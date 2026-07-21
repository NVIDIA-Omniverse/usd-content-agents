# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for validate_predictions and config_validate_predictions tasks."""

import json
import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from material_agent.materials import (
    DISALLOWED_UNKNOWN_VALIDATION_STATUS,
    FALLBACK_MATERIAL_NAME,
)
from material_agent.tasks.config_validate_predictions import (
    ValidatePredictionsConfigTask,
)
from material_agent.tasks.validate_predictions import (
    ValidatePredictionsTask,
    _best_fuzzy_match,
    _extract_material_name,
    _load_jsonl_predictions,
    _load_predictions_for_validation,
    _mark_materials_disallowed_unknown,
    _noop_disallowed_unknown_marker,
    _noop_fallback_marker,
    _prediction_material_records,
    _set_material_name,
    _write_predictions_atomically,
)

# ---------------------------------------------------------------------------
# _best_fuzzy_match
# ---------------------------------------------------------------------------


class TestBestFuzzyMatch:
    VALID_NAMES = [
        "Stainless Steel Polished",
        "Gold Polished",
        "Plastic White",
        "Copper Brushed",
        "Rubber Black",
    ]

    def test_exact_match_returns_score_one(self):
        match, score = _best_fuzzy_match("Stainless Steel Polished", self.VALID_NAMES)
        assert match == "Stainless Steel Polished"
        assert score == pytest.approx(1.0 + 0.1, abs=0.01)  # 1.0 ratio + 0.1 bonus

    def test_close_match_returns_high_score(self):
        match, score = _best_fuzzy_match("Stainles Steel Polished", self.VALID_NAMES)
        assert match == "Stainless Steel Polished"
        assert score > 0.8

    def test_token_containment_bonus(self):
        # "Steel Polished" tokens are a subset of "Stainless Steel Polished" tokens,
        # so "Stainless Steel Polished" should get a +0.1 bonus and beat "Gold Polished".
        match, score = _best_fuzzy_match("Steel Polished", self.VALID_NAMES)
        assert match == "Stainless Steel Polished"

    def test_no_match_returns_low_score(self):
        match, score = _best_fuzzy_match("Xylophone Unicorn", self.VALID_NAMES)
        assert score < 0.4


# ---------------------------------------------------------------------------
# _extract_material_name / _set_material_name
# ---------------------------------------------------------------------------


class TestExtractMaterialName:
    def test_extracts_from_dict(self):
        pred = {"materials": {"material": "Gold Polished"}}
        assert _extract_material_name(pred) == "Gold Polished"

    def test_returns_none_for_missing_materials(self):
        assert _extract_material_name({}) is None

    def test_extracts_from_string_materials(self):
        assert _extract_material_name({"materials": "string_value"}) == "string_value"

    def test_extracts_from_top_level_material(self):
        assert _extract_material_name({"material": "Gold Polished"}) == "Gold Polished"

    def test_returns_none_for_missing_material_key(self):
        assert _extract_material_name({"materials": {"other": "value"}}) is None


class TestSetMaterialName:
    def test_sets_correctly(self):
        pred = {"materials": {"material": "old"}}
        _set_material_name(pred, "new")
        assert pred["materials"]["material"] == "new"

    def test_creates_materials_dict_if_missing(self):
        pred = {}
        _set_material_name(pred, "Copper Brushed")
        assert pred["materials"]["material"] == "Copper Brushed"


class TestPredictionMaterialRecords:
    def test_missing_material_record_setter_persists_repair(self):
        pred = {"id": "/missing"}

        records = _prediction_material_records(pred)
        records[0].set_material("Gold Polished")

        assert pred["materials"]["material"] == "Gold Polished"

    def test_direct_marker_helpers_cover_top_level_and_noop_paths(self):
        string_material_pred = {"materials": "unknown"}
        _mark_materials_disallowed_unknown(string_material_pred)
        assert string_material_pred["materials"] == {
            "material": "unknown",
            "validation_status": DISALLOWED_UNKNOWN_VALIDATION_STATUS,
        }

        non_string_material_pred = {"materials": ["not", "a", "material"]}
        _mark_materials_disallowed_unknown(non_string_material_pred)
        assert non_string_material_pred["materials"] == {
            "material": "",
            "validation_status": DISALLOWED_UNKNOWN_VALIDATION_STATUS,
        }

        top_level_pred = {"id": "/top", "material": "__unknown__"}
        record = _prediction_material_records(top_level_pred)[0]
        record.mark_disallowed_unknown()
        record.set_material("Gold Polished")
        record.mark_fallback("__unknown__", "unknown_material")

        assert top_level_pred == {
            "id": "/top",
            "material": "Gold Polished",
            "validation_status": DISALLOWED_UNKNOWN_VALIDATION_STATUS,
            "fallback_source": "validation",
            "fallback_reason": "unknown_material",
            "fallback_original_material": "__unknown__",
        }

        assert _noop_disallowed_unknown_marker() is None
        assert _noop_fallback_marker(None, "unused") is None

    def test_container_item_markers_persist_list_and_dict_records(self):
        payload = {
            "predictions": ["bad list material"],
            "/World/DictMaterial": "bad dict material",
        }
        list_record, dict_record = _prediction_material_records(payload)

        list_record.set_material("Gold Polished")
        assert payload["predictions"][0] == "Gold Polished"
        payload["predictions"][0] = "bad list material"
        list_record.mark_disallowed_unknown()
        dict_record.mark_disallowed_unknown()
        assert payload["predictions"][0] == {
            "material": "",
            "validation_status": DISALLOWED_UNKNOWN_VALIDATION_STATUS,
        }
        assert payload["/World/DictMaterial"] == {
            "material": "",
            "validation_status": DISALLOWED_UNKNOWN_VALIDATION_STATUS,
        }

        payload = {"predictions": ["bad list material"]}
        record = _prediction_material_records(payload)[0]
        record.mark_fallback("bad list material", "missing_material")
        assert payload["predictions"][0] == {
            "material": "bad list material",
            "fallback_source": "validation",
            "fallback_reason": "missing_material",
            "fallback_original_material": "bad list material",
        }

        payload = {"predictions": [123]}
        assert _prediction_material_records(payload) == []


class TestPredictionLoading:
    def test_empty_predictions_file_loads_empty_list(self, tmp_path):
        path = tmp_path / "predictions.jsonl"
        path.write_text("  \n", encoding="utf-8")
        listener = MagicMock()

        predictions, json_document = _load_predictions_for_validation(path, listener)

        assert predictions == []
        assert json_document is False
        listener.warning.assert_called_once_with("Predictions file is empty")

    def test_json_suffix_falls_back_to_jsonl_and_skips_blank_lines(self, tmp_path):
        path = tmp_path / "predictions.json"
        path.write_text(
            '{"id": "/a", "materials": {"material": "Gold Polished"}}\n\n'
            '{"id": "/b", "materials": {"material": "Plastic White"}}\n',
            encoding="utf-8",
        )

        predictions, json_document = _load_predictions_for_validation(path, MagicMock())

        assert predictions == [
            {"id": "/a", "materials": {"material": "Gold Polished"}},
            {"id": "/b", "materials": {"material": "Plastic White"}},
        ]
        assert json_document is False

    def test_invalid_jsonl_reports_line_number(self):
        with pytest.raises(ValueError, match="line 2"):
            _load_jsonl_predictions('{"id": "/a"}\nnot-json\n')


# ---------------------------------------------------------------------------
# ValidatePredictionsTask.run()
# ---------------------------------------------------------------------------


def _write_predictions(path: Path, predictions: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")


def _read_predictions(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


VALID_NAMES = [
    "Stainless Steel Polished",
    "Gold Polished",
    "Plastic White",
    "Copper Brushed",
    "Rubber Black",
]


class TestValidatePredictionsTaskRun:
    def _make_context(self, predictions_path: Path) -> dict:
        return {
            "predictions_path": str(predictions_path),
            "material_names": VALID_NAMES,
        }

    def test_all_valid_predictions_no_changes(self, tmp_path):
        preds = [
            {"id": "/a", "materials": {"material": "Gold Polished"}},
            {"id": "/b", "materials": {"material": "Rubber Black"}},
        ]
        path = tmp_path / "predictions.jsonl"
        _write_predictions(path, preds)

        ctx = self._make_context(path)
        task = ValidatePredictionsTask()
        result = task.run(ctx)

        stats = result["validation_stats"]
        assert stats["valid"] == 2
        assert stats["auto_corrected"] == 0
        assert stats["no_material"] == 0

        # File unchanged (names still match)
        reloaded = _read_predictions(path)
        assert reloaded[0]["materials"]["material"] == "Gold Polished"
        assert reloaded[1]["materials"]["material"] == "Rubber Black"

    def test_auto_correctable_prediction(self, tmp_path):
        preds = [
            {"id": "/a", "materials": {"material": "Stainles Steel Polished"}},
        ]
        path = tmp_path / "predictions.jsonl"
        _write_predictions(path, preds)

        ctx = self._make_context(path)
        task = ValidatePredictionsTask()
        result = task.run(ctx)

        stats = result["validation_stats"]
        assert stats["auto_corrected"] == 1

        reloaded = _read_predictions(path)
        assert reloaded[0]["materials"]["material"] == "Stainless Steel Polished"

    def test_no_material_field_counted(self, tmp_path):
        preds = [
            {"id": "/a"},  # no materials at all
            {"id": "/b", "materials": {"material": "Gold Polished"}},
        ]
        path = tmp_path / "predictions.jsonl"
        _write_predictions(path, preds)

        ctx = self._make_context(path)
        task = ValidatePredictionsTask()
        result = task.run(ctx)

        stats = result["validation_stats"]
        assert stats["no_material"] == 1
        assert stats["valid"] == 1
        assert stats["fallback_applied"] == 1
        assert _read_predictions(path)[0]["materials"]["material"] == (
            FALLBACK_MATERIAL_NAME
        )

    def test_idless_jsonl_prediction_missing_material_is_counted(self, tmp_path):
        preds = [
            {"object": "part_without_selected_material"},
        ]
        path = tmp_path / "predictions.jsonl"
        _write_predictions(path, preds)

        ctx = self._make_context(path)
        result = ValidatePredictionsTask().run(ctx)

        stats = result["validation_stats"]
        assert stats["total"] == 1
        assert stats["no_material"] == 1
        assert stats["fallback_applied"] == 1
        assert _read_predictions(path)[0]["materials"]["material"] == (
            FALLBACK_MATERIAL_NAME
        )

    def test_mixed_top_level_and_nested_predictions_are_validated(self, tmp_path):
        preds = [
            {
                "id": "/parent",
                "materials": {"material": "Gold Polished"},
                "predictions": [
                    {
                        "id": "/child",
                        "materials": {"material": "Stainles Steel Polished"},
                    }
                ],
            },
        ]
        path = tmp_path / "predictions.jsonl"
        _write_predictions(path, preds)

        ctx = self._make_context(path)
        result = ValidatePredictionsTask().run(ctx)

        stats = result["validation_stats"]
        assert stats["total"] == 2
        assert stats["valid"] == 1
        assert stats["auto_corrected"] == 1

        reloaded = _read_predictions(path)
        assert reloaded[0]["materials"]["material"] == "Gold Polished"
        assert (
            reloaded[0]["predictions"][0]["materials"]["material"]
            == "Stainless Steel Polished"
        )

    def test_unknown_material_sentinel_is_preserved_when_allowed(self, tmp_path):
        preds = [
            {
                "id": "/a",
                "materials": {
                    "material": "__unknown__",
                    "reason": "no visible geometry",
                },
            },
            {"id": "/b", "materials": {"material": "Gold Polished"}},
        ]
        path = tmp_path / "predictions.jsonl"
        _write_predictions(path, preds)

        ctx = self._make_context(path)
        ctx["unknown_material_predictions"] = "from previous task"
        result = ValidatePredictionsTask().run(ctx)

        stats = result["validation_stats"]
        assert stats["unknown"] == 1
        assert stats["valid"] == 1
        assert stats["failed"] == 0
        assert stats["fallback_applied"] == 0
        assert result["unknown_material_predictions"] == 1

        reloaded = _read_predictions(path)
        assert reloaded[0]["materials"]["material"] == "__unknown__"

        report = json.loads((tmp_path / "validate_report.json").read_text())
        assert report["unknown"] == [
            {"index": 0, "id": "/a", "reason": "no visible geometry"}
        ]
        assert report["fallback_applied"] == []

    def test_unknown_material_sentinel_uses_fallback_when_disallowed(self, tmp_path):
        preds = [
            {
                "id": "/a",
                "materials": {
                    "material": "__unknown__",
                    "reason": "no visible geometry",
                },
            },
            {"id": "/b", "materials": {"material": "Gold Polished"}},
        ]
        path = tmp_path / "predictions.jsonl"
        _write_predictions(path, preds)

        ctx = self._make_context(path)
        ctx["allow_unknown_material"] = False
        task = ValidatePredictionsTask()
        result = task.run(ctx)

        stats = result["validation_stats"]
        assert stats["unknown"] == 0
        assert stats["unknown_disallowed"] == 1
        assert stats["valid"] == 1
        assert stats["failed"] == 0
        assert stats["fallback_applied"] == 1

        reloaded = _read_predictions(path)
        assert reloaded[0]["materials"]["material"] == FALLBACK_MATERIAL_NAME
        assert reloaded[0]["materials"]["reason"] == "no visible geometry"
        assert reloaded[0]["materials"]["fallback_source"] == "validation"
        assert reloaded[0]["materials"]["fallback_reason"] == "unknown_material"
        assert reloaded[0]["materials"]["fallback_original_material"] == "__unknown__"

        report = json.loads((tmp_path / "validate_report.json").read_text())
        assert report["unknown"] == []
        assert report["unknown_disallowed"] == [
            {"index": 0, "id": "/a", "reason": "no visible geometry"}
        ]
        assert report["fallback_applied"] == [
            {
                "index": 0,
                "new": FALLBACK_MATERIAL_NAME,
                "reason": "unknown_material",
                "id": "/a",
                "old": "__unknown__",
            }
        ]

    def test_string_materials_unknown_sentinel_uses_fallback(self, tmp_path):
        preds = [
            {"id": "/a", "materials": "__unknown__"},
            {"id": "/b", "materials": "Gold Polished"},
        ]
        path = tmp_path / "predictions.jsonl"
        _write_predictions(path, preds)

        ctx = self._make_context(path)
        ctx["allow_unknown_material"] = False
        task = ValidatePredictionsTask()
        result = task.run(ctx)

        stats = result["validation_stats"]
        assert stats["unknown"] == 0
        assert stats["unknown_disallowed"] == 1
        assert stats["valid"] == 1
        assert stats["fallback_applied"] == 1

        reloaded = _read_predictions(path)
        assert reloaded[0]["materials"]["material"] == FALLBACK_MATERIAL_NAME
        assert reloaded[0]["materials"]["fallback_source"] == "validation"

    def test_string_materials_auto_correction_preserves_shape(self, tmp_path):
        preds = [{"id": "/a", "materials": "Gold Polishd"}]
        path = tmp_path / "predictions.jsonl"
        _write_predictions(path, preds)

        result = ValidatePredictionsTask().run(self._make_context(path))

        assert result["validation_stats"]["auto_corrected"] == 1
        assert _read_predictions(path)[0]["materials"] == "Gold Polished"

    def test_unknown_material_uses_fallback_even_when_disallowed(self, tmp_path):
        preds = [
            {"id": "/a", "materials": {"material": "__UNKNOWN__"}},
            {"id": "/b", "materials": {"material": "Gold Polished"}},
        ]
        path = tmp_path / "predictions.jsonl"
        _write_predictions(path, preds)

        ctx = self._make_context(path)
        ctx["allow_unknown_material"] = False
        task = ValidatePredictionsTask()
        result = task.run(ctx)

        stats = result["validation_stats"]
        assert stats["unknown"] == 0
        assert stats["unknown_disallowed"] == 1
        assert stats["no_material"] == 0
        assert stats["valid"] == 1
        assert stats["auto_corrected"] == 0
        assert stats["llm_repaired"] == 0
        assert stats["fallback_applied"] == 1
        assert result["unknown_material_predictions"] == 0

        reloaded = _read_predictions(path)
        assert reloaded[0]["materials"]["material"] == FALLBACK_MATERIAL_NAME
        assert reloaded[0]["materials"]["fallback_reason"] == "unknown_material"

        report = json.loads((tmp_path / "validate_report.json").read_text())
        assert report["unknown"] == []
        assert report["unknown_disallowed"] == [{"index": 0, "id": "/a"}]

    def test_path_key_unknown_material_uses_fallback(self, tmp_path: Path) -> None:
        path = tmp_path / "predictions.json"
        path.write_text(
            json.dumps({"/World/Hidden": "__unknown__"}),
            encoding="utf-8",
        )

        ctx = self._make_context(path)
        ctx["allow_unknown_material"] = False
        result = ValidatePredictionsTask().run(ctx)

        assert result["validation_stats"]["unknown"] == 0
        assert result["validation_stats"]["unknown_disallowed"] == 1
        assert result["validation_stats"]["fallback_applied"] == 1
        reloaded = json.loads(path.read_text(encoding="utf-8"))
        assert reloaded["/World/Hidden"] == {
            "material": FALLBACK_MATERIAL_NAME,
            "fallback_source": "validation",
            "fallback_reason": "unknown_material",
            "fallback_original_material": "__unknown__",
        }

    def test_list_record_unknown_material_uses_fallback_and_preserves_metadata(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "predictions.json"
        path.write_text(
            json.dumps(
                [
                    {
                        "id": "/World/Hidden",
                        "materials": "__unknown__",
                        "reason": "no visible geometry",
                    }
                ]
            ),
            encoding="utf-8",
        )

        ctx = self._make_context(path)
        ctx["allow_unknown_material"] = False
        result = ValidatePredictionsTask().run(ctx)

        assert result["validation_stats"]["unknown"] == 0
        assert result["validation_stats"]["unknown_disallowed"] == 1
        assert result["validation_stats"]["fallback_applied"] == 1
        reloaded = json.loads(path.read_text(encoding="utf-8"))
        assert reloaded[0] == {
            "id": "/World/Hidden",
            "materials": {
                "material": FALLBACK_MATERIAL_NAME,
                "fallback_source": "validation",
                "fallback_reason": "unknown_material",
                "fallback_original_material": "__unknown__",
            },
            "reason": "no visible geometry",
        }

    def test_unexpected_llm_repair_index_is_skipped(self, tmp_path: Path) -> None:
        class UnexpectedIndexRepairTask(ValidatePredictionsTask):
            def _llm_repair(self, items, valid_names, llm_config, listener):
                return [(999, "Gold", None)]

        preds = [
            {"id": "/a", "materials": {"material": "Gold"}},
        ]
        path = tmp_path / "predictions.jsonl"
        _write_predictions(path, preds)

        ctx = self._make_context(path)
        ctx["llm_config"] = {"backend": "mock"}
        result = UnexpectedIndexRepairTask().run(ctx)

        stats = result["validation_stats"]
        assert stats["llm_repaired"] == 0
        assert stats["failed"] == 1
        assert _read_predictions(path) == preds

    def test_known_but_unrequested_llm_repair_index_skips_fuzzy_fallback(
        self, tmp_path: Path
    ) -> None:
        class UnexpectedExistingIndexRepairTask(ValidatePredictionsTask):
            def _llm_repair(self, items, valid_names, llm_config, listener):
                return [(1, "Gold Polished", None)]

        preds = [
            {"id": "/invalid", "materials": {"material": "Gold"}},
            {"id": "/valid", "materials": {"material": "Gold Polished"}},
        ]
        path = tmp_path / "predictions.jsonl"
        _write_predictions(path, preds)

        ctx = self._make_context(path)
        ctx["llm_config"] = {"backend": "mock"}
        result = UnexpectedExistingIndexRepairTask().run(ctx)

        stats = result["validation_stats"]
        assert stats["valid"] == 1
        assert stats["llm_repaired"] == 0
        assert stats["failed"] == 1
        assert _read_predictions(path) == preds

    def test_invalid_material_failed_llm_repair_uses_fallback(
        self, tmp_path: Path
    ) -> None:
        class FailedRepairTask(ValidatePredictionsTask):
            def _llm_repair(self, items, valid_names, llm_config, listener):
                return [(idx, old, None) for idx, old, _ in items]

        preds = [
            {
                "id": "/a",
                "materials": {"material": "Xylophone Unicorn"},
            },
        ]
        path = tmp_path / "predictions.jsonl"
        _write_predictions(path, preds)

        ctx = self._make_context(path)
        ctx["llm_config"] = {"backend": "mock"}
        result = FailedRepairTask().run(ctx)

        stats = result["validation_stats"]
        assert stats["llm_repaired"] == 0
        assert stats["failed"] == 1
        assert stats["fallback_applied"] == 1

        reloaded = _read_predictions(path)
        assert reloaded[0]["materials"]["material"] == FALLBACK_MATERIAL_NAME
        assert (
            reloaded[0]["materials"]["fallback_reason"] == "unrepaired_invalid_material"
        )
        assert (
            reloaded[0]["materials"]["fallback_original_material"]
            == "Xylophone Unicorn"
        )

        report = json.loads((tmp_path / "validate_report.json").read_text())
        assert report["fallback_applied"] == [
            {
                "index": 0,
                "new": FALLBACK_MATERIAL_NAME,
                "reason": "unrepaired_invalid_material",
                "id": "/a",
                "old": "Xylophone Unicorn",
            }
        ]

    def test_json_array_nested_and_path_keyed_predictions_are_validated(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "predictions.json"
        path.write_text(
            json.dumps(
                [
                    {
                        "results": [
                            {
                                "id": "/World/Nested",
                                "materials": {"material": "Stainles Steel Polished"},
                            }
                        ]
                    },
                    {"/World/PathKeyed": "Gold Polishd"},
                    {"/World/Hidden": "__unknown__"},
                ],
                indent=2,
            ),
            encoding="utf-8",
        )

        ctx = self._make_context(path)
        result = ValidatePredictionsTask().run(ctx)

        stats = result["validation_stats"]
        assert stats["total"] == 3
        assert stats["auto_corrected"] == 2
        assert stats["unknown"] == 1
        assert stats["no_material"] == 0

        reloaded = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(reloaded, list)
        assert (
            reloaded[0]["results"][0]["materials"]["material"]
            == "Stainless Steel Polished"
        )
        assert reloaded[1]["/World/PathKeyed"] == "Gold Polished"
        assert reloaded[2]["/World/Hidden"] == "__unknown__"

        report = json.loads((tmp_path / "validate_report.json").read_text())
        assert report["unknown"] == [{"index": 2, "id": "/World/Hidden"}]

    def test_atomic_rewrite_removes_stale_temp_files(self, tmp_path):
        path = tmp_path / "predictions.jsonl"
        path.write_text(json.dumps({"id": "/old"}) + "\n", encoding="utf-8")
        stale_path = tmp_path / ".predictions.jsonl.stale.tmp"
        stale_path.write_text("orphaned temp content", encoding="utf-8")
        old_time = time.time() - (2 * 60 * 60)
        os.utime(stale_path, (old_time, old_time))

        _write_predictions_atomically(
            path,
            [{"id": "/new", "materials": {"material": "Steel"}}],
        )

        assert not stale_path.exists()
        assert _read_predictions(path) == [
            {"id": "/new", "materials": {"material": "Steel"}}
        ]

    def test_atomic_rewrite_ignores_temp_stat_errors(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        path = tmp_path / "predictions.jsonl"
        path.write_text(json.dumps({"id": "/old"}) + "\n", encoding="utf-8")
        blocked_path = tmp_path / ".predictions.jsonl.blocked.tmp"
        blocked_path.write_text("blocked temp content", encoding="utf-8")

        original_stat = Path.stat

        def stat_or_permission_error(self, *args, **kwargs):
            if self == blocked_path:
                raise PermissionError("blocked")
            return original_stat(self, *args, **kwargs)

        monkeypatch.setattr(Path, "stat", stat_or_permission_error)

        _write_predictions_atomically(
            path,
            [{"id": "/new", "materials": {"material": "Steel"}}],
        )

        assert _read_predictions(path) == [
            {"id": "/new", "materials": {"material": "Steel"}}
        ]

    def test_atomic_rewrite_keeps_recent_temp_files(self, tmp_path):
        path = tmp_path / "predictions.jsonl"
        path.write_text(json.dumps({"id": "/old"}) + "\n", encoding="utf-8")
        recent_path = tmp_path / ".predictions.jsonl.active.tmp"
        recent_path.write_text("recent temp content", encoding="utf-8")

        _write_predictions_atomically(
            path,
            [{"id": "/new", "materials": {"material": "Steel"}}],
        )

        assert recent_path.exists()
        assert _read_predictions(path) == [
            {"id": "/new", "materials": {"material": "Steel"}}
        ]

    def test_predictions_file_rewritten_in_place(self, tmp_path):
        preds = [
            {"id": "/a", "materials": {"material": "Stainles Steel Polished"}},
            {"id": "/b", "materials": {"material": "Gold Polished"}},
        ]
        path = tmp_path / "predictions.jsonl"
        _write_predictions(path, preds)

        ctx = self._make_context(path)
        task = ValidatePredictionsTask()
        task.run(ctx)

        # File should exist at same path and content updated
        assert path.exists()
        reloaded = _read_predictions(path)
        assert len(reloaded) == 2
        assert reloaded[0]["materials"]["material"] == "Stainless Steel Polished"
        assert reloaded[1]["materials"]["material"] == "Gold Polished"
        assert not list(tmp_path.glob(".predictions.jsonl.*.tmp"))

    def test_atomic_rewrite_preserves_original_on_replace_failure(
        self, monkeypatch, tmp_path
    ):
        path = tmp_path / "predictions.jsonl"
        path.write_text(json.dumps({"id": "/old"}) + "\n", encoding="utf-8")

        def fail_replace(self, target):
            raise RuntimeError("replace failed")

        monkeypatch.setattr(Path, "replace", fail_replace)

        with pytest.raises(RuntimeError, match="replace failed"):
            _write_predictions_atomically(
                path,
                [{"id": "/new", "materials": {"material": "Steel"}}],
            )

        assert _read_predictions(path) == [{"id": "/old"}]
        assert not list(tmp_path.glob(".predictions.jsonl.*.tmp"))

    def test_validation_report_written(self, tmp_path):
        preds = [
            {"id": "/a", "materials": {"material": "Stainles Steel Polished"}},
        ]
        path = tmp_path / "predictions.jsonl"
        _write_predictions(path, preds)
        stale_report_tmp = tmp_path / ".validate_report.json.old.tmp"
        stale_report_tmp.write_text("stale", encoding="utf-8")
        stale_mtime = time.time() - 7200
        os.utime(stale_report_tmp, (stale_mtime, stale_mtime))

        ctx = self._make_context(path)
        task = ValidatePredictionsTask()
        task.run(ctx)

        report_path = tmp_path / "validate_report.json"
        assert report_path.exists()
        report = json.loads(report_path.read_text())
        assert "stats" in report
        assert "auto_corrected" in report
        assert report["stats"]["auto_corrected"] == 1
        assert not list(tmp_path.glob(".validate_report.json.*.tmp"))

    def test_missing_predictions_file_skips(self, tmp_path):
        path = tmp_path / "does_not_exist.jsonl"
        ctx = self._make_context(path)
        task = ValidatePredictionsTask()
        result = task.run(ctx)
        assert result["validation_stats"]["skipped"] is True


# ---------------------------------------------------------------------------
# ValidatePredictionsConfigTask.run()
# ---------------------------------------------------------------------------


class TestValidatePredictionsConfigTaskRun:
    def test_loads_config_sets_context(self, tmp_path):
        config = {
            "predictions_path": "/some/path/predictions.jsonl",
            "material_names": ["Mat A", "Mat B"],
            "llm": {"backend": "openai", "model": "gpt-4"},
        }
        cfg_path = tmp_path / "config.yaml"
        import yaml

        cfg_path.write_text(yaml.dump(config))

        ctx = {"config_path": str(cfg_path)}
        task = ValidatePredictionsConfigTask()
        result = task.run(ctx)

        assert result["predictions_path"] == "/some/path/predictions.jsonl"
        assert result["material_names"] == ["Mat A", "Mat B"]
        assert result["llm_config"]["backend"] == "openai"
        assert result["allow_unknown_material"] is True

    def test_loads_config_sets_unknown_material_policy(self, tmp_path):
        config = {
            "predictions_path": "/some/path/predictions.jsonl",
            "material_names": ["Mat A"],
            "allow_unknown_material": False,
        }
        cfg_path = tmp_path / "config.yaml"
        import yaml

        cfg_path.write_text(yaml.dump(config))

        ctx = {"config_path": str(cfg_path)}
        task = ValidatePredictionsConfigTask()
        result = task.run(ctx)

        assert result["allow_unknown_material"] is False

    def test_raises_on_missing_predictions_path(self, tmp_path):
        config = {"material_names": ["Mat A"]}
        cfg_path = tmp_path / "config.yaml"
        import yaml

        cfg_path.write_text(yaml.dump(config))

        ctx = {"config_path": str(cfg_path)}
        task = ValidatePredictionsConfigTask()
        with pytest.raises(ValueError, match="predictions_path"):
            task.run(ctx)

    def test_raises_on_missing_material_names(self, tmp_path):
        config = {"predictions_path": "/some/path.jsonl"}
        cfg_path = tmp_path / "config.yaml"
        import yaml

        cfg_path.write_text(yaml.dump(config))

        ctx = {"config_path": str(cfg_path)}
        task = ValidatePredictionsConfigTask()
        with pytest.raises(ValueError, match="material_names"):
            task.run(ctx)

    def test_raises_on_missing_config_path(self):
        ctx = {}
        task = ValidatePredictionsConfigTask()
        with pytest.raises(ValueError, match="config_path"):
            task.run(ctx)

    def test_raises_on_nonexistent_config_file(self, tmp_path):
        ctx = {"config_path": str(tmp_path / "nope.yaml")}
        task = ValidatePredictionsConfigTask()
        with pytest.raises(FileNotFoundError):
            task.run(ctx)
