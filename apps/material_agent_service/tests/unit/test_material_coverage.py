# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused tests for prim-level material coverage qualification."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ...service import coverage as coverage_module
from ...service.coverage import (
    build_material_coverage,
    coverage_is_release_ready,
    normalize_coverage_policy,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{json.dumps(record)}\n" for record in records),
        encoding="utf-8",
    )


def _result(
    session_dir: Path,
    *,
    predictions: list[dict],
    bound_ids: list[str],
    unbound_ids: list[str],
) -> SimpleNamespace:
    prediction_path = session_dir / "cache" / "predictions" / "predictions.jsonl"
    _write_jsonl(prediction_path, predictions)
    return SimpleNamespace(
        completed_steps=[
            "build_dataset_usd",
            "build_dataset_prepare_dataset",
            "predict",
            "apply",
        ],
        step_results={
            "build_dataset_usd": {"num_prims": 4},
            "build_dataset_prepare_dataset": {"num_entries": 4},
            "predict": {
                "predictions_count": len(predictions),
                "predictions_path": str(prediction_path),
            },
            "apply": {
                "assignment_stats": {
                    "total_prims": len(bound_ids),
                    "bound_prim_ids": bound_ids,
                    "unbound_prim_ids": unbound_ids,
                }
            },
        },
    )


def _write_targets(session_dir: Path) -> list[str]:
    target_ids = [f"/Root/Part_{index}" for index in range(4)]
    _write_jsonl(
        session_dir / "cache" / "dataset" / "prims.jsonl",
        [{"prim_path": prim_id} for prim_id in target_ids],
    )
    _write_jsonl(
        session_dir / "cache" / "dataset" / "dataset.jsonl",
        [{"id": prim_id} for prim_id in target_ids],
    )
    return target_ids


def _restored_result(
    session_dir: Path,
    *,
    restored_prim_sources: dict[str, str],
    restored_predictions: list[dict],
    bound_ids: list[str],
    unbound_ids: list[str],
    mapping_complete: bool = True,
    expected_target_count: int | None = None,
) -> SimpleNamespace:
    optimized_ids = sorted(set(restored_prim_sources.values()))
    prims_path = session_dir / "cache" / "dataset" / "prims.jsonl"
    dataset_path = session_dir / "cache" / "dataset" / "dataset.jsonl"
    raw_predictions_path = session_dir / "cache" / "predictions" / "predictions.jsonl"
    restored_predictions_path = (
        session_dir / "cache" / "restored" / "restored_predictions.jsonl"
    )
    _write_jsonl(prims_path, [{"prim_path": prim_id} for prim_id in optimized_ids])
    _write_jsonl(dataset_path, [{"id": prim_id} for prim_id in optimized_ids])
    _write_jsonl(
        raw_predictions_path,
        [{"id": prim_id, "material": "RawSteel"} for prim_id in optimized_ids],
    )
    _write_jsonl(restored_predictions_path, restored_predictions)
    return SimpleNamespace(
        completed_steps=[
            "build_dataset_usd",
            "build_dataset_prepare_dataset",
            "predict",
            "optimize_usd",
            "restore_usd",
            "apply",
        ],
        step_results={
            "build_dataset_usd": {"num_prims": len(optimized_ids)},
            "build_dataset_prepare_dataset": {
                "num_entries": len(optimized_ids),
                "dataset_jsonl_path": str(dataset_path),
            },
            "predict": {
                "predictions_count": len(optimized_ids),
                "predictions_path": str(raw_predictions_path),
            },
            "optimize_usd": {
                "optimization_metadata": {
                    "correspondence_map": {
                        "full_mapping": {"original_to_prototype": {}},
                        "split_mapping": {},
                    }
                }
            },
            "restore_usd": {
                "restored_predictions_path": str(restored_predictions_path),
                "restore_success": True,
                "restore_stats": {
                    "restored_prim_sources": restored_prim_sources,
                    "expected_target_count": (
                        len(restored_prim_sources)
                        if expected_target_count is None
                        else expected_target_count
                    ),
                    "mapping_complete": mapping_complete,
                    "mapping_warnings": [],
                },
            },
            "apply": {
                "assignment_stats": {
                    "total_prims": len(bound_ids),
                    "bound_prim_ids": bound_ids,
                    "unbound_prim_ids": unbound_ids,
                }
            },
        },
    )


def test_one_of_four_predictions_is_explicit_partial_coverage(tmp_path: Path) -> None:
    target_ids = _write_targets(tmp_path)
    result = _result(
        tmp_path,
        predictions=[{"id": target_ids[0], "material": "Steel"}],
        bound_ids=[target_ids[0]],
        unbound_ids=[],
    )

    coverage = build_material_coverage(result, tmp_path, policy="allow_partial")

    assert coverage["readiness_grade"] == "partial"
    assert coverage["target_count"] == 4
    assert coverage["prepared_count"] == 4
    assert coverage["predicted_count"] == 1
    assert coverage["usable_prediction_count"] == 1
    assert coverage["fallback_count"] == 0
    assert coverage["bound_count"] == 1
    assert coverage["unbound_count"] == 3
    assert coverage["prediction_coverage_ratio"] == 0.25
    assert coverage["binding_coverage_ratio"] == 0.25
    assert coverage["missing_prediction_prim_ids"] == target_ids[1:]
    assert coverage["unbound_prim_ids"] == target_ids[1:]
    assert not coverage_is_release_ready(coverage)


def test_fallback_is_separate_from_usable_prediction_and_can_be_ready(
    tmp_path: Path,
) -> None:
    target_ids = _write_targets(tmp_path)
    predictions = [
        {"id": target_ids[0], "material": "Steel"},
        {"id": target_ids[1], "material": "Rubber"},
        {"id": target_ids[2], "material": "Plastic"},
        {
            "id": target_ids[3],
            "materials": {
                "material": "__FALLBACK_MATERIAL__",
                "fallback_source": "validation",
                "fallback_reason": "missing_material",
            },
        },
    ]
    result = _result(
        tmp_path,
        predictions=predictions,
        bound_ids=target_ids,
        unbound_ids=[],
    )

    coverage = build_material_coverage(result, tmp_path, policy="strict")

    assert coverage["readiness_grade"] == "complete_with_fallback"
    assert coverage["predicted_count"] == 4
    assert coverage["usable_prediction_count"] == 3
    assert coverage["fallback_count"] == 1
    assert coverage["unknown_prediction_count"] == 0
    assert coverage["bound_count"] == 4
    assert coverage_is_release_ready(coverage)


def test_unknown_prediction_remains_unqualified(tmp_path: Path) -> None:
    target_ids = _write_targets(tmp_path)
    result = _result(
        tmp_path,
        predictions=[
            {"id": prim_id, "material": "Steel"} for prim_id in target_ids[:-1]
        ]
        + [{"id": target_ids[-1], "material": "__UNKNOWN__"}],
        bound_ids=target_ids[:-1],
        unbound_ids=[target_ids[-1]],
    )

    coverage = build_material_coverage(result, tmp_path, policy="strict")

    assert coverage["unknown_prediction_count"] == 1
    assert coverage["unknown_prim_ids"] == [target_ids[-1]]
    assert coverage["unbound_prim_ids"] == [target_ids[-1]]
    assert coverage["readiness_grade"] == "partial"
    assert not coverage_is_release_ready(coverage)


def test_missing_target_ids_cannot_shrink_denominator_to_prepared_subset(
    tmp_path: Path,
) -> None:
    only_known_id = "/Root/Part_0"
    _write_jsonl(
        tmp_path / "cache" / "dataset" / "dataset.jsonl",
        [{"id": only_known_id}],
    )
    result = _result(
        tmp_path,
        predictions=[{"id": only_known_id, "material": "Steel"}],
        bound_ids=[only_known_id],
        unbound_ids=[],
    )

    coverage = build_material_coverage(result, tmp_path, policy="strict")

    assert coverage["target_count"] == 4
    assert coverage["prepared_count"] == 1
    assert coverage["predicted_count"] == 1
    assert coverage["bound_count"] == 1
    assert coverage["unbound_count"] == 3
    assert coverage["readiness_grade"] == "not_evaluated"
    assert any("target prim ID" in warning for warning in coverage["warnings"])
    assert not coverage_is_release_ready(coverage)


def test_aggregate_binding_count_does_not_qualify_without_exact_ids(
    tmp_path: Path,
) -> None:
    target_ids = _write_targets(tmp_path)
    result = _result(
        tmp_path,
        predictions=[{"id": prim_id, "material": "Steel"} for prim_id in target_ids],
        bound_ids=target_ids,
        unbound_ids=[],
    )
    result.step_results["apply"] = {
        "assignment_stats": {"total_prims": len(target_ids)}
    }

    coverage = build_material_coverage(result, tmp_path, policy="strict")

    assert coverage["bound_count"] == 4
    assert coverage["binding_coverage_ratio"] == 1.0
    assert coverage["readiness_grade"] == "not_evaluated"
    assert any("Exact binding prim IDs" in warning for warning in coverage["warnings"])
    assert not coverage_is_release_ready(coverage)


def test_apply_only_regeneration_can_qualify_cached_prediction_evidence(
    tmp_path: Path,
) -> None:
    target_ids = _write_targets(tmp_path)
    result = _result(
        tmp_path,
        predictions=[{"id": prim_id, "material": "Steel"} for prim_id in target_ids],
        bound_ids=target_ids,
        unbound_ids=[],
    )
    result.completed_steps = ["apply"]
    result.step_results = {"apply": result.step_results["apply"]}

    coverage = build_material_coverage(result, tmp_path, policy="strict")

    assert coverage["readiness_grade"] == "complete"
    assert coverage_is_release_ready(coverage)


def test_restore_namespace_qualifies_identity_dedup_split_and_split_dedup(
    tmp_path: Path,
) -> None:
    restored_prim_sources = {
        "/Original/A": "/Optimized/P1",
        "/Original/B": "/Optimized/P1",
        "/Original/Split/Left": "/Optimized/P2",
        "/Original/Split/Right": "/Optimized/P2",
    }
    target_ids = list(restored_prim_sources)
    result = _restored_result(
        tmp_path,
        restored_prim_sources=restored_prim_sources,
        restored_predictions=[
            {"id": prim_id, "material": "Steel"} for prim_id in target_ids
        ],
        bound_ids=target_ids,
        unbound_ids=[],
    )

    coverage = build_material_coverage(result, tmp_path, policy="strict")

    assert coverage["readiness_grade"] == "complete"
    assert coverage["target_count"] == 4
    assert coverage["prepared_count"] == 4
    assert coverage["predicted_count"] == 4
    assert coverage["bound_count"] == 4
    for field in (
        "missing_prepared_prim_ids",
        "missing_prediction_prim_ids",
        "extra_prediction_prim_ids",
        "unbound_prim_ids",
    ):
        assert coverage[field] == []
        assert not any("/Optimized/" in prim_id for prim_id in coverage[field])


def test_restore_namespace_projects_full_map_to_selected_optimized_targets(
    tmp_path: Path,
) -> None:
    restored_prim_sources = {
        "/Original/SelectedA": "/Optimized/Selected",
        "/Original/SelectedB": "/Optimized/Selected",
        "/Original/FilteredOut": "/Optimized/Unrelated",
    }
    selected_restored_ids = ["/Original/SelectedA", "/Original/SelectedB"]
    result = _restored_result(
        tmp_path,
        restored_prim_sources=restored_prim_sources,
        restored_predictions=[
            {"id": prim_id, "material": "Steel"} for prim_id in selected_restored_ids
        ],
        bound_ids=selected_restored_ids,
        unbound_ids=[],
    )
    _write_jsonl(
        tmp_path / "cache" / "dataset" / "prims.jsonl",
        [{"prim_path": "/Optimized/Selected"}],
    )
    _write_jsonl(
        tmp_path / "cache" / "dataset" / "dataset.jsonl",
        [{"id": "/Optimized/Selected"}],
    )
    result.step_results["build_dataset_usd"]["num_prims"] = 1
    result.step_results["build_dataset_prepare_dataset"]["num_entries"] = 1

    coverage = build_material_coverage(result, tmp_path, policy="strict")

    assert coverage["readiness_grade"] == "complete"
    assert coverage["target_count"] == 2
    assert coverage["prepared_count"] == 2
    assert coverage["predicted_count"] == 2
    assert coverage["bound_count"] == 2
    assert coverage["missing_prepared_prim_ids"] == []
    assert coverage["missing_prediction_prim_ids"] == []
    assert coverage["unbound_prim_ids"] == []
    assert "/Original/FilteredOut" not in json.dumps(coverage)


def test_restore_namespace_keeps_missing_prediction_in_denominator(
    tmp_path: Path,
) -> None:
    missing_id = "/Original/Split/Right"
    restored_prim_sources = {
        "/Original/A": "/Optimized/P1",
        "/Original/B": "/Optimized/P1",
        "/Original/Split/Left": "/Optimized/P2",
        missing_id: "/Optimized/P2",
    }
    predicted_ids = [
        prim_id for prim_id in restored_prim_sources if prim_id != missing_id
    ]
    result = _restored_result(
        tmp_path,
        restored_prim_sources=restored_prim_sources,
        restored_predictions=[
            {"id": prim_id, "material": "Steel"} for prim_id in predicted_ids
        ],
        bound_ids=predicted_ids,
        unbound_ids=[],
    )

    coverage = build_material_coverage(result, tmp_path, policy="strict")

    assert coverage["readiness_grade"] == "partial"
    assert coverage["target_count"] == 4
    assert coverage["predicted_count"] == 3
    assert coverage["missing_prediction_prim_ids"] == [missing_id]
    assert coverage["unbound_prim_ids"] == [missing_id]


def test_restore_namespace_expands_missing_prepared_source_to_restored_targets(
    tmp_path: Path,
) -> None:
    restored_prim_sources = {
        "/Original/A": "/Optimized/P1",
        "/Original/B": "/Optimized/P1",
        "/Original/Split/Left": "/Optimized/P2",
        "/Original/Split/Right": "/Optimized/P2",
    }
    target_ids = list(restored_prim_sources)
    result = _restored_result(
        tmp_path,
        restored_prim_sources=restored_prim_sources,
        restored_predictions=[
            {"id": prim_id, "material": "Steel"} for prim_id in target_ids
        ],
        bound_ids=target_ids,
        unbound_ids=[],
    )
    dataset_path = tmp_path / "cache" / "dataset" / "dataset.jsonl"
    _write_jsonl(dataset_path, [{"id": "/Optimized/P1"}])

    coverage = build_material_coverage(result, tmp_path, policy="strict")

    assert coverage["readiness_grade"] == "partial"
    assert coverage["prepared_count"] == 2
    assert coverage["missing_prepared_prim_ids"] == [
        "/Original/Split/Left",
        "/Original/Split/Right",
    ]


@pytest.mark.parametrize(
    ("mapping_complete", "expected_target_count"),
    [(False, 1), (True, 2)],
)
def test_restore_namespace_incomplete_mapping_fails_closed(
    tmp_path: Path,
    mapping_complete: bool,
    expected_target_count: int,
) -> None:
    target_id = "/Original/A"
    result = _restored_result(
        tmp_path,
        restored_prim_sources={target_id: "/Optimized/P1"},
        restored_predictions=[{"id": target_id, "material": "Steel"}],
        bound_ids=[target_id],
        unbound_ids=[],
        mapping_complete=mapping_complete,
        expected_target_count=expected_target_count,
    )

    coverage = build_material_coverage(result, tmp_path, policy="strict")

    assert coverage["readiness_grade"] == "not_evaluated"
    assert not coverage_is_release_ready(coverage)
    assert any("Restore" in warning for warning in coverage["warnings"])


def test_restore_namespace_diagnostics_keep_canonical_denominator(
    tmp_path: Path,
) -> None:
    restored_prim_sources = {
        "/Original/A": "/Optimized/P1",
        "/Original/B": "/Optimized/P2",
    }
    restored_predictions = [
        {"id": prim_id, "material": "Steel"} for prim_id in restored_prim_sources
    ]

    missing_source_dir = tmp_path / "missing-source"
    missing_source = _restored_result(
        missing_source_dir,
        restored_prim_sources=restored_prim_sources,
        restored_predictions=[restored_predictions[0]],
        bound_ids=["/Original/A"],
        unbound_ids=[],
    )
    _write_jsonl(
        missing_source_dir / "cache" / "dataset" / "prims.jsonl",
        [{"prim_path": "/Optimized/P1"}],
    )
    missing_source_coverage = build_material_coverage(
        missing_source, missing_source_dir, policy="strict"
    )
    assert missing_source_coverage["target_count"] == 1
    assert missing_source_coverage["readiness_grade"] == "not_evaluated"
    assert not coverage_is_release_ready(missing_source_coverage)
    assert any(
        "optimized target prim ID(s) were unavailable" in warning
        for warning in missing_source_coverage["warnings"]
    )

    no_hint_dir = tmp_path / "missing-source-no-hint"
    no_hint = _restored_result(
        no_hint_dir,
        restored_prim_sources=restored_prim_sources,
        restored_predictions=[restored_predictions[0]],
        bound_ids=["/Original/A"],
        unbound_ids=[],
    )
    _write_jsonl(
        no_hint_dir / "cache" / "dataset" / "prims.jsonl",
        [{"prim_path": "/Optimized/P1"}],
    )
    no_hint.step_results["build_dataset_usd"]["num_prims"] = 0
    no_hint.step_results["build_dataset_prepare_dataset"]["num_entries"] = 0
    no_hint_coverage = build_material_coverage(no_hint, no_hint_dir, policy="strict")
    assert no_hint_coverage["readiness_grade"] == "not_evaluated"
    assert not coverage_is_release_ready(no_hint_coverage)
    assert any(
        "absent from the prepared target scope" in warning
        for warning in no_hint_coverage["warnings"]
    )

    extra_source_dir = tmp_path / "extra-source"
    extra_source = _restored_result(
        extra_source_dir,
        restored_prim_sources=restored_prim_sources,
        restored_predictions=restored_predictions,
        bound_ids=list(restored_prim_sources),
        unbound_ids=[],
    )
    extra_source.step_results["build_dataset_usd"]["num_prims"] = 1
    extra_source.step_results["build_dataset_prepare_dataset"]["num_entries"] = 1
    extra_source_coverage = build_material_coverage(
        extra_source, extra_source_dir, policy="strict"
    )
    assert extra_source_coverage["target_count"] == 2
    assert any(
        "unexpected optimized target prim ID(s)" in warning
        for warning in extra_source_coverage["warnings"]
    )

    unmapped_source_dir = tmp_path / "unmapped-source"
    unmapped_source = _restored_result(
        unmapped_source_dir,
        restored_prim_sources=restored_prim_sources,
        restored_predictions=restored_predictions,
        bound_ids=list(restored_prim_sources),
        unbound_ids=[],
    )
    _write_jsonl(
        unmapped_source_dir / "cache" / "dataset" / "prims.jsonl",
        [
            {"prim_path": "/Optimized/P1"},
            {"prim_path": "/Optimized/P2"},
            {"prim_path": "/Optimized/Unmapped"},
        ],
    )
    unmapped_source.step_results["build_dataset_usd"]["num_prims"] = 3
    unmapped_source.step_results["build_dataset_prepare_dataset"]["num_entries"] = 3
    unmapped_source_coverage = build_material_coverage(
        unmapped_source, unmapped_source_dir, policy="strict"
    )
    assert unmapped_source_coverage["readiness_grade"] == "not_evaluated"
    assert any(
        "absent from the restore correspondence map" in warning
        for warning in unmapped_source_coverage["warnings"]
    )


def test_restore_namespace_malformed_evidence_and_missing_artifact_fail_closed(
    tmp_path: Path,
) -> None:
    restore_ran = {"restore_usd"}

    assert coverage_module._restore_namespace_evidence(
        {"restore_usd": "invalid"}, restore_ran
    ) == (
        True,
        False,
        {},
        ["Restore ran without structured namespace evidence."],
    )

    _, complete, sources, warnings = coverage_module._restore_namespace_evidence(
        {"restore_usd": {"restore_success": False}}, restore_ran
    )
    assert complete is False
    assert sources == {}
    assert "Restore did not report successful namespace translation." in warnings
    assert "Restore target correspondence metadata was unavailable." in warnings

    _, complete, sources, warnings = coverage_module._restore_namespace_evidence(
        {
            "restore_usd": {
                "restore_success": True,
                "restore_stats": {
                    "restored_prim_sources": [],
                    "expected_target_count": 1,
                    "mapping_complete": True,
                },
            }
        },
        restore_ran,
    )
    assert complete is False
    assert sources == {}
    assert "Restore target correspondence map was malformed." in warnings

    _, complete, sources, warnings = coverage_module._restore_namespace_evidence(
        {
            "restore_usd": {
                "restore_success": True,
                "restore_stats": {
                    "restored_prim_sources": {"relative": "also-relative"},
                    "expected_target_count": 1,
                    "mapping_complete": True,
                },
            }
        },
        restore_ran,
    )
    assert complete is False
    assert sources == {}
    assert any("contained 1 invalid" in warning for warning in warnings)

    result = SimpleNamespace(
        completed_steps=["restore_usd"],
        step_results={
            "restore_usd": {
                "restored_predictions_path": str(tmp_path / "missing.jsonl")
            }
        },
    )
    assert coverage_module._prediction_path(result, tmp_path) is None


def test_coverage_policy_validation() -> None:
    assert normalize_coverage_policy(" strict ") == "strict"
    assert normalize_coverage_policy("allow_partial") == "allow_partial"
    with pytest.raises(ValueError, match="coverage_policy"):
        normalize_coverage_policy("best_effort")


def test_unscoped_binding_ids_are_clamped_to_target_scope(tmp_path: Path) -> None:
    target_ids = _write_targets(tmp_path)
    _write_jsonl(
        tmp_path / "cache" / "dataset" / "dataset.jsonl",
        [{"id": prim_id} for prim_id in target_ids[:-1]],
    )
    extra_id = "/Outside/Prediction"
    result = _result(
        tmp_path,
        predictions=[
            {"id": prim_id, "material": "Steel"} for prim_id in [*target_ids, extra_id]
        ],
        bound_ids=target_ids[:-1],
        unbound_ids=[target_ids[-1], "/Outside/Binding"],
    )

    coverage = build_material_coverage(result, tmp_path, policy="strict")

    assert coverage["missing_prepared_prim_ids"] == [target_ids[-1]]
    assert coverage["extra_prediction_prim_ids"] == [extra_id]
    assert coverage["unbound_prim_ids"] == [target_ids[-1]]
    assert any("not prepared" in warning for warning in coverage["warnings"])
    assert any("outside target scope" in warning for warning in coverage["warnings"])


def test_prediction_artifact_parser_handles_nested_and_invalid_shapes(
    tmp_path: Path,
) -> None:
    prediction_path = tmp_path / "predictions.jsonl"
    prediction_path.write_text(
        "\n"
        "not-json\n"
        + json.dumps(
            [
                {"id": "/Root/A", "materials": "Steel"},
                {
                    "id": "/Root/B",
                    "materials": {
                        "material": None,
                        "validation_status": "disallowed_unknown",
                    },
                },
                {"id": "/Root/C", "material": 123},
                {
                    "id": "/Root/E",
                    "predictions": {"material": "Glass"},
                },
                {"id": "/Root/J"},
                {
                    "/Root/F": "Rubber",
                    "/Root/G": {"material": "Plastic"},
                    "/Root/H": [{"material": "Steel"}],
                    "/Root/I": ["Copper"],
                },
                "Aluminum",
                7,
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    states = coverage_module._load_prediction_states(prediction_path)

    assert states == {
        "/Root/A": "usable",
        "/Root/B": "unknown",
        "/Root/C": "unknown",
        "/Root/E": "usable",
        "/Root/F": "usable",
        "/Root/G": "usable",
        "/Root/H": "usable",
        "/Root/I": "usable",
        "/Root/J": "unknown",
    }
    assert coverage_module._record_id(7) is None
    assert coverage_module._record_id({"id": "relative"}) is None
    assert coverage_module._binding_evidence(None) == (set(), set(), 0, False)
