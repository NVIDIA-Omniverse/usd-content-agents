# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for prediction symmetry and consistency analysis."""

import json

import pytest

import material_agent.tasks.prediction_analyzer as prediction_analyzer_module
from material_agent.tasks.prediction_analyzer import (
    PredictionAnalyzer,
    load_prims_metadata,
)


def _prediction(prim_id: str, material: str) -> dict:
    return {"id": prim_id, "materials": {"material": material}}


def _extent_for_center(x: float, y: float, z: float) -> str:
    return f"[({x - 0.1}, {y - 0.1}, {z - 0.1}), ({x + 0.1}, {y + 0.1}, {z + 0.1})]"


def _metadata(prim_id: str, center: tuple[float, float, float]) -> dict:
    return {
        "prim_path": prim_id,
        "metadata": {"extent": _extent_for_center(*center)},
    }


def test_detects_g1_left_right_path_symmetry_violations() -> None:
    """G1 side names start with left_/right_ and end in generic mesh leaves."""
    predictions = [
        _prediction(
            "/g1/humanoid__unitree__g1__left_shoulder_roll_link/visuals/"
            "left_shoulder_roll_link/mesh",
            "Aluminum",
        ),
        _prediction(
            "/g1/humanoid__unitree__g1__right_shoulder_roll_link/visuals/"
            "right_shoulder_roll_link/mesh",
            "Plastic Black",
        ),
        _prediction(
            "/g1/humanoid__unitree__g1__left_hip_pitch_link/visuals/"
            "left_hip_pitch_link/mesh",
            "Aluminum",
        ),
        _prediction(
            "/g1/humanoid__unitree__g1__right_hip_pitch_link/visuals/"
            "right_hip_pitch_link/mesh",
            "Plastic Black",
        ),
    ]

    result = PredictionAnalyzer(predictions).analyze()

    assert len(result.symmetry_pairs) == 2
    assert len(result.symmetry_violations) == 2
    assert {v.detection_method for v in result.symmetry_violations} == {"path"}
    assert {
        (v.prim_a, v.prim_b, v.material_a, v.material_b)
        for v in result.symmetry_violations
    } == {
        (
            "/g1/humanoid__unitree__g1__left_hip_pitch_link/visuals/"
            "left_hip_pitch_link/mesh",
            "/g1/humanoid__unitree__g1__right_hip_pitch_link/visuals/"
            "right_hip_pitch_link/mesh",
            "Aluminum",
            "Plastic Black",
        ),
        (
            "/g1/humanoid__unitree__g1__left_shoulder_roll_link/visuals/"
            "left_shoulder_roll_link/mesh",
            "/g1/humanoid__unitree__g1__right_shoulder_roll_link/visuals/"
            "right_shoulder_roll_link/mesh",
            "Aluminum",
            "Plastic Black",
        ),
    }


def test_detects_g1_geometry_leaf_symmetry_with_readable_names() -> None:
    predictions = [
        _prediction(
            "/visuals/left_ankle_roll_link/left_ankle_roll_link/mesh/Geometry",
            "Plastic Black",
        ),
        _prediction(
            "/visuals/right_ankle_roll_link/right_ankle_roll_link/mesh/Geometry",
            "Aluminum",
        ),
    ]

    result = PredictionAnalyzer(predictions).analyze()

    assert len(result.symmetry_pairs) == 1
    assert len(result.symmetry_violations) == 1
    assert result.symmetry_violations[0].detection_method == "path"
    assert "left_ankle_roll_link" in result.critique
    assert "right_ankle_roll_link" in result.critique
    assert "mesh" not in result.critique


def test_left_right_token_matching_does_not_match_unrelated_substrings() -> None:
    predictions = [
        _prediction("/robot/highlight_panel/mesh", "Aluminum"),
        _prediction("/robot/highright_panel/mesh", "Plastic Black"),
    ]

    result = PredictionAnalyzer(predictions).analyze()

    assert result.symmetry_pairs == []
    assert result.symmetry_violations == []


def test_numbered_path_symmetry_can_be_disabled() -> None:
    predictions = [
        _prediction("/asset/Geometry/Plastic1", "Plastic Black"),
        _prediction("/asset/Geometry/Plastic2", "Plastic Dark Blue"),
    ]

    default_result = PredictionAnalyzer(predictions).analyze()
    disabled_result = PredictionAnalyzer(
        predictions,
        detect_numbered_path_symmetry=False,
    ).analyze()

    assert len(default_result.symmetry_pairs) == 1
    assert disabled_result.symmetry_pairs == []
    assert disabled_result.symmetry_violations == []


def test_symmetry_resolution_can_be_left_to_vlm_feedback() -> None:
    predictions = [
        _prediction("/robot/left_arm/mesh", "Aluminum"),
        _prediction("/robot/right_arm/mesh", "Plastic Black"),
    ]

    result = PredictionAnalyzer(
        predictions,
        resolve_symmetry_directly=False,
        resolve_consistency_directly=False,
    ).analyze()

    assert len(result.symmetry_violations) == 1
    assert result.resolved_assignments == {}
    assert set(result.prim_feedback) == {
        "/robot/left_arm/mesh",
        "/robot/right_arm/mesh",
    }
    assert "Recommended:" not in result.critique
    assert "Both should use" not in result.critique
    assert "reference image" in result.critique


def test_consistency_resolution_can_be_left_to_vlm_feedback() -> None:
    predictions = [
        _prediction("/robot/visuals/panel_a/mesh", "Aluminum"),
        _prediction("/robot/visuals/panel_b/mesh", "Plastic Black"),
    ]

    result = PredictionAnalyzer(
        predictions,
        resolve_consistency_directly=False,
    ).analyze()

    assert len(result.consistency_violations) == 1
    assert result.resolved_assignments == {}
    assert set(result.prim_feedback) == {
        "/robot/visuals/panel_a/mesh",
        "/robot/visuals/panel_b/mesh",
    }
    assert "Recommended:" not in result.critique
    assert "Use '" not in result.critique
    assert "reference image" in result.critique


def test_initializes_string_materials_and_metadata_lookup() -> None:
    metadata = [{"prim_path": "/asset/part/mesh", "metadata": {"extent": ""}}]

    analyzer = PredictionAnalyzer(
        [{"id": "/asset/part/mesh", "materials": "Raw Material Label"}],
        prims_metadata=metadata,
    )

    assert analyzer._material_by_id["/asset/part/mesh"] == "Raw Material Label"
    assert analyzer._meta_by_path["/asset/part/mesh"] is metadata[0]


def test_path_short_name_fallback_matches_left_right_tokens() -> None:
    predictions = [
        _prediction("/variant_a/left_panel/mesh", "Aluminum"),
        _prediction("/variant_b/right_panel/mesh", "Plastic Black"),
    ]

    result = PredictionAnalyzer(predictions).analyze()

    assert result.symmetry_pairs == [
        ("/variant_a/left_panel/mesh", "/variant_b/right_panel/mesh")
    ]
    assert result.symmetry_violations[0].detection_method == "path"


def test_spatial_symmetry_detects_spatial_and_combined_pairs() -> None:
    predictions = [
        _prediction("/robot/left_arm/mesh", "Aluminum"),
        _prediction("/robot/right_arm/mesh", "Plastic Black"),
        _prediction("/robot/spatial_a/mesh", "Rubber"),
        _prediction("/robot/spatial_b/mesh", "Steel"),
        _prediction("/robot/center/mesh", "Plastic Black"),
        _prediction("/robot/unmatched/mesh", "Plastic Black"),
    ]
    metadata = [
        _metadata("/robot/left_arm/mesh", (-1.0, 0.0, 0.0)),
        _metadata("/robot/right_arm/mesh", (1.0, 0.0, 0.0)),
        _metadata("/robot/spatial_a/mesh", (-2.0, 1.0, 0.0)),
        _metadata("/robot/spatial_b/mesh", (2.0, 1.0, 0.0)),
        _metadata("/robot/center/mesh", (0.0, 2.0, 0.0)),
        _metadata("/robot/unmatched/mesh", (0.0, 3.0, 2.0)),
        {"prim_path": "/robot/not_predicted/mesh", "metadata": {"extent": ""}},
    ]

    result = PredictionAnalyzer(
        predictions,
        prims_metadata=metadata,
        symmetry_tolerance=0.25,
    ).analyze()

    methods_by_pair = {
        (violation.prim_a, violation.prim_b): violation.detection_method
        for violation in result.symmetry_violations
    }
    assert methods_by_pair[("/robot/left_arm/mesh", "/robot/right_arm/mesh")] == "both"
    assert (
        methods_by_pair[("/robot/spatial_a/mesh", "/robot/spatial_b/mesh")] == "spatial"
    )


def test_spatial_symmetry_skips_sparse_or_asymmetric_metadata() -> None:
    one_center = PredictionAnalyzer(
        [_prediction("/robot/only/mesh", "Aluminum")],
        prims_metadata=[_metadata("/robot/only/mesh", (0.0, 0.0, 0.0))],
    )
    asymmetric = PredictionAnalyzer(
        [
            _prediction("/robot/a/mesh", "Aluminum"),
            _prediction("/robot/b/mesh", "Plastic Black"),
        ],
        prims_metadata=[
            _metadata("/robot/a/mesh", (0.0, 0.0, 0.0)),
            _metadata("/robot/b/mesh", (10.0, 10.0, 10.0)),
        ],
        symmetry_tolerance=0.25,
    )

    assert one_center._detect_pairs_from_bounding_boxes() == []
    assert asymmetric._detect_pairs_from_bounding_boxes() == []
    assert (
        asymmetric._find_best_symmetry_axis(
            {"/robot/a/mesh": (0.0, 0.0, 0.0), "/robot/b/mesh": (10.0, 10.0, 10.0)}
        )
        is None
    )


def test_empty_and_already_consistent_predictions_have_no_feedback() -> None:
    empty_result = PredictionAnalyzer([]).analyze()
    consistent_result = PredictionAnalyzer(
        [
            _prediction("/robot/visuals/panel_a/mesh", "Aluminum"),
            _prediction("/robot/visuals/panel_b/mesh", "Aluminum"),
        ]
    ).analyze()

    assert empty_result.score == 0.0
    assert empty_result.critique == (
        "All predictions are symmetric and consistent. No issues found."
    )
    assert consistent_result.consistency_violations == []
    assert consistent_result.critique == (
        "All predictions are symmetric and consistent. No issues found."
    )


def test_direct_feedback_uses_globally_dominant_symmetric_material() -> None:
    predictions = [
        _prediction("/robot/left_arm/mesh", "Aluminum"),
        _prediction("/robot/right_arm/mesh", "Plastic Black"),
        _prediction("/robot/body/mesh", "Plastic Black"),
    ]

    result = PredictionAnalyzer(predictions).analyze()

    assert result.symmetry_violations[0].suggested == "Plastic Black"
    assert result.resolved_assignments["/robot/left_arm/mesh"] == "Plastic Black"
    assert "/robot/left_arm/mesh" in result.prim_feedback


def test_private_helpers_cover_short_paths_and_extent_parse_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analyzer = PredictionAnalyzer([])

    assert analyzer._extract_group_name("/single") == ""
    assert analyzer._parse_extent_center("") is None
    assert analyzer._parse_extent_center("[(1, 2, 3)]") is None
    assert analyzer._parse_extent_center("[(0, 2, 4), (2, 4, 6)]") == (
        1.0,
        3.0,
        5.0,
    )

    monkeypatch.setattr(
        prediction_analyzer_module.re,
        "findall",
        lambda _pattern, _text: ["not-a-number"] * 6,
    )
    assert analyzer._parse_extent_center("bad extent") is None


def test_load_prims_metadata_uses_usd_file_fallback_and_warns(tmp_path) -> None:
    dataset_dir = tmp_path / "dataset"
    usd_dir = dataset_dir / "usd"
    usd_dir.mkdir(parents=True)
    predictions_path = dataset_dir / "predictions.jsonl"
    predictions_path.write_text("", encoding="utf-8")
    (usd_dir / "prims.jsonl").write_text(
        "\n" + json.dumps({"prim_path": "/asset/a"}) + "\n",
        encoding="utf-8",
    )

    assert load_prims_metadata(predictions_path) == [{"prim_path": "/asset/a"}]

    (usd_dir / "prims.jsonl").unlink()
    (dataset_dir / "prims.jsonl").write_text(
        json.dumps({"prim_path": "/asset/b"}) + "\n",
        encoding="utf-8",
    )
    assert load_prims_metadata(predictions_path) == [{"prim_path": "/asset/b"}]

    (dataset_dir / "prims.jsonl").unlink()
    assert load_prims_metadata(predictions_path) == []
