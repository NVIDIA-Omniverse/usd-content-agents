# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collision candidate planner regressions."""

from __future__ import annotations

import json
from pathlib import Path

from geometry_repair.collision import _coacd_search_schedule
from geometry_repair.collision_planner import (
    evaluate_collision_candidate,
    select_collision_candidate,
)
from geometry_repair.models import RepairBudgets


def test_default_generated_collision_budget_matches_robocasa_profile() -> None:
    assert RepairBudgets().max_collision_hulls == 128


def test_coacd_search_schedule_is_bounded_and_escalates_fidelity() -> None:
    assert _coacd_search_schedule(maximum_hulls=16, threshold_m=0.01) == [("coarse", 16, 0.01)]
    assert _coacd_search_schedule(maximum_hulls=128, threshold_m=0.01) == [
        ("coarse", 32, 0.01),
        ("refined", 128, 0.005),
    ]
    assert _coacd_search_schedule(
        maximum_hulls=128,
        threshold_m=0.01,
        coarse_threshold_m=0.04,
    ) == [
        ("coarse", 32, 0.04),
        ("refined", 128, 0.01),
    ]


def _candidate(
    candidate_id: str,
    *,
    error: float,
    representation: str = "convex_hull",
    authored_source: bool = False,
    protected_features: list[str] | None = None,
    protected_features_satisfied: bool | None = None,
):
    return evaluate_collision_candidate(
        candidate_id=candidate_id,
        representation=representation,  # type: ignore[arg-type]
        generator=f"test.{candidate_id}",
        metrics={
            "volume_excess_ratio": error,
            "volume_deficit_ratio": min(error, 0.005),
            "false_positive_ratio": error,
            "false_negative_ratio": min(error, 0.005),
            "surface_gap_m": min(error, 0.004),
            "surface_overreach_m": min(error, 0.004),
        },
        hull_count=1,
        total_vertices=8,
        total_faces=12,
        maximum_vertices_per_hull=8,
        maximum_faces_per_hull=12,
        diagonal_m=1.0,
        budgets=RepairBudgets(),
        remaining_hull_budget=16,
        protected_features=protected_features,
        protected_features_satisfied=protected_features_satisfied,
        authored_source=authored_source,
    )


def test_selects_lowest_error_passing_candidate_independent_of_input_order(
    tmp_path: Path,
) -> None:
    coarse = _candidate("coarse", error=0.08)
    precise = _candidate("precise", error=0.01, representation="coacd")
    report_path = tmp_path / "candidate_search.json"

    forward = select_collision_candidate(
        "/Asset/Body",
        [coarse, precise],
        report_path=report_path,
    )
    reverse = select_collision_candidate("/Asset/Body", [precise, coarse])

    assert forward.selected_candidate_id == "precise"
    assert reverse.selected_candidate_id == "precise"
    assert json.loads(report_path.read_text(encoding="utf-8"))["selected_candidate_id"] == (
        "precise"
    )


def test_certified_authored_source_precedes_generated_substitute() -> None:
    source = _candidate(
        "source",
        error=0.04,
        representation="source_collision",
        authored_source=True,
    )
    generated = _candidate("generated", error=0.005, representation="coacd")

    result = select_collision_candidate("/Asset/Body", [generated, source])

    assert result.selected_candidate_id == "source"


def test_false_positive_and_unmeasured_protected_feature_fail_closed() -> None:
    overreach = _candidate("overreach", error=0.2)
    unmeasured = _candidate(
        "unmeasured",
        error=0.01,
        representation="coacd",
        protected_features=["handle_opening"],
    )
    measured = _candidate(
        "measured",
        error=0.01,
        representation="coacd",
        protected_features=["handle_opening"],
        protected_features_satisfied=True,
    )

    assert overreach.status == "fail"
    assert unmeasured.status == "conditional"
    result = select_collision_candidate("/Asset/Body", [overreach, unmeasured, measured])
    assert result.selected_candidate_id == "measured"


def test_no_passing_candidate_is_explicit_failure() -> None:
    failed = _candidate("failed", error=0.2)

    result = select_collision_candidate("/Asset/Body", [failed])

    assert result.status == "fail"
    assert result.selected_candidate_id is None
    assert result.failures
