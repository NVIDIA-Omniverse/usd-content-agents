# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Phase 0 tests for deterministic, budgeted intersection diagnosis."""

from __future__ import annotations

import math

import numpy as np
import pytest

from geometry_repair.scalable_audit import ScalableAuditBudget, audit_triangle_mesh


def _separated_triangles(count: int) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.arange(count, dtype=np.float64) * 2.0
    vertices = np.empty((count * 3, 3), dtype=np.float64)
    vertices[0::3] = np.column_stack((offsets, np.zeros(count), np.zeros(count)))
    vertices[1::3] = np.column_stack((offsets + 0.4, np.zeros(count), np.zeros(count)))
    vertices[2::3] = np.column_stack((offsets, np.full(count, 0.4), np.zeros(count)))
    triangles = np.arange(count * 3, dtype=np.int64).reshape((-1, 3))
    return vertices, triangles


def test_audit_evaluates_fifty_thousand_faces_without_legacy_triangle_cap() -> None:
    vertices, triangles = _separated_triangles(50_000)
    vertices_before = vertices.copy()
    triangles_before = triangles.copy()
    budget = ScalableAuditBudget(
        wall_time_s=30.0,
        memory_limit_bytes=256 * 1024 * 1024,
        chunk_size=4096,
    )

    report = audit_triangle_mesh(vertices, triangles, budget=budget)

    assert report.self_intersection.status == "evaluated_pass"
    assert report.coplanar_overlap.status == "evaluated_pass"
    assert report.self_intersection.complete is True
    assert report.resources.input_triangle_count == 50_000
    assert report.resources.evaluated_triangle_count == 50_000
    assert report.resources.chunk_count == math.ceil(50_000 / 4096)
    assert report.resources.chunks_completed == report.resources.chunk_count
    assert report.resources.resource_limit is None
    assert np.array_equal(vertices, vertices_before)
    assert np.array_equal(triangles, triangles_before)


def test_audit_is_deterministic_and_separates_cross_part_coplanar_findings() -> None:
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.2, 0.2, 0.0],
            [1.4, 0.2, 0.0],
            [0.2, 1.4, 0.0],
        ],
        dtype=np.float64,
    )
    triangles = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    budget = ScalableAuditBudget(chunk_size=1)

    first = audit_triangle_mesh(vertices, triangles, part_ids=["body", "insert"], budget=budget)
    second = audit_triangle_mesh(vertices, triangles, part_ids=["body", "insert"], budget=budget)

    assert first.geometry_sha256 == second.geometry_sha256
    assert first.self_intersection.status == "evaluated_fail"
    assert first.coplanar_overlap.status == "evaluated_fail"
    assert first.self_intersection.finding_count == 1
    assert first.coplanar_overlap.finding_count == 1
    assert first.self_intersection.cross_part_count == 1
    assert first.self_intersection.within_part_count == 0
    assert first.self_intersection.sample_pairs == second.self_intersection.sample_pairs
    assert first.resources.model_dump(exclude={"elapsed_s"}) == second.resources.model_dump(
        exclude={"elapsed_s"}
    )
    assert first.mesh_metrics_compatibility_fields() == {
        "self_intersection_status": "fail",
        "self_intersection_count": 1,
        "self_intersection_broad_phase_pairs": 1,
        "self_intersection_candidate_pairs": 1,
        "self_intersection_reason": None,
        "coplanar_overlap_status": "fail",
        "coplanar_overlap_count": 1,
    }


def test_memory_limit_is_a_skip_not_a_geometry_failure() -> None:
    vertices, triangles = _separated_triangles(8)
    report = audit_triangle_mesh(
        vertices,
        triangles,
        budget=ScalableAuditBudget(memory_limit_bytes=1024),
    )

    assert report.self_intersection.status == "skipped_resource_limit"
    assert report.coplanar_overlap.status == "skipped_resource_limit"
    assert report.self_intersection.finding_count == 0
    assert report.resources.resource_limit == "memory"
    assert report.resources.estimated_working_set_bytes > report.resources.memory_limit_bytes
    compatibility = report.mesh_metrics_compatibility_fields()
    assert compatibility["self_intersection_status"] == "not_evaluated"
    assert compatibility["coplanar_overlap_status"] == "not_evaluated"
    assert str(compatibility["self_intersection_reason"]).startswith("skipped_resource_limit:")


def test_invalid_indices_are_indeterminate_not_a_geometry_failure() -> None:
    vertices = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    triangles = np.asarray([[0, 1, 9]], dtype=np.int64)

    report = audit_triangle_mesh(vertices, triangles)

    assert report.self_intersection.status == "indeterminate"
    assert report.coplanar_overlap.status == "indeterminate"
    assert report.resources.resource_limit is None
    assert "out-of-range" in (report.self_intersection.reason or "")


def test_wall_time_limit_is_reported_independently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import geometry_repair.scalable_audit as scalable_audit

    call_count = 0

    def deterministic_clock() -> float:
        nonlocal call_count
        call_count += 1
        return 0.0 if call_count <= 2 else 2.0

    monkeypatch.setattr(scalable_audit.time, "monotonic", deterministic_clock)
    vertices, triangles = _separated_triangles(4)
    report = audit_triangle_mesh(
        vertices,
        triangles,
        budget=ScalableAuditBudget(wall_time_s=1.0),
    )

    assert report.self_intersection.status == "skipped_resource_limit"
    assert report.resources.resource_limit == "wall_time"
    assert report.resources.broad_pairs_examined == 0
    assert report.resources.exact_pair_tests == 0


def test_resource_cutoff_retains_concrete_failure_witness() -> None:
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.2, 0.2, 0.0],
            [2.0, 0.2, 0.0],
            [0.2, 2.0, 0.0],
            [0.4, 0.4, 0.0],
            [1.0, 0.4, 0.0],
            [0.4, 1.0, 0.0],
        ]
    )
    triangles = np.arange(9, dtype=np.int64).reshape((-1, 3))
    report = audit_triangle_mesh(
        vertices,
        triangles,
        part_ids=["a", "b", "c"],
        budget=ScalableAuditBudget(broad_pair_limit=1, chunk_size=1),
    )

    assert report.resources.resource_limit == "broad_pairs"
    assert report.resources.broad_pairs_examined == 1
    assert report.self_intersection.status == "evaluated_fail"
    assert report.self_intersection.complete is False
    assert report.self_intersection.finding_count == 1
    assert report.coplanar_overlap.status == "evaluated_fail"
    assert report.warnings


@pytest.mark.parametrize(
    ("budget", "expected_limit"),
    [
        (ScalableAuditBudget(broad_pair_limit=1, exact_test_limit=10), "broad_pairs"),
        (ScalableAuditBudget(broad_pair_limit=10, exact_test_limit=1), "exact_tests"),
    ],
)
def test_pair_budgets_stop_without_converting_no_finding_to_pass(
    monkeypatch: pytest.MonkeyPatch,
    budget: ScalableAuditBudget,
    expected_limit: str,
) -> None:
    import trimesh

    class DeterministicTree:
        def intersection(self, _bounds):
            return [0, 1, 2]

    monkeypatch.setattr(trimesh.util, "bounds_tree", lambda _bounds: DeterministicTree())
    vertices, triangles = _separated_triangles(3)

    report = audit_triangle_mesh(vertices, triangles, budget=budget)

    assert report.self_intersection.status == "skipped_resource_limit"
    assert report.coplanar_overlap.status == "skipped_resource_limit"
    assert report.resources.resource_limit == expected_limit
    assert report.self_intersection.finding_count == 0
    assert report.resources.broad_pairs_examined <= budget.broad_pair_limit
    assert report.resources.exact_pair_tests <= budget.exact_test_limit
