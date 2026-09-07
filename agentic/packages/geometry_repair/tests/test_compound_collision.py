# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused compound render/collision pairing and selective-reuse tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh

from geometry_repair.collision import build_collision_geometry
from geometry_repair.collision_audit import (
    CollisionAuditLimits,
    SourceCollisionAudit,
    audit_source_collision_proxy,
)
from geometry_repair.compound_collision import (
    compound_part_audit_map,
    pair_compound_collision_parts,
)
from geometry_repair.models import ProtectedFeature, ProtectedFeatureProbe, RepairBudgets


def _record(path: str, center: tuple[float, float, float], extent: float = 1.0):
    mesh = trimesh.creation.box(extents=(extent, extent, extent))
    mesh.apply_translation(center)
    return SimpleNamespace(
        path=path,
        world_vertices_m=np.asarray(mesh.vertices, dtype=np.float64),
        triangles=np.asarray(mesh.faces, dtype=np.int64),
    )


def _write_compound_stage(
    path: Path,
    *,
    renders: list[tuple[str, tuple[float, float, float], float]],
    collisions: list[tuple[str, tuple[float, float, float], float, str | None]],
) -> Path:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, Vt

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())

    def add_box(
        prim_path: str,
        center: tuple[float, float, float],
        extent: float,
        *,
        collision: bool,
        render_path: str | None = None,
    ) -> None:
        source = trimesh.creation.box(extents=(extent, extent, extent))
        source.apply_translation(center)
        vertices = np.asarray(source.vertices, dtype=np.float64)
        triangles = np.asarray(source.faces, dtype=np.int64)
        mesh = UsdGeom.Mesh.Define(stage, prim_path)
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices.astype(np.float32)))
        mesh.CreateFaceVertexCountsAttr(
            Vt.IntArray.FromNumpy(np.full(len(triangles), 3, dtype=np.int32))
        )
        mesh.CreateFaceVertexIndicesAttr(
            Vt.IntArray.FromNumpy(triangles.astype(np.int32).reshape(-1))
        )
        mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
        minimum = vertices.min(axis=0)
        maximum = vertices.max(axis=0)
        mesh.CreateExtentAttr(
            Vt.Vec3fArray(
                [
                    Gf.Vec3f(*[float(value) for value in minimum]),
                    Gf.Vec3f(*[float(value) for value in maximum]),
                ]
            )
        )
        if collision:
            mesh.CreatePurposeAttr().Set(UsdGeom.Tokens.guide)
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr().Set(True)
            UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr().Set(
                "convexHull"
            )
            if render_path is not None:
                mesh.GetPrim().SetCustomDataByKey("sourcePrimPath", render_path)

    for render_path, center, extent in renders:
        add_box(render_path, center, extent, collision=False)
    for collision_path, center, extent, render_path in collisions:
        add_box(
            collision_path,
            center,
            extent,
            collision=True,
            render_path=render_path,
        )
    stage.GetRootLayer().Save()
    return path


def _limits() -> CollisionAuditLimits:
    return CollisionAuditLimits(
        max_surface_gap_m=0.01,
        max_surface_overreach_m=0.01,
        advisory_hull_count=16,
        max_vertices_per_hull=255,
        max_faces_per_hull=255,
    )


def test_explicit_pairing_precedes_conflicting_geometry() -> None:
    left_render = _record("/Asset/LeftRender", (-2.0, 0.0, 0.0))
    right_render = _record("/Asset/RightRender", (2.0, 0.0, 0.0))
    left_collision = _record("/Asset/UnlabeledOne", (2.0, 0.0, 0.0))
    right_collision = _record("/Asset/UnlabeledTwo", (-2.0, 0.0, 0.0))

    result = pair_compound_collision_parts(
        [right_render, left_render],
        [right_collision, left_collision],
        explicit_targets={
            left_collision.path: left_render.path,
            right_collision.path: right_render.path,
        },
    )

    assert result.status == "pass"
    assert [item.method for item in result.collision_assignments] == [
        "explicit_metadata",
        "explicit_metadata",
    ]
    assert {item.collision_path: item.render_path for item in result.collision_assignments} == {
        left_collision.path: left_render.path,
        right_collision.path: right_render.path,
    }


def test_measured_fallback_is_unique_bounded_and_order_deterministic() -> None:
    renders = [
        _record("/Asset/Alpha", (-2.0, 0.0, 0.0)),
        _record("/Asset/Beta", (2.0, 0.0, 0.0)),
    ]
    collisions = [
        _record("/Asset/ProxyOne", (-2.0, 0.0, 0.0)),
        _record("/Asset/ProxyTwo", (2.0, 0.0, 0.0)),
    ]

    forward = pair_compound_collision_parts(renders, collisions)
    reversed_input = pair_compound_collision_parts(
        list(reversed(renders)),
        list(reversed(collisions)),
    )

    assert forward.model_dump(mode="json") == reversed_input.model_dump(mode="json")
    assert forward.status == "pass"
    assert all(item.method == "measured_unique" for item in forward.collision_assignments)
    assert all(
        sum(measurement.eligible for measurement in item.measurements) == 1
        for item in forward.collision_assignments
    )


@pytest.mark.parametrize(
    ("collision_center", "expected_status"),
    [((0.0, 0.0, 0.0), "ambiguous"), ((20.0, 0.0, 0.0), "unmatched")],
)
def test_non_unique_or_unmatched_measured_pairing_requires_review(
    collision_center: tuple[float, float, float],
    expected_status: str,
) -> None:
    renders = [
        _record("/Asset/Alpha", (0.0, 0.0, 0.0)),
        _record("/Asset/Beta", (0.0, 0.0, 0.0)),
    ]
    collision = _record("/Asset/Unknown", collision_center)

    result = pair_compound_collision_parts(renders, [collision])

    assert result.status == "review_required"
    assert result.collision_assignments[0].status == expected_status
    assert all(part.status == "unmatched" for part in result.render_parts)


def test_measured_pairing_stops_at_explicit_evaluation_budget() -> None:
    renders = [
        _record("/Asset/Alpha", (-2.0, 0.0, 0.0)),
        _record("/Asset/Beta", (2.0, 0.0, 0.0)),
    ]
    collisions = [
        _record("/Asset/UnknownOne", (-2.0, 0.0, 0.0)),
        _record("/Asset/UnknownTwo", (2.0, 0.0, 0.0)),
    ]

    result = pair_compound_collision_parts(
        renders,
        collisions,
        max_pair_evaluations=3,
    )

    assert result.status == "review_required"
    assert all(item.status == "not_evaluated" for item in result.collision_assignments)
    assert all("above limit 3" in item.reason for item in result.collision_assignments)


def test_part_audit_blocks_aggregate_union_false_pass(tmp_path: Path) -> None:
    source = _write_compound_stage(
        tmp_path / "swapped.usda",
        renders=[
            ("/Asset/LeftRender", (-2.0, 0.0, 0.0), 1.0),
            ("/Asset/RightRender", (2.0, 0.0, 0.0), 1.0),
        ],
        collisions=[
            ("/Asset/LeftCollision", (2.0, 0.0, 0.0), 1.0, "/Asset/LeftRender"),
            ("/Asset/RightCollision", (-2.0, 0.0, 0.0), 1.0, "/Asset/RightRender"),
        ],
    )

    report = audit_source_collision_proxy(
        source,
        source,
        limits=_limits(),
        surface_sample_limit=256,
        occupancy_grid_resolution=12,
    )
    parts = compound_part_audit_map(report.metrics.per_render_part_coverage)

    assert report.metrics.false_positive_ratio == pytest.approx(0.0)
    assert report.metrics.false_negative_ratio == pytest.approx(0.0)
    assert report.decision == "regenerate_candidate"
    assert report.status == "fail"
    assert set(parts) == {"/Asset/LeftRender", "/Asset/RightRender"}
    assert all(part.decision == "regenerate_candidate" for part in parts.values())
    assert all(part.coverage_ratio == pytest.approx(0.0) for part in parts.values())


def test_scoped_probe_is_measured_against_assigned_collision_subset(tmp_path: Path) -> None:
    source = _write_compound_stage(
        tmp_path / "scoped_probe.usda",
        renders=[
            ("/Asset/LeftRender", (0.0, 0.0, 0.0), 1.0),
            ("/Asset/RightRender", (0.0, 0.0, 0.0), 1.0),
        ],
        collisions=[
            ("/Asset/LeftCollision", (0.0, 0.0, 0.0), 0.98, "/Asset/LeftRender"),
            ("/Asset/RightCollision", (0.0, 0.0, 0.0), 1.0, "/Asset/RightRender"),
        ],
    )
    feature = ProtectedFeature(
        name="left_contact",
        kind="interface",
        scope_path="/Asset/LeftRender",
        probe=ProtectedFeatureProbe(
            kind="surface_support",
            points_m=[[0.5, 0.0, 0.0]],
            tolerance_m=0.001,
        ),
    )
    limits = CollisionAuditLimits(
        min_render_surface_coverage_ratio=0.95,
        max_volume_excess_ratio=0.10,
        max_volume_deficit_ratio=0.10,
        max_false_positive_ratio=0.10,
        max_false_negative_ratio=0.10,
        max_surface_gap_m=0.10,
        max_surface_overreach_m=0.10,
        advisory_hull_count=16,
        max_vertices_per_hull=255,
        max_faces_per_hull=255,
    )

    report = audit_source_collision_proxy(
        source,
        source,
        protected_features=[feature],
        limits=limits,
        surface_sample_limit=256,
        occupancy_grid_resolution=12,
    )
    parts = compound_part_audit_map(report.metrics.per_render_part_coverage)

    assert report.protected_feature_probes[0].status == "pass"
    assert parts["/Asset/LeftRender"].protected_feature_probes[0].status == "fail"
    assert parts["/Asset/LeftRender"].decision == "regenerate_candidate"
    assert parts["/Asset/RightRender"].decision == "preserve_source"
    assert report.decision == "regenerate_candidate"


def test_builder_reuses_passing_part_and_regenerates_only_failing_part(
    tmp_path: Path,
) -> None:
    source = _write_compound_stage(
        tmp_path / "selective.usda",
        renders=[
            ("/Asset/LeftRender", (-2.0, 0.0, 0.0), 1.0),
            ("/Asset/RightRender", (2.0, 0.0, 0.0), 1.0),
        ],
        collisions=[
            ("/Asset/LeftCollision", (-2.0, 0.0, 0.0), 1.0, "/Asset/LeftRender"),
            ("/Asset/RightCollision", (2.0, 0.0, 0.0), 2.0, "/Asset/RightRender"),
        ],
    )
    output = tmp_path / "collision.usda"
    report_path = tmp_path / "collision_report.json"
    audit_path = tmp_path / "source_collision_audit.json"

    report = build_collision_geometry(
        source,
        output,
        profile="rigid_pick_place",
        budgets=RepairBudgets(max_collision_hulls=1, sample_point_limit=256),
        coacd_enabled=False,
        runtime_engine="skip",
        report_path=report_path,
        source_collision_audit_path=audit_path,
    )
    audit = SourceCollisionAudit.model_validate_json(audit_path.read_text(encoding="utf-8"))
    parts = compound_part_audit_map(audit.metrics.per_render_part_coverage)

    assert output.is_file()
    assert report.representation == "hybrid"
    assert report.generator == "geometry_repair.hybrid_collision"
    assert report.source_collision_prim_count == 1
    assert report.generated_collision_prim_count == 1
    assert parts["/Asset/LeftRender"].decision == "preserve_source"
    assert parts["/Asset/RightRender"].decision == "regenerate_candidate"


def test_builder_does_not_replace_unmatched_compound_part(tmp_path: Path) -> None:
    source = _write_compound_stage(
        tmp_path / "unmatched.usda",
        renders=[
            ("/Asset/LeftRender", (-2.0, 0.0, 0.0), 1.0),
            ("/Asset/RightRender", (2.0, 0.0, 0.0), 1.0),
        ],
        collisions=[
            ("/Asset/LeftCollision", (-2.0, 0.0, 0.0), 1.0, "/Asset/LeftRender"),
        ],
    )
    output = tmp_path / "collision.usda"

    report = build_collision_geometry(
        source,
        output,
        profile="rigid_pick_place",
        budgets=RepairBudgets(sample_point_limit=256),
        runtime_engine="skip",
        report_path=tmp_path / "collision_report.json",
        source_collision_audit_path=tmp_path / "source_collision_audit.json",
    )

    assert report.status == "conditional"
    assert report.representation == "none"
    assert report.generator == "geometry_repair.source_collision_review"
    assert report.collision_path is None
    assert not output.exists()
    assert any("automatic collision replacement is prohibited" in item for item in report.warnings)


def test_builder_reserves_a_generated_hull_for_each_remaining_part(tmp_path: Path) -> None:
    source = _write_compound_stage(
        tmp_path / "two_render_parts.usda",
        renders=[
            ("/Asset/LeftRender", (-1.0, 0.0, 0.0), 1.0),
            ("/Asset/RightRender", (1.0, 0.0, 0.0), 1.0),
        ],
        collisions=[],
    )

    report = build_collision_geometry(
        source,
        tmp_path / "collision.usda",
        profile="rigid_pick_place",
        budgets=RepairBudgets(max_collision_hulls=2, sample_point_limit=256),
        coacd_enabled=False,
        runtime_engine="skip",
        report_path=tmp_path / "collision_report.json",
    )

    assert report.status == "conditional"
    assert report.hull_count == 2
    assert len(report.candidate_search_paths) == 2
    for search_path in report.candidate_search_paths:
        search = json.loads(Path(search_path).read_text(encoding="utf-8"))
        selected = next(
            candidate
            for candidate in search["candidates"]
            if candidate["candidate_id"] == search["selected_candidate_id"]
        )
        hull_gate = next(gate for gate in selected["gates"] if gate["gate_id"] == "hull_count")
        assert hull_gate["limit"] == 1
