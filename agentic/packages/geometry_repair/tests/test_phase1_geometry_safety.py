# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused safety tests for Phase 1 protection, correspondence, seams, and collision."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from geometry_repair.collision import _run_source_collision_audit
from geometry_repair.collision_audit import (
    CollisionAuditLimits,
    CollisionAuditMetrics,
    CollisionComplexity,
    _occupancy_metrics,
    _sample_probe_polyline,
    audit_source_collision_proxy,
    evaluate_source_collision_gates,
)
from geometry_repair.correspondence import (
    CornerMapping,
    identity_correspondence,
    resolve_categorical_assignment,
    resolve_generated_patch_categorical_assignment,
    transfer_face_varying_values,
)
from geometry_repair.models import (
    ProtectedFeature,
    ProtectedFeatureProbe,
    ProtectedFeatureProbeResult,
    RepairOperation,
    RepairRequest,
)
from geometry_repair.orchestrator import run_geometry_repair
from geometry_repair.protected_features import (
    ProtectedFeatureCandidate,
    _accessible_void_rays,
    _sample_polyline,
    candidate_to_protected_feature,
    compare_protected_feature_candidates,
    detect_protected_feature_candidates,
    evaluate_feature_probe,
)
from geometry_repair.workers.manifold_seam import (
    ManifoldSeamWorker,
    analyze_manifold_seams,
)


def _export_mesh(path: Path, mesh) -> Path:
    mesh.export(path)
    return path


def _open_cylinder():
    import trimesh

    mesh = trimesh.creation.cylinder(radius=0.5, height=2.0, sections=32)
    keep = np.asarray(mesh.face_normals)[:, 2] < 0.9
    return trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.faces)[keep],
        process=False,
    )


def _parallel_rods():
    import trimesh

    rods = []
    for x in (-0.15, 0.15):
        rod = trimesh.creation.cylinder(radius=0.025, height=1.0, sections=12)
        rod.apply_translation((x, 0.0, 0.0))
        rods.append(rod)
    return trimesh.util.concatenate(rods)


def _watertight_cup():
    import trimesh

    radial_height_profile = np.asarray(
        [
            [0.0, 0.0],
            [0.6, 0.0],
            [0.6, 1.2],
            [0.45, 1.2],
            [0.45, 0.15],
            [0.0, 0.15],
        ]
    )
    return trimesh.creation.revolve(radial_height_profile, sections=32)


def test_cavity_proximity_failure_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    import trimesh

    cup = _watertight_cup()

    def fail_proximity(*_args, **_kwargs):
        raise RuntimeError("proximity backend unavailable")

    monkeypatch.setattr(trimesh.proximity, "closest_point", fail_proximity)
    _candidates, warning = _accessible_void_rays(
        np.asarray(cup.vertices),
        np.asarray(cup.faces),
        tolerance=1e-6,
    )

    assert warning is not None
    assert "results may be incomplete" in warning
    assert "RuntimeError: proximity backend unavailable" in warning


def _property_vertex_cube() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import trimesh

    cube = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    vertices = np.asarray(cube.vertices)[np.asarray(cube.faces)].reshape((-1, 3))
    triangles = np.arange(len(vertices), dtype=np.int64).reshape((-1, 3))
    properties = np.column_stack(
        (
            np.arange(len(vertices), dtype=np.float64) / max(len(vertices) - 1, 1),
            np.tile([0.0, 1.0, 0.5], len(vertices) // 3),
        )
    )
    return vertices, triangles, properties


def test_explicit_protected_feature_behavior_is_unchanged(tmp_path: Path) -> None:
    import trimesh

    source = _export_mesh(tmp_path / "box.ply", trimesh.creation.box())
    feature = ProtectedFeature(
        name="top_support",
        kind="mating_surface",
        probe=ProtectedFeatureProbe(
            kind="surface_support",
            points_m=[[0.0, 0.0, 0.5]],
            tolerance_m=1e-6,
        ),
    )

    result = evaluate_feature_probe(source, feature)

    assert result.status == "pass"
    assert result.maximum_surface_distance_m == pytest.approx(0.0, abs=1e-12)
    assert feature.affected_roles == ["render", "collision"]


def test_small_topological_boundaries_remain_review_only(tmp_path: Path) -> None:
    import trimesh

    box = trimesh.creation.box()
    keep = np.asarray(box.face_normals)[:, 2] < 0.9
    open_box = trimesh.Trimesh(
        vertices=np.asarray(box.vertices),
        faces=np.asarray(box.faces)[keep],
        process=False,
    )
    report = detect_protected_feature_candidates(
        _export_mesh(tmp_path / "open_box.ply", open_box),
        confidence_threshold=0.0,
    )
    boundary_candidates = [
        candidate
        for candidate in report.candidates
        if candidate.evidence.get("measurement") == "position_welded_boundary_loop"
    ]

    assert boundary_candidates
    assert all(
        candidate.disposition == "review_only"
        for candidate in boundary_candidates
        if candidate.evidence.get("loop_vertex_count", 0) < 5
    )


def test_inferred_feature_adapter_is_render_scoped() -> None:
    candidate = ProtectedFeatureCandidate(
        candidate_id="/Asset/Mesh:opening:0000",
        kind="opening",
        scope_path="/Asset/Mesh",
        confidence=0.9,
        disposition="protect",
        mutation_authorized=False,
        probe=ProtectedFeatureProbe(
            kind="negative_space_path",
            points_m=[[0.0, 0.0, -0.1], [0.0, 0.0, 0.1]],
            radius_m=0.01,
        ),
        required_clearance_m=0.01,
        evidence={"measurement": "test"},
    )

    feature = candidate_to_protected_feature(candidate)

    assert feature.source == "inferred_hypothesis"
    assert feature.required is True
    assert feature.affected_roles == ["render"]


def test_bounded_polyline_sampling_covers_the_complete_path() -> None:
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
        ]
    )

    samples, truncated, required_count = _sample_polyline(points, step_m=0.01, limit=5)

    assert truncated is True
    assert required_count == 201
    assert samples.tolist() == [
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.5, 0.0],
        [1.0, 1.0, 0.0],
    ]


def test_truncated_negative_space_probe_cannot_pass_but_still_reports_obstruction(
    tmp_path: Path,
) -> None:
    import trimesh

    source = _export_mesh(tmp_path / "box.ply", trimesh.creation.box())
    clear = ProtectedFeature(
        name="clear_path",
        kind="opening",
        probe=ProtectedFeatureProbe(
            kind="negative_space_path",
            points_m=[[-1.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
            radius_m=0.1,
            tolerance_m=1e-5,
        ),
    )
    obstructed = clear.model_copy(
        update={
            "name": "obstructed_path",
            "probe": clear.probe.model_copy(
                update={"points_m": [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]}
            ),
        }
    )

    clear_result = evaluate_feature_probe(source, clear, sample_limit=8)
    obstructed_result = evaluate_feature_probe(source, obstructed, sample_limit=8)

    assert clear_result.status == "not_evaluated"
    assert clear_result.sample_count == 8
    assert "uniformly across the full path" in clear_result.warnings[0]
    assert obstructed_result.status == "fail"
    assert obstructed_result.occupied_sample_count > 0


def test_duplicate_review_candidates_share_bounded_probe_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.usd"
    candidate = tmp_path / "candidate.usd"
    source.write_bytes(b"source")
    candidate.write_bytes(b"candidate")
    probe = ProtectedFeatureProbe(
        kind="negative_space_path",
        points_m=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        radius_m=0.01,
    )
    inferred = [
        ProtectedFeatureCandidate(
            candidate_id=f"/Asset/Mesh:cavity:{index:04d}",
            kind="cavity",
            scope_path="/Asset/Mesh",
            confidence=0.4,
            disposition="review_only",
            mutation_authorized=False,
            probe=probe,
            evidence={"measurement": "duplicate-test"},
        )
        for index in range(2)
    ]
    calls: list[tuple[str, int]] = []

    def fake_evaluate(
        geometry_path,
        feature,
        *,
        include_guide_purpose=True,
        sample_limit=4096,
    ) -> ProtectedFeatureProbeResult:
        del feature, include_guide_purpose
        calls.append((str(geometry_path), sample_limit))
        return ProtectedFeatureProbeResult(
            feature_name="cached",
            probe_kind="negative_space_path",
            status="pass",
            sample_count=sample_limit,
        )

    monkeypatch.setattr(
        "geometry_repair.protected_features.evaluate_feature_probe",
        fake_evaluate,
    )

    report = compare_protected_feature_candidates(source, candidate, inferred)

    assert calls == [(str(source.resolve()), 128), (str(candidate.resolve()), 128)]
    assert [item.status for item in report.comparisons] == ["preserved", "preserved"]
    assert report.status == "conditional"


def test_candidate_detection_covers_negative_space_and_thin_features(tmp_path: Path) -> None:
    import trimesh

    cup = _export_mesh(tmp_path / "cup.ply", _open_cylinder())
    watertight_cup = _export_mesh(tmp_path / "watertight_cup.ply", _watertight_cup())
    rods = _export_mesh(tmp_path / "rods.ply", _parallel_rods())
    wire = _export_mesh(
        tmp_path / "wire.ply",
        trimesh.creation.cylinder(radius=0.025, height=1.0, sections=12),
    )
    torus = _export_mesh(
        tmp_path / "torus.ply",
        trimesh.creation.torus(major_radius=0.5, minor_radius=0.1),
    )

    candidates = [
        *detect_protected_feature_candidates(cup).candidates,
        *detect_protected_feature_candidates(watertight_cup).candidates,
        *detect_protected_feature_candidates(rods).candidates,
        *detect_protected_feature_candidates(wire).candidates,
        *detect_protected_feature_candidates(torus).candidates,
    ]
    kinds = {candidate.kind for candidate in candidates}

    assert {"opening", "cavity", "rim", "wire", "tine", "handle"} <= kinds
    assert all(candidate.mutation_authorized is False for candidate in candidates)
    cavity = next(candidate for candidate in candidates if candidate.kind == "cavity")
    assert cavity.disposition == "review_only"
    assert all(
        candidate.disposition == "review_only"
        for candidate in candidates
        if candidate.kind == "handle"
    )
    assert not any(
        candidate.kind == "opening"
        and candidate.evidence.get("measurement") == "convex_hull_accessible_void_rays"
        for candidate in candidates
    )
    assert any(
        candidate.evidence.get("measurement") == "convex_hull_accessible_void_rays"
        for candidate in candidates
        if candidate.kind in {"opening", "cavity"}
    )


def test_low_confidence_candidate_cannot_authorize_mutation_and_regression_fails(
    tmp_path: Path,
) -> None:
    import trimesh

    source_mesh = trimesh.creation.box()
    candidate_mesh = source_mesh.copy()
    candidate_mesh.apply_translation((2.0, 0.0, 0.0))
    source = _export_mesh(tmp_path / "source.ply", source_mesh)
    candidate = _export_mesh(tmp_path / "candidate.ply", candidate_mesh)
    inferred = ProtectedFeatureCandidate(
        candidate_id="/Asset/Mesh:wire:0000",
        kind="wire",
        scope_path="/Asset/Mesh",
        confidence=0.4,
        disposition="review_only",
        mutation_authorized=False,
        probe=ProtectedFeatureProbe(
            kind="surface_support",
            points_m=[[0.5, 0.0, 0.0]],
            tolerance_m=1e-4,
        ),
        evidence={"measurement": "test"},
    )

    report = compare_protected_feature_candidates(source, candidate, [inferred])

    assert report.comparisons[0].status == "regressed"
    assert report.comparisons[0].mutation_authorized is False
    assert report.status == "conditional"
    assert any("low-confidence" in warning for warning in report.warnings)


def test_identity_correspondence_is_hashed_and_different_output_refuses(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"identical source")
    same = identity_correspondence(
        source,
        vertex_count=8,
        face_count=12,
        part_ids=["body"],
        prim_paths=["/Asset/body"],
        attributes={"st": ("corner", "faceVarying")},
    )
    changed = tmp_path / "changed.bin"
    changed.write_bytes(b"changed output")
    refused = identity_correspondence(
        source,
        changed,
        vertex_count=8,
        face_count=12,
    )

    assert same.status == "pass"
    assert same.evidence_sha256 is not None
    assert same.attribute_transfers[0].method == "exact_source_corner"
    assert refused.status == "refused"
    assert "byte-identical" in refused.refusal_reasons[0]


def test_face_varying_transfer_uses_source_face_corners_and_barycentrics() -> None:
    source_uvs = np.asarray([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])
    mappings = [
        CornerMapping(output_face=0, output_corner=0, source_face=0, source_corner=0),
        CornerMapping(output_face=0, output_corner=1, source_face=0, source_corner=1),
        CornerMapping(
            output_face=0,
            output_corner=2,
            source_face=0,
            barycentric=[0.25, 0.25, 0.5],
        ),
    ]

    result = transfer_face_varying_values(
        "st",
        source_uvs,
        mappings,
        output_face_count=1,
    )

    assert result.status == "pass"
    assert result.transfer.method == "mixed_source_face_corner"
    assert np.asarray(result.values)[0, 2].tolist() == pytest.approx([0.25, 0.5])


def test_categorical_conflict_refuses_implicit_assignment() -> None:
    resolution = resolve_categorical_assignment("material", ["steel", "plastic", "steel"])

    assert resolution.status == "conflict"
    assert resolution.value is None
    assert resolution.contributing_values == ["plastic", "steel"]
    assert "split" in (resolution.reason or "")

    incomplete_patch = resolve_generated_patch_categorical_assignment(
        "material",
        ["steel"],
        boundary_complete=False,
    )
    assert incomplete_patch.status == "unassigned"
    assert "incomplete" in (incomplete_patch.reason or "")


def test_robocasa_mug_metrics_have_explicit_regeneration_reasons(tmp_path: Path) -> None:
    render = tmp_path / "render.usd"
    collision = tmp_path / "collision.usd"
    render.write_text("render", encoding="utf-8")
    collision.write_text("collision", encoding="utf-8")
    report = evaluate_source_collision_gates(
        render_path=render,
        collision_path=collision,
        metrics=CollisionAuditMetrics(
            render_surface_sample_count=4096,
            collision_surface_sample_count=4096,
            render_surface_coverage_ratio=1.0,
            surface_gap_p99_m=0.00258,
            surface_overreach_p99_m=0.01135,
            false_positive_ratio=0.238602,
            false_negative_ratio=0.0,
            volume_excess_ratio=0.238602,
            volume_deficit_ratio=0.0,
            occupancy_status="evaluated",
            occupancy_grid_resolution=24,
        ),
        complexity=CollisionComplexity(
            hull_count=32,
            total_vertices=4096,
            total_faces=8192,
            maximum_vertices_per_hull=128,
            maximum_faces_per_hull=254,
        ),
        limits=CollisionAuditLimits(
            max_surface_gap_m=0.005,
            max_surface_overreach_m=0.005,
            advisory_hull_count=16,
        ),
    )

    by_id = {gate.gate_id: gate for gate in report.gates}
    assert report.decision == "regenerate_candidate"
    assert by_id["false_positive_cavity_gap_occupation"].status == "fail"
    assert by_id["surface_overreach_p99_m"].status == "fail"
    assert by_id["source_proxy_hull_count"].status == "warning"
    assert by_id["source_proxy_hull_count"].blocking is False
    assert len(report.regeneration_reasons) == 3
    assert all("hull" not in reason for reason in report.regeneration_reasons)


def test_hull_count_alone_preserves_passing_source_proxy(tmp_path: Path) -> None:
    render = tmp_path / "render.usd"
    collision = tmp_path / "collision.usd"
    render.write_text("render", encoding="utf-8")
    collision.write_text("collision", encoding="utf-8")
    report = evaluate_source_collision_gates(
        render_path=render,
        collision_path=collision,
        metrics=CollisionAuditMetrics(
            render_surface_coverage_ratio=1.0,
            surface_gap_p99_m=0.001,
            surface_overreach_p99_m=0.001,
            volume_excess_ratio=0.01,
            volume_deficit_ratio=0.0,
            false_positive_ratio=0.01,
            false_negative_ratio=0.0,
            occupancy_status="evaluated",
            occupancy_grid_resolution=24,
        ),
        complexity=CollisionComplexity(
            hull_count=32,
            total_vertices=4096,
            total_faces=8192,
        ),
        limits=CollisionAuditLimits(
            max_surface_gap_m=0.005,
            max_surface_overreach_m=0.005,
            advisory_hull_count=16,
        ),
    )

    assert report.decision == "preserve_source"
    assert report.status == "conditional"
    assert report.regeneration_reasons == []
    assert "advisory only" in report.warnings[0]


def test_orchestrator_preserves_passing_authored_collision_with_audit(
    tmp_path: Path,
) -> None:
    from pxr import Usd, UsdGeom, UsdPhysics, Vt

    source = tmp_path / "source_collision.usda"
    stage = Usd.Stage.CreateNew(str(source))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    points = [
        (-0.5, -0.5, -0.5),
        (0.5, -0.5, -0.5),
        (0.5, 0.5, -0.5),
        (-0.5, 0.5, -0.5),
        (-0.5, -0.5, 0.5),
        (0.5, -0.5, 0.5),
        (0.5, 0.5, 0.5),
        (-0.5, 0.5, 0.5),
    ]
    faces = [
        (0, 2, 1),
        (0, 3, 2),
        (4, 5, 6),
        (4, 6, 7),
        (0, 1, 5),
        (0, 5, 4),
        (1, 2, 6),
        (1, 6, 5),
        (2, 3, 7),
        (2, 7, 6),
        (3, 0, 4),
        (3, 4, 7),
    ]
    for path, collision in (("/Asset/Render", False), ("/Asset/Collision", True)):
        mesh = UsdGeom.Mesh.Define(stage, path)
        mesh.CreatePointsAttr(Vt.Vec3fArray(points))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(faces)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([value for face in faces for value in face]))
        mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
        if collision:
            mesh.CreatePurposeAttr().Set(UsdGeom.Tokens.guide)
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr().Set(True)
            UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr().Set(
                "convexHull"
            )
    stage.GetRootLayer().Save()

    result = run_geometry_repair(
        RepairRequest(
            source_path=source,
            output_dir=tmp_path / "repair",
            profile="rigid_pick_place",
            mode="auto",
            collision_runtime_engine="fake",
        )
    )

    collision_report = Path(result.collision_report_path or "")
    report = json.loads(collision_report.read_text(encoding="utf-8"))
    audit = json.loads(Path(result.source_collision_audit_path or "").read_text(encoding="utf-8"))
    assert audit["decision"] == "preserve_source"
    assert Path(audit["collision_path"]).name == "source_collision_geometry.usda"
    assert Path(audit["collision_path"]).is_file()
    assert report["generator"] == "geometry_repair.source_collision_reuse"
    assert report["source_collision_prim_count"] == 1
    assert report["generated_collision_prim_count"] == 0


def test_collision_audit_detects_convex_proxy_filling_protected_torus_gap(
    tmp_path: Path,
) -> None:
    import trimesh

    render_mesh = trimesh.creation.torus(major_radius=0.5, minor_radius=0.15)
    collision_mesh = render_mesh.convex_hull
    render = _export_mesh(tmp_path / "torus.ply", render_mesh)
    collision = _export_mesh(tmp_path / "torus_collision.ply", collision_mesh)
    protected = ProtectedFeature(
        name="handle_gap",
        kind="clearance",
        minimum_clearance_m=0.1,
        probe=ProtectedFeatureProbe(
            kind="negative_space_path",
            points_m=[[0.0, 0.0, -0.1], [0.0, 0.0, 0.1]],
            radius_m=0.1,
            tolerance_m=1e-4,
        ),
    )

    report = audit_source_collision_proxy(
        render,
        collision,
        protected_features=[protected],
        limits=CollisionAuditLimits(
            max_surface_gap_m=0.1,
            max_surface_overreach_m=0.1,
            advisory_hull_count=16,
        ),
        occupancy_grid_resolution=16,
    )

    assert report.decision == "regenerate_candidate"
    assert report.protected_feature_probes[0].status == "fail"
    assert any(gate.gate_id == "protected_feature:handle_gap" for gate in report.gates)


def test_collision_occupancy_skips_before_queries_when_work_budget_is_exceeded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import trimesh

    render_mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.5)
    collision_mesh = render_mesh.convex_hull

    def _unexpected_contains(*_args, **_kwargs):
        raise AssertionError("over-budget occupancy must not execute containment queries")

    monkeypatch.setattr(
        "geometry_repair.collision._contains_bounded",
        _unexpected_contains,
    )

    result = _occupancy_metrics(
        [render_mesh],
        [collision_mesh],
        resolution=16,
        face_point_limit=100,
    )

    assert result["status"] == "not_evaluated"
    assert result["estimated_face_point_product"] > result["face_point_limit"]
    assert "exceeds budget" in str(result["reason"])


def test_collision_occupancy_reduces_resolution_to_fit_work_budget() -> None:
    import trimesh

    render_mesh = trimesh.creation.icosphere(subdivisions=1, radius=0.5)
    collision_mesh = render_mesh.copy()
    face_count = len(render_mesh.faces) + len(collision_mesh.faces)

    result = _occupancy_metrics(
        [render_mesh],
        [collision_mesh],
        resolution=16,
        face_point_limit=face_count * 10**3,
    )

    assert result["status"] == "evaluated"
    assert result["resolution"] == 10
    assert result["estimated_face_point_product"] == face_count * 10**3
    assert result["false_positive_ratio"] == pytest.approx(0.0)
    assert result["false_negative_ratio"] == pytest.approx(0.0)
    assert "reduced from 16 to 10" in str(result["reason"])


def test_compound_probe_polyline_covers_full_path_when_bounded() -> None:
    points = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
        dtype=np.float64,
    )

    samples, truncated, required_count = _sample_probe_polyline(
        points,
        step_m=0.001,
        limit=9,
    )

    assert truncated is True
    assert required_count == 2001
    assert len(samples) == 9
    assert samples[0].tolist() == points[0].tolist()
    assert samples[-1].tolist() == points[-1].tolist()


def test_isolated_source_collision_audit_timeout_is_review_required(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    render = tmp_path / "render.usda"
    collision = tmp_path / "source_collision.usda"
    report_path = tmp_path / "source_collision_audit.json"
    render.write_text("render\n", encoding="utf-8")
    collision.write_text("collision\n", encoding="utf-8")
    report_path.write_text('{"stale": true}\n', encoding="utf-8")

    def _timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd="collision-audit", timeout=0.01)

    monkeypatch.setattr("geometry_repair.collision.subprocess.run", _timeout)
    report = _run_source_collision_audit(
        render=render,
        collision=collision,
        report_path=report_path,
        limits=CollisionAuditLimits(),
        protected_features=[],
        payloads=[
            (
                "/Asset/Collision",
                np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
                np.asarray([[0, 1, 2]], dtype=np.int64),
                "source_collision_hull",
            )
        ],
        surface_sample_limit=128,
        occupancy_face_point_limit=1000,
        memory_mb=512,
        timeout_s=0.01,
    )

    assert report.status == "not_evaluated"
    assert report.decision == "review_required"
    assert "wall-clock budget" in report.warnings[0]
    assert report_path.is_file()


def test_manifold_seam_analysis_refuses_real_boundary_without_backend() -> None:
    vertices, triangles, properties = _property_vertex_cube()
    open_triangles = triangles[:-2]

    analysis = analyze_manifold_seams(vertices, open_triangles, properties=properties)

    assert analysis.status == "refused"
    assert analysis.positional_boundary_edge_count > 0
    assert any("real openings" in reason for reason in analysis.refusal_reasons)


def test_manifold_seam_analysis_fails_closed_when_capability_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from geometry_repair.workers import manifold_seam

    vertices, triangles, properties = _property_vertex_cube()
    monkeypatch.setattr(
        manifold_seam,
        "_load_manifold_backend",
        lambda: (None, None, "manifold3d unavailable for test"),
    )

    analysis = analyze_manifold_seams(vertices, triangles, properties=properties)

    assert analysis.status == "unavailable"
    assert analysis.refusal_reasons == ["manifold3d unavailable for test"]


def test_manifold_seam_analysis_preserves_exact_surface_and_properties() -> None:
    pytest.importorskip("manifold3d")
    available, reason = ManifoldSeamWorker().available()
    if not available:
        pytest.skip(reason or "audited manifold3d capability unavailable")
    vertices, triangles, properties = _property_vertex_cube()

    analysis = analyze_manifold_seams(vertices, triangles, properties=properties)

    assert analysis.status == "merge_vectors"
    assert analysis.indexed_boundary_edge_count > 0
    assert analysis.positional_boundary_edge_count == 0
    assert analysis.merge_from_vert
    assert analysis.positions_preserved
    assert analysis.properties_preserved
    assert analysis.surface_preserved
    assert analysis.face_ids_preserved
    assert analysis.run_original_ids_preserved
    assert analysis.source_surface_sha256 == analysis.roundtrip_surface_sha256
    assert analysis.property_rows_sha256 == analysis.roundtrip_property_rows_sha256


def test_manifold_worker_rejects_generic_repair_operation(tmp_path: Path) -> None:
    operation = RepairOperation(
        operation_id="attempt-00",
        worker="manifold_restore_merge_vectors",
        implementation="manifold3d",
        parameters={"operation": "generic_mesh_repair"},
        drift_band="identity",
        source_checkpoint="source",
    )

    result = ManifoldSeamWorker().execute(
        source=tmp_path / "unused.ply",
        output=tmp_path / "report.json",
        operation=operation,
    )

    assert result.status == "unavailable"
    assert "only the typed" in result.failures[0]
