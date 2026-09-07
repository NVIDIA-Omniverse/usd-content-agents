# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused tests for truthful, profile-driven geometry repair."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import trimesh

from geometry_repair import (
    ProtectedFeature,
    ProtectedFeatureProbe,
    RepairBudgets,
    RepairIntent,
    RepairRequest,
    run_geometry_repair,
)
from geometry_repair.artifacts import classify_unresolved_dependencies, file_sha256
from geometry_repair.brep import heal_brep
from geometry_repair.cli import _parser
from geometry_repair.collision import (
    _approximation_metrics,
    _bounded_convex_hull_reduction,
    _bounded_decomposition_input,
    _record_protected_feature_probe_outcomes,
    _run_coacd,
    _validated_source_collision_candidate,
    build_collision_geometry,
)
from geometry_repair.diagnosis import _mesh_issues, diagnose_asset
from geometry_repair.evaluation import phase2_candidate_decisions, run_cgal_exact_audit
from geometry_repair.fidelity import compare_geometry
from geometry_repair.mesh_io import _self_intersections
from geometry_repair.models import (
    GeometryMetrics,
    MeshMetrics,
    ProtectedFeatureProbeResult,
    RepairOperation,
)
from geometry_repair.orchestrator import (
    _admitted_worker_verified_issue_ids,
    _kill_worker_process_group,
    _pin_scene_optimizer_dependency_roots,
    _plan_repair,
    _run_worker_subprocess,
)
from geometry_repair.policy import assert_worker_enabled, worker_registry
from geometry_repair.worker_runner import _worker_version
from geometry_repair.workers.base import WorkerResult
from geometry_repair.workers.geogram_local_repair import (
    GEOGRAM_EXECUTABLE_SHA256_ENVIRONMENT_VARIABLE,
    GeogramLocalRepairWorker,
)
from geometry_repair.workers.scene_optimizer_deinstance import (
    SceneOptimizerDeinstanceWorker,
)
from geometry_repair.workers.sdf_rebuild import (
    OPENVDB_IMPLEMENTATION_VERSION,
    SdfRebuildWorker,
)
from geometry_repair.workers.trimesh_hole_fill import TrimeshBoundedHoleFillWorker


def _write_cube(
    path: Path,
    *,
    duplicate_face: bool = False,
    degenerate_face: bool = False,
    open_face: bool = False,
) -> Path:
    from pxr import Gf, Usd, UsdGeom, Vt

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/Asset/Mesh")
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
    if open_face:
        faces = faces[:-2]
    if duplicate_face:
        faces.append(faces[0])
    if degenerate_face:
        faces.append((0, 0, 0))
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*point) for point in points]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(faces)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([index for face in faces for index in face]))
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(-0.5, -0.5, -0.5), Gf.Vec3f(0.5, 0.5, 0.5)]))
    stage.GetRootLayer().Save()
    return path


def _write_trimesh(path: Path, source_mesh) -> Path:
    from pxr import Gf, Usd, UsdGeom, Vt

    vertices = source_mesh.vertices
    faces = source_mesh.faces
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/Asset/Mesh")
    mesh.CreatePointsAttr(
        Vt.Vec3fArray([Gf.Vec3f(*[float(value) for value in point]) for point in vertices])
    )
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(faces)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([int(index) for face in faces for index in face]))
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    minimum, maximum = source_mesh.bounds
    mesh.CreateExtentAttr(
        Vt.Vec3fArray(
            [
                Gf.Vec3f(*[float(value) for value in minimum]),
                Gf.Vec3f(*[float(value) for value in maximum]),
            ]
        )
    )
    stage.GetRootLayer().Save()
    return path


def _concave_l_prism():
    import numpy as np
    import trimesh

    profile_vertices = np.asarray(
        [[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0], [1.0, 2.0], [0.0, 2.0]]
    )
    profile_faces = np.asarray([[0, 1, 3], [1, 2, 3], [0, 3, 5], [3, 4, 5]])
    return trimesh.creation.extrude_triangulation(profile_vertices, profile_faces, height=0.3)


def test_needle_triangles_only_block_profiles_that_use_render_mesh_elements() -> None:
    metrics = GeometryMetrics(
        source_format="usd",
        mesh=MeshMetrics(
            mesh_count=1,
            needle_triangle_count=3,
            triangle_aspect_ratio_p95=54.0,
            triangle_aspect_ratio_max=14_000.0,
        ),
    )

    rigid_issue = next(
        issue
        for issue in _mesh_issues(metrics, "rigid_pick_place")
        if issue.issue_id == "mesh:needle_triangles"
    )
    deformable_issue = next(
        issue
        for issue in _mesh_issues(metrics, "deformable_or_cae")
        if issue.issue_id == "mesh:needle_triangles"
    )

    assert "rigid_pick_place" not in rigid_issue.blocking_profiles
    assert "articulated_rigid" not in rigid_issue.blocking_profiles
    assert "contact_rich" not in rigid_issue.blocking_profiles
    assert deformable_issue.blocking_profiles == ["deformable_or_cae"]


def test_open_planar_mesh_requires_sheet_intent_for_rigid_profiles(tmp_path: Path) -> None:
    source = _write_trimesh(
        tmp_path / "planar_sheet.usda",
        trimesh.Trimesh(
            vertices=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            faces=[[0, 1, 2]],
            process=False,
        ),
    )

    rigid = diagnose_asset(source, profile="rigid_pick_place", run_scalable_audit=False)
    visual = diagnose_asset(source, profile="visual_only", run_scalable_audit=False)

    assert rigid.metrics.mesh.geometric_dimension == 2
    assert rigid.metrics.mesh.zero_thickness_status == "not_evaluated"
    assert "mesh:sheet_intent_unconfirmed" in rigid.blocking_issue_ids
    assert "mesh:sheet_intent_unconfirmed" not in visual.blocking_issue_ids


def test_collapsed_closed_mesh_is_a_zero_thickness_simulation_defect(tmp_path: Path) -> None:
    collapsed_box = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    collapsed_box.vertices[:, 2] = 0.0
    source = _write_trimesh(tmp_path / "collapsed_box.usda", collapsed_box)

    diagnosis = diagnose_asset(source, profile="static_environment", run_scalable_audit=False)

    assert diagnosis.metrics.mesh.geometric_dimension == 2
    assert diagnosis.metrics.mesh.zero_thickness_status == "fail"
    assert "mesh:zero_thickness_geometry" in diagnosis.blocking_issue_ids


def test_fake_runtime_evidence_is_test_only_and_cannot_certify(tmp_path: Path) -> None:
    source = _write_cube(tmp_path / "cube.usda")
    source_digest = file_sha256(source)

    result = run_geometry_repair(
        RepairRequest(
            source_path=source,
            output_dir=tmp_path / "repair",
            profile="rigid_pick_place",
            mode="auto",
            collision_runtime_engine="fake",
        )
    )

    assert result.outcome == "conditional"
    assert result.attempts[0].status == "accepted"
    assert result.attempts[0].output_sha256 == file_sha256(result.attempts[0].output_path or "")
    assert Path(result.render_usd_path or "").is_file()
    assert Path(result.collision_usd_path or "").is_file()
    assert Path(result.composed_usd_path or "").is_file()
    assert file_sha256(source) == source_digest
    certificate = json.loads(Path(result.certificate_path).read_text())
    assert certificate["claim_scope"] == "geometry_repair.rigid_pick_place"
    assert certificate["deterministic_seed"] == 0
    assert certificate["validation_results"]["collision"] == "conditional"
    collision = json.loads(
        (Path(result.certificate_path).parent / "collision_report.json").read_text()
    )
    assert collision["runtime_status"] == "pass"
    assert collision["runtime_engine"] == "fake"
    assert collision["generator"] == "geometry_repair.primitive_fit"
    assert collision["representation"] == "primitive_fit"
    assert collision["primitive_fit_kinds"] == ["box"]
    assert len(collision["runtime_scenarios"]) == 4
    assert all(
        scenario["metrics"]["exact_collision_vertex_count"] == 8
        for scenario in collision["runtime_scenarios"]
    )
    assert collision["generator_version"] != ""
    assert collision["collision_sha256"] == file_sha256(collision["collision_path"])

    from pxr import Usd, UsdPhysics

    composed = Usd.Stage.Open(result.composed_usd_path or "")
    assert composed is not None
    assert composed.GetDefaultPrim().GetPath() == "/GeometryRepairAsset"
    assert composed.GetPrimAtPath("/GeometryRepairAsset/Render/Mesh")
    collision_prims = [prim for prim in composed.Traverse() if prim.HasAPI(UsdPhysics.CollisionAPI)]
    assert collision_prims
    enabled_collision_prims = [
        prim
        for prim in collision_prims
        if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False
    ]
    assert enabled_collision_prims
    assert all(
        str(prim.GetPath()).startswith("/GeometryRepairAsset/Collision/")
        for prim in enabled_collision_prims
    )

    consumer = Usd.Stage.CreateNew(str(tmp_path / "consumer.usda"))
    referenced = consumer.DefinePrim("/World/Object", "Xform")
    referenced.GetReferences().AddReference(result.composed_usd_path or "")
    consumer.GetRootLayer().Save()
    consumer = Usd.Stage.Open(str(tmp_path / "consumer.usda"))
    assert consumer is not None
    assert consumer.GetPrimAtPath("/World/Object/Render/Mesh")
    referenced_collision_prims = [
        prim
        for prim in consumer.Traverse()
        if prim.HasAPI(UsdPhysics.CollisionAPI)
        and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False
    ]
    assert referenced_collision_prims
    assert all(
        str(prim.GetPath()).startswith("/World/Object/Collision/")
        for prim in referenced_collision_prims
    )


def test_concave_rigid_collision_uses_bounded_coacd(tmp_path: Path) -> None:
    source = _write_trimesh(
        tmp_path / "concave_l_prism.usda",
        _concave_l_prism(),
    )
    result = run_geometry_repair(
        RepairRequest(
            source_path=source,
            output_dir=tmp_path / "repair",
            profile="rigid_pick_place",
            mode="auto",
            budgets=RepairBudgets(moderate_p99_ratio=0.05, max_collision_hulls=16),
            protected_features=[
                ProtectedFeature(
                    name="mating notch",
                    kind="mating_surface",
                    scope_path="/Asset/Mesh",
                    minimum_clearance_m=0.3,
                    tolerance_m=0.1,
                )
            ],
            collision_runtime_engine="fake",
        )
    )

    assert result.outcome == "conditional"
    collision = json.loads(
        (Path(result.certificate_path).parent / "collision_report.json").read_text()
    )
    assert collision["representation"] == "coacd"
    assert collision["generator"] == "coacd_collision"
    assert 1 < collision["hull_count"] <= 16
    assert collision["maximum_vertices_per_hull"] <= 255
    assert collision["maximum_faces_per_hull"] <= 255
    assert collision["total_vertices"] >= collision["maximum_vertices_per_hull"]
    assert collision["total_faces"] >= collision["maximum_faces_per_hull"]
    assert collision["collision_size_bytes"] > 0
    assert collision["source_render_sha256"] == file_sha256(collision["source_render_path"])
    assert collision["maximum_volume_deficit_ratio"] <= 0.01
    assert len(collision["decompositions"]) == 1
    decomposition = collision["decompositions"][0]
    assert decomposition["worker_version"] == "1.0.11"
    assert len(decomposition["input_sha256"]) == 64
    assert decomposition["requested_threshold_m"] == pytest.approx(0.1)
    assert decomposition["report_sha256"] == file_sha256(decomposition["report_path"])
    assert decomposition["stdout_sha256"] == file_sha256(decomposition["stdout_path"])
    assert decomposition["stderr_sha256"] == file_sha256(decomposition["stderr_path"])
    assert Path(collision["decomposition_report_paths"][0]).is_file()
    assert collision["protected_features_applied"] == ["mating notch"]
    assert collision["protected_features_unmeasured"] == ["mating notch"]
    assert collision["protected_feature_probes"][0]["status"] == "not_evaluated"
    assert any(
        "explicit human review is required and this handoff cannot be certified" in warning
        for warning in collision["warnings"]
    )
    assert not any("could not replace" in warning for warning in collision["warnings"])


def test_compound_collision_surface_error_ignores_internal_partition_faces() -> None:
    source = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    left = trimesh.creation.box(extents=(1.0, 2.0, 2.0))
    left.apply_translation((-0.5, 0.0, 0.0))
    right = trimesh.creation.box(extents=(1.0, 2.0, 2.0))
    right.apply_translation((0.5, 0.0, 0.0))

    metrics = _approximation_metrics(source, [left, right])

    assert metrics["false_positive_ratio"] == pytest.approx(0.0)
    assert metrics["false_negative_ratio"] == pytest.approx(0.0)
    assert metrics["surface_gap_m"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["surface_overreach_m"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["occupancy_grid_resolution"] == 24


def test_compound_collision_surface_error_ignores_subgrid_partition_gap() -> None:
    source = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    left = trimesh.creation.box(extents=(0.995, 2.0, 2.0))
    left.apply_translation((-0.5025, 0.0, 0.0))
    right = trimesh.creation.box(extents=(0.995, 2.0, 2.0))
    right.apply_translation((0.5025, 0.0, 0.0))

    metrics = _approximation_metrics(source, [left, right])

    assert metrics["surface_overreach_m"] == pytest.approx(0.0, abs=1e-9)


def test_compound_collision_surface_error_retains_exposed_cavity_fill() -> None:
    source = trimesh.creation.annulus(r_min=0.5, r_max=1.0, height=0.3, sections=32)

    metrics = _approximation_metrics(source, [source.convex_hull])

    assert metrics["false_positive_ratio"] > 0.1
    assert metrics["surface_overreach_m"] > 0.1


def test_source_collision_count_is_not_limited_by_generated_hull_budget() -> None:
    import numpy as np
    import trimesh

    source = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    proxy = SimpleNamespace(
        world_vertices_m=np.asarray(source.vertices, dtype=np.float64),
        triangles=np.asarray(source.faces, dtype=np.int64),
    )
    budgets = RepairBudgets(max_collision_hulls=16, max_source_collision_prims=64)

    result = _validated_source_collision_candidate(
        "/Asset/Mesh",
        source,
        [proxy] * 34,
        budgets,
    )

    assert result is not None
    payloads, metrics = result
    assert len(payloads) == 34
    assert metrics["false_positive_ratio"] == pytest.approx(0.0)
    assert metrics["false_negative_ratio"] == pytest.approx(0.0)


def test_source_collision_measurement_failure_is_retained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np
    import trimesh

    source = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    proxy = SimpleNamespace(
        world_vertices_m=np.asarray(source.vertices, dtype=np.float64),
        triangles=np.asarray(source.faces, dtype=np.int64),
    )
    reasons: list[str] = []

    def fail_measurement(*_args, **_kwargs):
        raise RuntimeError("distance backend unavailable")

    monkeypatch.setattr("geometry_repair.collision._approximation_metrics", fail_measurement)
    result = _validated_source_collision_candidate(
        "/Asset/Mesh",
        source,
        [proxy],
        RepairBudgets(),
        rejection_reasons=reasons,
    )

    assert result is None
    assert reasons == [
        "source collider approximation could not be measured: "
        "RuntimeError: distance backend unavailable"
    ]


def test_collision_working_copy_fallback_retains_failure_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import trimesh

    source = trimesh.creation.icosphere(subdivisions=5, radius=1.0)

    def fail_simplification(*_args, **_kwargs):
        raise RuntimeError("simplifier unavailable")

    monkeypatch.setattr(
        trimesh.Trimesh,
        "simplify_quadric_decimation",
        fail_simplification,
    )
    candidate, evidence = _bounded_decomposition_input(source, RepairBudgets())

    assert candidate is source
    assert evidence is not None
    assert "retained the original mesh" in evidence
    assert "RuntimeError: simplifier unavailable" in evidence


def test_over_budget_convex_hull_gets_bounded_candidate() -> None:
    import trimesh

    hull = trimesh.creation.icosphere(subdivisions=4).convex_hull
    budgets = RepairBudgets()

    reduced, evidence = _bounded_convex_hull_reduction(hull, budgets)

    assert reduced is not None
    assert evidence is not None
    assert len(reduced.vertices) <= budgets.max_collision_vertices_per_hull
    assert len(reduced.faces) <= budgets.max_collision_faces_per_hull
    assert reduced.is_watertight
    assert reduced.is_winding_consistent


def test_dense_convex_sphere_uses_bounded_primitive_instead_of_coacd(tmp_path: Path) -> None:
    import numpy as np
    import trimesh

    sphere = trimesh.creation.icosphere(subdivisions=3, radius=0.05)
    vertices, faces = np.asarray(sphere.vertices), np.asarray(sphere.faces)
    for _ in range(2):
        vertices, faces = trimesh.remesh.subdivide(vertices, faces)
    sphere = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    assert len(sphere.faces) > 10_000
    source = _write_trimesh(
        tmp_path / "sphere.usda",
        sphere,
    )
    collision = build_collision_geometry(
        source,
        tmp_path / "collision.usda",
        profile="rigid_pick_place",
        budgets=RepairBudgets(sample_point_limit=256),
        runtime_engine="fake",
        report_path=tmp_path / "collision_report.json",
    )

    assert collision.status == "conditional"
    assert collision.runtime_status == "pass"
    assert collision.representation == "primitive_fit"
    assert collision.primitive_fit_kinds == ["sphere"]
    assert collision.maximum_vertices_per_hull <= 255
    assert collision.maximum_faces_per_hull <= 255
    assert (collision.maximum_volume_deficit_ratio or 0.0) <= 0.01
    assert (collision.maximum_false_negative_ratio or 0.0) <= 0.01
    assert collision.decompositions == []
    assert not any("exact-volume pre-screen" in warning for warning in collision.warnings)


def test_single_threaded_coacd_seed_is_byte_deterministic(tmp_path: Path) -> None:
    import numpy as np

    source = _concave_l_prism()
    runs = []
    for index in range(2):
        payloads, evidence, _result_path, error = _run_coacd(
            source_path="/Asset/Mesh",
            vertices=np.asarray(source.vertices),
            triangles=np.asarray(source.faces),
            work_dir=tmp_path / f"run-{index}",
            threshold_m=0.1,
            max_hulls=4,
            max_vertices=255,
            max_faces=255,
            seed=17,
            memory_mb=1024,
            # Harness bound only — the assertion under test is byte
            # determinism across the two seeded single-threaded runs, not
            # speed. Coverage tracing plus a loaded shared runner in the
            # geometry-repair shard blew a 10s budget on a real decompose;
            # keep the ceiling generous so it only catches a genuine hang.
            timeout_s=120.0,
        )
        assert error is None
        assert evidence is not None
        assert evidence.thread_limit == 1
        runs.append(payloads)

    assert len(runs[0]) == len(runs[1]) == 2
    for left, right in zip(runs[0], runs[1], strict=True):
        assert np.array_equal(left[1], right[1])
        assert np.array_equal(left[2], right[2])


def test_coacd_allow_list_disablement_is_reported(tmp_path: Path) -> None:
    import trimesh

    source = _write_trimesh(
        tmp_path / "annulus.usda",
        trimesh.creation.annulus(r_min=0.5, r_max=1.0, height=0.3, sections=12),
    )
    result = run_geometry_repair(
        RepairRequest(
            source_path=source,
            output_dir=tmp_path / "repair",
            profile="rigid_pick_place",
            mode="auto",
            enabled_workers=[
                "ocp_shape_heal",
                "trimesh_conservative_cleanup",
                "usd_structure_repair",
            ],
            collision_runtime_engine="fake",
        )
    )

    collision = json.loads(
        (Path(result.certificate_path).parent / "collision_report.json").read_text()
    )
    assert collision["status"] == "fail"
    assert collision["representation"] == "none"
    assert collision["collision_path"] is None
    assert collision["decompositions"] == []
    assert any(
        "disabled by the request worker allow-list" in item for item in collision["warnings"]
    )


def test_coacd_worker_cannot_return_path_outside_assigned_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np

    escaped = tmp_path / "escaped.npz"
    np.savez_compressed(
        escaped,
        vertices=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        faces=np.asarray([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]]),
    )

    def fake_run(command, **_kwargs):
        result_path = Path(command[command.index("--result") + 1])
        result_path.write_text(
            json.dumps(
                {
                    "schema_version": "geometry-repair.coacd-result.v1",
                    "status": "pass",
                    "coacd_version": "1.0.11",
                    "elapsed_s": 0.0,
                    "parts": [
                        {
                            "path": str(escaped),
                            "sha256": file_sha256(escaped),
                            "vertex_count": 4,
                            "triangle_count": 4,
                        }
                    ],
                    "failures": [],
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("geometry_repair.collision.subprocess.run", fake_run)
    payloads, evidence, result_path, error = _run_coacd(
        source_path="/Asset/Mesh",
        vertices=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        triangles=np.asarray([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]]),
        work_dir=tmp_path / "worker",
        threshold_m=0.01,
        max_hulls=4,
        max_vertices=255,
        max_faces=255,
        seed=0,
        memory_mb=1024,
        timeout_s=10.0,
    )

    assert payloads == []
    assert evidence is None
    assert result_path.is_file()
    assert "escaped its output directory" in (error or "")


def test_coacd_timeout_emits_failure_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np

    def fake_timeout(command, **_kwargs):
        raise subprocess.TimeoutExpired(command, timeout=0.01)

    monkeypatch.setattr("geometry_repair.collision.subprocess.run", fake_timeout)
    payloads, evidence, result_path, error = _run_coacd(
        source_path="/Asset/Mesh",
        vertices=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        triangles=np.asarray([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]]),
        work_dir=tmp_path / "worker",
        threshold_m=0.01,
        max_hulls=4,
        max_vertices=255,
        max_faces=255,
        seed=0,
        memory_mb=1024,
        timeout_s=0.01,
    )

    assert payloads == []
    assert evidence is None
    assert "exceeded wall-clock budget" in (error or "")
    report = json.loads(result_path.read_text())
    assert report["status"] == "fail"


def test_worker_timeout_cannot_reuse_stale_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_cube(tmp_path / "source.usda")
    attempt_dir = tmp_path / "attempt"
    attempt_dir.mkdir()
    output = attempt_dir / "geometry.usda"
    shutil.copy2(source, output)
    stale = attempt_dir / "worker_result.json"
    stale.write_text(
        json.dumps(
            {
                "status": "completed",
                "output_path": str(output),
                "output_sha256": file_sha256(output),
                "changed": True,
            }
        ),
        encoding="utf-8",
    )

    class TimedOutProcess:
        pid = 424242
        returncode: int | None = None

        def communicate(self, timeout=None):
            if timeout is not None:
                raise subprocess.TimeoutExpired("worker", timeout=timeout)
            self.returncode = -9
            return "", ""

    monkeypatch.setattr(
        "geometry_repair.orchestrator.subprocess.Popen",
        lambda *_args, **_kwargs: TimedOutProcess(),
    )
    monkeypatch.setattr(
        "geometry_repair.orchestrator.os.killpg",
        lambda *_args: (_ for _ in ()).throw(ProcessLookupError()),
    )
    result = _run_worker_subprocess(
        operation=RepairOperation(
            operation_id="timeout",
            worker="trimesh_conservative_cleanup",
            implementation="test",
            drift_band="conservative",
            source_checkpoint=str(source),
        ),
        source=source,
        output=output,
        attempt_dir=attempt_dir,
        memory_mb=128,
        timeout_s=0.01,
    )

    assert result.status == "failed"
    assert "exceeded remaining wall-clock budget" in result.failures[0]
    assert not stale.exists()
    assert not output.exists()


def test_openvdb_worker_launch_bounds_thread_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    attempt_dir = tmp_path / "attempt"
    attempt_dir.mkdir()
    captured: dict[str, object] = {}

    class MissingResultProcess:
        pid = 424242
        returncode = 2

        def communicate(self, timeout=None):
            return "", ""

    def fake_popen(*_args, **kwargs):
        captured.update(kwargs)
        return MissingResultProcess()

    monkeypatch.setattr("geometry_repair.orchestrator.subprocess.Popen", fake_popen)
    result = _run_worker_subprocess(
        operation=RepairOperation(
            operation_id="bounded-openvdb",
            worker="sdf_rebuild",
            implementation="test",
            parameters={"deterministic_seed": 7},
            drift_band="reconstructive",
            source_checkpoint=str(source),
        ),
        source=source,
        output=tmp_path / "output.usda",
        attempt_dir=attempt_dir,
        memory_mb=8192,
        timeout_s=1.0,
    )

    assert result.status == "failed"
    environment = captured["env"]
    assert isinstance(environment, dict)
    assert environment["MALLOC_ARENA_MAX"] == "2"
    assert environment["MKL_NUM_THREADS"] == "1"
    assert environment["OMP_NUM_THREADS"] == "1"
    assert environment["OPENBLAS_NUM_THREADS"] == "1"
    assert environment["PXR_WORK_THREAD_LIMIT"] == "1"
    assert environment["TBB_NUM_THREADS"] == "1"
    assert environment["PYTHONHASHSEED"] == "7"


def test_scene_optimizer_dependency_roots_are_pinned_from_provenance(
    tmp_path: Path,
) -> None:
    upload = tmp_path / "upload"
    checkpoint_dir = tmp_path / "job" / "normalized"
    requested_root = tmp_path / "approved-external"
    localized_dir = tmp_path / "job" / "dependency_bundle"
    snapshot_dir = tmp_path / "job" / "source"
    for directory in (
        upload,
        checkpoint_dir,
        requested_root,
        localized_dir,
        snapshot_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    operation = RepairOperation(
        operation_id="deinstance",
        worker="scene_optimizer_deinstance",
        implementation="test",
        parameters={"approved_dependency_roots": ["/"]},
        drift_band="identity",
        source_checkpoint=str(checkpoint_dir / "source.usd"),
    )
    pinned = _pin_scene_optimizer_dependency_roots(
        operation,
        checkpoint_path=checkpoint_dir / "source.usd",
        source_path=upload / "source.usda",
        requested_dependency_roots=[requested_root],
        localized_source_path=localized_dir / "source.usda",
        resolved_snapshot_path=snapshot_dir / "resolved_snapshot.usdc",
    )

    assert pinned.parameters["approved_dependency_roots"] == [
        str(checkpoint_dir.resolve()),
        str(upload.resolve()),
        str(requested_root.resolve()),
        str(localized_dir.resolve()),
        str(snapshot_dir.resolve()),
    ]
    assert "/" not in pinned.parameters["approved_dependency_roots"]


def test_scene_optimizer_dependency_roots_omit_stale_unsafe_and_duplicate_roots(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    checkpoint_dir = tmp_path / "job" / "normalized"
    checkpoint_dir.mkdir(parents=True)
    requested_root = tmp_path / "approved-external"
    requested_root.mkdir()
    missing_requested = tmp_path / "removed-requested-root"
    missing_upload = tmp_path / "removed-upload"
    missing_localized = tmp_path / "removed-bundle"
    missing_snapshot = tmp_path / "removed-snapshot"

    operation = RepairOperation(
        operation_id="deinstance",
        worker="scene_optimizer_deinstance",
        implementation="test",
        drift_band="identity",
        source_checkpoint=str(checkpoint_dir / "source.usd"),
    )
    pinned = _pin_scene_optimizer_dependency_roots(
        operation,
        checkpoint_path=checkpoint_dir / "source.usd",
        source_path=missing_upload / "source.usda",
        requested_dependency_roots=[
            checkpoint_dir / ".." / "normalized",
            requested_root,
            missing_requested,
            Path(tmp_path.anchor),
        ],
        localized_source_path=missing_localized / "source.usda",
        resolved_snapshot_path=missing_snapshot / "resolved_snapshot.usdc",
    )

    assert pinned.parameters["approved_dependency_roots"] == [
        str(checkpoint_dir.resolve()),
        str(requested_root.resolve()),
    ]
    assert caplog.messages == [
        "Requested Scene Optimizer dependency root will not be forwarded because "
        f"it is not an existing directory: {missing_requested.resolve()}"
    ]


def test_scene_optimizer_worker_forwards_pinned_dependency_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import world_understanding.functions.graphics.scene_optimizer_local as so_local

    source = tmp_path / "source.usda"
    source.touch()
    output = tmp_path / "output.usda"
    approved_root = tmp_path / "approved"
    approved_root.mkdir()
    captured: dict[str, object] = {}

    def fake_optimize(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        output.touch()
        return {"status": "success"}

    monkeypatch.setattr(so_local, "optimize_usd_local", fake_optimize)
    result = SceneOptimizerDeinstanceWorker().execute(
        source=source,
        output=output,
        operation=RepairOperation(
            operation_id="deinstance",
            worker="scene_optimizer_deinstance",
            implementation="test",
            parameters={"approved_dependency_roots": [str(approved_root)]},
            drift_band="identity",
            source_checkpoint=str(source),
        ),
    )

    assert result.status == "completed"
    assert captured["approved_dependency_roots"] == [str(approved_root)]


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group semantics")
def test_worker_timeout_kills_native_descendants(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_cube(tmp_path / "source.usda")
    attempt_dir = tmp_path / "attempt"
    attempt_dir.mkdir()
    output = attempt_dir / "geometry.usda"
    marker = tmp_path / "descendant_survived"
    descendant = (
        f"import pathlib,time; time.sleep(0.4); pathlib.Path({str(marker)!r}).write_text('alive')"
    )
    parent = (
        "import subprocess,sys,time; "
        f"subprocess.Popen([sys.executable, '-c', {descendant!r}]); "
        "time.sleep(30)"
    )
    real_popen = subprocess.Popen

    def spawn_test_tree(_command, **kwargs):
        return real_popen([sys.executable, "-c", parent], **kwargs)

    monkeypatch.setattr(
        "geometry_repair.orchestrator.subprocess.Popen",
        spawn_test_tree,
    )
    result = _run_worker_subprocess(
        operation=RepairOperation(
            operation_id="timeout-tree",
            worker="trimesh_conservative_cleanup",
            implementation="test",
            drift_band="conservative",
            source_checkpoint=str(source),
        ),
        source=source,
        output=output,
        attempt_dir=attempt_dir,
        memory_mb=128,
        timeout_s=0.1,
    )

    time.sleep(0.5)
    assert result.status == "failed"
    assert not marker.exists()


def test_worker_group_kill_uses_direct_process_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProcess:
        pid = 42
        killed = False
        waited = False

        def kill(self) -> None:
            self.killed = True

        def communicate(self):
            self.waited = True
            return "", ""

    process = FakeProcess()
    monkeypatch.setattr("geometry_repair.orchestrator.os.name", "nt")

    _kill_worker_process_group(process)

    assert process.killed is True
    assert process.waited is True


def test_external_scene_instances_retain_node_transforms(tmp_path: Path) -> None:
    import trimesh

    source = tmp_path / "instances.glb"
    scene = trimesh.Scene()
    cube = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    scene.add_geometry(
        cube,
        geom_name="left_geometry",
        node_name="left_cube",
        transform=trimesh.transformations.translation_matrix([-2.0, 0.0, 0.0]),
    )
    scene.add_geometry(
        cube,
        geom_name="right_geometry",
        node_name="right_cube",
        transform=trimesh.transformations.translation_matrix([2.0, 0.0, 0.0]),
    )
    scene.export(source)

    diagnosis = diagnose_asset(source, profile="visual_only")

    assert diagnosis.metrics.mesh.mesh_count == 2
    assert len(diagnosis.metrics.meshes) == 2
    centers = []
    for record in diagnosis.metrics.meshes:
        assert record.metrics.bbox_min_m is not None
        assert record.metrics.bbox_max_m is not None
        centers.append((record.metrics.bbox_min_m[0] + record.metrics.bbox_max_m[0]) / 2.0)
    assert sorted(centers) == pytest.approx([-2.0, 2.0])


def test_vertex_touching_closed_shells_are_not_misclassified_as_watertight(
    tmp_path: Path,
) -> None:
    import numpy as np
    import trimesh

    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ]
    )
    faces = np.asarray(
        [
            [0, 2, 1],
            [0, 1, 3],
            [1, 2, 3],
            [2, 0, 3],
            [4, 5, 6],
            [4, 7, 5],
            [5, 7, 6],
            [6, 7, 4],
        ]
    )
    source = _write_trimesh(
        tmp_path / "vertex_touch.usda",
        trimesh.Trimesh(vertices=vertices, faces=faces, process=False),
    )

    diagnosis = diagnose_asset(source, profile="visual_only")

    assert diagnosis.metrics.mesh.non_manifold_vertex_count == 1
    assert diagnosis.metrics.mesh.watertight is False
    assert "mesh:non_manifold_vertices" in diagnosis.blocking_issue_ids


def test_authoritative_runtime_evidence_can_certify(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_cube(tmp_path / "cube.usda")

    def _runtime_stub(collision_path, output_dir, *, engine):
        assert Path(collision_path).is_file()
        assert engine == "ovphysx"
        runtime_dir = Path(output_dir)
        runtime_dir.mkdir(parents=True)
        proxy = runtime_dir / "temporary_proxy.usda"
        report = runtime_dir / "runtime_report.json"
        proxy.write_text("#usda 1.0\n", encoding="utf-8")
        report.write_text("{}\n", encoding="utf-8")
        return {
            "status": "pass",
            "engine": engine,
            "temporary_proxy_path": str(proxy),
            "runtime_report_path": str(report),
            "failures": [],
            "warnings": [],
        }

    monkeypatch.setattr(
        "geometry_repair.collision.validate_collision_runtime",
        _runtime_stub,
    )

    def _in_process_collision_build(**kwargs):
        kwargs.pop("attempt_dir")
        render_path = kwargs.pop("render_path")
        output_path = kwargs.pop("output_path")
        return build_collision_geometry(render_path, output_path, **kwargs)

    monkeypatch.setattr(
        "geometry_repair.orchestrator._run_collision_geometry_subprocess",
        _in_process_collision_build,
    )
    result = run_geometry_repair(
        RepairRequest(
            source_path=source,
            output_dir=tmp_path / "repair",
            profile="rigid_pick_place",
            mode="auto",
            collision_runtime_engine="ovphysx",
        )
    )

    assert result.outcome == "certified"
    collision = json.loads(
        (Path(result.certificate_path).parent / "collision_report.json").read_text()
    )
    assert collision["status"] == "pass"
    assert collision["runtime_engine"] == "ovphysx"


def test_clean_rigid_asset_without_runtime_evidence_is_conditional(
    tmp_path: Path,
) -> None:
    source = _write_cube(tmp_path / "cube.usda")

    result = run_geometry_repair(
        RepairRequest(
            source_path=source,
            output_dir=tmp_path / "repair",
            profile="rigid_pick_place",
            mode="auto",
        )
    )

    assert result.outcome == "conditional"
    collision = json.loads(
        (Path(result.certificate_path).parent / "collision_report.json").read_text()
    )
    assert collision["runtime_status"] == "not_evaluated"


def test_collision_source_vertex_budget_fails_closed(tmp_path: Path) -> None:
    source = _write_cube(tmp_path / "cube.usda")

    result = run_geometry_repair(
        RepairRequest(
            source_path=source,
            output_dir=tmp_path / "repair",
            profile="rigid_pick_place",
            mode="auto",
            budgets=RepairBudgets(max_collision_source_vertices_per_body=4),
        )
    )

    assert result.outcome == "rejected"
    certificate = json.loads(Path(result.certificate_path).read_text())
    assert any("above collision planning limit" in item for item in certificate["blockers"])


def test_over_complex_hull_cannot_fall_back_when_coacd_is_disabled(
    tmp_path: Path,
) -> None:
    source = _write_cube(tmp_path / "cube.usda")

    result = run_geometry_repair(
        RepairRequest(
            source_path=source,
            output_dir=tmp_path / "repair",
            profile="rigid_pick_place",
            mode="auto",
            budgets=RepairBudgets(max_collision_faces_per_hull=4),
            enabled_workers=[
                "ocp_shape_heal",
                "trimesh_conservative_cleanup",
                "usd_structure_repair",
            ],
            collision_runtime_engine="fake",
        )
    )

    assert result.outcome == "rejected"
    certificate = json.loads(Path(result.certificate_path).read_text())
    assert any("CoACD is required" in item for item in certificate["blockers"])


def test_static_collision_runs_static_contact_canary(
    tmp_path: Path,
) -> None:
    source = _write_cube(tmp_path / "cube.usda")

    result = run_geometry_repair(
        RepairRequest(
            source_path=source,
            output_dir=tmp_path / "repair",
            profile="static_environment",
            mode="auto",
            collision_runtime_engine="fake",
        )
    )

    assert result.outcome == "conditional"
    collision = json.loads(
        (Path(result.certificate_path).parent / "collision_report.json").read_text()
    )
    assert collision["representation"] == "static_triangle_mesh"
    assert collision["runtime_status"] == "pass"
    assert collision["runtime_scenarios"][0]["scenario_kind"] == "static_contact"
    assert any("test-only" in item for item in collision["warnings"])


def test_duplicate_and_degenerate_faces_are_repaired_conservatively(
    tmp_path: Path,
) -> None:
    source = _write_cube(
        tmp_path / "corrupt.usda",
        duplicate_face=True,
        degenerate_face=True,
    )

    result = run_geometry_repair(
        RepairRequest(
            source_path=source,
            output_dir=tmp_path / "repair",
            profile="rigid_pick_place",
            mode="auto",
            collision_runtime_engine="fake",
        )
    )

    assert result.outcome == "conditional"
    assert [attempt.status for attempt in result.attempts] == ["rejected", "accepted"]
    repaired = diagnose_asset(result.render_usd_path or "", profile="rigid_pick_place")
    assert repaired.status == "pass"
    assert repaired.metrics.mesh.duplicate_face_count == 0
    assert repaired.metrics.mesh.degenerate_face_count == 0
    certificate = json.loads(Path(result.certificate_path).read_text())
    assert certificate["render_geometry_changed"] is True
    assert certificate["visual_review_status"] == "required"
    assert certificate["validation_results"]["visual_review"] == "required"
    assert certificate["deleted_entities"]
    assert "remove_invalid_faces:/Asset/Mesh:2" in certificate["operations_applied"]
    worker_report = json.loads(Path(result.attempts[-1].worker_report_path or "").read_text())
    assert worker_report["metadata"]["execution_scope"] == "isolated_subprocess"
    assert worker_report["metadata"]["memory_limit_mb"] == 8192
    assert worker_report["metadata"]["implementation_version"] != "unknown"
    assert worker_report["output_sha256"] == file_sha256(result.attempts[-1].output_path or "")


def test_topology_and_stage_metric_repairs_advance_as_a_bounded_sequence(
    tmp_path: Path,
) -> None:
    from pxr import Usd, UsdGeom

    source = _write_cube(
        tmp_path / "corrupt_centimeter_y_up.usda",
        duplicate_face=True,
        degenerate_face=True,
    )
    stage = Usd.Stage.Open(str(source))
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
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

    assert result.outcome == "conditional"
    assert [attempt.status for attempt in result.attempts] == [
        "rejected",
        "advanced",
        "accepted",
    ]
    assert [attempt.operation.worker for attempt in result.attempts[1:]] == [
        "trimesh_conservative_cleanup",
        "usd_structure_repair",
    ]
    repaired = diagnose_asset(result.render_usd_path or "", profile="rigid_pick_place")
    assert repaired.status == "pass"
    assert repaired.metrics.mesh.duplicate_face_count == 0
    assert repaired.metrics.mesh.degenerate_face_count == 0
    assert repaired.metrics.meters_per_unit == pytest.approx(1.0)
    assert repaired.metrics.up_axis == "Z"
    certificate = json.loads(Path(result.certificate_path).read_text())
    assert any(
        item.startswith("remove_invalid_faces:") for item in certificate["operations_applied"]
    )
    assert "normalize_up_axis:Y->Z" in certificate["operations_applied"]


def test_open_dynamic_mesh_is_rejected_without_guessing_hole_intent(
    tmp_path: Path,
) -> None:
    source = _write_cube(tmp_path / "open.usda", open_face=True)

    result = run_geometry_repair(
        RepairRequest(
            source_path=source,
            output_dir=tmp_path / "repair",
            profile="rigid_pick_place",
            mode="auto",
        )
    )

    assert result.outcome == "rejected"
    assert len(result.attempts) == 1
    certificate = json.loads(Path(result.certificate_path).read_text())
    assert "mesh:boundary_edges" in certificate["blockers"]
    assert not result.render_usd_path


def test_intersecting_closed_shells_are_detected_and_rejected(tmp_path: Path) -> None:
    import numpy as np
    import trimesh
    from pxr import Gf, Usd, UsdGeom, Vt

    first = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    second = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    second.apply_translation([0.25, 0.25, 0.25])
    vertices = np.vstack([first.vertices, second.vertices])
    faces = np.vstack([first.faces, second.faces + len(first.vertices)])
    source = tmp_path / "intersecting_shells.usda"
    stage = Usd.Stage.CreateNew(str(source))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/Asset/Mesh")
    mesh.CreatePointsAttr(
        Vt.Vec3fArray([Gf.Vec3f(*[float(value) for value in point]) for point in vertices])
    )
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(faces)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(faces.astype(int).reshape(-1).tolist()))
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    stage.GetRootLayer().Save()

    diagnosis = diagnose_asset(source, profile="rigid_pick_place")
    result = run_geometry_repair(
        RepairRequest(
            source_path=source,
            output_dir=tmp_path / "repair",
            profile="rigid_pick_place",
            mode="auto",
        )
    )

    assert diagnosis.metrics.mesh.self_intersection_status == "fail"
    assert diagnosis.metrics.mesh.self_intersection_count > 0
    assert "mesh:self_intersections" in diagnosis.blocking_issue_ids
    assert result.outcome == "rejected"
    attempt = result.attempts[-1]
    assert attempt.operation.worker == "geogram_local_repair"
    assert attempt.status in {"rejected", "unavailable"}
    worker_report = json.loads(Path(attempt.worker_report_path or "").read_text())
    if attempt.status == "unavailable":
        assert not attempt.output_path
        assert any("Geogram" in reason or "vorpalite" in reason for reason in attempt.reasons)
        assert worker_report["status"] == "unavailable"
    else:
        assert attempt.output_path
        assert worker_report["metadata"]["vorpalite_version"] == GeogramLocalRepairWorker.version
        assert worker_report["metadata"]["hole_filling_enabled"] is False
        assert worker_report["metadata"]["component_removal_enabled"] is False


@pytest.mark.parametrize(
    ("reported_version", "expected_available"),
    (("1.10.0", True), ("1.8.5", False), ("1.11.0", False)),
)
def test_geogram_worker_requires_exact_audited_version(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    reported_version: str,
    expected_available: bool,
) -> None:
    executable = tmp_path / "vorpalite"
    executable.write_bytes(b"audited geogram test executable")
    monkeypatch.setenv(
        GEOGRAM_EXECUTABLE_SHA256_ENVIRONMENT_VARIABLE,
        file_sha256(executable),
    )
    monkeypatch.delenv(
        f"{GEOGRAM_EXECUTABLE_SHA256_ENVIRONMENT_VARIABLE}_FILE",
        raising=False,
    )
    monkeypatch.setattr(shutil, "which", lambda _name: str(executable))
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=f"vorpalite {reported_version}",
            stderr="",
        ),
    )

    available, reason = GeogramLocalRepairWorker().available()

    assert available is expected_available
    assert (reason is None) is expected_available
    if not expected_available:
        assert "audited vorpalite 1.10.0" in str(reason)


def test_geogram_worker_rejects_unapproved_executable_digest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    executable = tmp_path / "vorpalite"
    executable.write_bytes(b"unapproved geogram test executable")
    monkeypatch.setenv(
        GEOGRAM_EXECUTABLE_SHA256_ENVIRONMENT_VARIABLE,
        "0" * 64,
    )
    monkeypatch.delenv(
        f"{GEOGRAM_EXECUTABLE_SHA256_ENVIRONMENT_VARIABLE}_FILE",
        raising=False,
    )
    monkeypatch.setattr(shutil, "which", lambda _name: str(executable))
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail(
            "an unapproved executable must be rejected before version execution"
        ),
    )

    available, reason = GeogramLocalRepairWorker().available()

    assert available is False
    assert "digest mismatch" in str(reason)


def test_geogram_worker_records_approved_executable_digest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from pxr import Usd

    executable = tmp_path / "vorpalite"
    executable.write_bytes(b"approved geogram test executable")
    executable_sha256 = file_sha256(executable)
    monkeypatch.setenv(
        GEOGRAM_EXECUTABLE_SHA256_ENVIRONMENT_VARIABLE,
        executable_sha256,
    )
    monkeypatch.delenv(
        f"{GEOGRAM_EXECUTABLE_SHA256_ENVIRONMENT_VARIABLE}_FILE",
        raising=False,
    )
    monkeypatch.setattr(shutil, "which", lambda _name: str(executable))
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout="vorpalite 1.10.0",
            stderr="",
        ),
    )
    source = tmp_path / "empty.usda"
    stage = Usd.Stage.CreateNew(str(source))
    root = stage.DefinePrim("/Asset", "Xform")
    stage.SetDefaultPrim(root)
    stage.GetRootLayer().Save()

    result = GeogramLocalRepairWorker().execute(
        source=source,
        output=tmp_path / "candidate.usda",
        operation=RepairOperation(
            operation_id="geogram-identity",
            worker="geogram_local_repair",
            implementation="test",
            drift_band="conservative",
            source_checkpoint=str(source),
        ),
    )

    assert result.status == "unavailable"
    assert result.metadata["vorpalite_sha256"] == executable_sha256
    assert result.metadata["vorpalite_version"] == "1.10.0"


def test_self_intersection_diagnostic_stops_at_deterministic_pair_budget() -> None:
    import numpy as np

    base = np.asarray(
        [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    vertices = np.concatenate([base * (1.0 + 0.01 * index) for index in range(5)], axis=0)
    triangles = np.arange(15, dtype=np.int64).reshape((-1, 3))

    status, intersections, broad_pairs, candidate_pairs, reason = _self_intersections(
        vertices,
        triangles,
        diagonal=2.0,
        broad_pair_limit=3,
    )

    assert status == "not_evaluated"
    assert intersections == 3
    assert broad_pairs == 4
    assert candidate_pairs == 3
    assert reason == "non-adjacent broad-phase pair count exceeds exact-check limit 3"


def test_duplicate_index_cad_seams_are_not_self_intersections(tmp_path: Path) -> None:
    import numpy as np
    import trimesh
    from pxr import Gf, Usd, UsdGeom, Vt

    cube = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    cube.unmerge_vertices()
    source = tmp_path / "cad_face_soup.usda"
    stage = Usd.Stage.CreateNew(str(source))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/Asset/Mesh")
    mesh.CreatePointsAttr(
        Vt.Vec3fArray([Gf.Vec3f(*[float(value) for value in point]) for point in cube.vertices])
    )
    faces = np.asarray(cube.faces, dtype=np.int64)
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(faces)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(faces.reshape(-1).tolist()))
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    stage.GetRootLayer().Save()

    diagnosis = diagnose_asset(source, profile="rigid_pick_place")
    result = run_geometry_repair(
        RepairRequest(
            source_path=source,
            output_dir=tmp_path / "repair",
            profile="rigid_pick_place",
            mode="auto",
            collision_runtime_engine="fake",
        )
    )

    assert diagnosis.metrics.mesh.indexed_boundary_edge_count > 0
    assert diagnosis.metrics.mesh.boundary_edge_count == 0
    assert diagnosis.metrics.mesh.self_intersection_status == "pass"
    assert diagnosis.metrics.mesh.watertight is True
    assert result.outcome == "conditional"
    assert result.render_usd_path


def test_visual_profile_allows_intentional_open_surface(tmp_path: Path) -> None:
    source = _write_cube(tmp_path / "open.usda", open_face=True)

    result = run_geometry_repair(
        RepairRequest(
            source_path=source,
            output_dir=tmp_path / "repair",
            profile="visual_only",
            mode="auto",
        )
    )

    assert result.outcome == "conditional"
    assert result.collision_usd_path is None
    certificate = json.loads(Path(result.certificate_path).read_text())
    assert certificate["validation_results"]["diagnosis"] == "conditional"
    assert any("Open boundary edges" in item for item in certificate["remaining_warnings"])


def test_usd_dependency_snapshot_survives_original_source_removal(tmp_path: Path) -> None:
    from pxr import Usd

    dependency = _write_cube(tmp_path / "dependency.usda")
    source = tmp_path / "root.usda"
    source.write_text(
        """#usda 1.0
(
    defaultPrim = "Asset"
    subLayers = [@dependency.usda@]
)
""",
        encoding="utf-8",
    )
    source_digest = file_sha256(source)

    result = run_geometry_repair(
        RepairRequest(
            source_path=source,
            output_dir=tmp_path / "repair",
            profile="visual_only",
            mode="auto",
        )
    )
    source_package = json.loads(Path(result.source_package_path).read_text())
    certificate = json.loads(Path(result.certificate_path).read_text())

    source.unlink()
    dependency.unlink()
    repaired_stage = Usd.Stage.Open(result.render_usd_path or "")
    assert repaired_stage is not None
    assert repaired_stage.GetPrimAtPath("/Asset/Mesh")
    assert Path(source_package["resolved_snapshot_path"]).is_file()
    assert source_package["resolved_snapshot_sha256"]
    assert certificate["source_sha256"] == source_digest
    assert certificate["normalized_source_sha256"] != ""


def test_missing_texture_is_geometry_warning_not_package_failure(tmp_path: Path) -> None:
    from pxr import Sdf, Usd, UsdShade

    source = _write_cube(tmp_path / "missing_texture.usda")
    stage = Usd.Stage.Open(str(source))
    shader = UsdShade.Shader.Define(stage, "/Asset/Looks/PreviewSurface")
    shader.CreateIdAttr("UsdUVTexture")
    shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("Textures/missing_base_color.png")
    )
    stage.GetRootLayer().Save()

    result = run_geometry_repair(
        RepairRequest(
            source_path=source,
            output_dir=tmp_path / "repair",
            profile="visual_only",
            mode="auto",
        )
    )

    package = json.loads(
        (Path(result.certificate_path).parent / "usd_package_validation.json").read_text()
    )
    assert result.outcome == "certified"
    assert package["status"] == "warning"
    assert package["failures"] == []
    assert all(record["status"] != "fail" for record in package["layers"])
    assert any("missing_base_color.png" in warning for warning in package["warnings"])
    certificate = json.loads(Path(result.certificate_path).read_text())
    assert certificate["validation_results"]["source_format"] == ("pass_with_material_warnings")
    assert certificate["validation_results"]["source_format_raw"] == "fail"
    assert any("missing_base_color.png" in warning for warning in certificate["remaining_warnings"])


def test_unknown_unresolved_dependency_fails_closed() -> None:
    geometry, material = classify_unresolved_dependencies(
        [
            "/asset/missing_payload.usd",
            "/asset/missing_texture.png",
            "dependency discovery failed: RuntimeError: bad layer",
        ]
    )

    assert material == ["/asset/missing_texture.png"]
    assert geometry == [
        "/asset/missing_payload.usd",
        "dependency discovery failed: RuntimeError: bad layer",
    ]


def test_inferred_profile_cannot_certify(tmp_path: Path) -> None:
    source = _write_cube(tmp_path / "cube.usda")

    result = run_geometry_repair(
        RepairRequest(
            source_path=source,
            output_dir=tmp_path / "repair",
            profile="rigid_pick_place",
            profile_confirmed=False,
            mode="auto",
            collision_runtime_engine="fake",
        )
    )

    assert result.outcome == "conditional"
    certificate = json.loads(Path(result.certificate_path).read_text())
    assert certificate["profile_confirmed"] is False
    assert any("profile was inferred" in item for item in certificate["remaining_warnings"])


def test_production_certification_requires_source_rights_evidence(tmp_path: Path) -> None:
    source = _write_cube(tmp_path / "cube.usda")

    missing = run_geometry_repair(
        RepairRequest(
            source_path=source,
            output_dir=tmp_path / "missing",
            profile="visual_only",
            production_use=True,
            mode="auto",
        )
    )
    partial = run_geometry_repair(
        RepairRequest(
            source_path=source,
            output_dir=tmp_path / "partial",
            profile="visual_only",
            production_use=True,
            source_license="Apache-2.0",
            source_provenance={"origin": "synthetic-test-fixture"},
            mode="auto",
        )
    )
    documented = run_geometry_repair(
        RepairRequest(
            source_path=source,
            output_dir=tmp_path / "documented",
            profile="visual_only",
            production_use=True,
            source_uri="test-fixture://geometry-repair/cube.usda",
            source_license="Apache-2.0",
            source_provenance={"origin": "synthetic-test-fixture"},
            mode="auto",
        )
    )

    missing_certificate = json.loads(Path(missing.certificate_path).read_text())
    partial_certificate = json.loads(Path(partial.certificate_path).read_text())
    documented_certificate = json.loads(Path(documented.certificate_path).read_text())
    assert missing.outcome == "conditional"
    assert missing_certificate["source_rights_status"] == "missing"
    assert partial.outcome == "conditional"
    assert partial_certificate["source_rights_status"] == "missing"
    assert documented_certificate["source_rights_status"] == "pass"
    assert documented.outcome == "certified"


def test_stage_metrics_are_normalized_with_geometry_transform(tmp_path: Path) -> None:
    from pxr import Usd, UsdGeom

    source = _write_cube(tmp_path / "centimeter_y_up.usda")
    stage = Usd.Stage.Open(str(source))
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
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

    assert result.outcome == "conditional"
    assert result.attempts[-1].operation.worker == "usd_structure_repair"
    repaired_stage = Usd.Stage.Open(result.render_usd_path or "")
    assert str(UsdGeom.GetStageUpAxis(repaired_stage)) == "Z"
    assert UsdGeom.GetStageMetersPerUnit(repaired_stage) == pytest.approx(1.0)
    fidelity = json.loads(Path(result.attempts[-1].fidelity_path or "").read_text())
    assert fidelity["status"] == "pass"
    assert fidelity["surface_distance_p99_m"] == pytest.approx(0.0, abs=1e-9)
    relocated = tmp_path / "relocated-final"
    shutil.copytree(Path(result.render_usd_path or "").parent, relocated)
    repaired_stage = None
    shutil.rmtree(tmp_path / "repair")
    relocated_stage = Usd.Stage.Open(str(relocated / "asset.usda"))
    assert relocated_stage is not None
    assert relocated_stage.GetPrimAtPath("/GeometryRepairAsset/Render/Mesh")
    assert (relocated / "render.usd").is_file()
    assert (relocated / "collision.usda").is_file()


def test_structure_only_repair_does_not_deinstance_meshes(tmp_path: Path) -> None:
    from pxr import Usd, UsdGeom

    source = _write_cube(tmp_path / "instanced_centimeters.usda")
    stage = Usd.Stage.Open(str(source))
    asset = stage.GetPrimAtPath("/Asset")
    asset.SetInstanceable(True)
    instance = UsdGeom.Xform.Define(stage, "/SecondAsset").GetPrim()
    instance.GetReferences().AddInternalReference("/Asset")
    instance.SetInstanceable(True)
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    stage.GetRootLayer().Save()

    diagnosis = diagnose_asset(source, profile="static_environment")
    assert diagnosis.metrics.instance_proxy_mesh_count > 0
    assert "asset:noncanonical_stage_units" in diagnosis.blocking_issue_ids
    plan = _plan_repair(
        RepairRequest(
            source_path=source,
            output_dir=tmp_path / "repair",
            profile="static_environment",
            mode="auto",
        ),
        diagnosis,
        source,
        tmp_path / "repair_plan.json",
    )

    workers = [operation.worker for operation in plan.operations]
    assert "usd_structure_repair" in workers
    assert "scene_optimizer_deinstance" not in workers


def test_packaging_only_edit_proves_exact_world_geometry_match(tmp_path: Path) -> None:
    from pxr import Usd

    source = _write_cube(tmp_path / "source.usda")
    candidate = tmp_path / "candidate.usda"
    shutil.copy2(source, candidate)
    stage = Usd.Stage.Open(str(candidate))
    stage.GetRootLayer().customLayerData = {"packagingOperation": "deinstance"}
    stage.GetRootLayer().Save()

    report = compare_geometry(
        source,
        candidate,
        drift_band="identity",
        budgets=RepairBudgets(),
    )

    assert file_sha256(source) != file_sha256(candidate)
    assert report.status == "pass"
    assert report.exact_world_geometry_match is True
    assert report.exact_world_surface_match is True
    assert report.sample_count_source == 0
    assert report.surface_distance_p99_m == pytest.approx(0.0)


def test_unmeasurable_volume_drift_is_disclosed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pxr import Gf, Usd, UsdGeom, Vt

    import geometry_repair.fidelity as fidelity_module

    source = _write_cube(tmp_path / "source.usda", open_face=True)
    candidate = _write_cube(tmp_path / "candidate.usda", open_face=True)
    stage = Usd.Stage.Open(str(candidate))
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/Asset/Mesh"))
    points = list(mesh.GetPointsAttr().Get())
    first = points[0]
    points[0] = Gf.Vec3f(float(first[0]) + 1e-4, first[1], first[2])
    mesh.GetPointsAttr().Set(Vt.Vec3fArray(points))
    stage.GetRootLayer().Save()
    measured_drifts = iter((0.0, None))
    monkeypatch.setattr(
        fidelity_module,
        "_relative_drift",
        lambda _source, _candidate: next(measured_drifts),
    )

    report = fidelity_module.compare_geometry(
        source,
        candidate,
        drift_band="conservative",
        budgets=RepairBudgets(),
    )

    assert report.failures == []
    assert any("enclosed-volume drift could not be measured" in item for item in report.warnings)


def test_workers_cannot_self_verify_hard_geometry_blockers() -> None:
    admitted = _admitted_worker_verified_issue_ids(
        [
            "mesh:self_intersections",
            "mesh:self_intersections_not_evaluated",
        ],
        [
            "mesh:self_intersections",
            "mesh:self_intersections_not_evaluated",
        ],
    )

    assert admitted == ["mesh:self_intersections_not_evaluated"]


def test_exact_geometry_part_correspondence_does_not_use_proximity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pxr import Usd

    source = _write_cube(tmp_path / "source.usda")
    candidate = tmp_path / "candidate.usda"
    shutil.copy2(source, candidate)
    stage = Usd.Stage.Open(str(candidate))
    stage.GetRootLayer().customLayerData = {"packagingOperation": "identity-proof"}
    stage.GetRootLayer().Save()

    def _unexpected_proximity(*_args, **_kwargs):
        raise AssertionError("exact geometry must not use approximate part proximity")

    monkeypatch.setattr(trimesh.proximity, "closest_point", _unexpected_proximity)

    report = compare_geometry(
        source,
        candidate,
        drift_band="identity",
        budgets=RepairBudgets(),
    )

    assert report.status == "pass"
    assert report.ambiguous_part_mapping_count == 0
    assert report.part_correspondence
    assert all(
        record.get("evidence_mode") == "exact_world_geometry"
        for record in report.part_correspondence
    )


def test_exact_geometry_fidelity_skips_full_mesh_measurement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pxr import Usd

    import geometry_repair.fidelity as fidelity_module

    source = _write_cube(tmp_path / "source.usda")
    candidate = tmp_path / "candidate.usda"
    shutil.copy2(source, candidate)
    stage = Usd.Stage.Open(str(candidate))
    stage.GetRootLayer().customLayerData = {"packagingOperation": "identity-fast-path"}
    stage.GetRootLayer().Save()
    monkeypatch.setattr(
        fidelity_module,
        "_measure_asset",
        lambda *_args, **_kwargs: pytest.fail("exact geometry must skip full mesh measurement"),
    )

    report = fidelity_module.compare_geometry(
        source,
        candidate,
        drift_band="identity",
        budgets=RepairBudgets(),
    )

    assert report.status == "pass"
    assert report.exact_world_geometry_match is True
    assert report.part_correspondence


def test_conservative_cleanup_proves_exact_surface_set_match(tmp_path: Path) -> None:
    source = _write_cube(tmp_path / "source.usda", duplicate_face=True)
    candidate = _write_cube(tmp_path / "candidate.usda")

    report = compare_geometry(
        source,
        candidate,
        drift_band="conservative",
        budgets=RepairBudgets(),
        expected_issue_ids=["mesh:duplicate_faces"],
    )

    assert report.status == "pass"
    assert report.exact_world_geometry_match is False
    assert report.exact_world_surface_match is True
    assert report.surface_distance_p99_m == pytest.approx(0.0)
    assert report.source_face_coverage_ratio == pytest.approx(1.0)


def test_surface_set_match_handles_near_duplicate_vertex_indices(tmp_path: Path) -> None:
    from pxr import Gf, Usd, UsdGeom, Vt

    source = _write_cube(tmp_path / "source.usda")
    candidate = tmp_path / "candidate.usda"
    shutil.copy2(source, candidate)
    for path, add_face in ((source, True), (candidate, False)):
        stage = Usd.Stage.Open(str(path))
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/Asset/Mesh"))
        points = list(mesh.GetPointsAttr().Get())
        points.extend(Gf.Vec3f(point[0] + 5e-8, point[1], point[2]) for point in points[:3])
        mesh.GetPointsAttr().Set(Vt.Vec3fArray(points))
        if add_face:
            counts = list(mesh.GetFaceVertexCountsAttr().Get())
            indices = list(mesh.GetFaceVertexIndicesAttr().Get())
            mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray([*counts, 3]))
            mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray([*indices, 8, 10, 9]))
        stage.GetRootLayer().Save()

    report = compare_geometry(
        source,
        candidate,
        drift_band="conservative",
        budgets=RepairBudgets(),
        expected_issue_ids=["mesh:duplicate_faces"],
    )

    assert report.status == "pass"
    assert report.exact_world_geometry_match is False
    assert report.exact_world_surface_match is True
    assert report.surface_distance_p99_m == pytest.approx(0.0)


def test_unmeasured_protected_feature_makes_changed_candidate_conditional(
    tmp_path: Path,
) -> None:
    source = _write_cube(
        tmp_path / "corrupt.usda",
        duplicate_face=True,
        degenerate_face=True,
    )

    result = run_geometry_repair(
        RepairRequest(
            source_path=source,
            output_dir=tmp_path / "repair",
            profile="rigid_pick_place",
            mode="auto",
            collision_runtime_engine="fake",
            protected_features=[ProtectedFeature(name="service opening", kind="opening")],
        )
    )

    assert result.outcome == "conditional"
    fidelity = json.loads(Path(result.attempts[-1].fidelity_path or "").read_text())
    assert fidelity["protected_features_unmeasured"] == ["service opening"]


def test_collision_negative_space_probe_preserves_annulus_opening(tmp_path: Path) -> None:
    import trimesh

    source = _write_trimesh(
        tmp_path / "annulus.usda",
        trimesh.creation.annulus(r_min=0.5, r_max=1.0, height=0.3, sections=24),
    )
    result = run_geometry_repair(
        RepairRequest(
            source_path=source,
            output_dir=tmp_path / "repair",
            profile="rigid_pick_place",
            mode="auto",
            protected_features=[
                ProtectedFeature(
                    name="center bore",
                    kind="opening",
                    scope_path="/Asset/Mesh",
                    minimum_clearance_m=0.35,
                    tolerance_m=0.02,
                    probe=ProtectedFeatureProbe(
                        kind="negative_space_path",
                        points_m=[[0.0, 0.0, -0.3], [0.0, 0.0, 0.3]],
                        radius_m=0.35,
                        tolerance_m=0.02,
                    ),
                )
            ],
            collision_runtime_engine="fake",
        )
    )

    assert result.outcome == "conditional"
    collision = json.loads(Path(result.collision_report_path or "").read_text())
    assert collision["protected_features_unmeasured"] == []
    assert collision["protected_feature_probes"][0]["status"] == "pass"
    assert collision["protected_feature_probes"][0]["minimum_clearance_m"] >= 0.33


def test_collision_negative_space_probe_rejects_blocked_path(tmp_path: Path) -> None:
    source = _write_cube(tmp_path / "cube.usda")
    result = run_geometry_repair(
        RepairRequest(
            source_path=source,
            output_dir=tmp_path / "repair",
            profile="rigid_pick_place",
            mode="auto",
            protected_features=[
                ProtectedFeature(
                    name="impossible bore",
                    kind="opening",
                    probe=ProtectedFeatureProbe(
                        kind="negative_space_path",
                        points_m=[[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]],
                        radius_m=0.1,
                        tolerance_m=0.01,
                    ),
                )
            ],
            collision_runtime_engine="fake",
        )
    )

    assert result.outcome == "rejected"
    certificate = json.loads(Path(result.certificate_path).read_text())
    assert any("inside solid geometry" in item for item in certificate["blockers"])


def test_measured_collision_feature_failure_is_not_labeled_unmeasured() -> None:
    unmeasured: set[str] = set()
    failures: list[str] = []
    warnings: list[str] = []

    _record_protected_feature_probe_outcomes(
        protected_features=[ProtectedFeature(name="impossible bore", kind="opening")],
        probes=[
            ProtectedFeatureProbeResult(
                feature_name="impossible bore",
                probe_kind="negative_space_path",
                status="fail",
                sample_count=11,
                occupied_sample_count=4,
                failures=["4 path samples are inside solid geometry"],
            )
        ],
        protected_features_unmeasured=unmeasured,
        failures=failures,
        warnings=warnings,
    )

    assert unmeasured == set()
    assert failures == [
        "protected collision feature 'impossible bore': 4 path samples are inside solid geometry"
    ]
    assert warnings == []


def test_source_provenance_and_correspondence_are_preserved(tmp_path: Path) -> None:
    source = _write_cube(tmp_path / "cube.usda")
    result = run_geometry_repair(
        RepairRequest(
            source_path=source,
            output_dir=tmp_path / "repair",
            profile="visual_only",
            mode="auto",
            source_uri="s3://example-bucket/assets/cube.usda",
            source_license="CC-BY-4.0",
            source_provenance={"dataset": "phase-1-test"},
        )
    )

    source_package = json.loads(Path(result.source_package_path).read_text())
    source_format = json.loads(Path(result.source_format_validation_path or "").read_text())
    usd_intake = json.loads(Path(result.usd_intake_path or "").read_text())
    scalable_audit = json.loads(Path(result.scalable_audit_path or "").read_text())
    correspondence = json.loads(Path(result.correspondence_path or "").read_text())
    assert source_package["source_uri"] == "s3://example-bucket/assets/cube.usda"
    assert source_package["source_license"] == "CC-BY-4.0"
    assert source_package["source_provenance"] == {"dataset": "phase-1-test"}
    assert source_format["validator"] == "OpenUSD UsdUtils.ComplianceChecker"
    assert source_format["status"] == "pass"
    assert usd_intake["schema_version"] == "geometry-repair.usd-intake.v1"
    assert usd_intake["source_unchanged"] is True
    assert scalable_audit["schema_version"] == "geometry-repair.asset-scalable-audit.v1"
    assert scalable_audit["audit"]["self_intersection"]["status"] == "evaluated_pass"
    assert correspondence["status"] == "pass"
    assert correspondence["part_mapping_coverage_ratio"] == 1.0


def test_disabled_copyleft_worker_is_rejected_by_executable_policy(tmp_path: Path) -> None:
    source = _write_cube(tmp_path / "cube.usda")

    with pytest.raises(ValueError, match="Unknown or unapproved"):
        run_geometry_repair(
            RepairRequest(
                source_path=source,
                output_dir=tmp_path / "repair",
                profile="visual_only",
                enabled_workers=["cgal_local_repair"],
            )
        )


def test_worker_registry_allows_only_approved_oss_or_nvidia() -> None:
    registry = worker_registry()
    assert registry["schema_version"] == "geometry-repair.worker-registry.v1"
    enabled = {
        "enabled",
        "enabled_with_notice",
        "enabled_external",
    }
    for record in registry["workers"]:
        if record["approval_status"] in enabled:
            assert record["component_class"] in {"open_source", "nvidia_first_party"}
            assert record["license"]
            assert record["source"]
            assert record["version_evidence"]
    assert_worker_enabled("trimesh_conservative_cleanup")
    assert_worker_enabled("sdf_rebuild")
    assert_worker_enabled("sdf_collision_rebuild")
    assert_worker_enabled("openvdb_rebuild")
    assert_worker_enabled("openvdb_collision_rebuild")
    assert_worker_enabled("ocp_shape_heal")
    assert_worker_enabled("coacd_collision")
    records = {record["name"]: record for record in registry["workers"]}
    assert "openvdb_rebuild" not in records
    assert "openvdb_collision_rebuild" not in records
    for name in ("sdf_rebuild", "sdf_collision_rebuild"):
        record = records[name]
        assert record["component_class"] == "nvidia_first_party"
        assert record["license"].startswith("Geometry Repair Apache-2.0")
        assert record["source"] == "world-understanding/agentic/packages/geometry_repair"
        assert "backend_id" not in record
        assert "native_call_scope" not in record
        assert record["backend_qualification_policy"] == (
            "geometry_repair/sdf_backend_qualifications.json"
        )
        assert (
            "selected drivers require checked-in behavioral qualification"
            in record["version_evidence"]
        )
        assert record["execution_scope"] == "isolated_subprocess"
    with pytest.raises(ValueError, match="external shared workflow"):
        assert_worker_enabled("scene_optimizer")


def test_legacy_sdf_worker_ids_normalize_at_public_boundaries(tmp_path: Path) -> None:
    request = RepairRequest(
        source_path=tmp_path / "source.usda",
        output_dir=tmp_path / "repair",
        profile="visual_only",
        enabled_workers=["openvdb_rebuild", "openvdb_collision_rebuild"],
    )
    assert request.enabled_workers == ["sdf_rebuild", "sdf_collision_rebuild"]

    operation = RepairOperation(
        operation_id="legacy-rebuild",
        worker="openvdb_rebuild",
        implementation="legacy-record",
        drift_band="reconstructive",
        source_checkpoint=str(tmp_path / "source.usda"),
    )
    assert operation.worker == "sdf_rebuild"

    intent = RepairIntent(
        intent_id="legacy-collision-rebuild",
        authority="deterministic",
        target_role="collision",
        defect_class="collision_proxy",
        expected_topology_effects=["generate_collision_representation"],
        candidate_workers=["openvdb_collision_rebuild"],
    )
    assert intent.candidate_workers == ["sdf_collision_rebuild"]

    args = _parser().parse_args(
        [
            str(tmp_path / "source.usda"),
            "--out",
            str(tmp_path / "repair"),
            "--profile",
            "visual_only",
            "--worker",
            "openvdb_rebuild",
        ]
    )
    assert args.worker == ["sdf_rebuild"]


def test_sdf_worker_alias_collisions_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(
        ValueError, match="duplicate canonical worker 'sdf_rebuild'"
    ) as request_error:
        RepairRequest.model_validate(
            {
                "source_path": tmp_path / "source.usda",
                "output_dir": tmp_path / "repair",
                "profile": "visual_only",
                "enabled_workers": ("sdf_rebuild", "openvdb_rebuild"),
            }
        )
    assert "'sdf_rebuild' and 'openvdb_rebuild'" in str(request_error.value)

    with pytest.raises(
        ValueError, match="duplicate canonical worker 'sdf_rebuild'"
    ) as intent_error:
        RepairIntent(
            intent_id="duplicate-sdf-alias",
            authority="deterministic",
            target_role="render",
            defect_class="mesh_topology",
            candidate_workers=["openvdb_rebuild", "sdf_rebuild"],
        )
    assert "'openvdb_rebuild' and 'sdf_rebuild'" in str(intent_error.value)

    request = RepairRequest.model_validate(
        {
            "source_path": tmp_path / "source.usda",
            "output_dir": tmp_path / "repair",
            "profile": "visual_only",
            "enabled_workers": ("openvdb_rebuild",),
        }
    )
    assert request.enabled_workers == ["sdf_rebuild"]


def test_legacy_openvdb_worker_import_is_the_sdf_worker_module() -> None:
    from geometry_repair.workers import openvdb_rebuild, sdf_rebuild

    assert openvdb_rebuild is sdf_rebuild
    assert openvdb_rebuild.OpenVdbRebuildWorker is sdf_rebuild.SdfRebuildWorker


def test_openvdb_reconstruction_is_opt_in_and_escalates_bounded_closing(
    tmp_path: Path,
) -> None:
    source = _write_cube(tmp_path / "open_cube.usda", open_face=True)
    diagnosis = diagnose_asset(source, profile="rigid_pick_place")
    assert "mesh:boundary_edges" in diagnosis.blocking_issue_ids

    disabled_plan = _plan_repair(
        RepairRequest(
            source_path=source,
            output_dir=tmp_path / "disabled",
            profile="rigid_pick_place",
            mode="auto",
            enabled_workers=["sdf_rebuild"],
        ),
        diagnosis,
        source,
        tmp_path / "disabled_plan.json",
    )
    assert [operation.worker for operation in disabled_plan.operations] == ["noop"]

    enabled_plan = _plan_repair(
        RepairRequest(
            source_path=source,
            output_dir=tmp_path / "enabled",
            profile="rigid_pick_place",
            mode="auto",
            enabled_workers=["sdf_rebuild"],
            budgets=RepairBudgets(
                allow_reconstructive=True,
                max_attempts=5,
                timeout_s=900.0,
            ),
        ),
        diagnosis,
        source,
        tmp_path / "enabled_plan.json",
    )
    rebuilds = [
        operation for operation in enabled_plan.operations if operation.worker == "sdf_rebuild"
    ]
    assert [operation.parameters["max_grid_dimension"] for operation in rebuilds] == [
        256,
        192,
        160,
        128,
    ]
    assert [operation.parameters["closing_steps"] for operation in rebuilds] == [2, 2, 2, 2]
    assert [operation.parameters["smoothing_steps"] for operation in rebuilds] == [1, 1, 1, 1]
    assert [operation.parameters["timeout_s"] for operation in rebuilds] == [900.0] * 4
    assert all(operation.drift_band == "reconstructive" for operation in rebuilds)
    assert all(operation.parameters["mode"] == "signed" for operation in rebuilds)
    assert all(
        operation.implementation
        == f"geometry_repair.workers.sdf_rebuild+{OPENVDB_IMPLEMENTATION_VERSION}"
        for operation in rebuilds
    )
    assert _worker_version("sdf_rebuild") == OPENVDB_IMPLEMENTATION_VERSION


def test_invariant_worker_refusal_is_memoized_across_parameter_variants(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_cube(tmp_path / "open_cube.usda", open_face=True)
    calls: list[str] = []

    def _unavailable_worker(**kwargs) -> WorkerResult:
        calls.append(kwargs["operation"].operation_id)
        return WorkerResult(
            status="unavailable",
            failures=["source has authored attributes that cannot be remapped safely"],
        )

    monkeypatch.setattr(
        "geometry_repair.orchestrator._run_worker_subprocess",
        _unavailable_worker,
    )

    result = run_geometry_repair(
        RepairRequest(
            source_path=source,
            output_dir=tmp_path / "repair",
            profile="rigid_pick_place",
            mode="auto",
            enabled_workers=["sdf_rebuild"],
            budgets=RepairBudgets(allow_reconstructive=True, max_attempts=5),
        )
    )

    rebuild_attempts = [
        attempt for attempt in result.attempts if attempt.operation.worker == "sdf_rebuild"
    ]
    assert len(calls) == 1
    assert len(rebuild_attempts) == 4
    assert all(attempt.status == "unavailable" for attempt in rebuild_attempts)
    memoized_reports = [
        json.loads(Path(attempt.worker_report_path).read_text(encoding="utf-8"))
        for attempt in rebuild_attempts[1:]
    ]
    assert all(report["metadata"]["capability_memoized"] is True for report in memoized_reports)


def test_phase2_candidate_licenses_and_cgal_evaluation_boundary(tmp_path: Path) -> None:
    records = phase2_candidate_decisions()["candidates"]
    decisions = {item["name"]: item for item in records}
    assert len(records) == 30
    assert len(decisions) == len(records)
    assert all(item["operations"] for item in records)
    assert all(len(item["operations"]) == len(set(item["operations"])) for item in records)
    assert decisions["openvdb_sdf_driver"]["worker"] == "sdf_rebuild"
    assert decisions["openvdb_sdf_driver"]["operations"] == [
        "per_part_signed_level_set_reconstruction",
        "per_part_unsigned_offset_reconstruction",
    ]
    assert decisions["openvdb_sdf_driver"]["license"].startswith("Apache-2.0")
    assert decisions["openvdb_sdf_driver"]["decision"] == "promoted_bounded_production_candidate"
    assert decisions["cellocut_rebuild"]["decision"] == "shadow_evaluated_not_promoted"
    assert decisions["pamo_rebuild"]["decision"].startswith("disabled")

    with pytest.raises(ValueError, match="allow_gpl_evaluation=True"):
        run_cgal_exact_audit(tmp_path / "unused.usda", tmp_path / "audit.json")

    non_executable = tmp_path / "cgal-audit"
    non_executable.write_text("not executable", encoding="utf-8")
    with patch.dict(
        os.environ,
        {"GEOMETRY_REPAIR_CGAL_AUDIT_EXECUTABLE": str(non_executable)},
    ):
        with pytest.raises(RuntimeError, match="not executable"):
            run_cgal_exact_audit(
                tmp_path / "unused.usda",
                tmp_path / "audit.json",
                allow_gpl_evaluation=True,
            )


def test_bounded_hole_fill_restores_a_triangular_hole_exactly(tmp_path: Path) -> None:
    import numpy as np
    import trimesh

    ground_truth_mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.05)
    damaged_mesh = ground_truth_mesh.copy()
    keep = np.ones(len(damaged_mesh.faces), dtype=bool)
    keep[int(np.argmax(np.asarray(damaged_mesh.triangles_center)[:, 2]))] = False
    damaged_mesh.update_faces(keep)
    damaged_mesh.remove_unreferenced_vertices()
    source = _write_trimesh(tmp_path / "damaged.usda", damaged_mesh)
    ground_truth = _write_trimesh(tmp_path / "ground_truth.usda", ground_truth_mesh)

    result = run_geometry_repair(
        RepairRequest(
            source_path=source,
            output_dir=tmp_path / "repair",
            profile="rigid_pick_place",
            mode="auto",
            enabled_workers=["trimesh_bounded_hole_fill", "coacd_collision"],
            budgets=RepairBudgets(allow_reconstructive=True),
            collision_runtime_engine="fake",
        )
    )

    accepted = next(attempt for attempt in result.attempts if attempt.status == "accepted")
    assert accepted.operation.worker == "trimesh_bounded_hole_fill"
    assert result.render_usd_path is not None
    fidelity = compare_geometry(
        ground_truth,
        Path(result.render_usd_path),
        drift_band="reconstructive",
        budgets=RepairBudgets(),
    )
    assert fidelity.exact_world_surface_match is True
    assert fidelity.failures == []


def test_repaired_candidate_can_prove_feature_when_damaged_source_probe_is_unavailable(
    tmp_path: Path,
) -> None:
    import numpy as np
    import trimesh

    complete = trimesh.creation.torus(
        major_radius=0.035,
        minor_radius=0.009,
        major_sections=32,
        minor_sections=16,
    )
    damaged = complete.copy()
    keep = np.ones(len(damaged.faces), dtype=bool)
    keep[int(np.argmax(np.asarray(damaged.triangles_center)[:, 2]))] = False
    damaged.update_faces(keep)
    damaged.remove_unreferenced_vertices()
    source = _write_trimesh(tmp_path / "damaged_torus.usda", damaged)
    candidate = tmp_path / "repaired_torus.usda"
    worker = TrimeshBoundedHoleFillWorker()
    worker_result = worker.execute(
        source=source,
        output=candidate,
        operation=RepairOperation(
            operation_id="bounded-hole",
            worker=worker.name,
            implementation="test",
            drift_band="reconstructive",
            source_checkpoint=str(source),
        ),
    )
    assert worker_result.status == "completed"
    feature = ProtectedFeature(
        name="center opening",
        kind="opening",
        scope_path="/Asset/Mesh",
        minimum_clearance_m=0.02,
        probe=ProtectedFeatureProbe(
            kind="negative_space_path",
            points_m=[[0.0, 0.0, -0.02], [0.0, 0.0, 0.02]],
            radius_m=0.02,
        ),
    )

    report = compare_geometry(
        source,
        candidate,
        drift_band="reconstructive",
        budgets=RepairBudgets(),
        protected_features=[feature],
        expected_issue_ids=["mesh:boundary_edges"],
    )

    assert report.failures == []
    assert report.protected_features_passed == ["center opening"]
    assert report.protected_features_unmeasured == []
    assert report.protected_feature_probes[0].status == "pass"


def test_openvdb13_worker_rebuilds_a_closed_mesh_per_part(tmp_path: Path) -> None:
    source = _write_cube(tmp_path / "cube.usda")
    output = tmp_path / "rebuilt.usda"
    worker = SdfRebuildWorker()
    available, reason = worker.available()
    if not available:
        pytest.skip(reason or "source-locked OpenVDB 13 runtime is unavailable")

    result = worker.execute(
        source=source,
        output=output,
        operation=RepairOperation(
            operation_id="openvdb-integration",
            worker="sdf_rebuild",
            implementation="test",
            parameters={
                "mode": "signed",
                "max_grid_dimension": 64,
                "max_output_faces": 100_000,
                "half_width": 3.0,
                "offset_voxels": 0.75,
                "adaptivity": 0.01,
                "closing_steps": 2,
                "smoothing_steps": 1,
            },
            issue_ids=[],
            drift_band="reconstructive",
            source_checkpoint=str(source),
        ),
    )

    assert result.status == "completed"
    assert output.is_file()
    assert result.metadata["part_identity_strategy"] == "one_source_mesh_to_same_usd_prim"
    assert result.metadata["sdf_backend"]["provenance"]["distribution_version"] == "13.0.0+wu.3"
    assert result.metadata["invocations"][0]["backend_report"]["smoothing_steps"] == 1
    diagnosis = diagnose_asset(output, profile="rigid_pick_place")
    assert diagnosis.metrics.mesh.mesh_count == 1
    assert diagnosis.metrics.mesh.watertight is True


def test_sdf_reconstruction_preserves_elongated_open_mesh_grid_admission() -> None:
    import numpy as np
    import trimesh

    from geometry_repair.workers.sdf_rebuild import _bounded_voxel_size
    from geometry_repair.workers.sdf_reconstruction import (
        SdfReconstructionControls,
        reconstruct_sdf_mesh,
    )

    worker = SdfRebuildWorker()
    available, reason = worker.available()
    if not available:
        pytest.skip(reason or "source-locked OpenVDB 13 runtime is unavailable")
    mesh = trimesh.creation.box(extents=(1.0, 0.05, 0.05))
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.faces[:-1], dtype=np.int32)
    voxel_size, dimensions = _bounded_voxel_size(
        vertices,
        triangles,
        diagonal=float(np.linalg.norm(np.ptp(vertices, axis=0))),
        max_grid_dimension=64,
        mode="signed",
        half_width=3.0,
        offset_voxels=0.75,
        closing_steps=2,
    )
    result = reconstruct_sdf_mesh(
        vertices,
        triangles,
        SdfReconstructionControls(
            mode="signed",
            voxel_size=voxel_size,
            half_width=3.0,
            offset_voxels=0.75,
            adaptivity=0.01,
            closing_steps=2,
            smoothing_steps=1,
            deterministic_seed=0,
            max_grid_dimension=64,
            max_input_vertices=1_000,
            max_input_faces=1_000,
            max_active_voxels=2_000_000,
            max_output_faces=100_000,
        ),
    )

    assert max(dimensions) <= 64
    assert result.evidence["resource_usage"]["estimated_grid_dimensions"] == tuple(dimensions)
    assert result.evidence["algorithm"]["route"] == "signed_topology_closing"
    assert len(result.vertices) > 0
    assert len(result.triangles) > 0


def test_valid_native_brep_is_preserved_and_requires_usd_handoff(
    tmp_path: Path,
) -> None:
    pytest.importorskip("OCP")
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.BRepTools import BRepTools

    source = tmp_path / "box.brep"
    assert BRepTools.Write_s(BRepPrimAPI_MakeBox(10.0, 20.0, 30.0).Shape(), str(source))

    result = run_geometry_repair(
        RepairRequest(
            source_path=source,
            output_dir=tmp_path / "repair",
            profile="rigid_pick_place",
            mode="auto",
        )
    )

    assert result.outcome == "conditional"
    assert result.render_usd_path is None
    assert result.attempts[0].status == "accepted"
    diagnosis = diagnose_asset(source, profile="rigid_pick_place")
    assert diagnosis.metrics.brep.validity_status_counts["NoError"] > 0
    assert diagnosis.metrics.brep.subshape_status_counts["face"]["NoError"] == 6
    assert diagnosis.metrics.brep.minimum_edge_length_source_units == pytest.approx(10.0)


def test_native_brep_without_backend_fails_closed_with_provider_guidance(
    tmp_path: Path,
) -> None:
    source = tmp_path / "provider_source.step"
    source.write_bytes(b"inert native CAD provenance")

    with patch(
        "geometry_repair.diagnosis.ocp_available",
        return_value=(False, "optional backend is not installed"),
    ):
        diagnosis = diagnose_asset(source, profile="rigid_pick_place")

    assert diagnosis.metrics.brep.evaluated is False
    assert diagnosis.blocking_issue_ids == ["brep:backend_unavailable"]
    issue = diagnosis.issues[0]
    assert issue.candidate_repairs == []
    assert "validated USD or mesh representation" in issue.summary
    assert issue.evidence == {"reason": "optional backend is not installed"}


def test_ocp_worker_executes_bounded_sewing_before_shape_fix(tmp_path: Path) -> None:
    pytest.importorskip("OCP")
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.BRepTools import BRepTools
    from OCP.TopAbs import TopAbs_SHELL
    from OCP.TopExp import TopExp_Explorer

    source = tmp_path / "source.brep"
    output = tmp_path / "healed.brep"
    assert BRepTools.Write_s(BRepPrimAPI_MakeBox(10.0, 20.0, 30.0).Shape(), str(source))

    report = heal_brep(
        source,
        output,
        precision=1e-7,
        maximum_tolerance=1e-6,
        sew=True,
    )

    assert output.is_file()
    assert report["sewing"]["requested"] is True
    assert report["sewing"]["executed"] is False
    assert "solid identity" in report["sewing"]["skip_reason"]
    assert report["after"]["valid"] is True
    assert report["before"]["solid_count"] == report["after"]["solid_count"] == 1
    assert report["shape_fix"]["status_fail"] is False
    assert set(report["shape_fix"]["recorded_subshape_counts"]) == {
        "solid",
        "shell",
        "face",
        "wire",
        "edge",
    }

    shell_source = tmp_path / "shell_source.brep"
    shell_output = tmp_path / "shell_healed.brep"
    explorer = TopExp_Explorer(BRepPrimAPI_MakeBox(10.0, 20.0, 30.0).Shape(), TopAbs_SHELL)
    assert explorer.More()
    assert BRepTools.Write_s(explorer.Current(), str(shell_source))
    shell_report = heal_brep(
        shell_source,
        shell_output,
        precision=1e-7,
        maximum_tolerance=1e-6,
        sew=True,
    )
    assert shell_report["sewing"]["executed"] is True
    assert shell_report["sewing"]["deleted_face_count"] == 0


def test_numeric_diagnosis_covers_normals_uvs_and_singular_transforms(
    tmp_path: Path,
) -> None:
    import math

    from pxr import Gf, Sdf, Usd, UsdGeom, Vt

    source = _write_cube(tmp_path / "numeric_corruption.usda")
    stage = Usd.Stage.Open(str(source))
    root = stage.GetPrimAtPath("/Asset")
    UsdGeom.Xformable(root).AddScaleOp().Set(Gf.Vec3f(1.0, 0.0, 1.0))
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/Asset/Mesh"))
    mesh.CreateNormalsAttr(
        Vt.Vec3fArray([Gf.Vec3f(math.nan, 0.0, 1.0)] * len(mesh.GetFaceVertexIndicesAttr().Get()))
    )
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
    primvar = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st",
        Sdf.ValueTypeNames.TexCoord2fArray,
        UsdGeom.Tokens.vertex,
    )
    primvar.Set(Vt.Vec2fArray([Gf.Vec2f(math.nan, 0.0)] * len(mesh.GetPointsAttr().Get())))
    stage.GetRootLayer().Save()

    diagnosis = diagnose_asset(source, profile="visual_only")

    assert diagnosis.metrics.mesh.non_finite_normal_count > 0
    assert diagnosis.metrics.mesh.non_finite_uv_count > 0
    assert diagnosis.metrics.mesh.singular_transform_count == 1
    assert {
        "mesh:non_finite_normals",
        "mesh:non_finite_uvs",
        "mesh:singular_transforms",
    } <= set(diagnosis.blocking_issue_ids)
