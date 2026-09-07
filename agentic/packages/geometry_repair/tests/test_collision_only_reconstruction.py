# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collision-only reconstruction must preserve attributed render geometry."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh

from geometry_repair.artifacts import atomic_write_json, file_sha256
from geometry_repair.collision import (
    _dynamic_collision_reconstruction_reasons,
    build_collision_geometry,
)
from geometry_repair.mesh_io import load_meshes
from geometry_repair.models import ProtectedFeature, RepairBudgets
from geometry_repair.policy import assert_worker_operations_enabled
from geometry_repair.workers import sdf_rebuild as openvdb_worker
from geometry_repair.workers import sdf_reconstruction
from geometry_repair.workers.sdf_rebuild import (
    SdfCollisionReconstructionResult,
    reconstruct_collision_mesh_sdf,
)


def _write_attributed_open_box(path: Path) -> tuple[Path, trimesh.Trimesh]:
    from pxr import Gf, Sdf, Usd, UsdGeom, Vt

    closed = trimesh.creation.box(extents=(0.2, 0.1, 0.08))
    vertices = np.asarray(closed.vertices, dtype=np.float64)
    triangles = np.asarray(closed.faces, dtype=np.int64)
    open_triangles = triangles[:-2]

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/Asset/AttributedPart")
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(open_triangles)))
    mesh.CreateFaceVertexIndicesAttr(
        Vt.IntArray([int(value) for triangle in open_triangles for value in triangle])
    )
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    mesh.CreateExtentAttr().Set(
        Vt.Vec3fArray(
            [
                Gf.Vec3f(*(float(value) for value in vertices.min(axis=0))),
                Gf.Vec3f(*(float(value) for value in vertices.max(axis=0))),
            ]
        )
    )
    primvar = UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
        "st",
        Sdf.ValueTypeNames.TexCoord2fArray,
        UsdGeom.Tokens.faceVarying,
    )
    primvar.Set(
        Vt.Vec2fArray(
            [
                Gf.Vec2f(float(index % 2), float((index // 2) % 2))
                for index in range(3 * len(open_triangles))
            ]
        )
    )
    stage.GetRootLayer().Save()
    return path, closed


def test_collision_only_reconstruction_preserves_attributed_render(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, closed = _write_attributed_open_box(tmp_path / "attributed_open.usda")
    source_sha256 = file_sha256(source)

    def fake_reconstruction(**kwargs) -> SdfCollisionReconstructionResult:
        assert 600.0 < float(kwargs["timeout_s"]) <= 900.0
        work_dir = Path(kwargs["work_dir"])
        work_dir.mkdir(parents=True, exist_ok=True)
        evidence_path = work_dir / "collision_reconstruction.json"
        atomic_write_json(
            evidence_path,
            {
                "schema_version": "geometry-repair.sdf-collision-reconstruction.v1",
                "status": "success",
                "target_role": "collision",
                "source_render_unchanged": True,
            },
        )
        return SdfCollisionReconstructionResult(
            status="success",
            vertices=np.asarray(closed.vertices, dtype=np.float64),
            triangles=np.asarray(closed.faces, dtype=np.int64),
            evidence_path=str(evidence_path),
            warnings=["generated collision reference requires review"],
        )

    monkeypatch.setattr(
        "geometry_repair.collision.reconstruct_collision_mesh_sdf",
        fake_reconstruction,
    )
    report = build_collision_geometry(
        source,
        tmp_path / "collision.usda",
        profile="rigid_pick_place",
        budgets=RepairBudgets(allow_reconstructive=True, sample_point_limit=256),
        protected_features=[
            ProtectedFeature(
                name="render_only_inferred_opening",
                kind="opening",
                minimum_clearance_m=1e-9,
                source="inferred_hypothesis",
                affected_roles=["render"],
            )
        ],
        coacd_enabled=False,
        sdf_collision_rebuild_enabled=True,
        runtime_engine="skip",
        report_path=tmp_path / "collision_report.json",
    )

    assert report.status == "conditional"
    assert report.collision_reconstruction_count == 1
    assert len(report.collision_reconstruction_paths) == 1
    assert report.source_render_unchanged is True
    assert report.source_render_sha256 == source_sha256
    assert report.source_render_sha256_after == source_sha256
    assert file_sha256(source) == source_sha256
    assert Path(report.collision_path or "").is_file()
    assert report.role_excluded_protected_features == ["render_only_inferred_opening"]


def test_collision_builder_accepts_legacy_toggle_and_rejects_conflicts(
    tmp_path: Path,
) -> None:
    source = tmp_path / "render.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")

    report = build_collision_geometry(
        source,
        tmp_path / "collision.usda",
        profile="visual_only",
        budgets=RepairBudgets(),
        openvdb_collision_rebuild_enabled=False,
    )

    assert report.status == "not_required"
    with pytest.raises(ValueError, match="conflicting canonical and legacy SDF"):
        build_collision_geometry(
            source,
            tmp_path / "conflicting-collision.usda",
            profile="visual_only",
            budgets=RepairBudgets(),
            sdf_collision_rebuild_enabled=True,
            openvdb_collision_rebuild_enabled=False,
        )


def test_collision_only_reconstruction_requires_opt_in(tmp_path: Path) -> None:
    source, _closed = _write_attributed_open_box(tmp_path / "attributed_open.usda")

    report = build_collision_geometry(
        source,
        tmp_path / "collision.usda",
        profile="rigid_pick_place",
        budgets=RepairBudgets(allow_reconstructive=False, sample_point_limit=256),
        coacd_enabled=False,
        runtime_engine="skip",
    )

    assert report.status == "fail"
    assert report.representation == "none"
    assert report.collision_path is None
    assert any("not explicitly enabled" in failure for failure in report.failures)


def test_collision_reconstruction_operation_is_policy_authorized() -> None:
    assert_worker_operations_enabled(
        "sdf_collision_rebuild",
        "per_part_signed_collision_reconstruction",
    )


def test_watertight_self_intersecting_source_still_requires_reconstruction() -> None:
    measured = SimpleNamespace(
        watertight=True,
        enclosed_volume_m3=1.0,
        invalid_index_count=0,
        non_finite_vertex_count=0,
        degenerate_face_count=0,
        duplicate_face_count=0,
        over_connected_edge_count=0,
        non_manifold_vertex_count=0,
        inconsistent_orientation_edge_count=0,
        self_intersection_status="fail",
        coplanar_overlap_status="pass",
        inverted_shell_status="pass",
    )

    assert _dynamic_collision_reconstruction_reasons(measured) == ["self-intersection audit: fail"]


def test_in_process_collision_reconstruction_ignores_render_uv_topology_safely(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed = trimesh.creation.icosphere(subdivisions=3, radius=0.1)
    vertices = np.asarray(closed.vertices, dtype=np.float64)
    triangles = np.asarray(closed.faces, dtype=np.int64)
    open_mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=triangles[:-1],
        process=False,
    )
    source = tmp_path / "in_process_attributed_open.usda"
    from pxr import Gf, Sdf, Usd, UsdGeom, Vt

    stage = Usd.Stage.CreateNew(str(source))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    usd_mesh = UsdGeom.Mesh.Define(stage, "/Asset/AttributedPart")
    usd_mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices.astype(np.float32)))
    usd_mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(open_mesh.faces)))
    usd_mesh.CreateFaceVertexIndicesAttr(
        Vt.IntArray([int(value) for face in open_mesh.faces for value in face])
    )
    usd_mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    usd_mesh.CreateExtentAttr().Set(
        Vt.Vec3fArray(
            [
                Gf.Vec3f(*(float(value) for value in vertices.min(axis=0))),
                Gf.Vec3f(*(float(value) for value in vertices.max(axis=0))),
            ]
        )
    )
    primvar = UsdGeom.PrimvarsAPI(usd_mesh.GetPrim()).CreatePrimvar(
        "st",
        Sdf.ValueTypeNames.TexCoord2fArray,
        UsdGeom.Tokens.faceVarying,
    )
    primvar.Set(
        Vt.Vec2fArray(
            [
                Gf.Vec2f(float(index % 2), float((index // 2) % 2))
                for index in range(3 * len(open_mesh.faces))
            ]
        )
    )
    stage.GetRootLayer().Save()
    source_sha256 = file_sha256(source)
    meshes, _metadata = load_meshes(source)
    runtime_identity = {
        "library_version": [13, 0, 0],
        "distribution_version": openvdb_worker.OPENVDB_DISTRIBUTION_VERSION,
        "source_distribution_version": openvdb_worker.OPENVDB_DISTRIBUTION_VERSION,
        "source_lock_sha256": "1" * 64,
        "source_commit": openvdb_worker.OPENVDB_BACKEND_SOURCE_ID,
    }
    backend_identity = {
        "backend_id": "openvdb",
        "implementation_version": (
            f"openvdb-runtime-{openvdb_worker.OPENVDB_DISTRIBUTION_VERSION}"
        ),
        "operations": sorted(
            {operation.value for operation in openvdb_worker._OPENVDB13_REQUIRED_OPERATIONS}
            | {"write_fields"}
        ),
        "execution_mode": "in_process",
        "read_formats": [],
        "write_formats": ["vdb"],
        "supported_formats": ["vdb"],
        "provenance": runtime_identity,
    }
    captured: dict[str, object] = {}
    affinity_events: list[tuple[str, int | None]] = []
    monkeypatch.setattr(
        openvdb_worker,
        "_qualified_sdf_backend_identity",
        lambda _backend_id: dict(backend_identity),
    )

    def reconstruct(
        input_vertices: np.ndarray,
        input_triangles: np.ndarray,
        controls: sdf_reconstruction.SdfReconstructionControls,
    ) -> sdf_reconstruction.SdfReconstructionResult:
        affinity_events.append(("reconstruct", None))
        captured["vertices"] = input_vertices.copy()
        captured["triangles"] = input_triangles.copy()
        captured["controls"] = controls
        source_vertices = np.asarray(input_vertices, dtype=np.float32)
        source_triangles = np.asarray(input_triangles, dtype=np.int32)
        surface = sdf_reconstruction._canonical_surface(
            SimpleNamespace(
                vertices=np.asarray(closed.vertices, dtype=np.float32),
                triangles=np.asarray(closed.faces, dtype=np.int32),
                quads=np.empty((0, 4), dtype=np.int32),
            ),
            controls,
        )
        boundary_edges, non_manifold_edges, duplicate_faces = (
            sdf_reconstruction._topology_defect_counts(source_triangles)
        )
        topology_closing = controls.mode == "signed" and (
            boundary_edges + non_manifold_edges + duplicate_faces > 0
        )
        return sdf_reconstruction.SdfReconstructionResult(
            vertices=surface.vertices,
            triangles=surface.triangles,
            evidence=sdf_reconstruction._execution_evidence(
                controls=controls,
                qualification_id="geometry-repair.openvdb13.v1",
                required_operations=sdf_reconstruction._RECONSTRUCTION_OPERATIONS,
                limits=sdf_reconstruction._execution_limits(controls),
                selected_backend_identity=dict(backend_identity),
                selection_rejections=(),
                algorithm=sdf_reconstruction._algorithm_evidence(
                    controls,
                    topology_closing=topology_closing,
                ),
                estimated_dimensions=sdf_reconstruction._estimated_grid_dimensions(
                    source_vertices,
                    controls,
                    topology_closing=topology_closing,
                ),
                source_vertices=source_vertices,
                source_triangles=source_triangles,
                boundary_edges=boundary_edges,
                non_manifold_edges=non_manifold_edges,
                duplicate_faces=duplicate_faces,
                backend_call_status="succeeded",
                candidate_validation_status="accepted",
                surface=surface,
            ),
        )

    @contextmanager
    def temporary_affinity(cpu_count: int):
        affinity_events.append(("enter", cpu_count))
        try:
            yield
        finally:
            affinity_events.append(("exit", cpu_count))

    monkeypatch.setattr(openvdb_worker, "temporary_cpu_affinity", temporary_affinity)
    monkeypatch.setattr(openvdb_worker, "reconstruct_sdf_mesh", reconstruct)
    monkeypatch.setattr(openvdb_worker, "_validate_sdf_result", lambda *_a, **_k: None)

    result = reconstruct_collision_mesh_sdf(
        source_render=source,
        mesh=meshes[0],
        work_dir=tmp_path / "in_process_collision_rebuild",
        request_id="in-process-collision-rebuild",
        max_grid_dimension=64,
        max_output_faces=250_000,
        feature_voxels=6,
        minimum_feature_m=None,
        deterministic_seed=29,
        timeout_s=900.0,
        max_surface_p99_ratio=0.01,
    )

    assert result.status == "success", result.failures
    assert result.vertices is not None
    assert result.triangles is not None
    assert result.vertices.flags.owndata
    assert result.triangles.flags.owndata
    assert file_sha256(source) == source_sha256
    assert result.metadata["surface_p99_ratio"] <= 0.01
    assert np.array_equal(captured["vertices"], meshes[0].world_vertices_m)
    assert np.array_equal(captured["triangles"], meshes[0].triangles)
    controls = captured["controls"]
    assert isinstance(controls, sdf_reconstruction.SdfReconstructionControls)
    assert controls.mode == "signed"
    assert controls.backend_id == "openvdb"
    assert controls.voxel_size > 0.0
    assert result.metadata["backend"] == "sdf_tools.in_process"
    assert result.metadata["requested_outer_timeout_s"] == 900.0
    assert affinity_events == [("enter", 1), ("reconstruct", None), ("exit", 1)]
    assert result.metadata["backend_evidence"]["selected_backend_identity"]["backend_id"] == (
        "openvdb"
    )
    assert result.metadata["backend_evidence"]["selected_backend_identity"]["provenance"][
        "library_version"
    ] == [13, 0, 0]
    work_dir = tmp_path / "in_process_collision_rebuild"
    assert sorted(path.name for path in work_dir.iterdir()) == ["collision_reconstruction.json"]
    evidence = json.loads((work_dir / "collision_reconstruction.json").read_text(encoding="utf-8"))
    assert evidence["schema_version"] == "geometry-repair.sdf-collision-reconstruction.v1"


def test_collision_reconstruction_rejects_malformed_backend_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, _closed = _write_attributed_open_box(tmp_path / "source.usda")
    meshes, _metadata = load_meshes(source)
    monkeypatch.setattr(
        openvdb_worker,
        "_qualified_sdf_backend_identity",
        lambda _backend_id: {"library_version": [13, 0, 0]},
    )
    monkeypatch.setattr(
        openvdb_worker,
        "reconstruct_sdf_mesh",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )

    result = reconstruct_collision_mesh_sdf(
        source_render=source,
        mesh=meshes[0],
        work_dir=tmp_path / "malformed_collision_rebuild",
        request_id="malformed-collision-rebuild",
        max_grid_dimension=64,
        max_output_faces=250_000,
        feature_voxels=6,
        minimum_feature_m=None,
        deterministic_seed=29,
        timeout_s=60.0,
        max_surface_p99_ratio=0.01,
    )

    assert result.status == "failed"
    assert "unexpected result type" in result.failures[0]
    assert result.vertices is None
    assert result.triangles is None


def test_collision_reconstruction_refuses_symlink_work_dir_without_deleting_target(
    tmp_path: Path,
) -> None:
    source, _closed = _write_attributed_open_box(tmp_path / "source.usda")
    meshes, _metadata = load_meshes(source)
    target = tmp_path / "existing_collision_data"
    target.mkdir()
    sentinel = target / "keep.txt"
    sentinel.write_text("preserve me", encoding="utf-8")
    work_link = tmp_path / "collision_work_link"
    work_link.symlink_to(target, target_is_directory=True)

    result = reconstruct_collision_mesh_sdf(
        source_render=source,
        mesh=meshes[0],
        work_dir=work_link,
        request_id="symlink-work-dir",
        max_grid_dimension=64,
        max_output_faces=250_000,
        feature_voxels=6,
        minimum_feature_m=None,
        deterministic_seed=29,
        timeout_s=60.0,
        max_surface_p99_ratio=0.01,
    )

    assert result.status == "refused"
    assert result.failures == ["SDF collision work directory must not be a symbolic link"]
    assert work_link.is_symlink()
    assert sentinel.read_text(encoding="utf-8") == "preserve me"
