# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact attributed-topology rewrite regressions."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from geometry_repair.correspondence import (
    build_mesh_correspondence,
    write_correspondence_evidence,
)
from geometry_repair.models import RepairBudgets, RepairOperation, RepairRequest
from geometry_repair.orchestrator import run_geometry_repair
from geometry_repair.workers.geogram_local_repair import _exact_source_subset_rewrite
from geometry_repair.workers.trimesh_cleanup import TrimeshCleanupWorker
from geometry_repair.workers.trimesh_hole_fill import TrimeshBoundedHoleFillWorker


def _write_attributed_dirty_mesh(path: Path) -> Path:
    from pxr import Gf, Sdf, Usd, UsdGeom, Vt

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/Asset/Mesh")
    mesh.CreatePointsAttr(
        Vt.Vec3fArray(
            [
                Gf.Vec3f(0.0, 0.0, 0.0),
                Gf.Vec3f(1.0, 0.0, 0.0),
                Gf.Vec3f(1.0, 1.0, 0.0),
                Gf.Vec3f(0.0, 1.0, 0.0),
            ]
        )
    )
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3, 3, 3, 3]))
    mesh.CreateFaceVertexIndicesAttr(
        Vt.IntArray(
            [
                0,
                1,
                2,
                0,
                3,
                2,
                0,
                1,
                2,
                0,
                0,
                1,
            ]
        )
    )
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    normals = [Gf.Vec3f(float(index), 0.0, 1.0) for index in range(12)]
    mesh.CreateNormalsAttr(Vt.Vec3fArray(normals))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)

    primvars = UsdGeom.PrimvarsAPI(mesh.GetPrim())
    st = primvars.CreatePrimvar(
        "st",
        Sdf.ValueTypeNames.TexCoord2fArray,
        UsdGeom.Tokens.faceVarying,
    )
    st.Set(
        Vt.Vec2fArray(
            [
                Gf.Vec2f(0.0, 0.0),
                Gf.Vec2f(1.0, 0.0),
                Gf.Vec2f(1.0, 1.0),
                Gf.Vec2f(0.0, 1.0),
            ]
        )
    )
    st.SetIndices(Vt.IntArray([0, 1, 2, 0, 3, 2, 0, 1, 2, 0, 0, 1]))
    labels = primvars.CreatePrimvar(
        "faceWeight",
        Sdf.ValueTypeNames.FloatArray,
        UsdGeom.Tokens.uniform,
    )
    labels.Set(Vt.FloatArray([10.0, 20.0, 30.0, 40.0]))

    first = UsdGeom.Subset.Define(stage, "/Asset/Mesh/FirstMaterialFaces")
    first.CreateElementTypeAttr().Set(UsdGeom.Tokens.face)
    first.CreateIndicesAttr(Vt.IntArray([0, 2]))
    second = UsdGeom.Subset.Define(stage, "/Asset/Mesh/SecondMaterialFaces")
    second.CreateElementTypeAttr().Set(UsdGeom.Tokens.face)
    second.CreateIndicesAttr(Vt.IntArray([1]))
    stage.GetRootLayer().Save()
    return path


def _write_attributed_small_hole(path: Path, *, conflicting_boundary: bool) -> Path:
    from pxr import Gf, Sdf, Usd, UsdGeom, Vt

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/Asset/Mesh")
    points = [
        (-5.0, -5.0, -5.0),
        (5.0, -5.0, -5.0),
        (5.0, 5.0, -5.0),
        (-5.0, 5.0, -5.0),
        (-5.0, -5.0, 5.0),
        (5.0, -5.0, 5.0),
        (5.0, 5.0, 5.0),
        (-5.0, 5.0, 5.0),
        (6.0, 0.0, 0.0),
        (6.1, 0.0, 0.0),
        (6.0, 0.1, 0.0),
        (6.0, 0.0, 0.1),
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
        (8, 10, 9),
        (8, 9, 11),
        (9, 10, 11),
    ]
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*value) for value in points]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(faces)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([item for face in faces for item in face]))
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    mesh.CreateNormalsAttr(Vt.Vec3fArray([Gf.Vec3f(0.0, 0.0, 1.0)] * (len(faces) * 3)))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)

    primvars = UsdGeom.PrimvarsAPI(mesh.GetPrim())
    st = primvars.CreatePrimvar(
        "st",
        Sdf.ValueTypeNames.TexCoord2fArray,
        UsdGeom.Tokens.faceVarying,
    )
    st.Set(Vt.Vec2fArray([Gf.Vec2f(0.25, 0.75)]))
    st.SetIndices(Vt.IntArray([0] * (len(faces) * 3)))
    labels = primvars.CreatePrimvar(
        "faceWeight",
        Sdf.ValueTypeNames.FloatArray,
        UsdGeom.Tokens.uniform,
    )
    weights = [1.0] * len(faces)
    if conflicting_boundary:
        weights[-3:] = [1.0, 2.0, 3.0]
    labels.Set(Vt.FloatArray(weights))

    subset = UsdGeom.Subset.Define(stage, "/Asset/Mesh/MaterialFaces")
    subset.CreateElementTypeAttr().Set(UsdGeom.Tokens.face)
    subset.CreateIndicesAttr(Vt.IntArray(range(len(faces))))
    stage.GetRootLayer().Save()
    return path


def _write_dense_duplicate_vertex_mesh(
    path: Path,
    *,
    face_count: int,
    dropped_faces: set[int] | None = None,
) -> Path:
    from pxr import Gf, Usd, UsdGeom, Vt

    dropped = dropped_faces or set()
    points = []
    faces = []
    for face_id in range(face_count):
        group = face_id // 2
        x = float(group % 64)
        y = float(group // 64)
        points.extend([(x, y, 0.0), (x + 0.25, y, 0.0), (x, y + 0.25, 0.0)])
        if face_id not in dropped:
            faces.append((face_id * 3, face_id * 3 + 1, face_id * 3 + 2))

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/Asset/Mesh")
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*point) for point in points]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(faces)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([value for face in faces for value in face]))
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    stage.GetRootLayer().Save()
    return path


def test_cleanup_preserves_indexed_uv_normals_uniform_data_and_subsets(tmp_path: Path) -> None:
    from pxr import Usd, UsdGeom

    source = _write_attributed_dirty_mesh(tmp_path / "source.usda")
    output = tmp_path / "output.usda"
    result = TrimeshCleanupWorker().execute(
        source=source,
        output=output,
        operation=RepairOperation(
            operation_id="cleanup",
            worker="trimesh_conservative_cleanup",
            implementation="test",
            issue_ids=["mesh:duplicate_faces", "mesh:degenerate_faces"],
            source_checkpoint=str(source),
            drift_band="conservative",
        ),
    )

    assert result.status == "completed"
    assert result.changed is True
    assert result.metadata["correspondence_mode"] == "exact_source_face_corner"
    correspondence = build_mesh_correspondence(
        source,
        output,
        operation="trimesh_conservative_cleanup",
    )
    assert correspondence.status == "conditional"
    assert correspondence.refusal_reasons == []
    assert correspondence.material_mapping_coverage_ratio == 1.0
    assert all(item.status == "pass" for item in correspondence.attribute_transfers)
    stage = Usd.Stage.Open(str(output))
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/Asset/Mesh"))
    assert list(mesh.GetFaceVertexCountsAttr().Get()) == [3, 3]
    assert list(mesh.GetFaceVertexIndicesAttr().Get()) == [0, 1, 2, 0, 2, 3]
    assert [float(value[0]) for value in mesh.GetNormalsAttr().Get()] == [
        0.0,
        1.0,
        2.0,
        3.0,
        5.0,
        4.0,
    ]
    primvars = UsdGeom.PrimvarsAPI(mesh.GetPrim())
    assert list(primvars.GetPrimvar("st").GetIndices()) == [0, 1, 2, 0, 2, 3]
    assert list(primvars.GetPrimvar("faceWeight").Get()) == [10.0, 20.0]
    subsets = {
        subset.GetPrim().GetName(): list(subset.GetIndicesAttr().Get())
        for subset in UsdGeom.Subset.GetAllGeomSubsets(mesh)
    }
    assert subsets == {"FirstMaterialFaces": [0], "SecondMaterialFaces": [1]}
    assert np.asarray(mesh.GetPointsAttr().Get()).shape == (4, 3)


def test_small_hole_fill_extends_agreeing_attributes_and_records_correspondence(
    tmp_path: Path,
) -> None:
    from pxr import Usd, UsdGeom

    source = _write_attributed_small_hole(
        tmp_path / "source.usda",
        conflicting_boundary=False,
    )
    output = tmp_path / "output.usda"
    result = TrimeshBoundedHoleFillWorker().execute(
        source=source,
        output=output,
        operation=RepairOperation(
            operation_id="hole-fill",
            worker="trimesh_bounded_hole_fill",
            implementation="test",
            issue_ids=["mesh:small_boundary_loop"],
            source_checkpoint=str(source),
            drift_band="conservative",
        ),
    )

    assert result.status == "completed"
    correspondence = build_mesh_correspondence(
        source,
        output,
        operation="trimesh_bounded_hole_fill",
    )
    assert correspondence.status == "conditional"
    assert correspondence.refusal_reasons == []
    assert correspondence.material_mapping_coverage_ratio == 1.0
    assert all(item.status != "refused" for item in correspondence.attribute_transfers)
    assert correspondence.generated_output_regions == {"faces": [15]}

    stage = Usd.Stage.Open(str(output))
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/Asset/Mesh"))
    assert len(mesh.GetFaceVertexCountsAttr().Get()) == 16
    assert len(mesh.GetNormalsAttr().Get()) == 48
    primvars = UsdGeom.PrimvarsAPI(mesh.GetPrim())
    assert len(primvars.GetPrimvar("st").GetIndices()) == 48
    assert len(primvars.GetPrimvar("faceWeight").Get()) == 16
    subset = UsdGeom.Subset.GetAllGeomSubsets(mesh)[0]
    assert list(subset.GetIndicesAttr().Get()) == list(range(16))
    assert mesh.GetPrim().GetCustomDataByKey("geometryRepairGeneratedPatchMap")


def test_small_hole_fill_refuses_conflicting_boundary_attributes(tmp_path: Path) -> None:
    source = _write_attributed_small_hole(
        tmp_path / "source.usda",
        conflicting_boundary=True,
    )
    output = tmp_path / "output.usda"
    result = TrimeshBoundedHoleFillWorker().execute(
        source=source,
        output=output,
        operation=RepairOperation(
            operation_id="hole-fill",
            worker="trimesh_bounded_hole_fill",
            implementation="test",
            issue_ids=["mesh:small_boundary_loop"],
            source_checkpoint=str(source),
            drift_band="conservative",
        ),
    )

    assert result.status == "unavailable"
    assert "conflicting values" in result.failures[0]
    assert not output.exists()


def test_correspondence_refuses_incomplete_normal_transfer_evidence(
    tmp_path: Path,
) -> None:
    from pxr import Usd

    source = _write_attributed_small_hole(
        tmp_path / "source.usda",
        conflicting_boundary=False,
    )
    output = tmp_path / "output.usda"
    result = TrimeshBoundedHoleFillWorker().execute(
        source=source,
        output=output,
        operation=RepairOperation(
            operation_id="hole-fill",
            worker="trimesh_bounded_hole_fill",
            implementation="test",
            issue_ids=["mesh:small_boundary_loop"],
            source_checkpoint=str(source),
            drift_band="conservative",
        ),
    )
    assert result.status == "completed"
    stage = Usd.Stage.Open(str(output))
    mesh = stage.GetPrimAtPath("/Asset/Mesh")
    mesh.ClearCustomDataByKey("geometryRepairGeneratedPatchMap")
    stage.GetRootLayer().Save()

    correspondence = build_mesh_correspondence(
        source,
        output,
        operation="trimesh_bounded_hole_fill",
    )

    normals = next(
        item
        for item in correspondence.attribute_transfers
        if item.attribute_name == "source_normals"
    )
    assert correspondence.status == "refused"
    assert normals.status == "refused"
    assert normals.method == "refused"
    assert 0.0 < normals.coverage_ratio < 1.0


def test_geogram_exact_subset_mapping_accepts_only_unique_source_triangles() -> None:
    source_vertices = np.asarray(
        [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
        ],
        dtype=np.float64,
    )
    source_faces = np.asarray([(0, 1, 2), (0, 2, 3)], dtype=np.int64)
    candidate_vertices = source_vertices[[2, 3, 0, 1]]
    candidate_faces = np.asarray([(0, 1, 2)], dtype=np.int64)

    rewrite = _exact_source_subset_rewrite(
        source_vertices,
        source_faces,
        candidate_vertices,
        candidate_faces,
    )

    assert rewrite is not None
    assert rewrite.source_face_ids.tolist() == [1]
    assert rewrite.triangles.tolist() == [[0, 2, 3]]
    assert rewrite.source_corner_ids.tolist() == [[0, 1, 2]]
    generated_vertices = np.vstack((candidate_vertices, np.asarray([(0.5, 0.5, 0.0)])))
    generated_faces = np.asarray([(0, 1, 4)], dtype=np.int64)
    assert (
        _exact_source_subset_rewrite(
            source_vertices,
            source_faces,
            generated_vertices,
            generated_faces,
        )
        is None
    )


def test_dense_duplicate_property_vertices_use_compact_exact_index_correspondence(
    tmp_path: Path,
) -> None:
    source = _write_dense_duplicate_vertex_mesh(
        tmp_path / "source.usda",
        face_count=2048,
    )
    output = _write_dense_duplicate_vertex_mesh(
        tmp_path / "output.usda",
        face_count=2048,
        dropped_faces={7, 1024},
    )

    correspondence = build_mesh_correspondence(
        source,
        output,
        operation="exact_source_subset",
    )
    evidence_path = write_correspondence_evidence(
        tmp_path / "correspondence.json",
        correspondence,
    )

    assert correspondence.status == "conditional"
    assert correspondence.refusal_reasons == []
    assert correspondence.generated_output_regions == {}
    assert correspondence.changed_source_regions == {"faces": [7, 1024]}
    assert correspondence.vertex_mappings == []
    assert correspondence.corner_mappings == []
    assert correspondence.face_mappings == []
    assert len(correspondence.vertex_mapping_spans) == 1
    assert len(correspondence.face_mapping_spans) == 3
    assert len(correspondence.corner_mapping_spans) == 3
    assert evidence_path.stat().st_size < 10_000


def test_geogram_exact_subset_rewrite_restores_source_order_and_winding() -> None:
    source_vertices = np.asarray(
        [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
        ],
        dtype=np.float64,
    )
    source_faces = np.asarray([(0, 1, 2), (0, 2, 3)], dtype=np.int64)
    candidate_faces = np.asarray([(2, 1, 0), (3, 2, 0)], dtype=np.int64)

    rewrite = _exact_source_subset_rewrite(
        source_vertices,
        source_faces,
        source_vertices,
        candidate_faces[::-1],
    )

    assert rewrite is not None
    assert rewrite.source_face_ids.tolist() == [0, 1]
    assert rewrite.triangles.tolist() == source_faces.tolist()
    assert rewrite.source_corner_ids.tolist() == [[0, 1, 2], [0, 1, 2]]


def test_orchestrator_routes_attributed_small_hole_through_exact_patch_contract(
    tmp_path: Path,
) -> None:
    source = _write_attributed_small_hole(
        tmp_path / "source.usda",
        conflicting_boundary=False,
    )

    result = run_geometry_repair(
        RepairRequest(
            source_path=source,
            output_dir=tmp_path / "repair",
            profile="rigid_pick_place",
            mode="auto",
            enabled_workers=["trimesh_bounded_hole_fill", "coacd_collision"],
            budgets=RepairBudgets(allow_reconstructive=True),
        )
    )

    accepted = next(attempt for attempt in result.attempts if attempt.status == "accepted")
    assert accepted.operation.worker == "trimesh_bounded_hole_fill"
    assert accepted.correspondence_path is not None
    correspondence = build_mesh_correspondence(
        source,
        Path(accepted.output_path or ""),
        operation="trimesh_bounded_hole_fill",
    )
    assert correspondence.status == "conditional"
    assert correspondence.refusal_reasons == []
    assert correspondence.material_mapping_coverage_ratio == 1.0
