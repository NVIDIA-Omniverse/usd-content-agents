# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression witnesses for source-preserving visual intake, not task acceptance."""

import base64
import json
from types import SimpleNamespace

import numpy as np
import pytest
from content_agent_workflows.geometry.lossless_gltf import (
    import_static_gltf,
    verify_preserved_render,
)
from content_agent_workflows.geometry.workflow import (
    GeometryWorkflowInput,
    _cad_preflight_checks,
)
from geometry_repair.mesh_io import (
    positional_topology_mesh,
    positional_topology_mesh_with_source_indices,
)
from geometry_repair.protected_features import detect_protected_feature_candidates


def source_fixture(tmp_path, *, invalid_index=False):
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype="<f4")
    # The second triangle is intentionally source-degenerate and must survive.
    indices = np.array([0, 1, 2, 0, 3, 1], dtype="<u2")
    if invalid_index:
        indices[-1] = 99
    binary = positions.tobytes() + indices.tobytes()
    doc = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"name": "SourcePanel", "mesh": 0, "translation": [2, 3, 4]}],
        "materials": [{"name": "SourceMaterial"}],
        "meshes": [
            {
                "primitives": [
                    {"attributes": {"POSITION": 0}, "indices": 1, "material": 0}
                ]
            }
        ],
        "buffers": [
            {
                "byteLength": len(binary),
                "uri": "data:application/octet-stream;base64,"
                + base64.b64encode(binary).decode(),
            }
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": positions.nbytes},
            {"buffer": 0, "byteOffset": positions.nbytes, "byteLength": indices.nbytes},
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": len(positions),
                "type": "VEC3",
            },
            {
                "bufferView": 1,
                "componentType": 5123,
                "count": len(indices),
                "type": "SCALAR",
            },
        ],
    }
    path = tmp_path / "source.gltf"
    path.write_text(json.dumps(doc))
    return path, positions, indices


def test_static_import_keeps_exact_source_faces_and_meter_placement(tmp_path):
    from pxr import Usd, UsdGeom

    source, positions, indices = source_fixture(tmp_path)
    original = source.read_bytes()
    out = tmp_path / "source.usdc"
    receipt = import_static_gltf(source, out)
    stage = Usd.Stage.Open(str(out))
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(receipt["parts"][0]["path"]))
    np.testing.assert_array_equal(mesh.GetPointsAttr().Get(), positions)
    np.testing.assert_array_equal(mesh.GetFaceVertexIndicesAttr().Get(), indices)
    assert list(mesh.GetFaceVertexCountsAttr().Get()) == [3, 3]
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1
    assert stage.GetPrimAtPath("/Asset/Looks").GetTypeName() == "Scope"
    np.testing.assert_array_equal(mesh.GetExtentAttr().Get(), [[0, 0, 0], [1, 1, 0]])
    matrix = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(mesh.GetPrim()))
    np.testing.assert_array_equal(matrix[3, :3], [2, 3, 4])
    assert source.read_bytes() == original
    assert verify_preserved_render(out, out)["passed"]


def test_strict_default_still_blocks_open_and_degenerate_render(tmp_path):
    source, *_ = source_fixture(tmp_path)
    out = tmp_path / "source.usdc"
    import_static_gltf(source, out)
    strict = _cad_preflight_checks(
        out, tmp_path, "geometry-agent.static-visual-asset.v1"
    )
    assert strict[1] and any(c.status == "fail" for c in strict[0])
    scoped = _cad_preflight_checks(
        out,
        tmp_path,
        "geometry-agent.static-visual-asset.v1",
        preserved_render_reference=out,
    )
    assert not scoped[1]
    report = json.loads(scoped[-1].read_text())["reports"]["mesh_topology"]
    assert report["strict_topology_report"]["status"] == "fail"
    assert report["physical_acceptance"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        "faces",
        "units",
        "collision",
        "rigid_body",
        "nan",
        "invalid_index",
        "transform",
        "hidden",
        "purpose",
        "primitive",
        "subdivision",
        "animated_points",
        "animated_transform",
        "animated_visibility",
    ],
)
def test_visual_handoff_cannot_hide_source_changes_or_physics(tmp_path, mutation):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    source, *_ = source_fixture(tmp_path)
    ref = tmp_path / "source.usdc"
    receipt = import_static_gltf(source, ref)
    stage = Usd.Stage.Open(str(ref))
    candidate = tmp_path / "changed.usdc"
    stage.GetRootLayer().Export(str(candidate))
    stage = Usd.Stage.Open(str(candidate))
    p = stage.GetPrimAtPath(receipt["parts"][0]["path"])
    mesh = UsdGeom.Mesh(p)
    if mutation == "faces":
        mesh.GetFaceVertexIndicesAttr().Set([0, 2, 1, 0, 3, 1])
    if mutation == "units":
        UsdGeom.SetStageMetersPerUnit(stage, 0.001)
    if mutation == "collision":
        UsdPhysics.CollisionAPI.Apply(p)
    if mutation == "rigid_body":
        UsdPhysics.RigidBodyAPI.Apply(p.GetParent())
    if mutation == "nan":
        mesh.GetPointsAttr().Set([Gf.Vec3f(float("nan"), 0, 0)] * 4)
    if mutation == "invalid_index":
        mesh.GetFaceVertexIndicesAttr().Set([0, 1, 99, 0, 3, 1])
    if mutation == "transform":
        p.GetParent().GetAttribute("xformOp:transform").Set(Gf.Matrix4d(1))
    if mutation == "hidden":
        mesh.GetVisibilityAttr().Set("invisible")
    if mutation == "purpose":
        mesh.GetPurposeAttr().Set("guide")
    if mutation == "primitive":
        UsdGeom.Cube.Define(stage, "/Asset/Substitute")
    if mutation == "subdivision":
        mesh.GetSubdivisionSchemeAttr().Set("catmullClark")
    if mutation == "animated_points":
        mesh.GetPointsAttr().Set([Gf.Vec3f(2, 3, 4)] * 4, Usd.TimeCode(1))
    if mutation == "animated_transform":
        p.GetParent().GetAttribute("xformOp:transform").Set(Gf.Matrix4d(1), Usd.TimeCode(1))
    if mutation == "animated_visibility":
        mesh.GetVisibilityAttr().Set("invisible", Usd.TimeCode(1))
    stage.GetRootLayer().Save()
    with pytest.raises(ValueError):
        verify_preserved_render(ref, candidate)


def test_invalid_gltf_indices_fail_before_handoff(tmp_path):
    source, *_ = source_fixture(tmp_path, invalid_index=True)
    with pytest.raises(ValueError, match="triangle indices"):
        import_static_gltf(source, tmp_path / "bad.usdc")


def test_native_failure_retains_exception_type_and_traceback(tmp_path):
    from content_agent_workflows.geometry.workflow import run_geometry_workflow

    source, *_ = source_fixture(tmp_path, invalid_index=True)
    result = run_geometry_workflow(
        GeometryWorkflowInput(
            source_path=source,
            output_dir=tmp_path / "run",
            source_authoring_mode="lossless_gltf",
            optimization_policy="skip",
        )
    )
    assert not result.success and result.error_type == "ValueError"
    assert "Invalid triangle indices" in result.error
    diagnostic = json.loads(
        __import__("pathlib").Path(result.error_traceback_path).read_text()
    )
    assert (
        diagnostic["error_type"] == "ValueError"
        and "import_static_gltf" in diagnostic["traceback"]
    )


def test_visual_policy_is_explicit_and_cannot_skip_intake_or_mutate(tmp_path):
    assert GeometryWorkflowInput(output_dir=tmp_path).render_topology_policy == "strict"
    for change in [
        {},
        {"source_authoring_mode": "lossless_gltf"},
        {
            "source_authoring_mode": "lossless_gltf",
            "optimization_policy": "skip",
            "repair_mode": "auto",
        },
    ]:
        with pytest.raises(ValueError):
            GeometryWorkflowInput(
                output_dir=tmp_path, render_topology_policy="preserve_source", **change
            )
    request = GeometryWorkflowInput(
        output_dir=tmp_path,
        render_topology_policy="preserve_source",
        source_authoring_mode="lossless_gltf",
        optimization_policy="skip",
    )
    assert request.runtime_validation_mode == "skip"


def test_welded_boundary_uses_source_ids_after_nonidentical_average(
    tmp_path, monkeypatch
):
    import geometry_repair.protected_features as features

    corners = (
        np.array([[0.123, 0, 0], [1.123, 0, 0], [1.123, 1, 0], [0.123, 1, 0]], float)
        * 0.001
    )
    vertices = np.vstack([corners, corners + np.array([1e-13, 0, 0])])
    faces = np.array([[0, 1, 2], [4, 6, 7]], int)
    original = vertices.copy()
    original_faces = faces.copy()
    welded, triangles, source_ids = positional_topology_mesh_with_source_indices(
        vertices, faces
    )
    assert any(
        not np.array_equal(v, vertices[i])
        for v, i in zip(welded, source_ids, strict=True)
    )
    assert set(source_ids) == set(range(4))
    a, b = positional_topology_mesh(vertices, faces)
    np.testing.assert_array_equal(a, welded)
    np.testing.assert_array_equal(b, triangles)
    path = tmp_path / "source.fixture"
    path.write_text("source")
    monkeypatch.setattr(
        features,
        "load_meshes",
        lambda _: (
            [
                SimpleNamespace(
                    role="render",
                    path="/Panel",
                    world_vertices_m=vertices,
                    triangles=faces,
                )
            ],
            {},
        ),
    )
    monkeypatch.setattr(features, "_accessible_void_rays", lambda *_, **__: ([], None))
    report = detect_protected_feature_candidates(path)
    loops = [
        c
        for c in report.candidates
        if c.evidence.get("measurement") == "position_welded_boundary_loop"
    ]
    assert loops and all(
        set(c.evidence["loop_vertex_ids"]) == set(range(4)) for c in loops
    )
    np.testing.assert_array_equal(vertices, original)
    np.testing.assert_array_equal(faces, original_faces)
