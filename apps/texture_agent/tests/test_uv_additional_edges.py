# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pxr import Gf, Sdf, Usd, UsdGeom, Vt

from texture_agent.functions import uv_generation as uv
from texture_agent.functions import uv_robustness as robust
from texture_agent.tasks import prepare_uvs as prepare


def _quad(stage: Usd.Stage, path: str = "/World/Mesh") -> UsdGeom.Mesh:
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.GetPointsAttr().Set(
        [
            Gf.Vec3f(0, 0, 0),
            Gf.Vec3f(1, 0, 0),
            Gf.Vec3f(1, 1, 0),
            Gf.Vec3f(0, 1, 0),
        ]
    )
    mesh.GetFaceVertexCountsAttr().Set([4])
    mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
    return mesh


def _set_st(
    mesh: UsdGeom.Mesh,
    values: list[tuple[float, float]],
    *,
    interpolation: str = "faceVarying",
    indices: list[int] | None = None,
) -> None:
    st = UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
        "st",
        Sdf.ValueTypeNames.TexCoord2fArray,
        interpolation,
    )
    st.Set(Vt.Vec2fArray([Gf.Vec2f(float(u), float(v)) for u, v in values]))
    if indices is not None:
        st.SetIndices(Vt.IntArray(indices))


def test_uv_generation_private_helper_edges() -> None:
    topology = {"point_count": 4, "face_count": 2, "face_vertex_count": 8}
    assert uv._expected_count_for_interpolation("uniform", topology) == 2
    assert uv._expected_count_for_interpolation("mystery", topology) is None
    assert uv._uv_array(None).shape == (0, 2)
    assert uv._uv_array([]).shape == (0, 2)
    assert uv._uv_range(np.empty((0, 2))) is None
    assert uv._indices_are_valid(None, 4) is False
    assert uv._indices_are_valid([], 4) is False
    assert uv._face_varying_compatible({"face_vertex_count": 0}) is False
    assert uv._face_varying_compatible(
        {
            "face_vertex_count": 4,
            "indexed": True,
            "index_count": 4,
            "indices_valid": True,
            "value_count": 2,
        }
    )

    normals = uv._compute_face_normals(
        np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32),
        np.array([0, 1]),
        np.array([2]),
    )
    np.testing.assert_allclose(normals[0], [0, 1, 0])
    degenerate = uv._compute_face_normals(
        np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float32),
        np.array([0, 1, 2]),
        np.array([3]),
    )
    np.testing.assert_allclose(degenerate[0], [0, 1, 0])


def test_uv_generation_inspection_and_mutation_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage = Usd.Stage.CreateInMemory()
    empty = UsdGeom.Mesh.Define(stage, "/World/Empty")
    report = uv.inspect_uvs_for_mesh(empty.GetPrim())
    assert report["status"] == "invalid"
    assert report["issues"] == ["UV_EMPTY_TOPOLOGY"]

    mesh = _quad(stage, "/World/BadValues")
    _set_st(mesh, [], interpolation="faceVarying")
    report = uv.inspect_uvs_for_mesh(mesh.GetPrim())
    assert report["status"] == "missing"
    assert "UV_BAD_VALUE_COUNT" in report["issues"]

    unsupported = _quad(stage, "/World/Unsupported")
    _set_st(unsupported, [(0, 0)] * 4, interpolation="uniform")
    report = uv.inspect_uvs_for_mesh(unsupported.GetPrim())
    assert report["status"] == "invalid"
    assert "UV_BAD_INTERPOLATION" in report["issues"]

    nonfinite = _quad(stage, "/World/NonFinite")
    _set_st(nonfinite, [(0, 0), (float("nan"), 0), (1, 1), (0, 1)])
    normalize_count = uv.normalize_uvs(stage, target_prim_paths=["/World/NonFinite"])
    assert normalize_count == 0

    original_prepare_mutable = uv._prepare_mutable_prim
    monkeypatch.setattr(uv, "_prepare_mutable_prim", lambda *_args: False)
    assert uv.generate_uvs_for_stage(stage, target_prim_paths=["/World/Empty"]) == 0
    assert uv.fix_uv_interpolation(stage, target_prim_paths=["/World/Unsupported"]) == 0
    assert uv.normalize_uvs(stage, target_prim_paths=["/World/BadValues"]) == 0
    monkeypatch.setattr(uv, "_prepare_mutable_prim", original_prepare_mutable)

    empty_uvs_stage = Usd.Stage.CreateInMemory()
    empty_uvs_mesh = _quad(empty_uvs_stage, "/World/EmptyUvs")
    _set_st(empty_uvs_mesh, [], interpolation="faceVarying")
    assert uv.normalize_uvs(empty_uvs_stage, target_prim_paths=["/World/EmptyUvs"]) == 0

    instance_stage = Usd.Stage.CreateInMemory()
    instance_mesh = _quad(instance_stage, "/World/Instanceable")
    instance_mesh.GetPrim().SetInstanceable(True)
    _set_st(instance_mesh, [(2, 0), (3, 0), (3, 1), (2, 1)])
    assert (
        uv.normalize_uvs(
            instance_stage,
            target_prim_paths=["/World/Instanceable"],
        )
        == 1
    )
    assert instance_mesh.GetPrim().IsInstanceable() is False

    class FakeInstanceProxy:
        def IsInstanceProxy(self):
            return True

        def GetPath(self):
            return "/World/Proxy"

    assert uv._prepare_mutable_prim(FakeInstanceProxy(), "test") is False


def test_uv_robustness_private_helper_edges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert robust._as_float_array(None, width=3).shape == (0, 3)
    assert robust._as_float_array([], width=2).shape == (0, 2)
    assert robust._polygon_area_3d(np.array([[0, 0, 0], [1, 0, 0]])) == 0.0
    assert robust._polygon_area_2d(np.array([[0, 0], [1, 0]])) == 0.0

    stage = Usd.Stage.CreateInMemory()
    mesh = _quad(stage)
    assert robust._mesh_world_points(mesh.GetPrim(), np.empty((0, 3))).shape == (0, 3)
    assert (
        robust._expanded_face_vertex_uvs(
            mesh.GetPrim(),
            np.array([0, 1, 2, 3]),
            point_count=4,
        )
        is None
    )

    _set_st(mesh, [], interpolation="faceVarying")
    assert (
        robust._expanded_face_vertex_uvs(
            mesh.GetPrim(),
            np.array([0, 1, 2, 3]),
            point_count=4,
        )
        is None
    )

    stage2 = Usd.Stage.CreateInMemory()
    vertex_mesh = _quad(stage2)
    _set_st(vertex_mesh, [(0, 0), (1, 0)], interpolation="vertex", indices=[0, 1, 0])
    assert (
        robust._expanded_face_vertex_uvs(
            vertex_mesh.GetPrim(),
            np.array([0, 1, 2, 3]),
            point_count=4,
        )
        is None
    )

    class PrimvarsWithMissingIndices:
        class St:
            def IsDefined(self):
                return True

            def Get(self):
                return [(0, 0), (1, 0)]

            def GetInterpolation(self):
                return "faceVarying"

            def IsIndexed(self):
                return True

            def GetIndices(self):
                return None

        def __init__(self, prim):
            pass

        def GetPrimvar(self, name):
            return self.St()

    original_primvars_api = robust.UsdGeom.PrimvarsAPI
    try:
        robust.UsdGeom.PrimvarsAPI = PrimvarsWithMissingIndices
        assert (
            robust._expanded_face_vertex_uvs(
                mesh.GetPrim(),
                np.array([0, 1, 2, 3]),
                point_count=4,
            )
            is None
        )
    finally:
        robust.UsdGeom.PrimvarsAPI = original_primvars_api

    stage3 = Usd.Stage.CreateInMemory()
    indexed_vertex = _quad(stage3)
    _set_st(
        indexed_vertex,
        [(0, 0), (1, 0), (1, 1), (0, 1)],
        interpolation="vertex",
        indices=[0, 1, 2, 3],
    )
    assert robust._expanded_face_vertex_uvs(
        indexed_vertex.GetPrim(),
        np.array([0, 1, 2, 3]),
        point_count=4,
    ).shape == (4, 2)

    stage_bad_index = Usd.Stage.CreateInMemory()
    bad_index = _quad(stage_bad_index)
    _set_st(
        bad_index,
        [(0, 0), (1, 0)],
        interpolation="faceVarying",
        indices=[0, 1, 2, 1],
    )
    assert (
        robust._expanded_face_vertex_uvs(
            bad_index.GetPrim(),
            np.array([0, 1, 2, 3]),
            point_count=4,
        )
        is None
    )

    stage_mismatch = Usd.Stage.CreateInMemory()
    mismatch = _quad(stage_mismatch)
    _set_st(mismatch, [(0, 0), (1, 0)], interpolation="faceVarying")
    assert (
        robust._expanded_face_vertex_uvs(
            mismatch.GetPrim(),
            np.array([0, 1, 2, 3]),
            point_count=4,
        )
        is None
    )

    stage_vertex_indexed = Usd.Stage.CreateInMemory()
    vertex_indexed = _quad(stage_vertex_indexed)
    _set_st(
        vertex_indexed,
        [(0, 0), (1, 0), (1, 1), (0, 1)],
        interpolation="vertex",
        indices=[0, 1, 2, 3],
    )
    np.testing.assert_allclose(
        robust._expanded_face_vertex_uvs(
            vertex_indexed.GetPrim(),
            np.array([0, 1, 2, 3]),
            point_count=4,
        ),
        np.array([(0, 0), (1, 0), (1, 1), (0, 1)], dtype=np.float32),
    )

    stage_vertex_point_indices = Usd.Stage.CreateInMemory()
    vertex_point_indices = UsdGeom.Mesh.Define(
        stage_vertex_point_indices, "/World/TwoTri"
    )
    vertex_point_indices.GetPointsAttr().Set(
        [
            Gf.Vec3f(0, 0, 0),
            Gf.Vec3f(1, 0, 0),
            Gf.Vec3f(1, 1, 0),
            Gf.Vec3f(0, 1, 0),
        ]
    )
    vertex_point_indices.GetFaceVertexCountsAttr().Set([3, 3])
    vertex_point_indices.GetFaceVertexIndicesAttr().Set([0, 1, 2, 0, 2, 3])
    _set_st(
        vertex_point_indices,
        [(0, 0), (1, 0), (1, 1), (0, 1)],
        interpolation="vertex",
        indices=[0, 1, 2, 3],
    )
    assert robust._expanded_face_vertex_uvs(
        vertex_point_indices.GetPrim(),
        np.array([0, 1, 2, 0, 2, 3]),
        point_count=4,
    ).shape == (6, 2)

    stage_unknown = Usd.Stage.CreateInMemory()
    unknown = _quad(stage_unknown)
    _set_st(unknown, [(0, 0), (1, 0), (1, 1), (0, 1)], interpolation="constant")
    assert (
        robust._expanded_face_vertex_uvs(
            unknown.GetPrim(),
            np.array([0, 1, 2, 3]),
            point_count=4,
        )
        is None
    )

    stage4 = Usd.Stage.CreateInMemory()
    zero_world = UsdGeom.Mesh.Define(stage4, "/World/Zero")
    zero_world.GetPointsAttr().Set([Gf.Vec3f(0, 0, 0)] * 3)
    zero_world.GetFaceVertexCountsAttr().Set([3])
    zero_world.GetFaceVertexIndicesAttr().Set([0, 1, 2])
    _set_st(zero_world, [(0, 0), (1, 0), (0, 1)])
    metrics = robust.measure_uv_robustness_for_mesh(zero_world.GetPrim())
    assert metrics["zero_area_world_faces"] == 1


def test_prepare_uvs_private_policy_and_target_edges(tmp_path: Path) -> None:
    assert prepare._as_string_list("one") == ["one"]
    assert prepare._as_string_list(7) == ["7"]
    targets = prepare._collect_uv_target_prim_paths(
        {
            "texture_config": {"uv_prim_paths": {" /World/A/ "}},
            "material_textures": {
                "bad": "not-a-dict",
                "mat": {
                    "prim_paths": ["/World/B"],
                    "prim_path": "/World/C",
                    "per_prim": {"/World/D": {}, " ": {}},
                },
            },
        }
    )
    assert targets == ("/World/A", "/World/B", "/World/C", "/World/D")

    report = {
        "meshes": [
            {"prim_path": "/Skip", "status": "invalid"},
            {
                "prim_path": "/World/A",
                "status": "invalid",
                "diagnostics": [
                    {
                        "severity": "error",
                        "code": "UV_BAD",
                        "prim_path": "/World/A",
                        "recommended_action": "fix it",
                    }
                ],
            },
            {
                "prim_path": "/World/B",
                "status": "repairable",
                "recommended_action": "repair it",
            },
        ]
    }
    assert prepare._preflight_policy_errors(
        report,
        prepare.UVPreparePolicy.PRESERVE_OR_FIX,
        target_prim_paths=("/World/A",),
    ) == ["UV_BAD at /World/A: fix it"]
    assert prepare._preflight_policy_errors(
        report,
        prepare.UVPreparePolicy.GENERATE_MISSING,
        target_prim_paths=("/World/A",),
    ) == ["UV_BAD at /World/A: fix it"]
    assert prepare._preflight_policy_errors(
        report,
        prepare.UVPreparePolicy.VALIDATE,
        target_prim_paths=("/World/B",),
    ) == ["UV_NOT_READY at /World/B: repair it"]

    usd_path = tmp_path / "uvs.usda"
    stage = Usd.Stage.CreateNew(str(usd_path))
    mesh = _quad(stage)
    _set_st(mesh, [(2, 0), (3, 0), (3, 1), (2, 1)])
    stage.GetRootLayer().Save()

    prepared_path, actions = prepare._prepare_with_python_uvs(
        str(usd_path),
        tmp_path,
        uv_mode=prepare.UVProjectionMode.BOX,
        policy=prepare.UVPreparePolicy.PRESERVE_OR_FIX,
        normalize_out_of_range=True,
    )

    assert Path(prepared_path).is_file()
    assert actions["normalized"] == 1

    with pytest.raises(ValueError, match="Invalid Scene Optimizer UV projection"):
        prepare.PrepareUVsTask().run(
            {
                "usd_path": str(usd_path),
                "working_dir": str(tmp_path),
                "texture_config": {
                    "uv_backend": "scene_optimizer",
                    "uv_policy": "force_projection",
                    "uv_projection": "not-a-projection",
                },
            }
        )


def test_uv_robustness_manifest_filter_and_failure_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.usda"
    stage = Usd.Stage.CreateNew(str(source))
    _quad(stage)
    stage.GetRootLayer().Save()

    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        f"""
schema_version: {robust.UV_ROBUSTNESS_MANIFEST_SCHEMA_VERSION}
assets:
  - id: ""
    source_usd: missing.usda
  - id: keep
    source_usd: source.usda
modes:
  - id: ""
    texture_config: {{}}
  - id: fail
    texture_config: {{}}
""",
        encoding="utf-8",
    )

    class BrokenTask:
        def run(self, _context):
            raise RuntimeError("prepare failed")

    monkeypatch.setattr(robust, "PrepareUVsTask", lambda: BrokenTask())
    report = robust.evaluate_uv_robustness_manifest(
        manifest,
        tmp_path / "out",
        repo_root=tmp_path,
        asset_ids={"keep"},
        mode_ids={"fail"},
        fail_on_missing_required=False,
    )

    assert len(report["assets"]) == 1
    assert report["assets"][0]["modes"][0]["status"] == "failed"
    assert report["assets"][0]["modes"][0]["error"] == "prepare failed"
    persisted = json.loads((tmp_path / "out" / "uv_robustness_report.json").read_text())
    assert persisted["assets"][0]["id"] == "keep"


def test_uv_robustness_manifest_skips_blank_mode_id(tmp_path: Path) -> None:
    source = tmp_path / "source.usda"
    stage = Usd.Stage.CreateNew(str(source))
    _quad(stage)
    stage.GetRootLayer().Save()

    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        f"""
schema_version: {robust.UV_ROBUSTNESS_MANIFEST_SCHEMA_VERSION}
assets:
  - id: keep
    source_usd: source.usda
modes:
  - id: ""
    texture_config: {{}}
""",
        encoding="utf-8",
    )

    report = robust.evaluate_uv_robustness_manifest(
        manifest,
        tmp_path / "out",
        repo_root=tmp_path,
        fail_on_missing_required=False,
    )

    assert report["assets"][0]["modes"] == []
