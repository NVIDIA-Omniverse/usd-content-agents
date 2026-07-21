# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("pxr")
from pxr import Gf, Sdf, Usd, UsdGeom, Vt  # noqa: E402

from texture_agent.functions.uv_robustness import (  # noqa: E402
    UV_ROBUSTNESS_MANIFEST_SCHEMA_VERSION,
    evaluate_usd_uv_robustness,
    evaluate_uv_robustness_manifest,
)


def _write_uv_quad(
    path: Path,
    *,
    uvs: list[tuple[float, float]] | None = None,
    interpolation: str = "faceVarying",
    indexed: bool = False,
) -> Path:
    stage = Usd.Stage.CreateNew(str(path))
    mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
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
    if uvs is not None:
        st = UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, interpolation
        )
        st.Set(Vt.Vec2fArray([Gf.Vec2f(float(u), float(v)) for u, v in uvs]))
        if indexed:
            st.SetIndices(Vt.IntArray([0, 1, 2, 3]))
    stage.GetRootLayer().Save()
    return path


def _write_two_triangle_seam_usd(path: Path) -> Path:
    stage = Usd.Stage.CreateNew(str(path))
    mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
    mesh.GetPointsAttr().Set(
        [
            Gf.Vec3f(0, 0, 0),
            Gf.Vec3f(1, 0, 0),
            Gf.Vec3f(0, 1, 0),
            Gf.Vec3f(1, 1, 0),
        ]
    )
    mesh.GetFaceVertexCountsAttr().Set([3, 3])
    mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, 1, 3, 2])
    st = UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, "faceVarying"
    )
    st.Set(
        Vt.Vec2fArray(
            [
                Gf.Vec2f(0.0, 0.0),
                Gf.Vec2f(1.0, 0.0),
                Gf.Vec2f(0.0, 1.0),
                Gf.Vec2f(0.2, 0.0),
                Gf.Vec2f(1.0, 1.0),
                Gf.Vec2f(0.2, 1.0),
            ]
        )
    )
    stage.GetRootLayer().Save()
    return path


def _write_scaled_uv_quad(path: Path) -> Path:
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/World")
    root.AddScaleOp().Set(Gf.Vec3f(2.0, 2.0, 2.0))
    mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
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
    st = UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, "faceVarying"
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
    stage.GetRootLayer().Save()
    return path


def test_evaluate_usd_uv_robustness_reports_stretch_metrics(tmp_path: Path) -> None:
    usd_path = _write_uv_quad(
        tmp_path / "quad.usda",
        uvs=[(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)],
    )

    report = evaluate_usd_uv_robustness(
        usd_path,
        asset_id="quad",
        mode_id="source",
    )

    assert report["uv_report"]["summary"]["valid"] == 1
    assert report["robustness_summary"]["measurable_meshes"] == 1
    assert report["robustness_summary"]["zero_area_uv_faces"] == 0
    mesh = report["meshes"][0]
    assert mesh["uv_to_world_area_ratio_median"] == pytest.approx(2.0)
    assert mesh["stretch_p95_over_median"] == pytest.approx(1.0)


def test_evaluate_usd_uv_robustness_uses_world_space_area(
    tmp_path: Path,
) -> None:
    usd_path = _write_scaled_uv_quad(tmp_path / "scaled.usda")

    report = evaluate_usd_uv_robustness(
        usd_path,
        asset_id="scaled",
        mode_id="source",
    )

    mesh = report["meshes"][0]
    assert mesh["world_area"] == pytest.approx(4.0)
    assert mesh["uv_to_world_area_ratio_median"] == pytest.approx(0.25)


def test_evaluate_usd_uv_robustness_reports_zero_area_uv_faces(
    tmp_path: Path,
) -> None:
    usd_path = _write_uv_quad(
        tmp_path / "collapsed.usda",
        uvs=[(0.5, 0.5)] * 4,
    )

    report = evaluate_usd_uv_robustness(
        usd_path,
        asset_id="quad",
        mode_id="source",
    )

    assert report["robustness_summary"]["zero_area_uv_faces"] == 1
    assert report["meshes"][0]["zero_area_uv_faces"] == 1


def test_evaluate_usd_uv_robustness_reports_discontinuous_shared_edges(
    tmp_path: Path,
) -> None:
    usd_path = _write_two_triangle_seam_usd(tmp_path / "seam.usda")

    report = evaluate_usd_uv_robustness(
        usd_path,
        asset_id="seam",
        mode_id="source",
    )

    assert report["robustness_summary"]["shared_edges"] == 1
    assert report["robustness_summary"]["uv_discontinuous_shared_edges"] == 1
    assert report["robustness_summary"]["uv_discontinuous_shared_edge_ratio"] == 1.0


def test_evaluate_usd_uv_robustness_expands_vertex_uvs(tmp_path: Path) -> None:
    usd_path = _write_uv_quad(
        tmp_path / "vertex.usda",
        uvs=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        interpolation="vertex",
    )

    report = evaluate_usd_uv_robustness(
        usd_path,
        asset_id="quad",
        mode_id="source",
    )

    assert report["robustness_summary"]["measurable_meshes"] == 1
    assert report["meshes"][0]["measurable"] is True


def test_evaluate_usd_uv_robustness_expands_indexed_face_varying_uvs(
    tmp_path: Path,
) -> None:
    usd_path = _write_uv_quad(
        tmp_path / "indexed.usda",
        uvs=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        indexed=True,
    )

    report = evaluate_usd_uv_robustness(
        usd_path,
        asset_id="quad",
        mode_id="source",
    )

    assert report["uv_report"]["summary"]["indexed"] == 1
    assert report["robustness_summary"]["measurable_meshes"] == 1


def test_evaluate_usd_uv_robustness_reports_unmeasurable_missing_uvs(
    tmp_path: Path,
) -> None:
    usd_path = _write_uv_quad(tmp_path / "missing.usda")

    report = evaluate_usd_uv_robustness(
        usd_path,
        asset_id="quad",
        mode_id="source",
    )

    assert report["uv_report"]["summary"]["missing"] == 1
    assert report["robustness_summary"]["measurable_meshes"] == 0
    assert report["meshes"][0]["measurable"] is False


def test_evaluate_uv_robustness_manifest_records_missing_optional_asset(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        f"""
schema_version: {UV_ROBUSTNESS_MANIFEST_SCHEMA_VERSION}
assets:
  - id: optional_missing
    required: false
    source_usd: missing.usd
modes:
  - id: validate_only
    texture_config:
      uv_policy: validate
""",
        encoding="utf-8",
    )

    report = evaluate_uv_robustness_manifest(
        manifest,
        tmp_path / "out",
        repo_root=tmp_path,
    )

    assert report["assets"][0]["status"] == "missing"
    report_path = Path(report["report_path"])
    assert report_path.exists()
    persisted = json.loads(report_path.read_text(encoding="utf-8"))
    assert persisted["assets"][0]["id"] == "optional_missing"


def test_evaluate_uv_robustness_manifest_fails_missing_required_asset(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        f"""
schema_version: {UV_ROBUSTNESS_MANIFEST_SCHEMA_VERSION}
assets:
  - id: required_missing
    required: true
    source_usd: missing.usd
modes: []
""",
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="required_missing"):
        evaluate_uv_robustness_manifest(
            manifest,
            tmp_path / "out",
            repo_root=tmp_path,
        )

    persisted = json.loads(
        (tmp_path / "out" / "uv_robustness_report.json").read_text(encoding="utf-8")
    )
    assert persisted["missing_required_assets"] == ["required_missing"]
    assert persisted["assets"][0]["status"] == "missing_required"


def test_evaluate_uv_robustness_manifest_fails_empty_required_source(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        f"""
schema_version: {UV_ROBUSTNESS_MANIFEST_SCHEMA_VERSION}
assets:
  - id: required_empty_source
    required: true
    source_usd: ""
modes: []
""",
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="required_empty_source"):
        evaluate_uv_robustness_manifest(
            manifest,
            tmp_path / "out",
            repo_root=tmp_path,
        )

    persisted = json.loads(
        (tmp_path / "out" / "uv_robustness_report.json").read_text(encoding="utf-8")
    )
    assert persisted["missing_required_assets"] == ["required_empty_source"]
    assert persisted["assets"][0]["status"] == "missing_required"
    assert persisted["assets"][0]["error"] == "source_usd is required"


def test_evaluate_uv_robustness_manifest_treats_directory_source_as_missing(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        f"""
schema_version: {UV_ROBUSTNESS_MANIFEST_SCHEMA_VERSION}
assets:
  - id: optional_directory_source
    required: false
    source_usd: .
modes: []
""",
        encoding="utf-8",
    )

    report = evaluate_uv_robustness_manifest(
        manifest,
        tmp_path / "out",
        repo_root=tmp_path,
    )

    assert report["assets"][0]["status"] == "missing"
    assert report["assets"][0]["error"] == "source_usd must point to a USD file"


def test_evaluate_uv_robustness_manifest_runs_selected_mode(
    tmp_path: Path,
) -> None:
    source = _write_uv_quad(
        tmp_path / "source.usda",
        uvs=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
    )
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        f"""
schema_version: {UV_ROBUSTNESS_MANIFEST_SCHEMA_VERSION}
assets:
  - id: source_quad
    required: true
    source_usd: {source.name}
    target_prim_paths:
      - /World/Mesh
modes:
  - id: validate_only
    texture_config:
      uv_policy: validate
  - id: skipped_mode
    texture_config:
      uv_policy: generate_missing
""",
        encoding="utf-8",
    )

    report = evaluate_uv_robustness_manifest(
        manifest,
        tmp_path / "out",
        repo_root=tmp_path,
        mode_ids={"validate_only"},
    )

    asset = report["assets"][0]
    assert asset["status"] == "evaluated"
    assert asset["source_report"]["uv_report"]["summary"]["valid"] == 1
    assert [mode["id"] for mode in asset["modes"]] == ["validate_only"]
    assert asset["modes"][0]["status"] == "completed"
    assert asset["modes"][0]["texture_config"]["uv_scope"] == "target_prims"
    assert asset["modes"][0]["texture_config"]["uv_target_prim_paths"] == [
        "/World/Mesh"
    ]
    assert asset["modes"][0]["report"]["robustness_summary"]["measurable_meshes"] == 1
