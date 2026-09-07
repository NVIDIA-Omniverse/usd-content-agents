# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom, UsdShade, Vt

SKILL_DIR = (
    Path(__file__).resolve().parents[1]
    / ".agents"
    / "skills"
    / "content-workflow-mesh-segmentation"
)


def _write_three_component_mesh(
    path: Path,
    *,
    include_degenerate: bool = False,
    orientation: str = str(UsdGeom.Tokens.rightHanded),
    double_sided: bool = False,
    visibility: str = str(UsdGeom.Tokens.inherited),
    purpose: str = str(UsdGeom.Tokens.default_),
) -> None:
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [4.0, 1.0, 0.0],
            [3.0, 1.0, 0.0],
            [6.0, 0.0, 0.0],
            [7.0, 0.0, 0.0],
            [7.0, 1.0, 0.0],
            [6.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    triangles = np.asarray(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 5, 6],
            [4, 6, 7],
            [8, 9, 10],
            [8, 10, 11],
        ],
        dtype=np.int32,
    )
    if include_degenerate:
        triangles = np.vstack([triangles, np.asarray([[0, 0, 1]], dtype=np.int32)])
    stage = Usd.Stage.CreateNew(str(path))
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/World/Fused")
    mesh.CreatePointsAttr().Set(Vt.Vec3fArray.FromNumpy(points))
    mesh.CreateFaceVertexCountsAttr().Set(
        Vt.IntArray.FromNumpy(np.full(len(triangles), 3, dtype=np.int32))
    )
    mesh.CreateFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(triangles.reshape(-1)))
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    mesh.CreateOrientationAttr().Set(orientation)
    mesh.CreateDoubleSidedAttr().Set(double_sided)
    imageable = UsdGeom.Imageable(mesh.GetPrim())
    imageable.CreateVisibilityAttr().Set(visibility)
    imageable.CreatePurposeAttr().Set(purpose)
    stage.GetRootLayer().Save()


def _run(
    script: str, *arguments: object, check: bool = True
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(SKILL_DIR / "scripts" / script),
            *[str(argument) for argument in arguments],
        ],
        check=check,
        capture_output=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        text=True,
    )


def _oversegment(
    source: Path,
    output_dir: Path,
    *,
    patch_count: int = 0,
) -> Path:
    _run(
        "oversegment_mesh.py",
        "--source-usd",
        source,
        "--target",
        "/World/Fused",
        "--patch-count",
        patch_count,
        "--output-dir",
        output_dir,
    )
    manifest = json.loads(
        (output_dir / "fragment_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["implementation_language"] == "python"
    assert manifest["classifier_backend"] == "scipy"
    assert manifest["algorithm_reference"]["doi"] == "10.1111/cgf.12486"
    return output_dir / "fragment_ids.npy"


def test_superfacet_classifier_partitions_with_nonzero_patch_count(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.usda"
    _write_three_component_mesh(source)
    output_dir = tmp_path / "fragments"

    fragment_labels = _oversegment(
        source,
        output_dir,
        patch_count=3,
    )

    labels = np.load(fragment_labels)
    manifest = json.loads(
        (output_dir / "fragment_manifest.json").read_text(encoding="utf-8")
    )
    assert np.unique(labels).tolist() == list(range(6))
    assert manifest["requested_patch_count"] == 3
    assert manifest["effective_patch_count"] == 6
    assert manifest["fragment_count"] == 6
    assert manifest["disconnected_islands_split"] == 0


def test_fragment_edit_locking_frontier_and_export(tmp_path: Path) -> None:
    source = tmp_path / "source.usda"
    _write_three_component_mesh(source)

    prepared = tmp_path / "prepared"
    _run(
        "prepare_mesh.py",
        "--source-usd",
        source,
        "--target",
        "/World/Fused",
        "--output-dir",
        prepared,
    )
    topology = json.loads((prepared / "topology.json").read_text(encoding="utf-8"))
    assert topology["source_face_count"] == 6
    assert topology["topology_component_count"] == 3
    assert topology["largest_component_face_counts"] == [2, 2, 2]
    assert (prepared / "neutral.usdc").is_file()
    fragment_labels = _oversegment(source, tmp_path / "fragments")
    assert np.load(fragment_labels).tolist() == [0, 0, 1, 1, 2, 2]

    evidence = tmp_path / "evidence.json"
    evidence.write_text(
        json.dumps(
            {
                "events": [
                    {
                        "face_id": 0,
                        "polarity": "positive",
                        "view_id": "front",
                        "instance_id": "panel-a",
                    },
                    {
                        "face_id": 1,
                        "polarity": "positive",
                        "view_id": "rear",
                        "instance_id": "panel-a",
                    },
                    {
                        "face_id": 2,
                        "polarity": "negative",
                        "view_id": "front",
                        "instance_id": "panel-a",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    parent = tmp_path / "parent.u32le"
    np.asarray([0, 0, 2, 2, 0, 0], dtype="<u4").tofile(parent)
    candidate = tmp_path / "candidate.u32le"
    np.asarray([1, 1, 2, 2, 0, 0], dtype="<u4").tofile(candidate)

    comparison = tmp_path / "comparison.json"
    _run(
        "compare_face_labels.py",
        "--source-usd",
        source,
        "--target",
        "/World/Fused",
        "--fragment-labels",
        fragment_labels,
        "--parent-labels",
        parent,
        "--candidate-labels",
        candidate,
        "--evidence",
        evidence,
        "--active-segment-id",
        1,
        "--output",
        comparison,
    )
    assert json.loads(comparison.read_text(encoding="utf-8"))["status"] == "passed"

    frontier = tmp_path / "frontier"
    _run(
        "audit_selection_frontier.py",
        "--source-usd",
        source,
        "--target",
        "/World/Fused",
        "--fragment-labels",
        fragment_labels,
        "--face-labels",
        candidate,
        "--active-segment-id",
        1,
        "--output-dir",
        frontier,
    )
    frontier_payload = json.loads(
        (frontier / "frontier_audit.json").read_text(encoding="utf-8")
    )
    assert frontier_payload["selected_face_count"] == 2
    assert frontier_payload["selected_component_count"] == 1

    edits = tmp_path / "edits.json"
    edits.write_text(
        json.dumps(
            {
                "edits": [
                    {
                        "operation": "include",
                        "fragment_ids": [2],
                        "reason": "synthetic whole-fragment correction",
                        "evidence_render_ids": ["synthetic-view"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    edited = tmp_path / "edited"
    _run(
        "apply_fragment_edits.py",
        "--source-usd",
        source,
        "--target",
        "/World/Fused",
        "--fragment-labels",
        fragment_labels,
        "--parent-labels",
        candidate,
        "--edits",
        edits,
        "--active-segment-id",
        1,
        "--output-dir",
        edited,
    )
    edited_labels = np.fromfile(edited / "face_labels.u32le", dtype="<u4")
    assert edited_labels.tolist() == [1, 1, 2, 2, 1, 1]

    final_labels = tmp_path / "final.u32le"
    np.asarray([1, 1, 2, 2, 3, 3], dtype="<u4").tofile(final_labels)
    segments = tmp_path / "segments.json"
    segments.write_text(
        json.dumps(
            {
                "segments": [
                    {"segment_id": 1, "name": "panel_a", "color": [1.0, 0.0, 0.0]},
                    {"segment_id": 2, "name": "panel_b", "color": [0.0, 1.0, 0.0]},
                    {"segment_id": 3, "name": "body", "color": [0.0, 0.0, 1.0]},
                ]
            }
        ),
        encoding="utf-8",
    )
    output_usd = tmp_path / "segmented.usdc"
    export_manifest = tmp_path / "export_manifest.json"
    _run(
        "export_segmented_usd.py",
        "--source-usd",
        source,
        "--target",
        "/World/Fused",
        "--fragment-labels",
        fragment_labels,
        "--face-labels",
        final_labels,
        "--segments",
        segments,
        "--output-usd",
        output_usd,
        "--manifest",
        export_manifest,
    )
    exported = Usd.Stage.Open(str(output_usd))
    meshes = [prim for prim in exported.Traverse() if prim.IsA(UsdGeom.Mesh)]
    assert len(meshes) == 3
    assert all(
        len(prim.GetAttribute("meshSegmentation:sourceFaceIds").Get()) == 2
        for prim in meshes
    )
    export_payload = json.loads(export_manifest.read_text(encoding="utf-8"))
    assert export_payload["exact_source_face_coverage"] is True

    selected_usd = tmp_path / "selected.usdc"
    selected_manifest = tmp_path / "selected_manifest.json"
    _run(
        "build_selected_only_stage.py",
        "--source-usd",
        output_usd,
        "--target-prim",
        "/World/SegmentedAsset/Segments/panel_a",
        "--output-usd",
        selected_usd,
        "--manifest",
        selected_manifest,
    )
    selected = Usd.Stage.Open(str(selected_usd))
    segment_meshes = {
        prim.GetName(): prim.IsActive()
        for prim in selected.TraverseAll()
        if prim.IsA(UsdGeom.Mesh)
    }
    assert segment_meshes == {"body": False, "panel_a": True, "panel_b": False}


def test_export_assigns_stable_distinct_colors_when_omitted(tmp_path: Path) -> None:
    source = tmp_path / "source.usda"
    _write_three_component_mesh(source)
    fragment_labels = _oversegment(source, tmp_path / "fragments")
    final_labels = tmp_path / "final.u32le"
    np.asarray([1, 1, 2, 2, 3, 3], dtype="<u4").tofile(final_labels)

    def export(name: str, records: list[dict[str, object]]) -> dict[str, tuple]:
        segments = tmp_path / f"segments-{name}.json"
        segments.write_text(json.dumps({"segments": records}), encoding="utf-8")
        output_usd = tmp_path / f"segmented-{name}.usdc"
        _run(
            "export_segmented_usd.py",
            "--source-usd",
            source,
            "--target",
            "/World/Fused",
            "--fragment-labels",
            fragment_labels,
            "--face-labels",
            final_labels,
            "--segments",
            segments,
            "--output-usd",
            output_usd,
            "--manifest",
            tmp_path / f"export-{name}.json",
        )
        stage = Usd.Stage.Open(str(output_usd))
        return {
            prim.GetParent().GetName(): tuple(
                UsdShade.Shader(prim).GetInput("diffuseColor").Get()
            )
            for prim in stage.Traverse()
            if prim.IsA(UsdShade.Shader)
        }

    records = [
        {"segment_id": 1, "name": "panel_a"},
        {"segment_id": 2, "name": "panel_b"},
        {"segment_id": 3, "name": "body"},
    ]
    first = export("forward", records)
    second = export("reverse", list(reversed(records)))

    assert first == second
    assert len(first) == len(records)
    assert len(set(first.values())) == len(first)
    assert all(color != (0.5, 0.5, 0.5) for color in first.values())


def test_export_preserves_source_mesh_render_semantics(tmp_path: Path) -> None:
    source = tmp_path / "source.usda"
    _write_three_component_mesh(
        source,
        orientation=str(UsdGeom.Tokens.leftHanded),
        double_sided=False,
        visibility=str(UsdGeom.Tokens.inherited),
        purpose=str(UsdGeom.Tokens.render),
    )
    fragment_labels = _oversegment(source, tmp_path / "fragments")
    final_labels = tmp_path / "final.u32le"
    np.asarray([1, 1, 2, 2, 3, 3], dtype="<u4").tofile(final_labels)
    segments = tmp_path / "segments.json"
    segments.write_text(
        json.dumps(
            {
                "segments": [
                    {"segment_id": 1, "name": "panel_a"},
                    {"segment_id": 2, "name": "panel_b"},
                    {"segment_id": 3, "name": "body"},
                ]
            }
        ),
        encoding="utf-8",
    )
    output_usd = tmp_path / "segmented.usdc"
    manifest = tmp_path / "export.json"

    _run(
        "export_segmented_usd.py",
        "--source-usd",
        source,
        "--target",
        "/World/Fused",
        "--fragment-labels",
        fragment_labels,
        "--face-labels",
        final_labels,
        "--segments",
        segments,
        "--output-usd",
        output_usd,
        "--manifest",
        manifest,
    )

    stage = Usd.Stage.Open(str(output_usd))
    meshes = [UsdGeom.Mesh(prim) for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)]
    assert len(meshes) == 3
    for mesh in meshes:
        imageable = UsdGeom.Imageable(mesh.GetPrim())
        assert str(mesh.GetOrientationAttr().Get()) == str(UsdGeom.Tokens.leftHanded)
        assert mesh.GetDoubleSidedAttr().Get() is False
        assert str(imageable.ComputeVisibility()) == str(UsdGeom.Tokens.inherited)
        assert str(imageable.ComputePurpose()) == str(UsdGeom.Tokens.render)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["render_semantics"] == {
        "orientation": str(UsdGeom.Tokens.leftHanded),
        "double_sided": False,
        "visibility": str(UsdGeom.Tokens.inherited),
        "purpose": str(UsdGeom.Tokens.render),
    }


def test_picker_records_closest_hit_face(tmp_path: Path) -> None:
    source = tmp_path / "source.usda"
    _write_three_component_mesh(source)
    fragment_labels = _oversegment(source, tmp_path / "fragments")
    camera = tmp_path / "camera.json"
    camera.write_text(
        json.dumps(
            {
                "image_width": 100,
                "image_height": 100,
                "camera_state": {
                    "horizontal_aperture": 20.0,
                    "focal_length": 50.0,
                },
                "camera_world_transform": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.7, 0.2, 2.0, 1.0],
                ],
            }
        ),
        encoding="utf-8",
    )
    clicks = tmp_path / "clicks.json"
    clicks.write_text(
        json.dumps(
            {
                "events": [
                    {
                        "camera": str(camera),
                        "view_id": "top",
                        "instance_id": "panel-a",
                        "pixel": [49.5, 49.5],
                        "polarity": "positive",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    evidence = tmp_path / "picked_evidence.json"
    _run(
        "pick_face_evidence.py",
        "--source-usd",
        source,
        "--target",
        "/World/Fused",
        "--fragment-labels",
        fragment_labels,
        "--clicks",
        clicks,
        "--output",
        evidence,
    )
    payload = json.loads(evidence.read_text(encoding="utf-8"))
    assert payload["events"][0]["face_id"] == 0
    assert payload["events"][0]["fragment_id"] == 0
    assert payload["events"][0]["probe_passed"] is True
    assert payload["events"][0]["near_triangle_edge"] is False


def test_compare_rejects_locked_label_changes(tmp_path: Path) -> None:
    source = tmp_path / "source.usda"
    _write_three_component_mesh(source)
    fragment_labels = _oversegment(source, tmp_path / "fragments")
    parent = tmp_path / "parent.u32le"
    candidate = tmp_path / "candidate.u32le"
    np.asarray([0, 0, 2, 2, 0, 0], dtype="<u4").tofile(parent)
    np.asarray([1, 0, 1, 2, 0, 0], dtype="<u4").tofile(candidate)
    evidence = tmp_path / "evidence.json"
    evidence.write_text(
        json.dumps(
            {
                "events": [
                    {
                        "face_id": 0,
                        "polarity": "positive",
                        "view_id": "front",
                        "instance_id": "panel-a",
                    },
                    {
                        "face_id": 2,
                        "polarity": "negative",
                        "view_id": "front",
                        "instance_id": "panel-a",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    result = _run(
        "compare_face_labels.py",
        "--source-usd",
        source,
        "--target",
        "/World/Fused",
        "--fragment-labels",
        fragment_labels,
        "--parent-labels",
        parent,
        "--candidate-labels",
        candidate,
        "--evidence",
        evidence,
        "--active-segment-id",
        1,
        "--output",
        tmp_path / "comparison.json",
        check=False,
    )
    assert result.returncode == 2
    assert "candidate changes immutable labels" in result.stdout


def test_preserves_degenerate_source_face_as_unselectable_other(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source_with_degenerate.usda"
    _write_three_component_mesh(source, include_degenerate=True)
    prepared = tmp_path / "prepared"
    _run(
        "prepare_mesh.py",
        "--source-usd",
        source,
        "--target",
        "/World/Fused",
        "--output-dir",
        prepared,
    )
    topology = json.loads((prepared / "topology.json").read_text(encoding="utf-8"))
    assert topology["source_face_count"] == 7
    assert topology["degenerate_face_count"] == 1
    assert topology["degenerate_face_ids"] == [6]
    assert np.fromfile(
        prepared / "degenerate_face_ids.u32le",
        dtype="<u4",
    ).tolist() == [6]
    assert np.fromfile(
        prepared / "all_faces_candidate.u32le",
        dtype="<u4",
    ).tolist() == [1, 1, 1, 1, 1, 1, 0]
