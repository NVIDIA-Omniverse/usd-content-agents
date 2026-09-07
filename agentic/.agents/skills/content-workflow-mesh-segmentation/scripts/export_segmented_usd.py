#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export a complete dense label partition as one UsdGeomMesh per segment."""

from __future__ import annotations

import argparse
import colorsys
import json
from pathlib import Path
from typing import Any

import numpy as np
from mesh_geometry import (
    MeshData,
    fragment_atomic_conflicts,
    load_fragment_labels,
    load_labels,
    load_usd,
    sha256_file,
    source_metadata,
    write_json,
)
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-usd", type=Path, required=True)
    parser.add_argument("--target")
    parser.add_argument("--fragment-labels", type=Path, required=True)
    parser.add_argument("--face-labels", type=Path, required=True)
    parser.add_argument("--segments", type=Path, required=True)
    parser.add_argument("--output-usd", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def _safe_name(value: str) -> str:
    name = "".join(
        character if character.isalnum() else "_" for character in value
    ).strip("_")
    if not name:
        name = "segment"
    if name[0].isdigit():
        name = f"segment_{name}"
    return name


def _default_diagnostic_color(segment_id: int) -> list[float]:
    """Return a stable, visually distinct color when none is configured."""

    if segment_id == 0:
        return [0.35, 0.35, 0.35]
    hue = (segment_id * 0.618033988749895) % 1.0
    return [round(value, 6) for value in colorsys.hsv_to_rgb(hue, 0.72, 0.9)]


def _define_material(
    stage: Usd.Stage,
    path: str,
    color: list[float],
) -> UsdShade.Material:
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*map(float, color))
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.68)
    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(
        shader.ConnectableAPI(),
        "surface",
    )
    return material


def _author_segment(
    mesh: UsdGeom.Mesh,
    data: MeshData,
    source_faces: np.ndarray,
) -> int:
    selected = data.triangles[source_faces]
    point_ids, inverse = np.unique(selected.reshape(-1), return_inverse=True)
    points = data.points[point_ids]
    triangles = inverse.reshape(-1, 3).astype(np.int32)
    normals = data.normals[source_faces]
    mesh.CreatePointsAttr().Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(points)))
    mesh.CreateFaceVertexCountsAttr().Set(
        Vt.IntArray.FromNumpy(np.full(len(triangles), 3, dtype=np.int32))
    )
    mesh.CreateFaceVertexIndicesAttr().Set(
        Vt.IntArray.FromNumpy(np.ascontiguousarray(triangles.reshape(-1)))
    )
    mesh.CreateNormalsAttr().Set(
        Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(np.repeat(normals, 3, axis=0)))
    )
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
    mesh.CreateOrientationAttr().Set(data.orientation)
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr().Set(data.double_sided)
    imageable = UsdGeom.Imageable(mesh.GetPrim())
    imageable.CreateVisibilityAttr().Set(data.visibility)
    imageable.CreatePurposeAttr().Set(data.purpose)
    mesh.CreateExtentAttr().Set(
        Vt.Vec3fArray(
            [
                Gf.Vec3f(*points.min(axis=0).astype(float).tolist()),
                Gf.Vec3f(*points.max(axis=0).astype(float).tolist()),
            ]
        )
    )
    mesh.GetPrim().CreateAttribute(
        "meshSegmentation:sourceFaceIds",
        Sdf.ValueTypeNames.UIntArray,
    ).Set(Vt.UIntArray.FromNumpy(source_faces.astype(np.uint32)))
    return int(len(point_ids))


def _load_segments(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("segments")
    if not isinstance(records, list) or not records:
        raise ValueError("Segments JSON must contain a nonempty segments list")
    result = []
    ids: set[int] = set()
    names: set[str] = set()
    for index, raw in enumerate(records):
        if not isinstance(raw, dict):
            raise ValueError(f"Segment {index} is not an object")
        segment_id = int(raw["segment_id"])
        name = _safe_name(str(raw["name"]))
        if segment_id in ids:
            raise ValueError(f"Duplicate segment ID: {segment_id}")
        if name in names:
            raise ValueError(f"Duplicate sanitized segment name: {name}")
        color = [
            float(value)
            for value in raw.get("color", _default_diagnostic_color(segment_id))
        ]
        if len(color) != 3 or any(value < 0.0 or value > 1.0 for value in color):
            raise ValueError(f"Invalid diagnostic color for segment {segment_id}")
        ids.add(segment_id)
        names.add(name)
        result.append(
            {**raw, "segment_id": segment_id, "safe_name": name, "color": color}
        )
    return result


def main() -> None:
    args = parse_args()
    output_usd = args.output_usd.resolve()
    manifest_path = args.manifest.resolve()
    if output_usd.exists():
        raise FileExistsError(f"Refusing to overwrite output USD: {output_usd}")
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite manifest: {manifest_path}")
    output_usd.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    data = load_usd(args.source_usd, args.target)
    fragment_labels = load_fragment_labels(args.fragment_labels, data.face_count)
    labels = load_labels(args.face_labels, data.face_count)
    fragment_conflicts = fragment_atomic_conflicts(fragment_labels, labels)
    if len(fragment_conflicts):
        raise ValueError(
            "Final labels split immutable fragments: "
            f"{fragment_conflicts.astype(int).tolist()}"
        )
    degenerate_nonbackground = data.degenerate_face_ids[
        labels[data.degenerate_face_ids] != 0
    ]
    if len(degenerate_nonbackground):
        raise ValueError(
            "Degenerate source faces must remain in segment 0 (other): "
            f"{degenerate_nonbackground.astype(int).tolist()}"
        )
    segments_path = args.segments.resolve()
    segments = _load_segments(segments_path)
    configured_ids = {int(record["segment_id"]) for record in segments}
    observed_ids = {int(value) for value in np.unique(labels)}
    missing_config = observed_ids.difference(configured_ids)
    empty_config = configured_ids.difference(observed_ids)
    if missing_config:
        raise ValueError(f"Labels contain unconfigured segment IDs: {missing_config}")
    if empty_config:
        raise ValueError(f"Configured segments have no faces: {empty_config}")

    stage = Usd.Stage.CreateNew(str(output_usd))
    UsdGeom.SetStageUpAxis(
        stage,
        UsdGeom.Tokens.y if data.up_axis.lower() == "y" else UsdGeom.Tokens.z,
    )
    UsdGeom.SetStageMetersPerUnit(stage, data.meters_per_unit)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.Xform.Define(stage, "/World/SegmentedAsset")
    UsdGeom.Scope.Define(stage, "/World/SegmentedAsset/Segments")
    UsdGeom.Scope.Define(stage, "/World/SegmentedAsset/Looks")
    output_records = []
    source_face_total = 0
    for segment in segments:
        segment_id = int(segment["segment_id"])
        source_faces = np.flatnonzero(labels == segment_id).astype(np.uint32)
        source_face_total += len(source_faces)
        name = str(segment["safe_name"])
        prim_path = f"/World/SegmentedAsset/Segments/{name}"
        mesh = UsdGeom.Mesh.Define(stage, prim_path)
        point_count = _author_segment(mesh, data, source_faces)
        material = _define_material(
            stage,
            f"/World/SegmentedAsset/Looks/{name}",
            list(segment["color"]),
        )
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
        output_records.append(
            {
                "segment_id": segment_id,
                "name": str(segment["name"]),
                "output_prim_path": prim_path,
                "source_face_count": int(len(source_faces)),
                "output_point_count": point_count,
            }
        )
    if source_face_total != data.face_count:
        raise RuntimeError("Exported segment meshes do not exactly cover source faces")
    stage.GetRootLayer().Save()

    manifest = {
        "schema_version": "mesh-segmentation-fragment-export.v1",
        **source_metadata(data),
        "status": "passed",
        "semantic_decision_unit": "immutable_fragment",
        "fragment_labels": str(args.fragment_labels.resolve()),
        "fragment_labels_sha256": sha256_file(args.fragment_labels.resolve()),
        "fragment_count": int(fragment_labels.max()) + 1,
        "fragment_atomicity_conflict_ids": [],
        "face_labels": str(args.face_labels.resolve()),
        "face_labels_sha256": sha256_file(args.face_labels.resolve()),
        "segments": str(segments_path),
        "segments_sha256": sha256_file(segments_path),
        "output_usd": str(output_usd),
        "output_usd_sha256": sha256_file(output_usd),
        "exact_source_face_coverage": source_face_total == data.face_count,
        "output_segments": output_records,
        "limitations": [
            "The geometric exporter preserves source-face provenance and "
            "rebuilds geometry in world space; non-geometric source primvars "
            "and production material bindings are not copied."
        ],
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
