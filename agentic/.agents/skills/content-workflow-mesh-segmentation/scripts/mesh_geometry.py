#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared immutable-topology helpers for canonical mesh segmentation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt


@dataclass(frozen=True)
class MeshData:
    source_path: Path
    target_path: str
    points: np.ndarray
    triangles: np.ndarray
    normals: np.ndarray
    centroids: np.ndarray
    areas: np.ndarray
    valid_faces: np.ndarray
    face_adjacency: np.ndarray
    component_ids: np.ndarray
    component_sizes: np.ndarray
    topology_digest: str
    up_axis: str
    meters_per_unit: float
    orientation: str
    double_sided: bool
    visibility: str
    purpose: str

    @property
    def face_count(self) -> int:
        return int(len(self.triangles))

    @property
    def degenerate_face_ids(self) -> np.ndarray:
        return np.flatnonzero(~self.valid_faces)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Exactly the schemas that carried the `-v5-` infix before the rename. Deriving
# the legacy spelling mechanically also accepted `mesh-segmentation-v5-fragment-
# evidence.v1` and friends -- versions no producer ever emitted -- which widens
# the compat surface this shim promises to remove later.
RENAMED_SCHEMA_STEMS = frozenset(
    {
        "consistency-regions",
        "consistency-sheet",
        "falsification-plan",
        "falsification-review",
        "falsification-validation",
        "initializer-decision",
        "initializer-decision-validation",
        "initializer-runtime-gate",
        "revision-consistency-review",
        "selected-only-stage",
    }
)


def schema_matches(value: object, expected: str) -> bool:
    """Accept a schema version, tolerating the retired `-v5-` experiment infix.

    The launcher accepts both spellings so a run recorded before the rename
    stays readable. These scripts compared exactly, so the same artifact was
    readable by the launcher and rejected by the tooling that re-validates it.

    Only schemas that were actually renamed get the tolerance; see
    `RENAMED_SCHEMA_STEMS`.
    """

    if not isinstance(value, str):
        return False
    if value == expected:
        return True
    prefix, separator, remainder = expected.partition("mesh-segmentation-")
    if not separator:
        return False
    stem = remainder.rsplit(".", 1)[0]
    if stem not in RENAMED_SCHEMA_STEMS:
        return False
    return value == f"{prefix}mesh-segmentation-v5-{remainder}"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _resolve_mesh(stage: Usd.Stage, target: str | None) -> UsdGeom.Mesh:
    if target:
        prim = stage.GetPrimAtPath(target)
        if not prim or not prim.IsA(UsdGeom.Mesh):
            raise ValueError(f"Target is not a UsdGeomMesh: {target}")
        return UsdGeom.Mesh(prim)
    meshes = [UsdGeom.Mesh(prim) for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)]
    if len(meshes) != 1:
        raise ValueError(
            "USD must contain exactly one mesh when --target is omitted; "
            f"found {len(meshes)}"
        )
    return meshes[0]


def _world_points(mesh: UsdGeom.Mesh, stage: Usd.Stage) -> np.ndarray:
    raw = mesh.GetPointsAttr().Get()
    if raw is None:
        raise ValueError("Mesh has no points")
    matrix = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(
        mesh.GetPrim()
    )
    points = [
        matrix.Transform(Gf.Vec3d(float(point[0]), float(point[1]), float(point[2])))
        for point in raw
    ]
    return np.asarray(points, dtype=np.float32)


def _topology_digest(points: np.ndarray, triangles: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(points.astype("<f4", copy=False).tobytes())
    digest.update(triangles.astype("<i4", copy=False).tobytes())
    return f"sha256:{digest.hexdigest()}"


def load_usd(source: Path, target: str | None = None) -> MeshData:
    source = source.resolve()
    stage = Usd.Stage.Open(str(source))
    if stage is None:
        raise ValueError(f"Could not open USD: {source}")
    mesh = _resolve_mesh(stage, target)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int32)
    if not len(counts) or np.any(counts != 3):
        raise ValueError("Mesh segmentation requires a nonempty triangular mesh")
    indices = np.asarray(
        mesh.GetFaceVertexIndicesAttr().Get(),
        dtype=np.int32,
    )
    triangles = indices.reshape(-1, 3)
    points = _world_points(mesh, stage)
    vertices = points[triangles]
    cross = np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0])
    lengths = np.linalg.norm(cross, axis=1)
    valid_faces = lengths > 1.0e-12
    normals = np.zeros_like(cross, dtype=np.float32)
    normals[valid_faces] = (cross[valid_faces] / lengths[valid_faces, None]).astype(
        np.float32
    )
    centroids = vertices.mean(axis=1).astype(np.float32)
    areas = (0.5 * lengths).astype(np.float32)

    topology = trimesh.Trimesh(vertices=points, faces=triangles, process=False)
    adjacency = np.asarray(topology.face_adjacency, dtype=np.int64)
    adjacency = adjacency[valid_faces[adjacency[:, 0]] & valid_faces[adjacency[:, 1]]]
    components = trimesh.graph.connected_components(
        adjacency,
        nodes=np.arange(len(triangles), dtype=np.int64),
        min_len=1,
    )
    ordered = sorted(
        (np.asarray(component, dtype=np.int64) for component in components),
        key=lambda values: (-len(values), int(values.min())),
    )
    component_ids = np.full(len(triangles), -1, dtype=np.int32)
    component_sizes = np.zeros(len(triangles), dtype=np.int32)
    for component_id, faces in enumerate(ordered):
        component_ids[faces] = component_id
        component_sizes[faces] = len(faces)
    if np.any(component_ids < 0):
        raise RuntimeError("Failed to assign a topology component to every face")

    return MeshData(
        source_path=source,
        target_path=str(mesh.GetPath()),
        points=points,
        triangles=triangles,
        normals=normals,
        centroids=centroids,
        areas=areas,
        valid_faces=valid_faces,
        face_adjacency=adjacency,
        component_ids=component_ids,
        component_sizes=component_sizes,
        topology_digest=_topology_digest(points, triangles),
        up_axis=str(UsdGeom.GetStageUpAxis(stage)),
        meters_per_unit=float(UsdGeom.GetStageMetersPerUnit(stage)),
        orientation=str(mesh.GetOrientationAttr().Get()),
        double_sided=bool(mesh.GetDoubleSidedAttr().Get()),
        visibility=str(UsdGeom.Imageable(mesh.GetPrim()).ComputeVisibility()),
        purpose=str(UsdGeom.Imageable(mesh.GetPrim()).ComputePurpose()),
    )


def adjacency_lists(data: MeshData) -> list[np.ndarray]:
    neighbors: list[list[int]] = [[] for _ in range(data.face_count)]
    for first, second in data.face_adjacency:
        neighbors[int(first)].append(int(second))
        neighbors[int(second)].append(int(first))
    return [np.asarray(sorted(values), dtype=np.int64) for values in neighbors]


def source_metadata(data: MeshData) -> dict[str, Any]:
    return {
        "source_asset": str(data.source_path),
        "source_sha256": sha256_file(data.source_path),
        "target_prim_path": data.target_path,
        "source_face_count": data.face_count,
        "topology_digest": data.topology_digest,
        "up_axis": data.up_axis,
        "meters_per_unit": data.meters_per_unit,
        "render_semantics": {
            "orientation": data.orientation,
            "double_sided": data.double_sided,
            "visibility": data.visibility,
            "purpose": data.purpose,
        },
        "bounds_min": data.points.min(axis=0).astype(float).tolist(),
        "bounds_max": data.points.max(axis=0).astype(float).tolist(),
        "degenerate_face_count": int(len(data.degenerate_face_ids)),
        "degenerate_face_ids": [
            int(value) for value in data.degenerate_face_ids[:1024]
        ],
        "degenerate_face_ids_truncated": len(data.degenerate_face_ids) > 1024,
    }


def _create_stage(path: Path, data: MeshData) -> Usd.Stage:
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(
        stage,
        UsdGeom.Tokens.y if data.up_axis.lower() == "y" else UsdGeom.Tokens.z,
    )
    UsdGeom.SetStageMetersPerUnit(stage, data.meters_per_unit)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    return stage


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
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.72)
    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _author_source_mesh(stage: Usd.Stage, data: MeshData) -> UsdGeom.Mesh:
    mesh = UsdGeom.Mesh.Define(stage, "/World/FusedMesh")
    mesh.CreatePointsAttr().Set(
        Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(data.points))
    )
    mesh.CreateFaceVertexCountsAttr().Set(
        Vt.IntArray.FromNumpy(np.full(data.face_count, 3, dtype=np.int32))
    )
    mesh.CreateFaceVertexIndicesAttr().Set(
        Vt.IntArray.FromNumpy(
            np.ascontiguousarray(data.triangles.reshape(-1).astype(np.int32))
        )
    )
    mesh.CreateNormalsAttr().Set(
        Vt.Vec3fArray.FromNumpy(
            np.ascontiguousarray(np.repeat(data.normals, 3, axis=0))
        )
    )
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr().Set(True)
    minimum = data.points.min(axis=0).astype(float).tolist()
    maximum = data.points.max(axis=0).astype(float).tolist()
    mesh.CreateExtentAttr().Set(
        Vt.Vec3fArray(
            [
                Gf.Vec3f(*minimum),
                Gf.Vec3f(*maximum),
            ]
        )
    )
    return mesh


def write_neutral_usd(path: Path, data: MeshData) -> None:
    stage = _create_stage(path, data)
    mesh = _author_source_mesh(stage, data)
    material = _define_material(stage, "/World/Looks/Neutral", [0.43, 0.46, 0.48])
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    stage.GetRootLayer().Save()


def write_diagnostic_usd(
    path: Path,
    data: MeshData,
    labels: np.ndarray,
    *,
    active_segment_id: int = 1,
) -> None:
    if len(labels) != data.face_count:
        raise ValueError("Face-label count does not match source topology")
    stage = _create_stage(path, data)
    mesh = _author_source_mesh(stage, data)
    palette = {
        0: [0.34, 0.38, 0.42],
        active_segment_id: [0.95, 0.08, 0.62],
    }
    extra_colors = (
        [0.08, 0.82, 0.98],
        [1.0, 0.72, 0.05],
        [0.55, 1.0, 0.18],
        [0.72, 0.28, 1.0],
    )
    for index, segment_id in enumerate(
        sorted(int(value) for value in np.unique(labels))
    ):
        color = palette.get(segment_id, extra_colors[index % len(extra_colors)])
        material = _define_material(
            stage,
            f"/World/Looks/Segment_{segment_id:03d}",
            color,
        )
        subset = UsdGeom.Subset.Define(
            stage,
            f"/World/FusedMesh/segment_{segment_id:03d}",
        )
        subset.CreateElementTypeAttr().Set(UsdGeom.Tokens.face)
        subset.CreateFamilyNameAttr().Set("materialBind")
        subset.CreateIndicesAttr().Set(
            Vt.IntArray.FromNumpy(np.flatnonzero(labels == segment_id).astype(np.int32))
        )
        UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(material)
    UsdGeom.Subset.SetFamilyType(mesh, "materialBind", UsdGeom.Tokens.partition)
    stage.GetRootLayer().Save()


def load_labels(path: Path, face_count: int) -> np.ndarray:
    labels = np.fromfile(path.resolve(), dtype="<u4")
    if len(labels) != face_count:
        raise ValueError(
            f"Expected {face_count} face labels, found {len(labels)} in {path}"
        )
    return labels.astype(np.uint32, copy=False)


def load_fragment_labels(path: Path, face_count: int) -> np.ndarray:
    """Load one dense, zero-based fragment ID for every source face."""
    resolved = path.resolve()
    if resolved.suffix == ".npy":
        raw = np.load(resolved, allow_pickle=False)
    else:
        raw = np.fromfile(resolved, dtype="<u4")
    if raw.ndim != 1 or len(raw) != face_count:
        raise ValueError(
            f"Expected {face_count} fragment labels, found shape {raw.shape} "
            f"in {resolved}"
        )
    if not np.issubdtype(raw.dtype, np.integer):
        raise ValueError("Fragment labels must use an integer dtype")
    values = raw.astype(np.int64, copy=False)
    if np.any(values < 0):
        raise ValueError("Fragment labels must be nonnegative")
    unique = np.unique(values)
    if not len(unique) or not np.array_equal(
        unique,
        np.arange(int(unique[-1]) + 1, dtype=np.int64),
    ):
        raise ValueError("Fragment labels must be dense and zero-based")
    return values.astype(np.uint32, copy=False)


def fragment_atomic_conflicts(
    fragment_labels: np.ndarray,
    semantic_labels: np.ndarray,
) -> np.ndarray:
    """Return fragments whose faces do not all share one semantic label."""
    if fragment_labels.shape != semantic_labels.shape:
        raise ValueError("Fragment and semantic labels must have the same shape")
    fragment_count = int(fragment_labels.max()) + 1
    minimum = np.full(fragment_count, np.iinfo(np.uint32).max, dtype=np.uint32)
    maximum = np.zeros(fragment_count, dtype=np.uint32)
    np.minimum.at(minimum, fragment_labels, semantic_labels)
    np.maximum.at(maximum, fragment_labels, semantic_labels)
    return np.flatnonzero(minimum != maximum).astype(np.int64)


def faces_for_fragments(
    fragment_labels: np.ndarray,
    fragment_ids: np.ndarray,
) -> np.ndarray:
    """Return a dense face mask for a set of immutable fragments."""
    if not len(fragment_ids):
        return np.zeros(len(fragment_labels), dtype=bool)
    return np.isin(fragment_labels, np.asarray(fragment_ids, dtype=np.uint32))


def fragments_for_faces(
    fragment_labels: np.ndarray,
    face_ids: np.ndarray,
) -> np.ndarray:
    """Return sorted unique fragment IDs touched by source faces."""
    if not len(face_ids):
        return np.empty(0, dtype=np.int64)
    return np.unique(fragment_labels[np.asarray(face_ids, dtype=np.int64)]).astype(
        np.int64
    )


def load_evidence(path: Path, face_count: int) -> list[dict[str, Any]]:
    payload = json.loads(path.resolve().read_text(encoding="utf-8"))
    raw_events = payload.get("events")
    if not isinstance(raw_events, list) or not raw_events:
        raise ValueError("Evidence JSON must contain a nonempty events list")
    records: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_events):
        if not isinstance(raw, dict):
            raise ValueError(f"Evidence event {index} is not an object")
        if raw.get("rejection_reason"):
            continue
        if raw.get("probe_passed") is False:
            continue
        if raw.get("near_fragment_boundary") is True:
            continue
        if (
            "near_fragment_boundary" not in raw
            and raw.get("near_triangle_edge") is True
        ):
            continue
        face_id = raw.get("face_id", raw.get("hit_face_id"))
        if face_id is None:
            raise ValueError(f"Evidence event {index} has no face ID")
        face_id = int(face_id)
        if not 0 <= face_id < face_count:
            raise ValueError(f"Evidence face ID is out of range: {face_id}")
        polarity = str(raw.get("polarity", "")).lower()
        if polarity not in {"positive", "negative"}:
            raise ValueError(
                f"Evidence event {index} has unsupported polarity: {polarity}"
            )
        records.append(
            {
                **raw,
                "face_id": face_id,
                "polarity": polarity,
                "instance_id": str(raw.get("instance_id", "default")),
                "view_id": str(raw.get("view_id", raw.get("camera", "unknown"))),
            }
        )
    if not records:
        raise ValueError("No accepted evidence events remain after filtering")
    return records
