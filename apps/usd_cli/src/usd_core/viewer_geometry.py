# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Immutable mesh handoff for the realtime viewer's technical line pass."""

from __future__ import annotations

import hashlib
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LineGeometryCapture:
    """One immutable-in-practice topology capture reusable across style-only edits."""

    arrays: dict[str, np.ndarray]
    metadata: dict[str, Any]


def capture_line_geometry(stage: Any) -> LineGeometryCapture:
    arrays, metadata = _extract_line_geometry(stage)
    return LineGeometryCapture(arrays=arrays, metadata=metadata)


def publish_line_geometry(stage: Any, destination: Path) -> dict[str, Any]:
    """Extract deterministic world-space mesh topology and publish one NPZ.

    The handoff contains numeric arrays only and is written beside the flattened
    viewer snapshot. It never authors the source stage.
    """

    return publish_line_geometry_capture(capture_line_geometry(stage), destination)


def publish_line_geometry_capture(
    capture: LineGeometryCapture,
    destination: Path,
) -> dict[str, Any]:
    """Atomically publish a previously captured numeric topology sidecar."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez(stream, **capture.arrays)
            stream.flush()
            os.fsync(stream.fileno())
        _verify_line_geometry(temporary)
        os.chmod(temporary, 0o600)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

    hasher = hashlib.sha256()
    with destination.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return {
        "path": str(destination),
        "sha256": hasher.hexdigest(),
        "size_bytes": destination.stat().st_size,
        **capture.metadata,
    }


def _extract_line_geometry(stage: Any) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    from pxr import Usd, UsdGeom

    vertices: list[np.ndarray] = []
    triangles: list[np.ndarray] = []
    feature_edges: list[np.ndarray] = []
    silhouette_edges: list[np.ndarray] = []
    silhouette_normals: list[np.ndarray] = []
    reasons: set[str] = set()
    mesh_count = 0
    unsupported_prim_count = 0
    vertex_offset = 0
    xforms = UsdGeom.XformCache(Usd.TimeCode.Default())
    crease_cosine = math.cos(math.radians(30.0))
    coplanar_cosine = math.cos(math.radians(1.0))

    for prim in Usd.PrimRange(
        stage.GetPseudoRoot(),
        Usd.TraverseInstanceProxies(),
    ):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        imageable = UsdGeom.Imageable(prim)
        if imageable.ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue
        purpose = imageable.ComputePurpose()
        if purpose not in (UsdGeom.Tokens.default_, UsdGeom.Tokens.render):
            continue
        mesh = UsdGeom.Mesh(prim)
        points_value = mesh.GetPointsAttr().Get()
        counts_value = mesh.GetFaceVertexCountsAttr().Get()
        indices_value = mesh.GetFaceVertexIndicesAttr().Get()
        if points_value is None or counts_value is None or indices_value is None:
            unsupported_prim_count += 1
            reasons.add("mesh_topology_unavailable")
            continue

        local_points = np.asarray(points_value, dtype=np.float64)
        counts = np.asarray(counts_value, dtype=np.int64)
        indices = np.asarray(indices_value, dtype=np.int64)
        if (
            local_points.ndim != 2
            or local_points.shape[1] != 3
            or counts.ndim != 1
            or indices.ndim != 1
            or int(counts.sum()) != indices.size
            or not np.isfinite(local_points).all()
        ):
            unsupported_prim_count += 1
            reasons.add("invalid_mesh_topology")
            continue

        # Triangle-soup exports commonly repeat the same point for every face.
        # Weld exact duplicates before transforming/classifying topology: this
        # recovers the authored surface adjacency without inventing a spatial
        # tolerance that could join intentionally separate nearby parts.
        unique_local_points, point_remap = np.unique(
            local_points,
            axis=0,
            return_inverse=True,
        )
        point_remap = np.asarray(point_remap).reshape(-1)
        indices = point_remap[indices]
        transform = xforms.GetLocalToWorldTransform(prim)
        transform_array = np.asarray(
            [
                [float(transform[row][column]) for column in range(4)]
                for row in range(4)
            ],
            dtype=np.float64,
        )
        homogeneous_points = np.concatenate(
            (
                unique_local_points,
                np.ones((len(unique_local_points), 1), dtype=np.float64),
            ),
            axis=1,
        )
        world_points = np.ascontiguousarray(
            (homogeneous_points @ transform_array)[:, :3],
            dtype=np.float32,
        )
        if not np.isfinite(world_points).all():
            unsupported_prim_count += 1
            reasons.add("nonfinite_world_points")
            continue
        bounds_diagonal = float(np.linalg.norm(np.ptp(world_points, axis=0)))
        weld_step = max(bounds_diagonal * 1.0e-6, 1.0e-7)
        quantized = np.rint(
            (world_points.astype(np.float64) - world_points.min(axis=0)) / weld_step
        ).astype(np.int64)
        _, welded_point_indices, weld_remap = np.unique(
            quantized,
            axis=0,
            return_index=True,
            return_inverse=True,
        )
        world_points = np.ascontiguousarray(
            world_points[welded_point_indices],
            dtype=np.float32,
        )
        indices = weld_remap[indices]

        tri, polygonal, invalid_faces = _triangulate_faces(
            counts,
            indices,
            len(world_points),
        )
        if invalid_faces:
            reasons.add("invalid_face_skipped")
        if not len(tri):
            unsupported_prim_count += 1
            reasons.add("mesh_has_no_renderable_triangles")
            continue
        if polygonal:
            reasons.add("polygon_faces_fan_triangulated")
        if mesh.GetSubdivisionSchemeAttr().Get() not in (None, UsdGeom.Tokens.none):
            reasons.add("subdivision_surface_approximated")

        p0 = world_points[tri[:, 0]]
        p1 = world_points[tri[:, 1]]
        p2 = world_points[tri[:, 2]]
        normals = np.cross(p1 - p0, p2 - p0)
        lengths = np.linalg.norm(normals, axis=1)
        valid = lengths > 1.0e-12
        if not np.all(valid):
            reasons.add("degenerate_triangles_skipped")
            tri = tri[valid]
            normals = normals[valid]
            lengths = lengths[valid]
        if not len(tri):
            unsupported_prim_count += 1
            continue
        normals = np.ascontiguousarray(normals / lengths[:, None], dtype=np.float32)

        (
            local_features,
            local_silhouettes,
            local_silhouette_normals,
            has_nonmanifold_edges,
        ) = _classify_edges(
            tri,
            normals,
            crease_cosine=crease_cosine,
            coplanar_cosine=coplanar_cosine,
        )
        if has_nonmanifold_edges:
            reasons.add("nonmanifold_edge_visible_only")

        vertices.append(world_points)
        triangles.append(tri + vertex_offset)
        if len(local_features):
            feature_edges.append(local_features + vertex_offset)
        if len(local_silhouettes):
            silhouette_edges.append(local_silhouettes + vertex_offset)
            silhouette_normals.append(local_silhouette_normals)
        vertex_offset += len(world_points)
        mesh_count += 1

    arrays = {
        "vertices": _concatenate_or_empty(vertices, (0, 3), np.float32),
        "triangles": _concatenate_or_empty(triangles, (0, 3), np.uint32),
        "feature_edges": _concatenate_or_empty(feature_edges, (0, 2), np.uint32),
        "silhouette_edges": _concatenate_or_empty(silhouette_edges, (0, 2), np.uint32),
        "silhouette_normals": _concatenate_or_empty(
            silhouette_normals, (0, 2, 3), np.float32
        ),
    }
    metadata = {
        "mesh_count": mesh_count,
        "triangle_count": len(arrays["triangles"]),
        "feature_edge_count": len(arrays["feature_edges"]),
        "silhouette_edge_count": len(arrays["silhouette_edges"]),
        "unsupported_prim_count": unsupported_prim_count,
        "degradation_reasons": sorted(reasons),
    }
    return arrays, metadata


def _triangulate_faces(
    counts: np.ndarray,
    indices: np.ndarray,
    vertex_count: int,
) -> tuple[np.ndarray, bool, bool]:
    """Triangulate topology, keeping the all-triangle path fully vectorized."""

    if len(counts) and np.all(counts == 3):
        faces = indices.reshape(-1, 3)
        valid = np.all((faces >= 0) & (faces < vertex_count), axis=1)
        return (
            np.ascontiguousarray(faces[valid], dtype=np.uint32),
            False,
            not bool(np.all(valid)),
        )

    triangles: list[tuple[int, int, int]] = []
    cursor = 0
    polygonal = False
    invalid = False
    for count_value in counts:
        count = int(count_value)
        face = indices[cursor : cursor + count]
        cursor += count
        if count < 3 or np.any(face < 0) or np.any(face >= vertex_count):
            invalid = True
            continue
        polygonal |= count != 3
        triangles.extend(
            (int(face[0]), int(face[index]), int(face[index + 1]))
            for index in range(1, count - 1)
        )
    return np.asarray(triangles, dtype=np.uint32).reshape(-1, 3), polygonal, invalid


def _classify_edges(
    triangles: np.ndarray,
    normals: np.ndarray,
    *,
    crease_cosine: float,
    coplanar_cosine: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Return feature/silhouette topology without Python per-edge objects."""

    edges = np.concatenate(
        (
            triangles[:, (0, 1)],
            triangles[:, (1, 2)],
            triangles[:, (2, 0)],
        ),
        axis=0,
    )
    edges.sort(axis=1)
    face_indices = np.tile(np.arange(len(triangles), dtype=np.int64), 3)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    sorted_edges = edges[order]
    sorted_faces = face_indices[order]
    starts = np.concatenate(
        (
            np.asarray([0], dtype=np.int64),
            np.flatnonzero(np.any(sorted_edges[1:] != sorted_edges[:-1], axis=1)) + 1,
        )
    )
    counts = np.diff(np.append(starts, len(sorted_edges)))
    unique_edges = np.ascontiguousarray(sorted_edges[starts], dtype=np.uint32)

    boundary = counts == 1
    pair = counts == 2
    nonmanifold = counts > 2
    feature = boundary | nonmanifold
    silhouette = np.zeros(len(unique_edges), dtype=bool)
    pair_normals = np.empty((int(np.count_nonzero(pair)), 2, 3), dtype=np.float32)
    if np.any(pair):
        pair_starts = starts[pair]
        first = normals[sorted_faces[pair_starts]]
        second = normals[sorted_faces[pair_starts + 1]]
        pair_normals[:, 0] = first
        pair_normals[:, 1] = second
        dot = np.clip(np.sum(first * second, axis=1), -1.0, 1.0)
        feature[pair] = dot <= crease_cosine
        silhouette[pair] = dot <= coplanar_cosine

    return (
        np.ascontiguousarray(unique_edges[feature], dtype=np.uint32),
        np.ascontiguousarray(unique_edges[silhouette], dtype=np.uint32),
        np.ascontiguousarray(pair_normals[silhouette[pair]], dtype=np.float32),
        bool(np.any(nonmanifold)),
    )


def _concatenate_or_empty(
    chunks: list[np.ndarray], shape: tuple[int, ...], dtype: np.dtype[Any]
) -> np.ndarray:
    if not chunks:
        return np.empty(shape, dtype=dtype)
    return np.ascontiguousarray(np.concatenate(chunks, axis=0), dtype=dtype)


def _verify_line_geometry(path: Path) -> None:
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "vertices": ((-1, 3), np.float32),
            "triangles": ((-1, 3), np.uint32),
            "feature_edges": ((-1, 2), np.uint32),
            "silhouette_edges": ((-1, 2), np.uint32),
            "silhouette_normals": ((-1, 2, 3), np.float32),
        }
        for name, (shape, dtype) in required.items():
            value = archive[name]
            if value.dtype != dtype or value.ndim != len(shape):
                raise ValueError(f"invalid viewer line geometry array: {name}")
            if any(
                expected >= 0 and actual != expected
                for actual, expected in zip(value.shape, shape, strict=True)
            ):
                raise ValueError(f"invalid viewer line geometry shape: {name}")
        vertices = archive["vertices"]
        if not np.isfinite(vertices).all():
            raise ValueError("viewer line geometry contains nonfinite vertices")
        vertex_count = len(vertices)
        for name in ("triangles", "feature_edges", "silhouette_edges"):
            indices = archive[name]
            if indices.size and int(indices.max()) >= vertex_count:
                raise ValueError(f"viewer line geometry index escaped vertices: {name}")
        if len(archive["silhouette_edges"]) != len(archive["silhouette_normals"]):
            raise ValueError("viewer silhouette arrays have different lengths")
