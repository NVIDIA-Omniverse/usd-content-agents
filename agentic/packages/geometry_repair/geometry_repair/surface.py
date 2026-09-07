# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic surface sampling, distance, occupancy, and silhouette helpers."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .mesh_io import MeshData, load_meshes, positional_topology_mesh


@dataclass(frozen=True)
class SurfaceSamples:
    """Bounded points and corresponding source-triangle normals."""

    points: np.ndarray
    normals: np.ndarray


@lru_cache(maxsize=8)
def _cached_meshes(
    path_string: str,
    size_bytes: int,
    mtime_ns: int,
    include_guide_purpose: bool,
) -> tuple[MeshData, ...]:
    del size_bytes, mtime_ns
    meshes, _ = load_meshes(
        Path(path_string),
        include_guide_purpose=include_guide_purpose,
    )
    return tuple(meshes)


def _surface_meshes(
    path: Path,
    *,
    include_guide_purpose: bool = False,
) -> tuple[MeshData, ...]:
    resolved = path.resolve()
    stat = resolved.stat()
    return _cached_meshes(
        str(resolved),
        stat.st_size,
        stat.st_mtime_ns,
        include_guide_purpose,
    )


def _valid_triangles(mesh: MeshData) -> np.ndarray:
    if not len(mesh.triangles):
        return np.empty((0, 3), dtype=np.int64)
    return mesh.triangles[
        np.all(
            (mesh.triangles >= 0) & (mesh.triangles < len(mesh.world_vertices_m)),
            axis=1,
        )
    ]


@lru_cache(maxsize=8)
def _cached_trimesh_parts(
    path_string: str,
    size_bytes: int,
    mtime_ns: int,
    include_guide_purpose: bool,
) -> tuple:
    import trimesh

    del size_bytes, mtime_ns
    meshes = _surface_meshes(
        Path(path_string),
        include_guide_purpose=include_guide_purpose,
    )
    parts = []
    for mesh in meshes:
        triangles = _valid_triangles(mesh)
        if not len(triangles) or not np.isfinite(mesh.world_vertices_m).all():
            continue
        vertices, faces = positional_topology_mesh(mesh.world_vertices_m, triangles)
        if len(faces):
            coordinates = vertices[faces]
            areas = np.linalg.norm(
                np.cross(
                    coordinates[:, 1] - coordinates[:, 0],
                    coordinates[:, 2] - coordinates[:, 0],
                ),
                axis=1,
            )
            diagonal = float(np.linalg.norm(np.ptp(vertices, axis=0))) if len(vertices) else 0.0
            valid = np.isfinite(areas) & (areas > max(diagonal * diagonal * 2e-16, 2e-24))
            if np.any(valid):
                parts.append(
                    trimesh.Trimesh(
                        vertices=vertices,
                        faces=faces[valid],
                        process=False,
                    )
                )
    return tuple(parts)


def _trimesh_parts(path: Path, *, include_guide_purpose: bool = False) -> tuple:
    resolved = path.resolve()
    stat = resolved.stat()
    return _cached_trimesh_parts(
        str(resolved),
        stat.st_size,
        stat.st_mtime_ns,
        include_guide_purpose,
    )


@lru_cache(maxsize=8)
def _cached_trimesh_union(
    path_string: str,
    size_bytes: int,
    mtime_ns: int,
    include_guide_purpose: bool,
):
    import trimesh

    parts = _cached_trimesh_parts(
        path_string,
        size_bytes,
        mtime_ns,
        include_guide_purpose,
    )
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else trimesh.util.concatenate(parts)


def deterministic_surface_samples(path: str | Path, limit: int) -> SurfaceSamples:
    """Sample triangle interiors deterministically and retain face normals."""

    meshes = _surface_meshes(Path(path), include_guide_purpose=True)
    barycentric = np.asarray(
        [
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            [0.6, 0.2, 0.2],
            [0.2, 0.6, 0.2],
            [0.2, 0.2, 0.6],
        ],
        dtype=np.float64,
    )
    entries: list[tuple[str, np.ndarray, np.ndarray]] = []
    for mesh in meshes:
        triangles = _valid_triangles(mesh)
        if not len(triangles):
            continue
        coordinates = mesh.world_vertices_m[triangles]
        cross = np.cross(
            coordinates[:, 1] - coordinates[:, 0], coordinates[:, 2] - coordinates[:, 0]
        )
        lengths = np.linalg.norm(cross, axis=1)
        normals = np.zeros_like(cross)
        valid = lengths > 1e-30
        normals[valid] = cross[valid] / lengths[valid, None]
        if np.any(valid):
            entries.append((mesh.path, coordinates[valid], normals[valid]))
    if not entries:
        return SurfaceSamples(
            points=np.empty((0, 3), dtype=np.float64),
            normals=np.empty((0, 3), dtype=np.float64),
        )
    entries.sort(key=lambda item: item[0])
    entry_sample_counts = np.asarray(
        [len(coordinates) * len(barycentric) for _path, coordinates, _normals in entries],
        dtype=np.int64,
    )
    cumulative = np.cumsum(entry_sample_counts)
    total = int(cumulative[-1])
    selected = (
        np.arange(total, dtype=np.int64)
        if total <= limit
        else np.linspace(0, total - 1, num=limit, dtype=np.int64)
    )
    points: list[np.ndarray] = []
    sample_normals: list[np.ndarray] = []
    for global_index in selected:
        entry_index = int(np.searchsorted(cumulative, global_index, side="right"))
        previous = int(cumulative[entry_index - 1]) if entry_index else 0
        local_index = int(global_index) - previous
        face_index, barycentric_index = divmod(local_index, len(barycentric))
        _path, coordinates, normals = entries[entry_index]
        points.append(barycentric[barycentric_index] @ coordinates[face_index])
        sample_normals.append(normals[face_index])
    return SurfaceSamples(
        points=np.asarray(points, dtype=np.float64),
        normals=np.asarray(sample_normals, dtype=np.float64),
    )


def closest_surface(
    path: str | Path,
    points: np.ndarray,
    *,
    include_guide_purpose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return closest points, distances, and target face normals for a mesh union."""

    import trimesh

    query = np.asarray(points, dtype=np.float64).reshape((-1, 3))
    if not len(query):
        empty_points = np.empty((0, 3), dtype=np.float64)
        return empty_points, np.empty((0,), dtype=np.float64), empty_points
    resolved = Path(path).resolve()
    stat = resolved.stat()
    surface = _cached_trimesh_union(
        str(resolved),
        stat.st_size,
        stat.st_mtime_ns,
        include_guide_purpose,
    )
    if surface is None:
        raise ValueError(f"surface query found no valid triangles in {path}")
    nearest_chunks: list[np.ndarray] = []
    distance_chunks: list[np.ndarray] = []
    normal_chunks: list[np.ndarray] = []
    face_normals = np.asarray(surface.face_normals)
    for start in range(0, len(query), 128):
        chunk = query[start : start + 128]
        try:
            nearest, distances, triangle_ids = trimesh.proximity.closest_point(
                surface,
                chunk,
            )
        except (ModuleNotFoundError, ImportError):
            nearest, distances, triangle_ids = trimesh.proximity.closest_point_naive(
                surface,
                chunk,
            )
        nearest_chunks.append(np.asarray(nearest, dtype=np.float64))
        distance_chunks.append(np.asarray(distances, dtype=np.float64))
        normal_chunks.append(face_normals[np.asarray(triangle_ids, dtype=np.int64)])
    return (
        np.concatenate(nearest_chunks, axis=0),
        np.concatenate(distance_chunks, axis=0),
        np.concatenate(normal_chunks, axis=0),
    )


def points_inside_union(
    path: str | Path,
    points: np.ndarray,
    *,
    include_guide_purpose: bool = True,
) -> tuple[np.ndarray | None, str | None]:
    """Classify points against a watertight mesh union, or return why it is unavailable."""

    query = np.asarray(points, dtype=np.float64).reshape((-1, 3))
    parts = _trimesh_parts(Path(path).resolve(), include_guide_purpose=include_guide_purpose)
    if not parts:
        return None, "no valid mesh parts"
    if any(not part.is_watertight for part in parts):
        return None, "occupancy requires watertight mesh parts"
    occupied = np.zeros(len(query), dtype=bool)
    try:
        for part in parts:
            occupied |= np.asarray(part.contains(query), dtype=bool)
    except (ModuleNotFoundError, ImportError) as exc:
        return None, f"occupancy backend unavailable: {type(exc).__name__}: {exc}"
    return occupied, None


def oriented_bbox_extents(points: np.ndarray) -> np.ndarray | None:
    """Return stable minimum-volume oriented extents for a bounded point set."""

    values = np.asarray(points, dtype=np.float64).reshape((-1, 3))
    if len(values) < 3 or not np.isfinite(values).all():
        return None
    try:
        import trimesh

        _transform, extents = trimesh.bounds.oriented_bounds(
            values,
            angle_digits=2,
            ordered=True,
        )
        return np.sort(np.asarray(extents, dtype=np.float64))[::-1]
    except Exception:
        centered = values - values.mean(axis=0)
        covariance = centered.T @ centered / max(len(centered) - 1, 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        projected = centered @ eigenvectors[:, order]
        return np.sort(np.ptp(projected, axis=0))[::-1]


def unique_geometry_points(path: str | Path) -> np.ndarray:
    """Return sorted unique finite world-space source vertices."""

    meshes = _surface_meshes(Path(path))
    arrays = [mesh.world_vertices_m for mesh in meshes if len(mesh.world_vertices_m)]
    if not arrays:
        return np.empty((0, 3), dtype=np.float64)
    points = np.concatenate(arrays, axis=0)
    points = points[np.isfinite(points).all(axis=1)]
    return np.unique(points, axis=0)


def _triangle_coordinates(path: Path) -> np.ndarray:
    meshes = _surface_meshes(path)
    coordinates: list[np.ndarray] = []
    for mesh in meshes:
        triangles = _valid_triangles(mesh)
        if len(triangles):
            coordinates.append(mesh.world_vertices_m[triangles])
    if not coordinates:
        return np.empty((0, 3, 3), dtype=np.float64)
    return np.concatenate(coordinates, axis=0)


def silhouette_iou(
    source_path: str | Path,
    candidate_path: str | Path,
    *,
    resolution: int = 256,
) -> dict[str, float]:
    """Rasterize six deterministic orthographic silhouettes and compare IoU."""

    from PIL import Image, ImageDraw

    source = Path(source_path).resolve()
    candidate = Path(candidate_path).resolve()
    views: dict[str, tuple[int, int]] = {
        "front": (0, 2),
        "back": (0, 2),
        "left": (1, 2),
        "right": (1, 2),
        "top": (0, 1),
        "bottom": (0, 1),
    }
    source_coordinates = _triangle_coordinates(source)
    candidate_coordinates = _triangle_coordinates(candidate)
    projection_cache: dict[tuple[int, int], tuple[list[np.ndarray], list[np.ndarray]]] = {}
    iou_cache: dict[tuple[int, int], float] = {}
    output: dict[str, float] = {}
    for name, axes in views.items():
        if axes in iou_cache:
            output[name] = iou_cache[axes]
            continue
        if axes not in projection_cache:
            projection_cache[axes] = (
                list(source_coordinates[:, :, axes]),
                list(candidate_coordinates[:, :, axes]),
            )
        source_triangles, candidate_triangles = projection_cache[axes]
        all_triangles = source_triangles + candidate_triangles
        if not all_triangles:
            output[name] = 1.0
            continue
        all_points = np.concatenate(all_triangles, axis=0)
        minimum = all_points.min(axis=0)
        maximum = all_points.max(axis=0)
        span = np.maximum(maximum - minimum, 1e-12)
        padding = 4.0
        scale = (resolution - 2.0 * padding) / float(np.max(span))
        offset = (np.asarray([resolution, resolution], dtype=np.float64) - span * scale) * 0.5

        def rasterize(
            triangles: list[np.ndarray],
            *,
            view_minimum: np.ndarray = minimum,
            view_scale: float = scale,
            view_offset: np.ndarray = offset,
        ) -> np.ndarray:
            image = Image.new("1", (resolution, resolution), 0)
            draw = ImageDraw.Draw(image)
            for triangle in triangles:
                pixels = (triangle - view_minimum) * view_scale + view_offset
                pixels[:, 1] = resolution - 1 - pixels[:, 1]
                stable_pixels = np.rint(pixels).astype(np.int64)
                draw.polygon(
                    [tuple(int(value) for value in point) for point in stable_pixels],
                    fill=1,
                )
            return np.asarray(image, dtype=bool)

        left = rasterize(source_triangles)
        right = rasterize(candidate_triangles)
        union = int(np.count_nonzero(left | right))
        intersection = int(np.count_nonzero(left & right))
        value = 1.0 if union == 0 else intersection / union
        output[name] = value
        iou_cache[axes] = value
    return output


def occupancy_error(
    source_path: str | Path,
    candidate_path: str | Path,
    *,
    resolution: int = 24,
) -> tuple[dict[str, float] | None, str | None]:
    """Estimate union-volume excess and deficit on a deterministic common grid."""

    source_meshes, _ = load_meshes(Path(source_path).resolve(), include_guide_purpose=True)
    candidate_meshes, _ = load_meshes(Path(candidate_path).resolve(), include_guide_purpose=True)
    vertices = [
        mesh.world_vertices_m
        for mesh in [*source_meshes, *candidate_meshes]
        if len(mesh.world_vertices_m)
    ]
    if not vertices:
        return None, "occupancy grid has no vertices"
    all_vertices = np.concatenate(vertices, axis=0)
    minimum = all_vertices.min(axis=0)
    maximum = all_vertices.max(axis=0)
    extent = maximum - minimum
    padding = np.maximum(extent * 0.02, 1e-9)
    minimum -= padding
    maximum += padding
    coordinates = [
        np.linspace(minimum[index], maximum[index], num=resolution, endpoint=False)
        + (maximum[index] - minimum[index]) / (2.0 * resolution)
        for index in range(3)
    ]
    grid = np.stack(np.meshgrid(*coordinates, indexing="ij"), axis=-1).reshape((-1, 3))
    source_occupied, source_error = points_inside_union(source_path, grid)
    candidate_occupied, candidate_error = points_inside_union(candidate_path, grid)
    if source_occupied is None or candidate_occupied is None:
        return None, source_error or candidate_error
    source_count = int(np.count_nonzero(source_occupied))
    if source_count == 0:
        return None, "source occupies no grid samples"
    false_positive = int(np.count_nonzero(candidate_occupied & ~source_occupied))
    false_negative = int(np.count_nonzero(source_occupied & ~candidate_occupied))
    candidate_count = int(np.count_nonzero(candidate_occupied))
    return {
        "volume_excess_ratio": max(0, candidate_count - source_count) / source_count,
        "volume_deficit_ratio": max(0, source_count - candidate_count) / source_count,
        "false_positive_ratio": false_positive / source_count,
        "false_negative_ratio": false_negative / source_count,
        "grid_sample_count": float(len(grid)),
    }, None
