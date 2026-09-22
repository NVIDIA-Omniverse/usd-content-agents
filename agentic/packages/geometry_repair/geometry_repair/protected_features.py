# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic protected-feature probes for render and collision geometry."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .artifacts import file_sha256
from .mesh_io import load_meshes, positional_topology_mesh_with_source_indices
from .models import ProtectedFeature, ProtectedFeatureProbe, ProtectedFeatureProbeResult
from .surface import closest_surface, points_inside_union

PROTECTED_FEATURE_CANDIDATES_SCHEMA_VERSION = "geometry-repair.protected-feature-candidates.v1"
PROTECTED_FEATURE_COMPARISON_SCHEMA_VERSION = "geometry-repair.protected-feature-comparison.v1"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProtectedFeatureCandidate(_StrictModel):
    """Measured protection hypothesis that never grants mutation authority."""

    candidate_id: str = Field(min_length=1)
    kind: Literal["opening", "cavity", "handle", "wire", "rim", "tine"]
    scope_path: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    disposition: Literal["protect", "review_only"]
    mutation_authorized: Literal[False] = False
    probe: ProtectedFeatureProbe
    required_clearance_m: float | None = Field(default=None, ge=0.0)
    minimum_thickness_m: float | None = Field(default=None, ge=0.0)
    evidence: dict[str, Any] = Field(default_factory=dict)


class ProtectedFeatureCandidateReport(_StrictModel):
    """Deterministic source evidence for inferred negative space and thin features."""

    schema_version: Literal["geometry-repair.protected-feature-candidates.v1"] = (
        PROTECTED_FEATURE_CANDIDATES_SCHEMA_VERSION
    )
    source_path: str
    source_sha256: str
    confidence_threshold: float = Field(ge=0.0, le=1.0)
    candidates: list[ProtectedFeatureCandidate] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ProtectedFeatureCandidateComparison(_StrictModel):
    """Before/after probe result for one inferred protection candidate."""

    candidate_id: str
    kind: Literal["opening", "cavity", "handle", "wire", "rim", "tine"]
    confidence: float = Field(ge=0.0, le=1.0)
    disposition: Literal["protect", "review_only"]
    mutation_authorized: Literal[False] = False
    status: Literal["preserved", "regressed", "not_evaluated"]
    before: ProtectedFeatureProbeResult
    after: ProtectedFeatureProbeResult
    reasons: list[str] = Field(default_factory=list)


class ProtectedFeatureComparisonReport(_StrictModel):
    """Source-relative feature evidence used to block unsafe topology changes."""

    schema_version: Literal["geometry-repair.protected-feature-comparison.v1"] = (
        PROTECTED_FEATURE_COMPARISON_SCHEMA_VERSION
    )
    source_path: str
    source_sha256: str
    candidate_path: str
    candidate_sha256: str
    status: Literal["pass", "conditional", "fail"]
    comparisons: list[ProtectedFeatureCandidateComparison] = Field(default_factory=list)
    failures: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def _edge_owners(triangles: np.ndarray) -> dict[tuple[int, int], list[int]]:
    owners: dict[tuple[int, int], list[int]] = {}
    for face_index, face in enumerate(np.asarray(triangles, dtype=np.int64)):
        for start, end in zip(face, np.roll(face, -1), strict=True):
            edge = tuple(sorted((int(start), int(end))))
            owners.setdefault(edge, []).append(face_index)
    return owners


def _ordered_boundary_loops(triangles: np.ndarray) -> list[np.ndarray]:
    adjacency: dict[int, set[int]] = {}
    for (left, right), owners in _edge_owners(triangles).items():
        if len(owners) != 1:
            continue
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)
    loops: list[np.ndarray] = []
    visited_edges: set[tuple[int, int]] = set()
    for seed in sorted(adjacency):
        if len(adjacency[seed]) != 2:
            continue
        available = [
            neighbor
            for neighbor in sorted(adjacency[seed])
            if tuple(sorted((seed, neighbor))) not in visited_edges
        ]
        if not available:
            continue
        loop = [seed]
        previous = -1
        current = seed
        next_vertex = available[0]
        while True:
            edge = tuple(sorted((current, next_vertex)))
            if edge in visited_edges:
                break
            visited_edges.add(edge)
            previous, current = current, next_vertex
            if current == seed:
                loops.append(np.asarray(loop, dtype=np.int64))
                break
            loop.append(current)
            candidates = [
                value for value in sorted(adjacency.get(current, set())) if value != previous
            ]
            if len(candidates) != 1:
                break
            next_vertex = candidates[0]
    return sorted(loops, key=lambda item: tuple(int(value) for value in item))


def _face_components(triangles: np.ndarray) -> list[np.ndarray]:
    face_count = len(triangles)
    if not face_count:
        return []
    adjacency: list[set[int]] = [set() for _ in range(face_count)]
    for owners in _edge_owners(triangles).values():
        if len(owners) < 2:
            continue
        for left in owners:
            adjacency[left].update(right for right in owners if right != left)
    components: list[np.ndarray] = []
    visited: set[int] = set()
    for seed in range(face_count):
        if seed in visited:
            continue
        pending = [seed]
        component: list[int] = []
        while pending:
            current = pending.pop()
            if current in visited:
                continue
            visited.add(current)
            component.append(current)
            pending.extend(sorted(adjacency[current] - visited, reverse=True))
        components.append(np.asarray(sorted(component), dtype=np.int64))
    return components


def _stable_axis(vector: np.ndarray) -> np.ndarray:
    axis = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-18:
        return np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    axis = axis / norm
    significant = np.flatnonzero(np.abs(axis) > 1e-12)
    if len(significant) and axis[int(significant[0])] < 0.0:
        axis = -axis
    return axis


def _principal_extents(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = points - points.mean(axis=0)
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    axes = vectors[:, order]
    for index in range(3):
        axes[:, index] = _stable_axis(axes[:, index])
    extents = np.ptp(centered @ axes, axis=0)
    return extents, axes


def _loop_geometry(
    vertices: np.ndarray,
    loop: np.ndarray,
) -> dict[str, Any] | None:
    points = vertices[loop]
    if len(points) < 3:
        return None
    centered = points - points.mean(axis=0)
    _values, vectors = np.linalg.eigh(centered.T @ centered / max(len(points) - 1, 1))
    normal = _stable_axis(vectors[:, 0])
    plane_axes = vectors[:, [2, 1]]
    projected = centered @ plane_axes
    closed = np.vstack((points, points[0]))
    perimeter = float(np.linalg.norm(np.diff(closed, axis=0), axis=1).sum())
    area = abs(
        float(
            0.5
            * np.sum(
                projected[:, 0] * np.roll(projected[:, 1], -1)
                - projected[:, 1] * np.roll(projected[:, 0], -1)
            )
        )
    )
    radius = float(np.sqrt(area / np.pi)) if area > 0.0 else 0.0
    plane_residual = np.abs(centered @ normal)
    planarity = 1.0 - min(
        1.0,
        float(np.max(plane_residual, initial=0.0)) / max(radius, 1e-18),
    )
    circularity = min(1.0, 4.0 * np.pi * area / max(perimeter * perimeter, 1e-30))
    return {
        "points": points,
        "center": points.mean(axis=0),
        "normal": normal,
        "perimeter_m": perimeter,
        "area_m2": area,
        "equivalent_radius_m": radius,
        "planarity": planarity,
        "circularity": circularity,
    }


def _make_candidate(
    *,
    candidate_id: str,
    kind: Literal["opening", "cavity", "handle", "wire", "rim", "tine"],
    scope_path: str,
    confidence: float,
    confidence_threshold: float,
    probe: ProtectedFeatureProbe,
    evidence: dict[str, Any],
    required_clearance_m: float | None = None,
    minimum_thickness_m: float | None = None,
    protect_eligible: bool = True,
) -> ProtectedFeatureCandidate:
    bounded_confidence = min(max(float(confidence), 0.0), 1.0)
    return ProtectedFeatureCandidate(
        candidate_id=candidate_id,
        kind=kind,
        scope_path=scope_path,
        confidence=bounded_confidence,
        disposition=(
            "protect"
            if protect_eligible and bounded_confidence >= confidence_threshold
            else "review_only"
        ),
        mutation_authorized=False,
        probe=probe,
        required_clearance_m=required_clearance_m,
        minimum_thickness_m=minimum_thickness_m,
        evidence=evidence,
    )


def _accessible_void_rays(
    vertices: np.ndarray,
    triangles: np.ndarray,
    *,
    tolerance: float,
) -> tuple[list[dict[str, Any]], str | None]:
    """Find exterior-accessible concavities in a watertight component.

    A vessel mouth is not a topological boundary when the inner and outer walls
    meet at a rim. These bounded rays distinguish such empty space from a convex
    silhouette by requiring points that are inside the convex hull, outside the
    source solid, accessible from one side, and terminated by source solid.
    """

    import trimesh

    surface = trimesh.Trimesh(vertices=vertices, faces=triangles, process=False)
    if not surface.is_watertight or not surface.is_winding_consistent:
        return [], None
    try:
        hull = surface.convex_hull
    except Exception as exc:
        return [], f"convex-hull cavity audit failed: {type(exc).__name__}: {exc}"
    extents, axes = _principal_extents(vertices)
    center = vertices.mean(axis=0)
    projected = (vertices - center) @ axes
    minimum = projected.min(axis=0)
    maximum = projected.max(axis=0)
    results: list[dict[str, Any]] = []
    proximity_failures: list[str] = []
    offsets = np.linspace(-0.25, 0.25, num=5)
    for axis_index in range(3):
        axis_extent = float(extents[axis_index])
        if axis_extent <= tolerance * 8.0:
            continue
        perpendicular = [index for index in range(3) if index != axis_index]
        depths = np.linspace(-axis_extent * 0.02, axis_extent * 1.02, num=32)
        for sign in (-1.0, 1.0):
            edge_coordinate = (
                float(minimum[axis_index]) if sign < 0.0 else float(maximum[axis_index])
            )
            ray_records: list[dict[str, Any]] = []
            for left_offset in offsets:
                for right_offset in offsets:
                    local = np.zeros((len(depths), 3), dtype=np.float64)
                    local[:, axis_index] = edge_coordinate - sign * depths
                    local[:, perpendicular[0]] = left_offset * extents[perpendicular[0]]
                    local[:, perpendicular[1]] = right_offset * extents[perpendicular[1]]
                    points = center + local @ axes.T
                    try:
                        hull_inside = np.asarray(hull.contains(points), dtype=bool)
                        source_inside = np.asarray(surface.contains(points), dtype=bool)
                    except Exception as exc:
                        return [], f"cavity occupancy audit failed: {type(exc).__name__}: {exc}"
                    hull_indices = np.flatnonzero(hull_inside)
                    if not len(hull_indices):
                        continue
                    start = int(hull_indices[0])
                    solid_after = np.flatnonzero(source_inside[start:])
                    if not len(solid_after):
                        # A through-hole or fully open passage is not a cavity.
                        continue
                    end = start + int(solid_after[0])
                    if end - start < 3 or np.any(source_inside[start:end]):
                        continue
                    void_depth = float(depths[end] - depths[start])
                    if void_depth < max(axis_extent * 0.10, tolerance * 4.0):
                        continue
                    probe_end = start + max(1, int((end - start) * 0.65))
                    probe_points = points[start : probe_end + 1]
                    if len(probe_points) < 2:
                        continue
                    try:
                        _nearest, distances, _face_ids = trimesh.proximity.closest_point(
                            surface,
                            probe_points,
                        )
                    except Exception as exc:
                        proximity_failures.append(
                            f"ray clearance query failed: {type(exc).__name__}: {exc}"
                        )
                        continue
                    minimum_clearance = float(np.min(distances)) if len(distances) else 0.0
                    if minimum_clearance <= tolerance:
                        continue
                    ray_records.append(
                        {
                            "depth_m": void_depth,
                            "minimum_clearance_m": minimum_clearance,
                            "probe_points": probe_points,
                            "offsets": [float(left_offset), float(right_offset)],
                        }
                    )
            if not ray_records:
                continue
            ray_records.sort(
                key=lambda item: (
                    -float(item["depth_m"]),
                    -float(item["minimum_clearance_m"]),
                    item["offsets"],
                )
            )
            representative = ray_records[0]
            mouth_center = np.asarray(representative["probe_points"], dtype=np.float64)[0]
            ring_radius = max(
                float(representative["minimum_clearance_m"]),
                tolerance * 2.0,
            )
            angles = np.linspace(0.0, 2.0 * np.pi, num=16, endpoint=False)
            ring_queries = (
                mouth_center[None, :]
                + np.cos(angles)[:, None] * axes[:, perpendicular[0]][None, :] * ring_radius
                + np.sin(angles)[:, None] * axes[:, perpendicular[1]][None, :] * ring_radius
            )
            try:
                rim_points, _distances, _face_ids = trimesh.proximity.closest_point(
                    surface,
                    ring_queries,
                )
            except Exception as exc:
                proximity_failures.append(f"rim query failed: {type(exc).__name__}: {exc}")
                rim_points = np.empty((0, 3), dtype=np.float64)
            results.append(
                {
                    "axis_index": axis_index,
                    "direction_sign": int(sign),
                    "direction": (axes[:, axis_index] * sign).tolist(),
                    "depth_m": representative["depth_m"],
                    "minimum_clearance_m": representative["minimum_clearance_m"],
                    "probe_points": representative["probe_points"],
                    "rim_points": rim_points,
                    "supporting_ray_count": len(ray_records),
                    "sampled_ray_count": len(offsets) ** 2,
                }
            )
    ordered_results = sorted(
        results,
        key=lambda item: (
            int(item["axis_index"]),
            int(item["direction_sign"]),
        ),
    )
    warning = None
    if proximity_failures:
        warning = (
            f"cavity proximity audit skipped {len(proximity_failures)} query set(s); "
            f"results may be incomplete; last error: {proximity_failures[-1]}"
        )
    return ordered_results, warning


def detect_protected_feature_candidates(
    geometry_path: str | Path,
    *,
    confidence_threshold: float = 0.8,
    candidate_limit: int = 256,
) -> ProtectedFeatureCandidateReport:
    """Derive deterministic protection hypotheses without granting repair authority.

    Position-coincident property vertices are welded only in an analysis copy so
    UV/material seams cannot masquerade as physical openings. Every candidate is
    evidence for preservation or review; inferred intent never authorizes mutation.
    """

    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be in [0, 1]")
    if candidate_limit < 1:
        raise ValueError("candidate_limit must be positive")
    source = Path(geometry_path).expanduser().resolve()
    meshes, _metadata = load_meshes(source)
    candidates: list[ProtectedFeatureCandidate] = []
    warnings: list[str] = []
    for mesh in sorted(
        (item for item in meshes if item.role == "render"),
        key=lambda item: item.path,
    ):
        try:
            vertices, triangles, source_vertex_ids = positional_topology_mesh_with_source_indices(
                mesh.world_vertices_m,
                mesh.triangles,
            )
        except ValueError as exc:
            warnings.append(f"{mesh.path}: candidate analysis skipped: {exc}")
            continue
        if not len(vertices) or not len(triangles):
            continue
        diagonal = float(np.linalg.norm(np.ptp(vertices, axis=0)))
        tolerance = max(diagonal * 1e-5, 1e-7)
        object_center = vertices.mean(axis=0)
        loops = _ordered_boundary_loops(triangles)
        for loop_index, loop in enumerate(loops):
            geometry = _loop_geometry(vertices, loop)
            if geometry is None or geometry["equivalent_radius_m"] <= tolerance:
                continue
            center = geometry["center"]
            normal = geometry["normal"]
            radius = float(geometry["equivalent_radius_m"])
            clearance = max(radius * 0.2, tolerance)
            half_span = max(radius * 0.35, tolerance * 2.0)
            opening_points = [
                (center - normal * half_span).tolist(),
                (center + normal * half_span).tolist(),
            ]
            loop_evidence = {
                "measurement": "position_welded_boundary_loop",
                "loop_vertex_ids": [
                    int(source_vertex_ids[int(vertex_id)])
                    for vertex_id in loop
                ],
                "loop_vertex_count": len(loop),
                "perimeter_m": geometry["perimeter_m"],
                "area_m2": geometry["area_m2"],
                "equivalent_radius_m": radius,
                "planarity": geometry["planarity"],
                "circularity": geometry["circularity"],
            }
            # Planarity alone made tiny triangular defect boundaries look like
            # intentional openings with near-certain confidence.  Require
            # enough loop support and a plausible closed profile before an
            # inferred boundary can become hard render-protection evidence.
            loop_support = min(1.0, max(0.0, (len(loop) - 3.0) / 5.0))
            opening_confidence = (
                0.5
                + 0.18 * float(geometry["planarity"])
                + 0.12 * float(geometry["circularity"])
                + 0.1 * loop_support
            )
            rim_confidence = min(
                0.95,
                opening_confidence + (0.05 if len(loop) >= 5 else 0.0),
            )
            candidates.append(
                _make_candidate(
                    candidate_id=f"{mesh.path}:opening:{loop_index:04d}",
                    kind="opening",
                    scope_path=mesh.path,
                    confidence=opening_confidence,
                    confidence_threshold=confidence_threshold,
                    probe=ProtectedFeatureProbe(
                        kind="negative_space_path",
                        points_m=opening_points,
                        radius_m=clearance,
                        tolerance_m=tolerance,
                    ),
                    required_clearance_m=clearance,
                    evidence=loop_evidence,
                    protect_eligible=len(loop) >= 5,
                )
            )
            support_indices = np.linspace(
                0,
                len(loop) - 1,
                num=min(len(loop), 32),
                dtype=np.int64,
            )
            candidates.append(
                _make_candidate(
                    candidate_id=f"{mesh.path}:rim:{loop_index:04d}",
                    kind="rim",
                    scope_path=mesh.path,
                    confidence=rim_confidence,
                    confidence_threshold=confidence_threshold,
                    probe=ProtectedFeatureProbe(
                        kind="surface_support",
                        points_m=vertices[loop[support_indices]].tolist(),
                        tolerance_m=tolerance,
                    ),
                    minimum_thickness_m=clearance,
                    evidence=loop_evidence,
                    protect_eligible=len(loop) >= 5,
                )
            )
            toward_interior = object_center - center
            depth = float(abs(np.dot(toward_interior, normal)))
            if depth > max(radius, tolerance * 4.0):
                direction = normal * (1.0 if np.dot(toward_interior, normal) >= 0.0 else -1.0)
                candidates.append(
                    _make_candidate(
                        candidate_id=f"{mesh.path}:cavity:{loop_index:04d}",
                        kind="cavity",
                        scope_path=mesh.path,
                        confidence=0.58 + 0.16 * float(geometry["planarity"]),
                        confidence_threshold=confidence_threshold,
                        probe=ProtectedFeatureProbe(
                            kind="negative_space_path",
                            points_m=[
                                (center - direction * tolerance * 2.0).tolist(),
                                (center + direction * min(depth * 0.7, radius * 2.0)).tolist(),
                            ],
                            radius_m=max(clearance * 0.5, tolerance),
                            tolerance_m=tolerance,
                        ),
                        required_clearance_m=max(clearance * 0.5, tolerance),
                        evidence={**loop_evidence, "inferred_cavity_depth_m": depth},
                    )
                )

        slender_components: list[dict[str, Any]] = []
        for component_index, face_indices in enumerate(_face_components(triangles)):
            component_faces = triangles[face_indices]
            vertex_ids = np.unique(component_faces.reshape(-1))
            points = vertices[vertex_ids]
            if len(points) < 4:
                continue
            local_faces = np.searchsorted(vertex_ids, component_faces)
            extents, axes = _principal_extents(points)
            ordered_extents = np.maximum(extents, tolerance)
            length, width, thickness = [float(value) for value in ordered_extents]
            component_edges = _edge_owners(component_faces)
            boundary_count = sum(len(owners) == 1 for owners in component_edges.values())
            used_vertex_count = len(np.unique(component_faces))
            edge_count = len(component_edges)
            euler = used_vertex_count - edge_count + len(component_faces)
            genus = (2.0 - euler) / 2.0 if boundary_count == 0 else None
            sample_indices = np.linspace(
                0,
                len(vertex_ids) - 1,
                num=min(len(vertex_ids), 32),
                dtype=np.int64,
            )
            component_evidence = {
                "measurement": "position_welded_component_shape",
                "component_index": component_index,
                "face_count": len(component_faces),
                "extents_m": [length, width, thickness],
                "boundary_edge_count": boundary_count,
                "euler_characteristic": euler,
                "genus": genus,
            }
            accessible_voids, void_warning = _accessible_void_rays(
                points,
                local_faces,
                tolerance=tolerance,
            )
            if void_warning:
                warnings.append(f"{mesh.path}: {void_warning}")
            for void_index, void in enumerate(accessible_voids):
                support_ratio = float(void["supporting_ray_count"]) / float(
                    void["sampled_ray_count"]
                )
                confidence = min(0.78, 0.58 + support_ratio * 0.5)
                clearance = max(
                    float(void["minimum_clearance_m"]) * 0.5,
                    tolerance,
                )
                probe = ProtectedFeatureProbe(
                    kind="negative_space_path",
                    points_m=np.asarray(void["probe_points"], dtype=np.float64).tolist(),
                    radius_m=clearance,
                    tolerance_m=tolerance,
                )
                evidence = {
                    "measurement": "convex_hull_accessible_void_rays",
                    "component_index": component_index,
                    "axis_index": void["axis_index"],
                    "direction_sign": void["direction_sign"],
                    "direction": void["direction"],
                    "accessible_void_depth_m": void["depth_m"],
                    "minimum_clearance_m": void["minimum_clearance_m"],
                    "supporting_ray_count": void["supporting_ray_count"],
                    "sampled_ray_count": void["sampled_ray_count"],
                }
                suffix = f"{component_index:04d}:{void_index:02d}"
                candidates.append(
                    _make_candidate(
                        candidate_id=f"{mesh.path}:cavity:{suffix}",
                        kind="cavity",
                        scope_path=mesh.path,
                        confidence=min(0.79, confidence + 0.01),
                        confidence_threshold=confidence_threshold,
                        probe=probe,
                        required_clearance_m=clearance,
                        evidence=evidence,
                    )
                )
                rim_points = np.asarray(void["rim_points"], dtype=np.float64)
                if len(rim_points):
                    candidates.append(
                        _make_candidate(
                            candidate_id=f"{mesh.path}:rim:{suffix}",
                            kind="rim",
                            scope_path=mesh.path,
                            confidence=confidence,
                            confidence_threshold=confidence_threshold,
                            probe=ProtectedFeatureProbe(
                                kind="surface_support",
                                points_m=rim_points.tolist(),
                                tolerance_m=tolerance,
                            ),
                            minimum_thickness_m=clearance,
                            evidence=evidence,
                        )
                    )
            if genus is not None and genus >= 1.0 - 1e-9:
                candidates.append(
                    _make_candidate(
                        candidate_id=f"{mesh.path}:handle:{component_index:04d}",
                        kind="handle",
                        scope_path=mesh.path,
                        # Genus proves a topological loop, not semantic handle intent.
                        # Keep it review-only until a stronger affordance classifier exists.
                        confidence=min(0.79, 0.65 + 0.08 * genus),
                        confidence_threshold=confidence_threshold,
                        probe=ProtectedFeatureProbe(
                            kind="surface_support",
                            points_m=points[sample_indices].tolist(),
                            tolerance_m=tolerance,
                        ),
                        minimum_thickness_m=thickness,
                        evidence=component_evidence,
                    )
                )
            if length / max(width, tolerance) >= 5.0 and width / max(thickness, tolerance) <= 3.5:
                slender_components.append(
                    {
                        "component_index": component_index,
                        "axis": _stable_axis(axes[:, 0]),
                        "points": points,
                        "sample_indices": sample_indices,
                        "thickness": min(width, thickness),
                        "evidence": component_evidence,
                    }
                )

        parallel_groups: dict[int, int] = {}
        for left_index, left in enumerate(slender_components):
            count = sum(
                abs(float(np.dot(left["axis"], right["axis"]))) >= 0.95
                for right in slender_components
            )
            parallel_groups[left_index] = count
        for slender_index, descriptor in enumerate(slender_components):
            group_count = parallel_groups[slender_index]
            kind: Literal["wire", "tine"] = "tine" if group_count >= 2 else "wire"
            confidence = 0.82 if kind == "tine" else 0.68
            evidence = {
                **descriptor["evidence"],
                "parallel_slender_component_count": group_count,
                "principal_axis": descriptor["axis"].tolist(),
            }
            points = descriptor["points"]
            sample_indices = descriptor["sample_indices"]
            candidates.append(
                _make_candidate(
                    candidate_id=(f"{mesh.path}:{kind}:{int(descriptor['component_index']):04d}"),
                    kind=kind,
                    scope_path=mesh.path,
                    confidence=confidence,
                    confidence_threshold=confidence_threshold,
                    probe=ProtectedFeatureProbe(
                        kind="surface_support",
                        points_m=points[sample_indices].tolist(),
                        tolerance_m=tolerance,
                    ),
                    minimum_thickness_m=float(descriptor["thickness"]),
                    evidence=evidence,
                )
            )
    if len(candidates) > candidate_limit:
        detected_count = len(candidates)
        candidates = sorted(
            candidates,
            key=lambda item: (
                item.disposition != "protect",
                -item.confidence,
                item.candidate_id,
            ),
        )[:candidate_limit]
        warnings.append(
            f"protected-feature candidates were deterministically limited from "
            f"{detected_count} to {candidate_limit}; omitted hypotheses require review "
            "before reconstructive mutation"
        )
    candidates.sort(key=lambda item: item.candidate_id)
    return ProtectedFeatureCandidateReport(
        source_path=str(source),
        source_sha256=file_sha256(source),
        confidence_threshold=confidence_threshold,
        candidates=candidates,
        warnings=warnings,
    )


def candidate_to_protected_feature(candidate: ProtectedFeatureCandidate) -> ProtectedFeature:
    """Adapt detector evidence without granting collision or mutation authority."""

    explicit_kind = {
        "opening": "opening",
        "cavity": "cavity",
        "handle": "handle",
        "wire": "interface",
        "rim": "interface",
        "tine": "interface",
    }[candidate.kind]
    return ProtectedFeature(
        name=candidate.candidate_id,
        kind=explicit_kind,
        scope_path=candidate.scope_path,
        minimum_size_m=candidate.minimum_thickness_m,
        minimum_clearance_m=candidate.required_clearance_m,
        probe=candidate.probe,
        required=candidate.disposition == "protect",
        source="inferred_hypothesis",
        affected_roles=["render"],
    )


_REVIEW_ONLY_PROBE_SAMPLE_LIMIT = 128


def compare_protected_feature_candidates(
    source_path: str | Path,
    candidate_path: str | Path,
    candidates: list[ProtectedFeatureCandidate],
) -> ProtectedFeatureComparisonReport:
    """Probe inferred features before and after a mutation relative to the source."""

    source = Path(source_path).expanduser().resolve()
    candidate_geometry = Path(candidate_path).expanduser().resolve()
    comparisons: list[ProtectedFeatureCandidateComparison] = []
    failures: list[str] = []
    warnings: list[str] = []
    source_cache: dict[str, ProtectedFeatureProbeResult] = {}
    candidate_cache: dict[str, ProtectedFeatureProbeResult] = {}
    for inferred in sorted(candidates, key=lambda item: item.candidate_id):
        explicit = candidate_to_protected_feature(inferred)
        sample_limit = (
            _REVIEW_ONLY_PROBE_SAMPLE_LIMIT if inferred.disposition == "review_only" else 4096
        )
        probe_key = f"{sample_limit}:" + explicit.model_dump_json(
            exclude={"name", "kind", "scope_path", "source"}
        )
        if probe_key not in source_cache:
            source_cache[probe_key] = evaluate_feature_probe(
                source,
                explicit,
                include_guide_purpose=False,
                sample_limit=sample_limit,
            )
        if probe_key not in candidate_cache:
            candidate_cache[probe_key] = evaluate_feature_probe(
                candidate_geometry,
                explicit,
                include_guide_purpose=False,
                sample_limit=sample_limit,
            )
        before = source_cache[probe_key].model_copy(update={"feature_name": explicit.name})
        after = candidate_cache[probe_key].model_copy(update={"feature_name": explicit.name})
        reasons: list[str] = []
        if before.status == "pass" and after.status == "fail":
            status: Literal["preserved", "regressed", "not_evaluated"] = "regressed"
            reasons.extend(after.failures)
            message = f"protected candidate {inferred.candidate_id!r} regressed"
            (failures if inferred.disposition == "protect" else warnings).append(message)
        elif before.status == "pass" and after.status == "pass":
            status = "preserved"
        else:
            status = "not_evaluated"
            reasons.extend(before.warnings if before.status != "pass" else after.warnings)
            warnings.append(
                f"protected candidate {inferred.candidate_id!r} could not be compared deterministically"
            )
        if inferred.disposition == "review_only":
            warnings.append(
                f"protected candidate {inferred.candidate_id!r} is low-confidence review evidence "
                "and cannot authorize mutation"
            )
        comparisons.append(
            ProtectedFeatureCandidateComparison(
                candidate_id=inferred.candidate_id,
                kind=inferred.kind,
                confidence=inferred.confidence,
                disposition=inferred.disposition,
                mutation_authorized=False,
                status=status,
                before=before,
                after=after,
                reasons=reasons,
            )
        )
    return ProtectedFeatureComparisonReport(
        source_path=str(source),
        source_sha256=file_sha256(source),
        candidate_path=str(candidate_geometry),
        candidate_sha256=file_sha256(candidate_geometry),
        status="fail" if failures else "conditional" if warnings else "pass",
        comparisons=comparisons,
        failures=failures,
        warnings=warnings,
    )


def _sample_polyline(
    points: np.ndarray,
    step_m: float,
    limit: int = 4096,
) -> tuple[np.ndarray, bool, int]:
    """Sample a complete polyline uniformly and report resolution truncation."""

    if limit < 2:
        raise ValueError("polyline sample limit must be at least two")
    values = np.asarray(points, dtype=np.float64).reshape((-1, 3))
    if len(values) < 2:
        return values.copy(), False, len(values)
    segment_vectors = np.diff(values, axis=0)
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    total_length = float(np.sum(segment_lengths))
    if not np.isfinite(total_length) or total_length <= 1e-18:
        return values[:1].copy(), False, 1
    required_count = max(2, int(np.ceil(total_length / max(step_m, 1e-12))) + 1)
    sample_count = min(required_count, limit)
    distances = np.linspace(0.0, total_length, num=sample_count, dtype=np.float64)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    segment_ids = np.searchsorted(cumulative, distances, side="right") - 1
    segment_ids = np.clip(segment_ids, 0, len(segment_lengths) - 1)
    starts = cumulative[segment_ids]
    denominators = segment_lengths[segment_ids]
    ratios = np.divide(
        distances - starts,
        denominators,
        out=np.zeros_like(distances),
        where=denominators > 1e-18,
    )
    samples = values[segment_ids] + segment_vectors[segment_ids] * ratios[:, None]
    samples[-1] = values[-1]
    return samples, required_count > limit, required_count


def evaluate_feature_probe(
    geometry_path: str | Path,
    feature: ProtectedFeature,
    *,
    include_guide_purpose: bool = True,
    sample_limit: int = 4096,
) -> ProtectedFeatureProbeResult:
    """Evaluate one explicit metric-space feature probe against a geometry union."""

    if feature.probe is None:
        return ProtectedFeatureProbeResult(
            feature_name=feature.name,
            probe_kind="negative_space_path"
            if feature.kind in {"opening", "cavity", "clearance"}
            else "surface_support",
            status="not_evaluated",
            warnings=["protected feature has no deterministic probe geometry"],
        )
    probe = feature.probe
    control_points = np.asarray(probe.points_m, dtype=np.float64)
    tolerance = probe.tolerance_m or feature.tolerance_m or 1e-4
    required_clearance = (
        probe.radius_m
        if probe.radius_m is not None
        else feature.minimum_clearance_m
        if feature.minimum_clearance_m is not None
        else 0.0
    )
    if probe.kind == "negative_space_path":
        step = max(min(tolerance, max(required_clearance / 3.0, tolerance)), 1e-6)
        samples, resolution_truncated, required_sample_count = _sample_polyline(
            control_points,
            step,
            limit=sample_limit,
        )
        occupied, occupancy_error = points_inside_union(
            geometry_path,
            samples,
            include_guide_purpose=include_guide_purpose,
        )
        try:
            _nearest, distances, _normals = closest_surface(
                geometry_path,
                samples,
                include_guide_purpose=include_guide_purpose,
            )
        except Exception as exc:
            return ProtectedFeatureProbeResult(
                feature_name=feature.name,
                probe_kind=probe.kind,
                status="not_evaluated",
                sample_count=len(samples),
                warnings=[f"surface-distance query failed: {type(exc).__name__}: {exc}"],
            )
        failures: list[str] = []
        occupied_count = 0
        if occupied is None:
            return ProtectedFeatureProbeResult(
                feature_name=feature.name,
                probe_kind=probe.kind,
                status="not_evaluated",
                sample_count=len(samples),
                minimum_clearance_m=float(np.min(distances)) if len(distances) else None,
                warnings=[occupancy_error or "negative-space occupancy was not evaluated"],
            )
        occupied_count = int(np.count_nonzero(occupied))
        minimum_clearance = float(np.min(distances)) if len(distances) else None
        if occupied_count:
            failures.append(f"{occupied_count} path samples are inside solid geometry")
        if minimum_clearance is not None and minimum_clearance + tolerance < required_clearance:
            failures.append(
                f"minimum path clearance {minimum_clearance:.6g} m is below "
                f"required {required_clearance:.6g} m"
            )
        warnings = []
        status: Literal["pass", "fail", "not_evaluated"] = (
            "fail" if failures else "not_evaluated" if resolution_truncated else "pass"
        )
        if resolution_truncated:
            warnings.append(
                "negative-space path required "
                f"{required_sample_count} samples at the requested resolution; "
                f"the bounded probe evaluated {len(samples)} uniformly across the full path"
            )
        return ProtectedFeatureProbeResult(
            feature_name=feature.name,
            probe_kind=probe.kind,
            status=status,
            sample_count=len(samples),
            minimum_clearance_m=minimum_clearance,
            occupied_sample_count=occupied_count,
            failures=failures,
            warnings=warnings,
        )

    try:
        _nearest, distances, _normals = closest_surface(
            geometry_path,
            control_points,
            include_guide_purpose=include_guide_purpose,
        )
    except Exception as exc:
        return ProtectedFeatureProbeResult(
            feature_name=feature.name,
            probe_kind=probe.kind,
            status="not_evaluated",
            sample_count=len(control_points),
            warnings=[f"surface-distance query failed: {type(exc).__name__}: {exc}"],
        )
    maximum_distance = float(np.max(distances)) if len(distances) else None
    failures = []
    if maximum_distance is None:
        failures.append("surface support probe produced no distance samples")
    elif maximum_distance > tolerance:
        failures.append(
            f"maximum support distance {maximum_distance:.6g} m exceeds tolerance {tolerance:.6g} m"
        )
    return ProtectedFeatureProbeResult(
        feature_name=feature.name,
        probe_kind=probe.kind,
        status="fail" if failures else "pass",
        sample_count=len(control_points),
        maximum_surface_distance_m=maximum_distance,
        failures=failures,
    )


def evaluate_feature_probes(
    geometry_path: str | Path,
    features: list[ProtectedFeature],
) -> list[ProtectedFeatureProbeResult]:
    """Evaluate all explicitly declared protected features in stable name order."""

    return [
        evaluate_feature_probe(geometry_path, feature)
        for feature in sorted(features, key=lambda item: item.name)
    ]
