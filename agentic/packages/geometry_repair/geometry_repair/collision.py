# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Geometry-owned collision representation planning and authoring."""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import time
from collections import deque
from importlib import metadata
from pathlib import Path
from typing import Literal

import numpy as np

from .artifacts import atomic_write_json, file_sha256
from .collision_audit import (
    CollisionAuditLimits,
    CollisionAuditMetrics,
    CollisionComplexity,
    SourceCollisionAudit,
)
from .collision_planner import (
    CandidateRepresentation,
    CollisionCandidateEvaluation,
    evaluate_collision_candidate,
    select_collision_candidate,
)
from .compound_collision import compound_part_audit_map
from .mesh_io import load_meshes, measure_mesh, positional_topology_mesh
from .models import (
    CollisionDecompositionEvidence,
    CollisionReport,
    ProtectedFeature,
    ProtectedFeatureProbeResult,
    RepairBudgets,
    RepairProfile,
)
from .policy import assert_worker_operations_enabled
from .process_limits import bounded_process_environment
from .protected_features import evaluate_feature_probes
from .runtime import validate_collision_runtime, validate_static_collision_runtime
from .worker_ids import SDF_COLLISION_REBUILD_WORKER
from .workers.sdf_rebuild import reconstruct_collision_mesh_sdf

_COACD_THREAD_LIMIT = 1


def _record_protected_feature_probe_outcomes(
    *,
    protected_features: list[ProtectedFeature],
    probes: list[ProtectedFeatureProbeResult],
    protected_features_unmeasured: set[str],
    failures: list[str],
    warnings: list[str],
) -> None:
    """Record measured failures separately from probes that could not run."""

    by_name = {result.feature_name: result for result in probes}
    for feature in protected_features:
        result = by_name[feature.name]
        if result.status == "pass":
            continue
        if result.status == "fail":
            messages = [
                f"protected collision feature {feature.name!r}: {failure}"
                for failure in result.failures
            ]
            (failures if feature.required else warnings).extend(messages)
        elif feature.required:
            protected_features_unmeasured.add(feature.name)
            warnings.append(
                f"protected collision feature {feature.name!r} was not measured: "
                + "; ".join(result.warnings or ["no deterministic probe"])
                + "; explicit human review is required and this handoff cannot be certified"
            )


def _dynamic_collision_reconstruction_reasons(measured) -> list[str]:
    """Name source defects that prohibit direct dynamic collision derivation."""

    reasons: list[str] = []
    if measured.watertight is not True:
        reasons.append("source is not watertight")
    if measured.enclosed_volume_m3 is None or measured.enclosed_volume_m3 <= 0.0:
        reasons.append("source has no measured positive enclosed volume")
    for field_name, label in (
        ("invalid_index_count", "invalid indices"),
        ("non_finite_vertex_count", "non-finite vertices"),
        ("degenerate_face_count", "degenerate faces"),
        ("duplicate_face_count", "duplicate faces"),
        ("over_connected_edge_count", "over-connected edges"),
        ("non_manifold_vertex_count", "non-manifold vertices"),
        ("inconsistent_orientation_edge_count", "inconsistent orientation"),
    ):
        count = int(getattr(measured, field_name))
        if count:
            reasons.append(f"{label}: {count}")
    for field_name, label in (
        ("self_intersection_status", "self-intersection audit"),
        ("coplanar_overlap_status", "coplanar-overlap audit"),
        ("inverted_shell_status", "inverted-shell audit"),
    ):
        status = str(getattr(measured, field_name))
        if status != "pass":
            reasons.append(f"{label}: {status}")
    return reasons


def _unavailable_source_collision_audit(
    *,
    render: Path,
    collision: Path,
    report_path: Path,
    limits: CollisionAuditLimits,
    payloads: list[tuple[str, np.ndarray, np.ndarray, str]],
    reason: str,
    occupancy_face_point_limit: int,
) -> SourceCollisionAudit:
    complexity = CollisionComplexity(
        hull_count=len(payloads),
        total_vertices=sum(len(vertices) for _path, vertices, _faces, _kind in payloads),
        total_faces=sum(len(faces) for _path, _vertices, faces, _kind in payloads),
        maximum_vertices_per_hull=max(
            (len(vertices) for _path, vertices, _faces, _kind in payloads),
            default=0,
        ),
        maximum_faces_per_hull=max(
            (len(faces) for _path, _vertices, faces, _kind in payloads),
            default=0,
        ),
        collision_size_bytes=collision.stat().st_size,
    )
    report = SourceCollisionAudit(
        status="not_evaluated",
        decision="review_required",
        render_path=str(render),
        render_sha256=file_sha256(render),
        collision_path=str(collision),
        collision_sha256=file_sha256(collision),
        metrics=CollisionAuditMetrics(
            occupancy_status="not_evaluated",
            occupancy_reason=reason,
            occupancy_face_point_limit=occupancy_face_point_limit,
        ),
        complexity=complexity,
        limits=limits,
        warnings=[reason],
        report_path=str(report_path),
    )
    atomic_write_json(report_path, report)
    return report


def _run_source_collision_audit(
    *,
    render: Path,
    collision: Path,
    report_path: Path,
    limits: CollisionAuditLimits,
    protected_features: list[ProtectedFeature],
    payloads: list[tuple[str, np.ndarray, np.ndarray, str]],
    surface_sample_limit: int,
    occupancy_face_point_limit: int,
    memory_mb: int,
    timeout_s: float,
) -> SourceCollisionAudit:
    request_path = report_path.with_name("source_collision_audit_request.json")
    stdout_path = report_path.with_name("source_collision_audit_stdout.log")
    stderr_path = report_path.with_name("source_collision_audit_stderr.log")
    report_path.unlink(missing_ok=True)
    atomic_write_json(
        request_path,
        {
            "schema_version": "geometry-repair.source-collision-audit-request.v1",
            "render_path": str(render),
            "collision_path": str(collision),
            "report_path": str(report_path),
            "protected_features": [
                feature.model_dump(mode="json") for feature in protected_features
            ],
            "limits": limits.model_dump(mode="json"),
            "surface_sample_limit": surface_sample_limit,
            "occupancy_grid_resolution": 24,
            "occupancy_face_point_limit": occupancy_face_point_limit,
            "memory_mb": memory_mb,
        },
    )
    environment = bounded_process_environment()
    try:
        with (
            stdout_path.open("w", encoding="utf-8") as stdout,
            stderr_path.open("w", encoding="utf-8") as stderr,
        ):
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "geometry_repair.collision_audit_runner",
                    "--request",
                    str(request_path),
                ],
                check=False,
                env=environment,
                stdout=stdout,
                stderr=stderr,
                timeout=max(timeout_s, 0.001),
            )
    except subprocess.TimeoutExpired:
        return _unavailable_source_collision_audit(
            render=render,
            collision=collision,
            report_path=report_path,
            limits=limits,
            payloads=payloads,
            reason=f"source collision audit exceeded wall-clock budget of {timeout_s:.3f}s",
            occupancy_face_point_limit=occupancy_face_point_limit,
        )
    if completed.returncode != 0 or not report_path.is_file():
        return _unavailable_source_collision_audit(
            render=render,
            collision=collision,
            report_path=report_path,
            limits=limits,
            payloads=payloads,
            reason=f"source collision audit subprocess exited {completed.returncode}",
            occupancy_face_point_limit=occupancy_face_point_limit,
        )
    try:
        return SourceCollisionAudit.model_validate_json(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return _unavailable_source_collision_audit(
            render=render,
            collision=collision,
            report_path=report_path,
            limits=limits,
            payloads=payloads,
            reason=f"source collision audit report is invalid: {type(exc).__name__}: {exc}",
            occupancy_face_point_limit=occupancy_face_point_limit,
        )


def _surface_points(mesh, limit: int = 2048) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    centroids = np.asarray(mesh.triangles_center, dtype=np.float64)
    points = np.vstack((vertices, centroids)) if len(centroids) else vertices
    if len(points) <= limit:
        return points
    indices = np.linspace(0, len(points) - 1, num=limit, dtype=np.int64)
    return points[indices]


def _exterior_void_mask(occupied: np.ndarray) -> np.ndarray:
    """Return empty cells connected to the grid boundary in 26-neighborhood."""

    resolution = occupied.shape[0]
    exterior = np.zeros_like(occupied, dtype=bool)
    pending: deque[tuple[int, int, int]] = deque()
    for i in range(resolution):
        for j in range(resolution):
            for k in range(resolution):
                if (
                    i not in (0, resolution - 1)
                    and j not in (0, resolution - 1)
                    and k
                    not in (
                        0,
                        resolution - 1,
                    )
                ):
                    continue
                if not occupied[i, j, k] and not exterior[i, j, k]:
                    exterior[i, j, k] = True
                    pending.append((i, j, k))

    neighbor_offsets = tuple(
        (di, dj, dk)
        for di in (-1, 0, 1)
        for dj in (-1, 0, 1)
        for dk in (-1, 0, 1)
        if (di, dj, dk) != (0, 0, 0)
    )
    while pending:
        i, j, k = pending.popleft()
        for di, dj, dk in neighbor_offsets:
            neighbor = (i + di, j + dj, k + dk)
            if any(index < 0 or index >= resolution for index in neighbor):
                continue
            if occupied[neighbor] or exterior[neighbor]:
                continue
            exterior[neighbor] = True
            pending.append(neighbor)
    return exterior


def _exposed_union_surface_points(
    meshes: list,
    *,
    limit_per_mesh: int,
    grid_minimum: np.ndarray,
    grid_maximum: np.ndarray,
    occupied: np.ndarray,
) -> np.ndarray:
    """Sample the exterior-connected skin while excluding internal hull partitions."""

    if not meshes:
        return np.empty((0, 3), dtype=np.float64)
    points = np.vstack([_surface_points(mesh, limit_per_mesh) for mesh in meshes])
    if not len(points):
        return np.empty((0, 3), dtype=np.float64)

    exterior = _exterior_void_mask(occupied)
    resolution = occupied.shape[0]
    cell_size = (grid_maximum - grid_minimum) / resolution
    cell_indices = np.floor((points - grid_minimum) / cell_size).astype(np.int64)
    cell_indices = np.clip(cell_indices, 0, resolution - 1)
    adjacent_to_exterior = np.zeros(len(points), dtype=bool)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            for dk in (-1, 0, 1):
                neighbors = cell_indices + np.asarray((di, dj, dk), dtype=np.int64)
                valid = np.all((neighbors >= 0) & (neighbors < resolution), axis=1)
                adjacent_to_exterior[~valid] = True
                valid_neighbors = neighbors[valid]
                adjacent_to_exterior[valid] |= exterior[
                    valid_neighbors[:, 0],
                    valid_neighbors[:, 1],
                    valid_neighbors[:, 2],
                ]
    return points[adjacent_to_exterior]


def _closest_union_distance(points: np.ndarray, meshes: list) -> np.ndarray:
    import trimesh

    best = np.full(len(points), np.inf, dtype=np.float64)
    for mesh in meshes:
        try:
            _nearest, distances, _triangle_ids = trimesh.proximity.closest_point(mesh, points)
        except (ImportError, ModuleNotFoundError):
            chunks = []
            for start in range(0, len(points), 32):
                _nearest, distance, _triangle_ids = trimesh.proximity.closest_point_naive(
                    mesh,
                    points[start : start + 32],
                )
                chunks.append(distance)
            distances = np.concatenate(chunks, axis=0)
        best = np.minimum(best, distances)
    return best


def _contains_bounded(mesh, points: np.ndarray, chunk_size: int = 4096) -> np.ndarray:
    output = np.zeros(len(points), dtype=bool)
    if mesh.is_convex:
        normals = np.asarray(mesh.face_normals, dtype=np.float64)
        anchors = np.asarray(mesh.triangles[:, 0], dtype=np.float64)
        offsets = np.einsum("ij,ij->i", normals, anchors)
        tolerance = max(float(np.linalg.norm(np.ptp(mesh.vertices, axis=0))) * 1e-9, 1e-10)
        # Bound the temporary projection matrix while avoiding hundreds of
        # tiny allocations for common <=255-face convex collision hulls.
        effective_chunk_size = min(
            chunk_size,
            max(16, 2_000_000 // max(len(normals), 1)),
        )
        for start in range(0, len(points), effective_chunk_size):
            projections = points[start : start + effective_chunk_size] @ normals.T
            output[start : start + effective_chunk_size] = np.all(
                projections <= offsets[None, :] + tolerance,
                axis=1,
            )
        return output
    for start in range(0, len(points), chunk_size):
        output[start : start + chunk_size] = mesh.contains(points[start : start + chunk_size])
    return output


def _approximation_metrics(source_mesh, candidate_meshes: list) -> dict[str, float]:
    """Measure collision union error without summing overlapping hull volumes."""

    all_vertices = np.vstack(
        [
            np.asarray(source_mesh.vertices),
            *(np.asarray(mesh.vertices) for mesh in candidate_meshes),
        ]
    )
    minimum = all_vertices.min(axis=0)
    maximum = all_vertices.max(axis=0)
    extent = maximum - minimum
    padding = np.maximum(extent * 0.02, 1e-9)
    minimum -= padding
    maximum += padding
    source_face_count = len(source_mesh.faces)
    # A lower lattice for dense meshes caused orientation-sensitive false rejects;
    # 24^3 remains bounded while resolving the 1% occupied-space gate reliably.
    resolution = 24
    axes = [
        np.linspace(minimum[index], maximum[index], num=resolution, endpoint=False)
        + (maximum[index] - minimum[index]) / (2.0 * resolution)
        for index in range(3)
    ]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape((-1, 3))
    source_occupied = _contains_bounded(source_mesh, grid)
    candidate_occupied = np.zeros(len(grid), dtype=bool)
    for mesh in candidate_meshes:
        candidate_occupied |= _contains_bounded(mesh, grid)
    source_count = max(int(np.count_nonzero(source_occupied)), 1)
    candidate_count = int(np.count_nonzero(candidate_occupied))
    false_positive = int(np.count_nonzero(candidate_occupied & ~source_occupied))
    false_negative = int(np.count_nonzero(source_occupied & ~candidate_occupied))
    sample_limit = (
        512 if source_face_count > 10_000 else 1024 if source_face_count > 2_000 else 2048
    )
    source_points = _surface_points(source_mesh, sample_limit)
    candidate_points = _exposed_union_surface_points(
        candidate_meshes,
        limit_per_mesh=sample_limit,
        grid_minimum=minimum,
        grid_maximum=maximum,
        occupied=candidate_occupied.reshape((resolution, resolution, resolution)),
    )
    source_to_collision = _closest_union_distance(source_points, candidate_meshes)
    collision_to_source = (
        _closest_union_distance(candidate_points, [source_mesh])
        if len(candidate_points)
        else np.zeros((1,), dtype=np.float64)
    )
    return {
        "volume_excess_ratio": max(0, candidate_count - source_count) / source_count,
        "volume_deficit_ratio": max(0, source_count - candidate_count) / source_count,
        "false_positive_ratio": false_positive / source_count,
        "false_negative_ratio": false_negative / source_count,
        "surface_gap_m": float(np.percentile(source_to_collision, 99.0)),
        "surface_overreach_m": float(np.percentile(collision_to_source, 99.0)),
        "occupancy_grid_resolution": float(resolution),
    }


def _primitive_candidates(source_mesh) -> list[tuple[str, object]]:
    """Fit conservative box, sphere, cylinder, and capsule candidates."""

    import trimesh

    vertices = np.asarray(source_mesh.vertices, dtype=np.float64)
    if not len(vertices):
        return []
    minimum = vertices.min(axis=0)
    maximum = vertices.max(axis=0)
    center = (minimum + maximum) * 0.5
    extents = maximum - minimum
    box_transform = np.eye(4, dtype=np.float64)
    box_transform[:3, 3] = center
    candidates: list[tuple[str, object]] = [
        ("box", trimesh.creation.box(extents=extents, transform=box_transform))
    ]
    sphere_radius = float(np.max(np.linalg.norm(vertices - center, axis=1)))
    # A 127-point Fibonacci hull has 250 faces, stays below the PhysX convex
    # limit, and distributes approximation error more uniformly than deleting
    # vertices from an icosphere. The 1% radial guard offsets the finite hull's
    # inscribed bias; independent volume, occupancy, and surface gates still
    # decide whether the fit is usable for this source.
    point_count = 127
    point_indices = np.arange(point_count, dtype=np.float64)
    golden_ratio = (1.0 + 5.0**0.5) * 0.5
    sphere_z = 1.0 - 2.0 * (point_indices + 0.5) / point_count
    sphere_theta = 2.0 * np.pi * point_indices / golden_ratio
    sphere_radius_xy = np.sqrt(np.maximum(0.0, 1.0 - sphere_z**2))
    sphere_points = np.column_stack(
        (
            sphere_radius_xy * np.cos(sphere_theta),
            sphere_radius_xy * np.sin(sphere_theta),
            sphere_z,
        )
    )
    sphere_points = center + sphere_points * max(sphere_radius * 1.01, 1e-12)
    sphere = trimesh.convex.convex_hull(sphere_points)
    candidates.append(("sphere", sphere))

    centered = vertices - vertices.mean(axis=0)
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    axial = centered @ axis
    axial_min = float(axial.min())
    axial_max = float(axial.max())
    axial_center = vertices.mean(axis=0) + axis * ((axial_min + axial_max) * 0.5)
    radial = centered - np.outer(axial, axis)
    radius = float(np.max(np.linalg.norm(radial, axis=1)))
    height = max(axial_max - axial_min, 1e-12)
    rotation = trimesh.geometry.align_vectors([0.0, 0.0, 1.0], axis)
    transform = np.asarray(rotation, dtype=np.float64)
    transform[:3, 3] = axial_center
    candidates.append(
        (
            "cylinder",
            trimesh.creation.cylinder(
                radius=max(radius, 1e-12),
                height=height,
                sections=32,
                transform=transform,
            ),
        )
    )
    capsule_height = max(height - 2.0 * radius, 1e-12)
    candidates.append(
        (
            "capsule",
            trimesh.creation.capsule(
                height=capsule_height,
                radius=max(radius, 1e-12),
                count=[12, 12],
                transform=transform,
            ),
        )
    )
    return candidates


def _bounded_decomposition_input(source_mesh, budgets: RepairBudgets):
    """Simplify only the collision working copy when measured moderate drift permits it."""

    import trimesh

    if len(source_mesh.faces) <= 10_000:
        return source_mesh, None
    assert_worker_operations_enabled(
        "fast_simplification_collision_working_copy",
        "collision_only_quadric_decimation",
    )
    try:
        simplified = source_mesh.simplify_quadric_decimation(
            face_count=2_000,
            aggression=7,
        )
    except Exception as exc:
        return source_mesh, (
            "fast-simplification collision working-copy preprocessing failed; "
            f"retained the original mesh: {type(exc).__name__}: {exc}"
        )
    if (
        not isinstance(simplified, trimesh.Trimesh)
        or not simplified.is_watertight
        or not simplified.is_winding_consistent
        or len(simplified.faces) >= len(source_mesh.faces)
    ):
        return source_mesh, None
    source_volume = abs(float(source_mesh.volume))
    candidate_volume = abs(float(simplified.volume))
    if source_volume <= 0.0:
        return source_mesh, None
    volume_drift = abs(candidate_volume - source_volume) / source_volume
    if volume_drift > budgets.moderate_volume_drift:
        return source_mesh, None
    diagonal = float(np.linalg.norm(np.ptp(np.asarray(source_mesh.vertices), axis=0)))
    source_points = _surface_points(source_mesh, 1024)
    candidate_points = _surface_points(simplified, 1024)
    source_distance = _closest_union_distance(source_points, [simplified])
    candidate_distance = _closest_union_distance(candidate_points, [source_mesh])
    p99 = float(np.percentile(np.concatenate((source_distance, candidate_distance)), 99.0))
    if diagonal <= 0.0 or p99 / diagonal > budgets.moderate_p99_ratio:
        return source_mesh, None
    return simplified, (
        f"fast-simplification collision working copy: {len(source_mesh.faces)} -> "
        f"{len(simplified.faces)} faces; p99={p99:.6g} m; volume_drift={volume_drift:.6g}"
    )


def _bounded_convex_hull_reduction(hull, budgets: RepairBudgets):
    """Reduce only an over-budget convex candidate; independent gates decide fidelity."""

    import trimesh

    if (
        len(hull.vertices) <= budgets.max_collision_vertices_per_hull
        and len(hull.faces) <= budgets.max_collision_faces_per_hull
    ):
        return None, None
    assert_worker_operations_enabled(
        "fast_simplification_collision_working_copy",
        "collision_only_quadric_decimation",
    )
    maximum_target_faces = min(
        budgets.max_collision_faces_per_hull,
        max(4, 2 * budgets.max_collision_vertices_per_hull - 4),
    )
    targets = [maximum_target_faces]
    reduced_target = max(4, maximum_target_faces // 2)
    if reduced_target not in targets:
        targets.append(reduced_target)
    failures: list[str] = []
    for target_faces in targets:
        try:
            simplified = hull.simplify_quadric_decimation(
                face_count=target_faces,
                aggression=7,
            )
            reduced = simplified.convex_hull
        except Exception as exc:
            failures.append(f"target {target_faces}: {type(exc).__name__}: {exc}")
            continue
        if (
            not isinstance(reduced, trimesh.Trimesh)
            or not reduced.is_watertight
            or not reduced.is_winding_consistent
            or not math.isfinite(float(reduced.volume))
            or float(reduced.volume) <= 0.0
        ):
            failures.append(f"target {target_faces}: reduced hull is not a positive solid")
            continue
        if (
            len(reduced.vertices) > budgets.max_collision_vertices_per_hull
            or len(reduced.faces) > budgets.max_collision_faces_per_hull
        ):
            failures.append(
                f"target {target_faces}: reduced hull remains over complexity limits "
                f"({len(reduced.vertices)} vertices, {len(reduced.faces)} faces)"
            )
            continue
        return reduced, (
            f"bounded convex-hull reduction: {len(hull.vertices)} vertices/"
            f"{len(hull.faces)} faces -> {len(reduced.vertices)} vertices/"
            f"{len(reduced.faces)} faces"
        )
    return None, "; ".join(failures) or "bounded convex-hull reduction produced no candidate"


def _validated_source_collision_candidate(
    source_path: str,
    source_mesh,
    source_collision_meshes: list,
    budgets: RepairBudgets,
    audit: SourceCollisionAudit | None = None,
    *,
    rejection_reasons: list[str] | None = None,
) -> tuple[list[tuple[str, np.ndarray, np.ndarray, str]], dict[str, float]] | None:
    """Reuse authored source hulls only after independent convexity and error gates."""

    import trimesh

    def reject(reason: str) -> None:
        if rejection_reasons is not None:
            rejection_reasons.append(reason)
        return None

    if audit is not None and audit.decision != "preserve_source":
        return reject(f"source-collision audit decision was {audit.decision!r}")
    if not source_collision_meshes:
        return None
    if len(source_collision_meshes) > budgets.max_source_collision_prims:
        return reject(
            f"source collider count {len(source_collision_meshes)} exceeds the configured "
            f"limit {budgets.max_source_collision_prims}"
        )
    candidates = []
    payloads = []
    for source_index, source in enumerate(source_collision_meshes):
        vertices, faces = positional_topology_mesh(
            source.world_vertices_m,
            source.triangles,
        )
        candidate = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        if (
            not candidate.is_watertight
            or not candidate.is_winding_consistent
            or not candidate.is_convex
            or len(vertices) > budgets.max_collision_vertices_per_hull
            or len(faces) > budgets.max_collision_faces_per_hull
        ):
            return reject(
                f"source collider {source_index} failed convex hull topology or complexity gates"
            )
        candidates.append(candidate)
        payloads.append((source_path, vertices, faces, "source_collision_hull"))
    try:
        metrics = _approximation_metrics(source_mesh, candidates)
    except Exception as exc:
        return reject(
            f"source collider approximation could not be measured: {type(exc).__name__}: {exc}"
        )
    if (
        metrics["volume_excess_ratio"] > budgets.max_collision_volume_excess
        or metrics["volume_deficit_ratio"] > budgets.max_collision_volume_deficit
        or metrics["false_negative_ratio"] > budgets.max_collision_volume_deficit
    ):
        return reject(
            "source collider failed fixed approximation gates: "
            f"volume_excess_ratio={metrics['volume_excess_ratio']:.6g}, "
            f"volume_deficit_ratio={metrics['volume_deficit_ratio']:.6g}, "
            f"false_negative_ratio={metrics['false_negative_ratio']:.6g}"
        )
    return payloads, metrics


def _safe_name(value: str, index: int) -> str:
    name = re.sub(r"[^A-Za-z0-9_]", "_", value.strip("/"))
    if not name or not (name[0].isalpha() or name[0] == "_"):
        name = f"mesh_{name}"
    return f"{name}_{index:04d}"


def _coacd_threshold_m(
    source_path: str,
    diagonal_m: float,
    budgets: RepairBudgets,
    protected_features: list[ProtectedFeature],
    target_ratio: float = 0.01,
) -> float:
    epsilon = max(diagonal_m * 1e-9, 1e-9)
    # A slightly looser search threshold prevents pathological MCTS growth; the
    # independent collider error gate still enforces the stricter profile band.
    threshold = max(
        diagonal_m * max(budgets.moderate_p99_ratio, target_ratio),
        epsilon,
    )
    for feature in protected_features:
        if feature.source == "inferred_hypothesis":
            continue
        scope = feature.scope_path.rstrip("/") if feature.scope_path else None
        if scope and source_path != scope and not source_path.startswith(f"{scope}/"):
            continue
        caps = [
            feature.minimum_size_m / 3.0 if feature.minimum_size_m else None,
            feature.minimum_clearance_m / 3.0 if feature.minimum_clearance_m else None,
            feature.tolerance_m,
        ]
        active_caps = [value for value in caps if value is not None]
        if active_caps:
            threshold = min(threshold, *active_caps)
    return max(threshold, epsilon)


def _coacd_search_schedule(
    *,
    maximum_hulls: int,
    threshold_m: float,
    coarse_threshold_m: float | None = None,
) -> list[tuple[str, int, float]]:
    """Escalate complexity and fidelity only after a bounded coarse candidate fails."""

    if maximum_hulls < 2:
        return []
    coarse_hulls = min(maximum_hulls, 32)
    coarse_threshold = max(threshold_m, coarse_threshold_m or threshold_m)
    schedule = [("coarse", coarse_hulls, coarse_threshold)]
    if maximum_hulls > coarse_hulls:
        refined_threshold = (
            threshold_m if coarse_threshold_m is not None else max(threshold_m * 0.5, 1e-9)
        )
        schedule.append(("refined", maximum_hulls, refined_threshold))
    return schedule


def _protected_features_for_source(
    source_path: str,
    protected_features: list[ProtectedFeature],
) -> list[ProtectedFeature]:
    selected = []
    for feature in protected_features:
        scope = feature.scope_path.rstrip("/") if feature.scope_path else None
        if scope and source_path != scope and not source_path.startswith(f"{scope}/"):
            continue
        selected.append(feature)
    return selected


def _run_coacd(
    *,
    source_path: str,
    vertices: np.ndarray,
    triangles: np.ndarray,
    work_dir: Path,
    threshold_m: float,
    max_hulls: int,
    max_vertices: int,
    max_faces: int,
    seed: int,
    memory_mb: int,
    timeout_s: float,
    source_triangle_count: int | None = None,
    preprocessing: str | None = None,
) -> tuple[
    list[tuple[str, np.ndarray, np.ndarray, str]],
    CollisionDecompositionEvidence | None,
    Path,
    str | None,
]:
    """Run CoACD out of process and distrust every returned path and payload."""

    import trimesh

    assert_worker_operations_enabled("coacd_collision", "approximate_convex_decomposition")
    work_dir.mkdir(parents=True, exist_ok=True)
    parts_dir = (work_dir / "parts").resolve()
    input_path = (work_dir / "input.npz").resolve()
    result_path = (work_dir / "result.json").resolve()
    stdout_path = work_dir / "stdout.log"
    stderr_path = work_dir / "stderr.log"
    np.savez_compressed(input_path, vertices=vertices, faces=triangles)
    command = [
        sys.executable,
        "-m",
        "geometry_repair.coacd_runner",
        "--input",
        str(input_path),
        "--output-dir",
        str(parts_dir),
        "--result",
        str(result_path),
        "--threshold-m",
        repr(threshold_m),
        "--max-hulls",
        str(max_hulls),
        "--max-vertices",
        str(max_vertices),
        "--max-faces",
        str(max_faces),
        "--seed",
        str(seed),
        "--threads",
        str(_COACD_THREAD_LIMIT),
        "--memory-mb",
        str(memory_mb),
    ]
    environment = bounded_process_environment(deterministic_seed=seed)
    try:
        with (
            stdout_path.open("w", encoding="utf-8") as stdout,
            stderr_path.open("w", encoding="utf-8") as stderr,
        ):
            completed = subprocess.run(
                command,
                check=False,
                env=environment,
                stdout=stdout,
                stderr=stderr,
                timeout=max(timeout_s, 0.001),
            )
    except subprocess.TimeoutExpired:
        atomic_write_json(
            result_path,
            {
                "schema_version": "geometry-repair.coacd-result.v1",
                "status": "fail",
                "parts": [],
                "failures": [f"CoACD exceeded wall-clock budget of {timeout_s:.3f}s"],
            },
        )
        return [], None, result_path, f"CoACD exceeded wall-clock budget of {timeout_s:.3f}s"
    if not result_path.is_file():
        return [], None, result_path, f"CoACD exited {completed.returncode} without a result report"
    try:
        report = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [], None, result_path, f"invalid CoACD result report: {type(exc).__name__}: {exc}"
    if completed.returncode != 0 or report.get("status") != "pass":
        failures = report.get("failures") or [f"CoACD exited {completed.returncode}"]
        return [], None, result_path, "; ".join(str(item) for item in failures)
    if report.get("schema_version") != "geometry-repair.coacd-result.v1":
        return [], None, result_path, "CoACD result report has an unsupported schema version"
    records = report.get("parts")
    if not isinstance(records, list) or not records or len(records) > max_hulls:
        return [], None, result_path, "CoACD result report has an invalid part count"

    payloads: list[tuple[str, np.ndarray, np.ndarray, str]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            return [], None, result_path, f"CoACD part record {index} is not an object"
        part_path = Path(str(record.get("path", ""))).expanduser().resolve()
        if not part_path.is_relative_to(parts_dir) or not part_path.is_file():
            return [], None, result_path, f"CoACD part {index} escaped its output directory"
        if file_sha256(part_path) != record.get("sha256"):
            return [], None, result_path, f"CoACD part {index} failed digest verification"
        try:
            with np.load(part_path, allow_pickle=False) as part:
                part_vertices = np.asarray(part["vertices"], dtype=np.float64)
                part_triangles = np.asarray(part["faces"], dtype=np.int64)
        except (OSError, ValueError, KeyError) as exc:
            return [], None, result_path, f"CoACD part {index} is unreadable: {exc}"
        if (
            part_vertices.ndim != 2
            or part_vertices.shape[1:] != (3,)
            or part_triangles.ndim != 2
            or part_triangles.shape[1:] != (3,)
            or not np.isfinite(part_vertices).all()
            or len(part_vertices) > max_vertices
            or len(part_triangles) > max_faces
            or not len(part_triangles)
            or np.any(part_triangles < 0)
            or np.any(part_triangles >= len(part_vertices))
        ):
            return [], None, result_path, f"CoACD part {index} failed bounded topology checks"
        part_mesh = trimesh.Trimesh(
            vertices=part_vertices,
            faces=part_triangles,
            process=False,
        )
        if (
            not part_mesh.is_watertight
            or not part_mesh.is_winding_consistent
            or not part_mesh.is_convex
            or not np.isfinite(part_mesh.volume)
            or part_mesh.volume <= 0.0
        ):
            return [], None, result_path, f"CoACD part {index} is not a valid convex solid"
        payloads.append((source_path, part_vertices, part_triangles, "coacd_convex_hull"))

    version = str(report.get("coacd_version", ""))
    if version != "1.0.11":
        return [], None, result_path, f"unexpected CoACD version {version!r}"
    input_sha256 = file_sha256(input_path)
    if report.get("input_sha256") != input_sha256:
        return [], None, result_path, "CoACD result report has the wrong input digest"
    worker_warnings = [
        line.strip()
        for line in stdout_path.read_text(encoding="utf-8", errors="replace").splitlines()
        if "[warning]" in line or "exceeds the threshold" in line
    ][:20]
    evidence = CollisionDecompositionEvidence(
        source_path=source_path,
        worker_version=version,
        requested_threshold_m=threshold_m,
        thread_limit=int(report.get("thread_limit", 0)),
        input_sha256=input_sha256,
        source_triangle_count=source_triangle_count or len(triangles),
        decomposition_input_triangle_count=len(triangles),
        preprocessing=preprocessing,
        hull_count=len(payloads),
        elapsed_s=float(report.get("elapsed_s", 0.0)),
        report_path=str(result_path),
        report_sha256=file_sha256(result_path),
        stdout_path=str(stdout_path.resolve()),
        stdout_sha256=file_sha256(stdout_path),
        stderr_path=str(stderr_path.resolve()),
        stderr_sha256=file_sha256(stderr_path),
        warnings=worker_warnings,
    )
    return payloads, evidence, result_path, None


def _write_collision_stage(
    path: Path,
    payloads: list[tuple[str, np.ndarray, np.ndarray, str]],
    *,
    representation: str,
    preserve_source_paths: bool = False,
) -> Path:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt

    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    if preserve_source_paths:
        roots = sorted(
            {
                Sdf.Path(source_path).GetPrefixes()[0]
                for source_path, _vertices, _triangles, _kind in payloads
                if Sdf.Path(source_path).IsAbsolutePath()
                and len(Sdf.Path(source_path).GetPrefixes())
            },
            key=str,
        )
        if not roots:
            raise ValueError("source collision paths must be absolute USD prim paths")
        for source_root in roots:
            UsdGeom.Xform.Define(stage, source_root)
        stage.SetDefaultPrim(stage.GetPrimAtPath(roots[0]))
        stage.GetDefaultPrim().SetCustomDataByKey(
            "geometryRepairCollisionRepresentation",
            representation,
        )
    else:
        root = UsdGeom.Xform.Define(stage, "/CollisionGeometry")
        stage.SetDefaultPrim(root.GetPrim())
        root.GetPrim().SetCustomDataByKey(
            "geometryRepairCollisionRepresentation",
            representation,
        )
    for index, (source_path, vertices, triangles, kind) in enumerate(payloads):
        if preserve_source_paths:
            mesh_path = Sdf.Path(source_path)
            if not mesh_path.IsAbsolutePath() or mesh_path.IsRootPrimPath():
                raise ValueError(f"invalid source collision mesh path: {source_path}")
            for ancestor in mesh_path.GetParentPath().GetPrefixes():
                if ancestor.IsRootPrimPath() and not stage.GetPrimAtPath(ancestor):
                    UsdGeom.Xform.Define(stage, ancestor)
                elif not ancestor.IsRootPrimPath() and not stage.GetPrimAtPath(ancestor):
                    UsdGeom.Xform.Define(stage, ancestor)
        else:
            mesh_path = Sdf.Path(f"/CollisionGeometry/{_safe_name(source_path, index)}")
        mesh = UsdGeom.Mesh.Define(stage, mesh_path)
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices.astype(np.float32)))
        mesh.CreateFaceVertexCountsAttr(
            Vt.IntArray.FromNumpy(np.full(len(triangles), 3, dtype=np.int32))
        )
        mesh.CreateFaceVertexIndicesAttr(
            Vt.IntArray.FromNumpy(triangles.astype(np.int32).reshape(-1))
        )
        mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
        mesh.CreatePurposeAttr().Set(
            UsdGeom.Tokens.default_ if preserve_source_paths else UsdGeom.Tokens.guide
        )
        mesh.GetPrim().SetCustomDataByKey("sourcePrimPath", source_path)
        mesh.GetPrim().SetCustomDataByKey("collisionGeometryKind", kind)
        collision_api = UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        collision_api.CreateCollisionEnabledAttr().Set(True)
        mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
        mesh_collision.CreateApproximationAttr().Set("convexHull")
        if len(vertices):
            minimum = vertices.min(axis=0)
            maximum = vertices.max(axis=0)
            mesh.CreateExtentAttr(
                Vt.Vec3fArray(
                    [
                        Gf.Vec3f(*[float(value) for value in minimum]),
                        Gf.Vec3f(*[float(value) for value in maximum]),
                    ]
                )
            )
    stage.GetRootLayer().Save()
    return path


def _evaluate_collision_option(
    *,
    candidate_id: str,
    representation: CandidateRepresentation,
    generator: str,
    source_mesh,
    payloads: list[tuple[str, np.ndarray, np.ndarray, str]],
    diagonal_m: float,
    budgets: RepairBudgets,
    remaining_hull_budget: int,
    protected_features: list[ProtectedFeature],
    work_dir: Path,
    authored_source: bool = False,
) -> tuple[CollisionCandidateEvaluation, dict[str, float]]:
    """Materialize and independently measure one bounded collision option."""

    import trimesh

    meshes = [
        trimesh.Trimesh(vertices=vertices, faces=triangles, process=False)
        for _source_path, vertices, triangles, _kind in payloads
    ]
    metrics = _approximation_metrics(source_mesh, meshes)
    artifact_path = work_dir / f"{candidate_id}.usda"
    _write_collision_stage(
        artifact_path,
        payloads,
        representation=representation,
    )
    required_features = [feature for feature in protected_features if feature.required]
    protected_satisfied: bool | None = None
    probes = []
    if required_features:
        probes = evaluate_feature_probes(artifact_path, required_features)
        protected_satisfied = (
            False
            if any(probe.status == "fail" for probe in probes)
            else True
            if all(probe.status == "pass" for probe in probes)
            else None
        )
    evaluation = evaluate_collision_candidate(
        candidate_id=candidate_id,
        representation=representation,
        generator=generator,
        metrics=metrics,
        hull_count=len(payloads),
        total_vertices=sum(len(vertices) for _path, vertices, _faces, _kind in payloads),
        total_faces=sum(len(faces) for _path, _vertices, faces, _kind in payloads),
        maximum_vertices_per_hull=max(
            (len(vertices) for _path, vertices, _faces, _kind in payloads),
            default=0,
        ),
        maximum_faces_per_hull=max(
            (len(faces) for _path, _vertices, faces, _kind in payloads),
            default=0,
        ),
        diagonal_m=diagonal_m,
        budgets=budgets,
        remaining_hull_budget=remaining_hull_budget,
        protected_features=[feature.name for feature in required_features],
        protected_features_satisfied=protected_satisfied,
        protected_feature_probes=probes,
        authored_source=authored_source,
        artifact_path=str(artifact_path.resolve()),
        artifact_sha256=file_sha256(artifact_path),
    )
    return evaluation, metrics


def build_collision_geometry(
    render_path: str | Path,
    output_path: str | Path,
    *,
    profile: RepairProfile,
    budgets: RepairBudgets,
    protected_features: list[ProtectedFeature] | None = None,
    deterministic_seed: int = 0,
    coacd_enabled: bool = True,
    sdf_collision_rebuild_enabled: bool | None = None,
    openvdb_collision_rebuild_enabled: bool | None = None,
    timeout_s: float = 900.0,
    runtime_engine: Literal["skip", "fake", "ovphysx"] = "skip",
    report_path: str | Path | None = None,
    source_collision_audit_path: str | Path | None = None,
) -> CollisionReport:
    """Create a separate collision geometry layer without authoring physics."""

    if (
        sdf_collision_rebuild_enabled is not None
        and openvdb_collision_rebuild_enabled is not None
        and sdf_collision_rebuild_enabled != openvdb_collision_rebuild_enabled
    ):
        raise ValueError("conflicting canonical and legacy SDF collision toggles")
    if sdf_collision_rebuild_enabled is None:
        sdf_collision_rebuild_enabled = (
            True if openvdb_collision_rebuild_enabled is None else openvdb_collision_rebuild_enabled
        )

    output = Path(output_path).expanduser().resolve()
    render = Path(render_path).expanduser().resolve()
    source_render_sha256_before = file_sha256(render)
    requested_protected_features = list(protected_features or [])
    role_excluded_protected_features = sorted(
        {
            feature.name
            for feature in requested_protected_features
            if "collision" not in feature.affected_roles
        }
    )
    protected_features = [
        feature for feature in requested_protected_features if "collision" in feature.affected_roles
    ]
    if profile == "visual_only":
        report = CollisionReport(
            status="not_required",
            representation="none",
            source_render_path=str(render),
            source_render_sha256=source_render_sha256_before,
            source_render_sha256_after=file_sha256(render),
            source_render_unchanged=True,
        )
    else:
        import trimesh

        started = time.monotonic()
        meshes, _ = load_meshes(render)
        all_source_meshes, _ = load_meshes(render, include_guide_purpose=True)
        source_collision_meshes = [mesh for mesh in all_source_meshes if mesh.role == "collision"]
        explicit_protected_feature_names = {
            feature.name
            for feature in (protected_features or [])
            if feature.source != "inferred_hypothesis"
        }
        inferred_protected_feature_names = {
            feature.name
            for feature in (protected_features or [])
            if feature.source == "inferred_hypothesis"
        }
        payloads: list[tuple[str, np.ndarray, np.ndarray, str]] = []
        failures: list[str] = []
        warnings: list[str] = []
        source_collision_audit: SourceCollisionAudit | None = None
        resolved_source_collision_audit_path: Path | None = None
        source_collision_snapshot_path: Path | None = None
        if source_collision_meshes and profile == "rigid_pick_place":
            resolved_source_collision_audit_path = (
                Path(source_collision_audit_path).expanduser().resolve()
                if source_collision_audit_path
                else output.with_name("source_collision_audit.json")
            )
            try:
                source_collision_snapshot_path = resolved_source_collision_audit_path.with_name(
                    "source_collision_geometry.usda"
                )
                source_collision_payloads: list[tuple[str, np.ndarray, np.ndarray, str]] = []
                for source_mesh in source_collision_meshes:
                    triangles = np.asarray(source_mesh.triangles, dtype=np.int64).reshape((-1, 3))
                    valid = np.all(
                        (triangles >= 0) & (triangles < len(source_mesh.world_vertices_m)),
                        axis=1,
                    )
                    if np.any(valid):
                        source_collision_payloads.append(
                            (
                                source_mesh.path,
                                np.asarray(source_mesh.world_vertices_m, dtype=np.float64),
                                triangles[valid],
                                "source_collision_hull",
                            )
                        )
                if not source_collision_payloads:
                    raise ValueError("source collision inventory has no valid triangles")
                _write_collision_stage(
                    source_collision_snapshot_path,
                    source_collision_payloads,
                    representation="source_collision_audit",
                    preserve_source_paths=True,
                )
                source_collision_limits = CollisionAuditLimits(
                    max_volume_excess_ratio=budgets.max_collision_volume_excess,
                    max_volume_deficit_ratio=budgets.max_collision_volume_deficit,
                    max_false_positive_ratio=budgets.max_collision_volume_excess,
                    max_false_negative_ratio=budgets.max_collision_volume_deficit,
                    advisory_hull_count=budgets.max_collision_hulls,
                    max_vertices_per_hull=budgets.max_collision_vertices_per_hull,
                    max_faces_per_hull=budgets.max_collision_faces_per_hull,
                )
                source_collision_audit = _run_source_collision_audit(
                    render=render,
                    collision=source_collision_snapshot_path,
                    report_path=resolved_source_collision_audit_path,
                    limits=source_collision_limits,
                    protected_features=protected_features or [],
                    payloads=source_collision_payloads,
                    surface_sample_limit=budgets.sample_point_limit,
                    occupancy_face_point_limit=(budgets.max_collision_occupancy_face_point_product),
                    memory_mb=budgets.max_memory_mb,
                    timeout_s=max(
                        0.001,
                        min(
                            budgets.source_collision_audit_wall_time_s,
                            timeout_s - (time.monotonic() - started),
                        ),
                    ),
                )
                if source_collision_audit.decision != "preserve_source":
                    warnings.extend(
                        "source collision regeneration gate: " + reason
                        for reason in source_collision_audit.regeneration_reasons
                    )
                warnings.extend(source_collision_audit.warnings)
            except (OSError, RuntimeError, ValueError) as exc:
                warnings.append(
                    "source collision could not be certified for reuse: "
                    f"{type(exc).__name__}: {exc}"
                )
        compound_part_audits = {}
        compound_review_reason = None
        if profile == "rigid_pick_place" and source_collision_meshes and len(meshes) > 1:
            if len(source_collision_meshes) > budgets.max_source_collision_prims:
                compound_review_reason = (
                    f"compound source collision inventory has {len(source_collision_meshes)} "
                    f"prims, above intake limit {budgets.max_source_collision_prims}"
                )
            elif source_collision_audit is None:
                compound_review_reason = "compound source collision audit evidence is unavailable"
            else:
                try:
                    compound_part_audits = compound_part_audit_map(
                        source_collision_audit.metrics.per_render_part_coverage
                    )
                    expected_render_paths = {mesh.path for mesh in meshes}
                    audited_render_paths = set(compound_part_audits)
                    if audited_render_paths != expected_render_paths:
                        missing = sorted(expected_render_paths - audited_render_paths)
                        unexpected = sorted(audited_render_paths - expected_render_paths)
                        compound_review_reason = (
                            "compound source collision audit does not cover the render inventory: "
                            f"missing={missing}, unexpected={unexpected}"
                        )
                    audited_collision_paths = {
                        path
                        for part in compound_part_audits.values()
                        for path in part.collision_paths
                    }
                    expected_collision_paths = {mesh.path for mesh in source_collision_meshes}
                    if (
                        compound_review_reason is None
                        and audited_collision_paths != expected_collision_paths
                    ):
                        missing = sorted(expected_collision_paths - audited_collision_paths)
                        unexpected = sorted(audited_collision_paths - expected_collision_paths)
                        compound_review_reason = (
                            "compound source collision audit does not cover the collision "
                            f"inventory: missing={missing}, unexpected={unexpected}"
                        )
                except ValueError as exc:
                    compound_review_reason = (
                        "compound source collision audit evidence is invalid: "
                        f"{type(exc).__name__}: {exc}"
                    )
        if profile == "rigid_pick_place" and (
            compound_review_reason is not None
            or (
                source_collision_audit is not None
                and source_collision_audit.decision == "review_required"
            )
        ):
            if compound_review_reason is not None:
                warnings.append(compound_review_reason)
            warnings.append(
                "source collision audit requires review; automatic collision replacement "
                "is prohibited when required evidence is unavailable"
            )
            explicit_probes = [
                probe
                for probe in (
                    source_collision_audit.protected_feature_probes
                    if source_collision_audit is not None
                    else []
                )
                if probe.feature_name in explicit_protected_feature_names
            ]
            inferred_probes = [
                probe
                for probe in (
                    source_collision_audit.protected_feature_probes
                    if source_collision_audit is not None
                    else []
                )
                if probe.feature_name in inferred_protected_feature_names
            ]
            audited_complexity = (
                source_collision_audit.complexity
                if source_collision_audit is not None
                else CollisionComplexity(
                    hull_count=len(source_collision_meshes),
                    total_vertices=sum(
                        len(mesh.world_vertices_m) for mesh in source_collision_meshes
                    ),
                    total_faces=sum(len(mesh.triangles) for mesh in source_collision_meshes),
                    maximum_vertices_per_hull=max(
                        (len(mesh.world_vertices_m) for mesh in source_collision_meshes),
                        default=0,
                    ),
                    maximum_faces_per_hull=max(
                        (len(mesh.triangles) for mesh in source_collision_meshes),
                        default=0,
                    ),
                    collision_size_bytes=(
                        source_collision_snapshot_path.stat().st_size
                        if source_collision_snapshot_path is not None
                        and source_collision_snapshot_path.is_file()
                        else 0
                    ),
                )
            )
            report = CollisionReport(
                status="conditional",
                representation="none",
                source_render_path=str(render),
                source_render_sha256=source_render_sha256_before,
                source_render_sha256_after=file_sha256(render),
                source_render_unchanged=(file_sha256(render) == source_render_sha256_before),
                generator="geometry_repair.source_collision_review",
                source_body_count=len(meshes),
                source_collision_prim_count=0,
                total_vertices=audited_complexity.total_vertices,
                total_faces=audited_complexity.total_faces,
                collision_size_bytes=audited_complexity.collision_size_bytes,
                maximum_vertices_per_hull=audited_complexity.maximum_vertices_per_hull,
                maximum_faces_per_hull=audited_complexity.maximum_faces_per_hull,
                protected_feature_probes=explicit_probes,
                inferred_protected_feature_probes=inferred_probes,
                role_excluded_protected_features=role_excluded_protected_features,
                warnings=warnings,
                runtime_status="not_evaluated",
                runtime_engine=runtime_engine,
                report_path=(
                    str(Path(report_path).expanduser().resolve()) if report_path else None
                ),
                source_collision_audit_path=(
                    str(resolved_source_collision_audit_path)
                    if resolved_source_collision_audit_path is not None
                    and resolved_source_collision_audit_path.is_file()
                    else None
                ),
            )
            if report_path:
                atomic_write_json(report_path, report)
            return report
        decompositions: list[CollisionDecompositionEvidence] = []
        decomposition_report_paths: list[str] = []
        collision_reconstruction_paths: list[str] = []
        collision_reconstruction_count = 0
        generators: set[str] = set()
        protected_features_applied: set[str] = set()
        protected_features_unmeasured: set[str] = set()
        hull_count = 0
        maximum_vertices = 0
        maximum_faces = 0
        maximum_volume_excess: float | None = None
        maximum_volume_deficit: float | None = None
        maximum_surface_overreach: float | None = None
        maximum_surface_gap: float | None = None
        maximum_false_positive: float | None = None
        maximum_false_negative: float | None = None
        primitive_fit_kinds: list[str] = []
        candidate_search_paths: list[str] = []
        protected_feature_probes = []
        source_collision_prim_count = 0
        generated_collision_prim_count = 0
        preserve_source_collision_paths = False
        if not meshes:
            failures.append("collision planning found no renderable source bodies")
        if profile == "static_environment":
            protected_features_applied.update(
                feature.name for feature in (protected_features or [])
            )
            for mesh in meshes:
                if len(mesh.world_vertices_m) > budgets.max_collision_source_vertices_per_body:
                    failures.append(
                        f"{mesh.path}: source body has {len(mesh.world_vertices_m)} vertices, "
                        f"above collision planning limit "
                        f"{budgets.max_collision_source_vertices_per_body}"
                    )
                    continue
                payloads.append(
                    (
                        mesh.path,
                        mesh.world_vertices_m,
                        mesh.triangles,
                        "static_triangle_mesh",
                    )
                )
            representation = "static_triangle_mesh"
            generators.add("geometry_repair.static_triangle_collision")
        elif profile == "rigid_pick_place":
            assert_worker_operations_enabled(
                "trimesh_convex_hull_collision",
                "convex_hull_collision",
            )
            representation = "convex_hull"
            for mesh_index, mesh in enumerate(meshes):
                if len(mesh.world_vertices_m) > budgets.max_collision_source_vertices_per_body:
                    failures.append(
                        f"{mesh.path}: source body has {len(mesh.world_vertices_m)} vertices, "
                        f"above collision planning limit "
                        f"{budgets.max_collision_source_vertices_per_body}"
                    )
                    continue
                measured = measure_mesh(mesh)
                welded_vertices, welded_triangles = positional_topology_mesh(
                    mesh.world_vertices_m,
                    mesh.triangles,
                )
                source_mesh = trimesh.Trimesh(
                    vertices=welded_vertices,
                    faces=welded_triangles,
                    process=False,
                )
                mesh_protections = _protected_features_for_source(
                    mesh.path,
                    protected_features or [],
                )
                collision_sensitive = [
                    feature
                    for feature in mesh_protections
                    if feature.required
                    and feature.kind
                    in {
                        "opening",
                        "cavity",
                        "clearance",
                        "handle",
                        "mating_surface",
                        "interface",
                    }
                ]
                protected_features_applied.update(feature.name for feature in mesh_protections)
                reconstruction_reasons = _dynamic_collision_reconstruction_reasons(measured)
                if reconstruction_reasons:
                    if not budgets.allow_reconstructive:
                        failures.append(
                            f"{mesh.path}: dynamic collision source requires reconstruction "
                            f"({'; '.join(reconstruction_reasons)}), but collision-only "
                            "reconstruction was not explicitly enabled"
                        )
                        continue
                    if not sdf_collision_rebuild_enabled:
                        failures.append(
                            f"{mesh.path}: collision-only SDF reconstruction is disabled by "
                            "the request worker allow-list"
                        )
                        continue
                    assert_worker_operations_enabled(
                        SDF_COLLISION_REBUILD_WORKER,
                        "per_part_signed_collision_reconstruction",
                    )
                    feature_scales = [
                        value
                        for feature in mesh_protections
                        if feature.required and feature.source != "inferred_hypothesis"
                        for value in (
                            feature.minimum_size_m,
                            feature.minimum_clearance_m,
                            (
                                2.0 * feature.probe.radius_m
                                if feature.probe is not None and feature.probe.radius_m is not None
                                else None
                            ),
                        )
                        if value is not None
                    ]
                    maximum_grid = min(
                        budgets.max_collision_reconstruction_grid_dimension,
                        budgets.max_reconstruction_grid_dimension,
                    )
                    grid_ladder: list[int] = []
                    for ratio in (1.0, 0.75, 0.5, 0.375, 0.25):
                        grid_dimension = max(32, int(round(maximum_grid * ratio)))
                        if grid_dimension not in grid_ladder:
                            grid_ladder.append(grid_dimension)
                    reconstruction = None
                    reconstruction_failures: list[str] = []
                    for grid_dimension in grid_ladder:
                        remaining_time = timeout_s - (time.monotonic() - started)
                        candidate_reconstruction = reconstruct_collision_mesh_sdf(
                            source_render=render,
                            mesh=mesh,
                            work_dir=(
                                output.parent
                                / "collision_reconstruction"
                                / _safe_name(mesh.path, mesh_index)
                                / f"grid_{grid_dimension}"
                            ),
                            request_id=(
                                f"collision-rebuild:{mesh_index:04d}:grid-{grid_dimension}"
                            ),
                            max_grid_dimension=grid_dimension,
                            max_output_faces=budgets.max_reconstruction_output_faces,
                            feature_voxels=budgets.reconstruction_feature_voxels,
                            minimum_feature_m=(min(feature_scales) if feature_scales else None),
                            deterministic_seed=(deterministic_seed + mesh_index) % (2**31),
                            timeout_s=max(remaining_time, 0.001),
                            max_surface_p99_ratio=budgets.reconstructive_p99_ratio,
                        )
                        if candidate_reconstruction.evidence_path:
                            collision_reconstruction_paths.append(
                                candidate_reconstruction.evidence_path
                            )
                        if (
                            candidate_reconstruction.status == "success"
                            and candidate_reconstruction.vertices is not None
                            and candidate_reconstruction.triangles is not None
                        ):
                            reconstruction = candidate_reconstruction
                            break
                        reasons = candidate_reconstruction.failures or [
                            f"status {candidate_reconstruction.status!r}"
                        ]
                        reconstruction_failures.extend(
                            f"grid {grid_dimension}: {reason}" for reason in reasons
                        )
                        if candidate_reconstruction.status == "unavailable":
                            break
                    if reconstruction is None:
                        failures.extend(
                            f"{mesh.path}: collision-only reconstruction {reason}"
                            for reason in reconstruction_failures
                        )
                        continue
                    warnings.extend(
                        f"{mesh.path}: collision-only reconstruction excluded {reason}"
                        for reason in reconstruction_failures
                    )
                    warnings.extend(
                        f"{mesh.path}: {warning}" for warning in reconstruction.warnings
                    )
                    source_mesh = trimesh.Trimesh(
                        vertices=reconstruction.vertices,
                        faces=reconstruction.triangles,
                        process=False,
                    )
                    collision_reconstruction_count += 1
                    warnings.append(
                        f"{mesh.path}: collision candidates were measured against a bounded "
                        "collision-only reconstruction; render geometry was not rewritten; "
                        "source reasons: " + "; ".join(reconstruction_reasons)
                    )
                if (
                    not source_mesh.is_watertight
                    or not source_mesh.is_winding_consistent
                    or not math.isfinite(float(source_mesh.volume))
                    or float(source_mesh.volume) <= 0.0
                ):
                    failures.append(
                        f"{mesh.path}: dynamic collision working copy is not a positive, "
                        "consistently wound solid"
                    )
                    continue
                source_collisions_for_mesh = source_collision_meshes
                candidate_audit = source_collision_audit
                source_reuse_authorized = len(meshes) == 1
                if compound_part_audits:
                    part_audit = compound_part_audits[mesh.path]
                    source_reuse_authorized = part_audit.decision == "preserve_source"
                    selected_paths = set(part_audit.collision_paths)
                    source_collisions_for_mesh = sorted(
                        (
                            source_collision
                            for source_collision in source_collision_meshes
                            if source_collision.path in selected_paths
                        ),
                        key=lambda item: item.path,
                    )
                    candidate_audit = None
                candidate_prefix = _safe_name(mesh.path, mesh_index)
                candidate_work_dir = output.parent / "collision_candidates" / candidate_prefix
                candidate_work_dir.mkdir(parents=True, exist_ok=True)
                remaining_hulls = budgets.max_collision_hulls - generated_collision_prim_count
                remaining_body_count = len(meshes) - mesh_index - 1
                body_hull_budget = max(remaining_hulls - remaining_body_count, 0)
                candidate_evaluations: list[CollisionCandidateEvaluation] = []
                candidate_payloads: dict[
                    str,
                    list[tuple[str, np.ndarray, np.ndarray, str]],
                ] = {}
                candidate_generators: dict[str, str] = {}
                candidate_primitive_kinds: dict[str, str] = {}
                candidate_generation_failures: list[str] = []

                source_collision_rejections: list[str] = []
                source_collision_candidate = (
                    _validated_source_collision_candidate(
                        mesh.path,
                        source_mesh,
                        source_collisions_for_mesh,
                        budgets,
                        candidate_audit,
                        rejection_reasons=source_collision_rejections,
                    )
                    if source_reuse_authorized
                    else None
                )
                for rejection in source_collision_rejections:
                    message = f"source collision candidate excluded: {rejection}"
                    candidate_generation_failures.append(message)
                    warnings.append(f"{mesh.path}: {message}")
                if source_collision_candidate is not None:
                    source_payloads, _source_metrics = source_collision_candidate
                    candidate_id = f"{candidate_prefix}_source_collision"
                    evaluation, _metrics = _evaluate_collision_option(
                        candidate_id=candidate_id,
                        representation="source_collision",
                        generator="geometry_repair.source_collision_reuse",
                        source_mesh=source_mesh,
                        payloads=source_payloads,
                        diagonal_m=measured.bbox_diagonal_m or 0.0,
                        budgets=budgets,
                        remaining_hull_budget=body_hull_budget,
                        protected_features=collision_sensitive,
                        work_dir=candidate_work_dir,
                        authored_source=True,
                    )
                    candidate_evaluations.append(evaluation)
                    candidate_payloads[candidate_id] = source_payloads
                    candidate_generators[candidate_id] = "geometry_repair.source_collision_reuse"

                for primitive_kind, primitive_mesh in _primitive_candidates(source_mesh):
                    source_volume = abs(float(source_mesh.volume))
                    primitive_volume = abs(float(primitive_mesh.volume))
                    primitive_volume_excess = (
                        max(
                            0.0,
                            primitive_volume - source_volume,
                        )
                        / source_volume
                    )
                    primitive_volume_deficit = (
                        max(
                            0.0,
                            source_volume - primitive_volume,
                        )
                        / source_volume
                    )
                    if (
                        not math.isfinite(primitive_volume)
                        or primitive_volume_excess > budgets.max_collision_volume_excess
                        or primitive_volume_deficit > budgets.max_collision_volume_deficit
                    ):
                        candidate_generation_failures.append(
                            f"primitive {primitive_kind} candidate excluded by "
                            "exact-volume pre-screen: "
                            f"excess={primitive_volume_excess:.6g}, "
                            f"deficit={primitive_volume_deficit:.6g}"
                        )
                        continue
                    primitive_vertices = np.asarray(
                        primitive_mesh.vertices,
                        dtype=np.float64,
                    )
                    primitive_triangles = np.asarray(
                        primitive_mesh.faces,
                        dtype=np.int64,
                    )
                    candidate_id = f"{candidate_prefix}_primitive_{primitive_kind}"
                    primitive_payloads = [
                        (
                            mesh.path,
                            primitive_vertices,
                            primitive_triangles,
                            f"primitive_{primitive_kind}",
                        )
                    ]
                    try:
                        evaluation, _metrics = _evaluate_collision_option(
                            candidate_id=candidate_id,
                            representation=f"primitive_{primitive_kind}",  # type: ignore[arg-type]
                            generator="geometry_repair.primitive_fit",
                            source_mesh=source_mesh,
                            payloads=primitive_payloads,
                            diagonal_m=measured.bbox_diagonal_m or 0.0,
                            budgets=budgets,
                            remaining_hull_budget=body_hull_budget,
                            protected_features=collision_sensitive,
                            work_dir=candidate_work_dir,
                        )
                    except Exception as exc:
                        warnings.append(
                            f"{mesh.path}: primitive {primitive_kind} candidate could not "
                            f"be evaluated: {type(exc).__name__}: {exc}"
                        )
                        continue
                    candidate_evaluations.append(evaluation)
                    candidate_payloads[candidate_id] = primitive_payloads
                    candidate_generators[candidate_id] = "geometry_repair.primitive_fit"
                    candidate_primitive_kinds[candidate_id] = primitive_kind

                hull = source_mesh.convex_hull
                vertices = np.asarray(hull.vertices, dtype=np.float64)
                triangles = np.asarray(hull.faces, dtype=np.int64)
                hull_payloads = [(mesh.path, vertices, triangles, "convex_hull")]
                hull_candidate_id = f"{candidate_prefix}_convex_hull"
                try:
                    hull_evaluation, hull_metrics = _evaluate_collision_option(
                        candidate_id=hull_candidate_id,
                        representation="convex_hull",
                        generator="trimesh_convex_hull_collision",
                        source_mesh=source_mesh,
                        payloads=hull_payloads,
                        diagonal_m=measured.bbox_diagonal_m or 0.0,
                        budgets=budgets,
                        remaining_hull_budget=body_hull_budget,
                        protected_features=collision_sensitive,
                        work_dir=candidate_work_dir,
                    )
                    candidate_evaluations.append(hull_evaluation)
                    candidate_payloads[hull_candidate_id] = hull_payloads
                    candidate_generators[hull_candidate_id] = "trimesh_convex_hull_collision"
                    single_hull_excess = hull_metrics["volume_excess_ratio"]
                except Exception as exc:
                    single_hull_excess = math.inf
                    warnings.append(
                        f"{mesh.path}: convex-hull candidate could not be evaluated: "
                        f"{type(exc).__name__}: {exc}"
                    )
                reduced_hull, reduction_evidence = _bounded_convex_hull_reduction(
                    hull,
                    budgets,
                )
                if reduced_hull is not None:
                    reduced_vertices = np.asarray(reduced_hull.vertices, dtype=np.float64)
                    reduced_triangles = np.asarray(reduced_hull.faces, dtype=np.int64)
                    reduced_payloads = [
                        (
                            mesh.path,
                            reduced_vertices,
                            reduced_triangles,
                            "convex_hull",
                        )
                    ]
                    reduced_candidate_id = f"{candidate_prefix}_convex_hull_reduced"
                    try:
                        reduced_evaluation, _reduced_metrics = _evaluate_collision_option(
                            candidate_id=reduced_candidate_id,
                            representation="convex_hull",
                            generator="geometry_repair.bounded_convex_hull_reduction",
                            source_mesh=source_mesh,
                            payloads=reduced_payloads,
                            diagonal_m=measured.bbox_diagonal_m or 0.0,
                            budgets=budgets,
                            remaining_hull_budget=body_hull_budget,
                            protected_features=collision_sensitive,
                            work_dir=candidate_work_dir,
                        )
                        candidate_evaluations.append(reduced_evaluation)
                        candidate_payloads[reduced_candidate_id] = reduced_payloads
                        candidate_generators[reduced_candidate_id] = (
                            "geometry_repair.bounded_convex_hull_reduction"
                        )
                    except Exception as exc:
                        candidate_generation_failures.append(
                            "bounded convex-hull candidate could not be evaluated: "
                            f"{type(exc).__name__}: {exc}"
                        )
                elif reduction_evidence:
                    candidate_generation_failures.append(reduction_evidence)
                initial_search = select_collision_candidate(
                    mesh.path,
                    candidate_evaluations,
                )
                needs_decomposition = initial_search.status != "pass"
                if needs_decomposition:
                    if not coacd_enabled:
                        protected_reason = (
                            "; protected collision features: "
                            + ", ".join(feature.name for feature in collision_sensitive)
                            if collision_sensitive
                            else ""
                        )
                        message = (
                            f"CoACD is required because the single hull has "
                            f"{len(vertices)} vertices/{len(triangles)} faces and volume excess "
                            f"{single_hull_excess:.4f}{protected_reason}, but coacd_collision is disabled by the "
                            "request worker allow-list"
                        )
                        candidate_generation_failures.append(message)
                        warnings.append(f"{mesh.path}: {message}")
                    else:
                        if body_hull_budget <= 1:
                            message = "no hull budget remains for convex decomposition"
                            candidate_generation_failures.append(message)
                            warnings.append(f"{mesh.path}: {message}")
                        else:
                            decomposition_mesh, decomposition_preprocessing = (
                                _bounded_decomposition_input(source_mesh, budgets)
                            )
                            decomposition_vertices = np.asarray(
                                decomposition_mesh.vertices,
                                dtype=np.float64,
                            )
                            decomposition_triangles = np.asarray(
                                decomposition_mesh.faces,
                                dtype=np.int64,
                            )
                            threshold_m = _coacd_threshold_m(
                                mesh.path,
                                measured.bbox_diagonal_m or 0.0,
                                budgets,
                                mesh_protections,
                            )
                            coarse_threshold_m = _coacd_threshold_m(
                                mesh.path,
                                measured.bbox_diagonal_m or 0.0,
                                budgets,
                                mesh_protections,
                                target_ratio=min(
                                    0.04,
                                    budgets.max_collision_surface_distance_ratio * 0.8,
                                ),
                            )
                            coacd_candidate_produced = False
                            decomposition_deadline = time.monotonic() + min(
                                max(timeout_s - (time.monotonic() - started), 0.0),
                                budgets.max_collision_decomposition_time_s,
                            )
                            for variant, hull_limit, variant_threshold_m in _coacd_search_schedule(
                                maximum_hulls=body_hull_budget,
                                threshold_m=threshold_m,
                                coarse_threshold_m=coarse_threshold_m,
                            ):
                                remaining_time = min(
                                    timeout_s - (time.monotonic() - started),
                                    decomposition_deadline - time.monotonic(),
                                )
                                variant_name = _safe_name(mesh.path, mesh_index)
                                if variant != "coarse":
                                    variant_name = f"{variant_name}_{variant}"
                                if remaining_time <= 0.0:
                                    result_path = (
                                        output.parent
                                        / "collision_decomposition"
                                        / f"{variant_name}_timeout.json"
                                    )
                                    atomic_write_json(
                                        result_path,
                                        {
                                            "schema_version": "geometry-repair.coacd-result.v1",
                                            "status": "fail",
                                            "parts": [],
                                            "failures": [
                                                "no job or per-body wall-clock budget remained "
                                                "for CoACD"
                                            ],
                                        },
                                    )
                                    coacd_payloads = []
                                    evidence = None
                                    coacd_error = (
                                        "no job or per-body wall-clock budget remained for CoACD"
                                    )
                                else:
                                    try:
                                        (
                                            coacd_payloads,
                                            evidence,
                                            result_path,
                                            coacd_error,
                                        ) = _run_coacd(
                                            source_path=mesh.path,
                                            vertices=decomposition_vertices,
                                            triangles=decomposition_triangles,
                                            work_dir=(
                                                output.parent
                                                / "collision_decomposition"
                                                / variant_name
                                            ),
                                            threshold_m=variant_threshold_m,
                                            max_hulls=hull_limit,
                                            max_vertices=(budgets.max_collision_vertices_per_hull),
                                            max_faces=budgets.max_collision_faces_per_hull,
                                            seed=(deterministic_seed + mesh_index) % (2**31),
                                            memory_mb=budgets.max_memory_mb,
                                            timeout_s=remaining_time,
                                            source_triangle_count=len(welded_triangles),
                                            preprocessing=decomposition_preprocessing,
                                        )
                                    except Exception as exc:
                                        coacd_payloads = []
                                        evidence = None
                                        result_path = (
                                            output.parent
                                            / "collision_decomposition"
                                            / f"{variant_name}_exception.json"
                                        )
                                        atomic_write_json(
                                            result_path,
                                            {
                                                "schema_version": (
                                                    "geometry-repair.coacd-result.v1"
                                                ),
                                                "status": "fail",
                                                "parts": [],
                                                "failures": [f"{type(exc).__name__}: {exc}"],
                                            },
                                        )
                                        coacd_error = f"{type(exc).__name__}: {exc}"
                                decomposition_report_paths.append(str(result_path.resolve()))
                                if not coacd_payloads or evidence is None:
                                    warnings.append(
                                        f"{mesh.path}: CoACD {variant} candidate was unavailable "
                                        f"({coacd_error}); report: {result_path.resolve()}"
                                    )
                                    continue
                                coacd_candidate_produced = True
                                decompositions.append(evidence)
                                warnings.extend(
                                    f"{mesh.path}: {warning}" for warning in evidence.warnings
                                )
                                candidate_id = f"{candidate_prefix}_coacd"
                                if variant != "coarse":
                                    candidate_id = f"{candidate_id}_{variant}"
                                try:
                                    coacd_evaluation, _coacd_metrics = _evaluate_collision_option(
                                        candidate_id=candidate_id,
                                        representation="coacd",
                                        generator="coacd_collision",
                                        source_mesh=source_mesh,
                                        payloads=coacd_payloads,
                                        diagonal_m=measured.bbox_diagonal_m or 0.0,
                                        budgets=budgets,
                                        remaining_hull_budget=body_hull_budget,
                                        protected_features=collision_sensitive,
                                        work_dir=candidate_work_dir,
                                    )
                                    candidate_evaluations.append(coacd_evaluation)
                                    candidate_payloads[candidate_id] = coacd_payloads
                                    candidate_generators[candidate_id] = "coacd_collision"
                                    if coacd_evaluation.status != "fail":
                                        break
                                except Exception as exc:
                                    warnings.append(
                                        f"{mesh.path}: CoACD {variant} candidate evidence failed: "
                                        f"{type(exc).__name__}: {exc}"
                                    )
                            if not coacd_candidate_produced:
                                message = "CoACD did not produce any measurable candidate"
                                candidate_generation_failures.append(message)
                                warnings.append(f"{mesh.path}: {message}")

                candidate_search_path = candidate_work_dir / "candidate_search.json"
                candidate_search = select_collision_candidate(
                    mesh.path,
                    candidate_evaluations,
                    report_path=candidate_search_path,
                    failure_context=candidate_generation_failures,
                )
                candidate_search_paths.append(str(candidate_search_path.resolve()))
                if candidate_search.selected_candidate_id is None:
                    failures.extend(
                        f"{mesh.path}: {reason}" for reason in candidate_search.failures
                    )
                    continue
                warnings.extend(f"{mesh.path}: {warning}" for warning in candidate_search.warnings)
                selected_id = candidate_search.selected_candidate_id
                selected = candidate_payloads[selected_id]
                selected_generator = candidate_generators[selected_id]
                if selected_id in candidate_primitive_kinds:
                    primitive_fit_kinds.append(candidate_primitive_kinds[selected_id])

                selected_meshes = [
                    trimesh.Trimesh(
                        vertices=selected_vertices,
                        faces=selected_triangles,
                        process=False,
                    )
                    for _, selected_vertices, selected_triangles, _ in selected
                ]
                try:
                    approximation = _approximation_metrics(source_mesh, selected_meshes)
                except Exception as exc:
                    failures.append(
                        f"{mesh.path}: collision approximation error could not be measured: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    continue
                volume_excess = approximation["volume_excess_ratio"]
                volume_deficit = approximation["volume_deficit_ratio"]
                maximum_volume_excess = max(maximum_volume_excess or 0.0, volume_excess)
                maximum_volume_deficit = max(maximum_volume_deficit or 0.0, volume_deficit)
                maximum_surface_overreach = max(
                    maximum_surface_overreach or 0.0,
                    approximation["surface_overreach_m"],
                )
                maximum_surface_gap = max(
                    maximum_surface_gap or 0.0,
                    approximation["surface_gap_m"],
                )
                maximum_false_positive = max(
                    maximum_false_positive or 0.0,
                    approximation["false_positive_ratio"],
                )
                maximum_false_negative = max(
                    maximum_false_negative or 0.0,
                    approximation["false_negative_ratio"],
                )
                if volume_excess > budgets.max_collision_volume_excess:
                    failures.append(
                        f"{mesh.path}: selected collision volume excess {volume_excess:.4f} "
                        f"exceeds {budgets.max_collision_volume_excess:.4f}"
                    )
                if volume_deficit > budgets.max_collision_volume_deficit:
                    failures.append(
                        f"{mesh.path}: selected collision volume deficit {volume_deficit:.4f} "
                        f"exceeds {budgets.max_collision_volume_deficit:.4f}"
                    )
                if approximation["false_negative_ratio"] > budgets.max_collision_volume_deficit:
                    failures.append(
                        f"{mesh.path}: collision false-negative ratio "
                        f"{approximation['false_negative_ratio']:.4f} exceeds "
                        f"{budgets.max_collision_volume_deficit:.4f}"
                    )
                if approximation["false_positive_ratio"] > budgets.max_collision_volume_excess:
                    failures.append(
                        f"{mesh.path}: collision false-positive ratio "
                        f"{approximation['false_positive_ratio']:.4f} exceeds "
                        f"{budgets.max_collision_volume_excess:.4f}"
                    )
                surface_limit = max(
                    (measured.bbox_diagonal_m or 0.0)
                    * budgets.max_collision_surface_distance_ratio,
                    1e-6,
                )
                if approximation["surface_gap_m"] > surface_limit:
                    failures.append(
                        f"{mesh.path}: collision surface gap "
                        f"{approximation['surface_gap_m']:.6g} m exceeds "
                        f"{surface_limit:.6g} m"
                    )
                if approximation["surface_overreach_m"] > surface_limit:
                    failures.append(
                        f"{mesh.path}: collision surface overreach "
                        f"{approximation['surface_overreach_m']:.6g} m exceeds "
                        f"{surface_limit:.6g} m"
                    )
                payloads.extend(selected)
                hull_count += len(selected)
                source_collision_prim_count += sum(
                    item[3] == "source_collision_hull" for item in selected
                )
                generated_collision_prim_count += sum(
                    item[3] != "source_collision_hull" for item in selected
                )
                maximum_vertices = max(
                    maximum_vertices,
                    *(len(item[1]) for item in selected),
                )
                maximum_faces = max(
                    maximum_faces,
                    *(len(item[2]) for item in selected),
                )
                generators.add(selected_generator)
            if generated_collision_prim_count > budgets.max_collision_hulls:
                failures.append(
                    "generated collision hull count "
                    f"{generated_collision_prim_count} exceeds {budgets.max_collision_hulls}"
                )
        elif profile in {"articulated_rigid", "contact_rich"}:
            representation = "source_collision"
            if not source_collision_meshes:
                warnings.append(
                    f"{profile} requires caller-mapped source collision geometry; none was found"
                )
            elif len(source_collision_meshes) > budgets.max_source_collision_prims:
                failures.append(
                    "source collision prim count "
                    f"{len(source_collision_meshes)} exceeds intake resource limit "
                    f"{budgets.max_source_collision_prims}"
                )
            else:
                preserve_source_collision_paths = True
                for mesh in source_collision_meshes:
                    vertices, triangles = positional_topology_mesh(
                        mesh.world_vertices_m,
                        mesh.triangles,
                    )
                    if not len(vertices) or not len(triangles):
                        warnings.append(f"{mesh.path}: empty source collision mesh was omitted")
                        continue
                    payloads.append(
                        (
                            mesh.path,
                            vertices,
                            triangles,
                            "source_collision_hull",
                        )
                    )
                    maximum_vertices = max(maximum_vertices, len(vertices))
                    maximum_faces = max(maximum_faces, len(triangles))
                hull_count = len(payloads)
                source_collision_prim_count = len(payloads)
                generators.add("geometry_repair.source_collision_reuse")
                warnings.append(
                    f"{profile} collision was preserved for profile-specific geometry evidence; "
                    "joint/contact runtime validation remains downstream-owned"
                )
        else:
            representation = "none"
            failures.append(f"collision certification for profile {profile!r} is not implemented")
        if payloads:
            if profile == "rigid_pick_place":
                payload_kinds = {item[3] for item in payloads}
                if all(kind.startswith("primitive_") for kind in payload_kinds):
                    representation = "primitive_fit"
                elif payload_kinds == {"convex_hull"}:
                    representation = "convex_hull"
                elif payload_kinds == {"coacd_convex_hull"}:
                    representation = "coacd"
                elif payload_kinds == {"source_collision_hull"}:
                    representation = "source_collision"
                else:
                    representation = "hybrid"
            _write_collision_stage(
                output,
                payloads,
                representation=representation,
                preserve_source_paths=preserve_source_collision_paths,
            )
            protected_feature_probes = evaluate_feature_probes(
                output,
                protected_features or [],
            )
            _record_protected_feature_probe_outcomes(
                protected_features=protected_features or [],
                probes=protected_feature_probes,
                protected_features_unmeasured=protected_features_unmeasured,
                failures=failures,
                warnings=warnings,
            )
        runtime_status: Literal["pass", "fail", "not_evaluated"] = "not_evaluated"
        runtime_report_path = None
        temporary_proxy_path = None
        runtime_scenarios = []
        if (
            not failures
            and payloads
            and runtime_engine in {"fake", "ovphysx"}
            and profile == "rigid_pick_place"
        ):
            try:
                runtime = validate_collision_runtime(
                    output,
                    output.parent / "collision_runtime",
                    engine=runtime_engine,
                )
                runtime_status = runtime["status"]
                runtime_report_path = runtime["runtime_report_path"]
                temporary_proxy_path = runtime["temporary_proxy_path"]
                runtime_scenarios = runtime.get("runtime_scenarios", [])
                failures.extend(runtime["failures"])
                warnings.extend(runtime["warnings"])
                if runtime_engine == "fake" and runtime_status == "pass":
                    warnings.append(
                        "fake collision runtime evidence is test-only and cannot certify a production asset"
                    )
            except Exception as exc:
                runtime_status = "fail"
                failures.append(f"collision runtime validation failed: {type(exc).__name__}: {exc}")
        elif (
            not failures
            and payloads
            and runtime_engine in {"fake", "ovphysx"}
            and profile == "static_environment"
        ):
            try:
                runtime = validate_static_collision_runtime(
                    output,
                    output.parent / "collision_runtime",
                    engine=runtime_engine,
                )
                runtime_status = runtime["status"]
                runtime_report_path = runtime["runtime_report_path"]
                temporary_proxy_path = runtime["temporary_proxy_path"]
                runtime_scenarios = runtime.get("runtime_scenarios", [])
                failures.extend(runtime["failures"])
                warnings.extend(runtime["warnings"])
                if runtime_engine == "fake" and runtime_status == "pass":
                    warnings.append(
                        "fake collision runtime evidence is test-only and cannot certify a production asset"
                    )
            except Exception as exc:
                runtime_status = "fail"
                failures.append(
                    f"static collision runtime validation failed: {type(exc).__name__}: {exc}"
                )
        elif not failures and payloads:
            warnings.append("collision geometry was not cooked or stepped in a target runtime")
        source_render_sha256_after = file_sha256(render)
        source_render_unchanged = source_render_sha256_after == source_render_sha256_before
        if not source_render_unchanged:
            failures.append("collision planning altered the immutable render geometry source")
        handoff_available = bool(payloads) and not failures
        if not handoff_available:
            representation = "none"
            output.unlink(missing_ok=True)
        status = "fail" if failures else "conditional" if warnings else "pass"
        if generators == {"trimesh_convex_hull_collision"}:
            generator = "trimesh_convex_hull_collision"
            generator_version = metadata.version("trimesh")
        elif generators == {"coacd_collision"}:
            generator = "coacd_collision"
            generator_version = metadata.version("coacd")
        elif generators == {"geometry_repair.static_triangle_collision"}:
            generator = "geometry_repair.static_triangle_collision"
            generator_version = "source-tree"
        elif generators == {"geometry_repair.primitive_fit"}:
            generator = "geometry_repair.primitive_fit"
            generator_version = f"source-tree;trimesh={metadata.version('trimesh')}"
        elif generators == {"geometry_repair.bounded_convex_hull_reduction"}:
            generator = "geometry_repair.bounded_convex_hull_reduction"
            generator_version = (
                "source-tree;"
                f"trimesh={metadata.version('trimesh')};"
                f"fast-simplification={metadata.version('fast-simplification')}"
            )
        elif generators == {"geometry_repair.source_collision_reuse"}:
            generator = "geometry_repair.source_collision_reuse"
            generator_version = "source-authored; independently revalidated"
        elif generators:
            generator = "geometry_repair.hybrid_collision"
            generator_version = (
                f"trimesh={metadata.version('trimesh')};coacd={metadata.version('coacd')}"
            )
        else:
            generator = None
            generator_version = None
        if not handoff_available:
            generator = None
            generator_version = None
        report = CollisionReport(
            status=status,
            representation=representation,  # type: ignore[arg-type]
            source_render_path=str(render),
            source_render_sha256=source_render_sha256_before,
            source_render_sha256_after=source_render_sha256_after,
            source_render_unchanged=source_render_unchanged,
            collision_path=str(output) if handoff_available else None,
            collision_sha256=file_sha256(output) if handoff_available else None,
            generator=generator,
            generator_version=generator_version,
            source_body_count=len(meshes),
            hull_count=hull_count,
            source_collision_prim_count=source_collision_prim_count,
            generated_collision_prim_count=generated_collision_prim_count,
            total_vertices=sum(len(item[1]) for item in payloads),
            total_faces=sum(len(item[2]) for item in payloads),
            collision_size_bytes=output.stat().st_size if handoff_available else 0,
            maximum_vertices_per_hull=maximum_vertices,
            maximum_faces_per_hull=maximum_faces,
            maximum_volume_excess_ratio=maximum_volume_excess,
            maximum_volume_deficit_ratio=maximum_volume_deficit,
            maximum_surface_overreach_m=maximum_surface_overreach,
            maximum_surface_gap_m=maximum_surface_gap,
            maximum_false_positive_ratio=maximum_false_positive,
            maximum_false_negative_ratio=maximum_false_negative,
            primitive_fit_count=len(primitive_fit_kinds),
            primitive_fit_kinds=primitive_fit_kinds,
            candidate_search_paths=candidate_search_paths,
            decompositions=decompositions,
            decomposition_report_paths=decomposition_report_paths,
            collision_reconstruction_count=collision_reconstruction_count,
            collision_reconstruction_paths=collision_reconstruction_paths,
            protected_features_applied=sorted(
                protected_features_applied & explicit_protected_feature_names
            ),
            protected_features_unmeasured=sorted(
                protected_features_unmeasured & explicit_protected_feature_names
            ),
            protected_feature_probes=[
                probe
                for probe in protected_feature_probes
                if probe.feature_name in explicit_protected_feature_names
            ],
            inferred_protected_features_applied=sorted(
                protected_features_applied & inferred_protected_feature_names
            ),
            inferred_protected_features_unmeasured=sorted(
                protected_features_unmeasured & inferred_protected_feature_names
            ),
            inferred_protected_feature_probes=[
                probe
                for probe in protected_feature_probes
                if probe.feature_name in inferred_protected_feature_names
            ],
            role_excluded_protected_features=role_excluded_protected_features,
            warnings=warnings,
            failures=failures,
            runtime_status=runtime_status,
            runtime_engine=runtime_engine,
            runtime_report_path=runtime_report_path,
            temporary_proxy_path=temporary_proxy_path,
            runtime_scenarios=runtime_scenarios,
            report_path=str(Path(report_path).resolve()) if report_path else None,
            source_collision_audit_path=(
                str(resolved_source_collision_audit_path)
                if resolved_source_collision_audit_path is not None
                and resolved_source_collision_audit_path.is_file()
                else None
            ),
        )
    if report_path:
        atomic_write_json(report_path, report)
    return report


def compose_asset(
    render_path: str | Path,
    collision_path: str | Path | None,
    output_path: str | Path,
) -> Path:
    """Compose a reference-safe asset while retaining representation ownership.

    A sublayer-only composition leaves render and collision as sibling root
    prims.  Referencing that file through its default prim then imports only the
    render root, which silently drops generated collision in standard USD and
    Isaac Lab spawn paths.  Keep each representation in its own source layer,
    but reference both beneath one component root that is safe to reference as
    a complete asset.
    """

    from pxr import Usd, UsdGeom, UsdPhysics

    render = Path(render_path).expanduser().resolve()
    collision = Path(collision_path).expanduser().resolve() if collision_path else None
    output = Path(output_path).expanduser().resolve()
    if output == render or output == collision:
        raise ValueError("composed asset must not overwrite a representation layer")
    if collision is not None and not collision.is_file():
        raise FileNotFoundError(f"collision layer does not exist: {collision}")
    output.parent.mkdir(parents=True, exist_ok=True)
    render_stage = Usd.Stage.Open(str(render))
    if render_stage is None or not render_stage.GetDefaultPrim():
        raise ValueError(f"render layer has no default prim: {render}")

    output.unlink(missing_ok=True)
    stage = Usd.Stage.CreateNew(str(output))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.GetStageUpAxis(render_stage))
    UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.GetStageMetersPerUnit(render_stage))
    root = UsdGeom.Xform.Define(stage, "/GeometryRepairAsset")
    stage.SetDefaultPrim(root.GetPrim())
    root.GetPrim().SetCustomDataByKey("geometryRepairComposition", "reference_safe_v1")

    # A typeless over preserves the referenced default prim's schema. Defining
    # an Xform here would silently turn a source whose default prim is a Mesh
    # into an Xform with mesh attributes.
    render_prim = stage.OverridePrim("/GeometryRepairAsset/Render")
    render_prim.GetReferences().AddReference(
        _portable_layer_path(render, output.parent),
        render_stage.GetDefaultPrim().GetPath(),
    )
    render_prim.SetCustomDataByKey("geometryRepairRepresentation", "render")
    disabled_source_colliders = 0
    if collision is not None:
        render_root = "/GeometryRepairAsset/Render"
        for prim in stage.Traverse():
            path_text = str(prim.GetPath())
            if (
                path_text == render_root or path_text.startswith(f"{render_root}/")
            ) and prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr().Set(False)
                prim.SetCustomDataByKey(
                    "geometryRepairCollisionRole",
                    "inactive_source_evidence",
                )
                disabled_source_colliders += 1
    root.GetPrim().SetCustomDataByKey(
        "geometryRepairDisabledSourceColliderCount",
        disabled_source_colliders,
    )

    if collision is not None:
        collision_stage = Usd.Stage.Open(str(collision))
        if collision_stage is None or not collision_stage.GetDefaultPrim():
            raise ValueError(f"collision layer has no default prim: {collision}")
        collision_prim = stage.OverridePrim("/GeometryRepairAsset/Collision")
        collision_prim.GetReferences().AddReference(
            _portable_layer_path(collision, output.parent),
            collision_stage.GetDefaultPrim().GetPath(),
        )
        collision_prim.SetCustomDataByKey("geometryRepairRepresentation", "collision")

    stage.GetRootLayer().Save()
    return output


def _portable_layer_path(layer_path: Path, package_dir: Path) -> str:
    """Return a relocatable asset path when the layer is inside the package."""

    try:
        return layer_path.relative_to(package_dir).as_posix()
    except ValueError:
        return str(layer_path)
