# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inspectable source collision-proxy fidelity, task, and complexity audit."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .artifacts import atomic_write_json, file_sha256
from .compound_collision import (
    CompoundCollisionPartAudit,
    pair_compound_collision_parts,
)
from .mesh_io import load_meshes, positional_topology_mesh
from .models import ProtectedFeature, ProtectedFeatureProbeResult
from .protected_features import evaluate_feature_probes

SOURCE_COLLISION_AUDIT_SCHEMA_VERSION = "geometry-repair.source-collision-audit.v1"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CollisionAuditLimits(_StrictModel):
    """Profile limits; source proxy count is deliberately advisory only."""

    min_render_surface_coverage_ratio: float = Field(default=0.99, ge=0.0, le=1.0)
    max_volume_excess_ratio: float = Field(default=0.10, ge=0.0)
    max_volume_deficit_ratio: float = Field(default=0.01, ge=0.0)
    max_false_positive_ratio: float = Field(default=0.10, ge=0.0)
    max_false_negative_ratio: float = Field(default=0.01, ge=0.0)
    max_surface_gap_m: float | None = Field(default=None, ge=0.0)
    max_surface_overreach_m: float | None = Field(default=None, ge=0.0)
    advisory_hull_count: int | None = Field(default=16, ge=1)
    max_total_vertices: int | None = Field(default=None, ge=1)
    max_total_faces: int | None = Field(default=None, ge=1)
    max_vertices_per_hull: int | None = Field(default=255, ge=4)
    max_faces_per_hull: int | None = Field(default=255, ge=4)
    max_collision_size_bytes: int | None = Field(default=None, ge=1)
    max_cook_time_s: float | None = Field(default=None, gt=0.0)
    max_runtime_contact_cost: float | None = Field(default=None, gt=0.0)


class CollisionComplexity(_StrictModel):
    """Source proxy complexity measured independently from geometric fidelity."""

    hull_count: int = Field(ge=0)
    total_vertices: int = Field(ge=0)
    total_faces: int = Field(ge=0)
    maximum_vertices_per_hull: int = Field(default=0, ge=0)
    maximum_faces_per_hull: int = Field(default=0, ge=0)
    collision_size_bytes: int = Field(default=0, ge=0)
    cook_time_s: float | None = Field(default=None, ge=0.0)
    runtime_contact_cost: float | None = Field(default=None, ge=0.0)


class CollisionAuditMetrics(_StrictModel):
    """Render/collision union measurements used by explicit acceptance gates."""

    render_surface_sample_count: int = Field(default=0, ge=0)
    collision_surface_sample_count: int = Field(default=0, ge=0)
    render_surface_coverage_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    surface_gap_p99_m: float | None = Field(default=None, ge=0.0)
    surface_gap_max_m: float | None = Field(default=None, ge=0.0)
    surface_overreach_p99_m: float | None = Field(default=None, ge=0.0)
    surface_overreach_max_m: float | None = Field(default=None, ge=0.0)
    false_positive_ratio: float | None = Field(default=None, ge=0.0)
    false_negative_ratio: float | None = Field(default=None, ge=0.0)
    volume_excess_ratio: float | None = Field(default=None, ge=0.0)
    volume_deficit_ratio: float | None = Field(default=None, ge=0.0)
    occupancy_grid_resolution: int | None = Field(default=None, ge=2)
    occupancy_estimated_face_point_product: int | None = Field(default=None, ge=0)
    occupancy_face_point_limit: int | None = Field(default=None, ge=1)
    occupancy_status: Literal["evaluated", "not_evaluated"] = "not_evaluated"
    occupancy_reason: str | None = None
    per_render_part_coverage: list[dict[str, Any]] = Field(default_factory=list)


class CollisionAuditGate(_StrictModel):
    """One independently inspectable source-proxy acceptance predicate."""

    gate_id: str
    category: Literal["fidelity", "protected_space", "task", "complexity"]
    status: Literal["pass", "fail", "warning", "not_evaluated"]
    blocking: bool
    measured_value: float | int | str | None = None
    limit_value: float | int | str | None = None
    comparison: Literal[">=", "<=", "probe", "informational"]
    reason: str


class SourceCollisionAudit(_StrictModel):
    """Decision evidence for preserving or selectively regenerating source proxies."""

    schema_version: Literal["geometry-repair.source-collision-audit.v1"] = (
        SOURCE_COLLISION_AUDIT_SCHEMA_VERSION
    )
    status: Literal["pass", "conditional", "fail", "not_evaluated"]
    decision: Literal["preserve_source", "regenerate_candidate", "review_required"]
    render_path: str
    render_sha256: str
    collision_path: str
    collision_sha256: str
    metrics: CollisionAuditMetrics
    complexity: CollisionComplexity
    limits: CollisionAuditLimits
    gates: list[CollisionAuditGate] = Field(default_factory=list)
    protected_feature_probes: list[ProtectedFeatureProbeResult] = Field(default_factory=list)
    regeneration_reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    report_path: str | None = None


def _numeric_gate(
    *,
    gate_id: str,
    category: Literal["fidelity", "complexity"],
    measured: float | int | None,
    limit: float | int | None,
    comparison: Literal[">=", "<="],
    blocking: bool = True,
) -> CollisionAuditGate:
    if measured is None or limit is None:
        return CollisionAuditGate(
            gate_id=gate_id,
            category=category,
            status="not_evaluated",
            blocking=blocking,
            measured_value=measured,
            limit_value=limit,
            comparison=comparison,
            reason=f"{gate_id} has no measured value or configured limit",
        )
    passed = measured >= limit if comparison == ">=" else measured <= limit
    status: Literal["pass", "fail", "warning"]
    if passed:
        status = "pass"
    elif blocking:
        status = "fail"
    else:
        status = "warning"
    return CollisionAuditGate(
        gate_id=gate_id,
        category=category,
        status=status,
        blocking=blocking,
        measured_value=measured,
        limit_value=limit,
        comparison=comparison,
        reason=(
            f"{gate_id} measured {measured:.6g} and must be {comparison} {limit:.6g}"
            if isinstance(measured, int | float) and isinstance(limit, int | float)
            else f"{gate_id} did not meet its configured limit"
        ),
    )


def evaluate_source_collision_gates(
    *,
    render_path: str | Path,
    collision_path: str | Path,
    metrics: CollisionAuditMetrics,
    complexity: CollisionComplexity,
    limits: CollisionAuditLimits,
    protected_feature_probes: list[ProtectedFeatureProbeResult] | None = None,
    required_feature_names: set[str] | None = None,
    report_path: str | Path | None = None,
) -> SourceCollisionAudit:
    """Evaluate source-proxy metrics without allowing count alone to reject it."""

    render = Path(render_path).expanduser().resolve()
    collision = Path(collision_path).expanduser().resolve()
    gates = [
        _numeric_gate(
            gate_id="render_surface_coverage",
            category="fidelity",
            measured=metrics.render_surface_coverage_ratio,
            limit=limits.min_render_surface_coverage_ratio,
            comparison=">=",
        ),
        _numeric_gate(
            gate_id="collision_volume_excess",
            category="fidelity",
            measured=metrics.volume_excess_ratio,
            limit=limits.max_volume_excess_ratio,
            comparison="<=",
        ),
        _numeric_gate(
            gate_id="collision_volume_deficit",
            category="fidelity",
            measured=metrics.volume_deficit_ratio,
            limit=limits.max_volume_deficit_ratio,
            comparison="<=",
        ),
        _numeric_gate(
            gate_id="false_positive_cavity_gap_occupation",
            category="fidelity",
            measured=metrics.false_positive_ratio,
            limit=limits.max_false_positive_ratio,
            comparison="<=",
        ),
        _numeric_gate(
            gate_id="false_negative_render_occupation",
            category="fidelity",
            measured=metrics.false_negative_ratio,
            limit=limits.max_false_negative_ratio,
            comparison="<=",
        ),
        _numeric_gate(
            gate_id="surface_gap_p99_m",
            category="fidelity",
            measured=metrics.surface_gap_p99_m,
            limit=limits.max_surface_gap_m,
            comparison="<=",
        ),
        _numeric_gate(
            gate_id="surface_overreach_p99_m",
            category="fidelity",
            measured=metrics.surface_overreach_p99_m,
            limit=limits.max_surface_overreach_m,
            comparison="<=",
        ),
    ]
    if limits.advisory_hull_count is not None:
        gates.append(
            _numeric_gate(
                gate_id="source_proxy_hull_count",
                category="complexity",
                measured=complexity.hull_count,
                limit=limits.advisory_hull_count,
                comparison="<=",
                blocking=False,
            )
        )
    for gate_id, measured, limit in (
        ("total_collision_vertices", complexity.total_vertices, limits.max_total_vertices),
        ("total_collision_faces", complexity.total_faces, limits.max_total_faces),
        (
            "maximum_vertices_per_hull",
            complexity.maximum_vertices_per_hull,
            limits.max_vertices_per_hull,
        ),
        (
            "maximum_faces_per_hull",
            complexity.maximum_faces_per_hull,
            limits.max_faces_per_hull,
        ),
        (
            "collision_package_size_bytes",
            complexity.collision_size_bytes,
            limits.max_collision_size_bytes,
        ),
        ("collision_cook_time_s", complexity.cook_time_s, limits.max_cook_time_s),
        (
            "runtime_contact_cost",
            complexity.runtime_contact_cost,
            limits.max_runtime_contact_cost,
        ),
    ):
        if limit is not None:
            gates.append(
                _numeric_gate(
                    gate_id=gate_id,
                    category="complexity",
                    measured=measured,
                    limit=limit,
                    comparison="<=",
                )
            )

    required = required_feature_names or set()
    for result in sorted(protected_feature_probes or [], key=lambda item: item.feature_name):
        is_required = result.feature_name in required
        if result.status == "pass":
            gate_status: Literal["pass", "fail", "warning", "not_evaluated"] = "pass"
            reason = "protected task probe passed"
        elif result.status == "fail":
            gate_status = "fail" if is_required else "warning"
            reason = "; ".join(result.failures or ["protected task probe failed"])
        else:
            gate_status = "not_evaluated" if is_required else "warning"
            reason = "; ".join(result.warnings or ["protected task probe was not evaluated"])
        gates.append(
            CollisionAuditGate(
                gate_id=f"protected_feature:{result.feature_name}",
                category="protected_space"
                if result.probe_kind == "negative_space_path"
                else "task",
                status=gate_status,
                blocking=is_required,
                measured_value=result.status,
                comparison="probe",
                reason=reason,
            )
        )

    blocking_failures = [gate for gate in gates if gate.blocking and gate.status == "fail"]
    indeterminate = [gate for gate in gates if gate.blocking and gate.status == "not_evaluated"]
    advisory = [gate for gate in gates if not gate.blocking and gate.status == "warning"]
    if blocking_failures:
        decision: Literal["preserve_source", "regenerate_candidate", "review_required"] = (
            "regenerate_candidate"
        )
        status: Literal["pass", "conditional", "fail", "not_evaluated"] = "fail"
    elif indeterminate:
        decision = "review_required"
        status = "not_evaluated"
    else:
        decision = "preserve_source"
        status = "conditional" if advisory else "pass"
    regeneration_reasons = [gate.reason for gate in blocking_failures]
    warnings = [
        f"{gate.gate_id}: {gate.reason} (advisory only; hull count alone never rejects source)"
        if gate.gate_id == "source_proxy_hull_count"
        else f"{gate.gate_id}: {gate.reason}"
        for gate in advisory
    ]
    report = SourceCollisionAudit(
        status=status,
        decision=decision,
        render_path=str(render),
        render_sha256=file_sha256(render),
        collision_path=str(collision),
        collision_sha256=file_sha256(collision),
        metrics=metrics,
        complexity=complexity,
        limits=limits,
        gates=gates,
        protected_feature_probes=protected_feature_probes or [],
        regeneration_reasons=regeneration_reasons,
        warnings=warnings,
        report_path=str(Path(report_path).expanduser().resolve()) if report_path else None,
    )
    if report_path:
        atomic_write_json(report_path, report)
    return report


def _as_trimesh(mesh_data):
    import trimesh

    vertices, faces = positional_topology_mesh(
        mesh_data.world_vertices_m,
        mesh_data.triangles,
    )
    if not len(vertices) or not len(faces):
        return None
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _occupancy_metrics(
    render_meshes: list,
    collision_meshes: list,
    *,
    resolution: int,
    face_point_limit: int,
) -> dict[str, float | int | str | None]:
    from .collision import _contains_bounded

    if not render_meshes or not collision_meshes:
        return {"status": "not_evaluated", "reason": "render or collision union is empty"}
    if any(not mesh.is_watertight for mesh in render_meshes):
        return {
            "status": "not_evaluated",
            "reason": "render occupancy requires every audited render part to be watertight",
        }
    if any(not mesh.is_watertight for mesh in collision_meshes):
        return {
            "status": "not_evaluated",
            "reason": "collision occupancy requires every source proxy to be watertight",
        }
    face_count = sum(len(mesh.faces) for mesh in [*render_meshes, *collision_meshes])
    if face_count <= 0:
        return {"status": "not_evaluated", "reason": "occupancy meshes contain no faces"}
    maximum_point_count = face_point_limit // face_count
    maximum_resolution = int(np.floor(np.cbrt(maximum_point_count)))
    while (maximum_resolution + 1) ** 3 <= maximum_point_count:
        maximum_resolution += 1
    while maximum_resolution > 0 and maximum_resolution**3 > maximum_point_count:
        maximum_resolution -= 1
    if maximum_resolution < 8:
        requested_estimate = int(face_count * resolution**3)
        return {
            "status": "not_evaluated",
            "reason": (
                "occupancy face-point estimate "
                f"{requested_estimate} exceeds budget {face_point_limit}; "
                "even the minimum 8^3 grid exceeds the budget"
            ),
            "estimated_face_point_product": requested_estimate,
            "face_point_limit": face_point_limit,
        }
    effective_resolution = min(resolution, maximum_resolution)
    vertices = np.vstack(
        [
            *(np.asarray(mesh.vertices, dtype=np.float64) for mesh in render_meshes),
            *(np.asarray(mesh.vertices, dtype=np.float64) for mesh in collision_meshes),
        ]
    )
    minimum = vertices.min(axis=0)
    maximum = vertices.max(axis=0)
    extent = maximum - minimum
    padding = np.maximum(extent * 0.02, 1e-9)
    minimum -= padding
    maximum += padding
    axes = [
        np.linspace(minimum[index], maximum[index], num=effective_resolution, endpoint=False)
        + (maximum[index] - minimum[index]) / (2.0 * effective_resolution)
        for index in range(3)
    ]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape((-1, 3))
    estimated_face_point_product = int(face_count * len(grid))
    render_occupied = np.zeros(len(grid), dtype=bool)
    collision_occupied = np.zeros(len(grid), dtype=bool)
    try:
        for mesh in render_meshes:
            render_occupied |= _contains_bounded(mesh, grid)
        for mesh in collision_meshes:
            collision_occupied |= _contains_bounded(mesh, grid)
    except Exception as exc:
        return {
            "status": "not_evaluated",
            "reason": f"occupancy backend failed: {type(exc).__name__}: {exc}",
        }
    render_count = int(np.count_nonzero(render_occupied))
    if render_count == 0:
        return {"status": "not_evaluated", "reason": "render occupancy grid is empty"}
    collision_count = int(np.count_nonzero(collision_occupied))
    false_positive = int(np.count_nonzero(collision_occupied & ~render_occupied))
    false_negative = int(np.count_nonzero(render_occupied & ~collision_occupied))
    return {
        "status": "evaluated",
        "reason": (
            f"occupancy resolution reduced from {resolution} to {effective_resolution} "
            "to satisfy the face-point budget"
            if effective_resolution != resolution
            else None
        ),
        "resolution": effective_resolution,
        "estimated_face_point_product": estimated_face_point_product,
        "face_point_limit": face_point_limit,
        "false_positive_ratio": false_positive / render_count,
        "false_negative_ratio": false_negative / render_count,
        "volume_excess_ratio": max(0, collision_count - render_count) / render_count,
        "volume_deficit_ratio": max(0, render_count - collision_count) / render_count,
    }


def _measure_collision_metrics(
    render_pairs: list[tuple[Any, Any]],
    collision_pairs: list[tuple[Any, Any]],
    *,
    limits: CollisionAuditLimits,
    surface_sample_limit: int,
    occupancy_grid_resolution: int,
    occupancy_face_point_limit: int,
) -> CollisionAuditMetrics:
    import trimesh

    from .collision import _closest_union_distance, _surface_points

    render_meshes = [mesh for _record, mesh in render_pairs]
    collision_meshes = [mesh for _record, mesh in collision_pairs]
    render_distance_meshes = (
        []
        if not render_meshes
        else [render_meshes[0]]
        if len(render_meshes) == 1
        else [trimesh.util.concatenate(render_meshes)]
    )
    collision_distance_meshes = (
        []
        if not collision_meshes
        else [collision_meshes[0]]
        if len(collision_meshes) == 1
        else [trimesh.util.concatenate(collision_meshes)]
    )
    all_vertices = [np.asarray(mesh.vertices, dtype=np.float64) for mesh in render_meshes]
    diagonal = (
        float(np.linalg.norm(np.ptp(np.vstack(all_vertices), axis=0))) if all_vertices else 0.0
    )
    tolerance = limits.max_surface_gap_m or max(diagonal * 0.005, 1e-6)
    per_render_part_coverage: list[dict[str, Any]] = []
    render_distances: list[np.ndarray] = []
    collision_distances: list[np.ndarray] = []
    if collision_meshes:
        per_part_limit = max(1, surface_sample_limit // max(len(render_meshes), 1))
        for record, mesh in render_pairs:
            points = _surface_points(mesh, per_part_limit)
            distances = _closest_union_distance(points, collision_distance_meshes)
            render_distances.append(distances)
            per_render_part_coverage.append(
                {
                    "render_path": record.path,
                    "sample_count": len(points),
                    "coverage_ratio": float(np.count_nonzero(distances <= tolerance) / len(points))
                    if len(points)
                    else None,
                    "surface_gap_p99_m": float(np.percentile(distances, 99.0))
                    if len(distances)
                    else None,
                }
            )
        per_collision_limit = max(1, surface_sample_limit // max(len(collision_meshes), 1))
        if render_meshes:
            collision_points = [
                _surface_points(mesh, per_collision_limit) for mesh in collision_meshes
            ]
            collision_points = [points for points in collision_points if len(points)]
            if collision_points:
                collision_distances.append(
                    _closest_union_distance(
                        np.concatenate(collision_points, axis=0),
                        render_distance_meshes,
                    )
                )
    source_to_collision = (
        np.concatenate(render_distances) if render_distances else np.empty((0,), dtype=np.float64)
    )
    collision_to_source = (
        np.concatenate(collision_distances)
        if collision_distances
        else np.empty((0,), dtype=np.float64)
    )
    occupancy = _occupancy_metrics(
        render_meshes,
        collision_meshes,
        resolution=occupancy_grid_resolution,
        face_point_limit=occupancy_face_point_limit,
    )
    return CollisionAuditMetrics(
        render_surface_sample_count=len(source_to_collision),
        collision_surface_sample_count=len(collision_to_source),
        render_surface_coverage_ratio=(
            float(np.count_nonzero(source_to_collision <= tolerance) / len(source_to_collision))
            if len(source_to_collision)
            else None
        ),
        surface_gap_p99_m=(
            float(np.percentile(source_to_collision, 99.0)) if len(source_to_collision) else None
        ),
        surface_gap_max_m=(
            float(np.max(source_to_collision)) if len(source_to_collision) else None
        ),
        surface_overreach_p99_m=(
            float(np.percentile(collision_to_source, 99.0)) if len(collision_to_source) else None
        ),
        surface_overreach_max_m=(
            float(np.max(collision_to_source)) if len(collision_to_source) else None
        ),
        false_positive_ratio=occupancy.get("false_positive_ratio"),
        false_negative_ratio=occupancy.get("false_negative_ratio"),
        volume_excess_ratio=occupancy.get("volume_excess_ratio"),
        volume_deficit_ratio=occupancy.get("volume_deficit_ratio"),
        occupancy_grid_resolution=(
            int(occupancy["resolution"]) if occupancy.get("status") == "evaluated" else None
        ),
        occupancy_estimated_face_point_product=occupancy.get("estimated_face_point_product"),
        occupancy_face_point_limit=occupancy.get("face_point_limit"),
        occupancy_status=occupancy.get("status", "not_evaluated"),  # type: ignore[arg-type]
        occupancy_reason=occupancy.get("reason"),  # type: ignore[arg-type]
        per_render_part_coverage=per_render_part_coverage,
    )


def _part_limits(
    configured: CollisionAuditLimits | None,
    active: CollisionAuditLimits,
    render_mesh: Any,
) -> CollisionAuditLimits:
    vertices = np.asarray(render_mesh.vertices, dtype=np.float64)
    diagonal = float(np.linalg.norm(np.ptp(vertices, axis=0))) if len(vertices) else 0.0
    default_tolerance = max(diagonal * 0.005, 1e-6)
    return active.model_copy(
        update={
            "max_surface_gap_m": (
                configured.max_surface_gap_m
                if configured is not None and configured.max_surface_gap_m is not None
                else default_tolerance
            ),
            "max_surface_overreach_m": (
                configured.max_surface_overreach_m
                if configured is not None and configured.max_surface_overreach_m is not None
                else default_tolerance
            ),
            # These limits describe the complete collision package or runtime,
            # not one independently paired part. The aggregate audit owns them.
            "max_collision_size_bytes": None,
            "max_cook_time_s": None,
            "max_runtime_contact_cost": None,
        }
    )


def _source_hull_reuse_gates(
    collision_pairs: list[tuple[Any, Any]],
) -> list[CollisionAuditGate]:
    gates: list[CollisionAuditGate] = []
    for record, mesh in collision_pairs:
        failures = []
        if not mesh.is_watertight:
            failures.append("not watertight")
        if not mesh.is_winding_consistent:
            failures.append("winding is inconsistent")
        if not mesh.is_convex:
            failures.append("not convex")
        gates.append(
            CollisionAuditGate(
                gate_id=f"source_proxy_reuse:{record.path}",
                category="complexity",
                status="fail" if failures else "pass",
                blocking=True,
                measured_value="; ".join(failures) if failures else "eligible",
                limit_value="watertight, consistently wound, convex hull",
                comparison="informational",
                reason=(
                    f"{record.path}: " + "; ".join(failures)
                    if failures
                    else f"{record.path}: source hull is eligible for direct reuse"
                ),
            )
        )
    return gates


def _sample_probe_polyline(
    points: np.ndarray,
    step_m: float,
    limit: int = 4096,
) -> tuple[np.ndarray, bool, int]:
    from .protected_features import _sample_polyline

    return _sample_polyline(points, step_m, limit=limit)


def _evaluate_part_feature_probes(
    collision_meshes: list[Any],
    features: list[ProtectedFeature],
) -> list[ProtectedFeatureProbeResult]:
    from .collision import _closest_union_distance, _contains_bounded

    results: list[ProtectedFeatureProbeResult] = []
    for feature in sorted(features, key=lambda item: item.name):
        if feature.probe is None:
            results.append(
                ProtectedFeatureProbeResult(
                    feature_name=feature.name,
                    probe_kind=(
                        "negative_space_path"
                        if feature.kind in {"opening", "cavity", "clearance"}
                        else "surface_support"
                    ),
                    status="not_evaluated",
                    warnings=["protected feature has no deterministic probe geometry"],
                )
            )
            continue
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
        resolution_truncated = False
        required_sample_count = len(control_points)
        if probe.kind == "negative_space_path":
            points, resolution_truncated, required_sample_count = _sample_probe_polyline(
                control_points,
                max(tolerance, 1e-6),
            )
        else:
            points = control_points
        if not collision_meshes:
            results.append(
                ProtectedFeatureProbeResult(
                    feature_name=feature.name,
                    probe_kind=probe.kind,
                    status="not_evaluated",
                    sample_count=len(points),
                    warnings=["paired collision subset is empty"],
                )
            )
            continue
        try:
            distances = _closest_union_distance(points, collision_meshes)
        except Exception as exc:
            results.append(
                ProtectedFeatureProbeResult(
                    feature_name=feature.name,
                    probe_kind=probe.kind,
                    status="not_evaluated",
                    sample_count=len(points),
                    warnings=[f"surface-distance query failed: {type(exc).__name__}: {exc}"],
                )
            )
            continue
        if probe.kind == "surface_support":
            maximum_distance = float(np.max(distances)) if len(distances) else None
            failures = []
            if maximum_distance is None:
                failures.append("surface support probe produced no distance samples")
            elif maximum_distance > tolerance:
                failures.append(
                    f"maximum support distance {maximum_distance:.6g} m exceeds "
                    f"tolerance {tolerance:.6g} m"
                )
            results.append(
                ProtectedFeatureProbeResult(
                    feature_name=feature.name,
                    probe_kind=probe.kind,
                    status="fail" if failures else "pass",
                    sample_count=len(points),
                    maximum_surface_distance_m=maximum_distance,
                    failures=failures,
                )
            )
            continue
        try:
            occupied = np.zeros(len(points), dtype=bool)
            for mesh in collision_meshes:
                occupied |= _contains_bounded(mesh, points)
        except Exception as exc:
            results.append(
                ProtectedFeatureProbeResult(
                    feature_name=feature.name,
                    probe_kind=probe.kind,
                    status="not_evaluated",
                    sample_count=len(points),
                    minimum_clearance_m=float(np.min(distances)) if len(distances) else None,
                    warnings=[f"occupancy query failed: {type(exc).__name__}: {exc}"],
                )
            )
            continue
        occupied_count = int(np.count_nonzero(occupied))
        minimum_clearance = float(np.min(distances)) if len(distances) else None
        failures = []
        if occupied_count:
            failures.append(f"{occupied_count} path samples are inside solid geometry")
        if minimum_clearance is not None and minimum_clearance + tolerance < required_clearance:
            failures.append(
                f"minimum path clearance {minimum_clearance:.6g} m is below "
                f"required {required_clearance:.6g} m"
            )
        warnings = []
        if resolution_truncated:
            warnings.append(
                "negative-space path required "
                f"{required_sample_count} samples at the requested resolution; "
                f"the bounded probe evaluated {len(points)} uniformly across the full path"
            )
        results.append(
            ProtectedFeatureProbeResult(
                feature_name=feature.name,
                probe_kind=probe.kind,
                status=(
                    "fail" if failures else "not_evaluated" if resolution_truncated else "pass"
                ),
                sample_count=len(points),
                minimum_clearance_m=minimum_clearance,
                occupied_sample_count=occupied_count,
                failures=failures,
                warnings=warnings,
            )
        )
    return results


def audit_source_collision_proxy(
    render_path: str | Path,
    collision_path: str | Path,
    *,
    protected_features: list[ProtectedFeature] | None = None,
    limits: CollisionAuditLimits | None = None,
    surface_sample_limit: int = 4096,
    occupancy_grid_resolution: int = 24,
    occupancy_face_point_limit: int = 100_000_000,
    report_path: str | Path | None = None,
) -> SourceCollisionAudit:
    """Measure a source collision set before any regeneration is considered."""

    if surface_sample_limit < 64:
        raise ValueError("surface_sample_limit must be at least 64")
    if not 8 <= occupancy_grid_resolution <= 64:
        raise ValueError("occupancy_grid_resolution must be in [8, 64]")
    if occupancy_face_point_limit < 1:
        raise ValueError("occupancy_face_point_limit must be positive")
    render = Path(render_path).expanduser().resolve()
    collision = Path(collision_path).expanduser().resolve()
    render_records, _ = load_meshes(render)
    collision_records, _ = load_meshes(collision, include_guide_purpose=True)
    role_render = [record for record in render_records if record.role == "render"]
    if role_render:
        render_records = role_render
    role_collision = [record for record in collision_records if record.role == "collision"]
    if role_collision:
        collision_records = role_collision
    render_pairs = [
        (record, mesh) for record in render_records if (mesh := _as_trimesh(record)) is not None
    ]
    collision_pairs = [
        (record, mesh) for record in collision_records if (mesh := _as_trimesh(record)) is not None
    ]
    render_meshes = [mesh for _record, mesh in render_pairs]
    collision_meshes = [mesh for _record, mesh in collision_pairs]
    all_vertices = [np.asarray(mesh.vertices, dtype=np.float64) for mesh in render_meshes]
    diagonal = (
        float(np.linalg.norm(np.ptp(np.vstack(all_vertices), axis=0))) if all_vertices else 0.0
    )
    active_limits = limits or CollisionAuditLimits(
        max_surface_gap_m=max(diagonal * 0.005, 1e-6),
        max_surface_overreach_m=max(diagonal * 0.005, 1e-6),
    )
    if active_limits.max_surface_gap_m is None or active_limits.max_surface_overreach_m is None:
        active_limits = active_limits.model_copy(
            update={
                "max_surface_gap_m": active_limits.max_surface_gap_m
                if active_limits.max_surface_gap_m is not None
                else max(diagonal * 0.005, 1e-6),
                "max_surface_overreach_m": active_limits.max_surface_overreach_m
                if active_limits.max_surface_overreach_m is not None
                else max(diagonal * 0.005, 1e-6),
            }
        )
    metrics = _measure_collision_metrics(
        render_pairs,
        collision_pairs,
        limits=active_limits,
        surface_sample_limit=surface_sample_limit,
        occupancy_grid_resolution=occupancy_grid_resolution,
        occupancy_face_point_limit=occupancy_face_point_limit,
    )
    complexity = CollisionComplexity(
        hull_count=len(collision_meshes),
        total_vertices=sum(len(mesh.vertices) for mesh in collision_meshes),
        total_faces=sum(len(mesh.faces) for mesh in collision_meshes),
        maximum_vertices_per_hull=max((len(mesh.vertices) for mesh in collision_meshes), default=0),
        maximum_faces_per_hull=max((len(mesh.faces) for mesh in collision_meshes), default=0),
        collision_size_bytes=collision.stat().st_size,
    )
    probes = evaluate_feature_probes(collision, protected_features or [])
    required = {feature.name for feature in protected_features or [] if feature.required}
    if len(render_pairs) <= 1:
        return evaluate_source_collision_gates(
            render_path=render,
            collision_path=collision,
            metrics=metrics,
            complexity=complexity,
            limits=active_limits,
            protected_feature_probes=probes,
            required_feature_names=required,
            report_path=report_path,
        )

    pairing = pair_compound_collision_parts(
        [record for record, _mesh in render_pairs],
        [record for record, _mesh in collision_pairs],
        explicit_stage_paths=[collision, render],
        surface_sample_limit=max(
            16,
            min(256, surface_sample_limit // max(len(collision_pairs), 1)),
        ),
    )
    render_by_path = {record.path: (record, mesh) for record, mesh in render_pairs}
    collision_by_path = {record.path: (record, mesh) for record, mesh in collision_pairs}
    minimum_part_samples = 32
    part_sampling_available = surface_sample_limit >= minimum_part_samples * max(
        len(render_pairs),
        len(collision_pairs),
        1,
    )
    scoped_features: dict[str, list[ProtectedFeature]] = {
        record.path: [] for record, _mesh in render_pairs
    }
    render_paths = sorted(scoped_features)
    for feature in protected_features or []:
        if not feature.scope_path:
            continue
        scope = feature.scope_path.rstrip("/")
        matching_paths = [
            path for path in render_paths if path == scope or path.startswith(f"{scope}/")
        ]
        if len(matching_paths) == 1:
            scoped_features[matching_paths[0]].append(feature)
    assignments_by_collision = {
        assignment.collision_path: assignment for assignment in pairing.collision_assignments
    }
    compound_gates: list[CollisionAuditGate] = []
    for assignment in pairing.collision_assignments:
        compound_gates.append(
            CollisionAuditGate(
                gate_id=f"compound_pairing:collision:{assignment.collision_path}",
                category="fidelity",
                status="pass" if assignment.status == "matched" else "not_evaluated",
                blocking=True,
                measured_value=assignment.status,
                limit_value="matched",
                comparison="informational",
                reason=assignment.reason,
            )
        )

    part_audits: list[CompoundCollisionPartAudit] = []
    part_gate_models: list[CollisionAuditGate] = []
    for paired_part in pairing.render_parts:
        relevant_assignments = [
            assignment
            for assignment in pairing.collision_assignments
            if assignment.render_path == paired_part.render_path
            or paired_part.render_path in assignment.candidate_render_paths
        ]
        unresolved_for_part = [
            assignment for assignment in relevant_assignments if assignment.status != "matched"
        ]
        pairing_status: Literal["matched", "ambiguous", "unmatched", "not_evaluated"]
        if any(assignment.status == "ambiguous" for assignment in unresolved_for_part):
            pairing_status = "ambiguous"
        elif any(assignment.status == "not_evaluated" for assignment in unresolved_for_part):
            pairing_status = "not_evaluated"
        elif paired_part.status == "unmatched":
            pairing_status = "unmatched"
        else:
            pairing_status = "matched"
        if pairing_status != "matched":
            reason = paired_part.reason
            part_gate = CollisionAuditGate(
                gate_id=f"compound_pairing:render:{paired_part.render_path}",
                category="fidelity",
                status="not_evaluated",
                blocking=True,
                measured_value=pairing_status,
                limit_value="matched",
                comparison="informational",
                reason=reason,
            )
            compound_gates.append(part_gate)
            part_audits.append(
                CompoundCollisionPartAudit(
                    render_path=paired_part.render_path,
                    collision_paths=paired_part.collision_paths,
                    pairing_status=pairing_status,
                    pairing_methods=paired_part.assignment_methods,
                    pairing_evidence=relevant_assignments,
                    status="not_evaluated",
                    decision="review_required",
                    gates=[part_gate.model_dump(mode="json")],
                    warnings=[reason],
                )
            )
            continue

        if not part_sampling_available:
            reason = (
                f"part-level audit requires at least {minimum_part_samples} samples per render "
                f"and collision part within total limit {surface_sample_limit}"
            )
            part_gate = CollisionAuditGate(
                gate_id=f"compound_sampling:{paired_part.render_path}",
                category="fidelity",
                status="not_evaluated",
                blocking=True,
                measured_value=surface_sample_limit,
                limit_value=minimum_part_samples * max(len(render_pairs), len(collision_pairs), 1),
                comparison=">=",
                reason=reason,
            )
            compound_gates.append(part_gate)
            part_audits.append(
                CompoundCollisionPartAudit(
                    render_path=paired_part.render_path,
                    collision_paths=paired_part.collision_paths,
                    pairing_status="matched",
                    pairing_methods=paired_part.assignment_methods,
                    pairing_evidence=[
                        assignments_by_collision[path] for path in paired_part.collision_paths
                    ],
                    status="not_evaluated",
                    decision="review_required",
                    gates=[part_gate.model_dump(mode="json")],
                    warnings=[reason],
                )
            )
            continue

        render_pair = render_by_path[paired_part.render_path]
        selected_collision_pairs = [collision_by_path[path] for path in paired_part.collision_paths]
        limits_for_part = _part_limits(limits, active_limits, render_pair[1])
        render_sample_quota = surface_sample_limit // len(render_pairs)
        collision_sample_quota = surface_sample_limit // len(collision_pairs)
        part_sample_limit = min(
            render_sample_quota,
            collision_sample_quota * len(selected_collision_pairs),
        )
        metrics_for_part = _measure_collision_metrics(
            [render_pair],
            selected_collision_pairs,
            limits=limits_for_part,
            surface_sample_limit=part_sample_limit,
            occupancy_grid_resolution=occupancy_grid_resolution,
            occupancy_face_point_limit=occupancy_face_point_limit,
        )
        complexity_for_part = CollisionComplexity(
            hull_count=len(selected_collision_pairs),
            total_vertices=sum(len(mesh.vertices) for _record, mesh in selected_collision_pairs),
            total_faces=sum(len(mesh.faces) for _record, mesh in selected_collision_pairs),
            maximum_vertices_per_hull=max(
                (len(mesh.vertices) for _record, mesh in selected_collision_pairs),
                default=0,
            ),
            maximum_faces_per_hull=max(
                (len(mesh.faces) for _record, mesh in selected_collision_pairs),
                default=0,
            ),
            collision_size_bytes=collision.stat().st_size,
        )
        part_probes = _evaluate_part_feature_probes(
            [mesh for _record, mesh in selected_collision_pairs],
            scoped_features[paired_part.render_path],
        )
        required_part_features = {
            feature.name for feature in scoped_features[paired_part.render_path] if feature.required
        }
        part_report = evaluate_source_collision_gates(
            render_path=render,
            collision_path=collision,
            metrics=metrics_for_part,
            complexity=complexity_for_part,
            limits=limits_for_part,
            protected_feature_probes=part_probes,
            required_feature_names=required_part_features,
        )
        reuse_gates = _source_hull_reuse_gates(selected_collision_pairs)
        reuse_failures = [gate for gate in reuse_gates if gate.status == "fail"]
        part_decision = part_report.decision
        part_status = part_report.status
        part_reasons = list(part_report.regeneration_reasons)
        if reuse_failures:
            part_decision = "regenerate_candidate"
            part_status = "fail"
            part_reasons.extend(gate.reason for gate in reuse_failures)
        all_part_gates = [*part_report.gates, *reuse_gates]
        for gate in all_part_gates:
            part_gate_models.append(
                gate.model_copy(
                    update={"gate_id": f"compound_part:{paired_part.render_path}:{gate.gate_id}"}
                )
            )
        part_audits.append(
            CompoundCollisionPartAudit(
                render_path=paired_part.render_path,
                collision_paths=paired_part.collision_paths,
                pairing_status="matched",
                pairing_methods=paired_part.assignment_methods,
                pairing_evidence=[
                    assignments_by_collision[path] for path in paired_part.collision_paths
                ],
                status=part_status,
                decision=part_decision,
                sample_count=metrics_for_part.render_surface_sample_count,
                coverage_ratio=metrics_for_part.render_surface_coverage_ratio,
                surface_gap_p99_m=metrics_for_part.surface_gap_p99_m,
                metrics=metrics_for_part.model_dump(mode="json"),
                complexity=complexity_for_part.model_dump(mode="json"),
                gates=[gate.model_dump(mode="json") for gate in all_part_gates],
                protected_feature_probes=part_probes,
                regeneration_reasons=part_reasons,
                warnings=part_report.warnings,
            )
        )

    metrics = metrics.model_copy(
        update={"per_render_part_coverage": [part.model_dump(mode="json") for part in part_audits]}
    )
    aggregate = evaluate_source_collision_gates(
        render_path=render,
        collision_path=collision,
        metrics=metrics,
        complexity=complexity,
        limits=active_limits,
        protected_feature_probes=probes,
        required_feature_names=required,
    )
    part_reviews = [part for part in part_audits if part.decision == "review_required"]
    part_regenerations = [part for part in part_audits if part.decision == "regenerate_candidate"]
    unlocalized_aggregate_failure = (
        aggregate.decision == "regenerate_candidate" and not part_regenerations
    )
    if pairing.status == "review_required" or part_reviews:
        decision: Literal["preserve_source", "regenerate_candidate", "review_required"] = (
            "review_required"
        )
        status: Literal["pass", "conditional", "fail", "not_evaluated"] = "not_evaluated"
    elif aggregate.decision == "review_required" or unlocalized_aggregate_failure:
        decision = "review_required"
        status = "not_evaluated"
    elif aggregate.decision == "regenerate_candidate" or part_regenerations:
        decision = "regenerate_candidate"
        status = "fail"
    else:
        decision = "preserve_source"
        status = (
            "conditional"
            if aggregate.status == "conditional"
            or any(part.status == "conditional" for part in part_audits)
            else "pass"
        )
    part_reasons = [
        f"{part.render_path}: {reason}"
        for part in part_audits
        for reason in part.regeneration_reasons
    ]
    warnings = [
        *aggregate.warnings,
        *pairing.warnings,
        *(
            [
                "aggregate collision gates failed without a uniquely identified failing part; "
                "automatic replacement requires review"
            ]
            if unlocalized_aggregate_failure
            else []
        ),
        *[f"{part.render_path}: {warning}" for part in part_audits for warning in part.warnings],
    ]
    resolved_report_path = (
        str(Path(report_path).expanduser().resolve()) if report_path is not None else None
    )
    report = aggregate.model_copy(
        update={
            "status": status,
            "decision": decision,
            "metrics": metrics,
            "gates": [*aggregate.gates, *compound_gates, *part_gate_models],
            "regeneration_reasons": list(
                dict.fromkeys([*aggregate.regeneration_reasons, *part_reasons])
            ),
            "warnings": list(dict.fromkeys(warnings)),
            "report_path": resolved_report_path,
        }
    )
    if report_path is not None:
        atomic_write_json(report_path, report)
    return report
