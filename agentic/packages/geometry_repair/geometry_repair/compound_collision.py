# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic render/collision pairing for compound rigid assets."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .models import ProtectedFeatureProbeResult

COMPOUND_COLLISION_PAIRING_SCHEMA_VERSION = "geometry-repair.compound-collision-pairing.v1"
COMPOUND_COLLISION_PART_AUDIT_SCHEMA_VERSION = "geometry-repair.compound-collision-part-audit.v1"

_CUSTOM_RENDER_PATH_KEYS = (
    "geometryRepairRenderPrimPath",
    "renderPrimPath",
    "sourcePrimPath",
)
_RENDER_RELATIONSHIP_NAMES = (
    "geometryRepair:renderPart",
    "geometryRepair:renderPrim",
)
_ROLE_TOKENS = {
    "coll",
    "collider",
    "colliders",
    "collision",
    "col",
    "geom",
    "geometry",
    "mesh",
    "meshes",
    "proxy",
    "render",
    "shape",
    "vis",
    "visual",
    "visuals",
}
_COLLISION_DETAIL_TOKENS = {"convex", "hull", "hulls", "proxy"}
_COLLISION_ROLE_TOKENS = {
    "coll",
    "collider",
    "colliders",
    "collision",
    "col",
    "proxy",
}


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _MeshRecord(Protocol):
    path: str
    world_vertices_m: np.ndarray
    triangles: np.ndarray


class PairMeasurement(_StrictModel):
    """Bounded geometric evidence for one collision/render candidate pair."""

    render_path: str
    bbox_gap_m: float = Field(ge=0.0)
    bbox_gap_ratio: float = Field(ge=0.0)
    collision_surface_p95_m: float | None = Field(default=None, ge=0.0)
    collision_surface_p95_ratio: float | None = Field(default=None, ge=0.0)
    eligible: bool
    reason: str


class CollisionPartAssignment(_StrictModel):
    """Pairing disposition for one authored collision prim."""

    collision_path: str
    status: Literal["matched", "ambiguous", "unmatched", "not_evaluated"]
    method: Literal[
        "explicit_metadata",
        "path_convention",
        "measured_unique",
        "none",
    ] = "none"
    render_path: str | None = None
    explicit_render_paths: list[str] = Field(default_factory=list)
    candidate_render_paths: list[str] = Field(default_factory=list)
    measurements: list[PairMeasurement] = Field(default_factory=list)
    reason: str


class RenderCollisionPartPair(_StrictModel):
    """All source collision prims assigned to one render part."""

    render_path: str
    status: Literal["matched", "unmatched"]
    collision_paths: list[str] = Field(default_factory=list)
    assignment_methods: list[Literal["explicit_metadata", "path_convention", "measured_unique"]] = (
        Field(default_factory=list)
    )
    reason: str


class CompoundCollisionPairing(_StrictModel):
    """Persistable pairing evidence; review is mandatory unless every part is covered."""

    schema_version: Literal["geometry-repair.compound-collision-pairing.v1"] = (
        COMPOUND_COLLISION_PAIRING_SCHEMA_VERSION
    )
    status: Literal["pass", "review_required"]
    render_parts: list[RenderCollisionPartPair] = Field(default_factory=list)
    collision_assignments: list[CollisionPartAssignment] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class CompoundCollisionPartAudit(_StrictModel):
    """Part-level audit evidence nested in the existing source-audit metrics field."""

    schema_version: Literal["geometry-repair.compound-collision-part-audit.v1"] = (
        COMPOUND_COLLISION_PART_AUDIT_SCHEMA_VERSION
    )
    render_path: str
    collision_paths: list[str] = Field(default_factory=list)
    pairing_status: Literal["matched", "ambiguous", "unmatched", "not_evaluated"]
    pairing_methods: list[Literal["explicit_metadata", "path_convention", "measured_unique"]] = (
        Field(default_factory=list)
    )
    pairing_evidence: list[CollisionPartAssignment] = Field(default_factory=list)
    status: Literal["pass", "conditional", "fail", "not_evaluated"]
    decision: Literal["preserve_source", "regenerate_candidate", "review_required"]
    sample_count: int = Field(default=0, ge=0)
    coverage_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    surface_gap_p99_m: float | None = Field(default=None, ge=0.0)
    metrics: dict[str, Any] = Field(default_factory=dict)
    complexity: dict[str, Any] = Field(default_factory=dict)
    gates: list[dict[str, Any]] = Field(default_factory=list)
    protected_feature_probes: list[ProtectedFeatureProbeResult] = Field(default_factory=list)
    regeneration_reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def _path_tokens(path: str, *, role: Literal["render", "collision"]) -> tuple[str, ...]:
    value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", path.strip("/"))
    value = re.sub(r"(?<=[A-Za-z])(?=[0-9])|(?<=[0-9])(?=[A-Za-z])", "_", value)
    raw_tokens = [token.casefold() for token in re.split(r"[^A-Za-z0-9]+", value) if token]
    tokens: list[str] = []
    collision_role_seen = False
    drop_collision_index = False
    for token in raw_tokens:
        if token in _ROLE_TOKENS:
            if role == "collision" and token in _COLLISION_ROLE_TOKENS:
                collision_role_seen = True
            if role == "collision" and collision_role_seen and token in _COLLISION_DETAIL_TOKENS:
                drop_collision_index = True
            continue
        if role == "collision" and collision_role_seen and token in _COLLISION_DETAIL_TOKENS:
            drop_collision_index = True
            continue
        if drop_collision_index and token.isdecimal():
            continue
        drop_collision_index = False
        tokens.append(token)
    return tuple(tokens)


def _canonical_part_key(path: str, *, role: Literal["render", "collision"]) -> str | None:
    tokens = _path_tokens(path, role=role)
    return "/".join(tokens) if tokens else None


def _normalized_explicit_targets(
    value: Any,
) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values = [value]
    else:
        try:
            values = list(value)
        except TypeError:
            return []
    return sorted(
        {item for item in values if isinstance(item, str) and item.startswith("/") and item != "/"}
    )


def explicit_collision_render_targets(
    stage_paths: Sequence[str | Path],
    collision_paths: Sequence[str],
) -> dict[str, list[str]]:
    """Read narrowly recognized authored render targets without inferring intent."""

    from pxr import Sdf, Usd

    requested = sorted(set(collision_paths))
    targets: dict[str, set[str]] = {path: set() for path in requested}
    for raw_stage_path in stage_paths:
        stage_path = Path(raw_stage_path).expanduser().resolve()
        if stage_path.suffix.lower() not in {".usd", ".usda", ".usdc", ".usdz"}:
            continue
        stage = Usd.Stage.Open(str(stage_path))
        if stage is None:
            continue
        for collision_path in requested:
            prim = stage.GetPrimAtPath(collision_path)
            if not prim:
                continue
            for key in _CUSTOM_RENDER_PATH_KEYS:
                value = prim.GetCustomDataByKey(key)
                for target in _normalized_explicit_targets(value):
                    if target != collision_path and Sdf.Path(target).IsPrimPath():
                        targets[collision_path].add(target)
            for relationship_name in _RENDER_RELATIONSHIP_NAMES:
                relationship = prim.GetRelationship(relationship_name)
                if not relationship:
                    continue
                for target in relationship.GetTargets():
                    target_path = target.GetPrimPath()
                    if target_path.IsAbsolutePath() and str(target_path) != collision_path:
                        targets[collision_path].add(str(target_path))
    return {path: sorted(values) for path, values in targets.items() if values}


def _resolve_explicit_targets(
    targets: Sequence[str],
    render_paths: Sequence[str],
) -> list[str]:
    resolved: set[str] = set()
    for target in targets:
        if target in render_paths:
            resolved.add(target)
            continue
        prefix = target.rstrip("/") + "/"
        resolved.update(path for path in render_paths if path.startswith(prefix))
    return sorted(resolved)


def _bounds(record: _MeshRecord) -> tuple[np.ndarray, np.ndarray, float]:
    vertices = np.asarray(record.world_vertices_m, dtype=np.float64).reshape((-1, 3))
    if not len(vertices) or not bool(np.isfinite(vertices).all()):
        raise ValueError(f"{record.path}: pairing requires finite mesh vertices")
    minimum = vertices.min(axis=0)
    maximum = vertices.max(axis=0)
    return minimum, maximum, float(np.linalg.norm(maximum - minimum))


def _bbox_gap(
    left_minimum: np.ndarray,
    left_maximum: np.ndarray,
    right_minimum: np.ndarray,
    right_maximum: np.ndarray,
) -> float:
    separation = np.maximum(
        np.maximum(left_minimum - right_maximum, right_minimum - left_maximum),
        0.0,
    )
    return float(np.linalg.norm(separation))


def _surface_points(record: _MeshRecord, limit: int) -> np.ndarray:
    vertices = np.asarray(record.world_vertices_m, dtype=np.float64).reshape((-1, 3))
    triangles = np.asarray(record.triangles, dtype=np.int64).reshape((-1, 3))
    valid = np.all((triangles >= 0) & (triangles < len(vertices)), axis=1)
    triangles = triangles[valid]
    centroids = vertices[triangles].mean(axis=1) if len(triangles) else np.empty((0, 3))
    points = np.vstack((vertices, centroids)) if len(centroids) else vertices
    if len(points) <= limit:
        return points
    indices = np.linspace(0, len(points) - 1, num=limit, dtype=np.int64)
    return points[indices]


def _collision_surface_distance(
    collision: _MeshRecord,
    render: _MeshRecord,
    *,
    sample_limit: int,
) -> float:
    import trimesh

    render_vertices = np.asarray(render.world_vertices_m, dtype=np.float64).reshape((-1, 3))
    render_triangles = np.asarray(render.triangles, dtype=np.int64).reshape((-1, 3))
    valid = np.all((render_triangles >= 0) & (render_triangles < len(render_vertices)), axis=1)
    render_mesh = trimesh.Trimesh(
        vertices=render_vertices,
        faces=render_triangles[valid],
        process=False,
    )
    points = _surface_points(collision, sample_limit)
    if not len(points) or not len(render_mesh.faces):
        raise ValueError("pairing distance requires non-empty surfaces")
    try:
        _nearest, distances, _triangle_ids = trimesh.proximity.closest_point(render_mesh, points)
    except (ImportError, ModuleNotFoundError):
        _nearest, distances, _triangle_ids = trimesh.proximity.closest_point_naive(
            render_mesh,
            points,
        )
    return float(np.percentile(distances, 95.0))


def _measure_candidate(
    collision: _MeshRecord,
    render: _MeshRecord,
    *,
    max_bbox_gap_ratio: float,
    max_surface_distance_ratio: float,
    surface_sample_limit: int,
) -> PairMeasurement:
    collision_minimum, collision_maximum, collision_diagonal = _bounds(collision)
    render_minimum, render_maximum, render_diagonal = _bounds(render)
    scale = max(collision_diagonal, render_diagonal, 1e-9)
    bbox_gap = _bbox_gap(
        collision_minimum,
        collision_maximum,
        render_minimum,
        render_maximum,
    )
    bbox_gap_ratio = bbox_gap / scale
    if bbox_gap_ratio > max_bbox_gap_ratio:
        return PairMeasurement(
            render_path=render.path,
            bbox_gap_m=bbox_gap,
            bbox_gap_ratio=bbox_gap_ratio,
            eligible=False,
            reason=(f"bbox gap ratio {bbox_gap_ratio:.6g} exceeds {max_bbox_gap_ratio:.6g}"),
        )
    try:
        surface_distance = _collision_surface_distance(
            collision,
            render,
            sample_limit=surface_sample_limit,
        )
    except Exception as exc:
        return PairMeasurement(
            render_path=render.path,
            bbox_gap_m=bbox_gap,
            bbox_gap_ratio=bbox_gap_ratio,
            eligible=False,
            reason=f"surface proximity was not evaluated: {type(exc).__name__}: {exc}",
        )
    surface_ratio = surface_distance / scale
    eligible = surface_ratio <= max_surface_distance_ratio
    return PairMeasurement(
        render_path=render.path,
        bbox_gap_m=bbox_gap,
        bbox_gap_ratio=bbox_gap_ratio,
        collision_surface_p95_m=surface_distance,
        collision_surface_p95_ratio=surface_ratio,
        eligible=eligible,
        reason=(
            "bounded proximity gates passed"
            if eligible
            else (
                f"collision surface p95 ratio {surface_ratio:.6g} exceeds "
                f"{max_surface_distance_ratio:.6g}"
            )
        ),
    )


def pair_compound_collision_parts(
    render_records: Sequence[_MeshRecord],
    collision_records: Sequence[_MeshRecord],
    *,
    explicit_targets: Mapping[str, str | Sequence[str]] | None = None,
    explicit_stage_paths: Sequence[str | Path] = (),
    max_pair_evaluations: int = 4096,
    surface_sample_limit: int = 256,
    max_bbox_gap_ratio: float = 0.02,
    max_surface_distance_ratio: float = 0.10,
) -> CompoundCollisionPairing:
    """Pair every collision prim explicitly first, then by unique measured evidence."""

    if max_pair_evaluations < 1:
        raise ValueError("max_pair_evaluations must be positive")
    if surface_sample_limit < 16:
        raise ValueError("surface_sample_limit must be at least 16")
    renders = sorted(render_records, key=lambda item: item.path)
    collisions = sorted(collision_records, key=lambda item: item.path)
    render_paths = [record.path for record in renders]
    if len(render_paths) != len(set(render_paths)):
        raise ValueError("render inventory contains duplicate prim paths")
    collision_paths = [record.path for record in collisions]
    if len(collision_paths) != len(set(collision_paths)):
        raise ValueError("collision inventory contains duplicate prim paths")

    authored = explicit_collision_render_targets(explicit_stage_paths, collision_paths)
    for collision_path, raw_targets in (explicit_targets or {}).items():
        authored.setdefault(collision_path, [])
        authored[collision_path] = sorted(
            set(authored[collision_path]) | set(_normalized_explicit_targets(raw_targets))
        )

    assignments: list[CollisionPartAssignment] = []
    assigned: dict[str, list[CollisionPartAssignment]] = {path: [] for path in render_paths}
    unresolved: list[_MeshRecord] = []
    render_by_key: dict[str, list[str]] = {}
    for render in renders:
        key = _canonical_part_key(render.path, role="render")
        if key:
            render_by_key.setdefault(key, []).append(render.path)

    for collision in collisions:
        explicit = authored.get(collision.path, [])
        if explicit:
            resolved = _resolve_explicit_targets(explicit, render_paths)
            if len(resolved) == 1:
                assignment = CollisionPartAssignment(
                    collision_path=collision.path,
                    status="matched",
                    method="explicit_metadata",
                    render_path=resolved[0],
                    explicit_render_paths=explicit,
                    candidate_render_paths=resolved,
                    reason="one authored render target resolved to one render mesh",
                )
                assignments.append(assignment)
                assigned[resolved[0]].append(assignment)
            else:
                assignments.append(
                    CollisionPartAssignment(
                        collision_path=collision.path,
                        status="ambiguous" if len(resolved) > 1 else "unmatched",
                        explicit_render_paths=explicit,
                        candidate_render_paths=resolved,
                        reason=(
                            "authored render target resolves to multiple render meshes"
                            if resolved
                            else "authored render target does not resolve to a render mesh"
                        ),
                    )
                )
            continue

        key = _canonical_part_key(collision.path, role="collision")
        conventional = sorted(render_by_key.get(key, [])) if key else []
        if len(conventional) == 1:
            assignment = CollisionPartAssignment(
                collision_path=collision.path,
                status="matched",
                method="path_convention",
                render_path=conventional[0],
                candidate_render_paths=conventional,
                reason="role-normalized USD prim paths identify one render mesh",
            )
            assignments.append(assignment)
            assigned[conventional[0]].append(assignment)
        else:
            unresolved.append(collision)

    if len(unresolved) * len(renders) > max_pair_evaluations:
        reason = (
            f"measured pairing requires {len(unresolved) * len(renders)} pair evaluations, "
            f"above limit {max_pair_evaluations}"
        )
        assignments.extend(
            CollisionPartAssignment(
                collision_path=collision.path,
                status="not_evaluated",
                reason=reason,
            )
            for collision in unresolved
        )
    else:
        for collision in unresolved:
            measurements = [
                _measure_candidate(
                    collision,
                    render,
                    max_bbox_gap_ratio=max_bbox_gap_ratio,
                    max_surface_distance_ratio=max_surface_distance_ratio,
                    surface_sample_limit=surface_sample_limit,
                )
                for render in renders
            ]
            eligible = sorted(
                measurement.render_path for measurement in measurements if measurement.eligible
            )
            if len(eligible) == 1:
                assignment = CollisionPartAssignment(
                    collision_path=collision.path,
                    status="matched",
                    method="measured_unique",
                    render_path=eligible[0],
                    candidate_render_paths=eligible,
                    measurements=measurements,
                    reason="exactly one render mesh passed bounded proximity gates",
                )
                assignments.append(assignment)
                assigned[eligible[0]].append(assignment)
            else:
                assignments.append(
                    CollisionPartAssignment(
                        collision_path=collision.path,
                        status="ambiguous" if len(eligible) > 1 else "unmatched",
                        candidate_render_paths=eligible,
                        measurements=measurements,
                        reason=(
                            "multiple render meshes passed bounded proximity gates"
                            if eligible
                            else "no render mesh passed bounded proximity gates"
                        ),
                    )
                )

    assignments.sort(key=lambda item: item.collision_path)
    render_parts: list[RenderCollisionPartPair] = []
    for render_path in render_paths:
        part_assignments = sorted(
            assigned[render_path],
            key=lambda item: item.collision_path,
        )
        collision_part_paths = [item.collision_path for item in part_assignments]
        methods = sorted({item.method for item in part_assignments if item.method != "none"})
        render_parts.append(
            RenderCollisionPartPair(
                render_path=render_path,
                status="matched" if collision_part_paths else "unmatched",
                collision_paths=collision_part_paths,
                assignment_methods=methods,  # type: ignore[arg-type]
                reason=(
                    "every listed collision prim resolved to this render mesh"
                    if collision_part_paths
                    else "no collision prim resolved uniquely to this render mesh"
                ),
            )
        )

    unresolved_assignments = [item for item in assignments if item.status != "matched"]
    unmatched_renders = [item for item in render_parts if item.status != "matched"]
    warnings = [f"{item.collision_path}: {item.reason}" for item in unresolved_assignments] + [
        f"{item.render_path}: {item.reason}" for item in unmatched_renders
    ]
    return CompoundCollisionPairing(
        status="review_required" if warnings else "pass",
        render_parts=render_parts,
        collision_assignments=assignments,
        warnings=warnings,
    )


def compound_part_audit_map(
    entries: Sequence[Mapping[str, Any]],
) -> dict[str, CompoundCollisionPartAudit]:
    """Recover validated part decisions from a v1 source-audit metrics payload."""

    result: dict[str, CompoundCollisionPartAudit] = {}
    for entry in entries:
        if entry.get("schema_version") != COMPOUND_COLLISION_PART_AUDIT_SCHEMA_VERSION:
            continue
        part = CompoundCollisionPartAudit.model_validate(entry)
        if part.render_path in result:
            raise ValueError(f"duplicate compound audit entry for {part.render_path}")
        result[part.render_path] = part
    return result
