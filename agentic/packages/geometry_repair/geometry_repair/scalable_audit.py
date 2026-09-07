# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic, resource-budgeted non-mutating triangle intersection audit."""

from __future__ import annotations

import hashlib
import math
import time
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .mesh_io import _topology_vertex_ids, _triangles_intersect, _triangles_overlap_coplanar

SCALABLE_AUDIT_SCHEMA_VERSION = "geometry-repair.scalable-audit.v1"
AuditStatus = Literal[
    "evaluated_pass",
    "evaluated_fail",
    "skipped_resource_limit",
    "indeterminate",
]
PairRelation = Literal["within_part", "cross_part", "unknown"]
ResourceLimitKind = Literal["broad_pairs", "exact_tests", "wall_time", "memory"]


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ScalableAuditBudget(_FrozenModel):
    """Independent work limits for one non-mutating audit."""

    broad_pair_limit: int = Field(default=5_000_000, ge=1)
    exact_test_limit: int = Field(default=1_000_000, ge=1)
    wall_time_s: float = Field(default=60.0, gt=0.0, le=86_400.0)
    memory_limit_bytes: int = Field(default=1024 * 1024 * 1024, ge=1024)
    chunk_size: int = Field(default=2048, ge=1, le=65_536)
    finding_sample_limit: int = Field(default=128, ge=0, le=10_000)


class AuditPairFinding(_FrozenModel):
    left_triangle: int = Field(ge=0)
    right_triangle: int = Field(ge=0)
    relation: PairRelation


class AuditPredicateResult(_FrozenModel):
    predicate: Literal["self_intersection", "coplanar_overlap"]
    status: AuditStatus
    complete: bool
    finding_count: int = Field(default=0, ge=0)
    within_part_count: int = Field(default=0, ge=0)
    cross_part_count: int = Field(default=0, ge=0)
    unknown_relation_count: int = Field(default=0, ge=0)
    sample_pairs: list[AuditPairFinding] = Field(default_factory=list)
    reason: str | None = None


class AuditResourceEvidence(_FrozenModel):
    """Counters proving which independent budget stopped an audit."""

    input_triangle_count: int = Field(ge=0)
    evaluated_triangle_count: int = Field(ge=0)
    chunk_size: int = Field(ge=1)
    chunk_count: int = Field(ge=0)
    chunks_completed: int = Field(ge=0)
    broad_pair_limit: int = Field(ge=1)
    broad_pairs_examined: int = Field(ge=0)
    adjacency_pairs_skipped: int = Field(ge=0)
    exact_test_limit: int = Field(ge=1)
    exact_pair_tests: int = Field(ge=0)
    wall_time_limit_s: float = Field(gt=0.0)
    elapsed_s: float = Field(ge=0.0)
    memory_limit_bytes: int = Field(ge=1024)
    estimated_working_set_bytes: int = Field(ge=0)
    resource_limit: ResourceLimitKind | None = None
    stopped_at_triangle: int | None = Field(default=None, ge=0)


class ScalableMeshAuditReport(_FrozenModel):
    """Intersection predicates and complete resource evidence for one mesh."""

    schema_version: Literal["geometry-repair.scalable-audit.v1"] = SCALABLE_AUDIT_SCHEMA_VERSION
    geometry_sha256: str = Field(min_length=64, max_length=64)
    position_weld_tolerance: float = Field(ge=0.0)
    self_intersection: AuditPredicateResult
    coplanar_overlap: AuditPredicateResult
    resources: AuditResourceEvidence
    warnings: list[str] = Field(default_factory=list)

    def mesh_metrics_compatibility_fields(self) -> dict[str, int | str | None]:
        """Return the loss-aware subset accepted by the legacy ``MeshMetrics`` model.

        The legacy model cannot distinguish resource skips from indeterminate
        predicates. Both therefore map to ``not_evaluated``; callers must retain this
        full report as the authoritative status and budget evidence.
        """

        def legacy_status(status: AuditStatus) -> str:
            if status == "evaluated_pass":
                return "pass"
            if status == "evaluated_fail":
                return "fail"
            return "not_evaluated"

        reason = self.self_intersection.reason
        if self.self_intersection.status not in {"evaluated_pass", "evaluated_fail"}:
            reason = f"{self.self_intersection.status}: {reason or 'no reason reported'}"
        return {
            "self_intersection_status": legacy_status(self.self_intersection.status),
            "self_intersection_count": self.self_intersection.finding_count,
            "self_intersection_broad_phase_pairs": self.resources.broad_pairs_examined,
            "self_intersection_candidate_pairs": self.resources.exact_pair_tests,
            "self_intersection_reason": reason,
            "coplanar_overlap_status": legacy_status(self.coplanar_overlap.status),
            "coplanar_overlap_count": self.coplanar_overlap.finding_count,
        }


def _geometry_digest(
    vertices: np.ndarray,
    triangles: np.ndarray,
    part_ids: tuple[str, ...] | None,
) -> str:
    digest = hashlib.sha256()
    stable_vertices = np.ascontiguousarray(vertices, dtype="<f8")
    stable_triangles = np.ascontiguousarray(triangles, dtype="<i8")
    digest.update(str(stable_vertices.shape).encode("ascii"))
    digest.update(stable_vertices.tobytes())
    digest.update(str(stable_triangles.shape).encode("ascii"))
    digest.update(stable_triangles.tobytes())
    if part_ids is not None:
        for part_id in part_ids:
            encoded = part_id.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
    return digest.hexdigest()


def _estimated_working_set_bytes(vertex_count: int, triangle_count: int) -> int:
    """Conservative estimate for NumPy arrays plus the R-tree node payload."""

    input_views = vertex_count * (3 * 8 + 8)
    per_triangle = (
        3 * 3 * 8  # triangle coordinates
        + 2 * 3 * 8  # AABB values
        + 3 * 8  # positional topology IDs
        + 3 * 3 * 8  # quantization/temporary exact-test storage
        + 160  # R-tree node and Python query overhead allowance
    )
    return int(input_views + triangle_count * per_triangle + 4 * 1024 * 1024)


def _result_for_terminal_state(
    *,
    status: AuditStatus,
    reason: str,
) -> tuple[AuditPredicateResult, AuditPredicateResult]:
    return (
        AuditPredicateResult(
            predicate="self_intersection",
            status=status,
            complete=False,
            reason=reason,
        ),
        AuditPredicateResult(
            predicate="coplanar_overlap",
            status=status,
            complete=False,
            reason=reason,
        ),
    )


def _resource_evidence(
    *,
    budget: ScalableAuditBudget,
    input_triangle_count: int,
    evaluated_triangle_count: int,
    estimated_bytes: int,
    start_time: float,
    chunk_count: int = 0,
    chunks_completed: int = 0,
    broad_pairs: int = 0,
    adjacency_skipped: int = 0,
    exact_tests: int = 0,
    resource_limit: ResourceLimitKind | None = None,
    stopped_at_triangle: int | None = None,
) -> AuditResourceEvidence:
    return AuditResourceEvidence(
        input_triangle_count=input_triangle_count,
        evaluated_triangle_count=evaluated_triangle_count,
        chunk_size=budget.chunk_size,
        chunk_count=chunk_count,
        chunks_completed=chunks_completed,
        broad_pair_limit=budget.broad_pair_limit,
        broad_pairs_examined=broad_pairs,
        adjacency_pairs_skipped=adjacency_skipped,
        exact_test_limit=budget.exact_test_limit,
        exact_pair_tests=exact_tests,
        wall_time_limit_s=budget.wall_time_s,
        elapsed_s=max(0.0, time.monotonic() - start_time),
        memory_limit_bytes=budget.memory_limit_bytes,
        estimated_working_set_bytes=estimated_bytes,
        resource_limit=resource_limit,
        stopped_at_triangle=stopped_at_triangle,
    )


def _relation(part_ids: tuple[str, ...] | None, left: int, right: int) -> PairRelation:
    if part_ids is None:
        return "unknown"
    return "within_part" if part_ids[left] == part_ids[right] else "cross_part"


def _predicate_result(
    *,
    predicate: Literal["self_intersection", "coplanar_overlap"],
    findings: list[AuditPairFinding],
    finding_count: int,
    relation_counts: dict[PairRelation, int],
    complete: bool,
    terminal_status: AuditStatus | None,
    reason: str | None,
) -> AuditPredicateResult:
    if finding_count:
        status: AuditStatus = "evaluated_fail"
        predicate_reason = reason if not complete else None
    elif complete:
        status = "evaluated_pass"
        predicate_reason = None
    else:
        status = terminal_status or "indeterminate"
        predicate_reason = reason
    return AuditPredicateResult(
        predicate=predicate,
        status=status,
        complete=complete,
        finding_count=finding_count,
        within_part_count=relation_counts["within_part"],
        cross_part_count=relation_counts["cross_part"],
        unknown_relation_count=relation_counts["unknown"],
        sample_pairs=findings,
        reason=predicate_reason,
    )


def audit_triangle_mesh(
    vertices: np.ndarray,
    triangles: np.ndarray,
    *,
    part_ids: list[str] | tuple[str, ...] | np.ndarray | None = None,
    budget: ScalableAuditBudget | None = None,
) -> ScalableMeshAuditReport:
    """Audit triangle intersections without changing input arrays.

    A concrete witness is conclusive and remains ``evaluated_fail`` even when a
    later resource limit prevents exhaustive counting. A zero-finding result is
    ``evaluated_pass`` only after every non-adjacent broad-phase pair was tested.
    """

    selected_budget = budget or ScalableAuditBudget()
    source_vertices = np.asarray(vertices)
    source_triangles = np.asarray(triangles)
    if source_vertices.ndim != 2 or source_vertices.shape[1:] != (3,):
        raise ValueError("vertices must have shape (N, 3)")
    if source_triangles.ndim != 2 or source_triangles.shape[1:] != (3,):
        raise ValueError("triangles must have shape (M, 3)")
    stable_vertices = np.asarray(source_vertices, dtype=np.float64)
    stable_triangles = np.asarray(source_triangles, dtype=np.int64)
    stable_part_ids = (
        tuple(str(item) for item in np.asarray(part_ids).reshape(-1))
        if part_ids is not None
        else None
    )
    if stable_part_ids is not None and len(stable_part_ids) != len(stable_triangles):
        raise ValueError("part_ids must contain one value per triangle")

    start_time = time.monotonic()
    digest = _geometry_digest(stable_vertices, stable_triangles, stable_part_ids)
    triangle_count = len(stable_triangles)
    estimated_bytes = _estimated_working_set_bytes(len(stable_vertices), triangle_count)
    empty_resource = _resource_evidence(
        budget=selected_budget,
        input_triangle_count=triangle_count,
        evaluated_triangle_count=0,
        estimated_bytes=estimated_bytes,
        start_time=start_time,
    )

    if not np.isfinite(stable_vertices).all():
        reason = "mesh contains non-finite vertex coordinates"
        left, right = _result_for_terminal_state(status="indeterminate", reason=reason)
        return ScalableMeshAuditReport(
            geometry_sha256=digest,
            position_weld_tolerance=0.0,
            self_intersection=left,
            coplanar_overlap=right,
            resources=empty_resource,
        )
    if len(stable_triangles) and not np.all(
        (stable_triangles >= 0) & (stable_triangles < len(stable_vertices))
    ):
        reason = "mesh contains out-of-range triangle indices"
        left, right = _result_for_terminal_state(status="indeterminate", reason=reason)
        return ScalableMeshAuditReport(
            geometry_sha256=digest,
            position_weld_tolerance=0.0,
            self_intersection=left,
            coplanar_overlap=right,
            resources=empty_resource,
        )
    if estimated_bytes > selected_budget.memory_limit_bytes:
        reason = (
            f"estimated working set {estimated_bytes} bytes exceeds memory budget "
            f"{selected_budget.memory_limit_bytes} bytes"
        )
        left, right = _result_for_terminal_state(
            status="skipped_resource_limit",
            reason=reason,
        )
        resources = _resource_evidence(
            budget=selected_budget,
            input_triangle_count=triangle_count,
            evaluated_triangle_count=0,
            estimated_bytes=estimated_bytes,
            start_time=start_time,
            resource_limit="memory",
        )
        return ScalableMeshAuditReport(
            geometry_sha256=digest,
            position_weld_tolerance=0.0,
            self_intersection=left,
            coplanar_overlap=right,
            resources=resources,
        )

    if not triangle_count:
        left = AuditPredicateResult(
            predicate="self_intersection",
            status="evaluated_pass",
            complete=True,
        )
        right = AuditPredicateResult(
            predicate="coplanar_overlap",
            status="evaluated_pass",
            complete=True,
        )
        return ScalableMeshAuditReport(
            geometry_sha256=digest,
            position_weld_tolerance=0.0,
            self_intersection=left,
            coplanar_overlap=right,
            resources=_resource_evidence(
                budget=selected_budget,
                input_triangle_count=0,
                evaluated_triangle_count=0,
                estimated_bytes=estimated_bytes,
                start_time=start_time,
                chunk_count=0,
                chunks_completed=0,
            ),
        )

    minimum = stable_vertices.min(axis=0)
    maximum = stable_vertices.max(axis=0)
    diagonal = float(np.linalg.norm(maximum - minimum))
    tolerance = max(diagonal * 1e-10, 1e-12)
    coordinates = stable_vertices[stable_triangles]
    doubled_areas = np.linalg.norm(
        np.cross(coordinates[:, 1] - coordinates[:, 0], coordinates[:, 2] - coordinates[:, 0]),
        axis=1,
    )
    if not np.isfinite(doubled_areas).all() or np.any(doubled_areas <= tolerance * tolerance):
        reason = "mesh contains degenerate triangles that make exact predicates indeterminate"
        left, right = _result_for_terminal_state(status="indeterminate", reason=reason)
        return ScalableMeshAuditReport(
            geometry_sha256=digest,
            position_weld_tolerance=tolerance,
            self_intersection=left,
            coplanar_overlap=right,
            resources=_resource_evidence(
                budget=selected_budget,
                input_triangle_count=triangle_count,
                evaluated_triangle_count=0,
                estimated_bytes=estimated_bytes,
                start_time=start_time,
            ),
        )

    bounds = np.column_stack(
        (coordinates.min(axis=1) - tolerance, coordinates.max(axis=1) + tolerance)
    )
    topology_ids = _topology_vertex_ids(stable_vertices, diagonal)
    topology_triangles = topology_ids[stable_triangles]
    chunk_count = math.ceil(triangle_count / selected_budget.chunk_size)
    if time.monotonic() - start_time > selected_budget.wall_time_s:
        reason = "wall-time budget expired before broad-phase construction"
        left, right = _result_for_terminal_state(
            status="skipped_resource_limit",
            reason=reason,
        )
        return ScalableMeshAuditReport(
            geometry_sha256=digest,
            position_weld_tolerance=tolerance,
            self_intersection=left,
            coplanar_overlap=right,
            resources=_resource_evidence(
                budget=selected_budget,
                input_triangle_count=triangle_count,
                evaluated_triangle_count=0,
                estimated_bytes=estimated_bytes,
                start_time=start_time,
                chunk_count=chunk_count,
                resource_limit="wall_time",
            ),
        )

    try:
        import trimesh

        tree = trimesh.util.bounds_tree(bounds)
    except MemoryError:
        reason = "broad-phase spatial index exhausted the declared memory budget"
        left, right = _result_for_terminal_state(
            status="skipped_resource_limit",
            reason=reason,
        )
        return ScalableMeshAuditReport(
            geometry_sha256=digest,
            position_weld_tolerance=tolerance,
            self_intersection=left,
            coplanar_overlap=right,
            resources=_resource_evidence(
                budget=selected_budget,
                input_triangle_count=triangle_count,
                evaluated_triangle_count=0,
                estimated_bytes=estimated_bytes,
                start_time=start_time,
                chunk_count=chunk_count,
                resource_limit="memory",
            ),
        )
    except Exception as exc:
        reason = f"broad-phase spatial index failed: {type(exc).__name__}: {exc}"
        left, right = _result_for_terminal_state(status="indeterminate", reason=reason)
        return ScalableMeshAuditReport(
            geometry_sha256=digest,
            position_weld_tolerance=tolerance,
            self_intersection=left,
            coplanar_overlap=right,
            resources=_resource_evidence(
                budget=selected_budget,
                input_triangle_count=triangle_count,
                evaluated_triangle_count=0,
                estimated_bytes=estimated_bytes,
                start_time=start_time,
                chunk_count=chunk_count,
            ),
        )

    self_samples: list[AuditPairFinding] = []
    coplanar_samples: list[AuditPairFinding] = []
    self_count = 0
    coplanar_count = 0
    self_relations: dict[PairRelation, int] = {
        "within_part": 0,
        "cross_part": 0,
        "unknown": 0,
    }
    coplanar_relations = dict(self_relations)
    broad_pairs = 0
    exact_tests = 0
    adjacency_skipped = 0
    chunks_completed = 0
    stopped_at_triangle = None
    resource_limit: ResourceLimitKind | None = None
    terminal_status: AuditStatus | None = None
    terminal_reason = None

    try:
        stop = False
        for chunk_start in range(0, triangle_count, selected_budget.chunk_size):
            if time.monotonic() - start_time > selected_budget.wall_time_s:
                resource_limit = "wall_time"
                terminal_status = "skipped_resource_limit"
                terminal_reason = "wall-time budget expired between deterministic chunks"
                stopped_at_triangle = chunk_start
                break
            chunk_end = min(triangle_count, chunk_start + selected_budget.chunk_size)
            for index in range(chunk_start, chunk_end):
                stopped_at_triangle = index
                candidates = sorted(int(item) for item in tree.intersection(bounds[index]))
                for other in candidates:
                    if other <= index:
                        continue
                    if broad_pairs >= selected_budget.broad_pair_limit:
                        resource_limit = "broad_pairs"
                        terminal_status = "skipped_resource_limit"
                        terminal_reason = (
                            "broad-phase pair budget exhausted before exhaustive evaluation"
                        )
                        stop = True
                        break
                    broad_pairs += 1
                    if bool(
                        np.any(
                            topology_triangles[index, :, None] == topology_triangles[other, None, :]
                        )
                    ):
                        adjacency_skipped += 1
                        continue
                    if exact_tests >= selected_budget.exact_test_limit:
                        resource_limit = "exact_tests"
                        terminal_status = "skipped_resource_limit"
                        terminal_reason = "exact-test budget exhausted before exhaustive evaluation"
                        stop = True
                        break
                    exact_tests += 1
                    relation = _relation(stable_part_ids, index, other)
                    pair = AuditPairFinding(
                        left_triangle=index,
                        right_triangle=other,
                        relation=relation,
                    )
                    if _triangles_intersect(
                        coordinates[index],
                        coordinates[other],
                        tolerance,
                    ):
                        self_count += 1
                        self_relations[relation] += 1
                        if len(self_samples) < selected_budget.finding_sample_limit:
                            self_samples.append(pair)
                    if _triangles_overlap_coplanar(
                        coordinates[index],
                        coordinates[other],
                        tolerance,
                    ):
                        coplanar_count += 1
                        coplanar_relations[relation] += 1
                        if len(coplanar_samples) < selected_budget.finding_sample_limit:
                            coplanar_samples.append(pair)
                    if exact_tests % 256 == 0 and (
                        time.monotonic() - start_time > selected_budget.wall_time_s
                    ):
                        resource_limit = "wall_time"
                        terminal_status = "skipped_resource_limit"
                        terminal_reason = "wall-time budget expired during exact pair evaluation"
                        stop = True
                        break
                if stop:
                    break
            if stop:
                break
            chunks_completed += 1
        else:
            stopped_at_triangle = None
    except MemoryError:
        resource_limit = "memory"
        terminal_status = "skipped_resource_limit"
        terminal_reason = "audit exhausted memory while enumerating broad-phase pairs"
    except Exception as exc:
        terminal_status = "indeterminate"
        terminal_reason = f"exact audit failed: {type(exc).__name__}: {exc}"

    complete = terminal_status is None
    self_result = _predicate_result(
        predicate="self_intersection",
        findings=self_samples,
        finding_count=self_count,
        relation_counts=self_relations,
        complete=complete,
        terminal_status=terminal_status,
        reason=terminal_reason,
    )
    coplanar_result = _predicate_result(
        predicate="coplanar_overlap",
        findings=coplanar_samples,
        finding_count=coplanar_count,
        relation_counts=coplanar_relations,
        complete=complete,
        terminal_status=terminal_status,
        reason=terminal_reason,
    )
    resources = _resource_evidence(
        budget=selected_budget,
        input_triangle_count=triangle_count,
        evaluated_triangle_count=(triangle_count if complete else max(0, stopped_at_triangle or 0)),
        estimated_bytes=estimated_bytes,
        start_time=start_time,
        chunk_count=chunk_count,
        chunks_completed=chunks_completed,
        broad_pairs=broad_pairs,
        adjacency_skipped=adjacency_skipped,
        exact_tests=exact_tests,
        resource_limit=resource_limit,
        stopped_at_triangle=stopped_at_triangle,
    )
    warnings = []
    if terminal_reason and (self_count or coplanar_count):
        warnings.append(
            "a defect witness is conclusive, but finding counts are lower bounds because the "
            "audit did not complete"
        )
    return ScalableMeshAuditReport(
        geometry_sha256=digest,
        position_weld_tolerance=tolerance,
        self_intersection=self_result,
        coplanar_overlap=coplanar_result,
        resources=resources,
        warnings=warnings,
    )


__all__ = [
    "SCALABLE_AUDIT_SCHEMA_VERSION",
    "AuditPairFinding",
    "AuditPredicateResult",
    "AuditResourceEvidence",
    "ScalableAuditBudget",
    "ScalableMeshAuditReport",
    "audit_triangle_mesh",
]
