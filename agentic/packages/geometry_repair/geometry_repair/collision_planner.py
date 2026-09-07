# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed, deterministic candidate selection for rigid collision geometry."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .artifacts import atomic_write_json
from .models import ProtectedFeatureProbeResult, RepairBudgets

COLLISION_CANDIDATE_SEARCH_SCHEMA_VERSION = "geometry-repair.collision-candidate-search.v1"

CandidateRepresentation = Literal[
    "source_collision",
    "primitive_box",
    "primitive_sphere",
    "primitive_cylinder",
    "primitive_capsule",
    "convex_hull",
    "coacd",
]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CollisionCandidateGate(_StrictModel):
    """One immutable hard decision gate for a collision candidate."""

    gate_id: str
    status: Literal["pass", "fail", "not_evaluated"]
    measured: float | int | bool | None = None
    limit: float | int | bool | None = None
    units: str | None = None
    reason: str


class CollisionCandidateEvaluation(_StrictModel):
    """Measured fidelity, complexity, and intent evidence for one representation."""

    candidate_id: str
    representation: CandidateRepresentation
    generator: str
    authored_source: bool = False
    artifact_path: str | None = None
    artifact_sha256: str | None = Field(default=None, min_length=64, max_length=64)
    hull_count: int = Field(ge=1)
    total_vertices: int = Field(ge=4)
    total_faces: int = Field(ge=4)
    maximum_vertices_per_hull: int = Field(ge=4)
    maximum_faces_per_hull: int = Field(ge=4)
    metrics: dict[str, float]
    protected_features: list[str] = Field(default_factory=list)
    protected_features_satisfied: bool | None = None
    protected_feature_probes: list[ProtectedFeatureProbeResult] = Field(default_factory=list)
    gates: list[CollisionCandidateGate]
    status: Literal["pass", "conditional", "fail"]
    score: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def _validate_decision(self) -> CollisionCandidateEvaluation:
        expected = (
            "fail"
            if any(gate.status == "fail" for gate in self.gates)
            else "conditional"
            if any(gate.status == "not_evaluated" for gate in self.gates)
            else "pass"
        )
        if self.status != expected:
            raise ValueError("candidate status must match its hard and unmeasured gates")
        if self.status in {"pass", "conditional"} and self.score is None:
            raise ValueError("selectable collision candidates require a deterministic score")
        if self.status == "fail" and self.score is not None:
            raise ValueError("failed collision candidates cannot participate in scoring")
        return self


class CollisionCandidateSearch(_StrictModel):
    """Complete candidate ledger and deterministic selection for one render part."""

    schema_version: Literal["geometry-repair.collision-candidate-search.v1"] = (
        COLLISION_CANDIDATE_SEARCH_SCHEMA_VERSION
    )
    render_path: str
    status: Literal["pass", "conditional", "fail"]
    candidates: list[CollisionCandidateEvaluation]
    selected_candidate_id: str | None = None
    selection_policy: Literal["certified_source_then_lowest_normalized_error"] = (
        "certified_source_then_lowest_normalized_error"
    )
    runtime_gate: Literal["deferred_to_selected_composed_collision"] = (
        "deferred_to_selected_composed_collision"
    )
    failures: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    report_path: str | None = None

    @model_validator(mode="after")
    def _validate_selection(self) -> CollisionCandidateSearch:
        identifiers = [candidate.candidate_id for candidate in self.candidates]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("collision candidate IDs must be unique per render part")
        selectable = {
            candidate.candidate_id
            for candidate in self.candidates
            if candidate.status in {"pass", "conditional"}
        }
        if self.status in {"pass", "conditional"}:
            if self.selected_candidate_id not in selectable or self.failures:
                raise ValueError("selectable search requires one selected candidate")
            selected = next(
                item for item in self.candidates if item.candidate_id == self.selected_candidate_id
            )
            if self.status != selected.status:
                raise ValueError("search status must match the selected candidate")
            if self.status == "conditional" and not self.warnings:
                raise ValueError("conditional search requires an explicit warning")
        elif self.selected_candidate_id is not None or not self.failures:
            raise ValueError("failed search requires no selection and at least one failure")
        return self


_REQUIRED_METRICS = (
    "volume_excess_ratio",
    "volume_deficit_ratio",
    "false_positive_ratio",
    "false_negative_ratio",
    "surface_gap_m",
    "surface_overreach_m",
)
_REPRESENTATION_ORDER = {
    "primitive_box": 0,
    "primitive_sphere": 1,
    "primitive_cylinder": 2,
    "primitive_capsule": 3,
    "convex_hull": 4,
    "coacd": 5,
    "source_collision": 6,
}


def evaluate_collision_candidate(
    *,
    candidate_id: str,
    representation: CandidateRepresentation,
    generator: str,
    metrics: dict[str, float],
    hull_count: int,
    total_vertices: int,
    total_faces: int,
    maximum_vertices_per_hull: int,
    maximum_faces_per_hull: int,
    diagonal_m: float,
    budgets: RepairBudgets,
    remaining_hull_budget: int,
    protected_features: list[str] | None = None,
    protected_features_satisfied: bool | None = None,
    protected_feature_probes: list[ProtectedFeatureProbeResult] | None = None,
    authored_source: bool = False,
    artifact_path: str | None = None,
    artifact_sha256: str | None = None,
) -> CollisionCandidateEvaluation:
    """Evaluate one candidate without relaxing any profile or resource limit."""

    missing = [name for name in _REQUIRED_METRICS if name not in metrics]
    if missing:
        raise ValueError(f"collision candidate is missing metrics: {', '.join(missing)}")
    values = {name: float(metrics[name]) for name in _REQUIRED_METRICS}
    if any(not math.isfinite(value) or value < 0.0 for value in values.values()):
        raise ValueError("collision candidate metrics must be finite and non-negative")
    surface_limit_m = max(
        float(diagonal_m) * budgets.max_collision_surface_distance_ratio,
        1e-6,
    )
    features = sorted(set(protected_features or []))
    hull_limit = (
        budgets.max_source_collision_prims
        if authored_source
        else min(remaining_hull_budget, budgets.max_collision_hulls)
    )
    gate_specs: list[tuple[str, float | int | bool, float | int | bool, str, str]] = [
        (
            "hull_count",
            hull_count,
            hull_limit,
            "count",
            "candidate hull count must fit the remaining generated-hull budget",
        ),
        (
            "maximum_vertices_per_hull",
            maximum_vertices_per_hull,
            budgets.max_collision_vertices_per_hull,
            "count",
            "every convex hull must satisfy the target runtime vertex budget",
        ),
        (
            "maximum_faces_per_hull",
            maximum_faces_per_hull,
            budgets.max_collision_faces_per_hull,
            "count",
            "every convex hull must satisfy the target runtime face budget",
        ),
        (
            "volume_excess_ratio",
            values["volume_excess_ratio"],
            budgets.max_collision_volume_excess,
            "ratio",
            "collision volume excess must remain inside the profile limit",
        ),
        (
            "volume_deficit_ratio",
            values["volume_deficit_ratio"],
            budgets.max_collision_volume_deficit,
            "ratio",
            "collision volume deficit must remain inside the profile limit",
        ),
        (
            "false_positive_ratio",
            values["false_positive_ratio"],
            budgets.max_collision_volume_excess,
            "ratio",
            "collision occupied-space overreach must remain inside the profile limit",
        ),
        (
            "false_negative_ratio",
            values["false_negative_ratio"],
            budgets.max_collision_volume_deficit,
            "ratio",
            "collision occupied-space deficit must remain inside the profile limit",
        ),
        (
            "surface_gap_m",
            values["surface_gap_m"],
            surface_limit_m,
            "m",
            "render-to-collision surface gap must remain inside the profile limit",
        ),
        (
            "surface_overreach_m",
            values["surface_overreach_m"],
            surface_limit_m,
            "m",
            "collision-to-render surface overreach must remain inside the profile limit",
        ),
    ]
    gates = [
        CollisionCandidateGate(
            gate_id=gate_id,
            status="pass" if measured <= limit else "fail",
            measured=measured,
            limit=limit,
            units=units,
            reason=reason,
        )
        for gate_id, measured, limit, units, reason in gate_specs
    ]
    if features:
        gates.append(
            CollisionCandidateGate(
                gate_id="protected_features",
                status=(
                    "pass"
                    if protected_features_satisfied is True
                    else "fail"
                    if protected_features_satisfied is False
                    else "not_evaluated"
                ),
                measured=protected_features_satisfied,
                limit=True,
                reason=(
                    "all required protected collision features need explicit candidate evidence"
                ),
            )
        )
    failed = any(gate.status == "fail" for gate in gates)
    conditional = not failed and any(gate.status == "not_evaluated" for gate in gates)
    score = None
    if not failed:
        normalized_error = (
            values["volume_excess_ratio"] / max(budgets.max_collision_volume_excess, 1e-12)
            + values["volume_deficit_ratio"] / max(budgets.max_collision_volume_deficit, 1e-12)
            + values["false_positive_ratio"] / max(budgets.max_collision_volume_excess, 1e-12)
            + values["false_negative_ratio"] / max(budgets.max_collision_volume_deficit, 1e-12)
            + values["surface_gap_m"] / surface_limit_m
            + values["surface_overreach_m"] / surface_limit_m
        )
        complexity = (
            hull_count / max(budgets.max_collision_hulls, 1)
            + total_vertices
            / max(budgets.max_collision_hulls * budgets.max_collision_vertices_per_hull, 1)
            + total_faces
            / max(budgets.max_collision_hulls * budgets.max_collision_faces_per_hull, 1)
        )
        score = normalized_error + 0.05 * complexity
    return CollisionCandidateEvaluation(
        candidate_id=candidate_id,
        representation=representation,
        generator=generator,
        authored_source=authored_source,
        artifact_path=artifact_path,
        artifact_sha256=artifact_sha256,
        hull_count=hull_count,
        total_vertices=total_vertices,
        total_faces=total_faces,
        maximum_vertices_per_hull=maximum_vertices_per_hull,
        maximum_faces_per_hull=maximum_faces_per_hull,
        metrics=values,
        protected_features=features,
        protected_features_satisfied=protected_features_satisfied,
        protected_feature_probes=protected_feature_probes or [],
        gates=gates,
        status="fail" if failed else "conditional" if conditional else "pass",
        score=score,
    )


def select_collision_candidate(
    render_path: str,
    candidates: list[CollisionCandidateEvaluation],
    *,
    report_path: str | Path | None = None,
    failure_context: list[str] | None = None,
) -> CollisionCandidateSearch:
    """Select a certified source, then the lowest-error generated candidate."""

    selectable = [candidate for candidate in candidates if candidate.status != "fail"]
    selected = min(
        selectable,
        key=lambda candidate: (
            0 if candidate.status == "pass" else 1,
            0 if candidate.authored_source else 1,
            candidate.score if candidate.score is not None else math.inf,
            _REPRESENTATION_ORDER[candidate.representation],
            candidate.candidate_id,
        ),
        default=None,
    )
    resolved_report = (
        str(Path(report_path).expanduser().resolve()) if report_path is not None else None
    )
    probe_failures = sorted(
        {
            f"protected collision feature {probe.feature_name!r}: {failure}"
            for candidate in candidates
            for probe in candidate.protected_feature_probes
            if probe.status == "fail"
            for failure in probe.failures
        }
    )
    search = CollisionCandidateSearch(
        render_path=render_path,
        status=selected.status if selected is not None else "fail",
        candidates=sorted(candidates, key=lambda candidate: candidate.candidate_id),
        selected_candidate_id=selected.candidate_id if selected is not None else None,
        failures=(
            []
            if selected is not None
            else [
                "no collision candidate satisfied every fidelity, complexity, and intent gate",
                *probe_failures,
                *sorted(set(failure_context or [])),
            ]
        ),
        warnings=(
            [
                "selected collision candidate has unmeasured protected-feature evidence; "
                "the collision handoff remains conditional"
            ]
            if selected is not None and selected.status == "conditional"
            else []
        ),
        report_path=resolved_report,
    )
    if report_path is not None:
        atomic_write_json(Path(report_path), search)
    return search
