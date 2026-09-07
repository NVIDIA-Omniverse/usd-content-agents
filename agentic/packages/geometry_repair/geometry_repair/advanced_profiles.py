# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Geometry-only evidence for articulated and contact-rich repair profiles.

This module deliberately does not author joints, physics schemas, drives, or
runtime claims.  It evaluates caller-supplied geometric intent against explicit
render/collision mappings and returns evidence for the parent orchestrator.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from pydantic import Field, model_validator

from .artifacts import atomic_write_json, file_sha256
from .mesh_io import MeshData, _triangles_intersect, load_meshes, positional_topology_mesh
from .models import StrictModel

ADVANCED_PROFILE_SCHEMA_VERSION = "geometry-repair.advanced-profile-evidence.v1"
ADVANCED_PROFILE_REQUEST_SCHEMA_VERSION = "geometry-repair.advanced-profile-request.v1"
ADVANCED_PROFILE_REPORT_SCHEMA_VERSION = "geometry-repair.advanced-profile-report.v1"
ADVANCED_PROFILE_HANDOFF_SCHEMA_VERSION = "geometry-repair.advanced-profile-handoff.v1"

EvidenceStatus = Literal["pass", "fail", "not_evaluated", "indeterminate"]
EvidenceFactKind = Literal["source_fact", "measured_fact", "inferred_hypothesis"]
AdvancedProfileName = Literal["articulated_rigid", "contact_rich"]
GeometryOnlyClaimScope = Literal[
    "geometry_repair.articulated_rigid.geometry_only",
    "geometry_repair.contact_rich.geometry_only",
]
DownstreamOwner = Literal["articulation", "physics", "runtime_validation", "simready"]
DownstreamWorkflow = Literal[
    "content-workflow-articulation",
    "content-workflow-physics",
    "content-workflow-runtime-validation",
    "content-workflow-simready",
]
RouteReadiness = Literal["ready", "conditional", "blocked"]
AdvancedProfileArtifactRole = Literal["accepted_render_usd", "accepted_collision_usd"]
ArtifactAvailability = Literal["available", "not_evaluated"]
_ArtifactReferenceCache = dict[
    tuple[AdvancedProfileArtifactRole, str | None, str | None],
    tuple[str | None, str | None, ArtifactAvailability, list[str]],
]

_DOWNSTREAM_OWNER_ORDER: tuple[DownstreamOwner, ...] = (
    "articulation",
    "physics",
    "runtime_validation",
    "simready",
)


def _geometry_claim_scope(profile: AdvancedProfileName) -> GeometryOnlyClaimScope:
    if profile == "articulated_rigid":
        return "geometry_repair.articulated_rigid.geometry_only"
    return "geometry_repair.contact_rich.geometry_only"


def _finite_vector(value: tuple[float, float, float], *, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be a finite XYZ vector")
    return vector


def _authoritative(evidence: list[SourceEvidence]) -> bool:
    return any(item.fact_kind in {"source_fact", "measured_fact"} for item in evidence)


def _combined_status(checks: list[GeometryCheckEvidence]) -> EvidenceStatus:
    statuses = [check.status for check in checks if check.required]
    if not statuses:
        return "not_evaluated"
    for status in ("fail", "indeterminate", "not_evaluated"):
        if status in statuses:
            return cast(EvidenceStatus, status)
    return "pass"


class SourceEvidence(StrictModel):
    """Provenance for a semantic or geometric assertion."""

    fact_kind: EvidenceFactKind
    source_ref: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)


class SemanticLinkHypothesis(StrictModel):
    """Explicit part/link proposal; it carries no joint-authoring authority."""

    link_id: str = Field(min_length=1)
    semantic_label: str = Field(min_length=1)
    source_part_paths: list[str] = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[SourceEvidence] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_unique_paths(self) -> SemanticLinkHypothesis:
        if len(self.source_part_paths) != len(set(self.source_part_paths)):
            raise ValueError("source_part_paths must not contain duplicates")
        return self


class LinkGeometryMapping(StrictModel):
    """Caller-declared render and collision prims belonging to one rigid link."""

    link_id: str = Field(min_length=1)
    render_paths: list[str] = Field(min_length=1)
    collision_paths: list[str] = Field(min_length=1)
    evidence: list[SourceEvidence] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_unique_paths(self) -> LinkGeometryMapping:
        for name, paths in (
            ("render_paths", self.render_paths),
            ("collision_paths", self.collision_paths),
        ):
            if len(paths) != len(set(paths)):
                raise ValueError(f"{name} must not contain duplicates")
        return self


class JointSweepInput(StrictModel):
    """Caller-supplied joint geometry used only for sampled spatial checks."""

    joint_id: str = Field(min_length=1)
    parent_link_id: str = Field(min_length=1)
    child_link_id: str = Field(min_length=1)
    moving_link_ids: list[str] = Field(min_length=1)
    joint_type: Literal["revolute", "prismatic"]
    axis_world: tuple[float, float, float] | None = None
    origin_world_m: tuple[float, float, float] | None = None
    lower_limit: float | None = None
    upper_limit: float | None = None
    reference_value: float | None = None
    units: Literal["radians", "meters"] | None = None
    sample_count: int = Field(ge=2, le=361)
    input_authority: Literal["caller_supplied"] = "caller_supplied"
    evidence: list[SourceEvidence] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_geometry(self) -> JointSweepInput:
        if self.axis_world is not None:
            axis = _finite_vector(self.axis_world, name="axis_world")
            if float(np.linalg.norm(axis)) <= 1e-12:
                raise ValueError("axis_world must have non-zero length")
        if self.origin_world_m is not None:
            _finite_vector(self.origin_world_m, name="origin_world_m")
        supplied_values = [
            value
            for value in (self.lower_limit, self.upper_limit, self.reference_value)
            if value is not None
        ]
        if not all(math.isfinite(float(value)) for value in supplied_values):
            raise ValueError("supplied joint limits and reference_value must be finite")
        if self.lower_limit is not None and self.upper_limit is not None:
            if self.lower_limit > self.upper_limit:
                raise ValueError("lower_limit must not exceed upper_limit")
            if self.reference_value is not None and not (
                self.lower_limit <= self.reference_value <= self.upper_limit
            ):
                raise ValueError("reference_value must lie inside the supplied limits")
        expected_units = "radians" if self.joint_type == "revolute" else "meters"
        if self.units is not None and self.units != expected_units:
            raise ValueError(f"{self.joint_type} joints require units={expected_units!r}")
        if len(self.moving_link_ids) != len(set(self.moving_link_ids)):
            raise ValueError("moving_link_ids must not contain duplicates")
        if self.child_link_id not in self.moving_link_ids:
            raise ValueError("child_link_id must be included in moving_link_ids")
        if self.parent_link_id in self.moving_link_ids:
            raise ValueError("parent_link_id must not be included in moving_link_ids")
        return self


class AdjacentLinkExclusion(StrictModel):
    """Explicit collision-filter metadata; this does not author runtime filtering."""

    link_a: str = Field(min_length=1)
    link_b: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    evidence: list[SourceEvidence] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_pair(self) -> AdjacentLinkExclusion:
        if self.link_a == self.link_b:
            raise ValueError("an adjacent-link exclusion requires two distinct links")
        return self

    @property
    def key(self) -> tuple[str, str]:
        return tuple(sorted((self.link_a, self.link_b)))


class GeometryCheckEvidence(StrictModel):
    """One deterministic geometry check with explicit evaluation state."""

    check_id: str = Field(min_length=1)
    status: EvidenceStatus
    required: bool = True
    subjects: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    failures: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_status_payload(self) -> GeometryCheckEvidence:
        if self.status == "pass" and self.failures:
            raise ValueError("passing evidence cannot contain failures")
        if self.status == "fail" and not self.failures:
            raise ValueError("failing evidence must explain at least one failure")
        if self.status in {"not_evaluated", "indeterminate"} and not self.warnings:
            raise ValueError(f"{self.status} evidence must explain why it is incomplete")
        return self


class SweptMotionEvidence(StrictModel):
    """Sampled geometric motion result; it makes no dynamics or stability claim."""

    joint_id: str
    status: EvidenceStatus
    requested_sample_count: int = Field(ge=2)
    evaluated_sample_count: int = Field(ge=0)
    sample_values: list[float] = Field(default_factory=list)
    candidate_triangle_pairs_tested: int = Field(default=0, ge=0)
    collision_events: list[dict[str, Any]] = Field(default_factory=list)
    excluded_link_pairs: list[dict[str, Any]] = Field(default_factory=list)
    failures: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ArticulatedGeometryEvidence(StrictModel):
    """Geometry-owned evidence packet for the articulated-rigid profile."""

    schema_version: Literal["geometry-repair.advanced-profile-evidence.v1"] = (
        ADVANCED_PROFILE_SCHEMA_VERSION
    )
    profile: Literal["articulated_rigid"] = "articulated_rigid"
    claim_scope: Literal["geometry_repair.articulated_rigid.geometry_only"] = (
        "geometry_repair.articulated_rigid.geometry_only"
    )
    status: EvidenceStatus
    semantic_links: list[SemanticLinkHypothesis]
    link_mappings: list[LinkGeometryMapping]
    adjacent_link_exclusions: list[AdjacentLinkExclusion]
    configuration_checks: list[GeometryCheckEvidence]
    link_collision_checks: list[GeometryCheckEvidence]
    swept_motion_checks: list[SweptMotionEvidence]
    certification_eligible: bool
    blockers: list[str] = Field(default_factory=list)
    downstream_owners: list[str] = Field(
        default_factory=lambda: [
            "articulation: author and validate joints, limits, drives, and collision filtering",
            "physics: author mass, inertia, friction, restitution, and solver properties",
            "runtime-validation: prove actuation, contact stability, and target-runtime behavior",
        ]
    )


class ContactRichProbeInput(StrictModel):
    """Caller-supplied contact path and tolerances in world-space meters."""

    probe_id: str = Field(min_length=1)
    moving_part_id: str = Field(min_length=1)
    receiver_part_id: str = Field(min_length=1)
    receiver_collision_paths: list[str] | None = None
    axis_origin_world_m: tuple[float, float, float] | None = None
    axis_world: tuple[float, float, float] | None = None
    approach_direction: Literal["along_axis", "against_axis"]
    path_points_world_m: list[tuple[float, float, float]] | None = Field(
        default=None,
        min_length=2,
        max_length=256,
    )
    axis_tolerance_m: float = Field(gt=0.0)
    angular_tolerance_deg: float = Field(gt=0.0, le=90.0)
    moving_envelope_radius_m: float = Field(ge=0.0)
    required_radial_clearance_m: float = Field(ge=0.0)
    maximum_clearance_erosion_m: float = Field(default=0.0, ge=0.0)
    seated_stop_point_world_m: tuple[float, float, float] | None = None
    seated_stop_tolerance_m: float = Field(gt=0.0)
    minimum_protected_feature_m: float = Field(gt=0.0)
    collision_representation: Literal["convex", "sdf", "static_triangle_mesh"]
    sdf_voxel_size_m: float | None = Field(default=None, gt=0.0)
    required_voxels_across_feature: int = Field(default=3, ge=3, le=32)
    radial_surface_normal_tolerance_deg: float = Field(default=20.0, gt=0.0, lt=90.0)
    sample_step_m: float = Field(gt=0.0)
    max_path_samples: int = Field(default=4096, ge=2, le=100_000)
    evidence: list[SourceEvidence] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_probe(self) -> ContactRichProbeInput:
        if self.axis_world is not None:
            axis = _finite_vector(self.axis_world, name="axis_world")
            if float(np.linalg.norm(axis)) <= 1e-12:
                raise ValueError("axis_world must have non-zero length")
        if self.axis_origin_world_m is not None:
            _finite_vector(self.axis_origin_world_m, name="axis_origin_world_m")
        if self.seated_stop_point_world_m is not None:
            _finite_vector(self.seated_stop_point_world_m, name="seated_stop_point_world_m")
        for index, point in enumerate(self.path_points_world_m or []):
            _finite_vector(point, name=f"path_points_world_m[{index}]")
        if self.maximum_clearance_erosion_m >= self.required_radial_clearance_m:
            if self.required_radial_clearance_m > 0.0:
                raise ValueError(
                    "maximum_clearance_erosion_m must remain below required radial clearance"
                )
        return self


class ContactProbeEvidence(StrictModel):
    """Axis, path, clearance, stop, and optional SDF evidence for one pair."""

    probe_id: str
    status: EvidenceStatus
    checks: list[GeometryCheckEvidence]
    sampled_path_points: int = Field(ge=0)


class ContactRichGeometryEvidence(StrictModel):
    """Geometry-owned evidence packet for a contact-rich pair."""

    schema_version: Literal["geometry-repair.advanced-profile-evidence.v1"] = (
        ADVANCED_PROFILE_SCHEMA_VERSION
    )
    profile: Literal["contact_rich"] = "contact_rich"
    claim_scope: Literal["geometry_repair.contact_rich.geometry_only"] = (
        "geometry_repair.contact_rich.geometry_only"
    )
    status: EvidenceStatus
    receiver_collision_path: str | None
    probes: list[ContactRichProbeInput]
    probe_evidence: list[ContactProbeEvidence]
    certification_eligible: bool
    blockers: list[str] = Field(default_factory=list)
    downstream_owners: list[str] = Field(
        default_factory=lambda: [
            "physics: author contact material, mass, inertia, and solver properties",
            "articulation: author any constrained degrees of freedom or joints",
            "runtime-validation: execute approach-to-seat and disturbance tests",
        ]
    )


class AdvancedProfileArtifactReference(StrictModel):
    """Integrity-bound geometry artifact and exact prim selection for one subject."""

    reference_id: str = Field(min_length=1)
    role: AdvancedProfileArtifactRole
    artifact_path: str | None = None
    sha256: str | None = None
    prim_paths: list[str] = Field(default_factory=list)
    availability: ArtifactAvailability
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_reference(self) -> AdvancedProfileArtifactReference:
        if len(self.prim_paths) != len(set(self.prim_paths)):
            raise ValueError("artifact prim_paths must not contain duplicates")
        if self.sha256 is not None and (
            len(self.sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.sha256)
        ):
            raise ValueError("artifact sha256 must be a lowercase hexadecimal digest")
        if self.availability == "available" and (self.artifact_path is None or self.sha256 is None):
            raise ValueError("available artifact references require a path and SHA-256")
        if self.availability == "not_evaluated" and not self.warnings:
            raise ValueError("not-evaluated artifact references must explain missing evidence")
        return self


class LinkArtifactReferences(StrictModel):
    """Accepted render/collision artifacts and evidence locators for one proposed link."""

    subject_ref: str = Field(min_length=1)
    link_id: str = Field(min_length=1)
    render_artifact: AdvancedProfileArtifactReference
    collision_artifact: AdvancedProfileArtifactReference
    mapping_status: EvidenceStatus
    geometry_check_ids: list[str] = Field(min_length=1)
    source_evidence_refs: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_link_artifacts(self) -> LinkArtifactReferences:
        if self.render_artifact.role != "accepted_render_usd":
            raise ValueError("render_artifact must reference accepted render USD")
        if self.collision_artifact.role != "accepted_collision_usd":
            raise ValueError("collision_artifact must reference accepted collision USD")
        for name, values in (
            ("geometry_check_ids", self.geometry_check_ids),
            ("source_evidence_refs", self.source_evidence_refs),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must not contain duplicates")
        if self.render_artifact.reference_id == self.collision_artifact.reference_id:
            raise ValueError("render and collision artifact reference IDs must be distinct")
        return self


class ProbeArtifactReferences(StrictModel):
    """Exact receiver collision artifact and evidence locators for one contact probe."""

    subject_ref: str = Field(min_length=1)
    probe_id: str = Field(min_length=1)
    moving_part_id: str = Field(min_length=1)
    receiver_part_id: str = Field(min_length=1)
    receiver_collision_artifact: AdvancedProfileArtifactReference
    receiver_selection_status: EvidenceStatus
    geometry_check_ids: list[str] = Field(min_length=1)
    source_evidence_refs: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_probe_artifacts(self) -> ProbeArtifactReferences:
        if self.receiver_collision_artifact.role != "accepted_collision_usd":
            raise ValueError("receiver_collision_artifact must reference accepted collision USD")
        for name, values in (
            ("geometry_check_ids", self.geometry_check_ids),
            ("source_evidence_refs", self.source_evidence_refs),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must not contain duplicates")
        return self


class DownstreamRoutingRecord(StrictModel):
    """Deterministic request for work that geometry repair does not own."""

    route_id: str = Field(min_length=1)
    owner: DownstreamOwner
    workflow: DownstreamWorkflow
    status: Literal["not_evaluated"] = "not_evaluated"
    readiness: RouteReadiness
    upstream_claim_scope: GeometryOnlyClaimScope
    artifact_reference_ids: list[str] = Field(default_factory=list)
    subject_refs: list[str] = Field(default_factory=list)
    geometry_check_ids: list[str] = Field(default_factory=list)
    depends_on: list[DownstreamOwner] = Field(default_factory=list)
    required_inputs: list[str] = Field(min_length=1)
    requested_work: list[str] = Field(min_length=1)
    blockers: list[str] = Field(default_factory=list)
    result_claimed: Literal[False] = False

    @model_validator(mode="after")
    def _validate_route(self) -> DownstreamRoutingRecord:
        expected_workflow = {
            "articulation": "content-workflow-articulation",
            "physics": "content-workflow-physics",
            "runtime_validation": "content-workflow-runtime-validation",
            "simready": "content-workflow-simready",
        }[self.owner]
        if self.workflow != expected_workflow:
            raise ValueError(f"owner {self.owner!r} requires workflow {expected_workflow!r}")
        for name, values in (
            ("artifact_reference_ids", self.artifact_reference_ids),
            ("subject_refs", self.subject_refs),
            ("geometry_check_ids", self.geometry_check_ids),
            ("depends_on", self.depends_on),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must not contain duplicates")
        if self.owner in self.depends_on:
            raise ValueError("a downstream route cannot depend on itself")
        if self.readiness != "ready" and not self.blockers:
            raise ValueError("conditional or blocked routes must identify blockers")
        return self


class AdvancedProfileHandoff(StrictModel):
    """Geometry-owned handoff without downstream authoring or certification claims."""

    schema_version: Literal["geometry-repair.advanced-profile-handoff.v1"] = (
        ADVANCED_PROFILE_HANDOFF_SCHEMA_VERSION
    )
    profile: AdvancedProfileName
    claim_scope: GeometryOnlyClaimScope
    geometry_status: EvidenceStatus
    geometry_disposition: Literal["evidence_complete", "conditional", "failed"]
    downstream_status: Literal["not_evaluated"] = "not_evaluated"
    downstream_disposition: Literal["conditional", "blocked"]
    link_artifacts: list[LinkArtifactReferences] = Field(default_factory=list)
    probe_artifacts: list[ProbeArtifactReferences] = Field(default_factory=list)
    downstream_routes: list[DownstreamRoutingRecord] = Field(min_length=4, max_length=4)
    blockers: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_handoff(self) -> AdvancedProfileHandoff:
        expected_scope = _geometry_claim_scope(self.profile)
        if self.claim_scope != expected_scope:
            raise ValueError(f"profile {self.profile!r} requires claim_scope {expected_scope!r}")
        if self.profile == "articulated_rigid" and self.probe_artifacts:
            raise ValueError("articulated_rigid handoffs cannot contain probe artifacts")
        if self.profile == "contact_rich" and self.link_artifacts:
            raise ValueError("contact_rich handoffs cannot contain link artifacts")
        subject_refs = [
            *(item.subject_ref for item in self.link_artifacts),
            *(item.subject_ref for item in self.probe_artifacts),
        ]
        if len(subject_refs) != len(set(subject_refs)):
            raise ValueError("handoff subject_ref values must be unique")
        artifact_references = [
            *(
                reference
                for item in self.link_artifacts
                for reference in (item.render_artifact, item.collision_artifact)
            ),
            *(item.receiver_collision_artifact for item in self.probe_artifacts),
        ]
        artifact_ids = [reference.reference_id for reference in artifact_references]
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError("handoff artifact reference IDs must be unique")
        geometry_check_ids = list(
            dict.fromkeys(
                check_id
                for item in [*self.link_artifacts, *self.probe_artifacts]
                for check_id in item.geometry_check_ids
            )
        )
        owners = [route.owner for route in self.downstream_routes]
        if owners != list(_DOWNSTREAM_OWNER_ORDER):
            raise ValueError(
                "downstream routes must use deterministic articulation, physics, "
                "runtime_validation, simready order"
            )
        expected_dependencies: dict[DownstreamOwner, list[DownstreamOwner]] = {
            "articulation": [],
            "physics": [],
            "runtime_validation": ["articulation", "physics"],
            "simready": ["articulation", "physics", "runtime_validation"],
        }
        artifact_evidence_missing = any(
            reference.availability == "not_evaluated" for reference in artifact_references
        )
        for route in self.downstream_routes:
            if route.route_id != f"{self.profile}:{route.owner}":
                raise ValueError("downstream route_id must be deterministic for profile and owner")
            if route.upstream_claim_scope != self.claim_scope:
                raise ValueError("downstream routes must retain the geometry-only claim scope")
            if route.artifact_reference_ids != artifact_ids:
                raise ValueError("downstream routes must reference every handoff artifact in order")
            if route.subject_refs != subject_refs:
                raise ValueError("downstream routes must reference every handoff subject in order")
            if route.geometry_check_ids != geometry_check_ids:
                raise ValueError("downstream routes must reference every geometry check in order")
            if route.depends_on != expected_dependencies[route.owner]:
                raise ValueError("downstream route dependencies do not match the owner contract")
            expected_readiness: RouteReadiness
            if self.geometry_disposition == "failed":
                expected_readiness = "blocked"
            elif (
                self.geometry_status != "pass"
                or self.geometry_disposition != "evidence_complete"
                or artifact_evidence_missing
                or route.depends_on
            ):
                expected_readiness = "conditional"
            else:
                expected_readiness = "ready"
            if route.readiness != expected_readiness:
                raise ValueError("downstream route readiness does not match available evidence")
        if self.geometry_status == "fail" and self.geometry_disposition != "failed":
            raise ValueError("failed geometry evidence requires a failed geometry disposition")
        if self.geometry_status != "fail" and self.geometry_disposition == "failed":
            raise ValueError("only failed geometry evidence may use a failed disposition")
        expected_downstream_disposition = (
            "blocked" if self.geometry_disposition == "failed" else "conditional"
        )
        if self.downstream_disposition != expected_downstream_disposition:
            raise ValueError(
                "downstream disposition must remain conditional unless geometry failed"
            )
        return self


class AdvancedProfileRequest(StrictModel):
    """Serializable profile intent designed for optional embedding in RepairRequest."""

    schema_version: Literal["geometry-repair.advanced-profile-request.v1"] = (
        ADVANCED_PROFILE_REQUEST_SCHEMA_VERSION
    )
    profile: Literal["articulated_rigid", "contact_rich"]
    semantic_links: list[SemanticLinkHypothesis] = Field(default_factory=list)
    link_mappings: list[LinkGeometryMapping] = Field(default_factory=list)
    joints: list[JointSweepInput] = Field(default_factory=list)
    adjacent_link_exclusions: list[AdjacentLinkExclusion] = Field(default_factory=list)
    contact_probes: list[ContactRichProbeInput] = Field(default_factory=list)
    max_candidate_pairs: int = Field(default=50_000, ge=1, le=10_000_000)

    @model_validator(mode="after")
    def _validate_profile_fields(self) -> AdvancedProfileRequest:
        if self.profile == "articulated_rigid" and self.contact_probes:
            raise ValueError("articulated_rigid requests cannot contain contact_probes")
        if self.profile == "contact_rich" and any(
            (self.semantic_links, self.link_mappings, self.joints, self.adjacent_link_exclusions)
        ):
            raise ValueError(
                "contact_rich requests cannot contain articulated link or joint inputs"
            )
        return self


class AdvancedProfileEvidenceReport(StrictModel):
    """Single report envelope consumed by the parent repair certificate writer."""

    schema_version: Literal["geometry-repair.advanced-profile-report.v1"] = (
        ADVANCED_PROFILE_REPORT_SCHEMA_VERSION
    )
    request: AdvancedProfileRequest
    claim_scope: GeometryOnlyClaimScope
    status: EvidenceStatus
    disposition: Literal["evidence_complete", "conditional", "failed"]
    articulated: ArticulatedGeometryEvidence | None = None
    contact_rich: ContactRichGeometryEvidence | None = None
    blockers: list[str] = Field(default_factory=list)
    handoff: AdvancedProfileHandoff | None = None

    @model_validator(mode="before")
    @classmethod
    def _supply_legacy_claim_scope(cls, value: Any) -> Any:
        if not isinstance(value, dict) or "claim_scope" in value:
            return value
        request = value.get("request")
        profile = request.profile if isinstance(request, AdvancedProfileRequest) else None
        if isinstance(request, dict):
            profile = request.get("profile")
        if profile not in {"articulated_rigid", "contact_rich"}:
            return value
        populated = dict(value)
        populated["claim_scope"] = _geometry_claim_scope(cast(AdvancedProfileName, profile))
        return populated

    @model_validator(mode="after")
    def _validate_profile_payload(self) -> AdvancedProfileEvidenceReport:
        expected_scope = _geometry_claim_scope(self.request.profile)
        if self.claim_scope != expected_scope:
            raise ValueError(
                f"profile {self.request.profile!r} requires claim_scope {expected_scope!r}"
            )
        if self.request.profile == "articulated_rigid":
            if self.articulated is None or self.contact_rich is not None:
                raise ValueError("articulated_rigid reports require only articulated evidence")
        elif self.contact_rich is None or self.articulated is not None:
            raise ValueError("contact_rich reports require only contact_rich evidence")
        if self.handoff is not None:
            if self.handoff.profile != self.request.profile:
                raise ValueError("handoff profile must match the advanced-profile request")
            if self.handoff.claim_scope != expected_scope:
                raise ValueError("handoff claim_scope must match the report claim_scope")
            if self.handoff.geometry_status != self.status:
                raise ValueError("handoff geometry_status must match report status")
            if self.handoff.geometry_disposition != self.disposition:
                raise ValueError("handoff geometry_disposition must match report disposition")
        return self


def _valid_collision_mesh(mesh: MeshData) -> tuple[GeometryCheckEvidence, MeshData | None]:
    import trimesh

    failures: list[str] = []
    warnings: list[str] = []
    vertices = np.asarray(mesh.world_vertices_m, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int64).reshape((-1, 3))
    if not len(vertices) or not len(triangles):
        failures.append("collision mesh has no vertices or triangles")
    if len(vertices) and not np.isfinite(vertices).all():
        failures.append("collision mesh contains non-finite vertices")
    valid_indices = (
        np.all((triangles >= 0) & (triangles < len(vertices)), axis=1)
        if len(triangles)
        else np.empty((0,), dtype=bool)
    )
    invalid_count = int(len(triangles) - np.count_nonzero(valid_indices))
    if invalid_count:
        failures.append(f"collision mesh contains {invalid_count} invalid triangles")
    usable = triangles[valid_indices]
    degenerate_count = 0
    if len(usable) and np.isfinite(vertices).all():
        coordinates = vertices[usable]
        area_twice = np.linalg.norm(
            np.cross(coordinates[:, 1] - coordinates[:, 0], coordinates[:, 2] - coordinates[:, 0]),
            axis=1,
        )
        diagonal = float(np.linalg.norm(np.ptp(vertices, axis=0)))
        valid_area = area_twice > max(diagonal * diagonal * 2e-16, 2e-24)
        degenerate_count = int(len(usable) - np.count_nonzero(valid_area))
        usable = usable[valid_area]
        if degenerate_count:
            failures.append(f"collision mesh contains {degenerate_count} degenerate triangles")
    watertight = False
    winding_consistent = False
    convex = False
    volume_m3: float | None = None
    if not failures and len(usable):
        welded_vertices, welded_faces = positional_topology_mesh(vertices, usable)
        surface = trimesh.Trimesh(vertices=welded_vertices, faces=welded_faces, process=False)
        watertight = bool(surface.is_watertight)
        winding_consistent = bool(surface.is_winding_consistent)
        convex = bool(surface.is_convex)
        volume_m3 = abs(float(surface.volume)) if watertight else None
        if not watertight:
            failures.append("collision mesh is not watertight")
        if not winding_consistent:
            failures.append("collision mesh winding is inconsistent")
        if not convex:
            failures.append("articulated per-link collision part is not convex")
        if volume_m3 is not None and volume_m3 <= max(float(surface.scale) ** 3 * 1e-15, 1e-24):
            failures.append("collision mesh has negligible enclosed volume")
    check = GeometryCheckEvidence(
        check_id=f"per_link_collision:{mesh.path}",
        status="fail" if failures else "pass",
        subjects=[mesh.path],
        metrics={
            "vertex_count": int(len(vertices)),
            "triangle_count": int(len(triangles)),
            "invalid_triangle_count": invalid_count,
            "degenerate_triangle_count": degenerate_count,
            "watertight": watertight,
            "winding_consistent": winding_consistent,
            "convex": convex,
            "volume_m3": volume_m3,
        },
        failures=failures,
        warnings=warnings,
    )
    return check, mesh if not failures else None


def _rotation_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    x, y, z = axis / float(np.linalg.norm(axis))
    skew = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)
    identity = np.eye(3, dtype=np.float64)
    return identity + math.sin(angle_rad) * skew + (1.0 - math.cos(angle_rad)) * (skew @ skew)


def _transform_vertices(vertices: np.ndarray, joint: JointSweepInput, value: float) -> np.ndarray:
    if joint.axis_world is None or joint.origin_world_m is None or joint.reference_value is None:
        raise ValueError("joint transform requires caller-supplied axis, origin, and reference")
    axis = np.asarray(joint.axis_world, dtype=np.float64)
    axis /= float(np.linalg.norm(axis))
    delta = float(value - joint.reference_value)
    if joint.joint_type == "prismatic":
        return vertices + axis * delta
    origin = np.asarray(joint.origin_world_m, dtype=np.float64)
    rotation = _rotation_matrix(axis, delta)
    return (vertices - origin) @ rotation.T + origin


def _mesh_pair_intersection(
    left_vertices: np.ndarray,
    left_triangles: np.ndarray,
    right_vertices: np.ndarray,
    right_triangles: np.ndarray,
    *,
    candidate_budget: int,
) -> tuple[EvidenceStatus, bool, int, str | None]:
    """Return exact surface intersection under a deterministic broad-phase budget."""

    if candidate_budget < 1:
        return "indeterminate", False, 0, "candidate triangle-pair budget is exhausted"
    left_coordinates = left_vertices[left_triangles]
    right_coordinates = right_vertices[right_triangles]
    if not len(left_coordinates) or not len(right_coordinates):
        return "not_evaluated", False, 0, "one collision mesh has no valid triangles"
    combined = np.vstack((left_vertices, right_vertices))
    finite = combined[np.isfinite(combined).all(axis=1)]
    if not len(finite):
        return "not_evaluated", False, 0, "collision meshes contain no finite vertices"
    tolerance = max(float(np.linalg.norm(np.ptp(finite, axis=0))) * 1e-10, 1e-12)
    left_min = left_coordinates.min(axis=1)
    left_max = left_coordinates.max(axis=1)
    right_min = right_coordinates.min(axis=1)
    right_max = right_coordinates.max(axis=1)
    tested = 0
    for left_index, triangle in enumerate(left_coordinates):
        candidates = np.flatnonzero(
            np.all(left_max[left_index] + tolerance >= right_min, axis=1)
            & np.all(right_max + tolerance >= left_min[left_index], axis=1)
        )
        for right_index in candidates:
            tested += 1
            if tested > candidate_budget:
                return (
                    "indeterminate",
                    False,
                    tested - 1,
                    f"candidate triangle-pair count exceeds limit {candidate_budget}",
                )
            if _triangles_intersect(triangle, right_coordinates[int(right_index)], tolerance):
                return "fail", True, tested, None
    return "pass", False, tested, None


def _swept_motion_check(
    joint: JointSweepInput,
    collision_by_link: dict[str, list[MeshData]],
    exclusions: dict[tuple[str, str], AdjacentLinkExclusion],
    *,
    max_candidate_pairs: int,
) -> SweptMotionEvidence:
    missing_inputs = [
        name
        for name, value in (
            ("axis_world", joint.axis_world),
            ("origin_world_m", joint.origin_world_m),
            ("lower_limit", joint.lower_limit),
            ("upper_limit", joint.upper_limit),
            ("reference_value", joint.reference_value),
            ("units", joint.units),
        )
        if value is None
    ]
    if missing_inputs:
        return SweptMotionEvidence(
            joint_id=joint.joint_id,
            status="not_evaluated",
            requested_sample_count=joint.sample_count,
            evaluated_sample_count=0,
            warnings=[
                "caller did not supply required joint geometry: " + ", ".join(missing_inputs)
            ],
        )
    missing = sorted(set(joint.moving_link_ids + [joint.parent_link_id]) - collision_by_link.keys())
    if missing:
        return SweptMotionEvidence(
            joint_id=joint.joint_id,
            status="not_evaluated",
            requested_sample_count=joint.sample_count,
            evaluated_sample_count=0,
            warnings=[f"collision geometry is unavailable for links: {', '.join(missing)}"],
        )
    values = np.linspace(
        float(joint.lower_limit),
        float(joint.upper_limit),
        joint.sample_count,
        dtype=np.float64,
    )
    moving = set(joint.moving_link_ids)
    stationary = sorted(set(collision_by_link) - moving)
    collision_events: list[dict[str, Any]] = []
    excluded_pairs: dict[tuple[str, str], dict[str, Any]] = {}
    failures: list[str] = []
    warnings: list[str] = []
    tested_total = 0
    evaluated_samples = 0
    for sample_index, raw_value in enumerate(values):
        value = float(raw_value)
        for moving_link in sorted(moving):
            for stationary_link in stationary:
                pair = tuple(sorted((moving_link, stationary_link)))
                if pair in exclusions:
                    excluded_pairs.setdefault(
                        pair,
                        {
                            "links": list(pair),
                            "reason": exclusions[pair].reason,
                            "evidence": [
                                item.model_dump(mode="json") for item in exclusions[pair].evidence
                            ],
                        },
                    )
                    continue
                for moving_mesh in sorted(
                    collision_by_link[moving_link], key=lambda item: item.path
                ):
                    transformed = _transform_vertices(moving_mesh.world_vertices_m, joint, value)
                    for stationary_mesh in sorted(
                        collision_by_link[stationary_link], key=lambda item: item.path
                    ):
                        remaining = max_candidate_pairs - tested_total
                        status, intersects, tested, reason = _mesh_pair_intersection(
                            transformed,
                            moving_mesh.triangles,
                            stationary_mesh.world_vertices_m,
                            stationary_mesh.triangles,
                            candidate_budget=remaining,
                        )
                        tested_total += tested
                        if status == "indeterminate":
                            warnings.append(reason or "swept-motion intersection was indeterminate")
                            return SweptMotionEvidence(
                                joint_id=joint.joint_id,
                                status="indeterminate",
                                requested_sample_count=joint.sample_count,
                                evaluated_sample_count=evaluated_samples,
                                sample_values=[float(item) for item in values],
                                candidate_triangle_pairs_tested=tested_total,
                                collision_events=collision_events,
                                excluded_link_pairs=list(excluded_pairs.values()),
                                failures=failures,
                                warnings=warnings,
                            )
                        if status == "not_evaluated":
                            warnings.append(reason or "swept-motion intersection was not evaluated")
                            return SweptMotionEvidence(
                                joint_id=joint.joint_id,
                                status="not_evaluated",
                                requested_sample_count=joint.sample_count,
                                evaluated_sample_count=evaluated_samples,
                                sample_values=[float(item) for item in values],
                                candidate_triangle_pairs_tested=tested_total,
                                collision_events=collision_events,
                                excluded_link_pairs=list(excluded_pairs.values()),
                                failures=failures,
                                warnings=warnings,
                            )
                        if intersects:
                            event = {
                                "sample_index": sample_index,
                                "joint_value": value,
                                "units": joint.units,
                                "moving_link": moving_link,
                                "stationary_link": stationary_link,
                                "moving_collision_path": moving_mesh.path,
                                "stationary_collision_path": stationary_mesh.path,
                            }
                            collision_events.append(event)
                            failures.append(
                                f"{joint.joint_id} intersects {stationary_link!r} at "
                                f"sample {sample_index} ({value:.9g} {joint.units})"
                            )
        evaluated_samples += 1
    return SweptMotionEvidence(
        joint_id=joint.joint_id,
        status="fail" if failures else "pass",
        requested_sample_count=joint.sample_count,
        evaluated_sample_count=evaluated_samples,
        sample_values=[float(item) for item in values],
        candidate_triangle_pairs_tested=tested_total,
        collision_events=collision_events,
        excluded_link_pairs=list(excluded_pairs.values()),
        failures=failures,
        warnings=[
            "sampled geometry does not prove actuation, dynamics, contact stability, or runtime safety"
        ],
    )


def evaluate_articulated_geometry(
    render_path: str | Path,
    collision_path: str | Path,
    *,
    semantic_links: list[SemanticLinkHypothesis],
    link_mappings: list[LinkGeometryMapping],
    joints: list[JointSweepInput],
    adjacent_link_exclusions: list[AdjacentLinkExclusion] | None = None,
    max_candidate_pairs: int = 50_000,
) -> ArticulatedGeometryEvidence:
    """Evaluate articulated geometry without inferring or authoring joints."""

    if max_candidate_pairs < 1:
        raise ValueError("max_candidate_pairs must be positive")
    exclusions = adjacent_link_exclusions or []
    configuration_checks: list[GeometryCheckEvidence] = []
    blockers: list[str] = []

    link_ids = [item.link_id for item in semantic_links]
    duplicate_link_ids = sorted({item for item in link_ids if link_ids.count(item) > 1})
    mapping_ids = [item.link_id for item in link_mappings]
    duplicate_mapping_ids = sorted({item for item in mapping_ids if mapping_ids.count(item) > 1})
    configuration_failures = []
    if duplicate_link_ids:
        configuration_failures.append(
            f"duplicate semantic link IDs: {', '.join(duplicate_link_ids)}"
        )
    if duplicate_mapping_ids:
        configuration_failures.append(
            f"duplicate link mapping IDs: {', '.join(duplicate_mapping_ids)}"
        )
    if set(link_ids) != set(mapping_ids):
        configuration_failures.append(
            "semantic link IDs and link geometry mapping IDs must match exactly"
        )
    known_links = set(link_ids)
    exclusion_keys: list[tuple[str, str]] = []
    for exclusion in exclusions:
        exclusion_keys.append(exclusion.key)
        unknown = set(exclusion.key) - known_links
        if unknown:
            configuration_failures.append(
                f"adjacent exclusion {exclusion.key!r} references unknown links {sorted(unknown)!r}"
            )
    if len(exclusion_keys) != len(set(exclusion_keys)):
        configuration_failures.append("adjacent-link exclusions must not contain duplicate pairs")
    for joint in joints:
        unknown = (
            set(joint.moving_link_ids + [joint.parent_link_id, joint.child_link_id]) - known_links
        )
        if unknown:
            configuration_failures.append(
                f"joint {joint.joint_id!r} references unknown links {sorted(unknown)!r}"
            )
    configuration_checks.append(
        GeometryCheckEvidence(
            check_id="articulated_configuration",
            status="fail" if configuration_failures else "pass",
            subjects=sorted(known_links),
            metrics={"link_count": len(known_links), "joint_count": len(joints)},
            failures=configuration_failures,
        )
    )

    render_meshes: dict[str, MeshData] = {}
    collision_meshes: dict[str, MeshData] = {}
    load_warnings: list[str] = []
    try:
        # Explicit mappings may identify default-purpose Gprims that serve as
        # both visible and collision geometry in imported articulated USD.
        render_meshes = {
            item.path: item for item in load_meshes(render_path, include_guide_purpose=True)[0]
        }
    except Exception as exc:
        load_warnings.append(f"render geometry load failed: {type(exc).__name__}: {exc}")
    try:
        collision_meshes = {
            item.path: item for item in load_meshes(collision_path, include_guide_purpose=True)[0]
        }
    except Exception as exc:
        load_warnings.append(f"collision geometry load failed: {type(exc).__name__}: {exc}")
    if load_warnings:
        configuration_checks.append(
            GeometryCheckEvidence(
                check_id="articulated_geometry_load",
                status="indeterminate",
                warnings=load_warnings,
            )
        )

    path_owners: dict[tuple[str, str], str] = {}
    collision_by_link: dict[str, list[MeshData]] = {}
    link_checks: list[GeometryCheckEvidence] = []
    for mapping in sorted(link_mappings, key=lambda item: item.link_id):
        failures: list[str] = []
        for role, paths in (
            ("render", mapping.render_paths),
            ("collision", mapping.collision_paths),
        ):
            for path in paths:
                owner = path_owners.setdefault((role, path), mapping.link_id)
                if owner != mapping.link_id:
                    failures.append(
                        f"{role} path {path!r} is mapped to both {owner!r} and {mapping.link_id!r}"
                    )
        missing_render = sorted(set(mapping.render_paths) - render_meshes.keys())
        missing_collision = sorted(set(mapping.collision_paths) - collision_meshes.keys())
        if missing_render:
            failures.append(f"render paths were not found: {missing_render!r}")
        if missing_collision:
            failures.append(f"collision paths were not found: {missing_collision!r}")
        valid_collision: list[MeshData] = []
        child_checks: list[GeometryCheckEvidence] = []
        for path in sorted(set(mapping.collision_paths) & collision_meshes.keys()):
            check, valid_mesh = _valid_collision_mesh(collision_meshes[path])
            child_checks.append(check)
            if valid_mesh is not None:
                valid_collision.append(valid_mesh)
            failures.extend(check.failures)
        link_checks.append(
            GeometryCheckEvidence(
                check_id=f"link_mapping_and_collision:{mapping.link_id}",
                status="fail" if failures else "pass",
                subjects=[mapping.link_id, *mapping.render_paths, *mapping.collision_paths],
                metrics={
                    "render_mesh_count": len(mapping.render_paths) - len(missing_render),
                    "collision_mesh_count": len(mapping.collision_paths) - len(missing_collision),
                    "collision_part_checks": [
                        check.model_dump(mode="json") for check in child_checks
                    ],
                },
                failures=failures,
            )
        )
        if not failures:
            collision_by_link[mapping.link_id] = valid_collision

    exclusion_map = {item.key: item for item in exclusions}
    sweep_checks = [
        _swept_motion_check(
            joint,
            collision_by_link,
            exclusion_map,
            max_candidate_pairs=max_candidate_pairs,
        )
        for joint in sorted(joints, key=lambda item: item.joint_id)
    ]
    sweep_geometry_checks = [
        GeometryCheckEvidence(
            check_id=f"sampled_swept_motion:{item.joint_id}",
            status=item.status,
            subjects=[item.joint_id],
            metrics={
                "requested_sample_count": item.requested_sample_count,
                "evaluated_sample_count": item.evaluated_sample_count,
                "candidate_triangle_pairs_tested": item.candidate_triangle_pairs_tested,
                "collision_event_count": len(item.collision_events),
            },
            failures=item.failures,
            warnings=item.warnings,
        )
        for item in sweep_checks
    ]
    if not joints:
        sweep_geometry_checks.append(
            GeometryCheckEvidence(
                check_id="sampled_swept_motion",
                status="not_evaluated",
                warnings=["no caller-supplied joint geometry was provided"],
            )
        )

    all_checks = [*configuration_checks, *link_checks, *sweep_geometry_checks]
    status = _combined_status(all_checks)
    authoritative = all(_authoritative(item.evidence) for item in semantic_links) and all(
        _authoritative(item.evidence) for item in link_mappings
    )
    authoritative = authoritative and all(_authoritative(item.evidence) for item in joints)
    authoritative = authoritative and all(_authoritative(item.evidence) for item in exclusions)
    if not authoritative:
        blockers.append(
            "semantic, mapping, joint, or exclusion claims rely only on inferred hypotheses"
        )
    if not joints:
        blockers.append("sampled swept motion requires caller-supplied joint geometry")
    blockers.extend(
        failure for check in all_checks for failure in check.failures if failure not in blockers
    )
    if status in {"not_evaluated", "indeterminate"}:
        blockers.extend(
            warning for check in all_checks for warning in check.warnings if warning not in blockers
        )
    return ArticulatedGeometryEvidence(
        status=status,
        semantic_links=sorted(semantic_links, key=lambda item: item.link_id),
        link_mappings=sorted(link_mappings, key=lambda item: item.link_id),
        adjacent_link_exclusions=sorted(exclusions, key=lambda item: item.key),
        configuration_checks=configuration_checks,
        link_collision_checks=link_checks,
        swept_motion_checks=sweep_checks,
        certification_eligible=status == "pass" and authoritative and bool(joints),
        blockers=blockers,
    )


def _sample_path(
    points: np.ndarray,
    *,
    step_m: float,
    limit: int,
) -> tuple[np.ndarray, bool]:
    samples: list[np.ndarray] = []
    truncated = False
    for start, end in zip(points[:-1], points[1:], strict=True):
        count = max(2, int(math.ceil(float(np.linalg.norm(end - start)) / step_m)) + 1)
        values = np.linspace(0.0, 1.0, num=count, dtype=np.float64)
        segment = start[None, :] * (1.0 - values[:, None]) + end[None, :] * values[:, None]
        if samples:
            segment = segment[1:]
        remaining = limit - sum(len(item) for item in samples)
        if len(segment) > remaining:
            samples.append(segment[: max(remaining, 0)])
            truncated = True
            break
        samples.append(segment)
    return np.concatenate(samples, axis=0), truncated


def _receiver_collision_selection(
    probe: ContactRichProbeInput,
    collision_meshes_by_path: dict[str, list[MeshData]],
    *,
    load_error: str | None,
) -> tuple[GeometryCheckEvidence, list[MeshData]]:
    requested = probe.receiver_collision_paths
    if not requested:
        return (
            GeometryCheckEvidence(
                check_id=f"receiver_collision_selection:{probe.probe_id}",
                status="not_evaluated",
                subjects=[probe.receiver_part_id],
                warnings=["caller did not supply receiver_collision_paths"],
            ),
            [],
        )
    if load_error is not None:
        return (
            GeometryCheckEvidence(
                check_id=f"receiver_collision_selection:{probe.probe_id}",
                status="indeterminate",
                subjects=[probe.receiver_part_id, *requested],
                warnings=[load_error],
            ),
            [],
        )

    failures: list[str] = []
    if any(not path.strip() for path in requested):
        failures.append("receiver_collision_paths contains an empty prim path")
    duplicates = sorted({path for path in requested if requested.count(path) > 1})
    if duplicates:
        failures.append(
            "receiver collision selection is ambiguous because paths are duplicated: "
            f"{duplicates!r}"
        )
    selected: list[MeshData] = []
    for path in sorted(set(requested)):
        matches = collision_meshes_by_path.get(path, [])
        if not matches:
            failures.append(f"receiver collision prim path was not found: {path!r}")
            continue
        if len(matches) != 1:
            failures.append(
                f"receiver collision prim path is ambiguous: {path!r} resolved to "
                f"{len(matches)} meshes"
            )
            continue
        mesh = matches[0]
        if mesh.role != "collision":
            failures.append(
                f"receiver prim path {path!r} is classified as {mesh.role!r}, not collision"
            )
            continue
        selected.append(mesh)
    return (
        GeometryCheckEvidence(
            check_id=f"receiver_collision_selection:{probe.probe_id}",
            status="fail" if failures else "pass",
            subjects=[probe.receiver_part_id, *requested],
            metrics={
                "requested_paths": list(requested),
                "selected_paths": [mesh.path for mesh in selected],
                "selected_mesh_count": len(selected),
            },
            failures=failures,
        ),
        [] if failures else selected,
    )


def _selected_trimesh_parts(meshes: list[MeshData]) -> list[Any]:
    import trimesh

    parts = []
    for mesh in sorted(meshes, key=lambda item: item.path):
        vertices = np.asarray(mesh.world_vertices_m, dtype=np.float64)
        triangles = np.asarray(mesh.triangles, dtype=np.int64).reshape((-1, 3))
        if not len(vertices) or not np.isfinite(vertices).all() or not len(triangles):
            continue
        valid = np.all((triangles >= 0) & (triangles < len(vertices)), axis=1)
        vertices, triangles = positional_topology_mesh(vertices, triangles[valid])
        if not len(triangles):
            continue
        coordinates = vertices[triangles]
        area_twice = np.linalg.norm(
            np.cross(
                coordinates[:, 1] - coordinates[:, 0],
                coordinates[:, 2] - coordinates[:, 0],
            ),
            axis=1,
        )
        diagonal = float(np.linalg.norm(np.ptp(vertices, axis=0)))
        valid_area = np.isfinite(area_twice) & (
            area_twice > max(diagonal * diagonal * 2e-16, 2e-24)
        )
        if np.any(valid_area):
            parts.append(
                trimesh.Trimesh(
                    vertices=vertices,
                    faces=triangles[valid_area],
                    process=False,
                )
            )
    return parts


def _closest_selected_surface(
    meshes: list[MeshData],
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import trimesh

    query = np.asarray(points, dtype=np.float64).reshape((-1, 3))
    if not len(query):
        empty = np.empty((0, 3), dtype=np.float64)
        return empty, np.empty((0,), dtype=np.float64), empty
    parts = _selected_trimesh_parts(meshes)
    if not parts:
        raise ValueError("selected receiver collision prims contain no valid triangles")
    surface = parts[0] if len(parts) == 1 else trimesh.util.concatenate(parts)
    nearest_chunks: list[np.ndarray] = []
    distance_chunks: list[np.ndarray] = []
    normal_chunks: list[np.ndarray] = []
    face_normals = np.asarray(surface.face_normals, dtype=np.float64)
    for start in range(0, len(query), 128):
        chunk = query[start : start + 128]
        try:
            nearest, distances, triangle_ids = trimesh.proximity.closest_point(surface, chunk)
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


def _points_inside_selected_union(
    meshes: list[MeshData],
    points: np.ndarray,
) -> tuple[np.ndarray | None, str | None]:
    query = np.asarray(points, dtype=np.float64).reshape((-1, 3))
    parts = _selected_trimesh_parts(meshes)
    if not parts:
        return None, "selected receiver collision prims contain no valid mesh parts"
    if any(not part.is_watertight for part in parts):
        return None, "receiver occupancy requires watertight selected collision prims"
    occupied = np.zeros(len(query), dtype=bool)
    try:
        for part in parts:
            occupied |= np.asarray(part.contains(query), dtype=bool)
    except (ModuleNotFoundError, ImportError) as exc:
        return None, f"receiver occupancy backend unavailable: {type(exc).__name__}: {exc}"
    return occupied, None


def _closest_radial_surface_distances(
    meshes: list[MeshData],
    points: np.ndarray,
    *,
    axis: np.ndarray,
    normal_tolerance_deg: float,
) -> np.ndarray:
    """Measure clearance to receiver faces whose normals are radial to an axis."""

    import trimesh

    parts = _selected_trimesh_parts(meshes)
    if not parts:
        raise ValueError("selected receiver collision prims contain no valid triangles")
    surface = parts[0] if len(parts) == 1 else trimesh.util.concatenate(parts)
    unit_axis = np.asarray(axis, dtype=np.float64)
    unit_axis /= float(np.linalg.norm(unit_axis))
    maximum_alignment = math.sin(math.radians(normal_tolerance_deg))
    normals = np.asarray(surface.face_normals, dtype=np.float64)
    radial_faces = np.abs(normals @ unit_axis) <= maximum_alignment
    if not np.any(radial_faces):
        raise ValueError(
            "receiver collision has no radial faces within the declared normal tolerance"
        )
    radial_surface = trimesh.Trimesh(
        vertices=np.asarray(surface.vertices, dtype=np.float64),
        faces=np.asarray(surface.faces, dtype=np.int64)[radial_faces],
        process=False,
    )
    distances: list[np.ndarray] = []
    query = np.asarray(points, dtype=np.float64).reshape((-1, 3))
    for start in range(0, len(query), 128):
        chunk = query[start : start + 128]
        try:
            _nearest, chunk_distances, _triangle_ids = trimesh.proximity.closest_point(
                radial_surface,
                chunk,
            )
        except (ModuleNotFoundError, ImportError):
            _nearest, chunk_distances, _triangle_ids = trimesh.proximity.closest_point_naive(
                radial_surface, chunk
            )
        distances.append(np.asarray(chunk_distances, dtype=np.float64))
    return np.concatenate(distances) if distances else np.empty((0,), dtype=np.float64)


def _evaluate_contact_probe(
    probe: ContactRichProbeInput,
    collision_meshes_by_path: dict[str, list[MeshData]],
    *,
    collision_load_error: str | None,
) -> ContactProbeEvidence:
    selection_check, receiver_meshes = _receiver_collision_selection(
        probe,
        collision_meshes_by_path,
        load_error=collision_load_error,
    )
    if selection_check.status != "pass":
        reason = "receiver collision selection did not pass; geometry probes were not run"
        follow_up = [
            GeometryCheckEvidence(
                check_id=f"{check_name}:{probe.probe_id}",
                status="not_evaluated",
                required=not (
                    check_name == "sdf_resolution" and probe.collision_representation != "sdf"
                ),
                subjects=[probe.moving_part_id, probe.receiver_part_id],
                warnings=[reason],
            )
            for check_name in (
                "contact_axis",
                "contact_approach",
                "contact_path",
                "contact_clearance",
                "seated_stop",
                "sdf_resolution",
            )
        ]
        checks = [selection_check, *follow_up]
        return ContactProbeEvidence(
            probe_id=probe.probe_id,
            status=_combined_status(checks),
            checks=checks,
            sampled_path_points=0,
        )

    missing_motion_inputs = [
        name
        for name, value in (
            ("axis_origin_world_m", probe.axis_origin_world_m),
            ("axis_world", probe.axis_world),
            ("path_points_world_m", probe.path_points_world_m),
        )
        if value is None
    ]
    if missing_motion_inputs:
        reason = "caller did not supply required contact geometry: " + ", ".join(
            missing_motion_inputs
        )
        checks = [selection_check] + [
            GeometryCheckEvidence(
                check_id=f"{check_name}:{probe.probe_id}",
                status="not_evaluated",
                required=not (
                    check_name == "sdf_resolution" and probe.collision_representation != "sdf"
                ),
                subjects=[probe.moving_part_id, probe.receiver_part_id],
                warnings=[reason],
            )
            for check_name in (
                "contact_axis",
                "contact_approach",
                "contact_path",
                "contact_clearance",
                "seated_stop",
                "sdf_resolution",
            )
        ]
        return ContactProbeEvidence(
            probe_id=probe.probe_id,
            status="not_evaluated",
            checks=checks,
            sampled_path_points=0,
        )

    points = np.asarray(probe.path_points_world_m, dtype=np.float64)
    axis = np.asarray(probe.axis_world, dtype=np.float64)
    axis /= float(np.linalg.norm(axis))
    expected_axis = axis if probe.approach_direction == "along_axis" else -axis
    origin = np.asarray(probe.axis_origin_world_m, dtype=np.float64)
    relative = points - origin
    radial = relative - np.outer(relative @ axis, axis)
    maximum_axis_offset = float(np.max(np.linalg.norm(radial, axis=1)))
    segments = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(segments, axis=1)
    nonzero = segment_lengths > 1e-12
    normalized_segments = segments[nonzero] / segment_lengths[nonzero, None]
    minimum_alignment = (
        float(np.min(normalized_segments @ expected_axis)) if len(normalized_segments) else -1.0
    )
    cosine_limit = math.cos(math.radians(probe.angular_tolerance_deg))
    axis_failures: list[str] = []
    if maximum_axis_offset > probe.axis_tolerance_m:
        axis_failures.append(
            f"path offset from declared axis is {maximum_axis_offset:.6g} m, above "
            f"{probe.axis_tolerance_m:.6g} m"
        )
    if minimum_alignment < cosine_limit:
        axis_failures.append(
            f"path/approach alignment {minimum_alignment:.6g} is below {cosine_limit:.6g}"
        )
    axis_check = GeometryCheckEvidence(
        check_id=f"contact_axis:{probe.probe_id}",
        status="fail" if axis_failures else "pass",
        subjects=[probe.moving_part_id, probe.receiver_part_id],
        metrics={
            "maximum_axis_offset_m": maximum_axis_offset,
            "minimum_direction_alignment": minimum_alignment,
            "angular_tolerance_deg": probe.angular_tolerance_deg,
        },
        failures=axis_failures,
    )

    projections = points @ expected_axis
    projection_steps = np.diff(projections)
    approach_failures = []
    if not len(projection_steps) or np.any(projection_steps <= 1e-12):
        approach_failures.append(
            "approach path is not strictly monotonic in the declared direction"
        )
    approach_check = GeometryCheckEvidence(
        check_id=f"contact_approach:{probe.probe_id}",
        status="fail" if approach_failures else "pass",
        subjects=[probe.moving_part_id, probe.receiver_part_id],
        metrics={
            "signed_approach_distance_m": float(projections[-1] - projections[0]),
            "segment_count": int(len(segments)),
        },
        failures=approach_failures,
    )

    samples, truncated = _sample_path(
        points,
        step_m=probe.sample_step_m,
        limit=probe.max_path_samples,
    )
    path_check: GeometryCheckEvidence
    clearance_check: GeometryCheckEvidence
    try:
        distances = _closest_radial_surface_distances(
            receiver_meshes,
            samples,
            axis=axis,
            normal_tolerance_deg=probe.radial_surface_normal_tolerance_deg,
        )
        occupied, occupancy_error = _points_inside_selected_union(receiver_meshes, samples)
        minimum_surface_distance = float(np.min(distances)) if len(distances) else None
        minimum_allowed_clearance = (
            probe.required_radial_clearance_m - probe.maximum_clearance_erosion_m
        )
        required_distance = probe.moving_envelope_radius_m + minimum_allowed_clearance
        realized_clearance = (
            minimum_surface_distance - probe.moving_envelope_radius_m
            if minimum_surface_distance is not None
            else None
        )
        clearance_failures = []
        if minimum_surface_distance is None:
            clearance_failures.append("clearance query returned no samples")
        elif minimum_surface_distance + 1e-12 < required_distance:
            clearance_failures.append(
                f"minimum path distance {minimum_surface_distance:.6g} m is below moving-envelope "
                f"plus clearance requirement {required_distance:.6g} m"
            )
        clearance_check = GeometryCheckEvidence(
            check_id=f"contact_clearance:{probe.probe_id}",
            status="fail" if clearance_failures else "pass",
            subjects=[probe.moving_part_id, probe.receiver_part_id],
            metrics={
                "minimum_surface_distance_m": minimum_surface_distance,
                "moving_envelope_radius_m": probe.moving_envelope_radius_m,
                "required_radial_clearance_m": probe.required_radial_clearance_m,
                "maximum_clearance_erosion_m": probe.maximum_clearance_erosion_m,
                "minimum_allowed_clearance_m": minimum_allowed_clearance,
                "realized_radial_clearance_m": realized_clearance,
                "radial_surface_normal_tolerance_deg": (probe.radial_surface_normal_tolerance_deg),
            },
            failures=clearance_failures,
        )
        if occupied is None:
            path_check = GeometryCheckEvidence(
                check_id=f"contact_path:{probe.probe_id}",
                status="indeterminate",
                subjects=[probe.moving_part_id, probe.receiver_part_id],
                metrics={"sample_count": int(len(samples)), "truncated": truncated},
                warnings=[occupancy_error or "receiver occupancy was unavailable"],
            )
        else:
            occupied_count = int(np.count_nonzero(occupied))
            path_failures = (
                [f"{occupied_count} approach-path samples are inside receiver solid geometry"]
                if occupied_count
                else []
            )
            path_warnings = (
                ["path sampling reached max_path_samples before covering the complete path"]
                if truncated
                else []
            )
            path_status: EvidenceStatus = "fail" if path_failures else "pass"
            if truncated and not path_failures:
                path_status = "indeterminate"
            path_check = GeometryCheckEvidence(
                check_id=f"contact_path:{probe.probe_id}",
                status=path_status,
                subjects=[probe.moving_part_id, probe.receiver_part_id],
                metrics={
                    "sample_count": int(len(samples)),
                    "occupied_sample_count": occupied_count,
                    "truncated": truncated,
                },
                failures=path_failures,
                warnings=path_warnings,
            )
    except Exception as exc:
        reason = f"contact path query failed: {type(exc).__name__}: {exc}"
        path_check = GeometryCheckEvidence(
            check_id=f"contact_path:{probe.probe_id}",
            status="indeterminate",
            subjects=[probe.moving_part_id, probe.receiver_part_id],
            warnings=[reason],
        )
        clearance_check = GeometryCheckEvidence(
            check_id=f"contact_clearance:{probe.probe_id}",
            status="indeterminate",
            subjects=[probe.moving_part_id, probe.receiver_part_id],
            warnings=[reason],
        )

    if probe.seated_stop_point_world_m is None:
        stop_check = GeometryCheckEvidence(
            check_id=f"seated_stop:{probe.probe_id}",
            status="not_evaluated",
            subjects=[probe.receiver_part_id],
            warnings=["caller did not supply seated_stop_point_world_m"],
        )
    else:
        try:
            _nearest, stop_distances, _normals = _closest_selected_surface(
                receiver_meshes,
                np.asarray([probe.seated_stop_point_world_m], dtype=np.float64),
            )
            stop_distance = float(stop_distances[0]) if len(stop_distances) else None
            stop_failures = []
            if stop_distance is None:
                stop_failures.append("seated-stop query returned no surface distance")
            elif stop_distance > probe.seated_stop_tolerance_m:
                stop_failures.append(
                    f"seated-stop support is {stop_distance:.6g} m away, above tolerance "
                    f"{probe.seated_stop_tolerance_m:.6g} m"
                )
            stop_check = GeometryCheckEvidence(
                check_id=f"seated_stop:{probe.probe_id}",
                status="fail" if stop_failures else "pass",
                subjects=[probe.receiver_part_id],
                metrics={"surface_distance_m": stop_distance},
                failures=stop_failures,
            )
        except Exception as exc:
            stop_check = GeometryCheckEvidence(
                check_id=f"seated_stop:{probe.probe_id}",
                status="indeterminate",
                subjects=[probe.receiver_part_id],
                warnings=[f"seated-stop query failed: {type(exc).__name__}: {exc}"],
            )

    maximum_voxel_size = probe.minimum_protected_feature_m / probe.required_voxels_across_feature
    if probe.collision_representation != "sdf":
        sdf_check = GeometryCheckEvidence(
            check_id=f"sdf_resolution:{probe.probe_id}",
            status="not_evaluated",
            required=False,
            metrics={
                "collision_representation": probe.collision_representation,
                "maximum_voxel_size_m": maximum_voxel_size,
            },
            warnings=["SDF resolution is not applicable to the declared collision representation"],
        )
    elif probe.sdf_voxel_size_m is None:
        sdf_check = GeometryCheckEvidence(
            check_id=f"sdf_resolution:{probe.probe_id}",
            status="not_evaluated",
            metrics={"maximum_voxel_size_m": maximum_voxel_size},
            warnings=["SDF representation was declared without a voxel size"],
        )
    else:
        sdf_failures = []
        if probe.sdf_voxel_size_m > maximum_voxel_size * (1.0 + 1e-12):
            sdf_failures.append(
                f"SDF voxel size {probe.sdf_voxel_size_m:.6g} m exceeds protected-feature "
                f"limit {maximum_voxel_size:.6g} m"
            )
        sdf_check = GeometryCheckEvidence(
            check_id=f"sdf_resolution:{probe.probe_id}",
            status="fail" if sdf_failures else "pass",
            subjects=[probe.moving_part_id, probe.receiver_part_id],
            metrics={
                "voxel_size_m": probe.sdf_voxel_size_m,
                "minimum_protected_feature_m": probe.minimum_protected_feature_m,
                "required_voxels_across_feature": probe.required_voxels_across_feature,
                "maximum_voxel_size_m": maximum_voxel_size,
            },
            failures=sdf_failures,
        )
    checks = [
        selection_check,
        axis_check,
        approach_check,
        path_check,
        clearance_check,
        stop_check,
        sdf_check,
    ]
    return ContactProbeEvidence(
        probe_id=probe.probe_id,
        status=_combined_status(checks),
        checks=checks,
        sampled_path_points=int(len(samples)),
    )


def evaluate_contact_rich_geometry(
    receiver_collision_path: str | Path,
    *,
    probes: list[ContactRichProbeInput],
) -> ContactRichGeometryEvidence:
    """Evaluate explicit approach/clearance/stop probes against receiver collision."""

    path = Path(receiver_collision_path).expanduser().resolve()
    probe_ids = [probe.probe_id for probe in probes]
    blockers: list[str] = []
    if len(probe_ids) != len(set(probe_ids)):
        blockers.append("contact-rich probe IDs must be unique")
    collision_meshes_by_path: dict[str, list[MeshData]] = {}
    collision_load_error: str | None = None
    try:
        collision_meshes, _metadata = load_meshes(path, include_guide_purpose=True)
        for mesh in collision_meshes:
            collision_meshes_by_path.setdefault(mesh.path, []).append(mesh)
    except Exception as exc:
        collision_load_error = f"collision geometry load failed: {type(exc).__name__}: {exc}"
    evidence = [
        _evaluate_contact_probe(
            probe,
            collision_meshes_by_path,
            collision_load_error=collision_load_error,
        )
        for probe in sorted(probes, key=lambda item: item.probe_id)
    ]
    if not probes:
        status: EvidenceStatus = "not_evaluated"
        blockers.append("no contact-rich geometry probes were supplied")
    else:
        statuses = [item.status for item in evidence]
        status = "pass"
        for candidate in ("fail", "indeterminate", "not_evaluated"):
            if candidate in statuses:
                status = cast(EvidenceStatus, candidate)
                break
    authoritative = bool(probes) and all(_authoritative(probe.evidence) for probe in probes)
    if not authoritative and probes:
        blockers.append("contact intent relies only on inferred hypotheses")
    for item in evidence:
        for check in item.checks:
            blockers.extend(failure for failure in check.failures if failure not in blockers)
            if check.required and check.status in {"not_evaluated", "indeterminate"}:
                blockers.extend(warning for warning in check.warnings if warning not in blockers)
    return ContactRichGeometryEvidence(
        status=status,
        receiver_collision_path=str(path),
        probes=sorted(probes, key=lambda item: item.probe_id),
        probe_evidence=evidence,
        certification_eligible=status == "pass" and authoritative and not blockers,
        blockers=blockers,
    )


def _stable_subject_refs(kind: Literal["link", "probe"], identifiers: list[str]) -> list[str]:
    totals = {identifier: identifiers.count(identifier) for identifier in set(identifiers)}
    occurrences: dict[str, int] = {}
    references = []
    for identifier in identifiers:
        occurrences[identifier] = occurrences.get(identifier, 0) + 1
        suffix = f":{occurrences[identifier]}" if totals[identifier] > 1 else ""
        references.append(f"{kind}:{identifier}{suffix}")
    return references


def _artifact_reference(
    *,
    reference_id: str,
    role: AdvancedProfileArtifactRole,
    path: str | Path | None,
    prim_paths: list[str],
    unavailable_reason: str | None,
    cache: _ArtifactReferenceCache,
) -> AdvancedProfileArtifactReference:
    resolved_path = str(Path(path).expanduser().resolve()) if path is not None else None
    cache_key = (role, resolved_path, unavailable_reason)
    cached = cache.get(cache_key)
    if cached is None:
        digest: str | None = None
        warnings: list[str] = []
        availability: ArtifactAvailability = "not_evaluated"
        if unavailable_reason is not None:
            warnings.append(unavailable_reason)
        elif resolved_path is None:
            warnings.append(f"{role} artifact path was not supplied")
        elif not Path(resolved_path).is_file():
            warnings.append(f"{role} artifact is unavailable at {resolved_path!r}")
        else:
            try:
                digest = file_sha256(Path(resolved_path))
                availability = "available"
            except OSError as exc:
                warnings.append(
                    f"{role} artifact digest was not evaluated: {type(exc).__name__}: {exc}"
                )
        cached = (resolved_path, digest, availability, warnings)
        cache[cache_key] = cached
    artifact_path, digest, availability, warnings = cached
    return AdvancedProfileArtifactReference(
        reference_id=reference_id,
        role=role,
        artifact_path=artifact_path,
        sha256=digest,
        prim_paths=sorted(set(prim_paths)),
        availability=availability,
        warnings=list(warnings),
    )


def _link_artifact_references(
    request: AdvancedProfileRequest,
    evidence: ArticulatedGeometryEvidence,
    *,
    render_path: str | Path | None,
    collision_path: str | Path | None,
    collision_unavailable_reason: str | None,
) -> list[LinkArtifactReferences]:
    link_ids = sorted(
        {item.link_id for item in request.semantic_links}
        | {item.link_id for item in request.link_mappings}
    )
    cache: _ArtifactReferenceCache = {}
    records = []
    for link_id in link_ids:
        subject_ref = f"link:{link_id}"
        mappings = [item for item in request.link_mappings if item.link_id == link_id]
        semantic_evidence = [
            source
            for link in request.semantic_links
            if link.link_id == link_id
            for source in link.evidence
        ]
        joint_evidence = [
            source
            for joint in request.joints
            if link_id in set(joint.moving_link_ids + [joint.parent_link_id, joint.child_link_id])
            for source in joint.evidence
        ]
        exclusion_evidence = [
            source
            for exclusion in request.adjacent_link_exclusions
            if link_id in exclusion.key
            for source in exclusion.evidence
        ]
        source_refs = sorted(
            {
                item.source_ref
                for item in [
                    *semantic_evidence,
                    *(source for mapping in mappings for source in mapping.evidence),
                    *joint_evidence,
                    *exclusion_evidence,
                ]
            }
        )
        check_ids = [check.check_id for check in evidence.configuration_checks]
        mapping_checks = [
            check
            for check in evidence.link_collision_checks
            if check.check_id == f"link_mapping_and_collision:{link_id}"
        ]
        mapping_status: EvidenceStatus = "not_evaluated"
        if mapping_checks:
            check_ids.extend(check.check_id for check in mapping_checks)
            statuses = [check.status for check in mapping_checks]
            mapping_status = "pass"
            for candidate in ("fail", "indeterminate", "not_evaluated"):
                if candidate in statuses:
                    mapping_status = cast(EvidenceStatus, candidate)
                    break
        check_ids.extend(
            f"sampled_swept_motion:{joint.joint_id}"
            for joint in sorted(request.joints, key=lambda item: item.joint_id)
            if link_id in set(joint.moving_link_ids + [joint.parent_link_id, joint.child_link_id])
        )
        render_paths = sorted({path for mapping in mappings for path in mapping.render_paths})
        collision_paths = sorted({path for mapping in mappings for path in mapping.collision_paths})
        render_unavailable_reason = None
        if not render_paths:
            render_unavailable_reason = f"link {link_id!r} has no declared render prim mapping"
        elif render_path is None:
            render_unavailable_reason = (
                "accepted render artifact path was not supplied to the handoff"
            )
        link_collision_unavailable_reason = collision_unavailable_reason
        if link_collision_unavailable_reason is None and not collision_paths:
            link_collision_unavailable_reason = (
                f"link {link_id!r} has no declared collision prim mapping"
            )
        records.append(
            LinkArtifactReferences(
                subject_ref=subject_ref,
                link_id=link_id,
                render_artifact=_artifact_reference(
                    reference_id=f"{subject_ref}:render",
                    role="accepted_render_usd",
                    path=render_path,
                    prim_paths=render_paths,
                    unavailable_reason=render_unavailable_reason,
                    cache=cache,
                ),
                collision_artifact=_artifact_reference(
                    reference_id=f"{subject_ref}:collision",
                    role="accepted_collision_usd",
                    path=collision_path,
                    prim_paths=collision_paths,
                    unavailable_reason=link_collision_unavailable_reason,
                    cache=cache,
                ),
                mapping_status=mapping_status,
                geometry_check_ids=list(dict.fromkeys(check_ids)),
                source_evidence_refs=source_refs,
            )
        )
    return records


def _probe_artifact_references(
    request: AdvancedProfileRequest,
    evidence: ContactRichGeometryEvidence,
    *,
    collision_path: str | Path | None,
    collision_unavailable_reason: str | None,
) -> list[ProbeArtifactReferences]:
    probes = sorted(request.contact_probes, key=lambda item: item.probe_id)
    subject_refs = _stable_subject_refs("probe", [item.probe_id for item in probes])
    cache: _ArtifactReferenceCache = {}
    records = []
    for index, (subject_ref, probe) in enumerate(zip(subject_refs, probes, strict=True)):
        probe_evidence = (
            evidence.probe_evidence[index] if index < len(evidence.probe_evidence) else None
        )
        checks = probe_evidence.checks if probe_evidence is not None else []
        selection_status: EvidenceStatus = checks[0].status if checks else "not_evaluated"
        receiver_paths = probe.receiver_collision_paths or []
        probe_collision_unavailable_reason = collision_unavailable_reason
        if probe_collision_unavailable_reason is None and not receiver_paths:
            probe_collision_unavailable_reason = (
                f"probe {probe.probe_id!r} has no declared receiver_collision_paths"
            )
        records.append(
            ProbeArtifactReferences(
                subject_ref=subject_ref,
                probe_id=probe.probe_id,
                moving_part_id=probe.moving_part_id,
                receiver_part_id=probe.receiver_part_id,
                receiver_collision_artifact=_artifact_reference(
                    reference_id=f"{subject_ref}:receiver_collision",
                    role="accepted_collision_usd",
                    path=collision_path,
                    prim_paths=receiver_paths,
                    unavailable_reason=probe_collision_unavailable_reason,
                    cache=cache,
                ),
                receiver_selection_status=selection_status,
                geometry_check_ids=[check.check_id for check in checks]
                or [f"receiver_collision_selection:{probe.probe_id}"],
                source_evidence_refs=sorted({item.source_ref for item in probe.evidence}),
            )
        )
    return records


def _downstream_routes(
    *,
    profile: AdvancedProfileName,
    claim_scope: GeometryOnlyClaimScope,
    geometry_status: EvidenceStatus,
    geometry_disposition: Literal["evidence_complete", "conditional", "failed"],
    link_artifacts: list[LinkArtifactReferences],
    probe_artifacts: list[ProbeArtifactReferences],
    geometry_blockers: list[str],
) -> list[DownstreamRoutingRecord]:
    subject_refs = [
        *(item.subject_ref for item in link_artifacts),
        *(item.subject_ref for item in probe_artifacts),
    ]
    artifact_references = [
        *(
            reference
            for item in link_artifacts
            for reference in (item.render_artifact, item.collision_artifact)
        ),
        *(item.receiver_collision_artifact for item in probe_artifacts),
    ]
    artifact_reference_ids = [item.reference_id for item in artifact_references]
    geometry_check_ids = list(
        dict.fromkeys(
            check_id
            for item in [*link_artifacts, *probe_artifacts]
            for check_id in item.geometry_check_ids
        )
    )
    artifact_blockers = sorted(
        {
            warning
            for artifact in artifact_references
            if artifact.availability == "not_evaluated"
            for warning in artifact.warnings
        }
    )
    profile_requirements = {
        "articulated_rigid": {
            "articulation": [
                "authoritative source or caller joint definitions and collision-filter intent"
            ],
            "physics": ["authoritative physical-property and contact-material inputs per link"],
        },
        "contact_rich": {
            "articulation": [
                "authoritative constrained-degree-of-freedom intent when constraints are required"
            ],
            "physics": ["authoritative physical-property, contact-material, and solver inputs"],
        },
    }[profile]
    required_inputs: dict[DownstreamOwner, list[str]] = {
        "articulation": [
            "geometry-only advanced-profile evidence and referenced geometry artifacts",
            *profile_requirements["articulation"],
        ],
        "physics": [
            "geometry-only advanced-profile evidence and referenced collision artifacts",
            *profile_requirements["physics"],
        ],
        "runtime_validation": [
            "target runtime and scenario definition",
            "authored articulation and physics artifacts from upstream owners",
        ],
        "simready": [
            "formal SimReady profile identifier and version",
            "authored asset plus completed articulation, physics, and runtime evidence",
        ],
    }
    requested_work: dict[DownstreamOwner, list[str]] = {
        "articulation": [
            (
                "author and validate joints, limits, drives, and collision filtering only from "
                "authoritative inputs"
                if profile == "articulated_rigid"
                else "author and validate constrained degrees of freedom only when authoritative "
                "task intent requires them"
            )
        ],
        "physics": [
            "author and validate mass, inertia, friction, restitution, contact materials, and "
            "solver properties from authoritative inputs"
        ],
        "runtime_validation": [
            "execute target-runtime actuation, contact, limit, seating, and stability checks "
            "applicable to the profile"
        ],
        "simready": [
            "run formal SimReady profile validation after required authored and runtime evidence "
            "is available"
        ],
    }
    dependencies: dict[DownstreamOwner, list[DownstreamOwner]] = {
        "articulation": [],
        "physics": [],
        "runtime_validation": ["articulation", "physics"],
        "simready": ["articulation", "physics", "runtime_validation"],
    }
    workflow_by_owner: dict[DownstreamOwner, DownstreamWorkflow] = {
        "articulation": "content-workflow-articulation",
        "physics": "content-workflow-physics",
        "runtime_validation": "content-workflow-runtime-validation",
        "simready": "content-workflow-simready",
    }
    routes = []
    for owner in _DOWNSTREAM_OWNER_ORDER:
        route_blockers = sorted(set(geometry_blockers + artifact_blockers))
        if geometry_disposition == "failed":
            readiness: RouteReadiness = "blocked"
            if not route_blockers:
                route_blockers.append("geometry evidence failed")
        elif (
            geometry_status != "pass"
            or geometry_disposition != "evidence_complete"
            or artifact_blockers
        ):
            readiness = "conditional"
            if not route_blockers:
                route_blockers.append("geometry evidence is incomplete")
        elif dependencies[owner]:
            readiness = "conditional"
            route_blockers.extend(
                f"{dependency} result is not_evaluated" for dependency in dependencies[owner]
            )
        else:
            readiness = "ready"
            route_blockers = []
        routes.append(
            DownstreamRoutingRecord(
                route_id=f"{profile}:{owner}",
                owner=owner,
                workflow=workflow_by_owner[owner],
                readiness=readiness,
                upstream_claim_scope=claim_scope,
                artifact_reference_ids=artifact_reference_ids,
                subject_refs=subject_refs,
                geometry_check_ids=geometry_check_ids,
                depends_on=dependencies[owner],
                required_inputs=required_inputs[owner],
                requested_work=requested_work[owner],
                blockers=list(dict.fromkeys(route_blockers)),
            )
        )
    return routes


def _advanced_profile_handoff(
    request: AdvancedProfileRequest,
    *,
    status: EvidenceStatus,
    disposition: Literal["evidence_complete", "conditional", "failed"],
    blockers: list[str],
    articulated: ArticulatedGeometryEvidence | None = None,
    contact_rich: ContactRichGeometryEvidence | None = None,
    render_path: str | Path | None = None,
    collision_path: str | Path | None = None,
    collision_unavailable_reason: str | None = None,
) -> AdvancedProfileHandoff:
    claim_scope = _geometry_claim_scope(request.profile)
    link_artifacts = (
        _link_artifact_references(
            request,
            articulated,
            render_path=render_path,
            collision_path=collision_path,
            collision_unavailable_reason=collision_unavailable_reason,
        )
        if articulated is not None
        else []
    )
    probe_artifacts = (
        _probe_artifact_references(
            request,
            contact_rich,
            collision_path=collision_path,
            collision_unavailable_reason=collision_unavailable_reason,
        )
        if contact_rich is not None
        else []
    )
    routes = _downstream_routes(
        profile=request.profile,
        claim_scope=claim_scope,
        geometry_status=status,
        geometry_disposition=disposition,
        link_artifacts=link_artifacts,
        probe_artifacts=probe_artifacts,
        geometry_blockers=blockers,
    )
    return AdvancedProfileHandoff(
        profile=request.profile,
        claim_scope=claim_scope,
        geometry_status=status,
        geometry_disposition=disposition,
        downstream_disposition="blocked" if disposition == "failed" else "conditional",
        link_artifacts=link_artifacts,
        probe_artifacts=probe_artifacts,
        downstream_routes=routes,
        blockers=sorted(set(blockers)),
    )


def evaluate_advanced_profile(
    render_path: str | Path,
    collision_path: str | Path,
    request: AdvancedProfileRequest,
) -> AdvancedProfileEvidenceReport:
    """Evaluate one embeddable request and return a single serializable report."""

    if request.profile == "articulated_rigid":
        articulated = evaluate_articulated_geometry(
            render_path,
            collision_path,
            semantic_links=request.semantic_links,
            link_mappings=request.link_mappings,
            joints=request.joints,
            adjacent_link_exclusions=request.adjacent_link_exclusions,
            max_candidate_pairs=request.max_candidate_pairs,
        )
        disposition: Literal["evidence_complete", "conditional", "failed"]
        if articulated.status == "fail":
            disposition = "failed"
        elif articulated.certification_eligible:
            disposition = "evidence_complete"
        else:
            disposition = "conditional"
        return AdvancedProfileEvidenceReport(
            request=request,
            claim_scope=_geometry_claim_scope(request.profile),
            status=articulated.status,
            disposition=disposition,
            articulated=articulated,
            blockers=articulated.blockers,
            handoff=_advanced_profile_handoff(
                request,
                status=articulated.status,
                disposition=disposition,
                blockers=articulated.blockers,
                articulated=articulated,
                render_path=render_path,
                collision_path=collision_path,
            ),
        )

    contact_rich = evaluate_contact_rich_geometry(
        collision_path,
        probes=request.contact_probes,
    )
    if contact_rich.status == "fail":
        disposition = "failed"
    elif contact_rich.certification_eligible:
        disposition = "evidence_complete"
    else:
        disposition = "conditional"
    return AdvancedProfileEvidenceReport(
        request=request,
        claim_scope=_geometry_claim_scope(request.profile),
        status=contact_rich.status,
        disposition=disposition,
        contact_rich=contact_rich,
        blockers=contact_rich.blockers,
        handoff=_advanced_profile_handoff(
            request,
            status=contact_rich.status,
            disposition=disposition,
            blockers=contact_rich.blockers,
            contact_rich=contact_rich,
            render_path=render_path,
            collision_path=collision_path,
        ),
    )


def collision_unavailable_advanced_profile_report(
    request: AdvancedProfileRequest,
    *,
    reason: str,
    collision_path: str | Path | None = None,
) -> AdvancedProfileEvidenceReport:
    """Build conditional Phase 3 evidence without substituting render geometry."""

    message = reason.strip()
    if not message:
        raise ValueError("collision-unavailable evidence requires a non-empty reason")
    if request.profile == "articulated_rigid":
        articulated = ArticulatedGeometryEvidence(
            status="not_evaluated",
            semantic_links=sorted(request.semantic_links, key=lambda item: item.link_id),
            link_mappings=sorted(request.link_mappings, key=lambda item: item.link_id),
            adjacent_link_exclusions=sorted(
                request.adjacent_link_exclusions,
                key=lambda item: item.key,
            ),
            configuration_checks=[
                GeometryCheckEvidence(
                    check_id="accepted_collision_geometry",
                    status="not_evaluated",
                    warnings=[message],
                )
            ],
            link_collision_checks=[],
            swept_motion_checks=[
                SweptMotionEvidence(
                    joint_id=joint.joint_id,
                    status="not_evaluated",
                    requested_sample_count=joint.sample_count,
                    evaluated_sample_count=0,
                    warnings=[message],
                )
                for joint in sorted(request.joints, key=lambda item: item.joint_id)
            ],
            certification_eligible=False,
            blockers=[message],
        )
        return AdvancedProfileEvidenceReport(
            request=request,
            claim_scope=_geometry_claim_scope(request.profile),
            status="not_evaluated",
            disposition="conditional",
            articulated=articulated,
            blockers=[message],
            handoff=_advanced_profile_handoff(
                request,
                status="not_evaluated",
                disposition="conditional",
                blockers=[message],
                articulated=articulated,
                collision_path=collision_path,
                collision_unavailable_reason=message,
            ),
        )

    probe_evidence = []
    for probe in sorted(request.contact_probes, key=lambda item: item.probe_id):
        checks = [
            GeometryCheckEvidence(
                check_id=f"{check_name}:{probe.probe_id}",
                status="not_evaluated",
                required=not (
                    check_name == "sdf_resolution" and probe.collision_representation != "sdf"
                ),
                subjects=[probe.moving_part_id, probe.receiver_part_id],
                warnings=[message],
            )
            for check_name in (
                "receiver_collision_selection",
                "contact_axis",
                "contact_approach",
                "contact_path",
                "contact_clearance",
                "seated_stop",
                "sdf_resolution",
            )
        ]
        probe_evidence.append(
            ContactProbeEvidence(
                probe_id=probe.probe_id,
                status="not_evaluated",
                checks=checks,
                sampled_path_points=0,
            )
        )
    resolved_collision_path = (
        str(Path(collision_path).expanduser().resolve()) if collision_path is not None else None
    )
    contact_rich = ContactRichGeometryEvidence(
        status="not_evaluated",
        receiver_collision_path=resolved_collision_path,
        probes=sorted(request.contact_probes, key=lambda item: item.probe_id),
        probe_evidence=probe_evidence,
        certification_eligible=False,
        blockers=[message],
    )
    return AdvancedProfileEvidenceReport(
        request=request,
        claim_scope=_geometry_claim_scope(request.profile),
        status="not_evaluated",
        disposition="conditional",
        contact_rich=contact_rich,
        blockers=[message],
        handoff=_advanced_profile_handoff(
            request,
            status="not_evaluated",
            disposition="conditional",
            blockers=[message],
            contact_rich=contact_rich,
            collision_path=collision_path,
            collision_unavailable_reason=message,
        ),
    )


def write_advanced_profile_report(
    report: AdvancedProfileEvidenceReport,
    output_path: str | Path,
) -> Path:
    """Atomically serialize the complete request and evidence as one JSON report."""

    return atomic_write_json(
        Path(output_path).expanduser().resolve(),
        report.model_dump(mode="json"),
    )


def write_collision_unavailable_advanced_profile_report(
    request: AdvancedProfileRequest,
    output_path: str | Path,
    *,
    reason: str,
    collision_path: str | Path | None = None,
) -> Path:
    """Serialize conditional evidence when accepted collision is unavailable."""

    return write_advanced_profile_report(
        collision_unavailable_advanced_profile_report(
            request,
            reason=reason,
            collision_path=collision_path,
        ),
        output_path,
    )
