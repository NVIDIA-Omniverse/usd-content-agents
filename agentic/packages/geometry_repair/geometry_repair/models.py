# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Strict contracts for geometry repair jobs and certificates."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .worker_ids import canonical_worker_name, canonical_worker_names

REPAIR_REQUEST_SCHEMA_VERSION = "geometry-repair.request.v1"
SOURCE_PACKAGE_SCHEMA_VERSION = "geometry-repair.source-package.v1"
DIAGNOSIS_SCHEMA_VERSION = "geometry-repair.diagnosis.v1"
REPAIR_PLAN_SCHEMA_VERSION = "geometry-repair.plan.v1"
FIDELITY_SCHEMA_VERSION = "geometry-repair.fidelity.v1"
REPAIR_CERTIFICATE_SCHEMA_VERSION = "geometry-repair.certificate.v1"
REPAIR_RESULT_SCHEMA_VERSION = "geometry-repair.result.v1"
COLLISION_REPORT_SCHEMA_VERSION = "geometry-repair.collision-report.v1"
SOURCE_FORMAT_VALIDATION_SCHEMA_VERSION = "geometry-repair.source-format-validation.v1"

RepairProfile = Literal[
    "visual_only",
    "static_environment",
    "rigid_pick_place",
    "articulated_rigid",
    "contact_rich",
    "deformable_or_cae",
]
RepairMode = Literal["diagnose", "auto"]
RepairOutcome = Literal["certified", "conditional", "rejected"]
FactKind = Literal["source_fact", "measured_fact", "inferred_hypothesis", "generated_geometry"]
IssueSeverity = Literal["info", "warning", "error"]
DriftBand = Literal["identity", "conservative", "moderate", "reconstructive"]
GeometryRole = Literal["render", "collision", "brep_source", "helper"]
RoleValidationStatus = Literal["pass", "conditional", "fail", "not_evaluated", "not_applicable"]
RepairIntentAuthority = Literal["deterministic", "agent_proposal", "human_confirmed"]
TopologyEffect = Literal[
    "none",
    "delete_source_faces",
    "reverse_source_faces",
    "merge_source_vertices",
    "split_source_faces",
    "generate_local_faces",
    "replace_target_surface",
    "generate_collision_representation",
    "heal_brep_topology",
]


class StrictModel(BaseModel):
    """Shared strict Pydantic base."""

    model_config = ConfigDict(extra="forbid")


class RoleValidation(StrictModel):
    """Evidence-backed state for one geometry representation role."""

    role: GeometryRole
    required: bool
    status: RoleValidationStatus
    affected_prim_paths: list[str] = Field(default_factory=list)
    issue_ids: list[str] = Field(default_factory=list)
    evidence_paths: list[str] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_status_evidence(self) -> RoleValidation:
        for field_name in (
            "affected_prim_paths",
            "issue_ids",
            "evidence_paths",
            "blockers",
            "warnings",
        ):
            values = getattr(self, field_name)
            if len(values) != len(set(values)):
                raise ValueError(f"{field_name} must not contain duplicates")
        if self.status == "pass" and self.blockers:
            raise ValueError("passing role validation cannot contain blockers")
        if self.status == "fail" and not self.blockers:
            raise ValueError("failed role validation requires at least one blocker")
        if self.required and self.status == "not_applicable":
            raise ValueError("a required role cannot be not_applicable")
        return self


class RepairIntent(StrictModel):
    """Typed proposal that can rank operations but cannot relax validation gates."""

    intent_id: str = Field(min_length=1)
    authority: RepairIntentAuthority
    target_role: GeometryRole
    target_prim_paths: list[str] = Field(default_factory=list)
    issue_ids: list[str] = Field(default_factory=list)
    defect_class: str = Field(min_length=1)
    protected_feature_names: list[str] = Field(default_factory=list)
    expected_topology_effects: list[TopologyEffect] = Field(default_factory=list)
    candidate_workers: list[str] = Field(default_factory=list)
    evidence_paths: list[str] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    notes: list[str] = Field(default_factory=list)

    @field_validator("candidate_workers", mode="after")
    @classmethod
    def _canonicalize_candidate_workers(cls, value: list[str]) -> list[str]:
        return canonical_worker_names(value)

    @model_validator(mode="after")
    def _validate_bounded_proposal(self) -> RepairIntent:
        for field_name in (
            "target_prim_paths",
            "issue_ids",
            "protected_feature_names",
            "expected_topology_effects",
            "candidate_workers",
            "evidence_paths",
            "notes",
        ):
            values = getattr(self, field_name)
            if len(values) != len(set(values)):
                raise ValueError(f"{field_name} must not contain duplicates")
        if self.authority != "deterministic" and not self.issue_ids:
            raise ValueError("agent or human repair intent requires measured issue_ids")
        if self.authority == "agent_proposal" and not self.evidence_paths:
            raise ValueError("agent repair intent requires evidence_paths")
        if not self.candidate_workers:
            raise ValueError("repair intent requires at least one candidate worker")
        if not self.expected_topology_effects:
            raise ValueError("repair intent requires expected_topology_effects")
        return self


class ProtectedFeatureProbe(StrictModel):
    """Deterministic geometry query used to prove a protected feature survived."""

    kind: Literal["negative_space_path", "surface_support"]
    points_m: list[list[float]] = Field(min_length=1, max_length=256)
    radius_m: float | None = Field(default=None, ge=0.0)
    tolerance_m: float | None = Field(default=None, gt=0.0)

    @model_validator(mode="after")
    def _validate_points(self) -> ProtectedFeatureProbe:
        if any(
            len(point) != 3
            or any(
                not isinstance(value, int | float) or not math.isfinite(float(value))
                for value in point
            )
            for point in self.points_m
        ):
            raise ValueError("protected feature probe points_m must contain finite XYZ triples")
        if self.kind == "negative_space_path" and len(self.points_m) < 2:
            raise ValueError("negative_space_path probes require at least two points")
        return self


class ProtectedFeature(StrictModel):
    """Source feature that a repair operation must preserve."""

    name: str = Field(min_length=1)
    kind: Literal[
        "part",
        "material_region",
        "opening",
        "cavity",
        "clearance",
        "handle",
        "mating_surface",
        "interface",
    ]
    scope_path: str | None = None
    minimum_size_m: float | None = Field(default=None, gt=0.0)
    minimum_clearance_m: float | None = Field(default=None, gt=0.0)
    tolerance_m: float | None = Field(default=None, gt=0.0)
    probe: ProtectedFeatureProbe | None = None
    required: bool = True
    source: FactKind = "source_fact"
    affected_roles: list[GeometryRole] = Field(default_factory=lambda: ["render", "collision"])

    @model_validator(mode="after")
    def _validate_affected_roles(self) -> ProtectedFeature:
        if not self.affected_roles:
            raise ValueError("protected feature requires at least one affected role")
        if len(self.affected_roles) != len(set(self.affected_roles)):
            raise ValueError("protected feature affected_roles must not contain duplicates")
        return self


class ClassifiedHoleIntent(StrictModel):
    """Trusted intent for one accidental boundary loop; geometry is still revalidated."""

    target_mesh_path: str = Field(min_length=1)
    issue_id: str = Field(default="mesh:boundary_edges", min_length=1)
    intent_evidence_id: str = Field(min_length=1)
    confirmed_accidental_hole: Literal[True]
    boundary_loop_vertex_ids: list[int] = Field(min_length=5, max_length=4096)
    frozen_boundary_vertex_ids: list[int] = Field(default_factory=list, max_length=4096)
    protected_edge_vertex_pairs: list[tuple[int, int]] = Field(
        default_factory=list,
        max_length=4096,
    )
    max_loop_perimeter_ratio: float = Field(default=0.25, gt=0.0, le=0.5)
    max_patch_area_ratio: float = Field(default=0.05, gt=0.0, le=0.1)
    max_nonplanarity_ratio: float = Field(default=0.005, ge=0.0, le=0.02)
    max_boundary_turn_radians: float = Field(default=math.pi, gt=0.0, le=math.pi)
    max_envelope_ratio: float = Field(default=0.001, gt=0.0, le=0.005)
    max_new_vertices: int = Field(default=8192, ge=1, le=100_000)
    timeout_s: float = Field(default=120.0, ge=1.0, le=300.0)

    @model_validator(mode="after")
    def _complete_and_validate_boundary_guards(self) -> ClassifiedHoleIntent:
        loop = self.boundary_loop_vertex_ids
        if len(set(loop)) != len(loop) or any(value < 0 for value in loop):
            raise ValueError("boundary_loop_vertex_ids must be unique non-negative IDs")
        if not self.frozen_boundary_vertex_ids:
            self.frozen_boundary_vertex_ids = list(loop)
        if not self.protected_edge_vertex_pairs:
            self.protected_edge_vertex_pairs = [
                (loop[index], loop[(index + 1) % len(loop)]) for index in range(len(loop))
            ]
        if not set(loop) <= set(self.frozen_boundary_vertex_ids):
            raise ValueError("every classified boundary vertex must be frozen")
        required_edges = {
            tuple(sorted((loop[index], loop[(index + 1) % len(loop)])))
            for index in range(len(loop))
        }
        supplied_edges = {tuple(sorted(edge)) for edge in self.protected_edge_vertex_pairs}
        if not required_edges <= supplied_edges:
            raise ValueError("every classified boundary edge must be protected")
        return self


class RepairBudgets(StrictModel):
    """Resource, drift, and collision limits for one repair job."""

    max_attempts: int = Field(default=8, ge=1, le=32)
    timeout_s: float = Field(default=900.0, gt=0.0, le=86_400.0)
    max_memory_mb: int = Field(default=8192, ge=256, le=262_144)
    max_dependency_count: int = Field(default=20_000, ge=1, le=1_000_000)
    max_dependency_hash_bytes: int = Field(
        default=8 * 1024 * 1024 * 1024,
        ge=1,
        le=1024 * 1024 * 1024 * 1024,
    )
    audit_broad_pair_limit: int = Field(default=5_000_000, ge=1, le=100_000_000)
    audit_exact_test_limit: int = Field(default=1_000_000, ge=1, le=100_000_000)
    audit_wall_time_s: float = Field(default=60.0, gt=0.0, le=86_400.0)
    audit_chunk_size: int = Field(default=2048, ge=1, le=65_536)
    max_protected_feature_candidates: int = Field(default=256, ge=1, le=4096)
    sample_point_limit: int = Field(default=4096, ge=128, le=100_000)
    conservative_p99_ratio: float = Field(default=0.001, ge=0.0, le=0.1)
    conservative_volume_drift: float = Field(default=0.01, ge=0.0, le=1.0)
    moderate_p99_ratio: float = Field(default=0.005, ge=0.0, le=0.2)
    moderate_volume_drift: float = Field(default=0.03, ge=0.0, le=1.0)
    reconstructive_p99_ratio: float = Field(default=0.01, ge=0.0, le=0.5)
    reconstructive_volume_drift: float = Field(default=0.05, ge=0.0, le=1.0)
    max_reconstruction_grid_dimension: int = Field(default=256, ge=32, le=512)
    max_collision_reconstruction_grid_dimension: int = Field(default=128, ge=32, le=256)
    max_reconstruction_output_faces: int = Field(
        default=1_000_000,
        ge=1_000,
        le=5_000_000,
    )
    reconstruction_feature_voxels: int = Field(default=6, ge=2, le=32)
    # Generated decomposition is an optimization product and has a tight task
    # budget. Authored source collision has a separate, higher intake guard and
    # is accepted or rejected by fidelity/runtime evidence rather than count.
    max_collision_hulls: int = Field(default=128, ge=1, le=256)
    max_source_collision_prims: int = Field(default=1024, ge=1, le=10_000)
    max_collision_source_vertices_per_body: int = Field(
        default=250_000,
        ge=4,
        le=5_000_000,
    )
    max_collision_occupancy_face_point_product: int = Field(
        default=100_000_000,
        ge=1,
        le=10_000_000_000,
    )
    source_collision_audit_wall_time_s: float = Field(
        default=120.0,
        gt=0.0,
        le=3_600.0,
    )
    max_collision_decomposition_time_s: float = Field(
        default=120.0,
        gt=0.0,
        le=3_600.0,
    )
    max_collision_vertices_per_hull: int = Field(default=255, ge=4, le=255)
    max_collision_faces_per_hull: int = Field(default=255, ge=4, le=255)
    max_collision_volume_excess: float = Field(default=0.10, ge=0.0, le=5.0)
    max_collision_volume_deficit: float = Field(default=0.01, ge=0.0, le=1.0)
    max_collision_surface_distance_ratio: float = Field(default=0.05, ge=0.0, le=0.5)
    allow_reconstructive: bool = False


class RepairRequest(StrictModel):
    """Complete input to a deterministic geometry repair job."""

    schema_version: Literal["geometry-repair.request.v1"] = REPAIR_REQUEST_SCHEMA_VERSION
    source_path: Path
    output_dir: Path
    profile: RepairProfile
    mode: RepairMode = "diagnose"
    profile_confirmed: bool = True
    production_use: bool = False
    protected_features: list[ProtectedFeature] = Field(default_factory=list)
    classified_holes: list[ClassifiedHoleIntent] = Field(default_factory=list)
    proposed_intents: list[RepairIntent] = Field(default_factory=list)
    use_proposed_intent_ranking: bool = False
    budgets: RepairBudgets = Field(default_factory=RepairBudgets)
    enabled_workers: list[str] | None = None
    deterministic_seed: int = Field(default=0, ge=0, le=2**31 - 1)
    collision_runtime_engine: Literal["skip", "fake", "ovphysx"] = "skip"
    source_uri: str | None = None
    source_license: str | None = None
    source_provenance: dict[str, Any] = Field(default_factory=dict)
    source_meters_per_unit: float | None = Field(default=None, gt=0.0)
    source_up_axis: Literal["X", "Y", "Z"] | None = None
    dependency_roots: list[Path] = Field(default_factory=list)
    dependency_remap_manifest: Path | dict[str, Any] | None = None
    advanced_profile: dict[str, Any] | None = None

    @field_validator("enabled_workers", mode="after")
    @classmethod
    def _canonicalize_enabled_workers(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        return canonical_worker_names(value)

    @model_validator(mode="after")
    def _reject_duplicate_workers(self) -> RepairRequest:
        if self.enabled_workers is not None and len(set(self.enabled_workers)) != len(
            self.enabled_workers
        ):
            raise ValueError("enabled_workers must not contain duplicates")
        hole_keys = [
            (intent.target_mesh_path, tuple(intent.boundary_loop_vertex_ids))
            for intent in self.classified_holes
        ]
        if len(set(hole_keys)) != len(hole_keys):
            raise ValueError("classified_holes must not contain duplicate target loops")
        intent_ids = [intent.intent_id for intent in self.proposed_intents]
        if len(set(intent_ids)) != len(intent_ids):
            raise ValueError("proposed_intents must not contain duplicate intent_id values")
        resolved_roots = [path.expanduser().resolve() for path in self.dependency_roots]
        if len(set(resolved_roots)) != len(resolved_roots):
            raise ValueError("dependency_roots must not contain duplicates")
        if isinstance(self.dependency_remap_manifest, Path):
            self.dependency_remap_manifest = self.dependency_remap_manifest.expanduser().resolve()
        if self.advanced_profile is not None:
            if self.profile not in {"articulated_rigid", "contact_rich"}:
                raise ValueError(
                    "advanced_profile is valid only for articulated_rigid or contact_rich"
                )
            if self.advanced_profile.get("profile") != self.profile:
                raise ValueError("advanced_profile.profile must match the repair profile")
        return self


class SourceArtifact(StrictModel):
    """One immutable source or dependency copied into the repair job."""

    logical_role: Literal["source", "dependency"]
    original_path: str
    preserved_path: str
    sha256: str
    size_bytes: int = Field(ge=0)


class SourcePackage(StrictModel):
    """Immutable source snapshot and dependency-resolution evidence."""

    schema_version: Literal["geometry-repair.source-package.v1"] = SOURCE_PACKAGE_SCHEMA_VERSION
    source: SourceArtifact
    dependencies: list[SourceArtifact] = Field(default_factory=list)
    unresolved_dependencies: list[str] = Field(default_factory=list)
    unresolved_geometry_dependencies: list[str] = Field(default_factory=list)
    unresolved_material_dependencies: list[str] = Field(default_factory=list)
    resolved_snapshot_path: str | None = None
    resolved_snapshot_sha256: str | None = None
    source_uri: str | None = None
    source_license: str | None = None
    source_provenance: dict[str, Any] = Field(default_factory=dict)
    manifest_path: str


class SourceFormatValidationReport(StrictModel):
    """Authoritative source-format validation before geometry mutation."""

    schema_version: Literal["geometry-repair.source-format-validation.v1"] = (
        SOURCE_FORMAT_VALIDATION_SCHEMA_VERSION
    )
    source_path: str
    source_sha256: str
    source_format: str
    status: Literal["pass", "fail", "not_evaluated", "not_applicable"]
    validator: str | None = None
    validator_version: str | None = None
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    report_path: str | None = None


class MeshMetrics(StrictModel):
    """Topology and geometry measurements for one mesh or aggregate."""

    mesh_count: int = Field(default=0, ge=0)
    point_count: int = Field(default=0, ge=0)
    face_count: int = Field(default=0, ge=0)
    triangle_count: int = Field(default=0, ge=0)
    invalid_index_count: int = Field(default=0, ge=0)
    non_finite_vertex_count: int = Field(default=0, ge=0)
    degenerate_face_count: int = Field(default=0, ge=0)
    duplicate_face_count: int = Field(default=0, ge=0)
    duplicate_vertex_count: int = Field(default=0, ge=0)
    near_duplicate_vertex_count: int = Field(default=0, ge=0)
    unused_vertex_count: int = Field(default=0, ge=0)
    indexed_boundary_edge_count: int = Field(default=0, ge=0)
    boundary_edge_count: int = Field(default=0, ge=0)
    boundary_loop_count: int = Field(default=0, ge=0)
    open_boundary_chain_count: int = Field(default=0, ge=0)
    over_connected_edge_count: int = Field(default=0, ge=0)
    non_manifold_vertex_count: int = Field(default=0, ge=0)
    inconsistent_orientation_edge_count: int = Field(default=0, ge=0)
    non_triangular_face_count: int = Field(default=0, ge=0)
    needle_triangle_count: int = Field(default=0, ge=0)
    triangle_aspect_ratio_p95: float | None = Field(default=None, ge=1.0)
    triangle_aspect_ratio_max: float | None = Field(default=None, ge=1.0)
    connected_component_count: int = Field(default=0, ge=0)
    smallest_component_face_count: int | None = Field(default=None, ge=0)
    smallest_component_area_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    tiny_component_count: int = Field(default=0, ge=0)
    inverted_shell_count: int = Field(default=0, ge=0)
    inverted_shell_status: Literal["pass", "fail", "not_evaluated"] = "not_evaluated"
    nested_shell_count: int = Field(default=0, ge=0)
    nested_shell_status: Literal["pass", "fail", "not_evaluated"] = "not_evaluated"
    coplanar_overlap_count: int = Field(default=0, ge=0)
    coplanar_overlap_status: Literal["pass", "fail", "not_evaluated"] = "not_evaluated"
    material_subset_count: int = Field(default=0, ge=0)
    authored_normal_count: int = Field(default=0, ge=0)
    non_finite_normal_count: int = Field(default=0, ge=0)
    authored_uv_count: int = Field(default=0, ge=0)
    non_finite_uv_count: int = Field(default=0, ge=0)
    non_finite_transform_count: int = Field(default=0, ge=0)
    singular_transform_count: int = Field(default=0, ge=0)
    non_uniform_transform_count: int = Field(default=0, ge=0)
    sheared_transform_count: int = Field(default=0, ge=0)
    reflected_transform_count: int = Field(default=0, ge=0)
    surface_area_m2: float | None = Field(default=None, ge=0.0)
    enclosed_volume_m3: float | None = None
    bbox_min_m: list[float] | None = None
    bbox_max_m: list[float] | None = None
    bbox_diagonal_m: float | None = Field(default=None, ge=0.0)
    minimum_bbox_extent_m: float | None = Field(default=None, ge=0.0)
    geometric_dimension: Literal[0, 1, 2, 3] | None = None
    zero_thickness_status: Literal["pass", "fail", "not_evaluated"] = "not_evaluated"
    watertight: bool | None = None
    euler_characteristic: int | None = None
    genus: float | None = Field(default=None, ge=0.0)
    self_intersection_status: Literal["pass", "fail", "not_evaluated"] = "not_evaluated"
    self_intersection_count: int = Field(default=0, ge=0)
    self_intersection_broad_phase_pairs: int = Field(default=0, ge=0)
    self_intersection_candidate_pairs: int = Field(default=0, ge=0)
    self_intersection_reason: str | None = None


class MeshRecord(StrictModel):
    """Per-mesh diagnostic record retaining source identity."""

    path: str
    purpose: str | None = None
    role: Literal["render", "collision", "helper"] = "render"
    is_instance_proxy: bool = False
    metrics: MeshMetrics
    topology_status: Literal["pass", "warning", "fail", "not_evaluated"]
    has_face_material_subsets: bool = False
    has_authored_normals: bool = False
    has_authored_uvs: bool = False
    has_face_varying_data: bool = False


class BRepMetrics(StrictModel):
    """OpenCascade validity and mass-property measurements for native CAD."""

    evaluated: bool = False
    valid: bool | None = None
    solid_count: int = Field(default=0, ge=0)
    shell_count: int = Field(default=0, ge=0)
    face_count: int = Field(default=0, ge=0)
    wire_count: int = Field(default=0, ge=0)
    edge_count: int = Field(default=0, ge=0)
    surface_area_source_units2: float | None = Field(default=None, ge=0.0)
    volume_source_units3: float | None = None
    center_of_mass_source_units: list[float] | None = None
    bbox_min_source_units: list[float] | None = None
    bbox_max_source_units: list[float] | None = None
    maximum_tolerance_source_units: float | None = Field(default=None, ge=0.0)
    validity_status_counts: dict[str, int] = Field(default_factory=dict)
    subshape_status_counts: dict[str, dict[str, int]] = Field(default_factory=dict)
    minimum_edge_length_source_units: float | None = Field(default=None, ge=0.0)
    minimum_face_area_source_units2: float | None = Field(default=None, ge=0.0)
    tiny_edge_count: int = Field(default=0, ge=0)
    tiny_face_count: int = Field(default=0, ge=0)
    tiny_feature_threshold_source_units: float | None = Field(default=None, ge=0.0)
    error: str | None = None


class GeometryMetrics(StrictModel):
    """Asset-level geometry, stage, and correspondence measurements."""

    source_format: str
    up_axis: str | None = None
    meters_per_unit: float | None = Field(default=None, gt=0.0)
    root_count: int = Field(default=0, ge=0)
    default_prim_path: str | None = None
    mesh: MeshMetrics = Field(default_factory=MeshMetrics)
    meshes: list[MeshRecord] = Field(default_factory=list)
    brep: BRepMetrics = Field(default_factory=BRepMetrics)
    source_part_paths: list[str] = Field(default_factory=list)
    source_collision_paths: list[str] = Field(default_factory=list)
    source_helper_paths: list[str] = Field(default_factory=list)
    collision_source_mesh_count: int = Field(default=0, ge=0)
    helper_source_mesh_count: int = Field(default=0, ge=0)
    instance_proxy_mesh_count: int = Field(default=0, ge=0)
    material_binding_count: int = Field(default=0, ge=0)
    unresolved_dependency_count: int = Field(default=0, ge=0)
    unresolved_geometry_dependency_count: int = Field(default=0, ge=0)
    unresolved_material_dependency_count: int = Field(default=0, ge=0)
    self_intersection_status: Literal["pass", "fail", "not_evaluated"] = "not_evaluated"
    inter_part_intersection_status: Literal["pass", "fail", "not_evaluated"] = "not_evaluated"
    inter_part_intersection_count: int = Field(default=0, ge=0)
    inter_part_intersection_reason: str | None = None


class DiagnosisIssue(StrictModel):
    """Stable, evidence-backed diagnosis item."""

    issue_id: str = Field(min_length=1)
    category: Literal["format", "numeric", "topology", "brep", "semantic", "task"]
    scope: str
    severity: IssueSeverity
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    fact_kind: FactKind
    summary: str = Field(min_length=1)
    evidence: dict[str, Any] = Field(default_factory=dict)
    candidate_repairs: list[str] = Field(default_factory=list)
    blocking_profiles: list[RepairProfile] = Field(default_factory=list)
    affected_roles: list[GeometryRole] = Field(default_factory=lambda: ["render"])
    affected_prim_paths: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_role_scope(self) -> DiagnosisIssue:
        if not self.affected_roles:
            raise ValueError("diagnosis issue requires at least one affected role")
        if len(self.affected_roles) != len(set(self.affected_roles)):
            raise ValueError("affected_roles must not contain duplicates")
        if len(self.affected_prim_paths) != len(set(self.affected_prim_paths)):
            raise ValueError("affected_prim_paths must not contain duplicates")
        return self


class Diagnosis(StrictModel):
    """Deterministic diagnosis emitted before repair planning."""

    schema_version: Literal["geometry-repair.diagnosis.v1"] = DIAGNOSIS_SCHEMA_VERSION
    source_path: str
    source_sha256: str
    profile: RepairProfile
    metrics: GeometryMetrics
    issues: list[DiagnosisIssue] = Field(default_factory=list)
    blocking_issue_ids: list[str] = Field(default_factory=list)
    repairable_issue_ids: list[str] = Field(default_factory=list)
    verified_issue_ids: list[str] = Field(default_factory=list)
    role_validation: dict[GeometryRole, RoleValidation] = Field(default_factory=dict)
    status: Literal["pass", "conditional", "fail"]
    report_path: str | None = None
    scalable_audit_path: str | None = None


class RepairOperation(StrictModel):
    """One typed and bounded operation selected by a repair plan."""

    operation_id: str
    worker: str
    implementation: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    issue_ids: list[str] = Field(default_factory=list)
    drift_band: DriftBand
    source_checkpoint: str
    expected_changes: list[str] = Field(default_factory=list)
    intent_id: str | None = None
    target_role: GeometryRole = "render"
    target_prim_paths: list[str] = Field(default_factory=list)

    @field_validator("worker", mode="before")
    @classmethod
    def _canonicalize_worker(cls, value: object) -> object:
        return canonical_worker_name(value) if isinstance(value, str) else value


class RepairPlan(StrictModel):
    """Auditable sequence of deterministic repair operations."""

    schema_version: Literal["geometry-repair.plan.v1"] = REPAIR_PLAN_SCHEMA_VERSION
    source_sha256: str
    profile: RepairProfile
    operations: list[RepairOperation] = Field(default_factory=list)
    intents: list[RepairIntent] = Field(default_factory=list)
    protected_features: list[ProtectedFeature] = Field(default_factory=list)
    budgets: RepairBudgets
    deterministic_seed: int = Field(ge=0)
    routing_policy_id: str = "geometry-repair.route-policy.phase1-observed.v1"
    routing_evidence: dict[str, Any] = Field(default_factory=dict)
    rationale: str
    plan_path: str | None = None


class ProtectedFeatureProbeResult(StrictModel):
    """Measured result for a task-critical render or collision-space probe."""

    feature_name: str
    probe_kind: Literal["negative_space_path", "surface_support"]
    status: Literal["pass", "fail", "not_evaluated"]
    sample_count: int = Field(default=0, ge=0)
    minimum_clearance_m: float | None = Field(default=None, ge=0.0)
    maximum_surface_distance_m: float | None = Field(default=None, ge=0.0)
    occupied_sample_count: int = Field(default=0, ge=0)
    failures: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class FidelityReport(StrictModel):
    """Candidate drift measured directly against the normalized source."""

    schema_version: Literal["geometry-repair.fidelity.v1"] = FIDELITY_SCHEMA_VERSION
    source_path: str
    candidate_path: str
    drift_band: DriftBand
    status: Literal["pass", "conditional", "fail", "not_evaluated"]
    exact_world_geometry_match: bool = False
    exact_world_surface_match: bool = False
    sample_count_source: int = Field(default=0, ge=0)
    sample_count_candidate: int = Field(default=0, ge=0)
    surface_distance_median_m: float | None = Field(default=None, ge=0.0)
    surface_distance_p95_m: float | None = Field(default=None, ge=0.0)
    surface_distance_p99_m: float | None = Field(default=None, ge=0.0)
    surface_distance_max_m: float | None = Field(default=None, ge=0.0)
    normal_angle_median_deg: float | None = Field(default=None, ge=0.0, le=180.0)
    normal_angle_p95_deg: float | None = Field(default=None, ge=0.0, le=180.0)
    normal_angle_max_deg: float | None = Field(default=None, ge=0.0, le=180.0)
    p99_bbox_ratio: float | None = Field(default=None, ge=0.0)
    bbox_max_drift_m: float | None = Field(default=None, ge=0.0)
    bbox_max_drift_ratio: float | None = Field(default=None, ge=0.0)
    centroid_drift_ratio: float | None = Field(default=None, ge=0.0)
    oriented_bbox_extent_drift_ratio: float | None = Field(default=None, ge=0.0)
    surface_area_drift_ratio: float | None = Field(default=None, ge=0.0)
    volume_drift_ratio: float | None = Field(default=None, ge=0.0)
    part_count_delta: int = 0
    component_count_delta: int = 0
    boundary_loop_count_delta: int = 0
    genus_delta: float | None = None
    material_subset_count_delta: int = 0
    material_binding_count_delta: int = 0
    brep_identity_deltas: dict[str, int] = Field(default_factory=dict)
    missing_part_paths: list[str] = Field(default_factory=list)
    added_part_paths: list[str] = Field(default_factory=list)
    silhouette_iou_by_view: dict[str, float] = Field(default_factory=dict)
    silhouette_iou_min: float | None = Field(default=None, ge=0.0, le=1.0)
    source_face_coverage_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    candidate_face_coverage_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    material_mapping_coverage_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    part_mapping_coverage_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    part_correspondence: list[dict[str, Any]] = Field(default_factory=list)
    ambiguous_part_mapping_count: int = Field(default=0, ge=0)
    correspondence_mode: Literal[
        "identity",
        "exact_surface",
        "per_part_surface_projection",
        "unavailable",
    ] = "unavailable"
    protected_features_passed: list[str] = Field(default_factory=list)
    protected_features_unmeasured: list[str] = Field(default_factory=list)
    protected_feature_probes: list[ProtectedFeatureProbeResult] = Field(default_factory=list)
    failures: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    report_path: str | None = None


class AttemptRecord(StrictModel):
    """One immutable candidate attempt and independent remeasurement."""

    attempt_id: str
    operation: RepairOperation
    status: Literal["accepted", "advanced", "rejected", "unavailable", "failed"]
    output_path: str | None = None
    output_sha256: str | None = Field(
        default=None,
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    diagnosis_path: str | None = None
    fidelity_path: str | None = None
    worker_report_path: str | None = None
    worker_report_sha256: str | None = Field(
        default=None,
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    correspondence_path: str | None = None
    protected_feature_comparison_path: str | None = None
    reasons: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_worker_report_binding(self) -> AttemptRecord:
        if (self.worker_report_path is None) != (self.worker_report_sha256 is None):
            raise ValueError("worker report path and SHA-256 must be recorded together")
        return self


class CollisionDecompositionEvidence(StrictModel):
    """Verified evidence from one isolated convex-decomposition invocation."""

    source_path: str
    worker: Literal["coacd_collision"] = "coacd_collision"
    worker_version: str
    requested_threshold_m: float = Field(gt=0.0)
    thread_limit: int = Field(ge=1, le=64)
    input_sha256: str
    source_triangle_count: int = Field(default=0, ge=0)
    decomposition_input_triangle_count: int = Field(default=0, ge=0)
    preprocessing: str | None = None
    hull_count: int = Field(ge=1)
    elapsed_s: float = Field(ge=0.0)
    report_path: str
    report_sha256: str
    stdout_path: str
    stdout_sha256: str
    stderr_path: str
    stderr_sha256: str
    warnings: list[str] = Field(default_factory=list)


class RuntimeScenarioEvidence(StrictModel):
    """One deterministic simulator-backed collision canary."""

    scenario_id: str
    scenario_kind: Literal["static_contact", "rigid_drop_settle"]
    status: Literal["pass", "fail", "not_evaluated"]
    orientation_xyzw: list[float] | None = None
    runtime_report_path: str | None = None
    temporary_proxy_path: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    failures: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class CollisionReport(StrictModel):
    """Geometry-owned collision representation evidence."""

    schema_version: Literal["geometry-repair.collision-report.v1"] = COLLISION_REPORT_SCHEMA_VERSION
    status: Literal["pass", "conditional", "fail", "not_required"]
    representation: Literal[
        "none",
        "static_triangle_mesh",
        "primitive_fit",
        "source_collision",
        "convex_hull",
        "coacd",
        "hybrid",
    ] = "none"
    source_render_path: str | None = None
    source_render_sha256: str | None = None
    collision_path: str | None = None
    collision_sha256: str | None = None
    generator: str | None = None
    generator_version: str | None = None
    source_body_count: int = Field(default=0, ge=0)
    hull_count: int = Field(default=0, ge=0)
    source_collision_prim_count: int = Field(default=0, ge=0)
    generated_collision_prim_count: int = Field(default=0, ge=0)
    total_vertices: int = Field(default=0, ge=0)
    total_faces: int = Field(default=0, ge=0)
    collision_size_bytes: int = Field(default=0, ge=0)
    maximum_vertices_per_hull: int = Field(default=0, ge=0)
    maximum_faces_per_hull: int = Field(default=0, ge=0)
    maximum_volume_excess_ratio: float | None = Field(default=None, ge=0.0)
    maximum_volume_deficit_ratio: float | None = Field(default=None, ge=0.0)
    maximum_surface_overreach_m: float | None = Field(default=None, ge=0.0)
    maximum_surface_gap_m: float | None = Field(default=None, ge=0.0)
    maximum_false_positive_ratio: float | None = Field(default=None, ge=0.0)
    maximum_false_negative_ratio: float | None = Field(default=None, ge=0.0)
    primitive_fit_count: int = Field(default=0, ge=0)
    primitive_fit_kinds: list[str] = Field(default_factory=list)
    candidate_search_paths: list[str] = Field(default_factory=list)
    decompositions: list[CollisionDecompositionEvidence] = Field(default_factory=list)
    decomposition_report_paths: list[str] = Field(default_factory=list)
    collision_reconstruction_count: int = Field(default=0, ge=0)
    collision_reconstruction_paths: list[str] = Field(default_factory=list)
    source_render_sha256_after: str | None = None
    source_render_unchanged: bool | None = None
    protected_features_applied: list[str] = Field(default_factory=list)
    protected_features_unmeasured: list[str] = Field(default_factory=list)
    protected_feature_probes: list[ProtectedFeatureProbeResult] = Field(default_factory=list)
    inferred_protected_features_applied: list[str] = Field(default_factory=list)
    inferred_protected_features_unmeasured: list[str] = Field(default_factory=list)
    inferred_protected_feature_probes: list[ProtectedFeatureProbeResult] = Field(
        default_factory=list
    )
    role_excluded_protected_features: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    failures: list[str] = Field(default_factory=list)
    runtime_status: Literal["pass", "fail", "not_evaluated"] = "not_evaluated"
    runtime_engine: Literal["skip", "fake", "ovphysx"] = "skip"
    runtime_report_path: str | None = None
    temporary_proxy_path: str | None = None
    runtime_scenarios: list[RuntimeScenarioEvidence] = Field(default_factory=list)
    report_path: str | None = None
    source_collision_audit_path: str | None = None


class RepairCertificate(StrictModel):
    """Final geometry-scoped repair certificate."""

    schema_version: Literal["geometry-repair.certificate.v1"] = REPAIR_CERTIFICATE_SCHEMA_VERSION
    outcome: RepairOutcome
    claim_scope: str
    source_sha256: str
    normalized_source_sha256: str
    output_sha256: str | None = None
    profile: RepairProfile
    profile_confirmed: bool
    production_use: bool = False
    source_rights_status: Literal["pass", "missing", "not_required"] = "not_required"
    render_geometry_changed: bool = False
    visual_review_status: Literal["not_required", "required"] = "not_required"
    deterministic_seed: int = Field(ge=0)
    routing_policy_id: str | None = None
    accepted_attempt_id: str | None = None
    accepted_backend_identity: dict[str, Any] | None = None
    accepted_backend_qualification_id: str | None = Field(default=None, min_length=1)
    operations_applied: list[str] = Field(default_factory=list)
    deleted_entities: list[str] = Field(default_factory=list)
    merged_entities: list[str] = Field(default_factory=list)
    diagnosis_path: str
    source_format_validation_path: str | None = None
    repair_plan_path: str
    fidelity_path: str | None = None
    collision_report_path: str | None = None
    source_collision_audit_path: str | None = None
    dependency_localization_path: str | None = None
    protected_feature_candidates_path: str | None = None
    manifold_seam_analysis_path: str | None = None
    advanced_profile_evidence_path: str | None = None
    validation_results: dict[str, str] = Field(default_factory=dict)
    role_validation: dict[GeometryRole, RoleValidation] = Field(default_factory=dict)
    remaining_warnings: list[str] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    downstream_owners: list[str] = Field(default_factory=list)
    certificate_path: str | None = None

    @model_validator(mode="after")
    def _validate_accepted_backend_binding(self) -> RepairCertificate:
        if (self.accepted_backend_identity is None) != (
            self.accepted_backend_qualification_id is None
        ):
            raise ValueError("accepted backend identity and qualification ID must be paired")
        if self.accepted_backend_identity is not None:
            backend_id = self.accepted_backend_identity.get("backend_id")
            if not isinstance(backend_id, str) or not backend_id:
                raise ValueError("accepted backend identity requires a backend_id")
        return self


class RepairResult(StrictModel):
    """Canonical paths and outcome from one repair run."""

    schema_version: Literal["geometry-repair.result.v1"] = REPAIR_RESULT_SCHEMA_VERSION
    outcome: RepairOutcome
    claim_scope: str
    output_dir: str
    source_package_path: str
    source_format_validation_path: str | None = None
    usd_intake_path: str | None = None
    normalized_source_path: str
    render_usd_path: str | None = None
    collision_usd_path: str | None = None
    composed_usd_path: str | None = None
    diagnosis_path: str
    scalable_audit_path: str | None = None
    repair_plan_path: str
    attempts_path: str
    certificate_path: str
    manifest_path: str
    geometry_validation_evidence_path: str
    report_path: str
    collision_report_path: str | None = None
    correspondence_path: str | None = None
    source_collision_audit_path: str | None = None
    dependency_localization_path: str | None = None
    protected_feature_candidates_path: str | None = None
    manifold_seam_analysis_path: str | None = None
    advanced_profile_evidence_path: str | None = None
    attempts: list[AttemptRecord] = Field(default_factory=list)
    role_validation: dict[GeometryRole, RoleValidation] = Field(default_factory=dict)
    error: str | None = None
