# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SimReady Foundation workflow report contracts."""

from __future__ import annotations

from typing import Annotated, Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

DEFAULT_SIMREADY_PROFILE = "Prop-Robotics-Neutral"
DEFAULT_SIMREADY_PROFILE_VERSION = "1.0.0"
DEFAULT_SIMREADY_FOUNDATION_REPO_URL = "https://github.com/NVIDIA/simready-foundation"
DEFAULT_SIMREADY_FOUNDATION_REF = "main"

SIMREADY_PREFLIGHT_SCHEMA_VERSION = "content-agent-workflows.simready-preflight.v1"
SIMREADY_VALIDATION_SCHEMA_VERSION = (
    "content-agent-workflows.simready-profile-validation.v1"
)
SIMREADY_CONFORMANCE_SCHEMA_VERSION = (
    "content-agent-workflows.simready-profile-conformance.v1"
)
SIMREADY_GRASP_PLAN_SCHEMA_VERSION: Final[
    Literal["content-agent-workflows.simready-grasp-plan.v1"]
] = "content-agent-workflows.simready-grasp-plan.v1"
SIMREADY_GRASP_PLAN_ANALYTIC_SCHEMA_VERSION: Final[
    Literal["content-agent-workflows.simready-grasp-plan.v2"]
] = "content-agent-workflows.simready-grasp-plan.v2"
SIMREADY_GRASP_PLAN_COMPOSED_SCHEMA_VERSION: Final[
    Literal["content-agent-workflows.simready-grasp-plan.v3"]
] = "content-agent-workflows.simready-grasp-plan.v3"
SIMREADY_GRASP_PLAN_GENERATOR_IMPLEMENTATION: Final[
    Literal["content_agent_workflows.simready.grasp_plan_generator"]
] = "content_agent_workflows.simready.grasp_plan_generator"
SIMREADY_GRASP_PLAN_GENERATOR_VERSION: Final[Literal["1.0.0"]] = "1.0.0"
SIMREADY_GRASP_PLAN_ANALYTIC_GENERATOR_VERSION: Final[Literal["1.1.0"]] = "1.1.0"
SIMREADY_GRASP_PLAN_COMPOSED_GENERATOR_VERSION: Final[Literal["1.2.0"]] = "1.2.0"

_FLOAT32_MAX = 3.4028234663852886e38
_FLOAT32_MIN_POSITIVE = 1.401298464324817e-45
_StrictNonEmptyString = Annotated[
    str,
    StringConstraints(strict=True, min_length=1),
]
_ReadableStrictString = Annotated[
    str,
    StringConstraints(strict=True, strip_whitespace=True, min_length=1),
]
_Sha256 = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^[0-9a-f]{64}$"),
]
_FiniteFloat = Annotated[
    float,
    Field(
        strict=True,
        ge=-_FLOAT32_MAX,
        le=_FLOAT32_MAX,
        allow_inf_nan=False,
    ),
]
_PositiveFiniteFloat = Annotated[
    float,
    Field(
        strict=True,
        ge=_FLOAT32_MIN_POSITIVE,
        le=_FLOAT32_MAX,
        allow_inf_nan=False,
    ),
]
_Point3 = Annotated[list[_FiniteFloat], Field(min_length=3, max_length=3)]
_NonNegativeStrictInt = Annotated[int, Field(strict=True, ge=0)]


class SimReadyGraspPlanProvenance(BaseModel):
    """Owner approval and readable evidence for grasp geometry."""

    model_config = ConfigDict(extra="forbid", strict=True)

    source: Literal["owner_approved_plan"]
    approved_by: _ReadableStrictString
    evidence: Annotated[list[_ReadableStrictString], Field(min_length=1)]


class SimReadyGraspPlanDependencyProof(BaseModel):
    """One regular local file bound into a machine geometry proof."""

    model_config = ConfigDict(extra="forbid", strict=True)

    role: Literal["source_asset", "dependency"]
    relative_path: _ReadableStrictString
    sha256: _Sha256


class SimReadyGraspPlanTriangleProof(BaseModel):
    """Exact composed Mesh triangle selected by the machine generator."""

    model_config = ConfigDict(extra="forbid", strict=True)

    mesh_prim_path: _StrictNonEmptyString
    face_index: _NonNegativeStrictInt
    triangle_index: _NonNegativeStrictInt
    point_indices: Annotated[
        list[_NonNegativeStrictInt],
        Field(min_length=3, max_length=3),
    ]
    mesh_local_points: Annotated[list[_Point3], Field(min_length=3, max_length=3)]
    default_prim_local_points: Annotated[
        list[_Point3],
        Field(min_length=3, max_length=3),
    ]


class SimReadyGraspPlanMachineProofChecks(BaseModel):
    """Fail-closed checks required before machine provenance can be emitted."""

    model_config = ConfigDict(extra="forbid", strict=True)

    dependency_closure_complete: Literal[True]
    source_bytes_preserved: Literal[True]
    no_instances_or_prototypes: Literal[True]
    static_geometry_and_transforms: Literal[True]
    topology_valid: Literal[True]
    transforms_finite_and_nonsingular: Literal[True]
    triangle_finite_and_nondegenerate: Literal[True]
    endpoints_strictly_inside_triangle: Literal[True]
    line_nonzero: Literal[True]
    width_explicit_positive_stage_units: Literal[True]


class SimReadyGraspPlanMachineProvenance(BaseModel):
    """Auditable geometry proof produced without owner approval or model calls."""

    model_config = ConfigDict(extra="forbid", strict=True)

    source: Literal["machine_geometry_proof"]
    implementation: Literal["content_agent_workflows.simready.grasp_plan_generator"]
    implementation_version: Literal["1.0.0"]
    source_asset_sha256: _Sha256
    dependency_bundle_sha256: _Sha256
    dependencies: Annotated[
        list[SimReadyGraspPlanDependencyProof],
        Field(min_length=1),
    ]
    selected_triangle: SimReadyGraspPlanTriangleProof
    barycentric_coordinates: Annotated[
        list[Annotated[list[_ReadableStrictString], Field(min_length=3, max_length=3)]],
        Field(min_length=2, max_length=2),
    ]
    line_points_default_prim_local: Annotated[
        list[_Point3],
        Field(min_length=2, max_length=2),
    ]
    width_stage_units: _PositiveFiniteFloat
    proof_checks: SimReadyGraspPlanMachineProofChecks

    @model_validator(mode="after")
    def validate_machine_proof(self) -> SimReadyGraspPlanMachineProvenance:
        """Require canonical dependency and fixed rational proof records."""

        expected_barycentrics = [
            ["1/2", "1/4", "1/4"],
            ["1/4", "1/2", "1/4"],
        ]
        if self.barycentric_coordinates != expected_barycentrics:
            raise ValueError(
                "machine_geometry_proof barycentric_coordinates must use the "
                "generator's fixed rational coordinates"
            )
        dependency_keys = [item.relative_path for item in self.dependencies]
        if dependency_keys != sorted(dependency_keys) or len(dependency_keys) != len(
            set(dependency_keys)
        ):
            raise ValueError(
                "machine_geometry_proof dependencies must be unique and sorted"
            )
        source_dependencies = [
            item for item in self.dependencies if item.role == "source_asset"
        ]
        if len(source_dependencies) != 1:
            raise ValueError(
                "machine_geometry_proof must bind exactly one source asset"
            )
        if source_dependencies[0].sha256 != self.source_asset_sha256:
            raise ValueError(
                "machine_geometry_proof source dependency SHA-256 does not match"
            )
        if (
            self.line_points_default_prim_local[0]
            == (self.line_points_default_prim_local[1])
        ):
            raise ValueError("machine_geometry_proof line must be nonzero")
        return self


class SimReadyGraspPlanAnalyticSurfaceProof(BaseModel):
    """Exact canonical triangle on one composed analytic USD primitive."""

    model_config = ConfigDict(extra="forbid", strict=True)

    primitive_type: Literal["Cube"]
    prim_path: _StrictNonEmptyString
    size: _PositiveFiniteFloat
    face_index: Annotated[int, Field(strict=True, ge=0, le=5)]
    triangle_index: Annotated[int, Field(strict=True, ge=0, le=1)]
    corner_indices: Annotated[
        list[_NonNegativeStrictInt],
        Field(min_length=3, max_length=3),
    ]
    primitive_local_points: Annotated[
        list[_Point3],
        Field(min_length=3, max_length=3),
    ]
    default_prim_local_points: Annotated[
        list[_Point3],
        Field(min_length=3, max_length=3),
    ]

    @model_validator(mode="after")
    def validate_cube_triangle(self) -> SimReadyGraspPlanAnalyticSurfaceProof:
        """Bind the proof to the generator's exact Cube face topology."""

        half = self.size / 2.0
        points = (
            (-half, -half, -half),
            (half, -half, -half),
            (half, half, -half),
            (-half, half, -half),
            (-half, -half, half),
            (half, -half, half),
            (half, half, half),
            (-half, half, half),
        )
        faces = (
            (0, 1, 2, 3),
            (4, 5, 6, 7),
            (0, 1, 5, 4),
            (1, 2, 6, 5),
            (2, 3, 7, 6),
            (3, 0, 4, 7),
        )
        face = faces[self.face_index]
        triangles = (
            (face[0], face[1], face[2]),
            (face[0], face[2], face[3]),
        )
        expected_indices = triangles[self.triangle_index]
        if self.corner_indices != list(expected_indices):
            raise ValueError(
                "machine_analytic_geometry_proof corner indices do not match "
                "the canonical Cube face triangle"
            )
        expected_points = [list(points[index]) for index in expected_indices]
        if self.primitive_local_points != expected_points:
            raise ValueError(
                "machine_analytic_geometry_proof primitive-local points do not "
                "match the composed Cube size"
            )
        return self


class SimReadyGraspPlanAnalyticProofChecks(BaseModel):
    """Fail-closed checks required for an analytic primitive proof."""

    model_config = ConfigDict(extra="forbid", strict=True)

    dependency_closure_complete: Literal[True]
    source_bytes_preserved: Literal[True]
    no_instances_or_prototypes: Literal[True]
    static_geometry_and_transforms: Literal[True]
    schema_parameters_valid: Literal[True]
    transforms_finite_and_nonsingular: Literal[True]
    triangle_finite_and_nondegenerate: Literal[True]
    endpoints_strictly_inside_triangle: Literal[True]
    line_nonzero: Literal[True]
    width_explicit_positive_stage_units: Literal[True]


class SimReadyGraspPlanAnalyticMachineProvenance(BaseModel):
    """Auditable analytic-primitive proof produced without model calls."""

    model_config = ConfigDict(extra="forbid", strict=True)

    source: Literal["machine_analytic_geometry_proof"]
    implementation: Literal["content_agent_workflows.simready.grasp_plan_generator"]
    implementation_version: Literal["1.1.0"]
    source_asset_sha256: _Sha256
    dependency_bundle_sha256: _Sha256
    dependencies: Annotated[
        list[SimReadyGraspPlanDependencyProof],
        Field(min_length=1),
    ]
    selected_surface: SimReadyGraspPlanAnalyticSurfaceProof
    barycentric_coordinates: Annotated[
        list[Annotated[list[_ReadableStrictString], Field(min_length=3, max_length=3)]],
        Field(min_length=2, max_length=2),
    ]
    line_points_default_prim_local: Annotated[
        list[_Point3],
        Field(min_length=2, max_length=2),
    ]
    width_stage_units: _PositiveFiniteFloat
    proof_checks: SimReadyGraspPlanAnalyticProofChecks

    @model_validator(mode="after")
    def validate_machine_proof(
        self,
    ) -> SimReadyGraspPlanAnalyticMachineProvenance:
        """Require canonical dependencies and fixed triangle coordinates."""

        expected_barycentrics = [
            ["1/2", "1/4", "1/4"],
            ["1/4", "1/2", "1/4"],
        ]
        if self.barycentric_coordinates != expected_barycentrics:
            raise ValueError(
                "machine_analytic_geometry_proof barycentric_coordinates must "
                "use the generator's fixed rational coordinates"
            )
        dependency_keys = [item.relative_path for item in self.dependencies]
        if dependency_keys != sorted(dependency_keys) or len(dependency_keys) != len(
            set(dependency_keys)
        ):
            raise ValueError(
                "machine_analytic_geometry_proof dependencies must be unique and sorted"
            )
        source_dependencies = [
            item for item in self.dependencies if item.role == "source_asset"
        ]
        if len(source_dependencies) != 1:
            raise ValueError(
                "machine_analytic_geometry_proof must bind exactly one source asset"
            )
        if source_dependencies[0].sha256 != self.source_asset_sha256:
            raise ValueError(
                "machine_analytic_geometry_proof source dependency SHA-256 "
                "does not match"
            )
        if (
            self.line_points_default_prim_local[0]
            == (self.line_points_default_prim_local[1])
        ):
            raise ValueError("machine_analytic_geometry_proof line must be nonzero")
        p0, p1, p2 = self.selected_surface.default_prim_local_points
        expected_line = [
            [(2.0 * p0[index] + p1[index] + p2[index]) / 4.0 for index in range(3)],
            [(p0[index] + 2.0 * p1[index] + p2[index]) / 4.0 for index in range(3)],
        ]
        if self.line_points_default_prim_local != expected_line:
            raise ValueError(
                "machine_analytic_geometry_proof line points do not match the "
                "selected surface triangle"
            )
        return self


class SimReadyGraspPlanComposedProofChecks(BaseModel):
    """Fail-closed checks for geometry read through composed instance proxies."""

    model_config = ConfigDict(extra="forbid", strict=True)

    dependency_closure_complete: Literal[True]
    source_bytes_preserved: Literal[True]
    composed_instance_proxies_resolved: Literal[True]
    no_point_instancers: Literal[True]
    static_geometry_and_transforms: Literal[True]
    surface_schema_and_topology_valid: Literal[True]
    transforms_finite_and_nonsingular: Literal[True]
    triangle_finite_and_nondegenerate: Literal[True]
    endpoints_strictly_inside_triangle: Literal[True]
    line_nonzero: Literal[True]
    width_explicit_positive_stage_units: Literal[True]


class SimReadyGraspPlanComposedMachineProvenance(BaseModel):
    """Auditable proof for a Mesh or Cube resolved through USD composition."""

    model_config = ConfigDict(extra="forbid", strict=True)

    source: Literal["machine_composed_geometry_proof"]
    implementation: Literal["content_agent_workflows.simready.grasp_plan_generator"]
    implementation_version: Literal["1.2.0"]
    source_asset_sha256: _Sha256
    dependency_bundle_sha256: _Sha256
    dependencies: Annotated[
        list[SimReadyGraspPlanDependencyProof],
        Field(min_length=1),
    ]
    selected_triangle: SimReadyGraspPlanTriangleProof | None = None
    selected_surface: SimReadyGraspPlanAnalyticSurfaceProof | None = None
    barycentric_coordinates: Annotated[
        list[Annotated[list[_ReadableStrictString], Field(min_length=3, max_length=3)]],
        Field(min_length=2, max_length=2),
    ]
    line_points_default_prim_local: Annotated[
        list[_Point3],
        Field(min_length=2, max_length=2),
    ]
    width_stage_units: _PositiveFiniteFloat
    proof_checks: SimReadyGraspPlanComposedProofChecks

    @model_validator(mode="after")
    def validate_machine_proof(
        self,
    ) -> SimReadyGraspPlanComposedMachineProvenance:
        """Require one exact surface, canonical dependencies, and fixed endpoints."""

        if (self.selected_triangle is None) == (self.selected_surface is None):
            raise ValueError(
                "machine_composed_geometry_proof requires exactly one selected "
                "Mesh triangle or analytic surface"
            )
        expected_barycentrics = [
            ["1/2", "1/4", "1/4"],
            ["1/4", "1/2", "1/4"],
        ]
        if self.barycentric_coordinates != expected_barycentrics:
            raise ValueError(
                "machine_composed_geometry_proof barycentric_coordinates must "
                "use the generator's fixed rational coordinates"
            )
        dependency_keys = [item.relative_path for item in self.dependencies]
        if dependency_keys != sorted(dependency_keys) or len(dependency_keys) != len(
            set(dependency_keys)
        ):
            raise ValueError(
                "machine_composed_geometry_proof dependencies must be unique and sorted"
            )
        source_dependencies = [
            item for item in self.dependencies if item.role == "source_asset"
        ]
        if len(source_dependencies) != 1:
            raise ValueError(
                "machine_composed_geometry_proof must bind exactly one source asset"
            )
        if source_dependencies[0].sha256 != self.source_asset_sha256:
            raise ValueError(
                "machine_composed_geometry_proof source dependency SHA-256 "
                "does not match"
            )
        if (
            self.line_points_default_prim_local[0]
            == self.line_points_default_prim_local[1]
        ):
            raise ValueError("machine_composed_geometry_proof line must be nonzero")
        return self


class SimReadyGraspLinePlan(BaseModel):
    """One evidence-backed local-coordinate grasp BasisCurves plan."""

    model_config = ConfigDict(extra="forbid", strict=True)

    prim_path: _StrictNonEmptyString
    coordinate_space: Literal["local"]
    points: Annotated[list[_Point3], Field(min_length=2)]
    widths: Annotated[list[_PositiveFiniteFloat], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_width_count(self) -> SimReadyGraspLinePlan:
        """Require unambiguous constant or per-point width interpolation."""

        if len(self.widths) not in {1, len(self.points)}:
            raise ValueError(
                "widths must contain one constant width or one width per point"
            )
        return self


class SimReadyGraspPlan(BaseModel):
    """Strict GSP.001 repair plan bound to exact source asset bytes."""

    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal[
        "content-agent-workflows.simready-grasp-plan.v1",
        "content-agent-workflows.simready-grasp-plan.v2",
        "content-agent-workflows.simready-grasp-plan.v3",
    ]
    source_asset_sha256: _Sha256
    default_prim_path: _StrictNonEmptyString
    provenance: Annotated[
        SimReadyGraspPlanProvenance
        | SimReadyGraspPlanMachineProvenance
        | SimReadyGraspPlanAnalyticMachineProvenance
        | SimReadyGraspPlanComposedMachineProvenance,
        Field(discriminator="source"),
    ]
    grasp_lines: Annotated[list[SimReadyGraspLinePlan], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_unique_prim_paths(self) -> SimReadyGraspPlan:
        """Reject ambiguous plans that author the same prim more than once."""

        paths = [line.prim_path for line in self.grasp_lines]
        if len(paths) != len(set(paths)):
            raise ValueError("grasp_lines contains duplicate prim_path values")
        if isinstance(self.provenance, SimReadyGraspPlanMachineProvenance):
            if self.schema_version != SIMREADY_GRASP_PLAN_SCHEMA_VERSION:
                raise ValueError(
                    "machine_geometry_proof requires the v1 grasp-plan schema"
                )
            if self.source_asset_sha256 != self.provenance.source_asset_sha256:
                raise ValueError(
                    "machine_geometry_proof source SHA-256 must match the plan"
                )
            if len(self.grasp_lines) != 1:
                raise ValueError(
                    "machine_geometry_proof plans must contain exactly one grasp line"
                )
            line = self.grasp_lines[0]
            if line.points != self.provenance.line_points_default_prim_local:
                raise ValueError(
                    "machine_geometry_proof line points must match the grasp line"
                )
            if line.widths != [self.provenance.width_stage_units]:
                raise ValueError(
                    "machine_geometry_proof width must match the grasp line"
                )
        if isinstance(self.provenance, SimReadyGraspPlanAnalyticMachineProvenance):
            if self.schema_version != SIMREADY_GRASP_PLAN_ANALYTIC_SCHEMA_VERSION:
                raise ValueError(
                    "machine_analytic_geometry_proof requires the v2 grasp-plan schema"
                )
            if self.source_asset_sha256 != self.provenance.source_asset_sha256:
                raise ValueError(
                    "machine_analytic_geometry_proof source SHA-256 must match the plan"
                )
            if len(self.grasp_lines) != 1:
                raise ValueError(
                    "machine_analytic_geometry_proof plans must contain exactly "
                    "one grasp line"
                )
            line = self.grasp_lines[0]
            if line.points != self.provenance.line_points_default_prim_local:
                raise ValueError(
                    "machine_analytic_geometry_proof line points must match the "
                    "grasp line"
                )
            if line.widths != [self.provenance.width_stage_units]:
                raise ValueError(
                    "machine_analytic_geometry_proof width must match the grasp line"
                )
        if isinstance(self.provenance, SimReadyGraspPlanComposedMachineProvenance):
            if self.schema_version != SIMREADY_GRASP_PLAN_COMPOSED_SCHEMA_VERSION:
                raise ValueError(
                    "machine_composed_geometry_proof requires the v3 grasp-plan schema"
                )
            if self.source_asset_sha256 != self.provenance.source_asset_sha256:
                raise ValueError(
                    "machine_composed_geometry_proof source SHA-256 must match the plan"
                )
            if len(self.grasp_lines) != 1:
                raise ValueError(
                    "machine_composed_geometry_proof plans must contain exactly "
                    "one grasp line"
                )
            line = self.grasp_lines[0]
            if line.points != self.provenance.line_points_default_prim_local:
                raise ValueError(
                    "machine_composed_geometry_proof line points must match the "
                    "grasp line"
                )
            if line.widths != [self.provenance.width_stage_units]:
                raise ValueError(
                    "machine_composed_geometry_proof width must match the grasp line"
                )
        return self


class SimReadyPreflightReport(BaseModel):
    """Dependency and runtime readiness for SimReady Foundation workflows."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SIMREADY_PREFLIGHT_SCHEMA_VERSION
    passed: bool
    status: str
    foundation_repo_url: str = DEFAULT_SIMREADY_FOUNDATION_REPO_URL
    foundation_ref: str = DEFAULT_SIMREADY_FOUNDATION_REF
    foundation_root: str | None = None
    foundation_commit: str | None = None
    foundation_spec_root: str | None = None
    managed_foundation_checkout: bool = False
    venv_path: str | None = None
    validator_executable: str | None = None
    install_command: list[str] = Field(default_factory=list)
    command: list[str] = Field(default_factory=list)
    available_profiles: list[str] = Field(default_factory=list)
    specs_ready: bool = False
    runtime_ready: bool = False
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class SimReadyRuntimeInfo(BaseModel):
    """Resolved SimReady Foundation paths and executable state."""

    model_config = ConfigDict(extra="forbid")

    foundation_repo_url: str = DEFAULT_SIMREADY_FOUNDATION_REPO_URL
    foundation_ref: str = DEFAULT_SIMREADY_FOUNDATION_REF
    foundation_root: str | None = None
    foundation_commit: str | None = None
    foundation_spec_root: str | None = None
    managed_foundation_checkout: bool = False
    venv_path: str | None = None
    validator_executable: str | None = None
    install_command: list[str] = Field(default_factory=list)
    available_profiles: list[str] = Field(default_factory=list)
    specs_ready: bool = False
    runtime_ready: bool = False
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Return whether both Foundation specs and runtime are ready."""

        return self.specs_ready and self.runtime_ready and not self.errors

    def to_preflight_report(self) -> SimReadyPreflightReport:
        """Convert runtime discovery into a durable preflight report."""

        return SimReadyPreflightReport(
            passed=self.passed,
            status="PASS" if self.passed else "BLOCKED",
            foundation_repo_url=self.foundation_repo_url,
            foundation_ref=self.foundation_ref,
            foundation_root=self.foundation_root,
            foundation_commit=self.foundation_commit,
            foundation_spec_root=self.foundation_spec_root,
            managed_foundation_checkout=self.managed_foundation_checkout,
            venv_path=self.venv_path,
            validator_executable=self.validator_executable,
            install_command=self.install_command,
            available_profiles=self.available_profiles,
            specs_ready=self.specs_ready,
            runtime_ready=self.runtime_ready,
            warnings=self.warnings,
            errors=self.errors,
        )


class SimReadyValidationInput(BaseModel):
    """Input for formal SimReady Foundation profile validation."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    asset_path: str
    profile: str = DEFAULT_SIMREADY_PROFILE
    profile_version: str = DEFAULT_SIMREADY_PROFILE_VERSION
    report_path: str | None = None
    foundation_root: str | None = None
    foundation_spec_root: str | None = None
    venv_path: str | None = None
    install_missing: bool = True
    update_foundation: bool = False
    timeout_s: float = 300.0
    stdout_log_path: str | None = None
    stderr_log_path: str | None = None


class SimReadyValidationReport(BaseModel):
    """Normalized formal SimReady Foundation profile validation report."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SIMREADY_VALIDATION_SCHEMA_VERSION
    asset_path: str
    validator_skill: str = "content-workflow-simready"
    validator_tool: str = "simready-validate"
    passed: bool
    status: str
    profile_name: str
    profile_version: str
    profile_target: str
    command: list[str] = Field(default_factory=list)
    foundation_root: str | None = None
    foundation_commit: str | None = None
    foundation_spec_root: str | None = None
    validator_executable: str | None = None
    available_profiles: list[str] = Field(default_factory=list)
    profile_results: Any = None
    feature_results: Any = None
    requirement_counts: dict[str, int] = Field(default_factory=dict)
    issue_counts: dict[str, int] = Field(default_factory=dict)
    issues: list[dict[str, Any]] = Field(default_factory=list)
    ignored_issues: list[dict[str, Any]] = Field(default_factory=list)
    asset_topology: dict[str, Any] = Field(default_factory=dict)
    validation_policy: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    needs_rerun: bool = False
    rerun_reasons: list[str] = Field(default_factory=list)
    stdout_log_path: str | None = None
    stderr_log_path: str | None = None
    report_path: str | None = None
    raw_report_path: str | None = None
    next_step: str = "complete"


class SimReadyConformanceInput(BaseModel):
    """Input for staged SimReady Foundation profile conformance routing."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    asset_path: str
    output_dir: str
    profile: str = DEFAULT_SIMREADY_PROFILE
    profile_version: str = DEFAULT_SIMREADY_PROFILE_VERSION
    report_path: str | None = None
    validation_report_path: str | None = None
    grasp_plan_path: str | None = None
    source_asset: str | None = None
    grasp_prim_path: str | None = None
    foundation_root: str | None = None
    foundation_spec_root: str | None = None
    expected_physics_inventory_sha256: _Sha256 | None = Field(
        default=None,
        description=(
            "Trusted Joint Agent physics-inventory fingerprint; mandatory when "
            "G3A.HYG.001 is routed."
        ),
    )
    repair_requirements: list[str] = Field(default_factory=list)
    force: bool = False


class SimReadyConformanceReport(BaseModel):
    """Normalized staged SimReady Foundation conformance routing report."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SIMREADY_CONFORMANCE_SCHEMA_VERSION
    input_usd_path: str
    output_usd_path: str
    output_dir: str
    profile: str
    profile_version: str
    foundation_root: str | None = None
    foundation_commit: str | None = None
    foundation_spec_root: str | None = None
    validation_report: str | None = None
    failed_requirements: list[str] = Field(default_factory=list)
    requirements_repaired: list[str] = Field(default_factory=list)
    requirements_blocked: list[str] = Field(default_factory=list)
    requirements_skipped: list[str] = Field(default_factory=list)
    steps: list[dict[str, Any]] = Field(default_factory=list)
    reports: dict[str, str] = Field(default_factory=dict)
    passed: bool
    status: str
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    report_path: str | None = None
    next_step: str = "simready-validate"
