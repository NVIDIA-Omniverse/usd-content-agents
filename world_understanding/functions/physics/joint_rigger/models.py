# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Versioned, app-independent contracts for structured USD joint authoring."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

INPUT_SCHEMA_VERSION: Literal["world-understanding-joint-rigger-input-v1"] = (
    "world-understanding-joint-rigger-input-v1"
)
INPUT_SCHEMA_VERSION_V2: Literal["world-understanding-joint-rigger-input-v2"] = (
    "world-understanding-joint-rigger-input-v2"
)
PLAN_SCHEMA_VERSION: Literal["world-understanding-joint-rigger-plan-v1"] = (
    "world-understanding-joint-rigger-plan-v1"
)
PLAN_SCHEMA_VERSION_V2: Literal["world-understanding-joint-rigger-plan-v2"] = (
    "world-understanding-joint-rigger-plan-v2"
)
DIAGNOSTICS_SCHEMA_VERSION: Literal[
    "world-understanding-joint-rigger-diagnostics-v1"
] = "world-understanding-joint-rigger-diagnostics-v1"
RESULT_SCHEMA_VERSION: Literal["world-understanding-joint-rigger-result-v1"] = (
    "world-understanding-joint-rigger-result-v1"
)

type JointType = Literal["revolute", "prismatic", "spherical"]
type RigidLinkAuthoring = Literal["existing", "aggregate"]
type FieldDisposition = Literal[
    "accepted", "ignored", "defaulted", "rejected", "unresolved"
]
type ProvenanceSource = Literal[
    "accepted_manifest",
    "authored_metadata",
    "authored_reference",
    "owner_approved_plan",
    "source_metadata",
    "template_default",
]
type ResultStatus = Literal[
    "succeeded",
    "no_joints_authored",
    "unavailable",
    "incompatible",
    "rejected",
    "failed",
]

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_TOPOLOGY_PROVENANCE = frozenset({"joint_type", "body0", "body1"})
_ARTIFACT_BACKED_PROVENANCE_SOURCES = frozenset(
    {
        "accepted_manifest",
        "authored_metadata",
        "authored_reference",
        "source_metadata",
    }
)
_AXIS_TOLERANCE = 1e-6


class JointRiggerContractError(ValueError):
    """A fail-closed contract error with a stable machine-readable code."""

    def __init__(
        self,
        code: str,
        detail: str,
        *,
        unresolved_dependency_paths: tuple[str, ...] = (),
    ) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail
        self.unresolved_dependency_paths = unresolved_dependency_paths


class _ContractModel(BaseModel):
    """Strict immutable base for canonical JSON contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ArtifactIdentityV1(_ContractModel):
    """Logical artifact identity without a required local filesystem path."""

    uri: str = Field(min_length=1)
    root_sha256: str
    dependency_bundle_sha256: str | None = None

    @field_validator("uri")
    @classmethod
    def _nonblank_uri(cls, value: str) -> str:
        return _nonblank(value, "uri")

    @field_validator("root_sha256", "dependency_bundle_sha256")
    @classmethod
    def _valid_sha256(cls, value: str | None) -> str | None:
        if value is not None and not _SHA256_RE.fullmatch(value):
            raise ValueError("must be a lowercase 64-character SHA-256 digest")
        return value


class FieldProvenanceV1(_ContractModel):
    """Evidence that justifies one accepted structured authoring fact."""

    source: ProvenanceSource
    artifact: ArtifactIdentityV1 | None = None
    prim_path: str | None = None
    properties: tuple[str, ...] = ()
    derivation: str | None = None
    evidence: str = Field(min_length=1)

    @field_validator("prim_path")
    @classmethod
    def _valid_optional_prim_path(cls, value: str | None) -> str | None:
        if value is not None:
            _require_prim_path(value, "prim_path")
        return value

    @field_validator("properties")
    @classmethod
    def _canonical_properties(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(sorted({_nonblank(item, "property") for item in value}))
        return normalized

    @field_validator("derivation")
    @classmethod
    def _nonblank_derivation(cls, value: str | None) -> str | None:
        return _optional_nonblank(value, "derivation")

    @field_validator("evidence")
    @classmethod
    def _nonblank_evidence(cls, value: str) -> str:
        return _nonblank(value, "evidence")

    @model_validator(mode="after")
    def _validate_source_locator(self) -> FieldProvenanceV1:
        locator_parts = (
            self.artifact is not None,
            self.prim_path is not None,
            bool(self.properties),
        )
        if self.source not in _ARTIFACT_BACKED_PROVENANCE_SOURCES:
            if any(locator_parts) and not all(locator_parts):
                raise ValueError(
                    f"{self.source} provenance locator must provide artifact, "
                    "prim_path, and properties together"
                )
            return self
        if not locator_parts[0]:
            raise ValueError(f"{self.source} provenance requires an artifact identity")
        if not locator_parts[1]:
            raise ValueError(f"{self.source} provenance requires a prim_path")
        if not locator_parts[2]:
            raise ValueError(f"{self.source} provenance requires at least one property")
        return self


class JointTopologyV1(_ContractModel):
    """Required explicit topology for one supported authored joint."""

    joint_id: str = Field(min_length=1)
    joint_type: JointType
    body0: str
    body1: str
    axis_stage: tuple[float, float, float] | None = None
    field_provenance: Mapping[str, FieldProvenanceV1]

    @field_validator("joint_id")
    @classmethod
    def _nonblank_id(cls, value: str) -> str:
        return _nonblank(value, "joint_id")

    @field_validator("body0", "body1")
    @classmethod
    def _valid_body_path(cls, value: str) -> str:
        _require_prim_path(value, "body endpoint")
        return value

    @field_validator("axis_stage")
    @classmethod
    def _normalized_axis(
        cls, value: tuple[float, float, float] | None
    ) -> tuple[float, float, float] | None:
        if value is not None:
            _require_finite_vector(value, "axis_stage")
            norm = math.sqrt(sum(component * component for component in value))
            if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=_AXIS_TOLERANCE):
                raise ValueError("axis_stage must be a normalized vector")
        return value

    @field_validator("field_provenance")
    @classmethod
    def _immutable_field_provenance(
        cls,
        value: Mapping[str, FieldProvenanceV1],
    ) -> Mapping[str, FieldProvenanceV1]:
        return MappingProxyType(dict(value))

    @field_serializer("field_provenance")
    def _serialize_field_provenance(
        self,
        value: Mapping[str, FieldProvenanceV1],
    ) -> dict[str, FieldProvenanceV1]:
        return {key: value[key] for key in sorted(value)}

    def __deepcopy__(
        self,
        memo: dict[int, Any] | None = None,
    ) -> JointTopologyV1:
        """Copy the frozen contract without asking ``mappingproxy`` to pickle.

        Preserve Pydantic's field-set and private/extra state exactly; rebuilding
        through validation would incorrectly mark omitted defaults as explicit.
        """

        active_memo: dict[int, Any] = {} if memo is None else memo
        copied = type(self).__new__(type(self))
        active_memo[id(self)] = copied
        values = dict(self.__dict__)
        provenance = values.pop("field_provenance")
        copied_values = copy.deepcopy(values, memo=active_memo)
        copied_values["field_provenance"] = MappingProxyType(
            copy.deepcopy(dict(provenance), memo=active_memo)
        )
        object.__setattr__(copied, "__dict__", copied_values)
        object.__setattr__(
            copied,
            "__pydantic_extra__",
            copy.deepcopy(self.__pydantic_extra__, memo=active_memo),
        )
        object.__setattr__(
            copied,
            "__pydantic_fields_set__",
            self.__pydantic_fields_set__.copy(),
        )
        object.__setattr__(
            copied,
            "__pydantic_private__",
            copy.deepcopy(
                getattr(self, "__pydantic_private__", None),
                memo=active_memo,
            ),
        )
        return copied

    @model_validator(mode="after")
    def _validate_topology(self) -> JointTopologyV1:
        if self.body0 == self.body1:
            raise ValueError("same_body_endpoints: body0 and body1 must differ")
        expected = set(_REQUIRED_TOPOLOGY_PROVENANCE)
        if self.joint_type in {"revolute", "prismatic"}:
            if self.axis_stage is None:
                raise ValueError(
                    "axis_unresolved: revolute and prismatic joints require axis_stage"
                )
            expected.add("axis_stage")
        elif self.axis_stage is not None:
            raise ValueError("spherical joints must not carry axis_stage")
        actual = set(self.field_provenance)
        if actual != expected:
            raise ValueError(
                "field_provenance keys must exactly match required topology fields; "
                f"expected {sorted(expected)}, got {sorted(actual)}"
            )
        return self


class JointLimitV1(_ContractModel):
    """Optional source-backed scalar travel limits."""

    lower: float | None = None
    upper: float | None = None
    unit: Literal["degrees", "meters"]
    provenance: FieldProvenanceV1

    @model_validator(mode="after")
    def _validate_limits(self) -> JointLimitV1:
        if self.lower is None and self.upper is None:
            raise ValueError("at least one authored limit is required")
        for label, value in (("lower", self.lower), ("upper", self.upper)):
            if value is not None:
                _require_finite(value, label)
        if (
            self.lower is not None
            and self.upper is not None
            and self.lower > self.upper
        ):
            raise ValueError("invalid_limit_range: lower must not exceed upper")
        return self


class JointAnchorV1(_ContractModel):
    """Optional source-backed anchor expressed in the stage frame."""

    position_stage: tuple[float, float, float]
    provenance: FieldProvenanceV1

    @field_validator("position_stage")
    @classmethod
    def _finite_position(
        cls, value: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        _require_finite_vector(value, "position_stage")
        return value


class JointFrictionV1(_ContractModel):
    """Optional evidence-backed PhysX joint-friction coefficient."""

    coefficient: float
    provenance: FieldProvenanceV1

    @field_validator("coefficient")
    @classmethod
    def _finite_nonnegative_coefficient(cls, value: float) -> float:
        _require_finite(value, "coefficient")
        if value < 0.0:
            raise ValueError("coefficient must be nonnegative")
        return value


class JointDriveV1(_ContractModel):
    """Optional complete, source-backed drive opinion."""

    drive_type: Literal["force", "acceleration"]
    stiffness: float
    damping: float
    max_force: float
    target_position: float
    target_velocity: float
    max_joint_velocity: float | None = None
    provenance: FieldProvenanceV1

    @model_validator(mode="after")
    def _validate_drive(self) -> JointDriveV1:
        for label in (
            "stiffness",
            "damping",
            "max_force",
            "target_position",
            "target_velocity",
            "max_joint_velocity",
        ):
            value = getattr(self, label)
            if value is None:
                continue
            _require_finite(value, label)
            if label in {"stiffness", "damping", "max_force", "max_joint_velocity"}:
                if value < 0.0:
                    raise ValueError(f"{label} must be nonnegative")
        return self


class JointStateV1(_ContractModel):
    """Optional source-backed default joint state."""

    position: float
    velocity: float
    provenance: FieldProvenanceV1

    @model_validator(mode="after")
    def _validate_state(self) -> JointStateV1:
        _require_finite(self.position, "position")
        _require_finite(self.velocity, "velocity")
        return self


class JointMimicV1(_ContractModel):
    """Optional source-backed mimic relationship."""

    reference_joint_id: str = Field(min_length=1)
    gearing: float
    offset: float
    natural_frequency: float
    damping_ratio: float
    provenance: FieldProvenanceV1

    @model_validator(mode="after")
    def _validate_mimic(self) -> JointMimicV1:
        _nonblank(self.reference_joint_id, "reference_joint_id")
        for label in ("gearing", "offset", "natural_frequency", "damping_ratio"):
            _require_finite(getattr(self, label), label)
        if math.isclose(self.gearing, 0.0, rel_tol=0.0, abs_tol=0.0):
            raise ValueError("gearing must be nonzero")
        if self.natural_frequency <= 0.0:
            raise ValueError("natural_frequency must be positive")
        if self.damping_ratio < 0.0:
            raise ValueError("damping_ratio must be nonnegative")
        return self


class MassPropertiesV1(_ContractModel):
    """Optional SI mass and inertia facts."""

    mass_kg: float
    center_of_mass_m: tuple[float, float, float] | None = None
    diagonal_inertia_kg_m2: tuple[float, float, float]
    principal_axes: tuple[float, float, float, float] | None = None
    provenance: FieldProvenanceV1

    @model_validator(mode="after")
    def _validate_mass(self) -> MassPropertiesV1:
        _require_positive(self.mass_kg, "mass_kg")
        if self.center_of_mass_m is not None:
            _require_finite_vector(self.center_of_mass_m, "center_of_mass_m")
        _require_finite_vector(
            self.diagonal_inertia_kg_m2,
            "diagonal_inertia_kg_m2",
        )
        for value in self.diagonal_inertia_kg_m2:
            if value <= 0.0:
                raise ValueError("diagonal inertia values must be positive")
        first, second, third = sorted(self.diagonal_inertia_kg_m2)
        if first + second < third and not math.isclose(
            first + second, third, rel_tol=1e-9, abs_tol=1e-12
        ):
            raise ValueError("diagonal inertia must satisfy the inertia triangle")
        if self.principal_axes is not None:
            _require_finite_vector(self.principal_axes, "principal_axes")
            norm = math.sqrt(sum(value * value for value in self.principal_axes))
            if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=_AXIS_TOLERANCE):
                raise ValueError("principal_axes must be a normalized quaternion")
        return self


class ColliderPlanV1(_ContractModel):
    """One exact source-backed supported collision-owner target."""

    prim_path: str
    mesh_collision_api: Literal[True] | None = Field(
        default=None,
        description=(
            "True records applied PhysicsMeshCollisionAPI presence; omit the "
            "field when the API is absent. An authored mesh_approximation already "
            "implies API presence, so redundant True is normalized away. False is "
            "intentionally invalid."
        ),
    )
    mesh_approximation: (
        Literal[
            "none",
            "convexHull",
            "convexDecomposition",
            "sdf",
        ]
        | None
    ) = None
    provenance: FieldProvenanceV1

    @field_validator("mesh_collision_api", mode="before")
    @classmethod
    def _strict_mesh_api_presence(cls, value: Any) -> Literal[True] | None:
        """Reject values that compare equal to ``True`` but are not booleans."""

        if value is True or value is None:
            return value
        raise ValueError("mesh_collision_api must be exactly true or null")

    @model_validator(mode="after")
    def _canonical_mesh_api_presence(self) -> ColliderPlanV1:
        """Collapse the redundant API-plus-approximation representation."""

        if self.mesh_collision_api is True and self.mesh_approximation is not None:
            object.__setattr__(self, "mesh_collision_api", None)
        return self

    @field_validator("prim_path")
    @classmethod
    def _valid_path(cls, value: str) -> str:
        _require_prim_path(value, "collider prim_path")
        return value

    @property
    def has_mesh_collision_api(self) -> bool:
        """Resolve explicit v1 presence and the legacy approximation encoding."""

        return self.mesh_collision_api is True or self.mesh_approximation is not None


class RigidBodyPlanV1(_ContractModel):
    """Optional body physics facts for one exact prim."""

    prim_path: str
    mass: MassPropertiesV1 | None = None
    colliders: tuple[ColliderPlanV1, ...] = ()
    provenance: FieldProvenanceV1

    @field_validator("prim_path")
    @classmethod
    def _valid_path(cls, value: str) -> str:
        _require_prim_path(value, "rigid body prim_path")
        return value

    @field_validator("colliders")
    @classmethod
    def _canonical_colliders(
        cls, value: tuple[ColliderPlanV1, ...]
    ) -> tuple[ColliderPlanV1, ...]:
        paths = [item.prim_path for item in value]
        if len(paths) != len(set(paths)):
            raise ValueError("collider prim_path values must be unique")
        return tuple(sorted(value, key=lambda item: item.prim_path))

    @model_validator(mode="after")
    def _validate_collider_subtree(self) -> RigidBodyPlanV1:
        for collider in self.colliders:
            if not _is_same_or_descendant_prim_path(
                collider.prim_path,
                self.prim_path,
            ):
                raise ValueError(
                    "collider prim_path must equal its rigid body prim_path or be "
                    f"a descendant in that body's subtree: collider="
                    f"{collider.prim_path}, body={self.prim_path}"
                )
        return self


class ArticulationRootPlanV1(_ContractModel):
    """Optional explicit articulation-root designation."""

    prim_path: str
    provenance: FieldProvenanceV1

    @field_validator("prim_path")
    @classmethod
    def _valid_path(cls, value: str) -> str:
        _require_prim_path(value, "articulation root prim_path")
        return value


class JointPlanV1(_ContractModel):
    """One joint's required topology and optional source-backed facts."""

    topology: JointTopologyV1
    limit: JointLimitV1 | None = None
    anchor: JointAnchorV1 | None = None
    joint_friction: JointFrictionV1 | None = None
    drive: JointDriveV1 | None = None
    state: JointStateV1 | None = None
    mimic: JointMimicV1 | None = None

    @model_validator(mode="after")
    def _validate_joint_facts(self) -> JointPlanV1:
        if self.topology.joint_type == "spherical":
            if self.limit is not None:
                raise ValueError("scalar spherical limits are unsupported in v1")
            scalar_controls = tuple(
                field
                for field in ("joint_friction", "drive", "state", "mimic")
                if getattr(self, field) is not None
            )
            if scalar_controls:
                raise ValueError(
                    "scalar spherical control facts are unsupported in v1: "
                    f"{', '.join(scalar_controls)}"
                )
        if self.limit is not None:
            expected_unit = (
                "degrees" if self.topology.joint_type == "revolute" else "meters"
            )
            if self.limit.unit != expected_unit:
                raise ValueError(
                    f"{self.topology.joint_type} limits must use {expected_unit}"
                )
        if self.drive is not None and self.mimic is not None:
            raise ValueError("a joint cannot carry both drive and mimic opinions")
        if self.joint_friction is not None and self.mimic is not None:
            raise ValueError("joint friction with a mimic opinion is unsupported in v1")
        return self


class JointRiggerPlanV1(_ContractModel):
    """Deterministic authoring plan shared by apps and backends."""

    schema_version: Literal["world-understanding-joint-rigger-plan-v1"]
    joints: tuple[JointPlanV1, ...]
    rigid_bodies: tuple[RigidBodyPlanV1, ...] = ()
    articulation_root: ArticulationRootPlanV1 | None = None

    @field_validator("joints")
    @classmethod
    def _canonical_joints(
        cls, value: tuple[JointPlanV1, ...]
    ) -> tuple[JointPlanV1, ...]:
        identifiers = [item.topology.joint_id for item in value]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("joint_id values must be unique")
        return tuple(sorted(value, key=lambda item: item.topology.joint_id))

    @field_validator("rigid_bodies")
    @classmethod
    def _canonical_bodies(
        cls, value: tuple[RigidBodyPlanV1, ...]
    ) -> tuple[RigidBodyPlanV1, ...]:
        paths = [item.prim_path for item in value]
        if len(paths) != len(set(paths)):
            raise ValueError("rigid body prim_path values must be unique")
        collider_paths = [
            collider.prim_path for body in value for collider in body.colliders
        ]
        if len(collider_paths) != len(set(collider_paths)):
            raise ValueError("a collider prim_path may belong to only one body")
        for body in value:
            deeper_body_paths = tuple(
                path
                for path in paths
                if path != body.prim_path
                and _is_same_or_descendant_prim_path(path, body.prim_path)
            )
            for collider in body.colliders:
                owning_descendants = tuple(
                    path
                    for path in deeper_body_paths
                    if _is_same_or_descendant_prim_path(collider.prim_path, path)
                )
                if owning_descendants:
                    nearest = max(
                        owning_descendants,
                        key=lambda path: len(path.split("/")),
                    )
                    raise ValueError(
                        f"collider {collider.prim_path} must belong to its nearest "
                        f"planned rigid body {nearest}, not {body.prim_path}"
                    )
        return tuple(sorted(value, key=lambda item: item.prim_path))

    @model_validator(mode="after")
    def _validate_cross_references(self) -> JointRiggerPlanV1:
        joints_by_id = {item.topology.joint_id: item for item in self.joints}
        mimic_references: dict[str, str] = {}
        for item in self.joints:
            mimic = item.mimic
            if mimic is None:
                continue
            if mimic.reference_joint_id == item.topology.joint_id:
                raise ValueError("a mimic joint cannot reference itself")
            reference = joints_by_id.get(mimic.reference_joint_id)
            if reference is None:
                raise ValueError(
                    "mimic reference_joint_id must name another plan joint"
                )
            if reference.topology.joint_type == "spherical":
                raise ValueError(
                    "mimic reference_joint_id cannot name a spherical joint "
                    "without an explicit referenced degree of freedom"
                )
            mimic_references[item.topology.joint_id] = mimic.reference_joint_id
        cycle = _find_mimic_reference_cycle(mimic_references)
        if cycle is not None:
            raise ValueError(
                "mimic_reference_cycle: mimic references form a cycle: "
                f"{' -> '.join(cycle)}"
            )
        if self.articulation_root is not None:
            root_path = self.articulation_root.prim_path
            associated_paths = {item.prim_path for item in self.rigid_bodies}
            associated_paths.update(
                endpoint
                for joint in self.joints
                for endpoint in (joint.topology.body0, joint.topology.body1)
            )
            if not any(
                _is_same_or_descendant_prim_path(path, root_path)
                for path in associated_paths
            ):
                raise ValueError(
                    "articulation root must name a planned joint body endpoint or "
                    "rigid body, or one of their ancestors"
                )
        return self


class JointRiggerPlanV2(_ContractModel):
    """Deterministic multi-articulation plan for V2 owned authoring."""

    schema_version: Literal["world-understanding-joint-rigger-plan-v2"]
    joints: tuple[JointPlanV1, ...]
    rigid_bodies: tuple[RigidBodyPlanV1, ...] = ()
    articulation_roots: tuple[ArticulationRootPlanV1, ...]

    @field_validator("joints")
    @classmethod
    def _canonical_joints(
        cls, value: tuple[JointPlanV1, ...]
    ) -> tuple[JointPlanV1, ...]:
        identifiers = [item.topology.joint_id for item in value]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("joint_id values must be unique")
        return tuple(sorted(value, key=lambda item: item.topology.joint_id))

    @field_validator("rigid_bodies")
    @classmethod
    def _canonical_bodies(
        cls, value: tuple[RigidBodyPlanV1, ...]
    ) -> tuple[RigidBodyPlanV1, ...]:
        paths = [item.prim_path for item in value]
        if len(paths) != len(set(paths)):
            raise ValueError("rigid body prim_path values must be unique")
        collider_paths = [
            collider.prim_path for body in value for collider in body.colliders
        ]
        if len(collider_paths) != len(set(collider_paths)):
            raise ValueError("a collider prim_path may belong to only one body")
        for body in value:
            deeper_body_paths = tuple(
                path
                for path in paths
                if path != body.prim_path
                and _is_same_or_descendant_prim_path(path, body.prim_path)
            )
            for collider in body.colliders:
                owning_descendants = tuple(
                    path
                    for path in deeper_body_paths
                    if _is_same_or_descendant_prim_path(collider.prim_path, path)
                )
                if owning_descendants:
                    nearest = max(
                        owning_descendants,
                        key=lambda path: len(path.split("/")),
                    )
                    raise ValueError(
                        f"collider {collider.prim_path} must belong to its nearest "
                        f"planned rigid body {nearest}, not {body.prim_path}"
                    )
        return tuple(sorted(value, key=lambda item: item.prim_path))

    @field_validator("articulation_roots")
    @classmethod
    def _canonical_articulation_roots(
        cls,
        value: tuple[ArticulationRootPlanV1, ...],
    ) -> tuple[ArticulationRootPlanV1, ...]:
        paths = [root.prim_path for root in value]
        if len(paths) != len(set(paths)):
            raise ValueError("articulation root prim_path values must be unique")
        _require_nonoverlapping_paths(paths, "articulation root paths")
        return tuple(sorted(value, key=lambda root: root.prim_path))

    @model_validator(mode="after")
    def _validate_cross_references(self) -> JointRiggerPlanV2:
        _validate_mimic_references(self.joints)
        if not self.articulation_roots:
            if self.rigid_bodies or any(
                joint.state is not None
                or joint.drive is not None
                or joint.mimic is not None
                for joint in self.joints
            ):
                raise ValueError(
                    "empty articulation_roots is allowed only for a topology-only "
                    "internal projection"
                )
            return self

        graph_roots = _directed_joint_graph_roots(self.joints)
        planned_roots = {root.prim_path for root in self.articulation_roots}
        if planned_roots != graph_roots:
            raise ValueError(
                "articulation_roots must exactly match directed graph component "
                f"roots: missing={sorted(graph_roots - planned_roots)}, "
                f"extra={sorted(planned_roots - graph_roots)}"
            )
        return self


class LegacyComponentAssignmentV1(_ContractModel):
    """One explicit assignment for a component-name-only backend."""

    prim_path: str
    component_name: str = Field(min_length=1)
    source_field: Literal["role", "component_name"]

    @field_validator("prim_path")
    @classmethod
    def _valid_path(cls, value: str) -> str:
        _require_prim_path(value, "legacy assignment prim_path")
        return value

    @field_validator("component_name")
    @classmethod
    def _nonblank_name(cls, value: str) -> str:
        return _nonblank(value, "component_name")


class LegacyComponentNameCompatibilityV1(_ContractModel):
    """Explicit, diagnosed compatibility input for legacy external backends."""

    assignments: tuple[LegacyComponentAssignmentV1, ...] = Field(min_length=1)

    @field_validator("assignments")
    @classmethod
    def _canonical_assignments(
        cls, value: tuple[LegacyComponentAssignmentV1, ...]
    ) -> tuple[LegacyComponentAssignmentV1, ...]:
        paths = [item.prim_path for item in value]
        if len(paths) != len(set(paths)):
            raise ValueError("legacy assignment prim_path values must be unique")
        return tuple(sorted(value, key=lambda item: item.prim_path))


class JointRiggerInputV1(_ContractModel):
    """Stable input consumed by every owned or external backend."""

    schema_version: Literal["world-understanding-joint-rigger-input-v1"]
    source_asset: ArtifactIdentityV1
    plan: JointRiggerPlanV1
    legacy_component_names: LegacyComponentNameCompatibilityV1 | None = None
    conflict_policy: Literal["error"] = "error"


class RigidLinkMemberPlanV1(_ContractModel):
    """One exact source-to-authored member-root mapping for a rigid link."""

    source_prim_path: str
    authored_prim_path: str

    @field_validator("source_prim_path", "authored_prim_path")
    @classmethod
    def _valid_path(cls, value: str) -> str:
        _require_prim_path(value, "rigid-link member prim path")
        return value


class RigidLinkPlanV1(_ContractModel):
    """Explicit identity or aggregate authoring for one planned rigid body."""

    link_id: str = Field(min_length=1)
    body_authoring: RigidLinkAuthoring
    body_prim_path: str
    members: tuple[RigidLinkMemberPlanV1, ...] = Field(min_length=1)

    @field_validator("link_id")
    @classmethod
    def _nonblank_link_id(cls, value: str) -> str:
        return _nonblank(value, "link_id")

    @field_validator("body_prim_path")
    @classmethod
    def _valid_body_path(cls, value: str) -> str:
        _require_prim_path(value, "rigid-link body_prim_path")
        return value

    @field_validator("members")
    @classmethod
    def _canonical_members(
        cls,
        value: tuple[RigidLinkMemberPlanV1, ...],
    ) -> tuple[RigidLinkMemberPlanV1, ...]:
        source_paths = [member.source_prim_path for member in value]
        authored_paths = [member.authored_prim_path for member in value]
        if len(source_paths) != len(set(source_paths)):
            raise ValueError("rigid-link source member paths must be unique")
        if len(authored_paths) != len(set(authored_paths)):
            raise ValueError("rigid-link authored member paths must be unique")
        return tuple(sorted(value, key=lambda member: member.source_prim_path))

    @model_validator(mode="after")
    def _validate_authoring_shape(self) -> RigidLinkPlanV1:
        if self.body_authoring == "existing":
            if len(self.members) != 1:
                raise ValueError("existing rigid links require exactly one member")
            member = self.members[0]
            if (
                member.source_prim_path != self.body_prim_path
                or member.authored_prim_path != self.body_prim_path
            ):
                raise ValueError(
                    "existing rigid links require one identity member equal to "
                    "body_prim_path"
                )
            return self

        if len(self.members) < 2:
            raise ValueError("aggregate rigid links require at least two members")
        source_parent = _prim_parent_path(self.members[0].source_prim_path)
        if source_parent == "/":
            raise ValueError(
                "aggregate source members must live below one authored parent"
            )
        if any(
            _prim_parent_path(member.source_prim_path) != source_parent
            for member in self.members
        ):
            raise ValueError("aggregate source members must share one parent")
        if _prim_parent_path(self.body_prim_path) != source_parent:
            raise ValueError(
                "aggregate body_prim_path must be a sibling of its source members"
            )
        for member in self.members:
            if _paths_overlap(self.body_prim_path, member.source_prim_path):
                raise ValueError(
                    "aggregate body_prim_path must be disjoint from source members"
                )
            expected_authored_path = (
                f"{self.body_prim_path}/{_prim_name(member.source_prim_path)}"
            )
            if member.authored_prim_path != expected_authored_path:
                raise ValueError(
                    "aggregate authored member paths must be deterministic direct "
                    f"children of body_prim_path; expected {expected_authored_path}"
                )
        return self


class JointRiggerInputV2(JointRiggerInputV1):
    """V2 request with exact rigid-link membership and aggregate mappings."""

    # Runtime covariance is intentional: V2 remains facade-compatible with V1.
    schema_version: Literal[  # type: ignore[assignment]
        "world-understanding-joint-rigger-input-v2"
    ]
    plan: JointRiggerPlanV2  # type: ignore[assignment]
    rigid_links: tuple[RigidLinkPlanV1, ...] = Field(min_length=1)

    @field_validator("rigid_links")
    @classmethod
    def _canonical_rigid_links(
        cls,
        value: tuple[RigidLinkPlanV1, ...],
    ) -> tuple[RigidLinkPlanV1, ...]:
        link_ids = [link.link_id for link in value]
        body_paths = [link.body_prim_path for link in value]
        if len(link_ids) != len(set(link_ids)):
            raise ValueError("rigid-link link_id values must be unique")
        if len(body_paths) != len(set(body_paths)):
            raise ValueError("rigid-link body_prim_path values must be unique")

        source_paths = [
            member.source_prim_path for link in value for member in link.members
        ]
        authored_paths = [
            member.authored_prim_path for link in value for member in link.members
        ]
        if len(source_paths) != len(set(source_paths)):
            raise ValueError("source member paths may belong to only one rigid link")
        if len(authored_paths) != len(set(authored_paths)):
            raise ValueError("authored member paths may belong to only one rigid link")
        return tuple(sorted(value, key=lambda link: link.link_id))

    @model_validator(mode="after")
    def _validate_plan_body_coverage(self) -> JointRiggerInputV2:
        planned_body_paths = {body.prim_path for body in self.plan.rigid_bodies}
        planned_body_paths.update(
            endpoint
            for joint in self.plan.joints
            for endpoint in (joint.topology.body0, joint.topology.body1)
        )
        link_body_paths = {link.body_prim_path for link in self.rigid_links}
        if link_body_paths != planned_body_paths:
            raise ValueError(
                "rigid_links must exactly cover every planned body path: "
                f"missing={sorted(planned_body_paths - link_body_paths)}, "
                f"extra={sorted(link_body_paths - planned_body_paths)}"
            )
        self._validate_link_overlaps()
        return self

    def _validate_link_overlaps(self) -> None:
        links = tuple(self.rigid_links)
        for index, first in enumerate(links):
            for second in links[index + 1 :]:
                if not _paths_overlap(first.body_prim_path, second.body_prim_path):
                    continue
                if (
                    first.body_authoring == "aggregate"
                    or second.body_authoring == "aggregate"
                ):
                    raise ValueError(
                        "aggregate rigid-link body paths must be disjoint: "
                        f"{first.body_prim_path}, {second.body_prim_path}"
                    )
                _require_nested_existing_graph_ancestry(
                    first.body_prim_path,
                    second.body_prim_path,
                    self.plan.joints,
                )

        member_rows = tuple((link, member) for link in links for member in link.members)
        for attribute in ("source_prim_path", "authored_prim_path"):
            for index, (first_link, first_member) in enumerate(member_rows):
                first_path = getattr(first_member, attribute)
                for second_link, second_member in member_rows[index + 1 :]:
                    second_path = getattr(second_member, attribute)
                    if not _paths_overlap(first_path, second_path):
                        continue
                    if (
                        first_link.body_authoring == "aggregate"
                        or second_link.body_authoring == "aggregate"
                    ):
                        raise ValueError(
                            "aggregate rigid-link member paths must be disjoint: "
                            f"{first_path}, {second_path}"
                        )
                    _require_nested_existing_graph_ancestry(
                        first_link.body_prim_path,
                        second_link.body_prim_path,
                        self.plan.joints,
                    )

        for aggregate in (link for link in links if link.body_authoring == "aggregate"):
            for link in links:
                if link is aggregate:
                    continue
                candidate_paths = (link.body_prim_path,) + tuple(
                    path
                    for member in link.members
                    for path in (member.source_prim_path, member.authored_prim_path)
                )
                for candidate in candidate_paths:
                    if _paths_overlap(aggregate.body_prim_path, candidate):
                        raise ValueError(
                            "aggregate body paths must be disjoint from every other "
                            "rigid-link path: "
                            f"body={aggregate.body_prim_path}, path={candidate}"
                        )
            for member in (member for link in links for member in link.members):
                if _paths_overlap(
                    aggregate.body_prim_path,
                    member.source_prim_path,
                ):
                    raise ValueError(
                        "aggregate body paths must be disjoint from every source "
                        "member path: "
                        f"body={aggregate.body_prim_path}, "
                        f"source={member.source_prim_path}"
                    )


class FieldDecisionV1(_ContractModel):
    """Disposition and reason for one consumed or omitted contract field."""

    field: str = Field(min_length=1)
    disposition: FieldDisposition
    provenance: FieldProvenanceV1 | None = None
    reason_code: str | None = None
    detail: str = ""

    @field_validator("field")
    @classmethod
    def _nonblank_field(cls, value: str) -> str:
        return _nonblank(value, "field")

    @field_validator("reason_code")
    @classmethod
    def _nonblank_reason(cls, value: str | None) -> str | None:
        return _optional_nonblank(value, "reason_code")

    @model_validator(mode="after")
    def _validate_decision(self) -> FieldDecisionV1:
        if self.disposition == "accepted":
            if self.provenance is None:
                raise ValueError("accepted fields require provenance")
        elif self.reason_code is None:
            raise ValueError("non-accepted fields require reason_code")
        return self


class JointDiagnosticV1(_ContractModel):
    """Field decisions and stable reasons for one planned joint."""

    joint_id: str = Field(min_length=1)
    field_decisions: tuple[FieldDecisionV1, ...] = ()
    reason_codes: tuple[str, ...] = ()

    @field_validator("joint_id")
    @classmethod
    def _nonblank_id(cls, value: str) -> str:
        return _nonblank(value, "joint_id")

    @field_validator("field_decisions")
    @classmethod
    def _canonical_decisions(
        cls, value: tuple[FieldDecisionV1, ...]
    ) -> tuple[FieldDecisionV1, ...]:
        fields = [item.field for item in value]
        if len(fields) != len(set(fields)):
            raise ValueError("field decisions must be unique per joint")
        return tuple(sorted(value, key=lambda item: item.field))

    @field_validator("reason_codes")
    @classmethod
    def _canonical_reasons(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(sorted({_nonblank(item, "reason_code") for item in value}))

    @property
    def authored_prim_path(self) -> str | None:
        """Return the compatible v1 path decision without adding a wire field."""

        decision = next(
            (
                item
                for item in self.field_decisions
                if item.field == "usd.joint_prim_path"
                and item.disposition == "defaulted"
                and item.reason_code == "deterministic_joint_path"
            ),
            None,
        )
        if decision is None or not decision.detail:
            return None
        try:
            _require_prim_path(decision.detail, "usd.joint_prim_path detail")
        except ValueError:
            return None
        return decision.detail


class JointRiggerDiagnosticsV1(_ContractModel):
    """Machine-readable field dispositions for one backend invocation."""

    schema_version: Literal["world-understanding-joint-rigger-diagnostics-v1"]
    backend_name: str = Field(min_length=1)
    backend_version: str | None = None
    field_decisions: tuple[FieldDecisionV1, ...] = ()
    joint_diagnostics: tuple[JointDiagnosticV1, ...] = ()
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @field_validator("backend_name")
    @classmethod
    def _nonblank_backend(cls, value: str) -> str:
        return _nonblank(value, "backend_name")

    @field_validator("backend_version")
    @classmethod
    def _nonblank_version(cls, value: str | None) -> str | None:
        return _optional_nonblank(value, "backend_version")

    @field_validator("field_decisions")
    @classmethod
    def _canonical_decisions(
        cls, value: tuple[FieldDecisionV1, ...]
    ) -> tuple[FieldDecisionV1, ...]:
        fields = [item.field for item in value]
        if len(fields) != len(set(fields)):
            raise ValueError("top-level field decisions must be unique")
        return tuple(sorted(value, key=lambda item: item.field))

    @field_validator("joint_diagnostics")
    @classmethod
    def _canonical_joint_diagnostics(
        cls, value: tuple[JointDiagnosticV1, ...]
    ) -> tuple[JointDiagnosticV1, ...]:
        identifiers = [item.joint_id for item in value]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("joint diagnostics must have unique joint_id values")
        return tuple(sorted(value, key=lambda item: item.joint_id))

    @field_validator("errors", "warnings")
    @classmethod
    def _canonical_messages(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(sorted({_nonblank(item, "message") for item in value}))


class JointRiggerResultV1(_ContractModel):
    """Stable result and exact identities for one authoring invocation."""

    schema_version: Literal["world-understanding-joint-rigger-result-v1"]
    status: ResultStatus
    input_sha256: str = Field(
        description="SHA-256 of the canonical Joint Rigger input JSON payload."
    )
    plan_sha256: str = Field(
        description="SHA-256 of the canonical Joint Rigger plan JSON payload."
    )
    output_artifact: ArtifactIdentityV1 | None = None
    diagnostics: JointRiggerDiagnosticsV1

    @field_validator("input_sha256", "plan_sha256")
    @classmethod
    def _valid_sha256(cls, value: str) -> str:
        if not _SHA256_RE.fullmatch(value):
            raise ValueError("must be a lowercase 64-character SHA-256 digest")
        return value

    @model_validator(mode="after")
    def _validate_output_identity(self) -> JointRiggerResultV1:
        if self.status == "succeeded" and self.output_artifact is None:
            raise ValueError("succeeded results require output_artifact")
        if self.status != "succeeded" and self.output_artifact is not None:
            raise ValueError("non-succeeded results must not claim output_artifact")
        if self.status == "succeeded" and self.diagnostics.errors:
            raise ValueError("succeeded results must not contain diagnostics errors")
        return self


def canonical_json(value: BaseModel | Mapping[str, Any]) -> str:
    """Serialize a contract or mapping as deterministic compact JSON."""

    payload: Any
    if isinstance(value, BaseModel):
        payload = value.model_dump(mode="json", exclude_none=True)
    else:
        payload = dict(value)
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def canonical_sha256(value: BaseModel | Mapping[str, Any]) -> str:
    """Return the SHA-256 of :func:`canonical_json`."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _find_mimic_reference_cycle(
    references: Mapping[str, str],
) -> tuple[str, ...] | None:
    """Return one deterministic closed cycle from a single-reference graph."""

    resolved: set[str] = set()
    for start in sorted(references):
        path: list[str] = []
        positions: dict[str, int] = {}
        current = start
        while (
            current in references
            and current not in positions
            and current not in resolved
        ):
            positions[current] = len(path)
            path.append(current)
            current = references[current]
        if current in positions:
            cycle = path[positions[current] :]
            first = min(cycle)
            first_index = cycle.index(first)
            canonical = cycle[first_index:] + cycle[:first_index]
            return (*canonical, canonical[0])
        resolved.update(path)
    return None


def _nonblank(value: str, label: str) -> str:
    if not value.strip():
        raise ValueError(f"{label} must not be blank")
    return value


def _optional_nonblank(value: str | None, label: str) -> str | None:
    if value is not None:
        _nonblank(value, label)
    return value


def _require_prim_path(value: str, label: str) -> None:
    if not _is_absolute_prim_path(value):
        raise ValueError(
            f"{label} must be a valid absolute non-root USD prim path without "
            "property or variant selections"
        )


def _is_absolute_prim_path(value: str) -> bool:
    """Match ``Sdf.Path`` prim semantics with an optional-runtime fallback."""

    try:
        from pxr import Sdf
    except ImportError:
        return _is_portable_absolute_prim_path(value)

    validation = Sdf.Path.IsValidPathString(value)
    is_valid = validation[0] if isinstance(validation, tuple) else validation
    if not is_valid:
        return False
    path = Sdf.Path(value)
    return bool(
        str(path) == value
        and path.IsAbsolutePath()
        and path.IsPrimPath()
        and not path.ContainsPrimVariantSelection()
    )


def _is_portable_absolute_prim_path(value: str) -> bool:
    """Strict fallback for contract-only installs without OpenUSD bindings."""

    if not value.startswith("/") or value == "/":
        return False
    return all(segment.isidentifier() for segment in value[1:].split("/"))


def _is_same_or_descendant_prim_path(value: str, ancestor: str) -> bool:
    """Use USD path components, never textual prefixes, for subtree ownership."""

    try:
        from pxr import Sdf
    except ImportError:
        return value == ancestor or value.startswith(f"{ancestor}/")

    return bool(Sdf.Path(value).HasPrefix(Sdf.Path(ancestor)))


def _paths_overlap(first: str, second: str) -> bool:
    return _is_same_or_descendant_prim_path(
        first, second
    ) or _is_same_or_descendant_prim_path(second, first)


def _validate_mimic_references(joints: tuple[JointPlanV1, ...]) -> None:
    joints_by_id = {item.topology.joint_id: item for item in joints}
    mimic_references: dict[str, str] = {}
    for item in joints:
        mimic = item.mimic
        if mimic is None:
            continue
        if mimic.reference_joint_id == item.topology.joint_id:
            raise ValueError("a mimic joint cannot reference itself")
        reference = joints_by_id.get(mimic.reference_joint_id)
        if reference is None:
            raise ValueError("mimic reference_joint_id must name another plan joint")
        if reference.topology.joint_type == "spherical":
            raise ValueError(
                "mimic reference_joint_id cannot name a spherical joint without "
                "an explicit referenced degree of freedom"
            )
        mimic_references[item.topology.joint_id] = mimic.reference_joint_id
    cycle = _find_mimic_reference_cycle(mimic_references)
    if cycle is not None:
        raise ValueError(
            "mimic_reference_cycle: mimic references form a cycle: "
            f"{' -> '.join(cycle)}"
        )


def _directed_joint_graph_roots(joints: tuple[JointPlanV1, ...]) -> set[str]:
    body_paths = {
        endpoint
        for joint in joints
        for endpoint in (joint.topology.body0, joint.topology.body1)
    }
    incoming = dict.fromkeys(body_paths, 0)
    adjacency: dict[str, list[str]] = {path: [] for path in body_paths}
    for joint in joints:
        body0 = joint.topology.body0
        body1 = joint.topology.body1
        incoming[body1] += 1
        if incoming[body1] > 1:
            raise ValueError(f"body has multiple incoming joints: {body1}")
        adjacency[body0].append(body1)

    roots = {path for path, count in incoming.items() if count == 0}
    visited: set[str] = set()
    pending = sorted(roots, reverse=True)
    while pending:
        path = pending.pop()
        if path in visited:
            continue
        visited.add(path)
        pending.extend(sorted(adjacency[path], reverse=True))
    if visited != body_paths:
        raise ValueError(
            "joint graph contains a directed cycle without a component root: "
            f"{sorted(body_paths - visited)}"
        )
    return roots


def _require_nested_existing_graph_ancestry(
    first: str,
    second: str,
    joints: tuple[JointPlanV1, ...],
) -> None:
    if _is_same_or_descendant_prim_path(second, first):
        ancestor, descendant = first, second
    elif _is_same_or_descendant_prim_path(first, second):
        ancestor, descendant = second, first
    else:  # pragma: no cover - caller checks overlap
        return

    adjacency: dict[str, set[str]] = {}
    for joint in joints:
        adjacency.setdefault(joint.topology.body0, set()).add(joint.topology.body1)
    visited: set[str] = set()
    pending = [ancestor]
    while pending:
        path = pending.pop()
        if path == descendant:
            return
        if path in visited:
            continue
        visited.add(path)
        pending.extend(sorted(adjacency.get(path, ()), reverse=True))
    raise ValueError(
        "nested existing rigid links require matching transitive joint ancestry: "
        f"namespace ancestor={ancestor}, namespace descendant={descendant}"
    )


def _require_nonoverlapping_paths(paths: list[str], label: str) -> None:
    ordered = sorted(paths)
    for index, first in enumerate(ordered):
        for second in ordered[index + 1 :]:
            if _paths_overlap(first, second):
                raise ValueError(f"{label} must not overlap: {first}, {second}")


def _prim_parent_path(value: str) -> str:
    parent, _, _ = value.rpartition("/")
    return parent or "/"


def _prim_name(value: str) -> str:
    return value.rsplit("/", maxsplit=1)[-1]


def _require_finite(value: float, label: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")


def _require_positive(value: float, label: str) -> None:
    _require_finite(value, label)
    if value <= 0.0:
        raise ValueError(f"{label} must be positive")


def _require_finite_vector(value: tuple[float, ...], label: str) -> None:
    for index, component in enumerate(value):
        _require_finite(component, f"{label}[{index}]")


__all__ = [
    "DIAGNOSTICS_SCHEMA_VERSION",
    "INPUT_SCHEMA_VERSION",
    "INPUT_SCHEMA_VERSION_V2",
    "PLAN_SCHEMA_VERSION",
    "PLAN_SCHEMA_VERSION_V2",
    "RESULT_SCHEMA_VERSION",
    "ArticulationRootPlanV1",
    "ArtifactIdentityV1",
    "ColliderPlanV1",
    "FieldDecisionV1",
    "FieldDisposition",
    "FieldProvenanceV1",
    "JointAnchorV1",
    "JointDiagnosticV1",
    "JointDriveV1",
    "JointFrictionV1",
    "JointLimitV1",
    "JointMimicV1",
    "JointPlanV1",
    "JointRiggerContractError",
    "JointRiggerDiagnosticsV1",
    "JointRiggerInputV1",
    "JointRiggerInputV2",
    "JointRiggerPlanV1",
    "JointRiggerPlanV2",
    "JointRiggerResultV1",
    "JointStateV1",
    "JointTopologyV1",
    "JointType",
    "LegacyComponentAssignmentV1",
    "LegacyComponentNameCompatibilityV1",
    "MassPropertiesV1",
    "ProvenanceSource",
    "RigidLinkAuthoring",
    "RigidLinkMemberPlanV1",
    "RigidLinkPlanV1",
    "RigidBodyPlanV1",
    "ResultStatus",
    "canonical_json",
    "canonical_sha256",
]
