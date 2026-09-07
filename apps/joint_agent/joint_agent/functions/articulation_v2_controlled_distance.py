# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Strict axial controlled-distance overlays for articulation-v2.

The frozen articulation-v2 wire contract represents distance as an ordered
interval and intentionally owns no controls.  This sibling contract adds one
versioned release shape without changing those bytes: a prismatic joint whose
positive local-X coordinate is the attachment-frame distance, with a
source-backed linear position drive and joint state.

The frozen v1 overlay measures positive state from the body0 attachment toward
the body1 attachment.  The v2 successor follows the Isaac/PhysX prismatic state
convention instead: positive state translates the body1 attachment toward the
body0 attachment, so ``point0 - point1 = state * axis``.  V1 remains parseable
and byte-stable; callers must explicitly select v2 for the corrected body order.

Both versions are deliberately narrower than an omnidirectional
``DistanceJoint``.  Both
attachment frames must establish the same complete orthonormal basis, the
interval stays strictly positive, and the prismatic schema locks the other five
degrees of freedom.  Under those invariants the signed X coordinate and the
Euclidean frame-origin distance are the same value without a latent relative
twist.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Literal, NoReturn, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticCustomError, PydanticSerializationError
from world_understanding.functions.physics.joint_rigger import ArtifactIdentityV1

from joint_agent.functions.articulation_contract import LinkRecordV1
from joint_agent.functions.articulation_contract_v2 import (
    ArticulationContractV2,
    DistanceConstraintV2,
    FieldEvidenceV2,
    JointRecordV2,
    canonical_articulation_v2_sha256,
    parse_articulation_contract_v2,
)

CONTROLLED_DISTANCE_SCHEMA_VERSION: Literal[
    "joint-agent-articulation-v2-controlled-distance-v1"
] = "joint-agent-articulation-v2-controlled-distance-v1"
CONTROLLED_DISTANCE_SCHEMA_VERSION_V2: Literal[
    "joint-agent-articulation-v2-controlled-distance-v2"
] = "joint-agent-articulation-v2-controlled-distance-v2"

type Vector3 = tuple[float, float, float]

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_NORMALIZATION_TOLERANCE = 1e-6
_SOURCE_BACKED_PROVENANCE_SOURCES = frozenset(
    {
        "accepted_manifest",
        "authored_metadata",
        "authored_reference",
        "source_metadata",
    }
)
CONTROLLED_DISTANCE_FIELD_PROPERTIES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "representation": ("usd:schema:PrismaticJoint",),
        "body0_prim_path": ("physics:body0",),
        "body1_prim_path": ("physics:body1",),
        "axis_stage": (
            "physics:axis",
            "physics:localRot0",
            "physics:localRot1",
        ),
        "axial_interval.minimum_meters": ("physics:lowerLimit",),
        "axial_interval.maximum_meters": ("physics:upperLimit",),
        "drive.instance": ("usd:appliedSchema:PhysicsDriveAPI:linear",),
        "drive.drive_type": ("drive:linear:physics:type",),
        "drive.target_position_meters": ("drive:linear:physics:targetPosition",),
        "drive.target_velocity_meters_per_second": (
            "drive:linear:physics:targetVelocity",
        ),
        "drive.stiffness_newtons_per_meter": ("drive:linear:physics:stiffness",),
        "drive.damping_newton_seconds_per_meter": ("drive:linear:physics:damping",),
        "drive.maximum_force_newtons": ("drive:linear:physics:maxForce",),
        "state.instance": ("usd:appliedSchema:PhysicsJointStateAPI:linear",),
        "state.position_meters": ("state:linear:physics:position",),
        "state.velocity_meters_per_second": ("state:linear:physics:velocity",),
        "endpoint_frame.body0_world_transform": (
            "xformOp:translate",
            "xformOpOrder",
        ),
        "endpoint_frame.body1_world_transform": (
            "xformOp:translate",
            "xformOpOrder",
        ),
    }
)
CONTROLLED_DISTANCE_BASE_FIELD_PROPERTIES: Mapping[str, tuple[str, ...]] = (
    MappingProxyType(
        {
            "attachments.body0.orientation_wxyz": ("physics:localRot0",),
            "attachments.body0.position_meters": ("physics:localPos0",),
            "attachments.body1.orientation_wxyz": ("physics:localRot1",),
            "attachments.body1.position_meters": ("physics:localPos1",),
            "attachments.kind": (
                "physics:localPos0",
                "physics:localPos1",
                "physics:localRot0",
                "physics:localRot1",
            ),
            "body0_link": ("physics:body0",),
            "body1_link": ("physics:body1",),
            "constraint.kind": ("usd:schema:PrismaticJoint",),
            "constraint.minimum_meters": ("physics:lowerLimit",),
            "constraint.maximum_meters": ("physics:upperLimit",),
        }
    )
)


class ControlledDistanceError(ValueError):
    """Fail-closed controlled-distance error with a stable boundary code."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


class _ControlledDistanceModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class DistanceLinearDriveV1(_ControlledDistanceModel):
    """Source-backed force drive on the represented positive X distance."""

    classification: Literal["source_backed_position_drive"] = (
        "source_backed_position_drive"
    )
    instance: Literal["linear"] = "linear"
    drive_type: Literal["force"] = "force"
    target_position_meters: float
    target_velocity_meters_per_second: float = 0.0
    stiffness_newtons_per_meter: float
    damping_newton_seconds_per_meter: float
    maximum_force_newtons: float

    @field_validator(
        "target_position_meters",
        "stiffness_newtons_per_meter",
        "damping_newton_seconds_per_meter",
        "maximum_force_newtons",
    )
    @classmethod
    def _finite_values(cls, value: float) -> float:
        if not math.isfinite(value):
            raise PydanticCustomError(
                "controlled_distance_non_finite_drive_value",
                "controlled-distance drive values must be finite",
            )
        return 0.0 if value == 0.0 else value

    @field_validator("target_velocity_meters_per_second")
    @classmethod
    def _zero_target_velocity(cls, value: float) -> float:
        if not math.isfinite(value) or value != 0.0:
            raise PydanticCustomError(
                "controlled_distance_nonzero_target_velocity",
                "target_velocity_meters_per_second must be exactly zero",
            )
        return 0.0

    @model_validator(mode="after")
    def _active_bounded_force_drive(self) -> DistanceLinearDriveV1:
        if self.maximum_force_newtons <= 0.0:
            raise PydanticCustomError(
                "controlled_distance_non_positive_maximum_force",
                "maximum_force_newtons must be finite and positive",
            )
        if self.stiffness_newtons_per_meter <= 0.0:
            raise PydanticCustomError(
                "controlled_distance_non_positive_stiffness",
                "stiffness_newtons_per_meter must be finite and positive",
            )
        if self.damping_newton_seconds_per_meter <= 0.0:
            raise PydanticCustomError(
                "controlled_distance_non_positive_damping",
                "damping_newton_seconds_per_meter must be finite and positive",
            )
        return self


class DistanceLinearStateV1(_ControlledDistanceModel):
    """Source-backed initial state for the same linear degree of freedom."""

    instance: Literal["linear"] = "linear"
    position_meters: float
    velocity_meters_per_second: float = 0.0

    @field_validator("position_meters")
    @classmethod
    def _finite_position(cls, value: float) -> float:
        if not math.isfinite(value):
            raise PydanticCustomError(
                "controlled_distance_non_finite_state_value",
                "controlled-distance state position must be finite",
            )
        return 0.0 if value == 0.0 else value

    @field_validator("velocity_meters_per_second")
    @classmethod
    def _zero_state_velocity(cls, value: float) -> float:
        if not math.isfinite(value) or value != 0.0:
            raise PydanticCustomError(
                "controlled_distance_nonzero_state_velocity",
                "velocity_meters_per_second must be exactly zero",
            )
        return 0.0


class ControlledDistanceContractV1(_ControlledDistanceModel):
    """Identity-bound control overlay for one ready v2 distance record."""

    schema_version: Literal["joint-agent-articulation-v2-controlled-distance-v1"] = (
        CONTROLLED_DISTANCE_SCHEMA_VERSION
    )
    articulation_contract_sha256: str
    source_artifact: ArtifactIdentityV1
    source_joint_path: str = Field(min_length=1)
    body0_prim_path: str = Field(min_length=1)
    body1_prim_path: str = Field(min_length=1)
    joint_id: str = Field(min_length=1)
    representation: Literal["prismatic_axial_interval_drive_v1"] = (
        "prismatic_axial_interval_drive_v1"
    )
    axis_stage: Vector3
    axial_interval: DistanceConstraintV2
    drive: DistanceLinearDriveV1
    state: DistanceLinearStateV1
    field_evidence: tuple[FieldEvidenceV2, ...]
    static_qualified: Literal[False] = False
    dynamic_qualified: Literal[False] = False
    public_enabled: Literal[False] = False

    @field_validator("articulation_contract_sha256")
    @classmethod
    def _valid_contract_sha256(cls, value: str) -> str:
        if not _SHA256_RE.fullmatch(value):
            raise PydanticCustomError(
                "controlled_distance_invalid_contract_sha256",
                "articulation_contract_sha256 must be a lowercase SHA-256 digest",
            )
        return value

    @field_validator("source_joint_path", "body0_prim_path", "body1_prim_path")
    @classmethod
    def _valid_prim_path(cls, value: str) -> str:
        if value.strip() != value or not value.startswith("/") or value == "/":
            raise PydanticCustomError(
                "controlled_distance_invalid_prim_path",
                "controlled-distance prim paths must be non-root absolute paths",
            )
        return value

    @field_validator("joint_id")
    @classmethod
    def _nonblank_joint_id(cls, value: str) -> str:
        if value.strip() != value or not value:
            raise PydanticCustomError(
                "controlled_distance_invalid_joint_id",
                "joint_id must be nonblank and trimmed",
            )
        return value

    @field_validator("axis_stage")
    @classmethod
    def _normalized_axis(cls, value: Vector3) -> Vector3:
        if len(value) != 3 or any(not math.isfinite(item) for item in value):
            raise PydanticCustomError(
                "controlled_distance_invalid_axis",
                "axis_stage must contain three finite components",
            )
        norm = math.sqrt(sum(item * item for item in value))
        if not math.isclose(
            norm,
            1.0,
            rel_tol=0.0,
            abs_tol=_NORMALIZATION_TOLERANCE,
        ):
            raise PydanticCustomError(
                "controlled_distance_invalid_axis",
                "axis_stage must be a unit vector",
            )
        return tuple(0.0 if item == 0.0 else item for item in value)  # type: ignore[return-value]

    @field_validator("field_evidence")
    @classmethod
    def _canonical_field_evidence(
        cls,
        value: tuple[FieldEvidenceV2, ...],
    ) -> tuple[FieldEvidenceV2, ...]:
        fields = tuple(item.field for item in value)
        if len(fields) != len(set(fields)):
            raise PydanticCustomError(
                "controlled_distance_duplicate_field_evidence",
                "controlled-distance field evidence paths must be unique",
            )
        return tuple(sorted(value, key=lambda item: item.field))

    @model_validator(mode="after")
    def _complete_source_backed_control(self) -> ControlledDistanceContractV1:
        if self.body0_prim_path == self.body1_prim_path:
            raise PydanticCustomError(
                "controlled_distance_endpoint_identity_conflict",
                "controlled-distance endpoints must be distinct",
            )
        minimum = self.axial_interval.minimum_meters
        maximum = self.axial_interval.maximum_meters
        if minimum is None or maximum is None:
            raise PydanticCustomError(
                "controlled_distance_incomplete_axial_interval",
                "controlled distance requires a complete minimum/maximum pair",
            )
        if minimum <= 0.0 or minimum >= maximum:
            raise PydanticCustomError(
                "controlled_distance_invalid_axial_interval",
                "controlled distance requires 0 < minimum_meters < maximum_meters",
            )
        if not minimum < self.drive.target_position_meters < maximum:
            raise PydanticCustomError(
                "controlled_distance_target_outside_axial_interval",
                "drive target position must be strictly inside the axial interval",
            )
        if not minimum < self.state.position_meters < maximum:
            raise PydanticCustomError(
                "controlled_distance_state_outside_axial_interval",
                "state position must be strictly inside the axial interval",
            )

        actual = {item.field for item in self.field_evidence}
        expected = set(CONTROLLED_DISTANCE_FIELD_PROPERTIES)
        if actual != expected:
            missing = sorted(expected - actual)
            unknown = sorted(actual - expected)
            raise PydanticCustomError(
                "controlled_distance_incomplete_field_evidence",
                "controlled-distance evidence mismatch; missing={missing}; unknown={unknown}",
                {"missing": ", ".join(missing), "unknown": ", ".join(unknown)},
            )

        for item in self.field_evidence:
            provenance = item.provenance
            if provenance.source not in _SOURCE_BACKED_PROVENANCE_SOURCES:
                raise PydanticCustomError(
                    "controlled_distance_evidence_not_source_backed",
                    "controlled-distance evidence must be source-backed",
                )
            if provenance.artifact != self.source_artifact:
                raise PydanticCustomError(
                    "controlled_distance_evidence_artifact_conflict",
                    "controlled-distance evidence must use the exact source artifact",
                )
            expected_prim_path = (
                self.body0_prim_path
                if item.field == "endpoint_frame.body0_world_transform"
                else self.body1_prim_path
                if item.field == "endpoint_frame.body1_world_transform"
                else self.source_joint_path
            )
            if provenance.prim_path != expected_prim_path:
                raise PydanticCustomError(
                    "controlled_distance_evidence_prim_conflict",
                    "controlled-distance evidence uses the wrong exact source prim",
                )
            if (
                provenance.properties
                != CONTROLLED_DISTANCE_FIELD_PROPERTIES[item.field]
            ):
                raise PydanticCustomError(
                    "controlled_distance_evidence_property_conflict",
                    "controlled-distance evidence property locator conflicts with its field",
                )
        return self


class ControlledDistanceContractV2(ControlledDistanceContractV1):
    """Isaac-compatible controlled distance with explicit body1-to-body0 state."""

    schema_version: Literal[  # type: ignore[assignment]
        "joint-agent-articulation-v2-controlled-distance-v2"
    ]
    representation: Literal[  # type: ignore[assignment]
        "prismatic_axial_interval_drive_body1_to_body0_v2"
    ]


def bind_controlled_distance_contract_v1(
    articulation_contract: ArticulationContractV2,
    controlled_distance: ControlledDistanceContractV1,
) -> ControlledDistanceContractV1:
    """Bind the overlay to one exact ready, axial-source v2 distance record."""

    if type(articulation_contract) is not ArticulationContractV2:
        raise TypeError("articulation_contract must be an exact ArticulationContractV2")
    if type(controlled_distance) is not ControlledDistanceContractV1:
        raise TypeError(
            "controlled_distance must be an exact ControlledDistanceContractV1"
        )

    articulation = parse_articulation_contract_v2(
        articulation_contract.model_dump(mode="json", exclude_none=True)
    )
    controlled = parse_controlled_distance_contract_v1(
        controlled_distance.model_dump(mode="json", exclude_none=True)
    )
    return _bind_controlled_distance_contract(articulation, controlled)


def bind_controlled_distance_contract_v2(
    articulation_contract: ArticulationContractV2,
    controlled_distance: ControlledDistanceContractV2,
) -> ControlledDistanceContractV2:
    """Bind the v2 overlay to one exact ready, axial-source distance record."""

    if type(articulation_contract) is not ArticulationContractV2:
        raise TypeError("articulation_contract must be an exact ArticulationContractV2")
    if type(controlled_distance) is not ControlledDistanceContractV2:
        raise TypeError(
            "controlled_distance must be an exact ControlledDistanceContractV2"
        )

    articulation = parse_articulation_contract_v2(
        articulation_contract.model_dump(mode="json", exclude_none=True)
    )
    controlled = parse_controlled_distance_contract_v2(
        controlled_distance.model_dump(mode="json", exclude_none=True)
    )
    return cast(
        ControlledDistanceContractV2,
        _bind_controlled_distance_contract(articulation, controlled),
    )


def _bind_controlled_distance_contract(
    articulation: ArticulationContractV2,
    controlled: ControlledDistanceContractV1 | ControlledDistanceContractV2,
) -> ControlledDistanceContractV1 | ControlledDistanceContractV2:
    if (
        canonical_articulation_v2_sha256(articulation)
        != controlled.articulation_contract_sha256
    ):
        _fail(
            "controlled_distance_contract_identity_conflict",
            "controlled-distance overlay does not bind the exact articulation contract",
        )
    if controlled.source_artifact not in articulation.source_identities:
        _fail(
            "controlled_distance_source_artifact_undeclared",
            "controlled-distance source artifact is not declared by articulation-v2",
        )

    matches = tuple(
        record
        for record in articulation.records
        if isinstance(record, JointRecordV2) and record.joint_id == controlled.joint_id
    )
    if len(matches) != 1:
        _fail(
            "controlled_distance_joint_identity_conflict",
            "controlled-distance joint_id must select exactly one v2 joint",
        )
    joint = matches[0]
    if joint.review_status != "ready_for_rigger_input":
        _fail(
            "controlled_distance_joint_not_ready",
            "controlled-distance overlay requires a ready v2 joint",
        )
    if not isinstance(joint.constraint, DistanceConstraintV2):
        _fail(
            "controlled_distance_constraint_kind_conflict",
            "controlled-distance overlay requires a v2 distance constraint",
        )
    links = {
        record.link_id: record
        for record in articulation.records
        if isinstance(record, LinkRecordV1)
    }
    body0 = links.get(joint.body0_link)
    body1 = links.get(joint.body1_link)
    if (
        body0 is None
        or body1 is None
        or body0.body_prim_path != controlled.body0_prim_path
        or body1.body_prim_path != controlled.body1_prim_path
    ):
        _fail(
            "controlled_distance_endpoint_identity_conflict",
            "controlled-distance endpoints do not match the exact v2 link paths",
        )
    if (
        joint.constraint.minimum_meters != controlled.axial_interval.minimum_meters
        or joint.constraint.maximum_meters != controlled.axial_interval.maximum_meters
    ):
        _fail(
            "controlled_distance_axial_interval_conflict",
            "controlled-distance interval does not match articulation-v2",
        )

    evidence_by_field = {item.field: item for item in joint.field_evidence}
    for field, properties in CONTROLLED_DISTANCE_BASE_FIELD_PROPERTIES.items():
        item = evidence_by_field.get(field)
        if item is None:  # pragma: no cover - strict ready-v2 evidence invariant
            _fail(
                "controlled_distance_base_evidence_missing",
                f"articulation-v2 is missing controlled source evidence for {field}",
            )
        provenance = item.provenance
        if (
            provenance.source not in _SOURCE_BACKED_PROVENANCE_SOURCES
            or provenance.artifact != controlled.source_artifact
            or provenance.prim_path != controlled.source_joint_path
            or provenance.properties != properties
        ):
            _fail(
                "controlled_distance_base_evidence_conflict",
                f"articulation-v2 evidence for {field} is not the admitted axial source",
            )
    return controlled


def canonical_controlled_distance_json(
    value: ControlledDistanceContractV1 | Mapping[str, Any],
) -> str:
    """Validate and serialize one controlled-distance contract canonically."""

    try:
        payload = (
            value.model_dump(mode="json", exclude_none=True)
            if isinstance(value, ControlledDistanceContractV1)
            else value
        )
        contract = parse_controlled_distance_contract_v1(payload)
        return json.dumps(
            contract.model_dump(mode="json", exclude_none=True),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except ControlledDistanceError:
        raise
    except (PydanticSerializationError, TypeError, ValueError, OverflowError) as exc:
        raise ControlledDistanceError(
            "controlled_distance_serialization_failed",
            "validated controlled-distance data could not be serialized canonically",
        ) from exc


def canonical_controlled_distance_sha256(
    value: ControlledDistanceContractV1 | Mapping[str, Any],
) -> str:
    """Hash the canonical controlled-distance JSON representation."""

    return hashlib.sha256(
        canonical_controlled_distance_json(value).encode("utf-8")
    ).hexdigest()


def canonical_controlled_distance_v2_json(
    value: ControlledDistanceContractV2 | Mapping[str, Any],
) -> str:
    """Validate and serialize one v2 controlled-distance contract canonically."""

    try:
        payload = (
            value.model_dump(mode="json", exclude_none=True)
            if isinstance(value, ControlledDistanceContractV2)
            else value
        )
        contract = parse_controlled_distance_contract_v2(payload)
        return json.dumps(
            contract.model_dump(mode="json", exclude_none=True),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except ControlledDistanceError:
        raise
    except (PydanticSerializationError, TypeError, ValueError, OverflowError) as exc:
        raise ControlledDistanceError(
            "controlled_distance_serialization_failed",
            "validated controlled-distance data could not be serialized canonically",
        ) from exc


def canonical_controlled_distance_v2_sha256(
    value: ControlledDistanceContractV2 | Mapping[str, Any],
) -> str:
    """Hash the canonical v2 controlled-distance JSON representation."""

    return hashlib.sha256(
        canonical_controlled_distance_v2_json(value).encode("utf-8")
    ).hexdigest()


def parse_controlled_distance_contract_v1(
    payload: str | bytes | bytearray | Mapping[str, Any],
) -> ControlledDistanceContractV1:
    """Strictly parse v1 and collapse validator details to stable codes."""

    try:
        if not isinstance(payload, str | bytes | bytearray | Mapping):
            _fail(
                "controlled_distance_invalid_contract",
                "payload must be JSON text or an object mapping",
            )
        if isinstance(payload, str | bytes | bytearray):
            payload = json.loads(payload)
        return ControlledDistanceContractV1.model_validate(_python_wire_shape(payload))
    except (json.JSONDecodeError, UnicodeDecodeError, RecursionError) as exc:
        raise ControlledDistanceError(
            "controlled_distance_invalid_contract",
            "payload is not valid JSON",
        ) from exc
    except ValidationError as exc:
        first = exc.errors(include_url=False)[0]
        error_type = str(first["type"])
        detail = str(first["msg"])
        code = (
            error_type
            if error_type.startswith("controlled_distance_")
            else "controlled_distance_invalid_contract"
        )
        raise ControlledDistanceError(code, detail) from exc


def parse_controlled_distance_contract_v2(
    payload: str | bytes | bytearray | Mapping[str, Any],
) -> ControlledDistanceContractV2:
    """Strictly parse v2 and collapse validator details to stable codes."""

    try:
        if not isinstance(payload, str | bytes | bytearray | Mapping):
            _fail(
                "controlled_distance_invalid_contract",
                "payload must be JSON text or an object mapping",
            )
        if isinstance(payload, str | bytes | bytearray):
            payload = json.loads(payload)
        return ControlledDistanceContractV2.model_validate(_python_wire_shape(payload))
    except (json.JSONDecodeError, UnicodeDecodeError, RecursionError) as exc:
        raise ControlledDistanceError(
            "controlled_distance_invalid_contract",
            "payload is not valid JSON",
        ) from exc
    except ValidationError as exc:
        first = exc.errors(include_url=False)[0]
        error_type = str(first["type"])
        detail = str(first["msg"])
        code = (
            error_type
            if error_type.startswith("controlled_distance_")
            else "controlled_distance_invalid_contract"
        )
        raise ControlledDistanceError(code, detail) from exc


def _fail(code: str, detail: str) -> NoReturn:
    raise ControlledDistanceError(code, detail)


def _python_wire_shape(value: Any) -> Any:
    """Convert JSON arrays to immutable tuple shapes before strict validation."""

    if isinstance(value, Mapping):
        return {key: _python_wire_shape(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return tuple(_python_wire_shape(item) for item in value)
    return value


__all__ = [
    "CONTROLLED_DISTANCE_SCHEMA_VERSION",
    "CONTROLLED_DISTANCE_SCHEMA_VERSION_V2",
    "CONTROLLED_DISTANCE_BASE_FIELD_PROPERTIES",
    "CONTROLLED_DISTANCE_FIELD_PROPERTIES",
    "ControlledDistanceContractV1",
    "ControlledDistanceContractV2",
    "ControlledDistanceError",
    "DistanceLinearDriveV1",
    "DistanceLinearStateV1",
    "bind_controlled_distance_contract_v1",
    "bind_controlled_distance_contract_v2",
    "canonical_controlled_distance_json",
    "canonical_controlled_distance_sha256",
    "canonical_controlled_distance_v2_json",
    "canonical_controlled_distance_v2_sha256",
    "parse_controlled_distance_contract_v1",
    "parse_controlled_distance_contract_v2",
]
