# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Strict Joint Agent articulation-v2 wire contract.

This module is intentionally independent of the existing articulation-v1
implementation and of ``JointRiggerPlanV2``.  It defines data that a producer
may prove; it does not enable an authorer or a public capability.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticCustomError, PydanticSerializationError
from world_understanding.functions.physics.joint_rigger import (
    ArtifactIdentityV1,
    FieldProvenanceV1,
)

from joint_agent.functions.articulation_contract import (
    ArticulationContractV1,
    ContractDiagnosticV1,
    ContractSummaryV1,
    JointRecordV1,
    LinkRecordV1,
    PrimRecordV1,
)
from joint_agent.functions.articulation_types import ArticulationReviewStatus

ARTICULATION_CONTRACT_V2_SCHEMA_VERSION: Literal["joint-agent-articulation-v2"] = (
    "joint-agent-articulation-v2"
)
ARTICULATION_CONTRACT_V2_IMPLEMENTED_CAPABILITY_IDS = frozenset(
    {
        "articulation.full_attachment_frames",
        "distance.bounded_two_body_constraint",
        "fixed.explicit_two_body_constraint",
    }
)

type Vector3 = tuple[float, float, float]
type QuaternionWxyz = tuple[float, float, float, float]
type ReviewStatus = ArticulationReviewStatus
type ConstraintKind = Literal[
    "revolute",
    "prismatic",
    "spherical",
    "fixed",
    "distance",
]

_NORMALIZATION_TOLERANCE = 1e-6
_SOURCE_BACKED_PROVENANCE_SOURCES = frozenset(
    {
        "accepted_manifest",
        "authored_metadata",
        "authored_reference",
        "source_metadata",
    }
)
_CONVERSION_DIAGNOSTIC_PREFIX = "articulation_v2_from_v1_"
_RECORD_KIND_ORDER = {"prim": 0, "link": 1, "joint": 2}


class ArticulationContractV2Error(ValueError):
    """A fail-closed articulation-v2 error with a stable code."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


class _ContractModel(BaseModel):
    """Strict immutable base for the v2 wire contract."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class FieldEvidenceV2(_ContractModel):
    """Evidence for one fully-qualified v2 field path."""

    field: str = Field(min_length=1)
    provenance: FieldProvenanceV1

    @field_validator("field")
    @classmethod
    def _nonblank_field(cls, value: str) -> str:
        return _nonblank(value, "field")


class AttachmentFrameV2(_ContractModel):
    """One static joint frame expressed in an endpoint body's local space.

    ``orientation_wxyz`` is the active right-handed rotation from the canonical
    joint frame into the body-local frame.  The canonical motion axis is +X.
    """

    position_meters: Vector3
    orientation_wxyz: QuaternionWxyz
    sampling_mode: Literal["static"] = "static"

    @field_validator("position_meters")
    @classmethod
    def _finite_position(cls, value: Vector3) -> Vector3:
        finite = _finite_vector(
            value,
            code="articulation_v2_non_finite_position",
            label="attachment position_meters",
        )
        return (finite[0], finite[1], finite[2])

    @field_validator("orientation_wxyz")
    @classmethod
    def _canonical_orientation(cls, value: QuaternionWxyz) -> QuaternionWxyz:
        finite = _finite_vector(
            value,
            code="articulation_v2_non_finite_quaternion",
            label="attachment orientation_wxyz",
        )
        norm = math.sqrt(sum(component * component for component in finite))
        if not math.isclose(
            norm,
            1.0,
            rel_tol=0.0,
            abs_tol=_NORMALIZATION_TOLERANCE,
        ):
            raise PydanticCustomError(
                "articulation_v2_invalid_quaternion_norm",
                "attachment orientation_wxyz must be a unit quaternion",
            )
        canonical = tuple(
            0.0 if component == 0.0 else component for component in finite
        )
        for component in canonical:
            if component == 0.0:
                continue
            if component < 0.0:
                canonical = tuple(-item for item in canonical)
            break
        canonical = tuple(
            0.0 if component == 0.0 else component for component in canonical
        )
        return canonical  # type: ignore[return-value]


class ExplicitAttachmentFramesV2(_ContractModel):
    """Complete attachment frames for both directed joint endpoints."""

    kind: Literal["explicit"]
    body0: AttachmentFrameV2
    body1: AttachmentFrameV2


class Body1OriginAttachmentDefaultV2(_ContractModel):
    """Diagnosed v1 compatibility convention, never source-backed evidence."""

    kind: Literal["body1_origin_default"]


AttachmentFramesV2: TypeAlias = Annotated[  # noqa: UP040
    ExplicitAttachmentFramesV2 | Body1OriginAttachmentDefaultV2,
    Field(discriminator="kind"),
]


class RevoluteConstraintV2(_ContractModel):
    """One-axis angular constraint in degrees."""

    kind: Literal["revolute"]
    axis_stage: Vector3 | None = None
    limit_mode: Literal["bounded", "continuous", "unresolved"]
    lower_degrees: float | None = None
    upper_degrees: float | None = None

    @field_validator("axis_stage")
    @classmethod
    def _normalized_axis(cls, value: Vector3 | None) -> Vector3 | None:
        if value is None:
            return value
        return _normalized_axis(value)

    @field_validator("lower_degrees", "upper_degrees")
    @classmethod
    def _finite_limit(cls, value: float | None) -> float | None:
        return _finite_scalar(value, "articulation_v2_non_finite_limit")

    @model_validator(mode="after")
    def _validate_limit_shape(self) -> RevoluteConstraintV2:
        _validate_bounded_pair(
            mode=self.limit_mode,
            bounded_mode="bounded",
            lower=self.lower_degrees,
            upper=self.upper_degrees,
            label="revolute",
        )
        return self


class PrismaticConstraintV2(_ContractModel):
    """One-axis linear constraint in meters."""

    kind: Literal["prismatic"]
    axis_stage: Vector3 | None = None
    limit_mode: Literal["bounded", "unbounded", "unresolved"]
    lower_meters: float | None = None
    upper_meters: float | None = None

    @field_validator("axis_stage")
    @classmethod
    def _normalized_axis(cls, value: Vector3 | None) -> Vector3 | None:
        if value is None:
            return value
        return _normalized_axis(value)

    @field_validator("lower_meters", "upper_meters")
    @classmethod
    def _finite_limit(cls, value: float | None) -> float | None:
        return _finite_scalar(value, "articulation_v2_non_finite_limit")

    @model_validator(mode="after")
    def _validate_limit_shape(self) -> PrismaticConstraintV2:
        _validate_bounded_pair(
            mode=self.limit_mode,
            bounded_mode="bounded",
            lower=self.lower_meters,
            upper=self.upper_meters,
            label="prismatic",
        )
        return self


class SphericalConstraintV2(_ContractModel):
    """Passive spherical shape supported by current reference evidence.

    Per-axis angular limits remain outside v2 until a representative positive
    oracle fixes their semantics.
    """

    kind: Literal["spherical"]
    angular_limit_mode: Literal["free", "unresolved"]


class FixedConstraintV2(_ContractModel):
    """Rigid attachment with no independent degree of freedom."""

    kind: Literal["fixed"]


class DistanceConstraintV2(_ContractModel):
    """Distance interval between the two attachment-frame origins."""

    kind: Literal["distance"]
    minimum_meters: float | None = None
    maximum_meters: float | None = None

    @field_validator("minimum_meters", "maximum_meters")
    @classmethod
    def _finite_distance(cls, value: float | None) -> float | None:
        return _finite_scalar(value, "articulation_v2_non_finite_distance")

    @model_validator(mode="after")
    def _validate_bounds(self) -> DistanceConstraintV2:
        if self.minimum_meters is None and self.maximum_meters is None:
            raise PydanticCustomError(
                "articulation_v2_distance_bounds_missing",
                "distance constraint requires at least one bound",
            )
        if self.minimum_meters is not None and self.minimum_meters < 0.0:
            raise PydanticCustomError(
                "articulation_v2_distance_negative_minimum",
                "distance minimum_meters must be non-negative",
            )
        if self.maximum_meters is not None and self.maximum_meters < 0.0:
            raise PydanticCustomError(
                "articulation_v2_distance_negative_maximum",
                "distance maximum_meters must be non-negative",
            )
        if (
            self.minimum_meters is not None
            and self.maximum_meters is not None
            and self.minimum_meters > self.maximum_meters
        ):
            raise PydanticCustomError(
                "articulation_v2_distance_bounds_inverted",
                "distance minimum_meters must not exceed maximum_meters",
            )
        return self


JointConstraintV2: TypeAlias = Annotated[  # noqa: UP040
    RevoluteConstraintV2
    | PrismaticConstraintV2
    | SphericalConstraintV2
    | FixedConstraintV2
    | DistanceConstraintV2,
    Field(discriminator="kind"),
]


class JointRecordV2(_ContractModel):
    """One directed v2 joint with typed constraints and attachment frames."""

    kind: Literal["joint"]
    joint_id: str = Field(min_length=1)
    body0_link: str = Field(min_length=1)
    body1_link: str = Field(min_length=1)
    attachments: AttachmentFramesV2
    constraint: JointConstraintV2
    field_evidence: tuple[FieldEvidenceV2, ...]
    review_status: ReviewStatus

    @field_validator("joint_id", "body0_link", "body1_link")
    @classmethod
    def _nonblank_identifier(cls, value: str, info: Any) -> str:
        return _nonblank(value, info.field_name)

    @field_validator("field_evidence")
    @classmethod
    def _canonical_evidence(
        cls,
        value: tuple[FieldEvidenceV2, ...],
    ) -> tuple[FieldEvidenceV2, ...]:
        fields = [item.field for item in value]
        if len(fields) != len(set(fields)):
            raise PydanticCustomError(
                "articulation_v2_duplicate_field_evidence",
                "field_evidence paths must be unique",
            )
        return tuple(sorted(value, key=lambda item: item.field))

    @model_validator(mode="after")
    def _validate_joint(self) -> JointRecordV2:
        if self.body0_link == self.body1_link:
            raise PydanticCustomError(
                "articulation_v2_same_body_constraint",
                "body0_link and body1_link must reference distinct links",
            )
        if self.review_status == "ready_for_rigger_input" and not isinstance(
            self.attachments, ExplicitAttachmentFramesV2
        ):
            raise PydanticCustomError(
                "articulation_v2_ready_requires_explicit_frames",
                "ready v2 joints require complete explicit endpoint frames",
            )

        expected = _expected_joint_evidence_fields(self)
        actual = {item.field for item in self.field_evidence}
        unknown = sorted(actual - expected)
        if unknown:
            raise PydanticCustomError(
                "articulation_v2_unknown_evidence_field",
                "field_evidence contains unsupported paths: {paths}",
                {"paths": ", ".join(unknown)},
            )

        if self.review_status != "ready_for_rigger_input":
            return self
        if isinstance(
            self.constraint,
            RevoluteConstraintV2 | PrismaticConstraintV2,
        ):
            if self.constraint.axis_stage is None:
                raise PydanticCustomError(
                    "articulation_v2_ready_requires_axis",
                    "ready one-axis constraints require axis_stage",
                )
            if self.constraint.limit_mode == "unresolved":
                raise PydanticCustomError(
                    "articulation_v2_ready_has_unresolved_constraint",
                    "ready constraints may not use an unresolved limit mode",
                )
        if (
            isinstance(self.constraint, SphericalConstraintV2)
            and self.constraint.angular_limit_mode == "unresolved"
        ):
            raise PydanticCustomError(
                "articulation_v2_ready_has_unresolved_constraint",
                "ready spherical constraints may not use unresolved limits",
            )
        missing = sorted(expected - actual)
        if missing:
            raise PydanticCustomError(
                "articulation_v2_incomplete_ready_evidence",
                "ready joint field_evidence is incomplete: {paths}",
                {"paths": ", ".join(missing)},
            )
        fallback = sorted(
            item.field
            for item in self.field_evidence
            if not _is_source_backed(item.provenance)
        )
        if fallback:
            raise PydanticCustomError(
                "articulation_v2_ready_evidence_not_source_backed",
                "ready joint evidence must be source-backed: {paths}",
                {"paths": ", ".join(fallback)},
            )
        return self


ArticulationRecordV2: TypeAlias = Annotated[  # noqa: UP040
    PrimRecordV1 | LinkRecordV1 | JointRecordV2,
    Field(discriminator="kind"),
]


class ArticulationContractV2(_ContractModel):
    """Canonical provider-neutral articulation-v2 document."""

    schema_version: Literal["joint-agent-articulation-v2"]
    status: ReviewStatus
    articulation_roots: tuple[str, ...]
    source_identities: tuple[ArtifactIdentityV1, ...] = Field(min_length=1)
    records: tuple[ArticulationRecordV2, ...]
    diagnostics: tuple[ContractDiagnosticV1, ...] = ()
    summary: ContractSummaryV1

    @field_validator("source_identities")
    @classmethod
    def _canonical_source_identities(
        cls,
        value: tuple[ArtifactIdentityV1, ...],
    ) -> tuple[ArtifactIdentityV1, ...]:
        uris = [identity.uri for identity in value]
        if len(uris) != len(set(uris)):
            raise PydanticCustomError(
                "articulation_v2_duplicate_source_identity",
                "source identity uri values must be unique",
            )
        return tuple(
            sorted(
                value,
                key=lambda identity: (
                    identity.uri,
                    identity.root_sha256,
                    identity.dependency_bundle_sha256 or "",
                ),
            )
        )

    @field_validator("articulation_roots")
    @classmethod
    def _canonical_roots(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(_nonblank(item, "articulation root") for item in value)
        if len(normalized) != len(set(normalized)):
            raise PydanticCustomError(
                "articulation_v2_duplicate_root",
                "articulation root link ids must be unique",
            )
        return tuple(sorted(normalized))

    @field_validator("records")
    @classmethod
    def _canonical_records(
        cls,
        value: tuple[ArticulationRecordV2, ...],
    ) -> tuple[ArticulationRecordV2, ...]:
        return tuple(sorted(value, key=_record_sort_key))

    @field_validator("diagnostics")
    @classmethod
    def _canonical_diagnostics(
        cls,
        value: tuple[ContractDiagnosticV1, ...],
    ) -> tuple[ContractDiagnosticV1, ...]:
        keys = [
            (
                item.record_kind,
                item.record_id,
                item.field or "",
                item.code,
            )
            for item in value
        ]
        if len(keys) != len(set(keys)):
            raise PydanticCustomError(
                "articulation_v2_duplicate_diagnostic",
                "diagnostic target, field, and code tuples must be unique",
            )
        return tuple(
            sorted(
                value,
                key=lambda item: (
                    item.record_kind,
                    item.record_id,
                    item.field or "",
                    item.code,
                    item.detail,
                ),
            )
        )

    @model_validator(mode="after")
    def _validate_document(self) -> ArticulationContractV2:
        prims = tuple(
            record for record in self.records if isinstance(record, PrimRecordV1)
        )
        links = tuple(
            record for record in self.records if isinstance(record, LinkRecordV1)
        )
        joints = tuple(
            record for record in self.records if isinstance(record, JointRecordV2)
        )
        _require_unique(
            (record.prim_path for record in prims),
            "articulation_v2_duplicate_prim",
            "prim_path values",
        )
        _require_unique(
            (record.link_id for record in links),
            "articulation_v2_duplicate_link",
            "link_id values",
        )
        _require_unique(
            (record.body_prim_path for record in links),
            "articulation_v2_duplicate_body_path",
            "link body_prim_path values",
        )
        _require_unique(
            (record.joint_id for record in joints),
            "articulation_v2_duplicate_joint",
            "joint_id values",
        )
        links_by_id = {link.link_id: link for link in links}
        _validate_membership(prims, links, links_by_id)
        _validate_joint_graph(joints, links_by_id, self.articulation_roots)
        _validate_document_status_and_diagnostics(
            self.status,
            links,
            joints,
            self.diagnostics,
        )
        _validate_summary(self.summary, prims, links, joints, self.diagnostics)
        _validate_declared_evidence(
            set(self.source_identities),
            prims,
            links,
            joints,
        )
        return self


def canonical_articulation_v2_json(
    value: ArticulationContractV2 | Mapping[str, Any],
) -> str:
    """Validate and serialize a contract into its one canonical JSON form."""

    try:
        payload = (
            value.model_dump(mode="json", exclude_none=True)
            if isinstance(value, ArticulationContractV2)
            else value
        )
        contract = parse_articulation_contract_v2(payload)
        canonical_payload = _python_wire_shape(
            contract.model_dump(mode="json", exclude_none=True)
        )
        return json.dumps(
            canonical_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except ArticulationContractV2Error:
        raise
    except (PydanticSerializationError, TypeError, ValueError, OverflowError) as exc:
        raise ArticulationContractV2Error(
            "articulation_v2_serialization_failed",
            "validated articulation-v2 data could not be serialized canonically",
        ) from exc


def canonical_articulation_v2_sha256(
    value: ArticulationContractV2 | Mapping[str, Any],
) -> str:
    """Hash the canonical articulation-v2 JSON representation."""

    return hashlib.sha256(
        canonical_articulation_v2_json(value).encode("utf-8")
    ).hexdigest()


def parse_articulation_contract_v2(
    payload: str | bytes | bytearray | Mapping[str, Any],
) -> ArticulationContractV2:
    """Parse v2 and collapse validator details into stable boundary codes."""

    try:
        if isinstance(payload, str | bytes | bytearray):
            payload = json.loads(payload)
        return ArticulationContractV2.model_validate(_python_wire_shape(payload))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ArticulationContractV2Error(
            "articulation_v2_invalid_contract",
            "payload is not valid JSON",
        ) from exc
    except ValidationError as exc:
        code, detail = _stable_validation_error(exc)
        raise ArticulationContractV2Error(code, detail) from exc


def negotiate_articulation_contract_v2(
    accepted_versions: Sequence[str],
) -> Literal["joint-agent-articulation-v2"]:
    """Select v2 only when the peer explicitly accepts the exact version."""

    normalized = tuple(item.strip() for item in accepted_versions if item.strip())
    if ARTICULATION_CONTRACT_V2_SCHEMA_VERSION in normalized:
        return ARTICULATION_CONTRACT_V2_SCHEMA_VERSION
    raise ArticulationContractV2Error(
        "articulation_v2_no_common_version",
        "peer did not advertise joint-agent-articulation-v2",
    )


def convert_articulation_v1_to_v2(
    contract: ArticulationContractV1,
) -> ArticulationContractV2:
    """Preserve all represented v1 facts without promoting v1 defaults.

    V1 has no complete endpoint frames and an absent scalar limit is not proof
    of continuous/free behavior.  Every converted joint therefore remains
    review-required with explicit compatibility diagnostics.
    """

    contract = _strict_revalidate_v1_contract(contract)
    records: list[ArticulationRecordV2] = []
    diagnostics = [_convert_v1_diagnostic(item) for item in contract.diagnostics]
    for record in contract.records:
        if isinstance(record, PrimRecordV1 | LinkRecordV1):
            records.append(record)
            continue

        constraint = _convert_v1_constraint(record)
        evidence = _convert_v1_joint_evidence(record)
        converted_joint = JointRecordV2(
            kind="joint",
            joint_id=record.joint_id,
            body0_link=record.body0_link,
            body1_link=record.body1_link,
            attachments=Body1OriginAttachmentDefaultV2(kind="body1_origin_default"),
            constraint=constraint,
            field_evidence=evidence,
            review_status="review_required",
        )
        records.append(converted_joint)
        diagnostics.append(
            ContractDiagnosticV1(
                record_kind="joint",
                record_id=record.joint_id,
                code=f"{_CONVERSION_DIAGNOSTIC_PREFIX}body1_origin_default",
                severity="error",
                field="attachments.kind",
                detail=(
                    "articulation-v1 does not represent two source-backed endpoint "
                    "frames; the legacy body1-origin convention remains a diagnosed "
                    "default"
                ),
            )
        )
        if _constraint_is_unresolved(constraint):
            diagnostics.append(
                ContractDiagnosticV1(
                    record_kind="joint",
                    record_id=record.joint_id,
                    code=f"{_CONVERSION_DIAGNOSTIC_PREFIX}limit_unresolved",
                    severity="error",
                    field=_constraint_mode_field(constraint),
                    detail=(
                        "an absent v1 scalar limit does not prove continuous, "
                        "unbounded, or free v2 motion"
                    ),
                )
            )
        existing_diagnostic_fields = {
            item.field
            for item in diagnostics
            if item.record_kind == "joint"
            and item.record_id == record.joint_id
            and item.field is not None
        }
        for item in converted_joint.field_evidence:
            if (
                _is_source_backed(item.provenance)
                or item.field in existing_diagnostic_fields
                or item.field == "attachments.kind"
                or item.field == _constraint_mode_field(constraint)
                and _constraint_is_unresolved(constraint)
            ):
                continue
            diagnostics.append(
                ContractDiagnosticV1(
                    record_kind="joint",
                    record_id=record.joint_id,
                    code=f"{_CONVERSION_DIAGNOSTIC_PREFIX}fallback_preserved",
                    severity="error",
                    field=item.field,
                    detail=(
                        "v1 non-source-backed evidence remains a diagnosed "
                        "review fallback after conversion"
                    ),
                )
            )

    records_tuple = tuple(records)
    diagnostics_tuple = tuple(diagnostics)
    has_review = any(
        isinstance(record, LinkRecordV1 | JointRecordV2)
        and record.review_status == "review_required"
        for record in records_tuple
    )
    return ArticulationContractV2(
        schema_version=ARTICULATION_CONTRACT_V2_SCHEMA_VERSION,
        status="review_required" if has_review else "ready_for_rigger_input",
        articulation_roots=contract.articulation_roots,
        source_identities=contract.source_identities,
        records=records_tuple,
        diagnostics=diagnostics_tuple,
        summary=_summary(records_tuple, diagnostics_tuple),
    )


def downgrade_articulation_v2_to_v1(
    contract: ArticulationContractV2,
) -> ArticulationContractV1:
    """Fail closed because v2 frames and typed constraints have no v1 shape."""

    del contract
    raise ArticulationContractV2Error(
        "articulation_v2_downgrade_would_lose_facts",
        "articulation-v1 cannot represent v2 attachment frames and constraint kinds",
    )


def attachment_frame_from_basis_v2(
    *,
    position_meters: Sequence[float],
    basis_rows: Sequence[Sequence[float]],
) -> AttachmentFrameV2:
    """Build a frame from a finite right-handed orthonormal source basis.

    This producer-side adapter is the stable fail-closed boundary for singular
    or sheared source transforms.  It does not sample a USD stage.
    """

    if (
        len(position_meters) != 3
        or len(basis_rows) != 3
        or any(len(row) != 3 for row in basis_rows)
    ):
        raise ArticulationContractV2Error(
            "articulation_v2_invalid_frame_basis_shape",
            "position must have 3 values and basis_rows must be 3x3",
        )
    position = tuple(float(item) for item in position_meters)
    basis = tuple(tuple(float(item) for item in row) for row in basis_rows)
    if not all(math.isfinite(item) for row in basis for item in row):
        raise ArticulationContractV2Error(
            "articulation_v2_non_finite_frame_basis",
            "source frame basis must contain finite values",
        )
    determinant = _determinant3(basis)
    if math.isclose(determinant, 0.0, rel_tol=0.0, abs_tol=1e-12):
        raise ArticulationContractV2Error(
            "articulation_v2_singular_transform",
            "source frame basis is singular",
        )
    if determinant < 0.0:
        raise ArticulationContractV2Error(
            "articulation_v2_left_handed_transform",
            "source frame basis must be right-handed",
        )
    for row in basis:
        norm = math.sqrt(sum(item * item for item in row))
        if not math.isclose(
            norm,
            1.0,
            rel_tol=0.0,
            abs_tol=_NORMALIZATION_TOLERANCE,
        ):
            raise ArticulationContractV2Error(
                "articulation_v2_non_rigid_frame_basis",
                "source frame basis rows must be normalized",
            )
    for first, second in (
        (basis[0], basis[1]),
        (basis[0], basis[2]),
        (basis[1], basis[2]),
    ):
        dot = sum(left * right for left, right in zip(first, second, strict=True))
        if not math.isclose(
            dot,
            0.0,
            rel_tol=0.0,
            abs_tol=_NORMALIZATION_TOLERANCE,
        ):
            raise ArticulationContractV2Error(
                "articulation_v2_non_rigid_frame_basis",
                "source frame basis rows must be orthogonal",
            )
    return AttachmentFrameV2(
        position_meters=position,  # type: ignore[arg-type]
        orientation_wxyz=_quaternion_from_rotation_rows(basis),
        sampling_mode="static",
    )


def _convert_v1_constraint(record: JointRecordV1) -> JointConstraintV2:
    if record.motion_type == "revolute":
        if record.limit is None:
            return RevoluteConstraintV2(
                kind="revolute",
                axis_stage=record.axis_stage,
                limit_mode="unresolved",
            )
        return RevoluteConstraintV2(
            kind="revolute",
            axis_stage=record.axis_stage,
            limit_mode="bounded",
            lower_degrees=record.limit.lower,
            upper_degrees=record.limit.upper,
        )
    if record.motion_type == "prismatic":
        if record.limit is None:
            return PrismaticConstraintV2(
                kind="prismatic",
                axis_stage=record.axis_stage,
                limit_mode="unresolved",
            )
        return PrismaticConstraintV2(
            kind="prismatic",
            axis_stage=record.axis_stage,
            limit_mode="bounded",
            lower_meters=record.limit.lower,
            upper_meters=record.limit.upper,
        )
    return SphericalConstraintV2(
        kind="spherical",
        angular_limit_mode="unresolved",
    )


def _convert_v1_diagnostic(
    diagnostic: ContractDiagnosticV1,
) -> ContractDiagnosticV1:
    if diagnostic.record_kind != "joint" or diagnostic.field is None:
        return diagnostic
    renamed = {
        "motion_type": "constraint.kind",
        "axis_stage": "constraint.axis_stage",
        "limit": "constraint.limit_mode",
    }
    return diagnostic.model_copy(
        update={"field": renamed.get(diagnostic.field, diagnostic.field)}
    )


def _convert_v1_joint_evidence(
    record: JointRecordV1,
) -> tuple[FieldEvidenceV2, ...]:
    renamed = {
        "body0_link": "body0_link",
        "body1_link": "body1_link",
        "motion_type": "constraint.kind",
        "axis_stage": "constraint.axis_stage",
    }
    evidence = [
        FieldEvidenceV2(field=renamed[field], provenance=provenance)
        for field, provenance in record.field_evidence.items()
        if field in renamed
    ]
    evidence.append(
        FieldEvidenceV2(
            field="attachments.kind",
            provenance=FieldProvenanceV1(
                source="template_default",
                evidence=(
                    "articulation-v1 compatibility uses the diagnosed body1-origin "
                    "attachment convention"
                ),
            ),
        )
    )
    if isinstance(record.motion_type, str):
        mode_field = {
            "revolute": "constraint.limit_mode",
            "prismatic": "constraint.limit_mode",
            "spherical": "constraint.angular_limit_mode",
        }[record.motion_type]
        if record.limit is None:
            evidence.append(
                FieldEvidenceV2(
                    field=mode_field,
                    provenance=FieldProvenanceV1(
                        source="template_default",
                        evidence=(
                            "v1 limit absence is retained as unresolved v2 "
                            "constraint semantics"
                        ),
                    ),
                )
            )
        else:
            evidence.extend(
                (
                    FieldEvidenceV2(
                        field="constraint.limit_mode",
                        provenance=record.limit.provenance,
                    ),
                    FieldEvidenceV2(
                        field=(
                            "constraint.lower_degrees"
                            if record.motion_type == "revolute"
                            else "constraint.lower_meters"
                        ),
                        provenance=record.limit.provenance,
                    ),
                    FieldEvidenceV2(
                        field=(
                            "constraint.upper_degrees"
                            if record.motion_type == "revolute"
                            else "constraint.upper_meters"
                        ),
                        provenance=record.limit.provenance,
                    ),
                )
            )
    return tuple(evidence)


def _expected_joint_evidence_fields(record: JointRecordV2) -> set[str]:
    expected = {"body0_link", "body1_link", "attachments.kind", "constraint.kind"}
    if isinstance(record.attachments, ExplicitAttachmentFramesV2):
        expected.update(
            {
                "attachments.body0.position_meters",
                "attachments.body0.orientation_wxyz",
                "attachments.body1.position_meters",
                "attachments.body1.orientation_wxyz",
            }
        )
    constraint = record.constraint
    if isinstance(constraint, RevoluteConstraintV2):
        expected.add("constraint.limit_mode")
        if constraint.axis_stage is not None:
            expected.add("constraint.axis_stage")
        if constraint.limit_mode == "bounded":
            expected.update({"constraint.lower_degrees", "constraint.upper_degrees"})
    elif isinstance(constraint, PrismaticConstraintV2):
        expected.add("constraint.limit_mode")
        if constraint.axis_stage is not None:
            expected.add("constraint.axis_stage")
        if constraint.limit_mode == "bounded":
            expected.update({"constraint.lower_meters", "constraint.upper_meters"})
    elif isinstance(constraint, SphericalConstraintV2):
        expected.add("constraint.angular_limit_mode")
    elif isinstance(constraint, DistanceConstraintV2):
        if constraint.minimum_meters is not None:
            expected.add("constraint.minimum_meters")
        if constraint.maximum_meters is not None:
            expected.add("constraint.maximum_meters")
    return expected


def _validate_membership(
    prims: tuple[PrimRecordV1, ...],
    links: tuple[LinkRecordV1, ...],
    links_by_id: Mapping[str, LinkRecordV1],
) -> None:
    members: dict[str, list[str]] = {link.link_id: [] for link in links}
    for prim in prims:
        if prim.link_id not in links_by_id:
            raise PydanticCustomError(
                "articulation_v2_missing_member_link",
                "prim {path} references missing link {link}",
                {"path": prim.prim_path, "link": prim.link_id},
            )
        members[prim.link_id].append(prim.prim_path)
    for link in links:
        owned = members[link.link_id]
        if not owned:
            raise PydanticCustomError(
                "articulation_v2_empty_link",
                "link {link} has no explicit member prim",
                {"link": link.link_id},
            )
        if link.body_authoring == "existing" and (
            len(owned) != 1 or owned[0] != link.body_prim_path
        ):
            raise PydanticCustomError(
                "articulation_v2_invalid_existing_link_membership",
                "existing link {link} requires one identity member",
                {"link": link.link_id},
            )


def _validate_joint_graph(
    joints: tuple[JointRecordV2, ...],
    links_by_id: Mapping[str, LinkRecordV1],
    declared_roots: tuple[str, ...],
) -> None:
    parent_by_child: dict[str, str] = {}
    ordered_pairs: set[tuple[str, str]] = set()
    for joint in joints:
        missing = sorted({joint.body0_link, joint.body1_link}.difference(links_by_id))
        if missing:
            raise PydanticCustomError(
                "articulation_v2_missing_joint_link",
                "joint {joint} references missing links: {links}",
                {"joint": joint.joint_id, "links": ", ".join(missing)},
            )
        body1 = links_by_id[joint.body1_link]
        constraint = joint.constraint
        if isinstance(
            constraint,
            SphericalConstraintV2 | FixedConstraintV2 | DistanceConstraintV2,
        ):
            if body1.axis_stage is not None:
                raise PydanticCustomError(
                    "articulation_v2_body1_axis_not_applicable",
                    "non-axis joint {joint} requires body1 link {link} "
                    "axis_stage to be absent",
                    {"joint": joint.joint_id, "link": body1.link_id},
                )
        elif (
            isinstance(constraint, RevoluteConstraintV2 | PrismaticConstraintV2)
            and constraint.axis_stage is not None
        ):
            if body1.axis_stage is None:
                raise PydanticCustomError(
                    "articulation_v2_body1_axis_missing",
                    "joint {joint} axis_stage requires body1 link {link} axis_stage",
                    {"joint": joint.joint_id, "link": body1.link_id},
                )
            if not _axes_match(constraint.axis_stage, body1.axis_stage):
                raise PydanticCustomError(
                    "articulation_v2_body1_axis_conflict",
                    "joint {joint} axis_stage conflicts with body1 link {link} "
                    "axis_stage",
                    {"joint": joint.joint_id, "link": body1.link_id},
                )
        pair = (joint.body0_link, joint.body1_link)
        if pair in ordered_pairs:
            raise PydanticCustomError(
                "articulation_v2_duplicate_ordered_edge",
                "only one joint may exist per ordered link pair",
            )
        ordered_pairs.add(pair)
        if joint.body1_link in parent_by_child:
            raise PydanticCustomError(
                "articulation_v2_multiple_incoming_joints",
                "a link may have only one incoming joint",
            )
        parent_by_child[joint.body1_link] = joint.body0_link

    cycle = _find_cycle(parent_by_child)
    if cycle is not None:
        raise PydanticCustomError(
            "articulation_v2_closed_loop_unsupported",
            "closed-loop joint graphs are unsupported: {cycle}",
            {"cycle": " -> ".join(cycle)},
        )
    expected_roots = tuple(sorted(set(links_by_id) - set(parent_by_child)))
    if declared_roots != expected_roots:
        raise PydanticCustomError(
            "articulation_v2_root_mismatch",
            "articulation_roots do not match graph roots",
        )


def _validate_document_status_and_diagnostics(
    status: ReviewStatus,
    links: tuple[LinkRecordV1, ...],
    joints: tuple[JointRecordV2, ...],
    diagnostics: tuple[ContractDiagnosticV1, ...],
) -> None:
    records: tuple[LinkRecordV1 | JointRecordV2, ...] = (*links, *joints)
    review_records = {
        (record.kind, _record_identifier(record))
        for record in records
        if record.review_status == "review_required"
    }
    all_records = {(record.kind, _record_identifier(record)) for record in records}
    diagnosed: set[tuple[str, str]] = set()
    diagnosed_fields: dict[tuple[str, str], set[str]] = {}
    for diagnostic in diagnostics:
        target = (diagnostic.record_kind, diagnostic.record_id)
        if target not in all_records:
            raise PydanticCustomError(
                "articulation_v2_diagnostic_target_missing",
                "diagnostic targets a missing record",
            )
        if target not in review_records:
            raise PydanticCustomError(
                "articulation_v2_diagnostic_targets_ready_record",
                "diagnostics may target only review_required records",
            )
        diagnosed.add(target)
        if diagnostic.field is not None:
            diagnosed_fields.setdefault(target, set()).add(diagnostic.field)
    if diagnosed != review_records:
        raise PydanticCustomError(
            "articulation_v2_review_record_undiagnosed",
            "every review_required record requires a diagnostic",
        )
    expected_status: ReviewStatus = (
        "review_required" if review_records else "ready_for_rigger_input"
    )
    if status != expected_status:
        raise PydanticCustomError(
            "articulation_v2_status_mismatch",
            "contract status does not match record readiness",
        )
    for joint in joints:
        if joint.review_status != "review_required":
            continue
        target = ("joint", joint.joint_id)
        fallback_fields = {
            item.field
            for item in joint.field_evidence
            if not _is_source_backed(item.provenance)
        }
        if not fallback_fields.issubset(diagnosed_fields.get(target, set())):
            raise PydanticCustomError(
                "articulation_v2_fallback_evidence_undiagnosed",
                "review fallback evidence requires a targeted diagnostic",
            )


def _validate_summary(
    summary: ContractSummaryV1,
    prims: tuple[PrimRecordV1, ...],
    links: tuple[LinkRecordV1, ...],
    joints: tuple[JointRecordV2, ...],
    diagnostics: tuple[ContractDiagnosticV1, ...],
) -> None:
    expected = ContractSummaryV1(
        prim_count=len(prims),
        link_count=len(links),
        joint_count=len(joints),
        review_required_link_count=sum(
            link.review_status == "review_required" for link in links
        ),
        review_required_joint_count=sum(
            joint.review_status == "review_required" for joint in joints
        ),
        diagnostic_count=len(diagnostics),
    )
    if summary != expected:
        raise PydanticCustomError(
            "articulation_v2_summary_mismatch",
            "summary does not match canonical record and diagnostic counts",
        )


def _validate_declared_evidence(
    declared: set[ArtifactIdentityV1],
    prims: tuple[PrimRecordV1, ...],
    links: tuple[LinkRecordV1, ...],
    joints: tuple[JointRecordV2, ...],
) -> None:
    provenance: list[FieldProvenanceV1] = [prim.membership_evidence for prim in prims]
    provenance.extend(item for link in links for item in link.field_evidence.values())
    provenance.extend(
        item.provenance for joint in joints for item in joint.field_evidence
    )
    for item in provenance:
        if item.artifact is not None and item.artifact not in declared:
            raise PydanticCustomError(
                "articulation_v2_undeclared_evidence_artifact",
                "field evidence references an undeclared artifact",
            )


def _stable_validation_error(exc: ValidationError) -> tuple[str, str]:
    errors = exc.errors()
    first = errors[0]
    error_type = str(first["type"])
    location = tuple(str(part) for part in first["loc"])
    message = str(first["msg"])
    if error_type.startswith("articulation_v2_"):
        return error_type, message
    if error_type == "union_tag_invalid":
        if "constraint" in location:
            return "articulation_v2_unknown_constraint_kind", message
        if "attachments" in location:
            return "articulation_v2_unknown_attachment_kind", message
        return "articulation_v2_unknown_record_kind", message
    if error_type == "union_tag_not_found":
        if "constraint" in location:
            return "articulation_v2_constraint_kind_missing", message
        if "attachments" in location:
            return "articulation_v2_attachment_kind_missing", message
        return "articulation_v2_record_kind_missing", message
    if error_type == "missing" and "attachments" in location:
        return "articulation_v2_incomplete_attachment_frames", message
    if error_type == "literal_error" and "sampling_mode" in location:
        return "articulation_v2_time_varying_transform_unsupported", message
    if location == ("schema_version",):
        return "articulation_v2_version_mismatch", message
    return "articulation_v2_invalid_contract", message


def _python_wire_shape(value: Any) -> Any:
    """Translate JSON shapes and canonicalize semantic zero recursively."""

    if isinstance(value, Mapping):
        return {key: _python_wire_shape(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return tuple(_python_wire_shape(item) for item in value)
    if isinstance(value, float) and value == 0.0:
        return 0.0
    return value


def _strict_revalidate_v1_contract(
    contract: ArticulationContractV1,
) -> ArticulationContractV1:
    try:
        payload = contract.model_dump_json()
    except (PydanticSerializationError, TypeError, ValueError, OverflowError) as exc:
        raise ArticulationContractV2Error(
            "articulation_v2_serialization_failed",
            "articulation-v1 input could not be serialized for strict conversion",
        ) from exc
    try:
        return ArticulationContractV1.model_validate_json(payload)
    except ValidationError as exc:
        raise ArticulationContractV2Error(
            "articulation_v2_invalid_v1_contract",
            "articulation-v1 input failed strict revalidation before conversion",
        ) from exc


def _summary(
    records: tuple[ArticulationRecordV2, ...],
    diagnostics: tuple[ContractDiagnosticV1, ...],
) -> ContractSummaryV1:
    links = tuple(record for record in records if isinstance(record, LinkRecordV1))
    joints = tuple(record for record in records if isinstance(record, JointRecordV2))
    return ContractSummaryV1(
        prim_count=sum(isinstance(record, PrimRecordV1) for record in records),
        link_count=len(links),
        joint_count=len(joints),
        review_required_link_count=sum(
            link.review_status == "review_required" for link in links
        ),
        review_required_joint_count=sum(
            joint.review_status == "review_required" for joint in joints
        ),
        diagnostic_count=len(diagnostics),
    )


def _record_identifier(
    record: PrimRecordV1 | LinkRecordV1 | JointRecordV2,
) -> str:
    if isinstance(record, PrimRecordV1):
        return str(record.prim_path)
    if isinstance(record, LinkRecordV1):
        return str(record.link_id)
    return record.joint_id


def _record_sort_key(
    record: PrimRecordV1 | LinkRecordV1 | JointRecordV2,
) -> tuple[int, str]:
    return (_RECORD_KIND_ORDER[record.kind], _record_identifier(record))


def _constraint_is_unresolved(constraint: JointConstraintV2) -> bool:
    if isinstance(constraint, RevoluteConstraintV2 | PrismaticConstraintV2):
        return constraint.limit_mode == "unresolved"
    if isinstance(constraint, SphericalConstraintV2):
        return constraint.angular_limit_mode == "unresolved"
    return False


def _constraint_mode_field(constraint: JointConstraintV2) -> str:
    if isinstance(constraint, SphericalConstraintV2):
        return "constraint.angular_limit_mode"
    return "constraint.limit_mode"


def _is_source_backed(provenance: FieldProvenanceV1) -> bool:
    return bool(
        provenance.source in _SOURCE_BACKED_PROVENANCE_SOURCES
        and provenance.artifact is not None
        and provenance.prim_path is not None
        and provenance.properties
    )


def _validate_bounded_pair(
    *,
    mode: str,
    bounded_mode: str,
    lower: float | None,
    upper: float | None,
    label: str,
) -> None:
    if mode == bounded_mode:
        if lower is None or upper is None:
            raise PydanticCustomError(
                "articulation_v2_bounded_limits_incomplete",
                "{label} bounded mode requires lower and upper limits",
                {"label": label},
            )
        if lower > upper:
            raise PydanticCustomError(
                "articulation_v2_bounds_inverted",
                "{label} lower limit must not exceed upper limit",
                {"label": label},
            )
        return
    if lower is not None or upper is not None:
        raise PydanticCustomError(
            "articulation_v2_limits_not_applicable",
            "{label} non-bounded mode must not carry scalar limits",
            {"label": label},
        )


def _finite_scalar(value: float | None, code: str) -> float | None:
    if value is not None and not math.isfinite(value):
        raise PydanticCustomError(code, "value must be finite")
    return 0.0 if value == 0.0 else value


def _axes_match(first: Vector3, second: Vector3) -> bool:
    return all(
        math.isclose(left, right, rel_tol=0.0, abs_tol=_NORMALIZATION_TOLERANCE)
        for left, right in zip(first, second, strict=True)
    )


def _determinant3(matrix: tuple[tuple[float, ...], ...]) -> float:
    first, second, third = matrix
    return (
        first[0] * (second[1] * third[2] - second[2] * third[1])
        - first[1] * (second[0] * third[2] - second[2] * third[0])
        + first[2] * (second[0] * third[1] - second[1] * third[0])
    )


def _quaternion_from_rotation_rows(
    matrix: tuple[tuple[float, ...], ...],
) -> QuaternionWxyz:
    m00, m01, m02 = matrix[0]
    m10, m11, m12 = matrix[1]
    m20, m21, m22 = matrix[2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = (
            0.25 * scale,
            (m21 - m12) / scale,
            (m02 - m20) / scale,
            (m10 - m01) / scale,
        )
    elif m00 > m11 and m00 > m22:
        scale = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        quaternion = (
            (m21 - m12) / scale,
            0.25 * scale,
            (m01 + m10) / scale,
            (m02 + m20) / scale,
        )
    elif m11 > m22:
        scale = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        quaternion = (
            (m02 - m20) / scale,
            (m01 + m10) / scale,
            0.25 * scale,
            (m12 + m21) / scale,
        )
    else:
        scale = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        quaternion = (
            (m10 - m01) / scale,
            (m02 + m20) / scale,
            (m12 + m21) / scale,
            0.25 * scale,
        )
    norm = math.sqrt(sum(item * item for item in quaternion))
    return tuple(item / norm for item in quaternion)  # type: ignore[return-value]


def _finite_vector(
    value: tuple[float, ...],
    *,
    code: str,
    label: str,
) -> tuple[float, ...]:
    if not all(math.isfinite(component) for component in value):
        raise PydanticCustomError(
            code, "{label} must contain finite values", {"label": label}
        )
    return tuple(0.0 if component == 0.0 else component for component in value)


def _normalized_axis(value: Vector3) -> Vector3:
    finite = _finite_vector(
        value,
        code="articulation_v2_non_finite_axis",
        label="constraint axis_stage",
    )
    norm = math.sqrt(sum(component * component for component in finite))
    if not math.isclose(
        norm,
        1.0,
        rel_tol=0.0,
        abs_tol=_NORMALIZATION_TOLERANCE,
    ):
        raise PydanticCustomError(
            "articulation_v2_invalid_axis_norm",
            "constraint axis_stage must be a normalized vector",
        )
    return finite  # type: ignore[return-value]


def _require_unique(
    values: Iterable[str],
    code: str,
    label: str,
) -> None:
    values_tuple = tuple(values)
    if len(values_tuple) != len(set(values_tuple)):
        raise PydanticCustomError(code, "{label} must be unique", {"label": label})


def _find_cycle(
    parent_by_child: Mapping[str, str],
) -> tuple[str, ...] | None:
    resolved: set[str] = set()
    for start in sorted(parent_by_child):
        path: list[str] = []
        positions: dict[str, int] = {}
        current = start
        while (
            current in parent_by_child
            and current not in positions
            and current not in resolved
        ):
            positions[current] = len(path)
            path.append(current)
            current = parent_by_child[current]
        if current in positions:
            cycle = path[positions[current] :]
            first = min(cycle)
            index = cycle.index(first)
            canonical = cycle[index:] + cycle[:index]
            return (*canonical, canonical[0])
        resolved.update(path)
    return None


def _nonblank(value: str, label: str) -> str:
    if not value.strip():
        raise ValueError(f"{label} must not be blank")
    return value


__all__ = [
    "ARTICULATION_CONTRACT_V2_IMPLEMENTED_CAPABILITY_IDS",
    "ARTICULATION_CONTRACT_V2_SCHEMA_VERSION",
    "ArticulationContractV2",
    "ArticulationContractV2Error",
    "ArticulationRecordV2",
    "AttachmentFrameV2",
    "AttachmentFramesV2",
    "attachment_frame_from_basis_v2",
    "Body1OriginAttachmentDefaultV2",
    "ConstraintKind",
    "DistanceConstraintV2",
    "ExplicitAttachmentFramesV2",
    "FieldEvidenceV2",
    "FixedConstraintV2",
    "JointConstraintV2",
    "JointRecordV2",
    "PrismaticConstraintV2",
    "RevoluteConstraintV2",
    "SphericalConstraintV2",
    "canonical_articulation_v2_json",
    "canonical_articulation_v2_sha256",
    "convert_articulation_v1_to_v2",
    "downgrade_articulation_v2_to_v1",
    "negotiate_articulation_contract_v2",
    "parse_articulation_contract_v2",
]
