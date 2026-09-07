# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict contracts for the release-selected articulation-v2 workflow seam.

These models deliberately do not reuse or widen the frozen articulation-v1
request, Stage2-v0 candidate, review, checkpoint, or result contracts.
Qualification runners and their evidence roots are also intentionally absent:
the workflow consumes only the release-selection facts owned by Joint #870.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Literal, Self, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    field_validator,
    model_validator,
)

ARTICULATION_V2_REQUEST_SCHEMA_VERSION: Literal[
    "content-agent-workflows.articulation-request.v2"
] = "content-agent-workflows.articulation-request.v2"
ARTICULATION_V2_RELEASE_SELECTION_SCHEMA_VERSION: Literal[
    "content-agent-workflows.articulation-release-selection.v2"
] = "content-agent-workflows.articulation-release-selection.v2"
ARTICULATION_V2_REVIEW_CONTEXT_SCHEMA_VERSION: Literal[
    "content-agent-workflows.articulation-review-context.v2"
] = "content-agent-workflows.articulation-review-context.v2"
ARTICULATION_V2_REVIEW_RECEIPT_SCHEMA_VERSION: Literal[
    "content-agent-workflows.articulation-review-receipt.v2"
] = "content-agent-workflows.articulation-review-receipt.v2"
ARTICULATION_V2_AUTHORING_INTENT_SCHEMA_VERSION: Literal[
    "content-agent-workflows.articulation-authoring-intent.v2"
] = "content-agent-workflows.articulation-authoring-intent.v2"
ARTICULATION_V2_PUBLICATION_RESULT_SCHEMA_VERSION: Literal[
    "content-agent-workflows.articulation-publication-result.v2"
] = "content-agent-workflows.articulation-publication-result.v2"
ARTICULATION_V2_READBACK_RESULT_SCHEMA_VERSION: Literal[
    "content-agent-workflows.articulation-readback-result.v2"
] = "content-agent-workflows.articulation-readback-result.v2"
ARTICULATION_V2_CHECKPOINT_SCHEMA_VERSION: Literal[
    "content-agent-workflows.articulation-checkpoint.v2"
] = "content-agent-workflows.articulation-checkpoint.v2"
ARTICULATION_V2_RESULT_SCHEMA_VERSION: Literal[
    "content-agent-workflows.articulation-result.v2"
] = "content-agent-workflows.articulation-result.v2"

ARTICULATION_V2_CONTRACT_VERSION: Literal["joint-agent-articulation-v2"] = (
    "joint-agent-articulation-v2"
)
ARTICULATION_V2_RELEASE_FACADE_ID: Literal[
    "joint-agent.release-selected-articulation-v2"
] = "joint-agent.release-selected-articulation-v2"

ArticulationV2WorkflowMode = Literal["interactive", "batch"]
ArticulationV2SelectorKind = Literal["fixed", "distance"]
ArticulationV2Phase = Literal[
    "initialized",
    "needs_review",
    "ready_for_authoring",
    "published",
    "completed",
    "cancelled",
    "failed",
]
ArticulationV2Status = Literal[
    "awaiting_review_context",
    "needs_review",
    "ready_for_authoring",
    "published",
    "completed",
    "cancelled",
    "failed",
]

_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_QUATERNION_NORMALIZATION_TOLERANCE = 1e-6
# Mirrors the strict #870 facade; widening requires a coordinated contract change.
ARTICULATION_V2_READBACK_TOLERANCE = 1e-6


class _StrictV2Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _strict_model_input(value: Any) -> Any:
    """Rebuild Python input without dropping validation-bypassing extra fields."""

    if isinstance(value, BaseModel):
        document = {
            name: _strict_model_input(field_value)
            for name, field_value in vars(value).items()
        }
        if value.model_extra:
            document.update(
                {
                    name: _strict_model_input(field_value)
                    for name, field_value in value.model_extra.items()
                }
            )
        return document
    if isinstance(value, tuple):
        return tuple(_strict_model_input(item) for item in value)
    if isinstance(value, list):
        return [_strict_model_input(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _strict_model_input(item) for key, item in value.items()}
    return value


def _strict_revalidate_articulation_v2_model[ModelT: BaseModel](
    value: ModelT,
) -> ModelT:
    model_type = type(value)
    if (
        model_type.__module__ != __name__
        or globals().get(model_type.__name__) is not model_type
    ):
        raise TypeError(
            "articulation-v2 contracts require an exact models_v2 model class"
        )
    return type(value).model_validate(
        _strict_model_input(value),
        strict=True,
    )


def articulation_v2_canonical_sha256(
    value: BaseModel | Mapping[str, Any],
) -> str:
    """Digest one v2 model or mapping as canonical compact JSON.

    This convention binds in-memory release selections and idempotency
    payloads.  Artifact-binding SHA-256 fields instead bind the exact persisted
    JSON file bytes.
    """

    if isinstance(value, BaseModel):
        validated_value = _strict_revalidate_articulation_v2_model(value)
        document = validated_value.model_dump(mode="json")
    else:
        document = dict(value)
    payload = json.dumps(
        document,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _canonical_absolute_path(value: Path, *, label: str) -> Path:
    expanded = value.expanduser()
    if not expanded.is_absolute():
        raise ValueError(f"{label} paths must be absolute")
    return expanded.resolve()


def _is_canonical_absolute_prim_path(value: str) -> bool:
    try:
        from pxr import Sdf
    except ImportError:
        if not value.startswith("/") or value == "/":
            return False
        return all(segment.isidentifier() for segment in value[1:].split("/"))

    validation = Sdf.Path.IsValidPathString(value)
    is_valid = validation[0] if isinstance(validation, tuple) else validation
    if not is_valid:
        return False
    path = Sdf.Path(value)
    return bool(
        str(path) == value
        and path.IsAbsolutePath()
        and path.IsPrimPath()
        and not path.IsAbsoluteRootPath()
        and not path.ContainsPrimVariantSelection()
    )


class ArticulationV2ArtifactIdentity(_StrictV2Model):
    """Exact local identity for an opaque, non-USD evidence artifact."""

    uri: str = Field(min_length=1)
    path: Path
    sha256: str = Field(pattern=_SHA256_PATTERN)
    size_bytes: int = Field(gt=0)

    @field_validator("path")
    @classmethod
    def _absolute_path(cls, value: Path) -> Path:
        return _canonical_absolute_path(value, label="artifact identity")


class ArticulationV2UsdArtifactIdentity(ArticulationV2ArtifactIdentity):
    """Exact USD root plus its complete resolved dependency-closure identity."""

    dependency_bundle_sha256: str = Field(pattern=_SHA256_PATTERN)


class ArticulationV2OutputTarget(_StrictV2Model):
    """Pre-authoring logical and filesystem identity of the requested output."""

    uri: str = Field(min_length=1)
    path: Path
    format: Literal["raw_usd", "usdz"]

    @field_validator("path")
    @classmethod
    def _absolute_output_path(cls, value: Path) -> Path:
        return _canonical_absolute_path(value, label="output target")

    @model_validator(mode="after")
    def _format_matches_suffix(self) -> Self:
        suffix = self.path.suffix.lower()
        if self.format == "usdz" and suffix != ".usdz":
            raise ValueError("usdz output targets must use the .usdz suffix")
        if self.format == "raw_usd" and suffix not in {".usd", ".usda", ".usdc"}:
            raise ValueError("raw_usd output targets must use a USD layer suffix")
        return self


class ArticulationV2AttachmentFrame(_StrictV2Model):
    """One static body-local attachment frame in SI/WXYZ release semantics."""

    position_meters: tuple[StrictFloat, StrictFloat, StrictFloat]
    orientation_wxyz: tuple[
        StrictFloat,
        StrictFloat,
        StrictFloat,
        StrictFloat,
    ]
    sampling_mode: Literal["static"] = "static"

    @field_validator("position_meters", "orientation_wxyz")
    @classmethod
    def _finite_components(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if any(not math.isfinite(component) for component in value):
            raise ValueError("attachment-frame components must be finite")
        return tuple(0.0 if component == 0.0 else component for component in value)

    @field_validator("orientation_wxyz")
    @classmethod
    def _canonical_unit_orientation(
        cls,
        value: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        norm = math.sqrt(sum(component * component for component in value))
        if not math.isclose(
            norm,
            1.0,
            rel_tol=0.0,
            abs_tol=_QUATERNION_NORMALIZATION_TOLERANCE,
        ):
            raise ValueError("attachment orientation_wxyz must be a unit quaternion")
        canonical = tuple(0.0 if component == 0.0 else component for component in value)
        first_nonzero = next(
            (component for component in canonical if component != 0.0),
            None,
        )
        if first_nonzero is None:
            raise ValueError("attachment orientation_wxyz must not be zero")
        if first_nonzero < 0.0:
            canonical = tuple(-component for component in canonical)
        canonical = tuple(
            0.0 if component == 0.0 else component for component in canonical
        )
        return cast(tuple[float, float, float, float], canonical)


class ArticulationV2FixedContract(_StrictV2Model):
    kind: Literal["fixed"] = "fixed"
    policy: Literal["explicit_two_body_constraint"] = "explicit_two_body_constraint"
    source_joint_prim: str
    body0: str
    body1: str
    body0_attachment: ArticulationV2AttachmentFrame
    body1_attachment: ArticulationV2AttachmentFrame

    @field_validator("source_joint_prim", "body0", "body1")
    @classmethod
    def _canonical_prim_path(cls, value: str) -> str:
        if not _is_canonical_absolute_prim_path(value):
            raise ValueError("constraint paths must be canonical absolute prim paths")
        return value

    @model_validator(mode="after")
    def _distinct_endpoints(self) -> Self:
        if self.body0 == self.body1:
            raise ValueError("fixed body0 and body1 must be distinct")
        return self


class ArticulationV2DistanceContract(_StrictV2Model):
    kind: Literal["distance"] = "distance"
    policy: Literal["bounded_two_body_constraint"] = "bounded_two_body_constraint"
    source_joint_prim: str
    body0: str
    body1: str
    body0_attachment: ArticulationV2AttachmentFrame
    body1_attachment: ArticulationV2AttachmentFrame
    minimum_distance_meters: StrictFloat = Field(ge=0.0)
    maximum_distance_meters: StrictFloat = Field(ge=0.0)

    @field_validator("source_joint_prim", "body0", "body1")
    @classmethod
    def _canonical_prim_path(cls, value: str) -> str:
        if not _is_canonical_absolute_prim_path(value):
            raise ValueError("constraint paths must be canonical absolute prim paths")
        return value

    @field_validator("minimum_distance_meters", "maximum_distance_meters")
    @classmethod
    def _canonical_distance_zero(cls, value: float) -> float:
        return 0.0 if value == 0.0 else value

    @model_validator(mode="after")
    def _valid_distance_interval(self) -> Self:
        values = (self.minimum_distance_meters, self.maximum_distance_meters)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("distance bounds must be finite")
        if self.minimum_distance_meters > self.maximum_distance_meters:
            raise ValueError("distance bounds must form an ordered interval")
        if self.body0 == self.body1:
            raise ValueError("distance body0 and body1 must be distinct")
        return self


ArticulationV2Constraint = Annotated[
    ArticulationV2FixedContract | ArticulationV2DistanceContract,
    Field(discriminator="kind"),
]


class ArticulationV2FixedSelector(_StrictV2Model):
    kind: Literal["fixed"] = "fixed"
    capability_id: Literal["fixed.explicit_two_body_constraint"] = (
        "fixed.explicit_two_body_constraint"
    )
    property_profile: Literal["explicit_two_body_constraint"] = (
        "explicit_two_body_constraint"
    )
    contract: ArticulationV2FixedContract


class ArticulationV2DistanceSelector(_StrictV2Model):
    kind: Literal["distance"] = "distance"
    capability_id: Literal["distance.bounded_two_body_constraint"] = (
        "distance.bounded_two_body_constraint"
    )
    property_profile: Literal["source_backed_bounds"] = "source_backed_bounds"
    contract: ArticulationV2DistanceContract


ArticulationV2Selector = Annotated[
    ArticulationV2FixedSelector | ArticulationV2DistanceSelector,
    Field(discriminator="kind"),
]


class ArticulationV2StaticAttestation(_StrictV2Model):
    """The retained static outcome selected by #870, not its evidence root."""

    scorecard_sha256: str = Field(pattern=_SHA256_PATTERN)
    run_plan_sha256: str = Field(pattern=_SHA256_PATTERN)
    capability_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    run_id: str = Field(min_length=1)
    row_admission_sha256: str = Field(pattern=_SHA256_PATTERN)


class ArticulationV2ReleaseSelection(_StrictV2Model):
    """Exact release decision returned by the future #870 facade."""

    schema_version: Literal[
        "content-agent-workflows.articulation-release-selection.v2"
    ] = ARTICULATION_V2_RELEASE_SELECTION_SCHEMA_VERSION
    facade_id: Literal["joint-agent.release-selected-articulation-v2"] = (
        ARTICULATION_V2_RELEASE_FACADE_ID
    )
    manifest_schema_version: Literal["joint-agent-capability-manifest-v3"] = (
        "joint-agent-capability-manifest-v3"
    )
    manifest_version: str = Field(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+(?:-[a-z0-9.-]+)?$")
    manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    capability_id: str = Field(min_length=1)
    selector_kind: ArticulationV2SelectorKind
    property_profile: str = Field(min_length=1)
    row_admission_sha256: str = Field(pattern=_SHA256_PATTERN)
    selected: Literal[True] = True
    public_exposure: Literal["enabled"] = "enabled"
    disposition: Literal["supported"] = "supported"
    support_level: Literal["static_qualified"] = "static_qualified"
    static_status: Literal["pass"] = "pass"
    static_attestation: ArticulationV2StaticAttestation
    dynamic_qualified: Literal[False] = False
    authoring_contract_version: Literal["joint-agent-articulation-v2"] = (
        ARTICULATION_V2_CONTRACT_VERSION
    )
    reference_asset_id: str = Field(min_length=1)
    reference_artifact_key: str = Field(min_length=1)
    reference: ArticulationV2UsdArtifactIdentity

    @model_validator(mode="after")
    def _attestation_matches_admission(self) -> Self:
        if self.static_attestation.row_admission_sha256 != self.row_admission_sha256:
            raise ValueError(
                "static attestation must bind the selected row admission digest"
            )
        return self


class ArticulationV2WorkflowRequest(_StrictV2Model):
    """Version-discriminated fixed/distance workflow request."""

    schema_version: Literal["content-agent-workflows.articulation-request.v2"] = (
        ARTICULATION_V2_REQUEST_SCHEMA_VERSION
    )
    workflow_version: Literal["articulation-v2"] = "articulation-v2"
    mode: ArticulationV2WorkflowMode
    intent: str = Field(min_length=1)
    intent_sha256: str = Field(pattern=_SHA256_PATTERN)
    target_runtime: Literal["usd-cli"] = "usd-cli"
    selector: ArticulationV2Selector
    release_selection: ArticulationV2ReleaseSelection
    source: ArticulationV2UsdArtifactIdentity
    reference: ArticulationV2UsdArtifactIdentity
    output: ArticulationV2OutputTarget
    output_dir: Path
    require_explicit_review: Literal[True] = True
    apply_masses: Literal[False] = False
    apply_collision: Literal[False] = False
    claim_dynamic_qualification: Literal[False] = False

    @field_validator("output_dir")
    @classmethod
    def _absolute_output_dir(cls, value: Path) -> Path:
        return _canonical_absolute_path(value, label="workflow output_dir")

    @model_validator(mode="after")
    def _bind_exact_release_scope(self) -> Self:
        expected_intent_sha256 = hashlib.sha256(self.intent.encode("utf-8")).hexdigest()
        if self.intent_sha256 != expected_intent_sha256:
            raise ValueError("intent_sha256 must bind the exact workflow intent")
        selection = self.release_selection
        if selection.capability_id != self.selector.capability_id:
            raise ValueError("release selection capability does not match selector")
        if selection.selector_kind != self.selector.kind:
            raise ValueError("release selection kind does not match selector")
        if selection.property_profile != self.selector.property_profile:
            raise ValueError(
                "release selection property profile does not match selector"
            )
        if selection.reference != self.reference:
            raise ValueError(
                "release selection reference identity does not match request"
            )
        if self.source == self.reference:
            raise ValueError("source and reference identities must remain distinct")
        if self.output.path in {self.source.path, self.reference.path}:
            raise ValueError("output target must not alias source or reference paths")
        return self


class ArticulationV2ArtifactBinding(_StrictV2Model):
    path: Path
    sha256: str = Field(pattern=_SHA256_PATTERN)

    @field_validator("path")
    @classmethod
    def _absolute_path(cls, value: Path) -> Path:
        return _canonical_absolute_path(value, label="artifact binding")


class ArticulationV2ReviewContext(_StrictV2Model):
    schema_version: Literal[
        "content-agent-workflows.articulation-review-context.v2"
    ] = ARTICULATION_V2_REVIEW_CONTEXT_SCHEMA_VERSION
    request_sha256: str = Field(pattern=_SHA256_PATTERN)
    release_selection_sha256: str = Field(pattern=_SHA256_PATTERN)
    selector_kind: ArticulationV2SelectorKind
    capability_id: str = Field(min_length=1)
    source_sha256: str = Field(pattern=_SHA256_PATTERN)
    reference_sha256: str = Field(pattern=_SHA256_PATTERN)
    scene_evidence: ArticulationV2ArtifactIdentity


class ArticulationV2ReviewReceipt(_StrictV2Model):
    schema_version: Literal[
        "content-agent-workflows.articulation-review-receipt.v2"
    ] = ARTICULATION_V2_REVIEW_RECEIPT_SCHEMA_VERSION
    request_sha256: str = Field(pattern=_SHA256_PATTERN)
    release_selection_sha256: str = Field(pattern=_SHA256_PATTERN)
    review_context_sha256: str = Field(pattern=_SHA256_PATTERN)
    scene_evidence_sha256: str = Field(pattern=_SHA256_PATTERN)
    selector_kind: ArticulationV2SelectorKind
    capability_id: str = Field(min_length=1)
    decision: Literal["accept", "reject"]
    reviewer: str = Field(min_length=1)
    note: str | None = None


class ArticulationV2AuthoringIntent(_StrictV2Model):
    """Idempotent handoff for #870; this workflow never invokes it itself."""

    schema_version: Literal[
        "content-agent-workflows.articulation-authoring-intent.v2"
    ] = ARTICULATION_V2_AUTHORING_INTENT_SCHEMA_VERSION
    facade_id: Literal["joint-agent.release-selected-articulation-v2"] = (
        ARTICULATION_V2_RELEASE_FACADE_ID
    )
    request_sha256: str = Field(pattern=_SHA256_PATTERN)
    release_selection_sha256: str = Field(pattern=_SHA256_PATTERN)
    review_receipt_sha256: str = Field(pattern=_SHA256_PATTERN)
    idempotency_key: str = Field(pattern=_SHA256_PATTERN)
    selector: ArticulationV2Selector
    source: ArticulationV2UsdArtifactIdentity
    reference: ArticulationV2UsdArtifactIdentity
    output: ArticulationV2OutputTarget
    apply_masses: Literal[False] = False
    apply_collision: Literal[False] = False


class ArticulationV2PublicationResult(_StrictV2Model):
    schema_version: Literal[
        "content-agent-workflows.articulation-publication-result.v2"
    ] = ARTICULATION_V2_PUBLICATION_RESULT_SCHEMA_VERSION
    status: Literal["succeeded"] = "succeeded"
    facade_id: Literal["joint-agent.release-selected-articulation-v2"] = (
        ARTICULATION_V2_RELEASE_FACADE_ID
    )
    request_sha256: str = Field(pattern=_SHA256_PATTERN)
    release_selection_sha256: str = Field(pattern=_SHA256_PATTERN)
    review_receipt_sha256: str = Field(pattern=_SHA256_PATTERN)
    authoring_intent_sha256: str = Field(pattern=_SHA256_PATTERN)
    idempotency_key: str = Field(pattern=_SHA256_PATTERN)
    facade_call_id: str = Field(min_length=1)
    facade_call_count: Literal[1] = 1
    selector_kind: ArticulationV2SelectorKind
    capability_id: str = Field(min_length=1)
    source: ArticulationV2UsdArtifactIdentity
    reference: ArticulationV2UsdArtifactIdentity
    output_target: ArticulationV2OutputTarget
    output_artifact: ArticulationV2UsdArtifactIdentity
    authoring_contract_sha256: str = Field(pattern=_SHA256_PATTERN)
    dynamic_qualified: Literal[False] = False

    @model_validator(mode="after")
    def _output_matches_target(self) -> Self:
        if self.output_artifact.uri != self.output_target.uri:
            raise ValueError("published output URI must match the approved target")
        if self.output_artifact.path != self.output_target.path:
            raise ValueError("published output path must match the approved target")
        return self


class ArticulationV2FixedReadback(_StrictV2Model):
    kind: Literal["fixed"] = "fixed"
    joint_prim: str
    body0: str
    body1: str
    body0_attachment: ArticulationV2AttachmentFrame
    body1_attachment: ArticulationV2AttachmentFrame
    policy: Literal["explicit_two_body_constraint"] = "explicit_two_body_constraint"

    @field_validator("joint_prim", "body0", "body1")
    @classmethod
    def _canonical_prim_path(cls, value: str) -> str:
        if not _is_canonical_absolute_prim_path(value):
            raise ValueError("readback paths must be canonical absolute prim paths")
        return value


class ArticulationV2DistanceReadback(_StrictV2Model):
    kind: Literal["distance"] = "distance"
    policy: Literal["bounded_two_body_constraint"] = "bounded_two_body_constraint"
    joint_prim: str
    body0: str
    body1: str
    body0_attachment: ArticulationV2AttachmentFrame
    body1_attachment: ArticulationV2AttachmentFrame
    minimum_distance_meters: StrictFloat = Field(ge=0.0)
    maximum_distance_meters: StrictFloat = Field(ge=0.0)

    @field_validator("joint_prim", "body0", "body1")
    @classmethod
    def _canonical_prim_path(cls, value: str) -> str:
        if not _is_canonical_absolute_prim_path(value):
            raise ValueError("readback paths must be canonical absolute prim paths")
        return value

    @field_validator("minimum_distance_meters", "maximum_distance_meters")
    @classmethod
    def _canonical_distance_zero(cls, value: float) -> float:
        return 0.0 if value == 0.0 else value

    @model_validator(mode="after")
    def _ordered_interval(self) -> Self:
        values = (self.minimum_distance_meters, self.maximum_distance_meters)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("readback distance bounds must be finite")
        if self.minimum_distance_meters > self.maximum_distance_meters:
            raise ValueError("readback distance bounds must be ordered")
        return self


ArticulationV2ConstraintReadback = Annotated[
    ArticulationV2FixedReadback | ArticulationV2DistanceReadback,
    Field(discriminator="kind"),
]


class ArticulationV2ReadbackResult(_StrictV2Model):
    schema_version: Literal[
        "content-agent-workflows.articulation-readback-result.v2"
    ] = ARTICULATION_V2_READBACK_RESULT_SCHEMA_VERSION
    status: Literal["pass", "fail"]
    publication_result_sha256: str = Field(pattern=_SHA256_PATTERN)
    selector_kind: ArticulationV2SelectorKind
    capability_id: str = Field(min_length=1)
    source: ArticulationV2UsdArtifactIdentity
    reference: ArticulationV2UsdArtifactIdentity
    output_artifact: ArticulationV2UsdArtifactIdentity
    evidence_artifact: ArticulationV2ArtifactIdentity
    constraint: ArticulationV2ConstraintReadback | None
    readback_tolerance: StrictFloat = ARTICULATION_V2_READBACK_TOLERANCE
    exact_saved_stage_match: bool
    failures: tuple[str, ...] = ()
    dynamic_qualified: Literal[False] = False

    @field_validator("readback_tolerance")
    @classmethod
    def _supported_readback_tolerance(cls, value: float) -> float:
        if value != ARTICULATION_V2_READBACK_TOLERANCE:
            raise ValueError(
                "readback_tolerance must match the articulation-v2 release contract"
            )
        return value

    @model_validator(mode="after")
    def _status_matches_evidence(self) -> Self:
        passed = (
            self.exact_saved_stage_match
            and not self.failures
            and self.constraint is not None
        )
        if (self.status == "pass") != passed:
            raise ValueError("readback status must reflect exact saved-stage evidence")
        if self.status == "fail" and not self.failures:
            raise ValueError("failed readback requires at least one retained failure")
        if self.constraint is not None and self.constraint.kind != self.selector_kind:
            raise ValueError("readback constraint kind must match the selected kind")
        return self


class ArticulationV2Transition(_StrictV2Model):
    timestamp: str = Field(min_length=1)
    from_phase: ArticulationV2Phase
    to_phase: ArticulationV2Phase
    reason: str = Field(min_length=1)


class ArticulationV2Checkpoint(_StrictV2Model):
    schema_version: Literal["content-agent-workflows.articulation-checkpoint.v2"] = (
        ARTICULATION_V2_CHECKPOINT_SCHEMA_VERSION
    )
    revision: int = Field(default=0, ge=0)
    mode: ArticulationV2WorkflowMode
    phase: ArticulationV2Phase = "initialized"
    request: ArticulationV2ArtifactBinding
    release_selection_sha256: str = Field(pattern=_SHA256_PATTERN)
    source: ArticulationV2UsdArtifactIdentity
    reference: ArticulationV2UsdArtifactIdentity
    output: ArticulationV2OutputTarget
    review_context: ArticulationV2ArtifactBinding | None = None
    review_receipt: ArticulationV2ArtifactBinding | None = None
    authoring_intent: ArticulationV2ArtifactBinding | None = None
    publication_result: ArticulationV2ArtifactBinding | None = None
    readback_result: ArticulationV2ArtifactBinding | None = None
    idempotency_key: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    facade_call_id: str | None = None
    facade_call_count: int = Field(default=0, ge=0, le=1)
    transitions: tuple[ArticulationV2Transition, ...] = ()
    error: str | None = None

    @model_validator(mode="after")
    def _phase_artifacts_are_consistent(self) -> Self:
        if self.phase == "needs_review" and self.review_context is None:
            raise ValueError("needs_review requires a bound review context")
        if self.phase in {"ready_for_authoring", "published", "completed"}:
            if self.review_receipt is None or self.authoring_intent is None:
                raise ValueError(
                    "authoring-ready checkpoints require review and intent artifacts"
                )
            if self.idempotency_key is None:
                raise ValueError("authoring-ready checkpoints require idempotency")
        if self.phase in {"published", "completed"}:
            if self.publication_result is None or self.facade_call_count != 1:
                raise ValueError(
                    "published checkpoints require exactly one facade call"
                )
            if self.facade_call_id is None:
                raise ValueError("published checkpoints require a facade call ID")
        if self.phase == "completed" and self.readback_result is None:
            raise ValueError("completed checkpoints require saved-stage readback")
        if self.readback_result is not None and self.phase not in {
            "completed",
            "failed",
        }:
            raise ValueError("readback artifacts require a completed or failed phase")
        if self.phase == "failed" and self.readback_result is not None:
            if (
                self.publication_result is None
                or self.facade_call_count != 1
                or self.facade_call_id is None
                or self.idempotency_key is None
                or self.error is None
            ):
                raise ValueError(
                    "failed readback checkpoints retain publication and error identity"
                )
        return self


class ArticulationV2WorkflowResult(_StrictV2Model):
    schema_version: Literal["content-agent-workflows.articulation-result.v2"] = (
        ARTICULATION_V2_RESULT_SCHEMA_VERSION
    )
    success: bool
    status: ArticulationV2Status
    mode: ArticulationV2WorkflowMode
    capability_id: str = Field(min_length=1)
    selector_kind: ArticulationV2SelectorKind
    output_dir: Path
    checkpoint_path: Path
    request_path: Path
    review_context_path: Path | None = None
    review_receipt_path: Path | None = None
    authoring_intent_path: Path | None = None
    publication_result_path: Path | None = None
    readback_result_path: Path | None = None
    output_asset_path: Path | None = None
    idempotency_key: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    facade_call_count: int = Field(ge=0, le=1)
    dynamic_qualified: Literal[False] = False
    message: str = Field(min_length=1)

    @model_validator(mode="after")
    def _status_is_consistent(self) -> Self:
        if self.success != (self.status == "completed"):
            raise ValueError("success is true only for a completed v2 workflow")
        if self.status == "completed" and (
            self.output_asset_path is None
            or self.publication_result_path is None
            or self.readback_result_path is None
            or self.facade_call_count != 1
        ):
            raise ValueError(
                "completed v2 workflows require one publication and exact readback"
            )
        if self.status in {"ready_for_authoring", "published", "completed"} and (
            self.idempotency_key is None
        ):
            raise ValueError("post-review v2 results require an idempotency key")
        if (
            self.status == "failed"
            and self.readback_result_path is not None
            and (
                self.output_asset_path is None
                or self.publication_result_path is None
                or self.facade_call_count != 1
                or self.idempotency_key is None
            )
        ):
            raise ValueError(
                "failed readback results retain one publication and its output"
            )
        return self


__all__ = [
    "ARTICULATION_V2_AUTHORING_INTENT_SCHEMA_VERSION",
    "ARTICULATION_V2_CHECKPOINT_SCHEMA_VERSION",
    "ARTICULATION_V2_CONTRACT_VERSION",
    "ARTICULATION_V2_PUBLICATION_RESULT_SCHEMA_VERSION",
    "ARTICULATION_V2_READBACK_RESULT_SCHEMA_VERSION",
    "ARTICULATION_V2_READBACK_TOLERANCE",
    "ARTICULATION_V2_RELEASE_FACADE_ID",
    "ARTICULATION_V2_RELEASE_SELECTION_SCHEMA_VERSION",
    "ARTICULATION_V2_REQUEST_SCHEMA_VERSION",
    "ARTICULATION_V2_RESULT_SCHEMA_VERSION",
    "ARTICULATION_V2_REVIEW_CONTEXT_SCHEMA_VERSION",
    "ARTICULATION_V2_REVIEW_RECEIPT_SCHEMA_VERSION",
    "ArticulationV2AttachmentFrame",
    "ArticulationV2ArtifactBinding",
    "ArticulationV2ArtifactIdentity",
    "ArticulationV2AuthoringIntent",
    "ArticulationV2Checkpoint",
    "ArticulationV2Constraint",
    "ArticulationV2ConstraintReadback",
    "ArticulationV2DistanceContract",
    "ArticulationV2DistanceReadback",
    "ArticulationV2DistanceSelector",
    "ArticulationV2FixedContract",
    "ArticulationV2FixedReadback",
    "ArticulationV2FixedSelector",
    "ArticulationV2OutputTarget",
    "ArticulationV2Phase",
    "ArticulationV2PublicationResult",
    "ArticulationV2ReadbackResult",
    "ArticulationV2ReleaseSelection",
    "ArticulationV2ReviewContext",
    "ArticulationV2ReviewReceipt",
    "ArticulationV2Selector",
    "ArticulationV2SelectorKind",
    "ArticulationV2StaticAttestation",
    "ArticulationV2Status",
    "ArticulationV2Transition",
    "ArticulationV2UsdArtifactIdentity",
    "ArticulationV2WorkflowMode",
    "ArticulationV2WorkflowRequest",
    "ArticulationV2WorkflowResult",
    "articulation_v2_canonical_sha256",
]
