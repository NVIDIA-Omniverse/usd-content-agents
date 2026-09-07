# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Captured source-backed frame admission and owned USD authoring for v2.

This module is deliberately a sibling of the released articulation-v1 and
Stage 2 v0 paths. It accepts only immutable, identity-bound captures for the
canonical contract, the stripped authoring target, and the reviewed source
reference. The target supplies all published bytes; the reference supplies
only joint evidence. No path supplied by a caller is trusted as input.

The opt-in authorer supports self-contained raw USD roots and USDZ packages for
explicit revolute, prismatic, passive spherical, fixed, and distance v2
records. Fixed and distance records are promoted only from exact retained
source-joint evidence. Mobile joints may target exact aggregate rigid links
through the shared V2 membership policy; Gate 3, dynamic qualification, and
public selection remain separate capability gates.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import stat
import struct
import tempfile
import zipfile
from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Literal, NoReturn, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_validator,
    model_validator,
)
from world_understanding.functions.physics.joint_rigger import (
    ArtifactIdentityV1,
    FieldProvenanceV1,
    JointRiggerContractError,
    RigidLinkMemberPlanV1,
    RigidLinkPlanV1,
    identify_usd_artifact,
)
from world_understanding.functions.physics.joint_rigger.opaque_dependencies import (
    OPAQUE_DEPENDENCY_EXTENSIONS,
)
from world_understanding.functions.physics.joint_rigger.reference import (
    RetainedUsdArtifactInspection,
    UsdDependencyOpinion,
    retain_usd_artifact_inspection,
    usd_dependency_inventory,
)
from world_understanding.utils.artifacts import (
    ConfinedAtomicWrite,
    confined_atomic_writer,
    open_confined_directory,
)
from world_understanding.utils.captured_artifacts import (
    CapturedArtifactCleanupError,
    CapturedArtifactError,
    CapturedOpaqueArtifactResolver,
    CapturedOpaqueFile,
    OpaqueArtifactRequest,
    capture_resolved_opaque_file,
)
from world_understanding.utils.usd.package import (
    DEFAULT_MAX_USDZ_EXTRACTED_BYTES,
    DEFAULT_MAX_USDZ_MEMBER_BYTES,
    DEFAULT_MAX_USDZ_PACKAGE_TREE_MEMBERS,
    RetainedUsdzPackageTree,
    UsdzPackageError,
    UsdzPackageTreeManifest,
    extract_usdz_package_for_edit,
    find_usdz_root_layer,
    retain_usdz_package_tree,
    safe_usdz_member_name,
    validate_usdz_package_layout,
    validate_usdz_package_tree,
    write_usdz_package_from_directory,
)

from joint_agent.functions.articulation_contract import (
    LinkRecordV1,
    PrimRecordV1,
)
from joint_agent.functions.articulation_contract_v2 import (
    ArticulationContractV2,
    AttachmentFrameV2,
    DistanceConstraintV2,
    ExplicitAttachmentFramesV2,
    FieldEvidenceV2,
    FixedConstraintV2,
    JointConstraintV2,
    JointRecordV2,
    PrismaticConstraintV2,
    RevoluteConstraintV2,
    SphericalConstraintV2,
)

ARTICULATION_V2_FRAME_AUTHORING_SCHEMA_VERSION_V2: Literal[
    "joint-agent-articulation-v2-frame-authoring-v2"
] = "joint-agent-articulation-v2-frame-authoring-v2"
ARTICULATION_V2_FRAME_AUTHORING_SCHEMA_VERSION: Literal[
    "joint-agent-articulation-v2-frame-authoring-v3"
] = "joint-agent-articulation-v2-frame-authoring-v3"
ARTICULATION_V2_FRAME_AUTHORING_CONTRACT_SCHEMA_VERSION: Literal[
    "joint-agent-articulation-v2-frame-authoring-contract-v1"
] = "joint-agent-articulation-v2-frame-authoring-contract-v1"

_USD_FORMATS = frozenset({"usd", "usda", "usdc", "usdz"})
_SOURCE_BACKED_PROVENANCE_SOURCES = frozenset(
    {
        "accepted_manifest",
        "authored_metadata",
        "authored_reference",
        "source_metadata",
    }
)
_FRAME_TOLERANCE = 1e-5
_FRAME_COHERENCE_TOLERANCE = 1e-6
_READBACK_TOLERANCE = 1e-6
_MATRIX_TOLERANCE = 1e-9
_JOINT_SCOPE_NAME = "Joints"
_AUTHORING_VERSION_V2: Literal["joint-agent-articulation-v2-frame-author-v2"] = (
    "joint-agent-articulation-v2-frame-author-v2"
)
_AUTHORING_VERSION: Literal["joint-agent-articulation-v2-frame-author-v3"] = (
    "joint-agent-articulation-v2-frame-author-v3"
)
_AGGREGATE_CORE_AUTHOR_ERROR_CODES = {
    code: f"articulation_v2_{code}"
    for code in (
        "aggregate_authored_path_collision",
        "aggregate_body_collision",
        "aggregate_body_creation_failed",
        "aggregate_composition_arc_unsupported",
        "aggregate_dependency_authorship_unsupported",
        "aggregate_edit_target_mismatch",
        "aggregate_instance_unsupported",
        "aggregate_member_changed",
        "aggregate_member_world_transform_changed",
        "aggregate_metadata_authoring_failed",
        "aggregate_namespace_edit_failed",
        "aggregate_namespace_edit_rejected",
        "aggregate_rollback_snapshot_failed",
        "aggregate_root_not_editable",
        "aggregate_source_missing",
        "aggregate_source_parent_mismatch",
        "aggregate_subtree_scan_limit_exceeded",
        "aggregate_top_level_unsupported",
        "aggregate_variant_unsupported",
        "authored_aggregate_mismatch",
        "existing_link_missing",
        "invalid_stage",
        "openusd_unavailable",
        "rigid_link_cross_link_invalid",
    )
}
_MAX_CONTRACT_BYTES = 8 * 1024 * 1024
_SHA256_LENGTH = 64
_AGGREGATE_TARGET_SCAN_LIMIT = 1_000_000

type ProvenanceSource = Literal[
    "accepted_manifest",
    "authored_metadata",
    "authored_reference",
    "source_metadata",
]
type Vector3 = tuple[float, float, float]
type QuaternionWxyz = tuple[float, float, float, float]
type _FrameConstraintKind = Literal["revolute", "prismatic", "spherical"]
type _FixedDistanceConstraint = FixedConstraintV2 | DistanceConstraintV2
type FixedDistanceTopologyDecision = Literal[
    "explicit_two_body_constraint",
    "co_rigid_aggregation",
    "unresolved",
]

_SOURCE_FRAME_PROPERTY_BY_FIELD: Mapping[str, tuple[str, ...]] = {
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
}


class ArticulationV2FrameAuthoringError(ValueError):
    """Fail-closed v2 frame promotion/authoring error with a stable code."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


class _ResultModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class OpaqueArtifactIdentityV1(_ResultModel):
    """Provider-neutral identity and admission budget for captured bytes."""

    uri: str = Field(min_length=1)
    sha256: str
    size_bytes: int = Field(ge=0)
    max_bytes: int = Field(ge=0)

    @field_validator("uri")
    @classmethod
    def _valid_uri(cls, value: str) -> str:
        if not value.strip() or "\x00" in value:
            raise ValueError("uri must be nonblank")
        return value

    @field_validator("sha256")
    @classmethod
    def _valid_sha256(cls, value: str) -> str:
        if (
            len(value) != _SHA256_LENGTH
            or value.lower() != value
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError("sha256 must be lowercase 64-character hexadecimal")
        return value

    @model_validator(mode="after")
    def _budget_contains_size(self) -> OpaqueArtifactIdentityV1:
        if self.size_bytes > self.max_bytes:
            raise ValueError("size_bytes exceeds max_bytes")
        return self

    def to_request(self) -> OpaqueArtifactRequest:
        return OpaqueArtifactRequest(
            uri=self.uri,
            sha256=self.sha256,
            size_bytes=self.size_bytes,
            max_bytes=self.max_bytes,
        )

    @classmethod
    def from_request(
        cls,
        request: OpaqueArtifactRequest,
    ) -> OpaqueArtifactIdentityV1:
        return cls(
            uri=request.uri,
            sha256=request.sha256,
            size_bytes=request.size_bytes,
            max_bytes=request.max_bytes,
        )


class CapturedUsdArtifactBindingV1(_ResultModel):
    """One exact opaque file plus its composed USD identity."""

    capture: OpaqueArtifactIdentityV1
    format: Literal["usd", "usda", "usdc", "usdz"]
    usd_identity: ArtifactIdentityV1

    @model_validator(mode="after")
    def _identities_agree(self) -> CapturedUsdArtifactBindingV1:
        if (
            self.capture.uri != self.usd_identity.uri
            or self.capture.sha256 != self.usd_identity.root_sha256
        ):
            raise ValueError(
                "capture and composed USD identity must bind the same URI/root bytes"
            )
        return self


class SourceJointBindingV1(_ResultModel):
    """Complete source selector plus any mandatory topology decision."""

    joint_id: str = Field(min_length=1)
    source_joint_path: str = Field(min_length=1)
    topology_decision: Literal["explicit_two_body_constraint"] | None = None

    @field_validator("joint_id", "source_joint_path")
    @classmethod
    def _nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("joint binding fields must be nonblank")
        return value


class ArticulationV2FrameAuthoringContractV1(_ResultModel):
    """Canonical captured authoring document with explicit artifact roles."""

    schema_version: Literal["joint-agent-articulation-v2-frame-authoring-contract-v1"]
    target: CapturedUsdArtifactBindingV1
    source_reference: CapturedUsdArtifactBindingV1
    source_joint_bindings: tuple[SourceJointBindingV1, ...]
    articulation: ArticulationContractV2

    @field_validator("source_joint_bindings")
    @classmethod
    def _canonical_bindings(
        cls,
        value: tuple[SourceJointBindingV1, ...],
    ) -> tuple[SourceJointBindingV1, ...]:
        joint_ids = tuple(item.joint_id for item in value)
        source_paths = tuple(item.source_joint_path for item in value)
        if len(joint_ids) != len(set(joint_ids)):
            raise ValueError("source joint bindings must have unique joint_id values")
        if len(source_paths) != len(set(source_paths)):
            raise ValueError(
                "source joint bindings must have unique source_joint_path values"
            )
        return tuple(sorted(value, key=lambda item: item.joint_id))

    @model_validator(mode="after")
    def _artifact_roles_are_distinct(self) -> ArticulationV2FrameAuthoringContractV1:
        if self.target.capture == self.source_reference.capture:
            raise ValueError(
                "target and source_reference must be distinct captured artifacts"
            )
        return self


def canonical_frame_authoring_contract_bytes(
    contract: ArticulationV2FrameAuthoringContractV1,
) -> bytes:
    """Return the sole accepted canonical JSON encoding for a v2 authoring contract."""

    if type(contract) is not ArticulationV2FrameAuthoringContractV1:
        raise TypeError(
            "contract must be an exact ArticulationV2FrameAuthoringContractV1"
        )
    adapter = TypeAdapter(ArticulationV2FrameAuthoringContractV1)
    validated = adapter.validate_json(
        json.dumps(
            adapter.dump_python(contract, mode="json", exclude_none=True),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ),
        strict=True,
    )
    return json.dumps(
        adapter.dump_python(validated, mode="json", exclude_none=True),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


class SourceBackedAttachmentFramesV2(_ResultModel):
    """Canonical v2 frame bundle extracted from one exact source USD joint."""

    source_artifact: ArtifactIdentityV1
    source_joint_path: str
    body0_prim_path: str
    body1_prim_path: str
    attachments: ExplicitAttachmentFramesV2
    field_evidence: tuple[FieldEvidenceV2, ...]


class ConstraintPolicyDiagnosticV2(_ResultModel):
    """One deterministic fixed-vs-aggregation or distance admission decision."""

    code: Literal[
        "articulation_v2_explicit_fixed_constraint_selected",
        "articulation_v2_explicit_distance_constraint_selected",
    ]
    decision: Literal["explicit_two_body_constraint"]
    detail: str = Field(min_length=1)


class SourceBackedFixedDistanceConstraintV2(_ResultModel):
    """Exact captured source facts ready for fixed/distance record promotion."""

    source_artifact: ArtifactIdentityV1
    source_joint_path: str
    body0_prim_path: str
    body1_prim_path: str
    attachments: ExplicitAttachmentFramesV2
    constraint: _FixedDistanceConstraint
    field_evidence: tuple[FieldEvidenceV2, ...]
    policy_diagnostics: tuple[ConstraintPolicyDiagnosticV2, ...]

    @model_validator(mode="after")
    def _policy_matches_constraint(
        self,
    ) -> SourceBackedFixedDistanceConstraintV2:
        expected_code = (
            "articulation_v2_explicit_fixed_constraint_selected"
            if isinstance(self.constraint, FixedConstraintV2)
            else "articulation_v2_explicit_distance_constraint_selected"
        )
        if (
            len(self.policy_diagnostics) != 1
            or self.policy_diagnostics[0].code != expected_code
        ):
            raise ValueError(
                "fixed/distance source facts require one matching explicit "
                "two-body policy diagnostic"
            )
        return self


class AuthoredAttachmentFrameReadbackV2(_ResultModel):
    """Strict saved-stage readback shape emitted by result schema v2."""

    joint_id: str
    joint_path: str
    body0_prim_path: str
    body1_prim_path: str
    attachments: ExplicitAttachmentFramesV2
    axis_stage: Vector3 | None
    field_evidence: tuple[FieldEvidenceV2, ...]
    evidence_sha256: str


class AuthoredAttachmentFrameReadbackV3(_ResultModel):
    """Constraint-bearing saved-stage readback emitted by result schema v3."""

    joint_id: str
    joint_path: str
    body0_prim_path: str
    body1_prim_path: str
    attachments: ExplicitAttachmentFramesV2
    constraint: JointConstraintV2
    axis_stage: Vector3 | None
    field_evidence: tuple[FieldEvidenceV2, ...]
    evidence_sha256: str


class ArticulationV2FrameAuthoringResult(_ResultModel):
    """Identity-bound saved-stage evidence, explicitly not qualification."""

    schema_version: Literal[
        "joint-agent-articulation-v2-frame-authoring-v2",
        "joint-agent-articulation-v2-frame-authoring-v3",
    ]
    authoring_version: Literal[
        "joint-agent-articulation-v2-frame-author-v2",
        "joint-agent-articulation-v2-frame-author-v3",
    ]
    contract_sha256: str
    contract_artifact: OpaqueArtifactIdentityV1
    target_artifact: CapturedUsdArtifactBindingV1
    source_reference_artifact: CapturedUsdArtifactBindingV1
    output_artifact: ArtifactIdentityV1
    meters_per_unit: float
    readback_tolerance: float
    joint_readbacks: tuple[
        AuthoredAttachmentFrameReadbackV2 | AuthoredAttachmentFrameReadbackV3,
        ...,
    ]
    saved_stage_readback_complete: Literal[True]
    static_qualified: Literal[False] = False
    dynamic_qualified: Literal[False] = False
    public_enabled: Literal[False] = False

    @model_validator(mode="after")
    def _result_generation_is_exact(
        self,
    ) -> ArticulationV2FrameAuthoringResult:
        if self.schema_version == ARTICULATION_V2_FRAME_AUTHORING_SCHEMA_VERSION_V2:
            if self.authoring_version != _AUTHORING_VERSION_V2 or any(
                type(item) is not AuthoredAttachmentFrameReadbackV2
                for item in self.joint_readbacks
            ):
                raise ValueError(
                    "frame-authoring v2 requires v2 authoring and strict "
                    "constraint-free v2 readbacks"
                )
        elif self.authoring_version != _AUTHORING_VERSION or any(
            type(item) is not AuthoredAttachmentFrameReadbackV3
            for item in self.joint_readbacks
        ):
            raise ValueError(
                "frame-authoring v3 requires v3 authoring and "
                "constraint-bearing v3 readbacks"
            )
        return self


@dataclass(frozen=True)
class _PreparedJoint:
    record: JointRecordV2
    frame_constraint_kind: _FrameConstraintKind | None
    joint_path: str
    body0_prim_path: str
    body1_prim_path: str
    local_pos0_stage: Vector3
    local_pos1_stage: Vector3
    local_rot0: QuaternionWxyz
    local_rot1: QuaternionWxyz
    revolute_limits_degrees: tuple[float, float] | None
    prismatic_limits_stage_units: tuple[float, float] | None
    distance_bounds_stage_units: tuple[float, float] | None
    evidence_json: str
    evidence_sha256: str


@dataclass(frozen=True)
class _SourceSnapshot:
    default_prim_path: str
    meters_per_unit: float
    up_axis: str
    root_layer_metadata: tuple[tuple[str, Any], ...]
    prims: tuple[tuple[Any, ...], ...]


@dataclass(frozen=True)
class _PreparedTargetWorkspace:
    editable_root: Path
    staged_output: Path
    package_member_order: tuple[str, ...] = ()
    package_root_member: str | None = None
    non_root_member_hashes: Mapping[str, str] | None = None


@dataclass(frozen=True)
class _RetainedUsdArtifact:
    """One source artifact and the immutable projection used for all reads."""

    captured_path: Path
    inspection: RetainedUsdArtifactInspection
    package_tree: RetainedUsdzPackageTree | None = None
    publication_snapshot_path: Path | None = None

    @property
    def source_path(self) -> Path:
        return self.inspection.source_path

    @property
    def stage_path(self) -> Path:
        return self.inspection.stage_path

    @property
    def package_manifest(self) -> UsdzPackageTreeManifest | None:
        if self.package_tree is None:
            return None
        return self.package_tree.manifest

    def require_publication_snapshot_path(self) -> Path:
        """Return the retained generated bytes selected for publication."""

        if self.publication_snapshot_path is None:
            raise RuntimeError("retained artifact has no publication snapshot")
        return self.publication_snapshot_path


def _detached_model[ModelT: BaseModel](
    value: object,
    model_type: type[ModelT],
    *,
    label: str,
    error_code: str,
) -> ModelT:
    """Return a canonical deep snapshot of one caller-owned Pydantic model."""

    if type(value) is not model_type:
        raise TypeError(f"{label} must be an exact {model_type.__name__}")
    # The exact runtime type gate above is the authority; this cast is only
    # static narrowing and must remain inert under optimized Python.
    admitted = cast(ModelT, value)  # type: ignore[redundant-cast]
    try:
        adapter = TypeAdapter(model_type)
        payload = json.dumps(
            adapter.dump_python(admitted, mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        return adapter.validate_json(payload, strict=True)
    except (TypeError, ValueError) as exc:
        _fail(
            error_code,
            f"{label} is not a valid canonical model snapshot: "
            f"{type(exc).__name__}: {exc}",
        )


def _detached_artifact_identity(
    value: object,
    *,
    label: str,
    error_code: str,
) -> ArtifactIdentityV1:
    """Snapshot exact identity scalars without retaining a caller model."""

    if type(value) is not ArtifactIdentityV1:
        raise TypeError(f"{label} must be an exact ArtifactIdentityV1")
    admitted = cast(
        ArtifactIdentityV1,
        value,
    )  # type: ignore[redundant-cast]
    try:
        uri = admitted.uri
        root_sha256 = admitted.root_sha256
        dependency_bundle_sha256 = admitted.dependency_bundle_sha256
        if (
            type(uri) is not str
            or type(root_sha256) is not str
            or (
                dependency_bundle_sha256 is not None
                and type(dependency_bundle_sha256) is not str
            )
        ):
            raise TypeError("identity fields must use exact scalar types")
        return ArtifactIdentityV1(
            uri=uri,
            root_sha256=root_sha256,
            dependency_bundle_sha256=dependency_bundle_sha256,
        )
    except (TypeError, ValueError) as exc:
        _fail(
            error_code,
            f"{label} is not a valid exact identity: {type(exc).__name__}: {exc}",
        )


def _detached_opaque_request(
    value: object,
    *,
    label: str,
    error_code: str,
) -> OpaqueArtifactRequest:
    """Snapshot exact request scalars before any resolver callback."""

    if type(value) is not OpaqueArtifactRequest:
        raise TypeError(f"{label} must be an exact OpaqueArtifactRequest")
    admitted = cast(
        OpaqueArtifactRequest,
        value,
    )  # type: ignore[redundant-cast]
    try:
        return OpaqueArtifactRequest(
            uri=admitted.uri,
            sha256=admitted.sha256,
            size_bytes=admitted.size_bytes,
            max_bytes=admitted.max_bytes,
        )
    except (TypeError, ValueError) as exc:
        _fail(
            error_code,
            f"{label} is not a valid exact request: {type(exc).__name__}: {exc}",
        )


def _detached_exact_string(value: object, *, label: str) -> str:
    """Admit only immutable built-in strings across an untrusted callback."""

    if type(value) is not str:
        raise TypeError(f"{label} must be an exact string")
    return value


def _detached_output_path(value: object, *, entry_cwd: Path) -> Path:
    """Snapshot and anchor one accepted path before any resolver callback."""

    if type(value) is str:
        raw_path = value
    else:
        if not isinstance(value, Path):
            raise TypeError("output_usd_path must be a string or Path")
        try:
            raw_path = os.fspath(value)
        except (TypeError, ValueError, OSError) as exc:
            raise TypeError("output_usd_path is not a valid path") from exc
        if type(raw_path) is not str:
            raise TypeError("output_usd_path must resolve to an exact string")
    lexical_path = Path(raw_path)
    if "\x00" in raw_path or lexical_path.name in {"", ".", ".."}:
        _fail(
            "articulation_v2_output_path_invalid",
            f"output path must name one file: {lexical_path}",
        )
    return Path(os.path.normpath(entry_cwd / lexical_path))


def promote_source_backed_attachment_frames_v2(
    *,
    source_request: OpaqueArtifactRequest,
    resolver: CapturedOpaqueArtifactResolver,
    source_artifact: ArtifactIdentityV1,
    source_format: Literal["usd", "usda", "usdc", "usdz"],
    source_joint_path: str,
    expected_body0_prim_path: str,
    expected_body1_prim_path: str,
    provenance_source: ProvenanceSource = "authored_reference",
    body0_link: LinkRecordV1 | None = None,
    body1_link: LinkRecordV1 | None = None,
    body0_member_prim_paths: tuple[str, ...] | None = None,
    body1_member_prim_paths: tuple[str, ...] | None = None,
) -> SourceBackedAttachmentFramesV2:
    """Extract one complete frame pair from an immutable source capture.

    The source must carry explicit ``localPos0/1`` and ``localRot0/1`` default
    opinions. Schema fallbacks, half frames, connections, value blocks, time
    samples, endpoint rewrites, and singular transforms fail closed. USD
    stage-unit positions become body-local meters; USD axis-token semantics are
    folded into the v2 canonical ``+X`` joint frame. This convenience result is
    detached evidence; the authorer independently recaptures and revalidates
    the reference. Supplying the complete directed link context re-expresses a
    source-member frame in its deterministic aggregate body so the bundle can
    pass through the normal joint-record promotion seam. Omitting that context
    preserves the original existing-body extraction behavior.
    """

    admitted_request = _detached_opaque_request(
        source_request,
        label="source_request",
        error_code="articulation_v2_source_request_invalid",
    )
    admitted_artifact = _detached_artifact_identity(
        source_artifact,
        label="source_artifact",
        error_code="articulation_v2_source_artifact_invalid",
    )
    admitted_source_format = cast(
        Literal["usd", "usda", "usdc", "usdz"],
        _detached_exact_string(source_format, label="source_format"),
    )
    if admitted_source_format not in _USD_FORMATS:
        _fail(
            "articulation_v2_source_format_invalid",
            f"unsupported source format: {admitted_source_format!r}",
        )
    admitted_joint_path = _detached_exact_string(
        source_joint_path,
        label="source_joint_path",
    )
    admitted_body0_path = _detached_exact_string(
        expected_body0_prim_path,
        label="expected_body0_prim_path",
    )
    admitted_body1_path = _detached_exact_string(
        expected_body1_prim_path,
        label="expected_body1_prim_path",
    )
    admitted_provenance_source = cast(
        ProvenanceSource,
        _detached_exact_string(provenance_source, label="provenance_source"),
    )
    if admitted_provenance_source not in _SOURCE_BACKED_PROVENANCE_SOURCES:
        _fail(
            "articulation_v2_source_provenance_invalid",
            f"unsupported source provenance: {admitted_provenance_source!r}",
        )
    link_context = (
        body0_link,
        body1_link,
        body0_member_prim_paths,
        body1_member_prim_paths,
    )
    if any(value is not None for value in link_context) and not all(
        value is not None for value in link_context
    ):
        _fail(
            "articulation_v2_source_link_context_incomplete",
            "aggregate-aware frame extraction requires both directed links "
            "and both exact member-path tuples",
        )
    admitted_link_context: (
        tuple[
            LinkRecordV1,
            LinkRecordV1,
            tuple[str, ...],
            tuple[str, ...],
        ]
        | None
    ) = None
    if all(value is not None for value in link_context):
        if (
            type(body0_member_prim_paths) is not tuple
            or type(body1_member_prim_paths) is not tuple
        ):
            raise TypeError("aggregate member paths must be exact tuples")
        admitted_body0_link = _detached_model(
            body0_link,
            LinkRecordV1,
            label="body0_link",
            error_code="articulation_v2_source_link_context_invalid",
        )
        admitted_body1_link = _detached_model(
            body1_link,
            LinkRecordV1,
            label="body1_link",
            error_code="articulation_v2_source_link_context_invalid",
        )
        admitted_body0_members = tuple(
            _detached_exact_string(
                value,
                label=f"body0_member_prim_paths[{index}]",
            )
            for index, value in enumerate(body0_member_prim_paths)
        )
        admitted_body1_members = tuple(
            _detached_exact_string(
                value,
                label=f"body1_member_prim_paths[{index}]",
            )
            for index, value in enumerate(body1_member_prim_paths)
        )
        try:
            plans = (
                _rigid_link_plan_from_contract_link(
                    admitted_body0_link,
                    admitted_body0_members,
                ),
                _rigid_link_plan_from_contract_link(
                    admitted_body1_link,
                    admitted_body1_members,
                ),
            )
            _shared_aggregate_core_callable(
                "world_understanding.functions.physics.joint_rigger.models",
                "_validate_rigid_link_cross_link_invariants",
            )(plans)
        except ArticulationV2FrameAuthoringError:
            raise
        except (TypeError, ValueError) as exc:
            _fail(
                "articulation_v2_source_link_context_invalid",
                f"directed link membership is inconsistent: {exc}",
            )
        admitted_link_context = (
            admitted_body0_link,
            admitted_body1_link,
            admitted_body0_members,
            admitted_body1_members,
        )
    binding = _binding_from_request(
        admitted_request,
        source_artifact=admitted_artifact,
        source_format=admitted_source_format,
    )
    with _capture_from_resolver(
        resolver,
        admitted_request,
        role="source_reference",
    ) as capture:
        with _private_workspace() as workspace:
            source_path = _materialize_capture(
                capture,
                workspace / f"source-reference.{admitted_source_format}",
                role="source_reference",
            )
            with _retain_exact_usd_input(
                source_path,
                binding,
                role="source_reference",
            ) as source_input:
                _require_self_contained_usd(
                    source_path,
                    role="source_reference",
                    retained=source_input,
                )
                stage = _open_stage(
                    source_input.stage_path,
                    label="captured source reference",
                )
                try:
                    result = _extract_source_frames_from_stage(
                        stage,
                        source_artifact=admitted_artifact,
                        source_joint_path=admitted_joint_path,
                        expected_body0_prim_path=admitted_body0_path,
                        expected_body1_prim_path=admitted_body1_path,
                        provenance_source=admitted_provenance_source,
                    )
                    if admitted_link_context is not None:
                        (
                            admitted_body0_link,
                            admitted_body1_link,
                            admitted_body0_members,
                            admitted_body1_members,
                        ) = admitted_link_context
                        for endpoint_label, source_endpoint, link, member_paths in (
                            (
                                "body0",
                                result.body0_prim_path,
                                admitted_body0_link,
                                admitted_body0_members,
                            ),
                            (
                                "body1",
                                result.body1_prim_path,
                                admitted_body1_link,
                                admitted_body1_members,
                            ),
                        ):
                            expected_paths = (
                                (link.body_prim_path,)
                                if link.body_authoring == "existing"
                                else member_paths
                            )
                            if source_endpoint not in expected_paths:
                                _fail(
                                    "articulation_v2_source_endpoint_membership_conflict",
                                    f"source joint {admitted_joint_path!r} "
                                    f"{endpoint_label} endpoint "
                                    f"{source_endpoint!r} is not an exact member "
                                    f"of link {link.link_id!r}: "
                                    f"{sorted(expected_paths)}",
                                )
                        result = SourceBackedAttachmentFramesV2(
                            source_artifact=result.source_artifact,
                            source_joint_path=result.source_joint_path,
                            body0_prim_path=admitted_body0_link.body_prim_path,
                            body1_prim_path=admitted_body1_link.body_prim_path,
                            attachments=ExplicitAttachmentFramesV2(
                                kind="explicit",
                                body0=_attachment_for_authored_link(
                                    stage,
                                    source_endpoint_path=result.body0_prim_path,
                                    source_frame=result.attachments.body0,
                                    link=admitted_body0_link,
                                    joint_id=admitted_joint_path,
                                    endpoint_label="body0",
                                ),
                                body1=_attachment_for_authored_link(
                                    stage,
                                    source_endpoint_path=result.body1_prim_path,
                                    source_frame=result.attachments.body1,
                                    link=admitted_body1_link,
                                    joint_id=admitted_joint_path,
                                    endpoint_label="body1",
                                ),
                            ),
                            field_evidence=result.field_evidence,
                        )
                finally:
                    del stage
                capture.require_intact()
                _require_exact_usd_binding(
                    source_path,
                    binding,
                    role="source_reference",
                    retained=source_input,
                )
                return result


def promote_source_backed_fixed_distance_constraint_v2(
    *,
    source_request: OpaqueArtifactRequest,
    resolver: CapturedOpaqueArtifactResolver,
    source_artifact: ArtifactIdentityV1,
    source_format: Literal["usd", "usda", "usdc", "usdz"],
    source_joint_path: str,
    expected_body0_prim_path: str,
    expected_body1_prim_path: str,
    expected_constraint_kind: Literal["fixed", "distance"],
    topology_decision: FixedDistanceTopologyDecision,
    provenance_source: ProvenanceSource = "authored_reference",
) -> SourceBackedFixedDistanceConstraintV2:
    """Promote one exact captured fixed/distance joint into v2 facts.

    Fixed admission is an explicit two-body policy decision; it never turns a
    model ``fixed`` hint into permission to split a co-rigid link. Both schemas
    require exact static endpoint frames. Distance additionally requires a
    complete authored static bound pair. Raw USD and USDZ use the same retained
    capture, dependency-closure, and mutation gates as mobile frame promotion.
    """

    admitted_request = _detached_opaque_request(
        source_request,
        label="source_request",
        error_code="articulation_v2_source_request_invalid",
    )
    admitted_artifact = _detached_artifact_identity(
        source_artifact,
        label="source_artifact",
        error_code="articulation_v2_source_artifact_invalid",
    )
    admitted_source_format = cast(
        Literal["usd", "usda", "usdc", "usdz"],
        _detached_exact_string(source_format, label="source_format"),
    )
    if admitted_source_format not in _USD_FORMATS:
        _fail(
            "articulation_v2_source_format_invalid",
            f"unsupported source format: {admitted_source_format!r}",
        )
    admitted_joint_path = _detached_exact_string(
        source_joint_path,
        label="source_joint_path",
    )
    admitted_body0_path = _detached_exact_string(
        expected_body0_prim_path,
        label="expected_body0_prim_path",
    )
    admitted_body1_path = _detached_exact_string(
        expected_body1_prim_path,
        label="expected_body1_prim_path",
    )
    admitted_constraint_kind = cast(
        Literal["fixed", "distance"],
        _detached_exact_string(
            expected_constraint_kind,
            label="expected_constraint_kind",
        ),
    )
    admitted_topology_decision = cast(
        FixedDistanceTopologyDecision,
        _detached_exact_string(topology_decision, label="topology_decision"),
    )
    if admitted_constraint_kind not in {"fixed", "distance"}:
        _fail(
            "articulation_v2_source_constraint_kind_invalid",
            f"unsupported source constraint kind: {admitted_constraint_kind!r}",
        )
    if admitted_topology_decision == "co_rigid_aggregation":
        _fail(
            "articulation_v2_fixed_aggregation_evidence_conflict",
            "co-rigid aggregation evidence cannot be promoted as an explicit "
            "two-body constraint",
        )
    if admitted_topology_decision != "explicit_two_body_constraint":
        _fail(
            "articulation_v2_fixed_topology_decision_unresolved",
            "fixed/distance promotion requires an explicit two-body constraint "
            "decision",
        )
    admitted_provenance_source = cast(
        ProvenanceSource,
        _detached_exact_string(provenance_source, label="provenance_source"),
    )
    if admitted_provenance_source not in _SOURCE_BACKED_PROVENANCE_SOURCES:
        _fail(
            "articulation_v2_source_provenance_invalid",
            f"unsupported source provenance: {admitted_provenance_source!r}",
        )

    binding = _binding_from_request(
        admitted_request,
        source_artifact=admitted_artifact,
        source_format=admitted_source_format,
    )
    with _capture_from_resolver(
        resolver,
        admitted_request,
        role="source_reference",
    ) as capture:
        with _private_workspace() as workspace:
            source_path = _materialize_capture(
                capture,
                workspace / f"source-reference.{admitted_source_format}",
                role="source_reference",
            )
            with _retain_exact_usd_input(
                source_path,
                binding,
                role="source_reference",
            ) as source_input:
                _require_self_contained_usd(
                    source_path,
                    role="source_reference",
                    retained=source_input,
                )
                stage = _open_stage(
                    source_input.stage_path,
                    label="captured fixed/distance source reference",
                )
                try:
                    result = _extract_source_fixed_distance_from_stage(
                        stage,
                        source_artifact=admitted_artifact,
                        source_joint_path=admitted_joint_path,
                        expected_body0_prim_path=admitted_body0_path,
                        expected_body1_prim_path=admitted_body1_path,
                        expected_constraint_kind=admitted_constraint_kind,
                        provenance_source=admitted_provenance_source,
                    )
                finally:
                    del stage
                capture.require_intact()
                _require_exact_usd_binding(
                    source_path,
                    binding,
                    role="source_reference",
                    retained=source_input,
                )
                return result


def _extract_source_fixed_distance_from_stage(
    stage: Any,
    *,
    source_artifact: ArtifactIdentityV1,
    source_joint_path: str,
    expected_body0_prim_path: str,
    expected_body1_prim_path: str,
    expected_constraint_kind: Literal["fixed", "distance"],
    provenance_source: ProvenanceSource,
) -> SourceBackedFixedDistanceConstraintV2:
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    prim = _require_source_joint_prim(
        stage,
        source_joint_path,
        label="fixed/distance source joint",
    )
    schemas: Mapping[str, Any] = {
        "fixed": UsdPhysics.FixedJoint,
        "distance": UsdPhysics.DistanceJoint,
    }
    schema = schemas[expected_constraint_kind]
    if not prim.IsA(schema):
        _fail(
            "articulation_v2_source_constraint_kind_conflict",
            f"{source_joint_path!r} is not an authored "
            f"{expected_constraint_kind} joint",
        )
    joint = schema(prim)
    body0_path = _single_relationship_target(
        joint.GetBody0Rel(),
        field="physics:body0",
    )
    body1_path = _single_relationship_target(
        joint.GetBody1Rel(),
        field="physics:body1",
    )
    if body0_path == body1_path:
        _fail(
            "articulation_v2_source_same_body_constraint",
            "fixed/distance source endpoints must be distinct",
        )
    if body0_path != expected_body0_prim_path or body1_path != expected_body1_prim_path:
        _fail(
            "articulation_v2_source_endpoint_conflict",
            "fixed/distance source endpoints do not match the accepted "
            "directed pair: "
            f"observed=({body0_path}, {body1_path}), "
            f"expected=({expected_body0_prim_path}, "
            f"{expected_body1_prim_path})",
        )
    body0 = _require_endpoint(
        stage,
        body0_path,
        label="fixed/distance source body0",
        Sdf=Sdf,
        UsdGeom=UsdGeom,
    )
    body1 = _require_endpoint(
        stage,
        body1_path,
        label="fixed/distance source body1",
        Sdf=Sdf,
        UsdGeom=UsdGeom,
    )
    _require_distinct_rigid_body_prims(body0, body1, joint_id=None)
    for index, body in enumerate((body0, body1)):
        _require_static_endpoint_transform(
            body,
            label=f"fixed/distance source body{index}",
        )
        _require_invertible_world_transform(
            stage,
            body,
            label=f"fixed/distance source body{index}",
        )
    _require_no_conflicting_source_edge(
        stage,
        target_path=source_joint_path,
        body0_path=body0_path,
        body1_path=body1_path,
        target_kind=expected_constraint_kind,
        Usd=Usd,
        UsdPhysics=UsdPhysics,
    )
    _require_no_unsupported_fixed_distance_properties(
        prim,
        joint_kind=expected_constraint_kind,
    )
    _require_exact_fixed_distance_metadata(
        prim,
        joint_kind=expected_constraint_kind,
        expected_custom_data=None,
    )

    meters_per_unit = _meters_per_unit(stage)
    local_pos0 = _require_static_authored_vector(
        joint.GetLocalPos0Attr(),
        field="physics:localPos0",
    )
    local_pos1 = _require_static_authored_vector(
        joint.GetLocalPos1Attr(),
        field="physics:localPos1",
    )
    local_rot0 = _require_static_authored_quaternion(
        joint.GetLocalRot0Attr(),
        field="physics:localRot0",
    )
    local_rot1 = _require_static_authored_quaternion(
        joint.GetLocalRot1Attr(),
        field="physics:localRot1",
    )
    attachments = ExplicitAttachmentFramesV2(
        kind="explicit",
        body0=AttachmentFrameV2(
            position_meters=_multiply_vector(local_pos0, meters_per_unit),
            orientation_wxyz=local_rot0,
        ),
        body1=AttachmentFrameV2(
            position_meters=_multiply_vector(local_pos1, meters_per_unit),
            orientation_wxyz=local_rot1,
        ),
    )
    world_frame0 = _validate_endpoint_frame(
        stage,
        body0,
        attachments.body0,
        meters_per_unit=meters_per_unit,
        label="fixed/distance source body0",
    )
    world_frame1 = _validate_endpoint_frame(
        stage,
        body1,
        attachments.body1,
        meters_per_unit=meters_per_unit,
        label="fixed/distance source body1",
    )
    if expected_constraint_kind == "fixed":
        _require_fixed_frame_coherence(
            world_frame0=world_frame0,
            world_frame1=world_frame1,
            label=f"source joint {source_joint_path!r}",
            anchor_error_code="articulation_v2_source_endpoint_anchor_conflict",
            orientation_error_code=(
                "articulation_v2_source_endpoint_orientation_conflict"
            ),
        )

    if expected_constraint_kind == "fixed":
        constraint: _FixedDistanceConstraint = FixedConstraintV2(kind="fixed")
    else:
        minimum, maximum = _require_source_distance_bounds(joint)
        minimum_meters, maximum_meters = _distance_bounds_stage_units_to_meters(
            minimum,
            maximum,
            meters_per_unit=meters_per_unit,
        )
        constraint = DistanceConstraintV2(
            kind="distance",
            minimum_meters=minimum_meters,
            maximum_meters=maximum_meters,
        )
    evidence = tuple(
        sorted(
            (
                *_source_frame_evidence(
                    source_artifact=source_artifact,
                    joint_path=source_joint_path,
                    provenance_source=provenance_source,
                    axis_token_normalized=False,
                ),
                *_source_fixed_distance_evidence(
                    source_artifact=source_artifact,
                    joint_path=source_joint_path,
                    constraint=constraint,
                    provenance_source=provenance_source,
                ),
            ),
            key=lambda item: item.field,
        )
    )
    decision_code: Literal[
        "articulation_v2_explicit_fixed_constraint_selected",
        "articulation_v2_explicit_distance_constraint_selected",
    ] = (
        "articulation_v2_explicit_fixed_constraint_selected"
        if isinstance(constraint, FixedConstraintV2)
        else "articulation_v2_explicit_distance_constraint_selected"
    )
    return SourceBackedFixedDistanceConstraintV2(
        source_artifact=source_artifact,
        source_joint_path=source_joint_path,
        body0_prim_path=body0_path,
        body1_prim_path=body1_path,
        attachments=attachments,
        constraint=constraint,
        field_evidence=evidence,
        policy_diagnostics=(
            ConstraintPolicyDiagnosticV2(
                code=decision_code,
                decision="explicit_two_body_constraint",
                detail=(
                    "Exact captured USD constraint evidence keeps two distinct "
                    "rigid bodies; co-rigid aggregation was not selected."
                ),
            ),
        ),
    )


def _extract_source_frames_from_stage(
    stage: Any,
    *,
    source_artifact: ArtifactIdentityV1,
    source_joint_path: str,
    expected_body0_prim_path: str,
    expected_body1_prim_path: str,
    provenance_source: ProvenanceSource,
) -> SourceBackedAttachmentFramesV2:
    from pxr import Sdf, UsdGeom, UsdPhysics

    joint_prim = _require_source_joint_prim(
        stage,
        source_joint_path,
        label="source joint",
    )
    if not joint_prim.IsA(UsdPhysics.Joint):
        _fail(
            "articulation_v2_source_joint_missing",
            "source joint does not resolve to a USD physics joint: "
            f"{source_joint_path}",
        )
    constraint_kind: _FrameConstraintKind
    if joint_prim.IsA(UsdPhysics.RevoluteJoint):
        constraint_kind = "revolute"
    elif joint_prim.IsA(UsdPhysics.PrismaticJoint):
        constraint_kind = "prismatic"
    elif joint_prim.IsA(UsdPhysics.SphericalJoint):
        constraint_kind = "spherical"
    else:
        _fail(
            "articulation_v2_source_schema_unsupported",
            "source joint must be revolute, prismatic, or spherical",
        )
    if constraint_kind == "spherical":
        _require_free_spherical_source_controls_absent(joint_prim)
        axis_token = "X"
    else:
        axis_token = _source_axis_token(joint_prim)

    joint = UsdPhysics.Joint(joint_prim)
    body0_path = _single_relationship_target(
        joint.GetBody0Rel(),
        field="physics:body0",
    )
    body1_path = _single_relationship_target(
        joint.GetBody1Rel(),
        field="physics:body1",
    )
    if body0_path != expected_body0_prim_path or body1_path != expected_body1_prim_path:
        _fail(
            "articulation_v2_source_endpoint_conflict",
            "source joint endpoints do not match the accepted endpoint pair: "
            f"observed=({body0_path}, {body1_path}), "
            f"expected=({expected_body0_prim_path}, "
            f"{expected_body1_prim_path})",
        )
    if body0_path == body1_path:
        _fail(
            "articulation_v2_source_endpoint_conflict",
            "source joint endpoints must be distinct",
        )

    body0 = _require_endpoint(
        stage,
        body0_path,
        label="source joint body0",
        Sdf=Sdf,
        UsdGeom=UsdGeom,
    )
    body1 = _require_endpoint(
        stage,
        body1_path,
        label="source joint body1",
        Sdf=Sdf,
        UsdGeom=UsdGeom,
    )
    _require_static_endpoint_transform(body0, label="source joint body0")
    _require_static_endpoint_transform(body1, label="source joint body1")
    _require_invertible_world_transform(stage, body0, label="source joint body0")
    _require_invertible_world_transform(stage, body1, label="source joint body1")

    meters_per_unit = _meters_per_unit(stage)
    local_pos0 = _require_static_authored_vector(
        joint.GetLocalPos0Attr(),
        field="physics:localPos0",
    )
    local_pos1 = _require_static_authored_vector(
        joint.GetLocalPos1Attr(),
        field="physics:localPos1",
    )
    local_rot0 = _require_static_authored_quaternion(
        joint.GetLocalRot0Attr(),
        field="physics:localRot0",
    )
    local_rot1 = _require_static_authored_quaternion(
        joint.GetLocalRot1Attr(),
        field="physics:localRot1",
    )
    attachments = ExplicitAttachmentFramesV2(
        kind="explicit",
        body0=AttachmentFrameV2(
            position_meters=_multiply_vector(local_pos0, meters_per_unit),
            orientation_wxyz=_canonical_frame_orientation(
                local_rot0,
                axis_token=axis_token,
            ),
        ),
        body1=AttachmentFrameV2(
            position_meters=_multiply_vector(local_pos1, meters_per_unit),
            orientation_wxyz=_canonical_frame_orientation(
                local_rot1,
                axis_token=axis_token,
            ),
        ),
    )
    world_frame0 = _validate_endpoint_frame(
        stage,
        body0,
        attachments.body0,
        meters_per_unit=meters_per_unit,
        label="source joint body0",
    )
    world_frame1 = _validate_endpoint_frame(
        stage,
        body1,
        attachments.body1,
        meters_per_unit=meters_per_unit,
        label="source joint body1",
    )
    _require_full_frame_coherence(
        constraint_kind=constraint_kind,
        world_frame0=world_frame0,
        world_frame1=world_frame1,
        expected_axis=None,
        label=f"source joint {source_joint_path!r}",
        axis_error_code="articulation_v2_source_endpoint_frame_conflict",
        anchor_error_code="articulation_v2_source_endpoint_anchor_conflict",
        transverse_anchor_error_code=(
            "articulation_v2_transverse_anchor_frame_conflict"
        ),
    )
    evidence = _source_frame_evidence(
        source_artifact=source_artifact,
        joint_path=source_joint_path,
        provenance_source=provenance_source,
    )
    return SourceBackedAttachmentFramesV2(
        source_artifact=source_artifact,
        source_joint_path=source_joint_path,
        body0_prim_path=body0_path,
        body1_prim_path=body1_path,
        attachments=attachments,
        field_evidence=evidence,
    )


def promote_joint_record_with_source_backed_frames_v2(
    record: JointRecordV2,
    *,
    body0_link: LinkRecordV1,
    body1_link: LinkRecordV1,
    frames: SourceBackedAttachmentFramesV2,
    declared_source_identities: tuple[ArtifactIdentityV1, ...],
) -> JointRecordV2:
    """Replace a diagnosed/default frame with one exact source-backed pair.

    This is the versioned promotion seam between source extraction and the v2
    document builder.  It preserves the joint id, directed link ids,
    constraint, and all non-frame evidence byte-for-byte at model level.  It
    replaces only endpoint/frame fields and marks the joint ready only when the
    complete resulting v2 record validates.

    Document-level diagnostics, summary counts, and status remain the caller's
    responsibility so a partial multi-joint promotion cannot silently make an
    asset ready.
    """

    admitted_record = _detached_model(
        record,
        JointRecordV2,
        label="record",
        error_code="articulation_v2_frame_promotion_input_invalid",
    )
    admitted_body0 = _detached_model(
        body0_link,
        LinkRecordV1,
        label="body0_link",
        error_code="articulation_v2_frame_promotion_input_invalid",
    )
    admitted_body1 = _detached_model(
        body1_link,
        LinkRecordV1,
        label="body1_link",
        error_code="articulation_v2_frame_promotion_input_invalid",
    )
    admitted_frames = _detached_model(
        frames,
        SourceBackedAttachmentFramesV2,
        label="frames",
        error_code="articulation_v2_frame_promotion_input_invalid",
    )
    if type(declared_source_identities) is not tuple:
        raise TypeError("declared_source_identities must be an exact tuple")
    admitted_identities = tuple(
        _detached_artifact_identity(
            identity,
            label=f"declared_source_identities[{index}]",
            error_code="articulation_v2_frame_promotion_input_invalid",
        )
        for index, identity in enumerate(declared_source_identities)
    )
    if admitted_frames.source_artifact not in admitted_identities:
        _fail(
            "articulation_v2_frame_promotion_source_undeclared",
            f"frame source {admitted_frames.source_artifact.uri!r} is not declared",
        )
    if (
        admitted_record.body0_link != admitted_body0.link_id
        or admitted_record.body1_link != admitted_body1.link_id
        or admitted_frames.body0_prim_path != admitted_body0.body_prim_path
        or admitted_frames.body1_prim_path != admitted_body1.body_prim_path
    ):
        _fail(
            "articulation_v2_frame_promotion_endpoint_conflict",
            "promoted frame endpoints do not exactly match the directed v2 "
            f"record: joint=({admitted_record.body0_link}, "
            f"{admitted_record.body1_link}), "
            f"links=({admitted_body0.link_id}, {admitted_body1.link_id}), "
            f"source=({admitted_frames.body0_prim_path}, "
            f"{admitted_frames.body1_prim_path})",
        )

    promoted_evidence = {
        item.field: item.model_dump(mode="python")
        for item in admitted_frames.field_evidence
    }
    required_promoted_fields = set(_SOURCE_FRAME_PROPERTY_BY_FIELD)
    if set(promoted_evidence) != required_promoted_fields:
        _fail(
            "articulation_v2_frame_promotion_evidence_incomplete",
            "source frame bundle does not exactly cover endpoint/frame fields: "
            f"observed={sorted(promoted_evidence)}, "
            f"expected={sorted(required_promoted_fields)}",
        )
    evidence_by_field = {
        item.field: item.model_dump(mode="python")
        for item in admitted_record.field_evidence
        if item.field not in required_promoted_fields
    }
    evidence_by_field.update(promoted_evidence)
    payload = admitted_record.model_dump(mode="python")
    payload["attachments"] = admitted_frames.attachments.model_dump(mode="python")
    payload["field_evidence"] = tuple(
        evidence_by_field[field] for field in sorted(evidence_by_field)
    )
    payload["review_status"] = "ready_for_rigger_input"
    try:
        return JointRecordV2.model_validate(payload)
    except (TypeError, ValueError) as exc:
        _fail(
            "articulation_v2_frame_promotion_incomplete",
            f"promoted joint is not ready: {type(exc).__name__}: {exc}",
        )


def promote_joint_record_with_source_backed_fixed_distance_v2(
    record: JointRecordV2,
    *,
    body0_link: LinkRecordV1,
    body1_link: LinkRecordV1,
    source: SourceBackedFixedDistanceConstraintV2,
    declared_source_identities: tuple[ArtifactIdentityV1, ...],
) -> JointRecordV2:
    """Replace one review fixed/distance hint with exact captured source facts."""

    admitted_record = _detached_model(
        record,
        JointRecordV2,
        label="record",
        error_code="articulation_v2_constraint_promotion_input_invalid",
    )
    admitted_body0 = _detached_model(
        body0_link,
        LinkRecordV1,
        label="body0_link",
        error_code="articulation_v2_constraint_promotion_input_invalid",
    )
    admitted_body1 = _detached_model(
        body1_link,
        LinkRecordV1,
        label="body1_link",
        error_code="articulation_v2_constraint_promotion_input_invalid",
    )
    admitted_source = _detached_model(
        source,
        SourceBackedFixedDistanceConstraintV2,
        label="source",
        error_code="articulation_v2_constraint_promotion_input_invalid",
    )
    if type(declared_source_identities) is not tuple:
        raise TypeError("declared_source_identities must be an exact tuple")
    admitted_identities = tuple(
        _detached_artifact_identity(
            identity,
            label=f"declared_source_identities[{index}]",
            error_code="articulation_v2_constraint_promotion_input_invalid",
        )
        for index, identity in enumerate(declared_source_identities)
    )
    if admitted_source.source_artifact not in admitted_identities:
        _fail(
            "articulation_v2_constraint_promotion_source_undeclared",
            f"constraint source {admitted_source.source_artifact.uri!r} "
            "is not declared",
        )
    if (
        admitted_record.body0_link != admitted_body0.link_id
        or admitted_record.body1_link != admitted_body1.link_id
        or admitted_source.body0_prim_path != admitted_body0.body_prim_path
        or admitted_source.body1_prim_path != admitted_body1.body_prim_path
    ):
        _fail(
            "articulation_v2_constraint_promotion_endpoint_conflict",
            "source-backed constraint endpoints do not exactly match the "
            f"directed v2 record {admitted_record.joint_id!r}",
        )
    if (
        admitted_body0.body_authoring != "existing"
        or admitted_body1.body_authoring != "existing"
    ):
        _fail(
            "articulation_v2_constraint_aggregate_link_unsupported",
            "fixed/distance promotion requires two existing rigid-body links; "
            "aggregate-link creation remains a separate capability",
        )
    # Canonical detachment revalidates JointRecordV2, whose own invariant rejects
    # equal link IDs. Physical endpoint identity is independently checked during
    # source extraction and target preparation.
    if admitted_record.constraint.kind != admitted_source.constraint.kind:
        _fail(
            "articulation_v2_constraint_promotion_kind_conflict",
            f"review hint {admitted_record.constraint.kind!r} conflicts with "
            f"exact source schema {admitted_source.constraint.kind!r}",
        )

    expected_fields = _expected_fixed_distance_source_fields(admitted_source.constraint)
    source_evidence = {
        item.field: item.model_dump(mode="python")
        for item in admitted_source.field_evidence
    }
    if set(source_evidence) != expected_fields:
        _fail(
            "articulation_v2_constraint_promotion_evidence_incomplete",
            "fixed/distance source evidence does not exactly cover applicable "
            f"fields: observed={sorted(source_evidence)}, "
            f"expected={sorted(expected_fields)}",
        )
    evidence_by_field = {
        item.field: item.model_dump(mode="python")
        for item in admitted_record.field_evidence
        if item.field not in expected_fields
        and not item.field.startswith("constraint.")
    }
    evidence_by_field.update(source_evidence)
    payload = admitted_record.model_dump(mode="python")
    payload["attachments"] = admitted_source.attachments.model_dump(mode="python")
    payload["constraint"] = admitted_source.constraint.model_dump(mode="python")
    payload["field_evidence"] = tuple(
        evidence_by_field[field] for field in sorted(evidence_by_field)
    )
    payload["review_status"] = "ready_for_rigger_input"
    try:
        return JointRecordV2.model_validate(payload)
    except (TypeError, ValueError) as exc:
        _fail(
            "articulation_v2_constraint_promotion_incomplete",
            f"promoted fixed/distance joint is not ready: {type(exc).__name__}: {exc}",
        )


def author_articulation_v2_attachment_frames(
    *,
    contract_request: OpaqueArtifactRequest,
    target_request: OpaqueArtifactRequest,
    source_reference_request: OpaqueArtifactRequest,
    contract_resolver: CapturedOpaqueArtifactResolver,
    target_resolver: CapturedOpaqueArtifactResolver,
    source_reference_resolver: CapturedOpaqueArtifactResolver,
    output_usd_path: str | Path,
    output_uri: str,
) -> ArticulationV2FrameAuthoringResult:
    """Author from a stripped target after recapturing its reviewed reference.

    The captured canonical contract declares two distinct roles. Target bytes
    are the sole publication base. Source-reference bytes are reopened only to
    prove each bound joint's schema, directed endpoints, attachment frames,
    axis, limits, and evidence locators. Provider-owned captures are detached
    and released before their bytes become observable here. The detached
    captures remain live through private validation, are revalidated, and all
    close successfully before descriptor-confined atomic publication begins.
    """

    entry_cwd = Path.cwd()
    admitted_contract_request = _detached_opaque_request(
        contract_request,
        label="contract_request",
        error_code="articulation_v2_contract_request_invalid",
    )
    if admitted_contract_request.size_bytes > _MAX_CONTRACT_BYTES:
        _fail(
            "articulation_v2_contract_too_large",
            f"captured authoring contract exceeds {_MAX_CONTRACT_BYTES} bytes",
        )
    admitted_target_request = _detached_opaque_request(
        target_request,
        label="target_request",
        error_code="articulation_v2_target_request_invalid",
    )
    admitted_reference_request = _detached_opaque_request(
        source_reference_request,
        label="source_reference_request",
        error_code="articulation_v2_source_reference_request_invalid",
    )
    admitted_output_path = _detached_output_path(
        output_usd_path,
        entry_cwd=entry_cwd,
    )
    for resolver_value, label in (
        (contract_resolver, "contract_resolver"),
        (target_resolver, "target_resolver"),
        (source_reference_resolver, "source_reference_resolver"),
    ):
        if not isinstance(resolver_value, CapturedOpaqueArtifactResolver):
            raise TypeError(f"{label} must implement CapturedOpaqueArtifactResolver")
    if type(output_uri) is not str:
        raise TypeError("output_uri must be an exact string")
    if not output_uri.strip() or "\x00" in output_uri:
        _fail(
            "articulation_v2_output_uri_invalid",
            "output_uri must be nonblank and contain no null byte",
        )

    with ExitStack() as lifetime:
        # Hold the lexically selected parent before any resolver can rename or
        # replace it. Every later existence check and the atomic publication
        # use this same directory descriptor.
        try:
            output_parent_descriptor = lifetime.enter_context(
                open_confined_directory(admitted_output_path.parent)
            )
        except (OSError, RuntimeError, ValueError) as exc:
            _fail(
                "articulation_v2_output_commit_failed",
                "could not retain output parent before capture: "
                f"{type(exc).__name__}: {exc}",
            )

        # Materialized inputs must outlive every retained capture/input
        # finalizer, including exceptional unwinds. Enter the workspace before
        # the nested capture stack so captures close before it is deleted.
        workspace = lifetime.enter_context(_private_workspace())
        captures = lifetime.enter_context(ExitStack())
        contract_capture = captures.enter_context(
            _capture_from_resolver(
                contract_resolver,
                admitted_contract_request,
                role="contract",
            )
        )
        with nullcontext(workspace) as workspace:
            contract_path = _materialize_capture(
                contract_capture,
                workspace / "authoring-contract.json",
                role="contract",
            )
            contract = _parse_canonical_authoring_contract(
                contract_path,
                admitted_contract_request,
            )
            _require_declared_request(
                admitted_target_request,
                contract.target.capture,
                role="target",
            )
            _require_declared_request(
                admitted_reference_request,
                contract.source_reference.capture,
                role="source_reference",
            )
            output_path = _require_output_path(
                admitted_output_path,
                target_format=contract.target.format,
                parent_descriptor=output_parent_descriptor,
            )

            target_capture = captures.enter_context(
                _capture_from_resolver(
                    target_resolver,
                    admitted_target_request,
                    role="target",
                )
            )
            reference_capture = captures.enter_context(
                _capture_from_resolver(
                    source_reference_resolver,
                    admitted_reference_request,
                    role="source_reference",
                )
            )
            target_path = _materialize_capture(
                target_capture,
                workspace / f"captured-target.{contract.target.format}",
                role="target",
            )
            reference_path = _materialize_capture(
                reference_capture,
                workspace
                / f"captured-source-reference.{contract.source_reference.format}",
                role="source_reference",
            )
            target_input = captures.enter_context(
                _retain_exact_usd_input(
                    target_path,
                    contract.target,
                    role="target",
                )
            )
            reference_input = captures.enter_context(
                _retain_exact_usd_input(
                    reference_path,
                    contract.source_reference,
                    role="source_reference",
                )
            )
            _require_self_contained_usd(
                target_path,
                role="target",
                retained=target_input,
            )
            _require_self_contained_usd(
                reference_path,
                role="source_reference",
                retained=reference_input,
            )
            _validate_authoring_contract(contract)
            rigid_link_plans = _aggregate_rigid_link_plans(contract.articulation)
            _revalidate_source_reference(contract, reference_input.stage_path)

            prepared_target = _prepare_target_workspace(
                target_input.stage_path,
                contract.target,
                workspace=workspace,
            )
            editable_root = prepared_target.editable_root
            staged_output = prepared_target.staged_output
            stage = _open_stage(editable_root, label="private captured target")
            try:
                preserved_base = _capture_source_snapshot(stage)
                if rigid_link_plans:
                    _require_aggregate_source_target_member_alignment(
                        contract,
                        target_stage=stage,
                        reference_path=reference_input.stage_path,
                    )
                    _require_target_aggregate_members_not_rigid_bodies(
                        stage,
                        rigid_link_plans,
                    )
                    _author_contract_aggregate_links(stage, rigid_link_plans)
                    _author_aggregate_rigid_body_schemas(stage, rigid_link_plans)
                    preserved_base = _remap_source_snapshot_for_aggregate_links(
                        preserved_base,
                        rigid_link_plans,
                    )
                prepared, scope_path, create_scope = _prepare_contract(
                    stage,
                    contract.articulation,
                    contract_sha256=admitted_contract_request.sha256,
                )
                _author_prepared_joints(
                    stage,
                    prepared,
                    scope_path=scope_path,
                    create_scope=create_scope,
                    contract_sha256=admitted_contract_request.sha256,
                )
                root_layer = stage.GetRootLayer()
                if not root_layer.Save():
                    _fail(
                        "articulation_v2_output_save_failed",
                        f"could not save private output USD: {editable_root}",
                    )
            finally:
                del stage

            if contract.target.format == "usdz":
                package_root = workspace / "editable-package"
                root_member = editable_root.relative_to(package_root)
                try:
                    write_usdz_package_from_directory(
                        package_root,
                        root_member,
                        staged_output,
                        member_order=prepared_target.package_member_order,
                    )
                except (FileExistsError, OSError, UsdzPackageError) as exc:
                    _fail(
                        "articulation_v2_usdz_packaging_failed",
                        f"could not package private target: "
                        f"{type(exc).__name__}: {exc}",
                    )

            # Publication is prepared from the retained snapshot only after
            # provider callbacks close. Its private destination file remains
            # unpublished until the output retention context's final source
            # recheck succeeds.
            with _publish_without_overwrite(
                output_path,
                parent_descriptor=output_parent_descriptor,
            ) as publish:
                with _retain_generated_usd_output(
                    staged_output,
                    uri=output_uri,
                ) as retained_output:
                    if contract.target.format == "usdz":
                        _require_preserved_package_members(
                            retained_output.stage_path,
                            prepared_target,
                            package_manifest=retained_output.package_manifest,
                        )

                    reopened = _open_stage(
                        retained_output.stage_path,
                        label="saved private output USD",
                    )
                    try:
                        if rigid_link_plans:
                            _validate_contract_aggregate_links(
                                reopened,
                                rigid_link_plans,
                            )
                            _validate_aggregate_rigid_body_schemas(
                                reopened,
                                rigid_link_plans,
                            )
                        _validate_source_snapshot(
                            preserved_base,
                            reopened,
                            allowed_additions={
                                *(item.joint_path for item in prepared),
                                *({scope_path} if create_scope else set()),
                                *(
                                    link.body_prim_path
                                    for link in rigid_link_plans
                                    if link.body_authoring == "aggregate"
                                ),
                            },
                        )
                        readbacks = _validate_saved_readback(
                            reopened,
                            prepared,
                            contract_sha256=admitted_contract_request.sha256,
                        )
                        meters_per_unit = _meters_per_unit(reopened)
                    finally:
                        del reopened
                    _require_self_contained_usd(
                        staged_output,
                        role="output",
                        retained=retained_output,
                    )

                    # Commit gate: re-prove all three retained captures, all
                    # three private materializations, the canonical contract,
                    # and the source joint mapping immediately before
                    # publication.
                    for capture in (
                        contract_capture,
                        target_capture,
                        reference_capture,
                    ):
                        capture.require_intact()
                    _parse_canonical_authoring_contract(
                        contract_path,
                        admitted_contract_request,
                    )
                    _require_exact_usd_binding(
                        target_path,
                        contract.target,
                        role="target",
                        retained=target_input,
                    )
                    _require_exact_usd_binding(
                        reference_path,
                        contract.source_reference,
                        role="source_reference",
                        retained=reference_input,
                    )
                    _revalidate_source_reference(
                        contract,
                        reference_input.stage_path,
                    )
                    _require_self_contained_usd(
                        staged_output,
                        role="output",
                        retained=retained_output,
                    )
                    output_artifact = retained_output.inspection.identity
                    retained_output.inspection.require_stage_unchanged()

                    # No provider callback may run after this point. Copy the
                    # exact retained bytes into an unpublished destination
                    # file; the outer context commits it only after the
                    # retained output's final source digest succeeds.
                    captures.close()
                    publish(
                        retained_output.require_publication_snapshot_path(),
                        output_artifact.root_sha256,
                    )

            return ArticulationV2FrameAuthoringResult(
                schema_version=ARTICULATION_V2_FRAME_AUTHORING_SCHEMA_VERSION,
                authoring_version=_AUTHORING_VERSION,
                contract_sha256=admitted_contract_request.sha256,
                contract_artifact=OpaqueArtifactIdentityV1.from_request(
                    admitted_contract_request
                ),
                target_artifact=contract.target,
                source_reference_artifact=contract.source_reference,
                output_artifact=output_artifact,
                meters_per_unit=meters_per_unit,
                readback_tolerance=_READBACK_TOLERANCE,
                joint_readbacks=readbacks,
                saved_stage_readback_complete=True,
                static_qualified=False,
                dynamic_qualified=False,
                public_enabled=False,
            )


def _binding_from_request(
    request: OpaqueArtifactRequest,
    *,
    source_artifact: ArtifactIdentityV1,
    source_format: Literal["usd", "usda", "usdc", "usdz"],
) -> CapturedUsdArtifactBindingV1:
    try:
        return CapturedUsdArtifactBindingV1(
            capture=OpaqueArtifactIdentityV1.from_request(request),
            format=source_format,
            usd_identity=source_artifact,
        )
    except (TypeError, ValueError) as exc:
        _fail(
            "articulation_v2_source_request_mismatch",
            f"source request does not bind the declared USD identity: {exc}",
        )


@contextmanager
def _capture_from_resolver(
    resolver: CapturedOpaqueArtifactResolver,
    request: OpaqueArtifactRequest,
    *,
    role: str,
) -> Iterator[CapturedOpaqueFile]:
    body_completed = False
    try:
        with capture_resolved_opaque_file(resolver, request) as captured:
            yield captured
            body_completed = True
    except ArticulationV2FrameAuthoringError:
        raise
    except BaseException as exc:
        failures = _capture_failure_members(exc)
        if any(
            isinstance(failure, CapturedArtifactError)
            and failure.code
            in {
                "capture_representation_invalid",
                "capture_type_mismatch",
            }
            for failure in failures
        ):
            _fail(
                f"articulation_v2_{role}_capture_protocol_invalid",
                _capture_failure_summary(exc),
            )
        if any(
            isinstance(failure, CapturedArtifactError)
            and not isinstance(failure, CapturedArtifactCleanupError)
            for failure in failures
        ):
            _capture_fail(role, "capture failed integrity or resolution", exc)
        if body_completed or any(
            isinstance(failure, CapturedArtifactCleanupError) for failure in failures
        ):
            _fail(
                f"articulation_v2_{role}_capture_cleanup_failed",
                _capture_failure_summary(exc),
            )
        _capture_fail(role, "capture failed integrity or resolution", exc)


def _capture_failure_members(exc: BaseException) -> tuple[BaseException, ...]:
    if isinstance(exc, BaseExceptionGroup):
        return tuple(
            member
            for child in exc.exceptions
            for member in _capture_failure_members(child)
        )
    return (exc,)


def _capture_failure_summary(exc: BaseException) -> str:
    details: list[str] = []
    for failure in _capture_failure_members(exc):
        underlying = (
            f"{failure.code}: {failure}"
            if isinstance(failure, CapturedArtifactError)
            else str(failure)
        )
        details.append(f"{type(failure).__name__}: {underlying}")
        details.extend(f"note: {note}" for note in getattr(failure, "__notes__", ()))
    return "; ".join(details)


def _capture_fail(role: str, detail: str, exc: BaseException) -> NoReturn:
    _fail(
        f"articulation_v2_{role}_capture_failed",
        f"{detail}: {_capture_failure_summary(exc)}",
    )


@contextmanager
def _private_workspace() -> Iterator[Path]:
    with tempfile.TemporaryDirectory(
        prefix="joint-articulation-v2-captured-"
    ) as directory:
        path = Path(directory)
        os.chmod(path, 0o700)
        yield path


def _materialize_capture(
    capture: CapturedOpaqueFile,
    destination: Path,
    *,
    role: str,
) -> Path:
    capture.require_intact()
    if destination.parent.stat().st_mode & 0o077:
        _fail(
            f"articulation_v2_{role}_materialization_insecure",
            "private capture workspace grants group or other permissions",
        )
    flags = (
        os.O_CREAT
        | os.O_EXCL
        | os.O_WRONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = -1
    digest = hashlib.sha256()
    size = 0
    try:
        descriptor = os.open(destination, flags, 0o600)
        for chunk in capture.iter_chunks():
            view = memoryview(chunk)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:  # pragma: no cover - regular-file invariant
                    raise OSError("private materialization write made no progress")
                digest.update(view[:written])
                size += written
                view = view[written:]
        os.fsync(descriptor)
    except BaseException as exc:
        destination.unlink(missing_ok=True)
        if isinstance(exc, ArticulationV2FrameAuthoringError):
            raise
        _fail(
            f"articulation_v2_{role}_materialization_failed",
            f"could not materialize captured bytes: {type(exc).__name__}: {exc}",
        )
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if size != capture.size_bytes or digest.hexdigest() != capture.sha256:
        destination.unlink(missing_ok=True)
        _fail(
            f"articulation_v2_{role}_materialization_mismatch",
            "private materialization differs from retained capture",
        )
    try:
        destination.chmod(0o400)
        observed = destination.stat(follow_symlinks=False)
    except OSError as exc:
        _fail(
            f"articulation_v2_{role}_materialization_mismatch",
            f"could not finalize private materialization: {exc}",
        )
    capture.require_intact()
    if (
        destination.is_symlink()
        or not destination.is_file()
        or observed.st_size != capture.size_bytes
    ):
        _fail(
            f"articulation_v2_{role}_materialization_mismatch",
            "private materialization metadata changed after finalization",
        )
    return destination


def _parse_canonical_authoring_contract(
    path: Path,
    request: OpaqueArtifactRequest,
) -> ArticulationV2FrameAuthoringContractV1:
    if request.size_bytes > _MAX_CONTRACT_BYTES:
        _fail(
            "articulation_v2_contract_too_large",
            f"captured authoring contract exceeds {_MAX_CONTRACT_BYTES} bytes",
        )
    if path.stat(follow_symlinks=False).st_size != request.size_bytes:
        _fail(
            "articulation_v2_contract_materialization_mismatch",
            "captured authoring contract size changed",
        )
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != request.sha256:
        _fail(
            "articulation_v2_contract_materialization_mismatch",
            "captured authoring contract digest changed",
        )
    try:
        # Parse once to reject invalid UTF-8/JSON before Pydantic's JSON path.
        json.loads(payload)
        contract = ArticulationV2FrameAuthoringContractV1.model_validate_json(
            payload,
            strict=True,
        )
    except (TypeError, ValueError) as exc:
        _fail(
            "articulation_v2_contract_invalid",
            f"captured authoring contract is invalid: {type(exc).__name__}: {exc}",
        )
    if canonical_frame_authoring_contract_bytes(contract) != payload:
        _fail(
            "articulation_v2_contract_not_canonical",
            "captured authoring contract bytes are not the canonical JSON encoding",
        )
    return contract


def _require_declared_request(
    request: OpaqueArtifactRequest,
    declared: OpaqueArtifactIdentityV1,
    *,
    role: str,
) -> None:
    if request != declared.to_request():
        _fail(
            f"articulation_v2_{role}_request_mismatch",
            "supplied request differs from the canonical contract declaration",
        )


@contextmanager
def _retain_exact_usd_input(
    path: Path,
    binding: CapturedUsdArtifactBindingV1,
    *,
    role: str,
) -> Iterator[_RetainedUsdArtifact]:
    """Retain one identity/inventory/projection for every admission gate."""

    if binding.format == "usdz":
        try:
            with retain_usdz_package_tree(path) as package_tree:
                with retain_usd_artifact_inspection(
                    package_tree.snapshot_path,
                    uri=binding.usd_identity.uri,
                    expected_root_sha256=package_tree.manifest.root_sha256,
                ) as inspection:
                    retained = _RetainedUsdArtifact(
                        captured_path=path.expanduser().resolve(strict=True),
                        inspection=inspection,
                        package_tree=package_tree,
                    )
                    _require_exact_usd_binding(
                        path,
                        binding,
                        role=role,
                        retained=retained,
                    )
                    yield retained
            return
        except ArticulationV2FrameAuthoringError:
            raise
        except UsdzPackageError as exc:
            _fail(
                f"articulation_v2_{role}_package_invalid",
                f"captured USDZ is not canonical and bounded: {exc}",
            )
        except JointRiggerContractError as exc:
            _fail(
                f"articulation_v2_{role}_identity_unavailable",
                f"could not inspect retained USDZ: {exc.code}: {exc}",
            )

    try:
        with retain_usd_artifact_inspection(
            path,
            uri=binding.usd_identity.uri,
            expected_root_sha256=binding.capture.sha256,
            recheck_source_content_on_exit=True,
        ) as inspection:
            retained = _RetainedUsdArtifact(
                captured_path=path.expanduser().resolve(strict=True),
                inspection=inspection,
            )
            _require_exact_usd_binding(
                path,
                binding,
                role=role,
                retained=retained,
            )
            yield retained
    except ArticulationV2FrameAuthoringError:
        raise
    except (JointRiggerContractError, OSError) as exc:
        detail = (
            f"{exc.code}: {exc}"
            if isinstance(exc, JointRiggerContractError)
            else f"{type(exc).__name__}: {exc}"
        )
        _fail(
            f"articulation_v2_{role}_identity_unavailable",
            f"could not inspect retained USD: {detail}",
        )


@contextmanager
def _retain_generated_usd_output(
    path: Path,
    *,
    uri: str,
) -> Iterator[_RetainedUsdArtifact]:
    """Retain the sole output proof until its final prepublication digest."""

    if path.suffix.lower() == ".usdz":
        try:
            with retain_usdz_package_tree(path) as package_tree:
                with retain_usd_artifact_inspection(
                    package_tree.snapshot_path,
                    uri=uri,
                    expected_root_sha256=package_tree.manifest.root_sha256,
                ) as inspection:
                    retained = _RetainedUsdArtifact(
                        captured_path=path.expanduser().resolve(strict=True),
                        inspection=inspection,
                        package_tree=package_tree,
                        publication_snapshot_path=package_tree.snapshot_path,
                    )
                    retained.inspection.require_stage_unchanged()
                    yield retained
            return
        except ArticulationV2FrameAuthoringError:
            raise
        except UsdzPackageError as exc:
            _fail(
                "articulation_v2_output_package_invalid",
                f"generated USDZ is not canonical and bounded: {exc}",
            )
        except JointRiggerContractError as exc:
            _fail(
                "articulation_v2_output_identity_unavailable",
                f"could not inspect retained output USDZ: {exc.code}: {exc}",
            )

    try:
        with _retain_raw_usd_snapshot(path) as (
            source_path,
            snapshot_path,
            root_sha256,
        ):
            with retain_usd_artifact_inspection(
                snapshot_path,
                uri=uri,
                expected_root_sha256=root_sha256,
            ) as inspection:
                retained = _RetainedUsdArtifact(
                    captured_path=source_path,
                    inspection=inspection,
                    publication_snapshot_path=snapshot_path,
                )
                retained.inspection.require_stage_unchanged()
                yield retained
    except ArticulationV2FrameAuthoringError:
        raise
    except (JointRiggerContractError, OSError) as exc:
        detail = (
            f"{exc.code}: {exc}"
            if isinstance(exc, JointRiggerContractError)
            else f"{type(exc).__name__}: {exc}"
        )
        _fail(
            "articulation_v2_output_identity_unavailable",
            f"could not inspect retained output USD: {detail}",
        )


def _require_exact_usd_binding(
    path: Path,
    binding: CapturedUsdArtifactBindingV1,
    *,
    role: str,
    package_manifest: UsdzPackageTreeManifest | None = None,
    retained: _RetainedUsdArtifact | None = None,
) -> UsdzPackageTreeManifest | None:
    expected_suffix = f".{binding.format}"
    if path.suffix.lower() != expected_suffix:
        _fail(
            f"articulation_v2_{role}_format_mismatch",
            f"materialized capture must use {expected_suffix}",
        )
    if path.is_symlink() or not path.is_file():
        _fail(
            f"articulation_v2_{role}_identity_mismatch",
            "materialized bytes differ from the canonical capture declaration",
        )
    if retained is not None:
        if retained.captured_path != path.expanduser().resolve(strict=True):
            _fail(
                f"articulation_v2_{role}_identity_mismatch",
                "retained USD inspection belongs to a different capture",
            )
        retained.inspection.require_stage_unchanged()
        if binding.format == "usdz":
            if retained.package_tree is None:
                _fail(
                    f"articulation_v2_{role}_identity_mismatch",
                    "retained USDZ inspection has no package-tree proof",
                )
            retained.package_tree.require_snapshot_unchanged()
            package_manifest = cast(
                UsdzPackageTreeManifest,
                retained.package_manifest,
            )
            if not _package_manifest_path_state_matches(path, package_manifest):
                _fail(
                    f"articulation_v2_{role}_identity_mismatch",
                    "captured USDZ differs from its immutable package-tree proof",
                )
            size_bytes = package_manifest.root_size_bytes
            sha256 = package_manifest.root_sha256
        else:
            if retained.package_tree is not None:
                _fail(
                    f"articulation_v2_{role}_identity_mismatch",
                    "raw USD inspection unexpectedly has a package-tree proof",
                )
            observed_state = path.stat(follow_symlinks=False)
            size_bytes = observed_state.st_size
            sha256 = retained.inspection.identity.root_sha256
        if size_bytes != binding.capture.size_bytes or sha256 != binding.capture.sha256:
            _fail(
                f"articulation_v2_{role}_identity_mismatch",
                "materialized bytes differ from the canonical capture declaration",
            )
        if retained.inspection.identity != binding.usd_identity:
            _fail(
                f"articulation_v2_{role}_identity_mismatch",
                "composed USD identity differs from the canonical contract",
            )
        return package_manifest

    if binding.format == "usdz":
        reused_package_manifest = package_manifest is not None
        if package_manifest is None:
            try:
                package_manifest = validate_usdz_package_tree(path)
            except UsdzPackageError as exc:
                _fail(
                    f"articulation_v2_{role}_package_invalid",
                    f"captured USDZ is not canonical and bounded: {exc}",
                )
        manifest_matches = (
            _package_manifest_matches_path(path, package_manifest)
            if reused_package_manifest
            else _package_manifest_path_state_matches(path, package_manifest)
        )
        if not manifest_matches:
            _fail(
                f"articulation_v2_{role}_identity_mismatch",
                "captured USDZ differs from its immutable package-tree proof",
            )
        size_bytes = package_manifest.root_size_bytes
        sha256 = package_manifest.root_sha256
    else:
        size_bytes = path.stat(follow_symlinks=False).st_size
        sha256 = _file_sha256(path)
    if size_bytes != binding.capture.size_bytes or sha256 != binding.capture.sha256:
        _fail(
            f"articulation_v2_{role}_identity_mismatch",
            "materialized bytes differ from the canonical capture declaration",
        )
    try:
        observed = identify_usd_artifact(path, uri=binding.usd_identity.uri)
    except Exception as exc:
        _fail(
            f"articulation_v2_{role}_identity_unavailable",
            f"could not identify composed USD: {type(exc).__name__}: {exc}",
        )
    if observed != binding.usd_identity:
        _fail(
            f"articulation_v2_{role}_identity_mismatch",
            "composed USD identity differs from the canonical contract",
        )
    if package_manifest is not None and not _package_manifest_path_state_matches(
        path,
        package_manifest,
    ):
        _fail(
            f"articulation_v2_{role}_identity_mismatch",
            "captured USDZ changed during composed-identity validation",
        )
    return package_manifest


def _require_self_contained_usd(
    path: Path,
    *,
    role: str,
    package_manifest: UsdzPackageTreeManifest | None = None,
    retained: _RetainedUsdArtifact | None = None,
) -> None:
    expected = (
        retained.source_path
        if retained is not None
        else path.expanduser().resolve(strict=True)
    )
    if retained is not None:
        if retained.captured_path != path.expanduser().resolve(strict=True):
            _fail(
                f"articulation_v2_{role}_dependency_closure_invalid",
                "retained USD inspection belongs to a different capture",
            )
        retained.inspection.require_stage_unchanged()
        if path.suffix.lower() == ".usdz" and retained.package_tree is None:
            _fail(
                f"articulation_v2_{role}_dependency_closure_invalid",
                "retained USDZ inspection has no package-tree proof",
            )
        package_manifest = retained.package_manifest
    if path.suffix.lower() == ".usdz":
        if package_manifest is None:
            try:
                package_manifest = validate_usdz_package_tree(expected)
            except UsdzPackageError as exc:
                _fail(
                    f"articulation_v2_{role}_dependency_closure_invalid",
                    f"could not prove bounded canonical package closure: {exc}",
                )
        if retained is not None:
            package_tree = cast(RetainedUsdzPackageTree, retained.package_tree)
            package_tree.require_snapshot_unchanged()
            manifest_matches = _package_manifest_path_state_matches(
                path,
                package_manifest,
            )
        else:
            manifest_matches = _package_manifest_path_state_matches(
                expected,
                package_manifest,
            )
        if not manifest_matches:
            _fail(
                f"articulation_v2_{role}_dependency_closure_invalid",
                "captured USDZ differs from its package-tree proof before "
                "dependency enumeration",
            )
    if retained is not None:
        dependencies = retained.inspection.dependencies
    else:
        try:
            dependencies = usd_dependency_inventory(path)
        except Exception as exc:
            _fail(
                f"articulation_v2_{role}_dependency_closure_invalid",
                "could not prove complete dependency closure: "
                f"{type(exc).__name__}: {exc}",
            )
    if package_manifest is not None and retained is None:
        if not _package_manifest_matches_path(expected, package_manifest):
            _fail(
                f"articulation_v2_{role}_dependency_closure_invalid",
                "captured USDZ changed after its package-tree proof was built",
            )
    canonical_root_count = sum(
        _is_canonical_captured_dependency(
            dependency,
            expected=expected,
            package_root=False,
            package_manifest=None,
        )
        for dependency in dependencies
    )
    rejected = sorted(
        f"{dependency.kind}:{dependency.identifier}"
        for dependency in dependencies
        if not _is_canonical_captured_dependency(
            dependency,
            expected=expected,
            package_root=path.suffix.lower() == ".usdz",
            package_manifest=package_manifest,
        )
    )
    if canonical_root_count != 1:
        rejected.append("<missing-or-ambiguous-canonical-root-layer>")
    if rejected:
        _fail(
            f"articulation_v2_{role}_external_dependency",
            "captured USD is not self-contained; every dependency must be the "
            "canonical root or a canonical package member chain rooted at the "
            "captured USDZ; opaque package dependencies are unsupported in "
            f"this bounded raw-authoring slice: {rejected}",
        )


def _package_manifest_matches_path(
    path: Path,
    manifest: UsdzPackageTreeManifest,
) -> bool:
    if not _package_manifest_path_state_matches(path, manifest):
        return False
    try:
        expected = path.expanduser().resolve(strict=True)
        return _file_sha256(expected) == manifest.root_sha256
    except OSError:
        return False


def _package_manifest_path_state_matches(
    path: Path,
    manifest: UsdzPackageTreeManifest,
) -> bool:
    try:
        expected = path.expanduser().resolve(strict=True)
        observed = os.stat(expected, follow_symlinks=False)
        return (
            expected == manifest.root_path
            and not path.is_symlink()
            and observed.st_dev == manifest.root_device
            and observed.st_ino == manifest.root_inode
            and observed.st_size == manifest.root_size_bytes
            and observed.st_mtime_ns == manifest.root_mtime_ns
            and observed.st_ctime_ns == manifest.root_ctime_ns
        )
    except OSError:
        return False


def _is_canonical_captured_dependency(
    dependency: UsdDependencyOpinion,
    *,
    expected: Path,
    package_root: bool,
    package_manifest: UsdzPackageTreeManifest | None,
) -> bool:
    """Return whether one complete dependency opinion stays in captured bytes."""

    if dependency.local_path is None or dependency.lexical_path is None:
        return False
    try:
        local_path = dependency.local_path.expanduser().resolve(strict=True)
    except OSError:
        return False
    lexical_path = Path(os.path.abspath(dependency.lexical_path.expanduser()))
    if local_path != expected or lexical_path != expected:
        return False

    if not dependency.package_relative:
        return (
            dependency.kind == "layer"
            and dependency.asset_identifier == str(expected)
            and dependency.package_outer_identifier is None
            and not dependency.package_members
        )
    if (
        not package_root
        or package_manifest is None
        or dependency.kind == "opaque_asset"
        or not dependency.package_members
        or (
            Path(dependency.package_members[-1]).suffix.lower()
            in OPAQUE_DEPENDENCY_EXTENSIONS
        )
        or dependency.package_outer_identifier != str(expected)
    ):
        return False
    if any(
        safe_usdz_member_name(member, allow_leading_slash=False) != member
        for member in dependency.package_members
    ):
        return False
    if any(
        Path(member).suffix.lower() != ".usdz"
        for member in dependency.package_members[:-1]
    ):
        return False

    from pxr import Ar

    inner = dependency.package_members[-1]
    for package_member in reversed(dependency.package_members[:-1]):
        inner = str(Ar.JoinPackageRelativePath(package_member, inner))
    reconstructed = str(Ar.JoinPackageRelativePath(str(expected), inner))
    return reconstructed == dependency.asset_identifier and (
        dependency.package_members in package_manifest.member_chains
        or dependency.package_members in package_manifest.package_chains
    )


def _member_paths_by_link(
    articulation: ArticulationContractV2,
) -> dict[str, tuple[str, ...]]:
    """Project the contract's exact prim membership without path inference."""

    link_ids = {
        record.link_id
        for record in articulation.records
        if isinstance(record, LinkRecordV1)
    }
    return {
        link_id: tuple(
            sorted(
                record.prim_path
                for record in articulation.records
                if isinstance(record, PrimRecordV1) and record.link_id == link_id
            )
        )
        for link_id in link_ids
    }


def _shared_aggregate_core_callable(
    module_name: str,
    symbol_name: str,
) -> Callable[..., Any]:
    """Admit one post-0.5 aggregate seam without raising the import floor."""

    try:
        symbol = getattr(import_module(module_name), symbol_name)
    except (AttributeError, ImportError) as exc:
        _fail(
            "articulation_v2_aggregate_dependency_unavailable",
            "the installed world-understanding core does not provide the exact "
            f"aggregate seam {module_name}.{symbol_name}: "
            f"{type(exc).__name__}: {exc}",
        )
    if not callable(symbol):
        _fail(
            "articulation_v2_aggregate_dependency_unavailable",
            "the installed world-understanding aggregate seam is not callable: "
            f"{module_name}.{symbol_name}",
        )
    return cast(Callable[..., Any], symbol)


def _aggregate_rigid_link_plans(
    articulation: ArticulationContractV2,
) -> tuple[RigidLinkPlanV1, ...]:
    """Build the shared 0.5 aggregate mapping for an aggregate v2 contract."""

    links = tuple(
        record for record in articulation.records if isinstance(record, LinkRecordV1)
    )
    if not any(link.body_authoring == "aggregate" for link in links):
        return ()
    members_by_link = _member_paths_by_link(articulation)
    plans: list[RigidLinkPlanV1] = []
    try:
        for link in links:
            plans.append(
                _rigid_link_plan_from_contract_link(
                    link,
                    members_by_link.get(link.link_id, ()),
                )
            )
        _shared_aggregate_core_callable(
            "world_understanding.functions.physics.joint_rigger.models",
            "_validate_rigid_link_cross_link_invariants",
        )(tuple(plans))
    except ArticulationV2FrameAuthoringError:
        raise
    except (TypeError, ValueError) as exc:
        _fail(
            "articulation_v2_aggregate_link_plan_invalid",
            "contract membership cannot produce the shared deterministic "
            f"aggregate plan: {type(exc).__name__}: {exc}",
        )
    return tuple(sorted(plans, key=lambda item: item.link_id))


def _rigid_link_plan_from_contract_link(
    link: LinkRecordV1,
    source_paths: tuple[str, ...],
) -> RigidLinkPlanV1:
    members = tuple(
        RigidLinkMemberPlanV1(
            source_prim_path=source_path,
            authored_prim_path=(
                source_path
                if link.body_authoring == "existing"
                else f"{link.body_prim_path}/{source_path.rsplit('/', 1)[-1]}"
            ),
        )
        for source_path in source_paths
    )
    return RigidLinkPlanV1(
        link_id=link.link_id,
        body_authoring=link.body_authoring,
        body_prim_path=link.body_prim_path,
        members=members,
    )


def _author_contract_aggregate_links(
    stage: Any,
    rigid_link_plans: tuple[RigidLinkPlanV1, ...],
) -> None:
    """Apply the exact shared aggregate policy with app-stable failures."""

    try:
        _shared_aggregate_core_callable(
            "world_understanding.functions.physics.joint_rigger.rigid_links",
            "_author_aggregate_rigid_link_plans",
        )(stage, rigid_link_plans)
    except TypeError as exc:
        _fail(
            "articulation_v2_aggregate_link_plan_invalid",
            "shared aggregate authoring rejected the exact plan type: "
            f"{type(exc).__name__}: {exc}",
        )
    except JointRiggerContractError as exc:
        mapped_code = _AGGREGATE_CORE_AUTHOR_ERROR_CODES.get(exc.code)
        if mapped_code is None:
            _fail(
                "articulation_v2_aggregate_authoring_failed",
                "shared aggregate authoring returned an unmapped failure: "
                f"{exc.code}: {exc.detail}",
            )
        _fail(
            mapped_code,
            f"shared aggregate authoring rejected the target: {exc.detail}",
        )


def _require_target_aggregate_members_not_rigid_bodies(
    stage: Any,
    rigid_link_plans: tuple[RigidLinkPlanV1, ...],
) -> None:
    """Reject a target that would retain nested independent rigid bodies."""

    from pxr import UsdPhysics

    member_paths = tuple(
        member.source_prim_path
        for link in rigid_link_plans
        if link.body_authoring == "aggregate"
        for member in link.members
    )
    for member_path in member_paths:
        prim = stage.GetPrimAtPath(member_path)
        if (
            not prim
            or not prim.IsValid()
            or not prim.IsActive()
            or not prim.IsDefined()
        ):
            _fail(
                "articulation_v2_aggregate_source_missing",
                f"aggregate target member is missing or inactive: {member_path}",
            )
    conflicts = _aggregate_rigid_body_conflicts(
        stage,
        rigid_link_plans,
        authored=False,
        UsdPhysics=UsdPhysics,
    )
    if conflicts:
        _fail(
            "articulation_v2_aggregate_member_rigid_body_conflict",
            "aggregate target members must be geometry/components rather than "
            f"pre-existing independent rigid bodies: {conflicts}",
        )
    _reject_aggregate_member_targeting_properties(stage, member_paths)


def _aggregate_rigid_body_conflicts(
    stage: Any,
    rigid_link_plans: tuple[RigidLinkPlanV1, ...],
    *,
    authored: bool,
    UsdPhysics: Any,
) -> list[str]:
    member_paths = tuple(
        (member.authored_prim_path if authored else member.source_prim_path)
        for link in rigid_link_plans
        if link.body_authoring == "aggregate"
        for member in link.members
    )
    conflicts: set[str] = set()
    visits = 0
    for prim in stage.TraverseAll():
        visits += 1
        if visits > _AGGREGATE_TARGET_SCAN_LIMIT:
            _fail(
                "articulation_v2_aggregate_target_scan_limit_exceeded",
                "aggregate target exceeded the fixed prim scan limit",
            )
        path = str(prim.GetPath())
        if any(
            path == member_path or path.startswith(f"{member_path}/")
            for member_path in member_paths
        ) and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            conflicts.add(path)
    for link in rigid_link_plans:
        if link.body_authoring != "aggregate":
            continue
        anchor_path = (
            link.body_prim_path if authored else link.members[0].source_prim_path
        )
        parent = stage.GetPrimAtPath(anchor_path).GetParent()
        while parent and parent.IsValid() and str(parent.GetPath()) != "/":
            if parent.HasAPI(UsdPhysics.RigidBodyAPI):
                conflicts.add(str(parent.GetPath()))
            parent = parent.GetParent()
    return sorted(conflicts)


def _reject_aggregate_member_targeting_properties(
    stage: Any,
    member_paths: tuple[str, ...],
) -> None:
    conflicts: list[str] = []
    visits = 0
    for prim in stage.TraverseAll():
        visits += 1
        if visits > _AGGREGATE_TARGET_SCAN_LIMIT:
            _fail(
                "articulation_v2_aggregate_target_scan_limit_exceeded",
                "aggregate target exceeded the fixed prim scan limit",
            )
        for prop in prim.GetAuthoredProperties():
            attribute = prim.GetAttribute(prop.GetName())
            targets = (
                tuple(str(path) for path in attribute.GetConnections())
                if attribute
                else tuple(
                    str(path)
                    for path in prim.GetRelationship(prop.GetName()).GetTargets()
                )
            )
            if any(
                target == member_path
                or target.startswith(f"{member_path}/")
                or target.startswith(f"{member_path}.")
                for target in targets
                for member_path in member_paths
            ):
                conflicts.append(f"{prim.GetPath()}.{prop.GetName()}")
    if conflicts:
        _fail(
            "articulation_v2_aggregate_member_relationship_unsupported",
            "aggregate namespace authoring does not rewrite relationship or "
            f"connection targets into member subtrees: {sorted(conflicts)}",
        )


def _author_aggregate_rigid_body_schemas(
    stage: Any,
    rigid_link_plans: tuple[RigidLinkPlanV1, ...],
) -> None:
    """Make each authored aggregate endpoint one explicit USD rigid body."""

    from pxr import UsdPhysics

    for link in rigid_link_plans:
        if link.body_authoring != "aggregate":
            continue
        prim = stage.GetPrimAtPath(link.body_prim_path)
        schema = UsdPhysics.RigidBodyAPI.Apply(prim)
        if not schema or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            _fail(
                "articulation_v2_aggregate_rigid_body_authoring_failed",
                f"could not apply PhysicsRigidBodyAPI to {link.body_prim_path!r}",
            )


def _validate_aggregate_rigid_body_schemas(
    stage: Any,
    rigid_link_plans: tuple[RigidLinkPlanV1, ...],
) -> None:
    """Require one saved rigid body at the aggregate and none on its members."""

    from pxr import UsdPhysics

    member_conflicts = _aggregate_rigid_body_conflicts(
        stage,
        rigid_link_plans,
        authored=True,
        UsdPhysics=UsdPhysics,
    )
    for link in rigid_link_plans:
        if link.body_authoring != "aggregate":
            continue
        body = stage.GetPrimAtPath(link.body_prim_path)
        if not body.HasAPI(UsdPhysics.RigidBodyAPI) or member_conflicts:
            _fail(
                "articulation_v2_saved_readback_mismatch",
                "saved aggregate rigid-body schemas differ from the contract: "
                f"body={link.body_prim_path!r}, "
                f"body_has_api={body.HasAPI(UsdPhysics.RigidBodyAPI)}, "
                f"member_conflicts={member_conflicts}",
            )


def _validate_contract_aggregate_links(
    stage: Any,
    rigid_link_plans: tuple[RigidLinkPlanV1, ...],
) -> None:
    """Read back exact saved membership through the shared validator."""

    try:
        _shared_aggregate_core_callable(
            "world_understanding.functions.physics.joint_rigger.rigid_links",
            "_validate_authored_rigid_link_plans",
        )(stage, rigid_link_plans)
    except JointRiggerContractError as exc:
        _fail(
            "articulation_v2_saved_readback_mismatch",
            "saved aggregate membership or identity differs from the contract: "
            f"{exc.code}: {exc.detail}",
        )


def _validate_authoring_contract(
    contract: ArticulationV2FrameAuthoringContractV1,
) -> None:
    articulation = contract.articulation
    if articulation.status != "ready_for_rigger_input":
        _fail(
            "articulation_v2_contract_not_ready",
            "only a ready_for_rigger_input v2 contract may be authored",
        )
    expected_identities = {
        contract.target.usd_identity,
        contract.source_reference.usd_identity,
    }
    if set(articulation.source_identities) != expected_identities:
        _fail(
            "articulation_v2_contract_artifact_partition_invalid",
            "articulation source identities must be exactly target and "
            "source_reference",
        )
    joints = {
        record.joint_id: record
        for record in articulation.records
        if isinstance(record, JointRecordV2)
    }
    bindings = {binding.joint_id: binding for binding in contract.source_joint_bindings}
    if not joints or set(bindings) != set(joints):
        _fail(
            "articulation_v2_source_mapping_incomplete",
            "source_joint_bindings must name every and only contract joint: "
            f"joints={sorted(joints)}, bindings={sorted(bindings)}",
        )
    member_paths_by_link = _member_paths_by_link(articulation)
    links = {
        record.link_id: record
        for record in articulation.records
        if isinstance(record, LinkRecordV1)
    }

    for record in articulation.records:
        if isinstance(record, PrimRecordV1):
            _require_provenance_role(
                record.membership_evidence,
                expected_artifact=contract.target.usd_identity,
                expected_prim_path=record.prim_path,
                expected_properties=None,
                field=f"prim {record.prim_path} membership",
                code="articulation_v2_target_evidence_invalid",
            )
        elif isinstance(record, LinkRecordV1):
            member_paths = member_paths_by_link.get(record.link_id, ())
            for field, provenance in record.field_evidence.items():
                expected_prim_path = record.body_prim_path
                if record.body_authoring == "aggregate":
                    if provenance.prim_path not in member_paths:
                        _fail(
                            "articulation_v2_target_evidence_invalid",
                            f"aggregate link {record.link_id!r} {field} evidence "
                            "must select one exact declared target member",
                        )
                    expected_prim_path = cast(str, provenance.prim_path)
                _require_provenance_role(
                    provenance,
                    expected_artifact=contract.target.usd_identity,
                    expected_prim_path=expected_prim_path,
                    expected_properties=None,
                    field=f"link {record.link_id} {field}",
                    code="articulation_v2_target_evidence_invalid",
                )
        elif isinstance(record, JointRecordV2):
            binding = bindings[record.joint_id]
            if isinstance(
                record.constraint,
                FixedConstraintV2 | DistanceConstraintV2,
            ):
                body0_link = links.get(record.body0_link)
                body1_link = links.get(record.body1_link)
                if (
                    body0_link is not None and body0_link.body_authoring == "aggregate"
                ) or (
                    body1_link is not None and body1_link.body_authoring == "aggregate"
                ):
                    _fail(
                        "articulation_v2_constraint_aggregate_link_unsupported",
                        f"joint {record.joint_id!r} requires two existing rigid-body "
                        "links; fixed/distance aggregate policy is not represented",
                    )
                if binding.topology_decision != "explicit_two_body_constraint":
                    _fail(
                        "articulation_v2_fixed_topology_decision_unresolved",
                        f"joint {record.joint_id!r} requires a captured "
                        "explicit_two_body_constraint topology decision",
                    )
            elif binding.topology_decision is not None:
                _fail(
                    "articulation_v2_source_binding_policy_invalid",
                    f"joint {record.joint_id!r} cannot carry a fixed/distance "
                    "topology decision",
                )
            if isinstance(record.constraint, DistanceConstraintV2) and (
                record.constraint.minimum_meters is None
                or record.constraint.maximum_meters is None
            ):
                _fail(
                    "articulation_v2_distance_bounds_incomplete",
                    f"joint {record.joint_id!r} requires explicit source-backed "
                    "minimum and maximum bounds",
                )
            source_path = binding.source_joint_path
            expected = _expected_source_properties(record)
            observed_fields = {item.field for item in record.field_evidence}
            if observed_fields != set(expected):
                _fail(
                    "articulation_v2_source_evidence_incomplete",
                    f"joint {record.joint_id!r} evidence fields differ from the "
                    f"source-backed authoring profile: observed="
                    f"{sorted(observed_fields)}, expected={sorted(expected)}",
                )
            for item in record.field_evidence:
                _require_provenance_role(
                    item.provenance,
                    expected_artifact=contract.source_reference.usd_identity,
                    expected_prim_path=source_path,
                    expected_properties=expected[item.field],
                    field=f"joint {record.joint_id} {item.field}",
                    code="articulation_v2_source_evidence_invalid",
                )


def _require_provenance_role(
    provenance: FieldProvenanceV1,
    *,
    expected_artifact: ArtifactIdentityV1,
    expected_prim_path: str,
    expected_properties: tuple[str, ...] | None,
    field: str,
    code: str,
) -> None:
    if (
        provenance.source not in _SOURCE_BACKED_PROVENANCE_SOURCES
        or provenance.artifact != expected_artifact
        or provenance.prim_path != expected_prim_path
        or not provenance.properties
        or (
            expected_properties is not None
            and provenance.properties != tuple(sorted(expected_properties))
        )
    ):
        _fail(
            code,
            f"{field} does not bind the exact artifact, selector, and property "
            "set required by its role",
        )


def _source_endpoint_for_link(
    relationship: Any,
    *,
    field: str,
    link: LinkRecordV1,
    member_paths: tuple[str, ...],
    joint_id: str,
) -> str:
    """Bind a source endpoint to the exact declared member for one link."""

    source_path = _single_relationship_target(relationship, field=field)
    expected_paths = (
        (link.body_prim_path,) if link.body_authoring == "existing" else member_paths
    )
    if source_path not in expected_paths:
        _fail(
            "articulation_v2_source_endpoint_membership_conflict",
            f"joint {joint_id!r} {field} endpoint {source_path!r} is not an "
            f"exact member of link {link.link_id!r}: {sorted(expected_paths)}",
        )
    return source_path


def _attachment_for_authored_link(
    stage: Any,
    *,
    source_endpoint_path: str,
    source_frame: AttachmentFrameV2,
    link: LinkRecordV1,
    joint_id: str,
    endpoint_label: str,
) -> AttachmentFrameV2:
    """Express a source-member frame in its exact authored body space."""

    if link.body_authoring == "existing":
        if source_endpoint_path != link.body_prim_path:
            _fail(
                "articulation_v2_source_endpoint_membership_conflict",
                f"joint {joint_id!r} {endpoint_label} does not target existing "
                f"body {link.body_prim_path!r}",
            )
        return source_frame

    from pxr import Gf, Sdf, UsdGeom

    source_endpoint = _require_endpoint(
        stage,
        source_endpoint_path,
        label=f"source joint {joint_id!r} {endpoint_label}",
        Sdf=Sdf,
        UsdGeom=UsdGeom,
    )
    parent_path = str(Sdf.Path(link.body_prim_path).GetParentPath())
    aggregate_parent = _require_endpoint(
        stage,
        parent_path,
        label=f"aggregate link {link.link_id!r} parent",
        Sdf=Sdf,
        UsdGeom=UsdGeom,
        require_xformable=False,
    )
    _require_static_endpoint_transform(
        source_endpoint,
        label=f"source joint {joint_id!r} {endpoint_label}",
    )
    _require_static_endpoint_transform(
        aggregate_parent,
        label=f"aggregate link {link.link_id!r} parent",
    )
    meters_per_unit = _meters_per_unit(stage)
    source_world_frame = _validate_endpoint_frame(
        stage,
        source_endpoint,
        source_frame,
        meters_per_unit=meters_per_unit,
        label=f"source joint {joint_id!r} {endpoint_label}",
    )
    parent_world = _require_invertible_world_transform(
        stage,
        aggregate_parent,
        label=f"aggregate link {link.link_id!r} parent",
    )
    parent_inverse = parent_world.GetInverse()
    world_position, world_basis = source_world_frame
    local_position = parent_inverse.Transform(
        Gf.Vec3d(*(value / meters_per_unit for value in world_position))
    )
    local_basis = tuple(
        _normalized_direction(
            parent_inverse.TransformDir(Gf.Vec3d(*direction)),
            label=(
                f"joint {joint_id!r} {endpoint_label} aggregate-local frame direction"
            ),
        )
        for direction in world_basis
    )
    if any(
        abs(_dot(local_basis[left], local_basis[right])) > _FRAME_TOLERANCE
        for left, right in ((0, 1), (0, 2), (1, 2))
    ) or _dot(_cross(local_basis[0], local_basis[1]), local_basis[2]) < (
        1.0 - _FRAME_TOLERANCE
    ):
        _fail(
            "articulation_v2_aggregate_frame_reprojection_failed",
            f"joint {joint_id!r} {endpoint_label} frame cannot be represented "
            f"under aggregate body {link.body_prim_path!r}",
        )
    matrix = Gf.Matrix3d(1.0)
    for row, direction in enumerate(local_basis):
        matrix.SetRow(row, Gf.Vec3d(*direction))
    quaternion = matrix.ExtractRotation().GetQuat()
    imaginary = quaternion.GetImaginary()
    authored_frame = AttachmentFrameV2(
        position_meters=tuple(
            float(value) * meters_per_unit for value in local_position
        ),
        orientation_wxyz=_validated_orientation(
            (
                float(quaternion.GetReal()),
                float(imaginary[0]),
                float(imaginary[1]),
                float(imaginary[2]),
            )
        ),
    )
    authored_world_frame = _validate_endpoint_frame(
        stage,
        aggregate_parent,
        authored_frame,
        meters_per_unit=meters_per_unit,
        label=f"aggregate link {link.link_id!r} authored body",
    )
    _require_world_frame_close(
        authored_world_frame,
        source_world_frame,
        joint_id=joint_id,
        endpoint_label=endpoint_label,
    )
    return authored_frame


def _require_world_frame_close(
    observed: tuple[Vector3, tuple[Vector3, Vector3, Vector3]],
    expected: tuple[Vector3, tuple[Vector3, Vector3, Vector3]],
    *,
    joint_id: str,
    endpoint_label: str,
) -> None:
    observed_position, observed_basis = observed
    expected_position, expected_basis = expected
    if not _vectors_close(
        observed_position,
        expected_position,
        tolerance=_FRAME_COHERENCE_TOLERANCE,
    ) or any(
        not _vectors_close(
            observed_direction,
            expected_direction,
            tolerance=_FRAME_COHERENCE_TOLERANCE,
        )
        for observed_direction, expected_direction in zip(
            observed_basis,
            expected_basis,
            strict=True,
        )
    ):
        _fail(
            "articulation_v2_aggregate_frame_reprojection_failed",
            f"joint {joint_id!r} {endpoint_label} frame changes when projected "
            "into the authored aggregate body",
        )


def _expected_source_properties(
    record: JointRecordV2,
) -> dict[str, tuple[str, ...]]:
    result = {
        **_SOURCE_FRAME_PROPERTY_BY_FIELD,
        "constraint.kind": ("physics:body0", "physics:body1"),
    }
    constraint = record.constraint
    if isinstance(constraint, RevoluteConstraintV2 | PrismaticConstraintV2):
        result["constraint.axis_stage"] = ("physics:axis",)
        result["constraint.limit_mode"] = (
            ("physics:lowerLimit", "physics:upperLimit")
            if constraint.limit_mode == "bounded"
            else ("physics:axis",)
        )
        if isinstance(constraint, RevoluteConstraintV2) and (
            constraint.limit_mode == "bounded"
        ):
            result["constraint.lower_degrees"] = ("physics:lowerLimit",)
            result["constraint.upper_degrees"] = ("physics:upperLimit",)
        if isinstance(constraint, PrismaticConstraintV2) and (
            constraint.limit_mode == "bounded"
        ):
            result["constraint.lower_meters"] = ("physics:lowerLimit",)
            result["constraint.upper_meters"] = ("physics:upperLimit",)
    elif isinstance(constraint, SphericalConstraintV2):
        result["constraint.angular_limit_mode"] = (
            "physics:body0",
            "physics:body1",
        )
    elif isinstance(constraint, FixedConstraintV2):
        result["constraint.kind"] = ("usd:schema:FixedJoint",)
    elif isinstance(constraint, DistanceConstraintV2):
        result["constraint.kind"] = ("usd:schema:DistanceJoint",)
        result["constraint.minimum_meters"] = ("physics:minDistance",)
        result["constraint.maximum_meters"] = ("physics:maxDistance",)
    else:
        _fail(
            "articulation_v2_source_constraint_kind_invalid",
            "source evidence mapping received an unsupported constraint object",
        )
    return result


def _revalidate_source_reference(
    contract: ArticulationV2FrameAuthoringContractV1,
    reference_path: Path,
) -> None:
    from pxr import Usd, UsdPhysics

    stage = _open_stage(reference_path, label="captured source reference")
    try:
        all_joint_paths = {
            str(prim.GetPath())
            for prim in Usd.PrimRange.Stage(
                stage,
                Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate),
            )
            if prim.IsA(UsdPhysics.Joint)
        }
        bindings = {
            item.joint_id: item.source_joint_path
            for item in contract.source_joint_bindings
        }
        if all_joint_paths != set(bindings.values()):
            _fail(
                "articulation_v2_source_mapping_extra_or_missing",
                "reviewed source joints must be completely and uniquely bound: "
                f"source={sorted(all_joint_paths)}, "
                f"bindings={sorted(bindings.values())}",
            )
        links = {
            record.link_id: record
            for record in contract.articulation.records
            if isinstance(record, LinkRecordV1)
        }
        member_paths_by_link = _member_paths_by_link(contract.articulation)
        for record in contract.articulation.records:
            if not isinstance(record, JointRecordV2):
                continue
            body0 = links.get(record.body0_link)
            body1 = links.get(record.body1_link)
            if body0 is None or body1 is None:
                _fail(
                    "articulation_v2_link_missing",
                    f"joint {record.joint_id!r} references an undeclared link",
                )
            source_path = bindings[record.joint_id]
            source_prim = _require_source_joint_prim(
                stage,
                source_path,
                label=f"source joint for {record.joint_id!r}",
            )
            _require_source_schema_matches(source_prim, record)
            source_joint = UsdPhysics.Joint(source_prim)
            source_body0_path = _source_endpoint_for_link(
                source_joint.GetBody0Rel(),
                field="physics:body0",
                link=body0,
                member_paths=member_paths_by_link.get(body0.link_id, ()),
                joint_id=record.joint_id,
            )
            source_body1_path = _source_endpoint_for_link(
                source_joint.GetBody1Rel(),
                field="physics:body1",
                link=body1,
                member_paths=member_paths_by_link.get(body1.link_id, ()),
                joint_id=record.joint_id,
            )
            if isinstance(record.constraint, FixedConstraintV2 | DistanceConstraintV2):
                extracted_fixed_distance = _extract_source_fixed_distance_from_stage(
                    stage,
                    source_artifact=contract.source_reference.usd_identity,
                    source_joint_path=source_path,
                    expected_body0_prim_path=source_body0_path,
                    expected_body1_prim_path=source_body1_path,
                    expected_constraint_kind=record.constraint.kind,
                    provenance_source="authored_reference",
                )
                extracted_attachments = extracted_fixed_distance.attachments
                if extracted_fixed_distance.constraint != record.constraint:
                    _fail(
                        "articulation_v2_source_constraint_conflict",
                        f"joint {record.joint_id!r} constraint differs from "
                        "the captured source",
                    )
            else:
                extracted_frames = _extract_source_frames_from_stage(
                    stage,
                    source_artifact=contract.source_reference.usd_identity,
                    source_joint_path=source_path,
                    expected_body0_prim_path=source_body0_path,
                    expected_body1_prim_path=source_body1_path,
                    provenance_source="authored_reference",
                )
                _require_source_constraint_matches(
                    stage,
                    source_prim,
                    record,
                    extracted_frames.attachments,
                )
                extracted_attachments = ExplicitAttachmentFramesV2(
                    kind="explicit",
                    body0=_attachment_for_authored_link(
                        stage,
                        source_endpoint_path=source_body0_path,
                        source_frame=extracted_frames.attachments.body0,
                        link=body0,
                        joint_id=record.joint_id,
                        endpoint_label="body0",
                    ),
                    body1=_attachment_for_authored_link(
                        stage,
                        source_endpoint_path=source_body1_path,
                        source_frame=extracted_frames.attachments.body1,
                        link=body1,
                        joint_id=record.joint_id,
                        endpoint_label="body1",
                    ),
                )
            expected_attachments = record.attachments
            if not isinstance(expected_attachments, ExplicitAttachmentFramesV2):
                _fail(
                    "articulation_v2_explicit_frames_required",
                    f"joint {record.joint_id!r} lacks explicit frames",
                )
            _require_attachment_pair_close(
                extracted_attachments,
                expected_attachments,
                joint_id=record.joint_id,
            )
    finally:
        del stage


def _require_aggregate_source_target_member_alignment(
    contract: ArticulationV2FrameAuthoringContractV1,
    *,
    target_stage: Any,
    reference_path: Path,
) -> None:
    """Bind every aggregate source endpoint to the captured target geometry."""

    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    source_stage = _open_stage(
        reference_path,
        label="captured source reference for aggregate alignment",
    )
    try:
        links = {
            record.link_id: record
            for record in contract.articulation.records
            if isinstance(record, LinkRecordV1)
        }
        member_paths_by_link = _member_paths_by_link(contract.articulation)
        bindings = {
            item.joint_id: item.source_joint_path
            for item in contract.source_joint_bindings
        }
        compared: set[tuple[str, str]] = set()
        compared_parents: set[str] = set()
        for record in contract.articulation.records:
            if not isinstance(record, JointRecordV2) or isinstance(
                record.constraint,
                FixedConstraintV2 | DistanceConstraintV2,
            ):
                continue
            source_prim = _require_source_joint_prim(
                source_stage,
                bindings[record.joint_id],
                label=f"source joint for {record.joint_id!r}",
            )
            source_joint = UsdPhysics.Joint(source_prim)
            for endpoint_label, relationship, link_id in (
                ("body0", source_joint.GetBody0Rel(), record.body0_link),
                ("body1", source_joint.GetBody1Rel(), record.body1_link),
            ):
                link = links[link_id]
                if link.body_authoring != "aggregate":
                    continue
                source_path = _source_endpoint_for_link(
                    relationship,
                    field=f"physics:{endpoint_label}",
                    link=link,
                    member_paths=member_paths_by_link.get(link.link_id, ()),
                    joint_id=record.joint_id,
                )
                comparison_key = (link.link_id, source_path)
                if comparison_key in compared:
                    continue
                compared.add(comparison_key)
                parent_path = str(Sdf.Path(link.body_prim_path).GetParentPath())
                if parent_path not in compared_parents:
                    compared_parents.add(parent_path)
                    source_parent = _require_endpoint(
                        source_stage,
                        parent_path,
                        label=f"source aggregate parent {parent_path!r}",
                        Sdf=Sdf,
                        UsdGeom=UsdGeom,
                        require_xformable=False,
                    )
                    target_parent = _require_endpoint(
                        target_stage,
                        parent_path,
                        label=f"target aggregate parent {parent_path!r}",
                        Sdf=Sdf,
                        UsdGeom=UsdGeom,
                        require_xformable=False,
                    )
                    _require_static_endpoint_transform(
                        source_parent,
                        label=f"source aggregate parent {parent_path!r}",
                    )
                    _require_static_endpoint_transform(
                        target_parent,
                        label=f"target aggregate parent {parent_path!r}",
                    )
                    source_parent_signature = _physical_world_transform_signature(
                        _require_invertible_world_transform(
                            source_stage,
                            source_parent,
                            label=f"source aggregate parent {parent_path!r}",
                        ),
                        meters_per_unit=_meters_per_unit(source_stage),
                        Gf=Gf,
                    )
                    target_parent_signature = _physical_world_transform_signature(
                        _require_invertible_world_transform(
                            target_stage,
                            target_parent,
                            label=f"target aggregate parent {parent_path!r}",
                        ),
                        meters_per_unit=_meters_per_unit(target_stage),
                        Gf=Gf,
                    )
                    if any(
                        not _vectors_close(
                            source_value,
                            target_value,
                            tolerance=_FRAME_COHERENCE_TOLERANCE,
                        )
                        for source_value, target_value in zip(
                            source_parent_signature,
                            target_parent_signature,
                            strict=True,
                        )
                    ):
                        _fail(
                            "articulation_v2_target_source_member_transform_conflict",
                            f"aggregate parent {parent_path!r} for link "
                            f"{link.link_id!r} does not preserve the captured "
                            "target parent physical world transform",
                        )
                source_member = _require_endpoint(
                    source_stage,
                    source_path,
                    label=f"source aggregate member {source_path!r}",
                    Sdf=Sdf,
                    UsdGeom=UsdGeom,
                )
                target_member = _require_endpoint(
                    target_stage,
                    source_path,
                    label=f"target aggregate member {source_path!r}",
                    Sdf=Sdf,
                    UsdGeom=UsdGeom,
                )
                _require_static_endpoint_transform(
                    source_member,
                    label=f"source aggregate member {source_path!r}",
                )
                _require_static_endpoint_transform(
                    target_member,
                    label=f"target aggregate member {source_path!r}",
                )
                source_world = _require_invertible_world_transform(
                    source_stage,
                    source_member,
                    label=f"source aggregate member {source_path!r}",
                )
                target_world = _require_invertible_world_transform(
                    target_stage,
                    target_member,
                    label=f"target aggregate member {source_path!r}",
                )
                source_signature = _physical_world_transform_signature(
                    source_world,
                    meters_per_unit=_meters_per_unit(source_stage),
                    Gf=Gf,
                )
                target_signature = _physical_world_transform_signature(
                    target_world,
                    meters_per_unit=_meters_per_unit(target_stage),
                    Gf=Gf,
                )
                if any(
                    not _vectors_close(
                        source_value,
                        target_value,
                        tolerance=_FRAME_COHERENCE_TOLERANCE,
                    )
                    for source_value, target_value in zip(
                        source_signature,
                        target_signature,
                        strict=True,
                    )
                ):
                    _fail(
                        "articulation_v2_target_source_member_transform_conflict",
                        f"aggregate source endpoint {source_path!r} for joint "
                        f"{record.joint_id!r} {endpoint_label} does not preserve "
                        "the captured target member's physical world transform",
                    )
    finally:
        del source_stage


def _physical_world_transform_signature(
    world_transform: Any,
    *,
    meters_per_unit: float,
    Gf: Any,
) -> tuple[Vector3, Vector3, Vector3, Vector3]:
    origin = world_transform.Transform(Gf.Vec3d(0.0, 0.0, 0.0))
    position_meters = tuple(float(value) * meters_per_unit for value in origin)
    directions = tuple(
        tuple(float(value) for value in world_transform.TransformDir(axis))
        for axis in (
            Gf.Vec3d(1.0, 0.0, 0.0),
            Gf.Vec3d(0.0, 1.0, 0.0),
            Gf.Vec3d(0.0, 0.0, 1.0),
        )
    )
    return cast(
        tuple[Vector3, Vector3, Vector3, Vector3], (position_meters, *directions)
    )


def _require_source_schema_matches(source_prim: Any, record: JointRecordV2) -> None:
    from pxr import UsdPhysics

    schemas: Mapping[str, Any] = {
        "revolute": UsdPhysics.RevoluteJoint,
        "prismatic": UsdPhysics.PrismaticJoint,
        "spherical": UsdPhysics.SphericalJoint,
        "fixed": UsdPhysics.FixedJoint,
        "distance": UsdPhysics.DistanceJoint,
    }
    expected = schemas.get(record.constraint.kind)
    if expected is None or not source_prim.IsA(expected):
        _fail(
            "articulation_v2_source_schema_conflict",
            f"joint {record.joint_id!r} source schema does not match "
            f"{record.constraint.kind!r}",
        )


def _require_attachment_pair_close(
    observed: ExplicitAttachmentFramesV2,
    expected: ExplicitAttachmentFramesV2,
    *,
    joint_id: str,
) -> None:
    for field, left, right in (
        (
            "body0.position_meters",
            observed.body0.position_meters,
            expected.body0.position_meters,
        ),
        (
            "body0.orientation_wxyz",
            observed.body0.orientation_wxyz,
            expected.body0.orientation_wxyz,
        ),
        (
            "body1.position_meters",
            observed.body1.position_meters,
            expected.body1.position_meters,
        ),
        (
            "body1.orientation_wxyz",
            observed.body1.orientation_wxyz,
            expected.body1.orientation_wxyz,
        ),
    ):
        if not all(
            math.isclose(
                first,
                second,
                rel_tol=0.0,
                abs_tol=_READBACK_TOLERANCE,
            )
            for first, second in zip(left, right, strict=True)
        ):
            _fail(
                "articulation_v2_source_contract_mismatch",
                f"joint {joint_id!r} {field} differs from reviewed source",
            )


def _require_source_constraint_matches(
    stage: Any,
    source_prim: Any,
    record: JointRecordV2,
    attachments: ExplicitAttachmentFramesV2,
) -> None:
    from pxr import Sdf, UsdGeom, UsdPhysics

    constraint = record.constraint
    if not isinstance(
        constraint,
        RevoluteConstraintV2 | PrismaticConstraintV2 | SphericalConstraintV2,
    ):
        _fail(
            "articulation_v2_source_schema_conflict",
            f"joint {record.joint_id!r} is outside the frame authoring profile",
        )
    joint = UsdPhysics.Joint(source_prim)
    meters_per_unit = _meters_per_unit(stage)
    body0_path = _single_relationship_target(
        joint.GetBody0Rel(),
        field="physics:body0",
    )
    body1_path = _single_relationship_target(
        joint.GetBody1Rel(),
        field="physics:body1",
    )
    body0 = _require_endpoint(
        stage,
        body0_path,
        label=f"source joint {record.joint_id!r} body0",
        Sdf=Sdf,
        UsdGeom=UsdGeom,
    )
    body1 = _require_endpoint(
        stage,
        body1_path,
        label=f"source joint {record.joint_id!r} body1",
        Sdf=Sdf,
        UsdGeom=UsdGeom,
    )
    world_frame0 = _validate_endpoint_frame(
        stage,
        body0,
        attachments.body0,
        meters_per_unit=meters_per_unit,
        label=f"source joint {record.joint_id!r} body0",
    )
    world_frame1 = _validate_endpoint_frame(
        stage,
        body1,
        attachments.body1,
        meters_per_unit=meters_per_unit,
        label=f"source joint {record.joint_id!r} body1",
    )
    expected_axis = (
        constraint.axis_stage
        if isinstance(constraint, RevoluteConstraintV2 | PrismaticConstraintV2)
        else None
    )
    if (
        isinstance(
            constraint,
            RevoluteConstraintV2 | PrismaticConstraintV2,
        )
        and expected_axis is None
    ):
        _fail(
            "articulation_v2_source_axis_conflict",
            f"joint {record.joint_id!r} has no reviewed source axis",
        )
    _require_full_frame_coherence(
        constraint_kind=constraint.kind,
        world_frame0=world_frame0,
        world_frame1=world_frame1,
        expected_axis=expected_axis,
        label=f"source joint {record.joint_id!r}",
        axis_error_code="articulation_v2_source_axis_conflict",
        anchor_error_code="articulation_v2_source_endpoint_anchor_conflict",
        transverse_anchor_error_code=(
            "articulation_v2_transverse_anchor_frame_conflict"
        ),
    )
    if isinstance(constraint, RevoluteConstraintV2 | PrismaticConstraintV2):
        typed = (
            UsdPhysics.RevoluteJoint(source_prim)
            if isinstance(constraint, RevoluteConstraintV2)
            else UsdPhysics.PrismaticJoint(source_prim)
        )
        lower = typed.GetLowerLimitAttr()
        upper = typed.GetUpperLimitAttr()
        expected_bounded = constraint.limit_mode == "bounded"
        if expected_bounded:
            observed_lower = _require_source_static_scalar_limit(
                lower,
                field="physics:lowerLimit",
                joint_id=record.joint_id,
            )
            observed_upper = _require_source_static_scalar_limit(
                upper,
                field="physics:upperLimit",
                joint_id=record.joint_id,
            )
            if isinstance(constraint, PrismaticConstraintV2):
                observed_lower *= meters_per_unit
                observed_upper *= meters_per_unit
                expected_lower = constraint.lower_meters
                expected_upper = constraint.upper_meters
            else:
                expected_lower = constraint.lower_degrees
                expected_upper = constraint.upper_degrees
            if (
                expected_lower is None
                or expected_upper is None
                or not math.isclose(
                    observed_lower,
                    expected_lower,
                    rel_tol=0.0,
                    abs_tol=_READBACK_TOLERANCE,
                )
                or not math.isclose(
                    observed_upper,
                    expected_upper,
                    rel_tol=0.0,
                    abs_tol=_READBACK_TOLERANCE,
                )
            ):
                _fail(
                    "articulation_v2_source_limit_conflict",
                    f"joint {record.joint_id!r} limits differ from source",
                )
        else:
            _require_source_limit_absent(
                lower,
                field="physics:lowerLimit",
                joint_id=record.joint_id,
            )
            _require_source_limit_absent(
                upper,
                field="physics:upperLimit",
                joint_id=record.joint_id,
            )
    elif isinstance(constraint, SphericalConstraintV2):
        if constraint.angular_limit_mode != "free":
            _fail(
                "articulation_v2_source_limit_mode_conflict",
                f"joint {record.joint_id!r} spherical source is not free",
            )
        _require_free_spherical_source_controls_absent(source_prim)


def _prepare_target_workspace(
    target_path: Path,
    target: CapturedUsdArtifactBindingV1,
    *,
    workspace: Path,
) -> _PreparedTargetWorkspace:
    staged_output = workspace / f"validated-output.{target.format}"
    if target.format != "usdz":
        shutil.copyfile(target_path, staged_output)
        if _file_sha256(staged_output) != target.capture.sha256:
            _fail(
                "articulation_v2_target_copy_mismatch",
                "private authoring copy differs from captured target",
            )
        return _PreparedTargetWorkspace(
            editable_root=staged_output,
            staged_output=staged_output,
        )

    package_members, member_hashes = _package_member_hashes(target_path)
    try:
        root_member = find_usdz_root_layer(target_path)
        editable_root = extract_usdz_package_for_edit(
            target_path,
            workspace / "editable-package",
        )
    except (OSError, UsdzPackageError) as exc:
        _fail(
            "articulation_v2_usdz_extraction_failed",
            f"could not safely extract captured target: {type(exc).__name__}: {exc}",
        )
    if editable_root.relative_to(workspace / "editable-package") != root_member:
        _fail(
            "articulation_v2_usdz_root_mismatch",
            "extracted root differs from captured package root",
        )
    non_root = {
        member: digest
        for member, digest in member_hashes.items()
        if member != root_member.as_posix()
    }
    return _PreparedTargetWorkspace(
        editable_root=editable_root,
        staged_output=staged_output,
        package_member_order=(
            root_member.as_posix(),
            *(member for member in package_members if member != root_member.as_posix()),
        ),
        package_root_member=root_member.as_posix(),
        non_root_member_hashes=non_root,
    )


def _package_member_hashes(path: Path) -> tuple[tuple[str, ...], dict[str, str]]:
    order: list[str] = []
    hashes: dict[str, str] = {}
    try:
        with zipfile.ZipFile(path) as package:
            for info in package.infolist():
                if info.is_dir():
                    continue
                member = info.filename
                if member in hashes:
                    _fail(
                        "articulation_v2_usdz_member_invalid",
                        f"duplicate package member: {member}",
                    )
                digest = hashlib.sha256()
                with package.open(info) as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                order.append(member)
                hashes[member] = digest.hexdigest()
    except zipfile.BadZipFile as exc:
        _fail(
            "articulation_v2_usdz_member_invalid",
            f"captured target is not a valid package: {exc}",
        )
    return tuple(order), hashes


def _require_preserved_package_members(
    path: Path,
    prepared: _PreparedTargetWorkspace,
    *,
    package_manifest: UsdzPackageTreeManifest | None = None,
) -> None:
    expected = prepared.non_root_member_hashes
    expected_root = prepared.package_root_member
    if expected is None or expected_root is None:
        _fail(
            "articulation_v2_usdz_workspace_invalid",
            "package preservation requires prepared USDZ metadata",
        )
    if package_manifest is not None:
        if package_manifest.root_member_order != prepared.package_member_order:
            _fail(
                "articulation_v2_usdz_layout_invalid",
                "retained output package order differs from the prepared target: "
                f"observed={package_manifest.root_member_order}, "
                f"expected={prepared.package_member_order}",
            )
    else:
        try:
            validate_usdz_package_layout(
                path,
                expected_root_member=expected_root,
                expected_member_order=prepared.package_member_order,
                max_members=DEFAULT_MAX_USDZ_PACKAGE_TREE_MEMBERS,
                max_total_bytes=DEFAULT_MAX_USDZ_EXTRACTED_BYTES,
                max_member_bytes=DEFAULT_MAX_USDZ_MEMBER_BYTES,
            )
        except UsdzPackageError as exc:
            _fail(
                "articulation_v2_usdz_layout_invalid",
                f"private output package is not canonical and complete: {exc}",
            )
    _, observed = _package_member_hashes(path)
    changed = sorted(
        member for member, digest in expected.items() if observed.get(member) != digest
    )
    missing_or_extra = sorted(set(observed) ^ ({*expected} | {expected_root}))
    if changed or missing_or_extra:
        _fail(
            "articulation_v2_usdz_member_changed",
            "non-root package members were not preserved byte-for-byte: "
            f"changed={changed}, set_difference={missing_or_extra}",
        )


def _require_output_path(
    value: str | Path,
    *,
    target_format: str,
    parent_descriptor: int,
) -> Path:
    raw_path = os.fspath(value)
    if "\x00" in raw_path:
        _fail(
            "articulation_v2_output_path_invalid",
            "output path must not contain a null byte",
        )
    path = Path(raw_path)
    if path.name in {"", ".", ".."}:
        _fail(
            "articulation_v2_output_path_invalid",
            f"output path must name one file: {path}",
        )
    if target_format not in _USD_FORMATS or path.suffix.lower() != (
        f".{target_format}"
    ):
        _fail(
            "articulation_v2_output_format_mismatch",
            f"output must preserve captured target format .{target_format}",
        )
    try:
        os.stat(
            path.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        pass
    except OSError as exc:
        _fail(
            "articulation_v2_output_commit_failed",
            f"could not inspect held output destination: {type(exc).__name__}: {exc}",
        )
    else:
        _fail(
            "articulation_v2_output_exists",
            f"refusing to replace an existing output: {path}",
        )
    return path


type _StageRetainedPublication = Callable[[Path, str], None]


@contextmanager
def _publish_without_overwrite(
    destination: Path,
    *,
    parent_descriptor: int,
) -> Iterator[_StageRetainedPublication]:
    """Stage verified retained bytes and commit only on a clean context exit."""

    state: ConfinedAtomicWrite | None = None
    staged = False
    try:
        with ExitStack() as publication_lifetime:

            def stage_retained_snapshot(source: Path, expected_sha256: str) -> None:
                nonlocal staged, state
                if staged:
                    raise RuntimeError("retained output publication already staged")
                staged = True
                try:
                    state = publication_lifetime.enter_context(
                        confined_atomic_writer(
                            parent_descriptor,
                            destination.name,
                            overwrite=False,
                            file_mode=0o600,
                        )
                    )
                except (OSError, RuntimeError, ValueError) as exc:
                    _fail(
                        "articulation_v2_output_commit_failed",
                        "could not prepare atomic output publication: "
                        f"{type(exc).__name__}: {exc}",
                    )
                if state.stream is None:
                    _fail(
                        "articulation_v2_output_exists",
                        f"refusing to replace an existing output: {destination}",
                    )
                _copy_verified_retained_snapshot(
                    source,
                    state.stream,
                    expected_sha256=expected_sha256,
                )

            yield stage_retained_snapshot
    except ArticulationV2FrameAuthoringError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        _fail(
            "articulation_v2_output_commit_failed",
            f"could not atomically publish output: {type(exc).__name__}: {exc}",
        )

    if not staged or state is None:
        _fail(
            "articulation_v2_output_commit_failed",
            "retained output publication was not staged",
        )
    if not state.published:
        _fail(
            "articulation_v2_output_exists",
            f"refusing to replace an existing output: {destination}",
        )


def _copy_verified_retained_snapshot(
    source: Path,
    destination: Any,
    *,
    expected_sha256: str,
) -> None:
    """Copy one retained regular file through a stable descriptor and digest."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    descriptor = -1
    try:
        descriptor = os.open(source, flags)
        initial = os.fstat(descriptor)
        path_initial = os.stat(source, follow_symlinks=False)
        initial_state = _retained_raw_file_state(initial)
        if (
            not stat.S_ISREG(initial.st_mode)
            or _retained_raw_file_state(path_initial) != initial_state
        ):
            _fail(
                "articulation_v2_output_mutated",
                f"retained output snapshot is not a stable regular file: {source}",
            )

        digest = hashlib.sha256()
        offset = 0
        while offset < initial.st_size:
            chunk = os.pread(
                descriptor,
                min(1024 * 1024, initial.st_size - offset),
                offset,
            )
            if not chunk:
                _fail(
                    "articulation_v2_output_mutated",
                    f"retained output snapshot changed while copied: {source}",
                )
            digest.update(chunk)
            remaining = memoryview(chunk)
            while remaining:
                written = destination.write(remaining)
                if written is None or written <= 0:
                    raise OSError("retained output copy made no progress")
                remaining = remaining[written:]
            offset += len(chunk)

        final = os.fstat(descriptor)
        path_final = os.stat(source, follow_symlinks=False)
        if (
            os.pread(descriptor, 1, offset)
            or _retained_raw_file_state(final) != initial_state
            or _retained_raw_file_state(path_final) != initial_state
            or digest.hexdigest() != expected_sha256
        ):
            _fail(
                "articulation_v2_output_mutated",
                f"retained output snapshot changed while copied: {source}",
            )
    except ArticulationV2FrameAuthoringError:
        raise
    except OSError as exc:
        _fail(
            "articulation_v2_output_mutated",
            f"could not copy retained output snapshot: {type(exc).__name__}: {exc}",
        )
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _prepare_contract(
    stage: Any,
    contract: ArticulationContractV2,
    *,
    contract_sha256: str,
) -> tuple[tuple[_PreparedJoint, ...], str, bool]:
    from pxr import Sdf, Tf, Usd, UsdGeom, UsdPhysics

    default_prim = stage.GetDefaultPrim()
    if (
        not default_prim
        or not default_prim.IsValid()
        or not default_prim.IsActive()
        or not default_prim.IsDefined()
        or default_prim.IsInstance()
        or default_prim.IsInstanceProxy()
    ):
        _fail(
            "articulation_v2_invalid_default_prim",
            "input stage must have an active, defined, non-instance defaultPrim",
        )
    existing_joints = {
        str(prim.GetPath())
        for prim in Usd.PrimRange.Stage(
            stage,
            Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate),
        )
        if prim.IsA(UsdPhysics.Joint)
    }
    for prototype in stage.GetPrototypes():
        existing_joints.update(
            str(prim.GetPath())
            for prim in Usd.PrimRange.AllPrims(prototype)
            if prim.IsA(UsdPhysics.Joint)
        )
    if existing_joints:
        _fail(
            "articulation_v2_source_already_rigged",
            f"source must not contain USD physics joints: {sorted(existing_joints)}",
        )

    default_path = default_prim.GetPath()
    scope_path = str(default_path.AppendChild(_JOINT_SCOPE_NAME))
    existing_scope = stage.GetPrimAtPath(scope_path)
    create_scope = not existing_scope or not existing_scope.IsValid()
    if not create_scope and (
        not existing_scope.IsActive()
        or not existing_scope.IsDefined()
        or existing_scope.IsInstance()
        or existing_scope.IsInstanceProxy()
        or not existing_scope.IsA(UsdGeom.Scope)
    ):
        _fail(
            "articulation_v2_joint_scope_conflict",
            f"{scope_path} must be an active, defined, non-instance UsdGeom.Scope",
        )

    meters_per_unit = _meters_per_unit(stage)
    links = {
        record.link_id: record
        for record in contract.records
        if isinstance(record, LinkRecordV1)
    }
    joints = tuple(
        record for record in contract.records if isinstance(record, JointRecordV2)
    )
    if not joints:
        _fail(
            "articulation_v2_no_ready_joints",
            "ready contract contains no joint records",
        )
    prepared: list[_PreparedJoint] = []
    paths: set[str] = set()
    for record in joints:
        if record.review_status != "ready_for_rigger_input":
            _fail(
                "articulation_v2_joint_not_ready",
                f"joint {record.joint_id!r} is not ready_for_rigger_input",
            )
        if not isinstance(record.attachments, ExplicitAttachmentFramesV2):
            _fail(
                "articulation_v2_explicit_frames_required",
                f"joint {record.joint_id!r} does not carry explicit paired frames",
            )
        local_pos0_stage = _attachment_position_meters_to_stage_units(
            record.attachments.body0.position_meters,
            meters_per_unit=meters_per_unit,
            field=f"joint {record.joint_id!r} body0",
        )
        local_pos1_stage = _attachment_position_meters_to_stage_units(
            record.attachments.body1.position_meters,
            meters_per_unit=meters_per_unit,
            field=f"joint {record.joint_id!r} body1",
        )
        local_rot0 = _attachment_orientation_to_target_binary32(
            record.attachments.body0.orientation_wxyz,
            field=f"joint {record.joint_id!r} body0.orientation_wxyz",
        )
        local_rot1 = _attachment_orientation_to_target_binary32(
            record.attachments.body1.orientation_wxyz,
            field=f"joint {record.joint_id!r} body1.orientation_wxyz",
        )
        revolute_limits_degrees: tuple[float, float] | None = None
        if (
            isinstance(record.constraint, RevoluteConstraintV2)
            and record.constraint.limit_mode == "bounded"
        ):
            lower_degrees = record.constraint.lower_degrees
            upper_degrees = record.constraint.upper_degrees
            if lower_degrees is None or upper_degrees is None:
                _fail(
                    "articulation_v2_bounded_limits_incomplete",
                    f"joint {record.joint_id!r} has incomplete revolute limits",
                )
            revolute_limits_degrees = _revolute_limits_to_target_binary32(
                lower_degrees,
                upper_degrees,
            )
        prismatic_limits_stage_units: tuple[float, float] | None = None
        if (
            isinstance(record.constraint, PrismaticConstraintV2)
            and record.constraint.limit_mode == "bounded"
        ):
            lower_meters = record.constraint.lower_meters
            upper_meters = record.constraint.upper_meters
            if lower_meters is None or upper_meters is None:
                _fail(
                    "articulation_v2_bounded_limits_incomplete",
                    f"joint {record.joint_id!r} has incomplete prismatic limits",
                )
            prismatic_limits_stage_units = _prismatic_limits_meters_to_stage_units(
                lower_meters,
                upper_meters,
                meters_per_unit=meters_per_unit,
            )
        frame_constraint_kind = _mobile_frame_constraint_kind(record.constraint)
        fixed_distance = isinstance(
            record.constraint,
            FixedConstraintV2 | DistanceConstraintV2,
        )
        if frame_constraint_kind is None and not fixed_distance:
            _fail(
                "articulation_v2_constraint_not_supported_by_frame_authorer",
                f"joint {record.joint_id!r} uses an unsupported constraint object; "
                "fixed and distance authoring have separate capability gates",
            )
        if isinstance(record.constraint, DistanceConstraintV2) and (
            record.constraint.minimum_meters is None
            or record.constraint.maximum_meters is None
        ):
            _fail(
                "articulation_v2_distance_bounds_incomplete",
                f"joint {record.joint_id!r} requires explicit source-backed "
                "minimum and maximum bounds",
            )
        distance_bounds_stage_units = (
            _distance_bounds_meters_to_stage_units(
                cast(float, record.constraint.minimum_meters),
                cast(float, record.constraint.maximum_meters),
                meters_per_unit=meters_per_unit,
            )
            if isinstance(record.constraint, DistanceConstraintV2)
            else None
        )
        body0_link = links.get(record.body0_link)
        body1_link = links.get(record.body1_link)
        if body0_link is None or body1_link is None:
            _fail(
                "articulation_v2_link_missing",
                f"joint {record.joint_id!r} references an undeclared link",
            )
        if fixed_distance and (
            body0_link.body_authoring != "existing"
            or body1_link.body_authoring != "existing"
        ):
            _fail(
                "articulation_v2_constraint_aggregate_link_unsupported",
                f"joint {record.joint_id!r} requires two existing rigid-body "
                "links; aggregate-link creation remains a separate capability",
            )
        body0 = _require_endpoint(
            stage,
            body0_link.body_prim_path,
            label=f"joint {record.joint_id!r} body0",
            Sdf=Sdf,
            UsdGeom=UsdGeom,
        )
        body1 = _require_endpoint(
            stage,
            body1_link.body_prim_path,
            label=f"joint {record.joint_id!r} body1",
            Sdf=Sdf,
            UsdGeom=UsdGeom,
        )
        for body_index, body in enumerate((body0, body1)):
            _require_static_endpoint_transform(
                body,
                label=f"joint {record.joint_id!r} body{body_index}",
            )
            _require_invertible_world_transform(
                stage,
                body,
                label=f"joint {record.joint_id!r} body{body_index}",
            )
        if fixed_distance:
            _require_distinct_rigid_body_prims(
                body0,
                body1,
                joint_id=record.joint_id,
            )
            _require_exact_fixed_distance_record_evidence(record)
        frame0 = _validate_endpoint_frame(
            stage,
            body0,
            record.attachments.body0,
            meters_per_unit=meters_per_unit,
            label=f"joint {record.joint_id!r} body0",
        )
        frame1 = _validate_endpoint_frame(
            stage,
            body1,
            record.attachments.body1,
            meters_per_unit=meters_per_unit,
            label=f"joint {record.joint_id!r} body1",
        )
        axis = (
            record.constraint.axis_stage
            if isinstance(
                record.constraint,
                RevoluteConstraintV2 | PrismaticConstraintV2,
            )
            else None
        )
        if (
            isinstance(
                record.constraint,
                RevoluteConstraintV2 | PrismaticConstraintV2,
            )
            and axis is None
        ):  # pragma: no cover - v2 ready invariant
            _fail(
                "articulation_v2_axis_missing",
                f"joint {record.joint_id!r} has no stage axis",
            )
        if frame_constraint_kind is not None:
            _require_full_frame_coherence(
                constraint_kind=frame_constraint_kind,
                world_frame0=frame0,
                world_frame1=frame1,
                expected_axis=axis,
                label=f"joint {record.joint_id!r}",
                axis_error_code="articulation_v2_axis_frame_conflict",
                anchor_error_code=("articulation_v2_endpoint_anchor_frame_conflict"),
                transverse_anchor_error_code=(
                    "articulation_v2_transverse_anchor_frame_conflict"
                ),
            )
        elif isinstance(record.constraint, FixedConstraintV2):
            _require_fixed_frame_coherence(
                world_frame0=frame0,
                world_frame1=frame1,
                label=f"joint {record.joint_id!r}",
                anchor_error_code="articulation_v2_endpoint_anchor_frame_conflict",
                orientation_error_code=(
                    "articulation_v2_endpoint_orientation_frame_conflict"
                ),
            )

        stored_attachments = ExplicitAttachmentFramesV2(
            kind="explicit",
            body0=AttachmentFrameV2(
                position_meters=_multiply_vector(
                    local_pos0_stage,
                    meters_per_unit,
                ),
                orientation_wxyz=local_rot0,
            ),
            body1=AttachmentFrameV2(
                position_meters=_multiply_vector(
                    local_pos1_stage,
                    meters_per_unit,
                ),
                orientation_wxyz=local_rot1,
            ),
        )
        stored_frame0 = _validate_endpoint_frame(
            stage,
            body0,
            stored_attachments.body0,
            meters_per_unit=meters_per_unit,
            label=f"joint {record.joint_id!r} body0 target storage",
        )
        stored_frame1 = _validate_endpoint_frame(
            stage,
            body1,
            stored_attachments.body1,
            meters_per_unit=meters_per_unit,
            label=f"joint {record.joint_id!r} body1 target storage",
        )
        if frame_constraint_kind is not None:
            _require_full_frame_coherence(
                constraint_kind=frame_constraint_kind,
                world_frame0=stored_frame0,
                world_frame1=stored_frame1,
                expected_axis=axis,
                label=f"joint {record.joint_id!r} target storage",
                axis_error_code="articulation_v2_axis_frame_conflict",
                anchor_error_code="articulation_v2_endpoint_anchor_frame_conflict",
                transverse_anchor_error_code=(
                    "articulation_v2_transverse_anchor_frame_conflict"
                ),
            )
        elif isinstance(record.constraint, FixedConstraintV2):
            _require_fixed_frame_coherence(
                world_frame0=stored_frame0,
                world_frame1=stored_frame1,
                label=f"joint {record.joint_id!r} target storage",
                anchor_error_code="articulation_v2_endpoint_anchor_frame_conflict",
                orientation_error_code=(
                    "articulation_v2_endpoint_orientation_frame_conflict"
                ),
            )

        sanitized = Tf.MakeValidIdentifier(record.joint_id)
        suffix = hashlib.sha256(record.joint_id.encode("utf-8")).hexdigest()[:12]
        joint_path = str(Sdf.Path(scope_path).AppendChild(f"{sanitized}_{suffix}"))
        if joint_path in paths or stage.GetPrimAtPath(joint_path).IsValid():
            _fail(
                "articulation_v2_joint_target_collision",
                f"joint {record.joint_id!r} resolves to occupied path {joint_path}",
            )
        paths.add(joint_path)
        evidence_json = _canonical_field_evidence(record.field_evidence)
        prepared.append(
            _PreparedJoint(
                record=record,
                frame_constraint_kind=frame_constraint_kind,
                joint_path=joint_path,
                body0_prim_path=body0_link.body_prim_path,
                body1_prim_path=body1_link.body_prim_path,
                local_pos0_stage=local_pos0_stage,
                local_pos1_stage=local_pos1_stage,
                local_rot0=local_rot0,
                local_rot1=local_rot1,
                revolute_limits_degrees=revolute_limits_degrees,
                prismatic_limits_stage_units=prismatic_limits_stage_units,
                distance_bounds_stage_units=distance_bounds_stage_units,
                evidence_json=evidence_json,
                evidence_sha256=hashlib.sha256(
                    evidence_json.encode("utf-8")
                ).hexdigest(),
            )
        )
    return (
        tuple(sorted(prepared, key=lambda item: item.record.joint_id)),
        scope_path,
        create_scope,
    )


def _author_prepared_joints(
    stage: Any,
    prepared: tuple[_PreparedJoint, ...],
    *,
    scope_path: str,
    create_scope: bool,
    contract_sha256: str,
) -> None:
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    if create_scope:
        scope = UsdGeom.Scope.Define(stage, Sdf.Path(scope_path))
        if not scope or not scope.GetPrim().IsValid():
            _fail(
                "articulation_v2_joint_scope_authoring_failed",
                f"could not define joint scope {scope_path}",
            )
    schemas: Mapping[str, Any] = {
        "revolute": UsdPhysics.RevoluteJoint,
        "prismatic": UsdPhysics.PrismaticJoint,
        "spherical": UsdPhysics.SphericalJoint,
        "fixed": UsdPhysics.FixedJoint,
        "distance": UsdPhysics.DistanceJoint,
    }
    for item in prepared:
        record = item.record
        schema = schemas[record.constraint.kind]
        joint = schema.Define(stage, Sdf.Path(item.joint_path))
        prim = joint.GetPrim()
        if not prim or not prim.IsValid():
            _fail(
                "articulation_v2_joint_authoring_failed",
                f"could not define joint {item.joint_path}",
            )
        _require_set(
            joint.CreateBody0Rel().SetTargets([Sdf.Path(item.body0_prim_path)]),
            f"{item.joint_path} body0",
        )
        _require_set(
            joint.CreateBody1Rel().SetTargets([Sdf.Path(item.body1_prim_path)]),
            f"{item.joint_path} body1",
        )
        _require_set(
            joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*item.local_pos0_stage)),
            f"{item.joint_path} localPos0",
        )
        _require_set(
            joint.CreateLocalPos1Attr().Set(Gf.Vec3f(*item.local_pos1_stage)),
            f"{item.joint_path} localPos1",
        )
        _require_set(
            joint.CreateLocalRot0Attr().Set(_quatf(item.local_rot0, Gf=Gf)),
            f"{item.joint_path} localRot0",
        )
        _require_set(
            joint.CreateLocalRot1Attr().Set(_quatf(item.local_rot1, Gf=Gf)),
            f"{item.joint_path} localRot1",
        )
        if isinstance(
            record.constraint,
            RevoluteConstraintV2 | PrismaticConstraintV2,
        ):
            _require_set(
                joint.CreateAxisAttr().Set(UsdPhysics.Tokens.x),
                f"{item.joint_path} axis",
            )
        if (
            isinstance(record.constraint, RevoluteConstraintV2)
            and record.constraint.limit_mode == "bounded"
        ):
            revolute_limits = item.revolute_limits_degrees
            if (
                record.constraint.lower_degrees is None
                or record.constraint.upper_degrees is None
                or revolute_limits is None
            ):
                _fail(
                    "articulation_v2_bounded_limits_incomplete",
                    f"prepared joint {record.joint_id!r} has no revolute limits",
                )
            lower_degrees, upper_degrees = revolute_limits
            _require_set(
                joint.CreateLowerLimitAttr().Set(lower_degrees),
                f"{item.joint_path} lowerLimit",
            )
            _require_set(
                joint.CreateUpperLimitAttr().Set(upper_degrees),
                f"{item.joint_path} upperLimit",
            )
        if (
            isinstance(record.constraint, PrismaticConstraintV2)
            and record.constraint.limit_mode == "bounded"
        ):
            prismatic_limits = item.prismatic_limits_stage_units
            if (
                record.constraint.lower_meters is None
                or record.constraint.upper_meters is None
                or prismatic_limits is None
            ):
                _fail(
                    "articulation_v2_bounded_limits_incomplete",
                    f"prepared joint {record.joint_id!r} has no prismatic limits",
                )
            lower_stage, upper_stage = prismatic_limits
            _require_set(
                joint.CreateLowerLimitAttr().Set(lower_stage),
                f"{item.joint_path} lowerLimit",
            )
            _require_set(
                joint.CreateUpperLimitAttr().Set(upper_stage),
                f"{item.joint_path} upperLimit",
            )
        if isinstance(record.constraint, DistanceConstraintV2):
            distance_bounds = item.distance_bounds_stage_units
            if distance_bounds is None:  # pragma: no cover - preparation invariant
                _fail(
                    "articulation_v2_distance_bounds_incomplete",
                    f"prepared joint {record.joint_id!r} has no distance bounds",
                )
            minimum_stage, maximum_stage = distance_bounds
            _require_set(
                joint.CreateMinDistanceAttr().Set(minimum_stage),
                f"{item.joint_path} minDistance",
            )
            _require_set(
                joint.CreateMaxDistanceAttr().Set(maximum_stage),
                f"{item.joint_path} maxDistance",
            )
        metadata = {
            "authoringVersion": _AUTHORING_VERSION,
            "contractSha256": contract_sha256,
            "evidenceJson": item.evidence_json,
            "evidenceSha256": item.evidence_sha256,
            "jointId": record.joint_id,
        }
        for key, value in metadata.items():
            _set_custom_data_exact(
                prim,
                key=f"jointAgentV2:{key}",
                value=value,
                label=f"{item.joint_path} {key} customData",
            )


def _distance_bounds_meters_to_stage_units(
    minimum: float,
    maximum: float,
    *,
    meters_per_unit: float,
) -> tuple[float, float]:
    minimum_stage = _distance_meters_to_stage_units(
        minimum,
        meters_per_unit=meters_per_unit,
        field="minimum_meters",
    )
    maximum_stage = _distance_meters_to_stage_units(
        maximum,
        meters_per_unit=meters_per_unit,
        field="maximum_meters",
    )
    if minimum < maximum and minimum_stage >= maximum_stage:
        _fail(
            "articulation_v2_distance_unit_conversion_precision_loss",
            "distinct distance bounds collapse in target stage units",
        )
    return minimum_stage, maximum_stage


def _validate_saved_readback(
    stage: Any,
    prepared: tuple[_PreparedJoint, ...],
    *,
    contract_sha256: str,
) -> tuple[AuthoredAttachmentFrameReadbackV3, ...]:
    from pxr import Sdf, UsdGeom, UsdPhysics

    meters_per_unit = _meters_per_unit(stage)
    schemas: Mapping[str, Any] = {
        "revolute": UsdPhysics.RevoluteJoint,
        "prismatic": UsdPhysics.PrismaticJoint,
        "spherical": UsdPhysics.SphericalJoint,
        "fixed": UsdPhysics.FixedJoint,
        "distance": UsdPhysics.DistanceJoint,
    }
    readbacks: list[AuthoredAttachmentFrameReadbackV3] = []
    for item in prepared:
        record = item.record
        schema = schemas[record.constraint.kind]
        prim = stage.GetPrimAtPath(Sdf.Path(item.joint_path))
        if (
            not prim
            or not prim.IsValid()
            or not prim.IsActive()
            or not prim.IsDefined()
            or not prim.IsA(schema)
        ):
            _readback_fail(
                item,
                f"missing expected {record.constraint.kind} schema",
            )
        joint = schema(prim)
        if (
            _single_relationship_target(
                joint.GetBody0Rel(),
                field="physics:body0",
            )
            != item.body0_prim_path
            or _single_relationship_target(
                joint.GetBody1Rel(),
                field="physics:body1",
            )
            != item.body1_prim_path
        ):
            _readback_fail(item, "body endpoint relationships changed")
        local_pos0 = _readback_vector(
            joint.GetLocalPos0Attr(),
            field="physics:localPos0",
            item=item,
        )
        local_pos1 = _readback_vector(
            joint.GetLocalPos1Attr(),
            field="physics:localPos1",
            item=item,
        )
        if local_pos0 != item.local_pos0_stage or local_pos1 != item.local_pos1_stage:
            _readback_fail(
                item,
                "target-stage attachment positions differ from the approved "
                "binary32 values: "
                f"observed=({local_pos0!r}, {local_pos1!r}), "
                f"expected=({item.local_pos0_stage!r}, "
                f"{item.local_pos1_stage!r})",
            )
        local_rot0 = _readback_quaternion(
            joint.GetLocalRot0Attr(),
            field="physics:localRot0",
            item=item,
        )
        local_rot1 = _readback_quaternion(
            joint.GetLocalRot1Attr(),
            field="physics:localRot1",
            item=item,
        )
        if local_rot0 != item.local_rot0 or local_rot1 != item.local_rot1:
            _readback_fail(
                item,
                "target-stage attachment rotations differ from the approved "
                "binary32 values: "
                f"observed=({local_rot0!r}, {local_rot1!r}), "
                f"expected=({item.local_rot0!r}, {item.local_rot1!r})",
            )
        attachments = ExplicitAttachmentFramesV2(
            kind="explicit",
            body0=AttachmentFrameV2(
                position_meters=_multiply_vector(local_pos0, meters_per_unit),
                orientation_wxyz=local_rot0,
            ),
            body1=AttachmentFrameV2(
                position_meters=_multiply_vector(local_pos1, meters_per_unit),
                orientation_wxyz=local_rot1,
            ),
        )
        expected_attachments = record.attachments
        if not isinstance(expected_attachments, ExplicitAttachmentFramesV2):
            _readback_fail(item, "contract no longer carries explicit paired frames")
        _require_frame_close(
            attachments.body0,
            expected_attachments.body0,
            item=item,
            field="body0",
        )
        _require_frame_close(
            attachments.body1,
            expected_attachments.body1,
            item=item,
            field="body1",
        )

        body0 = _require_endpoint(
            stage,
            item.body0_prim_path,
            label=f"readback joint {record.joint_id!r} body0",
            Sdf=Sdf,
            UsdGeom=UsdGeom,
        )
        body1 = _require_endpoint(
            stage,
            item.body1_prim_path,
            label=f"readback joint {record.joint_id!r} body1",
            Sdf=Sdf,
            UsdGeom=UsdGeom,
        )
        world_frame0 = _validate_endpoint_frame(
            stage,
            body0,
            attachments.body0,
            meters_per_unit=meters_per_unit,
            label=f"readback joint {record.joint_id!r} body0",
        )
        world_frame1 = _validate_endpoint_frame(
            stage,
            body1,
            attachments.body1,
            meters_per_unit=meters_per_unit,
            label=f"readback joint {record.joint_id!r} body1",
        )
        axis_stage: Vector3 | None = None
        if isinstance(
            record.constraint,
            RevoluteConstraintV2 | PrismaticConstraintV2,
        ):
            if joint.GetAxisAttr().Get() != UsdPhysics.Tokens.x:
                _readback_fail(item, "canonical USD axis token is not X")
            expected_axis = record.constraint.axis_stage
            if expected_axis is None:
                _readback_fail(
                    item,
                    "saved one-axis joint has no requested stage axis",
                )
        else:
            expected_axis = None
        if item.frame_constraint_kind is not None:
            axis_stage = _require_full_frame_coherence(
                constraint_kind=item.frame_constraint_kind,
                world_frame0=world_frame0,
                world_frame1=world_frame1,
                expected_axis=expected_axis,
                label=f"saved joint {record.joint_id!r}",
                axis_error_code="articulation_v2_saved_readback_mismatch",
                anchor_error_code="articulation_v2_saved_readback_mismatch",
                transverse_anchor_error_code=(
                    "articulation_v2_saved_readback_mismatch"
                ),
            )
        else:
            _require_distinct_rigid_body_prims(
                body0,
                body1,
                joint_id=record.joint_id,
            )
            if isinstance(record.constraint, FixedConstraintV2):
                _require_fixed_frame_coherence(
                    world_frame0=world_frame0,
                    world_frame1=world_frame1,
                    label=f"saved joint {record.joint_id!r}",
                    anchor_error_code="articulation_v2_saved_readback_mismatch",
                    orientation_error_code="articulation_v2_saved_readback_mismatch",
                )
        _validate_limit_readback(joint, item, meters_per_unit=meters_per_unit)
        expected_metadata = {
            "authoringVersion": _AUTHORING_VERSION,
            "contractSha256": contract_sha256,
            "evidenceJson": item.evidence_json,
            "evidenceSha256": item.evidence_sha256,
            "jointId": record.joint_id,
        }
        for key, expected in expected_metadata.items():
            if prim.GetCustomDataByKey(f"jointAgentV2:{key}") != expected:
                _readback_fail(
                    item,
                    f"customData {key} does not match the accepted contract",
                )
        if isinstance(
            record.constraint,
            FixedConstraintV2 | DistanceConstraintV2,
        ):
            try:
                _require_exact_fixed_distance_metadata(
                    prim,
                    joint_kind=record.constraint.kind,
                    expected_custom_data={"jointAgentV2": expected_metadata},
                )
            except ArticulationV2FrameAuthoringError as exc:
                _readback_fail(
                    item,
                    f"saved fixed/distance metadata is not exact: {exc}",
                )
        readbacks.append(
            AuthoredAttachmentFrameReadbackV3(
                joint_id=record.joint_id,
                joint_path=item.joint_path,
                body0_prim_path=item.body0_prim_path,
                body1_prim_path=item.body1_prim_path,
                attachments=attachments,
                constraint=record.constraint,
                axis_stage=axis_stage,
                field_evidence=record.field_evidence,
                evidence_sha256=item.evidence_sha256,
            )
        )
    return tuple(readbacks)


def _validate_limit_readback(
    joint: Any,
    item: _PreparedJoint,
    *,
    meters_per_unit: float,
) -> None:
    constraint = item.record.constraint
    if isinstance(constraint, RevoluteConstraintV2):
        lower = joint.GetLowerLimitAttr()
        upper = joint.GetUpperLimitAttr()
        if constraint.limit_mode == "bounded":
            observed_lower = lower.Get()
            observed_upper = upper.Get()
            expected_limits = item.revolute_limits_degrees
            if (
                observed_lower is None
                or observed_upper is None
                or expected_limits is None
            ):
                _readback_fail(
                    item,
                    "bounded revolute limits are unauthored or unprepared: "
                    f"lowerLimit={observed_lower!r}, "
                    f"upperLimit={observed_upper!r}",
                )
            observed_limits = (float(observed_lower), float(observed_upper))
            if observed_limits != expected_limits:
                _readback_fail(
                    item,
                    "target-stage revolute limits differ from the approved "
                    "binary32 values: "
                    f"observed={observed_limits!r}, "
                    f"expected={expected_limits!r}",
                )
            _require_scalar_close(
                float(observed_lower),
                constraint.lower_degrees,
                item=item,
                field="lowerLimit",
            )
            _require_scalar_close(
                float(observed_upper),
                constraint.upper_degrees,
                item=item,
                field="upperLimit",
            )
        elif lower.HasAuthoredValueOpinion() or upper.HasAuthoredValueOpinion():
            _readback_fail(item, "continuous revolute authored scalar limits")
    elif isinstance(constraint, PrismaticConstraintV2):
        lower = joint.GetLowerLimitAttr()
        upper = joint.GetUpperLimitAttr()
        if constraint.limit_mode == "bounded":
            observed_lower = lower.Get()
            observed_upper = upper.Get()
            if observed_lower is None or observed_upper is None:
                _readback_fail(
                    item,
                    "bounded prismatic limits are unauthored or blocked: "
                    f"lowerLimit={observed_lower!r}, "
                    f"upperLimit={observed_upper!r}",
                )
            expected_limits = item.prismatic_limits_stage_units
            if expected_limits is None:  # pragma: no cover - preparation invariant
                _readback_fail(item, "prepared prismatic limits are missing")
            observed_limits = (float(observed_lower), float(observed_upper))
            if observed_limits != expected_limits:
                _readback_fail(
                    item,
                    "target-stage prismatic limits differ from the approved "
                    "binary32 values: "
                    f"observed={observed_limits!r}, "
                    f"expected={expected_limits!r}",
                )
            _require_scalar_close(
                float(observed_lower) * meters_per_unit,
                constraint.lower_meters,
                item=item,
                field="lowerLimit",
            )
            _require_scalar_close(
                float(observed_upper) * meters_per_unit,
                constraint.upper_meters,
                item=item,
                field="upperLimit",
            )
        elif lower.HasAuthoredValueOpinion() or upper.HasAuthoredValueOpinion():
            _readback_fail(item, "unbounded prismatic authored scalar limits")
    elif isinstance(constraint, SphericalConstraintV2):
        _validate_free_spherical_readback(joint.GetPrim(), item=item)
    elif isinstance(constraint, FixedConstraintV2):
        try:
            _require_no_unsupported_fixed_distance_properties(
                joint.GetPrim(),
                joint_kind="fixed",
            )
        except ArticulationV2FrameAuthoringError as exc:
            _readback_fail(item, f"fixed readback carries motion semantics: {exc}")
    elif isinstance(constraint, DistanceConstraintV2):
        expected_bounds = item.distance_bounds_stage_units
        if expected_bounds is None:  # pragma: no cover - preparation invariant
            _readback_fail(item, "prepared distance bounds are missing")
        try:
            _require_no_unsupported_fixed_distance_properties(
                joint.GetPrim(),
                joint_kind="distance",
            )
            minimum, maximum = _require_source_distance_bounds(joint)
        except ArticulationV2FrameAuthoringError as exc:
            _readback_fail(item, f"distance readback is incomplete: {exc}")
        observed_bounds = (minimum, maximum)
        if observed_bounds != expected_bounds:
            _readback_fail(
                item,
                "target-stage distance bounds differ from the approved "
                "binary32 values: "
                f"observed={observed_bounds!r}, expected={expected_bounds!r}",
            )
        _require_scalar_close(
            minimum * meters_per_unit,
            constraint.minimum_meters,
            item=item,
            field="minDistance",
        )
        _require_scalar_close(
            maximum * meters_per_unit,
            constraint.maximum_meters,
            item=item,
            field="maxDistance",
        )
    else:
        _readback_fail(item, "saved joint has an unsupported constraint object")


def _validate_free_spherical_readback(
    prim: Any,
    *,
    item: _PreparedJoint,
) -> None:
    constraint = item.record.constraint
    if (
        not isinstance(constraint, SphericalConstraintV2)
        or constraint.angular_limit_mode != "free"
    ):
        _readback_fail(item, "spherical readback is not the free profile")
    authored, applied_controls = _free_spherical_control_state(prim)
    if authored or applied_controls:
        _readback_fail(
            item,
            "free spherical joint authored scalar limits or controls: "
            f"attributes={authored}, schemas={applied_controls}",
        )


def _source_frame_evidence(
    *,
    source_artifact: ArtifactIdentityV1,
    joint_path: str,
    provenance_source: ProvenanceSource,
    axis_token_normalized: bool = True,
) -> tuple[FieldEvidenceV2, ...]:
    if provenance_source not in _SOURCE_BACKED_PROVENANCE_SOURCES:
        _fail(
            "articulation_v2_provenance_not_source_backed",
            f"unsupported frame provenance source: {provenance_source}",
        )
    evidence = []
    for field, properties in _SOURCE_FRAME_PROPERTY_BY_FIELD.items():
        evidence.append(
            FieldEvidenceV2(
                field=field,
                provenance=FieldProvenanceV1(
                    source=provenance_source,
                    artifact=source_artifact,
                    prim_path=joint_path,
                    properties=properties,
                    derivation=(
                        "usd_stage_units_to_body_local_meters"
                        if field.endswith("position_meters")
                        else (
                            "usd_axis_token_frame_to_canonical_positive_x"
                            if field.endswith("orientation_wxyz")
                            and axis_token_normalized
                            else (
                                "exact_usd_joint_frame_orientation"
                                if field.endswith("orientation_wxyz")
                                else None
                            )
                        )
                    ),
                    evidence=(
                        "Exact authored USD joint endpoint/frame opinion promoted "
                        "without geometry or model inference."
                    ),
                ),
            )
        )
    return tuple(sorted(evidence, key=lambda item: item.field))


def _source_fixed_distance_evidence(
    *,
    source_artifact: ArtifactIdentityV1,
    joint_path: str,
    constraint: _FixedDistanceConstraint,
    provenance_source: ProvenanceSource,
) -> tuple[FieldEvidenceV2, ...]:
    if provenance_source not in _SOURCE_BACKED_PROVENANCE_SOURCES:
        _fail(
            "articulation_v2_provenance_not_source_backed",
            f"unsupported constraint provenance source: {provenance_source}",
        )
    property_by_field: dict[str, tuple[str, ...]] = {
        "constraint.kind": (
            "usd:schema:FixedJoint"
            if isinstance(constraint, FixedConstraintV2)
            else "usd:schema:DistanceJoint",
        ),
    }
    if isinstance(constraint, DistanceConstraintV2):
        property_by_field.update(
            {
                "constraint.minimum_meters": ("physics:minDistance",),
                "constraint.maximum_meters": ("physics:maxDistance",),
            }
        )
    return tuple(
        FieldEvidenceV2(
            field=field,
            provenance=FieldProvenanceV1(
                source=provenance_source,
                artifact=source_artifact,
                prim_path=joint_path,
                properties=properties,
                derivation=(
                    "usd_stage_units_to_meters"
                    if field.endswith("_meters")
                    else "exact_usd_joint_schema"
                ),
                evidence=(
                    "Exact authored USD constraint opinion promoted without "
                    "model, geometry, or template inference."
                ),
            ),
        )
        for field, properties in sorted(property_by_field.items())
    )


def _expected_fixed_distance_source_fields(
    constraint: _FixedDistanceConstraint,
) -> set[str]:
    fields = {
        "attachments.body0.orientation_wxyz",
        "attachments.body0.position_meters",
        "attachments.body1.orientation_wxyz",
        "attachments.body1.position_meters",
        "attachments.kind",
        "body0_link",
        "body1_link",
        "constraint.kind",
    }
    if isinstance(constraint, DistanceConstraintV2):
        fields.update(
            {
                "constraint.minimum_meters",
                "constraint.maximum_meters",
            }
        )
    return fields


def _require_exact_fixed_distance_record_evidence(record: JointRecordV2) -> None:
    constraint = cast(_FixedDistanceConstraint, record.constraint)
    expected_fields = _expected_fixed_distance_source_fields(constraint)
    evidence = {item.field: item.provenance for item in record.field_evidence}
    if set(evidence) != expected_fields:
        _fail(
            "articulation_v2_fixed_distance_evidence_incomplete",
            f"joint {record.joint_id!r} fixed/distance evidence does not exactly "
            f"cover applicable fields: observed={sorted(evidence)}, "
            f"expected={sorted(expected_fields)}",
        )
    expected_properties = {
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
        "constraint.kind": (
            "usd:schema:FixedJoint"
            if isinstance(constraint, FixedConstraintV2)
            else "usd:schema:DistanceJoint",
        ),
    }
    if isinstance(constraint, DistanceConstraintV2):
        expected_properties.update(
            {
                "constraint.minimum_meters": ("physics:minDistance",),
                "constraint.maximum_meters": ("physics:maxDistance",),
            }
        )
    locators = {
        (item.artifact, item.prim_path, item.source) for item in evidence.values()
    }
    invalid_properties = sorted(
        field
        for field, provenance in evidence.items()
        if provenance.properties != expected_properties[field]
    )
    if len(locators) != 1 or invalid_properties:
        _fail(
            "articulation_v2_fixed_distance_evidence_conflict",
            f"joint {record.joint_id!r} requires one exact source joint locator "
            "and exact source properties: "
            f"locator_count={len(locators)}, "
            f"invalid_properties={invalid_properties}",
        )


def _require_distinct_rigid_body_prims(
    body0: Any,
    body1: Any,
    *,
    joint_id: str | None,
) -> None:
    from pxr import UsdPhysics

    label = f"joint {joint_id!r}" if joint_id is not None else "constraint"
    if body0.GetPath() == body1.GetPath():
        _fail(
            "articulation_v2_constraint_same_body",
            f"{label} endpoints must identify distinct rigid bodies",
        )
    for index, body in enumerate((body0, body1)):
        if not body.HasAPI(UsdPhysics.RigidBodyAPI):
            _fail(
                "articulation_v2_constraint_endpoint_not_rigid_body",
                f"{label} body{index} is not an explicit rigid body: {body.GetPath()}",
            )


def _require_no_conflicting_source_edge(
    stage: Any,
    *,
    target_path: str,
    body0_path: str,
    body1_path: str,
    target_kind: Literal["fixed", "distance"],
    Usd: Any,
    UsdPhysics: Any,
) -> None:
    expected_pair = frozenset((body0_path, body1_path))
    mobile_schemas = (
        UsdPhysics.RevoluteJoint,
        UsdPhysics.PrismaticJoint,
        UsdPhysics.SphericalJoint,
        UsdPhysics.DistanceJoint,
    )
    for prim in Usd.PrimRange.Stage(
        stage,
        Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate),
    ):
        path = str(prim.GetPath())
        if path == target_path or not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        body0_targets = tuple(
            str(target) for target in joint.GetBody0Rel().GetTargets()
        )
        body1_targets = tuple(
            str(target) for target in joint.GetBody1Rel().GetTargets()
        )
        has_distinct_single_targets = (
            len(body0_targets) == 1
            and len(body1_targets) == 1
            and body0_targets[0] != body1_targets[0]
        )
        if not has_distinct_single_targets:
            observed_targets = frozenset((*body0_targets, *body1_targets))
            if observed_targets.intersection(expected_pair):
                _fail(
                    "articulation_v2_competing_source_edge_malformed",
                    f"competing source joint {path} has malformed endpoints "
                    f"intersecting {target_path}: body0={body0_targets}, "
                    f"body1={body1_targets}",
                )
            continue
        candidate_pair = frozenset(
            (
                body0_targets[0],
                body1_targets[0],
            )
        )
        if candidate_pair != expected_pair:
            continue
        if target_kind == "fixed" and any(
            prim.IsA(schema) for schema in mobile_schemas
        ):
            _fail(
                "articulation_v2_fixed_mobile_evidence_conflict",
                f"fixed source {target_path} conflicts with mobile edge {path}",
            )
        _fail(
            "articulation_v2_constraint_edge_conflict",
            "source contains multiple constraints for endpoints "
            f"{sorted(expected_pair)}: {target_path}, {path}",
        )


def _require_no_unsupported_fixed_distance_properties(
    prim: Any,
    *,
    joint_kind: Literal["fixed", "distance"],
) -> None:
    allowed_properties = {
        "physics:body0",
        "physics:body1",
        "physics:localPos0",
        "physics:localPos1",
        "physics:localRot0",
        "physics:localRot1",
    }
    if joint_kind == "distance":
        allowed_properties.update(
            {
                "physics:minDistance",
                "physics:maxDistance",
            }
        )
    unsupported_properties = tuple(
        sorted(
            str(prop.GetName())
            for prop in prim.GetAuthoredProperties()
            if str(prop.GetName()) not in allowed_properties
        )
    )
    unsupported_schemas = tuple(sorted(str(name) for name in prim.GetAppliedSchemas()))
    if unsupported_properties or unsupported_schemas:
        code = (
            "articulation_v2_fixed_motion_opinion_conflict"
            if joint_kind == "fixed"
            else "articulation_v2_distance_motion_opinion_conflict"
        )
        _fail(
            code,
            f"{joint_kind} source contains semantics that this authorer would "
            "silently drop: "
            f"properties={unsupported_properties}, schemas={unsupported_schemas}",
        )


def _require_exact_fixed_distance_metadata(
    prim: Any,
    *,
    joint_kind: Literal["fixed", "distance"],
    expected_custom_data: Mapping[str, Any] | None,
) -> None:
    from pxr import Sdf

    expected_prim_metadata = {
        "specifier": Sdf.SpecifierDef,
        "typeName": (
            "PhysicsFixedJoint" if joint_kind == "fixed" else "PhysicsDistanceJoint"
        ),
    }
    if expected_custom_data is not None:
        expected_prim_metadata["customData"] = dict(expected_custom_data)
    expected_property_metadata: dict[str, dict[str, Any]] = {
        "physics:body0": {
            "custom": False,
            "variability": Sdf.VariabilityUniform,
        },
        "physics:body1": {
            "custom": False,
            "variability": Sdf.VariabilityUniform,
        },
        "physics:localPos0": {
            "custom": False,
            "typeName": "point3f",
            "variability": Sdf.VariabilityVarying,
        },
        "physics:localPos1": {
            "custom": False,
            "typeName": "point3f",
            "variability": Sdf.VariabilityVarying,
        },
        "physics:localRot0": {
            "custom": False,
            "typeName": "quatf",
            "variability": Sdf.VariabilityVarying,
        },
        "physics:localRot1": {
            "custom": False,
            "typeName": "quatf",
            "variability": Sdf.VariabilityVarying,
        },
    }
    if joint_kind == "distance":
        expected_property_metadata.update(
            {
                "physics:minDistance": {
                    "custom": False,
                    "typeName": "float",
                    "variability": Sdf.VariabilityVarying,
                },
                "physics:maxDistance": {
                    "custom": False,
                    "typeName": "float",
                    "variability": Sdf.VariabilityVarying,
                },
            }
        )
    actual_prim_metadata = prim.GetAllAuthoredMetadata()
    invalid_property_metadata = tuple(
        sorted(
            str(prop.GetName())
            for prop in prim.GetAuthoredProperties()
            if str(prop.GetName()) in expected_property_metadata
            and prop.GetAllAuthoredMetadata()
            != expected_property_metadata[str(prop.GetName())]
        )
    )
    connected_frame_properties = tuple(
        sorted(
            name
            for name in (
                "physics:localPos0",
                "physics:localPos1",
                "physics:localRot0",
                "physics:localRot1",
            )
            if (attribute := prim.GetAttribute(name)) and attribute.GetConnections()
        )
    )
    if (
        actual_prim_metadata != expected_prim_metadata
        or invalid_property_metadata
        or connected_frame_properties
    ):
        code = (
            "articulation_v2_fixed_source_metadata_conflict"
            if joint_kind == "fixed"
            else "articulation_v2_distance_source_metadata_conflict"
        )
        _fail(
            code,
            f"{joint_kind} source contains authored metadata that this "
            "authorer would silently drop: "
            f"prim_metadata_keys={tuple(sorted(actual_prim_metadata))}, "
            f"invalid_property_metadata={invalid_property_metadata}, "
            f"connected_frame_properties={connected_frame_properties}",
        )


def _require_source_distance_bounds(joint: Any) -> tuple[float, float]:
    minimum_attr = joint.GetMinDistanceAttr()
    maximum_attr = joint.GetMaxDistanceAttr()
    minimum_authored = bool(minimum_attr and minimum_attr.HasAuthoredValueOpinion())
    maximum_authored = bool(maximum_attr and maximum_attr.HasAuthoredValueOpinion())
    if minimum_authored != maximum_authored:
        _fail(
            "articulation_v2_distance_bounds_partial",
            "distance source must author both minimum and maximum bounds",
        )
    if not minimum_authored:
        _fail(
            "articulation_v2_distance_bounds_default_or_inferred",
            "distance schema fallbacks or inferred defaults are not evidence",
        )
    minimum_value = _require_explicit_static_attribute_default(
        minimum_attr,
        field="physics:minDistance",
        error_code="articulation_v2_distance_bound_state_invalid",
    )
    maximum_value = _require_explicit_static_attribute_default(
        maximum_attr,
        field="physics:maxDistance",
        error_code="articulation_v2_distance_bound_state_invalid",
    )
    try:
        minimum = float(minimum_value)
        maximum = float(maximum_value)
    except (TypeError, ValueError) as exc:
        _fail(
            "articulation_v2_distance_bounds_non_finite",
            f"distance bounds must be numeric: {exc}",
        )
    if not math.isfinite(minimum) or not math.isfinite(maximum):
        _fail(
            "articulation_v2_distance_bounds_non_finite",
            "distance bounds must be finite",
        )
    if minimum < 0.0 or maximum < 0.0:
        _fail(
            "articulation_v2_distance_bounds_negative",
            f"distance bounds must be non-negative: min={minimum}, max={maximum}",
        )
    if minimum > maximum:
        _fail(
            "articulation_v2_distance_bounds_inverted",
            f"distance minimum must not exceed maximum: min={minimum}, max={maximum}",
        )
    return minimum, maximum


def _distance_bounds_stage_units_to_meters(
    minimum: float,
    maximum: float,
    *,
    meters_per_unit: float,
) -> tuple[float, float]:
    minimum_meters = _distance_stage_units_to_meters(
        minimum,
        meters_per_unit=meters_per_unit,
        field="physics:minDistance",
    )
    maximum_meters = _distance_stage_units_to_meters(
        maximum,
        meters_per_unit=meters_per_unit,
        field="physics:maxDistance",
    )
    if minimum < maximum and minimum_meters >= maximum_meters:
        _fail(
            "articulation_v2_distance_unit_conversion_precision_loss",
            "distinct source distance bounds collapse in meters",
        )
    return minimum_meters, maximum_meters


def _distance_stage_units_to_meters(
    value: float,
    *,
    meters_per_unit: float,
    field: str,
) -> float:
    result = value * meters_per_unit
    if not math.isfinite(result):
        _fail(
            "articulation_v2_distance_unit_conversion_overflow",
            f"{field} cannot be represented in meters",
        )
    if value != 0.0 and result == 0.0:
        _fail(
            "articulation_v2_distance_unit_conversion_underflow",
            f"{field} collapses to zero meters",
        )
    round_trip_stage_units = result / meters_per_unit
    # Inputs and meters_per_unit are finite, and the product is finite/nonzero.
    # Any remaining IEEE-754 round-trip anomaly is therefore one precision-loss
    # condition; math.isclose also rejects an unexpected zero or infinity.
    if not math.isclose(
        round_trip_stage_units,
        value,
        rel_tol=1e-6,
        abs_tol=0.0,
    ):
        _fail(
            "articulation_v2_distance_unit_conversion_precision_loss",
            f"{field} cannot round-trip through meters without material precision loss",
        )
    return result


def _distance_meters_to_stage_units(
    value: float,
    *,
    meters_per_unit: float,
    field: str,
) -> float:
    return _meters_to_target_binary32(
        value,
        meters_per_unit=meters_per_unit,
        field=field,
        error_code_prefix="articulation_v2_distance_unit_conversion",
    )


def _attachment_position_meters_to_stage_units(
    position: Vector3,
    *,
    meters_per_unit: float,
    field: str,
) -> Vector3:
    stored = tuple(
        _meters_to_target_binary32(
            component,
            meters_per_unit=meters_per_unit,
            field=f"{field}.position_meters[{index}]",
            error_code_prefix="articulation_v2_frame_unit_conversion",
        )
        for index, component in enumerate(position)
    )
    return (stored[0], stored[1], stored[2])


def _attachment_orientation_to_target_binary32(
    orientation: QuaternionWxyz,
    *,
    field: str,
) -> QuaternionWxyz:
    stored = tuple(
        _scalar_to_target_binary32(
            component,
            field=f"{field}[{index}]",
            error_code_prefix="articulation_v2_frame_orientation_storage",
        )
        for index, component in enumerate(orientation)
    )
    result = (stored[0], stored[1], stored[2], stored[3])
    try:
        canonical = _validated_orientation(result)
    except (TypeError, ValueError) as exc:
        _fail(
            "articulation_v2_frame_orientation_storage_precision_loss",
            f"{field} is not a valid unit quaternion after target storage: {exc}",
        )
    if canonical != result:
        _fail(
            "articulation_v2_frame_orientation_storage_precision_loss",
            f"{field} changes canonical quaternion representation in target storage",
        )
    return result


def _revolute_limits_to_target_binary32(
    lower: float,
    upper: float,
) -> tuple[float, float]:
    lower_stored = _scalar_to_target_binary32(
        lower,
        field="lower_degrees",
        error_code_prefix="articulation_v2_revolute_limit_storage",
    )
    upper_stored = _scalar_to_target_binary32(
        upper,
        field="upper_degrees",
        error_code_prefix="articulation_v2_revolute_limit_storage",
    )
    if lower < upper and lower_stored >= upper_stored:
        _fail(
            "articulation_v2_revolute_limit_storage_precision_loss",
            "distinct revolute limits collapse in target USD float attributes",
        )
    return lower_stored, upper_stored


def _prismatic_limits_meters_to_stage_units(
    lower: float,
    upper: float,
    *,
    meters_per_unit: float,
) -> tuple[float, float]:
    lower_stage = _meters_to_target_binary32(
        lower,
        meters_per_unit=meters_per_unit,
        field="lower_meters",
        error_code_prefix="articulation_v2_prismatic_unit_conversion",
    )
    upper_stage = _meters_to_target_binary32(
        upper,
        meters_per_unit=meters_per_unit,
        field="upper_meters",
        error_code_prefix="articulation_v2_prismatic_unit_conversion",
    )
    if lower < upper and lower_stage >= upper_stage:
        _fail(
            "articulation_v2_prismatic_unit_conversion_precision_loss",
            "distinct prismatic limits collapse in target USD float attributes",
        )
    return lower_stage, upper_stage


def _scalar_to_target_binary32(
    value: float,
    *,
    field: str,
    error_code_prefix: str,
) -> float:
    try:
        encoded = struct.pack(">f", value)
        stored = float(struct.unpack(">f", encoded)[0])
    except (OverflowError, struct.error):
        _fail(
            f"{error_code_prefix}_overflow",
            f"{field} exceeds the target USD float attribute range",
        )
    if not math.isfinite(stored):
        _fail(
            f"{error_code_prefix}_overflow",
            f"{field} exceeds the target USD float attribute range",
        )
    if value != 0.0 and stored == 0.0:
        _fail(
            f"{error_code_prefix}_underflow",
            f"{field} collapses to zero in the target USD float attribute",
        )
    relative_round_trip = math.isclose(
        stored,
        value,
        rel_tol=1e-6,
        abs_tol=0.0,
    )
    saved_readback_round_trip = math.isclose(
        stored,
        value,
        rel_tol=0.0,
        abs_tol=_READBACK_TOLERANCE,
    )
    if not relative_round_trip or not saved_readback_round_trip:
        _fail(
            f"{error_code_prefix}_precision_loss",
            f"{field} cannot round-trip through the target USD float "
            "attribute within the relative and saved-readback tolerances",
        )
    return stored


def _meters_to_target_binary32(
    value: float,
    *,
    meters_per_unit: float,
    field: str,
    error_code_prefix: str,
) -> float:
    result = value / meters_per_unit
    if not math.isfinite(result):
        _fail(
            f"{error_code_prefix}_overflow",
            f"{field} cannot be represented in target stage units",
        )
    if value != 0.0 and result == 0.0:
        _fail(
            f"{error_code_prefix}_underflow",
            f"{field} collapses to zero target stage units",
        )
    try:
        encoded = struct.pack(">f", result)
        stored = float(struct.unpack(">f", encoded)[0])
    except (OverflowError, struct.error):
        _fail(
            f"{error_code_prefix}_overflow",
            f"{field} exceeds the target USD float attribute range",
        )
    if not math.isfinite(stored):
        _fail(
            f"{error_code_prefix}_overflow",
            f"{field} exceeds the target USD float attribute range",
        )
    if value != 0.0 and stored == 0.0:
        _fail(
            f"{error_code_prefix}_underflow",
            f"{field} collapses to zero in the target USD float attribute",
        )
    round_trip_meters = stored * meters_per_unit
    if not math.isfinite(round_trip_meters):
        _fail(
            f"{error_code_prefix}_overflow",
            f"{field} target USD float cannot round-trip to meters",
        )
    if value != 0.0 and round_trip_meters == 0.0:
        _fail(
            f"{error_code_prefix}_underflow",
            f"{field} target USD float collapses to zero meters",
        )
    relative_round_trip = math.isclose(
        round_trip_meters,
        value,
        rel_tol=1e-6,
        abs_tol=0.0,
    )
    result_round_trip = math.isclose(
        round_trip_meters,
        value,
        rel_tol=0.0,
        abs_tol=_READBACK_TOLERANCE,
    )
    if not relative_round_trip or not result_round_trip:
        _fail(
            f"{error_code_prefix}_precision_loss",
            f"{field} cannot round-trip through the target USD float "
            "attribute within the relative and saved-readback tolerances",
        )
    return stored


type _RetainedRawFileState = tuple[int, int, int, int, int, int, int, int, int]


@contextmanager
def _retain_raw_usd_snapshot(
    path: Path,
) -> Iterator[tuple[Path, Path, str]]:
    """Retain one descriptor-copied raw USD snapshot and final source gate."""

    requested = Path(os.path.abspath(path.expanduser()))
    if requested.suffix.lower() not in {".usd", ".usda", ".usdc"}:
        _fail(
            "articulation_v2_output_format_mismatch",
            f"raw output snapshot requires a USD layer: {path}",
        )
    with tempfile.TemporaryDirectory(
        prefix="joint-articulation-v2-output-"
    ) as directory:
        workspace = Path(directory)
        workspace.chmod(0o700)
        snapshot = workspace / f"root{requested.suffix.lower()}"
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        flags |= getattr(os, "O_NONBLOCK", 0)
        source_descriptor = -1
        target_descriptor = -1
        try:
            source_descriptor = os.open(requested, flags)
            initial = os.fstat(source_descriptor)
            path_initial = os.stat(requested, follow_symlinks=False)
            initial_state = _retained_raw_file_state(initial)
            if (
                not stat.S_ISREG(initial.st_mode)
                or _retained_raw_file_state(path_initial) != initial_state
            ):
                _fail(
                    "articulation_v2_output_identity_unavailable",
                    f"generated raw USD is not a stable regular file: {requested}",
                )
            target_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            target_flags |= getattr(os, "O_CLOEXEC", 0)
            target_flags |= getattr(os, "O_NOFOLLOW", 0)
            target_descriptor = os.open(snapshot, target_flags, 0o600)
            digest = hashlib.sha256()
            offset = 0
            while offset < initial.st_size:
                chunk = os.pread(
                    source_descriptor,
                    min(1024 * 1024, initial.st_size - offset),
                    offset,
                )
                if not chunk:
                    _fail(
                        "articulation_v2_output_identity_unavailable",
                        f"generated raw USD changed while copied: {requested}",
                    )
                digest.update(chunk)
                remaining = memoryview(chunk)
                while remaining:
                    written = os.write(target_descriptor, remaining)
                    if written <= 0:  # pragma: no cover - regular-file invariant
                        raise OSError("raw USD snapshot write made no progress")
                    remaining = remaining[written:]
                offset += len(chunk)
            if os.pread(source_descriptor, 1, offset):
                _fail(
                    "articulation_v2_output_identity_unavailable",
                    f"generated raw USD grew while copied: {requested}",
                )
            os.fsync(target_descriptor)
            final = os.fstat(source_descriptor)
            path_final = os.stat(requested, follow_symlinks=False)
            if (
                _retained_raw_file_state(final) != initial_state
                or _retained_raw_file_state(path_final) != initial_state
            ):
                _fail(
                    "articulation_v2_output_identity_unavailable",
                    f"generated raw USD changed while copied: {requested}",
                )
        finally:
            if target_descriptor >= 0:
                os.close(target_descriptor)
            if source_descriptor >= 0:
                os.close(source_descriptor)

        snapshot.chmod(0o400)
        primary_error: BaseException | None = None
        try:
            yield requested.resolve(strict=True), snapshot, digest.hexdigest()
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            try:
                _require_retained_raw_source(
                    requested,
                    expected_state=initial_state,
                    expected_sha256=digest.hexdigest(),
                )
            except BaseException as recheck_error:
                if primary_error is None:
                    raise
                primary_error.add_note(
                    "Final retained raw USD output recheck also failed: "
                    f"{type(recheck_error).__name__}: {recheck_error}"
                )


def _require_retained_raw_source(
    path: Path,
    *,
    expected_state: _RetainedRawFileState,
    expected_sha256: str,
) -> None:
    """Perform the one final descriptor digest of a generated raw output."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    descriptor = -1
    try:
        descriptor = os.open(path, flags)
        initial = os.fstat(descriptor)
        path_initial = os.stat(path, follow_symlinks=False)
        if (
            _retained_raw_file_state(initial) != expected_state
            or _retained_raw_file_state(path_initial) != expected_state
        ):
            _fail(
                "articulation_v2_output_mutated",
                f"generated raw USD changed before final recheck: {path}",
            )
        digest = hashlib.sha256()
        offset = 0
        while offset < initial.st_size:
            chunk = os.pread(
                descriptor,
                min(1024 * 1024, initial.st_size - offset),
                offset,
            )
            if not chunk:
                _fail(
                    "articulation_v2_output_mutated",
                    f"generated raw USD changed during final recheck: {path}",
                )
            digest.update(chunk)
            offset += len(chunk)
        final = os.fstat(descriptor)
        path_final = os.stat(path, follow_symlinks=False)
        if (
            os.pread(descriptor, 1, offset)
            or _retained_raw_file_state(final) != expected_state
            or _retained_raw_file_state(path_final) != expected_state
            or digest.hexdigest() != expected_sha256
        ):
            _fail(
                "articulation_v2_output_mutated",
                f"generated raw USD changed during final recheck: {path}",
            )
    except ArticulationV2FrameAuthoringError:
        raise
    except OSError as exc:
        _fail(
            "articulation_v2_output_mutated",
            f"could not complete final raw USD recheck: {path}: {exc}",
        )
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _retained_raw_file_state(value: os.stat_result) -> _RetainedRawFileState:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _open_stage(path: Path, *, label: str) -> Any:
    try:
        from pxr import Usd
    except ImportError as exc:  # pragma: no cover - optional runtime guard
        _fail(
            "articulation_v2_openusd_unavailable",
            f"OpenUSD bindings are required: {exc}",
        )
    try:
        stage = Usd.Stage.Open(str(path))
    except Exception as exc:
        _fail(
            "articulation_v2_stage_open_failed",
            f"could not open {label}: {path}: {type(exc).__name__}: {exc}",
        )
    if stage is None:
        _fail(
            "articulation_v2_stage_open_failed",
            f"could not open {label}: {path}",
        )
    return stage


def _meters_per_unit(stage: Any) -> float:
    from pxr import UsdGeom

    value = float(UsdGeom.GetStageMetersPerUnit(stage))
    if not math.isfinite(value) or value <= 0.0:
        _fail(
            "articulation_v2_invalid_stage_units",
            f"metersPerUnit must be positive and finite; got {value!r}",
        )
    return value


def _require_endpoint(
    stage: Any,
    path_value: str,
    *,
    label: str,
    Sdf: Any,
    UsdGeom: Any,
    require_xformable: bool = True,
) -> Any:
    path = Sdf.Path(path_value)
    prim = stage.GetPrimAtPath(path)
    if (
        str(path) != path_value
        or not path.IsAbsolutePath()
        or not path.IsPrimPath()
        or path.IsAbsoluteRootPath()
        or not prim
        or not prim.IsValid()
    ):
        _fail(
            "articulation_v2_endpoint_missing",
            f"{label} does not resolve to an exact absolute prim path: {path_value}",
        )
    if not prim.IsActive() or not prim.IsDefined():
        _fail(
            "articulation_v2_endpoint_inactive_or_undefined",
            f"{label} must be active and defined: {path_value}",
        )
    if prim.IsPrototype() or prim.IsInPrototype():
        _fail(
            "articulation_v2_endpoint_prototype",
            f"{label} is in a prototype namespace: {path_value}",
        )
    if prim.IsInstanceProxy():
        _fail(
            "articulation_v2_endpoint_instance_proxy",
            f"{label} is an instance proxy: {path_value}",
        )
    if prim.IsInstance():
        _fail(
            "articulation_v2_endpoint_instance",
            f"{label} is an instance root: {path_value}",
        )
    if prim.IsAbstract():
        _fail(
            "articulation_v2_endpoint_abstract",
            f"{label} is abstract: {path_value}",
        )
    if require_xformable and not UsdGeom.Xformable(prim):
        _fail(
            "articulation_v2_endpoint_not_xformable",
            f"{label} is not transformable: {path_value}",
        )
    return prim


def _require_source_joint_prim(
    stage: Any,
    path_value: str,
    *,
    label: str,
) -> Any:
    from pxr import Sdf

    path = Sdf.Path(path_value)
    prim = stage.GetPrimAtPath(path)
    if (
        str(path) != path_value
        or not path.IsAbsolutePath()
        or not path.IsPrimPath()
        or path.IsAbsoluteRootPath()
        or not prim
        or not prim.IsValid()
    ):
        _fail(
            "articulation_v2_source_joint_missing",
            f"{label} does not resolve to an exact absolute prim path: {path_value}",
        )
    if not prim.IsActive() or not prim.IsDefined():
        _fail(
            "articulation_v2_source_joint_inactive_or_undefined",
            f"{label} must be active and defined: {path_value}",
        )
    if prim.IsPrototype() or prim.IsInPrototype():
        _fail(
            "articulation_v2_source_joint_prototype",
            f"{label} is in a prototype namespace: {path_value}",
        )
    if prim.IsInstanceProxy():
        _fail(
            "articulation_v2_source_joint_instance_proxy",
            f"{label} is an instance proxy: {path_value}",
        )
    if prim.IsInstance():
        _fail(
            "articulation_v2_source_joint_instance",
            f"{label} is an instance root: {path_value}",
        )
    if prim.IsAbstract():
        _fail(
            "articulation_v2_source_joint_abstract",
            f"{label} is abstract: {path_value}",
        )
    return prim


def _require_static_endpoint_transform(prim: Any, *, label: str) -> None:
    from pxr import UsdGeom

    current = prim
    while current.IsValid() and not current.IsPseudoRoot():
        xformable = UsdGeom.Xformable(current)
        if xformable:
            samples = sorted(
                {
                    float(sample)
                    for op in xformable.GetOrderedXformOps()
                    for sample in op.GetAttr().GetTimeSamples()
                }
            )
            if samples:
                _fail(
                    "articulation_v2_time_varying_endpoint_transform",
                    f"{label} transform chain has time samples at "
                    f"{current.GetPath()}: {samples}",
                )
        current = current.GetParent()


def _require_invertible_world_transform(
    stage: Any,
    prim: Any,
    *,
    label: str,
) -> Any:
    from pxr import UsdGeom

    matrix = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    values = tuple(float(value) for row in matrix for value in row)
    determinant = float(matrix.GetDeterminant())
    if (
        any(not math.isfinite(value) for value in values)
        or not math.isfinite(determinant)
        or math.isclose(determinant, 0.0, rel_tol=0.0, abs_tol=1e-12)
    ):
        _fail(
            "articulation_v2_singular_endpoint_transform",
            f"{label} world transform must be finite and invertible",
        )
    return matrix


def _validate_endpoint_frame(
    stage: Any,
    prim: Any,
    frame: AttachmentFrameV2,
    *,
    meters_per_unit: float,
    label: str,
) -> tuple[Vector3, tuple[Vector3, Vector3, Vector3]]:
    from pxr import Gf

    world_transform = _require_invertible_world_transform(
        stage,
        prim,
        label=label,
    )
    local_position = Gf.Vec3d(
        *(component / meters_per_unit for component in frame.position_meters)
    )
    world_position_stage = world_transform.Transform(local_position)
    world_position = tuple(
        float(component) * meters_per_unit for component in world_position_stage
    )
    if any(not math.isfinite(component) for component in world_position):
        _fail(
            "articulation_v2_non_finite_reprojected_frame",
            f"{label} position reprojection is non-finite",
        )
    rotation = Gf.Rotation(_quatd(frame.orientation_wxyz, Gf=Gf))
    directions = tuple(
        _normalized_direction(
            world_transform.TransformDir(rotation.TransformDir(base)),
            label=f"{label} frame direction",
        )
        for base in (
            Gf.Vec3d(1.0, 0.0, 0.0),
            Gf.Vec3d(0.0, 1.0, 0.0),
            Gf.Vec3d(0.0, 0.0, 1.0),
        )
    )
    if any(
        abs(_dot(directions[left], directions[right])) > _FRAME_TOLERANCE
        for left, right in ((0, 1), (0, 2), (1, 2))
    ) or _dot(_cross(directions[0], directions[1]), directions[2]) < (
        1.0 - _FRAME_TOLERANCE
    ):
        _fail(
            "articulation_v2_unsupported_endpoint_frame",
            f"{label} transform cannot preserve a right-handed orthonormal frame",
        )
    return (
        (world_position[0], world_position[1], world_position[2]),
        (
            directions[0],
            directions[1],
            directions[2],
        ),
    )


def _require_full_frame_coherence(
    *,
    constraint_kind: _FrameConstraintKind,
    world_frame0: tuple[Vector3, tuple[Vector3, Vector3, Vector3]],
    world_frame1: tuple[Vector3, tuple[Vector3, Vector3, Vector3]],
    expected_axis: Vector3 | None,
    label: str,
    axis_error_code: str,
    anchor_error_code: str,
    transverse_anchor_error_code: str | None = None,
) -> Vector3 | None:
    """Enforce the one physical coherence rule for a complete frame pair.

    Revolute and spherical anchors must coincide. Prismatic anchors may differ
    only along their common motion axis; any transverse separation is a
    full-frame conflict. One-axis joints must also reproduce the same signed
    stage-space axis from both endpoint frames. A prismatic joint additionally
    locks all angular motion, so its endpoint frames must reproduce the whole
    orthonormal basis; a relative twist about the shared motion axis is a
    conflict even though revolute frames may differ that way by design. The
    same physical tolerance is used at promotion, source revalidation,
    preparation, and saved readback so an admitted pair cannot become
    intrinsically unpublishable later.
    """

    tolerance = _FRAME_COHERENCE_TOLERANCE
    position0, basis0 = world_frame0
    position1, basis1 = world_frame1
    separation: Vector3 = (
        position1[0] - position0[0],
        position1[1] - position0[1],
        position1[2] - position0[2],
    )
    separation_norm = math.sqrt(sum(component * component for component in separation))

    if constraint_kind == "spherical":
        if separation_norm > tolerance:
            _fail(
                anchor_error_code,
                f"{label} spherical endpoint anchors differ by "
                f"{separation_norm:.9g} meters",
            )
        return None

    axis0 = basis0[0]
    axis1 = basis1[0]
    admitted_axis = axis0 if expected_axis is None else expected_axis
    if not _vectors_close(
        axis0,
        admitted_axis,
        tolerance=tolerance,
    ) or not _vectors_close(
        axis1,
        admitted_axis,
        tolerance=tolerance,
    ):
        _fail(
            axis_error_code,
            f"{label} endpoint frames do not reproduce one signed "
            f"stage-space motion axis {admitted_axis}",
        )

    if constraint_kind == "revolute":
        if separation_norm > tolerance:
            _fail(
                anchor_error_code,
                f"{label} revolute endpoint anchors differ by "
                f"{separation_norm:.9g} meters",
            )
        return axis0

    for index, ordinal in ((1, "second"), (2, "third")):
        if not _vectors_close(
            basis0[index],
            basis1[index],
            tolerance=tolerance,
        ):
            _fail(
                axis_error_code,
                f"{label} prismatic endpoint frames are rotated relative to "
                f"each other about the motion axis; their {ordinal} basis "
                f"directions are {basis0[index]} and {basis1[index]}",
            )

    longitudinal = _multiply_vector(admitted_axis, _dot(separation, admitted_axis))
    transverse: Vector3 = (
        separation[0] - longitudinal[0],
        separation[1] - longitudinal[1],
        separation[2] - longitudinal[2],
    )
    transverse_norm = math.sqrt(sum(component * component for component in transverse))
    if transverse_norm > tolerance:
        _fail(
            transverse_anchor_error_code or anchor_error_code,
            f"{label} prismatic endpoint anchor separation has "
            f"{transverse_norm:.9g} meters transverse to the motion axis",
        )
    return axis0


def _require_fixed_frame_coherence(
    *,
    world_frame0: tuple[Vector3, tuple[Vector3, Vector3, Vector3]],
    world_frame1: tuple[Vector3, tuple[Vector3, Vector3, Vector3]],
    label: str,
    anchor_error_code: str,
    orientation_error_code: str,
) -> None:
    """Require one initially satisfied fixed world-space attachment frame."""

    tolerance = _FRAME_COHERENCE_TOLERANCE
    position0, basis0 = world_frame0
    position1, basis1 = world_frame1
    separation = (
        position1[0] - position0[0],
        position1[1] - position0[1],
        position1[2] - position0[2],
    )
    separation_norm = math.sqrt(sum(component * component for component in separation))
    if separation_norm > tolerance:
        _fail(
            anchor_error_code,
            f"{label} fixed endpoint anchors differ by {separation_norm:.9g} meters",
        )
    if any(
        not _vectors_close(left, right, tolerance=tolerance)
        for left, right in zip(basis0, basis1, strict=True)
    ):
        _fail(
            orientation_error_code,
            f"{label} fixed endpoint orientation bases differ",
        )


def _mobile_frame_constraint_kind(
    constraint: JointConstraintV2,
) -> _FrameConstraintKind | None:
    """Return the narrow kind accepted by the mobile-frame coherence seam."""

    if isinstance(constraint, RevoluteConstraintV2):
        return "revolute"
    if isinstance(constraint, PrismaticConstraintV2):
        return "prismatic"
    if isinstance(constraint, SphericalConstraintV2):
        return "spherical"
    return None


def _source_axis_token(prim: Any) -> str:
    attribute = prim.GetAttribute("physics:axis")
    value = _require_explicit_static_attribute_default(
        attribute,
        field="physics:axis",
        error_code="articulation_v2_source_axis_invalid",
    )
    token = str(value)
    if token not in {"X", "Y", "Z"}:
        _fail(
            "articulation_v2_source_axis_unsupported",
            f"source physics:axis must be X, Y, or Z; got {token!r}",
        )
    return token


def _require_explicit_static_attribute_default(
    attribute: Any,
    *,
    field: str,
    error_code: str,
) -> Any:
    from pxr import Usd

    if not attribute:
        _fail(
            error_code,
            f"{field} requires an explicit authored default",
        )
    samples = tuple(float(sample) for sample in attribute.GetTimeSamples())
    connections = tuple(str(path) for path in attribute.GetConnections())
    resolve_info = attribute.GetResolveInfo(Usd.TimeCode.Default())
    blocked = bool(resolve_info.ValueIsBlocked())
    value = attribute.Get(Usd.TimeCode.Default())
    if (
        not attribute.HasAuthoredValueOpinion()
        or samples
        or connections
        or blocked
        or value is None
    ):
        _fail(
            error_code,
            f"{field} must be one explicit static connected-free default: "
            f"authored={attribute.HasAuthoredValueOpinion()}, "
            f"blocked={blocked}, samples={samples}, "
            f"connections={connections}",
        )
    return value


def _require_no_authored_attribute_state(
    attribute: Any,
    *,
    field: str,
    error_code: str,
) -> None:
    from pxr import Usd

    if not attribute:
        return
    samples = tuple(float(sample) for sample in attribute.GetTimeSamples())
    connections = tuple(str(path) for path in attribute.GetConnections())
    blocked = bool(attribute.GetResolveInfo(Usd.TimeCode.Default()).ValueIsBlocked())
    authored = bool(attribute.HasAuthoredValueOpinion())
    if authored or samples or connections or blocked:
        _fail(
            error_code,
            f"{field} must have no authored default, samples, blocks, or "
            f"connections: authored={authored}, blocked={blocked}, "
            f"samples={samples}, connections={connections}",
        )


def _require_source_static_scalar_limit(
    attribute: Any,
    *,
    field: str,
    joint_id: str,
) -> float:
    label = f"joint {joint_id!r} {field}"
    value = _require_explicit_static_attribute_default(
        attribute,
        field=label,
        error_code="articulation_v2_source_limit_state_invalid",
    )
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        _fail(
            "articulation_v2_source_limit_state_invalid",
            f"{label} is not a scalar: {exc}",
        )
    if not math.isfinite(result):
        _fail(
            "articulation_v2_source_limit_state_invalid",
            f"{label} must be finite; got {result!r}",
        )
    return result


def _require_source_limit_absent(
    attribute: Any,
    *,
    field: str,
    joint_id: str,
) -> None:
    _require_no_authored_attribute_state(
        attribute,
        field=f"joint {joint_id!r} unbounded {field}",
        error_code="articulation_v2_source_limit_state_invalid",
    )


def _free_spherical_control_state(
    prim: Any,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    forbidden_attributes = (
        "physics:axis",
        "physics:coneAngle0Limit",
        "physics:coneAngle1Limit",
        "physics:lowerLimit",
        "physics:upperLimit",
    )
    authored_property_names = {
        str(prop.GetName()) for prop in prim.GetAuthoredProperties()
    }
    authored: list[str] = []
    for name in forbidden_attributes:
        attribute = prim.GetAttribute(name)
        if name in authored_property_names or (
            attribute
            and (
                attribute.HasAuthoredValueOpinion()
                or attribute.GetTimeSamples()
                or attribute.GetConnections()
            )
        ):
            authored.append(name)
    applied_controls = tuple(
        schema
        for schema in prim.GetAppliedSchemas()
        if schema.startswith(("PhysicsDriveAPI:", "PhysicsLimitAPI:"))
    )
    return tuple(authored), applied_controls


def _require_free_spherical_source_controls_absent(prim: Any) -> None:
    authored, applied_controls = _free_spherical_control_state(prim)
    if authored or applied_controls:
        _fail(
            "articulation_v2_source_spherical_control_invalid",
            "free spherical source carries axis, scalar limits, or controls: "
            f"attributes={authored}, schemas={applied_controls}",
        )


def _canonical_frame_orientation(
    source_orientation: QuaternionWxyz,
    *,
    axis_token: str,
) -> QuaternionWxyz:
    from pxr import Gf

    source_rotation = Gf.Rotation(_quatd(source_orientation, Gf=Gf))
    bases = {
        "X": (
            Gf.Vec3d(1.0, 0.0, 0.0),
            Gf.Vec3d(0.0, 1.0, 0.0),
            Gf.Vec3d(0.0, 0.0, 1.0),
        ),
        "Y": (
            Gf.Vec3d(0.0, 1.0, 0.0),
            Gf.Vec3d(0.0, 0.0, 1.0),
            Gf.Vec3d(1.0, 0.0, 0.0),
        ),
        "Z": (
            Gf.Vec3d(0.0, 0.0, 1.0),
            Gf.Vec3d(1.0, 0.0, 0.0),
            Gf.Vec3d(0.0, 1.0, 0.0),
        ),
    }[axis_token]
    matrix = Gf.Matrix3d(1.0)
    for row, base in enumerate(bases):
        matrix.SetRow(row, source_rotation.TransformDir(base))
    quaternion = matrix.ExtractRotation().GetQuat()
    imaginary = quaternion.GetImaginary()
    return _validated_orientation(
        (
            float(quaternion.GetReal()),
            float(imaginary[0]),
            float(imaginary[1]),
            float(imaginary[2]),
        )
    )


def _single_relationship_target(relationship: Any, *, field: str) -> str:
    targets = tuple(str(path) for path in relationship.GetTargets())
    if len(targets) != 1:
        _fail(
            "articulation_v2_source_endpoint_incomplete",
            f"{field} must have exactly one target; got {targets}",
        )
    return targets[0]


def _require_static_authored_vector(attribute: Any, *, field: str) -> Vector3:
    _require_static_attribute(attribute, field=field)
    value = attribute.Get()
    if value is None:
        _fail(
            "articulation_v2_source_frame_incomplete",
            f"{field} has no authored default value",
        )
    components = tuple(float(component) for component in value)
    if len(components) != 3 or any(
        not math.isfinite(component) for component in components
    ):
        _fail(
            "articulation_v2_source_frame_invalid",
            f"{field} must be a finite three-vector",
        )
    return (components[0], components[1], components[2])


def _require_static_authored_quaternion(
    attribute: Any,
    *,
    field: str,
) -> QuaternionWxyz:
    components = _require_static_authored_quaternion_components(
        attribute,
        field=field,
    )
    try:
        result = _validated_orientation(components)
    except (TypeError, ValueError) as exc:
        _fail(
            "articulation_v2_source_frame_invalid",
            f"{field} is not a finite canonical unit quaternion: {exc}",
        )
    return result


def _require_static_authored_quaternion_components(
    attribute: Any,
    *,
    field: str,
) -> QuaternionWxyz:
    _require_static_attribute(attribute, field=field)
    value = attribute.Get()
    if value is None:
        _fail(
            "articulation_v2_source_frame_incomplete",
            f"{field} has no authored default value",
        )
    try:
        imaginary = value.GetImaginary()
        result = (
            float(value.GetReal()),
            float(imaginary[0]),
            float(imaginary[1]),
            float(imaginary[2]),
        )
    except (AttributeError, IndexError, TypeError, ValueError) as exc:
        _fail(
            "articulation_v2_source_frame_invalid",
            f"{field} is not a quaternion: {exc}",
        )
    if any(not math.isfinite(component) for component in result):
        _fail(
            "articulation_v2_source_frame_invalid",
            f"{field} must be a finite quaternion",
        )
    return result


def _validated_orientation(value: QuaternionWxyz) -> QuaternionWxyz:
    orientation = AttachmentFrameV2(
        position_meters=(0.0, 0.0, 0.0),
        orientation_wxyz=value,
    ).orientation_wxyz
    return (
        float(orientation[0]),
        float(orientation[1]),
        float(orientation[2]),
        float(orientation[3]),
    )


def _require_static_attribute(attribute: Any, *, field: str) -> None:
    from pxr import Usd

    if not attribute:
        _fail(
            "articulation_v2_source_frame_incomplete",
            f"{field} requires an explicit authored opinion",
        )
    samples = tuple(float(sample) for sample in attribute.GetTimeSamples())
    connections = tuple(str(path) for path in attribute.GetConnections())
    blocked = bool(attribute.GetResolveInfo(Usd.TimeCode.Default()).ValueIsBlocked())
    if connections or blocked:
        _fail(
            "articulation_v2_source_frame_state_invalid",
            f"{field} must be connected-free and unblocked: "
            f"blocked={blocked}, connections={connections}",
        )
    if not attribute.HasAuthoredValueOpinion():
        _fail(
            "articulation_v2_source_frame_incomplete",
            f"{field} requires an explicit authored opinion",
        )
    if samples:
        _fail(
            "articulation_v2_source_frame_time_varying",
            f"{field} has time samples: {samples}",
        )


def _capture_source_snapshot(stage: Any) -> _SourceSnapshot:
    from pxr import Usd, UsdGeom

    default_prim = stage.GetDefaultPrim()
    prims: list[tuple[Any, ...]] = []
    cache = UsdGeom.XformCache()
    stage_prims = list(
        Usd.PrimRange.Stage(
            stage,
            Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate),
        )
    )
    prototype_prims = [
        prim
        for prototype in stage.GetPrototypes()
        for prim in Usd.PrimRange.AllPrims(prototype)
    ]
    seen_paths: set[str] = set()
    for prim in (*stage_prims, *prototype_prims):
        prim_path = str(prim.GetPath())
        if prim_path in seen_paths:
            continue
        seen_paths.add(prim_path)
        properties: list[tuple[Any, ...]] = []
        for prop in prim.GetAuthoredProperties():
            property_metadata = tuple(
                sorted(
                    (
                        str(key),
                        _canonical_metadata_value(value),
                    )
                    for key, value in prop.GetAllAuthoredMetadata().items()
                )
            )
            attribute = prim.GetAttribute(prop.GetName())
            if attribute:
                sample_values = tuple(
                    (
                        _canonical_float(float(sample)),
                        _canonical_attribute_value(
                            attribute.Get(Usd.TimeCode(float(sample)))
                        ),
                    )
                    for sample in attribute.GetTimeSamples()
                )
                properties.append(
                    (
                        str(prop.GetName()),
                        "attribute",
                        str(attribute.GetTypeName()),
                        str(attribute.GetVariability()),
                        _canonical_attribute_value(
                            attribute.Get(Usd.TimeCode.Default())
                        ),
                        sample_values,
                        tuple(str(path) for path in attribute.GetConnections()),
                        property_metadata,
                    )
                )
                continue
            relationship = prim.GetRelationship(prop.GetName())
            if relationship:
                properties.append(
                    (
                        str(prop.GetName()),
                        "relationship",
                        tuple(str(path) for path in relationship.GetTargets()),
                        property_metadata,
                    )
                )
        xformable = UsdGeom.Xformable(prim)
        world_transform = (
            tuple(
                float(value)
                for row in cache.GetLocalToWorldTransform(prim)
                for value in row
            )
            if xformable
            else None
        )
        prims.append(
            (
                prim_path,
                str(prim.GetTypeName()),
                bool(prim.IsActive()),
                bool(prim.IsDefined()),
                bool(prim.IsInstance()),
                bool(prim.IsInstanceable()),
                tuple(
                    sorted(
                        (
                            str(key),
                            _canonical_metadata_value(value),
                        )
                        for key, value in prim.GetAllAuthoredMetadata().items()
                    )
                ),
                tuple(sorted(str(token) for token in prim.GetAppliedSchemas())),
                tuple(sorted(properties)),
                world_transform,
            )
        )
    return _SourceSnapshot(
        default_prim_path=str(default_prim.GetPath()) if default_prim else "",
        meters_per_unit=_meters_per_unit(stage),
        up_axis=str(UsdGeom.GetStageUpAxis(stage)),
        root_layer_metadata=_root_layer_metadata_snapshot(stage.GetRootLayer()),
        prims=tuple(sorted(prims)),
    )


def _canonical_float(value: float) -> tuple[str, str]:
    """Return an exact, stable float encoding, including infinities and -0."""

    return ("float", float(value).hex())


def _canonical_attribute_value(value: Any) -> Any:
    """Canonicalize a USD attribute value without wrapper repr or identity."""

    from pxr import Sdf

    if value is None:
        return ("none",)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, str):
        return ("string", value)
    if isinstance(value, int):
        return ("integer", int(value))
    if isinstance(value, float):
        return _canonical_float(value)
    if isinstance(value, Sdf.AssetPath):
        return ("asset_path", value.path)
    if isinstance(value, Sdf.Path):
        return ("path", str(value))
    if isinstance(value, Mapping):
        items: list[tuple[str, Any]] = []
        for key, item in value.items():
            if not isinstance(key, str):
                _fail(
                    "articulation_v2_source_attribute_unreadable",
                    "USD attribute dictionary keys must be strings; "
                    f"got {type(key).__name__}",
                )
            items.append((key, _canonical_attribute_value(item)))
        return ("mapping", tuple(sorted(items)))
    if hasattr(value, "GetReal") and hasattr(value, "GetImaginary"):
        imaginary = value.GetImaginary()
        return (
            "quaternion",
            _canonical_type_name(value),
            _canonical_float(float(value.GetReal())),
            tuple(_canonical_float(float(component)) for component in imaginary),
        )
    if hasattr(value, "GetValue") and callable(value.GetValue):
        return (
            "wrapped_value",
            _canonical_type_name(value),
            _canonical_attribute_value(value.GetValue()),
        )
    try:
        components = tuple(value)
    except TypeError:
        _fail(
            "articulation_v2_source_attribute_unreadable",
            f"unsupported USD attribute value type: {_canonical_type_name(value)}",
        )
    return (
        "sequence",
        _canonical_type_name(value),
        tuple(_canonical_attribute_value(component) for component in components),
    )


def _canonical_type_name(value: Any) -> str:
    value_type = type(value)
    module = getattr(value_type, "__module__", "")
    qualified_name = getattr(value_type, "__qualname__", value_type.__name__)
    return f"{module}.{qualified_name}" if module else qualified_name


def _root_layer_metadata_snapshot(layer: Any) -> tuple[tuple[str, Any], ...]:
    """Return stable root-layer metadata, including USD's unwrapped offsets."""

    rows: list[tuple[str, Any]] = []
    for key_value in layer.pseudoRoot.ListInfoKeys():
        key = str(key_value)
        if key == "subLayerOffsets":
            value = tuple(
                (float(offset.offset), float(offset.scale))
                for offset in layer.subLayerOffsets
            )
        else:
            try:
                value = layer.pseudoRoot.GetInfo(key)
            except (RuntimeError, TypeError, ValueError) as exc:
                _fail(
                    "articulation_v2_source_stage_metadata_unreadable",
                    f"could not snapshot root-layer metadata {key!r}: "
                    f"{type(exc).__name__}: {exc}",
                )
        rows.append((key, _canonical_metadata_value(value)))
    return tuple(sorted(rows))


def _remap_source_snapshot_for_aggregate_links(
    snapshot: _SourceSnapshot,
    rigid_link_plans: tuple[RigidLinkPlanV1, ...],
) -> _SourceSnapshot:
    """Project the exact pre-edit snapshot through the declared namespace edit."""

    mappings = tuple(
        sorted(
            (
                (member.source_prim_path, member.authored_prim_path)
                for link in rigid_link_plans
                if link.body_authoring == "aggregate"
                for member in link.members
            ),
            key=lambda item: len(item[0]),
            reverse=True,
        )
    )

    def remap_path(path: str) -> str:
        for source_path, authored_path in mappings:
            if path == source_path:
                return authored_path
            if path.startswith(f"{source_path}/"):
                return f"{authored_path}{path[len(source_path) :]}"
        return path

    remapped_rows: list[tuple[Any, ...]] = []
    for row in snapshot.prims:
        values = list(row)
        values[0] = remap_path(str(row[0]))
        remapped_rows.append(tuple(values))
    return _SourceSnapshot(
        default_prim_path=snapshot.default_prim_path,
        meters_per_unit=snapshot.meters_per_unit,
        up_axis=snapshot.up_axis,
        root_layer_metadata=snapshot.root_layer_metadata,
        prims=tuple(sorted(remapped_rows)),
    )


def _validate_source_snapshot(
    before: _SourceSnapshot,
    stage: Any,
    *,
    allowed_additions: set[str],
) -> None:
    after = _capture_source_snapshot(stage)
    if (
        after.default_prim_path != before.default_prim_path
        or not math.isclose(
            after.meters_per_unit,
            before.meters_per_unit,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or after.up_axis != before.up_axis
        or after.root_layer_metadata != before.root_layer_metadata
    ):
        _fail(
            "articulation_v2_source_stage_metadata_changed",
            "root-layer metadata changed during authoring",
        )
    before_rows = {str(row[0]): row for row in before.prims}
    after_rows = {str(row[0]): row for row in after.prims}
    missing = set(before_rows) - set(after_rows)
    changed = sorted(
        path
        for path in before_rows.keys() & after_rows.keys()
        if not _snapshot_rows_close(before_rows[path], after_rows[path])
    )
    extras = set(after_rows) - set(before_rows)
    if missing or changed or extras != allowed_additions:
        _fail(
            "articulation_v2_source_stage_changed",
            "source hierarchy, properties, schemas, or transforms changed outside "
            "the owned joint additions: "
            f"missing={sorted(missing)}, changed={changed}, "
            f"extra={sorted(extras)}, allowed={sorted(allowed_additions)}",
        )


def _snapshot_rows_close(left: tuple[Any, ...], right: tuple[Any, ...]) -> bool:
    if left[:-1] != right[:-1]:
        return False
    left_matrix = left[-1]
    right_matrix = right[-1]
    if left_matrix is None or right_matrix is None:
        return left_matrix is right_matrix
    return all(
        math.isclose(
            first,
            second,
            rel_tol=_MATRIX_TOLERANCE,
            abs_tol=_MATRIX_TOLERANCE,
        )
        for first, second in zip(left_matrix, right_matrix, strict=True)
    )


def _canonical_metadata_value(value: Any) -> Any:
    """Remove Python-wrapper identity from authored USD metadata snapshots."""

    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        return tuple(
            sorted(
                (
                    str(key),
                    _canonical_metadata_value(item),
                )
                for key, item in value.items()
            )
        )
    if isinstance(value, list | tuple):
        return tuple(_canonical_metadata_value(item) for item in value)
    if all(
        hasattr(value, field)
        for field in (
            "isExplicit",
            "explicitItems",
            "addedItems",
            "prependedItems",
            "appendedItems",
            "deletedItems",
            "orderedItems",
        )
    ):
        return (
            "list_op",
            bool(value.isExplicit),
            tuple(str(item) for item in value.explicitItems),
            tuple(str(item) for item in value.addedItems),
            tuple(str(item) for item in value.prependedItems),
            tuple(str(item) for item in value.appendedItems),
            tuple(str(item) for item in value.deletedItems),
            tuple(str(item) for item in value.orderedItems),
        )
    return str(value)


def _canonical_field_evidence(
    evidence: tuple[FieldEvidenceV2, ...],
) -> str:
    return json.dumps(
        [
            item.model_dump(mode="json", exclude_none=True)
            for item in sorted(evidence, key=lambda value: value.field)
        ],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _require_frame_close(
    observed: AttachmentFrameV2,
    expected: AttachmentFrameV2,
    *,
    item: _PreparedJoint,
    field: str,
) -> None:
    if not all(
        math.isclose(
            left,
            right,
            rel_tol=0.0,
            abs_tol=_READBACK_TOLERANCE,
        )
        for left, right in zip(
            observed.position_meters,
            expected.position_meters,
            strict=True,
        )
    ) or not all(
        math.isclose(
            left,
            right,
            rel_tol=0.0,
            abs_tol=_READBACK_TOLERANCE,
        )
        for left, right in zip(
            observed.orientation_wxyz,
            expected.orientation_wxyz,
            strict=True,
        )
    ):
        _readback_fail(
            item,
            f"{field} frame differs from the requested canonical frame",
        )


def _readback_vector(
    attribute: Any,
    *,
    field: str,
    item: _PreparedJoint,
) -> Vector3:
    try:
        return _require_static_authored_vector(attribute, field=field)
    except ArticulationV2FrameAuthoringError as exc:
        _readback_fail(item, f"{field} is invalid: {exc.code}: {exc.detail}")


def _readback_quaternion(
    attribute: Any,
    *,
    field: str,
    item: _PreparedJoint,
) -> QuaternionWxyz:
    try:
        components = _require_static_authored_quaternion_components(
            attribute,
            field=field,
        )
        _validated_orientation(components)
        return components
    except (TypeError, ValueError) as exc:
        _readback_fail(item, f"{field} is not a canonical unit quaternion: {exc}")
    except ArticulationV2FrameAuthoringError as exc:
        _readback_fail(item, f"{field} is invalid: {exc.code}: {exc.detail}")


def _require_scalar_close(
    observed: float | None,
    expected: float | None,
    *,
    item: _PreparedJoint,
    field: str,
) -> None:
    if (
        observed is None
        or expected is None
        or not math.isclose(
            float(observed),
            float(expected),
            rel_tol=0.0,
            abs_tol=_READBACK_TOLERANCE,
        )
    ):
        _readback_fail(
            item,
            f"{field} differs: observed={observed!r}, expected={expected!r}",
        )


def _readback_fail(item: _PreparedJoint, detail: str) -> NoReturn:
    _fail(
        "articulation_v2_saved_readback_mismatch",
        f"joint {item.record.joint_id!r} at {item.joint_path}: {detail}",
    )


def _set_custom_data_exact(
    prim: Any,
    *,
    key: str,
    value: Any,
    label: str,
) -> None:
    prim.SetCustomDataByKey(key, value)
    observed = prim.GetCustomDataByKey(key)
    if type(observed) is not type(value) or observed != value:
        _fail(
            "articulation_v2_joint_authoring_failed",
            f"could not author {label}: observed={observed!r}, expected={value!r}",
        )


def _require_set(result: Any, label: str) -> None:
    if result is not True:
        _fail(
            "articulation_v2_joint_authoring_failed",
            f"could not author {label}",
        )


def _quatf(value: QuaternionWxyz, *, Gf: Any) -> Any:
    return Gf.Quatf(
        float(value[0]),
        Gf.Vec3f(float(value[1]), float(value[2]), float(value[3])),
    )


def _quatd(value: QuaternionWxyz, *, Gf: Any) -> Any:
    return Gf.Quatd(
        float(value[0]),
        Gf.Vec3d(float(value[1]), float(value[2]), float(value[3])),
    )


def _normalized_direction(value: Any, *, label: str) -> Vector3:
    result = tuple(float(component) for component in value)
    norm = math.sqrt(sum(component * component for component in result))
    if (
        any(not math.isfinite(component) for component in result)
        or not math.isfinite(norm)
        or math.isclose(norm, 0.0, rel_tol=0.0, abs_tol=1e-12)
    ):
        _fail(
            "articulation_v2_unsupported_endpoint_frame",
            f"{label} is not a finite nonzero direction",
        )
    return (
        result[0] / norm,
        result[1] / norm,
        result[2] / norm,
    )


def _dot(left: Vector3, right: Vector3) -> float:
    return sum(first * second for first, second in zip(left, right, strict=True))


def _cross(left: Vector3, right: Vector3) -> Vector3:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _vectors_close(
    left: Vector3,
    right: Vector3,
    *,
    tolerance: float = _FRAME_TOLERANCE,
) -> bool:
    return all(
        math.isclose(
            first,
            second,
            rel_tol=0.0,
            abs_tol=tolerance,
        )
        for first, second in zip(left, right, strict=True)
    )


def _multiply_vector(value: Vector3, scale: float) -> Vector3:
    return (
        value[0] * scale,
        value[1] * scale,
        value[2] * scale,
    )


def _fail(code: str, detail: str) -> NoReturn:
    raise ArticulationV2FrameAuthoringError(code, detail)


__all__ = [
    "ARTICULATION_V2_FRAME_AUTHORING_CONTRACT_SCHEMA_VERSION",
    "ARTICULATION_V2_FRAME_AUTHORING_SCHEMA_VERSION",
    "ARTICULATION_V2_FRAME_AUTHORING_SCHEMA_VERSION_V2",
    "ArticulationV2FrameAuthoringContractV1",
    "ArticulationV2FrameAuthoringError",
    "ArticulationV2FrameAuthoringResult",
    "AuthoredAttachmentFrameReadbackV2",
    "AuthoredAttachmentFrameReadbackV3",
    "CapturedUsdArtifactBindingV1",
    "ConstraintPolicyDiagnosticV2",
    "OpaqueArtifactIdentityV1",
    "SourceBackedAttachmentFramesV2",
    "SourceBackedFixedDistanceConstraintV2",
    "SourceJointBindingV1",
    "author_articulation_v2_attachment_frames",
    "canonical_frame_authoring_contract_bytes",
    "promote_joint_record_with_source_backed_fixed_distance_v2",
    "promote_joint_record_with_source_backed_frames_v2",
    "promote_source_backed_attachment_frames_v2",
    "promote_source_backed_fixed_distance_constraint_v2",
]
