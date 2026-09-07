# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provider-free ingestion of already verified domain validation results.

The domain owns execution and projection of its native report.  This module
only authenticates the supplied projection and every bound byte, exposes the
facts to the outer reasoner, and seals its exact assessment.  It deliberately
does not import or invoke a renderer, simulator, model provider, or Validation
template executor.
"""

from __future__ import annotations

import hashlib
import os
import stat
from datetime import datetime
from pathlib import Path
from typing import Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from world_understanding.utils.artifacts import (
    ArtifactPathError,
    open_confined_directory,
    open_confined_regular_file,
)

from content_agent_workflows.common.artifacts import (
    _directory_chain_matches,
    _open_directory_no_symlinks,
    atomic_write_json,
)
from content_agent_workflows.common.domain_execution import ExecutionArtifactBinding
from content_agent_workflows.common.embedded_domain_decision import (
    canonical_json_digest,
)

from .embedded_assessment import (
    VALIDATION_TERMINAL_RECEIPT_NAME,
    ValidationCoordinatorReviewDraft,
)
from .operations import (
    EXECUTED_VALIDATION_ARTIFACT_NAMES,
    PROVIDED_OPERATION_ARTIFACT_NAMES,
)

VERIFIED_OPERATION_ENVELOPE_SCHEMA_VERSION: Final = (
    "content-agent-workflows.verified-validation-operation-envelope.v1"
)
VERIFIED_OPERATION_PROJECTION_SCHEMA_VERSION: Final = (
    "content-agent-workflows.verified-validation-operation-projection.v1"
)
VERIFIED_OPERATION_INGEST_RECEIPT_SCHEMA_VERSION: Final = (
    "content-agent-workflows.verified-validation-operation-ingest-receipt.v1"
)
VERIFIED_OPERATION_INGEST_INDEX_SCHEMA_VERSION: Final = (
    "content-agent-workflows.verified-validation-operation-ingest-index.v1"
)
VERIFIED_OPERATION_EVIDENCE_INDEX_SCHEMA_VERSION: Final = (
    "content-agent-workflows.verified-validation-operation-evidence-index.v1"
)
VERIFIED_OPERATION_ASSESSMENT_SCHEMA_VERSION: Final = (
    "content-agent-workflows.verified-validation-operation-assessment.v1"
)
VERIFIED_OPERATION_EXECUTION_INDEX_SCHEMA_VERSION: Final = (
    "content-agent-workflows.verified-validation-operation-execution-index.v1"
)
VERIFIED_OPERATION_TERMINAL_RECEIPT_SCHEMA_VERSION: Final = (
    "content-agent-workflows.verified-validation-operation-terminal-receipt.v1"
)

VERIFIED_OPERATION_INGEST_INDEX_NAME: Final = "verified_operation_ingest_index.json"
VERIFIED_OPERATION_EVIDENCE_INDEX_NAME: Final = "verified_operation_evidence_index.json"
VERIFIED_OPERATION_EXECUTION_INDEX_NAME: Final = (
    "verified_operation_execution_index.json"
)
VERIFIED_OPERATION_CANONICAL_ASSESSMENT_NAME: Final = (
    "canonical_verified_operation_assessment.json"
)
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_HIERARCHICAL_ID_PATTERN = r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)+$"
_CHUNK_SIZE = 1024 * 1024
_LOCAL_PROVIDED_OPERATION_ARTIFACT_NAMES: Final = (
    VERIFIED_OPERATION_INGEST_INDEX_NAME,
    VERIFIED_OPERATION_EVIDENCE_INDEX_NAME,
    VERIFIED_OPERATION_EXECUTION_INDEX_NAME,
    VERIFIED_OPERATION_CANONICAL_ASSESSMENT_NAME,
    "verified_operations",
)
if _LOCAL_PROVIDED_OPERATION_ARTIFACT_NAMES != PROVIDED_OPERATION_ARTIFACT_NAMES:
    raise RuntimeError("provided Validation artifact inventory differs by mode")
_FIXED_PIPELINE_GATE_NAMES: Final = (
    "static_validation",
    "runtime_validation",
    "visual_quality",
    "package_integrity",
    "cross_stage_integrity",
)

VerifiedNativeStatus = Literal[
    "pass",
    "warn",
    "fail",
    "not_requested",
    "not_evaluated",
    "blocked",
    "error",
]
VerifiedOperationAuthority = Literal[
    "deterministic_fact",
    "outer_review_input",
    "advisory_critique",
]
VerifiedOperationDisposition = Literal[
    "pass",
    "fail",
    "waive",
    "defer",
    "not_evaluated",
    "advisory",
]
VerifiedTerminalDisposition = Literal[
    "pass",
    "fail",
    "needs_remediation",
    "deferred",
    "blocked",
]


def _validate_native_result_shape(
    *,
    result_label: Literal["verified projection", "verified operation"],
    native_report: ExecutionArtifactBinding | None,
    native_report_type: str | None,
    native_payload: ExecutionArtifactBinding | None,
    native_payload_type: str | None,
    native_status: VerifiedNativeStatus,
    required: bool,
    authority: VerifiedOperationAuthority,
    dependencies: tuple[ExecutionArtifactBinding, ...],
    artifacts: tuple[ExecutionArtifactBinding, ...],
) -> None:
    if native_report is None and native_payload is None:
        raise ValueError(f"{result_label} requires a native report or payload")
    if (native_report is None) != (native_report_type is None):
        raise ValueError(
            "native_report and native_report_type must be supplied together"
        )
    if (native_payload is None) != (native_payload_type is None):
        raise ValueError(
            "native_payload and native_payload_type must be supplied together"
        )
    if native_status == "not_requested" and required:
        raise ValueError("a not_requested operation cannot be required")
    if authority == "advisory_critique" and required:
        raise ValueError("advisory critique cannot be required")
    _validate_binding_sequence(dependencies, label="dependency")
    _validate_binding_sequence(artifacts, label="artifact")


class VerifiedOperationError(RuntimeError):
    """Raised when provided evidence cannot be accepted without weakening identity."""


_RegularFileIdentity = tuple[int, int, int, int, int, int]


class VerifiedOperationBindingCache:
    """Hash each identical binding once, then reject post-verification drift."""

    def __init__(self) -> None:
        self._verified: dict[
            tuple[str, str, int],
            tuple[ExecutionArtifactBinding, str, _RegularFileIdentity],
        ] = {}

    def verify(self, binding: ExecutionArtifactBinding, *, label: str) -> None:
        identity = (binding.path, binding.sha256, binding.size_bytes)
        if identity in self._verified:
            return
        metadata = _verify_execution_artifact_binding_streaming(
            binding,
            label=label,
        )
        self._verified[identity] = (binding, label, _stat_identity(metadata))

    def assert_unchanged(self) -> None:
        """Reject any binding that changed after its cached digest verification."""

        for binding, label, expected in self._verified.values():
            candidate = Path(binding.path).expanduser()
            observed = _read_regular_file_identity(candidate)
            if observed != expected:
                raise VerifiedOperationError(
                    f"{label} identity changed after verification: {candidate}"
                )


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class VerifiedOperationComponentIdentity(_FrozenModel):
    """Versioned producer, tool, profile, backend, or projector identity."""

    component_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    contract: ExecutionArtifactBinding
    configuration: ExecutionArtifactBinding | None = None


class VerifiedValidationOperationProjection(_FrozenModel):
    """Domain-authored statement binding its native report to exact artifacts."""

    schema_version: Literal[
        "content-agent-workflows.verified-validation-operation-projection.v1"
    ] = VERIFIED_OPERATION_PROJECTION_SCHEMA_VERSION
    operation_id: str = Field(pattern=_HIERARCHICAL_ID_PATTERN)
    gate_id: str = Field(pattern=_HIERARCHICAL_ID_PATTERN)
    evidence_type: str = Field(pattern=_HIERARCHICAL_ID_PATTERN)
    native_report_type: str | None = Field(
        default=None,
        pattern=_HIERARCHICAL_ID_PATTERN,
    )
    native_payload_type: str | None = Field(
        default=None,
        pattern=_HIERARCHICAL_ID_PATTERN,
    )
    claim_scope: str = Field(min_length=1)
    native_status: VerifiedNativeStatus
    required: bool
    authority: VerifiedOperationAuthority
    source: ExecutionArtifactBinding
    output: ExecutionArtifactBinding
    dependencies: tuple[ExecutionArtifactBinding, ...] = ()
    artifacts: tuple[ExecutionArtifactBinding, ...] = ()
    native_report: ExecutionArtifactBinding | None = None
    native_payload: ExecutionArtifactBinding | None = None
    producer_identity_sha256: str = Field(pattern=_SHA256_PATTERN)
    tool_identity_sha256: str = Field(pattern=_SHA256_PATTERN)
    profile_identity_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    backend_identity_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    verifier_identity_sha256: str = Field(pattern=_SHA256_PATTERN)
    projector_identity_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_native_result(self) -> Self:
        _validate_native_result_shape(
            result_label="verified projection",
            native_report=self.native_report,
            native_report_type=self.native_report_type,
            native_payload=self.native_payload,
            native_payload_type=self.native_payload_type,
            native_status=self.native_status,
            required=self.required,
            authority=self.authority,
            dependencies=self.dependencies,
            artifacts=self.artifacts,
        )
        return self


class VerifiedValidationOperationEnvelope(_FrozenModel):
    """Frozen external result accepted by the provider-free ingestion leaf."""

    schema_version: Literal[
        "content-agent-workflows.verified-validation-operation-envelope.v1"
    ] = VERIFIED_OPERATION_ENVELOPE_SCHEMA_VERSION
    operation_id: str = Field(pattern=_HIERARCHICAL_ID_PATTERN)
    gate_id: str = Field(pattern=_HIERARCHICAL_ID_PATTERN)
    evidence_type: str = Field(pattern=_HIERARCHICAL_ID_PATTERN)
    native_report_type: str | None = Field(
        default=None,
        pattern=_HIERARCHICAL_ID_PATTERN,
    )
    native_payload_type: str | None = Field(
        default=None,
        pattern=_HIERARCHICAL_ID_PATTERN,
    )
    claim_scope: str = Field(min_length=1)
    native_status: VerifiedNativeStatus
    required: bool = True
    authority: VerifiedOperationAuthority = "deterministic_fact"
    source: ExecutionArtifactBinding
    output: ExecutionArtifactBinding
    dependencies: tuple[ExecutionArtifactBinding, ...] = ()
    artifacts: tuple[ExecutionArtifactBinding, ...] = ()
    native_report: ExecutionArtifactBinding | None = None
    native_payload: ExecutionArtifactBinding | None = None
    producer: VerifiedOperationComponentIdentity
    tool: VerifiedOperationComponentIdentity
    profile: VerifiedOperationComponentIdentity | None = None
    backend: VerifiedOperationComponentIdentity | None = None
    verifier: VerifiedOperationComponentIdentity
    projector: VerifiedOperationComponentIdentity
    projection: ExecutionArtifactBinding

    @model_validator(mode="after")
    def validate_contract_shape(self) -> Self:
        _validate_native_result_shape(
            result_label="verified operation",
            native_report=self.native_report,
            native_report_type=self.native_report_type,
            native_payload=self.native_payload,
            native_payload_type=self.native_payload_type,
            native_status=self.native_status,
            required=self.required,
            authority=self.authority,
            dependencies=self.dependencies,
            artifacts=self.artifacts,
        )
        return self


class VerifiedOperationIngestReceipt(_FrozenModel):
    """One verified envelope retained without executing or normalizing it."""

    schema_version: Literal[
        "content-agent-workflows.verified-validation-operation-ingest-receipt.v1"
    ] = VERIFIED_OPERATION_INGEST_RECEIPT_SCHEMA_VERSION
    execution_mode: Literal["provided"] = "provided"
    envelope_binding: ExecutionArtifactBinding
    envelope_contract_sha256: str = Field(pattern=_SHA256_PATTERN)
    envelope: VerifiedValidationOperationEnvelope
    nested_agent_launched: Literal[False] = False

    @model_validator(mode="after")
    def validate_envelope_digest(self) -> Self:
        if self.envelope_contract_sha256 != canonical_json_digest(self.envelope):
            raise ValueError("envelope_contract_sha256 differs from envelope")
        return self


class VerifiedOperationIndexEntry(_FrozenModel):
    operation_id: str = Field(pattern=_HIERARCHICAL_ID_PATTERN)
    gate_id: str = Field(pattern=_HIERARCHICAL_ID_PATTERN)
    evidence_type: str = Field(pattern=_HIERARCHICAL_ID_PATTERN)
    native_report_type: str | None = Field(
        default=None,
        pattern=_HIERARCHICAL_ID_PATTERN,
    )
    native_payload_type: str | None = Field(
        default=None,
        pattern=_HIERARCHICAL_ID_PATTERN,
    )
    claim_scope: str = Field(min_length=1)
    native_status: VerifiedNativeStatus
    required: bool
    authority: VerifiedOperationAuthority
    ingest_receipt: ExecutionArtifactBinding


class VerifiedOperationIngestIndex(_FrozenModel):
    """Frozen list of independently ingested domain operations."""

    schema_version: Literal[
        "content-agent-workflows.verified-validation-operation-ingest-index.v1"
    ] = VERIFIED_OPERATION_INGEST_INDEX_SCHEMA_VERSION
    execution_mode: Literal["provided"] = "provided"
    operations: tuple[VerifiedOperationIndexEntry, ...] = Field(min_length=1)
    nested_agent_launched: Literal[False] = False

    @model_validator(mode="after")
    def validate_unique_operations(self) -> Self:
        operation_ids = [item.operation_id for item in self.operations]
        gate_ids = [item.gate_id for item in self.operations]
        if len(operation_ids) != len(set(operation_ids)):
            raise ValueError("verified operation IDs must be unique")
        if len(gate_ids) != len(set(gate_ids)):
            raise ValueError("verified gate IDs must be unique")
        return self


class VerifiedOperationEvidenceRecord(_FrozenModel):
    operation_id: str = Field(pattern=_HIERARCHICAL_ID_PATTERN)
    gate_id: str = Field(pattern=_HIERARCHICAL_ID_PATTERN)
    evidence_type: str = Field(pattern=_HIERARCHICAL_ID_PATTERN)
    native_report_type: str | None = Field(
        default=None,
        pattern=_HIERARCHICAL_ID_PATTERN,
    )
    native_payload_type: str | None = Field(
        default=None,
        pattern=_HIERARCHICAL_ID_PATTERN,
    )
    claim_scope: str = Field(min_length=1)
    native_status: VerifiedNativeStatus
    required: bool
    authority: VerifiedOperationAuthority
    ingest_receipt: ExecutionArtifactBinding
    source: ExecutionArtifactBinding
    output: ExecutionArtifactBinding
    dependencies: tuple[ExecutionArtifactBinding, ...] = ()
    artifacts: tuple[ExecutionArtifactBinding, ...] = ()
    native_report: ExecutionArtifactBinding | None = None
    native_payload: ExecutionArtifactBinding | None = None
    producer: VerifiedOperationComponentIdentity
    tool: VerifiedOperationComponentIdentity
    profile: VerifiedOperationComponentIdentity | None = None
    backend: VerifiedOperationComponentIdentity | None = None
    verifier: VerifiedOperationComponentIdentity
    projector: VerifiedOperationComponentIdentity
    projection: ExecutionArtifactBinding


class VerifiedOperationEvidenceIndex(_FrozenModel):
    """Exact facts presented to the one outer semantic reasoner."""

    schema_version: Literal[
        "content-agent-workflows.verified-validation-operation-evidence-index.v1"
    ] = VERIFIED_OPERATION_EVIDENCE_INDEX_SCHEMA_VERSION
    execution_mode: Literal["provided"] = "provided"
    assessment_identity_sha256: str = Field(pattern=_SHA256_PATTERN)
    ingest_index: ExecutionArtifactBinding
    records: tuple[VerifiedOperationEvidenceRecord, ...] = Field(min_length=1)
    nested_agent_launched: Literal[False] = False


class VerifiedOperationAssessmentDisposition(_FrozenModel):
    operation_id: str = Field(pattern=_HIERARCHICAL_ID_PATTERN)
    gate_id: str = Field(pattern=_HIERARCHICAL_ID_PATTERN)
    ingest_receipt_sha256: str = Field(pattern=_SHA256_PATTERN)
    native_status: VerifiedNativeStatus
    disposition: VerifiedOperationDisposition
    rationale: str = Field(min_length=1)


class VerifiedOperationCoordinatorAssessment(_FrozenModel):
    """Exact-digest independent operation assessment authored by the outer loop."""

    schema_version: Literal[
        "content-agent-workflows.verified-validation-operation-assessment.v1"
    ] = VERIFIED_OPERATION_ASSESSMENT_SCHEMA_VERSION
    assessment_id: str = Field(min_length=1)
    created_at: datetime
    assessment_identity_sha256: str = Field(pattern=_SHA256_PATTERN)
    operations: tuple[VerifiedOperationAssessmentDisposition, ...] = Field(min_length=1)
    terminal_disposition: VerifiedTerminalDisposition
    summary: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_shape(self) -> Self:
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("assessment created_at must include a timezone")
        operation_ids = [item.operation_id for item in self.operations]
        gate_ids = [item.gate_id for item in self.operations]
        if len(operation_ids) != len(set(operation_ids)):
            raise ValueError("assessment operation IDs must be unique")
        if len(gate_ids) != len(set(gate_ids)):
            raise ValueError("assessment gate IDs must be unique")
        return self


class VerifiedOperationExecutionIndex(_FrozenModel):
    schema_version: Literal[
        "content-agent-workflows.verified-validation-operation-execution-index.v1"
    ] = VERIFIED_OPERATION_EXECUTION_INDEX_SCHEMA_VERSION
    execution_mode: Literal["provided"] = "provided"
    assessment_identity_sha256: str = Field(pattern=_SHA256_PATTERN)
    assessment_id: str = Field(min_length=1)
    assessment_sha256: str = Field(pattern=_SHA256_PATTERN)
    evidence_index: ExecutionArtifactBinding
    canonical_assessment: ExecutionArtifactBinding | None = None
    authorized: bool
    nested_agent_launched: Literal[False] = False


class VerifiedOperationTerminalReceipt(_FrozenModel):
    """Terminal exact-readback receipt retaining every domain gate."""

    schema_version: Literal[
        "content-agent-workflows.verified-validation-operation-terminal-receipt.v1"
    ] = VERIFIED_OPERATION_TERMINAL_RECEIPT_SCHEMA_VERSION
    mode: Literal["provided"] = "provided"
    assessment_identity_sha256: str = Field(pattern=_SHA256_PATTERN)
    evidence_index: ExecutionArtifactBinding
    canonical_assessment: ExecutionArtifactBinding
    coordinator_review: ExecutionArtifactBinding
    operation_evidence: tuple[VerifiedOperationEvidenceRecord, ...] = Field(
        min_length=1
    )
    operation_dispositions: tuple[VerifiedOperationAssessmentDisposition, ...] = Field(
        min_length=1
    )
    gate_dispositions: dict[
        Literal[
            "static_validation",
            "runtime_validation",
            "visual_quality",
            "package_integrity",
            "cross_stage_integrity",
        ],
        Literal["not_evaluated"],
    ]
    terminal_disposition: VerifiedTerminalDisposition
    review_disposition: Literal[
        "accept", "reject", "revise", "retry", "stop", "cancelled"
    ]
    receipt_status: Literal["completed"] = "completed"
    nested_agent_launched: Literal[False] = False

    @model_validator(mode="after")
    def validate_fixed_pipeline_gate_compatibility(self) -> Self:
        if set(self.gate_dispositions) != set(_FIXED_PIPELINE_GATE_NAMES):
            raise ValueError(
                "provided-mode receipt must classify every fixed-pipeline gate"
            )
        if set(self.gate_dispositions.values()) != {"not_evaluated"}:
            raise ValueError(
                "provided results cannot assert fixed-pipeline five-gate passes"
            )
        evidence_identity = tuple(
            (
                item.operation_id,
                item.gate_id,
                item.ingest_receipt.sha256,
                item.native_status,
            )
            for item in self.operation_evidence
        )
        disposition_identity = tuple(
            (
                item.operation_id,
                item.gate_id,
                item.ingest_receipt_sha256,
                item.native_status,
            )
            for item in self.operation_dispositions
        )
        if evidence_identity != disposition_identity:
            raise ValueError("terminal operation evidence and dispositions differ")
        return self


def _validate_binding_sequence(
    bindings: tuple[ExecutionArtifactBinding, ...], *, label: str
) -> None:
    identities = [(item.path, item.sha256, item.size_bytes) for item in bindings]
    if len(identities) != len(set(identities)):
        raise ValueError(f"verified {label} bindings must be unique")


def _verified_run_root(
    output_dir: str | Path,
    *,
    create_missing: bool,
) -> Path:
    """Resolve no symlinks while pinning or creating a provided-mode run root."""

    root = Path(output_dir).expanduser()
    if not root.is_absolute():
        root = (Path.cwd() / root).absolute()
    if os.name == "nt":
        try:
            with open_confined_directory(root, create=create_missing):
                pass
        except (ArtifactPathError, OSError, RuntimeError, ValueError) as exc:
            raise VerifiedOperationError(
                f"verified operation run is missing or unsafe: {root}"
            ) from exc
        return root
    try:
        directory_fd, _chain = _open_directory_no_symlinks(
            root,
            create_missing=create_missing,
        )
    except OSError as exc:
        raise VerifiedOperationError(
            f"verified operation run is missing or unsafe: {root}"
        ) from exc
    os.close(directory_fd)
    return root


def _read_regular_file(
    path: Path,
    *,
    capture_bytes: bool,
) -> tuple[bytes | None, str, os.stat_result]:
    if not path.is_absolute():
        raise VerifiedOperationError(f"verified artifact path must be absolute: {path}")
    if os.name == "nt":
        try:
            with open_confined_directory(path.parent) as parent:
                with open_confined_regular_file(parent, path.name) as (
                    stream,
                    opened,
                ):
                    if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
                        raise VerifiedOperationError(
                            "verified artifact must be one regular non-linked "
                            f"file: {path}"
                        )
                    digest = hashlib.sha256()
                    chunks: list[bytes] | None = [] if capture_bytes else None
                    while chunk := os.read(stream.fileno(), _CHUNK_SIZE):
                        digest.update(chunk)
                        if chunks is not None:
                            chunks.append(chunk)
                    finished = os.fstat(stream.fileno())
                    with open_confined_regular_file(parent, path.name) as (
                        named_stream,
                        named_after,
                    ):
                        named_after = os.fstat(named_stream.fileno())
                    if _stat_identity(finished) != _stat_identity(
                        opened
                    ) or _stat_identity(named_after) != _stat_identity(opened):
                        raise VerifiedOperationError(
                            f"verified artifact identity changed while reading: {path}"
                        )
                    return (
                        b"".join(chunks) if chunks is not None else None,
                        digest.hexdigest(),
                        finished,
                    )
        except VerifiedOperationError:
            raise
        except (ArtifactPathError, OSError, RuntimeError, ValueError) as exc:
            raise VerifiedOperationError(
                f"verified artifact is missing or unsafe: {path}: {exc}"
            ) from exc
    try:
        parent_fd, parent_chain = _open_directory_no_symlinks(
            path.parent,
            create_missing=False,
        )
    except OSError as exc:
        raise VerifiedOperationError(
            f"verified artifact parent is missing or unsafe: {path}"
        ) from exc
    fd = -1
    try:
        flags = (
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        fd = os.open(path.name, flags, dir_fd=parent_fd)
        opened = os.fstat(fd)
        named = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        identity = (opened.st_dev, opened.st_ino)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or not stat.S_ISREG(named.st_mode)
            or named.st_nlink != 1
            or (named.st_dev, named.st_ino) != identity
        ):
            raise VerifiedOperationError(
                f"verified artifact must be one regular non-linked file: {path}"
            )
        digest = hashlib.sha256()
        chunks: list[bytes] | None = [] if capture_bytes else None
        while chunk := os.read(fd, _CHUNK_SIZE):
            digest.update(chunk)
            if chunks is not None:
                chunks.append(chunk)
        finished = os.fstat(fd)
        named_after = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            (
                finished.st_dev,
                finished.st_ino,
                finished.st_size,
                finished.st_mtime_ns,
                finished.st_ctime_ns,
                finished.st_nlink,
            )
            != (
                opened.st_dev,
                opened.st_ino,
                opened.st_size,
                opened.st_mtime_ns,
                opened.st_ctime_ns,
                opened.st_nlink,
            )
            or (
                named_after.st_dev,
                named_after.st_ino,
                named_after.st_size,
                named_after.st_mtime_ns,
                named_after.st_ctime_ns,
                named_after.st_nlink,
            )
            != (
                opened.st_dev,
                opened.st_ino,
                opened.st_size,
                opened.st_mtime_ns,
                opened.st_ctime_ns,
                opened.st_nlink,
            )
            or not _directory_chain_matches(path.parent, parent_chain)
        ):
            raise VerifiedOperationError(
                f"verified artifact identity changed while reading: {path}"
            )
        return (
            b"".join(chunks) if chunks is not None else None,
            digest.hexdigest(),
            finished,
        )
    except OSError as exc:
        raise VerifiedOperationError(
            f"verified artifact is missing or unsafe: {path}: {exc}"
        ) from exc
    finally:
        if fd >= 0:
            os.close(fd)
        os.close(parent_fd)


def _stat_identity(metadata: os.stat_result) -> _RegularFileIdentity:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
        metadata.st_nlink,
    )


def _read_regular_file_identity(path: Path) -> _RegularFileIdentity:
    """Recheck one path identity without hashing its already-verified bytes."""

    if not path.is_absolute():
        raise VerifiedOperationError(f"verified artifact path must be absolute: {path}")
    if os.name == "nt":
        try:
            with open_confined_directory(path.parent) as parent:
                with open_confined_regular_file(parent, path.name) as (
                    stream,
                    opened,
                ):
                    opened = os.fstat(stream.fileno())
                    with open_confined_regular_file(parent, path.name) as (
                        named_stream,
                        named,
                    ):
                        named = os.fstat(named_stream.fileno())
                    if (
                        not stat.S_ISREG(opened.st_mode)
                        or opened.st_nlink != 1
                        or _stat_identity(named) != _stat_identity(opened)
                    ):
                        raise VerifiedOperationError(
                            "verified artifact identity changed after verification: "
                            f"{path}"
                        )
                    return _stat_identity(opened)
        except VerifiedOperationError:
            raise
        except (ArtifactPathError, OSError, RuntimeError, ValueError) as exc:
            raise VerifiedOperationError(
                f"verified artifact is missing or unsafe: {path}: {exc}"
            ) from exc
    try:
        parent_fd, parent_chain = _open_directory_no_symlinks(
            path.parent,
            create_missing=False,
        )
    except OSError as exc:
        raise VerifiedOperationError(
            f"verified artifact parent is missing or unsafe: {path}"
        ) from exc
    fd = -1
    try:
        flags = (
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        fd = os.open(path.name, flags, dir_fd=parent_fd)
        opened = os.fstat(fd)
        named = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or not stat.S_ISREG(named.st_mode)
            or named.st_nlink != 1
            or _stat_identity(named) != _stat_identity(opened)
            or not _directory_chain_matches(path.parent, parent_chain)
        ):
            raise VerifiedOperationError(
                f"verified artifact identity changed after verification: {path}"
            )
        return _stat_identity(opened)
    except OSError as exc:
        raise VerifiedOperationError(
            f"verified artifact is missing or unsafe: {path}: {exc}"
        ) from exc
    finally:
        if fd >= 0:
            os.close(fd)
        os.close(parent_fd)


def execution_artifact_binding(path: str | Path) -> ExecutionArtifactBinding:
    """Bind one regular file through a no-symlink, single-read boundary."""

    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).absolute()
    _payload, digest, metadata = _read_regular_file(
        candidate,
        capture_bytes=False,
    )
    return ExecutionArtifactBinding(
        path=str(candidate),
        sha256=digest,
        size_bytes=metadata.st_size,
    )


def _verify_execution_artifact_binding_streaming(
    binding: ExecutionArtifactBinding,
    *,
    label: str,
) -> os.stat_result:
    """Reverify a binding without retaining its bytes in memory."""

    candidate = Path(binding.path).expanduser()
    _payload, observed, metadata = _read_regular_file(
        candidate,
        capture_bytes=False,
    )
    if metadata.st_size != binding.size_bytes or observed != binding.sha256:
        raise VerifiedOperationError(f"{label} binding is stale: {candidate}")
    return metadata


def verify_execution_artifact_binding(
    binding: ExecutionArtifactBinding, *, label: str = "verified artifact"
) -> bytes:
    """Re-read one binding and reject any byte, path, or identity drift."""

    candidate = Path(binding.path).expanduser()
    payload, observed, metadata = _read_regular_file(
        candidate,
        capture_bytes=True,
    )
    if metadata.st_size != binding.size_bytes or observed != binding.sha256:
        raise VerifiedOperationError(f"{label} binding is stale: {candidate}")
    assert payload is not None
    return payload


def _load_bound_model[ModelT: BaseModel](
    binding: ExecutionArtifactBinding,
    model: type[ModelT],
    *,
    label: str,
) -> ModelT:
    payload = verify_execution_artifact_binding(binding, label=label)
    try:
        return model.model_validate_json(payload)
    except (ValueError, ValidationError) as exc:
        raise VerifiedOperationError(f"invalid {label}: {exc}") from exc


def _projection_from_envelope(
    envelope: VerifiedValidationOperationEnvelope,
) -> VerifiedValidationOperationProjection:
    return VerifiedValidationOperationProjection(
        operation_id=envelope.operation_id,
        gate_id=envelope.gate_id,
        evidence_type=envelope.evidence_type,
        native_report_type=envelope.native_report_type,
        native_payload_type=envelope.native_payload_type,
        claim_scope=envelope.claim_scope,
        native_status=envelope.native_status,
        required=envelope.required,
        authority=envelope.authority,
        source=envelope.source,
        output=envelope.output,
        dependencies=envelope.dependencies,
        artifacts=envelope.artifacts,
        native_report=envelope.native_report,
        native_payload=envelope.native_payload,
        producer_identity_sha256=canonical_json_digest(envelope.producer),
        tool_identity_sha256=canonical_json_digest(envelope.tool),
        profile_identity_sha256=(
            canonical_json_digest(envelope.profile)
            if envelope.profile is not None
            else None
        ),
        backend_identity_sha256=(
            canonical_json_digest(envelope.backend)
            if envelope.backend is not None
            else None
        ),
        verifier_identity_sha256=canonical_json_digest(envelope.verifier),
        projector_identity_sha256=canonical_json_digest(envelope.projector),
    )


def build_verified_operation_projection(
    **values: object,
) -> VerifiedValidationOperationProjection:
    """Build the exact projection a domain must persist before enveloping it."""

    return VerifiedValidationOperationProjection.model_validate(values)


def _verify_component(
    component: VerifiedOperationComponentIdentity,
    *,
    label: str,
    binding_cache: VerifiedOperationBindingCache | None,
) -> None:
    _verify_binding(
        component.contract,
        label=f"{label} contract",
        binding_cache=binding_cache,
    )
    if component.configuration is not None:
        _verify_binding(
            component.configuration,
            label=f"{label} configuration",
            binding_cache=binding_cache,
        )


def _verify_binding(
    binding: ExecutionArtifactBinding,
    *,
    label: str,
    binding_cache: VerifiedOperationBindingCache | None,
) -> None:
    if binding_cache is None:
        _verify_execution_artifact_binding_streaming(binding, label=label)
    else:
        binding_cache.verify(binding, label=label)


def verify_operation_envelope(
    envelope: VerifiedValidationOperationEnvelope,
    *,
    binding_cache: VerifiedOperationBindingCache | None = None,
) -> VerifiedValidationOperationEnvelope:
    """Verify every supplied artifact and the domain's exact projection manifest."""

    _verify_binding(envelope.source, label="source", binding_cache=binding_cache)
    _verify_binding(envelope.output, label="output", binding_cache=binding_cache)
    for binding in envelope.dependencies:
        _verify_binding(binding, label="dependency", binding_cache=binding_cache)
    for binding in envelope.artifacts:
        _verify_binding(
            binding,
            label="evidence artifact",
            binding_cache=binding_cache,
        )
    if envelope.native_report is not None:
        _verify_binding(
            envelope.native_report,
            label="native report",
            binding_cache=binding_cache,
        )
    if envelope.native_payload is not None:
        _verify_binding(
            envelope.native_payload,
            label="native payload",
            binding_cache=binding_cache,
        )
    for label, component in (
        ("producer", envelope.producer),
        ("tool", envelope.tool),
        ("profile", envelope.profile),
        ("backend", envelope.backend),
        ("verifier", envelope.verifier),
        ("projector", envelope.projector),
    ):
        if component is not None:
            _verify_component(
                component,
                label=label,
                binding_cache=binding_cache,
            )
    projection = _load_bound_model(
        envelope.projection,
        VerifiedValidationOperationProjection,
        label="verified operation projection",
    )
    expected = _projection_from_envelope(envelope)
    if projection != expected:
        raise VerifiedOperationError(
            "verified operation projection differs from source, output, dependency, "
            "result, status, producer, tool, profile, backend, or projector identity"
        )
    return envelope


def load_verified_operation_envelope(
    path: str | Path,
) -> tuple[ExecutionArtifactBinding, VerifiedValidationOperationEnvelope]:
    """Load and fully verify one external envelope from its original bytes."""

    binding = execution_artifact_binding(path)
    envelope = _load_bound_model(
        binding,
        VerifiedValidationOperationEnvelope,
        label="verified operation envelope",
    )
    return binding, verify_operation_envelope(envelope)


def _assert_ingest_mode_only(root: Path) -> None:
    conflicting = [
        name
        for name in EXECUTED_VALIDATION_ARTIFACT_NAMES
        if (root / name).exists() or (root / name).is_symlink()
    ]
    if conflicting:
        raise VerifiedOperationError(
            "Validation execute and ingest modes are mutually exclusive; found "
            + ", ".join(conflicting)
        )


def _write_once(path: Path, value: BaseModel) -> None:
    if path.is_symlink():
        raise VerifiedOperationError(
            f"verified operation artifact is a symlink: {path}"
        )
    if path.exists():
        try:
            existing = value.__class__.model_validate_json(
                verify_execution_artifact_binding(
                    execution_artifact_binding(path),
                    label="existing verified operation artifact",
                )
            )
        except (ValueError, ValidationError, VerifiedOperationError) as exc:
            raise VerifiedOperationError(
                f"invalid existing verified operation artifact {path}: {exc}"
            ) from exc
        if existing != value:
            raise VerifiedOperationError(
                f"verified operation artifact already differs: {path}"
            )
        return
    atomic_write_json(path, value)


def _load_ingest_receipt(
    binding: ExecutionArtifactBinding,
) -> VerifiedOperationIngestReceipt:
    receipt = _load_bound_model(
        binding,
        VerifiedOperationIngestReceipt,
        label="verified operation ingest receipt",
    )
    bound_envelope = _load_bound_model(
        receipt.envelope_binding,
        VerifiedValidationOperationEnvelope,
        label="verified operation envelope",
    )
    if bound_envelope != receipt.envelope:
        raise VerifiedOperationError("ingested envelope bytes differ from receipt")
    verify_operation_envelope(receipt.envelope)
    return receipt


def _receipt_path(
    root: Path,
    envelope: VerifiedValidationOperationEnvelope,
) -> Path:
    return (
        root
        / "verified_operations"
        / canonical_json_digest(envelope)
        / "ingest_receipt.json"
    )


def load_verified_operation_ingest_index(
    output_dir: str | Path,
) -> tuple[VerifiedOperationIngestIndex, tuple[VerifiedOperationIngestReceipt, ...]]:
    """Revalidate an ingest index and every transitive artifact from disk."""

    root = _verified_run_root(output_dir, create_missing=False)
    _assert_ingest_mode_only(root)
    index_path = root / VERIFIED_OPERATION_INGEST_INDEX_NAME
    index = _load_bound_model(
        execution_artifact_binding(index_path),
        VerifiedOperationIngestIndex,
        label="verified operation ingest index",
    )
    receipts = tuple(
        _load_ingest_receipt(item.ingest_receipt) for item in index.operations
    )
    for entry, receipt in zip(index.operations, receipts, strict=True):
        envelope = receipt.envelope
        observed = (
            envelope.operation_id,
            envelope.gate_id,
            envelope.evidence_type,
            envelope.native_report_type,
            envelope.native_payload_type,
            envelope.claim_scope,
            envelope.native_status,
            envelope.required,
            envelope.authority,
        )
        expected = (
            entry.operation_id,
            entry.gate_id,
            entry.evidence_type,
            entry.native_report_type,
            entry.native_payload_type,
            entry.claim_scope,
            entry.native_status,
            entry.required,
            entry.authority,
        )
        if observed != expected:
            raise VerifiedOperationError(
                f"verified operation index entry differs for {entry.operation_id}"
            )
    return index, receipts


def ingest_verified_operation_result(
    envelope_path: str | Path,
    *,
    output_dir: str | Path,
) -> VerifiedOperationIngestReceipt:
    """Ingest exactly one verified result without executing any capability."""

    root = _verified_run_root(output_dir, create_missing=True)
    _assert_ingest_mode_only(root)
    for frozen_name in (
        VERIFIED_OPERATION_EVIDENCE_INDEX_NAME,
        VERIFIED_OPERATION_EXECUTION_INDEX_NAME,
        VALIDATION_TERMINAL_RECEIPT_NAME,
    ):
        if (root / frozen_name).exists() or (root / frozen_name).is_symlink():
            raise VerifiedOperationError(
                "verified operation set is already frozen for assessment"
            )
    envelope_binding, envelope = load_verified_operation_envelope(envelope_path)
    index_path = root / VERIFIED_OPERATION_INGEST_INDEX_NAME
    existing_entries: tuple[VerifiedOperationIndexEntry, ...] = ()
    if index_path.exists() or index_path.is_symlink():
        index, _ = load_verified_operation_ingest_index(root)
        existing_entries = index.operations
    if envelope.operation_id in {item.operation_id for item in existing_entries}:
        raise VerifiedOperationError(
            f"verified operation ID is already ingested: {envelope.operation_id}"
        )
    if envelope.gate_id in {item.gate_id for item in existing_entries}:
        raise VerifiedOperationError(
            f"verified gate ID is already ingested: {envelope.gate_id}"
        )
    receipt = VerifiedOperationIngestReceipt(
        envelope_binding=envelope_binding,
        envelope_contract_sha256=canonical_json_digest(envelope),
        envelope=envelope,
    )
    receipt_path = _receipt_path(root, envelope)
    receipt_dir = receipt_path.parent
    if receipt_dir.exists() or receipt_dir.is_symlink():
        raise VerifiedOperationError(
            f"verified operation receipt directory already exists: {receipt_dir}"
        )
    _verified_run_root(receipt_dir, create_missing=True)
    atomic_write_json(receipt_path, receipt)
    receipt_binding = execution_artifact_binding(receipt_path)
    entry = VerifiedOperationIndexEntry(
        operation_id=envelope.operation_id,
        gate_id=envelope.gate_id,
        evidence_type=envelope.evidence_type,
        native_report_type=envelope.native_report_type,
        native_payload_type=envelope.native_payload_type,
        claim_scope=envelope.claim_scope,
        native_status=envelope.native_status,
        required=envelope.required,
        authority=envelope.authority,
        ingest_receipt=receipt_binding,
    )
    try:
        updated = VerifiedOperationIngestIndex(operations=(*existing_entries, entry))
    except ValidationError as exc:
        raise VerifiedOperationError(
            f"verified operation index rejected: {exc}"
        ) from exc
    atomic_write_json(index_path, updated)
    _validated_index, validated_receipts = load_verified_operation_ingest_index(root)
    if validated_receipts[-1] != receipt:
        raise VerifiedOperationError(
            "verified operation receipt changed after ingestion"
        )
    return validated_receipts[-1]


def _build_evidence_index(root: Path) -> VerifiedOperationEvidenceIndex:
    _index, receipts = load_verified_operation_ingest_index(root)
    index_binding = execution_artifact_binding(
        root / VERIFIED_OPERATION_INGEST_INDEX_NAME
    )
    records = tuple(
        VerifiedOperationEvidenceRecord(
            operation_id=receipt.envelope.operation_id,
            gate_id=receipt.envelope.gate_id,
            evidence_type=receipt.envelope.evidence_type,
            native_report_type=receipt.envelope.native_report_type,
            native_payload_type=receipt.envelope.native_payload_type,
            claim_scope=receipt.envelope.claim_scope,
            native_status=receipt.envelope.native_status,
            required=receipt.envelope.required,
            authority=receipt.envelope.authority,
            ingest_receipt=execution_artifact_binding(
                _receipt_path(root, receipt.envelope)
            ),
            source=receipt.envelope.source,
            output=receipt.envelope.output,
            dependencies=receipt.envelope.dependencies,
            artifacts=receipt.envelope.artifacts,
            native_report=receipt.envelope.native_report,
            native_payload=receipt.envelope.native_payload,
            producer=receipt.envelope.producer,
            tool=receipt.envelope.tool,
            profile=receipt.envelope.profile,
            backend=receipt.envelope.backend,
            verifier=receipt.envelope.verifier,
            projector=receipt.envelope.projector,
            projection=receipt.envelope.projection,
        )
        for receipt in receipts
    )
    identity = canonical_json_digest(
        {
            "schema_version": "content-agent-workflows.verified-validation-assessment-identity.v1",
            "execution_mode": "provided",
            "ingest_index": index_binding.model_dump(mode="json"),
        }
    )
    return VerifiedOperationEvidenceIndex(
        assessment_identity_sha256=identity,
        ingest_index=index_binding,
        records=records,
    )


def collect_verified_operation_evidence(
    output_dir: str | Path,
) -> VerifiedOperationEvidenceIndex:
    """Freeze exact provided facts for direct outer assessment."""

    root = _verified_run_root(output_dir, create_missing=False)
    index = _build_evidence_index(root)
    _write_once(root / VERIFIED_OPERATION_EVIDENCE_INDEX_NAME, index)
    return index


def _load_evidence_index(root: Path) -> VerifiedOperationEvidenceIndex:
    path = root / VERIFIED_OPERATION_EVIDENCE_INDEX_NAME
    index = _load_bound_model(
        execution_artifact_binding(path),
        VerifiedOperationEvidenceIndex,
        label="verified operation evidence index",
    )
    refreshed = _build_evidence_index(root)
    if refreshed != index:
        raise VerifiedOperationError("verified operation evidence identity is stale")
    return index


def _load_model_from_path[ModelT: BaseModel](
    path: str | Path,
    model: type[ModelT],
    *,
    label: str,
) -> ModelT:
    return _load_bound_model(execution_artifact_binding(path), model, label=label)


def validate_verified_operation_assessment(
    assessment: VerifiedOperationCoordinatorAssessment,
    *,
    evidence: VerifiedOperationEvidenceIndex,
) -> VerifiedOperationCoordinatorAssessment:
    """Require exact coverage and fail closed on every required native status."""

    if assessment.assessment_identity_sha256 != evidence.assessment_identity_sha256:
        raise VerifiedOperationError(
            "outer assessment belongs to another evidence index"
        )
    actual = tuple(
        (
            item.operation_id,
            item.gate_id,
            item.ingest_receipt_sha256,
            item.native_status,
        )
        for item in assessment.operations
    )
    expected = tuple(
        (
            record.operation_id,
            record.gate_id,
            record.ingest_receipt.sha256,
            record.native_status,
        )
        for record in evidence.records
    )
    if actual != expected:
        raise VerifiedOperationError(
            "outer assessment must cover every imported operation exactly once in order"
        )
    for disposition, record in zip(
        assessment.operations,
        evidence.records,
        strict=True,
    ):
        if record.authority == "advisory_critique" and disposition.disposition not in {
            "advisory",
            "not_evaluated",
        }:
            raise VerifiedOperationError(
                f"advisory operation {record.operation_id} cannot become a pass"
            )
        if record.native_status in {"not_requested", "not_evaluated"} and (
            disposition.disposition != "not_evaluated"
        ):
            raise VerifiedOperationError(
                f"unevaluated operation {record.operation_id} cannot be relabeled"
            )
        if record.native_status in {"fail", "blocked", "error"} and (
            disposition.disposition == "pass"
        ):
            raise VerifiedOperationError(
                f"blocking operation {record.operation_id} cannot pass"
            )
    if assessment.terminal_disposition == "pass":
        for disposition, record in zip(
            assessment.operations,
            evidence.records,
            strict=True,
        ):
            if not record.required:
                continue
            expected_disposition = "waive" if record.native_status == "warn" else "pass"
            if record.native_status not in {"pass", "warn"} or (
                disposition.disposition != expected_disposition
            ):
                raise VerifiedOperationError(
                    f"required operation {record.operation_id} is unresolved: "
                    f"native={record.native_status}, disposition={disposition.disposition}"
                )
    return assessment


def assess_verified_operation_evidence(
    output_dir: str | Path,
    *,
    assessment_path: str | Path,
) -> VerifiedOperationExecutionIndex:
    """Validate and publish an exact-digest outer assessment in ingest mode."""

    root = _verified_run_root(output_dir, create_missing=False)
    evidence = _load_evidence_index(root)
    assessment = _load_model_from_path(
        assessment_path,
        VerifiedOperationCoordinatorAssessment,
        label="outer verified operation assessment",
    )
    validate_verified_operation_assessment(assessment, evidence=evidence)
    authorized = assessment.terminal_disposition == "pass"
    canonical_binding = None
    if authorized:
        canonical_path = root / VERIFIED_OPERATION_CANONICAL_ASSESSMENT_NAME
        _write_once(canonical_path, assessment)
        canonical_binding = execution_artifact_binding(canonical_path)
    execution = VerifiedOperationExecutionIndex(
        assessment_identity_sha256=evidence.assessment_identity_sha256,
        assessment_id=assessment.assessment_id,
        assessment_sha256=canonical_json_digest(assessment),
        evidence_index=execution_artifact_binding(
            root / VERIFIED_OPERATION_EVIDENCE_INDEX_NAME
        ),
        canonical_assessment=canonical_binding,
        authorized=authorized,
    )
    _write_once(root / VERIFIED_OPERATION_EXECUTION_INDEX_NAME, execution)
    return execution


def review_verified_operation_assessment(
    output_dir: str | Path,
    *,
    review_path: str | Path,
) -> VerifiedOperationTerminalReceipt:
    """Seal independent imported-operation dispositions after exact readback."""

    root = _verified_run_root(output_dir, create_missing=False)
    evidence = _load_evidence_index(root)
    execution = _load_model_from_path(
        root / VERIFIED_OPERATION_EXECUTION_INDEX_NAME,
        VerifiedOperationExecutionIndex,
        label="verified operation execution index",
    )
    if not execution.authorized or execution.canonical_assessment is None:
        raise VerifiedOperationError(
            "rejected or revision-required provided assessment has no output to review"
        )
    if (
        execution.assessment_identity_sha256 != evidence.assessment_identity_sha256
        or execution.evidence_index
        != execution_artifact_binding(root / VERIFIED_OPERATION_EVIDENCE_INDEX_NAME)
    ):
        raise VerifiedOperationError("verified operation execution identity is stale")
    assessment = _load_bound_model(
        execution.canonical_assessment,
        VerifiedOperationCoordinatorAssessment,
        label="canonical verified operation assessment",
    )
    if canonical_json_digest(assessment) != execution.assessment_sha256:
        raise VerifiedOperationError("canonical verified assessment digest is stale")
    validate_verified_operation_assessment(assessment, evidence=evidence)
    review_binding = execution_artifact_binding(review_path)
    review = _load_bound_model(
        review_binding,
        ValidationCoordinatorReviewDraft,
        label="outer verified assessment review",
    )
    if review.disposition == "accept" and assessment.terminal_disposition != "pass":
        raise VerifiedOperationError(
            "outer review cannot accept a non-passing assessment"
        )
    receipt = VerifiedOperationTerminalReceipt(
        assessment_identity_sha256=evidence.assessment_identity_sha256,
        evidence_index=execution.evidence_index,
        canonical_assessment=execution.canonical_assessment,
        coordinator_review=review_binding,
        operation_evidence=evidence.records,
        operation_dispositions=assessment.operations,
        gate_dispositions={
            name: "not_evaluated" for name in _FIXED_PIPELINE_GATE_NAMES
        },
        terminal_disposition=assessment.terminal_disposition,
        review_disposition=review.disposition,
        receipt_status="completed",
    )
    _write_once(root / VALIDATION_TERMINAL_RECEIPT_NAME, receipt)
    return load_verified_operation_terminal_receipt(root)


def load_verified_operation_terminal_receipt(
    output_dir: str | Path,
) -> VerifiedOperationTerminalReceipt:
    """Revalidate a terminal provided-mode receipt and all transitive bytes."""

    root = _verified_run_root(output_dir, create_missing=False)
    evidence = _load_evidence_index(root)
    receipt = _load_model_from_path(
        root / VALIDATION_TERMINAL_RECEIPT_NAME,
        VerifiedOperationTerminalReceipt,
        label="verified operation terminal receipt",
    )
    current_evidence_binding = execution_artifact_binding(
        root / VERIFIED_OPERATION_EVIDENCE_INDEX_NAME
    )
    if (
        receipt.assessment_identity_sha256 != evidence.assessment_identity_sha256
        or receipt.evidence_index != current_evidence_binding
        or receipt.operation_evidence != evidence.records
    ):
        raise VerifiedOperationError("terminal provided-operation evidence is stale")
    assessment = _load_bound_model(
        receipt.canonical_assessment,
        VerifiedOperationCoordinatorAssessment,
        label="terminal canonical verified assessment",
    )
    validate_verified_operation_assessment(assessment, evidence=evidence)
    review = _load_bound_model(
        receipt.coordinator_review,
        ValidationCoordinatorReviewDraft,
        label="terminal outer verified assessment review",
    )
    if (
        receipt.operation_dispositions != assessment.operations
        or receipt.terminal_disposition != assessment.terminal_disposition
        or receipt.review_disposition != review.disposition
        or receipt.receipt_status != "completed"
    ):
        raise VerifiedOperationError(
            "terminal provided-operation assessment or review is stale"
        )
    return receipt


def is_verified_operation_ingest_run(output_dir: str | Path) -> bool:
    """Return whether a run declares the explicit provider-free ingest mode."""

    root = _verified_run_root(output_dir, create_missing=False)
    return any(
        (root / name).exists() or (root / name).is_symlink()
        for name in PROVIDED_OPERATION_ARTIFACT_NAMES
    )


__all__ = [
    "VALIDATION_TERMINAL_RECEIPT_NAME",
    "VERIFIED_OPERATION_ASSESSMENT_SCHEMA_VERSION",
    "VERIFIED_OPERATION_CANONICAL_ASSESSMENT_NAME",
    "VERIFIED_OPERATION_ENVELOPE_SCHEMA_VERSION",
    "VERIFIED_OPERATION_EVIDENCE_INDEX_NAME",
    "VERIFIED_OPERATION_EVIDENCE_INDEX_SCHEMA_VERSION",
    "VERIFIED_OPERATION_EXECUTION_INDEX_NAME",
    "VERIFIED_OPERATION_EXECUTION_INDEX_SCHEMA_VERSION",
    "VERIFIED_OPERATION_INGEST_INDEX_NAME",
    "VERIFIED_OPERATION_INGEST_INDEX_SCHEMA_VERSION",
    "VERIFIED_OPERATION_INGEST_RECEIPT_SCHEMA_VERSION",
    "VERIFIED_OPERATION_PROJECTION_SCHEMA_VERSION",
    "VERIFIED_OPERATION_TERMINAL_RECEIPT_SCHEMA_VERSION",
    "VerifiedOperationAssessmentDisposition",
    "VerifiedOperationComponentIdentity",
    "VerifiedOperationCoordinatorAssessment",
    "VerifiedOperationError",
    "VerifiedOperationEvidenceIndex",
    "VerifiedOperationEvidenceRecord",
    "VerifiedOperationExecutionIndex",
    "VerifiedOperationIndexEntry",
    "VerifiedOperationIngestIndex",
    "VerifiedOperationIngestReceipt",
    "VerifiedOperationTerminalReceipt",
    "VerifiedValidationOperationEnvelope",
    "VerifiedValidationOperationProjection",
    "assess_verified_operation_evidence",
    "build_verified_operation_projection",
    "collect_verified_operation_evidence",
    "execution_artifact_binding",
    "ingest_verified_operation_result",
    "is_verified_operation_ingest_run",
    "load_verified_operation_envelope",
    "load_verified_operation_ingest_index",
    "load_verified_operation_terminal_receipt",
    "review_verified_operation_assessment",
    "validate_verified_operation_assessment",
    "verify_execution_artifact_binding",
    "verify_operation_envelope",
]
