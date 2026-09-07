# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Durable append-only storage for embedded-domain decision artifacts."""

from __future__ import annotations

import json
import os
import secrets
import stat
import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Annotated, Final, Literal, Self, TypeVar

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator
from world_understanding.utils.file_locking import (
    blocking_exclusive_descriptor_lock,
)

from .artifacts import _open_directory_no_symlinks
from .embedded_domain_decision import (
    BoundedExecutionAuthorization,
    ContractArtifact,
    ContractArtifactReference,
    EmbeddedBoundedExecutionResult,
    EmbeddedCoordinatorDecision,
    EmbeddedCoordinatorReview,
    EmbeddedDecisionContractError,
    EmbeddedDecisionIdentity,
    EmbeddedDecisionReceipt,
    EmbeddedDomainEvidence,
    EmbeddedDomainProposal,
    EmbeddedHumanDecision,
    PersistedExecutionLineage,
    Sha256Digest,
    artifact_reference,
    authorize_bounded_execution,
    canonical_json_digest,
    outer_stage_attempt_seal_key,
    validate_bounded_execution_result,
    validate_coordinator_decision_dependencies,
    validate_coordinator_review,
    validate_decision_receipt,
    validate_human_decision,
)

EMBEDDED_DECISION_ARTIFACT_JOURNAL_SCHEMA_VERSION: Final = (
    "content-agent-workflows.embedded-decision-artifact-journal.v1"
)
EMBEDDED_DECISION_MUTATION_LEASE_SCHEMA_VERSION: Final = (
    "content-agent-workflows.embedded-mutation-lease.v1"
)
EMBEDDED_DECISION_MUTATION_RECONCILIATION_SCHEMA_VERSION: Final = (
    "content-agent-workflows.embedded-mutation-reconciliation.v1"
)
_STORE_DIRECTORY = "embedded-domain-decision-artifacts"
_JOURNAL_NAME = "journal.json"
_LOCK_NAME = ".journal.lock"
_MUTATION_LEASE_LOCK_PREFIX = ".embedded-decision-mutation-lease-"
_MUTATION_LEASE_LOCK_SUFFIX = ".lock"
_MUTATION_LEASE_RECORD_SUFFIX = ".json"
_PERSISTED_ARTIFACT_KINDS = frozenset(
    {
        "evidence",
        "proposal",
        "coordinator_decision",
        "human_decision",
        "execution_authorization",
        "execution_result",
        "coordinator_review",
        "decision_receipt",
    }
)

PersistableEmbeddedDecisionArtifact = Annotated[
    EmbeddedDomainEvidence
    | EmbeddedDomainProposal
    | EmbeddedCoordinatorDecision
    | EmbeddedHumanDecision
    | BoundedExecutionAuthorization
    | EmbeddedBoundedExecutionResult
    | EmbeddedCoordinatorReview
    | EmbeddedDecisionReceipt,
    Field(discriminator="artifact_kind"),
]
_ARTIFACT_ADAPTER: TypeAdapter[PersistableEmbeddedDecisionArtifact] = TypeAdapter(
    PersistableEmbeddedDecisionArtifact
)
_ArtifactT = TypeVar("_ArtifactT", bound=ContractArtifact)
_ExecutionT = TypeVar("_ExecutionT")


class EmbeddedDecisionArtifactStoreError(EmbeddedDecisionContractError):
    """Raised when the durable artifact journal is unsafe or inconsistent."""


class EmbeddedDecisionAuthorizationReplayError(EmbeddedDecisionArtifactStoreError):
    """Raised when committed authority requires result reconciliation."""

    def __init__(
        self,
        outcome: EmbeddedDecisionAuthorizationReconciliationRequired,
    ) -> None:
        self.outcome = outcome
        super().__init__(
            "Execution authorization is already committed; reconcile operation "
            f"{outcome.operation_id} before executor invocation"
        )


class EmbeddedDecisionMutationLeaseReconciliationError(
    EmbeddedDecisionArtifactStoreError
):
    """Raised when a durable outer mutation lease forbids executor invocation."""

    def __init__(
        self,
        outcome: EmbeddedDecisionMutationLeaseReconciliationRequired,
    ) -> None:
        self.outcome = outcome
        EmbeddedDecisionArtifactStoreError.__init__(
            self,
            "Durable outer mutation lease requires "
            f"{outcome.disposition} for operation {outcome.operation_id}; "
            "executor invocation is forbidden",
        )


def _require_safe_artifact_id(artifact_id: str) -> None:
    if (
        artifact_id in {".", ".."}
        or "/" in artifact_id
        or "\\" in artifact_id
        or "\x00" in artifact_id
    ):
        raise ValueError("artifact_id cannot contain path traversal syntax")


def _entry_digest(
    *,
    sequence: int,
    reference: ContractArtifactReference,
    parent_artifact: ContractArtifactReference,
    relative_path: str,
    committed_at: datetime,
    previous_entry_sha256: str | None,
) -> str:
    return canonical_json_digest(
        {
            "schema_version": (
                "content-agent-workflows.embedded-decision-journal-entry.v1"
            ),
            "sequence": sequence,
            "reference": reference.model_dump(mode="json"),
            "parent_artifact": parent_artifact.model_dump(mode="json"),
            "relative_path": relative_path,
            "committed_at": committed_at.isoformat(),
            "previous_entry_sha256": previous_entry_sha256,
        }
    )


class EmbeddedDecisionArtifactJournalEntry(BaseModel):
    """One hash-chained durable append in the exact commit order."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sequence: int = Field(ge=1)
    reference: ContractArtifactReference
    parent_artifact: ContractArtifactReference
    relative_path: str = Field(min_length=1)
    committed_at: datetime
    previous_entry_sha256: Sha256Digest | None = None
    entry_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_entry(self) -> Self:
        _require_safe_artifact_id(self.reference.artifact_id)
        if self.committed_at.tzinfo is None or self.committed_at.utcoffset() is None:
            raise ValueError("journal commit timestamp must be timezone-aware")
        relative = PurePosixPath(self.relative_path)
        expected_path = PurePosixPath(
            "artifacts",
            self.reference.artifact_kind,
            f"{self.reference.sha256}.json",
        )
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative != expected_path
        ):
            raise ValueError("journal artifact path is not canonical or root-confined")
        expected_digest = _entry_digest(
            sequence=self.sequence,
            reference=self.reference,
            parent_artifact=self.parent_artifact,
            relative_path=self.relative_path,
            committed_at=self.committed_at,
            previous_entry_sha256=self.previous_entry_sha256,
        )
        if self.entry_sha256 != expected_digest:
            raise ValueError("journal entry digest is stale")
        return self


class EmbeddedDecisionArtifactJournal(BaseModel):
    """Versioned ordered index for a single embedded decision run root."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[
        "content-agent-workflows.embedded-decision-artifact-journal.v1"
    ] = EMBEDDED_DECISION_ARTIFACT_JOURNAL_SCHEMA_VERSION
    entries: tuple[EmbeddedDecisionArtifactJournalEntry, ...] = ()

    @model_validator(mode="after")
    def validate_ordered_lineage(self) -> Self:
        expected_previous: str | None = None
        previous_commit: datetime | None = None
        known_references: set[ContractArtifactReference] = set()
        known_ids: set[tuple[str, str]] = set()
        for expected_sequence, entry in enumerate(self.entries, start=1):
            if entry.sequence != expected_sequence:
                raise ValueError(
                    "journal entries must have contiguous sequence numbers"
                )
            if entry.previous_entry_sha256 != expected_previous:
                raise ValueError("journal entry chain does not bind its predecessor")
            identity = (
                entry.reference.artifact_kind,
                entry.reference.artifact_id,
            )
            if identity in known_ids:
                raise ValueError("journal contains a duplicate artifact identity")
            if previous_commit is not None and entry.committed_at < previous_commit:
                raise ValueError("journal commit timestamps must be nondecreasing")
            if entry.parent_artifact.artifact_kind in _PERSISTED_ARTIFACT_KINDS:
                if entry.parent_artifact not in known_references:
                    raise ValueError(
                        "journal artifact parent is absent or ordered after its child"
                    )
            known_ids.add(identity)
            known_references.add(entry.reference)
            expected_previous = entry.entry_sha256
            previous_commit = entry.committed_at
        return self


class DurableEmbeddedDecisionArtifactCommit(BaseModel):
    """Proof returned only after artifact bytes and journal entry are durable."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    store_schema_version: Literal[
        "content-agent-workflows.embedded-decision-artifact-journal.v1"
    ] = EMBEDDED_DECISION_ARTIFACT_JOURNAL_SCHEMA_VERSION
    reference: ContractArtifactReference
    sequence: int = Field(ge=1)
    entry_sha256: Sha256Digest
    relative_path: str = Field(min_length=1)
    newly_committed: bool


class EmbeddedDecisionAuthorizationReconciliationRequired(BaseModel):
    """Typed fail-closed outcome for exact authorization replay."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[
        "content-agent-workflows.embedded-authorization-reconciliation.v1"
    ] = "content-agent-workflows.embedded-authorization-reconciliation.v1"
    authorization: ContractArtifactReference
    operation_id: Sha256Digest
    mutation_id: Sha256Digest | None = None
    attempt: int = Field(ge=1)
    durable_commit: DurableEmbeddedDecisionArtifactCommit

    @model_validator(mode="after")
    def validate_replay(self) -> Self:
        if self.authorization.artifact_kind != "execution_authorization":
            raise ValueError("reconciliation must name an execution authorization")
        if self.durable_commit.reference != self.authorization:
            raise ValueError("reconciliation commit differs from authorization")
        if self.durable_commit.newly_committed:
            raise ValueError("reconciliation requires an exact replayed commit")
        return self


def _mutation_lease_digest(
    *,
    outer_stage_attempt_seal_key: str,
    identity: EmbeddedDecisionIdentity,
    accepted_decision: ContractArtifactReference,
    authorization: ContractArtifactReference,
    accepted_decision_digest: str,
    operation_id: str,
    mutation_id: str,
    attempt: int,
    anchor_run_root_device: int,
    anchor_run_root_inode: int,
    journal_store_device: int,
    journal_store_inode: int,
    created_at: datetime,
) -> str:
    return canonical_json_digest(
        {
            "schema_version": EMBEDDED_DECISION_MUTATION_LEASE_SCHEMA_VERSION,
            "outer_stage_attempt_seal_key": outer_stage_attempt_seal_key,
            "identity": identity.model_dump(mode="json"),
            "accepted_decision": accepted_decision.model_dump(mode="json"),
            "authorization": authorization.model_dump(mode="json"),
            "accepted_decision_digest": accepted_decision_digest,
            "operation_id": operation_id,
            "mutation_id": mutation_id,
            "attempt": attempt,
            "anchor_run_root_device": anchor_run_root_device,
            "anchor_run_root_inode": anchor_run_root_inode,
            "journal_store_device": journal_store_device,
            "journal_store_inode": journal_store_inode,
            "created_at": created_at.isoformat(),
        }
    )


class DurableEmbeddedMutationLease(BaseModel):
    """One-shot outer mutation claim anchored outside the journal store."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["content-agent-workflows.embedded-mutation-lease.v1"] = (
        EMBEDDED_DECISION_MUTATION_LEASE_SCHEMA_VERSION
    )
    outer_stage_attempt_seal_key: Sha256Digest
    identity: EmbeddedDecisionIdentity
    accepted_decision: ContractArtifactReference
    authorization: ContractArtifactReference
    accepted_decision_digest: Sha256Digest
    operation_id: Sha256Digest
    mutation_id: Sha256Digest
    attempt: int = Field(ge=1)
    anchor_run_root_device: int = Field(ge=0)
    anchor_run_root_inode: int = Field(ge=0)
    journal_store_device: int = Field(ge=0)
    journal_store_inode: int = Field(ge=0)
    created_at: datetime
    lease_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_lease(self) -> Self:
        if self.outer_stage_attempt_seal_key != outer_stage_attempt_seal_key(
            self.identity
        ):
            raise ValueError("mutation lease stage-attempt seal is stale")
        if self.accepted_decision.artifact_kind != "coordinator_decision":
            raise ValueError("mutation lease must bind a coordinator decision")
        if self.authorization.artifact_kind != "execution_authorization":
            raise ValueError("mutation lease must bind an execution authorization")
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("mutation lease timestamp must be timezone-aware")
        expected = _mutation_lease_digest(
            outer_stage_attempt_seal_key=self.outer_stage_attempt_seal_key,
            identity=self.identity,
            accepted_decision=self.accepted_decision,
            authorization=self.authorization,
            accepted_decision_digest=self.accepted_decision_digest,
            operation_id=self.operation_id,
            mutation_id=self.mutation_id,
            attempt=self.attempt,
            anchor_run_root_device=self.anchor_run_root_device,
            anchor_run_root_inode=self.anchor_run_root_inode,
            journal_store_device=self.journal_store_device,
            journal_store_inode=self.journal_store_inode,
            created_at=self.created_at,
        )
        if self.lease_sha256 != expected:
            raise ValueError("mutation lease digest is stale")
        return self


class EmbeddedDecisionMutationLeaseReconciliationRequired(BaseModel):
    """Typed outcome when a durable outer lease already owns the mutation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[
        "content-agent-workflows.embedded-mutation-reconciliation.v1"
    ] = EMBEDDED_DECISION_MUTATION_RECONCILIATION_SCHEMA_VERSION
    disposition: Literal["reconciliation_required", "stale_store"]
    authorization: ContractArtifactReference
    operation_id: Sha256Digest
    mutation_id: Sha256Digest
    attempt: int = Field(ge=1)
    outer_stage_attempt_seal_key: Sha256Digest
    lease_sha256: Sha256Digest
    leased_store_device: int = Field(ge=0)
    leased_store_inode: int = Field(ge=0)
    active_store_device: int = Field(ge=0)
    active_store_inode: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_reconciliation(self) -> Self:
        if self.authorization.artifact_kind != "execution_authorization":
            raise ValueError("mutation reconciliation must name an authorization")
        same_store = (
            self.leased_store_device,
            self.leased_store_inode,
        ) == (
            self.active_store_device,
            self.active_store_inode,
        )
        if same_store != (self.disposition == "reconciliation_required"):
            raise ValueError("mutation reconciliation disposition is inconsistent")
        return self


def _canonical_artifact_text(artifact: ContractArtifact) -> str:
    return json.dumps(
        artifact.model_dump(mode="json"),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _canonical_mutation_lease_bytes(lease: DurableEmbeddedMutationLease) -> bytes:
    return (
        json.dumps(
            lease.model_dump(mode="json"),
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
        )
        + "\n"
    ).encode("utf-8")


def _require_safe_relative_name(name: str, *, label: str) -> None:
    if name in {"", ".", ".."} or "/" in name or "\\" in name or "\x00" in name:
        raise EmbeddedDecisionArtifactStoreError(
            f"{label} must be one root-confined path component"
        )


def _directory_identity(descriptor: int, *, label: str) -> tuple[int, int]:
    metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.geteuid():
        raise EmbeddedDecisionArtifactStoreError(
            f"{label} must be a same-user directory inode"
        )
    return metadata.st_dev, metadata.st_ino


def _validate_named_directory_descriptor(
    parent_fd: int,
    name: str,
    descriptor: int,
    *,
    label: str,
) -> tuple[int, int]:
    descriptor_identity = _directory_identity(descriptor, label=label)
    named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    named_identity = (named.st_dev, named.st_ino)
    if (
        not stat.S_ISDIR(named.st_mode)
        or named.st_uid != os.geteuid()
        or named_identity != descriptor_identity
    ):
        raise EmbeddedDecisionArtifactStoreError(
            f"{label} inode was renamed or substituted"
        )
    return descriptor_identity


def _open_child_directory_at(
    parent_fd: int,
    name: str,
    *,
    label: str,
    create_missing: bool,
) -> int:
    _require_safe_relative_name(name, label=label)
    flags = (
        os.O_RDONLY
        | os.O_DIRECTORY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        descriptor = os.open(name, flags, dir_fd=parent_fd)
    except FileNotFoundError:
        if not create_missing:
            raise
        os.mkdir(name, 0o700, dir_fd=parent_fd)
        os.fsync(parent_fd)
        descriptor = os.open(name, flags, dir_fd=parent_fd)
    try:
        _validate_named_directory_descriptor(
            parent_fd,
            name,
            descriptor,
            label=label,
        )
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _read_regular_bytes_at(directory_fd: int, name: str, *, label: str) -> bytes:
    _require_safe_relative_name(name, label=label)
    descriptor = -1
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
            dir_fd=directory_fd,
        )
        before = os.fstat(descriptor)
        named_before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        identity = (before.st_dev, before.st_ino)
        if (
            not stat.S_ISREG(before.st_mode)
            or not stat.S_ISREG(named_before.st_mode)
            or before.st_nlink != 1
            or named_before.st_nlink != 1
            or identity != (named_before.st_dev, named_before.st_ino)
        ):
            raise EmbeddedDecisionArtifactStoreError(
                f"{label} must be a pinned single-link regular file"
            )
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
        named_after = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if (
            identity != (after.st_dev, after.st_ino)
            or identity != (named_after.st_dev, named_after.st_ino)
            or after.st_nlink != 1
            or named_after.st_nlink != 1
            or len(payload) != after.st_size
        ):
            raise EmbeddedDecisionArtifactStoreError(
                f"{label} changed while being read"
            )
        return payload
    except EmbeddedDecisionArtifactStoreError:
        raise
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise EmbeddedDecisionArtifactStoreError(
            f"Could not read descriptor-confined {label}: {exc}"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _regular_file_identity_at(
    directory_fd: int,
    name: str,
    *,
    label: str,
    expected_identity: tuple[int, int] | None = None,
) -> tuple[int, int]:
    _require_safe_relative_name(name, label=label)
    descriptor = -1
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
            dir_fd=directory_fd,
        )
        descriptor_stat = os.fstat(descriptor)
        named_stat = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        descriptor_identity = (descriptor_stat.st_dev, descriptor_stat.st_ino)
        named_identity = (named_stat.st_dev, named_stat.st_ino)
        if (
            not stat.S_ISREG(descriptor_stat.st_mode)
            or not stat.S_ISREG(named_stat.st_mode)
            or descriptor_stat.st_nlink != 1
            or named_stat.st_nlink != 1
            or descriptor_stat.st_uid != os.geteuid()
            or named_stat.st_uid != os.geteuid()
        ):
            raise EmbeddedDecisionArtifactStoreError(
                f"{label} must be a same-user single-link regular inode"
            )
        if descriptor_identity != named_identity:
            raise EmbeddedDecisionArtifactStoreError(
                f"{label} inode changed during validation"
            )
        if expected_identity is not None and descriptor_identity != expected_identity:
            raise EmbeddedDecisionArtifactStoreError(
                f"{label} inode was renamed or substituted"
            )
        return descriptor_identity
    except EmbeddedDecisionArtifactStoreError:
        raise
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise EmbeddedDecisionArtifactStoreError(
            f"Could not inspect descriptor-confined {label}: {exc}"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:  # pragma: no cover - defensive OS contract guard
            raise OSError("short write while persisting embedded decision artifact")
        offset += written


def _temporary_name(name: str) -> str:
    return f".{name}.tmp-{os.getpid()}-{secrets.token_hex(8)}"


def _write_temporary_bytes_at(directory_fd: int, name: str, payload: bytes) -> str:
    temporary = _temporary_name(name)
    descriptor = -1
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=directory_fd,
        )
        _write_all(descriptor, payload)
        os.fsync(descriptor)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
            descriptor = -1
        try:
            os.unlink(temporary, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return temporary


def _atomic_replace_bytes_at(
    directory_fd: int,
    name: str,
    payload: bytes,
    *,
    label: str,
) -> None:
    _require_safe_relative_name(name, label=label)
    temporary = _write_temporary_bytes_at(directory_fd, name, payload)
    try:
        try:
            _read_regular_bytes_at(directory_fd, name, label=label)
        except FileNotFoundError:
            pass
        os.replace(
            temporary,
            name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        os.fsync(directory_fd)
    finally:
        try:
            os.unlink(temporary, dir_fd=directory_fd)
        except FileNotFoundError:
            pass


def _atomic_publish_new_bytes_at(
    directory_fd: int,
    name: str,
    payload: bytes,
    *,
    label: str,
) -> bool:
    """Publish immutable bytes without overwriting an independently created name."""

    _require_safe_relative_name(name, label=label)
    temporary = _write_temporary_bytes_at(directory_fd, name, payload)
    published = False
    try:
        try:
            os.link(
                temporary,
                name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
            published = True
        except FileExistsError:
            existing = _read_regular_bytes_at(directory_fd, name, label=label)
            if existing != payload:
                raise EmbeddedDecisionArtifactStoreError(
                    f"{label} conflicts with independently published bytes"
                )
        os.unlink(temporary, dir_fd=directory_fd)
        os.fsync(directory_fd)
        return published
    finally:
        try:
            os.unlink(temporary, dir_fd=directory_fd)
        except FileNotFoundError:
            pass


def _linked_references(
    artifact: PersistableEmbeddedDecisionArtifact,
) -> tuple[ContractArtifactReference, ...]:
    links: list[ContractArtifactReference] = [artifact.parent_artifact]
    if isinstance(artifact, EmbeddedDomainProposal):
        links.extend(artifact.evidence_artifacts)
    elif isinstance(artifact, EmbeddedCoordinatorDecision):
        links.extend(artifact.evidence_artifacts)
        links.extend(artifact.proposal_artifacts)
    elif isinstance(artifact, EmbeddedHumanDecision):
        links.append(artifact.coordinator_decision)
    elif isinstance(artifact, BoundedExecutionAuthorization):
        links.append(artifact.accepted_decision)
        if artifact.human_decision is not None:
            links.append(artifact.human_decision)
        if artifact.resume_of is not None:
            links.append(artifact.resume_of)
        if artifact.resume_review is not None:
            links.append(artifact.resume_review)
    elif isinstance(artifact, EmbeddedBoundedExecutionResult):
        links.append(artifact.accepted_decision)
        if artifact.resume_of is not None:
            links.append(artifact.resume_of)
    elif isinstance(artifact, EmbeddedCoordinatorReview):
        links.append(artifact.execution_result)
    elif isinstance(artifact, EmbeddedDecisionReceipt):
        links.extend(
            (
                artifact.accepted_decision,
                artifact.execution_authorization,
                artifact.execution_result,
                artifact.coordinator_review,
            )
        )
        if artifact.human_decision is not None:
            links.append(artifact.human_decision)
        links.extend(artifact.prior_execution_authorizations)
        links.extend(artifact.prior_execution_results)
        links.extend(artifact.prior_coordinator_reviews)
    unique: list[ContractArtifactReference] = []
    for link in links:
        if link not in unique:
            unique.append(link)
    return tuple(unique)


def _require_link_chronology(
    artifact: PersistableEmbeddedDecisionArtifact,
    known_artifacts: dict[
        ContractArtifactReference,
        PersistableEmbeddedDecisionArtifact,
    ],
) -> None:
    for link in _linked_references(artifact):
        linked_artifact = known_artifacts.get(link)
        if (
            linked_artifact is not None
            and artifact.created_at < linked_artifact.created_at
        ):
            raise EmbeddedDecisionArtifactStoreError(
                "Artifact timestamp precedes a persisted dependency"
            )


def _require_known_artifact[ArtifactT: ContractArtifact](
    known_artifacts: dict[
        ContractArtifactReference,
        PersistableEmbeddedDecisionArtifact,
    ],
    reference: ContractArtifactReference,
    artifact_type: type[ArtifactT],
    *,
    label: str,
) -> ArtifactT:
    artifact = known_artifacts.get(reference)
    if artifact is None:
        raise EmbeddedDecisionArtifactStoreError(
            f"Exact {label} dependency is not committed in this journal"
        )
    if not isinstance(artifact, artifact_type):
        raise EmbeddedDecisionArtifactStoreError(
            f"Exact {label} dependency has the wrong artifact kind"
        )
    return artifact


def _require_unique_references(
    references: Sequence[ContractArtifactReference],
    *,
    label: str,
) -> None:
    if len(set(references)) != len(references):
        raise EmbeddedDecisionArtifactStoreError(
            f"{label} dependencies contain duplicate or extra references"
        )


def _decision_dependencies(
    decision: EmbeddedCoordinatorDecision,
    known_artifacts: dict[
        ContractArtifactReference,
        PersistableEmbeddedDecisionArtifact,
    ],
) -> tuple[
    tuple[EmbeddedDomainEvidence, ...],
    tuple[EmbeddedDomainProposal, ...],
]:
    _require_unique_references(decision.evidence_artifacts, label="decision evidence")
    _require_unique_references(decision.proposal_artifacts, label="decision proposal")
    evidence = tuple(
        _require_known_artifact(
            known_artifacts,
            reference,
            EmbeddedDomainEvidence,
            label="decision evidence",
        )
        for reference in decision.evidence_artifacts
    )
    proposals = tuple(
        _require_known_artifact(
            known_artifacts,
            reference,
            EmbeddedDomainProposal,
            label="decision proposal",
        )
        for reference in decision.proposal_artifacts
    )
    validate_coordinator_decision_dependencies(
        decision,
        evidence=evidence,
        proposals=proposals,
    )
    return evidence, proposals


def _validate_proposal_dependencies(
    proposal: EmbeddedDomainProposal,
    known_artifacts: dict[
        ContractArtifactReference,
        PersistableEmbeddedDecisionArtifact,
    ],
) -> None:
    _require_unique_references(proposal.evidence_artifacts, label="proposal evidence")
    for reference in proposal.evidence_artifacts:
        evidence = _require_known_artifact(
            known_artifacts,
            reference,
            EmbeddedDomainEvidence,
            label="proposal evidence",
        )
        if evidence.identity != proposal.identity:
            raise EmbeddedDecisionArtifactStoreError(
                "Proposal evidence identity or digests are stale"
            )
        if proposal.created_at < evidence.created_at:
            raise EmbeddedDecisionArtifactStoreError(
                "Proposal timestamp precedes its exact evidence"
            )


def _validate_outer_stage_attempt_identity(
    artifact: PersistableEmbeddedDecisionArtifact,
    known_artifacts: dict[
        ContractArtifactReference,
        PersistableEmbeddedDecisionArtifact,
    ],
) -> None:
    seal_key = outer_stage_attempt_seal_key(artifact.identity)
    if any(
        outer_stage_attempt_seal_key(known.identity) == seal_key
        and known.identity != artifact.identity
        for known in known_artifacts.values()
    ):
        raise EmbeddedDecisionArtifactStoreError(
            "Outer stage attempt already has a different full decision identity; "
            "bound digests or dependencies are stale"
        )


def _committed_human_decisions(
    decision_reference: ContractArtifactReference,
    known_artifacts: dict[
        ContractArtifactReference,
        PersistableEmbeddedDecisionArtifact,
    ],
) -> tuple[EmbeddedHumanDecision, ...]:
    return tuple(
        artifact
        for artifact in known_artifacts.values()
        if isinstance(artifact, EmbeddedHumanDecision)
        and artifact.coordinator_decision == decision_reference
    )


def _require_exact_committed_human_decision(
    decision_reference: ContractArtifactReference,
    expected_reference: ContractArtifactReference | None,
    known_artifacts: dict[
        ContractArtifactReference,
        PersistableEmbeddedDecisionArtifact,
    ],
) -> None:
    committed = _committed_human_decisions(decision_reference, known_artifacts)
    committed_references = tuple(artifact_reference(item) for item in committed)
    expected = () if expected_reference is None else (expected_reference,)
    if committed_references != expected:
        raise EmbeddedDecisionArtifactStoreError(
            "Coordinator decision requires exactly its one durable human decision; "
            "a revised human disposition requires a new coordinator decision"
        )


def _result_and_review_for_authorization(
    authorization: BoundedExecutionAuthorization,
    known_artifacts: dict[
        ContractArtifactReference,
        PersistableEmbeddedDecisionArtifact,
    ],
) -> tuple[EmbeddedBoundedExecutionResult, EmbeddedCoordinatorReview]:
    authorization_ref = artifact_reference(authorization)
    results = tuple(
        artifact
        for artifact in known_artifacts.values()
        if isinstance(artifact, EmbeddedBoundedExecutionResult)
        and artifact.parent_artifact == authorization_ref
    )
    if len(results) != 1:
        raise EmbeddedDecisionArtifactStoreError(
            "Prior authorization requires exactly one committed execution result"
        )
    result = results[0]
    result_ref = artifact_reference(result)
    reviews = tuple(
        artifact
        for artifact in known_artifacts.values()
        if isinstance(artifact, EmbeddedCoordinatorReview)
        and artifact.execution_result == result_ref
    )
    if len(reviews) != 1:
        raise EmbeddedDecisionArtifactStoreError(
            "Prior execution result requires exactly one committed coordinator review"
        )
    return result, reviews[0]


def _prior_authorization_lineage(
    authorization: BoundedExecutionAuthorization,
    known_artifacts: dict[
        ContractArtifactReference,
        PersistableEmbeddedDecisionArtifact,
    ],
) -> tuple[
    tuple[BoundedExecutionAuthorization, ...],
    tuple[EmbeddedBoundedExecutionResult, ...],
    tuple[EmbeddedCoordinatorReview, ...],
]:
    prior_authorizations = tuple(
        sorted(
            (
                artifact
                for artifact in known_artifacts.values()
                if isinstance(artifact, BoundedExecutionAuthorization)
                and artifact.operation_id == authorization.operation_id
            ),
            key=lambda item: item.attempt,
        )
    )
    expected_attempts = tuple(range(1, authorization.attempt))
    if tuple(item.attempt for item in prior_authorizations) != expected_attempts:
        raise EmbeddedDecisionArtifactStoreError(
            "Authorization dependency history is missing, extra, or noncontiguous"
        )
    results: list[EmbeddedBoundedExecutionResult] = []
    reviews: list[EmbeddedCoordinatorReview] = []
    for prior_authorization in prior_authorizations:
        result, review = _result_and_review_for_authorization(
            prior_authorization,
            known_artifacts,
        )
        results.append(result)
        reviews.append(review)
    return prior_authorizations, tuple(results), tuple(reviews)


def _validate_authorization_dependencies(
    authorization: BoundedExecutionAuthorization,
    known_artifacts: dict[
        ContractArtifactReference,
        PersistableEmbeddedDecisionArtifact,
    ],
) -> None:
    decision = _require_known_artifact(
        known_artifacts,
        authorization.accepted_decision,
        EmbeddedCoordinatorDecision,
        label="authorization decision",
    )
    evidence, proposals = _decision_dependencies(decision, known_artifacts)
    human_decision = (
        _require_known_artifact(
            known_artifacts,
            authorization.human_decision,
            EmbeddedHumanDecision,
            label="authorization human decision",
        )
        if authorization.human_decision is not None
        else None
    )
    _require_exact_committed_human_decision(
        artifact_reference(decision),
        authorization.human_decision,
        known_artifacts,
    )
    if authorization.outer_stage_attempt_seal_key != outer_stage_attempt_seal_key(
        authorization.identity
    ):
        raise EmbeddedDecisionArtifactStoreError(
            "Execution authorization carries a stale outer stage-attempt seal"
        )
    if any(
        isinstance(artifact, BoundedExecutionAuthorization)
        and artifact.outer_stage_attempt_seal_key
        == authorization.outer_stage_attempt_seal_key
        and artifact.identity != authorization.identity
        for artifact in known_artifacts.values()
    ):
        raise EmbeddedDecisionArtifactStoreError(
            "Outer stage attempt already has execution authority for a different "
            "full decision identity"
        )
    if authorization.execution_effect == "mutation" and any(
        isinstance(artifact, BoundedExecutionAuthorization)
        and artifact.execution_effect == "mutation"
        and artifact.outer_stage_attempt_seal_key
        == authorization.outer_stage_attempt_seal_key
        and (
            artifact.accepted_decision != authorization.accepted_decision
            or artifact.accepted_decision_digest
            != authorization.accepted_decision_digest
            or artifact.operation_id != authorization.operation_id
            or artifact.mutation_id != authorization.mutation_id
        )
        for artifact in known_artifacts.values()
    ):
        raise EmbeddedDecisionArtifactStoreError(
            "Outer stage attempt already has mutating authority for a different "
            "accepted decision or operation"
        )
    prior_authorizations, prior_results, prior_reviews = _prior_authorization_lineage(
        authorization, known_artifacts
    )
    expected = authorize_bounded_execution(
        decision,
        expected_identity=authorization.identity,
        evidence=evidence,
        existing_authorizations=prior_authorizations,
        proposals=proposals,
        executor=authorization.executor,
        execution_effect=authorization.execution_effect,
        human_decision=human_decision,
        historical_results=prior_results[:-1],
        historical_reviews=prior_reviews[:-1],
        prior_result=prior_results[-1] if prior_results else None,
        prior_review=prior_reviews[-1] if prior_reviews else None,
        created_at=authorization.created_at,
    )
    if expected != authorization:
        raise EmbeddedDecisionArtifactStoreError(
            "Execution authorization differs from its complete exact authority chain"
        )


def _validate_result_dependencies(
    result: EmbeddedBoundedExecutionResult,
    known_artifacts: dict[
        ContractArtifactReference,
        PersistableEmbeddedDecisionArtifact,
    ],
) -> None:
    authorization = _require_known_artifact(
        known_artifacts,
        result.parent_artifact,
        BoundedExecutionAuthorization,
        label="result authorization",
    )
    authorization_ref = artifact_reference(authorization)
    if any(
        isinstance(artifact, EmbeddedBoundedExecutionResult)
        and artifact.parent_artifact == authorization_ref
        for artifact in known_artifacts.values()
    ):
        raise EmbeddedDecisionArtifactStoreError(
            "Execution authorization already has a committed result"
        )
    validate_bounded_execution_result(result, authorization)


def _validate_review_dependencies(
    review: EmbeddedCoordinatorReview,
    known_artifacts: dict[
        ContractArtifactReference,
        PersistableEmbeddedDecisionArtifact,
    ],
) -> None:
    result = _require_known_artifact(
        known_artifacts,
        review.execution_result,
        EmbeddedBoundedExecutionResult,
        label="review result",
    )
    if any(
        isinstance(artifact, EmbeddedCoordinatorReview)
        and artifact.execution_result == review.execution_result
        for artifact in known_artifacts.values()
    ):
        raise EmbeddedDecisionArtifactStoreError(
            "Execution result already has a committed coordinator review"
        )
    validate_coordinator_review(review, result)
    authorization = _require_known_artifact(
        known_artifacts,
        result.parent_artifact,
        BoundedExecutionAuthorization,
        label="review authorization",
    )
    decision = _require_known_artifact(
        known_artifacts,
        authorization.accepted_decision,
        EmbeddedCoordinatorDecision,
        label="review decision",
    )
    if review.semantic_decision_owner != decision.producer:
        raise EmbeddedDecisionArtifactStoreError(
            "Coordinator review was not authored by the semantic decision owner"
        )


def _validate_receipt_dependencies(
    receipt: EmbeddedDecisionReceipt,
    known_artifacts: dict[
        ContractArtifactReference,
        PersistableEmbeddedDecisionArtifact,
    ],
) -> None:
    decision = _require_known_artifact(
        known_artifacts,
        receipt.accepted_decision,
        EmbeddedCoordinatorDecision,
        label="receipt decision",
    )
    evidence, proposals = _decision_dependencies(decision, known_artifacts)
    human_decision = (
        _require_known_artifact(
            known_artifacts,
            receipt.human_decision,
            EmbeddedHumanDecision,
            label="receipt human decision",
        )
        if receipt.human_decision is not None
        else None
    )
    _require_exact_committed_human_decision(
        artifact_reference(decision),
        receipt.human_decision,
        known_artifacts,
    )
    authorization = _require_known_artifact(
        known_artifacts,
        receipt.execution_authorization,
        BoundedExecutionAuthorization,
        label="receipt authorization",
    )
    result = _require_known_artifact(
        known_artifacts,
        receipt.execution_result,
        EmbeddedBoundedExecutionResult,
        label="receipt result",
    )
    review = _require_known_artifact(
        known_artifacts,
        receipt.coordinator_review,
        EmbeddedCoordinatorReview,
        label="receipt review",
    )
    _require_unique_references(
        receipt.prior_execution_authorizations,
        label="receipt prior authorization",
    )
    _require_unique_references(
        receipt.prior_execution_results,
        label="receipt prior result",
    )
    _require_unique_references(
        receipt.prior_coordinator_reviews,
        label="receipt prior review",
    )
    prior_lineage = PersistedExecutionLineage(
        authorizations=tuple(
            _require_known_artifact(
                known_artifacts,
                reference,
                BoundedExecutionAuthorization,
                label="receipt prior authorization",
            )
            for reference in receipt.prior_execution_authorizations
        ),
        results=tuple(
            _require_known_artifact(
                known_artifacts,
                reference,
                EmbeddedBoundedExecutionResult,
                label="receipt prior result",
            )
            for reference in receipt.prior_execution_results
        ),
        reviews=tuple(
            _require_known_artifact(
                known_artifacts,
                reference,
                EmbeddedCoordinatorReview,
                label="receipt prior review",
            )
            for reference in receipt.prior_coordinator_reviews
        ),
    )
    if any(
        isinstance(artifact, EmbeddedDecisionReceipt)
        and (
            artifact.execution_authorization == receipt.execution_authorization
            or artifact.coordinator_review == receipt.coordinator_review
        )
        for artifact in known_artifacts.values()
    ):
        raise EmbeddedDecisionArtifactStoreError(
            "Execution attempt already has a committed decision receipt"
        )
    validate_decision_receipt(
        receipt,
        decision=decision,
        evidence=evidence,
        proposals=proposals,
        authorization=authorization,
        result=result,
        review=review,
        prior_lineage=prior_lineage,
        human_decision=human_decision,
    )


def _validate_cross_artifact_dependencies(
    artifact: PersistableEmbeddedDecisionArtifact,
    known_artifacts: dict[
        ContractArtifactReference,
        PersistableEmbeddedDecisionArtifact,
    ],
) -> None:
    try:
        _validate_outer_stage_attempt_identity(artifact, known_artifacts)
        if isinstance(artifact, EmbeddedDomainProposal):
            _validate_proposal_dependencies(artifact, known_artifacts)
        elif isinstance(artifact, EmbeddedCoordinatorDecision):
            _decision_dependencies(artifact, known_artifacts)
        elif isinstance(artifact, EmbeddedHumanDecision):
            decision = _require_known_artifact(
                known_artifacts,
                artifact.coordinator_decision,
                EmbeddedCoordinatorDecision,
                label="human decision",
            )
            validate_human_decision(artifact, decision)
            if _committed_human_decisions(
                artifact.coordinator_decision,
                known_artifacts,
            ):
                raise EmbeddedDecisionArtifactStoreError(
                    "Coordinator decision already has a durable human decision; "
                    "a revised disposition requires a new coordinator decision"
                )
        elif isinstance(artifact, BoundedExecutionAuthorization):
            _validate_authorization_dependencies(artifact, known_artifacts)
        elif isinstance(artifact, EmbeddedBoundedExecutionResult):
            _validate_result_dependencies(artifact, known_artifacts)
        elif isinstance(artifact, EmbeddedCoordinatorReview):
            _validate_review_dependencies(artifact, known_artifacts)
        elif isinstance(artifact, EmbeddedDecisionReceipt):
            _validate_receipt_dependencies(artifact, known_artifacts)
    except EmbeddedDecisionArtifactStoreError:
        raise
    except (EmbeddedDecisionContractError, TypeError, ValueError) as exc:
        raise EmbeddedDecisionArtifactStoreError(
            f"Persisted cross-artifact decision chain is invalid: {exc}"
        ) from exc


class EmbeddedDecisionArtifactStore:
    """Append-only durable artifact store rooted inside one explicit run root."""

    def __init__(self, run_root: str | Path) -> None:
        candidate = Path(run_root).expanduser()
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        self._run_root = Path(os.path.abspath(candidate))
        self._store_root = self._run_root / _STORE_DIRECTORY
        self._thread_state = threading.local()
        self._run_root_identity, self._store_identity = self._initialize_store_root()

    @property
    def run_root(self) -> Path:
        return self._run_root

    @property
    def store_root(self) -> Path:
        return self._store_root

    def _initialize_store_root(
        self,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        for path, label in (
            (self._run_root, "run root"),
            (self._store_root, "artifact store root"),
        ):
            try:
                descriptor, _chain = _open_directory_no_symlinks(path)
            except OSError as exc:
                raise EmbeddedDecisionArtifactStoreError(
                    f"Embedded decision {label} must not contain symlinks: {exc}"
                ) from exc
            else:
                os.close(descriptor)
        run_fd = -1
        store_fd = -1
        try:
            run_fd, _chain = _open_directory_no_symlinks(
                self._run_root,
                create_missing=False,
            )
            store_fd = _open_child_directory_at(
                run_fd,
                _STORE_DIRECTORY,
                label="Embedded decision artifact store root",
                create_missing=False,
            )
            return (
                _directory_identity(run_fd, label="Embedded decision run root"),
                _directory_identity(
                    store_fd,
                    label="Embedded decision artifact store root",
                ),
            )
        except OSError as exc:  # pragma: no cover - initialized immediately above
            raise EmbeddedDecisionArtifactStoreError(
                f"Could not pin initialized embedded decision roots: {exc}"
            ) from exc
        finally:
            if store_fd >= 0:
                os.close(store_fd)
            if run_fd >= 0:
                os.close(run_fd)

    def _validate_run_root_path_binding(self, run_root_fd: int) -> tuple[int, int]:
        descriptor_identity = _directory_identity(
            run_root_fd,
            label="Embedded decision run root",
        )
        try:
            named = os.stat(self._run_root, follow_symlinks=False)
        except OSError as exc:
            raise EmbeddedDecisionArtifactStoreError(
                f"Embedded decision run root became stale: {exc}"
            ) from exc
        named_identity = (named.st_dev, named.st_ino)
        if (
            descriptor_identity != self._run_root_identity
            or named_identity != descriptor_identity
            or not stat.S_ISDIR(named.st_mode)
            or named.st_uid != os.geteuid()
        ):
            raise EmbeddedDecisionArtifactStoreError(
                "Embedded decision run root was renamed or replaced; "
                "the outer mutation lease anchor is stale"
            )
        return descriptor_identity

    def _pin_run_root(self) -> int:
        try:
            descriptor, _chain = _open_directory_no_symlinks(
                self._run_root,
                create_missing=False,
            )
        except OSError as exc:
            raise EmbeddedDecisionArtifactStoreError(
                f"Could not pin embedded decision run root: {exc}"
            ) from exc
        try:
            self._validate_run_root_path_binding(descriptor)
        except BaseException:
            os.close(descriptor)
            raise
        return descriptor

    def _validate_store_path_binding(self, store_fd: int) -> tuple[int, int]:
        """Verify the supplied pathname still names the initially pinned inode."""

        descriptor_identity = _directory_identity(
            store_fd,
            label="Embedded decision artifact store root",
        )
        try:
            named = os.stat(self._store_root, follow_symlinks=False)
        except OSError as exc:
            raise EmbeddedDecisionArtifactStoreError(
                f"Embedded decision artifact store root became stale: {exc}"
            ) from exc
        named_identity = (named.st_dev, named.st_ino)
        if (
            descriptor_identity != self._store_identity
            or named_identity != descriptor_identity
            or not stat.S_ISDIR(named.st_mode)
            or named.st_uid != os.geteuid()
        ):
            raise EmbeddedDecisionArtifactStoreError(
                "Embedded decision artifact store root was renamed or replaced; "
                "the run root is stale"
            )
        return descriptor_identity

    def _pin_store_root(self) -> int:
        try:
            descriptor, _chain = _open_directory_no_symlinks(
                self._store_root,
                create_missing=False,
            )
        except OSError as exc:
            raise EmbeddedDecisionArtifactStoreError(
                f"Could not pin embedded decision store root: {exc}"
            ) from exc
        try:
            self._validate_store_path_binding(descriptor)
        except BaseException:
            os.close(descriptor)
            raise
        return descriptor

    @contextmanager
    def _locked(self) -> Iterator[int]:
        active_store_fd = getattr(self._thread_state, "store_fd", None)
        if active_store_fd is not None:
            self._validate_store_path_binding(active_store_fd)
            try:
                yield active_store_fd
            finally:
                self._validate_store_path_binding(active_store_fd)
            return

        store_fd = self._pin_store_root()
        lock_fd = -1
        lock = None
        try:
            try:
                flags = (
                    os.O_RDWR
                    | os.O_CREAT
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                )
                lock_fd = os.open(_LOCK_NAME, flags, 0o600, dir_fd=store_fd)
                lock_identity = self._validate_lock_descriptor(
                    store_fd,
                    lock_fd,
                    name=_LOCK_NAME,
                    label="Journal lock",
                )
                lock = blocking_exclusive_descriptor_lock(lock_fd)
                lock.__enter__()
                self._validate_lock_descriptor(
                    store_fd,
                    lock_fd,
                    name=_LOCK_NAME,
                    label="Journal lock",
                    expected_identity=lock_identity,
                )
            except OSError as exc:
                raise EmbeddedDecisionArtifactStoreError(
                    f"Could not acquire safe embedded decision journal lock: {exc}"
                ) from exc
            self._validate_store_path_binding(store_fd)
            self._thread_state.store_fd = store_fd
            try:
                yield store_fd
            finally:
                del self._thread_state.store_fd
                self._validate_store_path_binding(store_fd)
                self._validate_lock_descriptor(
                    store_fd,
                    lock_fd,
                    name=_LOCK_NAME,
                    label="Journal lock",
                    expected_identity=lock_identity,
                )
        finally:
            if lock_fd >= 0:
                if lock is not None:
                    lock.__exit__(None, None, None)
                os.close(lock_fd)
            os.close(store_fd)

    @staticmethod
    def _validate_lock_descriptor(
        directory_fd: int,
        lock_fd: int,
        *,
        name: str,
        label: str,
        expected_identity: tuple[int, int] | None = None,
    ) -> tuple[int, int]:
        _require_safe_relative_name(name, label=label)
        descriptor_stat = os.fstat(lock_fd)
        named_stat = os.stat(
            name,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        descriptor_identity = (descriptor_stat.st_dev, descriptor_stat.st_ino)
        named_identity = (named_stat.st_dev, named_stat.st_ino)
        effective_uid = os.geteuid()
        if (
            not stat.S_ISREG(descriptor_stat.st_mode)
            or not stat.S_ISREG(named_stat.st_mode)
            or descriptor_stat.st_nlink != 1
            or named_stat.st_nlink != 1
            or descriptor_stat.st_uid != effective_uid
            or named_stat.st_uid != effective_uid
        ):
            raise EmbeddedDecisionArtifactStoreError(
                f"{label} must be a same-user single-link regular inode"
            )
        if descriptor_identity != named_identity:
            raise EmbeddedDecisionArtifactStoreError(
                f"{label} inode changed during acquisition"
            )
        if expected_identity is not None and descriptor_identity != expected_identity:
            raise EmbeddedDecisionArtifactStoreError(
                f"{label} inode was substituted while held"
            )
        return descriptor_identity

    @staticmethod
    def _mutation_lease_names(
        authorization: BoundedExecutionAuthorization,
    ) -> tuple[str, str]:
        seal_key = authorization.outer_stage_attempt_seal_key
        prefix = f"{_MUTATION_LEASE_LOCK_PREFIX}{seal_key}"
        lock_name = f"{prefix}{_MUTATION_LEASE_LOCK_SUFFIX}"
        record_name = (
            f"{prefix}-{artifact_reference(authorization).sha256}"
            f"{_MUTATION_LEASE_RECORD_SUFFIX}"
        )
        _require_safe_relative_name(lock_name, label="Mutation lease lock")
        _require_safe_relative_name(record_name, label="Mutation lease record")
        return lock_name, record_name

    @contextmanager
    def _mutation_lease_anchor_locked(
        self,
        authorization: BoundedExecutionAuthorization,
    ) -> Iterator[tuple[int, int, tuple[int, int], tuple[int, int], str]]:
        run_root_fd = self._pin_run_root()
        lock_fd = -1
        lock = None
        lock_name, record_name = self._mutation_lease_names(authorization)
        try:
            try:
                flags = (
                    os.O_RDWR
                    | os.O_CREAT
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                )
                lock_fd = os.open(lock_name, flags, 0o600, dir_fd=run_root_fd)
                lock_identity = self._validate_lock_descriptor(
                    run_root_fd,
                    lock_fd,
                    name=lock_name,
                    label="Mutation lease lock",
                )
                lock = blocking_exclusive_descriptor_lock(lock_fd)
                lock.__enter__()
                run_root_identity = self._validate_run_root_path_binding(run_root_fd)
                self._validate_lock_descriptor(
                    run_root_fd,
                    lock_fd,
                    name=lock_name,
                    label="Mutation lease lock",
                    expected_identity=lock_identity,
                )
            except OSError as exc:
                raise EmbeddedDecisionArtifactStoreError(
                    f"Could not acquire safe outer mutation lease lock: {exc}"
                ) from exc
            try:
                yield (
                    run_root_fd,
                    lock_fd,
                    run_root_identity,
                    lock_identity,
                    record_name,
                )
            finally:
                self._validate_run_root_path_binding(run_root_fd)
                self._validate_lock_descriptor(
                    run_root_fd,
                    lock_fd,
                    name=lock_name,
                    label="Mutation lease lock",
                    expected_identity=lock_identity,
                )
        finally:
            if lock_fd >= 0:
                if lock is not None:
                    lock.__exit__(None, None, None)
                os.close(lock_fd)
            os.close(run_root_fd)

    @staticmethod
    def _mutation_lease_record_prefix(seal_key: str) -> str:
        return f"{_MUTATION_LEASE_LOCK_PREFIX}{seal_key}-"

    def _load_mutation_lease_name_locked(
        self,
        run_root_fd: int,
        name: str,
    ) -> tuple[DurableEmbeddedMutationLease, tuple[int, int]]:
        identity = _regular_file_identity_at(
            run_root_fd,
            name,
            label="Mutation lease record",
        )
        raw = _read_regular_bytes_at(
            run_root_fd,
            name,
            label="Mutation lease record",
        )
        _regular_file_identity_at(
            run_root_fd,
            name,
            label="Mutation lease record",
            expected_identity=identity,
        )
        try:
            lease = DurableEmbeddedMutationLease.model_validate_json(raw)
        except ValueError as exc:
            raise EmbeddedDecisionArtifactStoreError(
                f"Outer mutation lease is malformed or stale: {exc}"
            ) from exc
        if raw != _canonical_mutation_lease_bytes(lease):
            raise EmbeddedDecisionArtifactStoreError(
                "Outer mutation lease bytes are not canonical"
            )
        return lease, identity

    def _load_mutation_leases_locked(
        self,
        run_root_fd: int,
        seal_key: str,
    ) -> tuple[tuple[DurableEmbeddedMutationLease, tuple[int, int]], ...]:
        prefix = self._mutation_lease_record_prefix(seal_key)
        try:
            names = sorted(os.listdir(run_root_fd))
        except OSError as exc:
            raise EmbeddedDecisionArtifactStoreError(
                f"Could not enumerate outer mutation leases: {exc}"
            ) from exc
        leases: list[tuple[DurableEmbeddedMutationLease, tuple[int, int]]] = []
        for name in names:
            if not name.startswith(prefix) or not name.endswith(
                _MUTATION_LEASE_RECORD_SUFFIX
            ):
                continue
            lease, identity = self._load_mutation_lease_name_locked(
                run_root_fd,
                name,
            )
            if lease.outer_stage_attempt_seal_key != seal_key:
                raise EmbeddedDecisionArtifactStoreError(
                    "Outer mutation lease filename differs from its stage seal"
                )
            expected_name = self._mutation_lease_names_from_reference(
                lease.outer_stage_attempt_seal_key,
                lease.authorization,
            )[1]
            if name != expected_name:
                raise EmbeddedDecisionArtifactStoreError(
                    "Outer mutation lease filename is not canonical"
                )
            leases.append((lease, identity))
        return tuple(leases)

    @staticmethod
    def _mutation_lease_names_from_reference(
        seal_key: str,
        authorization: ContractArtifactReference,
    ) -> tuple[str, str]:
        prefix = f"{_MUTATION_LEASE_LOCK_PREFIX}{seal_key}"
        return (
            f"{prefix}{_MUTATION_LEASE_LOCK_SUFFIX}",
            f"{prefix}-{authorization.sha256}{_MUTATION_LEASE_RECORD_SUFFIX}",
        )

    def _build_mutation_lease(
        self,
        authorization: BoundedExecutionAuthorization,
        *,
        run_root_identity: tuple[int, int],
        store_identity: tuple[int, int],
    ) -> DurableEmbeddedMutationLease:
        if authorization.execution_effect != "mutation":
            raise EmbeddedDecisionArtifactStoreError(
                "Outer mutation leases require mutating authorization"
            )
        if authorization.mutation_id is None:  # pragma: no cover - model invariant
            raise EmbeddedDecisionArtifactStoreError(
                "Mutating authorization is missing mutation identity"
            )
        authorization_reference = artifact_reference(authorization)
        lease_digest = _mutation_lease_digest(
            outer_stage_attempt_seal_key=(authorization.outer_stage_attempt_seal_key),
            identity=authorization.identity,
            accepted_decision=authorization.accepted_decision,
            authorization=authorization_reference,
            accepted_decision_digest=authorization.accepted_decision_digest,
            operation_id=authorization.operation_id,
            mutation_id=authorization.mutation_id,
            attempt=authorization.attempt,
            anchor_run_root_device=run_root_identity[0],
            anchor_run_root_inode=run_root_identity[1],
            journal_store_device=store_identity[0],
            journal_store_inode=store_identity[1],
            created_at=authorization.created_at,
        )
        return DurableEmbeddedMutationLease(
            outer_stage_attempt_seal_key=(authorization.outer_stage_attempt_seal_key),
            identity=authorization.identity,
            accepted_decision=authorization.accepted_decision,
            authorization=authorization_reference,
            accepted_decision_digest=authorization.accepted_decision_digest,
            operation_id=authorization.operation_id,
            mutation_id=authorization.mutation_id,
            attempt=authorization.attempt,
            anchor_run_root_device=run_root_identity[0],
            anchor_run_root_inode=run_root_identity[1],
            journal_store_device=store_identity[0],
            journal_store_inode=store_identity[1],
            created_at=authorization.created_at,
            lease_sha256=lease_digest,
        )

    def _validate_existing_mutation_leases(
        self,
        authorization: BoundedExecutionAuthorization,
        leases: Sequence[tuple[DurableEmbeddedMutationLease, tuple[int, int]]],
        *,
        run_root_identity: tuple[int, int],
        store_identity: tuple[int, int],
    ) -> DurableEmbeddedMutationLease | None:
        authorization_reference = artifact_reference(authorization)
        exact_authorization_lease: DurableEmbeddedMutationLease | None = None
        for lease, _lease_identity in leases:
            exact_operation = (
                lease.outer_stage_attempt_seal_key
                == authorization.outer_stage_attempt_seal_key
                and lease.identity == authorization.identity
                and lease.accepted_decision == authorization.accepted_decision
                and lease.accepted_decision_digest
                == authorization.accepted_decision_digest
                and lease.operation_id == authorization.operation_id
                and lease.mutation_id == authorization.mutation_id
            )
            if not exact_operation:
                raise EmbeddedDecisionArtifactStoreError(
                    "Outer stage-attempt mutation lease conflicts with stale identity "
                    "or operation"
                )
            if (
                lease.anchor_run_root_device,
                lease.anchor_run_root_inode,
            ) != run_root_identity:
                raise EmbeddedDecisionArtifactStoreError(
                    "Outer mutation lease is bound to a stale run-root anchor"
                )
            leased_store_identity = (
                lease.journal_store_device,
                lease.journal_store_inode,
            )
            if leased_store_identity != store_identity:
                raise EmbeddedDecisionMutationLeaseReconciliationError(
                    EmbeddedDecisionMutationLeaseReconciliationRequired(
                        disposition="stale_store",
                        authorization=lease.authorization,
                        operation_id=lease.operation_id,
                        mutation_id=lease.mutation_id,
                        attempt=lease.attempt,
                        outer_stage_attempt_seal_key=(
                            lease.outer_stage_attempt_seal_key
                        ),
                        lease_sha256=lease.lease_sha256,
                        leased_store_device=leased_store_identity[0],
                        leased_store_inode=leased_store_identity[1],
                        active_store_device=store_identity[0],
                        active_store_inode=store_identity[1],
                    )
                )
            if lease.attempt == authorization.attempt:
                if lease.authorization != authorization_reference:
                    raise EmbeddedDecisionArtifactStoreError(
                        "Outer stage-attempt mutation lease already binds a "
                        "different authorization for this operation attempt"
                    )
                exact_authorization_lease = lease
        return exact_authorization_lease

    @staticmethod
    def _same_store_mutation_reconciliation(
        lease: DurableEmbeddedMutationLease,
    ) -> EmbeddedDecisionMutationLeaseReconciliationError:
        return EmbeddedDecisionMutationLeaseReconciliationError(
            EmbeddedDecisionMutationLeaseReconciliationRequired(
                disposition="reconciliation_required",
                authorization=lease.authorization,
                operation_id=lease.operation_id,
                mutation_id=lease.mutation_id,
                attempt=lease.attempt,
                outer_stage_attempt_seal_key=lease.outer_stage_attempt_seal_key,
                lease_sha256=lease.lease_sha256,
                leased_store_device=lease.journal_store_device,
                leased_store_inode=lease.journal_store_inode,
                active_store_device=lease.journal_store_device,
                active_store_inode=lease.journal_store_inode,
            )
        )

    def _validate_artifact_outer_lease_binding(
        self,
        artifact: PersistableEmbeddedDecisionArtifact,
    ) -> None:
        run_root_fd = self._pin_run_root()
        try:
            run_root_identity = self._validate_run_root_path_binding(run_root_fd)
            seal_key = outer_stage_attempt_seal_key(artifact.identity)
            leases = self._load_mutation_leases_locked(run_root_fd, seal_key)
            for lease, _lease_identity in leases:
                if lease.identity != artifact.identity:
                    raise EmbeddedDecisionArtifactStoreError(
                        "Outer mutation lease rejects stale full decision identity"
                    )
                if (
                    lease.anchor_run_root_device,
                    lease.anchor_run_root_inode,
                ) != run_root_identity:
                    raise EmbeddedDecisionArtifactStoreError(
                        "Outer mutation lease is bound to a stale run-root anchor"
                    )
                if (
                    lease.journal_store_device,
                    lease.journal_store_inode,
                ) != self._store_identity:
                    raise EmbeddedDecisionArtifactStoreError(
                        "Outer mutation lease rejects a copied or replacement store"
                    )
            self._validate_run_root_path_binding(run_root_fd)
        finally:
            os.close(run_root_fd)

    def _publish_mutation_lease_locked(
        self,
        run_root_fd: int,
        record_name: str,
        lease: DurableEmbeddedMutationLease,
    ) -> tuple[int, int]:
        payload = _canonical_mutation_lease_bytes(lease)
        _atomic_publish_new_bytes_at(
            run_root_fd,
            record_name,
            payload,
            label="Mutation lease record",
        )
        loaded, lease_identity = self._load_mutation_lease_name_locked(
            run_root_fd,
            record_name,
        )
        if loaded != lease:
            raise EmbeddedDecisionArtifactStoreError(
                "Published outer mutation lease differs from exact authorization"
            )
        return lease_identity

    def _validate_mutation_anchor_before_invocation(
        self,
        *,
        run_root_fd: int,
        lock_fd: int,
        run_root_identity: tuple[int, int],
        lock_identity: tuple[int, int],
        lock_name: str,
        record_name: str,
        lease_identity: tuple[int, int],
    ) -> None:
        if self._validate_run_root_path_binding(run_root_fd) != run_root_identity:
            raise EmbeddedDecisionArtifactStoreError(
                "Outer mutation lease run-root anchor changed before invocation"
            )
        self._validate_lock_descriptor(
            run_root_fd,
            lock_fd,
            name=lock_name,
            label="Mutation lease lock",
            expected_identity=lock_identity,
        )
        _regular_file_identity_at(
            run_root_fd,
            record_name,
            label="Mutation lease record",
            expected_identity=lease_identity,
        )

    def _load_journal_locked(
        self,
        store_fd: int,
    ) -> tuple[
        EmbeddedDecisionArtifactJournal,
        dict[
            ContractArtifactReference,
            PersistableEmbeddedDecisionArtifact,
        ],
    ]:
        try:
            raw = _read_regular_bytes_at(
                store_fd,
                _JOURNAL_NAME,
                label="artifact journal",
            )
        except FileNotFoundError:
            return EmbeddedDecisionArtifactJournal(), {}
        try:
            journal = EmbeddedDecisionArtifactJournal.model_validate_json(raw)
        except ValueError as exc:
            raise EmbeddedDecisionArtifactStoreError(
                f"Embedded decision journal is malformed or stale: {exc}"
            ) from exc
        expected = (
            json.dumps(
                journal.model_dump(mode="json"),
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
            )
            + "\n"
        ).encode("utf-8")
        if raw != expected:
            raise EmbeddedDecisionArtifactStoreError(
                "Embedded decision journal bytes are not canonical"
            )
        known_references: set[ContractArtifactReference] = set()
        known_artifacts: dict[
            ContractArtifactReference,
            PersistableEmbeddedDecisionArtifact,
        ] = {}
        for entry in journal.entries:
            artifact = self._load_entry_locked(store_fd, entry)
            self._validate_artifact_outer_lease_binding(artifact)
            missing_links = [
                link
                for link in _linked_references(artifact)
                if link.artifact_kind in _PERSISTED_ARTIFACT_KINDS
                and link not in known_references
            ]
            if missing_links:
                raise EmbeddedDecisionArtifactStoreError(
                    "Journal artifact dependency is absent or ordered after its child"
                )
            _require_link_chronology(artifact, known_artifacts)
            _validate_cross_artifact_dependencies(artifact, known_artifacts)
            known_references.add(entry.reference)
            known_artifacts[entry.reference] = artifact
        return journal, known_artifacts

    def _write_journal_locked(
        self,
        store_fd: int,
        journal: EmbeddedDecisionArtifactJournal,
    ) -> None:
        EmbeddedDecisionArtifactJournal.model_validate(
            journal.model_dump(mode="python")
        )
        payload = (
            json.dumps(
                journal.model_dump(mode="json"),
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
            )
            + "\n"
        ).encode("utf-8")
        _atomic_replace_bytes_at(
            store_fd,
            _JOURNAL_NAME,
            payload,
            label="artifact journal",
        )

    @staticmethod
    def _artifact_components(
        reference: ContractArtifactReference,
    ) -> tuple[str, str, str]:
        if reference.artifact_kind not in _PERSISTED_ARTIFACT_KINDS:
            raise EmbeddedDecisionArtifactStoreError(
                f"Unsupported persisted artifact kind: {reference.artifact_kind}"
            )
        return "artifacts", reference.artifact_kind, f"{reference.sha256}.json"

    def _open_artifact_kind_directories(
        self,
        store_fd: int,
        artifact_kind: str,
        *,
        create_missing: bool,
    ) -> tuple[int, int]:
        if artifact_kind not in _PERSISTED_ARTIFACT_KINDS:
            raise EmbeddedDecisionArtifactStoreError(
                f"Unsupported persisted artifact kind: {artifact_kind}"
            )
        artifacts_fd = _open_child_directory_at(
            store_fd,
            "artifacts",
            label="Embedded decision artifacts directory",
            create_missing=create_missing,
        )
        try:
            kind_fd = _open_child_directory_at(
                artifacts_fd,
                artifact_kind,
                label=f"Embedded decision {artifact_kind} directory",
                create_missing=create_missing,
            )
        except BaseException:
            os.close(artifacts_fd)
            raise
        return artifacts_fd, kind_fd

    def _validate_artifact_kind_directories(
        self,
        store_fd: int,
        artifact_kind: str,
        artifacts_fd: int,
        kind_fd: int,
    ) -> None:
        _validate_named_directory_descriptor(
            store_fd,
            "artifacts",
            artifacts_fd,
            label="Embedded decision artifacts directory",
        )
        _validate_named_directory_descriptor(
            artifacts_fd,
            artifact_kind,
            kind_fd,
            label=f"Embedded decision {artifact_kind} directory",
        )

    def _load_entry_locked(
        self,
        store_fd: int,
        entry: EmbeddedDecisionArtifactJournalEntry,
    ) -> PersistableEmbeddedDecisionArtifact:
        components = self._artifact_components(entry.reference)
        if tuple(PurePosixPath(entry.relative_path).parts) != components:
            raise EmbeddedDecisionArtifactStoreError(
                "Journal artifact path differs from its exact content reference"
            )
        artifacts_fd, kind_fd = self._open_artifact_kind_directories(
            store_fd,
            entry.reference.artifact_kind,
            create_missing=False,
        )
        try:
            artifact = self._load_artifact_name_locked(
                kind_fd,
                entry.reference.artifact_kind,
                components[-1],
            )
            self._validate_artifact_kind_directories(
                store_fd,
                entry.reference.artifact_kind,
                artifacts_fd,
                kind_fd,
            )
        finally:
            os.close(kind_fd)
            os.close(artifacts_fd)
        if artifact_reference(artifact) != entry.reference:
            raise EmbeddedDecisionArtifactStoreError(
                "Persisted embedded decision artifact digest or identity is stale"
            )
        if artifact.parent_artifact != entry.parent_artifact:
            raise EmbeddedDecisionArtifactStoreError(
                "Persisted embedded decision artifact parent differs from journal"
            )
        return artifact

    def _load_artifact_name_locked(
        self,
        kind_fd: int,
        artifact_kind: str,
        name: str,
    ) -> PersistableEmbeddedDecisionArtifact:
        raw = _read_regular_bytes_at(
            kind_fd,
            name,
            label="embedded decision artifact",
        )
        try:
            artifact = _ARTIFACT_ADAPTER.validate_json(raw)
        except ValueError as exc:
            raise EmbeddedDecisionArtifactStoreError(
                f"Persisted embedded decision artifact is malformed: {exc}"
            ) from exc
        reference = artifact_reference(artifact)
        _artifacts_name, expected_kind, expected_name = self._artifact_components(
            reference
        )
        if artifact_kind != expected_kind or name != expected_name:
            raise EmbeddedDecisionArtifactStoreError(
                "Persisted artifact path differs from its canonical content digest"
            )
        expected = _canonical_artifact_text(artifact).encode("utf-8")
        if raw != expected:
            raise EmbeddedDecisionArtifactStoreError(
                "Persisted embedded decision artifact bytes are not canonical"
            )
        return artifact

    def _require_no_conflicting_artifact_file_locked(
        self,
        store_fd: int,
        reference: ContractArtifactReference,
    ) -> None:
        """Reject conflicting journaled or crash-orphaned bytes for one identity."""

        try:
            artifacts_fd, kind_fd = self._open_artifact_kind_directories(
                store_fd,
                reference.artifact_kind,
                create_missing=False,
            )
        except FileNotFoundError:
            return
        except OSError as exc:
            raise EmbeddedDecisionArtifactStoreError(
                f"Could not inspect root-confined artifact directory: {exc}"
            ) from exc
        try:
            for name in os.listdir(kind_fd):
                if not name.endswith(".json"):
                    continue
                artifact = self._load_artifact_name_locked(
                    kind_fd,
                    reference.artifact_kind,
                    name,
                )
                artifact_ref = artifact_reference(artifact)
                if (
                    artifact_ref.artifact_kind == reference.artifact_kind
                    and artifact_ref.artifact_id == reference.artifact_id
                    and artifact_ref != reference
                ):
                    raise EmbeddedDecisionArtifactStoreError(
                        "Artifact identity already exists with different canonical bytes"
                    )
            self._validate_artifact_kind_directories(
                store_fd,
                reference.artifact_kind,
                artifacts_fd,
                kind_fd,
            )
        finally:
            os.close(kind_fd)
            os.close(artifacts_fd)

    def _entry_for_reference_locked(
        self,
        journal: EmbeddedDecisionArtifactJournal,
        reference: ContractArtifactReference,
    ) -> EmbeddedDecisionArtifactJournalEntry | None:
        for entry in journal.entries:
            if (
                entry.reference.artifact_kind == reference.artifact_kind
                and entry.reference.artifact_id == reference.artifact_id
            ):
                if entry.reference != reference:
                    raise EmbeddedDecisionArtifactStoreError(
                        "Artifact identity already exists with different canonical "
                        "bytes"
                    )
                return entry
        return None

    def append(
        self,
        artifact: PersistableEmbeddedDecisionArtifact,
        *,
        committed_at: datetime | None = None,
    ) -> DurableEmbeddedDecisionArtifactCommit:
        """Durably append one artifact, replaying only exact identical bytes."""

        try:
            validated = _ARTIFACT_ADAPTER.validate_python(
                artifact.model_dump(mode="python")
            )
        except (TypeError, ValueError) as exc:
            raise EmbeddedDecisionArtifactStoreError(
                f"Embedded decision artifact is malformed or stale: {exc}"
            ) from exc
        try:
            _require_safe_artifact_id(validated.artifact_id)
        except ValueError as exc:
            raise EmbeddedDecisionArtifactStoreError(str(exc)) from exc
        self._validate_artifact_outer_lease_binding(validated)
        reference = artifact_reference(validated)
        canonical_text = _canonical_artifact_text(validated)
        if canonical_json_digest(validated) != reference.sha256:
            raise EmbeddedDecisionArtifactStoreError(
                "Embedded decision artifact digest is not canonical"
            )
        with self._locked() as store_fd:
            return self._append_locked(
                store_fd,
                validated,
                reference=reference,
                canonical_bytes=canonical_text.encode("utf-8"),
                committed_at=committed_at,
            )

    def _append_locked(
        self,
        store_fd: int,
        validated: PersistableEmbeddedDecisionArtifact,
        *,
        reference: ContractArtifactReference,
        canonical_bytes: bytes,
        committed_at: datetime | None,
    ) -> DurableEmbeddedDecisionArtifactCommit:
        journal, known_artifacts = self._load_journal_locked(store_fd)
        existing = self._entry_for_reference_locked(journal, reference)
        if existing is not None:
            self._load_entry_locked(store_fd, existing)
            return DurableEmbeddedDecisionArtifactCommit(
                reference=existing.reference,
                sequence=existing.sequence,
                entry_sha256=existing.entry_sha256,
                relative_path=existing.relative_path,
                newly_committed=False,
            )
        self._require_no_conflicting_artifact_file_locked(store_fd, reference)
        known_references = {entry.reference for entry in journal.entries}
        missing_links = [
            link
            for link in _linked_references(validated)
            if link.artifact_kind in _PERSISTED_ARTIFACT_KINDS
            and link not in known_references
        ]
        if missing_links:
            missing = ", ".join(
                f"{item.artifact_kind}:{item.artifact_id}" for item in missing_links
            )
            raise EmbeddedDecisionArtifactStoreError(
                "Artifact dependency is absent or ordered after its child: " + missing
            )
        _require_link_chronology(validated, known_artifacts)
        _validate_cross_artifact_dependencies(validated, known_artifacts)
        _artifacts_name, artifact_kind, artifact_name = self._artifact_components(
            reference
        )
        artifacts_fd, kind_fd = self._open_artifact_kind_directories(
            store_fd,
            artifact_kind,
            create_missing=True,
        )
        try:
            try:
                existing_bytes = _read_regular_bytes_at(
                    kind_fd,
                    artifact_name,
                    label="orphaned embedded decision artifact",
                )
            except FileNotFoundError:
                _atomic_publish_new_bytes_at(
                    kind_fd,
                    artifact_name,
                    canonical_bytes,
                    label="orphaned embedded decision artifact",
                )
            else:
                if existing_bytes != canonical_bytes:
                    raise EmbeddedDecisionArtifactStoreError(
                        "Orphaned artifact bytes conflict with identical digest path"
                    )
            self._validate_artifact_kind_directories(
                store_fd,
                artifact_kind,
                artifacts_fd,
                kind_fd,
            )
            sequence = len(journal.entries) + 1
            relative_path = PurePosixPath(
                "artifacts",
                artifact_kind,
                artifact_name,
            ).as_posix()
            timestamp = committed_at or datetime.now(UTC)
            if timestamp.tzinfo is None or timestamp.utcoffset() is None:
                raise EmbeddedDecisionArtifactStoreError(
                    "Artifact commit timestamp must be timezone-aware"
                )
            if timestamp < validated.created_at:
                raise EmbeddedDecisionArtifactStoreError(
                    "Artifact commit timestamp precedes artifact creation"
                )
            if journal.entries and timestamp < journal.entries[-1].committed_at:
                raise EmbeddedDecisionArtifactStoreError(
                    "Artifact commit timestamp precedes the journal tail"
                )
            previous = journal.entries[-1].entry_sha256 if journal.entries else None
            entry = EmbeddedDecisionArtifactJournalEntry(
                sequence=sequence,
                reference=reference,
                parent_artifact=validated.parent_artifact,
                relative_path=relative_path,
                committed_at=timestamp,
                previous_entry_sha256=previous,
                entry_sha256=_entry_digest(
                    sequence=sequence,
                    reference=reference,
                    parent_artifact=validated.parent_artifact,
                    relative_path=relative_path,
                    committed_at=timestamp,
                    previous_entry_sha256=previous,
                ),
            )
            updated = EmbeddedDecisionArtifactJournal(entries=(*journal.entries, entry))
            self._validate_store_path_binding(store_fd)
            self._write_journal_locked(store_fd, updated)
            self._validate_artifact_kind_directories(
                store_fd,
                artifact_kind,
                artifacts_fd,
                kind_fd,
            )
            artifact = self._load_artifact_name_locked(
                kind_fd,
                artifact_kind,
                artifact_name,
            )
            if artifact_reference(artifact) != entry.reference:
                raise EmbeddedDecisionArtifactStoreError(
                    "Persisted embedded decision artifact digest or identity is stale"
                )
            self._validate_store_path_binding(store_fd)
            return DurableEmbeddedDecisionArtifactCommit(
                reference=reference,
                sequence=entry.sequence,
                entry_sha256=entry.entry_sha256,
                relative_path=entry.relative_path,
                newly_committed=True,
            )
        finally:
            os.close(kind_fd)
            os.close(artifacts_fd)

    def journal(self) -> EmbeddedDecisionArtifactJournal:
        """Load and fully revalidate the ordered journal and all artifact bytes."""

        with self._locked() as store_fd:
            journal, _known_artifacts = self._load_journal_locked(store_fd)
            return journal

    def load_typed(
        self,
        reference: ContractArtifactReference,
        artifact_type: type[_ArtifactT],
    ) -> _ArtifactT:
        """Load one exact typed artifact only when journal and bytes still agree."""

        with self._locked() as store_fd:
            journal, known_artifacts = self._load_journal_locked(store_fd)
            entry = self._entry_for_reference_locked(journal, reference)
            if entry is None:
                raise EmbeddedDecisionArtifactStoreError(
                    "Artifact reference is not committed in this journal"
                )
            artifact = known_artifacts[entry.reference]
            if not isinstance(artifact, artifact_type):
                raise EmbeddedDecisionArtifactStoreError(
                    f"Committed artifact is not {artifact_type.__name__}"
                )
            return artifact

    def commit_authorization(
        self,
        authorization: BoundedExecutionAuthorization,
    ) -> DurableEmbeddedDecisionArtifactCommit:
        """Commit execution authority durably before any executor invocation."""

        return self.append(authorization)

    def invoke_after_authorization_commit(
        self,
        authorization: BoundedExecutionAuthorization,
        executor: Callable[[BoundedExecutionAuthorization], _ExecutionT],
    ) -> _ExecutionT:
        """Invoke an executor only after its exact authorization is durable."""

        if authorization.execution_effect == "mutation":
            return self._invoke_mutation_after_outer_lease(
                authorization,
                executor,
            )
        with self._locked() as store_fd:
            commit = self.commit_authorization(authorization)
            if not commit.newly_committed:
                raise EmbeddedDecisionAuthorizationReplayError(
                    EmbeddedDecisionAuthorizationReconciliationRequired(
                        authorization=commit.reference,
                        operation_id=authorization.operation_id,
                        mutation_id=authorization.mutation_id,
                        attempt=authorization.attempt,
                        durable_commit=commit,
                    )
                )
            self._validate_store_path_binding(store_fd)
            return executor(authorization)

    def _invoke_mutation_after_outer_lease(
        self,
        authorization: BoundedExecutionAuthorization,
        executor: Callable[[BoundedExecutionAuthorization], _ExecutionT],
    ) -> _ExecutionT:
        lock_name, expected_record_name = self._mutation_lease_names(authorization)
        with self._mutation_lease_anchor_locked(authorization) as anchor:
            (
                run_root_fd,
                lock_fd,
                run_root_identity,
                lock_identity,
                record_name,
            ) = anchor
            if record_name != expected_record_name:  # pragma: no cover - invariant
                raise EmbeddedDecisionArtifactStoreError(
                    "Outer mutation lease record name is inconsistent"
                )
            with self._locked() as store_fd:
                store_identity = self._validate_store_path_binding(store_fd)
                leases = self._load_mutation_leases_locked(
                    run_root_fd,
                    authorization.outer_stage_attempt_seal_key,
                )
                exact_lease = self._validate_existing_mutation_leases(
                    authorization,
                    leases,
                    run_root_identity=run_root_identity,
                    store_identity=store_identity,
                )
                if exact_lease is not None:
                    journal, _known_artifacts = self._load_journal_locked(store_fd)
                    entry = self._entry_for_reference_locked(
                        journal,
                        artifact_reference(authorization),
                    )
                    if entry is None:
                        raise self._same_store_mutation_reconciliation(exact_lease)
                    raise EmbeddedDecisionAuthorizationReplayError(
                        EmbeddedDecisionAuthorizationReconciliationRequired(
                            authorization=entry.reference,
                            operation_id=authorization.operation_id,
                            mutation_id=authorization.mutation_id,
                            attempt=authorization.attempt,
                            durable_commit=DurableEmbeddedDecisionArtifactCommit(
                                reference=entry.reference,
                                sequence=entry.sequence,
                                entry_sha256=entry.entry_sha256,
                                relative_path=entry.relative_path,
                                newly_committed=False,
                            ),
                        )
                    )
                commit = self.commit_authorization(authorization)
                if not commit.newly_committed:
                    raise EmbeddedDecisionAuthorizationReplayError(
                        EmbeddedDecisionAuthorizationReconciliationRequired(
                            authorization=commit.reference,
                            operation_id=authorization.operation_id,
                            mutation_id=authorization.mutation_id,
                            attempt=authorization.attempt,
                            durable_commit=commit,
                        )
                    )
                leases = self._load_mutation_leases_locked(
                    run_root_fd,
                    authorization.outer_stage_attempt_seal_key,
                )
                exact_lease = self._validate_existing_mutation_leases(
                    authorization,
                    leases,
                    run_root_identity=run_root_identity,
                    store_identity=store_identity,
                )
                if exact_lease is not None:
                    raise self._same_store_mutation_reconciliation(exact_lease)
                lease = self._build_mutation_lease(
                    authorization,
                    run_root_identity=run_root_identity,
                    store_identity=store_identity,
                )
                lease_identity = self._publish_mutation_lease_locked(
                    run_root_fd,
                    record_name,
                    lease,
                )
                self._validate_store_path_binding(store_fd)
                self._validate_mutation_anchor_before_invocation(
                    run_root_fd=run_root_fd,
                    lock_fd=lock_fd,
                    run_root_identity=run_root_identity,
                    lock_identity=lock_identity,
                    lock_name=lock_name,
                    record_name=record_name,
                    lease_identity=lease_identity,
                )
                try:
                    return executor(authorization)
                finally:
                    self._validate_mutation_anchor_before_invocation(
                        run_root_fd=run_root_fd,
                        lock_fd=lock_fd,
                        run_root_identity=run_root_identity,
                        lock_identity=lock_identity,
                        lock_name=lock_name,
                        record_name=record_name,
                        lease_identity=lease_identity,
                    )

    def append_result(
        self,
        result: EmbeddedBoundedExecutionResult,
    ) -> DurableEmbeddedDecisionArtifactCommit:
        return self.append(result)

    def append_review(
        self,
        review: EmbeddedCoordinatorReview,
    ) -> DurableEmbeddedDecisionArtifactCommit:
        return self.append(review)

    def append_receipt(
        self,
        receipt: EmbeddedDecisionReceipt,
    ) -> DurableEmbeddedDecisionArtifactCommit:
        return self.append(receipt)


__all__ = [
    "EMBEDDED_DECISION_ARTIFACT_JOURNAL_SCHEMA_VERSION",
    "EMBEDDED_DECISION_MUTATION_LEASE_SCHEMA_VERSION",
    "EMBEDDED_DECISION_MUTATION_RECONCILIATION_SCHEMA_VERSION",
    "DurableEmbeddedDecisionArtifactCommit",
    "DurableEmbeddedMutationLease",
    "EmbeddedDecisionAuthorizationReconciliationRequired",
    "EmbeddedDecisionAuthorizationReplayError",
    "EmbeddedDecisionArtifactJournal",
    "EmbeddedDecisionArtifactJournalEntry",
    "EmbeddedDecisionArtifactStore",
    "EmbeddedDecisionArtifactStoreError",
    "EmbeddedDecisionMutationLeaseReconciliationError",
    "EmbeddedDecisionMutationLeaseReconciliationRequired",
    "PersistableEmbeddedDecisionArtifact",
]
