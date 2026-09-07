# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Durable contracts for the resumable Validation Agent workflow."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from world_understanding.validation import (
    ValidationPlan,
    ValidationRequest,
    ValidationResult,
    ValidationTemplateResult,
)

VALIDATION_WORKFLOW_IDENTITY_SCHEMA_VERSION: Final = (
    "content-agent-workflows.validation-identity.v1"
)
VALIDATION_WORKFLOW_CHECKPOINT_SCHEMA_VERSION: Final = (
    "content-agent-workflows.validation-checkpoint.v1"
)
VALIDATION_WORKFLOW_EVIDENCE_SCHEMA_VERSION: Final = (
    "content-agent-workflows.validation-evidence.v1"
)
VALIDATION_WORKFLOW_SUMMARY_SCHEMA_VERSION: Final = (
    "content-agent-workflows.validation-summary.v1"
)
VALIDATION_WORK_ITEM_SCHEMA_VERSION: Final = (
    "content-agent-workflows.validation-work-item.v1"
)

_SHA256_PATTERN = r"^[0-9a-f]{64}$"
VALIDATION_EXTERNAL_RUNTIME_DEPENDENCY_PATHS: Final = frozenset(
    {"mdl://runtime/OmniPBR.mdl"}
)


def validation_external_dependency_digest(path: str) -> str:
    """Bind one declared runtime dependency that has no package-local bytes."""

    canonical = json.dumps(
        {
            "schema_version": "content-agent-workflows.validation-external-dependency.v1",
            "path": path,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ValidationWorkItemState(StrEnum):
    """Latest durable state for one ordered validation template."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class ValidationWorkflowStatus(StrEnum):
    """Terminal status for one workflow invocation."""

    COMPLETED = "completed"
    CANCELLED = "cancelled"


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ValidationArtifactIdentity(_FrozenModel):
    """Content identity for one source, reference, or generated artifact."""

    role: Literal["source", "reference", "evidence"]
    path: str = Field(min_length=1)
    kind: Literal["file", "directory", "external", "missing"]
    sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def _validate_digest_presence(self) -> Self:
        if self.kind == "missing" and self.sha256 is not None:
            raise ValueError("missing artifacts cannot carry a digest")
        if self.kind != "missing" and self.sha256 is None:
            raise ValueError("existing artifacts require a digest")
        if self.kind == "external":
            if self.role != "source":
                raise ValueError(
                    "external runtime dependencies must be source artifacts"
                )
            if self.path not in VALIDATION_EXTERNAL_RUNTIME_DEPENDENCY_PATHS:
                raise ValueError("external runtime dependency is not approved")
            if self.sha256 != validation_external_dependency_digest(self.path):
                raise ValueError("external runtime dependency digest is stale")
        return self


class ValidationWorkflowIdentity(_FrozenModel):
    """Immutable identity envelope governing checkpoint reuse."""

    schema_version: Literal["content-agent-workflows.validation-identity.v1"] = (
        VALIDATION_WORKFLOW_IDENTITY_SCHEMA_VERSION
    )
    request_digest: str = Field(pattern=_SHA256_PATTERN)
    policy_digest: str = Field(pattern=_SHA256_PATTERN)
    backend_digest: str = Field(pattern=_SHA256_PATTERN)
    source_artifacts: tuple[ValidationArtifactIdentity, ...]
    reference_artifacts: tuple[ValidationArtifactIdentity, ...]
    template_versions: dict[str, str]
    identity_digest: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def _validate_identity_digest(self) -> Self:
        expected = validation_workflow_identity_digest(
            schema_version=self.schema_version,
            request_digest=self.request_digest,
            policy_digest=self.policy_digest,
            backend_digest=self.backend_digest,
            source_artifacts=self.source_artifacts,
            reference_artifacts=self.reference_artifacts,
            template_versions=self.template_versions,
        )
        if self.identity_digest != expected:
            raise ValueError(
                "identity_digest must authenticate the complete workflow identity"
            )
        return self


def validation_workflow_identity_digest(
    *,
    schema_version: str,
    request_digest: str,
    policy_digest: str,
    backend_digest: str,
    source_artifacts: tuple[ValidationArtifactIdentity, ...],
    reference_artifacts: tuple[ValidationArtifactIdentity, ...],
    template_versions: dict[str, str],
) -> str:
    """Return the canonical digest for one complete workflow identity envelope."""

    payload: dict[str, Any] = {
        "schema_version": schema_version,
        "request_digest": request_digest,
        "policy_digest": policy_digest,
        "backend_digest": backend_digest,
        "source_artifacts": [
            identity.model_dump(mode="json") for identity in source_artifacts
        ],
        "reference_artifacts": [
            identity.model_dump(mode="json") for identity in reference_artifacts
        ],
        "template_versions": template_versions,
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ValidationAcceptedTemplateResult(_FrozenModel):
    """One template result and the artifacts validated for future resume."""

    work_item_identity_digest: str = Field(pattern=_SHA256_PATTERN)
    attempt: int = Field(ge=1)
    result: ValidationTemplateResult
    result_path: str = Field(min_length=1)
    result_sha256: str = Field(pattern=_SHA256_PATTERN)
    evidence_artifacts: tuple[ValidationArtifactIdentity, ...] = ()
    accepted_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("accepted_at")
    @classmethod
    def _timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("accepted_at must include a timezone")
        return value


class ValidationWorkItemRecord(_FrozenModel):
    """Checkpoint record for one ordered template work item."""

    work_item_id: str = Field(min_length=1)
    template_name: str = Field(min_length=1)
    identity_digest: str = Field(pattern=_SHA256_PATTERN)
    state: ValidationWorkItemState = ValidationWorkItemState.PENDING
    attempts: int = Field(default=0, ge=0)
    accepted_result: ValidationAcceptedTemplateResult | None = None
    last_error: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None

    @model_validator(mode="after")
    def _validate_accepted_result(self) -> Self:
        if self.accepted_result is None:
            return self
        if self.state != ValidationWorkItemState.COMPLETED:
            raise ValueError("accepted_result requires completed state")
        if self.accepted_result.work_item_identity_digest != self.identity_digest:
            raise ValueError("accepted result identity must match its work item")
        if self.accepted_result.result.template_name != self.template_name:
            raise ValueError("accepted result template must match its work item")
        return self

    @field_validator("started_at", "finished_at")
    @classmethod
    def _timezone_aware(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("work item timestamps must include a timezone")
        return value


class ValidationWorkflowCheckpoint(_FrozenModel):
    """Atomic resume boundary for one immutable validation workflow plan."""

    schema_version: Literal["content-agent-workflows.validation-checkpoint.v1"] = (
        VALIDATION_WORKFLOW_CHECKPOINT_SCHEMA_VERSION
    )
    revision: int = Field(default=0, ge=0)
    workflow_identity: ValidationWorkflowIdentity
    plan_digest: str = Field(pattern=_SHA256_PATTERN)
    ordered_work_item_ids: tuple[str, ...]
    records: tuple[ValidationWorkItemRecord, ...]
    cancellation_requested: bool = False
    cancellation_reason: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def _validate_records(self) -> Self:
        record_ids = tuple(record.work_item_id for record in self.records)
        if record_ids != self.ordered_work_item_ids:
            raise ValueError(
                "checkpoint records must match ordered_work_item_ids exactly"
            )
        if len(record_ids) != len(set(record_ids)):
            raise ValueError("checkpoint work item IDs must be unique")
        if not self.cancellation_requested and self.cancellation_reason is not None:
            raise ValueError("cancellation_reason requires cancellation_requested=true")
        return self

    @field_validator("created_at", "updated_at")
    @classmethod
    def _timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("checkpoint timestamps must include a timezone")
        return value


class ValidationWorkflowRun(_FrozenModel):
    """Final artifact index returned by a workflow invocation."""

    status: ValidationWorkflowStatus
    output_dir: str
    request: ValidationRequest
    plan: ValidationPlan
    result: ValidationResult
    checkpoint: ValidationWorkflowCheckpoint
    request_path: str
    plan_path: str
    result_path: str
    checkpoint_path: str
    evidence_path: str
    final_summary_path: str
