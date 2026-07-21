# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Immutable records for bounded Texture Plan execution."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

TEXTURE_EXECUTION_CHECKPOINT_SCHEMA_VERSION: Literal[
    "texture-agent-execution-checkpoint.v1"
] = "texture-agent-execution-checkpoint.v1"
TEXTURE_EXECUTION_SUMMARY_SCHEMA_VERSION: Literal[
    "texture-agent-execution-summary.v1"
] = "texture-agent-execution-summary.v1"


class TextureUnitExecutionState(StrEnum):
    """Latest attempt state for one immutable plan unit."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TextureExecutionStatus(StrEnum):
    """Terminal status of one bounded executor invocation."""

    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TextureArtifactRef(_FrozenModel):
    """One durable artifact accepted for a selected texture unit."""

    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
    uri: str = Field(min_length=1)
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @field_validator("name", "uri")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("artifact fields must be non-empty")
        return normalized


class TextureUnitExecutionResult(_FrozenModel):
    """Accepted output of exactly one selected plan unit."""

    unit_id: str = Field(pattern=r"^tu_[0-9a-f]{20}$")
    artifacts: tuple[TextureArtifactRef, ...] = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _unique_artifact_names(self) -> Self:
        names = [artifact.name for artifact in self.artifacts]
        if len(names) != len(set(names)):
            raise ValueError("artifact names must be unique within a unit result")
        return self


class TextureUnitExecutionRecord(_FrozenModel):
    """Checkpoint record for a selected unit.

    ``accepted_result`` records the latest accepted output for a unit. It is
    cleared before retry/regeneration attempts so failed requested work cannot
    be summarized as accepted from stale artifacts.
    """

    unit_id: str = Field(pattern=r"^tu_[0-9a-f]{20}$")
    state: TextureUnitExecutionState = TextureUnitExecutionState.PENDING
    attempts: int = Field(default=0, ge=0)
    cache_hit_count: int = Field(default=0, ge=0)
    accepted_result: TextureUnitExecutionResult | None = None
    last_error: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None

    @model_validator(mode="after")
    def _validate_result_identity(self) -> Self:
        if (
            self.accepted_result is not None
            and self.accepted_result.unit_id != self.unit_id
        ):
            raise ValueError("accepted result unit_id must match its checkpoint record")
        return self

    @field_validator("started_at", "finished_at")
    @classmethod
    def _timezone_aware(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("execution timestamps must include a timezone")
        return value


class TextureExecutionCheckpoint(_FrozenModel):
    """Atomic resume boundary for one immutable Texture Plan."""

    schema_version: Literal["texture-agent-execution-checkpoint.v1"] = (
        TEXTURE_EXECUTION_CHECKPOINT_SCHEMA_VERSION
    )
    plan_schema_version: str
    plan_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    selected_unit_ids: tuple[str, ...]
    records: tuple[TextureUnitExecutionRecord, ...]
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def _validate_records(self) -> Self:
        record_ids = tuple(record.unit_id for record in self.records)
        if record_ids != self.selected_unit_ids:
            raise ValueError(
                "checkpoint records must exactly match selected_unit_ids in plan order"
            )
        if len(record_ids) != len(set(record_ids)):
            raise ValueError("checkpoint selected_unit_ids must be unique")
        return self

    @field_validator("created_at", "updated_at")
    @classmethod
    def _timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("checkpoint timestamps must include a timezone")
        return value


class TextureExecutionSummary(_FrozenModel):
    """Result of a bounded execute, resume, or regeneration invocation."""

    schema_version: Literal["texture-agent-execution-summary.v1"] = (
        TEXTURE_EXECUTION_SUMMARY_SCHEMA_VERSION
    )
    status: TextureExecutionStatus
    plan_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    requested_unit_ids: tuple[str, ...]
    executed_unit_ids: tuple[str, ...]
    cache_hit_unit_ids: tuple[str, ...]
    accepted_unit_ids: tuple[str, ...]
    failed_unit_ids: tuple[str, ...]
    cancelled_unit_ids: tuple[str, ...]
    remaining_unit_ids: tuple[str, ...]
    records: tuple[TextureUnitExecutionRecord, ...]
