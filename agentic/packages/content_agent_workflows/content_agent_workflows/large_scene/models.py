# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed run-state contracts for large-scene orchestration."""

from __future__ import annotations

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

LEGACY_LARGE_SCENE_RUN_SCHEMA_VERSION: Final = (
    "content-agent-workflows.large-scene-run.v1"
)
LARGE_SCENE_RUN_SCHEMA_VERSION: Final = "content-agent-workflows.large-scene-run.v2"
HANDOFF_VALIDATION_SCHEMA_VERSION: Final = (
    "content-agent-workflows.large-scene-handoff-validation.v1"
)

PhaseName = Literal["decomposition", "asset_task_processing", "collection"]
SceneBackend = Literal["usd-cli"]
PhaseStatus = Literal[
    "pending", "ready", "running", "completed", "failed", "invalidated"
]
PHASE_ORDER: tuple[PhaseName, ...] = (
    "decomposition",
    "asset_task_processing",
    "collection",
)


class PhaseState(BaseModel):
    """Current durable state for one workflow phase."""

    model_config = ConfigDict(extra="forbid")

    status: PhaseStatus = "pending"
    result_path: str | None = None
    input_digest: str | None = None
    output_digest: str | None = None
    error: str | None = None


class PhaseTransition(BaseModel):
    """Append-only audit record for one phase-state transition."""

    model_config = ConfigDict(extra="forbid")

    timestamp: str
    phase: PhaseName
    from_status: PhaseStatus
    to_status: PhaseStatus
    reason: str
    actor: str
    input_digest: str | None = None
    result_path: str | None = None
    output_digest: str | None = None


class LargeSceneRun(BaseModel):
    """Durable coordinator state shared across agent sessions."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[
        LEGACY_LARGE_SCENE_RUN_SCHEMA_VERSION,
        LARGE_SCENE_RUN_SCHEMA_VERSION,
    ] = LARGE_SCENE_RUN_SCHEMA_VERSION
    revision: int = Field(default=0, ge=0)
    run_id: str = Field(min_length=1)
    source_scene: str
    scene_backend: SceneBackend
    additional_instructions: str | None = None
    request_artifact_paths: list[str] = Field(default_factory=list)
    requested_tasks: list[str] = Field(default_factory=list)
    source_input_digest: str
    current_phase: PhaseName | None = "decomposition"
    phases: dict[PhaseName, PhaseState]
    transitions: list[PhaseTransition] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def restore_legacy_scene_backend(cls, value: object) -> object:
        """Migrate pre-backend run state to the canonical usd-cli route."""

        if not isinstance(value, dict):
            return value
        if (
            value.get("schema_version") == LEGACY_LARGE_SCENE_RUN_SCHEMA_VERSION
            and "scene_backend" not in value
        ):
            return {**value, "scene_backend": "usd-cli"}
        return value

    @field_validator("additional_instructions", mode="before")
    @classmethod
    def normalize_additional_instructions(cls, value: object) -> object:
        if value is None:
            return None
        if not isinstance(value, str):
            return value
        normalized = value.strip()
        return normalized or None

    @model_validator(mode="after")
    def validate_phase_set(self) -> LargeSceneRun:
        if set(self.phases) != set(PHASE_ORDER):
            raise ValueError(f"phases must contain exactly {list(PHASE_ORDER)}")
        if len(self.requested_tasks) != len(set(self.requested_tasks)):
            raise ValueError("requested_tasks must be unique")
        return self


class HandoffValidationReport(BaseModel):
    """Machine-readable result of a phase handoff gate."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[HANDOFF_VALIDATION_SCHEMA_VERSION] = (
        HANDOFF_VALIDATION_SCHEMA_VERSION
    )
    phase: PhaseName
    valid: bool
    result_path: str
    input_digest: str | None = None
    output_digest: str | None = None
    computed_output_digest: str | None = None
    artifact_count: int = 0
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
