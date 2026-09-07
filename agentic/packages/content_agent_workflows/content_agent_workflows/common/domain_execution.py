# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed ownership and identity for standalone or composed domain execution."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

DOMAIN_EXECUTION_CONTEXT_METADATA_KEY: Final = (
    "content_agent_workflows.domain_execution_context"
)
DOMAIN_EXECUTION_CONTEXT_SCHEMA_VERSION: Final = (
    "content-agent-workflows.domain-execution-context.v1"
)
DOMAIN_EXECUTION_CONTEXT_SCHEMA_VERSION_V2: Final = (
    "content-agent-workflows.domain-execution-context.v2"
)
DOMAIN_EXECUTION_CONTEXT_SCHEMA_VERSION_V3: Final = (
    "content-agent-workflows.domain-execution-context.v3"
)

DomainName = Literal["texture", "articulation", "validation"]
DomainExecutionMode = Literal["standalone", "embedded"]
ReasoningLoopOwner = Literal[
    "compatibility_pipeline",
    "domain_child_agent",
    "asset_coordinator",
]


class ExecutionArtifactBinding(BaseModel):
    """Portable immutable artifact identity used at the workflow boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)


class EmbeddedStageBinding(BaseModel):
    """Exact outer coordinator attempt that owns one embedded domain run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    outer_run_id: str = Field(min_length=1)
    outer_request: ExecutionArtifactBinding
    stage: DomainName
    stage_attempt: int = Field(ge=1)
    coordinator_plan: ExecutionArtifactBinding
    input_asset: ExecutionArtifactBinding
    domain_run_root: str = Field(min_length=1)


class DomainExecutionContext(BaseModel):
    """Lifecycle boundary shared by embedded-capable domain requests."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[
        "content-agent-workflows.domain-execution-context.v1",
        "content-agent-workflows.domain-execution-context.v2",
        "content-agent-workflows.domain-execution-context.v3",
    ] = DOMAIN_EXECUTION_CONTEXT_SCHEMA_VERSION
    domain: DomainName
    mode: DomainExecutionMode
    reasoning_loop_owner: ReasoningLoopOwner
    embedded_stage: EmbeddedStageBinding | None = None

    @model_validator(mode="after")
    def validate_ownership(self) -> Self:
        if self.domain == "validation" and (
            self.schema_version != DOMAIN_EXECUTION_CONTEXT_SCHEMA_VERSION_V2
        ):
            raise ValueError(
                "validation execution requires domain execution context v2"
            )
        if self.domain == "texture" and (
            self.schema_version != DOMAIN_EXECUTION_CONTEXT_SCHEMA_VERSION
        ):
            raise ValueError("texture execution requires domain execution context v1")
        if self.domain == "articulation" and self.schema_version not in {
            DOMAIN_EXECUTION_CONTEXT_SCHEMA_VERSION,
            DOMAIN_EXECUTION_CONTEXT_SCHEMA_VERSION_V3,
        }:
            raise ValueError(
                "articulation execution requires domain execution context v1 or v3"
            )
        if (
            self.domain == "articulation"
            and (self.schema_version == DOMAIN_EXECUTION_CONTEXT_SCHEMA_VERSION_V3)
            and (
                self.mode != "standalone"
                or self.reasoning_loop_owner != "asset_coordinator"
            )
        ):
            raise ValueError(
                "articulation execution context v3 requires standalone "
                "asset_coordinator ownership"
            )
        if self.mode == "embedded":
            if self.reasoning_loop_owner != "asset_coordinator":
                raise ValueError(
                    "embedded execution requires asset_coordinator reasoning ownership"
                )
            if self.embedded_stage is None:
                raise ValueError("embedded execution requires an outer stage binding")
            if self.embedded_stage.stage != self.domain:
                raise ValueError("embedded stage must match the requested domain")
        else:
            if self.reasoning_loop_owner == "asset_coordinator":
                if not (
                    self.domain == "articulation"
                    and self.schema_version
                    == DOMAIN_EXECUTION_CONTEXT_SCHEMA_VERSION_V3
                ):
                    raise ValueError(
                        "standalone execution cannot claim asset_coordinator ownership"
                    )
            if self.embedded_stage is not None:
                raise ValueError(
                    "standalone execution cannot carry an outer stage binding"
                )
        return self


def domain_execution_context_from_metadata(
    metadata: Mapping[str, Any],
    *,
    expected_domain: DomainName,
) -> DomainExecutionContext | None:
    """Parse and domain-check the reserved request metadata when present."""

    raw = metadata.get(DOMAIN_EXECUTION_CONTEXT_METADATA_KEY)
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("Domain execution context metadata must be an object")
    context = DomainExecutionContext.model_validate(dict(raw))
    if context.domain != expected_domain:
        raise ValueError("Domain execution context does not match the workflow request")
    return context


def metadata_with_domain_execution_context(
    metadata: Mapping[str, Any],
    context: DomainExecutionContext,
) -> dict[str, Any]:
    """Return metadata containing one canonical typed execution context."""

    updated = dict(metadata)
    existing = updated.get(DOMAIN_EXECUTION_CONTEXT_METADATA_KEY)
    payload = context.model_dump(mode="json")
    if existing is not None and existing != payload:
        raise ValueError("Domain execution context metadata already differs")
    updated[DOMAIN_EXECUTION_CONTEXT_METADATA_KEY] = payload
    return updated


__all__ = [
    "DOMAIN_EXECUTION_CONTEXT_METADATA_KEY",
    "DOMAIN_EXECUTION_CONTEXT_SCHEMA_VERSION",
    "DOMAIN_EXECUTION_CONTEXT_SCHEMA_VERSION_V2",
    "DOMAIN_EXECUTION_CONTEXT_SCHEMA_VERSION_V3",
    "DomainExecutionContext",
    "DomainExecutionMode",
    "DomainName",
    "EmbeddedStageBinding",
    "ExecutionArtifactBinding",
    "ReasoningLoopOwner",
    "domain_execution_context_from_metadata",
    "metadata_with_domain_execution_context",
]
