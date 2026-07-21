# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared contracts for task-first Content Agents skills.

This module intentionally stays domain-neutral. Application packages such as
Material Agent or CAD Agent can specialize these contracts, but the base types
should not encode USD, CAD, rendering, or material-specific concepts.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field

IssueSeverity = Literal["info", "warning", "error"]
HarnessDecisionKind = Literal["accept", "continue", "blocked", "ask_user", "stop"]
HarnessRunStatus = Literal["succeeded", "failed", "canceled", "blocked"]


class TaskArtifact(BaseModel):
    """A file or directory produced by a task."""

    path: str
    kind: str
    label: str | None = None


class TaskIssue(BaseModel):
    """A non-fatal task issue intended for harness inspection."""

    severity: IssueSeverity
    message: str
    detail: dict[str, Any] = Field(default_factory=dict)


class HarnessArtifact(TaskArtifact):
    """Domain-neutral artifact reference for harness run results."""

    metadata: dict[str, Any] = Field(default_factory=dict)


class HarnessIssue(TaskIssue):
    """Structured issue surfaced to the outer harness for judgment."""

    code: str | None = None
    source: str | None = None
    blocking: bool = False


class HarnessDecision(BaseModel):
    """Harness-readable decision artifact produced by deterministic tasks."""

    decision: HarnessDecisionKind
    reason: str
    issue_codes: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)
    question: str | None = None


class HarnessRunResult(BaseModel):
    """Common result envelope for one bounded recipe execution."""

    run_id: str
    recipe_id: str
    output_dir: Path
    status: HarnessRunStatus
    summary: str
    artifacts: list[HarnessArtifact] = Field(default_factory=list)
    issues: list[HarnessIssue] = Field(default_factory=list)
    decision: HarnessDecision | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class HarnessRefinementAction(BaseModel):
    """One harness-authored action for a deterministic executor to perform."""

    action: str
    rationale: str
    target_ids: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)
    evidence_paths: list[str] = Field(default_factory=list)


class HarnessRefinementPlan(BaseModel):
    """Domain-neutral refinement plan written by an outer coding harness."""

    schema_version: Literal["agentic-harness-refinement-plan/v1"] = (
        "agentic-harness-refinement-plan/v1"
    )
    goal: str
    actions: list[HarnessRefinementAction] = Field(default_factory=list)
    evidence_paths: list[str] = Field(default_factory=list)
    observations: list[str] = Field(default_factory=list)
    stop_conditions: list[str] = Field(default_factory=list)
    created_by: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass
class RecipeContext:
    """Runtime context passed to a deterministic harness recipe."""

    run_id: str
    output_dir: Path
    metadata: dict[str, Any] = field(default_factory=dict)
    event_listener: Any | None = None
    cancel_checker: Callable[[], bool] | None = None
    _cancel_event_emitted: bool = field(default=False, init=False, repr=False)

    def emit(
        self,
        event_type: str,
        data: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Emit a structured event when an event listener is available."""
        if self.event_listener is None:
            return
        self.event_listener.event(event_type, data or {}, **kwargs)

    def cancel_requested(self) -> bool:
        """Return whether cooperative cancellation has been requested."""
        return bool(self.cancel_checker and self.cancel_checker())

    def emit_cancelled(self, data: dict[str, Any] | None = None) -> None:
        """Emit a cancellation lifecycle event once for this context."""
        if self._cancel_event_emitted:
            return
        payload = {"run_id": self.run_id, "message": "recipe cancellation requested"}
        if data:
            payload.update(data)
        self.emit("recipe.cancelled", payload)
        self._cancel_event_emitted = True

    def raise_if_cancelled(self) -> None:
        """Raise ``CancelledError`` when the caller requests cancellation."""
        if self.cancel_requested():
            self.emit_cancelled()
            raise asyncio.CancelledError("recipe cancellation requested")


type TaskCallable[InputT: BaseModel, OutputT: BaseModel] = Callable[
    [InputT], Awaitable[OutputT]
]


type RecipeCallable[InputT: BaseModel, OutputT: BaseModel] = Callable[
    [InputT, RecipeContext], OutputT | Awaitable[OutputT]
]


class TaskSkillSpec[InputT: BaseModel, OutputT: BaseModel](BaseModel):
    """Thin metadata connecting a harness-visible skill id to one task callable."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    domain: str
    name: str
    description: str
    when_to_use: str
    task: TaskCallable[InputT, OutputT]
    input_model: type[InputT]
    output_model: type[OutputT]
    examples: list[dict[str, Any]] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

    def input_schema(self) -> dict[str, Any]:
        """Return the JSON schema for the task input model."""
        return cast(dict[str, Any], self.input_model.model_json_schema())

    def output_schema(self) -> dict[str, Any]:
        """Return the JSON schema for the task output model."""
        return cast(dict[str, Any], self.output_model.model_json_schema())


class RecipeSpec[InputT: BaseModel, OutputT: BaseModel](BaseModel):
    """Metadata and callable for one bounded deterministic recipe."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    domain: str
    name: str
    description: str
    when_to_use: str
    recipe: RecipeCallable[InputT, OutputT]
    input_model: type[InputT]
    output_model: type[OutputT]
    examples: list[dict[str, Any]] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

    def input_schema(self) -> dict[str, Any]:
        """Return the JSON schema for the recipe input model."""
        return cast(dict[str, Any], self.input_model.model_json_schema())

    def output_schema(self) -> dict[str, Any]:
        """Return the JSON schema for the recipe output model."""
        return cast(dict[str, Any], self.output_model.model_json_schema())


def artifact(path: str, kind: str, label: str | None = None) -> TaskArtifact:
    """Create a task artifact with a small, consistent call site."""
    return TaskArtifact(path=path, kind=kind, label=label)
