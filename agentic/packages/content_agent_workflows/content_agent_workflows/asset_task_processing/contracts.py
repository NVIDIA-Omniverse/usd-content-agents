# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed durable artifacts for Workflow 2."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

TASK_CATALOG_SCHEMA_VERSION = "content-agent-workflows.task-catalog.v1"
ASSET_TASK_INVENTORY_SCHEMA_VERSION = "content-agent-workflows.asset-task-inventory.v1"
ASSET_TASK_RESULT_SCHEMA_VERSION = "content-agent-workflows.asset-task-result.v1"
ASSET_TASK_RESULTS_INDEX_SCHEMA_VERSION = (
    "content-agent-workflows.asset-task-results-index.v1"
)
ASSET_TASK_RUN_STATE_SCHEMA_VERSION = "content-agent-workflows.asset-task-run-state.v1"
DECISION_LEDGER_ENTRY_SCHEMA_VERSION = (
    "content-agent-workflows.decision-ledger-entry.v1"
)
AGENT_PLAN_POINTER_SCHEMA_VERSION = "content-agent-workflows.agent-plan-pointer.v1"
PROCESSING_PHASE_RESULT_SCHEMA_VERSION = (
    "content-agent-workflows.processing-phase-result.v1"
)

WorkItemStatus = Literal[
    "planned",
    "running",
    "produced",
    "preview_applied",
    "validated",
    "completed",
    "deferred",
    "failed",
    "waived",
]


class TaskSpec(BaseModel):
    """One domain operation requested over a decomposition view."""

    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(min_length=1)
    domain: str = Field(min_length=1)
    skill: str = Field(min_length=1)
    manifest_id: str = Field(min_length=1)
    request_path: str
    result_schema: str = ASSET_TASK_RESULT_SCHEMA_VERSION
    validator: str = Field(min_length=1)
    collector: str = Field(min_length=1)
    depends_on: list[str] = Field(default_factory=list)
    required: bool = True
    resource_hints: dict[str, Any] = Field(default_factory=dict)


class TaskCatalog(BaseModel):
    """Immutable task specifications for one processing run."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[TASK_CATALOG_SCHEMA_VERSION] = TASK_CATALOG_SCHEMA_VERSION
    tasks: list[TaskSpec]

    @model_validator(mode="after")
    def validate_tasks(self) -> TaskCatalog:
        task_ids = [task.task_id for task in self.tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("task_id values must be unique")
        known = set(task_ids)
        for task in self.tasks:
            missing = set(task.depends_on) - known
            if missing:
                raise ValueError(
                    f"Task {task.task_id} has unknown dependencies: {sorted(missing)}"
                )
            if task.task_id in task.depends_on:
                raise ValueError(f"Task {task.task_id} cannot depend on itself")

        visiting: set[str] = set()
        visited: set[str] = set()
        dependencies = {task.task_id: task.depends_on for task in self.tasks}

        def visit(task_id: str) -> None:
            if task_id in visiting:
                raise ValueError("task dependencies must be acyclic")
            if task_id in visited:
                return
            visiting.add(task_id)
            for dependency in dependencies[task_id]:
                visit(dependency)
            visiting.remove(task_id)
            visited.add(task_id)

        for task_id in task_ids:
            visit(task_id)
        return self


class AssetTaskWorkItem(BaseModel):
    """Mechanical unit of eligible asset-task work."""

    model_config = ConfigDict(extra="forbid")

    work_item_id: str = Field(min_length=1)
    manifest_id: str = Field(min_length=1)
    asset_id: str = Field(min_length=1)
    asset_label: str = ""
    task_id: str = Field(min_length=1)
    required: bool = True
    original_root_path: str
    working_usd_path: str | None = None
    working_root_path: str | None = None
    source_path_prefixes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_qualified_id(self) -> AssetTaskWorkItem:
        expected = f"{self.task_id}:{self.manifest_id}:{self.asset_id}"
        if self.work_item_id != expected:
            raise ValueError(f"work_item_id must be {expected!r}")
        return self


class AssetTaskInventory(BaseModel):
    """Immutable expansion of tasks over processable representatives."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[ASSET_TASK_INVENTORY_SCHEMA_VERSION] = (
        ASSET_TASK_INVENTORY_SCHEMA_VERSION
    )
    input_digest: str
    task_request_digests: dict[str, str] = Field(default_factory=dict)
    work_items: list[AssetTaskWorkItem]

    @model_validator(mode="after")
    def validate_unique_work_items(self) -> AssetTaskInventory:
        identities = [item.work_item_id for item in self.work_items]
        if len(identities) != len(set(identities)):
            raise ValueError("work_item_id values must be unique")
        return self


class AcceptedWaiver(BaseModel):
    """Explicit approval to omit one otherwise required work item."""

    model_config = ConfigDict(extra="forbid")

    waiver_id: str = Field(min_length=1)
    work_item_id: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    accepted_by: str = Field(min_length=1)
    accepted_at: str


class AssetTaskWorkItemState(BaseModel):
    """Mutable execution state for one immutable inventory item."""

    model_config = ConfigDict(extra="forbid")

    work_item_id: str
    status: WorkItemStatus = "planned"
    attempt_count: int = Field(default=0, ge=0)
    started_at: str | None = None
    completed_at: str | None = None
    result_path: str | None = None
    validation_path: str | None = None
    waiver_id: str | None = None
    error: str | None = None


class AssetTaskStateTransition(BaseModel):
    """Append-only work-item transition record."""

    model_config = ConfigDict(extra="forbid")

    timestamp: str
    work_item_id: str
    from_status: WorkItemStatus
    to_status: WorkItemStatus
    actor: str
    reason: str
    attempt_count: int = Field(ge=0)
    result_path: str | None = None
    validation_path: str | None = None


class AssetTaskRunState(BaseModel):
    """Mutable, locked Workflow 2 execution state."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[ASSET_TASK_RUN_STATE_SCHEMA_VERSION] = (
        ASSET_TASK_RUN_STATE_SCHEMA_VERSION
    )
    revision: int = Field(default=0, ge=0)
    input_digest: str
    inventory_path: str
    task_catalog_path: str
    manifest_catalog_path: str
    task_request_digests: dict[str, str] = Field(default_factory=dict)
    work_items: list[AssetTaskWorkItemState]
    accepted_waivers: list[AcceptedWaiver] = Field(default_factory=list)
    transitions: list[AssetTaskStateTransition] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unique_work_items(self) -> AssetTaskRunState:
        identities = [item.work_item_id for item in self.work_items]
        if len(identities) != len(set(identities)):
            raise ValueError("run-state work_item_id values must be unique")
        waiver_ids = [waiver.waiver_id for waiver in self.accepted_waivers]
        if len(waiver_ids) != len(set(waiver_ids)):
            raise ValueError("run-state waiver_id values must be unique")
        return self


class ResultMapping(BaseModel):
    """Canonical original-topology mapping for one task result."""

    model_config = ConfigDict(extra="forbid")

    path_space: Literal["original"] = "original"
    unresolved_paths: list[str] = Field(default_factory=list)


class ResultProvenance(BaseModel):
    """Agent plan and prior-result provenance for one work item."""

    model_config = ConfigDict(extra="forbid")

    agent_plan_revision: int = Field(ge=1)
    task_request_digest: str | None = None
    informed_by_results: list[str] = Field(default_factory=list)


class AssetTaskResult(BaseModel):
    """Generic independently validated result envelope."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[ASSET_TASK_RESULT_SCHEMA_VERSION] = (
        ASSET_TASK_RESULT_SCHEMA_VERSION
    )
    task_id: str
    domain: str
    manifest_id: str
    asset_id: str
    status: Literal["completed"] = "completed"
    original_root_path: str
    working_usd_path: str | None = None
    domain_outputs: dict[str, str] = Field(default_factory=dict)
    mapping: ResultMapping = Field(default_factory=ResultMapping)
    provenance: ResultProvenance
    warnings: list[str] = Field(default_factory=list)

    @property
    def work_item_id(self) -> str:
        return f"{self.task_id}:{self.manifest_id}:{self.asset_id}"


class ResultIndexEntry(BaseModel):
    """Index entry routing a qualified work item to its result."""

    model_config = ConfigDict(extra="forbid")

    work_item_id: str
    status: Literal["completed", "waived"]
    result_path: str | None = None
    validation_path: str | None = None


class AssetTaskResultsIndex(BaseModel):
    """Aggregate index of completed and waived work items."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[ASSET_TASK_RESULTS_INDEX_SCHEMA_VERSION] = (
        ASSET_TASK_RESULTS_INDEX_SCHEMA_VERSION
    )
    entries: list[ResultIndexEntry]

    @model_validator(mode="after")
    def validate_unique_entries(self) -> AssetTaskResultsIndex:
        identities = [entry.work_item_id for entry in self.entries]
        if len(identities) != len(set(identities)):
            raise ValueError("result-index work_item_id values must be unique")
        return self


class DecisionLedgerEntry(BaseModel):
    """Append-only durable memory for one committed work item."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[DECISION_LEDGER_ENTRY_SCHEMA_VERSION] = (
        DECISION_LEDGER_ENTRY_SCHEMA_VERSION
    )
    work_item_id: str
    domain: str
    task_id: str
    evidence_summary: str
    artifact_paths: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    validation_status: Literal["passed", "failed"]
    agent_plan_revision: int = Field(ge=1)
    task_request_digest: str | None = None
    informed_by_results: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)


class AgentPlanPointer(BaseModel):
    """Pointer to immutable agent-authored processing plan revisions."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[AGENT_PLAN_POINTER_SCHEMA_VERSION] = (
        AGENT_PLAN_POINTER_SCHEMA_VERSION
    )
    current_revision: int = Field(ge=1)
    current_plan_path: str
    revision_paths: list[str]

    @model_validator(mode="after")
    def validate_current_revision(self) -> AgentPlanPointer:
        if len(self.revision_paths) != len(set(self.revision_paths)):
            raise ValueError("agent plan revision_paths must be unique")
        if self.current_plan_path not in self.revision_paths:
            raise ValueError("current_plan_path must appear in revision_paths")
        if len(self.revision_paths) < self.current_revision:
            raise ValueError("current_revision exceeds available plan revisions")
        return self


class ProcessingPhaseResult(BaseModel):
    """Sealed Workflow 2 output consumed by the umbrella coordinator."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[PROCESSING_PHASE_RESULT_SCHEMA_VERSION] = (
        PROCESSING_PHASE_RESULT_SCHEMA_VERSION
    )
    phase: Literal["asset_task_processing"] = "asset_task_processing"
    success: bool
    input_digest: str
    task_catalog_path: str
    manifest_catalog_path: str
    asset_task_inventory_path: str
    work_item_state_path: str
    agent_plan_pointer_path: str
    decision_ledger_path: str
    results_index_path: str
    task_request_digests: dict[str, str] = Field(default_factory=dict)
    required_work_item_count: int = Field(ge=0)
    completed_required_count: int = Field(ge=0)
    optional_work_item_count: int = Field(default=0, ge=0)
    completed_optional_count: int = Field(default=0, ge=0)
    accepted_waivers: list[AcceptedWaiver] = Field(default_factory=list)
    artifact_paths: list[str] = Field(default_factory=list)
    output_digest: str | None = None
    completion_policy_satisfied: bool = False
    unresolved_issues: list[str] = Field(default_factory=list)
    error: str | None = None

    @model_validator(mode="after")
    def validate_completion_counts(self) -> ProcessingPhaseResult:
        if self.completed_required_count > self.required_work_item_count:
            raise ValueError(
                "completed_required_count cannot exceed required_work_item_count"
            )
        if self.completed_optional_count > self.optional_work_item_count:
            raise ValueError(
                "completed_optional_count cannot exceed optional_work_item_count"
            )
        return self
