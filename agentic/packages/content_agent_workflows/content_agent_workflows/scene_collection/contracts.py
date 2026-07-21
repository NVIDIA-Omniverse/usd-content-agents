# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed durable artifacts for Workflow 3."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DOMAIN_COLLECTION_RESULT_SCHEMA_VERSION = (
    "content-agent-workflows.domain-collection-result.v1"
)
COLLECTION_PHASE_RESULT_SCHEMA_VERSION = (
    "content-agent-workflows.collection-phase-result.v1"
)
COLLECTION_REQUEST_SCHEMA_VERSION = "content-agent-workflows.collection-request.v1"
COLLECTION_INPUT_INDEX_SCHEMA_VERSION = (
    "content-agent-workflows.collection-input-index.v1"
)
PROJECTED_MATERIAL_BINDING_SCHEMA_VERSION = (
    "content-agent-workflows.projected-material-binding.v1"
)


class CollectionRequest(BaseModel):
    """Concrete Workflow 3 input routed from a sealed processing phase."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[COLLECTION_REQUEST_SCHEMA_VERSION] = (
        COLLECTION_REQUEST_SCHEMA_VERSION
    )
    source_scene: str
    processing_result_path: str
    manifest_catalog_path: str
    task_catalog_path: str
    asset_task_inventory_path: str
    results_index_path: str
    output_dir: str
    input_digest: str
    requested_domains: list[str] = Field(default_factory=list)
    material_library_yaml: str | None = None


class CollectionInputArtifact(BaseModel):
    """One immutable Workflow 3 input and its content identity."""

    model_config = ConfigDict(extra="forbid")

    role: str
    path: str
    sha256: str


class CollectionInputIndex(BaseModel):
    """Stable preflight index used to resume domain collection."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[COLLECTION_INPUT_INDEX_SCHEMA_VERSION] = (
        COLLECTION_INPUT_INDEX_SCHEMA_VERSION
    )
    input_digest: str
    source_scene: str
    requested_domains: list[str]
    task_request_digests: dict[str, str] = Field(default_factory=dict)
    required_work_item_count: int = Field(ge=0)
    completed_work_item_count: int = Field(ge=0)
    artifacts: list[CollectionInputArtifact]


class ProjectedMaterialBinding(BaseModel):
    """One task-local material decision projected to original topology."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[PROJECTED_MATERIAL_BINDING_SCHEMA_VERSION] = (
        PROJECTED_MATERIAL_BINDING_SCHEMA_VERSION
    )
    work_item_id: str
    representative_asset_id: str
    member_asset_id: str
    representative_root_path: str
    member_root_path: str
    decision_target_path: str
    source_candidate_path: str
    instance_target_path: str
    authoring_target_path: str
    material_name: str
    propagation_basis: Literal["explicit", "instance_group"]
    mapping_method: Literal[
        "exact_relative",
        "stable_suffix",
        "ordered_structural",
        "equivalent_material_id",
    ]
    evidence_work_item_ids: list[str] = Field(default_factory=list)


class DomainCollectionResult(BaseModel):
    """Independently inspectable result from one domain collector."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[DOMAIN_COLLECTION_RESULT_SCHEMA_VERSION] = (
        DOMAIN_COLLECTION_RESULT_SCHEMA_VERSION
    )
    domain: str = Field(min_length=1)
    status: Literal["completed", "partial", "failed"]
    required: bool = True
    output_paths: list[str] = Field(default_factory=list)
    collection_report_path: str
    harmonization_report_path: str
    validation_report_path: str
    validation_passed: bool = False
    unresolved_issues: list[str] = Field(default_factory=list)
    artifact_paths: list[str] = Field(default_factory=list)
    output_digest: str | None = None


class CollectionPhaseResult(BaseModel):
    """Sealed Workflow 3 output consumed by the umbrella coordinator."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[COLLECTION_PHASE_RESULT_SCHEMA_VERSION] = (
        COLLECTION_PHASE_RESULT_SCHEMA_VERSION
    )
    phase: Literal["collection"] = "collection"
    success: bool
    input_digest: str
    domain_result_paths: list[str] = Field(default_factory=list)
    required_domain_count: int = Field(ge=0)
    completed_required_domain_count: int = Field(ge=0)
    composition_path: str | None = None
    topology_report_path: str
    cross_domain_validation_path: str | None = None
    final_output_paths: list[str] = Field(default_factory=list)
    artifact_paths: list[str] = Field(default_factory=list)
    output_digest: str | None = None
    completion_policy_satisfied: bool = False
    unresolved_issues: list[str] = Field(default_factory=list)
    error: str | None = None
