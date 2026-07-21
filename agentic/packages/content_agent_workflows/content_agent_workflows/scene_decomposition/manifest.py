# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic scene decomposition contracts for agentic asset workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from content_agent_workflows.common.artifacts import atomic_write_json

SCENE_MANIFEST_SCHEMA_VERSION = (
    "content-agent-workflows.scene-decomposition-manifest.v1"
)
SCENE_DECOMPOSITION_RESULT_SCHEMA_VERSION = (
    "content-agent-workflows.scene-decomposition-result.v1"
)
MANIFEST_CATALOG_SCHEMA_VERSION = "content-agent-workflows.manifest-catalog.v1"
DECOMPOSITION_PHASE_RESULT_SCHEMA_VERSION = (
    "content-agent-workflows.decomposition-phase-result.v1"
)

SceneGroupType = Literal["instance", "payload", "prototype"]


class StageMetadata(BaseModel):
    """Stage-level metadata preserved for downstream domain collectors."""

    model_config = ConfigDict(extra="forbid")

    default_prim_path: str | None = None
    up_axis: str | None = None
    meters_per_unit: float | None = None


class DecompositionPolicy(BaseModel):
    """Policy used to create a scene decomposition manifest."""

    model_config = ConfigDict(extra="forbid")

    method: str = "material_agent_scene_adapter"
    refinement_mode: Literal["none", "agent", "external_llm"] = "none"
    decomposition_intent: str = "generic_processing"
    preserve_instances: bool = True
    root_prim_path: str | None = None
    include_paths: list[str] = Field(default_factory=list)
    exclude_paths: list[str] = Field(default_factory=list)
    min_mesh_count: int = 0
    exclude_invisible_assets: bool = False
    detect_structural_duplicates: bool = False
    detect_payload_groups: bool = True
    detect_native_prototypes: bool = True
    extract_large_payload_representatives: bool = False
    extract_assets: bool = False
    flatten_extracts: bool = True
    skip_geometry: bool = False
    llm_refinement_enabled: bool = False


class ArtifactReference(BaseModel):
    """A file artifact associated with decomposition."""

    model_config = ConfigDict(extra="forbid")

    path: str
    kind: str
    description: str = ""


class SceneInstanceGroup(BaseModel):
    """Repeated or instanced scene members sharing source geometry."""

    model_config = ConfigDict(extra="forbid")

    group_id: str
    label: str
    group_type: SceneGroupType = "instance"
    source_file: str | list[str] | None = None
    instance_count: int = 0
    member_paths: list[str] = Field(default_factory=list)
    representative_asset_id: str | None = None


class ScenePayloadGroup(BaseModel):
    """Payload-backed scene group."""

    model_config = ConfigDict(extra="forbid")

    group_id: str
    label: str
    group_type: SceneGroupType = "payload"
    payload_file: str
    instance_count: int = 0
    instance_paths: list[str] = Field(default_factory=list)
    depth: int = 0
    child_payload_files: list[str] = Field(default_factory=list)
    parent_payload_files: list[str] = Field(default_factory=list)
    representative_usd_path: str | None = None
    modified_input_path: str | None = None
    output_usd_path: str | None = None
    status: str = "pending"


class ScenePrototypeGroup(BaseModel):
    """Native USD prototype group represented as a reusable working asset."""

    model_config = ConfigDict(extra="forbid")

    group_id: str
    label: str
    group_type: SceneGroupType = "prototype"
    prototype_path: str | None = None
    representative_usd_path: str | None = None
    instance_count: int = 0
    instance_paths: list[str] = Field(default_factory=list)
    status: str = "pending"


class DecomposedAsset(BaseModel):
    """A processable asset or sub-scene derived from the original scene."""

    model_config = ConfigDict(extra="forbid")

    asset_id: str
    label: str
    original_root_path: str
    working_usd_path: str | None = None
    working_root_path: str | None = None
    source_path_prefixes: list[str] = Field(default_factory=list)
    parent_group: str | None = None
    source_classification: str | None = None
    mesh_count: int = 0
    vertex_count: int = 0
    instance_group_id: str | None = None
    representative_asset_id: str | None = None
    processable: bool = True
    skip_reason: str | None = None
    status: str = "pending"
    context: dict[str, Any] = Field(default_factory=dict)


class SceneDecompositionManifest(BaseModel):
    """Generic manifest describing scene decomposition and topology mapping."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCENE_MANIFEST_SCHEMA_VERSION
    scene_id: str
    original_usd_path: str
    generated_at: str = ""
    stage_metadata: StageMetadata = Field(default_factory=StageMetadata)
    decomposition_policy: DecompositionPolicy = Field(
        default_factory=DecompositionPolicy
    )
    analysis: dict[str, Any] = Field(default_factory=dict)
    assets: list[DecomposedAsset] = Field(default_factory=list)
    instance_groups: list[SceneInstanceGroup] = Field(default_factory=list)
    payload_groups: list[ScenePayloadGroup] = Field(default_factory=list)
    prototype_groups: list[ScenePrototypeGroup] = Field(default_factory=list)
    mapping_artifacts: list[ArtifactReference] = Field(default_factory=list)
    diagnostics: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_identity_fields(self) -> SceneDecompositionManifest:
        asset_ids = [asset.asset_id for asset in self.assets]
        if any(not asset_id for asset_id in asset_ids):
            raise ValueError("asset_id values must be non-empty")
        if len(asset_ids) != len(set(asset_ids)):
            raise ValueError("asset_id values must be unique")

        for label, group_ids in (
            ("instance_groups", [group.group_id for group in self.instance_groups]),
            ("payload_groups", [group.group_id for group in self.payload_groups]),
            ("prototype_groups", [group.group_id for group in self.prototype_groups]),
        ):
            if any(not group_id for group_id in group_ids):
                raise ValueError(f"{label}.group_id values must be non-empty")
            if len(group_ids) != len(set(group_ids)):
                raise ValueError(f"{label}.group_id values must be unique")

        known_asset_ids = set(asset_ids)
        for group in self.instance_groups:
            representative = group.representative_asset_id
            if representative is not None and representative not in known_asset_ids:
                raise ValueError(
                    f"instance group {group.group_id} references unknown "
                    f"representative asset {representative}"
                )
        return self

    @property
    def processable_assets(self) -> list[DecomposedAsset]:
        """Return assets marked processable by the decomposition adapter."""

        return [asset for asset in self.assets if asset.processable]


class ManifestCatalogEntry(BaseModel):
    """One finalized decomposition view available to Workflow 2."""

    model_config = ConfigDict(extra="forbid")

    manifest_id: str = Field(min_length=1)
    intent: str = Field(min_length=1)
    path: str
    finalized: bool = True
    manifest_digest: str


class ManifestCatalog(BaseModel):
    """Catalog of same-scene finalized decomposition views."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[MANIFEST_CATALOG_SCHEMA_VERSION] = (
        MANIFEST_CATALOG_SCHEMA_VERSION
    )
    original_usd_path: str
    source_identity_digest: str
    structural_analysis_id: str
    manifests: list[ManifestCatalogEntry]

    @model_validator(mode="after")
    def validate_manifest_ids(self) -> ManifestCatalog:
        manifest_ids = [entry.manifest_id for entry in self.manifests]
        if len(manifest_ids) != len(set(manifest_ids)):
            raise ValueError("manifest_id values must be unique")
        return self


class SceneDecompositionRequest(BaseModel):
    """Input for creating a scene decomposition manifest."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    usd_path: Path
    output_dir: Path
    manifest_id: str = Field(default="default", min_length=1)
    decomposition_intent: str = Field(default="generic_processing", min_length=1)
    root_prim_path: str | None = None
    include_paths: list[str] = Field(default_factory=list)
    exclude_paths: list[str] = Field(default_factory=list)
    asset_filter: list[str] = Field(default_factory=list)
    min_mesh_count: int = Field(default=0, ge=0)
    exclude_invisible_assets: bool = False
    detect_structural_duplicates: bool = False
    detect_payload_groups: bool = True
    detect_native_prototypes: bool = True
    extract_large_payload_representatives: bool = False
    extract_assets: bool = False
    flatten_extracts: bool = True
    extract_workers: int = Field(default=1, ge=1)
    skip_geometry: bool = False
    building_block_min_reuse: int = Field(default=20, ge=1)
    enable_llm_refinement: bool = False
    llm_config: dict[str, Any] | None = None
    write_material_agent_manifest: bool = True


class SceneDecompositionResult(BaseModel):
    """Result and canonical artifacts from a scene decomposition run."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCENE_DECOMPOSITION_RESULT_SCHEMA_VERSION
    success: bool
    output_dir: str
    manifest_path: str | None = None
    manifest_catalog_path: str | None = None
    material_agent_manifest_path: str | None = None
    phase_result_path: str | None = None
    input_digest: str | None = None
    output_digest: str | None = None
    asset_count: int = 0
    processable_asset_count: int = 0
    instance_group_count: int = 0
    payload_group_count: int = 0
    prototype_group_count: int = 0
    error: str | None = None


class DecompositionPhaseResult(BaseModel):
    """Sealed Workflow 1 output consumed by the umbrella coordinator."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[DECOMPOSITION_PHASE_RESULT_SCHEMA_VERSION] = (
        DECOMPOSITION_PHASE_RESULT_SCHEMA_VERSION
    )
    phase: Literal["decomposition"] = "decomposition"
    success: bool
    input_digest: str
    source_scene: str
    source_identity_digest: str
    manifest_catalog_path: str | None = None
    manifest_paths: list[str] = Field(default_factory=list)
    extracted_asset_paths: list[str] = Field(default_factory=list)
    artifact_paths: list[str] = Field(default_factory=list)
    output_digest: str | None = None
    completion_policy_satisfied: bool = False
    unresolved_issues: list[str] = Field(default_factory=list)
    diagnostics: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None


def write_manifest_json(manifest: SceneDecompositionManifest, path: Path) -> Path:
    """Write a scene decomposition manifest as stable JSON."""

    return atomic_write_json(path, manifest)
