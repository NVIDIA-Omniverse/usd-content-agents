# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Texture-local catalog and non-executing adapter for the asset graph.

The provider-free UV operation and the focused prepare/apply-provided/evidence/
review/publish chain are registered. Every leaf has a native terminal
disposition plus exact evidence and saved-stage readback; proposal, provider
generation, critique, and umbrella workflow entrypoints are deliberately absent.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from content_agent_workflows.asset_composition import (
    ArtifactBinding,
    AssetLeafCatalog,
    AssetLeafDescriptor,
    AssetLeafProjectionContext,
    AssetLeafProjectionPayload,
    AssetLeafRuntimeBinding,
    AssetLeafRuntimeBundle,
    bind_usd_dependency_closure,
    canonical_asset_digest,
    verify_usd_dependency_closure,
)
from content_agent_workflows.asset_composition.models import (
    LeafReceiptArtifactCategory,
)
from content_agent_workflows.common.domain_execution import ExecutionArtifactBinding

from .capabilities import (
    TextureCandidateEvidencePacket,
    TextureCapabilityRequest,
    TextureGenerationPacket,
    TextureOuterPlan,
    TextureOuterReviewInput,
    TextureOuterReviewPacket,
    TexturePreparationPacket,
    TexturePublicationReceipt,
    _verify_candidate_evidence_artifacts,
    _verify_generation_artifacts,
    validate_texture_outer_plan,
)
from .scope_validation import (
    TextureScopeInvariantReport,
    validate_texture_scope_invariants,
)
from .uv_authoring import (
    TEXTURE_UV_LEAF_ID,
    TextureUvEvidence,
    TextureUvLeafInvocation,
    TextureUvLeafResult,
    TextureUvPolicy,
    TextureUvSavedStageReadback,
    _expected_native_disposition,
    _expected_result_detail,
    _HeldDirectory,
    _read_bound_bytes,
    _read_regular_fd,
    _recompute_saved_stage_contract,
    _verify_binding,
)

TEXTURE_UV_ASSET_ENTRYPOINT = (
    "content-workflow-cli texture agentic-leaf uv-prepare --invocation"
)
TEXTURE_UV_PROJECTION_SCHEMA_VERSION: Literal[
    "content-agent-workflows.texture-uv-verified-operation-projection.v1"
] = "content-agent-workflows.texture-uv-verified-operation-projection.v1"
TEXTURE_UV_PUBLICATION_SCHEMA_VERSION: Literal[
    "content-agent-workflows.texture-uv-asset-leaf-publication.v1"
] = "content-agent-workflows.texture-uv-asset-leaf-publication.v1"
TEXTURE_UV_PROJECTION_FILENAME = "texture_uv_verified_operation_projection.json"
TEXTURE_UV_PUBLICATION_FILENAME = "texture_uv_asset_leaf_publication.json"
TEXTURE_ASSET_LEAF_BUNDLE_ID = "texture"

_MAX_JSON_BYTES = 16 * 1024 * 1024


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _binding_identity(
    binding: ArtifactBinding | ExecutionArtifactBinding,
) -> tuple[str, str, int]:
    return binding.path, binding.sha256, binding.size_bytes


def _binding_identities(
    bindings: tuple[ArtifactBinding, ...] | tuple[ExecutionArtifactBinding, ...],
) -> tuple[tuple[str, str, int], ...]:
    return tuple(_binding_identity(binding) for binding in bindings)


class TextureUvVerifiedOperationProjection(_FrozenModel):
    """Typed non-executing projection of one exact native UV result."""

    schema_version: Literal[
        "content-agent-workflows.texture-uv-verified-operation-projection.v1"
    ] = TEXTURE_UV_PROJECTION_SCHEMA_VERSION
    leaf_id: Literal["texture.uv-prepare.v1"] = TEXTURE_UV_LEAF_ID
    descriptor_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    invocation: ExecutionArtifactBinding
    native_result: ExecutionArtifactBinding
    native_disposition: Literal["passed", "failed", "not_evaluated"]
    policy: TextureUvPolicy
    error: str | None = None
    output: ExecutionArtifactBinding
    output_dependencies: tuple[ArtifactBinding, ...]
    evidence: tuple[ExecutionArtifactBinding, ...] = Field(min_length=1)
    saved_stage_readbacks: tuple[ExecutionArtifactBinding, ...] = Field(min_length=1)
    saved_stage_readback_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    all_uv_ready: bool
    projector_native_leaf_invoked: Literal[False] = False
    projector_provider_invoked: Literal[False] = False
    projector_renderer_invoked: Literal[False] = False
    projector_service_constructed: Literal[False] = False
    projector_vlm_constructed: Literal[False] = False
    projector_image_generator_constructed: Literal[False] = False
    projector_graph_state_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_native_disposition(self) -> Self:
        expected = _expected_native_disposition(
            policy=self.policy,
            all_uv_ready=self.all_uv_ready,
        )
        if self.native_disposition != expected:
            raise ValueError(
                "Texture UV projection cannot upgrade or relabel native disposition"
            )
        if (expected == "failed") != bool(self.error):
            raise ValueError("Texture UV projection failure requires an exact error")
        return self


class TextureUvAssetLeafPublication(_FrozenModel):
    """Exact generic receipt arguments emitted by the Texture projector."""

    schema_version: Literal[
        "content-agent-workflows.texture-uv-asset-leaf-publication.v1"
    ] = TEXTURE_UV_PUBLICATION_SCHEMA_VERSION
    leaf_id: Literal["texture.uv-prepare.v1"] = TEXTURE_UV_LEAF_ID
    descriptor_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    projection: ExecutionArtifactBinding
    native_result: ExecutionArtifactBinding
    native_disposition: Literal["passed", "failed", "not_evaluated"]
    error: str | None = None
    graph_terminal_action: Literal["complete_leaf", "fail_leaf"]
    graph_invocation_path: str = Field(min_length=1)
    graph_result_path: str = Field(min_length=1)
    graph_operation_index_paths: tuple[str, ...] = Field(min_length=1)
    graph_evidence_index_paths: tuple[str, ...] = Field(min_length=1)
    graph_evidence_paths: tuple[str, ...] = Field(min_length=1)
    graph_saved_stage_readback_paths: tuple[str, ...] = Field(min_length=1)
    output: ExecutionArtifactBinding
    output_dependencies: tuple[ArtifactBinding, ...]

    @model_validator(mode="after")
    def validate_graph_paths(self) -> Self:
        expected_action = (
            "fail_leaf" if self.native_disposition == "failed" else "complete_leaf"
        )
        if self.graph_terminal_action != expected_action or (
            self.native_disposition == "failed"
        ) != bool(self.error):
            raise ValueError(
                "Texture UV publication action differs from native disposition"
            )
        paths = (
            self.graph_invocation_path,
            self.graph_result_path,
            *self.graph_operation_index_paths,
            *self.graph_evidence_index_paths,
            *self.graph_evidence_paths,
            *self.graph_saved_stage_readback_paths,
        )
        if len(paths) != len(set(paths)):
            raise ValueError("Texture UV publication graph paths must be unique")
        for value in paths:
            candidate = Path(value)
            if (
                not candidate.is_absolute()
                or ".." in candidate.parts
                or str(candidate) != value
            ):
                raise ValueError(
                    "Texture UV publication paths must be canonical and absolute"
                )
        return self


TEXTURE_PREPARE_LEAF_ID: Literal["texture.prepare.v1"] = "texture.prepare.v1"
TEXTURE_APPLY_PROVIDED_LEAF_ID: Literal["texture.apply-provided.v1"] = (
    "texture.apply-provided.v1"
)
TEXTURE_EVIDENCE_LEAF_ID: Literal["texture.evidence.v1"] = "texture.evidence.v1"
TEXTURE_REVIEW_LEAF_ID: Literal["texture.review.v1"] = "texture.review.v1"
TEXTURE_PUBLISH_LEAF_ID: Literal["texture.publish.v1"] = "texture.publish.v1"
TextureFocusedLeafId = Literal[
    "texture.prepare.v1",
    "texture.apply-provided.v1",
    "texture.evidence.v1",
    "texture.review.v1",
    "texture.publish.v1",
]
TEXTURE_FOCUSED_LEAF_IDS: tuple[TextureFocusedLeafId, ...] = (
    TEXTURE_PREPARE_LEAF_ID,
    TEXTURE_APPLY_PROVIDED_LEAF_ID,
    TEXTURE_EVIDENCE_LEAF_ID,
    TEXTURE_REVIEW_LEAF_ID,
    TEXTURE_PUBLISH_LEAF_ID,
)
TEXTURE_FOCUSED_INVOCATION_SCHEMA_VERSION: Literal[
    "content-agent-workflows.texture-focused-leaf-invocation.v1"
] = "content-agent-workflows.texture-focused-leaf-invocation.v1"
TEXTURE_FOCUSED_RESULT_SCHEMA_VERSION: Literal[
    "content-agent-workflows.texture-focused-leaf-result.v1"
] = "content-agent-workflows.texture-focused-leaf-result.v1"
TEXTURE_FOCUSED_FAILURE_SCHEMA_VERSION: Literal[
    "content-agent-workflows.texture-focused-native-failure.v1"
] = "content-agent-workflows.texture-focused-native-failure.v1"
TEXTURE_FOCUSED_FAILURE_READBACK_SCHEMA_VERSION: Literal[
    "content-agent-workflows.texture-focused-failure-readback.v1"
] = "content-agent-workflows.texture-focused-failure-readback.v1"
TEXTURE_FOCUSED_READBACK_SCHEMA_VERSION: Literal[
    "content-agent-workflows.texture-focused-saved-stage-readback.v1"
] = "content-agent-workflows.texture-focused-saved-stage-readback.v1"
TEXTURE_FOCUSED_RECEIPT_SCHEMA_VERSION: Literal[
    "content-agent-workflows.texture-focused-terminal-receipt.v1"
] = "content-agent-workflows.texture-focused-terminal-receipt.v1"
TEXTURE_FOCUSED_PROJECTION_SCHEMA_VERSION: Literal[
    "content-agent-workflows.texture-focused-verified-operation-projection.v1"
] = "content-agent-workflows.texture-focused-verified-operation-projection.v1"
TEXTURE_FOCUSED_PUBLICATION_SCHEMA_VERSION: Literal[
    "content-agent-workflows.texture-focused-asset-leaf-publication.v1"
] = "content-agent-workflows.texture-focused-asset-leaf-publication.v1"

_FOCUSED_ENTRYPOINTS: dict[TextureFocusedLeafId, str] = {
    leaf_id: (
        "content-workflow-cli texture agentic-leaf "
        f"{leaf_id.removeprefix('texture.').removesuffix('.v1')} --invocation"
    )
    for leaf_id in TEXTURE_FOCUSED_LEAF_IDS
}
_FOCUSED_DEPENDENCIES: dict[TextureFocusedLeafId, list[str]] = {
    TEXTURE_PREPARE_LEAF_ID: [TEXTURE_UV_LEAF_ID],
    TEXTURE_APPLY_PROVIDED_LEAF_ID: [TEXTURE_PREPARE_LEAF_ID],
    TEXTURE_EVIDENCE_LEAF_ID: [TEXTURE_APPLY_PROVIDED_LEAF_ID],
    TEXTURE_REVIEW_LEAF_ID: [TEXTURE_EVIDENCE_LEAF_ID],
    TEXTURE_PUBLISH_LEAF_ID: [TEXTURE_REVIEW_LEAF_ID],
}
_RENDERER_LEAF_IDS = frozenset(
    {
        TEXTURE_PREPARE_LEAF_ID,
        TEXTURE_EVIDENCE_LEAF_ID,
    }
)
_BASE_REQUIRED_ARTIFACT_CATEGORIES: tuple[LeafReceiptArtifactCategory, ...] = (
    "evidence",
    "saved_stage_readback",
)
_FOCUSED_REQUIRED_ARTIFACT_CATEGORIES: dict[
    TextureFocusedLeafId,
    tuple[LeafReceiptArtifactCategory, ...],
] = {
    leaf_id: (
        *_BASE_REQUIRED_ARTIFACT_CATEGORIES,
        *(("resource_release",) if leaf_id in _RENDERER_LEAF_IDS else ()),
    )
    for leaf_id in TEXTURE_FOCUSED_LEAF_IDS
}


def _focused_json_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _focused_json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_focused_json_value(item) for item in value]
    return value


class TextureFocusedLeafInvocation(_FrozenModel):
    """Exact outer-authored input for one focused Texture graph leaf."""

    schema_version: Literal[
        "content-agent-workflows.texture-focused-leaf-invocation.v1"
    ] = TEXTURE_FOCUSED_INVOCATION_SCHEMA_VERSION
    leaf_id: TextureFocusedLeafId
    attempt_root: str = Field(min_length=1)
    request: TextureCapabilityRequest | None = None
    uv_result: ExecutionArtifactBinding | None = None
    request_binding: ExecutionArtifactBinding | None = None
    preparation: ExecutionArtifactBinding | None = None
    outer_plan: ExecutionArtifactBinding | None = None
    generation: ExecutionArtifactBinding | None = None
    candidate_evidence: ExecutionArtifactBinding | None = None
    review_input: ExecutionArtifactBinding | None = None
    outer_review: ExecutionArtifactBinding | None = None
    output_asset_path: str | None = None
    selected_mode: Literal["agentic"] = "agentic"
    provider_status: Literal["not_requested"] = "not_requested"
    provider_invoked: Literal[False] = False
    fixed_pipeline_invoked: Literal[False] = False
    umbrella_run_invoked: Literal[False] = False
    compatibility_agent_step_invoked: Literal[False] = False
    nested_coordinator_invoked: Literal[False] = False

    @model_validator(mode="after")
    def validate_operation_fields(self) -> Self:
        root = Path(self.attempt_root)
        if (
            not root.is_absolute()
            or ".." in root.parts
            or str(root) != self.attempt_root
        ):
            raise ValueError(
                "Texture focused attempt root must be canonical and absolute"
            )
        fields = {
            "request": self.request,
            "uv_result": self.uv_result,
            "request_binding": self.request_binding,
            "preparation": self.preparation,
            "outer_plan": self.outer_plan,
            "generation": self.generation,
            "candidate_evidence": self.candidate_evidence,
            "review_input": self.review_input,
            "outer_review": self.outer_review,
            "output_asset_path": self.output_asset_path,
        }
        required: dict[TextureFocusedLeafId, set[str]] = {
            TEXTURE_PREPARE_LEAF_ID: {"request", "uv_result"},
            TEXTURE_APPLY_PROVIDED_LEAF_ID: {"preparation", "outer_plan"},
            TEXTURE_EVIDENCE_LEAF_ID: {
                "preparation",
                "outer_plan",
                "generation",
            },
            TEXTURE_REVIEW_LEAF_ID: {
                "outer_plan",
                "generation",
                "candidate_evidence",
                "review_input",
            },
            TEXTURE_PUBLISH_LEAF_ID: {
                "request_binding",
                "preparation",
                "outer_plan",
                "generation",
                "candidate_evidence",
                "outer_review",
                "output_asset_path",
            },
        }
        present = {name for name, value in fields.items() if value is not None}
        if present != required[self.leaf_id]:
            raise ValueError(
                f"Texture {self.leaf_id} invocation fields differ: "
                f"expected {sorted(required[self.leaf_id])}, got {sorted(present)}"
            )
        if self.leaf_id == TEXTURE_PREPARE_LEAF_ID:
            if self.request is None or self.request.output_dir != str(root / "native"):
                raise ValueError(
                    "Texture prepare request output_dir must be its native attempt root"
                )
        if self.output_asset_path is not None:
            output = Path(self.output_asset_path)
            native_root = root / "native"
            if (
                not output.is_absolute()
                or output.parent != native_root
                or str(output) != self.output_asset_path
            ):
                raise ValueError(
                    "Texture publication output must be direct in its native attempt root"
                )
        return self


class TextureFocusedNativeFailure(_FrozenModel):
    """Exact attempt-local failure emitted after a focused native root is claimed."""

    schema_version: Literal[
        "content-agent-workflows.texture-focused-native-failure.v1"
    ] = TEXTURE_FOCUSED_FAILURE_SCHEMA_VERSION
    leaf_id: TextureFocusedLeafId
    invocation: ExecutionArtifactBinding
    source: ExecutionArtifactBinding
    source_dependencies: tuple[ArtifactBinding, ...]
    output: ExecutionArtifactBinding
    output_dependencies: tuple[ArtifactBinding, ...]
    packet_chain: tuple[ExecutionArtifactBinding, ...] = ()
    renderer_evidence: tuple[ExecutionArtifactBinding, ...] = ()
    renderer_receipt_custody_error: str | None = Field(default=None, min_length=1)
    error: str = Field(min_length=1)
    provider_status: Literal["not_requested"] = "not_requested"
    provider_invoked: Literal[False] = False
    renderer_invoked: bool
    resources_released: Literal[True] = True
    failure_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(cls, **values: Any) -> TextureFocusedNativeFailure:
        payload = {
            "schema_version": TEXTURE_FOCUSED_FAILURE_SCHEMA_VERSION,
            **values,
        }
        return cls.model_validate(
            {
                **payload,
                "failure_identity_sha256": canonical_asset_digest(
                    _focused_json_value(payload)
                ),
            }
        )

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        payload = self.model_dump(
            mode="json",
            exclude={"failure_identity_sha256"},
        )
        if self.failure_identity_sha256 != canonical_asset_digest(payload):
            raise ValueError("Texture focused native failure identity is stale")
        if self.renderer_invoked and not self.renderer_evidence:
            raise ValueError(
                "Texture focused renderer failure requires exact receipt evidence"
            )
        if self.leaf_id not in _RENDERER_LEAF_IDS and (
            self.renderer_invoked or self.renderer_evidence
        ):
            raise ValueError(
                "Texture non-renderer failure cannot claim renderer receipt evidence"
            )
        return self


class TextureFocusedSavedStageReadback(_FrozenModel):
    """Recomputed exact source/output and non-target preservation facts."""

    schema_version: Literal[
        "content-agent-workflows.texture-focused-saved-stage-readback.v1"
    ] = TEXTURE_FOCUSED_READBACK_SCHEMA_VERSION
    leaf_id: TextureFocusedLeafId
    invocation: ExecutionArtifactBinding
    native_packet: ExecutionArtifactBinding
    source: ExecutionArtifactBinding
    source_dependencies: tuple[ArtifactBinding, ...]
    saved_stage: ExecutionArtifactBinding
    saved_stage_dependencies: tuple[ArtifactBinding, ...]
    packet_chain: tuple[ExecutionArtifactBinding, ...] = Field(min_length=1)
    scope_invariant_report: TextureScopeInvariantReport
    reopened_saved_stage: Literal[True] = True
    dependency_closure_verified: Literal[True] = True
    non_target_topology_preserved: bool
    non_target_physics_preserved: bool
    non_target_material_preserved: bool
    provider_status: Literal["not_requested"] = "not_requested"
    provider_invoked: Literal[False] = False
    readback_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(cls, **values: Any) -> TextureFocusedSavedStageReadback:
        payload = {
            "schema_version": TEXTURE_FOCUSED_READBACK_SCHEMA_VERSION,
            **values,
        }
        return cls.model_validate(
            {
                **payload,
                "readback_identity_sha256": canonical_asset_digest(
                    _focused_json_value(payload)
                ),
            }
        )

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        payload = self.model_dump(
            mode="json",
            exclude={"readback_identity_sha256"},
        )
        if self.readback_identity_sha256 != canonical_asset_digest(payload):
            raise ValueError("Texture focused saved-stage readback identity is stale")
        report = self.scope_invariant_report
        expected_topology = (
            report.geometry_unchanged and report.structure_unchanged_outside_target
        )
        expected_material = (
            report.non_target_materials_unchanged and report.bindings_unchanged
        )
        if (
            self.non_target_topology_preserved != expected_topology
            or self.non_target_physics_preserved != expected_topology
            or self.non_target_material_preserved != expected_material
        ):
            raise ValueError(
                "Texture focused preservation flags differ from the scope report"
            )
        return self


class TextureFocusedFailureReadback(_FrozenModel):
    """Exact source-stage readback retained after a native operation exception."""

    schema_version: Literal[
        "content-agent-workflows.texture-focused-failure-readback.v1"
    ] = TEXTURE_FOCUSED_FAILURE_READBACK_SCHEMA_VERSION
    leaf_id: TextureFocusedLeafId
    invocation: ExecutionArtifactBinding
    native_packet: ExecutionArtifactBinding
    source: ExecutionArtifactBinding
    source_dependencies: tuple[ArtifactBinding, ...]
    saved_stage: ExecutionArtifactBinding
    saved_stage_dependencies: tuple[ArtifactBinding, ...]
    packet_chain: tuple[ExecutionArtifactBinding, ...] = Field(min_length=1)
    error: str = Field(min_length=1)
    reopened_saved_stage: Literal[True] = True
    dependency_closure_verified: Literal[True] = True
    non_target_topology_preserved: Literal[True] = True
    non_target_physics_preserved: Literal[True] = True
    non_target_material_preserved: Literal[True] = True
    provider_status: Literal["not_requested"] = "not_requested"
    provider_invoked: Literal[False] = False
    readback_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(cls, **values: Any) -> TextureFocusedFailureReadback:
        payload = {
            "schema_version": TEXTURE_FOCUSED_FAILURE_READBACK_SCHEMA_VERSION,
            **values,
        }
        return cls.model_validate(
            {
                **payload,
                "readback_identity_sha256": canonical_asset_digest(
                    _focused_json_value(payload)
                ),
            }
        )

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        payload = self.model_dump(
            mode="json",
            exclude={"readback_identity_sha256"},
        )
        if self.readback_identity_sha256 != canonical_asset_digest(payload):
            raise ValueError("Texture focused failure readback identity is stale")
        if (
            self.saved_stage != self.source
            or self.saved_stage_dependencies != self.source_dependencies
        ):
            raise ValueError(
                "Texture focused failure readback must retain its exact source"
            )
        return self


class TextureFocusedLeafResult(_FrozenModel):
    """Native Texture graph result; never a relabeled generic disposition."""

    schema_version: Literal[
        "content-agent-workflows.texture-focused-leaf-result.v1"
    ] = TEXTURE_FOCUSED_RESULT_SCHEMA_VERSION
    leaf_id: TextureFocusedLeafId
    invocation: ExecutionArtifactBinding
    native_packet: ExecutionArtifactBinding
    native_disposition: Literal["passed", "failed", "not_evaluated"]
    error: str | None = None
    source: ExecutionArtifactBinding
    output: ExecutionArtifactBinding
    output_dependencies: tuple[ArtifactBinding, ...]
    evidence: tuple[ExecutionArtifactBinding, ...] = Field(min_length=1)
    saved_stage_readbacks: tuple[ExecutionArtifactBinding, ...] = Field(min_length=1)
    provider_status: Literal["not_requested"] = "not_requested"
    provider_invoked: Literal[False] = False
    renderer_invoked: bool
    fixed_pipeline_invoked: Literal[False] = False
    umbrella_run_invoked: Literal[False] = False
    compatibility_agent_step_invoked: Literal[False] = False
    nested_coordinator_invoked: Literal[False] = False

    @model_validator(mode="after")
    def validate_native_terminal_disposition(self) -> Self:
        if (self.native_disposition == "failed") != bool(self.error):
            raise ValueError(
                "Texture focused failed disposition requires an exact error"
            )
        return self


class TextureFocusedTerminalReceipt(_FrozenModel):
    """Self-digested native terminal receipt for one focused operation."""

    schema_version: Literal[
        "content-agent-workflows.texture-focused-terminal-receipt.v1"
    ] = TEXTURE_FOCUSED_RECEIPT_SCHEMA_VERSION
    leaf_id: TextureFocusedLeafId
    invocation: ExecutionArtifactBinding
    result: ExecutionArtifactBinding
    native_packet: ExecutionArtifactBinding
    saved_stage_readback: ExecutionArtifactBinding
    native_disposition: Literal["passed", "failed", "not_evaluated"]
    error: str | None = None
    output: ExecutionArtifactBinding
    provider_status: Literal["not_requested"] = "not_requested"
    provider_invoked: Literal[False] = False
    renderer_invoked: bool
    resources_released: Literal[True] = True
    receipt_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(cls, **values: Any) -> TextureFocusedTerminalReceipt:
        payload = {
            "schema_version": TEXTURE_FOCUSED_RECEIPT_SCHEMA_VERSION,
            **values,
        }
        return cls.model_validate(
            {
                **payload,
                "receipt_identity_sha256": canonical_asset_digest(
                    _focused_json_value(payload)
                ),
            }
        )

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        payload = self.model_dump(mode="json", exclude={"receipt_identity_sha256"})
        if self.receipt_identity_sha256 != canonical_asset_digest(payload):
            raise ValueError("Texture focused terminal receipt identity is stale")
        if (self.native_disposition == "failed") != bool(self.error):
            raise ValueError(
                "Texture focused terminal receipt disposition and error differ"
            )
        return self


class TextureFocusedVerifiedOperationProjection(_FrozenModel):
    """Non-executing generic projection of a recomputed focused result."""

    schema_version: Literal[
        "content-agent-workflows.texture-focused-verified-operation-projection.v1"
    ] = TEXTURE_FOCUSED_PROJECTION_SCHEMA_VERSION
    leaf_id: TextureFocusedLeafId
    descriptor_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    invocation: ExecutionArtifactBinding
    native_result: ExecutionArtifactBinding
    native_terminal_receipt: ExecutionArtifactBinding
    native_disposition: Literal["passed", "failed", "not_evaluated"]
    error: str | None = None
    output: ExecutionArtifactBinding
    output_dependencies: tuple[ArtifactBinding, ...]
    evidence: tuple[ExecutionArtifactBinding, ...] = Field(min_length=1)
    saved_stage_readbacks: tuple[ExecutionArtifactBinding, ...] = Field(min_length=1)
    provider_status: Literal["not_requested"] = "not_requested"
    native_renderer_invoked: bool
    projector_leaf_invoked: Literal[False] = False
    projector_provider_invoked: Literal[False] = False
    projector_renderer_invoked: Literal[False] = False
    projector_graph_state_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_native_terminal_disposition(self) -> Self:
        if (self.native_disposition == "failed") != bool(self.error):
            raise ValueError("Texture focused projection disposition and error differ")
        return self


class TextureFocusedAssetLeafPublication(_FrozenModel):
    """Exact generic complete-leaf arguments emitted after re-verification."""

    schema_version: Literal[
        "content-agent-workflows.texture-focused-asset-leaf-publication.v1"
    ] = TEXTURE_FOCUSED_PUBLICATION_SCHEMA_VERSION
    leaf_id: TextureFocusedLeafId
    descriptor_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    projection: ExecutionArtifactBinding
    native_result: ExecutionArtifactBinding
    native_disposition: Literal["passed", "failed", "not_evaluated"]
    error: str | None = None
    graph_terminal_action: Literal["complete_leaf", "fail_leaf"]
    graph_invocation_path: str = Field(min_length=1)
    graph_result_path: str = Field(min_length=1)
    graph_native_terminal_receipt_path: str = Field(min_length=1)
    graph_operation_index_paths: tuple[str, ...] = Field(min_length=1)
    graph_evidence_index_paths: tuple[str, ...] = Field(min_length=1)
    graph_evidence_paths: tuple[str, ...] = Field(min_length=1)
    graph_saved_stage_readback_paths: tuple[str, ...] = Field(min_length=1)
    output: ExecutionArtifactBinding
    output_dependencies: tuple[ArtifactBinding, ...]
    resource_claims: tuple[str, ...] = ()
    graph_resource_release_paths: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_graph_paths(self) -> Self:
        expected_action = (
            "fail_leaf" if self.native_disposition == "failed" else "complete_leaf"
        )
        if self.graph_terminal_action != expected_action or (
            self.native_disposition == "failed"
        ) != bool(self.error):
            raise ValueError(
                "Texture focused publication action differs from disposition"
            )
        paths = (
            self.graph_invocation_path,
            self.graph_result_path,
            self.graph_native_terminal_receipt_path,
            *self.graph_operation_index_paths,
            *self.graph_evidence_index_paths,
            *self.graph_evidence_paths,
            *self.graph_saved_stage_readback_paths,
        )
        if len(paths) != len(set(paths)):
            raise ValueError("Texture focused publication graph paths must be unique")
        if len(self.graph_resource_release_paths) != len(
            set(self.graph_resource_release_paths)
        ):
            raise ValueError(
                "Texture focused publication resource releases must be unique"
            )
        requires_renderer_release = self.leaf_id in _RENDERER_LEAF_IDS
        if (
            bool(self.resource_claims) != requires_renderer_release
            or bool(self.graph_resource_release_paths) != requires_renderer_release
        ):
            raise ValueError(
                "Texture focused publication renderer release differs from descriptor"
            )
        if self.resource_claims and (
            self.resource_claims != (f"texture.renderer:{self.leaf_id}",)
            or self.graph_resource_release_paths
            != (self.graph_native_terminal_receipt_path,)
        ):
            raise ValueError(
                "Texture focused publication renderer custody is inconsistent"
            )
        for value in (*paths, *self.graph_resource_release_paths):
            candidate = Path(value)
            if (
                not candidate.is_absolute()
                or ".." in candidate.parts
                or str(candidate) != value
            ):
                raise ValueError(
                    "Texture focused publication paths must be canonical and absolute"
                )
        return self


def _artifact_binding(
    binding: ArtifactBinding | ExecutionArtifactBinding,
) -> ArtifactBinding:
    return ArtifactBinding.model_validate(binding.model_dump(mode="json"))


def _load_contained_model[ModelT: BaseModel](
    attempt: _HeldDirectory,
    path: str | Path,
    model: type[ModelT],
    *,
    label: str,
) -> tuple[ExecutionArtifactBinding, ModelT]:
    try:
        binding, data, _identity = attempt.read(
            path,
            max_bytes=_MAX_JSON_BYTES,
        )
        payload = model.model_validate_json(data)
    except ValueError as exc:
        raise ValueError(f"Invalid {label}: {exc}") from exc
    return binding, payload


def _verify_contained_binding(
    attempt: _HeldDirectory,
    binding: ExecutionArtifactBinding,
    *,
    label: str,
) -> None:
    attempt.verify_binding(binding, label=label)


def _verify_native_bindings(
    attempt: _HeldDirectory,
    *,
    invocation_binding: ExecutionArtifactBinding,
    invocation: TextureUvLeafInvocation,
    result_binding: ExecutionArtifactBinding,
    result: TextureUvLeafResult,
) -> None:
    _verify_contained_binding(
        attempt,
        invocation_binding,
        label="Texture UV invocation",
    )
    _verify_contained_binding(
        attempt,
        result_binding,
        label="Texture UV native result",
    )
    for index, binding in enumerate(result.evidence, start=1):
        _verify_contained_binding(
            attempt,
            binding,
            label=f"Texture UV evidence {index}",
        )
    for index, binding in enumerate(result.saved_stage_readbacks, start=1):
        _verify_contained_binding(
            attempt,
            binding,
            label=f"Texture UV saved-stage readback {index}",
        )
    _verify_binding(invocation.source, label="Texture UV source")
    verify_usd_dependency_closure(
        invocation.source.path,
        invocation.source_dependencies,
    )
    if result.output == invocation.source:
        _verify_binding(result.output, label="Texture UV output")
    else:
        attempt.verify_binding(result.output, label="Texture UV output")
    verify_usd_dependency_closure(result.output.path, result.output_dependencies)
    attempt.verify_path()


def _validate_native_contract(
    attempt: _HeldDirectory,
    *,
    invocation_binding: ExecutionArtifactBinding,
    invocation: TextureUvLeafInvocation,
    result_binding: ExecutionArtifactBinding,
    result: TextureUvLeafResult,
) -> TextureUvSavedStageReadback:
    root = attempt.path
    if invocation.output_dir != str(root):
        raise ValueError(
            "Texture UV invocation output_dir is not its exact attempt root"
        )
    if (
        Path(invocation_binding.path).parent != root
        or Path(result_binding.path).parent != root
    ):
        raise ValueError(
            "Texture UV invocation and result must be direct attempt artifacts"
        )
    if result.invocation != invocation_binding:
        raise ValueError("Texture UV result does not bind its exact invocation")
    if result.source != invocation.source:
        raise ValueError("Texture UV result changed its source identity")
    if len(result.evidence) != 1 or len(result.saved_stage_readbacks) != 1:
        raise ValueError(
            "Texture UV projector requires exactly one native evidence and readback"
        )

    evidence_binding, evidence = _load_contained_model(
        attempt,
        result.evidence[0].path,
        TextureUvEvidence,
        label="Texture UV evidence",
    )
    if evidence_binding != result.evidence[0]:
        raise ValueError("Texture UV result evidence identity is stale")
    readback_binding, readback = _load_contained_model(
        attempt,
        result.saved_stage_readbacks[0].path,
        TextureUvSavedStageReadback,
        label="Texture UV saved-stage readback",
    )
    if readback_binding != result.saved_stage_readbacks[0]:
        raise ValueError("Texture UV result readback identity is stale")

    expected_disposition = _expected_native_disposition(
        policy=invocation.policy,
        all_uv_ready=readback.all_uv_ready,
    )
    if result.native_disposition != expected_disposition:
        raise ValueError(
            "Texture UV projector refuses to upgrade or relabel native disposition"
        )
    expected_authoring_method = (
        "bounded_box_fallback" if readback.authored_mesh_paths else None
    )
    if (
        readback.source != invocation.source
        or _binding_identities(readback.source_dependencies)
        != _binding_identities(invocation.source_dependencies)
        or readback.saved_stage != result.output
        or _binding_identities(readback.saved_stage_dependencies)
        != _binding_identities(result.output_dependencies)
        or readback.policy != invocation.policy
        or readback.authoring_method != expected_authoring_method
        or readback.requested_prim_paths != invocation.target_prim_paths
    ):
        raise ValueError(
            "Texture UV invocation, result, and saved-stage readback identities differ"
        )
    if not (
        readback.material_identities_unchanged
        and readback.preserved_uv_identities_unchanged
        and readback.external_dependencies_preserved
        and readback.reopened_saved_stage
    ):
        raise ValueError("Texture UV saved-stage invariants are not all satisfied")
    if (
        evidence.source != invocation.source
        or evidence.saved_stage != result.output
        or evidence.policy != invocation.policy
        or evidence.authoring_method != readback.authoring_method
        or evidence.requested_prim_paths != invocation.target_prim_paths
        or evidence.resolved_mesh_paths != readback.resolved_mesh_paths
        or evidence.authored_mesh_paths != readback.authored_mesh_paths
        or evidence.saved_stage_readback_identity_sha256
        != readback.readback_identity_sha256
        or evidence.external_dependencies_preserved
        != readback.external_dependencies_preserved
    ):
        raise ValueError("Texture UV evidence differs from its exact native readback")
    (
        recomputed_readback,
        recomputed_evidence,
        rejection_detail,
    ) = _recompute_saved_stage_contract(
        attempt=attempt,
        invocation=invocation,
        result=result,
    )
    if recomputed_readback != readback:
        raise ValueError(
            "Texture UV saved-stage readback differs from recomputed stage facts"
        )
    if recomputed_evidence != evidence:
        raise ValueError(
            "Texture UV evidence differs from recomputed source and output facts"
        )
    recomputed_disposition = _expected_native_disposition(
        policy=invocation.policy,
        all_uv_ready=recomputed_readback.all_uv_ready,
    )
    expected_detail = rejection_detail or _expected_result_detail(
        policy=invocation.policy,
        native_disposition=recomputed_disposition,
        mesh_readbacks=recomputed_readback.mesh_readbacks,
    )
    if result.detail != expected_detail:
        raise ValueError(
            "Texture UV native result detail differs from recomputed saved-stage facts"
        )
    expected_error = expected_detail if recomputed_disposition == "failed" else None
    if result.error != expected_error:
        raise ValueError(
            "Texture UV native result error differs from recomputed disposition"
        )
    if result.output != invocation.source:
        _verify_contained_binding(
            attempt,
            result.output,
            label="Texture UV authored output",
        )
    _verify_native_bindings(
        attempt,
        invocation_binding=invocation_binding,
        invocation=invocation,
        result_binding=result_binding,
        result=result,
    )
    return readback


def _project_texture_uv_runtime(
    invocation: BaseModel,
    result: BaseModel,
    context: AssetLeafProjectionContext,
) -> AssetLeafProjectionPayload:
    """Recompute one UV result and return only frozen generic projection facts."""

    request = cast(TextureUvLeafInvocation, invocation)
    native = cast(TextureUvLeafResult, result)
    root = Path(request.output_dir)
    with _HeldDirectory.open(root) as attempt:
        invocation_binding, loaded_invocation = _load_contained_model(
            attempt,
            context.invocation_artifact.path,
            TextureUvLeafInvocation,
            label="Texture UV invocation",
        )
        result_binding, loaded_result = _load_contained_model(
            attempt,
            context.result_artifact.path,
            TextureUvLeafResult,
            label="Texture UV native result",
        )
        if (
            Path(context.invocation_artifact.path).parent != attempt.path
            or Path(context.result_artifact.path).parent != attempt.path
            or request.output_dir != str(attempt.path)
        ):
            raise ValueError(
                "Texture UV projection context artifacts must be direct in the "
                "native attempt root"
            )
        if loaded_invocation != request or loaded_result != native:
            raise ValueError(
                "Texture UV projection models differ from their selected artifacts"
            )
        context.require_native_artifact_chain(
            invocation_artifact=_artifact_binding(native.invocation),
            result_artifact=_artifact_binding(result_binding),
        )
        _validate_native_contract(
            attempt,
            invocation_binding=invocation_binding,
            invocation=request,
            result_binding=result_binding,
            result=native,
        )
        attempt.verify_path()
    return AssetLeafProjectionPayload(
        native_disposition=native.native_disposition,
        native_status=native.native_disposition,
        native_terminal_receipt=context.result_artifact,
        evidence=tuple(_artifact_binding(item) for item in native.evidence),
        saved_stage_readbacks=tuple(
            _artifact_binding(item) for item in native.saved_stage_readbacks
        ),
        summary=native.detail,
        error=native.error,
    )


def project_texture_uv_verified_operation(
    *,
    attempt_root: str | Path,
    invocation_path: str | Path,
    native_result_path: str | Path,
) -> TextureUvAssetLeafPublication:
    """Reverify and project one native UV result without executing graph state."""

    with _HeldDirectory.open(attempt_root) as attempt:
        root = attempt.path
        invocation_binding, invocation = _load_contained_model(
            attempt,
            invocation_path,
            TextureUvLeafInvocation,
            label="Texture UV invocation",
        )
        result_binding, result = _load_contained_model(
            attempt,
            native_result_path,
            TextureUvLeafResult,
            label="Texture UV native result",
        )
        readback = _validate_native_contract(
            attempt,
            invocation_binding=invocation_binding,
            invocation=invocation,
            result_binding=result_binding,
            result=result,
        )
        descriptor = next(
            item
            for item in texture_asset_leaf_descriptors()
            if item.leaf_id == TEXTURE_UV_LEAF_ID
        )
        projection = TextureUvVerifiedOperationProjection(
            descriptor_digest=descriptor.descriptor_digest,
            invocation=invocation_binding,
            native_result=result_binding,
            native_disposition=result.native_disposition,
            policy=invocation.policy,
            error=result.error,
            output=result.output,
            output_dependencies=result.output_dependencies,
            evidence=result.evidence,
            saved_stage_readbacks=result.saved_stage_readbacks,
            saved_stage_readback_identity_sha256=(readback.readback_identity_sha256),
            all_uv_ready=readback.all_uv_ready,
        )

        projection_path = root / TEXTURE_UV_PROJECTION_FILENAME
        publication_path = root / TEXTURE_UV_PUBLICATION_FILENAME
        for name in (
            TEXTURE_UV_PROJECTION_FILENAME,
            TEXTURE_UV_PUBLICATION_FILENAME,
        ):
            if attempt.exists(name):
                raise FileExistsError(
                    f"Texture UV projector refuses to replace artifact: {root / name}"
                )

        _verify_native_bindings(
            attempt,
            invocation_binding=invocation_binding,
            invocation=invocation,
            result_binding=result_binding,
            result=result,
        )
        created: dict[str, tuple[int, int]] = {}
        created[TEXTURE_UV_PROJECTION_FILENAME] = attempt.write_json(
            TEXTURE_UV_PROJECTION_FILENAME,
            projection,
        )
        projection_binding = attempt.binding(
            TEXTURE_UV_PROJECTION_FILENAME,
            identity=created[TEXTURE_UV_PROJECTION_FILENAME],
        )
        publication = TextureUvAssetLeafPublication(
            descriptor_digest=descriptor.descriptor_digest,
            projection=projection_binding,
            native_result=result_binding,
            native_disposition=result.native_disposition,
            error=result.error,
            graph_terminal_action=(
                "fail_leaf"
                if result.native_disposition == "failed"
                else "complete_leaf"
            ),
            graph_invocation_path=invocation_binding.path,
            graph_result_path=result_binding.path,
            graph_operation_index_paths=(str(projection_path),),
            graph_evidence_index_paths=(str(publication_path),),
            graph_evidence_paths=tuple(item.path for item in result.evidence),
            graph_saved_stage_readback_paths=tuple(
                item.path for item in result.saved_stage_readbacks
            ),
            output=result.output,
            output_dependencies=result.output_dependencies,
        )
        created[TEXTURE_UV_PUBLICATION_FILENAME] = attempt.write_json(
            TEXTURE_UV_PUBLICATION_FILENAME,
            publication,
        )
        _validate_native_contract(
            attempt,
            invocation_binding=invocation_binding,
            invocation=invocation,
            result_binding=result_binding,
            result=result,
        )
        _verify_contained_binding(
            attempt,
            projection_binding,
            label="Texture UV projection",
        )
        publication_binding = attempt.binding(
            TEXTURE_UV_PUBLICATION_FILENAME,
            identity=created[TEXTURE_UV_PUBLICATION_FILENAME],
        )
        _verify_contained_binding(
            attempt,
            publication_binding,
            label="Texture UV publication",
        )
        attempt.verify_path()
        return publication


@dataclass(frozen=True)
class _FocusedNativeContract:
    source: ExecutionArtifactBinding
    source_dependencies: tuple[ArtifactBinding, ...]
    output: ExecutionArtifactBinding
    output_dependencies: tuple[ArtifactBinding, ...]
    packet_chain: tuple[ExecutionArtifactBinding, ...]
    evidence: tuple[ExecutionArtifactBinding, ...]
    native_disposition: Literal["passed", "failed", "not_evaluated"]
    error: str | None
    renderer_invoked: bool
    scope_invariant_report: TextureScopeInvariantReport | None


class _HeldAttemptTree:
    """No-follow reader for direct or nested artifacts below a held attempt."""

    def __init__(self, attempt: _HeldDirectory) -> None:
        self.attempt = attempt

    def _parts(self, path: str | Path) -> tuple[str, ...]:
        raw = str(path)
        candidate = Path(raw).expanduser()
        if (
            not candidate.is_absolute()
            or ".." in candidate.parts
            or str(candidate) != raw
        ):
            raise ValueError("Texture focused artifact path is not canonical absolute")
        try:
            relative = candidate.relative_to(self.attempt.path)
        except ValueError as exc:
            raise ValueError(
                f"Texture focused artifact escapes its attempt: {candidate}"
            ) from exc
        if not relative.parts:
            raise ValueError("Texture focused artifact cannot be the attempt directory")
        return relative.parts

    def _open(self, path: str | Path) -> int:
        parts = self._parts(path)
        parent = os.dup(self.attempt.descriptor)
        try:
            for part in parts[:-1]:
                child = os.open(
                    part,
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=parent,
                )
                os.close(parent)
                parent = child
                metadata = os.fstat(parent)
                if not stat.S_ISDIR(metadata.st_mode):
                    raise ValueError(
                        f"Texture focused artifact parent is not a directory: {part}"
                    )
            return os.open(
                parts[-1],
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0),
                dir_fd=parent,
            )
        except OSError as exc:
            raise ValueError(
                "Texture focused artifact path must not escape through symlinks: "
                f"{path}"
            ) from exc
        finally:
            os.close(parent)

    def read(
        self,
        path: str | Path,
        *,
        max_bytes: int | None = None,
    ) -> tuple[ExecutionArtifactBinding, bytes]:
        descriptor = self._open(path)
        try:
            data, identity = _read_regular_fd(
                descriptor,
                Path(path),
                max_bytes=max_bytes,
            )
        finally:
            os.close(descriptor)
        self.attempt.verify_path()
        rebound = self._open(path)
        try:
            metadata = os.fstat(rebound)
            if (metadata.st_dev, metadata.st_ino) != identity:
                raise ValueError(
                    f"Texture focused artifact identity changed after read: {path}"
                )
        finally:
            os.close(rebound)
        return (
            ExecutionArtifactBinding(
                path=str(path),
                sha256=hashlib.sha256(data).hexdigest(),
                size_bytes=len(data),
            ),
            data,
        )

    def verify_binding(
        self,
        binding: ExecutionArtifactBinding,
        *,
        label: str,
    ) -> bytes:
        observed, data = self.read(binding.path)
        if observed != binding:
            raise ValueError(f"{label} identity changed: {binding.path}")
        return data


@dataclass(frozen=True)
class _RendererReceiptEvidence:
    bindings: tuple[ExecutionArtifactBinding, ...]
    renderer_invoked: bool
    custody_error: str | None = None


def _renderer_invoked_from_journal(journal_bytes: bytes) -> bool:
    renderer_invoked = False
    for line in journal_bytes.splitlines():
        try:
            receipt = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                "Texture renderer command journal contains invalid JSON"
            ) from exc
        arguments = receipt.get("arguments") if isinstance(receipt, dict) else None
        if (
            not isinstance(receipt, dict)
            or receipt.get("schema_version")
            != "content-agent-workflows.usd-cli-command-receipt.v1"
            or not isinstance(arguments, list)
            or any(not isinstance(argument, str) for argument in arguments)
        ):
            raise ValueError("Texture renderer command receipt is invalid")
        if arguments and arguments[0] in {"render", "render-probe"}:
            renderer_invoked = True
    return renderer_invoked


def _renderer_receipt_evidence(
    attempt: _HeldDirectory,
) -> _RendererReceiptEvidence:
    """Bind attempt-owned usd-cli receipts, including an interrupted pair."""

    native_root = attempt.path / "native"
    if not native_root.is_dir():
        return _RendererReceiptEvidence((), False)
    receipt_names = {
        "usd_cli_command_receipts.jsonl",
        "usd_cli_command_receipts.checkpoint.json",
    }
    paths: list[Path] = []
    for directory, child_directories, file_names in os.walk(
        native_root,
        followlinks=False,
    ):
        parent = Path(directory)
        for child in child_directories:
            if (parent / child).is_symlink():
                raise ValueError(
                    "Texture renderer receipt evidence contains a symlink directory"
                )
        for file_name in file_names:
            if file_name in receipt_names:
                path = parent / file_name
                if path.is_symlink():
                    raise ValueError(
                        "Texture renderer receipt evidence must not be a symlink"
                    )
                paths.append(path)

    by_parent: dict[Path, dict[str, Path]] = {}
    for path in paths:
        by_parent.setdefault(path.parent, {})[path.name] = path
    evidence: list[ExecutionArtifactBinding] = []
    renderer_invoked = False
    custody_errors: list[str] = []
    tree = _HeldAttemptTree(attempt)
    for parent, artifacts in sorted(by_parent.items(), key=lambda item: str(item[0])):
        if set(artifacts) != receipt_names:
            relative_parent = parent.relative_to(native_root)
            custody_errors.append(
                "Texture renderer receipt journal and checkpoint are unpaired at "
                f"{relative_parent}"
            )
            journal_path = artifacts.get("usd_cli_command_receipts.jsonl")
            if journal_path is not None:
                journal_binding, journal_bytes = tree.read(
                    journal_path,
                    max_bytes=_MAX_JSON_BYTES,
                )
                renderer_invoked = (
                    _renderer_invoked_from_journal(journal_bytes) or renderer_invoked
                )
                evidence.append(journal_binding)
            checkpoint_path = artifacts.get("usd_cli_command_receipts.checkpoint.json")
            if checkpoint_path is not None:
                checkpoint_binding, checkpoint_bytes = tree.read(
                    checkpoint_path,
                    max_bytes=_MAX_JSON_BYTES,
                )
                try:
                    checkpoint = json.loads(checkpoint_bytes)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        "Texture renderer receipt checkpoint is invalid JSON"
                    ) from exc
                if (
                    not isinstance(checkpoint, dict)
                    or checkpoint.get("schema_version")
                    != "content-agent-workflows.usd-cli-receipt-checkpoint.v1"
                ):
                    raise ValueError("Texture renderer receipt checkpoint is invalid")
                evidence.append(checkpoint_binding)
            continue
        journal_binding, journal_bytes = tree.read(
            artifacts["usd_cli_command_receipts.jsonl"],
            max_bytes=_MAX_JSON_BYTES,
        )
        checkpoint_binding, checkpoint_bytes = tree.read(
            artifacts["usd_cli_command_receipts.checkpoint.json"],
            max_bytes=_MAX_JSON_BYTES,
        )
        try:
            checkpoint = json.loads(checkpoint_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                "Texture renderer receipt checkpoint is invalid JSON"
            ) from exc
        if (
            not isinstance(checkpoint, dict)
            or checkpoint.get("schema_version")
            != "content-agent-workflows.usd-cli-receipt-checkpoint.v1"
            or checkpoint.get("receipt_sha256") != journal_binding.sha256
            or checkpoint.get("receipt_size_bytes") != journal_binding.size_bytes
        ):
            raise ValueError(
                "Texture renderer receipt checkpoint does not bind its journal"
            )
        renderer_invoked = (
            _renderer_invoked_from_journal(journal_bytes) or renderer_invoked
        )
        evidence.extend((journal_binding, checkpoint_binding))
    return _RendererReceiptEvidence(
        tuple(evidence),
        renderer_invoked,
        "; ".join(custody_errors) if custody_errors else None,
    )


def _load_bound_model[ModelT: BaseModel](
    binding: ExecutionArtifactBinding,
    model: type[ModelT],
    *,
    label: str,
) -> ModelT:
    try:
        return model.model_validate_json(_read_bound_bytes(binding, label=label))
    except ValueError as exc:
        raise ValueError(f"Invalid {label}: {exc}") from exc


def _load_tree_model[ModelT: BaseModel](
    tree: _HeldAttemptTree,
    binding: ExecutionArtifactBinding,
    model: type[ModelT],
    *,
    label: str,
) -> ModelT:
    try:
        return model.model_validate_json(tree.verify_binding(binding, label=label))
    except ValueError as exc:
        raise ValueError(f"Invalid {label}: {exc}") from exc


def _focused_name(leaf_id: TextureFocusedLeafId, suffix: str) -> str:
    operation = leaf_id.removeprefix("texture.").removesuffix(".v1").replace("-", "_")
    return f"texture_{operation}_{suffix}.json"


def _validate_focused_selection(preparation: TexturePreparationPacket) -> None:
    operations = preparation.request.operations
    expected = (
        ("propose", "not_requested"),
        ("generate", "requested"),
        ("evidence", "requested"),
        ("critique", "not_requested"),
        ("review", "requested"),
        ("publish", "requested"),
    )
    observed = (
        ("propose", operations.state_for("propose")),
        ("generate", operations.state_for("generate")),
        ("evidence", operations.state_for("evidence")),
        ("critique", operations.state_for("critique")),
        ("review", operations.state_for("review")),
        ("publish", operations.state_for("publish")),
    )
    if observed != expected:
        raise ValueError(
            "Texture focused graph requires prepare/apply-provided/evidence/review/"
            "publish with proposal and critique not requested"
        )


def _read_preparation(
    binding: ExecutionArtifactBinding,
) -> TexturePreparationPacket:
    preparation = _load_bound_model(
        binding,
        TexturePreparationPacket,
        label="Texture preparation",
    )
    _validate_focused_selection(preparation)
    return preparation


def _validate_uv_prepare_dependency(
    binding: ExecutionArtifactBinding,
    request: TextureCapabilityRequest,
) -> TextureUvLeafResult:
    """Recompute and bind the exact passing UV result consumed by prepare."""

    root = Path(binding.path).parent
    with _HeldDirectory.open(root) as attempt:
        observed_binding, result = _load_contained_model(
            attempt,
            binding.path,
            TextureUvLeafResult,
            label="Texture UV dependency result",
        )
        invocation_binding, invocation = _load_contained_model(
            attempt,
            result.invocation.path,
            TextureUvLeafInvocation,
            label="Texture UV dependency invocation",
        )
        if observed_binding != binding or invocation_binding != result.invocation:
            raise ValueError("Texture prepare UV dependency binding is stale")
        _validate_native_contract(
            attempt,
            invocation_binding=invocation_binding,
            invocation=invocation,
            result_binding=observed_binding,
            result=result,
        )
    if result.native_disposition != "passed":
        raise ValueError("Texture prepare requires a passing UV dependency")
    if request.source != result.output or _binding_identities(
        request.source_dependencies
    ) != _binding_identities(result.output_dependencies):
        raise ValueError(
            "Texture prepare source and dependency closure differ from UV output"
        )
    return result


def _validate_apply_provided_chain(
    *,
    preparation: TexturePreparationPacket,
    preparation_binding: ExecutionArtifactBinding,
    outer_plan: TextureOuterPlan,
    outer_plan_binding: ExecutionArtifactBinding,
    generation: TextureGenerationPacket,
) -> None:
    """Recompute the provider-free plan and generation identities."""

    validate_texture_outer_plan(
        outer_plan,
        preparation=preparation,
        preparation_binding=preparation_binding,
    )
    expected_inputs = tuple(target.generator_inputs for target in outer_plan.targets)
    if (
        outer_plan.execution_mode != "apply_provided"
        or outer_plan.advisory_proposal is not None
        or generation.preparation != preparation_binding
        or generation.outer_plan != outer_plan_binding
        or generation.provider_proposal is not None
        or generation.execution_mode != "apply_provided"
        or generation.generator_provider != "outer_provided_image_apply"
        or generation.generator_capability != TEXTURE_APPLY_PROVIDED_LEAF_ID
        or generation.generator_inputs != expected_inputs
        or generation.provided_images != outer_plan.provided_images
        or generation.reference_artifacts != outer_plan.reference_artifacts
        or generation.execution.requested_unit_ids != outer_plan.target_unit_ids
        or tuple(item.unit_id for item in generation.unit_artifacts)
        != outer_plan.target_unit_ids
        or generation.unit_artifacts != generation.execution.unit_artifacts
        or generation.candidate.path
        != str(Path(generation.execution.output_asset_path).expanduser().resolve())
        or generation.operation_status.outcome("generate").state != "completed"
        or generation.execution.metadata.get("provider_invoked") is not False
        or generation.execution.metadata.get("model_invoked") is not False
        or generation.execution.metadata.get("live_backend_invoked") is not False
    ):
        raise ValueError("Texture apply-provided chain identities are inconsistent")
    _verify_generation_artifacts(generation)


def _validate_evidence_chain(
    *,
    preparation: TexturePreparationPacket,
    preparation_binding: ExecutionArtifactBinding,
    outer_plan: TextureOuterPlan,
    outer_plan_binding: ExecutionArtifactBinding,
    generation: TextureGenerationPacket,
    generation_binding: ExecutionArtifactBinding,
    evidence: TextureCandidateEvidencePacket,
) -> None:
    """Recompute the exact evidence packet over the provider-free candidate."""

    _validate_apply_provided_chain(
        preparation=preparation,
        preparation_binding=preparation_binding,
        outer_plan=outer_plan,
        outer_plan_binding=outer_plan_binding,
        generation=generation,
    )
    if (
        evidence.preparation != preparation_binding
        or evidence.outer_plan != outer_plan_binding
        or evidence.generation != generation_binding
        or evidence.source != preparation.request.source
        or evidence.source_dependencies != preparation.request.source_dependencies
        or evidence.candidate != generation.candidate
        or evidence.reference_artifacts != outer_plan.reference_artifacts
        or evidence.provided_images != outer_plan.provided_images
        or tuple(item.unit_id for item in evidence.unit_evidence)
        != outer_plan.target_unit_ids
        or evidence.operation_status.outcome("evidence").state != "completed"
        or not evidence.renderer_metadata
    ):
        raise ValueError("Texture evidence chain identities are inconsistent")
    _verify_candidate_evidence_artifacts(evidence)


def _verify_stage_binding(
    binding: ExecutionArtifactBinding,
    dependencies: tuple[ArtifactBinding, ...],
    *,
    label: str,
) -> None:
    _verify_binding(binding, label=label)
    verify_usd_dependency_closure(binding.path, dependencies)


def _scope_readback(
    *,
    source: ExecutionArtifactBinding,
    source_dependencies: tuple[ArtifactBinding, ...],
    output: ExecutionArtifactBinding,
    output_dependencies: tuple[ArtifactBinding, ...],
    preparation: TexturePreparationPacket,
) -> TextureScopeInvariantReport:
    _verify_stage_binding(source, source_dependencies, label="Texture focused source")
    _verify_stage_binding(output, output_dependencies, label="Texture focused output")
    report = validate_texture_scope_invariants(
        source_asset_path=source.path,
        output_asset_path=output.path,
        plan=preparation.scope_plan,
    )
    return report


def _scope_rejection_detail(report: TextureScopeInvariantReport) -> str | None:
    """Describe a deterministic scope rejection without losing its report."""

    if report.passed:
        return None
    violations = "; ".join(
        f"{item.code} at {item.prim_path}: {item.summary}" for item in report.violations
    )
    if not violations:
        violations = (
            "one or more preservation facts were false "
            f"(geometry={report.geometry_unchanged}, "
            f"non_target_materials={report.non_target_materials_unchanged}, "
            f"bindings={report.bindings_unchanged}, "
            "structure_outside_target="
            f"{report.structure_unchanged_outside_target})"
        )
    return f"Texture focused saved-stage scope readback rejected: {violations}"


@dataclass(frozen=True)
class _FocusedFailureContract:
    source: ExecutionArtifactBinding
    source_dependencies: tuple[ArtifactBinding, ...]
    output: ExecutionArtifactBinding
    output_dependencies: tuple[ArtifactBinding, ...]
    packet_chain: tuple[ExecutionArtifactBinding, ...]
    preparation: TexturePreparationPacket | None = None


def _focused_failure_contract(
    invocation: TextureFocusedLeafInvocation,
) -> _FocusedFailureContract:
    """Recompute the only failure custody allowed for one focused invocation."""

    leaf_id = invocation.leaf_id
    if leaf_id == TEXTURE_PREPARE_LEAF_ID:
        assert invocation.request is not None
        assert invocation.uv_result is not None
        _validate_uv_prepare_dependency(invocation.uv_result, invocation.request)
        return _FocusedFailureContract(
            source=invocation.request.source,
            source_dependencies=invocation.request.source_dependencies,
            output=invocation.request.source,
            output_dependencies=invocation.request.source_dependencies,
            packet_chain=(invocation.uv_result,),
        )

    if leaf_id == TEXTURE_APPLY_PROVIDED_LEAF_ID:
        assert invocation.preparation is not None
        assert invocation.outer_plan is not None
        preparation = _read_preparation(invocation.preparation)
        outer_plan = _load_bound_model(
            invocation.outer_plan,
            TextureOuterPlan,
            label="outer Texture plan",
        )
        validate_texture_outer_plan(
            outer_plan,
            preparation=preparation,
            preparation_binding=invocation.preparation,
        )
        if (
            outer_plan.execution_mode != "apply_provided"
            or outer_plan.advisory_proposal is not None
        ):
            raise ValueError(
                "Texture apply-provided failure changed the exact outer plan"
            )
        return _FocusedFailureContract(
            source=preparation.request.source,
            source_dependencies=preparation.request.source_dependencies,
            output=preparation.request.source,
            output_dependencies=preparation.request.source_dependencies,
            packet_chain=(invocation.preparation, invocation.outer_plan),
            preparation=preparation,
        )

    if leaf_id == TEXTURE_EVIDENCE_LEAF_ID:
        assert invocation.preparation is not None
        assert invocation.outer_plan is not None
        assert invocation.generation is not None
        preparation = _read_preparation(invocation.preparation)
        outer_plan = _load_bound_model(
            invocation.outer_plan,
            TextureOuterPlan,
            label="outer Texture plan",
        )
        generation = _load_bound_model(
            invocation.generation,
            TextureGenerationPacket,
            label="Texture generation",
        )
        _validate_apply_provided_chain(
            preparation=preparation,
            preparation_binding=invocation.preparation,
            outer_plan=outer_plan,
            outer_plan_binding=invocation.outer_plan,
            generation=generation,
        )
        return _FocusedFailureContract(
            source=preparation.request.source,
            source_dependencies=preparation.request.source_dependencies,
            output=generation.candidate,
            output_dependencies=tuple(
                bind_usd_dependency_closure(generation.candidate.path)
            ),
            packet_chain=(
                invocation.preparation,
                invocation.outer_plan,
                invocation.generation,
            ),
            preparation=preparation,
        )

    assert invocation.outer_plan is not None
    assert invocation.generation is not None
    assert invocation.candidate_evidence is not None
    outer_plan = _load_bound_model(
        invocation.outer_plan,
        TextureOuterPlan,
        label="outer Texture plan",
    )
    generation = _load_bound_model(
        invocation.generation,
        TextureGenerationPacket,
        label="Texture generation",
    )
    evidence = _load_bound_model(
        invocation.candidate_evidence,
        TextureCandidateEvidencePacket,
        label="Texture candidate evidence",
    )
    preparation = _read_preparation(generation.preparation)
    _validate_evidence_chain(
        preparation=preparation,
        preparation_binding=generation.preparation,
        outer_plan=outer_plan,
        outer_plan_binding=invocation.outer_plan,
        generation=generation,
        generation_binding=invocation.generation,
        evidence=evidence,
    )

    if leaf_id == TEXTURE_REVIEW_LEAF_ID:
        assert invocation.review_input is not None
        review_input = _load_bound_model(
            invocation.review_input,
            TextureOuterReviewInput,
            label="Texture outer review input",
        )
        if (
            review_input.outer_plan != invocation.outer_plan
            or review_input.generation != invocation.generation
            or review_input.candidate_evidence != invocation.candidate_evidence
            or review_input.candidate != generation.candidate
            or review_input.reference_artifacts != outer_plan.reference_artifacts
            or review_input.provided_images != outer_plan.provided_images
        ):
            raise ValueError("Texture review failure changed its exact input chain")
        packet_chain = (
            invocation.outer_plan,
            invocation.generation,
            invocation.candidate_evidence,
            invocation.review_input,
        )
    else:
        assert leaf_id == TEXTURE_PUBLISH_LEAF_ID
        assert invocation.request_binding is not None
        assert invocation.preparation is not None
        assert invocation.outer_review is not None
        request = _load_bound_model(
            invocation.request_binding,
            TextureCapabilityRequest,
            label="Texture capability request",
        )
        review = _load_bound_model(
            invocation.outer_review,
            TextureOuterReviewPacket,
            label="Texture outer review",
        )
        if (
            request != preparation.request
            or invocation.preparation != generation.preparation
            or review.outer_plan != invocation.outer_plan
            or review.generation != invocation.generation
            or review.candidate_evidence != invocation.candidate_evidence
            or review.candidate != generation.candidate
            or any(item.disposition != "accept" for item in review.unit_reviews)
        ):
            raise ValueError("Texture publish failure changed its exact input chain")
        packet_chain = (
            invocation.request_binding,
            invocation.preparation,
            invocation.outer_plan,
            invocation.generation,
            invocation.candidate_evidence,
            invocation.outer_review,
        )
    return _FocusedFailureContract(
        source=preparation.request.source,
        source_dependencies=preparation.request.source_dependencies,
        output=preparation.request.source,
        output_dependencies=preparation.request.source_dependencies,
        packet_chain=packet_chain,
        preparation=preparation,
    )


def _focused_native_contract(
    attempt: _HeldDirectory,
    invocation_binding: ExecutionArtifactBinding,
    invocation: TextureFocusedLeafInvocation,
    native_packet_binding: ExecutionArtifactBinding,
) -> _FocusedNativeContract:
    leaf_id = invocation.leaf_id
    native_payload = _read_bound_bytes(
        native_packet_binding,
        label="Texture focused native packet",
    )
    try:
        native_document = json.loads(native_payload)
    except json.JSONDecodeError as exc:
        raise ValueError("Texture focused native packet is not valid JSON") from exc
    if (
        isinstance(native_document, dict)
        and native_document.get("schema_version")
        == TEXTURE_FOCUSED_FAILURE_SCHEMA_VERSION
    ):
        failure = TextureFocusedNativeFailure.model_validate(native_document)
        if failure.leaf_id != leaf_id or failure.invocation != invocation_binding:
            raise ValueError(
                "Texture focused native failure differs from its exact invocation"
            )
        expected_failure = _focused_failure_contract(invocation)
        if (
            failure.source != expected_failure.source
            or failure.source_dependencies != expected_failure.source_dependencies
            or failure.output != expected_failure.output
            or failure.output_dependencies != expected_failure.output_dependencies
            or failure.packet_chain != expected_failure.packet_chain
        ):
            raise ValueError(
                f"Texture {leaf_id} failure changed its exact source, output, "
                "or upstream chain"
            )
        _verify_stage_binding(
            failure.source,
            failure.source_dependencies,
            label="Texture focused failed source",
        )
        _verify_stage_binding(
            failure.output,
            failure.output_dependencies,
            label="Texture focused failed attempted output",
        )
        for index, binding in enumerate(failure.packet_chain, start=1):
            _read_bound_bytes(
                binding,
                label=f"Texture focused failed upstream packet {index}",
            )
        renderer_receipts = _renderer_receipt_evidence(attempt)
        if (
            failure.renderer_evidence != renderer_receipts.bindings
            or failure.renderer_invoked != renderer_receipts.renderer_invoked
            or failure.renderer_receipt_custody_error != renderer_receipts.custody_error
        ):
            raise ValueError(
                "Texture focused native failure renderer receipt custody changed"
            )
        report: TextureScopeInvariantReport | None = None
        if leaf_id == TEXTURE_EVIDENCE_LEAF_ID:
            assert expected_failure.preparation is not None
            report = _scope_readback(
                source=failure.source,
                source_dependencies=failure.source_dependencies,
                output=failure.output,
                output_dependencies=failure.output_dependencies,
                preparation=expected_failure.preparation,
            )
        error = failure.error
        if renderer_receipts.custody_error is not None:
            error = f"{error}; {renderer_receipts.custody_error}"
        return _FocusedNativeContract(
            source=failure.source,
            source_dependencies=failure.source_dependencies,
            output=failure.output,
            output_dependencies=failure.output_dependencies,
            packet_chain=(*failure.packet_chain, native_packet_binding),
            evidence=(native_packet_binding, *renderer_receipts.bindings),
            native_disposition="failed",
            error=error,
            renderer_invoked=renderer_receipts.renderer_invoked,
            scope_invariant_report=report,
        )
    evidence: tuple[ExecutionArtifactBinding, ...]
    chain: tuple[ExecutionArtifactBinding, ...]
    disposition: Literal["passed", "failed", "not_evaluated"]
    error: str | None = None
    if leaf_id == TEXTURE_PREPARE_LEAF_ID:
        assert invocation.uv_result is not None
        preparation_packet = _load_bound_model(
            native_packet_binding,
            TexturePreparationPacket,
            label="Texture preparation result",
        )
        if (
            invocation.request is None
            or preparation_packet.request != invocation.request
        ):
            raise ValueError("Texture prepare result differs from its invocation")
        _validate_uv_prepare_dependency(
            invocation.uv_result,
            preparation_packet.request,
        )
        _validate_focused_selection(preparation_packet)
        if preparation_packet.operation_status.outcome("inspect").state != "completed":
            raise ValueError("Texture preparation did not complete inspection")
        source = preparation_packet.request.source
        source_dependencies = preparation_packet.request.source_dependencies
        output = source
        output_dependencies = source_dependencies
        evidence = (
            native_packet_binding,
            *preparation_packet.inspection.before_render_artifacts,
            *preparation_packet.inspection.inspection_artifacts,
        )
        chain = (invocation.uv_result, native_packet_binding)
        preparation = preparation_packet
        disposition = "passed"
        renderer_invoked = True
    elif leaf_id == TEXTURE_APPLY_PROVIDED_LEAF_ID:
        assert invocation.preparation is not None
        assert invocation.outer_plan is not None
        preparation = _read_preparation(invocation.preparation)
        outer_plan = _load_bound_model(
            invocation.outer_plan,
            TextureOuterPlan,
            label="outer Texture plan",
        )
        generation_packet = _load_bound_model(
            native_packet_binding,
            TextureGenerationPacket,
            label="Texture apply-provided result",
        )
        _validate_apply_provided_chain(
            preparation=preparation,
            preparation_binding=invocation.preparation,
            outer_plan=outer_plan,
            outer_plan_binding=invocation.outer_plan,
            generation=generation_packet,
        )
        source = preparation.request.source
        source_dependencies = preparation.request.source_dependencies
        output = generation_packet.candidate
        output_dependencies = tuple(bind_usd_dependency_closure(output.path))
        evidence = (native_packet_binding,)
        chain = (invocation.preparation, invocation.outer_plan, native_packet_binding)
        disposition = "passed"
        renderer_invoked = False
    elif leaf_id == TEXTURE_EVIDENCE_LEAF_ID:
        assert invocation.preparation is not None
        assert invocation.outer_plan is not None
        assert invocation.generation is not None
        preparation = _read_preparation(invocation.preparation)
        outer_plan = _load_bound_model(
            invocation.outer_plan,
            TextureOuterPlan,
            label="outer Texture plan",
        )
        generation = _load_bound_model(
            invocation.generation,
            TextureGenerationPacket,
            label="Texture generation",
        )
        candidate_packet = _load_bound_model(
            native_packet_binding,
            TextureCandidateEvidencePacket,
            label="Texture evidence result",
        )
        _validate_evidence_chain(
            preparation=preparation,
            preparation_binding=invocation.preparation,
            outer_plan=outer_plan,
            outer_plan_binding=invocation.outer_plan,
            generation=generation,
            generation_binding=invocation.generation,
            evidence=candidate_packet,
        )
        source = candidate_packet.source
        source_dependencies = candidate_packet.source_dependencies
        output = candidate_packet.candidate
        output_dependencies = tuple(bind_usd_dependency_closure(output.path))
        evidence = (
            native_packet_binding,
            *(
                item
                for unit in candidate_packet.unit_evidence
                for item in unit.source_images
            ),
            *(
                item
                for unit in candidate_packet.unit_evidence
                for item in unit.candidate_images
            ),
            *candidate_packet.static_evidence,
        )
        chain = (
            invocation.preparation,
            invocation.outer_plan,
            invocation.generation,
            native_packet_binding,
        )
        disposition = "passed"
        renderer_invoked = True
    elif leaf_id == TEXTURE_REVIEW_LEAF_ID:
        assert invocation.outer_plan is not None
        assert invocation.generation is not None
        assert invocation.candidate_evidence is not None
        assert invocation.review_input is not None
        outer_plan = _load_bound_model(
            invocation.outer_plan,
            TextureOuterPlan,
            label="outer Texture plan",
        )
        generation = _load_bound_model(
            invocation.generation,
            TextureGenerationPacket,
            label="Texture generation",
        )
        evidence_packet = _load_bound_model(
            invocation.candidate_evidence,
            TextureCandidateEvidencePacket,
            label="Texture candidate evidence",
        )
        review_input = _load_bound_model(
            invocation.review_input,
            TextureOuterReviewInput,
            label="Texture outer review input",
        )
        preparation = _read_preparation(generation.preparation)
        review_packet = _load_bound_model(
            native_packet_binding,
            TextureOuterReviewPacket,
            label="Texture review result",
        )
        _validate_evidence_chain(
            preparation=preparation,
            preparation_binding=generation.preparation,
            outer_plan=outer_plan,
            outer_plan_binding=invocation.outer_plan,
            generation=generation,
            generation_binding=invocation.generation,
            evidence=evidence_packet,
        )
        if (
            review_packet.model_dump(
                mode="python",
                exclude={"schema_version", "operation_status"},
            )
            != review_input.model_dump(mode="python", exclude={"schema_version"})
            or review_packet.outer_plan != invocation.outer_plan
            or review_packet.generation != invocation.generation
            or review_packet.candidate_evidence != invocation.candidate_evidence
            or review_packet.advisory_critique is not None
            or outer_plan.operations.critique != "not_requested"
            or review_packet.candidate != generation.candidate
            or review_packet.candidate != evidence_packet.candidate
            or review_packet.operation_status.outcome("review").state != "completed"
        ):
            raise ValueError(
                "Texture review native packet changed outer decision facts"
            )
        source = preparation.request.source
        source_dependencies = preparation.request.source_dependencies
        output = review_packet.candidate
        output_dependencies = tuple(bind_usd_dependency_closure(output.path))
        evidence = (native_packet_binding,)
        chain = (
            invocation.outer_plan,
            invocation.generation,
            invocation.candidate_evidence,
            invocation.review_input,
            native_packet_binding,
        )
        nonaccepted = tuple(
            item for item in review_packet.unit_reviews if item.disposition != "accept"
        )
        disposition = "failed" if nonaccepted else "passed"
        if nonaccepted:
            error = "Texture outer review did not accept every unit: " + "; ".join(
                f"{item.unit_id}: {item.disposition}: {item.rationale}"
                for item in nonaccepted
            )
        renderer_invoked = False
    else:
        assert leaf_id == TEXTURE_PUBLISH_LEAF_ID
        assert invocation.request_binding is not None
        assert invocation.preparation is not None
        assert invocation.outer_plan is not None
        assert invocation.generation is not None
        assert invocation.candidate_evidence is not None
        assert invocation.outer_review is not None
        preparation = _read_preparation(invocation.preparation)
        outer_plan = _load_bound_model(
            invocation.outer_plan,
            TextureOuterPlan,
            label="outer Texture plan",
        )
        generation = _load_bound_model(
            invocation.generation,
            TextureGenerationPacket,
            label="Texture generation",
        )
        evidence_packet = _load_bound_model(
            invocation.candidate_evidence,
            TextureCandidateEvidencePacket,
            label="Texture candidate evidence",
        )
        review = _load_bound_model(
            invocation.outer_review,
            TextureOuterReviewPacket,
            label="Texture outer review",
        )
        publication_packet = _load_bound_model(
            native_packet_binding,
            TexturePublicationReceipt,
            label="Texture publication result",
        )
        _validate_evidence_chain(
            preparation=preparation,
            preparation_binding=invocation.preparation,
            outer_plan=outer_plan,
            outer_plan_binding=invocation.outer_plan,
            generation=generation,
            generation_binding=invocation.generation,
            evidence=evidence_packet,
        )
        request = _load_bound_model(
            invocation.request_binding,
            TextureCapabilityRequest,
            label="Texture capability request",
        )
        if (
            request != preparation.request
            or publication_packet.request != invocation.request_binding
            or publication_packet.preparation != invocation.preparation
            or publication_packet.outer_plan != invocation.outer_plan
            or publication_packet.generation != invocation.generation
            or publication_packet.candidate_evidence != invocation.candidate_evidence
            or publication_packet.outer_review != invocation.outer_review
            or publication_packet.source != preparation.request.source
            or publication_packet.source_dependencies
            != preparation.request.source_dependencies
            or publication_packet.accepted_candidate != generation.candidate
            or publication_packet.accepted_candidate != evidence_packet.candidate
            or review.candidate != publication_packet.accepted_candidate
            or review.outer_plan != invocation.outer_plan
            or review.generation != invocation.generation
            or review.candidate_evidence != invocation.candidate_evidence
            or review.reference_artifacts != outer_plan.reference_artifacts
            or review.provided_images != outer_plan.provided_images
            or any(item.disposition != "accept" for item in review.unit_reviews)
            or review.operation_status.outcome("review").state != "completed"
            or publication_packet.reference_artifacts != outer_plan.reference_artifacts
            or publication_packet.provided_images != outer_plan.provided_images
            or invocation.output_asset_path != publication_packet.published_asset.path
            or publication_packet.operation_status.outcome("publish").state
            != "completed"
        ):
            raise ValueError(
                "Texture publication native packet changed chain identities"
            )
        source = publication_packet.source
        source_dependencies = publication_packet.source_dependencies
        output = publication_packet.published_asset
        output_dependencies = tuple(bind_usd_dependency_closure(output.path))
        evidence = (native_packet_binding, *publication_packet.verification_artifacts)
        chain = (
            invocation.request_binding,
            invocation.preparation,
            invocation.outer_plan,
            invocation.generation,
            invocation.candidate_evidence,
            invocation.outer_review,
            native_packet_binding,
        )
        disposition = "passed"
        renderer_invoked = False

    report = _scope_readback(
        source=source,
        source_dependencies=source_dependencies,
        output=output,
        output_dependencies=output_dependencies,
        preparation=preparation,
    )
    scope_rejection = _scope_rejection_detail(report)
    if scope_rejection is not None:
        disposition = "failed"
        error = f"{error}; {scope_rejection}" if error is not None else scope_rejection
    return _FocusedNativeContract(
        source=source,
        source_dependencies=source_dependencies,
        output=output,
        output_dependencies=output_dependencies,
        packet_chain=chain,
        evidence=evidence,
        native_disposition=disposition,
        error=error,
        renderer_invoked=renderer_invoked,
        scope_invariant_report=report,
    )


def _write_focused_native_result(
    attempt: _HeldDirectory,
    *,
    invocation_binding: ExecutionArtifactBinding,
    invocation: TextureFocusedLeafInvocation,
    native_packet_binding: ExecutionArtifactBinding,
) -> TextureFocusedLeafResult:
    tree = _HeldAttemptTree(attempt)
    contract = _focused_native_contract(
        attempt,
        invocation_binding,
        invocation,
        native_packet_binding,
    )
    for index, binding in enumerate(contract.evidence, start=1):
        tree.verify_binding(binding, label=f"Texture focused evidence {index}")
    report = contract.scope_invariant_report
    common_readback = {
        "leaf_id": invocation.leaf_id,
        "invocation": invocation_binding,
        "native_packet": native_packet_binding,
        "source": contract.source,
        "source_dependencies": contract.source_dependencies,
        "saved_stage": contract.output,
        "saved_stage_dependencies": contract.output_dependencies,
        "packet_chain": contract.packet_chain,
        "reopened_saved_stage": True,
        "dependency_closure_verified": True,
        "provider_status": "not_requested",
        "provider_invoked": False,
    }
    if report is None:
        assert contract.error is not None
        readback: BaseModel = TextureFocusedFailureReadback.create(
            **common_readback,
            error=contract.error,
            non_target_topology_preserved=True,
            non_target_physics_preserved=True,
            non_target_material_preserved=True,
        )
    else:
        readback = TextureFocusedSavedStageReadback.create(
            **common_readback,
            scope_invariant_report=report,
            non_target_topology_preserved=(
                report.geometry_unchanged and report.structure_unchanged_outside_target
            ),
            non_target_physics_preserved=(
                report.geometry_unchanged and report.structure_unchanged_outside_target
            ),
            non_target_material_preserved=(
                report.non_target_materials_unchanged and report.bindings_unchanged
            ),
        )
    readback_name = _focused_name(invocation.leaf_id, "saved_stage_readback")
    result_name = _focused_name(invocation.leaf_id, "leaf_result")
    receipt_name = _focused_name(invocation.leaf_id, "terminal_receipt")
    for name in (readback_name, result_name, receipt_name):
        if attempt.exists(name):
            raise FileExistsError(
                f"Texture focused leaf refuses to replace artifact: {attempt.path / name}"
            )
    readback_identity = attempt.write_json(readback_name, readback)
    readback_binding = attempt.binding(readback_name, identity=readback_identity)
    result = TextureFocusedLeafResult(
        leaf_id=invocation.leaf_id,
        invocation=invocation_binding,
        native_packet=native_packet_binding,
        native_disposition=contract.native_disposition,
        error=contract.error,
        source=contract.source,
        output=contract.output,
        output_dependencies=contract.output_dependencies,
        evidence=contract.evidence,
        saved_stage_readbacks=(readback_binding,),
        provider_status="not_requested",
        provider_invoked=False,
        renderer_invoked=contract.renderer_invoked,
    )
    result_identity = attempt.write_json(result_name, result)
    result_binding = attempt.binding(result_name, identity=result_identity)
    receipt = TextureFocusedTerminalReceipt.create(
        leaf_id=invocation.leaf_id,
        invocation=invocation_binding,
        result=result_binding,
        native_packet=native_packet_binding,
        saved_stage_readback=readback_binding,
        native_disposition=result.native_disposition,
        error=result.error,
        output=result.output,
        provider_status="not_requested",
        provider_invoked=False,
        renderer_invoked=result.renderer_invoked,
        resources_released=True,
    )
    receipt_identity = attempt.write_json(receipt_name, receipt)
    receipt_binding = attempt.binding(receipt_name, identity=receipt_identity)
    attempt.verify_binding(result_binding, label="Texture focused native result")
    attempt.verify_binding(receipt_binding, label="Texture focused terminal receipt")
    attempt.verify_path()
    return result


def _write_focused_native_failure(
    attempt: _HeldDirectory,
    native: _HeldDirectory,
    *,
    invocation_binding: ExecutionArtifactBinding,
    invocation: TextureFocusedLeafInvocation,
    error: str,
) -> TextureFocusedLeafResult:
    """Seal one exact failed native result without relabeling it as success."""

    expected = _focused_failure_contract(invocation)
    _verify_stage_binding(
        expected.source,
        expected.source_dependencies,
        label="Texture focused failed source",
    )
    _verify_stage_binding(
        expected.output,
        expected.output_dependencies,
        label="Texture focused failed attempted output",
    )
    renderer_receipts = _renderer_receipt_evidence(attempt)
    failure = TextureFocusedNativeFailure.create(
        leaf_id=invocation.leaf_id,
        invocation=invocation_binding,
        source=expected.source,
        source_dependencies=expected.source_dependencies,
        output=expected.output,
        output_dependencies=expected.output_dependencies,
        packet_chain=expected.packet_chain,
        renderer_evidence=renderer_receipts.bindings,
        renderer_receipt_custody_error=renderer_receipts.custody_error,
        error=error,
        provider_status="not_requested",
        provider_invoked=False,
        renderer_invoked=renderer_receipts.renderer_invoked,
        resources_released=True,
    )
    failure_name = _focused_name(invocation.leaf_id, "native_failure")
    if native.exists(failure_name):
        raise FileExistsError(
            f"Texture focused leaf refuses to replace artifact: "
            f"{native.path / failure_name}"
        )
    failure_identity = native.write_json(failure_name, failure)
    failure_binding = native.binding(failure_name, identity=failure_identity)
    native.verify_binding(failure_binding, label="Texture focused native failure")
    attempt.verify_path()
    return _write_focused_native_result(
        attempt,
        invocation_binding=invocation_binding,
        invocation=invocation,
        native_packet_binding=failure_binding,
    )


def _run_focused_operation(
    attempt: _HeldDirectory,
    *,
    invocation_binding: ExecutionArtifactBinding,
    invocation: TextureFocusedLeafInvocation,
    operation: Callable[[Path], ExecutionArtifactBinding],
) -> TextureFocusedLeafResult:
    """Run one operation below a descriptor-held native root and seal exceptions."""

    with attempt.claim_directory("native", mode=0o700) as native:
        attempt.verify_path()
        native.verify_path()
        try:
            native_packet_binding = operation(native.descriptor_path)
            native.verify_path()
            attempt.verify_path()
            return _write_focused_native_result(
                attempt,
                invocation_binding=invocation_binding,
                invocation=invocation,
                native_packet_binding=native_packet_binding,
            )
        except Exception as exc:
            native.verify_path()
            attempt.verify_path()
            detail = str(exc).strip() or "no exception detail"
            return _write_focused_native_failure(
                attempt,
                native,
                invocation_binding=invocation_binding,
                invocation=invocation,
                error=f"{type(exc).__name__}: {detail}",
            )


def _load_focused_invocation(
    attempt: _HeldDirectory,
    invocation_path: str | Path,
    *,
    expected_leaf_id: TextureFocusedLeafId,
) -> tuple[ExecutionArtifactBinding, TextureFocusedLeafInvocation]:
    binding, invocation = _load_contained_model(
        attempt,
        invocation_path,
        TextureFocusedLeafInvocation,
        label="Texture focused invocation",
    )
    if Path(binding.path).parent != attempt.path:
        raise ValueError("Texture focused invocation must be a direct attempt artifact")
    if invocation.attempt_root != str(attempt.path):
        raise ValueError("Texture focused invocation changed its attempt root")
    if invocation.leaf_id != expected_leaf_id:
        raise ValueError(
            f"Texture focused entrypoint {expected_leaf_id} rejects {invocation.leaf_id}"
        )
    native_root = attempt.path / "native"
    if attempt.exists("native"):
        raise FileExistsError(
            f"Texture focused native output root already exists: {native_root}"
        )
    return binding, invocation


def _focused_invocation_location(
    invocation_path: str | Path,
) -> tuple[Path, Path]:
    """Absolutize without following the invocation artifact itself."""

    candidate = Path(invocation_path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return candidate, candidate.parent.resolve(strict=True)


def run_texture_prepare_asset_leaf(
    invocation_path: str | Path,
    *,
    inspector: Any,
) -> TextureFocusedLeafResult:
    """Execute only provider-free preparation from an explicit graph invocation."""

    invocation_candidate, root = _focused_invocation_location(invocation_path)
    with _HeldDirectory.open(root) as attempt:
        binding, invocation = _load_focused_invocation(
            attempt,
            invocation_candidate,
            expected_leaf_id=TEXTURE_PREPARE_LEAF_ID,
        )
        assert invocation.request is not None
        assert invocation.uv_result is not None
        _validate_uv_prepare_dependency(invocation.uv_result, invocation.request)
        from .capabilities import prepare_texture_scope

        def execute(native_root: Path) -> ExecutionArtifactBinding:
            _packet, native_binding = prepare_texture_scope(
                invocation.request,
                inspector=inspector,
                output_dir=native_root,
                output_dir_claimed=True,
            )
            return native_binding

        return _run_focused_operation(
            attempt,
            invocation_binding=binding,
            invocation=invocation,
            operation=execute,
        )


def run_texture_apply_provided_asset_leaf(
    invocation_path: str | Path,
) -> TextureFocusedLeafResult:
    """Execute only deterministic apply-provided from an explicit graph input."""

    invocation_candidate, root = _focused_invocation_location(invocation_path)
    with _HeldDirectory.open(root) as attempt:
        binding, invocation = _load_focused_invocation(
            attempt,
            invocation_candidate,
            expected_leaf_id=TEXTURE_APPLY_PROVIDED_LEAF_ID,
        )
        assert invocation.preparation is not None
        assert invocation.outer_plan is not None
        preparation = _read_preparation(invocation.preparation)
        outer_plan = _load_bound_model(
            invocation.outer_plan,
            TextureOuterPlan,
            label="outer Texture plan",
        )
        from .capabilities import invoke_texture_generator
        from .provided_image_apply import ProvidedImageTextureApplyLeaf

        def execute(native_root: Path) -> ExecutionArtifactBinding:
            _packet, native_binding = invoke_texture_generator(
                outer_plan,
                outer_plan_binding=invocation.outer_plan,
                preparation=preparation,
                preparation_binding=invocation.preparation,
                generator=ProvidedImageTextureApplyLeaf(),
                output_dir=native_root,
                output_dir_claimed=True,
            )
            return native_binding

        return _run_focused_operation(
            attempt,
            invocation_binding=binding,
            invocation=invocation,
            operation=execute,
        )


def run_texture_evidence_asset_leaf(
    invocation_path: str | Path,
    *,
    collector: Any,
) -> TextureFocusedLeafResult:
    """Execute only deterministic/canonical evidence collection."""

    invocation_candidate, root = _focused_invocation_location(invocation_path)
    with _HeldDirectory.open(root) as attempt:
        binding, invocation = _load_focused_invocation(
            attempt,
            invocation_candidate,
            expected_leaf_id=TEXTURE_EVIDENCE_LEAF_ID,
        )
        assert invocation.preparation is not None
        assert invocation.outer_plan is not None
        assert invocation.generation is not None
        preparation = _read_preparation(invocation.preparation)
        outer_plan = _load_bound_model(
            invocation.outer_plan,
            TextureOuterPlan,
            label="outer Texture plan",
        )
        generation = _load_bound_model(
            invocation.generation,
            TextureGenerationPacket,
            label="Texture generation",
        )
        from .capabilities import collect_texture_candidate_evidence

        def execute(native_root: Path) -> ExecutionArtifactBinding:
            _packet, native_binding = collect_texture_candidate_evidence(
                outer_plan,
                generation,
                preparation=preparation,
                preparation_binding=invocation.preparation,
                outer_plan_binding=invocation.outer_plan,
                generation_binding=invocation.generation,
                collector=collector,
                output_dir=native_root,
                output_dir_claimed=True,
            )
            return native_binding

        return _run_focused_operation(
            attempt,
            invocation_binding=binding,
            invocation=invocation,
            operation=execute,
        )


def run_texture_review_asset_leaf(
    invocation_path: str | Path,
) -> TextureFocusedLeafResult:
    """Execute only exact outer-review recording; never semantic selection."""

    invocation_candidate, root = _focused_invocation_location(invocation_path)
    with _HeldDirectory.open(root) as attempt:
        binding, invocation = _load_focused_invocation(
            attempt,
            invocation_candidate,
            expected_leaf_id=TEXTURE_REVIEW_LEAF_ID,
        )
        assert invocation.outer_plan is not None
        assert invocation.generation is not None
        assert invocation.candidate_evidence is not None
        assert invocation.review_input is not None
        outer_plan = _load_bound_model(
            invocation.outer_plan,
            TextureOuterPlan,
            label="outer Texture plan",
        )
        generation = _load_bound_model(
            invocation.generation,
            TextureGenerationPacket,
            label="Texture generation",
        )
        evidence = _load_bound_model(
            invocation.candidate_evidence,
            TextureCandidateEvidencePacket,
            label="Texture candidate evidence",
        )
        review_input = _load_bound_model(
            invocation.review_input,
            TextureOuterReviewInput,
            label="Texture outer review input",
        )
        _read_preparation(generation.preparation)
        from .capabilities import record_texture_outer_review

        def execute(native_root: Path) -> ExecutionArtifactBinding:
            _packet, native_binding = record_texture_outer_review(
                review_input,
                outer_plan=outer_plan,
                generation=generation,
                evidence=evidence,
                output_dir=native_root,
                output_dir_claimed=True,
            )
            return native_binding

        return _run_focused_operation(
            attempt,
            invocation_binding=binding,
            invocation=invocation,
            operation=execute,
        )


def run_texture_publish_asset_leaf(
    invocation_path: str | Path,
) -> TextureFocusedLeafResult:
    """Execute only accepted publication and exact saved-stage readback."""

    invocation_candidate, root = _focused_invocation_location(invocation_path)
    with _HeldDirectory.open(root) as attempt:
        binding, invocation = _load_focused_invocation(
            attempt,
            invocation_candidate,
            expected_leaf_id=TEXTURE_PUBLISH_LEAF_ID,
        )
        assert invocation.request_binding is not None
        assert invocation.preparation is not None
        assert invocation.outer_plan is not None
        assert invocation.generation is not None
        assert invocation.candidate_evidence is not None
        assert invocation.outer_review is not None
        assert invocation.output_asset_path is not None
        preparation = _read_preparation(invocation.preparation)
        outer_plan = _load_bound_model(
            invocation.outer_plan,
            TextureOuterPlan,
            label="outer Texture plan",
        )
        generation = _load_bound_model(
            invocation.generation,
            TextureGenerationPacket,
            label="Texture generation",
        )
        evidence = _load_bound_model(
            invocation.candidate_evidence,
            TextureCandidateEvidencePacket,
            label="Texture candidate evidence",
        )
        review = _load_bound_model(
            invocation.outer_review,
            TextureOuterReviewPacket,
            label="Texture outer review",
        )
        from .capabilities import publish_texture_candidate

        def execute(native_root: Path) -> ExecutionArtifactBinding:
            publication_path = native_root / Path(invocation.output_asset_path).name
            _packet, native_binding = publish_texture_candidate(
                review,
                review_binding=invocation.outer_review,
                request_binding=invocation.request_binding,
                preparation=preparation,
                preparation_binding=invocation.preparation,
                outer_plan=outer_plan,
                outer_plan_binding=invocation.outer_plan,
                generation=generation,
                generation_binding=invocation.generation,
                evidence=evidence,
                evidence_binding=invocation.candidate_evidence,
                publication_path=publication_path,
                output_dir=native_root,
                output_dir_claimed=True,
            )
            return native_binding

        return _run_focused_operation(
            attempt,
            invocation_binding=binding,
            invocation=invocation,
            operation=execute,
        )


def _verify_focused_result_contract(
    attempt: _HeldDirectory,
    *,
    invocation_binding: ExecutionArtifactBinding,
    invocation: TextureFocusedLeafInvocation,
    result_binding: ExecutionArtifactBinding,
    result: TextureFocusedLeafResult,
    receipt_binding: ExecutionArtifactBinding,
    receipt: TextureFocusedTerminalReceipt,
) -> None:
    tree = _HeldAttemptTree(attempt)
    contract = _focused_native_contract(
        attempt,
        invocation_binding,
        invocation,
        result.native_packet,
    )
    if (
        result.invocation != invocation_binding
        or result.leaf_id != invocation.leaf_id
        or result.native_disposition != contract.native_disposition
        or result.error != contract.error
        or result.source != contract.source
        or result.output != contract.output
        or result.output_dependencies != contract.output_dependencies
        or result.evidence != contract.evidence
        or result.renderer_invoked != contract.renderer_invoked
        or len(result.saved_stage_readbacks) != 1
    ):
        raise ValueError("Texture focused native result differs from recomputed facts")
    for index, binding in enumerate(result.evidence, start=1):
        tree.verify_binding(binding, label=f"Texture focused evidence {index}")
    readback_binding = result.saved_stage_readbacks[0]
    report = contract.scope_invariant_report
    common_readback = {
        "leaf_id": invocation.leaf_id,
        "invocation": invocation_binding,
        "native_packet": result.native_packet,
        "source": contract.source,
        "source_dependencies": contract.source_dependencies,
        "saved_stage": contract.output,
        "saved_stage_dependencies": contract.output_dependencies,
        "packet_chain": contract.packet_chain,
        "reopened_saved_stage": True,
        "dependency_closure_verified": True,
        "provider_status": "not_requested",
        "provider_invoked": False,
    }
    if report is None:
        assert contract.error is not None
        readback = _load_tree_model(
            tree,
            readback_binding,
            TextureFocusedFailureReadback,
            label="Texture focused failure readback",
        )
        expected_readback: BaseModel = TextureFocusedFailureReadback.create(
            **common_readback,
            error=contract.error,
            non_target_topology_preserved=True,
            non_target_physics_preserved=True,
            non_target_material_preserved=True,
        )
    else:
        readback = _load_tree_model(
            tree,
            readback_binding,
            TextureFocusedSavedStageReadback,
            label="Texture focused saved-stage readback",
        )
        expected_readback = TextureFocusedSavedStageReadback.create(
            **common_readback,
            scope_invariant_report=report,
            non_target_topology_preserved=(
                report.geometry_unchanged and report.structure_unchanged_outside_target
            ),
            non_target_physics_preserved=(
                report.geometry_unchanged and report.structure_unchanged_outside_target
            ),
            non_target_material_preserved=(
                report.non_target_materials_unchanged and report.bindings_unchanged
            ),
        )
    if readback != expected_readback:
        raise ValueError(
            "Texture focused saved-stage readback differs from current stage facts"
        )
    if (
        receipt.leaf_id != invocation.leaf_id
        or receipt.invocation != invocation_binding
        or receipt.result != result_binding
        or receipt.native_packet != result.native_packet
        or receipt.saved_stage_readback != readback_binding
        or receipt.native_disposition != result.native_disposition
        or receipt.error != result.error
        or receipt.output != result.output
        or receipt.renderer_invoked != result.renderer_invoked
        or not receipt.resources_released
    ):
        raise ValueError("Texture focused terminal receipt differs from native result")
    attempt.verify_binding(result_binding, label="Texture focused result")
    attempt.verify_binding(receipt_binding, label="Texture focused terminal receipt")


def _project_texture_focused_runtime(
    invocation: BaseModel,
    result: BaseModel,
    context: AssetLeafProjectionContext,
) -> AssetLeafProjectionPayload:
    """Recompute one focused result and return frozen generic projection facts."""

    request = cast(TextureFocusedLeafInvocation, invocation)
    native = cast(TextureFocusedLeafResult, result)
    if request.leaf_id != context.leaf_id or native.leaf_id != context.leaf_id:
        raise ValueError(
            "Texture focused invocation and result leaf IDs must match the "
            "frozen projection context"
        )
    with _HeldDirectory.open(request.attempt_root) as attempt:
        invocation_binding, loaded_invocation = _load_contained_model(
            attempt,
            context.invocation_artifact.path,
            TextureFocusedLeafInvocation,
            label="Texture focused invocation",
        )
        result_binding, loaded_result = _load_contained_model(
            attempt,
            context.result_artifact.path,
            TextureFocusedLeafResult,
            label="Texture focused native result",
        )
        receipt_binding, receipt = _load_contained_model(
            attempt,
            attempt.path / _focused_name(request.leaf_id, "terminal_receipt"),
            TextureFocusedTerminalReceipt,
            label="Texture focused terminal receipt",
        )
        if (
            Path(context.invocation_artifact.path).parent != attempt.path
            or Path(context.result_artifact.path).parent != attempt.path
            or request.attempt_root != str(attempt.path)
            or Path(receipt_binding.path).parent != attempt.path
        ):
            raise ValueError(
                "Texture focused projection artifacts must be direct in the native "
                "attempt root"
            )
        if loaded_invocation != request or loaded_result != native:
            raise ValueError(
                "Texture focused projection models differ from their selected artifacts"
            )
        context.require_native_artifact_chain(
            invocation_artifact=_artifact_binding(native.invocation),
            result_artifact=_artifact_binding(receipt.result),
        )
        _verify_focused_result_contract(
            attempt,
            invocation_binding=invocation_binding,
            invocation=request,
            result_binding=result_binding,
            result=native,
            receipt_binding=receipt_binding,
            receipt=receipt,
        )
        attempt.verify_path()
    renderer_leaf = native.leaf_id in _RENDERER_LEAF_IDS
    renderer_resource = (f"texture.renderer:{native.leaf_id}",) if renderer_leaf else ()
    release_receipts = (_artifact_binding(receipt_binding),) if renderer_leaf else ()
    return AssetLeafProjectionPayload(
        native_disposition=native.native_disposition,
        native_status=native.native_disposition,
        native_terminal_receipt=_artifact_binding(receipt_binding),
        evidence=tuple(_artifact_binding(item) for item in native.evidence),
        saved_stage_readbacks=tuple(
            _artifact_binding(item) for item in native.saved_stage_readbacks
        ),
        resource_claims=renderer_resource,
        resource_release_receipts=release_receipts,
        summary=(
            f"Texture {native.leaf_id} ended with native disposition "
            f"{native.native_disposition}."
        ),
        error=native.error,
    )


def project_texture_focused_verified_operation(
    *,
    attempt_root: str | Path,
    invocation_path: str | Path,
    native_result_path: str | Path,
    native_terminal_receipt_path: str | Path,
) -> TextureFocusedAssetLeafPublication:
    """Reverify one focused result and emit only generic receipt arguments."""

    with _HeldDirectory.open(attempt_root) as attempt:
        invocation_binding, invocation = _load_contained_model(
            attempt,
            invocation_path,
            TextureFocusedLeafInvocation,
            label="Texture focused invocation",
        )
        result_binding, result = _load_contained_model(
            attempt,
            native_result_path,
            TextureFocusedLeafResult,
            label="Texture focused native result",
        )
        receipt_binding, receipt = _load_contained_model(
            attempt,
            native_terminal_receipt_path,
            TextureFocusedTerminalReceipt,
            label="Texture focused terminal receipt",
        )
        if (
            invocation.attempt_root != str(attempt.path)
            or Path(invocation_binding.path).parent != attempt.path
            or Path(result_binding.path).parent != attempt.path
            or Path(receipt_binding.path).parent != attempt.path
        ):
            raise ValueError(
                "Texture focused invocation, result, and receipt must be direct "
                "attempt artifacts"
            )
        _verify_focused_result_contract(
            attempt,
            invocation_binding=invocation_binding,
            invocation=invocation,
            result_binding=result_binding,
            result=result,
            receipt_binding=receipt_binding,
            receipt=receipt,
        )
        descriptor = next(
            item
            for item in texture_asset_leaf_descriptors()
            if item.leaf_id == invocation.leaf_id
        )
        projection = TextureFocusedVerifiedOperationProjection(
            leaf_id=invocation.leaf_id,
            descriptor_digest=descriptor.descriptor_digest,
            invocation=invocation_binding,
            native_result=result_binding,
            native_terminal_receipt=receipt_binding,
            native_disposition=result.native_disposition,
            error=result.error,
            output=result.output,
            output_dependencies=result.output_dependencies,
            evidence=result.evidence,
            saved_stage_readbacks=result.saved_stage_readbacks,
            provider_status="not_requested",
            native_renderer_invoked=result.renderer_invoked,
        )
        projection_name = _focused_name(invocation.leaf_id, "verified_projection")
        publication_name = _focused_name(invocation.leaf_id, "asset_publication")
        for name in (projection_name, publication_name):
            if attempt.exists(name):
                raise FileExistsError(
                    f"Texture focused projector refuses to replace artifact: "
                    f"{attempt.path / name}"
                )
        projection_identity = attempt.write_json(projection_name, projection)
        projection_binding = attempt.binding(
            projection_name,
            identity=projection_identity,
        )
        publication = TextureFocusedAssetLeafPublication(
            leaf_id=invocation.leaf_id,
            descriptor_digest=descriptor.descriptor_digest,
            projection=projection_binding,
            native_result=result_binding,
            native_disposition=result.native_disposition,
            error=result.error,
            graph_terminal_action=(
                "fail_leaf"
                if result.native_disposition == "failed"
                else "complete_leaf"
            ),
            graph_invocation_path=invocation_binding.path,
            graph_result_path=result_binding.path,
            graph_native_terminal_receipt_path=receipt_binding.path,
            graph_operation_index_paths=(str(attempt.path / projection_name),),
            graph_evidence_index_paths=(str(attempt.path / publication_name),),
            graph_evidence_paths=tuple(item.path for item in result.evidence),
            graph_saved_stage_readback_paths=tuple(
                item.path for item in result.saved_stage_readbacks
            ),
            output=result.output,
            output_dependencies=result.output_dependencies,
            resource_claims=(
                (f"texture.renderer:{result.leaf_id}",)
                if result.leaf_id in _RENDERER_LEAF_IDS
                else ()
            ),
            graph_resource_release_paths=(
                (receipt_binding.path,) if result.leaf_id in _RENDERER_LEAF_IDS else ()
            ),
        )
        publication_identity = attempt.write_json(publication_name, publication)
        publication_binding = attempt.binding(
            publication_name,
            identity=publication_identity,
        )
        _verify_focused_result_contract(
            attempt,
            invocation_binding=invocation_binding,
            invocation=invocation,
            result_binding=result_binding,
            result=result,
            receipt_binding=receipt_binding,
            receipt=receipt,
        )
        attempt.verify_binding(projection_binding, label="Texture focused projection")
        attempt.verify_binding(
            publication_binding,
            label="Texture focused publication",
        )
        attempt.verify_path()
        return publication


def texture_asset_leaf_runtime_bindings() -> tuple[AssetLeafRuntimeBinding, ...]:
    """Build six exact v3 Texture runtime registrations for repository discovery."""

    bindings = [
        AssetLeafRuntimeBinding.create(
            leaf_id=TEXTURE_UV_LEAF_ID,
            entrypoint=TEXTURE_UV_ASSET_ENTRYPOINT,
            invocation_model=TextureUvLeafInvocation,
            result_model=TextureUvLeafResult,
            projector_id="asset.projector.texture-uv-prepare.v1",
            projector=_project_texture_uv_runtime,
            required_artifact_categories=_BASE_REQUIRED_ARTIFACT_CATEGORIES,
        )
    ]
    for leaf_id in TEXTURE_FOCUSED_LEAF_IDS:
        bindings.append(
            AssetLeafRuntimeBinding.create(
                leaf_id=leaf_id,
                entrypoint=_FOCUSED_ENTRYPOINTS[leaf_id],
                invocation_model=TextureFocusedLeafInvocation,
                result_model=TextureFocusedLeafResult,
                projector_id=f"asset.projector.{leaf_id}",
                projector=_project_texture_focused_runtime,
                required_artifact_categories=(
                    _FOCUSED_REQUIRED_ARTIFACT_CATEGORIES[leaf_id]
                ),
                required_dependencies=_FOCUSED_DEPENDENCIES[leaf_id],
            )
        )
    return tuple(sorted(bindings, key=lambda binding: binding.descriptor.leaf_id))


def texture_asset_leaf_runtime_bundle() -> AssetLeafRuntimeBundle:
    """Return the side-effect-free bundle declared by the texture-agent package."""

    return AssetLeafRuntimeBundle.create(
        bundle_id=TEXTURE_ASSET_LEAF_BUNDLE_ID,
        bindings=texture_asset_leaf_runtime_bindings(),
    )


def texture_asset_leaf_descriptors() -> tuple[AssetLeafDescriptor, ...]:
    """Return exact v3 Texture descriptors derived from runtime registrations."""

    return tuple(
        binding.descriptor for binding in texture_asset_leaf_runtime_bindings()
    )


def texture_asset_leaf_catalog() -> AssetLeafCatalog:
    """Build the Texture-local v3 catalog without selecting or executing a leaf."""

    return AssetLeafCatalog.create(list(texture_asset_leaf_descriptors()))


__all__ = [
    "TEXTURE_ASSET_LEAF_BUNDLE_ID",
    "TEXTURE_APPLY_PROVIDED_LEAF_ID",
    "TEXTURE_EVIDENCE_LEAF_ID",
    "TEXTURE_FOCUSED_INVOCATION_SCHEMA_VERSION",
    "TEXTURE_FOCUSED_LEAF_IDS",
    "TEXTURE_FOCUSED_PROJECTION_SCHEMA_VERSION",
    "TEXTURE_FOCUSED_PUBLICATION_SCHEMA_VERSION",
    "TEXTURE_FOCUSED_READBACK_SCHEMA_VERSION",
    "TEXTURE_FOCUSED_RECEIPT_SCHEMA_VERSION",
    "TEXTURE_FOCUSED_RESULT_SCHEMA_VERSION",
    "TEXTURE_PREPARE_LEAF_ID",
    "TEXTURE_PUBLISH_LEAF_ID",
    "TEXTURE_REVIEW_LEAF_ID",
    "TEXTURE_UV_ASSET_ENTRYPOINT",
    "TEXTURE_UV_PROJECTION_FILENAME",
    "TEXTURE_UV_PROJECTION_SCHEMA_VERSION",
    "TEXTURE_UV_PUBLICATION_FILENAME",
    "TEXTURE_UV_PUBLICATION_SCHEMA_VERSION",
    "TextureFocusedAssetLeafPublication",
    "TextureFocusedLeafInvocation",
    "TextureFocusedLeafResult",
    "TextureFocusedSavedStageReadback",
    "TextureFocusedTerminalReceipt",
    "TextureFocusedVerifiedOperationProjection",
    "TextureUvAssetLeafPublication",
    "TextureUvVerifiedOperationProjection",
    "project_texture_focused_verified_operation",
    "project_texture_uv_verified_operation",
    "run_texture_apply_provided_asset_leaf",
    "run_texture_evidence_asset_leaf",
    "run_texture_prepare_asset_leaf",
    "run_texture_publish_asset_leaf",
    "run_texture_review_asset_leaf",
    "texture_asset_leaf_catalog",
    "texture_asset_leaf_descriptors",
    "texture_asset_leaf_runtime_bindings",
    "texture_asset_leaf_runtime_bundle",
]
