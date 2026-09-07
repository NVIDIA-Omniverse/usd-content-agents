# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Installed v3 asset-leaf runtime for provider-neutral Articulation.

The asset coordinator is the sole reasoner.  Preparation and an optional
provider proposal remain independently selectable.  The focused author,
evidence, review, and publish leaves are deterministic typed executors and
never launch a nested coordinator, a classic Joint pipeline, or a provider.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any, Literal, Self, cast

from filelock import FileLock
from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator

from content_agent_workflows.asset_composition import (
    ArtifactBinding as AssetArtifactBinding,
)
from content_agent_workflows.asset_composition import (
    AssetLeafCatalog,
    AssetLeafDescriptor,
    AssetLeafProjectionContext,
    AssetLeafProjectionPayload,
    AssetLeafRuntimeBinding,
    AssetLeafRuntimeBundle,
)
from content_agent_workflows.common.artifacts import (
    atomic_write_json,
    read_contained_artifact,
)
from content_agent_workflows.common.domain_execution import (
    DOMAIN_EXECUTION_CONTEXT_SCHEMA_VERSION_V3,
    DomainExecutionContext,
    ExecutionArtifactBinding,
    metadata_with_domain_execution_context,
)
from content_agent_workflows.common.embedded_domain_decision import (
    canonical_json_digest,
)
from content_agent_workflows.validation.verified_operations import (
    execution_artifact_binding,
    verify_execution_artifact_binding,
)

from .client import JointAgentGraphAuthoringClient
from .embedded_decision import EmbeddedArticulationCanonicalGraph
from .models import (
    ArticulationRunState,
    ArticulationWorkflowRequest,
    ArtifactBinding,
)
from .output_evidence import (
    EmbeddedArticulationOutputEvidence,
    build_embedded_articulation_output_evidence,
    validate_embedded_articulation_output_evidence,
)
from .preparation import ArticulationPreparationPublication
from .proposal_provider import (
    ArticulationProposalAttemptTerminalReceipt,
    validate_articulation_proposal_attempt_receipt,
)
from .standalone_decision import (
    StandaloneArticulationAuthoringReceipt,
    StandaloneArticulationCleanupReceipt,
    StandaloneArticulationDecisionLedger,
    StandaloneArticulationDecisionPatch,
    StandaloneArticulationIdentity,
    StandaloneArticulationIssuePacket,
    StandaloneArticulationObservation,
    StandaloneArticulationPostReviewPatch,
    StandaloneArticulationPreparation,
    StandaloneArticulationReadback,
    StandaloneArticulationTerminalReceipt,
    apply_standalone_articulation_decision_patch,
    build_standalone_articulation_observation,
    prepare_standalone_articulation_workflow,
)

ARTICULATION_PREPARATION_LEAF_ID = "articulation.preparation-publisher.v1"
ARTICULATION_PROPOSAL_LEAF_ID = "articulation.proposal-provider.v1"
ARTICULATION_AUTHOR_LEAF_ID: Literal["articulation.author.v1"] = (
    "articulation.author.v1"
)
ARTICULATION_EVIDENCE_LEAF_ID: Literal["articulation.evidence.v1"] = (
    "articulation.evidence.v1"
)
ARTICULATION_REVIEW_LEAF_ID: Literal["articulation.review.v1"] = (
    "articulation.review.v1"
)
ARTICULATION_PUBLISH_LEAF_ID: Literal["articulation.publish.v1"] = (
    "articulation.publish.v1"
)
ARTICULATION_ASSET_LEAF_BUNDLE_ID = "articulation"

ARTICULATION_PREPARATION_LEAF_INVOCATION_SCHEMA_VERSION: Literal[
    "content-agent-workflows.articulation-preparation-leaf-invocation.v1"
] = "content-agent-workflows.articulation-preparation-leaf-invocation.v1"
ARTICULATION_PREPARATION_FAILURE_RECEIPT_SCHEMA_VERSION: Literal[
    "content-agent-workflows.articulation-preparation-failure-receipt.v1"
] = "content-agent-workflows.articulation-preparation-failure-receipt.v1"
ARTICULATION_PREPARATION_TERMINAL_RESULT_SCHEMA_VERSION: Literal[
    "content-agent-workflows.articulation-preparation-terminal-result.v1"
] = "content-agent-workflows.articulation-preparation-terminal-result.v1"
ARTICULATION_PROPOSAL_LEAF_INVOCATION_SCHEMA_VERSION: Literal[
    "content-agent-workflows.articulation-proposal-leaf-invocation.v1"
] = "content-agent-workflows.articulation-proposal-leaf-invocation.v1"
ARTICULATION_FOCUSED_INVOCATION_SCHEMA_VERSION: Literal[
    "content-agent-workflows.articulation-focused-leaf-invocation.v1"
] = "content-agent-workflows.articulation-focused-leaf-invocation.v1"
ARTICULATION_FOCUSED_RESULT_SCHEMA_VERSION: Literal[
    "content-agent-workflows.articulation-focused-leaf-result.v1"
] = "content-agent-workflows.articulation-focused-leaf-result.v1"
ARTICULATION_AUTHOR_PROGRESS_SCHEMA_VERSION: Literal[
    "content-agent-workflows.articulation-author-leaf-progress.v1"
] = "content-agent-workflows.articulation-author-leaf-progress.v1"
ARTICULATION_ASSET_REVIEW_RECEIPT_SCHEMA_VERSION: Literal[
    "content-agent-workflows.articulation-asset-review-receipt.v1"
] = "content-agent-workflows.articulation-asset-review-receipt.v1"
ARTICULATION_ASSET_PHASE_RECEIPT_SCHEMA_VERSION: Literal[
    "content-agent-workflows.articulation-asset-phase-receipt.v1"
] = "content-agent-workflows.articulation-asset-phase-receipt.v1"

_PREPARATION_ENTRYPOINT = "content-workflow-cli articulation publish-preparation"
_PROPOSAL_ENTRYPOINT = "content-workflow-cli articulation propose"
_CANONICAL_OVRTX_EVIDENCE_LEAF_ID = "validation.canonical-ovrtx-evidence.v1"
_MAX_JSON_BYTES = 32 * 1024 * 1024

ArticulationFocusedLeafId = Literal[
    "articulation.author.v1",
    "articulation.evidence.v1",
    "articulation.review.v1",
    "articulation.publish.v1",
]
ARTICULATION_FOCUSED_LEAF_IDS: tuple[ArticulationFocusedLeafId, ...] = (
    ARTICULATION_AUTHOR_LEAF_ID,
    ARTICULATION_EVIDENCE_LEAF_ID,
    ARTICULATION_REVIEW_LEAF_ID,
    ARTICULATION_PUBLISH_LEAF_ID,
)
_FOCUSED_ENTRYPOINTS = {
    leaf_id: (
        "content-workflow-cli articulation agentic-leaf "
        f"{leaf_id.removeprefix('articulation.').removesuffix('.v1')} --invocation"
    )
    for leaf_id in ARTICULATION_FOCUSED_LEAF_IDS
}
_FOCUSED_DEPENDENCIES: dict[ArticulationFocusedLeafId, tuple[str, ...]] = {
    ARTICULATION_AUTHOR_LEAF_ID: (ARTICULATION_PREPARATION_LEAF_ID,),
    ARTICULATION_EVIDENCE_LEAF_ID: (
        ARTICULATION_AUTHOR_LEAF_ID,
        _CANONICAL_OVRTX_EVIDENCE_LEAF_ID,
    ),
    ARTICULATION_REVIEW_LEAF_ID: (ARTICULATION_EVIDENCE_LEAF_ID,),
    ARTICULATION_PUBLISH_LEAF_ID: (ARTICULATION_REVIEW_LEAF_ID,),
}


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _canonical_absolute(value: str, *, label: str) -> str:
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts or str(path) != value:
        raise ValueError(f"{label} must be canonical and absolute")
    return value


def _canonical_binding(binding: ExecutionArtifactBinding) -> ExecutionArtifactBinding:
    _canonical_absolute(binding.path, label="Articulation asset binding")
    return binding


def _binding(path: str | Path) -> ExecutionArtifactBinding:
    return execution_artifact_binding(Path(path).resolve())


def _direct_single_link_regular_file_exists(path: Path, *, label: str) -> bool:
    """Return whether one state leaf exists, rejecting links and reparses."""

    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise ValueError(f"{label} could not be inspected safely") from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or getattr(metadata, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    ):
        raise ValueError(f"{label} must be a direct single-link regular file")
    return True


def _native_binding(binding: ExecutionArtifactBinding) -> ArtifactBinding:
    return ArtifactBinding(path=binding.path, sha256=binding.sha256)


def _asset_binding(binding: ExecutionArtifactBinding) -> AssetArtifactBinding:
    return AssetArtifactBinding.model_validate(binding.model_dump(mode="json"))


def _verify_binding(
    binding: ExecutionArtifactBinding,
    *,
    label: str,
) -> bytes:
    return verify_execution_artifact_binding(binding, label=label)


def _load_bound[ModelT: BaseModel](
    binding: ExecutionArtifactBinding,
    model: type[ModelT],
    *,
    label: str,
) -> ModelT:
    try:
        return model.model_validate_json(_verify_binding(binding, label=label))
    except ValueError as exc:
        raise ValueError(f"invalid {label}: {exc}") from exc


def _write_once(
    path: Path, model: BaseModel, *, label: str
) -> ExecutionArtifactBinding:
    if path.exists():
        raise FileExistsError(f"{label} already exists: {path}")
    atomic_write_json(path, model)
    return _binding(path)


def _write_once_or_reuse_exact_preparation_failure_receipt(
    path: Path,
    receipt: ArticulationPreparationFailureReceipt,
) -> ExecutionArtifactBinding:
    """Recover only the exact failure receipt after a stdout crash window."""

    if not path.exists() and not path.is_symlink():
        return _write_once(
            path,
            receipt,
            label="Articulation preparation failure receipt",
        )
    try:
        existing = read_contained_artifact(
            path.parent,
            path,
            max_bytes=_MAX_JSON_BYTES,
            capture_bytes=True,
        )
    except (OSError, ValueError) as exc:
        raise ValueError(
            "Articulation preparation failure receipt is unavailable"
        ) from exc
    assert existing.data is not None
    expected_bytes = (
        json.dumps(
            receipt.model_dump(mode="json"),
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
        )
        + "\n"
    ).encode("utf-8")
    try:
        existing_receipt = ArticulationPreparationFailureReceipt.model_validate_json(
            existing.data
        )
    except ValueError as exc:
        raise ValueError(
            "Articulation preparation failure receipt is not valid"
        ) from exc
    if existing_receipt != receipt or existing.data != expected_bytes:
        raise ValueError(
            "Articulation preparation failure receipt differs from the exact "
            "expected receipt"
        )
    return _binding(path)


def _write_once_or_reuse_exact_phase_receipt(
    path: Path,
    receipt: ArticulationAssetPhaseReceipt,
    *,
    label: str,
) -> ExecutionArtifactBinding:
    """Write a phase receipt once, or reuse its exact canonical bytes.

    The author leaf writes its terminal phase receipt before its focused result.
    A process interruption between those two writes must be recoverable without
    weakening write-once custody for any other artifact.  Existing receipts are
    therefore accepted only when they are safe regular files containing the
    exact typed receipt and the exact bytes that ``atomic_write_json`` would
    have written for the current native state.
    """

    if not path.exists() and not path.is_symlink():
        return _write_once(path, receipt, label=label)
    try:
        existing = read_contained_artifact(
            path.parent,
            path,
            max_bytes=_MAX_JSON_BYTES,
            capture_bytes=True,
        )
    except (OSError, ValueError) as exc:
        raise ValueError(f"{label} is unavailable: {path}") from exc
    assert existing.data is not None
    existing_bytes = existing.data
    expected_bytes = (
        json.dumps(
            receipt.model_dump(mode="json"),
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
        )
        + "\n"
    ).encode("utf-8")
    try:
        existing_receipt = ArticulationAssetPhaseReceipt.model_validate_json(
            existing_bytes
        )
    except ValueError as exc:
        raise ValueError(f"{label} is not a valid typed receipt") from exc
    if existing_receipt != receipt or existing_bytes != expected_bytes:
        raise ValueError(f"{label} differs from the exact expected receipt")
    return _binding(path)


def _unique_bindings(
    bindings: tuple[ExecutionArtifactBinding, ...],
) -> tuple[ExecutionArtifactBinding, ...]:
    return tuple(
        {(item.path, item.sha256, item.size_bytes): item for item in bindings}.values()
    )


class ArticulationPreparationLeafInvocation(_FrozenModel):
    """Exact deterministic preparation publication selected by the graph."""

    schema_version: Literal[
        "content-agent-workflows.articulation-preparation-leaf-invocation.v1"
    ] = ARTICULATION_PREPARATION_LEAF_INVOCATION_SCHEMA_VERSION
    readback: ExecutionArtifactBinding
    retained_root: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_paths(self) -> Self:
        _canonical_binding(self.readback)
        _canonical_absolute(self.retained_root, label="retained root")
        if not Path(self.readback.path).is_relative_to(Path(self.retained_root)):
            raise ValueError("preparation readback must be inside the retained root")
        return self


class ArticulationPreparationFailureReceipt(_FrozenModel):
    """Attempt-confined failure that trusts only the selected invocation."""

    schema_version: Literal[
        "content-agent-workflows.articulation-preparation-failure-receipt.v1"
    ] = ARTICULATION_PREPARATION_FAILURE_RECEIPT_SCHEMA_VERSION
    invocation: ExecutionArtifactBinding
    native_status: Literal["invalid_preparation_input"] = "invalid_preparation_input"
    error: Literal[
        "Articulation preparation input failed deterministic validation."
    ] = "Articulation preparation input failed deterministic validation."
    nested_coordinator_invoked: Literal[False] = False
    provider_invoked: Literal[False] = False


class ArticulationPreparationLeafTerminalResult(_FrozenModel):
    """Closed graph result for one deterministic preparation failure."""

    schema_version: Literal[
        "content-agent-workflows.articulation-preparation-terminal-result.v1"
    ] = ARTICULATION_PREPARATION_TERMINAL_RESULT_SCHEMA_VERSION
    invocation: ExecutionArtifactBinding
    native_disposition: Literal["failed"] = "failed"
    native_status: Literal["invalid_preparation_input"] = "invalid_preparation_input"
    native_terminal_receipt: ExecutionArtifactBinding
    evidence: tuple[ExecutionArtifactBinding, ...] = Field(min_length=1)
    saved_stage_readbacks: tuple[ExecutionArtifactBinding, ...] = Field(min_length=1)
    error: Literal[
        "Articulation preparation input failed deterministic validation."
    ] = "Articulation preparation input failed deterministic validation."


class ArticulationPreparationLeafResult(
    RootModel[
        ArticulationPreparationPublication | ArticulationPreparationLeafTerminalResult
    ]
):
    """Wire-compatible success/failure union for the preparation graph leaf."""


def _preparation_invocation_location(
    invocation_path: str | Path,
) -> tuple[
    ExecutionArtifactBinding,
    ArticulationPreparationLeafInvocation,
    Path,
]:
    """Bind one safe preparation invocation without opening its inputs."""

    candidate = Path(os.path.abspath(Path(invocation_path).expanduser()))
    try:
        candidate_metadata = candidate.lstat()
        resolved = candidate.resolve(strict=True)
        root = candidate.parent.resolve(strict=True)
    except OSError as exc:
        raise ValueError(
            f"Articulation preparation invocation is unavailable: {candidate}"
        ) from exc
    if (
        resolved != candidate
        or root != candidate.parent
        or not stat.S_ISREG(candidate_metadata.st_mode)
        or candidate_metadata.st_nlink != 1
    ):
        raise ValueError("Articulation preparation invocation path is unsafe")
    binding = _binding(candidate)
    if binding.size_bytes > _MAX_JSON_BYTES:
        raise ValueError("Articulation preparation invocation exceeds its size bound")
    invocation = _load_bound(
        binding,
        ArticulationPreparationLeafInvocation,
        label="Articulation preparation invocation",
    )
    return binding, invocation, root


def fail_articulation_preparation_asset_leaf(
    invocation_path: str | Path,
    *,
    expected_invocation_binding: ExecutionArtifactBinding,
) -> ArticulationPreparationLeafTerminalResult:
    """Seal a generic typed failure without trusting malformed readback facts."""

    invocation_binding, _invocation, root = _preparation_invocation_location(
        invocation_path
    )
    if invocation_binding != expected_invocation_binding:
        raise ValueError(
            "Articulation preparation invocation changed before failure sealing"
        )
    receipt = ArticulationPreparationFailureReceipt(
        invocation=invocation_binding,
    )
    receipt_binding = _write_once_or_reuse_exact_preparation_failure_receipt(
        root / "articulation_preparation_failure_receipt.json",
        receipt,
    )
    return ArticulationPreparationLeafTerminalResult(
        invocation=invocation_binding,
        native_terminal_receipt=receipt_binding,
        evidence=(receipt_binding,),
        saved_stage_readbacks=(receipt_binding,),
    )


class ArticulationArtifactJsonProposalInvocation(_FrozenModel):
    adapter: Literal["artifact-json"] = "artifact-json"
    provider_id: str = Field(min_length=1)
    capability_id: str = Field(min_length=1)
    provider_payload: ExecutionArtifactBinding

    @model_validator(mode="after")
    def validate_payload(self) -> Self:
        _canonical_binding(self.provider_payload)
        return self


class ArticulationProposalLeafInvocation(_FrozenModel):
    """Exact explicitly selected advisory proposal attempt."""

    schema_version: Literal[
        "content-agent-workflows.articulation-proposal-leaf-invocation.v1"
    ] = ARTICULATION_PROPOSAL_LEAF_INVOCATION_SCHEMA_VERSION
    preparation: ExecutionArtifactBinding
    intent: str = Field(min_length=1, max_length=16_384)
    provider: ArticulationArtifactJsonProposalInvocation
    replaces_terminal_receipt: ExecutionArtifactBinding | None = None
    replacement_reason: str | None = Field(default=None, min_length=1, max_length=4096)

    @model_validator(mode="after")
    def validate_inputs(self) -> Self:
        _canonical_binding(self.preparation)
        if self.replaces_terminal_receipt is not None:
            _canonical_binding(self.replaces_terminal_receipt)
        if (self.replaces_terminal_receipt is None) != (
            self.replacement_reason is None
        ):
            raise ValueError(
                "proposal replacement receipt and reason must be provided together"
            )
        return self


class ArticulationFocusedLeafInvocation(_FrozenModel):
    """Exact outer-authored input for one deterministic Articulation leaf."""

    schema_version: Literal[
        "content-agent-workflows.articulation-focused-leaf-invocation.v1"
    ] = ARTICULATION_FOCUSED_INVOCATION_SCHEMA_VERSION
    leaf_id: ArticulationFocusedLeafId
    attempt_root: str = Field(min_length=1)
    source: ExecutionArtifactBinding | None = None
    preparation_publication: ExecutionArtifactBinding | None = None
    decision_patch_path: str | None = None
    proposal_result: ExecutionArtifactBinding | None = None
    intent: str | None = Field(default=None, min_length=1, max_length=16_384)
    allowed_motion_types: tuple[Literal["revolute", "prismatic"], ...] = (
        "revolute",
        "prismatic",
    )
    expected_candidate_count: int | None = Field(default=None, ge=0, le=256)
    max_candidate_count: int = Field(default=64, ge=1, le=256)
    author_result: ExecutionArtifactBinding | None = None
    canonical_visual_envelope: ExecutionArtifactBinding | None = None
    evidence_result: ExecutionArtifactBinding | None = None
    post_review_patch: ExecutionArtifactBinding | None = None
    review_result: ExecutionArtifactBinding | None = None
    selected_mode: Literal["agentic"] = "agentic"
    reasoning_loop_owner: Literal["asset_coordinator"] = "asset_coordinator"
    provider_status: Literal["not_requested", "provided"] = "not_requested"
    provider_invoked: Literal[False] = False
    nested_coordinator_invoked: Literal[False] = False
    fixed_pipeline_invoked: Literal[False] = False
    joint_agent_local_client_invoked: Literal[False] = False

    @model_validator(mode="after")
    def validate_leaf_scope(self) -> Self:
        _canonical_absolute(self.attempt_root, label="Articulation attempt root")
        fields = {
            "source": self.source,
            "preparation_publication": self.preparation_publication,
            "decision_patch_path": self.decision_patch_path,
            "proposal_result": self.proposal_result,
            "intent": self.intent,
            "author_result": self.author_result,
            "canonical_visual_envelope": self.canonical_visual_envelope,
            "evidence_result": self.evidence_result,
            "post_review_patch": self.post_review_patch,
            "review_result": self.review_result,
        }
        required: dict[ArticulationFocusedLeafId, set[str]] = {
            ARTICULATION_AUTHOR_LEAF_ID: {
                "source",
                "preparation_publication",
                "decision_patch_path",
                "intent",
            },
            ARTICULATION_EVIDENCE_LEAF_ID: {
                "author_result",
                "canonical_visual_envelope",
            },
            ARTICULATION_REVIEW_LEAF_ID: {
                "author_result",
                "evidence_result",
                "post_review_patch",
            },
            ARTICULATION_PUBLISH_LEAF_ID: {
                "author_result",
                "evidence_result",
                "review_result",
            },
        }
        present = {name for name, value in fields.items() if value is not None}
        allowed = required[self.leaf_id]
        if self.leaf_id == ARTICULATION_AUTHOR_LEAF_ID:
            allowed = allowed | {"proposal_result"}
        if not required[self.leaf_id].issubset(present) or not present.issubset(
            allowed
        ):
            raise ValueError(
                f"Articulation {self.leaf_id} invocation fields differ: "
                f"expected {sorted(required[self.leaf_id])} with optional "
                f"proposal_result, got {sorted(present)}"
            )
        for value in fields.values():
            if isinstance(value, ExecutionArtifactBinding):
                _canonical_binding(value)
        if self.decision_patch_path is not None:
            _canonical_absolute(
                self.decision_patch_path,
                label="Articulation decision patch path",
            )
            expected_patch_path = (
                Path(self.attempt_root) / "articulation_decision_patch.json"
            )
            if Path(self.decision_patch_path) != expected_patch_path:
                raise ValueError(
                    "Articulation decision patch must use its prescribed attempt path"
                )
        if self.post_review_patch is not None and (
            Path(self.post_review_patch.path).parent != Path(self.attempt_root)
        ):
            raise ValueError(
                "Articulation post-review patch must be inside its review attempt"
            )
        if (self.proposal_result is not None) != (self.provider_status == "provided"):
            raise ValueError(
                "Articulation provider status must match the optional proposal result"
            )
        if not self.allowed_motion_types or len(self.allowed_motion_types) != len(
            set(self.allowed_motion_types)
        ):
            raise ValueError("Articulation allowed motion types must be unique")
        if (
            self.expected_candidate_count is not None
            and self.expected_candidate_count > self.max_candidate_count
        ):
            raise ValueError(
                "expected candidate count must not exceed max candidate count"
            )
        return self


type ArticulationLeafInvocation = (
    ArticulationPreparationLeafInvocation
    | ArticulationProposalLeafInvocation
    | ArticulationFocusedLeafInvocation
)


class ArticulationAssetPhaseReceipt(_FrozenModel):
    schema_version: Literal[
        "content-agent-workflows.articulation-asset-phase-receipt.v1"
    ] = ARTICULATION_ASSET_PHASE_RECEIPT_SCHEMA_VERSION
    leaf_id: ArticulationFocusedLeafId
    invocation: ExecutionArtifactBinding
    upstream_results: tuple[ExecutionArtifactBinding, ...]
    retained_artifacts: tuple[ExecutionArtifactBinding, ...] = Field(min_length=1)
    native_status: str = Field(min_length=1)
    error: str | None = None
    nested_coordinator_invoked: Literal[False] = False
    provider_invoked: Literal[False] = False


class ArticulationAssetReviewReceipt(_FrozenModel):
    schema_version: Literal[
        "content-agent-workflows.articulation-asset-review-receipt.v1"
    ] = ARTICULATION_ASSET_REVIEW_RECEIPT_SCHEMA_VERSION
    invocation: ExecutionArtifactBinding
    author_result: ExecutionArtifactBinding
    evidence_result: ExecutionArtifactBinding
    post_review_patch: ExecutionArtifactBinding
    identity: ExecutionArtifactBinding
    canonical_graph: ExecutionArtifactBinding
    authoring_receipt: ExecutionArtifactBinding
    readback: ExecutionArtifactBinding
    output_evidence: ExecutionArtifactBinding
    disposition: Literal["accept", "reject", "revise"]
    inspected_render_image_sha256s: tuple[str, ...] = Field(min_length=1)
    findings: tuple[str, ...] = Field(min_length=1)
    semantic_authority: Literal["asset_coordinator"] = "asset_coordinator"


class ArticulationFocusedLeafResult(_FrozenModel):
    """Closed typed result used by all four focused Articulation leaves."""

    schema_version: Literal[
        "content-agent-workflows.articulation-focused-leaf-result.v1"
    ] = ARTICULATION_FOCUSED_RESULT_SCHEMA_VERSION
    leaf_id: ArticulationFocusedLeafId
    invocation: ExecutionArtifactBinding
    native_disposition: Literal["passed", "failed"]
    native_status: str = Field(min_length=1)
    native_terminal_receipt: ExecutionArtifactBinding
    evidence: tuple[ExecutionArtifactBinding, ...] = Field(min_length=1)
    saved_stage_readbacks: tuple[ExecutionArtifactBinding, ...] = Field(min_length=1)
    output: ExecutionArtifactBinding | None = None
    error: str | None = None
    provider_status: Literal["not_requested", "provided"] = "not_requested"
    provider_invoked: Literal[False] = False
    nested_coordinator_invoked: Literal[False] = False
    fixed_pipeline_invoked: Literal[False] = False
    joint_agent_local_client_invoked: Literal[False] = False

    @model_validator(mode="after")
    def validate_disposition(self) -> Self:
        if (self.native_disposition == "failed") != bool(self.error):
            raise ValueError("Articulation failed disposition requires one exact error")
        if self.native_disposition == "passed" and self.output is None:
            raise ValueError("passed Articulation leaf requires exact output bytes")
        return self


class ArticulationAuthorLeafProgress(_FrozenModel):
    """Non-terminal author handoff for the sole coordinator's decision patch."""

    schema_version: Literal[
        "content-agent-workflows.articulation-author-leaf-progress.v1"
    ] = ARTICULATION_AUTHOR_PROGRESS_SCHEMA_VERSION
    leaf_id: Literal["articulation.author.v1"] = ARTICULATION_AUTHOR_LEAF_ID
    invocation: ExecutionArtifactBinding
    native_status: Literal["awaiting_decision"] = "awaiting_decision"
    identity: ExecutionArtifactBinding
    observation: ExecutionArtifactBinding
    decision_patch_path: str = Field(min_length=1)
    semantic_authority: Literal["asset_coordinator"] = "asset_coordinator"
    provider_status: Literal["not_requested", "provided"] = "not_requested"
    provider_invoked: Literal[False] = False
    nested_coordinator_invoked: Literal[False] = False


def _invocation_location(
    invocation_path: str | Path,
) -> tuple[ExecutionArtifactBinding, ArticulationFocusedLeafInvocation, Path]:
    candidate = Path(os.path.abspath(Path(invocation_path).expanduser()))
    try:
        candidate_metadata = candidate.lstat()
        resolved = candidate.resolve(strict=True)
        root = candidate.parent.resolve(strict=True)
    except OSError as exc:
        raise ValueError(
            f"Articulation focused invocation is unavailable: {candidate}"
        ) from exc
    if (
        resolved != candidate
        or root != candidate.parent
        or not stat.S_ISREG(candidate_metadata.st_mode)
        or candidate_metadata.st_nlink != 1
    ):
        raise ValueError("Articulation focused invocation path is unsafe")
    binding = _binding(candidate)
    if binding.size_bytes > _MAX_JSON_BYTES:
        raise ValueError("Articulation focused invocation exceeds its size bound")
    invocation = ArticulationFocusedLeafInvocation.model_validate_json(
        _verify_binding(binding, label="Articulation focused invocation")
    )
    if Path(invocation.attempt_root) != root or candidate.parent != root:
        raise ValueError("Articulation focused invocation changed its attempt root")
    native = root / "native"
    result_path = root / "articulation_leaf_result.json"
    if result_path.exists():
        raise FileExistsError("Articulation focused attempt is already terminal")
    if native.exists() and invocation.leaf_id != ARTICULATION_AUTHOR_LEAF_ID:
        raise FileExistsError("Articulation focused attempt is not fresh")
    return binding, invocation, root


def _failure_result(
    *,
    root: Path,
    invocation_binding: ExecutionArtifactBinding,
    invocation: ArticulationFocusedLeafInvocation,
    error: str,
    retained: tuple[ExecutionArtifactBinding, ...] = (),
) -> ArticulationFocusedLeafResult:
    receipt = ArticulationAssetPhaseReceipt(
        leaf_id=invocation.leaf_id,
        invocation=invocation_binding,
        upstream_results=tuple(
            item
            for item in (
                invocation.proposal_result,
                invocation.author_result,
                invocation.evidence_result,
                invocation.review_result,
            )
            if item is not None
        ),
        retained_artifacts=(invocation_binding, *retained),
        native_status="failed",
        error=error,
    )
    receipt_binding = _write_once(
        root / "articulation_failure_receipt.json",
        receipt,
        label="Articulation failure receipt",
    )
    result = ArticulationFocusedLeafResult(
        leaf_id=invocation.leaf_id,
        invocation=invocation_binding,
        native_disposition="failed",
        native_status="failed",
        native_terminal_receipt=receipt_binding,
        evidence=_unique_bindings((receipt_binding, *retained)),
        saved_stage_readbacks=(receipt_binding,),
        error=error,
        provider_status=invocation.provider_status,
    )
    _write_once(
        root / "articulation_leaf_result.json",
        result,
        label="Articulation focused result",
    )
    return result


def _load_focused_result(
    binding: ExecutionArtifactBinding,
    *,
    expected_leaf_id: ArticulationFocusedLeafId,
) -> ArticulationFocusedLeafResult:
    result = _load_bound(
        binding,
        ArticulationFocusedLeafResult,
        label=f"{expected_leaf_id} result",
    )
    if result.leaf_id != expected_leaf_id or result.native_disposition != "passed":
        raise ValueError(f"{expected_leaf_id} did not produce a passing result")
    return result


def _binding_named(
    result: ArticulationFocusedLeafResult,
    filename: str,
) -> ExecutionArtifactBinding:
    matches = _unique_bindings(
        tuple(
            item
            for item in (*result.evidence, *result.saved_stage_readbacks)
            if Path(item.path).name == filename
        )
    )
    if len(matches) != 1:
        raise ValueError(f"Articulation result lacks one exact {filename}")
    _verify_binding(matches[0], label=filename)
    return matches[0]


def run_articulation_author_asset_leaf(
    invocation_path: str | Path,
    *,
    client: Any | None = None,
) -> ArticulationAuthorLeafProgress | ArticulationFocusedLeafResult:
    """Freeze outer semantics and author only the exact accepted graph."""

    invocation_binding, invocation, root = _invocation_location(invocation_path)
    if invocation.leaf_id != ARTICULATION_AUTHOR_LEAF_ID:
        raise ValueError("Articulation author entrypoint received another leaf")
    retained: list[ExecutionArtifactBinding] = []
    leaf_lock = FileLock(str(root / ".articulation-author-leaf.lock"))
    leaf_lock.acquire()
    try:
        concurrent_result_path = root / "articulation_leaf_result.json"
        if concurrent_result_path.exists():
            return _load_bound(
                _binding(concurrent_result_path),
                ArticulationFocusedLeafResult,
                label="Articulation author concurrent result",
            )
        assert invocation.source is not None
        assert invocation.preparation_publication is not None
        assert invocation.decision_patch_path is not None
        assert invocation.intent is not None
        publication = _load_bound(
            invocation.preparation_publication,
            ArticulationPreparationPublication,
            label="Articulation preparation publication",
        )
        publication_preparation = _load_bound(
            publication.preparation,
            StandaloneArticulationPreparation,
            label="published Articulation preparation",
        )
        if publication.preparation_digest != canonical_json_digest(
            publication_preparation
        ):
            raise ValueError("Articulation preparation publication digest is stale")
        preparation = publication_preparation
        proposal_result = None
        if invocation.proposal_result is not None:
            proposal_result = _load_bound(
                invocation.proposal_result,
                ArticulationProposalAttemptTerminalReceipt,
                label="Articulation proposal result",
            )
            if (
                proposal_result.disposition != "succeeded"
                or proposal_result.preparation != publication.preparation
                or proposal_result.preparation_digest != publication.preparation_digest
                or proposal_result.bound_preparation is None
            ):
                raise ValueError(
                    "Articulation author requires one successful matching proposal"
                )
            validated_proposal_result = validate_articulation_proposal_attempt_receipt(
                invocation.proposal_result.path
            )
            if validated_proposal_result != proposal_result:
                raise ValueError(
                    "Articulation proposal result differs from its validated terminal"
                )
            preparation = _load_bound(
                proposal_result.bound_preparation,
                StandaloneArticulationPreparation,
                label="proposal-bound Articulation preparation",
            )
        _verify_binding(invocation.source, label="Articulation source")
        if invocation.source != publication.source:
            raise ValueError("Articulation source differs from preparation publication")
        native = root / "native"
        identity_path = native / "standalone_articulation_identity.json"
        checkpoint_path = native / "checkpoint.json"
        initialize_native = False
        if not native.exists():
            native.mkdir(mode=0o700)
            initialize_native = True
        elif (
            native.is_symlink()
            or not native.is_dir()
            or getattr(native.lstat(), "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        ):
            raise ValueError("Articulation native workspace is not a real directory")
        else:
            identity_exists = _direct_single_link_regular_file_exists(
                identity_path,
                label="Articulation native identity",
            )
            checkpoint_exists = _direct_single_link_regular_file_exists(
                checkpoint_path,
                label="Articulation native checkpoint",
            )
            if identity_exists != checkpoint_exists:
                raise ValueError(
                    "Articulation native workspace is partially initialized"
                )
            if not identity_exists:
                if any(native.iterdir()):
                    raise ValueError(
                        "Articulation native workspace contains untrusted partial state"
                    )
                native.chmod(0o700)
                initialize_native = True
        if initialize_native:
            context = DomainExecutionContext(
                schema_version=DOMAIN_EXECUTION_CONTEXT_SCHEMA_VERSION_V3,
                domain="articulation",
                mode="standalone",
                reasoning_loop_owner="asset_coordinator",
            )
            request = ArticulationWorkflowRequest(
                source_asset=invocation.source.path,
                output_dir=native,
                intent=invocation.intent,
                review_policy="none",
                allowed_motion_types=invocation.allowed_motion_types,
                expected_candidate_count=invocation.expected_candidate_count,
                max_candidate_count=invocation.max_candidate_count,
                metadata=metadata_with_domain_execution_context({}, context),
            )
            prepare_standalone_articulation_workflow(
                request,
                mode="batch",
                preparation=preparation,
            )
        initial_observation_path = root / "articulation_author_observation.json"
        initial_decision_path = Path(invocation.decision_patch_path)
        native_decision_path = native / "articulation_decision_patch.json"
        try:
            identity_read = read_contained_artifact(
                native,
                identity_path,
                max_bytes=_MAX_JSON_BYTES,
            )
            checkpoint_read = read_contained_artifact(
                native,
                checkpoint_path,
                max_bytes=_MAX_JSON_BYTES,
                capture_bytes=True,
            )
        except (OSError, ValueError) as exc:
            raise ValueError(
                "Articulation native workspace state is unavailable or unsafe"
            ) from exc
        identity_binding = ExecutionArtifactBinding(
            path=str(identity_read.path),
            sha256=identity_read.sha256,
            size_bytes=identity_read.size_bytes,
        )
        assert checkpoint_read.data is not None
        current_state = ArticulationRunState.model_validate_json(checkpoint_read.data)
        observation_bindings: list[ExecutionArtifactBinding] = []
        outer_patch_bindings: list[ExecutionArtifactBinding] = []

        if current_state.phase == "awaiting_decision":
            observation = build_standalone_articulation_observation(native)
            if observation.refinement_attempt == 0:
                observation_path = initial_observation_path
                decision_path = initial_decision_path
            else:
                observation_path = root / (
                    "articulation_author_refinement_observation-"
                    f"{observation.refinement_attempt:02d}.json"
                )
                decision_path = Path(observation.decision_patch_path)
            if observation_path.exists():
                existing_observation = (
                    StandaloneArticulationObservation.model_validate_json(
                        observation_path.read_bytes()
                    )
                )
                if existing_observation != observation:
                    raise ValueError(
                        "Articulation author observation changed before resume"
                    )
                observation_binding = _binding(observation_path)
            else:
                observation_binding = _write_once(
                    observation_path,
                    observation,
                    label="Articulation author observation",
                )
            observation_bindings.append(observation_binding)
            if initial_observation_path.exists() and (
                initial_observation_path != observation_path
            ):
                observation_bindings.insert(0, _binding(initial_observation_path))
            if not decision_path.exists():
                return ArticulationAuthorLeafProgress(
                    invocation=invocation_binding,
                    identity=identity_binding,
                    observation=observation_binding,
                    decision_patch_path=str(decision_path),
                    provider_status=invocation.provider_status,
                )
        elif current_state.phase == "awaiting_post_review":
            if not native_decision_path.exists():
                raise ValueError(
                    "Articulation post-review state omitted its native decision"
                )
            if not initial_observation_path.exists():
                raise ValueError(
                    "Articulation native decision exists without its bound observation"
                )
            observation_binding = _binding(initial_observation_path)
            observation_bindings.append(observation_binding)
            for attempt in range(
                1, len(current_state.standalone_refinement_history) + 1
            ):
                refinement_observation_path = root / (
                    f"articulation_author_refinement_observation-{attempt:02d}.json"
                )
                if not refinement_observation_path.exists():
                    raise ValueError(
                        "Articulation refinement omitted its bound observation"
                    )
                observation_bindings.append(_binding(refinement_observation_path))
            decision_path = Path(
                cast(ArtifactBinding, current_state.standalone_decision_patch).path
            )
        else:
            raise RuntimeError(
                current_state.error
                or f"Articulation authoring reached unsupported {current_state.phase}"
            )

        if not decision_path.is_relative_to(root):
            raise ValueError("Articulation decision patch must stay inside the attempt")
        decision_metadata = decision_path.lstat()
        if (
            decision_path.resolve(strict=True) != decision_path
            or not stat.S_ISREG(decision_metadata.st_mode)
            or decision_metadata.st_nlink != 1
        ):
            raise ValueError("Articulation decision patch path is unsafe")
        outer_patch_binding = _binding(decision_path)
        if initial_decision_path.exists() and initial_decision_path != decision_path:
            outer_patch_bindings.append(_binding(initial_decision_path))
        outer_patch_bindings.append(outer_patch_binding)
        patch = _load_bound(
            outer_patch_binding,
            StandaloneArticulationDecisionPatch,
            label="Articulation decision patch",
        )
        if native_decision_path.exists() and current_state.phase == (
            "awaiting_post_review"
        ):
            state = current_state
        else:
            state = apply_standalone_articulation_decision_patch(
                native,
                patch,
                client=(
                    client if client is not None else JointAgentGraphAuthoringClient()
                ),
            )
        if state.phase == "awaiting_decision":
            refinement_observation = build_standalone_articulation_observation(native)
            refinement_observation_path = root / (
                "articulation_author_refinement_observation-"
                f"{refinement_observation.refinement_attempt:02d}.json"
            )
            refinement_observation_binding = _write_once(
                refinement_observation_path,
                refinement_observation,
                label="Articulation author refinement observation",
            )
            return ArticulationAuthorLeafProgress(
                invocation=invocation_binding,
                identity=identity_binding,
                observation=refinement_observation_binding,
                decision_patch_path=refinement_observation.decision_patch_path,
                provider_status=invocation.provider_status,
            )
        if state.phase != "awaiting_post_review":
            if state.standalone_refinement_history:
                retained.extend(
                    _binding(item.path) for item in state.standalone_refinement_history
                )
            if state.standalone_terminal_receipt is not None:
                retained.append(_binding(state.standalone_terminal_receipt.path))
            raise RuntimeError(
                state.error
                or "Articulation authoring did not reach exact post-review custody"
            )
        required = {
            "identity": state.standalone_identity,
            "preparation": state.standalone_preparation,
            "decision patch": state.standalone_decision_patch,
            "decision ledger": state.standalone_decision_ledger,
            "canonical graph": state.standalone_canonical_graph,
            "authoring receipt": state.standalone_authoring_receipt,
            "saved-stage readback": state.standalone_readback,
            "authoring result": state.authoring_result,
            "validation result": state.validation_result,
        }
        if any(value is None for value in required.values()):
            raise RuntimeError("Articulation authoring omitted required custody")
        bound = {
            name: _binding(cast(Any, value).path) for name, value in required.items()
        }
        refinement_bindings = tuple(
            _binding(item.path) for item in state.standalone_refinement_history
        )
        retained.extend(
            (
                *outer_patch_bindings,
                *observation_bindings,
                *refinement_bindings,
                *bound.values(),
            )
        )
        native_patch = _load_bound(
            bound["decision patch"],
            StandaloneArticulationDecisionPatch,
            label="normalized Articulation decision patch",
        )
        if native_patch != patch:
            raise RuntimeError(
                "Articulation normalized decision patch changed outer semantics"
            )
        decision_ledger = _load_bound(
            bound["decision ledger"],
            StandaloneArticulationDecisionLedger,
            label="Articulation decision ledger",
        )
        if decision_ledger.decision_patch != _native_binding(bound["decision patch"]):
            raise RuntimeError(
                "Articulation decision ledger does not bind the normalized patch"
            )
        optional_proposal = (
            _binding(state.standalone_proposal.path)
            if state.standalone_proposal is not None
            else None
        )
        authoring_receipt = _load_bound(
            bound["authoring receipt"],
            StandaloneArticulationAuthoringReceipt,
            label="Articulation authoring receipt",
        )
        outer_custody_bindings = tuple(
            item
            for item in outer_patch_bindings
            if item.path != bound["decision patch"].path
        )
        retained_artifacts = _unique_bindings(
            (
                *outer_custody_bindings,
                *observation_bindings,
                *refinement_bindings,
                *bound.values(),
                *((optional_proposal,) if optional_proposal is not None else ()),
            )
        )
        phase_receipt = ArticulationAssetPhaseReceipt(
            leaf_id=ARTICULATION_AUTHOR_LEAF_ID,
            invocation=invocation_binding,
            upstream_results=tuple(
                item
                for item in (
                    invocation.preparation_publication,
                    invocation.proposal_result,
                )
                if item is not None
            ),
            retained_artifacts=retained_artifacts,
            native_status="awaiting_post_review",
        )
        phase_binding = _write_once_or_reuse_exact_phase_receipt(
            root / "articulation_author_terminal_receipt.json",
            phase_receipt,
            label="Articulation author terminal receipt",
        )
        result = ArticulationFocusedLeafResult(
            leaf_id=ARTICULATION_AUTHOR_LEAF_ID,
            invocation=invocation_binding,
            native_disposition="passed",
            native_status="awaiting_post_review",
            native_terminal_receipt=phase_binding,
            evidence=(
                bound["decision patch"],
                *_unique_bindings(tuple(observation_bindings)),
                *_unique_bindings(refinement_bindings),
                bound["identity"],
                bound["preparation"],
                bound["decision ledger"],
                bound["canonical graph"],
                bound["authoring receipt"],
                *((optional_proposal,) if optional_proposal is not None else ()),
            ),
            saved_stage_readbacks=(
                bound["saved-stage readback"],
                bound["authoring result"],
                bound["validation result"],
            ),
            output=authoring_receipt.output_asset,
            provider_status=invocation.provider_status,
        )
        _write_once(
            root / "articulation_leaf_result.json",
            result,
            label="Articulation author result",
        )
        return result
    except Exception as exc:
        detail = str(exc).strip() or "no exception detail"
        return _failure_result(
            root=root,
            invocation_binding=invocation_binding,
            invocation=invocation,
            error=f"{type(exc).__name__}: {detail}",
            retained=tuple(retained),
        )
    finally:
        leaf_lock.release()


def run_articulation_evidence_asset_leaf(
    invocation_path: str | Path,
) -> ArticulationFocusedLeafResult:
    """Bind shared canonical OVRTX evidence without making a visual decision."""

    invocation_binding, invocation, root = _invocation_location(invocation_path)
    if invocation.leaf_id != ARTICULATION_EVIDENCE_LEAF_ID:
        raise ValueError("Articulation evidence entrypoint received another leaf")
    try:
        assert invocation.author_result is not None
        assert invocation.canonical_visual_envelope is not None
        author = _load_focused_result(
            invocation.author_result,
            expected_leaf_id=ARTICULATION_AUTHOR_LEAF_ID,
        )
        identity_binding = _binding_named(
            author, "standalone_articulation_identity.json"
        )
        identity = _load_bound(
            identity_binding,
            StandaloneArticulationIdentity,
            label="Articulation identity",
        )
        if author.output is None:
            raise ValueError("Articulation author result omitted its output")
        evidence = build_embedded_articulation_output_evidence(
            invocation.canonical_visual_envelope.path,
            expected_source=identity.source,
            expected_output=author.output,
        )
        native = root / "native"
        native.mkdir(mode=0o700)
        evidence_binding = _write_once(
            native / "standalone_articulation_output_evidence.json",
            evidence,
            label="Articulation output evidence",
        )
        phase_receipt = ArticulationAssetPhaseReceipt(
            leaf_id=ARTICULATION_EVIDENCE_LEAF_ID,
            invocation=invocation_binding,
            upstream_results=(
                invocation.author_result,
                invocation.canonical_visual_envelope,
            ),
            retained_artifacts=(
                evidence_binding,
                evidence.canonical_visual_envelope,
                evidence.canonical_visual_payload,
                evidence.render_report,
                *evidence.images,
            ),
            native_status="completed",
        )
        phase_binding = _write_once(
            root / "articulation_evidence_terminal_receipt.json",
            phase_receipt,
            label="Articulation evidence terminal receipt",
        )
        result = ArticulationFocusedLeafResult(
            leaf_id=ARTICULATION_EVIDENCE_LEAF_ID,
            invocation=invocation_binding,
            native_disposition="passed",
            native_status="completed",
            native_terminal_receipt=phase_binding,
            evidence=(
                evidence_binding,
                phase_binding,
            ),
            saved_stage_readbacks=(
                evidence_binding,
                phase_binding,
            ),
            output=author.output,
        )
        _write_once(
            root / "articulation_leaf_result.json",
            result,
            label="Articulation evidence result",
        )
        return result
    except Exception as exc:
        detail = str(exc).strip() or "no exception detail"
        return _failure_result(
            root=root,
            invocation_binding=invocation_binding,
            invocation=invocation,
            error=f"{type(exc).__name__}: {detail}",
        )


def run_articulation_review_asset_leaf(
    invocation_path: str | Path,
) -> ArticulationFocusedLeafResult:
    """Verify one outer-authored review against exact graph, readback, and images."""

    invocation_binding, invocation, root = _invocation_location(invocation_path)
    if invocation.leaf_id != ARTICULATION_REVIEW_LEAF_ID:
        raise ValueError("Articulation review entrypoint received another leaf")
    try:
        assert invocation.author_result is not None
        assert invocation.evidence_result is not None
        assert invocation.post_review_patch is not None
        author = _load_focused_result(
            invocation.author_result,
            expected_leaf_id=ARTICULATION_AUTHOR_LEAF_ID,
        )
        evidence_result = _load_focused_result(
            invocation.evidence_result,
            expected_leaf_id=ARTICULATION_EVIDENCE_LEAF_ID,
        )
        identity_binding = _binding_named(
            author, "standalone_articulation_identity.json"
        )
        graph_binding = _binding_named(author, "canonical_articulation_graph.json")
        receipt_binding = _binding_named(
            author, "standalone_articulation_authoring_receipt.json"
        )
        readback_binding = _binding_named(
            author, "standalone_articulation_readback.json"
        )
        output_evidence_binding = _binding_named(
            evidence_result, "standalone_articulation_output_evidence.json"
        )
        identity = _load_bound(
            identity_binding,
            StandaloneArticulationIdentity,
            label="Articulation identity",
        )
        graph = _load_bound(
            graph_binding,
            EmbeddedArticulationCanonicalGraph,
            label="Articulation canonical graph",
        )
        receipt = _load_bound(
            receipt_binding,
            StandaloneArticulationAuthoringReceipt,
            label="Articulation authoring receipt",
        )
        readback = _load_bound(
            readback_binding,
            StandaloneArticulationReadback,
            label="Articulation readback",
        )
        output_evidence = _load_bound(
            output_evidence_binding,
            EmbeddedArticulationOutputEvidence,
            label="Articulation output evidence",
        )
        validate_embedded_articulation_output_evidence(output_evidence)
        patch = _load_bound(
            invocation.post_review_patch,
            StandaloneArticulationPostReviewPatch,
            label="Articulation post-review patch",
        )
        expected_images = tuple(item.sha256 for item in output_evidence.images)
        if (
            patch.identity_digest != canonical_json_digest(identity)
            or patch.authoring_receipt_sha256 != receipt_binding.sha256
            or patch.canonical_graph_sha256 != graph_binding.sha256
            or patch.readback_sha256 != readback_binding.sha256
            or patch.output_evidence_sha256 != output_evidence_binding.sha256
            or patch.inspected_render_image_sha256s != expected_images
            or readback.canonical_graph_digest != canonical_json_digest(graph)
            or receipt.output_asset != output_evidence.post_mutation_output
        ):
            raise ValueError("Articulation review is stale or incomplete")
        review = ArticulationAssetReviewReceipt(
            invocation=invocation_binding,
            author_result=invocation.author_result,
            evidence_result=invocation.evidence_result,
            post_review_patch=invocation.post_review_patch,
            identity=identity_binding,
            canonical_graph=graph_binding,
            authoring_receipt=receipt_binding,
            readback=readback_binding,
            output_evidence=output_evidence_binding,
            disposition=patch.disposition,
            inspected_render_image_sha256s=patch.inspected_render_image_sha256s,
            findings=patch.findings,
        )
        review_binding = _write_once(
            root / "articulation_asset_review_receipt.json",
            review,
            label="Articulation asset review receipt",
        )
        disposition: Literal["passed", "failed"] = (
            "passed" if patch.disposition == "accept" else "failed"
        )
        error = (
            None
            if patch.disposition == "accept"
            else f"Articulation outer review {patch.disposition}: "
            + "; ".join(patch.findings)
        )
        result = ArticulationFocusedLeafResult(
            leaf_id=ARTICULATION_REVIEW_LEAF_ID,
            invocation=invocation_binding,
            native_disposition=disposition,
            native_status=patch.disposition,
            native_terminal_receipt=review_binding,
            evidence=(review_binding,),
            saved_stage_readbacks=(review_binding,),
            output=author.output if disposition == "passed" else None,
            error=error,
        )
        _write_once(
            root / "articulation_leaf_result.json",
            result,
            label="Articulation review result",
        )
        return result
    except Exception as exc:
        detail = str(exc).strip() or "no exception detail"
        return _failure_result(
            root=root,
            invocation_binding=invocation_binding,
            invocation=invocation,
            error=f"{type(exc).__name__}: {detail}",
        )


def run_articulation_publish_asset_leaf(
    invocation_path: str | Path,
) -> ArticulationFocusedLeafResult:
    """Publish one accepted immutable chain without mutating prior leaf attempts."""

    invocation_binding, invocation, root = _invocation_location(invocation_path)
    if invocation.leaf_id != ARTICULATION_PUBLISH_LEAF_ID:
        raise ValueError("Articulation publish entrypoint received another leaf")
    try:
        assert invocation.author_result is not None
        assert invocation.evidence_result is not None
        assert invocation.review_result is not None
        author = _load_focused_result(
            invocation.author_result,
            expected_leaf_id=ARTICULATION_AUTHOR_LEAF_ID,
        )
        evidence_result = _load_focused_result(
            invocation.evidence_result,
            expected_leaf_id=ARTICULATION_EVIDENCE_LEAF_ID,
        )
        review_result = _load_focused_result(
            invocation.review_result,
            expected_leaf_id=ARTICULATION_REVIEW_LEAF_ID,
        )
        review_binding = _binding_named(
            review_result, "articulation_asset_review_receipt.json"
        )
        review = _load_bound(
            review_binding,
            ArticulationAssetReviewReceipt,
            label="Articulation asset review receipt",
        )
        if review.disposition != "accept":
            raise ValueError("Articulation publication requires accepted review")
        if (
            review.author_result != invocation.author_result
            or review.evidence_result != invocation.evidence_result
            or _binding_named(
                evidence_result,
                "standalone_articulation_output_evidence.json",
            )
            != review.output_evidence
        ):
            raise ValueError("Articulation publication inputs belong to another chain")
        identity_binding = review.identity
        graph_binding = review.canonical_graph
        receipt_binding = review.authoring_receipt
        readback_binding = review.readback
        evidence_binding = review.output_evidence
        identity = _load_bound(
            identity_binding,
            StandaloneArticulationIdentity,
            label="Articulation identity",
        )
        graph = _load_bound(
            graph_binding,
            EmbeddedArticulationCanonicalGraph,
            label="Articulation canonical graph",
        )
        authoring = _load_bound(
            receipt_binding,
            StandaloneArticulationAuthoringReceipt,
            label="Articulation authoring receipt",
        )
        output_evidence = _load_bound(
            evidence_binding,
            EmbeddedArticulationOutputEvidence,
            label="Articulation output evidence",
        )
        validate_embedded_articulation_output_evidence(output_evidence)
        decision_patch = _binding_named(author, "articulation_decision_patch.json")
        decision_ledger = _binding_named(author, "articulation_decision_ledger.json")
        patch = _load_bound(
            decision_patch,
            StandaloneArticulationDecisionPatch,
            label="Articulation decision patch",
        )
        ledger = _load_bound(
            decision_ledger,
            StandaloneArticulationDecisionLedger,
            label="Articulation decision ledger",
        )
        if ledger.decision_patch != _native_binding(decision_patch):
            raise ValueError(
                "Articulation child decision differs from its validated ledger"
            )
        refinement_history = _unique_bindings(
            tuple(
                item
                for item in author.evidence
                if Path(item.path).name == "issue_packet.json"
            )
        )
        if len(refinement_history) > 1:
            raise ValueError("Articulation author result repeats refinement custody")
        if refinement_history:
            issue_binding = refinement_history[0]
            issue = _load_bound(
                issue_binding,
                StandaloneArticulationIssuePacket,
                label="Articulation refinement issue packet",
            )
            parent_patch_binding = _binding(issue.parent_patch.path)
            parent_ledger_binding = _binding(issue.parent_ledger.path)
            parent_ledger = _load_bound(
                parent_ledger_binding,
                StandaloneArticulationDecisionLedger,
                label="Articulation refinement parent ledger",
            )
            if (
                issue.identity_digest != canonical_json_digest(identity)
                or _native_binding(parent_patch_binding) != issue.parent_patch
                or _native_binding(parent_ledger_binding) != issue.parent_ledger
                or parent_ledger.decision_patch != issue.parent_patch
                or patch.refinement_attempt != issue.refinement_attempt
                or patch.parent_patch_sha256 != issue.parent_patch.sha256
                or patch.issue_packet_sha256 != issue_binding.sha256
                or Path(issue.replacement_patch_path).resolve(strict=True)
                != Path(decision_patch.path).resolve(strict=True)
            ):
                raise ValueError("Articulation refinement custody is inconsistent")
        elif patch.refinement_attempt != 0:
            raise ValueError("Articulation refined decision omitted its issue packet")
        preparation = _binding_named(author, "standalone_articulation_preparation.json")
        optional_proposal = tuple(
            item
            for item in author.evidence
            if Path(item.path).name == "standalone_articulation_provider_proposal.json"
        )
        if len(optional_proposal) > 1:
            raise ValueError("Articulation author result repeats its proposal")
        patch_binding = review.post_review_patch
        native = root / "native"
        native.mkdir(mode=0o700)
        retained = (
            identity_binding,
            preparation,
            decision_patch,
            decision_ledger,
            graph_binding,
            receipt_binding,
            readback_binding,
            evidence_binding,
            patch_binding,
            review_binding,
            *refinement_history,
            *optional_proposal,
        )
        cleanup = StandaloneArticulationCleanupReceipt(
            identity_digest=canonical_json_digest(identity),
            output_asset=authoring.output_asset,
            retained_artifacts=tuple(_native_binding(item) for item in retained),
        )
        cleanup_binding = _write_once(
            native / "standalone_articulation_cleanup_receipt.json",
            cleanup,
            label="Articulation cleanup receipt",
        )
        terminal = StandaloneArticulationTerminalReceipt(
            identity=_native_binding(identity_binding),
            identity_digest=canonical_json_digest(identity),
            request=identity.request,
            source=identity.source,
            source_dependency_bundle_sha256=identity.source_dependency_bundle_sha256,
            preparation=_native_binding(preparation),
            optional_proposal=(
                _native_binding(optional_proposal[0]) if optional_proposal else None
            ),
            decision_patch=_native_binding(decision_patch),
            decision_ledger=_native_binding(decision_ledger),
            canonical_graph=_native_binding(graph_binding),
            canonical_graph_digest=canonical_json_digest(graph),
            authoring_receipt=_native_binding(receipt_binding),
            output_asset=authoring.output_asset,
            readback=_native_binding(readback_binding),
            output_evidence=_native_binding(evidence_binding),
            post_review=_native_binding(patch_binding),
            cleanup=_native_binding(cleanup_binding),
            refinement_history=tuple(
                _native_binding(item) for item in refinement_history
            ),
        )
        terminal_binding = _write_once(
            native / "standalone_articulation_terminal_receipt.json",
            terminal,
            label="Articulation terminal receipt",
        )
        result = ArticulationFocusedLeafResult(
            leaf_id=ARTICULATION_PUBLISH_LEAF_ID,
            invocation=invocation_binding,
            native_disposition="passed",
            native_status="completed",
            native_terminal_receipt=terminal_binding,
            evidence=(terminal_binding, cleanup_binding),
            saved_stage_readbacks=(cleanup_binding, terminal_binding),
            output=authoring.output_asset,
        )
        _write_once(
            root / "articulation_leaf_result.json",
            result,
            label="Articulation publish result",
        )
        return result
    except Exception as exc:
        detail = str(exc).strip() or "no exception detail"
        return _failure_result(
            root=root,
            invocation_binding=invocation_binding,
            invocation=invocation,
            error=f"{type(exc).__name__}: {detail}",
        )


def _project_preparation(
    invocation: BaseModel,
    result: BaseModel,
    context: AssetLeafProjectionContext,
) -> AssetLeafProjectionPayload:
    request = cast(ArticulationPreparationLeafInvocation, invocation)
    resolved = cast(ArticulationPreparationLeafResult, result).root
    if isinstance(resolved, ArticulationPreparationLeafTerminalResult):
        expected_invocation = ExecutionArtifactBinding.model_validate(
            context.invocation_artifact.model_dump(mode="json")
        )
        _verify_binding(
            expected_invocation,
            label="Articulation preparation invocation",
        )
        if resolved.invocation != expected_invocation:
            raise ValueError(
                "Articulation preparation failure belongs to another invocation"
            )
        retained = (
            resolved.native_terminal_receipt,
            *resolved.evidence,
            *resolved.saved_stage_readbacks,
        )
        for item in retained:
            _verify_binding(item, label="Articulation preparation failure artifact")
        attempt_root = Path(context.invocation_artifact.path).parent
        if any(not Path(item.path).is_relative_to(attempt_root) for item in retained):
            raise ValueError(
                "Articulation preparation failure artifacts leave the active attempt"
            )
        if resolved.evidence != (
            resolved.native_terminal_receipt,
        ) or resolved.saved_stage_readbacks != (resolved.native_terminal_receipt,):
            raise ValueError(
                "Articulation preparation failure must bind only its terminal receipt"
            )
        receipt = _load_bound(
            resolved.native_terminal_receipt,
            ArticulationPreparationFailureReceipt,
            label="Articulation preparation failure receipt",
        )
        if (
            receipt.invocation != expected_invocation
            or receipt.native_status != resolved.native_status
            or receipt.error != resolved.error
        ):
            raise ValueError(
                "Articulation preparation failure receipt differs from its result"
            )
        result_execution = ExecutionArtifactBinding.model_validate(
            context.result_artifact.model_dump(mode="json")
        )
        _verify_binding(
            result_execution,
            label="Articulation preparation terminal result",
        )
        return AssetLeafProjectionPayload(
            native_disposition="failed",
            native_status=resolved.native_status,
            native_terminal_receipt=_asset_binding(resolved.native_terminal_receipt),
            evidence=tuple(_asset_binding(item) for item in resolved.evidence),
            saved_stage_readbacks=tuple(
                _asset_binding(item) for item in resolved.saved_stage_readbacks
            ),
            summary="Articulation deterministic preparation failed closed.",
            error=resolved.error,
        )

    publication = resolved
    if publication.readback != request.readback or (
        publication.retained_root != request.retained_root
    ):
        raise ValueError(
            "Articulation preparation result belongs to another invocation"
        )
    for item in (
        publication.readback,
        publication.source,
        *publication.dependencies,
        publication.configuration,
        publication.inspector_implementation,
        publication.saved_stage,
        *publication.renders,
        *publication.scene_artifacts,
        publication.preparation,
    ):
        _verify_binding(item, label="Articulation preparation artifact")
    terminal_execution = ExecutionArtifactBinding.model_validate(
        context.result_artifact.model_dump(mode="json")
    )
    _verify_binding(terminal_execution, label="Articulation preparation result")
    terminal = _asset_binding(terminal_execution)
    preparation_artifact = _asset_binding(publication.preparation)
    return AssetLeafProjectionPayload(
        native_disposition="passed",
        native_status="completed",
        native_terminal_receipt=terminal,
        evidence=(terminal, preparation_artifact),
        saved_stage_readbacks=(preparation_artifact,),
        summary="Articulation deterministic preparation completed.",
    )


def _project_proposal(
    invocation: BaseModel,
    result: BaseModel,
    context: AssetLeafProjectionContext,
) -> AssetLeafProjectionPayload:
    request = cast(ArticulationProposalLeafInvocation, invocation)
    receipt = cast(ArticulationProposalAttemptTerminalReceipt, result)
    if receipt.preparation != request.preparation:
        raise ValueError("Articulation proposal result belongs to another preparation")
    terminal_execution = ExecutionArtifactBinding.model_validate(
        context.result_artifact.model_dump(mode="json")
    )
    _verify_binding(terminal_execution, label="Articulation proposal result")
    terminal = _asset_binding(terminal_execution)
    evidence = _unique_bindings(
        tuple(
            item
            for item in (
                receipt.request,
                *receipt.provider_inputs,
                *receipt.partial_outputs,
                receipt.native_payload,
                receipt.proposal,
                receipt.bound_preparation,
            )
            if item is not None
        )
    )
    for item in evidence:
        _verify_binding(item, label="proposal artifact")
    attempt_root = Path(context.invocation_artifact.path).parent
    verified_evidence = tuple(
        _asset_binding(item)
        for item in evidence
        if Path(item.path).is_relative_to(attempt_root)
    )
    failed = receipt.disposition != "succeeded"
    return AssetLeafProjectionPayload(
        native_disposition="failed" if failed else "passed",
        native_status=receipt.disposition,
        native_terminal_receipt=terminal,
        evidence=verified_evidence or (terminal,),
        saved_stage_readbacks=(
            *(
                (_asset_binding(receipt.bound_preparation),)
                if receipt.bound_preparation is not None
                else ()
            ),
            terminal,
        ),
        summary=f"Articulation proposal attempt ended {receipt.disposition}.",
        error=receipt.failure_summary if failed else None,
    )


def _project_focused(
    invocation: BaseModel,
    result: BaseModel,
    context: AssetLeafProjectionContext,
) -> AssetLeafProjectionPayload:
    request = cast(ArticulationFocusedLeafInvocation, invocation)
    native = cast(ArticulationFocusedLeafResult, result)
    expected_invocation = ExecutionArtifactBinding.model_validate(
        context.invocation_artifact.model_dump(mode="json")
    )
    if native.leaf_id != request.leaf_id or native.invocation != expected_invocation:
        raise ValueError("Articulation focused result belongs to another invocation")
    for item in (
        native.native_terminal_receipt,
        *native.evidence,
        *native.saved_stage_readbacks,
    ):
        _verify_binding(item, label=f"{native.leaf_id} retained artifact")
    if native.output is not None:
        _verify_binding(native.output, label=f"{native.leaf_id} output")
    attempt_root = Path(context.invocation_artifact.path).parent
    if any(
        not Path(item.path).is_relative_to(attempt_root)
        for item in (
            native.native_terminal_receipt,
            *native.evidence,
            *native.saved_stage_readbacks,
        )
    ):
        raise ValueError(
            f"{native.leaf_id} projected artifacts leave the active attempt"
        )
    return AssetLeafProjectionPayload(
        native_disposition=native.native_disposition,
        native_status=native.native_status,
        native_terminal_receipt=_asset_binding(native.native_terminal_receipt),
        evidence=tuple(_asset_binding(item) for item in native.evidence),
        saved_stage_readbacks=tuple(
            _asset_binding(item) for item in native.saved_stage_readbacks
        ),
        summary=(
            f"Articulation {native.leaf_id} ended with native disposition "
            f"{native.native_disposition}."
        ),
        error=native.error,
    )


def articulation_asset_leaf_runtime_bindings() -> tuple[AssetLeafRuntimeBinding, ...]:
    """Build the six exact repository-owned Articulation registrations."""

    bindings = [
        AssetLeafRuntimeBinding.create(
            leaf_id=ARTICULATION_PREPARATION_LEAF_ID,
            entrypoint=f"{_PREPARATION_ENTRYPOINT} --invocation",
            invocation_model=ArticulationPreparationLeafInvocation,
            result_model=ArticulationPreparationLeafResult,
            projector_id="asset.projector.articulation.preparation.v1",
            projector=_project_preparation,
            required_artifact_categories=("evidence", "saved_stage_readback"),
        ),
        AssetLeafRuntimeBinding.create(
            leaf_id=ARTICULATION_PROPOSAL_LEAF_ID,
            entrypoint=f"{_PROPOSAL_ENTRYPOINT} --invocation",
            invocation_model=ArticulationProposalLeafInvocation,
            result_model=ArticulationProposalAttemptTerminalReceipt,
            projector_id="asset.projector.articulation.proposal.v1",
            projector=_project_proposal,
            required_artifact_categories=("evidence", "saved_stage_readback"),
            required_dependencies=(ARTICULATION_PREPARATION_LEAF_ID,),
        ),
    ]
    for leaf_id in ARTICULATION_FOCUSED_LEAF_IDS:
        bindings.append(
            AssetLeafRuntimeBinding.create(
                leaf_id=leaf_id,
                entrypoint=_FOCUSED_ENTRYPOINTS[leaf_id],
                invocation_model=ArticulationFocusedLeafInvocation,
                result_model=ArticulationFocusedLeafResult,
                projector_id=f"asset.projector.{leaf_id}",
                projector=_project_focused,
                required_artifact_categories=("evidence", "saved_stage_readback"),
                required_dependencies=_FOCUSED_DEPENDENCIES[leaf_id],
                required_dependents=(
                    (_CANONICAL_OVRTX_EVIDENCE_LEAF_ID,)
                    if leaf_id == ARTICULATION_AUTHOR_LEAF_ID
                    else ()
                ),
            )
        )
    return tuple(sorted(bindings, key=lambda item: item.descriptor.leaf_id))


def articulation_asset_leaf_runtime_bundle() -> AssetLeafRuntimeBundle:
    """Return the side-effect-free bundle declared by the joint-agent package."""

    return AssetLeafRuntimeBundle.create(
        bundle_id=ARTICULATION_ASSET_LEAF_BUNDLE_ID,
        bindings=articulation_asset_leaf_runtime_bindings(),
    )


def articulation_asset_leaf_descriptors() -> tuple[AssetLeafDescriptor, ...]:
    return tuple(item.descriptor for item in articulation_asset_leaf_runtime_bindings())


def articulation_asset_leaf_catalog() -> AssetLeafCatalog:
    """Compose Articulation with its shared static Validation requirement."""

    from content_agent_workflows.asset_composition import (
        compose_asset_leaf_runtime_bundles,
    )
    from content_agent_workflows.asset_composition.catalog_adapters import (
        shared_asset_leaf_runtime_bundle,
    )

    return compose_asset_leaf_runtime_bundles(
        (shared_asset_leaf_runtime_bundle(), articulation_asset_leaf_runtime_bundle())
    ).catalog


__all__ = [
    "ARTICULATION_ASSET_LEAF_BUNDLE_ID",
    "ARTICULATION_AUTHOR_LEAF_ID",
    "ARTICULATION_EVIDENCE_LEAF_ID",
    "ARTICULATION_FOCUSED_LEAF_IDS",
    "ARTICULATION_PREPARATION_LEAF_ID",
    "ARTICULATION_PREPARATION_FAILURE_RECEIPT_SCHEMA_VERSION",
    "ARTICULATION_PREPARATION_LEAF_INVOCATION_SCHEMA_VERSION",
    "ARTICULATION_PREPARATION_TERMINAL_RESULT_SCHEMA_VERSION",
    "ARTICULATION_PROPOSAL_LEAF_ID",
    "ARTICULATION_PROPOSAL_LEAF_INVOCATION_SCHEMA_VERSION",
    "ARTICULATION_PUBLISH_LEAF_ID",
    "ARTICULATION_REVIEW_LEAF_ID",
    "ArticulationArtifactJsonProposalInvocation",
    "ArticulationAssetPhaseReceipt",
    "ArticulationAssetReviewReceipt",
    "ArticulationAuthorLeafProgress",
    "ArticulationFocusedLeafInvocation",
    "ArticulationFocusedLeafResult",
    "ArticulationLeafInvocation",
    "ArticulationPreparationFailureReceipt",
    "ArticulationPreparationLeafInvocation",
    "ArticulationPreparationLeafResult",
    "ArticulationPreparationLeafTerminalResult",
    "ArticulationProposalLeafInvocation",
    "articulation_asset_leaf_catalog",
    "articulation_asset_leaf_descriptors",
    "articulation_asset_leaf_runtime_bindings",
    "articulation_asset_leaf_runtime_bundle",
    "fail_articulation_preparation_asset_leaf",
    "run_articulation_author_asset_leaf",
    "run_articulation_evidence_asset_leaf",
    "run_articulation_publish_asset_leaf",
    "run_articulation_review_asset_leaf",
]
