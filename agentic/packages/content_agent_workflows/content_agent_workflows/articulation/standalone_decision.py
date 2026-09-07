# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provider-neutral standalone Articulation decision custody.

The reasoning child may write only :class:`StandaloneArticulationDecisionPatch`.
Every mutation, saved-stage readback, evidence binding, review, and terminal
publication remains an outer-owned deterministic operation.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from content_agent_workflows.common.artifacts import (
    atomic_write_json,
    file_sha256,
)
from content_agent_workflows.common.domain_execution import (
    DomainExecutionContext,
    ExecutionArtifactBinding,
)
from content_agent_workflows.common.embedded_domain_decision import (
    ProducerIdentity,
    canonical_json_digest,
)

from .client import ArticulationAuthoringClient, CancelChecker
from .embedded_decision import (
    EMBEDDED_ARTICULATION_PREPARATION_SCHEMA_VERSION,
    EmbeddedArticulationCanonicalGraph,
    EmbeddedArticulationCapabilityLimits,
    EmbeddedArticulationError,
    EmbeddedArticulationPreparation,
    _execution_binding,
    _graph_candidate_document,
    _persist_state,
    _readback_groups,
    _readback_memberships,
    _readback_topology,
    _state,
    _transition,
    _validate_canonical_graph,
    _validate_saved_membership,
    _verify_execution_binding,
    state_to_source_binding,
)
from .finalizer import write_articulation_workflow_summary
from .models import (
    ARTICULATION_STANDALONE_RUN_STATE_SCHEMA_VERSION,
    ArticulationAuthoringRequest,
    ArticulationAuthoringResult,
    ArticulationFinalizationResult,
    ArticulationRunState,
    ArticulationWorkflowMode,
    ArticulationWorkflowRequest,
    ArtifactBinding,
)
from .output_evidence import (
    EmbeddedArticulationOutputEvidence,
    build_embedded_articulation_output_evidence,
    validate_embedded_articulation_output_evidence,
)

STANDALONE_ARTICULATION_PREPARATION_SCHEMA_VERSION: Literal[
    "content-agent-workflows.embedded-articulation-preparation.v1"
] = EMBEDDED_ARTICULATION_PREPARATION_SCHEMA_VERSION
STANDALONE_ARTICULATION_IDENTITY_SCHEMA_VERSION: Literal[
    "content-agent-workflows.standalone-articulation-identity.v1"
] = "content-agent-workflows.standalone-articulation-identity.v1"
STANDALONE_ARTICULATION_ASSET_IDENTITY_SCHEMA_VERSION: Literal[
    "content-agent-workflows.standalone-articulation-identity.v2"
] = "content-agent-workflows.standalone-articulation-identity.v2"
STANDALONE_ARTICULATION_PATCH_SCHEMA_VERSION: Literal[
    "content-agent-workflows.articulation-decision-patch.v2"
] = "content-agent-workflows.articulation-decision-patch.v2"
STANDALONE_ARTICULATION_LEDGER_SCHEMA_VERSION: Literal[
    "content-agent-workflows.articulation-decision-ledger.v2"
] = "content-agent-workflows.articulation-decision-ledger.v2"
STANDALONE_ARTICULATION_READBACK_SCHEMA_VERSION: Literal[
    "content-agent-workflows.standalone-articulation-readback.v1"
] = "content-agent-workflows.standalone-articulation-readback.v1"
STANDALONE_ARTICULATION_AUTHORING_RECEIPT_SCHEMA_VERSION: Literal[
    "content-agent-workflows.standalone-articulation-authoring-receipt.v1"
] = "content-agent-workflows.standalone-articulation-authoring-receipt.v1"
STANDALONE_ARTICULATION_POST_REVIEW_SCHEMA_VERSION: Literal[
    "content-agent-workflows.standalone-articulation-post-review.v1"
] = "content-agent-workflows.standalone-articulation-post-review.v1"
STANDALONE_ARTICULATION_CLEANUP_SCHEMA_VERSION: Literal[
    "content-agent-workflows.standalone-articulation-cleanup-receipt.v1"
] = "content-agent-workflows.standalone-articulation-cleanup-receipt.v1"
STANDALONE_ARTICULATION_TERMINAL_RECEIPT_SCHEMA_VERSION: Literal[
    "content-agent-workflows.standalone-articulation-terminal-receipt.v1"
] = "content-agent-workflows.standalone-articulation-terminal-receipt.v1"
STANDALONE_ARTICULATION_ISSUE_PACKET_SCHEMA_VERSION: Literal[
    "content-agent-workflows.standalone-articulation-issue-packet.v1"
] = "content-agent-workflows.standalone-articulation-issue-packet.v1"

_IMPLEMENTATION_MANIFEST = {
    "controller": "standalone-articulation-v1",
    "reasoning_boundary": "child-decision-patch-only",
    "default_provider_backend": None,
    "authoring": "joint-agent-owned-graph-adapter",
    "readback": "saved-stage-exact-graph-and-membership",
    "output_evidence": "canonical-ovrtx-required",
    "review": "independent-post-authoring",
    "max_refinement_attempts": 1,
    "masses": False,
    "colliders": False,
    "drives": False,
}


class StandaloneArticulationError(EmbeddedArticulationError):
    """Raised when standalone Articulation custody is incomplete or stale."""


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _now() -> datetime:
    return datetime.now(UTC)


def _binding(path: Path) -> ArtifactBinding:
    return ArtifactBinding(path=str(path.resolve()), sha256=file_sha256(path))


def _standalone_authoring_key(
    state: ArticulationRunState,
    graph_digest: str,
    output_dir: Path,
    accepted_authoring_plan_sha256: str | None = None,
) -> str:
    key = {
        "schema_version": (
            "content-agent-workflows.standalone-articulation-authoring-key.v1"
        ),
        "source_sha256": state.source_sha256,
        "dependency_sha256": state.source_dependency_bundle_sha256,
        "canonical_graph_digest": graph_digest,
        "output_dir": str(output_dir),
        "predictions_path": None,
    }
    if accepted_authoring_plan_sha256 is not None:
        key["accepted_authoring_plan_sha256"] = accepted_authoring_plan_sha256
    return cast(
        str,
        canonical_json_digest(key),
    )


def _accepted_rigid_link_authoring_plan(
    root: Path,
    state: ArticulationRunState,
    graph: EmbeddedArticulationCanonicalGraph,
    graph_binding: ArtifactBinding,
) -> ArtifactBinding | None:
    """Project only accepted rigid-link operations into the owned core request."""

    if not graph.rigid_link_operations:
        return None
    from world_understanding.functions.physics.joint_rigger import (
        INPUT_SCHEMA_VERSION_V2,
        PLAN_SCHEMA_VERSION_V2,
        ArticulationRootPlanV1,
        ArtifactIdentityV1,
        FieldProvenanceV1,
        JointLimitV1,
        JointPlanV1,
        JointRiggerInputV2,
        JointRiggerPlanV2,
        JointTopologyV1,
        RigidBodyPlanV1,
        RigidLinkMemberPlanV1,
        RigidLinkPlanV1,
        identify_usd_artifact,
    )

    source_path = Path(state.source_asset).expanduser().resolve()
    source_identity = identify_usd_artifact(source_path, uri=source_path.as_uri())
    if (
        source_identity.root_sha256 != state.source_sha256
        or source_identity.dependency_bundle_sha256
        != state.source_dependency_bundle_sha256
    ):
        raise StandaloneArticulationError(
            "Accepted rigid-link plan source identity changed before projection"
        )
    graph_identity = ArtifactIdentityV1(
        uri=Path(graph_binding.path).resolve().as_uri(),
        root_sha256=graph_binding.sha256,
    )

    def provenance(
        prim_path: str, property_names: str | tuple[str, ...]
    ) -> FieldProvenanceV1:
        return FieldProvenanceV1(
            source="accepted_manifest",
            artifact=graph_identity,
            prim_path=prim_path,
            properties=(
                (property_names,) if isinstance(property_names, str) else property_names
            ),
            derivation="articulation_contract_v1_to_joint_rigger_input_v1",
            evidence=(
                "Outer-frozen standalone Articulation graph accepted this exact fact."
            ),
        )

    axes = {
        "x": (1.0, 0.0, 0.0),
        "-x": (-1.0, 0.0, 0.0),
        "y": (0.0, 1.0, 0.0),
        "-y": (0.0, -1.0, 0.0),
        "z": (0.0, 0.0, 1.0),
        "-z": (0.0, 0.0, -1.0),
    }
    endpoint_paths = tuple(
        dict.fromkeys(
            endpoint
            for joint in graph.joints
            for endpoint in (joint.body0_owner_prim, joint.body1_owner_prim)
        )
    )
    # The shared owned-core evidence contract names each rigid link by the
    # source member identity used by Stage 2.  Keeping that stable lets the
    # adapter resolve diagnostics back to exactly one accepted candidate.
    link_ids = {path: path for path in endpoint_paths}
    joints = tuple(
        JointPlanV1(
            topology=JointTopologyV1(
                joint_id=joint.joint_id,
                joint_type=joint.joint_type,
                body0=joint.body0_owner_prim,
                body1=joint.body1_owner_prim,
                axis_stage=axes[joint.axis],
                field_provenance={
                    "joint_type": provenance(
                        joint.body1_owner_prim, f"joint:{joint.joint_id}.motion_type"
                    ),
                    "body0": provenance(
                        joint.body0_owner_prim,
                        (
                            f"joint:{joint.joint_id}.body0_link",
                            f"link:{link_ids[joint.body0_owner_prim]}.body_prim_path",
                        ),
                    ),
                    "body1": provenance(
                        joint.body1_owner_prim,
                        (
                            f"joint:{joint.joint_id}.body1_link",
                            f"link:{link_ids[joint.body1_owner_prim]}.body_prim_path",
                        ),
                    ),
                    "axis_stage": provenance(
                        joint.body1_owner_prim, f"joint:{joint.joint_id}.axis_stage"
                    ),
                },
            ),
            limit=(
                JointLimitV1(
                    lower=joint.lower_limit,
                    upper=joint.upper_limit,
                    unit=joint.limit_unit,
                    provenance=provenance(
                        joint.body1_owner_prim, f"joint:{joint.joint_id}.limits"
                    ),
                )
                if joint.lower_limit is not None or joint.upper_limit is not None
                else None
            ),
        )
        for joint in graph.joints
    )
    moving_paths = {joint.body1_owner_prim for joint in graph.joints}
    root_paths = tuple(path for path in endpoint_paths if path not in moving_paths)
    operation_body_paths = {
        operation.body_prim_path for operation in graph.rigid_link_operations
    }
    request = JointRiggerInputV2(
        schema_version=INPUT_SCHEMA_VERSION_V2,
        source_asset=source_identity,
        plan=JointRiggerPlanV2(
            schema_version=PLAN_SCHEMA_VERSION_V2,
            joints=joints,
            rigid_bodies=tuple(
                RigidBodyPlanV1(
                    prim_path=path,
                    provenance=provenance(
                        path,
                        (
                            "physics:rigidBodyEnabled",
                            "operation:apply_rigid_body_membership",
                        )
                        if path in operation_body_paths
                        else "physics:rigidBodyEnabled",
                    ),
                )
                for path in endpoint_paths
            ),
            articulation_roots=tuple(
                ArticulationRootPlanV1(
                    prim_path=path,
                    provenance=provenance(path, "physics:articulationRoot"),
                )
                for path in root_paths
            ),
        ),
        rigid_links=tuple(
            RigidLinkPlanV1(
                link_id=link_ids[path],
                body_authoring="existing",
                body_prim_path=path,
                members=(
                    RigidLinkMemberPlanV1(
                        source_prim_path=path,
                        authored_prim_path=path,
                    ),
                ),
            )
            for path in endpoint_paths
        ),
        legacy_component_names=None,
    )
    return _write_once_model(
        root / "accepted_joint_rigger_input.json",
        request,
        label="accepted rigid-link Joint Rigger input",
    )


def _load_bound[ModelT: BaseModel](
    binding: ArtifactBinding | None,
    model_type: type[ModelT],
) -> ModelT:
    if binding is None:
        raise StandaloneArticulationError(
            f"Required {model_type.__name__} artifact binding is missing"
        )
    path = Path(binding.path).expanduser().resolve()
    if not path.is_file() or file_sha256(path) != binding.sha256:
        raise StandaloneArticulationError(f"Bound artifact changed: {path}")
    return cast(ModelT, model_type.model_validate_json(path.read_bytes()))


def _persist_standalone_state(
    root: Path,
    state: ArticulationRunState,
) -> ArticulationRunState:
    return cast(ArticulationRunState, _persist_state(root, state))


def _write_once_model(
    path: Path,
    model: BaseModel,
    *,
    label: str,
) -> ArtifactBinding:
    if path.exists():
        existing = type(model).model_validate_json(path.read_bytes())
        if existing != model:
            raise StandaloneArticulationError(
                f"Existing {label} differs from the exact requested artifact"
            )
    else:
        atomic_write_json(path, model)
    return _binding(path)


class StandaloneArticulationPreparation(EmbeddedArticulationPreparation):
    """Standalone typing over the shared provider-neutral preparation wire."""

    schema_version: Literal[
        "content-agent-workflows.embedded-articulation-preparation.v1"
    ] = STANDALONE_ARTICULATION_PREPARATION_SCHEMA_VERSION


class StandaloneArticulationReviewPolicy(_FrozenModel):
    independent_review_required: Literal[True] = True
    canonical_output_evidence_required: Literal[True] = True
    max_refinement_attempts: Literal[1] = 1


class StandaloneArticulationIdentity(_FrozenModel):
    """Frozen request/source/preparation and implementation identity."""

    schema_version: Literal[
        "content-agent-workflows.standalone-articulation-identity.v1",
        "content-agent-workflows.standalone-articulation-identity.v2",
    ] = STANDALONE_ARTICULATION_IDENTITY_SCHEMA_VERSION
    execution_context: DomainExecutionContext
    request: ArtifactBinding
    source: ExecutionArtifactBinding
    source_dependency_bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    preparation: ArtifactBinding
    configuration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    capability_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    implementation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    default_provider_backend: None = None

    @model_validator(mode="after")
    def validate_standalone_identity(self) -> Self:
        if self.execution_context.domain != "articulation" or (
            self.execution_context.mode != "standalone"
        ):
            raise ValueError(
                "Standalone Articulation identity requires standalone ownership"
            )
        asset_owned = (
            self.schema_version == STANDALONE_ARTICULATION_ASSET_IDENTITY_SCHEMA_VERSION
        )
        expected_owner = "asset_coordinator" if asset_owned else "domain_child_agent"
        if self.execution_context.reasoning_loop_owner != expected_owner:
            raise ValueError(
                "Standalone Articulation identity reasoning owner is inconsistent"
            )
        return self


class StandaloneArticulationCandidateDecision(_FrozenModel):
    candidate_id: str = Field(min_length=1)
    disposition: Literal["accept", "reject", "refine"]
    evidence_ids: tuple[str, ...] = Field(min_length=1)
    issue_codes: tuple[
        Literal[
            "parent_unresolved",
            "axis_missing",
            "endpoint_unresolved",
            "frame_unresolved",
            "limit_unresolved",
            "membership_unresolved",
            "readback_mismatch",
        ],
        ...,
    ] = ()
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_candidate_decision(self) -> Self:
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("candidate evidence IDs must be unique")
        if len(self.issue_codes) != len(set(self.issue_codes)):
            raise ValueError("candidate issue codes must be unique")
        if self.disposition == "refine" and not self.issue_codes:
            raise ValueError("refine decisions require candidate-bound issue codes")
        if self.disposition != "refine" and self.issue_codes:
            raise ValueError("issue codes are reserved for refine decisions")
        return self


class StandaloneArticulationDecisionPatch(_FrozenModel):
    """The only semantic artifact the reasoning child may author."""

    schema_version: Literal[
        "content-agent-workflows.articulation-decision-patch.v2"
    ] = STANDALONE_ARTICULATION_PATCH_SCHEMA_VERSION
    expected_state_revision: int = Field(ge=0)
    refinement_attempt: int = Field(default=0, ge=0, le=1)
    parent_patch_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    issue_packet_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    identity_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    preparation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_dependency_bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    configuration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    disposition: Literal["accept", "reject", "revise"]
    candidate_decisions: tuple[StandaloneArticulationCandidateDecision, ...] = Field(
        min_length=1
    )
    canonical_graph: EmbeddedArticulationCanonicalGraph | None = None
    evidence_requirements: tuple[str, ...] = Field(min_length=1)
    review_policy: StandaloneArticulationReviewPolicy = Field(
        default_factory=StandaloneArticulationReviewPolicy
    )
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_patch_scope(self) -> Self:
        candidate_ids = tuple(item.candidate_id for item in self.candidate_decisions)
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("candidate decisions must contain unique IDs")
        if len(self.evidence_requirements) != len(set(self.evidence_requirements)):
            raise ValueError("evidence requirements must contain unique IDs")
        accepted = tuple(
            item.candidate_id
            for item in self.candidate_decisions
            if item.disposition == "accept"
        )
        refining = tuple(
            item.candidate_id
            for item in self.candidate_decisions
            if item.disposition == "refine"
        )
        if self.disposition == "accept":
            if self.canonical_graph is None or refining:
                raise ValueError(
                    "accepted patches require a complete graph and no refine decisions"
                )
            if self.canonical_graph.candidate_ids != accepted:
                raise ValueError(
                    "canonical graph IDs must exactly match accepted decision order"
                )
        elif self.canonical_graph is not None:
            raise ValueError("rejected or revised patches cannot freeze a graph")
        if self.disposition == "revise" and not refining:
            raise ValueError("revised patches require candidate-bound refine decisions")
        if self.disposition != "revise" and refining:
            raise ValueError("refine decisions require a revised patch")
        if self.refinement_attempt == 0:
            if (
                self.parent_patch_sha256 is not None
                or self.issue_packet_sha256 is not None
            ):
                raise ValueError("initial patches cannot claim refinement parents")
        elif self.parent_patch_sha256 is None or self.issue_packet_sha256 is None:
            raise ValueError("refinement patches require parent and issue bindings")
        return self


class StandaloneArticulationDecisionLedger(_FrozenModel):
    schema_version: Literal[
        "content-agent-workflows.articulation-decision-ledger.v2"
    ] = STANDALONE_ARTICULATION_LEDGER_SCHEMA_VERSION
    identity_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    preparation: ArtifactBinding
    decision_patch: ArtifactBinding
    refinement_attempt: int = Field(ge=0, le=1)
    disposition: Literal["accept", "reject", "revise", "cap_exhausted"]
    candidate_ids: tuple[str, ...] = Field(min_length=1)
    accepted_candidate_ids: tuple[str, ...] = ()
    rejected_candidate_ids: tuple[str, ...] = ()
    unresolved_candidate_ids: tuple[str, ...] = ()
    canonical_graph: ArtifactBinding | None = None
    canonical_graph_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    evidence_requirements: tuple[str, ...] = Field(min_length=1)
    validation_status: Literal["accepted", "non_success"]
    validated_at: datetime

    @model_validator(mode="after")
    def validate_ledger(self) -> Self:
        if self.validated_at.tzinfo is None or self.validated_at.utcoffset() is None:
            raise ValueError("ledger validation timestamp must be timezone-aware")
        accepted = self.disposition == "accept"
        if accepted != (self.validation_status == "accepted"):
            raise ValueError("only accepted decisions have accepted validation")
        if accepted != (self.canonical_graph is not None):
            raise ValueError("only accepted decisions bind a canonical graph")
        if accepted != (self.canonical_graph_digest is not None):
            raise ValueError("only accepted decisions bind a graph digest")
        return self


class StandaloneArticulationIssuePacket(_FrozenModel):
    schema_version: Literal[
        "content-agent-workflows.standalone-articulation-issue-packet.v1"
    ] = STANDALONE_ARTICULATION_ISSUE_PACKET_SCHEMA_VERSION
    refinement_attempt: Literal[1] = 1
    identity_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_patch: ArtifactBinding
    parent_ledger: ArtifactBinding
    failed_artifacts: tuple[ArtifactBinding, ...] = ()
    candidate_issue_codes: dict[str, tuple[str, ...]] = Field(min_length=1)
    required_evidence_ids: tuple[str, ...] = Field(min_length=1)
    allowed_operations: tuple[
        Literal[
            "hierarchy_inspection",
            "membership_inspection",
            "focused_evidence_collection",
            "saved_stage_readback",
        ],
        ...,
    ]
    replacement_patch_path: str = Field(min_length=1)


class StandaloneArticulationReadback(_FrozenModel):
    schema_version: Literal[
        "content-agent-workflows.standalone-articulation-readback.v1"
    ] = STANDALONE_ARTICULATION_READBACK_SCHEMA_VERSION
    identity_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    ledger_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    canonical_graph_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_asset: ExecutionArtifactBinding
    validation: ArtifactBinding
    joint_ids: tuple[str, ...] = Field(min_length=1)
    topology: tuple[dict[str, Any], ...] = Field(min_length=1)
    groups: tuple[dict[str, Any], ...] = Field(min_length=1)
    memberships: tuple[dict[str, Any], ...] = Field(min_length=1)
    rigid_link_operations: tuple[dict[str, Any], ...] = ()
    exact_topology_match: Literal[True] = True
    exact_membership_match: Literal[True] = True
    exact_co_rigid_disposition_match: Literal[True] = True


class StandaloneArticulationAuthoringReceipt(_FrozenModel):
    schema_version: Literal[
        "content-agent-workflows.standalone-articulation-authoring-receipt.v1"
    ] = STANDALONE_ARTICULATION_AUTHORING_RECEIPT_SCHEMA_VERSION
    identity_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision_patch: ArtifactBinding
    decision_ledger: ArtifactBinding
    canonical_graph: ArtifactBinding
    approved_candidates: ArtifactBinding
    authoring_request: ArtifactBinding
    authoring_result: ArtifactBinding
    validation: ArtifactBinding
    readback: ArtifactBinding
    output_asset: ExecutionArtifactBinding
    accepted_candidate_ids: tuple[str, ...] = Field(min_length=1)
    accepted_authoring_plan: ArtifactBinding | None = None
    membership_operation_receipt: ArtifactBinding | None = None
    rigid_link_operation_ids: tuple[str, ...] = ()
    masses_authored: Literal[False] = False
    colliders_authored: Literal[False] = False
    drives_authored: Literal[False] = False
    unrelated_topology_authored: Literal[False] = False
    status: Literal["verified"] = "verified"

    @model_validator(mode="after")
    def validate_membership_operation_custody(self) -> Self:
        selected = self.accepted_authoring_plan is not None
        if selected != (
            self.membership_operation_receipt is not None
        ) or selected != bool(self.rigid_link_operation_ids):
            raise ValueError(
                "accepted authoring plan, membership operation receipt, and "
                "rigid-link operation IDs must be selected together"
            )
        if len(self.rigid_link_operation_ids) != len(
            set(self.rigid_link_operation_ids)
        ):
            raise ValueError("rigid-link operation IDs must be unique")
        return self


class StandaloneArticulationPostReviewPatch(_FrozenModel):
    schema_version: Literal[
        "content-agent-workflows.standalone-articulation-post-review.v1"
    ] = STANDALONE_ARTICULATION_POST_REVIEW_SCHEMA_VERSION
    identity_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    authoring_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    canonical_graph_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    readback_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_evidence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    inspected_render_image_sha256s: tuple[str, ...] = Field(min_length=1)
    disposition: Literal["accept", "reject", "revise"]
    reviewer: ProducerIdentity
    findings: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_reviewer(self) -> Self:
        if self.reviewer.role != "outer_coordinator":
            raise ValueError("standalone post review requires an outer reviewer")
        if self.disposition == "accept" and len(
            self.inspected_render_image_sha256s
        ) != len(set(self.inspected_render_image_sha256s)):
            raise ValueError("accepted render image digests must be unique")
        return self


class StandaloneArticulationCleanupReceipt(_FrozenModel):
    schema_version: Literal[
        "content-agent-workflows.standalone-articulation-cleanup-receipt.v1"
    ] = STANDALONE_ARTICULATION_CLEANUP_SCHEMA_VERSION
    identity_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_asset: ExecutionArtifactBinding
    retained_artifacts: tuple[ArtifactBinding, ...] = Field(min_length=1)
    removed_transient_paths: tuple[str, ...] = ()
    status: Literal["completed"] = "completed"


class StandaloneArticulationTerminalReceipt(_FrozenModel):
    schema_version: Literal[
        "content-agent-workflows.standalone-articulation-terminal-receipt.v1"
    ] = STANDALONE_ARTICULATION_TERMINAL_RECEIPT_SCHEMA_VERSION
    identity: ArtifactBinding
    identity_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    request: ArtifactBinding
    source: ExecutionArtifactBinding
    source_dependency_bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    preparation: ArtifactBinding
    optional_proposal: ArtifactBinding | None = None
    decision_patch: ArtifactBinding
    decision_ledger: ArtifactBinding
    canonical_graph: ArtifactBinding | None = None
    canonical_graph_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    authoring_receipt: ArtifactBinding | None = None
    output_asset: ExecutionArtifactBinding | None = None
    readback: ArtifactBinding | None = None
    output_evidence: ArtifactBinding | None = None
    post_review: ArtifactBinding | None = None
    cleanup: ArtifactBinding | None = None
    refinement_history: tuple[ArtifactBinding, ...] = ()
    failed_artifacts: tuple[ArtifactBinding, ...] = ()
    issue_codes: tuple[str, ...] = ()
    source_mutated: Literal[False] = False
    terminal_disposition: Literal["accept", "reject", "refinement_cap"] = "accept"
    status: Literal["completed", "non_success"] = "completed"
    error: str | None = None

    @model_validator(mode="after")
    def validate_terminal_disposition(self) -> Self:
        if len(self.issue_codes) != len(set(self.issue_codes)):
            raise ValueError("terminal issue codes must be unique")
        graph_bound = self.canonical_graph is not None
        if graph_bound != (self.canonical_graph_digest is not None):
            raise ValueError("terminal canonical graph and digest must be paired")
        if self.terminal_disposition == "accept":
            required = (
                self.canonical_graph,
                self.authoring_receipt,
                self.output_asset,
                self.readback,
                self.output_evidence,
                self.post_review,
                self.cleanup,
            )
            if self.status != "completed" or any(item is None for item in required):
                raise ValueError(
                    "accepted terminal receipts require the complete output chain"
                )
            if self.error is not None or self.issue_codes or self.failed_artifacts:
                raise ValueError(
                    "accepted terminal receipts cannot carry non-success diagnostics"
                )
            return self
        if self.status != "non_success" or not self.error:
            raise ValueError(
                "non-success terminal receipts require an actionable error"
            )
        if self.terminal_disposition == "refinement_cap" and not self.issue_codes:
            raise ValueError("refinement-cap receipts require candidate issue codes")
        if any(
            item is not None
            for item in (
                self.authoring_receipt,
                self.output_asset,
                self.readback,
                self.output_evidence,
                self.post_review,
                self.cleanup,
            )
        ):
            raise ValueError(
                "non-success terminal receipts cannot claim verified output custody"
            )
        return self


class StandaloneArticulationObservation(_FrozenModel):
    schema_version: Literal[
        "content-agent-workflows.standalone-articulation-observation.v1"
    ] = "content-agent-workflows.standalone-articulation-observation.v1"
    state_revision: int = Field(ge=0)
    refinement_attempt: int = Field(ge=0, le=1)
    identity_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    preparation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_dependency_bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    configuration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_records: tuple[dict[str, Any], ...] = Field(min_length=1)
    optional_proposal: dict[str, Any] | None = None
    issue_packet: dict[str, Any] | None = None
    decision_patch_path: str = Field(min_length=1)
    child_authority: Literal["decision_patch_only"] = "decision_patch_only"
    forbidden_operations: tuple[
        Literal["usd_mutation", "authoring", "publication", "finalization"], ...
    ] = ("usd_mutation", "authoring", "publication", "finalization")


class _PreparationEvidenceView:
    def __init__(self, records: Sequence[Any]) -> None:
        self.records = tuple(records)


def _capabilities(
    preparation: StandaloneArticulationPreparation,
) -> EmbeddedArticulationCapabilityLimits:
    if preparation.capabilities.status != "available":
        raise StandaloneArticulationError(
            "Standalone Articulation capabilities are unavailable"
        )
    capabilities = EmbeddedArticulationCapabilityLimits.model_validate(
        dict(preparation.capabilities.facts)
    )
    if not capabilities.canonical_output_evidence_required:
        raise StandaloneArticulationError(
            "Standalone Articulation requires canonical output evidence"
        )
    return capabilities


def _validate_preparation_bindings(
    state: ArticulationRunState,
    preparation: StandaloneArticulationPreparation,
) -> None:
    if (
        preparation.source_sha256 != state.source_sha256
        or preparation.source_dependency_bundle_sha256
        != state.source_dependency_bundle_sha256
        or preparation.configuration_sha256 != state.backend_configuration_sha256
    ):
        raise StandaloneArticulationError(
            "Standalone Articulation preparation identity is stale"
        )
    _verify_execution_binding(state_to_source_binding(state), label="source asset")
    from .workflow import _source_identity

    try:
        current_source_sha256, current_dependency_sha256 = _source_identity(
            state.source_asset
        )
    except Exception as exc:
        raise StandaloneArticulationError(
            f"Standalone Articulation source identity cannot be recomputed: {exc}"
        ) from exc
    if (
        current_source_sha256 != state.source_sha256
        or current_dependency_sha256 != state.source_dependency_bundle_sha256
    ):
        raise StandaloneArticulationError(
            "Standalone Articulation source dependency identity changed"
        )
    for record in preparation.evidence_records:
        for artifact in record.artifacts:
            _verify_execution_binding(
                artifact,
                label=f"Articulation evidence {record.evidence_id}",
            )
    _capabilities(preparation)


def _build_identity(
    request: ArticulationWorkflowRequest,
    request_binding: ArtifactBinding,
    preparation: StandaloneArticulationPreparation,
    preparation_binding: ArtifactBinding,
) -> StandaloneArticulationIdentity:
    context = request.execution_context
    if context is None:
        raise StandaloneArticulationError(
            "Standalone preparation requires a typed execution context"
        )
    source_path = Path(request.source_asset).expanduser().resolve()
    return StandaloneArticulationIdentity(
        schema_version=(
            STANDALONE_ARTICULATION_ASSET_IDENTITY_SCHEMA_VERSION
            if context.reasoning_loop_owner == "asset_coordinator"
            else STANDALONE_ARTICULATION_IDENTITY_SCHEMA_VERSION
        ),
        execution_context=context,
        request=request_binding,
        source=ExecutionArtifactBinding(
            path=str(source_path),
            sha256=preparation.source_sha256,
            size_bytes=source_path.stat().st_size,
        ),
        source_dependency_bundle_sha256=(preparation.source_dependency_bundle_sha256),
        preparation=preparation_binding,
        configuration_sha256=preparation.configuration_sha256,
        capability_sha256=canonical_json_digest(_capabilities(preparation)),
        implementation_sha256=canonical_json_digest(_IMPLEMENTATION_MANIFEST),
    )


def prepare_standalone_articulation_workflow(
    request: ArticulationWorkflowRequest,
    *,
    mode: ArticulationWorkflowMode,
    preparation: StandaloneArticulationPreparation,
) -> ArticulationFinalizationResult:
    """Freeze deterministic preparation before the child reasons."""

    context = request.execution_context
    valid_owner = (
        context is not None
        and context.mode == "standalone"
        and context.reasoning_loop_owner in {"domain_child_agent", "asset_coordinator"}
    )
    if not valid_owner:
        raise StandaloneArticulationError(
            "Provider-neutral standalone preparation requires exact reasoning ownership"
        )
    from .workflow import _normalize_request, _source_identity, _write_once_json

    normalized = _normalize_request(request)
    root = normalized.output_dir
    root.mkdir(parents=True, exist_ok=True)
    request_binding = _write_once_json(
        root / "request.json", normalized, label="articulation request"
    )
    source_sha256, dependency_sha256 = _source_identity(normalized.source_asset)
    if (
        preparation.source_sha256 != source_sha256
        or preparation.source_dependency_bundle_sha256 != dependency_sha256
    ):
        raise StandaloneArticulationError(
            "Provider-neutral standalone preparation source identity is stale"
        )
    preparation_binding = _write_once_model(
        root / "standalone_articulation_preparation.json",
        preparation,
        label="standalone Articulation preparation",
    )
    proposal_binding = None
    if preparation.proposal is not None:
        proposal_binding = _write_once_model(
            root / "standalone_articulation_provider_proposal.json",
            preparation.proposal,
            label="standalone Articulation provider proposal",
        )
    identity = _build_identity(
        normalized,
        request_binding,
        preparation,
        preparation_binding,
    )
    identity_binding = _write_once_model(
        root / "standalone_articulation_identity.json",
        identity,
        label="standalone Articulation identity",
    )
    checkpoint = root / "checkpoint.json"
    if checkpoint.exists():
        state = _state(root)
        if (
            state.schema_version != ARTICULATION_STANDALONE_RUN_STATE_SCHEMA_VERSION
            or state.mode != mode
            or state.request != request_binding
            or state.standalone_identity != identity_binding
            or state.standalone_preparation != preparation_binding
            or state.standalone_proposal != proposal_binding
        ):
            raise StandaloneArticulationError(
                "Standalone preparation differs from the active checkpoint"
            )
    else:
        state = ArticulationRunState(
            schema_version=ARTICULATION_STANDALONE_RUN_STATE_SCHEMA_VERSION,
            mode=mode,
            request=request_binding,
            source_asset=normalized.source_asset,
            source_sha256=source_sha256,
            source_dependency_bundle_sha256=dependency_sha256,
            backend_configuration_sha256=preparation.configuration_sha256,
            scene_evidence_configuration_sha256=canonical_json_digest(
                preparation.scene
            ),
            standalone_identity=identity_binding,
            standalone_preparation=preparation_binding,
            standalone_proposal=proposal_binding,
        )
        state = _persist_standalone_state(root, state)
        state = _transition(
            state,
            "awaiting_decision",
            "Provider-neutral preparation is frozen before child reasoning.",
        )
        state = _persist_standalone_state(root, state)
    _validate_preparation_bindings(state, preparation)
    authoring = (
        _load_bound(state.authoring_result, ArticulationAuthoringResult)
        if state.authoring_result is not None
        else None
    )
    return write_articulation_workflow_summary(
        state,
        output_dir=root,
        authoring=authoring,
    )


def build_standalone_articulation_observation(
    output_dir: str | Path,
) -> StandaloneArticulationObservation:
    """Return the exact, non-mutating child handoff for the current attempt."""

    root = Path(output_dir).expanduser().resolve()
    state = _state(root)
    if (
        state.schema_version != ARTICULATION_STANDALONE_RUN_STATE_SCHEMA_VERSION
        or state.phase != "awaiting_decision"
    ):
        raise StandaloneArticulationError(
            "Standalone observation requires awaiting_decision"
        )
    identity = _load_bound(state.standalone_identity, StandaloneArticulationIdentity)
    preparation = _load_bound(
        state.standalone_preparation, StandaloneArticulationPreparation
    )
    _validate_preparation_bindings(state, preparation)
    issue_packet = None
    refinement_attempt = 0
    decision_patch_path = root / "articulation_decision_patch.json"
    if state.standalone_refinement_history:
        issue_binding = state.standalone_refinement_history[-1]
        packet = _load_bound(issue_binding, StandaloneArticulationIssuePacket)
        issue_packet = packet.model_dump(mode="json")
        refinement_attempt = packet.refinement_attempt
        decision_patch_path = Path(packet.replacement_patch_path)
    return StandaloneArticulationObservation(
        state_revision=state.revision,
        refinement_attempt=refinement_attempt,
        identity_digest=canonical_json_digest(identity),
        request_sha256=state.request.sha256,
        preparation_sha256=cast(ArtifactBinding, state.standalone_preparation).sha256,
        source_sha256=state.source_sha256,
        source_dependency_bundle_sha256=state.source_dependency_bundle_sha256,
        configuration_sha256=state.backend_configuration_sha256,
        evidence_records=tuple(
            item.model_dump(mode="json") for item in preparation.evidence_records
        ),
        optional_proposal=(
            preparation.proposal.model_dump(mode="json")
            if preparation.proposal is not None
            else None
        ),
        issue_packet=issue_packet,
        decision_patch_path=str(decision_patch_path.resolve()),
    )


def _validate_patch_identity(
    state: ArticulationRunState,
    identity: StandaloneArticulationIdentity,
    patch: StandaloneArticulationDecisionPatch,
) -> None:
    preparation_binding = state.standalone_preparation
    if preparation_binding is None:
        raise StandaloneArticulationError("Standalone preparation binding is missing")
    expected = (
        canonical_json_digest(identity),
        state.request.sha256,
        preparation_binding.sha256,
        state.source_sha256,
        state.source_dependency_bundle_sha256,
        state.backend_configuration_sha256,
    )
    observed = (
        patch.identity_digest,
        patch.request_sha256,
        patch.preparation_sha256,
        patch.source_sha256,
        patch.source_dependency_bundle_sha256,
        patch.configuration_sha256,
    )
    if observed != expected or patch.expected_state_revision != state.revision:
        raise StandaloneArticulationError(
            "Standalone decision patch is stale or belongs to another run"
        )
    if patch.refinement_attempt == 0:
        if state.standalone_refinement_history:
            raise StandaloneArticulationError(
                "Initial decision patch cannot replace an active refinement"
            )
    else:
        if not state.standalone_refinement_history:
            raise StandaloneArticulationError(
                "Refinement decision patch lacks an issue packet"
            )
        issue_binding = state.standalone_refinement_history[-1]
        issue = _load_bound(issue_binding, StandaloneArticulationIssuePacket)
        parent_ledger = _load_bound(
            issue.parent_ledger,
            StandaloneArticulationDecisionLedger,
        )
        if (
            patch.parent_patch_sha256
            != cast(ArtifactBinding, state.standalone_decision_patch).sha256
            or patch.issue_packet_sha256 != issue_binding.sha256
            or issue.parent_patch.sha256 != patch.parent_patch_sha256
            or issue.parent_ledger.sha256
            != cast(ArtifactBinding, state.standalone_decision_ledger).sha256
        ):
            raise StandaloneArticulationError(
                "Refinement decision patch does not bind its exact parent attempt"
            )
        candidate_ids = tuple(item.candidate_id for item in patch.candidate_decisions)
        issue_candidate_ids = tuple(issue.candidate_issue_codes)
        if (
            candidate_ids != parent_ledger.candidate_ids
            or not set(issue_candidate_ids).issubset(candidate_ids)
            or (
                parent_ledger.disposition == "revise"
                and issue_candidate_ids != parent_ledger.unresolved_candidate_ids
            )
        ):
            raise StandaloneArticulationError(
                "Refinement decision patch changes the bounded candidate scope"
            )
        required_evidence_ids = set(issue.required_evidence_ids)
        cited_evidence_ids = {
            evidence_id
            for item in patch.candidate_decisions
            for evidence_id in item.evidence_ids
        }
        if not required_evidence_ids.issubset(patch.evidence_requirements) or not (
            required_evidence_ids <= cited_evidence_ids
        ):
            raise StandaloneArticulationError(
                "Refinement decision patch omits required issue-packet evidence"
            )


def _patch_path(root: Path, patch: StandaloneArticulationDecisionPatch) -> Path:
    if patch.refinement_attempt == 0:
        return root / "articulation_decision_patch.json"
    return (
        root
        / "refinement"
        / f"attempt-{patch.refinement_attempt:03d}"
        / "articulation_decision_patch.json"
    )


def _ledger_path(root: Path, patch: StandaloneArticulationDecisionPatch) -> Path:
    if patch.refinement_attempt == 0:
        return root / "articulation_decision_ledger.json"
    return (
        root
        / "refinement"
        / f"attempt-{patch.refinement_attempt:03d}"
        / "articulation_decision_ledger.json"
    )


def _canonical_graph_path(
    root: Path,
    patch: StandaloneArticulationDecisionPatch,
) -> Path:
    if patch.refinement_attempt == 0:
        return root / "canonical_articulation_graph.json"
    return (
        root
        / "refinement"
        / f"attempt-{patch.refinement_attempt:03d}"
        / "canonical_articulation_graph.json"
    )


def _execution_root(
    root: Path,
    patch: StandaloneArticulationDecisionPatch,
) -> Path:
    if patch.refinement_attempt == 0:
        return root
    return root / "refinement" / f"attempt-{patch.refinement_attempt:03d}" / "execution"


def _issue_packet(
    root: Path,
    patch: StandaloneArticulationDecisionPatch,
    patch_binding: ArtifactBinding,
    ledger_binding: ArtifactBinding,
) -> ArtifactBinding:
    issue_codes = {
        item.candidate_id: cast(tuple[str, ...], item.issue_codes)
        for item in patch.candidate_decisions
        if item.disposition == "refine"
    }
    packet_path = root / "refinement" / "attempt-001" / "issue_packet.json"
    packet = StandaloneArticulationIssuePacket(
        identity_digest=patch.identity_digest,
        parent_patch=patch_binding,
        parent_ledger=ledger_binding,
        candidate_issue_codes=issue_codes,
        required_evidence_ids=tuple(
            dict.fromkeys(
                evidence_id
                for item in patch.candidate_decisions
                if item.disposition == "refine"
                for evidence_id in item.evidence_ids
            )
        ),
        allowed_operations=(
            "hierarchy_inspection",
            "membership_inspection",
            "focused_evidence_collection",
            "saved_stage_readback",
        ),
        replacement_patch_path=str(
            (
                root / "refinement" / "attempt-001" / "articulation_decision_patch.json"
            ).resolve()
        ),
    )
    return _write_once_model(
        packet_path,
        packet,
        label="standalone Articulation refinement issue packet",
    )


def _record_execution_refinement(
    root: Path,
    state: ArticulationRunState,
    patch: StandaloneArticulationDecisionPatch,
    *,
    failed_artifacts: tuple[ArtifactBinding, ...],
    issue_code: Literal[
        "frame_unresolved", "membership_unresolved", "readback_mismatch"
    ],
    detail: str,
) -> ArticulationRunState:
    graph = cast(EmbeddedArticulationCanonicalGraph, patch.canonical_graph)
    if patch.refinement_attempt >= patch.review_policy.max_refinement_attempts:
        error = (
            "Standalone Articulation refinement cap exhausted after "
            f"{issue_code}: {detail}"
        )
        state = state.model_copy(
            update={
                "accepted_candidate_ids": (),
                "unresolved_candidate_ids": graph.candidate_ids,
                "review_required_candidate_ids": (),
                "error": error,
            }
        )
        identity = _load_bound(
            state.standalone_identity, StandaloneArticulationIdentity
        )
        ledger = _load_bound(
            state.standalone_decision_ledger,
            StandaloneArticulationDecisionLedger,
        )
        state = _transition(state, "conditional", error)
        state = _publish_non_success_terminal_receipt(
            root,
            state,
            identity,
            ledger,
            terminal_disposition="refinement_cap",
            error=error,
            issue_codes=(issue_code,),
            failed_artifacts=failed_artifacts,
        )
        return _persist_standalone_state(
            root,
            state,
        )
    patch_binding = cast(ArtifactBinding, state.standalone_decision_patch)
    ledger_binding = cast(ArtifactBinding, state.standalone_decision_ledger)
    packet_path = root / "refinement" / "attempt-001" / "issue_packet.json"
    packet = StandaloneArticulationIssuePacket(
        identity_digest=patch.identity_digest,
        parent_patch=patch_binding,
        parent_ledger=ledger_binding,
        failed_artifacts=failed_artifacts,
        candidate_issue_codes={
            candidate_id: (issue_code,) for candidate_id in graph.candidate_ids
        },
        required_evidence_ids=(
            "joint-authoritative-owner-inspection"
            if issue_code == "membership_unresolved"
            else "joint-scene-inspection",
        ),
        allowed_operations=(
            "membership_inspection",
            "focused_evidence_collection",
            "saved_stage_readback",
        ),
        replacement_patch_path=str(
            (
                root / "refinement" / "attempt-001" / "articulation_decision_patch.json"
            ).resolve()
        ),
    )
    issue_binding = _write_once_model(
        packet_path,
        packet,
        label="standalone Articulation execution issue packet",
    )
    state = state.model_copy(
        update={
            "accepted_candidate_ids": (),
            "unresolved_candidate_ids": graph.candidate_ids,
            "review_required_candidate_ids": graph.candidate_ids,
            "standalone_refinement_history": (
                *state.standalone_refinement_history,
                issue_binding,
            ),
            "error": detail,
        }
    )
    state = _transition(
        state,
        "awaiting_decision",
        f"Candidate-bound {issue_code} requires one focused replacement proposal.",
    )
    return _persist_standalone_state(root, state)


def _publish_non_success_terminal_receipt(
    root: Path,
    state: ArticulationRunState,
    identity: StandaloneArticulationIdentity,
    ledger: StandaloneArticulationDecisionLedger,
    *,
    terminal_disposition: Literal["reject", "refinement_cap"],
    error: str,
    issue_codes: tuple[str, ...],
    failed_artifacts: tuple[ArtifactBinding, ...] = (),
) -> ArticulationRunState:
    """Seal one fail-closed terminal result without claiming output custody."""

    graph_binding = state.standalone_canonical_graph
    graph_digest = (
        canonical_json_digest(
            _load_bound(graph_binding, EmbeddedArticulationCanonicalGraph)
        )
        if graph_binding is not None
        else None
    )
    terminal = StandaloneArticulationTerminalReceipt(
        identity=cast(ArtifactBinding, state.standalone_identity),
        identity_digest=canonical_json_digest(identity),
        request=state.request,
        source=identity.source,
        source_dependency_bundle_sha256=state.source_dependency_bundle_sha256,
        preparation=cast(ArtifactBinding, state.standalone_preparation),
        optional_proposal=state.standalone_proposal,
        decision_patch=cast(ArtifactBinding, state.standalone_decision_patch),
        decision_ledger=cast(ArtifactBinding, state.standalone_decision_ledger),
        canonical_graph=graph_binding,
        canonical_graph_digest=graph_digest,
        refinement_history=state.standalone_refinement_history,
        failed_artifacts=failed_artifacts,
        issue_codes=tuple(dict.fromkeys(issue_codes)),
        terminal_disposition=terminal_disposition,
        status="non_success",
        error=error,
    )
    terminal_binding = _write_once_model(
        root / "standalone_articulation_terminal_receipt.json",
        terminal,
        label="standalone Articulation terminal receipt",
    )
    if ledger.decision_patch != terminal.decision_patch:
        raise StandaloneArticulationError(
            "Non-success terminal receipt does not bind the terminal decision ledger"
        )
    return state.model_copy(update={"standalone_terminal_receipt": terminal_binding})


def _freeze_decision(
    root: Path,
    state: ArticulationRunState,
    identity: StandaloneArticulationIdentity,
    preparation: StandaloneArticulationPreparation,
    patch: StandaloneArticulationDecisionPatch,
) -> tuple[ArticulationRunState, StandaloneArticulationDecisionLedger]:
    _validate_patch_identity(state, identity, patch)
    evidence_ids = {item.evidence_id for item in preparation.evidence_records}
    cited = {
        evidence_id
        for item in patch.candidate_decisions
        for evidence_id in item.evidence_ids
    }
    if (
        not set(patch.evidence_requirements) <= evidence_ids
        or not cited <= evidence_ids
    ):
        raise StandaloneArticulationError(
            "Standalone decision patch cites unavailable evidence"
        )
    request = _load_bound(state.request, ArticulationWorkflowRequest)
    candidate_count = len(patch.candidate_decisions)
    if candidate_count > request.max_candidate_count:
        raise StandaloneArticulationError(
            "Standalone decision patch exceeds the request max_candidate_count"
        )
    if (
        request.expected_candidate_count is not None
        and candidate_count != request.expected_candidate_count
    ):
        raise StandaloneArticulationError(
            "Standalone decision patch differs from the request "
            "expected_candidate_count"
        )
    if patch.disposition == "accept":
        graph = cast(EmbeddedArticulationCanonicalGraph, patch.canonical_graph)
        unsupported_motion_types = tuple(
            dict.fromkeys(
                joint.joint_type
                for joint in graph.joints
                if joint.joint_type not in request.allowed_motion_types
            )
        )
        if unsupported_motion_types:
            raise StandaloneArticulationError(
                "Standalone canonical graph contains motion types outside the "
                f"request scope: {unsupported_motion_types}"
            )
    patch_binding = _write_once_model(
        _patch_path(root, patch),
        patch,
        label="standalone Articulation decision patch",
    )
    graph_binding = None
    graph_digest = None
    if patch.disposition == "accept":
        graph = cast(EmbeddedArticulationCanonicalGraph, patch.canonical_graph)
        _validate_canonical_graph(
            graph,
            state=state,
            evidence=cast(Any, _PreparationEvidenceView(preparation.evidence_records)),
            capabilities=_capabilities(preparation),
            # Standalone preparation records existing static source membership
            # for exact custody; it does not ask the v1 authorer to create a
            # new fixed constraint for those preserved rows.
            allow_preserved_explicit_fixed_membership=True,
        )
        graph_binding = _write_once_model(
            _canonical_graph_path(root, patch),
            graph,
            label="canonical Articulation graph",
        )
        graph_digest = canonical_json_digest(graph)
    decisions = patch.candidate_decisions
    candidate_ids = tuple(item.candidate_id for item in decisions)
    accepted = tuple(
        item.candidate_id for item in decisions if item.disposition == "accept"
    )
    rejected = tuple(
        item.candidate_id for item in decisions if item.disposition == "reject"
    )
    unresolved = tuple(
        item.candidate_id for item in decisions if item.disposition == "refine"
    )
    effective_disposition: Literal["accept", "reject", "revise", "cap_exhausted"]
    if patch.disposition == "revise" and patch.refinement_attempt >= 1:
        effective_disposition = "cap_exhausted"
    else:
        effective_disposition = patch.disposition
    preparation_binding = cast(ArtifactBinding, state.standalone_preparation)
    ledger = StandaloneArticulationDecisionLedger(
        identity_digest=patch.identity_digest,
        preparation=preparation_binding,
        decision_patch=patch_binding,
        refinement_attempt=patch.refinement_attempt,
        disposition=effective_disposition,
        candidate_ids=candidate_ids,
        accepted_candidate_ids=accepted if patch.disposition == "accept" else (),
        rejected_candidate_ids=rejected,
        unresolved_candidate_ids=unresolved,
        canonical_graph=graph_binding,
        canonical_graph_digest=graph_digest,
        evidence_requirements=patch.evidence_requirements,
        validation_status=(
            "accepted" if patch.disposition == "accept" else "non_success"
        ),
        validated_at=_now(),
    )
    ledger_binding = _write_once_model(
        _ledger_path(root, patch),
        ledger,
        label="standalone Articulation decision ledger",
    )
    updates: dict[str, Any] = {
        "standalone_decision_patch": patch_binding,
        "standalone_decision_ledger": ledger_binding,
        "standalone_canonical_graph": graph_binding,
        "candidate_ids": candidate_ids,
        "accepted_candidate_ids": (accepted if patch.disposition == "accept" else ()),
        "rejected_candidate_ids": rejected,
        "unresolved_candidate_ids": unresolved,
        "review_required_candidate_ids": (
            accepted if patch.disposition == "accept" else unresolved
        ),
        "error": None,
    }
    state = state.model_copy(update=updates)
    if patch.disposition == "accept":
        return _persist_standalone_state(root, state), ledger
    if patch.disposition == "revise" and patch.refinement_attempt == 0:
        issue_binding = _issue_packet(root, patch, patch_binding, ledger_binding)
        state = _transition(
            state,
            "awaiting_decision",
            "Candidate-bound issues require one focused replacement proposal.",
            standalone_refinement_history=(
                *state.standalone_refinement_history,
                issue_binding,
            ),
        )
    else:
        error = (
            "Standalone Articulation refinement cap exhausted; inspect the bound "
            "candidate issue packet and revise the source or preparation."
            if effective_disposition == "cap_exhausted"
            else "The child decision patch rejected all authoring authority."
        )
        state = _transition(
            state,
            "conditional",
            error,
            error=error,
            review_required_candidate_ids=(),
        )
        terminal_disposition: Literal["reject", "refinement_cap"] = (
            "refinement_cap" if effective_disposition == "cap_exhausted" else "reject"
        )
        issue_codes = tuple(
            dict.fromkeys(
                code
                for decision in patch.candidate_decisions
                for code in decision.issue_codes
            )
        )
        state = _publish_non_success_terminal_receipt(
            root,
            state,
            identity,
            ledger,
            terminal_disposition=terminal_disposition,
            error=error,
            issue_codes=issue_codes,
        )
    return _persist_standalone_state(root, state), ledger


def apply_standalone_articulation_decision_patch(
    output_dir: str | Path,
    patch: StandaloneArticulationDecisionPatch,
    *,
    client: ArticulationAuthoringClient,
    cancel_checker: CancelChecker | None = None,
) -> ArticulationRunState:
    """Validate/freeze child semantics, then author only an accepted graph."""

    root = Path(output_dir).expanduser().resolve()
    state = _state(root)
    if (
        state.schema_version != ARTICULATION_STANDALONE_RUN_STATE_SCHEMA_VERSION
        or state.phase != "awaiting_decision"
    ):
        raise StandaloneArticulationError(
            "Standalone decision application requires awaiting_decision"
        )
    identity = _load_bound(state.standalone_identity, StandaloneArticulationIdentity)
    preparation = _load_bound(
        state.standalone_preparation, StandaloneArticulationPreparation
    )
    _validate_preparation_bindings(state, preparation)
    ledger: StandaloneArticulationDecisionLedger
    if state.standalone_decision_patch is not None:
        existing_patch = _load_bound(
            state.standalone_decision_patch,
            StandaloneArticulationDecisionPatch,
        )
        if existing_patch == patch:
            ledger = _load_bound(
                state.standalone_decision_ledger,
                StandaloneArticulationDecisionLedger,
            )
            if ledger.disposition != "accept":
                return state
        else:
            if patch.refinement_attempt == 0:
                raise StandaloneArticulationError(
                    "A different standalone decision patch is already frozen"
                )
            state, ledger = _freeze_decision(root, state, identity, preparation, patch)
    else:
        state, ledger = _freeze_decision(root, state, identity, preparation, patch)
    if patch.disposition != "accept":
        return state
    graph = cast(EmbeddedArticulationCanonicalGraph, patch.canonical_graph)
    graph_binding = cast(ArtifactBinding, state.standalone_canonical_graph)
    ledger_binding = cast(ArtifactBinding, state.standalone_decision_ledger)
    patch_binding = cast(ArtifactBinding, state.standalone_decision_patch)
    execution_root = _execution_root(root, patch)
    try:
        accepted_authoring_plan = _accepted_rigid_link_authoring_plan(
            execution_root,
            state,
            graph,
            graph_binding,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        return _record_execution_refinement(
            root,
            state,
            patch,
            failed_artifacts=(patch_binding, ledger_binding, graph_binding),
            issue_code="membership_unresolved",
            detail=f"Accepted rigid-link plan projection failed: {exc}",
        )
    approved = _graph_candidate_document(graph)
    approved_binding = _write_once_model(
        execution_root / "approved_articulation_candidates.json",
        approved,
        label="approved Articulation candidates",
    )
    request = ArticulationAuthoringRequest(
        source_asset=state.source_asset,
        source_sha256=state.source_sha256,
        source_dependency_bundle_sha256=state.source_dependency_bundle_sha256,
        candidate_document_path=approved_binding.path,
        candidate_document_sha256=approved_binding.sha256,
        accepted_candidate_ids=graph.candidate_ids,
        idempotency_key=_standalone_authoring_key(
            state,
            canonical_json_digest(graph),
            execution_root,
            (
                accepted_authoring_plan.sha256
                if accepted_authoring_plan is not None
                else None
            ),
        ),
        predictions_path=None,
        predictions_sha256=None,
        accepted_authoring_plan_path=(
            accepted_authoring_plan.path
            if accepted_authoring_plan is not None
            else None
        ),
        accepted_authoring_plan_sha256=(
            accepted_authoring_plan.sha256
            if accepted_authoring_plan is not None
            else None
        ),
        output_dir=execution_root,
    )
    request_binding = _write_once_model(
        execution_root / "authoring_request.json",
        request,
        label="standalone Articulation authoring request",
    )
    try:
        authored = client.author(request, cancel_checker=cancel_checker)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        state = state.model_copy(
            update={
                "approved_candidate_document": approved_binding,
                "authoring_request": request_binding,
            }
        )
        return _record_execution_refinement(
            root,
            state,
            patch,
            failed_artifacts=(
                patch_binding,
                ledger_binding,
                graph_binding,
                approved_binding,
                request_binding,
            ),
            issue_code="frame_unresolved",
            detail=f"Accepted graph authoring failed: {exc}",
        )
    authoring_binding = _write_once_model(
        execution_root / "authoring_result.json",
        authored,
        label="standalone Articulation authoring result",
    )
    try:
        validation = client.validate(
            authored,
            expected_candidate_ids=graph.candidate_ids,
            cancel_checker=cancel_checker,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        state = state.model_copy(
            update={
                "approved_candidate_document": approved_binding,
                "authoring_request": request_binding,
                "authoring_result": authoring_binding,
            }
        )
        return _record_execution_refinement(
            root,
            state,
            patch,
            failed_artifacts=(
                patch_binding,
                ledger_binding,
                graph_binding,
                approved_binding,
                request_binding,
                authoring_binding,
            ),
            issue_code="readback_mismatch",
            detail=f"Saved-stage validation failed: {exc}",
        )
    validation_binding = _write_once_model(
        execution_root / "validation_evidence.json",
        validation,
        label="standalone Articulation validation evidence",
    )
    if (
        validation.status != "pass"
        or validation.expected_candidate_ids != graph.candidate_ids
        or validation.validated_candidate_ids != graph.candidate_ids
        or authored.authored_candidate_ids != graph.candidate_ids
    ):
        state = state.model_copy(
            update={
                "approved_candidate_document": approved_binding,
                "authoring_request": request_binding,
                "authoring_result": authoring_binding,
                "validation_result": validation_binding,
            }
        )
        return _record_execution_refinement(
            root,
            state,
            patch,
            failed_artifacts=(
                patch_binding,
                ledger_binding,
                graph_binding,
                approved_binding,
                request_binding,
                authoring_binding,
                validation_binding,
            ),
            issue_code="readback_mismatch",
            detail="Saved-stage validation differs from the accepted graph.",
        )
    output_asset = _execution_binding(authored.output_asset_path)
    if output_asset.sha256 != authored.output_asset_sha256:
        raise StandaloneArticulationError(
            "Authored output bytes differ from the authoring receipt"
        )
    try:
        exact_membership = _validate_saved_membership(
            graph, Path(authored.output_asset_path)
        )
    except EmbeddedArticulationError as exc:
        state = state.model_copy(
            update={
                "approved_candidate_document": approved_binding,
                "authoring_request": request_binding,
                "authoring_result": authoring_binding,
                "validation_result": validation_binding,
            }
        )
        return _record_execution_refinement(
            root,
            state,
            patch,
            failed_artifacts=(
                patch_binding,
                ledger_binding,
                graph_binding,
                approved_binding,
                request_binding,
                authoring_binding,
                validation_binding,
            ),
            issue_code="membership_unresolved",
            detail=str(exc),
        )
    readback = StandaloneArticulationReadback(
        identity_digest=canonical_json_digest(identity),
        ledger_sha256=ledger_binding.sha256,
        canonical_graph_digest=cast(str, ledger.canonical_graph_digest),
        output_asset=output_asset,
        validation=validation_binding,
        joint_ids=graph.candidate_ids,
        topology=_readback_topology(graph),
        groups=_readback_groups(graph),
        memberships=_readback_memberships(graph),
        rigid_link_operations=tuple(
            item.model_dump(mode="json") for item in graph.rigid_link_operations
        ),
        exact_membership_match=exact_membership,
        exact_co_rigid_disposition_match=exact_membership,
    )
    readback_binding = _write_once_model(
        execution_root / "standalone_articulation_readback.json",
        readback,
        label="standalone Articulation readback",
    )
    membership_operation_receipt = (
        _binding(Path(authored.membership_operation_receipt_path))
        if authored.membership_operation_receipt_path is not None
        else None
    )
    if (membership_operation_receipt is None) != (not graph.rigid_link_operations) or (
        membership_operation_receipt is not None
        and membership_operation_receipt.sha256
        != authored.membership_operation_receipt_sha256
    ):
        raise StandaloneArticulationError(
            "Rigid-link graph and membership operation receipt differ"
        )
    receipt = StandaloneArticulationAuthoringReceipt(
        identity_digest=canonical_json_digest(identity),
        decision_patch=patch_binding,
        decision_ledger=ledger_binding,
        canonical_graph=graph_binding,
        approved_candidates=approved_binding,
        authoring_request=request_binding,
        authoring_result=authoring_binding,
        validation=validation_binding,
        readback=readback_binding,
        accepted_authoring_plan=accepted_authoring_plan,
        membership_operation_receipt=membership_operation_receipt,
        output_asset=output_asset,
        accepted_candidate_ids=graph.candidate_ids,
        rigid_link_operation_ids=tuple(
            item.operation_id for item in graph.rigid_link_operations
        ),
    )
    receipt_binding = _write_once_model(
        execution_root / "standalone_articulation_authoring_receipt.json",
        receipt,
        label="standalone Articulation authoring receipt",
    )
    state = state.model_copy(
        update={
            "approved_candidate_document": approved_binding,
            "authoring_request": request_binding,
            "authoring_result": authoring_binding,
            "validation_result": validation_binding,
            "standalone_authoring_receipt": receipt_binding,
            "standalone_readback": readback_binding,
            "review_required_candidate_ids": graph.candidate_ids,
            "error": None,
        }
    )
    state = _transition(
        state,
        "awaiting_post_review",
        "Exact saved-stage readback requires independent output review.",
    )
    return _persist_standalone_state(root, state)


def bind_standalone_articulation_output_evidence(
    output_dir: str | Path,
    canonical_visual_envelope_path: str | Path,
) -> ArticulationRunState:
    """Bind exact current-run canonical OVRTX evidence before review."""

    root = Path(output_dir).expanduser().resolve()
    state = _state(root)
    if (
        state.schema_version != ARTICULATION_STANDALONE_RUN_STATE_SCHEMA_VERSION
        or state.phase != "awaiting_post_review"
    ):
        raise StandaloneArticulationError(
            "Standalone output evidence binding requires awaiting_post_review"
        )
    authoring = _load_bound(state.authoring_result, ArticulationAuthoringResult)
    readback = _load_bound(state.standalone_readback, StandaloneArticulationReadback)
    output = _execution_binding(authoring.output_asset_path)
    if output != readback.output_asset:
        raise StandaloneArticulationError(
            "Canonical output evidence requires exact saved-stage bytes"
        )
    evidence = build_embedded_articulation_output_evidence(
        canonical_visual_envelope_path,
        expected_source=state_to_source_binding(state),
        expected_output=output,
    )
    binding = _write_once_model(
        root / "standalone_articulation_output_evidence.json",
        evidence,
        label="standalone Articulation output evidence",
    )
    if state.standalone_output_evidence is not None:
        if state.standalone_output_evidence != binding:
            raise StandaloneArticulationError(
                "Standalone output evidence conflicts with the checkpoint"
            )
        return state
    return _persist_standalone_state(
        root,
        state.model_copy(update={"standalone_output_evidence": binding}),
    )


def apply_standalone_articulation_post_review(
    output_dir: str | Path,
    patch: StandaloneArticulationPostReviewPatch,
) -> ArticulationRunState:
    """Publish only after independent review of exact graph/readback/render bytes."""

    root = Path(output_dir).expanduser().resolve()
    state = _state(root)
    if (
        state.schema_version != ARTICULATION_STANDALONE_RUN_STATE_SCHEMA_VERSION
        or state.phase != "awaiting_post_review"
    ):
        raise StandaloneArticulationError(
            "Standalone post review requires awaiting_post_review"
        )
    identity = _load_bound(state.standalone_identity, StandaloneArticulationIdentity)
    graph = _load_bound(
        state.standalone_canonical_graph, EmbeddedArticulationCanonicalGraph
    )
    receipt = _load_bound(
        state.standalone_authoring_receipt,
        StandaloneArticulationAuthoringReceipt,
    )
    readback = _load_bound(state.standalone_readback, StandaloneArticulationReadback)
    output_evidence = _load_bound(
        state.standalone_output_evidence,
        EmbeddedArticulationOutputEvidence,
    )
    validate_embedded_articulation_output_evidence(output_evidence)
    graph_binding = cast(ArtifactBinding, state.standalone_canonical_graph)
    receipt_binding = cast(ArtifactBinding, state.standalone_authoring_receipt)
    readback_binding = cast(ArtifactBinding, state.standalone_readback)
    evidence_binding = cast(ArtifactBinding, state.standalone_output_evidence)
    expected_images = tuple(item.sha256 for item in output_evidence.images)
    if (
        patch.identity_digest != canonical_json_digest(identity)
        or patch.authoring_receipt_sha256 != receipt_binding.sha256
        or patch.canonical_graph_sha256 != graph_binding.sha256
        or patch.readback_sha256 != readback_binding.sha256
        or patch.output_evidence_sha256 != evidence_binding.sha256
        or patch.inspected_render_image_sha256s != expected_images
        or readback.canonical_graph_digest != canonical_json_digest(graph)
        or receipt.output_asset != output_evidence.post_mutation_output
    ):
        raise StandaloneArticulationError(
            "Standalone post review is stale or does not cover exact output evidence"
        )
    review_binding = _write_once_model(
        root / "standalone_articulation_post_review.json",
        patch,
        label="standalone Articulation post review",
    )
    state = state.model_copy(update={"standalone_post_review": review_binding})
    if patch.disposition != "accept":
        error = f"Independent post review {patch.disposition}: " + "; ".join(
            patch.findings
        )
        state = _transition(
            state,
            "conditional",
            f"Independent post review {patch.disposition} blocks publication.",
            error=error,
            accepted_candidate_ids=(),
            unresolved_candidate_ids=(
                graph.candidate_ids if patch.disposition == "revise" else ()
            ),
            rejected_candidate_ids=(
                graph.candidate_ids if patch.disposition == "reject" else ()
            ),
            review_required_candidate_ids=(),
        )
        if patch.disposition == "reject":
            ledger = _load_bound(
                state.standalone_decision_ledger,
                StandaloneArticulationDecisionLedger,
            )
            state = _publish_non_success_terminal_receipt(
                root,
                state,
                identity,
                ledger,
                terminal_disposition="reject",
                error=error,
                issue_codes=(),
                failed_artifacts=(
                    receipt_binding,
                    readback_binding,
                    evidence_binding,
                    review_binding,
                ),
            )
        return _persist_standalone_state(root, state)
    retained = (
        tuple(
            cast(
                ArtifactBinding,
                item,
            )
            for item in (
                state.standalone_identity,
                state.standalone_preparation,
                state.standalone_decision_patch,
                state.standalone_decision_ledger,
                state.standalone_canonical_graph,
                state.standalone_authoring_receipt,
                state.standalone_readback,
                state.standalone_output_evidence,
                review_binding,
            )
        )
        + state.standalone_refinement_history
    )
    cleanup = StandaloneArticulationCleanupReceipt(
        identity_digest=canonical_json_digest(identity),
        output_asset=receipt.output_asset,
        retained_artifacts=retained,
    )
    cleanup_binding = _write_once_model(
        root / "standalone_articulation_cleanup_receipt.json",
        cleanup,
        label="standalone Articulation cleanup receipt",
    )
    terminal = StandaloneArticulationTerminalReceipt(
        identity=cast(ArtifactBinding, state.standalone_identity),
        identity_digest=canonical_json_digest(identity),
        request=state.request,
        source=identity.source,
        source_dependency_bundle_sha256=state.source_dependency_bundle_sha256,
        preparation=cast(ArtifactBinding, state.standalone_preparation),
        optional_proposal=state.standalone_proposal,
        decision_patch=cast(ArtifactBinding, state.standalone_decision_patch),
        decision_ledger=cast(ArtifactBinding, state.standalone_decision_ledger),
        canonical_graph=graph_binding,
        canonical_graph_digest=canonical_json_digest(graph),
        authoring_receipt=receipt_binding,
        output_asset=receipt.output_asset,
        readback=readback_binding,
        output_evidence=evidence_binding,
        post_review=review_binding,
        cleanup=cleanup_binding,
        refinement_history=state.standalone_refinement_history,
    )
    terminal_binding = _write_once_model(
        root / "standalone_articulation_terminal_receipt.json",
        terminal,
        label="standalone Articulation terminal receipt",
    )
    state = state.model_copy(
        update={
            "standalone_cleanup": cleanup_binding,
            "standalone_terminal_receipt": terminal_binding,
            "review_required_candidate_ids": (),
            "unresolved_candidate_ids": (),
            "error": None,
        }
    )
    state = _transition(
        state,
        "completed",
        "Independent review accepted exact standalone Articulation evidence.",
    )
    return _persist_standalone_state(root, state)


def validate_completed_standalone_articulation_checkpoint(
    state: ArticulationRunState,
    *,
    authoring: ArticulationAuthoringResult | None,
) -> None:
    """Revalidate the complete terminal chain for summary publication."""

    if (
        state.schema_version != ARTICULATION_STANDALONE_RUN_STATE_SCHEMA_VERSION
        or state.phase != "completed"
        or authoring is None
    ):
        raise StandaloneArticulationError(
            "Standalone completion requires v5 completed state and authoring result"
        )
    identity = _load_bound(state.standalone_identity, StandaloneArticulationIdentity)
    preparation = _load_bound(
        state.standalone_preparation, StandaloneArticulationPreparation
    )
    patch = _load_bound(
        state.standalone_decision_patch, StandaloneArticulationDecisionPatch
    )
    ledger = _load_bound(
        state.standalone_decision_ledger, StandaloneArticulationDecisionLedger
    )
    graph = _load_bound(
        state.standalone_canonical_graph, EmbeddedArticulationCanonicalGraph
    )
    receipt = _load_bound(
        state.standalone_authoring_receipt,
        StandaloneArticulationAuthoringReceipt,
    )
    readback = _load_bound(state.standalone_readback, StandaloneArticulationReadback)
    output_evidence = _load_bound(
        state.standalone_output_evidence, EmbeddedArticulationOutputEvidence
    )
    review = _load_bound(
        state.standalone_post_review, StandaloneArticulationPostReviewPatch
    )
    cleanup = _load_bound(
        state.standalone_cleanup, StandaloneArticulationCleanupReceipt
    )
    terminal = _load_bound(
        state.standalone_terminal_receipt,
        StandaloneArticulationTerminalReceipt,
    )
    _validate_preparation_bindings(state, preparation)
    validate_embedded_articulation_output_evidence(output_evidence)
    expected_terminal = StandaloneArticulationTerminalReceipt(
        identity=cast(ArtifactBinding, state.standalone_identity),
        identity_digest=canonical_json_digest(identity),
        request=state.request,
        source=identity.source,
        source_dependency_bundle_sha256=state.source_dependency_bundle_sha256,
        preparation=cast(ArtifactBinding, state.standalone_preparation),
        optional_proposal=state.standalone_proposal,
        decision_patch=cast(ArtifactBinding, state.standalone_decision_patch),
        decision_ledger=cast(ArtifactBinding, state.standalone_decision_ledger),
        canonical_graph=cast(ArtifactBinding, state.standalone_canonical_graph),
        canonical_graph_digest=canonical_json_digest(graph),
        authoring_receipt=cast(ArtifactBinding, state.standalone_authoring_receipt),
        output_asset=receipt.output_asset,
        readback=cast(ArtifactBinding, state.standalone_readback),
        output_evidence=cast(ArtifactBinding, state.standalone_output_evidence),
        post_review=cast(ArtifactBinding, state.standalone_post_review),
        cleanup=cast(ArtifactBinding, state.standalone_cleanup),
        refinement_history=state.standalone_refinement_history,
    )
    if (
        terminal != expected_terminal
        or patch.disposition != "accept"
        or ledger.disposition != "accept"
        or ledger.canonical_graph_digest != canonical_json_digest(graph)
        or receipt.accepted_candidate_ids != graph.candidate_ids
        or readback.joint_ids != graph.candidate_ids
        or review.disposition != "accept"
        or cleanup.output_asset != receipt.output_asset
        or authoring.output_asset_sha256 != receipt.output_asset.sha256
    ):
        raise StandaloneArticulationError(
            "Completed standalone Articulation custody chain is inconsistent"
        )


def run_standalone_articulation_workflow(
    request: ArticulationWorkflowRequest,
    *,
    mode: ArticulationWorkflowMode,
    preparation: StandaloneArticulationPreparation,
) -> ArticulationFinalizationResult:
    """Prepare or revalidate one standalone run without inventing child semantics."""

    result = prepare_standalone_articulation_workflow(
        request,
        mode=mode,
        preparation=preparation,
    )
    state = _state(request.output_dir.expanduser().resolve())
    authoring = (
        _load_bound(state.authoring_result, ArticulationAuthoringResult)
        if state.authoring_result is not None
        else None
    )
    return (
        write_articulation_workflow_summary(
            state,
            output_dir=request.output_dir.expanduser().resolve(),
            authoring=authoring,
        )
        if result.status != "awaiting_decision"
        else result
    )


__all__ = [
    "STANDALONE_ARTICULATION_AUTHORING_RECEIPT_SCHEMA_VERSION",
    "STANDALONE_ARTICULATION_CLEANUP_SCHEMA_VERSION",
    "STANDALONE_ARTICULATION_IDENTITY_SCHEMA_VERSION",
    "STANDALONE_ARTICULATION_ISSUE_PACKET_SCHEMA_VERSION",
    "STANDALONE_ARTICULATION_LEDGER_SCHEMA_VERSION",
    "STANDALONE_ARTICULATION_PATCH_SCHEMA_VERSION",
    "STANDALONE_ARTICULATION_POST_REVIEW_SCHEMA_VERSION",
    "STANDALONE_ARTICULATION_PREPARATION_SCHEMA_VERSION",
    "STANDALONE_ARTICULATION_READBACK_SCHEMA_VERSION",
    "STANDALONE_ARTICULATION_TERMINAL_RECEIPT_SCHEMA_VERSION",
    "StandaloneArticulationAuthoringReceipt",
    "StandaloneArticulationCandidateDecision",
    "StandaloneArticulationCleanupReceipt",
    "StandaloneArticulationDecisionLedger",
    "StandaloneArticulationDecisionPatch",
    "StandaloneArticulationError",
    "StandaloneArticulationIdentity",
    "StandaloneArticulationIssuePacket",
    "StandaloneArticulationObservation",
    "StandaloneArticulationPostReviewPatch",
    "StandaloneArticulationPreparation",
    "StandaloneArticulationReadback",
    "StandaloneArticulationReviewPolicy",
    "StandaloneArticulationTerminalReceipt",
    "apply_standalone_articulation_decision_patch",
    "apply_standalone_articulation_post_review",
    "bind_standalone_articulation_output_evidence",
    "build_standalone_articulation_observation",
    "prepare_standalone_articulation_workflow",
    "run_standalone_articulation_workflow",
    "validate_completed_standalone_articulation_checkpoint",
]
