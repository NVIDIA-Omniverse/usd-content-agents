# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Durable contracts for the articulation-v1 agentic workflow."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    StrictFloat,
    StrictInt,
    model_serializer,
    model_validator,
)

from content_agent_workflows.common.domain_execution import (
    DomainExecutionContext,
    domain_execution_context_from_metadata,
)

ARTICULATION_REQUEST_SCHEMA_VERSION: Literal[
    "content-agent-workflows.articulation-request.v1"
] = "content-agent-workflows.articulation-request.v1"
ARTICULATION_INFERENCE_SCHEMA_VERSION: Literal[
    "content-agent-workflows.articulation-inference.v1"
] = "content-agent-workflows.articulation-inference.v1"
ARTICULATION_REVIEW_RECEIPT_SCHEMA_VERSION: Literal[
    "content-agent-workflows.articulation-review-receipt.v1"
] = "content-agent-workflows.articulation-review-receipt.v1"
ARTICULATION_AUTHORING_REQUEST_SCHEMA_VERSION: Literal[
    "content-agent-workflows.articulation-authoring-request.v1"
] = "content-agent-workflows.articulation-authoring-request.v1"
ARTICULATION_AUTHORING_RESULT_SCHEMA_VERSION: Literal[
    "content-agent-workflows.articulation-authoring-result.v1"
] = "content-agent-workflows.articulation-authoring-result.v1"
ARTICULATION_VALIDATION_SCHEMA_VERSION: Literal[
    "content-agent-workflows.articulation-validation-evidence.v1"
] = "content-agent-workflows.articulation-validation-evidence.v1"
ARTICULATION_RUN_STATE_SCHEMA_VERSION: Literal[
    "content-agent-workflows.articulation-run-state.v1"
] = "content-agent-workflows.articulation-run-state.v1"
ARTICULATION_EMBEDDED_RUN_STATE_SCHEMA_VERSION: Literal[
    "content-agent-workflows.articulation-run-state.v4"
] = "content-agent-workflows.articulation-run-state.v4"
ARTICULATION_STANDALONE_RUN_STATE_SCHEMA_VERSION: Literal[
    "content-agent-workflows.articulation-run-state.v5"
] = "content-agent-workflows.articulation-run-state.v5"
ARTICULATION_FINALIZATION_RESULT_SCHEMA_VERSION: Literal[
    "content-agent-workflows.articulation-finalization-result.v1"
] = "content-agent-workflows.articulation-finalization-result.v1"
ARTICULATION_EMBEDDED_FINALIZATION_RESULT_SCHEMA_VERSION: Literal[
    "content-agent-workflows.articulation-finalization-result.v4"
] = "content-agent-workflows.articulation-finalization-result.v4"
ARTICULATION_STANDALONE_FINALIZATION_RESULT_SCHEMA_VERSION: Literal[
    "content-agent-workflows.articulation-finalization-result.v5"
] = "content-agent-workflows.articulation-finalization-result.v5"
STAGE2_SCHEMA_VERSION: Literal["joint-agent-stage2-v0"] = "joint-agent-stage2-v0"

ArticulationWorkflowMode = Literal["interactive", "batch"]
ArticulationReviewPolicy = Literal["uncertain", "all", "none"]
ArticulationV1MotionType = Literal["revolute", "prismatic"]
ArticulationWorkflowPhase = Literal[
    "initialized",
    "inferring",
    "collecting_evidence",
    "awaiting_decision",
    "needs_review",
    "authoring",
    "validating",
    "awaiting_post_review",
    "completed",
    "not_articulated",
    "conditional",
    "cancelled",
    "failed",
]
ArticulationFinalStatus = Literal[
    "awaiting_decision",
    "needs_review",
    "completed",
    "not_articulated",
    "conditional",
    "cancelled",
    "failed",
]
Stage2ReviewStatus = Literal["ready_for_rigger_input", "review_required"]
Stage2MotionType = Literal["revolute", "prismatic", "spherical", "fixed", "unknown"]
Stage2Confidence = Literal["high", "medium", "low"]
Stage2ParentResolutionSource = Literal[
    "stage1_hint",
    "stage1_rigger_evidence",
    "accepted_manifest",
    "structural_fallback",
    "unresolved",
]
Stage2FieldSource = Literal[
    "predicted",
    "consistency_corrected",
    "authored_metadata",
    "authored_reference",
    "source_metadata",
    "accepted_manifest",
    "stage1_hint",
    "stage1_rigger_evidence",
    "llm_adjudicated",
    "structural_fallback",
    "geometry_inferred",
    "template_default",
    "unknown",
]
Stage2ConnectivityEvidenceRole = Literal[
    "body0_body1_edge",
    "body1_ownership",
    "endpoint_canonicalization",
]
Stage2LimitReadiness = Literal[
    "not_provided",
    "source_backed",
    "rejected_conflicting_evidence",
    "rejected_untrusted_source",
    "rejected_missing_unit",
    "rejected_unsupported_joint_type",
    "rejected_unit_mismatch",
    "rejected_invalid_range",
]
Stage2UnresolvedReasonCode = Literal[
    "candidate_flag_conflict",
    "joint_type_conflict",
    "axis_missing",
    "axis_non_axis_aligned",
    "axis_geometry_ambiguous",
    "body1_unresolved",
    "parent_unresolved",
    "parent_self_reference",
    "compound_edge_conflict",
    "axis_evidence_conflict",
    "link_membership_conflict",
    "role_deferred_0_5",
]
MotionCapabilityDecisionReasonCode = (
    Stage2UnresolvedReasonCode
    | Literal[
        "motion_capability_conflict",
        "motion_capability_unsupported",
        "motion_capability_unresolved",
    ]
)
SemanticMotionCapabilityKind = Literal[
    "passive_rotation",
    "unsupported",
    "unresolved",
]
SemanticMotionMissingEvidence = Literal[
    "motion_kind",
    "passivity",
    "motion_contract",
]
MembershipDisposition = Literal[
    "co_rigid",
    "explicit_fixed",
    "independent_motion",
    "unresolved",
]
MembershipEvidenceSource = Literal[
    "predicted",
    "llm_adjudicated",
    "authored_metadata",
    "authored_reference",
    "source_metadata",
    "accepted_manifest",
    "human_reviewed",
    "unknown",
]
MembershipBoundary = Literal[
    "co_rigid_aggregation",
    "fixed_authoring_v2",
    "moving_candidate_generation",
    "human_review",
]
MembershipReviewStatus = Literal["resolved", "review_required"]
MembershipReasonCode = Literal[
    "fixed_authoring_evidence_required",
    "membership_evidence_conflict",
    "membership_evidence_missing",
    "physical_owner_unresolved",
]
ReviewDecision = Literal["accept", "reject"]

SUPPORTED_ARTICULATION_V1_TYPES = frozenset({"revolute", "prismatic"})
_DISALLOWED_FALLBACK_SOURCES = frozenset(
    {"geometry_inferred", "structural_fallback", "template_default", "unknown"}
)
_SOURCE_BACKED_LIMIT_SOURCES = frozenset(
    {
        "accepted_manifest",
        "authored_metadata",
        "authored_reference",
        "source_metadata",
        "template_default",
    }
)
TERMINAL_ARTICULATION_PHASES = frozenset(
    {"completed", "not_articulated", "conditional", "cancelled", "failed"}
)


def _is_canonical_absolute_prim_path(value: str) -> bool:
    """Match the Joint Rigger endpoint contract without requiring OpenUSD."""

    try:
        from pxr import Sdf
    except ImportError:
        if not value.startswith("/") or value == "/":
            return False
        return all(segment.isidentifier() for segment in value[1:].split("/"))

    validation = Sdf.Path.IsValidPathString(value)
    is_valid = validation[0] if isinstance(validation, tuple) else validation
    if not is_valid:
        return False
    path = Sdf.Path(value)
    return bool(
        str(path) == value
        and path.IsAbsolutePath()
        and path.IsPrimPath()
        and not path.IsAbsoluteRootPath()
        and not path.ContainsPrimVariantSelection()
    )


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _require_unique_ids(field_name: str, values: tuple[str, ...]) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must contain unique IDs")


class ArticulationWorkflowRequest(_StrictModel):
    """Source-bound request shared by interactive and batch entry points."""

    schema_version: Literal["content-agent-workflows.articulation-request.v1"] = (
        ARTICULATION_REQUEST_SCHEMA_VERSION
    )
    source_asset: str = Field(min_length=1)
    output_dir: Path
    intent: str = Field(
        default=(
            "Infer source-bound articulation candidates, review uncertain "
            "candidates, and author approved joint topology."
        ),
        min_length=1,
    )
    target_runtime: str = Field(default="usd-cli", min_length=1)
    review_policy: ArticulationReviewPolicy = "uncertain"
    allowed_motion_types: tuple[ArticulationV1MotionType, ...] = (
        "revolute",
        "prismatic",
    )
    expected_candidate_count: int | None = Field(default=None, ge=0, le=256)
    max_candidate_count: int = Field(default=64, ge=1, le=256)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_requested_scope(self) -> Self:
        if not self.allowed_motion_types:
            raise ValueError("allowed_motion_types must not be empty")
        _require_unique_ids("allowed_motion_types", self.allowed_motion_types)
        if (
            self.expected_candidate_count is not None
            and self.expected_candidate_count > self.max_candidate_count
        ):
            raise ValueError(
                "expected_candidate_count must not exceed max_candidate_count"
            )
        domain_execution_context_from_metadata(
            self.metadata,
            expected_domain="articulation",
        )
        return self

    @property
    def execution_context(self) -> DomainExecutionContext | None:
        """Return the validated lifecycle boundary without adding a model field."""

        return domain_execution_context_from_metadata(
            self.metadata,
            expected_domain="articulation",
        )


class Stage2CandidateSummary(BaseModel):
    """Compatibility view of the upstream Stage 2 summary."""

    model_config = ConfigDict(extra="allow", frozen=True)

    candidate_count: StrictInt = Field(ge=0)
    ready_candidate_count: StrictInt = Field(ge=0)
    review_required_candidate_count: StrictInt = Field(ge=0)


class Stage2EvidenceItem(BaseModel):
    """Strict compatibility view of structured Joint Agent evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: Stage2FieldSource = "unknown"
    description: str
    value: str | None = None
    prim_paths: tuple[str, ...] = ()
    connectivity_role: Stage2ConnectivityEvidenceRole | None = None


class Stage2SemanticMotionCapability(BaseModel):
    """Immutable compatibility view of one semantic motion decision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: SemanticMotionCapabilityKind
    source: Stage2FieldSource = "predicted"
    evidence: str = ""
    missing_evidence: tuple[SemanticMotionMissingEvidence, ...] = ()
    missing_contract: str | None = None

    @model_validator(mode="after")
    def _validate_disposition(self) -> Self:
        evidence = self.evidence.strip()
        missing_contract = (self.missing_contract or "").strip()
        if self.kind == "passive_rotation":
            if not evidence:
                raise ValueError("passive_rotation requires capability evidence")
            if self.missing_evidence or missing_contract:
                raise ValueError(
                    "passive_rotation cannot carry missing evidence or contract"
                )
        elif self.kind == "unsupported":
            if not missing_contract:
                raise ValueError("unsupported capability requires missing_contract")
        elif not self.missing_evidence:
            raise ValueError("unresolved capability requires missing_evidence")
        return self


class MembershipDispositionRecord(BaseModel):
    """Immutable compatibility view of one Joint membership decision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    disposition_id: str = Field(min_length=1)
    member_prim: str = Field(min_length=1)
    motion_candidate_prim: str | None = None
    physical_owner_prim: str | None = None
    physical_owner_candidate_prim: str | None = None
    disposition: MembershipDisposition
    attachment_parent_prim: str | None = None
    attachment_child_prim: str | None = None
    source: MembershipEvidenceSource
    confidence: Stage2Confidence
    rationale: str = ""
    source_prediction_ids: tuple[str, ...] = ()
    downstream_boundary: MembershipBoundary
    review_status: MembershipReviewStatus
    reason_codes: tuple[MembershipReasonCode, ...] = ()
    unresolved_questions: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _validate_disposition(self) -> Self:
        for name, path in (
            ("member_prim", self.member_prim),
            ("motion_candidate_prim", self.motion_candidate_prim),
            ("physical_owner_prim", self.physical_owner_prim),
            ("physical_owner_candidate_prim", self.physical_owner_candidate_prim),
            ("attachment_parent_prim", self.attachment_parent_prim),
            ("attachment_child_prim", self.attachment_child_prim),
        ):
            if path is not None and not _is_canonical_absolute_prim_path(path):
                raise ValueError(f"{name} must be an absolute USD prim path")
        if self.disposition in {"co_rigid", "independent_motion"} and not (
            self.physical_owner_prim
        ):
            raise ValueError("resolved membership requires a physical owner")
        if self.disposition == "explicit_fixed" and (
            not self.attachment_parent_prim or not self.attachment_child_prim
        ):
            raise ValueError("explicit fixed membership requires both endpoints")
        if self.disposition != "explicit_fixed" and (
            self.attachment_parent_prim or self.attachment_child_prim
        ):
            raise ValueError("only explicit fixed membership may carry endpoints")
        if self.review_status == "resolved" and self.disposition == "unresolved":
            raise ValueError("unresolved membership must require review")
        return self


class MembershipDispositionSummary(BaseModel):
    """Compatibility counts for the standalone Joint membership artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    disposition_count: StrictInt = Field(ge=0)
    disposition_counts: dict[MembershipDisposition, StrictInt] = Field(
        default_factory=dict
    )
    review_required_count: StrictInt = Field(ge=0)
    pending_downstream_count: StrictInt = Field(ge=0)


class MembershipDispositionDocument(BaseModel):
    """Typed membership output kept separate from Stage 2 v0."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["joint-agent-membership-disposition-v1"]
    summary: MembershipDispositionSummary
    dispositions: tuple[MembershipDispositionRecord, ...]

    @model_validator(mode="after")
    def _validate_summary(self) -> Self:
        ids = self.disposition_ids
        members = tuple(record.member_prim for record in self.dispositions)
        _require_unique_ids("membership disposition IDs", ids)
        _require_unique_ids("membership member prims", members)
        if tuple(sorted(members)) != members:
            raise ValueError("membership dispositions must use stable member order")
        counts = Counter(record.disposition for record in self.dispositions)
        if self.summary.disposition_count != len(self.dispositions):
            raise ValueError("membership disposition_count does not match records")
        if self.summary.disposition_counts != dict(sorted(counts.items())):
            raise ValueError("membership disposition_counts do not match records")
        if self.summary.review_required_count != sum(
            record.review_status == "review_required" for record in self.dispositions
        ):
            raise ValueError("membership review_required_count does not match records")
        if self.summary.pending_downstream_count != sum(
            record.disposition in {"explicit_fixed", "unresolved"}
            for record in self.dispositions
        ):
            raise ValueError(
                "membership pending_downstream_count does not match records"
            )
        return self

    @property
    def disposition_ids(self) -> tuple[str, ...]:
        """Return membership IDs in stable member order."""

        return tuple(record.disposition_id for record in self.dispositions)


class Stage2ArticulationCandidate(BaseModel):
    """Immutable compatibility envelope for one Joint Agent Stage 2 candidate."""

    model_config = ConfigDict(extra="allow", frozen=True)

    schema_version: Literal["joint-agent-stage2-v0"] = STAGE2_SCHEMA_VERSION
    candidate_id: str = Field(min_length=1)
    motion_type: Stage2MotionType = "unknown"
    moving_part_prims: tuple[str, ...] = ()
    fixed_parent_prim: str | None = None
    parent_resolution_source: Stage2ParentResolutionSource = "unresolved"
    joint_type_hint: str = "unknown"
    axis_hint: str = "unknown"
    motion_axis_world: tuple[StrictFloat, StrictFloat, StrictFloat] | None = None
    confidence: Stage2Confidence = "low"
    parent_hint: str = "unknown"
    child_hint: str = "unknown"
    component_name: str = "unknown"
    component_type: str = "unknown"
    role: str = "unknown"
    semantic_role: str | None = Field(
        default=None,
        pattern=r"^[a-z0-9]+(?:_[a-z0-9]+)*$",
    )
    motion_capability: Stage2SemanticMotionCapability | None = None
    source_prediction_ids: tuple[str, ...] = ()
    evidence: str = ""
    source_annotation_conflicts: dict[str, tuple[str, ...]] = Field(
        default_factory=dict
    )
    field_sources: dict[str, Stage2FieldSource] = Field(default_factory=dict)
    axis_evidence: tuple[Stage2EvidenceItem, ...] = ()
    connectivity_evidence: tuple[Stage2EvidenceItem, ...] = ()
    lower_limit: StrictFloat | None = None
    upper_limit: StrictFloat | None = None
    limit_unit: str = "unknown"
    limit_source: Stage2FieldSource = "unknown"
    limit_readiness: Stage2LimitReadiness = "not_provided"
    limit_evidence: tuple[Stage2EvidenceItem, ...] = ()
    review_status: Stage2ReviewStatus = "review_required"
    unresolved_reason_codes: tuple[Stage2UnresolvedReasonCode, ...] = ()
    unresolved_questions: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _validate_axis_contract(self) -> Self:
        axis_vectors = {
            "x": (1.0, 0.0, 0.0),
            "+x": (1.0, 0.0, 0.0),
            "-x": (-1.0, 0.0, 0.0),
            "y": (0.0, 1.0, 0.0),
            "+y": (0.0, 1.0, 0.0),
            "-y": (0.0, -1.0, 0.0),
            "z": (0.0, 0.0, 1.0),
            "+z": (0.0, 0.0, 1.0),
            "-z": (0.0, 0.0, -1.0),
        }
        if self.motion_axis_world is not None:
            expected = axis_vectors.get(self.axis_hint)
            if expected is None:
                raise ValueError(
                    "motion_axis_world requires an explicit axis-aligned axis_hint"
                )
            if self.motion_axis_world != expected:
                raise ValueError(
                    "motion_axis_world must exactly match the signed axis_hint"
                )
        return self

    @model_validator(mode="after")
    def _validate_parent_contract(self) -> Self:
        if (
            self.parent_resolution_source
            in {
                "stage1_hint",
                "stage1_rigger_evidence",
                "structural_fallback",
            }
            and not self.fixed_parent_prim
        ):
            raise ValueError(
                "fixed_parent_prim is required for a resolved parent source"
            )
        return self

    @model_validator(mode="after")
    def _validate_limit_contract(self) -> Self:
        has_limit = self.lower_limit is not None or self.upper_limit is not None
        for value in (self.lower_limit, self.upper_limit):
            if value is not None and not math.isfinite(value):
                raise ValueError("candidate limits must be finite")
        if (
            self.lower_limit is not None
            and self.upper_limit is not None
            and self.lower_limit > self.upper_limit
        ):
            raise ValueError("candidate lower_limit must not exceed upper_limit")
        if self.limit_readiness == "source_backed":
            if not has_limit or not self.limit_evidence:
                raise ValueError(
                    "source-backed limits require a value and limit evidence"
                )
            expected_unit = "degrees" if self.motion_type == "revolute" else "meters"
            if self.limit_unit != expected_unit:
                raise ValueError(f"{self.motion_type} limits must use {expected_unit}")
        elif has_limit:
            raise ValueError("numeric limits require limit_readiness=source_backed")
        return self

    @model_serializer(mode="wrap")
    def _omit_empty_semantic_motion_fields(  # type: ignore[no-untyped-def]
        self,
        handler: SerializerFunctionWrapHandler,
    ):
        """Preserve legacy Stage 2 serialization for candidates without decisions."""

        serialized = handler(self)
        if self.semantic_role is None:
            serialized.pop("semantic_role", None)
        if self.motion_capability is None:
            serialized.pop("motion_capability", None)
        return serialized

    @property
    def articulation_v1_type(self) -> str:
        """Return the normalized type used by the articulation-v1 gate."""

        if self.motion_type != "unknown":
            return self.motion_type
        return self.joint_type_hint.strip().lower()

    @property
    def is_native_ready(self) -> bool:
        """Return whether Joint Agent marked the source evidence authorable."""

        return self.review_status == "ready_for_rigger_input"

    def _has_exact_axis_evidence(self) -> bool:
        required_sources = (
            self.field_sources.get("axis_hint"),
            self.field_sources.get("motion_axis_world"),
        )
        if any(source is None for source in required_sources):
            return False
        if any(
            item.source in _DISALLOWED_FALLBACK_SOURCES for item in self.axis_evidence
        ):
            return False
        for source in required_sources:
            source_items = tuple(
                item for item in self.axis_evidence if item.source == source
            )
            if not source_items or any(
                (item.value or "").rsplit("->", maxsplit=1)[-1] != self.axis_hint
                for item in source_items
            ):
                return False
        return True

    @staticmethod
    def _is_strict_ancestor(ancestor: str, descendant: str) -> bool:
        prefix = ancestor.rstrip("/") + "/"
        return ancestor != descendant and descendant.startswith(prefix)

    def _typed_connectivity_item_is_valid(
        self,
        item: Stage2EvidenceItem,
        *,
        body0: str,
        body1: str,
    ) -> bool:
        if item.connectivity_role is None:
            return True
        if item.connectivity_role == "body0_body1_edge":
            return item.value == body0 and item.prim_paths == (body0, body1)
        if item.connectivity_role == "body1_ownership":
            return bool(
                item.value == body1
                and body1 in item.prim_paths
                and all(
                    path == body1 or self._is_strict_ancestor(body1, path)
                    for path in item.prim_paths
                )
            )
        if len(item.prim_paths) != 2:
            return False
        wrapper, leaf = item.prim_paths
        return bool(
            item.value == leaf
            and leaf in {body0, body1}
            and self._is_strict_ancestor(wrapper, leaf)
        )

    def _legacy_connectivity_role(
        self,
        item: Stage2EvidenceItem,
        *,
        body0: str,
        body1: str,
    ) -> Literal["edge", "other"] | None:
        paths = item.prim_paths
        if (
            not paths
            or len(paths) != len(set(paths))
            or any(not path.startswith("/") or path == "/" for path in paths)
        ):
            return None
        path_values = set(paths)
        endpoints = {body0, body1}
        if item.value == f"{body0}->{body1}":
            return "edge" if len(paths) == 2 and path_values == endpoints else None
        if item.value == body0:
            if len(paths) == 2 and path_values == endpoints:
                return "edge"
            if len(paths) == 3 and endpoints < path_values:
                (alias,) = path_values - endpoints
                if self._is_strict_ancestor(alias, body0):
                    return "edge"
            return None
        if item.value == body1:
            if paths == (body1,):
                return "other"
            if len(paths) == 2 and body1 in path_values:
                (alias,) = path_values - {body1}
                if self._is_strict_ancestor(alias, body1) or self._is_strict_ancestor(
                    body1,
                    alias,
                ):
                    return "other"
        return None

    def _has_exact_connectivity_evidence(self) -> bool:
        if self.fixed_parent_prim is None or len(self.moving_part_prims) != 1:
            return False
        body0 = self.fixed_parent_prim
        body1 = self.moving_part_prims[0]
        if body0 == body1:
            return False
        if any(
            item.source in _DISALLOWED_FALLBACK_SOURCES
            for item in self.connectivity_evidence
        ):
            return False
        if any(
            not self._typed_connectivity_item_is_valid(
                item,
                body0=body0,
                body1=body1,
            )
            for item in self.connectivity_evidence
        ):
            return False
        parent_source = self.field_sources.get("fixed_parent_prim")
        if parent_source is None:
            return False
        has_edge = False
        for item in self.connectivity_evidence:
            if item.source != parent_source:
                continue
            if item.connectivity_role == "body0_body1_edge":
                has_edge = True
            elif item.connectivity_role is None:
                legacy_role = self._legacy_connectivity_role(
                    item,
                    body0=body0,
                    body1=body1,
                )
                if legacy_role is None:
                    return False
                has_edge = has_edge or legacy_role == "edge"
        return has_edge

    def _has_exact_limit_evidence(self) -> bool:
        if self.limit_readiness != "source_backed":
            return True
        return bool(
            self.limit_source in _SOURCE_BACKED_LIMIT_SOURCES
            and self.limit_evidence
            and all(item.source == self.limit_source for item in self.limit_evidence)
        )

    def _has_valid_endpoint_paths(self) -> bool:
        return bool(
            len(self.moving_part_prims) == 1
            and _is_canonical_absolute_prim_path(self.fixed_parent_prim or "")
            and _is_canonical_absolute_prim_path(self.moving_part_prims[0])
        )

    @property
    def is_articulation_v1_authorable(self) -> bool:
        """Apply the workflow's additional fail-closed articulation-v1 gate."""

        required_sources = tuple(
            self.field_sources.get(field)
            for field in (
                "motion_type",
                "axis_hint",
                "motion_axis_world",
                "fixed_parent_prim",
            )
        )
        return bool(
            self.is_native_ready
            and self.articulation_v1_type in SUPPORTED_ARTICULATION_V1_TYPES
            and self.joint_type_hint == self.motion_type
            and self._has_valid_endpoint_paths()
            and self.parent_resolution_source
            not in {"structural_fallback", "unresolved"}
            and self.motion_axis_world is not None
            and not self.unresolved_reason_codes
            and not self.unresolved_questions
            and self.role.strip().lower()
            not in {"", "unknown", "none", "null", "n/a", "na"}
            and all(
                source is not None and source not in _DISALLOWED_FALLBACK_SOURCES
                for source in required_sources
            )
            and self._has_exact_axis_evidence()
            and self._has_exact_connectivity_evidence()
            and self._has_exact_limit_evidence()
        )


class Stage2CandidateDocument(BaseModel):
    """Immutable pass-through envelope for ``joint-agent-stage2-v0``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["joint-agent-stage2-v0"] = STAGE2_SCHEMA_VERSION
    summary: Stage2CandidateSummary
    candidates: tuple[Stage2ArticulationCandidate, ...]

    @model_validator(mode="after")
    def _validate_candidate_scope(self) -> Self:
        candidate_ids = self.candidate_ids
        _require_unique_ids("candidate IDs", candidate_ids)
        ready_count = sum(candidate.is_native_ready for candidate in self.candidates)
        review_count = len(self.candidates) - ready_count
        if self.summary.candidate_count != len(self.candidates):
            raise ValueError("summary candidate_count does not match candidates")
        if self.summary.ready_candidate_count != ready_count:
            raise ValueError("summary ready_candidate_count does not match candidates")
        if self.summary.review_required_candidate_count != review_count:
            raise ValueError(
                "summary review_required_candidate_count does not match candidates"
            )
        return self

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        """Return candidate IDs in immutable upstream order."""

        return tuple(candidate.candidate_id for candidate in self.candidates)

    def candidate_by_id(self) -> dict[str, Stage2ArticulationCandidate]:
        """Return the unique candidate lookup."""

        return {candidate.candidate_id: candidate for candidate in self.candidates}


class ArticulationInferenceResult(_StrictModel):
    """Normalized result of Joint Agent prediction and Stage 2 inference."""

    schema_version: Literal["content-agent-workflows.articulation-inference.v1"] = (
        ARTICULATION_INFERENCE_SCHEMA_VERSION
    )
    candidate_document: Stage2CandidateDocument
    membership_disposition_document: MembershipDispositionDocument | None = None
    backend_configuration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    predictions_path: str | None = None
    predictions_sha256: str | None = None
    report_path: str | None = None
    report_sha256: str | None = None
    backend_run_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_bound_optional_artifacts(self) -> Self:
        if bool(self.predictions_path) != bool(self.predictions_sha256):
            raise ValueError(
                "predictions_path and predictions_sha256 must be set together"
            )
        if bool(self.report_path) != bool(self.report_sha256):
            raise ValueError("report_path and report_sha256 must be set together")
        return self

    @model_validator(mode="after")
    def _validate_membership_candidate_bindings(self) -> Self:
        document = self.membership_disposition_document
        membership_schema = self.metadata.get("membership_disposition_schema_version")
        membership_required = (
            self.metadata.get("backend") == "joint-agent-local"
            or self.metadata.get("membership_disposition_required") is True
        )
        if membership_required:
            if membership_schema != "joint-agent-membership-disposition-v1":
                raise ValueError(
                    "membership-enabled inference requires the typed membership "
                    "disposition contract"
                )
            if document is None:
                raise ValueError(
                    "membership-enabled inference requires membership dispositions"
                )
            if (
                self.metadata.get("membership_disposition_classic_fallback_used")
                is not False
            ):
                raise ValueError(
                    "Joint Agent local inference must fail closed without a fixed-pipeline "
                    "membership fallback"
                )
        elif membership_schema is not None and document is None:
            raise ValueError(
                "membership disposition schema metadata requires its typed document"
            )
        if document is None:
            return self
        candidate_prims = {
            candidate.moving_part_prims[0]
            for candidate in self.candidate_document.candidates
            if candidate.moving_part_prims
        }
        for record in document.dispositions:
            if (
                record.motion_candidate_prim is not None
                and record.motion_candidate_prim not in candidate_prims
            ):
                raise ValueError(
                    "membership disposition references an unknown motion candidate"
                )
            if (
                record.physical_owner_candidate_prim is not None
                and record.physical_owner_candidate_prim not in candidate_prims
            ):
                raise ValueError(
                    "membership disposition references an unknown owner candidate"
                )
            if (
                record.disposition == "independent_motion"
                and record.motion_candidate_prim is None
            ):
                raise ValueError(
                    "independent membership requires a bound motion candidate"
                )
        authorable_candidate_prims = {
            candidate.moving_part_prims[0]
            for candidate in self.candidate_document.candidates
            if candidate.moving_part_prims
            and candidate.motion_type in SUPPORTED_ARTICULATION_V1_TYPES
        }
        covered_candidate_prims = {
            record.motion_candidate_prim
            for record in document.dispositions
            if record.motion_candidate_prim is not None
        }
        missing_candidate_prims = sorted(
            authorable_candidate_prims - covered_candidate_prims
        )
        if missing_candidate_prims:
            raise ValueError(
                "membership dispositions do not cover authorable Stage 2 candidates: "
                + ", ".join(missing_candidate_prims)
            )
        return self


class ArticulationReviewEntry(_StrictModel):
    """One explicit operator decision for a candidate needing workflow review."""

    candidate_id: str = Field(min_length=1)
    decision: ReviewDecision
    note: str | None = None


class ArticulationReviewReceipt(_StrictModel):
    """Immutable review decisions bound to exact request and candidate evidence."""

    schema_version: Literal[
        "content-agent-workflows.articulation-review-receipt.v1"
    ] = ARTICULATION_REVIEW_RECEIPT_SCHEMA_VERSION
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_dependency_bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    scene_evidence_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    reviewer: str = Field(min_length=1)
    decisions: tuple[ArticulationReviewEntry, ...]

    @model_validator(mode="after")
    def _validate_unique_decisions(self) -> Self:
        _require_unique_ids(
            "review decision candidate IDs",
            tuple(decision.candidate_id for decision in self.decisions),
        )
        return self


class ArtifactBinding(_StrictModel):
    """Path and digest pair used by durable run state."""

    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ArticulationAuthoringRequest(_StrictModel):
    """Exact accepted-only request sent to the Joint Agent authoring adapter."""

    schema_version: Literal[
        "content-agent-workflows.articulation-authoring-request.v1"
    ] = ARTICULATION_AUTHORING_REQUEST_SCHEMA_VERSION
    source_asset: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_dependency_bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_document_path: str = Field(min_length=1)
    candidate_document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    accepted_candidate_ids: tuple[str, ...] = Field(min_length=1)
    idempotency_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    predictions_path: str | None = None
    predictions_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    accepted_authoring_plan_path: str | None = None
    accepted_authoring_plan_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    output_dir: Path
    apply_masses: Literal[False] = False
    apply_collision: Literal[False] = False

    @model_validator(mode="after")
    def _validate_accepted_scope(self) -> Self:
        _require_unique_ids("accepted_candidate_ids", self.accepted_candidate_ids)
        if bool(self.predictions_path) != bool(self.predictions_sha256):
            raise ValueError(
                "predictions_path and predictions_sha256 must be set together"
            )
        if bool(self.accepted_authoring_plan_path) != bool(
            self.accepted_authoring_plan_sha256
        ):
            raise ValueError(
                "accepted authoring plan path and digest must be set together"
            )
        if self.accepted_authoring_plan_path is not None and (
            self.predictions_path is not None
        ):
            raise ValueError(
                "accepted authoring plans cannot be mixed with prediction authority"
            )
        return self


class ArticulationAuthoringResult(_StrictModel):
    """Identity-bound publication result returned by the Joint Agent adapter."""

    schema_version: Literal[
        "content-agent-workflows.articulation-authoring-result.v1"
    ] = ARTICULATION_AUTHORING_RESULT_SCHEMA_VERSION
    status: Literal["succeeded"]
    idempotency_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_dependency_bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_document_path: str = Field(min_length=1)
    candidate_document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_asset_path: str = Field(min_length=1)
    output_asset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    authored_candidate_ids: tuple[str, ...] = Field(min_length=1)
    authored_joint_count: int = Field(ge=1)
    diagnostics_path: str | None = None
    diagnostics_sha256: str | None = None
    joint_rigger_result_path: str | None = None
    joint_rigger_result_sha256: str | None = None
    membership_operation_receipt_path: str | None = None
    membership_operation_receipt_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    backend_call_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_publication_scope(self) -> Self:
        _require_unique_ids("authored_candidate_ids", self.authored_candidate_ids)
        if self.authored_joint_count != len(self.authored_candidate_ids):
            raise ValueError("authored_joint_count must match authored_candidate_ids")
        if bool(self.diagnostics_path) != bool(self.diagnostics_sha256):
            raise ValueError(
                "diagnostics_path and diagnostics_sha256 must be set together"
            )
        if bool(self.joint_rigger_result_path) != bool(self.joint_rigger_result_sha256):
            raise ValueError("joint_rigger_result_path and digest must be set together")
        if bool(self.membership_operation_receipt_path) != bool(
            self.membership_operation_receipt_sha256
        ):
            raise ValueError(
                "membership operation receipt path and digest must be set together"
            )
        return self


class ArticulationValidationResult(_StrictModel):
    """Exact saved-graph readback and package-portability evidence."""

    schema_version: Literal[
        "content-agent-workflows.articulation-validation-evidence.v1"
    ] = ARTICULATION_VALIDATION_SCHEMA_VERSION
    status: Literal["pass", "fail"]
    output_asset_path: str = Field(min_length=1)
    expected_output_asset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    observed_output_asset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_candidate_ids: tuple[str, ...] = Field(min_length=1)
    validated_candidate_ids: tuple[str, ...] = ()
    exact_graph_match: bool
    self_contained: bool
    failures: tuple[str, ...] = ()
    evidence_paths: tuple[str, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_readback_claim(self) -> Self:
        _require_unique_ids("expected_candidate_ids", self.expected_candidate_ids)
        _require_unique_ids("validated_candidate_ids", self.validated_candidate_ids)
        passed = bool(
            self.expected_output_asset_sha256 == self.observed_output_asset_sha256
            and self.exact_graph_match
            and self.self_contained
            and not self.failures
            and self.validated_candidate_ids == self.expected_candidate_ids
        )
        if (self.status == "pass") != passed:
            raise ValueError(
                "validation status must reflect exact graph, package, and ID checks"
            )
        return self


class ArticulationStateTransition(_StrictModel):
    """Append-only record for a durable articulation phase transition."""

    timestamp: str = Field(min_length=1)
    from_phase: ArticulationWorkflowPhase
    to_phase: ArticulationWorkflowPhase
    reason: str = Field(min_length=1)


class ArticulationRecoveryAction(_StrictModel):
    """Exact resume or clean-restart operation for a terminal Joint failure."""

    kind: Literal["resume", "restart"]
    operation: Literal["joint_agent.api.pipeline"]
    resume: bool
    clean: bool
    failed_step: str = Field(min_length=1)
    checkpoint_path: str = Field(min_length=1)

    @model_validator(mode="after")
    def _require_consistent_recovery_mode(self) -> Self:
        expected = (True, False) if self.kind == "resume" else (False, True)
        if (self.resume, self.clean) != expected:
            raise ValueError("recovery action flags must match its kind")
        return self


class ArticulationTerminalStatus(_StrictModel):
    """Strict non-success contract for required topology reconciliation."""

    requested: Literal[True]
    attempted: bool
    accepted: Literal[False]
    outcome: Literal["failed"]
    failure_stage: str = Field(min_length=1)
    error_type: str = Field(min_length=1)
    success: Literal[False]
    terminal: Literal[True]
    required: Literal[True]
    recovery_action: ArticulationRecoveryAction
    reason: str | None = None
    configured_max_images: int | None = None
    source_prediction_count: int | None = Field(default=None, ge=0)
    required_image_budget: int | None = Field(default=None, ge=0)
    authoritative_group_count: int | None = Field(default=None, ge=0)
    grouping_mode: Literal["source_prim", "rigid_body_owner"] | None = None
    attempt_diagnostics: tuple[dict[str, Any], ...] = ()
    diagnostics_artifact_path: str | None = Field(default=None, min_length=1)
    diagnostics_artifact_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    diagnostics_artifact_status: Literal["persisted", "unavailable"] | None = None
    diagnostics_persistence_error_type: str | None = Field(
        default=None,
        min_length=1,
    )
    unresolved_decisions: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _require_consistent_diagnostics_evidence(self) -> Self:
        status = self.diagnostics_artifact_status
        if status is None:
            if (
                self.diagnostics_artifact_sha256 is not None
                or self.diagnostics_persistence_error_type is not None
            ):
                raise ValueError(
                    "legacy terminal status cannot carry new diagnostics evidence"
                )
            if self.recovery_action.kind == "restart":
                raise ValueError(
                    "restart recovery requires unavailable diagnostics evidence"
                )
            return self

        if status == "persisted":
            if (
                self.diagnostics_artifact_path is None
                or self.diagnostics_artifact_sha256 is None
            ):
                raise ValueError(
                    "persisted diagnostics evidence requires path and digest"
                )
            if self.recovery_action.kind != "resume":
                raise ValueError(
                    "persisted diagnostics evidence requires resume recovery"
                )
            return self

        if (
            self.diagnostics_artifact_path is not None
            or self.diagnostics_artifact_sha256 is not None
        ):
            raise ValueError(
                "unavailable diagnostics evidence cannot carry path or digest"
            )
        if self.recovery_action.kind != "restart":
            raise ValueError(
                "unavailable diagnostics evidence requires restart recovery"
            )
        return self


class ArticulationRunState(_StrictModel):
    """Mutable, revisioned checkpoint for one articulation workflow run."""

    schema_version: Literal[
        "content-agent-workflows.articulation-run-state.v1",
        "content-agent-workflows.articulation-run-state.v2",
        "content-agent-workflows.articulation-run-state.v3",
        "content-agent-workflows.articulation-run-state.v4",
        "content-agent-workflows.articulation-run-state.v5",
    ] = ARTICULATION_RUN_STATE_SCHEMA_VERSION
    revision: int = Field(default=0, ge=0)
    mode: ArticulationWorkflowMode
    phase: ArticulationWorkflowPhase = "initialized"
    request: ArtifactBinding
    source_asset: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_dependency_bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    backend_configuration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    scene_evidence_configuration_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    inference_result: ArtifactBinding | None = None
    candidate_document: ArtifactBinding | None = None
    scene_evidence: ArtifactBinding | None = None
    review_receipt: ArtifactBinding | None = None
    approved_candidate_document: ArtifactBinding | None = None
    authoring_request: ArtifactBinding | None = None
    authoring_result: ArtifactBinding | None = None
    validation_result: ArtifactBinding | None = None
    embedded_evidence: ArtifactBinding | None = None
    embedded_proposal: ArtifactBinding | None = None
    embedded_canonical_graph: ArtifactBinding | None = None
    embedded_graph_revision: ArtifactBinding | None = None
    embedded_outer_review: ArtifactBinding | None = None
    embedded_coordinator_decision: ArtifactBinding | None = None
    embedded_human_decision: ArtifactBinding | None = None
    embedded_execution_authorization: ArtifactBinding | None = None
    embedded_readback: ArtifactBinding | None = None
    embedded_execution_result: ArtifactBinding | None = None
    embedded_output_evidence: ArtifactBinding | None = None
    embedded_coordinator_review: ArtifactBinding | None = None
    embedded_decision_receipt: ArtifactBinding | None = None
    embedded_terminal_receipt: ArtifactBinding | None = None
    standalone_identity: ArtifactBinding | None = None
    standalone_preparation: ArtifactBinding | None = None
    standalone_proposal: ArtifactBinding | None = None
    standalone_decision_patch: ArtifactBinding | None = None
    standalone_decision_ledger: ArtifactBinding | None = None
    standalone_canonical_graph: ArtifactBinding | None = None
    standalone_authoring_receipt: ArtifactBinding | None = None
    standalone_readback: ArtifactBinding | None = None
    standalone_output_evidence: ArtifactBinding | None = None
    standalone_post_review: ArtifactBinding | None = None
    standalone_cleanup: ArtifactBinding | None = None
    standalone_terminal_receipt: ArtifactBinding | None = None
    standalone_refinement_history: tuple[ArtifactBinding, ...] = ()
    candidate_ids: tuple[str, ...] = ()
    review_required_candidate_ids: tuple[str, ...] = ()
    auto_accepted_candidate_ids: tuple[str, ...] = ()
    accepted_candidate_ids: tuple[str, ...] = ()
    rejected_candidate_ids: tuple[str, ...] = ()
    unresolved_candidate_ids: tuple[str, ...] = ()
    review_required_membership_disposition_ids: tuple[str, ...] = ()
    unresolved_membership_disposition_ids: tuple[str, ...] = ()
    transitions: tuple[ArticulationStateTransition, ...] = ()
    error: str | None = None
    backend_terminal_status: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _validate_state_scopes(self) -> Self:
        embedded_bindings = (
            self.embedded_evidence,
            self.embedded_proposal,
            self.embedded_canonical_graph,
            self.embedded_graph_revision,
            self.embedded_outer_review,
            self.embedded_coordinator_decision,
            self.embedded_human_decision,
            self.embedded_execution_authorization,
            self.embedded_readback,
            self.embedded_execution_result,
            self.embedded_output_evidence,
            self.embedded_coordinator_review,
            self.embedded_decision_receipt,
            self.embedded_terminal_receipt,
        )
        standalone_bindings = (
            self.standalone_identity,
            self.standalone_preparation,
            self.standalone_proposal,
            self.standalone_decision_patch,
            self.standalone_decision_ledger,
            self.standalone_canonical_graph,
            self.standalone_authoring_receipt,
            self.standalone_readback,
            self.standalone_output_evidence,
            self.standalone_post_review,
            self.standalone_cleanup,
            self.standalone_terminal_receipt,
        )
        if self.schema_version == ARTICULATION_RUN_STATE_SCHEMA_VERSION and any(
            binding is not None for binding in embedded_bindings
        ):
            raise ValueError("embedded articulation bindings require an embedded run")
        if (
            self.schema_version == "content-agent-workflows.articulation-run-state.v2"
            and any(
                binding is not None
                for binding in (
                    self.embedded_graph_revision,
                    self.embedded_outer_review,
                    self.embedded_output_evidence,
                    self.embedded_terminal_receipt,
                )
            )
        ):
            raise ValueError("embedded output review fields require run-state v3")
        if (
            self.schema_version == "content-agent-workflows.articulation-run-state.v3"
            and self.embedded_graph_revision is not None
        ):
            raise ValueError("embedded graph revision requires run-state v4")
        if self.schema_version != ARTICULATION_STANDALONE_RUN_STATE_SCHEMA_VERSION and (
            any(binding is not None for binding in standalone_bindings)
            or self.standalone_refinement_history
        ):
            raise ValueError(
                "standalone decision custody bindings require run-state v5"
            )
        if (
            self.schema_version == ARTICULATION_STANDALONE_RUN_STATE_SCHEMA_VERSION
            and any(binding is not None for binding in embedded_bindings)
        ):
            raise ValueError("standalone decision custody cannot mix embedded bindings")
        if (
            self.schema_version == ARTICULATION_STANDALONE_RUN_STATE_SCHEMA_VERSION
            and self.standalone_preparation is None
        ):
            raise ValueError(
                "standalone decision custody requires a preparation binding"
            )
        scopes = (
            ("candidate_ids", self.candidate_ids),
            ("review_required_candidate_ids", self.review_required_candidate_ids),
            ("auto_accepted_candidate_ids", self.auto_accepted_candidate_ids),
            ("accepted_candidate_ids", self.accepted_candidate_ids),
            ("rejected_candidate_ids", self.rejected_candidate_ids),
            ("unresolved_candidate_ids", self.unresolved_candidate_ids),
        )
        for name, values in scopes:
            _require_unique_ids(name, values)
            if name != "candidate_ids" and not set(values).issubset(self.candidate_ids):
                raise ValueError(f"{name} must be a subset of candidate_ids")
        _require_unique_ids(
            "review_required_membership_disposition_ids",
            self.review_required_membership_disposition_ids,
        )
        _require_unique_ids(
            "unresolved_membership_disposition_ids",
            self.unresolved_membership_disposition_ids,
        )
        if set(self.accepted_candidate_ids) & set(self.rejected_candidate_ids):
            raise ValueError("accepted and rejected candidate IDs must be disjoint")
        if self.backend_terminal_status is not None:
            ArticulationTerminalStatus.model_validate(self.backend_terminal_status)
            if self.phase != "failed":
                raise ValueError("backend terminal status requires a failed state")
        if self.phase == "not_articulated" and (
            self.inference_result is None
            or self.candidate_document is None
            or self.candidate_ids
            or self.review_required_candidate_ids
            or self.auto_accepted_candidate_ids
            or self.accepted_candidate_ids
            or self.rejected_candidate_ids
            or self.unresolved_candidate_ids
            or self.review_required_membership_disposition_ids
            or self.unresolved_membership_disposition_ids
        ):
            raise ValueError(
                "not_articulated requires bound empty inference and candidate evidence"
            )
        return self

    @model_serializer(mode="wrap")
    def _omit_empty_backend_terminal_status(  # type: ignore[no-untyped-def]
        self,
        handler: SerializerFunctionWrapHandler,
    ):
        """Keep the v1 checkpoint shape stable on Pydantic 2.11."""

        serialized = handler(self)
        if self.schema_version == ARTICULATION_RUN_STATE_SCHEMA_VERSION:
            for field_name in (
                "embedded_evidence",
                "embedded_proposal",
                "embedded_canonical_graph",
                "embedded_graph_revision",
                "embedded_outer_review",
                "embedded_coordinator_decision",
                "embedded_human_decision",
                "embedded_execution_authorization",
                "embedded_readback",
                "embedded_execution_result",
                "embedded_output_evidence",
                "embedded_coordinator_review",
                "embedded_decision_receipt",
                "embedded_terminal_receipt",
            ):
                serialized.pop(field_name, None)
        elif self.schema_version == "content-agent-workflows.articulation-run-state.v2":
            for field_name in (
                "embedded_graph_revision",
                "embedded_outer_review",
                "embedded_output_evidence",
                "embedded_terminal_receipt",
            ):
                serialized.pop(field_name, None)
        elif self.schema_version == "content-agent-workflows.articulation-run-state.v3":
            serialized.pop("embedded_graph_revision", None)
        elif self.schema_version == ARTICULATION_STANDALONE_RUN_STATE_SCHEMA_VERSION:
            for field_name in (
                "embedded_evidence",
                "embedded_proposal",
                "embedded_canonical_graph",
                "embedded_graph_revision",
                "embedded_outer_review",
                "embedded_coordinator_decision",
                "embedded_human_decision",
                "embedded_execution_authorization",
                "embedded_readback",
                "embedded_execution_result",
                "embedded_output_evidence",
                "embedded_coordinator_review",
                "embedded_decision_receipt",
                "embedded_terminal_receipt",
            ):
                serialized.pop(field_name, None)
        elif self.embedded_graph_revision is None:
            serialized.pop("embedded_graph_revision", None)
        if self.schema_version != ARTICULATION_STANDALONE_RUN_STATE_SCHEMA_VERSION:
            for field_name in (
                "standalone_identity",
                "standalone_preparation",
                "standalone_proposal",
                "standalone_decision_patch",
                "standalone_decision_ledger",
                "standalone_canonical_graph",
                "standalone_authoring_receipt",
                "standalone_readback",
                "standalone_output_evidence",
                "standalone_post_review",
                "standalone_cleanup",
                "standalone_terminal_receipt",
                "standalone_refinement_history",
            ):
                serialized.pop(field_name, None)
        if self.backend_terminal_status is None:
            serialized.pop("backend_terminal_status", None)
        return serialized


class ArticulationMotionCapabilityDecision(_StrictModel):
    """Terminal-report view of one typed semantic motion disposition."""

    candidate_id: str = Field(min_length=1)
    semantic_role: str | None = None
    motion_capability: Stage2SemanticMotionCapability
    review_status: Stage2ReviewStatus
    unresolved_reason_codes: tuple[MotionCapabilityDecisionReasonCode, ...] = ()
    unresolved_questions: tuple[str, ...] = ()
    evidence: str = ""


class ArticulationFinalizationResult(_StrictModel):
    """User-facing index of current or terminal articulation artifacts."""

    schema_version: Literal[
        "content-agent-workflows.articulation-finalization-result.v1",
        "content-agent-workflows.articulation-finalization-result.v2",
        "content-agent-workflows.articulation-finalization-result.v3",
        "content-agent-workflows.articulation-finalization-result.v4",
        "content-agent-workflows.articulation-finalization-result.v5",
    ] = ARTICULATION_FINALIZATION_RESULT_SCHEMA_VERSION
    success: bool
    status: ArticulationFinalStatus
    mode: ArticulationWorkflowMode
    output_dir: str
    output_asset_path: str | None = None
    candidate_ids: tuple[str, ...] = ()
    review_required_candidate_ids: tuple[str, ...] = ()
    accepted_candidate_ids: tuple[str, ...] = ()
    rejected_candidate_ids: tuple[str, ...] = ()
    unresolved_candidate_ids: tuple[str, ...] = ()
    motion_capability_decisions: tuple[ArticulationMotionCapabilityDecision, ...] = ()
    membership_dispositions: tuple[MembershipDispositionRecord, ...] = ()
    review_required_membership_disposition_ids: tuple[str, ...] = ()
    unresolved_membership_disposition_ids: tuple[str, ...] = ()
    request_path: str | None = None
    checkpoint_path: str
    inference_result_path: str | None = None
    predictions_path: str | None = None
    report_path: str | None = None
    candidate_document_path: str | None = None
    scene_evidence_path: str | None = None
    review_receipt_path: str | None = None
    approved_candidate_document_path: str | None = None
    authoring_request_path: str | None = None
    authoring_result_path: str | None = None
    diagnostics_path: str | None = None
    joint_rigger_result_path: str | None = None
    validation_result_path: str | None = None
    embedded_evidence_path: str | None = None
    embedded_proposal_path: str | None = None
    embedded_canonical_graph_path: str | None = None
    embedded_graph_revision_path: str | None = None
    embedded_outer_review_path: str | None = None
    embedded_coordinator_decision_path: str | None = None
    embedded_human_decision_path: str | None = None
    embedded_execution_authorization_path: str | None = None
    embedded_readback_path: str | None = None
    embedded_execution_result_path: str | None = None
    embedded_output_evidence_path: str | None = None
    embedded_coordinator_review_path: str | None = None
    embedded_decision_receipt_path: str | None = None
    embedded_terminal_receipt_path: str | None = None
    standalone_identity_path: str | None = None
    standalone_preparation_path: str | None = None
    standalone_proposal_path: str | None = None
    standalone_decision_patch_path: str | None = None
    standalone_decision_ledger_path: str | None = None
    standalone_canonical_graph_path: str | None = None
    standalone_authoring_receipt_path: str | None = None
    standalone_readback_path: str | None = None
    standalone_output_evidence_path: str | None = None
    standalone_post_review_path: str | None = None
    standalone_cleanup_path: str | None = None
    standalone_terminal_receipt_path: str | None = None
    standalone_refinement_history_paths: tuple[str, ...] = ()
    workflow_progress_path: str
    final_summary_path: str
    message: str = Field(min_length=1)
    terminal_status: dict[str, Any] | None = None
    recovery_action: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _validate_status(self) -> Self:
        embedded_paths = (
            self.embedded_evidence_path,
            self.embedded_proposal_path,
            self.embedded_canonical_graph_path,
            self.embedded_graph_revision_path,
            self.embedded_outer_review_path,
            self.embedded_coordinator_decision_path,
            self.embedded_human_decision_path,
            self.embedded_execution_authorization_path,
            self.embedded_readback_path,
            self.embedded_execution_result_path,
            self.embedded_output_evidence_path,
            self.embedded_coordinator_review_path,
            self.embedded_decision_receipt_path,
            self.embedded_terminal_receipt_path,
        )
        standalone_paths = (
            self.standalone_identity_path,
            self.standalone_preparation_path,
            self.standalone_proposal_path,
            self.standalone_decision_patch_path,
            self.standalone_decision_ledger_path,
            self.standalone_canonical_graph_path,
            self.standalone_authoring_receipt_path,
            self.standalone_readback_path,
            self.standalone_output_evidence_path,
            self.standalone_post_review_path,
            self.standalone_cleanup_path,
            self.standalone_terminal_receipt_path,
        )
        if (
            self.schema_version == ARTICULATION_FINALIZATION_RESULT_SCHEMA_VERSION
            and any(path is not None for path in embedded_paths)
        ):
            raise ValueError(
                "embedded articulation paths require embedded finalization"
            )
        if (
            self.schema_version
            == "content-agent-workflows.articulation-finalization-result.v2"
            and any(
                path is not None
                for path in (
                    self.embedded_graph_revision_path,
                    self.embedded_outer_review_path,
                    self.embedded_output_evidence_path,
                    self.embedded_terminal_receipt_path,
                )
            )
        ):
            raise ValueError("embedded output review fields require finalization v3")
        if (
            self.schema_version
            == "content-agent-workflows.articulation-finalization-result.v3"
            and self.embedded_graph_revision_path is not None
        ):
            raise ValueError("embedded graph revision requires finalization v4")
        if (
            self.schema_version
            != ARTICULATION_STANDALONE_FINALIZATION_RESULT_SCHEMA_VERSION
            and (
                any(path is not None for path in standalone_paths)
                or self.standalone_refinement_history_paths
            )
        ):
            raise ValueError(
                "standalone decision custody paths require finalization v5"
            )
        if (
            self.schema_version
            == ARTICULATION_STANDALONE_FINALIZATION_RESULT_SCHEMA_VERSION
            and any(path is not None for path in embedded_paths)
        ):
            raise ValueError(
                "standalone decision custody cannot mix embedded finalization paths"
            )
        if (
            self.schema_version
            == ARTICULATION_STANDALONE_FINALIZATION_RESULT_SCHEMA_VERSION
            and self.standalone_preparation_path is None
        ):
            raise ValueError("standalone finalization requires a preparation path")
        disposition_ids = tuple(
            record.disposition_id for record in self.membership_dispositions
        )
        _require_unique_ids("membership disposition IDs", disposition_ids)
        for name, values in (
            (
                "review_required_membership_disposition_ids",
                self.review_required_membership_disposition_ids,
            ),
            (
                "unresolved_membership_disposition_ids",
                self.unresolved_membership_disposition_ids,
            ),
        ):
            _require_unique_ids(name, values)
            if not set(values).issubset(disposition_ids):
                raise ValueError(f"{name} must reference membership dispositions")
        if self.success != (self.status in {"completed", "not_articulated"}):
            raise ValueError(
                "success is true only for completed or not_articulated workflows"
            )
        if self.status == "completed" and (
            self.unresolved_candidate_ids
            or self.unresolved_membership_disposition_ids
            or not self.output_asset_path
            or not self.authoring_result_path
            or not self.validation_result_path
        ):
            raise ValueError(
                "completed workflow requires authoring and validation evidence, "
                "output, and no unresolved candidates"
            )
        if self.status == "needs_review" and not (
            self.review_required_candidate_ids
            or self.review_required_membership_disposition_ids
        ):
            raise ValueError("needs_review requires candidate or membership review IDs")
        if self.status == "not_articulated" and (
            self.candidate_ids
            or self.review_required_candidate_ids
            or self.accepted_candidate_ids
            or self.rejected_candidate_ids
            or self.unresolved_candidate_ids
            or self.review_required_membership_disposition_ids
            or self.unresolved_membership_disposition_ids
            or self.output_asset_path is not None
            or self.authoring_result_path is not None
            or self.validation_result_path is not None
            or self.inference_result_path is None
            or self.candidate_document_path is None
        ):
            raise ValueError(
                "not_articulated requires empty candidate scope and bound analysis evidence"
            )
        _require_unique_ids(
            "motion capability decision candidate IDs",
            tuple(
                decision.candidate_id for decision in self.motion_capability_decisions
            ),
        )
        if not {
            decision.candidate_id for decision in self.motion_capability_decisions
        }.issubset(self.candidate_ids):
            raise ValueError(
                "motion capability decisions must refer to reported candidate IDs"
            )
        if self.terminal_status is not None:
            if self.status != "failed" or self.success:
                raise ValueError("terminal status requires a failed non-success result")
            if self.recovery_action is None:
                raise ValueError("terminal status requires an exact recovery action")
            terminal_status = ArticulationTerminalStatus.model_validate(
                self.terminal_status
            )
            if self.recovery_action != terminal_status.recovery_action.model_dump(
                mode="json"
            ):
                raise ValueError(
                    "terminal status recovery action must match the final result"
                )
        elif self.recovery_action is not None:
            raise ValueError("recovery action requires a terminal status")
        return self

    @model_serializer(mode="wrap")
    def _omit_empty_motion_capability_decisions(  # type: ignore[no-untyped-def]
        self,
        handler: SerializerFunctionWrapHandler,
    ):
        """Keep legacy terminal summaries stable when no semantic decision exists."""

        serialized = handler(self)
        if self.schema_version == ARTICULATION_FINALIZATION_RESULT_SCHEMA_VERSION:
            for field_name in (
                "embedded_evidence_path",
                "embedded_proposal_path",
                "embedded_canonical_graph_path",
                "embedded_graph_revision_path",
                "embedded_outer_review_path",
                "embedded_coordinator_decision_path",
                "embedded_human_decision_path",
                "embedded_execution_authorization_path",
                "embedded_readback_path",
                "embedded_execution_result_path",
                "embedded_output_evidence_path",
                "embedded_coordinator_review_path",
                "embedded_decision_receipt_path",
                "embedded_terminal_receipt_path",
            ):
                serialized.pop(field_name, None)
        elif (
            self.schema_version
            == "content-agent-workflows.articulation-finalization-result.v2"
        ):
            for field_name in (
                "embedded_graph_revision_path",
                "embedded_outer_review_path",
                "embedded_output_evidence_path",
                "embedded_terminal_receipt_path",
            ):
                serialized.pop(field_name, None)
        elif (
            self.schema_version
            == "content-agent-workflows.articulation-finalization-result.v3"
        ):
            serialized.pop("embedded_graph_revision_path", None)
        elif (
            self.schema_version
            == ARTICULATION_STANDALONE_FINALIZATION_RESULT_SCHEMA_VERSION
        ):
            for field_name in (
                "embedded_evidence_path",
                "embedded_proposal_path",
                "embedded_canonical_graph_path",
                "embedded_graph_revision_path",
                "embedded_outer_review_path",
                "embedded_coordinator_decision_path",
                "embedded_human_decision_path",
                "embedded_execution_authorization_path",
                "embedded_readback_path",
                "embedded_execution_result_path",
                "embedded_output_evidence_path",
                "embedded_coordinator_review_path",
                "embedded_decision_receipt_path",
                "embedded_terminal_receipt_path",
            ):
                serialized.pop(field_name, None)
        elif self.embedded_graph_revision_path is None:
            serialized.pop("embedded_graph_revision_path", None)
        if (
            self.schema_version
            != ARTICULATION_STANDALONE_FINALIZATION_RESULT_SCHEMA_VERSION
        ):
            for field_name in (
                "standalone_identity_path",
                "standalone_preparation_path",
                "standalone_proposal_path",
                "standalone_decision_patch_path",
                "standalone_decision_ledger_path",
                "standalone_canonical_graph_path",
                "standalone_authoring_receipt_path",
                "standalone_readback_path",
                "standalone_output_evidence_path",
                "standalone_post_review_path",
                "standalone_cleanup_path",
                "standalone_terminal_receipt_path",
                "standalone_refinement_history_paths",
            ):
                serialized.pop(field_name, None)
        if not self.motion_capability_decisions:
            serialized.pop("motion_capability_decisions", None)
        if self.terminal_status is None:
            serialized.pop("terminal_status", None)
        if self.recovery_action is None:
            serialized.pop("recovery_action", None)
        return serialized


__all__ = [
    "ARTICULATION_AUTHORING_REQUEST_SCHEMA_VERSION",
    "ARTICULATION_AUTHORING_RESULT_SCHEMA_VERSION",
    "ARTICULATION_EMBEDDED_FINALIZATION_RESULT_SCHEMA_VERSION",
    "ARTICULATION_EMBEDDED_RUN_STATE_SCHEMA_VERSION",
    "ARTICULATION_FINALIZATION_RESULT_SCHEMA_VERSION",
    "ARTICULATION_INFERENCE_SCHEMA_VERSION",
    "ARTICULATION_REQUEST_SCHEMA_VERSION",
    "ARTICULATION_REVIEW_RECEIPT_SCHEMA_VERSION",
    "ARTICULATION_RUN_STATE_SCHEMA_VERSION",
    "ARTICULATION_STANDALONE_FINALIZATION_RESULT_SCHEMA_VERSION",
    "ARTICULATION_STANDALONE_RUN_STATE_SCHEMA_VERSION",
    "ARTICULATION_VALIDATION_SCHEMA_VERSION",
    "ArtifactBinding",
    "ArticulationAuthoringRequest",
    "ArticulationAuthoringResult",
    "ArticulationFinalStatus",
    "ArticulationFinalizationResult",
    "ArticulationInferenceResult",
    "ArticulationMotionCapabilityDecision",
    "MotionCapabilityDecisionReasonCode",
    "ArticulationReviewEntry",
    "ArticulationReviewPolicy",
    "ArticulationV1MotionType",
    "ArticulationReviewReceipt",
    "ArticulationRecoveryAction",
    "ArticulationRunState",
    "ArticulationStateTransition",
    "ArticulationTerminalStatus",
    "ArticulationValidationResult",
    "ArticulationWorkflowMode",
    "ArticulationWorkflowPhase",
    "ArticulationWorkflowRequest",
    "MembershipDispositionDocument",
    "MembershipDispositionRecord",
    "MembershipDispositionSummary",
    "ReviewDecision",
    "STAGE2_SCHEMA_VERSION",
    "SUPPORTED_ARTICULATION_V1_TYPES",
    "Stage2ArticulationCandidate",
    "Stage2CandidateDocument",
    "Stage2SemanticMotionCapability",
    "Stage2CandidateSummary",
    "Stage2EvidenceItem",
    "TERMINAL_ARTICULATION_PHASES",
]
