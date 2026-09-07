# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed physical-membership disposition for the shared Joint domain method."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from joint_agent.functions.stage1_schema import normalize_stage1_prediction_payload

MEMBERSHIP_DISPOSITION_SCHEMA_VERSION: Literal[
    "joint-agent-membership-disposition-v1"
] = "joint-agent-membership-disposition-v1"

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
Confidence = Literal["high", "medium", "low"]

_MOBILE_JOINT_TYPES = frozenset({"revolute", "prismatic"})
_TRUE_VALUES = frozenset({"true", "yes", "y", "1", "candidate", "articulated"})
_FALSE_VALUES = frozenset({"false", "no", "n", "0", "not_candidate", "not_articulated"})
_ACCEPTED_MEMBERSHIP_SOURCES = frozenset(
    {
        "accepted_manifest",
        "authored_metadata",
        "authored_reference",
        "human_reviewed",
        "source_metadata",
    }
)


class MembershipDispositionRecord(BaseModel):
    """One stable, provenance-carrying membership or attachment decision."""

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
    confidence: Confidence
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
            if path is not None and not _is_absolute_prim_path(path):
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
    """Deterministic counts for one membership disposition document."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    disposition_count: int = Field(ge=0)
    disposition_counts: dict[MembershipDisposition, int] = Field(default_factory=dict)
    review_required_count: int = Field(ge=0)
    pending_downstream_count: int = Field(ge=0)


class MembershipDispositionDocument(BaseModel):
    """Order-independent membership output kept separate from Stage 2 v0."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["joint-agent-membership-disposition-v1"] = (
        MEMBERSHIP_DISPOSITION_SCHEMA_VERSION
    )
    summary: MembershipDispositionSummary
    dispositions: tuple[MembershipDispositionRecord, ...]

    @model_validator(mode="after")
    def _validate_summary(self) -> Self:
        ids = tuple(record.disposition_id for record in self.dispositions)
        members = tuple(record.member_prim for record in self.dispositions)
        if len(ids) != len(set(ids)):
            raise ValueError("membership disposition IDs must be unique")
        if len(members) != len(set(members)):
            raise ValueError("membership member prims must be unique")
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


def infer_membership_dispositions(
    predictions: Iterable[dict[str, Any]],
    candidate_document: Mapping[str, Any] | BaseModel,
    *,
    output_key: str = "classification",
    candidate_joint_types: Sequence[str] = ("revolute", "prismatic"),
) -> MembershipDispositionDocument:
    """Classify physical membership without changing Stage 2 or authoring joints."""

    candidate_payload = _model_mapping(candidate_document)
    if candidate_payload.get("schema_version") != "joint-agent-stage2-v0":
        raise ValueError("membership disposition requires joint-agent-stage2-v0")
    candidates = candidate_payload.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("membership disposition requires a Stage 2 candidate list")
    candidate_index = _candidate_index(candidates)
    candidate_joint_type_set = {
        _clean_token(value, "unknown") for value in candidate_joint_types
    }
    if (
        not candidate_joint_type_set
        or not candidate_joint_type_set <= _MOBILE_JOINT_TYPES
    ):
        raise ValueError("membership disposition accepts only revolute/prismatic types")

    normalized_rows: dict[str, dict[str, Any]] = {}
    for row in predictions:
        member_prim = _clean_text(row.get("id"))
        if not member_prim:
            continue
        if member_prim in normalized_rows:
            raise ValueError(f"duplicate membership prediction id: {member_prim}")
        raw_payload = row.get(output_key)
        payload = (
            normalize_stage1_prediction_payload(raw_payload, output_key=output_key)
            if isinstance(raw_payload, dict)
            else {}
        )
        normalized_rows[member_prim] = cast(dict[str, Any], payload)

    records = [
        _disposition_for_row(
            member_prim,
            normalized_rows[member_prim],
            candidate_index=candidate_index,
            candidate_joint_type_set=candidate_joint_type_set,
        )
        for member_prim in sorted(normalized_rows)
    ]
    retained = tuple(record for record in records if record is not None)
    counts = Counter(record.disposition for record in retained)
    return MembershipDispositionDocument(
        summary=MembershipDispositionSummary(
            disposition_count=len(retained),
            disposition_counts=dict(sorted(counts.items())),
            review_required_count=sum(
                record.review_status == "review_required" for record in retained
            ),
            pending_downstream_count=sum(
                record.disposition in {"explicit_fixed", "unresolved"}
                for record in retained
            ),
        ),
        dispositions=retained,
    )


class _CandidateIndex:
    def __init__(self) -> None:
        self.primary_by_source: dict[str, str] = {}
        self.sources_by_primary: dict[str, tuple[str, ...]] = {}


def _candidate_index(candidates: list[Any]) -> _CandidateIndex:
    index = _CandidateIndex()
    for value in candidates:
        if not isinstance(value, Mapping):
            raise ValueError("Stage 2 candidates must be mappings")
        moving = value.get("moving_part_prims")
        sources = value.get("source_prediction_ids")
        if not isinstance(moving, list) or not moving:
            continue
        primary = _clean_text(moving[0])
        if not primary:
            continue
        source_ids = (
            tuple(
                sorted(
                    {
                        source
                        for raw_source in sources
                        if (source := _clean_text(raw_source))
                    }
                )
            )
            if isinstance(sources, list)
            else ()
        )
        index.sources_by_primary[primary] = source_ids
        for source in source_ids:
            prior = index.primary_by_source.setdefault(source, primary)
            if prior != primary:
                raise ValueError(
                    f"prediction {source} belongs to multiple Stage 2 candidates"
                )
    return index


def _disposition_for_row(
    member_prim: str,
    payload: dict[str, Any],
    *,
    candidate_index: _CandidateIndex,
    candidate_joint_type_set: set[str],
) -> MembershipDispositionRecord | None:
    candidate_primary = candidate_index.primary_by_source.get(member_prim)
    flag = _candidate_flag(payload.get("is_articulation_candidate"))
    joint_type = _clean_token(payload.get("joint_type_hint"), "unknown")
    membership = payload.get("membership")

    if isinstance(membership, Mapping):
        disposition = _disposition(membership.get("disposition"))
        source = _source(membership.get("source"))
        confidence = _confidence(membership.get("confidence"))
        rationale = _clean_text(membership.get("rationale"))
        owner = _optional_prim_path(membership.get("physical_owner_prim"))
        parent = _optional_prim_path(membership.get("attachment_parent_prim"))
        child = _optional_prim_path(membership.get("attachment_child_prim"))
        reasons: list[MembershipReasonCode] = []
        if disposition == "independent_motion":
            if owner is None:
                reasons.append("physical_owner_unresolved")
            if (
                (flag is not True and source not in _ACCEPTED_MEMBERSHIP_SOURCES)
                or joint_type not in candidate_joint_type_set
                or candidate_primary is None
                or owner not in {candidate_primary, member_prim}
            ):
                reasons.append("membership_evidence_conflict")
        elif disposition == "co_rigid":
            if owner is None or owner == member_prim:
                reasons.append("physical_owner_unresolved")
            if flag is True and source not in _ACCEPTED_MEMBERSHIP_SOURCES:
                reasons.append("membership_evidence_conflict")
        elif disposition == "explicit_fixed":
            if owner is None:
                reasons.append("physical_owner_unresolved")
            if parent is None or child is None or parent == child or child != owner:
                reasons.append("membership_evidence_conflict")
            if flag is True and source not in _ACCEPTED_MEMBERSHIP_SOURCES:
                reasons.append("membership_evidence_conflict")
        else:
            reasons.append("membership_evidence_missing")
    elif candidate_primary is not None and member_prim != candidate_primary:
        disposition = "co_rigid"
        source = "predicted"
        confidence = _confidence(payload.get("confidence"))
        rationale = "Stage 2 grouped this prediction under one physical owner."
        owner = candidate_primary
        parent = None
        child = None
        reasons = []
    elif flag is True and joint_type in candidate_joint_type_set:
        disposition = "independent_motion"
        source = "predicted"
        confidence = _confidence(payload.get("confidence"))
        rationale = _clean_text(payload.get("reasoning"))
        owner = candidate_primary or member_prim
        parent = None
        child = None
        reasons = (
            [] if candidate_primary is not None else ["membership_evidence_conflict"]
        )
    elif flag is False and joint_type in candidate_joint_type_set:
        disposition = "unresolved"
        source = "predicted"
        confidence = _confidence(payload.get("confidence"))
        rationale = _clean_text(payload.get("reasoning"))
        owner = None
        parent = None
        child = None
        reasons = ["membership_evidence_missing"]
    else:
        return None

    reasons = list(dict.fromkeys(reasons))
    if reasons and disposition != "explicit_fixed":
        disposition = "unresolved"
        parent = None
        child = None
    elif (
        reasons
        and disposition == "explicit_fixed"
        and any(reason != "fixed_authoring_evidence_required" for reason in reasons)
    ):
        disposition = "unresolved"
        parent = None
        child = None
    if disposition == "explicit_fixed" and not reasons:
        reasons.append("fixed_authoring_evidence_required")

    owner_candidate = (
        candidate_index.primary_by_source.get(owner)
        or (owner if owner in candidate_index.sources_by_primary else None)
        if owner is not None
        else None
    )
    boundary = cast(
        MembershipBoundary,
        {
            "co_rigid": "co_rigid_aggregation",
            "explicit_fixed": "fixed_authoring_v2",
            "independent_motion": "moving_candidate_generation",
            "unresolved": "human_review",
        }[disposition],
    )
    review_status: MembershipReviewStatus = (
        "review_required"
        if disposition == "unresolved"
        or (
            disposition == "explicit_fixed"
            and source not in _ACCEPTED_MEMBERSHIP_SOURCES
        )
        else "resolved"
    )
    source_ids = (
        candidate_index.sources_by_primary.get(candidate_primary, ())
        if candidate_primary is not None
        else (member_prim,)
    )
    return MembershipDispositionRecord(
        disposition_id=_disposition_id(member_prim),
        member_prim=member_prim,
        motion_candidate_prim=candidate_primary,
        physical_owner_prim=owner,
        physical_owner_candidate_prim=owner_candidate,
        disposition=disposition,
        attachment_parent_prim=parent,
        attachment_child_prim=child,
        source=source,
        confidence=confidence,
        rationale=rationale,
        source_prediction_ids=source_ids,
        downstream_boundary=boundary,
        review_status=review_status,
        reason_codes=tuple(reasons),
        unresolved_questions=tuple(_question(reason) for reason in reasons),
    )


def _model_mapping(value: Mapping[str, Any] | BaseModel) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return dict(value)


def _disposition_id(member_prim: str) -> str:
    digest = hashlib.sha256(member_prim.encode("utf-8")).hexdigest()[:16]
    return f"membership_{digest}"


def _is_absolute_prim_path(value: str) -> bool:
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


def _optional_prim_path(value: Any) -> str | None:
    path = _clean_text(value)
    return path if _is_absolute_prim_path(path) else None


def _candidate_flag(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return bool(value)
    token = _clean_token(value, "")
    if token in _TRUE_VALUES:
        return True
    if token in _FALSE_VALUES:
        return False
    return None


def _disposition(value: Any) -> MembershipDisposition:
    token = _clean_token(value, "unresolved")
    token = {
        "corigid": "co_rigid",
        "same_link": "co_rigid",
        "fixed": "explicit_fixed",
        "fixed_edge": "explicit_fixed",
        "independent": "independent_motion",
        "moving_link": "independent_motion",
        "conflicting": "unresolved",
        "unknown": "unresolved",
    }.get(token, token)
    if token not in {
        "co_rigid",
        "explicit_fixed",
        "independent_motion",
        "unresolved",
    }:
        return "unresolved"
    return cast(MembershipDisposition, token)


def _source(value: Any) -> MembershipEvidenceSource:
    token = _clean_token(value, "predicted")
    token = {"llm": "predicted", "model": "predicted", "vlm": "predicted"}.get(
        token, token
    )
    if token not in {
        "predicted",
        "llm_adjudicated",
        "authored_metadata",
        "authored_reference",
        "source_metadata",
        "accepted_manifest",
        "human_reviewed",
        "unknown",
    }:
        return "unknown"
    return cast(MembershipEvidenceSource, token)


def _confidence(value: Any) -> Confidence:
    token = _clean_token(value, "low")
    return cast(Confidence, token if token in {"high", "medium", "low"} else "low")


def _clean_token(value: Any, default: str) -> str:
    return _clean_text(value, default).lower().replace("-", "_").replace(" ", "_")


def _clean_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _question(reason: MembershipReasonCode) -> str:
    return {
        "fixed_authoring_evidence_required": (
            "Provide accepted attachment-frame evidence through the separate "
            "fixed-authoring capability."
        ),
        "membership_evidence_conflict": (
            "Resolve the conflicting physical membership evidence."
        ),
        "membership_evidence_missing": (
            "Select co-rigid membership, an explicit fixed attachment, or "
            "independent motion."
        ),
        "physical_owner_unresolved": "Resolve the stable physical owner identity.",
    }[reason]


def membership_disposition_json_schema() -> dict[str, Any]:
    """Return the schema of the standalone membership artifact."""

    return MembershipDispositionDocument.model_json_schema()


def membership_disposition_sha256(document: MembershipDispositionDocument) -> str:
    """Return a canonical digest for durable coordinator bindings."""

    payload = json.dumps(
        document.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "MEMBERSHIP_DISPOSITION_SCHEMA_VERSION",
    "MembershipDispositionDocument",
    "MembershipDispositionRecord",
    "infer_membership_dispositions",
    "membership_disposition_json_schema",
    "membership_disposition_sha256",
]
