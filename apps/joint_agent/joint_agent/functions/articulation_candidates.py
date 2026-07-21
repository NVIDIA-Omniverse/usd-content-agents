# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Report-only Stage 2 articulation candidate inference."""

from __future__ import annotations

import copy
import html
import json
import math
import re
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, NamedTuple, Self, cast, get_args

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)

from joint_agent.functions.articulation_types import ArticulationReviewStatus
from joint_agent.functions.axis_hints import normalize_axis_hint_token
from joint_agent.functions.consistency import (
    canonical_link_instance_id,
    is_model_supplied_link_instance_id,
)
from joint_agent.functions.stage1_schema import unwrap_stage1_prediction_payload

STAGE2_SCHEMA_VERSION: Literal["joint-agent-stage2-v0"] = "joint-agent-stage2-v0"
DEFAULT_CANDIDATE_JOINT_TYPES = ("revolute", "prismatic", "spherical")
Stage2MotionType = Literal["revolute", "prismatic", "spherical", "fixed", "unknown"]
Stage2Confidence = Literal["high", "medium", "low"]
# ``structural_fallback`` is accepted for legacy v0 artifacts. Current Stage 2
# inference does not emit it.
Stage2ParentResolutionSource = Literal[
    "stage1_hint",
    "stage1_rigger_evidence",
    "structural_fallback",
    "unresolved",
]
# ``structural_fallback`` and ``geometry_inferred`` are accepted for legacy v0
# artifacts. Current Stage 2 inference does not emit them.
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
_STAGE2_FIELD_SOURCES = set(get_args(Stage2FieldSource))
_MODEL_PREDICTED_STAGE2_SOURCES = frozenset({"predicted", "llm_adjudicated"})
Stage2ConnectivityEvidenceRole = Literal[
    "body0_body1_edge",
    "body1_ownership",
    "endpoint_canonicalization",
]
type Stage2ReviewStatus = ArticulationReviewStatus
READY_FOR_RIGGER_INPUT_STATUS: Stage2ReviewStatus = "ready_for_rigger_input"
REVIEW_REQUIRED_STATUS: Stage2ReviewStatus = "review_required"
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
# ``axis_geometry_ambiguous`` is accepted for legacy v0 artifacts only. Current
# Stage 2 inference does not emit geometry-derived axis reason codes.
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

_UNKNOWN_VALUES = {"", "unknown", "none", "null", "n/a", "na"}
_TRUE_VALUES = {"true", "yes", "y", "1", "candidate", "articulated"}
_FALSE_VALUES = {"false", "no", "n", "0", "none", "fixed", "static"}
_NON_LABEL_RE = re.compile(r"[^a-z0-9]+")
_AXIS_HINT_TO_WORLD: dict[str, tuple[float, float, float]] = {
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
_ALLOWED_AXIS_SET = frozenset(_AXIS_HINT_TO_WORLD)
_EXPLICIT_INSTANCE_INTERNAL_FIELDS = frozenset(
    {
        "_explicit_instance_link_conflict",
        "_explicit_instance_link_id",
        "_explicit_instance_link_members",
    }
)


class Stage2EvidenceItem(BaseModel):
    """Structured diagnostic evidence for one inferred Stage 2 field."""

    model_config = ConfigDict(extra="forbid")

    source: Stage2FieldSource = "unknown"
    description: str
    value: str | None = None
    prim_paths: list[str] = Field(default_factory=list)
    # Optional only so saved ``joint-agent-stage2-v0`` artifacts remain
    # readable. Current positive connectivity proofs always set this field;
    # axis, limit, conflict, and incomplete diagnostics omit it.
    connectivity_role: Stage2ConnectivityEvidenceRole | None = None

    # A runtime return annotation makes Pydantic replace this model's
    # serialization schema with the serializer's generic dict schema. Keep the
    # return type intentionally unannotated and confine the mypy exception here.
    @model_serializer(mode="wrap")
    def _omit_empty_connectivity_role(  # type: ignore[no-untyped-def]
        self,
        handler: SerializerFunctionWrapHandler,
    ):
        """Keep legacy role-less v0 JSON stable on every supported Pydantic."""
        serialized = cast(dict[str, Any], handler(self))
        if self.connectivity_role is None:
            serialized.pop("connectivity_role", None)
        return serialized


class Stage2ArticulationCandidate(BaseModel):
    """Minimal diagnostic Stage 2 joint-candidate edge contract."""

    model_config = ConfigDict(extra="allow")

    schema_version: Literal["joint-agent-stage2-v0"] = STAGE2_SCHEMA_VERSION
    candidate_id: str
    motion_type: Stage2MotionType = "unknown"
    moving_part_prims: list[str] = Field(default_factory=list)
    fixed_parent_prim: str | None = None
    parent_resolution_source: Stage2ParentResolutionSource = "unresolved"
    joint_type_hint: str = "unknown"
    axis_hint: str = "unknown"
    motion_axis_world: list[float] | None = Field(
        default=None,
        min_length=3,
        max_length=3,
    )
    confidence: Stage2Confidence = "low"
    parent_hint: str = "unknown"
    child_hint: str = "unknown"
    component_name: str = "unknown"
    component_type: str = "unknown"
    role: str = "unknown"
    source_prediction_ids: list[str] = Field(default_factory=list)
    evidence: str = ""
    source_annotation_conflicts: dict[str, list[str]] = Field(default_factory=dict)
    field_sources: dict[str, Stage2FieldSource] = Field(default_factory=dict)
    axis_evidence: list[Stage2EvidenceItem] = Field(default_factory=list)
    connectivity_evidence: list[Stage2EvidenceItem] = Field(default_factory=list)
    lower_limit: float | None = None
    upper_limit: float | None = None
    limit_unit: str = "unknown"
    limit_source: Stage2FieldSource = "unknown"
    limit_readiness: Stage2LimitReadiness = "not_provided"
    limit_evidence: list[Stage2EvidenceItem] = Field(default_factory=list)
    unresolved_reason_codes: list[Stage2UnresolvedReasonCode] = Field(
        default_factory=list
    )
    review_status: Stage2ReviewStatus = "review_required"
    unresolved_questions: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_cross_field_invariants(self) -> Self:
        if (
            self.motion_axis_world is not None
            and self.axis_hint not in _ALLOWED_AXIS_SET
        ):
            raise ValueError(
                "motion_axis_world requires axis_hint to be an explicit "
                "axis-aligned token"
            )
        # ``structural_fallback`` is accepted here only so legacy v0 artifacts
        # still validate; current inference does not emit it.
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
                "fixed_parent_prim must be set when parent_resolution_source is "
                "stage1_hint, stage1_rigger_evidence, or structural_fallback"
            )
        return self


def load_predictions_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load prediction rows from JSONL."""
    predictions: list[dict[str, Any]] = []
    predictions_path = Path(path)
    with predictions_path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if stripped:
                try:
                    predictions.append(json.loads(stripped))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in {predictions_path} at line {line_number}: "
                        f"{exc.msg}"
                    ) from exc
    return predictions


def load_prim_metadata_jsonl(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load legacy optional prim metadata rows keyed by prim path.

    Deprecated for Stage 2 inference: current articulation candidate inference
    accepts metadata plumbing for compatibility but does not consume it.
    """
    return _normalize_prim_metadata_index(load_predictions_jsonl(path))


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Atomically write JSON payload."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as f:
            tmp_path = Path(f.name)
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.write("\n")
        tmp_path.replace(output_path)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()


def write_articulation_candidate_report_html(
    path: str | Path,
    candidate_document: dict[str, Any],
) -> None:
    """Write a compact HTML review report for Stage 2 candidates."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    candidates = candidate_document.get("candidates", [])
    summary = candidate_document.get("summary", {})

    rows = []
    for candidate in candidates:
        questions = candidate.get("unresolved_questions", [])
        reason_codes = candidate.get("unresolved_reason_codes", [])
        rows.append(
            "<tr>"
            f"<td>{_e(candidate.get('candidate_id'))}</td>"
            f"<td>{_e(candidate.get('review_status', 'review_required'))}</td>"
            f"<td>{_e(candidate.get('joint_type_hint'))}</td>"
            f"<td>{_e(candidate.get('axis_hint'))}</td>"
            f"<td>{_e(candidate.get('confidence'))}</td>"
            f"<td>{_e(candidate.get('limit_readiness', 'not_provided'))}</td>"
            f"<td>{_e(_format_limits(candidate))}</td>"
            f"<td>{_e(', '.join(candidate.get('moving_part_prims', [])))}</td>"
            f"<td>{_e(candidate.get('fixed_parent_prim') or 'unresolved')}</td>"
            f"<td>{_e(candidate.get('parent_resolution_source', 'unresolved'))}</td>"
            f"<td>{_e(', '.join(reason_codes) if reason_codes else 'none')}</td>"
            f"<td>{_e(_format_evidence(candidate.get('axis_evidence', [])))}</td>"
            f"<td>{_e(_format_evidence(candidate.get('connectivity_evidence', [])))}</td>"
            f"<td>{_e(_format_evidence(candidate.get('limit_evidence', [])))}</td>"
            f"<td>{_e(_format_evidence(candidate.get('adjudication_evidence', [])))}</td>"
            f"<td>{_e('; '.join(questions) if questions else 'none')}</td>"
            f"<td>{_e(_format_annotation_conflicts(candidate))}</td>"
            f"<td>{_e(candidate.get('evidence'))}</td>"
            "</tr>"
        )

    body = "\n".join(rows) or (
        '<tr><td colspan="18">No articulation candidates found.</td></tr>'
    )
    joint_counts = summary.get("joint_type_counts", {})
    review_status_counts = summary.get("review_status_counts", {})
    limit_readiness_counts = summary.get("limit_readiness_counts", {})
    reason_code_counts = summary.get("reason_code_counts", {})
    count_bits = ", ".join(
        f"{_e(joint_type)}: {_e(count)}"
        for joint_type, count in sorted(joint_counts.items())
    )
    status_bits = ", ".join(
        f"{_e(status)}: {_e(count)}"
        for status, count in sorted(review_status_counts.items())
    )
    limit_bits = ", ".join(
        f"{_e(status)}: {_e(count)}"
        for status, count in sorted(limit_readiness_counts.items())
    )
    reason_bits = ", ".join(
        f"{_e(code)}: {_e(count)}" for code, count in sorted(reason_code_counts.items())
    )
    source_structure_bits = ", ".join(
        str(value) for value in summary.get("source_structure_diagnostics", [])
    )
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Joint Agent Articulation Candidates</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; table-layout: fixed; }}
    th, td {{ border: 1px solid #c9d1d9; padding: 8px; vertical-align: top; }}
    th {{ background: #eef2f7; text-align: left; }}
    td {{ word-wrap: break-word; }}
    .summary {{ margin-bottom: 16px; }}
    .meta {{ color: #52606d; }}
  </style>
</head>
<body>
  <h1>Articulation Candidates</h1>
  <div class="summary">
    <div>Total predictions: {_e(summary.get("total_predictions", 0))}</div>
    <div>Candidates: {_e(summary.get("candidate_count", 0))}</div>
    <div>Ready for rigger input: {_e(summary.get("ready_candidate_count", 0))}</div>
    <div>Needs review: {_e(summary.get("review_required_candidate_count", 0))}</div>
    <div>Missing axis hints: {_e(summary.get("unresolved_axis_count", 0))}</div>
    <div>Missing parent links: {_e(summary.get("unresolved_parent_count", 0))}</div>
    <div>Joint types: {_e(count_bits or "none")}</div>
    <div>Review status: {_e(status_bits or "none")}</div>
    <div>Limit readiness: {_e(limit_bits or "none")}</div>
    <div>Reason codes: {_e(reason_bits or "none")}</div>
    <div>Source structure diagnostics: {_e(source_structure_bits or "none")}</div>
  </div>
  <div class="meta">Schema: {_e(candidate_document.get("schema_version"))}</div>
  <table>
    <thead>
      <tr>
        <th>Candidate</th>
        <th>Status</th>
        <th>Joint Type</th>
        <th>Axis</th>
        <th>Confidence</th>
        <th>Limit Status</th>
        <th>Limits</th>
        <th>Moving Prims</th>
        <th>Fixed Parent</th>
        <th>Parent Source</th>
        <th>Reason Codes</th>
        <th>Axis Evidence</th>
        <th>Connectivity Evidence</th>
        <th>Limit Evidence</th>
        <th>Adjudication Evidence</th>
        <th>Review Questions</th>
        <th>Source Annotation Conflicts</th>
        <th>Evidence</th>
      </tr>
    </thead>
    <tbody>
      {body}
    </tbody>
  </table>
</body>
</html>
"""
    output_path.write_text(html_text, encoding="utf-8")


def infer_articulation_candidates(
    predictions: Iterable[dict[str, Any]],
    *,
    output_key: str = "classification",
    candidate_joint_types: Sequence[str] | None = None,
    prim_metadata: Iterable[dict[str, Any]]
    | Mapping[str, dict[str, Any]]
    | None = None,
) -> dict[str, Any]:
    """Infer report-only articulation candidates from Stage 1 predictions.

    Source prim metadata may validate exact authored rigid-body endpoint paths or
    an explicitly model-selected ancestor Xform from a bounded source hierarchy.
    Axis and parent/connectivity truth still must come from explicit Stage 1
    LLM/VLM fields; bbox, extent, center, names, and path structure never supply
    joint truth.
    """
    prediction_rows = list(predictions)
    candidate_joint_type_values = (
        DEFAULT_CANDIDATE_JOINT_TYPES
        if candidate_joint_types is None
        else candidate_joint_types
    )
    candidate_joint_type_set = {
        _clean_token(value) for value in candidate_joint_type_values
    }
    source_structure_index = _source_structure_index(prim_metadata)
    raw_payloads = [
        _PredictionPayload(
            prim_path=str(row.get("id", "")),
            payload=_classification_payload(row, output_key) or {},
        )
        for row in prediction_rows
    ]
    source_structure_index = _with_observable_hierarchy_leaf_aliases(
        source_structure_index,
        raw_payloads,
    )
    canonical_payloads = _canonicalize_transparent_hierarchy_endpoint_claims(
        raw_payloads,
        source_structure_index=source_structure_index,
    )
    canonical_payloads = _collapse_explicit_instance_link_members(
        canonical_payloads,
        candidate_joint_type_set=candidate_joint_type_set,
    )
    explicit_moving_link_representatives = _explicit_instance_link_representatives(
        canonical_payloads,
        candidate_joint_type_set=candidate_joint_type_set,
        source_structure_index=source_structure_index,
    )
    payloads = _synthesize_source_backed_body_payloads(
        canonical_payloads,
        source_structure_index=source_structure_index,
        candidate_joint_type_set=candidate_joint_type_set,
    )
    fixed_support_prediction_paths = _explicit_fixed_support_prediction_paths(payloads)
    prediction_prim_paths = {row.prim_path for row in raw_payloads if row.prim_path}
    _, known_prim_paths = _build_parent_index(
        payloads,
        additional_known_prim_paths={
            *source_structure_index.endpoint_paths,
            *source_structure_index.hierarchy_endpoint_paths,
        },
    )
    valid_endpoint_paths = (
        set(source_structure_index.endpoint_paths)
        if source_structure_index.endpoint_paths
        else known_prim_paths
    )
    payload_by_prim = {row.prim_path: row.payload for row in payloads if row.prim_path}
    compound_edge_collection = _collect_compound_edge_evidence(
        payloads=payloads,
        known_prim_paths=valid_endpoint_paths,
        prediction_prim_paths=prediction_prim_paths,
        source_structure_index=source_structure_index,
        candidate_joint_type_set=candidate_joint_type_set,
    )
    compound_edges_by_body1 = compound_edge_collection.edges_by_body1
    compound_edge_conflicts_by_body1 = (
        compound_edge_collection.conflict_evidence_by_body1
    )
    compound_edge_conflict_sources_by_body1 = (
        compound_edge_collection.conflict_source_prediction_ids_by_body1
    )
    compound_edge_conflict_candidate_body1_paths = (
        compound_edge_collection.conflict_candidate_body1_paths
    )
    candidates: list[dict[str, Any]] = []
    unresolved_axis_count = 0
    unresolved_parent_count = 0

    for payload_row in payloads:
        payload = payload_row.payload
        joint_type = _clean_token(payload.get("joint_type_hint"), "unknown")
        candidate_flag = _articulation_candidate_flag_state(
            payload.get("is_articulation_candidate")
        )
        joint_type_promotes = joint_type in candidate_joint_type_set
        source_support_joint_type_conflict = bool(
            payload.get("_source_support_joint_type_conflict")
        )
        if (
            candidate_flag is not True
            and not joint_type_promotes
            and not source_support_joint_type_conflict
        ):
            continue

        candidate_id = f"candidate_{len(candidates) + 1:04d}"
        source_owner_prim = source_structure_index.owner_by_prim_path.get(
            payload_row.source_prediction_ids[0]
            if payload_row.source_prediction_ids
            else payload_row.prim_path
        )
        source_prediction_paths = (
            payload_row.source_prediction_ids
            if payload_row.source_prediction_ids
            else [payload_row.prim_path]
        )
        candidate_endpoint_paths = _candidate_endpoint_vocabulary(
            source_structure_index=source_structure_index,
            source_prediction_paths=source_prediction_paths,
            prediction_prim_paths=prediction_prim_paths,
            legacy_known_prim_paths=valid_endpoint_paths,
        )
        body1_resolution = _resolve_rigger_body1_prim(
            payload=payload,
            valid_endpoint_paths=candidate_endpoint_paths,
            moving_prim_path=payload_row.prim_path,
            source_owner_prim=source_owner_prim,
            rigid_endpoint_vocabulary_present=bool(
                source_structure_index.endpoint_paths
            ),
            hierarchy_ancestor_paths=source_structure_index.hierarchy_ancestors_for(
                source_prediction_paths
            ),
        )
        moving_body_prim = body1_resolution.body1_prim or payload_row.prim_path
        direct_body0_endpoint_paths = _candidate_direct_body0_vocabulary(
            source_structure_index=source_structure_index,
            source_prediction_paths=source_prediction_paths,
            candidate_endpoint_paths=candidate_endpoint_paths,
            fixed_support_prediction_paths=fixed_support_prediction_paths,
            allow_fixed_support_gprims=(
                body1_resolution.evidence_present
                and body1_resolution.resolved
                and moving_body_prim == payload_row.prim_path
            ),
            explicit_moving_body0_paths=(
                _model_supplied_explicit_moving_body0_paths(
                    payload=payload,
                    moving_body_prim=moving_body_prim,
                    source_prediction_paths=source_prediction_paths,
                    source_structure_index=source_structure_index,
                    explicit_moving_link_representatives=(
                        explicit_moving_link_representatives
                    ),
                )
            ),
        )
        compound_edges_for_body1 = compound_edges_by_body1.get(moving_body_prim, ())
        compound_edge = _single_compound_edge_for_body1(
            moving_body_prim,
            compound_edges_by_body1,
        )
        raw_axis_hint, axis_source = _stage2_axis_hint_and_source(payload)
        compound_edge_conflict_evidence = list(
            compound_edge_conflicts_by_body1.get(moving_body_prim, ())
        )
        compound_edge_conflict_source_ids = list(
            compound_edge_conflict_sources_by_body1.get(moving_body_prim, ())
        )
        compound_axis_evidence: Stage2EvidenceItem | None = None
        parent_hint = _clean_text(payload.get("parent_hint"), "unknown")
        parent_resolution = _resolve_fixed_parent_prim(
            payload=payload,
            parent_hint=parent_hint,
            known_prim_paths=candidate_endpoint_paths,
            rigger_body0_paths=direct_body0_endpoint_paths,
            moving_prim_path=moving_body_prim,
        )
        fixed_parent_prim = parent_resolution.fixed_parent_prim
        parent_resolution_source = parent_resolution.source
        parent_resolved_to_self = parent_resolution.parent_resolved_to_self
        if len(compound_edges_for_body1) > 1:
            compound_edge = None
            compound_edge_conflict_evidence.append(
                _compound_edge_ambiguity_evidence(
                    moving_prim_path=moving_body_prim,
                    compound_edges=compound_edges_for_body1,
                )
            )
            compound_edge_conflict_source_ids.extend(
                source_prediction_id
                for edge in compound_edges_for_body1
                for source_prediction_id in edge.source_prediction_ids
            )
            shared_compound_axis = _compound_edges_shared_axis_hint(
                compound_edges_for_body1
            )
            if raw_axis_hint in _UNKNOWN_VALUES and shared_compound_axis is not None:
                raw_axis_hint = shared_compound_axis
                axis_source = "stage1_rigger_evidence"
                compound_axis_evidence = _compound_edges_axis_evidence(
                    moving_prim_path=moving_body_prim,
                    compound_edges=compound_edges_for_body1,
                    axis_hint=shared_compound_axis,
                )
        elif compound_edge is not None and not _compound_edge_matches_candidate(
            compound_edge,
            joint_type=joint_type,
            axis_hint=raw_axis_hint,
        ):
            compound_edge_conflict_evidence.append(
                _compound_edge_conflict_evidence(
                    compound_edge,
                    joint_type=joint_type,
                    axis_hint=raw_axis_hint,
                )
            )
            compound_edge_conflict_source_ids.extend(
                compound_edge.source_prediction_ids
            )
            compound_edge = None
        if (
            compound_edge is not None
            and body1_resolution.evidence_present
            and not body1_resolution.resolved
        ):
            compound_edge = None
        if (
            compound_edge is not None
            and fixed_parent_prim is not None
            and fixed_parent_prim != compound_edge.body0
        ):
            compound_edge_conflict_evidence.append(
                _compound_edge_parent_conflict_evidence(
                    compound_edge,
                    fixed_parent_prim=fixed_parent_prim,
                )
            )
            compound_edge_conflict_source_ids.extend(
                compound_edge.source_prediction_ids
            )
            compound_edge = None
        if (
            raw_axis_hint in _UNKNOWN_VALUES
            and compound_edge is not None
            and compound_edge.axis_hint not in _UNKNOWN_VALUES
        ):
            raw_axis_hint = compound_edge.axis_hint
            axis_source = compound_edge.source
            compound_axis_evidence = _compound_edge_axis_evidence(compound_edge)
        axis_hint, normalized_axis_evidence = _normalize_stage2_axis_hint(
            raw_axis_hint,
            moving_prim_path=moving_body_prim,
            axis_source=axis_source,
        )
        support_axis_conflict_evidence: Stage2EvidenceItem | None = None
        if payload.get("_source_support_axis_conflict"):
            support_axis_conflict_evidence = Stage2EvidenceItem(
                source="consistency_corrected",
                description=(
                    "Explicit Stage 1 rows for the same source-backed body edge "
                    "reported conflicting motion axes."
                ),
                value=", ".join(
                    str(value)
                    for value in payload.get("_source_support_axis_values", [])
                ),
                prim_paths=[moving_body_prim],
            )
        if (
            compound_edge_conflict_evidence
            and not parent_resolution.evidence_present
            and len(compound_edges_for_body1) > 1
        ):
            parent_resolution = _ParentResolution(
                fixed_parent_prim=None,
                source="unresolved",
                parent_resolved_to_self=False,
                evidence=compound_edge_conflict_evidence,
                evidence_present=True,
            )
            fixed_parent_prim = parent_resolution.fixed_parent_prim
            parent_resolution_source = parent_resolution.source
            parent_resolved_to_self = parent_resolution.parent_resolved_to_self
        if (
            compound_edge is not None
            and not parent_resolution.evidence_present
            and fixed_parent_prim is None
        ):
            parent_resolution = _ParentResolution(
                fixed_parent_prim=compound_edge.body0,
                source="stage1_rigger_evidence",
                parent_resolved_to_self=False,
                evidence=[_compound_edge_parent_evidence(compound_edge)],
                evidence_present=True,
            )
            fixed_parent_prim = parent_resolution.fixed_parent_prim
            parent_resolution_source = parent_resolution.source
            parent_resolved_to_self = parent_resolution.parent_resolved_to_self
        if compound_edge is not None and not body1_resolution.evidence_present:
            body1_resolution = _Body1Resolution(
                evidence=[_compound_edge_body1_evidence(compound_edge)],
                evidence_present=True,
                resolved=True,
            )

        motion_axis_world = _motion_axis_world_from_axis_hint(axis_hint)
        inferred_axis_evidence: Stage2EvidenceItem | None = (
            normalized_axis_evidence or compound_axis_evidence
        )

        unresolved_reason_codes = _candidate_unresolved_reason_codes(
            candidate_flag=candidate_flag,
            joint_type=joint_type,
            joint_type_promotes=joint_type_promotes,
            axis_hint=axis_hint,
            motion_axis_world=motion_axis_world,
            fixed_parent_prim=fixed_parent_prim,
            parent_resolved_to_self=parent_resolved_to_self,
            body1_evidence_present=body1_resolution.evidence_present,
            body1_resolved=body1_resolution.resolved,
            candidate_joint_type_set=candidate_joint_type_set,
            compound_edge_conflict=bool(compound_edge_conflict_evidence),
        )
        if support_axis_conflict_evidence is not None:
            unresolved_reason_codes.insert(0, "axis_evidence_conflict")
        if (
            payload.get("_source_support_candidate_flag_conflict")
            and "candidate_flag_conflict" not in unresolved_reason_codes
        ):
            unresolved_reason_codes.insert(0, "candidate_flag_conflict")
        if (
            source_support_joint_type_conflict
            and "joint_type_conflict" not in unresolved_reason_codes
        ):
            unresolved_reason_codes.insert(0, "joint_type_conflict")
        if payload.get("_explicit_instance_link_conflict"):
            unresolved_reason_codes.insert(0, "link_membership_conflict")
        unresolved_questions: list[str] = []

        if "axis_evidence_conflict" in unresolved_reason_codes:
            unresolved_questions.append(
                "Resolve the conflicting motion-axis evidence for this body edge."
            )
        if "compound_edge_conflict" in unresolved_reason_codes:
            unresolved_questions.append("Resolve the compound-edge evidence conflict.")
        if "joint_type_conflict" in unresolved_reason_codes:
            unresolved_questions.append(
                "Resolve the conflicting joint-type evidence for this body edge."
                if source_support_joint_type_conflict
                else (
                    "Confirm joint type because the candidate flag and joint hint "
                    "disagree."
                )
            )
        if "candidate_flag_conflict" in unresolved_reason_codes:
            unresolved_questions.append(
                "Confirm articulation candidate because the candidate flag and joint "
                "hint disagree."
            )
        if "link_membership_conflict" in unresolved_reason_codes:
            unresolved_questions.append(
                "Resolve the conflicting physical-link membership evidence before "
                "aggregating this candidate."
            )
        if "body1_unresolved" in unresolved_reason_codes:
            unresolved_questions.append("Resolve the moving body/body1 evidence.")
        if motion_axis_world is None:
            unresolved_axis_count += 1
            if "axis_non_axis_aligned" in unresolved_reason_codes:
                unresolved_questions.append(
                    "Resolve the joint axis because the hint is not axis-aligned."
                )
            else:
                unresolved_questions.append("Determine the joint axis.")
        if fixed_parent_prim is None:
            unresolved_parent_count += 1
            if "parent_self_reference" in unresolved_reason_codes:
                unresolved_questions.append(
                    "Resolve the fixed parent/connectivity because the parent hint "
                    "points to the moving prim."
                )
            else:
                unresolved_questions.append("Resolve the fixed parent/connectivity.")

        confidence = _candidate_confidence(
            payload.get("confidence"),
            unresolved_questions=unresolved_questions,
        )
        evidence = _clean_text(payload.get("evidence")) or _clean_text(
            payload.get("reasoning")
        )
        field_sources = _candidate_field_sources(
            joint_type=joint_type,
            axis_hint=axis_hint,
            motion_axis_world=motion_axis_world,
            fixed_parent_prim=fixed_parent_prim,
            parent_resolution_source=parent_resolution_source,
            parent_field_source=(
                parent_resolution.evidence[0].source
                if fixed_parent_prim is not None and parent_resolution.evidence
                else None
            ),
            payload=payload,
            axis_source=axis_source,
        )
        axis_evidence = _axis_evidence_for_candidate(
            axis_hint=axis_hint,
            motion_axis_world=motion_axis_world,
            moving_prim_path=moving_body_prim,
            axis_source=field_sources["axis_hint"],
            inferred_axis_evidence=inferred_axis_evidence,
        )
        if support_axis_conflict_evidence is not None:
            axis_evidence = [support_axis_conflict_evidence]
        limit_resolution = _candidate_limit_resolution(
            payload=payload,
            moving_prim_path=moving_body_prim,
            joint_type_hint=joint_type,
        )
        connectivity_evidence = _connectivity_evidence_for_candidate(
            moving_prim_path=moving_body_prim,
            fixed_parent_prim=fixed_parent_prim,
            parent_hint=parent_hint,
            parent_resolved_to_self=parent_resolved_to_self,
            parent_resolution_source=parent_resolution_source,
            explicit_parent_evidence=parent_resolution.evidence,
            explicit_body1_evidence=body1_resolution.evidence,
        )
        if compound_edge is not None:
            connectivity_evidence.extend(
                _compound_edge_canonicalization_evidence(compound_edge)
            )
        if parent_resolution.evidence is not compound_edge_conflict_evidence:
            connectivity_evidence.extend(compound_edge_conflict_evidence)
        review_status: Stage2ReviewStatus = (
            REVIEW_REQUIRED_STATUS
            if unresolved_reason_codes
            else READY_FOR_RIGGER_INPUT_STATUS
        )

        candidates.append(
            Stage2ArticulationCandidate(
                candidate_id=candidate_id,
                motion_type=_motion_type_from_joint_hint(joint_type),
                joint_type_hint=joint_type,
                axis_hint=axis_hint,
                motion_axis_world=motion_axis_world,
                confidence=confidence,
                moving_part_prims=_explicit_instance_moving_members(
                    payload,
                    moving_body_prim=moving_body_prim,
                ),
                fixed_parent_prim=fixed_parent_prim,
                parent_resolution_source=parent_resolution_source,
                parent_hint=parent_hint,
                child_hint=_clean_text(payload.get("child_hint"), "unknown"),
                component_name=_clean_text(payload.get("component_name"), "unknown"),
                component_type=_clean_text(payload.get("component_type"), "unknown"),
                role=_clean_token(payload.get("role"), "unknown"),
                source_prediction_ids=_dedupe_preserving_order(
                    [
                        value
                        for value in (
                            *payload_row.source_prediction_ids,
                            *(
                                compound_edge.source_prediction_ids
                                if compound_edge is not None
                                else ()
                            ),
                            *compound_edge_conflict_source_ids,
                        )
                        if value
                    ]
                ),
                evidence=evidence,
                source_annotation_conflicts={
                    str(field): [str(value) for value in values]
                    for field, values in (
                        payload.get("_source_support_annotation_conflicts", {})
                    ).items()
                    if isinstance(field, str) and isinstance(values, list)
                }
                if isinstance(
                    payload.get("_source_support_annotation_conflicts"),
                    Mapping,
                )
                else {},
                field_sources=field_sources,
                axis_evidence=axis_evidence,
                connectivity_evidence=connectivity_evidence,
                lower_limit=limit_resolution.lower_limit,
                upper_limit=limit_resolution.upper_limit,
                limit_unit=limit_resolution.unit,
                limit_source=limit_resolution.source,
                limit_readiness=limit_resolution.readiness,
                limit_evidence=limit_resolution.evidence,
                unresolved_reason_codes=unresolved_reason_codes,
                review_status=review_status,
                unresolved_questions=unresolved_questions,
            ).model_dump(mode="json")
        )

    existing_candidate_body1_paths = _candidate_moving_prim_paths(candidates)
    body1_paths_with_compound_evidence = _dedupe_preserving_order(
        [
            *compound_edges_by_body1,
            *compound_edge_conflicts_by_body1,
        ]
    )
    for body1 in body1_paths_with_compound_evidence:
        compound_edges = compound_edges_by_body1.get(body1, ())
        body1_conflict_evidence = list(compound_edge_conflicts_by_body1.get(body1, ()))
        body1_conflict_source_ids = list(
            compound_edge_conflict_sources_by_body1.get(body1, ())
        )
        if body1 in existing_candidate_body1_paths:
            continue
        if (
            body1_conflict_evidence
            and not compound_edges
            and body1 not in compound_edge_conflict_candidate_body1_paths
        ):
            continue
        if body1_conflict_evidence or len(compound_edges) > 1:
            candidate = _compound_edge_ambiguity_candidate(
                candidate_id=f"candidate_{len(candidates) + 1:04d}",
                body1=body1,
                compound_edges=compound_edges,
                conflict_evidence=body1_conflict_evidence,
                conflict_source_prediction_ids=body1_conflict_source_ids,
                payload_by_prim=payload_by_prim,
            )
            candidates.append(candidate)
            existing_candidate_body1_paths.add(body1)
            if candidate["motion_axis_world"] is None:
                unresolved_axis_count += 1
            unresolved_parent_count += 1
            continue
        for compound_edge in compound_edges:
            candidate = _compound_edge_candidate(
                candidate_id=f"candidate_{len(candidates) + 1:04d}",
                compound_edge=compound_edge,
                payload_by_prim=payload_by_prim,
                candidate_joint_type_set=candidate_joint_type_set,
            )
            candidates.append(candidate)
            if candidate["motion_axis_world"] is None:
                unresolved_axis_count += 1

    joint_counts = Counter(candidate["joint_type_hint"] for candidate in candidates)
    review_status_counts = Counter(
        candidate["review_status"] for candidate in candidates
    )
    limit_readiness_counts = Counter(
        candidate.get("limit_readiness", "not_provided") for candidate in candidates
    )
    reason_code_counts = Counter(
        code
        for candidate in candidates
        for code in candidate["unresolved_reason_codes"]
    )
    return {
        "schema_version": STAGE2_SCHEMA_VERSION,
        "summary": {
            "total_predictions": len(prediction_rows),
            "candidate_count": len(candidates),
            "ready_candidate_count": review_status_counts.get(
                READY_FOR_RIGGER_INPUT_STATUS, 0
            ),
            "review_required_candidate_count": review_status_counts.get(
                REVIEW_REQUIRED_STATUS, 0
            ),
            "joint_type_counts": dict(sorted(joint_counts.items())),
            "unresolved_axis_count": unresolved_axis_count,
            "unresolved_parent_count": unresolved_parent_count,
            "review_status_counts": dict(sorted(review_status_counts.items())),
            "limit_readiness_counts": dict(sorted(limit_readiness_counts.items())),
            "reason_code_counts": dict(sorted(reason_code_counts.items())),
            "source_structure_diagnostics": list(source_structure_index.diagnostics),
        },
        "candidates": candidates,
    }


class _PredictionPayload:
    def __init__(
        self,
        prim_path: str,
        payload: dict[str, Any],
        *,
        source_prediction_ids: Sequence[str] | None = None,
    ) -> None:
        self.prim_path = prim_path
        self.payload = payload
        self.source_prediction_ids = _dedupe_preserving_order(
            source_prediction_ids or ([prim_path] if prim_path else [])
        )


class _SourceStructureIndex(NamedTuple):
    endpoint_paths: frozenset[str]
    owner_by_prim_path: dict[str, str]
    hierarchy_endpoint_paths: frozenset[str] = frozenset()
    hierarchy_endpoint_paths_by_prim_path: dict[str, frozenset[str]] | None = None
    hierarchy_ancestor_paths_by_prim_path: dict[str, frozenset[str]] | None = None
    hierarchy_nearest_ancestor_by_prim_path: dict[str, str] | None = None
    hierarchy_transparent_leaf_by_wrapper: dict[str, str] | None = None
    diagnostics: tuple[str, ...] = ()
    structure_mode: Literal["legacy", "rigid_body", "hierarchy", "conflict"] = "legacy"

    def hierarchy_endpoints_for(self, prim_paths: Iterable[str]) -> set[str]:
        """Return only hierarchy choices shared by every supporting row."""
        endpoint_index = self.hierarchy_endpoint_paths_by_prim_path or {}
        row_paths = list(prim_paths)
        if not row_paths:
            return set()
        allowed = set(endpoint_index.get(row_paths[0], ()))
        for prim_path in row_paths[1:]:
            allowed.intersection_update(endpoint_index.get(prim_path, ()))
        return allowed

    def hierarchy_ancestors_for(self, prim_paths: Iterable[str]) -> set[str]:
        ancestor_index = self.hierarchy_ancestor_paths_by_prim_path or {}
        return {
            ancestor
            for prim_path in prim_paths
            for ancestor in ancestor_index.get(prim_path, ())
        }

    def shared_nearest_hierarchy_ancestor(
        self,
        prim_paths: Iterable[str],
    ) -> str | None:
        """Return one source-exported nearest ancestor shared by every row."""
        nearest_index = self.hierarchy_nearest_ancestor_by_prim_path or {}
        row_paths = list(prim_paths)
        if not row_paths:
            return None
        nearest_paths = {nearest_index.get(prim_path) for prim_path in row_paths}
        if None in nearest_paths or len(nearest_paths) != 1:
            return None
        return next(iter(nearest_paths))

    def transparent_leaf_aliases_for(
        self,
        prim_paths: Iterable[str],
    ) -> dict[str, str]:
        """Return structural aliases authorized by every supporting row."""
        allowed_wrappers = self.hierarchy_endpoints_for(prim_paths)
        transparent_leaf_by_wrapper = self.hierarchy_transparent_leaf_by_wrapper or {}
        return {
            wrapper_path: leaf_path
            for wrapper_path, leaf_path in transparent_leaf_by_wrapper.items()
            if wrapper_path in allowed_wrappers
        }


class _ParentResolution(NamedTuple):
    fixed_parent_prim: str | None
    source: Stage2ParentResolutionSource
    parent_resolved_to_self: bool
    evidence: list[Stage2EvidenceItem]
    evidence_present: bool


class _Body1Resolution(NamedTuple):
    evidence: list[Stage2EvidenceItem]
    evidence_present: bool
    resolved: bool
    body1_prim: str | None = None


class _LimitResolution(NamedTuple):
    lower_limit: float | None
    upper_limit: float | None
    unit: str
    source: Stage2FieldSource
    readiness: Stage2LimitReadiness
    evidence: list[Stage2EvidenceItem]


class _EndpointCanonicalization(NamedTuple):
    endpoint: Literal["body0", "body1"]
    wrapper_path: str
    leaf_path: str


class _CompoundEdgeResolution(NamedTuple):
    body0: str
    body1: str
    joint_type_hint: str
    raw_axis_hint: str
    axis_hint: str
    source: Stage2FieldSource
    confidence: Stage2Confidence
    rationale: str
    prim_paths: list[str]
    source_prediction_ids: tuple[str, ...]
    endpoint_canonicalizations: tuple[_EndpointCanonicalization, ...]

    @property
    def key(self) -> tuple[str, str]:
        return (self.body0, self.body1)


class _CompoundEdgeCollection(NamedTuple):
    edges_by_body1: dict[str, list[_CompoundEdgeResolution]]
    conflict_evidence_by_body1: dict[str, list[Stage2EvidenceItem]]
    conflict_source_prediction_ids_by_body1: dict[str, list[str]]
    conflict_candidate_body1_paths: set[str]


def _normalize_prim_metadata_index(
    prim_metadata: Iterable[dict[str, Any]] | Mapping[str, dict[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    """Normalize legacy prim metadata into a prim-path keyed mapping."""
    if prim_metadata is None:
        return {}
    if isinstance(prim_metadata, Mapping):
        index: dict[str, dict[str, Any]] = {}
        for prim_path, metadata in prim_metadata.items():
            if not isinstance(metadata, Mapping):
                continue
            metadata_dict = dict(metadata)
            metadata_dict.setdefault("id", str(prim_path))
            index[str(prim_path)] = metadata_dict
        return index

    index = {}
    for row in prim_metadata:
        if not isinstance(row, dict):
            continue
        prim_path = _clean_text(
            row.get("id") or row.get("prim_path") or row.get("path")
        )
        if prim_path:
            index[prim_path] = row
    return index


def _source_structure_index(
    prim_metadata: Iterable[dict[str, Any]] | Mapping[str, dict[str, Any]] | None,
) -> _SourceStructureIndex:
    """Extract only structure facts explicitly exported by dataset preparation."""
    metadata_index = _normalize_prim_metadata_index(prim_metadata)
    endpoint_vocabularies: set[frozenset[str]] = set()
    raw_owners: dict[str, str] = {}
    hierarchy_endpoint_paths: set[str] = set()
    hierarchy_endpoint_paths_by_prim_path: dict[str, frozenset[str]] = {}
    hierarchy_ancestor_paths_by_prim_path: dict[str, frozenset[str]] = {}
    hierarchy_nearest_ancestor_by_prim_path: dict[str, str] = {}
    structure_modes: set[str] = set()
    diagnostics: list[str] = []
    for prim_path, metadata in metadata_index.items():
        raw_structure = metadata.get("usd_metadata")
        if not isinstance(raw_structure, Mapping):
            raw_structure = metadata
        provenance = _clean_token(raw_structure.get("structure_provenance"), "unknown")
        if provenance == "source_metadata":
            structure_modes.add("rigid_body")
        elif provenance == "source_hierarchy":
            structure_modes.add("hierarchy")
            raw_hierarchy_paths = raw_structure.get("hierarchy_xform_paths")
            row_hierarchy_paths = frozenset(
                path
                for path in (
                    _clean_text(value)
                    for value in (
                        raw_hierarchy_paths
                        if isinstance(raw_hierarchy_paths, list)
                        else []
                    )
                )
                if path.startswith("/")
            )
            hierarchy_endpoint_paths.update(row_hierarchy_paths)
            hierarchy_endpoint_paths_by_prim_path[prim_path] = row_hierarchy_paths
            raw_ancestor_paths = raw_structure.get("hierarchy_ancestor_xform_paths")
            ordered_ancestor_paths = _dedupe_preserving_order(
                path
                for path in (
                    _clean_text(value)
                    for value in (
                        raw_ancestor_paths
                        if isinstance(raw_ancestor_paths, list)
                        else []
                    )
                )
                if path.startswith("/") and path in row_hierarchy_paths
            )
            ancestor_paths = frozenset(ordered_ancestor_paths)
            if ancestor_paths:
                hierarchy_ancestor_paths_by_prim_path[prim_path] = ancestor_paths
                hierarchy_nearest_ancestor_by_prim_path[prim_path] = (
                    ordered_ancestor_paths[0]
                )
            continue
        else:
            continue

        raw_endpoints = raw_structure.get("rigid_body_endpoint_paths")
        if isinstance(raw_endpoints, list):
            endpoint_vocabulary = frozenset(
                endpoint
                for endpoint in (_clean_text(value) for value in raw_endpoints)
                if endpoint.startswith("/")
            )
            if endpoint_vocabulary:
                endpoint_vocabularies.add(endpoint_vocabulary)

        owner_path = _clean_text(raw_structure.get("rigid_body_owner_path"))
        if owner_path.startswith("/"):
            raw_owners[prim_path] = owner_path

    if len(structure_modes) > 1:
        return _SourceStructureIndex(
            endpoint_paths=frozenset(),
            owner_by_prim_path={},
            diagnostics=("source_structure_mode_conflict",),
            structure_mode="conflict",
        )

    if len(endpoint_vocabularies) > 1:
        return _SourceStructureIndex(
            endpoint_paths=frozenset(),
            owner_by_prim_path={},
            diagnostics=("endpoint_vocabulary_conflict",),
            structure_mode="conflict",
        )

    endpoint_paths = (
        next(iter(endpoint_vocabularies)) if endpoint_vocabularies else frozenset()
    )

    owner_by_prim_path = {
        prim_path: owner_path
        for prim_path, owner_path in raw_owners.items()
        if owner_path in endpoint_paths
    }
    return _SourceStructureIndex(
        endpoint_paths=endpoint_paths,
        owner_by_prim_path=owner_by_prim_path,
        hierarchy_endpoint_paths=frozenset(hierarchy_endpoint_paths),
        hierarchy_endpoint_paths_by_prim_path=(hierarchy_endpoint_paths_by_prim_path),
        hierarchy_ancestor_paths_by_prim_path=(hierarchy_ancestor_paths_by_prim_path),
        hierarchy_nearest_ancestor_by_prim_path=(
            hierarchy_nearest_ancestor_by_prim_path
        ),
        diagnostics=tuple(diagnostics),
        structure_mode=(
            cast(
                Literal["rigid_body", "hierarchy"],
                next(iter(structure_modes)),
            )
            if structure_modes
            else "legacy"
        ),
    )


def _candidate_endpoint_vocabulary(
    *,
    source_structure_index: _SourceStructureIndex,
    source_prediction_paths: Sequence[str],
    prediction_prim_paths: set[str],
    legacy_known_prim_paths: set[str],
) -> set[str]:
    """Return endpoints trusted for one direct Stage 2 candidate.

    Once prepared metadata declares a structure mode, that mode is
    authoritative even when its sanitized endpoint vocabulary is empty. Only
    inputs with no prepared structure retain the legacy known-prediction-path
    behavior.
    """
    if source_structure_index.structure_mode == "conflict":
        return set()
    if source_structure_index.structure_mode == "rigid_body":
        return set(source_structure_index.endpoint_paths)
    if source_structure_index.structure_mode == "hierarchy":
        hierarchy_paths = source_structure_index.hierarchy_endpoints_for(
            source_prediction_paths
        )
        if not hierarchy_paths:
            return set()
        return {*prediction_prim_paths, *hierarchy_paths}
    return set(legacy_known_prim_paths)


def _candidate_direct_body0_vocabulary(
    *,
    source_structure_index: _SourceStructureIndex,
    source_prediction_paths: Sequence[str],
    candidate_endpoint_paths: set[str],
    fixed_support_prediction_paths: set[str],
    allow_fixed_support_gprims: bool,
    explicit_moving_body0_paths: set[str],
) -> set[str]:
    """Return fixed-body endpoints authorized for direct row-local evidence.

    Hierarchy-only candidates admit prediction-row Gprims so that the current
    row can name itself as ``body1``.  That narrow moving-body permission must
    not expand the direct ``body0`` vocabulary to every rendered sibling. An
    exact sibling Gprim is admitted only for an explicitly resolved direct edge
    whose ``body1`` is the current row, when either the sibling's own row
    independently labels it non-articulating and fixed or a separately validated
    explicit moving-link declaration identifies it as that link's representative.
    In both cases the exported hierarchy must place all supporting rows in the
    same nearest assembly. Ancestor-assembly ``body1`` claims do not get this
    Gprim exception. A transparent Xform wrapper may still be canonicalized to
    its sole observable leaf later, with explicit source-derived lineage on the
    claim.

    This restriction is local to direct Stage 1 hierarchy evidence.  A future
    asset-level resolver with independent graph provenance can define a
    separate endpoint vocabulary instead of weakening this row-local guard.
    """
    if source_structure_index.structure_mode == "hierarchy":
        return {
            *source_structure_index.hierarchy_endpoints_for(source_prediction_paths),
            *explicit_moving_body0_paths,
            *(
                _same_hierarchy_assembly_fixed_supports(
                    source_structure_index=source_structure_index,
                    source_prediction_paths=source_prediction_paths,
                    fixed_support_prediction_paths=fixed_support_prediction_paths,
                )
                if allow_fixed_support_gprims
                else set()
            ),
        }
    return set(candidate_endpoint_paths)


def _model_supplied_explicit_moving_body0_paths(
    *,
    payload: dict[str, Any],
    moving_body_prim: str,
    source_prediction_paths: Sequence[str],
    source_structure_index: _SourceStructureIndex,
    explicit_moving_link_representatives: Mapping[str, Sequence[str]],
) -> set[str]:
    """Authorize one exact nested parent backed by two explicit link groups.

    Hierarchy-only metadata intentionally does not make arbitrary sibling Gprims
    valid endpoints. The narrow exception here requires the current candidate to
    be a validated explicit moving-link representative, an exact model-supplied
    ``body0`` claim naming another validated representative, and one shared
    source-exported nearest Xform across both complete link declarations.
    """
    if (
        source_structure_index.structure_mode != "hierarchy"
        or moving_body_prim not in explicit_moving_link_representatives
    ):
        return set()

    body0_claim = _rigger_evidence_claim(payload, "body0")
    if (
        body0_claim is None
        or _rigger_claim_stage2_source(body0_claim)
        not in _MODEL_PREDICTED_STAGE2_SOURCES
    ):
        return set()
    body0 = _single_rigger_claim_endpoint(payload, "body0")
    if (
        body0 is None
        or body0 == moving_body_prim
        or body0 not in explicit_moving_link_representatives
    ):
        return set()

    shared_assembly = source_structure_index.shared_nearest_hierarchy_ancestor(
        [
            *source_prediction_paths,
            *explicit_moving_link_representatives[body0],
        ]
    )
    return {body0} if shared_assembly is not None else set()


def _explicit_fixed_support_prediction_paths(
    payloads: Sequence[_PredictionPayload],
) -> set[str]:
    """Return rows that independently identify an exact fixed support Gprim."""
    return {
        row.prim_path
        for row in payloads
        if row.prim_path
        and _articulation_candidate_flag_state(
            row.payload.get("is_articulation_candidate")
        )
        is False
        and _clean_token(row.payload.get("joint_type_hint"), "unknown") == "fixed"
        and _stage1_field_source(row.payload, "is_articulation_candidate")
        != "consistency_corrected"
        and _stage1_field_source(row.payload, "joint_type_hint")
        != "consistency_corrected"
    }


def _same_hierarchy_assembly_fixed_supports(
    *,
    source_structure_index: _SourceStructureIndex,
    source_prediction_paths: Sequence[str],
    fixed_support_prediction_paths: set[str],
) -> set[str]:
    """Return independently fixed rows in the same exact source assembly."""
    source_paths = [path for path in source_prediction_paths if path]
    if not source_paths:
        return set()
    return {
        support_path
        for support_path in fixed_support_prediction_paths
        if support_path not in source_paths
        and source_structure_index.shared_nearest_hierarchy_ancestor(
            [*source_paths, support_path]
        )
        is not None
    }


def _body0_path_is_authorized(
    candidate_parent: str,
    *,
    authorized_body0_paths: set[str],
    hierarchy_canonicalization: tuple[str, str] | None,
) -> bool:
    """Validate a direct body0 or its source-derived transparent alias."""
    return candidate_parent in authorized_body0_paths or (
        hierarchy_canonicalization is not None
        and hierarchy_canonicalization[0] in authorized_body0_paths
        and hierarchy_canonicalization[1] == candidate_parent
    )


_HIERARCHY_WRAPPER_PATH = "_source_hierarchy_wrapper_path"
_HIERARCHY_LEAF_PATH = "_source_hierarchy_leaf_path"
_HIERARCHY_ENDPOINT_CANONICALIZATIONS = "_source_hierarchy_endpoint_canonicalizations"


def _with_observable_hierarchy_leaf_aliases(
    source_structure_index: _SourceStructureIndex,
    payloads: Sequence[_PredictionPayload],
) -> _SourceStructureIndex:
    if source_structure_index.structure_mode != "hierarchy":
        return source_structure_index

    observable_paths = {row.prim_path for row in payloads if row.prim_path}
    descendant_leaves_by_wrapper: dict[str, set[str]] = {}
    ancestor_index = source_structure_index.hierarchy_ancestor_paths_by_prim_path or {}
    nearest_ancestor_index = (
        source_structure_index.hierarchy_nearest_ancestor_by_prim_path or {}
    )
    for observable_path in observable_paths:
        for wrapper_path in ancestor_index.get(observable_path, ()):
            descendant_leaves_by_wrapper.setdefault(wrapper_path, set()).add(
                observable_path
            )
    transparent_leaf_by_wrapper = {
        wrapper_path: next(iter(leaves))
        for wrapper_path, leaves in descendant_leaves_by_wrapper.items()
        if len(leaves) == 1
        and wrapper_path not in leaves
        and nearest_ancestor_index.get(next(iter(leaves))) == wrapper_path
    }
    return source_structure_index._replace(
        hierarchy_transparent_leaf_by_wrapper=transparent_leaf_by_wrapper
    )


def _canonicalize_transparent_hierarchy_endpoint_claims(
    payloads: Sequence[_PredictionPayload],
    *,
    source_structure_index: _SourceStructureIndex,
) -> list[_PredictionPayload]:
    """Canonicalize only hierarchy wrappers with one observable Gprim row.

    An unrigged hierarchy can expose an Xform as a bounded endpoint choice even
    when the Xform is only a transparent parent of one rendered Gprim.  In that
    one unambiguous case the observable prediction row is the authorable leaf
    endpoint.  Xforms containing zero or multiple observable rows are retained
    verbatim: they may be real rigid assemblies, and Stage 2 has no evidence to
    split them.

    The mapping uses only source-exported ancestor relationships and prediction
    row IDs.  It never inspects asset identity, path labels, or reference data.
    """
    result: list[_PredictionPayload] = []
    for row in payloads:
        source_prediction_ids = row.source_prediction_ids or [row.prim_path]
        row_transparent_leaf_by_wrapper = (
            source_structure_index.transparent_leaf_aliases_for(source_prediction_ids)
            if source_structure_index.structure_mode == "hierarchy"
            else {}
        )
        evidence = row.payload.get("rigger_evidence")
        if not isinstance(evidence, dict):
            result.append(row)
            continue

        references_transparent_wrapper = (
            _rigger_evidence_references_transparent_hierarchy_wrapper(
                evidence,
                transparent_leaf_by_wrapper=row_transparent_leaf_by_wrapper,
            )
        )
        has_internal_canonicalization = _rigger_evidence_has_internal_canonicalization(
            evidence
        )
        if not references_transparent_wrapper and not has_internal_canonicalization:
            result.append(row)
            continue

        payload = copy.deepcopy(row.payload)
        copied_evidence = cast(dict[str, Any], payload["rigger_evidence"])
        changed = _discard_internal_canonicalization(copied_evidence)
        for field in ("body0", "body1"):
            claim = copied_evidence.get(field)
            if isinstance(claim, dict):
                changed = (
                    _canonicalize_transparent_hierarchy_rigger_claim(
                        claim,
                        transparent_leaf_by_wrapper=row_transparent_leaf_by_wrapper,
                    )
                    or changed
                )

        raw_edges = copied_evidence.get("compound_edges")
        if isinstance(raw_edges, list):
            for raw_edge in raw_edges:
                if isinstance(raw_edge, dict):
                    changed = (
                        _canonicalize_transparent_hierarchy_compound_edge(
                            raw_edge,
                            transparent_leaf_by_wrapper=(
                                row_transparent_leaf_by_wrapper
                            ),
                        )
                        or changed
                    )

        result.append(
            _PredictionPayload(
                row.prim_path,
                payload if changed else row.payload,
                source_prediction_ids=row.source_prediction_ids,
            )
        )
    return result


def _rigger_evidence_has_internal_canonicalization(
    evidence: Mapping[str, Any],
) -> bool:
    """Return whether untrusted input contains producer-private lineage keys."""
    for field in ("body0", "body1"):
        claim = evidence.get(field)
        if isinstance(claim, Mapping) and (
            _HIERARCHY_WRAPPER_PATH in claim or _HIERARCHY_LEAF_PATH in claim
        ):
            return True

    raw_edges = evidence.get("compound_edges")
    return isinstance(raw_edges, list) and any(
        isinstance(raw_edge, Mapping)
        and _HIERARCHY_ENDPOINT_CANONICALIZATIONS in raw_edge
        for raw_edge in raw_edges
    )


def _discard_internal_canonicalization(evidence: dict[str, Any]) -> bool:
    """Strip caller-supplied private keys before deterministic correction.

    These keys are implementation details written only after the bounded source
    hierarchy proves a wrapper-to-leaf mapping. Accepting them from prediction
    JSON would let an input fabricate ``consistency_corrected`` evidence.
    """
    changed = False
    for field in ("body0", "body1"):
        claim = evidence.get(field)
        if not isinstance(claim, dict):
            continue
        for key in (_HIERARCHY_WRAPPER_PATH, _HIERARCHY_LEAF_PATH):
            if key in claim:
                claim.pop(key)
                changed = True

    raw_edges = evidence.get("compound_edges")
    if isinstance(raw_edges, list):
        for raw_edge in raw_edges:
            if (
                isinstance(raw_edge, dict)
                and _HIERARCHY_ENDPOINT_CANONICALIZATIONS in raw_edge
            ):
                raw_edge.pop(_HIERARCHY_ENDPOINT_CANONICALIZATIONS)
                changed = True
    return changed


def _rigger_evidence_references_transparent_hierarchy_wrapper(
    evidence: Mapping[str, Any],
    *,
    transparent_leaf_by_wrapper: Mapping[str, str],
) -> bool:
    """Return whether row-scoped evidence has an endpoint safe to rewrite."""
    if not transparent_leaf_by_wrapper:
        return False

    for field in ("body0", "body1"):
        claim = evidence.get(field)
        if not isinstance(claim, dict):
            continue
        claim_value = _clean_text(claim.get("value"), "unknown")
        exact_paths = _rigger_claim_exact_paths(claim, claim_value=claim_value)
        if len(exact_paths) == 1 and exact_paths[0] in transparent_leaf_by_wrapper:
            return True

    raw_edges = evidence.get("compound_edges")
    if not isinstance(raw_edges, list):
        return False
    for raw_edge in raw_edges:
        if not isinstance(raw_edge, Mapping):
            continue
        for field, alias in (
            ("body0", "fixed_parent_prim"),
            ("body1", "moving_body_prim"),
        ):
            endpoint_values = {
                value
                for key in (field, alias)
                if (
                    value := _compound_edge_endpoint_value(raw_edge.get(key))
                ).startswith("/")
            }
            if (
                len(endpoint_values) == 1
                and next(iter(endpoint_values)) in transparent_leaf_by_wrapper
            ):
                return True
    return False


def _canonicalize_transparent_hierarchy_rigger_claim(
    claim: dict[str, Any],
    *,
    transparent_leaf_by_wrapper: Mapping[str, str],
) -> bool:
    claim_value = _clean_text(claim.get("value"), "unknown")
    exact_paths = _rigger_claim_exact_paths(claim, claim_value=claim_value)
    if len(exact_paths) != 1:
        return False
    wrapper_path = exact_paths[0]
    leaf_path = transparent_leaf_by_wrapper.get(wrapper_path)
    if leaf_path is None:
        return False

    claim["value"] = leaf_path
    claim["prim_paths"] = [leaf_path]
    claim["source"] = "consistency_corrected"
    claim[_HIERARCHY_WRAPPER_PATH] = wrapper_path
    claim[_HIERARCHY_LEAF_PATH] = leaf_path
    claim["rationale"] = _append_hierarchy_canonicalization_rationale(
        claim.get("rationale"),
        wrapper_path=wrapper_path,
        leaf_path=leaf_path,
    )
    return True


def _canonicalize_transparent_hierarchy_compound_edge(
    edge: dict[str, Any],
    *,
    transparent_leaf_by_wrapper: Mapping[str, str],
) -> bool:
    canonicalizations: list[dict[str, str]] = []
    for field, alias in (
        ("body0", "fixed_parent_prim"),
        ("body1", "moving_body_prim"),
    ):
        endpoint_values = {
            value
            for key in (field, alias)
            if (value := _compound_edge_endpoint_value(edge.get(key))).startswith("/")
        }
        # Conflicting primary/alias endpoint claims are not safe to rewrite.
        if len(endpoint_values) != 1:
            continue
        wrapper_path = next(iter(endpoint_values))
        leaf_path = transparent_leaf_by_wrapper.get(wrapper_path)
        if leaf_path is None:
            continue
        for key in (field, alias):
            if key not in edge:
                continue
            value = edge[key]
            if isinstance(value, dict):
                value["value"] = leaf_path
                if isinstance(value.get("prim_paths"), list):
                    value["prim_paths"] = [leaf_path]
            else:
                edge[key] = leaf_path
        canonicalizations.append(
            {"field": field, "wrapper_path": wrapper_path, "leaf_path": leaf_path}
        )

    if not canonicalizations:
        return False

    edge["source"] = "consistency_corrected"
    edge[_HIERARCHY_ENDPOINT_CANONICALIZATIONS] = canonicalizations
    raw_prim_paths = edge.get("prim_paths")
    if isinstance(raw_prim_paths, list):
        replacements = {
            item["wrapper_path"]: item["leaf_path"] for item in canonicalizations
        }
        edge["prim_paths"] = _dedupe_preserving_order(
            [
                replacements.get(_clean_text(path), _clean_text(path))
                for path in raw_prim_paths
                if _clean_text(path).startswith("/")
            ]
        )
    rationale = edge.get("rationale") or edge.get("reasoning")
    for item in canonicalizations:
        rationale = _append_hierarchy_canonicalization_rationale(
            rationale,
            wrapper_path=item["wrapper_path"],
            leaf_path=item["leaf_path"],
        )
    edge["rationale"] = rationale
    return True


def _append_hierarchy_canonicalization_rationale(
    rationale: Any,
    *,
    wrapper_path: str,
    leaf_path: str,
) -> str:
    canonicalization = (
        "Source hierarchy canonicalized transparent wrapper "
        f"{wrapper_path} to its sole observable prediction row {leaf_path}."
    )
    existing = _clean_text(rationale)
    return f"{existing}; {canonicalization}" if existing else canonicalization


def _synthesize_source_backed_body_payloads(
    payloads: Sequence[_PredictionPayload],
    *,
    source_structure_index: _SourceStructureIndex,
    candidate_joint_type_set: set[str],
) -> list[_PredictionPayload]:
    """Collapse rows only after explicit claims pass source-structure checks."""
    if not (
        source_structure_index.endpoint_paths
        or source_structure_index.hierarchy_endpoint_paths
    ):
        return list(payloads)

    fixed_support_prediction_paths = _explicit_fixed_support_prediction_paths(payloads)
    groups: dict[tuple[str, str], list[tuple[int, _PredictionPayload]]] = {}
    grouped_indices: set[int] = set()
    for index, row in enumerate(payloads):
        candidate_flag = _articulation_candidate_flag_state(
            row.payload.get("is_articulation_candidate")
        )
        joint_type = _clean_token(row.payload.get("joint_type_hint"), "unknown")
        if candidate_flag is None and joint_type not in candidate_joint_type_set:
            continue

        source_owner = source_structure_index.owner_by_prim_path.get(row.prim_path)
        body0 = _single_rigger_claim_endpoint(row.payload, "body0")
        body1 = _single_rigger_claim_endpoint(row.payload, "body1")
        if body0 is None or body1 is None or body0 == body1:
            continue
        if source_structure_index.endpoint_paths:
            structure_valid = (
                source_owner is not None
                and body0 in source_structure_index.endpoint_paths
                and body1 == source_owner
            )
        else:
            fixed_support_paths = (
                _same_hierarchy_assembly_fixed_supports(
                    source_structure_index=source_structure_index,
                    source_prediction_paths=[row.prim_path],
                    fixed_support_prediction_paths=fixed_support_prediction_paths,
                )
                if body1 == row.prim_path
                else set()
            )
            allowed_body0_paths = {
                *source_structure_index.hierarchy_endpoints_for([row.prim_path]),
                *fixed_support_paths,
            }
            allowed_body1_paths = {
                row.prim_path,
                *source_structure_index.hierarchy_ancestors_for([row.prim_path]),
            }
            body0_claim = _rigger_evidence_claim(row.payload, "body0")
            body0_canonicalization = (
                _rigger_claim_hierarchy_canonicalization(body0_claim)
                if body0_claim is not None
                else None
            )
            structure_valid = (
                _body0_path_is_authorized(
                    body0,
                    authorized_body0_paths=allowed_body0_paths,
                    hierarchy_canonicalization=body0_canonicalization,
                )
                and body1 in allowed_body1_paths
            )
        if not structure_valid:
            continue
        groups.setdefault((body0, body1), []).append((index, row))
        grouped_indices.add(index)

    if not groups:
        return list(payloads)

    groups_by_body1: dict[
        str,
        list[tuple[str, list[tuple[int, _PredictionPayload]]]],
    ] = {}
    for (body0, body1), indexed_rows in groups.items():
        groups_by_body1.setdefault(body1, []).append((body0, indexed_rows))

    synthesized_by_first_index: dict[int, _PredictionPayload] = {}
    for body1, body1_groups in groups_by_body1.items():
        ordered_groups = sorted(
            body1_groups,
            key=lambda item: item[1][0][0],
        )
        first_index = ordered_groups[0][1][0][0]
        body0_paths = [body0 for body0, _ in ordered_groups]
        indexed_rows = sorted(
            (
                indexed_row
                for _, group_rows in ordered_groups
                for indexed_row in group_rows
            ),
            key=lambda item: item[0],
        )
        synthesized = _synthesize_body_edge_payload(
            body0=body0_paths[0],
            body1=body1,
            rows=[row for _, row in indexed_rows],
        )
        if len(body0_paths) > 1:
            evidence = synthesized.payload.get("rigger_evidence")
            if isinstance(evidence, dict):
                evidence["body0"] = {
                    "value": "conflicting explicit body0 endpoint claims",
                    "prim_paths": body0_paths,
                    "confidence": "low",
                    "rationale": (
                        "Stage 1 rows for the same source-backed moving body "
                        "selected different fixed endpoints."
                    ),
                    "source": "consistency_corrected",
                }
        synthesized_by_first_index[first_index] = synthesized

    result: list[_PredictionPayload] = []
    for index, row in enumerate(payloads):
        synthesized_row = synthesized_by_first_index.get(index)
        if synthesized_row is not None:
            result.append(synthesized_row)
        elif index not in grouped_indices:
            result.append(row)
    return result


def _collapse_explicit_instance_link_members(
    payloads: Sequence[_PredictionPayload],
    *,
    candidate_joint_type_set: set[str],
) -> list[_PredictionPayload]:
    """Collapse only a complete, internally coherent explicit link declaration.

    ``instance_id`` is prediction evidence, not a name heuristic. Aggregation is
    permitted only when every row explicitly agrees on role, promotable joint
    type, stage-space axis, and one link identifier; exactly one row must be the
    candidate anchor with exact body0/body1 claims. Missing or contradictory
    evidence marks the anchor review-required instead of inventing membership or
    a parent edge.
    """
    sanitized_payloads: list[_PredictionPayload] = []
    for row in payloads:
        if not _EXPLICIT_INSTANCE_INTERNAL_FIELDS.intersection(row.payload):
            sanitized_payloads.append(row)
            continue
        payload = copy.deepcopy(row.payload)
        for field in _EXPLICIT_INSTANCE_INTERNAL_FIELDS:
            payload.pop(field, None)
        sanitized_payloads.append(
            _PredictionPayload(
                row.prim_path,
                payload,
                source_prediction_ids=row.source_prediction_ids,
            )
        )

    groups: dict[str, list[tuple[int, _PredictionPayload]]] = {}
    for index, row in enumerate(sanitized_payloads):
        if not is_model_supplied_link_instance_id(row.payload):
            continue
        instance_id = canonical_link_instance_id(row.payload.get("instance_id"))
        if instance_id and instance_id not in _UNKNOWN_VALUES:
            groups.setdefault(instance_id, []).append((index, row))

    replacements: dict[int, _PredictionPayload] = {}
    consumed_indices: set[int] = set()
    conflict_rows: dict[int, list[str]] = {}
    for instance_id, indexed_rows in groups.items():
        if len(indexed_rows) < 2:
            continue
        reasons = _explicit_instance_link_conflicts(
            indexed_rows,
            candidate_joint_type_set=candidate_joint_type_set,
        )
        anchors = [
            (index, row)
            for index, row in indexed_rows
            if _articulation_candidate_flag_state(
                row.payload.get("is_articulation_candidate")
            )
            is True
            and _clean_token(row.payload.get("joint_type_hint"), "unknown")
            in candidate_joint_type_set
        ]
        if reasons:
            for index, _ in anchors:
                conflict_rows[index] = reasons
            continue
        if len(anchors) != 1:  # Defensive; the validator reports this above.
            continue

        anchor_index, anchor = anchors[0]
        member_paths = _dedupe_preserving_order(
            [anchor.prim_path]
            + [row.prim_path for _, row in indexed_rows if row is not anchor]
        )
        payload = copy.deepcopy(anchor.payload)
        payload["_explicit_instance_link_members"] = member_paths
        payload["_explicit_instance_link_id"] = instance_id
        source_prediction_ids = _dedupe_preserving_order(
            list(anchor.source_prediction_ids)
            + [
                source_id
                for _, row in indexed_rows
                if row is not anchor
                for source_id in row.source_prediction_ids
            ]
        )
        first_index = min(index for index, _ in indexed_rows)
        replacements[first_index] = _PredictionPayload(
            anchor.prim_path,
            payload,
            source_prediction_ids=source_prediction_ids,
        )
        consumed_indices.update(index for index, _ in indexed_rows)

    result: list[_PredictionPayload] = []
    for index, row in enumerate(sanitized_payloads):
        replacement = replacements.get(index)
        if replacement is not None:
            result.append(replacement)
            continue
        if index in consumed_indices:
            continue
        row_conflicts = conflict_rows.get(index)
        if row_conflicts is None:
            result.append(row)
            continue
        payload = copy.deepcopy(row.payload)
        payload["_explicit_instance_link_conflict"] = True
        raw_conflicts = payload.get("_source_support_annotation_conflicts")
        conflicts = dict(raw_conflicts) if isinstance(raw_conflicts, Mapping) else {}
        conflicts["instance_link_membership"] = list(row_conflicts)
        payload["_source_support_annotation_conflicts"] = conflicts
        result.append(
            _PredictionPayload(
                row.prim_path,
                payload,
                source_prediction_ids=row.source_prediction_ids,
            )
        )
    return result


def _explicit_instance_link_representatives(
    payloads: Sequence[_PredictionPayload],
    *,
    candidate_joint_type_set: set[str],
    source_structure_index: _SourceStructureIndex,
) -> dict[str, tuple[str, ...]]:
    """Return source-backed representatives allowed as nested moving parents.

    Multi-member links must have survived the strict collapse validator. A
    singleton cannot be collapsed, so it is admitted only when its sole row is
    an exact source-hierarchy prim with complete model-supplied moving-candidate
    semantics and predicted body0/body1 claims. Groups with two or more raw rows
    never take this singleton path, including ambiguous groups that failed
    collapse.
    """
    representatives: dict[str, tuple[str, ...]] = {}
    for row in payloads:
        instance_id = _clean_text(row.payload.get("_explicit_instance_link_id"))
        raw_members = row.payload.get("_explicit_instance_link_members")
        if (
            instance_id
            and isinstance(raw_members, list)
            and row.prim_path in raw_members
        ):
            representatives[row.prim_path] = tuple(row.source_prediction_ids)

    if source_structure_index.structure_mode != "hierarchy":
        return representatives

    singleton_groups: dict[str, list[_PredictionPayload]] = {}
    for row in payloads:
        if not is_model_supplied_link_instance_id(row.payload):
            continue
        instance_id = canonical_link_instance_id(row.payload.get("instance_id"))
        if instance_id and instance_id not in _UNKNOWN_VALUES:
            singleton_groups.setdefault(instance_id, []).append(row)

    nearest_ancestor_by_prim = (
        source_structure_index.hierarchy_nearest_ancestor_by_prim_path or {}
    )
    pending_singletons: dict[str, tuple[_PredictionPayload, str]] = {}
    for rows in singleton_groups.values():
        if len(rows) != 1:
            continue
        row = rows[0]
        if (
            row.prim_path in representatives
            or row.source_prediction_ids != [row.prim_path]
            or row.prim_path not in nearest_ancestor_by_prim
            or _explicit_instance_link_conflicts(
                [(0, row)],
                candidate_joint_type_set=candidate_joint_type_set,
            )
        ):
            continue
        body0_claim = _rigger_evidence_claim(row.payload, "body0")
        body1_claim = _rigger_evidence_claim(row.payload, "body1")
        body0 = _single_rigger_claim_endpoint(row.payload, "body0")
        body1 = _single_rigger_claim_endpoint(row.payload, "body1")
        if (
            body0_claim is None
            or body1_claim is None
            or _rigger_claim_stage2_source(body0_claim)
            not in _MODEL_PREDICTED_STAGE2_SOURCES
            or _rigger_claim_stage2_source(body1_claim)
            not in _MODEL_PREDICTED_STAGE2_SOURCES
            or body0 is None
            or body0 == row.prim_path
            or body1 != row.prim_path
        ):
            continue
        pending_singletons[row.prim_path] = (row, body0)

    while pending_singletons:
        admitted_paths: list[str] = []
        for prim_path, (row, body0) in pending_singletons.items():
            body0_is_source_endpoint = body0 in (
                source_structure_index.hierarchy_endpoints_for(
                    row.source_prediction_ids
                )
            )
            body0_is_validated_moving_link = (
                body0 in representatives
                and source_structure_index.shared_nearest_hierarchy_ancestor(
                    [
                        *row.source_prediction_ids,
                        *representatives[body0],
                    ]
                )
                is not None
            )
            if not body0_is_source_endpoint and not body0_is_validated_moving_link:
                continue
            representatives[prim_path] = tuple(row.source_prediction_ids)
            admitted_paths.append(prim_path)
        if not admitted_paths:
            break
        for prim_path in admitted_paths:
            pending_singletons.pop(prim_path)
    return representatives


def _explicit_instance_link_conflicts(
    indexed_rows: Sequence[tuple[int, _PredictionPayload]],
    *,
    candidate_joint_type_set: set[str],
) -> list[str]:
    rows = [row for _, row in indexed_rows]
    reasons: list[str] = []
    prim_paths = [row.prim_path for row in rows]
    if any(not path.startswith("/") for path in prim_paths) or len(
        set(prim_paths)
    ) != len(prim_paths):
        reasons.append("member_paths_not_distinct_absolute_prims")

    roles = {_clean_token(row.payload.get("role"), "unknown") for row in rows}
    if len(roles) != 1 or next(iter(roles), "unknown") in {
        *_UNKNOWN_VALUES,
        "body",
    }:
        reasons.append("roles_do_not_identify_one_moving_link")

    joint_types = {
        _clean_token(row.payload.get("joint_type_hint"), "unknown") for row in rows
    }
    if len(joint_types) != 1 or not joint_types.issubset(candidate_joint_type_set):
        reasons.append("joint_types_do_not_identify_one_moving_link")

    normalized_axes = [
        normalize_axis_hint_token(row.payload.get("axis_hint")) for row in rows
    ]
    axes = {
        normalized_axis.removeprefix("+")
        for normalized_axis in normalized_axes
        if normalized_axis is not None
    }
    if any(axis is None for axis in normalized_axes) or len(axes) != 1:
        reasons.append("axes_do_not_identify_one_moving_link")

    flags: list[bool | None] = []
    for row in rows:
        raw_flag = row.payload.get("is_articulation_candidate")
        flags.append(raw_flag if isinstance(raw_flag, bool) else None)
    if flags.count(True) != 1 or any(flag is None for flag in flags):
        reasons.append("link_requires_exactly_one_candidate_anchor")
    anchors = [row for row, flag in zip(rows, flags, strict=True) if flag is True]
    if len(anchors) != 1:
        return reasons

    anchor = anchors[0]
    body0 = _single_rigger_claim_endpoint(anchor.payload, "body0")
    body1 = _single_rigger_claim_endpoint(anchor.payload, "body1")
    if body0 is None:
        reasons.append("candidate_anchor_body0_not_exact")
    if body1 != anchor.prim_path:
        reasons.append("candidate_anchor_body1_not_exact_self")
    expected_axis = next(iter(axes)) if len(axes) == 1 else None
    for row in rows:
        rigger_evidence = row.payload.get("rigger_evidence")
        if rigger_evidence is not None and not isinstance(rigger_evidence, Mapping):
            reasons.append("member_rigger_evidence_malformed")
            continue
        if not isinstance(rigger_evidence, Mapping):
            rigger_evidence = {}
        row_body0 = _single_rigger_claim_endpoint(row.payload, "body0")
        row_body1 = _single_rigger_claim_endpoint(row.payload, "body1")
        if row is not anchor:
            if "body0" in rigger_evidence and row_body0 is None:
                reasons.append("member_body0_not_exact")
            elif row_body0 is not None and row_body0 != body0:
                reasons.append("member_body0_conflicts_with_anchor")
            if "body1" in rigger_evidence and row_body1 is None:
                reasons.append("member_body1_not_exact")
            elif row_body1 is not None and row_body1 != body1:
                reasons.append("member_body1_conflicts_with_anchor")
            if rigger_evidence.get("compound_edges"):
                reasons.append("member_compound_edge_evidence_not_anchor_scoped")
            if rigger_evidence.get("limits"):
                reasons.append("member_limit_evidence_not_anchor_scoped")
        motion_axis = rigger_evidence.get("motion_axis")
        if motion_axis is not None:
            motion_axis_value = (
                motion_axis.get("value") if isinstance(motion_axis, Mapping) else None
            )
            normalized_motion_axis = normalize_axis_hint_token(motion_axis_value)
            canonical_motion_axis = (
                normalized_motion_axis.removeprefix("+")
                if normalized_motion_axis is not None
                else None
            )
            if canonical_motion_axis is None or canonical_motion_axis != expected_axis:
                reasons.append("member_motion_axis_conflicts_with_link")
        parent_hint = _clean_text(row.payload.get("parent_hint"), "unknown")
        if parent_hint.startswith("/") and parent_hint != body0:
            reasons.append("member_parent_hint_conflicts_with_anchor")
        consistency = row.payload.get("consistency")
        flagged_fields = (
            consistency.get("flagged_fields")
            if isinstance(consistency, Mapping)
            else None
        )
        if isinstance(flagged_fields, Mapping) and "axis_hint" in flagged_fields:
            reasons.append("member_axis_consistency_unresolved")
        if any(
            row.payload.get(key)
            for key in (
                "_source_support_candidate_flag_conflict",
                "_source_support_joint_type_conflict",
                "_source_support_annotation_conflicts",
            )
        ):
            reasons.append("member_source_annotation_conflict")
    return sorted(set(reasons))


def _explicit_instance_moving_members(
    payload: Mapping[str, Any],
    *,
    moving_body_prim: str,
) -> list[str]:
    raw_members = payload.get("_explicit_instance_link_members")
    if not isinstance(raw_members, list):
        return [moving_body_prim]
    members = _dedupe_preserving_order(
        _clean_text(value)
        for value in raw_members
        if _clean_text(value).startswith("/")
    )
    if moving_body_prim not in members:
        return [moving_body_prim]
    return [
        moving_body_prim,
        *[member for member in members if member != moving_body_prim],
    ]


def _single_rigger_claim_endpoint(payload: dict[str, Any], field: str) -> str | None:
    claim = _rigger_evidence_claim(payload, field)
    if claim is None:
        return None
    claim_value = _clean_text(claim.get("value"), "unknown")
    exact_paths = _rigger_claim_exact_paths(claim, claim_value=claim_value)
    if len(exact_paths) == 1:
        return exact_paths[0]
    return None


def _synthesize_body_edge_payload(
    *,
    body0: str,
    body1: str,
    rows: Sequence[_PredictionPayload],
) -> _PredictionPayload:
    payload = copy.deepcopy(rows[0].payload)
    source_prediction_ids = _dedupe_preserving_order(
        source_id for row in rows for source_id in row.source_prediction_ids
    )
    evidence: dict[str, Any] = {}
    payload["rigger_evidence"] = evidence
    _set_normalized_endpoint_claim(evidence, "body0", body0, rows=rows)
    _set_normalized_endpoint_claim(evidence, "body1", body1, rows=rows)
    _merge_synthesized_auxiliary_evidence(
        payload=payload,
        evidence=evidence,
        rows=rows,
    )
    _merge_synthesized_descriptive_fields(payload=payload, rows=rows)

    candidate_flags = {
        state
        for row in rows
        if (
            state := _articulation_candidate_flag_state(
                row.payload.get("is_articulation_candidate")
            )
        )
        is not None
    }
    payload.pop("_source_support_candidate_flag_conflict", None)
    if candidate_flags == {True}:
        payload["is_articulation_candidate"] = True
    elif candidate_flags == {False}:
        payload["is_articulation_candidate"] = False
    elif candidate_flags == {True, False}:
        payload["is_articulation_candidate"] = True
        payload["_source_support_candidate_flag_conflict"] = True
    else:
        # Missing/unknown flags are not contradictory evidence. Preserve that
        # state so the existing promotable-joint-type rule can decide whether
        # this source-backed row participates without manufacturing a conflict.
        payload.pop("is_articulation_candidate", None)

    joint_types = _dedupe_preserving_order(
        joint_type
        for row in rows
        if (joint_type := _clean_token(row.payload.get("joint_type_hint"), "unknown"))
        not in _UNKNOWN_VALUES
    )
    if len(joint_types) == 1:
        payload["joint_type_hint"] = joint_types[0]
    elif len(joint_types) > 1:
        payload["joint_type_hint"] = "unknown"
        payload["_source_support_joint_type_conflict"] = True
    else:
        payload["joint_type_hint"] = "unknown"

    _apply_body_edge_axis_consensus(
        payload,
        evidence=evidence,
        rows=rows,
        source_prediction_ids=source_prediction_ids,
    )
    payload["confidence"] = _lowest_support_confidence(rows)
    return _PredictionPayload(
        body1,
        payload,
        source_prediction_ids=source_prediction_ids,
    )


def _merge_synthesized_auxiliary_evidence(
    *,
    payload: dict[str, Any],
    evidence: dict[str, Any],
    rows: Sequence[_PredictionPayload],
) -> None:
    """Merge non-endpoint evidence without choosing a sibling row by position.

    Semantically equivalent compound edges are coalesced without losing their
    paths, rationale, conservative confidence, or source-row provenance. Truly
    distinct edges remain separate so the existing Stage 2 conflict checks see
    every explicit claim. Motion limits have a single-value Stage 2 contract,
    so equivalent claims are coalesced while distinct claims fail closed
    instead of selecting one based on prediction order. Unsupported extra
    fields survive only when their values do not conflict across sibling rows;
    unsupported conflicting values are not promoted.
    """
    payload.pop("_source_support_limit_conflict", None)
    payload.pop("_source_support_limit_conflict_values", None)

    recognized_fields = {
        "body0",
        "body1",
        "motion_axis",
        "compound_edges",
        "limits",
    }
    compound_edges_by_key: dict[
        str,
        list[tuple[dict[str, Any], str]],
    ] = {}
    limit_claims_by_value: dict[str, list[dict[str, Any]]] = {}
    extra_values_by_field: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_evidence = row.payload.get("rigger_evidence")
        if not isinstance(row_evidence, Mapping):
            continue

        for field, value in row_evidence.items():
            if not isinstance(field, str) or field in recognized_fields:
                continue
            copied_value = copy.deepcopy(value)
            extra_values_by_field.setdefault(field, {}).setdefault(
                _stable_value_key(copied_value),
                copied_value,
            )

        raw_compound_edges = row_evidence.get("compound_edges")
        if isinstance(raw_compound_edges, list):
            for raw_edge in raw_compound_edges:
                if not isinstance(raw_edge, Mapping):
                    continue
                edge = copy.deepcopy(dict(raw_edge))
                source_prediction_id = min(row.source_prediction_ids or [row.prim_path])
                compound_edges_by_key.setdefault(
                    _compound_edge_semantic_key(edge),
                    [],
                ).append((edge, source_prediction_id))

        raw_limits = row_evidence.get("limits")
        if isinstance(raw_limits, Mapping):
            limits = _canonical_limit_claim(raw_limits)
            if limits is not None:
                limit_claims_by_value.setdefault(
                    _stable_mapping_key(limits),
                    [],
                ).append(
                    {
                        **limits,
                        "rationale": _clean_text(
                            raw_limits.get("rationale") or raw_limits.get("reasoning")
                        ),
                    }
                )

    if compound_edges_by_key:
        evidence["compound_edges"] = [
            _merge_equivalent_compound_edges(compound_edges_by_key[key])
            for key in sorted(compound_edges_by_key)
        ]

    if len(limit_claims_by_value) == 1:
        equivalent_claims = next(iter(limit_claims_by_value.values()))
        merged_limits = copy.deepcopy(min(equivalent_claims, key=_stable_mapping_key))
        rationales = sorted(
            {
                rationale
                for claim in equivalent_claims
                if (rationale := _clean_text(claim.get("rationale")))
            }
        )
        if rationales:
            merged_limits["rationale"] = "; ".join(rationales)
        evidence["limits"] = merged_limits
    elif len(limit_claims_by_value) > 1:
        conflicting_limits = [
            min(limit_claims_by_value[key], key=_stable_mapping_key)
            for key in sorted(limit_claims_by_value)
        ]
        payload["_source_support_limit_conflict"] = True
        payload["_source_support_limit_conflict_values"] = [
            _format_limit_value(
                lower_limit=_optional_float(limits.get("lower_limit")),
                upper_limit=_optional_float(limits.get("upper_limit")),
                unit=_normalize_limit_unit(limits.get("unit")),
                source=_normalize_limit_source(limits.get("source")),
            )
            for limits in conflicting_limits
        ]

    for field in sorted(extra_values_by_field):
        values_by_key = extra_values_by_field[field]
        if len(values_by_key) == 1:
            evidence[field] = next(iter(values_by_key.values()))


def _merge_synthesized_descriptive_fields(
    *,
    payload: dict[str, Any],
    rows: Sequence[_PredictionPayload],
) -> None:
    """Aggregate row-local descriptions without selecting a sibling by order.

    Multiple Gprims can legitimately describe different visible pieces of one
    rigid body, so disagreements are diagnostic rather than topology blockers.
    Scalar output fields retain only a semantic consensus; all disagreements
    remain available in a machine-readable candidate field.
    """
    scalar_fields = (
        "component_name",
        "component_type",
        "role",
        "parent_hint",
        "child_hint",
    )
    conflicts: dict[str, list[str]] = {}
    for field in scalar_fields:
        values_by_token: dict[str, set[str]] = {}
        for row in rows:
            value = _clean_text(row.payload.get(field))
            if not value or value.lower() in _UNKNOWN_VALUES:
                continue
            equivalence_key = (
                value
                if field in {"parent_hint", "child_hint"} and value.startswith("/")
                else _clean_token(value)
            )
            values_by_token.setdefault(equivalence_key, set()).add(value)

        if len(values_by_token) == 1:
            equivalent_values = next(iter(values_by_token.values()))
            payload[field] = min(
                equivalent_values,
                key=lambda value: (value.casefold(), value),
            )
        elif len(values_by_token) > 1:
            payload[field] = "unknown"
            conflicts[field] = sorted(
                {
                    value
                    for equivalent_values in values_by_token.values()
                    for value in equivalent_values
                },
                key=lambda value: (value.casefold(), value),
            )
        else:
            payload[field] = "unknown"

    narratives = sorted(
        {
            narrative
            for row in rows
            for field in ("evidence", "reasoning")
            if (narrative := _clean_text(row.payload.get(field)))
            and narrative.lower() not in _UNKNOWN_VALUES
        },
        key=lambda value: (value.casefold(), value),
    )
    payload.pop("reasoning", None)
    if narratives:
        payload["evidence"] = "; ".join(narratives)
    else:
        payload.pop("evidence", None)

    if conflicts:
        payload["_source_support_annotation_conflicts"] = conflicts
    else:
        payload.pop("_source_support_annotation_conflicts", None)


def _compound_edge_semantic_key(edge: Mapping[str, Any]) -> str:
    joint_type = _clean_token(edge.get("joint_type_hint"), "unknown")
    if joint_type in _UNKNOWN_VALUES:
        joint_type = _clean_token(edge.get("motion_type"), "unknown")
    axis_hint = _normalized_axis_hint_value(
        _clean_token(edge.get("axis_hint"), "unknown")
    )
    axis_world = _motion_axis_world_from_axis_hint(axis_hint)
    return _stable_mapping_key(
        {
            "body0": _compound_edge_endpoint(edge, "body0", "fixed_parent_prim"),
            "body1": _compound_edge_endpoint(edge, "body1", "moving_body_prim"),
            "joint_type_hint": joint_type,
            "axis": axis_world if axis_world is not None else axis_hint,
        }
    )


def _merge_equivalent_compound_edges(
    entries: Sequence[tuple[dict[str, Any], str]],
) -> dict[str, Any]:
    edge = copy.deepcopy(min((value for value, _ in entries), key=_stable_mapping_key))
    prim_paths = sorted(
        {
            prim_path
            for value, _ in entries
            for prim_path in _compound_edge_exact_prim_paths(value)
        }
    )
    if prim_paths:
        edge["prim_paths"] = prim_paths
    endpoint_canonicalizations = sorted(
        {
            (
                canonicalization.endpoint,
                canonicalization.wrapper_path,
                canonicalization.leaf_path,
            )
            for value, _ in entries
            for canonicalization in _compound_edge_endpoint_canonicalizations(value)
        }
    )
    if endpoint_canonicalizations:
        edge[_HIERARCHY_ENDPOINT_CANONICALIZATIONS] = [
            {
                "field": endpoint,
                "wrapper_path": wrapper_path,
                "leaf_path": leaf_path,
            }
            for endpoint, wrapper_path, leaf_path in endpoint_canonicalizations
        ]
    else:
        edge.pop(_HIERARCHY_ENDPOINT_CANONICALIZATIONS, None)
    rationales = sorted(
        {
            rationale
            for value, _ in entries
            if (
                rationale := _clean_text(
                    value.get("rationale") or value.get("reasoning")
                )
            )
        }
    )
    if rationales:
        edge["rationale"] = "; ".join(rationales)
    edge["confidence"] = _lowest_confidence(
        value.get("confidence") for value, _ in entries
    )
    sources = {_stage2_source_from_value(value.get("source")) for value, _ in entries}
    edge["source"] = (
        next(iter(sources)) if len(sources) == 1 else "consistency_corrected"
    )
    edge["_source_prediction_ids"] = sorted(
        {source_prediction_id for _, source_prediction_id in entries}
    )
    return edge


def _canonical_limit_claim(value: Mapping[str, Any]) -> dict[str, Any] | None:
    limit_dict = dict(value)
    lower_limit = _optional_float(_first_present(limit_dict, *_LIMIT_LOWER_ALIASES))
    upper_limit = _optional_float(_first_present(limit_dict, *_LIMIT_UPPER_ALIASES))
    if lower_limit is None and upper_limit is None:
        return None
    return {
        "lower_limit": lower_limit,
        "upper_limit": upper_limit,
        "unit": _normalize_limit_unit(_first_present(limit_dict, *_LIMIT_UNIT_ALIASES)),
        "source": _normalize_limit_source(
            _first_present(limit_dict, *_LIMIT_SOURCE_ALIASES)
        ),
    }


def _stable_mapping_key(value: Mapping[str, Any]) -> str:
    return _stable_value_key(value)


def _stable_value_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _set_normalized_endpoint_claim(
    evidence: dict[str, Any],
    field: str,
    endpoint: str,
    *,
    rows: Sequence[_PredictionPayload],
) -> None:
    claims = [
        claim
        for row in rows
        if (claim := _rigger_evidence_claim(row.payload, field)) is not None
    ]
    rationales = sorted(
        {
            rationale
            for claim in claims
            if (rationale := _clean_text(claim.get("rationale")))
        }
    )
    sources = {_stage2_source_from_value(claim.get("source")) for claim in claims}
    normalized_claim: dict[str, Any] = {
        "value": endpoint,
        "prim_paths": [endpoint],
        "confidence": _lowest_confidence(claim.get("confidence") for claim in claims),
        "rationale": (
            "; ".join(rationales) if rationales else "Explicit Stage 1 endpoint claim."
        ),
        "source": (
            next(iter(sources))
            if len(sources) == 1
            else ("consistency_corrected" if sources else "predicted")
        ),
    }
    canonicalizations = {
        canonicalization
        for claim in claims
        if (canonicalization := _rigger_claim_hierarchy_canonicalization(claim))
        is not None
    }
    if len(canonicalizations) == 1:
        wrapper_path, leaf_path = next(iter(canonicalizations))
        normalized_claim[_HIERARCHY_WRAPPER_PATH] = wrapper_path
        normalized_claim[_HIERARCHY_LEAF_PATH] = leaf_path
    evidence[field] = normalized_claim


def _apply_body_edge_axis_consensus(
    payload: dict[str, Any],
    *,
    evidence: dict[str, Any],
    rows: Sequence[_PredictionPayload],
    source_prediction_ids: list[str],
) -> None:
    explicit_axis_values: list[str] = []
    non_axis_values: list[str] = []
    for row in rows:
        raw_axis_hint, _ = _stage2_axis_hint_and_source(row.payload)
        normalized_axis_hint = _normalized_axis_hint_value(raw_axis_hint)
        if normalized_axis_hint in _ALLOWED_AXIS_SET:
            explicit_axis_values.append(normalized_axis_hint)
        elif normalized_axis_hint not in _UNKNOWN_VALUES:
            non_axis_values.append(normalized_axis_hint)

    axis_world_values = {
        tuple(axis_world)
        for axis_hint in explicit_axis_values
        if (axis_world := _motion_axis_world_from_axis_hint(axis_hint)) is not None
    }
    if len(axis_world_values) == 1 and not non_axis_values:
        consensus_axis = explicit_axis_values[0]
        payload["axis_hint"] = consensus_axis
        evidence["motion_axis"] = {
            "value": consensus_axis,
            "prim_paths": source_prediction_ids,
            "confidence": _lowest_support_confidence(rows),
            "rationale": (
                "Consensus of explicit Stage 1 axis evidence for one "
                "source-backed rigid-body edge."
            ),
            "source": "consistency_corrected",
        }
        provenance = payload.setdefault("provenance", {})
        if not isinstance(provenance, dict):
            provenance = {}
            payload["provenance"] = provenance
        field_sources = provenance.setdefault("field_sources", {})
        if not isinstance(field_sources, dict):
            field_sources = {}
            provenance["field_sources"] = field_sources
        field_sources["axis_hint"] = "consistency_corrected"
        return

    if len(axis_world_values) > 1 or non_axis_values:
        payload["axis_hint"] = "unknown"
        evidence.pop("motion_axis", None)
        payload["_source_support_axis_conflict"] = True
        payload["_source_support_axis_values"] = _dedupe_preserving_order(
            [*explicit_axis_values, *non_axis_values]
        )
        return

    payload["axis_hint"] = "unknown"
    evidence.pop("motion_axis", None)


def _lowest_support_confidence(rows: Sequence[_PredictionPayload]) -> str:
    return _lowest_confidence(row.payload.get("confidence") for row in rows)


def _lowest_confidence(values: Iterable[Any]) -> str:
    confidence_rank = {"low": 0, "medium": 1, "high": 2}
    confidences = [_clean_token(value, "low") for value in values]
    return min(
        (value for value in confidences if value in confidence_rank),
        key=confidence_rank.__getitem__,
        default="low",
    )


def _candidate_unresolved_reason_codes(
    *,
    candidate_flag: bool | None,
    joint_type: str,
    joint_type_promotes: bool,
    axis_hint: str,
    motion_axis_world: list[float] | None,
    fixed_parent_prim: str | None,
    parent_resolved_to_self: bool,
    body1_evidence_present: bool,
    body1_resolved: bool,
    candidate_joint_type_set: set[str],
    compound_edge_conflict: bool = False,
) -> list[Stage2UnresolvedReasonCode]:
    reason_codes: list[Stage2UnresolvedReasonCode] = []
    if compound_edge_conflict:
        reason_codes.append("compound_edge_conflict")
    if candidate_flag is True and joint_type not in candidate_joint_type_set:
        reason_codes.append("joint_type_conflict")
    if candidate_flag is False and joint_type_promotes:
        reason_codes.append("candidate_flag_conflict")
    if body1_evidence_present and not body1_resolved:
        reason_codes.append("body1_unresolved")
    if motion_axis_world is None:
        if axis_hint in _UNKNOWN_VALUES:
            reason_codes.append("axis_missing")
        else:
            reason_codes.append("axis_non_axis_aligned")
    if fixed_parent_prim is None:
        reason_codes.append(
            "parent_self_reference" if parent_resolved_to_self else "parent_unresolved"
        )
    return reason_codes


def _axis_evidence_for_candidate(
    *,
    axis_hint: str,
    motion_axis_world: list[float] | None,
    moving_prim_path: str,
    axis_source: Stage2FieldSource,
    inferred_axis_evidence: Stage2EvidenceItem | None,
) -> list[Stage2EvidenceItem]:
    if inferred_axis_evidence is not None:
        return [inferred_axis_evidence]
    if axis_hint in _UNKNOWN_VALUES:
        return []
    if motion_axis_world is None:
        return [
            Stage2EvidenceItem(
                source=axis_source,
                description="Stage 1 axis evidence is present but not axis-aligned.",
                value=axis_hint,
                prim_paths=[moving_prim_path],
            )
        ]
    return [
        Stage2EvidenceItem(
            source=axis_source,
            description="Motion axis resolved from explicit axis-aligned evidence.",
            value=axis_hint,
            prim_paths=[moving_prim_path],
        )
    ]


def _connectivity_evidence_for_candidate(
    *,
    moving_prim_path: str,
    fixed_parent_prim: str | None,
    parent_hint: str,
    parent_resolved_to_self: bool,
    parent_resolution_source: Stage2ParentResolutionSource,
    explicit_parent_evidence: list[Stage2EvidenceItem],
    explicit_body1_evidence: list[Stage2EvidenceItem],
) -> list[Stage2EvidenceItem]:
    evidence: list[Stage2EvidenceItem] = []
    if explicit_parent_evidence:
        evidence.extend(explicit_parent_evidence)
    elif fixed_parent_prim is not None:
        source = cast(Stage2FieldSource, parent_resolution_source)
        evidence.append(
            Stage2EvidenceItem(
                source=source,
                description="Fixed parent resolved from Stage 1 parent hint.",
                value=fixed_parent_prim,
                prim_paths=[fixed_parent_prim, moving_prim_path],
                connectivity_role="body0_body1_edge",
            )
        )
    elif parent_resolved_to_self:
        evidence.append(
            Stage2EvidenceItem(
                source="stage1_hint",
                description="Parent hint pointed to the moving prim.",
                value="self_reference",
                prim_paths=[moving_prim_path],
            )
        )
    else:
        evidence.extend(
            _unresolved_parent_hint_evidence(
                moving_prim_path=moving_prim_path,
                parent_hint=parent_hint,
            )
        )
    evidence.extend(explicit_body1_evidence)
    return evidence


_SOURCE_BACKED_LIMIT_SOURCES = frozenset(
    {
        "authored_metadata",
        "authored_reference",
        "source_metadata",
        "accepted_manifest",
        "template_default",
    }
)
_LIMIT_LOWER_ALIASES = ("lower_limit", "lower", "lowerLimit")
_LIMIT_UPPER_ALIASES = ("upper_limit", "upper", "upperLimit")
_LIMIT_UNIT_ALIASES = ("unit", "limit_unit", "limitUnit")
_LIMIT_SOURCE_ALIASES = ("source", "limit_source", "limitSource")


def _candidate_limit_resolution(
    *,
    payload: Mapping[str, Any],
    moving_prim_path: str,
    joint_type_hint: str,
) -> _LimitResolution:
    if payload.get("_source_support_limit_conflict"):
        raw_conflict_values = payload.get("_source_support_limit_conflict_values")
        conflict_values = (
            [str(value) for value in raw_conflict_values]
            if isinstance(raw_conflict_values, list)
            else []
        )
        return _rejected_limit_resolution(
            readiness="rejected_conflicting_evidence",
            evidence_source="consistency_corrected",
            description=(
                "Stage 1 rows for the same source-backed body edge reported "
                "conflicting motion limits; no numeric limits were selected."
            ),
            limit_value=" | ".join(conflict_values) or "multiple distinct claims",
            moving_prim_path=moving_prim_path,
        )

    raw_limits = _stage1_limit_evidence(payload)
    if raw_limits is None:
        return _empty_limit_resolution()

    lower_limit = _optional_float(_first_present(raw_limits, *_LIMIT_LOWER_ALIASES))
    upper_limit = _optional_float(_first_present(raw_limits, *_LIMIT_UPPER_ALIASES))
    if lower_limit is None and upper_limit is None:
        return _empty_limit_resolution()

    limit_source = _normalize_limit_source(
        _first_present(raw_limits, *_LIMIT_SOURCE_ALIASES)
    )
    limit_unit = _normalize_limit_unit(_first_present(raw_limits, *_LIMIT_UNIT_ALIASES))
    limit_value = _format_limit_value(
        lower_limit=lower_limit,
        upper_limit=upper_limit,
        unit=limit_unit,
        source=limit_source,
    )
    evidence_source = _stage2_source_from_value(limit_source)
    if limit_source not in _SOURCE_BACKED_LIMIT_SOURCES:
        return _rejected_limit_resolution(
            readiness="rejected_untrusted_source",
            evidence_source=evidence_source,
            description=(
                "Stage 1 limit evidence was not copied because the source is not "
                "authored/source metadata, authored reference, accepted manifest, "
                "or an accepted template."
            ),
            limit_value=limit_value,
            moving_prim_path=moving_prim_path,
        )

    if limit_unit == "unknown":
        return _rejected_limit_resolution(
            readiness="rejected_missing_unit",
            evidence_source=evidence_source,
            description=(
                "Stage 1 limit evidence was not copied because the unit is missing "
                "or not recognized."
            ),
            limit_value=limit_value,
            moving_prim_path=moving_prim_path,
        )

    if joint_type_hint not in {"prismatic", "revolute", "spherical"}:
        return _rejected_limit_resolution(
            readiness="rejected_unsupported_joint_type",
            evidence_source=evidence_source,
            description=(
                "Stage 1 limit evidence was not copied because the joint type does "
                "not support source-backed limits."
            ),
            limit_value=limit_value,
            moving_prim_path=moving_prim_path,
        )

    if not _limit_unit_matches_joint_type(limit_unit, joint_type_hint):
        return _rejected_limit_resolution(
            readiness="rejected_unit_mismatch",
            evidence_source=evidence_source,
            description=(
                "Stage 1 limit evidence was not copied because the unit does not "
                "match the joint type."
            ),
            limit_value=limit_value,
            moving_prim_path=moving_prim_path,
        )

    if (
        lower_limit is not None
        and upper_limit is not None
        and lower_limit > upper_limit
    ):
        return _rejected_limit_resolution(
            readiness="rejected_invalid_range",
            evidence_source=evidence_source,
            description=(
                "Stage 1 limit evidence was not copied because lower_limit is "
                "greater than upper_limit."
            ),
            limit_value=limit_value,
            moving_prim_path=moving_prim_path,
        )

    lower_limit, upper_limit, limit_unit = _canonical_stage2_limit_values(
        lower_limit=lower_limit,
        upper_limit=upper_limit,
        unit=limit_unit,
        joint_type_hint=joint_type_hint,
    )
    limit_value = _format_limit_value(
        lower_limit=lower_limit,
        upper_limit=upper_limit,
        unit=limit_unit,
        source=limit_source,
    )
    return _LimitResolution(
        lower_limit=lower_limit,
        upper_limit=upper_limit,
        unit=limit_unit,
        source=evidence_source,
        readiness="source_backed",
        evidence=[
            Stage2EvidenceItem(
                source=evidence_source,
                description=(
                    "Motion limits copied from authored/source Stage 1 evidence."
                ),
                value=limit_value,
                prim_paths=[moving_prim_path],
            )
        ],
    )


def _empty_limit_resolution() -> _LimitResolution:
    return _LimitResolution(
        lower_limit=None,
        upper_limit=None,
        unit="unknown",
        source="unknown",
        readiness="not_provided",
        evidence=[],
    )


def _rejected_limit_resolution(
    *,
    readiness: Stage2LimitReadiness,
    evidence_source: Stage2FieldSource,
    description: str,
    limit_value: str,
    moving_prim_path: str,
) -> _LimitResolution:
    return _LimitResolution(
        lower_limit=None,
        upper_limit=None,
        unit="unknown",
        source="unknown",
        readiness=readiness,
        evidence=[
            Stage2EvidenceItem(
                source=evidence_source,
                description=description,
                value=limit_value,
                prim_paths=[moving_prim_path],
            )
        ],
    )


def _stage1_limit_evidence(payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    evidence = payload.get("rigger_evidence")
    if not isinstance(evidence, Mapping):
        return None
    limits = evidence.get("limits")
    if not isinstance(limits, Mapping):
        return None
    return limits


def _normalize_limit_source(value: Any) -> str:
    source = _clean_alias_token(value, "unknown")
    if source in {"mjcf", "urdf", "usd", "source", "source_metadata"}:
        return "source_metadata"
    if source in {"authored", "authored_metadata", "authored_usd"}:
        return "authored_metadata"
    if source in {"reference", "authored_reference", "rigged_reference"}:
        return "authored_reference"
    if source in {"manifest", "accepted_manifest"}:
        return "accepted_manifest"
    if source in {"template", "accepted_template", "template_default"}:
        return "template_default"
    if source in {"predicted", "vlm", "llm", "visual", "mesh", "static_mesh"}:
        return "predicted"
    return "unknown"


def _normalize_limit_unit(value: Any) -> str:
    unit = _clean_alias_token(value, "unknown")
    if unit in {"degree", "degrees", "deg", "degs"}:
        return "degrees"
    if unit in {"radian", "radians", "rad", "rads"}:
        return "radians"
    if unit in {"meter", "meters", "metre", "metres", "m"}:
        return "meters"
    return "unknown"


def _canonical_stage2_limit_values(
    *,
    lower_limit: float | None,
    upper_limit: float | None,
    unit: str,
    joint_type_hint: str,
) -> tuple[float | None, float | None, str]:
    if joint_type_hint in {"revolute", "spherical"} and unit == "radians":
        return (
            _canonical_degrees_from_radians(lower_limit),
            _canonical_degrees_from_radians(upper_limit),
            "degrees",
        )
    return lower_limit, upper_limit, unit


def _canonical_degrees_from_radians(value: float | None) -> float | None:
    if value is None:
        return None
    degrees = math.degrees(value)
    nearest_whole_degree = round(degrees)
    if math.isclose(degrees, nearest_whole_degree, abs_tol=1e-3):
        return float(nearest_whole_degree)
    return degrees


def _limit_unit_matches_joint_type(unit: str, joint_type_hint: str) -> bool:
    if joint_type_hint == "prismatic":
        return unit == "meters"
    if joint_type_hint in {"revolute", "spherical"}:
        return unit in {"degrees", "radians"}
    return False


def _format_limit_value(
    *,
    lower_limit: float | None,
    upper_limit: float | None,
    unit: str,
    source: str,
) -> str:
    return (
        f"lower_limit={_format_optional_float(lower_limit)}, "
        f"upper_limit={_format_optional_float(upper_limit)}, "
        f"unit={unit}, source={source}"
    )


def _format_optional_float(value: float | None) -> str:
    return "unknown" if value is None else f"{value:g}"


def _collect_compound_edge_evidence(
    *,
    payloads: Sequence[_PredictionPayload],
    known_prim_paths: set[str],
    prediction_prim_paths: set[str],
    source_structure_index: _SourceStructureIndex,
    candidate_joint_type_set: set[str],
) -> _CompoundEdgeCollection:
    edges_by_key: dict[tuple[str, str], _CompoundEdgeResolution] = {}
    keys_by_unordered_endpoints: dict[frozenset[str], tuple[str, str]] = {}
    conflicting_keys: set[tuple[str, str]] = set()
    conflict_evidence_by_body1: dict[str, list[Stage2EvidenceItem]] = {}
    conflict_source_prediction_ids_by_body1: dict[str, list[str]] = {}
    conflict_candidate_body1_paths: set[str] = set()
    for row in payloads:
        evidence = row.payload.get("rigger_evidence")
        if not isinstance(evidence, Mapping):
            continue
        raw_edges = evidence.get("compound_edges")
        if not isinstance(raw_edges, list):
            continue
        for raw_edge in raw_edges:
            source_prediction_ids = _compound_edge_source_prediction_ids(
                raw_edge,
                row=row,
            )
            edge_known_prim_paths = _compound_edge_known_prim_paths(
                raw_edge=raw_edge,
                default_known_prim_paths=known_prim_paths,
                prediction_prim_paths=prediction_prim_paths,
                source_structure_index=source_structure_index,
                source_prediction_ids=source_prediction_ids,
            )
            edge = _compound_edge_resolution(
                raw_edge,
                source_prediction_ids=source_prediction_ids,
                known_prim_paths=edge_known_prim_paths,
                candidate_joint_type_set=candidate_joint_type_set,
            )
            if edge is None:
                continue
            unordered_key = frozenset((edge.body0, edge.body1))
            existing_unordered_key = keys_by_unordered_endpoints.get(unordered_key)
            if (
                existing_unordered_key is not None
                and existing_unordered_key != edge.key
            ):
                conflicting_keys.add(existing_unordered_key)
                conflicting_keys.add(edge.key)
                existing_edge = edges_by_key.get(existing_unordered_key)
                if existing_edge is not None:
                    _record_compound_edge_collection_conflict(
                        conflict_evidence_by_body1,
                        conflict_source_prediction_ids_by_body1,
                        compound_edges=(existing_edge, edge),
                        description=(
                            "Stage 1 compound edge evidence has reversed "
                            "endpoint direction conflicts."
                        ),
                    )
                continue
            keys_by_unordered_endpoints[unordered_key] = edge.key
            existing_edge = edges_by_key.get(edge.key)
            if existing_edge is None:
                edges_by_key[edge.key] = edge
                continue
            if existing_edge.joint_type_hint != edge.joint_type_hint:
                conflicting_keys.add(edge.key)
                _record_compound_edge_collection_conflict(
                    conflict_evidence_by_body1,
                    conflict_source_prediction_ids_by_body1,
                    compound_edges=(existing_edge, edge),
                    description=(
                        "Stage 1 compound edge evidence has duplicate endpoint "
                        "claims with conflicting joint types."
                    ),
                )
                conflict_candidate_body1_paths.add(edge.body1)
                continue
            combined_source_prediction_ids = tuple(
                _dedupe_preserving_order(
                    [
                        *existing_edge.source_prediction_ids,
                        *edge.source_prediction_ids,
                    ]
                )
            )
            if _axis_hints_equivalent(existing_edge.axis_hint, edge.axis_hint):
                edges_by_key[edge.key] = existing_edge._replace(
                    source_prediction_ids=combined_source_prediction_ids
                )
                continue
            if existing_edge.axis_hint in _UNKNOWN_VALUES:
                edges_by_key[edge.key] = edge._replace(
                    source_prediction_ids=combined_source_prediction_ids
                )
                continue
            if edge.axis_hint in _UNKNOWN_VALUES:
                edges_by_key[edge.key] = existing_edge._replace(
                    source_prediction_ids=combined_source_prediction_ids
                )
                continue
            conflicting_keys.add(edge.key)
            _record_compound_edge_collection_conflict(
                conflict_evidence_by_body1,
                conflict_source_prediction_ids_by_body1,
                compound_edges=(existing_edge, edge),
                description=(
                    "Stage 1 compound edge evidence has duplicate endpoint "
                    "claims with conflicting axes."
                ),
            )
            conflict_candidate_body1_paths.add(edge.body1)

    by_body1: dict[str, list[_CompoundEdgeResolution]] = {}
    for key, edge in edges_by_key.items():
        if key in conflicting_keys:
            continue
        by_body1.setdefault(edge.body1, []).append(edge)
    return _CompoundEdgeCollection(
        edges_by_body1=by_body1,
        conflict_evidence_by_body1=conflict_evidence_by_body1,
        conflict_source_prediction_ids_by_body1=(
            conflict_source_prediction_ids_by_body1
        ),
        conflict_candidate_body1_paths=conflict_candidate_body1_paths,
    )


def _compound_edge_source_prediction_ids(
    raw_edge: Any,
    *,
    row: _PredictionPayload,
) -> tuple[str, ...]:
    """Return only source rows that can actually own this edge.

    ``_source_prediction_ids`` is internal provenance added while equivalent
    evidence is merged. Treating model-provided values as authority would let
    one row borrow another row's bounded hierarchy vocabulary, so unknown IDs
    are discarded and the edge remains scoped to its real supporting row(s).
    """
    row_source_ids = tuple(
        _dedupe_preserving_order(
            source_id for source_id in row.source_prediction_ids if source_id
        )
    )
    if isinstance(raw_edge, Mapping):
        raw_source_ids = raw_edge.get("_source_prediction_ids")
        if isinstance(raw_source_ids, list):
            row_source_id_set = set(row_source_ids)
            source_ids = tuple(
                sorted(
                    {
                        source_id
                        for value in raw_source_ids
                        if (source_id := _clean_text(value))
                        and source_id in row_source_id_set
                    }
                )
            )
            if source_ids:
                return source_ids
    return row_source_ids


def _compound_edge_known_prim_paths(
    *,
    raw_edge: Any,
    default_known_prim_paths: set[str],
    prediction_prim_paths: set[str],
    source_structure_index: _SourceStructureIndex,
    source_prediction_ids: Sequence[str],
) -> set[str]:
    """Return the endpoint vocabulary trusted for one compound edge.

    Authored rigid-body metadata intentionally exports one shared endpoint
    vocabulary. Legacy rows without prepared structure retain their historical
    global known-path behavior. Hierarchy metadata is different: its bounded
    Xform choices belong to individual rendered rows, so an edge is valid only
    in the intersection of the vocabularies of the rows that actually emitted
    it. The exact claimed body1 Gprim is additionally admitted only when
    source-exported ancestry proves that every emitting row and that target
    share one exact nearest Xform. Sibling Gprims are never added as general
    endpoint choices, so this exception cannot validate a sibling body0.
    """
    if source_structure_index.structure_mode == "conflict":
        return set()
    if source_structure_index.structure_mode == "rigid_body":
        return set(source_structure_index.endpoint_paths)
    if source_structure_index.structure_mode == "legacy":
        return set(default_known_prim_paths)
    if not source_prediction_ids:
        return set()

    endpoint_index = source_structure_index.hierarchy_endpoint_paths_by_prim_path or {}
    shared_paths: set[str] | None = None
    for source_prediction_id in source_prediction_ids:
        hierarchy_paths = endpoint_index.get(source_prediction_id, ())
        transparent_leaf_by_wrapper = (
            source_structure_index.transparent_leaf_aliases_for([source_prediction_id])
        )
        row_paths = {
            source_prediction_id,
            *hierarchy_paths,
            *(
                transparent_leaf_by_wrapper[wrapper_path]
                for wrapper_path in hierarchy_paths
                if wrapper_path in transparent_leaf_by_wrapper
            ),
        }
        if shared_paths is None:
            shared_paths = row_paths
        else:
            shared_paths.intersection_update(row_paths)

    shared_nearest_ancestor = source_structure_index.shared_nearest_hierarchy_ancestor(
        source_prediction_ids
    )
    if shared_nearest_ancestor is not None and shared_paths is not None:
        nearest_index = (
            source_structure_index.hierarchy_nearest_ancestor_by_prim_path or {}
        )
        claimed_body1 = (
            _compound_edge_endpoint(raw_edge, "body1", "moving_body_prim")
            if isinstance(raw_edge, Mapping)
            else "unknown"
        )
        if (
            claimed_body1 in prediction_prim_paths
            and nearest_index.get(claimed_body1) == shared_nearest_ancestor
        ):
            shared_paths.add(claimed_body1)
    return shared_paths or set()


def _record_compound_edge_collection_conflict(
    conflict_evidence_by_body1: dict[str, list[Stage2EvidenceItem]],
    conflict_source_prediction_ids_by_body1: dict[str, list[str]],
    *,
    compound_edges: Sequence[_CompoundEdgeResolution],
    description: str,
) -> None:
    evidence = Stage2EvidenceItem(
        source="stage1_rigger_evidence",
        description=description,
        value="; ".join(
            f"{edge.body0}->{edge.body1} {edge.joint_type_hint}/{edge.axis_hint}"
            for edge in compound_edges
        ),
        prim_paths=_dedupe_preserving_order(
            prim_path for edge in compound_edges for prim_path in edge.prim_paths
        ),
    )
    canonicalization_evidence = _endpoint_canonicalizations_evidence(
        canonicalization
        for edge in compound_edges
        for canonicalization in edge.endpoint_canonicalizations
    )
    for body1 in _dedupe_preserving_order(edge.body1 for edge in compound_edges):
        conflict_evidence_by_body1.setdefault(body1, []).extend(
            [evidence, *canonicalization_evidence]
        )
        conflict_source_prediction_ids_by_body1[body1] = _dedupe_preserving_order(
            [
                *conflict_source_prediction_ids_by_body1.get(body1, ()),
                *(
                    source_prediction_id
                    for edge in compound_edges
                    for source_prediction_id in edge.source_prediction_ids
                ),
            ]
        )


def _compound_edge_resolution(
    raw_edge: Any,
    *,
    source_prediction_ids: Sequence[str],
    known_prim_paths: set[str],
    candidate_joint_type_set: set[str],
) -> _CompoundEdgeResolution | None:
    if not isinstance(raw_edge, Mapping):
        return None
    joint_type_hint = _clean_token(raw_edge.get("joint_type_hint"), "unknown")
    if joint_type_hint in _UNKNOWN_VALUES:
        joint_type_hint = _clean_token(raw_edge.get("motion_type"), "unknown")
    if joint_type_hint not in candidate_joint_type_set:
        return None
    body0 = _compound_edge_endpoint(raw_edge, "body0", "fixed_parent_prim")
    body1 = _compound_edge_endpoint(raw_edge, "body1", "moving_body_prim")
    if (
        not body0.startswith("/")
        or not body1.startswith("/")
        or body0 == body1
        or body0 not in known_prim_paths
        or body1 not in known_prim_paths
    ):
        return None
    raw_axis_hint = _clean_token(raw_edge.get("axis_hint"), "unknown")
    axis_hint = _normalized_axis_hint_value(raw_axis_hint)
    return _CompoundEdgeResolution(
        body0=body0,
        body1=body1,
        joint_type_hint=joint_type_hint,
        raw_axis_hint=raw_axis_hint,
        axis_hint=axis_hint,
        source=_stage2_source_from_value(raw_edge.get("source")),
        confidence=_candidate_confidence(
            raw_edge.get("confidence"),
            unresolved_questions=[],
        ),
        rationale=_clean_text(raw_edge.get("rationale") or raw_edge.get("reasoning")),
        prim_paths=_dedupe_preserving_order(
            [
                body1,
                body0,
                *_compound_edge_exact_prim_paths(raw_edge),
            ]
        ),
        source_prediction_ids=tuple(source_prediction_ids),
        endpoint_canonicalizations=_compound_edge_endpoint_canonicalizations(raw_edge),
    )


def _compound_edge_endpoint(
    edge: Mapping[str, Any],
    field: str,
    alias: str,
) -> str:
    value = _compound_edge_endpoint_value(edge.get(field))
    if value.lower() not in _UNKNOWN_VALUES:
        return value
    alias_value = _compound_edge_endpoint_value(edge.get(alias))
    if alias_value.lower() not in _UNKNOWN_VALUES:
        return alias_value
    return value


def _compound_edge_endpoint_value(value: Any) -> str:
    if isinstance(value, Mapping):
        value = value.get("value")
    return _clean_text(value, "unknown")


def _compound_edge_exact_prim_paths(edge: Mapping[str, Any]) -> list[str]:
    raw_paths = edge.get("prim_paths")
    if isinstance(raw_paths, list):
        candidates = raw_paths
    else:
        candidates = [raw_paths]
    return [
        prim_path
        for prim_path in (_clean_text(value) for value in candidates)
        if prim_path.startswith("/")
    ]


def _compound_edge_endpoint_canonicalizations(
    edge: Mapping[str, Any],
) -> tuple[_EndpointCanonicalization, ...]:
    raw_canonicalizations = edge.get(_HIERARCHY_ENDPOINT_CANONICALIZATIONS)
    if not isinstance(raw_canonicalizations, list):
        return ()

    canonicalizations: set[_EndpointCanonicalization] = set()
    for raw_canonicalization in raw_canonicalizations:
        if not isinstance(raw_canonicalization, Mapping):
            continue
        endpoint = _clean_text(raw_canonicalization.get("field"))
        if endpoint not in {"body0", "body1"}:
            continue
        wrapper_path = _clean_text(raw_canonicalization.get("wrapper_path"))
        leaf_path = _clean_text(raw_canonicalization.get("leaf_path"))
        if (
            not wrapper_path.startswith("/")
            or not leaf_path.startswith("/")
            or wrapper_path == leaf_path
        ):
            continue
        endpoint_value = _compound_edge_endpoint(
            edge,
            endpoint,
            "fixed_parent_prim" if endpoint == "body0" else "moving_body_prim",
        )
        if endpoint_value != leaf_path:
            continue
        canonicalizations.add(
            _EndpointCanonicalization(
                endpoint=cast(Literal["body0", "body1"], endpoint),
                wrapper_path=wrapper_path,
                leaf_path=leaf_path,
            )
        )
    return tuple(sorted(canonicalizations))


def _single_compound_edge_for_body1(
    moving_prim_path: str,
    compound_edges_by_body1: Mapping[str, Sequence[_CompoundEdgeResolution]],
) -> _CompoundEdgeResolution | None:
    compound_edges = compound_edges_by_body1.get(moving_prim_path, ())
    if len(compound_edges) != 1:
        return None
    return compound_edges[0]


def _compound_edge_matches_candidate(
    compound_edge: _CompoundEdgeResolution,
    *,
    joint_type: str,
    axis_hint: str,
) -> bool:
    if joint_type != compound_edge.joint_type_hint:
        return False
    normalized_axis_hint = _normalized_axis_hint_value(axis_hint)
    return (
        normalized_axis_hint in _UNKNOWN_VALUES
        or compound_edge.axis_hint in _UNKNOWN_VALUES
        or _axis_hints_equivalent(normalized_axis_hint, compound_edge.axis_hint)
    )


def _axis_hints_equivalent(left_axis_hint: str, right_axis_hint: str) -> bool:
    left_axis_world = _motion_axis_world_from_axis_hint(left_axis_hint)
    right_axis_world = _motion_axis_world_from_axis_hint(right_axis_hint)
    if left_axis_world is not None and right_axis_world is not None:
        return left_axis_world == right_axis_world
    return left_axis_hint == right_axis_hint


def _compound_edge_ambiguity_evidence(
    *,
    moving_prim_path: str,
    compound_edges: Sequence[_CompoundEdgeResolution],
) -> Stage2EvidenceItem:
    prim_paths = _dedupe_preserving_order(
        [
            moving_prim_path,
            *(
                prim_path
                for compound_edge in compound_edges
                for prim_path in compound_edge.prim_paths
            ),
        ]
    )
    values = ", ".join(compound_edge.body0 for compound_edge in compound_edges)
    return Stage2EvidenceItem(
        source="stage1_rigger_evidence",
        description=(
            "Stage 1 compound edge evidence referenced multiple candidate fixed "
            "parents for this moving body."
        ),
        value=values or "multiple_compound_edges",
        prim_paths=prim_paths,
    )


def _compound_edge_conflict_evidence(
    compound_edge: _CompoundEdgeResolution,
    *,
    joint_type: str,
    axis_hint: str,
) -> Stage2EvidenceItem:
    normalized_axis_hint = _normalized_axis_hint_value(axis_hint)
    return Stage2EvidenceItem(
        source=compound_edge.source,
        description=(
            "Stage 1 compound edge evidence conflicts with the candidate joint "
            "type or axis hint."
        ),
        value=(
            f"candidate={joint_type}/{normalized_axis_hint}; "
            f"compound_edge={compound_edge.joint_type_hint}/{compound_edge.axis_hint}"
        ),
        prim_paths=compound_edge.prim_paths,
    )


def _compound_edge_parent_conflict_evidence(
    compound_edge: _CompoundEdgeResolution,
    *,
    fixed_parent_prim: str,
) -> Stage2EvidenceItem:
    return Stage2EvidenceItem(
        source=compound_edge.source,
        description=(
            "Stage 1 compound edge evidence conflicts with the resolved fixed parent."
        ),
        value=(
            f"candidate_parent={fixed_parent_prim}; "
            f"compound_edge_parent={compound_edge.body0}"
        ),
        prim_paths=_dedupe_preserving_order(
            [compound_edge.body1, fixed_parent_prim, *compound_edge.prim_paths]
        ),
    )


def _compound_edge_parent_evidence(
    compound_edge: _CompoundEdgeResolution,
) -> Stage2EvidenceItem:
    return Stage2EvidenceItem(
        source=compound_edge.source,
        description="Fixed parent resolved from explicit Stage 1 compound edge evidence.",
        value=compound_edge.body0,
        prim_paths=[compound_edge.body0, compound_edge.body1],
        connectivity_role="body0_body1_edge",
    )


def _compound_edge_body1_evidence(
    compound_edge: _CompoundEdgeResolution,
) -> Stage2EvidenceItem:
    return Stage2EvidenceItem(
        source=compound_edge.source,
        description="Moving body confirmed from explicit Stage 1 compound edge evidence.",
        value=compound_edge.body1,
        prim_paths=[compound_edge.body1],
        connectivity_role="body1_ownership",
    )


def _endpoint_canonicalization_evidence(
    *,
    wrapper_path: str,
    leaf_path: str,
) -> Stage2EvidenceItem:
    return Stage2EvidenceItem(
        source="consistency_corrected",
        description=(
            "Explicit Stage 1 endpoint canonicalized from a transparent source "
            "hierarchy wrapper to its sole observable prediction row."
        ),
        value=leaf_path,
        prim_paths=[wrapper_path, leaf_path],
        connectivity_role="endpoint_canonicalization",
    )


def _compound_edge_canonicalization_evidence(
    compound_edge: _CompoundEdgeResolution,
) -> list[Stage2EvidenceItem]:
    return _endpoint_canonicalizations_evidence(
        compound_edge.endpoint_canonicalizations
    )


def _endpoint_canonicalizations_evidence(
    canonicalizations: Iterable[_EndpointCanonicalization],
) -> list[Stage2EvidenceItem]:
    return [
        _endpoint_canonicalization_evidence(
            wrapper_path=canonicalization.wrapper_path,
            leaf_path=canonicalization.leaf_path,
        )
        for canonicalization in sorted(set(canonicalizations))
    ]


def _compound_edge_axis_evidence(
    compound_edge: _CompoundEdgeResolution,
) -> Stage2EvidenceItem:
    description = "Motion axis resolved from explicit Stage 1 compound edge evidence."
    value = compound_edge.axis_hint
    if compound_edge.raw_axis_hint != compound_edge.axis_hint:
        description = (
            "Motion axis normalized to a canonical Stage 2 axis token from "
            "explicit Stage 1 compound edge evidence."
        )
        value = f"{compound_edge.raw_axis_hint}->{compound_edge.axis_hint}"
    if compound_edge.axis_hint not in _ALLOWED_AXIS_SET:
        description = (
            "Stage 1 compound edge axis evidence is present but not axis-aligned."
        )
        value = compound_edge.raw_axis_hint
    return Stage2EvidenceItem(
        source=compound_edge.source,
        description=description,
        value=value,
        prim_paths=compound_edge.prim_paths,
    )


def _compound_edges_shared_axis_hint(
    compound_edges: Sequence[_CompoundEdgeResolution],
) -> str | None:
    shared_axis_hint: str | None = None
    for compound_edge in compound_edges:
        if compound_edge.axis_hint in _UNKNOWN_VALUES:
            continue
        if shared_axis_hint is None:
            shared_axis_hint = compound_edge.axis_hint
            continue
        if not _axis_hints_equivalent(shared_axis_hint, compound_edge.axis_hint):
            return None
    return shared_axis_hint


def _compound_edges_axis_evidence(
    *,
    moving_prim_path: str,
    compound_edges: Sequence[_CompoundEdgeResolution],
    axis_hint: str,
) -> Stage2EvidenceItem:
    prim_paths = _dedupe_preserving_order(
        [
            moving_prim_path,
            *(
                prim_path
                for compound_edge in compound_edges
                for prim_path in compound_edge.prim_paths
            ),
        ]
    )
    if axis_hint not in _ALLOWED_AXIS_SET:
        return Stage2EvidenceItem(
            source="stage1_rigger_evidence",
            description=(
                "Stage 1 compound edge axis evidence is present but not axis-aligned."
            ),
            value=axis_hint,
            prim_paths=prim_paths,
        )
    return Stage2EvidenceItem(
        source="stage1_rigger_evidence",
        description=(
            "Motion axis resolved from consistent Stage 1 compound edge evidence."
        ),
        value=axis_hint,
        prim_paths=prim_paths,
    )


def _candidate_moving_prim_paths(candidates: Sequence[Mapping[str, Any]]) -> set[str]:
    moving_prim_paths: set[str] = set()
    for candidate in candidates:
        moving_part_prims = candidate.get("moving_part_prims")
        if not isinstance(moving_part_prims, list):
            continue
        for moving_part_prim in moving_part_prims:
            body1 = _clean_text(moving_part_prim)
            if body1:
                moving_prim_paths.add(body1)
    return moving_prim_paths


def _compound_edge_candidate(
    *,
    candidate_id: str,
    compound_edge: _CompoundEdgeResolution,
    payload_by_prim: Mapping[str, dict[str, Any]],
    candidate_joint_type_set: set[str],
) -> dict[str, Any]:
    axis_hint = compound_edge.axis_hint
    motion_axis_world = _motion_axis_world_from_axis_hint(axis_hint)
    body1_payload = payload_by_prim.get(compound_edge.body1, {})
    candidate_flag = _articulation_candidate_flag_state(
        body1_payload.get("is_articulation_candidate")
    )
    unresolved_reason_codes = _candidate_unresolved_reason_codes(
        candidate_flag=candidate_flag,
        joint_type=compound_edge.joint_type_hint,
        joint_type_promotes=compound_edge.joint_type_hint in candidate_joint_type_set,
        axis_hint=axis_hint,
        motion_axis_world=motion_axis_world,
        fixed_parent_prim=compound_edge.body0,
        parent_resolved_to_self=False,
        body1_evidence_present=True,
        body1_resolved=True,
        candidate_joint_type_set=candidate_joint_type_set,
    )
    unresolved_questions: list[str] = []
    if "candidate_flag_conflict" in unresolved_reason_codes:
        unresolved_questions.append(
            "Confirm articulation candidate because the candidate flag and joint "
            "hint disagree."
        )
    if motion_axis_world is None:
        if "axis_non_axis_aligned" in unresolved_reason_codes:
            unresolved_questions.append(
                "Resolve the joint axis because the hint is not axis-aligned."
            )
        else:
            unresolved_questions.append("Determine the joint axis.")

    candidate_evidence = (
        compound_edge.rationale
        or _clean_text(body1_payload.get("evidence"))
        or _clean_text(body1_payload.get("reasoning"))
    )
    axis_source: Stage2FieldSource = (
        "unknown" if axis_hint in _UNKNOWN_VALUES else compound_edge.source
    )
    motion_axis_source: Stage2FieldSource = (
        compound_edge.source if motion_axis_world is not None else "unknown"
    )
    axis_evidence = (
        []
        if axis_hint in _UNKNOWN_VALUES
        else [_compound_edge_axis_evidence(compound_edge)]
    )
    limit_resolution = _candidate_limit_resolution(
        payload=body1_payload,
        moving_prim_path=compound_edge.body1,
        joint_type_hint=compound_edge.joint_type_hint,
    )
    review_status: Stage2ReviewStatus = (
        REVIEW_REQUIRED_STATUS
        if unresolved_reason_codes
        else READY_FOR_RIGGER_INPUT_STATUS
    )

    return Stage2ArticulationCandidate(
        candidate_id=candidate_id,
        motion_type=_motion_type_from_joint_hint(compound_edge.joint_type_hint),
        joint_type_hint=compound_edge.joint_type_hint,
        axis_hint=axis_hint,
        motion_axis_world=motion_axis_world,
        confidence=_candidate_confidence(
            compound_edge.confidence,
            unresolved_questions=unresolved_questions,
        ),
        moving_part_prims=[compound_edge.body1],
        fixed_parent_prim=compound_edge.body0,
        parent_resolution_source="stage1_rigger_evidence",
        parent_hint=_clean_text(body1_payload.get("parent_hint"), "unknown"),
        child_hint=_clean_text(body1_payload.get("child_hint"), "unknown"),
        component_name=_clean_text(body1_payload.get("component_name"), "unknown"),
        component_type=_clean_text(body1_payload.get("component_type"), "unknown"),
        role=_clean_token(body1_payload.get("role"), "unknown"),
        source_prediction_ids=_dedupe_preserving_order(
            [
                value
                for value in (
                    compound_edge.body1,
                    *compound_edge.source_prediction_ids,
                )
                if value
            ]
        ),
        evidence=candidate_evidence,
        field_sources={
            "motion_type": compound_edge.source,
            "axis_hint": axis_source,
            "motion_axis_world": motion_axis_source,
            "fixed_parent_prim": compound_edge.source,
        },
        axis_evidence=axis_evidence,
        connectivity_evidence=[
            _compound_edge_parent_evidence(compound_edge),
            _compound_edge_body1_evidence(compound_edge),
            *_compound_edge_canonicalization_evidence(compound_edge),
        ],
        lower_limit=limit_resolution.lower_limit,
        upper_limit=limit_resolution.upper_limit,
        limit_unit=limit_resolution.unit,
        limit_source=limit_resolution.source,
        limit_readiness=limit_resolution.readiness,
        limit_evidence=limit_resolution.evidence,
        unresolved_reason_codes=unresolved_reason_codes,
        review_status=review_status,
        unresolved_questions=unresolved_questions,
    ).model_dump(mode="json")


def _compound_edge_ambiguity_candidate(
    *,
    candidate_id: str,
    body1: str,
    compound_edges: Sequence[_CompoundEdgeResolution],
    payload_by_prim: Mapping[str, dict[str, Any]],
    conflict_evidence: Sequence[Stage2EvidenceItem] = (),
    conflict_source_prediction_ids: Sequence[str] = (),
) -> dict[str, Any]:
    body1_payload = payload_by_prim.get(body1, {})
    joint_type_hints = {
        compound_edge.joint_type_hint for compound_edge in compound_edges
    }
    joint_type_hint = (
        next(iter(joint_type_hints)) if len(joint_type_hints) == 1 else "unknown"
    )
    axis_hint = _compound_edges_shared_axis_hint(compound_edges) or "unknown"
    motion_axis_world = _motion_axis_world_from_axis_hint(axis_hint)
    unresolved_reason_codes: list[Stage2UnresolvedReasonCode] = [
        "compound_edge_conflict"
    ]
    unresolved_questions = ["Resolve the compound-edge evidence conflict."]
    if motion_axis_world is None:
        if axis_hint in _UNKNOWN_VALUES:
            unresolved_reason_codes.append("axis_missing")
            unresolved_questions.append("Determine the joint axis.")
        else:
            unresolved_reason_codes.append("axis_non_axis_aligned")
            unresolved_questions.append(
                "Resolve the joint axis because the hint is not axis-aligned."
            )
    unresolved_reason_codes.append("parent_unresolved")
    unresolved_questions.append("Resolve the fixed parent/connectivity.")

    axis_source: Stage2FieldSource = (
        "stage1_rigger_evidence" if axis_hint not in _UNKNOWN_VALUES else "unknown"
    )
    axis_evidence = (
        []
        if axis_hint in _UNKNOWN_VALUES
        else [
            _compound_edges_axis_evidence(
                moving_prim_path=body1,
                compound_edges=compound_edges,
                axis_hint=axis_hint,
            )
        ]
    )
    candidate_evidence = (
        "; ".join(
            rationale
            for rationale in (edge.rationale for edge in compound_edges)
            if rationale
        )
        or _clean_text(body1_payload.get("evidence"))
        or _clean_text(body1_payload.get("reasoning"))
    )

    connectivity_evidence = list(conflict_evidence)
    if compound_edges:
        connectivity_evidence.append(
            _compound_edge_ambiguity_evidence(
                moving_prim_path=body1,
                compound_edges=compound_edges,
            )
        )
        connectivity_evidence.extend(
            evidence
            for compound_edge in compound_edges
            for evidence in _compound_edge_canonicalization_evidence(compound_edge)
        )
    limit_resolution = _candidate_limit_resolution(
        payload=body1_payload,
        moving_prim_path=body1,
        joint_type_hint=joint_type_hint,
    )

    return Stage2ArticulationCandidate(
        candidate_id=candidate_id,
        motion_type=_motion_type_from_joint_hint(joint_type_hint),
        joint_type_hint=joint_type_hint,
        axis_hint=axis_hint,
        motion_axis_world=motion_axis_world,
        confidence=_candidate_confidence(
            body1_payload.get("confidence"),
            unresolved_questions=unresolved_questions,
        ),
        moving_part_prims=[body1],
        fixed_parent_prim=None,
        parent_resolution_source="unresolved",
        parent_hint=_clean_text(body1_payload.get("parent_hint"), "unknown"),
        child_hint=_clean_text(body1_payload.get("child_hint"), "unknown"),
        component_name=_clean_text(body1_payload.get("component_name"), "unknown"),
        component_type=_clean_text(body1_payload.get("component_type"), "unknown"),
        role=_clean_token(body1_payload.get("role"), "unknown"),
        source_prediction_ids=_dedupe_preserving_order(
            [
                value
                for value in (
                    body1,
                    *(
                        source_prediction_id
                        for compound_edge in compound_edges
                        for source_prediction_id in (
                            compound_edge.source_prediction_ids
                        )
                    ),
                    *conflict_source_prediction_ids,
                )
                if value
            ]
        ),
        evidence=candidate_evidence,
        field_sources={
            "motion_type": (
                "stage1_rigger_evidence"
                if joint_type_hint not in _UNKNOWN_VALUES
                else "unknown"
            ),
            "axis_hint": axis_source,
            "motion_axis_world": (
                "stage1_rigger_evidence" if motion_axis_world is not None else "unknown"
            ),
            "fixed_parent_prim": "unknown",
        },
        axis_evidence=axis_evidence,
        connectivity_evidence=connectivity_evidence,
        lower_limit=limit_resolution.lower_limit,
        upper_limit=limit_resolution.upper_limit,
        limit_unit=limit_resolution.unit,
        limit_source=limit_resolution.source,
        limit_readiness=limit_resolution.readiness,
        limit_evidence=limit_resolution.evidence,
        unresolved_reason_codes=unresolved_reason_codes,
        review_status=REVIEW_REQUIRED_STATUS,
        unresolved_questions=unresolved_questions,
    ).model_dump(mode="json")


def _unresolved_parent_hint_evidence(
    *,
    moving_prim_path: str,
    parent_hint: str,
) -> list[Stage2EvidenceItem]:
    if not parent_hint or parent_hint.lower() in _UNKNOWN_VALUES:
        return []
    if parent_hint.startswith("/"):
        return [
            Stage2EvidenceItem(
                source="stage1_hint",
                description=(
                    "Stage 1 parent hint referenced a prim path that is not "
                    "present in the prediction rows."
                ),
                value=parent_hint,
                prim_paths=[moving_prim_path],
            )
        ]
    return [
        Stage2EvidenceItem(
            source="stage1_hint",
            description=(
                "Stage 1 parent hint did not provide an exact prim path; "
                "left parent/connectivity unresolved."
            ),
            value=parent_hint,
            prim_paths=[moving_prim_path],
        )
    ]


def _normalize_stage2_axis_hint(
    axis_hint: str,
    *,
    moving_prim_path: str,
    axis_source: Stage2FieldSource,
) -> tuple[str, Stage2EvidenceItem | None]:
    normalized = _normalized_axis_hint_value(axis_hint)
    if normalized == axis_hint:
        return axis_hint, None
    return (
        normalized,
        Stage2EvidenceItem(
            source=axis_source,
            description="Motion axis normalized to a canonical Stage 2 axis token.",
            value=f"{axis_hint}->{normalized}",
            prim_paths=[moving_prim_path],
        ),
    )


def _normalized_axis_hint_value(axis_hint: str) -> str:
    if axis_hint in _ALLOWED_AXIS_SET or axis_hint in _UNKNOWN_VALUES:
        return axis_hint
    return normalize_axis_hint_token(axis_hint) or axis_hint


def _build_parent_index(
    payloads: Sequence[_PredictionPayload],
    *,
    additional_known_prim_paths: Iterable[str] = (),
) -> tuple[dict[str, set[str]], set[str]]:
    index: dict[str, set[str]] = {}
    known_prim_paths: set[str] = {
        prim_path
        for prim_path in additional_known_prim_paths
        if isinstance(prim_path, str) and prim_path.startswith("/")
    }

    def add_label(label: str, prim_path: str) -> None:
        key = _label_key(label)
        if key:
            index.setdefault(key, set()).add(prim_path)

    for row in payloads:
        if row.prim_path:
            known_prim_paths.add(row.prim_path)
            add_label(row.prim_path, row.prim_path)
            payload = row.payload
            for field in ("component_name", "component_type", "role", "child_hint"):
                label = _clean_text(payload.get(field))
                if label and label.lower() not in _UNKNOWN_VALUES:
                    add_label(label, row.prim_path)
    return index, known_prim_paths


def _resolve_fixed_parent_prim(
    *,
    payload: dict[str, Any],
    parent_hint: str,
    known_prim_paths: set[str],
    rigger_body0_paths: set[str],
    moving_prim_path: str,
) -> _ParentResolution:
    rigger_resolution = _resolve_rigger_body0_prim(
        payload=payload,
        known_prim_paths=known_prim_paths,
        allowed_body0_paths=rigger_body0_paths,
        moving_prim_path=moving_prim_path,
    )
    # Structured body0 evidence is intentionally stricter than parent_hint:
    # when present but ambiguous or incomplete, it blocks weaker fallback hints.
    if rigger_resolution.evidence_present:
        return rigger_resolution

    fixed_parent_prim, parent_resolution_source = _resolve_parent_prim(
        parent_hint,
        known_prim_paths,
    )
    if fixed_parent_prim == moving_prim_path:
        return _ParentResolution(
            fixed_parent_prim=None,
            source="unresolved",
            parent_resolved_to_self=True,
            evidence=[],
            evidence_present=False,
        )
    return _ParentResolution(
        fixed_parent_prim=fixed_parent_prim,
        source=parent_resolution_source,
        parent_resolved_to_self=False,
        evidence=[],
        evidence_present=False,
    )


def _resolve_rigger_body0_prim(
    *,
    payload: dict[str, Any],
    known_prim_paths: set[str],
    moving_prim_path: str,
    allowed_body0_paths: set[str],
) -> _ParentResolution:
    claim = _rigger_evidence_claim(payload, "body0")
    if claim is None:
        return _ParentResolution(None, "unresolved", False, [], False)

    claim_value = _clean_text(claim.get("value"), "unknown")
    evidence_source = _rigger_claim_stage2_source(claim)
    exact_paths = _rigger_claim_exact_paths(claim, claim_value=claim_value)
    hierarchy_canonicalization = _rigger_claim_hierarchy_canonicalization(claim)
    canonicalization_evidence = (
        [
            _endpoint_canonicalization_evidence(
                wrapper_path=hierarchy_canonicalization[0],
                leaf_path=hierarchy_canonicalization[1],
            )
        ]
        if hierarchy_canonicalization is not None
        else []
    )
    evidence_prim_paths = _dedupe_preserving_order([moving_prim_path, *exact_paths])
    evidence_value = claim_value if claim_value.lower() not in _UNKNOWN_VALUES else None

    if len(exact_paths) == 1:
        candidate_parent = exact_paths[0]
        if (
            candidate_parent in known_prim_paths
            and candidate_parent == moving_prim_path
        ):
            return _ParentResolution(
                fixed_parent_prim=None,
                source="unresolved",
                parent_resolved_to_self=True,
                evidence=[
                    Stage2EvidenceItem(
                        source=evidence_source,
                        description=(
                            "Stage 1 rigger body0 evidence pointed to the moving prim."
                        ),
                        value="self_reference",
                        prim_paths=[moving_prim_path],
                    ),
                    *canonicalization_evidence,
                ],
                evidence_present=True,
            )

        if candidate_parent in known_prim_paths and (
            _body0_path_is_authorized(
                candidate_parent,
                authorized_body0_paths=allowed_body0_paths,
                hierarchy_canonicalization=hierarchy_canonicalization,
            )
        ):
            return _ParentResolution(
                fixed_parent_prim=candidate_parent,
                source="stage1_rigger_evidence",
                parent_resolved_to_self=False,
                evidence=[
                    Stage2EvidenceItem(
                        source=evidence_source,
                        description=(
                            "Fixed parent resolved from explicit Stage 1 "
                            "rigger body0 evidence."
                        ),
                        value=candidate_parent,
                        prim_paths=[candidate_parent, moving_prim_path],
                        connectivity_role="body0_body1_edge",
                    ),
                    *canonicalization_evidence,
                ],
                evidence_present=True,
            )
        description = (
            "Stage 1 rigger body0 evidence referenced a known prim path that "
            "is not in the allowed fixed-body endpoint vocabulary for this "
            "source-backed prediction row."
            if candidate_parent in known_prim_paths
            else (
                "Stage 1 rigger body0 evidence referenced a prim path that is "
                "not present in the prediction rows."
            )
        )
        return _ParentResolution(
            fixed_parent_prim=None,
            source="unresolved",
            parent_resolved_to_self=False,
            evidence=[
                Stage2EvidenceItem(
                    source=evidence_source,
                    description=description,
                    value=exact_paths[0],
                    prim_paths=evidence_prim_paths,
                ),
                *canonicalization_evidence,
            ],
            evidence_present=True,
        )

    description = (
        "Stage 1 rigger body0 evidence referenced multiple candidate parent "
        "prim paths; left parent/connectivity unresolved."
        if exact_paths
        else (
            "Stage 1 rigger body0 evidence did not provide an exact prim path; "
            "left parent/connectivity unresolved."
        )
    )
    return _ParentResolution(
        fixed_parent_prim=None,
        source="unresolved",
        parent_resolved_to_self=False,
        evidence=[
            Stage2EvidenceItem(
                source=evidence_source,
                description=description,
                value=evidence_value,
                prim_paths=evidence_prim_paths,
            ),
            *canonicalization_evidence,
        ],
        evidence_present=True,
    )


def _resolve_rigger_body1_prim(
    *,
    payload: dict[str, Any],
    valid_endpoint_paths: set[str],
    moving_prim_path: str,
    source_owner_prim: str | None = None,
    rigid_endpoint_vocabulary_present: bool = False,
    hierarchy_ancestor_paths: set[str] | None = None,
) -> _Body1Resolution:
    claim = _rigger_evidence_claim(payload, "body1")
    if claim is None:
        return _Body1Resolution([], False, True)

    claim_value = _clean_text(claim.get("value"), "unknown")
    evidence_source = _rigger_claim_stage2_source(claim)
    exact_paths = _rigger_claim_exact_paths(claim, claim_value=claim_value)
    hierarchy_canonicalization = _rigger_claim_hierarchy_canonicalization(claim)
    canonicalization_evidence = (
        [
            _endpoint_canonicalization_evidence(
                wrapper_path=hierarchy_canonicalization[0],
                leaf_path=hierarchy_canonicalization[1],
            )
        ]
        if hierarchy_canonicalization is not None
        else []
    )
    evidence_prim_paths = _dedupe_preserving_order([moving_prim_path, *exact_paths])
    evidence_value = claim_value if claim_value.lower() not in _UNKNOWN_VALUES else None

    if len(exact_paths) == 1:
        body1_prim = exact_paths[0]
        if rigid_endpoint_vocabulary_present:
            allowed_body1_paths = {moving_prim_path, source_owner_prim}
        else:
            allowed_body1_paths = {
                moving_prim_path,
                *(hierarchy_ancestor_paths or set()),
            }
        if body1_prim in valid_endpoint_paths and body1_prim in allowed_body1_paths:
            return _Body1Resolution(
                evidence=[
                    Stage2EvidenceItem(
                        source=evidence_source,
                        description=(
                            "Moving body confirmed from explicit Stage 1 rigger "
                            "body1 evidence and source-authored rigid-body ownership."
                            if rigid_endpoint_vocabulary_present
                            and source_owner_prim is not None
                            else (
                                "Moving body confirmed from explicit Stage 1 "
                                "rigger body1 evidence and the exported source "
                                "hierarchy."
                                if hierarchy_ancestor_paths
                                else (
                                    "Moving body confirmed from explicit Stage 1 "
                                    "rigger body1 evidence."
                                )
                            )
                        ),
                        value=body1_prim,
                        prim_paths=_dedupe_preserving_order(
                            [body1_prim, moving_prim_path]
                        ),
                        connectivity_role="body1_ownership",
                    ),
                    *canonicalization_evidence,
                ],
                evidence_present=True,
                resolved=True,
                body1_prim=body1_prim,
            )
        if rigid_endpoint_vocabulary_present:
            description = (
                "Stage 1 rigger body1 evidence referenced a rigid-body endpoint "
                "that does not own the candidate prediction row."
                if body1_prim in valid_endpoint_paths
                else (
                    "Stage 1 rigger body1 evidence referenced a prim path that is "
                    "not present in the source-authored endpoint vocabulary."
                )
            )
        elif hierarchy_ancestor_paths:
            description = (
                "Stage 1 rigger body1 evidence referenced a known Xform that is "
                "not a listed ancestor of the candidate prediction row."
                if body1_prim in valid_endpoint_paths
                else (
                    "Stage 1 rigger body1 evidence referenced a prim path that is "
                    "not present in the exported source hierarchy vocabulary."
                )
            )
        else:
            description = (
                "Stage 1 rigger body1 evidence referenced a different known prim "
                "than the candidate prediction row."
                if body1_prim in valid_endpoint_paths
                else (
                    "Stage 1 rigger body1 evidence referenced a prim path that is "
                    "not present in the prediction rows."
                )
            )
        return _Body1Resolution(
            evidence=[
                Stage2EvidenceItem(
                    source=evidence_source,
                    description=description,
                    value=body1_prim,
                    prim_paths=evidence_prim_paths,
                ),
                *canonicalization_evidence,
            ],
            evidence_present=True,
            resolved=False,
        )

    description = (
        "Stage 1 rigger body1 evidence referenced multiple candidate moving prim "
        "paths; left moving body unresolved."
        if exact_paths
        else (
            "Stage 1 rigger body1 evidence did not provide an exact prim path; "
            "left moving body unresolved."
        )
    )
    return _Body1Resolution(
        evidence=[
            Stage2EvidenceItem(
                source=evidence_source,
                description=description,
                value=evidence_value,
                prim_paths=evidence_prim_paths,
            ),
            *canonicalization_evidence,
        ],
        evidence_present=True,
        resolved=False,
    )


def _resolve_parent_prim(
    parent_hint: str,
    known_prim_paths: set[str],
) -> tuple[str | None, Stage2ParentResolutionSource]:
    if not parent_hint or parent_hint.lower() in _UNKNOWN_VALUES:
        return None, "unresolved"
    if parent_hint.startswith("/"):
        if parent_hint in known_prim_paths:
            return parent_hint, "stage1_hint"
        return None, "unresolved"
    return None, "unresolved"


def _motion_type_from_joint_hint(joint_type: str) -> Stage2MotionType:
    if joint_type == "revolute":
        return "revolute"
    if joint_type == "prismatic":
        return "prismatic"
    if joint_type == "spherical":
        return "spherical"
    if joint_type == "fixed":
        return "fixed"
    return "unknown"


def _motion_axis_world_from_axis_hint(axis_hint: str) -> list[float] | None:
    axis = _AXIS_HINT_TO_WORLD.get(axis_hint)
    if axis is None:
        return None
    return list(axis)


def _candidate_field_sources(
    *,
    joint_type: str,
    axis_hint: str,
    motion_axis_world: list[float] | None,
    fixed_parent_prim: str | None,
    parent_resolution_source: Stage2ParentResolutionSource,
    parent_field_source: Stage2FieldSource | None,
    payload: dict[str, Any],
    axis_source: Stage2FieldSource,
) -> dict[str, Stage2FieldSource]:
    parent_source: Stage2FieldSource = "unknown"
    if fixed_parent_prim is not None:
        if parent_field_source is not None:
            parent_source = parent_field_source
        elif parent_resolution_source in {"stage1_hint", "stage1_rigger_evidence"}:
            parent_source = cast(Stage2FieldSource, parent_resolution_source)

    if axis_hint in _UNKNOWN_VALUES:
        axis_source = "unknown"

    joint_type_stage1_source = _stage1_field_source(payload, "joint_type_hint")
    motion_type_source: Stage2FieldSource = "predicted"
    if _motion_type_from_joint_hint(joint_type) == "unknown":
        motion_type_source = "unknown"
    elif joint_type_stage1_source is not None:
        motion_type_source = joint_type_stage1_source

    motion_axis_source = axis_source
    if motion_axis_world is None:
        motion_axis_source = "unknown"

    return {
        "motion_type": motion_type_source,
        "axis_hint": axis_source,
        "motion_axis_world": motion_axis_source,
        "fixed_parent_prim": parent_source,
    }


def _stage1_field_source(
    payload: dict[str, Any], field: str
) -> Stage2FieldSource | None:
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        return None
    field_sources = provenance.get("field_sources")
    if not isinstance(field_sources, dict):
        return None
    source = field_sources.get(field)
    if isinstance(source, str) and source in _STAGE2_FIELD_SOURCES:
        return cast(Stage2FieldSource, source)
    return None


def _stage2_axis_hint_and_source(
    payload: dict[str, Any],
) -> tuple[str, Stage2FieldSource]:
    claim = _rigger_evidence_claim(payload, "motion_axis")
    if claim is not None:
        claim_value = _clean_text(claim.get("value"), "unknown").lower()
        if _normalized_axis_hint_value(claim_value) in _ALLOWED_AXIS_SET:
            return claim_value, _rigger_claim_stage2_source(claim)

    raw_axis_hint = _clean_token(payload.get("axis_hint"), "unknown")
    axis_source = _stage1_field_source(payload, "axis_hint") or "stage1_hint"
    return raw_axis_hint, axis_source


def _rigger_evidence_claim(
    payload: dict[str, Any],
    field: str,
) -> dict[str, Any] | None:
    evidence = payload.get("rigger_evidence")
    if not isinstance(evidence, dict):
        return None
    claim = evidence.get(field)
    if not isinstance(claim, dict):
        return None
    claim_value = _clean_text(claim.get("value"), "unknown")
    if claim_value.lower() in _UNKNOWN_VALUES and not _rigger_claim_exact_paths(
        claim, claim_value=claim_value
    ):
        return None
    return claim


def _rigger_claim_stage2_source(claim: dict[str, Any]) -> Stage2FieldSource:
    return _stage2_source_from_value(claim.get("source"))


def _rigger_claim_hierarchy_canonicalization(
    claim: Mapping[str, Any],
) -> tuple[str, str] | None:
    wrapper_path = _clean_text(claim.get(_HIERARCHY_WRAPPER_PATH))
    leaf_path = _clean_text(claim.get(_HIERARCHY_LEAF_PATH))
    if not wrapper_path.startswith("/") or not leaf_path.startswith("/"):
        return None
    if wrapper_path == leaf_path:
        return None
    claim_value = _clean_text(claim.get("value"), "unknown")
    if leaf_path not in _rigger_claim_exact_paths(
        claim,
        claim_value=claim_value,
    ):
        return None
    return (wrapper_path, leaf_path)


def _stage2_source_from_value(value: Any) -> Stage2FieldSource:
    source = _clean_token(value, "predicted")
    if source in _STAGE2_FIELD_SOURCES:
        return cast(Stage2FieldSource, source)
    return "unknown"


def _rigger_claim_exact_paths(
    claim: Mapping[str, Any],
    *,
    claim_value: str,
) -> list[str]:
    paths: list[str] = []
    if claim_value.startswith("/"):
        paths.append(claim_value)
    prim_paths = claim.get("prim_paths")
    if isinstance(prim_paths, list):
        for prim_path in prim_paths:
            prim_text = _clean_text(prim_path)
            if prim_text.startswith("/"):
                paths.append(prim_text)
    return _dedupe_preserving_order(paths)


def _dedupe_preserving_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _classification_payload(
    prediction: dict[str, Any], output_key: str
) -> dict[str, Any] | None:
    value = prediction.get(output_key)
    if not isinstance(value, dict):
        return None
    payload = cast(dict[str, Any], value)
    return cast(
        dict[str, Any],
        unwrap_stage1_prediction_payload(payload, output_key=output_key),
    )


def _candidate_confidence(
    confidence: Any,
    *,
    unresolved_questions: Sequence[str],
) -> Stage2Confidence:
    cleaned = _clean_token(confidence, "low")
    if cleaned not in {"high", "medium", "low"}:
        cleaned = "low"
    if len(unresolved_questions) >= 2:
        return "low"
    if unresolved_questions and cleaned == "high":
        return "medium"
    if cleaned == "medium":
        return "medium"
    if cleaned == "high":
        return "high"
    return "low"


def _articulation_candidate_flag_state(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return bool(value)
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in _TRUE_VALUES:
            return True
        if cleaned in _FALSE_VALUES:
            return False
    return None


def _clean_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _clean_token(value: Any, default: str = "") -> str:
    return _clean_text(value, default).lower().replace(" ", "_")


def _clean_alias_token(value: Any, default: str = "") -> str:
    return _clean_token(value, default).replace("-", "_")


def _first_present(value: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in value and _alias_value_is_usable(value[key]):
            return value[key]
    return None


def _alias_value_is_usable(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str) and value.strip().lower() in _UNKNOWN_VALUES:
        return False
    return True


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in _UNKNOWN_VALUES:
            return None
        try:
            number = float(text)
        except ValueError:
            return None
        return number if math.isfinite(number) else None
    return None


def _label_key(value: str) -> str:
    return _NON_LABEL_RE.sub("_", value.lower()).strip("_")


def _format_limits(candidate: Mapping[str, Any]) -> str:
    lower_limit = candidate.get("lower_limit")
    upper_limit = candidate.get("upper_limit")
    if lower_limit is None and upper_limit is None:
        return "none"
    unit = _clean_text(candidate.get("limit_unit"), "unknown")
    source = _clean_text(candidate.get("limit_source"), "unknown")
    return (
        f"lower={_format_optional_float(cast(float | None, lower_limit))}; "
        f"upper={_format_optional_float(cast(float | None, upper_limit))}; "
        f"unit={unit}; source={source}"
    )


def _format_evidence(items: Any) -> str:
    if not isinstance(items, list):
        return "none"
    bits: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        source = _clean_text(item.get("source"), "unknown")
        description = _clean_text(item.get("description"))
        value = _clean_text(item.get("value"))
        if description and value:
            bits.append(f"{source}: {description} ({value})")
        elif description:
            bits.append(f"{source}: {description}")
        elif value:
            bits.append(f"{source}: {value}")
    return "; ".join(bits) if bits else "none"


def _format_annotation_conflicts(candidate: Mapping[str, Any]) -> str:
    conflicts = candidate.get("source_annotation_conflicts")
    if not isinstance(conflicts, Mapping) or not conflicts:
        return "none"
    return json.dumps(conflicts, sort_keys=True, separators=(",", ":"))


def _e(value: Any) -> str:
    return html.escape(str(value), quote=True)
