# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rigged-reference validation for Stage 2 articulation candidates."""

from __future__ import annotations

import html
import json
import math
import re
import tempfile
from collections import Counter, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field

REFERENCE_MANIFEST_SCHEMA_VERSION: Literal[
    "joint-agent-rigged-reference-manifest-v0"
] = "joint-agent-rigged-reference-manifest-v0"
VALIDATION_SCHEMA_VERSION: Literal["joint-agent-rigged-reference-validation-v0"] = (
    "joint-agent-rigged-reference-validation-v0"
)
DEFAULT_LIMIT_TOLERANCE = 1e-4
_PATH_REVIEW_THRESHOLD = 45
_PATH_LEAF_SCORE = 65
_PATH_PASS_THRESHOLD = 80
_PATH_SUFFIX_SCORE = 85
_PATH_NORMALIZED_SCORE = 95
_PATH_EXACT_SCORE = 100
_MATCH_THRESHOLD = _PATH_PASS_THRESHOLD
_ASSIGNMENT_SCORE_WEIGHT = 1_000
_AXIS_DOT_TOLERANCE = 0.99
_AXIS_BY_VECTOR_INDEX = ("x", "y", "z")
_AXIS_UNIT_VECTORS = {
    "x": (1.0, 0.0, 0.0),
    "y": (0.0, 1.0, 0.0),
    "z": (0.0, 0.0, 1.0),
}
_PATH_TOKEN_RE = re.compile(r"[^a-z0-9]+")
_REFERENCE_SUBSET_IDENTITY_FIELDS = (
    "joint_prim_path",
    "joint_type",
    "body0",
    "body1",
    "axis",
    "lower_limit",
    "upper_limit",
)


ValidationCheckStatus = Literal[
    "pass",
    "fail",
    "missing",
    "not_applicable",
    "review",
]


class ReferenceJoint(BaseModel):
    """Authored joint data extracted from a rigged USD reference."""

    model_config = ConfigDict(extra="forbid")

    joint_prim_path: str
    joint_type: str
    body0: str | None = None
    body1: str | None = None
    axis: str | None = None
    axis_world: list[float] | None = Field(default=None, min_length=3, max_length=3)
    lower_limit: float | None = None
    upper_limit: float | None = None
    authored_metadata: dict[str, Any] = Field(default_factory=dict)


class RiggedReferenceManifest(BaseModel):
    """Ground-truth articulation manifest extracted from a rigged USD."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["joint-agent-rigged-reference-manifest-v0"] = (
        REFERENCE_MANIFEST_SCHEMA_VERSION
    )
    source_usd_path: str
    summary: dict[str, Any]
    joints: list[ReferenceJoint] = Field(default_factory=list)


class ValidationFieldCheck(BaseModel):
    """One validation check for a reference joint against a candidate."""

    model_config = ConfigDict(extra="forbid")

    status: ValidationCheckStatus
    expected: Any = None
    actual: Any = None
    detail: str = ""
    match_kind: str | None = None


class ValidationJointMatch(BaseModel):
    """Per-reference comparison result."""

    model_config = ConfigDict(extra="forbid")

    reference_joint_path: str
    candidate_id: str | None = None
    status: Literal["matched", "missing_candidate"]
    match_score: int = 0
    checks: dict[str, ValidationFieldCheck]
    mismatch_reasons: list[str] = Field(default_factory=list)
    reference: dict[str, Any]
    candidate: dict[str, Any] | None = None


@dataclass(frozen=True)
class _EndpointMatch:
    moving_check: ValidationFieldCheck
    fixed_check: ValidationFieldCheck
    orientation: Literal["authored", "reversed"]


@dataclass(frozen=True)
class _CandidateMatch:
    score: int
    moving_score: int
    endpoint_match: _EndpointMatch


@dataclass
class _FlowEdge:
    to_node: int
    reverse_index: int
    capacity: int
    cost: int
    reference_index: int | None = None
    candidate_index: int | None = None


def extract_reference_articulation_manifest(
    usd_path: str | Path,
) -> dict[str, Any]:
    """Extract authored USD physics joints into a reusable manifest."""
    from pxr import Usd, UsdPhysics

    source_path = Path(usd_path)
    stage = Usd.Stage.Open(str(source_path))
    if stage is None:
        raise ValueError(f"Could not open rigged USD reference: {source_path}")

    joints: list[ReferenceJoint] = []
    for prim in stage.Traverse():
        joint_type = _joint_type_from_prim(prim, UsdPhysics)
        if joint_type is None:
            continue

        body0 = _single_relationship_target(prim, "physics:body0")
        body1 = _single_relationship_target(prim, "physics:body1")
        axis = _normalize_axis(_effective_attribute_value(prim, "physics:axis"))
        reference_joint = ReferenceJoint(
            joint_prim_path=str(prim.GetPath()),
            joint_type=joint_type,
            body0=body0,
            body1=body1,
            axis=axis,
            axis_world=_reference_axis_world(stage, prim, body0, body1, axis),
            lower_limit=_optional_float(
                _authored_attribute_value(prim, "physics:lowerLimit")
            ),
            upper_limit=_optional_float(
                _authored_attribute_value(prim, "physics:upperLimit")
            ),
            authored_metadata=_extract_authored_metadata(stage, prim, body0, body1),
        )
        joints.append(reference_joint)

    joint_type_counts = Counter(joint.joint_type for joint in joints)
    manifest = RiggedReferenceManifest(
        source_usd_path=str(source_path),
        summary={
            "joint_count": len(joints),
            "joint_type_counts": dict(sorted(joint_type_counts.items())),
        },
        joints=joints,
    )
    return cast(dict[str, Any], manifest.model_dump(mode="json"))


def build_effective_reference_manifest_subset(
    extracted_manifest: RiggedReferenceManifest | Mapping[str, Any],
    declared_manifest: RiggedReferenceManifest | Mapping[str, Any],
    *,
    allowed_omitted_joint_types: frozenset[str],
) -> dict[str, Any]:
    """Validate a declared joint selection and bind it to current USD values.

    The declared manifest chooses joint paths, while the extracted manifest is
    authoritative for current graph, axis, limit, and authored-metadata values.
    Older manifests may omit ``axis_world`` because that field was added after
    the initial reference bundles; when present it must match exactly.
    """

    extracted = (
        extracted_manifest
        if isinstance(extracted_manifest, RiggedReferenceManifest)
        else RiggedReferenceManifest.model_validate(extracted_manifest)
    )
    declared = (
        declared_manifest
        if isinstance(declared_manifest, RiggedReferenceManifest)
        else RiggedReferenceManifest.model_validate(declared_manifest)
    )
    declared_document = declared.model_dump(mode="json")
    raw_declared_joints = (
        declared_manifest.get("joints", [])
        if isinstance(declared_manifest, Mapping)
        else declared_document["joints"]
    )
    if not isinstance(raw_declared_joints, list) or len(raw_declared_joints) != len(
        declared.joints
    ):
        raise ValueError("declared reference manifest joints are malformed")

    def joint_map(
        joints: Sequence[ReferenceJoint], *, label: str
    ) -> dict[str, ReferenceJoint]:
        by_path: dict[str, ReferenceJoint] = {}
        for joint in joints:
            if not joint.joint_prim_path:
                raise ValueError(f"{label} contains an empty joint path")
            if joint.joint_prim_path in by_path:
                raise ValueError(
                    f"{label} contains duplicate joint path {joint.joint_prim_path!r}"
                )
            by_path[joint.joint_prim_path] = joint
        return by_path

    extracted_by_path = joint_map(extracted.joints, label="extracted manifest")
    declared_by_path = joint_map(declared.joints, label="declared manifest")
    _validate_reference_manifest_summary(extracted, label="extracted manifest")
    _validate_reference_manifest_summary(declared, label="declared manifest")

    effective_joints: list[ReferenceJoint] = []
    for index, declared_joint in enumerate(declared.joints):
        extracted_joint = extracted_by_path.get(declared_joint.joint_prim_path)
        if extracted_joint is None:
            raise ValueError(
                "declared reference manifest contains a joint absent from the "
                f"materialized rigged USD: {declared_joint.joint_prim_path}"
            )
        declared_values = declared_joint.model_dump(mode="json")
        extracted_values = extracted_joint.model_dump(mode="json")
        drifted_fields = [
            field
            for field in _REFERENCE_SUBSET_IDENTITY_FIELDS
            if declared_values[field] != extracted_values[field]
        ]
        raw_declared_joint = raw_declared_joints[index]
        if not isinstance(raw_declared_joint, Mapping):
            raise ValueError("declared reference manifest joint is malformed")
        if "axis_world" in raw_declared_joint and (
            declared_values["axis_world"] != extracted_values["axis_world"]
        ):
            drifted_fields.append("axis_world")
        if drifted_fields:
            raise ValueError(
                f"declared reference joint {declared_joint.joint_prim_path!r} "
                "differs from the materialized rigged USD in fields "
                f"{drifted_fields}"
            )
        effective_joints.append(extracted_joint)

    omitted = [
        joint
        for path, joint in extracted_by_path.items()
        if path not in declared_by_path
    ]
    disallowed = sorted(
        joint.joint_prim_path
        for joint in omitted
        if joint.joint_type not in allowed_omitted_joint_types
    )
    if disallowed:
        raise ValueError(
            "declared reference manifest omits non-allowlisted joints from the "
            f"materialized rigged USD: {disallowed}"
        )
    omitted_counts = Counter(joint.joint_type for joint in omitted)
    for joint_type, count in sorted(omitted_counts.items()):
        summary_key = f"excluded_{joint_type}_joint_count"
        if declared.summary.get(summary_key) != count:
            raise ValueError(
                f"declared reference manifest summary {summary_key} is inconsistent"
            )
    for joint_type in allowed_omitted_joint_types - set(omitted_counts):
        summary_key = f"excluded_{joint_type}_joint_count"
        if summary_key in declared.summary and declared.summary[summary_key] != 0:
            raise ValueError(
                f"declared reference manifest summary {summary_key} is inconsistent"
            )

    effective_summary = dict(declared.summary)
    effective_summary["joint_count"] = len(effective_joints)
    effective_summary["joint_type_counts"] = dict(
        sorted(Counter(joint.joint_type for joint in effective_joints).items())
    )
    effective = RiggedReferenceManifest(
        source_usd_path=extracted.source_usd_path,
        summary=effective_summary,
        joints=effective_joints,
    )
    return cast(dict[str, Any], effective.model_dump(mode="json"))


def _validate_reference_manifest_summary(
    manifest: RiggedReferenceManifest, *, label: str
) -> None:
    expected_counts = dict(
        sorted(Counter(joint.joint_type for joint in manifest.joints).items())
    )
    if manifest.summary.get("joint_count") != len(manifest.joints):
        raise ValueError(f"{label} summary joint_count is inconsistent")
    if manifest.summary.get("joint_type_counts") != expected_counts:
        raise ValueError(f"{label} summary joint_type_counts is inconsistent")


def compare_articulation_candidates_to_reference(
    reference_manifest: RiggedReferenceManifest | Mapping[str, Any],
    candidate_document: Mapping[str, Any],
    *,
    limit_tolerance: float = DEFAULT_LIMIT_TOLERANCE,
) -> dict[str, Any]:
    """Compare Stage 2 candidates against a rigged-reference manifest."""
    _validate_limit_tolerance(limit_tolerance)
    manifest = (
        reference_manifest
        if isinstance(reference_manifest, RiggedReferenceManifest)
        else RiggedReferenceManifest.model_validate(reference_manifest)
    )
    candidates = [
        candidate
        for candidate in candidate_document.get("candidates", [])
        if isinstance(candidate, Mapping)
    ]

    assignments, candidate_matches = _assign_candidate_matches(
        manifest.joints,
        candidates,
    )
    matches: list[ValidationJointMatch] = []
    matched_candidate_indices = set(assignments.values())

    for reference_index, reference in enumerate(manifest.joints):
        candidate_index = assignments.get(reference_index)
        if candidate_index is None:
            matches.append(_missing_candidate_match(reference))
            continue

        candidate = dict(candidates[candidate_index])
        candidate_match = candidate_matches[(reference_index, candidate_index)]
        checks = _compare_reference_to_candidate(
            reference,
            candidate,
            endpoint_match=candidate_match.endpoint_match,
            limit_tolerance=limit_tolerance,
        )
        mismatch_reasons = _mismatch_reasons(checks)
        matches.append(
            ValidationJointMatch(
                reference_joint_path=reference.joint_prim_path,
                candidate_id=_candidate_id(candidate),
                status="matched",
                match_score=candidate_match.score,
                checks=checks,
                mismatch_reasons=mismatch_reasons,
                reference=reference.model_dump(mode="json"),
                candidate=dict(candidate),
            )
        )

    extra_candidates = [
        dict(candidate)
        for index, candidate in enumerate(candidates)
        if index not in matched_candidate_indices
    ]
    summary = _validation_summary(
        manifest=manifest,
        candidate_document=candidate_document,
        matches=matches,
        extra_candidate_count=len(extra_candidates),
    )
    validation_document = {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "reference_manifest": manifest.model_dump(mode="json"),
        "candidate_schema_version": candidate_document.get("schema_version"),
        "summary": summary,
        "matches": [match.model_dump(mode="json") for match in matches],
        "extra_candidates": extra_candidates,
    }
    return validation_document


def load_json_document(path: str | Path) -> dict[str, Any]:
    """Load a JSON document from disk."""
    with Path(path).open(encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def write_json_document(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Atomically write a JSON document."""
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
        if tmp_path is not None:
            # On success replace() moves the temp file; on write/replace failure
            # this removes the partial artifact left beside the destination.
            tmp_path.unlink(missing_ok=True)


def write_rigged_reference_validation_report_html(
    path: str | Path,
    validation_document: Mapping[str, Any],
) -> None:
    """Write an HTML report for rigged-reference validation."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = validation_document.get("summary", {})
    matches = validation_document.get("matches", [])
    extra_candidates = validation_document.get("extra_candidates", [])
    rows = []
    match_rows: Sequence[Any] = []
    if isinstance(matches, Sequence) and not isinstance(matches, str):
        match_rows = matches
    for raw_match in match_rows:
        if not isinstance(raw_match, Mapping):
            continue
        checks = raw_match.get("checks", {})
        check_bits = []
        if isinstance(checks, Mapping):
            for name in (
                "candidate_found",
                "joint_type",
                "body1",
                "body0",
                "axis",
                "limits",
            ):
                raw_check = checks.get(name)
                if not isinstance(raw_check, Mapping):
                    continue
                status = raw_check.get("status", "missing")
                detail = raw_check.get("detail", "")
                check_bits.append(
                    f"<div><strong>{_e(name)}:</strong> "
                    f'<span class="status-{_e(status)}">{_e(status)}</span>'
                    f"{' - ' + _e(detail) if detail else ''}</div>"
                )
        mismatch_reasons = raw_match.get("mismatch_reasons", [])
        rows.append(
            "<tr>"
            f"<td>{_e(raw_match.get('reference_joint_path'))}</td>"
            f"<td>{_e(raw_match.get('candidate_id') or 'missing')}</td>"
            f"<td>{_e(raw_match.get('status'))}</td>"
            f"<td>{_e(raw_match.get('match_score', 0))}</td>"
            f"<td>{''.join(check_bits)}</td>"
            f"<td>{_e(', '.join(mismatch_reasons) if isinstance(mismatch_reasons, list) else '')}</td>"
            "</tr>"
        )

    body = "\n".join(rows) or (
        '<tr><td colspan="6">No reference joints found.</td></tr>'
    )
    extra_bits = ", ".join(
        _e(_candidate_id(candidate))
        for candidate in extra_candidates
        if isinstance(candidate, Mapping)
    )
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Joint Agent Rigged-Reference Validation</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; table-layout: fixed; }}
    th, td {{ border: 1px solid #c9d1d9; padding: 8px; vertical-align: top; }}
    th {{ background: #eef2f7; text-align: left; }}
    td {{ word-wrap: break-word; }}
    .summary {{ margin-bottom: 16px; }}
    .meta {{ color: #52606d; margin-top: 12px; }}
    .status-pass {{ color: #087f5b; font-weight: 700; }}
    .status-fail, .status-missing {{ color: #c92a2a; font-weight: 700; }}
    .status-review {{ color: #b7791f; font-weight: 700; }}
    .status-not_applicable {{ color: #52606d; }}
  </style>
</head>
<body>
  <h1>Rigged-Reference Validation</h1>
  <div class="summary">
    <div>Reference joints: {_e(summary.get("reference_joint_count", 0))}</div>
    <div>Candidates: {_e(summary.get("candidate_count", 0))}</div>
    <div>Matched references: {_e(summary.get("matched_reference_count", 0))}</div>
    <div>Candidate recall: {_e(summary.get("candidate_recall", 0.0))}</div>
    <div>Joint type matches: {_e(summary.get("joint_type_match_count", 0))}</div>
    <div>Body1 matches: {_e(summary.get("body1_match_count", 0))}</div>
    <div>Body0 matches: {_e(summary.get("body0_match_count", 0))}</div>
    <div>Axis matches: {_e(summary.get("axis_match_count", 0))}</div>
    <div>Limit matches: {_e(summary.get("limit_value_match_count", 0))}</div>
    <div>Missing candidates: {_e(summary.get("missing_candidate_count", 0))}</div>
    <div>Extra candidates: {_e(summary.get("extra_candidate_count", 0))}</div>
  </div>
  <table>
    <thead>
      <tr>
        <th>Reference Joint</th>
        <th>Candidate</th>
        <th>Status</th>
        <th>Score</th>
        <th>Checks</th>
        <th>Mismatch Reasons</th>
      </tr>
    </thead>
    <tbody>
      {body}
    </tbody>
  </table>
  <div class="meta">Extra candidates: {extra_bits or "none"}</div>
  <div class="meta">Schema: {_e(validation_document.get("schema_version"))}</div>
</body>
</html>
"""
    output_path.write_text(html_text, encoding="utf-8")


def _assign_candidate_matches(
    references: Sequence[ReferenceJoint],
    candidates: list[Mapping[str, Any]],
) -> tuple[dict[int, int], dict[tuple[int, int], _CandidateMatch]]:
    candidate_matches: dict[tuple[int, int], _CandidateMatch] = {}
    for reference_index, reference in enumerate(references):
        for candidate_index, candidate in enumerate(candidates):
            candidate_match = _candidate_match(reference, candidate)
            if candidate_match.score >= _MATCH_THRESHOLD:
                candidate_matches[(reference_index, candidate_index)] = candidate_match
    if not candidate_matches:
        return {}, candidate_matches

    return (
        _assign_candidate_matches_by_flow(
            reference_count=len(references),
            candidate_count=len(candidates),
            candidate_matches=candidate_matches,
        ),
        candidate_matches,
    )


def _assign_candidate_matches_by_flow(
    *,
    reference_count: int,
    candidate_count: int,
    candidate_matches: Mapping[tuple[int, int], _CandidateMatch],
) -> dict[int, int]:
    source = 0
    reference_offset = 1
    candidate_offset = reference_offset + reference_count
    sink = candidate_offset + candidate_count
    graph: list[list[_FlowEdge]] = [[] for _ in range(sink + 1)]

    def add_edge(
        from_node: int,
        to_node: int,
        capacity: int,
        cost: int,
        *,
        reference_index: int | None = None,
        candidate_index: int | None = None,
    ) -> None:
        forward = _FlowEdge(
            to_node=to_node,
            reverse_index=len(graph[to_node]),
            capacity=capacity,
            cost=cost,
            reference_index=reference_index,
            candidate_index=candidate_index,
        )
        reverse = _FlowEdge(
            to_node=from_node,
            reverse_index=len(graph[from_node]),
            capacity=0,
            cost=-cost,
        )
        graph[from_node].append(forward)
        graph[to_node].append(reverse)

    for reference_index in range(reference_count):
        add_edge(source, reference_offset + reference_index, 1, 0)
    for candidate_index in range(candidate_count):
        add_edge(candidate_offset + candidate_index, sink, 1, 0)
    max_match_count = min(reference_count, candidate_count)
    max_edge_weight = max(
        (
            _assignment_edge_weight(candidate_match)
            for candidate_match in candidate_matches.values()
        ),
        default=0,
    )
    # One additional match must dominate any possible total score delta, so the
    # flow optimizes lexicographically: cardinality first, then score.
    cardinality_bonus = max_match_count * max_edge_weight + 1
    for (
        reference_index,
        candidate_index,
    ), candidate_match in candidate_matches.items():
        utility = cardinality_bonus + _assignment_edge_weight(candidate_match)
        add_edge(
            reference_offset + reference_index,
            candidate_offset + candidate_index,
            1,
            -utility,
            reference_index=reference_index,
            candidate_index=candidate_index,
        )

    while True:
        shortest_path = _shortest_augmenting_path(graph, source, sink)
        if shortest_path is None:
            break
        path_cost, previous_nodes, previous_edges = shortest_path
        if path_cost >= 0:
            break
        node = sink
        while node != source:
            previous_node = previous_nodes[node]
            edge = graph[previous_node][previous_edges[node]]
            edge.capacity -= 1
            graph[edge.to_node][edge.reverse_index].capacity += 1
            node = previous_node

    assignments: dict[int, int] = {}
    for reference_index in range(reference_count):
        node = reference_offset + reference_index
        for edge in graph[node]:
            if (
                edge.reference_index is not None
                and edge.candidate_index is not None
                and edge.capacity == 0
            ):
                assignments[edge.reference_index] = edge.candidate_index
    return assignments


def _assignment_edge_weight(candidate_match: _CandidateMatch) -> int:
    return (
        candidate_match.score * _ASSIGNMENT_SCORE_WEIGHT + candidate_match.moving_score
    )


def _shortest_augmenting_path(
    graph: Sequence[Sequence[_FlowEdge]],
    source: int,
    sink: int,
) -> tuple[int, list[int], list[int]] | None:
    distances = [math.inf] * len(graph)
    previous_nodes = [-1] * len(graph)
    previous_edges = [-1] * len(graph)
    queued = [False] * len(graph)
    distances[source] = 0
    queue: deque[int] = deque([source])
    queued[source] = True

    while queue:
        node = queue.popleft()
        queued[node] = False
        for edge_index, edge in enumerate(graph[node]):
            if edge.capacity <= 0:
                continue
            next_distance = distances[node] + edge.cost
            if next_distance >= distances[edge.to_node]:
                continue
            distances[edge.to_node] = next_distance
            previous_nodes[edge.to_node] = node
            previous_edges[edge.to_node] = edge_index
            if not queued[edge.to_node]:
                queue.append(edge.to_node)
                queued[edge.to_node] = True

    if previous_nodes[sink] == -1:
        return None
    return int(distances[sink]), previous_nodes, previous_edges


def _candidate_match(
    reference: ReferenceJoint,
    candidate: Mapping[str, Any],
) -> _CandidateMatch:
    endpoint_match = _endpoint_match_for_candidate(reference, candidate)
    score = _candidate_match_score(
        reference,
        candidate,
        endpoint_match=endpoint_match,
    )
    return _CandidateMatch(
        score=score,
        moving_score=_path_check_score(endpoint_match.moving_check),
        endpoint_match=endpoint_match,
    )


def _endpoint_match_for_candidate(
    reference: ReferenceJoint,
    candidate: Mapping[str, Any],
) -> _EndpointMatch:
    authored = _endpoint_match_for_orientation(
        moving_body=reference.body1,
        fixed_body=reference.body0,
        candidate=candidate,
        orientation="authored",
    )
    reversed_match = _endpoint_match_for_orientation(
        moving_body=reference.body0,
        fixed_body=reference.body1,
        candidate=candidate,
        orientation="reversed",
    )
    authored_score = _endpoint_match_score(authored)
    reversed_score = _endpoint_match_score(reversed_match)
    return reversed_match if reversed_score > authored_score else authored


def _endpoint_match_for_orientation(
    *,
    moving_body: str | None,
    fixed_body: str | None,
    candidate: Mapping[str, Any],
    orientation: Literal["authored", "reversed"],
) -> _EndpointMatch:
    moving_check = _compare_path_field(
        expected=moving_body,
        actual_values=_candidate_moving_part_prims(candidate),
        field_name="moving body",
    )
    fixed_check = _compare_path_field(
        expected=fixed_body,
        actual_values=[_candidate_fixed_parent(candidate)],
        field_name="fixed body",
    )
    if orientation == "reversed":
        moving_check = _with_orientation_detail(moving_check, "moving body")
        fixed_check = _with_orientation_detail(fixed_check, "fixed body")
    return _EndpointMatch(
        moving_check=moving_check,
        fixed_check=fixed_check,
        orientation=orientation,
    )


def _endpoint_match_score(endpoint_match: _EndpointMatch) -> tuple[int, int]:
    return (
        _path_check_score(endpoint_match.moving_check),
        _path_check_score(endpoint_match.fixed_check),
    )


def _with_orientation_detail(
    check: ValidationFieldCheck,
    field_name: str,
) -> ValidationFieldCheck:
    detail = (
        f"{check.detail} Authored USD joint endpoints are reversed relative to "
        f"Stage 2 {field_name} semantics."
    )
    return ValidationFieldCheck(
        status=check.status,
        expected=check.expected,
        actual=check.actual,
        detail=detail,
        match_kind=check.match_kind,
    )


def _candidate_match_score(
    reference: ReferenceJoint,
    candidate: Mapping[str, Any],
    *,
    endpoint_match: _EndpointMatch,
) -> int:
    score = _path_check_score(endpoint_match.moving_check)
    if _compare_joint_type(reference, candidate).status == "pass":
        score += 20
    if endpoint_match.fixed_check.status == "pass":
        score += 10
    axis_check = _compare_axis(reference, candidate)
    if axis_check.status == "pass":
        score += 8
    return score


def _compare_reference_to_candidate(
    reference: ReferenceJoint,
    candidate: Mapping[str, Any],
    *,
    endpoint_match: _EndpointMatch,
    limit_tolerance: float,
) -> dict[str, ValidationFieldCheck]:
    return {
        "candidate_found": ValidationFieldCheck(
            status="pass",
            expected=reference.joint_prim_path,
            actual=_candidate_id(candidate),
            detail="A Stage 2 candidate was matched to this reference joint.",
        ),
        "joint_type": _compare_joint_type(reference, candidate),
        "body1": (
            endpoint_match.moving_check
            if endpoint_match.orientation == "authored"
            else endpoint_match.fixed_check
        ),
        "body0": (
            endpoint_match.fixed_check
            if endpoint_match.orientation == "authored"
            else endpoint_match.moving_check
        ),
        "axis": _compare_axis(reference, candidate),
        "limits": _compare_limits(reference, candidate, limit_tolerance),
    }


def _missing_candidate_match(reference: ReferenceJoint) -> ValidationJointMatch:
    checks = {
        "candidate_found": ValidationFieldCheck(
            status="missing",
            expected=reference.joint_prim_path,
            detail="No Stage 2 candidate matched this reference joint.",
        ),
        "joint_type": ValidationFieldCheck(
            status="missing",
            expected=reference.joint_type,
            detail="Candidate is missing.",
        ),
        "body1": ValidationFieldCheck(
            status="missing",
            expected=reference.body1,
            detail="Candidate is missing.",
        ),
        "body0": ValidationFieldCheck(
            status="missing",
            expected=reference.body0,
            detail="Candidate is missing.",
        ),
        "axis": ValidationFieldCheck(
            status="missing",
            expected=reference.axis,
            detail="Candidate is missing.",
        ),
        "limits": ValidationFieldCheck(
            status="missing",
            expected=_reference_limits(reference),
            detail="Candidate is missing.",
        ),
    }
    return ValidationJointMatch(
        reference_joint_path=reference.joint_prim_path,
        status="missing_candidate",
        match_score=0,
        checks=checks,
        mismatch_reasons=["missing_candidate"],
        reference=reference.model_dump(mode="json"),
    )


def _compare_joint_type(
    reference: ReferenceJoint,
    candidate: Mapping[str, Any],
) -> ValidationFieldCheck:
    expected = _normalize_joint_type(reference.joint_type)
    actual = _normalize_joint_type(
        candidate.get("joint_type_hint") or candidate.get("motion_type")
    )
    if expected is None:
        return ValidationFieldCheck(
            status="not_applicable",
            expected=reference.joint_type,
            actual=actual,
            detail="Reference joint type is unavailable.",
        )
    if actual is None or actual == "unknown":
        return ValidationFieldCheck(
            status="missing",
            expected=expected,
            actual=actual or "unknown",
            detail="Candidate joint type is unresolved.",
        )
    if actual == expected:
        return ValidationFieldCheck(
            status="pass",
            expected=expected,
            actual=actual,
            detail="Joint type matches.",
        )
    return ValidationFieldCheck(
        status="fail",
        expected=expected,
        actual=actual,
        detail="Candidate joint type differs from authored reference.",
    )


def _compare_axis(
    reference: ReferenceJoint,
    candidate: Mapping[str, Any],
) -> ValidationFieldCheck:
    expected_vector = _reference_axis_vector(reference)
    actual_vector = _candidate_axis_vector(candidate)
    expected = reference.axis_world or reference.axis
    actual = actual_vector or _candidate_axis(candidate)
    if expected_vector is None:
        return ValidationFieldCheck(
            status="not_applicable",
            expected=reference.axis,
            actual=actual,
            detail="Reference axis is not authored.",
        )
    if actual_vector is None:
        return ValidationFieldCheck(
            status="missing",
            expected=expected,
            actual="unknown",
            detail="Candidate axis is unresolved.",
        )
    ignore_sign = reference.lower_limit is None and reference.upper_limit is None
    if _axes_equivalent(expected_vector, actual_vector, ignore_sign=ignore_sign):
        detail = (
            "Axis matches in world coordinates, ignoring sign."
            if ignore_sign
            else "Axis matches in world coordinates with signed direction."
        )
        return ValidationFieldCheck(
            status="pass",
            expected=expected,
            actual=actual,
            detail=detail,
        )
    detail = (
        "Candidate axis differs from authored reference."
        if ignore_sign
        else (
            "Candidate axis direction differs from limited authored reference; "
            "limited checks preserve candidate sign."
        )
    )
    return ValidationFieldCheck(
        status="fail",
        expected=expected,
        actual=actual,
        detail=detail,
    )


def _compare_path_field(
    *,
    expected: str | None,
    actual_values: Sequence[str | None],
    field_name: str,
) -> ValidationFieldCheck:
    actual_paths = [path for path in actual_values if path]
    if expected is None:
        return ValidationFieldCheck(
            status="not_applicable",
            expected=None,
            actual=actual_paths,
            detail=f"Reference {field_name} is not authored.",
        )
    if not actual_paths:
        return ValidationFieldCheck(
            status="missing",
            expected=expected,
            actual=None,
            detail=f"Candidate {field_name} is unresolved.",
        )

    best_actual = actual_paths[0]
    best_kind = "none"
    best_score = 0
    for actual in actual_paths:
        kind, score = _path_match_kind(expected, actual)
        if score > best_score:
            best_actual = actual
            best_kind = kind
            best_score = score

    if best_score >= _PATH_PASS_THRESHOLD:
        return ValidationFieldCheck(
            status="pass",
            expected=expected,
            actual=best_actual,
            detail=f"Candidate {field_name} matches the reference path.",
            match_kind=best_kind,
        )
    if best_score >= _PATH_REVIEW_THRESHOLD:
        return ValidationFieldCheck(
            status="review",
            expected=expected,
            actual=best_actual,
            detail=(
                f"Candidate {field_name} only weakly matches the reference path; "
                "review the pairing."
            ),
            match_kind=best_kind,
        )
    return ValidationFieldCheck(
        status="fail",
        expected=expected,
        actual=best_actual,
        detail=f"Candidate {field_name} differs from the authored reference.",
        match_kind=best_kind,
    )


def _compare_limits(
    reference: ReferenceJoint,
    candidate: Mapping[str, Any],
    limit_tolerance: float,
) -> ValidationFieldCheck:
    _validate_limit_tolerance(limit_tolerance)
    expected = _reference_limits(reference)
    if expected["lower"] is None and expected["upper"] is None:
        return ValidationFieldCheck(
            status="not_applicable",
            expected=expected,
            actual=_candidate_limits(candidate),
            detail="Reference joint has no authored limits.",
        )

    actual = _candidate_limits(candidate)
    missing_keys = [
        key
        for key, value in expected.items()
        if value is not None and actual[key] is None
    ]
    if missing_keys:
        return ValidationFieldCheck(
            status="missing",
            expected=expected,
            actual=actual,
            detail=f"Candidate is missing authored limit field(s): {', '.join(missing_keys)}.",
        )

    mismatched_keys = []
    for key, expected_value in expected.items():
        actual_value = actual[key]
        if expected_value is None or actual_value is None:
            continue
        if not math.isclose(expected_value, actual_value, abs_tol=limit_tolerance):
            mismatched_keys.append(key)
    if mismatched_keys:
        return ValidationFieldCheck(
            status="fail",
            expected=expected,
            actual=actual,
            detail=f"Candidate limit value differs for: {', '.join(mismatched_keys)}.",
        )
    return ValidationFieldCheck(
        status="pass",
        expected=expected,
        actual=actual,
        detail="Authored limit values match within tolerance.",
    )


def _validation_summary(
    *,
    manifest: RiggedReferenceManifest,
    candidate_document: Mapping[str, Any],
    matches: list[ValidationJointMatch],
    extra_candidate_count: int,
) -> dict[str, Any]:
    reference_joint_count = len(manifest.joints)
    matched_reference_count = sum(1 for match in matches if match.status == "matched")
    checks_by_name = {
        "joint_type": "joint_type_match_count",
        "body1": "body1_match_count",
        "body0": "body0_match_count",
        "axis": "axis_match_count",
        "limits": "limit_value_match_count",
    }
    summary = {
        "reference_joint_count": reference_joint_count,
        "candidate_count": _candidate_count(candidate_document),
        "matched_reference_count": matched_reference_count,
        "candidate_recall": (
            matched_reference_count / reference_joint_count
            if reference_joint_count
            else 0.0
        ),
        "missing_candidate_count": sum(
            1 for match in matches if match.status == "missing_candidate"
        ),
        "extra_candidate_count": extra_candidate_count,
        "ready_candidate_count": _candidate_summary_count(
            candidate_document,
            "ready_candidate_count",
            "ready_for_rigger_input",
        ),
        "review_required_candidate_count": _candidate_summary_count(
            candidate_document,
            "review_required_candidate_count",
            "review_required",
        ),
        "unresolved_reason_counts": _unresolved_reason_counts(candidate_document),
        "match_status_counts": dict(
            sorted(Counter(match.status for match in matches).items())
        ),
    }
    for check_name, summary_name in checks_by_name.items():
        summary[summary_name] = sum(
            1
            for match in matches
            if match.checks.get(check_name) is not None
            and match.checks[check_name].status == "pass"
        )

    summary["limit_reference_count"] = sum(
        1
        for joint in manifest.joints
        if joint.lower_limit is not None or joint.upper_limit is not None
    )
    summary["limit_presence_match_count"] = sum(
        1
        for match in matches
        if match.checks.get("limits") is not None
        and match.checks["limits"].status in {"pass", "fail"}
    )
    return summary


def _candidate_summary_count(
    candidate_document: Mapping[str, Any],
    key: str,
    review_status: str,
) -> int:
    raw_summary = candidate_document.get("summary", {})
    if isinstance(raw_summary, Mapping) and isinstance(raw_summary.get(key), int):
        return int(raw_summary[key])
    candidates = candidate_document.get("candidates", [])
    if not isinstance(candidates, Sequence):
        return 0
    return sum(
        1
        for candidate in candidates
        if isinstance(candidate, Mapping)
        and candidate.get("review_status") == review_status
    )


def _candidate_count(candidate_document: Mapping[str, Any]) -> int:
    raw_summary = candidate_document.get("summary", {})
    if isinstance(raw_summary, Mapping) and isinstance(
        raw_summary.get("candidate_count"), int
    ):
        return int(raw_summary["candidate_count"])
    candidates = candidate_document.get("candidates", [])
    if isinstance(candidates, Sequence) and not isinstance(candidates, str):
        return len(candidates)
    return 0


def _unresolved_reason_counts(candidate_document: Mapping[str, Any]) -> dict[str, int]:
    raw_summary = candidate_document.get("summary", {})
    if isinstance(raw_summary, Mapping) and isinstance(
        raw_summary.get("reason_code_counts"), Mapping
    ):
        return {
            str(key): int(value)
            for key, value in raw_summary["reason_code_counts"].items()
            if isinstance(value, int)
        }
    reason_counts: Counter[str] = Counter()
    candidates = candidate_document.get("candidates", [])
    if not isinstance(candidates, Sequence):
        return {}
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        reason_codes = candidate.get("unresolved_reason_codes", [])
        if not isinstance(reason_codes, Sequence) or isinstance(reason_codes, str):
            continue
        reason_counts.update(str(code) for code in reason_codes)
    return dict(sorted(reason_counts.items()))


def _mismatch_reasons(
    checks: Mapping[str, ValidationFieldCheck],
) -> list[str]:
    reasons = []
    for field_name, check in checks.items():
        if check.status in {"fail", "missing", "review"}:
            reasons.append(f"{field_name}_{check.status}")
    return reasons


def _path_check_score(check: ValidationFieldCheck) -> int:
    if check.status == "pass":
        if check.match_kind == "exact":
            return _PATH_EXACT_SCORE
        if check.match_kind == "normalized":
            return _PATH_NORMALIZED_SCORE
        if check.match_kind == "suffix":
            return _PATH_SUFFIX_SCORE
    if check.status == "review":
        return _PATH_REVIEW_THRESHOLD
    return 0


def _path_match_kind(expected: str, actual: str) -> tuple[str, int]:
    if expected == actual:
        return "exact", _PATH_EXACT_SCORE
    expected_segments = _path_segments(expected)
    actual_segments = _path_segments(actual)
    if expected_segments == actual_segments and expected_segments:
        return "normalized", _PATH_NORMALIZED_SCORE
    suffix_overlap = _suffix_overlap(expected_segments, actual_segments)
    if suffix_overlap >= 2:
        return "suffix", _PATH_SUFFIX_SCORE
    if suffix_overlap == 1:
        return "leaf", _PATH_LEAF_SCORE
    token_score = _leaf_token_score(expected_segments, actual_segments)
    if token_score >= _PATH_REVIEW_THRESHOLD:
        return "leaf_tokens", token_score
    return "none", 0


def _path_segments(path: str) -> list[str]:
    return [
        _normalize_path_segment(segment)
        for segment in path.split("/")
        if _normalize_path_segment(segment)
    ]


def _normalize_path_segment(segment: str) -> str:
    return _PATH_TOKEN_RE.sub("_", segment.lower()).strip("_")


def _suffix_overlap(first: Sequence[str], second: Sequence[str]) -> int:
    overlap = 0
    for left, right in zip(reversed(first), reversed(second), strict=False):
        if left != right:
            break
        overlap += 1
    return overlap


def _leaf_token_score(first: Sequence[str], second: Sequence[str]) -> int:
    if not first or not second:
        return 0
    left_tokens = {token for token in first[-1].split("_") if token}
    right_tokens = {token for token in second[-1].split("_") if token}
    if not left_tokens or not right_tokens:
        return 0
    overlap = left_tokens & right_tokens
    if not overlap:
        return 0
    return int(
        _PATH_REVIEW_THRESHOLD * len(overlap) / max(len(left_tokens), len(right_tokens))
    )


def _candidate_moving_part_prims(candidate: Mapping[str, Any]) -> list[str]:
    raw_prims = candidate.get("moving_part_prims", [])
    if isinstance(raw_prims, str):
        return [raw_prims]
    if not isinstance(raw_prims, Sequence):
        return []
    return [str(prim) for prim in raw_prims if prim]


def _candidate_fixed_parent(candidate: Mapping[str, Any]) -> str | None:
    raw_value = candidate.get("fixed_parent_prim")
    return str(raw_value) if raw_value else None


def _candidate_id(candidate: Mapping[str, Any]) -> str:
    return str(candidate.get("candidate_id") or candidate.get("id") or "unknown")


def _candidate_axis(candidate: Mapping[str, Any]) -> str | None:
    axis = _normalize_axis(candidate.get("axis_hint"))
    if axis is not None:
        return axis
    return _axis_from_vector(candidate.get("motion_axis_world"))


def _candidate_axis_vector(
    candidate: Mapping[str, Any],
) -> tuple[float, float, float] | None:
    vector = _normalize_vector(
        _vector_from_sequence(candidate.get("motion_axis_world"))
    )
    if vector is not None:
        return vector
    return _axis_vector_from_hint(candidate.get("axis_hint"))


def _reference_axis_vector(
    reference: ReferenceJoint,
) -> tuple[float, float, float] | None:
    return _normalize_vector(_vector_from_sequence(reference.axis_world)) or (
        _axis_unit_vector(reference.axis)
    )


def _axis_from_vector(value: Any) -> str | None:
    vector = _normalize_vector(_vector_from_sequence(value))
    if vector is None:
        return None
    components = [abs(component) for component in vector]
    max_value = max(components)
    max_index = components.index(max_value)
    if sum(1 for component in components if math.isclose(component, max_value)) > 1:
        return None
    if not math.isclose(max_value, 1.0, abs_tol=1e-3):
        return None
    return _AXIS_BY_VECTOR_INDEX[max_index]


def _axes_equivalent(
    expected: tuple[float, float, float],
    actual: tuple[float, float, float],
    *,
    ignore_sign: bool,
) -> bool:
    dot = sum(left * right for left, right in zip(expected, actual, strict=True))
    if ignore_sign:
        return abs(dot) >= _AXIS_DOT_TOLERANCE
    return dot >= _AXIS_DOT_TOLERANCE


def _axis_unit_vector(axis: str | None) -> tuple[float, float, float] | None:
    if axis is None:
        return None
    return _AXIS_UNIT_VECTORS.get(axis)


def _axis_vector_from_hint(value: Any) -> tuple[float, float, float] | None:
    vector = _axis_unit_vector(_normalize_axis(value))
    if vector is None:
        return None
    text = str(value).strip().lower() if value is not None else ""
    sign = -1.0 if text.startswith("-") else 1.0
    return (sign * vector[0], sign * vector[1], sign * vector[2])


def _vector_from_sequence(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, Sequence) or isinstance(value, str) or len(value) != 3:
        return None
    try:
        return (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError):
        return None


def _normalize_vector(
    vector: tuple[float, float, float] | None,
) -> tuple[float, float, float] | None:
    if vector is None:
        return None
    length = math.sqrt(sum(component * component for component in vector))
    if length <= 0:
        return None
    return (
        vector[0] / length,
        vector[1] / length,
        vector[2] / length,
    )


def _candidate_limits(candidate: Mapping[str, Any]) -> dict[str, float | None]:
    raw_limits = candidate.get("limits", {})
    limits = raw_limits if isinstance(raw_limits, Mapping) else {}
    return {
        "lower": _optional_float(
            _first_present(
                candidate.get("lower_limit"),
                candidate.get("lowerLimit"),
                limits.get("lower"),
                limits.get("lower_limit"),
                limits.get("lowerLimit"),
            )
        ),
        "upper": _optional_float(
            _first_present(
                candidate.get("upper_limit"),
                candidate.get("upperLimit"),
                limits.get("upper"),
                limits.get("upper_limit"),
                limits.get("upperLimit"),
            )
        ),
    }


def _reference_limits(reference: ReferenceJoint) -> dict[str, float | None]:
    return {"lower": reference.lower_limit, "upper": reference.upper_limit}


def _joint_type_from_prim(prim: Any, usd_physics: Any) -> str | None:
    schema_names = (
        ("RevoluteJoint", "revolute"),
        ("PrismaticJoint", "prismatic"),
        ("SphericalJoint", "spherical"),
        ("FixedJoint", "fixed"),
        ("D6Joint", "d6"),
        ("DistanceJoint", "distance"),
    )
    for schema_name, joint_type in schema_names:
        schema_type = getattr(usd_physics, schema_name, None)
        if schema_type is None:
            continue
        if prim.IsA(schema_type):
            return joint_type

    type_name = str(prim.GetTypeName())
    for schema_name, joint_type in schema_names:
        if type_name == f"Physics{schema_name}":
            return joint_type

    joint_schema = getattr(usd_physics, "Joint", None)
    if joint_schema is None or not prim.IsA(joint_schema):
        return None
    if not (type_name.startswith("Physics") and type_name.endswith("Joint")):
        return None
    normalized = _normalize_joint_type(type_name)
    return normalized if normalized not in {None, "unknown"} else None


def _validate_limit_tolerance(limit_tolerance: float) -> None:
    if limit_tolerance < 0:
        raise ValueError("limit_tolerance must be non-negative")


def _normalize_joint_type(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    text = text.removeprefix("physics")
    text = text.removesuffix("joint")
    text = text.strip("_:- ")
    return text or None


def _normalize_axis(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"", "unknown", "none", "null", "n/a", "na"}:
        return None
    text = text.removeprefix("+").removeprefix("-")
    return text if text in {"x", "y", "z"} else None


def _single_relationship_target(prim: Any, relationship_name: str) -> str | None:
    relationship = prim.GetRelationship(relationship_name)
    if not relationship or not relationship.IsValid():
        return None
    targets = relationship.GetTargets()
    if not targets:
        return None
    return str(targets[0])


def _authored_attribute_value(prim: Any, attribute_name: str) -> Any:
    attribute = prim.GetAttribute(attribute_name)
    if (
        not attribute
        or not attribute.IsValid()
        or not attribute.HasAuthoredValueOpinion()
    ):
        return None
    return _json_safe_value(attribute.Get())


def _effective_attribute_value(prim: Any, attribute_name: str) -> Any:
    attribute = prim.GetAttribute(attribute_name)
    if not attribute or not attribute.IsValid():
        return None
    return _json_safe_value(attribute.Get())


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _reference_axis_world(
    stage: Any,
    joint_prim: Any,
    body0: str | None,
    body1: str | None,
    axis: str | None,
) -> list[float] | None:
    vector = _axis_unit_vector(axis)
    if vector is None:
        return None

    if body0:
        vector = _rotate_vector_by_quaternion(
            vector,
            _authored_attribute_raw(joint_prim, "physics:localRot0"),
        )
        vector = _transform_direction_by_body(stage, body0, vector)
    else:
        # Frame 0 is world-space for joints anchored to the world. Preserve its
        # authored local rotation before considering the body1-side frame.
        local_rot0 = _authored_attribute_raw(joint_prim, "physics:localRot0")
        if local_rot0 is not None:
            vector = _rotate_vector_by_quaternion(vector, local_rot0)
        else:
            vector = _rotate_vector_by_quaternion(
                vector,
                _authored_attribute_raw(joint_prim, "physics:localRot1"),
            )
            if body1:
                vector = _transform_direction_by_body(stage, body1, vector)
    normalized = _normalize_vector(vector)
    return list(normalized) if normalized is not None else None


def _rotate_vector_by_quaternion(
    vector: tuple[float, float, float],
    quaternion: Any,
) -> tuple[float, float, float]:
    if quaternion is None:
        return vector
    from pxr import Gf

    rotated = Gf.Rotation(quaternion).TransformDir(Gf.Vec3d(*vector))
    return (float(rotated[0]), float(rotated[1]), float(rotated[2]))


def _transform_direction_by_body(
    stage: Any,
    body_path: str,
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    prim = stage.GetPrimAtPath(body_path)
    if not prim or not prim.IsValid():
        return vector
    from pxr import Gf, UsdGeom

    matrix = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    transformed = matrix.TransformDir(Gf.Vec3d(*vector))
    return (float(transformed[0]), float(transformed[1]), float(transformed[2]))


def _authored_attribute_raw(prim: Any, attribute_name: str) -> Any:
    attribute = prim.GetAttribute(attribute_name)
    if (
        not attribute
        or not attribute.IsValid()
        or not attribute.HasAuthoredValueOpinion()
    ):
        return None
    return attribute.Get()


def _extract_authored_metadata(
    stage: Any,
    prim: Any,
    body0: str | None,
    body1: str | None,
) -> dict[str, Any]:
    authored_properties = sorted(
        prop.GetName() for prop in prim.GetAuthoredProperties()
    )
    physics_attributes = {}
    for property_name in authored_properties:
        if not property_name.startswith("physics:"):
            continue
        value = _authored_attribute_value(prim, property_name)
        if value is not None:
            physics_attributes[property_name] = value

    return {
        "usd_type_name": str(prim.GetTypeName()),
        "applied_schemas": list(prim.GetAppliedSchemas()),
        "authored_properties": authored_properties,
        "physics_attributes": physics_attributes,
        "body0_applied_schemas": _applied_schemas_for_path(stage, body0),
        "body1_applied_schemas": _applied_schemas_for_path(stage, body1),
    }


def _applied_schemas_for_path(stage: Any, prim_path: str | None) -> list[str]:
    if not prim_path:
        return []
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return []
    return list(prim.GetAppliedSchemas())


def _json_safe_value(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, bytes):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, Sequence):
        return [_json_safe_value(item) for item in value]
    if hasattr(value, "__iter__"):
        return [_json_safe_value(item) for item in value]
    return str(value)


def _e(value: Any) -> str:
    return html.escape(str(value), quote=True)
