# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage 1 articulation schema helpers for Joint Agent predictions."""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from world_understanding.functions.classification.inference import extract_answer_block
from world_understanding.utils.llm_parsing import iter_json_dicts_from_llm_response

from joint_agent.functions.axis_hints import normalize_axis_hint_token
from joint_agent.prompts.prop_articulation import SUPPORTED_PROP_ROLES

STAGE1_SCHEMA_VERSION: Literal["joint-agent-stage1-v0"] = "joint-agent-stage1-v0"
PropRole = Literal[
    "body",
    "drawer",
    "door",
    "lid",
    "wheel",
    "caster_frame",
    "knob",
    "unknown",
]
JointTypeHint = Literal[
    "revolute",
    "prismatic",
    "spherical",
    "fixed",
    "none",
    "unknown",
]
AxisHint = Literal[
    "x",
    "y",
    "z",
    "+x",
    "-x",
    "+y",
    "-y",
    "+z",
    "-z",
    "unknown",
]
ConfidenceLevel = Literal["high", "medium", "low"]
ProvenanceSource = Literal[
    "predicted",
    "consistency_corrected",
    "llm_adjudicated",
    "template_default",
    "geometry_inferred",
    "unknown",
]
RiggerEvidenceSource = Literal[
    "predicted",
    "consistency_corrected",
    "llm_adjudicated",
    "unknown",
]
LimitEvidenceSource = Literal[
    "authored_metadata",
    "authored_reference",
    "source_metadata",
    "accepted_manifest",
    "template_default",
    "predicted",
    "unknown",
]
LimitUnit = Literal["degrees", "radians", "meters", "unknown"]


class Stage1Provenance(BaseModel):
    """Field-source provenance accepted by the Stage 1 contract."""

    model_config = ConfigDict(extra="allow")

    field_sources: dict[str, ProvenanceSource] = Field(default_factory=dict)


class Stage1RiggerEvidenceClaim(BaseModel):
    """Rigger-facing evidence claim for one Stage 1 field."""

    model_config = ConfigDict(extra="allow")

    value: str = "unknown"
    prim_paths: list[str] = Field(default_factory=list)
    confidence: ConfidenceLevel = "low"
    rationale: str = ""
    source: RiggerEvidenceSource = "predicted"


class Stage1CompoundEdgeEvidence(BaseModel):
    """Explicit Stage 1 evidence for one compound articulation edge.

    Endpoint claim dictionaries are normalized to string values here; nested
    endpoint rationale/provenance is intentionally not preserved until compound
    edge promotion has an owner-approved downstream contract.
    """

    model_config = ConfigDict(extra="allow")

    body0: str = "unknown"
    body1: str = "unknown"
    joint_type_hint: JointTypeHint = "unknown"
    axis_hint: AxisHint = "unknown"
    confidence: ConfidenceLevel = "low"
    rationale: str = ""
    source: RiggerEvidenceSource = "predicted"
    prim_paths: list[str] = Field(default_factory=list)


class Stage1MotionLimitEvidence(BaseModel):
    """Optional source-backed joint travel limits for one Stage 1 candidate."""

    model_config = ConfigDict(extra="allow")

    lower_limit: float | None = None
    upper_limit: float | None = None
    unit: LimitUnit = "unknown"
    source: LimitEvidenceSource = "unknown"
    rationale: str = ""


class Stage1RiggerEvidence(BaseModel):
    """Optional explicit evidence for downstream rigger-facing fields."""

    model_config = ConfigDict(extra="allow")

    body0: Stage1RiggerEvidenceClaim | None = None
    body1: Stage1RiggerEvidenceClaim | None = None
    motion_axis: Stage1RiggerEvidenceClaim | None = None
    compound_edges: list[Stage1CompoundEdgeEvidence] = Field(default_factory=list)
    limits: Stage1MotionLimitEvidence | None = None


class Stage1PredictionContract(BaseModel):
    """Backward-compatible per-prim Stage 1 articulation prediction payload."""

    model_config = ConfigDict(extra="allow")

    schema_version: Literal["joint-agent-stage1-v0"] = STAGE1_SCHEMA_VERSION
    asset_type: str = "unknown"
    component_type: str = "unknown"
    component_name: str = "unknown"
    role: PropRole = "unknown"
    instance_id: str | None = None
    is_articulation_candidate: bool = False
    joint_type_hint: JointTypeHint = "unknown"
    axis_hint: AxisHint = "unknown"
    parent_hint: str = "unknown"
    child_hint: str = "unknown"
    material: str = "unknown"
    confidence: ConfidenceLevel = "low"
    evidence: str = ""
    reasoning: str = ""
    provenance: Stage1Provenance | None = None
    rigger_evidence: Stage1RiggerEvidence | None = None


STAGE1_FIELDS = (
    "schema_version",
    "asset_type",
    "component_type",
    "component_name",
    "role",
    "instance_id",
    "is_articulation_candidate",
    "joint_type_hint",
    "axis_hint",
    "parent_hint",
    "child_hint",
    "material",
    "confidence",
    "evidence",
    "reasoning",
    "provenance",
    "rigger_evidence",
)

# Recognized Stage 1 fields may sit beside a wrapped classification object.
# They fill missing nested values without replacing the nested prediction.
_WRAPPER_MERGE_FIELDS = tuple(
    field for field in STAGE1_FIELDS if field != "schema_version"
)
_LIMIT_LOWER_ALIASES = ("lower_limit", "lower", "lowerLimit")
_LIMIT_UPPER_ALIASES = ("upper_limit", "upper", "upperLimit")
# Nested ``limits`` objects prefer canonical contract fields. Top-level wrapper
# aliases prefer explicit ``limit_*`` names so unrelated payload-level fields
# such as ``source`` do not override dedicated limit metadata.
_LIMIT_UNIT_ALIASES = ("unit", "limit_unit", "limitUnit")
_LIMIT_SOURCE_ALIASES = ("source", "limit_source", "limitSource")
_WRAPPER_LIMIT_UNIT_ALIASES = ("limit_unit", "unit", "limitUnit")
_WRAPPER_LIMIT_SOURCE_ALIASES = ("limit_source", "source", "limitSource")
_LIMIT_RATIONALE_ALIASES = ("rationale", "limit_rationale", "limitRationale")
_WRAPPER_LIMIT_RATIONALE_ALIASES = (
    "limit_rationale",
    "rationale",
    "limitRationale",
)
_LIMIT_EVIDENCE_ALIAS_FIELDS = (
    *_LIMIT_LOWER_ALIASES,
    *_LIMIT_UPPER_ALIASES,
    *_LIMIT_UNIT_ALIASES,
    *_LIMIT_SOURCE_ALIASES,
    *_LIMIT_RATIONALE_ALIASES,
)
_RIGGER_EVIDENCE_ALIAS_FIELDS = (
    "body0",
    "body1",
    "fixed_parent_prim",
    "moving_body_prim",
    "motion_axis",
    "compound_edges",
    "limits",
    *_LIMIT_EVIDENCE_ALIAS_FIELDS,
)
_RIGGER_ENDPOINT_ALIAS_GROUPS = {
    "body0": ("body0", "fixed_parent_prim"),
    "fixed_parent_prim": ("body0", "fixed_parent_prim"),
    "body1": ("body1", "moving_body_prim"),
    "moving_body_prim": ("body1", "moving_body_prim"),
}
_STAGE1_SIGNAL_FIELDS = {
    "asset_type",
    "component_type",
    "component_name",
    "role",
    "is_articulation_candidate",
    "joint_type_hint",
    "axis_hint",
    "parent_hint",
    "child_hint",
}

_MATERIAL_LABELS = {
    "metal",
    "steel",
    "aluminum",
    "plastic",
    "rubber",
    "glass",
    "ceramic",
    "fabric",
    "wood",
    "composite",
}

_TRUE_VALUES = {"true", "yes", "y", "1", "candidate", "articulated"}
_FALSE_VALUES = {"false", "no", "n", "0", "none", "fixed", "static"}
_UNKNOWN_VALUES = {"", "unknown", "none", "null", "n/a", "na"}

_MOVING_JOINT_HINTS = {"revolute", "prismatic", "spherical"}
_STATIC_JOINT_HINTS = {"fixed", "none"}
_VALID_JOINT_HINTS = {
    "revolute",
    "prismatic",
    "spherical",
    "fixed",
    "none",
    "unknown",
}
_ROLE_ALIASES = {
    "base": "body",
    "body": "body",
    "cabinet": "body",
    "cabinet_body": "body",
    "cart": "body",
    "chassis": "body",
    "frame": "body",
    "housing": "body",
    "main_body": "body",
    "support": "body",
    "drawer": "drawer",
    "drawer_body": "drawer",
    "drawer_front": "drawer",
    "slider": "drawer",
    "sliding_drawer": "drawer",
    "cabinet_door": "door",
    "door": "door",
    "door_body": "door",
    "door_leaf": "door",
    "door_panel": "door",
    "hinged_door": "door",
    "hinged_cabinet_door": "door",
    "single_hinged_door": "door",
    "swing_door": "door",
    "cover_lid": "lid",
    "hinged_lid": "lid",
    "lid": "lid",
    "lid_body": "lid",
    "lid_panel": "lid",
    "open_close_lid": "lid",
    "caster": "caster_frame",
    "caster_bracket": "caster_frame",
    "caster_fork": "caster_frame",
    "caster_frame": "caster_frame",
    "caster_mount": "caster_frame",
    "fork": "caster_frame",
    "mounting_plate": "caster_frame",
    "roller": "wheel",
    "rim": "wheel",
    "tire": "wheel",
    "wheel": "wheel",
    "dial": "knob",
    "knob": "knob",
    "rotary_control": "knob",
    "unknown": "unknown",
}
_WEAK_COMPONENT_TYPES = {"", "unknown", "other", "component", "part", "mesh", "object"}
_NON_MOVING_DOOR_LID_BODY_TOKENS = {
    "frame",
    "jamb",
    "sill",
    "threshold",
    "support",
}
# Supported controls such as door knobs are preserved before this blocker set is
# applied. These tokens prevent door/lid phrase fallback from promoting fixed,
# decorative, hardware, or deferred control parts to moving panel roles.
_NON_MOVING_DOOR_LID_UNKNOWN_TOKENS = {
    "bracket",
    "button",
    "buttons",
    "fastener",
    "fasteners",
    "faucet",
    "faucets",
    "gasket",
    "handle",
    "hardware",
    "hinge",
    "knob",
    "latch",
    "lever",
    "levers",
    "lock",
    "mount",
    "molding",
    "moulding",
    "pin",
    "rack",
    "racks",
    "screw",
    "screws",
    "seal",
    "shelf",
    "shelves",
    "spout",
    "spouts",
    "switch",
    "switches",
    "trim",
}
_DOOR_LID_SUPPORTED_CONTROL_ROLES = {"knob"}


def normalize_stage1_prediction_payload(
    payload: Any,
    *,
    output_key: str = "classification",
) -> Any:
    """Normalize a parsed VLM payload into the Joint Agent Stage 1 contract.

    The shared classification parser has material-agent compatibility behavior:
    when ``output_key`` is ``classification`` and it finds a top-level
    ``material`` field, it may move that material value into
    ``classification``. For Joint Agent 0.5, the outer prediction row still uses
    ``classification`` for compatibility, but the inner payload should be a
    joint-analysis object and should not contain another ambiguous
    ``classification`` field.
    """
    if not isinstance(payload, dict):
        return payload

    original_payload = copy.deepcopy(payload)
    body = _unwrap_output_payload(original_payload, output_key)
    recovered = _recover_stage1_json(body.get("original_response"), output_key)

    merged: dict[str, Any] = {}
    if recovered:
        merged.update(recovered)

    legacy_label = None
    for key, value in body.items():
        if key == output_key:
            legacy_label = value
            continue
        merged[key] = value

    if isinstance(legacy_label, str) and legacy_label.strip():
        label = legacy_label.strip()
        if _looks_like_material_label(label) and not merged.get("material"):
            merged["material"] = label
        elif (
            not merged.get("component_type")
            or merged.get("component_type") == "unknown"
        ):
            merged["component_type"] = label
        else:
            merged["legacy_label"] = label

    if (  # pragma: no cover - nested unwrap and merge paths copy this first.
        "original_response" not in merged and "original_response" in original_payload
    ):
        merged["original_response"] = original_payload["original_response"]

    return validate_stage1_prediction_payload(_fill_stage1_defaults(merged)).model_dump(
        mode="json",
        exclude_none=True,
    )


def unwrap_stage1_prediction_payload(
    payload: dict[str, Any],
    *,
    output_key: str = "classification",
) -> dict[str, Any]:
    """Unwrap a Stage 1 output-key wrapper without defaulting field values.

    Nested payload fields stay authoritative. Recognized wrapper fields fill
    only missing nested fields, and wrapper ``rigger_evidence`` is copied only
    when it contains exact prim-path evidence and the nested evidence is empty
    or lacks exact prim-path evidence. Top-level rigger-facing aliases such as
    ``body0``, ``body1``, ``fixed_parent_prim``, ``moving_body_prim``,
    ``motion_axis``, and ``compound_edges`` are folded into canonical
    ``rigger_evidence`` when they fill missing or weaker nested evidence.
    Copied rigger evidence is normalized so Stage 2 sees canonical claim shapes
    such as ``prim_paths``.
    """
    return _unwrap_output_payload(payload, output_key)


def validate_stage1_prediction_payload(
    payload: dict[str, Any],
) -> Stage1PredictionContract:
    """Validate a normalized payload against the Stage 1 contract."""
    return Stage1PredictionContract.model_validate(payload)


def stage1_prediction_json_schema() -> dict[str, Any]:
    """Return the JSON schema for the Stage 1 prediction contract."""
    return Stage1PredictionContract.model_json_schema()


def _unwrap_output_payload(payload: dict[str, Any], output_key: str) -> dict[str, Any]:
    nested = payload.get(output_key)
    if not isinstance(nested, dict):
        return _with_rigger_evidence_aliases(copy.deepcopy(payload))

    unwrapped = copy.deepcopy(nested)
    if "original_response" not in unwrapped and "original_response" in payload:
        unwrapped["original_response"] = payload["original_response"]
    unwrapped = _with_rigger_evidence_aliases(unwrapped)
    for wrapper_key in _WRAPPER_MERGE_FIELDS:
        if wrapper_key == "rigger_evidence":
            wrapper_evidence = _normalize_rigger_evidence(payload.get(wrapper_key))
            wrapper_evidence_has_prim_path = False
            if wrapper_evidence is not None:
                wrapper_evidence_has_prim_path = _rigger_evidence_has_prim_path(
                    wrapper_evidence
                )
                wrapper_evidence = _wrapper_rigger_evidence_for_merge(wrapper_evidence)
            if wrapper_evidence is not None:
                nested_evidence = (
                    _normalize_rigger_evidence(unwrapped.get(wrapper_key)) or {}
                )
                merged_evidence = _merge_rigger_evidence(
                    nested_evidence,
                    wrapper_evidence,
                    allow_pathless_motion_axis_insert=wrapper_evidence_has_prim_path,
                )
                if merged_evidence:
                    unwrapped[wrapper_key] = merged_evidence
            continue
        if _should_copy_wrapper_field(unwrapped, payload, wrapper_key):
            unwrapped[wrapper_key] = copy.deepcopy(payload[wrapper_key])
    wrapper_limits = _wrapper_motion_limit_evidence(payload)
    if wrapper_limits is not None:
        normalized_nested_evidence = _normalize_rigger_evidence(
            unwrapped.get("rigger_evidence")
        )
        merged_evidence = _merge_rigger_evidence(
            normalized_nested_evidence or {},
            {"limits": wrapper_limits},
        )
        if merged_evidence:
            unwrapped["rigger_evidence"] = merged_evidence
    for alias_key in _RIGGER_EVIDENCE_ALIAS_FIELDS:
        if alias_key == "limits" or alias_key in _LIMIT_EVIDENCE_ALIAS_FIELDS:
            continue
        if alias_key in payload and _should_copy_wrapper_rigger_alias(
            unwrapped,
            payload.get(alias_key),
            alias_key,
        ):
            unwrapped[alias_key] = copy.deepcopy(payload[alias_key])
    return _with_rigger_evidence_aliases(unwrapped)


def _should_copy_wrapper_field(
    unwrapped: dict[str, Any],
    payload: dict[str, Any],
    wrapper_key: str,
) -> bool:
    if wrapper_key not in payload:
        return False
    if wrapper_key not in unwrapped:
        return True
    return False


def _wrapper_rigger_evidence_for_merge(
    wrapper_evidence: dict[str, Any],
) -> dict[str, Any] | None:
    if _rigger_evidence_has_prim_path(wrapper_evidence):
        return wrapper_evidence

    merge_evidence: dict[str, Any] = {}
    motion_axis = _normalize_rigger_evidence_claim(wrapper_evidence.get("motion_axis"))
    if motion_axis is not None and _motion_axis_claim_is_explicit(motion_axis):
        merge_evidence["motion_axis"] = motion_axis

    limits = _normalize_motion_limit_evidence(wrapper_evidence.get("limits"))
    if limits is not None:
        merge_evidence["limits"] = limits
    return merge_evidence or None


def _rigger_evidence_has_prim_path(evidence: dict[str, Any]) -> bool:
    for claim_key in ("body0", "body1", "motion_axis"):
        claim = evidence.get(claim_key)
        if isinstance(claim, dict) and _rigger_claim_has_prim_path(claim):
            return True

    edges = evidence.get("compound_edges")
    if isinstance(edges, list):
        for edge in edges:
            if isinstance(edge, dict) and edge.get("prim_paths"):
                return True
    return False


def _rigger_claim_has_prim_path(claim: dict[str, Any]) -> bool:
    claim_value = _clean_string(claim.get("value"), "")
    return claim_value.startswith("/") or bool(claim.get("prim_paths"))


def _motion_axis_claim_is_explicit(claim: dict[str, Any]) -> bool:
    return normalize_axis_hint_token(claim.get("value")) is not None


def _should_replace_motion_axis_claim(
    nested_claim: dict[str, Any],
    alias_claim: dict[str, Any],
) -> bool:
    if _rigger_claim_has_prim_path(alias_claim) and not _rigger_claim_has_prim_path(
        nested_claim
    ):
        return True
    return _motion_axis_claim_is_explicit(alias_claim) and not (
        _motion_axis_claim_is_explicit(nested_claim)
    )


def _should_copy_wrapper_rigger_alias(
    unwrapped: dict[str, Any],
    wrapper_value: Any,
    alias_key: str,
) -> bool:
    if alias_key in {"body0", "body1", "fixed_parent_prim", "moving_body_prim"}:
        wrapper_claim = _normalize_rigger_evidence_claim(wrapper_value)
        if wrapper_claim is None or not _rigger_claim_has_prim_path(wrapper_claim):
            return False
        nested_claims = [
            _normalize_rigger_evidence_claim(unwrapped.get(nested_alias))
            for nested_alias in _RIGGER_ENDPOINT_ALIAS_GROUPS[alias_key]
            if nested_alias in unwrapped
        ]
        return not any(
            nested_claim is not None and _rigger_claim_has_prim_path(nested_claim)
            for nested_claim in nested_claims
        )

    if alias_key == "motion_axis":
        wrapper_claim = _normalize_rigger_evidence_claim(wrapper_value)
        return (
            wrapper_claim is not None
            and _normalize_rigger_evidence_claim(unwrapped.get(alias_key)) is None
        )

    if alias_key == "compound_edges":
        wrapper_edges = _normalize_compound_edge_evidence_list(wrapper_value)
        if not wrapper_edges or not _compound_edges_have_prim_path(wrapper_edges):
            return False
        nested_edges = _normalize_compound_edge_evidence_list(unwrapped.get(alias_key))
        return not nested_edges or not _compound_edges_have_prim_path(nested_edges)

    return False


def _wrapper_motion_limit_evidence(payload: dict[str, Any]) -> dict[str, Any] | None:
    limits = _normalize_motion_limit_evidence(payload.get("limits"))
    if limits is not None:
        return limits
    return _normalize_motion_limit_evidence(_limit_evidence_from_aliases(payload))


def _with_rigger_evidence_aliases(payload: dict[str, Any]) -> dict[str, Any]:
    """Fold top-level rigger aliases into ``payload`` in place."""
    alias_evidence = _rigger_evidence_from_top_level_aliases(payload)
    if alias_evidence is None:
        return payload

    nested_evidence = _normalize_rigger_evidence(payload.get("rigger_evidence")) or {}
    merged_evidence = _merge_rigger_evidence(nested_evidence, alias_evidence)
    if merged_evidence:
        payload["rigger_evidence"] = merged_evidence
    return payload


def _rigger_evidence_from_top_level_aliases(
    payload: dict[str, Any],
) -> dict[str, Any] | None:
    evidence: dict[str, Any] = {}
    for field, aliases in (
        ("body0", ("body0", "fixed_parent_prim")),
        ("body1", ("body1", "moving_body_prim")),
        ("motion_axis", ("motion_axis",)),
    ):
        for alias in aliases:
            if alias not in payload:
                continue
            claim = _normalize_rigger_evidence_claim(payload.get(alias))
            if claim is not None and (
                field == "motion_axis" or _rigger_claim_has_prim_path(claim)
            ):
                evidence[field] = claim
                break

    raw_edges = payload.get("compound_edges")
    compound_edges = _normalize_compound_edge_evidence_list(raw_edges)
    if compound_edges and _compound_edges_have_prim_path(compound_edges):
        evidence["compound_edges"] = compound_edges
    limits = _normalize_motion_limit_evidence(payload.get("limits"))
    if limits is None:
        limits = _normalize_motion_limit_evidence(_limit_evidence_from_aliases(payload))
    if limits is not None:
        evidence["limits"] = limits
    return evidence or None


def _merge_rigger_evidence(
    nested_evidence: dict[str, Any],
    alias_evidence: dict[str, Any],
    *,
    allow_pathless_motion_axis_insert: bool = True,
) -> dict[str, Any]:
    merged = copy.deepcopy(nested_evidence)
    for field in ("body0", "body1"):
        alias_claim = alias_evidence.get(field)
        if not isinstance(alias_claim, dict):
            continue
        if not _rigger_claim_has_prim_path(alias_claim):
            continue
        nested_claim = merged.get(field)
        if not isinstance(nested_claim, dict) or (
            not _rigger_claim_has_prim_path(nested_claim)
        ):
            merged[field] = copy.deepcopy(alias_claim)

    alias_motion_axis = alias_evidence.get("motion_axis")
    if isinstance(alias_motion_axis, dict):
        nested_motion_axis = merged.get("motion_axis")
        if (
            not isinstance(nested_motion_axis, dict)
            and (
                allow_pathless_motion_axis_insert
                or _rigger_claim_has_prim_path(alias_motion_axis)
            )
        ) or (
            isinstance(nested_motion_axis, dict)
            and _should_replace_motion_axis_claim(nested_motion_axis, alias_motion_axis)
        ):
            merged["motion_axis"] = copy.deepcopy(alias_motion_axis)

    alias_edges = alias_evidence.get("compound_edges")
    nested_edges = merged.get("compound_edges")
    if isinstance(alias_edges, list) and (
        not nested_edges
        or (
            not _compound_edges_have_prim_path(nested_edges)
            and _compound_edges_have_prim_path(alias_edges)
        )
    ):
        merged_edges: list[dict[str, Any]] = []
        for alias_edge in alias_edges:
            if isinstance(alias_edge, dict) and alias_edge not in merged_edges:
                merged_edges.append(copy.deepcopy(alias_edge))
        if merged_edges:
            merged["compound_edges"] = merged_edges

    alias_limits = _normalize_motion_limit_evidence(alias_evidence.get("limits"))
    nested_limits = _normalize_motion_limit_evidence(merged.get("limits"))
    if alias_limits is not None and (
        nested_limits is None
        or (
            not _limit_evidence_is_source_backed(nested_limits)
            and _limit_evidence_is_source_backed(alias_limits)
        )
    ):
        merged["limits"] = alias_limits
    return merged


def _limit_evidence_is_source_backed(limits: dict[str, Any]) -> bool:
    return limits.get("source") in {
        "authored_metadata",
        "authored_reference",
        "source_metadata",
        "accepted_manifest",
        "template_default",
    }


def _normalize_compound_edge_evidence_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [
        edge
        for raw_edge in value
        if (edge := _normalize_compound_edge_evidence(raw_edge)) is not None
    ]


def _compound_edges_have_prim_path(value: Any) -> bool:
    if not isinstance(value, list):
        return False
    return any(
        isinstance(edge, dict) and bool(edge.get("prim_paths")) for edge in value
    )


def _recover_stage1_json(
    original_response: Any,
    output_key: str,
) -> dict[str, Any] | None:
    if not isinstance(original_response, str) or not original_response.strip():
        return None

    answer = extract_answer_block(original_response)
    search_texts = [answer, original_response] if answer else [original_response]
    for text in search_texts:
        if not text:  # pragma: no cover - empty answers are filtered above.
            continue
        for candidate in iter_json_dicts_from_llm_response(text):
            candidate = _unwrap_output_payload(candidate, output_key)
            if _has_stage1_signal(candidate):
                return candidate
    return None


def _iter_complete_outer_json_dicts(text: str) -> list[dict[str, Any]]:
    """Return balanced outer JSON objects without recovering nested fragments."""
    results: list[dict[str, Any]] = []
    depth = 0
    start: int | None = None
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth != 0 or start is None:
                continue
            raw_candidate = text[start : index + 1]
            attempts = [raw_candidate]
            stripped = raw_candidate.strip()
            if stripped.startswith("{{") and stripped.endswith("}}"):
                attempts.append(stripped[1:-1])
            for attempt in attempts:
                try:
                    candidate = json.loads(attempt)
                except json.JSONDecodeError:
                    continue
                if isinstance(candidate, dict):
                    results.append(candidate)
                    break
            start = None
    return results


def _recover_complete_stage1_json(
    original_response: Any,
    output_key: str,
) -> dict[str, Any] | None:
    """Recover only a complete outer Stage 1 object from the source response."""
    if not isinstance(original_response, str) or not original_response.strip():
        return None

    def code_fence_contents(text: str) -> list[str]:
        contents: list[str] = []
        search_start = 0
        while True:
            fence_start = text.find("```", search_start)
            if fence_start == -1:
                break
            content_start = fence_start + 3
            fence_end = text.find("```", content_start)
            if fence_end == -1:
                break
            fenced = text[content_start:fence_end].strip()
            if fenced.lower().startswith("json") and (
                len(fenced) == 4 or fenced[4].isspace()
            ):
                fenced = fenced[4:].strip()
            contents.append(fenced)
            search_start = fence_end + 3
        return contents

    # Match ``iter_json_dicts_from_llm_response`` selection priority used by
    # normalization: fenced objects in the last answer, the whole last answer,
    # then fences and prose from the full response.
    search_texts: list[str] = []
    answer = extract_answer_block(original_response)
    if answer:
        search_texts.extend(code_fence_contents(answer))
        search_texts.append(answer)
    search_texts.extend(code_fence_contents(original_response))
    search_texts.append(original_response)

    for text in search_texts:
        if not text:  # pragma: no cover - empty answers are filtered above.
            continue
        for candidate in _iter_complete_outer_json_dicts(text):
            candidate = _unwrap_output_payload(candidate, output_key)
            if _has_stage1_signal(candidate):
                return candidate
    return None


def _has_stage1_signal(payload: dict[str, Any]) -> bool:
    return any(field in payload for field in _STAGE1_SIGNAL_FIELDS)


def has_parseable_stage1_source_response(
    payload_or_response: Any,
    *,
    output_key: str = "classification",
) -> bool:
    """Return whether the raw model response contains a complete Stage 1 JSON.

    Normalization intentionally fills contract defaults after malformed or
    truncated model output. Inference completion uses this pre-default signal
    to retry only rows whose source response never contained a parseable Stage
    1 object, while preserving complete responses that explicitly choose the
    supported ``unknown`` role.
    """
    if isinstance(payload_or_response, Mapping):
        body = _unwrap_output_payload(dict(payload_or_response), output_key)
        original_response = body.get("original_response")
    else:
        original_response = payload_or_response
    recovered = _recover_complete_stage1_json(original_response, output_key)
    if recovered is None:
        return False

    if "schema_version" in recovered:
        schema_version = recovered["schema_version"]
        return (
            isinstance(schema_version, str) and schema_version == STAGE1_SCHEMA_VERSION
        )

    # Legacy/custom prompts predate the versioned Stage 1 response contract.
    # The strict outer-object recovery above keeps them compatible without
    # accepting a balanced nested fragment from a truncated response.
    return True


def _fill_stage1_defaults(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    normalized["schema_version"] = STAGE1_SCHEMA_VERSION

    for field in ("asset_type", "component_type", "component_name"):
        normalized[field] = _clean_string(normalized.get(field), "unknown")

    normalized["role"] = _normalize_role(
        normalized.get("role"),
        component_type=normalized.get("component_type"),
        component_name=normalized.get("component_name"),
    )
    if "instance_id" in normalized:
        normalized["instance_id"] = (
            _clean_string(
                normalized.get("instance_id"),
                "",
            )
            or None
        )

    joint_hint = _normalize_joint_hint(normalized.get("joint_type_hint"))
    normalized["joint_type_hint"] = joint_hint
    normalized["is_articulation_candidate"] = _normalize_candidate_flag(
        normalized.get("is_articulation_candidate"),
        joint_hint=joint_hint,
    )
    normalized["axis_hint"] = _normalize_axis_hint(normalized.get("axis_hint"))

    for field in ("parent_hint", "child_hint", "material"):
        normalized[field] = _clean_string(normalized.get(field), "unknown")

    normalized["confidence"] = _normalize_confidence(normalized.get("confidence"))

    if not normalized.get("evidence") and normalized.get("reasoning"):
        normalized["evidence"] = normalized["reasoning"]
    normalized["evidence"] = _clean_string(normalized.get("evidence"), "")
    normalized["reasoning"] = _clean_string(normalized.get("reasoning"), "")

    rigger_evidence = _normalize_rigger_evidence(normalized.get("rigger_evidence"))
    if rigger_evidence is None:
        normalized.pop("rigger_evidence", None)
    else:
        normalized["rigger_evidence"] = rigger_evidence

    return normalized


def _normalize_candidate_flag(value: Any, *, joint_hint: str) -> bool:
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
    if joint_hint in _MOVING_JOINT_HINTS:
        return True
    if joint_hint in _STATIC_JOINT_HINTS:
        return False
    return False


def _normalize_joint_hint(value: Any) -> str:
    hint = _clean_string(value, "unknown").lower().replace(" ", "_")
    if hint in {"hinge", "rotary"}:
        return "revolute"
    if hint in {"slider", "linear"}:
        return "prismatic"
    if hint in {"ball", "ball_joint"}:
        return "spherical"
    if hint in {"static", "rigid"}:
        return "fixed"
    if hint in _VALID_JOINT_HINTS:
        return hint
    return "unknown"


def _normalize_axis_hint(value: Any) -> str:
    return normalize_axis_hint_token(value) or "unknown"


def _normalize_confidence(value: Any) -> str:
    confidence = _clean_string(value, "low").lower()
    if confidence in {"high", "medium", "low"}:
        return confidence
    if confidence in {"certain", "strong"}:
        return "high"
    if confidence in {"moderate", "mid"}:
        return "medium"
    return "low"


def _normalize_rigger_evidence(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None

    normalized: dict[str, Any] = {}
    for field in ("body0", "body1", "motion_axis"):
        claim = _normalize_rigger_evidence_claim(value.get(field))
        if claim is not None:
            normalized[field] = claim

    compound_edges: list[dict[str, Any]] = []
    raw_edges = value.get("compound_edges")
    if isinstance(raw_edges, list):
        for raw_edge in raw_edges:
            edge = _normalize_compound_edge_evidence(raw_edge)
            if edge is not None:
                compound_edges.append(edge)
    if compound_edges:
        normalized["compound_edges"] = compound_edges

    limits = _normalize_motion_limit_evidence(value.get("limits"))
    if limits is None:
        limits = _normalize_motion_limit_evidence(_limit_evidence_from_aliases(value))
    if limits is not None:
        normalized["limits"] = limits

    return normalized or None


def _normalize_rigger_evidence_claim(value: Any) -> dict[str, Any] | None:
    if isinstance(value, str):
        value = {"value": value}
    if not isinstance(value, dict):
        return None

    prim_paths = [
        *_normalize_prim_path_list(value.get("prim_path")),
        *_normalize_prim_path_list(value.get("prim_paths")),
    ]
    prim_paths = _dedupe_preserving_order(prim_paths)

    claim_value = _clean_string(value.get("value"), "unknown")
    if claim_value.lower() in _UNKNOWN_VALUES and len(prim_paths) == 1:
        claim_value = prim_paths[0]

    rationale = _clean_string(value.get("rationale"), "")
    if claim_value.lower() in _UNKNOWN_VALUES and not prim_paths:
        return None

    return {
        "value": claim_value,
        "prim_paths": prim_paths,
        "confidence": _normalize_confidence(value.get("confidence")),
        "rationale": rationale,
        "source": _normalize_rigger_evidence_source(value.get("source")),
    }


def _normalize_compound_edge_evidence(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None

    body0 = _rigger_edge_endpoint(value, "body0", "fixed_parent_prim")
    body1 = _rigger_edge_endpoint(value, "body1", "moving_body_prim")
    prim_paths = _dedupe_preserving_order(
        [
            *(_normalize_prim_path_list(value.get("prim_paths"))),
            *([body0] if body0.startswith("/") else []),
            *([body1] if body1.startswith("/") else []),
        ]
    )
    rationale = _clean_string(value.get("rationale"), "")
    if (
        body0.lower() in _UNKNOWN_VALUES
        and body1.lower() in _UNKNOWN_VALUES
        and not prim_paths
        and not rationale
    ):
        return None

    return {
        "body0": body0,
        "body1": body1,
        "joint_type_hint": _normalize_joint_hint(value.get("joint_type_hint")),
        "axis_hint": _normalize_axis_hint(value.get("axis_hint")),
        "confidence": _normalize_confidence(value.get("confidence")),
        "rationale": rationale,
        "source": _normalize_rigger_evidence_source(value.get("source")),
        "prim_paths": prim_paths,
    }


def _normalize_motion_limit_evidence(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None

    lower_limit = _optional_float(_first_present(value, *_LIMIT_LOWER_ALIASES))
    upper_limit = _optional_float(_first_present(value, *_LIMIT_UPPER_ALIASES))
    if lower_limit is None and upper_limit is None:
        return None
    rationale = _clean_string(value.get("rationale"), "")

    return {
        "lower_limit": lower_limit,
        "upper_limit": upper_limit,
        "unit": _normalize_limit_unit(_first_present(value, *_LIMIT_UNIT_ALIASES)),
        "source": _normalize_limit_evidence_source(
            _first_present(value, *_LIMIT_SOURCE_ALIASES)
        ),
        "rationale": _clean_string(
            _first_present(value, *_LIMIT_RATIONALE_ALIASES),
            rationale,
        ),
    }


def _limit_evidence_from_aliases(value: dict[str, Any]) -> dict[str, Any] | None:
    alias_values = {
        key: value[key] for key in _LIMIT_EVIDENCE_ALIAS_FIELDS if key in value
    }
    if not alias_values:
        return None
    return {
        "lower_limit": _first_present(alias_values, *_LIMIT_LOWER_ALIASES),
        "upper_limit": _first_present(alias_values, *_LIMIT_UPPER_ALIASES),
        "unit": _first_present(alias_values, *_WRAPPER_LIMIT_UNIT_ALIASES),
        "source": _first_present(alias_values, *_WRAPPER_LIMIT_SOURCE_ALIASES),
        "rationale": _first_present(alias_values, *_WRAPPER_LIMIT_RATIONALE_ALIASES),
    }


def _first_present(value: dict[str, Any], *keys: str) -> Any:
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


def _normalize_limit_unit(value: Any) -> str:
    unit = _clean_string(value, "unknown").lower().replace("-", "_")
    unit = unit.replace(" ", "_")
    if unit in {"degree", "degrees", "deg", "degs"}:
        return "degrees"
    if unit in {"radian", "radians", "rad", "rads"}:
        return "radians"
    if unit in {"meter", "meters", "metre", "metres", "m"}:
        return "meters"
    return "unknown"


def _normalize_limit_evidence_source(value: Any) -> str:
    source = _clean_string(value, "unknown").lower().replace("-", "_")
    source = source.replace(" ", "_")
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


def _rigger_edge_endpoint(
    value: dict[str, Any],
    primary_key: str,
    alias_key: str,
) -> str:
    raw_value = value.get(primary_key)
    if raw_value is None:
        raw_value = value.get(alias_key)
    if isinstance(raw_value, dict):
        claim = _normalize_rigger_evidence_claim(raw_value)
        if claim is not None:
            return _clean_string(claim.get("value"), "unknown")
        raw_value = None
    return _clean_string(raw_value, "unknown")


def _normalize_prim_path_list(value: Any) -> list[str]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list):
        values = value
    else:
        return []
    normalized: list[str] = []
    for item in values:
        text = _clean_string(item, "")
        if text.startswith("/"):
            normalized.append(text)
    return normalized


def _dedupe_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _normalize_rigger_evidence_source(value: Any) -> str:
    source = _clean_string(value, "predicted").lower().replace("-", "_")
    source = source.replace(" ", "_")
    if source in {"llm", "vlm", "llm_vlm", "model", "stage1", "stage1_model"}:
        return "predicted"
    if source in {
        "predicted",
        "consistency_corrected",
        "llm_adjudicated",
        "unknown",
    }:
        return source
    return "unknown"


def _normalize_role(
    value: Any,
    *,
    component_type: Any = None,
    component_name: Any = None,
) -> str:
    explicit_role = _clean_string(value, "")
    if explicit_role:
        return _canonical_role(explicit_role)

    component_type_text = _clean_string(component_type, "")
    component_type_role = _canonical_role(component_type_text)
    if component_type_role != "unknown":
        return component_type_role

    if component_type_text.lower().replace("-", "_").replace(" ", "_") not in (
        _WEAK_COMPONENT_TYPES
    ):
        return "unknown"

    component_name_role = _canonical_role(_clean_string(component_name, ""))
    if component_name_role != "unknown":
        return component_name_role
    return "unknown"


def infer_stage1_role(
    value: Any,
    *,
    component_type: Any = None,
    component_name: Any = None,
) -> str:
    """Infer a supported Stage 1 role from role/type/name hints."""
    return _normalize_role(
        value,
        component_type=component_type,
        component_name=component_name,
    )


def _canonical_role(value: str) -> str:
    cleaned = value.strip().lower().replace("-", "_").replace(" ", "_")
    if not cleaned:
        return "unknown"
    tokens = [token for token in cleaned.split("_") if token]
    non_moving_role = _non_moving_door_lid_role(tokens)
    if non_moving_role is not None:
        return non_moving_role
    if cleaned in SUPPORTED_PROP_ROLES:
        return cleaned
    if cleaned in _ROLE_ALIASES:
        return _ROLE_ALIASES[cleaned]
    door_lid_phrase_role = _door_lid_phrase_role(tokens)
    if door_lid_phrase_role is not None:
        return door_lid_phrase_role
    for token in tokens:
        if token in _ROLE_ALIASES:
            return _ROLE_ALIASES[token]
    return "unknown"


def _non_moving_door_lid_role(tokens: list[str]) -> str | None:
    if not ({"door", "lid"} & set(tokens)):
        return None
    control_role = _door_lid_supported_control_role(tokens)
    if control_role is not None:
        return control_role
    if _NON_MOVING_DOOR_LID_BODY_TOKENS & set(tokens):
        return "body"
    if _NON_MOVING_DOOR_LID_UNKNOWN_TOKENS & set(tokens):
        return "unknown"
    return None


def _door_lid_supported_control_role(tokens: list[str]) -> str | None:
    for alias in _role_alias_candidates(tokens):
        role = _ROLE_ALIASES.get(alias)
        if role in _DOOR_LID_SUPPORTED_CONTROL_ROLES:
            return role
    return None


def _role_alias_candidates(tokens: list[str]) -> list[str]:
    candidates: list[str] = []
    for width in range(len(tokens), 0, -1):
        for start in range(0, len(tokens) - width + 1):
            candidates.append("_".join(tokens[start : start + width]))
    return candidates


def _door_lid_phrase_role(tokens: list[str]) -> str | None:
    token_set = set(tokens)
    if "door" in token_set:
        return "door"
    if "lid" in token_set:
        return "lid"
    return None


def _clean_string(value: Any, default: str) -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _looks_like_material_label(value: str) -> bool:
    cleaned = value.strip().lower().replace("_", " ")
    return any(label in cleaned.split() for label in _MATERIAL_LABELS)
