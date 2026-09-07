# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Closed Stage 1 projection for the opt-in Joint 0.6 v1 adapter."""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from joint_agent.functions.joint_0_6_capabilities import (
    SOURCE_BACKED_STAGE2_SOURCES,
)
from joint_agent.functions.stage1_schema import (
    STAGE1_SCHEMA_VERSION,
    _limit_evidence_from_aliases,
    _normalize_compound_edge_evidence_list,
    _normalize_limit_evidence_source,
    _normalize_limit_unit,
    _normalize_motion_limit_evidence,
    _normalize_rigger_evidence_claim,
    unwrap_stage1_prediction_payload,
)

_PLACEHOLDERS = frozenset({"", "unknown", "none", "null", "n/a", "na"})
_TRANSPORT_FIELDS = frozenset({"legacy_label", "original_response"})
_ENDPOINT_ALIASES = {
    "body0": "body0",
    "body1": "body1",
    "fixed_parent_prim": "body0",
    "moving_body_prim": "body1",
    "motion_axis": "motion_axis",
}
_LIMIT_ALIAS_FIELDS = frozenset(
    {
        "limits",
        "lower_limit",
        "lower",
        "lowerLimit",
        "upper_limit",
        "upper",
        "upperLimit",
        "unit",
        "limit_unit",
        "limitUnit",
        "source",
        "limit_source",
        "limitSource",
        "rationale",
        "limit_rationale",
        "limitRationale",
    }
)
_TOP_LEVEL_ALIAS_FIELDS = frozenset(
    {
        *_ENDPOINT_ALIASES,
        "compound_edges",
        *_LIMIT_ALIAS_FIELDS,
    }
)
_PROVENANCE_FIELDS = frozenset({"field_sources"})
_PROVENANCE_FIELD_SOURCE_FIELDS = frozenset(
    {
        "asset_type",
        "component_type",
        "component_name",
        "role",
        "semantic_role",
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
        "source_joint_type",
    }
)
_BREADTH_FIELD_SOURCES = frozenset(
    {
        "predicted",
        "consistency_corrected",
        "authored_metadata",
        "authored_reference",
        "source_metadata",
        "accepted_manifest",
        "stage1_hint",
        "stage1_rigger_evidence",
        "llm_adjudicated",
        "template_default",
        "unknown",
    }
)
_RIGGER_FIELDS = frozenset(
    {"body0", "body1", "motion_axis", "compound_edges", "limits"}
)
_CLAIM_FIELDS = frozenset(
    {"value", "prim_path", "prim_paths", "confidence", "rationale", "source"}
)
_EDGE_FIELDS = frozenset(
    {
        "body0",
        "body1",
        "fixed_parent_prim",
        "moving_body_prim",
        "joint_type_hint",
        "motion_type",
        "axis_hint",
        "confidence",
        "rationale",
        "reasoning",
        "source",
        "prim_paths",
    }
)
_LIMIT_FIELDS = frozenset(_LIMIT_ALIAS_FIELDS - {"limits"})

type _BreadthFieldSource = Literal[
    "predicted",
    "consistency_corrected",
    "authored_metadata",
    "authored_reference",
    "source_metadata",
    "accepted_manifest",
    "stage1_hint",
    "stage1_rigger_evidence",
    "llm_adjudicated",
    "template_default",
    "unknown",
]
type _BreadthProvenanceField = Literal[
    "asset_type",
    "component_type",
    "component_name",
    "role",
    "semantic_role",
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
    "source_joint_type",
]


class _StrictProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    field_sources: dict[_BreadthProvenanceField, _BreadthFieldSource] = Field(
        default_factory=dict
    )


class _StrictSemanticMotionCapability(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    kind: Literal["passive_rotation", "unsupported", "unresolved"]
    source: _BreadthFieldSource = "predicted"
    evidence: str = ""
    missing_evidence: list[Literal["motion_kind", "passivity", "motion_contract"]] = (
        Field(default_factory=list)
    )
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


class _StrictClaim(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    value: str = "unknown"
    prim_paths: list[str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"] = "low"
    rationale: str = ""
    source: _BreadthFieldSource = "predicted"


class _StrictCompoundEdge(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    body0: str = "unknown"
    body1: str = "unknown"
    joint_type_hint: Literal[
        "revolute",
        "prismatic",
        "spherical",
        "fixed",
        "none",
        "unknown",
    ] = "unknown"
    axis_hint: Literal[
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
    ] = "unknown"
    confidence: Literal["high", "medium", "low"] = "low"
    rationale: str = ""
    source: _BreadthFieldSource = "predicted"
    prim_paths: list[str] = Field(default_factory=list)


class _StrictLimits(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    lower_limit: float | None = None
    upper_limit: float | None = None
    unit: Literal["degrees", "radians", "meters", "unknown"] = "unknown"
    source: _BreadthFieldSource = "unknown"
    rationale: str = ""


class _StrictRiggerEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    body0: _StrictClaim | None = None
    body1: _StrictClaim | None = None
    motion_axis: _StrictClaim | None = None
    compound_edges: list[_StrictCompoundEdge] = Field(default_factory=list)
    limits: _StrictLimits | None = None


class _StrictStage1Prediction(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal["joint-agent-stage1-v0"] = STAGE1_SCHEMA_VERSION
    asset_type: str = "unknown"
    component_type: str = "unknown"
    component_name: str = "unknown"
    role: str = "unknown"
    semantic_role: str | None = None
    motion_capability: _StrictSemanticMotionCapability | None = None
    instance_id: str | None = None
    is_articulation_candidate: bool = False
    joint_type_hint: Literal[
        "revolute",
        "prismatic",
        "spherical",
        "fixed",
        "none",
        "unknown",
    ] = "unknown"
    axis_hint: Literal[
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
    ] = "unknown"
    parent_hint: str = "unknown"
    child_hint: str = "unknown"
    material: str = "unknown"
    confidence: Literal["high", "medium", "low"] = "low"
    evidence: str = ""
    reasoning: str = ""
    provenance: _StrictProvenance | None = None
    rigger_evidence: _StrictRiggerEvidence | None = None


@dataclass(frozen=True)
class Stage1AxisVectorProjection:
    """One exact trusted Stage 1 vector retained outside the public v0 model."""

    axis: tuple[float, float, float]
    source: _BreadthFieldSource
    prim_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class Stage1BreadthProjection:
    """Trusted Stage 1 facts and every field excluded from the closed view."""

    payload: dict[str, Any]
    source_joint_type: Literal["continuous"] | None = None
    source_joint_type_source: _BreadthFieldSource | None = None
    motion_axis: Stage1AxisVectorProjection | None = None
    rejected_paths: tuple[str, ...] = ()


def project_stage1_for_v1_breadth(
    payload: Mapping[str, Any],
    *,
    output_key: str = "classification",
) -> Stage1BreadthProjection:
    """Normalize and validate one Stage 1 payload without forwarding raw extras."""

    raw_payload = copy.deepcopy(dict(payload))
    raw_nested = raw_payload.get(output_key)
    nested = isinstance(raw_nested, dict)
    if nested:
        raw_payload.pop(output_key)
        raw_body = cast(dict[str, Any], raw_nested)
        raw_wrapper = raw_payload
    else:
        raw_body = raw_payload
        raw_wrapper = raw_payload
    rejected: set[str] = set()
    motion_axis = _extract_private_motion_axis(
        raw_wrapper,
        raw_body=raw_body,
        nested=nested,
        rejected=rejected,
    )
    if _is_placeholder_container(raw_body):
        rejected.add("$")
    _record_raw_stage1_type_mismatches(
        raw_body,
        prefix="",
        rejected=rejected,
    )
    if nested:
        _record_raw_stage1_type_mismatches(
            raw_wrapper,
            prefix="wrapper.",
            rejected=rejected,
        )
    projection_payload = (
        {**raw_wrapper, output_key: raw_body} if nested else raw_payload
    )
    projected = unwrap_stage1_prediction_payload(
        projection_payload,
        output_key=output_key,
    )
    if nested:
        allowed_wrapper_fields = frozenset(
            {
                "physical_properties",
                "source_joint_type",
                *_StrictStage1Prediction.model_fields,
                *_TOP_LEVEL_ALIAS_FIELDS,
                *_TRANSPORT_FIELDS,
            }
        )
        _record_unknown_fields(
            raw_wrapper,
            allowed_wrapper_fields,
            prefix="wrapper",
            rejected=rejected,
        )
        _check_wrapper_field_consistency(
            wrapper=raw_wrapper,
            nested=raw_body,
            rejected=rejected,
        )
        wrapper_properties = raw_wrapper.get("physical_properties")
        if "physical_properties" in raw_wrapper and wrapper_properties != {}:
            rejected.add("wrapper.physical_properties")

    for field in _TRANSPORT_FIELDS:
        if field not in projected:
            continue
        if not isinstance(projected[field], str):
            rejected.add(field)
        projected.pop(field)

    if "physical_properties" in projected:
        if projected["physical_properties"] != {}:
            rejected.add("physical_properties")
        projected.pop("physical_properties")

    projected.pop("source_joint_type", None)
    source_joint_type: Literal["continuous"] | None = None
    source_joint_type_fields: list[tuple[str, Any]] = []
    if "source_joint_type" in raw_body:
        source_joint_type_fields.append(
            ("source_joint_type", raw_body["source_joint_type"])
        )
    if nested and "source_joint_type" in raw_wrapper:
        source_joint_type_fields.append(
            ("wrapper.source_joint_type", raw_wrapper["source_joint_type"])
        )
    for path, raw_source_joint_type in source_joint_type_fields:
        if raw_source_joint_type == "continuous":
            source_joint_type = "continuous"
        else:
            rejected.add(path)

    _check_top_level_alias_consistency(
        aliases=raw_body,
        normalized=projected,
        rejected=rejected,
    )
    if nested:
        _check_top_level_alias_consistency(
            aliases=raw_wrapper,
            normalized=projected,
            rejected=rejected,
        )
    for field in _TOP_LEVEL_ALIAS_FIELDS:
        projected.pop(field, None)

    provenance = projected.get("provenance")
    if isinstance(provenance, Mapping):
        provenance = _canonical_provenance(
            provenance,
            rejected=rejected,
        )
        projected["provenance"] = provenance
    rigger_evidence = projected.get("rigger_evidence")
    if isinstance(rigger_evidence, Mapping):
        raw_rigger_evidence = raw_body.get("rigger_evidence")
        canonical_raw_evidence: dict[str, Any] = {}
        if isinstance(raw_rigger_evidence, Mapping):
            canonical_raw_evidence = _normalize_canonical_rigger_evidence(
                _canonical_rigger_evidence(
                    raw_rigger_evidence,
                    path="rigger_evidence",
                    rejected=rejected,
                )
            )
        canonical_rigger_evidence = _normalize_canonical_rigger_evidence(
            _canonical_rigger_evidence(
                rigger_evidence,
                path="rigger_evidence",
                rejected=rejected,
            )
        )
        canonical_rigger_evidence.update(canonical_raw_evidence)
        if nested:
            wrapper_rigger_evidence = raw_wrapper.get("rigger_evidence")
            if isinstance(wrapper_rigger_evidence, Mapping):
                canonical_wrapper_evidence = _normalize_canonical_rigger_evidence(
                    _canonical_rigger_evidence(
                        wrapper_rigger_evidence,
                        path="wrapper.rigger_evidence",
                        rejected=rejected,
                    )
                )
                for field, value in canonical_wrapper_evidence.items():
                    if (
                        field in canonical_raw_evidence
                        and canonical_raw_evidence[field] != value
                    ):
                        rejected.add(f"wrapper.rigger_evidence.{field}")
                    elif field not in canonical_raw_evidence:
                        canonical_rigger_evidence[field] = value
        projected["rigger_evidence"] = canonical_rigger_evidence

    strict_payload = _strict_projection(projected, rejected=rejected)

    source: _BreadthFieldSource | None = None
    if source_joint_type is not None:
        field_sources = (
            provenance.get("field_sources") if isinstance(provenance, dict) else None
        )
        raw_source = (
            field_sources.get("source_joint_type")
            if isinstance(field_sources, dict)
            else None
        )
        if raw_source == "source_metadata":
            source = "source_metadata"
        else:
            rejected.add("provenance.field_sources.source_joint_type")

    return Stage1BreadthProjection(
        payload=strict_payload,
        source_joint_type=source_joint_type,
        source_joint_type_source=source,
        motion_axis=motion_axis,
        rejected_paths=tuple(sorted(rejected)),
    )


def _extract_private_motion_axis(
    raw_payload: dict[str, Any],
    *,
    raw_body: dict[str, Any],
    nested: bool,
    rejected: set[str],
) -> Stage1AxisVectorProjection | None:
    """Remove typed vector claims from the public projection and prove them.

    The released Stage 1 v0 model accepts only string axis hints. Joint 0.6
    therefore retains a vector in this private immutable side channel instead
    of widening or smuggling it through that public model.
    """

    claims: list[tuple[str, Stage1AxisVectorProjection]] = []
    containers: list[tuple[str, dict[str, Any]]] = [("", raw_body)]
    if nested:
        containers.append(("wrapper.", raw_payload))

    for prefix, container in containers:
        raw_evidence = container.get("rigger_evidence")
        if isinstance(raw_evidence, dict) and "motion_axis" in raw_evidence:
            path = f"{prefix}rigger_evidence.motion_axis"
            claim = raw_evidence["motion_axis"]
            if _is_vector_axis_claim(claim):
                raw_evidence.pop("motion_axis")
                projected = _closed_axis_vector_claim(
                    claim,
                    path=path,
                    rejected=rejected,
                )
                if projected is not None:
                    claims.append((path, projected))

        if "motion_axis" in container:
            path = f"{prefix}motion_axis"
            claim = container["motion_axis"]
            if _is_vector_axis_claim(claim):
                container.pop("motion_axis")
                projected = _closed_axis_vector_claim(
                    claim,
                    path=path,
                    rejected=rejected,
                )
                if projected is not None:
                    claims.append((path, projected))

    if not claims:
        return None

    axes = {claim.axis for _, claim in claims}
    sources = {claim.source for _, claim in claims}
    nonempty_paths = {claim.prim_paths for _, claim in claims if claim.prim_paths}
    if len(axes) != 1 or len(sources) != 1 or len(nonempty_paths) > 1:
        rejected.update(f"{path}.conflict" for path, _ in claims)
        return None

    first = claims[0][1]
    prim_paths = next(iter(nonempty_paths), ())
    return Stage1AxisVectorProjection(
        axis=first.axis,
        source=first.source,
        prim_paths=prim_paths,
    )


def _is_vector_axis_claim(value: Any) -> bool:
    raw_value = value.get("value") if isinstance(value, Mapping) else value
    if isinstance(raw_value, list | tuple):
        return True
    if not isinstance(raw_value, str):
        return False
    if raw_value.strip().startswith("["):
        return True
    try:
        decoded = json.loads(raw_value)
    except (json.JSONDecodeError, TypeError):
        return False
    return isinstance(decoded, list)


def _closed_axis_vector_claim(
    value: Any,
    *,
    path: str,
    rejected: set[str],
) -> Stage1AxisVectorProjection | None:
    if not isinstance(value, Mapping):
        rejected.add(path)
        return None
    _record_unknown_fields(value, _CLAIM_FIELDS, prefix=path, rejected=rejected)

    raw_axis = value.get("value")
    if isinstance(raw_axis, str):
        try:
            raw_axis = json.loads(raw_axis)
        except (json.JSONDecodeError, TypeError):
            rejected.add(f"{path}.value")
            return None
    if (
        not isinstance(raw_axis, list | tuple)
        or len(raw_axis) != 3
        or any(
            isinstance(component, bool) or not isinstance(component, int | float)
            for component in raw_axis
        )
    ):
        rejected.add(f"{path}.value")
        return None
    axis = (float(raw_axis[0]), float(raw_axis[1]), float(raw_axis[2]))
    if not all(math.isfinite(component) for component in axis):
        rejected.add(f"{path}.value")
        return None
    magnitude = math.sqrt(sum(component * component for component in axis))
    if not math.isclose(magnitude, 1.0, rel_tol=0.0, abs_tol=1e-6):
        rejected.add(f"{path}.value")
        return None

    raw_source = value.get("source")
    if (
        not isinstance(raw_source, str)
        or raw_source not in SOURCE_BACKED_STAGE2_SOURCES
    ):
        rejected.add(f"{path}.source")
        return None
    source = cast(_BreadthFieldSource, raw_source)

    if "confidence" in value and value["confidence"] not in {"high", "medium", "low"}:
        rejected.add(f"{path}.confidence")
    if "rationale" in value and not isinstance(value["rationale"], str):
        rejected.add(f"{path}.rationale")

    singular_paths, singular_invalid = _validated_absolute_paths(value.get("prim_path"))
    plural_paths, plural_invalid = _validated_absolute_paths(value.get("prim_paths"))
    if (
        ("prim_path" in value and singular_invalid)
        or ("prim_paths" in value and plural_invalid)
        or (singular_paths and plural_paths and singular_paths != plural_paths)
    ):
        rejected.add(f"{path}.prim_paths")
        return None
    prim_paths = tuple(_dedupe([*singular_paths, *plural_paths]))

    if any(
        rejected_path == path or rejected_path.startswith(f"{path}.")
        for rejected_path in rejected
    ):
        return None
    return Stage1AxisVectorProjection(
        axis=axis,
        source=source,
        prim_paths=prim_paths,
    )


def _canonical_provenance(
    provenance: Mapping[str, Any],
    *,
    rejected: set[str],
) -> dict[str, Any]:
    _record_unknown_fields(
        provenance,
        _PROVENANCE_FIELDS,
        prefix="provenance",
        rejected=rejected,
    )
    raw_field_sources = provenance.get("field_sources")
    if raw_field_sources is None:
        return {}
    if not isinstance(raw_field_sources, Mapping):
        rejected.add("provenance.field_sources")
        return {}

    result: dict[str, str] = {}
    for field, raw_source in raw_field_sources.items():
        path = f"provenance.field_sources.{field}"
        if field not in _PROVENANCE_FIELD_SOURCE_FIELDS:
            rejected.add(path)
            continue
        if not isinstance(raw_source, str) or raw_source not in _BREADTH_FIELD_SOURCES:
            rejected.add(path)
            continue
        result[str(field)] = raw_source
    return {"field_sources": result}


def _check_wrapper_field_consistency(
    *,
    wrapper: Mapping[str, Any],
    nested: Mapping[str, Any],
    rejected: set[str],
) -> None:
    separately_merged = {
        "physical_properties",
        "provenance",
        "rigger_evidence",
        "source_joint_type",
    }
    for field in _StrictStage1Prediction.model_fields:
        if field in separately_merged or field not in wrapper:
            continue
        wrapper_value = wrapper[field]
        nested_value = nested.get(field)
        if (
            not _value_is_unusable(wrapper_value)
            and not _value_is_unusable(nested_value)
            and wrapper_value != nested_value
        ):
            rejected.add(f"wrapper.{field}")

    wrapper_provenance = wrapper.get("provenance")
    nested_provenance = nested.get("provenance")
    if (
        not _value_is_unusable(wrapper_provenance)
        and not _value_is_unusable(nested_provenance)
        and wrapper_provenance != nested_provenance
    ):
        rejected.add("wrapper.provenance")


def _check_top_level_alias_consistency(
    *,
    aliases: Mapping[str, Any],
    normalized: Mapping[str, Any],
    rejected: set[str],
) -> None:
    raw_evidence = normalized.get("rigger_evidence")
    evidence = raw_evidence if isinstance(raw_evidence, Mapping) else {}

    for alias, canonical_field in _ENDPOINT_ALIASES.items():
        if alias not in aliases:
            continue
        raw_alias = aliases.get(alias)
        if _is_absent_or_scalar_placeholder(raw_alias):
            continue
        if _is_placeholder_container(raw_alias):
            rejected.add(alias)
            continue
        _canonical_claim(
            raw_alias,
            path=alias,
            endpoint=canonical_field in {"body0", "body1"},
            rejected=rejected,
        )
        alias_claim = _normalize_rigger_evidence_claim(raw_alias)
        canonical_claim = _normalize_rigger_evidence_claim(
            evidence.get(canonical_field)
        )
        if alias_claim is None or canonical_claim is None:
            rejected.add(alias)
        elif alias_claim != canonical_claim:
            rejected.add(alias)

    if "compound_edges" in aliases:
        raw_edges = aliases.get("compound_edges")
        if _is_placeholder_container(raw_edges):
            rejected.add("compound_edges")
        elif not _is_absent_or_scalar_placeholder(raw_edges):
            if isinstance(raw_edges, list):
                for index, raw_edge in enumerate(raw_edges):
                    _canonical_compound_edge(
                        raw_edge,
                        path=f"compound_edges.{index}",
                        rejected=rejected,
                    )
            alias_edges = _normalize_compound_edge_evidence_list(raw_edges)
            canonical_edges = _normalize_compound_edge_evidence_list(
                evidence.get("compound_edges")
            )
            if not alias_edges or alias_edges != canonical_edges:
                rejected.add("compound_edges")

    raw_limits = aliases.get("limits")
    if "limits" in aliases and _is_placeholder_container(raw_limits):
        rejected.add("limits")
    elif not _is_absent_or_scalar_placeholder(raw_limits):
        _canonical_limits(
            raw_limits,
            path="limits",
            rejected=rejected,
        )
    flat_limit_aliases = {
        field: aliases[field]
        for field in _LIMIT_ALIAS_FIELDS
        if field != "limits" and field in aliases
    }
    if flat_limit_aliases:
        rejected.update(
            f"limits.{field}"
            for field, value in flat_limit_aliases.items()
            if _is_placeholder_container(value)
        )
        if any(
            not _is_absent_or_scalar_placeholder(value)
            for value in flat_limit_aliases.values()
        ):
            _canonical_limits(
                flat_limit_aliases,
                path="limits",
                rejected=rejected,
            )

    alias_limits = _normalize_motion_limit_evidence(raw_limits)
    if alias_limits is None:
        alias_limits = _normalize_motion_limit_evidence(
            _limit_evidence_from_aliases(dict(aliases))
        )
    if alias_limits is not None:
        canonical_limits = _normalize_motion_limit_evidence(evidence.get("limits"))
        if canonical_limits != alias_limits:
            rejected.add("limits")


def _canonical_rigger_evidence(
    evidence: Mapping[str, Any],
    *,
    path: str,
    rejected: set[str],
) -> dict[str, Any]:
    _record_unknown_fields(
        evidence,
        _RIGGER_FIELDS,
        prefix=path,
        rejected=rejected,
    )
    result: dict[str, Any] = {}
    for field in ("body0", "body1", "motion_axis"):
        if field not in evidence:
            continue
        claim = _canonical_claim(
            evidence[field],
            path=f"{path}.{field}",
            endpoint=field in {"body0", "body1"},
            rejected=rejected,
        )
        if claim is not None:
            result[field] = claim

    raw_edges = evidence.get("compound_edges")
    if isinstance(raw_edges, list):
        if not raw_edges:
            rejected.add(f"{path}.compound_edges")
        else:
            result["compound_edges"] = [
                canonical
                for index, raw_edge in enumerate(raw_edges)
                if (
                    canonical := _canonical_compound_edge(
                        raw_edge,
                        path=f"{path}.compound_edges.{index}",
                        rejected=rejected,
                    )
                )
                is not None
            ]
    elif raw_edges is not None:
        rejected.add(f"{path}.compound_edges")

    if "limits" in evidence:
        limits = _canonical_limits(
            evidence["limits"],
            path=f"{path}.limits",
            rejected=rejected,
        )
        if limits is not None:
            result["limits"] = limits
    return result


def _canonical_claim(
    value: Any,
    *,
    path: str,
    endpoint: bool,
    rejected: set[str],
) -> dict[str, Any] | None:
    if _is_absent_or_scalar_placeholder(value):
        return None
    if _is_placeholder_container(value):
        rejected.add(path)
        return None
    if isinstance(value, str):
        return {"value": value}
    if not isinstance(value, Mapping):
        rejected.add(path)
        return None
    _record_unknown_fields(value, _CLAIM_FIELDS, prefix=path, rejected=rejected)

    raw_prim_path = value.get("prim_path")
    raw_prim_paths = value.get("prim_paths")
    singular_paths, singular_invalid = _validated_absolute_paths(raw_prim_path)
    plural_paths, plural_invalid = _validated_absolute_paths(raw_prim_paths)
    if ("prim_path" in value and singular_invalid) or (
        "prim_paths" in value and plural_invalid
    ):
        rejected.add(f"{path}.prim_paths")
    if singular_paths and plural_paths and singular_paths != plural_paths:
        rejected.add(f"{path}.prim_paths")
    paths = _dedupe([*singular_paths, *plural_paths])

    result = {
        field: copy.deepcopy(value[field])
        for field in ("value", "confidence", "rationale", "source")
        if field in value
    }
    if paths:
        result["prim_paths"] = paths
    raw_value = value.get("value")
    if endpoint and isinstance(raw_value, str) and raw_value.startswith("/"):
        if paths and paths != [raw_value]:
            rejected.add(f"{path}.prim_paths")
    if not _record_fragment_validation_errors(
        _StrictClaim,
        result,
        path=path,
        rejected=rejected,
    ):
        return None
    return result


def _canonical_compound_edge(
    value: Any,
    *,
    path: str,
    rejected: set[str],
) -> dict[str, Any] | None:
    if _is_placeholder_container(value):
        rejected.add(path)
        return None
    if not isinstance(value, Mapping):
        rejected.add(path)
        return None
    _record_unknown_fields(value, _EDGE_FIELDS, prefix=path, rejected=rejected)

    body0 = _one_edge_endpoint(
        value,
        ("body0", "fixed_parent_prim"),
        path=f"{path}.body0",
        rejected=rejected,
    )
    body1 = _one_edge_endpoint(
        value,
        ("body1", "moving_body_prim"),
        path=f"{path}.body1",
        rejected=rejected,
    )
    joint_type = _one_normalized_scalar(
        value,
        ("joint_type_hint", "motion_type"),
        path=f"{path}.joint_type_hint",
        normalize=_normalized_token,
        rejected=rejected,
    )
    rationale = _one_normalized_scalar(
        value,
        ("rationale", "reasoning"),
        path=f"{path}.rationale",
        normalize=_normalized_text,
        rejected=rejected,
    )
    result = {
        field: copy.deepcopy(value[field])
        for field in ("axis_hint", "confidence", "source")
        if field in value
    }
    if "prim_paths" in value:
        prim_paths, invalid_paths = _validated_absolute_paths(value["prim_paths"])
        if invalid_paths:
            rejected.add(f"{path}.prim_paths")
        if prim_paths:
            result["prim_paths"] = prim_paths
    if body0 is not None:
        result["body0"] = body0
    if body1 is not None:
        result["body1"] = body1
    if joint_type is not None:
        result["joint_type_hint"] = joint_type
    if rationale is not None:
        result["rationale"] = rationale
    if (
        "prim_paths" in result
        and isinstance(body0, str)
        and body0.startswith("/")
        and isinstance(body1, str)
        and body1.startswith("/")
        and result["prim_paths"] != [body0, body1]
    ):
        rejected.add(f"{path}.prim_paths")
    if not _record_fragment_validation_errors(
        _StrictCompoundEdge,
        result,
        path=path,
        rejected=rejected,
    ):
        return None
    return result


def _canonical_limits(
    value: Any,
    *,
    path: str,
    rejected: set[str],
) -> dict[str, Any] | None:
    if _is_placeholder_container(value):
        rejected.add(path)
        return None
    if not isinstance(value, Mapping):
        rejected.add(path)
        return None
    _record_unknown_fields(value, _LIMIT_FIELDS, prefix=path, rejected=rejected)
    lower = _one_normalized_scalar(
        value,
        ("lower_limit", "lower", "lowerLimit"),
        path=f"{path}.lower_limit",
        normalize=_finite_float,
        rejected=rejected,
    )
    upper = _one_normalized_scalar(
        value,
        ("upper_limit", "upper", "upperLimit"),
        path=f"{path}.upper_limit",
        normalize=_finite_float,
        rejected=rejected,
    )
    unit = _one_normalized_scalar(
        value,
        ("unit", "limit_unit", "limitUnit"),
        path=f"{path}.unit",
        normalize=_normalized_limit_unit,
        rejected=rejected,
    )
    source = _one_normalized_scalar(
        value,
        ("source", "limit_source", "limitSource"),
        path=f"{path}.source",
        normalize=_normalized_limit_source,
        rejected=rejected,
    )
    rationale = _one_normalized_scalar(
        value,
        ("rationale", "limit_rationale", "limitRationale"),
        path=f"{path}.rationale",
        normalize=_normalized_text,
        rejected=rejected,
    )
    result: dict[str, Any] = {}
    for field, normalized in (
        ("lower_limit", lower),
        ("upper_limit", upper),
        ("unit", unit),
        ("source", source),
        ("rationale", rationale),
    ):
        if normalized is not None:
            result[field] = normalized
    if lower is None and upper is None and not _value_is_unusable(value):
        rejected.add(path)
        return None
    if not _record_fragment_validation_errors(
        _StrictLimits,
        result,
        path=path,
        rejected=rejected,
    ):
        return None
    return result


def _one_edge_endpoint(
    value: Mapping[str, Any],
    fields: tuple[str, str],
    *,
    path: str,
    rejected: set[str],
) -> str | None:
    return cast(
        str | None,
        _one_normalized_scalar(
            value,
            fields,
            path=path,
            normalize=_edge_endpoint,
            rejected=rejected,
            allow_mapping=True,
        ),
    )


def _one_normalized_scalar(
    value: Mapping[str, Any],
    fields: tuple[str, ...],
    *,
    path: str,
    normalize: Any,
    rejected: set[str],
    allow_mapping: bool = False,
) -> Any:
    normalized: list[Any] = []
    for field in fields:
        if field not in value or _is_absent_or_scalar_placeholder(value[field]):
            continue
        if isinstance(value[field], Mapping | list | tuple) and not (
            allow_mapping and isinstance(value[field], Mapping)
        ):
            rejected.add(path)
            continue
        try:
            item = normalize(value[field])
        except (TypeError, ValueError):
            rejected.add(path)
            continue
        if item not in normalized:
            normalized.append(item)
    if len(normalized) > 1:
        rejected.add(path)
    return normalized[0] if normalized else None


def _edge_endpoint(value: Any) -> str:
    if isinstance(value, Mapping):
        value = value.get("value")
    if not isinstance(value, str) or not value.strip():
        raise ValueError
    return value.strip()


def _finite_float(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError
    parsed = float(value)
    if parsed != parsed or parsed in {float("inf"), float("-inf")}:
        raise ValueError
    return parsed


def _normalized_text(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError
    return value.strip()


def _normalized_token(value: Any) -> str:
    return _normalized_text(value).lower()


def _normalized_limit_unit(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError
    return _normalize_limit_unit(value)


def _normalized_limit_source(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError
    return _normalize_limit_evidence_source(value)


def _validated_absolute_paths(value: Any) -> tuple[list[str], bool]:
    if _is_absent_or_scalar_placeholder(value):
        return [], False
    values = value if isinstance(value, list) else [value]
    if not values:
        return [], True
    paths: list[str] = []
    invalid = False
    for item in values:
        if not isinstance(item, str):
            invalid = True
            continue
        path = item.strip()
        if not path.startswith("/"):
            invalid = True
            continue
        if path in paths:
            invalid = True
            continue
        paths.append(path)
    return paths, invalid


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _record_raw_stage1_type_mismatches(
    value: Mapping[str, Any],
    *,
    prefix: str,
    rejected: set[str],
) -> None:
    """Reject raw producer type coercions before compatibility normalization."""

    string_fields = {
        "schema_version",
        "asset_type",
        "component_type",
        "component_name",
        "role",
        "joint_type_hint",
        "axis_hint",
        "parent_hint",
        "child_hint",
        "material",
        "confidence",
        "evidence",
        "reasoning",
    }
    for field in string_fields:
        if field in value and not isinstance(value[field], str):
            rejected.add(f"{prefix}{field}")
    for field in _TRANSPORT_FIELDS:
        if field in value and not isinstance(value[field], str):
            rejected.add(f"{prefix}{field}")
    if (
        "instance_id" in value
        and value["instance_id"] is not None
        and not isinstance(value["instance_id"], str)
    ):
        rejected.add(f"{prefix}instance_id")
    if (
        "is_articulation_candidate" in value
        and type(value["is_articulation_candidate"]) is not bool
    ):
        rejected.add(f"{prefix}is_articulation_candidate")
    for field in ("provenance", "rigger_evidence"):
        if (
            field in value
            and value[field] is not None
            and not isinstance(value[field], Mapping)
        ):
            rejected.add(f"{prefix}{field}")
        elif field in value and _is_placeholder_container(value[field]):
            rejected.add(f"{prefix}{field}")


def _record_fragment_validation_errors(
    model: type[BaseModel],
    value: Mapping[str, Any],
    *,
    path: str,
    rejected: set[str],
) -> bool:
    try:
        model.model_validate(value, strict=True)
    except ValidationError as exc:
        rejected.update(f"{path}.{_error_path(error['loc'])}" for error in exc.errors())
        return False
    return True


def _normalize_canonical_rigger_evidence(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Fill only declared defaults after evidence passes the closed view."""

    result: dict[str, Any] = {}
    for field in ("body0", "body1", "motion_axis"):
        claim = value.get(field)
        if isinstance(claim, Mapping):
            result[field] = _StrictClaim.model_validate(
                claim,
                strict=True,
            ).model_dump(mode="json")
    raw_edges = value.get("compound_edges")
    if isinstance(raw_edges, list) and raw_edges:
        result["compound_edges"] = [
            _StrictCompoundEdge.model_validate(
                edge,
                strict=True,
            ).model_dump(mode="json")
            for edge in raw_edges
        ]
    raw_limits = value.get("limits")
    if isinstance(raw_limits, Mapping):
        result["limits"] = _StrictLimits.model_validate(
            raw_limits,
            strict=True,
        ).model_dump(mode="json")
    return result


def _record_unknown_fields(
    value: Mapping[str, Any],
    allowed_fields: frozenset[str],
    *,
    prefix: str,
    rejected: set[str],
) -> None:
    rejected.update(
        f"{prefix}.{field}" for field in value if field not in allowed_fields
    )


def _strict_projection(
    value: Mapping[str, Any],
    *,
    rejected: set[str],
) -> dict[str, Any]:
    """Return only fields that survive the closed model, even on partial failure."""

    candidate = copy.deepcopy(dict(value))
    while candidate:
        try:
            strict = _StrictStage1Prediction.model_validate(candidate)
        except ValidationError as exc:
            errors = exc.errors()
            rejected.update(_error_path(error["loc"]) for error in errors)
            removed = False
            for error in errors:
                removed = _drop_error_location(candidate, error["loc"]) or removed
            if not removed:
                return {}
        else:
            return strict.model_dump(
                mode="json",
                exclude_none=True,
                exclude_unset=True,
            )
    return {}


def _drop_error_location(
    value: dict[str, Any],
    location: tuple[int | str, ...],
) -> bool:
    if not location:
        return False
    current: Any = value
    for part in location[:-1]:
        if isinstance(part, str) and isinstance(current, dict):
            current = current.get(part)
        elif isinstance(part, int) and isinstance(current, list):
            if part < 0 or part >= len(current):
                return False
            current = current[part]
        else:
            return False
    leaf = location[-1]
    if isinstance(leaf, str) and isinstance(current, dict) and leaf in current:
        current.pop(leaf)
        return True
    if isinstance(leaf, int) and isinstance(current, list):
        if 0 <= leaf < len(current):
            current.pop(leaf)
            return True
    return False


def _value_is_unusable(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in _PLACEHOLDERS
    if isinstance(value, Mapping):
        return not value or all(_value_is_unusable(item) for item in value.values())
    if isinstance(value, list | tuple):
        return not value or all(_value_is_unusable(item) for item in value)
    return False


def _is_absent_or_scalar_placeholder(value: Any) -> bool:
    return value is None or (
        isinstance(value, str) and value.strip().lower() in _PLACEHOLDERS
    )


def _is_placeholder_container(value: Any) -> bool:
    return isinstance(value, Mapping | list | tuple) and _value_is_unusable(value)


def _error_path(location: tuple[int | str, ...]) -> str:
    return ".".join(str(part) for part in location)
