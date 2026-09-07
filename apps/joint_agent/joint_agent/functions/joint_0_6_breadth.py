# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Neutral source-proof contract for opt-in Joint 0.6 breadth.

This module deliberately depends on neither the Stage 2 candidate models nor
the articulation-contract adapter.  Both the Stage 2 producer and the
first-class-contract consumer validate the same plain mapping and receive the
same immutable proof.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

from joint_agent.functions.joint_0_6_capabilities import (
    CONTINUOUS_DERIVATION,
    INTERNAL_V1_BREADTH_CAPABILITY_IDS,
    INTERNAL_V1_BREADTH_STAGE2_TYPES,
    PUBLIC_OWNED_CORE_STAGE2_CAPABILITY_IDS,
    PUBLIC_OWNED_CORE_STAGE2_TYPES,
    SOURCE_BACKED_DERIVATION,
    SOURCE_BACKED_STAGE2_SOURCES,
)

_AXIS_TOLERANCE = 1e-6
_UNRESOLVED_VALUES = frozenset({"", "unknown", "none", "null", "n/a", "na"})
_TRUSTED_PARENT_RESOLUTIONS = frozenset({"stage1_hint", "stage1_rigger_evidence"})
# Released 0.5 treats accepted template limits as independently source-backed.
# Keep that authority scoped to limits; it must never authorize 0.6 topology.
_SOURCE_BACKED_LIMIT_SOURCES = SOURCE_BACKED_STAGE2_SOURCES | frozenset(
    {"template_default"}
)
_CARDINAL_AXES: Mapping[str, tuple[float, float, float]] = {
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

type BreadthProofFailureCode = Literal[
    "stage2_source_candidate_not_ready",
    "stage2_source_annotation_conflict",
    "stage2_joint_type_unsupported",
    "stage2_joint_type_conflict",
    "stage2_parent_resolution_untrusted",
    "stage2_topology_incomplete",
    "stage2_same_body_endpoints",
    "stage2_source_field_untrusted",
    "stage2_connectivity_source_untrusted",
    "stage2_connectivity_evidence_missing",
    "stage2_connectivity_evidence_conflict",
    "stage2_source_joint_type_unsupported",
    "stage2_continuous_normalization_conflict",
    "stage2_continuous_limit_conflict",
    "stage2_axis_unresolved",
    "stage2_axis_shape_invalid",
    "stage2_axis_non_finite",
    "stage2_axis_not_normalized",
    "stage2_axis_evidence_untrusted",
    "stage2_axis_hint_vector_conflict",
    "stage2_axis_hint_invalid",
    "stage2_axis_evidence_endpoint_conflict",
    "stage2_axis_evidence_value_invalid",
    "stage2_axis_evidence_value_conflict",
    "stage2_spherical_axis_not_applicable",
    "stage2_spherical_scalar_limit_unsupported",
    "stage2_limit_not_source_backed",
    "stage2_limit_source_untrusted",
    "stage2_limit_evidence_conflict",
    "stage2_limit_evidence_endpoint_conflict",
    "stage2_limit_unit_mismatch",
    "stage2_limit_evidence_value_conflict",
    "stage2_limit_invalid",
]


@dataclass(frozen=True, slots=True)
class BreadthProofFailure:
    """One stable, consumer-independent source-proof failure."""

    code: BreadthProofFailureCode
    detail: str


@dataclass(frozen=True, slots=True)
class SourceBackedLimitProof:
    """Exact typed limit and the source authority that represented it."""

    lower: float | None
    upper: float | None
    unit: Literal["degrees", "meters"]
    source: str


@dataclass(frozen=True, slots=True)
class SourceBackedBreadthProof:
    """Immutable proof accepted by both Stage 2 readiness and promotion."""

    candidate_id: str
    motion_type: Literal["prismatic", "revolute", "spherical"]
    source_joint_type: Literal["continuous"] | None
    body0: str
    body1: str
    moving_part_prims: tuple[str, ...]
    parent_resolution_source: Literal[
        "stage1_hint",
        "stage1_rigger_evidence",
    ]
    motion_source: str
    parent_source: str
    connectivity_source: str
    axis_hint: str | None
    axis: tuple[float, float, float] | None
    axis_source: str | None
    limit: SourceBackedLimitProof | None

    @property
    def derivation(self) -> str:
        return (
            CONTINUOUS_DERIVATION
            if self.source_joint_type == "continuous"
            else SOURCE_BACKED_DERIVATION
        )


@dataclass(frozen=True, slots=True)
class BreadthProofResult:
    """Closed validation outcome for one candidate mapping."""

    requested: bool
    proof: SourceBackedBreadthProof | None
    failures: tuple[BreadthProofFailure, ...]

    @property
    def accepted(self) -> bool:
        return self.proof is not None and not self.failures


@dataclass(frozen=True, slots=True)
class _RepresentedLimit:
    lower: float | None
    upper: float | None
    unit: str
    source: str | None


def requires_direct_breadth_construction(candidate: Mapping[str, Any]) -> bool:
    """Return whether a candidate needs private direct construction.

    Source-backed cardinal revolute/prismatic candidates remain on the public
    0.5 construction path.  The opt-in still validates their source proof, but
    only an actual 0.6 breadth discriminator selects direct construction.
    """

    if candidate.get("source_joint_type") is not None:
        return True
    if candidate.get("motion_type") == "spherical":
        return True

    raw_axis = candidate.get("motion_axis_world")
    axis = _numeric_axis(raw_axis)
    return axis is not None and axis not in _CARDINAL_AXES.values()


def requires_source_backed_proof(candidate: Mapping[str, Any]) -> bool:
    """Return whether opt-in validation must prove topology source authority.

    This is intentionally broader than direct-adapter selection, but limit
    provenance alone is not a topology selector.  Released 0.5 cardinal rows
    may combine predicted topology with independently source-backed limits,
    including accepted template defaults, and must remain on its preflight.
    A candidate carrying trusted *topology* opinions still cannot bypass exact
    evidence checks merely because the legacy builder can represent it.
    """

    if requires_direct_breadth_construction(candidate):
        return True
    field_sources = candidate.get("field_sources")
    return isinstance(field_sources, Mapping) and any(
        source in SOURCE_BACKED_STAGE2_SOURCES for source in field_sources.values()
    )


def validate_source_backed_breadth_proof(
    candidate: Mapping[str, Any],
) -> BreadthProofResult:
    """Validate one exact source-backed topology and return its typed proof.

    Validation is intentionally fail-closed and deterministic.  The result is
    safe to retain after the mutable candidate mapping has gone out of scope.
    """

    requested = requires_source_backed_proof(candidate)
    if not requested:
        return BreadthProofResult(requested=False, proof=None, failures=())

    candidate_id = candidate.get("candidate_id")
    displayed_id = candidate_id if isinstance(candidate_id, str) else "<unknown>"

    def rejected(
        code: BreadthProofFailureCode,
        detail: str,
    ) -> BreadthProofResult:
        return BreadthProofResult(
            requested=True,
            proof=None,
            failures=(
                BreadthProofFailure(
                    code=code,
                    detail=f"candidate {displayed_id!r} {detail}",
                ),
            ),
        )

    if candidate.get("review_status") != "ready_for_rigger_input":
        return rejected(
            "stage2_source_candidate_not_ready",
            "is not ready_for_rigger_input",
        )
    if candidate.get("unresolved_reason_codes") or candidate.get(
        "unresolved_questions"
    ):
        return rejected(
            "stage2_source_candidate_not_ready",
            "carries unresolved review state",
        )
    if candidate.get("source_annotation_conflicts"):
        return rejected(
            "stage2_source_annotation_conflict",
            "carries unresolved source annotation conflicts",
        )

    raw_motion_type = candidate.get("motion_type")
    if raw_motion_type not in INTERNAL_V1_BREADTH_STAGE2_TYPES:
        return rejected(
            "stage2_joint_type_unsupported",
            f"uses unsupported motion type {raw_motion_type!r}",
        )
    motion_type = cast(
        Literal["prismatic", "revolute", "spherical"],
        raw_motion_type,
    )
    if candidate.get("joint_type_hint") != motion_type:
        return rejected(
            "stage2_joint_type_conflict",
            "motion_type and joint_type_hint must match",
        )

    parent_resolution_source = candidate.get("parent_resolution_source")
    if parent_resolution_source not in _TRUSTED_PARENT_RESOLUTIONS:
        return rejected(
            "stage2_parent_resolution_untrusted",
            f"cannot use parent resolution {parent_resolution_source!r}",
        )

    body0 = candidate.get("fixed_parent_prim")
    moving = candidate.get("moving_part_prims")
    if (
        not _is_absolute_prim_path(body0)
        or not isinstance(moving, Sequence)
        or isinstance(moving, str | bytes)
        or not moving
        or any(not _is_absolute_prim_path(path) for path in moving)
        or len(set(moving)) != len(moving)
    ):
        return rejected(
            "stage2_topology_incomplete",
            "does not carry unique absolute body endpoints",
        )
    if requires_direct_breadth_construction(candidate) and len(moving) != 1:
        return rejected(
            "stage2_topology_incomplete",
            "private breadth must carry exactly one moving body endpoint",
        )
    moving_paths = tuple(str(path) for path in moving)
    body0 = str(body0)
    body1 = moving_paths[0]
    if body0 in moving_paths:
        return rejected(
            "stage2_same_body_endpoints",
            "body0 must differ from every moving body member",
        )

    field_sources = candidate.get("field_sources")
    if not isinstance(field_sources, Mapping):
        return rejected(
            "stage2_source_field_untrusted",
            "has no typed field_sources mapping",
        )
    motion_source = field_sources.get("motion_type")
    if motion_source not in SOURCE_BACKED_STAGE2_SOURCES:
        return rejected(
            "stage2_source_field_untrusted",
            "field 'motion_type' lacks accepted source provenance",
        )
    parent_source = field_sources.get("fixed_parent_prim")
    if parent_source not in SOURCE_BACKED_STAGE2_SOURCES:
        return rejected(
            "stage2_connectivity_source_untrusted",
            "fixed_parent_prim lacks accepted source provenance",
        )

    connectivity = candidate.get("connectivity_evidence")
    if not isinstance(connectivity, list) or not connectivity:
        return rejected(
            "stage2_connectivity_evidence_missing",
            "requires connectivity_evidence",
        )
    connectivity_shapes = [
        _connectivity_shape(item, body0=body0, body1=body1) for item in connectivity
    ]
    if any(shape is None for shape in connectivity_shapes):
        return rejected(
            "stage2_connectivity_evidence_conflict",
            "connectivity evidence does not bind its exact directed endpoints",
        )
    if not any(
        shape == "edge"
        and isinstance(item, Mapping)
        and item.get("source") == parent_source
        for item, shape in zip(connectivity, connectivity_shapes, strict=True)
    ):
        return rejected(
            "stage2_connectivity_evidence_conflict",
            "has no same-source proof of its exact body0/body1 edge",
        )
    if motion_type == "spherical" and not any(
        shape == "ownership"
        and isinstance(item, Mapping)
        and item.get("source") == parent_source
        and item.get("value") == body1
        and item.get("prim_paths") == [body1]
        for item, shape in zip(connectivity, connectivity_shapes, strict=True)
    ):
        return rejected(
            "stage2_connectivity_evidence_conflict",
            "passive spherical topology lacks exact same-source body1 ownership",
        )

    raw_source_joint_type = candidate.get("source_joint_type")
    source_joint_type: Literal["continuous"] | None
    if raw_source_joint_type is None:
        source_joint_type = None
        if "source_joint_type" in field_sources:
            return rejected(
                "stage2_source_field_untrusted",
                "source_joint_type provenance has no typed extension value",
            )
    elif raw_source_joint_type == "continuous":
        source_joint_type = "continuous"
        if field_sources.get("source_joint_type") != "source_metadata":
            return rejected(
                "stage2_source_field_untrusted",
                "continuous source_joint_type requires source_metadata provenance",
            )
        if motion_type != "revolute":
            return rejected(
                "stage2_continuous_normalization_conflict",
                "must normalize continuous to revolute",
            )
        if _has_limit_opinion(candidate):
            return rejected(
                "stage2_continuous_limit_conflict",
                "normalizes continuous to unbounded revolute and cannot carry limits",
            )
    else:
        return rejected(
            "stage2_source_joint_type_unsupported",
            f"carries unsupported source_joint_type {raw_source_joint_type!r}",
        )

    axis_hint: str | None
    axis: tuple[float, float, float] | None
    axis_source: str | None
    limit: SourceBackedLimitProof | None
    if motion_type == "spherical":
        if _has_axis_opinion(candidate, field_sources=field_sources):
            return rejected(
                "stage2_spherical_axis_not_applicable",
                "passive spherical topology must not carry axis evidence",
            )
        if _has_limit_opinion(candidate):
            return rejected(
                "stage2_spherical_scalar_limit_unsupported",
                "passive spherical topology must not carry scalar limits",
            )
        axis_hint = None
        axis = None
        axis_source = None
        limit = None
    else:
        raw_axis = candidate.get("motion_axis_world")
        if raw_axis is None:
            return rejected(
                "stage2_axis_unresolved",
                "has no stage-frame axis",
            )
        axis = _numeric_axis(raw_axis)
        if axis is None:
            return rejected(
                "stage2_axis_shape_invalid",
                "axis must contain exactly three non-boolean numbers",
            )
        if not all(math.isfinite(component) for component in axis):
            return rejected(
                "stage2_axis_non_finite",
                "has a non-finite axis",
            )
        magnitude = math.sqrt(sum(component * component for component in axis))
        if not math.isclose(
            magnitude,
            1.0,
            rel_tol=0.0,
            abs_tol=_AXIS_TOLERANCE,
        ):
            return rejected(
                "stage2_axis_not_normalized",
                f"axis magnitude is {magnitude:g}",
            )
        axis_source_value = field_sources.get("motion_axis_world")
        if axis_source_value not in SOURCE_BACKED_STAGE2_SOURCES:
            return rejected(
                "stage2_source_field_untrusted",
                "field 'motion_axis_world' lacks accepted source provenance",
            )
        axis_source = str(axis_source_value)
        normalized_hint = _normalized_token(candidate.get("axis_hint"))
        cardinal = _CARDINAL_AXES.get(normalized_hint)
        if cardinal is not None:
            if field_sources.get("axis_hint") != axis_source:
                return rejected(
                    "stage2_axis_evidence_untrusted",
                    "axis_hint and motion_axis_world provenance differ",
                )
            if cardinal != axis:
                return rejected(
                    "stage2_axis_hint_vector_conflict",
                    "axis vector conflicts with its cardinal axis_hint",
                )
            axis_hint = normalized_hint
        elif normalized_hint not in _UNRESOLVED_VALUES:
            return rejected(
                "stage2_axis_hint_invalid",
                "carries an invalid non-cardinal axis_hint",
            )
        elif field_sources.get("axis_hint") not in {None, "unknown"}:
            return rejected(
                "stage2_axis_hint_vector_conflict",
                "non-cardinal axis carries axis_hint provenance",
            )
        else:
            axis_hint = None

        axis_evidence = candidate.get("axis_evidence")
        if not isinstance(axis_evidence, list) or not axis_evidence:
            return rejected(
                "stage2_axis_evidence_untrusted",
                "requires source-backed axis_evidence",
            )
        for item in axis_evidence:
            if not isinstance(item, Mapping) or item.get("source") != axis_source:
                return rejected(
                    "stage2_axis_evidence_untrusted",
                    "axis evidence source differs from motion_axis_world provenance",
                )
            if item.get("prim_paths") != [body0, body1]:
                return rejected(
                    "stage2_axis_evidence_endpoint_conflict",
                    "axis evidence must bind exact directed body0/body1 paths",
                )
            represented_axis = _parse_axis_evidence(item.get("value"))
            if represented_axis is None:
                return rejected(
                    "stage2_axis_evidence_value_invalid",
                    "axis evidence has no exact cardinal token or JSON vector",
                )
            if represented_axis != axis:
                return rejected(
                    "stage2_axis_evidence_value_conflict",
                    "axis evidence does not equal motion_axis_world",
                )

        limit_result = _validate_limit(
            candidate,
            motion_type=motion_type,
            body1=body1,
            rejected=rejected,
        )
        if isinstance(limit_result, BreadthProofResult):
            return limit_result
        limit = limit_result

    return BreadthProofResult(
        requested=True,
        proof=SourceBackedBreadthProof(
            candidate_id=displayed_id,
            motion_type=motion_type,
            source_joint_type=source_joint_type,
            body0=body0,
            body1=body1,
            moving_part_prims=moving_paths,
            parent_resolution_source=parent_resolution_source,
            motion_source=str(motion_source),
            parent_source=str(parent_source),
            connectivity_source=str(parent_source),
            axis_hint=axis_hint,
            axis=axis,
            axis_source=axis_source,
            limit=limit,
        ),
        failures=(),
    )


def _validate_limit(
    candidate: Mapping[str, Any],
    *,
    motion_type: str,
    body1: str,
    rejected: Callable[
        [BreadthProofFailureCode, str],
        BreadthProofResult,
    ],
) -> SourceBackedLimitProof | BreadthProofResult | None:
    if not _has_limit_opinion(candidate):
        return None
    if candidate.get("limit_readiness") != "source_backed":
        return rejected(
            "stage2_limit_not_source_backed",
            "carries a non-source-backed limit opinion",
        )
    source = candidate.get("limit_source")
    if source not in _SOURCE_BACKED_LIMIT_SOURCES:
        return rejected(
            "stage2_limit_source_untrusted",
            "has an untrusted limit source",
        )
    evidence = candidate.get("limit_evidence")
    if (
        not isinstance(evidence, list)
        or not evidence
        or any(
            not isinstance(item, Mapping) or item.get("source") != source
            for item in evidence
        )
    ):
        return rejected(
            "stage2_limit_evidence_conflict",
            "limit evidence does not match limit_source",
        )
    if any(item.get("prim_paths") != [body1] for item in evidence):
        return rejected(
            "stage2_limit_evidence_endpoint_conflict",
            "limit evidence must bind the exact moving body1 endpoint",
        )

    expected_unit: Literal["degrees", "meters"] = (
        "degrees" if motion_type == "revolute" else "meters"
    )
    if candidate.get("limit_unit") != expected_unit:
        return rejected(
            "stage2_limit_unit_mismatch",
            f"{motion_type} limits must use {expected_unit}",
        )
    lower = _optional_number(candidate.get("lower_limit"))
    upper = _optional_number(candidate.get("upper_limit"))
    if (candidate.get("lower_limit") is not None and lower is None) or (
        candidate.get("upper_limit") is not None and upper is None
    ):
        return rejected(
            "stage2_limit_invalid",
            "limit bounds must be finite non-boolean numbers",
        )
    if lower is None and upper is None:
        return rejected(
            "stage2_limit_invalid",
            "source-backed limit must carry at least one bound",
        )
    if lower is not None and upper is not None and lower > upper:
        return rejected(
            "stage2_limit_invalid",
            "lower_limit exceeds upper_limit",
        )
    for item in evidence:
        represented = _parse_limit_evidence(item.get("value"))
        if (
            represented is None
            or represented.lower != lower
            or represented.upper != upper
            or represented.unit != expected_unit
            or (represented.source is not None and represented.source != source)
        ):
            return rejected(
                "stage2_limit_evidence_value_conflict",
                "limit evidence does not represent exact bounds, unit, and source",
            )
    return SourceBackedLimitProof(
        lower=lower,
        upper=upper,
        unit=expected_unit,
        source=str(source),
    )


def _connectivity_shape(
    item: Any,
    *,
    body0: str,
    body1: str,
) -> Literal["edge", "ownership", "canonicalization"] | None:
    if not isinstance(item, Mapping):
        return None
    if item.get("source") not in SOURCE_BACKED_STAGE2_SOURCES:
        return None
    role = item.get("connectivity_role")
    paths = item.get("prim_paths")
    value = item.get("value")
    if role == "body0_body1_edge":
        return "edge" if value == body0 and paths == [body0, body1] else None
    if role == "body1_ownership":
        return "ownership" if value == body1 and paths == [body1] else None
    if role == "endpoint_canonicalization":
        if (
            value not in {body0, body1}
            or not isinstance(paths, list)
            or len(paths) != 2
            or paths[-1] != value
            or not all(_is_absolute_prim_path(path) for path in paths)
            or not _strictly_related(paths[0], paths[1])
        ):
            return None
        return "canonicalization"
    if role is not None:
        return None

    # Legacy v0 evidence remains readable, but only in its exact directed form.
    if paths == [body0, body1] and value in {body0, f"{body0}->{body1}"}:
        return "edge"
    if paths == [body1] and value == body1:
        return "ownership"
    return None


def _has_axis_opinion(
    candidate: Mapping[str, Any],
    *,
    field_sources: Mapping[str, Any],
) -> bool:
    return bool(
        _normalized_token(candidate.get("axis_hint")) not in _UNRESOLVED_VALUES
        or candidate.get("motion_axis_world") is not None
        or candidate.get("axis_evidence")
        or any(
            field_sources.get(field) not in {None, "unknown"}
            for field in ("axis_hint", "motion_axis_world")
        )
    )


def _has_limit_opinion(candidate: Mapping[str, Any]) -> bool:
    return bool(
        candidate.get("lower_limit") is not None
        or candidate.get("upper_limit") is not None
        or candidate.get("limit_unit", "unknown") != "unknown"
        or candidate.get("limit_source", "unknown") != "unknown"
        or candidate.get("limit_readiness", "not_provided") != "not_provided"
        or candidate.get("limit_evidence")
    )


def _parse_axis_evidence(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, str):
        return None
    cardinal = _CARDINAL_AXES.get(value.strip().lower())
    if cardinal is not None:
        return cardinal
    try:
        decoded = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None
    axis = _numeric_axis(decoded)
    if axis is None or not all(math.isfinite(component) for component in axis):
        return None
    return axis


def _parse_limit_evidence(value: Any) -> _RepresentedLimit | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if "=" in stripped:
        fields: dict[str, str] = {}
        for part in stripped.split(","):
            key, separator, raw_value = part.strip().partition("=")
            if not separator or key in fields:
                return None
            fields[key] = raw_value.strip()
        if set(fields) != {
            "lower_limit",
            "upper_limit",
            "unit",
            "source",
        }:
            return None
        lower_valid, lower = _parse_optional_text_number(fields["lower_limit"])
        upper_valid, upper = _parse_optional_text_number(fields["upper_limit"])
        if not lower_valid or not upper_valid:
            return None
        return _RepresentedLimit(
            lower=lower,
            upper=upper,
            unit=fields["unit"],
            source=fields["source"],
        )

    parts = [part.strip() for part in stripped.split(":")]
    if len(parts) != 3:
        return None
    lower_valid, lower = _parse_optional_text_number(parts[0])
    upper_valid, upper = _parse_optional_text_number(parts[1])
    if not lower_valid or not upper_valid:
        return None
    return _RepresentedLimit(
        lower=lower,
        upper=upper,
        unit=parts[2],
        source=None,
    )


def _parse_optional_text_number(value: str) -> tuple[bool, float | None]:
    if value.strip().lower() in _UNRESOLVED_VALUES:
        return True, None
    try:
        parsed = float(value)
    except ValueError:
        return False, None
    if not math.isfinite(parsed):
        return False, None
    return True, parsed


def _optional_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _numeric_axis(value: Any) -> tuple[float, float, float] | None:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, str | bytes)
        or len(value) != 3
        or any(
            isinstance(component, bool) or not isinstance(component, int | float)
            for component in value
        )
    ):
        return None
    return (float(value[0]), float(value[1]), float(value[2]))


def _normalized_token(value: Any) -> str:
    return value.strip().lower() if isinstance(value, str) else ""


def _is_absolute_prim_path(value: Any) -> bool:
    if not isinstance(value, str) or value == "/" or not value.startswith("/"):
        return False
    return all(part not in {"", ".", ".."} for part in value.split("/")[1:])


def _strictly_related(left: str, right: str) -> bool:
    return left != right and (
        right.startswith(f"{left}/") or left.startswith(f"{right}/")
    )


__all__ = [
    "BreadthProofFailure",
    "BreadthProofFailureCode",
    "BreadthProofResult",
    "CONTINUOUS_DERIVATION",
    "INTERNAL_V1_BREADTH_CAPABILITY_IDS",
    "INTERNAL_V1_BREADTH_STAGE2_TYPES",
    "PUBLIC_OWNED_CORE_STAGE2_CAPABILITY_IDS",
    "PUBLIC_OWNED_CORE_STAGE2_TYPES",
    "SOURCE_BACKED_DERIVATION",
    "SOURCE_BACKED_STAGE2_SOURCES",
    "SourceBackedBreadthProof",
    "SourceBackedLimitProof",
    "requires_direct_breadth_construction",
    "requires_source_backed_proof",
    "validate_source_backed_breadth_proof",
]
