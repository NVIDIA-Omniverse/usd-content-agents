# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-asset consistency helpers for Joint Agent predictions."""

from __future__ import annotations

import copy
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from joint_agent.functions.axis_hints import normalize_axis_hint_token
from joint_agent.functions.stage1_schema import (
    infer_stage1_role,
    unwrap_stage1_prediction_payload,
)

DEFAULT_CONSISTENCY_FIELDS = (
    "role",
    "component_type",
    "component_name",
    "joint_type_hint",
    "is_articulation_candidate",
    "axis_hint",
)
PROVENANCE_PREDICTED = "predicted"
PROVENANCE_CONSISTENCY_CORRECTED = "consistency_corrected"

_SIDE_TOKEN_RE = re.compile(r"(left|right|front|rear|back)", re.IGNORECASE)
_NUMBER_RE = re.compile(r"\d+")
_NON_IDENTIFIER_RE = re.compile(r"[^a-z0-9_]+")
_LETTER_DIGIT_BOUNDARY_RE = re.compile(r"(?<=[a-z])_?(?=\d)")
_DIGIT_LETTER_BOUNDARY_RE = re.compile(r"(?<=\d)_?(?=[a-z])")
_PREDICTED_SOURCE_ALIASES = frozenset(
    {
        "llm",
        "vlm",
        "llm_vlm",
        "model",
        "stage1",
        "stage1_model",
        "llm_adjudicated",
        PROVENANCE_PREDICTED,
    }
)
_RAW_MODEL_SOURCE_ALIASES = frozenset(
    {
        "llm",
        "vlm",
        "llm_vlm",
        "model",
        "stage1",
        "stage1_model",
        PROVENANCE_PREDICTED,
    }
)
_TRUSTED_RIGGER_AXIS_SOURCES = frozenset(
    {
        "authored_metadata",
        "authored_reference",
        "source_metadata",
        "accepted_manifest",
        "template_default",
    }
)
_LINK_AXIS_CONFLICT_ACTIONS = frozenset(
    {
        "flagged_link_axis_tie",
        "preserved_stronger_rigger_axis_evidence",
    }
)
_TOPOLOGY_RECONCILED_FIELD_KEYS = frozenset(
    {
        "role",
        "instance_id",
        "is_articulation_candidate",
        "joint_type_hint",
        "axis_hint",
        "parent_hint",
        "child_hint",
        "confidence",
    }
)
_MOTION_PROFILE_FIELDS = (
    "role",
    "is_articulation_candidate",
    "joint_type_hint",
)
_UNKNOWN_ROLE_TOKENS = frozenset({"", "unknown", "none", "null"})
_UNKNOWN_JOINT_TOKENS = frozenset({"", "unknown", "null"})
_MIN_STRICT_OUTLIER_SUPPORT = 3


def load_predictions_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load JSONL predictions."""
    predictions: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            stripped_line = line.strip()
            if stripped_line:
                predictions.append(json.loads(stripped_line))
    return predictions


def write_predictions_jsonl(
    path: str | Path, predictions: Iterable[dict[str, Any]]
) -> None:
    """Write JSONL predictions."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as f:
            temp_path = Path(f.name)
            for prediction in predictions:
                f.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        temp_path.replace(output_path)
        temp_path = None
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass


def apply_prediction_consistency(
    predictions: Iterable[dict[str, Any]],
    *,
    output_key: str = "classification",
    min_group_size: int = 2,
    min_majority_fraction: float = 0.6,
    harmonize_fields: Sequence[str] | None = None,
    harmonize_motion_profiles: bool = False,
    add_role: bool = True,
    add_instance_id: bool = True,
    signature_depth: int = 2,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Annotate and optionally harmonize repeated-part prediction groups.

    The pass groups predictions by normalized USD path suffixes, e.g.
    ``wheel1/rim`` and ``wheel2/rim`` share a group after numeric and side tokens
    are normalized. By default the pass is conservative: it adds metadata,
    ``role``, and ``instance_id`` only. Callers can opt into field harmonization
    for selected fields once they have reviewed the behavior on a test asset.
    Motion-profile harmonization is a separate conservative opt-in: it repairs
    only one complete outlier when at least three repeated peers agree.
    """
    if min_group_size < 2:
        raise ValueError("min_group_size must be >= 2")
    if not 0 < min_majority_fraction <= 1:
        raise ValueError("min_majority_fraction must be in the range (0, 1]")
    if signature_depth < 1:
        raise ValueError("signature_depth must be >= 1")
    if not isinstance(harmonize_motion_profiles, bool):
        raise ValueError("harmonize_motion_profiles must be a boolean")

    harmonize_set = set(harmonize_fields or [])
    output_predictions = [copy.deepcopy(prediction) for prediction in predictions]
    sealed_topology_predictions = sum(
        has_topology_reconciliation_trace(prediction, output_key=output_key)
        for prediction in output_predictions
    )
    if sealed_topology_predictions:
        receipt_predictions = sum(
            has_topology_reconciliation_history(
                prediction,
                output_key=output_key,
            )
            for prediction in output_predictions
        )
        if sealed_topology_predictions != len(output_predictions) or (
            0 < receipt_predictions < len(output_predictions)
        ):
            raise ValueError(
                "topology reconciliation receipt is present on only part of the "
                "prediction set"
            )
        if receipt_predictions != len(output_predictions):
            raise ValueError("topology reconciliation receipt is invalid or stale")
        # Imported lazily because articulation adjudication imports Stage 2,
        # which in turn imports these consistency helpers.
        from joint_agent.functions.articulation_adjudication import (
            recover_articulation_topology_reconciliation_from_history,
        )

        if (
            recover_articulation_topology_reconciliation_from_history(
                output_predictions,
                output_key=output_key,
            )
            is None
        ):
            raise ValueError("topology reconciliation receipt is invalid or stale")
        # A topology overlay is an exact, recoverable receipt over the entire
        # prediction set. Mutating even diagnostic consistency metadata makes
        # that receipt stale, so a consistency rerun must preserve it byte-for-
        # byte at the payload level and leave validation to the Stage 2 reader.
        return output_predictions, {
            "total_predictions": len(output_predictions),
            "groups_total": 0,
            "groups_repeated": 0,
            "annotated_predictions": 0,
            "harmonized_predictions": 0,
            "field_conflicts": 0,
            "harmonize_fields": sorted(harmonize_set),
            "heuristic_paths_used": [],
            "groups": [],
            "link_axis_groups": 0,
            "link_axis_conflicts": 0,
            "link_axis_unresolved_conflicts": 0,
            "link_axis_harmonized_predictions": 0,
            "motion_profile_corrected_predictions": 0,
            "motion_profile_harmonization_enabled": harmonize_motion_profiles,
            "link_axis_group_details": [],
            "sealed_topology_predictions": sealed_topology_predictions,
            "consistency_skipped_reason": "sealed_topology_reconciliation_receipt",
        }
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, prediction in enumerate(output_predictions):
        prim_path = str(prediction.get("id", ""))
        signature = normalized_path_signature(prim_path, depth=signature_depth)
        if signature:
            groups[signature].append(idx)

    group_details: list[dict[str, Any]] = []
    annotated_count = 0
    harmonized_indices: set[int] = set()
    motion_profile_corrected_indices: set[int] = set()
    conflict_count = 0

    repeated_groups = {
        signature: indices
        for signature, indices in groups.items()
        if len(indices) >= min_group_size
    }

    for group_number, (signature, indices) in enumerate(
        sorted(repeated_groups.items()), start=1
    ):
        group_id = f"g{group_number:04d}"
        motion_profile_repairs = (
            _repair_repeated_motion_profiles(
                output_predictions,
                indices,
                output_key=output_key,
                signature_depth=signature_depth,
            )
            if harmonize_motion_profiles
            else {}
        )
        repaired_indices = set(motion_profile_repairs)
        harmonized_indices.update(repaired_indices)
        motion_profile_corrected_indices.update(repaired_indices)
        field_majorities = _compute_field_majorities(
            output_predictions,
            indices,
            output_key=output_key,
            fields=DEFAULT_CONSISTENCY_FIELDS,
            min_majority_fraction=min_majority_fraction,
        )
        conflicts = _compute_conflicts(
            output_predictions,
            indices,
            output_key=output_key,
            fields=DEFAULT_CONSISTENCY_FIELDS,
        )
        conflict_count += len(conflicts)

        for idx in indices:
            prediction = output_predictions[idx]
            payload = _classification_payload(prediction, output_key)
            if payload is None:
                continue
            existing_motion_profile_correction = (
                _validated_existing_motion_profile_correction(
                    payload,
                    prediction_id=str(prediction.get("id", "")),
                )
                if harmonize_motion_profiles
                else None
            )

            field_sources = _ensure_field_sources(payload)

            if add_instance_id and not payload.get("instance_id"):
                payload["instance_id"] = instance_id_from_path(
                    str(prediction.get("id", ""))
                )
                field_sources["instance_id"] = PROVENANCE_CONSISTENCY_CORRECTED

            changed_fields: dict[str, dict[str, Any]] = {}
            for field in harmonize_set:
                # Axis writes require explicit physical-link membership. Path
                # repetition alone cannot distinguish symmetric links from
                # legitimately differently oriented links.
                if field == "axis_hint":
                    continue
                if field not in field_majorities:
                    continue
                majority_value = field_majorities[field]
                current_value = payload.get(field)
                if current_value != majority_value:
                    changed_fields[field] = {
                        "before": current_value,
                        "after": majority_value,
                        "source_before": field_sources.get(field, PROVENANCE_PREDICTED),
                        "source_after": PROVENANCE_CONSISTENCY_CORRECTED,
                    }
                    payload[field] = majority_value
                    field_sources[field] = PROVENANCE_CONSISTENCY_CORRECTED

            if add_role and not payload.get("role"):
                role = infer_stage1_role(
                    None,
                    component_type=payload.get("component_type"),
                    component_name=payload.get("component_name"),
                )
                if role:
                    payload["role"] = role
                    field_sources["role"] = PROVENANCE_CONSISTENCY_CORRECTED

            if changed_fields:
                harmonized_indices.add(idx)
            flagged_fields = _flag_conflicting_fields(
                payload,
                field_majorities=field_majorities,
                conflicts=conflicts,
                harmonized_fields=set(changed_fields),
                field_sources=field_sources,
            )

            payload["consistency"] = {
                "group_id": group_id,
                "signature": signature,
                "group_size": len(indices),
                "majority": field_majorities,
                "conflicts": conflicts,
                "grouping_evidence": {
                    "strategy": "normalized_path_signature",
                    "signature_depth": signature_depth,
                    "caveat": "path_suffix_grouping_only",
                },
            }
            if changed_fields:
                payload["consistency"]["harmonized_fields"] = changed_fields
            if flagged_fields:
                payload["consistency"]["flagged_fields"] = flagged_fields
            motion_profile_repair = motion_profile_repairs.get(
                idx,
                existing_motion_profile_correction,
            )
            if motion_profile_repair is not None:
                payload["consistency"]["motion_profile_correction"] = (
                    motion_profile_repair
                )
            annotated_count += 1

        group_details.append(
            {
                "group_id": group_id,
                "signature": signature,
                "size": len(indices),
                "majority": field_majorities,
                "conflicts": conflicts,
                "motion_profile_corrections": [
                    motion_profile_repairs[idx]
                    for idx in indices
                    if idx in motion_profile_repairs
                ],
                "ids": [output_predictions[idx].get("id") for idx in indices],
            }
        )

    # Recompute after repeated-profile recovery. A corrected row has its stale
    # model link identity explicitly downgraded to consistency provenance and
    # must not participate in explicit-link axis voting.
    explicit_link_groups = _collect_explicit_link_groups(
        output_predictions,
        output_key=output_key,
    )
    link_axis_details, link_axis_changed_indices = _reconcile_explicit_link_axes(
        output_predictions,
        explicit_link_groups=explicit_link_groups,
        output_key=output_key,
        min_majority_fraction=min_majority_fraction,
        harmonize="axis_hint" in harmonize_set,
    )
    harmonized_indices.update(link_axis_changed_indices)

    stats = {
        "total_predictions": len(output_predictions),
        "groups_total": len(groups),
        "groups_repeated": len(repeated_groups),
        "annotated_predictions": annotated_count,
        "harmonized_predictions": len(harmonized_indices),
        "field_conflicts": conflict_count,
        "harmonize_fields": sorted(harmonize_set),
        "heuristic_paths_used": (
            ["normalized_path_signature_grouping"] if repeated_groups else []
        ),
        "groups": group_details,
        "link_axis_groups": len(link_axis_details),
        "link_axis_conflicts": sum(
            detail["status"]
            in {"harmonized", "unresolved_conflict", "conflict_not_harmonized"}
            for detail in link_axis_details
        ),
        "link_axis_unresolved_conflicts": sum(
            detail["status"] == "unresolved_conflict" for detail in link_axis_details
        ),
        "link_axis_harmonized_predictions": len(link_axis_changed_indices),
        "motion_profile_corrected_predictions": len(motion_profile_corrected_indices),
        "motion_profile_harmonization_enabled": harmonize_motion_profiles,
        "link_axis_group_details": link_axis_details,
        "sealed_topology_predictions": 0,
        "consistency_skipped_reason": None,
    }
    return output_predictions, stats


def normalized_path_signature(path: str, *, depth: int = 2) -> str:
    """Return a repeated-part grouping signature from a USD prim path."""
    parts = [part for part in path.split("/") if part]
    if not parts:
        return ""
    suffix = parts[-depth:]
    return "/".join(_normalize_part_token(part) for part in suffix)


def instance_id_from_path(path: str) -> str:
    """Create a stable instance id from the final USD path token."""
    parts = [part for part in path.split("/") if part]
    token = parts[-1] if parts else "unknown"
    token = token.lower()
    token = _NON_IDENTIFIER_RE.sub("_", token)
    token = token.strip("_")
    return token or "unknown"


def _repair_repeated_motion_profiles(
    predictions: list[dict[str, Any]],
    indices: list[int],
    *,
    output_key: str,
    signature_depth: int,
) -> dict[int, dict[str, Any]]:
    """Repair one complete model profile strongly outvoted by repeated peers.

    Normalized paths are not sufficient evidence for axis harmonization, but
    corresponding repeated parts should agree on their role, candidate flag,
    and joint type. A complete profile is changed only when at least three
    resolved siblings agree and it is the sole outlier. Missing profiles remain
    fail-closed for the bounded inference completion pass. Stronger-than-model
    provenance is never overwritten, and a corrected row's model-supplied link
    identity is invalidated so Stage 2 cannot retain stale membership from the
    bad profile.
    """
    if not _has_write_compatible_path_scope(
        predictions,
        indices,
        signature_depth=signature_depth,
    ):
        return {}

    profiles: dict[tuple[str, bool, str], list[int]] = defaultdict(list)
    unresolved_indices: list[int] = []
    for idx in indices:
        payload = _classification_payload(predictions[idx], output_key)
        if payload is None:
            continue
        profile = _resolved_motion_profile(payload)
        if profile is None:
            unresolved_indices.append(idx)
        else:
            profiles[profile].append(idx)

    repairs: dict[int, dict[str, Any]] = {}
    resolved_count = sum(len(profile_indices) for profile_indices in profiles.values())
    if unresolved_indices or resolved_count != len(indices):
        return repairs

    dominant_profile, supporting_indices = max(
        profiles.items(),
        key=lambda item: (len(item[1]), item[0]),
    )
    outlier_indices = [
        idx
        for profile, profile_indices in profiles.items()
        if profile != dominant_profile
        for idx in profile_indices
    ]
    if (
        len(supporting_indices) < _MIN_STRICT_OUTLIER_SUPPORT
        or len(outlier_indices) != 1
        or len(supporting_indices) / resolved_count < 0.75
    ):
        return repairs

    outlier_index = outlier_indices[0]
    repair = _apply_motion_profile_repair(
        predictions,
        outlier_index,
        profile=dominant_profile,
        supporting_indices=supporting_indices,
        group_size=len(indices),
        output_key=output_key,
        correction_kind="corrected_strict_outlier",
    )
    if repair is not None:
        repairs[outlier_index] = repair
    return repairs


def _resolved_motion_profile(
    payload: Mapping[str, Any],
) -> tuple[str, bool, str] | None:
    role = _normalize_identifier(payload.get("role"))
    joint_type = _normalize_identifier(payload.get("joint_type_hint"))
    candidate = payload.get("is_articulation_candidate")
    if (
        role in _UNKNOWN_ROLE_TOKENS
        or joint_type in _UNKNOWN_JOINT_TOKENS
        or not isinstance(candidate, bool)
    ):
        return None
    return role, candidate, joint_type


def _has_write_compatible_path_scope(
    predictions: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    *,
    signature_depth: int,
) -> bool:
    """Reject write-capable suffix groups that cross ancestor scopes."""
    scopes: set[tuple[int, tuple[str, ...]]] = set()
    for index in indices:
        parts = tuple(
            part for part in str(predictions[index].get("id", "")).split("/") if part
        )
        if not parts:
            return False
        scopes.add((len(parts), parts[:-signature_depth]))
    return len(scopes) == 1


def _validated_existing_motion_profile_correction(
    payload: Mapping[str, Any],
    *,
    prediction_id: str,
) -> dict[str, Any] | None:
    consistency = payload.get("consistency")
    if not isinstance(consistency, Mapping):
        return None
    correction = consistency.get("motion_profile_correction")
    if not isinstance(correction, dict):
        return None
    if (
        correction.get("strategy") != "resolved_repeated_part_motion_profile"
        or correction.get("kind") != "corrected_strict_outlier"
        or correction.get("target_prediction_id") != prediction_id
    ):
        return None
    changed_fields = correction.get("changed_fields")
    if not isinstance(changed_fields, Mapping) or not changed_fields:
        return None
    for field, change in changed_fields.items():
        if not isinstance(field, str) or not isinstance(change, Mapping):
            return None
        if payload.get(field) != change.get("after"):
            return None
    invalidated_instance_id = correction.get("invalidated_instance_id")
    if not isinstance(invalidated_instance_id, Mapping) or payload.get(
        "instance_id"
    ) != invalidated_instance_id.get("after"):
        return None
    return copy.deepcopy(correction)


def _apply_motion_profile_repair(
    predictions: list[dict[str, Any]],
    index: int,
    *,
    profile: tuple[str, bool, str],
    supporting_indices: Sequence[int],
    group_size: int,
    output_key: str,
    correction_kind: str,
) -> dict[str, Any] | None:
    payload = _classification_payload(predictions[index], output_key)
    if (
        payload is None
        or not _motion_profile_is_model_supplied(payload)
        or bool(payload.get("rigger_evidence"))
    ):
        return None

    role, candidate, joint_type = profile
    provenance = payload.get("provenance")
    raw_field_sources = (
        provenance.get("field_sources") if isinstance(provenance, Mapping) else {}
    )
    if not isinstance(raw_field_sources, Mapping):
        return None
    if payload.get("instance_id") is not None and not _field_is_raw_model_supplied(
        raw_field_sources,
        "instance_id",
    ):
        return None
    if (
        role == "body"
        and candidate is False
        and joint_type in {"fixed", "none"}
        and _normalize_identifier(payload.get("axis_hint")) != "unknown"
        and not _field_is_raw_model_supplied(raw_field_sources, "axis_hint")
    ):
        return None

    field_sources = _ensure_field_sources(payload)
    changed_fields: dict[str, dict[str, Any]] = {}
    for field, after in zip(
        _MOTION_PROFILE_FIELDS,
        (role, candidate, joint_type),
        strict=True,
    ):
        before = payload.get(field)
        if before == after:
            continue
        changed_fields[field] = {
            "before": before,
            "after": after,
            "source_before": field_sources.get(field, PROVENANCE_PREDICTED),
            "source_after": PROVENANCE_CONSISTENCY_CORRECTED,
        }
        payload[field] = after
        field_sources[field] = PROVENANCE_CONSISTENCY_CORRECTED

    if (
        role == "body"
        and candidate is False
        and joint_type in {"fixed", "none"}
        and _field_is_raw_model_supplied(field_sources, "axis_hint")
        and _normalize_identifier(payload.get("axis_hint")) != "unknown"
    ):
        before = payload.get("axis_hint")
        changed_fields["axis_hint"] = {
            "before": before,
            "after": "unknown",
            "source_before": field_sources.get("axis_hint", PROVENANCE_PREDICTED),
            "source_after": PROVENANCE_CONSISTENCY_CORRECTED,
        }
        payload["axis_hint"] = "unknown"
        field_sources["axis_hint"] = PROVENANCE_CONSISTENCY_CORRECTED

    if not changed_fields:
        return None

    prediction_id = str(predictions[index].get("id", ""))
    previous_instance_id = payload.get("instance_id")
    replacement_instance_id = instance_id_from_path(prediction_id)
    payload["instance_id"] = replacement_instance_id
    field_sources["instance_id"] = PROVENANCE_CONSISTENCY_CORRECTED

    return {
        "strategy": "resolved_repeated_part_motion_profile",
        "kind": correction_kind,
        "target_prediction_id": prediction_id,
        "support_count": len(supporting_indices),
        "group_size": group_size,
        "supporting_prediction_ids": [
            predictions[idx].get("id") for idx in supporting_indices
        ],
        "changed_fields": changed_fields,
        "invalidated_instance_id": {
            "before": previous_instance_id,
            "after": replacement_instance_id,
            "source_after": PROVENANCE_CONSISTENCY_CORRECTED,
        },
    }


def _motion_profile_is_model_supplied(payload: Mapping[str, Any]) -> bool:
    provenance = payload.get("provenance")
    field_sources = (
        provenance.get("field_sources") if isinstance(provenance, Mapping) else None
    )
    if not isinstance(field_sources, Mapping):
        return True
    return all(
        _field_is_raw_model_supplied(field_sources, field)
        for field in _MOTION_PROFILE_FIELDS
    )


def _field_is_raw_model_supplied(
    field_sources: Mapping[str, Any],
    field: str,
) -> bool:
    source = _normalize_identifier(field_sources.get(field))
    return not source or source in _RAW_MODEL_SOURCE_ALIASES


def _collect_explicit_link_groups(
    predictions: list[dict[str, Any]],
    *,
    output_key: str,
) -> dict[str, list[int]]:
    """Collect model-supplied physical-link memberships before path fallback."""
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, prediction in enumerate(predictions):
        payload = _classification_payload(prediction, output_key)
        if payload is None:
            continue
        if not is_model_supplied_link_instance_id(payload):
            continue
        instance_id = canonical_link_instance_id(payload.get("instance_id"))
        if not instance_id or instance_id in {"unknown", "none", "null"}:
            continue
        groups[instance_id].append(idx)
    return dict(groups)


def _reconcile_explicit_link_axes(
    predictions: list[dict[str, Any]],
    *,
    explicit_link_groups: dict[str, list[int]],
    output_key: str,
    min_majority_fraction: float,
    harmonize: bool,
) -> tuple[list[dict[str, Any]], set[int]]:
    """Reconcile one stage-space axis for every explicit multi-mesh link.

    A local link majority wins first. A tied link may use consensus from other
    repeated links with the same normalized instance family, role, and fixed
    parent. Links that are already internally consistent are never overwritten
    by a family vote, which preserves legitimate differently oriented siblings.
    """
    link_groups = {
        key: indices
        for key, indices in explicit_link_groups.items()
        if len(indices) >= 2
    }
    analyses: dict[str, dict[str, Any]] = {}
    family_votes: dict[tuple[str, str, str], list[str]] = defaultdict(list)

    for instance_id, indices in sorted(link_groups.items()):
        axis_counter = _axis_counter(predictions, indices, output_key=output_key)
        local_axis = _strict_majority(
            axis_counter,
            min_majority_fraction=min_majority_fraction,
        )
        role_counter: Counter[str] = Counter()
        exact_parent_counter: Counter[str] = Counter()
        parent_hint_counter: Counter[str] = Counter()
        for idx in indices:
            payload = _classification_payload(predictions[idx], output_key)
            if payload is None:
                continue
            role = _normalize_identifier(payload.get("role"))
            if role and role not in {"unknown", "none", "null"}:
                role_counter[role] += 1
            exact_parent_identity = _link_exact_parent_identity(payload)
            if exact_parent_identity:
                exact_parent_counter[exact_parent_identity] += 1
            parent_hint_identity = _link_parent_hint_identity(payload)
            if parent_hint_identity:
                parent_hint_counter[parent_hint_identity] += 1
        role = (
            _strict_majority(
                role_counter,
                min_majority_fraction=min_majority_fraction,
            )
            or "mixed"
        )
        if len(exact_parent_counter) == 1:
            parent_identity = next(iter(exact_parent_counter))
        elif not exact_parent_counter and len(parent_hint_counter) == 1:
            parent_identity = next(iter(parent_hint_counter))
        else:
            parent_identity = ""
        family_key = (
            _normalize_part_token(instance_id),
            role,
            parent_identity,
        )
        analyses[instance_id] = {
            "indices": indices,
            "axis_counter": axis_counter,
            "local_axis": local_axis,
            "family_key": family_key,
            "role": role,
            "parent_identity": parent_identity,
        }
        if local_axis is not None and parent_identity and role != "mixed":
            family_votes[family_key].append(local_axis)

    family_axes: dict[tuple[str, str, str], str] = {}
    for family_key, votes in family_votes.items():
        if len(votes) < 2:
            continue
        family_axis = _strict_majority(
            Counter(votes),
            min_majority_fraction=min_majority_fraction,
        )
        if family_axis is not None:
            family_axes[family_key] = family_axis

    repeated_member_axes = _repeated_member_axis_consensus(
        predictions,
        analyses=analyses,
        output_key=output_key,
    )

    details: list[dict[str, Any]] = []
    changed_indices: set[int] = set()
    for instance_id, analysis in analyses.items():
        indices = cast(list[int], analysis["indices"])
        axis_counter = cast(Counter[str], analysis["axis_counter"])
        local_axis = cast(str | None, analysis["local_axis"])
        family_key = cast(tuple[str, str, str], analysis["family_key"])
        role = cast(str, analysis["role"])
        parent_identity = cast(str, analysis["parent_identity"])
        family_axis = family_axes.get(family_key)
        repeated_member_evidence = [
            evidence
            for idx in indices
            if (
                evidence := repeated_member_axes.get(
                    (
                        family_key[0],
                        role,
                        normalized_path_signature(
                            str(predictions[idx].get("id", "")),
                        ),
                    )
                )
            )
            is not None
        ]
        repeated_member_evidence = [
            dict(item)
            for item in {
                json.dumps(evidence, sort_keys=True): evidence
                for evidence in repeated_member_evidence
            }.values()
        ]
        repeated_member_axis_values = {
            cast(str, evidence["axis"]) for evidence in repeated_member_evidence
        }
        has_conflict = len(axis_counter) > 1

        selected_axis = local_axis
        selection_basis = "link_axis"
        if has_conflict and local_axis is not None:
            selection_basis = "link_majority"
        elif has_conflict and family_axis is not None:
            selected_axis = family_axis
            selection_basis = "repeated_link_consensus"
        elif has_conflict and len(repeated_member_axis_values) == 1:
            selected_axis = next(iter(repeated_member_axis_values))
            selection_basis = "repeated_member_consensus"
        elif has_conflict:
            selection_basis = "unresolved_tie"

        link_changed_indices: set[int] = set()
        link_blocked_indices: set[int] = set()
        for idx in indices:
            payload = _classification_payload(predictions[idx], output_key)
            if payload is None:
                continue
            current_axis = _canonical_consistency_axis(payload.get("axis_hint"))
            consistency = payload.setdefault("consistency", {})
            if not isinstance(consistency, dict):
                consistency = {}
                payload["consistency"] = consistency

            axis_blocked = False
            if selected_axis is not None and harmonize:
                axis_blocked = _stronger_rigger_axis_blocks_harmonization(
                    payload,
                    selected_axis=selected_axis,
                )
                if axis_blocked:
                    flagged_fields = consistency.setdefault("flagged_fields", {})
                    if not isinstance(flagged_fields, dict):
                        flagged_fields = {}
                        consistency["flagged_fields"] = flagged_fields
                    flagged_fields["axis_hint"] = {
                        "current": payload.get("axis_hint"),
                        "rigger_motion_axis": _rigger_motion_axis_claim(payload),
                        "majority": selected_axis,
                        "source": _ensure_field_sources(payload).get(
                            "axis_hint", PROVENANCE_PREDICTED
                        ),
                        "conflict_values": [
                            {"value": value, "count": count}
                            for value, count in axis_counter.most_common()
                        ],
                        "action": "preserved_stronger_rigger_axis_evidence",
                    }
                    link_blocked_indices.add(idx)
                else:
                    axis_correction: dict[str, Any] | None = None
                    if current_axis != selected_axis:
                        field_sources = _ensure_field_sources(payload)
                        source_before = field_sources.get(
                            "axis_hint", PROVENANCE_PREDICTED
                        )
                        before = payload.get("axis_hint")
                        payload["axis_hint"] = selected_axis
                        field_sources["axis_hint"] = PROVENANCE_CONSISTENCY_CORRECTED
                        axis_correction = {
                            "before": before,
                            "after": selected_axis,
                            "source_before": source_before,
                            "source_after": PROVENANCE_CONSISTENCY_CORRECTED,
                            "basis": selection_basis,
                        }
                        current_axis = selected_axis

                    rigger_axis_correction = _harmonize_prediction_rigger_axis(
                        payload,
                        selected_axis=selected_axis,
                        selection_basis=selection_basis,
                    )
                    if (
                        axis_correction is not None
                        or rigger_axis_correction is not None
                    ):
                        harmonized_fields = consistency.setdefault(
                            "harmonized_fields", {}
                        )
                        if not isinstance(harmonized_fields, dict):
                            harmonized_fields = {}
                            consistency["harmonized_fields"] = harmonized_fields
                        if axis_correction is not None:
                            previous = harmonized_fields.get("axis_hint")
                            if isinstance(previous, dict):
                                axis_correction["before"] = previous.get(
                                    "before", axis_correction["before"]
                                )
                                axis_correction["source_before"] = previous.get(
                                    "source_before",
                                    axis_correction["source_before"],
                                )
                            harmonized_fields["axis_hint"] = axis_correction
                        if rigger_axis_correction is not None:
                            harmonized_fields["rigger_evidence.motion_axis"] = (
                                rigger_axis_correction
                            )
                        flagged_fields = consistency.get("flagged_fields")
                        if isinstance(flagged_fields, dict):
                            flagged_fields.pop("axis_hint", None)
                            if not flagged_fields:
                                consistency.pop("flagged_fields")
                        link_changed_indices.add(idx)
                        changed_indices.add(idx)

            if not axis_blocked and (
                not has_conflict or (selected_axis is not None and harmonize)
            ):
                _clear_stale_link_axis_conflict_flag(consistency)

            consistency["link_axis"] = {
                "instance_id": instance_id,
                "role": role,
                "fixed_parent": parent_identity or None,
                "group_size": len(indices),
                "family_signature": family_key[0],
                "axis_counts": [
                    {"value": value, "count": count}
                    for value, count in axis_counter.most_common()
                ],
                "selected_axis": selected_axis,
                "selection_basis": selection_basis,
                "harmonization_enabled": harmonize,
            }
            if repeated_member_evidence:
                consistency["link_axis"]["repeated_member_evidence"] = (
                    repeated_member_evidence
                )

            if has_conflict and selected_axis is None:
                flagged_fields = consistency.setdefault("flagged_fields", {})
                if not isinstance(flagged_fields, dict):
                    flagged_fields = {}
                    consistency["flagged_fields"] = flagged_fields
                flagged_fields["axis_hint"] = {
                    "current": payload.get("axis_hint"),
                    "majority": None,
                    "source": _ensure_field_sources(payload).get(
                        "axis_hint", PROVENANCE_PREDICTED
                    ),
                    "conflict_values": consistency["link_axis"]["axis_counts"],
                    "action": "flagged_link_axis_tie",
                }

        if link_blocked_indices:
            status = "unresolved_conflict"
        elif has_conflict and selected_axis is None:
            status = "unresolved_conflict"
        elif link_changed_indices:
            status = "harmonized"
        elif has_conflict and not harmonize:
            status = "conflict_not_harmonized"
        else:
            status = "consistent"
        details.append(
            {
                "instance_id": instance_id,
                "role": role,
                "fixed_parent": parent_identity or None,
                "ids": [predictions[idx].get("id") for idx in indices],
                "axis_counts": [
                    {"value": value, "count": count}
                    for value, count in axis_counter.most_common()
                ],
                "selected_axis": selected_axis,
                "selection_basis": selection_basis,
                "status": status,
                "repeated_member_evidence": repeated_member_evidence,
            }
        )

    return details, changed_indices


def _rigger_motion_axis_claim(payload: dict[str, Any]) -> dict[str, Any] | None:
    rigger_evidence = payload.get("rigger_evidence")
    if not isinstance(rigger_evidence, dict):
        return None
    motion_axis = rigger_evidence.get("motion_axis")
    return motion_axis if isinstance(motion_axis, dict) else None


def _stronger_rigger_axis_blocks_harmonization(
    payload: dict[str, Any],
    *,
    selected_axis: str,
) -> bool:
    claim = _rigger_motion_axis_claim(payload)
    if claim is None:
        return False
    claim_axis = _canonical_consistency_axis(claim.get("value"))
    if claim_axis is None or claim_axis == selected_axis:
        return False
    source = _normalize_rigger_axis_source(claim.get("source"))
    return source in _TRUSTED_RIGGER_AXIS_SOURCES


def _harmonize_prediction_rigger_axis(
    payload: dict[str, Any],
    *,
    selected_axis: str,
    selection_basis: str,
) -> dict[str, Any] | None:
    claim = _rigger_motion_axis_claim(payload)
    if claim is None:
        return None
    claim_axis = _canonical_consistency_axis(claim.get("value"))
    if claim_axis is None or claim_axis == selected_axis:
        return None
    source_before = _normalize_rigger_axis_source(claim.get("source"))
    before = claim.get("value")
    claim["value"] = selected_axis
    claim["source"] = PROVENANCE_CONSISTENCY_CORRECTED
    rationale = str(claim.get("rationale", "")).strip()
    correction_message = (
        "Consistency pass reconciled prediction-derived motion_axis with the "
        f"physical-link axis ({selection_basis})."
    )
    claim["rationale"] = (
        f"{rationale}; {correction_message}" if rationale else correction_message
    )
    correction = {
        "before": before,
        "after": selected_axis,
        "source_before": source_before,
        "source_after": PROVENANCE_CONSISTENCY_CORRECTED,
        "basis": selection_basis,
    }
    claim["consistency_correction"] = correction
    return correction


def _normalize_rigger_axis_source(value: Any) -> str:
    """Normalize model aliases and reject unknown axis-source authority."""
    source = _normalize_identifier(value)
    if not source or source in _PREDICTED_SOURCE_ALIASES:
        return PROVENANCE_PREDICTED
    if source in {
        PROVENANCE_CONSISTENCY_CORRECTED,
        "unknown",
        *_TRUSTED_RIGGER_AXIS_SOURCES,
    }:
        return source
    return "unknown"


def _clear_stale_link_axis_conflict_flag(consistency: dict[str, Any]) -> None:
    """Remove a prior link-axis conflict once the current link is resolved."""
    flagged_fields = consistency.get("flagged_fields")
    if not isinstance(flagged_fields, dict):
        return
    axis_flag = flagged_fields.get("axis_hint")
    if not isinstance(axis_flag, dict):
        return
    if axis_flag.get("action") not in _LINK_AXIS_CONFLICT_ACTIONS:
        return
    flagged_fields.pop("axis_hint", None)
    if not flagged_fields:
        consistency.pop("flagged_fields", None)


def _repeated_member_axis_consensus(
    predictions: list[dict[str, Any]],
    *,
    analyses: dict[str, dict[str, Any]],
    output_key: str,
) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Return unanimous axes for the same member kind across repeated links.

    The evidence is intentionally stricter than a majority vote: every
    occurrence of the repeated member kind must carry the same explicit axis,
    and the evidence must span at least two distinct physical-link groups.
    Unknown axes abstain by invalidating the consensus instead of being ignored.
    """
    occurrences: dict[
        tuple[str, str, str],
        list[tuple[str, str | None]],
    ] = defaultdict(list)
    for instance_id, analysis in analyses.items():
        indices = cast(list[int], analysis["indices"])
        family_key = cast(tuple[str, str, str], analysis["family_key"])
        role = cast(str, analysis["role"])
        if role == "mixed":
            continue
        for idx in indices:
            payload = _classification_payload(predictions[idx], output_key)
            if payload is None:
                continue
            member_signature = normalized_path_signature(
                str(predictions[idx].get("id", ""))
            )
            if not member_signature:
                continue
            occurrences[(family_key[0], role, member_signature)].append(
                (
                    instance_id,
                    _canonical_consistency_axis(payload.get("axis_hint")),
                )
            )

    consensus: dict[tuple[str, str, str], dict[str, Any]] = {}
    for key, member_occurrences in occurrences.items():
        link_ids = {instance_id for instance_id, _ in member_occurrences}
        axes = [axis for _, axis in member_occurrences]
        explicit_axes = {axis for axis in axes if axis is not None}
        if len(link_ids) < 2 or None in axes or len(explicit_axes) != 1:
            continue
        consensus[key] = {
            "member_signature": key[2],
            "axis": next(iter(explicit_axes)),
            "link_count": len(link_ids),
            "occurrence_count": len(member_occurrences),
        }
    return consensus


def _axis_counter(
    predictions: list[dict[str, Any]],
    indices: list[int],
    *,
    output_key: str,
) -> Counter[str]:
    counter: Counter[str] = Counter()
    for idx in indices:
        payload = _classification_payload(predictions[idx], output_key)
        if payload is None:
            continue
        axis = _canonical_consistency_axis(payload.get("axis_hint"))
        if axis is not None:
            counter[axis] += 1
    return counter


def _strict_majority(
    counter: Counter[str],
    *,
    min_majority_fraction: float,
) -> str | None:
    if not counter:
        return None
    ranked = counter.most_common()
    top_axis, top_count = ranked[0]
    if len(ranked) > 1 and ranked[1][1] == top_count:
        return None
    if top_count / counter.total() < min_majority_fraction:
        return None
    return top_axis


def _canonical_consistency_axis(value: Any) -> str | None:
    """Canonicalize unsigned and explicitly positive axes as equivalent."""
    axis = normalize_axis_hint_token(value)
    return axis.removeprefix("+") if axis is not None else None


def _normalize_identifier(value: Any) -> str:
    if value is None:
        return ""
    token = str(value).strip().lower()
    token = _NON_IDENTIFIER_RE.sub("_", token)
    token = re.sub(r"_+", "_", token)
    return token.strip("_")


def canonical_link_instance_id(value: Any) -> str:
    """Normalize cosmetic separator and zero-padding variants of one link id."""
    token = _normalize_identifier(value)
    if not token:
        return ""
    token = _LETTER_DIGIT_BOUNDARY_RE.sub("_", token)
    token = _DIGIT_LETTER_BOUNDARY_RE.sub("_", token)
    parts = [part for part in token.split("_") if part]
    return "_".join(str(int(part)) if part.isdigit() else part for part in parts)


def is_model_supplied_link_instance_id(payload: Mapping[str, Any]) -> bool:
    """Return whether ``instance_id`` is model evidence rather than a fallback."""
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        return True
    field_sources = provenance.get("field_sources")
    if not isinstance(field_sources, Mapping):
        return True
    source = _normalize_identifier(field_sources.get("instance_id"))
    if not source:
        return True
    return source in _PREDICTED_SOURCE_ALIASES


def _link_exact_parent_identity(payload: dict[str, Any]) -> str:
    rigger_evidence = payload.get("rigger_evidence")
    if isinstance(rigger_evidence, dict):
        body0 = rigger_evidence.get("body0")
        if isinstance(body0, dict):
            body0_value = body0.get("value")
            if isinstance(body0_value, str) and body0_value.strip():
                return _normalize_identifier(body0_value)
    return ""


def _link_parent_hint_identity(payload: dict[str, Any]) -> str:
    parent_hint = payload.get("parent_hint")
    parent_identity = _normalize_identifier(parent_hint)
    if parent_identity in {"", "unknown", "none", "null"}:
        return ""
    return parent_identity


def _normalize_part_token(token: str) -> str:
    token = token.lower()
    token = _SIDE_TOKEN_RE.sub("{side}", token)
    token = _NUMBER_RE.sub("{n}", token)
    token = _NON_IDENTIFIER_RE.sub("_", token)
    token = re.sub(r"_+", "_", token)
    return token.strip("_")


def _classification_payload(
    prediction: dict[str, Any], output_key: str
) -> dict[str, Any] | None:
    value = prediction.get(output_key)
    if not isinstance(value, dict):
        return None

    payload = cast(dict[str, Any], value)
    nested = payload.get(output_key)
    if isinstance(nested, dict):
        unwrapped = unwrap_stage1_prediction_payload(payload, output_key=output_key)
        payload[output_key] = unwrapped
        return cast(dict[str, Any], unwrapped)
    return payload


def prediction_payload_layers(
    prediction: Mapping[str, Any],
    *,
    output_key: str,
) -> tuple[Mapping[str, Any], ...]:
    """Return outer and nested prediction payloads without mutating either."""
    payload = prediction.get(output_key)
    if not isinstance(payload, Mapping):
        return ()
    nested = payload.get(output_key)
    if isinstance(nested, Mapping):
        return (payload, nested)
    return (payload,)


def effective_prediction_payload(
    prediction: Mapping[str, Any],
    *,
    output_key: str,
) -> Mapping[str, Any] | None:
    """Return the payload consumed by Stage 2 for one prediction row."""
    layers = prediction_payload_layers(prediction, output_key=output_key)
    return layers[-1] if layers else None


def _payload_has_topology_reconciliation_trace(payload: Mapping[str, Any]) -> bool:
    provenance = payload.get("provenance")
    if isinstance(provenance, Mapping):
        history = provenance.get("topology_reconciliation_history")
        if isinstance(history, list) and bool(history):
            return True
        field_sources = provenance.get("field_sources")
        if isinstance(field_sources, Mapping) and any(
            field_sources.get(field) == "llm_adjudicated"
            for field in _TOPOLOGY_RECONCILED_FIELD_KEYS
        ):
            return True
    consistency = payload.get("consistency")
    reconciliation = (
        consistency.get("topology_reconciliation")
        if isinstance(consistency, Mapping)
        else None
    )
    return bool(
        isinstance(reconciliation, Mapping)
        and reconciliation.get("source") == "llm_adjudicated"
    )


def has_topology_reconciliation_trace(
    prediction: Mapping[str, Any],
    *,
    output_key: str,
) -> bool:
    """Detect a topology overlay in either accepted payload representation."""
    return any(
        _payload_has_topology_reconciliation_trace(payload)
        for payload in prediction_payload_layers(
            prediction,
            output_key=output_key,
        )
    )


def has_topology_reconciliation_history(
    prediction: Mapping[str, Any],
    *,
    output_key: str,
) -> bool:
    """Detect reconciliation history in either outer or nested payload."""
    for payload in prediction_payload_layers(prediction, output_key=output_key):
        provenance = payload.get("provenance")
        history = (
            provenance.get("topology_reconciliation_history")
            if isinstance(provenance, Mapping)
            else None
        )
        if isinstance(history, list) and bool(history):
            return True
    return False


def _ensure_field_sources(payload: dict[str, Any]) -> dict[str, str]:
    provenance = payload.setdefault("provenance", {})
    if not isinstance(provenance, dict):
        provenance = {}
        payload["provenance"] = provenance

    field_sources = provenance.setdefault("field_sources", {})
    if not isinstance(field_sources, dict):
        field_sources = {}
        provenance["field_sources"] = field_sources

    for field in (*DEFAULT_CONSISTENCY_FIELDS, "instance_id"):
        if field in payload and field not in field_sources:
            field_sources[field] = PROVENANCE_PREDICTED
    return field_sources


def _flag_conflicting_fields(
    payload: dict[str, Any],
    *,
    field_majorities: dict[str, Any],
    conflicts: dict[str, list[dict[str, Any]]],
    harmonized_fields: set[str],
    field_sources: dict[str, str],
) -> dict[str, dict[str, Any]]:
    flagged_fields: dict[str, dict[str, Any]] = {}
    for field, conflict_values in conflicts.items():
        if field in harmonized_fields:
            continue
        if field not in field_majorities:
            continue
        current_value = payload.get(field)
        majority_value = field_majorities[field]
        if current_value == majority_value:
            continue
        flagged_fields[field] = {
            "current": current_value,
            "majority": majority_value,
            "source": field_sources.get(field, PROVENANCE_PREDICTED),
            "conflict_values": conflict_values,
            "action": "flagged",
        }
    return flagged_fields


def _compute_field_majorities(
    predictions: list[dict[str, Any]],
    indices: list[int],
    *,
    output_key: str,
    fields: Sequence[str],
    min_majority_fraction: float,
) -> dict[str, Any]:
    majorities: dict[str, Any] = {}
    for field in fields:
        values: list[Any] = []
        for idx in indices:
            payload = _classification_payload(predictions[idx], output_key)
            if payload is not None and field in payload and payload[field] is not None:
                values.append(_consistency_field_value(field, payload[field]))
        if not values:
            continue

        counter: Counter[str] = Counter(_value_key(value) for value in values)
        top_key, top_count = counter.most_common(1)[0]
        if top_count / len(indices) >= min_majority_fraction:
            majorities[field] = _value_from_key(top_key)
    return majorities


def _compute_conflicts(
    predictions: list[dict[str, Any]],
    indices: list[int],
    *,
    output_key: str,
    fields: Sequence[str],
) -> dict[str, list[dict[str, Any]]]:
    conflicts: dict[str, list[dict[str, Any]]] = {}
    for field in fields:
        counter: Counter[str] = Counter()
        for idx in indices:
            payload = _classification_payload(predictions[idx], output_key)
            if payload is not None and field in payload and payload[field] is not None:
                value = _consistency_field_value(field, payload[field])
                counter[_value_key(value)] += 1
        if len(counter) <= 1:
            continue
        conflicts[field] = [
            {"value": _value_from_key(key), "count": count}
            for key, count in counter.most_common()
        ]
    return conflicts


def _value_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _consistency_field_value(field: str, value: Any) -> Any:
    if field == "axis_hint":
        return _canonical_consistency_axis(value) or value
    return value


def _value_from_key(value_key: str) -> Any:
    return json.loads(value_key)
