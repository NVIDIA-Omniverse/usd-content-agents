# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Workflow-owned validation for agent material decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .manifest import ResolvedMaterialManifest

MATERIAL_DECISION_PATCH_SCHEMA_VERSION = "content-agents.material-decision-patch.v1"
VISIBLE_CANDIDATE_PRIMS_SCHEMA_VERSION = "content-agents.visible-candidate-prims.v1"


@dataclass(frozen=True, slots=True)
class DecisionContractError:
    code: str
    message: str
    path: str | None = None


@dataclass(frozen=True, slots=True)
class MaterialDecisionValidation:
    valid: bool
    canonical_candidate_paths: tuple[str, ...]
    claimed_candidate_paths: tuple[str, ...]
    errors: tuple[DecisionContractError, ...]


def validate_material_decision(
    patch: dict[str, Any],
    *,
    candidates: dict[str, Any],
    manifest: ResolvedMaterialManifest,
    clear_materials: bool,
) -> MaterialDecisionValidation:
    """Validate exact candidate coverage and authoritative material identity."""

    errors: list[DecisionContractError] = []
    if patch.get("schema_version") != MATERIAL_DECISION_PATCH_SCHEMA_VERSION:
        errors.append(
            DecisionContractError(
                "invalid_schema_version",
                "Material decision patch has an unsupported schema version.",
            )
        )
    if candidates.get("schema_version") != VISIBLE_CANDIDATE_PRIMS_SCHEMA_VERSION:
        errors.append(
            DecisionContractError(
                "invalid_candidate_schema_version",
                "Visible candidate artifact has an unsupported schema version.",
            )
        )
    raw_path_space = candidates.get("path_space")
    if raw_path_space == "inspection":
        path_space: Literal["source", "inspection"] = "inspection"
    elif raw_path_space == "source":
        path_space = "source"
    else:
        errors.append(
            DecisionContractError(
                "invalid_candidate_path_space",
                "Visible candidate artifact path_space must be source or inspection.",
            )
        )
        path_space = "source"
    canonical = _canonical_candidate_paths(candidates, path_space, errors)
    material_groups = _group_list(
        patch, "material_assignments", required=True, errors=errors
    )
    reviewed_groups = _group_list(
        patch, "reviewed_no_override", required=True, errors=errors
    )
    if clear_materials and reviewed_groups:
        errors.append(
            DecisionContractError(
                "clean_slate_reviewed_no_override",
                "Clean-slate material decisions cannot preserve existing appearance.",
            )
        )

    claimed: dict[str, str] = {}
    for index, group in enumerate(material_groups):
        location = f"material_assignments[{index}]"
        name = _required_group_string(group, "material_name", location, errors)
        material = manifest.by_name.get(name) if name else None
        if name and material is None:
            errors.append(
                DecisionContractError(
                    "unknown_material",
                    f"Material {name!r} is not present in the resolved manifest.",
                    location,
                )
            )
        requested_path = _required_group_string(
            group, "material_path", location, errors
        )
        if (
            material is not None
            and requested_path
            and requested_path != material.binding_path
        ):
            errors.append(
                DecisionContractError(
                    "material_path_mismatch",
                    f"Material {name!r} must bind {material.binding_path!r}.",
                    location,
                )
            )
        _claim_group_paths(
            group,
            location=location,
            path_space=path_space,
            canonical=canonical,
            claimed=claimed,
            errors=errors,
        )

    for index, group in enumerate(reviewed_groups):
        _claim_group_paths(
            group,
            location=f"reviewed_no_override[{index}]",
            path_space=path_space,
            canonical=canonical,
            claimed=claimed,
            errors=errors,
        )

    if canonical:
        missing = sorted(canonical.difference(claimed))
        for path in missing:
            errors.append(
                DecisionContractError(
                    "missing_candidate_decision",
                    "Canonical material candidate has no decision.",
                    path,
                )
            )
    else:
        explicit_count = patch.get("candidate_count")
        if explicit_count != 0:
            errors.append(
                DecisionContractError(
                    "implicit_zero_candidate_result",
                    "A zero-candidate decision must explicitly set candidate_count to 0.",
                )
            )
        if material_groups or reviewed_groups:
            errors.append(
                DecisionContractError(
                    "unexpected_zero_candidate_groups",
                    "A zero-candidate decision cannot contain assignment groups.",
                )
            )

    return MaterialDecisionValidation(
        valid=not errors,
        canonical_candidate_paths=tuple(sorted(canonical)),
        claimed_candidate_paths=tuple(sorted(claimed)),
        errors=tuple(errors),
    )


def _canonical_candidate_paths(
    document: dict[str, Any],
    path_space: Literal["source", "inspection"],
    errors: list[DecisionContractError],
) -> set[str]:
    raw = document.get("candidates")
    if not isinstance(raw, list):
        errors.append(
            DecisionContractError(
                "invalid_candidates",
                "Visible candidate artifact must contain a candidates list.",
            )
        )
        return set()
    result: set[str] = set()
    for index, candidate in enumerate(raw):
        if not isinstance(candidate, dict):
            errors.append(
                DecisionContractError(
                    "invalid_candidate",
                    "Candidate must be an object.",
                    f"candidates[{index}]",
                )
            )
            continue
        key = "runtime_path" if path_space == "inspection" else "source_path"
        value = candidate.get(key)
        if not isinstance(value, str) or not value.startswith("/"):
            errors.append(
                DecisionContractError(
                    "invalid_candidate_path",
                    f"Candidate requires an absolute {key}.",
                    f"candidates[{index}]",
                )
            )
            continue
        if value in result:
            errors.append(
                DecisionContractError(
                    "duplicate_candidate_path",
                    "Canonical candidate path appears more than once.",
                    value,
                )
            )
        result.add(value)
    declared = document.get("candidate_visible_prim_count")
    if (
        not isinstance(declared, int)
        or isinstance(declared, bool)
        or declared != len(result)
    ):
        errors.append(
            DecisionContractError(
                "candidate_count_mismatch",
                "candidate_visible_prim_count does not match canonical candidates.",
            )
        )
    return result


def _group_list(
    patch: dict[str, Any],
    key: str,
    *,
    required: bool,
    errors: list[DecisionContractError],
) -> list[dict[str, Any]]:
    value = patch.get(key)
    if not isinstance(value, list):
        if required:
            errors.append(
                DecisionContractError(
                    f"invalid_{key}",
                    f"Material decision patch requires a {key} list.",
                )
            )
        return []
    result = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            errors.append(
                DecisionContractError(
                    "invalid_decision_group",
                    "Decision group must be an object.",
                    f"{key}[{index}]",
                )
            )
        else:
            result.append(item)
    return result


def _required_group_string(
    group: dict[str, Any],
    key: str,
    location: str,
    errors: list[DecisionContractError],
) -> str:
    value = group.get(key)
    if not isinstance(value, str) or not value.strip():
        errors.append(
            DecisionContractError(
                f"missing_{key}",
                f"Decision group requires a non-empty {key}.",
                location,
            )
        )
        return ""
    return value.strip()


def _claim_group_paths(
    group: dict[str, Any],
    *,
    location: str,
    path_space: Literal["source", "inspection"],
    canonical: set[str],
    claimed: dict[str, str],
    errors: list[DecisionContractError],
) -> None:
    keys = (
        ("runtime_prim_paths", "prim_paths")
        if path_space == "inspection"
        else ("prim_paths", "source_prim_paths")
    )
    values: Any = None
    for key in keys:
        if isinstance(group.get(key), list):
            values = group[key]
            break
    if not isinstance(values, list) or not values:
        errors.append(
            DecisionContractError(
                "missing_decision_paths",
                f"Decision group requires non-empty {keys[0]}.",
                location,
            )
        )
        return
    seen_in_group: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value.startswith("/"):
            errors.append(
                DecisionContractError(
                    "invalid_decision_path",
                    "Decision target must be an absolute USD prim path.",
                    location,
                )
            )
            continue
        if value in seen_in_group:
            errors.append(
                DecisionContractError(
                    "duplicate_decision_path",
                    "Decision target is duplicated within its group.",
                    value,
                )
            )
            continue
        seen_in_group.add(value)
        if value not in canonical:
            errors.append(
                DecisionContractError(
                    "unknown_candidate_path",
                    "Decision target is not a canonical visible candidate.",
                    value,
                )
            )
            continue
        if value in claimed:
            errors.append(
                DecisionContractError(
                    "duplicate_candidate_decision",
                    f"Candidate was already claimed by {claimed[value]}.",
                    value,
                )
            )
            continue
        claimed[value] = location
