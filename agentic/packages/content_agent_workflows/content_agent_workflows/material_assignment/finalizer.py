# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Workflow policy for sealing Material assignment decisions.

The finalizer validates coverage and library identity before a launcher asks a
scene adapter to author operations.  It intentionally contains no usd-cli or
renderer import: those are primitive execution details owned by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .decisions import DecisionContractError, validate_material_decision
from .manifest import ResolvedMaterialManifest
from .policy import (
    MATERIAL_ASSIGNMENT_TARGET_PRIM_SOFT_CAP,
    PAINTED_OR_SATURATED_MATERIAL_TAGS,
    structured_finalizer_rejection,
)


@dataclass(frozen=True, slots=True)
class MaterialFinalizationPolicy:
    """Accepted/rejected decision groups plus exact coverage accounting."""

    material_assignments: tuple[dict[str, Any], ...]
    reviewed_no_override: tuple[dict[str, Any], ...]
    rejected_groups: tuple[dict[str, Any], ...]
    coverage: dict[str, int]


@dataclass(frozen=True, slots=True)
class MaterialDecisionNormalization:
    """Normalized workflow decisions before an authoring adapter receives them.

    Canonical source targets remain the coverage truth.  Optimizer/runtime paths
    are aliases only; this is deliberately evaluated before any scene-tool call.
    """

    material_assignments: tuple[dict[str, Any], ...]
    reviewed_no_override: tuple[dict[str, Any], ...]
    rejected_groups: tuple[dict[str, Any], ...]


def finalize_material_policy(
    decision_patch: dict[str, Any],
    *,
    candidates: dict[str, Any],
    manifest: ResolvedMaterialManifest,
    respect_existing_material_bindings: bool,
) -> MaterialFinalizationPolicy:
    """Apply workflow guardrails without executing any scene operation.

    Invalid path/material identity or incomplete coverage is a hard policy error;
    agent-proposed groups that violate the narrow structured guardrails are retained
    as explicit rejected artifacts so the parent can revise them.
    """

    normalized = normalize_material_decision_policy(
        decision_patch,
        candidates=candidates,
        manifest=manifest,
        respect_existing_material_bindings=respect_existing_material_bindings,
    )
    normalized_patch = {
        **decision_patch,
        "material_assignments": list(normalized.material_assignments),
        "reviewed_no_override": list(normalized.reviewed_no_override),
    }
    validation = validate_material_decision(
        normalized_patch,
        candidates=candidates,
        manifest=manifest,
        clear_materials=not respect_existing_material_bindings,
    )
    assignments = list(normalized.material_assignments)
    reviewed = list(normalized.reviewed_no_override)
    rejected = list(normalized.rejected_groups)
    errors: list[DecisionContractError] = []
    if rejected:
        first = rejected[0]
        errors.append(
            DecisionContractError(
                str(first.get("policy_code") or "material_assignment_policy_rejected"),
                str(
                    first.get("rejection_reason")
                    or "Material assignment policy rejected a decision."
                ),
            )
        )
    errors.extend(validation.errors)
    if errors:
        raise MaterialDecisionPolicyError(
            errors=tuple(errors), rejected=tuple(rejected)
        )
    if rejected:
        raise MaterialDecisionPolicyError(
            errors=(
                DecisionContractError(
                    "structured_finalizer_guardrail",
                    "One or more material assignment groups violate workflow policy.",
                ),
            ),
            rejected=tuple(rejected),
        )
    canonical = set(validation.canonical_candidate_paths)
    claimed = set(validation.claimed_candidate_paths)
    return MaterialFinalizationPolicy(
        material_assignments=tuple(assignments),
        reviewed_no_override=tuple(reviewed),
        rejected_groups=(),
        coverage={
            "candidate_visible_prim_count": len(canonical),
            "material_assignment_prim_count": len(
                _paths(assignments, candidates.get("path_space"))
            ),
            "reviewed_no_override_prim_count": len(
                _paths(reviewed, candidates.get("path_space"))
            ),
            "claimed_candidate_prim_count": len(claimed),
            "unassigned_visible_prim_count": len(canonical - claimed),
        },
    )


class MaterialDecisionPolicyError(ValueError):
    """A serializable policy rejection, distinct from scene-tool failures."""

    def __init__(
        self,
        *,
        errors: tuple[DecisionContractError, ...],
        rejected: tuple[dict[str, Any], ...],
    ) -> None:
        self.errors = errors
        self.rejected = rejected
        details = "; ".join(error.message for error in errors)
        super().__init__(details)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "content-agents.material-finalizer-rejections.v1",
            "errors": [
                {"code": error.code, "message": error.message, "path": error.path}
                for error in self.errors
            ],
            "rejected_groups": list(self.rejected),
        }


def normalize_material_decision_policy(
    decision_patch: dict[str, Any],
    *,
    candidates: dict[str, Any],
    manifest: ResolvedMaterialManifest,
    respect_existing_material_bindings: bool,
) -> MaterialDecisionNormalization:
    """Resolve aliases and enforce group-level authoring guardrails.

    This preserves the old finalizer's *workflow* rules while intentionally
    omitting its HTTP commands, renderer calls, progress previews, and session
    cleanup.  In particular, alias expansion is accepted only when it maps to
    exactly one canonical candidate; shared runtime or source-authoring aliases
    require a complete, same-material decision before authoring can begin.
    """

    canonical = _candidate_path_set(candidates)
    alias_targets = _candidate_alias_target_path_map(candidates)
    unambiguous_aliases = {
        alias: targets for alias, targets in alias_targets.items() if len(targets) == 1
    }
    ambiguous_aliases = {
        alias for alias, targets in alias_targets.items() if len(targets) > 1
    }
    assignments, rejected = _normalize_assignments(
        decision_patch,
        candidates=candidates,
        manifest=manifest,
        canonical=canonical,
        unambiguous_aliases=unambiguous_aliases,
        ambiguous_aliases=ambiguous_aliases,
    )
    reviewed, reviewed_rejected = _normalize_reviewed_no_override(
        decision_patch,
        candidates=candidates,
        canonical=canonical,
        unambiguous_aliases=unambiguous_aliases,
        ambiguous_aliases=ambiguous_aliases,
        allowed=respect_existing_material_bindings,
    )
    rejected.extend(reviewed_rejected)
    assignments, alias_rejected = _coalesce_shared_runtime_aliases(
        assignments,
        candidates=candidates,
        reviewed=reviewed,
    )
    rejected.extend(alias_rejected)
    return MaterialDecisionNormalization(
        material_assignments=tuple(assignments),
        reviewed_no_override=tuple(reviewed),
        rejected_groups=tuple(rejected),
    )


def _normalize_assignments(
    decision_patch: dict[str, Any],
    *,
    candidates: dict[str, Any],
    manifest: ResolvedMaterialManifest,
    canonical: set[str],
    unambiguous_aliases: dict[str, list[str]],
    ambiguous_aliases: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    path_space = _candidate_path_space(candidates)
    shapes = _shape_hints_by_path(candidates)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    claimed: dict[str, str] = {}
    for index, raw in enumerate(_groups(decision_patch, "material_assignments")):
        family = str(raw.get("family") or f"Material assignment {index + 1}")
        paths, invalid = _resolve_group_paths(
            raw,
            candidates=candidates,
            canonical=canonical,
            unambiguous_aliases=unambiguous_aliases,
        )
        for path in invalid:
            rejected.append(
                _rejected_group(
                    raw,
                    family=family,
                    paths=[path],
                    candidates=candidates,
                    reason=(
                        "Rejected material assignment for ambiguous aliases that map to "
                        "multiple visible material candidates. Use an exact canonical "
                        f"{path_space}-space target path instead."
                        if path in ambiguous_aliases
                        else f"Rejected material assignment for {path_space}-space paths that were not visible material candidates."
                    ),
                )
            )
        if not paths:
            continue
        name = str(raw.get("material_name") or "").strip()
        entry = manifest.by_name.get(name)
        requested_path = str(raw.get("material_path") or "").strip()
        if entry is None:
            rejected.append(
                _rejected_group(
                    raw,
                    family=family,
                    paths=paths,
                    candidates=candidates,
                    reason="Rejected material assignment with material_name not found in the resolved material manifest.",
                    policy_code="unknown_material",
                )
            )
            continue
        if requested_path and requested_path != entry.binding_path:
            rejected.append(
                _rejected_group(
                    raw,
                    family=family,
                    paths=paths,
                    candidates=candidates,
                    reason="Rejected material assignment whose material_path does not match the resolved material manifest.",
                    policy_code="material_path_mismatch",
                )
            )
            continue
        duplicate = [path for path in paths if path in claimed]
        if duplicate:
            rejected.append(
                _rejected_group(
                    raw,
                    family=family,
                    paths=duplicate,
                    candidates=candidates,
                    reason="Rejected duplicate material assignment for canonical candidate target path(s) already claimed by an earlier assignment.",
                )
            )
        paths = [path for path in paths if path not in claimed]
        if not paths:
            continue
        group = {
            "family": family,
            "coverage_status": "material_assignment",
            "material_name": entry.name,
            "material_path": entry.binding_path,
            "material_tags": list(entry.tags),
            "material_description": entry.description,
            "material_manifest_semantics": entry.as_palette_record()[
                "manifest_semantics"
            ],
            "path_space": path_space,
            "runtime_space": "inspection" if path_space == "inspection" else "source",
            "runtime_prim_paths": paths if path_space == "inspection" else [],
            "source_prim_paths": _source_paths_for_targets(paths, candidates),
            "prim_paths": paths,
            "rationale": str(raw.get("rationale") or "").strip(),
        }
        reason = _guardrail_reason(group, shapes)
        if reason:
            rejected.append(
                _rejected_group(
                    group,
                    family=family,
                    paths=paths,
                    candidates=candidates,
                    reason=reason,
                )
            )
            continue
        accepted.append(group)
        claimed.update({path: family for path in paths})
    return accepted, rejected


def _normalize_reviewed_no_override(
    decision_patch: dict[str, Any],
    *,
    candidates: dict[str, Any],
    canonical: set[str],
    unambiguous_aliases: dict[str, list[str]],
    ambiguous_aliases: set[str],
    allowed: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    path_space = _candidate_path_space(candidates)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for index, raw in enumerate(_groups(decision_patch, "reviewed_no_override")):
        family = str(raw.get("family") or f"Reviewed no-override {index + 1}")
        paths, invalid = _resolve_group_paths(
            raw,
            candidates=candidates,
            canonical=canonical,
            unambiguous_aliases=unambiguous_aliases,
        )
        if not allowed:
            rejected.append(
                _rejected_group(
                    raw,
                    family=family,
                    paths=paths or invalid,
                    candidates=candidates,
                    reason="Rejected reviewed-no-override decision because this session started from clean materials. Assign an explicit library material to each visible candidate instead.",
                )
            )
            continue
        for path in invalid:
            rejected.append(
                _rejected_group(
                    raw,
                    family=family,
                    paths=[path],
                    candidates=candidates,
                    reason=(
                        "Rejected reviewed-no-override decision for ambiguous aliases that map to multiple visible material candidates."
                        if path in ambiguous_aliases
                        else f"Rejected reviewed-no-override decision for {path_space}-space paths that were not visible material candidates."
                    ),
                )
            )
        if paths:
            accepted.append(
                {
                    "family": family,
                    "coverage_status": "preserved_existing",
                    "material_name": "Reviewed current material",
                    "material_path": None,
                    "path_space": path_space,
                    "runtime_space": "inspection"
                    if path_space == "inspection"
                    else "source",
                    "runtime_prim_paths": paths if path_space == "inspection" else [],
                    "source_prim_paths": _source_paths_for_targets(paths, candidates),
                    "prim_paths": paths,
                    "rationale": str(raw.get("rationale") or "").strip(),
                }
            )
    return accepted, rejected


def _groups(patch: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = patch.get(key)
    return (
        [dict(item) for item in value if isinstance(item, dict)]
        if isinstance(value, list)
        else []
    )


def _candidate_path_space(candidates: dict[str, Any]) -> str:
    return "inspection" if candidates.get("path_space") == "inspection" else "source"


def _values(value: Any) -> list[str]:
    if isinstance(value, str) and value:
        return [value]
    return (
        [item for item in value if isinstance(item, str) and item]
        if isinstance(value, list)
        else []
    )


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _candidate_targets(candidate: dict[str, Any], path_space: str) -> list[str]:
    if path_space == "inspection":
        return _dedupe(
            _values(candidate.get("runtime_paths"))
            or _values(candidate.get("inspection_paths"))
            or _values(candidate.get("runtime_path"))
            or _values(candidate.get("inspection_path"))
            or _values(candidate.get("prim_paths"))
        )
    return _dedupe(
        _values(candidate.get("source_path"))
        or _values(candidate.get("source_paths"))
        or _values(candidate.get("prim_path"))
        or _values(candidate.get("prim_paths"))
    )


def _candidate_path_set(candidates: dict[str, Any]) -> set[str]:
    space = _candidate_path_space(candidates)
    return {
        path
        for candidate in candidates.get("candidates", [])
        if isinstance(candidate, dict)
        for path in _candidate_targets(candidate, space)
    }


def _candidate_alias_target_path_map(
    candidates: dict[str, Any],
) -> dict[str, list[str]]:
    space = _candidate_path_space(candidates)
    aliases: dict[str, list[str]] = {}
    keys = (
        "runtime_path",
        "runtime_paths",
        "runtime_prim_paths",
        "inspection_path",
        "inspection_paths",
        "inspection_prim_paths",
        "source_path",
        "source_paths",
        "source_prim_paths",
        "original_source_path",
        "original_source_paths",
        "prim_path",
        "prim_paths",
    )
    for candidate in candidates.get("candidates", []):
        if not isinstance(candidate, dict):
            continue
        targets = _candidate_targets(candidate, space)
        for alias in _dedupe(
            [path for key in keys for path in _values(candidate.get(key))] + targets
        ):
            aliases.setdefault(alias, [])
            aliases[alias] = _dedupe(aliases[alias] + targets)
    return aliases


def _source_paths_for_targets(
    paths: list[str], candidates: dict[str, Any]
) -> list[str]:
    space = _candidate_path_space(candidates)
    sources: dict[str, list[str]] = {}
    for candidate in candidates.get("candidates", []):
        if not isinstance(candidate, dict):
            continue
        source_paths = _values(candidate.get("source_paths")) or _values(
            candidate.get("source_path")
        )
        for target in _candidate_targets(candidate, space):
            sources[target] = _dedupe(source_paths or [target])
    return _dedupe([source for path in paths for source in sources.get(path, [path])])


def _shape_hints_by_path(candidates: dict[str, Any]) -> dict[str, str]:
    space = _candidate_path_space(candidates)
    return {
        path: str(candidate.get("shape_hint") or "unknown")
        for candidate in candidates.get("candidates", [])
        if isinstance(candidate, dict)
        for path in _candidate_targets(candidate, space)
    }


def _resolve_group_paths(
    group: dict[str, Any],
    *,
    candidates: dict[str, Any],
    canonical: set[str],
    unambiguous_aliases: dict[str, list[str]],
) -> tuple[list[str], list[str]]:
    space = _candidate_path_space(candidates)
    ordered_keys = (
        (
            "runtime_prim_paths",
            "inspection_prim_paths",
            "prim_paths",
            "source_prim_paths",
            "source_paths",
        )
        if space == "inspection"
        else (
            "prim_paths",
            "source_prim_paths",
            "source_paths",
            "runtime_prim_paths",
            "inspection_prim_paths",
        )
    )
    raw = next(
        (_values(group.get(key)) for key in ordered_keys if _values(group.get(key))), []
    )
    paths: list[str] = []
    invalid: list[str] = []
    for path in _dedupe(raw):
        if path in canonical:
            paths.append(path)
        elif path in unambiguous_aliases:
            paths.extend(unambiguous_aliases[path])
        else:
            invalid.append(path)
    return _dedupe(paths), _dedupe(invalid)


def _rejected_group(
    group: dict[str, Any],
    *,
    family: str,
    paths: list[str],
    candidates: dict[str, Any],
    reason: str,
    policy_code: str | None = None,
) -> dict[str, Any]:
    space = _candidate_path_space(candidates)
    result = {
        "family": family,
        "coverage_status": str(group.get("coverage_status") or "material_assignment"),
        "material_name": group.get("material_name"),
        "material_path": group.get("material_path"),
        "prim_paths": _dedupe(paths),
        "runtime_prim_paths": _dedupe(paths) if space == "inspection" else [],
        "source_prim_paths": _source_paths_for_targets(_dedupe(paths), candidates),
        "rationale": str(group.get("rationale") or "").strip(),
        "rejection_reason": reason,
    }
    if policy_code:
        result["policy_code"] = policy_code
    return result


def _paths(groups: list[dict[str, Any]], path_space: Any) -> set[str]:
    keys = (
        ("runtime_prim_paths", "prim_paths")
        if path_space == "inspection"
        else ("source_prim_paths", "prim_paths")
    )
    return {
        str(path)
        for group in groups
        for key in keys
        for path in group.get(key, [])
        if isinstance(path, str) and path
    }


def _guardrail_reason(group: dict[str, Any], shapes: dict[str, str]) -> str | None:
    paths = _values(group.get("prim_paths"))
    shape_hints = {shapes.get(path, "unknown") for path in paths} - {"unknown"}
    tags = set(_values(group.get("material_tags")))
    saturated_or_painted = bool(tags & PAINTED_OR_SATURATED_MATERIAL_TAGS)
    if saturated_or_painted and "slender_bar" in shape_hints:
        return structured_finalizer_rejection("slender_bar_metal_family")
    if saturated_or_painted and len(paths) > 3 and len(shape_hints) > 1:
        return structured_finalizer_rejection("split_broad_painted_mixed_groups")
    if len(paths) > MATERIAL_ASSIGNMENT_TARGET_PRIM_SOFT_CAP and len(shape_hints) > 1:
        return structured_finalizer_rejection("split_large_mixed_groups")
    return None


def _coalesce_shared_runtime_aliases(
    groups: list[dict[str, Any]],
    *,
    candidates: dict[str, Any],
    reviewed: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Require atomic decisions for candidates sharing an authoring alias."""

    path_space = _candidate_path_space(candidates)
    if not groups:
        return groups, []
    alias_name = (
        "optimized runtime alias"
        if path_space == "source"
        else "shared source authoring alias"
    )
    aliases: dict[str, set[str]] = {}
    by_path: dict[str, dict[str, Any]] = {}
    for candidate in candidates.get("candidates", []):
        if not isinstance(candidate, dict):
            continue
        targets = _candidate_targets(candidate, path_space)
        alias_paths = (
            _dedupe(
                _values(candidate.get("runtime_paths"))
                or _values(candidate.get("inspection_paths"))
                or _values(candidate.get("runtime_path"))
                or _values(candidate.get("inspection_path"))
            )
            if path_space == "source"
            else _dedupe(
                _values(candidate.get("source_paths"))
                or _values(candidate.get("source_path"))
                or _values(candidate.get("original_source_paths"))
                or _values(candidate.get("original_source_path"))
            )
        )
        for target in targets:
            by_path[target] = candidate
            for alias in alias_paths:
                aliases.setdefault(alias, set()).add(target)
    components = [paths for paths in aliases.values() if len(paths) > 1]
    if not components:
        return groups, []
    accepted = [dict(group) for group in groups]
    rejected: list[dict[str, Any]] = []
    reviewed_paths = {
        path for group in reviewed for path in _values(group.get("prim_paths"))
    }
    component_counts: dict[str, int] = {}
    for component in components:
        for path in component:
            component_counts[path] = component_counts.get(path, 0) + 1
    overlapping = {path for path, count in component_counts.items() if count > 1}
    if overlapping:
        reason = (
            "Rejected material assignment for canonical candidates that participate "
            f"in overlapping {alias_name}s. No single atomic "
            "authoring target covers the component."
        )
        for index, group in enumerate(accepted):
            paths = set(_values(group.get("prim_paths"))) & overlapping
            if not paths:
                continue
            rejected.append(
                _rejected_group(
                    group,
                    family=str(group.get("family") or ""),
                    paths=sorted(paths),
                    candidates=candidates,
                    reason=reason,
                )
            )
            remaining = [
                path
                for path in _values(group.get("prim_paths"))
                if path not in overlapping
            ]
            accepted[index] = _group_with_paths(group, remaining, candidates)
    for component in components:
        indexed = [
            (index, set(_values(group.get("prim_paths"))) & component)
            for index, group in enumerate(accepted)
        ]
        indexed = [(index, paths) for index, paths in indexed if paths]
        if not indexed:
            continue
        covered = set().union(*(paths for _index, paths in indexed))
        keys = {
            (
                str(accepted[index].get("material_name") or ""),
                str(accepted[index].get("material_path") or ""),
            )
            for index, _paths in indexed
        }
        collapsed = path_space == "source" and any(
            bool(by_path.get(path, {}).get("instance_collapsed"))
            and str(by_path.get(path, {}).get("runtime_space") or "") == "inspection"
            for path in component
        )
        if covered != component or len(keys) != 1 or collapsed:
            if len(keys) != 1:
                reason = f"Rejected conflicting material assignments for canonical candidates that share one {alias_name}; they must use the same material."
            elif covered != component:
                reason = f"Rejected partial material assignment for canonical candidates that share one {alias_name}; every represented candidate must be decided together."
            else:
                reason = "Rejected material assignment for instance-collapsed candidates that share one optimized runtime alias; no faithful atomic source override exists."
            for index, paths in indexed:
                group = accepted[index]
                rejected.append(
                    _rejected_group(
                        group,
                        family=str(group.get("family") or ""),
                        paths=sorted(paths),
                        candidates=candidates,
                        reason=reason,
                    )
                )
                remaining = [
                    path
                    for path in _values(group.get("prim_paths"))
                    if path not in component
                ]
                accepted[index] = _group_with_paths(group, remaining, candidates)
            continue
        # A reviewed-no-override sibling is not sufficient: it would still be
        # altered by the shared authoring action.  The component must be explicit.
        if component & reviewed_paths:
            reason = f"Rejected material assignment because one {alias_name} has a reviewed-no-override sibling; author an explicit same-material decision for the full component."
            for index, paths in indexed:
                group = accepted[index]
                rejected.append(
                    _rejected_group(
                        group,
                        family=str(group.get("family") or ""),
                        paths=sorted(paths),
                        candidates=candidates,
                        reason=reason,
                    )
                )
                remaining = [
                    path
                    for path in _values(group.get("prim_paths"))
                    if path not in component
                ]
                accepted[index] = _group_with_paths(group, remaining, candidates)
            continue
        # Same-material complete components are coalesced to a single workflow
        # group so every adapter receives one atomic authoring instruction.
        if len(indexed) > 1:
            first_index = indexed[0][0]
            merged = dict(accepted[first_index])
            # A workflow group may cover more than one independent alias
            # component. Coalescing this component must retain those unrelated
            # paths so a later component can coalesce them in turn.
            first_remaining = [
                path
                for path in _values(accepted[first_index].get("prim_paths"))
                if path not in component
            ]
            merged_paths = sorted(_dedupe([*first_remaining, *component]))
            merged = _group_with_paths(merged, merged_paths, candidates)
            merged["family"] = " / ".join(
                _dedupe(
                    [
                        family
                        for index, _paths in indexed
                        for family in str(accepted[index].get("family") or "").split(
                            " / "
                        )
                        if family
                    ]
                )
            )
            for index, _paths in indexed:
                if index == first_index:
                    accepted[index] = merged
                    continue
                remaining = [
                    path
                    for path in _values(accepted[index].get("prim_paths"))
                    if path not in component
                ]
                accepted[index] = _group_with_paths(
                    accepted[index], remaining, candidates
                )
    return [group for group in accepted if _values(group.get("prim_paths"))], rejected


def _group_with_paths(
    group: dict[str, Any], paths: list[str], candidates: dict[str, Any]
) -> dict[str, Any]:
    """Keep normalized canonical/runtime/source path fields in lockstep."""

    canonical_paths = _dedupe(paths)
    return {
        **group,
        "prim_paths": canonical_paths,
        "runtime_prim_paths": (
            canonical_paths if _candidate_path_space(candidates) == "inspection" else []
        ),
        "source_prim_paths": _source_paths_for_targets(canonical_paths, candidates),
    }
