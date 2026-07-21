# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic, backend-free Texture Agent plan construction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from texture_agent.planning.contracts import (
    TEXTURE_PLAN_HARD_CAP,
    TextureDetailPolicy,
    TextureDiscoveryMode,
    TexturePlan,
    TexturePlanCounts,
    TexturePlanDecision,
    TexturePlanDecisionState,
    TexturePlanExecution,
    TexturePlanLimits,
    TexturePlanRequest,
    TexturePlanSkippedItem,
    TexturePlanUnit,
    TextureSelectionKind,
    TextureUnitMode,
)


def _paths(value: Any) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return ()
    return tuple(sorted({str(item) for item in value if str(item).startswith("/")}))


def _material_path(material: Any) -> str:
    return str(getattr(material, "prim_path", ""))


def _material_name(material: Any) -> str:
    return (
        str(getattr(material, "name", ""))
        or _material_path(material).rsplit("/", 1)[-1]
    )


def _material_alias_paths(material: Any) -> tuple[str, ...]:
    return _paths(
        (
            _material_path(material),
            *tuple(getattr(material, "material_alias_paths", ()) or ()),
        )
    )


def _bound_prim_paths(material: Any) -> tuple[str, ...]:
    return _paths(getattr(material, "bound_prim_paths", ()))


def _bound_subset_paths(material: Any) -> tuple[str, ...]:
    return _paths(getattr(material, "bound_subset_paths", ()))


def _material_sort_key(material: Any) -> tuple[str, str]:
    return (_material_path(material), _material_name(material))


def _effective_members(material: Any) -> tuple[tuple[str, ...], tuple[str, ...]]:
    return _bound_prim_paths(material), _bound_subset_paths(material)


def _matching_spec(
    material: Any,
    material_textures: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    """Resolve a legacy name key or an explicit path-scoped material spec."""
    name = _material_name(material)
    aliases = _material_alias_paths(material)
    for alias in aliases:
        direct = material_textures.get(alias)
        if isinstance(direct, Mapping):
            return direct

    named = material_textures.get(name)
    if not isinstance(named, Mapping):
        return None
    scoped_path = named.get("material_path")
    if scoped_path is not None and str(scoped_path) not in aliases:
        return None
    return named


def _detail_policy(
    request: TexturePlanRequest,
    spec: Mapping[str, Any] | None,
    *,
    member_path: str | None = None,
) -> TextureDetailPolicy:
    value: Any = request.detail_policy
    if spec is not None:
        value = spec.get("detail_policy", value)
        per_prim = spec.get("per_prim")
        if member_path and isinstance(per_prim, Mapping):
            leaf = member_path.rsplit("/", 1)[-1]
            override = per_prim.get(member_path) or per_prim.get(leaf)
            if isinstance(override, Mapping):
                value = override.get("detail_policy", value)
    return TextureDetailPolicy(value)


def _candidate_materials(
    request: TexturePlanRequest,
    authored: Sequence[Any],
    effective: Sequence[Any],
) -> list[Any]:
    if request.discovery_mode is TextureDiscoveryMode.ALL_AUTHORED:
        return sorted(authored, key=_material_sort_key)
    if request.discovery_mode is TextureDiscoveryMode.EFFECTIVE_BOUND:
        return sorted(effective, key=_material_sort_key)

    material_paths = set(request.explicit_material_paths)
    prim_paths = set(request.explicit_prim_paths)
    selected: list[Any] = []
    for material in authored:
        bound_prims, bound_subsets = _effective_members(material)
        if (
            material_paths.intersection(_material_alias_paths(material))
            or prim_paths.intersection(bound_prims)
            or prim_paths.intersection(bound_subsets)
        ):
            selected.append(material)
    return sorted(selected, key=_material_sort_key)


def _selection_reason(
    discovery_mode: TextureDiscoveryMode,
) -> tuple[str, str]:
    if discovery_mode is TextureDiscoveryMode.EXPLICIT:
        return "explicit_scope", "Selected by the request's explicit USD path scope."
    if discovery_mode is TextureDiscoveryMode.ALL_AUTHORED:
        return "all_authored", "Selected by the explicit all-authored discovery mode."
    return (
        "effectively_bound",
        "Selected because renderable geometry or a material subset uses this material.",
    )


def _unit_sort_key(unit: TexturePlanUnit) -> tuple[Any, ...]:
    return (
        unit.material_prim_paths,
        unit.member_prim_paths,
        unit.member_subset_paths,
        unit.group_key or "",
        unit.unit_id,
    )


def _build_units(
    request: TexturePlanRequest,
    materials: Sequence[Any],
    material_textures: Mapping[str, Any],
    *,
    auto_prompt_enabled: bool,
) -> tuple[list[TexturePlanUnit], list[TexturePlanSkippedItem]]:
    reason_code, reason = _selection_reason(request.discovery_mode)
    units: list[TexturePlanUnit] = []
    skipped: list[TexturePlanSkippedItem] = []

    for material in materials:
        path = _material_path(material)
        name = _material_name(material)
        spec = _matching_spec(material, material_textures)
        if not auto_prompt_enabled and spec is None:
            skipped.append(
                TexturePlanSkippedItem(
                    item_kind=TextureSelectionKind.MATERIAL,
                    canonical_id=path,
                    display_name=name,
                    reason_code="not_requested",
                    reason=(
                        "The material is outside material_textures while automatic "
                        "prompting is disabled."
                    ),
                )
            )
            continue

        prim_paths, subset_paths = _effective_members(material)
        if (
            request.discovery_mode is TextureDiscoveryMode.EXPLICIT
            and request.explicit_prim_paths
        ):
            explicit_members = set(request.explicit_prim_paths)
            prim_paths = tuple(path for path in prim_paths if path in explicit_members)
            subset_paths = tuple(
                path for path in subset_paths if path in explicit_members
            )
        if request.unit_mode is TextureUnitMode.PER_PRIM:
            members = [
                *((TextureSelectionKind.PRIM, member) for member in prim_paths),
                *((TextureSelectionKind.SUBSET, member) for member in subset_paths),
            ]
            if not members:
                skipped.append(
                    TexturePlanSkippedItem(
                        item_kind=TextureSelectionKind.MATERIAL,
                        canonical_id=path,
                        display_name=name,
                        reason_code="no_renderable_member",
                        reason=(
                            "Per-prim planning requires a renderable bound prim or "
                            "material-binding subset."
                        ),
                    )
                )
                continue
            for kind, member in members:
                units.append(
                    TexturePlanUnit.build(
                        unit_mode=request.unit_mode,
                        material_prim_paths=(path,),
                        member_prim_paths=(member,)
                        if kind is TextureSelectionKind.PRIM
                        else (),
                        member_subset_paths=(member,)
                        if kind is TextureSelectionKind.SUBSET
                        else (),
                        display_name=f"{name} on {member}",
                        selection_reason_code=reason_code,
                        selection_reason=reason,
                        detail_policy=_detail_policy(
                            request,
                            spec,
                            member_path=member,
                        ),
                    )
                )
            continue

        if request.unit_mode is TextureUnitMode.PER_GROUP:
            units.append(
                TexturePlanUnit.build(
                    unit_mode=request.unit_mode,
                    material_prim_paths=(path,),
                    member_prim_paths=prim_paths,
                    member_subset_paths=subset_paths,
                    group_key=path,
                    display_name=name,
                    selection_reason_code=reason_code,
                    selection_reason=reason,
                    detail_policy=_detail_policy(request, spec),
                )
            )
            continue

        units.append(
            TexturePlanUnit.build(
                unit_mode=request.unit_mode,
                material_prim_paths=(path,),
                member_prim_paths=prim_paths,
                member_subset_paths=subset_paths,
                display_name=name,
                selection_reason_code=reason_code,
                selection_reason=reason,
                detail_policy=_detail_policy(request, spec),
            )
        )

    return sorted(units, key=_unit_sort_key), skipped


def _discovery_skips(effective_discovery: Any) -> list[TexturePlanSkippedItem]:
    records = getattr(effective_discovery, "skipped_materials", ()) or ()
    skipped: list[TexturePlanSkippedItem] = []
    for record in records:
        path = str(getattr(record, "material_prim_path", ""))
        if not path.startswith("/"):
            continue
        skipped.append(
            TexturePlanSkippedItem(
                item_kind=TextureSelectionKind.MATERIAL,
                canonical_id=path,
                display_name=getattr(record, "material_name", None),
                reason_code=str(
                    getattr(record, "reason_code", "not_effectively_bound")
                ),
                reason=str(
                    getattr(
                        record,
                        "reason",
                        "No renderable scene member effectively uses this material.",
                    )
                ),
            )
        )
    return skipped


def _decision(
    job_count: int,
    effective_cap: int,
    selected_units: Sequence[TexturePlanUnit],
) -> TexturePlanDecision:
    if job_count == 0:
        return TexturePlanDecision(
            state=TexturePlanDecisionState.REQUIRES_NARROWING,
            execution_allowed=False,
            explicit_narrowing_required=True,
            reasons=("The plan contains zero executable texture-generation jobs.",),
            recommended_actions=(
                "Select at least one effectively bound material or provide a "
                "matching explicit material/prim scope.",
            ),
        )
    if job_count > TEXTURE_PLAN_HARD_CAP:
        return TexturePlanDecision(
            state=TexturePlanDecisionState.UNSUPPORTED,
            execution_allowed=False,
            consolidation_required=True,
            explicit_narrowing_required=True,
            reasons=(
                f"The plan contains {job_count} generation jobs and exceeds the "
                f"hard maximum of {TEXTURE_PLAN_HARD_CAP}.",
            ),
            recommended_actions=(
                "Consolidate semantically equivalent materials or provide a smaller "
                "explicit material/prim scope before executing.",
            ),
        )
    if any(unit.unit_mode is TextureUnitMode.PER_GROUP for unit in selected_units):
        return TexturePlanDecision(
            state=TexturePlanDecisionState.UNSUPPORTED,
            execution_allowed=False,
            reasons=(
                "The default Texture Agent executor cannot execute per_group "
                "texture-plan units.",
            ),
            recommended_actions=(
                "Use per_material or per_prim planning, or provide a custom "
                "group-aware executor.",
            ),
        )
    if any(
        unit.unit_mode is TextureUnitMode.PER_PRIM and unit.member_subset_paths
        for unit in selected_units
    ):
        return TexturePlanDecision(
            state=TexturePlanDecisionState.UNSUPPORTED,
            execution_allowed=False,
            reasons=(
                "The default Texture Agent executor cannot execute per_prim "
                "material-binding subset units.",
            ),
            recommended_actions=(
                "Use per_material planning for subset-bound materials or provide "
                "a subset-aware executor.",
            ),
        )
    if job_count > effective_cap:
        return TexturePlanDecision(
            state=TexturePlanDecisionState.REQUIRES_OPERATOR_OVERRIDE,
            execution_allowed=False,
            reasons=(
                f"The plan contains {job_count} generation jobs and exceeds the "
                f"effective backend default cap of {effective_cap}.",
            ),
            recommended_actions=(
                "Narrow the scope or provide an intentional operator override no "
                f"greater than {TEXTURE_PLAN_HARD_CAP}.",
            ),
        )
    return TexturePlanDecision(
        state=TexturePlanDecisionState.READY,
        execution_allowed=True,
    )


def build_texture_plan(
    request: TexturePlanRequest,
    *,
    discovered_materials: Sequence[Any],
    effective_discovery: Any = None,
    material_textures: Mapping[str, Any] | None = None,
    auto_prompt_enabled: bool = False,
) -> TexturePlan:
    """Build an immutable plan without invoking an LLM or image backend.

    ``effective_discovery`` is the typed WP1 result when available. The
    fallback accepts the legacy ``MaterialInfo.bound_prim_paths`` shape so the
    planner and its contract fixtures remain independently testable.
    """
    authored = list(
        getattr(effective_discovery, "authored_materials", None) or discovered_materials
    )
    effective = list(
        getattr(effective_discovery, "effective_materials", None)
        or [material for material in authored if any(_effective_members(material))]
    )
    candidates = _candidate_materials(request, authored, effective)
    selected_units, selection_skips = _build_units(
        request,
        candidates,
        material_textures or {},
        auto_prompt_enabled=auto_prompt_enabled,
    )

    skipped_by_path: dict[str, TexturePlanSkippedItem] = {}
    if request.discovery_mode is TextureDiscoveryMode.EFFECTIVE_BOUND:
        for item in _discovery_skips(effective_discovery):
            skipped_by_path[item.canonical_id] = item
    candidate_paths = {_material_path(material) for material in candidates}
    for material in authored:
        path = _material_path(material)
        if path not in candidate_paths and path not in skipped_by_path:
            skipped_by_path[path] = TexturePlanSkippedItem(
                item_kind=TextureSelectionKind.MATERIAL,
                canonical_id=path,
                display_name=_material_name(material),
                reason_code="outside_discovery_scope",
                reason="The material is outside the selected discovery scope.",
            )
    for item in selection_skips:
        skipped_by_path[item.canonical_id] = item
    skipped_items = tuple(skipped_by_path[path] for path in sorted(skipped_by_path))

    renderable_prims = _paths(
        getattr(effective_discovery, "renderable_prim_paths", None)
        or [path for material in authored for path in _bound_prim_paths(material)]
    )
    renderable_subsets = _paths(
        getattr(effective_discovery, "renderable_subset_paths", None)
        or [path for material in authored for path in _bound_subset_paths(material)]
    )
    selected_material_paths = {
        path for unit in selected_units for path in unit.material_prim_paths
    }
    limits = TexturePlanLimits.from_request(request)
    return TexturePlan(
        request=request,
        limits=limits,
        execution=TexturePlanExecution.from_request(request),
        counts=TexturePlanCounts(
            authored_material_count=int(
                getattr(effective_discovery, "authored_material_count", len(authored))
            ),
            renderable_prim_count=int(
                getattr(
                    effective_discovery,
                    "renderable_prim_count",
                    len(renderable_prims),
                )
            ),
            renderable_subset_count=int(
                getattr(
                    effective_discovery,
                    "renderable_subset_count",
                    len(renderable_subsets),
                )
            ),
            effective_bound_material_count=int(
                getattr(
                    effective_discovery,
                    "effective_bound_material_count",
                    len(effective),
                )
            ),
            selected_material_count=len(selected_material_paths),
            selected_unit_count=len(selected_units),
            skipped_item_count=len(skipped_items),
            planned_generation_job_count=len(selected_units),
        ),
        selected_units=tuple(selected_units),
        skipped_items=skipped_items,
        decision=_decision(len(selected_units), limits.effective_cap, selected_units),
    )
