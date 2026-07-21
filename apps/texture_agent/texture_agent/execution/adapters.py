# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adapters from legacy texture task units to immutable plan units."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

from texture_agent.functions.detail_policy import apply_detail_policy_to_prompt
from texture_agent.functions.material_discovery import PrimTextureUnit
from texture_agent.planning import TexturePlan, TexturePlanUnit, TextureUnitMode


def _runtime_identity(
    plan_unit: TexturePlanUnit,
    runtime_unit: PrimTextureUnit,
) -> bool:
    if runtime_unit.material_info.prim_path not in plan_unit.material_prim_paths:
        return False
    if plan_unit.unit_mode == TextureUnitMode.PER_MATERIAL:
        return not runtime_unit.prim_path
    if plan_unit.unit_mode == TextureUnitMode.PER_PRIM:
        if plan_unit.member_subset_paths:
            return False
        return tuple(plan_unit.member_prim_paths) == (runtime_unit.prim_path,)
    return False


def bind_prim_texture_units_to_plan(
    plan: TexturePlan,
    runtime_units: list[PrimTextureUnit],
) -> list[PrimTextureUnit]:
    """Return exact approved units, in plan order, rekeyed by stable ID.

    Extra discovered/runtime units are intentionally ignored. Missing or
    ambiguous approved identities fail before any backend invocation. The
    current ``PrimTextureUnit`` shape cannot express a multi-material group or
    a material-binding subset; those plan modes remain available to executor
    implementations with richer unit types and fail explicitly in this legacy
    adapter.
    """
    if not plan.decision.execution_allowed:
        raise ValueError(
            f"Texture plan is not approved for execution: {plan.decision.state.value}"
        )

    by_material: dict[str, list[PrimTextureUnit]] = defaultdict(list)
    for runtime_unit in runtime_units:
        by_material[runtime_unit.material_info.prim_path].append(runtime_unit)

    bound: list[PrimTextureUnit] = []
    for plan_unit in plan.selected_units:
        if plan_unit.unit_mode == TextureUnitMode.PER_GROUP:
            raise ValueError(
                "PrimTextureUnit cannot represent per_group Texture Plan units; "
                "use a group-aware unit runner"
            )
        if (
            plan_unit.unit_mode == TextureUnitMode.PER_PRIM
            and plan_unit.member_subset_paths
        ):
            raise ValueError(
                "PrimTextureUnit cannot represent material-binding subset units; "
                "use a subset-aware unit runner"
            )
        candidates = [
            runtime_unit
            for material_path in plan_unit.material_prim_paths
            for runtime_unit in by_material.get(material_path, [])
            if _runtime_identity(plan_unit, runtime_unit)
        ]
        if not candidates:
            raise ValueError(
                f"Approved texture unit {plan_unit.unit_id} has no matching runtime unit"
            )
        if len(candidates) > 1:
            raise ValueError(
                f"Approved texture unit {plan_unit.unit_id} matches multiple runtime units"
            )
        runtime_unit = candidates[0]
        detail_policy = plan_unit.detail_policy.value
        bound.append(
            replace(
                runtime_unit,
                key=plan_unit.unit_id,
                prompt=apply_detail_policy_to_prompt(
                    runtime_unit.prompt,
                    detail_policy,
                ),
                detail_policy=detail_policy,
            )
        )
    return bound
