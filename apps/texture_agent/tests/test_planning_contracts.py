# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contract tests for issue #466 bounded texture planning."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from texture_agent.planning import (
    TEXTURE_PLAN_SCHEMA_VERSION,
    TEXTURE_UV_AWARE_DEFAULT_CAP,
    TextureDiscoveryMode,
    TexturePlan,
    TexturePlanCounts,
    TexturePlanDecision,
    TexturePlanDecisionState,
    TexturePlanExecution,
    TexturePlanLimits,
    TexturePlanRequest,
    TexturePlanSkippedItem,
    TexturePlanSource,
    TexturePlanUnit,
    TextureUnitMode,
    stable_texture_unit_id,
    texture_plan_json_schema,
    validate_texture_plan_payload,
)


def _unit(index: int, *, display_name: str | None = None) -> TexturePlanUnit:
    return TexturePlanUnit.build(
        unit_mode=TextureUnitMode.PER_MATERIAL,
        material_prim_paths=(f"/World/Looks/Material_{index:03d}",),
        member_prim_paths=(f"/World/Geometry/Mesh_{index:03d}",),
        display_name=display_name or f"Material_{index:03d}",
        selection_reason_code="effectively_bound",
        selection_reason="Selected because renderable geometry uses this material.",
        detail_policy="surface_only",
    )


def _plan(
    job_count: int,
    *,
    backend_default_cap: int = 32,
    operator_override_cap: int | None = None,
    execution_allowed: bool,
) -> TexturePlan:
    request = TexturePlanRequest(
        source=TexturePlanSource(
            source_asset="s3://example-bucket/assets/board.usd",
            upstream_assignment_artifact=(
                "s3://example-bucket/assignments/materials.json"
            ),
        ),
        backend="service",
        backend_default_cap=backend_default_cap,
        operator_override_cap=operator_override_cap,
        detail_policy="surface_only",
        max_concurrency=2,
        unit_timeout_seconds=900,
    )
    units = tuple(_unit(index) for index in range(job_count))

    if execution_allowed:
        decision = TexturePlanDecision(
            state=TexturePlanDecisionState.READY,
            execution_allowed=True,
        )
    elif job_count > 64:
        decision = TexturePlanDecision(
            state=TexturePlanDecisionState.UNSUPPORTED,
            execution_allowed=False,
            consolidation_required=True,
            explicit_narrowing_required=True,
            reasons=("The plan exceeds the 64-unit hard cap.",),
            recommended_actions=(
                "Consolidate semantic materials or provide a smaller explicit scope.",
            ),
        )
    else:
        decision = TexturePlanDecision(
            state=TexturePlanDecisionState.REQUIRES_OPERATOR_OVERRIDE,
            execution_allowed=False,
            reasons=("The plan exceeds the effective backend default cap.",),
            recommended_actions=(
                "Narrow the scope or provide an intentional operator override.",
            ),
        )

    return TexturePlan(
        generated_at=datetime(2026, 6, 29, tzinfo=UTC),
        request=request,
        limits=TexturePlanLimits.from_request(request),
        execution=TexturePlanExecution.from_request(request),
        counts=TexturePlanCounts(
            authored_material_count=max(job_count, 1),
            renderable_prim_count=job_count,
            renderable_subset_count=0,
            effective_bound_material_count=job_count,
            selected_material_count=job_count,
            selected_unit_count=job_count,
            skipped_item_count=0,
            planned_generation_job_count=job_count,
        ),
        selected_units=units,
        decision=decision,
    )


def test_plan_round_trips_as_strict_versioned_json() -> None:
    plan = _plan(2, execution_allowed=True)

    payload = plan.model_dump_json(indent=2)
    restored = validate_texture_plan_payload(payload)

    assert restored == plan
    assert validate_texture_plan_payload(plan) is plan
    assert validate_texture_plan_payload(plan.model_dump()) == plan
    assert restored.schema_version == TEXTURE_PLAN_SCHEMA_VERSION
    assert restored.request.source.upstream_assignment_artifact is not None
    assert restored.counts.renderable_prim_count == 2
    assert restored.counts.planned_generation_job_count == 2
    assert restored.limits.global_default_cap == 32
    assert restored.limits.hard_cap == 64
    assert restored.execution.texture_size == 1024


def test_contract_models_are_immutable() -> None:
    plan = _plan(1, execution_allowed=True)

    with pytest.raises(ValidationError, match="frozen"):
        plan.counts.selected_unit_count = 2


def test_stable_unit_identity_uses_paths_not_display_names() -> None:
    first = TexturePlanUnit.build(
        unit_mode="per_material",
        material_prim_paths=("/World/Looks/A/Paint",),
        member_prim_paths=("/World/MeshA",),
        display_name="Paint",
        selection_reason_code="effectively_bound",
        selection_reason="Selected by effective binding.",
        detail_policy="default",
    )
    renamed = TexturePlanUnit.build(
        unit_mode="per_material",
        material_prim_paths=("/World/Looks/A/Paint",),
        member_prim_paths=("/World/MeshB",),
        display_name="Renamed Paint",
        selection_reason_code="effectively_bound",
        selection_reason="Selected by effective binding.",
        detail_policy="default",
    )
    duplicate_name_different_path = TexturePlanUnit.build(
        unit_mode="per_material",
        material_prim_paths=("/World/Looks/B/Paint",),
        display_name="Paint",
        selection_reason_code="effectively_bound",
        selection_reason="Selected by effective binding.",
        detail_policy="default",
    )

    assert first.unit_id == renamed.unit_id
    assert first.unit_id != duplicate_name_different_path.unit_id


def test_per_material_membership_is_auditable_but_not_identity() -> None:
    first = TexturePlanUnit.build(
        unit_mode="per_material",
        material_prim_paths=("/World/Looks/Paint",),
        member_prim_paths=("/World/A",),
        display_name="Paint",
        selection_reason_code="effectively_bound",
        selection_reason="Selected by effective binding.",
        detail_policy="default",
    )
    second = TexturePlanUnit.build(
        unit_mode="per_material",
        material_prim_paths=("/World/Looks/Paint",),
        member_prim_paths=("/World/B",),
        display_name="Paint",
        selection_reason_code="effectively_bound",
        selection_reason="Selected by effective binding.",
        detail_policy="default",
    )

    assert first.unit_id == second.unit_id
    assert first.member_prim_paths != second.member_prim_paths


@pytest.mark.parametrize("unit_mode", ["per_material", "per_prim"])
def test_non_group_units_reject_group_key(unit_mode: str) -> None:
    with pytest.raises(ValidationError, match="must not record group_key"):
        TexturePlanUnit.build(
            unit_mode=unit_mode,
            material_prim_paths=("/World/Looks/Paint",),
            member_prim_paths=("/World/A",),
            group_key="ignored-group",
            display_name="Paint",
            selection_reason_code="effectively_bound",
            selection_reason="Selected by effective binding.",
            detail_policy="default",
        )


def test_group_identity_is_order_independent() -> None:
    forward = stable_texture_unit_id(
        unit_mode="per_group",
        material_prim_paths=("/World/Looks/Steel", "/World/Looks/Iron"),
        member_prim_paths=("/World/B", "/World/A"),
    )
    reverse = stable_texture_unit_id(
        unit_mode="per_group",
        material_prim_paths=("/World/Looks/Iron", "/World/Looks/Steel"),
        member_prim_paths=("/World/A", "/World/B"),
    )

    assert forward == reverse

    keyed_unit = TexturePlanUnit.build(
        unit_mode="per_group",
        material_prim_paths=("/World/Looks/Iron", "/World/Looks/Steel"),
        member_prim_paths=("/World/A", "/World/B"),
        group_key="ferrous-metals",
        display_name="Ferrous metals",
        selection_reason_code="semantic_group",
        selection_reason="Selected from an upstream semantic appearance group.",
        detail_policy="default",
    )
    assert keyed_unit.group_key == "ferrous-metals"


def test_group_unit_requires_key_or_member_path() -> None:
    with pytest.raises(ValidationError, match="group_key or at least one member"):
        TexturePlanUnit.build(
            unit_mode="per_group",
            material_prim_paths=("/World/Looks/Steel",),
            display_name="Unidentified group",
            selection_reason_code="semantic_group",
            selection_reason="Selected from a semantic appearance group.",
            detail_policy="default",
        )


def test_per_prim_identity_requires_exactly_one_member() -> None:
    with pytest.raises(ValidationError, match="exactly one member"):
        TexturePlanUnit.build(
            unit_mode="per_prim",
            material_prim_paths=("/World/Looks/Steel",),
            member_prim_paths=("/World/A", "/World/B"),
            display_name="Steel on two prims",
            selection_reason_code="explicit_scope",
            selection_reason="Selected by explicit prim scope.",
            detail_policy="default",
        )


def test_explicit_discovery_requires_a_material_or_prim_scope() -> None:
    with pytest.raises(ValidationError, match="explicit discovery requires"):
        TexturePlanRequest(
            source=TexturePlanSource(source_asset="scene.usd"),
            discovery_mode=TextureDiscoveryMode.EXPLICIT,
        )


def test_explicit_scope_paths_are_canonical_and_sorted() -> None:
    request = TexturePlanRequest(
        source=TexturePlanSource(source_asset="scene.usd"),
        discovery_mode="explicit",
        explicit_material_paths=("/World/Looks/Z", "/World/Looks/A"),
    )

    assert request.explicit_material_paths == (
        "/World/Looks/A",
        "/World/Looks/Z",
    )

    with pytest.raises(ValidationError, match="absolute USD prim path"):
        TexturePlanRequest(
            source=TexturePlanSource(source_asset="scene.usd"),
            discovery_mode="explicit",
            explicit_material_paths=("World/Looks/A",),
        )


@pytest.mark.parametrize(
    ("jobs", "backend_cap", "override", "execution_allowed"),
    [
        (16, 16, None, True),
        (17, 16, None, False),
        (32, 32, None, True),
        (33, 32, None, False),
        (64, 32, 64, True),
        (65, 32, None, False),
    ],
)
def test_supported_envelope_boundaries(
    jobs: int,
    backend_cap: int,
    override: int | None,
    execution_allowed: bool,
) -> None:
    plan = _plan(
        jobs,
        backend_default_cap=backend_cap,
        operator_override_cap=override,
        execution_allowed=execution_allowed,
    )

    assert plan.decision.execution_allowed is execution_allowed
    if jobs == TEXTURE_UV_AWARE_DEFAULT_CAP:
        assert plan.limits.backend_default_cap == TEXTURE_UV_AWARE_DEFAULT_CAP
    if jobs == 65:
        assert plan.decision.state == TexturePlanDecisionState.UNSUPPORTED
        assert plan.decision.consolidation_required is True
        assert plan.decision.explicit_narrowing_required is True


def test_execution_cannot_be_approved_above_effective_cap() -> None:
    with pytest.raises(ValidationError, match="above the effective texture-unit cap"):
        _plan(33, execution_allowed=True)


def test_override_decision_requires_jobs_above_effective_cap() -> None:
    valid = _plan(1, execution_allowed=True)

    with pytest.raises(
        ValidationError,
        match="requires planned jobs above the effective texture-unit cap",
    ):
        TexturePlan(
            generated_at=valid.generated_at,
            request=valid.request,
            limits=valid.limits,
            execution=valid.execution,
            counts=valid.counts,
            selected_units=valid.selected_units,
            decision=TexturePlanDecision(
                state="requires_operator_override",
                execution_allowed=False,
                reasons=("Override requested.",),
                recommended_actions=("Provide a larger override.",),
            ),
        )


@pytest.mark.parametrize(
    ("state", "consolidation_required", "explicit_narrowing_required", "error"),
    [
        ("ready", True, False, "must not require consolidation"),
        (
            "requires_consolidation",
            False,
            False,
            "requires consolidation_required=true",
        ),
        (
            "requires_narrowing",
            False,
            False,
            "requires explicit_narrowing_required=true",
        ),
        (
            "requires_operator_override",
            True,
            False,
            "must not require consolidation",
        ),
    ],
)
def test_decision_state_rejects_contradictory_requirement_flags(
    state: str,
    consolidation_required: bool,
    explicit_narrowing_required: bool,
    error: str,
) -> None:
    with pytest.raises(ValidationError, match=error):
        TexturePlanDecision(
            state=state,
            execution_allowed=state == "ready",
            consolidation_required=consolidation_required,
            explicit_narrowing_required=explicit_narrowing_required,
            reasons=() if state == "ready" else ("Not ready.",),
            recommended_actions=() if state == "ready" else ("Revise scope.",),
        )


def test_operator_override_cannot_exceed_hard_cap() -> None:
    with pytest.raises(ValidationError, match="less than or equal to 64"):
        TexturePlanRequest(
            source=TexturePlanSource(source_asset="scene.usd"),
            operator_override_cap=65,
        )


def test_plan_counts_and_unit_ids_must_match_embedded_membership() -> None:
    valid = _plan(2, execution_allowed=True)
    duplicate_units = (valid.selected_units[0], valid.selected_units[0])

    with pytest.raises(ValidationError, match="unique unit_id"):
        TexturePlan(
            generated_at=valid.generated_at,
            request=valid.request,
            limits=valid.limits,
            execution=valid.execution,
            counts=valid.counts,
            selected_units=duplicate_units,
            decision=valid.decision,
        )

    with pytest.raises(ValidationError, match="selected_unit_count"):
        TexturePlan(
            generated_at=valid.generated_at,
            request=valid.request,
            limits=valid.limits,
            execution=valid.execution,
            counts=valid.counts.model_copy(update={"selected_unit_count": 1}),
            selected_units=valid.selected_units,
            decision=valid.decision,
        )


def test_skipped_items_record_machine_and_human_reasons() -> None:
    plan = _plan(1, execution_allowed=True)
    skipped = TexturePlanSkippedItem(
        item_kind="material",
        canonical_id="/World/Looks/Unused",
        display_name="Unused",
        reason_code="not_effectively_bound",
        reason="No renderable prim or material-binding subset uses this material.",
    )

    updated = TexturePlan(
        generated_at=plan.generated_at,
        request=plan.request,
        limits=plan.limits,
        execution=plan.execution,
        counts=plan.counts.model_copy(update={"skipped_item_count": 1}),
        selected_units=plan.selected_units,
        skipped_items=(skipped,),
        decision=plan.decision,
    )

    assert updated.skipped_items[0].reason_code == "not_effectively_bound"


def test_optional_contract_strings_accept_explicit_none() -> None:
    source = TexturePlanSource(
        source_asset="scene.usd",
        upstream_assignment_artifact=None,
    )
    skipped = TexturePlanSkippedItem(
        item_kind="material",
        canonical_id="/World/Looks/Unused",
        display_name=None,
        reason_code="not_effectively_bound",
        reason="No renderable scene member uses this material.",
    )

    assert source.upstream_assignment_artifact is None
    assert skipped.display_name is None


def test_json_schema_exposes_shared_modes_and_required_plan_sections() -> None:
    schema = texture_plan_json_schema()

    assert schema["properties"]["schema_version"]["const"] == (
        TEXTURE_PLAN_SCHEMA_VERSION
    )
    assert {
        "request",
        "limits",
        "execution",
        "counts",
        "selected_units",
        "skipped_items",
        "decision",
    }.issubset(schema["properties"])
    assert schema["$defs"]["TextureDiscoveryMode"]["enum"] == [
        "effective_bound",
        "explicit",
        "all_authored",
    ]
    assert schema["$defs"]["TextureUnitMode"]["enum"] == [
        "per_material",
        "per_group",
        "per_prim",
    ]
