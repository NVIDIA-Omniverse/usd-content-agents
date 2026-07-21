# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused tests for issue #466 immutable bounded execution."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from time import sleep

import pytest
from PIL import Image
from pxr import Usd, UsdGeom, UsdShade

from texture_agent.execution import (
    BoundedTextureExecutor,
    CancellationToken,
    FileTextureExecutionCheckpointStore,
    TextureArtifactRef,
    TextureExecutionCancelled,
    TextureExecutionStatus,
    TextureExecutionTimedOut,
    TextureUnitExecutionContext,
    TextureUnitExecutionResult,
    bind_prim_texture_units_to_plan,
)
from texture_agent.execution.adapters import _runtime_identity
from texture_agent.functions.material_discovery import MaterialInfo, PrimTextureUnit
from texture_agent.planning import (
    TexturePlan,
    TexturePlanCounts,
    TexturePlanDecision,
    TexturePlanExecution,
    TexturePlanLimits,
    TexturePlanRequest,
    TexturePlanSource,
    TexturePlanUnit,
)
from texture_agent.tasks import apply_textures as apply_textures_task
from texture_agent.tasks.apply_textures import ApplyTexturesTask
from texture_agent.tasks.blend_textures import BlendedTextures
from texture_agent.tasks.execute_texture_plan import ExecuteTexturePlanTask
from texture_agent.tasks.plan_textures import require_executable_texture_plan


def _plan(tmp_path: Path, count: int = 3, *, approved: bool = True) -> TexturePlan:
    request = TexturePlanRequest(
        source=TexturePlanSource(source_asset=str(tmp_path / "scene.usd")),
        backend="simple_image_gen",
        max_concurrency=1,
        unit_timeout_seconds=30,
    )
    units = tuple(
        TexturePlanUnit.build(
            unit_mode="per_material",
            material_prim_paths=(f"/World/Looks/Material_{index}",),
            member_prim_paths=(f"/World/Mesh_{index}",),
            display_name="Duplicate display name",
            selection_reason_code="effectively_bound",
            selection_reason="Used by renderable geometry.",
            detail_policy="surface_only",
        )
        for index in range(count)
    )
    decision = (
        TexturePlanDecision(state="ready", execution_allowed=True)
        if approved
        else TexturePlanDecision(
            state="unsupported",
            execution_allowed=False,
            reasons=("Not approved.",),
            recommended_actions=("Revise the plan.",),
        )
    )
    return TexturePlan(
        generated_at=datetime(2026, 6, 29, tzinfo=UTC),
        request=request,
        limits=TexturePlanLimits.from_request(request),
        execution=TexturePlanExecution.from_request(request),
        counts=TexturePlanCounts(
            authored_material_count=count,
            renderable_prim_count=count,
            renderable_subset_count=0,
            effective_bound_material_count=count,
            selected_material_count=count,
            selected_unit_count=count,
            skipped_item_count=0,
            planned_generation_job_count=count,
        ),
        selected_units=units,
        decision=decision,
    )


def _artifact_result(unit_id: str, output_dir: Path, marker: str = "ok"):
    path = output_dir / f"{unit_id}-{marker}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(marker.encode("utf-8"))
    return TextureUnitExecutionResult(
        unit_id=unit_id,
        artifacts=(TextureArtifactRef(name="albedo", uri=str(path)),),
        metadata={"marker": marker},
    )


def test_executor_invokes_exact_approved_ids_in_plan_order(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    invoked: list[str] = []

    def run(unit, execution_context):
        invoked.append(unit.unit_id)
        assert execution_context.timeout_seconds == 30
        return _artifact_result(unit.unit_id, tmp_path)

    summary = BoundedTextureExecutor(
        plan=plan,
        checkpoint_store=FileTextureExecutionCheckpointStore(
            tmp_path / "checkpoint.json"
        ),
        unit_runner=run,
    ).execute()

    expected = [unit.unit_id for unit in plan.selected_units]
    assert invoked == expected
    assert list(summary.executed_unit_ids) == expected
    assert summary.accepted_unit_ids == tuple(expected)
    assert summary.status == TextureExecutionStatus.COMPLETED


def test_file_checkpoint_store_load_returns_none_when_absent(tmp_path: Path) -> None:
    assert FileTextureExecutionCheckpointStore(tmp_path / "missing.json").load() is None


def test_non_executable_plan_fails_before_runner_invocation(tmp_path: Path) -> None:
    invoked = False

    def run(unit, execution_context):
        nonlocal invoked
        invoked = True
        return _artifact_result(unit.unit_id, tmp_path)

    with pytest.raises(ValueError, match="not approved"):
        BoundedTextureExecutor(
            plan=_plan(tmp_path, approved=False),
            checkpoint_store=FileTextureExecutionCheckpointStore(
                tmp_path / "checkpoint.json"
            ),
            unit_runner=run,
        )
    assert invoked is False


def test_execution_context_raises_on_cancel_and_timeout() -> None:
    token = CancellationToken()
    token.cancel()
    cancelled = TextureUnitExecutionContext(
        unit_id="tu_00000000000000000000",
        attempt=1,
        timeout_seconds=30,
        cancellation_token=token,
        started_monotonic=time.monotonic(),
    )

    with pytest.raises(TextureExecutionCancelled):
        cancelled.raise_if_cancelled()

    timed_out = TextureUnitExecutionContext(
        unit_id="tu_00000000000000000000",
        attempt=1,
        timeout_seconds=0,
        cancellation_token=CancellationToken(),
        started_monotonic=time.monotonic(),
    )

    with pytest.raises(TextureExecutionTimedOut):
        timed_out.raise_if_timed_out()


def test_executor_fails_non_cooperative_runner_after_unit_timeout(
    tmp_path: Path,
) -> None:
    original = _plan(tmp_path, count=3)
    plan = TexturePlan(
        generated_at=original.generated_at,
        request=original.request.model_copy(
            update={"unit_timeout_seconds": 1, "max_concurrency": 2}
        ),
        limits=original.limits,
        execution=original.execution.model_copy(
            update={"unit_timeout_seconds": 1, "max_concurrency": 2}
        ),
        counts=original.counts,
        selected_units=original.selected_units,
        decision=original.decision,
    )
    started = time.monotonic()

    def run(unit, execution_context):
        if unit.unit_id == plan.selected_units[1].unit_id:
            return _artifact_result(unit.unit_id, tmp_path)
        sleep(2.0)
        return _artifact_result(unit.unit_id, tmp_path)

    summary = BoundedTextureExecutor(
        plan=plan,
        checkpoint_store=FileTextureExecutionCheckpointStore(
            tmp_path / "checkpoint.json"
        ),
        unit_runner=run,
    ).execute()

    assert time.monotonic() - started < 1.8
    assert summary.status == TextureExecutionStatus.PARTIAL
    assert summary.failed_unit_ids == (plan.selected_units[0].unit_id,)
    assert summary.cancelled_unit_ids == (plan.selected_units[2].unit_id,)


def test_executor_abandons_active_futures_on_external_cancel(
    tmp_path: Path,
) -> None:
    original = _plan(tmp_path, count=2)
    plan = TexturePlan(
        generated_at=original.generated_at,
        request=original.request.model_copy(update={"max_concurrency": 2}),
        limits=original.limits,
        execution=original.execution.model_copy(update={"max_concurrency": 2}),
        counts=original.counts,
        selected_units=original.selected_units,
        decision=original.decision,
    )
    lock = Lock()
    started = 0

    def run(unit, execution_context):
        nonlocal started
        with lock:
            started += 1
        sleep(1.0)
        return _artifact_result(unit.unit_id, tmp_path)

    def is_cancelled() -> bool:
        with lock:
            return started == 2

    summary = BoundedTextureExecutor(
        plan=plan,
        checkpoint_store=FileTextureExecutionCheckpointStore(
            tmp_path / "checkpoint.json"
        ),
        unit_runner=run,
        external_cancellation_check=is_cancelled,
    ).execute()

    assert summary.status == TextureExecutionStatus.CANCELLED
    assert summary.cancelled_unit_ids == tuple(
        unit.unit_id for unit in plan.selected_units
    )


def test_default_artifact_validator_accepts_remote_and_rejects_bad_digest(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path, count=1)
    local = tmp_path / "artifact.png"
    local.write_bytes(b"actual")
    calls = 0

    def run(unit, execution_context):
        nonlocal calls
        calls += 1
        if calls == 1:
            return TextureUnitExecutionResult(
                unit_id=unit.unit_id,
                artifacts=(
                    TextureArtifactRef(
                        name="remote",
                        uri="session-artifact:///cache/texture.png",
                    ),
                ),
            )
        return TextureUnitExecutionResult(
            unit_id=unit.unit_id,
            artifacts=(
                TextureArtifactRef(
                    name="local",
                    uri=str(local),
                    sha256="0" * 64,
                ),
            ),
        )

    accepted = BoundedTextureExecutor(
        plan=plan,
        checkpoint_store=FileTextureExecutionCheckpointStore(
            tmp_path / "remote-checkpoint.json"
        ),
        unit_runner=run,
    ).execute()

    assert accepted.status == TextureExecutionStatus.COMPLETED
    failed = BoundedTextureExecutor(
        plan=plan,
        checkpoint_store=FileTextureExecutionCheckpointStore(
            tmp_path / "digest-checkpoint.json"
        ),
        unit_runner=run,
    ).execute()
    assert failed.status == TextureExecutionStatus.FAILED


def test_executor_accepts_mapping_results_and_reports_partial_selection(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path, count=2)
    target = plan.selected_units[0].unit_id

    def run(unit, execution_context):
        path = tmp_path / f"{unit.unit_id}.png"
        path.write_bytes(b"png")
        return {"albedo": str(path)}

    summary = BoundedTextureExecutor(
        plan=plan,
        checkpoint_store=FileTextureExecutionCheckpointStore(
            tmp_path / "checkpoint.json"
        ),
        unit_runner=run,
    ).execute(regenerate_unit_ids=(target,))

    assert summary.status == TextureExecutionStatus.PARTIAL
    assert summary.accepted_unit_ids == (target,)
    assert summary.remaining_unit_ids == (plan.selected_units[1].unit_id,)


def test_executor_promotes_external_cancellation_check_to_token(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path, count=1)
    token = CancellationToken()
    executor = BoundedTextureExecutor(
        plan=plan,
        checkpoint_store=FileTextureExecutionCheckpointStore(
            tmp_path / "checkpoint.json"
        ),
        unit_runner=lambda unit, _: _artifact_result(unit.unit_id, tmp_path),
        cancellation_token=token,
        external_cancellation_check=lambda: True,
    )

    assert executor._is_cancelled() is True
    assert token.is_cancelled() is True


def test_cancel_then_resume_does_not_repeat_completed_units(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    checkpoint_store = FileTextureExecutionCheckpointStore(tmp_path / "checkpoint.json")
    token = CancellationToken()
    first_invocation: list[str] = []

    def cancel_after_first(checkpoint):
        completed = [
            record
            for record in checkpoint.records
            if record.accepted_result is not None
        ]
        if completed:
            token.cancel()

    first = BoundedTextureExecutor(
        plan=plan,
        checkpoint_store=checkpoint_store,
        unit_runner=lambda unit, _: (
            first_invocation.append(unit.unit_id)
            or _artifact_result(unit.unit_id, tmp_path)
        ),
        cancellation_token=token,
        progress_callback=cancel_after_first,
    ).execute()

    assert first.status == TextureExecutionStatus.CANCELLED
    assert first_invocation == [plan.selected_units[0].unit_id]

    resumed_invocation: list[str] = []
    resumed = BoundedTextureExecutor(
        plan=plan,
        checkpoint_store=checkpoint_store,
        unit_runner=lambda unit, _: (
            resumed_invocation.append(unit.unit_id)
            or _artifact_result(unit.unit_id, tmp_path, marker="resumed")
        ),
    ).execute(resume=True)

    assert resumed.cache_hit_unit_ids == (plan.selected_units[0].unit_id,)
    assert resumed_invocation == [unit.unit_id for unit in plan.selected_units[1:]]
    assert resumed.remaining_unit_ids == ()
    assert resumed.status == TextureExecutionStatus.COMPLETED


def test_targeted_regeneration_only_replaces_requested_unit(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    store = FileTextureExecutionCheckpointStore(tmp_path / "checkpoint.json")
    initial = BoundedTextureExecutor(
        plan=plan,
        checkpoint_store=store,
        unit_runner=lambda unit, _: _artifact_result(unit.unit_id, tmp_path, "v1"),
    ).execute()
    before = {record.unit_id: record for record in initial.records}

    target = plan.selected_units[1].unit_id
    invoked: list[str] = []
    regenerated = BoundedTextureExecutor(
        plan=plan,
        checkpoint_store=store,
        unit_runner=lambda unit, _: (
            invoked.append(unit.unit_id)
            or _artifact_result(unit.unit_id, tmp_path, "v2")
        ),
    ).execute(regenerate_unit_ids=(target,))
    after = {record.unit_id: record for record in regenerated.records}

    assert invoked == [target]
    assert after[target].accepted_result != before[target].accepted_result
    for untouched in (plan.selected_units[0].unit_id, plan.selected_units[2].unit_id):
        assert after[untouched] == before[untouched]


def test_failed_targeted_regeneration_clears_previous_acceptance(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    store = FileTextureExecutionCheckpointStore(tmp_path / "checkpoint.json")
    BoundedTextureExecutor(
        plan=plan,
        checkpoint_store=store,
        unit_runner=lambda unit, _: _artifact_result(unit.unit_id, tmp_path),
    ).execute()
    target = plan.selected_units[0].unit_id

    failed = BoundedTextureExecutor(
        plan=plan,
        checkpoint_store=store,
        unit_runner=lambda unit, _: (_ for _ in ()).throw(RuntimeError("retry failed")),
    ).execute(regenerate_unit_ids=(target,))

    assert failed.failed_unit_ids == (target,)
    assert failed.records[0].accepted_result is None
    assert target not in failed.accepted_unit_ids
    assert target in failed.remaining_unit_ids
    assert failed.status == TextureExecutionStatus.FAILED

    invoked: list[str] = []
    resumed = BoundedTextureExecutor(
        plan=plan,
        checkpoint_store=store,
        unit_runner=lambda unit, _: (
            invoked.append(unit.unit_id)
            or _artifact_result(unit.unit_id, tmp_path, marker="recovered")
        ),
    ).execute(resume=True)
    assert invoked == [target]
    assert resumed.status == TextureExecutionStatus.COMPLETED
    assert resumed.failed_unit_ids == ()
    assert resumed.records[0].last_error is None


def test_regeneration_rejects_id_outside_immutable_plan(tmp_path: Path) -> None:
    executor = BoundedTextureExecutor(
        plan=_plan(tmp_path),
        checkpoint_store=FileTextureExecutionCheckpointStore(
            tmp_path / "checkpoint.json"
        ),
        unit_runner=lambda unit, _: _artifact_result(unit.unit_id, tmp_path),
    )

    with pytest.raises(ValueError, match="outside the approved plan"):
        executor.execute(regenerate_unit_ids=("tu_00000000000000000000",))


def test_resume_reexecutes_unit_when_cached_artifact_is_missing(tmp_path: Path) -> None:
    plan = _plan(tmp_path, count=1)
    store = FileTextureExecutionCheckpointStore(tmp_path / "checkpoint.json")
    initial = BoundedTextureExecutor(
        plan=plan,
        checkpoint_store=store,
        unit_runner=lambda unit, _: _artifact_result(unit.unit_id, tmp_path),
    ).execute()
    accepted = initial.records[0].accepted_result
    assert accepted is not None
    cached_path = Path(accepted.artifacts[0].uri)
    cached_path.unlink()
    invoked: list[str] = []

    resumed = BoundedTextureExecutor(
        plan=plan,
        checkpoint_store=store,
        unit_runner=lambda unit, _: (
            invoked.append(unit.unit_id)
            or _artifact_result(unit.unit_id, tmp_path, "recovered")
        ),
    ).execute(resume=True)

    assert invoked == [plan.selected_units[0].unit_id]
    assert resumed.cache_hit_unit_ids == ()
    assert resumed.records[0].attempts == 2


def test_resume_rejects_checkpoint_from_different_plan(tmp_path: Path) -> None:
    store = FileTextureExecutionCheckpointStore(tmp_path / "checkpoint.json")
    first_plan = _plan(tmp_path, count=1)
    BoundedTextureExecutor(
        plan=first_plan,
        checkpoint_store=store,
        unit_runner=lambda unit, _: _artifact_result(unit.unit_id, tmp_path),
    ).execute()

    with pytest.raises(ValueError, match="different immutable Texture Plan"):
        BoundedTextureExecutor(
            plan=_plan(tmp_path, count=2),
            checkpoint_store=store,
            unit_runner=lambda unit, _: _artifact_result(unit.unit_id, tmp_path),
        ).execute(resume=True)


def test_executor_never_exceeds_plan_concurrency(tmp_path: Path) -> None:
    original = _plan(tmp_path, count=6)
    plan = TexturePlan(
        generated_at=original.generated_at,
        request=original.request.model_copy(update={"max_concurrency": 2}),
        limits=original.limits,
        execution=original.execution.model_copy(update={"max_concurrency": 2}),
        counts=original.counts,
        selected_units=original.selected_units,
        decision=original.decision,
    )
    lock = Lock()
    active = 0
    maximum_active = 0

    def run(unit, execution_context):
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        sleep(0.01)
        with lock:
            active -= 1
        return _artifact_result(unit.unit_id, tmp_path)

    BoundedTextureExecutor(
        plan=plan,
        checkpoint_store=FileTextureExecutionCheckpointStore(
            tmp_path / "checkpoint.json"
        ),
        unit_runner=run,
    ).execute()

    assert maximum_active == 2


def _runtime_unit(index: int, *, key: str | None = None) -> PrimTextureUnit:
    return PrimTextureUnit(
        prim_path="",
        material_info=MaterialInfo(
            prim_path=f"/World/Looks/Material_{index}",
            name="Duplicate display name",
            bound_prim_paths=[f"/World/Mesh_{index}"],
        ),
        key=key or f"legacy-{index}",
        prompt="PCB traces on painted metal" if index == 0 else "painted metal",
        opacity=0.85,
    )


def test_legacy_adapter_filters_extras_and_rekeys_duplicate_names(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path, count=2)
    bound = bind_prim_texture_units_to_plan(
        plan,
        [_runtime_unit(99), _runtime_unit(1), _runtime_unit(0)],
    )

    assert [unit.key for unit in bound] == [
        unit.unit_id for unit in plan.selected_units
    ]
    assert all(unit.detail_policy == "surface_only" for unit in bound)
    assert "Avoid traces, vias, pads" in bound[0].prompt


def test_legacy_adapter_per_prim_identity_rejects_mismatched_shapes(
    tmp_path: Path,
) -> None:
    plan_unit = TexturePlanUnit.build(
        unit_mode="per_prim",
        material_prim_paths=("/World/Looks/Paint",),
        member_subset_paths=("/World/Mesh/Subset",),
        display_name="Paint subset",
        selection_reason_code="effectively_bound",
        selection_reason="Used by renderable geometry.",
        detail_policy="surface_only",
    )
    runtime = PrimTextureUnit(
        prim_path="/World/Mesh",
        material_info=MaterialInfo(prim_path="/World/Looks/Paint", name="Paint"),
        key="legacy",
        prompt="paint",
        opacity=1.0,
    )

    matching_plan_unit = TexturePlanUnit.build(
        unit_mode="per_prim",
        material_prim_paths=("/World/Looks/Paint",),
        member_prim_paths=("/World/Mesh",),
        display_name="Paint mesh",
        selection_reason_code="effectively_bound",
        selection_reason="Used by renderable geometry.",
        detail_policy="surface_only",
    )
    assert _runtime_identity(matching_plan_unit, runtime) is True
    group_plan_unit = TexturePlanUnit.build(
        unit_mode="per_group",
        material_prim_paths=("/World/Looks/Paint",),
        member_prim_paths=("/World/Mesh",),
        group_key="/World/Looks/Paint",
        display_name="Paint group",
        selection_reason_code="effectively_bound",
        selection_reason="Used by renderable geometry.",
        detail_policy="surface_only",
    )
    assert _runtime_identity(group_plan_unit, runtime) is False
    assert _runtime_identity(plan_unit, runtime) is False
    mismatched_material = PrimTextureUnit(
        prim_path=runtime.prim_path,
        material_info=MaterialInfo(
            prim_path="/World/Looks/Other",
            name="Other",
        ),
        key=runtime.key,
        prompt=runtime.prompt,
        opacity=runtime.opacity,
    )
    assert _runtime_identity(plan_unit, mismatched_material) is False


def test_legacy_adapter_accepts_per_material_subset_evidence(
    tmp_path: Path,
) -> None:
    plan_unit = TexturePlanUnit.build(
        unit_mode="per_material",
        material_prim_paths=("/World/Looks/Paint",),
        member_subset_paths=("/World/Mesh/Subset",),
        display_name="Paint subset",
        selection_reason_code="effectively_bound",
        selection_reason="Used by renderable geometry.",
        detail_policy="surface_only",
    )
    original = _plan(tmp_path, count=1)
    plan = TexturePlan(
        generated_at=original.generated_at,
        request=original.request,
        limits=original.limits,
        execution=original.execution,
        counts=original.counts,
        selected_units=(plan_unit,),
        decision=original.decision,
    )
    bound = bind_prim_texture_units_to_plan(
        plan,
        [
            PrimTextureUnit(
                prim_path="",
                material_info=MaterialInfo(
                    prim_path="/World/Looks/Paint", name="Paint"
                ),
                key="legacy",
                prompt="paint",
                opacity=1.0,
            )
        ],
    )

    assert bound[0].key == plan_unit.unit_id


def test_execute_task_publishes_accepted_results_by_stable_unit_id(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path, count=2)
    lock = Lock()
    invoked: list[str] = []

    def unit_runner(plan_unit, runtime_unit, base_context, execution_context):
        with lock:
            invoked.append(runtime_unit.key)
        artifacts = []
        for channel in ("albedo", "normal", "orm"):
            path = tmp_path / f"{plan_unit.unit_id}_{channel}.png"
            path.write_bytes(channel.encode("utf-8"))
            artifacts.append(TextureArtifactRef(name=channel, uri=str(path)))
        return TextureUnitExecutionResult(
            unit_id=plan_unit.unit_id,
            artifacts=tuple(artifacts),
        )

    context = ExecuteTexturePlanTask(unit_runner=unit_runner).run(
        {
            "texture_plan": plan,
            "prim_texture_units": [_runtime_unit(1), _runtime_unit(0)],
            "working_dir": str(tmp_path),
            "texture_config": {},
        }
    )

    expected = [unit.unit_id for unit in plan.selected_units]
    assert invoked == expected
    assert list(context["generated_textures"]) == expected
    assert context["texture_execution_status"] == "completed"
    assert context["texture_execution_remaining_unit_ids"] == []


def test_execute_task_loads_plan_from_path_and_accepts_planning_regenerate_ids(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path, count=1)
    plan_path = tmp_path / "texture_plan.json"
    plan_path.write_text(plan.model_dump_json(), encoding="utf-8")

    def unit_runner(plan_unit, runtime_unit, base_context, execution_context):
        return _artifact_result(plan_unit.unit_id, tmp_path)

    context = ExecuteTexturePlanTask(unit_runner=unit_runner).run(
        {
            "texture_plan_path": str(plan_path),
            "prim_texture_units": [_runtime_unit(0)],
            "working_dir": str(tmp_path),
            "planning_config": {
                "regenerate_unit_ids": [plan.selected_units[0].unit_id]
            },
        }
    )

    assert context["texture_execution_status"] == "completed"


def test_execute_task_records_fallback_error_when_unit_observation_is_empty(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path, count=1)
    context = {
        "texture_plan": plan,
        "prim_texture_units": [_runtime_unit(0)],
        "working_dir": str(tmp_path),
    }

    def unit_runner(plan_unit, runtime_unit, base_context, execution_context):
        raise RuntimeError("backend failed before diagnostics")

    with pytest.raises(RuntimeError, match="texture generation requests failed"):
        ExecuteTexturePlanTask(unit_runner=unit_runner).run(context)

    assert context["generate_textures_errors"] == [
        {
            "material": plan.selected_units[0].unit_id,
            "type": "TextureUnitExecutionError",
            "status": None,
            "message": "RuntimeError: texture unit execution failed",
        }
    ]


def test_execute_task_honors_failure_threshold_for_partial_failures(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path, count=2)
    context = {
        "texture_plan": plan,
        "prim_texture_units": [_runtime_unit(0), _runtime_unit(1)],
        "working_dir": str(tmp_path),
        "texture_config": {"failure_threshold": 0.5},
    }

    def unit_runner(plan_unit, runtime_unit, base_context, execution_context):
        if plan_unit.unit_id == plan.selected_units[0].unit_id:
            artifacts = []
            for channel in ("albedo", "normal", "orm"):
                path = tmp_path / f"{plan_unit.unit_id}_{channel}.png"
                path.write_bytes(channel.encode("utf-8"))
                artifacts.append(TextureArtifactRef(name=channel, uri=str(path)))
            return TextureUnitExecutionResult(
                unit_id=plan_unit.unit_id,
                artifacts=tuple(artifacts),
            )
        raise RuntimeError("backend failed")

    with pytest.raises(RuntimeError, match="1/2 texture generation requests"):
        ExecuteTexturePlanTask(unit_runner=unit_runner).run(context)

    assert context["texture_execution_status"] == "partial"


def test_execute_task_isolates_unit_runner_context_mutations(tmp_path: Path) -> None:
    plan = _plan(tmp_path, count=2)
    context = {
        "texture_plan": plan,
        "prim_texture_units": [_runtime_unit(0), _runtime_unit(1)],
        "working_dir": str(tmp_path),
    }

    def unit_runner(plan_unit, runtime_unit, base_context, execution_context):
        del runtime_unit, execution_context
        assert "mutated_by_unit" not in base_context
        base_context["mutated_by_unit"] = plan_unit.unit_id
        return _artifact_result(plan_unit.unit_id, tmp_path)

    result = ExecuteTexturePlanTask(unit_runner=unit_runner).run(context)

    assert result["texture_execution_accepted_unit_ids"] == [
        unit.unit_id for unit in plan.selected_units
    ]
    assert "mutated_by_unit" not in context


def test_execute_task_requires_plan_payload_or_path() -> None:
    with pytest.raises(ValueError, match="requires texture_plan"):
        ExecuteTexturePlanTask._load_plan({})


def test_execute_task_empty_regenerate_ids_when_explicitly_none() -> None:
    assert (
        ExecuteTexturePlanTask._regenerate_ids({"texture_regenerate_unit_ids": None})
        == ()
    )


def test_require_executable_texture_plan_loads_default_working_dir_artifact(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path, count=1)
    (tmp_path / "texture_plan.json").write_text(
        plan.model_dump_json(),
        encoding="utf-8",
    )
    context = {"working_dir": str(tmp_path)}

    loaded = require_executable_texture_plan(context)

    assert loaded == plan
    assert context["texture_plan"] == plan
    assert context["texture_plan_path"] == str(tmp_path / "texture_plan.json")


def test_apply_keeps_duplicate_material_names_separated_by_canonical_path(
    tmp_path: Path,
) -> None:
    input_usd = tmp_path / "scene.usda"
    stage = Usd.Stage.CreateNew(str(input_usd))
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Scope.Define(stage, "/World/A")
    UsdGeom.Scope.Define(stage, "/World/B")
    first_material_path = "/World/A/Paint"
    second_material_path = "/World/B/Paint"
    UsdShade.Material.Define(stage, first_material_path)
    UsdShade.Material.Define(stage, second_material_path)
    stage.GetRootLayer().Save()

    unit_ids = [
        "tu_00000000000000000000",
        "tu_11111111111111111111",
    ]
    units = [
        PrimTextureUnit(
            prim_path="",
            material_info=MaterialInfo(prim_path=material_path, name="Paint"),
            key=unit_id,
            prompt="paint",
            opacity=1.0,
        )
        for material_path, unit_id in zip(
            (first_material_path, second_material_path),
            unit_ids,
            strict=True,
        )
    ]
    blended = {}
    for index, unit_id in enumerate(unit_ids):
        paths = {}
        for channel, color in (
            ("albedo", (index * 255, 0, 0)),
            ("normal", (128, 128, 255)),
            ("orm", (255, 128, 0)),
        ):
            path = tmp_path / "textures" / f"{unit_id}_{channel}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (2, 2), color).save(path)
            paths[channel] = str(path)
        blended[unit_id] = BlendedTextures(**paths)

    context = ApplyTexturesTask().run(
        {
            "usd_path": str(input_usd),
            "blended_textures": blended,
            "prim_texture_units": units,
            "working_dir": str(tmp_path),
        }
    )

    assert context["apply_textures_stats"]["applied_count"] == 2
    output = Usd.Stage.Open(context["output_usd_paths"][0])
    for material_path, unit_id in zip(
        (first_material_path, second_material_path),
        unit_ids,
        strict=True,
    ):
        value = (
            output.GetPrimAtPath(material_path)
            .GetAttribute("inputs:base_color_texture_file")
            .Get()
        )
        assert unit_id in value.path


def test_apply_deinstances_instance_proxy_material_before_authoring(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model.usda"
    model_stage = Usd.Stage.CreateNew(str(model_path))
    model = UsdGeom.Xform.Define(model_stage, "/Model")
    model_stage.SetDefaultPrim(model.GetPrim())
    cube = UsdGeom.Cube.Define(model_stage, "/Model/Cube")
    material = UsdShade.Material.Define(model_stage, "/Model/Looks/Shared")
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)
    model_stage.GetRootLayer().Save()

    input_usd = tmp_path / "scene.usda"
    stage = Usd.Stage.CreateNew(str(input_usd))
    instance = UsdGeom.Xform.Define(stage, "/World/InstanceA").GetPrim()
    instance.GetReferences().AddReference(str(model_path), "/Model")
    instance.SetInstanceable(True)
    stage.GetRootLayer().Save()

    unit_id = "tu_00000000000000000000"
    paths = {}
    for channel, color in (
        ("albedo", (255, 0, 0)),
        ("normal", (128, 128, 255)),
        ("orm", (255, 128, 0)),
    ):
        path = tmp_path / "textures" / f"{unit_id}_{channel}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (2, 2), color).save(path)
        paths[channel] = str(path)

    context = ApplyTexturesTask().run(
        {
            "usd_path": str(input_usd),
            "blended_textures": {unit_id: BlendedTextures(**paths)},
            "prim_texture_units": [
                PrimTextureUnit(
                    prim_path="",
                    material_info=MaterialInfo(
                        prim_path="/World/InstanceA/Looks/Shared",
                        name="Shared",
                    ),
                    key=unit_id,
                    prompt="paint",
                    opacity=1.0,
                )
            ],
            "working_dir": str(tmp_path),
        }
    )

    output = Usd.Stage.Open(context["output_usd_paths"][0])
    instance_prim = output.GetPrimAtPath("/World/InstanceA")
    assert instance_prim.IsValid()
    assert not instance_prim.IsInstanceable()
    material_prim = output.GetPrimAtPath("/World/InstanceA/Looks/Shared")
    assert material_prim.IsValid()
    assert not material_prim.IsInstanceProxy()
    assert (
        material_prim.GetAttribute("inputs:base_color_texture_file")
        .Get()
        .path.endswith(f"{unit_id}_albedo.png")
    )


def test_apply_skips_material_when_deinstance_cannot_make_it_editable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _UneditablePrim:
        def IsValid(self) -> bool:
            return True

        def IsInstanceProxy(self) -> bool:
            return True

        def IsPrototype(self) -> bool:
            return False

        def IsInPrototype(self) -> bool:
            return False

    monkeypatch.setattr(
        apply_textures_task,
        "_editable_prim_for_path",
        lambda stage, mat_path: _UneditablePrim(),
    )

    assert apply_textures_task._apply_pbr_textures(
        Usd.Stage.CreateInMemory(),
        "/World/Looks/Shared",
        BlendedTextures(albedo="", normal="", orm=""),
        tmp_path,
        "tu_00000000000000000000",
        str(tmp_path / "scene.usda"),
        tmp_path / "output.usda",
    ) == (0, [], [], [])
