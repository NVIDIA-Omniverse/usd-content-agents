# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared interactive and batch orchestration for durable texture workflows."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from filelock import FileLock

from .client import (
    TextureAgentServiceCancellationRequested,
    TexturePlannerExecutorClient,
)
from .decision import (
    TextureDecisionPatch,
    TextureStepObservation,
    build_texture_step_observation,
    record_texture_decision_patch,
    texture_decision_ledger_path,
    validate_texture_decision_patch,
    verify_texture_resume_decision_state,
)
from .embedded_decision import (
    TextureEmbeddedDecisionPatch,
    TextureEmbeddedStepObservation,
)
from .finalizer import (
    CanonicalTextureWorkflowFinalizer,
    TextureWorkflowFinalizer,
    write_texture_planning_artifacts,
)
from .models import (
    TextureExecutionResult,
    TextureFinalizationResult,
    TextureFinalizationStatus,
    TextureFinalizerInput,
    TexturePlanDocument,
    TextureUnitArtifact,
    TextureValidationResult,
    TextureWorkflowMode,
    TextureWorkflowPhase,
    TextureWorkflowProgress,
    TextureWorkflowRequest,
)
from .runtime import (
    CancellationCheck,
    TextureWorkflowAction,
    TextureWorkflowCheckpoint,
    TextureWorkflowCheckpointStore,
    TextureWorkflowRuntimeError,
    collect_artifact_digests,
    collect_output_asset_digest,
    collect_validation_evidence_digests,
    texture_plan_digest,
    texture_request_digest,
    texture_source_identity_digest,
    validate_resume_identity,
    verify_accepted_unit_material_state,
    verify_artifact_digests,
    verify_checkpoint_identity,
    verify_output_asset_digest,
    verify_validation_evidence_digests,
)
from .scene_validation import TextureSceneValidator
from .scope_validation import texture_unit_material_state_digests

ProgressCallback = Callable[[TextureWorkflowProgress], None]


class _TextureWorkflowStepPaused(RuntimeError):
    """Internal control signal emitted only after a durable transition."""

    def __init__(self, checkpoint: TextureWorkflowCheckpoint) -> None:
        super().__init__(f"Texture workflow paused before {checkpoint.next_action}")
        self.checkpoint = checkpoint


def _require_execution_scope(
    execution: TextureExecutionResult,
    requested_unit_ids: tuple[str, ...],
) -> None:
    if execution.requested_unit_ids != requested_unit_ids:
        raise RuntimeError(
            "Texture executor response scope differs from the requested unit IDs."
        )
    artifact_ids = tuple(artifact.unit_id for artifact in execution.unit_artifacts)
    if len(artifact_ids) != len(set(artifact_ids)):
        raise RuntimeError("Texture executor returned duplicate unit artifacts.")
    if set(artifact_ids) != set(requested_unit_ids):
        raise RuntimeError(
            "Texture executor artifacts differ from the requested unit IDs."
        )


def _require_validation_scope(
    validation: TextureValidationResult,
    requested_unit_ids: tuple[str, ...],
    *,
    output_asset_path: str,
    iteration: int,
) -> None:
    if validation.evaluated_unit_ids != requested_unit_ids:
        raise RuntimeError(
            "usd-cli validation response scope differs from the requested unit IDs."
        )
    failed = validation.failed_unit_ids
    if len(failed) != len(set(failed)):
        raise RuntimeError("usd-cli validation returned duplicate failed unit IDs.")
    unknown_failed = sorted(set(failed) - set(requested_unit_ids))
    if unknown_failed:
        raise RuntimeError(
            "usd-cli validation failed unit IDs outside the requested scope: "
            + ", ".join(unknown_failed)
        )
    if validation.iteration != iteration:
        raise RuntimeError(
            "usd-cli validation response iteration differs from the requested "
            "iteration."
        )
    if (
        Path(validation.output_asset_path).resolve()
        != Path(output_asset_path).resolve()
    ):
        raise RuntimeError(
            "usd-cli validation response asset differs from the candidate output asset."
        )


def _ordered_partition(
    selected_unit_ids: tuple[str, ...],
    remaining_unit_ids: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    remaining = set(remaining_unit_ids)
    accepted = tuple(
        unit_id for unit_id in selected_unit_ids if unit_id not in remaining
    )
    ordered_remaining = tuple(
        unit_id for unit_id in selected_unit_ids if unit_id in remaining
    )
    return accepted, ordered_remaining


def _plan_unit_paths(
    plan: TexturePlanDocument,
    field_name: str,
) -> tuple[str, ...]:
    paths: list[str] = []
    for unit in plan.selected_units:
        raw_paths = getattr(unit, field_name, None)
        if raw_paths is None and unit.model_extra:
            raw_paths = unit.model_extra.get(field_name)
        if raw_paths is None:
            continue
        if not isinstance(raw_paths, list | tuple):
            raise RuntimeError(f"Texture plan {field_name} must be an array.")
        paths.extend(str(path) for path in raw_paths)
    return tuple(dict.fromkeys(paths))


def _require_plan_request_scope(
    request: TextureWorkflowRequest,
    plan: TexturePlanDocument,
    *,
    exact: bool = False,
) -> None:
    def explicit_paths(field_name: str) -> tuple[str, ...]:
        raw_paths = request.metadata.get(field_name) or ()
        if not isinstance(raw_paths, list | tuple):
            raise RuntimeError(
                f"Texture request metadata.{field_name} must be an array."
            )
        return tuple(str(path) for path in raw_paths)

    if exact and request.metadata.get("unit_mode") == "per_material":
        unit_materials = tuple(
            tuple(
                str(path)
                for path in (
                    getattr(unit, "material_prim_paths", None)
                    or (unit.model_extra or {}).get("material_prim_paths")
                    or ()
                )
            )
            for unit in plan.selected_units
        )
        if any(len(paths) != 1 for paths in unit_materials) or len(
            {paths[0] for paths in unit_materials if paths}
        ) != len(unit_materials):
            raise RuntimeError(
                "Texture per-material proposal must retain one unique material "
                "target per unit"
            )

    explicit_material_paths = explicit_paths("explicit_material_paths")
    if explicit_material_paths:
        planned_material_paths = _plan_unit_paths(plan, "material_prim_paths")
        missing_materials = sorted(
            set(explicit_material_paths) - set(planned_material_paths)
        )
        unexpected_materials = sorted(
            set(planned_material_paths) - set(explicit_material_paths)
        )
        if (
            (exact and missing_materials)
            or unexpected_materials
            or (not planned_material_paths)
        ):
            if not exact:
                raise RuntimeError(
                    "Texture plan expands beyond explicit material scope: "
                    + ", ".join(unexpected_materials or ("<missing material paths>",))
                )
            raise RuntimeError(
                "Texture plan must exactly cover explicit material scope; missing: "
                + ", ".join(missing_materials or ("<none>",))
                + "; unexpected: "
                + ", ".join(unexpected_materials or ("<none>",))
            )

    explicit_prim_paths = explicit_paths("explicit_prim_paths")
    if explicit_prim_paths:
        planned_member_paths = (
            *_plan_unit_paths(plan, "member_prim_paths"),
            *_plan_unit_paths(plan, "member_subset_paths"),
        )
        unexpected_members = sorted(
            path
            for path in planned_member_paths
            if not any(
                path == root or path.startswith(f"{root}/")
                for root in explicit_prim_paths
            )
        )
        missing_roots = sorted(
            root
            for root in explicit_prim_paths
            if not any(
                path == root or path.startswith(f"{root}/")
                for path in planned_member_paths
            )
        )
        if not planned_member_paths or unexpected_members or (exact and missing_roots):
            if not exact:
                raise RuntimeError(
                    "Texture plan expands beyond explicit prim scope: "
                    + ", ".join(unexpected_members or ("<missing member paths>",))
                )
            raise RuntimeError(
                "Texture plan must cover every explicit prim root without expansion; "
                "missing roots: "
                + ", ".join(missing_roots or ("<none>",))
                + "; unexpected members: "
                + ", ".join(unexpected_members or ("<none>",))
            )


def _export_client_resume_state(
    client: TexturePlannerExecutorClient,
    plan: TexturePlanDocument,
) -> dict[str, Any]:
    exporter = getattr(client, "export_resume_state", None)
    if not callable(exporter):
        return {}
    state = exporter(plan)
    if not isinstance(state, dict):
        raise TextureWorkflowRuntimeError(
            "Texture client resume state must be a JSON object"
        )
    return state


def _restore_client_resume_state(
    client: TexturePlannerExecutorClient,
    plan: TexturePlanDocument,
    state: Mapping[str, Any],
) -> None:
    if not state:
        return
    restorer = getattr(client, "restore_resume_state", None)
    if not callable(restorer):
        raise TextureWorkflowRuntimeError(
            "Texture client cannot restore the checkpointed adapter state"
        )
    restorer(plan, state)


def _execute_client_units(
    client: TexturePlannerExecutorClient,
    plan: TexturePlanDocument,
    unit_ids: tuple[str, ...],
    *,
    output_dir: Path,
    preserved_artifacts: Mapping[str, TextureUnitArtifact],
    persist_resume_state: Callable[[], None],
    cancellation_check: CancellationCheck | None,
) -> TextureExecutionResult:
    resumable_executor = getattr(client, "execute_resumable", None)
    if callable(resumable_executor):
        resumable_kwargs: dict[str, Any] = {
            "output_dir": output_dir,
            "preserved_artifacts": preserved_artifacts,
            "persist_resume_state": persist_resume_state,
        }
        if (
            cancellation_check is not None
            and getattr(client, "supports_resumable_cancellation", False) is True
        ):
            resumable_kwargs["cancellation_check"] = cancellation_check
        result = resumable_executor(plan, unit_ids, **resumable_kwargs)
        if not isinstance(result, TextureExecutionResult):
            raise TextureWorkflowRuntimeError(
                "Texture resumable executor must return TextureExecutionResult"
            )
        return result
    return client.execute(
        plan,
        unit_ids,
        output_dir=output_dir,
        preserved_artifacts=preserved_artifacts,
    )


def run_texture_workflow(
    request: TextureWorkflowRequest,
    *,
    mode: TextureWorkflowMode,
    client: TexturePlannerExecutorClient,
    validator: TextureSceneValidator,
    finalizer: TextureWorkflowFinalizer | None = None,
    progress_callback: ProgressCallback | None = None,
    resume: bool = False,
    cancellation_check: CancellationCheck | None = None,
    checkpoint_store: TextureWorkflowCheckpointStore | None = None,
    decision_patch: TextureDecisionPatch | None = None,
    pause_after_transition: bool = False,
    workflow_lock: FileLock | None = None,
    decision_patch_recovered: bool = False,
) -> TextureFinalizationResult:
    """Compatibility controller for the durable Texture state machine.

    New skill-routed launchers use :func:`run_texture_workflow_step` so a
    reasoning loop reviews evidence and supplies a typed decision before each
    mutation, validation, refinement, or publication step. Keeping the
    uninterrupted controller here preserves the explicit legacy/baseline API.
    """

    output_dir = request.output_dir.resolve()
    canonical_lock_path = output_dir / ".texture_workflow.lock"
    if pause_after_transition and (
        workflow_lock is None
        or not workflow_lock.is_locked
        or Path(workflow_lock.lock_file) != canonical_lock_path
    ):
        raise TextureWorkflowRuntimeError(
            "Focused Texture workflow transitions require the active canonical "
            "output lock"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    store = checkpoint_store or TextureWorkflowCheckpointStore(output_dir)
    finalizer_impl = finalizer or CanonicalTextureWorkflowFinalizer()
    checkpoint: TextureWorkflowCheckpoint
    client_resume_restored = not resume

    if resume:
        if not store.exists():
            raise TextureWorkflowRuntimeError(
                f"No Texture workflow checkpoint exists at {store.path}"
            )
        checkpoint = store.load()
        validate_resume_identity(checkpoint, request=request, mode=mode)
    else:
        if store.exists():
            raise TextureWorkflowRuntimeError(
                "Texture workflow checkpoint already exists; pass resume=True "
                "or choose a new output directory"
            )
        request_identity_digest = texture_request_digest(request)
        preplanning_source_identity_digest = texture_source_identity_digest(request)

        def verify_precheckpoint_request_identity(*, boundary: str) -> None:
            if request_identity_digest != texture_request_digest(request):
                raise TextureWorkflowRuntimeError(
                    f"Texture workflow request changed {boundary}; "
                    "start a new output directory"
                )
            if request.output_dir.resolve() != output_dir:
                raise TextureWorkflowRuntimeError(
                    f"Texture workflow output directory changed {boundary}; "
                    "start a new output directory"
                )

        raw_plan = client.plan(request)
        plan = TexturePlanDocument.model_validate(
            raw_plan.model_dump(mode="python", round_trip=True)
        )
        verify_precheckpoint_request_identity(boundary="during planning")
        if preplanning_source_identity_digest != texture_source_identity_digest(
            request
        ):
            raise TextureWorkflowRuntimeError(
                "Texture workflow source bytes changed during planning; "
                "start a new output directory"
            )
        source_identity_digest = texture_source_identity_digest(request, plan=plan)
        planned_plan_digest = texture_plan_digest(plan)
        client_resume_state = _export_client_resume_state(client, plan)
        if planned_plan_digest != texture_plan_digest(plan):
            raise TextureWorkflowRuntimeError(
                "Texture workflow plan changed before checkpointing"
            )
        verify_precheckpoint_request_identity(
            boundary="before writing planning artifacts"
        )
        write_texture_planning_artifacts(request, plan)
        _require_plan_request_scope(request, plan)
        if not plan.decision.execution_allowed:
            raise RuntimeError(
                "Texture plan is not executable: "
                f"decision state is {plan.decision.state!r}."
            )
        if not plan.selected_unit_ids:
            raise RuntimeError("Texture plan contains no selected units.")
        planned = TextureWorkflowProgress.build(
            mode=mode,
            phase="planned",
            iteration=0,
            selected_unit_ids=plan.selected_unit_ids,
            accepted_unit_ids=(),
            remaining_unit_ids=plan.selected_unit_ids,
            message="Immutable texture plan accepted before backend work.",
        )
        if progress_callback is not None:
            progress_callback(planned)
        if planned_plan_digest != texture_plan_digest(plan):
            raise TextureWorkflowRuntimeError(
                "Texture workflow plan changed before checkpointing"
            )
        verify_precheckpoint_request_identity(boundary="before checkpointing")
        checkpoint = store.create(
            mode=mode,
            request=request,
            plan=plan,
            source_identity_digest=source_identity_digest,
            next_action="execute",
            progress=(planned,),
            client_resume_state=client_resume_state,
        )
        if pause_after_transition:
            if decision_patch is not None:
                raise TextureWorkflowRuntimeError(
                    "The initial planning step does not accept a decision patch"
                )
            raise _TextureWorkflowStepPaused(checkpoint)

    plan = checkpoint.plan
    _require_plan_request_scope(request, plan)
    selected_unit_ids = checkpoint.selected_unit_ids
    accepted_unit_ids = checkpoint.accepted_unit_ids
    remaining_unit_ids = checkpoint.remaining_unit_ids
    pending_validation_unit_ids = checkpoint.pending_validation_unit_ids
    executions = list(checkpoint.executions)
    validations = list(checkpoint.validations)
    progress = list(checkpoint.progress)
    unit_artifacts = dict(checkpoint.unit_artifacts)
    accepted_unit_material_state_digests = dict(
        checkpoint.accepted_unit_material_state_digests
    )
    output_asset_path = checkpoint.output_asset_path
    iteration = checkpoint.iteration
    action: TextureWorkflowAction = checkpoint.next_action
    decision_consumed = False

    def verify_identity() -> None:
        verify_checkpoint_identity(
            checkpoint,
            request=request,
            active_plan=plan,
        )
        if request.output_dir.resolve() != output_dir:
            raise TextureWorkflowRuntimeError(
                "Texture workflow output directory changed after checkpointing"
            )

    def report(
        phase: TextureWorkflowPhase,
        message: str,
        *,
        event_iteration: int | None = None,
    ) -> None:
        item = TextureWorkflowProgress.build(
            mode=mode,
            phase=phase,
            iteration=iteration if event_iteration is None else event_iteration,
            selected_unit_ids=selected_unit_ids,
            accepted_unit_ids=accepted_unit_ids,
            remaining_unit_ids=remaining_unit_ids,
            message=message,
        )
        progress.append(item)
        if progress_callback is not None:
            progress_callback(item)

    def save(
        next_action: TextureWorkflowAction,
        *,
        terminal_status: TextureFinalizationStatus | None = None,
        cancellation_reason: str | None = None,
        refresh_execution_integrity: bool = False,
        refresh_validation_integrity: bool = False,
    ) -> None:
        nonlocal checkpoint, action
        verify_identity()
        client_resume_state = (
            _export_client_resume_state(client, plan)
            if client_resume_restored
            else checkpoint.client_resume_state
        )
        verify_identity()
        checkpoint = checkpoint.model_copy(
            update={
                "next_action": next_action,
                "iteration": iteration,
                "accepted_unit_ids": accepted_unit_ids,
                "remaining_unit_ids": remaining_unit_ids,
                "pending_validation_unit_ids": pending_validation_unit_ids,
                "executions": tuple(executions),
                "validations": tuple(validations),
                "progress": tuple(progress),
                "unit_artifacts": unit_artifacts,
                "artifact_digests": (
                    collect_artifact_digests(unit_artifacts)
                    if refresh_execution_integrity
                    else checkpoint.artifact_digests
                ),
                "output_asset_sha256": (
                    collect_output_asset_digest(output_asset_path)
                    if refresh_execution_integrity
                    else checkpoint.output_asset_sha256
                ),
                "validation_evidence_sha256_by_path": (
                    collect_validation_evidence_digests(tuple(validations))
                    if refresh_validation_integrity
                    else checkpoint.validation_evidence_sha256_by_path
                ),
                "accepted_unit_material_state_digests": (
                    accepted_unit_material_state_digests
                ),
                "output_asset_path": output_asset_path,
                "client_resume_state": client_resume_state,
                "terminal_status": terminal_status,
                "cancellation_reason": cancellation_reason,
            }
        )
        checkpoint = store.save(checkpoint)
        verify_identity()
        action = next_action

    def ensure_client_resume_restored() -> None:
        nonlocal client_resume_restored
        if client_resume_restored:
            return
        verify_identity()
        _restore_client_resume_state(
            client,
            plan,
            checkpoint.client_resume_state,
        )
        verify_identity()
        client_resume_restored = True

    def require_decision() -> None:
        nonlocal decision_consumed
        if not pause_after_transition:
            return
        if decision_consumed:
            raise TextureWorkflowRuntimeError(
                "A Texture step decision cannot authorize multiple transitions"
            )
        if decision_patch is None:
            raise TextureWorkflowRuntimeError(
                f"Texture action {action!r} requires an evidence-bound decision patch"
            )
        try:
            validate_texture_decision_patch(
                decision_patch,
                checkpoint=checkpoint,
                output_dir=output_dir,
            )
            record_texture_decision_patch(
                decision_patch,
                output_dir=output_dir,
                workflow_lock=workflow_lock,
            )
        except ValueError as exc:
            raise TextureWorkflowRuntimeError(str(exc)) from exc
        decision_consumed = True

    def pause() -> None:
        if pause_after_transition:
            raise _TextureWorkflowStepPaused(checkpoint)

    def finalize(
        status: TextureFinalizationStatus,
        *,
        cancellation_reason: str | None = None,
    ) -> TextureFinalizationResult:
        verify_identity()
        finalizer_plan = TexturePlanDocument.model_validate(
            plan.model_dump(mode="python", round_trip=True)
        )
        finalizer_request = request.model_copy(deep=True)
        result = finalizer_impl.finalize(
            TextureFinalizerInput(
                mode=mode,
                request=finalizer_request,
                plan=finalizer_plan,
                terminal_status=status,
                cancellation_reason=cancellation_reason,
                executions=tuple(executions),
                validations=tuple(validations),
                progress=tuple(progress),
                unit_artifacts=unit_artifacts,
                accepted_unit_ids=accepted_unit_ids,
                remaining_unit_ids=remaining_unit_ids,
                output_asset_path=output_asset_path,
                output_asset_sha256=checkpoint.output_asset_sha256,
                workflow_checkpoint_path=str(store.path),
                decision_ledger_path=(
                    str(texture_decision_ledger_path(output_dir))
                    if texture_decision_ledger_path(output_dir).is_file()
                    or (pause_after_transition and status != "cancelled")
                    else None
                ),
            )
        )
        if checkpoint.request_digest != texture_request_digest(finalizer_request):
            raise TextureWorkflowRuntimeError(
                "Texture workflow finalizer mutated the request identity"
            )
        if finalizer_request.output_dir.resolve() != output_dir:
            raise TextureWorkflowRuntimeError(
                "Texture workflow finalizer mutated the output directory"
            )
        if checkpoint.plan_digest != texture_plan_digest(finalizer_plan):
            raise TextureWorkflowRuntimeError(
                "Texture workflow finalizer mutated the immutable plan"
            )
        verify_identity()
        return result

    def cancel() -> TextureFinalizationResult:
        # Cancellation can race the first action after an exact orphan patch was
        # recovered. Bind those already-reviewed bytes before advancing the
        # checkpoint; the cancelled revision then requires a fresh decision.
        if (
            pause_after_transition
            and decision_patch_recovered
            and decision_patch is not None
            and not decision_consumed
        ):
            require_decision()
        reason = "Texture workflow cancellation requested."
        report("cancelled", reason)
        save(
            action,
            terminal_status="cancelled",
            cancellation_reason=reason,
        )
        return finalize("cancelled", cancellation_reason=reason)

    if checkpoint.next_action == "done":
        if checkpoint.terminal_status not in {"pass", "conditional"}:
            raise TextureWorkflowRuntimeError(
                "Completed Texture checkpoint has no valid terminal status"
            )
        return finalize(checkpoint.terminal_status)

    if resume:
        if checkpoint.terminal_status == "cancelled":
            checkpoint = checkpoint.model_copy(
                update={"terminal_status": None, "cancellation_reason": None}
            )
        if not pause_after_transition:
            report(
                "resuming",
                f"Resuming durable Texture workflow at action {action!r}.",
            )
            save(action)

    while True:
        verify_identity()
        if cancellation_check is not None and cancellation_check():
            return cancel()

        if action == "execute":
            require_decision()
            ensure_client_resume_restored()
            report("executing", "Executing all approved selected-unit IDs.")
            save("execute")
            try:
                execution = _execute_client_units(
                    client,
                    plan,
                    remaining_unit_ids,
                    output_dir=output_dir,
                    preserved_artifacts={},
                    persist_resume_state=lambda: save("execute"),
                    cancellation_check=cancellation_check,
                )
            except TextureAgentServiceCancellationRequested:
                return cancel()
            verify_identity()
            _require_execution_scope(execution, remaining_unit_ids)
            executions.append(execution)
            unit_artifacts.update(
                {artifact.unit_id: artifact for artifact in execution.unit_artifacts}
            )
            output_asset_path = execution.output_asset_path
            pending_validation_unit_ids = remaining_unit_ids
            save("validate", refresh_execution_integrity=True)
            pause()
            continue

        if action == "validate":
            require_decision()
            if output_asset_path is None:
                raise TextureWorkflowRuntimeError(
                    "Validation checkpoint has no output asset"
                )
            verify_output_asset_digest(checkpoint)
            verify_validation_evidence_digests(checkpoint)
            verify_accepted_unit_material_state(
                checkpoint,
                output_asset_path=output_asset_path,
            )
            validation = validator.validate(
                request=request,
                plan=plan,
                output_asset_path=output_asset_path,
                unit_artifacts=unit_artifacts,
                unit_ids=pending_validation_unit_ids,
                iteration=iteration,
                output_dir=output_dir,
            )
            verify_identity()
            verify_output_asset_digest(checkpoint)
            verify_validation_evidence_digests(checkpoint)
            _require_validation_scope(
                validation,
                pending_validation_unit_ids,
                output_asset_path=output_asset_path,
                iteration=iteration,
            )
            validations.append(validation)
            previously_accepted_unit_ids = accepted_unit_ids
            accepted_unit_ids, remaining_unit_ids = _ordered_partition(
                selected_unit_ids,
                validation.failed_unit_ids,
            )
            if previously_accepted_unit_ids:
                verify_accepted_unit_material_state(
                    checkpoint,
                    output_asset_path=output_asset_path,
                )
            accepted_unit_material_state_digests = texture_unit_material_state_digests(
                output_asset_path=output_asset_path,
                plan=plan,
                unit_ids=accepted_unit_ids,
            )
            pending_validation_unit_ids = ()
            report(
                "validating",
                "usd-cli VQA updated accepted and remaining selected units.",
            )
            save(
                "refine"
                if remaining_unit_ids and iteration < request.max_vqa_iterations
                else "finalize",
                refresh_validation_integrity=True,
            )
            pause()
            continue

        if action == "refine":
            require_decision()
            ensure_client_resume_restored()
            verify_artifact_digests(
                checkpoint,
                unit_ids=accepted_unit_ids,
            )
            regeneration_ids = remaining_unit_ids
            next_iteration = iteration + 1
            report(
                "refining",
                "Regenerating only exact unit IDs that failed usd-cli VQA.",
                event_iteration=next_iteration,
            )
            save("refine")
            preserved_artifacts = {
                unit_id: unit_artifacts[unit_id] for unit_id in accepted_unit_ids
            }
            try:
                execution = _execute_client_units(
                    client,
                    plan,
                    regeneration_ids,
                    output_dir=output_dir,
                    preserved_artifacts=preserved_artifacts,
                    persist_resume_state=lambda: save("refine"),
                    cancellation_check=cancellation_check,
                )
            except TextureAgentServiceCancellationRequested:
                return cancel()
            verify_identity()
            _require_execution_scope(execution, regeneration_ids)
            verify_artifact_digests(
                checkpoint,
                unit_ids=accepted_unit_ids,
            )
            verify_accepted_unit_material_state(
                checkpoint,
                output_asset_path=execution.output_asset_path,
            )
            executions.append(execution)
            unit_artifacts.update(
                {artifact.unit_id: artifact for artifact in execution.unit_artifacts}
            )
            output_asset_path = execution.output_asset_path
            pending_validation_unit_ids = regeneration_ids
            iteration = next_iteration
            save("validate", refresh_execution_integrity=True)
            pause()
            continue

        if action == "finalize":
            require_decision()
            verify_artifact_digests(checkpoint)
            verify_accepted_unit_material_state(checkpoint)
            verify_output_asset_digest(checkpoint)
            verify_validation_evidence_digests(checkpoint)
            report("finalizing", "Writing canonical texture workflow artifacts.")
            save("finalize")
            status: TextureFinalizationStatus = (
                "pass" if not remaining_unit_ids else "conditional"
            )
            report(
                "completed",
                (
                    "All selected units are accepted."
                    if status == "pass"
                    else "Workflow stopped with bounded unresolved selected units."
                ),
            )
            save("finalize")
            result = finalize(status)
            save("done", terminal_status=status)
            return result

        raise TextureWorkflowRuntimeError(f"Unknown Texture workflow action: {action}")


def run_texture_workflow_step(
    request: TextureWorkflowRequest,
    *,
    mode: TextureWorkflowMode,
    client: TexturePlannerExecutorClient,
    validator: TextureSceneValidator,
    decision_patch: TextureDecisionPatch | TextureEmbeddedDecisionPatch | None = None,
    finalizer: TextureWorkflowFinalizer | None = None,
    progress_callback: ProgressCallback | None = None,
    cancellation_check: CancellationCheck | None = None,
    checkpoint_store: TextureWorkflowCheckpointStore | None = None,
) -> (
    TextureStepObservation | TextureEmbeddedStepObservation | TextureFinalizationResult
):
    """Advance exactly one focused Texture capability boundary.

    The first call performs immutable planning and yields an observation. Every
    later non-terminal call requires one patch bound to that observation. This
    is the shared control surface for a standalone child and an embedded outer
    coordinator; neither needs to call the fixed compatibility controller.
    """

    context = request.execution_context
    if context is not None and context.mode == "embedded":
        from .embedded_workflow import run_embedded_texture_workflow_step

        if decision_patch is not None and not isinstance(
            decision_patch,
            TextureEmbeddedDecisionPatch,
        ):
            raise TextureWorkflowRuntimeError(
                "Embedded Texture requires an outer-owned decision patch"
            )
        return run_embedded_texture_workflow_step(
            request,
            mode=mode,
            client=client,
            validator=validator,
            decision_patch=decision_patch,
            finalizer=finalizer,
            progress_callback=progress_callback,
            cancellation_check=cancellation_check,
            checkpoint_store=checkpoint_store,
        )
    if decision_patch is not None and not isinstance(
        decision_patch, TextureDecisionPatch
    ):
        raise TextureWorkflowRuntimeError(
            "Standalone Texture requires the fixed-pipeline decision patch"
        )
    store = checkpoint_store or TextureWorkflowCheckpointStore(request.output_dir)
    with store.transaction(ledger_output_dir=request.output_dir) as workflow_lock:
        resume = store.exists()
        decision_patch_recovered = False
        if resume:
            checkpoint = store.load()
            validate_resume_identity(checkpoint, request=request, mode=mode)
            try:
                recovered_decision_patch = verify_texture_resume_decision_state(
                    checkpoint,
                    output_dir=request.output_dir,
                )
            except ValueError as exc:
                raise TextureWorkflowRuntimeError(str(exc)) from exc
            if recovered_decision_patch is not None:
                if (
                    decision_patch is not None
                    and decision_patch != recovered_decision_patch
                ):
                    raise TextureWorkflowRuntimeError(
                        "Texture resume must reuse the exact persisted current "
                        "decision patch"
                    )
                decision_patch = recovered_decision_patch
                decision_patch_recovered = True
            if checkpoint.next_action == "done" and decision_patch is not None:
                raise TextureWorkflowRuntimeError(
                    "Completed Texture workflow does not accept another decision patch"
                )
            if decision_patch is None and checkpoint.next_action != "done":
                return build_texture_step_observation(
                    checkpoint,
                    output_dir=request.output_dir,
                )
        try:
            return run_texture_workflow(
                request,
                mode=mode,
                client=client,
                validator=validator,
                finalizer=finalizer,
                progress_callback=progress_callback,
                resume=resume,
                cancellation_check=cancellation_check,
                checkpoint_store=store,
                decision_patch=decision_patch,
                pause_after_transition=True,
                workflow_lock=workflow_lock,
                decision_patch_recovered=decision_patch_recovered,
            )
        except _TextureWorkflowStepPaused as paused:
            return build_texture_step_observation(
                paused.checkpoint,
                output_dir=request.output_dir,
            )


def run_interactive_texture_workflow(
    request: TextureWorkflowRequest,
    *,
    client: TexturePlannerExecutorClient,
    validator: TextureSceneValidator,
    finalizer: TextureWorkflowFinalizer | None = None,
    progress_callback: ProgressCallback | None = None,
    resume: bool = False,
    cancellation_check: CancellationCheck | None = None,
) -> TextureFinalizationResult:
    """Interactive entry point using the shared durable workflow."""

    return run_texture_workflow(
        request,
        mode="interactive",
        client=client,
        validator=validator,
        finalizer=finalizer,
        progress_callback=progress_callback,
        resume=resume,
        cancellation_check=cancellation_check,
    )


def run_batch_texture_workflow(
    request: TextureWorkflowRequest,
    *,
    client: TexturePlannerExecutorClient,
    validator: TextureSceneValidator,
    finalizer: TextureWorkflowFinalizer | None = None,
    progress_callback: ProgressCallback | None = None,
    resume: bool = False,
    cancellation_check: CancellationCheck | None = None,
) -> TextureFinalizationResult:
    """Batch-wrapper entry point using the shared durable workflow."""

    return run_texture_workflow(
        request,
        mode="batch",
        client=client,
        validator=validator,
        finalizer=finalizer,
        progress_callback=progress_callback,
        resume=resume,
        cancellation_check=cancellation_check,
    )
