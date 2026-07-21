# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared interactive and batch orchestration for bounded texture workflows."""

from __future__ import annotations

from collections.abc import Callable

from .client import TexturePlannerExecutorClient
from .finalizer import (
    CanonicalTextureWorkflowFinalizer,
    TextureWorkflowFinalizer,
    write_texture_planning_artifacts,
)
from .models import (
    TextureExecutionResult,
    TextureFinalizationResult,
    TextureFinalizerInput,
    TextureUnitArtifact,
    TextureValidationResult,
    TextureWorkflowMode,
    TextureWorkflowPhase,
    TextureWorkflowProgress,
    TextureWorkflowRequest,
)
from .workbench_validation import TextureWorkbenchValidator

ProgressCallback = Callable[[TextureWorkflowProgress], None]


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
) -> None:
    if validation.evaluated_unit_ids != requested_unit_ids:
        raise RuntimeError(
            "Workbench validation response scope differs from the requested unit IDs."
        )
    failed = validation.failed_unit_ids
    if len(failed) != len(set(failed)):
        raise RuntimeError("Workbench validation returned duplicate failed unit IDs.")
    unknown_failed = sorted(set(failed) - set(requested_unit_ids))
    if unknown_failed:
        raise RuntimeError(
            "Workbench validation failed unit IDs outside the requested scope: "
            + ", ".join(unknown_failed)
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


def run_texture_workflow(
    request: TextureWorkflowRequest,
    *,
    mode: TextureWorkflowMode,
    client: TexturePlannerExecutorClient,
    validator: TextureWorkbenchValidator,
    finalizer: TextureWorkflowFinalizer | None = None,
    progress_callback: ProgressCallback | None = None,
) -> TextureFinalizationResult:
    """Plan, execute, validate, refine exact failures, and finalize artifacts."""

    output_dir = request.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_unit_ids: tuple[str, ...] = ()
    accepted_unit_ids: tuple[str, ...] = ()
    remaining_unit_ids: tuple[str, ...] = ()
    progress: list[TextureWorkflowProgress] = []

    def report(
        phase: TextureWorkflowPhase,
        message: str,
        *,
        iteration: int = 0,
    ) -> None:
        item = TextureWorkflowProgress.build(
            mode=mode,
            phase=phase,
            iteration=iteration,
            selected_unit_ids=selected_unit_ids,
            accepted_unit_ids=accepted_unit_ids,
            remaining_unit_ids=remaining_unit_ids,
            message=message,
        )
        progress.append(item)
        if progress_callback is not None:
            progress_callback(item)

    plan = client.plan(request)
    write_texture_planning_artifacts(request, plan)
    if not plan.decision.execution_allowed:
        raise RuntimeError(
            "Texture plan is not executable: "
            f"decision state is {plan.decision.state!r}."
        )
    selected_unit_ids = plan.selected_unit_ids
    if not selected_unit_ids:
        raise RuntimeError("Texture plan contains no selected units.")
    remaining_unit_ids = selected_unit_ids
    report("planned", "Immutable texture plan accepted before backend work.")
    report("executing", "Executing all approved selected-unit IDs.")

    executions = [
        client.execute(
            plan,
            selected_unit_ids,
            output_dir=output_dir,
            preserved_artifacts={},
        )
    ]
    _require_execution_scope(executions[0], selected_unit_ids)
    unit_artifacts: dict[str, TextureUnitArtifact] = {
        item.unit_id: item for item in executions[0].unit_artifacts
    }
    output_asset_path = executions[0].output_asset_path

    validations = [
        validator.validate(
            output_asset_path=output_asset_path,
            unit_artifacts=unit_artifacts,
            unit_ids=selected_unit_ids,
            iteration=0,
            output_dir=output_dir,
        )
    ]
    _require_validation_scope(validations[0], selected_unit_ids)
    accepted_unit_ids, remaining_unit_ids = _ordered_partition(
        selected_unit_ids, validations[0].failed_unit_ids
    )
    report(
        "validating",
        "Workbench VQA reported accepted and remaining selected units.",
    )

    for iteration in range(1, request.max_vqa_iterations + 1):
        if not remaining_unit_ids:
            break
        regeneration_ids = remaining_unit_ids
        report(
            "refining",
            "Regenerating only the exact unit IDs that failed Workbench VQA.",
            iteration=iteration,
        )
        preserved_artifacts = {
            unit_id: unit_artifacts[unit_id] for unit_id in accepted_unit_ids
        }
        execution = client.execute(
            plan,
            regeneration_ids,
            output_dir=output_dir,
            preserved_artifacts=preserved_artifacts,
        )
        _require_execution_scope(execution, regeneration_ids)
        executions.append(execution)
        for artifact in execution.unit_artifacts:
            unit_artifacts[artifact.unit_id] = artifact
        output_asset_path = execution.output_asset_path

        validation = validator.validate(
            output_asset_path=output_asset_path,
            unit_artifacts=unit_artifacts,
            unit_ids=regeneration_ids,
            iteration=iteration,
            output_dir=output_dir,
        )
        _require_validation_scope(validation, regeneration_ids)
        validations.append(validation)
        accepted_unit_ids, remaining_unit_ids = _ordered_partition(
            selected_unit_ids, validation.failed_unit_ids
        )
        report(
            "validating",
            "Workbench VQA updated accepted and remaining selected units.",
            iteration=iteration,
        )

    report("finalizing", "Writing canonical texture workflow artifacts.")
    final_phase = TextureWorkflowProgress.build(
        mode=mode,
        phase="completed",
        iteration=len(validations) - 1,
        selected_unit_ids=selected_unit_ids,
        accepted_unit_ids=accepted_unit_ids,
        remaining_unit_ids=remaining_unit_ids,
        message=(
            "All selected units are accepted."
            if not remaining_unit_ids
            else "Workflow stopped with bounded unresolved selected units."
        ),
    )
    progress.append(final_phase)
    if progress_callback is not None:
        progress_callback(final_phase)

    finalizer_impl = finalizer or CanonicalTextureWorkflowFinalizer()
    return finalizer_impl.finalize(
        TextureFinalizerInput(
            mode=mode,
            request=request,
            plan=plan,
            executions=tuple(executions),
            validations=tuple(validations),
            progress=tuple(progress),
            unit_artifacts=unit_artifacts,
            accepted_unit_ids=accepted_unit_ids,
            remaining_unit_ids=remaining_unit_ids,
            output_asset_path=output_asset_path,
        )
    )


def run_interactive_texture_workflow(
    request: TextureWorkflowRequest,
    *,
    client: TexturePlannerExecutorClient,
    validator: TextureWorkbenchValidator,
    finalizer: TextureWorkflowFinalizer | None = None,
    progress_callback: ProgressCallback | None = None,
) -> TextureFinalizationResult:
    """Interactive entry point using the shared workflow contracts."""

    return run_texture_workflow(
        request,
        mode="interactive",
        client=client,
        validator=validator,
        finalizer=finalizer,
        progress_callback=progress_callback,
    )


def run_batch_texture_workflow(
    request: TextureWorkflowRequest,
    *,
    client: TexturePlannerExecutorClient,
    validator: TextureWorkbenchValidator,
    finalizer: TextureWorkflowFinalizer | None = None,
    progress_callback: ProgressCallback | None = None,
) -> TextureFinalizationResult:
    """Batch-wrapper entry point using the shared workflow contracts."""

    return run_texture_workflow(
        request,
        mode="batch",
        client=client,
        validator=validator,
        finalizer=finalizer,
        progress_callback=progress_callback,
    )
