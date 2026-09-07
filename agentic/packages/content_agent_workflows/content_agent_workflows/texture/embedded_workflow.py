# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Outer-owned embedded Texture workflow over bounded provider leaves."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pxr import UsdUtils
from pydantic import JsonValue

from content_agent_workflows.common.artifacts import atomic_write_json, load_json
from content_agent_workflows.common.domain_execution import ExecutionArtifactBinding
from content_agent_workflows.common.embedded_domain_artifact_store import (
    EmbeddedDecisionArtifactStore,
    EmbeddedDecisionAuthorizationReplayError,
)
from content_agent_workflows.common.embedded_domain_decision import (
    BoundedExecutionAuthorization,
    ContractArtifactReference,
    EmbeddedBoundedExecutionResult,
    EmbeddedCoordinatorDecision,
    EmbeddedCoordinatorReview,
    EmbeddedDomainEvidence,
    EmbeddedDomainProposal,
    artifact_reference,
    canonical_json_digest,
)

from .client import TexturePlannerExecutorClient
from .decision import texture_checkpoint_decision_digest
from .embedded_decision import (
    TextureEmbeddedDecisionPatch,
    TextureEmbeddedDecisionState,
    TextureEmbeddedStepObservation,
    authorize_candidate_generation,
    authorize_publication,
    bind_file,
    build_candidate_review,
    build_plan_decision,
    build_publication_decision,
    build_publication_review,
    build_review_receipt,
    candidate_evidence_artifact,
    candidate_result_artifact,
    candidate_review_evidence_artifact,
    ensure_embedded_patch_identity,
    initialize_embedded_texture_decisions,
    load_generation_chain,
    publication_result_artifact,
    texture_candidate_dependency_closure_facts,
    validate_completed_texture_decision_chain,
    validate_rejected_texture_candidate_chain,
)
from .finalizer import (
    CanonicalTextureWorkflowFinalizer,
    TextureWorkflowFinalizer,
    write_texture_planning_artifacts,
)
from .models import (
    TextureExecutionResult,
    TextureFinalizationResult,
    TextureFinalizerInput,
    TexturePlanDocument,
    TextureUnitArtifact,
    TextureValidationResult,
    TextureWorkflowMode,
    TextureWorkflowProgress,
    TextureWorkflowRequest,
)
from .runtime import (
    CancellationCheck,
    TextureWorkflowCheckpoint,
    TextureWorkflowCheckpointStore,
    TextureWorkflowRuntimeError,
    collect_artifact_digests,
    collect_output_asset_digest,
    collect_validation_evidence_digests,
    texture_request_digest,
    texture_source_identity_digest,
    validate_resume_identity,
    verify_artifact_digests,
    verify_output_asset_digest,
    verify_validation_evidence_digests,
)
from .scene_validation import TextureSceneValidator
from .scope_validation import (
    texture_unit_material_state_digests,
    validate_texture_scope_invariants,
)
from .workflow import (
    ProgressCallback,
    _execute_client_units,
    _export_client_resume_state,
    _require_execution_scope,
    _require_plan_request_scope,
    _require_validation_scope,
    _restore_client_resume_state,
)


def _state(checkpoint: TextureWorkflowCheckpoint) -> TextureEmbeddedDecisionState:
    state = checkpoint.embedded_decision_state
    if state is None:
        raise TextureWorkflowRuntimeError(
            "Embedded Texture checkpoint lacks the shared decision state"
        )
    return state


def _contract_path(
    store: EmbeddedDecisionArtifactStore,
    reference: ContractArtifactReference,
) -> Path:
    return (
        store.store_root
        / "artifacts"
        / reference.artifact_kind
        / f"{reference.sha256}.json"
    ).resolve()


def _committed_result_for_authorization(
    store: EmbeddedDecisionArtifactStore,
    authorization: BoundedExecutionAuthorization,
) -> EmbeddedBoundedExecutionResult | None:
    authorization_ref = artifact_reference(authorization)
    results: list[EmbeddedBoundedExecutionResult] = []
    for entry in store.journal().entries:
        if entry.reference.artifact_kind != "execution_result":
            continue
        result = store.load_typed(entry.reference, EmbeddedBoundedExecutionResult)
        if result.parent_artifact == authorization_ref:
            results.append(result)
    if len(results) > 1:
        raise TextureWorkflowRuntimeError(
            "Texture authorization has more than one committed execution result"
        )
    return results[0] if results else None


def _require_reconcilable_candidate_resume(
    client: TexturePlannerExecutorClient,
    plan: TexturePlanDocument,
    unit_ids: tuple[str, ...],
    *,
    preserved_artifacts: Mapping[str, TextureUnitArtifact],
) -> None:
    checker = getattr(client, "can_reconcile_authorized_execution", None)
    if (
        not callable(checker)
        or checker(
            plan,
            unit_ids,
            preserved_artifacts=preserved_artifacts,
        )
        is not True
    ):
        raise TextureWorkflowRuntimeError(
            "Committed Texture candidate authorization lacks an exact pending "
            "adapter execution; regeneration is forbidden"
        )


def _target_unit_ids(checkpoint: TextureWorkflowCheckpoint) -> tuple[str, ...]:
    if checkpoint.next_action in {"execute", "refine"}:
        return checkpoint.remaining_unit_ids
    if checkpoint.next_action == "validate":
        return checkpoint.pending_validation_unit_ids
    if checkpoint.next_action in {"finalize", "review_publication"}:
        return checkpoint.selected_unit_ids
    return ()


def build_embedded_texture_observation(
    checkpoint: TextureWorkflowCheckpoint,
    *,
    output_dir: Path,
) -> TextureEmbeddedStepObservation:
    """Return the exact evidence and shared refs reviewed by the outer loop."""

    state = _state(checkpoint)
    decision_store = EmbeddedDecisionArtifactStore(output_dir)
    candidate_output = None
    visual_evidence: tuple[ExecutionArtifactBinding, ...] = ()
    if state.current_result is not None:
        candidate_result = decision_store.load_typed(
            state.current_result,
            EmbeddedBoundedExecutionResult,
        )
        candidate_output = (
            candidate_result.outputs[0] if candidate_result.outputs else None
        )
        visual_evidence = tuple(
            artifact
            for record in candidate_result.evidence
            if record.evidence_type == "critique"
            for artifact in record.artifacts
        )
    publication_output = None
    if state.publication_result is not None:
        publication_result = decision_store.load_typed(
            state.publication_result,
            EmbeddedBoundedExecutionResult,
        )
        publication_output = (
            publication_result.outputs[0] if publication_result.outputs else None
        )
    action = checkpoint.next_action
    if action not in {
        "execute",
        "validate",
        "refine",
        "finalize",
        "review_publication",
        "done",
    }:
        raise TextureWorkflowRuntimeError(
            f"Unsupported embedded Texture action: {action}"
        )
    patch_path = None
    if action != "done":
        patch_path = str(
            output_dir
            / "decisions"
            / f"{checkpoint.revision:04d}-{action}-embedded-decision.json"
        )
    return TextureEmbeddedStepObservation(
        request_digest=checkpoint.request_digest,
        source_identity_digest=checkpoint.source_identity_digest,
        proposal_plan_digest=checkpoint.plan_digest,
        checkpoint_decision_digest=texture_checkpoint_decision_digest(checkpoint),
        checkpoint_revision=checkpoint.revision,
        action=action,
        iteration=checkpoint.iteration,
        target_unit_ids=_target_unit_ids(checkpoint),
        accepted_unit_ids=checkpoint.accepted_unit_ids,
        remaining_unit_ids=checkpoint.remaining_unit_ids,
        inspection=state.inspection,
        service_plan_proposal=checkpoint.plan,
        canonical_plan=state.canonical_plan,
        current_candidate_result=state.current_result,
        current_candidate_evidence=state.current_candidate_evidence,
        candidate_output=candidate_output,
        visual_evidence_artifacts=visual_evidence,
        accepted_candidate_receipt=state.accepted_candidate_receipt,
        publication_result=state.publication_result,
        publication_output=publication_output,
        completed_receipt=state.completed_receipt,
        decision_patch_path=patch_path,
        terminal=action == "done",
    )


def _load_decision_inputs(
    store: EmbeddedDecisionArtifactStore,
    decision: EmbeddedCoordinatorDecision,
) -> tuple[tuple[EmbeddedDomainEvidence, ...], tuple[EmbeddedDomainProposal, ...]]:
    evidence = tuple(
        store.load_typed(reference, EmbeddedDomainEvidence)
        for reference in decision.evidence_artifacts
    )
    proposals = tuple(
        store.load_typed(reference, EmbeddedDomainProposal)
        for reference in decision.proposal_artifacts
    )
    return evidence, proposals


def _publication_path(
    output_dir: Path,
    candidate: ExecutionArtifactBinding,
) -> Path:
    suffix = Path(candidate.path).suffix.lower()
    if suffix not in {".usd", ".usda", ".usdc", ".usdz"}:
        raise TextureWorkflowRuntimeError(
            "Texture accepted candidate must be a USD or USDZ asset"
        )
    publication_dir = (output_dir / "published").resolve()
    return publication_dir / f"texture-accepted-{candidate.sha256[:20]}{suffix}"


def _validate_publication_result_bindings(
    result: EmbeddedBoundedExecutionResult,
    *,
    expected_path: Path | None = None,
    candidate: ExecutionArtifactBinding | None = None,
) -> ExecutionArtifactBinding:
    """Rebind every persisted publication output/evidence artifact."""

    if len(result.outputs) != 1:
        raise TextureWorkflowRuntimeError(
            "Texture publication result must bind exactly one output"
        )
    published = result.outputs[0]
    published_path = Path(published.path).expanduser()
    if published_path.is_symlink():
        raise TextureWorkflowRuntimeError(
            "Texture publication result rejects a symlinked output"
        )
    if (
        expected_path is not None
        and published_path.resolve() != expected_path.resolve()
    ):
        raise TextureWorkflowRuntimeError(
            "Texture publication result names a non-canonical output path"
        )
    if candidate is not None and (
        published.sha256 != candidate.sha256
        or published.size_bytes != candidate.size_bytes
    ):
        raise TextureWorkflowRuntimeError(
            "Texture publication result differs from the accepted candidate"
        )
    bindings = (
        published,
        *(artifact for record in result.evidence for artifact in record.artifacts),
    )
    for binding in bindings:
        path = Path(binding.path).expanduser()
        if path.is_symlink():
            raise TextureWorkflowRuntimeError(
                "Texture publication result artifact bytes changed after commit"
            )
        try:
            current = bind_file(path)
        except OSError as exc:
            raise TextureWorkflowRuntimeError(
                "Texture publication result artifact is unavailable after commit"
            ) from exc
        if current != binding:
            raise TextureWorkflowRuntimeError(
                "Texture publication result artifact bytes changed after commit"
            )
    return published


def _copy_candidate_once(
    candidate: ExecutionArtifactBinding,
    destination: Path,
) -> ExecutionArtifactBinding:
    """Publish exact candidate bytes once without overwrite or substitution."""

    source = Path(candidate.path).expanduser().resolve()
    if bind_file(source) != candidate:
        raise TextureWorkflowRuntimeError(
            "Accepted Texture candidate changed before deterministic publication"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(destination, flags, 0o600)
    try:
        with (
            source.open("rb") as source_file,
            os.fdopen(
                descriptor,
                "wb",
                closefd=False,
            ) as output_file,
        ):
            while chunk := source_file.read(1024 * 1024):
                output_file.write(chunk)
            output_file.flush()
            os.fsync(output_file.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    finally:
        os.close(descriptor)
    published = bind_file(destination)
    if published.sha256 != candidate.sha256:
        destination.unlink(missing_ok=True)
        raise TextureWorkflowRuntimeError(
            "Published Texture bytes differ from the accepted candidate"
        )
    return published


def _record_candidate_dependency_closure(
    candidate_path: str | Path,
    *,
    output_dir: Path,
) -> tuple[ExecutionArtifactBinding, dict[str, JsonValue]]:
    candidate = bind_file(candidate_path)
    try:
        facts = texture_candidate_dependency_closure_facts(candidate)
    except ValueError as exc:
        raise TextureWorkflowRuntimeError(str(exc)) from exc
    manifest_path = atomic_write_json(
        output_dir
        / "candidate-closure"
        / f"{candidate.sha256}-dependency-closure.json",
        facts,
    )
    return bind_file(manifest_path), facts


def _load_candidate_dependency_closure(
    decision_store: EmbeddedDecisionArtifactStore,
    state: TextureEmbeddedDecisionState,
) -> tuple[ExecutionArtifactBinding, dict[str, JsonValue]]:
    result_ref = state.accepted_candidate_result
    if result_ref is None:
        raise TextureWorkflowRuntimeError(
            "Texture publication lacks an accepted candidate result"
        )
    result = decision_store.load_typed(result_ref, EmbeddedBoundedExecutionResult)
    records = tuple(
        record
        for record in result.evidence
        if record.evidence_id == "texture-candidate-dependency-closure"
    )
    if len(records) != 1 or len(records[0].artifacts) != 1:
        raise TextureWorkflowRuntimeError(
            "Accepted Texture candidate lacks one dependency-closure manifest"
        )
    binding = records[0].artifacts[0]
    if bind_file(binding.path) != binding:
        raise TextureWorkflowRuntimeError(
            "Accepted Texture candidate dependency-closure manifest changed"
        )
    raw_facts = load_json(Path(binding.path))
    recorded_facts = dict(records[0].facts)
    if not isinstance(raw_facts, dict) or canonical_json_digest(
        raw_facts
    ) != canonical_json_digest(recorded_facts):
        raise TextureWorkflowRuntimeError(
            "Texture candidate dependency-closure facts differ from their manifest"
        )
    facts = recorded_facts
    if (
        not result.outputs
        or facts.get("candidate_sha256") != result.outputs[0].sha256
        or facts.get("candidate_size_bytes") != result.outputs[0].size_bytes
        or facts.get("closure_sealed_by_candidate_bytes") is not True
    ):
        raise TextureWorkflowRuntimeError(
            "Texture candidate dependency closure binds another candidate"
        )
    return binding, facts


def _verify_publication(
    *,
    request: TextureWorkflowRequest,
    plan: TexturePlanDocument,
    published: ExecutionArtifactBinding,
    output_dir: Path,
    candidate_closure_facts: Mapping[str, JsonValue],
) -> tuple[tuple[ExecutionArtifactBinding, ...], dict[str, JsonValue]]:
    """Verify target scope, dependency closure, and preserved non-target state."""

    report = validate_texture_scope_invariants(
        source_asset_path=request.source_asset,
        output_asset_path=published.path,
        plan=plan,
    )
    report_path = atomic_write_json(
        output_dir / "publication" / "scope_invariants.json",
        report,
    )
    if not report.passed:
        raise TextureWorkflowRuntimeError(
            "Texture publication violates UV, material, or preserved-content scope"
        )
    try:
        published_closure_facts = texture_candidate_dependency_closure_facts(published)
    except ValueError as exc:
        raise TextureWorkflowRuntimeError(str(exc)) from exc
    if canonical_json_digest(published_closure_facts) != canonical_json_digest(
        candidate_closure_facts
    ):
        raise TextureWorkflowRuntimeError(
            "Texture publication dependency closure differs from the reviewed candidate"
        )
    try:
        layers, assets, unresolved = UsdUtils.ComputeAllDependencies(published.path)
    except Exception as exc:
        raise TextureWorkflowRuntimeError(
            f"Could not inspect Texture publication dependency closure: {exc}"
        ) from exc
    unresolved_paths = tuple(sorted(str(item) for item in unresolved))
    if unresolved_paths:
        raise TextureWorkflowRuntimeError(
            "Texture publication dependency closure is unresolved: "
            + ", ".join(unresolved_paths)
        )
    published_path = Path(published.path).resolve()
    if published_path.suffix.lower() != ".usdz":
        external_paths: list[str] = []
        run_root = output_dir.expanduser().resolve()
        for item in (*layers, *assets):
            raw_path = getattr(item, "realPath", None) or str(item)
            dependency = Path(str(raw_path)).expanduser()
            if not dependency.is_absolute():
                continue
            resolved = dependency.resolve()
            if resolved != published_path and not resolved.is_relative_to(run_root):
                external_paths.append(str(resolved))
        if external_paths:
            raise TextureWorkflowRuntimeError(
                "Texture publication dependency closure escapes the run: "
                + ", ".join(sorted(set(external_paths)))
            )
    closure_path = atomic_write_json(
        output_dir / "publication" / "dependency_closure.json",
        {
            "schema_version": (
                "content-agent-workflows.texture-publication-closure.v1"
            ),
            "published_asset": published.model_dump(mode="json"),
            "layer_count": len(layers),
            "asset_count": len(assets),
            "unresolved": [],
            "candidate_closure": dict(candidate_closure_facts),
            "published_closure": published_closure_facts,
        },
    )
    return (
        (bind_file(report_path), bind_file(closure_path)),
        {
            "published_asset_sha256": published.sha256,
            "scope_invariants_passed": True,
            "geometry_unchanged": report.geometry_unchanged,
            "non_target_materials_unchanged": (report.non_target_materials_unchanged),
            "bindings_unchanged": report.bindings_unchanged,
            "structure_unchanged_outside_target": (
                report.structure_unchanged_outside_target
            ),
            "dependency_closure_complete": True,
            "exact_candidate_identity_preserved": (
                published_closure_facts == dict(candidate_closure_facts)
            ),
        },
    )


def _finalize(
    checkpoint: TextureWorkflowCheckpoint,
    *,
    request: TextureWorkflowRequest,
    store: TextureWorkflowCheckpointStore,
    finalizer: TextureWorkflowFinalizer,
) -> TextureFinalizationResult:
    state = _state(checkpoint)
    if checkpoint.terminal_status == "conditional":
        validate_rejected_texture_candidate_chain(
            EmbeddedDecisionArtifactStore(request.output_dir),
            state,
        )
        return finalizer.finalize(
            TextureFinalizerInput(
                mode=checkpoint.mode,
                request=request.model_copy(deep=True),
                plan=checkpoint.plan.model_copy(deep=True),
                terminal_status="conditional",
                executions=checkpoint.executions,
                validations=checkpoint.validations,
                progress=checkpoint.progress,
                unit_artifacts=checkpoint.unit_artifacts,
                accepted_unit_ids=checkpoint.accepted_unit_ids,
                remaining_unit_ids=checkpoint.remaining_unit_ids,
                output_asset_path=checkpoint.output_asset_path,
                output_asset_sha256=checkpoint.output_asset_sha256,
                workflow_checkpoint_path=str(store.path),
            )
        )
    if checkpoint.terminal_status not in {None, "pass"}:
        raise TextureWorkflowRuntimeError(
            "Embedded Texture terminal state is neither accepted nor rejected"
        )
    receipt_ref = state.completed_receipt
    if receipt_ref is None:
        raise TextureWorkflowRuntimeError(
            "Embedded Texture completion requires a reviewed decision receipt"
        )
    decision_store = EmbeddedDecisionArtifactStore(request.output_dir)
    publication_result_ref = state.publication_result
    if publication_result_ref is None:
        raise TextureWorkflowRuntimeError(
            "Embedded Texture completion lacks its publication result"
        )
    publication_result = decision_store.load_typed(
        publication_result_ref,
        EmbeddedBoundedExecutionResult,
    )
    published = _validate_publication_result_bindings(publication_result)
    if (
        checkpoint.output_asset_path != published.path
        or checkpoint.output_asset_sha256 != published.sha256
    ):
        raise TextureWorkflowRuntimeError(
            "Embedded Texture checkpoint differs from the committed publication"
        )
    receipt = validate_completed_texture_decision_chain(
        decision_store,
        state,
        checkpoint.plan,
    )
    if artifact_reference(receipt) != receipt_ref:
        raise TextureWorkflowRuntimeError(
            "Embedded Texture completion references another decision receipt"
        )
    receipt_path = _contract_path(decision_store, receipt_ref)
    return finalizer.finalize(
        TextureFinalizerInput(
            mode=checkpoint.mode,
            request=request.model_copy(deep=True),
            plan=checkpoint.plan.model_copy(deep=True),
            terminal_status="pass",
            executions=checkpoint.executions,
            validations=checkpoint.validations,
            progress=checkpoint.progress,
            unit_artifacts=checkpoint.unit_artifacts,
            accepted_unit_ids=checkpoint.accepted_unit_ids,
            remaining_unit_ids=checkpoint.remaining_unit_ids,
            output_asset_path=checkpoint.output_asset_path,
            output_asset_sha256=checkpoint.output_asset_sha256,
            workflow_checkpoint_path=str(store.path),
            embedded_decision_receipt_path=str(receipt_path),
        )
    )


def _progress(
    checkpoint: TextureWorkflowCheckpoint,
    *,
    phase: str,
    message: str,
    iteration: int | None = None,
) -> TextureWorkflowProgress:
    return TextureWorkflowProgress.build(
        mode=checkpoint.mode,
        phase=phase,  # type: ignore[arg-type]
        iteration=checkpoint.iteration if iteration is None else iteration,
        selected_unit_ids=checkpoint.selected_unit_ids,
        accepted_unit_ids=checkpoint.accepted_unit_ids,
        remaining_unit_ids=checkpoint.remaining_unit_ids,
        message=message,
    )


def _save(
    checkpoint_store: TextureWorkflowCheckpointStore,
    checkpoint: TextureWorkflowCheckpoint,
    *,
    next_action: str,
    state: TextureEmbeddedDecisionState,
    executions: tuple[TextureExecutionResult, ...] | None = None,
    validations: tuple[TextureValidationResult, ...] | None = None,
    progress: tuple[TextureWorkflowProgress, ...] | None = None,
    unit_artifacts: Mapping[str, TextureUnitArtifact] | None = None,
    accepted_unit_ids: tuple[str, ...] | None = None,
    remaining_unit_ids: tuple[str, ...] | None = None,
    pending_validation_unit_ids: tuple[str, ...] | None = None,
    output_asset_path: str | None = None,
    iteration: int | None = None,
    client_resume_state: dict[str, Any] | None = None,
    terminal_status: str | None = None,
) -> TextureWorkflowCheckpoint:
    execution_values = executions if executions is not None else checkpoint.executions
    validation_values = (
        validations if validations is not None else checkpoint.validations
    )
    artifact_values = dict(
        unit_artifacts if unit_artifacts is not None else checkpoint.unit_artifacts
    )
    accepted_values = (
        accepted_unit_ids
        if accepted_unit_ids is not None
        else checkpoint.accepted_unit_ids
    )
    output_value = (
        output_asset_path
        if output_asset_path is not None
        else checkpoint.output_asset_path
    )
    material_state = (
        texture_unit_material_state_digests(
            output_asset_path=output_value,
            plan=checkpoint.plan,
            unit_ids=accepted_values,
        )
        if accepted_values and output_value is not None
        else {}
    )
    retained_accepted_ids = set(checkpoint.accepted_unit_ids) & set(accepted_values)
    for unit_id in retained_accepted_ids:
        prior_digest = checkpoint.accepted_unit_material_state_digests.get(unit_id)
        if prior_digest is not None and material_state.get(unit_id) != prior_digest:
            raise TextureWorkflowRuntimeError(
                "Texture candidate changed previously accepted material state before "
                f"fresh outer review: {unit_id}"
            )
    updated = checkpoint.model_copy(
        update={
            "next_action": next_action,
            "iteration": checkpoint.iteration if iteration is None else iteration,
            "accepted_unit_ids": accepted_values,
            "remaining_unit_ids": (
                remaining_unit_ids
                if remaining_unit_ids is not None
                else checkpoint.remaining_unit_ids
            ),
            "pending_validation_unit_ids": (
                pending_validation_unit_ids
                if pending_validation_unit_ids is not None
                else checkpoint.pending_validation_unit_ids
            ),
            "executions": execution_values,
            "validations": validation_values,
            "progress": progress if progress is not None else checkpoint.progress,
            "unit_artifacts": artifact_values,
            "artifact_digests": collect_artifact_digests(artifact_values),
            "output_asset_path": output_value,
            "output_asset_sha256": collect_output_asset_digest(output_value),
            "validation_evidence_sha256_by_path": (
                collect_validation_evidence_digests(validation_values)
            ),
            "accepted_unit_material_state_digests": material_state,
            "client_resume_state": (
                client_resume_state
                if client_resume_state is not None
                else checkpoint.client_resume_state
            ),
            "terminal_status": terminal_status,
            "embedded_decision_state": state,
        }
    )
    return checkpoint_store.save(updated)


def _initialize(
    request: TextureWorkflowRequest,
    *,
    mode: TextureWorkflowMode,
    client: TexturePlannerExecutorClient,
    validator: TextureSceneValidator,
    checkpoint_store: TextureWorkflowCheckpointStore,
    progress_callback: ProgressCallback | None,
) -> TextureWorkflowCheckpoint:
    request_digest = texture_request_digest(request)
    source_digest = texture_source_identity_digest(request)
    raw_plan = client.plan(request)
    plan = TexturePlanDocument.model_validate(
        raw_plan.model_dump(mode="python", round_trip=True)
    )
    if request_digest != texture_request_digest(request):
        raise TextureWorkflowRuntimeError(
            "Texture request changed during provider planning"
        )
    if source_digest != texture_source_identity_digest(request):
        raise TextureWorkflowRuntimeError(
            "Texture source changed during provider planning"
        )
    write_texture_planning_artifacts(request, plan)
    _require_plan_request_scope(request, plan, exact=True)
    if not plan.decision.execution_allowed or not plan.selected_unit_ids:
        raise TextureWorkflowRuntimeError(
            "Texture service proposal is not executable or has no selected units"
        )
    inspector = getattr(validator, "inspect", None)
    if not callable(inspector):
        raise TextureWorkflowRuntimeError(
            "Embedded Texture requires provider-neutral pre-plan inspection"
        )
    inspection = inspector(request=request, plan=plan, output_dir=request.output_dir)
    decision_state = initialize_embedded_texture_decisions(
        request,
        plan,
        inspection,
    )
    planned = TextureWorkflowProgress.build(
        mode=mode,
        phase="planned",
        iteration=0,
        selected_unit_ids=plan.selected_unit_ids,
        accepted_unit_ids=(),
        remaining_unit_ids=plan.selected_unit_ids,
        message=(
            "Service plan retained as a proposal; digest-bound inspection awaits "
            "the outer canonical Texture plan."
        ),
    )
    if progress_callback is not None:
        progress_callback(planned)
    return checkpoint_store.create(
        mode=mode,
        request=request,
        plan=plan,
        source_identity_digest=texture_source_identity_digest(request, plan=plan),
        next_action="execute",
        progress=(planned,),
        client_resume_state=_export_client_resume_state(client, plan),
        embedded_decision_state=decision_state,
    )


def _generate_and_assess_candidate(
    checkpoint: TextureWorkflowCheckpoint,
    patch: TextureEmbeddedDecisionPatch,
    *,
    request: TextureWorkflowRequest,
    client: TexturePlannerExecutorClient,
    validator: TextureSceneValidator,
    checkpoint_store: TextureWorkflowCheckpointStore,
    decision_store: EmbeddedDecisionArtifactStore,
    cancellation_check: CancellationCheck | None,
    progress_callback: ProgressCallback | None,
) -> TextureWorkflowCheckpoint:
    state = _state(checkpoint)
    requested_ids = (
        checkpoint.remaining_unit_ids
        if checkpoint.next_action == "execute"
        else patch.regeneration_unit_ids
    )
    if checkpoint.next_action == "refine":
        if requested_ids != checkpoint.remaining_unit_ids:
            raise TextureWorkflowRuntimeError(
                "Texture refinement must regenerate the exact unresolved units"
            )
        if checkpoint.iteration >= request.max_vqa_iterations:
            raise TextureWorkflowRuntimeError(
                "Texture refinement limit reached; rejected visual evidence cannot "
                "be published"
            )
    decision, evidence, proposals = build_plan_decision(
        patch,
        store=decision_store,
        state=state,
        proposal=checkpoint.plan,
    )
    decision_commit = decision_store.append(decision)
    if decision_commit.reference != artifact_reference(decision):
        raise TextureWorkflowRuntimeError(
            "Texture decision store committed another plan decision"
        )
    authorization = authorize_candidate_generation(
        decision,
        evidence=evidence,
        proposals=proposals,
    )
    _restore_client_resume_state(
        client, checkpoint.plan, checkpoint.client_resume_state
    )
    next_iteration = (
        checkpoint.iteration + 1
        if checkpoint.next_action == "refine"
        else checkpoint.iteration
    )
    preserved_artifacts = {
        unit_id: checkpoint.unit_artifacts[unit_id]
        for unit_id in checkpoint.accepted_unit_ids
    }

    def persist_resume_state() -> None:
        nonlocal checkpoint
        checkpoint = checkpoint_store.save_client_resume_state(
            checkpoint,
            _export_client_resume_state(client, checkpoint.plan),
        )

    def execute_leaf(
        _authorization: BoundedExecutionAuthorization,
    ) -> tuple[
        TextureExecutionResult,
        TextureValidationResult,
        ExecutionArtifactBinding,
        dict[str, JsonValue],
    ]:
        execution = _execute_client_units(
            client,
            checkpoint.plan,
            requested_ids,
            output_dir=request.output_dir,
            preserved_artifacts=preserved_artifacts,
            persist_resume_state=persist_resume_state,
            cancellation_check=cancellation_check,
        )
        _require_execution_scope(execution, requested_ids)
        closure_artifact, closure_facts = _record_candidate_dependency_closure(
            execution.output_asset_path,
            output_dir=request.output_dir,
        )
        all_artifacts = dict(checkpoint.unit_artifacts)
        all_artifacts.update(
            {artifact.unit_id: artifact for artifact in execution.unit_artifacts}
        )
        validation_unit_ids = checkpoint.selected_unit_ids
        validation = validator.validate(
            request=request,
            plan=checkpoint.plan,
            output_asset_path=execution.output_asset_path,
            unit_artifacts=all_artifacts,
            unit_ids=validation_unit_ids,
            iteration=next_iteration,
            output_dir=request.output_dir,
        )
        _require_validation_scope(
            validation,
            validation_unit_ids,
            output_asset_path=execution.output_asset_path,
            iteration=next_iteration,
        )
        return execution, validation, closure_artifact, closure_facts

    committed_result: EmbeddedBoundedExecutionResult | None = None
    try:
        (
            execution,
            validation,
            closure_artifact,
            closure_facts,
        ) = decision_store.invoke_after_authorization_commit(
            authorization,
            execute_leaf,
        )
    except EmbeddedDecisionAuthorizationReplayError as exc:
        if exc.outcome.authorization != artifact_reference(authorization):
            raise TextureWorkflowRuntimeError(
                "Texture candidate reconciliation names another authorization"
            ) from exc
        _require_reconcilable_candidate_resume(
            client,
            checkpoint.plan,
            requested_ids,
            preserved_artifacts=preserved_artifacts,
        )
        committed_result = _committed_result_for_authorization(
            decision_store,
            authorization,
        )
        (
            execution,
            validation,
            closure_artifact,
            closure_facts,
        ) = execute_leaf(authorization)
    unit_artifacts = dict(checkpoint.unit_artifacts)
    unit_artifacts.update(
        {artifact.unit_id: artifact for artifact in execution.unit_artifacts}
    )
    result_artifact = candidate_result_artifact(
        authorization,
        execution,
        validation,
        candidate_closure_artifact=closure_artifact,
        candidate_closure_facts=closure_facts,
        candidate_artifacts=tuple(
            unit_artifacts[unit_id] for unit_id in checkpoint.selected_unit_ids
        ),
        created_at=(
            committed_result.created_at if committed_result is not None else None
        ),
    )
    if committed_result is not None and result_artifact != committed_result:
        raise TextureWorkflowRuntimeError(
            "Reconciled Texture candidate differs from its committed result"
        )
    result_commit = decision_store.append_result(result_artifact)
    if committed_result is None and not result_commit.newly_committed:
        raise TextureWorkflowRuntimeError(
            "Texture candidate result was committed by another execution"
        )
    candidate_evidence = candidate_evidence_artifact(result_artifact)
    decision_store.append(candidate_evidence)
    state = state.model_copy(
        update={
            "canonical_plan": patch.canonical_plan,
            "current_decision": artifact_reference(decision),
            "current_authorization": artifact_reference(authorization),
            "current_result": artifact_reference(result_artifact),
            "current_candidate_evidence": artifact_reference(candidate_evidence),
            "current_candidate_domain_review": None,
            "current_candidate_review": None,
            "current_candidate_receipt": None,
        }
    )
    event = _progress(
        checkpoint,
        phase="validating",
        iteration=next_iteration,
        message=(
            "Candidate generation and usd-cli/VQA critique are persisted as "
            "evidence; accepted and remaining units are unchanged."
        ),
    )
    if progress_callback is not None:
        progress_callback(event)
    return _save(
        checkpoint_store,
        checkpoint,
        next_action="validate",
        state=state,
        executions=(*checkpoint.executions, execution),
        validations=(*checkpoint.validations, validation),
        progress=(*checkpoint.progress, event),
        unit_artifacts=unit_artifacts,
        pending_validation_unit_ids=checkpoint.selected_unit_ids,
        output_asset_path=execution.output_asset_path,
        iteration=next_iteration,
        client_resume_state=_export_client_resume_state(client, checkpoint.plan),
    )


def _review_candidate(
    checkpoint: TextureWorkflowCheckpoint,
    patch: TextureEmbeddedDecisionPatch,
    *,
    request: TextureWorkflowRequest,
    checkpoint_store: TextureWorkflowCheckpointStore,
    decision_store: EmbeddedDecisionArtifactStore,
    progress_callback: ProgressCallback | None,
) -> TextureWorkflowCheckpoint:
    state = _state(checkpoint)
    decision, authorization, result = load_generation_chain(
        decision_store,
        state,
    )
    review = build_candidate_review(
        patch,
        state=state,
        decision=decision,
        result=result,
    )
    domain_review_evidence = candidate_review_evidence_artifact(
        patch,
        state=state,
        result=result,
    )
    decision_store.append(domain_review_evidence)
    decision_store.append_review(review)
    evidence, proposals = _load_decision_inputs(decision_store, decision)
    receipt = build_review_receipt(
        artifact_id=f"texture-candidate-receipt-{patch.checkpoint_revision:04d}",
        decision=decision,
        authorization=authorization,
        result=result,
        review=review,
        evidence=evidence,
        proposals=proposals,
    )
    decision_store.append_receipt(receipt)
    domain_review = patch.candidate_review
    if domain_review is None:  # pragma: no cover - patch invariant
        raise TextureWorkflowRuntimeError("Texture candidate review payload is absent")
    accepted: set[str] = set()
    for disposition in domain_review.unit_dispositions:
        if disposition.disposition == "accept":
            accepted.add(disposition.unit_id)
        else:
            accepted.discard(disposition.unit_id)
    accepted_ids = tuple(
        unit_id for unit_id in checkpoint.selected_unit_ids if unit_id in accepted
    )
    remaining_ids = tuple(
        unit_id for unit_id in checkpoint.selected_unit_ids if unit_id not in accepted
    )
    all_accepted = not remaining_ids and review.disposition == "accept"
    terminal_rejection = review.disposition == "reject"
    if terminal_rejection:
        accepted_ids = ()
        remaining_ids = checkpoint.selected_unit_ids
    state = state.model_copy(
        update={
            "current_candidate_review": artifact_reference(review),
            "current_candidate_domain_review": artifact_reference(
                domain_review_evidence
            ),
            "current_candidate_receipt": artifact_reference(receipt),
            "accepted_candidate_result": (
                artifact_reference(result) if all_accepted else None
            ),
            "accepted_candidate_domain_review": (
                artifact_reference(domain_review_evidence) if all_accepted else None
            ),
            "accepted_candidate_review": (
                artifact_reference(review) if all_accepted else None
            ),
            "accepted_candidate_receipt": (
                artifact_reference(receipt) if all_accepted else None
            ),
        }
    )
    event = _progress(
        checkpoint,
        phase="validating",
        message=(
            "Outer coordinator accepted the exact reviewed candidate."
            if all_accepted
            else (
                "Outer coordinator rejected the candidate; Texture is terminally "
                "closed to refinement and publication."
                if terminal_rejection
                else "Outer coordinator requested candidate revision; publication "
                "remains forbidden."
            )
        ),
    )
    if progress_callback is not None:
        progress_callback(event)
    return _save(
        checkpoint_store,
        checkpoint,
        next_action=(
            "done" if terminal_rejection else ("finalize" if all_accepted else "refine")
        ),
        state=state,
        progress=(*checkpoint.progress, event),
        accepted_unit_ids=accepted_ids,
        remaining_unit_ids=remaining_ids,
        pending_validation_unit_ids=(),
        terminal_status="conditional" if terminal_rejection else None,
    )


def _publish_candidate(
    checkpoint: TextureWorkflowCheckpoint,
    patch: TextureEmbeddedDecisionPatch,
    *,
    request: TextureWorkflowRequest,
    checkpoint_store: TextureWorkflowCheckpointStore,
    decision_store: EmbeddedDecisionArtifactStore,
    progress_callback: ProgressCallback | None,
) -> TextureWorkflowCheckpoint:
    state = _state(checkpoint)
    publication = patch.publication
    if publication is None:  # pragma: no cover - patch invariant
        raise TextureWorkflowRuntimeError("Texture publication payload is absent")
    expected_path = _publication_path(request.output_dir, publication.candidate_output)
    if Path(publication.publication_path).expanduser().resolve() != expected_path:
        raise TextureWorkflowRuntimeError(
            "Texture publication path is not the canonical accepted-candidate path"
        )
    verify_artifact_digests(checkpoint)
    verify_output_asset_digest(checkpoint)
    verify_validation_evidence_digests(checkpoint)
    if checkpoint.output_asset_sha256 != publication.candidate_output.sha256:
        raise TextureWorkflowRuntimeError(
            "Texture publication candidate differs from the reviewed checkpoint"
        )
    _closure_artifact, candidate_closure_facts = _load_candidate_dependency_closure(
        decision_store, state
    )
    decision, evidence = build_publication_decision(
        patch,
        store=decision_store,
        state=state,
    )
    decision_store.append(decision)
    authorization = authorize_publication(decision, evidence=evidence)

    def publish_leaf(
        _authorization: BoundedExecutionAuthorization,
    ) -> EmbeddedBoundedExecutionResult:
        published = _copy_candidate_once(
            publication.candidate_output,
            expected_path,
        )
        verification_artifacts, verification_facts = _verify_publication(
            request=request,
            plan=checkpoint.plan,
            published=published,
            output_dir=request.output_dir,
            candidate_closure_facts=candidate_closure_facts,
        )
        return publication_result_artifact(
            authorization,
            published_asset=published,
            verification_artifacts=verification_artifacts,
            verification_facts=verification_facts,
        )

    committed_result: EmbeddedBoundedExecutionResult | None = None
    try:
        result = decision_store.invoke_after_authorization_commit(
            authorization,
            publish_leaf,
        )
    except EmbeddedDecisionAuthorizationReplayError as exc:
        if exc.outcome.authorization != artifact_reference(authorization):
            raise TextureWorkflowRuntimeError(
                "Texture publication reconciliation names another authorization"
            ) from exc
        committed_result = _committed_result_for_authorization(
            decision_store,
            authorization,
        )
        if committed_result is not None:
            result = committed_result
        else:
            if expected_path.is_symlink():
                raise TextureWorkflowRuntimeError(
                    "Texture publication reconciliation rejects a symlinked output"
                )
            published = bind_file(expected_path)
            if (
                published.sha256 != publication.candidate_output.sha256
                or published.size_bytes != publication.candidate_output.size_bytes
            ):
                raise TextureWorkflowRuntimeError(
                    "Texture publication reconciliation found substituted output bytes"
                )
            verification_artifacts, verification_facts = _verify_publication(
                request=request,
                plan=checkpoint.plan,
                published=published,
                output_dir=request.output_dir,
                candidate_closure_facts=candidate_closure_facts,
            )
            result = publication_result_artifact(
                authorization,
                published_asset=published,
                verification_artifacts=verification_artifacts,
                verification_facts=verification_facts,
            )
    result_commit = decision_store.append_result(result)
    if committed_result is None and not result_commit.newly_committed:
        raise TextureWorkflowRuntimeError(
            "Texture publication result was committed by another execution"
        )
    published = _validate_publication_result_bindings(
        result,
        expected_path=expected_path,
        candidate=publication.candidate_output,
    )
    state = state.model_copy(
        update={
            "publication_decision": artifact_reference(decision),
            "publication_authorization": artifact_reference(authorization),
            "publication_result": artifact_reference(result),
            "publication_review": None,
            "completed_receipt": None,
        }
    )
    event = _progress(
        checkpoint,
        phase="finalizing",
        message=(
            "Exact accepted candidate published once; outer review of publication "
            "proof is still required."
        ),
    )
    if progress_callback is not None:
        progress_callback(event)
    return _save(
        checkpoint_store,
        checkpoint,
        next_action="review_publication",
        state=state,
        progress=(*checkpoint.progress, event),
        output_asset_path=published.path,
    )


def _review_publication(
    checkpoint: TextureWorkflowCheckpoint,
    patch: TextureEmbeddedDecisionPatch,
    *,
    request: TextureWorkflowRequest,
    checkpoint_store: TextureWorkflowCheckpointStore,
    decision_store: EmbeddedDecisionArtifactStore,
    finalizer: TextureWorkflowFinalizer,
    progress_callback: ProgressCallback | None,
) -> TextureFinalizationResult:
    state = _state(checkpoint)
    required = (
        state.publication_decision,
        state.publication_authorization,
        state.publication_result,
    )
    if any(reference is None for reference in required):
        raise TextureWorkflowRuntimeError(
            "Texture publication review lacks its exact mutation chain"
        )
    decision_ref, authorization_ref, result_ref = required
    assert decision_ref is not None
    assert authorization_ref is not None
    assert result_ref is not None
    decision = decision_store.load_typed(decision_ref, EmbeddedCoordinatorDecision)
    authorization = decision_store.load_typed(
        authorization_ref,
        BoundedExecutionAuthorization,
    )
    result = decision_store.load_typed(result_ref, EmbeddedBoundedExecutionResult)
    published = _validate_publication_result_bindings(result)
    if (
        checkpoint.output_asset_path != published.path
        or checkpoint.output_asset_sha256 != published.sha256
    ):
        raise TextureWorkflowRuntimeError(
            "Texture publication review checkpoint differs from committed bytes"
        )
    if state.completed_receipt is not None:
        review_ref = state.publication_review
        domain_review = patch.publication_review
        if review_ref is None or domain_review is None:
            raise TextureWorkflowRuntimeError(
                "Texture completed receipt lacks its exact publication review"
            )
        persisted_review = decision_store.load_typed(
            review_ref,
            EmbeddedCoordinatorReview,
        )
        if (
            domain_review.publication_result_sha256 != result_ref.sha256
            or domain_review.published_asset != published
            or domain_review.disposition != persisted_review.disposition
            or domain_review.findings != persisted_review.findings
        ):
            raise TextureWorkflowRuntimeError(
                "Texture publication recovery substituted the completed review"
            )
        result_index = _finalize(
            checkpoint,
            request=request,
            store=checkpoint_store,
            finalizer=finalizer,
        )
        _save(
            checkpoint_store,
            checkpoint,
            next_action="done",
            state=state,
            terminal_status="pass",
        )
        return result_index
    review = build_publication_review(patch, state=state, result=result)
    if review.disposition != "accept":
        raise TextureWorkflowRuntimeError(
            "Rejected Texture publication cannot complete or be republished"
        )
    decision_store.append_review(review)
    evidence, proposals = _load_decision_inputs(decision_store, decision)
    receipt = build_review_receipt(
        artifact_id=f"texture-completed-receipt-{patch.checkpoint_revision:04d}",
        decision=decision,
        authorization=authorization,
        result=result,
        review=review,
        evidence=evidence,
        proposals=proposals,
    )
    if receipt.receipt_status != "completed":
        raise TextureWorkflowRuntimeError(
            "Texture publication review did not produce a completed receipt"
        )
    decision_store.append_receipt(receipt)
    state = state.model_copy(
        update={
            "publication_review": artifact_reference(review),
            "completed_receipt": artifact_reference(receipt),
        }
    )
    reviewing = _progress(
        checkpoint,
        phase="reviewing_publication",
        message="Outer coordinator accepted exact publication proof.",
    )
    completed = _progress(
        checkpoint,
        phase="completed",
        message="All Texture targets and the exact publication are accepted.",
    )
    if progress_callback is not None:
        progress_callback(reviewing)
        progress_callback(completed)
    checkpoint = _save(
        checkpoint_store,
        checkpoint,
        next_action="review_publication",
        state=state,
        progress=(*checkpoint.progress, reviewing, completed),
    )
    result_index = _finalize(
        checkpoint,
        request=request,
        store=checkpoint_store,
        finalizer=finalizer,
    )
    _save(
        checkpoint_store,
        checkpoint,
        next_action="done",
        state=state,
        terminal_status="pass",
    )
    return result_index


def run_embedded_texture_workflow_step(
    request: TextureWorkflowRequest,
    *,
    mode: TextureWorkflowMode,
    client: TexturePlannerExecutorClient,
    validator: TextureSceneValidator,
    decision_patch: TextureEmbeddedDecisionPatch | None = None,
    finalizer: TextureWorkflowFinalizer | None = None,
    progress_callback: ProgressCallback | None = None,
    cancellation_check: CancellationCheck | None = None,
    checkpoint_store: TextureWorkflowCheckpointStore | None = None,
) -> TextureEmbeddedStepObservation | TextureFinalizationResult:
    """Advance one outer-owned Texture decision or bounded execution boundary."""

    context = request.execution_context
    if context is None or context.mode != "embedded":
        raise TextureWorkflowRuntimeError(
            "Embedded Texture workflow requires an embedded execution context"
        )
    output_dir = request.output_dir.expanduser().resolve()
    store = checkpoint_store or TextureWorkflowCheckpointStore(output_dir)
    finalizer_impl = finalizer or CanonicalTextureWorkflowFinalizer()
    with store.transaction(ledger_output_dir=output_dir):
        if not store.exists():
            if decision_patch is not None:
                raise TextureWorkflowRuntimeError(
                    "Initial embedded Texture inspection does not accept a decision"
                )
            checkpoint = _initialize(
                request,
                mode=mode,
                client=client,
                validator=validator,
                checkpoint_store=store,
                progress_callback=progress_callback,
            )
            return build_embedded_texture_observation(
                checkpoint,
                output_dir=output_dir,
            )
        checkpoint = store.load()
        validate_resume_identity(checkpoint, request=request, mode=mode)
        _state(checkpoint)
        decision_store = EmbeddedDecisionArtifactStore(output_dir)
        decision_store.journal()
        if checkpoint.next_action == "done":
            if decision_patch is not None:
                raise TextureWorkflowRuntimeError(
                    "Completed embedded Texture workflow rejects another decision"
                )
            return _finalize(
                checkpoint,
                request=request,
                store=store,
                finalizer=finalizer_impl,
            )
        if decision_patch is None:
            return build_embedded_texture_observation(
                checkpoint,
                output_dir=output_dir,
            )
        if cancellation_check is not None and cancellation_check():
            raise TextureWorkflowRuntimeError(
                "Embedded Texture cancellation stopped before decision execution"
            )
        try:
            ensure_embedded_patch_identity(
                decision_patch,
                request_digest=checkpoint.request_digest,
                source_identity_digest=checkpoint.source_identity_digest,
                proposal_plan_digest=checkpoint.plan_digest,
                checkpoint_decision_digest=texture_checkpoint_decision_digest(
                    checkpoint
                ),
                checkpoint_revision=checkpoint.revision,
                action=checkpoint.next_action,
                iteration=checkpoint.iteration,
            )
            if checkpoint.next_action in {"execute", "refine"}:
                checkpoint = _generate_and_assess_candidate(
                    checkpoint,
                    decision_patch,
                    request=request,
                    client=client,
                    validator=validator,
                    checkpoint_store=store,
                    decision_store=decision_store,
                    cancellation_check=cancellation_check,
                    progress_callback=progress_callback,
                )
                return build_embedded_texture_observation(
                    checkpoint,
                    output_dir=output_dir,
                )
            if checkpoint.next_action == "validate":
                checkpoint = _review_candidate(
                    checkpoint,
                    decision_patch,
                    request=request,
                    checkpoint_store=store,
                    decision_store=decision_store,
                    progress_callback=progress_callback,
                )
                return build_embedded_texture_observation(
                    checkpoint,
                    output_dir=output_dir,
                )
            if checkpoint.next_action == "finalize":
                checkpoint = _publish_candidate(
                    checkpoint,
                    decision_patch,
                    request=request,
                    checkpoint_store=store,
                    decision_store=decision_store,
                    progress_callback=progress_callback,
                )
                return build_embedded_texture_observation(
                    checkpoint,
                    output_dir=output_dir,
                )
            if checkpoint.next_action == "review_publication":
                return _review_publication(
                    checkpoint,
                    decision_patch,
                    request=request,
                    checkpoint_store=store,
                    decision_store=decision_store,
                    finalizer=finalizer_impl,
                    progress_callback=progress_callback,
                )
        except TextureWorkflowRuntimeError:
            raise
        except (OSError, RuntimeError, ValueError) as exc:
            raise TextureWorkflowRuntimeError(str(exc)) from exc
        raise TextureWorkflowRuntimeError(
            f"Unknown embedded Texture action: {checkpoint.next_action}"
        )


__all__ = [
    "build_embedded_texture_observation",
    "run_embedded_texture_workflow_step",
]
