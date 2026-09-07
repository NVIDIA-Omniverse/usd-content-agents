# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared interactive and batch entry points for one asset reasoning loop."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from filelock import FileLock, Timeout
from pydantic import BaseModel, ConfigDict, Field

from content_agent_workflows.common.domain_execution import DomainName
from content_agent_workflows.common.embedded_domain_decision import (
    EmbeddedDecisionIdentity,
)
from content_agent_workflows.common.usd_cli_session import (
    ParentUsdCliSessionIdentity,
)

from .models import (
    ArtifactBinding,
    AssetCompositionRun,
    AssetGeometryStageResult,
    StageName,
)
from .state import (
    AssetCompositionStateError,
    activate_single_reasoning_loop,
    begin_leaf,
    begin_stage,
    build_combined_report,
    build_embedded_domain_decision_identity,
    cancel_leaf,
    cancel_stage,
    complete_leaf,
    complete_stage,
    execute_geometry_stage,
    fail_leaf,
    fail_stage,
    finalize_graph_run,
    freeze_execution_graph,
    leaf_directory,
    load_run_state,
    load_verified_asset_request,
    load_verified_run,
    record_coordinator_evidence_review,
    record_coordinator_plan,
    recover_leaf,
    recover_stage,
    require_review,
    stage_directory,
)

AssetCoordinatorInvocationMode = Literal["interactive", "batch"]


@dataclass(frozen=True)
class AssetCoordinatorSession:
    """Typed tool surface presented to one long-running reasoning loop."""

    run_state_path: Path
    mode: AssetCoordinatorInvocationMode
    parent_usd_cli_session_identity: ParentUsdCliSessionIdentity | None = None
    parent_usd_cli_session_identity_artifact: ArtifactBinding | None = None

    def _verified_run(self) -> AssetCompositionRun:
        """Revalidate durable state and every frozen external input."""

        run = load_verified_run(self.run_state_path)
        load_verified_asset_request(self.run_state_path, run=run)
        return run

    def status(self) -> AssetCompositionRun:
        return self._verified_run()

    def _require_mode(
        self,
        selected_mode: Literal["agentic", "compatibility_fixed"],
    ) -> AssetCompositionRun:
        # This gate accepts no artifact claim. The dispatched state surface owns
        # authoritative verification at its mutation or artifact boundary.
        run = load_run_state(self.run_state_path)
        if run.selected_mode != selected_mode:
            raise AssetCompositionStateError(
                f"{selected_mode} transition is unavailable in {run.selected_mode} mode"
            )
        return run

    def stage_directory(self, stage: StageName) -> Path:
        """Return the canonical directory for the current stage attempt."""

        self._require_mode("compatibility_fixed")
        return stage_directory(self.run_state_path, stage)

    def freeze_graph(
        self,
        graph_path: str | Path,
        *,
        actor: str = "asset-coordinator",
    ) -> AssetCompositionRun:
        """Bind exactly one graph supplied by this session's sole reasoner."""

        self._require_mode("agentic")
        return freeze_execution_graph(
            self.run_state_path,
            graph_path=graph_path,
            actor=actor,
        )

    def leaf_directory(self, leaf_id: str, *, attempt: int | None = None) -> Path:
        """Return the canonical directory for one selected leaf attempt."""

        self._require_mode("agentic")
        return leaf_directory(self.run_state_path, leaf_id, attempt=attempt)

    def begin_leaf(
        self,
        leaf_id: str,
        *,
        actor: str = "asset-coordinator",
    ) -> AssetCompositionRun:
        self._require_mode("agentic")
        return begin_leaf(self.run_state_path, leaf_id, actor=actor)

    def complete_leaf(
        self,
        leaf_id: str,
        *,
        invocation_path: str | Path,
        result_path: str | Path,
        native_terminal_receipt_path: str | Path | None = None,
        operation_index_paths: Sequence[str | Path] = (),
        evidence_index_paths: Sequence[str | Path] = (),
        evidence_paths: Sequence[str | Path] = (),
        saved_stage_readback_paths: Sequence[str | Path] = (),
        native_disposition: Literal["passed", "not_evaluated"] | None = None,
        resource_claims: Sequence[str] = (),
        resource_release_paths: Sequence[str | Path] = (),
        summary: str | None = None,
        actor: str = "asset-coordinator",
    ) -> AssetCompositionRun:
        self._require_mode("agentic")
        return complete_leaf(
            self.run_state_path,
            leaf_id,
            invocation_path=invocation_path,
            result_path=result_path,
            native_terminal_receipt_path=native_terminal_receipt_path,
            operation_index_paths=operation_index_paths,
            evidence_index_paths=evidence_index_paths,
            evidence_paths=evidence_paths,
            saved_stage_readback_paths=saved_stage_readback_paths,
            native_disposition=native_disposition,
            resource_claims=resource_claims,
            resource_release_paths=resource_release_paths,
            summary=summary,
            actor=actor,
        )

    def fail_leaf(
        self,
        leaf_id: str,
        *,
        reason: str,
        invocation_path: str | Path | None = None,
        result_path: str | Path | None = None,
        native_terminal_receipt_path: str | Path | None = None,
        operation_index_paths: Sequence[str | Path] = (),
        evidence_index_paths: Sequence[str | Path] = (),
        evidence_paths: Sequence[str | Path] = (),
        saved_stage_readback_paths: Sequence[str | Path] = (),
        resource_claims: Sequence[str] = (),
        resource_release_paths: Sequence[str | Path] = (),
        actor: str = "asset-coordinator",
    ) -> AssetCompositionRun:
        self._require_mode("agentic")
        return fail_leaf(
            self.run_state_path,
            leaf_id,
            reason=reason,
            invocation_path=invocation_path,
            result_path=result_path,
            native_terminal_receipt_path=native_terminal_receipt_path,
            operation_index_paths=operation_index_paths,
            evidence_index_paths=evidence_index_paths,
            evidence_paths=evidence_paths,
            saved_stage_readback_paths=saved_stage_readback_paths,
            resource_claims=resource_claims,
            resource_release_paths=resource_release_paths,
            actor=actor,
        )

    def cancel_leaf(
        self,
        leaf_id: str,
        *,
        reason: str,
        invocation_path: str | Path | None = None,
        result_path: str | Path | None = None,
        native_terminal_receipt_path: str | Path | None = None,
        operation_index_paths: Sequence[str | Path] = (),
        evidence_index_paths: Sequence[str | Path] = (),
        evidence_paths: Sequence[str | Path] = (),
        saved_stage_readback_paths: Sequence[str | Path] = (),
        resource_claims: Sequence[str] = (),
        resource_release_paths: Sequence[str | Path] = (),
        actor: str = "asset-coordinator",
    ) -> AssetCompositionRun:
        self._require_mode("agentic")
        return cancel_leaf(
            self.run_state_path,
            leaf_id,
            reason=reason,
            invocation_path=invocation_path,
            result_path=result_path,
            native_terminal_receipt_path=native_terminal_receipt_path,
            operation_index_paths=operation_index_paths,
            evidence_index_paths=evidence_index_paths,
            evidence_paths=evidence_paths,
            saved_stage_readback_paths=saved_stage_readback_paths,
            resource_claims=resource_claims,
            resource_release_paths=resource_release_paths,
            actor=actor,
        )

    def recover_leaf(
        self,
        leaf_id: str,
        *,
        reason: str,
        actor: str = "asset-coordinator",
    ) -> AssetCompositionRun:
        self._require_mode("agentic")
        return recover_leaf(self.run_state_path, leaf_id, reason=reason, actor=actor)

    def finalize_graph(
        self,
        *,
        parent_release_receipt_path: str | Path | None = None,
        parent_command_receipt_journal_path: str | Path | None = None,
        parent_command_receipt_checkpoint_path: str | Path | None = None,
        actor: str = "asset-coordinator",
    ) -> AssetCompositionRun:
        self._require_mode("agentic")
        if (
            self.parent_usd_cli_session_identity is not None
            or self.parent_usd_cli_session_identity_artifact is not None
        ):
            raise AssetCompositionStateError(
                "Launcher-bound interactive reasoning must return at "
                "finalize_receipts; the launcher releases parent resources "
                "before graph finalization"
            )
        return finalize_graph_run(
            self.run_state_path,
            parent_release_receipt_path=parent_release_receipt_path,
            parent_command_receipt_journal_path=(parent_command_receipt_journal_path),
            parent_command_receipt_checkpoint_path=(
                parent_command_receipt_checkpoint_path
            ),
            actor=actor,
        )

    def record_plan(
        self,
        plan_path: str | Path,
        *,
        actor: str = "asset-coordinator",
    ) -> AssetCompositionRun:
        self._require_mode("compatibility_fixed")
        return record_coordinator_plan(
            self.run_state_path,
            plan_path=plan_path,
            actor=actor,
        )

    def begin_stage(
        self,
        stage: StageName,
        *,
        actor: str = "asset-coordinator",
    ) -> AssetCompositionRun:
        self._require_mode("compatibility_fixed")
        return begin_stage(self.run_state_path, stage, actor=actor)

    def execute_geometry(
        self,
        *,
        actor: str = "asset-geometry-executor",
    ) -> AssetGeometryStageResult:
        """Execute the active deterministic Geometry stage without nested reasoning."""

        return execute_geometry_stage(self.run_state_path, actor=actor)

    def embedded_decision_identity(
        self,
        *,
        domain: DomainName,
        input_asset: str | Path,
        output_dir: str | Path,
        capability_digests: Mapping[str, str],
        implementation_digests: Mapping[str, str],
        configuration_digests: Mapping[str, str] | None = None,
    ) -> EmbeddedDecisionIdentity:
        """Return the exact identity future embedded domain leaves must use."""

        self._require_mode("compatibility_fixed")
        return build_embedded_domain_decision_identity(
            self.run_state_path,
            domain=domain,
            input_asset=input_asset,
            output_dir=output_dir,
            capability_digests=capability_digests,
            implementation_digests=implementation_digests,
            configuration_digests=configuration_digests,
        )

    def review_evidence(
        self,
        review_path: str | Path,
        *,
        actor: str = "asset-coordinator",
    ) -> AssetCompositionRun:
        self._require_mode("compatibility_fixed")
        return record_coordinator_evidence_review(
            self.run_state_path,
            review_path=review_path,
            actor=actor,
        )

    def require_articulation_review(
        self,
        candidates_path: str | Path,
        *,
        actor: str = "asset-coordinator",
    ) -> AssetCompositionRun:
        self._require_mode("compatibility_fixed")
        return require_review(
            self.run_state_path,
            candidates_path=candidates_path,
            actor=actor,
        )

    def complete_stage(
        self,
        stage: StageName,
        *,
        output_asset: str | Path,
        evidence_paths: Sequence[str | Path],
        summary: str,
        actor: str = "asset-coordinator",
    ) -> AssetCompositionRun:
        self._require_mode("compatibility_fixed")
        return complete_stage(
            self.run_state_path,
            stage,
            output_asset=output_asset,
            evidence_paths=evidence_paths,
            summary=summary,
            actor=actor,
        )

    def fail_stage(
        self,
        stage: StageName,
        *,
        reason: str,
        actor: str = "asset-coordinator",
    ) -> AssetCompositionRun:
        """Durably stop when an executor cannot produce reviewable evidence."""

        self._require_mode("compatibility_fixed")
        return fail_stage(
            self.run_state_path,
            stage,
            reason=reason,
            actor=actor,
        )

    def cancel_stage(
        self,
        stage: StageName,
        *,
        reason: str,
        actor: str = "asset-coordinator",
    ) -> AssetCompositionRun:
        """Durably record an operator or coordinator cancellation."""

        self._require_mode("compatibility_fixed")
        return cancel_stage(
            self.run_state_path,
            stage,
            reason=reason,
            actor=actor,
        )

    def recover_stage(
        self,
        stage: StageName,
        *,
        reason: str,
        actor: str = "asset-coordinator",
    ) -> AssetCompositionRun:
        """Reopen the current stopped stage after its cause is corrected."""

        self._require_mode("compatibility_fixed")
        return recover_stage(
            self.run_state_path,
            stage,
            reason=reason,
            actor=actor,
        )

    def build_combined_report(
        self,
        *,
        final_asset: str | Path,
        validation_summary: str | Path,
        output_path: str | Path,
    ) -> ArtifactBinding:
        self._require_mode("compatibility_fixed")
        return build_combined_report(
            self.run_state_path,
            final_asset=final_asset,
            validation_summary=validation_summary,
            output_path=output_path,
        )


class AssetCoordinatorLoopResult(BaseModel):
    """Observable result from the shared coordinator invocation boundary."""

    model_config = ConfigDict(extra="forbid")

    mode: AssetCoordinatorInvocationMode
    returncode: int = Field(
        ge=0,
        description=(
            "Accepted reasoning-loop status; negative raw process statuses are "
            "rejected before result construction"
        ),
    )
    run: AssetCompositionRun


AssetReasoningLoop = Callable[[AssetCoordinatorSession], int | None]


class AssetCoordinatorLeaseError(AssetCompositionStateError):
    """Raised when another reasoning loop already owns the composed run."""


def _coordinator_lease_path(state_path: Path) -> Path:
    """Place the lease outside the child-writable composed-run directory."""

    resolved = state_path.expanduser().resolve()
    run_dir = resolved.parent
    identity = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:16]
    return run_dir.parent / f".{run_dir.name}.{identity}.asset-coordinator.lock"


def _acquire_coordinator_lease(state_path: Path) -> FileLock:
    lease_path = _coordinator_lease_path(state_path)
    # FileLock ownership is OS-backed. The pathname may remain after release or
    # process termination, but an inert file does not retain the lease.
    lease = FileLock(str(lease_path))
    try:
        lease.acquire(timeout=0)
    except Timeout as exc:
        raise AssetCoordinatorLeaseError(
            "Another asset coordinator reasoning loop already owns this run"
        ) from exc
    return lease


def run_asset_coordinator_transition[TransitionResultT](
    run_state_path: str | Path,
    *,
    transition: Callable[[], TransitionResultT],
) -> TransitionResultT:
    """Apply one operator transition while excluding the active reasoning loop."""

    state_path = Path(run_state_path).expanduser().resolve()
    lease = _acquire_coordinator_lease(state_path)
    try:
        run = load_verified_run(state_path)
        load_verified_asset_request(state_path, run=run)
        result = transition()
        final = load_verified_run(state_path)
        load_verified_asset_request(state_path, run=final)
        return result
    finally:
        lease.release()


def run_asset_coordinator(
    run_state_path: str | Path,
    *,
    mode: AssetCoordinatorInvocationMode,
    reasoning_loop: AssetReasoningLoop,
) -> AssetCoordinatorLoopResult:
    """Invoke exactly one reasoning loop over the shared durable coordinator."""

    state_path = Path(run_state_path).expanduser().resolve()
    lease = _acquire_coordinator_lease(state_path)
    try:
        initial = load_verified_run(state_path)
        load_verified_asset_request(state_path, run=initial)
        if initial.coordinator.mode == "legacy":
            initial = activate_single_reasoning_loop(state_path)
        if initial.coordinator.mode != "single_reasoning_loop":  # pragma: no cover
            raise AssetCompositionStateError("Unsupported asset coordinator mode")
        expected_coordinator_mode = initial.coordinator.mode
        session = AssetCoordinatorSession(run_state_path=state_path, mode=mode)
        returncode = reasoning_loop(session)
        if returncode is None:
            returncode = 0
        if returncode < 0:
            raise AssetCompositionStateError(
                "Asset coordinator reasoning loop returned a negative status"
            )
        final = load_verified_run(state_path)
        load_verified_asset_request(state_path, run=final)
        if final.coordinator.mode != expected_coordinator_mode:
            raise AssetCompositionStateError(
                "Asset coordinator mode changed during the reasoning-loop invocation"
            )
        if final.coordinator.next_action not in {
            "await_human_review",
            "finalize_receipts",
            "stopped",
            "terminal",
        }:
            raise AssetCompositionStateError(
                "Asset coordinator reasoning loop returned before reaching an "
                "await_human_review, finalize_receipts, stopped, or terminal "
                "boundary; "
                f"next_action={final.coordinator.next_action}"
            )
        return AssetCoordinatorLoopResult(
            mode=mode,
            returncode=returncode,
            run=final,
        )
    finally:
        lease.release()


def run_interactive_asset_coordinator(
    run_state_path: str | Path,
    *,
    reasoning_loop: AssetReasoningLoop,
) -> AssetCoordinatorLoopResult:
    """Run the shared coordinator from an interactive ``agentic/`` session."""

    return run_asset_coordinator(
        run_state_path,
        mode="interactive",
        reasoning_loop=reasoning_loop,
    )


def run_batch_asset_coordinator(
    run_state_path: str | Path,
    *,
    reasoning_loop: AssetReasoningLoop,
) -> AssetCoordinatorLoopResult:
    """Run the same coordinator behind ``content-workflow-cli asset``."""

    return run_asset_coordinator(
        run_state_path,
        mode="batch",
        reasoning_loop=reasoning_loop,
    )


__all__ = [
    "AssetCoordinatorInvocationMode",
    "AssetCoordinatorLeaseError",
    "AssetCoordinatorLoopResult",
    "AssetCoordinatorSession",
    "AssetReasoningLoop",
    "run_asset_coordinator",
    "run_asset_coordinator_transition",
    "run_batch_asset_coordinator",
    "run_interactive_asset_coordinator",
]
