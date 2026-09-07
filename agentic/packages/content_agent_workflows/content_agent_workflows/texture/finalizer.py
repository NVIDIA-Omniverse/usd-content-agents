# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic finalization for bounded agentic texture workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel

from content_agent_workflows.common.artifacts import (
    atomic_write_json,
    file_sha256,
    load_json,
)

from .decision import verify_texture_decision_ledger
from .models import (
    TextureFinalizationResult,
    TextureFinalizerInput,
    TexturePlanDocument,
    TextureWorkflowRequest,
    TextureWorkflowValidationEvidence,
)
from .runtime import (
    TextureWorkflowCheckpoint,
    texture_plan_digest,
    texture_request_digest,
    texture_source_identity_digest,
)


class TextureWorkflowFinalizer(Protocol):
    """Finalizer boundary shared by interactive and batch wrappers."""

    def finalize(self, payload: TextureFinalizerInput) -> TextureFinalizationResult:
        """Write canonical workflow artifacts and return their index."""


def _json_payload(value: BaseModel | dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return value


def _write_json(path: Path, payload: BaseModel | dict[str, Any]) -> Path:
    return atomic_write_json(path, _json_payload(payload))


def write_texture_planning_artifacts(
    request: TextureWorkflowRequest,
    plan: TexturePlanDocument,
) -> tuple[Path, Path]:
    """Persist the request and immutable plan before executor work begins."""

    output_dir = request.output_dir.resolve()
    request_path = _write_json(output_dir / "request.json", request)
    plan_path = _write_json(output_dir / "texture_plan.json", plan)
    return request_path, plan_path


class CanonicalTextureWorkflowFinalizer:
    """Write canonical artifacts backed by mandatory durable checkpoint evidence."""

    def finalize(self, payload: TextureFinalizerInput) -> TextureFinalizationResult:
        output_dir = payload.request.output_dir.resolve()
        checkpoint_path = Path(
            payload.workflow_checkpoint_path or output_dir / "workflow_checkpoint.json"
        ).expanduser()
        checkpoint_path = checkpoint_path.resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Texture workflow checkpoint does not exist: {checkpoint_path}"
            )
        checkpoint = TextureWorkflowCheckpoint.model_validate(
            load_json(checkpoint_path)
        )
        decision_ledger_path: Path | None = None
        if payload.decision_ledger_path is not None:
            decision_ledger_path = (
                Path(payload.decision_ledger_path).expanduser().resolve()
            )
            verify_texture_decision_ledger(
                decision_ledger_path,
                output_dir=output_dir,
                request_digest=texture_request_digest(payload.request),
                source_identity_digest=texture_source_identity_digest(
                    payload.request,
                    plan=payload.plan,
                ),
                plan_digest=texture_plan_digest(payload.plan),
                require_final_decision=payload.terminal_status != "cancelled",
                checkpoint=checkpoint,
            )
        request_path, plan_path = write_texture_planning_artifacts(
            payload.request, payload.plan
        )
        execution_summary_path = _write_json(
            output_dir / "texture_execution_summary.json",
            {
                "schema_version": "content-agent-workflows.texture-execution.v1",
                "executions": [
                    execution.model_dump(mode="json")
                    for execution in payload.executions
                ],
                "unit_artifacts": {
                    unit_id: artifact.model_dump(mode="json")
                    for unit_id, artifact in payload.unit_artifacts.items()
                },
            },
        )
        vqa_path = _write_json(
            output_dir / "visual_quality_assessment.json",
            {
                "schema_version": "content-agent-workflows.texture-vqa.v1",
                "passes": [
                    validation.model_dump(mode="json")
                    for validation in payload.validations
                ],
                "accepted_unit_ids": payload.accepted_unit_ids,
                "remaining_unit_ids": payload.remaining_unit_ids,
            },
        )
        visual_evidence_paths = tuple(
            dict.fromkeys(
                evidence_path
                for validation in payload.validations
                for finding in validation.findings
                for evidence_path in finding.evidence_artifact_paths
            )
        )
        evidence_output_path = payload.output_asset_path
        output_asset_sha256 = payload.output_asset_sha256
        if evidence_output_path is not None:
            output_path = Path(evidence_output_path).expanduser().resolve()
            if output_path.is_file():
                evidence_output_path = str(output_path)
                observed_output_sha256 = file_sha256(output_path)
                if (
                    output_asset_sha256 is not None
                    and output_asset_sha256 != observed_output_sha256
                ):
                    raise ValueError(
                        "Texture finalizer output digest differs from current bytes"
                    )
                output_asset_sha256 = observed_output_sha256
            elif payload.terminal_status != "cancelled":
                raise FileNotFoundError(
                    f"Texture output asset does not exist: {output_path}"
                )
            else:
                evidence_output_path = None
                output_asset_sha256 = None
        validation_evidence = TextureWorkflowValidationEvidence(
            target_runtime=payload.request.target_runtime,
            status=payload.terminal_status,
            selected_unit_ids=payload.plan.selected_unit_ids,
            accepted_unit_ids=payload.accepted_unit_ids,
            remaining_unit_ids=payload.remaining_unit_ids,
            selected_unit_count=len(payload.plan.selected_unit_ids),
            backend_job_count=sum(
                len(execution.requested_unit_ids) for execution in payload.executions
            ),
            cache_hit_count=sum(
                len(execution.cache_hit_unit_ids) for execution in payload.executions
            ),
            retry_count=sum(execution.retry_count for execution in payload.executions),
            output_asset_path=evidence_output_path,
            output_asset_sha256=output_asset_sha256,
            unit_artifact_paths={
                unit_id: artifact.artifact_paths
                for unit_id, artifact in payload.unit_artifacts.items()
            },
            visual_evidence_paths=visual_evidence_paths,
        )
        validation_evidence_path = _write_json(
            output_dir / "validation_evidence.json", validation_evidence
        )
        progress_path = _write_json(
            output_dir / "workflow_progress.json",
            {
                "schema_version": "content-agent-workflows.texture-progress-log.v2",
                "events": [item.model_dump(mode="json") for item in payload.progress],
            },
        )

        success = payload.terminal_status == "pass"
        summary_payload = {
            "schema_version": "content-agent-workflows.texture-summary.v3",
            "status": payload.terminal_status,
            "mode": payload.mode,
            "source_asset": payload.request.source_asset,
            "output_asset_path": evidence_output_path,
            "output_asset_sha256": validation_evidence.output_asset_sha256,
            "selected_unit_ids": payload.plan.selected_unit_ids,
            "accepted_unit_ids": payload.accepted_unit_ids,
            "remaining_unit_ids": payload.remaining_unit_ids,
            "cancellation_reason": payload.cancellation_reason,
            "artifacts": {
                "request": str(request_path),
                "texture_plan": str(plan_path),
                "execution_summary": str(execution_summary_path),
                "visual_quality_assessment": str(vqa_path),
                "validation_evidence": str(validation_evidence_path),
                "workflow_progress": str(progress_path),
                "workflow_checkpoint": str(checkpoint_path),
                **(
                    {"decision_ledger": str(decision_ledger_path)}
                    if decision_ledger_path is not None
                    else {}
                ),
                **(
                    {
                        "embedded_decision_receipt": (
                            payload.embedded_decision_receipt_path
                        )
                    }
                    if payload.embedded_decision_receipt_path is not None
                    else {}
                ),
            },
        }
        final_summary_path = _write_json(
            output_dir / "final_summary.json", summary_payload
        )
        return TextureFinalizationResult(
            success=success,
            status=payload.terminal_status,
            mode=payload.mode,
            output_dir=str(output_dir),
            output_asset_path=evidence_output_path,
            accepted_unit_ids=payload.accepted_unit_ids,
            remaining_unit_ids=payload.remaining_unit_ids,
            cancellation_reason=payload.cancellation_reason,
            request_path=str(request_path),
            texture_plan_path=str(plan_path),
            execution_summary_path=str(execution_summary_path),
            visual_quality_assessment_path=str(vqa_path),
            validation_evidence_path=str(validation_evidence_path),
            workflow_progress_path=str(progress_path),
            workflow_checkpoint_path=str(checkpoint_path),
            decision_ledger_path=(
                str(decision_ledger_path) if decision_ledger_path is not None else None
            ),
            embedded_decision_receipt_path=(payload.embedded_decision_receipt_path),
            final_summary_path=str(final_summary_path),
        )
