# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic finalization for bounded agentic texture workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel

from .models import (
    TextureFinalizationResult,
    TextureFinalizerInput,
    TexturePlanDocument,
    TextureWorkflowRequest,
    TextureWorkflowValidationEvidence,
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_payload(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path.resolve()


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
    """Write the canonical mock workflow artifacts under the request run dir."""

    def finalize(self, payload: TextureFinalizerInput) -> TextureFinalizationResult:
        output_dir = payload.request.output_dir.resolve()
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
            evidence_path
            for validation in payload.validations
            for finding in validation.findings
            for evidence_path in finding.evidence_artifact_paths
        )
        validation_evidence = TextureWorkflowValidationEvidence(
            target_runtime=payload.request.target_runtime,
            status="pass" if not payload.remaining_unit_ids else "conditional",
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
            output_asset_path=payload.output_asset_path,
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
                "schema_version": "content-agent-workflows.texture-progress-log.v1",
                "events": [item.model_dump(mode="json") for item in payload.progress],
            },
        )

        success = not payload.remaining_unit_ids
        summary_payload = {
            "schema_version": "content-agent-workflows.texture-summary.v1",
            "status": "pass" if success else "conditional",
            "mode": payload.mode,
            "source_asset": payload.request.source_asset,
            "output_asset_path": payload.output_asset_path,
            "selected_unit_ids": payload.plan.selected_unit_ids,
            "accepted_unit_ids": payload.accepted_unit_ids,
            "remaining_unit_ids": payload.remaining_unit_ids,
            "artifacts": {
                "request": str(request_path),
                "texture_plan": str(plan_path),
                "execution_summary": str(execution_summary_path),
                "visual_quality_assessment": str(vqa_path),
                "validation_evidence": str(validation_evidence_path),
                "workflow_progress": str(progress_path),
            },
        }
        final_summary_path = _write_json(
            output_dir / "final_summary.json", summary_payload
        )
        return TextureFinalizationResult(
            success=success,
            status="pass" if success else "conditional",
            mode=payload.mode,
            output_dir=str(output_dir),
            output_asset_path=payload.output_asset_path,
            accepted_unit_ids=payload.accepted_unit_ids,
            remaining_unit_ids=payload.remaining_unit_ids,
            request_path=str(request_path),
            texture_plan_path=str(plan_path),
            execution_summary_path=str(execution_summary_path),
            visual_quality_assessment_path=str(vqa_path),
            validation_evidence_path=str(validation_evidence_path),
            workflow_progress_path=str(progress_path),
            final_summary_path=str(final_summary_path),
        )
