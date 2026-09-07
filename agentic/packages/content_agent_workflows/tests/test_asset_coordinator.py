# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared single-loop composed-asset coordinator."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path

import content_workflow_cli.cli as workflow_cli_module
import pytest
from content_workflow_cli.cli import main as workflow_cli_main
from filelock import Timeout
from PIL import Image
from world_understanding.utils.usd.package import write_usdz_package_from_directory
from world_understanding.validation import (
    ValidationIssue,
    ValidationPlan,
    ValidationPlanStep,
    ValidationRequest,
    ValidationResult,
    ValidationTemplateContext,
    ValidationTemplateResult,
)

import content_agent_workflows.articulation as articulation_api
import content_agent_workflows.asset_composition.coordinator as asset_coordinator
import content_agent_workflows.asset_composition.state as asset_state
import content_agent_workflows.validation.embedded_assessment as validation_assessment
import content_agent_workflows.validation.workflow as validation_workflow_module
from content_agent_workflows.articulation import (
    SKILL_ROUTED_DECISION_METADATA_KEY,
    ArticulationAuthoringRequest,
    ArticulationAuthoringResult,
    ArticulationCandidateDecision,
    ArticulationDecisionPatch,
    ArticulationInferenceResult,
    ArticulationReviewEntry,
    ArticulationReviewReceipt,
    ArticulationRunState,
    ArticulationSceneCandidateEvidence,
    ArticulationSceneEvidenceIdentity,
    ArticulationSceneEvidenceResult,
    ArticulationSceneRenderEvidence,
    ArticulationValidationResult,
    ArticulationWorkflowRequest,
    MembershipDispositionDocument,
    MembershipDispositionRecord,
    MembershipDispositionSummary,
    Stage2ArticulationCandidate,
    Stage2CandidateDocument,
    Stage2CandidateSummary,
    apply_articulation_decision_patch,
    build_articulation_step_observation,
    write_articulation_workflow_summary,
)
from content_agent_workflows.articulation import (
    ArtifactBinding as ArticulationArtifactBinding,
)
from content_agent_workflows.asset_composition import (
    LEGACY_STAGE_ORDER,
    ArtifactBinding,
    AssetCompositionRun,
    AssetCompositionStateError,
    AssetCoordinatorLoopResult,
    AssetCoordinatorSession,
    AssetCoordinatorState,
    AssetCrossStageClaim,
    AssetCrossStageValidation,
    StageName,
    begin_stage,
    build_combined_report,
    build_embedded_domain_decision_identity,
    build_embedded_domain_execution_context,
    complete_stage,
    create_run,
    fail_stage,
    load_verified_run,
    record_coordinator_evidence_review,
    record_coordinator_plan,
    record_review_decisions,
    recover_stage,
    require_review,
    run_batch_asset_coordinator,
    run_interactive_asset_coordinator,
    stage_directory,
    validate_terminal,
)
from content_agent_workflows.asset_composition.cli import main as state_cli_main
from content_agent_workflows.common.artifacts import atomic_write_json, file_sha256
from content_agent_workflows.common.domain_execution import (
    DOMAIN_EXECUTION_CONTEXT_METADATA_KEY,
    ExecutionArtifactBinding,
    metadata_with_domain_execution_context,
)
from content_agent_workflows.common.embedded_domain_artifact_store import (
    EmbeddedDecisionArtifactStore,
)
from content_agent_workflows.common.embedded_domain_decision import (
    EmbeddedDomainEvidence,
    ProviderNeutralEvidenceRecord,
    artifact_reference,
    canonical_json_digest,
)
from content_agent_workflows.common.validation_evidence import (
    CheckStatus,
    EvidenceArtifact,
    material_assignment_validation_evidence,
    physics_validation_evidence,
)
from content_agent_workflows.texture import (
    TEXTURE_EMBEDDED_DECISION_IDENTITY_METADATA_KEY,
    TextureAcceptanceCriteria,
    TextureArtifactDigest,
    TextureCandidateReviewDecision,
    TextureCanonicalPlan,
    TextureDecisionPatch,
    TextureEmbeddedDecisionPatch,
    TextureEmbeddedDecisionState,
    TextureExecutionResult,
    TextureGeneratorInputs,
    TextureInspectionResult,
    TextureInspectionUnit,
    TexturePlanCounts,
    TexturePlanDecision,
    TexturePlanDocument,
    TexturePlanSelectedUnit,
    TexturePlanTarget,
    TexturePreservationConstraints,
    TexturePublicationDecision,
    TexturePublicationReviewDecision,
    TextureUnitArtifact,
    TextureUnitDisposition,
    TextureValidationFinding,
    TextureValidationResult,
    TextureWorkflowCheckpoint,
    TextureWorkflowRequest,
    TextureWorkflowValidationEvidence,
    record_texture_decision_patch,
    texture_embedded_capability_digests,
    texture_embedded_implementation_digests,
    texture_plan_digest,
    texture_request_digest,
    texture_source_identity_digest,
)
from content_agent_workflows.texture.embedded_decision import (
    authorize_candidate_generation,
    authorize_publication,
    build_candidate_review,
    build_plan_decision,
    build_publication_decision,
    build_publication_review,
    build_review_receipt,
    candidate_evidence_artifact,
    candidate_result_artifact,
    candidate_review_evidence_artifact,
    initialize_embedded_texture_decisions,
    publication_result_artifact,
    texture_candidate_dependency_closure_facts,
    validate_completed_texture_decision_chain,
)
from content_agent_workflows.validation import (
    CANONICAL_VALIDATION_ASSESSMENT_NAME,
    EMBEDDED_VALIDATION_EVIDENCE_INDEX_NAME,
    EMBEDDED_VALIDATION_EXECUTION_INDEX_NAME,
    EMBEDDED_VALIDATION_RECEIPT_INDEX_NAME,
    VALIDATION_OPERATION_INDEX_NAME,
    VALIDATION_TERMINAL_RECEIPT_NAME,
    VALIDATION_WORKFLOW_EVIDENCE_SCHEMA_VERSION,
    VALIDATION_WORKFLOW_SUMMARY_SCHEMA_VERSION,
    EmbeddedValidationAssessmentError,
    ValidationAcceptedTemplateResult,
    ValidationArtifactIdentity,
    ValidationAssessmentFinding,
    ValidationCoordinatorAssessment,
    ValidationCoordinatorReviewDraft,
    ValidationGateAssessment,
    ValidationWorkflowCheckpoint,
    ValidationWorkflowIdentity,
    ValidationWorkflowRun,
    ValidationWorkItemRecord,
    ValidationWorkItemState,
    assess_embedded_validation,
    load_embedded_validation_run,
    prepare_embedded_validation_evidence,
    review_embedded_validation_assessment,
    validate_completed_embedded_validation_receipt,
    validate_coordinator_assessment,
)
from content_agent_workflows.validation.models import (
    validation_workflow_identity_digest,
)


class _EmbeddedValidationCliExecutor:
    """Deterministic dependency seam for the real embedded CLI workflow."""

    def __init__(self, base_dir: Path) -> None:
        del base_dir

    @property
    def template_versions(self) -> Mapping[str, str]:
        return {
            "render_valid": "test.render-valid.v1",
            "look_right": "test.look-right.v1",
        }

    def plan(
        self,
        request: ValidationRequest,
        *,
        working_dir: Path,
    ) -> ValidationPlan:
        del working_dir
        return ValidationPlan(
            steps=tuple(
                ValidationPlanStep(
                    template_name=template_name,
                    reason="Exercise the real embedded Validation workflow.",
                )
                for template_name in request.requested_templates
            )
        )

    def run(
        self,
        template_name: str,
        context: ValidationTemplateContext,
    ) -> ValidationTemplateResult:
        context.working_dir.mkdir(parents=True, exist_ok=True)
        evidence = context.working_dir / f"{template_name}.png"
        Image.new("RGB", (8, 8), (40, 80, 120)).save(evidence, format="PNG")
        if template_name == "look_right":
            return ValidationTemplateResult(
                template_name=template_name,
                status="failed",
                issues=(
                    ValidationIssue(
                        code="validation.visual_mismatch",
                        severity="fail",
                        message="Judge proposes a visual mismatch.",
                        template_name=template_name,
                    ),
                ),
                metrics={"issue_count": 1, "vlm_invoked": True},
                evidence={"image_paths": [str(evidence)]},
                metadata={"judge_output_role": "critique_proposal"},
            )
        return ValidationTemplateResult(
            template_name=template_name,
            status="passed",
            evidence={"image_paths": [str(evidence)]},
            metadata={
                "deterministic_template": True,
                "runtime_render": {
                    "status": "passed",
                    "backend": "fake",
                    "image_paths": [str(evidence)],
                    "render_response": None,
                    "render_output_dir": str(context.working_dir),
                    "issues": [],
                    "metadata": {},
                },
                "adapter_result": {
                    "status": "pass",
                    "verdict": "pass",
                    "issues": [],
                },
            },
        )


def _create(
    tmp_path: Path,
    name: str = "run",
    *,
    legacy: bool = False,
    historical_runtime: bool = False,
    physics_validation_mode: str = "runtime_required",
) -> Path:
    run_dir = tmp_path / name
    run_dir.mkdir()
    (run_dir / "raw").mkdir()
    source = tmp_path / f"{name}-source.usda"
    source.write_text('#usda 1.0\ndef Xform "World" {}\n', encoding="utf-8")
    joint_config = tmp_path / f"{name}-joint.yaml"
    joint_config.write_text("review_policy: all\n", encoding="utf-8")
    materials_yaml = tmp_path / f"{name}-materials.yaml"
    materials_yaml.write_text("materials: []\n", encoding="utf-8")
    materials_usd = tmp_path / f"{name}-materials.usda"
    materials_usd.write_text("#usda 1.0\n", encoding="utf-8")
    request = run_dir / "request.json"
    request.write_text(
        json.dumps(
            {
                "schema_version": "content-agents.asset-composition-request.v1",
                "created_at": "2026-08-03T00:00:00+00:00",
                "workflow": "asset.run",
                **({} if legacy else {"coordinator_mode": "single_reasoning_loop"}),
                "run_id": name,
                "run_dir": str(run_dir),
                "run_state": str(run_dir / "asset_run.json"),
                "prompt": "compose the cabinet",
                "physics_validation_mode": physics_validation_mode,
                "repository_root": str(tmp_path),
                "source_asset": str(source),
                "joint_config": str(joint_config),
                "joint_config_binding": {
                    "path": str(joint_config),
                    "sha256": file_sha256(joint_config),
                    "size_bytes": joint_config.stat().st_size,
                },
                "materials_yaml": str(materials_yaml),
                "materials_yaml_binding": {
                    "path": str(materials_yaml),
                    "sha256": file_sha256(materials_yaml),
                    "size_bytes": materials_yaml.stat().st_size,
                },
                "materials_usd": str(materials_usd),
                "materials_usd_binding": {
                    "path": str(materials_usd),
                    "sha256": file_sha256(materials_usd),
                    "size_bytes": materials_usd.stat().st_size,
                },
                "materials_usd_dependencies": [],
                "reference_images": [],
                "reference_files": [],
                "reference_bindings": [],
                "runtime": {
                    "runner": "codex",
                    "model": None,
                    "model_reasoning_effort": None,
                    **(
                        {
                            "workbench_url": "http://127.0.0.1:8088",
                            "start_workbench": False,
                            "keep_workbench": False,
                            "workbench_timeout_seconds": 60.0,
                        }
                        if legacy or historical_runtime
                        else {"scene_tool_timeout_seconds": 60.0}
                    ),
                    "child_timeout_seconds": 3600.0,
                    "codex_base_url": None,
                    "codex_sandbox_mode": "workspace-write",
                    "codex_config": {},
                    "claude_config": {},
                    "claude_permission_mode": "default",
                    "claude_max_turns": None,
                    "claude_execution_mode": "sdk",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    state_path = run_dir / "asset_run.json"
    create_run(
        state_path,
        run_id=name,
        request_path=request,
        source_asset=source,
        coordinator_mode="legacy" if legacy else "single_reasoning_loop",
    )
    return state_path


def _write_plan(
    state_path: Path,
    stage: StageName,
    *,
    evidence: Path | None = None,
    reason: str = "Initial stage plan.",
) -> Path:
    index = len(load_verified_run(state_path).coordinator.plan_revisions) + 1
    draft = state_path.parent / "raw" / f"plan-draft-{index:03d}.json"
    draft.write_text(
        json.dumps(
            {
                "schema_version": (
                    "content-agent-workflows.asset-coordinator-plan-draft.v1"
                ),
                "stage": stage,
                "objective": f"Produce accepted {stage} evidence.",
                "steps": [
                    {
                        "stage": stage,
                        "objective": f"Run the typed {stage} executor.",
                        "acceptance_evidence": [f"accepted {stage} result"],
                        "may_revisit": True,
                    }
                ],
                "evidence_paths": [
                    str(evidence or (state_path.parent / "request.json"))
                ],
                "revision_reason": reason,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    record_coordinator_plan(state_path, plan_path=draft, actor="test-agent")
    return draft


def _write_review(
    state_path: Path,
    stage: StageName,
    *,
    decision: str,
    evidence: list[Path],
    output: Path | None = None,
    target_stage: StageName | None = None,
    repair_scope: list[str] | None = None,
) -> Path:
    index = len(load_verified_run(state_path).coordinator.evidence_reviews) + 1
    draft = state_path.parent / "raw" / f"review-draft-{index:03d}.json"
    draft.write_text(
        json.dumps(
            {
                "schema_version": (
                    "content-agent-workflows.asset-coordinator-review-draft.v1"
                ),
                "stage": stage,
                "output_asset_path": str(output) if output is not None else None,
                "evidence_paths": [str(path) for path in evidence],
                "findings": [f"Reviewed exact {stage} evidence."],
                "decision": decision,
                "target_stage": target_stage,
                "decision_summary": f"Coordinator selected {decision}.",
                "repair_scope": repair_scope or [],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    record_coordinator_evidence_review(
        state_path,
        review_path=draft,
        actor="test-agent",
    )
    return draft


def _begin(state_path: Path, stage: StageName) -> Path:
    _write_plan(state_path, stage)
    begin_stage(state_path, stage, actor="test-agent")
    return stage_directory(state_path, stage)


def _write_material_evidence(
    state_path: Path,
    directory: Path,
    output: Path,
    evidence: Path,
) -> list[Path]:
    run = load_verified_run(state_path)
    stage_state = run.stages["material"]
    assert stage_state.input_asset is not None
    frozen_request = json.loads(
        (state_path.parent / "request.json").read_text(encoding="utf-8")
    )
    coordinator_request = directory / "coordinator_request.json"
    coordinator_request.write_text(
        json.dumps(
            {
                "schema_version": "content-agents.material-coordinator-request.v1",
                "run_dir": str(directory),
                "repository_root": frozen_request["repository_root"],
                "source": {
                    "path": stage_state.input_asset.path,
                    "sha256": stage_state.input_asset.sha256,
                },
                "materials_yaml": {
                    "path": frozen_request["materials_yaml_binding"]["path"],
                    "sha256": frozen_request["materials_yaml_binding"]["sha256"],
                },
                "materials_usd": {
                    "path": frozen_request["materials_usd_binding"]["path"],
                    "sha256": frozen_request["materials_usd_binding"]["sha256"],
                },
                "materials_usd_dependencies": frozen_request[
                    "materials_usd_dependencies"
                ],
                "reference_images": [],
                "reference_files": [],
                "output_usd_path": str(output),
                "scene_tool_timeout_seconds": frozen_request["runtime"][
                    "scene_tool_timeout_seconds"
                ],
                "optimize": False,
                "root_prim_path": None,
                "material_candidate_space": "renderable",
                "skip_instances": False,
                "skip_prototypes": False,
                "skip_invisible": False,
                "flatten_prototypes": None,
                "enable_deinstance": None,
                "enable_split": None,
                "enable_deduplicate": None,
                "respect_existing_material_bindings": True,
                "material_restore_timeout_seconds": 60.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    raw = directory / "raw"
    raw.mkdir()
    decision_patch = raw / "material_applied_decision_patch.json"
    initial_render = directory / "evidence_renders" / "initial.png"
    initial_render.parent.mkdir()
    initial_render.write_bytes(b"initial render")
    final_render = directory / "final_renders" / "final.png"
    final_render.parent.mkdir()
    final_render.write_bytes(b"final render")
    turntable_frames: list[Path] = []
    for index in range(24):
        frame = final_render.parent / f"final_turntable_{index:03d}.png"
        frame.write_bytes(f"turntable frame {index}".encode())
        turntable_frames.append(frame)
    turntable_gif = final_render.parent / "final_turntable.gif"
    turntable_gif.write_bytes(b"turntable gif")
    visual_quality = {
        "schema_version": "content-agents.visual-quality-assessment.v1",
        "status": "pass",
        "checked_views": [str(final_render)],
        "reference_images": [],
        "reference_files": [],
        "issues_found": [],
        "issues_fixed": [],
        "unresolved_issues": [],
        "assessment_notes": "Final Material render passed visual review.",
    }
    decision_patch.write_text(
        json.dumps(
            {
                "schema_version": "content-agents.material-decision-patch.v1",
                "material_assignments": [],
                "reviewed_no_override": [],
                "visual_quality_assessment": {
                    **visual_quality,
                    "checked_views": [str(initial_render)],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (directory / "visual_quality_assessment.json").write_text(
        json.dumps(visual_quality) + "\n",
        encoding="utf-8",
    )
    (directory / "assignments.json").write_text(
        json.dumps(
            {
                "schema_version": "content-agents.assignments.v1",
                "source_usd": stage_state.input_asset.path,
                "coverage": {
                    "unassigned_visible_prim_count": 0,
                    "missing_assignment_prim_count": 0,
                    "rejected_assignment_prim_count": 0,
                },
                "visual_quality_assessment": visual_quality,
                "materialized_usd": {
                    "status": "succeeded",
                    "requested_output_path": str(output),
                    "output_path": str(output),
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    validation = material_assignment_validation_evidence(
        asset=stage_state.input_asset.path,
        target_runtime="usd-cli/ovrtx",
        visual_materials_status="pass",
        evidence_artifacts=[
            EvidenceArtifact(
                kind="render",
                path=str(final_render),
                description="Final Material verification render.",
            )
        ],
    )
    (directory / "validation_evidence.json").write_text(
        validation.model_dump_json() + "\n",
        encoding="utf-8",
    )
    (raw / "material_restore_response.json").write_text(
        json.dumps(
            {
                "scene_tool": "usd-cli",
                "output_usd_path": str(output),
                "unresolved_mappings": [],
                "restored_edit_count": 0,
                "restored_source_prim_paths": [],
                "unbound_source_prim_paths": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (raw / "final_render_records.json").write_text(
        json.dumps(
            {
                "schema_version": "content-agents.material-final-renders.v1",
                "scene_tool": "usd-cli",
                "render_engine": "ovrtx",
                "transports": ["ovrtx"],
                "turntable": {
                    "frame_count": len(turntable_frames),
                    "gif_path": str(turntable_gif),
                },
                "renders": [
                    {
                        "kind": "verification_view",
                        "renderer": "ovrtx",
                        "image_path": str(final_render),
                    },
                    *[
                        {
                            "kind": "turntable_frame",
                            "renderer": "ovrtx",
                            "image_path": str(frame),
                        }
                        for frame in turntable_frames
                    ],
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    final_render_paths = [final_render, *turntable_frames, turntable_gif]
    final_bindings = [
        {
            "path": str(path),
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in final_render_paths
    ]
    (raw / "material_application_receipt.json").write_text(
        json.dumps(
            {
                "schema_version": "content-agents.material-application-receipt.v1",
                "status": "review_required",
                "final_render_bindings": final_bindings,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (raw / "material_post_apply_review.json").write_text(
        json.dumps(
            {
                "schema_version": "content-agents.material-post-apply-review.v1",
                "status": "pass",
                "checked_views": [str(path) for path in final_render_paths],
                "checked_view_bindings": final_bindings,
                "unresolved_issues": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (directory / "api_operation_counts.json").write_text(
        json.dumps(
            {
                "schema_version": "content-agents.api-operation-counts.v1",
                "api_operation_count_total": 1,
                "render_count_total": 25,
                "material_override_commands": 0,
                "final_renders": 25,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (raw / "material_binding_audit.json").write_text(
        json.dumps(
            {
                "schema_version": "content-agents.material-binding-audit.v1",
                "status": "pass",
                "expected_target_count": 0,
                "verified_target_count": 0,
                "records": [],
                "errors": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    trace = directory / "trace"
    trace.mkdir()
    canonical_evidence = [
        coordinator_request,
        directory / "coordinator_preparation.json",
        directory / "assignments.json",
        directory / "api_operation_counts.json",
        directory / "visual_quality_assessment.json",
        directory / "validation_evidence.json",
        directory / "final_summary.md",
        raw / "material_restore_response.json",
        raw / "material_binding_audit.json",
        raw / "material_application_receipt.json",
        raw / "material_post_apply_review.json",
        raw / "ovrtx_render_probe.json",
        raw / "material_operation_receipts.json",
        decision_patch,
        raw / "material_finalization_policy.json",
        raw / "material_run_packet.json",
        raw / "visible_candidate_prims.json",
        raw / "material_palette.json",
        raw / "material_authoring_context.md",
        raw / "material_assignment_seed.json",
        raw / "visible_candidate_table.tsv",
        raw / "final_render_records.json",
        trace / "operation_trace.json",
        trace / "operation_trace.md",
        trace / "run_retrospective.json",
        trace / "replay_manifest.json",
    ]
    for path in canonical_evidence:
        if not path.exists():
            path.write_text("{}\n", encoding="utf-8")
    result_evidence = [
        evidence,
        *canonical_evidence,
        initial_render,
        *final_render_paths,
    ]
    coordinator_result = directory / "coordinator_result.json"
    coordinator_result.write_text(
        json.dumps(
            {
                "schema_version": "content-agents.material-coordinator-result.v1",
                "status": "pass",
                "output_usd_path": str(output),
                "output_usd_sha256": file_sha256(output),
                "request": {
                    "path": str(coordinator_request),
                    "sha256": file_sha256(coordinator_request),
                },
                "decision_patch": {
                    "path": str(decision_patch),
                    "sha256": file_sha256(decision_patch),
                },
                "evidence": [
                    {
                        "path": str(path),
                        "sha256": file_sha256(path),
                    }
                    for path in result_evidence
                ],
                "unresolved_issues": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return [*result_evidence, coordinator_result]


def _complete_simple_stage(state_path: Path, stage: StageName) -> Path:
    directory = _begin(state_path, stage)
    output = directory / f"{stage}.usda"
    if stage in {"material", "texture", "validation"}:
        run = load_verified_run(state_path)
        input_asset = run.stages[stage].input_asset
        assert input_asset is not None
        output.write_bytes(Path(input_asset.path).read_bytes())
    else:
        output.write_text(
            f'#usda 1.0\ndef Xform "{stage.title()}" {{}}\n',
            encoding="utf-8",
        )
    if stage == "texture":
        evidence_paths = _write_texture_evidence(state_path, directory, output)
    elif stage == "validation":
        evidence_paths = _write_validation_evidence(state_path, directory)
    else:
        evidence = directory / f"{stage}-evidence.json"
        evidence.write_text('{"status":"pass"}\n', encoding="utf-8")
        evidence_paths = (
            _write_material_evidence(state_path, directory, output, evidence)
            if stage == "material"
            else [evidence]
        )
    _write_review(
        state_path,
        stage,
        decision="accept",
        evidence=evidence_paths,
        output=output,
    )
    complete_stage(
        state_path,
        stage,
        output_asset=output,
        evidence_paths=evidence_paths,
        summary=f"Accepted {stage}.",
        actor="test-agent",
    )
    return output


def _write_texture_shared_receipt(
    *,
    domain_run: Path,
    request: TextureWorkflowRequest,
    plan: TexturePlanDocument,
    output: Path,
    unit_id: str,
    visual_evidence_path: Path,
) -> tuple[TextureEmbeddedDecisionState, list[Path], Path]:
    context = request.execution_context
    assert context is not None and context.embedded_stage is not None
    source = context.embedded_stage.input_asset
    visual = ExecutionArtifactBinding(
        path=str(visual_evidence_path),
        sha256=file_sha256(visual_evidence_path),
        size_bytes=visual_evidence_path.stat().st_size,
    )
    published = ExecutionArtifactBinding(
        path=str(output),
        sha256=file_sha256(output),
        size_bytes=output.stat().st_size,
    )
    generator_inputs = TextureGeneratorInputs(
        backend="test",
        prompt="test surface appearance",
    )
    inspection = TextureInspectionResult(
        source=source,
        proposal_plan_digest=texture_plan_digest(plan),
        units=(
            TextureInspectionUnit(
                unit_id=unit_id,
                material_prim_paths=("/World/Looks/Test",),
                member_prim_paths=("/World/Geometry/Test",),
                uv_status="ready",
                uv_facts={
                    "uv_scope": "target_prims",
                    "material_bindings": [
                        {
                            "prim_path": "/World/Geometry/Test",
                            "material_prim_path": "/World/Looks/Test",
                        }
                    ],
                    "primvars": [
                        {
                            "prim_path": "/World/Geometry/Test",
                            "status": "ready",
                        }
                    ],
                },
                proposed_generator_inputs=generator_inputs,
            ),
        ),
        before_render_artifacts=(visual,),
        inspection_artifacts=(visual,),
        capability_constraints=("surface texturing only",),
        renderer_metadata={"interface": "test-renderer"},
        tool_metadata={"adapter": "test"},
    )
    timestamp = datetime.now(UTC)
    state = initialize_embedded_texture_decisions(
        request,
        plan,
        inspection,
        created_at=timestamp,
    )
    store = EmbeddedDecisionArtifactStore(domain_run)
    canonical_plan = TextureCanonicalPlan(
        source=source,
        proposal_plan_digest=texture_plan_digest(plan),
        targets=(
            TexturePlanTarget(
                unit_id=unit_id,
                material_prim_paths=("/World/Looks/Test",),
                member_prim_paths=("/World/Geometry/Test",),
                requested_appearance=generator_inputs.prompt,
                generator_inputs=generator_inputs,
            ),
        ),
        preservation=TexturePreservationConstraints(),
        acceptance=TextureAcceptanceCriteria(
            appearance_requirements=("Test appearance is visible.",)
        ),
        capability_constraints=inspection.capability_constraints,
    )

    def patch(
        action: str, revision: int, **values: object
    ) -> TextureEmbeddedDecisionPatch:
        return TextureEmbeddedDecisionPatch(
            request_digest=texture_request_digest(request),
            source_identity_digest=texture_source_identity_digest(request, plan=plan),
            proposal_plan_digest=texture_plan_digest(plan),
            checkpoint_decision_digest=f"{revision}" * 64,
            checkpoint_revision=revision,
            action=action,
            iteration=0,
            created_at=timestamp,
            rationale=f"Test outer coordinator accepted {action}.",
            confidence=1.0,
            **values,
        )

    plan_patch = patch("execute", 1, canonical_plan=canonical_plan)
    candidate_decision, evidence, proposals = build_plan_decision(
        plan_patch,
        store=store,
        state=state,
        proposal=plan,
    )
    store.append(candidate_decision)
    candidate_authorization = authorize_candidate_generation(
        candidate_decision,
        evidence=evidence,
        proposals=proposals,
    )
    store.invoke_after_authorization_commit(
        candidate_authorization,
        lambda _claim: None,
    )
    candidate_path = domain_run / "texture-candidate.usda"
    candidate_path.write_bytes(output.read_bytes())
    candidate_artifact_path = domain_run / "candidate-texture.png"
    candidate_artifact_path.write_bytes(b"generated candidate texture")
    candidate_execution = TextureExecutionResult(
        requested_unit_ids=(unit_id,),
        unit_artifacts=(
            TextureUnitArtifact(
                unit_id=unit_id,
                artifact_paths=(str(candidate_artifact_path),),
            ),
        ),
        output_asset_path=str(candidate_path),
    )
    candidate_validation = TextureValidationResult(
        iteration=0,
        evaluated_unit_ids=(unit_id,),
        findings=(
            TextureValidationFinding(
                unit_id=unit_id,
                status="pass",
                summary="Test visual critique.",
                evidence_artifact_paths=(str(visual_evidence_path),),
            ),
        ),
        output_asset_path=str(candidate_path),
    )
    candidate_binding = ExecutionArtifactBinding(
        path=str(candidate_path.resolve()),
        sha256=file_sha256(candidate_path),
        size_bytes=candidate_path.stat().st_size,
    )
    closure_facts = texture_candidate_dependency_closure_facts(candidate_binding)
    closure_path = atomic_write_json(
        domain_run / "candidate-closure.json",
        closure_facts,
    )
    closure_binding = ExecutionArtifactBinding(
        path=str(closure_path.resolve()),
        sha256=file_sha256(closure_path),
        size_bytes=closure_path.stat().st_size,
    )
    candidate_result = candidate_result_artifact(
        candidate_authorization,
        candidate_execution,
        candidate_validation,
        candidate_closure_artifact=closure_binding,
        candidate_closure_facts=closure_facts,
        created_at=timestamp,
    )
    store.append_result(candidate_result)
    candidate_evidence = candidate_evidence_artifact(
        candidate_result,
        created_at=timestamp,
    )
    store.append(candidate_evidence)
    state = state.model_copy(
        update={
            "canonical_plan": canonical_plan,
            "current_decision": artifact_reference(candidate_decision),
            "current_authorization": artifact_reference(candidate_authorization),
            "current_result": artifact_reference(candidate_result),
            "current_candidate_evidence": artifact_reference(candidate_evidence),
        }
    )
    candidate_review_payload = TextureCandidateReviewDecision(
        candidate_result_sha256=artifact_reference(candidate_result).sha256,
        output_asset=candidate_result.outputs[0],
        plan_digest=canonical_json_digest(canonical_plan),
        unit_dispositions=(
            TextureUnitDisposition(
                unit_id=unit_id,
                disposition="accept",
                rationale="Fresh visual evidence accepted.",
            ),
        ),
        visual_evidence_artifacts=(visual,),
        visual_evidence_accepted=True,
        findings=("Fresh candidate accepted.",),
    )
    candidate_review_patch = patch(
        "validate",
        2,
        candidate_review=candidate_review_payload,
    )
    candidate_review = build_candidate_review(
        candidate_review_patch,
        state=state,
        decision=candidate_decision,
        result=candidate_result,
    )
    candidate_domain_review = candidate_review_evidence_artifact(
        candidate_review_patch,
        state=state,
        result=candidate_result,
    )
    store.append(candidate_domain_review)
    store.append_review(candidate_review)
    candidate_receipt = build_review_receipt(
        artifact_id="texture-test-candidate-receipt",
        decision=candidate_decision,
        authorization=candidate_authorization,
        result=candidate_result,
        review=candidate_review,
        evidence=evidence,
        proposals=proposals,
    )
    store.append_receipt(candidate_receipt)
    state = state.model_copy(
        update={
            "current_candidate_domain_review": artifact_reference(
                candidate_domain_review
            ),
            "current_candidate_review": artifact_reference(candidate_review),
            "current_candidate_receipt": artifact_reference(candidate_receipt),
            "accepted_candidate_result": artifact_reference(candidate_result),
            "accepted_candidate_domain_review": artifact_reference(
                candidate_domain_review
            ),
            "accepted_candidate_review": artifact_reference(candidate_review),
            "accepted_candidate_receipt": artifact_reference(candidate_receipt),
        }
    )
    publication_payload = TexturePublicationDecision(
        candidate_result_sha256=artifact_reference(candidate_result).sha256,
        candidate_output=candidate_result.outputs[0],
        plan_digest=canonical_json_digest(canonical_plan),
        accepted_unit_ids=(unit_id,),
        candidate_review_sha256=artifact_reference(candidate_review).sha256,
        publication_path=str(output),
    )
    publication_patch = patch(
        "finalize",
        3,
        publication=publication_payload,
    )
    publication_decision, publication_evidence = build_publication_decision(
        publication_patch,
        store=store,
        state=state,
    )
    store.append(publication_decision)
    publication_authorization = authorize_publication(
        publication_decision,
        evidence=publication_evidence,
    )
    store.invoke_after_authorization_commit(
        publication_authorization,
        lambda _claim: None,
    )
    publication_result = publication_result_artifact(
        publication_authorization,
        published_asset=published,
        verification_artifacts=(visual,),
        verification_facts={
            "published_asset_sha256": published.sha256,
            "scope_invariants_passed": True,
            "geometry_unchanged": True,
            "non_target_materials_unchanged": True,
            "bindings_unchanged": True,
            "structure_unchanged_outside_target": True,
            "dependency_closure_complete": True,
            "exact_candidate_identity_preserved": True,
        },
        created_at=timestamp,
    )
    store.append_result(publication_result)
    state = state.model_copy(
        update={
            "publication_decision": artifact_reference(publication_decision),
            "publication_authorization": artifact_reference(publication_authorization),
            "publication_result": artifact_reference(publication_result),
        }
    )
    publication_review_payload = TexturePublicationReviewDecision(
        publication_result_sha256=artifact_reference(publication_result).sha256,
        published_asset=published,
        disposition="accept",
        findings=("Exact publication proof accepted.",),
    )
    publication_review_patch = patch(
        "review_publication",
        4,
        publication_review=publication_review_payload,
    )
    publication_review = build_publication_review(
        publication_review_patch,
        state=state,
        result=publication_result,
    )
    store.append_review(publication_review)
    completed_receipt = build_review_receipt(
        artifact_id="texture-test-completed-receipt",
        decision=publication_decision,
        authorization=publication_authorization,
        result=publication_result,
        review=publication_review,
        evidence=publication_evidence,
    )
    store.append_receipt(completed_receipt)
    receipt_ref = artifact_reference(completed_receipt)
    state = state.model_copy(
        update={
            "publication_review": artifact_reference(publication_review),
            "completed_receipt": receipt_ref,
        }
    )
    assert validate_completed_texture_decision_chain(store, state, plan) == (
        completed_receipt
    )
    paths = [
        (store.store_root / "journal.json").resolve(),
        *(
            (store.store_root / entry.relative_path).resolve()
            for entry in store.journal().entries
        ),
    ]
    receipt_path = (
        store.store_root
        / "artifacts"
        / receipt_ref.artifact_kind
        / f"{receipt_ref.sha256}.json"
    ).resolve()
    return (
        state,
        [*paths, candidate_path, candidate_artifact_path, closure_path],
        receipt_path,
    )


def _write_texture_evidence(
    state_path: Path,
    directory: Path,
    output: Path,
    *,
    status: str = "pass",
    omit_execution_context: bool = False,
    include_request_evidence: bool = True,
) -> list[Path]:
    domain_run = directory / "domain-run"
    domain_run.mkdir(exist_ok=True)
    input_asset = load_verified_run(state_path).stages["texture"].input_asset
    assert input_asset is not None
    context = build_embedded_domain_execution_context(
        state_path,
        domain="texture",
        input_asset=input_asset.path,
        output_dir=domain_run,
    )
    request = TextureWorkflowRequest(
        source_asset=input_asset.path,
        output_dir=domain_run,
        metadata=(
            {}
            if omit_execution_context
            else metadata_with_domain_execution_context({}, context)
        ),
    )
    if (
        request.execution_context is not None
        and load_verified_run(state_path).schema_version
        != "content-agent-workflows.asset-composition-run.v1"
    ):
        identity = build_embedded_domain_decision_identity(
            state_path,
            domain="texture",
            input_asset=input_asset.path,
            output_dir=domain_run,
            capability_digests=texture_embedded_capability_digests(),
            implementation_digests=texture_embedded_implementation_digests(),
            configuration_digests={
                "texture_request": texture_request_digest(request),
            },
        )
        request = request.model_copy(
            update={
                "metadata": {
                    **request.metadata,
                    TEXTURE_EMBEDDED_DECISION_IDENTITY_METADATA_KEY: (
                        identity.model_dump(mode="json")
                    ),
                }
            }
        )
    request_path = domain_run / "request.json"
    request_path.write_text(
        request.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    unit_id = "tu_11111111111111111111"
    accepted = (unit_id,) if status == "pass" else ()
    remaining = () if status == "pass" else (unit_id,)
    unit_artifact_path = domain_run / "generated-texture.png"
    unit_artifact_path.write_bytes(b"generated texture")
    visual_evidence_path = domain_run / "texture-validation.png"
    visual_evidence_path.write_bytes(b"texture validation")
    unit_artifact_paths = (
        {unit_id: (str(unit_artifact_path),)} if status != "cancelled" else {}
    )
    validation = TextureWorkflowValidationEvidence(
        target_runtime="usd-cli",
        status=status,
        selected_unit_ids=(unit_id,),
        accepted_unit_ids=accepted,
        remaining_unit_ids=remaining,
        selected_unit_count=1,
        backend_job_count=1,
        cache_hit_count=0,
        retry_count=0,
        output_asset_path=str(output),
        output_asset_sha256=file_sha256(output),
        unit_artifact_paths=unit_artifact_paths,
        visual_evidence_paths=(str(visual_evidence_path),),
    )
    validation_path = domain_run / "validation_evidence.json"
    validation_path.write_text(
        validation.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    progress_path = domain_run / "workflow_progress.json"
    progress_path.write_text(
        json.dumps(
            {
                "schema_version": "content-agent-workflows.texture-progress-log.v2",
                "events": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    summary_path = domain_run / "final_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": "content-agent-workflows.texture-summary.v3",
                "status": status,
                "mode": "batch",
                "source_asset": input_asset.path,
                "output_asset_path": str(output),
                "output_asset_sha256": file_sha256(output),
                "selected_unit_ids": [unit_id],
                "accepted_unit_ids": list(accepted),
                "remaining_unit_ids": list(remaining),
                "cancellation_reason": (
                    "test cancellation" if status == "cancelled" else None
                ),
                "artifacts": {
                    "workflow_checkpoint": str(domain_run / "workflow_checkpoint.json"),
                    "workflow_progress": str(progress_path),
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    plan = TexturePlanDocument(
        counts=TexturePlanCounts(selected_unit_count=1),
        selected_units=(TexturePlanSelectedUnit(unit_id=unit_id),),
        decision=TexturePlanDecision(state="approved", execution_allowed=True),
    )
    artifact = TextureUnitArtifact(
        unit_id=unit_id,
        artifact_paths=(str(unit_artifact_path),),
    )
    validation_result = TextureValidationResult(
        iteration=1,
        evaluated_unit_ids=(unit_id,),
        findings=(
            TextureValidationFinding(
                unit_id=unit_id,
                status="pass" if status == "pass" else "fail",
                summary="Bounded Texture validation result.",
                evidence_artifact_paths=(str(visual_evidence_path),),
            ),
        ),
        output_asset_path=str(output),
    )
    embedded_state = None
    shared_decision_evidence: list[Path] = []
    embedded_receipt_path: Path | None = None
    if (
        status == "pass"
        and request.execution_context is not None
        and load_verified_run(state_path).schema_version
        != "content-agent-workflows.asset-composition-run.v1"
    ):
        (
            embedded_state,
            shared_decision_evidence,
            embedded_receipt_path,
        ) = _write_texture_shared_receipt(
            domain_run=domain_run,
            request=request,
            plan=plan,
            output=output,
            unit_id=unit_id,
            visual_evidence_path=visual_evidence_path,
        )
    checkpoint = TextureWorkflowCheckpoint(
        updated_at="2026-08-03T00:00:00Z",
        mode="batch",
        request_digest=texture_request_digest(request),
        source_identity_digest=input_asset.sha256,
        plan_digest=texture_plan_digest(plan),
        plan=plan,
        next_action="done" if status != "cancelled" else "execute",
        selected_unit_ids=(unit_id,),
        accepted_unit_ids=accepted,
        remaining_unit_ids=remaining,
        unit_artifacts=({unit_id: artifact} if status != "cancelled" else {}),
        artifact_digests=(
            {
                unit_id: TextureArtifactDigest(
                    unit_id=unit_id,
                    sha256_by_path={
                        str(unit_artifact_path): file_sha256(unit_artifact_path)
                    },
                )
            }
            if status != "cancelled"
            else {}
        ),
        output_asset_path=str(output),
        output_asset_sha256=file_sha256(output),
        validations=(validation_result,),
        validation_evidence_sha256_by_path={
            str(visual_evidence_path): file_sha256(visual_evidence_path)
        },
        accepted_unit_material_state_digests=(
            {unit_id: "3" * 64} if status == "pass" else {}
        ),
        terminal_status=status,
        cancellation_reason=("test cancellation" if status == "cancelled" else None),
        embedded_decision_state=embedded_state,
    )
    checkpoint_path = domain_run / "workflow_checkpoint.json"
    checkpoint_path.write_text(
        checkpoint.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    decision_evidence: list[Path] = []
    if (
        load_verified_run(state_path).schema_version
        != "content-agent-workflows.asset-composition-run.v1"
    ):
        decision_specs = (
            (
                1,
                "execute",
                (
                    "inspect_scope",
                    "inspect_uv",
                    "generate_candidate",
                    "preview_apply",
                ),
            ),
            (2, "validate", ("render_verification", "assess_visual_quality")),
            (
                3,
                "finalize",
                (
                    "verify_dependency_closure",
                    "finalize_decision_patch",
                    "publish_portable_asset",
                ),
            ),
        )
        ledger_path = domain_run / "texture_decision_ledger.json"
        for revision, action, operations in decision_specs:
            patch = TextureDecisionPatch(
                request_digest=checkpoint.request_digest,
                source_identity_digest=checkpoint.source_identity_digest,
                plan_digest=checkpoint.plan_digest,
                checkpoint_decision_digest=f"{revision}" * 64,
                checkpoint_revision=revision,
                action=action,
                iteration=0,
                target_unit_ids=(unit_id,),
                operations=operations,
                evidence_sha256_by_path={},
                rationale=f"Coordinator approved the bounded {action} step.",
                confidence=0.9,
            )
            ledger_path = record_texture_decision_patch(
                patch,
                output_dir=domain_run,
            )
        ledger_payload = json.loads(ledger_path.read_text(encoding="utf-8"))
        decision_evidence.extend(
            Path(record["decision_patch_path"]) for record in ledger_payload["records"]
        )
        decision_evidence.append(ledger_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["artifacts"]["decision_ledger"] = str(ledger_path)
        if embedded_receipt_path is not None:
            summary["artifacts"]["embedded_decision_receipt"] = str(
                embedded_receipt_path
            )
        summary_path.write_text(
            json.dumps(summary) + "\n",
            encoding="utf-8",
        )
    return [
        *([request_path] if include_request_evidence else []),
        validation_path,
        progress_path,
        summary_path,
        checkpoint_path,
        *decision_evidence,
        *shared_decision_evidence,
        unit_artifact_path,
        visual_evidence_path,
    ]


def _write_validation_evidence(
    state_path: Path,
    directory: Path,
    *,
    template_name: str = "render_valid",
    include_outer_visual_gate: bool = False,
) -> list[Path]:
    run = load_verified_run(state_path)
    input_asset = run.stages["validation"].input_asset
    assert input_asset is not None
    domain = directory / "domain-run"
    domain.mkdir()
    context = build_embedded_domain_execution_context(
        state_path,
        domain="validation",
        input_asset=input_asset.path,
        output_dir=domain,
    )
    request = ValidationRequest(
        task_description="Validate the accepted composed asset.",
        inputs=(input_asset.path,),
        requested_templates=(template_name,),
        metadata=metadata_with_domain_execution_context({}, context),
    )
    plan = ValidationPlan(
        steps=(
            ValidationPlanStep(
                template_name=template_name,
                reason="Prove load and render validity.",
            ),
        )
    )
    render_path = domain / "render.png"
    evidence_artifacts: tuple[ValidationArtifactIdentity, ...] = ()
    template_metadata: dict[str, object] = {}
    if template_name == "render_valid":
        Image.new("RGB", (8, 8), (20, 80, 140)).save(render_path, format="PNG")
        evidence_artifacts = (
            ValidationArtifactIdentity(
                role="evidence",
                path=str(render_path),
                kind="file",
                sha256=file_sha256(render_path),
            ),
        )
        template_metadata = {
            "runtime_render": {
                "status": "available",
                "image_paths": [str(render_path)],
            }
        }
    template_result = ValidationTemplateResult(
        template_name=template_name,
        status="passed",
        metadata=template_metadata,
    )
    request_path = domain / "validation_request.json"
    request_path.write_text(request.model_dump_json(indent=2) + "\n", encoding="utf-8")
    plan_path = domain / "validation_plan.json"
    plan_path.write_text(plan.model_dump_json(indent=2) + "\n", encoding="utf-8")
    template_result_path = domain / "template_result.json"
    template_result_path.write_text(
        template_result.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    source_identity = ValidationArtifactIdentity(
        role="source",
        path=input_asset.path,
        kind="file",
        sha256=input_asset.sha256,
    )
    identity_values = {
        "schema_version": "content-agent-workflows.validation-identity.v1",
        "request_digest": canonical_json_digest(request),
        "policy_digest": "1" * 64,
        "backend_digest": "2" * 64,
        "source_artifacts": (source_identity,),
        "reference_artifacts": (),
        "template_versions": {template_name: "1"},
    }
    identity = ValidationWorkflowIdentity(
        **identity_values,
        identity_digest=validation_workflow_identity_digest(**identity_values),
    )
    work_item_identity = "3" * 64
    accepted = ValidationAcceptedTemplateResult(
        work_item_identity_digest=work_item_identity,
        attempt=1,
        result=template_result,
        result_path=str(template_result_path),
        result_sha256=file_sha256(template_result_path),
        evidence_artifacts=evidence_artifacts,
    )
    checkpoint = ValidationWorkflowCheckpoint(
        revision=1,
        workflow_identity=identity,
        plan_digest=canonical_json_digest(plan),
        ordered_work_item_ids=(template_name.replace("_", "-"),),
        records=(
            ValidationWorkItemRecord(
                work_item_id=template_name.replace("_", "-"),
                template_name=template_name,
                identity_digest=work_item_identity,
                state=ValidationWorkItemState.COMPLETED,
                attempts=1,
                accepted_result=accepted,
            ),
        ),
    )
    result = ValidationResult(
        verdict="pass",
        request=request,
        plan=plan,
        template_results=(template_result,),
    )
    result_path = domain / "validation_result.json"
    result_path.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    checkpoint_path = domain / "validation_checkpoint.json"
    checkpoint_path.write_text(
        checkpoint.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    evidence_path = domain / "validation_evidence.json"
    evidence_path.write_text(
        json.dumps(
            {
                "schema_version": VALIDATION_WORKFLOW_EVIDENCE_SCHEMA_VERSION,
                "workflow_identity_digest": identity.identity_digest,
                "plan_digest": checkpoint.plan_digest,
                "source_before": [source_identity.model_dump(mode="json")],
                "source_after": [source_identity.model_dump(mode="json")],
                "source_unchanged": True,
                "templates": {
                    template_name: {
                        "status": "passed",
                        "result_path": str(template_result_path),
                        "result_sha256": file_sha256(template_result_path),
                        "evidence_artifacts": [
                            artifact.model_dump(mode="json")
                            for artifact in evidence_artifacts
                        ],
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    summary_path = domain / "final_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": VALIDATION_WORKFLOW_SUMMARY_SCHEMA_VERSION,
                "status": "completed",
                "verdict": "pass",
                "recommendation": None,
                "source_asset_unchanged": True,
                "completed_templates": [template_name],
                "remaining_templates": [],
                "artifacts": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    cross_stage_path = _write_validation_cross_stage_evidence(
        state_path,
        directory,
        result_path,
    )
    cross_stage_payload = AssetCrossStageValidation.model_validate_json(
        cross_stage_path.read_text(encoding="utf-8")
    )
    cross_stage_warns = any(
        claim.status == "warn" for claim in cross_stage_payload.claims
    )
    native_run = load_embedded_validation_run(domain)
    prepare_embedded_validation_evidence(
        native_run,
        run_state_path=state_path,
    )
    gate_name = (
        "visual_quality"
        if template_name == "look_right"
        else (
            "runtime_validation"
            if template_name == "physical_behavior"
            else ("static_validation")
        )
    )
    gates = [
        ValidationGateAssessment(
            gate=gate_name,
            evidence_ids=(f"validation-template-{template_name}",),
            disposition="pass",
            rationale="Outer coordinator accepted the exact factual evidence.",
        ),
        ValidationGateAssessment(
            gate="package_integrity",
            evidence_ids=("validation-package-integrity",),
            disposition="pass",
            rationale="The saved native artifact package passed readback.",
        ),
        ValidationGateAssessment(
            gate="cross_stage_integrity",
            evidence_ids=("validation-cross-stage-integrity",),
            disposition="waive" if cross_stage_warns else "pass",
            rationale=(
                "The disclosed schema-readback runtime limitation is explicitly "
                "waived; accepted handoffs remain intact."
                if cross_stage_warns
                else "The accepted handoffs and cross-stage claims are intact."
            ),
        ),
    ]
    if include_outer_visual_gate:
        gates[0] = ValidationGateAssessment(
            gate="visual_quality",
            evidence_ids=(
                f"validation-template-{template_name}",
                "validation-outer-visual-evidence",
            ),
            disposition="pass",
            rationale="The outer reasoner inspected the bound render image.",
        )
    elif template_name == "render_valid":
        gates.append(
            ValidationGateAssessment(
                gate="visual_quality",
                evidence_ids=("validation-outer-visual-evidence",),
                disposition="pass",
                rationale="The outer reasoner inspected the bound render image.",
            )
        )
    assessment = ValidationCoordinatorAssessment(
        assessment_id=f"test-{template_name}",
        created_at=checkpoint.updated_at,
        gates=tuple(gates),
        terminal_disposition="pass",
        summary="Outer coordinator accepts every required Validation gate.",
    )
    assessment_draft = directory / "validation_assessment_draft.json"
    assessment_draft.write_text(
        assessment.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    assess_embedded_validation(
        domain,
        run_state_path=state_path,
        assessment_path=assessment_draft,
    )
    review_draft = ValidationCoordinatorReviewDraft(
        created_at=checkpoint.updated_at,
        disposition="accept",
        findings=("Exact assessment output matches the accepted decision.",),
    )
    review_draft_path = directory / "validation_assessment_review_draft.json"
    review_draft_path.write_text(
        review_draft.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    review_embedded_validation_assessment(
        domain,
        run_state_path=state_path,
        review_path=review_draft_path,
    )
    return [
        summary_path,
        result_path,
        evidence_path,
        checkpoint_path,
        cross_stage_path,
        domain / EMBEDDED_VALIDATION_EVIDENCE_INDEX_NAME,
        domain / EMBEDDED_VALIDATION_EXECUTION_INDEX_NAME,
        domain / EMBEDDED_VALIDATION_RECEIPT_INDEX_NAME,
        domain / CANONICAL_VALIDATION_ASSESSMENT_NAME,
    ]


def _write_validation_cross_stage_evidence(
    state_path: Path,
    directory: Path,
    result_path: Path,
) -> Path:
    run = load_verified_run(state_path)
    request_payload = json.loads(Path(run.request.path).read_text(encoding="utf-8"))
    schema_readback = request_payload.get("physics_validation_mode") == (
        "schema_readback"
    )
    input_asset = run.stages["validation"].input_asset
    assert input_asset is not None
    upstream = {
        stage: run.stages[stage]
        for stage in ("articulation", "material", "texture", "physics")
    }
    accepted_handoffs = {}
    for stage, stage_state in upstream.items():
        assert stage_state.handoff is not None
        accepted_handoffs[stage] = stage_state.handoff
    result_path_binding = asset_state._binding(result_path, label="test result")
    cross_stage = AssetCrossStageValidation(
        run_id=run.run_id,
        validation_input=input_asset,
        accepted_handoffs=accepted_handoffs,
        claims=(
            AssetCrossStageClaim(
                name="joint_graph",
                status="pass",
                summary="The reviewed Joint graph remains accepted.",
                evidence=[
                    upstream["articulation"].evidence[0],
                    result_path_binding,
                ],
            ),
            AssetCrossStageClaim(
                name="appearance",
                status="pass",
                summary="Material and Texture appearance evidence remains accepted.",
                evidence=[
                    upstream["material"].evidence[0],
                    upstream["texture"].evidence[0],
                ],
            ),
            AssetCrossStageClaim(
                name="physics_behavior",
                status="warn" if schema_readback else "pass",
                summary=(
                    "Physics schema/readback is accepted; runtime behavior was not "
                    "qualified."
                    if schema_readback
                    else "Physics authoring and runtime behavior remain accepted."
                ),
                evidence=[upstream["physics"].evidence[0]],
            ),
            AssetCrossStageClaim(
                name="render_and_package",
                status="pass",
                summary="The exact Physics output passes native Validation.",
                evidence=[result_path_binding],
            ),
            AssetCrossStageClaim(
                name="non_target_preservation",
                status="pass",
                summary="The unchanged source identity proves non-target preservation.",
                evidence=[result_path_binding],
            ),
        ),
    )
    cross_stage_path = directory / "cross_stage_validation.json"
    cross_stage_path.write_text(
        cross_stage.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    return cross_stage_path


def _complete_physics_stage(
    state_path: Path,
    *,
    omit_joint: bool = False,
    joint_axis: str | None = None,
    dependency_contents: str | None = None,
    omit_coordinator_patch: bool = False,
    decision_source_digest: str | None = None,
    simulation_report_overrides: dict[str, object] | None = None,
    with_topology_plan: bool = False,
    topology_operations: list[dict[str, object]] | None = None,
    topology_promotions: list[dict[str, object]] | None = None,
    topology_report_overrides: dict[str, object] | None = None,
    applied_decision_overrides: dict[str, object] | None = None,
    raw_applied_decision: dict[str, object] | None = None,
    coordinator_uses_target_ids: bool = False,
    assignment_path_space: str = "source",
    assignment_source_path_expansions: dict[str, list[str]] | None = None,
    assignment_overrides: dict[str, object] | None = None,
    runtime_loadability_status: CheckStatus = "pass",
    no_explosions_status: CheckStatus = "pass",
) -> Path:
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics
    from world_understanding.functions.physics.physics_topology import sha256_file

    from content_agent_workflows.physics import (
        PhysicsComponentDecision,
        add_physics_component_target_catalog,
        inspect_physics_components,
        physics_decision_assignment_payload,
    )

    directory = _begin(state_path, "physics")
    output = directory / "physics.usda"
    stage = Usd.Stage.CreateNew(str(output))
    UsdGeom.Xform.Define(stage, "/World")
    cabinet = UsdGeom.Xform.Define(stage, "/World/Cabinet").GetPrim()
    drawer = UsdGeom.Xform.Define(stage, "/World/Drawer").GetPrim()
    if coordinator_uses_target_ids:
        UsdGeom.Cube.Define(stage, "/World/Drawer/Visual")
    UsdPhysics.RigidBodyAPI.Apply(cabinet).CreateRigidBodyEnabledAttr(True)
    UsdPhysics.RigidBodyAPI.Apply(drawer).CreateRigidBodyEnabledAttr(True)
    if not omit_joint:
        joint = UsdPhysics.PrismaticJoint.Define(stage, "/World/DrawerJoint")
        joint.CreateBody0Rel().SetTargets([Sdf.Path("/World/Cabinet")])
        joint.CreateBody1Rel().SetTargets([Sdf.Path("/World/Drawer")])
        if joint_axis is not None:
            joint.CreateAxisAttr(joint_axis)
    if dependency_contents is not None:
        dependency = directory / "dependency.usda"
        dependency.write_text(dependency_contents, encoding="utf-8")
        referenced = stage.OverridePrim("/World/ReferencedDecoration")
        referenced.GetReferences().AddReference("dependency.usda")
    stage.GetRootLayer().Save()
    native = physics_validation_evidence(
        asset=str(output.resolve()),
        target_runtime="fake",
        physics_properties_status="pass",
        runtime_loadability_status=runtime_loadability_status,
        no_explosions_status=no_explosions_status,
    )
    native.metadata["asset_sha256"] = file_sha256(output)
    evidence = directory / "validation_evidence.json"
    evidence.write_text(native.model_dump_json() + "\n", encoding="utf-8")
    run = load_verified_run(state_path)
    input_asset = run.stages["physics"].input_asset
    assert input_asset is not None
    decision_component_id = "drawer-component"
    collider_path = "/World/Drawer"
    collider_target_id: str | None = None
    if coordinator_uses_target_ids:
        input_components = inspect_physics_components(input_asset.path)
        target_catalog = add_physics_component_target_catalog(
            {
                "components": [
                    component.model_dump(mode="json") for component in input_components
                ]
            }
        )
        drawer_components = [
            component
            for component in target_catalog["components"]
            if component["body_root_path"] == "/World/Drawer"
        ]
        assert len(drawer_components) == 1
        visual_targets = [
            target
            for target in drawer_components[0]["authoring_targets"]
            if target["role"] == "visual"
        ]
        assert len(visual_targets) == 1
        decision_component_id = drawer_components[0]["component_id"]
        collider_path = visual_targets[0]["prim_path"]
        collider_target_id = visual_targets[0]["target_id"]
    decision = {
        "decision_id": "drawer-physics",
        "component_id": decision_component_id,
        "body_root_path": "/World/Drawer",
        "visual_evidence_paths": [collider_path],
        "collider_paths": [collider_path],
        "collision_mode": "author_on_targets",
        "mass_authoring_path": "/World/Drawer",
        "inferred_material_family": "wood",
        "inferred_material_name": None,
        "collision_approximation": "convexHull",
        "physical_properties": {
            "density": 700.0,
            "estimated_mass_kg": 4.0,
            "static_friction": 0.5,
            "dynamic_friction": 0.4,
            "restitution": 0.1,
        },
        "confidence": 0.9,
        "rationale": "Drawer component requires bounded rigid-body properties.",
    }
    coordinator_decision = decision
    if coordinator_uses_target_ids:
        assert collider_target_id is not None
        coordinator_decision = {
            key: value
            for key, value in decision.items()
            if key
            not in {
                "body_root_path",
                "visual_evidence_paths",
                "collider_paths",
                "mass_authoring_path",
            }
        }
        coordinator_decision["collider_target_ids"] = [collider_target_id]
    patch_payload = {
        "schema_version": "content-agent-workflows.physics-decision-patch.v2",
        "asset": input_asset.path,
        "source_digest": decision_source_digest or sha256_file(input_asset.path),
        "decisions": [coordinator_decision],
        "unresolved_components": [],
    }
    coordinator_patch = directory / "coordinator_physics_decision_patch.json"
    coordinator_patch.write_text(
        json.dumps(patch_payload) + "\n",
        encoding="utf-8",
    )
    raw = directory / "domain-run" / "raw"
    raw.mkdir(parents=True)
    native_patch = raw / "physics_decision_patch.json"
    native_patch.write_text(json.dumps(patch_payload) + "\n", encoding="utf-8")
    prepared_asset = Path(input_asset.path)
    applied_patch = native_patch
    applied_decision_model = PhysicsComponentDecision.model_validate(
        {**decision, **(applied_decision_overrides or {})}
    )
    applied_decision = (
        raw_applied_decision
        if raw_applied_decision is not None
        else applied_decision_model.model_dump(mode="json")
    )
    topology_evidence: list[Path] = []
    if with_topology_plan:
        prepared_asset = directory / "domain-run" / "prepared.usda"
        prepared_asset.write_text(
            '#usda 1.0\ndef Xform "Prepared" {}\n', encoding="utf-8"
        )
        applied_patch = raw / "physics_decision_patch_apply.json"
        applied_payload = {
            **patch_payload,
            "asset": str(prepared_asset),
            "source_digest": sha256_file(prepared_asset),
            "decisions": [applied_decision],
        }
        applied_patch.write_text(json.dumps(applied_payload) + "\n", encoding="utf-8")
        topology_plan_payload = {
            "schema_version": "content-workflows.physics-topology-plan.v1",
            "expected_source_digest": sha256_file(input_asset.path),
            "mobility_intent": "preserve",
            "operations": topology_operations or [],
            "joint_endpoint_owner_promotions": topology_promotions or [],
            "invariants": {
                "enabled_collider_count": 0,
                "reject_articulation_changes": True,
            },
        }
        topology_plan = directory / "coordinator_physics_topology_plan.json"
        topology_plan.write_text(
            json.dumps(topology_plan_payload) + "\n", encoding="utf-8"
        )
        topology_report_payload: dict[str, object] = {
            "operation": "physics.apply_topology_plan",
            "schema_version": "content-workflows.physics-topology-plan.v1",
            "input_usd_path": input_asset.path,
            "output_usd_path": str(prepared_asset),
            "source_digest": sha256_file(input_asset.path),
            "output_digest": sha256_file(prepared_asset),
            "mobility_intent": "preserve",
            "applied_operations": topology_operations or [],
            "applied_joint_endpoint_owner_promotions": [
                {
                    **promotion,
                    "before_rigid_body_paths": [],
                    "after_rigid_body_paths": [
                        promotion["requested_rigid_body_ancestor_path"]
                    ],
                }
                for promotion in (topology_promotions or [])
            ],
            "rejected_operations": [],
            "warnings": [],
            "invariants": topology_plan_payload["invariants"],
            "invariant_results": {
                "enabled_collider_count_preserved": True,
                "articulation_changes_rejected": True,
            },
        }
        topology_report_payload.update(topology_report_overrides or {})
        topology_report = raw / "physics_topology_report.json"
        topology_report.write_text(
            json.dumps(topology_report_payload) + "\n", encoding="utf-8"
        )
        topology_evidence = [
            prepared_asset,
            applied_patch,
            topology_plan,
            topology_report,
        ]
    elif coordinator_uses_target_ids:
        applied_patch = raw / "physics_decision_patch_apply.json"
        applied_patch.write_text(
            json.dumps(
                {
                    **patch_payload,
                    "asset": str(prepared_asset),
                    "source_digest": sha256_file(prepared_asset),
                    "decisions": [applied_decision],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        topology_evidence = [applied_patch]
    components = raw / "physics_components.json"
    components.write_text(
        json.dumps(
            add_physics_component_target_catalog(
                {
                    "schema_version": ("content-agent-workflows.physics-components.v2"),
                    "asset": str(prepared_asset),
                    "source_digest": sha256_file(prepared_asset),
                    "path_space": "source",
                    "component_count": 1,
                    "components": [
                        {
                            "component_id": decision_component_id,
                            "body_root_path": "/World/Drawer",
                            "visual_evidence_paths": [collider_path],
                        }
                    ],
                }
            )
        )
        + "\n",
        encoding="utf-8",
    )
    apply_report_payload = {"operation": "physics.apply_schema", "status": "pass"}
    apply_report = raw / "physics_apply_report.json"
    apply_report.write_text(
        json.dumps(apply_report_payload) + "\n",
        encoding="utf-8",
    )
    trajectory = directory / "domain-run" / "trajectory.jsonl"
    trajectory.write_text('{"t":0.0,"pose":[0,0,0]}\n', encoding="utf-8")
    simulation_report_payload: dict[str, object] = {
        "engine": "fake",
        "physics_usd": str(output),
        "trajectory_jsonl": str(trajectory),
        "failures": [],
        "warnings": [],
    }
    simulation_report_payload.update(simulation_report_overrides or {})
    simulation_report = directory / "domain-run" / "simulation_report.json"
    simulation_report.write_text(
        json.dumps(simulation_report_payload) + "\n", encoding="utf-8"
    )
    assignments = directory / "domain-run" / "physics_assignments.json"
    assignment_expansions = assignment_source_path_expansions or {}
    assignments_payload: dict[str, object] = {
        "schema_version": "content-agent-workflows.physics-assignments.v1",
        "asset": input_asset.path,
        "source_asset_sha256": file_sha256(input_asset.path),
        "prepared_asset": str(prepared_asset),
        "prepared_asset_sha256": file_sha256(prepared_asset),
        "physics_usd": str(output),
        "path_space": assignment_path_space,
        "source_path_expansions": assignment_expansions,
        "candidate_count": 1,
        "component_count": 1,
        "decision_count": 1,
        "decision_patch": str(native_patch),
        "apply_decision_patch": str(applied_patch),
        "decisions": [
            physics_decision_assignment_payload(
                applied_decision_model,
                path_space=assignment_path_space,
                source_path_expansions=assignment_expansions,
            )
        ],
        "unresolved_components": [],
        "mobility_intent": "preserve",
        "apply_report": apply_report_payload,
        "validation_evidence": str(evidence),
        "simulation_report": str(simulation_report),
    }
    assignments_payload.update(assignment_overrides or {})
    assignments.write_text(
        json.dumps(assignments_payload) + "\n",
        encoding="utf-8",
    )
    evidence_paths = [
        evidence,
        native_patch,
        components,
        apply_report,
        trajectory,
        simulation_report,
        assignments,
        *topology_evidence,
    ]
    if not omit_coordinator_patch:
        evidence_paths.append(coordinator_patch)
    _write_review(
        state_path,
        "physics",
        decision="accept",
        evidence=evidence_paths,
        output=output,
    )
    complete_stage(
        state_path,
        "physics",
        output_asset=output,
        evidence_paths=evidence_paths,
        summary="Accepted simulation-backed physics.",
        actor="test-agent",
    )
    return output


def _complete_articulation(
    state_path: Path,
    *,
    stale_candidate_receipt: bool = False,
    with_rigid_bodies: bool = True,
    drawer_only_rigid_body: bool = False,
    with_drawer_visual: bool = False,
    receipt_reviewer: str = "asset-owner",
    omit_execution_context: bool = False,
    include_request_evidence: bool = True,
    omit_decision_ledger_evidence: bool = False,
    strip_terminal_checkpoint_bindings: bool = False,
    include_scene_evidence: bool = False,
    omit_scene_nested_evidence: bool = False,
    omit_articulation_evidence_name: str | None = None,
    stub_terminal_summary: bool = False,
) -> Path:
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics
    from world_understanding.functions.physics.joint_rigger import (
        identify_usd_artifact,
    )
    from world_understanding.functions.physics.physics_topology import sha256_file

    first_attempt = _begin(state_path, "articulation")
    domain_run = first_attempt / "domain-run"
    domain_run.mkdir()
    initial_run = load_verified_run(state_path)
    source_identity = identify_usd_artifact(
        Path(initial_run.source_asset.path).resolve(),
        uri=Path(initial_run.source_asset.path).resolve().as_uri(),
    )
    assert source_identity.dependency_bundle_sha256 is not None
    source_dependency_bundle_sha256 = source_identity.dependency_bundle_sha256
    context = build_embedded_domain_execution_context(
        state_path,
        domain="articulation",
        input_asset=initial_run.source_asset.path,
        output_dir=domain_run,
    )
    metadata = {SKILL_ROUTED_DECISION_METADATA_KEY: True}
    if not omit_execution_context:
        metadata = metadata_with_domain_execution_context(metadata, context)
    request = ArticulationWorkflowRequest(
        source_asset=initial_run.source_asset.path,
        output_dir=domain_run,
        review_policy="all",
        metadata=metadata,
    )
    request_path = domain_run / "request.json"
    atomic_write_json(request_path, request)
    candidate = Stage2ArticulationCandidate(
        candidate_id="drawer",
        motion_type="prismatic",
        joint_type_hint="prismatic",
        moving_part_prims=("/World/Drawer",),
        fixed_parent_prim="/World/Cabinet",
        parent_resolution_source="stage1_hint",
        axis_hint="x",
        motion_axis_world=(1.0, 0.0, 0.0),
        confidence="high",
        role="drawer",
        field_sources={
            "motion_type": "predicted",
            "axis_hint": "predicted",
            "motion_axis_world": "predicted",
            "fixed_parent_prim": "stage1_hint",
        },
        axis_evidence=(
            {
                "source": "predicted",
                "description": "source-bound signed axis",
                "value": "x",
                "prim_paths": ("/World/Drawer",),
            },
        ),
        connectivity_evidence=(
            {
                "source": "stage1_hint",
                "description": "source-bound parent-child edge",
                "value": "/World/Cabinet",
                "prim_paths": ("/World/Cabinet", "/World/Drawer"),
                "connectivity_role": "body0_body1_edge",
            },
        ),
        review_status="ready_for_rigger_input",
    )
    candidate_document = Stage2CandidateDocument(
        summary=Stage2CandidateSummary(
            candidate_count=1,
            ready_candidate_count=1,
            review_required_candidate_count=0,
        ),
        candidates=(candidate,),
    )
    candidates = domain_run / "articulation_candidates.json"
    atomic_write_json(candidates, candidate_document)
    predictions_path = domain_run / "predictions.json"
    predictions_path.write_text('{"drawer":"prismatic"}\n', encoding="utf-8")
    report_path = domain_run / "candidate_report.json"
    report_path.write_text('{"candidate_count":1}\n', encoding="utf-8")
    membership_document = MembershipDispositionDocument(
        schema_version="joint-agent-membership-disposition-v1",
        summary=MembershipDispositionSummary(
            disposition_count=1,
            disposition_counts={"independent_motion": 1},
            review_required_count=0,
            pending_downstream_count=0,
        ),
        dispositions=(
            MembershipDispositionRecord(
                disposition_id="membership-drawer",
                member_prim="/World/Drawer",
                motion_candidate_prim="/World/Drawer",
                physical_owner_prim="/World/Drawer",
                physical_owner_candidate_prim="/World/Drawer",
                disposition="independent_motion",
                source="accepted_manifest",
                confidence="high",
                rationale="Fixture explicitly accepts the drawer as an independent link.",
                downstream_boundary="moving_candidate_generation",
                review_status="resolved",
            ),
        ),
    )
    inference = ArticulationInferenceResult(
        candidate_document=candidate_document,
        membership_disposition_document=membership_document,
        backend_configuration_sha256="3" * 64,
        predictions_path=str(predictions_path.resolve()),
        predictions_sha256=file_sha256(predictions_path),
        report_path=str(report_path.resolve()),
        report_sha256=file_sha256(report_path),
        metadata={
            "membership_disposition_schema_version": (
                "joint-agent-membership-disposition-v1"
            ),
            "membership_disposition_required": True,
            "membership_disposition_classic_fallback_used": False,
        },
    )
    inference_path = domain_run / "inference_result.json"
    atomic_write_json(inference_path, inference)
    scene_manifest_path: Path | None = None
    scene_nested_paths: list[Path] = []
    if include_scene_evidence:
        evidence_root = domain_run / "scene_evidence"
        evidence_root.mkdir()

        def write_scene_artifact(name: str) -> ArticulationArtifactBinding:
            path = evidence_root / name
            path.write_text(f"{name}\n", encoding="utf-8")
            scene_nested_paths.append(path)
            return ArticulationArtifactBinding(
                path=str(path.resolve()),
                sha256=file_sha256(path),
            )

        render = ArticulationSceneRenderEvidence(
            candidate_id="drawer",
            focus_prim_path="/World/Drawer",
            direction="front",
            width=512,
            height=512,
            render_quality="final",
            preview_scene_path=request.source_asset,
            renderer="OVRTX",
            ovrtx_render_mode="PathTracing",
            ovrtx_num_sensor_updates=1,
            active_aov="LdrColor",
            image_artifact=write_scene_artifact("drawer.png"),
            response_artifact=write_scene_artifact("render-response.json"),
            camera_artifact=write_scene_artifact("camera.json"),
        )
        scene_manifest = ArticulationSceneEvidenceResult(
            identity=ArticulationSceneEvidenceIdentity(
                request_sha256=file_sha256(request_path),
                source_asset=request.source_asset,
                source_sha256=initial_run.source_asset.sha256,
                source_dependency_bundle_sha256=source_dependency_bundle_sha256,
                candidate_document_sha256=file_sha256(candidates),
                candidate_ids=("drawer",),
                collector_configuration_sha256="5" * 64,
            ),
            scene_session_id="test-session",
            scene_workspace_dir=str(evidence_root),
            source_scene_path=request.source_asset,
            inspection_scene_path=request.source_asset,
            scene_source_digest=sha256_file(request.source_asset),
            inspected_prim_paths=("/World/Cabinet", "/World/Drawer"),
            session_response_artifact=write_scene_artifact("session.json"),
            scene_snapshot_artifact=write_scene_artifact("snapshot.json"),
            topology_inspection_artifact=write_scene_artifact("topology.json"),
            prim_properties_artifact=write_scene_artifact("properties.json"),
            candidates=(
                ArticulationSceneCandidateEvidence(
                    candidate_id="drawer",
                    fixed_parent_prim="/World/Cabinet",
                    moving_part_prims=("/World/Drawer",),
                    inspected_prim_paths=("/World/Cabinet", "/World/Drawer"),
                    renders=(render,),
                ),
            ),
        )
        scene_manifest_path = evidence_root / "manifest.json"
        atomic_write_json(scene_manifest_path, scene_manifest)
    requires_embedded_context = (
        initial_run.schema_version != "content-agent-workflows.asset-composition-run.v1"
    )
    review_candidates = candidates
    approved_candidates = candidates
    decision_evidence: list[Path] = []
    if requires_embedded_context:
        checkpoint_path = domain_run / "checkpoint.json"
        checkpoint = ArticulationRunState(
            revision=1,
            mode="batch",
            phase="needs_review",
            request=ArticulationArtifactBinding(
                path=str(request_path.resolve()),
                sha256=file_sha256(request_path),
            ),
            source_asset=request.source_asset,
            source_sha256=initial_run.source_asset.sha256,
            source_dependency_bundle_sha256=source_dependency_bundle_sha256,
            backend_configuration_sha256="3" * 64,
            inference_result=ArticulationArtifactBinding(
                path=str(inference_path.resolve()),
                sha256=file_sha256(inference_path),
            ),
            candidate_document=ArticulationArtifactBinding(
                path=str(candidates.resolve()),
                sha256=file_sha256(candidates),
            ),
            scene_evidence_configuration_sha256=(
                "5" * 64 if scene_manifest_path is not None else None
            ),
            scene_evidence=(
                ArticulationArtifactBinding(
                    path=str(scene_manifest_path.resolve()),
                    sha256=file_sha256(scene_manifest_path),
                )
                if scene_manifest_path is not None
                else None
            ),
            candidate_ids=("drawer",),
            review_required_candidate_ids=("drawer",),
        )
        atomic_write_json(checkpoint_path, checkpoint)
        observation = build_articulation_step_observation(
            checkpoint,
            output_dir=domain_run,
        )
        patch = ArticulationDecisionPatch(
            request_sha256=observation.request_sha256,
            source_sha256=observation.source_sha256,
            source_dependency_bundle_sha256=(
                observation.source_dependency_bundle_sha256
            ),
            candidate_document_sha256=observation.candidate_document_sha256,
            scene_evidence_sha256=observation.scene_evidence_sha256,
            checkpoint_revision=observation.checkpoint_revision,
            decisions=(
                ArticulationCandidateDecision(
                    candidate_id="drawer",
                    decision="accept",
                    rationale="Coordinator accepted the exact native-ready proposal.",
                    confidence=0.9,
                    evidence_paths=tuple(observation.evidence_sha256_by_path),
                ),
            ),
            evidence_summary="Coordinator reviewed the exact Joint evidence.",
        )
        apply_articulation_decision_patch(domain_run, patch)
        review_candidates = domain_run / "agent_reviewed_articulation_candidates.json"
        reviewed_document = Stage2CandidateDocument.model_validate_json(
            review_candidates.read_text(encoding="utf-8")
        )
        approved_candidates = domain_run / "approved_articulation_candidates.json"
        approved_document = reviewed_document.model_copy(
            update={
                "summary": reviewed_document.summary.model_copy(
                    update={
                        "joint_type_counts": {"prismatic": 1},
                        "unresolved_axis_count": 0,
                        "unresolved_parent_count": 0,
                        "review_status_counts": {"ready_for_rigger_input": 1},
                        "limit_readiness_counts": {"not_provided": 1},
                        "reason_code_counts": {},
                    }
                )
            }
        )
        atomic_write_json(approved_candidates, approved_document)
        decision_evidence = [
            checkpoint_path,
            candidates,
            domain_run / "articulation_decision_patch.json",
            review_candidates,
            domain_run / "articulation_decision_ledger.json",
            approved_candidates,
            *([scene_manifest_path] if scene_manifest_path is not None else []),
            *([] if omit_scene_nested_evidence else scene_nested_paths),
        ]
    _write_review(
        state_path,
        "articulation",
        decision="await_review",
        evidence=[review_candidates],
    )
    require_review(
        state_path,
        candidates_path=review_candidates,
        actor="test-agent",
    )
    decisions = state_path.parent / "raw" / "joint-decisions.json"
    decisions.write_text('{"drawer":"accept"}\n', encoding="utf-8")
    record_review_decisions(
        state_path,
        decisions_path=decisions,
        reviewer=receipt_reviewer,
    )

    _write_plan(
        state_path,
        "articulation",
        evidence=review_candidates,
        reason="Resume authoring after the exact Joint review receipt.",
    )
    assert stage_directory(state_path, "articulation") == first_attempt
    continued = begin_stage(state_path, "articulation", actor="test-agent")
    assert continued.stages["articulation"].attempt_count == 1
    second_attempt = stage_directory(state_path, "articulation")
    assert second_attempt == first_attempt
    output = domain_run / "articulated.usda"
    stage = Usd.Stage.CreateNew(str(output))
    UsdGeom.Xform.Define(stage, "/World")
    cabinet = UsdGeom.Xform.Define(stage, "/World/Cabinet").GetPrim()
    drawer = UsdGeom.Xform.Define(stage, "/World/Drawer").GetPrim()
    if with_rigid_bodies:
        UsdPhysics.RigidBodyAPI.Apply(cabinet).CreateRigidBodyEnabledAttr(True)
        UsdPhysics.RigidBodyAPI.Apply(drawer).CreateRigidBodyEnabledAttr(True)
    elif drawer_only_rigid_body:
        UsdPhysics.RigidBodyAPI.Apply(drawer).CreateRigidBodyEnabledAttr(True)
    if with_drawer_visual:
        UsdGeom.Cube.Define(stage, "/World/Drawer/Visual")
    joint = UsdPhysics.PrismaticJoint.Define(stage, "/World/DrawerJoint")
    joint.CreateBody0Rel().SetTargets([Sdf.Path("/World/Cabinet")])
    joint.CreateBody1Rel().SetTargets([Sdf.Path("/World/Drawer")])
    stage.GetRootLayer().Save()
    run = load_verified_run(state_path)
    receipt_candidate_sha256 = (
        "4" * 64 if stale_candidate_receipt else file_sha256(review_candidates)
    )
    output_sha256 = file_sha256(output)
    diagnostics_path = domain_run / "joint_rigger_diagnostics.json"
    diagnostics_path.write_text('{"status":"succeeded"}\n', encoding="utf-8")
    joint_rigger_result_path = domain_run / "joint_rigger_result.json"
    joint_rigger_result_path.write_text(
        '{"authored_joint_count":1}\n',
        encoding="utf-8",
    )
    authoring = ArticulationAuthoringResult(
        status="succeeded",
        idempotency_key="1" * 64,
        source_sha256=run.source_asset.sha256,
        source_dependency_bundle_sha256=source_dependency_bundle_sha256,
        candidate_document_path=str(approved_candidates),
        candidate_document_sha256=file_sha256(approved_candidates),
        output_asset_path=str(output),
        output_asset_sha256=output_sha256,
        authored_candidate_ids=("drawer",),
        authored_joint_count=1,
        diagnostics_path=str(diagnostics_path.resolve()),
        diagnostics_sha256=file_sha256(diagnostics_path),
        joint_rigger_result_path=str(joint_rigger_result_path.resolve()),
        joint_rigger_result_sha256=file_sha256(joint_rigger_result_path),
    )
    authoring_path = domain_run / "authoring_result.json"
    authoring_path.write_text(
        authoring.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    authoring_request = ArticulationAuthoringRequest(
        source_asset=request.source_asset,
        source_sha256=run.source_asset.sha256,
        source_dependency_bundle_sha256=source_dependency_bundle_sha256,
        candidate_document_path=str(approved_candidates),
        candidate_document_sha256=file_sha256(approved_candidates),
        accepted_candidate_ids=("drawer",),
        idempotency_key="1" * 64,
        predictions_path=str(predictions_path.resolve()),
        predictions_sha256=file_sha256(predictions_path),
        output_dir=domain_run,
    )
    authoring_request_path = domain_run / "authoring_request.json"
    atomic_write_json(authoring_request_path, authoring_request)
    receipt = ArticulationReviewReceipt(
        request_sha256=file_sha256(request_path),
        source_sha256=run.source_asset.sha256,
        source_dependency_bundle_sha256=source_dependency_bundle_sha256,
        candidate_document_sha256=receipt_candidate_sha256,
        scene_evidence_sha256=(
            file_sha256(scene_manifest_path)
            if scene_manifest_path is not None
            else None
        ),
        reviewer="asset-owner",
        decisions=(ArticulationReviewEntry(candidate_id="drawer", decision="accept"),),
    )
    receipt_path = domain_run / "review_receipt.json"
    receipt_path.write_text(
        receipt.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    validation = ArticulationValidationResult(
        status="pass",
        output_asset_path=str(output),
        expected_output_asset_sha256=output_sha256,
        observed_output_asset_sha256=output_sha256,
        expected_candidate_ids=("drawer",),
        validated_candidate_ids=("drawer",),
        exact_graph_match=True,
        self_contained=True,
    )
    validation_path = domain_run / "validation_evidence.json"
    validation_path.write_text(
        validation.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    if requires_embedded_context:
        terminal_checkpoint = ArticulationRunState.model_validate_json(
            (domain_run / "checkpoint.json").read_text(encoding="utf-8")
        ).model_copy(
            update={
                "revision": 2,
                "phase": "completed",
                "review_receipt": ArticulationArtifactBinding(
                    path=str(receipt_path.resolve()),
                    sha256=file_sha256(receipt_path),
                ),
                "approved_candidate_document": ArticulationArtifactBinding(
                    path=str(approved_candidates.resolve()),
                    sha256=file_sha256(approved_candidates),
                ),
                "authoring_request": ArticulationArtifactBinding(
                    path=str(authoring_request_path.resolve()),
                    sha256=file_sha256(authoring_request_path),
                ),
                "authoring_result": ArticulationArtifactBinding(
                    path=str(authoring_path.resolve()),
                    sha256=file_sha256(authoring_path),
                ),
                "validation_result": ArticulationArtifactBinding(
                    path=str(validation_path.resolve()),
                    sha256=file_sha256(validation_path),
                ),
                "accepted_candidate_ids": ("drawer",),
            }
        )
        if stub_terminal_summary:
            atomic_write_json(domain_run / "workflow_progress.json", {})
            atomic_write_json(domain_run / "final_summary.json", {})
        else:
            write_articulation_workflow_summary(
                terminal_checkpoint,
                output_dir=domain_run,
                authoring=authoring,
            )
        if strip_terminal_checkpoint_bindings:
            terminal_checkpoint = terminal_checkpoint.model_copy(
                update={
                    "review_receipt": None,
                    "approved_candidate_document": None,
                    "authoring_request": None,
                    "authoring_result": None,
                    "validation_result": None,
                    "accepted_candidate_ids": (),
                }
            )
        atomic_write_json(domain_run / "checkpoint.json", terminal_checkpoint)
    evidence = [
        *([request_path] if include_request_evidence else []),
        inference_path,
        predictions_path,
        report_path,
        authoring_request_path,
        authoring_path,
        diagnostics_path,
        joint_rigger_result_path,
        receipt_path,
        validation_path,
        *([domain_run / "workflow_progress.json"] if requires_embedded_context else []),
        *([domain_run / "final_summary.json"] if requires_embedded_context else []),
        *(
            [
                path
                for path in decision_evidence
                if path.name != "articulation_decision_ledger.json"
            ]
            if omit_decision_ledger_evidence
            else decision_evidence
        ),
    ]
    if omit_articulation_evidence_name is not None:
        evidence = [
            path for path in evidence if path.name != omit_articulation_evidence_name
        ]
    _write_review(
        state_path,
        "articulation",
        decision="accept",
        evidence=evidence,
        output=output,
    )
    complete_stage(
        state_path,
        "articulation",
        output_asset=output,
        evidence_paths=evidence,
        summary="Accepted exact reviewed articulation graph.",
        actor="test-agent",
    )
    return output


def test_stage_cannot_begin_before_evidence_backed_plan(tmp_path: Path) -> None:
    state_path = _create(tmp_path)

    with pytest.raises(AssetCompositionStateError, match="requires a coordinator plan"):
        begin_stage(state_path, "articulation")

    _write_plan(state_path, "articulation")
    run = begin_stage(state_path, "articulation")

    assert run.coordinator.next_action == "execute_stage"
    assert len(run.coordinator.plan_revisions) == 1


def test_embedded_domain_context_binds_exact_active_attempt_and_input(
    tmp_path: Path,
) -> None:
    state_path = _create(tmp_path)
    run = load_verified_run(state_path)

    with pytest.raises(AssetCompositionStateError, match="running stage attempt"):
        build_embedded_domain_execution_context(
            state_path,
            domain="articulation",
            input_asset=run.source_asset.path,
            output_dir=stage_directory(state_path, "articulation") / "domain-run",
        )

    _begin(state_path, "articulation")
    other = tmp_path / "other.usda"
    other.write_text('#usda 1.0\ndef Xform "Other" {}\n', encoding="utf-8")
    with pytest.raises(
        AssetCompositionStateError,
        match="Embedded articulation input differs from the active stage handoff",
    ):
        build_embedded_domain_execution_context(
            state_path,
            domain="articulation",
            input_asset=other,
            output_dir=stage_directory(state_path, "articulation") / "domain-run",
        )

    with pytest.raises(AssetCompositionStateError, match="output directory must be"):
        build_embedded_domain_execution_context(
            state_path,
            domain="articulation",
            input_asset=load_verified_run(state_path).source_asset.path,
            output_dir=tmp_path / "outside-domain-run",
        )

    active = load_verified_run(state_path)
    context = build_embedded_domain_execution_context(
        state_path,
        domain="articulation",
        input_asset=active.source_asset.path,
        output_dir=stage_directory(state_path, "articulation") / "domain-run",
    )

    assert context.mode == "embedded"
    assert context.reasoning_loop_owner == "asset_coordinator"
    assert context.embedded_stage is not None
    assert context.embedded_stage.outer_run_id == active.run_id
    assert context.embedded_stage.stage_attempt == 1
    assert context.embedded_stage.domain_run_root == str(
        (stage_directory(state_path, "articulation") / "domain-run").resolve()
    )
    assert context.embedded_stage.outer_request.sha256 == active.request.sha256
    assert (
        context.embedded_stage.coordinator_plan.sha256
        == active.coordinator.plan_revisions[-1].sha256
    )


def test_embedded_decision_identity_binds_frozen_inputs_and_provider_manifests(
    tmp_path: Path,
) -> None:
    state_path = _create(tmp_path)
    _begin(state_path, "articulation")
    active = load_verified_run(state_path)
    domain_run = stage_directory(state_path, "articulation") / "domain-run"

    identity = build_embedded_domain_decision_identity(
        state_path,
        domain="articulation",
        input_asset=active.source_asset.path,
        output_dir=domain_run,
        capability_digests={"joint_inspection": "a" * 64},
        implementation_digests={"articulation_adapter": "b" * 64},
        configuration_digests={"candidate_graph_schema": "c" * 64},
    )

    request = json.loads(
        (state_path.parent / "request.json").read_text(encoding="utf-8")
    )
    assert identity.source.sha256 == active.source_asset.sha256
    assert identity.execution_context.embedded_stage is not None
    assert identity.execution_context.embedded_stage.outer_request.sha256 == (
        active.request.sha256
    )
    assert (
        identity.coordinator_plan.artifact_id
        == Path(active.coordinator.plan_revisions[-1].path).stem
    )
    assert identity.coordinator_plan.schema_version == (
        "content-agent-workflows.asset-coordinator-plan.v1"
    )
    assert (
        identity.coordinator_plan.sha256 == active.coordinator.plan_revisions[-1].sha256
    )
    assert identity.digests.configuration["asset_request"] == active.request.sha256
    assert identity.digests.configuration["candidate_graph_schema"] == "c" * 64
    assert (
        identity.digests.prompt["asset_prompt"]
        == hashlib.sha256(request["prompt"].encode("utf-8")).hexdigest()
    )
    assert identity.digests.capabilities == {"joint_inspection": "a" * 64}
    assert identity.digests.implementations == {"articulation_adapter": "b" * 64}

    session = AssetCoordinatorSession(run_state_path=state_path, mode="interactive")
    assert (
        session.embedded_decision_identity(
            domain="articulation",
            input_asset=active.source_asset.path,
            output_dir=domain_run,
            capability_digests={"joint_inspection": "a" * 64},
            implementation_digests={"articulation_adapter": "b" * 64},
            configuration_digests={"candidate_graph_schema": "c" * 64},
        )
        == identity
    )


def test_embedded_decision_identity_uses_one_locked_plan_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_path = _create(tmp_path)
    _begin(state_path, "articulation")
    active = load_verified_run(state_path)
    original_load = asset_state._load_coordinator_plan
    loaded_digests: list[str] = []

    def tracked_load(binding: ArtifactBinding) -> asset_state.AssetCoordinatorPlan:
        loaded_digests.append(binding.sha256)
        return original_load(binding)

    monkeypatch.setattr(asset_state, "_load_coordinator_plan", tracked_load)
    identity = build_embedded_domain_decision_identity(
        state_path,
        domain="articulation",
        input_asset=active.source_asset.path,
        output_dir=stage_directory(state_path, "articulation") / "domain-run",
        capability_digests={"joint_inspection": "a" * 64},
        implementation_digests={"articulation_adapter": "b" * 64},
    )

    assert identity.coordinator_plan.sha256 == loaded_digests[0]
    assert loaded_digests == [active.coordinator.plan_revisions[-1].sha256]


def test_validation_decision_identity_binds_active_input_dependency_closure(
    tmp_path: Path,
) -> None:
    state_path = _create(tmp_path, "validation-decision-dependencies")
    _complete_articulation(state_path)
    _complete_simple_stage(state_path, "material")
    _complete_simple_stage(state_path, "texture")
    _complete_physics_stage(
        state_path,
        dependency_contents='#usda 1.0\ndef Xform "ReviewedDependency" {}\n',
    )
    _begin(state_path, "validation")
    active = load_verified_run(state_path)
    validation_state = active.stages["validation"]
    assert validation_state.input_asset is not None
    assert validation_state.input_dependencies

    identity = build_embedded_domain_decision_identity(
        state_path,
        domain="validation",
        input_asset=validation_state.input_asset.path,
        output_dir=stage_directory(state_path, "validation") / "domain-run",
        capability_digests={"validation_checks": "a" * 64},
        implementation_digests={"validation_adapter": "b" * 64},
    )

    assert {
        name: digest
        for name, digest in identity.digests.references.items()
        if name.startswith("active_input_dependency_")
    } == {
        f"active_input_dependency_{index:03d}": binding.sha256
        for index, binding in enumerate(
            validation_state.input_dependencies,
            start=1,
        )
    }

    Path(validation_state.input_dependencies[0].path).write_text(
        '#usda 1.0\ndef Xform "DriftedDependency" {}\n',
        encoding="utf-8",
    )
    with pytest.raises(
        AssetCompositionStateError,
        match="dependency closure identity changed",
    ):
        build_embedded_domain_decision_identity(
            state_path,
            domain="validation",
            input_asset=validation_state.input_asset.path,
            output_dir=stage_directory(state_path, "validation") / "domain-run",
            capability_digests={"validation_checks": "a" * 64},
            implementation_digests={"validation_adapter": "b" * 64},
        )


def test_embedded_decision_identity_normalizes_invalid_adapter_digest_error(
    tmp_path: Path,
) -> None:
    state_path = _create(tmp_path)
    _begin(state_path, "articulation")
    active = load_verified_run(state_path)
    domain_run = stage_directory(state_path, "articulation") / "domain-run"

    with pytest.raises(
        AssetCompositionStateError,
        match="Invalid embedded articulation decision identity",
    ):
        build_embedded_domain_decision_identity(
            state_path,
            domain="articulation",
            input_asset=active.source_asset.path,
            output_dir=domain_run,
            capability_digests={"joint_inspection": "not-a-digest"},
            implementation_digests={"articulation_adapter": "b" * 64},
        )


def test_accept_review_without_output_raises_typed_state_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_path = _create(tmp_path)
    directory = _begin(state_path, "articulation")
    evidence = directory / "classification-evidence.json"
    evidence.write_text('{"status":"pass"}\n', encoding="utf-8")
    malformed_draft = asset_state.AssetCoordinatorReviewDraft.model_construct(
        schema_version="content-agent-workflows.asset-coordinator-review-draft.v1",
        stage="articulation",
        output_asset_path=None,
        evidence_paths=[str(evidence)],
        findings=["Malformed accept review."],
        decision="accept",
        target_stage=None,
        decision_summary="Attempted acceptance without output.",
        repair_scope=[],
    )
    monkeypatch.setattr(
        asset_state,
        "_load_review_draft",
        lambda _path: malformed_draft,
    )

    with pytest.raises(
        AssetCompositionStateError,
        match="Accept decision requires a reviewed output asset",
    ):
        record_coordinator_evidence_review(
            state_path,
            review_path=state_path.parent / "malformed-review.json",
            actor="test-agent",
        )


def test_articulation_acceptance_without_candidates_raises_typed_state_error(
    tmp_path: Path,
) -> None:
    state_path = _create(tmp_path)
    directory = _begin(state_path, "articulation")
    output = directory / "articulated.usda"
    output.write_text('#usda 1.0\ndef Xform "Articulated" {}\n', encoding="utf-8")
    decisions = directory / "review-decisions.json"
    decisions.write_text('{"accepted":[]}\n', encoding="utf-8")
    run = load_verified_run(state_path)
    input_asset = run.stages["articulation"].input_asset
    assert input_asset is not None
    run.stages["articulation"].review_decisions = ArtifactBinding(
        path=str(decisions.resolve()),
        sha256=file_sha256(decisions),
        size_bytes=decisions.stat().st_size,
    )
    output_binding = ArtifactBinding(
        path=str(output.resolve()),
        sha256=file_sha256(output),
        size_bytes=output.stat().st_size,
    )

    with pytest.raises(
        AssetCompositionStateError,
        match="requires frozen Joint review decisions and candidates",
    ):
        asset_state._validate_stage_acceptance(
            "articulation",
            state_path=state_path,
            run=run,
            input_asset=input_asset,
            input_dependencies=run.stages["articulation"].input_dependencies,
            output=output_binding,
            output_dependencies=[],
            evidence=[],
        )


def test_plan_rejects_mutable_coordinator_control_evidence(tmp_path: Path) -> None:
    state_path = _create(tmp_path)
    nested_log = state_path.parent / "stages" / "child-output.log"
    nested_log.parent.mkdir(parents=True)
    nested_log.write_text("live\n", encoding="utf-8")
    composed_log = state_path.parent / "stages" / "child-run-output.log"
    composed_log.write_text("live composed run\n", encoding="utf-8")
    composed_final = state_path.parent / "stages" / "child-review-final.md"
    composed_final.write_text("partial final\n", encoding="utf-8")
    coordinator_output = state_path.parent / "stages" / "coordinator-output.log"
    coordinator_output.write_text("live coordinator output\n", encoding="utf-8")
    coordinator_final = state_path.parent / "stages" / "coordinator-final.md"
    coordinator_final.write_text("partial coordinator final\n", encoding="utf-8")
    review_prompt = state_path.parent / "stages" / "agent_review_prompt.md"
    review_prompt.write_text("live review prompt\n", encoding="utf-8")
    checkpoint = state_path.parent / "stages" / "checkpoint.json"
    checkpoint.write_text("{}\n", encoding="utf-8")
    workflow_checkpoint = state_path.parent / "stages" / "workflow_checkpoint.json"
    workflow_checkpoint.write_text("{}\n", encoding="utf-8")
    validation_checkpoint = state_path.parent / "stages" / "validation_checkpoint.json"
    validation_checkpoint.write_text("{}\n", encoding="utf-8")
    scene_tool_log = state_path.parent / "usd-cli.log"
    scene_tool_log.write_text("live usd-cli output\n", encoding="utf-8")
    material_release = state_path.parent / "raw" / "material_session_release.json"
    material_release.write_text('{"status":"released"}\n', encoding="utf-8")
    material_decision = state_path.parent / "raw" / "material_decision_patch.json"
    material_decision.write_text('{"assignments":[]}\n', encoding="utf-8")

    for mutable_path in (
        state_path,
        nested_log,
        composed_log,
        composed_final,
        coordinator_output,
        coordinator_final,
        review_prompt,
        checkpoint,
        workflow_checkpoint,
        validation_checkpoint,
        scene_tool_log,
        material_release,
        material_decision,
    ):
        with pytest.raises(
            AssetCompositionStateError,
            match="mutable coordinator control artifact",
        ):
            _write_plan(state_path, "articulation", evidence=mutable_path)

    run = load_verified_run(state_path)
    assert run.coordinator.plan_revisions == []
    assert run.coordinator.next_action == "plan"


def test_material_accept_preserves_bindings_without_appearance_clear_report(
    tmp_path: Path,
) -> None:
    state_path = _create(tmp_path)
    _complete_articulation(state_path)
    directory = _begin(state_path, "material")
    output = directory / "material.usda"
    output.write_text('#usda 1.0\ndef Xform "Material" {}\n', encoding="utf-8")
    evidence = directory / "material-evidence.json"
    evidence.write_text('{"status":"pass"}\n', encoding="utf-8")

    evidence_paths = _write_material_evidence(
        state_path,
        directory,
        output,
        evidence,
    )

    assert not (directory / "raw" / "appearance_clear_report.json").exists()
    _write_review(
        state_path,
        "material",
        decision="accept",
        evidence=evidence_paths,
        output=output,
    )


def test_material_accept_requires_clear_report_when_bindings_are_not_preserved(
    tmp_path: Path,
) -> None:
    state_path = _create(tmp_path)
    _complete_articulation(state_path)
    directory = _begin(state_path, "material")
    output = directory / "material.usda"
    output.write_text('#usda 1.0\ndef Xform "Material" {}\n', encoding="utf-8")
    evidence = directory / "material-evidence.json"
    evidence.write_text('{"status":"pass"}\n', encoding="utf-8")
    evidence_paths = _write_material_evidence(
        state_path,
        directory,
        output,
        evidence,
    )
    coordinator_request = directory / "coordinator_request.json"
    request_payload = json.loads(coordinator_request.read_text(encoding="utf-8"))
    request_payload["respect_existing_material_bindings"] = False
    coordinator_request.write_text(
        json.dumps(request_payload) + "\n",
        encoding="utf-8",
    )
    coordinator_result = directory / "coordinator_result.json"
    result_payload = json.loads(coordinator_result.read_text(encoding="utf-8"))
    request_sha256 = file_sha256(coordinator_request)
    result_payload["request"]["sha256"] = request_sha256
    for item in result_payload["evidence"]:
        if item["path"] == str(coordinator_request):
            item["sha256"] = request_sha256
    coordinator_result.write_text(
        json.dumps(result_payload) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        AssetCompositionStateError,
        match="omits canonical evidence: raw/appearance_clear_report.json",
    ):
        _write_review(
            state_path,
            "material",
            decision="accept",
            evidence=evidence_paths,
            output=output,
        )


def test_material_accept_rejects_child_substitution_of_frozen_source(
    tmp_path: Path,
) -> None:
    state_path = _create(tmp_path)
    _complete_articulation(state_path)
    directory = _begin(state_path, "material")
    output = directory / "material.usda"
    output.write_text('#usda 1.0\ndef Xform "Material" {}\n', encoding="utf-8")
    evidence = directory / "material-evidence.json"
    evidence.write_text('{"status":"pass"}\n', encoding="utf-8")
    evidence_paths = _write_material_evidence(
        state_path,
        directory,
        output,
        evidence,
    )
    coordinator_request = directory / "coordinator_request.json"
    substituted_source = directory / "substituted-source.usda"
    substituted_source.write_text(
        '#usda 1.0\ndef Xform "Substituted" {}\n',
        encoding="utf-8",
    )
    request_payload = json.loads(coordinator_request.read_text(encoding="utf-8"))
    request_payload["source"] = {
        "path": str(substituted_source),
        "sha256": file_sha256(substituted_source),
    }
    coordinator_request.write_text(
        json.dumps(request_payload) + "\n",
        encoding="utf-8",
    )
    coordinator_result = directory / "coordinator_result.json"
    result_payload = json.loads(coordinator_result.read_text(encoding="utf-8"))
    request_sha256 = file_sha256(coordinator_request)
    result_payload["request"]["sha256"] = request_sha256
    for item in result_payload["evidence"]:
        if item["path"] == str(coordinator_request):
            item["sha256"] = request_sha256
    coordinator_result.write_text(
        json.dumps(result_payload) + "\n",
        encoding="utf-8",
    )

    review_count = len(load_verified_run(state_path).coordinator.evidence_reviews)
    with pytest.raises(
        AssetCompositionStateError,
        match="Material source differs from the frozen coordinator input",
    ):
        _write_review(
            state_path,
            "material",
            decision="accept",
            evidence=evidence_paths,
            output=output,
        )

    run = load_verified_run(state_path)
    assert len(run.coordinator.evidence_reviews) == review_count
    assert run.coordinator.next_action == "execute_stage"


def test_material_accept_rejects_conditional_result(tmp_path: Path) -> None:
    state_path = _create(tmp_path)
    _complete_articulation(state_path)
    directory = _begin(state_path, "material")
    output = directory / "material.usda"
    output.write_text('#usda 1.0\ndef Xform "Material" {}\n', encoding="utf-8")
    evidence = directory / "material-evidence.json"
    evidence.write_text('{"status":"conditional"}\n', encoding="utf-8")
    evidence_paths = _write_material_evidence(
        state_path,
        directory,
        output,
        evidence,
    )
    coordinator_result = directory / "coordinator_result.json"
    result_payload = json.loads(coordinator_result.read_text(encoding="utf-8"))
    result_payload["status"] = "conditional"
    result_payload["unresolved_issues"] = ["drawer front coverage is incomplete"]
    coordinator_result.write_text(
        json.dumps(result_payload) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        AssetCompositionStateError,
        match="must pass before acceptance",
    ):
        _write_review(
            state_path,
            "material",
            decision="accept",
            evidence=evidence_paths,
            output=output,
        )


@pytest.mark.parametrize("status", ["conditional", "cancelled"])
def test_texture_accept_rejects_nonpassing_native_result(
    tmp_path: Path,
    status: str,
) -> None:
    state_path = _create(tmp_path)
    _complete_articulation(state_path)
    _complete_simple_stage(state_path, "material")
    directory = _begin(state_path, "texture")
    output = directory / "texture.usda"
    output.write_text('#usda 1.0\ndef Xform "Texture" {}\n', encoding="utf-8")
    evidence = _write_texture_evidence(
        state_path,
        directory,
        output,
        status=status,
    )

    with pytest.raises(
        AssetCompositionStateError,
        match="Texture native terminal status must pass before acceptance",
    ):
        _write_review(
            state_path,
            "texture",
            decision="accept",
            evidence=evidence,
            output=output,
        )

    run = load_verified_run(state_path)
    assert run.current_stage == "texture"
    assert run.coordinator.next_action == "execute_stage"


def test_texture_accept_requires_output_bound_native_evidence(tmp_path: Path) -> None:
    state_path = _create(tmp_path)
    _complete_articulation(state_path)
    _complete_simple_stage(state_path, "material")
    directory = _begin(state_path, "texture")
    output = directory / "texture.usda"
    output.write_text('#usda 1.0\ndef Xform "Texture" {}\n', encoding="utf-8")
    evidence = _write_texture_evidence(state_path, directory, output)
    validation_path = next(
        path for path in evidence if path.name == "validation_evidence.json"
    )
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["output_asset_path"] = str(directory / "other.usda")
    validation_path.write_text(json.dumps(validation) + "\n", encoding="utf-8")

    with pytest.raises(
        AssetCompositionStateError,
        match="does not bind the accepted output",
    ):
        _write_review(
            state_path,
            "texture",
            decision="accept",
            evidence=evidence,
            output=output,
        )


def test_texture_accept_requires_shared_decision_receipt(tmp_path: Path) -> None:
    state_path = _create(tmp_path)
    _complete_articulation(state_path)
    _complete_simple_stage(state_path, "material")
    directory = _begin(state_path, "texture")
    output = directory / "texture.usda"
    output.write_text('#usda 1.0\ndef Xform "Texture" {}\n', encoding="utf-8")
    evidence = _write_texture_evidence(state_path, directory, output)
    summary_path = next(path for path in evidence if path.name == "final_summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    receipt_path = Path(summary["artifacts"]["embedded_decision_receipt"])
    evidence = [path for path in evidence if path != receipt_path]

    with pytest.raises(
        AssetCompositionStateError,
        match="Texture embedded decision receipt",
    ):
        _write_review(
            state_path,
            "texture",
            decision="accept",
            evidence=evidence,
            output=output,
        )


def test_texture_accept_requires_complete_candidate_decision_chain(
    tmp_path: Path,
) -> None:
    state_path = _create(tmp_path)
    _complete_articulation(state_path)
    _complete_simple_stage(state_path, "material")
    directory = _begin(state_path, "texture")
    output = directory / "texture.usda"
    output.write_text('#usda 1.0\ndef Xform "Texture" {}\n', encoding="utf-8")
    evidence = _write_texture_evidence(state_path, directory, output)
    checkpoint_path = next(
        path for path in evidence if path.name == "workflow_checkpoint.json"
    )
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    embedded = checkpoint["embedded_decision_state"]
    for field_name in (
        "current_decision",
        "current_authorization",
        "current_result",
        "current_candidate_evidence",
        "current_candidate_domain_review",
        "current_candidate_review",
        "current_candidate_receipt",
        "accepted_candidate_result",
        "accepted_candidate_domain_review",
        "accepted_candidate_review",
        "accepted_candidate_receipt",
    ):
        embedded[field_name] = None
    checkpoint_path.write_text(json.dumps(checkpoint) + "\n", encoding="utf-8")

    with pytest.raises(
        AssetCompositionStateError,
        match="completed Texture state lacks its canonical plan or decision chain",
    ):
        _write_review(
            state_path,
            "texture",
            decision="accept",
            evidence=evidence,
            output=output,
        )


@pytest.mark.parametrize(
    "drift",
    [
        "frozen_identity",
        "checkpoint_plan_bytes",
        "proposal_reference",
        "inspection_state",
    ],
)
def test_texture_accept_rejects_frozen_decision_input_drift(
    tmp_path: Path,
    drift: str,
) -> None:
    state_path = _create(tmp_path)
    _complete_articulation(state_path)
    _complete_simple_stage(state_path, "material")
    directory = _begin(state_path, "texture")
    output = directory / "texture.usda"
    output.write_text('#usda 1.0\ndef Xform "Texture" {}\n', encoding="utf-8")
    evidence = _write_texture_evidence(state_path, directory, output)
    request_path = next(path for path in evidence if path.name == "request.json")
    checkpoint_path = next(
        path for path in evidence if path.name == "workflow_checkpoint.json"
    )
    if drift == "frozen_identity":
        request = json.loads(request_path.read_text(encoding="utf-8"))
        identity = request["metadata"][TEXTURE_EMBEDDED_DECISION_IDENTITY_METADATA_KEY]
        identity["digests"]["configuration"]["texture_request"] = "f" * 64
        request_path.write_text(json.dumps(request) + "\n", encoding="utf-8")
    else:
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        embedded = checkpoint["embedded_decision_state"]
        if drift == "checkpoint_plan_bytes":
            checkpoint["plan"]["decision"]["state"] = "drifted"
        elif drift == "proposal_reference":
            embedded["plan_proposal"]["sha256"] = "0" * 64
        else:
            embedded["inspection"]["units"][0]["uv_facts"]["primvars"][0]["status"] = (
                "missing"
            )
        checkpoint_path.write_text(json.dumps(checkpoint) + "\n", encoding="utf-8")

    with pytest.raises(AssetCompositionStateError):
        _write_review(
            state_path,
            "texture",
            decision="accept",
            evidence=evidence,
            output=output,
        )


def test_texture_accept_requires_shared_decision_journal(tmp_path: Path) -> None:
    state_path = _create(tmp_path)
    _complete_articulation(state_path)
    _complete_simple_stage(state_path, "material")
    directory = _begin(state_path, "texture")
    output = directory / "texture.usda"
    output.write_text('#usda 1.0\ndef Xform "Texture" {}\n', encoding="utf-8")
    evidence = _write_texture_evidence(state_path, directory, output)
    evidence = [path for path in evidence if path.name != "journal.json"]

    with pytest.raises(
        AssetCompositionStateError,
        match="Texture embedded decision journal",
    ):
        _write_review(
            state_path,
            "texture",
            decision="accept",
            evidence=evidence,
            output=output,
        )


def test_texture_accept_requires_terminal_progress_index(tmp_path: Path) -> None:
    state_path = _create(tmp_path)
    _complete_articulation(state_path)
    _complete_simple_stage(state_path, "material")
    directory = _begin(state_path, "texture")
    output = directory / "texture.usda"
    output.write_text('#usda 1.0\ndef Xform "Texture" {}\n', encoding="utf-8")
    evidence = _write_texture_evidence(state_path, directory, output)
    evidence = [path for path in evidence if path.name != "workflow_progress.json"]

    with pytest.raises(
        AssetCompositionStateError,
        match="Texture acceptance requires exactly one",
    ):
        _write_review(
            state_path,
            "texture",
            decision="accept",
            evidence=evidence,
            output=output,
        )


@pytest.mark.parametrize(
    "context_change",
    ["missing", "stale_attempt", "wrong_output_root"],
)
def test_texture_accept_requires_exact_embedded_execution_context(
    tmp_path: Path,
    context_change: str,
) -> None:
    state_path = _create(tmp_path)
    _complete_articulation(state_path)
    _complete_simple_stage(state_path, "material")
    directory = _begin(state_path, "texture")
    output = directory / "texture.usda"
    output.write_text('#usda 1.0\ndef Xform "Texture" {}\n', encoding="utf-8")
    evidence = _write_texture_evidence(state_path, directory, output)
    request_path = next(path for path in evidence if path.name == "request.json")
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if context_change == "missing":
        request["metadata"].pop(DOMAIN_EXECUTION_CONTEXT_METADATA_KEY)
    elif context_change == "stale_attempt":
        context = request["metadata"][DOMAIN_EXECUTION_CONTEXT_METADATA_KEY]
        context["embedded_stage"]["stage_attempt"] += 1
    else:
        context = request["metadata"][DOMAIN_EXECUTION_CONTEXT_METADATA_KEY]
        context["embedded_stage"]["domain_run_root"] = str(tmp_path / "elsewhere")
    request_path.write_text(json.dumps(request) + "\n", encoding="utf-8")

    with pytest.raises(
        AssetCompositionStateError,
        match=(
            "requires an embedded execution context"
            if context_change == "missing"
            else "differs from the active coordinator attempt"
        ),
    ):
        _write_review(
            state_path,
            "texture",
            decision="accept",
            evidence=evidence,
            output=output,
        )


def test_legacy_texture_attempt_can_resume_without_embedded_request_evidence(
    tmp_path: Path,
) -> None:
    state_path = _create(tmp_path)
    _complete_articulation(state_path)
    _complete_simple_stage(state_path, "material")
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    payload["schema_version"] = "content-agent-workflows.asset-composition-run.v1"
    state_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    directory = _begin(state_path, "texture")
    output = directory / "texture.usda"
    output.write_text('#usda 1.0\ndef Xform "Texture" {}\n', encoding="utf-8")
    evidence = _write_texture_evidence(
        state_path,
        directory,
        output,
        omit_execution_context=True,
        include_request_evidence=False,
    )
    _write_review(
        state_path,
        "texture",
        decision="accept",
        evidence=evidence,
        output=output,
    )
    complete_stage(
        state_path,
        "texture",
        output_asset=output,
        evidence_paths=evidence,
        summary="Accepted pre-upgrade Texture attempt.",
        actor="test-agent",
    )

    run = load_verified_run(state_path)
    assert run.schema_version == "content-agent-workflows.asset-composition-run.v1"
    assert run.current_stage == "physics"


def test_texture_accept_rejects_substituted_stage_input(tmp_path: Path) -> None:
    state_path = _create(tmp_path)
    _complete_articulation(state_path)
    _complete_simple_stage(state_path, "material")
    directory = _begin(state_path, "texture")
    output = directory / "texture.usda"
    output.write_text('#usda 1.0\ndef Xform "Texture" {}\n', encoding="utf-8")
    evidence = _write_texture_evidence(state_path, directory, output)
    summary_path = next(path for path in evidence if path.name == "final_summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["source_asset"] = str(directory / "stale-material.usda")
    summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")

    with pytest.raises(
        AssetCompositionStateError,
        match="does not bind a passing accepted output",
    ):
        _write_review(
            state_path,
            "texture",
            decision="accept",
            evidence=evidence,
            output=output,
        )


def test_texture_accept_rejects_contradictory_remaining_units(tmp_path: Path) -> None:
    state_path = _create(tmp_path)
    _complete_articulation(state_path)
    _complete_simple_stage(state_path, "material")
    directory = _begin(state_path, "texture")
    output = directory / "texture.usda"
    output.write_text('#usda 1.0\ndef Xform "Texture" {}\n', encoding="utf-8")
    evidence = _write_texture_evidence(state_path, directory, output)
    summary_path = next(path for path in evidence if path.name == "final_summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["remaining_unit_ids"] = ["unexpected-unit"]
    summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")

    with pytest.raises(
        AssetCompositionStateError,
        match="summary unit partition differs",
    ):
        _write_review(
            state_path,
            "texture",
            decision="accept",
            evidence=evidence,
            output=output,
        )


def test_texture_accept_requires_every_manifest_artifact_binding(
    tmp_path: Path,
) -> None:
    state_path = _create(tmp_path)
    _complete_articulation(state_path)
    _complete_simple_stage(state_path, "material")
    directory = _begin(state_path, "texture")
    output = directory / "texture.usda"
    output.write_text('#usda 1.0\ndef Xform "Texture" {}\n', encoding="utf-8")
    evidence = _write_texture_evidence(state_path, directory, output)
    incomplete = [path for path in evidence if path.name != "texture-validation.png"]

    with pytest.raises(
        AssetCompositionStateError,
        match="visual validation evidence must match exactly one sealed",
    ):
        _write_review(
            state_path,
            "texture",
            decision="accept",
            evidence=incomplete,
            output=output,
        )


def test_texture_handoff_rejects_post_acceptance_artifact_mutation(
    tmp_path: Path,
) -> None:
    state_path = _create(tmp_path)
    _complete_articulation(state_path)
    _complete_simple_stage(state_path, "material")
    _complete_simple_stage(state_path, "texture")
    artifact = next(
        binding
        for binding in load_verified_run(state_path).stages["texture"].evidence
        if Path(binding.path).name == "generated-texture.png"
    )
    Path(artifact.path).write_bytes(b"mutated texture")

    with pytest.raises(AssetCompositionStateError, match="identity changed"):
        load_verified_run(state_path)


def test_texture_accept_rejects_output_mutated_after_native_evidence(
    tmp_path: Path,
) -> None:
    state_path = _create(tmp_path)
    _complete_articulation(state_path)
    _complete_simple_stage(state_path, "material")
    directory = _begin(state_path, "texture")
    output = directory / "texture.usda"
    output.write_text('#usda 1.0\ndef Xform "Texture" {}\n', encoding="utf-8")
    evidence = _write_texture_evidence(state_path, directory, output)
    output.write_text('#usda 1.0\ndef Xform "Mutated" {}\n', encoding="utf-8")

    with pytest.raises(
        AssetCompositionStateError,
        match="output digest differs from accepted bytes",
    ):
        _write_review(
            state_path,
            "texture",
            decision="accept",
            evidence=evidence,
            output=output,
        )


def test_texture_accept_rejects_stale_native_evidence_schema(tmp_path: Path) -> None:
    state_path = _create(tmp_path)
    _complete_articulation(state_path)
    _complete_simple_stage(state_path, "material")
    directory = _begin(state_path, "texture")
    output = directory / "texture.usda"
    output.write_text('#usda 1.0\ndef Xform "Texture" {}\n', encoding="utf-8")
    evidence = _write_texture_evidence(state_path, directory, output)
    validation_path = next(
        path for path in evidence if path.name == "validation_evidence.json"
    )
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["schema_version"] = (
        "content-agent-workflows.texture-validation-evidence.v2"
    )
    validation_path.write_text(json.dumps(validation) + "\n", encoding="utf-8")

    with pytest.raises(AssetCompositionStateError, match="Invalid Texture"):
        _write_review(
            state_path,
            "texture",
            decision="accept",
            evidence=evidence,
            output=output,
        )


def test_material_accept_requires_explicit_dependency_closure_binding(
    tmp_path: Path,
) -> None:
    state_path = _create(tmp_path)
    _complete_articulation(state_path)
    directory = _begin(state_path, "material")
    output = directory / "material.usda"
    output.write_text('#usda 1.0\ndef Xform "Material" {}\n', encoding="utf-8")
    evidence = directory / "material-evidence.json"
    evidence.write_text('{"status":"pass"}\n', encoding="utf-8")
    evidence_paths = _write_material_evidence(
        state_path,
        directory,
        output,
        evidence,
    )
    coordinator_request = directory / "coordinator_request.json"
    request_payload = json.loads(coordinator_request.read_text(encoding="utf-8"))
    request_payload.pop("materials_usd_dependencies")
    coordinator_request.write_text(json.dumps(request_payload) + "\n", encoding="utf-8")
    coordinator_result = directory / "coordinator_result.json"
    result_payload = json.loads(coordinator_result.read_text(encoding="utf-8"))
    request_sha256 = file_sha256(coordinator_request)
    result_payload["request"]["sha256"] = request_sha256
    for item in result_payload["evidence"]:
        if item["path"] == str(coordinator_request):
            item["sha256"] = request_sha256
    coordinator_result.write_text(json.dumps(result_payload) + "\n", encoding="utf-8")

    with pytest.raises(
        AssetCompositionStateError,
        match="dependency bindings must be lists",
    ):
        _write_review(
            state_path,
            "material",
            decision="accept",
            evidence=evidence_paths,
            output=output,
        )


def test_material_accept_seals_every_result_evidence_path(tmp_path: Path) -> None:
    state_path = _create(tmp_path)
    _complete_articulation(state_path)
    directory = _begin(state_path, "material")
    output = directory / "material.usda"
    output.write_text('#usda 1.0\ndef Xform "Material" {}\n', encoding="utf-8")
    evidence = directory / "material-evidence.json"
    evidence.write_text('{"status":"pass"}\n', encoding="utf-8")
    evidence_paths = _write_material_evidence(
        state_path,
        directory,
        output,
        evidence,
    )

    with pytest.raises(
        AssetCompositionStateError,
        match="result evidence item 1 must be sealed exactly once",
    ):
        _write_review(
            state_path,
            "material",
            decision="accept",
            evidence=evidence_paths[1:],
            output=output,
        )


def test_material_accept_binds_scene_tool_timeout_to_frozen_request(
    tmp_path: Path,
) -> None:
    state_path = _create(tmp_path)
    _complete_articulation(state_path)
    directory = _begin(state_path, "material")
    output = directory / "material.usda"
    output.write_text('#usda 1.0\ndef Xform "Material" {}\n', encoding="utf-8")
    evidence = directory / "material-evidence.json"
    evidence.write_text('{"status":"pass"}\n', encoding="utf-8")
    evidence_paths = _write_material_evidence(
        state_path,
        directory,
        output,
        evidence,
    )
    coordinator_request = directory / "coordinator_request.json"
    request_payload = json.loads(coordinator_request.read_text(encoding="utf-8"))
    request_payload["scene_tool_timeout_seconds"] = 120.0
    coordinator_request.write_text(
        json.dumps(request_payload) + "\n",
        encoding="utf-8",
    )
    coordinator_result = directory / "coordinator_result.json"
    result_payload = json.loads(coordinator_result.read_text(encoding="utf-8"))
    request_sha256 = file_sha256(coordinator_request)
    result_payload["request"]["sha256"] = request_sha256
    for item in result_payload["evidence"]:
        if item["path"] == str(coordinator_request):
            item["sha256"] = request_sha256
    coordinator_result.write_text(
        json.dumps(result_payload) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        AssetCompositionStateError,
        match="scene-tool timeout differs from the frozen asset request",
    ):
        _write_review(
            state_path,
            "material",
            decision="accept",
            evidence=evidence_paths,
            output=output,
        )


def test_material_accept_requires_canonical_executor_evidence(tmp_path: Path) -> None:
    state_path = _create(tmp_path)
    _complete_articulation(state_path)
    directory = _begin(state_path, "material")
    output = directory / "material.usda"
    output.write_text('#usda 1.0\ndef Xform "Material" {}\n', encoding="utf-8")
    evidence = directory / "material-evidence.json"
    evidence.write_text('{"status":"pass"}\n', encoding="utf-8")
    evidence_paths = _write_material_evidence(
        state_path,
        directory,
        output,
        evidence,
    )
    omitted = directory / "visual_quality_assessment.json"
    coordinator_result = directory / "coordinator_result.json"
    result_payload = json.loads(coordinator_result.read_text(encoding="utf-8"))
    result_payload["evidence"] = [
        item for item in result_payload["evidence"] if item["path"] != str(omitted)
    ]
    coordinator_result.write_text(json.dumps(result_payload) + "\n", encoding="utf-8")

    with pytest.raises(
        AssetCompositionStateError,
        match="omits canonical evidence: visual_quality_assessment.json",
    ):
        _write_review(
            state_path,
            "material",
            decision="accept",
            evidence=[path for path in evidence_paths if path != omitted],
            output=output,
        )


def test_material_accept_requires_sealed_post_apply_review(tmp_path: Path) -> None:
    state_path = _create(tmp_path)
    _complete_articulation(state_path)
    directory = _begin(state_path, "material")
    output = directory / "material.usda"
    output.write_text('#usda 1.0\ndef Xform "Material" {}\n', encoding="utf-8")
    evidence = directory / "material-evidence.json"
    evidence.write_text('{"status":"pass"}\n', encoding="utf-8")
    evidence_paths = _write_material_evidence(
        state_path,
        directory,
        output,
        evidence,
    )
    omitted = directory / "raw" / "material_post_apply_review.json"
    coordinator_result = directory / "coordinator_result.json"
    result_payload = json.loads(coordinator_result.read_text(encoding="utf-8"))
    result_payload["evidence"] = [
        item for item in result_payload["evidence"] if item["path"] != str(omitted)
    ]
    coordinator_result.write_text(json.dumps(result_payload) + "\n", encoding="utf-8")

    with pytest.raises(
        AssetCompositionStateError,
        match="omits canonical evidence: raw/material_post_apply_review.json",
    ):
        _write_review(
            state_path,
            "material",
            decision="accept",
            evidence=[path for path in evidence_paths if path != omitted],
            output=output,
        )


@pytest.mark.parametrize(
    ("relative_path", "mutation", "message"),
    [
        (
            "assignments.json",
            lambda payload: payload["materialized_usd"].update(status="partial"),
            "must report a succeeded materialized USD",
        ),
        (
            "visual_quality_assessment.json",
            lambda payload: payload.update(
                status="unresolved_issues", unresolved_issues=["wrong finish"]
            ),
            "embed a different visual quality assessment",
        ),
        (
            "validation_evidence.json",
            lambda payload: payload.update(sim_ready_status="fail"),
            "does not prove a clean pass",
        ),
        (
            "raw/material_restore_response.json",
            lambda payload: payload.update(
                unresolved_mappings=[{"source_path": "/World/Drawer"}]
            ),
            "does not prove complete source coverage",
        ),
        (
            "raw/material_post_apply_review.json",
            lambda payload: payload.update(checked_view_bindings=[]),
            "differs from the applied final renders",
        ),
        (
            "raw/final_render_records.json",
            lambda payload: payload["turntable"].update(frame_count=1),
            "lack the required OVRTX turntable",
        ),
    ],
)
def test_material_accept_rejects_contradictory_native_evidence(
    tmp_path: Path,
    relative_path: str,
    mutation: object,
    message: str,
) -> None:
    state_path = _create(tmp_path)
    _complete_articulation(state_path)
    directory = _begin(state_path, "material")
    output = directory / "material.usda"
    output.write_text('#usda 1.0\ndef Xform "Material" {}\n', encoding="utf-8")
    evidence = directory / "material-evidence.json"
    evidence.write_text('{"status":"pass"}\n', encoding="utf-8")
    evidence_paths = _write_material_evidence(
        state_path,
        directory,
        output,
        evidence,
    )
    native_path = directory / relative_path
    native_payload = json.loads(native_path.read_text(encoding="utf-8"))
    assert callable(mutation)
    mutation(native_payload)
    native_path.write_text(json.dumps(native_payload) + "\n", encoding="utf-8")
    coordinator_result = directory / "coordinator_result.json"
    result_payload = json.loads(coordinator_result.read_text(encoding="utf-8"))
    for item in result_payload["evidence"]:
        if item["path"] == str(native_path):
            item["sha256"] = file_sha256(native_path)
    coordinator_result.write_text(json.dumps(result_payload) + "\n", encoding="utf-8")

    with pytest.raises(AssetCompositionStateError, match=message):
        _write_review(
            state_path,
            "material",
            decision="accept",
            evidence=evidence_paths,
            output=output,
        )


def test_material_accept_rejects_result_evidence_changed_after_finalization(
    tmp_path: Path,
) -> None:
    state_path = _create(tmp_path)
    _complete_articulation(state_path)
    directory = _begin(state_path, "material")
    output = directory / "material.usda"
    output.write_text('#usda 1.0\ndef Xform "Material" {}\n', encoding="utf-8")
    evidence = directory / "material-evidence.json"
    evidence.write_text('{"status":"pass","version":1}\n', encoding="utf-8")
    evidence_paths = _write_material_evidence(
        state_path,
        directory,
        output,
        evidence,
    )
    evidence.write_text(
        '{"status":"pass","version":2}\n',
        encoding="utf-8",
    )

    with pytest.raises(
        AssetCompositionStateError,
        match="result evidence item 1 must be sealed exactly once",
    ):
        _write_review(
            state_path,
            "material",
            decision="accept",
            evidence=evidence_paths,
            output=output,
        )


def test_legacy_material_completion_does_not_require_coordinator_packet(
    tmp_path: Path,
) -> None:
    state_path = _create(tmp_path, legacy=True)
    begin_stage(state_path, "articulation", actor="legacy-test")
    first_attempt = stage_directory(state_path, "articulation")
    candidates = first_attempt / "candidates.json"
    candidates.write_text('{"joints":[]}\n', encoding="utf-8")
    require_review(state_path, candidates_path=candidates, actor="legacy-test")
    decisions = state_path.parent / "raw" / "legacy-decisions.json"
    decisions.write_text('{"accepted":[]}\n', encoding="utf-8")
    record_review_decisions(
        state_path,
        decisions_path=decisions,
        reviewer="legacy-reviewer",
    )
    begin_stage(state_path, "articulation", actor="legacy-test")
    articulation_dir = stage_directory(state_path, "articulation")
    articulated = articulation_dir / "articulated.usda"
    articulated.write_text(
        '#usda 1.0\ndef Xform "Articulated" {}\n',
        encoding="utf-8",
    )
    articulation_evidence = articulation_dir / "evidence.json"
    articulation_evidence.write_text('{"status":"pass"}\n', encoding="utf-8")
    complete_stage(
        state_path,
        "articulation",
        output_asset=articulated,
        evidence_paths=[articulation_evidence],
        summary="Legacy articulation accepted.",
        actor="legacy-test",
    )

    begin_stage(state_path, "material", actor="legacy-test")
    material_dir = stage_directory(state_path, "material")
    materialized = material_dir / "material.usda"
    materialized.write_text(
        '#usda 1.0\ndef Xform "Material" {}\n',
        encoding="utf-8",
    )
    material_evidence = material_dir / "evidence.json"
    material_evidence.write_text('{"status":"pass"}\n', encoding="utf-8")
    completed = complete_stage(
        state_path,
        "material",
        output_asset=materialized,
        evidence_paths=[material_evidence],
        summary="Legacy Material accepted.",
        actor="legacy-test",
    )

    assert completed.stages["material"].status == "completed"


def test_articulation_accept_requires_frozen_joint_review(tmp_path: Path) -> None:
    state_path = _create(tmp_path)
    directory = _begin(state_path, "articulation")
    output = directory / "articulated.usda"
    output.write_text('#usda 1.0\ndef Xform "Articulated" {}\n', encoding="utf-8")
    evidence = directory / "joint-evidence.json"
    evidence.write_text('{"status":"pass"}\n', encoding="utf-8")

    with pytest.raises(
        AssetCompositionStateError,
        match="requires frozen Joint review decisions",
    ):
        _write_review(
            state_path,
            "articulation",
            decision="accept",
            evidence=[evidence],
            output=output,
        )

    run = load_verified_run(state_path)
    assert run.coordinator.next_action == "execute_stage"


def test_articulation_accept_rejects_receipt_for_stale_candidate_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_path = _create(tmp_path)
    monkeypatch.setattr(
        articulation_api,
        "verify_articulation_workflow_summary",
        lambda _checkpoint, *, output_dir, authoring: None,
    )

    with pytest.raises(
        AssetCompositionStateError,
        match="Review receipt candidate digest does not match current evidence",
    ):
        _complete_articulation(
            state_path,
            stale_candidate_receipt=True,
            stub_terminal_summary=True,
        )

    run = load_verified_run(state_path)
    assert run.current_stage == "articulation"
    assert run.coordinator.next_action == "execute_stage"


def test_articulation_accept_rejects_a_different_receipt_reviewer(
    tmp_path: Path,
) -> None:
    state_path = _create(tmp_path)

    with pytest.raises(
        AssetCompositionStateError,
        match="receipt reviewer differs from the recorded human reviewer",
    ):
        _complete_articulation(state_path, receipt_reviewer="different-reviewer")

    run = load_verified_run(state_path)
    assert run.current_stage == "articulation"
    assert run.coordinator.next_action == "execute_stage"


def test_articulation_accept_requires_embedded_execution_context(
    tmp_path: Path,
) -> None:
    state_path = _create(tmp_path)

    with pytest.raises(
        AssetCompositionStateError,
        match="requires an embedded execution context",
    ):
        _complete_articulation(state_path, omit_execution_context=True)

    run = load_verified_run(state_path)
    assert run.current_stage == "articulation"
    assert run.coordinator.next_action == "execute_stage"


def test_articulation_accept_requires_coordinator_decision_ledger(
    tmp_path: Path,
) -> None:
    state_path = _create(tmp_path)

    with pytest.raises(
        AssetCompositionStateError,
        match="Articulation acceptance requires exactly one",
    ):
        _complete_articulation(
            state_path,
            omit_decision_ledger_evidence=True,
        )

    run = load_verified_run(state_path)
    assert run.current_stage == "articulation"
    assert run.coordinator.next_action == "execute_stage"


def test_articulation_accept_rejects_stripped_terminal_checkpoint_bindings(
    tmp_path: Path,
) -> None:
    state_path = _create(tmp_path)

    with pytest.raises(
        AssetCompositionStateError,
        match="Invalid embedded Articulation terminal checkpoint",
    ):
        _complete_articulation(
            state_path,
            strip_terminal_checkpoint_bindings=True,
        )

    run = load_verified_run(state_path)
    assert run.current_stage == "articulation"
    assert run.coordinator.next_action == "execute_stage"


@pytest.mark.parametrize(
    "artifact_name",
    (
        "inference_result.json",
        "authoring_request.json",
        "workflow_progress.json",
        "final_summary.json",
    ),
)
def test_articulation_accept_requires_every_terminal_artifact_binding(
    tmp_path: Path,
    artifact_name: str,
) -> None:
    state_path = _create(tmp_path)

    with pytest.raises(
        AssetCompositionStateError,
        match="Articulation acceptance requires exactly one",
    ):
        _complete_articulation(
            state_path,
            omit_articulation_evidence_name=artifact_name,
        )

    run = load_verified_run(state_path)
    assert run.current_stage == "articulation"
    assert run.coordinator.next_action == "execute_stage"


def test_articulation_workflow_progress_remains_sealed_after_acceptance(
    tmp_path: Path,
) -> None:
    state_path = _create(tmp_path)
    _complete_articulation(state_path)
    completed = load_verified_run(state_path)
    progress_binding = next(
        binding
        for binding in completed.stages["articulation"].evidence
        if Path(binding.path).name == "workflow_progress.json"
    )

    Path(progress_binding.path).write_text(
        '{"phase":"tampered"}\n',
        encoding="utf-8",
    )

    with pytest.raises(AssetCompositionStateError, match="identity changed"):
        load_verified_run(state_path)


@pytest.mark.parametrize(
    "artifact_name",
    (
        "predictions.json",
        "candidate_report.json",
        "joint_rigger_diagnostics.json",
        "joint_rigger_result.json",
    ),
)
def test_articulation_accept_requires_every_transitive_artifact_binding(
    tmp_path: Path,
    artifact_name: str,
) -> None:
    state_path = _create(tmp_path)

    with pytest.raises(
        AssetCompositionStateError,
        match="must match exactly one sealed evidence artifact",
    ):
        _complete_articulation(
            state_path,
            omit_articulation_evidence_name=artifact_name,
        )

    run = load_verified_run(state_path)
    assert run.current_stage == "articulation"
    assert run.coordinator.next_action == "execute_stage"


def test_articulation_accept_requires_every_scene_artifact_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_path = _create(tmp_path)
    monkeypatch.setattr(
        articulation_api,
        "validate_completed_articulation_checkpoint",
        lambda _checkpoint, *, request: None,
    )
    monkeypatch.setattr(
        articulation_api,
        "verify_articulation_workflow_summary",
        lambda _checkpoint, *, output_dir, authoring: None,
    )

    with pytest.raises(
        AssetCompositionStateError,
        match="Articulation scene nested evidence must match exactly one",
    ):
        _complete_articulation(
            state_path,
            include_scene_evidence=True,
            omit_scene_nested_evidence=True,
            stub_terminal_summary=True,
        )

    run = load_verified_run(state_path)
    assert run.current_stage == "articulation"
    assert run.coordinator.next_action == "execute_stage"


def test_legacy_articulation_attempt_can_resume_without_embedded_request_evidence(
    tmp_path: Path,
) -> None:
    state_path = _create(tmp_path)
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    payload["schema_version"] = "content-agent-workflows.asset-composition-run.v1"
    state_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    _complete_articulation(
        state_path,
        omit_execution_context=True,
        include_request_evidence=False,
    )

    run = load_verified_run(state_path)
    assert run.schema_version == "content-agent-workflows.asset-composition-run.v1"
    assert run.current_stage == "material"


def test_state_cli_records_plan_and_evidence_review(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    state_path = _create(tmp_path)
    plan_draft = state_path.parent / "raw" / "cli-plan.json"
    plan_draft.write_text(
        json.dumps(
            {
                "schema_version": (
                    "content-agent-workflows.asset-coordinator-plan-draft.v1"
                ),
                "stage": "articulation",
                "objective": "Inspect and classify the articulation evidence.",
                "steps": [
                    {
                        "stage": "articulation",
                        "objective": "Run the typed Joint executor.",
                        "acceptance_evidence": ["reviewed candidates"],
                        "may_revisit": False,
                    }
                ],
                "evidence_paths": [str(state_path.parent / "request.json")],
                "revision_reason": "Initial prompt and source review.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert (
        state_cli_main(
            [
                "record-plan",
                "--run-state",
                str(state_path),
                "--plan-file",
                str(plan_draft),
            ]
        )
        == 0
    )
    assert (
        state_cli_main(
            [
                "begin-stage",
                "--run-state",
                str(state_path),
                "--stage",
                "articulation",
            ]
        )
        == 0
    )
    stage_dir = stage_directory(state_path, "articulation")
    evidence = stage_dir / "candidate-evidence.json"
    evidence.write_text('{"status":"pass"}\n', encoding="utf-8")
    review_draft = state_path.parent / "raw" / "cli-review.json"
    review_draft.write_text(
        json.dumps(
            {
                "schema_version": (
                    "content-agent-workflows.asset-coordinator-review-draft.v1"
                ),
                "stage": "articulation",
                "output_asset_path": None,
                "evidence_paths": [str(evidence)],
                "findings": ["The typed executor produced reviewable evidence."],
                "decision": "await_review",
                "target_stage": None,
                "decision_summary": "Pause for exact Joint review decisions.",
                "repair_scope": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert (
        state_cli_main(
            [
                "record-evidence-review",
                "--run-state",
                str(state_path),
                "--review-file",
                str(review_draft),
            ]
        )
        == 0
    )

    assert load_verified_run(state_path).coordinator.next_action == "await_human_review"
    capsys.readouterr()


def test_interactive_and_batch_share_one_coordinator_implementation(
    tmp_path: Path,
) -> None:
    observed: list[tuple[str, str]] = []

    def reason(session: AssetCoordinatorSession) -> int:
        observed.append(
            (session.mode, load_verified_run(session.run_state_path).run_id)
        )
        _write_plan(session.run_state_path, "articulation")
        session.fail_stage(
            "articulation",
            reason="Stop after proving the shared coordinator invocation.",
        )
        return 0

    interactive = run_interactive_asset_coordinator(
        _create(tmp_path, "interactive"),
        reasoning_loop=reason,
    )
    batch = run_batch_asset_coordinator(
        _create(tmp_path, "batch"),
        reasoning_loop=reason,
    )

    assert observed == [("interactive", "interactive"), ("batch", "batch")]
    assert interactive.run.coordinator.next_action == "stopped"
    assert batch.run.coordinator.next_action == "stopped"
    assert (
        interactive.run.coordinator.plan_revisions[0].sha256
        != batch.run.coordinator.plan_revisions[0].sha256
    )


@pytest.mark.parametrize(
    "entrypoint",
    [run_interactive_asset_coordinator, run_batch_asset_coordinator],
)
def test_shared_coordinator_rejects_intermediate_callback_return(
    tmp_path: Path,
    entrypoint: Callable[..., AssetCoordinatorLoopResult],
) -> None:
    state_path = _create(tmp_path, "intermediate-return")

    def reason(session: AssetCoordinatorSession) -> int:
        _write_plan(session.run_state_path, "articulation")
        return 0

    with pytest.raises(
        AssetCompositionStateError,
        match=r"returned before reaching.*next_action=begin_stage",
    ):
        entrypoint(state_path, reasoning_loop=reason)


def test_asset_coordinator_rejects_a_competing_reasoning_loop(tmp_path: Path) -> None:
    state_path = _create(tmp_path, "leased")

    def reason(session: AssetCoordinatorSession) -> None:
        lease_path = asset_coordinator._coordinator_lease_path(state_path)
        assert not lease_path.is_relative_to(state_path.parent)
        state_path.with_name(f".{state_path.name}.coordinator.lock").unlink(
            missing_ok=True
        )
        with pytest.raises(
            AssetCompositionStateError,
            match="already owns this run",
        ):
            run_batch_asset_coordinator(
                state_path,
                reasoning_loop=lambda _competing_session: None,
            )
        session.fail_stage(
            "articulation",
            reason="Stop after proving competing coordinators are rejected.",
        )

    result = run_interactive_asset_coordinator(state_path, reasoning_loop=reason)

    assert result.run.run_id == "leased"


def test_asset_coordinator_rejects_a_transient_mode_downgrade(
    tmp_path: Path,
) -> None:
    state_path = _create(tmp_path, "mode-downgrade")

    def reason(session: AssetCoordinatorSession) -> None:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        payload["coordinator"]["mode"] = "legacy"
        state_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        try:
            with pytest.raises(
                AssetCompositionStateError,
                match="Frozen request coordinator mode differs",
            ):
                begin_stage(state_path, "articulation")
        finally:
            payload["coordinator"]["mode"] = "single_reasoning_loop"
            state_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        session.fail_stage(
            "articulation",
            reason="Stop after proving transient mode drift is rejected.",
        )

    result = run_interactive_asset_coordinator(state_path, reasoning_loop=reason)

    assert result.run.stages["articulation"].status == "failed"
    assert result.run.coordinator.mode == "single_reasoning_loop"


def test_migrated_coordinator_rejects_a_transient_mode_downgrade(
    tmp_path: Path,
) -> None:
    state_path = _create(tmp_path, "migrated-mode-downgrade", legacy=True)

    def reason(session: AssetCoordinatorSession) -> None:
        upgraded_request = json.loads(
            (state_path.parent / "request.json").read_text(encoding="utf-8")
        )
        assert upgraded_request["coordinator_mode"] == "single_reasoning_loop"
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        payload["coordinator"]["mode"] = "legacy"
        state_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        try:
            with pytest.raises(
                AssetCompositionStateError,
                match="Frozen request coordinator mode differs",
            ):
                begin_stage(state_path, "articulation")
        finally:
            payload["coordinator"]["mode"] = "single_reasoning_loop"
            state_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        session.fail_stage(
            "articulation",
            reason="Stop after proving migrated mode drift is rejected.",
        )

    result = run_interactive_asset_coordinator(state_path, reasoning_loop=reason)

    assert result.run.stages["articulation"].status == "failed"
    assert result.run.coordinator.mode == "single_reasoning_loop"


def test_state_cli_transition_revalidates_frozen_inputs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    state_path = _create(tmp_path, "batch-transition-drift")
    _write_plan(state_path, "articulation")
    payload = json.loads(
        (state_path.parent / "request.json").read_text(encoding="utf-8")
    )
    Path(payload["joint_config"]).write_text(
        "review_policy: changed\n",
        encoding="utf-8",
    )

    assert (
        state_cli_main(
            [
                "begin-stage",
                "--run-state",
                str(state_path),
                "--stage",
                "articulation",
            ]
        )
        == 2
    )
    assert "Joint config identity changed" in capsys.readouterr().err
    run = load_verified_run(state_path)
    assert len(run.coordinator.plan_revisions) == 1
    assert run.stages["articulation"].status == "ready"


def test_state_cli_record_review_respects_active_coordinator_lease(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    state_path = _create(tmp_path, "leased-cli-review")

    def reason(session: AssetCoordinatorSession) -> None:
        assert (
            state_cli_main(
                [
                    "record-review",
                    "--run-state",
                    str(state_path),
                    "--decisions",
                    str(state_path.parent / "decisions.json"),
                    "--reviewer",
                    "operator",
                ]
            )
            == 2
        )
        assert "already owns this run" in capsys.readouterr().err
        session.fail_stage(
            "articulation",
            reason="Stop after proving CLI transitions respect the lease.",
        )

    result = run_interactive_asset_coordinator(state_path, reasoning_loop=reason)

    assert result.run.run_id == "leased-cli-review"


def test_state_cli_recover_respects_active_coordinator_lease(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    state_path = _create(tmp_path, "leased-cli-recover")
    _write_plan(state_path, "articulation")
    begin_stage(state_path, "articulation", actor="test-agent")
    fail_stage(
        state_path,
        "articulation",
        reason="Executor stopped before review.",
    )

    def reason(_session: AssetCoordinatorSession) -> None:
        assert (
            state_cli_main(
                [
                    "recover-stage",
                    "--run-state",
                    str(state_path),
                    "--stage",
                    "articulation",
                    "--reason",
                    "Competing recovery.",
                ]
            )
            == 2
        )
        assert "already owns this run" in capsys.readouterr().err

    result = run_interactive_asset_coordinator(state_path, reasoning_loop=reason)

    assert result.run.terminal_status == "failed"
    assert result.run.stages["articulation"].status == "failed"


def test_asset_coordinator_does_not_relabel_inner_filelock_timeout(
    tmp_path: Path,
) -> None:
    state_path = _create(tmp_path, "inner-timeout")

    def time_out(_session: AssetCoordinatorSession) -> None:
        raise Timeout("domain-executor.lock")

    with pytest.raises(Timeout, match="domain-executor.lock"):
        run_interactive_asset_coordinator(
            state_path,
            reasoning_loop=time_out,
        )


def test_asset_coordinator_releases_lease_when_reasoning_callback_raises(
    tmp_path: Path,
) -> None:
    state_path = _create(tmp_path, "callback-error-release")

    def crash(_session: AssetCoordinatorSession) -> None:
        raise RuntimeError("reasoning callback failed")

    with pytest.raises(RuntimeError, match="reasoning callback failed"):
        run_interactive_asset_coordinator(state_path, reasoning_loop=crash)

    def stop(session: AssetCoordinatorSession) -> None:
        _write_plan(session.run_state_path, "articulation")
        session.fail_stage(
            "articulation",
            reason="Stop after proving the released lease can be reacquired.",
        )

    result = run_batch_asset_coordinator(state_path, reasoning_loop=stop)

    assert result.run.terminal_status == "failed"
    assert result.run.stages["articulation"].status == "failed"


def test_unknown_stage_cannot_bypass_coordinator_acceptance_validation(
    tmp_path: Path,
) -> None:
    state_path = _create(tmp_path, "unknown-acceptance-stage")
    stage_dir = _begin(state_path, "articulation")
    run = load_verified_run(state_path)
    state = run.stages["articulation"]
    assert state.input_asset is not None
    output = stage_dir / "output.usda"
    output.write_text('#usda 1.0\ndef Xform "World" {}\n', encoding="utf-8")
    output_binding = ArtifactBinding(
        path=str(output.resolve()),
        sha256=file_sha256(output),
        size_bytes=output.stat().st_size,
    )

    with pytest.raises(
        AssetCompositionStateError,
        match="No coordinator acceptance validator is registered",
    ):
        asset_state._validate_stage_acceptance(
            "future-stage",  # type: ignore[arg-type]
            state_path=state_path,
            run=run,
            input_asset=state.input_asset,
            input_dependencies=state.input_dependencies,
            output=output_binding,
            output_dependencies=[],
            evidence=[],
        )


def test_interactive_session_can_fail_recover_and_cancel_without_evidence(
    tmp_path: Path,
) -> None:
    state_path = _create(tmp_path, "typed-stop")

    def reason(session: AssetCoordinatorSession) -> None:
        session.fail_stage(
            "articulation",
            reason="usd-cli was unreachable before executor evidence existed.",
        )

    failed = run_interactive_asset_coordinator(state_path, reasoning_loop=reason)
    assert failed.run.terminal_status == "failed"

    session = AssetCoordinatorSession(run_state_path=state_path, mode="interactive")
    recovered = session.recover_stage(
        "articulation",
        reason="usd-cli availability was restored.",
    )
    assert recovered.terminal_status == "active"
    cancelled = session.cancel_stage(
        "articulation",
        reason="Operator cancelled before retry.",
    )
    assert cancelled.terminal_status == "cancelled"


def test_interactive_coordinator_rejects_frozen_input_drift_before_reasoning(
    tmp_path: Path,
) -> None:
    state_path = _create(tmp_path, "interactive-drift")
    request = json.loads(
        (state_path.parent / "request.json").read_text(encoding="utf-8")
    )
    Path(request["joint_config"]).write_text(
        "review_policy: changed\n",
        encoding="utf-8",
    )
    invoked = False

    def reason(_session: AssetCoordinatorSession) -> None:
        nonlocal invoked
        invoked = True

    with pytest.raises(
        AssetCompositionStateError, match="Joint config identity changed"
    ):
        run_interactive_asset_coordinator(state_path, reasoning_loop=reason)

    assert not invoked


def test_interactive_session_revalidates_frozen_inputs_before_transition(
    tmp_path: Path,
) -> None:
    state_path = _create(tmp_path, "interactive-transition-drift")
    request = json.loads(
        (state_path.parent / "request.json").read_text(encoding="utf-8")
    )
    joint_config = Path(request["joint_config"])
    original = joint_config.read_text(encoding="utf-8")

    def reason(session: AssetCoordinatorSession) -> None:
        joint_config.write_text("review_policy: changed\n", encoding="utf-8")
        try:
            with pytest.raises(
                AssetCompositionStateError,
                match="Joint config identity changed",
            ):
                session.fail_stage(
                    "articulation",
                    reason="This transition must not accept drifted inputs.",
                )
        finally:
            joint_config.write_text(original, encoding="utf-8")
        session.fail_stage(
            "articulation",
            reason="Stop after proving interactive transitions revalidate inputs.",
        )

    result = run_interactive_asset_coordinator(state_path, reasoning_loop=reason)

    assert result.run.terminal_status == "failed"
    assert result.run.stages["articulation"].status == "failed"


def test_batch_coordinator_migrates_legacy_run_before_reasoning(
    tmp_path: Path,
) -> None:
    state_path = _create(tmp_path, "legacy", legacy=True)
    observed_modes: list[str] = []

    def reason(session: AssetCoordinatorSession) -> None:
        observed_modes.append(session.status().coordinator.mode)
        session.fail_stage(
            "articulation",
            reason="Stop after proving legacy migration precedes reasoning.",
        )

    result = run_batch_asset_coordinator(state_path, reasoning_loop=reason)

    assert result.returncode == 0
    assert observed_modes == ["single_reasoning_loop"]
    assert result.run.coordinator.next_action == "stopped"
    upgraded_request = json.loads(
        (state_path.parent / "request.json").read_text(encoding="utf-8")
    )
    assert upgraded_request["runtime"]["scene_tool_timeout_seconds"] == 60.0
    assert (
        not {
            "workbench_url",
            "start_workbench",
            "keep_workbench",
            "workbench_timeout_seconds",
        }
        & upgraded_request["runtime"].keys()
    )


def test_active_run_reads_historical_runtime_without_mutating_frozen_request(
    tmp_path: Path,
) -> None:
    state_path = _create(
        tmp_path,
        "historical-runtime",
        historical_runtime=True,
    )

    run = load_verified_run(state_path)
    request = asset_state.load_verified_asset_request(state_path, run=run)

    assert request.runtime.scene_tool_timeout_seconds == 60.0
    frozen = json.loads(
        (state_path.parent / "request.json").read_text(encoding="utf-8")
    )
    assert "scene_tool_timeout_seconds" not in frozen["runtime"]
    assert frozen["runtime"]["workbench_timeout_seconds"] == 60.0


def test_interactive_session_exercises_shared_typed_surface(tmp_path: Path) -> None:
    state_path = _create(tmp_path, "typed-surface")

    def reason(session: AssetCoordinatorSession) -> None:
        plan = state_path.parent / "raw" / "session-plan.json"
        plan.write_text(
            json.dumps(
                {
                    "schema_version": (
                        "content-agent-workflows.asset-coordinator-plan-draft.v1"
                    ),
                    "stage": "articulation",
                    "objective": "Inspect articulation candidates.",
                    "steps": [
                        {
                            "stage": "articulation",
                            "objective": "Run the typed Joint executor.",
                            "acceptance_evidence": ["candidate evidence"],
                            "may_revisit": False,
                        }
                    ],
                    "evidence_paths": [str(state_path.parent / "request.json")],
                    "revision_reason": "Initial interactive plan.",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        session.record_plan(plan, actor="interactive-test")
        planned_directory = session.stage_directory("articulation")
        session.begin_stage("articulation", actor="interactive-test")
        assert session.stage_directory("articulation") == planned_directory
        candidates = planned_directory / "candidates.json"
        candidates.write_text('{"candidate_ids":["drawer"]}\n', encoding="utf-8")
        review = state_path.parent / "raw" / "session-review.json"
        review.write_text(
            json.dumps(
                {
                    "schema_version": (
                        "content-agent-workflows.asset-coordinator-review-draft.v1"
                    ),
                    "stage": "articulation",
                    "output_asset_path": None,
                    "evidence_paths": [str(candidates)],
                    "findings": ["Review the exact drawer candidate."],
                    "decision": "await_review",
                    "target_stage": None,
                    "decision_summary": "Pause for Joint review.",
                    "repair_scope": [],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        session.review_evidence(review, actor="interactive-test")
        session.require_articulation_review(candidates, actor="interactive-test")

    result = run_interactive_asset_coordinator(state_path, reasoning_loop=reason)

    assert result.returncode == 0
    assert result.run.stages["articulation"].status == "needs_review"
    assert result.run.coordinator.next_action == "await_human_review"


def test_typed_session_completes_an_exact_reviewed_stage(tmp_path: Path) -> None:
    state_path = _create(tmp_path, "session-complete")
    _complete_articulation(state_path)
    directory = _begin(state_path, "material")
    output = directory / "material.usda"
    output.write_text('#usda 1.0\ndef Xform "Material" {}\n', encoding="utf-8")
    evidence = directory / "material-evidence.json"
    evidence.write_text('{"status":"pass"}\n', encoding="utf-8")
    evidence_paths = _write_material_evidence(
        state_path,
        directory,
        output,
        evidence,
    )
    _write_review(
        state_path,
        "material",
        decision="accept",
        evidence=evidence_paths,
        output=output,
    )
    session = AssetCoordinatorSession(
        run_state_path=state_path,
        mode="interactive",
    )

    completed = session.complete_stage(
        "material",
        output_asset=output,
        evidence_paths=evidence_paths,
        summary="Accepted exact Material evidence.",
        actor="interactive-test",
    )

    assert completed.stages["material"].status == "completed"
    assert completed.current_stage == "texture"


def test_shared_coordinator_rejects_negative_reasoning_status(tmp_path: Path) -> None:
    with pytest.raises(
        AssetCompositionStateError,
        match="reasoning loop returned a negative status",
    ):
        run_interactive_asset_coordinator(
            _create(tmp_path, "negative-status"),
            reasoning_loop=lambda _session: -9,
        )


def test_second_joint_review_cannot_replace_frozen_receipt(tmp_path: Path) -> None:
    state_path = _create(tmp_path)
    first_attempt = _begin(state_path, "articulation")
    first_candidates = first_attempt / "articulation-candidates.json"
    first_candidates.write_text('{"candidate_ids":["drawer"]}\n', encoding="utf-8")
    _write_review(
        state_path,
        "articulation",
        decision="await_review",
        evidence=[first_candidates],
    )
    require_review(state_path, candidates_path=first_candidates, actor="test-agent")
    decisions = state_path.parent / "raw" / "joint-decisions.json"
    decisions.write_text('{"drawer":"accept"}\n', encoding="utf-8")
    recorded = record_review_decisions(
        state_path,
        decisions_path=decisions,
        reviewer="asset-owner",
    )
    receipt = recorded.stages["articulation"].review_decisions
    assert receipt is not None

    _write_plan(
        state_path,
        "articulation",
        evidence=first_candidates,
        reason="Author the reviewed Joint graph.",
    )
    begin_stage(state_path, "articulation", actor="test-agent")
    second_attempt = stage_directory(state_path, "articulation")
    second_candidates = second_attempt / "unexpected-candidates.json"
    second_candidates.write_text('{"candidate_ids":["door"]}\n', encoding="utf-8")

    with pytest.raises(AssetCompositionStateError, match="frozen Joint review"):
        _write_review(
            state_path,
            "articulation",
            decision="await_review",
            evidence=[second_candidates],
        )
    refinement_evidence = second_attempt / "refinement-evidence.json"
    refinement_evidence.write_text(
        '{"status":"needs_refinement"}\n',
        encoding="utf-8",
    )
    with pytest.raises(AssetCompositionStateError, match="frozen Joint review"):
        _write_review(
            state_path,
            "articulation",
            decision="refine",
            evidence=[refinement_evidence],
            repair_scope=["retry authoring without changing reviewed candidates"],
        )

    current = load_verified_run(state_path)
    assert current.stages["articulation"].review_decisions == receipt
    assert current.coordinator.next_action == "execute_stage"


def test_coordinator_rejects_duplicate_review_evidence_paths(tmp_path: Path) -> None:
    state_path = _create(tmp_path)
    stage_dir = _begin(state_path, "articulation")
    evidence = stage_dir / "classification-evidence.json"
    evidence.write_text('{"status":"conditional"}\n', encoding="utf-8")

    with pytest.raises(AssetCompositionStateError, match="duplicate evidence path"):
        _write_review(
            state_path,
            "articulation",
            decision="refine",
            evidence=[evidence, evidence],
            repair_scope=["retry classification"],
        )

    assert load_verified_run(state_path).coordinator.evidence_reviews == []


def test_coordinator_review_loader_parses_only_digest_bound_bytes(
    tmp_path: Path,
) -> None:
    state_path = _create(tmp_path)
    directory = _begin(state_path, "articulation")
    candidates = directory / "articulation_candidates.json"
    candidates.write_text('{"candidates":[]}\n', encoding="utf-8")
    _write_review(
        state_path,
        "articulation",
        decision="await_review",
        evidence=[candidates],
    )
    review_binding = load_verified_run(state_path).coordinator.evidence_reviews[-1]
    review_path = Path(review_binding.path)
    payload = json.loads(review_path.read_text(encoding="utf-8"))
    payload["decision_summary"] = "Temporarily replaced accepting review."
    review_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(AssetCompositionStateError, match="identity changed"):
        asset_state._load_coordinator_review(review_binding)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {"decisions": [{"candidate_id": "drawer", "decision": "accept"}]},
            "'accept' or 'reject'",
        ),
        (
            {"drawer": "accept", "door": "reject"},
            "unexpected=\\['door'\\]",
        ),
        ({"drawer": "approve"}, "'accept' or 'reject'"),
    ],
)
def test_joint_review_rejects_invalid_decisions_before_sealing(
    tmp_path: Path,
    payload: dict[str, object],
    message: str,
) -> None:
    state_path = _create(tmp_path)
    first_attempt = _begin(state_path, "articulation")
    candidates = first_attempt / "articulation-candidates.json"
    candidates.write_text('{"candidate_ids":["drawer"]}\n', encoding="utf-8")
    _write_review(
        state_path,
        "articulation",
        decision="await_review",
        evidence=[candidates],
    )
    require_review(state_path, candidates_path=candidates, actor="test-agent")
    decisions = state_path.parent / "raw" / "invalid-joint-decisions.json"
    decisions.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(AssetCompositionStateError, match=message):
        record_review_decisions(
            state_path,
            decisions_path=decisions,
            reviewer="asset-owner",
        )

    current = load_verified_run(state_path)
    assert current.stages["articulation"].status == "needs_review"
    assert current.stages["articulation"].review_decisions is None


def test_orphaned_plan_record_is_reconciled_after_interrupted_state_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_path = _create(tmp_path)
    real_write_run = asset_state._write_run
    failed = False

    def fail_once(path: Path, run: AssetCompositionRun) -> AssetCompositionRun:
        nonlocal failed
        if not failed:
            failed = True
            raise RuntimeError("simulated interrupted state commit")
        return real_write_run(path, run)

    monkeypatch.setattr(asset_state, "_write_run", fail_once)
    with pytest.raises(RuntimeError, match="interrupted state commit"):
        _write_plan(state_path, "articulation")
    orphan = state_path.parent / "coordinator" / "plans" / "plan-001.json"
    assert orphan.is_file()

    monkeypatch.setattr(asset_state, "_write_run", real_write_run)
    _write_plan(state_path, "articulation")

    run = load_verified_run(state_path)
    assert len(run.coordinator.plan_revisions) == 1
    assert run.coordinator.plan_revisions[0].path == str(orphan)


def test_orphaned_review_record_is_reconciled_after_interrupted_state_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_path = _create(tmp_path)
    stage_dir = _begin(state_path, "articulation")
    evidence = stage_dir / "classification-evidence.json"
    evidence.write_text('{"status":"pass"}\n', encoding="utf-8")
    real_write_run = asset_state._write_run
    failed = False

    def fail_once(path: Path, run: AssetCompositionRun) -> AssetCompositionRun:
        nonlocal failed
        if not failed:
            failed = True
            raise RuntimeError("simulated interrupted state commit")
        return real_write_run(path, run)

    monkeypatch.setattr(asset_state, "_write_run", fail_once)
    with pytest.raises(RuntimeError, match="interrupted state commit"):
        _write_review(
            state_path,
            "articulation",
            decision="refine",
            evidence=[evidence],
            repair_scope=["retry exact reviewed attempt"],
        )
    orphan = state_path.parent / "coordinator" / "reviews" / "review-001.json"
    assert orphan.is_file()

    monkeypatch.setattr(asset_state, "_write_run", real_write_run)
    _write_review(
        state_path,
        "articulation",
        decision="refine",
        evidence=[evidence],
        repair_scope=["retry exact reviewed attempt"],
    )

    run = load_verified_run(state_path)
    assert len(run.coordinator.evidence_reviews) == 1
    assert run.coordinator.evidence_reviews[0].path == str(orphan)
    assert run.stages["articulation"].status == "ready"


def test_coordinator_enforces_refinement_budget(tmp_path: Path) -> None:
    state_path = _create(tmp_path)
    stage_dir = _begin(state_path, "articulation")

    for index in range(3):
        evidence = stage_dir / "classification-evidence.json"
        evidence.write_text(
            json.dumps({"status": "conditional", "iteration": index + 1}) + "\n",
            encoding="utf-8",
        )
        _write_review(
            state_path,
            "articulation",
            decision="refine",
            evidence=[evidence],
            repair_scope=[f"candidate classification revision {index + 1}"],
        )
        _write_plan(
            state_path,
            "articulation",
            evidence=evidence,
            reason=f"Apply bounded refinement {index + 1}.",
        )
        assert stage_directory(state_path, "articulation") != stage_dir
        begin_stage(state_path, "articulation", actor="test-agent")
        stage_dir = stage_directory(state_path, "articulation")

    fourth_evidence = stage_dir / "classification-evidence.json"
    fourth_evidence.write_text(
        '{"status":"conditional","iteration":4}\n', encoding="utf-8"
    )

    with pytest.raises(AssetCompositionStateError, match="refinement budget"):
        _write_review(
            state_path,
            "articulation",
            decision="refine",
            evidence=[fourth_evidence],
            repair_scope=["unbounded fourth revision"],
        )

    assert (
        load_verified_run(state_path).coordinator.refinement_counts["articulation"] == 3
    )
    assert stage_dir.name == "04"


def test_default_plan_budget_covers_all_declared_refinements_and_revisits() -> None:
    coordinator = AssetCoordinatorState(mode="single_reasoning_loop")
    stage_count = len(LEGACY_STAGE_ORDER)
    required_plans = (
        stage_count * (1 + coordinator.max_refinements_per_stage)
        + 1  # second articulation plan after the frozen Joint review receipt
        + coordinator.max_revisits * (stage_count - 1)
    )

    assert coordinator.max_plan_revisions >= required_plans


def test_coordinator_state_caps_evidence_reviews_explicitly() -> None:
    binding = ArtifactBinding(path="/tmp/review.json", sha256="0" * 64, size_bytes=0)

    with pytest.raises(ValueError, match="evidence review budget exceeded"):
        AssetCoordinatorState(
            mode="single_reasoning_loop",
            max_evidence_reviews=1,
            evidence_reviews=[binding, binding],
        )


def test_evidence_review_budget_fails_before_writing_orphan(tmp_path: Path) -> None:
    state_path = _create(tmp_path)
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    payload["coordinator"]["max_evidence_reviews"] = 1
    state_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    first_dir = _begin(state_path, "articulation")
    first_evidence = first_dir / "first.json"
    first_evidence.write_text('{"status":"conditional"}\n', encoding="utf-8")
    _write_review(
        state_path,
        "articulation",
        decision="refine",
        evidence=[first_evidence],
        repair_scope=["retry classification"],
    )
    _write_plan(state_path, "articulation", evidence=first_evidence)
    begin_stage(state_path, "articulation")
    second_dir = stage_directory(state_path, "articulation")
    second_evidence = second_dir / "second.json"
    second_evidence.write_text('{"status":"conditional"}\n', encoding="utf-8")

    with pytest.raises(AssetCompositionStateError, match="review budget exceeded"):
        _write_review(
            state_path,
            "articulation",
            decision="refine",
            evidence=[second_evidence],
            repair_scope=["another retry"],
        )

    assert not (state_path.parent / "coordinator/reviews/review-002.json").exists()


def test_validation_accept_rejects_modified_output(tmp_path: Path) -> None:
    state_path = _create(tmp_path)
    _complete_articulation(state_path)
    _complete_simple_stage(state_path, "material")
    _complete_simple_stage(state_path, "texture")
    _complete_physics_stage(state_path)
    directory = _begin(state_path, "validation")
    output = directory / "validation.usda"
    output.write_text('#usda 1.0\ndef Xform "Modified" {}\n', encoding="utf-8")
    evidence = _write_validation_evidence(state_path, directory)

    with pytest.raises(
        AssetCompositionStateError,
        match="stage output bytes differ from the input",
    ):
        _write_review(
            state_path,
            "validation",
            decision="accept",
            evidence=evidence,
            output=output,
        )


def test_validation_accept_rejects_changed_dependency_closure(tmp_path: Path) -> None:
    state_path = _create(tmp_path, "validation-dependency-drift")
    _complete_articulation(state_path)
    _complete_simple_stage(state_path, "material")
    _complete_simple_stage(state_path, "texture")
    _complete_physics_stage(
        state_path,
        dependency_contents='#usda 1.0\ndef Xform "ReviewedDependency" {}\n',
    )
    directory = _begin(state_path, "validation")
    input_asset = load_verified_run(state_path).stages["validation"].input_asset
    assert input_asset is not None
    output = directory / "validation.usda"
    output.write_bytes(Path(input_asset.path).read_bytes())
    (directory / "dependency.usda").write_text(
        '#usda 1.0\ndef Xform "SubstitutedDependency" {}\n',
        encoding="utf-8",
    )
    evidence = _write_validation_evidence(state_path, directory)

    with pytest.raises(
        AssetCompositionStateError,
        match="stage output dependency closure differs from the input",
    ):
        _write_review(
            state_path,
            "validation",
            decision="accept",
            evidence=evidence,
            output=output,
        )


def test_validation_accept_rejects_failing_native_result(tmp_path: Path) -> None:
    state_path = _create(tmp_path)
    _complete_articulation(state_path)
    _complete_simple_stage(state_path, "material")
    _complete_simple_stage(state_path, "texture")
    _complete_physics_stage(state_path)
    directory = _begin(state_path, "validation")
    input_asset = load_verified_run(state_path).stages["validation"].input_asset
    assert input_asset is not None
    output = directory / "validation.usda"
    output.write_bytes(Path(input_asset.path).read_bytes())
    evidence = _write_validation_evidence(state_path, directory)
    result_path = next(
        path for path in evidence if path.name == "validation_result.json"
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["verdict"] = "fail"
    result_path.write_text(json.dumps(result) + "\n", encoding="utf-8")

    with pytest.raises(
        AssetCompositionStateError,
        match="stale validation_result",
    ):
        _write_review(
            state_path,
            "validation",
            decision="accept",
            evidence=evidence,
            output=output,
        )


def test_validation_accept_rejects_result_status_that_contradicts_checkpoint(
    tmp_path: Path,
) -> None:
    state_path = _create(tmp_path)
    _complete_articulation(state_path)
    _complete_simple_stage(state_path, "material")
    _complete_simple_stage(state_path, "texture")
    _complete_physics_stage(state_path)
    directory = _begin(state_path, "validation")
    input_asset = load_verified_run(state_path).stages["validation"].input_asset
    assert input_asset is not None
    output = directory / "validation.usda"
    output.write_bytes(Path(input_asset.path).read_bytes())
    evidence = _write_validation_evidence(state_path, directory)
    result_path = next(
        path for path in evidence if path.name == "validation_result.json"
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["template_results"][0]["status"] = "failed"
    result_path.write_text(json.dumps(result) + "\n", encoding="utf-8")

    with pytest.raises(
        AssetCompositionStateError,
        match="stale validation_result",
    ):
        _write_review(
            state_path,
            "validation",
            decision="accept",
            evidence=evidence,
            output=output,
        )


def test_embedded_validation_cli_requires_outer_assessment_for_asset_acceptance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_path = _create(tmp_path, "embedded-validation-cli")
    _complete_articulation(state_path)
    _complete_simple_stage(state_path, "material")
    _complete_simple_stage(state_path, "texture")
    _complete_physics_stage(state_path)
    directory = _begin(state_path, "validation")
    stage = load_verified_run(state_path).stages["validation"]
    assert stage.input_asset is not None
    domain = directory / "domain-run"
    reference = directory / "reference.png"
    Image.new("RGB", (8, 8), (40, 80, 120)).save(reference, format="PNG")
    monkeypatch.setattr(
        validation_workflow_module,
        "ScaffoldValidationStepExecutor",
        _EmbeddedValidationCliExecutor,
    )
    classic_prepare_calls: list[bool] = []
    real_classic_prepare = (
        workflow_cli_module._prepare_fixed_pipeline_embedded_validation_evidence
    )

    def record_classic_prepare(
        run: ValidationWorkflowRun,
        *,
        run_state_path: Path,
        defer_until_cross_stage: bool,
    ) -> bool:
        classic_prepare_calls.append(defer_until_cross_stage)
        return real_classic_prepare(
            run,
            run_state_path=run_state_path,
            defer_until_cross_stage=defer_until_cross_stage,
        )

    monkeypatch.setattr(
        workflow_cli_module,
        "_prepare_fixed_pipeline_embedded_validation_evidence",
        record_classic_prepare,
    )

    # The native aggregate is deliberately failed by look_right. The CLI still
    # persists its exact digest-bound evidence and critique proposal, but it has
    # no authority to complete the composed stage.
    assert (
        workflow_cli_main(
            [
                "validate",
                "run",
                "--usd",
                stage.input_asset.path,
                "--task",
                "Validate the exact accepted composed asset without mutation.",
                "--output-dir",
                str(domain),
                "--reference-image",
                str(reference),
                "--template",
                "render_valid",
                "--template",
                "look_right",
                "--embedded-run-state",
                str(state_path),
            ]
        )
        == 1
    )
    assert classic_prepare_calls == [True]
    raw_result = json.loads(
        (domain / "validation_result.json").read_text(encoding="utf-8")
    )
    assert raw_result["verdict"] == "fail"
    assert raw_result["metadata"] == {
        **raw_result["metadata"],
        "embedded_execution": True,
        "native_verdict_role": "factual_evidence_only",
        "look_right_role": "proposal_or_critique_only",
        "semantic_completion_authority": (
            "completed_embedded_validation_decision_receipt"
        ),
    }
    cross_stage = _write_validation_cross_stage_evidence(
        state_path,
        directory,
        domain / "validation_result.json",
    )
    output = directory / "validation.usda"
    output.write_bytes(Path(stage.input_asset.path).read_bytes())
    native_run = load_embedded_validation_run(domain)
    operation_index_target = directory / "operation-index-target.json"
    operation_index_target.write_text("{}\n", encoding="utf-8")
    operation_index_link = domain / VALIDATION_OPERATION_INDEX_NAME
    operation_index_link.symlink_to(operation_index_target)
    with pytest.raises(
        EmbeddedValidationAssessmentError,
        match="operation index must not be a symlink",
    ):
        prepare_embedded_validation_evidence(
            native_run,
            run_state_path=state_path,
        )
    operation_index_link.unlink()

    binding_target = directory / "binding-target.json"
    binding_target.write_text("{}\n", encoding="utf-8")
    binding_link = directory / "binding-link.json"
    binding_link.symlink_to(binding_target)
    with pytest.raises(
        EmbeddedValidationAssessmentError,
        match="must not be a symlink",
    ):
        validation_assessment._binding(binding_link)
    unavailable_records = validation_assessment._template_evidence_records(
        native_run,
        ValidationTemplateResult(
            template_name="render_valid",
            status="error",
            issues=(
                ValidationIssue(
                    code="validation.render_error",
                    severity="fail",
                    message="Deterministic render validation failed.",
                    template_name="render_valid",
                ),
            ),
        ),
    )
    assert [record.status for record in unavailable_records] == [
        "error",
        "available",
    ]
    assert unavailable_records[0].facts == {}
    assert unavailable_records[1].required is False
    assert unavailable_records[1].facts["tool_identity"] == {
        "name": "render_valid",
        "version": "test.render-valid.v1",
    }
    assert unavailable_records[1].facts["profile_identity"] == {
        "request_digest": native_run.checkpoint.workflow_identity.request_digest,
        "policy_digest": native_run.checkpoint.workflow_identity.policy_digest,
        "backend_digest": native_run.checkpoint.workflow_identity.backend_digest,
    }
    assert unavailable_records[1].facts["issue_codes"] == ("validation.render_error",)
    raw_unavailable_report = unavailable_records[1].facts["raw_template_result"]
    assert isinstance(raw_unavailable_report, Mapping)
    assert raw_unavailable_report["status"] == "error"
    unavailable_judge_records = validation_assessment._template_evidence_records(
        native_run,
        ValidationTemplateResult(
            template_name="look_right",
            status="skipped",
            issues=(
                ValidationIssue(
                    code="visual.judge_unavailable",
                    severity="warn",
                    message="The configured visual judge is unavailable.",
                    template_name="look_right",
                ),
            ),
        ),
    )
    assert len(unavailable_judge_records) == 2
    assert unavailable_judge_records[0].status == "unsupported"
    assert unavailable_judge_records[0].facts == {}
    assert unavailable_judge_records[1].status == "available"
    assert unavailable_judge_records[1].required is False
    assert unavailable_judge_records[1].facts["issue_codes"] == (
        "visual.judge_unavailable",
    )
    assert unavailable_judge_records[1].facts["raw_template_result"]["status"] == (
        "skipped"
    )
    judge_unavailable_assessment = ValidationCoordinatorAssessment(
        assessment_id="judge-unavailable-assessment",
        created_at=native_run.checkpoint.updated_at,
        gates=(
            ValidationGateAssessment(
                gate="visual_quality",
                required=False,
                evidence_ids=(
                    "validation-template-look_right",
                    "validation-template-look_right-report",
                ),
                disposition="defer",
                rationale="The advisory judge dependency was explicitly unavailable.",
            ),
        ),
        findings=(
            ValidationAssessmentFinding(
                finding_id="judge-unavailable",
                source_evidence_ids=("validation-template-look_right-report",),
                source_issue_codes=("visual.judge_unavailable",),
                severity="warning",
                summary="The configured visual judge was unavailable.",
                disposition="deferred",
                rationale="The unavailable advisory dependency remains explicit.",
            ),
        ),
        terminal_disposition="blocked",
        summary="The advisory visual judge did not produce a critique.",
    )
    assert (
        validate_coordinator_assessment(
            judge_unavailable_assessment,
            evidence=unavailable_judge_records,
        )
        == judge_unavailable_assessment
    )
    assessment = ValidationCoordinatorAssessment(
        assessment_id="embedded-cli-outer-assessment",
        created_at=native_run.checkpoint.updated_at,
        gates=(
            ValidationGateAssessment(
                gate="static_validation",
                evidence_ids=("validation-template-render_valid",),
                disposition="pass",
                rationale="The deterministic render gate is an explicit pass.",
            ),
            ValidationGateAssessment(
                gate="visual_quality",
                evidence_ids=(
                    "validation-outer-visual-evidence",
                    "validation-template-look_right",
                ),
                disposition="waive",
                rationale=(
                    "The outer coordinator reviewed the judge critique and records "
                    "its bounded waiver independently."
                ),
            ),
            ValidationGateAssessment(
                gate="package_integrity",
                evidence_ids=("validation-package-integrity",),
                disposition="pass",
                rationale="The exact native bundle passed saved-artifact readback.",
            ),
            ValidationGateAssessment(
                gate="cross_stage_integrity",
                evidence_ids=("validation-cross-stage-integrity",),
                disposition="pass",
                rationale="Accepted upstream handoffs and claims are intact.",
            ),
        ),
        findings=(
            ValidationAssessmentFinding(
                finding_id="visual-mismatch-proposal",
                source_evidence_ids=("validation-template-look_right",),
                source_issue_codes=("validation.visual_mismatch",),
                severity="warning",
                summary="The optional judge proposed a visual mismatch.",
                affected_artifacts=(stage.input_asset.path,),
                disposition="waived",
                rationale=(
                    "The proposal is not factual completion authority; the outer "
                    "review accepts the exact artifact under this explicit waiver."
                ),
            ),
        ),
        terminal_disposition="pass",
        summary="The outer coordinator accepts every required factual gate.",
    )
    assessment_path = directory / "outer-assessment.json"
    assessment_path.write_text(
        assessment.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    # Classic run -> assess completes evidence preparation without a separately
    # invoked collect-evidence command. The focused leaf has its own CLI test.
    assert (
        workflow_cli_main(
            [
                "validate",
                "assess",
                "--output-dir",
                str(domain),
                "--embedded-run-state",
                str(state_path),
                "--assessment",
                str(assessment_path),
            ]
        )
        == 0
    )
    assert classic_prepare_calls == [True, False]
    evidence_index = json.loads(
        (domain / EMBEDDED_VALIDATION_EVIDENCE_INDEX_NAME).read_text(encoding="utf-8")
    )
    assert len(evidence_index["evidence"]) == 5
    assert len({item["artifact_id"] for item in evidence_index["evidence"]}) == 5
    assert len(evidence_index["proposals"]) == 1
    native_evidence = [
        domain / "final_summary.json",
        domain / "validation_result.json",
        domain / "validation_evidence.json",
        domain / "validation_checkpoint.json",
        cross_stage,
        domain / EMBEDDED_VALIDATION_EVIDENCE_INDEX_NAME,
    ]
    with pytest.raises(
        AssetCompositionStateError,
        match="requires exactly one embedded evidence index, execution index",
    ):
        _write_review(
            state_path,
            "validation",
            decision="accept",
            evidence=native_evidence,
            output=output,
        )
    review = ValidationCoordinatorReviewDraft(
        created_at=native_run.checkpoint.updated_at,
        disposition="accept",
        findings=("Accepted the exact canonical outer assessment.",),
    )
    review_path = directory / "outer-assessment-review.json"
    review_path.write_text(
        review.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    assert (
        workflow_cli_main(
            [
                "validate",
                "review-assessment",
                "--output-dir",
                str(domain),
                "--embedded-run-state",
                str(state_path),
                "--review",
                str(review_path),
            ]
        )
        == 0
    )
    validate_completed_embedded_validation_receipt(domain)
    terminal_path = domain / VALIDATION_TERMINAL_RECEIPT_NAME
    receipt_index_path = domain / EMBEDDED_VALIDATION_RECEIPT_INDEX_NAME
    original_terminal = terminal_path.read_bytes()
    original_receipt_index = receipt_index_path.read_bytes()
    terminal_payload = json.loads(original_terminal)
    terminal_payload["review_disposition"] = "reject"
    terminal_path.write_text(
        json.dumps(terminal_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    receipt_index_payload = json.loads(original_receipt_index)
    receipt_index_payload["terminal_receipt"]["sha256"] = file_sha256(terminal_path)
    receipt_index_payload["terminal_receipt"]["size_bytes"] = (
        terminal_path.stat().st_size
    )
    receipt_index_path.write_text(
        json.dumps(receipt_index_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        EmbeddedValidationAssessmentError,
        match="terminal receipt is incomplete or stale",
    ):
        validate_completed_embedded_validation_receipt(domain)
    terminal_path.write_bytes(original_terminal)
    receipt_index_path.write_bytes(original_receipt_index)
    validate_completed_embedded_validation_receipt(domain)
    completed_evidence = [
        *native_evidence,
        domain / EMBEDDED_VALIDATION_EXECUTION_INDEX_NAME,
        domain / EMBEDDED_VALIDATION_RECEIPT_INDEX_NAME,
        domain / CANONICAL_VALIDATION_ASSESSMENT_NAME,
    ]
    _write_review(
        state_path,
        "validation",
        decision="accept",
        evidence=completed_evidence,
        output=output,
    )
    complete_stage(
        state_path,
        "validation",
        output_asset=output,
        evidence_paths=completed_evidence,
        summary="Accepted only after the outer assessment receipt completed.",
        actor="test-agent",
    )
    assert load_verified_run(state_path).stages["validation"].status == "completed"


@pytest.mark.parametrize(
    ("records", "evidence_ids", "gate_required", "message"),
    (
        (
            (
                ProviderNeutralEvidenceRecord(
                    evidence_id="static-pass",
                    evidence_type="validation",
                    status="available",
                    summary="Deterministic static pass.",
                    facts={"gate": "static_validation", "verdict": "pass"},
                ),
                ProviderNeutralEvidenceRecord(
                    evidence_id="static-fail",
                    evidence_type="validation",
                    status="available",
                    summary="Deterministic static failure.",
                    facts={"gate": "static_validation", "verdict": "fail"},
                ),
            ),
            ("static-pass", "static-fail"),
            True,
            "required gate static_validation cannot pass",
        ),
        (
            (
                ProviderNeutralEvidenceRecord(
                    evidence_id="static-error",
                    evidence_type="validation",
                    status="error",
                    summary="Static tool execution failed.",
                ),
            ),
            ("static-error",),
            True,
            "required gate static_validation cannot pass",
        ),
        (
            (
                ProviderNeutralEvidenceRecord(
                    evidence_id="static-pass",
                    evidence_type="validation",
                    status="available",
                    summary="Deterministic static pass.",
                    facts={"gate": "static_validation", "verdict": "pass"},
                ),
                ProviderNeutralEvidenceRecord(
                    evidence_id="runtime-pass",
                    evidence_type="validation",
                    status="available",
                    summary="Deterministic runtime pass.",
                    facts={"gate": "runtime_validation", "verdict": "pass"},
                ),
            ),
            ("static-pass",),
            True,
            "assessment evidence coverage is incomplete",
        ),
        (
            (
                ProviderNeutralEvidenceRecord(
                    evidence_id="static-pass",
                    evidence_type="validation",
                    status="available",
                    summary="Required deterministic static pass.",
                    facts={"gate": "static_validation", "verdict": "pass"},
                ),
            ),
            ("static-pass",),
            False,
            "required evidence cannot be downgraded",
        ),
    ),
)
def test_embedded_validation_assessment_fails_closed_on_required_evidence(
    records: tuple[ProviderNeutralEvidenceRecord, ...],
    evidence_ids: tuple[str, ...],
    gate_required: bool,
    message: str,
) -> None:
    # This validator consumes only provider-neutral records; model_construct
    # intentionally omits unrelated persisted identity fields in this unit test.
    evidence = tuple(
        EmbeddedDomainEvidence.model_construct(records=(record,)) for record in records
    )
    assessment = ValidationCoordinatorAssessment(
        assessment_id="invalid-required-evidence",
        created_at="2026-08-10T00:00:00Z",
        gates=(
            ValidationGateAssessment(
                gate="static_validation",
                required=gate_required,
                evidence_ids=evidence_ids,
                disposition="pass",
                rationale="Attempted pass for fail-closed coverage.",
            ),
        ),
        terminal_disposition="pass",
        summary="This assessment must be rejected.",
    )

    with pytest.raises(EmbeddedValidationAssessmentError, match=message):
        validate_coordinator_assessment(assessment, evidence=evidence)


def test_embedded_validation_required_gate_must_cite_required_evidence() -> None:
    advisory_failure = ProviderNeutralEvidenceRecord(
        evidence_id="advisory-static-fail",
        evidence_type="validation",
        status="available",
        required=False,
        summary="Advisory evidence reports a static failure.",
        facts={"gate": "static_validation", "verdict": "fail"},
    )
    evidence = (EmbeddedDomainEvidence.model_construct(records=(advisory_failure,)),)
    assessment = ValidationCoordinatorAssessment(
        assessment_id="required-gate-with-advisory-only-evidence",
        created_at="2026-08-23T00:00:00Z",
        gates=(
            ValidationGateAssessment(
                gate="static_validation",
                required=True,
                evidence_ids=(advisory_failure.evidence_id,),
                disposition="waive",
                rationale="An advisory result alone cannot satisfy a required gate.",
            ),
        ),
        terminal_disposition="pass",
        summary="This assessment must be rejected.",
    )

    with pytest.raises(
        EmbeddedValidationAssessmentError,
        match="required gate static_validation must cite required evidence",
    ):
        validate_coordinator_assessment(assessment, evidence=evidence)


def test_validation_refinement_preserves_prior_assessment_evidence(
    tmp_path: Path,
) -> None:
    state_path = _create(tmp_path, "validation-remediation-lineage")
    _complete_articulation(state_path)
    _complete_simple_stage(state_path, "material")
    _complete_simple_stage(state_path, "texture")
    _complete_physics_stage(state_path)
    first_directory = _begin(state_path, "validation")
    evidence = _write_validation_evidence(state_path, first_directory)
    evidence_hashes = {str(path): file_sha256(path) for path in evidence}
    review_evidence = [
        path for path in evidence if path.name != "validation_checkpoint.json"
    ]

    _write_review(
        state_path,
        "validation",
        decision="refine",
        evidence=review_evidence,
        repair_scope=[
            "Remediate the targeted finding and revalidate in a new attempt."
        ],
    )
    refined = load_verified_run(state_path)
    archived = refined.stages["validation"].superseded_attempts[-1]
    assert [item.path for item in archived.evidence] == [
        str(path) for path in review_evidence
    ]
    assert all(file_sha256(path) == digest for path, digest in evidence_hashes.items())

    _write_plan(
        state_path,
        "validation",
        evidence=evidence[0],
        reason="Targeted remediation with append-only prior evidence.",
    )
    begin_stage(state_path, "validation", actor="test-agent")
    second_directory = stage_directory(state_path, "validation")
    assert second_directory != first_directory
    assert second_directory.is_dir()
    assert load_verified_run(state_path).stages["validation"].attempt_count == 2


def test_validation_accept_requires_cross_stage_receipt(tmp_path: Path) -> None:
    state_path = _create(tmp_path)
    _complete_articulation(state_path)
    _complete_simple_stage(state_path, "material")
    _complete_simple_stage(state_path, "texture")
    _complete_physics_stage(state_path)
    directory = _begin(state_path, "validation")
    input_asset = load_verified_run(state_path).stages["validation"].input_asset
    assert input_asset is not None
    output = directory / "validation.usda"
    output.write_bytes(Path(input_asset.path).read_bytes())
    evidence = _write_validation_evidence(state_path, directory)
    native_only = [
        path for path in evidence if path.name != "cross_stage_validation.json"
    ]

    with pytest.raises(
        AssetCompositionStateError,
        match="requires one cross_stage_validation.json",
    ):
        _write_review(
            state_path,
            "validation",
            decision="accept",
            evidence=native_only,
            output=output,
        )


def test_validation_assessment_identity_rejects_stale_cross_stage_bytes(
    tmp_path: Path,
) -> None:
    state_path = _create(tmp_path, "stale-cross-stage-assessment-identity")
    _complete_articulation(state_path)
    _complete_simple_stage(state_path, "material")
    _complete_simple_stage(state_path, "texture")
    _complete_physics_stage(state_path)
    directory = _begin(state_path, "validation")
    _write_validation_evidence(state_path, directory)
    cross_stage_path = directory / "cross_stage_validation.json"
    cross_stage_path.write_text(
        cross_stage_path.read_text(encoding="utf-8") + " ",
        encoding="utf-8",
    )
    with pytest.raises(EmbeddedValidationAssessmentError, match="stale"):
        assess_embedded_validation(
            directory / "domain-run",
            run_state_path=state_path,
            assessment_path=directory / "validation_assessment_draft.json",
        )


def test_validation_assessment_requires_render_valid_gate(tmp_path: Path) -> None:
    state_path = _create(tmp_path)
    _complete_articulation(state_path)
    _complete_simple_stage(state_path, "material")
    _complete_simple_stage(state_path, "texture")
    _complete_physics_stage(state_path)
    directory = _begin(state_path, "validation")

    with pytest.raises(
        EmbeddedValidationAssessmentError,
        match="without available required current-render evidence",
    ):
        _write_validation_evidence(
            state_path,
            directory,
            template_name="look_right",
        )


def test_validation_accept_requires_render_valid_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def current_render_evidence(
        run: ValidationWorkflowRun,
        *,
        required: bool | None = None,
    ) -> ProviderNeutralEvidenceRecord:
        del required
        render_path = Path(run.output_dir) / "outer-render.png"
        Image.new("RGB", (8, 8), (20, 80, 140)).save(render_path, format="PNG")
        binding = ExecutionArtifactBinding(
            path=str(render_path.resolve()),
            sha256=file_sha256(render_path),
            size_bytes=render_path.stat().st_size,
        )
        return ProviderNeutralEvidenceRecord(
            evidence_id="validation-outer-visual-evidence",
            evidence_type="render",
            status="available",
            required=True,
            summary="Digest-bound current render for the outer coordinator.",
            artifacts=(binding,),
            facts={
                "gate": "visual_quality",
                "semantic_authority": "outer_coordinator",
                "render_images": [binding.model_dump(mode="json")],
                "reference_images": [],
            },
        )

    monkeypatch.setattr(
        validation_assessment,
        "_visual_evidence_record",
        current_render_evidence,
    )
    state_path = _create(tmp_path)
    _complete_articulation(state_path)
    _complete_simple_stage(state_path, "material")
    _complete_simple_stage(state_path, "texture")
    _complete_physics_stage(state_path)
    directory = _begin(state_path, "validation")
    input_asset = load_verified_run(state_path).stages["validation"].input_asset
    assert input_asset is not None
    output = directory / "validation.usda"
    output.write_bytes(Path(input_asset.path).read_bytes())
    evidence = _write_validation_evidence(
        state_path,
        directory,
        template_name="look_right",
        include_outer_visual_gate=True,
    )

    with pytest.raises(
        AssetCompositionStateError,
        match="omits the required render_valid gate",
    ):
        _write_review(
            state_path,
            "validation",
            decision="accept",
            evidence=evidence,
            output=output,
        )


def test_validation_accept_requires_preserved_joint_graph(tmp_path: Path) -> None:
    state_path = _create(tmp_path)
    _complete_articulation(state_path)
    _complete_simple_stage(state_path, "material")
    _complete_simple_stage(state_path, "texture")
    _complete_physics_stage(state_path, omit_joint=True)
    directory = _begin(state_path, "validation")
    input_asset = load_verified_run(state_path).stages["validation"].input_asset
    assert input_asset is not None
    output = directory / "validation.usda"
    output.write_bytes(Path(input_asset.path).read_bytes())
    evidence = _write_validation_evidence(state_path, directory)

    with pytest.raises(
        AssetCompositionStateError,
        match="Joint graph differs from the accepted Articulation graph",
    ):
        _write_review(
            state_path,
            "validation",
            decision="accept",
            evidence=evidence,
            output=output,
        )


def test_validation_accept_rejects_changed_joint_motion_definition(
    tmp_path: Path,
) -> None:
    state_path = _create(tmp_path)
    _complete_articulation(state_path)
    _complete_simple_stage(state_path, "material")
    _complete_simple_stage(state_path, "texture")
    _complete_physics_stage(state_path, joint_axis="Y")
    directory = _begin(state_path, "validation")
    input_asset = load_verified_run(state_path).stages["validation"].input_asset
    assert input_asset is not None
    output = directory / "validation.usda"
    output.write_bytes(Path(input_asset.path).read_bytes())
    evidence = _write_validation_evidence(state_path, directory)

    with pytest.raises(
        AssetCompositionStateError,
        match="Joint graph differs from the accepted Articulation graph",
    ):
        _write_review(
            state_path,
            "validation",
            decision="accept",
            evidence=evidence,
            output=output,
        )


def test_validation_accepts_preserved_authored_joints_when_physics_adds_bodies(
    tmp_path: Path,
) -> None:
    state_path = _create(tmp_path)
    _complete_articulation(state_path, with_rigid_bodies=False)
    _complete_simple_stage(state_path, "material")
    _complete_simple_stage(state_path, "texture")
    _complete_physics_stage(state_path)

    output = _complete_simple_stage(state_path, "validation")

    assert output.is_file()
    assert load_verified_run(state_path).stages["validation"].status == "completed"


def test_articulated_physics_accepts_reset_nested_body(tmp_path: Path) -> None:
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    output = tmp_path / "reset-nested.usda"
    stage = Usd.Stage.CreateNew(str(output))
    cabinet = UsdGeom.Xform.Define(stage, "/Cabinet").GetPrim()
    stage.SetDefaultPrim(cabinet)
    drawer = UsdGeom.Xform.Define(stage, "/Cabinet/Drawer").GetPrim()
    UsdGeom.Xformable(drawer).SetResetXformStack(True)
    UsdPhysics.RigidBodyAPI.Apply(cabinet).CreateRigidBodyEnabledAttr(True)
    UsdPhysics.RigidBodyAPI.Apply(drawer).CreateRigidBodyEnabledAttr(True)
    cabinet_collider = UsdGeom.Cube.Define(stage, "/Cabinet/Collision").GetPrim()
    drawer_collider = UsdGeom.Cube.Define(stage, "/Cabinet/Drawer/Collision").GetPrim()
    UsdPhysics.CollisionAPI.Apply(cabinet_collider).CreateCollisionEnabledAttr(True)
    UsdPhysics.CollisionAPI.Apply(drawer_collider).CreateCollisionEnabledAttr(True)
    joint = UsdPhysics.PrismaticJoint.Define(stage, "/Cabinet/DrawerJoint")
    joint.CreateBody0Rel().SetTargets([Sdf.Path("/Cabinet")])
    joint.CreateBody1Rel().SetTargets([Sdf.Path("/Cabinet/Drawer")])
    assert stage.GetRootLayer().Save()

    asset_state._validate_articulated_physics_output(output)


def test_joint_graph_signature_ignores_authored_sibling_order(tmp_path: Path) -> None:
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    def write_stage(path: Path, joint_paths: tuple[str, ...]) -> None:
        stage = Usd.Stage.CreateNew(str(path))
        UsdGeom.Xform.Define(stage, "/World")
        UsdGeom.Xform.Define(stage, "/World/Cabinet")
        UsdGeom.Xform.Define(stage, "/World/DrawerA")
        UsdGeom.Xform.Define(stage, "/World/DrawerB")
        for joint_path in joint_paths:
            joint = UsdPhysics.PrismaticJoint.Define(stage, joint_path)
            joint.CreateBody0Rel().SetTargets([Sdf.Path("/World/Cabinet")])
            suffix = joint_path.removeprefix("/World/Joint")
            joint.CreateBody1Rel().SetTargets([Sdf.Path(f"/World/Drawer{suffix}")])
        stage.GetRootLayer().Save()

    first = tmp_path / "first.usda"
    second = tmp_path / "second.usda"
    write_stage(first, ("/World/JointA", "/World/JointB"))
    write_stage(second, ("/World/JointB", "/World/JointA"))

    assert asset_state._joint_graph_signature(
        first, label="first"
    ) == asset_state._joint_graph_signature(second, label="second")


def test_joint_graph_value_encoding_is_typed_exact_and_repr_free() -> None:
    from pxr import Gf, Sdf, Vt

    value = {
        "axis": Gf.Vec3f(0.1, -0.0, 2.0),
        "orientation": Gf.Quatf(0.5, Gf.Vec3f(0.1, 0.2, 0.3)),
        "targets": Vt.StringArray(["/World/Base", "/World/Link"]),
        "asset": Sdf.AssetPath("relative/model.usd"),
    }

    encoded = asset_state._canonical_joint_graph_value(value)
    serialized = json.dumps(encoded, sort_keys=True, separators=(",", ":"))

    assert encoded == asset_state._canonical_joint_graph_value(value)
    assert "Gf." not in serialized
    assert "Vt." not in serialized
    assert "Sdf." not in serialized
    assert "-0x0.0p+0" in serialized
    assert "relative/model.usd" in serialized


def test_joint_graph_value_encoding_fails_closed_for_unknown_types() -> None:
    class UnsupportedAuthoredValue:
        pass

    with pytest.raises(
        AssetCompositionStateError,
        match="unsupported authored USD value type",
    ):
        asset_state._canonical_joint_graph_value(UnsupportedAuthoredValue())


def test_composed_joint_graph_detects_proxies_without_changing_shared_topology(
    tmp_path: Path,
) -> None:
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics
    from world_understanding.functions.physics.physics_topology import (
        inspect_physics_topology,
    )

    def write_stage(root_path: Path, model_path: Path, *, axis: str) -> None:
        model_stage = Usd.Stage.CreateNew(str(model_path))
        model = UsdGeom.Xform.Define(model_stage, "/Model")
        model_stage.SetDefaultPrim(model.GetPrim())
        base = UsdGeom.Xform.Define(model_stage, "/Model/Base").GetPrim()
        link = UsdGeom.Xform.Define(model_stage, "/Model/Link").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(base).CreateRigidBodyEnabledAttr(True)
        UsdPhysics.RigidBodyAPI.Apply(link).CreateRigidBodyEnabledAttr(True)
        joint = UsdPhysics.RevoluteJoint.Define(model_stage, "/Model/HiddenJoint")
        joint.CreateBody0Rel().SetTargets([Sdf.Path("/Model/Base")])
        joint.CreateBody1Rel().SetTargets([Sdf.Path("/Model/Link")])
        joint.CreateAxisAttr(axis)
        assert model_stage.GetRootLayer().Save()

        root_stage = Usd.Stage.CreateNew(str(root_path))
        world = UsdGeom.Xform.Define(root_stage, "/World")
        root_stage.SetDefaultPrim(world.GetPrim())
        instance = root_stage.OverridePrim("/World/InstancedRig")
        assert instance.GetReferences().AddReference(str(model_path))
        assert instance.SetInstanceable(True)
        assert root_stage.GetRootLayer().Save()

    first = tmp_path / "instanced-first.usda"
    second = tmp_path / "instanced-second.usda"
    write_stage(first, tmp_path / "model-first.usda", axis="X")
    write_stage(second, tmp_path / "model-second.usda", axis="Y")

    first_signature = asset_state._joint_graph_signature(first, label="first")
    second_signature = asset_state._joint_graph_signature(second, label="second")
    topology = inspect_physics_topology(first)

    assert first_signature
    assert first_signature != second_signature
    assert [joint["prim_path"] for joint in topology["joints"]] == [
        "/World/InstancedRig/HiddenJoint"
    ]
    assert topology["rigid_body_paths"] == [
        "/World/InstancedRig/Base",
        "/World/InstancedRig/Link",
    ]
    with pytest.raises(AssetCompositionStateError, match="instance proxies"):
        asset_state._validate_articulated_physics_output(first)


def test_required_bound_path_deduplicates_same_output_and_evidence() -> None:
    binding = ArtifactBinding(path="/tmp/output.usda", sha256="1" * 64, size_bytes=1)

    assert (
        asset_state._required_bound_path(
            binding.path,
            evidence=[binding],
            additional=[binding],
            label="Texture unit artifact",
        )
        == binding
    )


def test_dependency_verification_recomputes_a_retargeted_symlink_closure(
    tmp_path: Path,
) -> None:
    from pxr import Usd, UsdGeom

    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "dependency.usda").write_text(
        '#usda 1.0\ndef Xform "First" {}\n', encoding="utf-8"
    )
    (second / "dependency.usda").write_text(
        '#usda 1.0\ndef Xform "Second" {}\n', encoding="utf-8"
    )
    link = tmp_path / "active"
    link.symlink_to(first, target_is_directory=True)
    root = tmp_path / "root.usda"
    stage = Usd.Stage.CreateNew(str(root))
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    referenced = stage.OverridePrim("/World/Referenced")
    assert referenced.GetReferences().AddReference("active/dependency.usda")
    assert stage.GetRootLayer().Save()
    dependencies = asset_state.bind_usd_dependency_closure(root)
    asset_state.verify_usd_dependency_closure(root, dependencies)

    link.unlink()
    link.symlink_to(second, target_is_directory=True)

    with pytest.raises(
        AssetCompositionStateError,
        match="dependency closure identity changed",
    ):
        asset_state.verify_usd_dependency_closure(root, dependencies)


def test_validation_accept_rejects_mismatched_native_template_bundle(
    tmp_path: Path,
) -> None:
    state_path = _create(tmp_path)
    _complete_articulation(state_path)
    _complete_simple_stage(state_path, "material")
    _complete_simple_stage(state_path, "texture")
    _complete_physics_stage(state_path)
    directory = _begin(state_path, "validation")
    input_asset = load_verified_run(state_path).stages["validation"].input_asset
    assert input_asset is not None
    output = directory / "validation.usda"
    output.write_bytes(Path(input_asset.path).read_bytes())
    evidence = _write_validation_evidence(state_path, directory)
    summary_path = next(path for path in evidence if path.name == "final_summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["completed_templates"] = ["look_right"]
    summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")

    with pytest.raises(
        AssetCompositionStateError,
        match="Validation final summary completed_templates differs",
    ):
        _write_review(
            state_path,
            "validation",
            decision="accept",
            evidence=evidence,
            output=output,
        )


def test_coordinator_rejects_stale_final_report_then_completes_with_valid_report(
    tmp_path: Path,
) -> None:
    state_path = _create(tmp_path)
    _complete_articulation(state_path)
    _complete_simple_stage(state_path, "material")
    _complete_simple_stage(state_path, "texture")
    _complete_physics_stage(state_path)
    _complete_simple_stage(state_path, "validation")
    directory = _begin(state_path, "finalization")
    package_source = directory / "package-source"
    package_source.mkdir()
    (package_source / "asset.usda").write_text("#usda 1.0\n", encoding="utf-8")
    output = directory / "asset.usdz"
    write_usdz_package_from_directory(package_source, Path("asset.usda"), output)
    validation_evidence = load_verified_run(state_path).stages["validation"].evidence
    validation_summary = next(
        binding
        for binding in validation_evidence
        if Path(binding.path).name == "final_summary.json"
    )
    noncanonical_validation = next(
        binding
        for binding in validation_evidence
        if Path(binding.path).name == "validation_result.json"
    )
    with pytest.raises(
        AssetCompositionStateError,
        match="canonical accepted final_summary.json",
    ):
        build_combined_report(
            state_path,
            final_asset=output,
            validation_summary=noncanonical_validation.path,
            output_path=directory / "noncanonical-combined-report.json",
        )
    stale_report = directory / "stale-combined-report.json"
    build_combined_report(
        state_path,
        final_asset=output,
        validation_summary=validation_summary.path,
        output_path=stale_report,
    )
    stale_payload = json.loads(stale_report.read_text(encoding="utf-8"))
    stale_payload["run_id"] = "stale-run"
    stale_report.write_text(json.dumps(stale_payload) + "\n", encoding="utf-8")

    with pytest.raises(
        AssetCompositionStateError,
        match="Combined final report does not match accepted workflow state",
    ):
        _write_review(
            state_path,
            "finalization",
            decision="accept",
            evidence=[stale_report],
            output=output,
        )

    revisable = load_verified_run(state_path)
    assert revisable.coordinator.next_action == "execute_stage"
    assert revisable.stages["finalization"].status == "running"

    valid_report = directory / "combined_report.json"
    session = AssetCoordinatorSession(
        run_state_path=state_path,
        mode="interactive",
    )
    session.build_combined_report(
        final_asset=output,
        validation_summary=validation_summary.path,
        output_path=valid_report,
    )
    duplicate_report = directory / "duplicate-combined-report.json"
    duplicate_report.write_bytes(valid_report.read_bytes())
    with pytest.raises(
        AssetCompositionStateError,
        match="exactly one combined asset report",
    ):
        _write_review(
            state_path,
            "finalization",
            decision="accept",
            evidence=[valid_report, duplicate_report],
            output=output,
        )
    assert load_verified_run(state_path).coordinator.next_action == "execute_stage"

    _write_review(
        state_path,
        "finalization",
        decision="accept",
        evidence=[valid_report],
        output=output,
    )
    complete_stage(
        state_path,
        "finalization",
        output_asset=output,
        evidence_paths=[valid_report],
        summary="Accepted exact combined report.",
        actor="test-agent",
    )

    completed = load_verified_run(state_path)
    assert completed.coordinator.next_action == "terminal"
    assert validate_terminal(state_path).valid

    request = json.loads(
        (state_path.parent / "request.json").read_text(encoding="utf-8")
    )
    Path(request["joint_config"]).write_text(
        "review_policy: changed after completion\n",
        encoding="utf-8",
    )
    terminal = validate_terminal(state_path)
    assert not terminal.valid
    assert "Joint config identity changed" in terminal.errors[0]


def test_refinement_rejects_reusing_a_sealed_output_path(tmp_path: Path) -> None:
    state_path = _create(tmp_path)
    _complete_articulation(state_path)
    stage_dir = _begin(state_path, "material")
    output = stage_dir / "material.usda"
    output.write_text('#usda 1.0\ndef Xform "Material" {}\n', encoding="utf-8")
    first_evidence = stage_dir / "first-evidence.json"
    first_evidence.write_text('{"status":"conditional"}\n', encoding="utf-8")
    _write_review(
        state_path,
        "material",
        decision="refine",
        evidence=[first_evidence],
        output=output,
        repair_scope=["repair scoped Material coverage"],
    )
    _write_plan(
        state_path,
        "material",
        evidence=first_evidence,
        reason="Retry in an isolated attempt directory.",
    )
    begin_stage(state_path, "material", actor="test-agent")
    second_dir = stage_directory(state_path, "material")
    second_evidence = second_dir / "second-evidence.json"
    second_evidence.write_text('{"status":"conditional"}\n', encoding="utf-8")

    with pytest.raises(
        AssetCompositionStateError,
        match="reviewed output must belong to the current stage attempt",
    ):
        _write_review(
            state_path,
            "material",
            decision="refine",
            evidence=[second_evidence],
            output=output,
            repair_scope=["do not overwrite the prior attempt"],
        )


def test_first_attempt_rejects_evidence_from_a_future_retry_directory(
    tmp_path: Path,
) -> None:
    state_path = _create(tmp_path)
    first_dir = _begin(state_path, "articulation")
    future_dir = first_dir / "attempts" / "02"
    future_dir.mkdir(parents=True)
    future_evidence = future_dir / "future-evidence.json"
    future_evidence.write_text('{"status":"conditional"}\n', encoding="utf-8")

    with pytest.raises(
        AssetCompositionStateError,
        match="review evidence must belong to the current stage attempt",
    ):
        _write_review(
            state_path,
            "articulation",
            decision="refine",
            evidence=[future_evidence],
            repair_scope=["Do not pre-populate a later attempt."],
        )


def test_refinement_rejects_review_evidence_from_a_prior_attempt(
    tmp_path: Path,
) -> None:
    state_path = _create(tmp_path)
    first_dir = _begin(state_path, "articulation")
    first_evidence = first_dir / "first-evidence.json"
    first_evidence.write_text('{"status":"conditional"}\n', encoding="utf-8")
    _write_review(
        state_path,
        "articulation",
        decision="refine",
        evidence=[first_evidence],
        repair_scope=["retry in a distinct attempt"],
    )
    _write_plan(state_path, "articulation", evidence=first_evidence)
    begin_stage(state_path, "articulation", actor="test-agent")
    second_dir = stage_directory(state_path, "articulation")
    second_output = second_dir / "articulation.usda"
    second_output.write_text("#usda 1.0\n", encoding="utf-8")

    with pytest.raises(
        AssetCompositionStateError,
        match="review evidence must belong to the current stage attempt",
    ):
        _write_review(
            state_path,
            "articulation",
            decision="refine",
            evidence=[first_evidence],
            output=second_output,
            repair_scope=["do not reuse archived evidence"],
        )


def test_review_can_seal_realistic_texture_evidence_manifest(tmp_path: Path) -> None:
    state_path = _create(tmp_path)
    stage_dir = _begin(state_path, "articulation")
    evidence: list[Path] = []
    for index in range(80):
        path = stage_dir / f"artifact-{index:03d}.json"
        path.write_text(json.dumps({"index": index}) + "\n", encoding="utf-8")
        evidence.append(path)

    _write_review(
        state_path,
        "articulation",
        decision="refine",
        evidence=evidence,
        repair_scope=["Retry after reviewing the complete artifact manifest."],
    )

    run = load_verified_run(state_path)
    review = json.loads(
        Path(run.coordinator.evidence_reviews[-1].path).read_text(encoding="utf-8")
    )
    assert len(review["evidence"]) == 80


def test_verified_run_hashes_repeated_chain_bindings_once_per_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_path = _create(tmp_path)
    _write_plan(state_path, "articulation")
    request_path = str((state_path.parent / "request.json").resolve())
    plan_path = load_verified_run(state_path).coordinator.plan_revisions[-1].path
    calls: list[str] = []
    real_binding = asset_state._binding

    def counting_binding(
        path: str | Path,
        *,
        label: str,
        required_root: Path | None = None,
    ) -> ArtifactBinding:
        calls.append(str(Path(path).expanduser().resolve()))
        return real_binding(path, label=label, required_root=required_root)

    monkeypatch.setattr(asset_state, "_binding", counting_binding)

    load_verified_run(state_path)

    # The request is checked once as an external frozen input and once under the
    # stricter run-root policy used for plan evidence. Same-policy chain
    # references, such as the plan binding itself, are hashed only once.
    assert calls.count(request_path) == 2
    assert calls.count(plan_path) == 1

    calls.clear()
    load_verified_run(state_path)

    assert calls.count(request_path) == 2
    assert calls.count(plan_path) == 1


def test_physics_accept_requires_coordinator_owned_decision_patch(
    tmp_path: Path,
) -> None:
    state_path = _create(tmp_path)
    _complete_articulation(state_path)
    _complete_simple_stage(state_path, "material")
    _complete_simple_stage(state_path, "texture")

    with pytest.raises(
        AssetCompositionStateError,
        match="requires one coordinator decision patch",
    ):
        _complete_physics_stage(state_path, omit_coordinator_patch=True)

    run = load_verified_run(state_path)
    assert run.current_stage == "physics"
    assert run.coordinator.next_action == "execute_stage"


def test_physics_catalog_rejects_retired_workbench_schema() -> None:
    from content_agent_workflows.physics import add_physics_component_target_catalog

    payload = {
        "schema_version": "content-" + "workbench.physics-components.v2",
        "components": [],
    }

    with pytest.raises(
        RuntimeError,
        match="Unsupported physics component inspection schema_version",
    ):
        add_physics_component_target_catalog(payload)


def test_physics_accept_rejects_decisions_for_stale_input(tmp_path: Path) -> None:
    state_path = _create(tmp_path)
    _complete_articulation(state_path)
    _complete_simple_stage(state_path, "material")
    _complete_simple_stage(state_path, "texture")

    with pytest.raises(
        AssetCompositionStateError,
        match="do not match the coordinator-authored input",
    ):
        _complete_physics_stage(state_path, decision_source_digest="9" * 64)

    run = load_verified_run(state_path)
    assert run.current_stage == "physics"
    assert run.coordinator.next_action == "execute_stage"


def test_physics_accept_rejects_unbound_simulation_report(tmp_path: Path) -> None:
    state_path = _create(tmp_path)
    _complete_articulation(state_path)
    _complete_simple_stage(state_path, "material")
    _complete_simple_stage(state_path, "texture")

    with pytest.raises(
        AssetCompositionStateError,
        match="Physics runtime output path differs",
    ):
        _complete_physics_stage(
            state_path,
            simulation_report_overrides={"physics_usd": str(tmp_path / "other.usda")},
        )

    run = load_verified_run(state_path)
    assert run.current_stage == "physics"
    assert run.coordinator.next_action == "execute_stage"


def test_physics_accept_binds_topology_report_to_coordinator_plan(
    tmp_path: Path,
) -> None:
    state_path = _create(tmp_path)
    _complete_articulation(state_path)
    _complete_simple_stage(state_path, "material")
    _complete_simple_stage(state_path, "texture")

    output = _complete_physics_stage(state_path, with_topology_plan=True)

    assert output.is_file()
    assert load_verified_run(state_path).stages["physics"].status == "completed"


def test_physics_accept_resolves_coordinator_target_ids_before_rebase(
    tmp_path: Path,
) -> None:
    state_path = _create(tmp_path)
    _complete_articulation(
        state_path,
        with_rigid_bodies=False,
        drawer_only_rigid_body=True,
        with_drawer_visual=True,
    )
    _complete_simple_stage(state_path, "material")
    _complete_simple_stage(state_path, "texture")

    output = _complete_physics_stage(
        state_path,
        with_topology_plan=True,
        coordinator_uses_target_ids=True,
    )

    assert output.is_file()
    assert load_verified_run(state_path).stages["physics"].status == "completed"


def test_physics_accept_resolves_target_ids_without_topology_repair(
    tmp_path: Path,
) -> None:
    state_path = _create(tmp_path)
    _complete_articulation(
        state_path,
        with_rigid_bodies=False,
        drawer_only_rigid_body=True,
        with_drawer_visual=True,
    )
    _complete_simple_stage(state_path, "material")
    _complete_simple_stage(state_path, "texture")

    output = _complete_physics_stage(
        state_path,
        coordinator_uses_target_ids=True,
    )

    assert output.is_file()
    assert load_verified_run(state_path).stages["physics"].status == "completed"


def test_physics_accept_rejects_assignment_path_provenance_mismatch(
    tmp_path: Path,
) -> None:
    state_path = _create(tmp_path)
    _complete_articulation(state_path)
    _complete_simple_stage(state_path, "material")
    _complete_simple_stage(state_path, "texture")

    with pytest.raises(
        AssetCompositionStateError,
        match="assignment path provenance differs",
    ):
        _complete_physics_stage(
            state_path,
            assignment_path_space="inspection",
            assignment_source_path_expansions={
                "/World/Drawer": ["/Stale/Drawer"],
            },
        )


@pytest.mark.parametrize(
    "digest_field",
    ["source_asset_sha256", "prepared_asset_sha256"],
)
def test_physics_accept_rejects_assignment_asset_digest_mismatch(
    tmp_path: Path,
    digest_field: str,
) -> None:
    state_path = _create(tmp_path)
    _complete_articulation(state_path)
    _complete_simple_stage(state_path, "material")
    _complete_simple_stage(state_path, "texture")

    with pytest.raises(
        AssetCompositionStateError,
        match="stage input, output, or native patch differs",
    ):
        _complete_physics_stage(
            state_path,
            assignment_overrides={digest_field: "0" * 64},
        )


def test_physics_accept_allows_executor_reset_annotation(tmp_path: Path) -> None:
    state_path = _create(tmp_path)
    _complete_articulation(state_path)
    _complete_simple_stage(state_path, "material")
    _complete_simple_stage(state_path, "texture")
    operation = {
        "op": "ensure_rigid_body_api",
        "prim_path": "/World/Drawer",
    }

    output = _complete_physics_stage(
        state_path,
        with_topology_plan=True,
        topology_operations=[operation],
        topology_report_overrides={
            "applied_operations": [{**operation, "reset_xform_stack": "preserve_world"}]
        },
    )

    assert output.is_file()
    assert load_verified_run(state_path).stages["physics"].status == "completed"


def test_physics_accept_binds_joint_endpoint_owner_promotion_report(
    tmp_path: Path,
) -> None:
    state_path = _create(tmp_path)
    _complete_articulation(state_path)
    _complete_simple_stage(state_path, "material")
    _complete_simple_stage(state_path, "texture")
    operation = {
        "op": "ensure_rigid_body_api",
        "prim_path": "/World/Drawer",
    }
    promotion = {
        "joint_prim_path": "/World/DrawerJoint",
        "relationship": "body1",
        "relationship_target_path": "/World/Drawer",
        "requested_rigid_body_ancestor_path": "/World/Drawer",
    }

    output = _complete_physics_stage(
        state_path,
        with_topology_plan=True,
        topology_operations=[operation],
        topology_promotions=[promotion],
    )

    assert output.is_file()
    assert load_verified_run(state_path).stages["physics"].status == "completed"


def test_physics_accept_rejects_mismatched_owner_promotion_report(
    tmp_path: Path,
) -> None:
    state_path = _create(tmp_path)
    _complete_articulation(state_path)
    _complete_simple_stage(state_path, "material")
    _complete_simple_stage(state_path, "texture")
    promotion = {
        "joint_prim_path": "/World/DrawerJoint",
        "relationship": "body1",
        "relationship_target_path": "/World/Drawer",
        "requested_rigid_body_ancestor_path": "/World/Drawer",
    }

    with pytest.raises(
        AssetCompositionStateError,
        match="joint endpoint owner promotions differs",
    ):
        _complete_physics_stage(
            state_path,
            with_topology_plan=True,
            topology_operations=[
                {
                    "op": "ensure_rigid_body_api",
                    "prim_path": "/World/Drawer",
                }
            ],
            topology_promotions=[promotion],
            topology_report_overrides={"applied_joint_endpoint_owner_promotions": []},
        )


def test_physics_accept_rejects_tampered_rebased_decisions(tmp_path: Path) -> None:
    state_path = _create(tmp_path)
    _complete_articulation(state_path)
    _complete_simple_stage(state_path, "material")
    _complete_simple_stage(state_path, "texture")

    with pytest.raises(
        AssetCompositionStateError,
        match="not the deterministic rebase of coordinator decisions",
    ):
        _complete_physics_stage(
            state_path,
            with_topology_plan=True,
            applied_decision_overrides={
                "physical_properties": {
                    "density": 1.0,
                    "estimated_mass_kg": 0.01,
                    "static_friction": 0.5,
                    "dynamic_friction": 0.4,
                    "restitution": 0.1,
                }
            },
        )

    run = load_verified_run(state_path)
    assert run.current_stage == "physics"
    assert run.coordinator.next_action == "execute_stage"


def test_physics_accept_rejects_unresolved_target_id_applied_patch(
    tmp_path: Path,
) -> None:
    state_path = _create(tmp_path)
    _complete_articulation(state_path)
    _complete_simple_stage(state_path, "material")
    _complete_simple_stage(state_path, "texture")

    with pytest.raises(
        AssetCompositionStateError,
        match="applied patch must contain resolved V2 component decisions",
    ):
        _complete_physics_stage(
            state_path,
            with_topology_plan=True,
            raw_applied_decision={
                "decision_id": "drawer-decision",
                "component_id": "drawer-component",
                "collider_target_ids": ["drawer-component:visual:target"],
                "collision_mode": "author_on_targets",
                "inferred_material_family": "wood",
                "inferred_material_name": None,
                "collision_approximation": "convexHull",
                "physical_properties": {
                    "density": 650.0,
                    "estimated_mass_kg": 1.0,
                    "static_friction": 0.6,
                    "dynamic_friction": 0.5,
                    "restitution": 0.1,
                },
                "confidence": 0.9,
                "rationale": "Fixture target-ID decision.",
                "rigid_body_grouping": None,
                "quality_warnings": [],
            },
        )

    run = load_verified_run(state_path)
    assert run.current_stage == "physics"
    assert run.coordinator.next_action == "execute_stage"


def test_physics_accept_rejects_topology_report_for_different_plan(
    tmp_path: Path,
) -> None:
    state_path = _create(tmp_path)
    _complete_articulation(state_path)
    _complete_simple_stage(state_path, "material")
    _complete_simple_stage(state_path, "texture")

    with pytest.raises(
        AssetCompositionStateError,
        match="Physics topology report applied_operations differs",
    ):
        _complete_physics_stage(
            state_path,
            with_topology_plan=True,
            topology_report_overrides={
                "applied_operations": [
                    {"op": "remove_fixed_joint", "prim_path": "/World/Other"}
                ]
            },
        )

    run = load_verified_run(state_path)
    assert run.current_stage == "physics"
    assert run.coordinator.next_action == "execute_stage"


def test_invalid_physics_accept_remains_revisable_and_archives_reviewed_attempt(
    tmp_path: Path,
) -> None:
    state_path = _create(tmp_path)
    _complete_articulation(state_path)
    _complete_simple_stage(state_path, "material")
    _complete_simple_stage(state_path, "texture")
    stage_dir = _begin(state_path, "physics")
    output = stage_dir / "conditional-physics.usda"
    output.write_text('#usda 1.0\ndef Xform "Physics" {}\n', encoding="utf-8")
    native = physics_validation_evidence(
        asset=str(output.resolve()),
        target_runtime="fake",
        physics_properties_status="pass",
        runtime_loadability_status="pass",
        no_explosions_status="pass",
    )
    payload = native.model_dump(mode="json")
    metadata = payload["metadata"]
    assert isinstance(metadata, dict)
    metadata["asset_sha256"] = file_sha256(output)
    payload["sim_ready_status"] = "conditional"
    evidence = stage_dir / "validation_evidence.json"
    evidence.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    review_count = len(load_verified_run(state_path).coordinator.evidence_reviews)

    with pytest.raises(
        AssetCompositionStateError,
        match="native runtime validation must pass",
    ):
        _write_review(
            state_path,
            "physics",
            decision="accept",
            evidence=[evidence],
            output=output,
        )

    rejected = load_verified_run(state_path)
    assert rejected.coordinator.next_action == "execute_stage"
    assert len(rejected.coordinator.evidence_reviews) == review_count

    _write_review(
        state_path,
        "physics",
        decision="refine",
        evidence=[evidence],
        output=output,
        repair_scope=["Repair the conditional runtime result."],
    )
    refined = load_verified_run(state_path)
    archived = refined.stages["physics"].superseded_attempts[-1]
    assert archived.output_asset is not None
    assert archived.output_asset.path == str(output)
    assert [item.path for item in archived.evidence] == [str(evidence)]
    assert refined.coordinator.next_action == "plan"


def test_schema_readback_mode_accepts_disclosed_runtime_gap(tmp_path: Path) -> None:
    state_path = _create(
        tmp_path,
        physics_validation_mode="schema_readback",
    )
    _complete_articulation(state_path)
    _complete_simple_stage(state_path, "material")
    _complete_simple_stage(state_path, "texture")

    _complete_physics_stage(
        state_path,
        runtime_loadability_status="not_evaluated",
        no_explosions_status="not_evaluated",
        simulation_report_overrides={
            "not_evaluated": True,
            "trajectory_jsonl": None,
            "warnings": ["Multi-body runtime simulation is not supported."],
        },
    )

    run = load_verified_run(state_path)
    assert run.stages["physics"].status == "completed"
    assert run.current_stage == "validation"

    _complete_simple_stage(state_path, "validation")
    validated = load_verified_run(state_path)
    assert validated.stages["validation"].status == "completed"


def test_runtime_required_rejects_not_evaluated_simulation_report(
    tmp_path: Path,
) -> None:
    state_path = _create(tmp_path)
    _complete_articulation(state_path)
    _complete_simple_stage(state_path, "material")
    _complete_simple_stage(state_path, "texture")

    with pytest.raises(
        AssetCompositionStateError,
        match="runtime report is marked not_evaluated",
    ):
        _complete_physics_stage(
            state_path,
            simulation_report_overrides={
                "not_evaluated": True,
                "trajectory_jsonl": None,
                "warnings": ["Runtime was not evaluated."],
            },
        )


def test_schema_readback_validation_rejects_a_physics_pass_claim(
    tmp_path: Path,
) -> None:
    state_path = _create(tmp_path, physics_validation_mode="schema_readback")
    _complete_articulation(state_path)
    _complete_simple_stage(state_path, "material")
    _complete_simple_stage(state_path, "texture")
    _complete_physics_stage(
        state_path,
        runtime_loadability_status="not_evaluated",
        no_explosions_status="not_evaluated",
        simulation_report_overrides={
            "not_evaluated": True,
            "trajectory_jsonl": None,
            "warnings": ["Multi-body runtime simulation is not supported."],
        },
    )
    directory = _begin(state_path, "validation")
    input_asset = load_verified_run(state_path).stages["validation"].input_asset
    assert input_asset is not None
    output = directory / "validation.usda"
    output.write_bytes(Path(input_asset.path).read_bytes())
    evidence = _write_validation_evidence(state_path, directory)
    cross_stage = next(
        path for path in evidence if path.name == "cross_stage_validation.json"
    )
    payload = json.loads(cross_stage.read_text(encoding="utf-8"))
    physics_claim = next(
        claim for claim in payload["claims"] if claim["name"] == "physics_behavior"
    )
    physics_claim["status"] = "pass"
    cross_stage.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(
        AssetCompositionStateError,
        match="physics_behavior claim must match",
    ):
        _write_review(
            state_path,
            "validation",
            decision="accept",
            evidence=evidence,
            output=output,
        )


def test_coordinator_records_evidence_backed_stop_decision(tmp_path: Path) -> None:
    state_path = _create(tmp_path)
    stage_dir = _begin(state_path, "articulation")
    evidence = stage_dir / "backend-error.json"
    evidence.write_text('{"error":"Joint backend unavailable"}\n', encoding="utf-8")

    _write_review(
        state_path,
        "articulation",
        decision="stop_failed",
        evidence=[evidence],
    )

    stopped = load_verified_run(state_path)
    assert stopped.terminal_status == "failed"
    assert stopped.coordinator.next_action == "stopped"
    assert stopped.coordinator.stop_reason == "Coordinator selected stop_failed."
    assert stopped.stages["articulation"].status == "failed"

    stopped_attempt = stage_dir
    recovered = recover_stage(
        state_path,
        "articulation",
        reason="Joint backend was restored.",
        actor="operator",
    )
    assert recovered.coordinator.next_action == "plan"
    assert recovered.coordinator.stop_reason is None
    assert not recovered.stages["articulation"].continue_current_attempt
    _write_plan(state_path, "articulation", reason="Retry after backend recovery.")
    begin_stage(state_path, "articulation", actor="test-agent")
    fresh_attempt = stage_directory(state_path, "articulation")

    assert fresh_attempt != stopped_attempt
    assert fresh_attempt.name == "02"
    assert evidence.read_text(encoding="utf-8") == (
        '{"error":"Joint backend unavailable"}\n'
    )
    load_verified_run(state_path)


@pytest.mark.parametrize("decision", ["accept", "await_review"])
def test_interruption_after_sealed_review_recovers_into_fresh_attempt(
    tmp_path: Path,
    decision: str,
) -> None:
    state_path = _create(tmp_path)
    stage: StageName = "material" if decision == "accept" else "articulation"
    if stage == "material":
        _complete_articulation(state_path)
    directory = _begin(state_path, stage)
    output = directory / "reviewed.usda"
    output.write_text('#usda 1.0\ndef Xform "Reviewed" {}\n', encoding="utf-8")
    evidence = directory / "reviewed-evidence.json"
    evidence.write_text('{"status":"pass"}\n', encoding="utf-8")
    evidence_paths = (
        _write_material_evidence(state_path, directory, output, evidence)
        if stage == "material"
        else [evidence]
    )
    _write_review(
        state_path,
        stage,
        decision=decision,
        evidence=evidence_paths,
        output=output if decision == "accept" else None,
    )
    sealed_bytes = evidence.read_bytes()

    failed = fail_stage(
        state_path,
        stage,
        reason="Child exited after sealing its review.",
        actor="test-runner",
    )
    assert not failed.stages[stage].continue_current_attempt
    recover_stage(
        state_path,
        stage,
        reason="Restart in an isolated attempt.",
        actor="operator",
    )
    _write_plan(state_path, stage, reason="Retry after reviewed exit.")
    begin_stage(state_path, stage, actor="test-agent")

    fresh_attempt = stage_directory(state_path, stage)
    assert fresh_attempt != directory
    assert fresh_attempt.name == "02"
    assert evidence.read_bytes() == sealed_bytes
    load_verified_run(state_path)


def test_interruption_after_joint_decisions_preserves_receipt_bound_attempt(
    tmp_path: Path,
) -> None:
    state_path = _create(tmp_path)
    first_attempt = _begin(state_path, "articulation")
    candidates = first_attempt / "articulation_candidates.json"
    candidates.write_text('{"candidate_ids":["drawer"]}\n', encoding="utf-8")
    _write_review(
        state_path,
        "articulation",
        decision="await_review",
        evidence=[candidates],
    )
    require_review(state_path, candidates_path=candidates, actor="test-agent")
    decisions = state_path.parent / "raw" / "joint-decisions.json"
    decisions.write_text('{"drawer":"accept"}\n', encoding="utf-8")
    reviewed = record_review_decisions(
        state_path,
        decisions_path=decisions,
        reviewer="asset-owner",
    )
    assert reviewed.stages["articulation"].continue_current_attempt
    assert reviewed.stages["articulation"].status == "ready"

    failed = fail_stage(
        state_path,
        "articulation",
        reason="usd-cli failed before begin-stage.",
        actor="test-runner",
    )
    assert failed.stages["articulation"].continue_current_attempt
    recover_stage(
        state_path,
        "articulation",
        reason="usd-cli restored.",
        actor="operator",
    )
    _write_plan(
        state_path,
        "articulation",
        evidence=candidates,
        reason="Continue the receipt-bound authoring attempt.",
    )
    begin_stage(state_path, "articulation", actor="test-agent")

    assert stage_directory(state_path, "articulation") == first_attempt
    assert load_verified_run(state_path).stages["articulation"].attempt_count == 1


def test_interrupted_attempt_plan_cannot_seal_live_attempt_evidence(
    tmp_path: Path,
) -> None:
    state_path = _create(tmp_path)
    attempt = _begin(state_path, "articulation")
    partial = attempt / "partial-executor-evidence.json"
    partial.write_text('{"progress":1}\n', encoding="utf-8")
    fail_stage(
        state_path,
        "articulation",
        reason="Executor stopped before evidence review.",
        actor="test-runner",
    )
    recover_stage(
        state_path,
        "articulation",
        reason="Resume the unreviewed executor attempt.",
        actor="operator",
    )

    with pytest.raises(
        AssetCompositionStateError,
        match="cannot bind mutable current stage-attempt evidence",
    ):
        _write_plan(
            state_path,
            "articulation",
            evidence=partial,
            reason="Unsafe partial-evidence plan.",
        )

    _write_plan(state_path, "articulation", reason="Resume from frozen evidence.")
    begin_stage(state_path, "articulation", actor="test-agent")
    partial.write_text('{"progress":2}\n', encoding="utf-8")

    assert stage_directory(state_path, "articulation") == attempt
    load_verified_run(state_path)


def test_downstream_evidence_revisits_material_and_preserves_joint_receipt(
    tmp_path: Path,
) -> None:
    state_path = _create(tmp_path)
    _complete_articulation(state_path)
    before_material = load_verified_run(state_path)
    joint_receipt = before_material.stages["articulation"].review_decisions
    request_binding = before_material.request
    assert joint_receipt is not None

    _complete_simple_stage(state_path, "material")
    _complete_simple_stage(state_path, "texture")
    physics_dir = _begin(state_path, "physics")
    rejected_output = physics_dir / "single-body-physics.usda"
    rejected_output.write_text(
        '#usda 1.0\ndef Xform "SingleBody" {}\n',
        encoding="utf-8",
    )
    downstream_evidence = physics_dir / "runtime-review.json"
    downstream_evidence.write_text(
        '{"status":"conditional","finding":"same rigid body endpoints"}\n',
        encoding="utf-8",
    )

    _write_review(
        state_path,
        "physics",
        decision="revisit",
        evidence=[downstream_evidence],
        output=rejected_output,
        target_stage="material",
        repair_scope=["drawer-front appearance and downstream physical grouping"],
    )

    revisited = load_verified_run(state_path)
    assert revisited.current_stage == "material"
    assert revisited.stages["material"].status == "ready"
    assert revisited.stages["texture"].status == "pending"
    assert revisited.stages["physics"].status == "pending"
    assert revisited.coordinator.revisit_count == 1
    assert revisited.coordinator.next_action == "plan"
    assert revisited.request == request_binding
    assert revisited.stages["articulation"].review_decisions == joint_receipt
    assert revisited.stages["articulation"].status == "completed"
    assert len(revisited.stages["material"].superseded_attempts) == 1
    assert len(revisited.stages["texture"].superseded_attempts) == 1
    assert len(revisited.stages["physics"].superseded_attempts) == 1
    archived_physics = revisited.stages["physics"].superseded_attempts[0]
    assert archived_physics.output_asset is not None
    assert archived_physics.output_asset.path == str(rejected_output)
    assert [item.path for item in archived_physics.evidence] == [
        str(downstream_evidence)
    ]

    _write_plan(
        state_path,
        "material",
        evidence=downstream_evidence,
        reason="Physics evidence requires a bounded Material-stage revisit.",
    )
    begin_stage(state_path, "material", actor="test-agent")
    assert stage_directory(state_path, "material").name == "02"


def test_revisit_rejects_a_stage_not_enabled_for_the_run(tmp_path: Path) -> None:
    state_path = _create(tmp_path)
    _complete_articulation(state_path)
    material_dir = _begin(state_path, "material")
    evidence = material_dir / "material-review.json"
    evidence.write_text('{"status":"repair"}\n', encoding="utf-8")
    draft = material_dir / "revisit-disabled-geometry.json"
    draft.write_text(
        json.dumps(
            {
                "schema_version": (
                    "content-agent-workflows.asset-coordinator-review-draft.v1"
                ),
                "stage": "material",
                "output_asset_path": None,
                "evidence_paths": [str(evidence)],
                "findings": ["Malformed target stage."],
                "decision": "revisit",
                "target_stage": "geometry",
                "decision_summary": "Attempt to revisit a disabled stage.",
                "repair_scope": ["geometry"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(AssetCompositionStateError, match="is not enabled"):
        record_coordinator_evidence_review(
            state_path,
            review_path=draft,
            actor="test-agent",
        )


def test_revisit_cannot_invalidate_frozen_joint_review(tmp_path: Path) -> None:
    state_path = _create(tmp_path)
    _complete_articulation(state_path)
    material_dir = _begin(state_path, "material")
    evidence = material_dir / "material-review.json"
    evidence.write_text('{"status":"repair"}\n', encoding="utf-8")
    draft = material_dir / "revisit-articulation.json"
    draft.write_text(
        json.dumps(
            {
                "schema_version": (
                    "content-agent-workflows.asset-coordinator-review-draft.v1"
                ),
                "stage": "material",
                "output_asset_path": None,
                "evidence_paths": [str(evidence)],
                "findings": ["A downstream issue was observed."],
                "decision": "revisit",
                "target_stage": "articulation",
                "decision_summary": "Attempt to replace reviewed joints.",
                "repair_scope": ["joint topology"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(AssetCompositionStateError, match="frozen Joint review"):
        record_coordinator_evidence_review(
            state_path,
            review_path=draft,
            actor="test-agent",
        )

    unchanged = load_verified_run(state_path)
    assert unchanged.current_stage == "material"
    assert unchanged.stages["articulation"].status == "completed"
    assert unchanged.coordinator.revisit_count == 0
