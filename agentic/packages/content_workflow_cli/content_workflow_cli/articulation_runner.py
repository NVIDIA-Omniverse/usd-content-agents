# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Thin public launcher for the durable articulation-v1 workflow."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import stat
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields, replace
from functools import partial
from pathlib import Path
from typing import Any, ClassVar, Literal, cast, overload

from content_agent_workflows.articulation import (
    EMBEDDED_ARTICULATION_IMPLEMENTATION_MANIFEST,
    SKILL_ROUTED_DECISION_METADATA_KEY,
    ArticulationAuthoringClient,
    ArticulationFinalizationResult,
    ArticulationRunState,
    ArticulationWorkflowError,
    ArticulationWorkflowRequest,
    EmbeddedArticulationCanonicalGraph,
    EmbeddedArticulationCapabilityLimits,
    EmbeddedArticulationDecisionPatch,
    EmbeddedArticulationGraphRevisionPatch,
    EmbeddedArticulationPostReviewPatch,
    EmbeddedArticulationPreparation,
    EmbeddedArticulationRuntime,
    JointAgentGraphAuthoringClient,
    JointAgentLocalClient,
    StandaloneArticulationDecisionPatch,
    StandaloneArticulationPostReviewPatch,
    StandaloneArticulationPreparation,
    apply_embedded_articulation_decision_patch,
    apply_embedded_articulation_graph_revision,
    apply_embedded_articulation_post_review,
    apply_standalone_articulation_decision_patch,
    apply_standalone_articulation_post_review,
    bind_embedded_articulation_output_evidence,
    bind_standalone_articulation_output_evidence,
    build_articulation_review_receipt,
    build_articulation_step_observation,
    build_standalone_articulation_observation,
    run_batch_articulation_workflow,
    run_embedded_articulation_workflow,
    run_standalone_articulation_workflow,
    write_articulation_workflow_summary,
)
from content_agent_workflows.articulation import (
    run_articulation_workflow as advance_articulation_state_machine,
)
from content_agent_workflows.articulation.models import (
    TERMINAL_ARTICULATION_PHASES,
)
from content_agent_workflows.asset_composition import (
    build_embedded_domain_decision_identity,
    build_embedded_domain_execution_context,
    load_embedded_articulation_human_acceptance,
    load_verified_run,
    record_articulation_graph_revision,
    run_asset_coordinator_transition,
    validate_articulation_graph_revision_request,
)
from content_agent_workflows.common.artifacts import atomic_write_json
from content_agent_workflows.common.domain_execution import (
    DomainExecutionContext,
    metadata_with_domain_execution_context,
)
from content_agent_workflows.common.embedded_domain_decision import (
    EmbeddedDomainEvidence,
    EmbeddedDomainProposal,
    EmbeddedHumanDecision,
    ProducerIdentity,
    artifact_reference,
    canonical_json_digest,
)
from world_understanding.utils.artifacts import (
    ArtifactPathError,
    open_regular_file_no_follow,
)

from .child_launch import (
    ARTICULATION_CHILD_LAUNCH_PROFILE,
    ChildLaunchArtifactIdentity,
    child_launch_artifact_identity,
)
from .runner import (
    CLAUDE_EXECUTION_CLI,
    CLAUDE_EXECUTION_SDK,
    CODEX_SANDBOX_WORKSPACE_WRITE,
    RUNNER_CLAUDE,
    RUNNER_CODEX,
    _lexical_absolute_path,
    _reject_unsafe_run_links,
    _skill_routed_child_session_lock,
    _skill_routed_run_lock,
    parent_usd_cli_prompt_contract,
    run_child_agent,
    start_parent_usd_cli_capability,
    stop_parent_usd_cli_capability,
)
from .trace import TraceWriter, build_trace

ARTICULATION_LAUNCHER_METADATA_KEY = "content_workflow_cli"
ARTICULATION_LAUNCHER_SCHEMA_VERSION = "content-workflow-cli.articulation-launcher.v2"
ARTICULATION_LEGACY_LAUNCHER_SCHEMA_VERSION = (
    "content-workflow-cli.articulation-launcher.v1"
)
ARTICULATION_SCENE_POLICY_ID = "content-workflow-cli.articulation-source-evidence.v1"
ARTICULATION_EXECUTION_SKILL_ROUTED = "skill-routed"
ARTICULATION_EXECUTION_FIXED = "fixed"
ARTICULATION_AGENT_LAUNCHER_SCHEMA_VERSION = (
    "content-workflow-cli.articulation-agent-launcher.v2"
)
ARTICULATION_LEGACY_AGENT_LAUNCHER_SCHEMA_VERSION = (
    "content-workflow-cli.articulation-agent-launcher.v1"
)
ARTICULATION_AGENT_LAUNCHER_POLICY_SCHEMA_VERSION = (
    "content-workflow-cli.articulation-agent-launcher-policy.v1"
)
MAX_ARTICULATION_AGENT_LAUNCHER_BYTES = 1024 * 1024
MAX_ARTICULATION_AGENT_LAUNCHER_POLICY_BYTES = 16 * 1024
MAX_ARTICULATION_CHILD_FINAL_BYTES = 1024 * 1024
_REQUESTED_REVIEW_POLICY_METADATA_KEY = "content_workflow_cli.requested_review_policy"
_ARTICULATION_SCENE_EVIDENCE_REQUIRED_METADATA_KEY = (
    "content_agent_workflows.scene_evidence_required"
)
ArticulationReviewPolicy = Literal["uncertain", "all", "none"]
ArticulationMotionType = Literal["revolute", "prismatic"]
ArticulationReviewDecision = Literal["accept", "reject"]
EmbeddedArticulationReviewDecision = Literal["accept", "reject", "revise"]
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ArticulationRunConfig:
    """Resolved inputs for one public articulation launcher invocation."""

    repo_root: Path
    source_asset: Path
    output_dir: Path
    joint_config: Path | None
    intent: str
    review_policy: ArticulationReviewPolicy = "uncertain"
    allowed_motion_types: tuple[ArticulationMotionType, ...] = (
        "revolute",
        "prismatic",
    )
    expected_candidate_count: int | None = None
    max_candidate_count: int = 64
    joint_session_id: str | None = None
    embedded_run_state: Path | None = None
    embedded_preparation: Path | None = None
    verbose: bool = False
    scene_timeout_seconds: float = 60.0
    execution_mode: str = ARTICULATION_EXECUTION_SKILL_ROUTED
    runner: str = RUNNER_CODEX
    model: str | None = None
    model_reasoning_effort: str | None = None
    codex_base_url: str | None = None
    codex_sandbox_mode: str = CODEX_SANDBOX_WORKSPACE_WRITE
    codex_config: dict[str, object] | None = None
    claude_config: dict[str, object] | None = None
    claude_permission_mode: str = "default"
    claude_max_turns: int | None = None
    claude_execution_mode: str = CLAUDE_EXECUTION_SDK
    child_timeout_seconds: float = 1800.0
    agent_cwd: Path | None = None
    dry_run: bool = False


@dataclass(frozen=True)
class _StoredLauncherConfig:
    joint_config: Path | None
    joint_session_id: str | None
    embedded_run_state: Path | None = None
    embedded_preparation: Path | None = None
    # Launcher v1 records created before skill routing have no execution_mode.
    # Preserve their fixed-controller behavior instead of silently upgrading them.
    execution_mode: str = ARTICULATION_EXECUTION_FIXED
    requested_review_policy: ArticulationReviewPolicy = "uncertain"
    runner: str = RUNNER_CODEX
    model: str | None = None
    model_reasoning_effort: str | None = None
    codex_base_url: str | None = None
    codex_sandbox_mode: str = CODEX_SANDBOX_WORKSPACE_WRITE
    codex_config: dict[str, object] | None = None
    claude_config: dict[str, object] | None = None
    claude_permission_mode: str = "default"
    claude_max_turns: int | None = None
    claude_execution_mode: str = CLAUDE_EXECUTION_SDK
    child_timeout_seconds: float = 1800.0
    agent_cwd: Path | None = None


@dataclass(frozen=True)
class _ArticulationChildRuntimeConfig:
    child_launch_profile: ClassVar[str] = ARTICULATION_CHILD_LAUNCH_PROFILE
    repo_root: Path
    usd_path: Path
    runner: str
    model: str | None
    model_reasoning_effort: str | None
    codex_base_url: str | None
    codex_sandbox_mode: str
    codex_config: dict[str, object] | None
    claude_config: dict[str, object] | None
    claude_permission_mode: str
    claude_max_turns: int | None
    claude_execution_mode: str
    child_timeout_seconds: float
    agent_cwd: Path | None
    reference_images: list[Path]
    reference_files: list[Path] | None
    child_capability_inventory: ChildLaunchArtifactIdentity | None = None
    child_domain_policy_bounds: ChildLaunchArtifactIdentity | None = None
    child_forbidden_environment_names: tuple[str, ...] = ()
    parent_usd_cli_session_identity: Path | None = None
    parent_usd_cli_session_identity_sha256: str | None = None
    workflow_skill: ClassVar[str] = "content-workflow-articulation"


@dataclass(frozen=True, slots=True)
class _ArticulationChildLaunchRequired:
    """Signal that the parent lease must be released before child launch."""


_ARTICULATION_CHILD_LAUNCH_REQUIRED = _ArticulationChildLaunchRequired()


def run_articulation_workflow(
    config: ArticulationRunConfig,
) -> ArticulationFinalizationResult:
    """Run or resume the exact articulation request described by ``config``."""

    request, normalized_config = _build_request(config)
    return _run_request(request, normalized_config)


def resume_articulation_workflow(
    run_dir: str | Path,
    *,
    repo_root: Path,
    scene_timeout_seconds: float = 60.0,
    verbose: bool = False,
) -> ArticulationFinalizationResult:
    """Resume an existing CLI-created articulation run without changing its request."""

    resolved_run_dir = _verified_existing_run_dir(run_dir)
    request = _load_request(resolved_run_dir)
    launcher = _load_launcher_config(request)
    if launcher.execution_mode == ARTICULATION_EXECUTION_SKILL_ROUTED:
        with _skill_routed_run_lock(resolved_run_dir, domain="articulation"):
            outcome = _resume_articulation_workflow_locked(
                resolved_run_dir,
                repo_root=repo_root,
                scene_timeout_seconds=scene_timeout_seconds,
                verbose=verbose,
            )
        return _complete_locked_articulation_resume(
            outcome,
            resolved_run_dir,
            repo_root=repo_root,
            scene_timeout_seconds=scene_timeout_seconds,
            verbose=verbose,
        )
    config = _articulation_resume_config(
        request,
        launcher,
        repo_root=repo_root,
        scene_timeout_seconds=scene_timeout_seconds,
        verbose=verbose,
    )
    return _run_request(request, config)


def _articulation_resume_config(
    request: ArticulationWorkflowRequest,
    launcher: _StoredLauncherConfig,
    *,
    repo_root: Path,
    scene_timeout_seconds: float,
    verbose: bool,
) -> ArticulationRunConfig:
    return ArticulationRunConfig(
        repo_root=repo_root.resolve(),
        source_asset=Path(request.source_asset),
        output_dir=request.output_dir,
        joint_config=launcher.joint_config,
        intent=request.intent,
        review_policy=request.review_policy,
        allowed_motion_types=request.allowed_motion_types,
        expected_candidate_count=request.expected_candidate_count,
        max_candidate_count=request.max_candidate_count,
        joint_session_id=launcher.joint_session_id,
        embedded_run_state=launcher.embedded_run_state,
        embedded_preparation=launcher.embedded_preparation,
        verbose=verbose,
        scene_timeout_seconds=scene_timeout_seconds,
        execution_mode=launcher.execution_mode,
        runner=launcher.runner,
        model=launcher.model,
        model_reasoning_effort=launcher.model_reasoning_effort,
        codex_base_url=launcher.codex_base_url,
        codex_sandbox_mode=launcher.codex_sandbox_mode,
        codex_config=launcher.codex_config,
        claude_config=launcher.claude_config,
        claude_permission_mode=launcher.claude_permission_mode,
        claude_max_turns=launcher.claude_max_turns,
        claude_execution_mode=launcher.claude_execution_mode,
        child_timeout_seconds=launcher.child_timeout_seconds,
        agent_cwd=launcher.agent_cwd,
    )


def _resume_articulation_workflow_locked(
    run_dir: Path,
    *,
    repo_root: Path,
    scene_timeout_seconds: float,
    verbose: bool,
) -> ArticulationFinalizationResult | _ArticulationChildLaunchRequired:
    """Resume a skill-routed run while its parent-owned lease is already held."""

    request = _load_request(run_dir)
    launcher = _load_launcher_config(request)
    if launcher.execution_mode != ARTICULATION_EXECUTION_SKILL_ROUTED:
        raise ValueError("Locked Articulation resume requires skill-routed execution")
    config = _articulation_resume_config(
        request,
        launcher,
        repo_root=repo_root,
        scene_timeout_seconds=scene_timeout_seconds,
        verbose=verbose,
    )
    return _run_skill_routed_articulation_request_locked(request, config)


def _complete_locked_articulation_resume(
    outcome: ArticulationFinalizationResult | _ArticulationChildLaunchRequired,
    run_dir: Path,
    *,
    repo_root: Path,
    scene_timeout_seconds: float,
    verbose: bool,
) -> ArticulationFinalizationResult:
    if not isinstance(outcome, _ArticulationChildLaunchRequired):
        return outcome
    request = _load_request(run_dir)
    launcher = _load_launcher_config(request)
    config = _articulation_resume_config(
        request,
        launcher,
        repo_root=repo_root,
        scene_timeout_seconds=scene_timeout_seconds,
        verbose=verbose,
    )
    return _complete_articulation_child_launch(outcome, request, config)


def review_articulation_workflow(
    run_dir: str | Path,
    decisions_path: str | Path,
    *,
    reviewer: str,
    repo_root: Path,
    scene_timeout_seconds: float = 60.0,
    verbose: bool = False,
) -> ArticulationFinalizationResult:
    """Bind exact review decisions and immediately resume the same workflow."""

    resolved_run_dir = _verified_existing_run_dir(run_dir)
    request, state = _load_verified_request_and_state(resolved_run_dir)
    if state.mode != "batch":
        raise ValueError(
            "Articulation review requires a content-workflow-cli batch run."
        )
    launcher = _load_launcher_config(request)
    if request.execution_context is not None:
        if launcher.embedded_run_state is None:
            raise ValueError("Embedded articulation review lacks the outer run state")
        # Parsing remains an input sanity check; authority comes only from the
        # exact decisions already persisted by the selected review policy.
        _load_decisions(decisions_path, allow_revise=True)
        with _skill_routed_run_lock(resolved_run_dir, domain="articulation"):
            outcome = _resume_articulation_workflow_locked(
                resolved_run_dir,
                repo_root=repo_root,
                scene_timeout_seconds=scene_timeout_seconds,
                verbose=verbose,
            )
        return _complete_locked_articulation_resume(
            outcome,
            resolved_run_dir,
            repo_root=repo_root,
            scene_timeout_seconds=scene_timeout_seconds,
            verbose=verbose,
        )
    decisions = _load_decisions(decisions_path)
    if launcher.execution_mode == ARTICULATION_EXECUTION_SKILL_ROUTED:
        with _skill_routed_run_lock(resolved_run_dir, domain="articulation"):
            request, state = _load_verified_request_and_state(resolved_run_dir)
            if state.mode != "batch":
                raise ValueError(
                    "Articulation review requires a content-workflow-cli batch run."
                )
            locked_launcher = _load_launcher_config(request)
            if locked_launcher.execution_mode != ARTICULATION_EXECUTION_SKILL_ROUTED:
                raise ValueError(
                    "Articulation execution mode changed while waiting for review"
                )
            build_articulation_review_receipt(
                resolved_run_dir,
                decisions,
                reviewer=reviewer,
            )
            outcome = _resume_articulation_workflow_locked(
                resolved_run_dir,
                repo_root=repo_root,
                scene_timeout_seconds=scene_timeout_seconds,
                verbose=verbose,
            )
        return _complete_locked_articulation_resume(
            outcome,
            resolved_run_dir,
            repo_root=repo_root,
            scene_timeout_seconds=scene_timeout_seconds,
            verbose=verbose,
        )
    build_articulation_review_receipt(
        resolved_run_dir,
        decisions,
        reviewer=reviewer,
    )
    return resume_articulation_workflow(
        resolved_run_dir,
        repo_root=repo_root,
        scene_timeout_seconds=scene_timeout_seconds,
        verbose=verbose,
    )


def revise_articulation_graph_workflow(
    run_dir: str | Path,
    revision_patch_path: str | Path,
    *,
    repo_root: Path,
    scene_timeout_seconds: float = 60.0,
    verbose: bool = False,
) -> ArticulationFinalizationResult:
    """Supersede one reviewed graph and reopen its exact-digest human gate."""

    root = _verified_existing_run_dir(run_dir)
    patch_path = Path(revision_patch_path).expanduser().resolve()
    if not patch_path.is_relative_to(root):
        raise ValueError("Articulation graph revision patch must stay inside the run")
    request, state = _load_verified_request_and_state(root)
    if state.mode != "batch" or request.execution_context is None:
        raise ValueError(
            "Graph supersession requires an embedded batch Articulation run"
        )
    launcher = _load_launcher_config(request)
    if launcher.execution_mode != ARTICULATION_EXECUTION_SKILL_ROUTED:
        raise ValueError("Graph supersession requires skill-routed execution")
    if launcher.embedded_run_state is None:
        raise ValueError("Graph supersession lacks the outer asset run state")
    config = _articulation_resume_config(
        request,
        launcher,
        repo_root=repo_root,
        scene_timeout_seconds=scene_timeout_seconds,
        verbose=verbose,
    )
    outer_state_path = config.embedded_run_state
    if outer_state_path is None:  # defensive parity with the frozen launcher check
        raise ValueError("Graph supersession lacks the outer asset run state")

    def transition() -> ArticulationFinalizationResult:
        with _skill_routed_run_lock(root, domain="articulation"):
            _reject_unsafe_run_links(root)
            patch = EmbeddedArticulationGraphRevisionPatch.model_validate_json(
                patch_path.read_bytes()
            )
            frozen_request, frozen_state = _load_verified_request_and_state(root)
            frozen_config = _load_articulation_agent_launcher(
                root,
                request=frozen_request,
            )
            if frozen_config.embedded_run_state != outer_state_path:
                raise ValueError(
                    "Outer asset run state changed while waiting for graph revision"
                )
            preparation = _load_embedded_preparation(frozen_config)
            runtime_client = None
            if preparation is None:
                if frozen_config.joint_config is None:
                    raise ValueError(
                        "Provider-backed Articulation requires joint_config"
                    )
                runtime_client = JointAgentLocalClient(
                    frozen_config.joint_config,
                    session_id=frozen_config.joint_session_id,
                    verbose=frozen_config.verbose,
                )
            if frozen_state.embedded_canonical_graph is None:
                raise ValueError(
                    "Graph supersession lacks its parent canonical graph binding"
                )
            validate_articulation_graph_revision_request(
                outer_state_path,
                candidates_path=frozen_state.embedded_canonical_graph.path,
                human_decisions=patch.human_decisions,
            )
            revised_state = apply_embedded_articulation_graph_revision(
                root,
                patch,
                runtime=_embedded_articulation_runtime(
                    frozen_request,
                    frozen_config,
                    runtime_client,
                    preparation,
                ),
            )
            if (
                revised_state.embedded_graph_revision is None
                or revised_state.embedded_canonical_graph is None
            ):
                raise ValueError(
                    "Graph supersession did not persist its revision bindings"
                )
            record_articulation_graph_revision(
                outer_state_path,
                revision_receipt_path=revised_state.embedded_graph_revision.path,
                candidates_path=revised_state.embedded_canonical_graph.path,
                actor="asset-coordinator",
            )
            result = write_articulation_workflow_summary(
                revised_state,
                output_dir=root,
            )
            _write_articulation_observation_if_needed(root, result)
            TraceWriter(root).write(
                "canonical_graph_superseded",
                phase="articulation_review",
                summary=(
                    "Immutable parent/child graph revision is persisted and the "
                    "exact revised digest requires complete human review."
                ),
                artifacts=[
                    path
                    for path in (
                        result.embedded_graph_revision_path,
                        result.embedded_canonical_graph_path,
                        result.embedded_outer_review_path,
                        result.embedded_coordinator_decision_path,
                    )
                    if path
                ],
                data={"status": result.status},
            )
            return result

    return run_asset_coordinator_transition(
        outer_state_path,
        transition=transition,
    )


def _build_request(
    config: ArticulationRunConfig,
) -> tuple[ArticulationWorkflowRequest, ArticulationRunConfig]:
    if config.execution_mode not in {
        ARTICULATION_EXECUTION_SKILL_ROUTED,
        ARTICULATION_EXECUTION_FIXED,
    }:
        raise ValueError("Unsupported Articulation execution mode")
    if (
        config.embedded_run_state is not None
        and config.execution_mode == ARTICULATION_EXECUTION_FIXED
    ):
        raise ValueError(
            "Embedded Articulation execution cannot use the fixed compatibility mode"
        )
    if (
        config.embedded_preparation is not None
        and config.execution_mode == ARTICULATION_EXECUTION_FIXED
    ):
        raise ValueError(
            "Provider-neutral Articulation preparation requires skill-routed execution"
        )
    _ensure_articulation_agent_configs_safe(config)
    source_asset = config.source_asset.expanduser().resolve()
    if not source_asset.is_file():
        raise FileNotFoundError(
            f"Articulation source asset is not a file: {source_asset}"
        )
    output_dir_candidate = config.output_dir.expanduser()
    _reject_unsafe_run_links(output_dir_candidate, allow_missing=True)
    output_dir = _lexical_absolute_path(output_dir_candidate)
    embedded_preparation_candidate = (
        config.embedded_preparation.expanduser()
        if config.embedded_preparation is not None
        else None
    )
    embedded_preparation = (
        embedded_preparation_candidate.resolve()
        if embedded_preparation_candidate is not None
        else None
    )
    if embedded_preparation is not None:
        assert embedded_preparation_candidate is not None
        if config.joint_config is not None:
            raise ValueError(
                "joint_config and embedded_preparation are mutually exclusive"
            )
        if (
            embedded_preparation_candidate.is_symlink()
            or not embedded_preparation.is_file()
        ):
            raise FileNotFoundError(
                "Embedded Articulation preparation is not a regular file: "
                f"{embedded_preparation}"
            )
        preparation_model = (
            EmbeddedArticulationPreparation
            if config.embedded_run_state is not None
            else StandaloneArticulationPreparation
        )
        preparation_model.model_validate_json(embedded_preparation.read_bytes())
    joint_config = (
        config.joint_config.expanduser().resolve()
        if config.joint_config is not None
        else None
    )
    if joint_config is None and embedded_preparation is None:
        raise ValueError(
            "Joint Agent config is required without provider-neutral preparation"
        )
    if joint_config is not None and not joint_config.is_file():
        raise FileNotFoundError(f"Joint Agent config is not a file: {joint_config}")
    normalized = ArticulationRunConfig(
        repo_root=config.repo_root.expanduser().resolve(),
        source_asset=source_asset,
        output_dir=output_dir,
        joint_config=joint_config,
        intent=config.intent,
        review_policy=config.review_policy,
        allowed_motion_types=config.allowed_motion_types,
        expected_candidate_count=config.expected_candidate_count,
        max_candidate_count=config.max_candidate_count,
        joint_session_id=config.joint_session_id,
        embedded_run_state=(
            config.embedded_run_state.expanduser().resolve()
            if config.embedded_run_state is not None
            else None
        ),
        embedded_preparation=embedded_preparation,
        verbose=config.verbose,
        scene_timeout_seconds=config.scene_timeout_seconds,
        execution_mode=config.execution_mode,
        runner=config.runner,
        model=config.model,
        model_reasoning_effort=config.model_reasoning_effort,
        codex_base_url=config.codex_base_url,
        codex_sandbox_mode=config.codex_sandbox_mode,
        codex_config=config.codex_config,
        claude_config=config.claude_config,
        claude_permission_mode=config.claude_permission_mode,
        claude_max_turns=config.claude_max_turns,
        claude_execution_mode=config.claude_execution_mode,
        child_timeout_seconds=config.child_timeout_seconds,
        agent_cwd=(
            config.agent_cwd.expanduser().resolve()
            if config.agent_cwd is not None
            else None
        ),
        dry_run=config.dry_run,
    )
    _validate_articulation_agent_runtime(normalized)
    metadata: dict[str, Any] = {
        ARTICULATION_LAUNCHER_METADATA_KEY: {
            "schema_version": ARTICULATION_LAUNCHER_SCHEMA_VERSION,
            "joint_config": str(joint_config) if joint_config is not None else None,
            "joint_session_id": config.joint_session_id,
            "embedded_run_state": (
                str(normalized.embedded_run_state)
                if normalized.embedded_run_state is not None
                else None
            ),
            "embedded_preparation": (
                str(embedded_preparation) if embedded_preparation is not None else None
            ),
            "embedded_preparation_sha256": (
                hashlib.sha256(embedded_preparation.read_bytes()).hexdigest()
                if embedded_preparation is not None
                else None
            ),
            "execution_mode": config.execution_mode,
            "requested_review_policy": config.review_policy,
            "runner": config.runner,
            "model": config.model,
            "model_reasoning_effort": config.model_reasoning_effort,
            "codex_base_url": config.codex_base_url,
            "codex_sandbox_mode": config.codex_sandbox_mode,
            "codex_config": config.codex_config,
            "claude_config": config.claude_config,
            "claude_permission_mode": config.claude_permission_mode,
            "claude_max_turns": config.claude_max_turns,
            "claude_execution_mode": config.claude_execution_mode,
            "child_timeout_seconds": config.child_timeout_seconds,
            "agent_cwd": str(config.agent_cwd) if config.agent_cwd else None,
        }
    }
    if config.execution_mode == ARTICULATION_EXECUTION_SKILL_ROUTED:
        metadata[SKILL_ROUTED_DECISION_METADATA_KEY] = True
        metadata[_REQUESTED_REVIEW_POLICY_METADATA_KEY] = config.review_policy
    if normalized.embedded_run_state is not None:
        execution_context = build_embedded_domain_execution_context(
            normalized.embedded_run_state,
            domain="articulation",
            input_asset=source_asset,
            output_dir=output_dir,
        )
        metadata = metadata_with_domain_execution_context(
            metadata,
            execution_context,
        )
    elif embedded_preparation is not None:
        metadata = metadata_with_domain_execution_context(
            metadata,
            DomainExecutionContext(
                domain="articulation",
                mode="standalone",
                reasoning_loop_owner="domain_child_agent",
            ),
        )
    request = ArticulationWorkflowRequest(
        source_asset=str(source_asset),
        output_dir=output_dir,
        intent=config.intent,
        # This field controls the operator gate. Skill-routed execution reviews
        # every candidate; a requested `none` policy is separately persisted and
        # publishes the validated agent decision as the exact review receipt.
        review_policy=(
            "all"
            if config.execution_mode == ARTICULATION_EXECUTION_SKILL_ROUTED
            else config.review_policy
        ),
        allowed_motion_types=config.allowed_motion_types,
        expected_candidate_count=config.expected_candidate_count,
        max_candidate_count=config.max_candidate_count,
        metadata=metadata,
    )
    return request, normalized


def _run_request(
    request: ArticulationWorkflowRequest,
    config: ArticulationRunConfig,
) -> ArticulationFinalizationResult:
    run_dir = config.output_dir
    _reject_unsafe_run_links(run_dir, allow_missing=True)
    run_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    run_dir.chmod(0o700)
    _reject_unsafe_run_links(run_dir)

    if config.execution_mode == ARTICULATION_EXECUTION_FIXED:
        return _execute_articulation_controller(
            request,
            config,
            controller=_fixed_articulation_controller,
        )

    with _skill_routed_run_lock(run_dir, domain="articulation"):
        outcome = _run_skill_routed_articulation_request_locked(request, config)
    return _complete_articulation_child_launch(outcome, request, config)


def _run_skill_routed_articulation_request_locked(
    request: ArticulationWorkflowRequest,
    config: ArticulationRunConfig,
) -> ArticulationFinalizationResult | _ArticulationChildLaunchRequired:
    """Execute one Articulation parent while its sibling lease is held."""

    run_dir = config.output_dir
    checkpoint_path = run_dir / "checkpoint.json"
    state: ArticulationRunState | None = None
    if checkpoint_path.is_file():
        stored_request, state = _load_verified_request_and_state(run_dir)
        if _request_invocation_payload(stored_request) != (
            _request_invocation_payload(request)
        ):
            raise ValueError(
                "Articulation invocation differs from the checkpointed request"
            )
    prepared_request_path = run_dir / "articulation_agent_request.json"
    if prepared_request_path.is_file():
        prepared_request = ArticulationWorkflowRequest.model_validate_json(
            prepared_request_path.read_text(encoding="utf-8")
        )
        if _request_invocation_payload(prepared_request) != (
            _request_invocation_payload(request)
        ):
            raise ValueError(
                "Articulation invocation differs from the prepared child request"
            )
    _write_articulation_agent_launcher(run_dir, config, request=request)
    if (
        request.execution_context is not None
        and request.execution_context.mode == "embedded"
    ):
        result = _execute_articulation_controller(
            request,
            config,
            controller=partial(
                _embedded_articulation_controller,
                config=config,
            ),
        )
        _write_articulation_observation_if_needed(run_dir, result)
        return result

    if (
        request.execution_context is not None
        and request.execution_context.mode == "standalone"
    ):
        result = _execute_articulation_controller(
            request,
            config,
            controller=_standalone_articulation_controller,
        )
        _write_articulation_observation_if_needed(run_dir, result)
        current = ArticulationRunState.model_validate_json(
            (run_dir / "checkpoint.json").read_bytes()
        )
        if current.phase == "awaiting_decision":
            return _ARTICULATION_CHILD_LAUNCH_REQUIRED
        return result

    if state is not None:
        # Existence selects the deterministic parent verifier, not a trust
        # shortcut: advance_articulation_state_machine reloads and validates the
        # ledger against the current checkpoint before any review or authoring.
        ledger_exists = (run_dir / "articulation_decision_ledger.json").is_file()
        if (
            state.phase in TERMINAL_ARTICULATION_PHASES
            or state.review_receipt is not None
            or (state.phase == "needs_review" and ledger_exists)
        ):
            return _execute_articulation_controller(
                request,
                config,
                controller=advance_articulation_state_machine,
            )
    return _ARTICULATION_CHILD_LAUNCH_REQUIRED


def _complete_articulation_child_launch(
    outcome: ArticulationFinalizationResult | _ArticulationChildLaunchRequired,
    request: ArticulationWorkflowRequest,
    config: ArticulationRunConfig,
) -> ArticulationFinalizationResult:
    if not isinstance(outcome, _ArticulationChildLaunchRequired):
        return outcome
    with _skill_routed_child_session_lock(
        config.output_dir,
        domain="articulation",
    ):
        # A second parent may have waited while the first child completed. Recheck
        # the durable state under the mutation lease before launching another
        # reasoning session.
        with _skill_routed_run_lock(config.output_dir, domain="articulation"):
            current = _run_skill_routed_articulation_request_locked(request, config)
        if not isinstance(current, _ArticulationChildLaunchRequired):
            return current
        return _launch_articulation_child(request, config)


def _fixed_articulation_controller(
    request: ArticulationWorkflowRequest,
    *,
    mode: str,
    **kwargs: Any,
) -> ArticulationFinalizationResult:
    # Adapt the common controller signature; batch compatibility has one mode.
    del mode
    return run_batch_articulation_workflow(request, **kwargs)


def _load_embedded_preparation(
    config: ArticulationRunConfig,
) -> EmbeddedArticulationPreparation | None:
    path = config.embedded_preparation
    if path is None:
        return None
    expanded = path.expanduser()
    resolved = expanded.resolve()
    if expanded.is_symlink() or not resolved.is_file():
        raise ValueError(
            f"Embedded Articulation preparation is not a regular file: {resolved}"
        )
    return EmbeddedArticulationPreparation.model_validate_json(resolved.read_bytes())


def _load_standalone_preparation(
    config: ArticulationRunConfig,
) -> StandaloneArticulationPreparation | None:
    path = config.embedded_preparation
    if path is None:
        return None
    if config.embedded_run_state is not None:
        raise ValueError(
            "Standalone Articulation preparation cannot carry embedded_run_state"
        )
    expanded = path.expanduser()
    resolved = expanded.resolve()
    if expanded.is_symlink() or not resolved.is_file():
        raise ValueError(
            f"Standalone Articulation preparation is not a regular file: {resolved}"
        )
    return StandaloneArticulationPreparation.model_validate_json(resolved.read_bytes())


def _standalone_articulation_controller(
    request: ArticulationWorkflowRequest,
    *,
    mode: str,
    preparation: StandaloneArticulationPreparation,
    **kwargs: Any,
) -> ArticulationFinalizationResult:
    """Prepare/revalidate standalone custody without invoking Joint inference."""

    del kwargs
    context = request.execution_context
    if context is None or context.mode != "standalone":
        raise ValueError("Standalone Articulation requires standalone context")
    return run_standalone_articulation_workflow(
        request,
        mode=cast(Any, mode),
        preparation=preparation,
    )


def _embedded_articulation_runtime(
    request: ArticulationWorkflowRequest,
    config: ArticulationRunConfig,
    client: JointAgentLocalClient | None,
    preparation: EmbeddedArticulationPreparation | None = None,
) -> EmbeddedArticulationRuntime:
    """Bind provider-backed or provider-neutral preparation to one identity."""

    if config.embedded_run_state is None:
        raise ValueError("Embedded articulation requires the outer run state path")
    capabilities = EmbeddedArticulationCapabilityLimits(
        canonical_output_evidence_required=True
    )
    implementation_digests = {
        name: canonical_json_digest(manifest)
        for name, manifest in EMBEDDED_ARTICULATION_IMPLEMENTATION_MANIFEST.items()
    }
    embedded_implementation = implementation_digests["embedded_articulation_controller"]
    joint_implementation = implementation_digests["joint_agent_graph_adapter"]
    coordinator_implementation = implementation_digests["asset_coordinator"]
    capability_digests = {
        "embedded_articulation_graph_authoring": canonical_json_digest(capabilities)
    }
    if preparation is not None:
        implementation_digests["deterministic_evidence_provider"] = (
            preparation.evidence_provider.implementation_digest
        )
        if preparation.proposal is not None:
            implementation_digests["optional_proposal_provider"] = (
                preparation.proposal.producer.implementation_digest
            )
        configuration_digests = {
            "embedded_preparation_configuration": preparation.configuration_sha256
        }
        if preparation.proposal is not None and preparation.proposal.capability:
            capability = preparation.proposal.capability
            configuration_digests["optional_proposal_provider_configuration"] = (
                capability.provider_configuration_sha256
            )
            capability_digests["optional_proposal_provider"] = canonical_json_digest(
                capability
            )
        evidence_provider = preparation.evidence_provider
        proposal_provider = (
            preparation.proposal.producer if preparation.proposal is not None else None
        )
    else:
        if client is None:
            raise ValueError("Provider-backed Articulation requires a Joint client")
        configuration_digests = {
            "joint_runtime_configuration": client.configuration_sha256(request)
        }
        evidence_provider = ProducerIdentity(
            producer_id="joint-agent-inspection",
            role="evidence_provider",
            implementation="JointAgentLocalClient.infer+Scene",
            implementation_digest=joint_implementation,
        )
        proposal_provider = ProducerIdentity(
            producer_id="joint-agent-proposal",
            role="proposal_provider",
            implementation="JointAgentLocalClient.infer",
            implementation_digest=joint_implementation,
        )
    identity = build_embedded_domain_decision_identity(
        config.embedded_run_state,
        domain="articulation",
        input_asset=request.source_asset,
        output_dir=request.output_dir,
        capability_digests=capability_digests,
        implementation_digests=implementation_digests,
        configuration_digests=configuration_digests,
    )
    return EmbeddedArticulationRuntime(
        identity=identity,
        evidence_provider=evidence_provider,
        proposal_provider=proposal_provider,
        outer_coordinator=ProducerIdentity(
            producer_id="asset-coordinator",
            role="outer_coordinator",
            implementation="content-workflow-asset",
            implementation_digest=coordinator_implementation,
        ),
        executor=ProducerIdentity(
            producer_id="joint-agent-graph-authorer",
            role="executor",
            implementation="embedded-articulation-graph-adapter",
            implementation_digest=embedded_implementation,
        ),
        capabilities=capabilities,
    )


def _embedded_articulation_controller(
    request: ArticulationWorkflowRequest,
    *,
    config: ArticulationRunConfig,
    mode: str,
    client: ArticulationAuthoringClient,
    provider_client: JointAgentLocalClient | None = None,
    scene_evidence_collector: Any = None,
    preparation: EmbeddedArticulationPreparation | None = None,
    **kwargs: Any,
) -> ArticulationFinalizationResult:
    if config.embedded_run_state is None:
        raise ValueError("Embedded articulation requires embedded_run_state")
    embedded_run_state = config.embedded_run_state
    if preparation is None and provider_client is None:
        raise ValueError(
            "Provider-backed Articulation requires a JointAgentLocalClient"
        )
    if preparation is None and provider_client is not client:
        raise ValueError(
            "Provider-backed Articulation must use one client for inference and "
            "authoring"
        )
    if preparation is not None and provider_client is not None:
        raise ValueError(
            "Provider-neutral Articulation forbids a JointAgentLocalClient"
        )
    runtime = _embedded_articulation_runtime(
        request,
        config,
        provider_client,
        preparation,
    )

    def load_human_acceptance(
        _run_dir: Path,
        canonical_graph: Any,
    ) -> Any:
        return load_embedded_articulation_human_acceptance(
            embedded_run_state,
            canonical_graph=canonical_graph.path,
        )

    return run_embedded_articulation_workflow(
        request,
        mode=cast(Any, mode),
        client=client,
        runtime=runtime,
        human_acceptance_loader=load_human_acceptance,
        preparation=preparation,
        scene_evidence_collector=scene_evidence_collector,
        **kwargs,
    )


def _execute_articulation_controller(
    request: ArticulationWorkflowRequest,
    config: ArticulationRunConfig,
    *,
    controller: Any,
) -> ArticulationFinalizationResult:
    _reject_unsafe_run_links(config.output_dir, allow_missing=True)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    _reject_unsafe_run_links(config.output_dir)
    context = request.execution_context
    standalone_preparation = (
        _load_standalone_preparation(config)
        if context is not None and context.mode == "standalone"
        else None
    )
    preparation = (
        _load_embedded_preparation(config)
        if context is not None and context.mode == "embedded"
        else None
    )
    provider_client: JointAgentLocalClient | None
    if standalone_preparation is not None or preparation is not None:
        client: ArticulationAuthoringClient = JointAgentGraphAuthoringClient()
        provider_client = None
        collector = None
    else:
        if config.joint_config is None:
            raise ValueError("Provider-backed Articulation requires joint_config")
        provider_client = JointAgentLocalClient(
            config.joint_config,
            session_id=config.joint_session_id,
            verbose=config.verbose,
        )
        client = provider_client
        collector = _build_live_evidence_collector(config.scene_timeout_seconds)
    controller_kwargs = {
        "mode": "batch",
        "client": client,
        "scene_evidence_collector": collector,
    }
    selected_preparation = standalone_preparation or preparation
    if selected_preparation is not None:
        controller_kwargs["preparation"] = selected_preparation
    if config.embedded_run_state is not None:
        controller_kwargs["provider_client"] = provider_client
    try:
        return cast(
            ArticulationFinalizationResult,
            controller(
                request,
                **controller_kwargs,
            ),
        )
    except ArticulationWorkflowError:
        if not _has_verified_failed_checkpoint(config.output_dir):
            raise
        return cast(
            ArticulationFinalizationResult,
            controller(
                request,
                **controller_kwargs,
            ),
        )


def _launch_articulation_child(
    request: ArticulationWorkflowRequest,
    config: ArticulationRunConfig,
) -> ArticulationFinalizationResult:
    run_dir = config.output_dir
    request_path = atomic_write_json(
        run_dir / "articulation_agent_request.json",
        request.model_dump(mode="json"),
    )
    capability_inventory, domain_policy_bounds = (
        _prepare_articulation_child_launch_contract(request)
    )
    prompt = _build_articulation_agent_prompt(request)
    prompt_path = run_dir / "prompts" / "articulation_skill_routed.md"
    prompt_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    prompt_path.parent.chmod(0o700)
    prompt_path.write_text(prompt, encoding="utf-8")
    trace_writer = TraceWriter(run_dir)
    trace_writer.write(
        "workflow_started",
        phase="articulation_skill_routed",
        summary="Prepared one long-running Articulation child task.",
        artifacts=[
            str(request_path),
            capability_inventory.path,
            domain_policy_bounds.path,
            str(prompt_path),
        ],
    )
    if config.dry_run:
        trace_writer.write(
            "workflow_dry_run",
            phase="articulation_skill_routed",
            summary="Dry run stopped before launching the Articulation child.",
        )
        build_trace(run_dir)
        return ArticulationFinalizationResult(
            success=False,
            status="conditional",
            mode="batch",
            output_dir=str(run_dir),
            checkpoint_path=str(run_dir / "checkpoint.json"),
            workflow_progress_path=str(run_dir / "workflow_progress.json"),
            final_summary_path=str(run_dir / "final_summary.json"),
            message="Dry run prepared the skill-routed Articulation task.",
        )

    child_config = _articulation_child_runtime_config(
        config,
        capability_inventory=capability_inventory,
        domain_policy_bounds=domain_policy_bounds,
    )
    parent_usd_cli_capability = start_parent_usd_cli_capability(
        config=child_config,
        run_dir=run_dir,
        workflow="articulation.author",
        session_workflow="articulation-evidence",
        input_roots=(config.source_asset,),
        initial_scene=config.source_asset,
        timeout_seconds=max(config.scene_timeout_seconds, 900.0),
    )
    try:
        child_config = replace(
            child_config,
            parent_usd_cli_session_identity=parent_usd_cli_capability.identity_path,
            parent_usd_cli_session_identity_sha256=(
                parent_usd_cli_capability.identity_sha256
            ),
        )
        prompt += parent_usd_cli_prompt_contract(parent_usd_cli_capability)
        prompt_path.write_text(prompt, encoding="utf-8")
        return _run_articulation_child_with_parent_capability(
            request=request,
            config=config,
            child_config=child_config,
            prompt=prompt,
            trace_writer=trace_writer,
        )
    finally:
        stop_parent_usd_cli_capability(parent_usd_cli_capability)


def _run_articulation_child_with_parent_capability(
    *,
    request: ArticulationWorkflowRequest,
    config: ArticulationRunConfig,
    child_config: _ArticulationChildRuntimeConfig,
    prompt: str,
    trace_writer: TraceWriter,
) -> ArticulationFinalizationResult:
    """Run the child and parent verification while its renderer lease is live."""

    run_dir = config.output_dir
    child_output_path = run_dir / "raw" / "articulation_child_output.jsonl"
    child_final_path = run_dir / "raw" / "articulation_child_final.json"
    child_output_path.parent.mkdir(parents=True, exist_ok=True)
    returncode = run_child_agent(
        config=child_config,
        prompt=prompt,
        run_dir=run_dir,
        child_output_path=child_output_path,
        child_final_path=child_final_path,
        bridge_artifact_prefix="articulation_skill_routed",
    )
    with _skill_routed_run_lock(run_dir, domain="articulation"):
        return _finalize_articulation_child_launch(
            request,
            config,
            returncode=returncode,
            child_output_path=child_output_path,
            child_final_path=child_final_path,
            trace_writer=trace_writer,
        )


def _finalize_articulation_child_launch(
    request: ArticulationWorkflowRequest,
    config: ArticulationRunConfig,
    *,
    returncode: int,
    child_output_path: Path,
    child_final_path: Path,
    trace_writer: TraceWriter,
) -> ArticulationFinalizationResult:
    """Finalize child output while the parent again owns the run lease."""

    run_dir = config.output_dir
    trace_writer.write(
        "child_finished",
        phase="articulation_skill_routed",
        summary="Articulation child exited after proposal review.",
        artifacts=[str(child_output_path), str(child_final_path)],
        data={"returncode": returncode},
    )
    if returncode != 0:
        build_trace(run_dir)
        raise RuntimeError(f"Articulation child agent exited with code {returncode}")
    child_status = _articulation_child_status(child_final_path)
    if child_status not in {
        "awaiting_decision",
        "awaiting_post_review",
        "needs_review",
        "completed",
        "not_articulated",
        "conditional",
        "cancelled",
        "failed",
    }:
        trace_writer.write(
            "workflow_failed",
            phase="articulation_skill_routed",
            summary="Articulation child exited without a reportable checkpoint.",
            artifacts=[str(child_final_path), str(run_dir / "checkpoint.json")],
            data={"child_status": child_status},
        )
        build_trace(run_dir)
        raise RuntimeError(
            "Articulation child exited without a reportable checkpoint "
            f"(status={child_status!r})"
        )
    if (
        request.execution_context is not None
        and request.execution_context.mode == "standalone"
    ):
        state = ArticulationRunState.model_validate_json(
            (run_dir / "checkpoint.json").read_bytes()
        )
        authoring = None
        if state.authoring_result is not None:
            from content_agent_workflows.articulation import ArticulationAuthoringResult

            authoring = ArticulationAuthoringResult.model_validate_json(
                Path(state.authoring_result.path).read_bytes()
            )
        result = write_articulation_workflow_summary(
            state,
            output_dir=run_dir,
            authoring=authoring,
        )
    else:
        result = _execute_articulation_controller(
            request,
            config,
            controller=advance_articulation_state_machine,
        )
    final_path = run_dir / "final_summary.json"
    trace_writer.write(
        "workflow_finished",
        phase="articulation_skill_routed",
        summary="Articulation child decision is bound to durable evidence.",
        artifacts=[str(final_path)],
        data={"status": result.status},
    )
    build_trace(run_dir)
    return result


def _articulation_child_status(path: Path) -> str | None:
    """Read the child's terminal envelope before parent re-verification."""

    try:
        raw_text = _read_articulation_agent_file(
            path,
            label="Articulation child final summary",
            max_bytes=MAX_ARTICULATION_CHILD_FINAL_BYTES,
        ).decode("utf-8")
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    payload = _extract_last_articulation_child_envelope(raw_text)
    status = payload.get("status") if isinstance(payload, dict) else None
    return status if isinstance(status, str) else None


def _extract_last_articulation_child_envelope(text: str) -> dict[str, Any] | None:
    """Recover the final JSON object from a child response with optional prose.

    Child runners can persist their final answer verbatim.  Accepting the last
    complete JSON object lets the parent tolerate a Markdown fence or a short
    prose preamble while still requiring a valid object-shaped envelope.
    """

    decoder = json.JSONDecoder()
    last_envelope: dict[str, Any] | None = None
    index = text.find("{")
    while index != -1:
        try:
            candidate, end = decoder.raw_decode(text, index)
        except (json.JSONDecodeError, RecursionError):
            index = text.find("{", index + 1)
            continue
        if isinstance(candidate, dict):
            last_envelope = candidate
        index = text.find("{", end)
    return last_envelope


def prepare_articulation_agent_step(
    run_dir: str | Path,
) -> ArticulationFinalizationResult:
    """Internal focused command: inspect, infer, collect evidence, then pause."""

    root = Path(run_dir).expanduser().resolve()
    request = ArticulationWorkflowRequest.model_validate_json(
        (root / "articulation_agent_request.json").read_text(encoding="utf-8")
    )
    config = _load_articulation_agent_launcher(root, request=request)
    context = request.execution_context
    controller: Any
    if context is not None and context.mode == "embedded":
        controller = partial(_embedded_articulation_controller, config=config)
    elif context is not None and context.mode == "standalone":
        controller = _standalone_articulation_controller
    else:
        controller = advance_articulation_state_machine
    result = _execute_articulation_controller(
        request,
        config,
        controller=controller,
    )
    _write_articulation_observation_if_needed(root, result)
    TraceWriter(root).write(
        "step_finished",
        phase="articulation_proposal",
        summary="Inspection, proposal, and Scene evidence are checkpointed.",
        artifacts=[
            path
            for path in (
                result.candidate_document_path,
                result.scene_evidence_path,
            )
            if path
        ],
        data={"status": result.status},
    )
    return result


def apply_articulation_agent_step(
    run_dir: str | Path,
    decision_patch_path: str | Path,
) -> ArticulationFinalizationResult:
    """Internal focused command: bind a decision, then honor the review gate."""

    from content_agent_workflows.articulation import (
        ArticulationDecisionPatch,
        apply_articulation_decision_patch,
    )

    root = Path(run_dir).expanduser().resolve()
    patch_path = Path(decision_patch_path).expanduser().resolve()
    if not patch_path.is_relative_to(root):
        raise ValueError("Articulation decision patch must stay inside the run")
    request = _load_request(root)
    config = _load_articulation_agent_launcher(root, request=request)
    if (
        request.execution_context is not None
        and request.execution_context.mode == "standalone"
    ):
        with _skill_routed_run_lock(root, domain="articulation"):
            _reject_unsafe_run_links(root)
            request = _load_request(root)
            context = request.execution_context
            if context is None or context.mode != "standalone":
                raise ValueError(
                    "Standalone Articulation context changed while waiting for apply"
                )
            patch = StandaloneArticulationDecisionPatch.model_validate_json(
                patch_path.read_bytes()
            )
            state = apply_standalone_articulation_decision_patch(
                root,
                patch,
                client=JointAgentGraphAuthoringClient(),
            )
            authoring = None
            if state.authoring_result is not None:
                from content_agent_workflows.articulation import (
                    ArticulationAuthoringResult,
                )

                authoring = ArticulationAuthoringResult.model_validate_json(
                    Path(state.authoring_result.path).read_bytes()
                )
            result = write_articulation_workflow_summary(
                state,
                output_dir=root,
                authoring=authoring,
            )
            _write_articulation_observation_if_needed(root, result)
            TraceWriter(root).write(
                "decision_bound",
                phase="articulation_review",
                summary=(
                    "Child decision passed outer validation; accepted-only graph "
                    "authoring and exact saved-stage readback are checkpointed."
                ),
                artifacts=[
                    path
                    for path in (
                        result.standalone_decision_patch_path,
                        result.standalone_decision_ledger_path,
                        result.standalone_canonical_graph_path,
                        result.standalone_readback_path,
                    )
                    if path
                ],
                data={"status": result.status},
            )
            return result
    if (
        request.execution_context is not None
        and request.execution_context.mode == "embedded"
    ):
        with _skill_routed_run_lock(root, domain="articulation"):
            _reject_unsafe_run_links(root)
            request = _load_request(root)
            if request.execution_context is None:
                raise ValueError(
                    "Articulation execution context changed while waiting for apply"
                )
            config = _load_articulation_agent_launcher(root, request=request)
            embedded_patch = EmbeddedArticulationDecisionPatch.model_validate_json(
                patch_path.read_bytes()
            )
            preparation = _load_embedded_preparation(config)
            runtime_client = None
            if preparation is None:
                if config.joint_config is None:
                    raise ValueError(
                        "Provider-backed Articulation requires joint_config"
                    )
                runtime_client = JointAgentLocalClient(
                    config.joint_config,
                    session_id=config.joint_session_id,
                    verbose=config.verbose,
                )
            runtime = _embedded_articulation_runtime(
                request,
                config,
                runtime_client,
                preparation,
            )
            apply_embedded_articulation_decision_patch(
                root,
                embedded_patch,
                runtime=runtime,
            )
            result = _execute_articulation_controller(
                request,
                config,
                controller=partial(_embedded_articulation_controller, config=config),
            )
            TraceWriter(root).write(
                "decision_bound",
                phase="articulation_review",
                summary=(
                    "Exact outer accept/reject/revise graph review is persisted "
                    "before mutation or optional human review."
                ),
                artifacts=[
                    str(root / "canonical_articulation_graph.json"),
                    result.embedded_outer_review_path or "",
                    result.embedded_coordinator_decision_path or "",
                ],
                data={"status": result.status},
            )
            return result
    standalone_patch = ArticulationDecisionPatch.model_validate_json(
        patch_path.read_text(encoding="utf-8")
    )
    apply_articulation_decision_patch(root, standalone_patch)
    launcher = _load_launcher_config(request)
    if launcher.requested_review_policy == "none":
        decisions = {
            item.candidate_id: item.decision for item in standalone_patch.decisions
        }
        build_articulation_review_receipt(
            root,
            decisions,
            reviewer="skill-routed-articulation-child",
        )
    result = _execute_articulation_controller(
        request,
        config,
        controller=advance_articulation_state_machine,
    )
    TraceWriter(root).write(
        "decision_bound",
        phase="articulation_review",
        summary="Agent decision patch passed deterministic scope validation.",
        artifacts=[
            str(root / "articulation_decision_patch.json"),
            str(root / "articulation_decision_ledger.json"),
        ],
        data={"status": result.status},
    )
    return result


def finalize_articulation_agent_step(
    run_dir: str | Path,
    post_review_patch_path: str | Path,
) -> ArticulationFinalizationResult:
    """Internal focused command: bind outer post-readback review and receipt."""

    root = Path(run_dir).expanduser().resolve()
    patch_path = Path(post_review_patch_path).expanduser().resolve()
    if not patch_path.is_relative_to(root):
        raise ValueError("Articulation post-review patch must stay inside the run")
    request = _load_request(root)
    if request.execution_context is None:
        raise ValueError("Post-readback review requires embedded or standalone custody")
    with _skill_routed_run_lock(root, domain="articulation"):
        _reject_unsafe_run_links(root)
        request = _load_request(root)
        if request.execution_context is None:
            raise ValueError(
                "Articulation execution context changed while waiting for post review"
            )
        if request.execution_context.mode == "standalone":
            patch = StandaloneArticulationPostReviewPatch.model_validate_json(
                patch_path.read_bytes()
            )
            state = apply_standalone_articulation_post_review(root, patch)
            authoring = None
            if state.authoring_result is not None:
                from content_agent_workflows.articulation import (
                    ArticulationAuthoringResult,
                )

                authoring = ArticulationAuthoringResult.model_validate_json(
                    Path(state.authoring_result.path).read_bytes()
                )
            result = write_articulation_workflow_summary(
                state,
                output_dir=root,
                authoring=authoring,
            )
            TraceWriter(root).write(
                "post_review_bound",
                phase="articulation_post_review",
                summary=(
                    "Independent standalone review is bound before terminal "
                    "publication."
                ),
                artifacts=[
                    path
                    for path in (
                        result.standalone_post_review_path,
                        result.standalone_terminal_receipt_path,
                    )
                    if path
                ],
                data={"status": result.status},
            )
            return result
        config = _load_articulation_agent_launcher(root, request=request)
        preparation = _load_embedded_preparation(config)
        runtime_client = None
        if preparation is None:
            if config.joint_config is None:
                raise ValueError("Provider-backed Articulation requires joint_config")
            runtime_client = JointAgentLocalClient(
                config.joint_config,
                session_id=config.joint_session_id,
                verbose=config.verbose,
            )
        patch = EmbeddedArticulationPostReviewPatch.model_validate_json(
            patch_path.read_bytes()
        )
        state = apply_embedded_articulation_post_review(
            root,
            patch,
            runtime=_embedded_articulation_runtime(
                request,
                config,
                runtime_client,
                preparation,
            ),
        )
        authoring = None
        if state.authoring_result is not None:
            from content_agent_workflows.articulation import ArticulationAuthoringResult

            authoring = ArticulationAuthoringResult.model_validate_json(
                Path(state.authoring_result.path).read_bytes()
            )
        result = write_articulation_workflow_summary(
            state,
            output_dir=root,
            authoring=authoring,
        )
        TraceWriter(root).write(
            "post_review_bound",
            phase="articulation_post_review",
            summary="Outer post-readback review is persisted before stage completion.",
            artifacts=[
                path
                for path in (
                    result.embedded_coordinator_review_path,
                    result.embedded_decision_receipt_path,
                )
                if path
            ],
            data={"status": result.status},
        )
        return result


def bind_articulation_output_evidence_step(
    run_dir: str | Path,
    canonical_visual_envelope_path: str | Path,
) -> ArticulationFinalizationResult:
    """Public focused leaf: bind exact canonical post-authoring OVRTX evidence."""

    root = Path(run_dir).expanduser().resolve()
    envelope_path = Path(canonical_visual_envelope_path).expanduser().resolve()
    request = _load_request(root)
    if request.execution_context is None:
        raise ValueError(
            "Canonical output binding requires embedded or standalone custody"
        )
    with _skill_routed_run_lock(root, domain="articulation"):
        _reject_unsafe_run_links(root)
        request = _load_request(root)
        if request.execution_context is None:
            raise ValueError(
                "Articulation execution context changed while waiting for output binding"
            )
        context = request.execution_context
        assert context is not None
        state = (
            bind_standalone_articulation_output_evidence(root, envelope_path)
            if context.mode == "standalone"
            else bind_embedded_articulation_output_evidence(root, envelope_path)
        )
        authoring = None
        if state.authoring_result is not None:
            from content_agent_workflows.articulation import ArticulationAuthoringResult

            authoring = ArticulationAuthoringResult.model_validate_json(
                Path(state.authoring_result.path).read_bytes()
            )
        result = write_articulation_workflow_summary(
            state,
            output_dir=root,
            authoring=authoring,
        )
        TraceWriter(root).write(
            "canonical_output_evidence_bound",
            phase="articulation_post_review",
            summary=("Exact post-authoring OVRTX evidence is bound for outer review."),
            artifacts=[
                path
                for path in (
                    result.embedded_output_evidence_path,
                    result.standalone_output_evidence_path,
                    str(envelope_path),
                )
                if path
            ],
            data={"status": result.status},
        )
        return result


def _write_articulation_observation_if_needed(
    run_dir: Path,
    result: ArticulationFinalizationResult,
) -> Path | None:
    if result.status not in {"awaiting_decision", "needs_review", "conditional"}:
        return None
    if result.standalone_preparation_path:
        state = ArticulationRunState.model_validate_json(
            (run_dir / "checkpoint.json").read_bytes()
        )
        if state.phase != "awaiting_decision":
            return None
        observation = build_standalone_articulation_observation(run_dir)
        return cast(
            Path,
            atomic_write_json(
                run_dir / "articulation_agent_observation.json",
                observation.model_dump(mode="json"),
            ),
        )
    if result.embedded_evidence_path:
        evidence = EmbeddedDomainEvidence.model_validate_json(
            Path(result.embedded_evidence_path).read_bytes()
        )
        proposal = (
            EmbeddedDomainProposal.model_validate_json(
                Path(result.embedded_proposal_path).read_bytes()
            )
            if result.embedded_proposal_path is not None
            else None
        )
        source_members = next(
            (
                item
                for item in evidence.records
                if item.evidence_id == "joint-source-member-inspection"
            ),
            None,
        )
        authoritative_owners = next(
            (
                item
                for item in evidence.records
                if item.evidence_id == "joint-authoritative-owner-inspection"
            ),
            None,
        )
        capabilities = next(
            (
                item
                for item in evidence.records
                if item.evidence_id == "joint-authoring-capabilities"
            ),
            None,
        )
        preparation_record = next(
            (
                item
                for item in evidence.records
                if item.evidence_id == "articulation-preparation"
            ),
            None,
        )
        source_member_values = (
            source_members.facts.get("source_member_prims")
            if source_members is not None
            else None
        )
        authoritative_owner_values = (
            authoritative_owners.facts.get("authoritative_owner_prims")
            if authoritative_owners is not None
            else None
        )
        checkpoint = ArticulationRunState.model_validate_json(
            (run_dir / "checkpoint.json").read_bytes()
        )
        canonical_graph = (
            EmbeddedArticulationCanonicalGraph.model_validate_json(
                Path(result.embedded_canonical_graph_path).read_bytes()
            )
            if result.embedded_canonical_graph_path is not None
            else None
        )
        graph_revision_inputs: dict[str, object] | None = None
        if checkpoint.embedded_human_decision is not None:
            human_decision = EmbeddedHumanDecision.model_validate_json(
                Path(checkpoint.embedded_human_decision.path).read_bytes()
            )
            if human_decision.disposition == "revise":
                frozen_request = _load_request(run_dir)
                launcher = _load_launcher_config(frozen_request)
                if launcher.embedded_run_state is None:
                    raise ValueError(
                        "Human-revised embedded Articulation lacks outer run state"
                    )
                outer_run = load_verified_run(launcher.embedded_run_state)
                outer_stage = outer_run.stages["articulation"]
                if (
                    outer_stage.review_candidates is None
                    or outer_stage.review_decisions is None
                    or checkpoint.embedded_canonical_graph is None
                    or outer_stage.review_candidates.path
                    != checkpoint.embedded_canonical_graph.path
                    or outer_stage.review_candidates.sha256
                    != checkpoint.embedded_canonical_graph.sha256
                ):
                    raise ValueError(
                        "Human-revised Articulation observation lacks exact outer "
                        "review bindings"
                    )
                graph_revision_inputs = {
                    "human_decision": checkpoint.embedded_human_decision.model_dump(
                        mode="json"
                    ),
                    "human_decisions": outer_stage.review_decisions.model_dump(
                        mode="json"
                    ),
                }
        return atomic_write_json(
            run_dir / "articulation_agent_observation.json",
            {
                "schema_version": (
                    "content-workflow-cli.embedded-articulation-observation.v4"
                ),
                "identity_digest": canonical_json_digest(evidence.identity),
                "evidence_digest": artifact_reference(evidence).sha256,
                "proposal_digest": (
                    artifact_reference(proposal).sha256
                    if proposal is not None
                    else None
                ),
                "evidence_path": result.embedded_evidence_path,
                "proposal_path": result.embedded_proposal_path,
                "proposal_status": (
                    preparation_record.facts.get("proposal_status")
                    if preparation_record is not None
                    else ("available" if proposal is not None else "not_evaluated")
                ),
                "provider_candidate_document_path": result.candidate_document_path,
                "scene_evidence_path": result.scene_evidence_path,
                "checkpoint_revision": checkpoint.revision,
                "canonical_graph_path": result.embedded_canonical_graph_path,
                "canonical_graph_sha256": (
                    checkpoint.embedded_canonical_graph.sha256
                    if checkpoint.embedded_canonical_graph is not None
                    else None
                ),
                "canonical_graph_digest": (
                    canonical_json_digest(canonical_graph)
                    if canonical_graph is not None
                    else None
                ),
                "graph_revision_path": result.embedded_graph_revision_path,
                "graph_revision_patch_path": str(run_dir / "graph_revision_patch.json"),
                "graph_revision_inputs": graph_revision_inputs,
                "source_member_prims": (
                    [str(item) for item in source_member_values]
                    if isinstance(source_member_values, Sequence)
                    and not isinstance(source_member_values, str | bytes)
                    else []
                ),
                "authoritative_owner_prims": (
                    [str(item) for item in authoritative_owner_values]
                    if isinstance(authoritative_owner_values, Sequence)
                    and not isinstance(authoritative_owner_values, str | bytes)
                    else []
                ),
                "capability_limits": (dict(capabilities.facts) if capabilities else {}),
                "canonical_graph_schema_version": (
                    "content-agent-workflows.embedded-articulation-graph.v1"
                ),
                "decision_patch_schema_version": (
                    "content-agent-workflows.embedded-articulation-decision-patch.v3"
                ),
                "graph_revision_patch_schema_version": (
                    "content-agent-workflows.embedded-articulation-graph-revision-"
                    "patch.v1"
                ),
                "outer_review_dispositions": ["accept", "reject", "revise"],
                "human_review_statuses": ["not_requested", "human_required"],
                "authority": (
                    "Provider proposals and raw predictions are evidence only. "
                    "The outer organizer must explicitly review the exact graph. "
                    "Human review is enforced only when frozen policy requires it."
                ),
            },
        )
    ledger_path = run_dir / "articulation_decision_ledger.json"
    if ledger_path.is_file():
        return None
    state = ArticulationRunState.model_validate_json(
        (run_dir / "checkpoint.json").read_text(encoding="utf-8")
    )
    observation = build_articulation_step_observation(state, output_dir=run_dir)
    return atomic_write_json(
        run_dir / "articulation_agent_observation.json",
        observation.model_dump(mode="json"),
    )


def _write_articulation_agent_launcher(
    run_dir: Path,
    config: ArticulationRunConfig,
    *,
    request: ArticulationWorkflowRequest,
) -> Path:
    _validate_articulation_agent_runtime(config)
    _ensure_articulation_agent_configs_safe(config)
    payload = asdict(config)
    for field_name in (
        "repo_root",
        "source_asset",
        "output_dir",
        "joint_config",
        "embedded_run_state",
        "embedded_preparation",
        "agent_cwd",
    ):
        value = payload[field_name]
        if value is not None:
            payload[field_name] = str(value)
    path = run_dir / "articulation_agent_launcher.json"
    atomic_write_json(
        path,
        {
            "schema_version": ARTICULATION_AGENT_LAUNCHER_SCHEMA_VERSION,
            "config": payload,
        },
    )
    launcher_bytes = _read_articulation_agent_file(
        path,
        label="Articulation agent launcher",
        max_bytes=MAX_ARTICULATION_AGENT_LAUNCHER_BYTES,
    )
    atomic_write_json(
        _articulation_agent_launcher_policy_path(run_dir),
        {
            "schema_version": ARTICULATION_AGENT_LAUNCHER_POLICY_SCHEMA_VERSION,
            "run_dir": str(run_dir.expanduser().resolve()),
            "launcher_sha256": hashlib.sha256(launcher_bytes).hexdigest(),
            "request_sha256": hashlib.sha256(
                _stable_articulation_invocation_bytes(request)
            ).hexdigest(),
        },
    )
    return path


def _articulation_agent_launcher_policy_path(run_dir: Path) -> Path:
    resolved_run_dir = run_dir.expanduser().resolve()
    return (
        resolved_run_dir.parent
        / f".{resolved_run_dir.name}.articulation-agent-launcher-policy.json"
    )


def _stable_articulation_invocation_bytes(
    request: ArticulationWorkflowRequest,
) -> bytes:
    return (
        json.dumps(
            _request_invocation_payload(request),
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
        )
        + "\n"
    ).encode("utf-8")


def _read_articulation_agent_file(
    path: Path,
    *,
    label: str,
    max_bytes: int,
) -> bytes:
    try:
        with open_regular_file_no_follow(path) as (stream, metadata):
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise ValueError(f"{label} must be a single-link regular file: {path}")
            if metadata.st_size > max_bytes:
                raise ValueError(f"{label} exceeds the {max_bytes}-byte limit: {path}")
            contents = stream.read(max_bytes + 1)
            final_metadata = os.fstat(stream.fileno())
            if final_metadata.st_nlink != 1:
                raise ValueError(
                    f"{label} must remain a single-link regular file: {path}"
                )
    except (ArtifactPathError, OSError) as exc:
        raise ValueError(f"Unable to read {label} safely at {path}: {exc}") from exc
    if len(contents) > max_bytes:
        raise ValueError(f"{label} exceeds the {max_bytes}-byte limit: {path}")
    return contents


def _load_articulation_agent_launcher(
    run_dir: Path,
    *,
    request: ArticulationWorkflowRequest,
) -> ArticulationRunConfig:
    resolved_run_dir = run_dir.expanduser().resolve()
    policy_bytes = _read_articulation_agent_file(
        _articulation_agent_launcher_policy_path(resolved_run_dir),
        label="Articulation agent launcher policy",
        max_bytes=MAX_ARTICULATION_AGENT_LAUNCHER_POLICY_BYTES,
    )
    try:
        policy = json.loads(policy_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid Articulation agent launcher policy JSON") from exc
    if not isinstance(policy, dict) or set(policy) != {
        "schema_version",
        "run_dir",
        "launcher_sha256",
        "request_sha256",
    }:
        raise ValueError("Articulation agent launcher policy has invalid fields")
    if (
        policy.get("schema_version")
        != ARTICULATION_AGENT_LAUNCHER_POLICY_SCHEMA_VERSION
    ):
        raise ValueError("Unsupported Articulation agent launcher policy schema")
    if Path(str(policy.get("run_dir"))).expanduser().resolve() != resolved_run_dir:
        raise ValueError("Articulation agent launcher policy run_dir mismatch")
    for digest_field in ("launcher_sha256", "request_sha256"):
        digest = policy.get(digest_field)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(
                f"Articulation agent launcher policy {digest_field} is invalid"
            )
    actual_request_sha256 = hashlib.sha256(
        _stable_articulation_invocation_bytes(request)
    ).hexdigest()
    if actual_request_sha256 != policy["request_sha256"]:
        raise ValueError(
            "Articulation request digest does not match the parent-owned policy"
        )

    launcher_bytes = _read_articulation_agent_file(
        resolved_run_dir / "articulation_agent_launcher.json",
        label="Articulation agent launcher",
        max_bytes=MAX_ARTICULATION_AGENT_LAUNCHER_BYTES,
    )
    if hashlib.sha256(launcher_bytes).hexdigest() != policy["launcher_sha256"]:
        raise ValueError(
            "Articulation agent launcher digest does not match the parent-owned policy"
        )
    try:
        payload = json.loads(launcher_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid Articulation agent launcher JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "config"}:
        raise ValueError("Articulation agent launcher has invalid fields")
    launcher_schema = payload.get("schema_version")
    if launcher_schema not in {
        ARTICULATION_AGENT_LAUNCHER_SCHEMA_VERSION,
        ARTICULATION_LEGACY_AGENT_LAUNCHER_SCHEMA_VERSION,
    }:
        raise ValueError("Unsupported Articulation agent launcher schema")
    raw = payload.get("config")
    if not isinstance(raw, dict):
        raise ValueError("Articulation agent launcher config must be an object")
    expected_config_fields = {field.name for field in fields(ArticulationRunConfig)}
    if launcher_schema == ARTICULATION_LEGACY_AGENT_LAUNCHER_SCHEMA_VERSION:
        expected_config_fields.remove("embedded_preparation")
    if set(raw) != expected_config_fields:
        missing = sorted(expected_config_fields - set(raw))
        unknown = sorted(set(raw) - expected_config_fields)
        raise ValueError(
            "Articulation agent launcher config fields do not match the schema: "
            f"missing={missing}, unknown={unknown}"
        )
    raw.setdefault("embedded_preparation", None)
    for field_name in (
        "repo_root",
        "source_asset",
        "output_dir",
        "joint_config",
        "embedded_run_state",
        "embedded_preparation",
        "agent_cwd",
    ):
        raw_value = raw.get(field_name)
        if raw_value is not None:
            if not isinstance(raw_value, str):
                raise ValueError(
                    "Articulation agent launcher config field "
                    f"{field_name!r} must be a string or null"
                )
            raw[field_name] = Path(raw_value)
    if raw.get("allowed_motion_types") is not None:
        if not isinstance(raw["allowed_motion_types"], list):
            raise ValueError(
                "Articulation agent launcher config field 'allowed_motion_types' "
                "must be an array"
            )
        raw["allowed_motion_types"] = tuple(raw["allowed_motion_types"])
    try:
        result = ArticulationRunConfig(**raw)
        _validate_articulation_agent_runtime(result)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid Articulation agent launcher config") from exc
    _ensure_articulation_agent_configs_safe(result)
    if result.output_dir.expanduser().resolve() != resolved_run_dir:
        raise ValueError("Articulation agent launcher output_dir mismatch")
    if (
        result.source_asset.expanduser().resolve()
        != Path(request.source_asset).expanduser().resolve()
    ):
        raise ValueError("Articulation agent launcher source_asset mismatch")
    return result


def _validate_articulation_agent_runtime(config: ArticulationRunConfig) -> None:
    required_strings = {
        "intent",
        "review_policy",
        "execution_mode",
        "runner",
        "codex_sandbox_mode",
        "claude_permission_mode",
        "claude_execution_mode",
    }
    optional_strings = {
        "joint_session_id",
        "model",
        "model_reasoning_effort",
        "codex_base_url",
    }
    for field_name in required_strings:
        value = getattr(config, field_name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                "Articulation agent launcher field "
                f"{field_name!r} must be a non-empty string"
            )
    for field_name in optional_strings:
        value = getattr(config, field_name)
        if value is not None and not isinstance(value, str):
            raise ValueError(
                "Articulation agent launcher field "
                f"{field_name!r} must be a string or null"
            )

    for field_name in ("repo_root", "source_asset", "output_dir"):
        if not isinstance(getattr(config, field_name), Path):
            raise ValueError(
                f"Articulation agent launcher field {field_name!r} must be a path"
            )
    for field_name in (
        "joint_config",
        "embedded_run_state",
        "embedded_preparation",
        "agent_cwd",
    ):
        value = getattr(config, field_name)
        if value is not None and not isinstance(value, Path):
            raise ValueError(
                "Articulation agent launcher field "
                f"{field_name!r} must be a path or null"
            )

    if (
        not isinstance(config.allowed_motion_types, tuple)
        or not config.allowed_motion_types
        or len(set(config.allowed_motion_types)) != len(config.allowed_motion_types)
        or any(
            motion_type not in {"revolute", "prismatic"}
            for motion_type in config.allowed_motion_types
        )
    ):
        raise ValueError(
            "Articulation agent launcher field 'allowed_motion_types' must be a "
            "non-empty unique tuple of supported motion types"
        )
    if (
        isinstance(config.max_candidate_count, bool)
        or not isinstance(config.max_candidate_count, int)
        or not 1 <= config.max_candidate_count <= 256
    ):
        raise ValueError(
            "Articulation agent launcher field 'max_candidate_count' must be an "
            "integer from 1 through 256"
        )
    if config.expected_candidate_count is not None and (
        isinstance(config.expected_candidate_count, bool)
        or not isinstance(config.expected_candidate_count, int)
        or not 0 <= config.expected_candidate_count <= config.max_candidate_count
    ):
        raise ValueError(
            "Articulation agent launcher field 'expected_candidate_count' must be "
            "null or an integer from 0 through max_candidate_count"
        )

    for field_name in ("scene_timeout_seconds", "child_timeout_seconds"):
        value = getattr(config, field_name)
        minimum_inclusive = field_name == "child_timeout_seconds"
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
            or (float(value) < 0 if minimum_inclusive else float(value) <= 0)
        ):
            qualifier = "non-negative" if minimum_inclusive else "positive"
            raise ValueError(
                f"Articulation agent launcher field {field_name!r} must be {qualifier}"
            )
    for field_name in ("verbose", "dry_run"):
        if not isinstance(getattr(config, field_name), bool):
            raise ValueError(
                f"Articulation agent launcher field {field_name!r} must be a boolean"
            )
    for field_name in ("codex_config", "claude_config"):
        value = getattr(config, field_name)
        if value is not None and not isinstance(value, dict):
            raise ValueError(
                "Articulation agent launcher field "
                f"{field_name!r} must be an object or null"
            )
    if config.claude_max_turns is not None and (
        isinstance(config.claude_max_turns, bool)
        or not isinstance(config.claude_max_turns, int)
        or config.claude_max_turns <= 0
    ):
        raise ValueError(
            "Articulation agent launcher field 'claude_max_turns' must be a "
            "positive integer or null"
        )

    if config.review_policy not in {"uncertain", "all", "none"}:
        raise ValueError(
            "Articulation agent launcher uses an unsupported review policy"
        )
    if config.execution_mode not in {
        ARTICULATION_EXECUTION_SKILL_ROUTED,
        ARTICULATION_EXECUTION_FIXED,
    }:
        raise ValueError(
            "Articulation agent launcher uses an unsupported execution mode"
        )
    if config.embedded_preparation is not None and config.joint_config is not None:
        raise ValueError(
            "Articulation agent launcher cannot select joint_config and "
            "embedded_preparation together"
        )
    if config.embedded_preparation is None and config.joint_config is None:
        raise ValueError(
            "Articulation agent launcher requires joint_config or embedded_preparation"
        )
    if config.runner not in {RUNNER_CODEX, RUNNER_CLAUDE}:
        raise ValueError("Articulation agent launcher uses an unsupported child runner")
    if config.codex_sandbox_mode != CODEX_SANDBOX_WORKSPACE_WRITE:
        raise ValueError(
            "Articulation agent launcher uses an unsupported Codex sandbox mode"
        )
    if config.claude_permission_mode not in {
        "default",
        "acceptEdits",
        "bypassPermissions",
        "plan",
    }:
        raise ValueError(
            "Articulation agent launcher uses an unsupported Claude permission mode"
        )
    if config.claude_execution_mode not in {
        CLAUDE_EXECUTION_SDK,
        CLAUDE_EXECUTION_CLI,
    }:
        raise ValueError(
            "Articulation agent launcher uses an unsupported Claude execution mode"
        )


def _ensure_articulation_agent_configs_safe(config: ArticulationRunConfig) -> None:
    """Reject inline credentials before request or launcher persistence."""

    from world_understanding.utils.credentials import ensure_no_inline_secrets

    ensure_no_inline_secrets(
        {
            "codex_config": config.codex_config,
            "claude_config": config.claude_config,
        },
        context="Articulation agent configuration",
        path_context=True,
    )


def _articulation_child_runtime_config(
    config: ArticulationRunConfig,
    *,
    capability_inventory: ChildLaunchArtifactIdentity,
    domain_policy_bounds: ChildLaunchArtifactIdentity,
) -> _ArticulationChildRuntimeConfig:
    return _ArticulationChildRuntimeConfig(
        repo_root=config.repo_root,
        usd_path=config.source_asset,
        runner=config.runner,
        model=config.model,
        model_reasoning_effort=config.model_reasoning_effort,
        codex_base_url=config.codex_base_url,
        codex_sandbox_mode=config.codex_sandbox_mode,
        codex_config=config.codex_config,
        claude_config=config.claude_config,
        claude_permission_mode=config.claude_permission_mode,
        claude_max_turns=config.claude_max_turns,
        claude_execution_mode=config.claude_execution_mode,
        child_timeout_seconds=config.child_timeout_seconds,
        agent_cwd=config.agent_cwd,
        reference_images=[],
        reference_files=None,
        child_capability_inventory=capability_inventory,
        child_domain_policy_bounds=domain_policy_bounds,
        child_forbidden_environment_names=(
            _articulation_forbidden_environment_names(config.joint_config)
        ),
    )


def _prepare_articulation_child_launch_contract(
    request: ArticulationWorkflowRequest,
) -> tuple[ChildLaunchArtifactIdentity, ChildLaunchArtifactIdentity]:
    """Freeze the wrapper-owned capabilities and semantic policy bounds."""

    run_dir = request.output_dir.expanduser().resolve()
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    raw_dir.chmod(0o700)
    request_identity = child_launch_artifact_identity(
        run_dir,
        run_dir / "articulation_agent_request.json",
    )
    capability_path = atomic_write_json(
        raw_dir / "articulation_child_capability_inventory.json",
        {
            "schema_version": "content-workflow-cli.articulation-child-capabilities.v1",
            "workflow": "articulation.author",
            "request_identity": request_identity.model_dump(mode="json"),
            "selection_owner": "articulation-domain-wrapper",
            "capabilities": [
                {
                    "capability_id": "articulation.agent-prepare",
                    "execution_owner": "articulation-domain-wrapper",
                    "transport": "typed-artifact",
                },
                {
                    "capability_id": "articulation.agent-apply",
                    "execution_owner": "articulation-domain-wrapper",
                    "transport": "typed-artifact",
                },
            ],
        },
    )
    policy_path = atomic_write_json(
        raw_dir / "articulation_child_domain_policy.json",
        {
            "schema_version": "content-workflow-cli.articulation-child-policy.v1",
            "workflow": "articulation.author",
            "request_identity": request_identity.model_dump(mode="json"),
            "allowed_motion_types": list(request.allowed_motion_types),
            "expected_candidate_count": request.expected_candidate_count,
            "max_candidate_count": request.max_candidate_count,
            "requested_review_policy": request.metadata.get(
                _REQUESTED_REVIEW_POLICY_METADATA_KEY,
                request.review_policy,
            ),
            "child_authority": {
                "domain_client": False,
                "domain_credentials": False,
                "domain_network": False,
                "renderer_network": False,
                "semantic_operation_selection": False,
            },
        },
    )
    return (
        child_launch_artifact_identity(run_dir, capability_path),
        child_launch_artifact_identity(run_dir, policy_path),
    )


def _articulation_forbidden_environment_names(
    joint_config: Path | None,
) -> tuple[str, ...]:
    """Collect only named domain credentials, never their values."""

    if joint_config is None:
        return ()
    import yaml

    names: set[str] = set()
    try:
        payload = yaml.safe_load(joint_config.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(
            f"Unable to load Joint Agent config {joint_config}: {exc}"
        ) from exc

    def visit(value: object, key: str | None = None) -> None:
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                visit(child_value, str(child_key))
            return
        if isinstance(value, list):
            for child in value:
                visit(child, key)
            return
        if not isinstance(value, str):
            return
        if key is not None and key.lower().endswith("_env"):
            candidate = value.strip()
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", candidate):
                names.add(candidate)
        for match in re.finditer(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", value):
            names.add(match.group(1))

    visit(payload)
    return tuple(sorted(name for name in names if name))


def _build_articulation_agent_prompt(request: ArticulationWorkflowRequest) -> str:
    run_dir = request.output_dir.expanduser().resolve()
    task = {
        "schema_version": "content-agents.skill-routed-task.v1",
        "workflow": "articulation.author",
        "run_dir": str(run_dir),
        "required_skills": [
            "usd-cli",
            "content-articulation-inspection",
            "content-articulation-proposal",
            "content-articulation-review",
            "content-articulation-authoring",
            "content-workflow-articulation",
        ],
    }
    if (
        request.execution_context is not None
        and request.execution_context.mode == "standalone"
    ):
        return f"""You are the single long-running child for a standalone Articulation workflow.

Load and follow every skill in this compact task:
{json.dumps(task, indent=2)}

Inspect `final_summary.json` when it already exists. If the run is already at
`awaiting_post_review`, stop and return that status for the outer coordinator.
Otherwise, first run:
`content-workflow-cli articulation _agent-prepare --run-dir {run_dir}`

Read `articulation_agent_observation.json` and every exact evidence record it
binds. Write the complete typed `StandaloneArticulationDecisionPatch` to the
observation's `decision_patch_path`. Bind the exact state, request,
preparation, source, dependency, and configuration identities. Include every
candidate disposition and, for accepted candidates, the complete owner/body
membership, joint type, axis, frame policy, limits, evidence requirements, and
review policy in `canonical_graph`. Only observation-bound evidence is semantic
authority. Exact observation-bound hierarchy, transforms, bounds, and authored
properties may establish geometric facts when the patch names the bound evidence
IDs and states the derivation. Set the top-level `evidence_requirements` to a
non-empty list and set non-empty `evidence_ids` on every `candidate_decisions`
entry and every entry in `canonical_graph.groups`,
`canonical_graph.memberships`, `canonical_graph.joints`, and
`canonical_graph.rigid_link_operations`. Use only exact IDs from the
observation's `evidence_records`, and verify every required list before the
first `_agent-apply` call. Do not use web search, repository fixtures,
dataset labels, an unbound reference asset, unbound geometry, labels alone, or
visual plausibility to invent an axis, endpoint, frame, or limit. Emit a typed
`revise` decision when a required fact is not established by the bound evidence.
Then run:
`content-workflow-cli articulation _agent-apply --run-dir {run_dir} --decision-patch <path>`

If the observation contains a candidate-bound issue packet, gather only the
listed focused evidence and write its one immutable replacement patch. Respect
the single-attempt cap; if the focused evidence still does not establish every
required fact, preserve the issue codes and let the outer workflow publish the
non-success cap receipt without source mutation. When `_agent-apply` reaches
`awaiting_post_review`, stop and return that status for the outer coordinator.
The child must not render, bind output evidence, inspect output images, perform
post-authoring review, or finalize the run. The outer coordinator alone owns
canonical OVRTX evidence, post-review, and terminal finalization.

Never mutate USD directly, invoke JointAgentLocalClient or
`joint_agent.api.pipeline`, write a ledger or canonical graph, bypass output
evidence, or write outside the run directory. Return a concise JSON object with
`status` and `final_summary_path`, with JSON only: no Markdown fence or prose,
and without private reasoning.
"""
    return f"""You are the single long-running child for a skill-routed Articulation workflow.

Load and follow every skill in this compact task:
{json.dumps(task, indent=2)}

First run:
`content-workflow-cli articulation _agent-prepare --run-dir {run_dir}`

That command can take many minutes. Wait for it to exit before inspecting its
artifacts or returning. Never report `blocked` while it or another workflow
command is still running.

Read `articulation_agent_observation.json`, the exact candidate document, and
Scene evidence it binds. Write one ordered `ArticulationDecisionPatch`
entry for every candidate. Accept, reject, or provide a complete
`replacement_candidate` edit without changing candidate identity, endpoints,
motion type, or native readiness. Cite only observation-bound evidence paths.
Write the patch to `decision_patch_path`, then run:
`content-workflow-cli articulation _agent-apply --run-dir {run_dir} --decision-patch <path>`

Never author directly, bypass exact outer review or selected human-review policy,
edit the source asset, or
write outside the run directory. Deterministic code owns accepted-only
authoring, saved-stage readback, publication, and review receipts. Return a
concise JSON object with `status` and `final_summary_path`, with JSON only: no
Markdown fence or prose, and without private reasoning.
"""


def _build_live_evidence_collector(timeout_seconds: float) -> Any:
    from content_agent_workflows.articulation.scene_evidence import (
        LiveUsdCliArticulationEvidenceCollector,
    )

    return LiveUsdCliArticulationEvidenceCollector(
        evidence_policy_id=ARTICULATION_SCENE_POLICY_ID,
        timeout=timeout_seconds,
    )


def _load_request(run_dir: str | Path) -> ArticulationWorkflowRequest:
    request, _ = _load_verified_request_and_state(run_dir)
    return request


def _load_verified_request_and_state(
    run_dir: str | Path,
) -> tuple[ArticulationWorkflowRequest, ArticulationRunState]:
    resolved_run_dir = _verified_existing_run_dir(run_dir)
    request_path = resolved_run_dir / "request.json"
    if not request_path.is_file():
        raise FileNotFoundError(
            f"Articulation run request does not exist: {request_path}"
        )
    checkpoint_path = resolved_run_dir / "checkpoint.json"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Articulation run checkpoint does not exist: {checkpoint_path}"
        )
    try:
        state = ArticulationRunState.model_validate_json(checkpoint_path.read_bytes())
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"Invalid articulation run checkpoint at {checkpoint_path}: {exc}"
        ) from exc
    bound_request_path = Path(state.request.path).expanduser().resolve()
    if bound_request_path != request_path:
        raise ValueError(
            "Articulation checkpoint request path does not match the run directory."
        )
    request_bytes = request_path.read_bytes()
    observed_sha256 = hashlib.sha256(request_bytes).hexdigest()
    if observed_sha256 != state.request.sha256:
        raise ValueError(
            "Articulation request digest differs from the checkpointed run."
        )
    try:
        request = ArticulationWorkflowRequest.model_validate_json(request_bytes)
    except ValueError as exc:
        raise ValueError(
            f"Invalid articulation run request at {request_path}: {exc}"
        ) from exc
    if request.output_dir.expanduser().resolve() != resolved_run_dir:
        raise ValueError(
            "Articulation request output_dir does not match the requested run directory."
        )
    return request, state


def _has_verified_failed_checkpoint(run_dir: str | Path) -> bool:
    """Return whether a canonical terminal-failure replay is available."""

    try:
        _, state = _load_verified_request_and_state(run_dir)
    except (FileNotFoundError, OSError, RuntimeError, ValueError):
        return False
    return state.phase == "failed"


def _verified_existing_run_dir(run_dir: str | Path) -> Path:
    candidate = Path(run_dir).expanduser()
    _reject_unsafe_run_links(candidate)
    return _lexical_absolute_path(candidate)


def _scene_evidence_is_pending(
    request: ArticulationWorkflowRequest,
) -> bool:
    """Return whether this invocation can still need a live Scene call."""

    output_dir = request.output_dir.expanduser().resolve()
    checkpoint_path = output_dir / "checkpoint.json"
    scene_evidence_path = output_dir / "scene_evidence" / "manifest.json"
    invocation_payload = _request_invocation_payload(request)
    if not checkpoint_path.exists():
        request_path = output_dir / "request.json"
        if request_path.exists():
            try:
                stored_request = ArticulationWorkflowRequest.model_validate_json(
                    request_path.read_bytes()
                )
            except (OSError, ValueError) as exc:
                raise ValueError(
                    f"Invalid uncheckpointed articulation request at "
                    f"{request_path}: {exc}"
                ) from exc
            if _request_invocation_payload(stored_request) != invocation_payload:
                raise ValueError(
                    "Existing uncheckpointed articulation request conflicts with "
                    "this invocation."
                )
        return True

    stored_request, state = _load_verified_request_and_state(output_dir)
    if _request_invocation_payload(stored_request) != invocation_payload:
        raise ValueError(
            "Articulation request differs from the checkpointed invocation."
        )
    return bool(
        state.phase not in TERMINAL_ARTICULATION_PHASES
        and state.scene_evidence is None
        and not scene_evidence_path.is_file()
    )


def _request_invocation_payload(
    request: ArticulationWorkflowRequest,
) -> dict[str, Any]:
    """Return request identity without the workflow-injected evidence marker."""

    metadata = dict(request.metadata)
    if metadata.get(_ARTICULATION_SCENE_EVIDENCE_REQUIRED_METADATA_KEY) is True:
        metadata.pop(_ARTICULATION_SCENE_EVIDENCE_REQUIRED_METADATA_KEY)
    return request.model_copy(update={"metadata": metadata}).model_dump(mode="json")


def _load_launcher_config(
    request: ArticulationWorkflowRequest,
) -> _StoredLauncherConfig:
    raw = request.metadata.get(ARTICULATION_LAUNCHER_METADATA_KEY)
    if not isinstance(raw, dict):
        raise ValueError(
            "Articulation request was not created by content-workflow-cli."
        )
    launcher_schema = raw.get("schema_version")
    if launcher_schema not in {
        ARTICULATION_LAUNCHER_SCHEMA_VERSION,
        ARTICULATION_LEGACY_LAUNCHER_SCHEMA_VERSION,
    }:
        raise ValueError("Unsupported content-workflow-cli articulation metadata.")
    joint_config_value = raw.get("joint_config")
    joint_session_id = raw.get("joint_session_id")
    embedded_run_state_value = raw.get("embedded_run_state")
    embedded_preparation_value = raw.get("embedded_preparation")
    if joint_config_value is not None and (
        not isinstance(joint_config_value, str) or not joint_config_value
    ):
        raise ValueError("Articulation launcher joint_config is malformed.")
    joint_config = (
        Path(joint_config_value).expanduser().resolve()
        if isinstance(joint_config_value, str)
        else None
    )
    if joint_config is not None and not joint_config.is_file():
        raise FileNotFoundError(f"Joint Agent config is not a file: {joint_config}")
    if joint_session_id is not None and not isinstance(joint_session_id, str):
        raise ValueError("Articulation launcher joint_session_id must be a string.")
    if embedded_run_state_value is not None and not isinstance(
        embedded_run_state_value, str
    ):
        raise ValueError("Articulation launcher embedded_run_state must be a string.")
    if embedded_preparation_value is not None and not isinstance(
        embedded_preparation_value, str
    ):
        raise ValueError("Articulation launcher embedded_preparation must be a string.")
    embedded_preparation_candidate = (
        Path(embedded_preparation_value).expanduser()
        if embedded_preparation_value
        else None
    )
    embedded_preparation = (
        embedded_preparation_candidate.resolve()
        if embedded_preparation_candidate is not None
        else None
    )
    if embedded_preparation is not None:
        if joint_config is not None:
            raise ValueError(
                "Articulation metadata cannot select joint_config and "
                "embedded_preparation together."
            )
        expected_preparation_sha256 = raw.get("embedded_preparation_sha256")
        if (
            embedded_preparation_candidate is None
            or embedded_preparation_candidate.is_symlink()
            or not embedded_preparation.is_file()
            or not isinstance(expected_preparation_sha256, str)
            or hashlib.sha256(embedded_preparation.read_bytes()).hexdigest()
            != expected_preparation_sha256
        ):
            raise ValueError("Embedded Articulation preparation is missing or changed.")
    elif launcher_schema == ARTICULATION_LEGACY_LAUNCHER_SCHEMA_VERSION:
        if joint_config is None:
            raise ValueError("Legacy Articulation metadata lacks joint_config.")
    elif joint_config is None:
        raise ValueError(
            "Articulation metadata requires joint_config or embedded_preparation."
        )
    raw_runner = raw.get("runner", RUNNER_CODEX)
    if not isinstance(raw_runner, str) or raw_runner not in {
        RUNNER_CODEX,
        RUNNER_CLAUDE,
    }:
        raise ValueError("Articulation launcher runner is unsupported.")
    raw_child_timeout = raw.get("child_timeout_seconds", 1800.0)
    if (
        isinstance(raw_child_timeout, bool)
        or not isinstance(raw_child_timeout, int | float)
        or not math.isfinite(float(raw_child_timeout))
        or float(raw_child_timeout) < 0
    ):
        raise ValueError(
            "Articulation launcher child_timeout_seconds must be non-negative."
        )
    execution_mode = str(raw.get("execution_mode", ARTICULATION_EXECUTION_FIXED))
    if execution_mode not in {
        ARTICULATION_EXECUTION_SKILL_ROUTED,
        ARTICULATION_EXECUTION_FIXED,
    }:
        raise ValueError("Articulation launcher execution_mode is unsupported.")
    requested_review_policy = str(
        raw.get("requested_review_policy", request.review_policy)
    )
    if requested_review_policy not in {"uncertain", "all", "none"}:
        raise ValueError("Articulation requested review policy is unsupported.")
    agent_cwd_value = raw.get("agent_cwd")
    if agent_cwd_value is not None and not isinstance(agent_cwd_value, str):
        raise ValueError("Articulation launcher agent_cwd must be a string.")
    return _StoredLauncherConfig(
        joint_config=joint_config,
        joint_session_id=joint_session_id,
        embedded_run_state=(
            Path(embedded_run_state_value).expanduser().resolve()
            if embedded_run_state_value
            else None
        ),
        embedded_preparation=embedded_preparation,
        execution_mode=execution_mode,
        requested_review_policy=cast(
            ArticulationReviewPolicy,
            requested_review_policy,
        ),
        runner=raw_runner,
        model=cast(str | None, raw.get("model")),
        model_reasoning_effort=cast(
            str | None,
            raw.get("model_reasoning_effort"),
        ),
        codex_base_url=cast(str | None, raw.get("codex_base_url")),
        codex_sandbox_mode=str(
            raw.get("codex_sandbox_mode", CODEX_SANDBOX_WORKSPACE_WRITE)
        ),
        codex_config=cast(dict[str, object] | None, raw.get("codex_config")),
        claude_config=cast(dict[str, object] | None, raw.get("claude_config")),
        claude_permission_mode=str(raw.get("claude_permission_mode", "default")),
        claude_max_turns=cast(int | None, raw.get("claude_max_turns")),
        claude_execution_mode=str(
            raw.get("claude_execution_mode", CLAUDE_EXECUTION_SDK)
        ),
        child_timeout_seconds=float(raw_child_timeout),
        agent_cwd=Path(agent_cwd_value).expanduser().resolve()
        if agent_cwd_value
        else None,
    )


@overload
def _load_decisions(
    path: str | Path,
    *,
    allow_revise: Literal[False] = False,
) -> dict[str, ArticulationReviewDecision]: ...


@overload
def _load_decisions(
    path: str | Path,
    *,
    allow_revise: Literal[True],
) -> dict[str, EmbeddedArticulationReviewDecision]: ...


def _load_decisions(
    path: str | Path,
    *,
    allow_revise: bool = False,
) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Articulation decisions file is not a file: {resolved}"
        )
    try:
        payload: Any = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid articulation decisions JSON at {resolved}: {exc}"
        ) from exc
    if isinstance(payload, dict) and isinstance(payload.get("decisions"), dict):
        payload = payload["decisions"]
    if not isinstance(payload, dict) or not payload:
        raise ValueError(
            "Articulation decisions JSON must be a non-empty candidate-to-decision object."
        )
    decisions: dict[str, Any] = {}
    for candidate_id, decision in payload.items():
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError("Articulation decision candidate IDs must be strings.")
        allowed = (
            {"accept", "reject", "revise"}
            if allow_revise
            else {
                "accept",
                "reject",
            }
        )
        if decision not in allowed:
            raise ValueError(
                f"Decision for {candidate_id!r} must be one of {sorted(allowed)}."
            )
        decisions[candidate_id] = cast(EmbeddedArticulationReviewDecision, decision)
    return decisions


__all__ = [
    "ARTICULATION_LAUNCHER_METADATA_KEY",
    "ARTICULATION_LAUNCHER_SCHEMA_VERSION",
    "ARTICULATION_SCENE_POLICY_ID",
    "ArticulationRunConfig",
    "resume_articulation_workflow",
    "revise_articulation_graph_workflow",
    "review_articulation_workflow",
    "run_articulation_workflow",
]
