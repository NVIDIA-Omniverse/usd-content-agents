# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thin public launcher for one composed, durable asset workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import secrets
import shutil
import stat
import sys
import time
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal, cast

from content_agent_workflows.asset_composition import (
    LEGACY_ASSET_LEAF_CATALOG_SCHEMA_VERSION,
    ArtifactBinding,
    AssetCadModelingRequest,
    AssetCadParameterVariant,
    AssetCompositionStateError,
    AssetCoordinatorLeaseError,
    AssetCoordinatorSession,
    AssetGeometryRequest,
    AssetLeafCatalog,
    AssetRunRequest,
    AssetRuntimeRequest,
    AssetSegmentationRunBinding,
    AssetSoleCoordinatorIdentity,
    AssetSourceStaging,
    AssetTerminalValidation,
    PhysicsValidationMode,
    bind_usd_dependency_closure,
    cancel_stage,
    canonical_asset_digest,
    create_run,
    discover_repository_asset_leaf_catalog,
    fail_stage,
    finalize_graph_run,
    load_verified_asset_request,
    load_verified_run,
    record_review_decisions,
    recover_leaf,
    recover_stage,
    resolve_repository_asset_leaf_catalog,
    run_asset_coordinator_transition,
    run_batch_asset_coordinator,
    run_interactive_asset_coordinator,
    validate_terminal,
)
from content_agent_workflows.asset_composition.coordinator import AssetReasoningLoop
from content_agent_workflows.asset_composition.models import (
    LegacyAssetCadModelingRequest,
)
from content_agent_workflows.common.artifacts import (
    atomic_write_json,
    atomic_write_text,
    file_sha256,
    read_contained_artifact,
)
from content_agent_workflows.common.usd_cli import (
    resolve_package_owned_usd_cli_route,
    run_bounded_usd_cli_subprocess,
    sanitized_usd_cli_execution_env,
)
from content_agent_workflows.common.usd_cli_session import (
    MAX_PARENT_USD_CLI_SESSION_IDENTITY_BYTES,
    MAX_USD_CLI_RECEIPT_CHECKPOINT_BYTES,
    ParentUsdCliArtifactIdentity,
    ParentUsdCliGeneratedSourceIdentity,
    ParentUsdCliSessionIdentity,
    ParentUsdCliStagedSourceIdentity,
    WorkflowUsdCliSession,
    parent_usd_cli_daemon_identity_sha256,
)
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .prompts import CONTROLLED_JSON_ARTIFACT_WRITE
from .runner import (
    CLAUDE_EXECUTION_CLI,
    CLAUDE_EXECUTION_SDK,
    CODEX_SANDBOX_WORKSPACE_WRITE,
    RUNNER_CLAUDE,
    RUNNER_CODEX,
    SUPPORTED_CLAUDE_EXECUTION_MODES,
    SUPPORTED_CODEX_SANDBOX_MODES,
    SUPPORTED_RUNNERS,
    USD_CLI_ASSET_MAX_IDLE_TIMEOUT_SECONDS,
    AssetCompositionChildConfig,
    ChildProcessInterrupted,
    UsdCliDaemonCleanupEvidence,
    UsdCliDaemonIdentity,
    UsdCliDaemonLease,
    UsdCliDaemonStrictStartError,
    UsdCliDaemonTeardownEvidence,
    UsdCliStagedInput,
    UsdCliTelemetryRoute,
    _activate_usd_cli_child_config,
    _append_child_runner_error,
    _attach_workflow_usd_cli_session,
    _lexical_absolute_path,
    _prepare_usd_cli_telemetry_route,
    _reject_unsafe_run_links,
    _resolve_materials_usd_from_manifest,
    _run_child_agent,
    _stage_usd_cli_input_tree,
    _staged_input_integrity_errors,
    _start_usd_cli_run_daemon_strict,
    _stop_usd_cli_run_daemon_strict,
    find_repo_root,
)
from .source_dependency_closure import (
    discover_non_usd_source_closure,
    source_root_stages_whole_tree,
)
from .usd_cli_backend import UsdCliReadiness, ensure_usd_cli_ovrtx_ready

PROMPT_REFERENCE_RELATIVE_PATH = Path("inputs/prompt-reference.md")
SOURCE_INTENT_RELATIVE_PATH = Path("inputs/source-intent.json")
SEGMENTATION_INPUT_RELATIVE_ROOT = Path("inputs/geometry-segmentation")
DEFAULT_GEOMETRY_AUTHORING_PROVIDER_ID = os.getenv(
    "GEOMETRY_AUTHORING_PROVIDER_ID",
    "external-geometry-authoring",
)
CAD_SOURCE_IMAGE_SUFFIXES = frozenset({".jpeg", ".jpg", ".png", ".webp"})
USD_SOURCE_SUFFIXES = frozenset({".usd", ".usda", ".usdc", ".usdz"})
IMMUTABLE_SOURCE_CLOSURE_SCHEMA_VERSION: Final = (
    "content-workflow-cli.immutable-source-closure.v1"
)
MAX_STAGED_SEGMENTATION_FILES = 20_000
MAX_STAGED_SEGMENTATION_BYTES = 8 * 1024 * 1024 * 1024
ASSET_USD_CLI_TEARDOWN_SCHEMA_VERSION: Final = (
    "content-workflow-cli.asset-usd-cli-teardown.v1"
)
ASSET_USD_CLI_LAUNCH_SCHEMA_VERSION: Final = (
    "content-workflow-cli.asset-usd-cli-launch.v1"
)
MAX_ASSET_USD_CLI_TEARDOWN_RECEIPT_BYTES: Final = 1024 * 1024
MAX_ASSET_USD_CLI_DAEMON_LOG_BYTES: Final = 16 * 1024 * 1024
MAX_ASSET_USD_CLI_RECEIPT_JOURNAL_BYTES: Final = 64 * 1024 * 1024
MAX_ASSET_USD_CLI_LIFECYCLE_RECORDS: Final = 128
ASSET_USD_CLI_PROVIDER_FREE_READINESS_SCHEMA_VERSION: Final = (
    "content-workflow-cli.asset-usd-cli-provider-free-readiness.v1"
)


@dataclass(frozen=True)
class AssetRunConfig(AssetCompositionChildConfig):
    """Resolved launcher configuration satisfying the shared runner contract."""

    repo_root: Path
    usd_path: Path | None
    prompt: str
    source_root: Path | None = None
    selected_mode: Literal["agentic", "compatibility_fixed"] = "agentic"
    leaf_catalog: AssetLeafCatalog | None = None
    required_leaf_ids: list[str] = field(default_factory=list)
    required_terminal_leaf_ids: list[str] = field(default_factory=list)
    required_leaf_dependencies: dict[str, list[str]] = field(default_factory=dict)
    exact_leaf_scope: bool = False
    joint_config: Path | None = None
    materials_yaml: Path | None = None
    materials_usd: Path | None = None
    source_images: list[Path] = field(default_factory=list)
    reference_images: list[Path] = field(default_factory=list)
    reference_files: list[Path] | None = None
    output_dir: Path | None = None
    run_id: str | None = None
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
    scene_tool_timeout_seconds: float = 60.0
    child_timeout_seconds: float = 3600.0
    physics_validation_mode: PhysicsValidationMode | None = None
    geometry_target_profile: str = "geometry-agent.static-visual-asset.v1"
    geometry_optimization_policy: str = "preserve_correspondence"
    geometry_optimizer_backend: str = "local"
    geometry_repair_mode: str = "off"
    geometry_repair_profile: str = "visual_only"
    geometry_render_evidence: bool = True
    geometry_segmentation_run_dir: Path | None = None
    geometry_segmentation_required: bool = False
    geometry_required_parts: list[str] = field(default_factory=list)
    include_geometry_stage: bool = True
    cad_provider_id: str = DEFAULT_GEOMETRY_AUTHORING_PROVIDER_ID
    cad_parameter_values: dict[str, str | int | float | bool] = field(
        default_factory=dict
    )
    cad_parameter_variants: list[AssetCadParameterVariant] = field(default_factory=list)
    cad_required_outputs: list[str] = field(default_factory=lambda: ["usd"])
    dry_run: bool = False
    agent_workspace: Path | None = None
    agent_cwd: Path | None = None
    parent_usd_cli_session_identity: Path | None = None
    parent_usd_cli_session_identity_sha256: str | None = None


class AssetUsdCliTeardownReceipt(BaseModel):
    """Terminal release evidence for one parent-owned asset launcher session."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["content-workflow-cli.asset-usd-cli-teardown.v1"] = (
        ASSET_USD_CLI_TEARDOWN_SCHEMA_VERSION
    )
    created_at: str
    launch_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    run_id: str = Field(min_length=1)
    session_identity_path: str | None = None
    session_identity_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    daemon_identity_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    daemon_was_started: bool
    status: Literal["not_started", "released", "failed"]
    boundary: Literal[
        "completed",
        "human_review",
        "failed",
        "cancelled",
        "incomplete",
        "setup_failed",
    ]
    child_returncode: int
    interrupted: bool
    process_released: bool | None = None
    descendants_released: bool | None = None
    sessions_released: bool | None = None
    listener_released: bool | None = None
    daemon_leases_released: bool | None = None
    state_directory_released: bool | None = None
    source_integrity_verified: bool
    listener_host: str | None = None
    listener_port: int | None = Field(default=None, ge=1, le=65535)
    renderer_credentials: Literal["parent_confined"] = "parent_confined"
    daemon_log_path: str | None = None
    daemon_log_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    command_receipt_journal: ArtifactBinding | None = None
    command_receipt_checkpoint: ArtifactBinding | None = None
    setup_error: str | None = None
    errors: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_release_status(self) -> AssetUsdCliTeardownReceipt:
        if (self.session_identity_path is None) != (
            self.session_identity_sha256 is None
        ):
            raise ValueError("session identity path and digest must appear together")
        if (self.daemon_log_path is None) != (self.daemon_log_sha256 is None):
            raise ValueError("daemon log path and digest must appear together")
        if (self.command_receipt_journal is None) != (
            self.command_receipt_checkpoint is None
        ):
            raise ValueError(
                "command receipt journal and checkpoint bindings must appear together"
            )
        if self.daemon_was_started != (self.daemon_identity_sha256 is not None):
            raise ValueError(
                "started daemon status must match the captured identity digest"
            )
        released = (
            all(
                value is True
                for value in (
                    self.process_released,
                    self.descendants_released,
                    self.sessions_released,
                    self.listener_released,
                    self.daemon_leases_released,
                    self.state_directory_released,
                )
            )
            and self.source_integrity_verified
        )
        if self.status == "released" and (not released or self.errors):
            raise ValueError("released teardown requires complete clean evidence")
        if (
            self.status == "released"
            and self.session_identity_path is not None
            and self.command_receipt_journal is None
        ):
            raise ValueError(
                "released child session requires sealed command receipt evidence"
            )
        if self.status == "not_started":
            resource_evidence = (
                self.process_released,
                self.descendants_released,
                self.sessions_released,
                self.listener_released,
                self.daemon_leases_released,
                self.state_directory_released,
            )
            if (
                self.daemon_was_started
                or self.daemon_identity_sha256 is not None
                or any(value is not None for value in resource_evidence)
                or self.setup_error is None
                or self.errors
            ):
                raise ValueError(
                    "not-started teardown requires only a concrete setup error"
                )
        if self.status == "failed" and not self.errors:
            raise ValueError("failed teardown requires a concrete error")
        return self


class AssetUsdCliLaunchIntent(BaseModel):
    """Durable proof that one parent lifecycle must end in a receipt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["content-workflow-cli.asset-usd-cli-launch.v1"] = (
        ASSET_USD_CLI_LAUNCH_SCHEMA_VERSION
    )
    created_at: str
    launch_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    run_id: str = Field(min_length=1)


@dataclass(frozen=True)
class AssetRunResult:
    """Observable result for run, review, and resume commands."""

    run_dir: Path
    request_path: Path
    run_state_path: Path
    prompt_path: Path
    child_output_path: Path
    child_final_path: Path
    terminal_validation_path: Path | None
    returncode: int
    completed: bool
    needs_review: bool


def add_asset_subcommands(subparsers: Any) -> None:
    """Register public asset catalog/run/review/resume commands."""

    asset = subparsers.add_parser(
        "asset",
        help=(
            "Run one frozen asset graph or the explicit fixed CAD/Geometry "
            "compatibility workflow."
        ),
    )
    asset_subparsers = asset.add_subparsers(dest="asset_command")

    catalog = asset_subparsers.add_parser(
        "catalog",
        help="Print the deterministic repository-owned public leaf catalog.",
    )
    catalog.set_defaults(handler=_handle_catalog)

    run = asset_subparsers.add_parser(
        "run",
        help="Start one durable composed-asset workflow from a natural-language goal.",
    )
    _add_run_args(run)
    run.set_defaults(handler=_handle_run)

    review = asset_subparsers.add_parser(
        "review",
        help="Bind exact Joint candidate decisions and resume the same workflow.",
    )
    _add_existing_run_args(review)
    review.add_argument("--decisions-json", required=True, type=Path)
    review.add_argument("--reviewer", required=True)
    review.set_defaults(handler=_handle_review)

    resume = asset_subparsers.add_parser(
        "resume",
        help="Resume the next frozen leaf or compatibility stage.",
    )
    _add_existing_run_args(resume)
    resume.add_argument(
        "--recover",
        required=False,
        help="Explicit reason for reopening a failed or cancelled current stage.",
    )
    resume.set_defaults(handler=_handle_resume)


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--source",
        type=Path,
        help=(
            "Provided CAD, mesh, USD, URDF, MJCF, or provider-exported source. "
            "Use geometry-agent before asset composition for text/image authoring."
        ),
    )
    source.add_argument(
        "--usd",
        type=Path,
        help="Compatibility alias for --source.",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        help=(
            "Approved dependency-tree root for an opaque multi-file --source or "
            "a parsed source with references above its directory. Parsed formats "
            "stage only their reachable dependency graph; opaque formats freeze "
            "every regular file below this root."
        ),
    )
    prompt = parser.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt", help="Natural-language goal for the final asset.")
    prompt.add_argument(
        "--prompt-file", type=Path, help="UTF-8 file containing the goal."
    )
    parser.add_argument(
        "--leaf-catalog",
        type=Path,
        help=(
            "Compatibility readback of repository catalog bytes; arbitrary "
            "run-local descriptors are rejected."
        ),
    )
    parser.add_argument(
        "--required-leaf",
        action="append",
        default=[],
        help=(
            "Repository leaf ID that the frozen graph must select as required. "
            "May be repeated."
        ),
    )
    parser.add_argument(
        "--required-terminal-leaf",
        action="append",
        default=[],
        help=("Required leaf ID that must be a terminal output. May be repeated."),
    )
    parser.add_argument(
        "--required-leaf-dependency",
        action="append",
        default=[],
        metavar="LEAF=DEPENDENCY",
        help=(
            "Prompt-specific dependency edge that the frozen graph must include. "
            "Both IDs must also be supplied with --required-leaf. May be repeated."
        ),
    )
    parser.add_argument(
        "--exact-leaf-scope",
        action="store_true",
        help=(
            "Reject every graph leaf outside --required-leaf and require a "
            "prompt-relevance rationale for each selected leaf."
        ),
    )
    parser.add_argument(
        "--compatibility-fixed-order",
        action="store_true",
        help=(
            "Explicitly use the historical fixed-stage compatibility path. This "
            "mode is never entered from agentic qualification."
        ),
    )
    parser.add_argument(
        "--joint-config",
        type=Path,
        help="Fixed compatibility only: Joint Agent stage configuration.",
    )
    parser.add_argument(
        "--materials-yaml",
        type=Path,
        help="Fixed compatibility only: Material stage library manifest.",
    )
    parser.add_argument(
        "--materials-usd",
        type=Path,
        help="Material library USD. Defaults to library_path in --materials-yaml.",
    )
    parser.add_argument(
        "--reference-image",
        action="append",
        default=[],
        type=Path,
        help="Optional appearance or behavior reference image. May be repeated.",
    )
    parser.add_argument(
        "--reference",
        action="append",
        default=[],
        type=Path,
        help="Optional readable reference file. May be repeated.",
    )
    parser.add_argument(
        "--geometry-target-profile",
        default="geometry-agent.static-visual-asset.v1",
        help="Frozen Geometry validation profile.",
    )
    parser.add_argument(
        "--geometry-optimization-policy",
        choices=["skip", "preserve_correspondence", "runtime_efficiency"],
        default="preserve_correspondence",
    )
    parser.add_argument(
        "--geometry-optimizer-backend",
        choices=["local", "remote"],
        default="local",
    )
    parser.add_argument(
        "--geometry-repair-mode",
        choices=["off", "diagnose", "auto"],
        default="off",
    )
    parser.add_argument(
        "--geometry-repair-profile",
        choices=[
            "visual_only",
            "static_environment",
            "rigid_pick_place",
            "articulated_rigid",
            "contact_rich",
            "deformable_or_cae",
        ],
        default="visual_only",
    )
    parser.add_argument(
        "--geometry-render-evidence",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require digest-bound OVRTX Geometry evidence.",
    )
    parser.add_argument(
        "--geometry-segmentation-run-dir",
        type=Path,
        help=(
            "Completed mesh-segmentation run to validate and freeze below this "
            "composed run's immutable inputs."
        ),
    )
    parser.add_argument(
        "--geometry-segmentation-required",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--geometry-required-part",
        action="append",
        default=[],
        help="Required semantic part name. May be repeated.",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument(
        "--runner",
        choices=[RUNNER_CODEX, RUNNER_CLAUDE],
        default=RUNNER_CODEX,
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--model-reasoning-effort", default=None)
    parser.add_argument(
        "--codex-base-url",
        default=os.getenv("CONTENT_AGENT_CODEX_BASE_URL"),
    )
    parser.add_argument(
        "--codex-sandbox-mode",
        choices=[CODEX_SANDBOX_WORKSPACE_WRITE],
        default=CODEX_SANDBOX_WORKSPACE_WRITE,
    )
    parser.add_argument("--codex-config-json", action="append", default=[])
    parser.add_argument(
        "--claude-permission-mode",
        choices=["default", "acceptEdits", "bypassPermissions", "plan"],
        default="default",
    )
    parser.add_argument("--claude-max-turns", type=int, default=None)
    parser.add_argument(
        "--claude-execution-mode",
        choices=[CLAUDE_EXECUTION_SDK, CLAUDE_EXECUTION_CLI],
        default=CLAUDE_EXECUTION_SDK,
    )
    parser.add_argument("--claude-config-json", action="append", default=[])
    parser.add_argument("--scene-tool-timeout", type=float, default=60.0)
    parser.add_argument("--child-timeout", type=float, default=3600.0)
    parser.add_argument(
        "--physics-validation-mode",
        choices=["runtime-required", "schema-readback"],
        default=None,
        help=(
            "Require passing runtime evidence, or accept exact Physics schema "
            "readback while preserving explicit non-qualified runtime warnings."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")


def _add_existing_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Update/verify durable state without launching a child agent.",
    )


def _parse_required_leaf_dependencies(values: list[str]) -> dict[str, list[str]]:
    parsed: dict[str, list[str]] = {}
    for raw in values:
        owner, separator, dependency = raw.partition("=")
        owner = owner.strip()
        dependency = dependency.strip()
        if separator != "=" or not owner or not dependency:
            raise ValueError("--required-leaf-dependency requires LEAF=DEPENDENCY")
        parsed.setdefault(owner, []).append(dependency)
    for owner, dependencies in parsed.items():
        if len(dependencies) != len(set(dependencies)):
            raise ValueError(
                f"--required-leaf-dependency must not repeat an edge: {owner}"
            )
    return {
        owner: sorted(dependencies) for owner, dependencies in sorted(parsed.items())
    }


def _handle_run(args: argparse.Namespace) -> int:
    repo_root = (
        args.repo_root.expanduser().resolve() if args.repo_root else find_repo_root()
    )
    if args.prompt_file is not None:
        prompt = args.prompt_file.expanduser().read_text(encoding="utf-8")
    else:
        prompt = str(args.prompt)
    source_path = args.source if args.source is not None else args.usd
    reference_images = list(
        dict.fromkeys([path.expanduser().resolve() for path in args.reference_image])
    )
    config = AssetRunConfig(
        repo_root=repo_root,
        usd_path=(source_path.expanduser().resolve() if source_path else None),
        prompt=prompt,
        source_root=(
            args.source_root.expanduser().resolve()
            if args.source_root is not None
            else None
        ),
        selected_mode=(
            "compatibility_fixed" if args.compatibility_fixed_order else "agentic"
        ),
        leaf_catalog=(
            _load_leaf_catalog(args.leaf_catalog)
            if args.leaf_catalog is not None
            else None
        ),
        required_leaf_ids=list(args.required_leaf),
        required_terminal_leaf_ids=list(args.required_terminal_leaf),
        required_leaf_dependencies=_parse_required_leaf_dependencies(
            list(args.required_leaf_dependency)
        ),
        exact_leaf_scope=bool(args.exact_leaf_scope),
        joint_config=(
            args.joint_config.expanduser().resolve()
            if args.joint_config is not None
            else None
        ),
        materials_yaml=(
            args.materials_yaml.expanduser().resolve()
            if args.materials_yaml is not None
            else None
        ),
        materials_usd=(
            args.materials_usd.expanduser().resolve()
            if args.materials_usd is not None
            else None
        ),
        source_images=[],
        reference_images=reference_images,
        reference_files=[path.expanduser().resolve() for path in args.reference],
        output_dir=args.output_dir.expanduser() if args.output_dir else None,
        run_id=args.run_id,
        runner=args.runner,
        model=args.model,
        model_reasoning_effort=args.model_reasoning_effort,
        codex_base_url=args.codex_base_url,
        codex_sandbox_mode=args.codex_sandbox_mode,
        codex_config=_merge_json_objects(args.codex_config_json),
        claude_config=_merge_json_objects(args.claude_config_json),
        claude_permission_mode=args.claude_permission_mode,
        claude_max_turns=args.claude_max_turns,
        claude_execution_mode=args.claude_execution_mode,
        scene_tool_timeout_seconds=args.scene_tool_timeout,
        child_timeout_seconds=args.child_timeout,
        physics_validation_mode=(
            cast(
                PhysicsValidationMode,
                args.physics_validation_mode.replace("-", "_"),
            )
            if args.physics_validation_mode is not None
            else None
        ),
        geometry_target_profile=args.geometry_target_profile,
        geometry_optimization_policy=args.geometry_optimization_policy,
        geometry_optimizer_backend=args.geometry_optimizer_backend,
        geometry_repair_mode=args.geometry_repair_mode,
        geometry_repair_profile=args.geometry_repair_profile,
        geometry_render_evidence=args.geometry_render_evidence,
        geometry_segmentation_run_dir=(
            args.geometry_segmentation_run_dir.expanduser()
            if args.geometry_segmentation_run_dir is not None
            else None
        ),
        geometry_segmentation_required=args.geometry_segmentation_required,
        geometry_required_parts=list(args.geometry_required_part),
        dry_run=args.dry_run,
    )
    return _print_result(run_asset_workflow(config))


def _handle_catalog(_args: argparse.Namespace) -> int:
    print(discover_repository_asset_leaf_catalog().model_dump_json(indent=2))
    return 0


def _handle_review(args: argparse.Namespace) -> int:
    result = review_asset_workflow(
        args.run_dir,
        decisions_path=args.decisions_json,
        reviewer=args.reviewer,
        dry_run=args.dry_run,
    )
    return _print_result(result)


def _handle_resume(args: argparse.Namespace) -> int:
    result = resume_asset_workflow(
        args.run_dir,
        recovery_reason=args.recover,
        dry_run=args.dry_run,
    )
    return _print_result(result)


def _print_result(result: AssetRunResult) -> int:
    print(f"Run directory: {result.run_dir}")
    print(f"Request: {result.request_path}")
    print(f"Run state: {result.run_state_path}")
    print(f"Prompt: {result.prompt_path}")
    print(f"Needs review: {str(result.needs_review).lower()}")
    print(f"Completed: {str(result.completed).lower()}")
    if result.terminal_validation_path is not None:
        print(f"Terminal validation: {result.terminal_validation_path}")
    return result.returncode


def run_asset_workflow(config: AssetRunConfig) -> AssetRunResult:
    """Create and optionally execute a fresh composed-asset run."""

    return _run_fresh_asset_workflow(
        config,
        invocation_mode="batch",
        reasoning_loop=None,
    )


def run_interactive_asset_workflow(
    config: AssetRunConfig,
    *,
    reasoning_loop: AssetReasoningLoop,
) -> AssetRunResult:
    """Run one public agentic callback inside the launcher-owned lifecycle."""

    if config.selected_mode != "agentic":
        raise ValueError(
            "The public interactive asset lifecycle requires selected_mode=agentic"
        )
    return _run_fresh_asset_workflow(
        config,
        invocation_mode="interactive",
        reasoning_loop=reasoning_loop,
    )


def _run_fresh_asset_workflow(
    config: AssetRunConfig,
    *,
    invocation_mode: Literal["interactive", "batch"],
    reasoning_loop: AssetReasoningLoop | None,
) -> AssetRunResult:
    """Share exact request/source preparation across public launcher modes."""

    normalized = _validated_config(config)
    if (
        normalized.geometry_segmentation_required or normalized.geometry_required_parts
    ) and normalized.geometry_segmentation_run_dir is None:
        raise ValueError(
            "Required composed Geometry segmentation needs "
            "--geometry-segmentation-run-dir"
        )
    run_id, run_dir = _create_run_dir(normalized)
    request_path = run_dir / "request.json"
    run_state_path = run_dir / "asset_run.json"
    try:
        normalized = replace(normalized, agent_cwd=run_dir)
        if normalized.selected_mode == "compatibility_fixed":
            normalized = _with_material_reference(normalized, run_dir=run_dir)
        source_staging: AssetSourceStaging | None = None
        if normalized.usd_path is not None:
            if _is_usd_source_path(normalized.usd_path):
                staged_source = _stage_usd_cli_input_tree(
                    label="asset_source",
                    source_usd_path=normalized.usd_path,
                    run_dir=run_dir,
                )
                source_staging = _source_staging_contract(staged_source)
                source_asset = staged_source.staged_usd_path
            else:
                source_asset, source_staging = _stage_immutable_source_file(
                    source_path=normalized.usd_path,
                    source_root=normalized.source_root,
                    run_dir=run_dir,
                )
            normalized = replace(normalized, usd_path=source_asset)
        else:
            source_asset = _write_source_intent(normalized, run_dir=run_dir)
        segmentation_run = (
            _stage_completed_segmentation_run(
                normalized.geometry_segmentation_run_dir,
                destination_parent=run_dir / SEGMENTATION_INPUT_RELATIVE_ROOT,
                source_asset=source_asset,
                required=normalized.geometry_segmentation_required,
                required_parts=normalized.geometry_required_parts,
            )
            if normalized.geometry_segmentation_run_dir is not None
            else None
        )
        if segmentation_run is not None:
            normalized = replace(
                normalized,
                geometry_segmentation_run_dir=Path(segmentation_run.run_dir),
            )
        request = _build_request(
            normalized,
            run_id=run_id,
            run_dir=run_dir,
            run_state_path=run_state_path,
            source_asset=source_asset,
            segmentation_run=segmentation_run,
            source_staging=source_staging,
        )
    except (Exception, KeyboardInterrupt) as exc:
        try:
            _remove_uncommitted_fresh_run(run_dir, request_path=request_path)
        except Exception as cleanup_exc:  # noqa: BLE001 - preserve both failures
            exc.add_note(
                "Fresh-run cleanup also failed: "
                f"{type(cleanup_exc).__name__}: {cleanup_exc}"
            )
        raise
    atomic_write_json(request_path, request)
    create_run(
        run_state_path,
        run_id=run_id,
        request_path=request_path,
        source_asset=source_asset,
        include_geometry_stage=(
            normalized.selected_mode == "compatibility_fixed"
            and normalized.include_geometry_stage
        ),
        include_cad_modeling_stage=(
            normalized.selected_mode == "compatibility_fixed"
            and normalized.usd_path is None
        ),
    )
    _verify_source_intent(request, run_dir=run_dir)
    prompt_path = run_dir / "agent_prompt.md"
    prompt = _build_agent_prompt(
        request=request,
        request_path=request_path,
        run_state_path=run_state_path,
        run_dir=run_dir,
        resume=False,
        include_cad_modeling=request.cad_modeling is not None,
    )
    if normalized.dry_run:
        atomic_write_text(prompt_path, prompt, within=run_dir)
    return _prepared_or_execute(
        config=normalized,
        request=request,
        request_path=request_path,
        run_state_path=run_state_path,
        prompt_path=prompt_path,
        prompt=prompt,
        suffix="run",
        invocation_mode=invocation_mode,
        reasoning_loop=reasoning_loop,
    )


def review_asset_workflow(
    run_dir: str | Path,
    *,
    decisions_path: str | Path,
    reviewer: str,
    dry_run: bool = False,
) -> AssetRunResult:
    """Bind exact articulation review bytes and resume the frozen workflow."""

    resolved, request, config = _load_existing(run_dir, dry_run=dry_run)
    if request.selected_mode != "compatibility_fixed":
        raise AssetCompositionStateError(
            "asset review is available only in fixed compatibility mode"
        )
    run_state_path = resolved / "asset_run.json"

    def record_guarded_review() -> object:
        _require_safe_asset_usd_cli_teardown_history(
            resolved,
            run_id=request.run_id,
            terminal_valid=False,
            reserve_launch_slot=True,
        )
        return record_review_decisions(
            run_state_path,
            decisions_path=decisions_path,
            reviewer=reviewer,
        )

    run_asset_coordinator_transition(
        run_state_path,
        transition=record_guarded_review,
    )
    return _resume_prepared(resolved, request=request, config=config, suffix="review")


def resume_asset_workflow(
    run_dir: str | Path,
    *,
    recovery_reason: str | None = None,
    dry_run: bool = False,
) -> AssetRunResult:
    """Resume from the exact next stage, optionally reopening a stopped run."""

    return _resume_asset_workflow(
        run_dir,
        recovery_reason=recovery_reason,
        dry_run=dry_run,
        invocation_mode="batch",
        reasoning_loop=None,
    )


def resume_interactive_asset_workflow(
    run_dir: str | Path,
    *,
    reasoning_loop: AssetReasoningLoop,
    recovery_reason: str | None = None,
    dry_run: bool = False,
) -> AssetRunResult:
    """Resume one public agentic callback inside a new parent lifecycle."""

    return _resume_asset_workflow(
        run_dir,
        recovery_reason=recovery_reason,
        dry_run=dry_run,
        invocation_mode="interactive",
        reasoning_loop=reasoning_loop,
    )


def _resume_asset_workflow(
    run_dir: str | Path,
    *,
    recovery_reason: str | None,
    dry_run: bool,
    invocation_mode: Literal["interactive", "batch"],
    reasoning_loop: AssetReasoningLoop | None,
) -> AssetRunResult:
    """Share exact resume/recovery gates across public launcher modes."""

    resolved, request, config = _load_existing(run_dir, dry_run=dry_run)
    if invocation_mode == "interactive" and request.selected_mode != "agentic":
        raise AssetCompositionStateError(
            "The public interactive asset lifecycle can resume only agentic runs"
        )
    run_state_path = resolved / "asset_run.json"
    run = load_verified_run(run_state_path)
    current_stage = run.current_stage
    current_leaf = run.current_leaf_id
    if run.terminal_status in {"failed", "cancelled"} and (
        current_stage is not None or current_leaf is not None
    ):
        if not recovery_reason or not recovery_reason.strip():
            raise AssetCompositionStateError(
                "A failed or cancelled run requires --recover with an explicit reason"
            )

        def recover_guarded_stage() -> object:
            _require_safe_asset_usd_cli_teardown_history(
                resolved,
                run_id=request.run_id,
                terminal_valid=False,
                reserve_launch_slot=True,
            )
            if run.selected_mode == "agentic":
                assert current_leaf is not None
                return recover_leaf(
                    run_state_path,
                    current_leaf,
                    reason=recovery_reason,
                    actor="content-workflow-cli",
                )
            assert current_stage is not None
            return recover_stage(
                run_state_path,
                current_stage,
                reason=recovery_reason,
                actor="content-workflow-cli",
            )

        run_asset_coordinator_transition(
            run_state_path,
            transition=recover_guarded_stage,
        )
    elif recovery_reason is not None:
        raise AssetCompositionStateError(
            "--recover is accepted only for a failed or cancelled current stage"
        )
    return _resume_prepared(
        resolved,
        request=request,
        config=config,
        suffix="resume",
        invocation_mode=invocation_mode,
        reasoning_loop=reasoning_loop,
    )


def _resume_prepared(
    run_dir: Path,
    *,
    request: AssetRunRequest,
    config: AssetRunConfig,
    suffix: str,
    invocation_mode: Literal["interactive", "batch"] = "batch",
    reasoning_loop: AssetReasoningLoop | None = None,
) -> AssetRunResult:
    run_state_path = run_dir / "asset_run.json"
    terminal = validate_terminal(run_state_path)
    if terminal.valid:
        run_asset_coordinator_transition(
            run_state_path,
            transition=lambda: _require_safe_asset_usd_cli_teardown_history(
                run_dir,
                run_id=request.run_id,
                terminal_valid=True,
            ),
        )
        return _result_without_child(
            run_dir,
            terminal=terminal,
            suffix=suffix,
            returncode=0,
        )
    run = load_verified_run(run_state_path)
    if (
        run.selected_mode == "compatibility_fixed"
        and run.current_stage == "articulation"
        and run.stages["articulation"].status == "needs_review"
    ):
        raise AssetCompositionStateError(
            "Articulation needs review; use asset review with exact decisions"
        )
    prompt_path = run_dir / f"agent_{suffix}_prompt.md"
    prompt = _build_agent_prompt(
        request=request,
        request_path=run_dir / "request.json",
        run_state_path=run_state_path,
        run_dir=run_dir,
        resume=True,
        include_cad_modeling=request.cad_modeling is not None,
    )
    if config.dry_run:
        run_asset_coordinator_transition(
            run_state_path,
            transition=lambda: _require_safe_asset_usd_cli_teardown_history(
                run_dir,
                run_id=request.run_id,
                terminal_valid=False,
            ),
        )
        atomic_write_text(prompt_path, prompt, within=run_dir)
    return _prepared_or_execute(
        config=replace(config, agent_cwd=run_dir),
        request=request,
        request_path=run_dir / "request.json",
        run_state_path=run_state_path,
        prompt_path=prompt_path,
        prompt=prompt,
        suffix=suffix,
        invocation_mode=invocation_mode,
        reasoning_loop=reasoning_loop,
    )


def _prepared_or_execute(
    *,
    config: AssetRunConfig,
    request: AssetRunRequest,
    request_path: Path,
    run_state_path: Path,
    prompt_path: Path,
    prompt: str,
    suffix: str,
    invocation_mode: Literal["interactive", "batch"],
    reasoning_loop: AssetReasoningLoop | None,
) -> AssetRunResult:
    run_dir = Path(request.run_dir)
    child_output_path = run_dir / f"child-{suffix}-output.log"
    child_final_path = run_dir / f"child-{suffix}-final.md"
    if config.dry_run:
        run = load_verified_run(run_state_path)
        return AssetRunResult(
            run_dir=run_dir,
            request_path=request_path,
            run_state_path=run_state_path,
            prompt_path=prompt_path,
            child_output_path=child_output_path,
            child_final_path=child_final_path,
            terminal_validation_path=None,
            returncode=0,
            completed=False,
            needs_review=(
                run.selected_mode == "compatibility_fixed"
                and run.current_stage == "articulation"
                and run.stages["articulation"].status == "needs_review"
            ),
        )
    return _execute_with_parent_usd_cli_lifecycle(
        config=config,
        request=request,
        request_path=request_path,
        run_state_path=run_state_path,
        prompt_path=prompt_path,
        prompt=prompt,
        child_output_path=child_output_path,
        child_final_path=child_final_path,
        suffix=suffix,
        invocation_mode=invocation_mode,
        reasoning_loop=reasoning_loop,
    )


def _parent_artifact_identity(
    binding: ArtifactBinding,
) -> ParentUsdCliArtifactIdentity:
    return ParentUsdCliArtifactIdentity(
        path=binding.path,
        sha256=binding.sha256,
        size_bytes=binding.size_bytes,
    )


def _daemon_identity_sha256(identity: UsdCliDaemonIdentity) -> str:
    return parent_usd_cli_daemon_identity_sha256(
        pid=identity.pid,
        process_start_token=identity.process_start_token,
        project_id=identity.project_id,
        instance_id=identity.instance_id,
        process_group_id=identity.process_group_id,
        os_session_id=identity.os_session_id,
    )


def _verify_asset_usd_cli_launch_intent(
    run_dir: Path,
    *,
    binding: ArtifactBinding,
    expected: AssetUsdCliLaunchIntent,
) -> None:
    """Re-attest the exact parent launch intent after child write access ends."""

    observed = read_contained_artifact(
        run_dir,
        binding.path,
        max_bytes=MAX_ASSET_USD_CLI_TEARDOWN_RECEIPT_BYTES,
        capture_bytes=True,
    )
    if observed.sha256 != binding.sha256 or observed.size_bytes != binding.size_bytes:
        raise RuntimeError("asset usd-cli launch intent changed during child execution")
    assert observed.data is not None
    try:
        current = AssetUsdCliLaunchIntent.model_validate_json(observed.data)
    except Exception as exc:  # noqa: BLE001 - normalize custody failure
        raise RuntimeError("asset usd-cli launch intent is no longer valid") from exc
    if current != expected:
        raise RuntimeError("asset usd-cli launch intent identity changed")


def _bind_asset_usd_cli_lifecycle_artifact(
    run_dir: Path,
    *,
    path: str,
    sha256: str,
    max_bytes: int,
    expected_size_bytes: int | None = None,
    expected_path: Path | None = None,
    expected_name_pattern: str | None = None,
    label: str,
) -> ArtifactBinding:
    """Confine and bind one immutable artifact named by a lifecycle receipt."""

    try:
        observed = read_contained_artifact(
            run_dir,
            path,
            max_bytes=max_bytes,
        )
    except Exception as exc:  # noqa: BLE001 - historical custody blocks reuse
        raise AssetCompositionStateError(
            f"Asset usd-cli {label} is invalid: {path}"
        ) from exc
    if expected_path is not None and observed.path != expected_path:
        raise AssetCompositionStateError(
            f"Asset usd-cli {label} path is inconsistent: {path}"
        )
    if expected_name_pattern is not None and not re.fullmatch(
        expected_name_pattern,
        observed.path.name,
    ):
        raise AssetCompositionStateError(
            f"Asset usd-cli {label} path is inconsistent: {path}"
        )
    expected_raw_dir = run_dir.resolve(strict=True) / "raw"
    if observed.path.parent != expected_raw_dir:
        raise AssetCompositionStateError(
            f"Asset usd-cli {label} must be stored directly in the run raw directory"
        )
    if observed.sha256 != sha256:
        raise AssetCompositionStateError(
            f"Asset usd-cli {label} digest is inconsistent: {path}"
        )
    if expected_size_bytes is not None and observed.size_bytes != expected_size_bytes:
        raise AssetCompositionStateError(
            f"Asset usd-cli {label} size is inconsistent: {path}"
        )
    return ArtifactBinding(
        path=str(observed.path),
        sha256=observed.sha256,
        size_bytes=observed.size_bytes,
    )


def _capture_asset_usd_cli_command_receipt_evidence(
    run_dir: Path,
    session: WorkflowUsdCliSession,
) -> tuple[ArtifactBinding, ArtifactBinding]:
    """Capture the exact parent journal/checkpoint pair at teardown."""

    session.verify_receipt_journal_integrity()
    run_root = run_dir.resolve(strict=True)
    expected_journal = run_root / "raw" / "usd_cli_command_receipts.jsonl"
    expected_checkpoint = run_root / "raw" / "usd_cli_command_receipts.checkpoint.json"
    observed_journal = read_contained_artifact(
        run_root,
        session.receipt_file,
        max_bytes=MAX_ASSET_USD_CLI_RECEIPT_JOURNAL_BYTES,
    )
    observed_checkpoint = read_contained_artifact(
        run_root,
        session.receipt_checkpoint_file,
        max_bytes=MAX_USD_CLI_RECEIPT_CHECKPOINT_BYTES,
    )
    if observed_journal.path != expected_journal:
        raise RuntimeError("asset usd-cli command receipt journal path changed")
    if observed_checkpoint.path != expected_checkpoint:
        raise RuntimeError("asset usd-cli command receipt checkpoint path changed")
    return (
        ArtifactBinding(
            path=str(observed_journal.path),
            sha256=observed_journal.sha256,
            size_bytes=observed_journal.size_bytes,
        ),
        ArtifactBinding(
            path=str(observed_checkpoint.path),
            sha256=observed_checkpoint.sha256,
            size_bytes=observed_checkpoint.size_bytes,
        ),
    )


def _verify_latest_asset_usd_cli_command_receipt_evidence(
    run_dir: Path,
    *,
    receipts: dict[str, AssetUsdCliTeardownReceipt],
) -> None:
    """Bind the live journal/checkpoint to the latest clean release receipt."""

    run_root = run_dir.resolve(strict=True)
    expected_journal = run_root / "raw" / "usd_cli_command_receipts.jsonl"
    expected_checkpoint = run_root / "raw" / "usd_cli_command_receipts.checkpoint.json"
    released: list[tuple[ArtifactBinding, ArtifactBinding]] = []
    for receipt in receipts.values():
        if receipt.status != "released":
            continue
        journal = receipt.command_receipt_journal
        checkpoint = receipt.command_receipt_checkpoint
        if journal is None or checkpoint is None:
            continue
        if Path(journal.path) != expected_journal:
            raise AssetCompositionStateError(
                "Asset usd-cli command receipt journal path is inconsistent"
            )
        if Path(checkpoint.path) != expected_checkpoint:
            raise AssetCompositionStateError(
                "Asset usd-cli command receipt checkpoint path is inconsistent"
            )
        released.append((journal, checkpoint))

    journal_exists = expected_journal.exists() or expected_journal.is_symlink()
    checkpoint_exists = expected_checkpoint.exists() or expected_checkpoint.is_symlink()
    if not released:
        if journal_exists or checkpoint_exists:
            raise AssetCompositionStateError(
                "Asset usd-cli command receipt evidence has no released lifecycle seal"
            )
        return

    released_by_size: dict[int, tuple[ArtifactBinding, ArtifactBinding]] = {}
    for pair in released:
        journal, _checkpoint = pair
        prior = released_by_size.setdefault(journal.size_bytes, pair)
        if prior != pair:
            raise AssetCompositionStateError(
                "Asset usd-cli released command receipt history is ambiguous"
            )
    latest_journal, latest_checkpoint = max(
        released_by_size.values(),
        key=lambda item: item[0].size_bytes,
    )
    if latest_journal.size_bytes <= 0:
        raise AssetCompositionStateError(
            "Asset usd-cli released command receipt journal is empty"
        )
    _bind_asset_usd_cli_lifecycle_artifact(
        run_root,
        path=latest_journal.path,
        sha256=latest_journal.sha256,
        max_bytes=MAX_ASSET_USD_CLI_RECEIPT_JOURNAL_BYTES,
        expected_size_bytes=latest_journal.size_bytes,
        expected_path=expected_journal,
        label="command receipt journal",
    )
    _bind_asset_usd_cli_lifecycle_artifact(
        run_root,
        path=latest_checkpoint.path,
        sha256=latest_checkpoint.sha256,
        max_bytes=MAX_USD_CLI_RECEIPT_CHECKPOINT_BYTES,
        expected_size_bytes=latest_checkpoint.size_bytes,
        expected_path=expected_checkpoint,
        label="command receipt checkpoint",
    )
    checkpoint_observed = read_contained_artifact(
        run_root,
        latest_checkpoint.path,
        max_bytes=MAX_USD_CLI_RECEIPT_CHECKPOINT_BYTES,
        capture_bytes=True,
    )
    assert checkpoint_observed.data is not None
    try:
        checkpoint_payload = json.loads(checkpoint_observed.data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AssetCompositionStateError(
            "Asset usd-cli command receipt checkpoint is invalid"
        ) from exc
    if (
        not isinstance(checkpoint_payload, dict)
        or checkpoint_payload.get("schema_version")
        != "content-agent-workflows.usd-cli-receipt-checkpoint.v1"
        or checkpoint_payload.get("workflow") != "asset.run"
        or checkpoint_payload.get("receipt_sha256") != latest_journal.sha256
        or checkpoint_payload.get("receipt_size_bytes") != latest_journal.size_bytes
    ):
        raise AssetCompositionStateError(
            "Asset usd-cli command receipt checkpoint does not seal the journal"
        )


def _require_safe_asset_usd_cli_teardown_history(
    run_dir: Path,
    *,
    run_id: str,
    terminal_valid: bool,
    reserve_launch_slot: bool = False,
) -> tuple[ArtifactBinding, ...]:
    """Fail resume/review closed on stale state or any failed prior release."""

    for state_name in (".usd-cli", ".ov", ".3dsc"):
        state_path = run_dir / state_name
        if state_path.exists() or state_path.is_symlink():
            raise AssetCompositionStateError(
                f"Prior asset usd-cli state remains at {state_path}; start a new "
                "run after ownership-safe cleanup"
            )
    raw_dir = run_dir / "raw"
    if not raw_dir.is_dir():
        if terminal_valid:
            raise AssetCompositionStateError(
                "Terminal asset run is missing its usd-cli teardown receipt"
            )
        return ()
    intents: dict[str, AssetUsdCliLaunchIntent] = {}
    lifecycle_bindings: list[ArtifactBinding] = []
    intent_paths = sorted(raw_dir.glob("asset_usd_cli_launch_*.json"))
    if len(intent_paths) > MAX_ASSET_USD_CLI_LIFECYCLE_RECORDS or (
        reserve_launch_slot and len(intent_paths) >= MAX_ASSET_USD_CLI_LIFECYCLE_RECORDS
    ):
        raise AssetCompositionStateError(
            "Asset usd-cli launch history has no bounded slot for another launch; "
            "start a new empty run and do not resume this identity"
        )
    for path in intent_paths:
        try:
            observed = read_contained_artifact(
                run_dir,
                path,
                max_bytes=MAX_ASSET_USD_CLI_TEARDOWN_RECEIPT_BYTES,
                capture_bytes=True,
            )
            assert observed.data is not None
            intent = AssetUsdCliLaunchIntent.model_validate_json(observed.data)
        except Exception as exc:  # noqa: BLE001 - malformed custody blocks reuse
            raise AssetCompositionStateError(
                f"Asset usd-cli launch intent is invalid: {path}"
            ) from exc
        if path.name != f"asset_usd_cli_launch_{intent.launch_id}.json":
            raise AssetCompositionStateError(
                f"Asset usd-cli launch intent path is inconsistent: {path}"
            )
        if intent.run_id != run_id:
            raise AssetCompositionStateError(
                f"Asset usd-cli launch intent belongs to another run: {path}"
            )
        if intent.launch_id in intents:
            raise AssetCompositionStateError(
                f"Asset usd-cli launch identity is duplicated: {intent.launch_id}"
            )
        intents[intent.launch_id] = intent
        lifecycle_bindings.append(
            ArtifactBinding(
                path=str(observed.path),
                sha256=observed.sha256,
                size_bytes=observed.size_bytes,
            )
        )
    receipts: dict[str, AssetUsdCliTeardownReceipt] = {}
    receipt_paths = sorted(raw_dir.glob("asset_usd_cli_teardown_*.json"))
    if len(receipt_paths) > MAX_ASSET_USD_CLI_LIFECYCLE_RECORDS or (
        reserve_launch_slot
        and len(receipt_paths) >= MAX_ASSET_USD_CLI_LIFECYCLE_RECORDS
    ):
        raise AssetCompositionStateError(
            "Asset usd-cli teardown history has no bounded slot for another launch; "
            "start a new empty run and do not resume this identity"
        )
    for path in receipt_paths:
        try:
            observed = read_contained_artifact(
                run_dir,
                path,
                max_bytes=MAX_ASSET_USD_CLI_TEARDOWN_RECEIPT_BYTES,
                capture_bytes=True,
            )
            assert observed.data is not None
            receipt = AssetUsdCliTeardownReceipt.model_validate_json(observed.data)
        except Exception as exc:  # noqa: BLE001 - malformed custody blocks reuse
            raise AssetCompositionStateError(
                f"Asset usd-cli teardown receipt is invalid: {path}"
            ) from exc
        if receipt.status == "failed":
            raise AssetCompositionStateError(
                "A prior asset usd-cli teardown failed; this run is not resumable "
                f"({path})"
            )
        if path.name != f"asset_usd_cli_teardown_{receipt.launch_id}.json":
            raise AssetCompositionStateError(
                f"Asset usd-cli teardown receipt path is inconsistent: {path}"
            )
        if receipt.run_id != run_id:
            raise AssetCompositionStateError(
                f"Asset usd-cli teardown receipt belongs to another run: {path}"
            )
        if receipt.launch_id in receipts:
            raise AssetCompositionStateError(
                f"Asset usd-cli teardown identity is duplicated: {receipt.launch_id}"
            )
        receipts[receipt.launch_id] = receipt
        lifecycle_bindings.append(
            ArtifactBinding(
                path=str(observed.path),
                sha256=observed.sha256,
                size_bytes=observed.size_bytes,
            )
        )
        if receipt.session_identity_path is not None:
            session_identity_sha256 = receipt.session_identity_sha256
            if session_identity_sha256 is None:
                raise AssetCompositionStateError(
                    "Asset usd-cli session identity digest is missing"
                )
            lifecycle_bindings.append(
                _bind_asset_usd_cli_lifecycle_artifact(
                    run_dir,
                    path=receipt.session_identity_path,
                    sha256=session_identity_sha256,
                    max_bytes=MAX_PARENT_USD_CLI_SESSION_IDENTITY_BYTES,
                    expected_path=(
                        run_dir.resolve(strict=True)
                        / "raw"
                        / f"asset_usd_cli_session_{receipt.launch_id}.json"
                    ),
                    label="session identity",
                )
            )
        if receipt.daemon_log_path is not None:
            daemon_log_sha256 = receipt.daemon_log_sha256
            if daemon_log_sha256 is None:
                raise AssetCompositionStateError(
                    "Asset usd-cli preserved daemon log digest is missing"
                )
            lifecycle_bindings.append(
                _bind_asset_usd_cli_lifecycle_artifact(
                    run_dir,
                    path=receipt.daemon_log_path,
                    sha256=daemon_log_sha256,
                    max_bytes=MAX_ASSET_USD_CLI_DAEMON_LOG_BYTES,
                    expected_name_pattern=r"usd_cli_daemon_[0-9a-f]{16}\.log",
                    label="preserved daemon log",
                )
            )
    missing_receipts = sorted(set(intents) - set(receipts))
    if missing_receipts:
        raise AssetCompositionStateError(
            "A prior asset usd-cli launch has no teardown receipt; this run is "
            f"not resumable ({', '.join(missing_receipts)})"
        )
    missing_intents = sorted(set(receipts) - set(intents))
    if missing_intents:
        raise AssetCompositionStateError(
            "A prior asset usd-cli teardown has no launch intent; this run is "
            f"not resumable ({', '.join(missing_intents)})"
        )
    _verify_latest_asset_usd_cli_command_receipt_evidence(
        run_dir,
        receipts=receipts,
    )
    if terminal_valid and not any(
        receipt.status == "released" for receipt in receipts.values()
    ):
        raise AssetCompositionStateError(
            "Terminal asset run has no proven released usd-cli lifecycle receipt"
        )
    return tuple(sorted(lifecycle_bindings, key=lambda binding: binding.path))


def _verify_asset_usd_cli_lifecycle_history(
    run_dir: Path,
    *,
    bindings: tuple[ArtifactBinding, ...],
) -> None:
    """Re-attest the complete pre-child lifecycle record set and exact bytes."""

    raw_dir = run_dir / "raw"
    expected_paths = tuple(sorted(Path(binding.path) for binding in bindings))
    expected_record_paths = tuple(
        path
        for path in expected_paths
        if re.fullmatch(
            r"asset_usd_cli_(?:launch|teardown)_[A-Za-z0-9][A-Za-z0-9._-]*\.json",
            path.name,
        )
    )

    def current_paths() -> tuple[Path, ...]:
        if not raw_dir.is_dir():
            return ()
        intent_paths = tuple(raw_dir.glob("asset_usd_cli_launch_*.json"))
        receipt_paths = tuple(raw_dir.glob("asset_usd_cli_teardown_*.json"))
        if (
            len(intent_paths) > MAX_ASSET_USD_CLI_LIFECYCLE_RECORDS
            or len(receipt_paths) > MAX_ASSET_USD_CLI_LIFECYCLE_RECORDS
        ):
            raise RuntimeError("asset usd-cli lifecycle history exceeds its bound")
        return tuple(sorted((*intent_paths, *receipt_paths)))

    if current_paths() != expected_record_paths:
        raise RuntimeError("asset usd-cli lifecycle record set changed")
    for binding in bindings:
        binding_path = Path(binding.path)
        if re.fullmatch(r"usd_cli_daemon_[0-9a-f]{16}\.log", binding_path.name):
            max_bytes = MAX_ASSET_USD_CLI_DAEMON_LOG_BYTES
        elif re.fullmatch(
            r"asset_usd_cli_session_[A-Za-z0-9][A-Za-z0-9._-]*\.json",
            binding_path.name,
        ):
            max_bytes = MAX_PARENT_USD_CLI_SESSION_IDENTITY_BYTES
        elif binding_path.name == "usd_cli_command_receipts.jsonl":
            max_bytes = MAX_ASSET_USD_CLI_RECEIPT_JOURNAL_BYTES
        elif binding_path.name == "usd_cli_command_receipts.checkpoint.json":
            max_bytes = MAX_USD_CLI_RECEIPT_CHECKPOINT_BYTES
        else:
            max_bytes = MAX_ASSET_USD_CLI_TEARDOWN_RECEIPT_BYTES
        observed = read_contained_artifact(
            run_dir,
            binding.path,
            max_bytes=max_bytes,
        )
        if (
            observed.sha256 != binding.sha256
            or observed.size_bytes != binding.size_bytes
        ):
            raise RuntimeError(
                f"asset usd-cli lifecycle artifact changed: {binding.path}"
            )
    if current_paths() != expected_record_paths:
        raise RuntimeError("asset usd-cli lifecycle record set changed during readback")


def _write_parent_usd_cli_session_identity(
    *,
    config: AssetRunConfig,
    request: AssetRunRequest,
    launch_id: str,
    route: UsdCliTelemetryRoute,
    lease: UsdCliDaemonLease,
    session: WorkflowUsdCliSession,
    readiness: UsdCliReadiness,
) -> tuple[ParentUsdCliSessionIdentity, Path, str]:
    source_staging = request.source_staging
    if source_staging is not None:
        source_identity: (
            ParentUsdCliStagedSourceIdentity | ParentUsdCliGeneratedSourceIdentity
        ) = ParentUsdCliStagedSourceIdentity(
            original_source=_parent_artifact_identity(source_staging.original_source),
            staged_source=_parent_artifact_identity(source_staging.staged_source),
            staging_manifest=_parent_artifact_identity(source_staging.manifest),
            dependency_digest_set_sha256=(source_staging.dependency_digest_set_sha256),
        )
    elif request.source_mode == "cad_modeling":
        source_identity = ParentUsdCliGeneratedSourceIdentity(
            source_intent=_parent_artifact_identity(
                _frozen_file_binding(Path(request.source_asset))
            ),
            source_images=[
                _parent_artifact_identity(binding)
                for binding in request.source_image_bindings
            ],
        )
    else:
        raise RuntimeError("Asset request has no typed source identity")
    if route.wrapper_path is None or route.target_path is None:
        raise RuntimeError("Asset usd-cli route is incomplete")
    if readiness.artifact_path is None:
        raise RuntimeError("Asset usd-cli readiness omitted its evidence artifact")
    if lease.identity.instance_id is None:
        raise RuntimeError("Asset usd-cli daemon omitted its instance identity")
    if lease.identity.process_group_id is None or lease.identity.os_session_id is None:
        raise RuntimeError("Asset usd-cli daemon omitted its complete OS identity")
    try:
        server_state = json.loads(lease.server_state_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Asset usd-cli daemon state is invalid") from exc
    if (
        not isinstance(server_state, dict)
        or server_state.get("host") != "127.0.0.1"
        or server_state.get("lifecycle_owner") != "external"
        or not isinstance(server_state.get("port"), int)
        or isinstance(server_state.get("port"), bool)
    ):
        raise RuntimeError("Asset usd-cli daemon endpoint is invalid")
    identity = ParentUsdCliSessionIdentity(
        created_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        launch_id=launch_id,
        run_id=request.run_id,
        run_dir=request.run_dir,
        repository_root=str(config.repo_root),
        parent_session_id=session.session_id,
        project_id=lease.identity.project_id,
        instance_id=lease.identity.instance_id,
        daemon_identity_sha256=_daemon_identity_sha256(lease.identity),
        server_host="127.0.0.1",
        server_port=int(server_state["port"]),
        allowed_roots=[request.run_dir],
        source=source_identity,
        readiness_artifact=_parent_artifact_identity(
            _frozen_file_binding(readiness.artifact_path)
        ),
        launcher_implementation=_parent_artifact_identity(
            _frozen_file_binding(Path(__file__).resolve())
        ),
        usd_cli_version=readiness.version,
        usd_cli_source_revision=readiness.source_revision,
        usd_cli_wrapper=str(route.wrapper_path.resolve()),
        usd_cli_executable=str(route.target_path.resolve()),
    )
    identity_path = (
        Path(request.run_dir) / "raw" / f"asset_usd_cli_session_{launch_id}.json"
    )
    atomic_write_json(identity_path, identity, within=request.run_dir)
    identity_sha256 = file_sha256(identity_path)
    return identity, identity_path, identity_sha256


def _parent_usd_cli_session_integrity_errors(
    run_dir: Path,
    *,
    expected_identity: ParentUsdCliSessionIdentity | None,
    identity_path: Path | None,
    expected_sha256: str | None,
) -> list[str]:
    """Re-attest the complete parent handoff after child write access ends."""

    expected_values = (expected_identity, identity_path, expected_sha256)
    if all(value is None for value in expected_values):
        return []
    if any(value is None for value in expected_values):
        return ["parent session attestation is incomplete"]
    assert expected_identity is not None
    assert identity_path is not None
    assert expected_sha256 is not None
    try:
        observed = read_contained_artifact(
            run_dir,
            identity_path,
            max_bytes=MAX_PARENT_USD_CLI_SESSION_IDENTITY_BYTES,
            capture_bytes=True,
        )
        if observed.sha256 != expected_sha256:
            raise ValueError("identity digest changed")
        assert observed.data is not None
        current_identity = ParentUsdCliSessionIdentity.model_validate_json(
            observed.data
        )
        if current_identity != expected_identity:
            raise ValueError("identity fields changed")
    except Exception as exc:  # noqa: BLE001 - any custody loss fails release closed
        return [f"parent session attestation: {type(exc).__name__}: {exc}"]
    return []


def _prompt_with_parent_usd_cli_session(
    prompt: str,
    *,
    identity: ParentUsdCliSessionIdentity,
    identity_path: Path,
    identity_sha256: str,
) -> str:
    if isinstance(identity.source, ParentUsdCliStagedSourceIdentity):
        source_summary = f"- Staged source: `{identity.source.staged_source.path}`\n"
    else:
        source_summary = (
            f"- Generated-source intent: `{identity.source.source_intent.path}`\n"
            f"- Frozen source images: {len(identity.source.source_images)}\n"
            "- Initial USD: not available until the CAD-modeling stage completes\n"
        )
    return (
        prompt.rstrip()
        + "\n\n"
        + "## Parent-owned usd-cli session\n\n"
        + f"- Typed identity: `{identity_path}`\n"
        + f"- Identity SHA-256: `{identity_sha256}`\n"
        + source_summary
        + f"- Allowed root: `{identity.run_dir}`\n\n"
        + "The launcher already owns the only usd-cli daemon. Reuse the typed "
        + "session contract for every domain command. Direct usd-cli-tel commands must "
        + "omit `--server` so the pinned attached-project route can recover the "
        + "parent daemon token, and must include "
        + f"`--session {identity.parent_session_id}` before the verb. Do not start, stop, or "
        + "replace a daemon, do not widen allowed roots, and do not request or "
        + "copy renderer credentials. The launcher tears down all child sessions "
        + "at the terminal or human-review boundary.\n"
    )


def _initialize_parent_usd_cli_session(
    session: WorkflowUsdCliSession,
    *,
    source_asset: Path | None,
) -> None:
    """Open only inspectable USD sources in the parent-owned named session."""

    if source_asset is not None:
        session.open(source_asset)
        return
    # CAD, mesh, URDF, MJCF, and source-free inputs do not become inspectable
    # USD until their Geometry/CAD workflow publishes one.
    session.run_json(["info"])


def _asset_execution_boundary(
    run_state_path: Path,
    *,
    interrupted: bool,
    setup_failed: bool,
) -> Literal[
    "completed",
    "human_review",
    "failed",
    "cancelled",
    "incomplete",
    "setup_failed",
]:
    if interrupted:
        return "cancelled"
    if setup_failed:
        return "setup_failed"
    try:
        run = load_verified_run(run_state_path)
    except AssetCompositionStateError:
        return "failed"
    if run.terminal_status == "completed":
        return "completed"
    if run.terminal_status == "failed":
        return "failed"
    if run.terminal_status == "cancelled":
        return "cancelled"
    if (
        run.selected_mode == "agentic"
        and run.coordinator.next_action == "finalize_receipts"
    ):
        return "completed"
    if (
        run.current_stage == "articulation"
        and run.stages["articulation"].status == "needs_review"
    ):
        return "human_review"
    return "incomplete"


def _asset_usd_cli_readiness(
    config: AssetRunConfig,
    *,
    run_dir: Path,
    route: UsdCliTelemetryRoute,
    workflow_session: WorkflowUsdCliSession,
    launch_id: str,
) -> UsdCliReadiness:
    """Prepare usd-cli without making an agentic pre-selection provider call."""

    if config.selected_mode == "compatibility_fixed":
        if route.target_path is None:  # pragma: no cover - caller validates route
            raise RuntimeError("Asset usd-cli route is incomplete")
        return ensure_usd_cli_ovrtx_ready(
            config.repo_root,
            run_dir=run_dir,
            executable=route.target_path,
            timeout_seconds=max(config.scene_tool_timeout_seconds, 900.0),
            session=workflow_session,
            artifact_stem=f"asset_usd_cli_probe_{launch_id}",
        )
    package_route = resolve_package_owned_usd_cli_route(config.repo_root)
    if route.target_path is None or route.target_path.resolve(strict=True) != (
        package_route.target
    ):
        raise RuntimeError("Agentic usd-cli route is not the package-owned target")
    process = run_bounded_usd_cli_subprocess(
        [str(package_route.target), "--version"],
        check=True,
        env=sanitized_usd_cli_execution_env(
            executable_dir=package_route.target.parent,
        ),
        timeout=min(config.scene_tool_timeout_seconds, 60.0),
    )
    version = process.stdout.strip()
    if not version:
        raise RuntimeError("Agentic usd-cli version check returned no identity")
    artifact_path = run_dir / "raw" / f"asset_usd_cli_probe_{launch_id}.json"
    probe = {
        "selected_mode": "agentic",
        "provider_readiness": "not_requested",
    }
    atomic_write_json(
        artifact_path,
        {
            "schema_version": ASSET_USD_CLI_PROVIDER_FREE_READINESS_SCHEMA_VERSION,
            "usd_cli_version": version,
            "usd_cli_source_revision": package_route.source_revision,
            "probe": probe,
        },
        within=run_dir,
    )
    return UsdCliReadiness(
        version=version,
        source_revision=package_route.source_revision,
        probe=probe,
        artifact_path=artifact_path,
    )


def _execute_with_parent_usd_cli_lifecycle(
    *,
    config: AssetRunConfig,
    request: AssetRunRequest,
    request_path: Path,
    run_state_path: Path,
    prompt_path: Path,
    prompt: str,
    child_output_path: Path,
    child_final_path: Path,
    suffix: str,
    invocation_mode: Literal["interactive", "batch"],
    reasoning_loop: AssetReasoningLoop | None,
) -> AssetRunResult:
    if invocation_mode == "interactive" and reasoning_loop is None:
        raise ValueError("Interactive asset lifecycle requires one reasoning loop")
    if invocation_mode == "batch" and reasoning_loop is not None:
        raise ValueError("Batch asset lifecycle owns its child reasoning adapter")
    child_returncode = 2
    interrupted = False
    run_dir = Path(request.run_dir)
    started = time.monotonic()
    launch_id = f"{suffix}-{time.time_ns()}-{secrets.token_hex(4)}"
    daemon_lease: UsdCliDaemonLease | None = None
    daemon_cleanup_lease: UsdCliDaemonCleanupEvidence | None = None
    session_identity: ParentUsdCliSessionIdentity | None = None
    session_identity_path: Path | None = None
    session_identity_sha256: str | None = None
    daemon_release: UsdCliDaemonTeardownEvidence | None = None
    coordinator_failed = False
    setup_failed = False
    setup_error: BaseException | None = None
    launch_intent: AssetUsdCliLaunchIntent | None = None
    launch_intent_binding: ArtifactBinding | None = None
    lifecycle_history_bindings: tuple[ArtifactBinding, ...] | None = None
    command_receipt_journal_binding: ArtifactBinding | None = None
    command_receipt_checkpoint_binding: ArtifactBinding | None = None
    teardown_receipt_path: Path | None = None

    def lifecycle_reasoning_loop(session: AssetCoordinatorSession) -> int:
        nonlocal child_returncode
        nonlocal coordinator_failed
        nonlocal daemon_lease
        nonlocal daemon_cleanup_lease
        nonlocal daemon_release
        nonlocal interrupted
        nonlocal session_identity
        nonlocal session_identity_path
        nonlocal session_identity_sha256
        nonlocal setup_error
        nonlocal setup_failed
        nonlocal launch_intent
        nonlocal launch_intent_binding
        nonlocal lifecycle_history_bindings
        nonlocal command_receipt_journal_binding
        nonlocal command_receipt_checkpoint_binding
        nonlocal teardown_receipt_path
        execution_config = config
        execution_prompt = prompt
        workflow_session: WorkflowUsdCliSession | None = None
        teardown_errors: list[str] = []
        teardown_receipt_binding: ArtifactBinding | None = None

        def record_teardown_exception(label: str, exc: BaseException) -> None:
            nonlocal child_returncode
            nonlocal interrupted
            if isinstance(exc, KeyboardInterrupt):
                interrupted = True
                child_returncode = 130
            teardown_errors.append(f"{label}: {type(exc).__name__}: {exc}")

        try:
            prior_lifecycle_bindings = _require_safe_asset_usd_cli_teardown_history(
                run_dir,
                run_id=request.run_id,
                terminal_valid=False,
                reserve_launch_slot=True,
            )
            # The coordinator lease must cover every prompt mutation. A competing
            # resume therefore cannot clobber the active owner's prompt artifact.
            atomic_write_text(prompt_path, prompt, within=run_dir)
            launch_intent = AssetUsdCliLaunchIntent(
                created_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                launch_id=launch_id,
                run_id=request.run_id,
            )
            launch_intent_path = (
                run_dir / "raw" / f"asset_usd_cli_launch_{launch_id}.json"
            )
            atomic_write_json(
                launch_intent_path,
                launch_intent,
                within=run_dir,
            )
            launch_intent_binding = _frozen_file_binding(launch_intent_path)
            lifecycle_history_bindings = (
                *prior_lifecycle_bindings,
                launch_intent_binding,
            )
            pre_reasoning_staged_errors = _staged_input_integrity_errors(run_dir)
            if pre_reasoning_staged_errors:
                raise RuntimeError(
                    "Staged source failed pre-child integrity: "
                    + "; ".join(pre_reasoning_staged_errors)
                )
            route = _prepare_usd_cli_telemetry_route(
                run_dir,
                repo_root=config.repo_root,
            )
            if not route.active or route.target_path is None:
                raise RuntimeError(
                    route.reason or "The package-owned usd-cli route is unavailable"
                )
            idle_timeout_seconds = (
                max(3600, int(config.child_timeout_seconds) + 1800)
                if config.child_timeout_seconds > 0
                else USD_CLI_ASSET_MAX_IDLE_TIMEOUT_SECONDS
            )
            daemon_lease = _start_usd_cli_run_daemon_strict(
                route=route,
                run_dir=run_dir,
                idle_timeout_seconds=idle_timeout_seconds,
            )
            workflow_session = _attach_workflow_usd_cli_session(
                repo_root=config.repo_root,
                run_dir=run_dir,
                workflow="asset.run",
            )
            readiness = _asset_usd_cli_readiness(
                config,
                run_dir=run_dir,
                route=route,
                workflow_session=workflow_session,
                launch_id=launch_id,
            )
            # ``render-probe`` is a parent-side readiness command. Provided-source
            # runs establish the named session with the staged USD immediately.
            # Source-free CAD runs reuse that same named session after the CAD
            # stage creates its first USD. In both cases, remove parent-only
            # renderer credentials before the child can access the run directory.
            _initialize_parent_usd_cli_session(
                workflow_session,
                source_asset=(
                    Path(request.source_asset)
                    if request.source_staging is not None
                    and _is_usd_source_path(Path(request.source_asset))
                    else None
                ),
            )
            _activate_usd_cli_child_config(daemon_lease)
            (
                session_identity,
                session_identity_path,
                session_identity_sha256,
            ) = _write_parent_usd_cli_session_identity(
                config=config,
                request=request,
                launch_id=launch_id,
                route=route,
                lease=daemon_lease,
                session=workflow_session,
                readiness=readiness,
            )
            execution_prompt = _prompt_with_parent_usd_cli_session(
                prompt,
                identity=session_identity,
                identity_path=session_identity_path,
                identity_sha256=session_identity_sha256,
            )
            atomic_write_text(prompt_path, execution_prompt, within=run_dir)
            execution_config = replace(
                config,
                parent_usd_cli_session_identity=session_identity_path,
                parent_usd_cli_session_identity_sha256=session_identity_sha256,
            )
            session_identity_artifact = _frozen_file_binding(session_identity_path)
            if session_identity_artifact.sha256 != session_identity_sha256:
                raise RuntimeError(
                    "Parent usd-cli session identity changed before reasoning"
                )
            bound_session = replace(
                session,
                parent_usd_cli_session_identity=session_identity,
                parent_usd_cli_session_identity_artifact=(session_identity_artifact),
            )
            if reasoning_loop is None:
                child_returncode = _run_child_agent(
                    config=execution_config,
                    prompt=execution_prompt,
                    run_dir=run_dir,
                    child_output_path=child_output_path,
                    child_final_path=child_final_path,
                    scene_service=None,
                    prompt_image_inputs=[
                        {
                            "label": f"CAD source image {index}",
                            "path": str(path),
                        }
                        for index, path in enumerate(config.source_images, start=1)
                    ],
                    bridge_artifact_prefix="asset_composition",
                )
            else:
                callback_returncode = reasoning_loop(bound_session)
                child_returncode = (
                    0 if callback_returncode is None else callback_returncode
                )
        except KeyboardInterrupt:
            interrupted = True
            child_returncode = 130
            error = RuntimeError("KeyboardInterrupt during asset workflow")
            if session_identity is None:
                setup_failed = True
                setup_error = error
            _append_child_runner_error(child_output_path, error, run_dir=run_dir)
        except ChildProcessInterrupted as exc:
            interrupted = True
            child_returncode = 130
            _append_child_runner_error(child_output_path, exc, run_dir=run_dir)
        except UsdCliDaemonStrictStartError as exc:
            setup_failed = True
            setup_error = exc
            daemon_release = exc.teardown_evidence
            daemon_cleanup_lease = exc.cleanup_lease
            interrupted = exc.interrupted
            child_returncode = 130 if interrupted else 2
            if daemon_release is None and daemon_cleanup_lease is None:
                teardown_errors.append(
                    "strict daemon startup cleanup could not be proven"
                )
            _append_child_runner_error(child_output_path, exc, run_dir=run_dir)
        except Exception as exc:  # noqa: BLE001 - preserve resumable state and logs
            setup_failed = session_identity is None
            setup_error = exc if setup_failed else None
            child_returncode = 2
            _append_child_runner_error(child_output_path, exc, run_dir=run_dir)
        finally:
            # Release the exact daemon/process-group authority before any slower
            # state or artifact hashing so an interrupt in those later steps
            # cannot strand the parent-owned sidecar.
            teardown_lease = daemon_lease or daemon_cleanup_lease
            if teardown_lease is not None and daemon_release is None:
                try:
                    daemon_release = _stop_usd_cli_run_daemon_strict(
                        lease=teardown_lease
                    )
                except KeyboardInterrupt:
                    interrupted = True
                    child_returncode = 130
                    try:
                        daemon_release = _stop_usd_cli_run_daemon_strict(
                            lease=teardown_lease
                        )
                    except KeyboardInterrupt as retry_exc:
                        record_teardown_exception(
                            "usd-cli teardown retry after KeyboardInterrupt",
                            retry_exc,
                        )
                    except Exception as retry_exc:  # noqa: BLE001 - release is a gate
                        record_teardown_exception(
                            "usd-cli teardown retry after KeyboardInterrupt",
                            retry_exc,
                        )
                except Exception as exc:  # noqa: BLE001 - release is a hard gate
                    record_teardown_exception("usd-cli teardown", exc)
            if workflow_session is not None:
                try:
                    # Attached children never append to the launcher's receipt
                    # journal. Verify its pinned inode and digest after their run
                    # directory access and before accepting lifecycle release.
                    workflow_session.verify_receipt_journal_integrity()
                    receipt_file = workflow_session.receipt_file
                    checkpoint_file = workflow_session.receipt_checkpoint_file
                    receipt_exists = receipt_file.exists() or receipt_file.is_symlink()
                    checkpoint_exists = (
                        checkpoint_file.exists() or checkpoint_file.is_symlink()
                    )
                    if receipt_exists != checkpoint_exists:
                        raise RuntimeError(
                            "parent usd-cli receipt journal/checkpoint pair is incomplete"
                        )
                    if receipt_exists:
                        (
                            command_receipt_journal_binding,
                            command_receipt_checkpoint_binding,
                        ) = _capture_asset_usd_cli_command_receipt_evidence(
                            run_dir,
                            workflow_session,
                        )
                        lifecycle_history_bindings = (
                            *(lifecycle_history_bindings or ()),
                            command_receipt_journal_binding,
                            command_receipt_checkpoint_binding,
                        )
                except BaseException as exc:  # evidence failure cannot skip release
                    record_teardown_exception(
                        "parent usd-cli receipt journal integrity",
                        exc,
                    )
            try:
                teardown_errors.extend(
                    _parent_usd_cli_session_integrity_errors(
                        run_dir,
                        expected_identity=session_identity,
                        identity_path=session_identity_path,
                        expected_sha256=session_identity_sha256,
                    )
                )
            except BaseException as exc:  # attestation cannot bypass final receipt
                record_teardown_exception("parent session attestation", exc)
            staged_errors: list[str] = []
            try:
                staged_errors = _staged_input_integrity_errors(run_dir)
                teardown_errors.extend(
                    f"staged source integrity: {error}" for error in staged_errors
                )
            except BaseException as exc:  # attestation cannot bypass final receipt
                staged_errors = ["attestation did not complete"]
                record_teardown_exception("staged source integrity", exc)
            if lifecycle_history_bindings is not None:
                try:
                    _verify_asset_usd_cli_lifecycle_history(
                        run_dir,
                        bindings=lifecycle_history_bindings,
                    )
                except BaseException as exc:  # custody cannot bypass final receipt
                    record_teardown_exception(
                        "asset lifecycle history integrity",
                        exc,
                    )
            if launch_intent is not None and launch_intent_binding is not None:
                try:
                    _verify_asset_usd_cli_launch_intent(
                        run_dir,
                        binding=launch_intent_binding,
                        expected=launch_intent,
                    )
                except BaseException as exc:  # custody cannot bypass final receipt
                    record_teardown_exception(
                        "asset launch intent integrity",
                        exc,
                    )

            if teardown_errors:
                coordinator_failed = True
                child_returncode = 2
            try:
                _record_unfinished_child_exit(
                    run_state_path,
                    returncode=child_returncode,
                    interrupted=interrupted,
                    elapsed_seconds=time.monotonic() - started,
                )
            except BaseException as exc:  # state recording cannot bypass receipt
                record_teardown_exception("asset child terminal-state recording", exc)
                coordinator_failed = True
                child_returncode = 2

            status: Literal["not_started", "released", "failed"]
            if teardown_errors:
                status = "failed"
            elif daemon_release is not None:
                status = "released"
            else:
                status = "not_started"
            daemon_identity = (
                daemon_lease.identity
                if daemon_lease is not None
                else (
                    daemon_release.identity
                    if daemon_release is not None
                    else (
                        daemon_cleanup_lease.identities[0]
                        if daemon_cleanup_lease is not None
                        and len(daemon_cleanup_lease.identities) == 1
                        else None
                    )
                )
            )
            try:
                if launch_intent_binding is None:
                    raise RuntimeError(
                        "asset usd-cli launch intent was not durably recorded"
                    )
                receipt = AssetUsdCliTeardownReceipt(
                    created_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                    launch_id=launch_id,
                    run_id=request.run_id,
                    session_identity_path=(
                        str(session_identity_path)
                        if session_identity_path is not None
                        else None
                    ),
                    session_identity_sha256=session_identity_sha256,
                    daemon_identity_sha256=(
                        _daemon_identity_sha256(daemon_identity)
                        if daemon_identity is not None
                        else None
                    ),
                    daemon_was_started=(
                        daemon_release.daemon_was_started
                        if daemon_release is not None
                        else daemon_identity is not None
                    ),
                    status=status,
                    boundary=_asset_execution_boundary(
                        run_state_path,
                        interrupted=interrupted,
                        setup_failed=setup_failed,
                    ),
                    child_returncode=child_returncode,
                    interrupted=interrupted,
                    process_released=(
                        daemon_release.process_released
                        if daemon_release is not None
                        else None
                    ),
                    descendants_released=(
                        daemon_release.descendants_released
                        if daemon_release is not None
                        else None
                    ),
                    sessions_released=(
                        daemon_release.sessions_released
                        if daemon_release is not None
                        else None
                    ),
                    listener_released=(
                        daemon_release.listener_released
                        if daemon_release is not None
                        else None
                    ),
                    daemon_leases_released=(
                        daemon_release.daemon_leases_released
                        if daemon_release is not None
                        else None
                    ),
                    state_directory_released=(
                        daemon_release.state_directory_released
                        if daemon_release is not None
                        else None
                    ),
                    source_integrity_verified=not staged_errors,
                    listener_host=(
                        daemon_release.host if daemon_release is not None else None
                    ),
                    listener_port=(
                        daemon_release.port if daemon_release is not None else None
                    ),
                    daemon_log_path=(
                        str(daemon_release.daemon_log_path)
                        if daemon_release is not None
                        and daemon_release.daemon_log_path is not None
                        else None
                    ),
                    daemon_log_sha256=(
                        daemon_release.daemon_log_sha256
                        if daemon_release is not None
                        else None
                    ),
                    command_receipt_journal=command_receipt_journal_binding,
                    command_receipt_checkpoint=command_receipt_checkpoint_binding,
                    setup_error=(
                        f"{type(setup_error).__name__}: {setup_error}"
                        if setup_error is not None
                        else None
                    ),
                    errors=teardown_errors,
                )
                teardown_receipt_path = (
                    run_dir / "raw" / f"asset_usd_cli_teardown_{launch_id}.json"
                )
                atomic_write_json(
                    teardown_receipt_path,
                    receipt,
                    within=run_dir,
                )
            except BaseException as exc:  # receipt failure cannot skip fallback
                if isinstance(exc, KeyboardInterrupt):
                    interrupted = True
                    child_returncode = 130
                receipt_error = f"teardown receipt {type(exc).__name__}: {exc}"
                teardown_errors.append(receipt_error)
                # A construction failure must still leave a schema-valid terminal
                # receipt paired with the durable launch intent. Keep only facts
                # that cannot themselves invalidate the fallback model.
                if launch_intent_binding is not None:
                    fallback_daemon_started = daemon_identity is not None and (
                        daemon_release.daemon_was_started
                        if daemon_release is not None
                        else True
                    )
                    fallback_daemon_sha256 = (
                        _daemon_identity_sha256(daemon_identity)
                        if daemon_identity is not None and fallback_daemon_started
                        else None
                    )
                    fallback = AssetUsdCliTeardownReceipt(
                        created_at=(
                            datetime.now(UTC).isoformat().replace("+00:00", "Z")
                        ),
                        launch_id=launch_id,
                        run_id=request.run_id,
                        session_identity_path=(
                            str(session_identity_path)
                            if session_identity_path is not None
                            and session_identity_sha256 is not None
                            else None
                        ),
                        session_identity_sha256=(
                            session_identity_sha256
                            if session_identity_path is not None
                            and session_identity_sha256 is not None
                            else None
                        ),
                        daemon_identity_sha256=fallback_daemon_sha256,
                        daemon_was_started=fallback_daemon_started,
                        status="failed",
                        boundary=(
                            "cancelled"
                            if interrupted
                            else (
                                "setup_failed" if session_identity is None else "failed"
                            )
                        ),
                        child_returncode=child_returncode,
                        interrupted=interrupted,
                        source_integrity_verified=False,
                        setup_error=(
                            f"{type(setup_error).__name__}: {setup_error}"
                            if setup_error is not None
                            else None
                        ),
                        errors=[receipt_error],
                    )
                    teardown_receipt_path = (
                        run_dir / "raw" / f"asset_usd_cli_teardown_{launch_id}.json"
                    )
                    atomic_write_json(
                        teardown_receipt_path,
                        fallback,
                        within=run_dir,
                    )
            if teardown_receipt_path is not None:
                try:
                    teardown_receipt_binding = _frozen_file_binding(
                        teardown_receipt_path
                    )
                    lifecycle_history_bindings = (
                        *(lifecycle_history_bindings or ()),
                        teardown_receipt_binding,
                    )
                    _verify_asset_usd_cli_lifecycle_history(
                        run_dir,
                        bindings=lifecycle_history_bindings,
                    )
                except BaseException as exc:
                    record_teardown_exception(
                        "asset teardown receipt integrity",
                        exc,
                    )
            if teardown_errors:
                coordinator_failed = True
                child_returncode = 2
                error = RuntimeError(
                    "Asset usd-cli teardown failed: " + "; ".join(teardown_errors)
                )
                _append_child_runner_error(child_output_path, error, run_dir=run_dir)
            elif setup_error is not None:
                coordinator_failed = True
        graph_run = load_verified_run(run_state_path)
        if (
            graph_run.selected_mode == "agentic"
            and graph_run.coordinator.next_action == "finalize_receipts"
        ):
            if teardown_receipt_path is None:
                raise AssetCompositionStateError(
                    "Agentic graph reached receipt finalization without a parent "
                    "resource-release receipt"
                )
            if teardown_receipt_binding is None:
                raise AssetCompositionStateError(
                    "Agentic graph release receipt lacks an exact frozen binding"
                )
            teardown = AssetUsdCliTeardownReceipt.model_validate_json(
                teardown_receipt_path.read_text(encoding="utf-8")
            )
            if teardown.status != "released":
                raise AssetCompositionStateError(
                    "Agentic graph parent resources were not fully released"
                )
            if (
                teardown.command_receipt_journal is None
                or teardown.command_receipt_checkpoint is None
            ):
                raise AssetCompositionStateError(
                    "Agentic graph release lacks the sealed command receipt pair"
                )
            legacy_graph = (
                request.leaf_catalog is not None
                and request.leaf_catalog.schema_version
                == LEGACY_ASSET_LEAF_CATALOG_SCHEMA_VERSION
            )
            if legacy_graph:
                finalize_graph_run(
                    run_state_path,
                    resource_release_paths=[teardown_receipt_path],
                    actor="content-workflow-cli",
                )
            else:
                finalize_graph_run(
                    run_state_path,
                    parent_release_receipt_path=teardown_receipt_path,
                    parent_command_receipt_journal_path=(
                        teardown.command_receipt_journal.path
                    ),
                    parent_command_receipt_checkpoint_path=(
                        teardown.command_receipt_checkpoint.path
                    ),
                    expected_parent_release_receipt=teardown_receipt_binding,
                    expected_parent_command_receipt_journal=(
                        command_receipt_journal_binding
                    ),
                    expected_parent_command_receipt_checkpoint=(
                        command_receipt_checkpoint_binding
                    ),
                    actor="content-workflow-cli",
                )
        return child_returncode

    try:
        coordinator_entrypoint = (
            run_interactive_asset_coordinator
            if invocation_mode == "interactive"
            else run_batch_asset_coordinator
        )
        coordinator_result = coordinator_entrypoint(
            run_state_path,
            reasoning_loop=lifecycle_reasoning_loop,
        )
        child_returncode = coordinator_result.returncode
    except AssetCoordinatorLeaseError as exc:
        # The active owner holds every setup/state/teardown mutation. A competing
        # invocation returns without touching its run artifacts or durable state.
        print(f"Asset coordinator lease unavailable: {exc}", file=sys.stderr)
        return AssetRunResult(
            run_dir=run_dir,
            request_path=request_path,
            run_state_path=run_state_path,
            prompt_path=prompt_path,
            child_output_path=child_output_path,
            child_final_path=child_final_path,
            terminal_validation_path=None,
            returncode=2,
            completed=False,
            needs_review=False,
        )
    except Exception as exc:  # noqa: BLE001 - preserve resumable state and logs
        coordinator_failed = True
        if not interrupted:
            child_returncode = 2
        _append_child_runner_error(child_output_path, exc, run_dir=run_dir)
    terminal = validate_terminal(run_state_path)
    terminal_path = run_dir / "terminal_validation.json"
    atomic_write_json(terminal_path, terminal)
    try:
        run = load_verified_run(run_state_path)
    except AssetCompositionStateError as exc:
        _append_child_runner_error(child_output_path, exc, run_dir=run_dir)
        return AssetRunResult(
            run_dir=run_dir,
            request_path=request_path,
            run_state_path=run_state_path,
            prompt_path=prompt_path,
            child_output_path=child_output_path,
            child_final_path=child_final_path,
            terminal_validation_path=terminal_path,
            returncode=2,
            completed=False,
            needs_review=False,
        )
    needs_review = (
        not coordinator_failed
        and run.selected_mode == "compatibility_fixed"
        and run.current_stage == "articulation"
        and run.stages["articulation"].status == "needs_review"
    )
    returncode = child_returncode
    if terminal.valid and not coordinator_failed:
        returncode = 0
    elif needs_review and child_returncode == 0:
        returncode = 3
    elif child_returncode == 0 and not terminal.valid:
        returncode = 1
    return AssetRunResult(
        run_dir=run_dir,
        request_path=request_path,
        run_state_path=run_state_path,
        prompt_path=prompt_path,
        child_output_path=child_output_path,
        child_final_path=child_final_path,
        terminal_validation_path=terminal_path,
        returncode=returncode,
        completed=terminal.valid and not coordinator_failed,
        needs_review=needs_review,
    )


def _record_unfinished_child_exit(
    state_path: Path,
    *,
    returncode: int,
    interrupted: bool,
    elapsed_seconds: float,
) -> None:
    try:
        run = load_verified_run(state_path)
        if run.selected_mode == "agentic":
            leaf_id = run.current_leaf_id
            if (
                leaf_id is None
                or run.terminal_status != "active"
                or run.coordinator.next_action == "finalize_receipts"
            ):
                return
            # A launcher exit is not a descriptor-resolved native result. Leave
            # the exact running attempt resumable instead of fabricating a
            # failed/cancelled leaf receipt without its typed result/projector.
            return
        stage = run.current_stage
        if stage is None or run.terminal_status != "active":
            return
        state = run.stages[stage]
        if state.status == "needs_review":
            return
        reason = (
            f"Child agent was interrupted after {elapsed_seconds:.1f} seconds."
            if interrupted
            else (
                "Child agent exited before the composed workflow reached a "
                "terminal state."
                if returncode == 0
                else f"Child agent exited with code {returncode}."
            )
        )
        if interrupted:
            cancel_stage(state_path, stage, reason=reason, actor="content-workflow-cli")
        else:
            fail_stage(state_path, stage, reason=reason, actor="content-workflow-cli")
    except AssetCompositionStateError:
        # Preserve the original runner status. Any state-integrity error will be
        # reported by terminal validation and fail resume closed.
        return


def _is_usd_source_path(path: Path) -> bool:
    return path.suffix.lower() in USD_SOURCE_SUFFIXES


def _stage_immutable_source_file(
    *,
    source_path: Path,
    source_root: Path | None,
    run_dir: Path,
) -> tuple[Path, AssetSourceStaging]:
    """Freeze one dependency-closed non-USD source without USD APIs."""

    run_root = run_dir.resolve(strict=True)
    if source_root is not None and source_root_stages_whole_tree(source_path):
        # Opaque formats recursively freeze the explicit root, so a nested run
        # directory would become part of its own input walk. Parsed and
        # self-contained formats enumerate a finite closure instead.
        approved_root = source_root.expanduser().resolve(strict=True)
        if run_root == approved_root or run_root.is_relative_to(approved_root):
            raise ValueError(
                "Run directory must be outside the approved non-USD source root"
            )
    closure = discover_non_usd_source_closure(
        source_path,
        explicit_source_root=source_root,
    )
    original_dependencies = [_frozen_file_binding(path) for path in closure.files]
    original = next(
        binding
        for binding in original_dependencies
        if binding.path == str(closure.source)
    )
    staged_root = run_root / "inputs" / "asset_source"
    if staged_root.exists():
        raise ValueError(f"Refusing pre-existing staged input tree: {staged_root}")
    staged_root.mkdir(parents=True, exist_ok=False)
    staged_dependencies: list[ArtifactBinding] = []
    manifest_files: list[dict[str, object]] = []
    for source_binding in original_dependencies:
        source_file = Path(source_binding.path)
        relative = source_file.relative_to(closure.source_root)
        staged_file = staged_root / relative
        staged_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_file, staged_file)
        os.chmod(staged_file, 0o444)
        staged_binding = _frozen_file_binding(staged_file)
        if (
            staged_binding.sha256 != source_binding.sha256
            or staged_binding.size_bytes != source_binding.size_bytes
        ):
            raise ValueError(
                "Staged source bytes diverged from the approved non-USD source: "
                f"{source_file}"
            )
        staged_dependencies.append(staged_binding)
        manifest_files.append(
            {
                "source_path": source_binding.path,
                "staged_path": staged_binding.path,
                "relative_path": relative.as_posix(),
                "sha256": staged_binding.sha256,
                "size_bytes": staged_binding.size_bytes,
            }
        )
    staged_path = staged_root / closure.source.relative_to(closure.source_root)
    staged = next(
        binding for binding in staged_dependencies if binding.path == str(staged_path)
    )
    digest_payload = json.dumps(
        sorted(binding.sha256 for binding in staged_dependencies),
        separators=(",", ":"),
        ensure_ascii=True,
    )
    digest_set = hashlib.sha256(digest_payload.encode("utf-8")).hexdigest()
    manifest_path = run_root / "raw" / "staged_source_asset_source.json"
    atomic_write_json(
        manifest_path,
        {
            "schema_version": IMMUTABLE_SOURCE_CLOSURE_SCHEMA_VERSION,
            "source_path": original.path,
            "source_sha256": original.sha256,
            "staged_path": staged.path,
            "common_source_root": str(closure.source_root),
            "dependency_discovery_strategy": closure.strategy,
            "file_count": len(manifest_files),
            "total_size_bytes": sum(
                binding.size_bytes for binding in staged_dependencies
            ),
            "dependency_digest_set_sha256": digest_set,
            "unresolved_dependencies": [],
            "files": manifest_files,
            "self_containment": {
                "status": "verified",
                "escaped_paths": [],
            },
        },
        within=run_root,
    )
    return staged_path, AssetSourceStaging(
        original_source=original,
        original_dependencies=original_dependencies,
        staged_source=staged,
        staged_dependencies=staged_dependencies,
        manifest=_frozen_file_binding(manifest_path),
        dependency_digest_set_sha256=digest_set,
    )


def _source_staging_contract(staged: UsdCliStagedInput) -> AssetSourceStaging:
    """Bind both sides of one exact, dependency-closed source relocation."""

    try:
        payload = json.loads(staged.manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Staged source manifest is unreadable") from exc
    files = payload.get("files") if isinstance(payload, dict) else None
    unresolved = (
        payload.get("unresolved_dependencies") if isinstance(payload, dict) else None
    )
    if not isinstance(files, list) or not files:
        raise ValueError("Staged source manifest omitted its dependency closure")
    if unresolved != []:
        raise ValueError(
            "Asset source dependency closure is unresolved; refusing child launch: "
            f"{unresolved!r}"
        )
    original_dependencies: list[ArtifactBinding] = []
    staged_dependencies: list[ArtifactBinding] = []
    for item in files:
        if not isinstance(item, dict):
            raise ValueError("Staged source manifest contains a non-object file entry")
        try:
            source_path = str(item["source_path"])
            staged_path = str(item["staged_path"])
            sha256 = str(item["sha256"])
            size_bytes = int(item["size_bytes"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Staged source manifest contains malformed file identity"
            ) from exc
        original_dependencies.append(
            ArtifactBinding(path=source_path, sha256=sha256, size_bytes=size_bytes)
        )
        staged_dependencies.append(
            ArtifactBinding(path=staged_path, sha256=sha256, size_bytes=size_bytes)
        )
    original_source = next(
        (
            binding
            for binding in original_dependencies
            if binding.path == str(staged.source_usd_path)
        ),
        None,
    )
    staged_source = next(
        (
            binding
            for binding in staged_dependencies
            if binding.path == str(staged.staged_usd_path)
        ),
        None,
    )
    if original_source is None or staged_source is None:
        raise ValueError("Staged source manifest lost its root USD identity")
    manifest_binding = _frozen_file_binding(staged.manifest_path)
    return AssetSourceStaging(
        original_source=original_source,
        original_dependencies=original_dependencies,
        staged_source=staged_source,
        staged_dependencies=staged_dependencies,
        manifest=manifest_binding,
        dependency_digest_set_sha256=staged.dependency_digest_set_sha256,
    )


def _build_request(
    config: AssetRunConfig,
    *,
    run_id: str,
    run_dir: Path,
    run_state_path: Path,
    source_asset: Path | None = None,
    segmentation_run: AssetSegmentationRunBinding | None = None,
    source_staging: AssetSourceStaging | None = None,
) -> AssetRunRequest:
    if config.selected_mode == "compatibility_fixed" and config.materials_usd is None:
        raise ValueError("Validated asset config is missing the materials library")
    resolved_source_asset = source_asset or config.usd_path
    if resolved_source_asset is None:
        raise ValueError("Source-free request is missing its frozen source intent")
    fixed_geometry = (
        AssetGeometryRequest(
            target_profile=config.geometry_target_profile,
            optimization_policy=cast(Any, config.geometry_optimization_policy),
            optimizer_backend=cast(Any, config.geometry_optimizer_backend),
            repair_mode=cast(Any, config.geometry_repair_mode),
            repair_profile=cast(Any, config.geometry_repair_profile),
            render_evidence=config.geometry_render_evidence,
            segmentation_run=segmentation_run,
            segmentation_required=config.geometry_segmentation_required,
            segmentation_required_parts=list(config.geometry_required_parts),
        )
        if config.include_geometry_stage
        else None
    )
    created_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    runtime = AssetRuntimeRequest(
        runner=config.runner,
        model=config.model,
        model_reasoning_effort=config.model_reasoning_effort,
        scene_tool_timeout_seconds=config.scene_tool_timeout_seconds,
        child_timeout_seconds=config.child_timeout_seconds,
        codex_base_url=config.codex_base_url,
        codex_sandbox_mode=config.codex_sandbox_mode,
        codex_config=config.codex_config or {},
        claude_config=config.claude_config or {},
        claude_permission_mode=config.claude_permission_mode,
        claude_max_turns=config.claude_max_turns,
        claude_execution_mode=config.claude_execution_mode,
    )
    references = [
        _frozen_file_binding(path)
        for path in [*config.reference_images, *(config.reference_files or [])]
    ]
    cad_modeling = (
        _cad_modeling_request(config)
        if config.selected_mode == "compatibility_fixed" and config.usd_path is None
        else None
    )
    common: dict[str, object] = {
        "created_at": created_at,
        "coordinator_mode": "single_reasoning_loop",
        "run_id": run_id,
        "run_dir": str(run_dir),
        "run_state": str(run_state_path),
        "repository_root": str(config.repo_root),
        "source_asset": str(resolved_source_asset),
        "source_mode": "cad_modeling" if cad_modeling is not None else "provided",
        "cad_modeling": cad_modeling,
        "source_images": [str(path) for path in config.source_images],
        "source_image_bindings": [
            _frozen_file_binding(path) for path in config.source_images
        ],
        "prompt": config.prompt.strip(),
        "reference_images": [str(path) for path in config.reference_images],
        "reference_files": [str(path) for path in config.reference_files or []],
        "reference_bindings": references,
        "runtime": runtime,
    }
    if source_staging is not None:
        common["source_staging"] = source_staging
    if config.selected_mode == "agentic":
        if source_staging is None:
            raise ValueError(
                "Validated agentic config is missing staged source identity"
            )
        if config.usd_path is None:
            raise ValueError("Agentic asset runs currently require a provided source")
        if config.leaf_catalog is None:
            raise ValueError("Validated agentic config is missing its leaf catalog")
        coordinator = AssetSoleCoordinatorIdentity.create(
            coordinator_id=f"asset-coordinator:{run_id}",
            invocation_id=f"single-prompt:{run_id}",
            actor=config.runner,
            implementation=f"content-workflow-cli:{config.runner}",
        )
        configuration_identity: dict[str, object] = {
            "repository_root": str(config.repo_root),
            "runtime": runtime.model_dump(mode="json"),
            "requires_parent_resource_release": True,
        }
        if (
            config.required_leaf_ids
            or config.required_terminal_leaf_ids
            or config.required_leaf_dependencies
        ):
            configuration_identity.update(
                {
                    "required_leaf_ids": config.required_leaf_ids,
                    "required_terminal_leaf_ids": (config.required_terminal_leaf_ids),
                    "required_leaf_dependencies": config.required_leaf_dependencies,
                }
            )
        if config.exact_leaf_scope:
            configuration_identity["exact_leaf_scope"] = True
        configuration_digest = canonical_asset_digest(configuration_identity)
        return AssetRunRequest(
            **common,
            schema_version="content-agents.asset-composition-request.v3",
            selected_mode="agentic",
            prompt_digest=hashlib.sha256(
                config.prompt.strip().encode("utf-8")
            ).hexdigest(),
            source_digest=source_staging.staged_source.sha256,
            configuration_digest=configuration_digest,
            reference_digest=canonical_asset_digest(
                [binding.model_dump(mode="json") for binding in references]
            ),
            sole_coordinator_identity=coordinator,
            leaf_catalog=config.leaf_catalog,
            required_leaf_ids=config.required_leaf_ids,
            required_terminal_leaf_ids=config.required_terminal_leaf_ids,
            required_leaf_dependencies=config.required_leaf_dependencies,
            exact_leaf_scope=config.exact_leaf_scope,
            requires_parent_resource_release=True,
        )
    if (
        config.joint_config is None
        or config.materials_yaml is None
        or config.materials_usd is None
    ):
        raise ValueError("Validated fixed compatibility config is incomplete")
    schema_version = (
        "content-agents.asset-composition-request.v5"
        if fixed_geometry is not None or cad_modeling is not None
        else "content-agents.asset-composition-request.v2"
    )
    return AssetRunRequest(
        **common,
        schema_version=cast(Any, schema_version),
        selected_mode="compatibility_fixed",
        geometry=fixed_geometry,
        physics_validation_mode=cast(
            PhysicsValidationMode,
            config.physics_validation_mode or "runtime_required",
        ),
        joint_config=str(config.joint_config),
        joint_config_binding=_frozen_file_binding(config.joint_config),
        materials_yaml=str(config.materials_yaml),
        materials_yaml_binding=_frozen_file_binding(config.materials_yaml),
        materials_usd=str(config.materials_usd),
        materials_usd_binding=_frozen_file_binding(config.materials_usd),
        materials_usd_dependencies=bind_usd_dependency_closure(config.materials_usd),
    )


def _cad_modeling_request(config: AssetRunConfig) -> AssetCadModelingRequest:
    return AssetCadModelingRequest(
        provider_id=config.cad_provider_id,
        target_profile=config.geometry_target_profile,
        parameter_values=config.cad_parameter_values,
        parameter_variants=config.cad_parameter_variants,
        required_outputs=cast(Any, config.cad_required_outputs),
    )


def _write_source_intent(config: AssetRunConfig, *, run_dir: Path) -> Path:
    """Freeze prompt/image CAD intent as the initial stage artifact."""

    policy = _cad_modeling_request(config)
    source_intent = run_dir / SOURCE_INTENT_RELATIVE_PATH
    source_intent.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        source_intent,
        {
            "schema_version": "content-agents.asset-source-intent.v1",
            "mode": "cad_modeling",
            "prompt_sha256": hashlib.sha256(config.prompt.encode("utf-8")).hexdigest(),
            "source_images": [
                _frozen_file_binding(path).model_dump(mode="json")
                for path in config.source_images
            ],
            "cad_modeling": policy.model_dump(mode="json"),
        },
    )
    return source_intent.resolve()


def _load_existing(
    run_dir: str | Path,
    *,
    dry_run: bool,
) -> tuple[Path, AssetRunRequest, AssetRunConfig]:
    candidate = Path(run_dir).expanduser()
    _reject_unsafe_run_links(candidate)
    resolved = candidate.resolve()
    request_path = resolved / "request.json"
    state_path = resolved / "asset_run.json"
    if not request_path.is_file() or not state_path.is_file():
        raise FileNotFoundError(
            f"Composed asset request or run state is missing below {resolved}"
        )
    run = load_verified_run(state_path)
    request = load_verified_asset_request(state_path, run=run)
    if request.source_staging is None and request.source_mode != "cad_modeling":
        raise AssetCompositionStateError(
            "Legacy asset request predates run-confined source staging; start a "
            "new asset run instead of resuming this v1 request"
        )
    if request.source_staging is not None:
        staged_errors = _staged_input_integrity_errors(resolved)
        if staged_errors:
            raise AssetCompositionStateError(
                "Staged source integrity changed: " + "; ".join(staged_errors)
            )
    _verify_prompt_reference(request, run_dir=resolved)
    _verify_source_intent(request, run_dir=resolved)
    config = _config_from_request(request, dry_run=dry_run)
    frozen_legacy_graph = (
        run.execution_graph is not None
        and request.leaf_catalog is not None
        and request.leaf_catalog.schema_version
        == LEGACY_ASSET_LEAF_CATALOG_SCHEMA_VERSION
    )
    return (
        resolved,
        request,
        _validated_config(
            config,
            frozen_legacy_graph=frozen_legacy_graph,
        ),
    )


def _prompt_reference_text(prompt: str) -> str:
    """Return the deterministic Material reference for a prompt-only request."""

    return (
        "# Prompt-derived material reference\n\n"
        "No external appearance reference was supplied. Use this frozen user "
        "goal as the text reference for material selection and visual review.\n\n"
        f"{prompt.strip()}\n"
    )


def _with_material_reference(
    config: AssetRunConfig,
    *,
    run_dir: Path,
) -> AssetRunConfig:
    """Freeze the prompt as a text reference when Material has no other input."""

    if config.reference_images or config.reference_files:
        return config
    reference_path = run_dir / PROMPT_REFERENCE_RELATIVE_PATH
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(reference_path, _prompt_reference_text(config.prompt))
    return replace(config, reference_files=[reference_path])


def _verify_prompt_reference(request: AssetRunRequest, *, run_dir: Path) -> None:
    """Fail resume closed if a generated prompt reference changed."""

    expected_path = (run_dir / PROMPT_REFERENCE_RELATIVE_PATH).resolve()
    if str(expected_path) not in request.reference_files:
        return
    if request.reference_images or request.reference_files != [str(expected_path)]:
        raise ValueError(
            "Generated prompt reference must be the only reference in its frozen request"
        )
    try:
        actual = expected_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(
            f"Generated prompt reference is unavailable: {expected_path}"
        ) from exc
    if actual != _prompt_reference_text(request.prompt):
        raise ValueError("Generated prompt reference identity changed")


def _verify_source_intent(request: AssetRunRequest, *, run_dir: Path) -> None:
    """Fail resume closed if source-free CAD intent differs from the request."""

    if request.source_mode != "cad_modeling":
        return
    expected_path = (run_dir / SOURCE_INTENT_RELATIVE_PATH).resolve()
    if Path(request.source_asset).expanduser().resolve() != expected_path:
        raise ValueError("CAD source intent is not at the canonical run path")
    if request.cad_modeling is None:
        raise ValueError("CAD source intent lacks frozen modeling policy")
    try:
        payload = json.loads(expected_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"CAD source intent is unavailable: {expected_path}") from exc
    expected = {
        "schema_version": "content-agents.asset-source-intent.v1",
        "mode": "cad_modeling",
        "prompt_sha256": hashlib.sha256(request.prompt.encode("utf-8")).hexdigest(),
        "source_images": [
            binding.model_dump(mode="json") for binding in request.source_image_bindings
        ],
        "cad_modeling": request.cad_modeling.model_dump(mode="json"),
    }
    if payload != expected:
        raise ValueError("CAD source intent differs from the frozen request")


def _frozen_file_binding(path: Path) -> ArtifactBinding:
    """Capture the exact bytes of one normalized, regular workflow input."""

    candidate = path.expanduser().absolute()
    try:
        metadata = candidate.lstat()
    except OSError as exc:
        raise FileNotFoundError(
            f"Frozen workflow input is unavailable: {candidate}"
        ) from exc
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise FileNotFoundError(
            "Frozen workflow input is not a regular file with unique identity: "
            f"{candidate}"
        )
    resolved = candidate.resolve(strict=True)
    return ArtifactBinding(
        path=str(resolved),
        sha256=file_sha256(resolved),
        size_bytes=metadata.st_size,
    )


def _segmentation_run_files(root: Path) -> list[Path]:
    """Return one bounded, symlink-free regular-file closure."""

    files: list[Path] = []
    total_bytes = 0
    for current_text, directory_names, file_names in os.walk(root, followlinks=False):
        current = Path(current_text)
        directory_names.sort()
        file_names.sort()
        for name in directory_names:
            candidate = current / name
            metadata = candidate.lstat()
            if candidate.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
                raise ValueError(
                    f"Segmentation run contains an unsafe directory: {candidate}"
                )
        for name in file_names:
            candidate = current / name
            metadata = candidate.lstat()
            if candidate.is_symlink() or not stat.S_ISREG(metadata.st_mode):
                raise ValueError(
                    f"Segmentation run contains a non-regular file: {candidate}"
                )
            if metadata.st_nlink != 1:
                raise ValueError(
                    f"Segmentation run contains a hard-linked file: {candidate}"
                )
            files.append(candidate)
            total_bytes += metadata.st_size
            if len(files) > MAX_STAGED_SEGMENTATION_FILES:
                raise ValueError("Segmentation run exceeds the staged file-count limit")
            if total_bytes > MAX_STAGED_SEGMENTATION_BYTES:
                raise ValueError("Segmentation run exceeds the staged byte limit")
    return sorted(files, key=lambda path: str(path))


def _copy_stable_regular_file(
    source: Path,
    destination: Path,
    *,
    maximum_bytes: int,
) -> int:
    """Copy one descriptor-pinned input without following links."""

    source_descriptor: int | None = None
    destination_descriptor: int | None = None
    try:
        source_descriptor = os.open(
            source,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
        before = os.fstat(source_descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError(
                f"Segmentation input is not a unique regular file: {source}"
            )
        if before.st_size > maximum_bytes:
            raise ValueError("Segmentation run exceeds the staged byte limit")
        destination_descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        copied = 0
        while chunk := os.read(source_descriptor, 1024 * 1024):
            view = memoryview(chunk)
            while view:
                written = os.write(destination_descriptor, view)
                if written <= 0:
                    raise OSError("segmentation staging write made no progress")
                view = view[written:]
            copied += len(chunk)
            if copied > maximum_bytes:
                raise ValueError("Segmentation run exceeds the staged byte limit")
        os.fsync(destination_descriptor)
        after = os.fstat(source_descriptor)
        if (
            (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino)
            or after.st_size != before.st_size
            or after.st_mtime_ns != before.st_mtime_ns
            or after.st_ctime_ns != before.st_ctime_ns
            or after.st_nlink != before.st_nlink
            or copied != before.st_size
        ):
            raise ValueError(f"Segmentation input changed while staging: {source}")
        return copied
    except Exception:
        if destination_descriptor is not None:
            try:
                destination.unlink(missing_ok=True)
            except OSError:
                pass
        raise
    finally:
        if destination_descriptor is not None:
            os.close(destination_descriptor)
        if source_descriptor is not None:
            os.close(source_descriptor)


def _rebase_segmentation_value(
    value: object,
    *,
    source_root: Path,
    destination_root: Path,
) -> object:
    if isinstance(value, str):
        candidate = Path(value)
        if candidate.is_absolute():
            try:
                relative = candidate.relative_to(source_root)
            except ValueError:
                return value
            return str(destination_root / relative)
        return value
    if isinstance(value, list):
        return [
            _rebase_segmentation_value(
                item,
                source_root=source_root,
                destination_root=destination_root,
            )
            for item in value
        ]
    if isinstance(value, dict):
        return {
            key: _rebase_segmentation_value(
                item,
                source_root=source_root,
                destination_root=destination_root,
            )
            for key, item in value.items()
        }
    return value


def _rebase_staged_segmentation_run(
    *,
    source_root: Path,
    destination_root: Path,
) -> None:
    path_bearing_documents = (
        "request.json",
        "prepare/topology.json",
        "segments.json",
        "final/renders/render_manifest.json",
        "final/export_manifest.json",
    )
    for relative in path_bearing_documents:
        path = destination_root / relative
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Could not rebase staged segmentation artifact {relative}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise ValueError(
                f"Staged segmentation artifact must be a JSON object: {relative}"
            )
        rebased = _rebase_segmentation_value(
            payload,
            source_root=source_root,
            destination_root=destination_root,
        )
        assert isinstance(rebased, dict)
        if relative == "final/export_manifest.json":
            rebased["segments_sha256"] = file_sha256(destination_root / "segments.json")
        atomic_write_json(path, rebased)
    _rebind_staged_usd_cli_receipts(
        source_root=source_root,
        destination_root=destination_root,
    )


def _staged_render_artifact(
    *,
    destination_root: Path,
    manifest_path: Path,
    value: object,
    label: str,
) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Staged render evidence has an invalid {label} path")
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = manifest_path.parent / candidate
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(destination_root.resolve(strict=True))
        metadata = resolved.lstat()
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"Staged render evidence {label} is outside the copied run"
        ) from exc
    if (
        resolved.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
    ):
        raise ValueError(f"Staged render evidence {label} is not a unique regular file")
    return resolved


def _rebase_staged_json_document(
    path: Path,
    *,
    source_root: Path,
    destination_root: Path,
    label: str,
) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not rebase staged {label}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Staged {label} must be a JSON object")
    rebased = _rebase_segmentation_value(
        payload,
        source_root=source_root,
        destination_root=destination_root,
    )
    assert isinstance(rebased, dict)
    atomic_write_json(path, rebased)
    return rebased


def _rebind_staged_usd_cli_receipts(
    *,
    source_root: Path,
    destination_root: Path,
) -> None:
    """Make copied usd-cli evidence portable without changing the producer run."""

    manifest_path = destination_root / "final/renders/render_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("scene_tool") != "usd-cli":
        return
    renders = manifest.get("renders")
    if not isinstance(renders, list):
        raise ValueError("Staged usd-cli render manifest has invalid renders")
    for index, record in enumerate(renders):
        if not isinstance(record, dict):
            raise ValueError(f"Staged usd-cli render record {index} is invalid")
        for artifact_kind in ("camera", "response"):
            artifact = _staged_render_artifact(
                destination_root=destination_root,
                manifest_path=manifest_path,
                value=record.get(artifact_kind),
                label=f"render {index} {artifact_kind}",
            )
            _rebase_staged_json_document(
                artifact,
                source_root=source_root,
                destination_root=destination_root,
                label=f"render {index} {artifact_kind}",
            )
            record[f"{artifact_kind}_sha256"] = file_sha256(artifact)

    receipts = _staged_render_artifact(
        destination_root=destination_root,
        manifest_path=manifest_path,
        value=manifest.get("usd_cli_command_receipts"),
        label="usd-cli command receipts",
    )
    checkpoint = _staged_render_artifact(
        destination_root=destination_root,
        manifest_path=manifest_path,
        value=manifest.get("usd_cli_receipt_checkpoint"),
        label="usd-cli receipt checkpoint",
    )
    try:
        receipt_lines = receipts.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"Could not read staged usd-cli receipts: {exc}") from exc
    if not receipt_lines:
        raise ValueError("Staged usd-cli receipt journal is empty")
    rebased_receipts: list[dict[str, Any]] = []
    for index, line in enumerate(receipt_lines):
        try:
            receipt = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Staged usd-cli receipt {index} is invalid JSON") from exc
        if not isinstance(receipt, dict):
            raise ValueError(f"Staged usd-cli receipt {index} must be an object")
        rebased = _rebase_segmentation_value(
            receipt,
            source_root=source_root,
            destination_root=destination_root,
        )
        assert isinstance(rebased, dict)
        rebased_receipts.append(rebased)
    atomic_write_text(
        receipts,
        "".join(
            json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n"
            for receipt in rebased_receipts
        ),
    )
    receipt_stat = receipts.stat(follow_symlinks=False)
    checkpoint_payload = _rebase_staged_json_document(
        checkpoint,
        source_root=source_root,
        destination_root=destination_root,
        label="usd-cli receipt checkpoint",
    )
    checkpoint_payload.update(
        {
            "receipt_device": receipt_stat.st_dev,
            "receipt_inode": receipt_stat.st_ino,
            "receipt_sha256": file_sha256(receipts),
            "receipt_size_bytes": receipt_stat.st_size,
        }
    )
    atomic_write_json(checkpoint, checkpoint_payload)
    manifest["usd_cli_command_receipts_sha256"] = file_sha256(receipts)
    manifest["usd_cli_receipt_checkpoint_sha256"] = file_sha256(checkpoint)
    atomic_write_json(manifest_path, manifest)


def _stage_completed_segmentation_run(
    source_root: Path,
    *,
    destination_parent: Path,
    source_asset: Path | None,
    required: bool,
    required_parts: list[str],
) -> AssetSegmentationRunBinding:
    """Validate, snapshot, rebase, and revalidate a completed producer run."""

    if source_asset is None:
        raise ValueError(
            "A completed segmentation run must be paired with a provided source asset"
        )
    from content_agent_workflows.geometry.segmentation import (
        consume_segmentation_handoff,
    )

    source_root = source_root.resolve(strict=True)
    source_handoff = consume_segmentation_handoff(
        run_dir=source_root,
        expected_source_usd=source_asset,
        required=required,
        required_parts=required_parts,
        require_ovrtx_evidence=True,
    )
    if source_handoff.failures or source_handoff.outcome == "rejected":
        raise ValueError(
            "Completed segmentation run failed Geometry handoff validation: "
            + "; ".join(source_handoff.failures)
        )
    producer_run_id = source_handoff.producer_run_id
    if not producer_run_id:
        raise ValueError("Completed segmentation run lacks a producer run ID")

    destination_parent.mkdir(parents=True, exist_ok=False)
    destination_root = destination_parent / producer_run_id
    destination_root.mkdir()
    source_files = _segmentation_run_files(source_root)
    copied_bytes = 0
    for source in source_files:
        relative = source.relative_to(source_root)
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        copied_bytes += _copy_stable_regular_file(
            source,
            destination,
            maximum_bytes=MAX_STAGED_SEGMENTATION_BYTES - copied_bytes,
        )
    if _segmentation_run_files(source_root) != source_files:
        raise ValueError("Segmentation run file closure changed while staging")

    _rebase_staged_segmentation_run(
        source_root=source_root,
        destination_root=destination_root,
    )
    staged_handoff = consume_segmentation_handoff(
        run_dir=destination_root,
        expected_source_usd=source_asset,
        required=required,
        required_parts=required_parts,
        require_ovrtx_evidence=True,
    )
    if staged_handoff.failures or staged_handoff.outcome == "rejected":
        raise ValueError(
            "Staged segmentation run failed Geometry handoff validation: "
            + "; ".join(staged_handoff.failures)
        )
    if (
        staged_handoff.source_asset_sha256 != source_handoff.source_asset_sha256
        or staged_handoff.topology_digest != source_handoff.topology_digest
        or staged_handoff.fragment_labels_sha256
        != source_handoff.fragment_labels_sha256
        or staged_handoff.face_labels_sha256 != source_handoff.face_labels_sha256
        or staged_handoff.segmented_usd_sha256 != source_handoff.segmented_usd_sha256
        or staged_handoff.parts != source_handoff.parts
    ):
        raise ValueError("Staged segmentation handoff differs from its producer run")

    artifacts = tuple(
        _frozen_file_binding(path) for path in _segmentation_run_files(destination_root)
    )
    return AssetSegmentationRunBinding(
        producer_run_id=producer_run_id,
        run_dir=str(destination_root.resolve()),
        artifacts=artifacts,
    )


def _load_leaf_catalog(path: Path) -> AssetLeafCatalog:
    """Resolve compatibility bytes to the exact repository-owned catalog."""

    candidate = path.expanduser().absolute()
    try:
        metadata = candidate.lstat()
    except OSError as exc:
        raise FileNotFoundError(f"Leaf catalog is unavailable: {candidate}") from exc
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > 8 * 1024 * 1024:
        raise ValueError("Leaf catalog must be a bounded regular JSON file")
    try:
        candidate_catalog = AssetLeafCatalog.model_validate_json(
            candidate.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, ValidationError) as exc:
        raise ValueError(f"Leaf catalog is invalid: {exc}") from exc
    try:
        return resolve_repository_asset_leaf_catalog(candidate_catalog).catalog
    except ValueError as exc:
        raise ValueError(f"Leaf catalog is not repository-owned: {exc}") from exc


def _config_from_request(request: AssetRunRequest, *, dry_run: bool) -> AssetRunConfig:
    if isinstance(request.cad_modeling, LegacyAssetCadModelingRequest):
        raise ValueError(
            "asset request v4 uses the retired direct-authoring policy; "
            "inspect it with this release or resume it with the pre-provider runner"
        )
    runtime = request.runtime
    return AssetRunConfig(
        repo_root=Path(request.repository_root),
        usd_path=(
            Path(request.source_asset) if request.source_mode == "provided" else None
        ),
        prompt=request.prompt,
        selected_mode=request.selected_mode,
        leaf_catalog=request.leaf_catalog,
        required_leaf_ids=list(request.required_leaf_ids),
        required_terminal_leaf_ids=list(request.required_terminal_leaf_ids),
        required_leaf_dependencies={
            leaf_id: list(dependencies)
            for leaf_id, dependencies in request.required_leaf_dependencies.items()
        },
        exact_leaf_scope=request.exact_leaf_scope,
        joint_config=(
            Path(request.joint_config) if request.joint_config is not None else None
        ),
        materials_yaml=(
            Path(request.materials_yaml) if request.materials_yaml is not None else None
        ),
        materials_usd=(
            Path(request.materials_usd) if request.materials_usd is not None else None
        ),
        source_images=[Path(path) for path in request.source_images],
        reference_images=[Path(path) for path in request.reference_images],
        reference_files=[Path(path) for path in request.reference_files],
        output_dir=Path(request.run_dir),
        run_id=request.run_id,
        runner=runtime.runner,
        model=runtime.model,
        model_reasoning_effort=runtime.model_reasoning_effort,
        codex_base_url=runtime.codex_base_url,
        codex_sandbox_mode=runtime.codex_sandbox_mode,
        codex_config=runtime.codex_config or None,
        claude_config=runtime.claude_config or None,
        claude_permission_mode=runtime.claude_permission_mode,
        claude_max_turns=runtime.claude_max_turns,
        claude_execution_mode=runtime.claude_execution_mode,
        scene_tool_timeout_seconds=runtime.scene_tool_timeout_seconds,
        child_timeout_seconds=runtime.child_timeout_seconds,
        physics_validation_mode=(
            request.physics_validation_mode
            if request.selected_mode == "compatibility_fixed"
            else None
        ),
        geometry_target_profile=(
            request.geometry.target_profile
            if request.geometry is not None
            else "geometry-agent.static-visual-asset.v1"
        ),
        geometry_optimization_policy=(
            request.geometry.optimization_policy
            if request.geometry is not None
            else "preserve_correspondence"
        ),
        geometry_optimizer_backend=(
            request.geometry.optimizer_backend
            if request.geometry is not None
            else "local"
        ),
        geometry_repair_mode=(
            request.geometry.repair_mode if request.geometry is not None else "off"
        ),
        geometry_repair_profile=(
            request.geometry.repair_profile
            if request.geometry is not None
            else "visual_only"
        ),
        geometry_render_evidence=(
            request.geometry.render_evidence if request.geometry is not None else False
        ),
        geometry_segmentation_run_dir=(
            Path(request.geometry.segmentation_run.run_dir)
            if request.geometry is not None
            and request.geometry.segmentation_run is not None
            else None
        ),
        geometry_segmentation_required=(
            request.geometry.segmentation_required
            if request.geometry is not None
            else False
        ),
        geometry_required_parts=(
            list(request.geometry.segmentation_required_parts)
            if request.geometry is not None
            else []
        ),
        include_geometry_stage=request.geometry is not None,
        cad_provider_id=(
            request.cad_modeling.provider_id
            if request.cad_modeling is not None
            else DEFAULT_GEOMETRY_AUTHORING_PROVIDER_ID
        ),
        cad_parameter_values=(
            dict(request.cad_modeling.parameter_values)
            if request.cad_modeling is not None
            else {}
        ),
        cad_parameter_variants=(
            list(request.cad_modeling.parameter_variants)
            if request.cad_modeling is not None
            else []
        ),
        cad_required_outputs=(
            list(request.cad_modeling.required_outputs)
            if request.cad_modeling is not None
            else ["usd"]
        ),
        dry_run=dry_run,
        agent_cwd=Path(request.run_dir),
    )


def _validated_config(
    config: AssetRunConfig,
    *,
    frozen_legacy_graph: bool = False,
) -> AssetRunConfig:
    required_leaf_ids = [leaf_id.strip() for leaf_id in config.required_leaf_ids]
    required_terminal_leaf_ids = [
        leaf_id.strip() for leaf_id in config.required_terminal_leaf_ids
    ]
    for label, values in (
        ("--required-leaf", required_leaf_ids),
        ("--required-terminal-leaf", required_terminal_leaf_ids),
    ):
        if any(not value for value in values):
            raise ValueError(f"{label} requires a non-empty repository leaf ID")
        if len(values) != len(set(values)):
            raise ValueError(f"{label} must not contain duplicates")
    required_leaf_ids.sort()
    required_terminal_leaf_ids.sort()
    nonrequired_terminals = sorted(
        set(required_terminal_leaf_ids).difference(required_leaf_ids)
    )
    if nonrequired_terminals:
        raise ValueError(
            "--required-terminal-leaf values must also be supplied with "
            f"--required-leaf: {nonrequired_terminals}"
        )
    required_leaf_dependencies: dict[str, list[str]] = {}
    for raw_owner, raw_dependencies in config.required_leaf_dependencies.items():
        owner = raw_owner.strip()
        dependencies = [dependency.strip() for dependency in raw_dependencies]
        if not owner or any(not dependency for dependency in dependencies):
            raise ValueError("required leaf dependency IDs must be non-empty")
        if len(dependencies) != len(set(dependencies)):
            raise ValueError(
                f"required leaf dependencies must not contain duplicate edges: {owner}"
            )
        if owner in dependencies:
            raise ValueError(f"required leaf cannot depend on itself: {owner}")
        required_leaf_dependencies[owner] = sorted(dependencies)
    required_leaf_dependencies = {
        owner: required_leaf_dependencies[owner]
        for owner in sorted(required_leaf_dependencies)
    }
    dependency_ids = {
        leaf_id
        for owner, dependencies in required_leaf_dependencies.items()
        for leaf_id in (owner, *dependencies)
    }
    nonrequired_dependencies = sorted(dependency_ids.difference(required_leaf_ids))
    if nonrequired_dependencies:
        raise ValueError(
            "--required-leaf-dependency IDs must also be supplied with "
            f"--required-leaf: {nonrequired_dependencies}"
        )
    if config.exact_leaf_scope and not required_leaf_ids:
        raise ValueError("--exact-leaf-scope requires at least one --required-leaf")
    normalized_source_images = [
        path.expanduser().resolve() for path in config.source_images
    ]
    normalized_reference_images = [
        path.expanduser().resolve() for path in config.reference_images
    ]
    normalized_segmentation_run = None
    if config.geometry_segmentation_run_dir is not None:
        requested_segmentation_run = (
            config.geometry_segmentation_run_dir.expanduser().absolute()
        )
        try:
            segmentation_metadata = requested_segmentation_run.lstat()
        except OSError as exc:
            raise FileNotFoundError(
                "Geometry segmentation run directory is unavailable: "
                f"{requested_segmentation_run}"
            ) from exc
        if requested_segmentation_run.is_symlink() or not stat.S_ISDIR(
            segmentation_metadata.st_mode
        ):
            raise ValueError(
                "Geometry segmentation run must be a non-symlink directory"
            )
        normalized_segmentation_run = requested_segmentation_run.resolve(strict=True)
    if len(normalized_reference_images) != len(set(normalized_reference_images)):
        raise ValueError("Material references must not repeat within a reference kind")
    if config.selected_mode not in {"agentic", "compatibility_fixed"}:
        raise ValueError(f"Unsupported asset selected mode: {config.selected_mode}")
    if config.selected_mode == "agentic":
        if frozen_legacy_graph:
            if (
                config.leaf_catalog is None
                or config.leaf_catalog.schema_version
                != LEGACY_ASSET_LEAF_CATALOG_SCHEMA_VERSION
            ):
                raise ValueError(
                    "Frozen graph-v1 resume requires its exact legacy leaf catalog"
                )
            leaf_catalog = config.leaf_catalog
        else:
            leaf_catalog = (
                discover_repository_asset_leaf_catalog()
                if config.leaf_catalog is None
                else resolve_repository_asset_leaf_catalog(config.leaf_catalog).catalog
            )
        catalog_ids = {descriptor.leaf_id for descriptor in leaf_catalog.descriptors}
        missing_required = sorted(set(required_leaf_ids).difference(catalog_ids))
        if missing_required:
            raise ValueError(
                "Required asset capabilities are absent from the repository "
                f"leaf catalog: {missing_required}"
            )
        if any(
            value is not None
            for value in (
                config.joint_config,
                config.materials_yaml,
                config.materials_usd,
                config.physics_validation_mode,
            )
        ):
            raise ValueError(
                "Agentic asset runs cannot accept fixed-stage domain configuration"
            )
        joint_config = None
        materials_yaml = None
        materials_usd = None
        physics_validation_mode = None
    else:
        if (
            required_leaf_ids
            or required_terminal_leaf_ids
            or required_leaf_dependencies
        ):
            raise ValueError("Required graph leaves are accepted only in agentic mode")
        if config.leaf_catalog is not None:
            raise ValueError(
                "Fixed compatibility mode cannot accept an agentic leaf catalog"
            )
        if config.joint_config is None or config.materials_yaml is None:
            raise ValueError(
                "--compatibility-fixed-order requires --joint-config and "
                "--materials-yaml"
            )
        joint_config = config.joint_config.expanduser().resolve()
        materials_yaml = config.materials_yaml.expanduser().resolve()
        if not materials_yaml.is_file():
            raise FileNotFoundError(
                f"materials manifest is not a file: {materials_yaml}"
            )
        materials_usd = (
            config.materials_usd.expanduser().resolve()
            if config.materials_usd is not None
            else _resolve_materials_usd_from_manifest(materials_yaml)
        )
        physics_validation_mode = config.physics_validation_mode or "runtime_required"
        leaf_catalog = None
    resolved = replace(
        config,
        repo_root=config.repo_root.expanduser().resolve(),
        usd_path=(
            config.usd_path.expanduser().resolve()
            if config.usd_path is not None
            else None
        ),
        source_root=(
            config.source_root.expanduser().resolve()
            if config.source_root is not None
            else None
        ),
        joint_config=joint_config,
        materials_yaml=materials_yaml,
        materials_usd=materials_usd,
        physics_validation_mode=physics_validation_mode,
        leaf_catalog=leaf_catalog,
        required_leaf_ids=required_leaf_ids,
        required_terminal_leaf_ids=required_terminal_leaf_ids,
        required_leaf_dependencies=required_leaf_dependencies,
        exact_leaf_scope=config.exact_leaf_scope,
        source_images=normalized_source_images,
        reference_images=list(
            dict.fromkeys([*normalized_source_images, *normalized_reference_images])
        ),
        reference_files=[
            path.expanduser().resolve() for path in (config.reference_files or [])
        ],
        prompt=config.prompt.strip(),
        cad_provider_id=config.cad_provider_id.strip(),
        geometry_target_profile=config.geometry_target_profile.strip(),
        geometry_segmentation_run_dir=normalized_segmentation_run,
        geometry_required_parts=[
            name.strip() for name in config.geometry_required_parts
        ],
        agent_workspace=(config.repo_root / "agentic").resolve(),
    )
    required_files: list[tuple[str, Path]] = []
    if resolved.usd_path is not None:
        required_files.append(("source USD", resolved.usd_path))
    if resolved.selected_mode == "compatibility_fixed":
        assert resolved.joint_config is not None
        assert resolved.materials_yaml is not None
        assert materials_usd is not None
        required_files.extend(
            [
                ("Joint config", resolved.joint_config),
                ("materials manifest", resolved.materials_yaml),
                ("materials library", materials_usd),
            ]
        )
    for label, path in required_files:
        if not path.is_file():
            raise FileNotFoundError(f"{label} is not a file: {path}")
    if resolved.source_root is not None:
        if resolved.usd_path is None:
            raise ValueError("--source-root requires --source")
        if _is_usd_source_path(resolved.usd_path):
            raise ValueError("--source-root is not accepted for USD sources")
        if not resolved.source_root.is_dir():
            raise FileNotFoundError(
                f"Source dependency root is not a directory: {resolved.source_root}"
            )
    if resolved.usd_path is not None and resolved.source_images:
        raise ValueError("--source-image is accepted only when --source is omitted")
    if resolved.usd_path is None:
        raise ValueError(
            "Asset composition requires a provided source. Use Geometry Agent "
            "generation or export first, then pass its immutable artifact."
        )
    if resolved.geometry_segmentation_run_dir is not None:
        if not resolved.include_geometry_stage:
            raise ValueError("A segmentation run requires the Geometry stage")
        if resolved.usd_path is None:
            raise ValueError(
                "A completed segmentation run must be bound to a provided source asset"
            )
    for path in resolved.source_images:
        if not path.is_file():
            raise FileNotFoundError(f"CAD source image is not a file: {path}")
        if path.suffix.lower() not in CAD_SOURCE_IMAGE_SUFFIXES:
            raise ValueError(
                f"CAD source image has unsupported format: {path.suffix or '<none>'}"
            )
    for path in [*resolved.reference_images, *(resolved.reference_files or [])]:
        if not path.is_file():
            raise FileNotFoundError(f"Reference is not a file: {path}")
    all_references = [
        *resolved.reference_images,
        *(resolved.reference_files or []),
    ]
    if len(all_references) != len(set(all_references)):
        raise ValueError(
            "Material references must not repeat within or across reference kinds"
        )
    if not resolved.prompt:
        raise ValueError("The asset prompt must not be empty")
    if not resolved.cad_provider_id:
        raise ValueError("Geometry authoring provider id must not be empty")
    if resolved.runner not in SUPPORTED_RUNNERS:
        raise ValueError(f"Unsupported runner: {resolved.runner}")
    if resolved.codex_sandbox_mode not in SUPPORTED_CODEX_SANDBOX_MODES:
        raise ValueError(f"Unsupported Codex sandbox: {resolved.codex_sandbox_mode}")
    if resolved.claude_execution_mode not in SUPPORTED_CLAUDE_EXECUTION_MODES:
        raise ValueError(
            f"Unsupported Claude execution mode: {resolved.claude_execution_mode}"
        )
    if resolved.scene_tool_timeout_seconds <= 0:
        raise ValueError("Scene-tool timeout must be greater than zero")
    if resolved.child_timeout_seconds < 0:
        raise ValueError("Child timeout must be zero or greater")
    if resolved.physics_validation_mode is not None and (
        resolved.physics_validation_mode
        not in {
            "runtime_required",
            "schema_readback",
        }
    ):
        raise ValueError(
            "Physics validation mode must be runtime_required or schema_readback"
        )
    if not resolved.geometry_target_profile:
        raise ValueError("Geometry target profile must not be empty")
    if resolved.geometry_optimization_policy not in {
        "skip",
        "preserve_correspondence",
        "runtime_efficiency",
    }:
        raise ValueError("Unsupported Geometry optimization policy")
    if resolved.geometry_optimizer_backend not in {"local", "remote"}:
        raise ValueError("Unsupported Geometry optimizer backend")
    if resolved.geometry_repair_mode not in {"off", "diagnose", "auto"}:
        raise ValueError("Unsupported Geometry repair mode")
    if resolved.geometry_repair_profile not in {
        "visual_only",
        "static_environment",
        "rigid_pick_place",
        "articulated_rigid",
        "contact_rich",
        "deformable_or_cae",
    }:
        raise ValueError("Unsupported Geometry repair profile")
    if any(
        not name
        or len(name) > 2048
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
        or any(ord(character) < 32 for character in name)
        for name in resolved.geometry_required_parts
    ):
        raise ValueError("Geometry required parts must contain bounded plain names")
    if len(resolved.geometry_required_parts) != len(
        set(resolved.geometry_required_parts)
    ):
        raise ValueError("Geometry required part names must not repeat")
    if resolved.geometry_required_parts and not resolved.geometry_segmentation_required:
        raise ValueError(
            "Geometry required parts require --geometry-segmentation-required"
        )
    if resolved.usd_path is None:
        _cad_modeling_request(resolved)
    trusted_workspace = (resolved.repo_root / "agentic").resolve()
    if not (trusted_workspace / ".agents" / "skills").is_dir():
        raise FileNotFoundError(
            f"Agent workflow skills are unavailable below {trusted_workspace}"
        )
    return resolved


def _create_run_dir(config: AssetRunConfig) -> tuple[str, Path]:
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
    source_name = (
        config.usd_path.stem
        if config.usd_path is not None
        else "-".join(config.prompt.split()[:6])
    )
    default_id = f"{_slug(source_name)}-{stamp}"
    run_id = (config.run_id or default_id).strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id):
        raise ValueError("Invalid --run-id")
    candidate = config.output_dir or config.repo_root / "runs" / run_id
    _reject_unsafe_run_links(candidate, allow_missing=True)
    run_dir = _lexical_absolute_path(candidate)
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    # The run can briefly hold parent-only renderer configuration before child
    # launch, and every workflow receipt writer requires an owner-controlled raw
    # directory. Pass restrictive modes explicitly instead of inheriting a
    # process umask (commonly 0022 in CI). Normalize a propagated setgid bit on
    # newly created directories before any untrusted process can observe them.
    run_dir.mkdir(mode=0o700)
    run_dir.chmod(0o700)
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(mode=0o700)
    raw_dir.chmod(0o700)
    return run_id, run_dir


def _remove_uncommitted_fresh_run(run_dir: Path, *, request_path: Path) -> None:
    """Remove only a just-created run that never acquired a durable request."""

    if request_path.exists() or request_path.is_symlink():
        raise RuntimeError(
            f"Refusing cleanup after a durable asset request exists: {request_path}"
        )
    _reject_unsafe_run_links(run_dir, allow_missing=False)
    resolved = run_dir.resolve(strict=True)
    if not resolved.is_dir():
        raise RuntimeError(f"Refusing unsafe fresh-run cleanup: {run_dir}")
    shutil.rmtree(resolved)
    if resolved.exists() or resolved.is_symlink():
        raise RuntimeError(f"Fresh asset run remains after cleanup: {resolved}")


def _build_agent_prompt(
    *,
    request: AssetRunRequest,
    request_path: Path,
    run_state_path: Path,
    run_dir: Path,
    resume: bool,
    include_cad_modeling: bool,
) -> str:
    action = "Resume" if resume else "Execute"
    first_executor = (
        "Start with the typed CAD-modeling executor. Inspect its immutable "
        "exported geometry, external authoring evidence, candidate ledger, "
        "source manifest, and requested parameter-family evidence before "
        "accepting its output into Geometry."
        if include_cad_modeling
        else "Start with the typed Geometry executor."
    )
    if request.selected_mode == "agentic":
        windows_artifact_guidance = (
            """On managed Windows, do not invoke `Copy-Item`, `Get-Item`, or
`Get-FileHash`. Parent-sealed source, readiness, session, and staged-skill
artifacts are already below the run directory with exact path, SHA-256, and
size bindings. Reference those bindings in place instead of duplicating or
rehashing them. When deterministic Articulation preparation needs one retained
root, use the run directory itself and name the existing run-relative artifacts
in the readback. For new child-authored JSON, use the controlled writer and copy
the exact binding it returns. Do not probe for an alternate copy, stat, or hash
mechanism. Invoke every frozen native leaf only through
`content-workflow-asset-state invoke-leaf --run-state <state> --leaf <leaf>
--invocation <absolute invocation> --result <absolute new result JSON>`. Do not
invoke the descriptor's domain executable directly. Do not invoke `usd-cli-tel`
directly from the coordinator. The launcher-owned readiness artifact, command
journal/checkpoint, staged source, and exact session identity are the retained
preparation facts available before native execution. Domain leaves invoked
through `invoke-leaf` own any authenticated scene command they require. Read
each staged skill once with a literal `Get-Content -LiteralPath
'<path>' -Raw` command and wait for its completed result. If a long file must be
continued, use one literal `Get-Content -LiteralPath '<path>' | Select-Object
-Skip <offset> -First 200` command at a time. Never assign a PowerShell variable,
index a line array, or launch parallel retry reads.
"""
            if os.name == "nt"
            else ""
        )
        legacy_graph_resume = (
            resume
            and request.leaf_catalog is not None
            and request.leaf_catalog.schema_version
            == LEGACY_ASSET_LEAF_CATALOG_SCHEMA_VERSION
        )
        required_graph_guidance = (
            " Select every `required_leaf_ids` entry as `required`, mark exactly "
            "`required_terminal_leaf_ids` as terminal outputs, and do not add "
            "another terminal. Include every edge in "
            "`required_leaf_dependencies` before graph freeze."
            if request.required_leaf_ids
            else ""
        )
        exact_scope_guidance = (
            " The request freezes an exact leaf scope: select every and only "
            "`required_leaf_ids` entry. Give every selected node a concise "
            "`selection_rationale` that explains its direct prompt relevance or "
            "its required dependency role. Any extra leaf or missing rationale "
            "will fail graph freeze."
            if request.exact_leaf_scope
            else ""
        )
        graph_guidance = (
            "Continue the exact frozen `content-agents.asset-execution-graph.v1` "
            "in its caller-supplied topological order. Never regenerate, reorder, "
            "or upgrade its graph/catalog/receipt identities."
            if legacy_graph_resume
            else (
                "Supply exactly one typed "
                "`content-agents.asset-execution-graph.v2`: select only the opaque "
                "leaf IDs that are needed, list every catalog omission, declare "
                "only prompt-specific dependency edges, bind every request/catalog "
                "digest, and freeze the graph before any leaf executes."
                f"{required_graph_guidance}{exact_scope_guidance} Nodes and "
                "omissions are stored lexically; that storage order is not "
                "execution policy."
            )
        )
        return f"""Use the `content-workflow-asset` skill.

{action} one frozen agentic asset request as its sole outer coordinator:

- Request: `{request_path}`
- Durable state: `{run_state_path}`
- Run directory: `{run_dir}`

Read and verify the request, sole-coordinator identity, and embedded public leaf
catalog before acting. {graph_guidance}

For the initial graph, write exactly `execution_graph.json` with the controlled
JSON procedure below. Do not invoke the provider `Write`, `Edit`, or `MultiEdit`
tools, and do not probe for or retry another write mechanism.
{CONTROLLED_JSON_ARTIFACT_WRITE}

{windows_artifact_guidance}

Never edit `request.json` or `asset_run.json` directly. Use only the graph and
leaf transitions documented by the umbrella skill. State validates and follows
your frozen graph; it cannot select, insert, or remove leaves. Record each exact
typed invocation and native result. The descriptor-resolved projector binds the
native terminal receipt, operation/evidence indexes, evidence, readback, native
status, and resource releases. Preserve an opaque failed or cancelled result
exactly; never upgrade it to `passed` or `not_evaluated`. Optional selected leaves
may end `not_evaluated` only when no selected dependent requires them. Omitted
leaves remain `not_requested`. If `articulation.proposal-provider.v1` is selected,
construct its predecessor preparation with `proposal_status=not_evaluated` and
no proposal; use `proposal_status=not_requested` only when that leaf is omitted.
For every invocation artifact inherited from a predecessor leaf, copy its absolute `path`, `sha256`, and `size_bytes` verbatim from that predecessor's
verified `leaf_receipt.json`; never reconstruct, abbreviate, or relocate the
path. Before calling the opaque entrypoint, require that each copied path is an
existing regular file below the declared run directory and rehashes to the
copied identity. Never create, pre-create, or write a descriptor-owned `native/`
directory; only the opaque entrypoint initializes that workspace. Correct an
invocation-construction error before the native call; preserve a native failed
result once called. The standalone Articulation
preparation capability must set
`canonical_output_evidence_required=true`; this declares the mandatory
post-author policy and does not supply a pre-author visual envelope or change
the author-before-canonical-render dependency. At the Articulation author pause,
set the top-level `evidence_requirements` to a non-empty list and set non-empty
`evidence_ids` on every `candidate_decisions` entry and every entry in
`canonical_graph.groups`, `canonical_graph.memberships`,
`canonical_graph.joints`, and `canonical_graph.rigid_link_operations`. Use only
exact `evidence_id` values present in the current
`articulation_author_observation.json.evidence_records`. Verify every required
list before the first native call; do not submit an empty list and try to repair
it after a native failure; these fields are not future post-author validation
goals.

The `material.assignment.v1` entrypoint has two typed pauses inside one active
leaf attempt. Keep its invocation bytes unchanged. The first call seals
preparation and returns `awaiting_decision`; author only the prescribed
`native/raw/material_decision_patch.json`, then call the same entrypoint again.
The second call applies the decision, seals exact post-apply OVRTX evidence, and
returns `awaiting_review`; visually review every prescribed binding, write only
`native/raw/material_post_apply_review.json`, and call the same entrypoint a
third time. Only a terminal `passed` result with the exact output bytes and
resource-release receipt may reach `complete-leaf`. A conditional result is a
failed required leaf, not Material success.

Keep the two canonical visual identities distinct in a complete release graph.
`validation.canonical-ovrtx-evidence.v1` is the Articulation review envelope and
must retain the exact authored Articulation bytes. After accepted Articulation
publication, carry the exact output through Material, Texture, Physics
inspection/apply, SimReady conformance, and portable packaging. Run
`asset.final-ovrtx-evidence.v1` only on that exact portable package, alongside
SimReady validation. The combined terminal invocation must copy every required
same-run predecessor receipt and may pass only when its projector proves the
Articulation-to-Material-to-Texture-to-Physics-to-SimReady/package byte chain.

Stop only at `finalize_receipts`, a failed/cancelled leaf, or a complete terminal
receipt. The launcher owns parent resource teardown and seals that release after
you return at `finalize_receipts`. Do not call a selector, nested coordinator,
provider, fixed stage pipeline, compatibility command, classic fallback, or any
hidden fallback. There is no built-in total domain or stage order in agentic
mode.
"""
    return f"""Use the `content-workflow-asset` skill.

{action} the frozen composed-asset request through every required stage:

- Request: `{request_path}`
- Durable state: `{run_state_path}`
- Run directory: `{run_dir}`

Read the request and verified state before acting. Never edit `request.json` or
`asset_run.json` directly. Use only the transition command documented by the
umbrella skill. Before every stage attempt, inspect evidence and seal a typed
coordinator plan; after every domain result, seal a typed evidence review and
choose accept, await_review, refine, revisit, or stop. Preserve every accepted
predecessor and pass its exact output asset into the next domain workflow.

{first_executor} Inspect Geometry's durable USDC, manifest,
validation evidence, optimization status, and required OVRTX evidence; a
rejected or incomplete Geometry handoff cannot enter Articulation. Stop after
publishing articulation candidates when review is required. After review,
continue through Material, Texture, Physics, Validation, and final self-contained
package/report publication. On a concrete blocker, record the current stage
failure and leave all completed stages resumable.

You are the one long-running cross-domain reasoning loop. Interpret the prompt,
author Material and Physics decision artifacts, inspect Texture and simulation
evidence, select the Validation task/templates, and revise or revisit stages
within the recorded budgets. Follow the umbrella skill's typed domain
invocations; do not launch recursive coding agents from any domain workflow.
The external geometry-authoring job is a typed nested workflow operation. Its
provider may return only immutable exported geometry and bounded typed
evidence; provider-native source remains opaque, it never owns this asset run,
and the workflow must never import or execute it.
"""


def _result_without_child(
    run_dir: Path,
    *,
    terminal: AssetTerminalValidation,
    suffix: str,
    returncode: int,
) -> AssetRunResult:
    terminal_path = run_dir / "terminal_validation.json"
    atomic_write_json(terminal_path, terminal)
    return AssetRunResult(
        run_dir=run_dir,
        request_path=run_dir / "request.json",
        run_state_path=run_dir / "asset_run.json",
        prompt_path=run_dir / f"agent_{suffix}_prompt.md",
        child_output_path=run_dir / f"child-{suffix}-output.log",
        child_final_path=run_dir / f"child-{suffix}-final.md",
        terminal_validation_path=terminal_path,
        returncode=returncode,
        completed=terminal.valid,
        needs_review=False,
    )


def _merge_json_objects(values: list[str]) -> dict[str, object] | None:
    merged: dict[str, object] = {}
    for raw in values:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Agent config JSON is not valid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("Agent config JSON must be an object")
        merged.update(parsed)
    return merged or None


def _parameter_values(
    entries: list[str],
) -> dict[str, str | int | float | bool]:
    values: dict[str, str | int | float | bool] = {}
    for entry in entries:
        name, separator, raw = entry.partition("=")
        name = name.strip()
        if not separator or not name:
            raise ValueError(f"Invalid CAD parameter {entry!r}; expected NAME=VALUE")
        try:
            parsed: object = json.loads(raw)
        except json.JSONDecodeError:
            parsed = raw
        if not isinstance(parsed, str | int | float | bool):
            raise ValueError(f"CAD parameter {name!r} must be a scalar")
        if isinstance(parsed, float) and not math.isfinite(parsed):
            raise ValueError(f"CAD parameter {name!r} must be finite")
        if name in values:
            raise ValueError(f"CAD parameter {name!r} was repeated")
        values[name] = parsed
    return values


def _required_cad_outputs(entries: list[str]) -> list[str]:
    outputs = list(dict.fromkeys(entries))
    if not {"usd", "usda"}.intersection(outputs):
        outputs.append("usd")
    return outputs


def _parameter_variants(path: Path | None) -> list[AssetCadParameterVariant]:
    if path is None:
        return []
    resolved = path.expanduser().resolve()
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Could not read CAD variants JSON at {resolved}: {exc}"
        ) from exc
    if not isinstance(payload, list):
        raise ValueError("CAD variants JSON must contain an array")
    return [AssetCadParameterVariant.model_validate(item) for item in payload]


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower() or "asset"
