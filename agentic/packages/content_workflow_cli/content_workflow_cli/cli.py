# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Command-line interface for content-workflow-cli."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from content_agent_workflows.common.artifacts import atomic_write_text
from dotenv import dotenv_values

from .controlled_artifact_cli import (
    write_controlled_json_artifact,
)
from .live_view import add_live_view_args, run_with_live_view
from .material_authoring_runner import (
    add_material_authoring_subcommand,
    material_authoring_skills_available,
)
from .mesh_segmentation_runner import (
    CODEX_EXECUTION_HOST,
    DEFAULT_CODEX_CONTAINER_IMAGE,
    DEFAULT_MESH_SEGMENTATION_CASE_TIMEOUT_SECONDS,
    DEFAULT_MESH_SEGMENTATION_CHILD_TIMEOUT_SECONDS,
    MeshSegmentationConfig,
    run_mesh_segmentation,
)

if TYPE_CHECKING:
    from content_agent_workflows.articulation import ArticulationFinalizationResult
    from content_agent_workflows.physics import PhysicsVompMassConfig

from .articulation_capability_runner import (
    add_articulation_capability_subcommands,
)
from .articulation_runner import (
    ARTICULATION_EXECUTION_FIXED,
    ARTICULATION_EXECUTION_SKILL_ROUTED,
    ArticulationRunConfig,
    apply_articulation_agent_step,
    finalize_articulation_agent_step,
    prepare_articulation_agent_step,
    resume_articulation_workflow,
    review_articulation_workflow,
    revise_articulation_graph_workflow,
    run_articulation_workflow,
)
from .asset_runner import add_asset_subcommands
from .cad_to_simready_runner import (
    add_cad_to_simready_subcommands,
    add_physics_runtime_preflight_args,
    handle_physics_runtime_preflight,
)
from .external_tuning_broker import (
    DEFAULT_EXTERNAL_REFINE_MAX_ITERATIONS,
    DEFAULT_EXTERNAL_SWEEP_DEADLINE_SECONDS,
)
from .geometry_runner import add_geometry_subcommands
from .runner import (
    CLAUDE_EXECUTION_CLI,
    CLAUDE_EXECUTION_SDK,
    CODEX_SANDBOX_DANGER_FULL_ACCESS,
    CODEX_SANDBOX_WORKSPACE_WRITE,
    DEFAULT_MATERIAL_RESTORE_TIMEOUT_SECONDS,
    DEFAULT_VQA_REFINEMENT_MAX_ITERATIONS,
    PROMPT_MODE_SKILL_ROUTED,
    RUNNER_CLAUDE,
    RUNNER_CODEX,
    USD_CLI_UNAVAILABLE_HINT,
    MaterialAssignConfig,
    PhysicsApplyConfig,
    _latest_physics_finalize_record,
    _resolve_materials_usd_from_manifest,
    codex_base_url_from_responses_url,
    find_repo_root,
    preflight_codex_windows_support,
    run_material_assignment,
    run_physics_apply,
)
from .scene_runner import SceneRunConfig, resume_scene_workflow, run_scene_workflow
from .texture_runner import add_texture_subcommands
from .trace import build_trace
from .tuning_broker import (
    DEFAULT_AGENTIC_TUNE_MAX_ITERATIONS,
    DEFAULT_SWEEP_DEADLINE_SECONDS,
)
from .usd_cli_backend import usd_cli_source_distributed

if TYPE_CHECKING:
    # Type-only: the runtime imports stay inside the handlers so unrelated
    # subcommands do not pay for the workflow and validation packages.
    from content_agent_workflows.validation import ValidationWorkflowRun
    from world_understanding.validation import ValidationRequest

SUPPORTED_CLAUDE_CONFIG_KEYS = frozenset({"env", "maxBudgetUsd", "settings"})
CLI_NAME = "content-workflow-cli"
CONVERT_TO_USD_OUTPUT_FORMATS = ("usd", "usda", "usdc", "usdz")
DEFAULT_CONVERTER_TIMEOUT_S = 120.0
IMAGE_REFERENCE_SUFFIXES = frozenset(
    {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
)
VIDEO_REFERENCE_SUFFIXES = frozenset({".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"})
REFERENCE_DIRECTORY_SUFFIXES = IMAGE_REFERENCE_SUFFIXES | frozenset(
    {".doc", ".docx", ".md", ".pdf", ".txt"}
)


class _DirectExecutorAction(argparse.Action):
    """Track use of the deprecated direct-executor spelling."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: object,
        option_string: str | None = None,
    ) -> None:
        del parser, values
        setattr(namespace, self.dest, True)
        if option_string == "--deterministic-workflow":
            setattr(namespace, "used_deterministic_workflow_alias", True)


def _normalize_cli_argv(argv: list[str]) -> list[str]:
    """Attach option-like Texture view values before argparse sees them."""

    normalized: list[str] = []
    index = 0
    while index < len(argv):
        if (
            argv[index] == "--evidence-view"
            and index + 1 < len(argv)
            and argv[index + 1] == "-z"
        ):
            normalized.append("--evidence-view=-z")
            index += 2
            continue
        normalized.append(argv[index])
        index += 1
    return normalized


def main(argv: list[str] | None = None) -> int:
    try:
        parser = build_parser()
        raw_argv = sys.argv[1:] if argv is None else argv
        args = parser.parse_args(_normalize_cli_argv(raw_argv))
        if not hasattr(args, "handler"):
            parser.print_help()
            return 1
        if getattr(args, "live_view", False):
            return run_with_live_view(args)
        return int(args.handler(args))
    except SystemExit as exc:
        # argparse reports malformed command plans with SystemExit.  Treat that
        # as an ordinary CLI result so workflow callers can persist their final
        # failure artifact instead of losing control of the orchestration run.
        if exc.code is None:
            return 0
        return exc.code if isinstance(exc.code, int) else 2
    except KeyboardInterrupt:
        # Ctrl-C is an interruption, not a configuration error, and must not be
        # reported as one. Handlers print their own recovery guidance first.
        return 130
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"{CLI_NAME}: error: {exc}", file=sys.stderr)
        return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=CLI_NAME,
        description="Run agentic asset workflows with usd-cli scene operations.",
    )
    subparsers = parser.add_subparsers(dest="command")

    convert_to_usd = subparsers.add_parser(
        "convert-to-usd",
        aliases=["convert"],
        help="Route a source asset to a requested OpenUSD file.",
    )
    _add_convert_to_usd_args(convert_to_usd)
    convert_to_usd.set_defaults(handler=_handle_convert_to_usd)

    preflight = subparsers.add_parser(
        "preflight",
        help="Prepare workflow dependencies without running the full workflow.",
    )
    preflight_subparsers = preflight.add_subparsers(dest="preflight_command")
    convert_to_usd_preflight = preflight_subparsers.add_parser(
        "convert-to-usd",
        aliases=["convert"],
        help="Install/check the converter dependency implied by a source asset.",
    )
    _add_convert_to_usd_preflight_args(convert_to_usd_preflight)
    convert_to_usd_preflight.set_defaults(handler=_handle_convert_to_usd_preflight)
    simready_preflight = preflight_subparsers.add_parser(
        "simready-foundation",
        aliases=["simready"],
        help="Install/check SimReady Foundation profile tooling.",
    )
    _add_simready_preflight_args(simready_preflight)
    simready_preflight.set_defaults(handler=_handle_simready_preflight)
    physics_runtime_preflight = preflight_subparsers.add_parser(
        "physics-runtime",
        aliases=["physics"],
        help="Install/check the isolated OvPhysX runtime used by physics validation.",
    )
    add_physics_runtime_preflight_args(physics_runtime_preflight)
    physics_runtime_preflight.set_defaults(handler=handle_physics_runtime_preflight)
    articulation_platform_preflight = preflight_subparsers.add_parser(
        "articulation-authoring-platform",
        aliases=["articulation-platform"],
        help="Check that descriptor-sealed Joint Rigger authoring is supported.",
    )
    articulation_platform_preflight.set_defaults(
        handler=_handle_articulation_authoring_platform_preflight
    )
    claude_sandbox_preflight = preflight_subparsers.add_parser(
        "claude-sandbox",
        help="Report whether the current host supports the Claude child sandbox.",
    )
    claude_sandbox_preflight.set_defaults(handler=_handle_claude_sandbox_preflight)

    simready = subparsers.add_parser(
        "simready",
        help="SimReady static conformance and dynamic runtime validation workflows.",
    )
    simready_subparsers = simready.add_subparsers(dest="simready_command")
    simready_validate = simready_subparsers.add_parser(
        "validate-profile",
        aliases=["validate"],
        help="Run formal SimReady Foundation profile validation.",
    )
    _add_simready_validate_args(simready_validate)
    simready_validate.set_defaults(handler=_handle_simready_validate_profile)
    simready_conform = simready_subparsers.add_parser(
        "conform-profile",
        aliases=["conform"],
        help="Route staged SimReady profile conformance through Foundation.",
    )
    _add_simready_conform_args(simready_conform)
    simready_conform.set_defaults(handler=_handle_simready_conform_profile)
    simready_runtime_validate = simready_subparsers.add_parser(
        "validate-runtime",
        aliases=["runtime"],
        help="Run dynamic SimReady Benchmark validation in an isolated runtime.",
    )
    _add_simready_runtime_validation_args(simready_runtime_validate)
    simready_runtime_validate.set_defaults(handler=_handle_simready_validate_runtime)

    auth = subparsers.add_parser("auth", help="Codex authentication utilities.")
    auth_subparsers = auth.add_subparsers(dest="auth_command")
    auth_login = auth_subparsers.add_parser(
        "login",
        help="Start a Codex ChatGPT/OAuth login.",
    )
    auth_login.add_argument(
        "--device-code",
        action="store_true",
        help="Use Codex device auth instead of browser callback login.",
    )
    auth_login.set_defaults(handler=_handle_auth_login)

    auth_status = auth_subparsers.add_parser(
        "status",
        help="Print the active Codex account state.",
    )
    auth_status.add_argument(
        "--sandbox-smoke",
        action="store_true",
        help=(
            "Explicit alias for auth status's mandatory direct and SDK-bridge "
            "workspace-write execution probes."
        ),
    )
    auth_status.set_defaults(handler=_handle_auth_status)

    materials = subparsers.add_parser(
        "materials", help="Material authoring and assignment workflows."
    )
    materials_subparsers = materials.add_subparsers(dest="materials_command")
    if material_authoring_skills_available():
        add_material_authoring_subcommand(materials_subparsers)
    for name, help_text in (
        (
            "assign",
            "Assign material-library materials to a USD asset through workflow skills.",
        ),
        (
            "apply",
            "Alias for materials assign, matching physics apply command naming.",
        ),
    ):
        assign = materials_subparsers.add_parser(
            name,
            help=help_text,
        )
        _add_assign_args(assign)
        assign.set_defaults(handler=_handle_materials_assign)

    articulation = subparsers.add_parser(
        "articulation",
        help="Reviewed, resumable Joint Agent articulation workflows.",
    )
    articulation_subparsers = articulation.add_subparsers(
        dest="articulation_command",
        metavar=(
            "{run,review,revise-graph,resume,agentic-leaf,publish-preparation,"
            "validate-preparation,propose,validate-attempt,bind-output-evidence,"
            "project-graph-apply,project-gate3a,project-gate3b,project-dynamic}"
        ),
    )
    add_articulation_capability_subcommands(articulation_subparsers)
    articulation_run = articulation_subparsers.add_parser(
        "run",
        help="Infer, review, and author source-bound articulation-v1 joints.",
    )
    _add_articulation_run_args(articulation_run)
    articulation_run.set_defaults(handler=_handle_articulation_run)
    articulation_review = articulation_subparsers.add_parser(
        "review",
        help="Bind exact candidate decisions and resume an articulation run.",
    )
    _add_articulation_review_args(articulation_review)
    articulation_review.set_defaults(handler=_handle_articulation_review)
    articulation_revision = articulation_subparsers.add_parser(
        "revise-graph",
        help=(
            "Persist a typed immutable graph revision and reopen exact-digest review."
        ),
    )
    _add_articulation_revision_args(articulation_revision)
    articulation_revision.set_defaults(handler=_handle_articulation_revision)
    articulation_resume = articulation_subparsers.add_parser(
        "resume",
        help="Resume an interrupted articulation run without changing its request.",
    )
    _add_articulation_resume_args(articulation_resume)
    articulation_resume.set_defaults(handler=_handle_articulation_resume)
    articulation_agent_prepare = articulation_subparsers.add_parser(
        "_agent-prepare",
        help=argparse.SUPPRESS,
    )
    articulation_agent_prepare.add_argument("--run-dir", required=True, type=Path)
    articulation_agent_prepare.set_defaults(handler=_handle_articulation_agent_prepare)
    articulation_agent_apply = articulation_subparsers.add_parser(
        "_agent-apply",
        help=argparse.SUPPRESS,
    )
    articulation_agent_apply.add_argument("--run-dir", required=True, type=Path)
    articulation_agent_apply.add_argument(
        "--decision-patch",
        required=True,
        type=Path,
    )
    articulation_agent_apply.set_defaults(handler=_handle_articulation_agent_apply)
    articulation_agent_finalize = articulation_subparsers.add_parser(
        "_agent-finalize",
        help=argparse.SUPPRESS,
    )
    articulation_agent_finalize.add_argument("--run-dir", required=True, type=Path)
    articulation_agent_finalize.add_argument(
        "--post-review-patch",
        required=True,
        type=Path,
    )
    articulation_agent_finalize.set_defaults(
        handler=_handle_articulation_agent_finalize
    )

    physics = subparsers.add_parser("physics", help="Physics authoring workflows.")
    physics_subparsers = physics.add_subparsers(dest="physics_command")
    physics_apply = physics_subparsers.add_parser(
        "apply",
        help="Infer and apply USD physics schemas, then optionally validate by simulation.",
    )
    _add_physics_apply_args(physics_apply)
    physics_apply.set_defaults(handler=_handle_physics_apply)
    physics_refine_external = physics_subparsers.add_parser(
        "refine-external",
        help=(
            "Agent-managed external-runtime (BYOR) refinement: budgeted "
            "qualification-gated tune-external sweeps with the coding agent "
            "as judge and refiner."
        ),
    )
    _add_physics_refine_external_args(physics_refine_external)
    physics_refine_external.set_defaults(handler=_handle_physics_refine_external)

    mesh_segmentation = subparsers.add_parser(
        "mesh-segmentation",
        aliases=["segment-mesh"],
        help="Fresh-session fused-mesh segmentation workflows.",
    )
    mesh_segmentation_subparsers = mesh_segmentation.add_subparsers(
        dest="mesh_segmentation_command"
    )
    mesh_segmentation_run = mesh_segmentation_subparsers.add_parser(
        "run",
        help="Invoke the mesh-segmentation skill in a new isolated child-agent session.",
    )
    _add_mesh_segmentation_run_args(mesh_segmentation_run)
    mesh_segmentation_run.set_defaults(handler=_handle_mesh_segmentation_run)

    validate = subparsers.add_parser(
        "validate",
        help="Resumable prompt-to-report validation workflows.",
    )
    validate_subparsers = validate.add_subparsers(dest="validate_command")
    validate_run = validate_subparsers.add_parser(
        "run",
        help="Validate an asset against a prompt and publish an evidence-backed report.",
    )
    _add_validate_run_args(validate_run)
    validate_run.set_defaults(handler=_handle_validate_run)
    validate_resume = validate_subparsers.add_parser(
        "resume",
        help="Resume an interrupted validation run at its next unfinished check.",
    )
    _add_validate_resume_args(validate_resume)
    validate_resume.set_defaults(handler=_handle_validate_resume)
    validate_prepare = validate_subparsers.add_parser(
        "prepare",
        help="Freeze an explicit outer-selected Validation capability request.",
    )
    _add_validate_prepare_args(validate_prepare)
    validate_prepare.set_defaults(handler=_handle_validate_prepare)
    validate_check = validate_subparsers.add_parser(
        "check",
        help="Execute exactly one capability from a focused Validation request.",
    )
    _add_validate_check_args(validate_check)
    validate_check.set_defaults(handler=_handle_validate_check)
    validate_finalize = validate_subparsers.add_parser(
        "finalize",
        help="Finalize selected operation results into the standard report bundle.",
    )
    _add_validate_finalize_args(validate_finalize)
    validate_finalize.set_defaults(handler=_handle_validate_finalize)
    validate_ingest_verified = validate_subparsers.add_parser(
        "ingest-verified-operation-result",
        help="Re-verify and ingest one digest-bound external operation result.",
    )
    _add_validate_ingest_verified_args(validate_ingest_verified)
    validate_ingest_verified.set_defaults(
        handler=_handle_validate_ingest_verified_operation_result
    )
    validate_produce_visual = validate_subparsers.add_parser(
        "produce-canonical-visual-evidence",
        help="Produce digest-bound post-mutation OVRTX evidence without judging it.",
    )
    _add_validate_produce_visual_args(validate_produce_visual)
    validate_produce_visual.set_defaults(
        handler=_handle_validate_produce_canonical_visual_evidence
    )
    validate_collect_evidence = validate_subparsers.add_parser(
        "collect-evidence",
        help="Bind native, image, package, and optional cross-stage assessment evidence.",
    )
    _add_validate_collect_evidence_args(validate_collect_evidence)
    validate_collect_evidence.set_defaults(handler=_handle_validate_collect_evidence)
    validate_assess = validate_subparsers.add_parser(
        "assess",
        help=(
            "Persist an outer-authored assessment and run the bounded "
            "non-mutating finalizer."
        ),
    )
    _add_validate_assess_args(validate_assess)
    validate_assess.set_defaults(handler=_handle_validate_assess)
    validate_review_assessment = validate_subparsers.add_parser(
        "review-assessment",
        help="Review the exact assessment output and seal its terminal receipt.",
    )
    _add_validate_review_assessment_args(validate_review_assessment)
    validate_review_assessment.set_defaults(handler=_handle_validate_review_assessment)

    scene = subparsers.add_parser(
        "scene",
        help="Multi-phase large-scene workflows.",
    )
    scene_subparsers = scene.add_subparsers(dest="scene_command")
    scene_run = scene_subparsers.add_parser(
        "run",
        help="Run decomposition, asset tasks, and collection through one agent.",
    )
    _add_scene_run_args(scene_run)
    scene_run.set_defaults(handler=_handle_scene_run)
    scene_resume = scene_subparsers.add_parser(
        "resume",
        help="Resume a prepared, interrupted, or failed large-scene run.",
    )
    _add_scene_resume_args(scene_resume)
    scene_resume.set_defaults(handler=_handle_scene_resume)
    for name, help_text, handler, operation_name in (
        (
            "phase",
            "Operate the durable large-scene phase state.",
            _handle_scene_phase,
            "PHASE_COMMAND",
        ),
        (
            "decompose",
            "Decompose the staged source scene for a large-scene run.",
            _handle_scene_decompose,
            "SCENE",
        ),
        (
            "process",
            "Operate durable per-asset Workflow 2 state.",
            _handle_scene_process,
            "PROCESS_COMMAND",
        ),
        (
            "material-task",
            "Survey or execute Workflow 2 material work items.",
            _handle_scene_material_task,
            "MATERIAL_COMMAND",
        ),
    ):
        command = scene_subparsers.add_parser(name, help=help_text)
        command.add_argument("scene_operation", metavar=operation_name)
        command.add_argument("scene_operation_args", nargs=argparse.REMAINDER)
        command.set_defaults(handler=handler)
    scene_collect = scene_subparsers.add_parser(
        "collect",
        help="Collect validated per-asset results onto the original topology.",
    )
    scene_collect.add_argument("--request", type=Path, required=True)
    scene_collect.set_defaults(handler=_handle_scene_collect)

    add_texture_subcommands(subparsers)
    add_geometry_subcommands(subparsers)
    add_asset_subcommands(subparsers)
    add_cad_to_simready_subcommands(subparsers)

    artifact = subparsers.add_parser(
        "artifact",
        help="Confined child-workflow artifact utilities.",
    )
    artifact_subparsers = artifact.add_subparsers(dest="artifact_command")
    artifact_write_json = artifact_subparsers.add_parser(
        "write-json",
        help="Atomically publish one non-clobbering JSON artifact from stdin.",
    )
    artifact_write_json.add_argument(
        "--output",
        required=True,
        help="Canonical forward-slash path relative to the child run directory.",
    )
    artifact_write_json.add_argument(
        "--json",
        dest="json_document",
        help="Complete JSON document. When omitted, the document is read from stdin.",
    )
    artifact_write_json.add_argument(
        "--replace",
        action="store_true",
        help="Atomically replace an approved child-owned decision patch.",
    )
    artifact_write_json.set_defaults(handler=_handle_artifact_write_json)

    trace = subparsers.add_parser("trace", help="Trace utilities.")
    trace_subparsers = trace.add_subparsers(dest="trace_command")
    trace_build = trace_subparsers.add_parser(
        "build",
        help="Build operation_trace and replay_manifest files from a run directory.",
    )
    trace_build.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help="Existing content-workflow-cli run directory.",
    )
    trace_build.set_defaults(handler=_handle_trace_build)
    return parser


def _handle_artifact_write_json(args: argparse.Namespace) -> int:
    """Publish model-authored JSON as data beneath the confined child run."""

    write_controlled_json_artifact(
        output=args.output,
        json_document=args.json_document,
        replace_existing=args.replace,
    )
    return 0


def _add_convert_to_usd_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "source_asset",
        type=Path,
        help="Input source asset path.",
    )
    parser.add_argument(
        "output_usd",
        type=Path,
        nargs="?",
        help=(
            "Output USD file. Defaults to ./<source-stem>.usda in the current "
            "working directory."
        ),
    )
    parser.add_argument(
        "--install-missing",
        dest="install_missing",
        action="store_true",
        default=True,
        help="Install only the converter package implied by the source extension. This is the default.",
    )
    parser.add_argument(
        "--no-install-missing",
        dest="install_missing",
        action="store_false",
        help="Do not install missing converter dependencies before converting.",
    )
    parser.add_argument(
        "--converter-timeout",
        dest="converter_timeout_s",
        type=_positive_float,
        default=DEFAULT_CONVERTER_TIMEOUT_S,
        metavar="SECONDS",
        help=(
            "Positive finite converter execution timeout in seconds. "
            f"Defaults to {DEFAULT_CONVERTER_TIMEOUT_S:g}."
        ),
    )
    parser.add_argument(
        "--output-format",
        choices=CONVERT_TO_USD_OUTPUT_FORMATS,
        default=None,
        help=(
            "Output USD format to use when OUTPUT_USD is omitted. If OUTPUT_USD "
            "is provided, its extension must match this format."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory for canonical workflow artifacts. This does not "
            "change the default output USD location."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume an existing --output-dir run only when its recorded request "
            "and source identity match."
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional path to write the normalized conversion report JSON.",
    )
    parser.add_argument(
        "--markdown-report",
        type=Path,
        default=None,
        help="Optional path to write the normalized conversion report Markdown.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full conversion report or workflow result JSON.",
    )


def _add_convert_to_usd_preflight_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "source_asset",
        type=Path,
        help="Input source asset path.",
    )
    parser.add_argument(
        "--install-missing",
        dest="install_missing",
        action="store_true",
        default=True,
        help="Install the converter package implied by the source extension. This is the default.",
    )
    parser.add_argument(
        "--no-install-missing",
        dest="install_missing",
        action="store_false",
        help="Check the implied converter dependency without installing it.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional path to write the normalized preflight report JSON.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full preflight report JSON.",
    )


def _add_scene_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--usd", required=True, type=Path, help="Input USD scene.")
    parser.add_argument(
        "--task",
        action="append",
        required=True,
        help="Requested asset-task domain, such as material or physics. May be repeated.",
    )
    parser.add_argument(
        "--reference-image",
        action="append",
        type=Path,
        default=[],
        help="Explicit reference image. May be repeated.",
    )
    parser.add_argument(
        "--reference",
        action="append",
        type=Path,
        default=[],
        help="Explicit image or document reference. May be repeated.",
    )
    parser.add_argument(
        "--reference-dir",
        action="append",
        type=Path,
        default=[],
        help=(
            "Directory of image/document references. Direct child files are expanded "
            "in deterministic filename order. May be repeated."
        ),
    )
    parser.add_argument(
        "--materials-yaml",
        type=Path,
        default=None,
        help="Material library metadata YAML; required for --task material.",
    )
    parser.add_argument(
        "--materials-usd",
        type=Path,
        default=None,
        help="Optional material library USD override.",
    )
    parser.add_argument(
        "--material-candidate-space",
        choices=["source", "inspection"],
        default="source",
        help="Material candidate path space. Defaults to authorable source prims.",
    )
    parser.add_argument(
        "--respect-existing-material-bindings",
        "--respect-existing-materials",
        dest="respect_existing_material_bindings",
        action="store_true",
        default=False,
        help="Preserve existing material bindings as task evidence.",
    )
    parser.add_argument(
        "--ignore-existing-material-bindings",
        "--ignore-existing-materials",
        dest="respect_existing_material_bindings",
        action="store_false",
        help="Do not preserve existing material bindings. This is the default.",
    )
    parser.add_argument(
        "--additional-instructions",
        default=None,
        help="Scene-level task guidance carried through all workflow phases.",
    )
    parser.add_argument(
        "--additional-instructions-file",
        type=Path,
        default=None,
        help="Markdown or text file containing scene-level task guidance.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional durable run identifier. Defaults to a timestamped scene name.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Run directory. Defaults to runs/<scene>-<timestamp> at repo root.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root. The child agent launches from <repo-root>/agentic.",
    )
    parser.add_argument(
        "--runner",
        choices=[RUNNER_CODEX, RUNNER_CLAUDE],
        default=RUNNER_CODEX,
        help="Child agent runner.",
    )
    _add_child_model_args(parser)
    parser.add_argument(
        "--codex-base-url",
        default=os.getenv("CONTENT_AGENT_CODEX_BASE_URL"),
        help="Optional OpenAI-compatible base URL for the Codex SDK.",
    )
    parser.add_argument(
        "--codex-sandbox-mode",
        choices=[CODEX_SANDBOX_WORKSPACE_WRITE],
        default=_restricted_default_codex_sandbox_mode(),
        help="Codex child sandbox mode.",
    )
    parser.add_argument("--codex-config-json", action="append", default=[])
    parser.add_argument("--codex-config-file", action="append", type=Path, default=[])
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
        help=(
            "How --runner claude launches the child agent: 'sdk' (default, "
            "Claude Agent SDK via Node) or 'cli' (spawn the local `claude` CLI "
            "directly, reusing its `claude login` OAuth session)."
        ),
    )
    parser.add_argument("--claude-config-json", action="append", default=[])
    parser.add_argument("--claude-config-file", action="append", type=Path, default=[])
    parser.add_argument(
        "--scene-tool-timeout",
        type=float,
        default=60.0,
        help="Seconds to wait for each low-level usd-cli operation.",
    )
    parser.add_argument(
        "--child-timeout",
        type=float,
        default=_env_float("CONTENT_AGENT_SCENE_CHILD_TIMEOUT", 1800.0),
        help="Seconds to wait for the long-running child. Use 0 to disable the timeout.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the resolved request, run state, prompt, and trace without launching.",
    )


def _add_mesh_segmentation_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--workflow-skill",
        choices=[
            "content-workflow-mesh-segmentation",
        ],
        default="content-workflow-mesh-segmentation",
        help="Mesh-segmentation workflow contract staged into the fresh session.",
    )
    parser.add_argument(
        "--asset",
        required=True,
        type=Path,
        help="Single fused USD asset to segment.",
    )
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=None,
        help=(
            "Optional dependency root copied with the asset. Use this for USD "
            "assets with relative textures, payloads, or references."
        ),
    )
    parser.add_argument(
        "--target-prim",
        default=None,
        help="Optional absolute UsdGeomMesh prim path.",
    )
    parser.add_argument(
        "--target-semantic-part",
        action="append",
        default=[],
        help=(
            "Semantic part to segment without running part recognition. May be "
            "repeated to provide a complete user vocabulary; the agent chooses "
            "processing order and every non-target face is assigned to `other`."
        ),
    )
    parser.add_argument(
        "--continue-from-run",
        type=Path,
        default=None,
        help=(
            "Validated completed run whose immutable fragment map and "
            "nonzero semantic labels are copied into this fresh run as locked "
            "continuation state."
        ),
    )
    parser.add_argument(
        "--reference-image",
        action="append",
        type=Path,
        default=[],
        help="Guiding appearance reference image. May be repeated.",
    )
    parser.add_argument(
        "--reference-dir",
        action="append",
        type=Path,
        default=[],
        help=(
            "Directory of guiding reference images. Direct child images are "
            "expanded in deterministic filename order. May be repeated."
        ),
    )
    parser.add_argument(
        "--expected-asset-sha256",
        default=None,
        help=(
            "Optional lowercase SHA-256 that the staged asset must match. This "
            "binds a previously validated publisher input to child staging."
        ),
    )
    parser.add_argument(
        "--expected-reference-sha256",
        action="append",
        default=[],
        help=(
            "Expected lowercase SHA-256 for each expanded reference image, in "
            "the same order. Must be repeated for the complete set when used."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "New run directory. Existing directories are rejected to preserve "
            "fresh-session isolation."
        ),
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional durable run identifier used when --output-dir is omitted.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root containing agentic/.agents/skills.",
    )
    parser.add_argument(
        "--runner",
        choices=[RUNNER_CODEX, RUNNER_CLAUDE],
        default=RUNNER_CODEX,
        help="Fresh child-agent runner.",
    )
    _add_child_model_args(parser)
    parser.add_argument(
        "--codex-base-url",
        default=None,
        help="Optional OpenAI-compatible base URL for the Codex SDK.",
    )
    parser.add_argument(
        "--codex-responses-url",
        default=os.getenv("CONTENT_AGENT_CODEX_RESPONSES_URL"),
        help=(
            "Exact OpenAI-compatible Responses endpoint. The launcher records "
            "this URL and derives the SDK base URL from its /responses suffix."
        ),
    )
    parser.add_argument(
        "--codex-api-key-env",
        default=os.getenv("CONTENT_AGENT_CODEX_API_KEY_ENV"),
        help=(
            "Environment variable containing the Codex provider key. Its value "
            "is mapped to OPENAI_API_KEY only in the child process."
        ),
    )
    parser.add_argument(
        "--allow-codex-configured-auth",
        action="store_true",
        help=(
            "Explicitly allow the host Codex SDK to use its configured local "
            "authentication instead of a named provider key and endpoint."
        ),
    )
    parser.add_argument(
        "--codex-sandbox-mode",
        choices=[
            CODEX_SANDBOX_WORKSPACE_WRITE,
            CODEX_SANDBOX_DANGER_FULL_ACCESS,
        ],
        default=_default_codex_sandbox_mode(allow_danger_full_access=True),
        help=(
            "Codex child sandbox mode. Use danger-full-access only for a "
            "host-mode child when nested bubblewrap is unavailable."
        ),
    )
    parser.add_argument(
        "--codex-execution-mode",
        choices=[CODEX_EXECUTION_HOST],
        default=CODEX_EXECUTION_HOST,
        help="Run the mesh-segmentation Codex child on the host.",
    )
    parser.add_argument(
        "--allow-unsafe-host-child",
        action="store_true",
        help=(
            "Explicitly authorize an unsandboxed host Codex child when "
            "--codex-sandbox-mode=danger-full-access is selected."
        ),
    )
    parser.add_argument(
        "--codex-container-image",
        default=os.getenv(
            "CONTENT_AGENT_CODEX_CONTAINER_IMAGE", DEFAULT_CODEX_CONTAINER_IMAGE
        ),
        help=(
            "Reserved compatibility setting. Mesh segmentation currently runs "
            "only in host mode and never starts this image."
        ),
    )
    parser.add_argument("--codex-config-json", action="append", default=[])
    parser.add_argument("--codex-config-file", action="append", type=Path, default=[])
    parser.add_argument(
        "--image-gen-backend",
        default=os.getenv("CONTENT_AGENT_IMAGE_GEN_BACKEND"),
        help=(
            "Explicit World Understanding image-generation backend. When omitted, "
            "the workflow defaults to the coding agent's companion image generator."
        ),
    )
    parser.add_argument(
        "--image-gen-model",
        default=None,
        help=(
            "Image-generation model for an explicit external endpoint, or a "
            "preference recorded for the coding agent's companion image "
            "generator when no endpoint is supplied."
        ),
    )
    parser.add_argument(
        "--image-gen-base-url",
        default=None,
        help=(
            "Optional base URL for the explicitly selected World Understanding "
            "image backend. It does not select a provider by itself."
        ),
    )
    parser.add_argument(
        "--image-gen-api-key-env",
        default=os.getenv("CONTENT_AGENT_IMAGE_GEN_API_KEY_ENV"),
        help="Environment variable containing the image-generation API key.",
    )
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
    parser.add_argument("--claude-config-file", action="append", type=Path, default=[])
    parser.add_argument(
        "--scene-tool-timeout",
        type=float,
        default=60.0,
        help="Seconds to wait for usd-cli readiness.",
    )
    parser.add_argument(
        "--child-timeout",
        type=float,
        default=_env_float(
            "CONTENT_AGENT_MESH_SEGMENTATION_TIMEOUT",
            DEFAULT_MESH_SEGMENTATION_CHILD_TIMEOUT_SECONDS,
        ),
        help=(
            "Seconds to wait for each child turn. This bounds one turn, not "
            "the run: the launcher grants up to --iteration-budget turns, so "
            "worst-case wall clock is also capped by --case-timeout. The "
            f"default {DEFAULT_MESH_SEGMENTATION_CHILD_TIMEOUT_SECONDS:g}-second "
            "turn budget accommodates the shipped small-fixture acceptance run. "
            "Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--iteration-budget",
        type=_positive_int,
        default=12,
        help="Maximum autonomous segmentation/refinement revisions.",
    )
    parser.add_argument(
        "--case-timeout",
        type=float,
        default=DEFAULT_MESH_SEGMENTATION_CASE_TIMEOUT_SECONDS,
        help=(
            "Total seconds across every turn of this case. Each turn is "
            "clamped to whatever remains, so this bounds the case rather than "
            "multiplying with --child-timeout. The default bounds the whole case "
            f"to {DEFAULT_MESH_SEGMENTATION_CASE_TIMEOUT_SECONDS:g} seconds. Use 0 "
            "to disable and let the iteration budget alone decide."
        ),
    )
    parser.add_argument(
        "--memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable required run-scoped observation memory (default: enabled). "
            "Use --no-memory only for controlled baselines."
        ),
    )
    parser.add_argument(
        "--memory-root",
        type=Path,
        default=None,
        help=(
            "Wrapper-owned agent-memory root outside the child output directory. "
            "Mesh runs default to <repo-root>/runs/.memory."
        ),
    )
    parser.add_argument(
        "--additional-instructions",
        default=None,
        help="Optional task guidance included in the frozen request.",
    )
    parser.add_argument(
        "--additional-instructions-file",
        type=Path,
        default=None,
        help="Optional task-guidance file included in the frozen request.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Stage inputs, skills, request, and prompt without launching the child.",
    )


def _add_validate_run_args(
    parser: argparse.ArgumentParser,
    *,
    include_coordinator: bool = True,
) -> None:
    parser.add_argument(
        "--usd",
        required=True,
        type=Path,
        help="USD asset to validate. Validation never modifies this file.",
    )
    parser.add_argument(
        "--task",
        required=True,
        help="Prompt describing what to validate and what to report.",
    )
    parser.add_argument(
        "--reference-image",
        action="append",
        default=[],
        type=Path,
        help="Reference image for look_right. Repeat for multiple references.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Run directory for request, plan, evidence, and report artifacts.",
    )
    parser.add_argument(
        "--template",
        action="append",
        default=[],
        help="Restrict the plan to these templates, in order. Repeat per template.",
    )
    parser.add_argument(
        "--focus-prim",
        action="append",
        default=[],
        help="Limit rendering and review to this prim path. Repeat per prim.",
    )
    parser.add_argument(
        "--embedded-run-state",
        type=Path,
        help=(
            "Exact composed asset_run.json that owns this Validation attempt. "
            "Omit for standalone Validation."
        ),
    )
    _add_validate_render_args(parser)
    _add_validate_report_args(parser)
    parser.add_argument(
        "--policy-file",
        type=Path,
        help=(
            "JSON object merged into the Validation request policy. Relative "
            "artifact paths are resolved against --base-dir by the adapters."
        ),
    )
    if not include_coordinator:
        return
    parser.add_argument(
        "--direct-executor",
        action="store_true",
        help=(
            "Use the legacy outer-preselected deterministic workflow. The "
            "default launches one decision-only planning child and validates "
            "its plan before calling exact Validation adapters."
        ),
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root containing the agentic workflow packages.",
    )
    parser.add_argument(
        "--runner",
        choices=(RUNNER_CODEX, RUNNER_CLAUDE),
        default=None,
        help="Decision-only planning child runner.",
    )
    _add_child_model_args(parser)
    parser.add_argument(
        "--codex-base-url",
        default=None,
    )
    parser.add_argument(
        "--codex-sandbox-mode",
        choices=(CODEX_SANDBOX_WORKSPACE_WRITE,),
        default=None,
    )
    parser.add_argument("--codex-config-json", action="append", default=[])
    parser.add_argument("--codex-config-file", action="append", type=Path, default=[])
    parser.add_argument(
        "--claude-permission-mode",
        choices=("default", "acceptEdits", "bypassPermissions", "plan"),
        default=None,
    )
    parser.add_argument("--claude-max-turns", type=int, default=None)
    parser.add_argument(
        "--claude-execution-mode",
        choices=(CLAUDE_EXECUTION_SDK, CLAUDE_EXECUTION_CLI),
        default=None,
        help="Use the Claude Agent SDK or equivalent authenticated CLI path.",
    )
    parser.add_argument("--claude-config-json", action="append", default=[])
    parser.add_argument("--claude-config-file", action="append", type=Path, default=[])
    parser.add_argument(
        "--child-timeout",
        type=_non_negative_float,
        default=None,
        help="Seconds to wait for the planning child; use 0 to disable.",
    )
    parser.add_argument("--agent-cwd", type=Path, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write selection-free preparation and child prompt without launching.",
    )
    parser.add_argument(
        "--required-capability",
        action="append",
        default=[],
        help=(
            "Capability ID the child plan must select. Repeat for multiple "
            "mandatory coordinator constraints."
        ),
    )


def _add_validate_resume_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Existing validation run directory to resume.",
    )
    parser.add_argument(
        "--recover-orphaned-claims",
        action="store_true",
        help=(
            "Release the previous runner's in-flight check before resuming. "
            "Use only when that runner is confirmed gone, such as after Ctrl-C; "
            "resume otherwise refuses to run alongside a live claim."
        ),
    )
    parser.add_argument(
        "--embedded-run-state",
        type=Path,
        help=(
            "Exact composed asset_run.json for an explicitly embedded request. "
            "Standalone resume omits this option."
        ),
    )
    _add_validate_report_args(parser)


def _add_validate_prepare_args(parser: argparse.ArgumentParser) -> None:
    _add_validate_run_args(parser, include_coordinator=False)
    parser.add_argument(
        "--rule",
        action="append",
        default=[],
        help="Explicit atomic rule ID chosen by the outer reasoner. Repeat in order.",
    )
    parser.add_argument(
        "--profile",
        help="Explicit documented named profile chosen by the outer reasoner.",
    )


def _add_validate_check_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--template", required=True)
    parser.add_argument(
        "--prior-result",
        action="append",
        default=[],
        type=Path,
        help="Exact prior operation_result.json dependency. Repeat as needed.",
    )
    parser.add_argument("--json", action="store_true")


def _add_validate_finalize_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", required=True, type=Path)
    _add_validate_report_args(parser)


def _add_validate_ingest_verified_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--envelope",
        required=True,
        type=Path,
        help="Versioned verified-operation envelope produced by a domain projector.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--json", action="store_true")


def _add_validate_produce_visual_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--usd",
        required=True,
        type=Path,
        help="Exact post-mutation USD to render through shared OVRTX.",
    )
    parser.add_argument(
        "--source-usd",
        required=True,
        type=Path,
        help="Exact pre-mutation source USD whose digest must be retained.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--render-backend",
        required=True,
        choices=("remote", "ovrtx"),
        help="Explicit shared OVRTX backend alias.",
    )
    parser.add_argument(
        "--render-view",
        action="append",
        default=[],
        help="OVRTX view token. Repeat for multiple views; defaults to +x+y+z.",
    )
    parser.add_argument("--render-width", type=_positive_int, default=1024)
    parser.add_argument("--render-height", type=_positive_int, default=1024)
    parser.add_argument(
        "--operation-id",
        default="visual.canonical-ovrtx",
        help="Stable hierarchical operation ID.",
    )
    parser.add_argument(
        "--gate-id",
        default="visual.canonical-evidence",
        help="Stable hierarchical gate ID.",
    )
    parser.add_argument("--json", action="store_true")


def _add_validate_collect_evidence_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--embedded-run-state",
        type=Path,
        help=("Exact composed asset_run.json. Omit for focused standalone Validation."),
    )
    parser.add_argument("--json", action="store_true")


def _add_validate_assess_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Completed focused or embedded Validation run directory to assess.",
    )
    parser.add_argument(
        "--embedded-run-state",
        type=Path,
        help="Exact composed asset_run.json; omit for standalone assessment.",
    )
    parser.add_argument(
        "--assessment",
        required=True,
        type=Path,
        help="Outer-authored typed Validation assessment to persist.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the execution index payload instead of a status line.",
    )


def _add_validate_review_assessment_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Assessed focused or embedded Validation run directory to review.",
    )
    parser.add_argument(
        "--embedded-run-state",
        type=Path,
        help="Exact composed asset_run.json; omit for standalone review.",
    )
    parser.add_argument(
        "--review",
        required=True,
        type=Path,
        help="Outer-authored review draft of the exact assessment output.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the receipt index payload instead of a status line.",
    )


def _add_validate_render_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--render-backend",
        help="Rendering backend used to produce current-run visual evidence.",
    )
    parser.add_argument(
        "--render-view",
        action="append",
        default=[],
        help="Camera view to render. Repeat per view.",
    )
    parser.add_argument(
        "--render-width",
        type=int,
        help="Rendered evidence width in pixels.",
    )
    parser.add_argument(
        "--render-height",
        type=int,
        help="Rendered evidence height in pixels.",
    )


def _add_validate_report_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--base-dir",
        type=Path,
        help="Base directory for resolving relative policy paths. Defaults to cwd.",
    )
    parser.add_argument(
        "--fail-on-warn",
        action="store_true",
        help="Exit non-zero when the verdict is warn.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the validation result as JSON instead of a summary.",
    )


def _add_scene_resume_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help="Existing large-scene run directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and write the resume prompt without launching a child agent.",
    )
    parser.add_argument(
        "--adopt-legacy-request-sha256",
        help=(
            "Explicitly adopt a genuine pre-policy v1 run by supplying the exact "
            "SHA-256 of its request.json. Modern or mixed-schema runs are rejected."
        ),
    )


def _add_simready_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--foundation-root", type=Path)
    parser.add_argument("--foundation-spec-root", type=Path)
    parser.add_argument("--venv", dest="venv_path", type=Path)
    parser.add_argument(
        "--install-missing",
        dest="install_missing",
        action="store_true",
        default=True,
        help="Install missing SimReady Foundation dependencies. This is the default.",
    )
    parser.add_argument(
        "--no-install-missing",
        dest="install_missing",
        action="store_false",
        help="Check only; do not clone or install missing dependencies.",
    )
    parser.add_argument(
        "--update-foundation",
        action="store_true",
        help="Fetch/update a managed Foundation checkout before checking it.",
    )


def _add_simready_preflight_args(parser: argparse.ArgumentParser) -> None:
    _add_simready_runtime_args(parser)
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional path to write the normalized preflight report JSON.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full preflight report JSON.",
    )


def _add_simready_validate_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("asset_path", type=Path, help="USD asset path.")
    parser.add_argument("--profile", default="Prop-Robotics-Neutral")
    parser.add_argument("--profile-version", default="1.0.0")
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--stdout-log", type=Path, default=None)
    parser.add_argument("--stderr-log", type=Path, default=None)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume only when the recorded request and source identity match.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero when profile validation fails.",
    )
    _add_simready_runtime_args(parser)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full normalized validation report JSON.",
    )


def _add_simready_runtime_validation_args(
    parser: argparse.ArgumentParser,
) -> None:
    parser.add_argument("asset_path", type=Path, help="USD asset path.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "Single-run workspace. Reuse invalidates and removes any prior "
            "verified-operation publication before preflight completes."
        ),
    )
    parser.add_argument(
        "--sr-specs",
        dest="sr_specs_path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--engines-toml",
        dest="engines_toml_path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--project-config",
        dest="project_config_path",
        type=Path,
    )
    parser.add_argument(
        "--tests-path",
        dest="tests_paths",
        action="append",
        type=Path,
        default=[],
    )
    parser.add_argument("--feature", dest="features", action="append", default=[])
    parser.add_argument("--test", dest="tests", action="append", default=[])
    parser.add_argument("--runtime", dest="runtimes", action="append", default=[])
    parser.add_argument("--benchmark-executable", type=Path)
    parser.add_argument("--report", dest="report_path", type=Path)
    parser.add_argument("--max-concurrent", type=_positive_int, default=1)
    parser.add_argument(
        "--timeout", dest="timeout_s", type=_positive_float, default=3600.0
    )
    parser.add_argument("--stamp-results", action="store_true")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full normalized runtime report JSON.",
    )


def _add_simready_conform_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("asset_path", type=Path, help="USD asset path.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--profile", default="Prop-Robotics-Neutral")
    parser.add_argument("--profile-version", default="1.0.0")
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--validation-report", type=Path, default=None)
    parser.add_argument("--source-asset", default=None)
    parser.add_argument(
        "--expected-physics-inventory-sha256",
        default=None,
        help=(
            "Mandatory trusted Joint Agent physics-inventory fingerprint when "
            "routing G3A.HYG.001."
        ),
    )
    parser.add_argument(
        "--grasp-prim",
        dest="grasp_prim_path",
        default=None,
        help="Explicit prim whose bounds provide local-coordinate grasp evidence.",
    )
    parser.add_argument("--foundation-root", type=Path, default=None)
    parser.add_argument("--foundation-spec-root", type=Path, default=None)
    parser.add_argument(
        "--venv",
        dest="venv_path",
        type=Path,
        default=None,
        help="Validator venv used to produce the identity-bound validation report.",
    )
    parser.add_argument(
        "--repair", dest="repair_requirements", action="append", default=[]
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume only when the recorded request and source identity match.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero when conformance is blocked or failed.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full normalized conformance report JSON.",
    )


def _add_child_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        metavar="MODEL",
        default=None,
        help=(
            "Optional provider-specific child-agent model ID forwarded without "
            "wrapper enum validation."
        ),
    )
    parser.add_argument(
        "--model-reasoning-effort",
        metavar="EFFORT",
        default=None,
        help=(
            "Optional provider/model-specific reasoning effort forwarded without "
            "wrapper enum validation."
        ),
    )


def _add_assign_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--usd", required=True, type=Path, help="Input USD asset path.")
    parser.add_argument(
        "--reference-image",
        action="append",
        type=Path,
        default=[],
        help="Reference image path. May be repeated.",
    )
    parser.add_argument(
        "--reference",
        action="append",
        type=Path,
        default=[],
        help=(
            "Generic reference file path. Images are attached as reference images; "
            "other readable files such as PDFs/docs are passed by path. May be repeated."
        ),
    )
    parser.add_argument(
        "--materials-yaml",
        required=True,
        type=Path,
        help="Material library metadata YAML.",
    )
    parser.add_argument(
        "--materials-usd",
        type=Path,
        default=None,
        help=(
            "Optional USD material library override. Defaults to library_path "
            "resolved from --materials-yaml."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Run output directory. Defaults to runs/<asset>-<timestamp> under "
            "the current working directory."
        ),
    )
    parser.add_argument(
        "--output-usd",
        type=Path,
        default=None,
        help=(
            "Optional durable USD containing the accepted material assignments. "
            "The source asset is not overwritten. Prefer a .usdc suffix for "
            "geometry-heavy assets to avoid large ASCII outputs."
        ),
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root for the child agent. Defaults to git root.",
    )
    parser.add_argument(
        "--runner",
        choices=[RUNNER_CODEX, RUNNER_CLAUDE],
        default=RUNNER_CODEX,
        help="Child agent runner.",
    )
    _add_child_model_args(parser)
    parser.add_argument(
        "--codex-base-url",
        default=os.getenv("CONTENT_AGENT_CODEX_BASE_URL"),
        help=(
            "Optional OpenAI-compatible base URL for the Codex SDK. Defaults to "
            "the normal Codex CLI provider/auth configuration."
        ),
    )
    parser.add_argument(
        "--codex-responses-url",
        default=os.getenv("CONTENT_AGENT_CODEX_RESPONSES_URL"),
        help=(
            "Exact custom Codex Responses endpoint URL ending in /responses. "
            "Use this for NVIDIA Inference Hub or another Responses-compatible "
            "provider; it cannot be combined with --codex-base-url."
        ),
    )
    parser.add_argument(
        "--codex-api-key-env",
        default=os.getenv("CONTENT_AGENT_CODEX_API_KEY_ENV"),
        help=(
            "Environment variable containing the custom Codex provider key. "
            "Its value is mapped to OPENAI_API_KEY only in the child process."
        ),
    )
    parser.add_argument(
        "--codex-sandbox-mode",
        choices=[CODEX_SANDBOX_WORKSPACE_WRITE],
        default=_restricted_default_codex_sandbox_mode(),
        help=(
            "Codex child sandbox mode. Defaults to workspace-write, confined "
            "to the run directory."
        ),
    )
    parser.add_argument(
        "--codex-config-json",
        action="append",
        default=[],
        help=(
            "JSON object of Codex SDK config overrides. May be repeated; later "
            "values recursively override earlier values. Do not put secrets here; "
            "prefer env_key/auth helpers in Codex provider config. JSON null "
            "values are rejected because Codex config CLI serialization cannot "
            "represent them."
        ),
    )
    parser.add_argument(
        "--codex-config-file",
        action="append",
        type=Path,
        default=[],
        help=(
            "Path to a JSON object containing Codex SDK config overrides. May be "
            "repeated; inline --codex-config-json overrides files. JSON null "
            "values are rejected."
        ),
    )
    parser.add_argument(
        "--vision-backend",
        default=os.getenv("CONTENT_AGENT_VISION_BACKEND"),
        help=(
            "Optional World Understanding VLM backend. When set with "
            "--vision-model, images are analyzed by this VLM and structured "
            "observations are sent to the Codex driver."
        ),
    )
    parser.add_argument(
        "--vision-model",
        default=os.getenv("CONTENT_AGENT_VISION_MODEL"),
        help="VLM model used for delegated material visual analysis.",
    )
    parser.add_argument(
        "--vision-api-key-env",
        default=os.getenv("CONTENT_AGENT_VISION_API_KEY_ENV"),
        help=(
            "Optional environment variable containing the delegated VLM key. "
            "The backend's registered credential lookup is used when omitted."
        ),
    )
    parser.add_argument(
        "--vision-base-url",
        default=os.getenv("CONTENT_AGENT_VISION_BASE_URL"),
        help="Optional delegated VLM endpoint override.",
    )
    parser.add_argument(
        "--vision-max-tokens",
        type=int,
        default=_env_int("CONTENT_AGENT_VISION_MAX_TOKENS", 4096),
        help="Maximum output tokens for each delegated VLM observation.",
    )
    parser.add_argument(
        "--claude-permission-mode",
        choices=[
            "default",
            "acceptEdits",
            "bypassPermissions",
            "plan",
        ],
        default="default",
        help=(
            "Claude Agent SDK permission mode. The default uses the SDK's "
            "`default` mode with the bridge's explicit allowed tool set."
        ),
    )
    parser.add_argument(
        "--claude-max-turns",
        type=int,
        default=None,
        help="Optional Claude Agent SDK maxTurns limit. Not supported with "
        "--claude-execution-mode=cli.",
    )
    parser.add_argument(
        "--claude-execution-mode",
        choices=[CLAUDE_EXECUTION_SDK, CLAUDE_EXECUTION_CLI],
        default=CLAUDE_EXECUTION_SDK,
        help=(
            "How --runner claude launches the child agent. 'sdk' (default) uses "
            "the Claude Agent SDK over Node, which requires ANTHROPIC_API_KEY or "
            "another provider-specific SDK auth environment. 'cli' spawns the "
            "local `claude` CLI directly (no Node/SDK dependency) and reuses "
            "whatever auth that CLI already has, including an OAuth session from "
            "`claude login`."
        ),
    )
    parser.add_argument(
        "--claude-config-json",
        action="append",
        default=[],
        help=(
            "JSON object of Claude Agent SDK option overrides. May be repeated; "
            "later values recursively override earlier values. Supported top-level "
            "keys: env, maxBudgetUsd, settings. Do not put secrets here."
        ),
    )
    parser.add_argument(
        "--claude-config-file",
        action="append",
        type=Path,
        default=[],
        help=(
            "Path to a JSON object containing Claude Agent SDK option overrides. "
            "May be repeated; inline --claude-config-json overrides files. "
            "Supported top-level keys: env, maxBudgetUsd, settings."
        ),
    )
    parser.add_argument(
        "--optimize",
        dest="optimize",
        action="store_true",
        default=None,
        help=(
            "Run the workflow-owned Scene Optimizer before inspection in fixed "
            "selection mode. For the usd-cli backend, inspection uses the "
            "optimized derivative while accepted bindings are translated back to "
            "a derivative of the immutable source."
        ),
    )
    parser.add_argument(
        "--no-optimize",
        dest="optimize",
        action="store_false",
        help=(
            "Load the source USD directly without Scene Optimizer in fixed "
            "selection mode."
        ),
    )
    parser.add_argument(
        "--optimizer-selection",
        choices=["fixed", "agent"],
        default="fixed",
        help=(
            "Choose fixed CLI optimizer settings or let a child-agent inspection "
            "turn select per-asset settings."
        ),
    )
    parser.add_argument(
        "--root-prim-path",
        "--root-prim",
        dest="root_prim_path",
        default=None,
        help="Limit material candidate discovery to a USD prim subtree.",
    )
    parser.add_argument(
        "--material-candidate-space",
        choices=["source", "inspection"],
        default="source",
        help=(
            "Path space for prediction/coverage candidates. The default `source` "
            "matches material-agent by predicting authorable source/prototype "
            "targets and retaining runtime paths as evidence."
        ),
    )
    parser.add_argument(
        "--skip-instances",
        dest="skip_instances",
        action="store_true",
        default=True,
        help=(
            "Collapse instance-proxy/runtime candidates to authorable "
            "source/prototype targets. This is the default."
        ),
    )
    parser.add_argument(
        "--include-instances",
        dest="skip_instances",
        action="store_false",
        help="Predict per-instance candidates instead of collapsing them.",
    )
    parser.add_argument(
        "--skip-prototypes",
        dest="skip_prototypes",
        action="store_true",
        default=False,
        help="Skip candidates whose authoring target is a local prototype source.",
    )
    parser.add_argument(
        "--include-prototypes",
        dest="skip_prototypes",
        action="store_false",
        help="Keep prototype/source candidates. This is the default.",
    )
    parser.add_argument(
        "--skip-invisible",
        dest="skip_invisible",
        action="store_true",
        default=False,
        help=(
            "Skip invisible candidates. Scene visibility hints already apply "
            "effective visibility."
        ),
    )
    parser.add_argument(
        "--include-invisible",
        dest="skip_invisible",
        action="store_false",
        help="Do not add extra invisible filtering beyond scene visibility hints.",
    )
    parser.add_argument(
        "--flatten-prototypes",
        dest="flatten_prototypes",
        action="store_true",
        default=None,
        help="Pass flatten_prototypes=true to the workflow-owned Scene Optimizer.",
    )
    parser.add_argument(
        "--no-flatten-prototypes",
        dest="flatten_prototypes",
        action="store_false",
        help="Pass flatten_prototypes=false to the workflow-owned Scene Optimizer.",
    )
    parser.add_argument(
        "--enable-deinstance",
        dest="enable_deinstance",
        action="store_true",
        default=None,
        help="Pass enable_deinstance=true to the workflow-owned Scene Optimizer.",
    )
    parser.add_argument(
        "--disable-deinstance",
        dest="enable_deinstance",
        action="store_false",
        help="Pass enable_deinstance=false to the workflow-owned Scene Optimizer.",
    )
    parser.add_argument(
        "--enable-split",
        dest="enable_split",
        action="store_true",
        default=None,
        help="Pass enable_split=true to the workflow-owned Scene Optimizer.",
    )
    parser.add_argument(
        "--disable-split",
        dest="enable_split",
        action="store_false",
        help="Pass enable_split=false to the workflow-owned Scene Optimizer.",
    )
    parser.add_argument(
        "--enable-deduplicate",
        dest="enable_deduplicate",
        action="store_true",
        default=None,
        help=(
            "Pass enable_deduplicate=true to the workflow-owned Scene Optimizer. "
            "Material assignment keeps deduplication disabled by default so "
            "visually identical parts can still receive distinct materials."
        ),
    )
    parser.add_argument(
        "--disable-deduplicate",
        dest="enable_deduplicate",
        action="store_false",
        help="Pass enable_deduplicate=false to the workflow-owned Scene Optimizer.",
    )
    parser.add_argument(
        "--preflight",
        dest="preflight",
        action="store_true",
        default=True,
        help=(
            "Prepare a usd-cli material run packet before launching the child "
            "agent. This is the default for real runs."
        ),
    )
    parser.add_argument(
        "--no-preflight",
        dest="preflight",
        action="store_false",
        help="Let the child agent perform usd-cli setup and initial inspection.",
    )
    parser.add_argument(
        "--respect-existing-material-bindings",
        "--respect-existing-materials",
        dest="respect_existing_material_bindings",
        action="store_true",
        default=False,
        help=(
            "Use existing material bindings as preserved seed decisions. By "
            "default the usd-cli session clears existing material bindings "
            "and authored display colors before inspection/rendering."
        ),
    )
    parser.add_argument(
        "--ignore-existing-material-bindings",
        "--ignore-existing-materials",
        dest="respect_existing_material_bindings",
        action="store_false",
        help=(
            "Clear existing authored appearance before inspection/rendering, "
            "including material bindings and display colors. This is the default."
        ),
    )
    parser.add_argument(
        "--scene-tool-timeout",
        type=float,
        default=60.0,
        help="Seconds to wait for each low-level usd-cli operation.",
    )
    parser.add_argument(
        "--material-restore-timeout",
        type=float,
        default=DEFAULT_MATERIAL_RESTORE_TIMEOUT_SECONDS,
        help=(
            "Seconds to wait for final material application and USD restoration. "
            "Defaults to the material operation limit."
        ),
    )
    parser.add_argument(
        "--child-timeout",
        type=float,
        default=_env_float("CONTENT_AGENT_CHILD_TIMEOUT", 1800.0),
        help=(
            "Seconds to wait for the Codex SDK child turn. Use 0 to disable "
            "the timeout."
        ),
    )
    parser.add_argument(
        "--prompt-mode",
        choices=[PROMPT_MODE_SKILL_ROUTED],
        default=PROMPT_MODE_SKILL_ROUTED,
        help=(
            "Child prompt shape. `skill-routed` passes compact structured inputs "
            "and asks the child agent to use workflow skills."
        ),
    )
    parser.add_argument(
        "--vqa-refinement-max-iterations",
        type=_positive_int,
        default=DEFAULT_VQA_REFINEMENT_MAX_ITERATIONS,
        help=(
            "Maximum total VQA review/refinement iterations, including the "
            "initial child final review pass. Defaults to 3."
        ),
    )
    parser.add_argument(
        "--no-vqa-refinement",
        dest="vqa_refinement_max_iterations",
        action="store_const",
        const=1,
        help=(
            "Disable wrapper-launched follow-up VQA refinement turns after the "
            "initial child final review pass."
        ),
    )
    parser.add_argument(
        "--codex-persistent-refinement",
        dest="codex_persistent_refinement",
        action="store_true",
        default=False,
        help=(
            "Deprecated compatibility flag. Confined execution always uses "
            "fresh compact Codex turns seeded by artifact pointers so each "
            "child process group can be terminated between turns."
        ),
    )
    parser.add_argument(
        "--no-codex-persistent-refinement",
        dest="codex_persistent_refinement",
        action="store_false",
        help="Use fresh compact Codex turns for VQA refinement. This is the default.",
    )
    parser.add_argument(
        "--memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable run-scoped observation memory (default). Use --no-memory "
            "for a controlled baseline."
        ),
    )
    parser.add_argument(
        "--memory-root",
        type=Path,
        default=None,
        help=(
            "Wrapper-owned memory root. It must be outside the child-writable "
            "run directory. Defaults to <repo-root>/agentic/.memory."
        ),
    )
    parser.add_argument(
        "--additional-instructions",
        default=None,
        help="Extra instruction text appended to the child-agent prompt.",
    )
    parser.add_argument(
        "--additional-instructions-file",
        type=Path,
        default=None,
        help="File containing extra instruction text appended to the child-agent prompt.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write request, prompt, and trace skeleton without launching the child agent.",
    )
    add_live_view_args(parser, workflow="materials.assign")


def _add_articulation_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--usd", required=True, type=Path, help="Input USD/USDZ asset.")
    parser.add_argument(
        "--joint-config",
        type=Path,
        default=None,
        help=(
            "Joint Agent YAML config used for provider-backed inference. Required "
            "unless --preparation supplies deterministic evidence."
        ),
    )
    parser.add_argument(
        "--preparation",
        "--embedded-preparation",
        dest="embedded_preparation",
        type=Path,
        default=None,
        help=(
            "Provider-neutral Articulation preparation with source, owner, "
            "capability, render, and usd-cli evidence. Runs standalone unless "
            "--embedded-run-state selects outer Asset custody; skips Joint Agent "
            "inference in both modes."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for durable articulation workflow artifacts.",
    )
    parser.add_argument(
        "--embedded-run-state",
        type=Path,
        default=None,
        help=(
            "Active composed-asset run state that owns this Articulation stage "
            "attempt. Omit for standalone execution."
        ),
    )
    parser.add_argument(
        "--intent",
        required=True,
        help="Exact user articulation request bound into the durable workflow.",
    )
    parser.add_argument(
        "--review-policy",
        choices=("uncertain", "all", "none"),
        default="uncertain",
        help=(
            "Which native-ready candidates require an explicit review receipt. "
            "Skill-routed execution always sends every candidate through its "
            "agent decision gate; 'none' publishes that decision as the receipt, "
            "while 'uncertain' and 'all' retain the fail-closed all-candidate "
            "operator review gate."
        ),
    )
    parser.add_argument(
        "--allowed-motion-type",
        action="append",
        choices=("revolute", "prismatic"),
        default=None,
        help=(
            "Allowed articulation-v1 motion type. May be repeated; defaults to "
            "revolute and prismatic."
        ),
    )
    parser.add_argument(
        "--expected-candidate-count",
        type=_non_negative_int,
        default=None,
        help="Optional exact candidate-count expectation, including zero.",
    )
    parser.add_argument(
        "--max-candidate-count",
        type=_positive_int,
        default=64,
        help="Fail closed if inference returns more candidates. Defaults to 64.",
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="Optional stable Joint Agent backend session identifier.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose Joint Agent backend logging.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root containing the workflow and usd-cli packages.",
    )
    _add_articulation_scene_runtime_args(parser)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the complete articulation finalization result as JSON.",
    )
    parser.add_argument(
        "--execution-mode",
        choices=(
            ARTICULATION_EXECUTION_SKILL_ROUTED,
            ARTICULATION_EXECUTION_FIXED,
        ),
        default=ARTICULATION_EXECUTION_SKILL_ROUTED,
        help=(
            "Use the skill-routed child controller (default) or the explicit "
            "fixed compatibility baseline."
        ),
    )
    parser.add_argument(
        "--runner",
        choices=(RUNNER_CODEX, RUNNER_CLAUDE),
        default=RUNNER_CODEX,
        help="Long-running child-agent runner for standalone execution.",
    )
    _add_child_model_args(parser)
    parser.add_argument(
        "--codex-base-url",
        default=os.getenv("CONTENT_AGENT_CODEX_BASE_URL"),
    )
    parser.add_argument(
        "--codex-sandbox-mode",
        choices=(CODEX_SANDBOX_WORKSPACE_WRITE,),
        default=CODEX_SANDBOX_WORKSPACE_WRITE,
    )
    parser.add_argument("--codex-config-json", action="append", default=[])
    parser.add_argument("--codex-config-file", action="append", type=Path, default=[])
    parser.add_argument(
        "--claude-permission-mode",
        choices=("default", "acceptEdits", "bypassPermissions", "plan"),
        default="default",
    )
    parser.add_argument("--claude-max-turns", type=int, default=None)
    parser.add_argument(
        "--claude-execution-mode",
        choices=(CLAUDE_EXECUTION_SDK, CLAUDE_EXECUTION_CLI),
        default=CLAUDE_EXECUTION_SDK,
    )
    parser.add_argument("--claude-config-json", action="append", default=[])
    parser.add_argument("--claude-config-file", action="append", type=Path, default=[])
    parser.add_argument(
        "--child-timeout",
        type=_non_negative_float,
        default=1800.0,
        help="Seconds to wait for the child agent. Use 0 to disable the timeout.",
    )
    parser.add_argument("--agent-cwd", type=Path, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Freeze the request and compact child task without launching.",
    )


def _add_articulation_review_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help="Existing articulation run directory in needs_review.",
    )
    parser.add_argument(
        "--decisions-json",
        required=True,
        type=Path,
        help="Candidate-to-decision JSON object using accept or reject values.",
    )
    parser.add_argument(
        "--reviewer",
        required=True,
        help="Reviewer identity bound into the exact review receipt.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root containing the workflow and usd-cli packages.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose Joint Agent backend logging while resuming.",
    )
    _add_articulation_scene_runtime_args(parser)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the complete articulation finalization result as JSON.",
    )


def _add_articulation_resume_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help="Existing articulation run directory.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root containing the workflow and usd-cli packages.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose Joint Agent backend logging while resuming.",
    )
    _add_articulation_scene_runtime_args(parser)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the complete articulation finalization result as JSON.",
    )


def _add_articulation_revision_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help="Existing embedded Articulation run with a human revise decision.",
    )
    parser.add_argument(
        "--revision-patch",
        required=True,
        type=Path,
        help=(
            "Run-confined typed graph-revision patch binding the parent review "
            "and exact revised graph."
        ),
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root containing the frozen workflow implementation.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose configuration loading without invoking inference.",
    )
    _add_articulation_scene_runtime_args(parser)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the complete articulation finalization result as JSON.",
    )


def _add_articulation_scene_runtime_args(
    parser: argparse.ArgumentParser,
) -> None:
    parser.add_argument(
        "--scene-tool-timeout",
        type=_positive_float,
        default=60.0,
        help="Seconds to wait for each low-level usd-cli operation.",
    )


def _add_physics_refine_external_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--runtime-config",
        type=Path,
        required=True,
        help="Trusted local external-runtime (BYOR) config YAML.",
    )
    parser.add_argument(
        "--user-prompt",
        required=True,
        help="Natural-language behavior goal the agent judges evidence against.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "Run directory for qualification, sweeps, and the final bundle. "
            "Re-running an approval invocation archives a previous run's "
            "final/ deliverable to <output-dir>.final.<n>/ beside the run "
            "directory and clears prior decision and iteration artifacts."
        ),
    )
    parser.add_argument(
        "--approve-qualification",
        default=None,
        metavar="DIGEST",
        help=(
            "Exact digest printed after reviewing qualification evidence. "
            "Without it, the run qualifies the runtime and stops."
        ),
    )
    parser.add_argument(
        "--reference-image",
        dest="reference_images",
        type=Path,
        action="append",
        default=[],
        help="Optional reference image for the agent's visual comparison.",
    )
    parser.add_argument(
        "--max-iterations",
        type=_positive_int,
        default=DEFAULT_EXTERNAL_REFINE_MAX_ITERATIONS,
        help="Sweep budget for the agent-owned refinement loop.",
    )
    parser.add_argument(
        "--max-trials",
        type=_positive_int,
        default=6,
        help=(
            "Per-sweep trial budget. External trials start a customer "
            "simulator process each, so keep this small."
        ),
    )
    parser.add_argument(
        "--sweep-deadline-seconds",
        type=_positive_float,
        default=DEFAULT_EXTERNAL_SWEEP_DEADLINE_SECONDS,
        help="Per-sweep wall-clock ceiling enforced by the broker.",
    )
    parser.add_argument(
        "--phase-deadline-seconds",
        type=_positive_float,
        default=None,
        help="Optional whole-phase wall-clock ceiling.",
    )
    parser.add_argument(
        "--additional-instructions",
        default=None,
        help="Optional operator instructions appended to the session prompt.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root (defaults to auto-detection).",
    )
    parser.add_argument(
        "--runner",
        choices=[RUNNER_CODEX, RUNNER_CLAUDE],
        default=RUNNER_CODEX,
        help="Child agent runner.",
    )
    _add_child_model_args(parser)
    parser.add_argument(
        "--codex-base-url",
        default=os.getenv("CONTENT_AGENT_CODEX_BASE_URL"),
        help="Optional Codex-compatible base URL.",
    )
    parser.add_argument(
        "--child-timeout",
        type=_non_negative_float,
        default=None,
        help=(
            "Seconds to wait for the child session. Must comfortably exceed "
            "max_iterations x sweep deadline; defaults to that product plus "
            "an hour of review slack. Use 0 to disable."
        ),
    )


def _handle_physics_refine_external(args: argparse.Namespace) -> int:
    from content_workflow_cli.external_refine_runner import (
        PhysicsExternalRefineConfig,
        run_physics_external_refine,
    )

    repo_root = args.repo_root.resolve() if args.repo_root else find_repo_root()
    runtime_config = args.runtime_config.expanduser().resolve()
    if not runtime_config.is_file():
        print(f"runtime config does not exist: {runtime_config}", file=sys.stderr)
        return 2
    # Strip here so a quoted-whitespace value fails the runner's non-empty
    # behavior-goal check instead of reaching the approval agent as a
    # criterion-free prompt.
    config = PhysicsExternalRefineConfig(
        repo_root=repo_root,
        runtime_config=runtime_config,
        user_prompt=args.user_prompt.strip(),
        output_dir=args.output_dir,
        approval_digest=args.approve_qualification,
        reference_images=list(args.reference_images),
        max_iterations=args.max_iterations,
        max_trials=args.max_trials,
        sweep_deadline_seconds=args.sweep_deadline_seconds,
        phase_deadline_seconds=args.phase_deadline_seconds,
        additional_instructions=args.additional_instructions,
        runner=args.runner,
        model=args.model,
        model_reasoning_effort=args.model_reasoning_effort,
        codex_base_url=args.codex_base_url,
        # The documented constraint on --child-timeout is that it exceeds the
        # total sweep budget; when the flag is omitted, derive it from the
        # budget instead of shipping a fixed value the default budget breaks.
        child_timeout_seconds=(
            args.child_timeout
            if args.child_timeout is not None
            else args.max_iterations * args.sweep_deadline_seconds + 3600.0
        ),
    )
    try:
        result = run_physics_external_refine(config)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"physics refine-external failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "run_dir": str(result.run_dir),
                "status": result.status,
                "validated": result.validated,
                "qualification_digest": result.qualification_digest,
                "final_dir": str(result.final_dir) if result.final_dir else None,
                "reasons": list(result.reasons),
            },
            indent=2,
        )
    )
    return result.returncode


def _add_physics_apply_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--usd", required=True, type=Path, help="Input USD/USDZ asset.")
    parser.add_argument(
        "--reference-image",
        action="append",
        type=Path,
        default=[],
        help="Optional physics behavior reference image path. May be repeated.",
    )
    parser.add_argument(
        "--reference",
        action="append",
        type=Path,
        default=[],
        help=(
            "Optional generic physics behavior reference file path. Images are "
            "attached as reference images; other readable files are passed by path."
        ),
    )
    parser.add_argument(
        "--behavior-prompt",
        default=None,
        help="Natural-language physics behavior goal for agentic tune/refine turns.",
    )
    parser.add_argument(
        "--behavior-prompt-file",
        type=Path,
        default=None,
        help="File containing the physics behavior goal.",
    )
    parser.add_argument(
        "--vomp-root",
        type=Path,
        default=None,
        help=(
            "Enable agentic VoMP mass authoring with this pinned official "
            "VoMP checkout."
        ),
    )
    parser.add_argument(
        "--vomp-target-prim",
        default=None,
        help=(
            "Rigid-body prim for VoMP mass authoring. Defaults to the sole "
            "mass_authoring_path selected by the long-running agent."
        ),
    )
    parser.add_argument(
        "--vomp-python",
        type=Path,
        default=None,
        help="VoMP interpreter; defaults to <vomp-root>/.venv/bin/python.",
    )
    parser.add_argument(
        "--vomp-config",
        type=Path,
        default=None,
        help="VoMP inference JSON; defaults to weights/inference.json below the root.",
    )
    parser.add_argument(
        "--vomp-ovrtx-venv",
        type=Path,
        default=None,
        help="Optional OVRTX renderer environment used for calibrated evidence.",
    )
    parser.add_argument(
        "--vomp-num-views",
        type=_positive_int,
        default=None,
        help="Override the number of calibrated OVRTX evidence views.",
    )
    parser.add_argument(
        "--vomp-image-size",
        type=_positive_int,
        default=None,
        help="Override the square OVRTX evidence render resolution.",
    )
    parser.add_argument(
        "--vomp-seed",
        type=_vomp_seed,
        default=None,
        help="Override the deterministic VoMP camera and inference seed.",
    )
    parser.add_argument(
        "--vomp-render-mode",
        choices=("rt1", "rt2", "pt"),
        default=None,
        help="OVRTX evidence render mode.",
    )
    parser.add_argument(
        "--vomp-num-sensor-updates",
        type=_positive_int,
        default=None,
        help="Override OVRTX progressive render iterations.",
    )
    parser.add_argument(
        "--vomp-material-target",
        choices=("auto", "preview_surface", "openpbr_materialx"),
        default=None,
        help="OVRTX material conversion target.",
    )
    parser.add_argument(
        "--vomp-attention-backend",
        choices=("xformers", "sdpa", "naive"),
        default=None,
        help="Attention implementation used by the isolated VoMP worker.",
    )
    parser.add_argument(
        "--vomp-timeout-seconds",
        type=_positive_float,
        default=None,
        help="VoMP worker deadline in seconds.",
    )
    parser.add_argument(
        "--vomp-max-complete-voxels",
        type=_positive_int,
        default=None,
        help="Fail above this complete-field size instead of subsampling.",
    )
    parser.add_argument(
        "--scenario",
        type=Path,
        default=None,
        help="Optional legacy-compatible physics tuning scenario YAML for agent context.",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Enable agentic physics tuning intent after schema application.",
    )
    parser.add_argument(
        "--refine",
        action="store_true",
        help="Enable iterative agentic physics refine intent; implies --tune.",
    )
    parser.add_argument(
        "--tune-engine",
        choices=("ovphysx", "newton"),
        default=None,
        help="Physics tuning engine to target in the agentic loop.",
    )
    parser.add_argument(
        "--optimizer",
        default="auto",
        help=(
            "Optimizer for agentic physics tuning/refine: 'auto' (BoTorch when "
            "installed, else hard error), 'botorch' (production BO), 'random' "
            "(baseline), 'cma-es' (baseline), or the name of an installed "
            "optimizer extension."
        ),
    )
    parser.add_argument(
        "--max-trials",
        type=_positive_int,
        default=30,
        help="Maximum tuning trials per agentic tune/refine iteration.",
    )
    parser.add_argument(
        "--max-iterations",
        type=_positive_int,
        default=DEFAULT_AGENTIC_TUNE_MAX_ITERATIONS,
        help=(
            "Hard sweep budget for the agent-owned tuning loop; the sweep "
            "broker refuses reservations beyond this."
        ),
    )
    parser.add_argument(
        "--sweep-deadline-seconds",
        type=float,
        default=DEFAULT_SWEEP_DEADLINE_SECONDS,
        help=(
            "Per-sweep wall-clock deadline; the broker cancels a sweep "
            "cooperatively when it expires."
        ),
    )
    parser.add_argument(
        "--tuning-phase-deadline-seconds",
        type=float,
        default=None,
        help=(
            "Optional total tuning-phase deadline; the broker refuses new "
            "sweep reservations after it expires."
        ),
    )
    parser.add_argument(
        "--revalidation-max-penetration-m",
        type=float,
        default=None,
        help=(
            "Ground-penetration limit (meters, at most 1.0) for every runtime "
            "ground-clearance gate in the run: the apply-phase validation, the "
            "visual-refinement and tuning agent prompts, and the wrapper's "
            "independent revalidation of an accepted tuned candidate. Defaults "
            "to the runtime validator's scale-relative limit, "
            "min(max(0.005, min(0.025 * bbox_diagonal, "
            "0.5 * smallest_bbox_extent)), 1.0) meters — the 0.005 m floor "
            "applies after the extent cap — when measured from exact collider "
            "geometry, or 0.005 m on the conservative bbox fallback. Raise for "
            "unit-normalized assets whose soft contacts legitimately rest "
            "deeper."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for canonical physics workflow artifacts.",
    )
    parser.add_argument(
        "--output-usd",
        type=Path,
        default=None,
        help="Authored USD/USDZ output path. Defaults to <output-dir>/physics.usdc.",
    )
    parser.add_argument(
        "--collision-approximation",
        default="convexHull",
        help="UsdPhysics.MeshCollisionAPI approximation for mesh colliders.",
    )
    parser.add_argument(
        "--simulation-engine",
        choices=("ovphysx", "fake", "none"),
        default="ovphysx",
        help="Runtime validation engine.",
    )
    parser.add_argument(
        "--no-simulation",
        action="store_true",
        help=(
            "Skip runtime simulation and retain schema-authoring artifacts; "
            "the workflow result is conditional and non-passing and does not "
            "satisfy required runtime and visual evidence."
        ),
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=3.0,
        help="Validation simulation duration in seconds.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=1.0 / 240.0,
        help="Validation simulation timestep.",
    )
    parser.add_argument(
        "--sample-fps",
        type=int,
        default=30,
        help="Validation trajectory sample rate.",
    )
    parser.add_argument(
        "--drop-height-m",
        type=float,
        default=None,
        help="Drop-settle validation gap in meters. Defaults to asset bbox height.",
    )
    parser.add_argument(
        "--fail-on-validation-error",
        action="store_true",
        help="Return a non-zero exit code when simulation validation fails.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume a deterministic physics run only when its recorded request "
            "and source identity match."
        ),
    )
    parser.add_argument(
        "--scene-tool-timeout",
        type=float,
        default=300.0,
        help="Timeout in seconds for low-level usd-cli operations.",
    )
    parser.add_argument(
        "--decision-patch",
        type=Path,
        help=(
            "Coordinator-authored physics decision patch. Supported only with "
            "--direct-executor."
        ),
    )
    parser.add_argument(
        "--topology-plan",
        type=Path,
        help=(
            "Coordinator-authored physics topology plan. Supported only with "
            "--direct-executor."
        ),
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root for the child agent. Defaults to git root.",
    )
    parser.add_argument(
        "--runner",
        choices=[RUNNER_CODEX, RUNNER_CLAUDE],
        default=RUNNER_CODEX,
        help="Child agent runner.",
    )
    _add_child_model_args(parser)
    parser.add_argument(
        "--codex-base-url",
        default=os.getenv("CONTENT_AGENT_CODEX_BASE_URL"),
        help="Optional OpenAI-compatible base URL for the Codex SDK.",
    )
    parser.add_argument(
        "--codex-sandbox-mode",
        choices=[CODEX_SANDBOX_WORKSPACE_WRITE],
        default=_restricted_default_codex_sandbox_mode(),
        help="Codex child sandbox mode.",
    )
    parser.add_argument("--codex-config-json", action="append", default=[])
    parser.add_argument("--codex-config-file", action="append", type=Path, default=[])
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
        help=(
            "How --runner claude launches the child agent: 'sdk' (default, "
            "Claude Agent SDK via Node) or 'cli' (spawn the local `claude` CLI "
            "directly, reusing its `claude login` OAuth session)."
        ),
    )
    parser.add_argument("--claude-config-json", action="append", default=[])
    parser.add_argument("--claude-config-file", action="append", type=Path, default=[])
    parser.add_argument(
        "--optimize",
        dest="optimize",
        action="store_true",
        default=True,
        help="Run the workflow-owned Scene Optimizer before inspection.",
    )
    parser.add_argument(
        "--no-optimize",
        dest="optimize",
        action="store_false",
        help="Load the source USD directly without Scene Optimizer.",
    )
    parser.add_argument(
        "--optimizer-selection",
        choices=["fixed", "agent"],
        default="fixed",
        help=(
            "Choose fixed physics inspection settings or let a child-agent "
            "topology inspection select per-asset settings."
        ),
    )
    parser.add_argument(
        "--flatten-prototypes",
        dest="flatten_prototypes",
        action="store_true",
        default=None,
        help="Pass flatten_prototypes=true in fixed optimizer selection mode.",
    )
    parser.add_argument(
        "--no-flatten-prototypes",
        dest="flatten_prototypes",
        action="store_false",
        help="Pass flatten_prototypes=false in fixed optimizer selection mode.",
    )
    parser.add_argument(
        "--enable-deinstance",
        dest="enable_deinstance",
        action="store_true",
        default=None,
        help="Enable Scene Optimizer deinstancing in fixed selection mode.",
    )
    parser.add_argument(
        "--disable-deinstance",
        dest="enable_deinstance",
        action="store_false",
        help="Disable Scene Optimizer deinstancing in fixed selection mode.",
    )
    parser.add_argument(
        "--enable-split",
        dest="enable_split",
        action="store_true",
        default=None,
        help="Enable Scene Optimizer mesh splitting in fixed selection mode.",
    )
    parser.add_argument(
        "--disable-split",
        dest="enable_split",
        action="store_false",
        help="Disable Scene Optimizer mesh splitting in fixed selection mode.",
    )
    parser.add_argument(
        "--enable-deduplicate",
        dest="enable_deduplicate",
        action="store_true",
        default=None,
        help="Enable Scene Optimizer deduplication in fixed selection mode.",
    )
    parser.add_argument(
        "--disable-deduplicate",
        dest="enable_deduplicate",
        action="store_false",
        help="Disable Scene Optimizer deduplication in fixed selection mode.",
    )
    parser.add_argument(
        "--child-timeout",
        type=float,
        default=_env_float("CONTENT_AGENT_CHILD_TIMEOUT", 1800.0),
        help="Seconds to wait for each child-agent turn. Use 0 to disable.",
    )
    parser.add_argument(
        "--log-to-stderr",
        dest="log_to_stderr",
        action="store_true",
        default=None,
        help="Route progress logs to stderr.",
    )
    parser.add_argument(
        "--log-to-stdout",
        dest="log_to_stderr",
        action="store_false",
        help="Route progress logs to stdout.",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=[PROMPT_MODE_SKILL_ROUTED],
        default=PROMPT_MODE_SKILL_ROUTED,
        help="Skill-routed child prompt shape.",
    )
    parser.add_argument(
        "--visual-validation-max-iterations",
        "--vqa-refinement-max-iterations",
        dest="vqa_refinement_max_iterations",
        type=_positive_int,
        default=DEFAULT_VQA_REFINEMENT_MAX_ITERATIONS,
        help="Maximum physics visual review/refinement iterations. Defaults to 3.",
    )
    parser.add_argument(
        "--no-visual-validation-refinement",
        "--no-vqa-refinement",
        dest="vqa_refinement_max_iterations",
        action="store_const",
        const=1,
        help="Run only one physics visual review turn.",
    )
    parser.add_argument(
        "--codex-persistent-refinement",
        dest="codex_persistent_refinement",
        action="store_true",
        default=False,
        help=(
            "Deprecated compatibility flag. Confined execution always uses "
            "fresh Codex turns so child process groups can be terminated."
        ),
    )
    parser.add_argument(
        "--no-codex-persistent-refinement",
        dest="codex_persistent_refinement",
        action="store_false",
    )
    parser.add_argument(
        "--additional-instructions",
        default=None,
        help="Extra instruction text appended to the child-agent prompt.",
    )
    parser.add_argument(
        "--additional-instructions-file",
        type=Path,
        default=None,
        help="File containing extra instruction text appended to the prompt.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write request, prompt, and trace skeleton without launching the child agent.",
    )
    parser.add_argument(
        "--direct-executor",
        "--deterministic-workflow",
        dest="deterministic_workflow",
        action=_DirectExecutorAction,
        nargs=0,
        default=False,
        help=(
            "Use the lower-level direct executor inside the agentic "
            "Physics command, without child-agent visual review. This is not "
            "the fixed pipeline apps/physics_agent pipeline; --deterministic-workflow "
            "is a legacy alias."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full result JSON.",
    )
    add_live_view_args(parser, workflow="physics.apply")


def _default_codex_sandbox_mode(
    *,
    allow_danger_full_access: bool = False,
) -> str:
    value = os.getenv("CONTENT_AGENT_CODEX_SANDBOX_MODE")
    if value in {None, ""}:
        return CODEX_SANDBOX_WORKSPACE_WRITE
    allowed = {CODEX_SANDBOX_WORKSPACE_WRITE}
    if allow_danger_full_access:
        allowed.add(CODEX_SANDBOX_DANGER_FULL_ACCESS)
    if value not in allowed:
        supported = ", ".join(sorted(allowed))
        raise ValueError(
            "Invalid CONTENT_AGENT_CODEX_SANDBOX_MODE: "
            f"{value!r}. Expected one of: {supported}"
        )
    return value


def _restricted_default_codex_sandbox_mode() -> str:
    """Keep full-access mesh configuration from leaking into other workflows."""

    if os.getenv("CONTENT_AGENT_CODEX_SANDBOX_MODE") == (
        CODEX_SANDBOX_DANGER_FULL_ACCESS
    ):
        return CODEX_SANDBOX_WORKSPACE_WRITE
    return _default_codex_sandbox_mode()


def _handle_convert_to_usd(args: argparse.Namespace) -> int:
    from content_agent_workflows.convert_to_usd import (
        ConvertToUsdWorkflowInput,
        convert_source_to_usd_file,
        resolve_output_usd_path,
        run_convert_to_usd_workflow,
    )

    source_asset = args.source_asset.expanduser().resolve()
    try:
        output_usd = resolve_output_usd_path(
            source_asset,
            args.output_usd,
            output_format=args.output_format,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.resume and args.output_dir is None:
        print("error: --resume requires --output-dir", file=sys.stderr)
        return 2

    if args.output_dir is not None:
        result = run_convert_to_usd_workflow(
            ConvertToUsdWorkflowInput(
                source_asset_path=source_asset,
                output_dir=args.output_dir.expanduser().resolve(),
                output_usd_path=output_usd,
                output_format=args.output_format,
                install_missing=args.install_missing,
                converter_timeout_s=args.converter_timeout_s,
                resume=args.resume,
            )
        )
        if args.report is not None:
            _copy_text_artifact(Path(result.conversion_report_path), args.report)
        if args.markdown_report is not None:
            _copy_text_artifact(Path(result.markdown_report_path), args.markdown_report)
        if args.json:
            print(json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True))
        else:
            status = "ok" if result.success else "failed"
            print(f"convert-to-usd {status}: {result.output_usd_path or 'no output'}")
            print(f"Run directory: {result.output_dir}")
            print(f"Report: {result.conversion_report_path}")
            print(f"Converter timeout: {result.converter_timeout_s:g}s")
            if result.error:
                print(f"error: {result.error}", file=sys.stderr)
        return 0 if result.success else 1

    report, _probe_artifact = convert_source_to_usd_file(
        source_asset,
        output_usd,
        output_format=args.output_format,
        install_missing=args.install_missing,
        timeout_s=args.converter_timeout_s,
    )
    report_json = json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True)
    if args.report is not None:
        report_path = args.report.expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report_json + "\n", encoding="utf-8")
    if args.markdown_report is not None:
        args.markdown_report.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_report.write_text(report.to_markdown(), encoding="utf-8")
    if args.json:
        print(report_json)
    else:
        status = "ok" if report.passed else report.status
        print(f"convert-to-usd {status}: {report.output_usd_path or 'no output'}")
        print(f"Converter timeout: {report.converter_timeout_s:g}s")
        if args.report is not None:
            print(f"Report: {args.report.expanduser().resolve()}")
        if report.errors:
            print("error: " + "; ".join(report.errors), file=sys.stderr)
    return 0 if report.passed else 1


def _handle_convert_to_usd_preflight(args: argparse.Namespace) -> int:
    from content_agent_workflows.convert_to_usd import (
        preflight_convert_to_usd_dependencies,
    )

    source_asset = args.source_asset.expanduser().resolve()
    report = preflight_convert_to_usd_dependencies(
        source_asset,
        install_missing=args.install_missing,
    )
    report_json = json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True)
    if args.report is not None:
        report_path = args.report.expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report_json + "\n", encoding="utf-8")
    if args.json:
        print(report_json)
    else:
        status = "ok" if report.passed else report.status
        target = report.converter_reference or "no converter"
        print(f"convert-to-usd preflight {status}: {target}")
        if report.install_command:
            print("Install command: " + " ".join(report.install_command))
        if args.report is not None:
            print(f"Report: {args.report.expanduser().resolve()}")
        if report.errors:
            print("error: " + "; ".join(report.errors), file=sys.stderr)
    return 0 if report.passed else 1


def _handle_simready_preflight(args: argparse.Namespace) -> int:
    from content_agent_workflows.simready import preflight_simready_foundation

    report = preflight_simready_foundation(
        foundation_root=args.foundation_root.expanduser().resolve()
        if args.foundation_root is not None
        else None,
        foundation_spec_root=args.foundation_spec_root.expanduser().resolve()
        if args.foundation_spec_root is not None
        else None,
        venv_path=args.venv_path.expanduser().resolve()
        if args.venv_path is not None
        else None,
        install_missing=args.install_missing,
        update_foundation=args.update_foundation,
    )
    report_json = json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True)
    if args.report is not None:
        report_path = args.report.expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report_json + "\n", encoding="utf-8")
    if args.json:
        print(report_json)
    else:
        status = "ok" if report.passed else report.status.lower()
        print(f"simready-foundation preflight {status}")
        if report.foundation_root:
            print(f"Foundation root: {report.foundation_root}")
        if report.foundation_spec_root:
            print(f"Spec root: {report.foundation_spec_root}")
        if report.validator_executable:
            print(f"Validator: {report.validator_executable}")
        if args.report is not None:
            print(f"Report: {args.report.expanduser().resolve()}")
        if report.errors:
            print("error: " + "; ".join(report.errors), file=sys.stderr)
    return 0 if report.passed else 1


def _handle_articulation_authoring_platform_preflight(
    _args: argparse.Namespace,
) -> int:
    from world_understanding.functions.physics.joint_rigger import (
        require_joint_rigger_authoring_platform,
    )

    require_joint_rigger_authoring_platform()
    print("articulation authoring platform preflight ok")
    return 0


def _handle_claude_sandbox_preflight(_args: argparse.Namespace) -> int:
    from content_workflow_cli.runner import preflight_claude_windows_sandbox

    preflight_claude_windows_sandbox(Path.cwd())
    print("Claude sandbox preflight: host supported")
    return 0


def _handle_simready_validate_profile(args: argparse.Namespace) -> int:
    from content_agent_workflows.simready import (
        SimReadyValidationInput,
        run_simready_profile_validation,
    )

    report = run_simready_profile_validation(
        SimReadyValidationInput(
            asset_path=str(args.asset_path.expanduser().resolve()),
            profile=args.profile,
            profile_version=args.profile_version,
            report_path=str(args.report.expanduser().resolve())
            if args.report is not None
            else None,
            foundation_root=str(args.foundation_root.expanduser().resolve())
            if args.foundation_root is not None
            else None,
            foundation_spec_root=str(args.foundation_spec_root.expanduser().resolve())
            if args.foundation_spec_root is not None
            else None,
            venv_path=str(args.venv_path.expanduser().resolve())
            if args.venv_path is not None
            else None,
            install_missing=args.install_missing,
            update_foundation=args.update_foundation,
            timeout_s=args.timeout,
            stdout_log_path=str(args.stdout_log.expanduser().resolve())
            if args.stdout_log is not None
            else None,
            stderr_log_path=str(args.stderr_log.expanduser().resolve())
            if args.stderr_log is not None
            else None,
            resume=args.resume,
        )
    )
    payload = report.model_dump(mode="json")
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        status = "ok" if report.passed else report.status.lower()
        print(f"simready validate-profile {status}: {report.profile_target}")
        print(f"Report: {report.report_path or args.report or report.raw_report_path}")
        if report.needs_rerun:
            print("Needs rerun: " + ", ".join(report.rerun_reasons))
        if report.errors:
            print("error: " + "; ".join(report.errors), file=sys.stderr)
    if report.passed:
        return 0
    if args.strict:
        return 1
    if (
        report.status == "BLOCKED"
        or report.errors
        or report.next_step == "fix-simready-validator-runtime"
    ):
        return 1
    return 0


def _handle_simready_validate_runtime(args: argparse.Namespace) -> int:
    from content_agent_workflows.simready import (
        SimReadyRuntimeValidationInput,
        run_simready_runtime_validation,
    )

    report = run_simready_runtime_validation(
        SimReadyRuntimeValidationInput(
            asset_path=str(args.asset_path.expanduser().resolve()),
            output_dir=str(args.output_dir.expanduser().resolve()),
            sr_specs_path=str(args.sr_specs_path.expanduser().resolve()),
            engines_toml_path=str(args.engines_toml_path.expanduser().resolve()),
            project_config_path=(
                str(args.project_config_path.expanduser().resolve())
                if args.project_config_path is not None
                else None
            ),
            tests_paths=tuple(
                str(path.expanduser().resolve()) for path in args.tests_paths
            ),
            features=tuple(args.features),
            tests=tuple(args.tests),
            runtimes=tuple(args.runtimes),
            benchmark_executable=(
                str(args.benchmark_executable.expanduser().resolve())
                if args.benchmark_executable is not None
                else None
            ),
            report_path=(
                str(args.report_path.expanduser().resolve())
                if args.report_path is not None
                else None
            ),
            max_concurrent=args.max_concurrent,
            timeout_s=args.timeout_s,
            stamp_results=args.stamp_results,
        )
    )
    payload = report.model_dump(mode="json")
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"simready validate-runtime {report.status}: {report.asset_path}")
        if report.report_path:
            print(f"Report: {report.report_path}")
        if report.verified_operation_publication_path:
            print(f"Verified operations: {report.verified_operation_publication_path}")
        if report.errors:
            print("error: " + "; ".join(report.errors), file=sys.stderr)
    if report.passed:
        return 0
    return 4 if report.status == "blocked" else 1


def _handle_simready_conform_profile(args: argparse.Namespace) -> int:
    from content_agent_workflows.simready import (
        SimReadyConformanceInput,
        run_simready_profile_conformance,
    )

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(args.asset_path.expanduser().resolve()),
            output_dir=str(args.output_dir.expanduser().resolve()),
            profile=args.profile,
            profile_version=args.profile_version,
            report_path=str(args.report.expanduser().resolve())
            if args.report is not None
            else None,
            validation_report_path=str(args.validation_report.expanduser().resolve())
            if args.validation_report is not None
            else None,
            source_asset=args.source_asset,
            expected_physics_inventory_sha256=(args.expected_physics_inventory_sha256),
            grasp_prim_path=args.grasp_prim_path,
            foundation_root=str(args.foundation_root.expanduser().resolve())
            if args.foundation_root is not None
            else None,
            foundation_spec_root=str(args.foundation_spec_root.expanduser().resolve())
            if args.foundation_spec_root is not None
            else None,
            venv_path=str(args.venv_path.expanduser().resolve())
            if args.venv_path is not None
            else None,
            repair_requirements=args.repair_requirements,
            force=args.force,
            resume=args.resume,
        )
    )
    payload = report.model_dump(mode="json")
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        status = "ok" if report.passed else report.status.lower()
        print(f"simready conform-profile {status}: {report.output_usd_path}")
        if report.report_path:
            print(f"Report: {report.report_path}")
        if report.requirements_blocked:
            print("Blocked requirements: " + ", ".join(report.requirements_blocked))
        if report.errors:
            print("error: " + "; ".join(report.errors), file=sys.stderr)
    if report.passed:
        return 0
    if args.strict or report.status == "FAIL":
        return 1
    return 0


def _copy_text_artifact(source: Path, destination: Path) -> None:
    atomic_write_text(destination, source.read_text(encoding="utf-8"))


def _codex_base_url_from_args(args: argparse.Namespace) -> str | None:
    base_url = str(args.codex_base_url) if args.codex_base_url else None
    responses_url = str(args.codex_responses_url) if args.codex_responses_url else None
    if base_url and responses_url:
        raise ValueError(
            "Use either --codex-base-url or --codex-responses-url, not both."
        )
    if not responses_url:
        return base_url
    return codex_base_url_from_responses_url(responses_url)


def _load_mesh_segmentation_provider_secrets(
    repo_root: Path,
    *,
    environment_names: list[str | None],
) -> None:
    requested = sorted({name for name in environment_names if name})
    missing = [name for name in requested if not os.environ.get(name)]
    if not missing:
        return
    env_path = repo_root / ".env"
    if not env_path.is_file():
        return
    values = dotenv_values(env_path)
    for name in missing:
        value = values.get(name)
        if value:
            os.environ[name] = value


def _handle_mesh_segmentation_run(args: argparse.Namespace) -> int:
    repo_root = args.repo_root.resolve() if args.repo_root else find_repo_root()
    _load_mesh_segmentation_provider_secrets(
        repo_root,
        environment_names=[
            args.codex_api_key_env,
            args.image_gen_api_key_env,
        ],
    )
    _reference_directories, directory_references = _expand_reference_directories(
        args.reference_dir
    )
    directory_images, _directory_files = _split_generic_references(directory_references)
    reference_images = _dedupe_paths(
        [path.expanduser().resolve() for path in args.reference_image]
        + directory_images
    )

    instructions = args.additional_instructions
    if args.additional_instructions_file is not None:
        file_text = _read_additional_instructions_file(
            args.additional_instructions_file
        )
        instructions = (
            f"{instructions.rstrip()}\n{file_text.strip()}"
            if instructions and file_text.strip()
            else instructions or file_text
        )
    normalized_instructions = (
        instructions.strip() if instructions and instructions.strip() else None
    )
    target_semantic_parts = [
        value.strip() for value in args.target_semantic_part if value.strip()
    ]

    result = run_mesh_segmentation(
        MeshSegmentationConfig(
            repo_root=repo_root,
            workflow_skill=args.workflow_skill,
            asset_path=args.asset.expanduser().resolve(),
            asset_root=(
                args.asset_root.expanduser().resolve()
                if args.asset_root is not None
                else None
            ),
            target_prim_path=args.target_prim,
            target_semantic_parts=target_semantic_parts,
            continue_from_run=(
                args.continue_from_run.expanduser().resolve()
                if args.continue_from_run is not None
                else None
            ),
            reference_images=reference_images,
            expected_asset_sha256=args.expected_asset_sha256,
            expected_reference_sha256=list(args.expected_reference_sha256),
            output_dir=(
                args.output_dir.expanduser() if args.output_dir is not None else None
            ),
            run_id=args.run_id,
            runner=args.runner,
            model=args.model,
            model_reasoning_effort=args.model_reasoning_effort,
            codex_base_url=_codex_base_url_from_args(args),
            codex_responses_url=args.codex_responses_url,
            codex_api_key_env=args.codex_api_key_env,
            allow_codex_configured_auth=args.allow_codex_configured_auth,
            codex_sandbox_mode=args.codex_sandbox_mode,
            codex_config=_load_codex_config(args),
            codex_execution_mode=args.codex_execution_mode,
            allow_unsafe_host_child=args.allow_unsafe_host_child,
            codex_container_image=args.codex_container_image,
            claude_config=_load_claude_config(args),
            claude_permission_mode=args.claude_permission_mode,
            claude_max_turns=args.claude_max_turns,
            claude_execution_mode=args.claude_execution_mode,
            scene_tool_timeout_seconds=args.scene_tool_timeout,
            child_timeout_seconds=args.child_timeout,
            iteration_budget=args.iteration_budget,
            case_timeout_seconds=args.case_timeout,
            memory_enabled=args.memory,
            memory_root=(
                args.memory_root.expanduser().resolve()
                if args.memory_root is not None
                else None
            ),
            image_gen_backend=args.image_gen_backend,
            image_gen_model=args.image_gen_model,
            image_gen_base_url=args.image_gen_base_url,
            image_gen_api_key_env=args.image_gen_api_key_env,
            additional_instructions=normalized_instructions,
            dry_run=args.dry_run,
        )
    )
    print(f"Run directory: {result.run_dir}")
    print(f"Request: {result.request_path}")
    print(f"Prompt: {result.prompt_path}")
    print("Fresh child session: true")
    print(f"Completed: {str(result.completed).lower()}")
    if result.terminal_validation_path is not None:
        print(f"Terminal validation: {result.terminal_validation_path}")
    return result.returncode


def _handle_scene_run(args: argparse.Namespace) -> int:
    repo_root = args.repo_root.resolve() if args.repo_root else find_repo_root()
    reference_directories, directory_references = _expand_reference_directories(
        args.reference_dir
    )
    explicit_reference_images = [
        path.expanduser().resolve() for path in args.reference_image
    ]
    generic_images, generic_files = _split_generic_references(args.reference)
    directory_images, directory_files = _split_generic_references(directory_references)
    reference_images = _dedupe_paths(
        explicit_reference_images + generic_images + directory_images
    )
    reference_files = _dedupe_paths(generic_files + directory_files)

    instructions = args.additional_instructions
    instruction_sources: list[Path] = []
    if args.additional_instructions_file is not None:
        instruction_path = args.additional_instructions_file.expanduser().resolve()
        file_text = _read_additional_instructions_file(instruction_path)
        instruction_sources.append(instruction_path)
        instructions = (
            f"{instructions.rstrip()}\n{file_text.strip()}"
            if instructions and file_text.strip()
            else instructions or file_text
        )
    normalized_instructions = (
        instructions.strip() if instructions and instructions.strip() else None
    )

    tasks = list(args.task)
    materials_yaml = (
        args.materials_yaml.expanduser().resolve()
        if args.materials_yaml is not None
        else None
    )
    materials_usd = None
    if args.materials_usd is not None:
        materials_usd = args.materials_usd.expanduser().resolve()
    elif materials_yaml is not None:
        materials_usd = _resolve_materials_usd_from_manifest(materials_yaml)

    config = SceneRunConfig(
        repo_root=repo_root,
        usd_path=args.usd.expanduser().resolve(),
        requested_tasks=tasks,
        reference_images=reference_images,
        reference_files=reference_files,
        reference_directories=reference_directories,
        materials_yaml=materials_yaml,
        materials_usd=materials_usd,
        material_candidate_space=args.material_candidate_space,
        respect_existing_material_bindings=args.respect_existing_material_bindings,
        additional_instructions=normalized_instructions,
        additional_instruction_sources=instruction_sources,
        output_dir=(
            args.output_dir.expanduser() if args.output_dir is not None else None
        ),
        run_id=args.run_id,
        runner=args.runner,
        model=args.model,
        model_reasoning_effort=args.model_reasoning_effort,
        codex_base_url=args.codex_base_url,
        codex_sandbox_mode=args.codex_sandbox_mode,
        codex_config=_load_codex_config(args),
        claude_config=_load_claude_config(args),
        claude_permission_mode=args.claude_permission_mode,
        claude_max_turns=args.claude_max_turns,
        claude_execution_mode=args.claude_execution_mode,
        scene_tool_timeout_seconds=args.scene_tool_timeout,
        child_timeout_seconds=args.child_timeout,
        dry_run=args.dry_run,
    )
    result = run_scene_workflow(config)
    _print_scene_result(result)
    return result.returncode


def _handle_scene_resume(args: argparse.Namespace) -> int:
    result = resume_scene_workflow(
        args.run_dir.expanduser(),
        dry_run=args.dry_run,
        adopt_legacy_request_sha256=args.adopt_legacy_request_sha256,
    )
    _print_scene_result(result)
    return result.returncode


def _handle_scene_phase(args: argparse.Namespace) -> int:
    from content_agent_workflows.large_scene.cli import main as phase_main

    return phase_main([args.scene_operation, *args.scene_operation_args])


def _handle_scene_decompose(args: argparse.Namespace) -> int:
    from content_agent_workflows.scene_decomposition.cli import main as decompose_main

    return decompose_main([args.scene_operation, *args.scene_operation_args])


def _handle_scene_process(args: argparse.Namespace) -> int:
    from content_agent_workflows.asset_task_processing.__main__ import (
        main as process_main,
    )

    return process_main([args.scene_operation, *args.scene_operation_args])


def _handle_scene_material_task(args: argparse.Namespace) -> int:
    from content_agent_workflows.asset_task_processing.material_task import (
        main as material_task_main,
    )

    return material_task_main([args.scene_operation, *args.scene_operation_args])


def _handle_scene_collect(args: argparse.Namespace) -> int:
    from content_agent_workflows.scene_collection.__main__ import main as collect_main

    return collect_main(["--request", str(args.request)])


def _print_scene_result(result: Any) -> None:
    print(f"Run directory: {result.run_dir}")
    print(f"Request: {result.request_path}")
    print(f"Run state: {result.run_state_path}")
    print(f"Prompt: {result.prompt_path}")
    print(f"Completed: {str(result.completed).lower()}")
    if result.terminal_validation_path is not None:
        print(f"Terminal validation: {result.terminal_validation_path}")
    for label, path in result.trace_paths.items():
        print(f"{label}: {path}")


def _handle_materials_assign(args: argparse.Namespace) -> int:
    repo_root = args.repo_root.resolve() if args.repo_root else find_repo_root()
    if not usd_cli_source_distributed(repo_root):
        print(USD_CLI_UNAVAILABLE_HINT, file=sys.stderr)
        return 2
    optimize = False if args.optimize is None else args.optimize
    prompt_mode = PROMPT_MODE_SKILL_ROUTED

    materials_yaml = args.materials_yaml.expanduser().resolve()
    materials_usd = (
        args.materials_usd.expanduser().resolve()
        if args.materials_usd is not None
        else _resolve_materials_usd_from_manifest(materials_yaml)
    )
    additional_instructions = args.additional_instructions
    if args.additional_instructions_file:
        file_text = _read_additional_instructions_file(
            args.additional_instructions_file
        )
        additional_instructions = (
            f"{additional_instructions}\n{file_text}"
            if additional_instructions
            else file_text
        )
    generic_reference_images, reference_files = _split_generic_references(
        args.reference
    )
    reference_images = _dedupe_paths(
        [path.expanduser().resolve() for path in args.reference_image]
        + generic_reference_images
    )

    config = MaterialAssignConfig(
        repo_root=repo_root,
        usd_path=args.usd.expanduser().resolve(),
        reference_images=reference_images,
        materials_yaml=materials_yaml,
        materials_usd=materials_usd,
        reference_files=reference_files,
        output_dir=args.output_dir.expanduser()
        if args.output_dir is not None
        else None,
        output_usd_path=args.output_usd.expanduser().resolve()
        if args.output_usd is not None
        else None,
        default_output_root=Path.cwd().resolve(),
        runner=args.runner,
        model=args.model,
        model_reasoning_effort=args.model_reasoning_effort,
        codex_base_url=_codex_base_url_from_args(args),
        codex_responses_url=args.codex_responses_url,
        codex_api_key_env=args.codex_api_key_env,
        codex_sandbox_mode=args.codex_sandbox_mode,
        codex_config=_load_codex_config(args),
        vision_backend=args.vision_backend,
        vision_model=args.vision_model,
        vision_api_key_env=args.vision_api_key_env,
        vision_base_url=args.vision_base_url,
        vision_max_tokens=args.vision_max_tokens,
        claude_config=_load_claude_config(args),
        claude_permission_mode=args.claude_permission_mode,
        claude_max_turns=args.claude_max_turns,
        claude_execution_mode=args.claude_execution_mode,
        dry_run=args.dry_run,
        optimize=optimize,
        optimizer_selection=args.optimizer_selection,
        root_prim_path=args.root_prim_path,
        material_candidate_space=args.material_candidate_space,
        skip_instances=args.skip_instances,
        skip_prototypes=args.skip_prototypes,
        skip_invisible=args.skip_invisible,
        flatten_prototypes=args.flatten_prototypes,
        enable_deinstance=args.enable_deinstance,
        enable_split=args.enable_split,
        enable_deduplicate=args.enable_deduplicate,
        preflight=args.preflight,
        respect_existing_material_bindings=args.respect_existing_material_bindings,
        scene_tool_timeout_seconds=args.scene_tool_timeout,
        material_restore_timeout_seconds=args.material_restore_timeout,
        child_timeout_seconds=args.child_timeout,
        prompt_mode=prompt_mode,
        vqa_refinement_max_iterations=args.vqa_refinement_max_iterations,
        codex_persistent_refinement=args.codex_persistent_refinement,
        memory_enabled=args.memory,
        memory_root=(
            args.memory_root.expanduser().resolve()
            if args.memory_root is not None
            else None
        ),
        additional_instructions=additional_instructions,
    )
    result = run_material_assignment(config)
    print(f"Run directory: {result.run_dir}")
    print(f"Prompt: {result.prompt_path}")
    print(f"Request: {result.request_path}")
    print(f"Child output: {result.child_output_path}")
    print(f"Child final: {result.child_final_path}")
    for label, path in result.trace_paths.items():
        print(f"{label}: {path}")
    return result.returncode


def _handle_articulation_run(args: argparse.Namespace) -> int:
    if (
        args.embedded_run_state is not None
        and args.execution_mode == ARTICULATION_EXECUTION_FIXED
    ):
        raise ValueError(
            "Embedded Articulation execution cannot use the fixed compatibility mode"
        )
    repo_root = args.repo_root.resolve() if args.repo_root else find_repo_root()
    allowed_motion_types = tuple(
        dict.fromkeys(args.allowed_motion_type or ("revolute", "prismatic"))
    )
    result = run_articulation_workflow(
        ArticulationRunConfig(
            repo_root=repo_root,
            source_asset=args.usd,
            output_dir=args.output_dir,
            joint_config=args.joint_config,
            intent=args.intent,
            review_policy=args.review_policy,
            allowed_motion_types=allowed_motion_types,
            expected_candidate_count=args.expected_candidate_count,
            max_candidate_count=args.max_candidate_count,
            joint_session_id=args.session_id,
            embedded_run_state=args.embedded_run_state,
            embedded_preparation=args.embedded_preparation,
            verbose=args.verbose,
            scene_timeout_seconds=args.scene_tool_timeout,
            execution_mode=args.execution_mode,
            runner=args.runner,
            model=args.model,
            model_reasoning_effort=args.model_reasoning_effort,
            codex_base_url=args.codex_base_url,
            codex_sandbox_mode=args.codex_sandbox_mode,
            codex_config=_load_codex_config(args),
            claude_config=_load_claude_config(args),
            claude_permission_mode=args.claude_permission_mode,
            claude_max_turns=args.claude_max_turns,
            claude_execution_mode=args.claude_execution_mode,
            child_timeout_seconds=args.child_timeout,
            agent_cwd=args.agent_cwd,
            dry_run=args.dry_run,
        )
    )
    _print_articulation_result(result, as_json=args.json)
    return 0 if args.dry_run else _articulation_returncode(result)


def _handle_articulation_review(args: argparse.Namespace) -> int:
    repo_root = args.repo_root.resolve() if args.repo_root else find_repo_root()
    result = review_articulation_workflow(
        args.run_dir,
        args.decisions_json,
        reviewer=args.reviewer,
        repo_root=repo_root,
        scene_timeout_seconds=args.scene_tool_timeout,
        verbose=args.verbose,
    )
    _print_articulation_result(result, as_json=args.json)
    return _articulation_returncode(result)


def _handle_articulation_resume(args: argparse.Namespace) -> int:
    repo_root = args.repo_root.resolve() if args.repo_root else find_repo_root()
    result = resume_articulation_workflow(
        args.run_dir,
        repo_root=repo_root,
        scene_timeout_seconds=args.scene_tool_timeout,
        verbose=args.verbose,
    )
    _print_articulation_result(result, as_json=args.json)
    return _articulation_returncode(result)


def _handle_articulation_revision(args: argparse.Namespace) -> int:
    repo_root = args.repo_root.resolve() if args.repo_root else find_repo_root()
    result = revise_articulation_graph_workflow(
        args.run_dir,
        args.revision_patch,
        repo_root=repo_root,
        scene_timeout_seconds=args.scene_tool_timeout,
        verbose=args.verbose,
    )
    _print_articulation_result(result, as_json=args.json)
    return _articulation_returncode(result)


def _handle_articulation_agent_prepare(args: argparse.Namespace) -> int:
    result = prepare_articulation_agent_step(args.run_dir)
    print(result.model_dump_json(indent=2))
    return _articulation_returncode(result)


def _handle_articulation_agent_apply(args: argparse.Namespace) -> int:
    result = apply_articulation_agent_step(args.run_dir, args.decision_patch)
    print(result.model_dump_json(indent=2))
    return _articulation_returncode(result)


def _handle_articulation_agent_finalize(args: argparse.Namespace) -> int:
    result = finalize_articulation_agent_step(
        args.run_dir,
        args.post_review_patch,
    )
    print(result.model_dump_json(indent=2))
    return _articulation_returncode(result)


def _print_articulation_result(
    result: ArticulationFinalizationResult,
    *,
    as_json: bool,
) -> None:
    payload = result.model_dump(mode="json")
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    print(f"articulation {result.status}: {result.output_dir}")
    if result.status in {"not_articulated", "conditional", "cancelled", "failed"}:
        print(f"message: {result.message}")
    for field_name, label in (
        ("candidate_document_path", "candidates"),
        ("scene_evidence_path", "scene_evidence"),
        ("review_receipt_path", "review_receipt"),
        ("approved_candidate_document_path", "approved_candidates"),
        ("output_asset_path", "output_usdz"),
        ("diagnostics_path", "diagnostics"),
        ("validation_result_path", "validation_evidence"),
        ("final_summary_path", "final_summary"),
    ):
        path = payload.get(field_name)
        if path:
            print(f"{label}: {path}")
    if result.status == "needs_review":
        candidate_ids = ", ".join(result.review_required_candidate_ids)
        print(f"review_required_candidate_ids: {candidate_ids}")


def _articulation_returncode(result: ArticulationFinalizationResult) -> int:
    return (
        0
        if result.status
        in {"awaiting_decision", "needs_review", "completed", "not_articulated"}
        else 1
    )


def _build_validate_request(
    args: argparse.Namespace,
) -> tuple[ValidationRequest, Path, Path]:
    from world_understanding.validation.cli import (
        build_validation_request_from_inputs,
        scaffold_policy_from_request,
    )

    base_dir = _validate_base_dir(args)
    # Resolve inputs here so the published request is self-contained: resume
    # reloads it without needing the original --base-dir to reproduce identity.
    source = args.usd.expanduser().resolve()
    references = [Path(path).expanduser().resolve() for path in args.reference_image]
    for reference in references:
        # Reject a non-image reference here rather than deep inside the VLM step.
        if reference.suffix.lower() not in IMAGE_REFERENCE_SUFFIXES:
            raise ValueError(
                f"--reference-image expects an image file, got {reference}."
            )
    request = build_validation_request_from_inputs(
        task_description=args.task,
        inputs=(str(source),),
        output_dir=args.output_dir,
        template_overrides=tuple(args.template),
        focus_prim_overrides=tuple(args.focus_prim),
        reference_image_paths=tuple(str(path) for path in references),
        render_backend=args.render_backend,
        render_views=tuple(args.render_view),
        render_image_width=args.render_width,
        render_image_height=args.render_height,
        base_dir=base_dir,
    )
    if args.policy_file is not None:
        policy = dict(request.policy)
        # Validation policy is evidence data, not an agent/provider config.
        # Local OVRTX receipts preserve an unavailable renderer identity as
        # JSON null, and the qualified-evidence validator owns that semantic
        # check after path resolution.
        overrides = _read_json_object(
            args.policy_file,
            config_name="Validation policy",
            allow_null=True,
        )
        _validate_validation_policy_nulls(overrides, source=str(args.policy_file))
        cli_owned_policy_keys = set(policy)
        if args.render_view:
            cli_owned_policy_keys.update(
                {
                    "expected_cameras",
                    "render_view_directions",
                    "runtime_render_views",
                }
            )
        if args.render_width is not None:
            cli_owned_policy_keys.add("render_image_width")
        if args.render_height is not None:
            cli_owned_policy_keys.add("render_image_height")
        conflicts = sorted(set(overrides) & cli_owned_policy_keys)
        if conflicts:
            raise ValueError(
                "--policy-file cannot override policy keys already set by CLI "
                "flags: " + ", ".join(conflicts)
            )
        policy.update(overrides)
        request = request.model_copy(update={"policy": policy})
    request = request.model_copy(
        update={"policy": scaffold_policy_from_request(request, base_dir=base_dir)}
    )
    required_capabilities = tuple(
        dict.fromkeys(getattr(args, "required_capability", ()))
    )
    if required_capabilities:
        metadata = dict(request.metadata)
        metadata["validation_required_capability_ids"] = list(required_capabilities)
        request = request.model_copy(update={"metadata": metadata})
    output_dir = request.project.working_dir
    if not output_dir:
        raise ValueError("validation output directory could not be resolved")
    if args.embedded_run_state is not None:
        request = _bind_embedded_validation_request(
            request,
            run_state_path=args.embedded_run_state,
            output_dir=Path(output_dir),
        )
    return request, Path(output_dir), base_dir


def _handle_validate_run(args: argparse.Namespace) -> int:
    request, output_dir, base_dir = _build_validate_request(args)
    # ``--embedded-run-state`` is the established composed-asset compatibility
    # surface.  Keep routing it through the classic deterministic executor even
    # when older callers do not spell the newer ``--direct-executor`` alias.
    if args.direct_executor or args.embedded_run_state is not None:
        coordinator_only = tuple(
            flag
            for flag, supplied in (
                ("--dry-run", args.dry_run),
                ("--repo-root", args.repo_root is not None),
                ("--runner", args.runner is not None),
                ("--model", args.model is not None),
                ("--model-reasoning-effort", args.model_reasoning_effort is not None),
                ("--codex-base-url", args.codex_base_url is not None),
                ("--codex-sandbox-mode", args.codex_sandbox_mode is not None),
                ("--codex-config-json", bool(args.codex_config_json)),
                ("--codex-config-file", bool(args.codex_config_file)),
                (
                    "--claude-permission-mode",
                    args.claude_permission_mode is not None,
                ),
                ("--claude-max-turns", args.claude_max_turns is not None),
                (
                    "--claude-execution-mode",
                    args.claude_execution_mode is not None,
                ),
                ("--claude-config-json", bool(args.claude_config_json)),
                ("--claude-config-file", bool(args.claude_config_file)),
                ("--child-timeout", args.child_timeout is not None),
                ("--agent-cwd", args.agent_cwd is not None),
                ("--required-capability", bool(args.required_capability)),
            )
            if supplied
        )
        if coordinator_only:
            raise ValueError(
                "these options require the agentic coordinator path: "
                + ", ".join(coordinator_only)
            )
        return _run_validation_workflow_command(
            args,
            request=request,
            output_dir=output_dir,
            base_dir=base_dir,
            resume=False,
        )
    if args.template:
        raise ValueError(
            "--template selects the legacy plan and requires --direct-executor; "
            "agentic Validation delegates check selection to the planning child"
        )
    if args.runner is None:
        raise ValueError(
            "--runner is required for agentic Validation; select codex or claude "
            "explicitly"
        )
    if args.model is None or not args.model.strip():
        raise ValueError(
            "--model is required for agentic Validation; select the child model "
            "explicitly"
        )
    if args.runner == RUNNER_CLAUDE and args.claude_execution_mode is None:
        raise ValueError(
            "--claude-execution-mode is required with --runner claude; select "
            "sdk or cli explicitly"
        )
    if args.runner == RUNNER_CODEX and args.claude_execution_mode is not None:
        raise ValueError(
            "--claude-execution-mode may only be selected with --runner claude"
        )
    from world_understanding.validation.cli import validation_exit_code

    from content_workflow_cli.validation_runner import (
        ValidationCoordinatorRunConfig,
        run_validation_coordinator,
    )

    repo_root = args.repo_root.resolve() if args.repo_root else find_repo_root()
    result = run_validation_coordinator(
        ValidationCoordinatorRunConfig(
            repo_root=repo_root,
            request=request.model_copy(update={"requested_templates": ()}),
            output_dir=output_dir,
            config_base_dir=base_dir,
            runner=args.runner,
            model=args.model.strip(),
            model_reasoning_effort=args.model_reasoning_effort,
            codex_base_url=(
                args.codex_base_url
                if args.codex_base_url is not None
                else os.getenv("CONTENT_AGENT_CODEX_BASE_URL")
            ),
            codex_sandbox_mode=(
                args.codex_sandbox_mode
                if args.codex_sandbox_mode is not None
                else _restricted_default_codex_sandbox_mode()
            ),
            codex_config=_load_codex_config(args),
            claude_config=_load_claude_config(args),
            claude_permission_mode=args.claude_permission_mode or "default",
            claude_max_turns=args.claude_max_turns,
            claude_execution_mode=args.claude_execution_mode,
            child_timeout_seconds=(
                1800.0 if args.child_timeout is None else args.child_timeout
            ),
            agent_cwd=args.agent_cwd,
            dry_run=args.dry_run,
        )
    )
    if args.json:
        payload: dict[str, object] = {
            "status": result.status,
            "output_dir": str(result.output_dir),
            "preparation_path": str(result.preparation_path),
            "plan_patch_path": str(result.plan_patch_path),
            "execution_receipt_path": (
                str(result.execution_receipt_path)
                if result.execution_receipt_path is not None
                else None
            ),
            "validation_result": (
                result.run.result.model_dump(mode="json")
                if result.run is not None
                else None
            ),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif result.status == "prepared":
        print(f"Validation coordinator prepared: {result.output_dir}")
        print(f"Preparation: {result.preparation_path}")
    else:
        if result.run is None:  # pragma: no cover - launcher invariant
            raise RuntimeError("Validation coordinator omitted its workflow result")
        _print_validation_run_summary(result.run)
        print("Status: assessment_required")
        print(
            "Next: content-workflow-cli validate collect-evidence "
            f"--output-dir {result.output_dir}"
        )
    if result.run is None:
        return 0
    return validation_exit_code(
        result.run.result.verdict,
        fail_on_warn=bool(args.fail_on_warn),
    )


def _handle_validate_prepare(args: argparse.Namespace) -> int:
    from content_agent_workflows.validation import prepare_validation_operations

    request, output_dir, base_dir = _build_validate_request(args)
    preparation = prepare_validation_operations(
        request,
        output_dir=output_dir,
        config_base_dir=base_dir,
        requested_rules=tuple(args.rule),
        named_profile=args.profile,
    )
    payload = preparation.model_dump(mode="json")
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"Validation operations prepared: {output_dir}")
        print(
            "Expanded templates: " + ", ".join(preparation.request.requested_templates)
        )
        for capability in preparation.capabilities:
            print(
                f"  {capability.template_name}: {capability.selection_state}; "
                + ", ".join(capability.required_capabilities)
            )
    return 0


def _handle_validate_check(args: argparse.Namespace) -> int:
    from content_agent_workflows.validation import run_validation_operation

    result = run_validation_operation(
        args.output_dir,
        template_name=args.template,
        prior_result_paths=tuple(args.prior_result),
    )
    payload = result.model_dump(mode="json")
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            f"Validation operation {result.template_name}: "
            f"{result.template_result.status}"
        )
        if (
            result.template_name == "look_right"
            and result.template_result.status == "skipped"
        ):
            print("Semantic outcome: not_evaluated (not passed)")
        print(f"Result: {result.template_result_path}")
    optional_not_evaluated = (
        result.template_name == "look_right"
        and result.template_result.status == "skipped"
    )
    return 0 if result.template_result.passed or optional_not_evaluated else 1


def _handle_validate_finalize(args: argparse.Namespace) -> int:
    from content_agent_workflows.validation import finalize_validation_operations
    from world_understanding.validation.cli import validation_exit_code

    run = finalize_validation_operations(args.output_dir)
    if args.json:
        print(json.dumps(run.result.model_dump(mode="json"), indent=2, sort_keys=True))
    else:
        _print_validation_run_summary(run)
    return validation_exit_code(
        run.result.verdict,
        fail_on_warn=bool(args.fail_on_warn),
    )


def _handle_validate_ingest_verified_operation_result(
    args: argparse.Namespace,
) -> int:
    from content_agent_workflows.validation import ingest_verified_operation_result

    receipt = ingest_verified_operation_result(
        args.envelope,
        output_dir=args.output_dir,
    )
    if args.json:
        print(json.dumps(receipt.model_dump(mode="json"), indent=2, sort_keys=True))
    else:
        print(f"Verified operation ingested: {receipt.envelope.operation_id}")
        print(f"Native status: {receipt.envelope.native_status}")
        print(f"Mode: {receipt.execution_mode}")
    return 0


def _handle_validate_produce_canonical_visual_evidence(
    args: argparse.Namespace,
) -> int:
    from content_agent_workflows.validation import produce_canonical_visual_evidence

    publication = produce_canonical_visual_evidence(
        post_mutation_usd=args.usd,
        source_usd=args.source_usd,
        output_dir=args.output_dir,
        backend=args.render_backend,
        views=tuple(args.render_view) or ("+x+y+z",),
        image_width=args.render_width,
        image_height=args.render_height,
        operation_id=args.operation_id,
        gate_id=args.gate_id,
    )
    if args.json:
        print(json.dumps(publication.model_dump(mode="json"), indent=2, sort_keys=True))
    else:
        print(f"Canonical visual evidence: {publication.envelope.path}")
        print(f"Output USD SHA-256: {publication.result.output.sha256}")
        print("Semantic authority: outer coordinator")
    return 0


def _handle_validate_collect_evidence(args: argparse.Namespace) -> int:
    from content_agent_workflows.validation import (
        EmbeddedValidationAssessmentError,
        VerifiedOperationError,
        collect_verified_operation_evidence,
        is_verified_operation_ingest_run,
        load_embedded_validation_run,
        prepare_embedded_validation_evidence,
        prepare_standalone_validation_evidence,
    )

    try:
        if is_verified_operation_ingest_run(args.output_dir):
            if args.embedded_run_state is not None:
                raise VerifiedOperationError(
                    "provided-operation mode cannot use --embedded-run-state"
                )
            provided_index = collect_verified_operation_evidence(args.output_dir)
            payload = provided_index.model_dump(mode="json")
            record_count = len(provided_index.records)
        elif args.embedded_run_state is None:
            standalone_index = prepare_standalone_validation_evidence(args.output_dir)
            payload = standalone_index.model_dump(mode="json")
            record_count = len(standalone_index.records)
        else:
            run = load_embedded_validation_run(args.output_dir)
            embedded_index = prepare_embedded_validation_evidence(
                run,
                run_state_path=args.embedded_run_state,
            )
            payload = embedded_index.model_dump(mode="json")
            record_count = len(embedded_index.evidence)
    except (EmbeddedValidationAssessmentError, VerifiedOperationError) as exc:
        print(f"{CLI_NAME}: error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"Validation assessment evidence: {args.output_dir}")
        print(f"Evidence records: {record_count}")
    return 0


def _prepare_fixed_pipeline_embedded_validation_evidence(
    run: ValidationWorkflowRun,
    *,
    run_state_path: Path,
    defer_until_cross_stage: bool,
) -> bool:
    """Preserve fixed-pipeline embedded evidence preparation around cross-stage timing."""

    # Legacy combined ``validate run --embedded-run-state`` prepared its
    # evidence automatically. Cross-stage evidence is now part of assessment
    # identity, but the outer coordinator can only author it after the native
    # result exists. Defer that one timing dependency to the fixed-pipeline ``assess``
    # command; focused/standalone operation flows remain explicitly decomposed.
    cross_stage_path = Path(run.output_dir).parent / "cross_stage_validation.json"
    if (
        defer_until_cross_stage
        and not cross_stage_path.exists()
        and not cross_stage_path.is_symlink()
    ):
        return False

    from content_agent_workflows.validation import (
        prepare_embedded_validation_evidence,
    )

    prepare_embedded_validation_evidence(
        run,
        run_state_path=run_state_path,
    )
    return True


def _handle_validate_resume(args: argparse.Namespace) -> int:
    from world_understanding.validation import ValidationRequest

    base_dir = _validate_base_dir(args)
    output_dir = args.output_dir.expanduser().resolve()
    request_path = output_dir / "validation_request.json"
    if not request_path.is_file():
        print(
            f"{CLI_NAME}: error: no validation run to resume at {output_dir}. "
            "Expected validation_request.json.",
            file=sys.stderr,
        )
        return 2
    # Resume replays the request the first attempt published, so prompt, input,
    # reference, template, and policy identity are reproduced exactly. A changed
    # asset or prompt fails closed inside the workflow rather than here.
    request = ValidationRequest.model_validate_json(
        request_path.read_text(encoding="utf-8")
    )
    from content_agent_workflows.common.domain_execution import (
        domain_execution_context_from_metadata,
    )

    embedded_context = domain_execution_context_from_metadata(
        request.metadata,
        expected_domain="validation",
    )
    if embedded_context is not None and args.embedded_run_state is None:
        print(
            f"{CLI_NAME}: error: embedded Validation resume requires "
            "--embedded-run-state.",
            file=sys.stderr,
        )
        return 2
    if embedded_context is None and args.embedded_run_state is not None:
        print(
            f"{CLI_NAME}: error: --embedded-run-state cannot reinterpret a "
            "standalone Validation request.",
            file=sys.stderr,
        )
        return 2
    if embedded_context is not None:
        rebound = _bind_embedded_validation_request(
            request,
            run_state_path=args.embedded_run_state,
            output_dir=output_dir,
        )
        if rebound != request:
            print(
                f"{CLI_NAME}: error: persisted embedded Validation context is stale.",
                file=sys.stderr,
            )
            return 2
    from content_agent_workflows.validation import (
        VALIDATION_COORDINATOR_PREPARATION_NAME,
        VALIDATION_COORDINATOR_SAFE_RESTART_NAME,
        write_validation_coordinator_safe_restart,
    )

    coordinator_preparation = output_dir / VALIDATION_COORDINATOR_PREPARATION_NAME
    if coordinator_preparation.exists() or coordinator_preparation.is_symlink():
        receipt = write_validation_coordinator_safe_restart(
            output_dir,
            request=request,
        )
        if args.json:
            print(json.dumps(receipt.model_dump(mode="json"), indent=2, sort_keys=True))
        else:
            print(
                f"{CLI_NAME}: error: agentic Validation coordinator runs do not "
                "resume in place. The prior run was preserved and no legacy "
                "Validation executor was invoked. Safe-restart disposition: "
                f"{output_dir / VALIDATION_COORDINATOR_SAFE_RESTART_NAME}. Start "
                "a fresh `content-workflow-cli validate run` with the same "
                "inputs and explicit runner configuration plus "
                "`--output-dir <new-empty-directory>`.",
                file=sys.stderr,
            )
        return 2
    if args.recover_orphaned_claims:
        # Resume cannot tell a crashed runner from a live concurrent one, so it
        # fails closed on an in-flight claim. Releasing it is the operator's
        # explicit assertion that the previous runner is gone.
        from content_agent_workflows.validation import (
            recover_orphaned_validation_claims,
        )

        recover_orphaned_validation_claims(output_dir)
    return _run_validation_workflow_command(
        args,
        request=request,
        output_dir=output_dir,
        base_dir=base_dir,
        resume=True,
    )


def _bind_embedded_validation_request(
    request: ValidationRequest,
    *,
    run_state_path: Path,
    output_dir: Path,
) -> ValidationRequest:
    from content_agent_workflows.asset_composition import (
        build_embedded_domain_execution_context,
    )
    from content_agent_workflows.common.domain_execution import (
        metadata_with_domain_execution_context,
    )

    if len(request.inputs) != 1:
        raise ValueError("embedded Validation requires exactly one input asset")
    context = build_embedded_domain_execution_context(
        run_state_path,
        domain="validation",
        input_asset=request.inputs[0],
        output_dir=output_dir,
    )
    return request.model_copy(
        update={
            "metadata": metadata_with_domain_execution_context(
                request.metadata,
                context,
            )
        }
    )


def _handle_validate_assess(args: argparse.Namespace) -> int:
    from content_agent_workflows.validation import (
        EMBEDDED_VALIDATION_EVIDENCE_INDEX_NAME,
        EmbeddedValidationAssessmentError,
        VerifiedOperationError,
        assess_embedded_validation,
        assess_standalone_validation,
        assess_verified_operation_evidence,
        is_verified_operation_ingest_run,
        load_embedded_validation_run,
    )

    try:
        if is_verified_operation_ingest_run(args.output_dir):
            if args.embedded_run_state is not None:
                raise VerifiedOperationError(
                    "provided-operation mode cannot use --embedded-run-state"
                )
            provided_index = assess_verified_operation_evidence(
                args.output_dir,
                assessment_path=args.assessment,
            )
            payload = provided_index.model_dump(mode="json")
            authorized = provided_index.authorized
            assessment_id = provided_index.assessment_id
        elif args.embedded_run_state is None:
            standalone_index = assess_standalone_validation(
                args.output_dir,
                assessment_path=args.assessment,
            )
            payload = standalone_index.model_dump(mode="json")
            authorized = standalone_index.authorized
            assessment_id = standalone_index.assessment_id
        else:
            evidence_index_path = (
                Path(args.output_dir) / EMBEDDED_VALIDATION_EVIDENCE_INDEX_NAME
            )
            if (
                not evidence_index_path.exists()
                and not evidence_index_path.is_symlink()
            ):
                _prepare_fixed_pipeline_embedded_validation_evidence(
                    load_embedded_validation_run(args.output_dir),
                    run_state_path=args.embedded_run_state,
                    defer_until_cross_stage=False,
                )
            embedded_index = assess_embedded_validation(
                args.output_dir,
                run_state_path=args.embedded_run_state,
                assessment_path=args.assessment,
            )
            payload = embedded_index.model_dump(mode="json")
            authorized = embedded_index.authorized
            assessment_id = embedded_index.assessment_id
    except (EmbeddedValidationAssessmentError, VerifiedOperationError) as exc:
        print(f"{CLI_NAME}: error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        state = "authorized" if authorized else "not authorized"
        print(f"Validation assessment {state}: {assessment_id}")
    return 0 if authorized else 1


def _handle_validate_review_assessment(args: argparse.Namespace) -> int:
    from content_agent_workflows.validation import (
        EmbeddedValidationAssessmentError,
        ValidationTerminalReceipt,
        VerifiedOperationError,
        VerifiedOperationTerminalReceipt,
        is_verified_operation_ingest_run,
        review_embedded_validation_assessment,
        review_standalone_validation_assessment,
        review_verified_operation_assessment,
    )

    receipt: VerifiedOperationTerminalReceipt | ValidationTerminalReceipt
    try:
        if is_verified_operation_ingest_run(args.output_dir):
            if args.embedded_run_state is not None:
                raise VerifiedOperationError(
                    "provided-operation mode cannot use --embedded-run-state"
                )
            receipt = review_verified_operation_assessment(
                args.output_dir,
                review_path=args.review,
            )
        elif args.embedded_run_state is None:
            receipt = review_standalone_validation_assessment(
                args.output_dir,
                review_path=args.review,
            )
        else:
            embedded_index = review_embedded_validation_assessment(
                args.output_dir,
                run_state_path=args.embedded_run_state,
                review_path=args.review,
            )
            receipt = ValidationTerminalReceipt.model_validate_json(
                Path(embedded_index.terminal_receipt.path).read_text(encoding="utf-8")
            )
    except (EmbeddedValidationAssessmentError, VerifiedOperationError) as exc:
        print(f"{CLI_NAME}: error: {exc}", file=sys.stderr)
        return 2
    payload = receipt.model_dump(mode="json")
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"Validation assessment receipt: {receipt.receipt_status}")
    return 0 if receipt.receipt_status == "completed" else 1


def _validate_base_dir(args: argparse.Namespace) -> Path:
    if args.base_dir is not None:
        return Path(args.base_dir).expanduser().resolve()
    return Path.cwd().resolve()


def _validation_resume_command(
    output_dir: Path,
    embedded_run_state: Path | None,
) -> str:
    command = [
        CLI_NAME,
        "validate",
        "resume",
        "--output-dir",
        str(output_dir),
        "--recover-orphaned-claims",
    ]
    if embedded_run_state is not None:
        command.extend(["--embedded-run-state", str(embedded_run_state)])
    return shlex.join(command)


def _run_validation_workflow_command(
    args: argparse.Namespace,
    *,
    request: ValidationRequest,
    output_dir: Path,
    base_dir: Path,
    resume: bool,
) -> int:
    from content_agent_workflows.validation import run_validation_workflow
    from world_understanding.validation.cli import validation_exit_code

    try:
        run = run_validation_workflow(
            request,
            output_dir=output_dir,
            config_base_dir=base_dir,
            resume=resume,
        )
        if args.embedded_run_state is not None and run.status.value == "completed":
            _prepare_fixed_pipeline_embedded_validation_evidence(
                run,
                run_state_path=args.embedded_run_state,
                defer_until_cross_stage=True,
            )
    except KeyboardInterrupt:
        # Accepted checks are already durable, so point at the exact recovery
        # command instead of leaving the operator with a bare traceback.
        print(
            f"{CLI_NAME}: interrupted. Completed checks are preserved; resume "
            f"with:\n  "
            f"{_validation_resume_command(output_dir, args.embedded_run_state)}",
            file=sys.stderr,
        )
        raise
    if args.json:
        print(json.dumps(run.result.model_dump(mode="json"), indent=2, sort_keys=True))
    else:
        _print_validation_run_summary(run)
    if run.status.value == "cancelled":
        # A cancelled run keeps its completed evidence, but it is not a pass.
        return 3
    exit_code: int = validation_exit_code(
        run.result.verdict,
        fail_on_warn=bool(args.fail_on_warn),
    )
    return exit_code


def _print_validation_run_summary(run: ValidationWorkflowRun) -> None:
    print(f"Verdict: {run.result.verdict} ({run.status.value})")
    print(f"Run directory: {run.output_dir}")
    print(f"Request: {run.request_path}")
    print(f"Plan: {run.plan_path}")
    print(f"Result: {run.result_path}")
    print(f"Evidence: {run.evidence_path}")
    print(f"Summary: {run.final_summary_path}")
    for record in run.checkpoint.records:
        state = record.state.value
        detail = f" ({record.last_error})" if record.last_error else ""
        print(f"  {record.template_name}: {state}{detail}")
    if run.result.recommended_action:
        print(f"Recommended action: {run.result.recommended_action}")


def _physics_vomp_config_from_args(
    args: argparse.Namespace,
) -> PhysicsVompMassConfig | None:
    from content_agent_workflows.physics import PhysicsVompMassConfig

    option_names = (
        "vomp_target_prim",
        "vomp_python",
        "vomp_config",
        "vomp_ovrtx_venv",
        "vomp_num_views",
        "vomp_image_size",
        "vomp_seed",
        "vomp_render_mode",
        "vomp_num_sensor_updates",
        "vomp_material_target",
        "vomp_attention_backend",
        "vomp_timeout_seconds",
        "vomp_max_complete_voxels",
    )
    if args.vomp_root is None:
        supplied = [name for name in option_names if getattr(args, name) is not None]
        if supplied:
            flags = ", ".join(f"--{name.replace('_', '-')}" for name in supplied)
            raise ValueError(f"{flags} require --vomp-root")
        return None

    values: dict[str, object] = {
        "runtime_root": args.vomp_root.expanduser().resolve(),
    }
    if args.vomp_target_prim is not None:
        values["target_prim_path"] = args.vomp_target_prim
    if args.vomp_python is not None:
        values["python_executable"] = args.vomp_python.expanduser()
    if args.vomp_config is not None:
        values["config_path"] = args.vomp_config.expanduser()
    if args.vomp_ovrtx_venv is not None:
        values["ovrtx_venv_dir"] = args.vomp_ovrtx_venv.expanduser().resolve()
    if args.vomp_num_views is not None:
        values["num_views"] = args.vomp_num_views
    if args.vomp_image_size is not None:
        values["image_width"] = args.vomp_image_size
        values["image_height"] = args.vomp_image_size
    if args.vomp_seed is not None:
        values["seed"] = args.vomp_seed
    if args.vomp_render_mode is not None:
        values["render_mode"] = args.vomp_render_mode
    if args.vomp_num_sensor_updates is not None:
        values["num_sensor_updates"] = args.vomp_num_sensor_updates
    if args.vomp_material_target is not None:
        values["material_target"] = args.vomp_material_target
    if args.vomp_attention_backend is not None:
        values["attention_backend"] = args.vomp_attention_backend
    if args.vomp_timeout_seconds is not None:
        values["timeout_seconds"] = args.vomp_timeout_seconds
    if args.vomp_max_complete_voxels is not None:
        values["max_complete_voxels"] = args.vomp_max_complete_voxels
    return PhysicsVompMassConfig.model_validate(values)


def _handle_physics_apply(args: argparse.Namespace) -> int:
    if getattr(args, "used_deterministic_workflow_alias", False):
        print(
            "WARNING: --deterministic-workflow is deprecated; "
            "use --direct-executor instead.",
            file=sys.stderr,
        )
    repo_root = args.repo_root.resolve() if args.repo_root else find_repo_root()
    if not usd_cli_source_distributed(repo_root):
        print(USD_CLI_UNAVAILABLE_HINT, file=sys.stderr)
        return 2

    vomp_mass = _physics_vomp_config_from_args(args)
    explicit_reference_images = [
        path.expanduser().resolve() for path in args.reference_image
    ]
    generic_reference_images, reference_files = _split_generic_references(
        args.reference
    )
    video_references = [
        path
        for path in [*explicit_reference_images, *reference_files]
        if path.suffix.lower() in VIDEO_REFERENCE_SUFFIXES
    ]
    if video_references:
        print(
            "Video references are unsupported in the public 0.6 workflow; "
            "supply representative reference images instead: "
            + ", ".join(str(path) for path in video_references),
            file=sys.stderr,
        )
        return 2
    reference_images = _dedupe_paths(
        explicit_reference_images + generic_reference_images
    )
    if not args.deterministic_workflow:
        if args.resume:
            print(
                "--resume is only supported with --deterministic-workflow.",
                file=sys.stderr,
            )
            return 2
        if args.decision_patch is not None or args.topology_plan is not None:
            print(
                "--decision-patch and --topology-plan require --direct-executor.",
                file=sys.stderr,
            )
            return 2
        additional_instructions = args.additional_instructions
        if args.additional_instructions_file:
            file_text = _read_additional_instructions_file(
                args.additional_instructions_file
            )
            additional_instructions = (
                f"{additional_instructions}\n{file_text}"
                if additional_instructions
                else file_text
            )
        behavior_prompt = args.behavior_prompt
        if args.behavior_prompt_file:
            file_text = _read_additional_instructions_file(args.behavior_prompt_file)
            behavior_prompt = (
                f"{behavior_prompt}\n{file_text}" if behavior_prompt else file_text
            )
        scenario_path = (
            args.scenario.expanduser().resolve() if args.scenario is not None else None
        )
        agent_result = run_physics_apply(
            PhysicsApplyConfig(
                repo_root=repo_root,
                usd_path=args.usd.expanduser().resolve(),
                reference_images=reference_images,
                reference_files=reference_files,
                output_dir=args.output_dir.expanduser(),
                output_usd_path=args.output_usd.expanduser()
                if args.output_usd is not None
                else None,
                collision_approximation=args.collision_approximation,
                run_simulation=not args.no_simulation
                and args.simulation_engine != "none",
                simulation_engine=args.simulation_engine,
                simulation_duration_s=args.duration_s,
                simulation_dt=args.dt,
                simulation_sample_fps=args.sample_fps,
                drop_height_m=args.drop_height_m,
                vomp_mass=vomp_mass,
                fail_on_validation_error=args.fail_on_validation_error,
                runner=args.runner,
                model=args.model,
                model_reasoning_effort=args.model_reasoning_effort,
                codex_base_url=args.codex_base_url,
                codex_sandbox_mode=args.codex_sandbox_mode,
                codex_config=_load_codex_config(args),
                claude_config=_load_claude_config(args),
                claude_permission_mode=args.claude_permission_mode,
                claude_max_turns=args.claude_max_turns,
                claude_execution_mode=args.claude_execution_mode,
                dry_run=args.dry_run,
                optimize=args.optimize,
                optimizer_selection=args.optimizer_selection,
                flatten_prototypes=args.flatten_prototypes,
                enable_deinstance=args.enable_deinstance,
                enable_split=args.enable_split,
                enable_deduplicate=args.enable_deduplicate,
                scene_tool_timeout_seconds=args.scene_tool_timeout,
                child_timeout_seconds=args.child_timeout,
                prompt_mode=args.prompt_mode,
                vqa_refinement_max_iterations=args.vqa_refinement_max_iterations,
                codex_persistent_refinement=args.codex_persistent_refinement,
                additional_instructions=additional_instructions,
                behavior_prompt=behavior_prompt,
                scenario_path=scenario_path,
                tune=args.tune or args.refine,
                refine=args.refine,
                tune_engine=args.tune_engine,
                optimizer=args.optimizer,
                max_trials=args.max_trials,
                max_iterations=args.max_iterations,
                sweep_deadline_seconds=args.sweep_deadline_seconds,
                tuning_phase_deadline_seconds=args.tuning_phase_deadline_seconds,
                revalidation_max_penetration_m=args.revalidation_max_penetration_m,
                log_to_stderr=(
                    args.log_to_stderr if args.log_to_stderr is not None else True
                ),
            )
        )
        if args.json:
            print(
                json.dumps(
                    {
                        "run_dir": str(agent_result.run_dir),
                        "prompt_path": str(agent_result.prompt_path),
                        "request_path": str(agent_result.request_path),
                        "child_output_path": str(agent_result.child_output_path),
                        "child_final_path": str(agent_result.child_final_path),
                        "returncode": agent_result.returncode,
                        "trace_paths": agent_result.trace_paths,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            print(f"Run directory: {agent_result.run_dir}")
            print(f"Prompt: {agent_result.prompt_path}")
            print(f"Request: {agent_result.request_path}")
            print(f"Child output: {agent_result.child_output_path}")
            print(f"Child final: {agent_result.child_final_path}")
            for label, path in agent_result.trace_paths.items():
                print(f"{label}: {path}")
        if agent_result.returncode != 0:
            # The child's final message is often a success-looking summary
            # (it authored its patch before a later stage failed); without an
            # explicit error line here a failed run reads as a passed one and
            # the real cause stays buried in a raw/ JSON artifact.
            detail = ""
            try:
                # On the tune/refine path the finalize phase succeeded before
                # tuning started, so the latest finalize record is a CLEAN
                # one — the actual cause lives in the tuning summary. Prefer
                # it there; otherwise fall back to the finalize record.
                tuning_path = (
                    agent_result.run_dir / "raw" / "physics_agentic_tuning_result.json"
                )
                if (args.tune or args.refine) and tuning_path.is_file():
                    tuning = json.loads(tuning_path.read_text(encoding="utf-8"))
                    error = tuning.get("error") if isinstance(tuning, dict) else None
                    detail = f": tuning failed: {error}" if error else ""
                    detail += (
                        f" (tuning report at {tuning_path})"
                        if detail
                        else f": tuning report at {tuning_path}"
                    )
                else:
                    record, record_path = _latest_physics_finalize_record(
                        agent_result.run_dir
                    )
                    if record.get("error"):
                        detail = f": finalize failed: {record['error']}"
                    if record:
                        # A record without a recorded error is still the
                        # closest artifact to the failure; always point at
                        # it, with an explicit label so the line reads as a
                        # diagnostic pointer rather than an opaque path.
                        detail += (
                            f": finalize report at {record_path}"
                            if not detail
                            else f" (finalize report at {record_path})"
                        )
            except Exception:  # noqa: BLE001 — diagnostics only, never mask the exit
                pass
            print(
                "error: physics apply failed with exit code "
                f"{agent_result.returncode}{detail}",
                file=sys.stderr,
            )
        return agent_result.returncode

    deterministic_unsupported = (
        args.tune
        or args.refine
        or args.behavior_prompt
        or args.behavior_prompt_file
        or args.scenario
    )
    if deterministic_unsupported:
        print(
            "agentic physics tune/refine flags require the default long-running "
            "workflow; remove --direct-executor.",
            file=sys.stderr,
        )
        return 2

    from content_agent_workflows.physics import (
        PhysicsApplyWorkflowInput,
        run_physics_apply_workflow,
    )

    input_usd = args.usd.expanduser().resolve()
    # Keep the caller's lexical output directory: the workflow's own preflight
    # rejects symlink traversal, and resolve() here would erase that evidence.
    output_dir = Path(os.path.abspath(args.output_dir.expanduser()))
    workflow_input = PhysicsApplyWorkflowInput(
        usd_path=input_usd,
        output_dir=output_dir,
        output_usd_path=args.output_usd.expanduser().resolve()
        if args.output_usd is not None
        else None,
        decision_patch_path=(
            args.decision_patch.expanduser().resolve()
            if args.decision_patch is not None
            else None
        ),
        topology_plan_path=(
            args.topology_plan.expanduser().resolve()
            if args.topology_plan is not None
            else None
        ),
        collision_approximation=args.collision_approximation,
        run_simulation=(not args.no_simulation and args.simulation_engine != "none"),
        simulation_engine=args.simulation_engine,
        simulation_duration_s=args.duration_s,
        simulation_dt=args.dt,
        simulation_sample_fps=args.sample_fps,
        drop_height_m=args.drop_height_m,
        vomp_mass=vomp_mass,
        # The agentic path honours this override at the apply-phase
        # runtime gate; silently dropping it here would leave the
        # deterministic branch enforcing the 0.005 m default the flag
        # exists to relax.
        max_ground_penetration_m=args.revalidation_max_penetration_m,
        fail_on_validation_error=args.fail_on_validation_error,
        scene_tool_timeout_seconds=args.scene_tool_timeout,
        resume=args.resume,
    )
    workflow_result = run_physics_apply_workflow(workflow_input)
    payload = workflow_result.model_dump(mode="json")
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        status = "ok" if workflow_result.success else "failed"
        print(f"physics apply {status}: {workflow_result.validation_status}")
        if workflow_result.physics_usd_path:
            print(f"physics_usd: {workflow_result.physics_usd_path}")
        if workflow_result.assignments_path:
            print(f"assignments: {workflow_result.assignments_path}")
        if workflow_result.validation_evidence_path:
            print(f"validation_evidence: {workflow_result.validation_evidence_path}")
        if workflow_result.vomp_result_path:
            print(f"vomp_result: {workflow_result.vomp_result_path}")
        if workflow_result.vomp_provenance_path:
            print(f"vomp_provenance: {workflow_result.vomp_provenance_path}")
        if workflow_result.error:
            print(f"error: {workflow_result.error}", file=sys.stderr)
    return 0 if workflow_result.success else 1


def _split_generic_references(paths: list[Path]) -> tuple[list[Path], list[Path]]:
    reference_images: list[Path] = []
    reference_files: list[Path] = []
    for path in paths:
        resolved = path.expanduser().resolve()
        if _is_image_reference(resolved):
            reference_images.append(resolved)
        else:
            reference_files.append(resolved)
    return _dedupe_paths(reference_images), _dedupe_paths(reference_files)


def _expand_reference_directories(
    paths: list[Path],
) -> tuple[list[Path], list[Path]]:
    directories: list[Path] = []
    references: list[Path] = []
    for path in paths:
        directory = path.expanduser().resolve()
        if not directory.exists():
            raise FileNotFoundError(f"--reference-dir does not exist: {directory}")
        if not directory.is_dir():
            raise NotADirectoryError(f"--reference-dir is not a directory: {directory}")
        directories.append(directory)
        children = sorted(
            (
                child.resolve()
                for child in directory.iterdir()
                if child.is_file()
                and child.suffix.lower() in REFERENCE_DIRECTORY_SUFFIXES
            ),
            key=lambda child: (child.name.casefold(), child.name),
        )
        references.extend(children)
    return _dedupe_paths(directories), _dedupe_paths(references)


def _is_image_reference(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_REFERENCE_SUFFIXES


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    deduped: list[Path] = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def _read_additional_instructions_file(path: Path) -> str:
    expanded = path.expanduser()
    if not expanded.exists():
        raise FileNotFoundError(
            f"--additional-instructions-file does not exist: {expanded}"
        )
    if not expanded.is_file():
        raise IsADirectoryError(
            f"--additional-instructions-file is not a file: {expanded}"
        )
    try:
        return expanded.read_text(encoding="utf-8")
    except OSError as exc:
        raise OSError(
            f"could not read --additional-instructions-file {expanded}: {exc}"
        ) from exc


def _handle_auth_login(args: argparse.Namespace) -> int:
    command = [_codex_executable(), "login"]
    if args.device_code:
        command.append("--device-auth")
    return _run_codex_command(command)


def _handle_auth_status(args: argparse.Namespace) -> int:
    executable = _codex_executable()
    status_returncode = _run_codex_command([executable, "login", "status"])
    if status_returncode != 0:
        return status_returncode
    prerequisite_error = _linux_codex_sandbox_prerequisite_error()
    if prerequisite_error:
        print(f"error: {prerequisite_error}", file=sys.stderr)
        return 1
    return _probe_codex_workspace_write_access(executable)


def _linux_codex_sandbox_prerequisite_error() -> str | None:
    """Return an actionable Linux sandbox prerequisite failure, if any."""

    if not sys.platform.startswith("linux"):
        return None
    bwrap = shutil.which("bwrap")
    if bwrap is None:
        return (
            "Codex workspace-write requires bubblewrap (bwrap) on Linux. Install "
            "your distribution's bubblewrap package, enable unprivileged user "
            "namespaces, then rerun `content-workflow-cli auth status`."
        )
    smoke_executable = shutil.which("true")
    if smoke_executable is None:
        return (
            "Codex workspace-write requires an executable `true` command to verify "
            "bubblewrap on Linux. Install your distribution's core utilities, then "
            "rerun `content-workflow-cli auth status`."
        )
    try:
        completed = subprocess.run(
            [
                bwrap,
                "--ro-bind",
                "/",
                "/",
                "--dev",
                "/dev",
                "--proc",
                "/proc",
                "--",
                smoke_executable,
            ],
            check=False,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return (
            "bubblewrap could not start a user namespace for Codex workspace-write "
            f"({type(exc).__name__}: {_sanitize_native_diagnostic(str(exc))}). "
            "Enable unprivileged user namespaces and verify bwrap is permitted by "
            "the host/container security policy."
        )
    if completed.returncode:
        details = _sanitize_native_diagnostic(completed.stderr or completed.stdout)
        return (
            "bubblewrap cannot create the Codex workspace-write sandbox "
            f"(exit {completed.returncode}{': ' + details if details else ''}). "
            "Enable unprivileged user namespaces and verify bwrap is permitted by "
            "the host/container security policy."
        )
    return None


def _sanitize_native_diagnostic(value: str, *, limit: int = 1200) -> str:
    """Keep native diagnostics actionable without copying credentials to the terminal."""

    redacted = re.sub(
        r"(?i)\b\"?([a-z0-9 _-]*(?:authorization|cookie|credential|password|secret|token|api[ _-]?key|access[ _-]?key)(?![a-z0-9])[a-z0-9 _-]*)\"?\s*([=:])\s*(?:(?:bearer|basic|token)\s+)?(?:\"(?:\\[\s\S]|[^\"\\])*\"|'(?:\\[\s\S]|[^'\\])*'|`(?:\\[\s\S]|[^`\\])*`|\S+)",
        r"\1\2[redacted]",
        value,
    )
    redacted = re.sub(
        r"(?i)\b(api[_ -]?key|access[_ -]?(?:key|token)|authorization|credential|password|secret|token)\b(?:\s+(?:is|was|provided)){1,3}\s*[:=]?\s*(?:\"(?:\\[\s\S]|[^\"\\])*\"|'(?:\\[\s\S]|[^'\\])*'|`(?:\\[\s\S]|[^`\\])*`|\S+)",
        r"\1 [redacted]",
        redacted,
    )
    redacted = re.sub(
        r"(?i)\b(bearer|basic|token)\s+(?:\"(?:\\[\s\S]|[^\"\\])*\"|'(?:\\[\s\S]|[^'\\])*'|`(?:\\[\s\S]|[^`\\])*`|\S+)",
        r"\1 [redacted]",
        redacted,
    )
    return " ".join(redacted.split())[:limit]


def _smoke_write_command(marker: Path, contents: str) -> str:
    """Return a native-shell command that writes the smoke-test marker."""

    if sys.platform == "win32":
        # The Codex command tool uses PowerShell on native Windows hosts. Keep
        # values literal so temporary-directory names are not parsed as syntax.
        def quote(value: str) -> str:
            return value.replace("'", "''")

        return (
            "Set-Content -NoNewline -LiteralPath "
            f"'{quote(str(marker))}' -Value '{quote(contents)}'"
        )
    return f"printf {shlex.quote(contents)} > {shlex.quote(str(marker))}"


def _read_exact_smoke_marker(marker: Path, expected: str) -> bool:
    """Read only an exact regular marker, without following a replacement link."""

    expected_bytes = expected.encode("utf-8")
    try:
        before = os.lstat(marker)
        if not stat.S_ISREG(before.st_mode) or before.st_size != len(expected_bytes):
            return False
        descriptor = os.open(marker, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError:
        return False
    try:
        after = os.fstat(descriptor)
        if (
            not stat.S_ISREG(after.st_mode)
            or after.st_size != len(expected_bytes)
            or (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino)
        ):
            return False
        return os.read(descriptor, len(expected_bytes) + 1) == expected_bytes
    except OSError:
        return False
    finally:
        os.close(descriptor)


def _probe_codex_workspace_write_access(executable: str) -> int:
    """Prove direct and SDK-bridge Codex children can execute workspace writes."""

    try:
        preflight_codex_windows_support()
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory(prefix="content-workflow-codex-auth-") as cwd:
        marker = Path(cwd) / ".content-workflow-codex-direct-smoke"
        command = [
            executable,
            "exec",
            "--ephemeral",
            "--ignore-user-config",
            "--skip-git-repo-check",
            "--sandbox",
            "workspace-write",
            "--color",
            "never",
            "--cd",
            cwd,
            (
                "Run exactly this "
                f"{'PowerShell' if sys.platform == 'win32' else 'shell'} command using "
                f"the command tool: {_smoke_write_command(marker, 'direct-ok')}. "
                "Then reply exactly OK."
            ),
        ]
        try:
            completed = _run_codex_model_probe(command)
        except FileNotFoundError as exc:
            raise RuntimeError(
                "codex CLI is not installed locally or on PATH. Run `npm ci --prefix "
                f"agentic/packages/content_workflow_cli` before using {CLI_NAME} auth helpers."
            ) from exc
        except subprocess.TimeoutExpired:
            print(
                "error: Codex reports a login, but its workspace-write sandbox probe timed out.",
                file=sys.stderr,
            )
            return 1

        if completed.returncode != 0 or not _read_exact_smoke_marker(
            marker, "direct-ok"
        ):
            print(
                "error: Codex reports a login, but its direct workspace-write sandbox "
                "cannot execute a command.",
                file=sys.stderr,
            )
            details = _sanitize_native_diagnostic(completed.stderr or completed.stdout)
            if details:
                print(details, file=sys.stderr)
            return int(completed.returncode) or 1

        bridge = Path(__file__).with_name("codex_sdk_bridge.mjs")
        try:
            bridged = _run_codex_model_probe(
                [shutil.which("node") or "node", str(bridge), "--sandbox-smoke", cwd]
            )
        except FileNotFoundError:
            print(
                "error: Node.js is required to verify the Codex SDK bridge sandbox.",
                file=sys.stderr,
            )
            return 1
        except subprocess.TimeoutExpired:
            print(
                "error: Codex SDK bridge workspace-write sandbox probe timed out.",
                file=sys.stderr,
            )
            return 1
        if bridged.returncode == 0:
            print("Codex direct and SDK bridge workspace-write sandboxes are usable.")
            return 0

    print(
        "error: Codex direct sandbox passed, but the SDK bridge cannot execute a "
        "workspace-write command.",
        file=sys.stderr,
    )
    details = _sanitize_native_diagnostic(bridged.stderr or bridged.stdout)
    if details:
        print(details, file=sys.stderr)
    return int(bridged.returncode) or 1


_CODEX_MODEL_PROBE_TIMEOUT_SECONDS = 60.0
_CODEX_MODEL_PROBE_CLEANUP_TIMEOUT_SECONDS = 5.0


def _run_codex_model_probe(
    command: list[str],
) -> subprocess.CompletedProcess[str]:
    """Run an isolated Codex probe and clean up its full process group."""

    windows_job: Any | None = None
    creation_flags = 0
    if os.name == "nt":
        from world_understanding.utils.windows_process import WindowsKillOnCloseJob

        windows_job = WindowsKillOnCloseJob()
        creation_flags = windows_job.creation_flags
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            start_new_session=hasattr(os, "setsid"),
            creationflags=creation_flags,
        )
    except BaseException:
        if windows_job is not None:
            windows_job.close()
        raise
    try:
        if windows_job is not None:
            windows_job.assign_process(process)
            windows_job.resume_process(process)
        stdout, stderr = process.communicate(timeout=_CODEX_MODEL_PROBE_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as timeout_error:
        if windows_job is not None:
            windows_job.terminate(exit_code=1)
            try:
                windows_job.wait_for_empty(
                    timeout_s=_CODEX_MODEL_PROBE_CLEANUP_TIMEOUT_SECONDS
                )
            finally:
                windows_job.close()
                windows_job = None
        elif hasattr(os, "killpg"):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        elif process.poll() is None:  # pragma: no cover - legacy fallback
            process.kill()
        try:
            process.communicate(timeout=_CODEX_MODEL_PROBE_CLEANUP_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=_CODEX_MODEL_PROBE_CLEANUP_TIMEOUT_SECONDS)
        raise timeout_error
    finally:
        if windows_job is not None:
            windows_job.close()
    return subprocess.CompletedProcess(command, process.wait(), stdout, stderr)


def _run_codex_command(command: list[str]) -> int:
    try:
        completed = subprocess.run(command, check=False)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "codex CLI is not installed locally or on PATH. Run `npm ci --prefix "
            f"agentic/packages/content_workflow_cli` before using {CLI_NAME} auth helpers."
        ) from exc
    return int(completed.returncode)


def _codex_executable() -> str:
    package_root = Path(__file__).resolve().parent.parent
    executable = "codex.cmd" if sys.platform == "win32" else "codex"
    local_codex = package_root / "node_modules" / ".bin" / executable
    if local_codex.exists():
        return str(local_codex)
    return "codex"


def _handle_trace_build(args: argparse.Namespace) -> int:
    result = build_trace(args.run_dir.expanduser().resolve())
    print(f"operation_trace_json: {result['operation_trace_json']}")
    print(f"operation_trace_md: {result['operation_trace_md']}")
    print(f"run_retrospective_json: {result['run_retrospective_json']}")
    print(f"replay_manifest_json: {result['replay_manifest_json']}")
    return 0


def _load_codex_config(args: argparse.Namespace) -> dict[str, object] | None:
    config: dict[str, object] = {}
    for path in args.codex_config_file:
        _merge_json_object(config, _read_json_object(path, config_name="Codex config"))
    for text in args.codex_config_json:
        _merge_json_object(
            config,
            _parse_json_object(
                text,
                "--codex-config-json",
                config_name="Codex config",
            ),
        )
    return config or None


def _load_claude_config(args: argparse.Namespace) -> dict[str, object] | None:
    config: dict[str, object] = {}
    for path in args.claude_config_file:
        _merge_json_object(config, _read_json_object(path, config_name="Claude config"))
    for text in args.claude_config_json:
        _merge_json_object(
            config,
            _parse_json_object(
                text,
                "--claude-config-json",
                config_name="Claude config",
            ),
        )
    _validate_claude_config(config)
    return config or None


def _validate_claude_config(config: dict[str, object]) -> None:
    unsupported = sorted(set(config) - SUPPORTED_CLAUDE_CONFIG_KEYS)
    if unsupported:
        supported = ", ".join(sorted(SUPPORTED_CLAUDE_CONFIG_KEYS))
        rejected = ", ".join(unsupported)
        raise ValueError(
            "Claude config supports only these top-level keys: "
            f"{supported}. Unsupported keys: {rejected}"
        )
    settings = config.get("settings")
    if settings is not None and not isinstance(settings, dict):
        raise ValueError("Claude config settings must be a JSON object.")


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a floating point number.") from exc


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise SystemExit(f"{name} must be an integer.") from exc


def _positive_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected an integer") from exc
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def _non_negative_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected an integer") from exc
    if value < 0:
        raise argparse.ArgumentTypeError("must be at least 0")
    return value


def _vomp_seed(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected an integer") from exc
    if not 0 <= value <= 2**32 - 1:
        raise argparse.ArgumentTypeError("must be between 0 and 4294967295")
    return value


def _positive_float(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a number") from exc
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return value


def _non_negative_float(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a number") from exc
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("must be at least 0")
    return value


def _read_json_object(
    path: Path,
    *,
    config_name: str,
    allow_null: bool = False,
) -> dict[str, object]:
    return _parse_json_object(
        path.expanduser().read_text(encoding="utf-8"),
        str(path),
        config_name=config_name,
        allow_null=allow_null,
    )


def _validate_validation_policy_nulls(
    value: object,
    *,
    source: str,
    path: tuple[str | int, ...] = (),
) -> None:
    if value is None:
        allowed_renderer_identity = (
            len(path) == 5
            and path[0] == "qualified_render_evidence"
            and isinstance(path[1], int)
            and path[2] == "ovrtx_render_metadata"
            and path[3] == "renderer_identities"
            and isinstance(path[4], int)
        )
        if not allowed_renderer_identity:
            raise ValueError(
                f"{source} contains null outside a qualified OVRTX renderer identity."
            )
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_validation_policy_nulls(
                item,
                source=source,
                path=(*path, index),
            )
    elif isinstance(value, dict):
        for key, item in value.items():
            _validate_validation_policy_nulls(
                item,
                source=source,
                path=(*path, key),
            )


def _parse_json_object(
    text: str,
    source: str,
    *,
    config_name: str = "agent config",
    allow_null: bool = False,
) -> dict[str, object]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{source} must contain a valid JSON object: {exc}") from exc
    if value is None:
        raise ValueError(
            f"{source} contains null, which {config_name} does not accept."
        )
    if not isinstance(value, dict):
        raise ValueError(f"{source} must contain a JSON object.")
    _validate_config_value(
        value,
        source,
        config_name=config_name,
        allow_null=allow_null,
    )
    return value


MAX_AGENT_CONFIG_DEPTH = 32


def _validate_config_value(
    value: object,
    source: str,
    *,
    config_name: str,
    depth: int = 0,
    allow_null: bool = False,
) -> None:
    if depth > MAX_AGENT_CONFIG_DEPTH:
        raise ValueError(
            f"{source} exceeds the maximum {config_name} nesting depth "
            f"of {MAX_AGENT_CONFIG_DEPTH}."
        )
    if value is None:
        if allow_null:
            return
        raise ValueError(
            f"{source} contains null, which {config_name} does not accept."
        )
    if isinstance(value, str | bool | int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{source} contains a non-finite number.")
        return
    if isinstance(value, list):
        for item in value:
            _validate_config_value(
                item,
                source,
                config_name=config_name,
                depth=depth + 1,
                allow_null=allow_null,
            )
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{source} contains a non-string or empty config key.")
            _validate_config_value(
                item,
                source,
                config_name=config_name,
                depth=depth + 1,
                allow_null=allow_null,
            )
        return
    raise ValueError(
        f"{source} contains unsupported value type: {type(value).__name__}"
    )


def _merge_json_object(
    target: dict[str, object], update: dict[str, object]
) -> dict[str, object]:
    for key, value in update.items():
        existing = target.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            _merge_json_object(existing, value)
        else:
            target[key] = value
    return target
