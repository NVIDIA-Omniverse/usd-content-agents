# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Command-line interface for content-workflow-cli."""

from __future__ import annotations

import argparse
import ipaddress
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

from .runner import (
    CLAUDE_EXECUTION_CLI,
    CLAUDE_EXECUTION_SDK,
    CODEX_SANDBOX_WORKSPACE_WRITE,
    DEFAULT_MATERIAL_RESTORE_TIMEOUT_SECONDS,
    DEFAULT_PROMPT_MODE,
    DEFAULT_VQA_REFINEMENT_MAX_ITERATIONS,
    PROMPT_MODE_LEGACY_EXPANDED,
    PROMPT_MODE_SKILL_ROUTED,
    RUNNER_CLAUDE,
    RUNNER_CODEX,
    MaterialAssignConfig,
    PhysicsApplyConfig,
    find_repo_root,
    run_material_assignment,
    run_physics_apply,
)
from .scene_runner import SceneRunConfig, resume_scene_workflow, run_scene_workflow
from .trace import build_trace

SUPPORTED_CLAUDE_CONFIG_KEYS = frozenset({"env", "maxBudgetUsd", "settings"})
CLI_NAME = "content-workflow-cli"
CONVERT_TO_USD_OUTPUT_FORMATS = ("usd", "usda", "usdc", "usdz")
IMAGE_REFERENCE_SUFFIXES = frozenset(
    {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
)
REFERENCE_DIRECTORY_SUFFIXES = IMAGE_REFERENCE_SUFFIXES | frozenset(
    {".doc", ".docx", ".md", ".pdf", ".txt"}
)


def main(argv: list[str] | None = None) -> int:
    try:
        parser = build_parser()
        args = parser.parse_args(argv)
        if not hasattr(args, "handler"):
            parser.print_help()
            return 1
        return int(args.handler(args))
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"{CLI_NAME}: error: {exc}", file=sys.stderr)
        return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=CLI_NAME,
        description="Run agentic asset workflows against Content Workbench.",
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

    simready = subparsers.add_parser(
        "simready",
        help="SimReady Foundation profile conformance and validation workflows.",
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
    auth_status.set_defaults(handler=_handle_auth_status)

    materials = subparsers.add_parser(
        "materials", help="Material-assignment workflows."
    )
    materials_subparsers = materials.add_subparsers(dest="materials_command")
    for name, help_text in (
        (
            "assign",
            "Assign material-library materials to a USD asset through Content Workbench.",
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

    physics = subparsers.add_parser("physics", help="Physics authoring workflows.")
    physics_subparsers = physics.add_subparsers(dest="physics_command")
    physics_apply = physics_subparsers.add_parser(
        "apply",
        help="Infer and apply USD physics schemas, then optionally validate by simulation.",
    )
    _add_physics_apply_args(physics_apply)
    physics_apply.set_defaults(handler=_handle_physics_apply)

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
        help="Run directory. Defaults to agentic/runs/<scene>-<timestamp>.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root. The child agent launches from <repo-root>/agentic.",
    )
    parser.add_argument(
        "--workbench-url",
        default=os.getenv("CONTENT_WORKBENCH_URL", "http://127.0.0.1:8088"),
        help="Content Workbench endpoint.",
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
        default=_default_codex_sandbox_mode(),
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
        "--start-workbench",
        dest="start_workbench",
        action="store_true",
        default=None,
        help="Start a local Workbench sidecar for loopback URLs when needed.",
    )
    parser.add_argument(
        "--no-start-workbench",
        dest="start_workbench",
        action="store_false",
        help="Require an already-running Workbench endpoint.",
    )
    parser.add_argument(
        "--keep-workbench",
        action="store_true",
        help="Leave a wrapper-started Workbench process running after the run.",
    )
    parser.add_argument(
        "--workbench-timeout",
        type=float,
        default=60.0,
        help="Seconds to wait for Workbench to become healthy.",
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
        "--repair", dest="repair_requirements", action="append", default=[]
    )
    parser.add_argument("--force", action="store_true")
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
        type=_model_reasoning_effort,
        default=None,
        help=(
            "Optional provider/model-specific reasoning effort forwarded without "
            "wrapper enum validation. `max` is rejected; use `xhigh` for Codex."
        ),
    )


def _model_reasoning_effort(value: str) -> str:
    if value == "max":
        raise argparse.ArgumentTypeError(
            "unsupported model reasoning effort 'max'; use 'xhigh'"
        )
    return value


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
        "--workbench-url",
        default=os.getenv("CONTENT_WORKBENCH_URL", "http://127.0.0.1:8088"),
        help="Content Workbench endpoint.",
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
            "The source asset is not overwritten."
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
        "--codex-sandbox-mode",
        choices=[CODEX_SANDBOX_WORKSPACE_WRITE],
        default=_default_codex_sandbox_mode(),
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
        "--start-workbench",
        dest="start_workbench",
        action="store_true",
        default=None,
        help=(
            "Start a local host Workbench sidecar if the endpoint is not already "
            "healthy. Defaults to true for localhost URLs."
        ),
    )
    parser.add_argument(
        "--no-start-workbench",
        dest="start_workbench",
        action="store_false",
        help="Do not start a Workbench sidecar automatically.",
    )
    parser.add_argument(
        "--optimize",
        dest="optimize",
        action="store_true",
        default=True,
        help=(
            "Ask Content Workbench to run Scene Optimizer before inspection in "
            "fixed selection mode."
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
            "Skip invisible candidates. Workbench visible hints already apply "
            "effective visibility."
        ),
    )
    parser.add_argument(
        "--include-invisible",
        dest="skip_invisible",
        action="store_false",
        help="Do not add extra invisible filtering beyond Workbench visible hints.",
    )
    parser.add_argument(
        "--flatten-prototypes",
        dest="flatten_prototypes",
        action="store_true",
        default=None,
        help="Pass flatten_prototypes=true to Workbench Scene Optimizer.",
    )
    parser.add_argument(
        "--no-flatten-prototypes",
        dest="flatten_prototypes",
        action="store_false",
        help="Pass flatten_prototypes=false to Workbench Scene Optimizer.",
    )
    parser.add_argument(
        "--enable-deinstance",
        dest="enable_deinstance",
        action="store_true",
        default=None,
        help="Pass enable_deinstance=true to Workbench Scene Optimizer.",
    )
    parser.add_argument(
        "--disable-deinstance",
        dest="enable_deinstance",
        action="store_false",
        help="Pass enable_deinstance=false to Workbench Scene Optimizer.",
    )
    parser.add_argument(
        "--enable-split",
        dest="enable_split",
        action="store_true",
        default=None,
        help="Pass enable_split=true to Workbench Scene Optimizer.",
    )
    parser.add_argument(
        "--disable-split",
        dest="enable_split",
        action="store_false",
        help="Pass enable_split=false to Workbench Scene Optimizer.",
    )
    parser.add_argument(
        "--enable-deduplicate",
        dest="enable_deduplicate",
        action="store_true",
        default=None,
        help="Pass enable_deduplicate=true to Workbench Scene Optimizer.",
    )
    parser.add_argument(
        "--disable-deduplicate",
        dest="enable_deduplicate",
        action="store_false",
        help="Pass enable_deduplicate=false to Workbench Scene Optimizer.",
    )
    parser.add_argument(
        "--preflight",
        dest="preflight",
        action="store_true",
        default=True,
        help=(
            "Prepare a Workbench material-run packet before launching the child "
            "agent. This is the default for real runs."
        ),
    )
    parser.add_argument(
        "--no-preflight",
        dest="preflight",
        action="store_false",
        help="Let the child agent perform Workbench setup and initial inspection.",
    )
    parser.add_argument(
        "--respect-existing-material-bindings",
        "--respect-existing-materials",
        dest="respect_existing_material_bindings",
        action="store_true",
        default=False,
        help=(
            "Use existing material bindings as preserved seed decisions. By "
            "default the Workbench session clears existing material bindings "
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
        "--keep-workbench",
        action="store_true",
        help="Leave a wrapper-started Content Workbench process running after the run.",
    )
    parser.add_argument(
        "--workbench-timeout",
        type=float,
        default=60.0,
        help="Seconds to wait for the configured Workbench endpoint to become healthy.",
    )
    parser.add_argument(
        "--material-restore-timeout",
        type=float,
        default=DEFAULT_MATERIAL_RESTORE_TIMEOUT_SECONDS,
        help=(
            "Seconds to wait for final material application and USD restoration. "
            "Defaults to the Workbench material operation limit."
        ),
    )
    parser.add_argument(
        "--child-timeout",
        type=float,
        default=_env_float("CONTENT_AGENT_CHILD_TIMEOUT", 1800.0),
        help=(
            "Seconds to wait for the Codex SDK child turn. Use 0 to disable "
            "the timeout; with --no-start-workbench there is no managed "
            "Workbench watchdog, so this can wait indefinitely."
        ),
    )
    parser.add_argument(
        "--prompt-mode",
        choices=[PROMPT_MODE_LEGACY_EXPANDED, PROMPT_MODE_SKILL_ROUTED],
        default=DEFAULT_PROMPT_MODE,
        help=(
            "Child prompt shape. `legacy-expanded` embeds the full workflow "
            "contract for compatibility; `skill-routed` passes compact structured "
            "inputs and asks the child agent to use preview skills."
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
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for canonical physics workflow artifacts.",
    )
    parser.add_argument(
        "--output-usd",
        type=Path,
        default=None,
        help="Authored USD/USDZ output path. Defaults to <output-dir>/physics.usda.",
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
        help="Apply physics without running runtime validation.",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=1.0,
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
        "--workbench-url",
        default=None,
        help="Optional Content Workbench endpoint for physics operations.",
    )
    parser.add_argument(
        "--workbench-session-id",
        default=None,
        help="Optional existing Workbench session ID for physics operations.",
    )
    parser.add_argument(
        "--workbench-timeout",
        type=float,
        default=300.0,
        help="Timeout in seconds for Workbench physics operation requests.",
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
        default=_default_codex_sandbox_mode(),
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
        "--start-workbench",
        dest="start_workbench",
        action="store_true",
        default=None,
        help="Start a local Workbench sidecar for loopback URLs when needed.",
    )
    parser.add_argument(
        "--no-start-workbench",
        dest="start_workbench",
        action="store_false",
        help="Do not start a Workbench sidecar automatically.",
    )
    parser.add_argument(
        "--optimize",
        dest="optimize",
        action="store_true",
        default=True,
        help="Ask Content Workbench to run Scene Optimizer before inspection.",
    )
    parser.add_argument(
        "--no-optimize",
        dest="optimize",
        action="store_false",
        help="Load the source USD directly without Workbench Scene Optimizer.",
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
        "--keep-workbench",
        action="store_true",
        help="Leave a wrapper-started Content Workbench process running after the run.",
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
        choices=[PROMPT_MODE_LEGACY_EXPANDED, PROMPT_MODE_SKILL_ROUTED],
        default=DEFAULT_PROMPT_MODE,
        help="Child prompt shape.",
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
        "--deterministic-workflow",
        action="store_true",
        help="Use the lower-level deterministic physics workflow without child-agent visual review.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full result JSON.",
    )


def _default_codex_sandbox_mode() -> str:
    value = os.getenv("CONTENT_AGENT_CODEX_SANDBOX_MODE")
    if value in {None, ""}:
        return CODEX_SANDBOX_WORKSPACE_WRITE
    allowed = {CODEX_SANDBOX_WORKSPACE_WRITE}
    if value not in allowed:
        supported = ", ".join(sorted(allowed))
        raise ValueError(
            "Invalid CONTENT_AGENT_CODEX_SANDBOX_MODE: "
            f"{value!r}. Expected one of: {supported}"
        )
    return value


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

    if args.output_dir is not None:
        result = run_convert_to_usd_workflow(
            ConvertToUsdWorkflowInput(
                source_asset_path=source_asset,
                output_dir=args.output_dir.expanduser().resolve(),
                output_usd_path=output_usd,
                output_format=args.output_format,
                install_missing=args.install_missing,
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
            if result.error:
                print(f"error: {result.error}", file=sys.stderr)
        return 0 if result.success else 1

    report, _probe_artifact = convert_source_to_usd_file(
        source_asset,
        output_usd,
        output_format=args.output_format,
        install_missing=args.install_missing,
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
            repair_requirements=args.repair_requirements,
            force=args.force,
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
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


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
        workbench_url=args.workbench_url.rstrip("/"),
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
        start_workbench=_should_start_workbench(args),
        keep_workbench=args.keep_workbench,
        workbench_timeout_seconds=args.workbench_timeout,
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
    )
    _print_scene_result(result)
    return result.returncode


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
        workbench_url=args.workbench_url.rstrip("/"),
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
        start_workbench=_should_start_workbench(args),
        keep_workbench=args.keep_workbench,
        workbench_timeout_seconds=args.workbench_timeout,
        material_restore_timeout_seconds=args.material_restore_timeout,
        child_timeout_seconds=args.child_timeout,
        prompt_mode=args.prompt_mode,
        vqa_refinement_max_iterations=args.vqa_refinement_max_iterations,
        codex_persistent_refinement=args.codex_persistent_refinement,
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


def _handle_physics_apply(args: argparse.Namespace) -> int:
    if not args.deterministic_workflow:
        if args.workbench_session_id is not None:
            print(
                "--workbench-session-id is only supported with "
                "--deterministic-workflow.",
                file=sys.stderr,
            )
            return 2
        repo_root = args.repo_root.resolve() if args.repo_root else find_repo_root()
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
        workbench_url = (
            args.workbench_url
            or os.getenv("CONTENT_WORKBENCH_URL")
            or "http://127.0.0.1:8088"
        ).rstrip("/")
        start_workbench_args = argparse.Namespace(
            **{**vars(args), "workbench_url": workbench_url}
        )
        result = run_physics_apply(
            PhysicsApplyConfig(
                repo_root=repo_root,
                usd_path=args.usd.expanduser().resolve(),
                reference_images=reference_images,
                reference_files=reference_files,
                workbench_url=workbench_url,
                output_dir=args.output_dir.expanduser(),
                output_usd_path=args.output_usd.expanduser().resolve()
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
                start_workbench=_should_start_workbench(start_workbench_args),
                keep_workbench=args.keep_workbench,
                workbench_timeout_seconds=args.workbench_timeout,
                child_timeout_seconds=args.child_timeout,
                prompt_mode=args.prompt_mode,
                vqa_refinement_max_iterations=args.vqa_refinement_max_iterations,
                codex_persistent_refinement=args.codex_persistent_refinement,
                additional_instructions=additional_instructions,
                log_to_stderr=(
                    args.log_to_stderr if args.log_to_stderr is not None else True
                ),
            )
        )
        if args.json:
            print(
                json.dumps(
                    {
                        "run_dir": str(result.run_dir),
                        "prompt_path": str(result.prompt_path),
                        "request_path": str(result.request_path),
                        "child_output_path": str(result.child_output_path),
                        "child_final_path": str(result.child_final_path),
                        "returncode": result.returncode,
                        "trace_paths": result.trace_paths,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            print(f"Run directory: {result.run_dir}")
            print(f"Prompt: {result.prompt_path}")
            print(f"Request: {result.request_path}")
            print(f"Child output: {result.child_output_path}")
            print(f"Child final: {result.child_final_path}")
            for label, path in result.trace_paths.items():
                print(f"{label}: {path}")
        return result.returncode

    from content_agent_workflows.physics import (
        PhysicsApplyWorkflowInput,
        run_physics_apply_workflow,
    )

    result = run_physics_apply_workflow(
        PhysicsApplyWorkflowInput(
            usd_path=args.usd.expanduser().resolve(),
            output_dir=args.output_dir.expanduser().resolve(),
            output_usd_path=args.output_usd.expanduser().resolve()
            if args.output_usd is not None
            else None,
            collision_approximation=args.collision_approximation,
            run_simulation=not args.no_simulation and args.simulation_engine != "none",
            simulation_engine=args.simulation_engine,
            simulation_duration_s=args.duration_s,
            simulation_dt=args.dt,
            simulation_sample_fps=args.sample_fps,
            drop_height_m=args.drop_height_m,
            fail_on_validation_error=args.fail_on_validation_error,
            workbench_url=args.workbench_url.rstrip("/")
            if args.workbench_url is not None
            else None,
            workbench_session_id=args.workbench_session_id,
            workbench_timeout_s=args.workbench_timeout,
        )
    )
    payload = result.model_dump(mode="json")
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        status = "ok" if result.success else "failed"
        print(f"physics apply {status}: {result.validation_status}")
        if result.physics_usd_path:
            print(f"physics_usd: {result.physics_usd_path}")
        if result.assignments_path:
            print(f"assignments: {result.assignments_path}")
        if result.validation_evidence_path:
            print(f"validation_evidence: {result.validation_evidence_path}")
        if result.error:
            print(f"error: {result.error}", file=sys.stderr)
    return 0 if result.success else 1


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


def _resolve_materials_usd_from_manifest(materials_yaml: Path) -> Path:
    manifest = _load_yaml_manifest(materials_yaml)
    library_path = manifest.get("library_path")
    if not isinstance(library_path, str) or not library_path.strip():
        raise ValueError(
            f"{materials_yaml} does not define a non-empty top-level library_path. "
            "Pass --materials-usd to override it."
        )
    resolved = Path(library_path).expanduser()
    if not resolved.is_absolute():
        resolved = materials_yaml.parent / resolved
    return resolved.resolve()


def _load_yaml_manifest(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(
            f"Failed to parse material YAML manifest {path}: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise ValueError(f"Material YAML manifest {path} must be a mapping.")
    return data


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
    return _probe_codex_model_access(executable)


def _probe_codex_model_access(executable: str) -> int:
    """Verify that the active login can complete a minimal model request."""

    with tempfile.TemporaryDirectory(prefix="content-workflow-codex-auth-") as cwd:
        command = [
            executable,
            "exec",
            "--ephemeral",
            "--ignore-user-config",
            "--skip-git-repo-check",
            "--sandbox",
            "read-only",
            "--color",
            "never",
            "--cd",
            cwd,
            "Reply with exactly OK. Do not use tools.",
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
                "error: Codex reports a login, but its model usability probe timed out.",
                file=sys.stderr,
            )
            return 1

    if completed.returncode == 0:
        print("Codex login is usable for model calls.")
        return 0

    print(
        "error: Codex reports a login, but it cannot complete a model call.",
        file=sys.stderr,
    )
    details = (completed.stderr or completed.stdout or "").strip()
    if details:
        print(details, file=sys.stderr)
    return int(completed.returncode) or 1


def _run_codex_model_probe(
    command: list[str],
) -> subprocess.CompletedProcess[str]:
    """Run an isolated Codex probe and clean up its full process group."""

    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        start_new_session=hasattr(os, "setsid"),
    )
    try:
        stdout, stderr = process.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        if hasattr(os, "killpg"):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        elif process.poll() is None:  # pragma: no cover - non-POSIX fallback
            process.kill()
        process.communicate()
        raise
    return subprocess.CompletedProcess(
        command,
        process.wait(),
        stdout,
        stderr,
    )


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


def _should_start_workbench(args: argparse.Namespace) -> bool:
    if args.start_workbench is not None:
        return bool(args.start_workbench)
    parsed = urlparse(str(args.workbench_url))
    hostname = parsed.hostname
    if hostname == "localhost":
        return True
    if hostname is None:
        return False
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a floating point number.") from exc


def _positive_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected an integer") from exc
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def _read_json_object(path: Path, *, config_name: str) -> dict[str, object]:
    return _parse_json_object(
        path.expanduser().read_text(encoding="utf-8"),
        str(path),
        config_name=config_name,
    )


def _parse_json_object(
    text: str, source: str, *, config_name: str = "agent config"
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
    _validate_config_value(value, source, config_name=config_name)
    return value


MAX_AGENT_CONFIG_DEPTH = 32


def _validate_config_value(
    value: object,
    source: str,
    *,
    config_name: str,
    depth: int = 0,
) -> None:
    if depth > MAX_AGENT_CONFIG_DEPTH:
        raise ValueError(
            f"{source} exceeds the maximum {config_name} nesting depth "
            f"of {MAX_AGENT_CONFIG_DEPTH}."
        )
    if value is None:
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
