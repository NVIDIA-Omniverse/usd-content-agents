# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Thin CLI launcher for the reusable durable Texture workflow."""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import math
import os
import shutil
import signal
import stat
import sys
import threading
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from types import FrameType
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast
from urllib.parse import urlparse

from content_agent_workflows.asset_composition import (
    build_embedded_domain_decision_identity,
    build_embedded_domain_execution_context,
)
from content_agent_workflows.common.artifacts import (
    atomic_write_json,
    atomic_write_text,
    file_sha256,
    load_json,
)
from content_agent_workflows.common.domain_execution import (
    metadata_with_domain_execution_context,
)
from world_understanding.utils.artifacts import (
    ArtifactPathError,
    open_regular_file_no_follow,
)

from .child_launch import (
    TEXTURE_CHILD_LAUNCH_PROFILE,
    ChildLaunchArtifactIdentity,
    child_launch_artifact_identity,
)
from .runner import (
    CLAUDE_EXECUTION_CLI,
    CLAUDE_EXECUTION_SDK,
    CODEX_SANDBOX_WORKSPACE_WRITE,
    RUNNER_CLAUDE,
    RUNNER_CODEX,
    _child_turn_used_tools,
    _lexical_absolute_path,
    _skill_routed_run_lock,
    find_repo_root,
    parent_usd_cli_prompt_contract,
    run_child_agent,
    start_parent_usd_cli_capability,
    stop_parent_usd_cli_capability,
)
from .trace import TraceWriter, build_trace

if TYPE_CHECKING:
    from content_agent_workflows.texture import (
        TextureAgenticExecutionAdapter,
        TextureFinalizationResult,
        TextureWorkflowCancellationToken,
        TextureWorkflowCheckpoint,
        TextureWorkflowProgress,
        TextureWorkflowRequest,
    )
    from content_agent_workflows.texture.models import TextureWorkflowMode

DEFAULT_VLM_TIMEOUT_SECONDS = 120.0
TEXTURE_VALIDATION_POLICY_VERSION = "content-workflow-cli.texture-vqa.v1"
TEXTURE_VALIDATION_POLICY_METADATA_KEY = "content_workflow_cli_validation_policy_id"
TEXTURE_EXECUTION_MODE_METADATA_KEY = "content_workflow_cli_execution_mode"
TEXTURE_CONDITIONAL_EXIT_CODE = 3
TEXTURE_CANCELLED_EXIT_CODE = 130
PUBLIC_VLM_BACKENDS = ("nim", "openai", "anthropic", "gemini")
TEXTURE_AGENT_LAUNCHER_SCHEMA_VERSION = "content-workflow-cli.texture-agent-launcher.v1"
TEXTURE_AGENT_LAUNCHER_POLICY_SCHEMA_VERSION = (
    "content-workflow-cli.texture-agent-launcher-policy.v1"
)
MAX_TEXTURE_AGENT_LAUNCHER_BYTES = 1024 * 1024
MAX_TEXTURE_AGENT_LAUNCHER_POLICY_BYTES = 16 * 1024
TEXTURE_EXECUTION_SKILL_ROUTED = "skill-routed"
TEXTURE_EXECUTION_FIXED = "fixed"
TEXTURE_AGENTIC_PLAN_METADATA_KEY = "content_workflow_cli_texture_plan_schema"
TEXTURE_AGENTIC_PLAN_SCHEMA_VERSION = "content-agent-workflows.texture-plan.v2"
TEXTURE_AGENTIC_PLAN_OUTPUT_CONTRACT_SCHEMA_VERSION = (
    "content-workflow-cli.texture-plan-output-contract.v1"
)
TEXTURE_AGENTIC_PLAN_OUTPUT_CONTRACT_NAME = "texture_plan_output_contract.json"
TEXTURE_AGENTIC_PROVIDED_CANDIDATES_METADATA_KEY = (
    "content_workflow_cli_texture_agentic_provided_candidates"
)
TEXTURE_AGENTIC_ACTION_POLICY_METADATA_KEY = (
    "content_workflow_cli_texture_agentic_action_policy"
)
TEXTURE_AGENTIC_APPEARANCE_POLICY_METADATA_KEY = (
    "content_workflow_cli_texture_agentic_appearance_policy"
)
TEXTURE_AGENTIC_EVIDENCE_VIEWS_METADATA_KEY = (
    "content_workflow_cli_texture_agentic_evidence_views"
)
TEXTURE_AGENTIC_UV_POLICY_METADATA_KEY = (
    "content_workflow_cli_texture_agentic_uv_policy"
)
TEXTURE_AGENTIC_SOURCE_PREPARATION_METADATA_KEY = (
    "content_workflow_cli_texture_agentic_source_preparation"
)
TEXTURE_COMPANION_IMAGE_BACKEND = "coding_agent_companion"
TEXTURE_COMPANION_GENERATION_HANDOFF_SCHEMA_VERSION = (
    "content-workflow-cli.texture-companion-generation-handoff.v1"
)


@dataclass(frozen=True)
class TextureRuntimeConfig:
    """Runtime-only endpoint and validation settings.

    Credentials are resolved from the named environment variables and are never
    copied into the durable workflow request.
    """

    texture_agent_url: str | None
    texture_agent_token_env: str | None
    texture_timeout_seconds: float
    texture_poll_interval_seconds: float
    vlm_backend: str | None
    vlm_model: str | None
    vlm_base_url: str | None
    vlm_api_key_env: str | None
    vlm_timeout_seconds: float
    json_output: bool
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
    execution_mode: str = TEXTURE_EXECUTION_SKILL_ROUTED


@dataclass(frozen=True)
class _TextureChildRuntimeConfig:
    """Adapter from Texture runtime settings to the shared child launcher."""

    child_launch_profile: ClassVar[str] = TEXTURE_CHILD_LAUNCH_PROFILE
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
    workflow_skill: ClassVar[str] = "content-workflow-texture"


class _LazyVlm:
    """Delay credential resolution until VQA actually needs the model.

    Resume identity checks therefore fail before contacting a model provider or
    requiring credentials when the persisted request/source has been changed.
    """

    def __init__(
        self,
        *,
        backend: str,
        model: str,
        base_url: str | None,
        api_key_env: str | None,
        timeout_seconds: float,
    ) -> None:
        self._config = {
            "backend": backend,
            "model": model,
            "base_url": base_url,
            "api_key_env": api_key_env,
            "timeout": timeout_seconds,
        }
        self._vlm: Any | None = None

    def generate(self, **kwargs: Any) -> str:
        if self._vlm is None:
            config = {
                key: value for key, value in self._config.items() if value is not None
            }
            backend = str(config.pop("backend"))
            api_key = _resolve_vlm_api_key(backend, self._config, "VLM")
            config.pop("api_key_env", None)
            if api_key:
                config["api_key"] = api_key
            self._vlm = _create_vlm(backend=backend, **config)
        return str(self._vlm.generate(**kwargs))


def _resolve_vlm_api_key(
    backend: str,
    config: dict[str, Any],
    label: str,
) -> str | None:
    from world_understanding.agentic.config import get_api_key_for_model_config

    return get_api_key_for_model_config(backend, config, label)


def _create_vlm(*, backend: str, **config: Any) -> Any:
    from world_understanding.functions.models.vision_language_models import create_vlm

    return create_vlm(backend=backend, **config)


class _UnavailableRuntimeAdapter:
    """Fail loudly if a terminal-only resume unexpectedly reaches a runtime."""

    def __getattr__(self, name: str) -> Any:
        raise RuntimeError(
            f"Terminal Texture resume unexpectedly requested runtime method {name!r}"
        )


def add_texture_subcommands(subparsers: Any) -> None:
    """Register the public ``texture run`` and ``texture resume`` commands."""

    texture = subparsers.add_parser(
        "texture",
        help="Run the bounded prompt-to-textured-USD workflow.",
    )
    texture_subparsers = texture.add_subparsers(
        dest="texture_command",
        required=True,
        metavar=(
            "{agentic-leaf,prepare,propose,generate,apply-provided,evidence,"
            "critique,review,publish,run,resume}"
        ),
    )

    from .texture_capability_runner import add_texture_capability_subcommands

    add_texture_capability_subcommands(texture_subparsers)

    texture_run = texture_subparsers.add_parser(
        "run",
        help="Texture an explicitly selected USD material or prim scope.",
    )
    texture_run.add_argument("--usd", required=True, type=Path, help="Input USD asset.")
    texture_run.add_argument(
        "--prompt",
        required=True,
        help="Natural-language appearance request for the selected scope.",
    )
    texture_run.add_argument(
        "--material-path",
        action="append",
        default=[],
        help=(
            "Exact material prim path to texture. Exactly one of --material-path "
            "or --prim-path is required; this option may be repeated."
        ),
    )
    texture_run.add_argument(
        "--prim-path",
        action="append",
        default=[],
        help=(
            "Exact renderable prim path to texture. Exactly one of --material-path "
            "or --prim-path is required; this option may be repeated."
        ),
    )
    texture_run.add_argument(
        "--provided-image",
        action="append",
        default=[],
        metavar="TARGET=PNG",
        help=(
            "Bind one exact outer-provided albedo PNG to a selected material or "
            "prim target. May be repeated; only the accepted plan may select it."
        ),
    )
    texture_run.add_argument(
        "--reference-image",
        action="append",
        default=[],
        metavar="ROLE=PATH",
        help=(
            "Bind one exact appearance reference image under a unique semantic "
            "role. May be repeated; references remain evidence inputs and never "
            "become provided candidates."
        ),
    )
    texture_run.add_argument(
        "--unit-action",
        action="append",
        default=[],
        metavar="TARGET=ACTION",
        help=(
            "Constrain one exact selected target to preserve, generate, "
            "apply_provided, defer, or reject. May be repeated."
        ),
    )
    texture_run.add_argument(
        "--unit-appearance",
        action="append",
        default=[],
        metavar="TARGET=TEXT",
        help=(
            "Bind one exact requested appearance to a selected generate or "
            "apply_provided target. May be repeated."
        ),
    )
    texture_run.add_argument(
        "--evidence-view",
        action="append",
        default=[],
        choices=("+x-y+z", "+z", "-z"),
        help="Require one exact current-run OVRTX direction. May be repeated.",
    )
    texture_run.add_argument(
        "--uv-policy",
        choices=("inspect", "generate_missing"),
        default="inspect",
        help=(
            "Provider-free pre-child UV policy. generate_missing requires an "
            "explicit prim scope and freezes original/prepared source identities."
        ),
    )
    texture_run.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="New durable run directory.",
    )
    texture_run.add_argument(
        "--max-vqa-iterations",
        type=_non_negative_int,
        default=None,
        help=(
            "Maximum unit-specific VQA refinement retries. Standalone "
            "skill-routed Texture currently supports only 0 (one reviewed "
            "attempt); embedded and fixed modes default to 2."
        ),
    )
    texture_run.add_argument(
        "--texture-backend",
        default=None,
        help="Optional Texture Agent generation backend override.",
    )
    texture_run.add_argument(
        "--texture-endpoint",
        default=None,
        help="Optional Texture Agent generation endpoint override.",
    )
    texture_run.add_argument(
        "--backend-engine",
        default=None,
        help="Optional Texture Agent backend engine/model override.",
    )
    texture_run.add_argument(
        "--embedded-run-state",
        type=Path,
        default=None,
        help=(
            "Bind this execution to the exact active Texture attempt in a composed "
            "asset_run.json. This option is valid only for texture run; resume uses "
            "the context frozen in request.json."
        ),
    )
    _add_texture_runtime_args(texture_run)
    texture_run.set_defaults(handler=_handle_texture_run)

    texture_resume = texture_subparsers.add_parser(
        "resume",
        help="Resume the exact request stored in a Texture run directory.",
    )
    texture_resume.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help="Existing Texture workflow run directory.",
    )
    _add_texture_runtime_args(texture_resume)
    texture_resume.set_defaults(handler=_handle_texture_resume)

    texture_agent_step = texture_subparsers.add_parser(
        "_agent-step",
        help=argparse.SUPPRESS,
    )
    texture_agent_step.add_argument("--run-dir", required=True, type=Path)
    texture_agent_step.add_argument("--decision-patch", type=Path, default=None)
    texture_agent_step.set_defaults(handler=_handle_texture_agent_step)


def _add_texture_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--texture-agent-url",
        default=os.getenv("CONTENT_TEXTURE_AGENT_URL"),
        help=(
            "Optional Texture Agent service endpoint. It is required only when "
            "the accepted plan explicitly selects a service generation backend."
        ),
    )
    parser.add_argument(
        "--texture-agent-token-env",
        default=None,
        metavar="ENV_VAR",
        help="Environment variable containing an optional Texture Agent bearer token.",
    )
    parser.add_argument(
        "--texture-timeout",
        type=_positive_float,
        default=1800.0,
        help="Texture Agent request timeout in seconds.",
    )
    parser.add_argument(
        "--texture-poll-interval",
        type=_non_negative_float,
        default=1.0,
        help="Texture Agent status polling interval in seconds.",
    )
    parser.add_argument(
        "--vlm-backend",
        choices=PUBLIC_VLM_BACKENDS,
        default=os.getenv("CONTENT_TEXTURE_VLM_BACKEND"),
        help=(
            "Public VLM provider used for visual validation. Required unless "
            "CONTENT_TEXTURE_VLM_BACKEND is configured."
        ),
    )
    parser.add_argument(
        "--vlm-model",
        default=os.getenv("CONTENT_TEXTURE_VLM_MODEL"),
        help=(
            "VLM model used for visual validation. Required unless "
            "CONTENT_TEXTURE_VLM_MODEL is configured."
        ),
    )
    parser.add_argument(
        "--vlm-base-url",
        default=os.getenv("CONTENT_TEXTURE_VLM_BASE_URL"),
        help="Optional provider-compatible VLM endpoint.",
    )
    parser.add_argument(
        "--vlm-api-key-env",
        default=None,
        metavar="ENV_VAR",
        help=(
            "Environment variable containing the VLM credential. Provider default "
            "credential variables are used when omitted."
        ),
    )
    parser.add_argument(
        "--vlm-timeout",
        type=_positive_float,
        default=DEFAULT_VLM_TIMEOUT_SECONDS,
        help=(
            "VLM assessment request timeout in seconds. This runtime-only "
            "setting does not change validation policy identity."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the terminal Texture result as JSON; progress remains on stderr.",
    )
    parser.add_argument(
        "--runner",
        choices=(RUNNER_CODEX, RUNNER_CLAUDE),
        default=RUNNER_CODEX,
        help="Long-running child-agent runner for standalone execution.",
    )
    parser.add_argument("--model", default=None, help="Optional child-agent model.")
    parser.add_argument(
        "--model-reasoning-effort",
        default=None,
        help="Optional child-agent reasoning effort.",
    )
    parser.add_argument(
        "--codex-base-url",
        default=os.getenv("CONTENT_AGENT_CODEX_BASE_URL"),
        help="Optional OpenAI-compatible Codex SDK base URL.",
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
        help="Seconds to wait for the standalone long-running child; 0 disables.",
    )
    parser.add_argument(
        "--agent-cwd",
        type=Path,
        default=None,
        help="Optional child working directory; defaults to <repo>/agentic.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Freeze the request, run packet, prompt, and trace without launching.",
    )
    parser.add_argument(
        "--execution-mode",
        choices=(TEXTURE_EXECUTION_SKILL_ROUTED, TEXTURE_EXECUTION_FIXED),
        default=TEXTURE_EXECUTION_SKILL_ROUTED,
        help=(
            "Use the skill-routed child controller (default) or the explicit "
            "fixed compatibility baseline."
        ),
    )


def _handle_texture_run(args: argparse.Namespace) -> int:
    from content_agent_workflows.texture import (
        TEXTURE_EMBEDDED_DECISION_IDENTITY_METADATA_KEY,
        TextureReferenceArtifact,
        TextureWorkflowRequest,
        bind_texture_agentic_artifact,
        texture_embedded_capability_digests,
        texture_embedded_implementation_digests,
        texture_request_digest,
    )

    source_asset = args.usd.expanduser().resolve()
    if not source_asset.is_file():
        raise FileNotFoundError(f"Texture source USD does not exist: {source_asset}")
    if source_asset.suffix.lower() not in {".usd", ".usda", ".usdc", ".usdz"}:
        raise ValueError(f"Texture source must be a USD file: {source_asset}")

    material_paths = _normalize_prim_paths(args.material_path, "--material-path")
    prim_paths = _normalize_prim_paths(args.prim_path, "--prim-path")
    if not material_paths and not prim_paths:
        raise ValueError(
            "Texture scope must include at least one --material-path or --prim-path."
        )
    if material_paths and prim_paths:
        raise ValueError(
            "Texture scope cannot mix --material-path and --prim-path; "
            "choose one scope type per run."
        )
    intent = str(args.prompt).strip()
    if not intent:
        raise ValueError("--prompt must not be empty.")

    runtime = _runtime_config(args)
    standalone_skill_routed = (
        runtime.execution_mode == TEXTURE_EXECUTION_SKILL_ROUTED
        and args.embedded_run_state is None
    )
    requested_max_vqa_iterations = args.max_vqa_iterations
    if requested_max_vqa_iterations is None:
        max_vqa_iterations = 0 if standalone_skill_routed else 2
    else:
        max_vqa_iterations = int(requested_max_vqa_iterations)
    if standalone_skill_routed and max_vqa_iterations != 0:
        raise ValueError(
            "--max-vqa-iterations must be 0 in standalone skill-routed Texture "
            "mode; the current standalone agentic executor owns one reviewed "
            "attempt and does not implement refinement retries"
        )
    if runtime.execution_mode == TEXTURE_EXECUTION_FIXED:
        _require_texture_agent_url(runtime)
    elif runtime.texture_agent_url is not None and args.texture_backend is None:
        print(
            "Warning: --texture-agent-url/CONTENT_TEXTURE_AGENT_URL does not "
            "select generation in skill-routed mode by itself; generate actions default to the "
            "coding-agent companion. Set --texture-backend to select a Texture "
            "service backend, or use --execution-mode fixed.",
            file=sys.stderr,
        )
    allowed_actions = {
        "preserve",
        "generate",
        "apply_provided",
        "defer",
        "reject",
    }
    action_policy: dict[str, str] = {}
    appearance_policy: dict[str, str] = {}
    evidence_views = tuple(args.evidence_view)
    if len(evidence_views) != len(set(evidence_views)):
        raise ValueError("--evidence-view values must be unique")
    uv_policy = str(args.uv_policy)
    if uv_policy == "generate_missing" and not prim_paths:
        raise ValueError(
            "--uv-policy generate_missing requires explicit --prim-path scope"
        )
    allowed_candidate_targets = {*material_paths, *prim_paths}
    reference_artifacts: list[TextureReferenceArtifact] = []
    reference_roles: set[str] = set()
    reference_paths: set[str] = set()
    for raw in args.reference_image:
        if "=" not in raw:
            raise ValueError("--reference-image must use ROLE=PATH")
        role, raw_path = (item.strip() for item in raw.split("=", 1))
        if not role or not raw_path:
            raise ValueError("--reference-image must use non-empty ROLE=PATH")
        if role in reference_roles:
            raise ValueError("--reference-image roles must be unique")
        binding = bind_texture_agentic_artifact(Path(raw_path).expanduser())
        if binding.path in reference_paths:
            raise ValueError("--reference-image paths must be unique")
        reference_roles.add(role)
        reference_paths.add(binding.path)
        reference_artifacts.append(
            TextureReferenceArtifact(role=role, artifact=binding)
        )
    for raw in args.unit_action:
        if "=" not in raw:
            raise ValueError("--unit-action must use TARGET=ACTION")
        target, action = (item.strip() for item in raw.split("=", 1))
        if target not in allowed_candidate_targets:
            raise ValueError(
                "--unit-action target must be one exact selected material or prim"
            )
        if action not in allowed_actions:
            raise ValueError("--unit-action names an unsupported Texture action")
        if target in action_policy:
            raise ValueError("--unit-action target must be unique")
        action_policy[target] = action
    for raw in args.unit_appearance:
        if "=" not in raw:
            raise ValueError("--unit-appearance must use TARGET=TEXT")
        target, appearance = (item.strip() for item in raw.split("=", 1))
        if target not in allowed_candidate_targets:
            raise ValueError(
                "--unit-appearance target must be one exact selected material or prim"
            )
        if not appearance:
            raise ValueError("--unit-appearance text must not be empty")
        if target in appearance_policy:
            raise ValueError("--unit-appearance target must be unique")
        required_action = action_policy.get(target)
        if required_action is not None and required_action not in {
            "generate",
            "apply_provided",
        }:
            raise ValueError(
                "--unit-appearance requires generate or apply_provided action"
            )
        appearance_policy[target] = appearance
    provided_candidates: list[dict[str, Any]] = []
    observed_candidate_paths: set[str] = set()
    for raw in args.provided_image:
        if "=" not in raw:
            raise ValueError("--provided-image must use TARGET=PNG")
        target, raw_path = raw.split("=", 1)
        target = target.strip()
        if target not in allowed_candidate_targets:
            raise ValueError(
                "--provided-image target must be one exact selected material or prim"
            )
        if any(item["target_path"] == target for item in provided_candidates):
            raise ValueError("--provided-image target must be unique")
        candidate_path = Path(raw_path).expanduser()
        if candidate_path.suffix.lower() != ".png":
            raise ValueError("--provided-image must name a PNG file")
        binding = bind_texture_agentic_artifact(candidate_path)
        if binding.path in observed_candidate_paths:
            raise ValueError("--provided-image paths must be unique")
        observed_candidate_paths.add(binding.path)
        provided_candidates.append(
            {
                "target_path": target,
                "artifact": binding.model_dump(mode="json"),
                "producer": {
                    "provider": "content-workflow-cli",
                    "capability": "texture.bind-provided-image.v1",
                    "invocation_id": f"provided-{binding.sha256[:20]}",
                    "provenance": {"source": "explicit_cli_argument"},
                },
            }
        )
    output_dir_candidate = _lexical_absolute_path(args.output_dir)
    metadata: dict[str, Any] = {
        # Material targets can consume the user's request directly. Prim-only
        # targets still need the Texture Agent's material-aware prompt step.
        "auto_prompt_enabled": not material_paths,
        "detail_policy": "surface_only",
        "discovery_mode": "explicit",
        "unit_mode": "per_material",
        "uv_scope": "target_prims",
        "explicit_material_paths": list(material_paths),
        "explicit_prim_paths": list(prim_paths),
        TEXTURE_VALIDATION_POLICY_METADATA_KEY: _validation_policy_id(runtime),
        TEXTURE_EXECUTION_MODE_METADATA_KEY: runtime.execution_mode,
    }
    if runtime.execution_mode == TEXTURE_EXECUTION_SKILL_ROUTED:
        metadata[TEXTURE_AGENTIC_PLAN_METADATA_KEY] = (
            TEXTURE_AGENTIC_PLAN_SCHEMA_VERSION
        )
        if provided_candidates:
            metadata[TEXTURE_AGENTIC_PROVIDED_CANDIDATES_METADATA_KEY] = (
                provided_candidates
            )
        if action_policy:
            metadata[TEXTURE_AGENTIC_ACTION_POLICY_METADATA_KEY] = action_policy
        if appearance_policy:
            metadata[TEXTURE_AGENTIC_APPEARANCE_POLICY_METADATA_KEY] = appearance_policy
        if evidence_views:
            metadata[TEXTURE_AGENTIC_EVIDENCE_VIEWS_METADATA_KEY] = list(evidence_views)
        metadata[TEXTURE_AGENTIC_UV_POLICY_METADATA_KEY] = uv_policy
    elif (
        provided_candidates
        or action_policy
        or appearance_policy
        or evidence_views
        or uv_policy != "inspect"
    ):
        raise ValueError(
            "agentic Texture action, evidence, and UV controls require "
            "--execution-mode skill-routed"
        )
    if material_paths:
        metadata["material_textures"] = {
            material_path: {
                "prompt": intent,
                "detail_policy": "surface_only",
            }
            for material_path in material_paths
        }
    for key in ("texture_backend", "texture_endpoint", "backend_engine"):
        value = getattr(args, key)
        if value is not None:
            normalized = str(value).strip()
            if not normalized:
                raise ValueError(f"--{key.replace('_', '-')} must not be empty.")
            if key == "texture_endpoint":
                normalized = _normalized_url(normalized, "--texture-endpoint")
            metadata[key] = normalized

    if args.embedded_run_state is not None:
        execution_context = build_embedded_domain_execution_context(
            args.embedded_run_state,
            domain="texture",
            input_asset=source_asset,
            output_dir=output_dir_candidate,
        )
        metadata = metadata_with_domain_execution_context(
            metadata,
            execution_context,
        )

    output_dir = _prepare_run_dir(output_dir_candidate, resume=False)
    request = TextureWorkflowRequest(
        source_asset=str(source_asset),
        output_dir=output_dir,
        intent=intent,
        reference_artifacts=tuple(reference_artifacts),
        target_runtime="usd-cli",
        max_vqa_iterations=max_vqa_iterations,
        metadata=metadata,
    )
    if args.embedded_run_state is not None:
        identity = build_embedded_domain_decision_identity(
            args.embedded_run_state,
            domain="texture",
            input_asset=source_asset,
            output_dir=output_dir,
            capability_digests=texture_embedded_capability_digests(),
            implementation_digests=texture_embedded_implementation_digests(),
            configuration_digests={
                "texture_request": texture_request_digest(request),
            },
        )
        if identity.execution_context != request.execution_context:
            raise ValueError(
                "Texture shared decision identity differs from the active stage"
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
    return _execute_texture_request(
        request,
        runtime=runtime,
        resume=False,
    )


def _handle_texture_resume(args: argparse.Namespace) -> int:
    from content_agent_workflows.texture import TextureWorkflowRequest

    run_dir = _prepare_run_dir(args.run_dir, resume=True)
    request_path = run_dir / "request.json"
    if not request_path.is_file():
        raise FileNotFoundError(f"Texture run request does not exist: {request_path}")
    request = TextureWorkflowRequest.model_validate_json(
        request_path.read_text(encoding="utf-8")
    )
    if request.output_dir.expanduser().resolve() != run_dir:
        raise ValueError(
            "Texture request output_dir does not match --run-dir; "
            "the persisted request cannot be changed for resume."
        )
    persisted_execution_mode = request.metadata.get(
        TEXTURE_EXECUTION_MODE_METADATA_KEY,
        TEXTURE_EXECUTION_FIXED,
    )
    if persisted_execution_mode not in {
        TEXTURE_EXECUTION_SKILL_ROUTED,
        TEXTURE_EXECUTION_FIXED,
    }:
        raise ValueError("Persisted Texture execution mode is unsupported")
    if persisted_execution_mode == TEXTURE_EXECUTION_SKILL_ROUTED:
        runtime = _load_texture_agent_launcher(run_dir, request=request)
    else:
        runtime = replace(
            _runtime_config(args),
            execution_mode=TEXTURE_EXECUTION_FIXED,
        )
        _require_texture_agent_url(runtime)
    return _execute_texture_request(
        request,
        runtime=runtime,
        resume=True,
    )


def _runtime_config(args: argparse.Namespace) -> TextureRuntimeConfig:
    raw_texture_agent_url = args.texture_agent_url
    requested_execution_mode = str(
        getattr(args, "execution_mode", TEXTURE_EXECUTION_SKILL_ROUTED)
    )
    if requested_execution_mode == TEXTURE_EXECUTION_FIXED and (
        raw_texture_agent_url is None or not str(raw_texture_agent_url).strip()
    ):
        raise ValueError(
            "--texture-agent-url or CONTENT_TEXTURE_AGENT_URL is required "
            "for --execution-mode fixed."
        )
    texture_agent_url = (
        _normalized_url(
            str(raw_texture_agent_url),
            "--texture-agent-url",
        )
        if raw_texture_agent_url is not None and str(raw_texture_agent_url).strip()
        else None
    )
    raw_vlm_backend = args.vlm_backend
    raw_vlm_model = args.vlm_model
    has_vlm_backend = bool(raw_vlm_backend is not None and str(raw_vlm_backend).strip())
    has_vlm_model = bool(raw_vlm_model is not None and str(raw_vlm_model).strip())
    if requested_execution_mode == TEXTURE_EXECUTION_FIXED and not has_vlm_backend:
        raise ValueError("--vlm-backend or CONTENT_TEXTURE_VLM_BACKEND is required.")
    if requested_execution_mode == TEXTURE_EXECUTION_FIXED and not has_vlm_model:
        raise ValueError("--vlm-model or CONTENT_TEXTURE_VLM_MODEL is required.")
    if has_vlm_backend != has_vlm_model:
        raise ValueError("--vlm-backend and --vlm-model must be configured together.")
    vlm_backend = str(raw_vlm_backend).strip() if has_vlm_backend else None
    vlm_model = str(raw_vlm_model).strip() if has_vlm_model else None
    if vlm_backend is not None and vlm_backend not in PUBLIC_VLM_BACKENDS:
        choices = ", ".join(PUBLIC_VLM_BACKENDS)
        raise ValueError(f"--vlm-backend must be one of: {choices}.")
    raw_vlm_base_url = args.vlm_base_url
    if raw_vlm_base_url is None and vlm_backend == "openai":
        raw_vlm_base_url = _first_present_environment_value(
            "OPENAI_BASE_URL",
            "OPENAI_API_BASE",
        )
    if raw_vlm_base_url is None and vlm_backend == "anthropic":
        raw_vlm_base_url = _first_present_environment_value(
            "ANTHROPIC_API_URL",
            "ANTHROPIC_BASE_URL",
        )
    vlm_base_url = (
        _normalized_url(raw_vlm_base_url, "--vlm-base-url")
        if raw_vlm_base_url is not None
        else None
    )
    vlm_api_key_env = _optional_env_name(
        args.vlm_api_key_env,
        "--vlm-api-key-env",
    )
    if (
        vlm_base_url is not None or vlm_api_key_env is not None
    ) and vlm_backend is None:
        raise ValueError(
            "VLM endpoint or credential options require --vlm-backend and --vlm-model."
        )
    if (
        vlm_base_url is not None
        and vlm_backend in {"anthropic", "gemini"}
        and vlm_api_key_env is None
    ):
        raise ValueError(
            "--vlm-api-key-env is required with --vlm-base-url for "
            f"{vlm_backend}; provider-default credentials are not forwarded "
            "to custom endpoints."
        )
    texture_agent_token_env = _optional_env_name(
        args.texture_agent_token_env, "--texture-agent-token-env"
    )
    if texture_agent_token_env is not None and texture_agent_url is None:
        raise ValueError(
            "--texture-agent-token-env requires --texture-agent-url or "
            "CONTENT_TEXTURE_AGENT_URL."
        )
    if texture_agent_url is not None:
        _validate_bearer_transport(
            texture_agent_url,
            credential_env=texture_agent_token_env,
            option="--texture-agent-url",
        )
    return TextureRuntimeConfig(
        texture_agent_url=texture_agent_url,
        texture_agent_token_env=texture_agent_token_env,
        texture_timeout_seconds=float(args.texture_timeout),
        texture_poll_interval_seconds=float(args.texture_poll_interval),
        vlm_backend=vlm_backend,
        vlm_model=vlm_model,
        vlm_base_url=vlm_base_url,
        vlm_api_key_env=vlm_api_key_env,
        vlm_timeout_seconds=float(args.vlm_timeout),
        json_output=bool(args.json),
        runner=str(getattr(args, "runner", RUNNER_CODEX)),
        model=getattr(args, "model", None),
        model_reasoning_effort=getattr(args, "model_reasoning_effort", None),
        codex_base_url=getattr(args, "codex_base_url", None),
        codex_sandbox_mode=str(
            getattr(args, "codex_sandbox_mode", CODEX_SANDBOX_WORKSPACE_WRITE)
        ),
        codex_config=_load_agent_config(
            getattr(args, "codex_config_file", []),
            getattr(args, "codex_config_json", []),
            option="--codex-config-json",
        ),
        claude_config=_load_agent_config(
            getattr(args, "claude_config_file", []),
            getattr(args, "claude_config_json", []),
            option="--claude-config-json",
        ),
        claude_permission_mode=str(getattr(args, "claude_permission_mode", "default")),
        claude_max_turns=getattr(args, "claude_max_turns", None),
        claude_execution_mode=str(
            getattr(args, "claude_execution_mode", CLAUDE_EXECUTION_SDK)
        ),
        child_timeout_seconds=float(getattr(args, "child_timeout", 1800.0)),
        agent_cwd=(
            Path(getattr(args, "agent_cwd")).expanduser().resolve()
            if getattr(args, "agent_cwd", None) is not None
            else None
        ),
        dry_run=bool(getattr(args, "dry_run", False)),
        execution_mode=str(requested_execution_mode),
    )


def _require_texture_agent_url(runtime: TextureRuntimeConfig) -> str:
    """Resolve a service endpoint only after the accepted plan selected it."""

    if runtime.texture_agent_url is None:
        raise ValueError(
            "The accepted Texture plan selected a service generation capability, "
            "but --texture-agent-url and CONTENT_TEXTURE_AGENT_URL are unset."
        )
    return runtime.texture_agent_url


def _require_vlm_identity(runtime: TextureRuntimeConfig) -> tuple[str, str]:
    """Return the fixed compatibility review provider selected by the caller."""

    if runtime.vlm_backend is None or runtime.vlm_model is None:
        raise ValueError(
            "The fixed Texture workflow requires --vlm-backend and --vlm-model."
        )
    return runtime.vlm_backend, runtime.vlm_model


def _load_agent_config(
    paths: list[Path],
    payloads: list[str],
    *,
    option: str,
) -> dict[str, object] | None:
    config: dict[str, object] = {}
    for path in paths:
        raw = json.loads(path.expanduser().read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"{path} must contain a JSON object")
        config.update(raw)
    for payload in payloads:
        raw = json.loads(payload)
        if not isinstance(raw, dict):
            raise ValueError(f"{option} must be a JSON object")
        config.update(raw)
    return config or None


def _first_present_environment_value(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return None


def _execute_texture_request(
    request: TextureWorkflowRequest,
    *,
    runtime: TextureRuntimeConfig,
    resume: bool,
) -> int:
    """Route normal runs through one child; keep fixed mode explicit."""

    embedded = _is_embedded_texture_request(request)
    if runtime.execution_mode == TEXTURE_EXECUTION_FIXED:
        if embedded:
            raise ValueError(
                "Embedded Texture execution cannot use the fixed compatibility mode"
            )
        return _execute_texture_compatibility_request(
            request,
            runtime=runtime,
            resume=resume,
        )
    with _skill_routed_run_lock(request.output_dir, domain="texture"):
        if embedded:
            from content_agent_workflows.texture import TextureFinalizationResult

            # The outer asset coordinator owns the reasoning loop and invokes one
            # focused embedded transition at a time; embedded domains never spawn
            # a nested long-running child here.
            if resume:
                raise ValueError(
                    "Embedded Texture continuation must use the focused _agent-step "
                    "surface with an exact decision patch"
                )
            _write_texture_agent_launcher(request.output_dir, runtime, request=request)
            outcome = _execute_texture_skill_step(
                request,
                runtime=runtime,
                decision_patch=None,
            )
            _print_texture_step_outcome(
                outcome,
                run_dir=request.output_dir,
                json_output=runtime.json_output,
            )
            return (
                _result_exit_code(outcome)
                if isinstance(outcome, TextureFinalizationResult)
                else 0
            )
        return _launch_texture_child(request, runtime=runtime, resume=resume)


def _is_embedded_texture_request(request: TextureWorkflowRequest) -> bool:
    context = request.execution_context
    return context is not None and context.mode == "embedded"


def _execute_texture_compatibility_request(
    request: TextureWorkflowRequest,
    *,
    runtime: TextureRuntimeConfig,
    resume: bool,
) -> int:
    """Run the pre-#1061 fixed controller as an explicit compatibility path."""

    from content_agent_workflows.texture import (
        LiveUsdCliTextureValidator,
        TextureAgentServiceClient,
        TextureWorkflowCancellationToken,
        VlmTextureVisualAssessor,
        run_batch_texture_workflow,
    )

    cancellation_token = TextureWorkflowCancellationToken()
    workflow_active = False

    try:
        with _cooperative_signal_handlers(
            cancellation_token,
            checkpoint_path=request.output_dir / "workflow_checkpoint.json",
            workflow_active=lambda: workflow_active,
        ):
            checkpoint = _preflight_resume(request, runtime=runtime) if resume else None

            needs_runtime_services = (
                checkpoint is None
                or checkpoint.next_action
                not in {
                    "done",
                    "finalize",
                }
            )
            token_value = None
            if needs_runtime_services and runtime.texture_agent_token_env is not None:
                token_value = os.getenv(runtime.texture_agent_token_env)
                if not token_value:
                    raise ValueError(
                        f"{runtime.texture_agent_token_env} is not set or is empty."
                    )

            client: Any
            validator: Any
            if needs_runtime_services:
                vlm_backend, vlm_model = _require_vlm_identity(runtime)
                client = TextureAgentServiceClient(
                    base_url=_require_texture_agent_url(runtime),
                    timeout_seconds=runtime.texture_timeout_seconds,
                    poll_interval_seconds=runtime.texture_poll_interval_seconds,
                    max_status_poll_failures=5,
                    token=token_value,
                )
                lazy_vlm = _LazyVlm(
                    backend=vlm_backend,
                    model=vlm_model,
                    base_url=runtime.vlm_base_url,
                    api_key_env=runtime.vlm_api_key_env,
                    timeout_seconds=runtime.vlm_timeout_seconds,
                )
                validator = LiveUsdCliTextureValidator(
                    assessor=VlmTextureVisualAssessor(lazy_vlm),
                    validation_policy_id=_validation_policy_id(runtime),
                )
            else:
                client = _UnavailableRuntimeAdapter()
                validator = _UnavailableRuntimeAdapter()

            request.output_dir.mkdir(parents=True, exist_ok=True)

            workflow_active = True
            try:
                result = run_batch_texture_workflow(
                    request,
                    client=client,
                    validator=validator,
                    progress_callback=_print_progress,
                    resume=resume,
                    cancellation_check=cancellation_token.is_cancelled,
                )
            finally:
                workflow_active = False
    finally:
        pass
    _print_result(result, json_output=runtime.json_output)
    return _result_exit_code(result)


def _launch_texture_child(
    request: TextureWorkflowRequest,
    *,
    runtime: TextureRuntimeConfig,
    resume: bool,
) -> int:
    if (
        request.metadata.get(TEXTURE_AGENTIC_PLAN_METADATA_KEY)
        == TEXTURE_AGENTIC_PLAN_SCHEMA_VERSION
    ):
        return _launch_texture_plan_child(
            request,
            runtime=runtime,
            resume=resume,
        )
    return _launch_texture_legacy_child(
        request,
        runtime=runtime,
        resume=resume,
    )


def _prepare_texture_agentic_source_request(
    request: TextureWorkflowRequest,
    *,
    resume: bool,
) -> TextureWorkflowRequest:
    """Freeze optional provider-free UV preparation into the durable request."""

    from content_agent_workflows.texture import (
        TextureAgenticSourcePreparation,
        bind_texture_agentic_artifact,
        prepare_texture_agentic_source,
        validate_texture_agentic_source_preparation,
    )

    policy = request.metadata.get(TEXTURE_AGENTIC_UV_POLICY_METADATA_KEY, "inspect")
    if policy not in {"inspect", "generate_missing"}:
        raise ValueError("agentic Texture UV policy is unsupported")
    raw_preparation = request.metadata.get(
        TEXTURE_AGENTIC_SOURCE_PREPARATION_METADATA_KEY
    )
    if raw_preparation is not None:
        if not isinstance(raw_preparation, Mapping):
            raise ValueError("Texture source-preparation binding must be an object")
        raw_path = raw_preparation.get("path")
        if not isinstance(raw_path, str):
            raise ValueError("Texture source-preparation binding omitted its path")
        preparation_binding = bind_texture_agentic_artifact(raw_path)
        if preparation_binding.model_dump(mode="json") != dict(raw_preparation):
            raise ValueError("Texture source-preparation binding changed")
        preparation = TextureAgenticSourcePreparation.model_validate(
            load_json(preparation_binding.path)
        )
        validate_texture_agentic_source_preparation(
            preparation,
            receipt_binding=preparation_binding,
        )
        if policy != preparation.policy:
            raise ValueError("Texture request changed its source-preparation policy")
        if request.source_asset != preparation.effective_source.path:
            raise ValueError("Texture request source differs from UV-prepared bytes")
        expected_targets = tuple(
            str(item) for item in request.metadata.get("explicit_prim_paths") or ()
        )
        if preparation.target_prim_paths != expected_targets:
            raise ValueError("Texture request changed its UV-preparation scope")
        return request
    if resume:
        if policy == "generate_missing":
            raise ValueError("Texture resume omitted frozen source preparation")
        return request
    if policy == "inspect":
        return request
    target_prim_paths = tuple(
        str(item) for item in request.metadata.get("explicit_prim_paths") or ()
    )
    if not target_prim_paths:
        raise ValueError(
            "agentic Texture missing-UV preparation requires explicit prim scope"
        )
    preparation, preparation_binding = prepare_texture_agentic_source(
        request.source_asset,
        output_dir=request.output_dir / "source_preparation",
        target_prim_paths=target_prim_paths,
    )
    return request.model_copy(
        update={
            "source_asset": preparation.effective_source.path,
            "metadata": {
                **request.metadata,
                TEXTURE_AGENTIC_SOURCE_PREPARATION_METADATA_KEY: (
                    preparation_binding.model_dump(mode="json")
                ),
            },
        }
    )


def _validate_texture_plan_proposal_inputs(
    request: TextureWorkflowRequest,
    *,
    request_path: Path,
    request_sha256: str,
    preparation: Any,
    preparation_binding: Any,
    output_contract_identity: ChildLaunchArtifactIdentity,
) -> None:
    """Revalidate every immutable input before accepting a child proposal."""

    from content_agent_workflows.texture import (
        bind_texture_agentic_artifact,
        validate_texture_preparation,
    )

    validate_texture_preparation(
        preparation,
        preparation_binding=preparation_binding,
        expected_request=_texture_agentic_capability_request(request),
    )
    if _prepare_texture_agentic_source_request(request, resume=True) != request:
        raise RuntimeError("Texture plan proposal changed frozen source preparation")
    if (
        file_sha256(request_path) != request_sha256
        or bind_texture_agentic_artifact(preparation_binding.path)
        != preparation_binding
        or bind_texture_agentic_artifact(preparation.request.source.path)
        != preparation.request.source
        or child_launch_artifact_identity(
            request.output_dir.expanduser().resolve(),
            Path(output_contract_identity.path),
        )
        != output_contract_identity
    ):
        raise RuntimeError("Texture plan proposal modified an immutable input")


def _texture_plan_output_contract_document() -> dict[str, object]:
    """Return the package-owned structural authority for plan-only output."""

    return {
        "schema_version": TEXTURE_AGENTIC_PLAN_OUTPUT_CONTRACT_SCHEMA_VERSION,
        "output_schema": TEXTURE_AGENTIC_PLAN_SCHEMA_VERSION,
        "json_schema": _texture_plan_output_schema(),
    }


def _texture_provider_output_schema(schema: dict[str, object]) -> dict[str, object]:
    """Make one canonical Texture schema valid for strict structured output."""

    definitions = schema.get("$defs")
    if not isinstance(definitions, dict):
        raise RuntimeError("Texture provider schema must declare definitions")
    for definition, field_name in (
        ("TextureGeneratorInputs", "parameters"),
        ("TextureProvidedImageProducer", "provenance"),
    ):
        definition_schema = definitions.get(definition)
        if definition_schema is None:
            continue
        properties = (
            definition_schema.get("properties")
            if isinstance(definition_schema, dict)
            else None
        )
        if not isinstance(properties, dict) or field_name not in properties:
            raise RuntimeError("Texture provider schema projection drifted")
        properties[field_name] = {
            "type": "string",
            "description": "JSON object encoded as a string for strict output",
        }

    def make_strict(node: object) -> None:
        if isinstance(node, list):
            for item in node:
                make_strict(item)
            return
        if not isinstance(node, dict):
            return
        node.pop("default", None)
        properties = node.get("properties")
        if properties is not None:
            if not isinstance(properties, dict):
                raise RuntimeError("Texture provider properties must be a map")
            node["required"] = list(properties)
            node["additionalProperties"] = False
        elif (additional := node.get("additionalProperties")) is not None and (
            additional is not False
        ):
            raise RuntimeError(
                "Texture provider schema contains an unsupported free-form map"
            )
        definitions = node.get("$defs")
        if definitions is not None:
            if not isinstance(definitions, dict):
                raise RuntimeError("Texture provider definitions must be a map")
            for definition in definitions.values():
                make_strict(definition)
        if isinstance(properties, dict):
            for property_schema in properties.values():
                make_strict(property_schema)
        for keyword in ("items", "anyOf", "allOf", "oneOf", "prefixItems"):
            if keyword in node:
                make_strict(node[keyword])

    make_strict(schema)
    return schema


def _decode_texture_plan_provider_maps(payload: object) -> object:
    """Decode the two free-form maps carried as strict-schema JSON strings."""

    if not isinstance(payload, dict):
        return payload

    def decode(value: object, *, label: str) -> dict[str, object]:
        if isinstance(value, dict):
            return value
        if not isinstance(value, str):
            raise ValueError(f"{label} must be a JSON object string")
        decoded = json.loads(value)
        if not isinstance(decoded, dict):
            raise ValueError(f"{label} must decode to a JSON object")
        return decoded

    dispositions = payload.get("dispositions")
    if not isinstance(dispositions, list):
        return payload
    for disposition in dispositions:
        if not isinstance(disposition, dict):
            continue
        generator_inputs = disposition.get("generator_inputs")
        if not isinstance(generator_inputs, dict):
            continue
        if "parameters" in generator_inputs:
            generator_inputs["parameters"] = decode(
                generator_inputs["parameters"],
                label="Texture generator parameters",
            )
        provided_images = generator_inputs.get("provided_images")
        if not isinstance(provided_images, list):
            continue
        for provided_image in provided_images:
            producer = (
                provided_image.get("producer")
                if isinstance(provided_image, dict)
                else None
            )
            if isinstance(producer, dict) and "provenance" in producer:
                producer["provenance"] = decode(
                    producer["provenance"],
                    label="Texture provided-image provenance",
                )
    return payload


def _texture_plan_output_schema() -> dict[str, object]:
    """Return the provider-enforced schema for the tool-free planning turn."""

    from content_agent_workflows.texture import TextureAgenticPlan

    return cast(
        dict[str, object],
        _texture_provider_output_schema(
            TextureAgenticPlan.model_json_schema(mode="validation")
        ),
    )


def _texture_review_output_schema() -> dict[str, object]:
    """Return the provider-enforced schema for the tool-free review turn."""

    from content_agent_workflows.texture import TextureAgenticReviewReceipt

    return cast(
        dict[str, object],
        _texture_provider_output_schema(
            TextureAgenticReviewReceipt.model_json_schema(mode="validation")
        ),
    )


def _load_texture_plan_child_output(
    *,
    run_dir: Path,
    child_final_path: Path,
    runtime: TextureRuntimeConfig,
) -> Any:
    """Load one plan only when its structured turn was clean and tool-free."""

    from content_agent_workflows.texture import TextureAgenticPlan

    if child_final_path.is_symlink() or not child_final_path.is_file():
        raise RuntimeError("Texture plan child did not return structured output")
    if _child_turn_used_tools(
        run_dir=run_dir,
        artifact_prefix="texture_plan_only",
        output_schema=_texture_plan_output_schema(),
        child_final_path=child_final_path,
        allow_claude_structured_output=runtime.runner == RUNNER_CLAUDE,
    ):
        raise RuntimeError(
            "Texture plan structured output lacks clean tool-free event evidence"
        )
    return TextureAgenticPlan.model_validate(
        _decode_texture_plan_provider_maps(load_json(child_final_path))
    )


def _validate_texture_plan_child_provenance(
    *,
    run_dir: Path,
    child_final_path: Path,
    plan_path: Path,
    runtime: TextureRuntimeConfig,
) -> None:
    """Reject a canonical plan without its exact clean structured source."""

    from content_agent_workflows.texture import TextureAgenticPlan

    child_plan = _load_texture_plan_child_output(
        run_dir=run_dir,
        child_final_path=child_final_path,
        runtime=runtime,
    )
    if plan_path.is_symlink() or not plan_path.is_file():
        raise RuntimeError("Texture parent-owned plan is not a regular file")
    canonical_plan = TextureAgenticPlan.model_validate(load_json(plan_path))
    if canonical_plan != child_plan:
        raise RuntimeError("Texture parent-owned plan differs from child output")


def _materialize_texture_plan_child_output(
    *,
    run_dir: Path,
    child_final_path: Path,
    plan_path: Path,
    runtime: TextureRuntimeConfig,
) -> None:
    """Validate structured provider output before the parent writes the plan."""

    plan = _load_texture_plan_child_output(
        run_dir=run_dir,
        child_final_path=child_final_path,
        runtime=runtime,
    )
    if plan_path.exists() or plan_path.is_symlink():
        raise RuntimeError("Texture plan child wrote the parent-owned plan path")
    atomic_write_json(plan_path, plan.model_dump(mode="json"))


def _load_texture_review_child_output(
    *,
    run_dir: Path,
    child_final_path: Path,
    runtime: TextureRuntimeConfig,
) -> Any:
    """Load one review only when its structured turn was clean and tool-free."""

    from content_agent_workflows.texture import TextureAgenticReviewReceipt

    if child_final_path.is_symlink() or not child_final_path.is_file():
        raise RuntimeError("Texture review child did not return structured output")
    if _child_turn_used_tools(
        run_dir=run_dir,
        artifact_prefix="texture_review_only",
        output_schema=_texture_review_output_schema(),
        child_final_path=child_final_path,
        allow_claude_structured_output=runtime.runner == RUNNER_CLAUDE,
    ):
        raise RuntimeError(
            "Texture review structured output lacks clean tool-free event evidence"
        )
    return TextureAgenticReviewReceipt.model_validate(load_json(child_final_path))


def _validate_texture_review_child_provenance(
    *,
    run_dir: Path,
    child_final_path: Path,
    proposal_path: Path,
    runtime: TextureRuntimeConfig,
) -> None:
    """Reject a canonical review without its exact clean structured source."""

    from content_agent_workflows.texture import TextureAgenticReviewReceipt

    child_review = _load_texture_review_child_output(
        run_dir=run_dir,
        child_final_path=child_final_path,
        runtime=runtime,
    )
    if proposal_path.is_symlink() or not proposal_path.is_file():
        raise RuntimeError("Texture parent-owned review is not a regular file")
    canonical_review = TextureAgenticReviewReceipt.model_validate(
        load_json(proposal_path)
    )
    if canonical_review != child_review:
        raise RuntimeError("Texture parent-owned review differs from child output")


def _materialize_texture_review_child_output(
    *,
    run_dir: Path,
    child_final_path: Path,
    proposal_path: Path,
    runtime: TextureRuntimeConfig,
) -> None:
    """Validate structured provider output before the parent writes the review."""

    review = _load_texture_review_child_output(
        run_dir=run_dir,
        child_final_path=child_final_path,
        runtime=runtime,
    )
    if proposal_path.exists() or proposal_path.is_symlink():
        raise RuntimeError("Texture review child wrote the parent-owned proposal path")
    atomic_write_json(proposal_path, review.model_dump(mode="json"))


def _texture_reasoning_attempt_started(
    run_dir: Path,
    *,
    bridge_artifact_prefix: str,
    child_output_name: str,
) -> bool:
    """Detect a prior provider turn that lacks adoptable structured output."""

    raw_dir = run_dir / "raw"
    return any(
        path.exists() or path.is_symlink()
        for path in (
            raw_dir / child_output_name,
            raw_dir / f"{bridge_artifact_prefix}_launch_descriptor.json",
            raw_dir / f"{bridge_artifact_prefix}_request.json",
            raw_dir / f"{bridge_artifact_prefix}_items.json",
            raw_dir / f"{bridge_artifact_prefix}_result.json",
            raw_dir / f"{bridge_artifact_prefix}_observable_events.jsonl",
        )
    )


def _prepare_texture_plan_output_contract(
    run_dir: Path,
) -> ChildLaunchArtifactIdentity:
    """Create or validate the exact run-local plan output contract."""

    resolved_run_dir = run_dir.expanduser().resolve()
    path = resolved_run_dir / "raw" / TEXTURE_AGENTIC_PLAN_OUTPUT_CONTRACT_NAME
    expected = _texture_plan_output_contract_document()
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file():
            raise ValueError("Texture plan output contract must be a regular file")
        if load_json(path) != expected:
            raise ValueError("Texture plan output contract differs from package schema")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(path, expected)
    return child_launch_artifact_identity(resolved_run_dir, path)


def _replay_texture_agentic_terminal_plan_if_present(
    *,
    request: TextureWorkflowRequest,
    runtime: TextureRuntimeConfig,
    request_path: Path,
    prompt_path: Path,
    preparation_path: Path,
    accepted_path: Path,
) -> int | None:
    """Replay a terminal plan before any mutable launch preamble is written."""

    run_dir = request.output_dir.expanduser().resolve()
    terminal_path = run_dir / "texture_terminal_receipt.json"
    result_path = run_dir / "workflow_result.json"
    if not (terminal_path.exists() or terminal_path.is_symlink()) and not (
        result_path.exists() or result_path.is_symlink()
    ):
        return None

    from content_agent_workflows.texture import (
        TextureAcceptedPlan,
        TexturePreparationPacket,
        bind_texture_agentic_artifact,
        validate_texture_accepted_plan,
        validate_texture_preparation,
    )

    if not preparation_path.is_file() or preparation_path.is_symlink():
        raise ValueError(
            "Texture terminal resume requires a regular preparation packet"
        )
    if not accepted_path.is_file() or accepted_path.is_symlink():
        raise ValueError("Texture terminal resume requires a regular accepted plan")
    preparation = TexturePreparationPacket.model_validate(load_json(preparation_path))
    preparation_binding = bind_texture_agentic_artifact(preparation_path)
    validate_texture_preparation(
        preparation,
        preparation_binding=preparation_binding,
        expected_request=_texture_agentic_capability_request(request),
    )
    accepted = TextureAcceptedPlan.model_validate(load_json(accepted_path))
    accepted_binding = bind_texture_agentic_artifact(accepted_path)
    validate_texture_accepted_plan(
        accepted,
        accepted_binding=accepted_binding,
        preparation=preparation,
        preparation_binding=preparation_binding,
    )
    terminal_result = _replay_texture_agentic_terminal_result(
        request=request,
        request_path=request_path,
        prompt_path=prompt_path,
        preparation_binding=preparation_binding,
        accepted=accepted,
        accepted_binding=accepted_binding,
    )
    if terminal_result is None:
        raise ValueError("Texture terminal artifacts disappeared during resume")
    _print_texture_plan_handoff(
        terminal_result,
        json_output=runtime.json_output,
    )
    return 0 if terminal_result["status"] == "published" else 2


def _launch_texture_plan_child(
    request: TextureWorkflowRequest,
    *,
    runtime: TextureRuntimeConfig,
    resume: bool,
) -> int:
    """Prepare first, launch one plan-only child, and freeze its exact proposal."""

    from content_agent_workflows.texture import (
        TextureAcceptedPlan,
        TexturePreparationPacket,
        accept_texture_agentic_plan,
        bind_texture_agentic_artifact,
        validate_texture_accepted_plan,
        validate_texture_preparation,
    )

    run_dir = request.output_dir.expanduser().resolve()
    if resume:
        _validate_texture_resume_policy(request, runtime=runtime)
    run_dir.mkdir(parents=True, exist_ok=True)
    request_path = run_dir / "request.json"
    if not runtime.dry_run:
        request = _prepare_texture_agentic_source_request(request, resume=resume)
    if resume:
        if not request_path.is_file():
            raise FileNotFoundError(f"Texture request does not exist: {request_path}")
        persisted = type(request).model_validate(load_json(request_path))
        if persisted != request:
            raise ValueError("Texture resume request differs from frozen request bytes")
    elif request_path.exists():
        raise FileExistsError(f"Texture request already exists: {request_path}")
    else:
        atomic_write_json(request_path, request.model_dump(mode="json"))
    request_sha256 = file_sha256(request_path)
    preparation_path = run_dir / "preparation" / "texture_preparation.json"
    prompt_path = run_dir / "prompts" / "texture_plan_only.md"
    accepted_path = run_dir / "accepted_texture_plan.json"
    if resume:
        terminal_exit_code = _replay_texture_agentic_terminal_plan_if_present(
            request=request,
            runtime=runtime,
            request_path=request_path,
            prompt_path=prompt_path,
            preparation_path=preparation_path,
            accepted_path=accepted_path,
        )
        if terminal_exit_code is not None:
            return terminal_exit_code
    _write_texture_agent_launcher(run_dir, runtime, request=request)
    trace_writer = TraceWriter(run_dir)
    trace_writer.write(
        "workflow_started",
        phase="texture_plan_only",
        summary="Started one agentic Texture plan-only workflow.",
        artifacts=[str(request_path)],
        data={"resume": resume},
    )
    if runtime.dry_run:
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(
            prompt_path,
            _build_texture_plan_dry_run_prompt(request),
        )
        trace_writer.write(
            "workflow_dry_run",
            phase="texture_plan_only",
            summary="Dry run stopped before deterministic preparation and child launch.",
        )
        build_trace(run_dir)
        _print_texture_plan_handoff(
            {
                "schema_version": "content-workflow-cli.texture-plan-handoff.v1",
                "status": "dry_run",
                "run_dir": str(run_dir),
                "request_path": str(request_path),
                "prompt_path": str(prompt_path),
            },
            json_output=runtime.json_output,
        )
        return 0

    if preparation_path.is_file():
        preparation = TexturePreparationPacket.model_validate(
            load_json(preparation_path)
        )
        preparation_binding = bind_texture_agentic_artifact(preparation_path)
        validate_texture_preparation(
            preparation,
            preparation_binding=preparation_binding,
            expected_request=_texture_agentic_capability_request(request),
        )
    else:
        preparation, preparation_binding = _prepare_texture_agentic_preparation(
            request,
            runtime=runtime,
            resume_incomplete=resume,
        )
    if preparation_binding.path != str(preparation_path):
        raise ValueError("Texture preparation was written outside its canonical path")
    trace_writer.write(
        "preparation_completed",
        phase="texture_plan_only",
        summary="Provider-neutral Texture preparation completed before reasoning.",
        artifacts=[preparation_binding.path],
        data={
            "source_sha256": preparation.request.source.sha256,
            "preparation_sha256": preparation_binding.sha256,
            "prepared_unit_ids": list(preparation.scope_plan.selected_unit_ids),
        },
    )
    plan_path = run_dir / "texture_plan.json"
    child_output_path = run_dir / "raw" / "texture_child_output.jsonl"
    child_final_path = run_dir / "raw" / "texture_child_final.json"
    if plan_path.is_symlink():
        raise ValueError("Texture plan proposal must be a regular file")
    if accepted_path.is_file():
        _validate_texture_plan_child_provenance(
            run_dir=run_dir,
            child_final_path=child_final_path,
            plan_path=plan_path,
            runtime=runtime,
        )
        accepted = TextureAcceptedPlan.model_validate(load_json(accepted_path))
        accepted_binding = bind_texture_agentic_artifact(accepted_path)
        validate_texture_accepted_plan(
            accepted,
            accepted_binding=accepted_binding,
            preparation=preparation,
            preparation_binding=preparation_binding,
        )
        return _finish_texture_plan_handoff(
            request=request,
            runtime=runtime,
            request_path=request_path,
            prompt_path=prompt_path,
            preparation_binding=preparation_binding,
            accepted=accepted,
            accepted_binding=accepted_binding,
            trace_writer=trace_writer,
            resumed=True,
        )

    if resume and (plan_path.exists() or plan_path.is_symlink()):
        _validate_texture_plan_child_provenance(
            run_dir=run_dir,
            child_final_path=child_final_path,
            plan_path=plan_path,
            runtime=runtime,
        )
        output_contract_identity = _prepare_texture_plan_output_contract(run_dir)
        _validate_texture_plan_proposal_inputs(
            request,
            request_path=request_path,
            request_sha256=request_sha256,
            preparation=preparation,
            preparation_binding=preparation_binding,
            output_contract_identity=output_contract_identity,
        )
        accepted, accepted_binding = accept_texture_agentic_plan(
            plan_path,
            preparation=preparation,
            preparation_binding=preparation_binding,
            output_path=accepted_path,
        )
        return _finish_texture_plan_handoff(
            request=request,
            runtime=runtime,
            request_path=request_path,
            prompt_path=prompt_path,
            preparation_binding=preparation_binding,
            accepted=accepted,
            accepted_binding=accepted_binding,
            trace_writer=trace_writer,
            resumed=True,
        )

    if resume and (child_final_path.exists() or child_final_path.is_symlink()):
        output_contract_identity = _prepare_texture_plan_output_contract(run_dir)
        _validate_texture_plan_proposal_inputs(
            request,
            request_path=request_path,
            request_sha256=request_sha256,
            preparation=preparation,
            preparation_binding=preparation_binding,
            output_contract_identity=output_contract_identity,
        )
        _materialize_texture_plan_child_output(
            run_dir=run_dir,
            child_final_path=child_final_path,
            plan_path=plan_path,
            runtime=runtime,
        )
        accepted, accepted_binding = accept_texture_agentic_plan(
            plan_path,
            preparation=preparation,
            preparation_binding=preparation_binding,
            output_path=accepted_path,
        )
        return _finish_texture_plan_handoff(
            request=request,
            runtime=runtime,
            request_path=request_path,
            prompt_path=prompt_path,
            preparation_binding=preparation_binding,
            accepted=accepted,
            accepted_binding=accepted_binding,
            trace_writer=trace_writer,
            resumed=True,
        )

    if resume and _texture_reasoning_attempt_started(
        run_dir,
        bridge_artifact_prefix="texture_plan_only",
        child_output_name="texture_child_output.jsonl",
    ):
        raise RuntimeError(
            "Texture plan resume found an incomplete provider turn without "
            "adoptable structured output; start a fresh run rather than repeating it"
        )

    capability_inventory, domain_policy_bounds, output_contract_identity = (
        _prepare_texture_plan_child_launch_contract(
            request,
            runtime=runtime,
            preparation_binding=preparation_binding,
        )
    )
    child_config = _texture_child_runtime_config(
        request,
        runtime,
        capability_inventory=capability_inventory,
        domain_policy_bounds=domain_policy_bounds,
    )
    prompt = _build_texture_plan_prompt(
        request,
        preparation=preparation,
        preparation_binding=preparation_binding,
        output_contract_identity=output_contract_identity,
    )
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(prompt_path, prompt)
    child_output_path.parent.mkdir(parents=True, exist_ok=True)
    returncode = run_child_agent(
        config=child_config,
        prompt=prompt,
        run_dir=run_dir,
        child_output_path=child_output_path,
        child_final_path=child_final_path,
        scene_service=None,
        bridge_artifact_prefix="texture_plan_only",
        output_schema=_texture_plan_output_schema(),
        stage_skills=False,
        tools_disabled=True,
    )
    trace_writer.write(
        "child_finished",
        phase="texture_plan_only",
        summary="Tool-free Texture child returned one structured plan proposal.",
        artifacts=[str(child_output_path), str(child_final_path)],
        data={"returncode": returncode},
    )
    if returncode != 0:
        build_trace(run_dir)
        raise RuntimeError(f"Texture plan child agent exited with code {returncode}")
    try:
        _validate_texture_plan_proposal_inputs(
            request,
            request_path=request_path,
            request_sha256=request_sha256,
            preparation=preparation,
            preparation_binding=preparation_binding,
            output_contract_identity=output_contract_identity,
        )
    except Exception:
        build_trace(run_dir)
        raise
    try:
        _materialize_texture_plan_child_output(
            run_dir=run_dir,
            child_final_path=child_final_path,
            plan_path=plan_path,
            runtime=runtime,
        )
    except Exception:
        build_trace(run_dir)
        raise
    accepted, accepted_binding = accept_texture_agentic_plan(
        plan_path,
        preparation=preparation,
        preparation_binding=preparation_binding,
        output_path=accepted_path,
    )
    return _finish_texture_plan_handoff(
        request=request,
        runtime=runtime,
        request_path=request_path,
        prompt_path=prompt_path,
        preparation_binding=preparation_binding,
        accepted=accepted,
        accepted_binding=accepted_binding,
        trace_writer=trace_writer,
        resumed=False,
    )


def _texture_agentic_capability_request(request: TextureWorkflowRequest) -> Any:
    """Rebuild the exact deterministic capability request bound to a run."""

    from content_agent_workflows.texture import (
        TextureOperationSelection,
        build_texture_capability_request,
    )

    material_paths = tuple(
        str(path) for path in request.metadata.get("explicit_material_paths") or ()
    )
    prim_paths = tuple(
        str(path) for path in request.metadata.get("explicit_prim_paths") or ()
    )
    if bool(material_paths) == bool(prim_paths):
        raise ValueError(
            "agentic Texture preparation requires exactly one explicit scope type"
        )
    preparation_root = request.output_dir.expanduser().resolve() / "preparation"
    return build_texture_capability_request(
        source_asset=request.source_asset,
        output_dir=preparation_root,
        intent=request.intent,
        material_prim_paths=material_paths,
        prim_paths=prim_paths,
        reference_artifacts=tuple(
            (item.role, item.artifact.path) for item in request.reference_artifacts
        ),
        operations=TextureOperationSelection(),
        texture_size=int(request.metadata.get("texture_size") or 1024),
        metadata=request.metadata,
    )


def _prepare_texture_agentic_preparation(
    request: TextureWorkflowRequest,
    *,
    runtime: TextureRuntimeConfig,
    resume_incomplete: bool = False,
) -> tuple[Any, Any]:
    """Run provider-neutral scope and OVRTX preparation without semantic clients."""

    from content_agent_workflows.texture import (
        LiveUsdCliTextureValidator,
        prepare_texture_scope,
    )

    capability_request = _texture_agentic_capability_request(request)
    inspector = LiveUsdCliTextureValidator(
        assessor=None,
        validation_policy_id=_validation_policy_id(runtime),
    )
    return cast(
        tuple[Any, Any],
        prepare_texture_scope(
            capability_request,
            inspector=inspector,
            resume_incomplete=resume_incomplete,
        ),
    )


def _verify_texture_terminal_nested_artifacts(
    *,
    verify: Callable[..., Any],
    ledger: Any | None = None,
    cleanup: Any | None = None,
    evidence: Any | None = None,
    readback: Any | None = None,
    review: Any | None = None,
    publication: Any | None = None,
) -> None:
    """Replay every nested byte binding reachable from terminal receipts."""

    if ledger is not None:
        verify(ledger.source, label="Texture ledger source")
        verify(ledger.final_candidate, label="Texture final candidate")
        for record in ledger.records:
            verify(record.input_asset, label="Texture adapter input")
            verify(record.output_asset, label="Texture adapter output")
            for unit in record.unit_artifacts:
                for artifact in unit.artifacts:
                    verify(artifact, label=f"Texture unit artifact {unit.unit_id}")
            for artifact in record.evidence_artifacts:
                verify(artifact, label="Texture adapter evidence")
    if cleanup is not None:
        for artifact in cleanup.retained_artifacts:
            verify(artifact, label="Texture cleanup retained artifact")
    if evidence is not None:
        for unit in evidence.unit_evidence:
            for artifact in (*unit.source_images, *unit.candidate_images):
                verify(artifact, label=f"Texture visual evidence {unit.unit_id}")
        for artifact in evidence.static_evidence:
            verify(artifact, label="Texture static evidence")
    if readback is not None:
        verify(readback.saved_stage, label="Texture saved stage")
        for artifact in readback.verification_artifacts:
            verify(artifact, label="Texture readback verification artifact")
    if review is not None:
        for artifact in review.inspected_visual_artifacts:
            verify(artifact, label="Texture reviewed visual artifact")
    if publication is not None:
        verify(publication.published_asset, label="Texture published asset")
        for artifact in publication.verification_artifacts:
            verify(artifact, label="Texture publication verification artifact")


def _texture_agentic_result_payload(
    *,
    run_dir: Path,
    request_path: Path,
    prompt_path: Path,
    preparation_binding: Any,
    accepted: Any,
    accepted_binding: Any,
    ledger: Any,
    ledger_binding: Any,
    evidence_binding: Any | None,
    review_binding: Any | None,
    publication_binding: Any | None,
    terminal: Any,
    terminal_binding: Any,
    resumed: bool,
) -> dict[str, Any]:
    """Build the sole workflow-result projection from verified terminal facts."""

    return {
        "schema_version": "content-workflow-cli.texture-agentic-result.v1",
        "status": terminal.disposition,
        "run_dir": str(run_dir),
        "request_path": str(request_path),
        "prompt_path": str(prompt_path),
        "preparation": preparation_binding.model_dump(mode="json"),
        "proposed_plan": accepted.proposal.model_dump(mode="json"),
        "accepted_plan": accepted_binding.model_dump(mode="json"),
        "adapter_ledger": ledger_binding.model_dump(mode="json"),
        "candidate": ledger.final_candidate.model_dump(mode="json"),
        "evidence": (
            evidence_binding.model_dump(mode="json")
            if evidence_binding is not None
            else None
        ),
        "review": (
            review_binding.model_dump(mode="json")
            if review_binding is not None
            else None
        ),
        "publication": (
            publication_binding.model_dump(mode="json")
            if publication_binding is not None
            else None
        ),
        "terminal_receipt": terminal_binding.model_dump(mode="json"),
        "prepared_unit_ids": list(accepted.plan.unit_ids),
        "selected_actions": {
            action: list(accepted.plan.units_for(action))
            for action in (
                "preserve",
                "generate",
                "apply_provided",
                "defer",
                "reject",
            )
        },
        "execution_owner": "texture-domain-wrapper",
        "resumed": resumed,
    }


def _load_or_rebuild_texture_agentic_result(
    *,
    result_path: Path,
    run_dir: Path,
    request_path: Path,
    prompt_path: Path,
    preparation_binding: Any,
    accepted: Any,
    accepted_binding: Any,
    ledger: Any,
    ledger_binding: Any,
    evidence_binding: Any | None,
    review_binding: Any | None,
    publication_binding: Any | None,
    terminal: Any,
    terminal_binding: Any,
) -> dict[str, Any]:
    """Validate a result projection or rebuild it from a verified terminal."""

    if result_path.exists() or result_path.is_symlink():
        if not result_path.is_file() or result_path.is_symlink():
            raise ValueError("Texture terminal workflow result must be a regular file")
        result = load_json(result_path)
        if not isinstance(result, dict) or not isinstance(result.get("resumed"), bool):
            raise ValueError("Texture terminal workflow result must be an object")
        expected_result = _texture_agentic_result_payload(
            run_dir=run_dir,
            request_path=request_path,
            prompt_path=prompt_path,
            preparation_binding=preparation_binding,
            accepted=accepted,
            accepted_binding=accepted_binding,
            ledger=ledger,
            ledger_binding=ledger_binding,
            evidence_binding=evidence_binding,
            review_binding=review_binding,
            publication_binding=publication_binding,
            terminal=terminal,
            terminal_binding=terminal_binding,
            resumed=result["resumed"],
        )
        if result != expected_result:
            raise ValueError(
                "Texture terminal workflow result is stale or inconsistent"
            )
        return result

    result = _texture_agentic_result_payload(
        run_dir=run_dir,
        request_path=request_path,
        prompt_path=prompt_path,
        preparation_binding=preparation_binding,
        accepted=accepted,
        accepted_binding=accepted_binding,
        ledger=ledger,
        ledger_binding=ledger_binding,
        evidence_binding=evidence_binding,
        review_binding=review_binding,
        publication_binding=publication_binding,
        terminal=terminal,
        terminal_binding=terminal_binding,
        resumed=True,
    )
    atomic_write_json(result_path, result)
    return result


def _validate_texture_terminal_replay_semantics(
    *,
    terminal: Any,
    accepted: Any,
    ledger: Any,
    evidence: Any | None,
    evidence_binding: Any | None,
    readback_binding: Any | None,
    review: Any | None,
    review_binding: Any | None,
    publication: Any | None,
    publication_binding: Any | None,
) -> None:
    """Replay the semantic publication gates after exact byte validation."""

    if review is not None:
        if evidence is None or evidence_binding is None:
            raise ValueError("Texture terminal review requires exact evidence")
        if (
            tuple(item.unit_id for item in review.unit_reviews)
            != accepted.plan.unit_ids
        ):
            raise ValueError("Texture terminal review is stale")
        required_visuals = tuple(
            artifact
            for item in evidence.unit_evidence
            for artifact in (*item.source_images, *item.candidate_images)
        )
        if review.inspected_visual_artifacts != required_visuals:
            raise ValueError("Texture terminal review did not inspect exact visuals")

    if terminal.disposition == "published":
        if ledger.unresolved_unit_ids:
            raise ValueError("unresolved Texture units block publication")
        if evidence is None or evidence_binding is None or readback_binding is None:
            raise ValueError("Texture publication requires exact evidence")
        if review is None or review_binding is None or not review.accepted:
            raise ValueError("Texture publication requires an accepted separate review")
        if publication is None or publication_binding is None:
            raise ValueError("Texture publication requires a publication binding")
        if (
            terminal.saved_stage_readback != evidence.saved_stage_readback
            or publication.saved_stage_readback != readback_binding
        ):
            raise ValueError("Texture terminal publication is stale")
    elif publication is not None or publication_binding is not None:
        raise ValueError("non-published Texture terminal cannot bind publication")
    if terminal.disposition == "rejected" and (review is None or review.accepted):
        raise ValueError("rejected Texture terminal requires a rejecting review")


def _validate_texture_terminal_replay_evidence(
    *,
    accepted: Any,
    accepted_binding: Any,
    ledger: Any,
    ledger_binding: Any,
    readback: Any | None,
    readback_binding: Any | None,
    evidence: Any | None,
    evidence_binding: Any | None,
) -> None:
    """Reapply the first-run readback and evidence validators on replay."""

    from content_agent_workflows.texture import (
        validate_texture_agentic_evidence,
        validate_texture_agentic_saved_stage_readback,
    )

    if readback is not None:
        if readback_binding is None:
            raise ValueError("Texture terminal readback omitted its binding")
        validate_texture_agentic_saved_stage_readback(
            readback,
            readback_binding=readback_binding,
            accepted=accepted,
            accepted_binding=accepted_binding,
            ledger=ledger,
            ledger_binding=ledger_binding,
        )
    if evidence is not None:
        if evidence_binding is None or readback is None or readback_binding is None:
            raise ValueError(
                "Texture terminal evidence omitted its saved-stage readback"
            )
        validate_texture_agentic_evidence(
            evidence,
            evidence_binding=evidence_binding,
            accepted=accepted,
            accepted_binding=accepted_binding,
            ledger=ledger,
            ledger_binding=ledger_binding,
            readback=readback,
            readback_binding=readback_binding,
        )
        _validate_texture_agentic_ovrtx_metadata(
            evidence.renderer_metadata,
            expected_directions=accepted.plan.evidence.required_views,
        )


def _replay_texture_agentic_terminal_result(
    *,
    request: Any,
    request_path: Path,
    prompt_path: Path,
    preparation_binding: Any,
    accepted: Any,
    accepted_binding: Any,
) -> dict[str, Any] | None:
    """Verify and return an existing terminal result without re-execution."""

    from content_agent_workflows.common.domain_execution import (
        ExecutionArtifactBinding,
    )
    from content_agent_workflows.texture import (
        TextureAdapterCallLedger,
        TextureAgenticCleanupReceipt,
        TextureAgenticEvidenceReceipt,
        TextureAgenticPublicationReceipt,
        TextureAgenticReviewReceipt,
        TextureAgenticSavedStageReadback,
        TextureAgenticTerminalReceipt,
        bind_texture_agentic_artifact,
    )

    run_dir = request.output_dir.expanduser().resolve()
    terminal_path = run_dir / "texture_terminal_receipt.json"
    result_path = run_dir / "workflow_result.json"
    if not (terminal_path.exists() or terminal_path.is_symlink()) and not (
        result_path.exists() or result_path.is_symlink()
    ):
        return None
    if not terminal_path.is_file() or terminal_path.is_symlink():
        raise ValueError("Texture terminal resume requires a regular terminal receipt")

    def verify(raw: object, *, label: str) -> ExecutionArtifactBinding:
        binding = ExecutionArtifactBinding.model_validate(raw)
        if bind_texture_agentic_artifact(binding.path) != binding:
            raise ValueError(f"{label} bytes changed before terminal resume")
        return binding

    terminal_binding = bind_texture_agentic_artifact(terminal_path)
    terminal = TextureAgenticTerminalReceipt.model_validate(load_json(terminal_path))
    request_binding = bind_texture_agentic_artifact(request_path)
    if (
        terminal.request != request_binding
        or terminal.preparation != preparation_binding
        or terminal.proposed_plan != accepted.proposal
        or terminal.accepted_plan != accepted_binding
    ):
        raise ValueError("Texture terminal receipt binds another accepted request")

    ledger_binding = verify(terminal.adapter_ledger, label="Texture adapter ledger")
    ledger = TextureAdapterCallLedger.model_validate(load_json(ledger_binding.path))
    if (
        ledger.accepted_plan != accepted_binding
        or ledger.preparation != preparation_binding
        or ledger.source != accepted.source
        or terminal.candidate != ledger.final_candidate
    ):
        raise ValueError("Texture terminal adapter ledger is stale")
    _verify_texture_terminal_nested_artifacts(verify=verify, ledger=ledger)

    cleanup_binding = verify(terminal.cleanup, label="Texture cleanup")
    cleanup = TextureAgenticCleanupReceipt.model_validate(
        load_json(cleanup_binding.path)
    )
    if (
        cleanup.accepted_plan != accepted_binding
        or cleanup.adapter_ledger != ledger_binding
        or cleanup.candidate != ledger.final_candidate
        or cleanup.status != terminal.cleanup_status
    ):
        raise ValueError("Texture terminal cleanup is stale")
    _verify_texture_terminal_nested_artifacts(verify=verify, cleanup=cleanup)

    evidence_binding = terminal.evidence
    readback_binding = terminal.saved_stage_readback
    review_binding = terminal.review
    publication_binding = terminal.publication
    evidence = None
    readback = None
    review = None
    publication = None
    if evidence_binding is not None:
        evidence_binding = verify(evidence_binding, label="Texture evidence")
        evidence = TextureAgenticEvidenceReceipt.model_validate(
            load_json(evidence_binding.path)
        )
        if (
            evidence.accepted_plan != accepted_binding
            or evidence.adapter_ledger != ledger_binding
            or evidence.candidate != ledger.final_candidate
            or evidence.saved_stage_readback != readback_binding
        ):
            raise ValueError("Texture terminal evidence is stale")
        _verify_texture_terminal_nested_artifacts(verify=verify, evidence=evidence)
    if readback_binding is not None:
        readback_binding = verify(
            readback_binding, label="Texture saved-stage readback"
        )
        readback = TextureAgenticSavedStageReadback.model_validate(
            load_json(readback_binding.path)
        )
        if (
            readback.accepted_plan != accepted_binding
            or readback.adapter_ledger != ledger_binding
            or readback.candidate != ledger.final_candidate
            or readback.scope_plan_digest != accepted.plan.scope_plan_digest
        ):
            raise ValueError("Texture terminal saved-stage readback is stale")
        _verify_texture_terminal_nested_artifacts(verify=verify, readback=readback)
    if review_binding is not None:
        review_binding = verify(review_binding, label="Texture review")
        review = TextureAgenticReviewReceipt.model_validate(
            load_json(review_binding.path)
        )
        if (
            review.accepted_plan != accepted_binding
            or review.adapter_ledger != ledger_binding
            or review.evidence != evidence_binding
            or review.candidate != ledger.final_candidate
            or review.plan_digest != accepted.proposal_digest
        ):
            raise ValueError("Texture terminal review is stale")
        _verify_texture_terminal_nested_artifacts(verify=verify, review=review)
    if publication_binding is not None:
        publication_binding = verify(publication_binding, label="Texture publication")
        publication = TextureAgenticPublicationReceipt.model_validate(
            load_json(publication_binding.path)
        )
        if (
            publication.accepted_plan != accepted_binding
            or publication.adapter_ledger != ledger_binding
            or publication.candidate != ledger.final_candidate
            or publication.evidence != evidence_binding
            or publication.saved_stage_readback != readback_binding
            or publication.review != review_binding
        ):
            raise ValueError("Texture terminal publication is stale")
        _verify_texture_terminal_nested_artifacts(
            verify=verify,
            publication=publication,
        )

    _validate_texture_terminal_replay_evidence(
        accepted=accepted,
        accepted_binding=accepted_binding,
        ledger=ledger,
        ledger_binding=ledger_binding,
        readback=readback,
        readback_binding=readback_binding,
        evidence=evidence,
        evidence_binding=evidence_binding,
    )
    _validate_texture_terminal_replay_semantics(
        terminal=terminal,
        accepted=accepted,
        ledger=ledger,
        evidence=evidence,
        evidence_binding=evidence_binding,
        readback_binding=readback_binding,
        review=review,
        review_binding=review_binding,
        publication=publication,
        publication_binding=publication_binding,
    )

    return _load_or_rebuild_texture_agentic_result(
        result_path=result_path,
        run_dir=run_dir,
        request_path=request_path,
        prompt_path=prompt_path,
        preparation_binding=preparation_binding,
        accepted=accepted,
        accepted_binding=accepted_binding,
        ledger=ledger,
        ledger_binding=ledger_binding,
        evidence_binding=evidence_binding,
        review_binding=review_binding,
        publication_binding=publication_binding,
        terminal=terminal,
        terminal_binding=terminal_binding,
    )


def _finish_texture_plan_handoff(
    *,
    request: TextureWorkflowRequest,
    runtime: TextureRuntimeConfig,
    request_path: Path,
    prompt_path: Path,
    preparation_binding: Any,
    accepted: Any,
    accepted_binding: Any,
    trace_writer: TraceWriter,
    resumed: bool,
) -> int:
    """Record plan acceptance, then continue through outer-owned execution."""

    terminal_result = _replay_texture_agentic_terminal_result(
        request=request,
        request_path=request_path,
        prompt_path=prompt_path,
        preparation_binding=preparation_binding,
        accepted=accepted,
        accepted_binding=accepted_binding,
    )
    if terminal_result is not None:
        _print_texture_plan_handoff(
            terminal_result,
            json_output=runtime.json_output,
        )
        return 0 if terminal_result["status"] == "published" else 2

    trace_writer.write(
        "plan_accepted",
        phase="texture_plan_only",
        summary="Outer wrapper validated and froze the exact child Texture plan.",
        artifacts=[
            preparation_binding.path,
            accepted.proposal.path,
            accepted_binding.path,
        ],
        data={"status": "accepted_plan", "resumed": resumed},
    )
    companion_handoff = _prepare_texture_companion_generation_handoff(
        accepted=accepted,
        accepted_binding=accepted_binding,
        run_dir=request.output_dir.expanduser().resolve(),
    )
    if companion_handoff is not None:
        handoff, handoff_binding, ready = companion_handoff
        if not ready:
            runner_guidance = ""
            if runtime.runner == RUNNER_CLAUDE:
                runner_guidance = (
                    " --runner claude selects the plan/review child only and "
                    "does not provide outer image generation. If the outer "
                    "coordinator has no image generator, start a fresh run with "
                    "both --texture-backend <name> and "
                    "--texture-agent-url <url>; this frozen run cannot change "
                    "provider."
                )
            print(
                "Warning: the accepted Texture plan selected coding-agent "
                "companion generation, so the outer coordinator must record "
                "the exact image-generation result before resume."
                f"{runner_guidance}",
                file=sys.stderr,
            )
            trace_writer.write(
                "companion_generation_requested",
                phase="texture_agentic_execution",
                summary=(
                    "Outer execution paused for exactly the selected coding-agent "
                    "companion image-generation attempts."
                ),
                artifacts=[handoff_binding.path],
                data={
                    "status": "awaiting_companion_generation",
                    "unit_ids": [item["unit_id"] for item in handoff["units"]],
                },
            )
            build_trace(request.output_dir)
            _print_texture_plan_handoff(
                {
                    "schema_version": ("content-workflow-cli.texture-plan-handoff.v1"),
                    "status": "awaiting_companion_generation",
                    "run_dir": str(request.output_dir.expanduser().resolve()),
                    "request_path": str(request_path),
                    "prompt_path": str(prompt_path),
                    "accepted_plan": accepted_binding.model_dump(mode="json"),
                    "companion_generation_handoff": handoff_binding.model_dump(
                        mode="json"
                    ),
                },
                json_output=runtime.json_output,
            )
            return TEXTURE_CONDITIONAL_EXIT_CODE
        trace_writer.write(
            "companion_generation_ready",
            phase="texture_agentic_execution",
            summary=(
                "Found one recorded coding-agent companion result for every selected "
                "generation unit."
            ),
            artifacts=[
                handoff_binding.path,
                *(str(item["manifest_path"]) for item in handoff["units"]),
            ],
            data={"unit_ids": [item["unit_id"] for item in handoff["units"]]},
        )
    return _execute_texture_accepted_plan(
        request=request,
        runtime=runtime,
        request_path=request_path,
        prompt_path=prompt_path,
        preparation_binding=preparation_binding,
        accepted=accepted,
        accepted_binding=accepted_binding,
        trace_writer=trace_writer,
        resumed=resumed,
        companion_handoff=(
            companion_handoff[0] if companion_handoff is not None else None
        ),
    )


def _prepare_texture_companion_generation_handoff(
    *,
    accepted: Any,
    accepted_binding: Any,
    run_dir: Path,
) -> tuple[dict[str, Any], Any, bool] | None:
    """Freeze selected companion prompts and wait for outer-recorded results."""

    from content_agent_workflows.texture import bind_texture_agentic_artifact

    units = tuple(
        item
        for item in accepted.plan.dispositions
        if item.action == "generate"
        and item.generator_inputs is not None
        and item.generator_inputs.backend == TEXTURE_COMPANION_IMAGE_BACKEND
    )
    if not units:
        return None
    if len(units) != len(accepted.plan.units_for("generate")):
        raise ValueError(
            "one-attempt Texture generation cannot mix companion and service backends"
        )
    handoff_units: list[dict[str, Any]] = []
    all_results_present = True
    for unit in units:
        inputs = unit.generator_inputs
        assert inputs is not None
        attempt_dir = _prepare_texture_companion_attempt_dir(run_dir, unit.unit_id)
        prompt_path = attempt_dir / "prompt.txt"
        if prompt_path.exists():
            if prompt_path.is_symlink() or not prompt_path.is_file():
                raise ValueError("Texture companion prompt must be a regular file")
            if prompt_path.read_text(encoding="utf-8") != inputs.prompt:
                raise ValueError(
                    "Texture companion prompt changed after plan acceptance"
                )
        else:
            atomic_write_text(prompt_path, inputs.prompt)
        prompt_binding = bind_texture_agentic_artifact(prompt_path)
        output_path = attempt_dir / "raw.png"
        manifest_path = attempt_dir / "image_generation.json"
        if not manifest_path.is_file() or manifest_path.is_symlink():
            all_results_present = False
        handoff_units.append(
            {
                "unit_id": unit.unit_id,
                "prompt": prompt_binding.model_dump(mode="json"),
                "conditioning_images": [
                    item.model_dump(mode="json") for item in inputs.reference_artifacts
                ],
                "texture_size": inputs.texture_size,
                "output_path": str(output_path),
                "manifest_path": str(manifest_path),
                "tool_id": "companion-image-generation",
            }
        )
    handoff = {
        "schema_version": TEXTURE_COMPANION_GENERATION_HANDOFF_SCHEMA_VERSION,
        "accepted_plan": accepted_binding.model_dump(mode="json"),
        "backend": TEXTURE_COMPANION_IMAGE_BACKEND,
        "units": handoff_units,
        "fallback": False,
    }
    handoff_path = run_dir / "texture_companion_generation_handoff.json"
    if handoff_path.exists():
        if handoff_path.is_symlink() or not handoff_path.is_file():
            raise ValueError("Texture companion handoff must be a regular file")
        if load_json(handoff_path) != handoff:
            raise ValueError("Texture companion handoff changed after plan acceptance")
    else:
        atomic_write_json(handoff_path, handoff)
    return (
        handoff,
        bind_texture_agentic_artifact(handoff_path),
        all_results_present,
    )


def _prepare_texture_companion_attempt_dir(run_dir: Path, unit_id: str) -> Path:
    """Create the bounded companion attempt without following directory symlinks."""

    attempt_dir = run_dir / "companion_generation" / unit_id / "attempt-001"
    for directory in (
        run_dir / "companion_generation",
        run_dir / "companion_generation" / unit_id,
        attempt_dir,
    ):
        if directory.exists() or directory.is_symlink():
            if directory.is_symlink() or not directory.is_dir():
                raise ValueError(
                    "Texture companion attempt path must be a regular directory"
                )
            continue
        directory.mkdir()
    return attempt_dir


def _texture_agentic_adapter_factories(
    accepted: Any,
    accepted_binding: Any,
    runtime: TextureRuntimeConfig,
    *,
    preparation: Any,
    companion_handoff: Mapping[str, Any] | None = None,
) -> dict[
    Literal["generate", "apply_provided"],
    Callable[[], TextureAgenticExecutionAdapter],
]:
    """Build lazy factories for exactly the mutating actions in the plan."""

    from content_agent_workflows.texture import (
        ProvidedImageTextureApplyLeaf,
        TextureAgenticGeneratorLeafAdapter,
        TextureAgentServiceClient,
    )

    factories: dict[
        Literal["generate", "apply_provided"],
        Callable[[], TextureAgenticExecutionAdapter],
    ] = {}
    generate_units = tuple(
        item for item in accepted.plan.dispositions if item.action == "generate"
    )
    if generate_units:
        backends = {
            item.generator_inputs.backend
            for item in generate_units
            if item.generator_inputs is not None
        }
        if len(backends) != 1:
            raise ValueError(
                "one-attempt Texture generation requires one selected backend"
            )
        backend = next(iter(backends))

        if backend == TEXTURE_COMPANION_IMAGE_BACKEND:

            def companion_leaf() -> Any:
                from .texture_capability_runner import (
                    _RecordedCompanionGeneratorLeaf,
                )

                if companion_handoff is None:
                    raise ValueError(
                        "selected coding-agent companion generation has no recorded "
                        "handoff"
                    )
                return _RecordedCompanionGeneratorLeaf(
                    companion_handoff,
                    accepted_plan=accepted_binding,
                )

            leaf_factory = companion_leaf
            adapter_id = (
                "coding_agent_companion:image-generation+texture.apply-provided.v1"
            )
        else:

            def service_leaf() -> Any:
                from .texture_capability_runner import _ServiceGeneratorLeaf

                token = None
                if runtime.texture_agent_token_env is not None:
                    token = os.getenv(runtime.texture_agent_token_env)
                    if not token:
                        raise ValueError(
                            f"{runtime.texture_agent_token_env} is not set or is empty"
                        )
                client = TextureAgentServiceClient(
                    base_url=_require_texture_agent_url(runtime),
                    timeout_seconds=runtime.texture_timeout_seconds,
                    poll_interval_seconds=runtime.texture_poll_interval_seconds,
                    max_status_poll_failures=5,
                    token=token,
                )
                return _ServiceGeneratorLeaf(
                    provider_id=backend,
                    client=client,
                    endpoint=preparation.request.metadata.get("texture_endpoint"),
                )

            leaf_factory = service_leaf
            adapter_id = f"{backend}:texture-agent-service.execute.v1"

        factories["generate"] = lambda: TextureAgenticGeneratorLeafAdapter(
            action="generate",
            leaf_factory=leaf_factory,
            adapter_id=adapter_id,
        )
    if accepted.plan.units_for("apply_provided"):
        factories["apply_provided"] = lambda: TextureAgenticGeneratorLeafAdapter(
            action="apply_provided",
            leaf_factory=ProvidedImageTextureApplyLeaf,
            adapter_id=("outer_provided_image_apply:texture.apply-provided.v1"),
        )
    return factories


def _texture_agentic_failure_ledger(
    *,
    accepted: Any,
    accepted_binding: Any,
    preparation_binding: Any,
    output_path: Path,
) -> tuple[Any, Any]:
    from content_agent_workflows.texture import (
        TextureAdapterCallLedger,
        bind_texture_agentic_artifact,
    )

    ledger = TextureAdapterCallLedger(
        accepted_plan=accepted_binding,
        preparation=preparation_binding,
        source=accepted.source,
        final_candidate=accepted.source,
        unresolved_unit_ids=accepted.plan.unit_ids,
    )
    atomic_write_json(output_path, ledger)
    return ledger, bind_texture_agentic_artifact(output_path)


def _texture_agentic_readback_and_evidence(
    *,
    request: Any,
    accepted: Any,
    accepted_binding: Any,
    preparation: Any,
    ledger: Any,
    ledger_binding: Any,
    run_dir: Path,
) -> tuple[Any, Any, Any, Any]:
    """Collect full-unit readback and fresh matched OVRTX evidence."""

    from content_agent_workflows.asset_composition import bind_usd_dependency_closure
    from content_agent_workflows.texture import (
        LiveUsdCliTextureValidator,
        TextureAgenticEvidenceReceipt,
        TextureAgenticSavedStageReadback,
        TextureUnitArtifact,
        TextureWorkflowRequest,
        bind_texture_agentic_artifact,
        record_texture_agentic_evidence,
        record_texture_agentic_saved_stage_readback,
        texture_unit_material_state_digests,
        validate_texture_scope_invariants,
    )

    candidate = ledger.final_candidate
    invariant_report = validate_texture_scope_invariants(
        source_asset_path=accepted.source.path,
        output_asset_path=candidate.path,
        plan=preparation.scope_plan,
    )
    if not invariant_report.passed:
        raise ValueError("Texture candidate failed deterministic scope readback")
    preserved_ids = accepted.plan.units_for("preserve")
    source_preserved = texture_unit_material_state_digests(
        output_asset_path=accepted.source.path,
        plan=preparation.scope_plan,
        unit_ids=preserved_ids,
        normalize_texture_asset_relocations=True,
    )
    candidate_preserved = texture_unit_material_state_digests(
        output_asset_path=candidate.path,
        plan=preparation.scope_plan,
        unit_ids=preserved_ids,
        normalize_texture_asset_relocations=True,
    )
    if source_preserved != candidate_preserved:
        raise ValueError("Texture preserve disposition changed material state")
    dependencies = tuple(bind_usd_dependency_closure(candidate.path))
    if dependencies:
        raise ValueError("Texture candidate has external dependency closure")
    readback_dir = run_dir / "readback"
    readback_dir.mkdir(parents=True, exist_ok=True)
    verification_path = atomic_write_json(
        readback_dir / "saved_stage_verification.json",
        {
            "schema_version": (
                "content-workflow-cli.texture-agentic-readback-verification.v1"
            ),
            "source": accepted.source.model_dump(mode="json"),
            "candidate": candidate.model_dump(mode="json"),
            "scope_plan_digest": accepted.plan.scope_plan_digest,
            "scope_invariants": invariant_report.model_dump(mode="json"),
            "preserved_unit_source_digests": source_preserved,
            "preserved_unit_candidate_digests": candidate_preserved,
            "candidate_dependencies": [
                item.model_dump(mode="json") for item in dependencies
            ],
        },
    )
    readback = TextureAgenticSavedStageReadback(
        accepted_plan=accepted_binding,
        adapter_ledger=ledger_binding,
        candidate=candidate,
        saved_stage=candidate,
        scope_plan_digest=accepted.plan.scope_plan_digest,
        verification_artifacts=(bind_texture_agentic_artifact(verification_path),),
    )
    readback, readback_binding = record_texture_agentic_saved_stage_readback(
        readback,
        accepted=accepted,
        accepted_binding=accepted_binding,
        ledger=ledger,
        ledger_binding=ledger_binding,
        output_path=readback_dir / "texture_saved_stage_readback.json",
    )

    artifacts_by_unit: dict[str, TextureUnitArtifact] = {}
    for record in ledger.records:
        for item in record.unit_artifacts:
            artifacts_by_unit[item.unit_id] = TextureUnitArtifact(
                unit_id=item.unit_id,
                artifact_paths=tuple(binding.path for binding in item.artifacts),
                metadata={**item.metadata, "action": record.action},
            )
    action_by_id = {item.unit_id: item.action for item in accepted.plan.dispositions}
    for unit_id in accepted.plan.unit_ids:
        artifacts_by_unit.setdefault(
            unit_id,
            TextureUnitArtifact(
                unit_id=unit_id,
                artifact_paths=(candidate.path,),
                metadata={"action": action_by_id[unit_id], "mutation": False},
            ),
        )
    evidence_dir = run_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    collector = LiveUsdCliTextureValidator(
        assessor=None,
        validation_policy_id="content-workflow-cli.texture-agentic-evidence.v1",
        directions=accepted.plan.evidence.required_views,
    )
    workflow_request = TextureWorkflowRequest(
        source_asset=accepted.source.path,
        output_dir=evidence_dir,
        intent=preparation.request.intent,
        reference_artifacts=accepted.plan.reference_artifacts,
        metadata={"source_asset_binding": accepted.source.model_dump(mode="json")},
    )
    provided_images = tuple(
        image
        for disposition in accepted.plan.dispositions
        if disposition.generator_inputs is not None
        for image in disposition.generator_inputs.provided_images
    )
    unit_evidence, static_evidence, raw_renderer = collector.collect_candidate_evidence(
        request=workflow_request,
        plan=preparation.scope_plan,
        output_asset_path=candidate.path,
        unit_artifacts=artifacts_by_unit,
        unit_ids=accepted.plan.unit_ids,
        output_dir=evidence_dir,
        reference_artifacts=accepted.plan.reference_artifacts,
        provided_images=provided_images,
    )
    renderer_metadata = dict(raw_renderer)
    _validate_texture_agentic_ovrtx_metadata(
        renderer_metadata,
        expected_directions=accepted.plan.evidence.required_views,
    )
    evidence = TextureAgenticEvidenceReceipt(
        accepted_plan=accepted_binding,
        adapter_ledger=ledger_binding,
        candidate=candidate,
        view_names=accepted.plan.evidence.required_views,
        unit_evidence=unit_evidence,
        static_evidence=static_evidence,
        saved_stage_readback=readback_binding,
        renderer_metadata=renderer_metadata,
    )
    evidence, evidence_binding = record_texture_agentic_evidence(
        evidence,
        accepted=accepted,
        accepted_binding=accepted_binding,
        ledger=ledger,
        ledger_binding=ledger_binding,
        readback=readback,
        readback_binding=readback_binding,
        output_path=evidence_dir / "texture_agentic_evidence.json",
    )
    return readback, readback_binding, evidence, evidence_binding


def _validate_texture_agentic_ovrtx_metadata(
    renderer_metadata: Mapping[str, Any],
    *,
    expected_directions: tuple[str, ...],
) -> None:
    if (
        renderer_metadata.get("provider") != "usd-cli-ovrtx"
        or renderer_metadata.get("renderer") != "ovrtx"
        or renderer_metadata.get("current_run") is not True
        or renderer_metadata.get("directions") != list(expected_directions)
        or renderer_metadata.get("per_image_provenance") != "bound-response-camera-v1"
    ):
        raise ValueError("Texture evidence has invalid OVRTX provenance")


def _load_texture_agentic_readback_and_evidence(
    *,
    accepted: Any,
    accepted_binding: Any,
    ledger: Any,
    ledger_binding: Any,
    run_dir: Path,
) -> tuple[Any, Any, Any, Any] | None:
    """Revalidate and reuse a complete persisted readback/evidence pair."""

    from content_agent_workflows.texture import (
        TextureAgenticEvidenceReceipt,
        TextureAgenticSavedStageReadback,
        bind_texture_agentic_artifact,
        validate_texture_agentic_evidence,
        validate_texture_agentic_saved_stage_readback,
    )

    readback_path = run_dir / "readback" / "texture_saved_stage_readback.json"
    evidence_path = run_dir / "evidence" / "texture_agentic_evidence.json"
    readback_present = readback_path.exists() or readback_path.is_symlink()
    evidence_present = evidence_path.exists() or evidence_path.is_symlink()
    if not readback_present and not evidence_present:
        return None
    if evidence_present and not readback_present:
        raise ValueError("Texture evidence exists without its saved-stage readback")

    readback_binding = bind_texture_agentic_artifact(readback_path)
    readback = TextureAgenticSavedStageReadback.model_validate(load_json(readback_path))
    validate_texture_agentic_saved_stage_readback(
        readback,
        readback_binding=readback_binding,
        accepted=accepted,
        accepted_binding=accepted_binding,
        ledger=ledger,
        ledger_binding=ledger_binding,
    )
    if not evidence_present:
        # A crash between deterministic readback and OVRTX collection has no
        # complete evidence to reuse. The caller may continue collection.
        return None

    evidence_binding = bind_texture_agentic_artifact(evidence_path)
    evidence = TextureAgenticEvidenceReceipt.model_validate(load_json(evidence_path))
    validate_texture_agentic_evidence(
        evidence,
        evidence_binding=evidence_binding,
        accepted=accepted,
        accepted_binding=accepted_binding,
        ledger=ledger,
        ledger_binding=ledger_binding,
        readback=readback,
        readback_binding=readback_binding,
    )
    _validate_texture_agentic_ovrtx_metadata(
        evidence.renderer_metadata,
        expected_directions=accepted.plan.evidence.required_views,
    )
    return readback, readback_binding, evidence, evidence_binding


def _load_or_collect_texture_agentic_readback_and_evidence(
    *,
    request: Any,
    accepted: Any,
    accepted_binding: Any,
    preparation: Any,
    ledger: Any,
    ledger_binding: Any,
    run_dir: Path,
) -> tuple[Any, Any, Any, Any]:
    persisted = _load_texture_agentic_readback_and_evidence(
        accepted=accepted,
        accepted_binding=accepted_binding,
        ledger=ledger,
        ledger_binding=ledger_binding,
        run_dir=run_dir,
    )
    if persisted is not None:
        return persisted
    return _texture_agentic_readback_and_evidence(
        request=request,
        accepted=accepted,
        accepted_binding=accepted_binding,
        preparation=preparation,
        ledger=ledger,
        ledger_binding=ledger_binding,
        run_dir=run_dir,
    )


def _obtain_texture_agentic_review(
    *,
    request: Any,
    runtime: TextureRuntimeConfig,
    accepted: Any,
    accepted_binding: Any,
    preparation: Any,
    preparation_binding: Any,
    ledger: Any,
    ledger_binding: Any,
    evidence: Any,
    evidence_binding: Any,
    run_dir: Path,
) -> tuple[Any, Any]:
    """Launch a separate multimodal review child and validate exact custody."""

    from content_agent_workflows.texture import (
        TextureAgenticReviewReceipt,
        bind_texture_agentic_artifact,
        record_texture_agentic_review,
        validate_texture_accepted_plan,
        validate_texture_preparation,
    )

    proposal_path = run_dir / "texture_review_proposal.json"
    child_output_path = run_dir / "raw" / "texture_review_child_output.jsonl"
    child_final_path = run_dir / "raw" / "texture_review_child_final.json"
    if proposal_path.is_symlink():
        raise ValueError("Texture review proposal must be a regular file")
    materialize_child_output = False
    if not proposal_path.is_file():
        if child_final_path.exists() or child_final_path.is_symlink():
            materialize_child_output = True
        else:
            if _texture_reasoning_attempt_started(
                run_dir,
                bridge_artifact_prefix="texture_review_only",
                child_output_name="texture_review_child_output.jsonl",
            ):
                raise RuntimeError(
                    "Texture review resume found an incomplete provider turn without "
                    "adoptable structured output; start a fresh run rather than "
                    "repeating it"
                )
            capability_inventory, domain_policy_bounds = (
                _prepare_texture_review_child_launch_contract(
                    request,
                    runtime=runtime,
                    accepted_binding=accepted_binding,
                    ledger_binding=ledger_binding,
                    evidence_binding=evidence_binding,
                )
            )
            child_config = _texture_child_runtime_config(
                request,
                runtime,
                capability_inventory=capability_inventory,
                domain_policy_bounds=domain_policy_bounds,
            )
            review_attachments = _texture_review_image_attachments(
                accepted=accepted,
                evidence=evidence,
            )
            visual_paths = [
                Path(binding.path) for _role, _unit_id, binding in review_attachments
            ]
            child_config = replace(child_config, reference_images=visual_paths)
            review_prompt = _build_texture_review_prompt(
                request,
                accepted=accepted,
                accepted_binding=accepted_binding,
                ledger=ledger,
                ledger_binding=ledger_binding,
                evidence=evidence,
                evidence_binding=evidence_binding,
            )
            prompt_path = run_dir / "prompts" / "texture_review_only.md"
            atomic_write_text(prompt_path, review_prompt)
            returncode = run_child_agent(
                config=child_config,
                prompt=review_prompt,
                run_dir=run_dir,
                child_output_path=child_output_path,
                child_final_path=child_final_path,
                scene_service=None,
                bridge_artifact_prefix="texture_review_only",
                output_schema=_texture_review_output_schema(),
                stage_skills=False,
                tools_disabled=True,
            )
            if returncode != 0:
                raise RuntimeError(
                    f"Texture review child exited with code {returncode}"
                )
            materialize_child_output = True
        validate_texture_preparation(
            preparation,
            preparation_binding=preparation_binding,
            expected_request=_texture_agentic_capability_request(request),
        )
        validate_texture_accepted_plan(
            accepted,
            accepted_binding=accepted_binding,
            preparation=preparation,
            preparation_binding=preparation_binding,
        )
        if _prepare_texture_agentic_source_request(request, resume=True) != request:
            raise RuntimeError("Texture review child changed frozen source preparation")
        for binding, label in (
            (accepted_binding, "accepted Texture plan"),
            (ledger_binding, "Texture adapter ledger"),
            (evidence_binding, "Texture evidence"),
            (ledger.final_candidate, "Texture candidate"),
        ):
            if bind_texture_agentic_artifact(binding.path) != binding:
                raise RuntimeError(f"Texture review child modified {label}")
        if materialize_child_output:
            _materialize_texture_review_child_output(
                run_dir=run_dir,
                child_final_path=child_final_path,
                proposal_path=proposal_path,
                runtime=runtime,
            )
    if not proposal_path.is_file():
        raise RuntimeError("Texture review child did not write its review proposal")
    _validate_texture_review_child_provenance(
        run_dir=run_dir,
        child_final_path=child_final_path,
        proposal_path=proposal_path,
        runtime=runtime,
    )
    review = TextureAgenticReviewReceipt.model_validate(load_json(proposal_path))
    return cast(
        tuple[Any, Any],
        record_texture_agentic_review(
            review,
            accepted=accepted,
            accepted_binding=accepted_binding,
            ledger=ledger,
            ledger_binding=ledger_binding,
            evidence=evidence,
            evidence_binding=evidence_binding,
            output_path=run_dir / "texture_review.json",
        ),
    )


def _publish_texture_agentic_candidate(
    *,
    accepted: Any,
    accepted_binding: Any,
    preparation: Any,
    ledger: Any,
    ledger_binding: Any,
    readback_binding: Any,
    evidence: Any,
    evidence_binding: Any,
    review: Any,
    review_binding: Any,
    run_dir: Path,
) -> tuple[Any, Any]:
    """Atomically publish the exact reviewed self-contained candidate."""

    from content_agent_workflows.asset_composition import bind_usd_dependency_closure
    from content_agent_workflows.texture import (
        TextureAgenticPublicationReceipt,
        bind_texture_agentic_artifact,
        record_texture_agentic_publication,
        validate_texture_scope_invariants,
    )

    if ledger.unresolved_unit_ids:
        raise ValueError("unresolved Texture units block publication")
    if not review.accepted:
        raise ValueError("Texture review rejected or left unresolved units")
    candidate = ledger.final_candidate
    dependencies = tuple(bind_usd_dependency_closure(candidate.path))
    if dependencies:
        raise ValueError("Texture publication requires self-contained candidate bytes")
    destination_dir = run_dir / "published"
    destination_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(candidate.path).suffix.lower()
    destination = destination_dir / f"textured_asset{suffix}"
    publication_created = False
    if destination.exists():
        published = bind_texture_agentic_artifact(destination)
        if published.sha256 != candidate.sha256:
            raise ValueError("existing Texture publication differs from candidate")
    else:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(destination, flags, 0o600)
        try:
            with (
                Path(candidate.path).open("rb") as source_stream,
                os.fdopen(descriptor, "wb") as destination_stream,
            ):
                descriptor = -1
                shutil.copyfileobj(source_stream, destination_stream)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        published = bind_texture_agentic_artifact(destination)
        if published.sha256 != candidate.sha256:
            destination.unlink(missing_ok=True)
            raise ValueError("published Texture bytes differ from candidate")
        publication_created = True
    try:
        readback = validate_texture_scope_invariants(
            source_asset_path=accepted.source.path,
            output_asset_path=published.path,
            plan=preparation.scope_plan,
        )
        if not readback.passed:
            raise ValueError("published Texture stage failed deterministic readback")
        verification_path = atomic_write_json(
            destination_dir / "publication_verification.json",
            {
                "schema_version": (
                    "content-workflow-cli.texture-agentic-publication-verification.v1"
                ),
                "accepted_plan": accepted_binding.model_dump(mode="json"),
                "candidate": candidate.model_dump(mode="json"),
                "published_asset": published.model_dump(mode="json"),
                "saved_stage_readback": readback_binding.model_dump(mode="json"),
                "scope_invariants": readback.model_dump(mode="json"),
                "dependency_count": len(dependencies),
            },
        )
        publication = TextureAgenticPublicationReceipt(
            accepted_plan=accepted_binding,
            adapter_ledger=ledger_binding,
            candidate=candidate,
            evidence=evidence_binding,
            saved_stage_readback=readback_binding,
            review=review_binding,
            published_asset=published,
            verification_artifacts=(bind_texture_agentic_artifact(verification_path),),
        )
        return cast(
            tuple[Any, Any],
            record_texture_agentic_publication(
                publication,
                accepted_binding=accepted_binding,
                ledger=ledger,
                ledger_binding=ledger_binding,
                evidence=evidence,
                evidence_binding=evidence_binding,
                readback_binding=readback_binding,
                review=review,
                review_binding=review_binding,
                output_path=destination_dir / "texture_publication.json",
            ),
        )
    except Exception:
        if publication_created:
            destination.unlink(missing_ok=True)
        raise


def _rollback_texture_agentic_publication(
    *,
    run_dir: Path,
    publication: Any,
    publication_binding: Any,
) -> None:
    """Remove the fixed publication projection before sealing a failed terminal."""

    publication_dir = run_dir / "published"
    if publication_dir.is_symlink() or not publication_dir.is_dir():
        raise ValueError("Texture publication rollback requires its fixed directory")
    publication_root = publication_dir.resolve()
    rollback_paths = (
        Path(publication.published_asset.path),
        *(Path(item.path) for item in publication.verification_artifacts),
        Path(publication_binding.path),
    )
    resolved_paths: list[Path] = []
    for path in rollback_paths:
        expanded = path.expanduser()
        if expanded.is_symlink():
            raise ValueError("Texture publication rollback refuses symbolic links")
        resolved = expanded.resolve()
        if not resolved.is_relative_to(publication_root):
            raise ValueError("Texture publication rollback escaped its fixed directory")
        if resolved.exists() and not resolved.is_file():
            raise ValueError("Texture publication rollback requires regular files")
        resolved_paths.append(resolved)
    for resolved in resolved_paths:
        resolved.unlink(missing_ok=True)
    cleanup_path = run_dir / "texture_cleanup.json"
    if cleanup_path.exists() or cleanup_path.is_symlink():
        if cleanup_path.is_symlink() or not cleanup_path.is_file():
            raise ValueError("Texture cleanup rollback requires a regular file")
        cleanup_path.unlink()


def _execute_texture_accepted_plan(
    *,
    request: Any,
    runtime: TextureRuntimeConfig,
    request_path: Path,
    prompt_path: Path,
    preparation_binding: Any,
    accepted: Any,
    accepted_binding: Any,
    trace_writer: TraceWriter,
    resumed: bool,
    companion_handoff: Mapping[str, Any] | None = None,
) -> int:
    """Own execution, evidence, review, publication, cleanup, and terminal truth."""

    from content_agent_workflows.texture import (
        TextureAgenticCleanupReceipt,
        TexturePreparationPacket,
        bind_texture_agentic_artifact,
        execute_texture_agentic_plan,
        record_texture_agentic_cleanup,
        seal_texture_agentic_terminal_receipt,
    )

    run_dir = request.output_dir.expanduser().resolve()
    preparation = TexturePreparationPacket.model_validate(
        load_json(preparation_binding.path)
    )
    request_binding = bind_texture_agentic_artifact(request_path)
    ledger = ledger_binding = None
    evidence = evidence_binding = None
    review = review_binding = None
    publication = publication_binding = None
    failure_issue_code = "adapter_execution_failed"
    disposition: Literal["published", "rejected", "blocked", "failed"]
    try:
        ledger, ledger_binding = execute_texture_agentic_plan(
            accepted,
            accepted_binding=accepted_binding,
            preparation=preparation,
            preparation_binding=preparation_binding,
            adapter_factories=_texture_agentic_adapter_factories(
                accepted,
                accepted_binding,
                runtime,
                preparation=preparation,
                companion_handoff=companion_handoff,
            ),
            output_dir=run_dir / "operations",
        )
        trace_writer.write(
            "execution_completed",
            phase="texture_agentic_execution",
            summary="Outer wrapper executed exactly the selected Texture adapters.",
            artifacts=[ledger_binding.path, ledger.final_candidate.path],
            data={"adapter_calls": len(ledger.records)},
        )
        failure_issue_code = "saved_stage_evidence_failed"
        readback, readback_binding, evidence, evidence_binding = (
            _load_or_collect_texture_agentic_readback_and_evidence(
                request=request,
                accepted=accepted,
                accepted_binding=accepted_binding,
                preparation=preparation,
                ledger=ledger,
                ledger_binding=ledger_binding,
                run_dir=run_dir,
            )
        )
        failure_issue_code = "separate_review_failed"
        review, review_binding = _obtain_texture_agentic_review(
            request=request,
            runtime=runtime,
            accepted=accepted,
            accepted_binding=accepted_binding,
            preparation=preparation,
            preparation_binding=preparation_binding,
            ledger=ledger,
            ledger_binding=ledger_binding,
            evidence=evidence,
            evidence_binding=evidence_binding,
            run_dir=run_dir,
        )
        if ledger.unresolved_unit_ids or not review.accepted:
            disposition = (
                "rejected"
                if any(item.disposition == "reject" for item in review.unit_reviews)
                else "blocked"
            )
            issue_codes = tuple(
                code
                for code, present in (
                    ("unresolved_plan_unit", bool(ledger.unresolved_unit_ids)),
                    ("review_not_accepted", not review.accepted),
                )
                if present
            )
        else:
            failure_issue_code = "publication_failed"
            publication, publication_binding = _publish_texture_agentic_candidate(
                accepted=accepted,
                accepted_binding=accepted_binding,
                preparation=preparation,
                ledger=ledger,
                ledger_binding=ledger_binding,
                readback_binding=readback_binding,
                evidence=evidence,
                evidence_binding=evidence_binding,
                review=review,
                review_binding=review_binding,
                run_dir=run_dir,
            )
            disposition = "published"
            issue_codes = ()
        failure_issue_code = "terminal_finalization_failed"
        retained = tuple(
            binding
            for binding in (
                accepted_binding,
                ledger_binding,
                ledger.final_candidate,
                readback_binding,
                evidence_binding,
                review_binding,
                publication_binding,
            )
            if binding is not None
        )
        cleanup = TextureAgenticCleanupReceipt(
            accepted_plan=accepted_binding,
            adapter_ledger=ledger_binding,
            candidate=ledger.final_candidate,
            status="completed",
            retained_artifacts=retained,
        )
        cleanup, cleanup_binding = record_texture_agentic_cleanup(
            cleanup,
            accepted_binding=accepted_binding,
            ledger=ledger,
            ledger_binding=ledger_binding,
            output_path=run_dir / "texture_cleanup.json",
        )
        terminal, terminal_binding = seal_texture_agentic_terminal_receipt(
            request=request_binding,
            accepted=accepted,
            accepted_binding=accepted_binding,
            ledger=ledger,
            ledger_binding=ledger_binding,
            evidence=evidence,
            evidence_binding=evidence_binding,
            review=review,
            review_binding=review_binding,
            publication=publication,
            publication_binding=publication_binding,
            cleanup=cleanup,
            cleanup_binding=cleanup_binding,
            disposition=disposition,
            issue_codes=issue_codes,
            output_path=run_dir / "texture_terminal_receipt.json",
        )
    except Exception as exc:
        if publication is not None or publication_binding is not None:
            if publication is None or publication_binding is None:
                raise ValueError(
                    "partial Texture publication state is inconsistent"
                ) from exc
            _rollback_texture_agentic_publication(
                run_dir=run_dir,
                publication=publication,
                publication_binding=publication_binding,
            )
            publication = publication_binding = None
        if ledger is None or ledger_binding is None:
            observed_ledger_path = (
                run_dir / "operations" / "texture_adapter_ledger.json"
            )
            if observed_ledger_path.is_file() and not observed_ledger_path.is_symlink():
                from content_agent_workflows.texture import TextureAdapterCallLedger

                observed_ledger = TextureAdapterCallLedger.model_validate(
                    load_json(observed_ledger_path)
                )
                observed_binding = bind_texture_agentic_artifact(observed_ledger_path)
                if (
                    observed_ledger.accepted_plan != accepted_binding
                    or observed_ledger.preparation != preparation_binding
                    or observed_ledger.source != accepted.source
                ):
                    raise ValueError(
                        "partial Texture adapter ledger binds another execution"
                    ) from exc
                ledger, ledger_binding = observed_ledger, observed_binding
            else:
                ledger, ledger_binding = _texture_agentic_failure_ledger(
                    accepted=accepted,
                    accepted_binding=accepted_binding,
                    preparation_binding=preparation_binding,
                    output_path=run_dir / "texture_adapter_failure_ledger.json",
                )
        cleanup = TextureAgenticCleanupReceipt(
            accepted_plan=accepted_binding,
            adapter_ledger=ledger_binding,
            candidate=ledger.final_candidate,
            status="completed",
            retained_artifacts=(
                accepted_binding,
                ledger_binding,
                ledger.final_candidate,
            ),
        )
        cleanup, cleanup_binding = record_texture_agentic_cleanup(
            cleanup,
            accepted_binding=accepted_binding,
            ledger=ledger,
            ledger_binding=ledger_binding,
            output_path=run_dir / "texture_cleanup.json",
        )
        terminal, terminal_binding = seal_texture_agentic_terminal_receipt(
            request=request_binding,
            accepted=accepted,
            accepted_binding=accepted_binding,
            ledger=ledger,
            ledger_binding=ledger_binding,
            evidence=evidence,
            evidence_binding=evidence_binding,
            review=review,
            review_binding=review_binding,
            publication=None,
            publication_binding=None,
            cleanup=cleanup,
            cleanup_binding=cleanup_binding,
            disposition="failed",
            issue_codes=(failure_issue_code,),
            failure=str(exc),
            output_path=run_dir / "texture_terminal_receipt.json",
        )
    result = _texture_agentic_result_payload(
        run_dir=run_dir,
        request_path=request_path,
        prompt_path=prompt_path,
        preparation_binding=preparation_binding,
        accepted=accepted,
        accepted_binding=accepted_binding,
        ledger=ledger,
        ledger_binding=ledger_binding,
        evidence_binding=evidence_binding,
        review_binding=review_binding,
        publication_binding=publication_binding,
        terminal=terminal,
        terminal_binding=terminal_binding,
        resumed=resumed,
    )
    result_path = run_dir / "workflow_result.json"
    atomic_write_json(result_path, result)
    trace_writer.write(
        "workflow_finished",
        phase="texture_agentic_terminal",
        summary=f"Texture agentic workflow ended as {terminal.disposition}.",
        artifacts=[terminal_binding.path, str(result_path)],
        data={"status": terminal.disposition, "resumed": resumed},
    )
    build_trace(run_dir)
    _print_texture_plan_handoff(result, json_output=runtime.json_output)
    return 0 if terminal.disposition == "published" else 2


def _print_texture_plan_handoff(
    payload: Mapping[str, Any],
    *,
    json_output: bool,
) -> None:
    if json_output:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    print(f"texture_run_dir: {payload['run_dir']}")
    print(f"status: {payload['status']}")
    accepted = payload.get("accepted_plan")
    if isinstance(accepted, Mapping):
        print(f"accepted_plan: {accepted.get('path')}")


def _launch_texture_legacy_child(
    request: TextureWorkflowRequest,
    *,
    runtime: TextureRuntimeConfig,
    resume: bool,
) -> int:
    from content_agent_workflows.texture import TextureFinalizationResult

    run_dir = request.output_dir.expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    request_path = run_dir / "request.json"
    if resume:
        _preflight_resume(request, runtime=runtime)
    elif request_path.exists():
        raise FileExistsError(f"Texture request already exists: {request_path}")
    else:
        atomic_write_json(request_path, request.model_dump(mode="json"))

    parent_usd_cli_capability = None
    try:
        _write_texture_agent_launcher(run_dir, runtime, request=request)
        prompt = _build_texture_agent_prompt(request)
        prompt_path = run_dir / "prompts" / "texture_skill_routed.md"
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(prompt, encoding="utf-8")
        trace_writer = TraceWriter(run_dir)
        trace_writer.write(
            "workflow_started",
            phase="texture_skill_routed",
            summary="Prepared one long-running Texture child task.",
            artifacts=[str(request_path), str(prompt_path)],
            data={"resume": resume},
        )
        if runtime.dry_run:
            trace_writer.write(
                "workflow_dry_run",
                phase="texture_skill_routed",
                summary="Dry run stopped before launching the Texture child.",
            )
            build_trace(run_dir)
            print(f"texture_run_dir: {run_dir}")
            print(f"prompt_path: {prompt_path}")
            return 0

        capability_inventory, domain_policy_bounds = (
            _prepare_texture_child_launch_contract(request, runtime=runtime)
        )
        child_config = _texture_child_runtime_config(
            request,
            runtime,
            capability_inventory=capability_inventory,
            domain_policy_bounds=domain_policy_bounds,
        )
        parent_usd_cli_capability = start_parent_usd_cli_capability(
            config=child_config,
            run_dir=run_dir,
            workflow="texture.generate",
            session_workflow="texture-validation",
            input_roots=(Path(request.source_asset),),
            initial_scene=Path(request.source_asset),
            timeout_seconds=max(runtime.child_timeout_seconds, 900.0),
        )
        child_config = replace(
            child_config,
            parent_usd_cli_session_identity=parent_usd_cli_capability.identity_path,
            parent_usd_cli_session_identity_sha256=(
                parent_usd_cli_capability.identity_sha256
            ),
        )
        prompt += parent_usd_cli_prompt_contract(parent_usd_cli_capability)
        atomic_write_text(prompt_path, prompt)
        child_output_path = run_dir / "raw" / "texture_child_output.jsonl"
        child_final_path = run_dir / "raw" / "texture_child_final.json"
        child_output_path.parent.mkdir(parents=True, exist_ok=True)
        returncode = run_child_agent(
            config=child_config,
            prompt=prompt,
            run_dir=run_dir,
            child_output_path=child_output_path,
            child_final_path=child_final_path,
            scene_service=None,
            bridge_artifact_prefix="texture_skill_routed",
        )
        trace_writer.write(
            "child_finished",
            phase="texture_skill_routed",
            summary="Texture child exited after driving focused workflow steps.",
            artifacts=[str(child_output_path), str(child_final_path)],
            data={"returncode": returncode},
        )
        if returncode != 0:
            build_trace(run_dir)
            raise RuntimeError(f"Texture child agent exited with code {returncode}")
        verified_outcome = _execute_texture_skill_step(
            request,
            runtime=runtime,
            decision_patch=None,
        )
        if not isinstance(verified_outcome, TextureFinalizationResult):
            build_trace(run_dir)
            raise RuntimeError(
                "Texture child exited before deterministic terminal finalization"
            )
        result = verified_outcome
        final_path = run_dir / "final_summary.json"
        trace_writer.write(
            "workflow_finished",
            phase="texture_skill_routed",
            summary="Deterministic Texture finalization completed.",
            artifacts=[str(final_path)],
            data={"status": result.status},
        )
        build_trace(run_dir)
        _print_result(result, json_output=runtime.json_output)
        return _result_exit_code(result)
    finally:
        if parent_usd_cli_capability is not None:
            stop_parent_usd_cli_capability(parent_usd_cli_capability)


def _execute_texture_skill_step(
    request: TextureWorkflowRequest,
    *,
    runtime: TextureRuntimeConfig,
    decision_patch: Any | None,
) -> Any:
    from content_agent_workflows.texture import (
        LiveUsdCliTextureValidator,
        TextureAgentServiceClient,
        TextureWorkflowCancellationToken,
        VlmTextureVisualAssessor,
        run_texture_workflow_step,
    )

    checkpoint_path = request.output_dir / "workflow_checkpoint.json"
    checkpoint = (
        _preflight_resume(
            request,
            runtime=runtime,
            mode="interactive" if _is_embedded_texture_request(request) else "batch",
        )
        if checkpoint_path.is_file()
        else None
    )
    if checkpoint is not None and decision_patch is None:
        if not _is_embedded_texture_request(request):
            from content_agent_workflows.texture import (
                verify_texture_resume_decision_state,
            )

            # A standalone child may have durably authored the current patch before
            # interruption. Reuse those exact reviewed bytes.
            decision_patch = verify_texture_resume_decision_state(
                checkpoint,
                output_dir=request.output_dir,
            )
    if _is_embedded_texture_request(request):
        needs_runtime_services = checkpoint is None or (
            decision_patch is not None
            and checkpoint.next_action in {"execute", "refine"}
        )
    else:
        needs_runtime_services = (
            checkpoint is None or decision_patch is not None
        ) and (checkpoint is None or checkpoint.next_action not in {"done", "finalize"})
    token_value = None
    if needs_runtime_services and runtime.texture_agent_token_env is not None:
        token_value = os.getenv(runtime.texture_agent_token_env)
        if not token_value:
            raise ValueError(
                f"{runtime.texture_agent_token_env} is not set or is empty."
            )
    if needs_runtime_services:
        vlm_backend, vlm_model = _require_vlm_identity(runtime)
        client: Any = TextureAgentServiceClient(
            base_url=_require_texture_agent_url(runtime),
            timeout_seconds=runtime.texture_timeout_seconds,
            poll_interval_seconds=runtime.texture_poll_interval_seconds,
            max_status_poll_failures=5,
            token=token_value,
        )
        lazy_vlm = _LazyVlm(
            backend=vlm_backend,
            model=vlm_model,
            base_url=runtime.vlm_base_url,
            api_key_env=runtime.vlm_api_key_env,
            timeout_seconds=runtime.vlm_timeout_seconds,
        )
        validator: Any = LiveUsdCliTextureValidator(
            assessor=VlmTextureVisualAssessor(lazy_vlm),
            validation_policy_id=_validation_policy_id(runtime),
        )
    else:
        client = _UnavailableRuntimeAdapter()
        validator = _UnavailableRuntimeAdapter()

    cancellation_token = TextureWorkflowCancellationToken()
    workflow_active = False
    try:
        with _cooperative_signal_handlers(
            cancellation_token,
            checkpoint_path=request.output_dir / "workflow_checkpoint.json",
            workflow_active=lambda: workflow_active,
        ):
            workflow_active = True
            try:
                return run_texture_workflow_step(
                    request,
                    mode=(
                        "interactive"
                        if _is_embedded_texture_request(request)
                        else "batch"
                    ),
                    client=client,
                    validator=validator,
                    decision_patch=decision_patch,
                    progress_callback=_print_progress,
                    cancellation_check=cancellation_token.is_cancelled,
                )
            finally:
                workflow_active = False
    finally:
        pass


def _handle_texture_agent_step(args: argparse.Namespace) -> int:
    from content_agent_workflows.texture import (
        TextureDecisionPatch,
        TextureEmbeddedDecisionPatch,
        TextureWorkflowRequest,
    )

    run_dir = _prepare_run_dir(args.run_dir, resume=True)
    request = TextureWorkflowRequest.model_validate_json(
        (run_dir / "request.json").read_text(encoding="utf-8")
    )
    runtime = _load_texture_agent_launcher(run_dir, request=request)
    patch = None
    patch_path: Path | None = None
    if args.decision_patch is not None:
        patch_path = args.decision_patch.expanduser().resolve()
        if not patch_path.is_relative_to(run_dir):
            raise ValueError(
                "Texture decision patch must stay inside the run directory"
            )
        patch_model = (
            TextureEmbeddedDecisionPatch
            if _is_embedded_texture_request(request)
            else TextureDecisionPatch
        )
        patch = patch_model.model_validate_json(patch_path.read_text(encoding="utf-8"))
    outcome = _execute_texture_skill_step(
        request,
        runtime=runtime,
        decision_patch=patch,
    )
    outcome_action = getattr(outcome, "action", None)
    outcome_status = getattr(outcome, "status", None)
    _print_texture_step_outcome(outcome, run_dir=run_dir, json_output=True)
    artifacts: list[str] = []
    if outcome_action is not None:
        artifacts.append(str(run_dir / "agent_step_observation.json"))
    if patch_path is not None:
        artifacts.append(str(patch_path))
    if outcome_status is not None:
        artifacts.append(str(run_dir / "final_summary.json"))
    TraceWriter(run_dir).write(
        "step_finished",
        phase=f"texture_{outcome_action or outcome_status or 'step'}",
        summary="One focused Texture capability boundary completed.",
        artifacts=artifacts,
        data={
            "action": outcome_action,
            "status": outcome_status,
        },
    )
    build_trace(run_dir)
    if outcome_status is None:
        return 0
    return _result_exit_code(outcome)


def _print_texture_step_outcome(
    outcome: Any,
    *,
    run_dir: Path,
    json_output: bool,
) -> None:
    from content_agent_workflows.texture import (
        TextureEmbeddedStepObservation,
        TextureStepObservation,
    )

    if isinstance(outcome, TextureStepObservation | TextureEmbeddedStepObservation):
        path = run_dir / "agent_step_observation.json"
        atomic_write_json(path, outcome.model_dump(mode="json"))
        if json_output:
            print(outcome.model_dump_json(indent=2))
        else:
            print(f"texture_next_action: {outcome.action}")
            print(f"texture_observation: {path}")
        return
    _print_result(outcome, json_output=json_output)


def _write_texture_agent_launcher(
    run_dir: Path,
    runtime: TextureRuntimeConfig,
    *,
    request: TextureWorkflowRequest,
) -> Path:
    from world_understanding.utils.credentials import ensure_no_inline_secrets

    ensure_no_inline_secrets(
        {
            "codex_config": runtime.codex_config,
            "claude_config": runtime.claude_config,
        },
        context="Texture agent configuration",
        path_context=True,
    )
    _validate_texture_agent_runtime(runtime)
    resolved_run_dir = run_dir.expanduser().resolve()
    if request.output_dir.expanduser().resolve() != resolved_run_dir:
        raise ValueError("Texture launcher request output_dir mismatch")
    payload = asdict(runtime)
    if runtime.agent_cwd is not None:
        payload["agent_cwd"] = str(runtime.agent_cwd)
    path = run_dir / "texture_agent_launcher.json"
    launcher_document = {
        "schema_version": TEXTURE_AGENT_LAUNCHER_SCHEMA_VERSION,
        "runtime": payload,
    }
    serialized_launcher = (
        json.dumps(launcher_document, indent=2, sort_keys=True, ensure_ascii=True)
        + "\n"
    )
    if len(serialized_launcher.encode("utf-8")) > MAX_TEXTURE_AGENT_LAUNCHER_BYTES:
        raise ValueError(
            "Texture agent launcher exceeds the maximum allowed size of "
            f"{MAX_TEXTURE_AGENT_LAUNCHER_BYTES} bytes"
        )
    atomic_write_text(path, serialized_launcher)
    launcher_bytes = _read_texture_agent_launcher_file(
        path,
        label="Texture agent launcher",
        max_bytes=MAX_TEXTURE_AGENT_LAUNCHER_BYTES,
    )
    policy_path = _texture_agent_launcher_policy_path(run_dir)
    atomic_write_json(
        policy_path,
        {
            "schema_version": TEXTURE_AGENT_LAUNCHER_POLICY_SCHEMA_VERSION,
            "run_dir": str(resolved_run_dir),
            "launcher_sha256": hashlib.sha256(launcher_bytes).hexdigest(),
            "request_sha256": hashlib.sha256(
                _stable_texture_request_bytes(request)
            ).hexdigest(),
        },
    )
    return path


def _texture_agent_launcher_policy_path(run_dir: Path) -> Path:
    resolved_run_dir = run_dir.expanduser().resolve()
    return (
        resolved_run_dir.parent
        / f".{resolved_run_dir.name}.texture-agent-launcher-policy.json"
    )


def _stable_texture_request_bytes(request: TextureWorkflowRequest) -> bytes:
    return (
        json.dumps(
            request.model_dump(mode="json"),
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
        )
        + "\n"
    ).encode("utf-8")


def _read_texture_agent_launcher_file(
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


def _validate_texture_agent_runtime(runtime: TextureRuntimeConfig) -> None:
    required_strings = {
        "runner",
        "codex_sandbox_mode",
        "claude_permission_mode",
        "claude_execution_mode",
        "execution_mode",
    }
    optional_strings = {
        "texture_agent_url",
        "texture_agent_token_env",
        "vlm_backend",
        "vlm_model",
        "vlm_base_url",
        "vlm_api_key_env",
        "model",
        "model_reasoning_effort",
        "codex_base_url",
    }
    for field_name in required_strings:
        value = getattr(runtime, field_name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Texture agent launcher field {field_name!r} must be a non-empty string"
            )
    for field_name in optional_strings:
        value = getattr(runtime, field_name)
        if value is not None and not isinstance(value, str):
            raise ValueError(
                f"Texture agent launcher field {field_name!r} must be a string or null"
            )

    for field_name in (
        "texture_timeout_seconds",
        "texture_poll_interval_seconds",
        "vlm_timeout_seconds",
    ):
        value = getattr(runtime, field_name)
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
            or float(value) <= 0
        ):
            raise ValueError(
                f"Texture agent launcher field {field_name!r} must be positive"
            )
    if (
        isinstance(runtime.child_timeout_seconds, bool)
        or not isinstance(runtime.child_timeout_seconds, int | float)
        or not math.isfinite(float(runtime.child_timeout_seconds))
        or float(runtime.child_timeout_seconds) < 0
    ):
        raise ValueError(
            "Texture agent launcher field 'child_timeout_seconds' must be non-negative"
        )
    for field_name in ("json_output", "dry_run"):
        if not isinstance(getattr(runtime, field_name), bool):
            raise ValueError(
                f"Texture agent launcher field {field_name!r} must be a boolean"
            )
    for field_name in ("codex_config", "claude_config"):
        value = getattr(runtime, field_name)
        if value is not None and not isinstance(value, dict):
            raise ValueError(
                f"Texture agent launcher field {field_name!r} must be an object or null"
            )
    if runtime.claude_max_turns is not None and (
        isinstance(runtime.claude_max_turns, bool)
        or not isinstance(runtime.claude_max_turns, int)
        or runtime.claude_max_turns <= 0
    ):
        raise ValueError(
            "Texture agent launcher field 'claude_max_turns' must be a positive integer or null"
        )
    if runtime.agent_cwd is not None and not isinstance(runtime.agent_cwd, Path):
        raise ValueError(
            "Texture agent launcher field 'agent_cwd' must be a path or null"
        )

    if runtime.texture_agent_url is not None:
        if (
            _normalized_url(runtime.texture_agent_url, "texture_agent_url")
            != runtime.texture_agent_url
        ):
            raise ValueError(
                "Texture agent launcher field 'texture_agent_url' is not normalized"
            )
    if runtime.vlm_base_url is not None:
        if (
            _normalized_url(runtime.vlm_base_url, "vlm_base_url")
            != runtime.vlm_base_url
        ):
            raise ValueError(
                "Texture agent launcher field 'vlm_base_url' is not normalized"
            )
    for field_name in ("texture_agent_token_env", "vlm_api_key_env"):
        value = getattr(runtime, field_name)
        if _optional_env_name(value, field_name) != value:
            raise ValueError(
                f"Texture agent launcher field {field_name!r} is not normalized"
            )

    if (runtime.vlm_backend is None) != (runtime.vlm_model is None):
        raise ValueError("Texture agent launcher VLM backend/model must be paired")
    if (
        runtime.vlm_backend is not None
        and runtime.vlm_backend not in PUBLIC_VLM_BACKENDS
    ):
        raise ValueError("Texture agent launcher uses an unsupported VLM backend")
    if runtime.runner not in {RUNNER_CODEX, RUNNER_CLAUDE}:
        raise ValueError("Texture agent launcher uses an unsupported child runner")
    if runtime.codex_sandbox_mode != CODEX_SANDBOX_WORKSPACE_WRITE:
        raise ValueError(
            "Texture agent launcher uses an unsupported Codex sandbox mode"
        )
    if runtime.claude_permission_mode not in {
        "default",
        "acceptEdits",
        "bypassPermissions",
        "plan",
    }:
        raise ValueError(
            "Texture agent launcher uses an unsupported Claude permission mode"
        )
    if runtime.claude_execution_mode not in {
        CLAUDE_EXECUTION_SDK,
        CLAUDE_EXECUTION_CLI,
    }:
        raise ValueError(
            "Texture agent launcher uses an unsupported Claude execution mode"
        )
    if runtime.execution_mode not in {
        TEXTURE_EXECUTION_SKILL_ROUTED,
        TEXTURE_EXECUTION_FIXED,
    }:
        raise ValueError("Texture agent launcher uses an unsupported execution mode")
    if runtime.execution_mode == TEXTURE_EXECUTION_FIXED:
        _require_vlm_identity(runtime)
    elif (
        runtime.vlm_base_url is not None or runtime.vlm_api_key_env is not None
    ) and runtime.vlm_backend is None:
        raise ValueError(
            "Texture agent launcher VLM endpoint options require a backend/model"
        )
    if (
        runtime.vlm_base_url is not None
        and runtime.vlm_backend in {"anthropic", "gemini"}
        and runtime.vlm_api_key_env is None
    ):
        raise ValueError(
            "Texture agent launcher requires a scoped VLM key for a custom endpoint"
        )
    if (
        runtime.texture_agent_token_env is not None
        and runtime.texture_agent_url is None
    ):
        raise ValueError(
            "Texture agent launcher cannot configure a service token without a URL"
        )
    if runtime.texture_agent_url is not None:
        _validate_bearer_transport(
            runtime.texture_agent_url,
            credential_env=runtime.texture_agent_token_env,
            option="texture_agent_url",
        )
    from world_understanding.utils.credentials import ensure_no_inline_secrets

    ensure_no_inline_secrets(
        {
            "codex_config": runtime.codex_config,
            "claude_config": runtime.claude_config,
        },
        context="Texture agent launcher",
        path_context=True,
    )


def _load_texture_agent_launcher(
    run_dir: Path,
    *,
    request: TextureWorkflowRequest,
) -> TextureRuntimeConfig:
    resolved_run_dir = run_dir.expanduser().resolve()
    policy_path = _texture_agent_launcher_policy_path(resolved_run_dir)
    policy_bytes = _read_texture_agent_launcher_file(
        policy_path,
        label="Texture agent launcher policy",
        max_bytes=MAX_TEXTURE_AGENT_LAUNCHER_POLICY_BYTES,
    )
    try:
        policy = json.loads(policy_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid Texture agent launcher policy JSON") from exc
    if not isinstance(policy, dict) or set(policy) != {
        "schema_version",
        "run_dir",
        "launcher_sha256",
        "request_sha256",
    }:
        raise ValueError("Texture agent launcher policy has invalid fields")
    if policy.get("schema_version") != TEXTURE_AGENT_LAUNCHER_POLICY_SCHEMA_VERSION:
        raise ValueError("Unsupported Texture agent launcher policy schema")
    if Path(str(policy.get("run_dir"))).expanduser().resolve() != resolved_run_dir:
        raise ValueError("Texture agent launcher policy run_dir mismatch")
    for digest_field in ("launcher_sha256", "request_sha256"):
        digest = policy.get(digest_field)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"Texture agent launcher policy {digest_field} is invalid")
    actual_request_sha256 = hashlib.sha256(
        _stable_texture_request_bytes(request)
    ).hexdigest()
    if actual_request_sha256 != policy["request_sha256"]:
        raise ValueError(
            "Texture request digest does not match the parent-owned policy"
        )

    path = resolved_run_dir / "texture_agent_launcher.json"
    launcher_bytes = _read_texture_agent_launcher_file(
        path,
        label="Texture agent launcher",
        max_bytes=MAX_TEXTURE_AGENT_LAUNCHER_BYTES,
    )
    actual_sha256 = hashlib.sha256(launcher_bytes).hexdigest()
    if actual_sha256 != policy["launcher_sha256"]:
        raise ValueError(
            "Texture agent launcher digest does not match the parent-owned policy"
        )
    try:
        payload = json.loads(launcher_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid Texture agent launcher JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "runtime"}:
        raise ValueError("Texture agent launcher has invalid fields")
    if payload.get("schema_version") != TEXTURE_AGENT_LAUNCHER_SCHEMA_VERSION:
        raise ValueError("Unsupported Texture agent launcher schema")
    runtime = payload.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError("Texture agent launcher runtime must be a JSON object")
    expected_runtime_fields = {field.name for field in fields(TextureRuntimeConfig)}
    if set(runtime) != expected_runtime_fields:
        missing = sorted(expected_runtime_fields - set(runtime))
        unknown = sorted(set(runtime) - expected_runtime_fields)
        raise ValueError(
            "Texture agent launcher runtime fields do not match the schema: "
            f"missing={missing}, unknown={unknown}"
        )
    raw_agent_cwd = runtime.get("agent_cwd")
    if raw_agent_cwd is not None:
        if not isinstance(raw_agent_cwd, str):
            raise ValueError(
                "Texture agent launcher runtime field 'agent_cwd' must be a string or null"
            )
        runtime["agent_cwd"] = Path(raw_agent_cwd)
    try:
        result = TextureRuntimeConfig(**runtime)
        _validate_texture_agent_runtime(result)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid Texture agent launcher runtime") from exc
    return result


def _texture_child_runtime_config(
    request: TextureWorkflowRequest,
    runtime: TextureRuntimeConfig,
    *,
    capability_inventory: ChildLaunchArtifactIdentity,
    domain_policy_bounds: ChildLaunchArtifactIdentity,
) -> _TextureChildRuntimeConfig:
    return _TextureChildRuntimeConfig(
        repo_root=find_repo_root(),
        usd_path=Path(request.source_asset).expanduser().resolve(),
        runner=runtime.runner,
        model=runtime.model,
        model_reasoning_effort=runtime.model_reasoning_effort,
        codex_base_url=runtime.codex_base_url,
        codex_sandbox_mode=runtime.codex_sandbox_mode,
        codex_config=runtime.codex_config,
        claude_config=runtime.claude_config,
        claude_permission_mode=runtime.claude_permission_mode,
        claude_max_turns=runtime.claude_max_turns,
        claude_execution_mode=runtime.claude_execution_mode,
        child_timeout_seconds=runtime.child_timeout_seconds,
        agent_cwd=runtime.agent_cwd,
        reference_images=[],
        reference_files=None,
        child_capability_inventory=capability_inventory,
        child_domain_policy_bounds=domain_policy_bounds,
        child_forbidden_environment_names=tuple(
            sorted(
                {
                    name
                    for name in (
                        runtime.texture_agent_token_env,
                        runtime.vlm_api_key_env,
                    )
                    if name
                }
            )
        ),
    )


def _prepare_texture_child_launch_contract(
    request: TextureWorkflowRequest,
    *,
    runtime: TextureRuntimeConfig,
) -> tuple[ChildLaunchArtifactIdentity, ChildLaunchArtifactIdentity]:
    """Freeze Texture capabilities and bounds without endpoint authority."""

    from content_agent_workflows.texture import texture_embedded_capability_digests

    run_dir = request.output_dir.expanduser().resolve()
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    request_identity = child_launch_artifact_identity(
        run_dir,
        run_dir / "request.json",
    )
    capability_path = atomic_write_json(
        raw_dir / "texture_child_capability_inventory.json",
        {
            "schema_version": "content-workflow-cli.texture-child-capabilities.v1",
            "workflow": "texture.generate",
            "request_identity": request_identity.model_dump(mode="json"),
            "selection_owner": "texture-domain-wrapper",
            "capability_digests": texture_embedded_capability_digests(),
            "adapter": {
                "capability_id": "texture.agent-step",
                "execution_owner": "texture-domain-wrapper",
                "transport": "typed-artifact",
            },
        },
    )
    policy_path = atomic_write_json(
        raw_dir / "texture_child_domain_policy.json",
        {
            "schema_version": "content-workflow-cli.texture-child-policy.v1",
            "workflow": "texture.generate",
            "request_identity": request_identity.model_dump(mode="json"),
            "validation_policy_id": _validation_policy_id(runtime),
            "execution_mode": runtime.execution_mode,
            "max_vqa_iterations": request.max_vqa_iterations,
            "target_runtime": request.target_runtime,
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


def _prepare_texture_plan_child_launch_contract(
    request: TextureWorkflowRequest,
    *,
    runtime: TextureRuntimeConfig,
    preparation_binding: Any,
) -> tuple[
    ChildLaunchArtifactIdentity,
    ChildLaunchArtifactIdentity,
    ChildLaunchArtifactIdentity,
]:
    """Freeze plan-only semantic authority over exact preparation bytes."""

    from content_agent_workflows.texture import texture_embedded_capability_digests

    run_dir = request.output_dir.expanduser().resolve()
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    request_identity = child_launch_artifact_identity(
        run_dir,
        run_dir / "request.json",
    )
    preparation_identity = child_launch_artifact_identity(
        run_dir,
        preparation_binding.path,
    )
    output_contract_identity = _prepare_texture_plan_output_contract(run_dir)
    capability_path = atomic_write_json(
        raw_dir / "texture_child_capability_inventory.json",
        {
            "schema_version": "content-workflow-cli.texture-child-capabilities.v4",
            "workflow": "texture.generate",
            "request_identity": request_identity.model_dump(mode="json"),
            "preparation_identity": preparation_identity.model_dump(mode="json"),
            "output_contract_identity": output_contract_identity.model_dump(
                mode="json"
            ),
            "selection_owner": "texture-reasoning-child",
            "allowed_actions": [
                "preserve",
                "generate",
                "apply_provided",
                "defer",
                "reject",
            ],
            "capability_digests": texture_embedded_capability_digests(),
            "adapter": {
                "capability_id": "texture.plan-only",
                "execution_owner": "texture-domain-wrapper",
                "transport": "typed-artifact",
                "output_owner": "texture-domain-wrapper",
                "provider_output_path": str(
                    run_dir / "raw" / "texture_child_final.json"
                ),
                "canonical_output_path": str(run_dir / "texture_plan.json"),
                "output_schema": TEXTURE_AGENTIC_PLAN_SCHEMA_VERSION,
            },
        },
    )
    policy_path = atomic_write_json(
        raw_dir / "texture_child_domain_policy.json",
        {
            "schema_version": "content-workflow-cli.texture-child-policy.v5",
            "workflow": "texture.generate",
            "request_identity": request_identity.model_dump(mode="json"),
            "preparation_identity": preparation_identity.model_dump(mode="json"),
            "validation_policy_id": _validation_policy_id(runtime),
            "execution_mode": runtime.execution_mode,
            "bounded_attempts_per_unit": 1,
            "tool_policy": _texture_tool_free_child_policy(),
            "child_authority": {
                "domain_client": False,
                "domain_credentials": False,
                "domain_network": False,
                "renderer_network": False,
                "semantic_operation_selection": True,
                "usd_mutation": False,
                "review": False,
                "publication": False,
            },
        },
    )
    return (
        child_launch_artifact_identity(run_dir, capability_path),
        child_launch_artifact_identity(run_dir, policy_path),
        output_contract_identity,
    )


def _prepare_texture_review_child_launch_contract(
    request: TextureWorkflowRequest,
    *,
    runtime: TextureRuntimeConfig,
    accepted_binding: Any,
    ledger_binding: Any,
    evidence_binding: Any,
) -> tuple[ChildLaunchArtifactIdentity, ChildLaunchArtifactIdentity]:
    """Freeze a review-only child boundary over exact post-operation evidence."""

    run_dir = request.output_dir.expanduser().resolve()
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    request_identity = child_launch_artifact_identity(
        run_dir,
        run_dir / "request.json",
    )
    inputs = {
        "accepted_plan": child_launch_artifact_identity(
            run_dir, accepted_binding.path
        ).model_dump(mode="json"),
        "adapter_ledger": child_launch_artifact_identity(
            run_dir, ledger_binding.path
        ).model_dump(mode="json"),
        "evidence": child_launch_artifact_identity(
            run_dir, evidence_binding.path
        ).model_dump(mode="json"),
    }
    capability_path = atomic_write_json(
        raw_dir / "texture_review_child_capability_inventory.json",
        {
            "schema_version": "content-workflow-cli.texture-child-capabilities.v4",
            "workflow": "texture.review",
            "request_identity": request_identity.model_dump(mode="json"),
            "input_identities": inputs,
            "selection_owner": "texture-reasoning-child",
            "adapter": {
                "capability_id": "texture.review-only",
                "execution_owner": "texture-domain-wrapper",
                "transport": "typed-artifact",
                "output_owner": "texture-domain-wrapper",
                "provider_output_path": str(
                    run_dir / "raw" / "texture_review_child_final.json"
                ),
                "canonical_output_path": str(run_dir / "texture_review_proposal.json"),
                "output_schema": ("content-agent-workflows.texture-agentic-review.v1"),
            },
        },
    )
    policy_path = atomic_write_json(
        raw_dir / "texture_review_child_domain_policy.json",
        {
            "schema_version": "content-workflow-cli.texture-child-policy.v5",
            "workflow": "texture.review",
            "request_identity": request_identity.model_dump(mode="json"),
            "input_identities": inputs,
            "validation_policy_id": _validation_policy_id(runtime),
            "execution_mode": runtime.execution_mode,
            "tool_policy": _texture_tool_free_child_policy(),
            "child_authority": {
                "domain_client": False,
                "domain_credentials": False,
                "domain_network": False,
                "renderer_network": False,
                "semantic_operation_selection": False,
                "usd_mutation": False,
                "visual_review": True,
                "publication": False,
            },
        },
    )
    return (
        child_launch_artifact_identity(run_dir, capability_path),
        child_launch_artifact_identity(run_dir, policy_path),
    )


def _texture_tool_free_child_policy() -> dict[str, object]:
    """Describe the runner-enforced boundary for Texture reasoning turns."""

    return {
        "mode": "tools_disabled",
        "filesystem_access": "none",
        "command_execution": "none",
        "network_access": "reasoning_transport_only",
        "enforcement": [
            "provider_tool_configuration",
            "runner_tool_denial",
            "observable_tool_event_rejection",
        ],
    }


def _build_texture_plan_prompt(
    request: TextureWorkflowRequest,
    *,
    preparation: Any,
    preparation_binding: Any,
    output_contract_identity: ChildLaunchArtifactIdentity,
) -> str:
    task = {
        "schema_version": "content-agents.texture-plan-only-task.v2",
        "workflow": "texture.generate",
        "request": request.model_dump(mode="json"),
        "preparation": preparation.model_dump(mode="json"),
        "preparation_identity": preparation_binding.model_dump(mode="json"),
        "output_contract_identity": output_contract_identity.model_dump(mode="json"),
        "output_schema": TEXTURE_AGENTIC_PLAN_SCHEMA_VERSION,
        "tool_policy": _texture_tool_free_child_policy(),
    }
    return f"""You are the plan-only reasoner for one bounded Texture attempt.

The parent validated and embedded every immutable input in this task:
{json.dumps(task, indent=2)}

Return exactly one JSON object matching the provider-enforced
`{TEXTURE_AGENTIC_PLAN_SCHEMA_VERSION}` output schema. The parent alone validates
that object and writes the canonical plan. Do not invoke any command, tool,
service, generator, renderer, VLM, or network capability. Do not read, search,
open, or write any filesystem path. Do not mutate USD, apply an image, review
evidence, publish, or create another workflow artifact. If the embedded inputs
are insufficient, return no substitute result and fail closed.

The strict wire schema carries `generator_inputs.parameters` and each
`provided_images[].producer.provenance` object as a JSON-encoded string. Encode
those two maps exactly as JSON objects; the parent decodes them before canonical
model validation.

The plan must copy the exact `preparation_identity`, the exact `source` from
`preparation.request`, the exact `scope_plan_digest`, references, capability
constraints, and every inspection unit exactly once in inspection order. Copy
each unit's unit ID and material, member-prim, and member-subset paths without
change. Select exactly one action per unit: `preserve`, `generate`,
`apply_provided`, `defer`, or `reject`.
When request metadata contains
`content_workflow_cli_texture_agentic_action_policy`, match each exact target to
its prepared unit and select that required action without substitution.
When request metadata contains
`content_workflow_cli_texture_agentic_appearance_policy`, match each exact target
to its prepared unit and copy that value byte-for-byte as the unit's
`requested_appearance`; do not summarize, expand, or paraphrase it.

For `generate`, include a concise non-empty requested appearance specific to that
unit. Copy the prepared unit's typed `proposed_generator_inputs` exactly with
`execution_mode=provider_generate`, except replace `prompt` with the deterministic
detail-policy form of that unit's requested appearance. For `surface_only`, use
the exact guardrail envelope already present in the prepared prompt, with only
the description replaced by the unit's requested appearance; never include
instructions for another unit in this prompt. When the request did not explicitly
select a World Understanding backend, preparation freezes
`coding_agent_companion` as the repository's image-generation default; this is
provider authority, not an invented fallback. For `apply_provided`, bind exactly
one existing
albedo artifact from request metadata
`content_workflow_cli_texture_agentic_provided_candidates`, match its target to
the prepared unit, copy its exact artifact and producer identity, and copy the
prepared generator inputs while changing only `execution_mode` to
`apply_provided`, `backend` to `outer_provided_image_apply`, clearing engine,
seed, and parameters, adding that one approved image, and replacing `prompt` by
the same deterministic unit-specific detail-policy form described above.
References are never candidates. Preserve,
defer, and reject must set requested_appearance and generator_inputs to null.

Use fail-closed preservation with every boolean true. Acceptance must require
fresh before/after renders, provider-neutral renderer metadata, exact UV and
material scope, package closure, and non-target preservation. Evidence must name
one or more supported OVRTX directions from `+x-y+z`, `+z`, and `-z` as its
required views; when request metadata contains
`content_workflow_cli_texture_agentic_evidence_views`, copy that exact ordered
list. Set every required boolean true. Stop policy is exactly one attempt per
unit, zero plan revisions, no provider retry, and no fallback. Your final
response must be the one plan JSON object only; do not include private reasoning.
"""


def _build_texture_plan_dry_run_prompt(request: TextureWorkflowRequest) -> str:
    """Describe the eventual tool-free turn without creating preparation."""

    task = {
        "schema_version": "content-agents.texture-plan-only-dry-run.v1",
        "workflow": "texture.generate",
        "request": request.model_dump(mode="json"),
        "output_schema": TEXTURE_AGENTIC_PLAN_SCHEMA_VERSION,
        "tool_policy": _texture_tool_free_child_policy(),
        "status": "preparation_not_created_in_dry_run",
    }
    return (
        "Dry-run preview only. No reasoning child was launched. On a live run, "
        "the parent embeds the validated preparation and invokes one tool-free "
        "structured-output turn:\n"
        f"{json.dumps(task, indent=2)}\n"
    )


def _build_texture_review_prompt(
    request: TextureWorkflowRequest,
    *,
    accepted: Any,
    accepted_binding: Any,
    ledger: Any,
    ledger_binding: Any,
    evidence: Any,
    evidence_binding: Any,
) -> str:
    """Build the separate review-only task over immutable current-run evidence."""

    review_attachments = _texture_review_image_attachments(
        accepted=accepted,
        evidence=evidence,
    )
    task = {
        "schema_version": "content-agents.texture-review-only-task.v3",
        "workflow": "texture.review",
        "request": request.model_dump(mode="json"),
        "accepted_plan": accepted.model_dump(mode="json"),
        "accepted_plan_identity": accepted_binding.model_dump(mode="json"),
        "adapter_ledger": ledger.model_dump(mode="json"),
        "adapter_ledger_identity": ledger_binding.model_dump(mode="json"),
        "evidence": evidence.model_dump(mode="json"),
        "evidence_identity": evidence_binding.model_dump(mode="json"),
        "candidate": ledger.final_candidate.model_dump(mode="json"),
        "expected_plan_digest": accepted.proposal_digest,
        "required_unit_ids": list(accepted.plan.unit_ids),
        "review_image_attachments": [
            {
                "role": role,
                **({"unit_id": unit_id} if unit_id is not None else {}),
                "artifact": binding.model_dump(mode="json"),
            }
            for role, unit_id, binding in review_attachments
        ],
        "required_visual_artifacts": [
            binding.model_dump(mode="json")
            for item in evidence.unit_evidence
            for binding in (*item.source_images, *item.candidate_images)
        ],
        "output_schema": "content-agent-workflows.texture-agentic-review.v1",
        "tool_policy": _texture_tool_free_child_policy(),
        "output_contract": {
            "schema_version": "content-agent-workflows.texture-agentic-review.v1",
            "accepted_plan": "copy task.accepted_plan_identity exactly",
            "adapter_ledger": "copy task.adapter_ledger_identity exactly",
            "evidence": "copy task.evidence_identity exactly",
            "candidate": "copy task.candidate exactly",
            "plan_digest": "copy task.expected_plan_digest exactly",
            "unit_reviews": [
                {
                    "unit_id": "one task.required_unit_ids entry",
                    "disposition": "accept, reject, or unresolved",
                    "rationale": "non-empty review rationale",
                }
            ],
            "inspected_visual_artifacts": (
                "copy task.required_visual_artifacts exactly"
            ),
            "findings": ["at least one concise overall finding"],
        },
    }
    return f"""You are the separate review-only reasoner for one Texture attempt.

The parent validated and embedded every immutable document in this task. The
accepted appearance references, outer-provided candidate images, and exact
source and candidate OVRTX images are attached to the turn in the same order as
`review_image_attachments`:
{json.dumps(task, indent=2)}

Inspect every attached image according to its explicit role and unit mapping,
then compare every source/candidate pair against the accepted appearance
references, requested appearance, adapter ledger, and evidence. Return exactly
one JSON object matching the provider-enforced
`content-agent-workflows.texture-agentic-review.v1` schema.
The parent alone validates that object and writes the canonical review proposal.
Do not invoke any command, tool, service, generator, renderer, VLM, or network
capability. Do not read, search, open, or write any filesystem path. If an
embedded document or attached image is absent or insufficient, mark the affected
unit unresolved or reject it; never search for substitute evidence.

Copy the accepted-plan, adapter-ledger, evidence, and candidate identities from
`output_contract` without change. Set `plan_digest` to the task's exact
`expected_plan_digest`; do not substitute the scope-plan digest. Cover the exact
`required_unit_ids` once in order under `unit_reviews`, using only the exact
fields `unit_id`, `disposition`, and `rationale`. Copy
`required_visual_artifacts` exactly into `inspected_visual_artifacts`. Put the
overall review strings in `findings`. Follow `output_contract` literally; do
not emit `action`, `decision`, `overall_findings`, or
`required_visual_artifacts` as output fields.

For generate, apply_provided, and preserve, choose accept, reject, or unresolved
from the exact requested appearance, acceptance criteria, and matched current-run
views. A preserve unit must remain visually and semantically unchanged. A plan
unit already marked defer must remain unresolved; a plan unit already marked
reject must remain reject. Include a non-empty rationale for every unit and at
least one concise overall finding.

Do not modify plan facts, evidence, USD, or candidate bytes. Do not publish. Your
final response must be the one review JSON object only; do not include private
reasoning.
"""


def _texture_review_image_attachments(
    *,
    accepted: Any,
    evidence: Any,
) -> tuple[tuple[str, str | None, Any], ...]:
    """Return review images in one explicit, tool-free attachment order."""

    attachments: list[tuple[str, str | None, Any]] = []
    attachments.extend(
        (reference.role, None, reference.artifact)
        for reference in accepted.plan.reference_artifacts
    )
    for disposition in accepted.plan.dispositions:
        generator_inputs = disposition.generator_inputs
        if generator_inputs is None:
            continue
        attachments.extend(
            (provided.role, disposition.unit_id, provided.artifact)
            for provided in generator_inputs.provided_images
        )
    for item in evidence.unit_evidence:
        attachments.extend(
            ("source_evidence", item.unit_id, binding) for binding in item.source_images
        )
        attachments.extend(
            ("candidate_evidence", item.unit_id, binding)
            for binding in item.candidate_images
        )
    return tuple(attachments)


def _build_texture_agent_prompt(request: TextureWorkflowRequest) -> str:
    run_dir = request.output_dir.expanduser().resolve()
    task = {
        "schema_version": "content-agents.skill-routed-task.v1",
        "workflow": "texture.generate",
        "run_dir": str(run_dir),
        "request_path": str(run_dir / "request.json"),
        "required_skills": [
            "usd-cli",
            "content-texture-scope",
            "content-texture-candidate",
            "content-texture-quality",
            "content-texture-publish",
            "content-workflow-texture",
        ],
    }
    return f"""You are the single long-running child for a skill-routed Texture workflow.

Load and follow every skill listed in this compact task:
{json.dumps(task, indent=2)}

Drive the workflow one focused boundary at a time. First run:
`content-workflow-cli texture _agent-step --run-dir {run_dir}`

Each workflow command can take many minutes. Wait for it to exit before
inspecting its artifacts or returning. Never report `blocked` while it or
another workflow command is still running.

After every non-terminal step, read `agent_step_observation.json` and the bound
evidence it lists. Write the exact typed decision to `decision_patch_path` using
schema `content-agent-workflows.texture-decision-patch.v1`. Copy
`request_digest`, `source_identity_digest`, `plan_digest`,
`checkpoint_decision_digest`, `checkpoint_revision`, `action`, `iteration`,
`target_unit_ids`, and `evidence_sha256_by_path` unchanged. Set `operations` to
the observation's `required_operations`, then add only your concise `rationale`
and numeric `confidence`. Invoke the same command with `--decision-patch <path>`.
Never broaden unit IDs, skip an operation, reuse a stale patch, edit the source
USD, or write outside the run directory. Continue until deterministic
finalization writes `final_summary.json`.

For failed visual units, review the new evidence and authorize only the exact
failed IDs in the observation. Do not rerun accepted units. Your final response
must be a concise JSON object with `status` and `final_summary_path`; do not
include private reasoning.
"""


def _validation_policy_id(runtime: TextureRuntimeConfig) -> str:
    descriptor: dict[str, object]
    if runtime.execution_mode == TEXTURE_EXECUTION_SKILL_ROUTED:
        descriptor = {
            "schema_version": TEXTURE_VALIDATION_POLICY_VERSION,
            "review_routes": {
                "agentic_plan": {
                    "backend": "coding-agent-companion",
                    "runner": runtime.runner,
                    "model": runtime.model or "configured-default",
                },
                "compatibility_step": {
                    "backend": (
                        "vlm-provider"
                        if runtime.vlm_backend is not None
                        else "unavailable"
                    ),
                    "vlm_backend": runtime.vlm_backend,
                    "vlm_model": runtime.vlm_model,
                    "vlm_base_url": runtime.vlm_base_url,
                },
            },
            "scene_tool": "usd-cli",
            "renderer": "ovrtx",
        }
    else:
        vlm_backend, vlm_model = _require_vlm_identity(runtime)
        descriptor = {
            "schema_version": TEXTURE_VALIDATION_POLICY_VERSION,
            "review_backend": "vlm-provider",
            "vlm_backend": vlm_backend,
            "vlm_model": vlm_model,
            "vlm_base_url": runtime.vlm_base_url,
            "scene_tool": "usd-cli",
            "renderer": "ovrtx",
        }
    canonical = json.dumps(
        descriptor,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"{TEXTURE_VALIDATION_POLICY_VERSION}:{digest}"


def _validate_texture_resume_policy(
    request: TextureWorkflowRequest,
    *,
    runtime: TextureRuntimeConfig,
) -> None:
    """Require the exact frozen review and renderer policy before resume writes."""

    persisted_policy = request.metadata.get(TEXTURE_VALIDATION_POLICY_METADATA_KEY)
    active_policy = _validation_policy_id(runtime)
    if persisted_policy != active_policy:
        raise ValueError(
            "Texture validation policy changed; resume with the original "
            "coding-agent runner/model and usd-cli renderer contract."
        )


def _preflight_resume(
    request: TextureWorkflowRequest,
    *,
    runtime: TextureRuntimeConfig,
    mode: TextureWorkflowMode = "batch",
) -> TextureWorkflowCheckpoint:
    """Validate all durable identity before credentials or runtime contact."""

    from content_agent_workflows.texture import (
        TextureWorkflowCheckpointStore,
        validate_resume_identity,
    )

    _reject_symlink_entries(request.output_dir)
    checkpoint = TextureWorkflowCheckpointStore(request.output_dir).load()
    validate_resume_identity(checkpoint, request=request, mode=mode)
    _validate_texture_resume_policy(request, runtime=runtime)
    if (
        runtime.execution_mode == TEXTURE_EXECUTION_SKILL_ROUTED
        and not _is_embedded_texture_request(request)
    ):
        _verify_texture_resume_decision_ledger(request, checkpoint)
    return checkpoint


def _verify_texture_resume_decision_ledger(
    request: TextureWorkflowRequest,
    checkpoint: TextureWorkflowCheckpoint,
) -> None:
    """Fail before child launch when persisted skill decisions are incomplete."""

    from content_agent_workflows.texture import verify_texture_resume_decision_state

    verify_texture_resume_decision_state(
        checkpoint,
        output_dir=request.output_dir,
    )


def _print_progress(progress: TextureWorkflowProgress) -> None:
    print(
        (
            f"[texture:{progress.phase}] iteration={progress.iteration} "
            f"accepted={progress.accepted_unit_count}/"
            f"{progress.selected_unit_count} "
            f"remaining={progress.remaining_unit_count}: {progress.message}"
        ),
        file=sys.stderr,
        flush=True,
    )


def _print_result(
    result: TextureFinalizationResult,
    *,
    json_output: bool,
) -> None:
    if json_output:
        print(result.model_dump_json(indent=2))
        return
    print(f"Status: {result.status}")
    print(f"Run directory: {result.output_dir}")
    print(f"Output USD: {result.output_asset_path or '(none)'}")
    print(f"Accepted units: {len(result.accepted_unit_ids)}")
    print(f"Remaining units: {len(result.remaining_unit_ids)}")
    print(f"Request: {result.request_path}")
    print(f"Plan: {result.texture_plan_path}")
    print(f"Validation evidence: {result.validation_evidence_path}")
    print(f"Checkpoint: {result.workflow_checkpoint_path}")
    print(f"Final summary: {result.final_summary_path}")


def _result_exit_code(result: TextureFinalizationResult) -> int:
    if result.status == "pass":
        return 0
    if result.status == "conditional":
        return TEXTURE_CONDITIONAL_EXIT_CODE
    return TEXTURE_CANCELLED_EXIT_CODE


@contextmanager
def _cooperative_signal_handlers(
    cancellation_token: TextureWorkflowCancellationToken,
    *,
    checkpoint_path: Path,
    workflow_active: Callable[[], bool],
) -> Iterator[None]:
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    previous: dict[signal.Signals, Any] = {}

    def request_cancel(signum: int, _frame: FrameType | None) -> None:
        checkpoint_exists = checkpoint_path.is_file()
        if not workflow_active() or not checkpoint_exists:
            if checkpoint_exists:
                message = (
                    "Texture setup received an interrupt before workflow execution "
                    "was active; aborting immediately with the existing checkpoint "
                    "preserved."
                )
            else:
                message = (
                    "Texture setup received an interrupt before a durable checkpoint "
                    "existed; aborting immediately."
                )
            print(
                message,
                file=sys.stderr,
                flush=True,
            )
            raise KeyboardInterrupt
        if cancellation_token.is_cancelled():
            print(
                "Texture workflow received a second interrupt; "
                "aborting immediately with the latest checkpoint preserved.",
                file=sys.stderr,
                flush=True,
            )
            raise KeyboardInterrupt
        cancellation_token.cancel()
        try:
            signal_name = signal.Signals(signum).name
        except ValueError:
            signal_name = str(signum)
        print(
            f"Texture workflow received {signal_name}; "
            "saving a resumable cancelled result. Interrupt again to force exit.",
            file=sys.stderr,
            flush=True,
        )

    signals = (signal.SIGINT, signal.SIGTERM)
    try:
        for item in signals:
            previous[item] = signal.getsignal(item)
            signal.signal(item, request_cancel)
        yield
    finally:
        for registered_signal, handler in previous.items():
            signal.signal(registered_signal, handler)


def _prepare_run_dir(path: Path, *, resume: bool) -> Path:
    expanded = path.expanduser()
    lexical = Path(os.path.abspath(expanded))
    try:
        resolved = expanded.resolve()
    except OSError as exc:
        raise ValueError(f"Unable to resolve Texture run directory: {path}") from exc
    if lexical != resolved:
        raise ValueError(
            "Texture run directory must not traverse symlinks: "
            f"{lexical} resolves to {resolved}"
        )
    if resume:
        if not resolved.is_dir():
            raise FileNotFoundError(f"Texture run directory does not exist: {resolved}")
        _reject_symlink_entries(resolved)
    else:
        if resolved.exists():
            raise ValueError(
                f"Texture run directory must not already exist: {resolved}"
            )
        try:
            resolved.mkdir(parents=True, exist_ok=False)
        except OSError as exc:
            raise ValueError(
                f"Unable to create Texture run directory: {resolved}"
            ) from exc
    return resolved


def _reject_symlink_entries(root: Path) -> None:
    """Reject symlinks before a durable run is read or rewritten."""

    try:
        for current_root, directory_names, file_names in os.walk(
            root,
            followlinks=False,
        ):
            current = Path(current_root)
            for name in (*directory_names, *file_names):
                candidate = current / name
                if candidate.is_symlink():
                    raise ValueError(
                        f"Texture run directory must not contain symlinks: {candidate}"
                    )
    except OSError as exc:
        raise ValueError(
            f"Unable to inspect Texture run directory for symlinks: {root}"
        ) from exc


def _normalize_prim_paths(values: list[str], option: str) -> tuple[str, ...]:
    normalized: list[str] = []
    for raw in values:
        value = str(raw).strip()
        if not value or not value.startswith("/") or value == "/":
            raise ValueError(f"{option} must be an absolute non-root USD prim path.")
        if value not in normalized:
            normalized.append(value)
    return tuple(normalized)


def _normalized_url(raw: str, option: str) -> str:
    value = str(raw).strip().rstrip("/")
    parsed = urlparse(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(f"{option} must be an http:// or https:// URL.")
    return value


def _validate_bearer_transport(
    url: str,
    *,
    credential_env: str | None,
    option: str,
) -> None:
    """Keep bearer credentials off cleartext non-loopback transports."""
    if credential_env is None:
        return
    parsed = urlparse(url)
    if parsed.scheme == "https":
        return
    host = parsed.hostname or ""
    if host.lower() == "localhost":
        return
    try:
        if ipaddress.ip_address(host).is_loopback:
            return
    except ValueError:
        pass
    raise ValueError(
        f"{option} must use https:// when --texture-agent-token-env is set "
        "for a non-loopback endpoint."
    )


def _optional_env_name(raw: str | None, option: str) -> str | None:
    if raw is None:
        return None
    value = str(raw).strip()
    if not value or not value.isidentifier():
        raise ValueError(f"{option} must name a valid environment variable.")
    return value


def _non_negative_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected an integer") from exc
    if value < 0:
        raise argparse.ArgumentTypeError("must be at least 0")
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
