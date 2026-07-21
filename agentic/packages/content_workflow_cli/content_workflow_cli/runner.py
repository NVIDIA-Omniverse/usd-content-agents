# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runner implementation for content-workflow-cli workflows."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import logging
import os
import queue
import re
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
import zipfile
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NoReturn, Protocol
from urllib.parse import urlparse
from uuid import uuid4

import yaml
from content_workbench_agent_client import client as workbench_client

from .prompts import (
    build_material_assignment_prompt,
    build_material_optimizer_selection_prompt,
    build_material_refinement_prompt,
    build_physics_apply_prompt,
    build_physics_optimizer_selection_prompt,
    build_physics_visual_refinement_prompt,
    build_skill_routed_material_assignment_prompt,
)
from .trace import (
    TraceWriter,
    UnsafeRunArtifactError,
    append_run_text,
    build_trace,
    utc_now,
)
from .workbench_tools.material_finalize import (
    MaterialFinalizeConfig,
    finalize_material_decisions,
)
from .workbench_tools.material_run_packet import (
    MaterialRunPacketConfig,
    packet_image_inputs,
    prepare_material_run_packet,
)

RUNNER_CODEX = "codex"
RUNNER_CLAUDE = "claude"
SUPPORTED_RUNNERS = {RUNNER_CODEX, RUNNER_CLAUDE}
CLAUDE_EXECUTION_SDK = "sdk"
CLAUDE_EXECUTION_CLI = "cli"
SUPPORTED_CLAUDE_EXECUTION_MODES = {CLAUDE_EXECUTION_SDK, CLAUDE_EXECUTION_CLI}
CLAUDE_CLI_BINARY_ENV = "CONTENT_AGENTS_CLAUDE_CLI_PATH"
# Keep in sync with claude_bridge.mjs DEFAULT_TOOLS. These read-only tools may
# run unattended. Bash is auto-approved separately only when the Claude OS
# sandbox is active and cannot be bypassed.
CLAUDE_CLI_ALLOWED_TOOLS = [
    "Read",
    "Glob",
    "Grep",
    "LS",
    "TodoWrite",
    "Skill",
]
CLAUDE_CLI_AVAILABLE_TOOLS = [
    "Bash",
    "Read",
    "Write",
    "Edit",
    "MultiEdit",
    "Glob",
    "Grep",
    "LS",
    "TodoWrite",
    "Skill",
]
CLAUDE_CLI_SANDBOXED_TOOLS = [*CLAUDE_CLI_ALLOWED_TOOLS, "Bash"]
CLAUDE_SANDBOXED_PERMISSION_MODES = {"acceptEdits", "bypassPermissions"}
# Keep in sync with claude_bridge.mjs BASE_SYSTEM_PROMPT_APPEND.
CLAUDE_CLI_SYSTEM_PROMPT_APPEND = (
    "You are running as a non-interactive child agent inside content-workflow-cli. "
    "Follow the user prompt artifact contract exactly. "
    "You are a single continuous turn with no later turn to deliver asynchronous "
    "notifications: tools like Monitor are not in your allowed toolset, and "
    "backgrounding a Bash command (run_in_background) will not report its result "
    "back to you either. To wait on a long-running command (for example a batch "
    "job), run it as one blocking Bash call, such as a shell loop that polls and "
    "sleeps until the work is done (e.g. `until <condition>; do sleep N; done`), "
    "or simply run it in the foreground and wait for it to exit. "
    "Bash commands run in a mandatory OS sandbox: use Bash for Content "
    "Workbench requests and for creating artifacts inside the run directory. "
    "The sandbox blocks writes outside that directory; only the configured "
    "Workbench host is pre-authorized for Bash network access. Use sandboxed "
    "Bash to read input paths outside the run directory; those paths are not "
    "added as writable Claude workspaces."
)
# Keep in sync with claude_bridge.mjs DANGEROUS_CLAUDE_ENV_KEYS.
CLAUDE_CLI_DANGEROUS_ENV_KEYS = {
    "ALL_PROXY",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "LD_PRELOAD",
    "LD_LIBRARY_PATH",
    "NODE_OPTIONS",
    "NO_PROXY",
    "PATH",
    "all_proxy",
    "http_proxy",
    "https_proxy",
    "no_proxy",
}
CODEX_SANDBOX_WORKSPACE_WRITE = "workspace-write"
SUPPORTED_CODEX_SANDBOX_MODES = {CODEX_SANDBOX_WORKSPACE_WRITE}
SUPPORTED_CODEX_AUTH_CREDENTIALS_STORES = {
    "auto",
    "ephemeral",
    "file",
    "keyring",
}
CHILD_HEARTBEAT_INTERVAL_SECONDS = 30.0
TERMINAL_SUCCESS_GRACE_SECONDS_ENV = "CONTENT_AGENTS_TERMINAL_SUCCESS_GRACE_SECONDS"
SCENE_TERMINAL_BRIDGE_PREFIXES = frozenset({"scene_run", "scene_resume"})


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


TERMINAL_SUCCESS_GRACE_SECONDS = _env_float(TERMINAL_SUCCESS_GRACE_SECONDS_ENV, 15.0)
WORKBENCH_WATCHDOG_INTERVAL_SECONDS = 15.0
WORKBENCH_HEALTH_FAILURE_LIMIT = 3
LIVE_PROGRESS_MAX_BYTES = 1024 * 1024
LIVE_PROGRESS_MAX_DIRECTORY_ENTRIES = 4096
MATERIAL_LIBRARY_ROOTS_ENV = "CONTENT_WORKBENCH_MATERIAL_LIBRARY_ROOTS"
ALLOW_FALLBACK_SUCCESS_ENV = "CONTENT_AGENTS_ALLOW_FALLBACK_SUCCESS"
DISABLE_FALLBACK_SUCCESS_ENV = "CONTENT_AGENTS_DISABLE_FALLBACK_SUCCESS"
DEFAULT_VQA_REFINEMENT_MAX_ITERATIONS = 3
DEFAULT_MATERIAL_RESTORE_TIMEOUT_SECONDS = 300.0
WORKBENCH_OUTPUT_STAGING_DIR_PREFIX = ".workbench-output-"
WORKBENCH_OUTPUT_STAGING_SHARED_MODE = 0o1777
WORKBENCH_OUTPUT_STAGING_PRIVATE_MODE = 0o700
WORKBENCH_RUN_DIR_TRAVERSE_BITS = 0o011
MATERIALIZED_OUTPUT_SUMMARY_START = (
    "<!-- content-workflow-cli:materialized-usd:start -->"
)
MATERIALIZED_OUTPUT_SUMMARY_END = "<!-- content-workflow-cli:materialized-usd:end -->"
OPTIMIZER_SELECTION_FIXED = "fixed"
OPTIMIZER_SELECTION_AGENT = "agent"
SUPPORTED_OPTIMIZER_SELECTION_MODES = {
    OPTIMIZER_SELECTION_FIXED,
    OPTIMIZER_SELECTION_AGENT,
}
PROMPT_MODE_LEGACY_EXPANDED = "legacy-expanded"
PROMPT_MODE_SKILL_ROUTED = "skill-routed"
SUPPORTED_PROMPT_MODES = {PROMPT_MODE_LEGACY_EXPANDED, PROMPT_MODE_SKILL_ROUTED}
DEFAULT_PROMPT_MODE = os.getenv(
    "CONTENT_AGENTS_PROMPT_MODE",
    PROMPT_MODE_LEGACY_EXPANDED,
)
ENV_TRUE_VALUES = {"1", "true", "yes", "on"}
ENV_FALSE_VALUES = {"0", "false", "no", "off"}
STEP_ARTIFACT_MANIFEST_SCHEMA_VERSION = "content-agents.workflow-step-manifest.v1"
STEP_ARTIFACT_SCHEMA_VERSION = "content-agents.workflow-step-artifacts.v1"
INCOMPLETE_STEP_SCHEMA_VERSION = "content-agents.workflow-incomplete-step.v1"
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WatchdogFailure:
    reason: str
    fatal: bool = False


class ChildProcessInterrupted(RuntimeError):
    """Raised when the wrapper receives a termination signal during a child run."""

    def __init__(self, signum: int, label: str) -> None:
        super().__init__(f"{label} interrupted by signal {signum}")
        self.signum = signum


class MaterialRestoreCoverageError(RuntimeError):
    """A durable restore response with structured unresolved prim mappings."""

    def __init__(
        self,
        message: str,
        *,
        unresolved_mappings: list[dict[str, Any]],
    ) -> None:
        super().__init__(message)
        self.unresolved_mappings = unresolved_mappings


class AgentRuntimeConfig(Protocol):
    """Runner fields shared by every wrapper-launched agent workflow."""

    @property
    def repo_root(self) -> Path: ...

    @property
    def agent_cwd(self) -> Path | None: ...

    @property
    def usd_path(self) -> Path: ...

    @property
    def reference_images(self) -> list[Path]: ...

    @property
    def reference_files(self) -> list[Path] | None: ...

    @property
    def workbench_url(self) -> str: ...

    @property
    def runner(self) -> str: ...

    @property
    def model(self) -> str | None: ...

    @property
    def model_reasoning_effort(self) -> str | None: ...

    @property
    def codex_base_url(self) -> str | None: ...

    @property
    def codex_sandbox_mode(self) -> str: ...

    @property
    def codex_config(self) -> dict[str, object] | None: ...

    @property
    def claude_config(self) -> dict[str, object] | None: ...

    @property
    def claude_permission_mode(self) -> str: ...

    @property
    def claude_max_turns(self) -> int | None: ...

    @property
    def claude_execution_mode(self) -> str: ...

    @property
    def child_timeout_seconds(self) -> float: ...


@dataclass(frozen=True)
class MaterialAssignConfig:
    repo_root: Path
    usd_path: Path
    reference_images: list[Path]
    materials_yaml: Path
    materials_usd: Path
    workbench_url: str
    reference_files: list[Path] | None = None
    output_dir: Path | None = None
    output_usd_path: Path | None = None
    default_output_root: Path | None = None
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
    dry_run: bool = False
    optimize: bool = True
    optimizer_selection: str = OPTIMIZER_SELECTION_FIXED
    root_prim_path: str | None = None
    material_candidate_space: str = "source"
    skip_instances: bool = True
    skip_prototypes: bool = False
    skip_invisible: bool = False
    flatten_prototypes: bool | None = None
    enable_deinstance: bool | None = None
    enable_split: bool | None = None
    enable_deduplicate: bool | None = None
    preflight: bool = True
    respect_existing_material_bindings: bool = False
    start_workbench: bool = False
    keep_workbench: bool = False
    workbench_timeout_seconds: float = 60.0
    material_restore_timeout_seconds: float = DEFAULT_MATERIAL_RESTORE_TIMEOUT_SECONDS
    child_timeout_seconds: float = 1800.0
    prompt_mode: str = DEFAULT_PROMPT_MODE
    vqa_refinement_max_iterations: int = DEFAULT_VQA_REFINEMENT_MAX_ITERATIONS
    codex_persistent_refinement: bool = False
    additional_instructions: str | None = None
    agent_cwd: Path | None = None


@dataclass(frozen=True)
class PhysicsApplyConfig:
    repo_root: Path
    usd_path: Path
    workbench_url: str
    reference_images: list[Path] = field(default_factory=list)
    reference_files: list[Path] = field(default_factory=list)
    output_dir: Path | None = None
    output_usd_path: Path | None = None
    collision_approximation: str = "convexHull"
    run_simulation: bool = True
    simulation_engine: str = "ovphysx"
    simulation_duration_s: float = 1.0
    simulation_dt: float = 1.0 / 240.0
    simulation_sample_fps: int = 30
    drop_height_m: float | None = None
    fail_on_validation_error: bool = False
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
    dry_run: bool = False
    optimize: bool = True
    optimizer_selection: str = OPTIMIZER_SELECTION_FIXED
    flatten_prototypes: bool | None = None
    enable_deinstance: bool | None = None
    enable_split: bool | None = None
    enable_deduplicate: bool | None = None
    start_workbench: bool = False
    keep_workbench: bool = False
    workbench_timeout_seconds: float = 60.0
    child_timeout_seconds: float = 1800.0
    prompt_mode: str = DEFAULT_PROMPT_MODE
    vqa_refinement_max_iterations: int = DEFAULT_VQA_REFINEMENT_MAX_ITERATIONS
    codex_persistent_refinement: bool = False
    additional_instructions: str | None = None
    agent_cwd: Path | None = None
    log_to_stderr: bool = False


@dataclass(frozen=True)
class RunResult:
    run_dir: Path
    prompt_path: Path
    request_path: Path
    child_output_path: Path
    child_final_path: Path
    returncode: int
    trace_paths: dict[str, str]


def run_material_assignment(config: MaterialAssignConfig) -> RunResult:
    _validate_config(config)
    run_started_monotonic = time.monotonic()
    run_dir = _prepare_run_dir(config)
    trace_writer = TraceWriter(run_dir)
    request_path = run_dir / "request.json"
    prompt_path = run_dir / "agent_prompt.md"
    child_output_path = run_dir / "child-output.log"
    child_final_path = run_dir / "child-final.md"

    request = _build_request(config, run_dir)
    request_path.write_text(json.dumps(request, indent=2), encoding="utf-8")
    _print_run_start(
        run_dir=run_dir,
        request_path=request_path,
        child_output_path=child_output_path,
    )
    trace_writer.write(
        "run_created",
        phase="setup",
        summary="Created content-workflow-cli run directory and request metadata.",
        artifacts=[str(request_path)],
        data={"run_dir": str(run_dir), "workflow": request["workflow"]},
    )

    if config.dry_run:
        prompt = _build_material_assignment_child_prompt(
            config=config,
            run_dir=run_dir,
            preflight_packet=None,
        )
        prompt_path.write_text(prompt, encoding="utf-8")
        trace_writer.write(
            "prompt_written",
            phase="setup",
            summary="Wrote the child-agent prompt and artifact contract.",
            artifacts=[str(prompt_path)],
        )
        trace_paths = build_trace(run_dir)
        return RunResult(
            run_dir=run_dir,
            prompt_path=prompt_path,
            request_path=request_path,
            child_output_path=child_output_path,
            child_final_path=child_final_path,
            returncode=0,
            trace_paths=_trace_path_summary(trace_paths),
        )

    managed_workbench: ManagedWorkbench | None = None
    preflight_packet: dict[str, Any] | None = None
    codex_session: CodexThreadSession | None = None
    try:
        if config.start_workbench:
            managed_workbench = ManagedWorkbench(
                repo_root=config.repo_root,
                workbench_url=config.workbench_url,
                run_dir=run_dir,
                timeout_seconds=config.workbench_timeout_seconds,
                material_library_roots=[
                    config.usd_path.parent,
                    config.materials_usd.parent,
                ],
            )
            managed_workbench.start()
            trace_writer.write(
                "workbench_started",
                phase="setup",
                summary="Started or reused a Content Workbench service for this run.",
                artifacts=[str(run_dir / "workbench.log")],
                data={"workbench_url": config.workbench_url},
            )
        else:
            wait_for_workbench(
                config.workbench_url,
                timeout_seconds=config.workbench_timeout_seconds,
                output_root=run_dir,
            )
            trace_writer.write(
                "workbench_reachable",
                phase="setup",
                summary="Verified the configured Content Workbench endpoint is reachable.",
                data={"workbench_url": config.workbench_url},
            )

        prompt_image_inputs: list[dict[str, str]] = []
        if config.optimizer_selection == OPTIMIZER_SELECTION_AGENT:
            config, optimizer_decision = _run_material_optimizer_selection(
                config=config,
                run_dir=run_dir,
                trace_writer=trace_writer,
                managed_workbench=managed_workbench,
            )
            request = _build_request(config, run_dir)
            request["optimizer_selection_decision"] = optimizer_decision
            request_path.write_text(json.dumps(request, indent=2), encoding="utf-8")

        if config.preflight:
            try:
                preflight_packet = prepare_material_run_packet(
                    MaterialRunPacketConfig(
                        workbench_url=config.workbench_url,
                        run_dir=run_dir,
                        usd_path=config.usd_path,
                        materials_yaml=config.materials_yaml,
                        materials_usd=config.materials_usd,
                        optimize=config.optimize,
                        root_prim_path=config.root_prim_path,
                        material_candidate_space=config.material_candidate_space,
                        skip_instances=config.skip_instances,
                        skip_prototypes=config.skip_prototypes,
                        skip_invisible=config.skip_invisible,
                        flatten_prototypes=config.flatten_prototypes,
                        enable_deinstance=config.enable_deinstance,
                        enable_split=config.enable_split,
                        enable_deduplicate=config.enable_deduplicate,
                        respect_existing_material_bindings=(
                            config.respect_existing_material_bindings
                        ),
                    )
                )
            except Exception as exc:
                hint = _material_library_roots_recovery_hint(config, exc)
                if hint is not None:
                    trace_writer.write(
                        "preflight_failed",
                        phase="setup",
                        summary=(
                            "Content Workbench rejected the configured material "
                            "library allowlist during preflight."
                        ),
                        data={
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "materials_usd": str(config.materials_usd),
                            "materials_root": str(config.materials_usd.parent),
                            "allowlist_env": MATERIAL_LIBRARY_ROOTS_ENV,
                        },
                    )
                    raise RuntimeError(hint) from exc
                raise
            prompt_image_inputs = packet_image_inputs(preflight_packet)
            request["preflight"] = {
                "enabled": True,
                "packet_path": str(run_dir / "raw" / "material_run_packet.json"),
                "image_inputs": prompt_image_inputs,
            }
            request_path.write_text(json.dumps(request, indent=2), encoding="utf-8")
            trace_writer.write(
                "preflight_completed",
                phase="setup",
                summary=(
                    "Prepared reusable Workbench material-run packet before "
                    "launching the child agent."
                ),
                artifacts=[str(run_dir / "raw" / "material_run_packet.json")],
                data={"image_inputs": prompt_image_inputs},
            )
        else:
            request["preflight"] = {"enabled": False}
            request_path.write_text(json.dumps(request, indent=2), encoding="utf-8")

        prompt = _build_material_assignment_child_prompt(
            config=config,
            run_dir=run_dir,
            preflight_packet=preflight_packet,
        )
        prompt_path.write_text(prompt, encoding="utf-8")
        trace_writer.write(
            "prompt_written",
            phase="setup",
            summary="Wrote the child-agent prompt and artifact contract.",
            artifacts=[str(prompt_path)],
            data={
                "preflight_enabled": bool(preflight_packet),
                "prompt_mode": config.prompt_mode,
            },
        )

        try:
            if _should_start_codex_thread_session(config):
                codex_session = CodexThreadSession(
                    config=config,
                    run_dir=run_dir,
                    managed_workbench=managed_workbench,
                )
                trace_writer.write(
                    "codex_thread_started",
                    phase="runner",
                    summary=(
                        "Started a persistent Codex SDK thread for initial and "
                        "VQA refinement turns."
                    ),
                    artifacts=[
                        str(run_dir / "raw" / "codex_thread_session.json"),
                        str(run_dir / "raw" / "codex_thread_session.log"),
                    ],
                )
            returncode = _run_child_agent(
                config=config,
                prompt=prompt,
                run_dir=run_dir,
                child_output_path=child_output_path,
                child_final_path=child_final_path,
                managed_workbench=managed_workbench,
                prompt_image_inputs=prompt_image_inputs,
                codex_session=codex_session,
            )
            trace_writer.write(
                "child_agent_finished",
                phase="runner",
                summary="Child agent process exited.",
                artifacts=[str(child_output_path), str(child_final_path)],
                data={"returncode": returncode},
            )
        except UnsafeRunArtifactError:
            raise
        except Exception as exc:  # noqa: BLE001 - preserve trace on runner failure
            returncode = 2
            _append_child_runner_error(child_output_path, exc, run_dir=run_dir)
            trace_writer.write(
                "child_agent_failed",
                phase="runner",
                summary="Child agent runner failed before completion.",
                artifacts=[str(child_output_path), str(child_final_path)],
                data={
                    "returncode": returncode,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )

        try:
            _finalize_structured_material_decisions(
                config=config,
                run_dir=run_dir,
                preflight_packet=preflight_packet,
                trace_writer=trace_writer,
            )
        except UnsafeRunArtifactError:
            raise
        except Exception as exc:  # noqa: BLE001 - preserve run cleanup/trace
            if returncode == 0:
                returncode = 2
            trace_writer.write(
                "warning",
                phase="artifact finalization",
                summary="Deterministic material-decision finalizer failed.",
                artifacts=[
                    str(run_dir / "raw" / "material_decision_finalizer_error.json")
                ],
                data={
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "effective_returncode": returncode,
                },
            )

        finalized = _ensure_material_assignment_artifacts(
            config=config,
            run_dir=run_dir,
            request=request,
            trace_writer=trace_writer,
            child_output_path=child_output_path,
            child_final_path=child_final_path,
            child_returncode=returncode,
        )
        if finalized:
            _snapshot_material_step_artifacts(
                run_dir=run_dir,
                trace_writer=trace_writer,
                step_id="01_initial_prediction",
                step_role="initial_prediction",
                iteration=1,
                prompt_path=prompt_path,
                child_output_path=child_output_path,
                child_final_path=child_final_path,
                bridge_artifact_prefix=config.runner,
                summary=(
                    "Captured canonical artifacts after the initial child material "
                    "prediction and deterministic finalization."
                ),
            )
        if returncode == 0 and finalized:
            refinement_codex_session = codex_session
            refinement_returncode = _run_vqa_refinement_loop(
                config=config,
                run_dir=run_dir,
                request=request,
                preflight_packet=preflight_packet,
                trace_writer=trace_writer,
                managed_workbench=managed_workbench,
                initial_child_output_path=child_output_path,
                initial_child_final_path=child_final_path,
                prompt_image_inputs=prompt_image_inputs,
                codex_session=refinement_codex_session,
            )
            if refinement_returncode != 0:
                returncode = refinement_returncode
            finalized = _ensure_material_assignment_artifacts(
                config=config,
                run_dir=run_dir,
                request=request,
                trace_writer=trace_writer,
                child_output_path=child_output_path,
                child_final_path=child_final_path,
                child_returncode=returncode,
            )
        fallback_success_enabled = _fallback_success_enabled()
        if returncode != 0 and finalized:
            trace_writer.write(
                "warning",
                phase="runner",
                summary=(
                    "Child agent did not finish cleanly, but the wrapper recovered "
                    "the required material-assignment artifacts from observable "
                    "Workbench trace data."
                ),
                artifacts=[
                    str(run_dir / "assignments.json"),
                    str(run_dir / "api_operation_counts.json"),
                    str(run_dir / "final_summary.md"),
                ],
                data={
                    "original_returncode": returncode,
                    "fallback_recovered_returncode": 0,
                    "effective_returncode": 0
                    if fallback_success_enabled
                    else returncode,
                    "fallback_success_enabled": fallback_success_enabled,
                    "allow_env": ALLOW_FALLBACK_SUCCESS_ENV,
                    "legacy_disable_env": DISABLE_FALLBACK_SUCCESS_ENV,
                },
            )
            if fallback_success_enabled:
                returncode = 0
        if returncode == 0 and finalized and config.output_usd_path is not None:
            try:
                _record_materialized_output_status(
                    run_dir=run_dir,
                    output_usd_path=config.output_usd_path,
                    status="pending",
                )
            except Exception as status_exc:  # noqa: BLE001 - retain diagnostics
                returncode = 2
                trace_writer.write(
                    "error",
                    phase="artifact finalization",
                    summary=(
                        "Failed to record pending materialized USD status before "
                        "restore."
                    ),
                    artifacts=[
                        str(run_dir / "assignments.json"),
                        str(run_dir / "final_summary.md"),
                    ],
                    data={
                        "error_type": type(status_exc).__name__,
                        "error": str(status_exc),
                    },
                )
            else:
                try:
                    restored_output_path = _restore_materialized_output(
                        config=config,
                        run_dir=run_dir,
                        preflight_packet=preflight_packet,
                        trace_writer=trace_writer,
                    )
                except Exception as exc:  # noqa: BLE001 - preserve cleanup and trace
                    returncode = 2
                    annotation_error: Exception | None = None
                    try:
                        _record_materialized_output_status(
                            run_dir=run_dir,
                            output_usd_path=config.output_usd_path,
                            status="failed",
                            error=exc,
                        )
                    except Exception as status_exc:  # noqa: BLE001 - retain root error
                        annotation_error = status_exc
                    trace_writer.write(
                        "error",
                        phase="material output",
                        summary="Failed to restore accepted materials to durable USD.",
                        artifacts=[
                            str(run_dir / "assignments.json"),
                            str(run_dir / "final_summary.md"),
                        ],
                        data={
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            **(
                                {
                                    "artifact_status_error_type": type(
                                        annotation_error
                                    ).__name__,
                                    "artifact_status_error": str(annotation_error),
                                }
                                if annotation_error is not None
                                else {}
                            ),
                        },
                    )
                else:
                    try:
                        _record_materialized_output_status(
                            run_dir=run_dir,
                            output_usd_path=restored_output_path,
                            status="succeeded",
                        )
                    except Exception as status_exc:  # noqa: BLE001 - output is durable
                        returncode = 2
                        trace_writer.write(
                            "error",
                            phase="artifact finalization",
                            summary=(
                                "Restored durable USD but failed to record its "
                                "successful materialization status."
                            ),
                            artifacts=[
                                str(restored_output_path),
                                str(run_dir / "assignments.json"),
                                str(run_dir / "final_summary.md"),
                            ],
                            data={
                                "error_type": type(status_exc).__name__,
                                "error": str(status_exc),
                            },
                        )
    finally:
        if codex_session is not None:
            close_error: Exception | None = None
            try:
                codex_session.close()
            except UnsafeRunArtifactError:
                raise
            except Exception as exc:  # noqa: BLE001 - cleanup must not mask the run
                close_error = exc
                logger.warning(
                    "Failed to stop persistent Codex SDK thread cleanly: %s",
                    exc,
                    exc_info=True,
                )
            trace_writer.write(
                "codex_thread_stopped",
                phase="cleanup",
                summary=(
                    "Stopped the persistent Codex SDK thread."
                    if close_error is None
                    else "Failed to stop the persistent Codex SDK thread cleanly."
                ),
                artifacts=[str(run_dir / "raw" / "codex_thread_session.log")],
                data=(
                    {}
                    if close_error is None
                    else {
                        "error_type": type(close_error).__name__,
                        "error": str(close_error),
                    }
                ),
            )
        session_id = _preflight_session_id(preflight_packet)
        if session_id:
            close_error = None
            try:
                close_workbench_session(config.workbench_url, session_id)
            except Exception as exc:  # noqa: BLE001 - cleanup must not mask the run
                close_error = exc
                logger.warning(
                    "Failed to close Content Workbench session %s: %s",
                    session_id,
                    exc,
                    exc_info=True,
                )
            trace_writer.write(
                "workbench_session_closed",
                phase="cleanup",
                summary=(
                    "Closed the preflight Content Workbench session."
                    if close_error is None
                    else "Failed to close the preflight Content Workbench session."
                ),
                data=(
                    {"session_id": session_id}
                    if close_error is None
                    else {
                        "session_id": session_id,
                        "error_type": type(close_error).__name__,
                        "error": str(close_error),
                    }
                ),
            )
        if managed_workbench and not config.keep_workbench:
            stop_error: Exception | None = None
            try:
                managed_workbench.stop()
            except Exception as exc:  # noqa: BLE001 - cleanup must not mask the run
                stop_error = exc
                logger.warning(
                    "Failed to stop Content Workbench process cleanly: %s",
                    exc,
                    exc_info=True,
                )
            trace_writer.write(
                "workbench_stopped",
                phase="cleanup",
                summary=(
                    "Stopped the Content Workbench service started by the wrapper."
                    if stop_error is None
                    else "Content Workbench stop failed after the run completed."
                ),
                artifacts=[str(run_dir / "workbench.log")],
                data=(
                    {}
                    if stop_error is None
                    else {
                        "error_type": type(stop_error).__name__,
                        "error": str(stop_error),
                    }
                ),
            )

    _write_run_cost_metrics(
        config=config,
        run_dir=run_dir,
        request=request,
        wall_time_seconds=time.monotonic() - run_started_monotonic,
    )
    trace_paths = build_trace(run_dir)
    return RunResult(
        run_dir=run_dir,
        prompt_path=prompt_path,
        request_path=request_path,
        child_output_path=child_output_path,
        child_final_path=child_final_path,
        returncode=returncode,
        trace_paths=_trace_path_summary(trace_paths),
    )


def run_physics_apply(config: PhysicsApplyConfig) -> RunResult:
    _validate_physics_config(config)
    run_started_monotonic = time.monotonic()
    run_dir = _prepare_run_dir(config)  # type: ignore[arg-type]
    trace_writer = TraceWriter(run_dir)
    request_path = run_dir / "request.json"
    prompt_path = run_dir / "agent_prompt.md"
    child_output_path = run_dir / "child-output.log"
    child_final_path = run_dir / "child-final.md"

    request = _build_physics_request(config, run_dir)
    request_path.write_text(json.dumps(request, indent=2), encoding="utf-8")
    _print_run_start(
        run_dir=run_dir,
        request_path=request_path,
        child_output_path=child_output_path,
        stream=_console_stream(config),
    )
    trace_writer.write(
        "run_created",
        phase="setup",
        summary="Created content-workflow-cli physics run directory.",
        artifacts=[str(request_path)],
        data={"run_dir": str(run_dir), "workflow": request["workflow"]},
    )

    if config.dry_run:
        prompt = build_physics_apply_prompt(
            repo_root=config.repo_root,
            run_dir=run_dir,
            usd_path=config.usd_path,
            workbench_url=config.workbench_url,
            session_id="<dry-run-session>",
            reference_images=config.reference_images,
            reference_files=config.reference_files,
            additional_instructions=config.additional_instructions,
            collision_approximation=config.collision_approximation,
            visual_validation_max_iterations=config.vqa_refinement_max_iterations,
        )
        prompt_path.write_text(prompt, encoding="utf-8")
        trace_writer.write(
            "prompt_written",
            phase="setup",
            summary="Wrote the physics child-agent prompt and artifact contract.",
            artifacts=[str(prompt_path)],
            data={"dry_run": True},
        )
        trace_paths = build_trace(run_dir)
        return RunResult(
            run_dir=run_dir,
            prompt_path=prompt_path,
            request_path=request_path,
            child_output_path=child_output_path,
            child_final_path=child_final_path,
            returncode=0,
            trace_paths=_trace_path_summary(trace_paths),
        )

    managed_workbench: ManagedWorkbench | None = None
    session_id: str | None = None
    codex_session: CodexThreadSession | None = None
    returncode = 0
    try:
        if config.start_workbench:
            managed_workbench = ManagedWorkbench(
                repo_root=config.repo_root,
                workbench_url=config.workbench_url,
                run_dir=run_dir,
                timeout_seconds=config.workbench_timeout_seconds,
                material_library_roots=[config.usd_path.parent],
            )
            managed_workbench.start()
            trace_writer.write(
                "workbench_started",
                phase="setup",
                summary="Started or reused a Content Workbench service for this run.",
                artifacts=[str(run_dir / "workbench.log")],
                data={"workbench_url": config.workbench_url},
            )
        else:
            wait_for_workbench(
                config.workbench_url,
                timeout_seconds=config.workbench_timeout_seconds,
                output_root=run_dir,
            )
            trace_writer.write(
                "workbench_reachable",
                phase="setup",
                summary="Verified the configured Content Workbench endpoint is reachable.",
                data={"workbench_url": config.workbench_url},
            )

        if config.optimizer_selection == OPTIMIZER_SELECTION_AGENT:
            config, optimizer_decision = _run_physics_optimizer_selection(
                config=config,
                run_dir=run_dir,
                trace_writer=trace_writer,
                managed_workbench=managed_workbench,
            )
            request = _build_physics_request(config, run_dir)
            request["optimizer_selection_decision"] = optimizer_decision
            request_path.write_text(json.dumps(request, indent=2), encoding="utf-8")

        physics_packet = _prepare_physics_run_packet(config, run_dir)
        session_id = str(physics_packet["session_id"])
        request["preflight"] = {
            "enabled": True,
            "packet_path": str(run_dir / "raw" / "physics_run_packet.json"),
            "session_id": session_id,
        }
        request_path.write_text(json.dumps(request, indent=2), encoding="utf-8")
        trace_writer.write(
            "preflight_completed",
            phase="setup",
            summary="Prepared reusable Workbench physics-run packet.",
            artifacts=[
                str(run_dir / "raw" / "physics_run_packet.json"),
                str(run_dir / "raw" / "physics_components.json"),
                str(run_dir / "raw" / "physics_topology.json"),
            ],
            data={"session_id": session_id},
        )

        prompt = build_physics_apply_prompt(
            repo_root=config.repo_root,
            run_dir=run_dir,
            usd_path=config.usd_path,
            workbench_url=config.workbench_url,
            session_id=session_id,
            reference_images=config.reference_images or [],
            reference_files=config.reference_files or [],
            additional_instructions=config.additional_instructions,
            collision_approximation=config.collision_approximation,
            visual_validation_max_iterations=config.vqa_refinement_max_iterations,
        )
        prompt_path.write_text(prompt, encoding="utf-8")
        trace_writer.write(
            "prompt_written",
            phase="setup",
            summary="Wrote the physics child-agent prompt and artifact contract.",
            artifacts=[str(prompt_path)],
            data={"session_id": session_id},
        )

        try:
            if _should_start_codex_thread_session(config):  # type: ignore[arg-type]
                codex_session = CodexThreadSession(
                    config=config,  # type: ignore[arg-type]
                    run_dir=run_dir,
                    managed_workbench=managed_workbench,
                )
                trace_writer.write(
                    "codex_thread_started",
                    phase="runner",
                    summary=(
                        "Started a persistent Codex SDK thread for physics "
                        "visual validation/refinement turns."
                    ),
                    artifacts=[
                        str(run_dir / "raw" / "codex_thread_session.json"),
                        str(run_dir / "raw" / "codex_thread_session.log"),
                    ],
                )
            returncode = _run_child_agent(
                config=config,
                prompt=prompt,
                run_dir=run_dir,
                child_output_path=child_output_path,
                child_final_path=child_final_path,
                managed_workbench=managed_workbench,
                prompt_image_inputs=[],
                codex_session=codex_session,
            )
            trace_writer.write(
                "child_agent_finished",
                phase="runner",
                summary="Physics decision child agent process exited.",
                artifacts=[str(child_output_path), str(child_final_path)],
                data={"returncode": returncode},
            )
        except UnsafeRunArtifactError:
            raise
        except Exception as exc:  # noqa: BLE001 - preserve trace on runner failure
            returncode = 2
            _append_child_runner_error(child_output_path, exc, run_dir=run_dir)
            trace_writer.write(
                "child_agent_failed",
                phase="runner",
                summary="Physics decision child agent runner failed.",
                artifacts=[str(child_output_path), str(child_final_path)],
                data={"error_type": type(exc).__name__, "error": str(exc)},
            )

        patch_path = run_dir / "raw" / "physics_decision_patch.json"
        if patch_path.exists():
            try:
                refinement_returncode, session_id = _run_physics_visual_refinement_loop(
                    config=config,
                    run_dir=run_dir,
                    session_id=session_id,
                    trace_writer=trace_writer,
                    managed_workbench=managed_workbench,
                    codex_session=codex_session,
                )
                if returncode == 0:
                    returncode = refinement_returncode
            except UnsafeRunArtifactError:
                raise
            except Exception as exc:  # noqa: BLE001 - preserve trace on finalizer failure
                if returncode == 0:
                    returncode = 2
                trace_writer.write(
                    "physics_finalization_failed",
                    phase="finalization",
                    summary="Physics finalization or visual validation failed.",
                    artifacts=[str(patch_path)],
                    data={"error_type": type(exc).__name__, "error": str(exc)},
                )
        elif returncode == 0:
            returncode = 2
            trace_writer.write(
                "warning",
                phase="finalization",
                summary="Physics child finished without raw/physics_decision_patch.json.",
                data={"missing_artifact": str(patch_path)},
            )

    finally:
        if codex_session is not None:
            codex_session.close()
        if session_id and not config.keep_workbench:
            try:
                close_workbench_session(
                    config.workbench_url,
                    session_id,
                    timeout=config.workbench_timeout_seconds,
                )
                trace_writer.write(
                    "workbench_session_closed",
                    phase="cleanup",
                    summary="Closed the Workbench session for this physics run.",
                    data={"session_id": session_id},
                )
            except Exception as exc:  # noqa: BLE001 - cleanup best effort
                trace_writer.write(
                    "warning",
                    phase="cleanup",
                    summary="Failed to close Workbench session after physics run.",
                    data={"error_type": type(exc).__name__, "error": str(exc)},
                )
        if managed_workbench is not None and not config.keep_workbench:
            try:
                managed_workbench.stop()
                trace_writer.write(
                    "workbench_stopped",
                    phase="cleanup",
                    summary="Stopped the managed Content Workbench service.",
                    artifacts=[str(run_dir / "workbench.log")],
                )
            except Exception as exc:  # noqa: BLE001 - cleanup best effort
                trace_writer.write(
                    "warning",
                    phase="cleanup",
                    summary="Failed to stop managed Content Workbench service.",
                    artifacts=[str(run_dir / "workbench.log")],
                    data={"error_type": type(exc).__name__, "error": str(exc)},
                )

    _write_run_cost_metrics(
        config=config,
        run_dir=run_dir,
        request=request,
        wall_time_seconds=time.monotonic() - run_started_monotonic,
    )
    trace_paths = build_trace(run_dir)
    return RunResult(
        run_dir=run_dir,
        prompt_path=prompt_path,
        request_path=request_path,
        child_output_path=child_output_path,
        child_final_path=child_final_path,
        returncode=returncode,
        trace_paths=_trace_path_summary(trace_paths),
    )


class ManagedWorkbench:
    """Manage a local Content Workbench uvicorn process."""

    def __init__(
        self,
        *,
        repo_root: Path,
        workbench_url: str,
        run_dir: Path,
        timeout_seconds: float,
        material_library_roots: list[Path] | None = None,
    ) -> None:
        self.repo_root = repo_root
        self.workbench_url = workbench_url.rstrip("/")
        self.run_dir = run_dir
        self.timeout_seconds = timeout_seconds
        self.material_library_roots = material_library_roots or []
        self.process: subprocess.Popen[str] | None = None
        self.log_stream = None

    def start(self) -> None:
        if is_workbench_healthy(self.workbench_url):
            wait_for_workbench(
                self.workbench_url,
                timeout_seconds=self.timeout_seconds,
                output_root=self.run_dir,
            )
            return

        parsed = urlparse(self.workbench_url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 8088
        if not _is_loopback_host(host):
            raise ValueError(
                "Content Workbench auto-start only supports loopback hosts; "
                f"got {host!r} from {self.workbench_url}."
            )
        log_path = self.run_dir / "workbench.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_stream = log_path.open("w", encoding="utf-8")
        try:
            env = os.environ.copy()
            env["PYTHONPATH"] = _prepend_pythonpath(
                env.get("PYTHONPATH"),
                *_content_workbench_pythonpath_roots(self.repo_root),
            )
            env["CONTENT_WORKBENCH_MATERIAL_LIBRARY_ROOTS"] = _append_env_paths(
                env.get("CONTENT_WORKBENCH_MATERIAL_LIBRARY_ROOTS"),
                self.material_library_roots,
            )
            # This sidecar serves an untrusted child agent. Never inherit a
            # broader output allowlist from the launcher environment.
            env["CONTENT_WORKBENCH_OUTPUT_ROOTS"] = str(self.run_dir.resolve())
            command = [
                sys.executable,
                "-m",
                "uvicorn",
                "content_workbench.main:app",
                "--host",
                host,
                "--port",
                str(port),
            ]
            self.process = subprocess.Popen(
                command,
                cwd=self.repo_root,
                env=env,
                stdout=self.log_stream,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
                start_new_session=hasattr(os, "setsid"),
            )
            wait_for_workbench(
                self.workbench_url,
                timeout_seconds=self.timeout_seconds,
                output_root=self.run_dir,
            )
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        terminate_error: Exception | None = None
        try:
            if self.process is not None and self.process.poll() is None:
                _terminate_subprocess(self.process)
        except Exception as exc:  # noqa: BLE001 - close log before surfacing cleanup
            terminate_error = exc
        finally:
            if self.log_stream is not None:
                self.log_stream.close()
                self.log_stream = None
        if terminate_error is not None:
            raise terminate_error

    def returncode(self) -> int | None:
        if self.process is None:
            return None
        return self.process.poll()


def wait_for_workbench(
    workbench_url: str,
    *,
    timeout_seconds: float,
    output_root: Path | None = None,
) -> None:
    workbench_client.wait_until_healthy(
        workbench_url,
        timeout_seconds=timeout_seconds,
        output_root=output_root,
    )


def close_workbench_session(
    workbench_url: str, session_id: str, *, timeout: float = 300.0
) -> None:
    workbench_client.close_session(workbench_url, session_id, timeout=timeout)


def is_workbench_healthy(
    workbench_url: str,
    *,
    output_root: Path | None = None,
) -> bool:
    return workbench_client.is_healthy(workbench_url, output_root=output_root)


def _material_library_roots_recovery_hint(
    config: MaterialAssignConfig,
    error: Exception,
) -> str | None:
    error_text = _exception_chain_text(error)
    if MATERIAL_LIBRARY_ROOTS_ENV not in error_text:
        return None
    if "Material library path is outside" not in error_text:
        return None

    source_root = config.usd_path.parent.resolve()
    materials_root = config.materials_usd.parent.resolve()
    allowlist = _append_env_paths(None, [source_root, materials_root])
    if config.start_workbench:
        recovery = (
            "A Workbench endpoint was already healthy, so the wrapper reused it "
            "instead of launching a sidecar with the run-specific material "
            "allowlist. Stop the existing local Workbench and rerun, or restart "
            "it manually with the allowlist below."
        )
    else:
        recovery = (
            "The wrapper is using an existing Workbench endpoint. Restart that "
            "Workbench on the service host with the allowlist below, or stop the "
            "local service and let the wrapper start a managed sidecar."
        )

    return (
        "Content Workbench rejected the material library because it is outside "
        f"{MATERIAL_LIBRARY_ROOTS_ENV}.\n"
        f"Material library USD: {config.materials_usd}\n"
        f"Required material library root: {materials_root}\n"
        f"{recovery}\n"
        "Suggested Workbench host environment:\n"
        f"  export {MATERIAL_LIBRARY_ROOTS_ENV}={shlex.quote(allowlist)}\n"
        "Original Workbench error:\n"
        f"  {error}"
    )


def _exception_chain_text(error: BaseException) -> str:
    messages: list[str] = []
    current: BaseException | None = error
    while current is not None:
        messages.append(str(current))
        current = current.__cause__ or current.__context__
    return "\n".join(messages)


def _preflight_session_id(preflight_packet: dict[str, Any] | None) -> str | None:
    if not isinstance(preflight_packet, dict):
        return None
    session_id = preflight_packet.get("session_id")
    return session_id if isinstance(session_id, str) and session_id else None


def _is_loopback_host(hostname: str | None) -> bool:
    if hostname == "localhost":
        return True
    if hostname is None:
        return False
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _make_workbench_watchdog(
    workbench_url: str,
    managed_workbench: ManagedWorkbench | None,
    *,
    output_root: Path,
) -> Callable[[], WatchdogFailure | None]:
    health_url = f"{workbench_url.rstrip('/')}/healthz"

    def watchdog() -> WatchdogFailure | None:
        if managed_workbench is not None:
            returncode = managed_workbench.returncode()
            if returncode is not None:
                return WatchdogFailure(
                    f"Content Workbench process exited with return code {returncode}.",
                    fatal=True,
                )
        try:
            workbench_client.check_health(
                workbench_url,
                output_root=output_root,
            )
        except workbench_client.WorkbenchHealthConfigurationError as exc:
            return WatchdogFailure(
                f"Content Workbench endpoint configuration changed: {exc}",
                fatal=True,
            )
        except RuntimeError:
            return WatchdogFailure(
                f"Content Workbench endpoint is not healthy: {health_url}",
                fatal=False,
            )
        return None

    return watchdog


def find_repo_root(start: Path | None = None) -> Path:
    cwd = (start or Path.cwd()).resolve()
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            errors="replace",
        )
        return Path(completed.stdout.strip()).resolve()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return cwd


def _agent_working_directory(config: AgentRuntimeConfig, run_dir: Path) -> Path:
    """Confine every child runner to its resolved run directory."""

    resolved_run_dir = run_dir.resolve()
    configured_cwd = getattr(config, "agent_cwd", None)
    if (
        configured_cwd is not None
        and Path(configured_cwd).resolve() != resolved_run_dir
    ):
        raise ValueError(
            "Child agent working directory must resolve to the run directory: "
            f"{Path(configured_cwd).resolve()} != {resolved_run_dir}"
        )
    return resolved_run_dir


def _stage_agent_skills(config: AgentRuntimeConfig, run_dir: Path) -> None:
    """Copy trusted project skills into both child-agent discovery roots."""

    resolved_run_dir = run_dir.resolve()
    target_agents = resolved_run_dir / ".agents"
    target_claude = resolved_run_dir / ".claude"
    target_skill_roots = [target_agents / "skills", target_claude / "skills"]
    configured_workspace = getattr(config, "agent_workspace", None)
    candidates = [
        configured_workspace,
        config.repo_root / "agentic",
        config.repo_root,
    ]
    source_skills = None
    for workspace in candidates:
        if workspace is None:
            continue
        candidate = (Path(workspace).resolve() / ".agents" / "skills").resolve()
        # The run directory is child-writable. Never treat skills staged there
        # by an earlier turn as the trusted source for the next turn.
        if candidate in target_skill_roots or candidate.is_relative_to(
            resolved_run_dir
        ):
            continue
        if candidate.is_dir():
            source_skills = candidate
            break
    # A previous child turn can modify its writable run directory. Replace
    # symlinks before copying so a resume cannot redirect trusted skills to an
    # arbitrary host path.
    for target_root, target_skills in zip(
        [target_agents, target_claude], target_skill_roots, strict=True
    ):
        if target_root.is_symlink() or target_root.is_file():
            target_root.unlink()
        target_root.mkdir(parents=True, exist_ok=True)
        if target_skills.is_symlink() or target_skills.is_file():
            target_skills.unlink()
        elif target_skills.exists():
            shutil.rmtree(target_skills)
    if source_skills is None:
        return
    for target_skills in target_skill_roots:
        shutil.copytree(source_skills, target_skills)


def _lexical_absolute_path(path: Path) -> Path:
    """Return an absolute path without following any symlink components."""

    return Path(os.path.abspath(path.expanduser()))


def _reject_unsafe_run_links(
    run_dir: Path,
    *,
    allow_missing: bool = False,
) -> None:
    """Reject child-controlled links before an unsandboxed parent writes."""

    lexical_run_dir = _lexical_absolute_path(run_dir)
    try:
        resolved_run_dir = run_dir.resolve()
    except OSError as exc:
        raise UnsafeRunArtifactError(
            f"Unable to inspect the run directory safely: {run_dir}"
        ) from exc
    if lexical_run_dir != resolved_run_dir:
        raise UnsafeRunArtifactError(
            f"Run directory path must resolve without traversing symlinks: {run_dir}"
        )
    try:
        run_metadata = resolved_run_dir.lstat()
    except FileNotFoundError:
        if allow_missing:
            return
        raise UnsafeRunArtifactError(
            f"Unable to inspect the run directory safely: {run_dir}"
        ) from None
    except OSError as exc:
        raise UnsafeRunArtifactError(
            f"Unable to inspect the run directory safely: {run_dir}"
        ) from exc
    if not stat.S_ISDIR(run_metadata.st_mode):
        raise UnsafeRunArtifactError(f"Run directory is not a directory: {run_dir}")

    def reject_walk_error(error: OSError) -> NoReturn:
        raise UnsafeRunArtifactError(
            f"Unable to inspect the run directory safely: {error.filename or run_dir}"
        ) from error

    try:
        for root, directories, files in os.walk(
            resolved_run_dir,
            followlinks=False,
            onerror=reject_walk_error,
        ):
            for name in [*directories, *files]:
                candidate = Path(root) / name
                try:
                    metadata = candidate.lstat()
                except OSError as exc:
                    raise UnsafeRunArtifactError(
                        f"Unable to inspect run artifact safely: {candidate}"
                    ) from exc
                if stat.S_ISLNK(metadata.st_mode):
                    raise UnsafeRunArtifactError(
                        "Child-created symlinks are not allowed in the run directory: "
                        f"{candidate}"
                    )
                if stat.S_ISDIR(metadata.st_mode):
                    continue
                if not stat.S_ISREG(metadata.st_mode):
                    raise UnsafeRunArtifactError(
                        "Child-created special files are not allowed in the run "
                        f"directory: {candidate}"
                    )
                if metadata.st_nlink != 1:
                    raise UnsafeRunArtifactError(
                        "Child-created hard links are not allowed in the run directory: "
                        f"{candidate}"
                    )
    except UnsafeRunArtifactError:
        raise
    except OSError as exc:
        raise UnsafeRunArtifactError(
            f"Unable to inspect the run directory safely: {run_dir}"
        ) from exc


class CodexThreadSession:
    """Long-lived Codex SDK thread used across initial and refinement turns."""

    def __init__(
        self,
        *,
        config: MaterialAssignConfig,
        run_dir: Path,
        managed_workbench: ManagedWorkbench | None = None,
    ) -> None:
        self.config = config
        self.run_dir = run_dir
        self.managed_workbench = managed_workbench
        self.bridge_path = Path(__file__).with_name("codex_sdk_bridge.mjs")
        self.session_request_path = run_dir / "raw" / "codex_thread_session.json"
        self.session_log_path = run_dir / "raw" / "codex_thread_session.log"
        self.turn_index = 0
        self.log_stream = None
        self.process: subprocess.Popen[str] | None = None
        self.output_reader: threading.Thread | None = None
        self.output_queue: queue.Queue[str] = queue.Queue()
        self.control_messages: list[dict[str, Any]] = []
        self._closed = False
        _reject_unsafe_run_links(run_dir)
        _stage_agent_skills(config, run_dir)
        session_request = _build_codex_session_request(config=config, run_dir=run_dir)
        _write_private_json(self.session_request_path, session_request)
        self.command = [
            "node",
            str(self.bridge_path),
            "--server",
            str(self.session_request_path),
        ]
        try:
            self.log_stream = self.session_log_path.open("w", encoding="utf-8")
            self.log_stream.write("$ " + " ".join(self.command) + "\n")
            self.log_stream.write(f"session_request: {self.session_request_path}\n")
            self.log_stream.flush()
            self.process = subprocess.Popen(
                _descendant_reaper_command(self.command),
                cwd=_agent_working_directory(config, self.run_dir),
                env=_codex_bridge_env(config),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
                bufsize=1,
                start_new_session=hasattr(os, "setsid"),
            )
            self.output_reader = threading.Thread(
                target=_read_subprocess_output,
                args=(self.process, self.output_queue),
                daemon=True,
            )
            self.output_reader.start()
            self._wait_until_ready()
        except BaseException:
            self.close()
            raise

    def run_turn(
        self,
        *,
        prompt: str,
        run_dir: Path,
        child_output_path: Path,
        child_final_path: Path,
        managed_workbench: ManagedWorkbench | None = None,
        prompt_image_inputs: list[dict[str, str]] | None = None,
        output_schema: dict[str, object] | None = None,
        bridge_artifact_prefix: str | None = None,
    ) -> int:
        self.turn_index += 1
        artifact_prefix = _bridge_artifact_prefix(bridge_artifact_prefix, "codex")
        sdk_request_path = run_dir / "raw" / f"{artifact_prefix}_request.json"
        sdk_request = _build_codex_sdk_request(
            config=self.config,
            prompt=prompt,
            run_dir=run_dir,
            child_final_path=child_final_path,
            prompt_image_inputs=prompt_image_inputs,
            output_schema=output_schema,
            bridge_artifact_prefix=artifact_prefix,
        )
        _write_private_json(sdk_request_path, sdk_request)

        request_id = f"{artifact_prefix}-{self.turn_index}"
        with child_output_path.open("w", encoding="utf-8") as log_stream:
            log_stream.write("$ " + " ".join(self.command) + "  # persistent\n")
            log_stream.write(f"request: {sdk_request_path}\n")
            log_stream.write(f"codex_thread_session: {self.session_request_path}\n")
            log_stream.flush()
            self._send_turn(request_id=request_id, request_path=sdk_request_path)
            return self._wait_for_turn(
                request_id=request_id,
                request_path=sdk_request_path,
                child_output_path=child_output_path,
                run_dir=run_dir,
                log_stream=log_stream,
                managed_workbench=managed_workbench or self.managed_workbench,
            )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        process = self.process
        if process is not None and process.poll() is None and process.stdin is not None:
            try:
                process.stdin.write(json.dumps({"type": "shutdown"}) + "\n")
                process.stdin.flush()
            except (BrokenPipeError, OSError):
                pass
        if process is not None and process.poll() is None:
            _terminate_subprocess(process)
        cleanup_error: BaseException | None = None
        try:
            if self.output_reader is not None:
                self.output_reader.join(timeout=2)
                self._drain_output()
        except BaseException as exc:  # noqa: BLE001 - validate reaper below
            cleanup_error = exc
        try:
            if self.log_stream is not None:
                self.log_stream.close()
        except BaseException as exc:  # noqa: BLE001 - validate reaper below
            cleanup_error = cleanup_error or exc
        if process is not None:
            try:
                _validated_descendant_supervisor_returncode(
                    process=process,
                    timeout_label="codex persistent child session",
                )
            except RuntimeError as exc:
                raise UnsafeRunArtifactError(
                    "Codex persistent child cleanup could not prove that all "
                    "descendants were reaped; refusing artifact processing: "
                    f"{exc}"
                ) from exc
        if cleanup_error is not None:
            raise cleanup_error

    def _wait_until_ready(self) -> None:
        start_time = time.monotonic()
        while True:
            self._drain_output()
            for index, message in enumerate(self.control_messages):
                if message.get("type") == "ready":
                    del self.control_messages[index]
                    return
                if message.get("type") == "error":
                    raise RuntimeError(str(message.get("error") or message))
            if self.process is None or self.process.poll() is not None:
                raise RuntimeError(
                    f"Codex thread bridge exited before ready with return code "
                    f"{self.process.returncode if self.process is not None else None}."
                )
            if time.monotonic() - start_time > 60:
                assert self.process is not None
                _terminate_subprocess(self.process)
                raise TimeoutError("Codex thread bridge did not become ready.")
            time.sleep(0.05)

    def _send_turn(self, *, request_id: str, request_path: Path) -> None:
        if self.process is None:
            raise RuntimeError("Codex thread bridge is not running.")
        if self.process.stdin is None or self.process.poll() is not None:
            raise RuntimeError("Codex thread bridge is not running.")
        message = {
            "type": "turn",
            "request_id": request_id,
            "request_path": str(request_path),
        }
        self.process.stdin.write(json.dumps(message) + "\n")
        self.process.stdin.flush()

    def _wait_for_turn(
        self,
        *,
        request_id: str,
        request_path: Path,
        child_output_path: Path,
        run_dir: Path,
        log_stream: Any,
        managed_workbench: ManagedWorkbench | None,
    ) -> int:
        timeout_seconds = self.config.child_timeout_seconds
        timeout_label = "codex persistent child turn"
        if self.process is None:
            raise RuntimeError("Codex thread bridge is not running.")
        process = self.process
        restore_signal_handlers = _install_child_signal_handlers(
            process=process,
            log_stream=log_stream,
            timeout_label=timeout_label,
        )
        workbench_watchdog = _make_workbench_watchdog(
            self.config.workbench_url,
            managed_workbench,
            output_root=run_dir,
        )
        start_time = time.monotonic()
        last_heartbeat = start_time
        last_watchdog_check = start_time
        workbench_failure_count = 0
        try:
            while True:
                self._drain_output(turn_log_stream=log_stream)
                response = self._pop_turn_response(
                    request_id=request_id,
                    request_path=request_path,
                )
                if response is not None:
                    if response.get("error"):
                        _write_runner_log_line(log_stream, str(response["error"]))
                    return int(response.get("returncode") or 0)

                if process.poll() is not None:
                    raise RuntimeError(
                        f"Codex thread bridge exited before turn completed: "
                        f"{process.returncode}"
                    )

                now = time.monotonic()
                elapsed = now - start_time
                if timeout_seconds > 0 and elapsed >= timeout_seconds:
                    _terminate_subprocess(process)
                    message = f"{timeout_label} exceeded {timeout_seconds:.1f} seconds"
                    _write_runner_log_line(log_stream, f"TimeoutError: {message}")
                    raise TimeoutError(message)

                if (now - last_heartbeat) >= CHILD_HEARTBEAT_INTERVAL_SECONDS:
                    last_heartbeat = now
                    _write_runner_log_line(
                        log_stream,
                        _child_heartbeat_message(
                            timeout_label=timeout_label,
                            elapsed_seconds=elapsed,
                            timeout_seconds=timeout_seconds,
                            progress_reporter=_run_progress_summary_for_log,
                            run_dir=child_output_path.parent,
                        ),
                        _console_stream(self.config),
                    )

                if (now - last_watchdog_check) >= WORKBENCH_WATCHDOG_INTERVAL_SECONDS:
                    last_watchdog_check = now
                    failure = workbench_watchdog()
                    if failure is None:
                        workbench_failure_count = 0
                    else:
                        workbench_failure_count += 1
                        _write_runner_log_line(
                            log_stream,
                            (
                                "Workbench watchdog warning "
                                f"{workbench_failure_count}/"
                                f"{WORKBENCH_HEALTH_FAILURE_LIMIT}: {failure.reason}"
                            ),
                            _console_stream(self.config),
                        )
                        if (
                            failure.fatal
                            or workbench_failure_count >= WORKBENCH_HEALTH_FAILURE_LIMIT
                        ):
                            _terminate_subprocess(process)
                            message = (
                                f"{timeout_label} stopped because {failure.reason}"
                            )
                            _write_runner_log_line(
                                log_stream,
                                f"RuntimeError: {message}",
                                _console_stream(self.config),
                            )
                            raise RuntimeError(message)

                time.sleep(0.05)
        except BaseException:
            cleanup_error: BaseException | None = None
            try:
                if process.poll() is None:
                    _terminate_subprocess(process)
            except BaseException as exc:  # noqa: BLE001 - validate reaper below
                cleanup_error = exc
            try:
                self._drain_output(turn_log_stream=log_stream)
            except BaseException as exc:  # noqa: BLE001 - preserve primary failure
                cleanup_error = cleanup_error or exc
            try:
                _validated_descendant_supervisor_returncode(
                    process=process,
                    timeout_label=timeout_label,
                )
            except RuntimeError as exc:
                raise UnsafeRunArtifactError(
                    f"{timeout_label} cleanup could not prove that all child "
                    "descendants were reaped; refusing artifact processing: "
                    f"{exc}"
                ) from exc
            if cleanup_error is not None:
                logger.debug(
                    "%s cleanup reported %s after the primary failure",
                    timeout_label,
                    cleanup_error,
                )
            raise
        finally:
            restore_signal_handlers()
            log_stream.flush()
            _chmod_private(child_output_path)

    def _pop_turn_response(
        self,
        *,
        request_id: str,
        request_path: Path,
    ) -> dict[str, Any] | None:
        for index, message in enumerate(self.control_messages):
            if message.get("type") != "turn_finished":
                continue
            if message.get("request_id") == request_id or message.get(
                "request_path"
            ) == str(request_path):
                return self.control_messages.pop(index)
        return None

    def _drain_output(self, turn_log_stream: Any | None = None) -> None:
        while True:
            try:
                line = self.output_queue.get_nowait()
            except queue.Empty:
                break
            self.log_stream.write(line)
            if turn_log_stream is not None:
                turn_log_stream.write(line)
            parsed = _parse_bridge_control_line(line)
            if parsed is not None:
                self.control_messages.append(parsed)
            else:
                logger.debug("%s", line.rstrip("\n"))
        self.log_stream.flush()
        if turn_log_stream is not None:
            turn_log_stream.flush()


def _parse_bridge_control_line(line: str) -> dict[str, Any] | None:
    try:
        value = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict):
        return None
    if value.get("type") in {
        "ready",
        "turn_finished",
        "error",
        "shutdown_ack",
    }:
        return value
    return None


def _run_child_agent(
    *,
    config: AgentRuntimeConfig,
    prompt: str,
    run_dir: Path,
    child_output_path: Path,
    child_final_path: Path,
    managed_workbench: ManagedWorkbench | None = None,
    prompt_image_inputs: list[dict[str, str]] | None = None,
    output_schema: dict[str, object] | None = None,
    bridge_artifact_prefix: str | None = None,
    codex_session: CodexThreadSession | None = None,
) -> int:
    _reject_unsafe_run_links(run_dir)
    _stage_agent_skills(config, run_dir)
    try:
        returncode: int
        if config.runner == RUNNER_CODEX:
            if codex_session is not None:
                returncode = codex_session.run_turn(
                    prompt=prompt,
                    run_dir=run_dir,
                    child_output_path=child_output_path,
                    child_final_path=child_final_path,
                    managed_workbench=managed_workbench,
                    prompt_image_inputs=prompt_image_inputs,
                    output_schema=output_schema,
                    bridge_artifact_prefix=bridge_artifact_prefix,
                )
            else:
                returncode = _run_child_agent_codex(
                    config=config,
                    prompt=prompt,
                    run_dir=run_dir,
                    child_output_path=child_output_path,
                    child_final_path=child_final_path,
                    managed_workbench=managed_workbench,
                    prompt_image_inputs=prompt_image_inputs,
                    output_schema=output_schema,
                    bridge_artifact_prefix=bridge_artifact_prefix,
                )
        elif config.runner == RUNNER_CLAUDE:
            if getattr(config, "claude_execution_mode", CLAUDE_EXECUTION_SDK) == (
                CLAUDE_EXECUTION_CLI
            ):
                returncode = _run_child_agent_claude_cli(
                    config=config,
                    prompt=prompt,
                    run_dir=run_dir,
                    child_output_path=child_output_path,
                    child_final_path=child_final_path,
                    managed_workbench=managed_workbench,
                    prompt_image_inputs=prompt_image_inputs,
                    bridge_artifact_prefix=bridge_artifact_prefix,
                )
            else:
                returncode = _run_child_agent_claude(
                    config=config,
                    prompt=prompt,
                    run_dir=run_dir,
                    child_output_path=child_output_path,
                    child_final_path=child_final_path,
                    managed_workbench=managed_workbench,
                    prompt_image_inputs=prompt_image_inputs,
                    bridge_artifact_prefix=bridge_artifact_prefix,
                )
        else:
            raise ValueError(f"Unsupported runner: {config.runner}")
        return returncode
    finally:
        # Provider failures return here only after their descendant supervisor
        # has terminated and reaped the child tree. Validate before the caller
        # performs any unsandboxed tracing, recovery, or artifact processing.
        # A safe tree preserves the provider's primary exception; an unsafe
        # tree deliberately fails closed with UnsafeRunArtifactError.
        _reject_unsafe_run_links(run_dir)


def _should_start_codex_thread_session(config: MaterialAssignConfig) -> bool:
    # Reusing a provider process would also reuse child-controlled workspace
    # state across trusted finalization boundaries. Retain the request field for
    # resume-schema compatibility, but always use fresh confined turns.
    del config
    return False


def _run_child_agent_codex(
    *,
    config: AgentRuntimeConfig,
    prompt: str,
    run_dir: Path,
    child_output_path: Path,
    child_final_path: Path,
    managed_workbench: ManagedWorkbench | None = None,
    prompt_image_inputs: list[dict[str, str]] | None = None,
    output_schema: dict[str, object] | None = None,
    bridge_artifact_prefix: str | None = None,
) -> int:
    bridge_path = Path(__file__).with_name("codex_sdk_bridge.mjs")
    artifact_prefix = _bridge_artifact_prefix(bridge_artifact_prefix, "codex")
    sdk_request_path = run_dir / "raw" / f"{artifact_prefix}_request.json"
    sdk_request = _build_codex_sdk_request(
        config=config,
        prompt=prompt,
        run_dir=run_dir,
        child_final_path=child_final_path,
        prompt_image_inputs=prompt_image_inputs,
        output_schema=output_schema,
        bridge_artifact_prefix=artifact_prefix,
    )
    _write_private_json(sdk_request_path, sdk_request)

    command = ["node", str(bridge_path), str(sdk_request_path)]
    with child_output_path.open("w", encoding="utf-8") as log_stream:
        log_stream.write("$ " + " ".join(command) + "\n")
        log_stream.write(f"request: {sdk_request_path}\n")
        log_stream.flush()
        return _run_subprocess_with_timeout(
            command=command,
            cwd=_agent_working_directory(config, run_dir),
            env=_codex_bridge_env(config),
            timeout_seconds=config.child_timeout_seconds,
            log_stream=log_stream,
            timeout_label="codex child turn",
            workbench_watchdog=_make_workbench_watchdog(
                config.workbench_url,
                managed_workbench,
                output_root=run_dir,
            ),
            progress_reporter=_run_progress_summary_for_log,
            run_dir=run_dir,
            terminal_success_detector=_terminal_success_detector_for_bridge(
                artifact_prefix
            ),
            console_stream=_console_stream(config),
        )


def _build_codex_sdk_request(
    *,
    config: AgentRuntimeConfig,
    prompt: str,
    run_dir: Path,
    child_final_path: Path,
    prompt_image_inputs: list[dict[str, str]] | None = None,
    output_schema: dict[str, object] | None = None,
    bridge_artifact_prefix: str | None = None,
) -> dict[str, object]:
    artifact_prefix = _bridge_artifact_prefix(bridge_artifact_prefix, "codex")
    sdk_items_path = run_dir / "raw" / f"{artifact_prefix}_items.json"
    sdk_result_path = run_dir / "raw" / f"{artifact_prefix}_result.json"
    child_cwd = _agent_working_directory(config, run_dir)
    credentials_store = _codex_cli_auth_credentials_store(child_cwd=child_cwd)
    return {
        "schema_version": "content-agents.codex-request.v1",
        "repo_root": str(child_cwd),
        "run_dir": str(run_dir),
        "prompt": prompt,
        "reference_images": [str(path) for path in config.reference_images],
        "reference_files": [str(path) for path in config.reference_files or []],
        "prompt_image_inputs": prompt_image_inputs or [],
        "output_schema": output_schema,
        "child_final_path": str(child_final_path),
        "items_path": str(sdk_items_path),
        "result_path": str(sdk_result_path),
        "model": config.model,
        "model_reasoning_effort": config.model_reasoning_effort,
        "codex_base_url": config.codex_base_url,
        "codex_sandbox_mode": config.codex_sandbox_mode,
        "codex_config": config.codex_config or {},
        **(
            {"cli_auth_credentials_store": credentials_store}
            if credentials_store is not None
            else {}
        ),
        "child_timeout_seconds": config.child_timeout_seconds,
    }


def _build_codex_session_request(
    *,
    config: AgentRuntimeConfig,
    run_dir: Path,
) -> dict[str, object]:
    child_cwd = _agent_working_directory(config, run_dir)
    credentials_store = _codex_cli_auth_credentials_store(child_cwd=child_cwd)
    return {
        "schema_version": "content-agents.codex-thread-session.v1",
        "repo_root": str(child_cwd),
        "run_dir": str(run_dir),
        "model": config.model,
        "model_reasoning_effort": config.model_reasoning_effort,
        "codex_base_url": config.codex_base_url,
        "codex_sandbox_mode": config.codex_sandbox_mode,
        "codex_config": config.codex_config or {},
        **(
            {"cli_auth_credentials_store": credentials_store}
            if credentials_store is not None
            else {}
        ),
        "child_timeout_seconds": config.child_timeout_seconds,
    }


def _effective_codex_home(*, child_cwd: Path | None = None) -> Path:
    configured_home = os.environ.get("CODEX_HOME")
    if not configured_home:
        return Path.home() / ".codex"
    codex_home = Path(configured_home).expanduser()
    if not codex_home.is_absolute():
        codex_home = (child_cwd or Path.cwd()) / codex_home
    return codex_home


def _codex_cli_auth_credentials_store(*, child_cwd: Path | None = None) -> str | None:
    config_path = _effective_codex_home(child_cwd=child_cwd) / "config.toml"
    try:
        with config_path.open("rb") as stream:
            config = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError, UnicodeDecodeError):
        return None
    credentials_store = config.get("cli_auth_credentials_store")
    if (
        not isinstance(credentials_store, str)
        or credentials_store not in SUPPORTED_CODEX_AUTH_CREDENTIALS_STORES
    ):
        return None
    return credentials_store


def _codex_bridge_env(config: AgentRuntimeConfig) -> dict[str, str]:
    env = os.environ.copy()
    # The Node bridge may invoke Python through shell tools; keep local packages
    # importable.
    env["PYTHONPATH"] = _prepend_pythonpath(
        env.get("PYTHONPATH"),
        *_agentic_workflow_pythonpath_roots(config.repo_root),
    )
    return env


def _run_child_agent_claude(
    *,
    config: AgentRuntimeConfig,
    prompt: str,
    run_dir: Path,
    child_output_path: Path,
    child_final_path: Path,
    managed_workbench: ManagedWorkbench | None = None,
    prompt_image_inputs: list[dict[str, str]] | None = None,
    bridge_artifact_prefix: str | None = None,
) -> int:
    bridge_path = Path(__file__).with_name("claude_bridge.mjs")
    artifact_prefix = _bridge_artifact_prefix(bridge_artifact_prefix, "claude")
    sdk_request_path = run_dir / "raw" / f"{artifact_prefix}_request.json"
    sdk_items_path = run_dir / "raw" / f"{artifact_prefix}_items.json"
    sdk_result_path = run_dir / "raw" / f"{artifact_prefix}_result.json"
    sdk_request = {
        "schema_version": "content-agents.claude-request.v1",
        "repo_root": str(_agent_working_directory(config, run_dir)),
        "run_dir": str(run_dir),
        "workbench_url": config.workbench_url,
        "prompt": prompt,
        "reference_images": [str(path) for path in config.reference_images],
        "reference_files": [str(path) for path in config.reference_files or []],
        "prompt_image_inputs": prompt_image_inputs or [],
        "child_final_path": str(child_final_path),
        "items_path": str(sdk_items_path),
        "result_path": str(sdk_result_path),
        "model": config.model,
        "model_reasoning_effort": config.model_reasoning_effort,
        "claude_config": config.claude_config or {},
        "claude_permission_mode": config.claude_permission_mode,
        "claude_max_turns": config.claude_max_turns,
        "child_timeout_seconds": config.child_timeout_seconds,
    }
    _write_private_json(sdk_request_path, sdk_request)

    env = os.environ.copy()
    # The Node bridge may invoke Python through shell tools; keep local packages
    # importable.
    env["PYTHONPATH"] = _prepend_pythonpath(
        env.get("PYTHONPATH"),
        *_agentic_workflow_pythonpath_roots(config.repo_root),
    )
    command = ["node", str(bridge_path), str(sdk_request_path)]
    with child_output_path.open("w", encoding="utf-8") as log_stream:
        log_stream.write("$ " + " ".join(command) + "\n")
        log_stream.write(f"request: {sdk_request_path}\n")
        log_stream.flush()
        return _run_subprocess_with_timeout(
            command=command,
            cwd=_agent_working_directory(config, run_dir),
            env=env,
            timeout_seconds=config.child_timeout_seconds,
            log_stream=log_stream,
            timeout_label="claude child turn",
            workbench_watchdog=_make_workbench_watchdog(
                config.workbench_url,
                managed_workbench,
                output_root=run_dir,
            ),
            progress_reporter=_run_progress_summary_for_log,
            run_dir=run_dir,
            terminal_success_detector=_terminal_success_detector_for_bridge(
                artifact_prefix
            ),
            console_stream=_console_stream(config),
        )


def _find_claude_cli_binary() -> str:
    override = os.environ.get(CLAUDE_CLI_BINARY_ENV)
    resolved = shutil.which(override) if override else shutil.which("claude")
    if not resolved:
        raise RuntimeError(
            "The 'claude' CLI was not found on PATH. Install Claude Code and run "
            "`claude login` once (so it has an OAuth session), or set "
            f"{CLAUDE_CLI_BINARY_ENV} to the executable path."
        )
    return resolved


def _map_claude_cli_effort(value: str | None) -> str | None:
    if not value:
        return None
    if value == "minimal":
        return "low"
    return value


def _claude_cli_add_dirs(*, run_dir: Path) -> list[str]:
    """Expose only the writable run directory as a Claude workspace root."""

    return [str(run_dir.resolve())]


def _stage_claude_cli_reference_images(
    *,
    run_dir: Path,
    artifact_prefix: str,
    reference_images: list[Path],
    prompt_image_inputs: list[dict[str, str]],
) -> tuple[list[Path], list[dict[str, str]]]:
    """Copy image inputs into the confined workspace for Claude's Read tool."""

    target_root = run_dir.resolve() / "raw" / f"{artifact_prefix}_reference_images"
    if target_root.is_symlink() or target_root.is_file():
        target_root.unlink()
    elif target_root.exists():
        shutil.rmtree(target_root)
    target_root.mkdir(parents=True, mode=0o700)
    _chmod_private(target_root)

    def stage(source: Path, *, kind: str, index: int) -> Path:
        resolved_source = source.expanduser().resolve()
        if not resolved_source.is_file():
            raise ValueError(f"Claude CLI reference image is not a file: {source}")
        suffix = resolved_source.suffix.lower()
        if not re.fullmatch(r"\.[a-z0-9]{1,10}", suffix):
            suffix = ""
        destination = target_root / f"{kind}-{index:03d}{suffix}"
        with (
            resolved_source.open("rb") as source_stream,
            destination.open("xb") as destination_stream,
        ):
            shutil.copyfileobj(source_stream, destination_stream)
        _chmod_private(destination)
        return destination

    staged_references = [
        stage(path, kind="reference", index=index)
        for index, path in enumerate(reference_images)
    ]
    staged_prompt_inputs: list[dict[str, str]] = []
    for index, item in enumerate(prompt_image_inputs):
        raw_path = item.get("path")
        if not raw_path:
            continue
        staged_item = dict(item)
        staged_item["path"] = str(stage(Path(raw_path), kind="prompt", index=index))
        staged_prompt_inputs.append(staged_item)
    return staged_references, staged_prompt_inputs


def _build_claude_cli_prompt(
    *,
    prompt: str,
    reference_images: list[Path],
    reference_files: list[Path],
    prompt_image_inputs: list[dict[str, str]],
) -> str:
    sections = [prompt]
    image_lines = [f"- {path}" for path in reference_images]
    for item in prompt_image_inputs:
        raw_path = item.get("path")
        if not raw_path:
            continue
        label = item.get("label") or "Prompt image"
        image_lines.append(f"- {raw_path} ({label})")
    if image_lines:
        sections.append(
            "Reference images (use the Read tool to inspect each staged image "
            "before reasoning about it):\n" + "\n".join(image_lines)
        )
    if reference_files:
        file_lines = [f"- {path}" for path in reference_files]
        sections.append(
            "Reference files (use sandboxed Bash to inspect each as needed):\n"
            + "\n".join(file_lines)
        )
    return "\n\n".join(sections)


def _filter_claude_cli_env(config_env: dict[str, object]) -> dict[str, str]:
    filtered: dict[str, str] = {}
    dropped: list[str] = []
    for key, value in config_env.items():
        if key in CLAUDE_CLI_DANGEROUS_ENV_KEYS:
            dropped.append(key)
            continue
        filtered[key] = str(value)
    if dropped:
        logger.warning(
            "Ignoring dangerous Claude config env key(s): %s.", ", ".join(dropped)
        )
    return filtered


def _build_claude_cli_command(
    *,
    claude_bin: str,
    config: AgentRuntimeConfig,
    prompt_image_inputs: list[dict[str, str]],
    run_dir: Path,
) -> list[str]:
    permission_mode = config.claude_permission_mode
    # acceptEdits and bypassPermissions auto-approve direct file mutations, so
    # merely keeping mutating tools out of --allowedTools is not a confinement
    # boundary in those modes. Restrict the actual tool surface to read-only
    # tools plus Bash; Bash remains useful for Workbench calls and artifact
    # creation, while the mandatory OS sandbox confines its writes and network.
    available_tools = (
        CLAUDE_CLI_SANDBOXED_TOOLS
        if permission_mode in CLAUDE_SANDBOXED_PERMISSION_MODES
        else CLAUDE_CLI_AVAILABLE_TOOLS
    )
    parsed_workbench = urlparse(config.workbench_url)
    allowed_domains = [parsed_workbench.hostname] if parsed_workbench.hostname else []
    sandbox_settings = {
        "sandbox": {
            "enabled": True,
            "failIfUnavailable": True,
            "autoAllowBashIfSandboxed": True,
            "allowUnsandboxedCommands": False,
            "network": {"allowedDomains": allowed_domains},
        }
    }
    command = [
        claude_bin,
        "--print",
        "--output-format",
        "json",
        "--input-format",
        "text",
        "--no-session-persistence",
        "--permission-mode",
        permission_mode,
        "--append-system-prompt",
        CLAUDE_CLI_SYSTEM_PROMPT_APPEND,
        "--settings",
        json.dumps(sandbox_settings, separators=(",", ":")),
        "--setting-sources",
        "",
        # `--allowedTools` only pre-approves these tools so they skip an
        # interactive permission prompt under the default permission mode;
        # it does not restrict which built-in tools are available to call.
        # `--tools` is what actually shrinks the available toolset, which is
        # what keeps Monitor (and anything else not listed here) out of
        # reach regardless of permission mode -- including
        # `bypassPermissions`, under which an unlisted tool would otherwise
        # be auto-approved and callable, reproducing the exact async-stall
        # this execution mode's system prompt is meant to prevent.
        "--allowedTools",
        ",".join(CLAUDE_CLI_ALLOWED_TOOLS),
        "--tools",
        ",".join(available_tools),
    ]
    for directory in _claude_cli_add_dirs(run_dir=run_dir):
        command.extend(["--add-dir", directory])
    if permission_mode == "bypassPermissions":
        command.append("--dangerously-skip-permissions")
    if config.model:
        command.extend(["--model", config.model])
    effort = _map_claude_cli_effort(config.model_reasoning_effort)
    if effort:
        command.extend(["--effort", effort])
    claude_config = config.claude_config or {}
    max_budget_usd = claude_config.get("maxBudgetUsd")
    if isinstance(max_budget_usd, int | float):
        command.extend(["--max-budget-usd", str(max_budget_usd)])
    # The prompt is sent via stdin (see `stdin_input` at the call site)
    # instead of as a positional argument: an unbounded
    # `--additional-instructions-file` can make the prompt exceed the
    # ~128 KiB Linux argv limit, failing the spawn with E2BIG.
    return command


def _extract_last_json_object(text: str) -> dict[str, Any] | None:
    """Recover the trailing JSON object from CLI stdout that may also
    contain non-JSON diagnostic lines merged in from stderr.

    Advances past each successfully parsed span (rather than by one
    character) so nested objects, such as a `usage` sub-dict, are not
    mistaken for the outer result envelope.
    """
    decoder = json.JSONDecoder()
    last_match: dict[str, Any] | None = None
    index = text.find("{")
    while index != -1:
        try:
            candidate, end = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            index = text.find("{", index + 1)
            continue
        if isinstance(candidate, dict):
            last_match = candidate
        index = text.find("{", end)
    return last_match


def _finalize_claude_cli_output(
    *,
    child_output_path: Path,
    child_final_path: Path,
    items_path: Path,
    result_path: Path,
) -> None:
    raw_text = child_output_path.read_text(encoding="utf-8", errors="replace")
    parsed = _extract_last_json_object(raw_text)
    if parsed is None:
        _write_private_json(items_path, [])
        _write_private_json(result_path, {})
        return
    final_text = parsed.get("result")
    if not isinstance(final_text, str) or not final_text:
        final_text = json.dumps(parsed, indent=2)
    child_final_path.write_text(final_text, encoding="utf-8")
    _chmod_private(child_final_path)
    _write_private_json(items_path, [parsed])
    _write_private_json(result_path, parsed)


def _run_child_agent_claude_cli(
    *,
    config: AgentRuntimeConfig,
    prompt: str,
    run_dir: Path,
    child_output_path: Path,
    child_final_path: Path,
    managed_workbench: ManagedWorkbench | None = None,
    prompt_image_inputs: list[dict[str, str]] | None = None,
    bridge_artifact_prefix: str | None = None,
) -> int:
    claude_bin = _find_claude_cli_binary()
    artifact_prefix = _bridge_artifact_prefix(bridge_artifact_prefix, "claude")
    items_path = run_dir / "raw" / f"{artifact_prefix}_items.json"
    result_path = run_dir / "raw" / f"{artifact_prefix}_result.json"
    image_inputs = prompt_image_inputs or []
    staged_reference_images, staged_image_inputs = _stage_claude_cli_reference_images(
        run_dir=run_dir,
        artifact_prefix=artifact_prefix,
        reference_images=config.reference_images,
        prompt_image_inputs=image_inputs,
    )

    full_prompt = _build_claude_cli_prompt(
        prompt=prompt,
        reference_images=staged_reference_images,
        reference_files=config.reference_files or [],
        prompt_image_inputs=staged_image_inputs,
    )
    command = _build_claude_cli_command(
        claude_bin=claude_bin,
        config=config,
        prompt_image_inputs=staged_image_inputs,
        run_dir=run_dir,
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = _prepend_pythonpath(
        env.get("PYTHONPATH"),
        *_agentic_workflow_pythonpath_roots(config.repo_root),
    )
    config_env = (config.claude_config or {}).get("env")
    if isinstance(config_env, dict):
        env.update(_filter_claude_cli_env(config_env))

    with child_output_path.open("w", encoding="utf-8") as log_stream:
        log_stream.write(
            "$ "
            + " ".join(shlex.quote(part) for part in command)
            + " <prompt sent via stdin, omitted for brevity>\n"
        )
        log_stream.flush()
        returncode = _run_subprocess_with_timeout(
            command=command,
            cwd=_agent_working_directory(config, run_dir),
            env=env,
            timeout_seconds=config.child_timeout_seconds,
            log_stream=log_stream,
            timeout_label="claude CLI child turn",
            workbench_watchdog=_make_workbench_watchdog(
                config.workbench_url,
                managed_workbench,
                output_root=run_dir,
            ),
            progress_reporter=_run_progress_summary_for_log,
            run_dir=run_dir,
            terminal_success_detector=_terminal_success_detector_for_bridge(
                artifact_prefix
            ),
            console_stream=_console_stream(config),
            stdin_input=full_prompt,
        )

    _reject_unsafe_run_links(run_dir)
    _finalize_claude_cli_output(
        child_output_path=child_output_path,
        child_final_path=child_final_path,
        items_path=items_path,
        result_path=result_path,
    )
    return returncode


def _run_subprocess_with_timeout(
    *,
    command: list[str],
    cwd: Path,
    env: dict[str, str],
    timeout_seconds: float,
    log_stream: Any,
    timeout_label: str,
    workbench_watchdog: Callable[[], WatchdogFailure | None] | None = None,
    heartbeat_interval_seconds: float = CHILD_HEARTBEAT_INTERVAL_SECONDS,
    watchdog_interval_seconds: float = WORKBENCH_WATCHDOG_INTERVAL_SECONDS,
    workbench_health_failure_limit: int = WORKBENCH_HEALTH_FAILURE_LIMIT,
    progress_reporter: Callable[[Path], str | None] | None = None,
    run_dir: Path | None = None,
    terminal_success_detector: Callable[[Path], str | None] | None = None,
    terminal_success_grace_seconds: float = TERMINAL_SUCCESS_GRACE_SECONDS,
    console_stream: Any | None = None,
    stdin_input: str | None = None,
) -> int:
    supervised_command = _descendant_reaper_command(command)
    process = subprocess.Popen(
        supervised_command,
        cwd=cwd,
        env=env,
        stdin=subprocess.PIPE if stdin_input is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        bufsize=1,
        start_new_session=hasattr(os, "setsid"),
    )
    output_queue: queue.Queue[str] = queue.Queue()
    output_reader = threading.Thread(
        target=_read_subprocess_output,
        args=(process, output_queue),
        daemon=True,
    )
    output_reader.start()
    if stdin_input is not None and process.stdin is not None:
        # Pass large prompts via stdin instead of argv: Linux caps a single
        # exec() argument list around 128 KiB, and an unbounded
        # `--additional-instructions-file` can exceed that, failing the
        # spawn with E2BIG before the child ever starts. Write from a
        # dedicated thread, started only after the stdout reader is already
        # running: writing a large prompt synchronously here, before stdout
        # is being drained, can deadlock if the child starts emitting output
        # before it finishes reading stdin (its stdout pipe fills while we're
        # still blocked writing stdin, and neither side can make progress).
        threading.Thread(
            target=_write_and_close_stdin,
            args=(process.stdin, stdin_input),
            daemon=True,
        ).start()
    restore_signal_handlers = _install_child_signal_handlers(
        process=process,
        log_stream=log_stream,
        timeout_label=timeout_label,
    )
    start_time = time.monotonic()
    last_heartbeat = start_time
    last_watchdog_check = start_time
    terminal_success_detected_at: float | None = None
    workbench_failure_count = 0
    try:
        while True:
            _drain_subprocess_output(output_queue, log_stream, console_stream)
            if process.poll() is not None:
                break

            now = time.monotonic()
            elapsed = now - start_time
            if terminal_success_detector is not None and run_dir is not None:
                terminal_summary = terminal_success_detector(run_dir)
                if terminal_summary:
                    if terminal_success_detected_at is None:
                        terminal_success_detected_at = now
                        _write_runner_log_line(
                            log_stream,
                            (
                                f"{timeout_label} terminal run state detected: "
                                f"{terminal_summary}; waiting "
                                f"{terminal_success_grace_seconds:.1f}s for child "
                                "to exit"
                            ),
                            console_stream,
                        )
                    elif (
                        terminal_success_grace_seconds <= 0
                        or (now - terminal_success_detected_at)
                        >= terminal_success_grace_seconds
                    ):
                        _write_runner_log_line(
                            log_stream,
                            (
                                f"{timeout_label} still running after terminal "
                                f"run state: {terminal_summary}; terminating child "
                                "and treating run as successful"
                            ),
                            console_stream,
                        )
                        _terminate_subprocess(process)
                        output_reader.join(timeout=2)
                        _drain_subprocess_output(
                            output_queue,
                            log_stream,
                            console_stream,
                        )
                        _validated_descendant_supervisor_returncode(
                            process=process,
                            timeout_label=timeout_label,
                        )
                        return 0
                else:
                    terminal_success_detected_at = None
            if timeout_seconds > 0 and elapsed >= timeout_seconds:
                _terminate_subprocess(process)
                message = f"{timeout_label} exceeded {timeout_seconds:.1f} seconds"
                _write_runner_log_line(
                    log_stream,
                    f"TimeoutError: {message}",
                    console_stream,
                )
                raise TimeoutError(message)

            if (
                heartbeat_interval_seconds > 0
                and (now - last_heartbeat) >= heartbeat_interval_seconds
            ):
                last_heartbeat = now
                _write_runner_log_line(
                    log_stream,
                    _child_heartbeat_message(
                        timeout_label=timeout_label,
                        elapsed_seconds=elapsed,
                        timeout_seconds=timeout_seconds,
                        progress_reporter=progress_reporter,
                        run_dir=run_dir,
                    ),
                    console_stream,
                )

            if (
                workbench_watchdog is not None
                and watchdog_interval_seconds > 0
                and (now - last_watchdog_check) >= watchdog_interval_seconds
            ):
                last_watchdog_check = now
                failure = workbench_watchdog()
                if failure is None:
                    workbench_failure_count = 0
                else:
                    workbench_failure_count += 1
                    _write_runner_log_line(
                        log_stream,
                        (
                            "Workbench watchdog warning "
                            f"{workbench_failure_count}/"
                            f"{workbench_health_failure_limit}: {failure.reason}"
                        ),
                        console_stream,
                    )
                    if (
                        failure.fatal
                        or workbench_failure_count >= workbench_health_failure_limit
                    ):
                        _terminate_subprocess(process)
                        message = f"{timeout_label} stopped because {failure.reason}"
                        _write_runner_log_line(
                            log_stream,
                            f"RuntimeError: {message}",
                            console_stream,
                        )
                        raise RuntimeError(message)

            wait_timeout = 0.5
            if timeout_seconds > 0:
                wait_timeout = min(wait_timeout, max(0.0, timeout_seconds - elapsed))
            if heartbeat_interval_seconds > 0:
                wait_timeout = min(
                    wait_timeout,
                    max(0.0, heartbeat_interval_seconds - (now - last_heartbeat)),
                )
            if workbench_watchdog is not None and watchdog_interval_seconds > 0:
                wait_timeout = min(
                    wait_timeout,
                    max(
                        0.0,
                        watchdog_interval_seconds - (now - last_watchdog_check),
                    ),
                )
            try:
                process.wait(timeout=max(0.05, wait_timeout))
            except subprocess.TimeoutExpired:
                pass

        output_reader.join(timeout=2)
        _drain_subprocess_output(output_queue, log_stream, console_stream)
        return _validated_descendant_supervisor_returncode(
            process=process,
            timeout_label=timeout_label,
        )
    except BaseException:
        cleanup_error: BaseException | None = None
        try:
            if process.poll() is None:
                _terminate_subprocess(process)
        except BaseException as exc:  # noqa: BLE001 - validate reaper below
            cleanup_error = exc
        try:
            output_reader.join(timeout=2)
            _drain_subprocess_output(output_queue, log_stream, console_stream)
        except BaseException as exc:  # noqa: BLE001 - preserve primary failure
            cleanup_error = cleanup_error or exc
        try:
            if process.pid is not None and process.poll() is None:
                _terminate_subprocess_group(process.pid)
        except BaseException as exc:  # noqa: BLE001 - validate reaper below
            cleanup_error = cleanup_error or exc
        try:
            _validated_descendant_supervisor_returncode(
                process=process,
                timeout_label=timeout_label,
            )
        except RuntimeError as exc:
            raise UnsafeRunArtifactError(
                f"{timeout_label} cleanup could not prove that all child "
                "descendants were reaped; refusing artifact processing: "
                f"{exc}"
            ) from exc
        if cleanup_error is not None:
            logger.debug(
                "%s cleanup reported %s after the primary failure",
                timeout_label,
                cleanup_error,
            )
        raise
    finally:
        restore_signal_handlers()


def _validated_descendant_supervisor_returncode(
    *,
    process: subprocess.Popen[str],
    timeout_label: str,
) -> int:
    if process.returncode is None:
        raise RuntimeError(f"{timeout_label} exited without a return code")
    returncode = int(process.returncode)
    if sys.platform == "linux":
        if returncode < 0:
            raise RuntimeError(
                f"{timeout_label} descendant supervisor died from signal "
                f"{-returncode}; refusing artifact post-processing"
            )
        if returncode == 125:
            raise RuntimeError(
                f"{timeout_label} descendant supervisor failed; refusing "
                "artifact post-processing"
            )
    return returncode


def _descendant_reaper_command(command: list[str]) -> list[str]:
    if sys.platform != "linux":
        raise RuntimeError("confined child turns require Linux descendant supervision")
    reaper_path = Path(__file__).with_name("descendant_reaper.py").resolve()
    return [sys.executable, "-I", "-S", str(reaper_path), "--", *command]


def _read_subprocess_output(
    process: subprocess.Popen[str],
    output_queue: queue.Queue[str],
) -> None:
    if process.stdout is None:
        return
    for line in process.stdout:
        output_queue.put(line)


def _write_and_close_stdin(stdin: Any, data: str) -> None:
    try:
        stdin.write(data)
    except (BrokenPipeError, OSError):
        # The child may exit, or stop reading stdin, before consuming the
        # whole prompt; that is the child's business, not a wrapper failure.
        pass
    finally:
        try:
            stdin.close()
        except OSError:
            pass


def _drain_subprocess_output(
    output_queue: queue.Queue[str],
    log_stream: Any,
    console_stream: Any | None = None,
) -> None:
    stream = console_stream or sys.stdout
    while True:
        try:
            line = output_queue.get_nowait()
        except queue.Empty:
            break
        print(line, end="", file=stream)
        log_stream.write(line)
    log_stream.flush()


def _write_runner_log_line(
    log_stream: Any,
    message: str,
    console_stream: Any | None = None,
) -> None:
    line = f"content-workflow-cli: {message}\n"
    print(line, end="", file=console_stream or sys.stdout)
    log_stream.write(line)
    log_stream.flush()


def _print_run_start(
    *,
    run_dir: Path,
    request_path: Path,
    child_output_path: Path,
    stream: Any | None = None,
) -> None:
    output_stream = stream or sys.stdout
    print(
        f"content-workflow-cli: run id: {run_dir.name}", file=output_stream, flush=True
    )
    print(
        f"content-workflow-cli: run directory: {run_dir}",
        file=output_stream,
        flush=True,
    )
    print(
        f"content-workflow-cli: request: {request_path}", file=output_stream, flush=True
    )
    print(
        f"content-workflow-cli: child output: {child_output_path}",
        file=output_stream,
        flush=True,
    )


def _console_stream(config: AgentRuntimeConfig) -> Any:
    return sys.stderr if getattr(config, "log_to_stderr", False) else sys.stdout


def _child_heartbeat_message(
    *,
    timeout_label: str,
    elapsed_seconds: float,
    timeout_seconds: float,
    progress_reporter: Callable[[Path], str | None] | None = None,
    run_dir: Path | None = None,
) -> str:
    timeout_text = "disabled" if timeout_seconds <= 0 else f"{timeout_seconds:.0f}s"
    message = (
        f"{timeout_label} still running "
        f"(elapsed={elapsed_seconds:.0f}s, timeout={timeout_text})"
    )
    if progress_reporter is None or run_dir is None:
        return message
    try:
        progress = progress_reporter(run_dir)
    except Exception as exc:  # noqa: BLE001 - progress logging must be best effort
        logger.debug("Failed to build child progress heartbeat: %s", exc)
        return message
    if not progress:
        return message
    return f"{message}; progress: {progress}"


def _run_progress_summary_for_log(run_dir: Path) -> str | None:
    """Return a compact progress summary from durable run artifacts, if present."""

    parts: list[str] = []
    request = _load_progress_json_object(run_dir, run_dir / "request.json")
    if request is not None:
        workflow = request.get("workflow")
        if isinstance(workflow, str) and workflow:
            parts.append(f"workflow={workflow}")

    scene_state = _load_progress_json_object(
        run_dir,
        run_dir / "large_scene_run.json",
    )
    if scene_state is not None:
        current_phase = scene_state.get("current_phase")
        if isinstance(current_phase, str) and current_phase:
            parts.append(f"scene phase={current_phase}")
        phases = scene_state.get("phases")
        if isinstance(phases, dict):
            phase_bits: list[str] = []
            for name in ("decomposition", "asset_task_processing", "collection"):
                raw_phase = phases.get(name)
                if not isinstance(raw_phase, dict):
                    continue
                status = raw_phase.get("status")
                if isinstance(status, str) and status:
                    phase_bits.append(f"{name}={status}")
            if phase_bits:
                parts.append("phases " + ", ".join(phase_bits))

    asset_task_dir = _active_asset_task_dir(run_dir)
    asset_state = (
        _load_progress_json_object(
            run_dir,
            asset_task_dir / "asset_task_run_state.json",
        )
        if asset_task_dir is not None
        else None
    )
    if asset_state is not None:
        work_items = asset_state.get("work_items")
        if isinstance(work_items, list):
            status_counts: dict[str, int] = {}
            for item in work_items:
                if not isinstance(item, dict):
                    continue
                status = item.get("status")
                if not isinstance(status, str) or not status:
                    status = "unknown"
                status_counts[status] = status_counts.get(status, 0) + 1
            total = len(work_items)
            results_count = _asset_task_result_count(asset_task_dir)
            status_text = ", ".join(
                f"{status}={count}" for status, count in sorted(status_counts.items())
            )
            phase_dir = asset_task_dir.relative_to(run_dir).as_posix()
            workflow2 = f"workflow2 dir={phase_dir}, results={results_count}/{total}"
            if status_text:
                workflow2 += f", work_items {status_text}"
            parts.append(workflow2)

    material_summary = _material_progress_summary_for_log(run_dir)
    if material_summary:
        parts.append(material_summary)

    physics_summary = _physics_progress_summary_for_log(run_dir)
    if physics_summary:
        parts.append(physics_summary)

    trace_summary = _trace_progress_summary_for_log(run_dir)
    if trace_summary:
        parts.append(trace_summary)

    latest = _latest_progress_artifact(run_dir)
    if latest is not None:
        rel_path, age_seconds = latest
        parts.append(
            f"latest={rel_path} updated {_format_progress_age(age_seconds)} ago"
        )

    return "; ".join(parts) if parts else None


def _terminal_success_summary_for_log(run_dir: Path) -> str | None:
    """Return a terminal success summary for wrapper-supervised runs."""

    scene_state = _load_progress_json_object(
        run_dir,
        run_dir / "large_scene_run.json",
    )
    if scene_state is None:
        return None
    if scene_state.get("failed_at") is not None:
        return None
    if scene_state.get("current_phase") is not None:
        return None
    phases = scene_state.get("phases")
    if not isinstance(phases, dict):
        return None
    required = ("decomposition", "asset_task_processing", "collection")
    statuses: list[str] = []
    for name in required:
        raw_phase = phases.get(name)
        if not isinstance(raw_phase, dict):
            return None
        status = raw_phase.get("status")
        if status != "completed":
            return None
        statuses.append(f"{name}=completed")
    return "large-scene run completed; " + ", ".join(statuses)


def _terminal_success_detector_for_bridge(
    bridge_artifact_prefix: str,
) -> Callable[[Path], str | None] | None:
    """Enable scene terminal detection only for parent-scoped scene launches."""

    if bridge_artifact_prefix in SCENE_TERMINAL_BRIDGE_PREFIXES:
        return _terminal_success_summary_for_log
    return None


def _live_progress_artifact_metadata(
    run_dir: Path,
    path: Path,
    *,
    directory: bool = False,
) -> os.stat_result | None:
    """Stat a live progress path through no-follow directory descriptors."""

    lexical_run_dir = _lexical_absolute_path(run_dir)
    lexical_path = _lexical_absolute_path(path)
    try:
        relative_path = lexical_path.relative_to(lexical_run_dir)
    except ValueError:
        return None
    if not relative_path.parts:
        return None

    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    directory_fds: list[int] = []
    try:
        directory_fds.append(os.open(lexical_run_dir, directory_flags))
        current_fd = directory_fds[-1]
        for component in relative_path.parts[:-1]:
            next_fd = os.open(component, directory_flags, dir_fd=current_fd)
            directory_fds.append(next_fd)
            current_fd = next_fd
        metadata = os.stat(
            relative_path.name,
            dir_fd=current_fd,
            follow_symlinks=False,
        )
        if directory:
            return metadata if stat.S_ISDIR(metadata.st_mode) else None
        if stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1:
            return metadata
        return None
    except OSError:
        return None
    finally:
        for directory_fd in reversed(directory_fds):
            os.close(directory_fd)


def _active_asset_task_dir(run_dir: Path) -> Path | None:
    candidates: list[Path] = []
    root_fd: int | None = None
    try:
        root_fd = os.open(
            _lexical_absolute_path(run_dir),
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        with os.scandir(root_fd) as entries:
            for index, entry in enumerate(entries):
                if index >= LIVE_PROGRESS_MAX_DIRECTORY_ENTRIES:
                    return None
                if not entry.name.startswith("02-asset-tasks"):
                    continue
                metadata = entry.stat(follow_symlinks=False)
                if stat.S_ISDIR(metadata.st_mode):
                    candidates.append(run_dir / entry.name)
    except OSError:
        return None
    finally:
        if root_fd is not None:
            os.close(root_fd)
    if not candidates:
        return None

    def retry_revision(path: Path) -> int:
        if path.name == "02-asset-tasks":
            return 0
        match = re.fullmatch(r"02-asset-tasks-r(\d+)", path.name)
        if not match:
            return -1
        return int(match.group(1))

    def modified_time(path: Path) -> float:
        directory_metadata = _live_progress_artifact_metadata(
            run_dir,
            path,
            directory=True,
        )
        if directory_metadata is None:
            return 0.0
        latest = 0.0
        for name in (
            "asset_task_run_state.json",
            "asset_task_results_index.json",
            "decision_ledger.jsonl",
            "processing_result.json",
        ):
            metadata = _live_progress_artifact_metadata(run_dir, path / name)
            if metadata is not None:
                latest = max(latest, metadata.st_mtime)
        if latest > 0:
            return latest
        return directory_metadata.st_mtime

    return max(
        candidates,
        key=lambda path: (modified_time(path), retry_revision(path), path.name),
    )


def _asset_task_result_count(asset_task_dir: Path) -> int:
    results_index = _load_progress_json_object(
        asset_task_dir.parent, asset_task_dir / "asset_task_results_index.json"
    )
    if results_index is None:
        return 0
    entries = results_index.get("entries")
    if isinstance(entries, list):
        return len(entries)
    return 0


def _material_progress_summary_for_log(run_dir: Path) -> str | None:
    parts: list[str] = []
    assignments = _load_progress_json_object(run_dir, run_dir / "assignments.json")
    if assignments is not None:
        groups = assignments.get("assignments")
        if isinstance(groups, list):
            parts.append(f"assignments={len(groups)}")
        coverage = assignments.get("coverage")
        if isinstance(coverage, dict):
            decision_count = coverage.get("material_decision_prim_count")
            missing_count = coverage.get("missing_assignment_prim_count")
            if isinstance(decision_count, int):
                parts.append(f"decision_prims={decision_count}")
            if isinstance(missing_count, int):
                parts.append(f"missing_prims={missing_count}")

    patch = _load_progress_json_object(
        run_dir,
        run_dir / "raw" / "material_decision_patch.json",
    )
    if patch is not None:
        material_assignments = patch.get("material_assignments")
        reviewed = patch.get("reviewed_no_override")
        if isinstance(material_assignments, list):
            parts.append(f"patch_assignments={len(material_assignments)}")
        if isinstance(reviewed, list):
            parts.append(f"patch_reviewed_no_override={len(reviewed)}")

    if not parts:
        return None
    return "material " + ", ".join(parts)


def _physics_progress_summary_for_log(run_dir: Path) -> str | None:
    parts: list[str] = []
    patch = _load_progress_json_object(
        run_dir,
        run_dir / "raw" / "physics_decision_patch.json",
    )
    if patch is not None:
        decisions = patch.get("decisions")
        if isinstance(decisions, list):
            parts.append(f"patch_decisions={len(decisions)}")

    assignments = _load_progress_json_object(
        run_dir,
        run_dir / "physics_assignments.json",
    )
    if assignments is not None:
        decisions = assignments.get("decisions")
        if isinstance(decisions, list):
            parts.append(f"assignments={len(decisions)}")
        status = assignments.get("validation_status")
        if isinstance(status, str) and status:
            parts.append(f"validation={status}")

    assessment = _load_progress_json_object(
        run_dir, run_dir / "physics_behavior_assessment.json"
    )
    if assessment is not None:
        status = assessment.get("status")
        if isinstance(status, str) and status:
            parts.append(f"assessment={status}")

    history = _load_progress_json_object(
        run_dir, run_dir / "raw" / "physics_visual_validation_history.json"
    )
    if history is not None:
        status = history.get("status")
        iterations = history.get("iterations")
        if isinstance(status, str) and status:
            parts.append(f"visual_history={status}")
        if isinstance(iterations, list):
            parts.append(f"visual_iterations={len(iterations)}")

    if not parts:
        return None
    return "physics " + ", ".join(parts)


def _trace_progress_summary_for_log(run_dir: Path) -> str | None:
    last_event = _last_jsonl_object(
        run_dir,
        run_dir / "trace" / "events.jsonl",
    )
    if last_event is None:
        return None
    event_type = last_event.get("event_type")
    phase = last_event.get("phase")
    if not isinstance(event_type, str) or not event_type:
        return None
    if isinstance(phase, str) and phase:
        return f"last_event={event_type}/{phase}"
    return f"last_event={event_type}"


def _read_live_progress_artifact(
    run_dir: Path,
    path: Path,
    *,
    tail: bool = False,
) -> bytes | None:
    """Read a child-writable progress file without blocking or following links."""

    lexical_run_dir = _lexical_absolute_path(run_dir)
    lexical_path = _lexical_absolute_path(path)
    try:
        relative_path = lexical_path.relative_to(lexical_run_dir)
    except ValueError:
        return None
    if not relative_path.parts:
        return None

    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    file_flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
    directory_fds: list[int] = []
    file_fd: int | None = None
    try:
        directory_fds.append(os.open(lexical_run_dir, directory_flags))
        current_fd = directory_fds[-1]
        for component in relative_path.parts[:-1]:
            next_fd = os.open(component, directory_flags, dir_fd=current_fd)
            directory_fds.append(next_fd)
            current_fd = next_fd

        initial = os.stat(
            relative_path.name,
            dir_fd=current_fd,
            follow_symlinks=False,
        )
        if not stat.S_ISREG(initial.st_mode) or initial.st_nlink != 1:
            return None
        if not tail and initial.st_size > LIVE_PROGRESS_MAX_BYTES:
            return None

        file_fd = os.open(relative_path.name, file_flags, dir_fd=current_fd)
        opened = os.fstat(file_fd)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_dev != initial.st_dev
            or opened.st_ino != initial.st_ino
        ):
            return None

        if tail and opened.st_size > LIVE_PROGRESS_MAX_BYTES:
            os.lseek(file_fd, opened.st_size - LIVE_PROGRESS_MAX_BYTES, os.SEEK_SET)
            remaining = LIVE_PROGRESS_MAX_BYTES
        else:
            remaining = LIVE_PROGRESS_MAX_BYTES + 1
        content = bytearray()
        while remaining > 0:
            chunk = os.read(file_fd, min(64 * 1024, remaining))
            if not chunk:
                break
            content.extend(chunk)
            remaining -= len(chunk)
        if len(content) > LIVE_PROGRESS_MAX_BYTES:
            return None

        final = os.fstat(file_fd)
        current = os.stat(
            relative_path.name,
            dir_fd=current_fd,
            follow_symlinks=False,
        )
        expected_identity = (opened.st_dev, opened.st_ino)
        expected_version = (opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns)
        if (
            not stat.S_ISREG(final.st_mode)
            or final.st_nlink != 1
            or (final.st_dev, final.st_ino) != expected_identity
            or (current.st_dev, current.st_ino) != expected_identity
            or (final.st_size, final.st_mtime_ns, final.st_ctime_ns) != expected_version
            or (current.st_size, current.st_mtime_ns, current.st_ctime_ns)
            != expected_version
        ):
            return None
        return bytes(content)
    except OSError:
        return None
    finally:
        if file_fd is not None:
            os.close(file_fd)
        for directory_fd in reversed(directory_fds):
            os.close(directory_fd)


def _last_jsonl_object(run_dir: Path, path: Path) -> dict[str, Any] | None:
    raw = _read_live_progress_artifact(run_dir, path, tail=True)
    if raw is None:
        return None
    try:
        for line in reversed(raw.splitlines()):
            if not line.strip():
                continue
            value = json.loads(line.decode("utf-8"))
            return value if isinstance(value, dict) else None
    except (UnicodeDecodeError, ValueError, RecursionError):
        return None
    return None


def _load_progress_json_object(run_dir: Path, path: Path) -> dict[str, Any] | None:
    raw = _read_live_progress_artifact(run_dir, path)
    if raw is None:
        return None
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError):
        return None
    return value if isinstance(value, dict) else None


def _latest_progress_artifact(run_dir: Path) -> tuple[str, float] | None:
    candidates = [
        run_dir / "request.json",
        run_dir / "child-output.log",
        run_dir / "child-final.md",
        run_dir / "workbench.log",
        run_dir / "large_scene_run.json",
        run_dir / "trace" / "events.jsonl",
        run_dir / "raw" / "material_run_packet.json",
        run_dir / "raw" / "material_decision_patch.json",
        run_dir / "raw" / "physics_run_packet.json",
        run_dir / "raw" / "physics_candidate_prims.json",
        run_dir / "raw" / "physics_decision_patch.json",
        run_dir / "raw" / "physics_visual_validation_history.json",
        run_dir / "assignments.json",
        run_dir / "api_operation_counts.json",
        run_dir / "visual_quality_assessment.json",
        run_dir / "physics_assignments.json",
        run_dir / "physics_behavior_assessment.json",
        run_dir / "validation_evidence.json",
        run_dir / "final_summary.md",
        run_dir / "01-decomposition" / "decomposition_result.json",
    ]
    asset_task_dir = _active_asset_task_dir(run_dir)
    if asset_task_dir is not None:
        candidates.extend(
            [
                asset_task_dir / "agent_plan" / "current.json",
                asset_task_dir / "agent_plan" / "revision-0001.md",
                asset_task_dir / "asset_task_run_state.json",
                asset_task_dir / "asset_task_results_index.json",
                asset_task_dir / "decision_ledger.jsonl",
                asset_task_dir / "processing_result.json",
            ]
        )
    latest_path: Path | None = None
    latest_mtime = 0.0
    for path in candidates:
        metadata = _live_progress_artifact_metadata(run_dir, path)
        if metadata is None:
            continue
        mtime = metadata.st_mtime
        if mtime > latest_mtime:
            latest_path = path
            latest_mtime = mtime
    if latest_path is None:
        return None
    try:
        rel_path = latest_path.relative_to(run_dir).as_posix()
    except ValueError:
        rel_path = str(latest_path)
    return rel_path, max(0.0, time.time() - latest_mtime)


def _format_progress_age(age_seconds: float) -> str:
    if age_seconds < 90:
        return f"{age_seconds:.0f}s"
    if age_seconds < 3600:
        return f"{age_seconds / 60:.0f}m"
    return f"{age_seconds / 3600:.1f}h"


def _install_child_signal_handlers(
    *,
    process: subprocess.Popen[str],
    log_stream: Any,
    timeout_label: str,
) -> Callable[[], None]:
    """Install main-thread signal handlers that terminate the child process.

    When called from a non-main thread, Python cannot install signal handlers,
    so this intentionally returns a no-op restore callback.
    """
    if threading.current_thread() is not threading.main_thread():
        return lambda: None

    candidate_signals = [signal.SIGINT, signal.SIGTERM]
    if hasattr(signal, "SIGHUP"):
        candidate_signals.append(signal.SIGHUP)

    previous_handlers: dict[signal.Signals, Any] = {}

    def handler(signum: int, _frame: Any) -> None:
        _write_runner_log_line(
            log_stream,
            f"received signal {signum}; terminating {timeout_label}",
        )
        _request_subprocess_termination(process)
        raise ChildProcessInterrupted(signum, timeout_label)

    for signum in candidate_signals:
        try:
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, handler)
        except (OSError, RuntimeError, ValueError):
            previous_handlers.pop(signum, None)

    def restore() -> None:
        for signum, previous in previous_handlers.items():
            try:
                signal.signal(signum, previous)
            except (OSError, RuntimeError, ValueError):
                continue

    return restore


def _request_subprocess_termination(process: subprocess.Popen[str]) -> None:
    """Ask a child process or process group to stop without waiting."""
    if process.poll() is not None:
        return
    pid = getattr(process, "pid", None)
    if hasattr(os, "killpg") and pid is not None:
        try:
            os.killpg(pid, signal.SIGTERM)
            return
        except ProcessLookupError:
            return
        except OSError:
            pass
    process.terminate()


def _terminate_subprocess(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    pid = getattr(process, "pid", None)
    if hasattr(os, "killpg") and pid is not None:
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        except OSError:
            process.terminate()
    else:
        process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if hasattr(os, "killpg") and pid is not None:
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                return
            except OSError:
                try:
                    process.kill()
                except ProcessLookupError:
                    return
                except OSError as exc:
                    message = (
                        f"Failed to kill process {pid or '<unknown>'} after "
                        f"process-group SIGKILL failed: {exc}"
                    )
                    logger.warning(message)
                    raise RuntimeError(message) from exc
        else:
            process.kill()
        _wait_after_sigkill(process, pid)


def _wait_after_sigkill(process: subprocess.Popen[str], pid: int | None) -> None:
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        message = f"Process {pid or '<unknown>'} did not exit after SIGKILL"
        logger.warning(message)
        raise RuntimeError(message)


def _terminate_subprocess_group(pid: int | None) -> None:
    if pid is None or not hasattr(os, "killpg"):
        return
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except OSError:
        return
    time.sleep(0.2)
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except OSError:
        return


def _prepare_run_dir(config: MaterialAssignConfig) -> Path:
    if config.output_dir is not None:
        run_dir_candidate = config.output_dir.expanduser()
    else:
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
        asset_name = _slug(config.usd_path.stem)
        output_root = getattr(config, "default_output_root", None) or config.repo_root
        run_dir_candidate = output_root / "runs" / f"{asset_name}-{stamp}"
    # Validate the caller-provided lexical path before mkdir or resolve can
    # erase evidence that the run directory itself is a symlink.
    _reject_unsafe_run_links(run_dir_candidate, allow_missing=True)
    run_dir = _lexical_absolute_path(run_dir_candidate)
    run_dir.mkdir(parents=True, exist_ok=True)
    _reject_unsafe_run_links(run_dir)
    _chmod_private(run_dir)
    (run_dir / "raw").mkdir(exist_ok=True)
    (run_dir / "evidence_renders").mkdir(exist_ok=True)
    (run_dir / "final_renders").mkdir(exist_ok=True)
    (run_dir / "trace").mkdir(exist_ok=True)
    _chmod_private(run_dir / "raw")
    for stale_optional_artifact in (
        run_dir / "raw" / "physics_topology_plan.json",
        run_dir / "raw" / "physics_decision_patch_apply.json",
    ):
        stale_optional_artifact.unlink(missing_ok=True)
    return run_dir


def _write_private_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _chmod_private(path)


def _bridge_artifact_prefix(value: str | None, default: str) -> str:
    raw = value or default
    pieces = []
    for char in raw.lower():
        if char.isalnum() or char == "_":
            pieces.append(char)
        elif pieces and pieces[-1] != "_":
            pieces.append("_")
    prefix = "".join(pieces).strip("_")
    return prefix or default


def _load_required_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"Required JSON artifact does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Required JSON artifact is invalid: {path}: {exc}") from exc


def _write_run_cost_metrics(
    *,
    config: AgentRuntimeConfig,
    run_dir: Path,
    request: dict[str, Any],
    wall_time_seconds: float,
) -> None:
    counts = _load_json(run_dir / "api_operation_counts.json", default={})
    if not isinstance(counts, dict):
        counts = {}
    usage = _load_model_usage(run_dir)
    command_records = _load_child_command_records(run_dir)
    metrics = {
        "wall_time_seconds": round(wall_time_seconds, 3),
        "total_tokens": _total_tokens(usage),
        "model_turn_count": _model_turn_count(run_dir),
        "workbench_api_calls_total": _int_metric(counts, "api_operation_count_total"),
        "render_calls_total": _int_metric(counts, "render_count_total"),
        "pick_calls": _int_metric(counts, "pick_calls"),
        "command_calls": _int_metric(counts, "material_override_commands"),
        "shell_commands_total": len(command_records),
        "jq_commands_total": sum(
            1
            for record in command_records
            if _command_has_program(record["command"], "jq")
        ),
        "python_glue_commands_total": sum(
            1
            for record in command_records
            if _command_uses_python_glue(record["command"])
        ),
        "large_file_reads_total": sum(
            1
            for record in command_records
            if _command_reads_large_artifact(record["command"])
        ),
        "docs_reads_total": sum(
            1
            for record in command_records
            if _command_mentions_path_fragment(
                record["command"], ("README", "agent-api")
            )
        ),
        "repeated_file_reads_total": None,
        "failed_commands_total": sum(
            1 for record in command_records if record.get("exit_code") not in (0, None)
        ),
        "failed_api_calls": _optional_int_metric(counts, "failed_api_calls"),
        "retried_api_calls": _optional_int_metric(counts, "retried_api_calls"),
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "cached_input_tokens": usage.get("cached_input_tokens"),
        "estimated_model_cost_usd": None,
        "context": {
            "asset": str(config.usd_path),
            "prompt_id": str(request.get("workflow") or ""),
            "model": config.model or "",
            "reasoning_effort": config.model_reasoning_effort or "",
            "git_sha": _git_head_sha(config.repo_root),
            "workbench_version": "",
            "renderer": "content-workbench",
        },
    }
    (run_dir / "run_cost_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _load_model_usage(run_dir: Path) -> dict[str, Any]:
    usage_total: dict[str, int] = {}
    usage_count = 0
    explicit_total_count = 0
    derived_missing_total = 0
    for path in sorted((run_dir / "raw").glob("*_result.json")):
        payload = _load_json(path, default={})
        if not isinstance(payload, dict):
            continue
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            continue
        usage_count += 1
        for key, value in usage.items():
            if isinstance(value, int):
                usage_total[key] = usage_total.get(key, 0) + value
                if key == "total_tokens":
                    explicit_total_count += 1
        if "total_tokens" not in usage:
            input_tokens = usage.get("input_tokens")
            output_tokens = usage.get("output_tokens")
            if isinstance(input_tokens, int) and isinstance(output_tokens, int):
                derived_missing_total += input_tokens + output_tokens
    if explicit_total_count == 0:
        usage_total.pop("total_tokens", None)
    elif explicit_total_count != usage_count:
        usage_total["total_tokens"] = (
            usage_total.get("total_tokens", 0) + derived_missing_total
        )
    return dict(usage_total)


def _total_tokens(usage: dict[str, Any]) -> int | None:
    explicit_total = usage.get("total_tokens")
    if isinstance(explicit_total, int):
        return explicit_total
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    if isinstance(input_tokens, int) and isinstance(output_tokens, int):
        return input_tokens + output_tokens
    return None


def _model_turn_count(run_dir: Path) -> int:
    count = 0
    for path in sorted((run_dir / "raw").glob("*_result.json")):
        result = _load_json(path, default={})
        if isinstance(result, dict) and result:
            count += 1
    return count


def _load_child_command_records(run_dir: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted((run_dir / "raw").glob("*_items.json")):
        items = _load_json(path, default=[])
        if isinstance(items, dict):
            items = items.get("items") or []
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            command = item.get("command")
            if not isinstance(command, str) or not command.strip():
                continue
            records.append({"command": command, "exit_code": item.get("exit_code")})
    return records


def _command_tokens(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return command.split()


def _command_has_program(command: str, program: str) -> bool:
    return any(Path(token).name == program for token in _command_tokens(command))


def _command_uses_python_glue(command: str) -> bool:
    for token in _command_tokens(command):
        name = Path(token).name
        if name.startswith("python") or name.endswith(".py"):
            return True
    return False


def _command_mentions_path_fragment(command: str, fragments: tuple[str, ...]) -> bool:
    return any(
        any(fragment in token for fragment in fragments)
        for token in _command_tokens(command)
    )


def _command_reads_large_artifact(command: str) -> bool:
    return _command_has_program(command, "cat") and _command_mentions_path_fragment(
        command, ("scene_snapshot",)
    )


def _int_metric(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    return value if isinstance(value, int) else 0


def _optional_int_metric(payload: dict[str, Any], key: str) -> int | None:
    value = payload.get(key)
    return value if isinstance(value, int) else None


def _git_head_sha(repo_root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""
    return completed.stdout.strip()


def _chmod_private(path: Path) -> None:
    try:
        os.chmod(path, 0o600 if path.is_file() else 0o700)
    except OSError as exc:
        logger.warning("Unable to restrict permissions on %s: %s", path, exc)


def _build_material_assignment_child_prompt(
    *,
    config: MaterialAssignConfig,
    run_dir: Path,
    preflight_packet: dict[str, Any] | None,
) -> str:
    builder = (
        build_skill_routed_material_assignment_prompt
        if config.prompt_mode == PROMPT_MODE_SKILL_ROUTED
        else build_material_assignment_prompt
    )
    return builder(
        repo_root=config.repo_root,
        run_dir=run_dir,
        usd_path=config.usd_path,
        reference_images=config.reference_images,
        materials_yaml=config.materials_yaml,
        materials_usd=config.materials_usd,
        workbench_url=config.workbench_url,
        reference_files=config.reference_files or [],
        optimize=config.optimize,
        optimizer_options=_optimizer_options(config),
        material_candidate_policy=_material_candidate_policy(config),
        respect_existing_material_bindings=config.respect_existing_material_bindings,
        additional_instructions=config.additional_instructions,
        preflight_packet=preflight_packet,
        vqa_refinement_max_iterations=config.vqa_refinement_max_iterations,
    )


def _build_request(config: MaterialAssignConfig, run_dir: Path) -> dict[str, object]:
    optimizer_options = _optimizer_options(config)
    material_candidate_policy = _material_candidate_policy(config)
    return {
        "schema_version": "content-agents.request.v1",
        "created_at": utc_now(),
        "workflow": "materials.assign",
        "run_dir": str(run_dir),
        "dry_run": config.dry_run,
        "workbench_url": config.workbench_url,
        "workbench_optimize": config.optimize,
        "optimizer_selection": config.optimizer_selection,
        "optimizer_options": optimizer_options,
        "material_candidate_policy": material_candidate_policy,
        "appearance_evidence_policy": {
            "schema_version": "content-agent-workflows.appearance-evidence-policy.v1",
            "default": "ignore",
            "global_sources": [],
            "scopes": [],
        },
        "clear_materials": not config.respect_existing_material_bindings,
        "respect_existing_material_bindings": config.respect_existing_material_bindings,
        "preflight_enabled": config.preflight and not config.dry_run,
        "runner": config.runner,
        "model": config.model,
        "model_reasoning_effort": config.model_reasoning_effort,
        "codex_base_url": config.codex_base_url,
        "codex_sandbox_mode": config.codex_sandbox_mode,
        "codex_config": config.codex_config or {},
        "claude_config": config.claude_config or {},
        "claude_permission_mode": config.claude_permission_mode,
        "claude_max_turns": config.claude_max_turns,
        "child_timeout_seconds": config.child_timeout_seconds,
        "prompt_mode": config.prompt_mode,
        "vqa_refinement_max_iterations": config.vqa_refinement_max_iterations,
        "codex_persistent_refinement": config.codex_persistent_refinement,
        "additional_instructions": (
            config.additional_instructions.strip()
            if config.additional_instructions and config.additional_instructions.strip()
            else None
        ),
        "inputs": {
            "usd": str(config.usd_path),
            "output_usd": str(config.output_usd_path)
            if config.output_usd_path is not None
            else None,
            "reference_images": [str(path) for path in config.reference_images],
            "reference_files": [str(path) for path in config.reference_files or []],
            "materials_yaml": str(config.materials_yaml),
            "materials_usd": str(config.materials_usd),
        },
        "constraints": {
            "use_workbench_only": True,
            "workbench_optimize": config.optimize,
            "session_scoped_material_overrides": True,
            "clear_materials": not config.respect_existing_material_bindings,
            "source_usd_edits_allowed": False,
            "optimizer_options": optimizer_options,
            "material_candidate_policy": material_candidate_policy,
        },
    }


def _optimizer_options(config: MaterialAssignConfig) -> dict[str, object]:
    return {
        "flatten_prototypes": config.flatten_prototypes,
        "enable_deinstance": config.enable_deinstance,
        "enable_split": config.enable_split,
        "enable_deduplicate": config.enable_deduplicate,
    }


def _physics_optimizer_options(config: PhysicsApplyConfig) -> dict[str, object]:
    return {
        "flatten_prototypes": config.flatten_prototypes,
        "enable_deinstance": config.enable_deinstance,
        "enable_split": config.enable_split,
        "enable_deduplicate": config.enable_deduplicate,
    }


def _run_material_optimizer_selection(
    *,
    config: MaterialAssignConfig,
    run_dir: Path,
    trace_writer: TraceWriter,
    managed_workbench: ManagedWorkbench | None,
) -> tuple[MaterialAssignConfig, dict[str, Any]]:
    analysis_dir = run_dir / "optimizer_analysis"
    analysis_packet = prepare_material_run_packet(
        MaterialRunPacketConfig(
            workbench_url=config.workbench_url,
            run_dir=analysis_dir,
            usd_path=config.usd_path,
            materials_yaml=config.materials_yaml,
            materials_usd=config.materials_usd,
            optimize=False,
            root_prim_path=config.root_prim_path,
            material_candidate_space=config.material_candidate_space,
            skip_instances=config.skip_instances,
            skip_prototypes=config.skip_prototypes,
            skip_invisible=config.skip_invisible,
            respect_existing_material_bindings=(
                config.respect_existing_material_bindings
            ),
        )
    )
    analysis_session_id = _preflight_session_id(analysis_packet)
    if not analysis_session_id:
        raise RuntimeError(
            "Material optimizer analysis did not return a Workbench session ID."
        )
    decision_path = run_dir / "raw" / "optimizer_decision.json"
    prompt_path = run_dir / "raw" / "optimizer_selection_prompt.md"
    child_output_path = run_dir / "raw" / "optimizer_selection_child-output.log"
    child_final_path = run_dir / "raw" / "optimizer_selection_child-final.md"
    prompt = build_material_optimizer_selection_prompt(
        asset_path=config.usd_path,
        run_dir=run_dir,
        analysis_run_dir=analysis_dir,
        workbench_url=config.workbench_url,
        session_id=analysis_session_id,
        decision_path=decision_path,
        additional_instructions=config.additional_instructions,
    )
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(prompt, encoding="utf-8")
    trace_writer.write(
        "optimizer_selection_started",
        phase="optimizer selection",
        summary=(
            "Started an agent turn to select Scene Optimizer settings from an "
            "unoptimized Workbench inspection."
        ),
        artifacts=[
            str(prompt_path),
            str(analysis_dir / "raw" / "material_run_packet.json"),
        ],
        data={"session_id": analysis_session_id},
    )
    try:
        returncode = _run_child_agent(
            config=config,
            prompt=prompt,
            run_dir=run_dir,
            child_output_path=child_output_path,
            child_final_path=child_final_path,
            managed_workbench=managed_workbench,
            prompt_image_inputs=packet_image_inputs(analysis_packet),
            bridge_artifact_prefix="optimizer_selection",
        )
    finally:
        if analysis_session_id:
            try:
                close_workbench_session(
                    config.workbench_url,
                    analysis_session_id,
                    timeout=config.workbench_timeout_seconds,
                )
            except Exception as exc:  # noqa: BLE001 - preserve selection outcome
                trace_writer.write(
                    "warning",
                    phase="optimizer selection",
                    summary="Failed to close the optimizer-analysis session.",
                    data={"error_type": type(exc).__name__, "error": str(exc)},
                )
    if returncode != 0:
        raise RuntimeError(
            f"Optimizer-selection agent exited with return code {returncode}."
        )
    decision = _load_optimizer_decision(
        decision_path, expected_task="material_assignment"
    )
    _write_private_json(decision_path, decision)
    selected = replace(
        config,
        optimize=bool(decision["optimize"]),
        flatten_prototypes=decision["flatten_prototypes"],
        enable_deinstance=decision["enable_deinstance"],
        enable_split=decision["enable_split"],
        enable_deduplicate=decision["enable_deduplicate"],
    )
    trace_writer.write(
        "optimizer_selection_finished",
        phase="optimizer selection",
        summary="Applied the agent-selected Scene Optimizer settings.",
        artifacts=[str(decision_path), str(child_final_path)],
        data={
            "optimize": selected.optimize,
            "optimizer_options": _optimizer_options(selected),
        },
    )
    return selected, decision


def _load_material_optimizer_decision(path: Path) -> dict[str, Any]:
    """Compatibility wrapper for material optimizer-decision tests and callers."""

    return _load_optimizer_decision(path, expected_task="material_assignment")


def _load_optimizer_decision(
    path: Path,
    *,
    expected_task: str,
) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Optimizer-selection agent did not write {path}.")
    payload = _load_required_json(path)
    if not isinstance(payload, dict):
        raise ValueError("optimizer_decision.json must contain a JSON object.")
    if payload.get("schema_version") != "content-agents.optimizer-decision.v1":
        raise ValueError(
            "optimizer_decision.json must use "
            "schema_version content-agents.optimizer-decision.v1."
        )
    if payload.get("task") != expected_task:
        raise ValueError(f"optimizer_decision.json task must be {expected_task!r}.")
    optimize = payload.get("optimize")
    if not isinstance(optimize, bool):
        raise ValueError("optimizer_decision.json optimize must be a boolean.")
    option_names = (
        "flatten_prototypes",
        "enable_deinstance",
        "enable_split",
        "enable_deduplicate",
    )
    for name in option_names:
        if payload.get(name) is not None and not isinstance(payload.get(name), bool):
            raise ValueError(
                f"optimizer_decision.json {name} must be a boolean or null."
            )
    rationale = payload.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("optimizer_decision.json rationale must be non-empty.")
    evidence = payload.get("evidence")
    if (
        not isinstance(evidence, list)
        or not evidence
        or not all(isinstance(item, str) and item.strip() for item in evidence)
    ):
        raise ValueError(
            "optimizer_decision.json evidence must contain non-empty strings."
        )
    if not optimize:
        for name in option_names:
            payload[name] = None
    elif all(payload.get(name) is False for name in option_names):
        raise ValueError(
            "An optimized decision must enable at least one of flatten, deinstance, "
            "split, or deduplicate, or leave an operation null to use its backend "
            "default."
        )
    return payload


def _run_physics_optimizer_selection(
    *,
    config: PhysicsApplyConfig,
    run_dir: Path,
    trace_writer: TraceWriter,
    managed_workbench: ManagedWorkbench | None,
) -> tuple[PhysicsApplyConfig, dict[str, Any]]:
    analysis_dir = run_dir / "optimizer_analysis"
    analysis_config = replace(
        config,
        optimize=False,
        optimizer_selection=OPTIMIZER_SELECTION_FIXED,
        flatten_prototypes=None,
        enable_deinstance=None,
        enable_split=None,
        enable_deduplicate=None,
    )
    analysis_packet = _prepare_physics_run_packet(analysis_config, analysis_dir)
    analysis_session_id = _preflight_session_id(analysis_packet)
    if not analysis_session_id:
        raise RuntimeError(
            "Physics optimizer analysis did not return a Workbench session ID."
        )
    decision_path = run_dir / "raw" / "optimizer_decision.json"
    prompt_path = run_dir / "raw" / "optimizer_selection_prompt.md"
    child_output_path = run_dir / "raw" / "optimizer_selection_child-output.log"
    child_final_path = run_dir / "raw" / "optimizer_selection_child-final.md"
    prompt = build_physics_optimizer_selection_prompt(
        asset_path=config.usd_path,
        run_dir=run_dir,
        analysis_run_dir=analysis_dir,
        workbench_url=config.workbench_url,
        session_id=analysis_session_id,
        decision_path=decision_path,
        collision_approximation=config.collision_approximation,
        runtime_validation_enabled=config.run_simulation,
        additional_instructions=config.additional_instructions,
    )
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(prompt, encoding="utf-8")
    trace_writer.write(
        "optimizer_selection_started",
        phase="optimizer selection",
        summary=(
            "Started a physics-task agent turn to select Scene Optimizer "
            "settings from unoptimized component and topology evidence."
        ),
        artifacts=[
            str(prompt_path),
            str(analysis_dir / "raw" / "physics_components.json"),
            str(analysis_dir / "raw" / "physics_topology.json"),
        ],
        data={"session_id": analysis_session_id, "task": "physics_authoring"},
    )
    try:
        returncode = _run_child_agent(
            config=config,
            prompt=prompt,
            run_dir=run_dir,
            child_output_path=child_output_path,
            child_final_path=child_final_path,
            managed_workbench=managed_workbench,
            prompt_image_inputs=[],
            bridge_artifact_prefix="optimizer_selection",
        )
    finally:
        if analysis_session_id:
            try:
                close_workbench_session(
                    config.workbench_url,
                    analysis_session_id,
                    timeout=config.workbench_timeout_seconds,
                )
            except Exception as exc:  # noqa: BLE001 - preserve selection outcome
                trace_writer.write(
                    "warning",
                    phase="optimizer selection",
                    summary="Failed to close the physics optimizer-analysis session.",
                    data={"error_type": type(exc).__name__, "error": str(exc)},
                )
    if returncode != 0:
        raise RuntimeError(
            f"Physics optimizer-selection agent exited with return code {returncode}."
        )
    decision = _load_optimizer_decision(
        decision_path,
        expected_task="physics_authoring",
    )
    _write_private_json(decision_path, decision)
    selected = replace(
        config,
        optimize=bool(decision["optimize"]),
        flatten_prototypes=decision["flatten_prototypes"],
        enable_deinstance=decision["enable_deinstance"],
        enable_split=decision["enable_split"],
        enable_deduplicate=decision["enable_deduplicate"],
    )
    trace_writer.write(
        "optimizer_selection_finished",
        phase="optimizer selection",
        summary="Applied the physics-task agent-selected optimizer settings.",
        artifacts=[str(decision_path), str(child_final_path)],
        data={
            "task": "physics_authoring",
            "optimize": selected.optimize,
            "optimizer_options": _physics_optimizer_options(selected),
        },
    )
    return selected, decision


def _material_candidate_policy(config: MaterialAssignConfig) -> dict[str, object]:
    return {
        "material_candidate_space": config.material_candidate_space,
        "root_prim_path": config.root_prim_path,
        "skip_instances": config.skip_instances,
        "skip_prototypes": config.skip_prototypes,
        "skip_invisible": config.skip_invisible,
    }


def _build_physics_request(
    config: PhysicsApplyConfig,
    run_dir: Path,
) -> dict[str, object]:
    return {
        "schema_version": "content-agents.request.v1",
        "created_at": utc_now(),
        "workflow": "physics.apply",
        "run_dir": str(run_dir),
        "dry_run": config.dry_run,
        "workbench_url": config.workbench_url,
        "workbench_optimize": config.optimize,
        "optimizer_selection": config.optimizer_selection,
        "optimizer_options": _physics_optimizer_options(config),
        "runner": config.runner,
        "model": config.model,
        "model_reasoning_effort": config.model_reasoning_effort,
        "codex_base_url": config.codex_base_url,
        "codex_sandbox_mode": config.codex_sandbox_mode,
        "codex_config": config.codex_config or {},
        "claude_config": config.claude_config or {},
        "claude_permission_mode": config.claude_permission_mode,
        "claude_max_turns": config.claude_max_turns,
        "child_timeout_seconds": config.child_timeout_seconds,
        "prompt_mode": config.prompt_mode,
        "visual_validation_max_iterations": config.vqa_refinement_max_iterations,
        "codex_persistent_refinement": config.codex_persistent_refinement,
        "additional_instructions": (
            config.additional_instructions.strip()
            if config.additional_instructions and config.additional_instructions.strip()
            else None
        ),
        "inputs": {
            "usd": str(config.usd_path),
            "reference_images": [str(path) for path in config.reference_images or []],
            "reference_files": [str(path) for path in config.reference_files or []],
            "output_usd": str(config.output_usd_path)
            if config.output_usd_path is not None
            else None,
        },
        "runtime_validation": {
            "enabled": config.run_simulation,
            "engine": config.simulation_engine,
            "duration_s": config.simulation_duration_s,
            "dt": config.simulation_dt,
            "sample_fps": config.simulation_sample_fps,
            "drop_height_m": config.drop_height_m,
        },
        "constraints": {
            "use_workbench_only": True,
            "source_usd_edits_allowed": False,
            "ovphysx_authoritative": True,
            "visual_issues_default_to_conditional": True,
            "collision_approximation": config.collision_approximation,
            "optimizer_task": "physics_authoring",
            "optimizer_options": _physics_optimizer_options(config),
        },
    }


def _prepare_physics_run_packet(
    config: PhysicsApplyConfig,
    run_dir: Path,
) -> dict[str, Any]:
    raw_dir = run_dir / "raw"
    docs_dir = raw_dir / "workbench_docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    try:
        docs = workbench_client.download_agent_api_docs(
            config.workbench_url,
            docs_dir,
            timeout=config.workbench_timeout_seconds,
        )
    except Exception as exc:  # noqa: BLE001 - docs are useful but not critical
        docs = {"download_error": f"{type(exc).__name__}: {exc}"}

    session = workbench_client.create_session(
        config.workbench_url,
        {
            "scene_path": str(config.usd_path),
            "optimize": config.optimize,
            "clear_materials": False,
            "width": 1024,
            "height": 768,
            **{
                key: value
                for key, value in _physics_optimizer_options(config).items()
                if value is not None
            },
        },
        timeout=config.workbench_timeout_seconds,
    )
    session_id = str(session.get("session_id") or "")
    if not session_id:
        raise RuntimeError("Content Workbench did not return a session_id.")
    try:
        components = workbench_client.inspect_physics_components(
            config.workbench_url,
            session_id,
            {
                "usd_path": str(config.usd_path),
                "path_space": "source",
            },
            timeout=config.workbench_timeout_seconds,
        )
        topology = workbench_client.inspect_physics_topology(
            config.workbench_url,
            session_id,
            {
                "usd_path": str(config.usd_path),
                "path_space": "source",
            },
            timeout=config.workbench_timeout_seconds,
        )
    except Exception:
        try:
            close_workbench_session(
                config.workbench_url,
                session_id,
                timeout=config.workbench_timeout_seconds,
            )
        except Exception:  # noqa: BLE001 - preserve the inspection failure
            pass
        raise
    components_path = raw_dir / "physics_components.json"
    components_path.write_text(
        json.dumps(components, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    topology_path = raw_dir / "physics_topology.json"
    topology_path.write_text(
        json.dumps(topology, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    packet = {
        "schema_version": "content-agents.physics-run-packet.v2",
        "session_id": session_id,
        "session": session,
        "agent_api_docs": docs,
        "components_path": str(components_path),
        "topology_path": str(topology_path),
        "component_count": components.get("component_count")
        if isinstance(components, dict)
        else None,
        "workbench": {
            "endpoint": config.workbench_url,
            "optimize": config.optimize,
            "optimizer_options": _physics_optimizer_options(config),
        },
    }
    _write_private_json(raw_dir / "physics_run_packet.json", packet)
    return packet


def _run_physics_visual_refinement_loop(
    *,
    config: PhysicsApplyConfig,
    run_dir: Path,
    session_id: str,
    trace_writer: TraceWriter,
    managed_workbench: ManagedWorkbench | None,
    codex_session: CodexThreadSession | None = None,
) -> tuple[int, str]:
    from content_agent_workflows.common import ValidationEvidence
    from content_agent_workflows.physics import (
        default_physics_behavior_assessment,
        load_physics_behavior_assessment,
        merge_physics_behavior_assessment,
    )

    max_iterations = max(1, config.vqa_refinement_max_iterations)
    history_path = run_dir / "raw" / "physics_visual_validation_history.json"
    history: dict[str, Any] = {
        "schema_version": "content-agents.physics-visual-validation-history.v1",
        "run_dir": str(run_dir),
        "max_iterations": max_iterations,
        "iterations": [],
    }
    previous_assessment_path: Path | None = None
    returncode = 0

    for iteration in range(1, max_iterations + 1):
        patch_path = run_dir / "raw" / "physics_decision_patch.json"
        finalize_record = _finalize_physics_once(
            config=config,
            run_dir=run_dir,
            session_id=session_id,
            iteration=iteration,
            trace_writer=trace_writer,
        )
        # Deterministic finalization may normalize the patch. Only changes made
        # by the subsequent visual-review turn require a physics rerun.
        patch_digest_before = _file_digest(patch_path)
        validation_path = Path(str(finalize_record.get("validation_evidence_path")))
        runtime_report_path = (
            Path(str(finalize_record["simulation_report_path"]))
            if finalize_record.get("simulation_report_path")
            else None
        )
        rendered_frames = [
            str(path) for path in finalize_record.get("rendered_frames") or []
        ]
        issue_packet_path = _write_physics_visual_issue_packet(
            run_dir=run_dir,
            iteration=iteration,
            finalize_record=finalize_record,
            previous_assessment_path=previous_assessment_path,
        )
        prompt = build_physics_visual_refinement_prompt(
            repo_root=config.repo_root,
            run_dir=run_dir,
            usd_path=config.usd_path,
            workbench_url=config.workbench_url,
            session_id=session_id,
            iteration=iteration,
            max_iterations=max_iterations,
            decision_patch_path=patch_path,
            validation_evidence_path=validation_path,
            runtime_report_path=runtime_report_path,
            rendered_frames=rendered_frames,
            previous_assessment_path=previous_assessment_path,
            issue_packet_path=issue_packet_path,
        )
        prompt_path = run_dir / "raw" / f"physics_visual_review_prompt_{iteration}.md"
        child_output_path = (
            run_dir / "raw" / f"physics_visual_review_{iteration}_child-output.log"
        )
        child_final_path = (
            run_dir / "raw" / f"physics_visual_review_{iteration}_child-final.md"
        )
        prompt_path.write_text(prompt, encoding="utf-8")
        trace_writer.write(
            "physics_visual_review_started",
            phase="visual validation",
            summary=(
                f"Started physics visual validation/refinement iteration "
                f"{iteration}/{max_iterations}."
            ),
            artifacts=[str(prompt_path), str(issue_packet_path)],
            data={"rendered_frame_count": len(rendered_frames)},
        )
        try:
            child_returncode = _run_child_agent(
                config=config,
                prompt=prompt,
                run_dir=run_dir,
                child_output_path=child_output_path,
                child_final_path=child_final_path,
                managed_workbench=managed_workbench,
                prompt_image_inputs=[
                    {"label": f"Physics validation frame {index}", "path": path}
                    for index, path in enumerate(rendered_frames, start=1)
                ],
                bridge_artifact_prefix=f"physics_visual_review_{iteration}",
                codex_session=codex_session,
            )
        except UnsafeRunArtifactError:
            raise
        except Exception as exc:  # noqa: BLE001 - preserve review artifacts
            child_returncode = 2
            _append_child_runner_error(child_output_path, exc, run_dir=run_dir)
            trace_writer.write(
                "child_agent_failed",
                phase="visual validation",
                summary="Physics visual review child agent runner failed.",
                artifacts=[str(child_output_path), str(child_final_path)],
                data={"error_type": type(exc).__name__, "error": str(exc)},
            )
        returncode = child_returncode if returncode == 0 else returncode

        assessment_path = run_dir / "physics_behavior_assessment.json"
        if assessment_path.exists():
            assessment = load_physics_behavior_assessment(assessment_path)
        else:
            assessment = default_physics_behavior_assessment(
                runtime_report=runtime_report_path,
                rendered_frames=rendered_frames,
                unresolved_issue=(
                    "Physics visual review did not produce "
                    "physics_behavior_assessment.json."
                ),
            )
            assessment_path.write_text(
                json.dumps(assessment.model_dump(mode="json"), indent=2),
                encoding="utf-8",
            )

        patch_digest_after = _file_digest(patch_path)
        patch_changed = patch_digest_after != patch_digest_before
        if patch_changed and iteration < max_iterations:
            history["iterations"].append(
                {
                    "iteration": iteration,
                    "child_returncode": child_returncode,
                    "patch_changed": True,
                    "assessment_path": str(assessment_path),
                    "rerun_required": True,
                }
            )
            _write_private_json(history_path, history)
            trace_writer.write(
                "physics_patch_updated",
                phase="visual validation",
                summary=(
                    "Physics visual review updated the decision patch; "
                    "rerunning schema apply and runtime validation."
                ),
                artifacts=[str(patch_path), str(assessment_path)],
                data={"iteration": iteration},
            )
            session_id = _restart_physics_refinement_session(
                config=config,
                run_dir=run_dir,
                current_session_id=session_id,
                managed_workbench=managed_workbench,
            )
            trace_writer.write(
                "physics_refinement_session_restarted",
                phase="visual validation",
                summary=(
                    "Restarted Workbench with a fresh source session before "
                    "reapplying the updated physics patch."
                ),
                data={"iteration": iteration, "session_id": session_id},
            )
            previous_assessment_path = assessment_path
            continue

        if patch_changed:
            assessment.unresolved_issues.append(
                {
                    "severity": "warning",
                    "description": (
                        "Physics decision patch changed on the final visual "
                        "validation iteration and was not rerun through ovphysx."
                    ),
                }
            )
            if assessment.status == "pass":
                assessment.status = "unresolved_issues"
            assessment_path.write_text(
                json.dumps(assessment.model_dump(mode="json"), indent=2),
                encoding="utf-8",
            )

        evidence_payload = _load_required_json(validation_path)
        evidence = ValidationEvidence.model_validate(evidence_payload)
        evidence = merge_physics_behavior_assessment(
            evidence,
            assessment,
            assessment_path=assessment_path,
        )
        validation_path.write_text(
            json.dumps(evidence.model_dump(mode="json"), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        _update_physics_assignments_after_visual(
            run_dir=run_dir,
            assessment_path=assessment_path,
            rendered_frames=rendered_frames,
            validation_status=evidence.sim_ready_status,
        )
        _write_physics_final_summary(
            run_dir=run_dir,
            validation_status=evidence.sim_ready_status,
            assessment_path=assessment_path,
            validation_evidence_path=validation_path,
            runtime_report_path=runtime_report_path,
            rendered_frames=rendered_frames,
        )
        unresolved = [str(item) for item in assessment.unresolved_issues]
        history["iterations"].append(
            {
                "iteration": iteration,
                "child_returncode": child_returncode,
                "patch_changed": patch_changed,
                "assessment_status": assessment.status,
                "unresolved_issues": unresolved,
                "validation_status": evidence.sim_ready_status,
                "assessment_path": str(assessment_path),
            }
        )
        _write_private_json(history_path, history)
        trace_writer.write(
            "physics_visual_review_finished",
            phase="visual validation",
            summary=(
                f"Physics visual review iteration {iteration} finished with "
                f"status {assessment.status}."
            ),
            artifacts=[str(assessment_path), str(validation_path)],
            data={
                "validation_status": evidence.sim_ready_status,
                "unresolved_issue_count": len(unresolved),
            },
        )
        if config.fail_on_validation_error and evidence.sim_ready_status == "fail":
            history["status"] = "runtime_failed"
            history["stop_reason"] = (
                "Physics runtime validation failed and fail-on-validation-error "
                "is enabled."
            )
            _write_private_json(history_path, history)
            return 1, session_id
        if not unresolved and child_returncode == 0:
            history["status"] = "satisfied"
            history["stop_reason"] = (
                f"Physics visual validation iteration {iteration} passed."
            )
            _write_private_json(history_path, history)
            return 0, session_id
        previous_assessment_path = assessment_path
        if child_returncode != 0:
            history["status"] = "child_failed"
            history["stop_reason"] = (
                f"Physics visual validation iteration {iteration} failed with "
                f"return code {child_returncode}."
            )
            _write_private_json(history_path, history)
            return child_returncode, session_id
        break

    history.setdefault("status", "max_iterations_reached")
    history.setdefault(
        "stop_reason",
        "Physics visual validation stopped with unresolved or conditional issues.",
    )
    _write_private_json(history_path, history)
    return returncode, session_id


def _restart_physics_refinement_session(
    *,
    config: PhysicsApplyConfig,
    run_dir: Path,
    current_session_id: str,
    managed_workbench: ManagedWorkbench | None,
) -> str:
    try:
        close_workbench_session(
            config.workbench_url,
            current_session_id,
            timeout=config.workbench_timeout_seconds,
        )
    except Exception:
        pass

    if managed_workbench is not None:
        managed_workbench.stop()
        managed_workbench.start()
    else:
        wait_for_workbench(
            config.workbench_url,
            timeout_seconds=config.workbench_timeout_seconds,
            output_root=run_dir,
        )

    session = workbench_client.create_session(
        config.workbench_url,
        {
            "scene_path": str(config.usd_path),
            "optimize": config.optimize,
            "clear_materials": False,
            "width": 1024,
            "height": 768,
            **{
                key: value
                for key, value in _physics_optimizer_options(config).items()
                if value is not None
            },
        },
        timeout=config.workbench_timeout_seconds,
    )
    session_id = str(session.get("session_id") or "")
    if not session_id:
        raise RuntimeError(
            "Content Workbench did not return a session_id after physics refinement restart."
        )
    return session_id


def _finalize_physics_once(
    *,
    config: PhysicsApplyConfig,
    run_dir: Path,
    session_id: str,
    iteration: int,
    trace_writer: TraceWriter,
) -> dict[str, Any]:
    from content_agent_workflows.physics import (
        PhysicsApplyWorkflowInput,
        run_physics_apply_workflow,
    )

    result = run_physics_apply_workflow(
        PhysicsApplyWorkflowInput(
            usd_path=config.usd_path,
            output_dir=run_dir,
            output_usd_path=config.output_usd_path or (run_dir / "physics.usda"),
            decision_patch_path=run_dir / "raw" / "physics_decision_patch.json",
            topology_plan_path=(
                run_dir / "raw" / "physics_topology_plan.json"
                if (run_dir / "raw" / "physics_topology_plan.json").is_file()
                else None
            ),
            collision_approximation=config.collision_approximation,
            run_simulation=config.run_simulation and config.simulation_engine != "none",
            simulation_engine=config.simulation_engine,
            simulation_duration_s=config.simulation_duration_s,
            simulation_dt=config.simulation_dt,
            simulation_sample_fps=config.simulation_sample_fps,
            drop_height_m=config.drop_height_m,
            fail_on_validation_error=False,
            workbench_url=config.workbench_url,
            workbench_session_id=session_id,
            workbench_timeout_s=config.workbench_timeout_seconds,
        )
    )
    record = result.model_dump(mode="json")
    record_path = run_dir / "raw" / f"physics_finalize_result_{iteration}.json"
    _write_private_json(record_path, record)
    if not result.validation_evidence_path:
        raise RuntimeError(
            "Physics deterministic finalizer did not produce validation evidence: "
            f"{result.error or 'unknown error'}"
        )
    rendered_frames = _render_physics_validation_frames(
        config=config,
        run_dir=run_dir,
        session_id=session_id,
        simulation_report_path=Path(result.simulation_report_path)
        if result.simulation_report_path
        else None,
        iteration=iteration,
        trace_writer=trace_writer,
    )
    record["rendered_frames"] = rendered_frames
    _write_private_json(record_path, record)
    trace_writer.write(
        "physics_finalized",
        phase="finalization",
        summary=(
            "Applied physics decision patch, ran runtime validation, and "
            "prepared render evidence."
        ),
        artifacts=[
            path
            for path in [
                result.physics_usd_path,
                result.assignments_path,
                result.validation_evidence_path,
                result.simulation_report_path,
                str(record_path),
            ]
            if path
        ],
        data={
            "validation_status": result.validation_status,
            "rendered_frame_count": len(rendered_frames),
            "success": result.success,
        },
    )
    return record


def _render_physics_validation_frames(
    *,
    config: PhysicsApplyConfig,
    run_dir: Path,
    session_id: str,
    simulation_report_path: Path | None,
    iteration: int,
    trace_writer: TraceWriter,
) -> list[str]:
    if simulation_report_path is None or not simulation_report_path.exists():
        return []
    report = _load_json(simulation_report_path, default={})
    if not isinstance(report, dict):
        return []
    recording = report.get("recording_usda")
    if not isinstance(recording, str) or not recording:
        return []
    frames_dir = run_dir / "runtime" / f"visual_review_frames_{iteration}"
    try:
        response = workbench_client.render_frames(
            config.workbench_url,
            session_id,
            {
                "scene_path": recording,
                "width": 640,
                "height": 480,
                "render_quality": "inspection",
                "save_camera_json": True,
                "camera_path": "+x+y+z",
                "make_mp4": False,
                "max_duration_seconds": max(config.simulation_duration_s, 0.1),
            },
            timeout=config.workbench_timeout_seconds,
        )
    except Exception as exc:  # noqa: BLE001 - visual review can be conditional
        trace_writer.write(
            "warning",
            phase="visual validation",
            summary="Failed to render physics validation recording frames.",
            artifacts=[str(simulation_report_path)],
            data={"error_type": type(exc).__name__, "error": str(exc)},
        )
        return []
    response_path = run_dir / "raw" / f"physics_render_frames_{iteration}.json"
    _write_private_json(response_path, response)
    frame_paths = [
        str(path) for path in response.get("frame_paths", []) if isinstance(path, str)
    ]
    copied_frame_paths: list[str] = []
    for frame_path in frame_paths:
        source = Path(frame_path)
        if not source.exists():
            copied_frame_paths.append(frame_path)
            continue
        frames_dir.mkdir(parents=True, exist_ok=True)
        destination = frames_dir / source.name
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
        copied_frame_paths.append(str(destination))
    frame_paths = copied_frame_paths
    trace_writer.write(
        "physics_validation_frames_rendered",
        phase="visual validation",
        summary="Rendered frame sequence from physics validation recording.",
        artifacts=[str(response_path), *frame_paths],
        data={"frame_count": len(frame_paths), "recording_usda": recording},
    )
    return frame_paths


def _write_physics_visual_issue_packet(
    *,
    run_dir: Path,
    iteration: int,
    finalize_record: dict[str, Any],
    previous_assessment_path: Path | None,
) -> Path:
    path = run_dir / "raw" / f"physics_visual_issue_packet_{iteration}.json"
    packet = {
        "schema_version": "content-agents.physics-visual-issue-packet.v1",
        "iteration": iteration,
        "validation_status": finalize_record.get("validation_status"),
        "validation_evidence_path": finalize_record.get("validation_evidence_path"),
        "simulation_report_path": finalize_record.get("simulation_report_path"),
        "rendered_frames": finalize_record.get("rendered_frames") or [],
        "previous_assessment_path": str(previous_assessment_path)
        if previous_assessment_path
        else None,
    }
    _write_private_json(path, packet)
    return path


def _update_physics_assignments_after_visual(
    *,
    run_dir: Path,
    assessment_path: Path,
    rendered_frames: list[str],
    validation_status: str,
) -> None:
    assignments_path = run_dir / "physics_assignments.json"
    assignments = _load_json(assignments_path, default={})
    if not isinstance(assignments, dict):
        return
    assignments["physics_behavior_assessment"] = str(assessment_path)
    assignments["visual_validation_frames"] = rendered_frames
    assignments["validation_status"] = validation_status
    assignments_path.write_text(
        json.dumps(assignments, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_physics_final_summary(
    *,
    run_dir: Path,
    validation_status: str,
    assessment_path: Path,
    validation_evidence_path: Path,
    runtime_report_path: Path | None,
    rendered_frames: list[str],
) -> None:
    lines = [
        "# Physics Apply Summary",
        "",
        f"- Status: `{validation_status}`",
        f"- Behavior assessment: `{assessment_path}`",
        f"- Validation evidence: `{validation_evidence_path}`",
    ]
    if runtime_report_path is not None:
        lines.append(f"- Runtime report: `{runtime_report_path}`")
    lines.append(f"- Rendered validation frames: {len(rendered_frames)}")
    if rendered_frames:
        lines.extend(["", "## Validation Frames", ""])
        lines.extend(f"- `{path}`" for path in rendered_frames)
    (run_dir / "final_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _file_digest(path: Path) -> str:
    if not path.exists():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_config(config: MaterialAssignConfig) -> None:
    if config.runner not in SUPPORTED_RUNNERS:
        supported = ", ".join(sorted(SUPPORTED_RUNNERS))
        raise ValueError(
            f"Unsupported --runner: {config.runner}. Expected one of: {supported}."
        )
    if config.prompt_mode not in SUPPORTED_PROMPT_MODES:
        supported = ", ".join(sorted(SUPPORTED_PROMPT_MODES))
        raise ValueError(
            f"Unsupported prompt mode: {config.prompt_mode}. Expected one of: {supported}."
        )
    if config.optimizer_selection not in SUPPORTED_OPTIMIZER_SELECTION_MODES:
        supported = ", ".join(sorted(SUPPORTED_OPTIMIZER_SELECTION_MODES))
        raise ValueError(
            "Unsupported optimizer selection mode: "
            f"{config.optimizer_selection}. Expected one of: {supported}."
        )
    if config.optimizer_selection == OPTIMIZER_SELECTION_AGENT:
        if not config.preflight:
            raise ValueError("Agent optimizer selection requires material preflight.")
        explicit_options = {
            name: value
            for name, value in _optimizer_options(config).items()
            if value is not None
        }
        if explicit_options:
            raise ValueError(
                "Agent optimizer selection cannot be combined with fixed optimizer "
                f"options: {', '.join(sorted(explicit_options))}."
            )
    if config.output_usd_path is not None and not config.preflight:
        raise ValueError("--output-usd requires material preflight.")
    if (
        config.output_usd_path is not None
        and config.output_usd_path.resolve() == config.usd_path.resolve()
    ):
        raise ValueError(
            "--output-usd must differ from --usd after resolving relative paths "
            "and symlinks; the source asset is never overwritten."
        )
    if config.material_candidate_space not in {"source", "inspection"}:
        raise ValueError(
            "Unsupported material candidate space: "
            f"{config.material_candidate_space}. Expected one of: inspection, source."
        )
    if config.root_prim_path and not config.root_prim_path.startswith("/"):
        raise ValueError(
            f"--root-prim-path must be an absolute USD prim path: {config.root_prim_path}"
        )
    reference_files = config.reference_files or []
    paths = [
        ("USD", config.usd_path),
        ("materials YAML", config.materials_yaml),
        ("materials USD", config.materials_usd),
        *[
            (f"reference image {index + 1}", path)
            for index, path in enumerate(config.reference_images)
        ],
        *[
            (f"reference file {index + 1}", path)
            for index, path in enumerate(reference_files)
        ],
    ]
    for label, path in paths:
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")
        if not path.is_file():
            raise ValueError(f"{label} is not a file: {path}")
        if not os.access(path, os.R_OK):
            raise PermissionError(f"{label} is not readable: {path}")
    if (
        config.output_usd_path is not None
        and not config.output_usd_path.parent.exists()
    ):
        raise FileNotFoundError(
            f"Output USD directory does not exist: {config.output_usd_path.parent}"
        )
    if not config.reference_images and not reference_files:
        raise ValueError("At least one --reference-image or --reference is required.")
    parsed = urlparse(config.workbench_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"Invalid --workbench-url: {config.workbench_url}")
    if parsed.path not in {"", "/"} or parsed.params or parsed.query or parsed.fragment:
        raise ValueError(
            "--workbench-url must be a root http(s) URL without path, params, "
            f"query, or fragment: {config.workbench_url}"
        )
    if config.start_workbench and not _is_loopback_host(parsed.hostname):
        raise ValueError(
            "--start-workbench only supports loopback --workbench-url hosts; "
            f"got {parsed.hostname!r}."
        )
    if config.codex_base_url:
        parsed_codex = urlparse(config.codex_base_url)
        if parsed_codex.scheme not in {"http", "https"} or not parsed_codex.netloc:
            raise ValueError(f"Invalid --codex-base-url: {config.codex_base_url}")
    if config.codex_sandbox_mode not in SUPPORTED_CODEX_SANDBOX_MODES:
        supported = ", ".join(sorted(SUPPORTED_CODEX_SANDBOX_MODES))
        raise ValueError(
            f"Unsupported --codex-sandbox-mode: {config.codex_sandbox_mode}. "
            f"Expected one of: {supported}"
        )
    if config.claude_max_turns is not None and config.claude_max_turns <= 0:
        raise ValueError("--claude-max-turns must be greater than 0.")
    if config.claude_execution_mode not in SUPPORTED_CLAUDE_EXECUTION_MODES:
        supported = ", ".join(sorted(SUPPORTED_CLAUDE_EXECUTION_MODES))
        raise ValueError(
            "Unsupported --claude-execution-mode: "
            f"{config.claude_execution_mode}. Expected one of: {supported}."
        )
    if (
        config.claude_execution_mode == CLAUDE_EXECUTION_CLI
        and config.claude_max_turns is not None
    ):
        raise ValueError(
            "--claude-max-turns is not supported with "
            "--claude-execution-mode=cli; the claude CLI print mode has no "
            "max-turns equivalent."
        )
    if config.child_timeout_seconds < 0:
        raise ValueError("--child-timeout must be greater than or equal to 0.")
    if config.material_restore_timeout_seconds <= 0:
        raise ValueError("--material-restore-timeout must be greater than 0.")
    if config.vqa_refinement_max_iterations < 0:
        raise ValueError(
            "--vqa-refinement-max-iterations must be greater than or equal to 0."
        )
    if config.child_timeout_seconds == 0:
        if config.start_workbench:
            logger.warning(
                "--child-timeout=0 disables the child process timeout; use only "
                "for local debugging because the wrapper can wait indefinitely."
            )
        else:
            logger.warning(
                "--child-timeout=0 disables the child process timeout and "
                "--no-start-workbench leaves no managed Workbench watchdog; the "
                "wrapper can wait indefinitely for the child or external "
                "Workbench endpoint."
            )
    if (
        config.runner == RUNNER_CLAUDE
        and config.claude_permission_mode == "bypassPermissions"
    ):
        logger.warning(
            "--claude-permission-mode=bypassPermissions disables Claude SDK "
            "permission prompts; use only in a trusted local workspace."
        )


def _validate_physics_config(config: PhysicsApplyConfig) -> None:
    if config.runner not in SUPPORTED_RUNNERS:
        supported = ", ".join(sorted(SUPPORTED_RUNNERS))
        raise ValueError(
            f"Unsupported --runner: {config.runner}. Expected one of: {supported}."
        )
    if config.prompt_mode not in SUPPORTED_PROMPT_MODES:
        supported = ", ".join(sorted(SUPPORTED_PROMPT_MODES))
        raise ValueError(
            f"Unsupported prompt mode: {config.prompt_mode}. Expected one of: {supported}."
        )
    if config.optimizer_selection not in SUPPORTED_OPTIMIZER_SELECTION_MODES:
        supported = ", ".join(sorted(SUPPORTED_OPTIMIZER_SELECTION_MODES))
        raise ValueError(
            "Unsupported optimizer selection mode: "
            f"{config.optimizer_selection}. Expected one of: {supported}."
        )
    if config.optimizer_selection == OPTIMIZER_SELECTION_AGENT:
        explicit_options = {
            name: value
            for name, value in _physics_optimizer_options(config).items()
            if value is not None
        }
        if explicit_options:
            raise ValueError(
                "Agent optimizer selection cannot be combined with fixed physics "
                f"optimizer options: {', '.join(sorted(explicit_options))}."
            )
    paths = [
        ("USD", config.usd_path),
        *[
            (f"reference image {index + 1}", path)
            for index, path in enumerate(config.reference_images or [])
        ],
        *[
            (f"reference file {index + 1}", path)
            for index, path in enumerate(config.reference_files or [])
        ],
    ]
    for label, path in paths:
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")
        if not path.is_file():
            raise ValueError(f"{label} is not a file: {path}")
        if not os.access(path, os.R_OK):
            raise PermissionError(f"{label} is not readable: {path}")
    if (
        config.output_usd_path is not None
        and not config.output_usd_path.parent.exists()
    ):
        raise FileNotFoundError(
            f"Output USD directory does not exist: {config.output_usd_path.parent}"
        )
    parsed = urlparse(config.workbench_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"Invalid --workbench-url: {config.workbench_url}")
    if parsed.path not in {"", "/"} or parsed.params or parsed.query or parsed.fragment:
        raise ValueError(
            "--workbench-url must be a root http(s) URL without path, params, "
            f"query, or fragment: {config.workbench_url}"
        )
    if config.start_workbench and not _is_loopback_host(parsed.hostname):
        raise ValueError(
            "--start-workbench only supports loopback --workbench-url hosts; "
            f"got {parsed.hostname!r}."
        )
    if config.codex_base_url:
        parsed_codex = urlparse(config.codex_base_url)
        if parsed_codex.scheme not in {"http", "https"} or not parsed_codex.netloc:
            raise ValueError(f"Invalid --codex-base-url: {config.codex_base_url}")
    if config.codex_sandbox_mode not in SUPPORTED_CODEX_SANDBOX_MODES:
        supported = ", ".join(sorted(SUPPORTED_CODEX_SANDBOX_MODES))
        raise ValueError(
            f"Unsupported --codex-sandbox-mode: {config.codex_sandbox_mode}. "
            f"Expected one of: {supported}"
        )
    if config.claude_max_turns is not None and config.claude_max_turns <= 0:
        raise ValueError("--claude-max-turns must be greater than 0.")
    if config.claude_execution_mode not in SUPPORTED_CLAUDE_EXECUTION_MODES:
        supported = ", ".join(sorted(SUPPORTED_CLAUDE_EXECUTION_MODES))
        raise ValueError(
            "Unsupported --claude-execution-mode: "
            f"{config.claude_execution_mode}. Expected one of: {supported}."
        )
    if (
        config.claude_execution_mode == CLAUDE_EXECUTION_CLI
        and config.claude_max_turns is not None
    ):
        raise ValueError(
            "--claude-max-turns is not supported with "
            "--claude-execution-mode=cli; the claude CLI print mode has no "
            "max-turns equivalent."
        )
    if config.child_timeout_seconds < 0:
        raise ValueError("--child-timeout must be greater than or equal to 0.")
    if config.workbench_timeout_seconds <= 0:
        raise ValueError("--workbench-timeout must be greater than 0.")
    if config.vqa_refinement_max_iterations < 0:
        raise ValueError(
            "--visual-validation-max-iterations must be greater than or equal to 0."
        )
    if config.simulation_engine not in {"ovphysx", "fake", "none"}:
        raise ValueError("--simulation-engine must be one of: ovphysx, fake, none.")
    if config.simulation_duration_s <= 0:
        raise ValueError("--duration-s must be greater than 0.")
    if config.simulation_dt <= 0:
        raise ValueError("--dt must be greater than 0.")
    if config.simulation_sample_fps <= 0:
        raise ValueError("--sample-fps must be greater than 0.")
    if (
        config.runner == RUNNER_CLAUDE
        and config.claude_permission_mode == "bypassPermissions"
    ):
        logger.warning(
            "--claude-permission-mode=bypassPermissions disables Claude SDK "
            "permission prompts; use only in a trusted local workspace."
        )


def _trace_path_summary(trace_paths: dict[str, object]) -> dict[str, str]:
    return {
        key: value
        for key, value in trace_paths.items()
        if isinstance(value, str) and key.endswith(("_json", "_md"))
    }


def _agentic_workflow_pythonpath_roots(repo_root: Path) -> list[Path]:
    roots = [
        repo_root / "agentic" / "packages" / "content_workflow_cli",
        repo_root / "agentic" / "packages" / "content_workbench",
        repo_root / "agentic" / "packages" / "content_workbench_agent_client",
        repo_root / "agentic" / "packages" / "content_agent_workflows",
        repo_root / "packages" / "content_workflow_cli",
        repo_root / "packages" / "content_workbench",
        repo_root / "packages" / "content_workbench_agent_client",
        repo_root / "packages" / "content_agent_workflows",
        repo_root / "apps" / "content_workflow_cli",
    ]
    if repo_root.name == "agentic":
        parent = repo_root.parent
        roots.extend(
            [
                parent / "apps" / "content_workflow_cli",
            ]
        )
    return _existing_pythonpath_roots(roots)


def _content_workbench_pythonpath_roots(repo_root: Path) -> list[Path]:
    roots = [
        repo_root / "agentic" / "packages" / "content_workbench",
        repo_root / "packages" / "content_workbench",
    ]
    return _existing_pythonpath_roots(roots)


def _existing_pythonpath_roots(paths: list[Path]) -> list[Path]:
    return list(dict.fromkeys(path.resolve() for path in paths if path.exists()))


def _prepend_pythonpath(existing: str | None, *paths: Path) -> str:
    parts = [str(path) for path in paths]
    if existing:
        parts.append(existing)
    return os.pathsep.join(parts)


def _append_env_paths(existing: str | None, paths: list[Path]) -> str:
    parts = [item.strip() for item in (existing or "").split(",") if item.strip()]
    parts.extend(str(path.resolve()) for path in paths)
    return ",".join(list(dict.fromkeys(parts)))


def _fallback_success_enabled() -> bool:
    value = os.environ.get(ALLOW_FALLBACK_SUCCESS_ENV)
    if value is not None:
        return _parse_bool_env(
            ALLOW_FALLBACK_SUCCESS_ENV,
            value,
            true_values=ENV_TRUE_VALUES,
            false_values=ENV_FALSE_VALUES,
        )

    legacy_disable = os.environ.get(DISABLE_FALLBACK_SUCCESS_ENV)
    if legacy_disable is not None and legacy_disable.strip() != "":
        return _parse_bool_env(
            DISABLE_FALLBACK_SUCCESS_ENV,
            legacy_disable,
            true_values=ENV_FALSE_VALUES,
            false_values=ENV_TRUE_VALUES,
        )

    return False


def _parse_bool_env(
    name: str,
    raw_value: str,
    *,
    true_values: set[str],
    false_values: set[str],
) -> bool:
    value = raw_value.strip().lower()
    if value == "":
        return False
    if value in true_values:
        return True
    if value in false_values:
        return False
    logger.warning(
        "Ignoring unrecognized %s value %r; expected one of: %s",
        name,
        raw_value,
        ", ".join(sorted(true_values | false_values)),
    )
    return False


def _slug(value: str) -> str:
    chars = []
    for char in value.lower():
        if char.isalnum():
            chars.append(char)
        elif chars and chars[-1] != "-":
            chars.append("-")
    slug = "".join(chars).strip("-")
    return slug or "asset"


def _append_child_runner_error(
    path: Path,
    error: Exception,
    *,
    run_dir: Path,
) -> None:
    append_run_text(
        run_dir,
        path,
        "content-workflow-cli runner failed before child completion\n"
        f"{type(error).__name__}: {error}\n",
    )


def _snapshot_material_step_artifacts(
    *,
    run_dir: Path,
    trace_writer: TraceWriter,
    step_id: str,
    step_role: str,
    iteration: int,
    prompt_path: Path,
    child_output_path: Path,
    child_final_path: Path,
    bridge_artifact_prefix: str,
    summary: str,
) -> Path | None:
    """Preserve the current canonical material artifacts as a durable step."""

    required = [
        run_dir / "assignments.json",
        run_dir / "api_operation_counts.json",
        run_dir / "visual_quality_assessment.json",
        run_dir / "final_summary.md",
    ]
    if not all(path.exists() for path in required):
        return None

    steps_dir = run_dir / "steps"
    step_dir = steps_dir / step_id
    if step_dir.exists():
        shutil.rmtree(step_dir)
    for subdir in ("canonical", "raw", "renders", "agent", "evidence"):
        (step_dir / subdir).mkdir(parents=True, exist_ok=True)

    copied: dict[str, Any] = {
        "canonical": {},
        "raw": {},
        "renders": [],
        "agent": {},
        "evidence": [],
    }

    for name, source in {
        "assignments": run_dir / "assignments.json",
        "api_operation_counts": run_dir / "api_operation_counts.json",
        "visual_quality_assessment": run_dir / "visual_quality_assessment.json",
        "final_summary": run_dir / "final_summary.md",
    }.items():
        dest = _copy_step_artifact(source, step_dir / "canonical" / source.name)
        if dest is not None:
            copied["canonical"][name] = _run_relative_path(run_dir, dest)

    raw_artifacts = {
        "material_decision_patch": run_dir / "raw" / "material_decision_patch.json",
        "rejected_material_assignments": (
            run_dir / "raw" / "rejected_material_assignments.json"
        ),
        "final_render_records": run_dir / "raw" / "final_render_records.json",
    }
    if iteration == 1:
        raw_artifacts["material_override_summary"] = (
            run_dir / "raw" / "material_override_summary_iter1.json"
        )
    else:
        for source in sorted(
            (run_dir / "raw").glob(f"vqa_refinement_iter{iteration}_*.json")
        ):
            raw_artifacts[source.stem] = source
    for name, source in raw_artifacts.items():
        dest = _copy_step_artifact(source, step_dir / "raw" / source.name)
        if dest is not None:
            copied["raw"][name] = _run_relative_path(run_dir, dest)

    for source in sorted((run_dir / "final_renders").glob("*")):
        if not source.is_file():
            continue
        dest = _copy_step_artifact(source, step_dir / "renders" / source.name)
        if dest is not None:
            copied["renders"].append(_run_relative_path(run_dir, dest))

    agent_artifacts = {
        "prompt": prompt_path,
        "child_output": child_output_path,
        "child_final": child_final_path,
        "runner_request": run_dir / "raw" / f"{bridge_artifact_prefix}_request.json",
        "runner_items": run_dir / "raw" / f"{bridge_artifact_prefix}_items.json",
        "runner_result": run_dir / "raw" / f"{bridge_artifact_prefix}_result.json",
    }
    for name, source in agent_artifacts.items():
        dest = _copy_step_artifact(source, step_dir / "agent" / source.name)
        if dest is not None:
            copied["agent"][name] = _run_relative_path(run_dir, dest)

    for source in _step_evidence_sources(run_dir, iteration):
        dest = _copy_step_artifact(source, step_dir / "evidence" / source.name)
        if dest is not None:
            copied["evidence"].append(_run_relative_path(run_dir, dest))

    assignments = _load_json(run_dir / "assignments.json", default={})
    visual_quality = _load_json(run_dir / "visual_quality_assessment.json", default={})
    counts = _load_json(run_dir / "api_operation_counts.json", default={})
    coverage = assignments.get("coverage") if isinstance(assignments, dict) else {}
    if not isinstance(coverage, dict):
        coverage = {}
    if not isinstance(visual_quality, dict):
        visual_quality = {}
    if not isinstance(counts, dict):
        counts = {}
    assignment_groups = (
        assignments.get("assignments") if isinstance(assignments, dict) else []
    )
    if not isinstance(assignment_groups, list):
        assignment_groups = []
    rejected_groups = _load_json(
        run_dir / "raw" / "rejected_material_assignments.json",
        default=[],
    )
    if not isinstance(rejected_groups, list):
        rejected_groups = []
    material_assignment_groups = [
        group
        for group in assignment_groups
        if isinstance(group, dict)
        and _is_material_assignment_status(group.get("coverage_status"))
    ]
    material_assignment_group_count: object = len(material_assignment_groups)
    material_assignment_target_prims: object = sum(
        len(_string_list(group.get("prim_paths")))
        for group in material_assignment_groups
    )
    if not material_assignment_groups:
        material_assignment_group_count = counts.get("material_assignment_groups")
        material_assignment_target_prims = counts.get(
            "material_assignment_target_prims"
        )

    step_record = {
        "schema_version": STEP_ARTIFACT_SCHEMA_VERSION,
        "step_id": step_id,
        "step_role": step_role,
        "iteration": iteration,
        "created_at": utc_now(),
        "summary": summary,
        "run_dir": str(run_dir),
        "artifacts": copied,
        "coverage": {
            "candidate_visible_prim_count": coverage.get(
                "candidate_visible_prim_count"
            ),
            "material_decision_prim_count": coverage.get(
                "material_decision_prim_count"
            ),
            "material_assignment_prim_count": coverage.get(
                "material_assignment_prim_count"
            ),
            "ambiguous_unassigned_prim_count": coverage.get(
                "ambiguous_unassigned_prim_count"
            ),
            "unassigned_visible_prim_count": coverage.get(
                "unassigned_visible_prim_count"
            ),
            "missing_assignment_prim_count": coverage.get(
                "missing_assignment_prim_count"
            ),
            "rejected_assignment_prim_count": coverage.get(
                "rejected_assignment_prim_count"
            ),
        },
        "visual_quality_status": visual_quality.get("status"),
        "visual_quality_unresolved_issues": _string_list(
            visual_quality.get("unresolved_issues")
        ),
        "material_assignment_groups": material_assignment_group_count,
        "material_assignment_target_prims": material_assignment_target_prims,
        "context": _material_step_context_metadata(
            run_dir=run_dir,
            assignments=assignments if isinstance(assignments, dict) else {},
        ),
        "decision_group_summaries": _material_step_group_summaries(
            assignment_groups,
            row_kind="assignment",
        ),
        "rejected_assignment_count": len(
            [group for group in rejected_groups if isinstance(group, dict)]
        ),
        "rejected_assignment_summaries": _material_step_group_summaries(
            rejected_groups,
            row_kind="rejected_assignment",
        ),
    }
    step_json = step_dir / "step.json"
    step_json.write_text(json.dumps(step_record, indent=2), encoding="utf-8")

    manifest_path = _upsert_step_manifest(
        run_dir=run_dir,
        step_record=step_record,
        step_json=step_json,
    )
    trace_writer.write(
        "workflow_step_artifacts_captured",
        phase="artifact snapshot",
        summary=summary,
        artifacts=[str(step_json), str(manifest_path)],
        data={
            "step_id": step_id,
            "iteration": iteration,
            "step_role": step_role,
            "render_artifacts": len(copied["renders"]),
            "canonical_artifacts": len(copied["canonical"]),
        },
    )
    return step_dir


def _material_step_context_metadata(
    *,
    run_dir: Path,
    assignments: dict[str, Any],
) -> dict[str, Any]:
    raw_dir = run_dir / "raw"
    packet = _load_json(raw_dir / "material_run_packet.json", default={})
    if not isinstance(packet, dict):
        packet = {}
    request = _load_json(run_dir / "request.json", default={})
    if not isinstance(request, dict):
        request = {}
    inputs = request.get("inputs")
    if not isinstance(inputs, dict):
        inputs = {}
    palette = _load_json(raw_dir / "material_palette.json", default={})
    if not isinstance(palette, dict):
        palette = {}
    materials = palette.get("materials")
    if not isinstance(materials, list):
        materials = []
    candidates = _load_json(raw_dir / "visible_candidate_prims.json", default={})
    if not isinstance(candidates, dict):
        candidates = {}
    candidate_rows = candidates.get("candidates")
    if not isinstance(candidate_rows, list):
        candidate_rows = []
    session = packet.get("session")
    if not isinstance(session, dict):
        session = {}
    docs = packet.get("docs")
    if not isinstance(docs, dict):
        docs = {}
    coverage = assignments.get("coverage")
    if not isinstance(coverage, dict):
        coverage = {}
    return {
        "source_usd": (
            assignments.get("source_usd")
            or packet.get("source_usd")
            or inputs.get("usd")
        ),
        "inspection_usd": assignments.get("inspection_usd"),
        "session_id": assignments.get("session_id") or packet.get("session_id"),
        "references": {
            "images": _string_list(inputs.get("reference_images")),
            "files": _string_list(inputs.get("reference_files")),
            "docs": docs,
        },
        "materials": {
            "materials_yaml": packet.get("materials_yaml")
            or inputs.get("materials_yaml"),
            "materials_usd": (
                assignments.get("library_path")
                or packet.get("materials_usd")
                or inputs.get("materials_usd")
            ),
            "palette_path": _run_relative_path(
                run_dir, raw_dir / "material_palette.json"
            ),
            "palette_sha256": _file_sha256(raw_dir / "material_palette.json"),
            "palette_material_count": len(materials),
        },
        "clean_slate": {
            "clear_materials": packet.get(
                "clear_materials",
                request.get("clear_materials"),
            ),
            "respect_existing_material_bindings": packet.get(
                "respect_existing_material_bindings",
                request.get("respect_existing_material_bindings"),
            ),
        },
        "optimizer": {
            "optimize": packet.get("optimize", request.get("workbench_optimize")),
            "optimization_artifact": session.get("optimization"),
        },
        "candidate_universe": {
            "path": _run_relative_path(
                run_dir, raw_dir / "visible_candidate_prims.json"
            ),
            "path_space": assignments.get("path_space") or candidates.get("path_space"),
            "candidate_count": len(candidate_rows),
            "candidate_visible_prim_count": coverage.get(
                "candidate_visible_prim_count"
            ),
        },
    }


def _material_step_group_summaries(
    groups: list[Any],
    *,
    row_kind: str,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for index, group in enumerate(groups):
        if not isinstance(group, dict):
            continue
        prim_paths = _string_list(group.get("prim_paths"))
        runtime_paths = _string_list(group.get("runtime_prim_paths"))
        source_paths = _string_list(group.get("source_prim_paths"))
        summaries.append(
            {
                "row_kind": row_kind,
                "group_index": index,
                "coverage_status": group.get("coverage_status"),
                "family": group.get("family"),
                "material_name": group.get("material_name"),
                "material_path": group.get("material_path"),
                "path_space": group.get("path_space"),
                "runtime_space": group.get("runtime_space"),
                "prim_count": len(prim_paths),
                "runtime_prim_count": len(runtime_paths),
                "source_prim_count": len(source_paths),
                "prim_paths": prim_paths,
                "runtime_prim_paths": runtime_paths,
                "source_prim_paths": source_paths,
                "rationale": group.get("rationale"),
                "rejection_reason": group.get("rejection_reason"),
            }
        )
    return summaries


def _file_sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_step_artifact(source: Path, dest: Path) -> Path | None:
    if not source.exists() or not source.is_file():
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    return dest


def _step_evidence_sources(run_dir: Path, iteration: int) -> list[Path]:
    if iteration == 1:
        patterns = [
            "initial_*.png",
            "child_iter1_*.png",
        ]
    else:
        patterns = [
            f"*iter{iteration}*.png",
            f"vqa_refinement_iter{iteration}_*.png",
        ]
    sources: list[Path] = []
    seen: set[Path] = set()
    for base in (run_dir / "evidence_renders", run_dir / "raw"):
        for pattern in patterns:
            for path in sorted(base.glob(pattern)):
                if path.is_file() and path not in seen:
                    seen.add(path)
                    sources.append(path)
    return sources


def _upsert_step_manifest(
    *,
    run_dir: Path,
    step_record: dict[str, Any],
    step_json: Path,
) -> Path:
    steps_dir = run_dir / "steps"
    manifest_path = steps_dir / "manifest.json"
    manifest = _load_json(manifest_path, default={})
    if not isinstance(manifest, dict):
        manifest = {}
    steps = manifest.get("steps")
    if not isinstance(steps, list):
        steps = []

    step_id = str(step_record["step_id"])
    entry = {
        "step_id": step_id,
        "step_role": step_record["step_role"],
        "iteration": step_record["iteration"],
        "created_at": step_record["created_at"],
        "summary": step_record["summary"],
        "step_json": _run_relative_path(run_dir, step_json),
        "step_dir": _run_relative_path(run_dir, step_json.parent),
        "coverage": step_record["coverage"],
        "visual_quality_status": step_record.get("visual_quality_status"),
        "visual_quality_unresolved_issue_count": len(
            step_record.get("visual_quality_unresolved_issues") or []
        ),
        "material_assignment_groups": step_record.get("material_assignment_groups"),
        "material_assignment_target_prims": step_record.get(
            "material_assignment_target_prims"
        ),
        "rejected_assignment_count": step_record.get("rejected_assignment_count"),
        "context": _manifest_step_context_summary(step_record.get("context")),
        "renders": step_record["artifacts"].get("renders", []),
    }
    steps = [
        item
        for item in steps
        if not isinstance(item, dict) or item.get("step_id") != step_id
    ]
    steps.append(entry)
    steps.sort(
        key=lambda item: (
            int(item.get("iteration") or 0) if isinstance(item, dict) else 0
        )
    )

    manifest = {
        "schema_version": STEP_ARTIFACT_MANIFEST_SCHEMA_VERSION,
        "run_dir": str(run_dir),
        "updated_at": utc_now(),
        "latest_step_id": step_id,
        "steps": steps,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def _manifest_step_context_summary(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    materials = value.get("materials")
    if not isinstance(materials, dict):
        materials = {}
    clean_slate = value.get("clean_slate")
    if not isinstance(clean_slate, dict):
        clean_slate = {}
    optimizer = value.get("optimizer")
    if not isinstance(optimizer, dict):
        optimizer = {}
    candidate_universe = value.get("candidate_universe")
    if not isinstance(candidate_universe, dict):
        candidate_universe = {}
    references = value.get("references")
    if not isinstance(references, dict):
        references = {}
    return {
        "source_usd": value.get("source_usd"),
        "inspection_usd": value.get("inspection_usd"),
        "session_id": value.get("session_id"),
        "materials_usd": materials.get("materials_usd"),
        "palette_sha256": materials.get("palette_sha256"),
        "reference_image_count": len(_string_list(references.get("images"))),
        "reference_file_count": len(_string_list(references.get("files"))),
        "clear_materials": clean_slate.get("clear_materials"),
        "respect_existing_material_bindings": clean_slate.get(
            "respect_existing_material_bindings"
        ),
        "optimize": optimizer.get("optimize"),
        "candidate_path_space": candidate_universe.get("path_space"),
        "candidate_count": candidate_universe.get("candidate_count"),
    }


def _record_incomplete_material_step(
    *,
    run_dir: Path,
    trace_writer: TraceWriter,
    step_id: str,
    step_role: str,
    iteration: int,
    prompt_path: Path,
    child_output_path: Path,
    child_final_path: Path,
    bridge_artifact_prefix: str,
    summary: str,
    reason: str,
    error: str | None = None,
) -> Path:
    steps_dir = run_dir / "steps"
    steps_dir.mkdir(parents=True, exist_ok=True)
    incomplete_path = steps_dir / "incomplete_steps.jsonl"
    raw_dir = run_dir / "raw"
    artifacts = {
        "prompt": _run_relative_path(run_dir, prompt_path),
        "child_output": _run_relative_path(run_dir, child_output_path),
        "child_final": _run_relative_path(run_dir, child_final_path),
        "runner_request": _run_relative_path(
            run_dir,
            raw_dir / f"{bridge_artifact_prefix}_request.json",
        ),
        "runner_items": _run_relative_path(
            run_dir,
            raw_dir / f"{bridge_artifact_prefix}_items.json",
        ),
        "runner_result": _run_relative_path(
            run_dir,
            raw_dir / f"{bridge_artifact_prefix}_result.json",
        ),
        "material_decision_patch": _run_relative_path(
            run_dir,
            raw_dir / "material_decision_patch.json",
        ),
        "finalizer_error": _run_relative_path(
            run_dir,
            raw_dir / "material_decision_finalizer_error.json",
        ),
    }
    record = {
        "schema_version": INCOMPLETE_STEP_SCHEMA_VERSION,
        "status": "incomplete",
        "step_id": step_id,
        "step_role": step_role,
        "iteration": iteration,
        "created_at": utc_now(),
        "summary": summary,
        "reason": reason,
        "error": error,
        "run_dir": str(run_dir),
        "artifacts": artifacts,
    }
    records = [
        row for row in _load_jsonl(incomplete_path) if row.get("step_id") != step_id
    ]
    records.append(record)
    records.sort(
        key=lambda row: (
            int(row.get("iteration") or 0),
            str(row.get("step_id") or ""),
        )
    )
    incomplete_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
        encoding="utf-8",
    )
    manifest_path = _upsert_incomplete_step_manifest(
        run_dir=run_dir,
        incomplete_record=record,
        incomplete_path=incomplete_path,
    )
    trace_writer.write(
        "workflow_step_incomplete",
        phase="artifact snapshot",
        summary=summary,
        artifacts=[str(incomplete_path), str(manifest_path)],
        data={
            "step_id": step_id,
            "iteration": iteration,
            "step_role": step_role,
            "reason": reason,
            "error": error,
        },
    )
    return incomplete_path


def _upsert_incomplete_step_manifest(
    *,
    run_dir: Path,
    incomplete_record: dict[str, Any],
    incomplete_path: Path,
) -> Path:
    steps_dir = run_dir / "steps"
    manifest_path = steps_dir / "manifest.json"
    manifest = _load_json(manifest_path, default={})
    if not isinstance(manifest, dict):
        manifest = {}
    incomplete_steps = manifest.get("incomplete_steps")
    if not isinstance(incomplete_steps, list):
        incomplete_steps = []
    step_id = str(incomplete_record["step_id"])
    entry = {
        "step_id": step_id,
        "step_role": incomplete_record["step_role"],
        "iteration": incomplete_record["iteration"],
        "created_at": incomplete_record["created_at"],
        "summary": incomplete_record["summary"],
        "status": incomplete_record["status"],
        "reason": incomplete_record["reason"],
        "error": incomplete_record.get("error"),
        "incomplete_steps": _run_relative_path(run_dir, incomplete_path),
    }
    incomplete_steps = [
        item
        for item in incomplete_steps
        if not isinstance(item, dict) or item.get("step_id") != step_id
    ]
    incomplete_steps.append(entry)
    incomplete_steps.sort(
        key=lambda item: (
            int(item.get("iteration") or 0) if isinstance(item, dict) else 0
        )
    )
    manifest.setdefault("schema_version", STEP_ARTIFACT_MANIFEST_SCHEMA_VERSION)
    manifest.setdefault("run_dir", str(run_dir))
    manifest["updated_at"] = utc_now()
    manifest["incomplete_steps_path"] = _run_relative_path(run_dir, incomplete_path)
    manifest["incomplete_steps"] = incomplete_steps
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def _run_relative_path(run_dir: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(run_dir.resolve()))
    except ValueError:
        return str(path)


def _run_vqa_refinement_loop(
    *,
    config: MaterialAssignConfig,
    run_dir: Path,
    request: dict[str, object],
    preflight_packet: dict[str, Any] | None,
    trace_writer: TraceWriter,
    managed_workbench: ManagedWorkbench | None,
    initial_child_output_path: Path,
    initial_child_final_path: Path,
    prompt_image_inputs: list[dict[str, str]],
    codex_session: CodexThreadSession | None = None,
) -> int:
    max_iterations = config.vqa_refinement_max_iterations
    history_path = run_dir / "raw" / "vqa_refinement_history.json"
    session_id = _preflight_session_id(preflight_packet)
    assessment = _current_vqa_refinement_assessment(run_dir)
    history = _load_vqa_refinement_history(history_path)
    history.update(
        {
            "schema_version": "content-agents.vqa-refinement-history.v1",
            "run_dir": str(run_dir),
            "max_iterations": max_iterations,
        }
    )
    history.setdefault(
        "initial_child_artifacts",
        {
            "child_output": str(initial_child_output_path),
            "child_final": str(initial_child_final_path),
        },
    )
    history["initial_assessment"] = assessment

    if _vqa_refinement_satisfied(assessment):
        history["status"] = "satisfied_initial"
        history["stop_reason"] = "initial_canonical_artifacts_satisfied_vqa_gate"
        _write_private_json(history_path, history)
        trace_writer.write(
            "vqa_refinement_finished",
            phase="vqa refinement",
            summary="Skipped VQA refinement because canonical artifacts already satisfy the VQA gate.",
            artifacts=[str(history_path)],
            data={"max_iterations": max_iterations},
        )
        return 0

    if _vqa_refinement_systematic_unfixable(assessment):
        history["status"] = "systematic_unfixable_initial"
        history["stop_reason"] = (
            "VQA refinement skipped because all initial remaining active "
            "issues are recorded as material-library, Workbench picking, "
            "or prim-granularity limitations."
        )
        _write_private_json(history_path, history)
        trace_writer.write(
            "warning",
            phase="vqa refinement",
            summary=history["stop_reason"],
            artifacts=[str(history_path)],
            data={
                "max_iterations": max_iterations,
                "issue_signature": assessment.get("signature"),
                "active_issues": assessment.get("active_issues"),
            },
        )
        return 0

    if max_iterations <= 1:
        history["status"] = "max_iterations_reached"
        history["stop_reason"] = (
            "VQA issues remain after the initial review, but no additional "
            "refinement iterations are configured."
        )
        _write_private_json(history_path, history)
        trace_writer.write(
            "warning",
            phase="vqa refinement",
            summary=history["stop_reason"],
            artifacts=[str(history_path)],
            data={
                "max_iterations": max_iterations,
                "issue_signature": assessment.get("signature"),
            },
        )
        return 0

    if not session_id:
        history["status"] = "skipped_no_session"
        history["stop_reason"] = (
            "VQA issues remain, but wrapper-owned refinement requires a "
            "preflight Workbench session."
        )
        _write_private_json(history_path, history)
        trace_writer.write(
            "warning",
            phase="vqa refinement",
            summary=history["stop_reason"],
            artifacts=[str(history_path)],
            data={"preflight_enabled": preflight_packet is not None},
        )
        return 0

    previous_signature = assessment.get("signature")
    for iteration in range(2, max_iterations + 1):
        prompt_path = run_dir / "raw" / f"vqa_refinement_prompt_{iteration}.md"
        child_output_path = (
            run_dir / "raw" / f"vqa_refinement_{iteration}_child-output.log"
        )
        child_final_path = (
            run_dir / "raw" / f"vqa_refinement_{iteration}_child-final.md"
        )
        artifact_index_path = _write_vqa_refinement_artifact_index(
            run_dir=run_dir,
            iteration=iteration,
            assessment=assessment,
        )
        issue_packet_path = _write_vqa_refinement_issue_packet(
            run_dir=run_dir,
            iteration=iteration,
            assessment=assessment,
            history=history,
            artifact_index_path=artifact_index_path,
        )
        prompt = build_material_refinement_prompt(
            run_dir=run_dir,
            usd_path=config.usd_path,
            reference_images=config.reference_images,
            reference_files=config.reference_files or [],
            materials_yaml=config.materials_yaml,
            materials_usd=config.materials_usd,
            workbench_url=config.workbench_url,
            session_id=session_id,
            iteration=iteration,
            max_iterations=max_iterations,
            issue_summary=assessment,
            history_path=history_path,
            artifact_index_path=artifact_index_path,
            issue_packet_path=issue_packet_path,
            repair_attempt_ledger=_vqa_refinement_attempt_ledger(history),
            previous_child_artifacts=_vqa_refinement_child_artifacts(
                history,
                initial_child_output_path=initial_child_output_path,
                initial_child_final_path=initial_child_final_path,
            ),
            optimize=config.optimize,
            respect_existing_material_bindings=(
                config.respect_existing_material_bindings
            ),
            additional_instructions=config.additional_instructions,
        )
        iteration_started_at = utc_now()
        prompt_path.write_text(prompt, encoding="utf-8")
        trace_writer.write(
            "vqa_refinement_started",
            phase="vqa refinement",
            summary=(
                f"Started VQA refinement iteration {iteration}/{max_iterations} "
                "against canonical unresolved issues."
            ),
            artifacts=[str(prompt_path), str(history_path)],
            data={
                "iteration": iteration,
                "max_iterations": max_iterations,
                "issue_signature": previous_signature,
                "artifact_index": str(artifact_index_path),
                "issue_packet": str(issue_packet_path),
            },
        )

        try:
            child_returncode = _run_child_agent(
                config=config,
                prompt=prompt,
                run_dir=run_dir,
                child_output_path=child_output_path,
                child_final_path=child_final_path,
                managed_workbench=managed_workbench,
                prompt_image_inputs=_vqa_refinement_image_inputs(
                    run_dir,
                    prompt_image_inputs,
                ),
                bridge_artifact_prefix=f"vqa_refinement_{iteration}",
                codex_session=codex_session,
            )
        except UnsafeRunArtifactError:
            raise
        except Exception as exc:  # noqa: BLE001 - preserve refinement history
            child_returncode = 2
            _append_child_runner_error(child_output_path, exc, run_dir=run_dir)
            trace_writer.write(
                "child_agent_failed",
                phase="vqa refinement",
                summary="VQA refinement child agent runner failed before completion.",
                artifacts=[str(child_output_path), str(child_final_path)],
                data={
                    "iteration": iteration,
                    "returncode": child_returncode,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )

        finalizer_error: str | None = None
        if child_returncode == 0:
            try:
                _finalize_structured_material_decisions(
                    config=config,
                    run_dir=run_dir,
                    preflight_packet=preflight_packet,
                    trace_writer=trace_writer,
                )
            except UnsafeRunArtifactError:
                raise
            except Exception as exc:  # noqa: BLE001 - keep history inspectable
                child_returncode = 2
                finalizer_error = f"{type(exc).__name__}: {exc}"
                trace_writer.write(
                    "warning",
                    phase="vqa refinement",
                    summary="Deterministic finalizer failed after a VQA refinement turn.",
                    artifacts=[
                        str(run_dir / "raw" / "material_decision_finalizer_error.json")
                    ],
                    data={
                        "iteration": iteration,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )

        if finalizer_error:
            incomplete_path = _record_incomplete_material_step(
                run_dir=run_dir,
                trace_writer=trace_writer,
                step_id=f"{iteration:02d}_vqa_refinement_iter{iteration}",
                step_role="vqa_refinement",
                iteration=iteration,
                prompt_path=prompt_path,
                child_output_path=child_output_path,
                child_final_path=child_final_path,
                bridge_artifact_prefix=f"vqa_refinement_{iteration}",
                summary=(
                    f"VQA refinement iteration {iteration} wrote child artifacts, "
                    "but deterministic finalization failed."
                ),
                reason="deterministic_finalization_failed",
                error=finalizer_error,
            )
            attempt_record = {
                "iteration": iteration,
                "started_at": iteration_started_at,
                "completed_at": utc_now(),
                "child_returncode": child_returncode,
                "prompt": str(prompt_path),
                "child_output": str(child_output_path),
                "child_final": str(child_final_path),
                "before": assessment,
                "after": assessment,
                "converged": True,
                "finalizer_error": finalizer_error,
                "incomplete_step": str(incomplete_path),
            }
            history.setdefault("iterations", []).append(attempt_record)
            history["status"] = "finalizer_failed"
            history["stop_reason"] = (
                f"VQA refinement iteration {iteration} could not be finalized: "
                f"{finalizer_error}"
            )
            _write_private_json(history_path, history)
            trace_writer.write(
                "warning",
                phase="vqa refinement",
                summary=history["stop_reason"],
                artifacts=[str(history_path), str(incomplete_path)],
                data={"iteration": iteration, "returncode": child_returncode},
            )
            return child_returncode

        _ensure_material_assignment_artifacts(
            config=config,
            run_dir=run_dir,
            request=request,
            trace_writer=trace_writer,
            child_output_path=initial_child_output_path,
            child_final_path=initial_child_final_path,
            child_returncode=child_returncode,
        )
        _snapshot_material_step_artifacts(
            run_dir=run_dir,
            trace_writer=trace_writer,
            step_id=f"{iteration:02d}_vqa_refinement_iter{iteration}",
            step_role="vqa_refinement",
            iteration=iteration,
            prompt_path=prompt_path,
            child_output_path=child_output_path,
            child_final_path=child_final_path,
            bridge_artifact_prefix=f"vqa_refinement_{iteration}",
            summary=(
                f"Captured canonical artifacts after VQA refinement iteration "
                f"{iteration} and deterministic finalization."
            ),
        )
        next_assessment = _current_vqa_refinement_assessment(run_dir)
        next_signature = next_assessment.get("signature")
        converged = next_signature == previous_signature
        attempt_record = {
            "iteration": iteration,
            "started_at": iteration_started_at,
            "completed_at": utc_now(),
            "child_returncode": child_returncode,
            "prompt": str(prompt_path),
            "child_output": str(child_output_path),
            "child_final": str(child_final_path),
            "before": assessment,
            "after": next_assessment,
            "converged": converged,
        }
        if finalizer_error:
            attempt_record["finalizer_error"] = finalizer_error
        history.setdefault("iterations", []).append(attempt_record)

        if child_returncode != 0:
            history["status"] = "child_failed"
            history["stop_reason"] = (
                f"VQA refinement iteration {iteration} exited with return code "
                f"{child_returncode}."
            )
            _write_private_json(history_path, history)
            trace_writer.write(
                "warning",
                phase="vqa refinement",
                summary=history["stop_reason"],
                artifacts=[str(history_path), str(child_output_path)],
                data={"iteration": iteration, "returncode": child_returncode},
            )
            return child_returncode

        if _vqa_refinement_satisfied(next_assessment):
            history["status"] = "satisfied"
            history["stop_reason"] = (
                f"VQA refinement iteration {iteration} satisfied the VQA gate."
            )
            _write_private_json(history_path, history)
            trace_writer.write(
                "vqa_refinement_finished",
                phase="vqa refinement",
                summary=history["stop_reason"],
                artifacts=[str(history_path), str(child_final_path)],
                data={"iteration": iteration, "max_iterations": max_iterations},
            )
            return 0

        if _vqa_refinement_systematic_unfixable(next_assessment):
            history["status"] = "systematic_unfixable"
            history["stop_reason"] = (
                f"VQA refinement stopped after iteration {iteration}; all "
                "remaining active issues are recorded as material-library, "
                "Workbench picking, or prim-granularity limitations."
            )
            _write_private_json(history_path, history)
            trace_writer.write(
                "warning",
                phase="vqa refinement",
                summary=history["stop_reason"],
                artifacts=[str(history_path), str(child_final_path)],
                data={
                    "iteration": iteration,
                    "max_iterations": max_iterations,
                    "issue_signature": next_signature,
                    "active_issues": next_assessment.get("active_issues"),
                },
            )
            return 0

        if converged:
            history["status"] = "converged_unresolved"
            history["stop_reason"] = (
                f"VQA refinement converged after iteration {iteration}; "
                "canonical issue signature did not change."
            )
            _write_private_json(history_path, history)
            trace_writer.write(
                "warning",
                phase="vqa refinement",
                summary=history["stop_reason"],
                artifacts=[str(history_path), str(child_final_path)],
                data={
                    "iteration": iteration,
                    "max_iterations": max_iterations,
                    "issue_signature": next_signature,
                },
            )
            return 0

        assessment = next_assessment
        previous_signature = next_signature
        _write_private_json(history_path, history)

    history["status"] = "max_iterations_reached"
    history["stop_reason"] = (
        f"VQA refinement reached the configured maximum of {max_iterations} "
        "iterations with unresolved issues still present."
    )
    _write_private_json(history_path, history)
    trace_writer.write(
        "warning",
        phase="vqa refinement",
        summary=history["stop_reason"],
        artifacts=[str(history_path)],
        data={
            "max_iterations": max_iterations,
            "issue_signature": previous_signature,
        },
    )
    return 0


def _write_vqa_refinement_artifact_index(
    *,
    run_dir: Path,
    iteration: int,
    assessment: dict[str, object],
) -> Path:
    raw_dir = run_dir / "raw"
    path = raw_dir / f"vqa_refinement_artifact_index_{iteration}.json"
    final_renders = {
        render_path.stem: _run_relative_path(run_dir, render_path)
        for render_path in sorted((run_dir / "final_renders").glob("*.png"))
    }
    steps_manifest = run_dir / "steps" / "manifest.json"
    index = {
        "schema_version": "content-agents.vqa-refinement-artifact-index.v1",
        "iteration": iteration,
        "run_dir": str(run_dir),
        "active_issue_ids": [
            str(item.get("id"))
            for item in assessment.get("active_issues", [])
            if isinstance(item, dict) and item.get("id")
        ],
        "current_views": final_renders,
        "canonical_artifacts": {
            "assignments": _run_relative_path(run_dir, run_dir / "assignments.json"),
            "visual_quality_assessment": _run_relative_path(
                run_dir, run_dir / "visual_quality_assessment.json"
            ),
            "material_decision_patch": _run_relative_path(
                run_dir, raw_dir / "material_decision_patch.json"
            ),
            "rejected_material_assignments": _run_relative_path(
                run_dir, raw_dir / "rejected_material_assignments.json"
            ),
            "material_palette": _run_relative_path(
                run_dir, raw_dir / "material_palette.json"
            ),
            "final_render_records": _run_relative_path(
                run_dir, raw_dir / "final_render_records.json"
            ),
            "vqa_refinement_history": _run_relative_path(
                run_dir, raw_dir / "vqa_refinement_history.json"
            ),
            "steps_manifest": _run_relative_path(run_dir, steps_manifest)
            if steps_manifest.exists()
            else None,
        },
        "step_dirs": [
            _run_relative_path(run_dir, step_dir)
            for step_dir in sorted((run_dir / "steps").glob("*"))
            if step_dir.is_dir()
        ],
    }
    _write_private_json(path, index)
    return path


def _write_vqa_refinement_issue_packet(
    *,
    run_dir: Path,
    iteration: int,
    assessment: dict[str, object],
    history: dict[str, object],
    artifact_index_path: Path,
) -> Path:
    raw_dir = run_dir / "raw"
    path = raw_dir / f"vqa_refinement_issue_packet_{iteration}.json"
    active_issues = [
        item for item in assessment.get("active_issues", []) if isinstance(item, dict)
    ]
    packet = {
        "schema_version": "content-agents.vqa-refinement-issue-packet.v1",
        "iteration": iteration,
        "status": assessment.get("status"),
        "active_issues": active_issues,
        "coverage": assessment.get("coverage") or {},
        "current_material_decisions": assessment.get("current_material_decisions")
        or [],
        "assessment_notes": assessment.get("assessment_notes") or "",
        "recent_attempts": _vqa_refinement_attempt_ledger(history)[-3:],
        "artifact_index": _run_relative_path(run_dir, artifact_index_path),
    }
    _write_private_json(path, packet)
    return path


def _current_vqa_refinement_assessment(run_dir: Path) -> dict[str, object]:
    assignments = _load_json(run_dir / "assignments.json", default={})
    if not isinstance(assignments, dict):
        assignments = {}
    visual_quality = _load_json(run_dir / "visual_quality_assessment.json", default={})
    if not isinstance(visual_quality, dict):
        embedded = assignments.get("visual_quality_assessment")
        visual_quality = embedded if isinstance(embedded, dict) else {}
    final_review = assignments.get("final_review")
    if not isinstance(final_review, dict):
        final_review = {}
    coverage = assignments.get("coverage")
    if not isinstance(coverage, dict):
        coverage = {}

    status = str(visual_quality.get("status") or "unknown")
    unresolved_fallback = (
        visual_quality.get("issues_found") if status == "unresolved_issues" else None
    )
    unresolved_vqa = _vqa_issue_items(
        visual_quality.get("unresolved_issues"),
        fallback_items=unresolved_fallback,
    )
    final_review_unresolved = _vqa_issue_items(final_review.get("unresolved_issues"))
    coverage_gaps = _coverage_gap_issues(coverage)
    rejected_assignment_issues = _rejected_material_assignment_issues(run_dir)
    if status == "unresolved_issues" and not unresolved_vqa:
        unresolved_vqa.append(
            {
                "description": (
                    "Visual quality status is unresolved_issues but no unresolved "
                    "issue was listed."
                ),
                "affected_prim_paths": [],
            }
        )
    active_issues = _vqa_refinement_active_issues(
        unresolved_vqa=unresolved_vqa,
        final_review_unresolved=final_review_unresolved,
        rejected_assignments=rejected_assignment_issues,
        coverage_gaps=coverage_gaps,
    )
    decision_summary = _decision_patch_repair_summary(run_dir)
    signature_payload = _vqa_refinement_signature_payload(
        status=status,
        active_issues=active_issues,
        coverage=coverage,
        decision_summary=decision_summary,
    )

    issue_summary = {
        "status": status,
        "active_issues": active_issues,
        "unresolved_vqa_issues": [
            str(item.get("description") or "") for item in unresolved_vqa
        ],
        "vqa_issues_found": _refinement_issue_descriptions(
            visual_quality.get("issues_found")
        ),
        "vqa_issues_fixed": _refinement_issue_descriptions(
            visual_quality.get("issues_fixed")
        ),
        "final_review_unresolved_issues": [
            str(item.get("description") or "") for item in final_review_unresolved
        ],
        "rejected_assignment_issues": [
            str(item.get("description") or "") for item in rejected_assignment_issues
        ],
        "coverage_gaps": coverage_gaps,
        "checked_views": _string_list(visual_quality.get("checked_views")),
        "assessment_notes": str(visual_quality.get("assessment_notes") or ""),
        "coverage": {
            "candidate_visible_prim_count": coverage.get(
                "candidate_visible_prim_count"
            ),
            "material_decision_prim_count": coverage.get(
                "material_decision_prim_count"
            ),
            "ambiguous_unassigned_prim_count": coverage.get(
                "ambiguous_unassigned_prim_count"
            ),
            "unassigned_visible_prim_count": coverage.get(
                "unassigned_visible_prim_count"
            ),
            "missing_assignment_prim_count": coverage.get(
                "missing_assignment_prim_count"
            ),
            "rejected_assignment_prim_count": coverage.get(
                "rejected_assignment_prim_count"
            ),
        },
        "current_material_decisions": decision_summary,
    }
    issue_summary["signature"] = _short_json_signature(signature_payload)
    return issue_summary


def _vqa_refinement_satisfied(assessment: dict[str, object]) -> bool:
    status = str(assessment.get("status") or "")
    return (
        status in {"pass", "fixed"}
        and not assessment.get("unresolved_vqa_issues")
        and not assessment.get("final_review_unresolved_issues")
        and not assessment.get("rejected_assignment_issues")
        and not assessment.get("coverage_gaps")
    )


def _coverage_gap_issues(coverage: dict[str, Any]) -> list[str]:
    candidate_count = _int_or_none(coverage.get("candidate_visible_prim_count"))
    decision_count = _int_or_none(coverage.get("material_decision_prim_count"))
    if candidate_count is None or decision_count is None:
        return []
    if decision_count >= candidate_count:
        return []
    missing_count = _int_or_none(coverage.get("missing_assignment_prim_count"))
    rejected_count = _int_or_none(coverage.get("rejected_assignment_prim_count"))
    issues: list[str] = []
    if missing_count and missing_count > 0:
        issues.append(
            f"Coverage gap: {missing_count} visible candidate prim(s) have no proposed material assignment."
        )
    if rejected_count and rejected_count > 0:
        issues.append(
            f"Coverage gap: {rejected_count} visible candidate prim(s) had proposed assignments rejected by the finalizer."
        )
    if issues:
        return issues
    return [
        f"Coverage gap: {candidate_count - decision_count} visible candidate prim(s) lack a material decision."
    ]


def _rejected_material_assignment_issues(run_dir: Path) -> list[dict[str, object]]:
    rejected = _load_json(
        run_dir / "raw" / "rejected_material_assignments.json", default=[]
    )
    if not isinstance(rejected, list):
        return []
    issues: list[dict[str, object]] = []
    for item in rejected:
        if not isinstance(item, dict):
            continue
        family = str(item.get("family") or "unnamed material decision")
        reason = str(item.get("rejection_reason") or "material assignment was rejected")
        paths = _dedupe_strings(
            _string_list(item.get("prim_paths"))
            + _string_list(item.get("runtime_prim_paths"))
            + _string_list(item.get("source_prim_paths"))
        )
        issues.append(
            {
                "description": f"Rejected material decision for {family}: {reason}",
                "affected_prim_paths": paths,
                "raw": item,
            }
        )
    return issues


def _int_or_none(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _refinement_issue_descriptions(value: object) -> list[str]:
    if not isinstance(value, list):
        return _string_list(value)
    descriptions: list[str] = []
    for item in value:
        if isinstance(item, dict):
            description = item.get("description") or item.get("issue") or item
            descriptions.append(str(description))
        elif item is not None:
            descriptions.append(str(item))
    return descriptions


def _vqa_issue_items(
    value: object,
    *,
    fallback_items: object | None = None,
) -> list[dict[str, object]]:
    items = _vqa_issue_items_from_value(value)
    if items:
        return items
    return _vqa_issue_items_from_value(fallback_items)


def _vqa_issue_items_from_value(value: object) -> list[dict[str, object]]:
    if value is None:
        return []
    if isinstance(value, dict):
        description = (
            value.get("description")
            or value.get("issue")
            or value.get("summary")
            or value.get("message")
            or ""
        )
        return [
            {
                "description": str(description),
                "affected_prim_paths": _issue_affected_prim_paths(value),
                "raw": value,
            }
        ]
    if isinstance(value, str):
        return [{"description": value, "affected_prim_paths": []}]
    if isinstance(value, list | tuple):
        items: list[dict[str, object]] = []
        for item in value:
            items.extend(_vqa_issue_items_from_value(item))
        return items
    return [{"description": str(value), "affected_prim_paths": []}]


def _issue_affected_prim_paths(value: dict[str, object]) -> list[str]:
    paths: list[str] = []
    for key in (
        "affected_prim_paths",
        "prim_paths",
        "runtime_prim_paths",
        "source_prim_paths",
        "target_prim_paths",
        "affected_prims",
        "prims",
    ):
        paths.extend(_string_list(value.get(key)))
    affected = value.get("affected")
    if isinstance(affected, dict):
        paths.extend(_issue_affected_prim_paths(affected))
    return _dedupe_strings([path for path in paths if path])


def _vqa_refinement_active_issues(
    *,
    unresolved_vqa: list[dict[str, object]],
    final_review_unresolved: list[dict[str, object]],
    rejected_assignments: list[dict[str, object]],
    coverage_gaps: list[str],
) -> list[dict[str, object]]:
    issues: list[dict[str, object]] = []
    seen: set[str] = set()
    sources = [
        ("rejected_assignment", rejected_assignments),
        ("visual_quality", unresolved_vqa),
        ("final_review", final_review_unresolved),
    ]
    for source, issue_items in sources:
        for issue_item in issue_items:
            description = str(issue_item.get("description") or "")
            normalized = _normalize_refinement_issue(description)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            affected_prim_paths = _string_list(issue_item.get("affected_prim_paths"))
            systematic_limitation = (
                False
                if source == "rejected_assignment"
                else _issue_looks_systematic_unfixable(description)
            )
            actionable = (
                True
                if source == "rejected_assignment"
                else bool(affected_prim_paths) or not systematic_limitation
            )
            issues.append(
                {
                    "id": f"issue-{len(issues) + 1}",
                    "source": source,
                    "description": description,
                    "affected_prim_paths": affected_prim_paths,
                    "actionable": actionable,
                    "systematic_limitation": systematic_limitation,
                }
            )
    for description in coverage_gaps:
        normalized = _normalize_refinement_issue(description)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        issues.append(
            {
                "id": f"issue-{len(issues) + 1}",
                "source": "coverage",
                "description": description,
                "affected_prim_paths": [],
                "actionable": True,
                "systematic_limitation": False,
            }
        )
    return issues


def _decision_patch_repair_summary(run_dir: Path) -> list[dict[str, object]]:
    patch = _load_json(run_dir / "raw" / "material_decision_patch.json", default={})
    if not isinstance(patch, dict):
        return []
    groups = []
    for key in ("material_assignments", "reviewed_no_override"):
        raw_groups = patch.get(key)
        if not isinstance(raw_groups, list):
            continue
        for group in raw_groups:
            if not isinstance(group, dict):
                continue
            paths = _string_list(group.get("prim_paths"))
            runtime_paths = _string_list(group.get("runtime_prim_paths"))
            source_paths = _string_list(group.get("source_prim_paths"))
            groups.append(
                {
                    "kind": key,
                    "family": group.get("family"),
                    "material_name": group.get("material_name"),
                    "material_path": group.get("material_path"),
                    "target_prim_count": max(
                        len(paths),
                        len(runtime_paths),
                        len(source_paths),
                    ),
                    "rationale": str(group.get("rationale") or "")[:500],
                }
            )
    return sorted(groups, key=lambda item: json.dumps(item, sort_keys=True))


def _vqa_refinement_signature_payload(
    *,
    status: str,
    active_issues: list[dict[str, object]],
    coverage: dict[str, object],
    decision_summary: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "status": status,
        "active_issues": [
            {
                "source": item.get("source"),
                "description": _normalize_refinement_issue(item.get("description")),
                "affected_prim_paths": sorted(
                    _string_list(item.get("affected_prim_paths"))
                ),
                "actionable": bool(item.get("actionable")),
                "systematic_limitation": bool(item.get("systematic_limitation")),
            }
            for item in active_issues
        ],
        "coverage": {
            "candidate_visible_prim_count": coverage.get(
                "candidate_visible_prim_count"
            ),
            "material_decision_prim_count": coverage.get(
                "material_decision_prim_count"
            ),
            "ambiguous_unassigned_prim_count": coverage.get(
                "ambiguous_unassigned_prim_count"
            ),
            "unassigned_visible_prim_count": coverage.get(
                "unassigned_visible_prim_count"
            ),
            "missing_assignment_prim_count": coverage.get(
                "missing_assignment_prim_count"
            ),
            "rejected_assignment_prim_count": coverage.get(
                "rejected_assignment_prim_count"
            ),
        },
        "material_decisions": [
            {
                "kind": item.get("kind"),
                "family": item.get("family"),
                "material_name": item.get("material_name"),
                "material_path": item.get("material_path"),
                "target_prim_count": item.get("target_prim_count"),
            }
            for item in decision_summary
        ],
    }


def _short_json_signature(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalize_refinement_issue(value: object) -> str:
    text = str(value or "").lower()
    text = re.sub(r"\biteration\s+\d+\b", "iteration", text)
    text = re.sub(r"/\S+", "<path>", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _issue_looks_systematic_unfixable(description: object) -> bool:
    text = _normalize_refinement_issue(description)
    markers = [
        "no suitable",
        "no semantically valid",
        "no exact",
        "no matching",
        "does not provide",
        "no wood or laminate material",
        "no exact supplied library material",
        "no exact supplied material",
        "no exact material",
        "material library has no",
        "material library lacks",
        "material palette has no",
        "material palette lacks",
        "library has no",
        "library lacks",
        "palette has no",
        "palette lacks",
        "no wood material",
        "not fixable with",
        "cannot be fixed with material overrides",
        "cannot be fixed with material overrides",
        "cannot be fixed with material assignments",
        "unfixable",
        "missing source geometry",
        "no source geometry",
        "granularity",
        "prim granularity",
        "granularity prevents",
        "coupled",
        "couples",
        "exposes that surface as part of",
        "cannot be split",
        "not separately authorable",
        "not separately bindable",
        "not separately targetable",
        "not independently addressable",
        "not independently assignable",
        "not independently bindable",
        "not independently targetable",
        "not separately material-addressable",
        "not exposed as individually assignable",
        "not exposed as separately assignable",
        "single bindable prim",
        "share workbench target paths",
        "shares workbench target paths",
        "share target paths",
        "shares target paths",
        "would also recolor",
        "would recolor",
        "did not expose a safe separate",
        "does not expose a safe separate",
        "no safe separate material path",
        "not safe exact material target",
        "not safe exact material targets",
        "no safe exact material target",
        "no safe exact material targets",
        "mixed generic-geometry buckets",
        "mixed generic geometry buckets",
        "too broad",
        "cannot pick",
        "unable to pick",
        "workbench cannot",
    ]
    return any(marker in text for marker in markers)


def _vqa_refinement_systematic_unfixable(assessment: dict[str, object]) -> bool:
    active_issues = assessment.get("active_issues")
    if not isinstance(active_issues, list) or not active_issues:
        return False
    if _assessment_notes_claim_systematic_limitations(
        assessment.get("assessment_notes")
    ):
        return True
    return all(
        isinstance(item, dict)
        and (
            bool(item.get("systematic_limitation"))
            or _issue_looks_systematic_unfixable(item.get("description"))
        )
        for item in active_issues
    )


def _assessment_notes_claim_systematic_limitations(notes: object) -> bool:
    text = str(notes or "").lower()
    if not text:
        return False
    limitation_markers = (
        "remaining differences are palette or geometry granularity limitations",
        "remaining differences are material-library or geometry limitations",
        "remaining differences are palette limitations",
        "remaining differences are geometry granularity limitations",
        "targeted picks did not provide safe isolated prims",
        "targeted pick evidence showed unsafe or missing targets",
        "not safely separable by pick",
    )
    if any(marker in text for marker in limitation_markers):
        return True
    return (
        "all fixable" in text
        and "remaining differences" in text
        and (
            "limitations rather than missed" in text
            or "not missed safe material assignments" in text
        )
    )


def _vqa_refinement_attempt_ledger(
    history: dict[str, object],
) -> list[dict[str, object]]:
    iterations = history.get("iterations")
    if not isinstance(iterations, list):
        return []
    ledger: list[dict[str, object]] = []
    for item in iterations[-3:]:
        if not isinstance(item, dict):
            continue
        after = item.get("after")
        before = item.get("before")
        ledger.append(
            {
                "iteration": item.get("iteration"),
                "child_returncode": item.get("child_returncode"),
                "before_active_issues": _active_issue_descriptions(before),
                "after_active_issues": _active_issue_descriptions(after),
                "converged": bool(item.get("converged")),
                "stop_reason": item.get("stop_reason"),
            }
        )
    return ledger


def _active_issue_descriptions(value: object) -> list[str]:
    if not isinstance(value, dict):
        return []
    active_issues = value.get("active_issues")
    if not isinstance(active_issues, list):
        return _string_list(value.get("unresolved_vqa_issues"))
    descriptions: list[str] = []
    for item in active_issues:
        if isinstance(item, dict):
            description = item.get("description")
            if description is not None:
                descriptions.append(str(description))
    return descriptions


def _load_vqa_refinement_history(path: Path) -> dict[str, object]:
    value = _load_json(path, default={})
    if isinstance(value, dict):
        value.setdefault("iterations", [])
        if not isinstance(value.get("iterations"), list):
            value["iterations"] = []
        return value
    return {"iterations": []}


def _vqa_refinement_child_artifacts(
    history: dict[str, object],
    *,
    initial_child_output_path: Path,
    initial_child_final_path: Path,
) -> list[dict[str, str]]:
    artifacts = [
        {
            "iteration": "1",
            "child_output": str(initial_child_output_path),
            "child_final": str(initial_child_final_path),
        }
    ]
    iterations = history.get("iterations")
    if isinstance(iterations, list):
        for item in iterations:
            if not isinstance(item, dict):
                continue
            artifacts.append(
                {
                    "iteration": str(item.get("iteration")),
                    "prompt": str(item.get("prompt") or ""),
                    "child_output": str(item.get("child_output") or ""),
                    "child_final": str(item.get("child_final") or ""),
                }
            )
    return artifacts


def _vqa_refinement_image_inputs(
    run_dir: Path,
    base_inputs: list[dict[str, str]],
) -> list[dict[str, str]]:
    inputs: list[dict[str, str]] = []
    seen: set[str] = set()
    # Refinement is a repair turn against current output. Re-attaching the
    # initial evidence renders substantially expands the child context without
    # helping targeted patch decisions.
    _ = base_inputs
    preferred_names = [
        "final_oblique.png",
        "final_front_py.png",
        "final_top.png",
        "final_side_px.png",
    ]
    final_dir = run_dir / "final_renders"
    preferred_paths = [final_dir / name for name in preferred_names]
    fallback_paths = sorted(final_dir.glob("*.png"))
    for path in [*preferred_paths, *fallback_paths]:
        if len(inputs) >= 3:
            break
        key = str(path)
        if not path.exists() or key in seen:
            continue
        seen.add(key)
        inputs.append({"label": f"Current final render {path.name}", "path": key})
    return inputs


def _finalize_structured_material_decisions(
    *,
    config: MaterialAssignConfig,
    run_dir: Path,
    preflight_packet: dict[str, Any] | None,
    trace_writer: TraceWriter,
) -> bool:
    session_id = _preflight_session_id(preflight_packet)
    if not session_id:
        trace_writer.write(
            "warning",
            phase="artifact finalization",
            summary=(
                "Skipped deterministic material-decision finalizer because no "
                "preflight Workbench session id was available."
            ),
            data={"preflight_enabled": preflight_packet is not None},
        )
        return False

    decision_patch, source = _material_decision_patch_from_run(
        run_dir,
        trace_writer=trace_writer,
    )
    if decision_patch is None:
        trace_writer.write(
            "warning",
            phase="artifact finalization",
            summary=(
                "Skipped deterministic material-decision finalizer because no "
                "decision patch or compatible assignments artifact was available."
            ),
        )
        return False

    try:
        paths = finalize_material_decisions(
            MaterialFinalizeConfig(
                workbench_url=config.workbench_url,
                run_dir=run_dir,
                session_id=session_id,
                source_usd=config.usd_path,
                materials_usd=config.materials_usd,
                reference_images=config.reference_images,
                reference_files=config.reference_files or [],
                decision_patch=decision_patch,
            )
        )
    except Exception as exc:
        error_path = run_dir / "raw" / "material_decision_finalizer_error.json"
        error_path.parent.mkdir(parents=True, exist_ok=True)
        error_path.write_text(
            json.dumps(
                {
                    "schema_version": (
                        "content-agents.material-decision-finalizer-error.v1"
                    ),
                    "decision_patch_source": source,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        raise

    trace_writer.write(
        "material_decision_finalized",
        phase="artifact finalization",
        summary=(
            "Applied deterministic material-decision finalizer to write canonical "
            "material assignment artifacts."
        ),
        artifacts=list(paths.values()),
        data={"decision_patch_source": source},
    )
    return True


def _record_materialized_output_status(
    *,
    run_dir: Path,
    output_usd_path: Path,
    status: str,
    error: Exception | None = None,
) -> None:
    """Keep user-facing artifacts aligned with durable materialization state."""

    if status not in {"pending", "succeeded", "failed"}:
        raise ValueError(f"Unsupported materialized USD status: {status}")
    if status == "failed" and error is None:
        raise ValueError("A failed materialized USD status requires an error")

    resolved_output_path = output_usd_path.expanduser().resolve()
    status_record: dict[str, object] = {
        "status": status,
        "requested_output_path": str(resolved_output_path),
    }
    if status == "succeeded":
        status_record["output_path"] = str(resolved_output_path)
    if error is not None:
        status_record["error_type"] = type(error).__name__
        status_record["error"] = str(error)
        if isinstance(error, MaterialRestoreCoverageError):
            status_record["unresolved_mappings"] = error.unresolved_mappings

    summary_path = run_dir / "final_summary.md"
    if not summary_path.is_file():
        raise RuntimeError(
            "Cannot record materialized USD status without final_summary.md."
        )
    summary = summary_path.read_text(encoding="utf-8")
    start_count = summary.count(MATERIALIZED_OUTPUT_SUMMARY_START)
    end_count = summary.count(MATERIALIZED_OUTPUT_SUMMARY_END)
    if start_count not in {0, 1} or end_count != start_count:
        raise RuntimeError("Malformed Materialized USD summary delimiters.")
    start_index = summary.find(MATERIALIZED_OUTPUT_SUMMARY_START)
    end_index = summary.find(MATERIALIZED_OUTPUT_SUMMARY_END)
    if start_count == 1 and end_index < start_index:
        raise RuntimeError("Malformed Materialized USD summary delimiters.")

    assignments_path = run_dir / "assignments.json"
    assignments = _load_json(assignments_path, default=None)
    if not isinstance(assignments, dict):
        raise RuntimeError(
            "Cannot record materialized USD status without valid assignments.json."
        )
    assignments["materialized_usd"] = status_record
    assignments_path.write_text(
        json.dumps(assignments, indent=2) + "\n",
        encoding="utf-8",
    )

    status_label = {
        "pending": "PENDING",
        "succeeded": "SUCCEEDED",
        "failed": "FAILED",
    }[status]
    safe_output_path = str(resolved_output_path).replace("`", "'")
    section_lines = [
        MATERIALIZED_OUTPUT_SUMMARY_START,
        "## Materialized USD",
        "",
        f"- Status: **{status_label}**",
        f"- Requested output: `{safe_output_path}`",
    ]
    if error is not None:
        safe_error = " ".join(str(error).splitlines()).replace("`", "'")
        section_lines.append(f"- Error: `{type(error).__name__}: {safe_error}`")
        if isinstance(error, MaterialRestoreCoverageError):
            section_lines.append("- Unresolved mappings:")
            for mapping in error.unresolved_mappings:
                safe_mapping = json.dumps(mapping, sort_keys=True).replace("`", "'")
                section_lines.append(f"  - `{safe_mapping}`")
    section_lines.append(MATERIALIZED_OUTPUT_SUMMARY_END)
    section = "\n".join(section_lines)

    if start_count == 1:
        end_index += len(MATERIALIZED_OUTPUT_SUMMARY_END)
        updated_summary = summary[:start_index] + section + summary[end_index:]
    else:
        updated_summary = summary.rstrip() + "\n\n" + section + "\n"
    summary_path.write_text(updated_summary, encoding="utf-8")


def _restore_materialized_output(
    *,
    config: MaterialAssignConfig,
    run_dir: Path,
    preflight_packet: dict[str, Any] | None,
    trace_writer: TraceWriter,
) -> Path:
    session_id = _preflight_session_id(preflight_packet)
    if not session_id:
        raise RuntimeError(
            "Cannot restore materialized USD without a preflight Workbench session."
        )
    if config.output_usd_path is None:
        raise RuntimeError("Materialized USD output path was not configured.")

    output_path = config.output_usd_path.resolve()
    try:
        with _shared_workbench_output_staging_dir(run_dir) as workbench_staging_dir:
            workbench_output_path = (
                workbench_staging_dir
                / f"materialized-output{output_path.suffix.lower()}"
            )
            response = workbench_client.restore_scene(
                config.workbench_url,
                session_id,
                {
                    "output_usd_path": str(workbench_output_path),
                    "output_mode": "flattened",
                    "overwrite": True,
                    "include_preview_artifact": False,
                    "fail_on_invalid_assignment": False,
                },
                timeout=config.material_restore_timeout_seconds,
            )
    finally:
        _reject_unsafe_run_links(run_dir)
    _verify_resealed_workbench_output(
        workbench_staging_dir,
        workbench_output_path,
    )
    response_path = run_dir / "raw" / "material_restore_response.json"
    response_path.write_text(
        json.dumps(response, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    unresolved_mappings = response.get("unresolved_mappings")
    if not isinstance(unresolved_mappings, list) or any(
        not isinstance(mapping, dict) for mapping in unresolved_mappings
    ):
        raise RuntimeError(
            "Workbench material restore returned an invalid unresolved_mappings value."
        )
    if unresolved_mappings:
        raise MaterialRestoreCoverageError(
            "Workbench material restore could not map all accepted assignments "
            f"to source prims ({len(unresolved_mappings)} unresolved mapping(s)).",
            unresolved_mappings=[dict(mapping) for mapping in unresolved_mappings],
        )
    restored_edit_count = response.get("restored_edit_count")
    if isinstance(restored_edit_count, bool) or not isinstance(
        restored_edit_count, int
    ):
        raise RuntimeError(
            "Workbench material restore returned an invalid restored_edit_count."
        )
    restored_source_prim_paths = response.get("restored_source_prim_paths")
    if (
        not isinstance(restored_source_prim_paths, list)
        or any(
            not isinstance(path, str) or not path.startswith("/")
            for path in restored_source_prim_paths
        )
        or len(set(restored_source_prim_paths)) != len(restored_source_prim_paths)
    ):
        raise RuntimeError(
            "Workbench material restore returned invalid restored source coverage."
        )
    if restored_edit_count != len(restored_source_prim_paths):
        raise RuntimeError(
            "Workbench material restore count did not match restored source "
            f"coverage: expected {len(restored_source_prim_paths)}, got "
            f"{restored_edit_count}."
        )
    uncovered_groups = _validate_material_restore_source_coverage(
        run_dir,
        restored_source_prim_paths=restored_source_prim_paths,
    )
    raw_unbound_paths = response.get("unbound_source_prim_paths")
    if not isinstance(raw_unbound_paths, list) or any(
        not isinstance(path, str) or not path.startswith("/")
        for path in raw_unbound_paths
    ):
        raise RuntimeError(
            "Workbench material restore returned invalid unbound source coverage."
        )
    unbound_source_prim_paths = sorted(set(raw_unbound_paths))
    restored_path_value = response.get("output_usd_path")
    restored_path = (
        _lexical_absolute_path(Path(restored_path_value))
        if isinstance(restored_path_value, str) and restored_path_value.strip()
        else None
    )
    if restored_path != workbench_output_path:
        raise RuntimeError(
            "Workbench material restore did not produce the requested USD: "
            f"{workbench_output_path}"
        )
    _validate_materialized_usd_output(workbench_output_path)
    _publish_materialized_output(workbench_output_path, output_path)
    _validate_materialized_usd_output(output_path)
    if unbound_source_prim_paths or uncovered_groups:
        trace_writer.write(
            "warning",
            phase="material output",
            summary=(
                "Restored a durable USD with partial material assignment coverage."
            ),
            artifacts=[str(output_path), str(response_path)],
            data={
                "unbound_source_prim_paths": unbound_source_prim_paths,
                "uncovered_assignment_groups": uncovered_groups,
            },
        )
    trace_writer.write(
        "material_output_restored",
        phase="material output",
        summary="Restored accepted material assignments to durable USD.",
        artifacts=[str(output_path), str(response_path)],
        data={
            "output_mode": response.get("output_mode"),
            "restored_edit_count": restored_edit_count,
            "restored_source_prim_paths": restored_source_prim_paths,
            "unbound_source_prim_paths": unbound_source_prim_paths,
            "uncovered_assignment_groups": uncovered_groups,
        },
    )
    return output_path


@contextmanager
def _shared_workbench_output_staging_dir(run_dir: Path) -> Iterator[Path]:
    """Share one anchored staging directory only during a Workbench request."""

    lexical_run_dir = _lexical_absolute_path(run_dir)
    staging_name = f"{WORKBENCH_OUTPUT_STAGING_DIR_PREFIX}{uuid4().hex}"
    staging_path = lexical_run_dir / staging_name
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    run_fd: int | None = None
    staging_fd: int | None = None
    run_identity: tuple[int, int] | None = None
    run_original_mode: int | None = None
    staging_identity: tuple[int, int] | None = None

    def reseal() -> OSError | None:
        reseal_error: OSError | None = None
        if staging_fd is not None:
            try:
                os.fchmod(staging_fd, WORKBENCH_OUTPUT_STAGING_PRIVATE_MODE)
                private = os.fstat(staging_fd)
                if (
                    not stat.S_ISDIR(private.st_mode)
                    or stat.S_IMODE(private.st_mode)
                    != WORKBENCH_OUTPUT_STAGING_PRIVATE_MODE
                ):
                    raise OSError("Workbench output staging reseal verification failed")
                if staging_identity is not None:
                    if (private.st_dev, private.st_ino) != staging_identity:
                        raise OSError(
                            "Workbench output staging changed while being resealed"
                        )
                    if run_fd is None:
                        raise OSError(
                            "Workbench output staging run descriptor is unavailable"
                        )
                    current = os.stat(
                        staging_name,
                        dir_fd=run_fd,
                        follow_symlinks=False,
                    )
                    if (
                        not stat.S_ISDIR(current.st_mode)
                        or (current.st_dev, current.st_ino) != staging_identity
                        or stat.S_IMODE(current.st_mode)
                        != WORKBENCH_OUTPUT_STAGING_PRIVATE_MODE
                    ):
                        raise OSError(
                            "Workbench output staging reseal verification failed"
                        )
            except OSError as exc:
                reseal_error = exc

        if run_fd is not None and run_original_mode is not None:
            try:
                os.fchmod(run_fd, run_original_mode)
                private_run = os.fstat(run_fd)
                current_run = os.stat(lexical_run_dir, follow_symlinks=False)
                if (
                    not stat.S_ISDIR(private_run.st_mode)
                    or run_identity is None
                    or (private_run.st_dev, private_run.st_ino) != run_identity
                    or (current_run.st_dev, current_run.st_ino) != run_identity
                    or stat.S_IMODE(private_run.st_mode) != run_original_mode
                    or stat.S_IMODE(current_run.st_mode) != run_original_mode
                ):
                    raise OSError("Workbench run directory reseal verification failed")
            except OSError as exc:
                if reseal_error is None:
                    reseal_error = exc
        return reseal_error

    def close_descriptors() -> None:
        if staging_fd is not None:
            os.close(staging_fd)
        if run_fd is not None:
            os.close(run_fd)

    setup_error: UnsafeRunArtifactError | OSError | None = None
    try:
        run_fd = os.open(lexical_run_dir, directory_flags)
        initial_run = os.fstat(run_fd)
        current_run = os.stat(lexical_run_dir, follow_symlinks=False)
        run_identity = (initial_run.st_dev, initial_run.st_ino)
        run_original_mode = stat.S_IMODE(initial_run.st_mode)
        if (
            not stat.S_ISDIR(initial_run.st_mode)
            or (current_run.st_dev, current_run.st_ino) != run_identity
            or stat.S_IMODE(current_run.st_mode) != run_original_mode
        ):
            raise UnsafeRunArtifactError(
                "Workbench run directory changed while opening"
            )
        # A collision or precreated link must fail rather than reusing a path
        # that was not created for this synchronous restore request.
        os.mkdir(staging_name, mode=0o700, dir_fd=run_fd)
        initial = os.stat(staging_name, dir_fd=run_fd, follow_symlinks=False)
        if not stat.S_ISDIR(initial.st_mode):
            raise UnsafeRunArtifactError(
                "Workbench output staging path must be a directory"
            )
        staging_fd = os.open(staging_name, directory_flags, dir_fd=run_fd)
        opened = os.fstat(staging_fd)
        staging_identity = (initial.st_dev, initial.st_ino)
        if (opened.st_dev, opened.st_ino) != staging_identity:
            raise UnsafeRunArtifactError(
                "Workbench output staging directory changed while opening"
            )

        # A different service UID needs r+w+x to create the output in this
        # leaf. Sticky deletion semantics and the randomized name bound that
        # access to this one synchronous restore request.
        os.fchmod(staging_fd, WORKBENCH_OUTPUT_STAGING_SHARED_MODE)
        shared = os.fstat(staging_fd)
        current = os.stat(staging_name, dir_fd=run_fd, follow_symlinks=False)
        if (
            not stat.S_ISDIR(shared.st_mode)
            or (shared.st_dev, shared.st_ino) != staging_identity
            or (current.st_dev, current.st_ino) != staging_identity
            or stat.S_IMODE(shared.st_mode) != WORKBENCH_OUTPUT_STAGING_SHARED_MODE
            or stat.S_IMODE(current.st_mode) != WORKBENCH_OUTPUT_STAGING_SHARED_MODE
        ):
            raise UnsafeRunArtifactError(
                "Workbench output staging directory changed while sharing"
            )

        # Existing remote Workbench processes need only traversal through the
        # private run root. Do not grant directory-read permission or change
        # any child artifact directory such as raw/.
        shared_run_mode = run_original_mode | WORKBENCH_RUN_DIR_TRAVERSE_BITS
        os.fchmod(run_fd, shared_run_mode)
        shared_run = os.fstat(run_fd)
        current_run = os.stat(lexical_run_dir, follow_symlinks=False)
        if (
            not stat.S_ISDIR(shared_run.st_mode)
            or (shared_run.st_dev, shared_run.st_ino) != run_identity
            or (current_run.st_dev, current_run.st_ino) != run_identity
            or stat.S_IMODE(shared_run.st_mode) != shared_run_mode
            or stat.S_IMODE(current_run.st_mode) != shared_run_mode
        ):
            raise UnsafeRunArtifactError(
                "Workbench run directory changed while sharing"
            )
    except (UnsafeRunArtifactError, OSError) as exc:
        setup_error = exc

    if setup_error is not None:
        reseal_error = reseal()
        close_descriptors()
        if reseal_error is not None:
            raise UnsafeRunArtifactError(
                f"Unable to reseal Workbench output staging below {run_dir}"
            ) from reseal_error
        if isinstance(setup_error, UnsafeRunArtifactError):
            raise setup_error
        raise UnsafeRunArtifactError(
            f"Unable to prepare Workbench output staging below {run_dir}"
        ) from setup_error

    try:
        yield staging_path
    finally:
        # Revoke cross-UID access before inspecting or publishing any
        # service-created artifact, including when the request raises.
        reseal_error = reseal()
        close_descriptors()
        if reseal_error is not None:
            raise UnsafeRunArtifactError(
                f"Unable to reseal Workbench output staging below {run_dir}"
            ) from reseal_error


def _verify_resealed_workbench_output(staging_dir: Path, output_path: Path) -> None:
    """Verify one Workbench output without following links or special files."""

    lexical_staging_dir = _lexical_absolute_path(staging_dir)
    lexical_output_path = _lexical_absolute_path(output_path)
    if lexical_output_path.parent != lexical_staging_dir:
        raise UnsafeRunArtifactError(
            f"Workbench output escapes its staging directory: {output_path}"
        )
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    file_flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
    staging_fd: int | None = None
    output_fd: int | None = None
    try:
        staging_fd = os.open(lexical_staging_dir, directory_flags)
        staging_metadata = os.fstat(staging_fd)
        if (
            stat.S_IMODE(staging_metadata.st_mode)
            != WORKBENCH_OUTPUT_STAGING_PRIVATE_MODE
        ):
            raise UnsafeRunArtifactError(
                "Workbench output staging directory was not resealed"
            )
        initial = os.stat(
            lexical_output_path.name,
            dir_fd=staging_fd,
            follow_symlinks=False,
        )
        if not stat.S_ISREG(initial.st_mode) or initial.st_nlink != 1:
            raise UnsafeRunArtifactError(
                "Workbench output must be a single-link regular file"
            )
        output_fd = os.open(lexical_output_path.name, file_flags, dir_fd=staging_fd)
        opened = os.fstat(output_fd)
        current = os.stat(
            lexical_output_path.name,
            dir_fd=staging_fd,
            follow_symlinks=False,
        )
        expected_identity = (initial.st_dev, initial.st_ino)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or (opened.st_dev, opened.st_ino) != expected_identity
            or (current.st_dev, current.st_ino) != expected_identity
            or not stat.S_ISREG(current.st_mode)
            or current.st_nlink != 1
        ):
            raise UnsafeRunArtifactError(
                "Workbench output changed while being verified"
            )
    except UnsafeRunArtifactError:
        raise
    except OSError as exc:
        raise UnsafeRunArtifactError(
            f"Unable to verify Workbench output safely: {output_path}"
        ) from exc
    finally:
        if output_fd is not None:
            os.close(output_fd)
        if staging_fd is not None:
            os.close(staging_fd)


@contextmanager
def _anchored_materialized_output_parent(
    output_path: Path,
) -> Iterator[tuple[Path, int, Callable[[], None]]]:
    """Open or create an output parent without following path-component links."""

    lexical_output_path = _lexical_absolute_path(output_path)
    if not lexical_output_path.name or not lexical_output_path.anchor:
        raise UnsafeRunArtifactError(
            f"Materialized output path must name an absolute file: {output_path}"
        )
    anchor_path = Path(lexical_output_path.anchor)
    relative_parent = lexical_output_path.parent.relative_to(anchor_path)
    directory_access_flag = getattr(os, "O_PATH", os.O_RDONLY)
    directory_flags = (
        directory_access_flag | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    )
    directory_fds: list[int] = []
    links: list[tuple[int, str, int, tuple[int, int]]] = []

    def close_descriptors() -> None:
        for directory_fd in reversed(directory_fds):
            os.close(directory_fd)

    def validate_chain() -> None:
        try:
            for parent_fd, component, child_fd, expected_identity in links:
                opened = os.fstat(child_fd)
                current = os.stat(
                    component,
                    dir_fd=parent_fd,
                    follow_symlinks=False,
                )
                if (
                    not stat.S_ISDIR(opened.st_mode)
                    or not stat.S_ISDIR(current.st_mode)
                    or (opened.st_dev, opened.st_ino) != expected_identity
                    or (current.st_dev, current.st_ino) != expected_identity
                ):
                    raise UnsafeRunArtifactError(
                        "Materialized output parent changed during publication"
                    )
        except UnsafeRunArtifactError:
            raise
        except OSError as exc:
            raise UnsafeRunArtifactError(
                "Materialized output parent changed during publication"
            ) from exc

    try:
        root_fd = os.open(anchor_path, directory_flags)
        directory_fds.append(root_fd)
        root_metadata = os.fstat(root_fd)
        if not stat.S_ISDIR(root_metadata.st_mode):
            raise UnsafeRunArtifactError(
                f"Materialized output anchor is not a directory: {anchor_path}"
            )
        for component in relative_parent.parts:
            if component in {"", os.curdir, os.pardir}:
                raise UnsafeRunArtifactError(
                    f"Invalid materialized output parent component: {component!r}"
                )
            parent_fd = directory_fds[-1]
            try:
                child_fd = os.open(component, directory_flags, dir_fd=parent_fd)
            except FileNotFoundError:
                try:
                    os.mkdir(component, mode=0o777, dir_fd=parent_fd)
                except FileExistsError:
                    pass
                child_fd = os.open(component, directory_flags, dir_fd=parent_fd)
            directory_fds.append(child_fd)
            opened = os.fstat(child_fd)
            current = os.stat(
                component,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
            expected_identity = (opened.st_dev, opened.st_ino)
            if (
                not stat.S_ISDIR(opened.st_mode)
                or not stat.S_ISDIR(current.st_mode)
                or (current.st_dev, current.st_ino) != expected_identity
            ):
                raise UnsafeRunArtifactError(
                    "Materialized output parent changed while being opened"
                )
            links.append((parent_fd, component, child_fd, expected_identity))
        validate_chain()
    except UnsafeRunArtifactError:
        close_descriptors()
        raise
    except OSError as exc:
        close_descriptors()
        raise UnsafeRunArtifactError(
            f"Unable to open materialized output parent safely: {output_path}"
        ) from exc

    try:
        yield lexical_output_path, directory_fds[-1], validate_chain
    finally:
        close_descriptors()


def _write_anchored_materialized_output_temp(
    *,
    parent_fd: int,
    scratch_path: Path,
    output_suffix: str,
) -> tuple[str, int, tuple[int, int]]:
    """Copy into a private temp and return its still-open trusted descriptor."""

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
    temporary_fd: int | None = None
    temporary_name: str | None = None
    for _attempt in range(16):
        candidate = f".materialized-publish-{uuid4().hex}{output_suffix}"
        try:
            temporary_fd = os.open(candidate, flags, 0o600, dir_fd=parent_fd)
        except FileExistsError:
            continue
        temporary_name = candidate
        break
    if temporary_fd is None or temporary_name is None:
        raise UnsafeRunArtifactError(
            "Unable to allocate a private materialized output temporary file"
        )

    try:
        os.fchmod(temporary_fd, 0o600)
        initial = os.fstat(temporary_fd)
        expected_identity = (initial.st_dev, initial.st_ino)
        if not stat.S_ISREG(initial.st_mode) or initial.st_nlink != 1:
            raise UnsafeRunArtifactError(
                "Materialized output temporary must be a single-link regular file"
            )
        with scratch_path.open("rb") as source_stream:
            with os.fdopen(
                temporary_fd,
                "wb",
                closefd=False,
            ) as destination_stream:
                shutil.copyfileobj(source_stream, destination_stream)
                destination_stream.flush()
                os.fsync(destination_stream.fileno())
                final = os.fstat(destination_stream.fileno())
                if (
                    not stat.S_ISREG(final.st_mode)
                    or final.st_nlink != 1
                    or (final.st_dev, final.st_ino) != expected_identity
                    or stat.S_IMODE(final.st_mode) != 0o600
                ):
                    raise UnsafeRunArtifactError(
                        "Materialized output temporary changed while being written"
                    )
        current = os.stat(
            temporary_name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(current.st_mode)
            or current.st_nlink != 1
            or (current.st_dev, current.st_ino) != expected_identity
        ):
            raise UnsafeRunArtifactError(
                "Materialized output temporary changed after being written"
            )
        return temporary_name, temporary_fd, expected_identity
    except BaseException:
        if temporary_fd is not None:
            os.close(temporary_fd)
        try:
            os.unlink(temporary_name, dir_fd=parent_fd)
        except FileNotFoundError:
            pass
        raise


def _unlink_anchored_materialized_output_temp_if_present(
    *,
    parent_fd: int,
    temporary_name: str,
    temporary_fd: int,
    expected_identity: tuple[int, int],
) -> None:
    """Remove an unpublished trusted temporary without following a swapped name."""

    try:
        current = os.stat(
            temporary_name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return
    trusted = os.fstat(temporary_fd)
    if (
        not stat.S_ISREG(current.st_mode)
        or not stat.S_ISREG(trusted.st_mode)
        or current.st_nlink != 1
        or trusted.st_nlink != 1
        or (current.st_dev, current.st_ino) != expected_identity
        or (trusted.st_dev, trusted.st_ino) != expected_identity
    ):
        raise UnsafeRunArtifactError(
            "Materialized output temporary changed before cleanup"
        )
    os.unlink(temporary_name, dir_fd=parent_fd)


def _close_materialized_output_temporary_lifetime(
    temporary_lifetime: ExitStack,
    *,
    primary_error: BaseException | None = None,
) -> None:
    """Close temporary resources without hiding an active publication error."""

    try:
        temporary_lifetime.close()
    except BaseException as cleanup_error:
        if primary_error is None:
            raise
        primary_error.add_note(
            "Additional materialized-output temporary cleanup failure: "
            f"{type(cleanup_error).__name__}: {cleanup_error}"
        )


def _link_materialized_output_backup(
    *,
    parent_fd: int,
    output_name: str,
    expected_identity: tuple[int, int],
) -> str:
    """Preserve an existing output under a random anchored backup name."""

    backup_name: str | None = None
    for _attempt in range(16):
        candidate = f".materialized-backup-{uuid4().hex}"
        try:
            os.link(
                output_name,
                candidate,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileExistsError:
            continue
        backup_name = candidate
        break
    if backup_name is None:
        raise UnsafeRunArtifactError(
            "Unable to preserve the existing materialized output"
        )

    try:
        output_metadata = os.stat(
            output_name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        backup_metadata = os.stat(
            backup_name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(output_metadata.st_mode)
            or not stat.S_ISREG(backup_metadata.st_mode)
            or output_metadata.st_nlink != 2
            or backup_metadata.st_nlink != 2
            or (output_metadata.st_dev, output_metadata.st_ino) != expected_identity
            or (backup_metadata.st_dev, backup_metadata.st_ino) != expected_identity
        ):
            raise UnsafeRunArtifactError(
                "Existing materialized output changed while being preserved"
            )
    except BaseException:
        try:
            os.unlink(backup_name, dir_fd=parent_fd)
        except FileNotFoundError:
            pass
        except OSError as cleanup_exc:
            raise UnsafeRunArtifactError(
                "Unable to clean up an unverified materialized output backup"
            ) from cleanup_exc
        raise
    return backup_name


def _publish_materialized_output(source_path: Path, output_path: Path) -> None:
    """Atomically publish a confined Workbench result from the parent process."""

    with _anchored_materialized_output_parent(output_path) as (
        lexical_output_path,
        parent_fd,
        validate_parent,
    ):
        output_name = lexical_output_path.name
        output_suffix = lexical_output_path.suffix.lower()
        temporary_lifetime = ExitStack()
        try:
            with tempfile.TemporaryDirectory(
                prefix="content-workflow-materialized-publish-"
            ) as scratch_directory:
                scratch_path = Path(scratch_directory) / f"output{output_suffix}"
                if output_suffix == ".usdz":
                    shutil.copy2(source_path, scratch_path)
                else:
                    _export_materialized_output_with_rebased_assets(
                        source_path,
                        scratch_path,
                        logical_output_parent=lexical_output_path.parent,
                    )
                _validate_materialized_usd_output(scratch_path)
                temporary_name, temporary_fd, temporary_identity = (
                    _write_anchored_materialized_output_temp(
                        parent_fd=parent_fd,
                        scratch_path=scratch_path,
                        output_suffix=output_suffix,
                    )
                )
                temporary_lifetime.callback(os.close, temporary_fd)
                temporary_lifetime.callback(
                    _unlink_anchored_materialized_output_temp_if_present,
                    parent_fd=parent_fd,
                    temporary_name=temporary_name,
                    temporary_fd=temporary_fd,
                    expected_identity=temporary_identity,
                )
        except BaseException as primary_error:
            _close_materialized_output_temporary_lifetime(
                temporary_lifetime,
                primary_error=primary_error,
            )
            raise

        backup_name: str | None = None
        backup_identity: tuple[int, int] | None = None
        published = False
        try:
            try:
                existing = os.stat(
                    output_name,
                    dir_fd=parent_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                existing = None
            if existing is not None:
                if not stat.S_ISREG(existing.st_mode) or existing.st_nlink != 1:
                    raise UnsafeRunArtifactError(
                        "Existing materialized output must be a single-link "
                        "regular file"
                    )
                backup_identity = (existing.st_dev, existing.st_ino)
                backup_name = _link_materialized_output_backup(
                    parent_fd=parent_fd,
                    output_name=output_name,
                    expected_identity=backup_identity,
                )

            validate_parent()
            current_temporary = os.stat(
                temporary_name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISREG(current_temporary.st_mode)
                or current_temporary.st_nlink != 1
                or (current_temporary.st_dev, current_temporary.st_ino)
                != temporary_identity
            ):
                raise UnsafeRunArtifactError(
                    "Materialized output temporary changed before atomic publication"
                )
            os.replace(
                temporary_name,
                output_name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
            published = True
            current = os.stat(
                output_name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISREG(current.st_mode)
                or current.st_nlink != 1
                or (current.st_dev, current.st_ino) != temporary_identity
            ):
                raise UnsafeRunArtifactError(
                    "Materialized output changed during atomic publication"
                )
            validate_parent()
            if backup_name is not None:
                os.unlink(backup_name, dir_fd=parent_fd)
                backup_name = None
                backup_identity = None
        except BaseException:
            rollback_error: OSError | UnsafeRunArtifactError | None = None
            try:
                if published:
                    current = os.stat(
                        output_name,
                        dir_fd=parent_fd,
                        follow_symlinks=False,
                    )
                    if (
                        not stat.S_ISREG(current.st_mode)
                        or current.st_nlink != 1
                        or (current.st_dev, current.st_ino) != temporary_identity
                    ):
                        raise UnsafeRunArtifactError(
                            "Cannot safely roll back a replaced materialized output"
                        )
                    if backup_name is not None:
                        if backup_identity is None:
                            raise UnsafeRunArtifactError(
                                "Materialized output backup identity is unavailable"
                            )
                        current_backup = os.stat(
                            backup_name,
                            dir_fd=parent_fd,
                            follow_symlinks=False,
                        )
                        if (
                            not stat.S_ISREG(current_backup.st_mode)
                            or current_backup.st_nlink != 1
                            or (current_backup.st_dev, current_backup.st_ino)
                            != backup_identity
                        ):
                            raise UnsafeRunArtifactError(
                                "Materialized output backup changed before rollback"
                            )
                        os.replace(
                            backup_name,
                            output_name,
                            src_dir_fd=parent_fd,
                            dst_dir_fd=parent_fd,
                        )
                        restored = os.stat(
                            output_name,
                            dir_fd=parent_fd,
                            follow_symlinks=False,
                        )
                        if (
                            not stat.S_ISREG(restored.st_mode)
                            or restored.st_nlink != 1
                            or (restored.st_dev, restored.st_ino) != backup_identity
                        ):
                            raise UnsafeRunArtifactError(
                                "Materialized output backup changed during rollback"
                            )
                        backup_name = None
                        backup_identity = None
                    else:
                        os.unlink(output_name, dir_fd=parent_fd)
                elif backup_name is not None:
                    os.unlink(backup_name, dir_fd=parent_fd)
                    backup_name = None
            except (OSError, UnsafeRunArtifactError) as rollback_exc:
                rollback_error = rollback_exc
            if rollback_error is not None:
                raise UnsafeRunArtifactError(
                    "Unable to roll back materialized output publication safely"
                ) from rollback_error
            raise
        finally:
            _close_materialized_output_temporary_lifetime(
                temporary_lifetime,
                primary_error=sys.exception(),
            )


def _export_materialized_output_with_rebased_assets(
    source_path: Path,
    output_path: Path,
    *,
    logical_output_parent: Path,
) -> None:
    """Export a relocated USD while preserving local asset-path targets."""

    from pxr import Sdf, UsdUtils

    anchor_layer = Sdf.Layer.FindOrOpen(str(source_path))
    editable_layer = Sdf.Layer.OpenAsAnonymous(str(source_path))
    if anchor_layer is None or editable_layer is None:
        raise RuntimeError(f"Could not open confined materialized USD: {source_path}")

    def rebase_asset_path(asset_path: str) -> str:
        if (
            not asset_path
            or Path(asset_path).is_absolute()
            or urlparse(asset_path).scheme
        ):
            return asset_path
        absolute_path = anchor_layer.ComputeAbsolutePath(asset_path)
        if not absolute_path:
            return asset_path
        return os.path.relpath(absolute_path, logical_output_parent).replace("\\", "/")

    UsdUtils.ModifyAssetPaths(
        editable_layer,
        rebase_asset_path,
        keepEmptyPathsInArrays=True,
    )
    if not editable_layer.Export(str(output_path)):
        raise RuntimeError(f"Could not export materialized USD: {output_path}")


def _validate_materialized_usd_output(output_path: Path) -> None:
    output_suffix = output_path.suffix.lower()
    if output_suffix == ".usdz" and not zipfile.is_zipfile(output_path):
        raise RuntimeError(
            f"Workbench material restore produced an invalid USDZ: {output_path}"
        )
    try:
        from pxr import Usd
    except ImportError as exc:
        raise RuntimeError(
            "Cannot validate Workbench material restore without OpenUSD."
        ) from exc
    try:
        stage = Usd.Stage.Open(str(output_path))
    except Exception as exc:  # pragma: no cover - OpenUSD exception types vary
        raise RuntimeError(
            f"Workbench material restore produced an invalid USD: {output_path}"
        ) from exc
    if stage is None:
        raise RuntimeError(
            f"Workbench material restore produced an invalid USD: {output_path}"
        )
    expected_format = {".usda": "usda", ".usdc": "usdc"}.get(output_suffix)
    actual_format = stage.GetRootLayer().GetFileFormat().formatId
    if expected_format is not None and actual_format != expected_format:
        raise RuntimeError(
            "Workbench material restore output encoding does not match its "
            f"{output_suffix} suffix: expected {expected_format}, got {actual_format}"
        )


def _validate_material_restore_source_coverage(
    run_dir: Path,
    *,
    restored_source_prim_paths: list[str],
) -> list[dict[str, object]]:
    assignments = _load_json(run_dir / "assignments.json", default=None)
    if not isinstance(assignments, dict):
        raise RuntimeError(
            "Cannot validate material restore without canonical assignments.json."
        )
    groups = assignments.get("assignments")
    if not isinstance(groups, list):
        raise RuntimeError(
            "Cannot validate material restore because assignments.json is invalid."
        )
    restored_paths = set(restored_source_prim_paths)
    aliases_by_source_path = _material_restore_aliases_by_source_path(run_dir)
    allowed_paths: set[str] = set()
    accepted_groups: list[tuple[set[str], set[str], dict[str, set[str]]]] = []
    for group in groups:
        if not isinstance(group, dict):
            continue
        if group.get("coverage_status") != "material_assignment":
            continue
        group_source_paths = _string_list(group.get("source_prim_paths"))
        if not group_source_paths:
            raise RuntimeError(
                "Accepted material assignment is missing source-space prim paths."
            )
        runtime_paths = _string_list(group.get("runtime_prim_paths"))
        source_path_set = set(group_source_paths)
        runtime_path_set = set(runtime_paths)
        source_path_aliases = {
            source_path: {source_path} | aliases_by_source_path.get(source_path, set())
            for source_path in source_path_set
        }
        accepted_groups.append((source_path_set, runtime_path_set, source_path_aliases))
        allowed_paths.update(source_path_set)
        allowed_paths.update(runtime_path_set)
        for aliases in source_path_aliases.values():
            allowed_paths.update(aliases)

    unexpected_paths = sorted(restored_paths - allowed_paths)
    if unexpected_paths:
        raise RuntimeError(
            "Workbench material restore authored source paths outside accepted "
            f"assignment coverage: {unexpected_paths}."
        )
    uncovered_groups = [
        {
            "source_prim_paths": sorted(source_paths),
            "runtime_prim_paths": sorted(runtime_paths),
            "original_source_aliases": {
                source_path: sorted(aliases - {source_path})
                for source_path, aliases in sorted(source_path_aliases.items())
                if aliases - {source_path}
            },
        }
        for source_paths, runtime_paths, source_path_aliases in accepted_groups
        if not (
            source_paths.issubset(restored_paths)
            or (runtime_paths and runtime_paths.issubset(restored_paths))
            or all(
                bool(aliases & restored_paths)
                for aliases in source_path_aliases.values()
            )
        )
    ]
    return uncovered_groups


def _material_restore_aliases_by_source_path(
    run_dir: Path,
) -> dict[str, set[str]]:
    candidates = _load_json(
        run_dir / "raw" / "visible_candidate_prims.json",
        default={},
    )
    if not isinstance(candidates, dict):
        return {}
    rows = candidates.get("candidates")
    if not isinstance(rows, list):
        return {}

    aliases_by_source_path: dict[str, set[str]] = {}
    for candidate in rows:
        if not isinstance(candidate, dict):
            continue
        source_paths = set(_string_list(candidate.get("source_path")))
        source_paths.update(_string_list(candidate.get("source_paths")))
        original_paths = set(_string_list(candidate.get("original_source_path")))
        original_paths.update(_string_list(candidate.get("original_source_paths")))
        for source_path in source_paths:
            aliases_by_source_path.setdefault(source_path, set()).update(original_paths)
    return aliases_by_source_path


def _material_decision_patch_from_run(
    run_dir: Path,
    *,
    trace_writer: TraceWriter,
) -> tuple[dict[str, Any] | None, str]:
    raw_dir = run_dir / "raw"
    patch_path = raw_dir / "material_decision_patch.json"
    patch = _load_json(
        patch_path,
        default=None,
        trace_writer=trace_writer,
        phase="artifact finalization",
    )
    if isinstance(patch, dict):
        return patch, "raw/material_decision_patch.json"
    if patch_path.exists():
        trace_writer.write(
            "warning",
            phase="artifact finalization",
            summary=(
                "Ignored material_decision_patch.json because it did not contain "
                "a JSON object."
            ),
            artifacts=[str(patch_path)],
        )

    assignments = _load_json(
        run_dir / "assignments.json",
        default={},
        trace_writer=trace_writer,
        phase="artifact finalization",
    )
    if not isinstance(assignments, dict):
        return None, "missing"
    patch = _decision_patch_from_assignments(run_dir, assignments)
    if patch is None:
        return None, "missing"
    return patch, "assignments.json"


def _decision_patch_from_assignments(
    run_dir: Path,
    assignments: dict[str, Any],
) -> dict[str, Any] | None:
    assignment_groups = _assignment_group_records(assignments.get("assignments"))
    if assignment_groups is None:
        return None

    candidates = _load_json(
        run_dir / "raw" / "visible_candidate_prims.json",
        default={},
    )
    path_space = (
        "inspection"
        if isinstance(candidates, dict)
        and str(candidates.get("path_space") or "source") == "inspection"
        else "source"
    )
    material_assignments = []
    reviewed_no_override = []
    for group in assignment_groups:
        status = str(group.get("coverage_status") or "")
        patch_group = _decision_patch_group(group, path_space=path_space)
        if patch_group is None:
            continue
        if _is_material_assignment_status(status):
            material_name = str(group.get("material_name") or "").strip()
            if not material_name:
                continue
            patch_group["material_name"] = material_name
            material_path = group.get("material_path")
            if material_path is not None:
                patch_group["material_path"] = str(material_path)
            material_assignments.append(patch_group)
        elif status == "preserved_existing":
            reviewed_no_override.append(patch_group)

    final_review = assignments.get("final_review")
    if not isinstance(final_review, dict):
        final_review = {}
    visual_quality = assignments.get("visual_quality_assessment")
    if not isinstance(visual_quality, dict):
        visual_quality = _load_json(
            run_dir / "visual_quality_assessment.json",
            default={},
        )
    if not isinstance(visual_quality, dict):
        visual_quality = {}

    return {
        "schema_version": "content-agents.material-decision-patch.v1",
        "material_assignments": material_assignments,
        "reviewed_no_override": reviewed_no_override,
        "final_review_issues_found": _string_list(final_review.get("issues_found")),
        "final_review_issues_fixed": _string_list(final_review.get("issues_fixed")),
        "final_review_notes": str(final_review.get("review_notes") or ""),
        "visual_quality_assessment": visual_quality,
    }


def _assignment_group_records(value: object) -> list[dict[str, Any]] | None:
    if isinstance(value, list):
        return [dict(item) for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        groups = []
        for status, raw_items in value.items():
            if isinstance(raw_items, dict):
                items = [raw_items]
            elif isinstance(raw_items, list):
                items = raw_items
            else:
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                group = dict(item)
                group.setdefault("coverage_status", str(status))
                groups.append(group)
        return groups
    return None


def _decision_patch_group(
    group: dict[str, Any],
    *,
    path_space: str,
) -> dict[str, Any] | None:
    prim_paths = _string_list(group.get("prim_paths"))
    runtime_paths = _string_list(group.get("runtime_prim_paths")) or _string_list(
        group.get("inspection_prim_paths")
    )
    source_paths = _string_list(group.get("source_prim_paths")) or _string_list(
        group.get("source_paths")
    )
    group_path_space = str(group.get("path_space") or group.get("runtime_space") or "")
    result = {
        "family": str(group.get("family") or group.get("authoring_family") or "group"),
        "rationale": str(group.get("rationale") or "").strip(),
    }
    if path_space == "inspection":
        if not runtime_paths and group_path_space in {"inspection", "runtime"}:
            runtime_paths = prim_paths
        if not source_paths and not runtime_paths:
            source_paths = prim_paths
        if runtime_paths:
            result["runtime_prim_paths"] = _dedupe_strings(runtime_paths)
        if source_paths:
            result["source_prim_paths"] = _dedupe_strings(source_paths)
        if not runtime_paths and not source_paths:
            return None
        return result

    paths = prim_paths or source_paths
    if not paths:
        return None
    result["prim_paths"] = _dedupe_strings(paths)
    return result


def _ensure_material_assignment_artifacts(
    *,
    config: MaterialAssignConfig,
    run_dir: Path,
    request: dict[str, object],
    trace_writer: TraceWriter,
    child_output_path: Path,
    child_final_path: Path,
    child_returncode: int,
) -> bool:
    assignments_path = run_dir / "assignments.json"
    counts_path = run_dir / "api_operation_counts.json"
    visual_quality_path = run_dir / "visual_quality_assessment.json"
    summary_path = run_dir / "final_summary.md"
    existing_assignments = _load_json(
        assignments_path,
        default={},
        trace_writer=trace_writer,
        phase="artifact finalization",
    )
    if not isinstance(existing_assignments, dict):
        existing_assignments = {}
    embedded_visual_quality = existing_assignments.get("visual_quality_assessment")
    if not visual_quality_path.exists() and isinstance(embedded_visual_quality, dict):
        visual_quality_path.write_text(
            json.dumps(embedded_visual_quality, indent=2),
            encoding="utf-8",
        )

    required_paths = [assignments_path, counts_path, visual_quality_path, summary_path]
    if all(path.exists() for path in required_paths) and child_final_path.exists():
        return True

    raw_dir = run_dir / "raw"
    events = _load_jsonl(run_dir / "trace" / "events.jsonl")
    groups = _assignment_groups_from_run(run_dir, config, trace_writer=trace_writer)
    coverage, final_review, groups = _fallback_coverage_review_and_assignments(
        run_dir=run_dir,
        groups=groups,
        child_returncode=child_returncode,
    )
    final_render_paths = sorted((run_dir / "final_renders").glob("*.png"))
    visual_quality = _visual_quality_from_assignments_or_fallback(
        assignments=existing_assignments,
        final_review=final_review,
        final_render_paths=final_render_paths,
        reference_images=config.reference_images,
        reference_files=config.reference_files or [],
        child_returncode=child_returncode,
    )
    counts = _operation_counts(
        run_dir,
        events,
        groups,
        coverage=coverage,
        final_review=final_review,
        visual_quality=visual_quality,
    )

    generated: list[str] = []
    if groups and not assignments_path.exists():
        assignments = {
            "schema_version": "content-agents.assignments.v1",
            "session_id": _session_id_from_run(run_dir, events),
            "source_usd": str(config.usd_path),
            "library_path": str(config.materials_usd),
            "per_prim_material_assignment_count": sum(
                len(group["prim_paths"])
                for group in groups
                if _is_material_assignment_status(group.get("coverage_status"))
            ),
            "coverage": coverage,
            "assignments": groups,
            "final_review": final_review,
            "visual_quality_assessment": visual_quality,
            "uncertainty": _assignment_uncertainty(
                events=events,
                child_returncode=child_returncode,
                final_render_count=len(final_render_paths),
            ),
            "generated_by": "content-workflow-cli fallback finalizer",
        }
        assignments_path.write_text(json.dumps(assignments, indent=2), encoding="utf-8")
        generated.append(str(assignments_path))
    elif (
        assignments_path.exists()
        and "visual_quality_assessment" not in existing_assignments
    ):
        existing_assignments["visual_quality_assessment"] = visual_quality
        assignments_path.write_text(
            json.dumps(existing_assignments, indent=2),
            encoding="utf-8",
        )
        generated.append(str(assignments_path))

    if not counts_path.exists():
        counts_path.write_text(json.dumps(counts, indent=2), encoding="utf-8")
        generated.append(str(counts_path))

    if not visual_quality_path.exists():
        visual_quality_path.write_text(
            json.dumps(visual_quality, indent=2),
            encoding="utf-8",
        )
        generated.append(str(visual_quality_path))

    if groups and final_render_paths and not summary_path.exists():
        summary_path.write_text(
            _fallback_final_summary(
                run_dir=run_dir,
                groups=groups,
                final_render_paths=final_render_paths,
                counts=counts,
                coverage=coverage,
                final_review=final_review,
                visual_quality=visual_quality,
                child_returncode=child_returncode,
            ),
            encoding="utf-8",
        )
        generated.append(str(summary_path))

    if not child_final_path.exists() and summary_path.exists():
        child_final_path.write_text(
            "Child agent did not return a final message. "
            "content-workflow-cli synthesized the required final artifacts from "
            "observable Workbench trace data.\n\n"
            f"Run directory: {run_dir}\n"
            f"Assignments: {assignments_path}\n"
            f"Final renders: {run_dir / 'final_renders'}\n"
            f"Trace: {run_dir / 'trace'}\n",
            encoding="utf-8",
        )
        generated.append(str(child_final_path))

    complete = (
        assignments_path.exists() and counts_path.exists() and summary_path.exists()
    )
    if generated:
        trace_writer.write(
            "verification" if complete else "warning",
            phase="artifact finalization",
            summary=(
                "Generated missing final material-assignment artifacts from "
                "observable Workbench trace data."
                if complete
                else "Generated partial fallback artifacts; final contract remains incomplete."
            ),
            artifacts=generated,
            data={
                "complete": complete,
                "assignment_groups": len(groups),
                "final_renders": len(final_render_paths),
                "child_returncode": child_returncode,
                "raw_applied_override_groups": str(
                    raw_dir / "applied_override_groups.json"
                ),
            },
        )
    return complete


def _assignment_groups_from_run(
    run_dir: Path,
    config: MaterialAssignConfig,
    *,
    trace_writer: TraceWriter | None = None,
) -> list[dict[str, object]]:
    raw_groups = _load_json(
        run_dir / "raw" / "applied_override_groups.json",
        default=[],
        trace_writer=trace_writer,
        phase="artifact finalization",
    )
    material_bindings = _load_material_bindings(config.materials_yaml)
    groups: list[dict[str, object]] = []
    if isinstance(raw_groups, list):
        for record in raw_groups:
            if not isinstance(record, dict):
                continue
            material_name = str(record.get("material_name") or "").strip()
            prim_paths = _string_list(record.get("source_paths")) or _string_list(
                [record.get("prim_path")]
            )
            if not material_name or not prim_paths:
                continue
            material_path = str(
                record.get("material_path")
                or material_bindings.get(material_name)
                or _fallback_material_path(material_name)
            )
            groups.append(
                {
                    "family": str(record.get("name") or _slug(material_name)),
                    "material_name": material_name,
                    "material_path": material_path,
                    "prim_paths": prim_paths,
                    "rationale": str(
                        record.get("rationale")
                        or f"Recovered from material assignment trace for {material_name}."
                    ),
                }
            )
    if groups:
        return _dedupe_assignment_groups(groups)

    events = _load_jsonl(run_dir / "trace" / "events.jsonl")
    for event in events:
        if event.get("event_type") != "assignment":
            continue
        phase = str(event.get("phase") or "")
        if "override" not in phase:
            continue
        data = event.get("data")
        if not isinstance(data, dict):
            continue
        material_names = _string_list(data.get("material_names"))
        prim_paths = _string_list(data.get("prim_paths"))
        if not material_names or not prim_paths:
            continue
        material_name = material_names[0]
        groups.append(
            {
                "family": _slug(material_name),
                "material_name": material_name,
                "material_path": material_bindings.get(material_name)
                or _fallback_material_path(material_name),
                "prim_paths": prim_paths,
                "rationale": str(event.get("summary") or "Recovered from trace event."),
            }
        )
    return _dedupe_assignment_groups(groups)


def _fallback_coverage_review_and_assignments(
    *,
    run_dir: Path,
    groups: list[dict[str, object]],
    child_returncode: int,
) -> tuple[dict[str, object], dict[str, object], list[dict[str, object]]]:
    material_groups = []
    for group in groups:
        updated = dict(group)
        updated.setdefault("coverage_status", "material_assignment")
        material_groups.append(updated)

    candidate_paths = _visible_candidate_prim_paths(run_dir)
    material_paths = _unique_group_prim_paths(material_groups)
    if candidate_paths:
        covered_candidates = [
            candidate
            for candidate in candidate_paths
            if any(_paths_overlap(candidate, assigned) for assigned in material_paths)
        ]
        uncovered_candidates = [
            candidate
            for candidate in candidate_paths
            if candidate not in set(covered_candidates)
        ]
        candidate_count = len(candidate_paths)
        assigned_count = len(covered_candidates)
    else:
        uncovered_candidates = []
        candidate_count = len(material_paths)
        assigned_count = len(material_paths)

    ambiguous_count = len(uncovered_candidates)
    assignments = list(material_groups)
    if uncovered_candidates:
        assignments.append(
            {
                "family": "fallback-unreviewed-visible-candidates",
                "coverage_status": "ambiguous_unassigned",
                "material_name": None,
                "material_path": None,
                "prim_paths": uncovered_candidates,
                "rationale": (
                    "Fallback finalization could not verify material decisions for "
                    "these canonical material candidates before the child run stopped."
                ),
            }
        )

    coverage = {
        "candidate_visible_prim_count": candidate_count,
        "material_decision_prim_count": assigned_count + ambiguous_count,
        "material_assignment_prim_count": assigned_count,
        "preserved_existing_prim_count": 0,
        "ambiguous_unassigned_prim_count": ambiguous_count,
        "coverage_notes": (
            "Fallback coverage was reconstructed from Workbench trace events. "
            "Material assignment coverage is counted against visible /visuals/ mesh "
            "candidates when hierarchy data is available; uncovered candidates "
            "are marked ambiguous rather than silently omitted."
        ),
    }

    unresolved_issues: list[str] = []
    if child_returncode != 0:
        unresolved_issues.append(
            f"Child agent exited with return code {child_returncode}; wrapper synthesized final artifacts."
        )
    if ambiguous_count:
        unresolved_issues.append(
            f"{ambiguous_count} visible candidate prim(s) were not assigned or reviewed by the child agent."
        )

    final_review = {
        "issues_found": len(unresolved_issues),
        "issues_fixed": 0,
        "unresolved_issues": unresolved_issues,
        "review_notes": (
            "Fallback final review is conservative and only reflects observable "
            "trace artifacts; it does not replace a completed child-agent "
            "visual review pass."
        ),
    }
    return coverage, final_review, assignments


def _visible_candidate_prim_paths(run_dir: Path) -> list[str]:
    visible_candidates = _load_json(
        run_dir / "raw" / "visible_candidate_prims.json",
        default={},
    )
    if isinstance(visible_candidates, dict):
        path_space = str(visible_candidates.get("path_space") or "source")
        paths: list[str] = []
        for candidate in visible_candidates.get("candidates", []):
            if not isinstance(candidate, dict):
                continue
            if path_space == "inspection":
                path = candidate.get("runtime_path") or candidate.get("inspection_path")
            else:
                path = candidate.get("source_path")
            if isinstance(path, str) and path:
                paths.append(path)
        if paths:
            return _dedupe_strings(paths)

    hierarchy = _load_json(run_dir / "raw" / "hierarchy_flat.json", default=[])
    if not isinstance(hierarchy, list):
        return []

    visible_meshes: list[str] = []
    fallback_meshes: list[str] = []
    for item in hierarchy:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or "")
        if not path or item.get("active") is False:
            continue
        if str(item.get("type_name") or "") != "Mesh":
            continue
        if _path_has_component(path, "collisions") or _path_has_component(
            path, "Looks"
        ):
            continue
        fallback_meshes.append(path)
        if _path_has_component(path, "visuals"):
            visible_meshes.append(path)
    return _dedupe_strings(visible_meshes or fallback_meshes)


def _unique_group_prim_paths(groups: list[dict[str, object]]) -> list[str]:
    paths: list[str] = []
    for group in groups:
        paths.extend(_string_list(group.get("prim_paths")))
    return _dedupe_strings(paths)


def _is_material_assignment_status(value: object) -> bool:
    return str(value or "") == "material_assignment"


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _path_has_component(path: str, component: str) -> bool:
    return component in [part for part in path.split("/") if part]


def _paths_overlap(left: str, right: str) -> bool:
    return left == right or left.startswith(f"{right}/") or right.startswith(f"{left}/")


def _dedupe_assignment_groups(
    groups: list[dict[str, object]],
) -> list[dict[str, object]]:
    seen: set[tuple[str, tuple[str, ...]]] = set()
    deduped: list[dict[str, object]] = []
    for group in groups:
        material_name = str(group.get("material_name") or "")
        prim_paths = tuple(_string_list(group.get("prim_paths")))
        key = (material_name, prim_paths)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(group)
    return deduped


def _visual_quality_from_assignments_or_fallback(
    *,
    assignments: dict[str, object],
    final_review: dict[str, object],
    final_render_paths: list[Path],
    reference_images: list[Path],
    child_returncode: int,
    reference_files: list[Path] | None = None,
) -> dict[str, object]:
    embedded = assignments.get("visual_quality_assessment")
    if isinstance(embedded, dict):
        return embedded

    issues_found: list[dict[str, object]] = []
    unresolved_issues: list[str] = []
    if child_returncode != 0:
        unresolved_issues.append(
            f"Child agent exited with return code {child_returncode}; visual quality review was not completed by the child."
        )
    if not final_render_paths:
        unresolved_issues.append(
            "No final render images were available for visual quality assessment."
        )
    if not embedded and child_returncode == 0:
        unresolved_issues.append(
            "Child agent did not produce visual_quality_assessment.json or an embedded visual_quality_assessment object."
        )
    for issue in unresolved_issues:
        issues_found.append(
            {
                "severity": "high",
                "description": issue,
                "affected_prim_paths": [],
                "evidence_artifacts": [str(path) for path in final_render_paths],
                "expected_appearance": "Child-authored visual quality review.",
                "actual_appearance": "Visual quality review missing or incomplete.",
                "status": "unresolved",
            }
        )

    final_unresolved = final_review.get("unresolved_issues")
    if isinstance(final_unresolved, list):
        for issue in final_unresolved:
            unresolved_issues.append(str(issue))

    status = (
        "pass" if not issues_found and not unresolved_issues else "unresolved_issues"
    )
    return {
        "schema_version": "content-agents.visual-quality-assessment.v1",
        "status": status,
        "checked_views": [str(path) for path in final_render_paths],
        "reference_images": [str(path) for path in reference_images],
        "reference_files": [str(path) for path in reference_files or []],
        "issues_found": issues_found,
        "issues_fixed": [],
        "unresolved_issues": unresolved_issues,
        "assessment_notes": (
            "Fallback visual quality assessment is conservative. It records whether "
            "the child produced the required visual QA artifact; it does not replace "
            "a child-authored perceptual review of final renders against references."
        ),
    }


def _count_items(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return 1 if value else 0
    if isinstance(value, list | tuple | set):
        return len(value)
    return None


def _operation_counts(
    run_dir: Path,
    events: list[dict[str, Any]],
    groups: list[dict[str, object]],
    *,
    coverage: dict[str, object] | None = None,
    final_review: dict[str, object] | None = None,
    visual_quality: dict[str, object] | None = None,
) -> dict[str, object]:
    raw_dir = run_dir / "raw"
    api_operation_count = 0
    for event in events:
        data = event.get("data")
        if isinstance(data, dict):
            api_operation_count += len(_string_list(data.get("api_calls")))
    render_count = len(list(raw_dir.glob("*render*.json")))
    pick_count = len(list(raw_dir.glob("pick_*.json"))) + len(
        list(raw_dir.glob("grounding_pick_*.json"))
    )
    material_override_count = sum(
        len(_string_list(group.get("prim_paths")))
        for group in groups
        if _is_material_assignment_status(group.get("coverage_status"))
    )
    final_render_dir = run_dir / "final_renders"
    final_render_count = len(list(final_render_dir.glob("*.png"))) + len(
        list(final_render_dir.glob("*.gif"))
    )
    counts: dict[str, object] = {
        "schema_version": "content-agents.api-operation-counts.v1",
        "api_operation_count_total": api_operation_count,
        "render_count_total": render_count,
        "pick_calls": pick_count,
        "material_override_commands": material_override_count,
        "final_renders": final_render_count,
        "fallback_generated": True,
    }
    if coverage:
        counts["coverage_candidate_visible_prims"] = coverage.get(
            "candidate_visible_prim_count"
        )
        counts["coverage_material_decision_prims"] = coverage.get(
            "material_decision_prim_count"
        )
        counts["coverage_unassigned_visible_prims"] = coverage.get(
            "unassigned_visible_prim_count"
        )
        counts["coverage_missing_assignment_prims"] = coverage.get(
            "missing_assignment_prim_count"
        )
        counts["coverage_rejected_assignment_prims"] = coverage.get(
            "rejected_assignment_prim_count"
        )
    if final_review:
        counts["final_review_issues_found"] = _count_items(
            final_review.get("issues_found")
        )
        counts["final_review_issues_fixed"] = _count_items(
            final_review.get("issues_fixed")
        )
    if visual_quality:
        counts["visual_quality_issues_found"] = _count_items(
            visual_quality.get("issues_found")
        )
        counts["visual_quality_issues_fixed"] = _count_items(
            visual_quality.get("issues_fixed")
        )
    return counts


def _fallback_final_summary(
    *,
    run_dir: Path,
    groups: list[dict[str, object]],
    final_render_paths: list[Path],
    counts: dict[str, object],
    coverage: dict[str, object],
    final_review: dict[str, object],
    visual_quality: dict[str, object],
    child_returncode: int,
) -> str:
    lines = [
        "# Material Assignment Summary",
        "",
        "Generated by content-workflow-cli fallback finalization from observable Workbench trace data.",
        "",
        "## Coverage Summary",
        "",
        "| Status | Prim Count |",
        "| --- | ---: |",
        f"| Candidate visible prims | {coverage.get('candidate_visible_prim_count')} |",
        f"| Material decisions | {coverage.get('material_decision_prim_count')} |",
        f"| Material assignments | {coverage.get('material_assignment_prim_count')} |",
        f"| Preserved existing/default | {coverage.get('preserved_existing_prim_count')} |",
        f"| Ambiguous/unassigned | {coverage.get('ambiguous_unassigned_prim_count')} |",
        "",
        f"Coverage notes: {coverage.get('coverage_notes')}",
        "",
        "## Material Map",
        "",
    ]
    for group in groups:
        prims = ", ".join(_string_list(group.get("prim_paths")))
        lines.append(
            f"- [{group.get('coverage_status')}] {group.get('family')}: "
            f"{group.get('material_name')} "
            f"({group.get('material_path')}) -> {prims}"
        )
        lines.append(f"  Rationale: {group.get('rationale')}")
    lines.extend(["", "## Final Renders", ""])
    for path in final_render_paths:
        lines.append(f"- {path.relative_to(run_dir)}")
    lines.extend(
        [
            "",
            "## Operation Counts",
            "",
            f"- API operations recorded in trace: {counts.get('api_operation_count_total')}",
            f"- Render responses: {counts.get('render_count_total')}",
            f"- Pick calls: {counts.get('pick_calls')}",
            f"- Material override commands: {counts.get('material_override_commands')}",
            f"- Final renders: {counts.get('final_renders')}",
            "",
            "## Final Review",
            "",
            f"- Issues found: {final_review.get('issues_found')}",
            f"- Issues fixed: {final_review.get('issues_fixed')}",
        ]
    )
    unresolved = final_review.get("unresolved_issues")
    if isinstance(unresolved, list) and unresolved:
        lines.append("- Unresolved issues:")
        for issue in unresolved:
            lines.append(f"  - {issue}")
    lines.extend(
        [
            f"- Review notes: {final_review.get('review_notes')}",
            "",
            "## Visual Quality Assessment",
            "",
            f"- Status: {visual_quality.get('status')}",
            f"- Issues found: {_count_items(visual_quality.get('issues_found'))}",
            f"- Issues fixed: {_count_items(visual_quality.get('issues_fixed'))}",
            f"- Assessment notes: {visual_quality.get('assessment_notes')}",
            "",
            "## Uncertainty",
            "",
        ]
    )
    if child_returncode != 0:
        lines.append(
            f"- Child agent return code was {child_returncode}; final contract files were recovered by the wrapper."
        )
    lines.append(
        "- Some pixel picks may be ambiguous; assignments are based on successful material assignment trace and final renders."
    )
    lines.append("")
    return "\n".join(lines)


def _assignment_uncertainty(
    *,
    events: list[dict[str, Any]],
    child_returncode: int,
    final_render_count: int,
) -> list[str]:
    uncertainty: list[str] = []
    if child_returncode != 0:
        uncertainty.append(
            f"Child agent exited with return code {child_returncode}; wrapper recovered final artifacts from trace."
        )
    if final_render_count == 0:
        uncertainty.append("No final verification renders were found.")
    if any(_pick_event_has_no_prim_path(event) for event in events):
        uncertainty.append(
            "Some pixel-pick events returned no prim path; fallback used successful override records and hierarchy evidence."
        )
    return uncertainty


def _pick_event_has_no_prim_path(event: dict[str, Any]) -> bool:
    if event.get("event_type") != "pick":
        return False
    data = event.get("data")
    if not isinstance(data, dict) or "prim_paths" not in data:
        return False
    prim_paths = data["prim_paths"]
    if prim_paths is None:
        return True
    if isinstance(prim_paths, list | tuple):
        return len(prim_paths) == 0 or any(item is None for item in prim_paths)
    if isinstance(prim_paths, str):
        return prim_paths.strip() == ""
    return False


def _load_material_bindings(path: Path) -> dict[str, str]:
    bindings: dict[str, str] = {}
    if not path.exists():
        return bindings
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse material library YAML: {path}") from exc
    entries: object
    if isinstance(loaded, list):
        entries = loaded
    elif isinstance(loaded, dict):
        entries = loaded.get("entries", loaded.get("materials", []))
    else:
        raise ValueError(f"Material library YAML must be a mapping or list: {path}")
    if not isinstance(entries, list):
        raise ValueError(f"Material library YAML entries must be a list: {path}")
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        binding = entry.get("binding")
        if isinstance(name, str) and isinstance(binding, str):
            bindings[name] = binding
    return bindings


def _fallback_material_path(material_name: str) -> str:
    return f"/World/Looks/{_safe_usd_name(material_name)}"


def _safe_usd_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")
    if not name:
        return "Material"
    if name[0].isdigit():
        return f"Material_{name}"
    return name


def _session_id_from_run(run_dir: Path, events: list[dict[str, Any]]) -> str | None:
    session_id_path = run_dir / "raw" / "session_id.txt"
    if session_id_path.exists():
        value = session_id_path.read_text(encoding="utf-8").strip()
        if value:
            return value
    for name in [
        "session_create_response.json",
        "session_create_loaded.json",
        "session_create.json",
    ]:
        value = _find_json_key(
            _load_json(run_dir / "raw" / name, default={}), "session_id"
        )
        if isinstance(value, str) and value:
            return value
    for event in events:
        data = event.get("data")
        if isinstance(data, dict) and isinstance(data.get("session_id"), str):
            return str(data["session_id"])
    return None


def _find_json_key(value: object, key: str) -> object:
    if isinstance(value, dict):
        if key in value:
            return value[key]
        for child in value.values():
            found = _find_json_key(child, key)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_json_key(child, key)
            if found is not None:
                return found
    return None


def _string_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list | tuple):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def _load_json(
    path: Path,
    *,
    default: object,
    trace_writer: TraceWriter | None = None,
    phase: str = "json load",
) -> object:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        if trace_writer is not None:
            trace_writer.write(
                "warning",
                phase=phase,
                summary=(
                    "Failed to parse JSON artifact; using fallback data for "
                    "material-assignment finalization."
                ),
                artifacts=[str(path)],
                data={"path": str(path), "error": str(exc)},
            )
        return default


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            events.append(value)
    return events
