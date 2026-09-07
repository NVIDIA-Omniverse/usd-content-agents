# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public batch launcher for the agent-driven large-scene workflow."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import stat
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, Literal
from urllib.parse import urlparse

import yaml
from content_agent_workflows.common.artifacts import (
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_text,
    file_sha256,
    load_json,
    read_contained_artifact,
)
from content_agent_workflows.common.run_record import (
    WorkflowRunManifest,
    WorkflowRunRecorder,
)
from content_agent_workflows.large_scene import (
    LargeSceneRun,
    create_run,
    invalidate_from,
    load_run_state,
    validate_phase_handoff,
    verify_run_source_inputs,
)
from content_agent_workflows.large_scene.models import (
    LARGE_SCENE_RUN_SCHEMA_VERSION,
    LEGACY_LARGE_SCENE_RUN_SCHEMA_VERSION,
)
from content_agent_workflows.large_scene.state import _source_input_digest
from pydantic import BaseModel, ConfigDict, Field, model_validator
from world_understanding.utils.artifacts import fsync_directory
from world_understanding.utils.file_locking import exclusive_descriptor_lock

from .prompts import CONTROLLED_JSON_ARTIFACT_WRITE
from .runner import (
    CLAUDE_EXECUTION_CLI,
    CLAUDE_EXECUTION_SDK,
    CODEX_SANDBOX_WORKSPACE_WRITE,
    SCENE_BACKEND_USD_CLI,
    SUPPORTED_CLAUDE_EXECUTION_MODES,
    SUPPORTED_CODEX_SANDBOX_MODES,
    SUPPORTED_RUNNERS,
    SUPPORTED_SCENE_BACKENDS,
    ChildProcessInterrupted,
    ParentUsdCliCapability,
    UsdCliDaemonTeardownEvidence,
    UsdCliStagedInput,
    _append_child_runner_error,
    _chmod_private,
    _lexical_absolute_path,
    _prepare_contained_run_subdirectory,
    _reject_unsafe_run_links,
    _run_child_agent,
    _stage_usd_cli_input_tree,
    _staged_input_integrity_errors,
    _trace_path_summary,
    _write_run_cost_metrics,
    bind_parent_usd_cli_capability,
    parent_usd_cli_prompt_contract,
    start_parent_usd_cli_capability,
    stop_parent_usd_cli_capability_strict,
)
from .trace import TraceWriter, UnsafeRunArtifactError, build_trace, utc_now

LEGACY_SCENE_REQUEST_SCHEMA_VERSION = "content-agents.large-scene-request.v1"
SCENE_REQUEST_SCHEMA_VERSION = "content-agents.large-scene-request.v2"
SCENE_TERMINAL_VALIDATION_SCHEMA_VERSION = (
    "content-agents.large-scene-terminal-validation.v1"
)
SCENE_USD_CLI_TEARDOWN_SCHEMA_VERSION: Literal[
    "content-workflow-cli.scene-usd-cli-teardown.v1"
] = "content-workflow-cli.scene-usd-cli-teardown.v1"
SceneBackend = Literal["usd-cli"]
MAX_COLLECTION_RESULT_BYTES = 16 * 1024 * 1024
SCENE_LAUNCHER_POLICY_SCHEMA_VERSION = "content-agents.scene-launcher-policy.v1"
LEGACY_SCENE_ADOPTION_SCHEMA_VERSION = "content-agents.legacy-scene-adoption.v1"
LEGACY_SCENE_MIGRATION_INTENT_SCHEMA_VERSION = (
    "content-agents.legacy-scene-migration-intent.v1"
)
MAX_SCENE_LAUNCHER_POLICY_BYTES = 256 * 1024
MAX_LEGACY_SCENE_ADOPTION_BYTES = 256 * 1024
MAX_SCENE_REQUEST_BYTES = 4 * 1024 * 1024
MAX_SCENE_RUN_STATE_BYTES = 16 * 1024 * 1024
MAX_SCENE_MATERIALS_YAML_BYTES = 4 * 1024 * 1024
SCENE_RUN_INITIALIZATION_LOCK_TIMEOUT_SECONDS = 30.0
DEFAULT_SCENE_TOOL_TIMEOUT_SECONDS = 60.0
# Budget for the usd-cli OVRTX readiness/render probe. Deliberately not
# the generic scene-tool timeout (60 s default): a
# healthy local OVRTX daemon cold start may take up to 600 s
# (``OVRTX_DAEMON_START_TIMEOUT`` in ``usd_core/render/ovrtx.py``), so the
# probe gets the backend's own startup budget plus render headroom, matching
# the ``ensure_usd_cli_ovrtx_ready`` default.
USD_CLI_READINESS_TIMEOUT_SECONDS = 900.0
logger = logging.getLogger(__name__)


class SceneUsdCliTeardownReceipt(BaseModel):
    """Terminal release evidence for one parent-owned scene session."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["content-workflow-cli.scene-usd-cli-teardown.v1"] = (
        SCENE_USD_CLI_TEARDOWN_SCHEMA_VERSION
    )
    created_at: str
    launch_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    run_id: str = Field(min_length=1)
    session_identity_path: str
    session_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["released", "failed"]
    boundary: Literal["completed", "failed", "cancelled", "incomplete"]
    child_returncode: int
    interrupted: bool
    process_released: bool | None = None
    descendants_released: bool | None = None
    sessions_released: bool | None = None
    listener_released: bool | None = None
    daemon_leases_released: bool | None = None
    state_directory_released: bool | None = None
    listener_host: str | None = None
    listener_port: int | None = Field(default=None, ge=1, le=65535)
    daemon_log_path: str | None = None
    daemon_log_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    errors: list[str] = Field(default_factory=list)

    @model_validator(mode="after")  # type: ignore[misc]
    def validate_release_status(self) -> SceneUsdCliTeardownReceipt:
        if (self.daemon_log_path is None) != (self.daemon_log_sha256 is None):
            raise ValueError("daemon log path and digest must appear together")
        released = all(
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
        if self.status == "released" and (not released or self.errors):
            raise ValueError("released teardown requires complete clean evidence")
        if self.status == "failed" and not self.errors:
            raise ValueError("failed teardown requires a concrete error")
        return self


class SceneReferencesRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    directories: list[str] = Field(default_factory=list)
    images: list[str] = Field(default_factory=list)
    files: list[str] = Field(default_factory=list)


class SceneTaskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    domain: str
    scene_backend: SceneBackend
    scene_session_scope: Literal["per_asset"]
    inputs: dict[str, Any] = Field(default_factory=dict)
    policy: dict[str, Any] = Field(default_factory=dict)


class SceneRuntimeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    runner: str
    scene_backend: SceneBackend
    model: str | None = None
    model_reasoning_effort: str | None = None
    scene_tool_timeout_seconds: float
    child_timeout_seconds: float
    codex_base_url: str | None = None
    codex_sandbox_mode: str
    codex_config: dict[str, Any] = Field(default_factory=dict)
    claude_config: dict[str, Any] = Field(default_factory=dict)
    claude_permission_mode: str
    claude_max_turns: int | None = None
    claude_execution_mode: str = CLAUDE_EXECUTION_SDK


class SceneRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[
        LEGACY_SCENE_REQUEST_SCHEMA_VERSION,
        SCENE_REQUEST_SCHEMA_VERSION,
    ] = SCENE_REQUEST_SCHEMA_VERSION
    created_at: str
    workflow: str = "scene.run"
    dry_run: bool
    run_id: str
    run_dir: str
    run_state: str
    repository_root: str
    agent_workspace: str
    child_workspace: str | None = None
    source_scene: str
    source_scene_original: str | None = None
    requested_tasks: list[str]
    references: SceneReferencesRequest
    additional_instructions: str | None = None
    additional_instruction_sources: list[str] = Field(default_factory=list)
    tasks: list[SceneTaskRequest]
    decomposition: dict[str, Any]
    collection: dict[str, Any]
    runtime: SceneRuntimeRequest

    @model_validator(mode="before")
    @classmethod
    def restore_legacy_backend_contract(cls, value: object) -> object:
        """Upgrade v1 requests to the supported usd-cli/per-asset contract."""

        if not isinstance(value, dict):
            return value
        if value.get("schema_version") != LEGACY_SCENE_REQUEST_SCHEMA_VERSION:
            return value

        restored = dict(value)
        runtime = restored.get("runtime")
        if isinstance(runtime, dict):
            restored_runtime = dict(runtime)
            legacy_timeout = restored_runtime.pop("workbench_timeout_seconds", None)
            if "scene_tool_timeout_seconds" not in restored_runtime:
                restored_runtime["scene_tool_timeout_seconds"] = (
                    legacy_timeout
                    if isinstance(legacy_timeout, int | float)
                    and not isinstance(legacy_timeout, bool)
                    and legacy_timeout > 0
                    else DEFAULT_SCENE_TOOL_TIMEOUT_SECONDS
                )
            for retired_field in (
                "workbench_url",
                "start_workbench",
                "keep_workbench",
            ):
                restored_runtime.pop(retired_field, None)
            restored_runtime["scene_backend"] = restored_runtime.get(
                "scene_backend", SCENE_BACKEND_USD_CLI
            )
            restored["runtime"] = restored_runtime
        tasks = restored.get("tasks")
        if isinstance(tasks, list):
            restored["tasks"] = [
                {
                    **task,
                    "scene_backend": task.get("scene_backend", SCENE_BACKEND_USD_CLI),
                    "scene_session_scope": task.get("scene_session_scope", "per_asset"),
                }
                if isinstance(task, dict)
                else task
                for task in tasks
            ]
        return restored

    @model_validator(mode="after")
    def validate_backend_contract(self) -> SceneRunRequest:
        mismatched_tasks = [
            task.domain
            for task in self.tasks
            if task.scene_backend != self.runtime.scene_backend
        ]
        if mismatched_tasks:
            raise ValueError(
                "Task scene_backend must match runtime.scene_backend for: "
                + ", ".join(mismatched_tasks)
            )
        return self


class SceneLauncherPolicy(BaseModel):
    """Parent-owned integrity policy for a resumable scene request."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[SCENE_LAUNCHER_POLICY_SCHEMA_VERSION] = (
        SCENE_LAUNCHER_POLICY_SCHEMA_VERSION
    )
    run_dir: str
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    legacy_adoption_receipt_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )


class LegacySceneAdoptionReceipt(BaseModel):
    """Operator-authorized trust bootstrap for one genuine pre-policy v1 run."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[LEGACY_SCENE_ADOPTION_SCHEMA_VERSION] = (
        LEGACY_SCENE_ADOPTION_SCHEMA_VERSION
    )
    adopted_at: str
    authorization: Literal["externally_supplied_request_sha256"] = (
        "externally_supplied_request_sha256"
    )
    run_dir: str
    request_schema_before: Literal[LEGACY_SCENE_REQUEST_SCHEMA_VERSION] = (
        LEGACY_SCENE_REQUEST_SCHEMA_VERSION
    )
    request_schema_after: Literal[SCENE_REQUEST_SCHEMA_VERSION] = (
        SCENE_REQUEST_SCHEMA_VERSION
    )
    request_sha256_before: str = Field(pattern=r"^[0-9a-f]{64}$")
    request_sha256_after: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_state_schema_before: Literal[LEGACY_LARGE_SCENE_RUN_SCHEMA_VERSION] = (
        LEGACY_LARGE_SCENE_RUN_SCHEMA_VERSION
    )
    run_state_schema_after: Literal[LARGE_SCENE_RUN_SCHEMA_VERSION] = (
        LARGE_SCENE_RUN_SCHEMA_VERSION
    )
    run_state_sha256_before: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_state_sha256_after: str = Field(pattern=r"^[0-9a-f]{64}$")
    legacy_request_artifact: str
    legacy_run_state_artifact: str
    migration_intent_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_scene: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_input_digest_before: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_input_digest_after: str = Field(pattern=r"^[0-9a-f]{64}$")
    scene_backend: Literal["usd-cli"] = "usd-cli"


class LegacySceneMigrationIntent(BaseModel):
    """Parent-owned journal created before replacing either legacy artifact."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[LEGACY_SCENE_MIGRATION_INTENT_SCHEMA_VERSION] = (
        LEGACY_SCENE_MIGRATION_INTENT_SCHEMA_VERSION
    )
    prepared_at: str
    authorization: Literal["externally_supplied_request_sha256"] = (
        "externally_supplied_request_sha256"
    )
    run_dir: str
    request_schema_before: Literal[LEGACY_SCENE_REQUEST_SCHEMA_VERSION] = (
        LEGACY_SCENE_REQUEST_SCHEMA_VERSION
    )
    request_schema_after: Literal[SCENE_REQUEST_SCHEMA_VERSION] = (
        SCENE_REQUEST_SCHEMA_VERSION
    )
    request_sha256_before: str = Field(pattern=r"^[0-9a-f]{64}$")
    request_sha256_after: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_state_schema_before: Literal[LEGACY_LARGE_SCENE_RUN_SCHEMA_VERSION] = (
        LEGACY_LARGE_SCENE_RUN_SCHEMA_VERSION
    )
    run_state_schema_after: Literal[LARGE_SCENE_RUN_SCHEMA_VERSION] = (
        LARGE_SCENE_RUN_SCHEMA_VERSION
    )
    run_state_sha256_before: str = Field(pattern=r"^[0-9a-f]{64}$")
    legacy_request_artifact: str
    legacy_run_state_artifact: str
    source_scene: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_input_digest_before: str = Field(pattern=r"^[0-9a-f]{64}$")
    scene_backend: Literal["usd-cli"] = "usd-cli"


@dataclass(frozen=True)
class LegacySceneMigrationPlan:
    request_bytes_before: bytes
    request: SceneRunRequest
    run_state_bytes_before: bytes
    run_state: LargeSceneRun
    source_scene: Path
    source_sha256: str
    source_input_digest_before: str


class SceneLauncherPolicyError(RuntimeError):
    """Raised when a scene run cannot satisfy its launcher integrity policy."""


class SceneTerminalValidation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCENE_TERMINAL_VALIDATION_SCHEMA_VERSION
    checked_at: str
    valid: bool
    current_phase: str | None
    phase_statuses: dict[str, str]
    errors: list[str]
    final_handoff: dict[str, Any] | None = None


@dataclass(frozen=True)
class SceneRunConfig:
    child_launch_profile: ClassVar[str] = "scene.run"
    repo_root: Path
    usd_path: Path
    requested_tasks: list[str]
    reference_images: list[Path] = field(default_factory=list)
    reference_files: list[Path] = field(default_factory=list)
    reference_directories: list[Path] = field(default_factory=list)
    materials_yaml: Path | None = None
    materials_usd: Path | None = None
    material_candidate_space: str = "source"
    respect_existing_material_bindings: bool = False
    additional_instructions: str | None = None
    additional_instruction_sources: list[Path] = field(default_factory=list)
    output_dir: Path | None = None
    run_id: str | None = None
    runner: str = "codex"
    model: str | None = None
    model_reasoning_effort: str | None = None
    codex_base_url: str | None = None
    codex_sandbox_mode: str = CODEX_SANDBOX_WORKSPACE_WRITE
    codex_config: dict[str, object] | None = None
    claude_config: dict[str, object] | None = None
    claude_permission_mode: str = "default"
    claude_max_turns: int | None = None
    claude_execution_mode: str = CLAUDE_EXECUTION_SDK
    scene_tool_timeout_seconds: float = DEFAULT_SCENE_TOOL_TIMEOUT_SECONDS
    child_timeout_seconds: float = 1800.0
    dry_run: bool = False
    agent_workspace: Path | None = None
    agent_cwd: Path | None = None
    parent_usd_cli_session_identity: Path | None = None
    parent_usd_cli_session_identity_sha256: str | None = None
    # Set by bind_parent_usd_cli_capability after the parent daemon is attested.
    usd_cli_server_url: str | None = None


@dataclass(frozen=True)
class SceneRunResult:
    run_dir: Path
    request_path: Path
    run_state_path: Path
    prompt_path: Path
    child_output_path: Path
    child_final_path: Path
    terminal_validation_path: Path | None
    returncode: int
    completed: bool
    trace_paths: dict[str, str]

    @property
    def workflow_run_manifest_path(self) -> Path:
        """Return the workflow-owned durable run manifest."""

        return self.run_dir / "workflow_run_manifest.json"


def run_scene_workflow(config: SceneRunConfig) -> SceneRunResult:
    """Create and optionally execute a fresh three-phase large-scene run."""

    _validate_scene_config(config)
    with _prepare_scene_run_dir(config) as (run_id, run_dir):
        config = replace(
            config,
            agent_cwd=_confined_child_workspace(config.agent_cwd, run_dir=run_dir),
        )
        request_path = run_dir / "request.json"
        run_state_path = run_dir / "large_scene_run.json"
        prompt_path = run_dir / "agent_prompt.md"
        child_output_path = run_dir / "child-output.log"
        child_final_path = run_dir / "child-final.md"

        if (
            request_path.exists()
            or request_path.is_symlink()
            or run_state_path.exists()
            or run_state_path.is_symlink()
        ):
            raise FileExistsError(f"Large-scene run already exists: {run_dir}")

        staged_inputs_root = _prepare_scene_staged_inputs_root(run_dir)
        staged_source = _stage_usd_cli_input_tree(
            label="source",
            source_usd_path=config.usd_path,
            run_dir=run_dir,
            staging_base=staged_inputs_root,
        )
        staged_material_library = (
            _stage_usd_cli_input_tree(
                label="material_library",
                source_usd_path=config.materials_usd,
                run_dir=run_dir,
                staging_base=staged_inputs_root,
            )
            if config.materials_usd is not None
            else None
        )
        staged_materials_yaml = (
            _stage_scene_materials_yaml(
                source_path=config.materials_yaml,
                staged_material_library=staged_material_library,
                run_dir=run_dir,
            )
            if config.materials_yaml is not None and staged_material_library is not None
            else None
        )
        _seal_scene_staged_inputs(staged_inputs_root)
        request = _build_scene_request(
            config,
            run_id=run_id,
            run_dir=run_dir,
            run_state_path=run_state_path,
            staged_source=staged_source,
            staged_material_library=staged_material_library,
            staged_materials_yaml=staged_materials_yaml,
        )
        _write_request(run_dir, request_path, request)
        run_recorder = _start_scene_run_recorder(
            run_dir=run_dir,
            request=request,
            source_path=config.usd_path,
            resume=False,
        )
        # The run recorder republishes request.json in its canonical form, so
        # bind the immutable launcher policy to those final bytes.
        _write_scene_launcher_policy(
            run_dir,
            request_bytes=_read_scene_protected_file(
                request_path,
                label="protected scene request",
                max_bytes=MAX_SCENE_REQUEST_BYTES,
            ),
        )
        input_artifacts = _request_input_artifacts(
            config,
            request_path,
            staged_inputs=[
                staged_input.manifest_path
                for staged_input in (staged_source, staged_material_library)
                if staged_input is not None
            ],
        )
        if staged_materials_yaml is not None:
            input_artifacts.append(staged_materials_yaml)
        try:
            create_run(
                run_state_path,
                run_id=run_id,
                source_scene=Path(request.source_scene),
                requested_tasks=config.requested_tasks,
                request_artifact_paths=input_artifacts,
                additional_instructions=config.additional_instructions,
                scene_backend=SCENE_BACKEND_USD_CLI,
                actor="content-workflow-cli",
            )

            prompt = _build_scene_agent_prompt(
                request_path=request_path,
                run_state_path=run_state_path,
                run_dir=run_dir,
                resume=False,
                scene_backend=SCENE_BACKEND_USD_CLI,
            )
            atomic_write_text(prompt_path, prompt, within=run_dir)
        except Exception as exc:
            _finalize_scene_run_manifest(
                recorder=run_recorder,
                run_dir=run_dir,
                prompt_path=prompt_path,
                terminal_path=None,
                status="fail",
                failure={
                    "code": "scene_run_setup_failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            raise

        trace_writer = TraceWriter(run_dir)
        trace_writer.write(
            "run_created",
            phase="setup",
            summary="Created a large-scene batch request and durable phase state.",
            artifacts=[str(request_path), str(run_state_path), str(prompt_path)],
            data={
                "run_id": run_id,
                "workflow": request.workflow,
                "scene_backend": SCENE_BACKEND_USD_CLI,
            },
        )

    if config.dry_run:
        trace_paths = build_trace(run_dir)
        _finalize_scene_run_manifest(
            recorder=run_recorder,
            run_dir=run_dir,
            prompt_path=prompt_path,
            terminal_path=None,
            status="blocked",
            failure={
                "code": "dry_run",
                "message": "The batch workflow was prepared but not executed.",
            },
        )
        return SceneRunResult(
            run_dir=run_dir,
            request_path=request_path,
            run_state_path=run_state_path,
            prompt_path=prompt_path,
            child_output_path=child_output_path,
            child_final_path=child_final_path,
            terminal_validation_path=None,
            returncode=0,
            completed=False,
            trace_paths=_trace_path_summary(trace_paths),
        )

    return _execute_scene_agent(
        config=config,
        request=request,
        request_path=request_path,
        run_state_path=run_state_path,
        prompt=prompt,
        prompt_path=prompt_path,
        child_output_path=child_output_path,
        child_final_path=child_final_path,
        trace_writer=trace_writer,
        bridge_artifact_prefix="scene_run",
        run_recorder=run_recorder,
    )


def resume_scene_workflow(
    run_dir: Path,
    *,
    dry_run: bool = False,
    adopt_legacy_request_sha256: str | None = None,
) -> SceneRunResult:
    """Resume a previously prepared or interrupted large-scene run."""

    lexical_run_dir = run_dir.expanduser()
    _reject_unsafe_run_links(lexical_run_dir)
    resolved_run_dir = lexical_run_dir.resolve()
    for directory_name in ("raw", "trace"):
        _prepare_contained_run_subdirectory(
            resolved_run_dir,
            resolved_run_dir / directory_name,
            create=False,
        )
    request_path = resolved_run_dir / "request.json"
    run_state_path = resolved_run_dir / "large_scene_run.json"
    if not request_path.is_file():
        raise FileNotFoundError(f"Large-scene request does not exist: {request_path}")
    if not run_state_path.is_file():
        raise FileNotFoundError(
            f"Large-scene run state does not exist: {run_state_path}"
        )

    explicit_legacy_adoption = adopt_legacy_request_sha256 is not None
    if explicit_legacy_adoption:
        assert adopt_legacy_request_sha256 is not None
        run_recorder = _adopt_legacy_scene_run(
            run_dir=resolved_run_dir,
            request_path=request_path,
            run_state_path=run_state_path,
            expected_request_sha256=adopt_legacy_request_sha256,
        )
        request_bytes = _verify_scene_launcher_policy(
            resolved_run_dir,
            request_path=request_path,
        )
    else:
        if _scene_launcher_policy_identity(
            resolved_run_dir
        ) is None and _looks_like_recognized_legacy_scene_run(
            request_path=request_path,
            run_state_path=run_state_path,
        ):
            raise SceneLauncherPolicyError(
                "Recognized a pre-policy v1 scene run, but ordinary resume will "
                "not trust child-writable legacy state. Supply the exact request "
                "digest through --adopt-legacy-request-sha256 to migrate it."
            )
        request_bytes = _verify_scene_launcher_policy(
            resolved_run_dir,
            request_path=request_path,
        )
    request = SceneRunRequest.model_validate_json(request_bytes)
    run_state = load_run_state(run_state_path)
    if request.schema_version != SCENE_REQUEST_SCHEMA_VERSION:
        raise SceneLauncherPolicyError(
            "Scene resume requires a current v2 request. A genuine pre-policy v1 "
            "run must be explicitly migrated with "
            "--adopt-legacy-request-sha256."
        )
    if run_state.schema_version != LARGE_SCENE_RUN_SCHEMA_VERSION:
        raise SceneLauncherPolicyError(
            "Scene resume requires current v2 run state. A genuine pre-policy v1 "
            "run must be explicitly migrated with "
            "--adopt-legacy-request-sha256."
        )
    if run_state.scene_backend != request.runtime.scene_backend:
        raise ValueError(
            "Large-scene request and run-state scene_backend values differ: "
            f"{request.runtime.scene_backend!r} != {run_state.scene_backend!r}"
        )
    if Path(request.run_dir).resolve() != resolved_run_dir:
        raise ValueError(
            "Resolved request run_dir does not match --run-dir: "
            f"{request.run_dir} != {resolved_run_dir}"
        )
    config = _config_from_request(request, dry_run=dry_run)
    _validate_scene_config(config)
    if not explicit_legacy_adoption:
        run_recorder = _start_scene_run_recorder(
            run_dir=resolved_run_dir,
            request=request,
            source_path=config.usd_path,
            resume=True,
        )

    prompt_path = resolved_run_dir / "agent_resume_prompt.md"
    child_output_path = resolved_run_dir / "child-resume-output.log"
    child_final_path = resolved_run_dir / "child-resume-final.md"
    prompt = _build_scene_agent_prompt(
        request_path=request_path,
        run_state_path=run_state_path,
        run_dir=resolved_run_dir,
        resume=True,
        scene_backend=request.runtime.scene_backend,
    )
    atomic_write_text(prompt_path, prompt, within=resolved_run_dir)
    trace_writer = TraceWriter(resolved_run_dir)
    trace_writer.write(
        "run_resume_requested",
        phase="setup",
        summary="Prepared a child agent to resume durable large-scene state.",
        artifacts=[str(request_path), str(run_state_path), str(prompt_path)],
        data={
            "run_id": request.run_id,
            "dry_run": dry_run,
            "scene_backend": request.runtime.scene_backend,
        },
    )

    terminal = _validate_terminal_state(run_state_path)
    teardown_receipt_path, teardown_errors = _recorded_scene_usd_cli_teardown(
        resolved_run_dir,
        request=request,
        recorder=run_recorder,
        require_receipt=terminal.valid,
    )
    if teardown_errors:
        terminal = terminal.model_copy(
            update={
                "valid": False,
                "errors": [*terminal.errors, *teardown_errors],
            }
        )
    if terminal.valid or teardown_errors:
        terminal_path = _write_terminal_validation(resolved_run_dir, terminal)
        trace_writer.write(
            "run_already_complete" if terminal.valid else "run_release_unproven",
            phase="validation",
            summary=(
                "The large-scene run was already complete and valid."
                if terminal.valid
                else "The terminal scene state has no valid usd-cli release proof."
            ),
            artifacts=[
                str(path)
                for path in (terminal_path, teardown_receipt_path)
                if path is not None
            ],
        )
        trace_paths = build_trace(resolved_run_dir)
        manifest = _finalize_scene_run_manifest(
            recorder=run_recorder,
            run_dir=resolved_run_dir,
            prompt_path=prompt_path,
            terminal_path=terminal_path,
            teardown_receipt_path=teardown_receipt_path,
            status="pass" if terminal.valid else "fail",
            failure=(
                None
                if terminal.valid
                else {
                    "code": "scene_usd_cli_release_unproven",
                    "terminal_errors": terminal.errors,
                }
            ),
        )
        return SceneRunResult(
            run_dir=resolved_run_dir,
            request_path=request_path,
            run_state_path=run_state_path,
            prompt_path=prompt_path,
            child_output_path=child_output_path,
            child_final_path=child_final_path,
            terminal_validation_path=terminal_path,
            returncode=0 if manifest.status == "pass" else 1,
            completed=manifest.status == "pass",
            trace_paths=_trace_path_summary(trace_paths),
        )

    if dry_run:
        trace_paths = build_trace(resolved_run_dir)
        _finalize_scene_run_manifest(
            recorder=run_recorder,
            run_dir=resolved_run_dir,
            prompt_path=prompt_path,
            terminal_path=None,
            status="blocked",
            failure={
                "code": "dry_run",
                "message": "The resumable batch was inspected but not executed.",
            },
        )
        return SceneRunResult(
            run_dir=resolved_run_dir,
            request_path=request_path,
            run_state_path=run_state_path,
            prompt_path=prompt_path,
            child_output_path=child_output_path,
            child_final_path=child_final_path,
            terminal_validation_path=None,
            returncode=0,
            completed=False,
            trace_paths=_trace_path_summary(trace_paths),
        )

    _prepare_resumable_phase(run_state_path)
    return _execute_scene_agent(
        config=config,
        request=request,
        request_path=request_path,
        run_state_path=run_state_path,
        prompt=prompt,
        prompt_path=prompt_path,
        child_output_path=child_output_path,
        child_final_path=child_final_path,
        trace_writer=trace_writer,
        bridge_artifact_prefix="scene_resume",
        run_recorder=run_recorder,
    )


def _scene_parent_session_integrity_errors(
    run_dir: Path,
    *,
    capability: ParentUsdCliCapability,
) -> tuple[str, list[str]]:
    """Re-attest the parent handoff after child access has ended."""

    errors: list[str] = []
    match = re.fullmatch(
        r"parent_usd_cli_session_([A-Za-z0-9][A-Za-z0-9._-]*)[.]json",
        capability.identity_path.name,
    )
    if match is None:
        launch_id = (
            "invalid-"
            + hashlib.sha256(str(capability.identity_path).encode()).hexdigest()[:16]
        )
        errors.append("parent session identity path is malformed")
    else:
        launch_id = match.group(1)
    try:
        observed = read_contained_artifact(
            run_dir,
            capability.identity_path,
            max_bytes=MAX_SCENE_REQUEST_BYTES,
        )
        if observed.sha256 != capability.identity_sha256:
            raise ValueError("identity digest changed")
    except Exception as exc:  # noqa: BLE001 - any custody loss fails closed
        errors.append(f"parent session attestation: {type(exc).__name__}: {exc}")
    return launch_id, errors


def _recorded_scene_usd_cli_teardown(
    run_dir: Path,
    *,
    request: SceneRunRequest,
    recorder: WorkflowRunRecorder,
    require_receipt: bool,
) -> tuple[Path | None, list[str]]:
    """Validate prior release custody before completion or another launch."""

    record = next(
        (
            artifact
            for artifact in recorder.manifest.artifacts
            if artifact.logical_name == "usd_cli_teardown"
        ),
        None,
    )
    if record is None:
        residual = run_dir / ".usd-cli"
        if residual.exists() or residual.is_symlink():
            return None, ["scene run retains .usd-cli without a teardown receipt"]
        explicitly_adopted_legacy = (
            "legacy_adoption_receipt" in recorder.manifest.required_artifacts
        )
        backend_probe_path = run_dir / "raw" / "scene_backend_probe.json"
        prior_launch_evidence = (
            any(
                artifact.logical_name == "backend_probe"
                for artifact in recorder.manifest.artifacts
            )
            or backend_probe_path.exists()
            or backend_probe_path.is_symlink()
        )
        if prior_launch_evidence or (require_receipt and not explicitly_adopted_legacy):
            return None, ["modern scene run is missing its usd-cli teardown receipt"]
        return None, []

    path = run_dir / record.path
    try:
        observed = read_contained_artifact(
            run_dir,
            path,
            max_bytes=MAX_SCENE_REQUEST_BYTES,
            capture_bytes=True,
        )
        if observed.sha256 != record.sha256:
            raise ValueError("receipt digest differs from the workflow manifest")
        receipt = SceneUsdCliTeardownReceipt.model_validate_json(observed.data)
        expected_path = (
            run_dir / "raw" / f"scene_usd_cli_teardown_{receipt.launch_id}.json"
        )
        if observed.path != expected_path:
            raise ValueError("receipt path and launch identity differ")
        if receipt.run_id != request.run_id:
            raise ValueError("receipt belongs to another scene run")
        session_identity = read_contained_artifact(
            run_dir,
            Path(receipt.session_identity_path),
            max_bytes=MAX_SCENE_REQUEST_BYTES,
        )
        if session_identity.sha256 != receipt.session_identity_sha256:
            raise ValueError("receipt session identity digest changed")
        if receipt.status != "released":
            raise ValueError("latest scene usd-cli teardown was not released")
    except (OSError, ValueError) as exc:
        return path, [f"usd-cli teardown receipt: {exc}"]
    return path, []


def _write_scene_usd_cli_teardown_receipt(
    *,
    run_dir: Path,
    request: SceneRunRequest,
    capability: ParentUsdCliCapability,
    release: UsdCliDaemonTeardownEvidence | None,
    boundary: Literal["completed", "failed", "cancelled", "incomplete"],
    child_returncode: int,
    interrupted: bool,
    launch_id: str,
    errors: list[str],
) -> Path:
    """Seal one scene launch's strict resource-release evidence."""

    receipt = SceneUsdCliTeardownReceipt(
        created_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        launch_id=launch_id,
        run_id=request.run_id,
        session_identity_path=str(capability.identity_path),
        session_identity_sha256=capability.identity_sha256,
        status="released" if release is not None and not errors else "failed",
        boundary=boundary,
        child_returncode=child_returncode,
        interrupted=interrupted,
        process_released=(release.process_released if release is not None else None),
        descendants_released=(
            release.descendants_released if release is not None else None
        ),
        sessions_released=(release.sessions_released if release is not None else None),
        listener_released=(release.listener_released if release is not None else None),
        daemon_leases_released=(
            release.daemon_leases_released if release is not None else None
        ),
        state_directory_released=(
            release.state_directory_released if release is not None else None
        ),
        listener_host=release.host if release is not None else None,
        listener_port=release.port if release is not None else None,
        daemon_log_path=(
            str(release.daemon_log_path)
            if release is not None and release.daemon_log_path is not None
            else None
        ),
        daemon_log_sha256=(release.daemon_log_sha256 if release is not None else None),
        errors=errors,
    )
    path = run_dir / "raw" / f"scene_usd_cli_teardown_{launch_id}.json"
    atomic_write_json(path, receipt, within=run_dir)
    return path


def _retry_scene_terminal_action[TerminalActionResult](
    action: Callable[[], TerminalActionResult],
    *,
    on_interrupt: Callable[[], None],
) -> tuple[TerminalActionResult | None, BaseException | None]:
    """Retry one post-child terminal action after a first Ctrl-C."""

    try:
        return action(), None
    except KeyboardInterrupt:
        on_interrupt()
        try:
            return action(), None
        except BaseException as exc:
            return None, exc


def _execute_scene_agent(
    *,
    config: SceneRunConfig,
    request: SceneRunRequest,
    request_path: Path,
    run_state_path: Path,
    prompt: str,
    prompt_path: Path,
    child_output_path: Path,
    child_final_path: Path,
    trace_writer: TraceWriter,
    bridge_artifact_prefix: str,
    run_recorder: WorkflowRunRecorder,
) -> SceneRunResult:
    run_started = time.monotonic()
    parent_usd_cli_capability: ParentUsdCliCapability | None = None
    daemon_release: UsdCliDaemonTeardownEvidence | None = None
    teardown_receipt_path: Path | None = None
    terminal: SceneTerminalValidation | None = None
    interrupted = False
    child_returncode = 2
    run_dir = Path(request.run_dir).expanduser().resolve(strict=True)
    backend_probe_path = run_dir / "raw" / "scene_backend_probe.json"

    def mark_terminal_interrupt() -> None:
        nonlocal child_returncode, interrupted
        interrupted = True
        child_returncode = 130

    try:
        required_capabilities = (
            ("appearance.clear.v1",)
            if "material" in config.requested_tasks
            and not config.respect_existing_material_bindings
            else ()
        )
        parent_usd_cli_capability = start_parent_usd_cli_capability(
            config=config,
            run_dir=run_dir,
            workflow="scene.run",
            session_workflow="large-scene",
            input_roots=_scene_parent_input_roots(
                run_dir,
                request=request,
            ),
            initial_scene=Path(request.source_scene),
            required_capabilities=required_capabilities,
            timeout_seconds=USD_CLI_READINESS_TIMEOUT_SECONDS,
        )
        readiness = parent_usd_cli_capability.readiness
        telemetry_route = parent_usd_cli_capability.route
        atomic_write_json(
            backend_probe_path,
            {
                "schema_version": "content-agents.scene-backend-probe.v1",
                "status": "ready",
                "scene_backend": SCENE_BACKEND_USD_CLI,
                "renderer": "ovrtx",
                "usd_cli_version": readiness.version,
                "telemetry_route": telemetry_route.metadata(),
                "probe_artifact": (
                    str(readiness.artifact_path)
                    if readiness.artifact_path is not None
                    else None
                ),
                "probe": getattr(readiness, "probe", {}),
            },
            within=run_dir,
        )
        execution_prompt = prompt.rstrip() + parent_usd_cli_prompt_contract(
            parent_usd_cli_capability
        )
        atomic_write_text(prompt_path, execution_prompt, within=run_dir)
        execution_config = bind_parent_usd_cli_capability(
            config, parent_usd_cli_capability
        )
        trace_writer.write(
            "usd_cli_ready",
            phase="setup",
            summary="Verified the selected usd-cli backend and OVRTX renderer.",
            artifacts=(
                [str(readiness.artifact_path)]
                if readiness.artifact_path is not None
                else []
            ),
            data={
                "scene_backend": SCENE_BACKEND_USD_CLI,
                "usd_cli_version": readiness.version,
                "telemetry_route": telemetry_route.metadata(),
            },
        )
        trace_writer.write(
            "usd_cli_daemon_ready",
            phase="setup",
            summary=(
                "Started and captured the project-scoped usd-cli daemon "
                "before granting the large-scene child write access."
            ),
            data={"telemetry_route": telemetry_route.metadata()},
        )
        child_returncode = _run_child_agent(
            config=execution_config,
            prompt=execution_prompt,
            run_dir=Path(request.run_dir),
            child_output_path=child_output_path,
            child_final_path=child_final_path,
            prompt_image_inputs=[],
            bridge_artifact_prefix=bridge_artifact_prefix,
        )
        trace_writer.write(
            "child_agent_finished",
            phase="runner",
            summary="Large-scene child agent exited.",
            artifacts=[str(child_output_path), str(child_final_path)],
            data={"returncode": child_returncode},
        )
    except UnsafeRunArtifactError:
        raise
    except (KeyboardInterrupt, ChildProcessInterrupted) as exc:
        interrupted = True
        child_returncode = 130
        _append_child_runner_error(
            child_output_path,
            RuntimeError(f"Scene workflow interrupted: {exc}"),
            run_dir=run_dir,
        )
    except Exception as exc:  # noqa: BLE001 - preserve partial run artifacts
        child_returncode = 2
        _append_child_runner_error(
            child_output_path,
            exc,
            run_dir=Path(request.run_dir),
        )
        if not backend_probe_path.exists() or backend_probe_path.is_symlink():
            try:
                atomic_write_json(
                    backend_probe_path,
                    {
                        "schema_version": "content-agents.scene-backend-probe.v1",
                        "status": "failed",
                        "scene_backend": SCENE_BACKEND_USD_CLI,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                    within=Path(request.run_dir),
                )
            except (OSError, ValueError) as probe_error:
                logger.warning(
                    "Could not preserve failed scene-backend probe at %s: %s",
                    backend_probe_path,
                    probe_error,
                )
        trace_writer.write(
            "child_agent_failed",
            phase="runner",
            summary="Large-scene child-agent runner failed.",
            artifacts=[str(child_output_path), str(child_final_path)],
            data={"error_type": type(exc).__name__, "error": str(exc)},
        )
    finally:
        if parent_usd_cli_capability is not None:
            teardown_errors: list[str] = []
            try:
                daemon_release, retry_error = _retry_scene_terminal_action(
                    lambda: stop_parent_usd_cli_capability_strict(
                        parent_usd_cli_capability
                    ),
                    on_interrupt=mark_terminal_interrupt,
                )
            except Exception as exc:  # noqa: BLE001 - teardown is a terminal gate
                teardown_errors.append(f"usd-cli teardown: {type(exc).__name__}: {exc}")
            else:
                if retry_error is not None:
                    teardown_errors.append(
                        "usd-cli teardown retry after KeyboardInterrupt: "
                        f"{type(retry_error).__name__}: {retry_error}"
                    )
            if daemon_release is not None:
                unreleased = [
                    name
                    for name in (
                        "process_released",
                        "descendants_released",
                        "sessions_released",
                        "listener_released",
                        "daemon_leases_released",
                        "state_directory_released",
                    )
                    if getattr(daemon_release, name) is not True
                ]
                if unreleased:
                    teardown_errors.append(
                        "strict usd-cli teardown returned incomplete release "
                        "evidence: " + ", ".join(unreleased)
                    )

            identity_result, identity_retry_error = _retry_scene_terminal_action(
                lambda: _scene_parent_session_integrity_errors(
                    run_dir,
                    capability=parent_usd_cli_capability,
                ),
                on_interrupt=mark_terminal_interrupt,
            )
            if identity_result is None:
                launch_id = (
                    "invalid-"
                    + hashlib.sha256(
                        str(parent_usd_cli_capability.identity_path).encode()
                    ).hexdigest()[:16]
                )
                assert identity_retry_error is not None
                identity_errors = [
                    "parent session attestation retry after KeyboardInterrupt: "
                    f"{type(identity_retry_error).__name__}: {identity_retry_error}"
                ]
            else:
                launch_id, identity_errors = identity_result
            teardown_errors.extend(identity_errors)
            if daemon_release is None and not teardown_errors:
                teardown_errors.append("strict usd-cli teardown returned no evidence")

            try:
                terminal, terminal_retry_error = _retry_scene_terminal_action(
                    lambda: _validate_terminal_state(run_state_path),
                    on_interrupt=mark_terminal_interrupt,
                )
            except Exception as exc:  # noqa: BLE001 - receipt remains fail-closed
                terminal_retry_error = exc
            if terminal is None:
                assert terminal_retry_error is not None
                terminal_error = (
                    "scene terminal-state attestation: "
                    f"{type(terminal_retry_error).__name__}: {terminal_retry_error}"
                )
                terminal = SceneTerminalValidation(
                    checked_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                    valid=False,
                    current_phase=None,
                    phase_statuses={},
                    errors=[terminal_error],
                )
                teardown_errors.append(terminal_error)
            boundary: Literal["completed", "failed", "cancelled", "incomplete"]
            if interrupted:
                boundary = "cancelled"
            elif child_returncode != 0:
                boundary = "failed"
            elif terminal.valid:
                boundary = "completed"
            else:
                boundary = "incomplete"

            if teardown_errors:
                child_returncode = 2
            try:
                teardown_receipt_path, receipt_retry_error = (
                    _retry_scene_terminal_action(
                        lambda: _write_scene_usd_cli_teardown_receipt(
                            run_dir=run_dir,
                            request=request,
                            capability=parent_usd_cli_capability,
                            release=daemon_release,
                            boundary="cancelled" if interrupted else boundary,
                            child_returncode=child_returncode,
                            interrupted=interrupted,
                            launch_id=launch_id,
                            errors=teardown_errors,
                        ),
                        on_interrupt=mark_terminal_interrupt,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - receipt is a terminal gate
                child_returncode = 2
                teardown_errors.append(f"usd-cli teardown receipt failed: {exc}")
            else:
                if teardown_errors:
                    child_returncode = 2
                if receipt_retry_error is not None:
                    child_returncode = 2
                    teardown_errors.append(
                        "usd-cli teardown receipt retry after KeyboardInterrupt: "
                        f"{type(receipt_retry_error).__name__}: {receipt_retry_error}"
                    )
            if teardown_errors:
                try:
                    _append_child_runner_error(
                        child_output_path,
                        RuntimeError(
                            "Scene usd-cli teardown failed: "
                            + "; ".join(teardown_errors)
                        ),
                        run_dir=run_dir,
                    )
                except BaseException as log_error:  # receipt sealing is authoritative
                    logger.warning(
                        "Could not record scene usd-cli teardown errors at %s: %s",
                        child_output_path,
                        log_error,
                    )

    if terminal is None:
        terminal = _validate_terminal_state(run_state_path)
    terminal_path = _write_terminal_validation(Path(request.run_dir), terminal)
    if teardown_receipt_path is not None:
        trace_writer.write(
            "usd_cli_teardown_recorded",
            phase="runner",
            summary="Recorded strict parent-owned scene usd-cli teardown evidence.",
            artifacts=[str(teardown_receipt_path)],
            data={"interrupted": interrupted},
        )
    trace_writer.write(
        "terminal_validation",
        phase="validation",
        summary=(
            "Large-scene run reached a valid terminal state."
            if terminal.valid
            else "Large-scene run did not reach a valid terminal state."
        ),
        artifacts=[str(terminal_path)],
        data={"valid": terminal.valid, "errors": terminal.errors},
    )

    request_payload = request.model_dump(mode="json")
    _write_run_cost_metrics(
        config=config,
        run_dir=Path(request.run_dir),
        request=request_payload,
        wall_time_seconds=time.monotonic() - run_started,
    )
    trace_paths = build_trace(Path(request.run_dir))
    returncode = child_returncode
    if child_returncode == 0 and not terminal.valid:
        returncode = 1
    manifest = _finalize_scene_run_manifest(
        recorder=run_recorder,
        run_dir=Path(request.run_dir),
        prompt_path=prompt_path,
        terminal_path=terminal_path,
        teardown_receipt_path=teardown_receipt_path,
        status="pass" if returncode == 0 and terminal.valid else "fail",
        failure=(
            None
            if returncode == 0 and terminal.valid
            else {
                "code": "large_scene_run_failed",
                "child_returncode": child_returncode,
                "effective_returncode": returncode,
                "terminal_errors": terminal.errors,
            }
        ),
    )
    if returncode == 0 and manifest.status != "pass":
        returncode = 1
    return SceneRunResult(
        run_dir=Path(request.run_dir),
        request_path=request_path,
        run_state_path=run_state_path,
        prompt_path=prompt_path,
        child_output_path=child_output_path,
        child_final_path=child_final_path,
        terminal_validation_path=terminal_path,
        returncode=returncode,
        completed=terminal.valid and manifest.status == "pass",
        trace_paths=_trace_path_summary(trace_paths),
    )


def _build_scene_request(
    config: SceneRunConfig,
    *,
    run_id: str,
    run_dir: Path,
    run_state_path: Path,
    staged_source: UsdCliStagedInput | None = None,
    staged_material_library: UsdCliStagedInput | None = None,
    staged_materials_yaml: Path | None = None,
) -> SceneRunRequest:
    tasks: list[SceneTaskRequest] = []
    for domain in config.requested_tasks:
        if domain == "material":
            tasks.append(
                SceneTaskRequest(
                    domain=domain,
                    scene_backend=SCENE_BACKEND_USD_CLI,
                    scene_session_scope="per_asset",
                    inputs={
                        "materials_yaml": str(
                            staged_materials_yaml
                            if staged_materials_yaml is not None
                            else config.materials_yaml
                        ),
                        "materials_yaml_source": str(config.materials_yaml),
                        "materials_usd": str(
                            staged_material_library.staged_usd_path
                            if staged_material_library is not None
                            else config.materials_usd
                        ),
                        "materials_usd_source": str(config.materials_usd),
                    },
                    policy={
                        "candidate_space": config.material_candidate_space,
                        "respect_existing_material_bindings": (
                            config.respect_existing_material_bindings
                        ),
                        "appearance_evidence_policy": {
                            "schema_version": (
                                "content-agent-workflows.appearance-evidence-policy.v1"
                            ),
                            "default": "ignore",
                            "global_sources": [],
                            "scopes": [],
                        },
                    },
                )
            )
        else:
            tasks.append(
                SceneTaskRequest(
                    domain=domain,
                    scene_backend=SCENE_BACKEND_USD_CLI,
                    scene_session_scope="per_asset",
                )
            )

    return SceneRunRequest(
        created_at=utc_now(),
        dry_run=config.dry_run,
        run_id=run_id,
        run_dir=str(run_dir),
        run_state=str(run_state_path),
        repository_root=str(config.repo_root),
        agent_workspace=str(_agent_workspace(config)),
        child_workspace=(
            str(config.agent_cwd.resolve()) if config.agent_cwd is not None else None
        ),
        source_scene=str(
            staged_source.staged_usd_path
            if staged_source is not None
            else config.usd_path
        ),
        source_scene_original=(
            str(config.usd_path) if staged_source is not None else None
        ),
        requested_tasks=config.requested_tasks,
        references=SceneReferencesRequest(
            directories=[str(path) for path in config.reference_directories],
            images=[str(path) for path in config.reference_images],
            files=[str(path) for path in config.reference_files],
        ),
        additional_instructions=(
            config.additional_instructions.strip()
            if config.additional_instructions and config.additional_instructions.strip()
            else None
        ),
        additional_instruction_sources=[
            str(path) for path in config.additional_instruction_sources
        ],
        tasks=tasks,
        decomposition={"mode": "agent_planned", "overrides": {}},
        collection={"mode": "domain_aware", "overrides": {}},
        runtime=SceneRuntimeRequest(
            runner=config.runner,
            scene_backend=SCENE_BACKEND_USD_CLI,
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
        ),
    )


def _stage_scene_materials_yaml(
    *,
    source_path: Path,
    staged_material_library: UsdCliStagedInput,
    run_dir: Path,
) -> Path:
    """Stage the manifest with its library path bound to the staged USD."""

    source = source_path.expanduser().resolve(strict=True)
    destination = staged_material_library.staged_usd_path.parent / source.name
    if destination.exists() or destination.is_symlink():
        raise ValueError(f"Refusing pre-existing staged materials YAML: {destination}")
    artifact = read_contained_artifact(
        source.parent,
        source,
        max_bytes=MAX_SCENE_MATERIALS_YAML_BYTES,
        capture_bytes=True,
        allow_hardlinks=True,
    )
    assert artifact.data is not None
    try:
        payload = yaml.safe_load(artifact.data.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"Invalid materials YAML manifest {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Material YAML manifest {source} must be a mapping.")
    payload["library_path"] = os.path.relpath(
        staged_material_library.staged_usd_path,
        start=destination.parent,
    )
    staged_bytes = yaml.safe_dump(payload, sort_keys=False).encode("utf-8")
    atomic_write_bytes(
        destination,
        staged_bytes,
        within=staged_material_library.staged_root,
    )
    os.chmod(destination, 0o444)
    return destination


def _scene_staged_inputs_root(run_dir: Path) -> Path:
    run_root = run_dir.expanduser().resolve(strict=True)
    return run_root.parent / f".{run_root.name}.scene-inputs"


def _prepare_scene_staged_inputs_root(run_dir: Path) -> Path:
    staged_inputs_root = _scene_staged_inputs_root(run_dir)
    if staged_inputs_root.exists() or staged_inputs_root.is_symlink():
        raise FileExistsError(
            f"Large-scene staged inputs already exist: {staged_inputs_root}"
        )
    staged_inputs_root.mkdir(mode=0o700)
    return staged_inputs_root.resolve(strict=True)


def _remove_stale_scene_staged_inputs(run_dir: Path) -> None:
    staged_inputs_root = _scene_staged_inputs_root(run_dir)
    try:
        root_metadata = staged_inputs_root.lstat()
    except FileNotFoundError:
        return
    if not stat.S_ISDIR(root_metadata.st_mode) or _scene_metadata_is_reparse_point(
        root_metadata
    ):
        raise FileExistsError(
            f"Refusing unsafe stale scene staged inputs: {staged_inputs_root}"
        )

    files: list[Path] = []
    directories: list[Path] = []

    def inspect(directory: Path) -> None:
        try:
            with os.scandir(directory) as stream:
                entries = tuple(stream)
        except OSError as exc:
            raise FileExistsError(
                f"Refusing unsafe stale scene staged inputs: {staged_inputs_root}"
            ) from exc
        for entry in entries:
            path = Path(entry.path)
            try:
                # ``DirEntry.stat(follow_symlinks=False)`` returns zeroed
                # identity/link fields for read-only files on Windows. Path's
                # no-follow stat preserves the NTFS identity we need here.
                metadata = path.lstat()
            except OSError as exc:
                raise FileExistsError(
                    f"Refusing unsafe stale scene staged inputs: {staged_inputs_root}"
                ) from exc
            if _scene_metadata_is_reparse_point(metadata):
                raise FileExistsError(
                    f"Refusing unsafe stale scene staged inputs: {staged_inputs_root}"
                )
            if stat.S_ISDIR(metadata.st_mode):
                inspect(path)
                directories.append(path)
                continue
            if stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1:
                files.append(path)
                continue
            raise FileExistsError(
                f"Refusing unsafe stale scene staged inputs: {staged_inputs_root}"
            )

    inspect(staged_inputs_root)
    # Sealed inputs are deliberately read-only. Windows maps that mode to the
    # read-only file attribute, which must be cleared before recursive cleanup.
    # Validate the complete tree first so chmod never follows a rejected link.
    for file_path in files:
        os.chmod(file_path, 0o600)
    for directory in directories:
        os.chmod(directory, 0o700)
    os.chmod(staged_inputs_root, 0o700)
    shutil.rmtree(staged_inputs_root)


def _seal_scene_staged_inputs(staged_inputs_root: Path) -> None:
    for path in staged_inputs_root.rglob("*"):
        os.chmod(path, 0o555 if path.is_dir() else 0o444)
    os.chmod(staged_inputs_root, 0o555)


def _scene_parent_input_roots(
    run_dir: Path,
    *,
    request: SceneRunRequest,
) -> tuple[Path, ...]:
    staged_inputs_root = _scene_staged_inputs_root(run_dir)
    working_inputs = [Path(request.source_scene)]
    for task in request.tasks:
        if task.domain != "material":
            continue
        for name in ("materials_yaml", "materials_usd"):
            value = task.inputs.get(name)
            if value:
                working_inputs.append(Path(str(value)))

    staged = False
    external: list[Path] = []
    for input_path in working_inputs:
        resolved = input_path.expanduser().resolve(strict=True)
        try:
            resolved.relative_to(staged_inputs_root)
        except ValueError:
            external.append(resolved)
        else:
            staged = True

    if not external:
        return (staged_inputs_root,)

    # Current runs use the sealed sibling depot above. Preserve resumability for
    # accepted v2 requests written before that depot existed: the launcher policy
    # and workflow manifest verify these exact input paths and hashes before the
    # parent capability is created. New runs never take this compatibility path.
    roots = ([staged_inputs_root] if staged else []) + external
    return tuple(dict.fromkeys(roots))


def _config_from_request(
    request: SceneRunRequest,
    *,
    dry_run: bool,
) -> SceneRunConfig:
    material_task = next(
        (task for task in request.tasks if task.domain == "material"),
        None,
    )
    materials_yaml = None
    materials_usd = None
    candidate_space = "source"
    respect_existing = False
    if material_task is not None:
        raw_yaml = material_task.inputs.get("materials_yaml")
        raw_usd = material_task.inputs.get("materials_usd")
        materials_yaml = Path(str(raw_yaml)).resolve() if raw_yaml else None
        materials_usd = Path(str(raw_usd)).resolve() if raw_usd else None
        candidate_space = str(material_task.policy.get("candidate_space") or "source")
        respect_existing = bool(
            material_task.policy.get("respect_existing_material_bindings", False)
        )

    repository_root = Path(request.repository_root).resolve()
    agent_workspace = Path(request.agent_workspace).resolve()
    # Requests produced before child_workspace was added wrote the default run
    # directory into agent_workspace. Recover their real skill workspace so
    # existing production runs remain resumable after the confinement change.
    if (
        request.child_workspace is None
        and agent_workspace == Path(request.run_dir).resolve()
    ):
        agent_workspace = (repository_root / "agentic").resolve()
    trusted_agent_workspace = (repository_root / "agentic").resolve()
    if agent_workspace != trusted_agent_workspace:
        raise ValueError(
            "Scene agent_workspace must resolve to the trusted repository "
            f"workspace: {agent_workspace} != {trusted_agent_workspace}"
        )

    return SceneRunConfig(
        repo_root=repository_root,
        agent_workspace=agent_workspace,
        agent_cwd=_confined_child_workspace(
            Path(request.child_workspace)
            if request.child_workspace is not None
            else None,
            run_dir=Path(request.run_dir),
        ),
        usd_path=Path(request.source_scene_original or request.source_scene).resolve(),
        requested_tasks=list(request.requested_tasks),
        reference_images=[Path(path).resolve() for path in request.references.images],
        reference_files=[Path(path).resolve() for path in request.references.files],
        reference_directories=[
            Path(path).resolve() for path in request.references.directories
        ],
        materials_yaml=materials_yaml,
        materials_usd=materials_usd,
        material_candidate_space=candidate_space,
        respect_existing_material_bindings=respect_existing,
        additional_instructions=request.additional_instructions,
        additional_instruction_sources=[
            Path(path).resolve() for path in request.additional_instruction_sources
        ],
        output_dir=Path(request.run_dir).resolve(),
        run_id=request.run_id,
        runner=request.runtime.runner,
        model=request.runtime.model,
        model_reasoning_effort=request.runtime.model_reasoning_effort,
        codex_base_url=request.runtime.codex_base_url,
        codex_sandbox_mode=request.runtime.codex_sandbox_mode,
        codex_config=request.runtime.codex_config or None,
        claude_config=request.runtime.claude_config or None,
        claude_permission_mode=request.runtime.claude_permission_mode,
        claude_max_turns=request.runtime.claude_max_turns,
        claude_execution_mode=request.runtime.claude_execution_mode,
        scene_tool_timeout_seconds=request.runtime.scene_tool_timeout_seconds,
        child_timeout_seconds=request.runtime.child_timeout_seconds,
        dry_run=dry_run,
    )


def _build_scene_agent_prompt(
    *,
    request_path: Path,
    run_state_path: Path,
    run_dir: Path,
    resume: bool,
    scene_backend: str,
) -> str:
    action = "Resume" if resume else "Execute"
    artifact_authoring_guidance = (
        CONTROLLED_JSON_ARTIFACT_WRITE
        if os.name == "nt"
        else (
            "Author declarative JSON, YAML, and Markdown with the child "
            "file-editing tools, never with inline or module Python."
        )
    )
    fail_fast_guidance = (
        "On Windows, every `scene material-task run-batch` command MUST include "
        "`--fail-fast`. Treat its nonzero result as the originating system "
        "blocker; record the phase failure without attempting later work items."
        if os.name == "nt"
        else ""
    )
    return f"""Use the `content-workflow-large-scene` skill.

{action} every required phase for this batch request:

- Resolved request: `{request_path}`
- Durable run state: `{run_state_path}`
- Run directory: `{run_dir}`
- Frozen scene backend: `{scene_backend}`

Read the resolved request and run state before acting. Use the task, domain, and
scene-backend skills selected by the umbrella skill. The request is frozen
launcher input; do not rewrite it. Never edit `large_scene_run.json` directly.
You are already inside the launcher-owned child turn. Never invoke
`content-workflow-cli scene run` or `content-workflow-cli scene resume` here;
those parent-only commands would recursively launch another agent and replace
active phase state. Continue this turn with `scene phase`, `scene decompose`,
`scene process`, `scene material-task`, and `scene collect` commands only.
Use the internal transition helper required by the umbrella skill, validate
each handoff, and do not advance past a failed gate.

Keep semantic decomposition, task planning, per-asset decisions, evidence reuse,
and collection judgment agent-owned. Read scene-level `additional_instructions`
from run state and preserve them exactly in applicable task requests. For
material task requests, keep the clean-slate `appearance_evidence_policy`
unless explicit user guidance asks for scoped display-color or existing-material
evidence on named roots. Copy each task's `scene_backend` and
`scene_session_scope` into the frozen domain task request's `processing_policy`.
Copy task inputs exactly as frozen: a material request's `material_library_path`
must use `tasks[].inputs.materials_usd` and `material_library_yaml` must use
`tasks[].inputs.materials_yaml`. Never substitute a `*_source` provenance path
for its run-confined working input.
Give every per-asset scene operation an independent session identity and
checkpoint lineage; never reuse a usd-cli project/session
across asset work items. A retry must preserve completed item artifacts and
reopen only the failed or deferred item. Write all generated artifacts under
the run directory.

Continue until collection completes and `current_phase` becomes null. If a
concrete blocker prevents completion, record the phase failure and clearly
identify the resumable state in the final response.

Use `content-workflow-cli scene ...` for every workflow operation.
{artifact_authoring_guidance}
{fail_fast_guidance}
Do not infer that the execution policy blocks an operation from missing
aliases, documentation, or a different failed command.
Report a policy blocker only after the exact required command was attempted and
the retained command result contains the denial reason.
"""


def _scene_run_manifest_backend(request: SceneRunRequest) -> dict[str, Any]:
    return {
        "scene_backend": request.runtime.scene_backend,
        "scene_tool": "usd-cli",
        "renderer": "ovrtx",
        "renderer_probe_required": True,
        "backend_readiness_evidence": "ovrtx_render_probe",
        "runner": request.runtime.runner,
        "scene_session_scope": "per_asset",
    }


def _scene_run_manifest_policy(request: SceneRunRequest) -> dict[str, Any]:
    input_paths = [
        request.source_scene_original or request.source_scene,
        *request.references.images,
        *request.references.files,
        *request.additional_instruction_sources,
    ]
    for task in request.tasks:
        if task.domain != "material":
            continue
        material_yaml_name = (
            "materials_yaml_source"
            if task.inputs.get("materials_yaml_source")
            else "materials_yaml"
        )
        material_usd_name = (
            "materials_usd_source"
            if task.inputs.get("materials_usd_source")
            else "materials_usd"
        )
        for name in (material_yaml_name, material_usd_name):
            value = task.inputs.get(name)
            if value:
                input_paths.append(str(value))
    input_hashes = {
        str(path): file_sha256(Path(path)) for path in sorted(set(input_paths))
    }
    return {
        "requested_tasks": list(request.requested_tasks),
        "tasks": [task.model_dump(mode="json") for task in request.tasks],
        "decomposition": request.decomposition,
        "collection": request.collection,
        "additional_instructions": request.additional_instructions,
        "input_sha256": input_hashes,
    }


def _scene_run_manifest_required_artifacts(
    request: SceneRunRequest,
    *,
    legacy_adopted: bool,
) -> list[str]:
    required = [
        "request",
        "run_state",
        "terminal_validation",
        "collection_output",
        "operation_trace",
    ]
    # Adopted pre-manifest runs predate the backend-readiness artifact. Their
    # v1 bytes and migration receipt are the durable evidence for the explicit
    # trust transition; a completed historical run must not replay a backend
    # operation merely to manufacture new readiness evidence.
    if legacy_adopted:
        required[2:2] = [
            "legacy_adoption_receipt",
            "legacy_request_evidence",
            "legacy_run_state_evidence",
        ]
    else:
        required.insert(2, "backend_probe")
    return required


def _scene_run_recorder_contract(
    *,
    run_dir: Path,
    request: SceneRunRequest,
    source_path: Path,
) -> dict[str, Any]:
    request_document = load_json(run_dir / "request.json")
    intent_path = _legacy_scene_migration_intent_path(run_dir)
    receipt_path = _legacy_scene_adoption_receipt_path(run_dir)
    intent_exists = intent_path.exists() or intent_path.is_symlink()
    receipt_exists = receipt_path.exists() or receipt_path.is_symlink()
    if intent_exists != receipt_exists:
        raise SceneLauncherPolicyError(
            "Legacy scene migration intent and adoption receipt must either both "
            "exist or both be absent."
        )
    return {
        "workflow": "scene.run",
        "request": request_document,
        "source_path": source_path,
        "backend": _scene_run_manifest_backend(request),
        "policy": _scene_run_manifest_policy(request),
        "required_artifacts": _scene_run_manifest_required_artifacts(
            request,
            legacy_adopted=intent_exists,
        ),
    }


def _start_scene_run_recorder(
    *,
    run_dir: Path,
    request: SceneRunRequest,
    source_path: Path,
    resume: bool,
) -> WorkflowRunRecorder:
    contract = _scene_run_recorder_contract(
        run_dir=run_dir,
        request=request,
        source_path=source_path,
    )
    if not resume:
        return WorkflowRunRecorder.start(run_dir, **contract)
    manifest_path = run_dir / "workflow_run_manifest.json"
    if manifest_path.exists() or manifest_path.is_symlink():
        return WorkflowRunRecorder.start(run_dir, **contract, resume=True)
    raise SceneLauncherPolicyError(
        "Scene workflow manifest is missing; modern runs cannot be adopted, and "
        "a genuine pre-policy v1 run requires explicit operator adoption with "
        "--adopt-legacy-request-sha256."
    )


def _finalize_scene_run_manifest(
    *,
    recorder: WorkflowRunRecorder,
    run_dir: Path,
    prompt_path: Path,
    terminal_path: Path | None,
    status: Literal["pass", "fail", "blocked"],
    failure: dict[str, Any] | None,
    teardown_receipt_path: Path | None = None,
) -> WorkflowRunManifest:
    """Record one durable scene checkpoint without replacing TraceWriter."""

    artifact_errors: list[str] = []
    collection_record = None
    try:
        current_request = SceneRunRequest.model_validate(
            load_json(run_dir / "request.json")
        )
        current_policy = _scene_run_manifest_policy(current_request)
        current_backend = _scene_run_manifest_backend(current_request)
    except (OSError, ValueError) as exc:
        artifact_errors.append(f"workflow_contract: {exc}")
    else:
        if current_policy != recorder.manifest.policy:
            artifact_errors.append(
                "workflow_contract: workflow policy or referenced inputs changed"
            )
        if current_backend != recorder.manifest.backend:
            artifact_errors.append("workflow_contract: backend route changed")

    candidates: list[tuple[str, Path, str, bool]] = [
        ("request", run_dir / "request.json", "request", True),
        ("run_state", run_dir / "large_scene_run.json", "checkpoint", True),
        (
            "legacy_adoption_receipt",
            _legacy_scene_adoption_receipt_path(run_dir),
            "adoption_receipt",
            "legacy_adoption_receipt" in recorder.manifest.required_artifacts,
        ),
        (
            "legacy_request_evidence",
            _legacy_scene_evidence_paths(run_dir)[0],
            "migration_evidence",
            "legacy_request_evidence" in recorder.manifest.required_artifacts,
        ),
        (
            "legacy_run_state_evidence",
            _legacy_scene_evidence_paths(run_dir)[1],
            "migration_evidence",
            "legacy_run_state_evidence" in recorder.manifest.required_artifacts,
        ),
        ("prompt", prompt_path, "prompt", False),
        (
            "backend_probe",
            run_dir / "raw" / "scene_backend_probe.json",
            "backend_probe",
            "backend_probe" in recorder.manifest.required_artifacts,
        ),
        (
            "usd_cli_teardown",
            teardown_receipt_path
            or run_dir / "raw" / "scene_usd_cli_teardown_missing.json",
            "resource_release_receipt",
            False,
        ),
        (
            "terminal_validation",
            terminal_path or run_dir / "terminal_validation.json",
            "validation",
            True,
        ),
        ("child_output", run_dir / "child-output.log", "runner_log", False),
        ("child_final", run_dir / "child-final.md", "agent_response", False),
        (
            "child_resume_output",
            run_dir / "child-resume-output.log",
            "runner_log",
            False,
        ),
        (
            "child_resume_final",
            run_dir / "child-resume-final.md",
            "agent_response",
            False,
        ),
        (
            "run_cost_metrics",
            run_dir / "run_cost_metrics.json",
            "metrics",
            False,
        ),
        (
            "operation_trace",
            run_dir / "trace" / "operation_trace.json",
            "trace",
            True,
        ),
        (
            "replay_manifest",
            run_dir / "trace" / "replay_manifest.json",
            "trace",
            False,
        ),
    ]
    collection_output: Path | None = None
    try:
        run_state = load_run_state(run_dir / "large_scene_run.json")
        result_path = run_state.phases["collection"].result_path
        if result_path:
            collection_output = Path(result_path).expanduser()
            if not collection_output.is_absolute():
                collection_output = run_dir / collection_output
            candidates.append(("collection_output", collection_output, "output", True))
    except Exception:
        collection_output = None

    for logical_name, path, kind, required in candidates:
        if not path.exists() and not path.is_symlink():
            continue
        try:
            record = recorder.record_artifact(
                logical_name,
                path,
                kind=kind,
                required=required,
            )
        except (OSError, ValueError) as exc:
            artifact_errors.append(f"{logical_name}: {exc}")
        else:
            if logical_name == "collection_output":
                collection_record = record

    if collection_record is not None:
        try:
            collection_read = read_contained_artifact(
                run_dir,
                run_dir / collection_record.path,
                max_bytes=MAX_COLLECTION_RESULT_BYTES,
                parse_json=True,
            )
        except ValueError as exc:
            artifact_errors.append(f"collection_output: {exc}")
            collection_document: dict[str, Any] = {}
        else:
            if (
                collection_read.sha256 != collection_record.sha256
                or collection_read.size_bytes != collection_record.size_bytes
            ):
                artifact_errors.append(
                    "collection_output: artifact changed after it was recorded"
                )
                collection_document = {}
            else:
                collection_document = collection_read.json_object or {}
        artifact_paths = collection_document.get("artifact_paths")
        if isinstance(artifact_paths, list):
            for index, value in enumerate(artifact_paths, start=1):
                if not isinstance(value, str) or not value:
                    artifact_errors.append(
                        f"collection_artifact_{index}: invalid artifact path"
                    )
                    continue
                path = Path(value).expanduser()
                if not path.is_absolute():
                    path = collection_output.parent / path
                if _is_scene_staged_input(run_dir, path):
                    # Collection receipts include their frozen inputs for
                    # provenance. They are already digest-bound by the staged
                    # input manifests and are not workflow output artifacts.
                    continue
                try:
                    recorder.record_artifact(
                        f"collection_artifact_{index}",
                        path,
                        kind="output_artifact",
                        required=False,
                    )
                except (OSError, ValueError) as exc:
                    artifact_errors.append(f"collection_artifact_{index}: {exc}")

    # large_scene_run.json is an independently validated atomic state machine
    # and is expected to change between launcher turns.  Record its current
    # digest in the manifest, but do not seal it into the cross-turn checkpoint.
    checkpoint_artifacts = [
        record.logical_name
        for record in recorder.manifest.artifacts
        if record.logical_name != "run_state"
    ]
    if checkpoint_artifacts:
        try:
            recorder.checkpoint(
                "validated" if status == "pass" else status,
                checkpoint_artifacts,
            )
        except (OSError, ValueError) as exc:
            artifact_errors.append(f"checkpoint: {exc}")
    effective_failure = dict(failure or {})
    if artifact_errors:
        effective_failure.setdefault("artifact_errors", artifact_errors)
        if status == "pass":
            status = "fail"
            effective_failure.setdefault("code", "artifact_recording_failed")
    return recorder.finalize(status, failure=effective_failure or None)


def _is_scene_staged_input(run_dir: Path, path: Path) -> bool:
    try:
        resolved = path.resolve(strict=True)
        staged_inputs_root = _scene_staged_inputs_root(run_dir)
        resolved.relative_to(staged_inputs_root)
    except (OSError, ValueError):
        return False
    return resolved != staged_inputs_root


def _validate_scene_config(config: SceneRunConfig) -> None:
    tasks = config.requested_tasks
    if not tasks:
        raise ValueError("At least one --task is required.")
    if len(tasks) != len(set(tasks)):
        raise ValueError("--task values must be unique.")
    for task in tasks:
        if not re.fullmatch(r"[a-z][a-z0-9_-]*", task):
            raise ValueError(f"Invalid --task value: {task!r}")

    paths: list[tuple[str, Path]] = [("USD", config.usd_path)]
    paths.extend(
        (f"reference image {index + 1}", path)
        for index, path in enumerate(config.reference_images)
    )
    paths.extend(
        (f"reference file {index + 1}", path)
        for index, path in enumerate(config.reference_files)
    )
    paths.extend(
        (f"additional instruction source {index + 1}", path)
        for index, path in enumerate(config.additional_instruction_sources)
    )
    if "material" in tasks:
        if config.materials_yaml is None:
            raise ValueError("--materials-yaml is required for --task material.")
        if config.materials_usd is None:
            raise ValueError("A materials USD is required for --task material.")
        paths.extend(
            [
                ("materials YAML", config.materials_yaml),
                ("materials USD", config.materials_usd),
            ]
        )
    elif config.materials_yaml is not None or config.materials_usd is not None:
        raise ValueError("Material library inputs require --task material.")

    for label, path in paths:
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")
        if not path.is_file():
            raise ValueError(f"{label} is not a file: {path}")
        if not os.access(path, os.R_OK):
            raise PermissionError(f"{label} is not readable: {path}")
    for directory in config.reference_directories:
        if not directory.is_dir():
            raise ValueError(f"Reference directory does not exist: {directory}")

    if config.material_candidate_space != "source":
        raise ValueError(
            "scene workflows currently support only --material-candidate-space=source."
        )
    if SCENE_BACKEND_USD_CLI not in SUPPORTED_SCENE_BACKENDS:
        raise ValueError(
            f"Unsupported --scene-backend: {SCENE_BACKEND_USD_CLI!r}; expected "
            f"one of {sorted(SUPPORTED_SCENE_BACKENDS)}"
        )
    if config.runner not in SUPPORTED_RUNNERS:
        raise ValueError(f"Unsupported --runner: {config.runner}")
    if config.codex_sandbox_mode not in SUPPORTED_CODEX_SANDBOX_MODES:
        raise ValueError(
            f"Unsupported --codex-sandbox-mode: {config.codex_sandbox_mode}"
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
    if config.scene_tool_timeout_seconds <= 0:
        raise ValueError("--scene-tool-timeout must be greater than 0.")
    if config.codex_base_url:
        parsed_codex = urlparse(config.codex_base_url)
        if parsed_codex.scheme not in {"http", "https"} or not parsed_codex.netloc:
            raise ValueError(f"Invalid --codex-base-url: {config.codex_base_url}")

    workspace = _agent_workspace(config)
    trusted_workspace = (config.repo_root / "agentic").resolve()
    if workspace != trusted_workspace:
        raise ValueError(
            "Scene agent_workspace must resolve to the trusted repository "
            f"workspace: {workspace} != {trusted_workspace}"
        )
    if not workspace.is_dir():
        raise FileNotFoundError(f"Agent workspace does not exist: {workspace}")
    if not (workspace / ".agents" / "skills").is_dir():
        raise FileNotFoundError(
            f"Agent workspace does not expose .agents/skills: {workspace}"
        )


@contextmanager
def _prepare_scene_run_dir(config: SceneRunConfig) -> Iterator[tuple[str, Path]]:
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
    default_run_id = f"{_slug(config.usd_path.stem)}-{stamp}"
    run_id = (config.run_id or default_run_id).strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id):
        raise ValueError(
            "--run-id must start with an alphanumeric character and contain only "
            "letters, numbers, dot, underscore, or hyphen."
        )
    run_dir_candidate = (
        config.output_dir
        if config.output_dir is not None
        else config.repo_root / "runs" / run_id
    )
    # Check the uncanonicalized path first so a precreated run-directory
    # symlink cannot be hidden by resolve() and granted as the child workspace.
    _reject_unsafe_run_links(run_dir_candidate, allow_missing=True)
    run_dir = _lexical_absolute_path(run_dir_candidate)
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    _reject_unsafe_run_links(run_dir, allow_missing=True)
    with _scene_run_initialization_lock(run_dir):
        stale_policy_identity = _scene_launcher_policy_identity(run_dir)
        created_run_dir = False
        try:
            run_dir.mkdir()
            created_run_dir = True
        except FileExistsError:
            pass
        _reject_unsafe_run_links(run_dir)
        if created_run_dir and stale_policy_identity is not None:
            _remove_stale_scene_launcher_policy(
                run_dir,
                expected_identity=stale_policy_identity,
            )
        if created_run_dir:
            _remove_stale_scene_staged_inputs(run_dir)
        for directory_name in ("raw", "trace"):
            _prepare_contained_run_subdirectory(
                run_dir,
                run_dir / directory_name,
                create=True,
            )
        _chmod_private(run_dir / "raw")
        yield run_id, run_dir


def _write_request(run_dir: Path, path: Path, request: SceneRunRequest) -> bytes:
    request_text = request.model_dump_json(indent=2) + "\n"
    request_bytes = request_text.encode("utf-8")
    if len(request_bytes) > MAX_SCENE_REQUEST_BYTES:
        raise SceneLauncherPolicyError(
            f"Scene request exceeds the {MAX_SCENE_REQUEST_BYTES}-byte limit: {path}"
        )
    atomic_write_text(path, request_text, within=run_dir)
    _chmod_private(path, within=run_dir)
    return request_bytes


def _scene_launcher_policy_path(run_dir: Path) -> Path:
    resolved_run_dir = run_dir.expanduser().resolve()
    return (
        resolved_run_dir.parent / f".{resolved_run_dir.name}.scene-launcher-policy.json"
    )


def _legacy_scene_adoption_receipt_path(run_dir: Path) -> Path:
    return run_dir / "legacy_scene_adoption_receipt.json"


def _legacy_scene_migration_intent_path(run_dir: Path) -> Path:
    resolved_run_dir = run_dir.expanduser().resolve()
    return (
        resolved_run_dir.parent
        / f".{resolved_run_dir.name}.legacy-scene-migration-intent.json"
    )


def _legacy_scene_evidence_paths(run_dir: Path) -> tuple[Path, Path]:
    evidence_dir = run_dir / "raw" / "legacy_scene_adoption"
    return evidence_dir / "request.v1.json", evidence_dir / "large_scene_run.v1.json"


def _scene_run_initialization_lock_path(run_dir: Path) -> Path:
    resolved_run_dir = run_dir.expanduser().resolve()
    return resolved_run_dir.parent / f".{resolved_run_dir.name}.scene-launcher.lock"


def _scene_metadata_is_reparse_point(metadata: os.stat_result) -> bool:
    reparse_attribute = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return stat.S_ISLNK(metadata.st_mode) or bool(
        reparse_attribute
        and getattr(metadata, "st_file_attributes", 0) & reparse_attribute
    )


@contextmanager
def _scene_run_initialization_lock(run_dir: Path) -> Iterator[None]:
    """Serialize parent-owned publication of a scene request and its policy."""

    lock_path = _scene_run_initialization_lock_path(run_dir)
    initial_metadata: os.stat_result | None = None
    if os.name == "nt":
        try:
            initial_metadata = lock_path.lstat()
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise SceneLauncherPolicyError(
                f"Unable to open scene initialization lock safely at {lock_path}: {exc}"
            ) from exc
        if initial_metadata is not None and _scene_metadata_is_reparse_point(
            initial_metadata
        ):
            raise SceneLauncherPolicyError(
                "Unable to open scene initialization lock safely at "
                f"{lock_path}: reparse points are not allowed"
            )
    flags = os.O_RDWR | os.O_CREAT
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise SceneLauncherPolicyError(
            f"Unable to open scene initialization lock safely at {lock_path}: {exc}"
        ) from exc

    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise SceneLauncherPolicyError(
                "Scene initialization lock must be a single-link regular file: "
                f"{lock_path}"
            )
        if os.name == "nt":
            try:
                current_metadata = lock_path.lstat()
            except OSError as exc:
                raise SceneLauncherPolicyError(
                    "Unable to open scene initialization lock safely at "
                    f"{lock_path}: {exc}"
                ) from exc
            if (
                _scene_metadata_is_reparse_point(current_metadata)
                or (current_metadata.st_dev, current_metadata.st_ino)
                != (metadata.st_dev, metadata.st_ino)
                or (
                    initial_metadata is not None
                    and (initial_metadata.st_dev, initial_metadata.st_ino)
                    != (metadata.st_dev, metadata.st_ino)
                )
            ):
                raise SceneLauncherPolicyError(
                    "Unable to open scene initialization lock safely at "
                    f"{lock_path}: the directory entry changed or is a reparse point"
                )
        if hasattr(os, "fchmod"):
            os.fchmod(descriptor, 0o600)
        if os.name == "nt" and metadata.st_size == 0:
            os.write(descriptor, b"\0")
            os.fsync(descriptor)
            os.lseek(descriptor, 0, os.SEEK_SET)

        deadline = time.monotonic() + SCENE_RUN_INITIALIZATION_LOCK_TIMEOUT_SECONDS
        lock = None
        while True:
            candidate = exclusive_descriptor_lock(descriptor)
            try:
                candidate.__enter__()
                lock = candidate
                break
            except BlockingIOError as exc:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise SceneLauncherPolicyError(
                        "Timed out waiting for another scene launcher to finish "
                        f"initializing {run_dir}."
                    ) from exc
                time.sleep(min(0.01, remaining))
            except OSError as exc:
                raise SceneLauncherPolicyError(
                    f"Unable to lock scene run initialization at {lock_path}: {exc}"
                ) from exc
        try:
            yield
        finally:
            assert lock is not None
            lock.__exit__(None, None, None)
    finally:
        os.close(descriptor)


def _scene_launcher_policy_identity(run_dir: Path) -> tuple[int, int] | None:
    """Return the current policy inode without following its directory entry."""

    policy_path = _scene_launcher_policy_path(run_dir)
    try:
        metadata = policy_path.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise SceneLauncherPolicyError(
            f"Unable to inspect scene launcher policy at {policy_path}: {exc}"
        ) from exc
    return metadata.st_dev, metadata.st_ino


def _remove_stale_scene_launcher_policy(
    run_dir: Path,
    *,
    expected_identity: tuple[int, int],
) -> None:
    """Remove a valid orphan policy after recreating its missing run directory."""

    resolved_run_dir = run_dir.expanduser().resolve()
    policy_path = _scene_launcher_policy_path(resolved_run_dir)
    try:
        initial_metadata = policy_path.lstat()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise SceneLauncherPolicyError(
            f"Unable to inspect stale scene launcher policy at {policy_path}: {exc}"
        ) from exc
    if (
        (initial_metadata.st_dev, initial_metadata.st_ino) != expected_identity
        or not stat.S_ISREG(initial_metadata.st_mode)
        or initial_metadata.st_nlink != 1
    ):
        return

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(policy_path, flags)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise SceneLauncherPolicyError(
            f"Unable to open stale scene launcher policy at {policy_path}: {exc}"
        ) from exc

    with os.fdopen(descriptor, "rb") as stream:
        opened_metadata = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(opened_metadata.st_mode)
            or opened_metadata.st_nlink != 1
            or (opened_metadata.st_dev, opened_metadata.st_ino)
            != (initial_metadata.st_dev, initial_metadata.st_ino)
            or opened_metadata.st_size > MAX_SCENE_LAUNCHER_POLICY_BYTES
        ):
            return
        policy_bytes = stream.read(MAX_SCENE_LAUNCHER_POLICY_BYTES + 1)
    if len(policy_bytes) > MAX_SCENE_LAUNCHER_POLICY_BYTES:
        return

    try:
        policy = SceneLauncherPolicy.model_validate_json(policy_bytes)
    except ValueError:
        return
    if Path(policy.run_dir).expanduser().resolve() != resolved_run_dir:
        return

    try:
        current_metadata = policy_path.lstat()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise SceneLauncherPolicyError(
            f"Unable to recheck stale scene launcher policy at {policy_path}: {exc}"
        ) from exc
    if (current_metadata.st_dev, current_metadata.st_ino) != (
        opened_metadata.st_dev,
        opened_metadata.st_ino,
    ):
        return
    try:
        policy_path.unlink()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise SceneLauncherPolicyError(
            f"Unable to remove stale scene launcher policy at {policy_path}: {exc}"
        ) from exc


def _write_scene_launcher_policy(
    run_dir: Path,
    *,
    request_bytes: bytes,
    legacy_adoption_receipt_sha256: str | None = None,
) -> Path:
    """Create the parent-owned policy without replacing or following a symlink."""

    resolved_run_dir = run_dir.expanduser().resolve()
    policy_path = _scene_launcher_policy_path(resolved_run_dir)
    policy = SceneLauncherPolicy(
        run_dir=str(resolved_run_dir),
        request_sha256=hashlib.sha256(request_bytes).hexdigest(),
        legacy_adoption_receipt_sha256=legacy_adoption_receipt_sha256,
    )
    policy_bytes = (policy.model_dump_json(indent=2, exclude_none=True) + "\n").encode(
        "utf-8"
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(policy_path, flags, 0o600)
    except FileExistsError as exc:
        raise FileExistsError(
            f"Refusing to replace existing scene launcher policy: {policy_path}"
        ) from exc
    except OSError as exc:
        raise SceneLauncherPolicyError(
            f"Unable to create scene launcher policy safely at {policy_path}: {exc}"
        ) from exc

    with os.fdopen(descriptor, "wb") as stream:
        if hasattr(os, "fchmod"):
            os.fchmod(stream.fileno(), 0o600)
        stream.write(policy_bytes)
        stream.flush()
        os.fsync(stream.fileno())
    return policy_path


def _verify_scene_launcher_policy(run_dir: Path, *, request_path: Path) -> bytes:
    """Verify the immutable parent policy before parsing a persisted request."""

    resolved_run_dir = run_dir.expanduser().resolve()
    policy_path = _scene_launcher_policy_path(resolved_run_dir)
    if not policy_path.exists():
        raise SceneLauncherPolicyError(
            "Scene launcher policy is missing; refusing to resume a legacy or "
            f"unverified run: {policy_path}"
        )
    policy_bytes = _read_scene_protected_file(
        policy_path,
        label="scene launcher policy",
        max_bytes=MAX_SCENE_LAUNCHER_POLICY_BYTES,
    )

    try:
        policy = SceneLauncherPolicy.model_validate_json(policy_bytes)
    except ValueError as exc:
        raise SceneLauncherPolicyError(
            f"Invalid scene launcher policy at {policy_path}: {exc}"
        ) from exc

    policy_run_dir = Path(policy.run_dir).expanduser().resolve()
    if policy_run_dir != resolved_run_dir:
        raise SceneLauncherPolicyError(
            "Scene launcher policy run_dir mismatch: "
            f"{policy_run_dir} != {resolved_run_dir}"
        )

    request_bytes = _read_scene_protected_file(
        request_path,
        label="protected scene request",
        max_bytes=MAX_SCENE_REQUEST_BYTES,
    )
    request_sha256 = hashlib.sha256(request_bytes).hexdigest()
    if request_sha256 != policy.request_sha256:
        raise SceneLauncherPolicyError(
            "Scene request digest does not match the parent-owned launcher policy; "
            f"refusing resume: {request_path}"
        )
    _verify_policy_legacy_scene_evidence(
        resolved_run_dir,
        policy=policy,
    )
    return request_bytes


def _verify_policy_legacy_scene_evidence(
    run_dir: Path,
    *,
    policy: SceneLauncherPolicy,
) -> None:
    """Verify migration evidence bound by the parent-owned launcher policy."""

    receipt_path = _legacy_scene_adoption_receipt_path(run_dir)
    intent_path = _legacy_scene_migration_intent_path(run_dir)
    receipt_exists = receipt_path.exists() or receipt_path.is_symlink()
    intent_exists = intent_path.exists() or intent_path.is_symlink()
    expected_receipt_sha256 = policy.legacy_adoption_receipt_sha256
    if expected_receipt_sha256 is None:
        if receipt_exists or intent_exists:
            raise SceneLauncherPolicyError(
                "Scene launcher policy does not authorize the present legacy "
                "migration metadata."
            )
        return
    if not receipt_exists or not intent_exists:
        raise SceneLauncherPolicyError(
            "Scene launcher policy requires complete legacy migration metadata."
        )

    receipt_bytes = _read_scene_protected_file(
        receipt_path,
        label="legacy scene adoption receipt",
        max_bytes=MAX_LEGACY_SCENE_ADOPTION_BYTES,
    )
    if hashlib.sha256(receipt_bytes).hexdigest() != expected_receipt_sha256:
        raise SceneLauncherPolicyError(
            "Legacy scene adoption receipt differs from the parent-owned policy."
        )
    try:
        receipt = LegacySceneAdoptionReceipt.model_validate_json(receipt_bytes)
    except ValueError as exc:
        raise SceneLauncherPolicyError(
            f"Invalid legacy scene adoption receipt at {receipt_path}: {exc}"
        ) from exc

    intent_bytes = _read_scene_protected_file(
        intent_path,
        label="legacy scene migration intent",
        max_bytes=MAX_LEGACY_SCENE_ADOPTION_BYTES,
    )
    if hashlib.sha256(intent_bytes).hexdigest() != receipt.migration_intent_sha256:
        raise SceneLauncherPolicyError(
            "Legacy scene migration intent differs from the adoption receipt."
        )
    try:
        intent = LegacySceneMigrationIntent.model_validate_json(intent_bytes)
    except ValueError as exc:
        raise SceneLauncherPolicyError(
            f"Invalid legacy scene migration intent at {intent_path}: {exc}"
        ) from exc

    legacy_request_path, legacy_state_path = _legacy_scene_evidence_paths(run_dir)
    canonical_paths = (
        Path(receipt.run_dir).expanduser().resolve() == run_dir
        and Path(intent.run_dir).expanduser().resolve() == run_dir
        and Path(receipt.legacy_request_artifact).expanduser().resolve()
        == legacy_request_path
        and Path(intent.legacy_request_artifact).expanduser().resolve()
        == legacy_request_path
        and Path(receipt.legacy_run_state_artifact).expanduser().resolve()
        == legacy_state_path
        and Path(intent.legacy_run_state_artifact).expanduser().resolve()
        == legacy_state_path
    )
    matching_claims = (
        receipt.request_sha256_after == policy.request_sha256
        and receipt.request_sha256_before == intent.request_sha256_before
        and receipt.request_sha256_after == intent.request_sha256_after
        and receipt.run_state_sha256_before == intent.run_state_sha256_before
        and receipt.source_scene == intent.source_scene
        and receipt.source_sha256 == intent.source_sha256
        and receipt.source_input_digest_before == intent.source_input_digest_before
    )
    if not canonical_paths or not matching_claims:
        raise SceneLauncherPolicyError(
            "Legacy migration receipt and intent contain contradictory claims."
        )

    legacy_request_bytes = _read_scene_protected_file(
        legacy_request_path,
        label="legacy request evidence",
        max_bytes=MAX_SCENE_REQUEST_BYTES,
    )
    legacy_state_bytes = _read_scene_protected_file(
        legacy_state_path,
        label="legacy run-state evidence",
        max_bytes=MAX_SCENE_RUN_STATE_BYTES,
    )
    if (
        hashlib.sha256(legacy_request_bytes).hexdigest()
        != receipt.request_sha256_before
        or hashlib.sha256(legacy_state_bytes).hexdigest()
        != receipt.run_state_sha256_before
    ):
        raise SceneLauncherPolicyError(
            "Preserved legacy scene evidence differs from the adoption receipt."
        )


def _decode_scene_json_object(contents: bytes, *, label: str) -> dict[str, Any]:
    try:
        document = json.loads(contents)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SceneLauncherPolicyError(f"Invalid {label}: {exc}") from exc
    if not isinstance(document, dict):
        raise SceneLauncherPolicyError(f"Invalid {label}: expected a JSON object")
    return document


def _looks_like_recognized_legacy_scene_run(
    *,
    request_path: Path,
    run_state_path: Path,
) -> bool:
    """Identify known v1 metadata without treating it as an integrity anchor."""

    try:
        request_document = _decode_scene_json_object(
            _read_scene_protected_file(
                request_path,
                label="legacy scene request candidate",
                max_bytes=MAX_SCENE_REQUEST_BYTES,
            ),
            label=f"legacy scene request candidate at {request_path}",
        )
        state_document = _decode_scene_json_object(
            _read_scene_protected_file(
                run_state_path,
                label="legacy scene state candidate",
                max_bytes=MAX_SCENE_RUN_STATE_BYTES,
            ),
            label=f"legacy scene state candidate at {run_state_path}",
        )
    except SceneLauncherPolicyError:
        return False
    runtime = request_document.get("runtime")
    tasks = request_document.get("tasks")
    return bool(
        request_document.get("schema_version") == LEGACY_SCENE_REQUEST_SCHEMA_VERSION
        and isinstance(runtime, dict)
        and "scene_backend" not in runtime
        and isinstance(tasks, list)
        and all(
            isinstance(task, dict)
            and "scene_backend" not in task
            and "scene_session_scope" not in task
            for task in tasks
        )
        and state_document.get("schema_version")
        == LEGACY_LARGE_SCENE_RUN_SCHEMA_VERSION
        and "scene_backend" not in state_document
    )


def _legacy_request_input_paths(
    request: SceneRunRequest,
    *,
    request_path: Path,
) -> list[Path]:
    paths = [
        request_path,
        *(Path(value).expanduser() for value in request.additional_instruction_sources),
        *(Path(value).expanduser() for value in request.references.images),
        *(Path(value).expanduser() for value in request.references.files),
    ]
    for task in request.tasks:
        if task.domain != "material":
            continue
        for name in ("materials_yaml", "materials_usd"):
            value = task.inputs.get(name)
            if value:
                paths.append(Path(str(value)).expanduser())
    return paths


def _require_legacy_input_without_links(path: Path, *, label: str) -> Path:
    """Validate one adopted input lexically without following a link component."""

    absolute = _lexical_absolute_path(path)
    current = Path(absolute.anchor)
    for index, component in enumerate(absolute.parts[1:], start=1):
        current /= component
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise SceneLauncherPolicyError(
                f"Legacy {label} is missing or cannot be inspected safely: {current}"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise SceneLauncherPolicyError(
                f"Legacy {label} must not traverse a symlink: {current}"
            )
        if index < len(absolute.parts) - 1 and not stat.S_ISDIR(metadata.st_mode):
            raise SceneLauncherPolicyError(
                f"Legacy {label} parent is not a directory: {current}"
            )
    metadata = absolute.stat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise SceneLauncherPolicyError(
            f"Legacy {label} must be a single-link regular file: {absolute}"
        )
    return absolute


def _require_legacy_directory_without_links(path: Path, *, label: str) -> Path:
    absolute = _lexical_absolute_path(path)
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise SceneLauncherPolicyError(
                f"Legacy {label} is missing or cannot be inspected safely: {current}"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise SceneLauncherPolicyError(
                f"Legacy {label} must not traverse a symlink: {current}"
            )
    if not absolute.is_dir():
        raise SceneLauncherPolicyError(f"Legacy {label} is not a directory: {absolute}")
    return absolute


def _legacy_scene_adoption_contract(
    *,
    run_dir: Path,
    request_path: Path,
    run_state_path: Path,
    expected_request_sha256: str,
    request_bytes: bytes | None = None,
    run_state_bytes: bytes | None = None,
) -> LegacySceneMigrationPlan:
    """Validate a genuine v1 request/state pair before any trust migration."""

    normalized_expected = expected_request_sha256.strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", normalized_expected) is None:
        raise SceneLauncherPolicyError(
            "Legacy scene adoption requires a 64-character hexadecimal request "
            "SHA-256 digest."
        )

    if request_bytes is None:
        request_bytes = _read_scene_protected_file(
            request_path,
            label="protected legacy scene request",
            max_bytes=MAX_SCENE_REQUEST_BYTES,
        )
    request_sha256 = hashlib.sha256(request_bytes).hexdigest()
    if request_sha256 != normalized_expected:
        raise SceneLauncherPolicyError(
            "Legacy scene request digest does not match the operator-supplied "
            f"SHA-256; refusing adoption: {request_path}"
        )
    request_document = _decode_scene_json_object(
        request_bytes,
        label=f"legacy scene request at {request_path}",
    )
    runtime_document = request_document.get("runtime")
    task_documents = request_document.get("tasks")
    if (
        request_document.get("schema_version") != LEGACY_SCENE_REQUEST_SCHEMA_VERSION
        or not isinstance(runtime_document, dict)
        or "scene_backend" in runtime_document
        or not isinstance(task_documents, list)
        or any(
            not isinstance(task, dict)
            or "scene_backend" in task
            or "scene_session_scope" in task
            for task in task_documents
        )
    ):
        raise SceneLauncherPolicyError(
            "Legacy scene adoption accepts only a genuine v1 request without "
            "v2 backend or session-scope fields."
        )
    try:
        request = SceneRunRequest.model_validate(request_document)
    except ValueError as exc:
        raise SceneLauncherPolicyError(
            f"Invalid legacy scene request at {request_path}: {exc}"
        ) from exc

    if run_state_bytes is None:
        run_state_bytes = _read_scene_protected_file(
            run_state_path,
            label="protected legacy scene run state",
            max_bytes=MAX_SCENE_RUN_STATE_BYTES,
        )
    run_state_document = _decode_scene_json_object(
        run_state_bytes,
        label=f"legacy scene run state at {run_state_path}",
    )
    if (
        run_state_document.get("schema_version")
        != LEGACY_LARGE_SCENE_RUN_SCHEMA_VERSION
        or "scene_backend" in run_state_document
    ):
        raise SceneLauncherPolicyError(
            "Legacy scene adoption accepts only genuine v1 state without the "
            "v2 scene_backend field."
        )
    try:
        run_state = LargeSceneRun.model_validate(run_state_document)
    except ValueError as exc:
        raise SceneLauncherPolicyError(
            f"Invalid legacy scene run state at {run_state_path}: {exc}"
        ) from exc

    if request.runtime.scene_backend != SCENE_BACKEND_USD_CLI or any(
        task.scene_backend != SCENE_BACKEND_USD_CLI
        or task.scene_session_scope != "per_asset"
        for task in request.tasks
    ):
        raise SceneLauncherPolicyError(
            "Historical v1 scene runs must migrate to the usd-cli/per-asset route."
        )
    if run_state.scene_backend != SCENE_BACKEND_USD_CLI:
        raise SceneLauncherPolicyError(
            "Historical v1 scene state must migrate to the usd-cli backend."
        )

    resolved_run_dir = run_dir.expanduser().resolve(strict=True)
    path_mismatches: list[str] = []
    if Path(request.run_dir).expanduser().resolve() != resolved_run_dir:
        path_mismatches.append("request.run_dir")
    if Path(request.run_state).expanduser().resolve() != run_state_path:
        path_mismatches.append("request.run_state")
    if run_state.run_id != request.run_id:
        path_mismatches.append("run_id")
    if (
        Path(run_state.source_scene).expanduser().resolve()
        != Path(request.source_scene).expanduser().resolve()
    ):
        path_mismatches.append("source_scene")
    if run_state.requested_tasks != request.requested_tasks:
        path_mismatches.append("requested_tasks")
    if [task.domain for task in request.tasks] != request.requested_tasks:
        path_mismatches.append("tasks")
    if run_state.additional_instructions != request.additional_instructions:
        path_mismatches.append("additional_instructions")
    legacy_request_inputs = _legacy_request_input_paths(
        request,
        request_path=request_path,
    )
    lexical_request_artifacts = [
        _lexical_absolute_path(Path(path)) for path in run_state.request_artifact_paths
    ]
    resolved_request_artifacts = {path.resolve() for path in lexical_request_artifacts}
    if request_path not in resolved_request_artifacts:
        path_mismatches.append("request_artifact_paths")
    expected_request_artifacts = {
        _lexical_absolute_path(path).resolve() for path in legacy_request_inputs
    }
    if resolved_request_artifacts != expected_request_artifacts or len(
        lexical_request_artifacts
    ) != len(resolved_request_artifacts):
        path_mismatches.append("request_artifact_inventory")
    if path_mismatches:
        raise SceneLauncherPolicyError(
            "Legacy scene request/state contract mismatch: "
            + ", ".join(path_mismatches)
        )

    _require_legacy_input_without_links(
        Path(request.source_scene),
        label="source scene",
    )
    for index, path in enumerate(legacy_request_inputs, start=1):
        _require_legacy_input_without_links(
            path,
            label=f"request input {index}",
        )
    _require_legacy_directory_without_links(
        Path(request.repository_root),
        label="repository root",
    )
    _require_legacy_directory_without_links(
        Path(request.agent_workspace),
        label="agent workspace",
    )
    for index, value in enumerate(request.references.directories, start=1):
        _require_legacy_directory_without_links(
            Path(value),
            label=f"reference directory {index}",
        )

    try:
        source_input_digest = verify_run_source_inputs(run_state)
    except Exception as exc:
        raise SceneLauncherPolicyError(
            "Legacy scene source or request artifacts do not match the frozen v1 "
            f"input digest: {exc}"
        ) from exc
    decomposition = run_state.phases["decomposition"]
    if decomposition.input_digest != source_input_digest:
        raise SceneLauncherPolicyError(
            "Legacy scene decomposition input digest does not match the verified "
            "v1 source/request digest."
        )
    if (
        not run_state.transitions
        or run_state.transitions[0].phase != "decomposition"
        or run_state.transitions[0].input_digest != source_input_digest
    ):
        raise SceneLauncherPolicyError(
            "Legacy scene state does not contain the expected v1 creation "
            "transition digest."
        )

    source_scene = Path(request.source_scene).expanduser().resolve(strict=True)
    return LegacySceneMigrationPlan(
        request_bytes_before=request_bytes,
        request=request,
        run_state_bytes_before=run_state_bytes,
        run_state=run_state,
        source_scene=source_scene,
        source_sha256=file_sha256(source_scene),
        source_input_digest_before=source_input_digest,
    )


def _scene_model_bytes(model: BaseModel) -> bytes:
    return (model.model_dump_json(indent=2) + "\n").encode("utf-8")


def _normalize_legacy_scene_request(
    plan: LegacySceneMigrationPlan,
    *,
    run_dir: Path,
) -> SceneRunRequest:
    repository_root = Path(plan.request.repository_root).expanduser().resolve()
    trusted_agent_workspace = (repository_root / "agentic").resolve()
    legacy_agent_workspace = Path(plan.request.agent_workspace).expanduser().resolve()
    if legacy_agent_workspace not in {trusted_agent_workspace, run_dir}:
        raise SceneLauncherPolicyError(
            "Legacy scene request has an unrecognized agent workspace: "
            f"{legacy_agent_workspace}"
        )
    document = plan.request.model_dump(mode="json")
    document.update(
        {
            "schema_version": SCENE_REQUEST_SCHEMA_VERSION,
            "agent_workspace": str(trusted_agent_workspace),
            "child_workspace": str(run_dir),
        }
    )
    return SceneRunRequest.model_validate(document)


def _normalize_legacy_scene_state(
    plan: LegacySceneMigrationPlan,
    *,
    request_path: Path,
) -> LargeSceneRun:
    source_input_digest_after = _source_input_digest(
        plan.run_state.source_scene,
        plan.run_state.request_artifact_paths,
        plan.run_state.requested_tasks,
        plan.run_state.additional_instructions,
        scene_backend=SCENE_BACKEND_USD_CLI,
        schema_version=LARGE_SCENE_RUN_SCHEMA_VERSION,
    )
    document = plan.run_state.model_dump(mode="json")
    document.update(
        {
            "schema_version": LARGE_SCENE_RUN_SCHEMA_VERSION,
            "scene_backend": SCENE_BACKEND_USD_CLI,
            "revision": plan.run_state.revision + 1,
            "source_input_digest": source_input_digest_after,
        }
    )
    decomposition = document["phases"]["decomposition"]
    decomposition["input_digest"] = source_input_digest_after
    for transition in document["transitions"]:
        if (
            transition.get("phase") == "decomposition"
            and transition.get("input_digest") == plan.source_input_digest_before
        ):
            transition["input_digest"] = source_input_digest_after
    migrated = LargeSceneRun.model_validate(document)
    if request_path not in {
        Path(value).expanduser().resolve() for value in migrated.request_artifact_paths
    }:
        raise SceneLauncherPolicyError(
            "Migrated scene state lost the canonical request artifact path."
        )
    return migrated


def _write_parent_owned_scene_migration_intent(
    path: Path,
    intent: LegacySceneMigrationIntent,
) -> None:
    contents = _scene_model_bytes(intent)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise FileExistsError(
            f"Refusing to replace existing legacy migration intent: {path}"
        ) from exc
    except OSError as exc:
        raise SceneLauncherPolicyError(
            f"Unable to create legacy migration intent safely at {path}: {exc}"
        ) from exc
    with os.fdopen(descriptor, "wb") as stream:
        if hasattr(os, "fchmod"):
            os.fchmod(stream.fileno(), 0o600)
        stream.write(contents)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        fsync_directory(path.parent)
    except OSError as exc:
        raise SceneLauncherPolicyError(
            "Unable to durably sync the legacy migration intent directory at "
            f"{path.parent}: {exc}"
        ) from exc


def _read_legacy_scene_migration_intent(path: Path) -> LegacySceneMigrationIntent:
    contents = _read_scene_protected_file(
        path,
        label="legacy scene migration intent",
        max_bytes=MAX_LEGACY_SCENE_ADOPTION_BYTES,
    )
    try:
        return LegacySceneMigrationIntent.model_validate_json(contents)
    except ValueError as exc:
        raise SceneLauncherPolicyError(
            f"Invalid legacy scene migration intent at {path}: {exc}"
        ) from exc


def _write_or_verify_legacy_evidence(
    path: Path,
    contents: bytes,
    *,
    run_dir: Path,
    label: str,
) -> None:
    if path.exists() or path.is_symlink():
        recorded = _read_scene_protected_file(
            path,
            label=label,
            max_bytes=max(MAX_SCENE_REQUEST_BYTES, MAX_SCENE_RUN_STATE_BYTES),
        )
        if recorded != contents:
            raise SceneLauncherPolicyError(
                f"{label.capitalize()} does not match the validated legacy bytes."
            )
        return
    atomic_write_bytes(path, contents, within=run_dir)
    _chmod_private(path, within=run_dir)


def _read_legacy_scene_adoption_receipt(
    receipt_path: Path,
) -> LegacySceneAdoptionReceipt:
    receipt_bytes = _read_scene_protected_file(
        receipt_path,
        label="legacy scene adoption receipt",
        max_bytes=MAX_LEGACY_SCENE_ADOPTION_BYTES,
    )
    try:
        return LegacySceneAdoptionReceipt.model_validate_json(receipt_bytes)
    except ValueError as exc:
        raise SceneLauncherPolicyError(
            f"Invalid legacy scene adoption receipt at {receipt_path}: {exc}"
        ) from exc


def _require_matching_scene_model(
    actual: BaseModel,
    expected: BaseModel,
    *,
    label: str,
) -> None:
    if actual.model_dump(mode="json") != expected.model_dump(mode="json"):
        raise SceneLauncherPolicyError(
            f"{label.capitalize()} does not match the validated migration plan."
        )


def _adopt_legacy_scene_run(
    *,
    run_dir: Path,
    request_path: Path,
    run_state_path: Path,
    expected_request_sha256: str,
) -> WorkflowRunRecorder:
    """Crash-resumably migrate an explicitly authorized pre-policy v1 run."""

    with _scene_run_initialization_lock(run_dir):
        _reject_unsafe_run_links(run_dir)
        receipt_path = _legacy_scene_adoption_receipt_path(run_dir)
        intent_path = _legacy_scene_migration_intent_path(run_dir)
        legacy_request_path, legacy_state_path = _legacy_scene_evidence_paths(run_dir)
        manifest_path = run_dir / "workflow_run_manifest.json"
        policy_exists = _scene_launcher_policy_identity(run_dir) is not None
        receipt_exists = receipt_path.exists() or receipt_path.is_symlink()
        intent_exists = intent_path.exists() or intent_path.is_symlink()
        manifest_exists = manifest_path.exists() or manifest_path.is_symlink()

        if not intent_exists and (receipt_exists or policy_exists or manifest_exists):
            raise SceneLauncherPolicyError(
                "Refusing legacy adoption of a mixed or partially modern run: "
                "migration intent is missing."
            )
        if intent_exists and not receipt_exists and (policy_exists or manifest_exists):
            raise SceneLauncherPolicyError(
                "Refusing legacy adoption of an invalid partial migration: policy "
                "or manifest exists before the completed adoption receipt."
            )

        if intent_exists:
            intent = _read_legacy_scene_migration_intent(intent_path)
            normalized_expected = expected_request_sha256.strip().lower()
            if intent.request_sha256_before != normalized_expected:
                raise SceneLauncherPolicyError(
                    "Legacy migration intent does not match the operator-supplied "
                    "request SHA-256."
                )
            if (
                Path(intent.run_dir).expanduser().resolve() != run_dir
                or Path(intent.legacy_request_artifact).expanduser().resolve()
                != legacy_request_path
                or Path(intent.legacy_run_state_artifact).expanduser().resolve()
                != legacy_state_path
            ):
                raise SceneLauncherPolicyError(
                    "Legacy migration intent contains an invalid run or evidence path."
                )
            request_before = _read_scene_protected_file(
                legacy_request_path,
                label="legacy request evidence",
                max_bytes=MAX_SCENE_REQUEST_BYTES,
            )
            state_before = _read_scene_protected_file(
                legacy_state_path,
                label="legacy run-state evidence",
                max_bytes=MAX_SCENE_RUN_STATE_BYTES,
            )
            live_request = _read_scene_protected_file(
                request_path,
                label="scene request during legacy migration",
                max_bytes=MAX_SCENE_REQUEST_BYTES,
            )
            if hashlib.sha256(live_request).hexdigest() not in {
                intent.request_sha256_before,
                intent.request_sha256_after,
            }:
                raise SceneLauncherPolicyError(
                    "Scene request changed outside the recorded legacy migration."
                )
            # Recreate the exact v1 input boundary before recomputing its digest.
            # The parent-owned intent makes this replacement recoverable if the
            # process stops before the v2 request is republished below.
            atomic_write_bytes(request_path, request_before, within=run_dir)
            plan = _legacy_scene_adoption_contract(
                run_dir=run_dir,
                request_path=request_path,
                run_state_path=run_state_path,
                expected_request_sha256=expected_request_sha256,
                request_bytes=request_before,
                run_state_bytes=state_before,
            )
        else:
            plan = _legacy_scene_adoption_contract(
                run_dir=run_dir,
                request_path=request_path,
                run_state_path=run_state_path,
                expected_request_sha256=expected_request_sha256,
            )
            _prepare_contained_run_subdirectory(
                run_dir,
                legacy_request_path.parent,
                create=True,
            )
            _write_or_verify_legacy_evidence(
                legacy_request_path,
                plan.request_bytes_before,
                run_dir=run_dir,
                label="legacy request evidence",
            )
            _write_or_verify_legacy_evidence(
                legacy_state_path,
                plan.run_state_bytes_before,
                run_dir=run_dir,
                label="legacy run-state evidence",
            )

        migrated_request = _normalize_legacy_scene_request(plan, run_dir=run_dir)
        migrated_request_bytes = _scene_model_bytes(migrated_request)
        expected_intent = LegacySceneMigrationIntent(
            prepared_at=(intent.prepared_at if intent_exists else utc_now()),
            run_dir=str(run_dir),
            request_sha256_before=hashlib.sha256(plan.request_bytes_before).hexdigest(),
            request_sha256_after=hashlib.sha256(migrated_request_bytes).hexdigest(),
            run_state_sha256_before=hashlib.sha256(
                plan.run_state_bytes_before
            ).hexdigest(),
            legacy_request_artifact=str(legacy_request_path),
            legacy_run_state_artifact=str(legacy_state_path),
            source_scene=str(plan.source_scene),
            source_sha256=plan.source_sha256,
            source_input_digest_before=plan.source_input_digest_before,
        )
        if intent_exists:
            _require_matching_scene_model(
                intent,
                expected_intent,
                label="legacy migration intent",
            )
        else:
            _write_parent_owned_scene_migration_intent(intent_path, expected_intent)
            intent = expected_intent
        intent_bytes = _read_scene_protected_file(
            intent_path,
            label="legacy scene migration intent",
            max_bytes=MAX_LEGACY_SCENE_ADOPTION_BYTES,
        )
        try:
            persisted_intent = LegacySceneMigrationIntent.model_validate_json(
                intent_bytes
            )
        except ValueError as exc:
            raise SceneLauncherPolicyError(
                f"Invalid legacy scene migration intent at {intent_path}: {exc}"
            ) from exc
        _require_matching_scene_model(
            persisted_intent,
            expected_intent,
            label="persisted legacy migration intent",
        )
        intent_sha256 = hashlib.sha256(intent_bytes).hexdigest()

        atomic_write_bytes(request_path, migrated_request_bytes, within=run_dir)
        _chmod_private(request_path, within=run_dir)
        migrated_state = _normalize_legacy_scene_state(
            plan,
            request_path=request_path,
        )
        migrated_state_bytes = _scene_model_bytes(migrated_state)
        source_input_digest_after = migrated_state.source_input_digest
        receipt = (
            _read_legacy_scene_adoption_receipt(receipt_path)
            if receipt_exists
            else None
        )
        expected_receipt = LegacySceneAdoptionReceipt(
            adopted_at=receipt.adopted_at if receipt is not None else utc_now(),
            run_dir=str(run_dir),
            request_sha256_before=intent.request_sha256_before,
            request_sha256_after=intent.request_sha256_after,
            run_state_sha256_before=intent.run_state_sha256_before,
            run_state_sha256_after=hashlib.sha256(migrated_state_bytes).hexdigest(),
            legacy_request_artifact=str(legacy_request_path),
            legacy_run_state_artifact=str(legacy_state_path),
            migration_intent_sha256=intent_sha256,
            source_scene=str(plan.source_scene),
            source_sha256=plan.source_sha256,
            source_input_digest_before=plan.source_input_digest_before,
            source_input_digest_after=source_input_digest_after,
        )
        if receipt is not None:
            _require_matching_scene_model(
                receipt,
                expected_receipt,
                label="legacy scene adoption receipt",
            )
        else:
            atomic_write_json(receipt_path, expected_receipt, within=run_dir)
            _chmod_private(receipt_path, within=run_dir)
        receipt_bytes = _read_scene_protected_file(
            receipt_path,
            label="legacy scene adoption receipt",
            max_bytes=MAX_LEGACY_SCENE_ADOPTION_BYTES,
        )
        try:
            persisted_receipt = LegacySceneAdoptionReceipt.model_validate_json(
                receipt_bytes
            )
        except ValueError as exc:
            raise SceneLauncherPolicyError(
                f"Invalid legacy scene adoption receipt at {receipt_path}: {exc}"
            ) from exc
        _require_matching_scene_model(
            persisted_receipt,
            expected_receipt,
            label="persisted legacy scene adoption receipt",
        )
        receipt_sha256 = hashlib.sha256(receipt_bytes).hexdigest()

        live_state = _read_scene_protected_file(
            run_state_path,
            label="scene state during legacy migration",
            max_bytes=MAX_SCENE_RUN_STATE_BYTES,
        )
        if hashlib.sha256(live_state).hexdigest() not in {
            intent.run_state_sha256_before,
            expected_receipt.run_state_sha256_after,
        }:
            raise SceneLauncherPolicyError(
                "Scene run state changed outside the recorded legacy migration."
            )
        atomic_write_bytes(run_state_path, migrated_state_bytes, within=run_dir)
        _chmod_private(run_state_path, within=run_dir)

        if policy_exists:
            protected_request = _verify_scene_launcher_policy(
                run_dir,
                request_path=request_path,
            )
            if protected_request != migrated_request_bytes:
                raise SceneLauncherPolicyError(
                    "Migrated scene request differs from its launcher policy."
                )
        else:
            _write_scene_launcher_policy(
                run_dir,
                request_bytes=migrated_request_bytes,
                legacy_adoption_receipt_sha256=receipt_sha256,
            )

        contract = _scene_run_recorder_contract(
            run_dir=run_dir,
            request=migrated_request,
            source_path=plan.source_scene,
        )
        if manifest_exists:
            return WorkflowRunRecorder.start(run_dir, **contract, resume=True)
        return WorkflowRunRecorder.adopt(run_dir, **contract)


def _read_scene_protected_file(
    path: Path,
    *,
    label: str,
    max_bytes: int,
) -> bytes:
    """Read a bounded, single-link regular file through a no-follow descriptor."""

    path_metadata: os.stat_result | None = None
    if os.name == "nt":
        try:
            path_metadata = path.lstat()
        except OSError as exc:
            raise SceneLauncherPolicyError(
                f"Unable to read {label} safely at {path}: {exc}"
            ) from exc
        if _scene_metadata_is_reparse_point(path_metadata):
            raise SceneLauncherPolicyError(
                f"Unable to read {label} safely at {path}: "
                "reparse points are not allowed"
            )

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise SceneLauncherPolicyError(
            f"Unable to read {label} safely at {path}: {exc}"
        ) from exc

    with os.fdopen(descriptor, "rb") as stream:
        metadata = os.fstat(stream.fileno())
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise SceneLauncherPolicyError(
                f"{label.capitalize()} must be a single-link regular file: {path}"
            )
        if path_metadata is not None and (
            not stat.S_ISREG(path_metadata.st_mode)
            or path_metadata.st_nlink != 1
            or (path_metadata.st_dev, path_metadata.st_ino)
            != (metadata.st_dev, metadata.st_ino)
        ):
            raise SceneLauncherPolicyError(
                f"Unable to read {label} safely at {path}: "
                "the directory entry changed before it was opened"
            )
        if metadata.st_size > max_bytes:
            raise SceneLauncherPolicyError(
                f"{label.capitalize()} exceeds the {max_bytes}-byte limit: {path}"
            )
        contents = stream.read(max_bytes + 1)
        final_metadata = os.fstat(stream.fileno())
        initial_identity = (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
            metadata.st_nlink,
        )
        final_identity = (
            final_metadata.st_dev,
            final_metadata.st_ino,
            final_metadata.st_size,
            final_metadata.st_mtime_ns,
            final_metadata.st_ctime_ns,
            final_metadata.st_nlink,
        )
        if final_metadata.st_nlink != 1:
            raise SceneLauncherPolicyError(
                f"{label.capitalize()} must remain a single-link regular file: {path}"
            )
        if final_identity != initial_identity:
            raise SceneLauncherPolicyError(
                f"{label.capitalize()} changed while it was being read: {path}"
            )
        if os.name == "nt":
            try:
                current_metadata = path.lstat()
            except OSError as exc:
                raise SceneLauncherPolicyError(
                    f"Unable to read {label} safely at {path}: {exc}"
                ) from exc
            if _scene_metadata_is_reparse_point(current_metadata) or (
                current_metadata.st_dev,
                current_metadata.st_ino,
            ) != (final_metadata.st_dev, final_metadata.st_ino):
                raise SceneLauncherPolicyError(
                    f"Unable to read {label} safely at {path}: "
                    "the directory entry changed or is a reparse point"
                )
    if len(contents) > max_bytes:
        raise SceneLauncherPolicyError(
            f"{label.capitalize()} exceeds the {max_bytes}-byte limit: {path}"
        )
    return contents


def _request_input_artifacts(
    config: SceneRunConfig,
    request_path: Path,
    *,
    staged_inputs: list[Path] | None = None,
) -> list[Path]:
    paths = [
        request_path,
        config.usd_path,
        *(staged_inputs or []),
        *config.additional_instruction_sources,
        *config.reference_images,
        *config.reference_files,
    ]
    if config.materials_yaml is not None:
        paths.append(config.materials_yaml)
    if config.materials_usd is not None:
        paths.append(config.materials_usd)
    return list(dict.fromkeys(path.resolve() for path in paths))


def _prepare_resumable_phase(run_state_path: Path) -> None:
    run = load_run_state(run_state_path)
    phase = run.current_phase
    if phase is None:
        return
    status = run.phases[phase].status
    if status in {"running", "failed"}:
        invalidate_from(
            run_state_path,
            phase,
            reason="Batch launcher resumed an interrupted or failed phase.",
            actor="content-workflow-cli",
        )
    elif status != "ready":
        raise RuntimeError(
            f"Cannot resume phase {phase} from status {status}; expected ready, running, or failed"
        )


def _validate_terminal_state(run_state_path: Path) -> SceneTerminalValidation:
    errors: list[str] = []
    final_handoff: dict[str, Any] | None = None
    try:
        run = load_run_state(run_state_path)
    except Exception as exc:  # noqa: BLE001 - report validation, do not hide it
        return SceneTerminalValidation(
            checked_at=utc_now(),
            valid=False,
            current_phase=None,
            phase_statuses={},
            errors=[f"Cannot load run state: {exc}"],
        )

    statuses = {phase: state.status for phase, state in run.phases.items()}
    try:
        verify_run_source_inputs(run)
    except Exception as exc:  # noqa: BLE001 - report the frozen-input gate
        errors.append(f"Frozen scene source or request input changed: {exc}")
    if run.current_phase is not None:
        errors.append(f"current_phase is still {run.current_phase}")
    incomplete = [phase for phase, status in statuses.items() if status != "completed"]
    if incomplete:
        errors.append("Phases are not completed: " + ", ".join(incomplete))

    collection = run.phases.get("collection")
    if collection is None:
        errors.append("Run state is missing collection phase")
    elif collection.status == "completed" and collection.result_path:
        try:
            report = validate_phase_handoff(
                run_state_path,
                "collection",
                collection.result_path,
            )
            final_handoff = report.model_dump(mode="json")
            if not report.valid:
                errors.extend(report.errors)
        except Exception as exc:  # noqa: BLE001 - convert gate failure to report
            errors.append(f"Final collection handoff validation failed: {exc}")
    else:
        errors.append("Collection has no completed result to validate")

    errors.extend(_staged_input_integrity_errors(run_state_path.parent))

    return SceneTerminalValidation(
        checked_at=utc_now(),
        valid=not errors,
        current_phase=run.current_phase,
        phase_statuses=statuses,
        errors=errors,
        final_handoff=final_handoff,
    )


def _write_terminal_validation(
    run_dir: Path,
    terminal: SceneTerminalValidation,
) -> Path:
    path = run_dir / "terminal_validation.json"
    return atomic_write_json(path, terminal, within=run_dir)


def _agent_workspace(config: SceneRunConfig) -> Path:
    """Return the workspace that provides the large-scene skills."""

    return (config.agent_workspace or config.repo_root / "agentic").resolve()


def _confined_child_workspace(candidate: Path | None, *, run_dir: Path) -> Path:
    """Reject persisted or caller-provided child workspaces outside the run."""

    resolved_run_dir = run_dir.resolve()
    if candidate is not None and candidate.resolve() != resolved_run_dir:
        raise ValueError(
            "Scene child_workspace must resolve to the run directory: "
            f"{candidate.resolve()} != {resolved_run_dir}"
        )
    return resolved_run_dir


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return slug or "scene"
