# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public batch launcher for the agent-driven large-scene workflow."""

from __future__ import annotations

import hashlib
import os
import re
import stat
import time
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from content_agent_workflows.large_scene import (
    create_run,
    invalidate_from,
    load_run_state,
    validate_phase_handoff,
)
from pydantic import BaseModel, ConfigDict, Field

from .runner import (
    CLAUDE_EXECUTION_CLI,
    CLAUDE_EXECUTION_SDK,
    CODEX_SANDBOX_WORKSPACE_WRITE,
    SUPPORTED_CLAUDE_EXECUTION_MODES,
    SUPPORTED_CODEX_SANDBOX_MODES,
    SUPPORTED_RUNNERS,
    ManagedWorkbench,
    _append_child_runner_error,
    _chmod_private,
    _is_loopback_host,
    _lexical_absolute_path,
    _reject_unsafe_run_links,
    _run_child_agent,
    _trace_path_summary,
    _write_run_cost_metrics,
    wait_for_workbench,
)
from .trace import TraceWriter, UnsafeRunArtifactError, build_trace, utc_now

SCENE_REQUEST_SCHEMA_VERSION = "content-agents.large-scene-request.v1"
SCENE_TERMINAL_VALIDATION_SCHEMA_VERSION = (
    "content-agents.large-scene-terminal-validation.v1"
)
SCENE_LAUNCHER_POLICY_SCHEMA_VERSION = "content-agents.scene-launcher-policy.v1"
MAX_SCENE_LAUNCHER_POLICY_BYTES = 256 * 1024
MAX_SCENE_REQUEST_BYTES = 4 * 1024 * 1024


class SceneReferencesRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    directories: list[str] = Field(default_factory=list)
    images: list[str] = Field(default_factory=list)
    files: list[str] = Field(default_factory=list)


class SceneTaskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    domain: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    policy: dict[str, Any] = Field(default_factory=dict)


class SceneRuntimeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    runner: str
    model: str | None = None
    model_reasoning_effort: str | None = None
    workbench_url: str
    start_workbench: bool
    keep_workbench: bool
    workbench_timeout_seconds: float
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

    schema_version: str = SCENE_REQUEST_SCHEMA_VERSION
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
    requested_tasks: list[str]
    references: SceneReferencesRequest
    additional_instructions: str | None = None
    additional_instruction_sources: list[str] = Field(default_factory=list)
    tasks: list[SceneTaskRequest]
    decomposition: dict[str, Any]
    collection: dict[str, Any]
    runtime: SceneRuntimeRequest


class SceneLauncherPolicy(BaseModel):
    """Parent-owned integrity policy for a resumable scene request."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[SCENE_LAUNCHER_POLICY_SCHEMA_VERSION] = (
        SCENE_LAUNCHER_POLICY_SCHEMA_VERSION
    )
    run_dir: str
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


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
    repo_root: Path
    usd_path: Path
    requested_tasks: list[str]
    workbench_url: str
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
    start_workbench: bool = True
    keep_workbench: bool = False
    workbench_timeout_seconds: float = 60.0
    child_timeout_seconds: float = 1800.0
    dry_run: bool = False
    agent_workspace: Path | None = None
    agent_cwd: Path | None = None


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


def run_scene_workflow(config: SceneRunConfig) -> SceneRunResult:
    """Create and optionally execute a fresh three-phase large-scene run."""

    _validate_scene_config(config)
    run_id, run_dir = _prepare_scene_run_dir(config)
    config = replace(
        config,
        agent_cwd=_confined_child_workspace(config.agent_cwd, run_dir=run_dir),
    )
    request_path = run_dir / "request.json"
    run_state_path = run_dir / "large_scene_run.json"
    prompt_path = run_dir / "agent_prompt.md"
    child_output_path = run_dir / "child-output.log"
    child_final_path = run_dir / "child-final.md"

    if request_path.exists() or run_state_path.exists():
        raise FileExistsError(f"Large-scene run already exists: {run_dir}")

    request = _build_scene_request(
        config,
        run_id=run_id,
        run_dir=run_dir,
        run_state_path=run_state_path,
    )
    request_bytes = _write_request(request_path, request)
    _write_scene_launcher_policy(run_dir, request_bytes=request_bytes)
    input_artifacts = _request_input_artifacts(config, request_path)
    create_run(
        run_state_path,
        run_id=run_id,
        source_scene=config.usd_path,
        requested_tasks=config.requested_tasks,
        request_artifact_paths=input_artifacts,
        additional_instructions=config.additional_instructions,
        actor="content-workflow-cli",
    )

    prompt = _build_scene_agent_prompt(
        request_path=request_path,
        run_state_path=run_state_path,
        run_dir=run_dir,
        resume=False,
    )
    prompt_path.write_text(prompt, encoding="utf-8")

    trace_writer = TraceWriter(run_dir)
    trace_writer.write(
        "run_created",
        phase="setup",
        summary="Created a large-scene batch request and durable phase state.",
        artifacts=[str(request_path), str(run_state_path), str(prompt_path)],
        data={"run_id": run_id, "workflow": request.workflow},
    )

    if config.dry_run:
        trace_paths = build_trace(run_dir)
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
    )


def resume_scene_workflow(run_dir: Path, *, dry_run: bool = False) -> SceneRunResult:
    """Resume a previously prepared or interrupted large-scene run."""

    lexical_run_dir = run_dir.expanduser()
    _reject_unsafe_run_links(lexical_run_dir)
    resolved_run_dir = lexical_run_dir.resolve()
    request_path = resolved_run_dir / "request.json"
    run_state_path = resolved_run_dir / "large_scene_run.json"
    if not request_path.is_file():
        raise FileNotFoundError(f"Large-scene request does not exist: {request_path}")
    if not run_state_path.is_file():
        raise FileNotFoundError(
            f"Large-scene run state does not exist: {run_state_path}"
        )

    request_bytes = _verify_scene_launcher_policy(
        resolved_run_dir,
        request_path=request_path,
    )
    request = SceneRunRequest.model_validate_json(request_bytes)
    if Path(request.run_dir).resolve() != resolved_run_dir:
        raise ValueError(
            "Resolved request run_dir does not match --run-dir: "
            f"{request.run_dir} != {resolved_run_dir}"
        )
    config = _config_from_request(request, dry_run=dry_run)
    _validate_scene_config(config)

    prompt_path = resolved_run_dir / "agent_resume_prompt.md"
    child_output_path = resolved_run_dir / "child-resume-output.log"
    child_final_path = resolved_run_dir / "child-resume-final.md"
    prompt = _build_scene_agent_prompt(
        request_path=request_path,
        run_state_path=run_state_path,
        run_dir=resolved_run_dir,
        resume=True,
    )
    prompt_path.write_text(prompt, encoding="utf-8")
    trace_writer = TraceWriter(resolved_run_dir)
    trace_writer.write(
        "run_resume_requested",
        phase="setup",
        summary="Prepared a child agent to resume durable large-scene state.",
        artifacts=[str(request_path), str(run_state_path), str(prompt_path)],
        data={"run_id": request.run_id, "dry_run": dry_run},
    )

    terminal = _validate_terminal_state(run_state_path)
    if terminal.valid:
        terminal_path = _write_terminal_validation(resolved_run_dir, terminal)
        trace_writer.write(
            "run_already_complete",
            phase="validation",
            summary="The large-scene run was already complete and valid.",
            artifacts=[str(terminal_path)],
        )
        trace_paths = build_trace(resolved_run_dir)
        return SceneRunResult(
            run_dir=resolved_run_dir,
            request_path=request_path,
            run_state_path=run_state_path,
            prompt_path=prompt_path,
            child_output_path=child_output_path,
            child_final_path=child_final_path,
            terminal_validation_path=terminal_path,
            returncode=0,
            completed=True,
            trace_paths=_trace_path_summary(trace_paths),
        )

    if dry_run:
        trace_paths = build_trace(resolved_run_dir)
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
    )


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
) -> SceneRunResult:
    run_started = time.monotonic()
    managed_workbench: ManagedWorkbench | None = None
    child_returncode = 2
    try:
        if config.start_workbench:
            managed_workbench = ManagedWorkbench(
                repo_root=config.repo_root,
                workbench_url=config.workbench_url,
                run_dir=Path(request.run_dir),
                timeout_seconds=config.workbench_timeout_seconds,
                material_library_roots=_workbench_roots(config),
            )
            managed_workbench.start()
            trace_writer.write(
                "workbench_started",
                phase="setup",
                summary="Started or reused Content Workbench for the scene run.",
                artifacts=[str(Path(request.run_dir) / "workbench.log")],
                data={"workbench_url": config.workbench_url},
            )
        else:
            wait_for_workbench(
                config.workbench_url,
                timeout_seconds=config.workbench_timeout_seconds,
                output_root=Path(request.run_dir),
            )
            trace_writer.write(
                "workbench_reachable",
                phase="setup",
                summary="Verified the configured Content Workbench endpoint.",
                data={"workbench_url": config.workbench_url},
            )

        child_returncode = _run_child_agent(
            config=config,
            prompt=prompt,
            run_dir=Path(request.run_dir),
            child_output_path=child_output_path,
            child_final_path=child_final_path,
            managed_workbench=managed_workbench,
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
    except Exception as exc:  # noqa: BLE001 - preserve partial run artifacts
        child_returncode = 2
        _append_child_runner_error(
            child_output_path,
            exc,
            run_dir=Path(request.run_dir),
        )
        trace_writer.write(
            "child_agent_failed",
            phase="runner",
            summary="Large-scene child-agent runner failed.",
            artifacts=[str(child_output_path), str(child_final_path)],
            data={"error_type": type(exc).__name__, "error": str(exc)},
        )
    finally:
        if managed_workbench is not None and not config.keep_workbench:
            try:
                managed_workbench.stop()
                trace_writer.write(
                    "workbench_stopped",
                    phase="cleanup",
                    summary="Stopped the managed Content Workbench service.",
                    artifacts=[str(Path(request.run_dir) / "workbench.log")],
                )
            except Exception as exc:  # noqa: BLE001 - cleanup is best effort
                trace_writer.write(
                    "warning",
                    phase="cleanup",
                    summary="Failed to stop managed Content Workbench service.",
                    data={"error_type": type(exc).__name__, "error": str(exc)},
                )

    terminal = _validate_terminal_state(run_state_path)
    terminal_path = _write_terminal_validation(Path(request.run_dir), terminal)
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
    return SceneRunResult(
        run_dir=Path(request.run_dir),
        request_path=request_path,
        run_state_path=run_state_path,
        prompt_path=prompt_path,
        child_output_path=child_output_path,
        child_final_path=child_final_path,
        terminal_validation_path=terminal_path,
        returncode=returncode,
        completed=terminal.valid,
        trace_paths=_trace_path_summary(trace_paths),
    )


def _build_scene_request(
    config: SceneRunConfig,
    *,
    run_id: str,
    run_dir: Path,
    run_state_path: Path,
) -> SceneRunRequest:
    tasks: list[SceneTaskRequest] = []
    for domain in config.requested_tasks:
        if domain == "material":
            tasks.append(
                SceneTaskRequest(
                    domain=domain,
                    inputs={
                        "materials_yaml": str(config.materials_yaml),
                        "materials_usd": str(config.materials_usd),
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
            tasks.append(SceneTaskRequest(domain=domain))

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
        source_scene=str(config.usd_path),
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
            model=config.model,
            model_reasoning_effort=config.model_reasoning_effort,
            workbench_url=config.workbench_url,
            start_workbench=config.start_workbench,
            keep_workbench=config.keep_workbench,
            workbench_timeout_seconds=config.workbench_timeout_seconds,
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
        usd_path=Path(request.source_scene).resolve(),
        requested_tasks=list(request.requested_tasks),
        workbench_url=request.runtime.workbench_url,
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
        start_workbench=request.runtime.start_workbench,
        keep_workbench=request.runtime.keep_workbench,
        workbench_timeout_seconds=request.runtime.workbench_timeout_seconds,
        child_timeout_seconds=request.runtime.child_timeout_seconds,
        dry_run=dry_run,
    )


def _build_scene_agent_prompt(
    *,
    request_path: Path,
    run_state_path: Path,
    run_dir: Path,
    resume: bool,
) -> str:
    action = "Resume" if resume else "Execute"
    return f"""Use the `content-workflow-large-scene` skill.

{action} every required phase for this batch request:

- Resolved request: `{request_path}`
- Durable run state: `{run_state_path}`
- Run directory: `{run_dir}`

Read the resolved request and run state before acting. Use the task, domain, and
Workbench skills selected by the umbrella skill. The request is frozen launcher
input; do not rewrite it. Never edit `large_scene_run.json` directly. Use the
internal transition helper required by the umbrella skill, validate each
handoff, and do not advance past a failed gate.

Keep semantic decomposition, task planning, per-asset decisions, evidence reuse,
and collection judgment agent-owned. Read scene-level `additional_instructions`
from run state and preserve them exactly in applicable task requests. For
material task requests, keep the clean-slate `appearance_evidence_policy`
unless explicit user guidance asks for scoped display-color or existing-material
evidence on named roots. Write all generated artifacts under the run directory.

Continue until collection completes and `current_phase` becomes null. If a
concrete blocker prevents completion, record the phase failure and clearly
identify the resumable state in the final response.
"""


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
    if config.workbench_timeout_seconds <= 0:
        raise ValueError("--workbench-timeout must be greater than 0.")

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


def _prepare_scene_run_dir(config: SceneRunConfig) -> tuple[str, Path]:
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
        else config.repo_root / "agentic" / "runs" / run_id
    )
    # Check the uncanonicalized path first so a precreated run-directory
    # symlink cannot be hidden by resolve() and granted as the child workspace.
    _reject_unsafe_run_links(run_dir_candidate, allow_missing=True)
    run_dir = _lexical_absolute_path(run_dir_candidate)
    run_dir.mkdir(parents=True, exist_ok=True)
    _reject_unsafe_run_links(run_dir)
    (run_dir / "raw").mkdir(exist_ok=True)
    (run_dir / "trace").mkdir(exist_ok=True)
    _chmod_private(run_dir / "raw")
    return run_id, run_dir


def _write_request(path: Path, request: SceneRunRequest) -> bytes:
    request_bytes = (request.model_dump_json(indent=2) + "\n").encode("utf-8")
    if len(request_bytes) > MAX_SCENE_REQUEST_BYTES:
        raise SceneLauncherPolicyError(
            f"Scene request exceeds the {MAX_SCENE_REQUEST_BYTES}-byte limit: {path}"
        )
    path.write_bytes(request_bytes)
    _chmod_private(path)
    return request_bytes


def _scene_launcher_policy_path(run_dir: Path) -> Path:
    resolved_run_dir = run_dir.expanduser().resolve()
    return (
        resolved_run_dir.parent / f".{resolved_run_dir.name}.scene-launcher-policy.json"
    )


def _write_scene_launcher_policy(run_dir: Path, *, request_bytes: bytes) -> Path:
    """Create the parent-owned policy without replacing or following a symlink."""

    resolved_run_dir = run_dir.expanduser().resolve()
    policy_path = _scene_launcher_policy_path(resolved_run_dir)
    policy = SceneLauncherPolicy(
        run_dir=str(resolved_run_dir),
        request_sha256=hashlib.sha256(request_bytes).hexdigest(),
    )
    policy_bytes = (policy.model_dump_json(indent=2) + "\n").encode("utf-8")
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
    return request_bytes


def _read_scene_protected_file(
    path: Path,
    *,
    label: str,
    max_bytes: int,
) -> bytes:
    """Read a bounded, single-link regular file through a no-follow descriptor."""

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
        if metadata.st_size > max_bytes:
            raise SceneLauncherPolicyError(
                f"{label.capitalize()} exceeds the {max_bytes}-byte limit: {path}"
            )
        contents = stream.read(max_bytes + 1)
        final_metadata = os.fstat(stream.fileno())
        if final_metadata.st_nlink != 1:
            raise SceneLauncherPolicyError(
                f"{label.capitalize()} must remain a single-link regular file: {path}"
            )
    if len(contents) > max_bytes:
        raise SceneLauncherPolicyError(
            f"{label.capitalize()} exceeds the {max_bytes}-byte limit: {path}"
        )
    return contents


def _request_input_artifacts(
    config: SceneRunConfig,
    request_path: Path,
) -> list[Path]:
    paths = [
        request_path,
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
    path.write_text(terminal.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def _workbench_roots(config: SceneRunConfig) -> list[Path]:
    roots = [config.usd_path.parent]
    if config.materials_usd is not None:
        roots.append(config.materials_usd.parent)
    return list(dict.fromkeys(path.resolve() for path in roots))


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
