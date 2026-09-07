# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thin launcher for the decision-only Validation coordinator child."""

from __future__ import annotations

import json
import re
import secrets
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal

from content_agent_workflows.common.artifacts import (
    atomic_write_json,
    atomic_write_text,
    file_sha256,
    load_json,
)
from content_agent_workflows.common.embedded_domain_decision import (
    canonical_json_digest,
)
from content_agent_workflows.validation import (
    VALIDATION_COORDINATOR_EXECUTION_RECEIPT_NAME,
    VALIDATION_COORDINATOR_PLAN_PATCH_NAME,
    VALIDATION_COORDINATOR_PREPARATION_NAME,
    ValidationCoordinatorExecutionReceipt,
    ValidationCoordinatorPlanPatch,
    ValidationWorkflowRun,
    accept_validation_coordinator_plan,
    execute_validation_coordinator_plan,
    prepare_validation_coordinator,
)
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from world_understanding.utils.credentials import (
    ensure_no_inline_secrets,
    parse_env_reference,
)
from world_understanding.validation import ValidationRequest

from .child_launch import (
    VALIDATION_CHILD_LAUNCH_PROFILE,
    ChildLaunchArtifactIdentity,
    build_child_launch_descriptor,
    child_launch_artifact_identity,
)
from .runner import (
    CHILD_CACHE_RELPATH,
    CLAUDE_CLI_TRUSTED_SKILLS_PLUGIN_MANIFEST,
    CLAUDE_EXECUTION_CLI,
    CLAUDE_EXECUTION_SDK,
    CODEX_SANDBOX_WORKSPACE_WRITE,
    RUNNER_CLAUDE,
    RUNNER_CODEX,
    run_child_agent,
)

VALIDATION_COORDINATOR_WORKFLOW_SKILL = "content-workflow-validation"
CLAUDE_CLI_PROJECT_SCRATCH = Path(".claude") / ".cc-writes"


@dataclass(frozen=True)
class ValidationCoordinatorRunConfig:
    """One public agentic Validation invocation."""

    repo_root: Path
    request: ValidationRequest
    output_dir: Path
    config_base_dir: Path
    runner: str
    model: str
    model_reasoning_effort: str | None = None
    codex_base_url: str | None = None
    codex_sandbox_mode: str = CODEX_SANDBOX_WORKSPACE_WRITE
    codex_config: dict[str, object] | None = None
    claude_config: dict[str, object] | None = None
    claude_permission_mode: str = "default"
    claude_max_turns: int | None = None
    claude_execution_mode: str | None = None
    child_timeout_seconds: float = 1800.0
    agent_cwd: Path | None = None
    dry_run: bool = False


@dataclass(frozen=True)
class _ValidationChildConfig:
    child_launch_profile: ClassVar[str] = VALIDATION_CHILD_LAUNCH_PROFILE
    repo_root: Path
    usd_path: Path
    reference_images: list[Path]
    reference_files: list[Path] | None
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
    workflow_skill: str = VALIDATION_COORDINATOR_WORKFLOW_SKILL
    scene_backend: str = "none"
    child_capability_inventory: ChildLaunchArtifactIdentity | None = None
    child_domain_policy_bounds: ChildLaunchArtifactIdentity | None = None
    child_forbidden_environment_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class ValidationCoordinatorLauncherResult:
    status: Literal["prepared", "assessment_required"]
    output_dir: Path
    preparation_path: Path
    plan_patch_path: Path
    execution_receipt_path: Path | None = None
    run: ValidationWorkflowRun | None = None
    execution_receipt: ValidationCoordinatorExecutionReceipt | None = None


class _ValidationPlannerFinal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["plan_authored"]
    plan_id: str = Field(min_length=1)
    plan_patch_path: str = Field(min_length=1)
    plan_patch: ValidationCoordinatorPlanPatch


def _ensure_validation_agent_configs_safe(
    config: ValidationCoordinatorRunConfig,
) -> None:
    """Reject inline credentials before request or launcher persistence."""

    ensure_no_inline_secrets(
        {
            "codex_config": config.codex_config,
            "claude_config": config.claude_config,
        },
        context="Validation agent configuration",
        path_context=True,
    )


def _validation_forbidden_environment_names(
    request: ValidationRequest,
) -> tuple[str, ...]:
    """Collect only named Validation leaf credentials, never their values."""

    names: set[str] = set()

    def visit(value: object, key: str | None = None) -> None:
        if isinstance(value, Mapping):
            for child_key, child_value in value.items():
                visit(child_value, str(child_key))
            return
        if isinstance(value, Sequence) and not isinstance(
            value, str | bytes | bytearray
        ):
            for child in value:
                visit(child, key)
            return
        if key != "api_key_env":
            return
        name = parse_env_reference(value, allow_legacy_bare=True)
        if name is not None:
            names.add(name)

    for policy_key in ("look_right_vlm", "look_right_llm_judge"):
        visit(request.policy.get(policy_key))
    return tuple(sorted(names))


def _validated_claude_cli_reference_artifacts(
    *,
    config: ValidationCoordinatorRunConfig,
    raw_dir: Path,
    reference_images: list[Path],
) -> set[Path]:
    """Bind the shared launcher's confined Claude CLI image staging."""

    if not (
        config.runner == RUNNER_CLAUDE
        and config.claude_execution_mode == CLAUDE_EXECUTION_CLI
    ):
        return set()
    staged_root = raw_dir / "validation_planner_reference_images"
    if staged_root.is_symlink() or not staged_root.is_dir():
        raise RuntimeError(
            "Validation Claude CLI reference staging is missing or unsafe"
        )
    expected: list[Path] = []
    for index, source in enumerate(reference_images):
        suffix = source.expanduser().resolve().suffix.lower()
        if not re.fullmatch(r"\.[a-z0-9]{1,10}", suffix):
            suffix = ""
        destination = staged_root / f"reference-{index:03d}{suffix}"
        if (
            destination.is_symlink()
            or not destination.is_file()
            or file_sha256(destination) != file_sha256(source)
        ):
            raise RuntimeError(
                "Validation Claude CLI reference staging changed exact image bytes"
            )
        expected.append(destination)
    if set(staged_root.iterdir()) != set(expected):
        raise RuntimeError(
            "Validation Claude CLI reference staging contains unexpected artifacts"
        )
    return {staged_root, *expected}


def _planner_prompt(
    *,
    preparation_path: Path,
    preparation_payload: Mapping[str, object],
    plan_patch_path: Path,
    task_description: str,
    producer: Literal["codex", "claude"],
    child_session_id: str,
) -> str:
    schema = json.dumps(
        ValidationCoordinatorPlanPatch.model_json_schema(),
        indent=2,
        sort_keys=True,
    )
    preparation_json = json.dumps(preparation_payload, indent=2, sort_keys=True)
    return f"""You are the decision-only planning child for Validation Agent v0.6.

Use exactly this parent-supplied immutable preparation from:
{preparation_path}

<validation_coordinator_preparation>
{preparation_json}
</validation_coordinator_preparation>

Its exact preparation-bound task_description is:
{json.dumps(task_description)}

Copy every plan_id and check_id explicitly named by that task_description
exactly; do not invent aliases or substitute other IDs.

Return exactly one proposal in the structured final response's `plan_patch`
field. The trusted parent will validate and atomically persist it to:
{plan_patch_path}

The proposal must validate against this JSON schema:
{schema}

Use producer={producer!r} and child_session_id={child_session_id!r}. Select a
finite set of unique approved capabilities that directly supports explicit
claims and acceptance criteria in the task. Preserve exact capability,
template, rule, target, parameter, evidence, and dependency identities. Every
required check uses all_required_evidence; every advisory check uses
best_effort_advisory. Select at least two distinct capabilities named by
mandatory_constraints.provider_free_capability_ids and include at least one
explicit dependency. look_right is advisory and explicitly depends on
render_valid.

This turn is decision-only. Do not execute a Validation operation, renderer,
VLM/LLM leaf, simulator, nested agent, or arbitrary code. Do not create
validation_plan.json, validation_request.json, operations/, results, evidence,
assessment, review, or publication artifacts. Do not modify the source or the
preparation. Do not author Python or another executable file. The trusted outer
process will reject the proposal before any operation if any identity, target,
parameter, dependency, policy, or source binding is invalid.

Do not use a shell, file-write tool, or another side effect to create the plan
patch. Finish by returning only one JSON object matching the requested
structured-output schema, with status="plan_authored", the exact plan_id,
plan_patch_path={json.dumps(str(plan_patch_path))}, and the complete proposal in
plan_patch. Do not wrap the object in Markdown.
"""


def _child_final_schema() -> dict[str, object]:
    schema = _ValidationPlannerFinal.model_json_schema()
    definitions = schema.get("$defs")
    if not isinstance(definitions, dict):  # pragma: no cover - Pydantic contract
        raise RuntimeError("Validation planner schema omitted its definitions")
    selected_check = definitions.get("ValidationCoordinatorSelectedCheck")
    if not isinstance(selected_check, dict):  # pragma: no cover - Pydantic contract
        raise RuntimeError("Validation planner schema omitted selected checks")
    properties = selected_check.get("properties")
    if not isinstance(properties, dict):  # pragma: no cover - Pydantic contract
        raise RuntimeError("Validation planner selected-check schema is malformed")

    # OpenAI strict structured output does not admit an arbitrary-key object.
    # Validation v1 has exactly one child-authored parameter, so preserve the
    # domain model while exposing its complete bounded transport shape.
    properties["parameters"] = {
        "anyOf": [
            {
                "additionalProperties": False,
                "properties": {},
                "required": [],
                "type": "object",
            },
            {
                "additionalProperties": False,
                "properties": {
                    "evidence_paths": {
                        "items": {"type": "string"},
                        "type": "array",
                    }
                },
                "required": ["evidence_paths"],
                "type": "object",
            },
        ],
        "title": "Parameters",
    }
    definitions.pop("JsonValue", None)
    strict_schema = _strict_structured_output_schema(schema)
    if not isinstance(strict_schema, dict):  # pragma: no cover - root contract
        raise RuntimeError("Validation planner schema root is not an object")
    return strict_schema


def _strict_structured_output_schema(value: object) -> object:
    """Return the OpenAI/Claude-compatible strict subset of one JSON schema."""

    if isinstance(value, list):
        return [_strict_structured_output_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {
        key: _strict_structured_output_schema(child)
        for key, child in value.items()
        if key != "default"
    }
    properties = result.get("properties")
    if result.get("type") == "object" and isinstance(properties, dict):
        result["additionalProperties"] = False
        result["required"] = list(properties)
    return result


def _tree_snapshot(root: Path) -> tuple[tuple[str, str, int, str], ...]:
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(
            f"Validation decision-only support directory is missing or unsafe: {root}"
        )
    snapshot: list[tuple[str, str, int, str]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).parts):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise RuntimeError(
                "Validation planning child crossed the decision-only support "
                f"boundary with a symlink: {path}"
            )
        mode = stat.S_IMODE(path.stat().st_mode)
        if path.is_dir():
            snapshot.append((relative, "directory", mode, ""))
        elif path.is_file():
            snapshot.append((relative, "file", mode, file_sha256(path)))
        else:
            raise RuntimeError(
                "Validation planning child crossed the decision-only support "
                f"boundary with a special file: {path}"
            )
    return tuple(snapshot)


def _validate_decision_only_support_files(
    *,
    output_dir: Path,
    repo_root: Path,
    prompt_path: Path,
    prompt_sha256: str,
    prompt_mode: int,
) -> None:
    if (
        prompt_path.is_symlink()
        or not prompt_path.is_file()
        or file_sha256(prompt_path) != prompt_sha256
        or stat.S_IMODE(prompt_path.stat().st_mode) != prompt_mode
    ):
        raise RuntimeError("Validation planning child changed its frozen prompt")
    prompts_dir = prompt_path.parent
    unexpected_prompts = tuple(
        sorted(
            str(path.relative_to(output_dir))
            for path in prompts_dir.rglob("*")
            if path != prompt_path
        )
    )
    if unexpected_prompts:
        raise RuntimeError(
            "Validation planning child crossed the decision-only prompt boundary: "
            + ", ".join(unexpected_prompts)
        )

    skill_name = VALIDATION_COORDINATOR_WORKFLOW_SKILL
    source_skill = next(
        (
            candidate
            for candidate in (
                repo_root.resolve() / "agentic" / ".agents" / "skills" / skill_name,
                repo_root.resolve() / ".agents" / "skills" / skill_name,
            )
            if candidate.is_dir()
        ),
        None,
    )
    if source_skill is None:
        raise RuntimeError("Trusted Validation child skill source is unavailable")
    source_snapshot = _tree_snapshot(source_skill)
    for discovery_root_name in (".agents", ".claude"):
        discovery_root = output_dir / discovery_root_name
        skills_root = discovery_root / "skills"
        target_skill = skills_root / skill_name
        expected_root_entries = {"skills"}
        if discovery_root_name == ".claude":
            expected_root_entries.add(".claude-plugin")
        if (
            discovery_root.is_symlink()
            or skills_root.is_symlink()
            or target_skill.is_symlink()
            or not target_skill.is_dir()
            or {path.name for path in discovery_root.iterdir()} != expected_root_entries
            or {path.name for path in skills_root.iterdir()} != {skill_name}
            or _tree_snapshot(target_skill) != source_snapshot
        ):
            raise RuntimeError(
                "Validation planning child changed its trusted staged skill tree: "
                f"{discovery_root_name}"
            )
    plugin_metadata_root = output_dir / ".claude" / ".claude-plugin"
    plugin_manifest_path = plugin_metadata_root / "plugin.json"
    if (
        plugin_metadata_root.is_symlink()
        or not plugin_metadata_root.is_dir()
        or {path.name for path in plugin_metadata_root.iterdir()} != {"plugin.json"}
        or plugin_manifest_path.is_symlink()
        or not plugin_manifest_path.is_file()
        or load_json(plugin_manifest_path) != CLAUDE_CLI_TRUSTED_SKILLS_PLUGIN_MANIFEST
    ):
        raise RuntimeError(
            "Validation planning child changed its trusted staged skill tree: "
            ".claude-plugin"
        )


def _discard_empty_claude_cli_project_scratch(output_dir: Path) -> None:
    """Remove Claude CLI's empty sandbox bookkeeping directory.

    Claude CLI creates ``.claude/.cc-writes`` while enforcing its write
    sandbox, including on a tool-free turn. Only that exact empty directory is
    provider-owned scratch; links, files, special entries, or content remain a
    hard failure at the trusted-skill boundary.
    """

    current = output_dir
    for component in CLAUDE_CLI_PROJECT_SCRATCH.parts:
        current /= component
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            return
        except OSError as exc:
            raise RuntimeError(
                "Validation planning child changed its trusted staged skill tree: "
                + str(CLAUDE_CLI_PROJECT_SCRATCH)
            ) from exc
        if not stat.S_ISDIR(metadata.st_mode) or getattr(
            metadata, "st_file_attributes", 0
        ) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0):
            raise RuntimeError(
                "Validation planning child changed its trusted staged skill tree: "
                + str(CLAUDE_CLI_PROJECT_SCRATCH)
            )
    scratch = current
    if any(scratch.iterdir()):
        raise RuntimeError(
            "Validation planning child changed its trusted staged skill tree: "
            + str(CLAUDE_CLI_PROJECT_SCRATCH)
        )
    scratch.rmdir()


def _validate_rebuildable_child_cache_root(output_dir: Path) -> None:
    """Accept only the package-owned directory used for rebuildable child caches.

    The generic child launcher pins ``XDG_CACHE_HOME`` inside the writable run
    directory and never consumes that tree as workflow evidence. Cache contents
    therefore remain opaque and untrusted, but the top-level entry itself must
    still be a real directory rather than a link, file, or special entry.
    """

    cache_root = output_dir / CHILD_CACHE_RELPATH
    try:
        metadata = cache_root.lstat()
    except FileNotFoundError:
        return
    if not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError(
            "Validation planning child changed its rebuildable cache root: "
            + str(CHILD_CACHE_RELPATH)
        )


def run_validation_coordinator(
    config: ValidationCoordinatorRunConfig,
) -> ValidationCoordinatorLauncherResult:
    """Prepare, ask one child for a plan, accept it, then run exact adapters."""

    output_dir = config.output_dir.expanduser().resolve()
    if config.runner not in {RUNNER_CODEX, RUNNER_CLAUDE}:
        raise ValueError(f"Unsupported Validation child runner: {config.runner!r}")
    if not isinstance(config.model, str) or not config.model.strip():
        raise ValueError("Validation requires an explicit non-empty child model")
    if config.runner == RUNNER_CLAUDE and config.claude_execution_mode is None:
        raise ValueError(
            "Validation requires an explicit Claude execution mode: sdk or cli"
        )
    if config.runner == RUNNER_CODEX and config.claude_execution_mode is not None:
        raise ValueError(
            "Validation Claude execution mode may only be selected with runner=claude"
        )
    if config.claude_execution_mode not in {
        None,
        CLAUDE_EXECUTION_SDK,
        CLAUDE_EXECUTION_CLI,
    }:
        raise ValueError(
            "Unsupported Validation Claude execution mode: "
            f"{config.claude_execution_mode!r}"
        )
    if config.claude_max_turns is not None and config.claude_max_turns <= 0:
        raise ValueError("--claude-max-turns must be greater than 0.")
    if (
        config.claude_execution_mode == CLAUDE_EXECUTION_CLI
        and config.claude_max_turns is not None
    ):
        raise ValueError(
            "--claude-max-turns is not supported with "
            "--claude-execution-mode=cli; the claude CLI print mode has no "
            "max-turns equivalent."
        )
    _ensure_validation_agent_configs_safe(config)
    preparation = prepare_validation_coordinator(
        config.request,
        output_dir=output_dir,
        config_base_dir=config.config_base_dir,
    )
    preparation_path = output_dir / VALIDATION_COORDINATOR_PREPARATION_NAME
    plan_patch_path = output_dir / VALIDATION_COORDINATOR_PLAN_PATCH_NAME
    prompts_dir = output_dir / "prompts"
    raw_dir = output_dir / "raw"
    for directory in (prompts_dir, raw_dir):
        if directory.exists() or directory.is_symlink():
            raise RuntimeError(
                "Validation coordinator run directory already contains "
                f"{directory.name}/; use a fresh --output-dir: {output_dir}"
            )
    prompts_dir.mkdir()
    raw_dir.mkdir()
    producer: Literal["codex", "claude"] = (
        "codex" if config.runner == RUNNER_CODEX else "claude"
    )
    child_session_id = secrets.token_hex(16)
    prompt = _planner_prompt(
        preparation_path=preparation_path,
        preparation_payload=preparation.model_dump(mode="json"),
        plan_patch_path=plan_patch_path,
        task_description=preparation.request.task_description,
        producer=producer,
        child_session_id=child_session_id,
    )
    prompt_path = prompts_dir / "validation_coordinator_plan.md"
    atomic_write_text(prompt_path, prompt)
    if config.dry_run:
        return ValidationCoordinatorLauncherResult(
            status="prepared",
            output_dir=output_dir,
            preparation_path=preparation_path,
            plan_patch_path=plan_patch_path,
        )

    preparation_sha256 = file_sha256(preparation_path)
    prompt_sha256 = file_sha256(prompt_path)
    prompt_mode = stat.S_IMODE(prompt_path.stat().st_mode)
    preparation_identity = child_launch_artifact_identity(
        output_dir,
        preparation_path,
    )
    source = Path(preparation.request.inputs[0])
    reference_images = [
        Path(artifact.path)
        for artifact in preparation.adapter_identity.reference_artifacts
        if artifact.kind == "file"
    ]
    child_config = _ValidationChildConfig(
        repo_root=config.repo_root,
        usd_path=source,
        reference_images=reference_images,
        reference_files=None,
        runner=config.runner,
        model=config.model,
        model_reasoning_effort=config.model_reasoning_effort,
        codex_base_url=config.codex_base_url,
        codex_sandbox_mode=config.codex_sandbox_mode,
        codex_config=config.codex_config,
        claude_config=config.claude_config,
        claude_permission_mode=config.claude_permission_mode,
        claude_max_turns=config.claude_max_turns,
        # The generic child interface requires a concrete value. It is ignored
        # for Codex; Claude was required to select one explicitly above.
        claude_execution_mode=(config.claude_execution_mode or CLAUDE_EXECUTION_SDK),
        child_timeout_seconds=config.child_timeout_seconds,
        agent_cwd=config.agent_cwd,
        child_capability_inventory=preparation_identity,
        child_domain_policy_bounds=preparation_identity,
        child_forbidden_environment_names=(
            _validation_forbidden_environment_names(config.request)
        ),
    )
    child_output_path = raw_dir / "validation_planner_output.jsonl"
    child_final_path = raw_dir / "validation_planner_final.json"
    launch_descriptor = build_child_launch_descriptor(
        config=child_config,
        run_dir=output_dir,
        child_output_path=child_output_path,
        child_final_path=child_final_path,
        bridge_artifact_prefix="validation_planner",
        runner=child_config.runner,
        model=child_config.model,
        model_reasoning_effort=child_config.model_reasoning_effort,
        claude_execution_mode=child_config.claude_execution_mode,
        extra_allowed_hosts=(),
    )
    expected_launch_descriptor_digest = canonical_json_digest(
        launch_descriptor.model_dump(mode="json")
    )
    returncode = run_child_agent(
        config=child_config,
        prompt=prompt,
        run_dir=output_dir,
        child_output_path=child_output_path,
        child_final_path=child_final_path,
        output_schema=_child_final_schema(),
        bridge_artifact_prefix="validation_planner",
        stage_skills=True,
        tools_disabled=True,
        extra_allowed_hosts=[],
        launch_descriptor=launch_descriptor,
    )
    if returncode != 0:
        raise RuntimeError(
            f"Validation planning child failed with exit code {returncode}"
        )
    if file_sha256(preparation_path) != preparation_sha256:
        raise RuntimeError("Validation planning child changed its preparation")
    if (
        config.runner == RUNNER_CLAUDE
        and config.claude_execution_mode == CLAUDE_EXECUTION_CLI
    ):
        _discard_empty_claude_cli_project_scratch(output_dir)
    _validate_rebuildable_child_cache_root(output_dir)
    _validate_decision_only_support_files(
        output_dir=output_dir,
        repo_root=config.repo_root,
        prompt_path=prompt_path,
        prompt_sha256=prompt_sha256,
        prompt_mode=prompt_mode,
    )
    try:
        child_final = _ValidationPlannerFinal.model_validate(
            load_json(child_final_path)
        )
    except (OSError, ValueError, ValidationError) as exc:
        raise RuntimeError(
            f"Validation planning child returned an invalid final response: {exc}"
        ) from exc
    if Path(child_final.plan_patch_path).expanduser().resolve() != plan_patch_path:
        raise RuntimeError("Validation planning child reported another plan patch path")
    if child_final.plan_patch.plan_id != child_final.plan_id:
        raise RuntimeError(
            "Validation planning child final response changed its plan ID"
        )
    allowed_top_level = {
        VALIDATION_COORDINATOR_PREPARATION_NAME,
        CHILD_CACHE_RELPATH.name,
        ".agents",
        ".claude",
        "prompts",
        "raw",
    }
    unexpected = tuple(
        sorted(
            path.name
            for path in output_dir.iterdir()
            if path.name not in allowed_top_level
        )
    )
    if unexpected:
        raise RuntimeError(
            "Validation planning child crossed the decision-only boundary: "
            + ", ".join(unexpected)
        )
    allowed_raw = {
        child_output_path,
        child_final_path,
        raw_dir / "validation_planner_request.json",
        raw_dir / "validation_planner_items.json",
        raw_dir / "validation_planner_result.json",
        raw_dir / "validation_planner_observable_events.jsonl",
        raw_dir / "validation_planner_launch_descriptor.json",
        *_validated_claude_cli_reference_artifacts(
            config=config,
            raw_dir=raw_dir,
            reference_images=reference_images,
        ),
    }
    unexpected_raw = tuple(
        sorted(
            str(path.relative_to(output_dir))
            for path in raw_dir.rglob("*")
            if path not in allowed_raw
        )
    )
    if unexpected_raw:
        raise RuntimeError(
            "Validation planning child crossed the decision-only raw boundary: "
            + ", ".join(unexpected_raw)
        )
    atomic_write_json(
        plan_patch_path,
        child_final.plan_patch,
        within=output_dir,
    )
    accepted = accept_validation_coordinator_plan(
        output_dir,
        plan_patch_path=plan_patch_path,
        child_output_path=child_final_path,
        producer=producer,
        expected_child_launch_descriptor_digest=(expected_launch_descriptor_digest),
        child_session_id=child_session_id,
        child_plan_id=child_final.plan_id,
    )
    if accepted.plan_id != child_final.plan_id:
        raise RuntimeError(
            "Validation planning child final response changed its plan ID"
        )
    run, execution_receipt = execute_validation_coordinator_plan(output_dir)
    return ValidationCoordinatorLauncherResult(
        status="assessment_required",
        output_dir=output_dir,
        preparation_path=preparation_path,
        plan_patch_path=plan_patch_path,
        execution_receipt_path=(
            output_dir / VALIDATION_COORDINATOR_EXECUTION_RECEIPT_NAME
        ),
        run=run,
        execution_receipt=execution_receipt,
    )


__all__ = [
    "VALIDATION_COORDINATOR_WORKFLOW_SKILL",
    "ValidationCoordinatorLauncherResult",
    "ValidationCoordinatorRunConfig",
    "run_validation_coordinator",
]
