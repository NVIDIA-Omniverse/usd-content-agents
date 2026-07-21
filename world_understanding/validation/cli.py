# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validation Agent CLI helpers.

This module keeps command parsing in ``world_understanding.cli`` thin while
providing a testable local/CI entry point for Validation Agent V1 configs.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import typer
import yaml
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from world_understanding.agentic.validation_scaffold import (
    DraftValidationError,
    create_draft_validation_request,
    run_validation_scaffold,
)
from world_understanding.validation.models import (
    ISSUE_CODE_PATTERN,
    ValidationFocusConfig,
    ValidationIssue,
    ValidationProject,
    ValidationRenderConfig,
    ValidationRequest,
    ValidationResult,
    ValidationTemplateResult,
    ValidationVerdict,
    aggregate_validation_verdict,
)
from world_understanding.validation.rendering_backend_contract import (
    normalize_validation_rendering_backend,
)
from world_understanding.validation.scaffold_compat import (
    validation_result_from_scaffold_result,
)
from world_understanding.validation.templates import (
    ValidationContractError,
    create_default_template_registry,
)
from world_understanding.validation.usd_rendering import expand_runtime_render_views

DEFAULT_OUTPUT_DIR: Final = Path(".validation-runs") / "validation-agent"
REQUEST_ARTIFACT_NAME: Final = "validation_request.json"
PLAN_ARTIFACT_NAME: Final = "validation_plan.json"
RESULT_ARTIFACT_NAME: Final = "validation_result.json"

PASS_EXIT_CODE: Final = 0
VALIDATION_FAILURE_EXIT_CODE: Final = 1
VALIDATION_CLI_ERROR_EXIT_CODE: Final = 2
DEFAULT_DEPENDENCY_UNAVAILABLE_ISSUE_CODES: Final = frozenset(
    {
        "render.renderer_unavailable",
        "visual.judge_unavailable",
        "physics.behavior_judge_unavailable",
        "physics.behavior_refiner_unavailable",
    }
)
DEPENDENCY_UNAVAILABLE_GATE_ISSUE_CODE: Final = "validation.dependency_unavailable"
DEPENDENCY_UNAVAILABLE_RECOMMENDED_ACTION: Final = (
    "Configure the required validation dependency and rerun the release gate. "
    "For visual gates, provide a usable look_right VLM model/credential path; "
    "for render gates, provide an available renderer endpoint; for behavior "
    "gates, provide the required judge evidence or model path."
)


class ValidationCliError(RuntimeError):
    """Raised when a Validation Agent CLI run cannot be prepared or executed."""


@dataclass(frozen=True)
class ValidationCliRun:
    """Result bundle returned by the testable CLI helper."""

    request: ValidationRequest
    result: ValidationResult
    output_dir: Path
    artifact_paths: dict[str, str]
    exit_code: int


def load_validation_request_config(config_path: str | Path) -> ValidationRequest:
    """Load a JSON or YAML Validation Agent V1 request config."""

    path = _resolve_existing_file(config_path)
    try:
        raw_config = _load_config_mapping(path)
        return ValidationRequest.model_validate(raw_config)
    except ValidationError as exc:
        raise ValidationCliError(f"Invalid validation config {path}: {exc}") from exc


def run_validation_from_config(
    config_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    dry_run: bool = False,
    template_overrides: Sequence[str] = (),
    focus_prim_overrides: Sequence[str] = (),
    fail_on_warn: bool = False,
) -> ValidationCliRun:
    """Run Validation Agent from a V1 config file and write report artifacts."""

    config_file = _resolve_existing_file(config_path)
    request = load_validation_request_config(config_file)
    resolved_output_dir = _resolve_output_dir(
        request,
        config_file=config_file,
        output_dir=output_dir,
    )
    validation_request = _apply_validation_overrides(
        request,
        output_dir=resolved_output_dir,
        template_overrides=template_overrides,
        focus_prim_overrides=focus_prim_overrides,
    )
    return run_validation_request(
        validation_request,
        config_base_dir=config_file.parent,
        output_dir=resolved_output_dir,
        dry_run=dry_run,
        fail_on_warn=fail_on_warn,
    )


def build_validation_request_from_inputs(
    *,
    task_description: str,
    inputs: str | Path | Sequence[str | Path],
    output_dir: str | Path | None = None,
    template_overrides: Sequence[str] = (),
    focus_prim_overrides: Sequence[str] = (),
    reference_image_paths: str | Path | Sequence[str | Path] = (),
    render_backend: str | None = None,
    render_views: Sequence[str] = (),
    render_image_width: int | None = None,
    render_image_height: int | None = None,
    base_dir: str | Path | None = None,
) -> ValidationRequest:
    """Build a V1 request for ``validation-agent validate --task ... INPUT...``."""

    resolved_base_dir = _resolve_base_dir(base_dir)
    task = task_description.strip()
    if not task:
        raise ValidationCliError("Validation task must not be empty")
    input_paths = _normalize_direct_input_paths(inputs)
    reference_paths = _normalize_optional_direct_paths(reference_image_paths)

    resolved_output_dir = _resolve_direct_output_dir(
        output_dir,
        base_dir=resolved_base_dir,
    )
    policy = {"reference_image_paths": reference_paths} if reference_paths else {}
    try:
        return ValidationRequest(
            task_description=task,
            inputs=input_paths,
            project=ValidationProject(working_dir=str(resolved_output_dir)),
            render=ValidationRenderConfig(
                backend=render_backend,
                views=tuple(render_views) if render_views else None,
                image_width=render_image_width,
                image_height=render_image_height,
            ),
            focus=ValidationFocusConfig(prim_paths=tuple(focus_prim_overrides)),
            requested_templates=tuple(template_overrides),
            policy=policy,
        )
    except ValidationError as exc:
        raise ValidationCliError(f"Invalid validation request: {exc}") from exc


def run_validation_from_inputs(
    *,
    task_description: str,
    inputs: str | Path | Sequence[str | Path],
    output_dir: str | Path | None = None,
    dry_run: bool = False,
    template_overrides: Sequence[str] = (),
    focus_prim_overrides: Sequence[str] = (),
    fail_on_warn: bool = False,
    reference_image_paths: str | Path | Sequence[str | Path] = (),
    render_backend: str | None = None,
    render_views: Sequence[str] = (),
    render_image_width: int | None = None,
    render_image_height: int | None = None,
    base_dir: str | Path | None = None,
) -> ValidationCliRun:
    """Run Validation Agent from a task string and direct input paths."""

    resolved_base_dir = _resolve_base_dir(base_dir)
    request = build_validation_request_from_inputs(
        task_description=task_description,
        inputs=inputs,
        output_dir=output_dir,
        template_overrides=template_overrides,
        focus_prim_overrides=focus_prim_overrides,
        reference_image_paths=reference_image_paths,
        render_backend=render_backend,
        render_views=render_views,
        render_image_width=render_image_width,
        render_image_height=render_image_height,
        base_dir=resolved_base_dir,
    )
    if request.project.working_dir is None:
        raise ValidationCliError("Validation output directory could not be resolved")
    return run_validation_request(
        request,
        config_base_dir=resolved_base_dir,
        output_dir=request.project.working_dir,
        dry_run=dry_run,
        fail_on_warn=fail_on_warn,
    )


def run_validation_cli_command(
    *,
    config_path: Path,
    output_dir: Path | None,
    dry_run: bool,
    template_overrides: Sequence[str] = (),
    focus_prim_overrides: Sequence[str] = (),
    fail_on_warn: bool,
    output_format: str,
    console: Console,
) -> None:
    """Run the shared Validation Agent CLI command implementation."""

    if output_format not in {"text", "json"}:
        console.print(f"[red]Error: Unknown format: {output_format}[/red]")
        raise typer.Exit(VALIDATION_CLI_ERROR_EXIT_CODE)

    try:
        cli_run = run_validation_from_config(
            config_path,
            output_dir=output_dir,
            dry_run=dry_run,
            template_overrides=tuple(template_overrides),
            focus_prim_overrides=tuple(focus_prim_overrides),
            fail_on_warn=fail_on_warn,
        )
    except ValidationCliError as e:
        console.print(f"[red]Validation Agent error:[/red] {e}")
        raise typer.Exit(VALIDATION_CLI_ERROR_EXIT_CODE) from e

    if output_format == "json":
        console.print_json(cli_run.result.model_dump_json(indent=2))
    else:
        print_validation_summary(cli_run, console=console)

    if cli_run.exit_code:
        raise typer.Exit(cli_run.exit_code)


def run_validation_inputs_cli_command(
    *,
    task_description: str,
    inputs: Sequence[Path],
    output_dir: Path | None,
    dry_run: bool,
    template_overrides: Sequence[str] = (),
    focus_prim_overrides: Sequence[str] = (),
    fail_on_warn: bool,
    output_format: str,
    console: Console,
    reference_image_paths: Sequence[Path] = (),
    render_backend: str | None = None,
    render_views: Sequence[str] = (),
    render_image_width: int | None = None,
    render_image_height: int | None = None,
) -> None:
    """Run the shared direct-input Validation Agent CLI implementation."""

    if output_format not in {"text", "json"}:
        console.print(f"[red]Error: Unknown format: {output_format}[/red]")
        raise typer.Exit(VALIDATION_CLI_ERROR_EXIT_CODE)

    try:
        cli_run = run_validation_from_inputs(
            task_description=task_description,
            inputs=inputs,
            output_dir=output_dir,
            dry_run=dry_run,
            template_overrides=tuple(template_overrides),
            focus_prim_overrides=tuple(focus_prim_overrides),
            fail_on_warn=fail_on_warn,
            reference_image_paths=tuple(reference_image_paths),
            render_backend=render_backend,
            render_views=tuple(render_views),
            render_image_width=render_image_width,
            render_image_height=render_image_height,
        )
    except ValidationCliError as e:
        console.print(f"[red]Validation Agent error:[/red] {e}")
        raise typer.Exit(VALIDATION_CLI_ERROR_EXIT_CODE) from e

    if output_format == "json":
        console.print_json(cli_run.result.model_dump_json(indent=2))
    else:
        print_validation_summary(cli_run, console=console)

    if cli_run.exit_code:
        raise typer.Exit(cli_run.exit_code)


def print_validation_summary(
    cli_run: ValidationCliRun,
    *,
    console: Console,
) -> None:
    """Print the concise Rich summary shared by Validation Agent CLIs."""

    verdict = cli_run.result.verdict
    color = {
        "pass": "green",
        "planned": "cyan",
        "warn": "yellow",
        "fail": "red",
        "needs_refinement": "red",
    }.get(verdict, "white")
    console.print(
        f"[bold]Validation Agent verdict:[/bold] [{color}]{verdict}[/{color}]"
    )
    console.print(f"[bold]Output directory:[/bold] {cli_run.output_dir}")

    table = Table(title="Validation Agent Templates")
    table.add_column("Template", style="cyan")
    table.add_column("Status", style="magenta")
    table.add_column("Issues", justify="right")
    for template_result in cli_run.result.template_results:
        table.add_row(
            template_result.template_name,
            template_result.status,
            str(len(template_result.issues)),
        )
    console.print(table)

    if cli_run.result.issues:
        issues_table = Table(title="Validation Agent Issues")
        issues_table.add_column("Severity", style="yellow")
        issues_table.add_column("Code", style="cyan")
        issues_table.add_column("Template", style="magenta")
        issues_table.add_column("Message", style="white")
        for issue in cli_run.result.issues:
            issues_table.add_row(
                issue.severity,
                issue.code,
                issue.template_name or "",
                issue.message,
            )
        console.print(issues_table)

    console.print("[bold]Artifacts:[/bold]")
    for name, path in cli_run.artifact_paths.items():
        console.print(f"  {name}: {path}")


def run_validation_request(
    request: ValidationRequest,
    *,
    config_base_dir: str | Path,
    output_dir: str | Path,
    dry_run: bool = False,
    fail_on_warn: bool = False,
) -> ValidationCliRun:
    """Run a prepared Validation Agent request and write stable V1 artifacts."""

    try:
        _validate_requested_templates(request.requested_templates)
        base_dir = Path(config_base_dir).expanduser().resolve(strict=False)
        resolved_output_dir = Path(output_dir).expanduser().resolve(strict=False)

        # Validate and normalize the policy before creating or replacing run
        # artifacts so selector conflicts cannot leave a mixed artifact set.
        scaffold_policy = _scaffold_policy_from_request(request, base_dir=base_dir)

        resolved_output_dir.mkdir(parents=True, exist_ok=True)

        request_artifact_path = resolved_output_dir / REQUEST_ARTIFACT_NAME
        plan_artifact_path = resolved_output_dir / PLAN_ARTIFACT_NAME
        result_artifact_path = resolved_output_dir / RESULT_ARTIFACT_NAME
        artifact_paths = {
            "validation_request": str(request_artifact_path),
            "validation_plan": str(plan_artifact_path),
            "validation_result": str(result_artifact_path),
        }

        _write_model_json(request, request_artifact_path)

        draft_request = create_draft_validation_request(
            task_description=request.task_description,
            inputs=request.inputs,
            working_dir=resolved_output_dir,
            base_dir=base_dir,
            focus_prim_paths=request.focus.prim_paths,
            requested_templates=request.requested_templates,
            policy=scaffold_policy,
            dry_run=dry_run,
            metadata=_scaffold_metadata_from_request(request),
        )
        try:
            draft_result = run_validation_scaffold(draft_request)
        except DraftValidationError as exc:
            raise ValidationCliError(str(exc)) from exc

        stable_result = validation_result_from_scaffold_result(draft_result)
        stable_result = _finalize_result(
            stable_result,
            request=request,
            artifact_paths=artifact_paths,
        )
        _write_model_json(stable_result.plan, plan_artifact_path)
        _write_model_json(stable_result, result_artifact_path)

        return ValidationCliRun(
            request=request,
            result=stable_result,
            output_dir=resolved_output_dir,
            artifact_paths=artifact_paths,
            exit_code=validation_exit_code(
                stable_result.verdict,
                fail_on_warn=fail_on_warn,
            ),
        )
    except ValidationCliError:
        raise
    except Exception as exc:
        raise ValidationCliError(f"Validation Agent run failed: {exc}") from exc


def validation_exit_code(
    verdict: ValidationVerdict,
    *,
    fail_on_warn: bool = False,
) -> int:
    """Return a CI-friendly process status code for a validation verdict."""

    if verdict in {"pass", "planned"}:
        return PASS_EXIT_CODE
    if verdict == "warn" and not fail_on_warn:
        return PASS_EXIT_CODE
    return VALIDATION_FAILURE_EXIT_CODE


def _load_config_mapping(path: Path) -> Mapping[str, Any]:
    suffix = path.suffix.lower()
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValidationCliError(
            f"Unable to read validation config {path}: {exc}"
        ) from exc
    try:
        if suffix == ".json":
            raw_config = json.loads(text)
        elif suffix in {".yaml", ".yml"}:
            raw_config = yaml.safe_load(text)
        else:
            raise ValidationCliError(
                "Validation config must be JSON or YAML: "
                f"{path} (extension {suffix or '<none>'})"
            )
    except json.JSONDecodeError as exc:
        raise ValidationCliError(
            f"Invalid JSON validation config {path}: {exc}"
        ) from exc
    except yaml.YAMLError as exc:
        raise ValidationCliError(
            f"Invalid YAML validation config {path}: {exc}"
        ) from exc
    if not isinstance(raw_config, Mapping):
        raise ValidationCliError(
            "Validation config must be a mapping: "
            f"{path} (got {type(raw_config).__name__})"
        )
    return raw_config


def _resolve_existing_file(path: str | Path | None) -> Path:
    if path is None:
        raise ValidationCliError("Missing config path")
    resolved = Path(path).expanduser().resolve(strict=False)
    if not resolved.is_file():
        raise ValidationCliError(f"Validation config not found: {resolved}")
    return resolved


def _resolve_base_dir(base_dir: str | Path | None) -> Path:
    if base_dir is None:
        return Path.cwd().resolve(strict=False)
    return Path(base_dir).expanduser().resolve(strict=False)


def _normalize_direct_input_paths(
    inputs: str | Path | Sequence[str | Path],
) -> tuple[str, ...]:
    input_paths = _normalize_optional_direct_paths(inputs)
    if not input_paths:
        raise ValidationCliError("At least one validation input is required")
    return input_paths


def _normalize_optional_direct_paths(
    paths: str | Path | Sequence[str | Path],
) -> tuple[str, ...]:
    if isinstance(paths, str | Path):
        return (str(paths),)
    return tuple(str(path) for path in paths)


def _resolve_output_dir(
    request: ValidationRequest,
    *,
    config_file: Path,
    output_dir: str | Path | None,
) -> Path:
    if output_dir is not None:
        path = Path(output_dir).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        return path.resolve(strict=False)

    if request.project.working_dir:
        path = Path(request.project.working_dir).expanduser()
        if not path.is_absolute():
            path = config_file.parent / path
        return path.resolve(strict=False)

    return (config_file.parent / DEFAULT_OUTPUT_DIR).resolve(strict=False)


def _resolve_direct_output_dir(
    output_dir: str | Path | None,
    *,
    base_dir: Path,
) -> Path:
    if output_dir is not None:
        path = Path(output_dir).expanduser()
        if not path.is_absolute():
            path = base_dir / path
        return path.resolve(strict=False)
    return (base_dir / DEFAULT_OUTPUT_DIR).resolve(strict=False)


def _apply_validation_overrides(
    request: ValidationRequest,
    *,
    output_dir: Path,
    template_overrides: Sequence[str],
    focus_prim_overrides: Sequence[str],
) -> ValidationRequest:
    requested_templates = (
        tuple(template_overrides) if template_overrides else request.requested_templates
    )
    focus = (
        ValidationFocusConfig(prim_paths=tuple(focus_prim_overrides))
        if focus_prim_overrides
        else request.focus
    )
    project = request.project.model_copy(update={"working_dir": str(output_dir)})
    return request.model_copy(
        update={
            "project": project,
            "focus": focus,
            "requested_templates": requested_templates,
        }
    )


def _validate_requested_templates(template_names: Sequence[str]) -> None:
    try:
        create_default_template_registry().validate_template_names(template_names)
    except ValidationContractError as exc:
        raise ValidationCliError(str(exc)) from exc


def _scaffold_policy_from_request(
    request: ValidationRequest,
    *,
    base_dir: str | Path | None = None,
) -> dict[str, Any]:
    policy = dict(request.policy)
    if base_dir is not None:
        policy = _resolve_policy_path_fields(policy, base_dir=Path(base_dir))
    if request.render.backend is not None:
        legacy_backend = policy.get("render_backend")
        if legacy_backend is not None and (
            normalize_validation_rendering_backend(legacy_backend)
            != normalize_validation_rendering_backend(request.render.backend)
        ):
            raise ValidationCliError(
                "Conflicting rendering backends: render.backend is "
                f"{request.render.backend!r}, but legacy policy.render_backend is "
                f"{legacy_backend!r}. Remove policy.render_backend or make the values "
                "match."
            )
        policy["render_backend"] = request.render.backend
    if request.render.views is not None:
        policy.setdefault(
            "expected_cameras",
            list(expand_runtime_render_views(request.render.views)),
        )
    if isinstance(request.render.animation_frames, str):
        policy.setdefault("expected_frames", [request.render.animation_frames])
    elif isinstance(request.render.animation_frames, tuple):
        policy.setdefault("expected_frames", list(request.render.animation_frames))
    if request.render.image_width is not None:
        policy.setdefault("render_image_width", request.render.image_width)
    if request.render.image_height is not None:
        policy.setdefault("render_image_height", request.render.image_height)
    if request.render.metadata:
        policy.setdefault("render_metadata", request.render.metadata)
    return policy


_POLICY_PATH_SEQUENCE_KEYS: Final = (
    "animation_frame_paths",
    "animation_usd_paths",
    "behavior_video_paths",
    "current_image_paths",
    "reference_image_paths",
    "render_image_paths",
    "sampled_video_frame_paths",
    "simulation_json_paths",
    "time_sampled_usd_paths",
    "trajectory_metrics_paths",
    "video_paths",
)
_POLICY_PATH_SCALAR_OR_SEQUENCE_KEYS: Final = (
    "physical_behavior_refine_summary_path",
    "refine_summary_path",
)
_POLICY_PATH_VALUE_KEYS: Final = (
    "physical_behavior_refine_output_dir",
    "physics_refine_output_dir",
    "refine_output_dir",
    "render_output_dir",
)


def _resolve_policy_path_fields(
    policy: Mapping[str, Any],
    *,
    base_dir: Path,
) -> dict[str, Any]:
    resolved = dict(policy)
    for key in _POLICY_PATH_SEQUENCE_KEYS:
        if key in resolved:
            resolved[key] = _resolve_path_sequence_value(
                resolved[key],
                base_dir,
                field_name=key,
            )
    for key in _POLICY_PATH_SCALAR_OR_SEQUENCE_KEYS:
        if key in resolved:
            resolved[key] = _resolve_path_scalar_or_sequence_value(
                resolved[key],
                base_dir,
                field_name=key,
            )
    for key in _POLICY_PATH_VALUE_KEYS:
        if key in resolved and resolved[key] is not None:
            if not isinstance(resolved[key], str | Path):
                raise ValidationCliError(
                    f"policy.{key} must be a path string, got "
                    f"{type(resolved[key]).__name__}"
                )
            resolved[key] = str(
                _resolve_policy_path_value(
                    resolved[key],
                    base_dir,
                    field_name=key,
                )
            )

    focused = resolved.get("focused_image_paths")
    if isinstance(focused, Mapping):
        focused_paths: dict[str, Any] = {}
        for prim_path, paths in focused.items():
            if not isinstance(prim_path, str):
                raise ValidationCliError(
                    "policy.focused_image_paths keys must be prim-path strings, "
                    f"got {type(prim_path).__name__}"
                )
            focused_paths[prim_path] = _resolve_path_sequence_value(
                paths,
                base_dir,
                field_name=f"focused_image_paths[{prim_path!r}]",
            )
        resolved["focused_image_paths"] = focused_paths
    elif focused is not None:
        raise ValidationCliError(
            "policy.focused_image_paths must be a mapping of prim-path strings "
            f"to path strings, got {type(focused).__name__}"
        )
    return resolved


def _resolve_path_sequence_value(
    value: Any,
    base_dir: Path,
    *,
    field_name: str,
) -> Any:
    if value is None:
        return value
    if isinstance(value, str | Path):
        return [str(_resolve_policy_path_value(value, base_dir, field_name=field_name))]
    if not isinstance(value, Sequence) or isinstance(value, bytes | bytearray):
        raise ValidationCliError(
            f"policy.{field_name} must be a path string or sequence of path "
            f"strings, got {type(value).__name__}"
        )
    resolved_values: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str | Path):
            raise ValidationCliError(
                f"policy.{field_name}[{index}] must be a path string, got "
                f"{type(item).__name__}"
            )
        resolved_values.append(
            str(
                _resolve_policy_path_value(
                    item,
                    base_dir,
                    field_name=f"{field_name}[{index}]",
                )
            )
        )
    return resolved_values


def _resolve_path_scalar_or_sequence_value(
    value: Any,
    base_dir: Path,
    *,
    field_name: str,
) -> Any:
    if value is None:
        return value
    if isinstance(value, str | Path):
        return str(_resolve_policy_path_value(value, base_dir, field_name=field_name))
    return _resolve_path_sequence_value(value, base_dir, field_name=field_name)


def _resolve_policy_path_value(
    value: str | Path,
    base_dir: Path,
    *,
    field_name: str,
) -> Path:
    if isinstance(value, str) and not value.strip():
        raise ValidationCliError(f"policy.{field_name} path must not be empty")
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve(strict=False)
    return (base_dir / path).resolve(strict=False)


def _scaffold_metadata_from_request(request: ValidationRequest) -> dict[str, Any]:
    metadata = dict(request.metadata)
    metadata.setdefault("schema_version", request.schema_version)
    metadata.setdefault("planner", request.planner.model_dump(mode="json"))
    metadata.setdefault("render", request.render.model_dump(mode="json"))
    metadata.setdefault("project", request.project.model_dump(mode="json"))
    return metadata


def _finalize_result(
    result: ValidationResult,
    *,
    request: ValidationRequest,
    artifact_paths: Mapping[str, str],
) -> ValidationResult:
    plan = result.plan.model_copy(update={"artifact_paths": dict(artifact_paths)})
    metadata = dict(result.metadata)
    metadata.setdefault("runner", "validation-agent-cli")
    stable_result = result.model_copy(
        update={
            "request": request,
            "plan": plan,
            "artifact_paths": dict(artifact_paths),
            "metadata": metadata,
        }
    )
    expected_result = _apply_expected_result_policy(stable_result)
    return _apply_gate_policy(expected_result, request=request)


def _apply_expected_result_policy(result: ValidationResult) -> ValidationResult:
    """Downgrade exactly-matched known-negative failures to reportable warnings.

    Callers opt in by setting ``policy.expected_verdict: fail`` and an exact
    ``policy.expected_issue_codes`` set. Missing, extra, errored, or internally
    inconsistent failure evidence stays blocking. Planned dry-run results are
    not evaluated because no template evidence has been produced yet.
    """

    expected_verdict = result.request.policy.get("expected_verdict")
    if expected_verdict != "fail" or result.verdict == "planned":
        return result

    expected_issue_codes = _expected_issue_codes(result.request.policy)
    observed_fail_codes = _observed_fail_issue_codes(result)
    match_metadata = {
        "configured": True,
        "expected_verdict": expected_verdict,
        "expected_issue_codes": sorted(expected_issue_codes),
        "observed_fail_issue_codes": sorted(observed_fail_codes),
        "matched": False,
    }
    if not expected_issue_codes:
        match_metadata["reason"] = "expected_issue_codes_missing"
        return _result_with_expected_result_metadata(result, match_metadata)

    unexpected_issue_codes = observed_fail_codes - expected_issue_codes
    missing_issue_codes = expected_issue_codes - observed_fail_codes
    if unexpected_issue_codes or missing_issue_codes:
        match_metadata.update(
            {
                "unexpected_issue_codes": sorted(unexpected_issue_codes),
                "missing_expected_issue_codes": sorted(missing_issue_codes),
                "reason": "issue_code_mismatch",
            }
        )
        return _result_with_expected_result_metadata(result, match_metadata)

    if result.verdict != "fail" or not observed_fail_codes:
        match_metadata["reason"] = "verdict_mismatch"
        return _result_with_expected_result_metadata(result, match_metadata)

    # Guard status/issue desync: a failed template without one of the expected
    # fail issues remains blocking even if the aggregate issue-code set matches.
    if _has_unexpected_failed_template(result, expected_issue_codes):
        match_metadata["reason"] = "unexpected_failed_template"
        return _result_with_expected_result_metadata(result, match_metadata)

    if any(template.status == "error" for template in result.template_results):
        match_metadata["reason"] = "template_error"
        return _result_with_expected_result_metadata(result, match_metadata)

    match_metadata["matched"] = True
    issues = tuple(
        _downgrade_expected_failure_issue(issue, expected_issue_codes)
        for issue in result.issues
    )
    template_results = tuple(
        _downgrade_expected_failure_template_result(template, expected_issue_codes)
        for template in result.template_results
    )
    metadata = dict(result.metadata)
    metadata["expected_result"] = match_metadata
    return result.model_copy(
        update={
            "verdict": aggregate_validation_verdict(template_results),
            "template_results": template_results,
            "issues": issues,
            "metadata": metadata,
            "recommended_action": (
                result.recommended_action
                or "Known-negative fixture matched the expected failure. "
                "Treat this as reportable coverage, not proof that the "
                "asset has passing generated physics authoring."
            ),
        }
    )


def _result_with_expected_result_metadata(
    result: ValidationResult,
    expected_result: Mapping[str, Any],
) -> ValidationResult:
    metadata = dict(result.metadata)
    metadata["expected_result"] = dict(expected_result)
    issues = tuple(result.issues)
    if expected_result.get("matched") is False:
        issues = (
            *issues,
            _expected_result_mismatch_issue(expected_result),
        )
    return result.model_copy(
        update={
            "verdict": "fail",
            "issues": issues,
            "metadata": metadata,
        }
    )


def _expected_result_mismatch_issue(
    expected_result: Mapping[str, Any],
) -> ValidationIssue:
    reason = expected_result.get("reason", "unknown")
    return ValidationIssue(
        code="validation.expected_result_mismatch",
        severity="fail",
        message=(
            "Known-negative expected result did not match observed validation "
            f"result ({reason})."
        ),
        details={"expected_result": dict(expected_result)},
    )


def _expected_issue_codes(policy: Mapping[str, Any]) -> set[str]:
    value = policy.get("expected_issue_codes")
    if isinstance(value, str):
        return {value} if value.strip() else set()
    if isinstance(value, bytes | bytearray) or not isinstance(value, Sequence):
        return set()
    return {str(item) for item in value if isinstance(item, str) and item.strip()}


def _observed_fail_issue_codes(result: ValidationResult) -> set[str]:
    """Return aggregate fail issue codes after scaffold result normalization."""

    return {issue.code for issue in result.issues if issue.severity == "fail"}


def _has_unexpected_failed_template(
    result: ValidationResult,
    expected_issue_codes: set[str],
) -> bool:
    for template in result.template_results:
        if template.status != "failed":
            continue
        fail_issue_codes = {
            issue.code for issue in template.issues if issue.severity == "fail"
        }
        if not fail_issue_codes or fail_issue_codes - expected_issue_codes:
            return True
    return False


def _downgrade_expected_failure_template_result(
    template: ValidationTemplateResult,
    expected_issue_codes: set[str],
) -> ValidationTemplateResult:
    issues = tuple(
        _downgrade_expected_failure_issue(issue, expected_issue_codes)
        for issue in template.issues
    )
    has_expected_failure_issue = any(
        issue.details.get("expected_failure") for issue in issues
    )
    status = (
        "warn"
        if template.status == "failed" and has_expected_failure_issue
        else template.status
    )
    metadata = dict(template.metadata)
    if has_expected_failure_issue:
        metadata["expected_failure_matched"] = True
    return template.model_copy(
        update={
            "status": status,
            "issues": issues,
            "metadata": metadata,
        }
    )


def _downgrade_expected_failure_issue(
    issue: ValidationIssue,
    expected_issue_codes: set[str],
) -> ValidationIssue:
    if issue.severity != "fail" or issue.code not in expected_issue_codes:
        return issue

    details = dict(issue.details)
    details.setdefault("original_severity", issue.severity)
    details["expected_failure"] = True
    return issue.model_copy(
        update={
            "severity": "warn",
            "details": details,
        }
    )


def _apply_gate_policy(
    result: ValidationResult,
    *,
    request: ValidationRequest,
) -> ValidationResult:
    gate_policy = _optional_mapping(request.policy.get("gate_policy"))
    if not _gate_policy_blocks(gate_policy.get("dependency_unavailable")):
        return result

    blocked_issue_codes = _dependency_unavailable_issue_codes(gate_policy)
    blocked_issues = tuple(
        issue for issue in result.issues if issue.code in blocked_issue_codes
    )
    if not blocked_issues:
        return result

    issue_codes = sorted({issue.code for issue in blocked_issues})
    gate_issue = ValidationIssue(
        code=DEPENDENCY_UNAVAILABLE_GATE_ISSUE_CODE,
        severity="fail",
        message=(
            "A validation dependency required by this release gate is unavailable."
        ),
        details={
            "gate_policy": "dependency_unavailable:block",
            "blocked_issue_codes": issue_codes,
            "blocked_issue_count": len(blocked_issues),
        },
    )
    metadata = dict(result.metadata)
    metadata["gate_policy_evaluation"] = {
        "blocked": True,
        "reason": "dependency_unavailable",
        "blocked_issue_codes": issue_codes,
    }
    return result.model_copy(
        update={
            "verdict": "fail",
            "issues": result.issues + (gate_issue,),
            "recommended_action": _dependency_unavailable_recommended_action(
                result.recommended_action
            ),
            "metadata": metadata,
        }
    )


def _dependency_unavailable_recommended_action(existing_action: str | None) -> str:
    if not existing_action:
        return DEPENDENCY_UNAVAILABLE_RECOMMENDED_ACTION
    if DEPENDENCY_UNAVAILABLE_RECOMMENDED_ACTION in existing_action:
        return existing_action
    return (
        f"{DEPENDENCY_UNAVAILABLE_RECOMMENDED_ACTION} "
        f"Existing recommendation: {existing_action}"
    )


def _dependency_unavailable_issue_codes(
    gate_policy: Mapping[str, Any],
) -> frozenset[str]:
    if "dependency_unavailable_issue_codes" not in gate_policy:
        return DEFAULT_DEPENDENCY_UNAVAILABLE_ISSUE_CODES
    configured_codes = _gate_policy_string_sequence(
        gate_policy.get("dependency_unavailable_issue_codes"),
        field_name="gate_policy.dependency_unavailable_issue_codes",
    )
    invalid_codes = [
        code
        for code in configured_codes
        if re.fullmatch(ISSUE_CODE_PATTERN, code) is None
    ]
    if invalid_codes:
        raise ValidationCliError(
            "gate_policy.dependency_unavailable_issue_codes contains invalid "
            f"issue code(s): {invalid_codes}"
        )
    return frozenset(configured_codes)


def _gate_policy_blocks(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return value.strip().lower() == "block"


def _optional_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _gate_policy_string_sequence(value: Any, *, field_name: str) -> tuple[str, ...]:
    """Normalize string/list fields before caller-specific validation."""

    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, Sequence) or isinstance(value, bytes | bytearray):
        raise ValidationCliError(f"{field_name} must be a string or sequence")
    values = tuple(value)
    invalid_types = [
        type(item).__name__ for item in values if not isinstance(item, str)
    ]
    if invalid_types:
        raise ValidationCliError(
            f"{field_name} must contain only strings; got {invalid_types}"
        )
    return values


def _write_model_json(model: Any, path: Path) -> None:
    path.write_text(model.model_dump_json(indent=2) + "\n", encoding="utf-8")
