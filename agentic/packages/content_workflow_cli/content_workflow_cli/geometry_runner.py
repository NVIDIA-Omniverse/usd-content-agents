# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thin public launcher for the deterministic Geometry workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from content_agent_workflows.common.artifacts import atomic_write_json, file_sha256

if TYPE_CHECKING:
    from content_agent_workflows.geometry import (
        GeometryWorkflowInput,
        GeometryWorkflowResult,
    )

GEOMETRY_CLI_REQUEST_SCHEMA_VERSION = "content-workflow-cli.geometry-request.v1"
GEOMETRY_CLI_RESULT_FILENAME = "geometry_workflow_result.json"
GEOMETRY_CLI_REQUEST_FILENAME = "geometry_request.json"
GEOMETRY_CLI_FAILURE_FILENAME = "geometry_cli_failure.json"

SourceAuthoringMode = Literal[
    "auto",
    "shared_conversion",
    "opaque_import",
    "parametric_recovery",
    "direct_preserve",
]
OptimizationPolicy = Literal["skip", "preserve_correspondence", "runtime_efficiency"]
RepairMode = Literal["off", "diagnose", "auto"]
RepairProfile = Literal[
    "visual_only",
    "static_environment",
    "rigid_pick_place",
    "articulated_rigid",
    "contact_rich",
    "deformable_or_cae",
]
RenderPreset = Literal["hero", "4view", "six_view", "vertical4", "material_review"]
OvRTXRenderMode = Literal["rt1", "rt2", "pt"]
RuntimeValidationMode = Literal[
    "skip", "authored_physics", "temporary_loadability_proxy"
]
RuntimeEngine = Literal["ovphysx", "fake", "none"]
SimReadyMode = Literal["skip", "validate", "validate_and_route_conformance"]


@dataclass(frozen=True)
class GeometryRunConfig:
    """Resolved user-facing configuration for one fresh Geometry run."""

    source_asset: Path | None
    source_manifest: Path | None
    output_dir: Path
    source_role: str | None = None
    prompt: str | None = None
    prompt_file: Path | None = None
    reference_image: Path | None = None
    output_usd: Path | None = None
    source_authoring_mode: SourceAuthoringMode = "auto"
    allow_lossy_recovery: bool = False
    target_profile: str = "geometry-agent.static-visual-asset.v1"
    target_runtime: str = "isaac-lab"
    optimization_policy: OptimizationPolicy = "preserve_correspondence"
    optimizer_backend: Literal["local", "remote"] = "local"
    repair_mode: RepairMode = "off"
    repair_profile: RepairProfile = "visual_only"
    render_evidence: bool = True
    render_preset: RenderPreset = "six_view"
    render_backend: Literal["ovrtx", "remote"] = "ovrtx"
    render_ovrtx_mode: OvRTXRenderMode = "pt"
    render_ovrtx_num_sensor_updates: int = 64
    segmentation_run_dir: Path | None = None
    segmentation_required: bool = False
    required_parts: list[str] = field(default_factory=list)
    runtime_validation_mode: RuntimeValidationMode = "skip"
    runtime_engine: RuntimeEngine = "none"
    simready_mode: SimReadyMode = "skip"
    simready_profile: str | None = None
    install_missing_converters: bool = False
    dry_run: bool = False
    json_output: bool = False


@dataclass(frozen=True)
class ResolvedGeometryRun:
    """Validated paths and typed workflow request for one CLI invocation."""

    output_dir: Path
    request_path: Path
    workflow_input: GeometryWorkflowInput
    request_record: dict[str, Any]
    render_evidence: bool
    dry_run: bool
    json_output: bool


def add_geometry_subcommands(subparsers: Any) -> None:
    """Register the public ``geometry run`` command."""

    geometry = subparsers.add_parser(
        "geometry",
        help="Prepare, validate, render, and package one Geometry handoff.",
    )
    geometry_subparsers = geometry.add_subparsers(dest="geometry_command")
    run = geometry_subparsers.add_parser(
        "run",
        help="Run one fresh deterministic Geometry workflow.",
    )
    _add_geometry_run_args(run)
    run.set_defaults(handler=_handle_geometry_run)


def _add_geometry_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "source_asset",
        type=Path,
        nargs="?",
        help="CAD, mesh, USD, or an admitted geometry source representation.",
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        help="Use a typed source manifest instead of SOURCE_ASSET.",
    )
    parser.add_argument(
        "--source-role",
        help="Optional declared representation role to select from a manifest.",
    )
    prompt = parser.add_mutually_exclusive_group()
    prompt.add_argument("--prompt", help="Optional source or acceptance context.")
    prompt.add_argument(
        "--prompt-file",
        type=Path,
        help="UTF-8 file containing optional source or acceptance context.",
    )
    parser.add_argument(
        "--reference-image",
        type=Path,
        help="Optional reference image bound to the workflow request.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Fresh directory for the request, handoff, evidence, and result.",
    )
    parser.add_argument(
        "--output-usd",
        type=Path,
        help="Output filename inside OUTPUT_DIR. Defaults to durable binary USDC.",
    )
    parser.add_argument(
        "--source-authoring-mode",
        choices=[
            "auto",
            "shared_conversion",
            "opaque_import",
            "parametric_recovery",
            "direct_preserve",
        ],
        default="auto",
    )
    parser.add_argument(
        "--allow-lossy-recovery",
        action="store_true",
        help="Allow an explicitly labeled lossy source-recovery route.",
    )
    parser.add_argument(
        "--target-profile",
        default="geometry-agent.static-visual-asset.v1",
        help="Frozen Geometry validation profile.",
    )
    parser.add_argument("--target-runtime", default="isaac-lab")
    parser.add_argument(
        "--optimization-policy",
        choices=["skip", "preserve_correspondence", "runtime_efficiency"],
        default="preserve_correspondence",
    )
    parser.add_argument(
        "--optimizer-backend", choices=["local", "remote"], default="local"
    )
    parser.add_argument(
        "--repair-mode", choices=["off", "diagnose", "auto"], default="off"
    )
    parser.add_argument(
        "--repair-profile",
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
        "--render-evidence",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require digest-bound OVRTX evidence. Enabled by default.",
    )
    parser.add_argument(
        "--render-preset",
        choices=[
            "hero",
            "4view",
            "six_view",
            "vertical4",
            "material_review",
        ],
        default="six_view",
    )
    parser.add_argument(
        "--render-backend",
        choices=["ovrtx", "remote"],
        default="ovrtx",
        help=(
            "Final Geometry evidence uses local OVRTX or a configured remote "
            "OVRTX service."
        ),
    )
    parser.add_argument(
        "--ovrtx-mode",
        choices=["rt1", "rt2", "pt"],
        default="pt",
        help="OVRTX render mode. The public launcher defaults to final-quality PT.",
    )
    parser.add_argument(
        "--ovrtx-sensor-updates",
        type=int,
        default=64,
        help="OVRTX accumulation iterations per view. Defaults to 64.",
    )
    parser.add_argument(
        "--segmentation-run-dir",
        type=Path,
        help="Completed mesh-segmentation run to consume and validate.",
    )
    parser.add_argument(
        "--segmentation-required",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--required-part",
        action="append",
        default=[],
        help="Required semantic part name. May be repeated.",
    )
    parser.add_argument(
        "--runtime-validation-mode",
        choices=["skip", "authored_physics", "temporary_loadability_proxy"],
        default="skip",
    )
    parser.add_argument(
        "--runtime-engine", choices=["ovphysx", "fake", "none"], default="none"
    )
    parser.add_argument(
        "--simready-mode",
        choices=["skip", "validate", "validate_and_route_conformance"],
        default="skip",
    )
    parser.add_argument("--simready-profile")
    parser.add_argument(
        "--install-missing-converters",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Explicitly allow the conversion workflow to install its allowlisted "
            "converter for this source type. Prefer converters provisioned in the "
            "runtime image."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and freeze the request without executing Geometry.",
    )
    parser.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="Print the complete request or workflow result as JSON.",
    )


def _handle_geometry_run(args: argparse.Namespace) -> int:
    config = GeometryRunConfig(
        source_asset=args.source_asset,
        source_manifest=args.source_manifest,
        source_role=args.source_role,
        prompt=args.prompt,
        prompt_file=args.prompt_file,
        reference_image=args.reference_image,
        output_dir=args.output_dir,
        output_usd=args.output_usd,
        source_authoring_mode=cast(SourceAuthoringMode, args.source_authoring_mode),
        allow_lossy_recovery=args.allow_lossy_recovery,
        target_profile=args.target_profile,
        target_runtime=args.target_runtime,
        optimization_policy=cast(OptimizationPolicy, args.optimization_policy),
        optimizer_backend=args.optimizer_backend,
        repair_mode=cast(RepairMode, args.repair_mode),
        repair_profile=cast(RepairProfile, args.repair_profile),
        render_evidence=args.render_evidence,
        render_preset=cast(RenderPreset, args.render_preset),
        render_backend=args.render_backend,
        render_ovrtx_mode=cast(OvRTXRenderMode, args.ovrtx_mode),
        render_ovrtx_num_sensor_updates=args.ovrtx_sensor_updates,
        segmentation_run_dir=args.segmentation_run_dir,
        segmentation_required=args.segmentation_required,
        required_parts=list(args.required_part),
        runtime_validation_mode=cast(
            RuntimeValidationMode, args.runtime_validation_mode
        ),
        runtime_engine=cast(RuntimeEngine, args.runtime_engine),
        simready_mode=cast(SimReadyMode, args.simready_mode),
        simready_profile=args.simready_profile,
        install_missing_converters=args.install_missing_converters,
        dry_run=args.dry_run,
        json_output=args.json_output,
    )
    return run_geometry_cli(config)


def run_geometry_cli(config: GeometryRunConfig) -> int:
    """Validate, freeze, execute, and report one Geometry workflow run."""

    resolved = _resolve_run(config)
    resolved.output_dir.mkdir(parents=True)
    atomic_write_json(resolved.request_path, resolved.request_record)

    if resolved.dry_run:
        if resolved.json_output:
            print(json.dumps(resolved.request_record, indent=2, sort_keys=True))
        else:
            print("Geometry dry run: ready")
            print(f"Request: {resolved.request_path}")
            print(f"Output directory: {resolved.output_dir}")
        return 0

    try:
        result = _execute_geometry_workflow(resolved.workflow_input)
    except Exception as exc:
        atomic_write_json(
            resolved.output_dir / GEOMETRY_CLI_FAILURE_FILENAME,
            {
                "schema_version": "content-workflow-cli.geometry-failure.v1",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "request_path": str(resolved.request_path),
            },
        )
        raise

    result_payload = result.model_dump(mode="json")
    result_path = resolved.output_dir / GEOMETRY_CLI_RESULT_FILENAME
    atomic_write_json(result_path, result_payload)

    if resolved.json_output:
        print(json.dumps(result_payload, indent=2, sort_keys=True))
    else:
        _print_geometry_summary(
            result,
            result_path=result_path,
            render_evidence=resolved.render_evidence,
        )
    return _geometry_exit_code(result)


def _resolve_run(config: GeometryRunConfig) -> ResolvedGeometryRun:
    from content_agent_workflows.geometry import GeometryWorkflowInput

    if (config.source_asset is None) == (config.source_manifest is None):
        raise ValueError("provide exactly one SOURCE_ASSET or --source-manifest")

    source = (
        _existing_file(config.source_asset, label="source asset")
        if config.source_asset is not None
        else None
    )
    source_manifest = (
        _existing_file(config.source_manifest, label="source manifest")
        if config.source_manifest is not None
        else None
    )
    reference_image = (
        _existing_file(config.reference_image, label="reference image")
        if config.reference_image is not None
        else None
    )
    prompt, prompt_path = _resolve_prompt(config.prompt, config.prompt_file)

    output_dir = _fresh_output_path(config.output_dir)
    output_usd = _resolve_output_usd(config.output_usd, output_dir=output_dir)

    segmentation_run_dir = None
    if config.segmentation_run_dir is not None:
        segmentation_run_dir = config.segmentation_run_dir.expanduser().resolve()
        if not segmentation_run_dir.is_dir():
            raise FileNotFoundError(
                f"segmentation run directory is not a directory: {segmentation_run_dir}"
            )
    segmentation_required = config.segmentation_required or bool(config.required_parts)
    if segmentation_required and segmentation_run_dir is None:
        raise ValueError(
            "--segmentation-required and --required-part require --segmentation-run-dir"
        )
    if config.runtime_validation_mode != "skip" and config.runtime_engine == "none":
        raise ValueError("runtime validation requires --runtime-engine ovphysx or fake")
    if not config.target_profile.strip():
        raise ValueError("--target-profile must not be empty")
    if not config.target_runtime.strip():
        raise ValueError("--target-runtime must not be empty")
    if config.render_ovrtx_num_sensor_updates < 1:
        raise ValueError("--ovrtx-sensor-updates must be positive")

    workflow_input = GeometryWorkflowInput(
        source_path=source,
        source_manifest_path=source_manifest,
        expected_source_sha256=file_sha256(source) if source is not None else None,
        expected_source_manifest_sha256=(
            file_sha256(source_manifest) if source_manifest is not None else None
        ),
        source_representation_role=config.source_role,
        prompt=prompt,
        image_path=reference_image,
        output_dir=output_dir,
        output_usd_path=output_usd,
        source_authoring_mode=config.source_authoring_mode,
        allow_lossy_recovery=config.allow_lossy_recovery,
        target_profile=config.target_profile.strip(),
        target_runtime=config.target_runtime.strip(),
        runtime_validation_mode=config.runtime_validation_mode,
        runtime_engine=config.runtime_engine,
        render_evidence=config.render_evidence,
        render_preset=config.render_preset,
        render_backend=config.render_backend,
        render_ovrtx_mode=config.render_ovrtx_mode,
        render_ovrtx_num_sensor_updates=config.render_ovrtx_num_sensor_updates,
        optimization_policy=config.optimization_policy,
        optimizer_backend=config.optimizer_backend,
        segmentation_run_dir=segmentation_run_dir,
        segmentation_required=segmentation_required,
        segmentation_required_parts=config.required_parts,
        repair_mode=config.repair_mode,
        repair_profile=config.repair_profile,
        install_missing_converters=config.install_missing_converters,
        simready_mode=config.simready_mode,
        simready_profile=config.simready_profile,
    )
    request_payload = workflow_input.model_dump(mode="json")
    request_record: dict[str, Any] = {
        "schema_version": GEOMETRY_CLI_REQUEST_SCHEMA_VERSION,
        "workflow": "geometry.run",
        "request": request_payload,
        "source_sha256": (
            workflow_input.expected_source_sha256
            or workflow_input.expected_source_manifest_sha256
        ),
        "prompt_sha256": (
            hashlib.sha256(prompt.encode("utf-8")).hexdigest() if prompt else None
        ),
        "prompt_file": str(prompt_path) if prompt_path else None,
        "reference_image_sha256": (
            file_sha256(reference_image) if reference_image else None
        ),
    }
    return ResolvedGeometryRun(
        output_dir=output_dir,
        request_path=output_dir / GEOMETRY_CLI_REQUEST_FILENAME,
        workflow_input=workflow_input,
        request_record=request_record,
        render_evidence=config.render_evidence,
        dry_run=config.dry_run,
        json_output=config.json_output,
    )


def _existing_file(path: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is not a file: {resolved}")
    return resolved


def _resolve_prompt(
    prompt: str | None,
    prompt_file: Path | None,
) -> tuple[str | None, Path | None]:
    if prompt is not None and prompt_file is not None:
        raise ValueError("provide only one --prompt or --prompt-file")
    if prompt_file is not None:
        resolved = _existing_file(prompt_file, label="prompt file")
        text = resolved.read_text(encoding="utf-8")
        if not text.strip():
            raise ValueError("prompt file must not be empty")
        return text, resolved
    if prompt is not None and not prompt.strip():
        raise ValueError("--prompt must not be empty")
    return prompt, None


def _fresh_output_path(path: Path) -> Path:
    candidate = Path(os.path.abspath(os.fspath(path.expanduser())))
    if candidate.is_symlink():
        raise FileExistsError(
            f"output directory already exists; choose a fresh run path: {candidate}"
        )
    try:
        resolved = candidate.parent.resolve() / candidate.name
    except OSError as exc:
        raise ValueError(f"unable to inspect output directory: {candidate}") from exc
    if resolved.exists() or resolved.is_symlink():
        raise FileExistsError(
            f"output directory already exists; choose a fresh run path: {resolved}"
        )
    return resolved


def _resolve_output_usd(path: Path | None, *, output_dir: Path) -> Path | None:
    if path is None:
        return None
    candidate = path.expanduser()
    resolved = (
        candidate.resolve()
        if candidate.is_absolute()
        else (output_dir / candidate).resolve()
    )
    if not resolved.is_relative_to(output_dir):
        raise ValueError("--output-usd must resolve inside --output-dir")
    if resolved.suffix.lower() not in {".usd", ".usda", ".usdc"}:
        raise ValueError("--output-usd must end in .usd, .usda, or .usdc")
    return resolved


def _execute_geometry_workflow(
    params: GeometryWorkflowInput,
) -> GeometryWorkflowResult:
    from content_agent_workflows.geometry import run_geometry_workflow

    return run_geometry_workflow(params)


def _geometry_exit_code(result: GeometryWorkflowResult) -> int:
    if result.handoff_ready == "yes":
        return 0
    if result.handoff_ready == "conditional":
        return 3
    return 1


def _print_geometry_summary(
    result: GeometryWorkflowResult,
    *,
    result_path: Path,
    render_evidence: bool,
) -> None:
    print(f"Geometry validation: {result.validation_status}")
    print(f"Handoff ready: {result.handoff_ready}")
    if result.geometry_usd_path:
        print(f"Geometry USD: {result.geometry_usd_path}")
    if result.handoff_manifest_path:
        print(f"Manifest: {result.handoff_manifest_path}")
    if result.validation_evidence_path:
        print(f"Validation evidence: {result.validation_evidence_path}")
    if result.render_report_path:
        print(f"OVRTX evidence: {result.render_report_path}")
    elif not render_evidence:
        print("OVRTX evidence: not requested (visual qualification is incomplete)")
    if result.error:
        print(f"Error: {result.error}")
    print(f"Result: {result_path}")


__all__ = [
    "GEOMETRY_CLI_FAILURE_FILENAME",
    "GEOMETRY_CLI_REQUEST_FILENAME",
    "GEOMETRY_CLI_REQUEST_SCHEMA_VERSION",
    "GEOMETRY_CLI_RESULT_FILENAME",
    "GeometryRunConfig",
    "add_geometry_subcommands",
    "run_geometry_cli",
]
