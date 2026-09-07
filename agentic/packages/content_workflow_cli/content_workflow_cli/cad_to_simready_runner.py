# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thin launcher for the canonical composed CAD-to-SimReady workflow."""

from __future__ import annotations

import argparse
import json
import math
import re
import shlex
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from content_agent_workflows.cad_to_simready import (
    CAD_TO_SIMREADY_WORKFLOW_ENTRYPOINT,
    CAD_TO_SIMREADY_WORKFLOW_SKILL,
    CadToSimReadyInvocation,
    CadToSimReadyRequest,
    CadToSimReadyResult,
    bind_artifact,
    execute_cad_to_simready_workflow,
    flatten_usd_for_physics,
    render_final_simready_evidence,
    source_format,
)

from .entrypoint import UNSUPPORTED_CAD_TO_SIMREADY_EXECUTION_HOST_MESSAGE
from .runner import PROMPT_MODE_SKILL_ROUTED, find_repo_root

_CAD_CONVERTER_FORMATS = frozenset(
    {"sldasm", "sldprt", "step", "stp", "catpart", "catproduct"}
)
_CANONICAL_REPAIRS = ("NP.006", "UN.006")
_FINAL_REPAIRS = (*_CANONICAL_REPAIRS, "NP.005", "UN.007", "GSP.001")


@dataclass(frozen=True)
class CadToSimReadyRunConfig:
    """Resolved launcher configuration for one source asset."""

    source_asset: Path
    output_dir: Path
    repo_root: Path
    materials_yaml: Path
    materials_usd: Path | None = None
    asset_id: str | None = None
    source_meters_per_unit: float | None = None
    profile: str = "Prop-Robotics-Neutral"
    profile_version: str = "1.0.0"
    runner: str = "codex"
    model: str | None = None
    model_reasoning_effort: str | None = None
    max_iterations: int = 3
    child_timeout_seconds: float = 3600.0
    scene_tool_timeout_seconds: float = 300.0
    collision_approximation: str = "convexHull"
    simulation_duration_seconds: float = 1.0
    simulation_sample_fps: int = 30
    render_width: int = 768
    render_height: int = 576
    turntable_frame_count: int = 24
    turntable_fps: float = 12.5
    install_missing: bool = True
    dry_run: bool = False


def add_cad_to_simready_subcommands(subparsers: Any) -> None:
    """Register ``cad-to-simready run`` as the public workflow entry point."""

    workflow = subparsers.add_parser(
        "cad-to-simready",
        help="Convert a source asset, assign materials and physics, and validate SimReady USD.",
    )
    workflow_subparsers = workflow.add_subparsers(dest="cad_to_simready_command")
    run = workflow_subparsers.add_parser(
        "run",
        help="Run the canonical composed CAD-to-SimReady workflow.",
    )
    _add_run_args(run)
    run.set_defaults(handler=_handle_run)
    flatten = workflow_subparsers.add_parser(
        "flatten-for-physics",
        help="Create the deterministic, non-instanceable physics-authoring handoff.",
    )
    flatten.add_argument("source_usd", type=Path)
    flatten.add_argument("output_usd", type=Path)
    flatten.add_argument("--report", required=True, type=Path)
    flatten.add_argument("--source-meters-per-unit", type=float, default=None)
    flatten.add_argument("--json", action="store_true")
    flatten.set_defaults(handler=_handle_flatten_for_physics)
    render = workflow_subparsers.add_parser(
        "render-final",
        help="Render presentation evidence from a strictly validated final SimReady USD.",
    )
    render.add_argument("final_usd", type=Path)
    render.add_argument("--validation-report", required=True, type=Path)
    render.add_argument("--output-dir", required=True, type=Path)
    render.add_argument("--report", required=True, type=Path)
    render.add_argument("--width", type=int, default=768)
    render.add_argument("--height", type=int, default=576)
    render.add_argument("--turntable-frames", type=int, default=24)
    render.add_argument("--turntable-fps", type=float, default=12.5)
    render.add_argument("--scene-tool-timeout", type=float, default=300.0)
    render.add_argument("--json", action="store_true")
    render.set_defaults(handler=_handle_render_final)


def add_physics_runtime_preflight_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the reusable OvPhysX readiness command."""

    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument(
        "--install-missing",
        dest="install_missing",
        action="store_true",
        default=True,
        help="Install and probe the exact locked runtime when absent. This is the default.",
    )
    parser.add_argument(
        "--no-install-missing",
        dest="install_missing",
        action="store_false",
        help="Check readiness and emit installation commands without modifying the host.",
    )
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--json", action="store_true")


def handle_physics_runtime_preflight(args: argparse.Namespace) -> int:
    """Run the public physics-runtime readiness check."""

    from content_agent_workflows.physics import preflight_ovphysx_runtime

    repo_root = (
        args.repo_root.expanduser().resolve() if args.repo_root else find_repo_root()
    )
    report = preflight_ovphysx_runtime(
        repo_root=repo_root,
        install_missing=args.install_missing,
    )
    payload = report.model_dump(mode="json")
    if args.report is not None:
        report_path = args.report.expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        target = report.remote_url or report.venv_path
        print(
            f"physics-runtime preflight {report.status.lower()} "
            f"({report.executor}): {target}"
        )
        if report.errors:
            for error in report.errors:
                print(f"error: {error}")
        if not report.passed and report.install_commands:
            print("Install commands:")
            for command in report.install_commands:
                print("  " + shlex.join(command))
    return 0 if report.passed else 1


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("source_asset", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--asset-id", default=None)
    parser.add_argument(
        "--source-meters-per-unit",
        type=float,
        default=None,
        help=(
            "Explicit coordinate-unit scale for unitless source formats. "
            "Omit to preserve converter-authored stage units."
        ),
    )
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--materials-yaml", required=True, type=Path)
    parser.add_argument("--materials-usd", type=Path, default=None)
    parser.add_argument("--profile", default="Prop-Robotics-Neutral")
    parser.add_argument("--profile-version", default="1.0.0")
    parser.add_argument("--runner", choices=["codex", "claude"], default="codex")
    parser.add_argument(
        "--model",
        default=None,
        help="Optional child-model override. Omit to inherit the user's runner setting.",
    )
    parser.add_argument("--model-reasoning-effort", default=None)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--child-timeout", type=float, default=3600.0)
    parser.add_argument("--scene-tool-timeout", type=float, default=300.0)
    parser.add_argument("--collision-approximation", default="convexHull")
    parser.add_argument("--duration-s", type=float, default=1.0)
    parser.add_argument("--sample-fps", type=int, default=30)
    parser.add_argument("--render-width", type=int, default=768)
    parser.add_argument("--render-height", type=int, default=576)
    parser.add_argument("--turntable-frames", type=int, default=24)
    parser.add_argument("--turntable-fps", type=float, default=12.5)
    parser.add_argument(
        "--install-missing",
        dest="install_missing",
        action="store_true",
        default=True,
        help="Install missing converter, physics runtime, and SimReady dependencies.",
    )
    parser.add_argument(
        "--no-install-missing",
        dest="install_missing",
        action="store_false",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")


def _handle_run(args: argparse.Namespace) -> int:
    config = CadToSimReadyRunConfig(
        source_asset=args.source_asset.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        repo_root=(
            args.repo_root.expanduser().resolve()
            if args.repo_root is not None
            else find_repo_root()
        ),
        materials_yaml=args.materials_yaml.expanduser().resolve(),
        materials_usd=(
            args.materials_usd.expanduser().resolve()
            if args.materials_usd is not None
            else None
        ),
        asset_id=args.asset_id,
        source_meters_per_unit=args.source_meters_per_unit,
        profile=args.profile,
        profile_version=args.profile_version,
        runner=args.runner,
        model=args.model,
        model_reasoning_effort=args.model_reasoning_effort,
        max_iterations=args.max_iterations,
        child_timeout_seconds=args.child_timeout,
        scene_tool_timeout_seconds=args.scene_tool_timeout,
        collision_approximation=args.collision_approximation,
        simulation_duration_seconds=args.duration_s,
        simulation_sample_fps=args.sample_fps,
        render_width=args.render_width,
        render_height=args.render_height,
        turntable_frame_count=args.turntable_frames,
        turntable_fps=args.turntable_fps,
        install_missing=args.install_missing,
        dry_run=args.dry_run,
    )
    try:
        result = run_cad_to_simready(config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"cad-to-simready failed: {exc}", file=sys.stderr)
        return 2
    payload = result.model_dump(mode="json")
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"cad-to-simready {result.status}: {result.asset_id}")
        print(f"run_dir: {config.output_dir}")
        if result.output_asset is not None:
            print(f"output_usd: {config.output_dir / result.output_asset.path}")
        if result.error:
            print(f"error: {result.error}")
    if result.status in {"completed", "planned"}:
        return 0
    return 2 if result.status == "blocked" else 1


def _handle_flatten_for_physics(args: argparse.Namespace) -> int:
    try:
        payload = flatten_usd_for_physics(
            args.source_usd,
            args.output_usd,
            args.report,
            source_meters_per_unit=args.source_meters_per_unit,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"cad-to-simready flatten-for-physics failed: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"cad-to-simready flatten-for-physics: {args.output_usd}")
    return 0


def _handle_render_final(args: argparse.Namespace) -> int:
    try:
        payload = render_final_simready_evidence(
            args.final_usd,
            args.validation_report,
            args.output_dir,
            args.report,
            width=args.width,
            height=args.height,
            turntable_frame_count=args.turntable_frames,
            turntable_fps=args.turntable_fps,
            scene_tool_timeout_seconds=args.scene_tool_timeout,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"cad-to-simready render-final failed: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"cad-to-simready final renders: {args.output_dir}")
        print(f"turntable_image: {payload['artifacts']['turntable_image']['path']}")
    return 0


def _asset_id(config: CadToSimReadyRunConfig) -> str:
    if config.asset_id:
        return config.asset_id
    value = re.sub(r"[^a-zA-Z0-9_.-]+", "-", config.source_asset.stem).strip("-.")
    return value or "asset"


def _paths(config: CadToSimReadyRunConfig) -> dict[str, Path]:
    stem = config.source_asset.name
    if stem.lower().endswith(".forge.step"):
        stem = stem[: -len(".forge.step")]
    else:
        stem = config.source_asset.stem
    root = config.output_dir
    converted = root / "convert" / f"{stem}.usda"
    return {
        "source": config.source_asset,
        "request": root / "request.json",
        "result": root / "result.json",
        "plan": root / "run_plan.json",
        "log": root / "run.log",
        "converter_preflight": root / "preflight" / "convert-to-usd.json",
        "physics_runtime_preflight": root / "preflight" / "physics-runtime.json",
        "simready_preflight": root / "preflight" / "simready-foundation.json",
        "conversion_report": root / "convert" / "conversion_report.json",
        "conversion_markdown": root / "convert" / "conversion_report.md",
        "converted_usd": converted,
        "canonicalization_report": root
        / "convert"
        / "canonical"
        / "canonicalization.json",
        "canonical_usd": root / "convert" / "canonical" / "staged" / converted.name,
        "materialized_usd": root / "material" / "materialized.usdc",
        "flatten_report": root / "physics" / "flatten_report.json",
        "flattened_physics_usd": root / "convert" / f"{stem}.physics-input.usda",
        "physics_units_report": root / "physics" / "unit-normalization.json",
        "physics_input_usd": root
        / "physics"
        / "normalized"
        / "staged"
        / f"{stem}.physics-input.usda",
        # SimReady Foundation accepts the generic .usd and text .usda suffixes,
        # but silently emits an empty result for a binary layer named .usdc.
        # Keep the efficient binary encoding while using the accepted .usd
        # container suffix at this workflow boundary.
        "physics_usd": root / "physics" / "physics.usd",
        "initial_validation_report": root / "simready" / "simready-profile.json",
        "conformance_report": root
        / "simready"
        / "conform"
        / "simready-conform-profile.json",
        "final_validation_report": root
        / "simready"
        / "simready-profile-after-conform.json",
        "final_render_manifest": root / "simready" / "final-render-manifest.json",
        "final_render_receipt": root
        / "simready"
        / "final_renders"
        / "raw"
        / "usd_cli_command_receipts.jsonl",
        "final_render_receipt_checkpoint": root
        / "simready"
        / "final_renders"
        / "raw"
        / "usd_cli_command_receipts.checkpoint.json",
        "final_render_records": root
        / "simready"
        / "final_renders"
        / "final_render_records.json",
        "final_hero_render": root / "simready" / "final_renders" / "final_hero.png",
        "final_multiview_render": root
        / "simready"
        / "final_renders"
        / "final_six_view.png",
        "final_turntable_render": root
        / "simready"
        / "final_renders"
        / "final_turntable.gif",
    }


def _agent_args(config: CadToSimReadyRunConfig) -> list[str]:
    args = ["--runner", config.runner]
    if config.model:
        args.extend(["--model", config.model])
    if config.model_reasoning_effort:
        args.extend(["--model-reasoning-effort", config.model_reasoning_effort])
    args.extend(["--child-timeout", str(config.child_timeout_seconds)])
    return args


def _scene_tool_args(config: CadToSimReadyRunConfig) -> list[str]:
    return ["--scene-tool-timeout", str(config.scene_tool_timeout_seconds)]


def _repair_args(requirements: tuple[str, ...]) -> list[str]:
    return [
        value for requirement in requirements for value in ("--repair", requirement)
    ]


def _command_plan(
    config: CadToSimReadyRunConfig,
    paths: dict[str, Path],
) -> list[dict[str, Any]]:
    install_flag = (
        "--install-missing" if config.install_missing else "--no-install-missing"
    )
    reference_asset = paths["converted_usd"]
    commands: list[dict[str, Any]] = [
        {
            "stage": "preflight",
            "step": "convert-to-usd-preflight",
            "argv": [
                "preflight",
                "convert-to-usd",
                str(config.source_asset),
                install_flag,
                "--report",
                str(paths["converter_preflight"]),
            ],
        },
        {
            "stage": "preflight",
            "step": "physics-runtime-preflight",
            "argv": [
                "preflight",
                "physics-runtime",
                "--repo-root",
                str(config.repo_root),
                install_flag,
                "--report",
                str(paths["physics_runtime_preflight"]),
            ],
        },
        {
            "stage": "preflight",
            "step": "simready-foundation-preflight",
            "argv": [
                "preflight",
                "simready-foundation",
                install_flag,
                "--report",
                str(paths["simready_preflight"]),
            ],
        },
        {
            "stage": "convert",
            "step": "convert-to-usd",
            "argv": [
                "convert-to-usd",
                str(config.source_asset),
                str(paths["converted_usd"]),
                "--output-dir",
                str(paths["converted_usd"].parent),
                install_flag,
                "--report",
                str(paths["conversion_report"]),
                "--markdown-report",
                str(paths["conversion_markdown"]),
            ],
        },
        {
            "stage": "convert",
            "step": "canonicalize-usd",
            "argv": [
                "simready",
                "conform-profile",
                str(paths["converted_usd"]),
                "--output-dir",
                str(paths["canonical_usd"].parents[1]),
                "--source-asset",
                str(config.source_asset),
                "--profile",
                config.profile,
                "--profile-version",
                config.profile_version,
                *_repair_args(_CANONICAL_REPAIRS),
                "--report",
                str(paths["canonicalization_report"]),
                "--strict",
                "--force",
            ],
        },
        {
            "stage": "material",
            "step": "assign-materials",
            "argv": [
                "materials",
                "assign",
                "--usd",
                str(paths["canonical_usd"]),
                "--reference",
                str(reference_asset),
                "--materials-yaml",
                str(config.materials_yaml),
                "--output-dir",
                str(paths["materialized_usd"].parent),
                "--output-usd",
                str(paths["materialized_usd"]),
                "--repo-root",
                str(config.repo_root),
                "--optimizer-selection",
                "agent",
                "--prompt-mode",
                PROMPT_MODE_SKILL_ROUTED,
                "--vqa-refinement-max-iterations",
                str(config.max_iterations),
                *_agent_args(config),
                *_scene_tool_args(config),
            ],
        },
        {
            "stage": "physics",
            "step": "flatten-for-physics",
            "argv": [
                "cad-to-simready",
                "flatten-for-physics",
                str(paths["materialized_usd"]),
                str(paths["flattened_physics_usd"]),
                "--report",
                str(paths["flatten_report"]),
            ],
        },
        {
            "stage": "physics",
            "step": "normalize-units-for-physics",
            "argv": [
                "simready",
                "conform-profile",
                str(paths["flattened_physics_usd"]),
                "--output-dir",
                str(paths["physics_input_usd"].parents[1]),
                "--source-asset",
                str(config.source_asset),
                "--profile",
                config.profile,
                "--profile-version",
                config.profile_version,
                "--repair",
                "UN.007",
                "--report",
                str(paths["physics_units_report"]),
                "--strict",
                "--force",
            ],
        },
        {
            "stage": "physics",
            "step": "apply-physics",
            "argv": [
                "physics",
                "apply",
                "--usd",
                str(paths["physics_input_usd"]),
                "--reference",
                str(reference_asset),
                "--output-dir",
                str(paths["physics_usd"].parent),
                "--output-usd",
                str(paths["physics_usd"]),
                "--repo-root",
                str(config.repo_root),
                "--optimizer-selection",
                "agent",
                "--prompt-mode",
                PROMPT_MODE_SKILL_ROUTED,
                "--collision-approximation",
                config.collision_approximation,
                "--duration-s",
                str(config.simulation_duration_seconds),
                "--sample-fps",
                str(config.simulation_sample_fps),
                "--visual-validation-max-iterations",
                str(config.max_iterations),
                *_agent_args(config),
                *_scene_tool_args(config),
            ],
        },
        {
            "stage": "validation",
            "step": "initial-simready-validation",
            "argv": [
                "simready",
                "validate-profile",
                str(paths["physics_usd"]),
                "--profile",
                config.profile,
                "--profile-version",
                config.profile_version,
                "--report",
                str(paths["initial_validation_report"]),
            ],
        },
        {
            "stage": "validation",
            "step": "conform-simready-profile",
            "argv": [
                "simready",
                "conform-profile",
                str(paths["physics_usd"]),
                "--output-dir",
                str(paths["conformance_report"].parent),
                "--profile",
                config.profile,
                "--profile-version",
                config.profile_version,
                "--validation-report",
                str(paths["initial_validation_report"]),
                "--source-asset",
                str(config.source_asset),
                "--grasp-prim",
                "<default-prim-from-flatten-report>",
                *_repair_args(_FINAL_REPAIRS),
                "--report",
                str(paths["conformance_report"]),
                "--strict",
                "--force",
            ],
        },
        {
            "stage": "validation",
            "step": "final-simready-validation",
            "argv": [
                "simready",
                "validate-profile",
                "<conformed-output-usd>",
                "--profile",
                config.profile,
                "--profile-version",
                config.profile_version,
                "--report",
                str(paths["final_validation_report"]),
                "--strict",
            ],
        },
        {
            "stage": "validation",
            "step": "render-final-simready",
            "argv": [
                "cad-to-simready",
                "render-final",
                "<conformed-output-usd>",
                "--validation-report",
                str(paths["final_validation_report"]),
                "--output-dir",
                str(paths["final_hero_render"].parent),
                "--report",
                str(paths["final_render_manifest"]),
                "--width",
                str(config.render_width),
                "--height",
                str(config.render_height),
                "--turntable-frames",
                str(config.turntable_frame_count),
                "--turntable-fps",
                str(config.turntable_fps),
                "--scene-tool-timeout",
                str(config.scene_tool_timeout_seconds),
            ],
        },
    ]
    if config.materials_usd is not None:
        material = next(
            command for command in commands if command["step"] == "assign-materials"
        )
        material["argv"].extend(["--materials-usd", str(config.materials_usd)])
    if config.source_meters_per_unit is not None:
        flatten = next(
            command for command in commands if command["step"] == "flatten-for-physics"
        )
        flatten["argv"].extend(
            ["--source-meters-per-unit", str(config.source_meters_per_unit)]
        )
    return commands


def _failed_planned_invocation(
    command: dict[str, Any],
    *,
    log: TextIO,
    error: str,
    argv: list[str] | None = None,
) -> CadToSimReadyInvocation:
    """Persist a typed failure when a planned command cannot be resolved."""

    log.write(f"error: {error}\n")
    log.flush()
    failed_argv = command["argv"] if argv is None else argv
    return CadToSimReadyInvocation(
        stage=command["stage"],
        step=command["step"],
        argv=["content-workflow-cli", *failed_argv],
        exit_code=1,
    )


def _replace_placeholder(
    command: dict[str, Any],
    placeholder: str,
    value: str,
    *,
    log: TextIO,
) -> list[str] | CadToSimReadyInvocation:
    """Resolve one planned placeholder or return an auditable typed failure."""

    updated = list(command["argv"])
    indexes = [index for index, item in enumerate(updated) if item == placeholder]
    if len(indexes) != 1:
        return _failed_planned_invocation(
            command,
            log=log,
            error=(
                f"command plan for {command['step']} must contain exactly one "
                f"{placeholder!r} placeholder; found {len(indexes)}"
            ),
        )
    updated[indexes[0]] = value
    return updated


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _resolved_conformed_output_path(*, report_path: Path, run_dir: Path) -> Path:
    """Resolve a report-selected output only when it is a contained file."""

    output = _read_json(report_path).get("output_usd_path")
    if not isinstance(output, str) or not output.strip():
        raise ValueError("conformance report has no non-empty output USD path")
    try:
        candidate = Path(output).expanduser().resolve(strict=True)
        contained_run_dir = run_dir.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError(
            "conformance report output USD path does not resolve to an existing file"
        ) from exc
    if not candidate.is_relative_to(contained_run_dir) or not candidate.is_file():
        raise ValueError(
            "conformance report output USD path is not a file contained by the run"
        )
    return candidate


def _run_cli(argv: list[str], log: TextIO) -> int:
    from .cli import main

    with redirect_stdout(log), redirect_stderr(log):
        return int(main(argv))


def _invoke_cli(
    command: dict[str, Any],
    *,
    log: TextIO,
    argv: list[str] | None = None,
) -> CadToSimReadyInvocation:
    resolved = list(argv or command["argv"])
    log.write(f"\n=== CAD-to-SimReady {command['stage']}: {command['step']} ===\n")
    log.write("$ content-workflow-cli " + shlex.join(resolved) + "\n")
    log.flush()
    try:
        exit_code = _run_cli(resolved, log)
    except SystemExit as exc:
        # Keep command-plan/parser drift inside the auditable workflow result,
        # including when a replaced or older CLI entry point still raises.
        exit_code = exc.code if isinstance(exc.code, int) else 2
        log.write(
            "content-workflow-cli terminated while parsing the invocation "
            f"(exit code {exit_code}).\n"
        )
        log.flush()
    except Exception:  # noqa: BLE001 - persist an auditable failed invocation
        log.write("content-workflow-cli raised an unexpected exception:\n")
        traceback.print_exc(file=log)
        log.flush()
        exit_code = 1
    return CadToSimReadyInvocation(
        stage=command["stage"],
        step=command["step"],
        argv=["content-workflow-cli", *resolved],
        exit_code=exit_code,
    )


def _request(config: CadToSimReadyRunConfig) -> CadToSimReadyRequest:
    source_kind = source_format(config.source_asset)
    return CadToSimReadyRequest(
        asset_id=_asset_id(config),
        source=bind_artifact(config.source_asset),
        source_format=source_kind,
        source_meters_per_unit=config.source_meters_per_unit,
        requires_cad_converter=source_kind in _CAD_CONVERTER_FORMATS,
        profile=config.profile,
        profile_version=config.profile_version,
        materials_yaml=bind_artifact(config.materials_yaml),
        materials_usd=(
            bind_artifact(config.materials_usd)
            if config.materials_usd is not None
            else None
        ),
        runner=config.runner,  # type: ignore[arg-type]
        model=config.model,
        model_reasoning_effort=config.model_reasoning_effort,
        max_iterations=config.max_iterations,
        install_missing=config.install_missing,
        scene_tool_timeout_seconds=config.scene_tool_timeout_seconds,
        render_width=config.render_width,
        render_height=config.render_height,
        turntable_frame_count=config.turntable_frame_count,
        turntable_fps=config.turntable_fps,
    )


def run_cad_to_simready(config: CadToSimReadyRunConfig) -> CadToSimReadyResult:
    """Execute the typed stage contract through existing public workflows."""

    # This is a coarse native-host prefilter. Child runners remain authoritative
    # for provider-specific sandbox prerequisites before provider startup.
    if not config.dry_run and sys.platform not in {"linux", "win32"}:
        raise ValueError(UNSUPPORTED_CAD_TO_SIMREADY_EXECUTION_HOST_MESSAGE)
    if not config.source_asset.is_file():
        raise FileNotFoundError(f"source asset is missing: {config.source_asset}")
    if not config.materials_yaml.is_file():
        raise FileNotFoundError(f"materials YAML is missing: {config.materials_yaml}")
    if config.materials_usd is not None and not config.materials_usd.is_file():
        raise FileNotFoundError(f"materials USD is missing: {config.materials_usd}")
    if config.max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")
    if config.source_meters_per_unit is not None and (
        not math.isfinite(config.source_meters_per_unit)
        or config.source_meters_per_unit <= 0
    ):
        raise ValueError("source_meters_per_unit must be finite and positive")
    if (
        config.render_width < 64
        or config.render_height < 64
        or config.render_width % 2
        or config.render_height % 2
    ):
        raise ValueError("render width and height must be even integers >= 64")
    if config.turntable_frame_count < 2:
        raise ValueError("turntable frame count must be at least 2")
    if config.turntable_fps <= 0:
        raise ValueError("turntable fps must be positive")
    if config.scene_tool_timeout_seconds <= 0:
        raise ValueError("scene_tool_timeout_seconds must be positive")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    paths = _paths(config)
    session_owned_paths = {
        "final_render_receipt",
        "final_render_receipt_checkpoint",
    }
    for name, path in paths.items():
        if name != "source" and name not in session_owned_paths:
            path.parent.mkdir(parents=True, exist_ok=True)
    request = _request(config)
    paths["request"].write_text(
        request.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    commands = _command_plan(config, paths)
    paths["plan"].write_text(
        json.dumps(
            {
                "schema_version": "content-agent-workflows.cad-to-simready-plan.v1",
                "asset_id": request.asset_id,
                "workflow_skill": CAD_TO_SIMREADY_WORKFLOW_SKILL,
                "workflow_entrypoint": list(CAD_TO_SIMREADY_WORKFLOW_ENTRYPOINT),
                "prompt_mode": PROMPT_MODE_SKILL_ROUTED,
                "model": config.model,
                "source_meters_per_unit": config.source_meters_per_unit,
                "commands": [
                    {**command, "argv": ["content-workflow-cli", *command["argv"]]}
                    for command in commands
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if config.dry_run:

        def unexpected_invocation(step: str) -> CadToSimReadyInvocation:
            raise AssertionError(f"planned workflow invoked {step}")

        result = execute_cad_to_simready_workflow(
            request=request,
            request_path=paths["request"],
            run_dir=config.output_dir,
            artifact_paths=paths,
            invoke=unexpected_invocation,
            planned=True,
        )
        paths["result"].write_text(
            result.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        return result

    command_by_step = {command["step"]: command for command in commands}
    with paths["log"].open("a", encoding="utf-8") as log:

        def invoke(step: str) -> CadToSimReadyInvocation:
            command = command_by_step[step]
            argv: list[str] | None = None
            if step == "conform-simready-profile":
                default_prim = _read_json(paths["flatten_report"]).get(
                    "default_prim_path"
                )
                if not isinstance(default_prim, str) or not default_prim.startswith(
                    "/"
                ):
                    return _failed_planned_invocation(
                        command,
                        log=log,
                        error="flatten report has no valid default prim path",
                    )
                resolved = _replace_placeholder(
                    command,
                    "<default-prim-from-flatten-report>",
                    default_prim,
                    log=log,
                )
                if isinstance(resolved, CadToSimReadyInvocation):
                    return resolved
                argv = resolved
            elif step in {"final-simready-validation", "render-final-simready"}:
                resolved = _replace_placeholder(
                    command,
                    "<conformed-output-usd>",
                    str(paths["final_usd"]),
                    log=log,
                )
                if isinstance(resolved, CadToSimReadyInvocation):
                    return resolved
                argv = resolved
            invocation = _invoke_cli(command, log=log, argv=argv)
            if step == "conform-simready-profile" and invocation.exit_code == 0:
                try:
                    paths["final_usd"] = _resolved_conformed_output_path(
                        report_path=paths["conformance_report"],
                        run_dir=config.output_dir,
                    )
                except ValueError as exc:
                    return _failed_planned_invocation(
                        command,
                        log=log,
                        error=str(exc),
                        argv=invocation.argv[1:],
                    )
            return invocation

        result = execute_cad_to_simready_workflow(
            request=request,
            request_path=paths["request"],
            run_dir=config.output_dir,
            artifact_paths=paths,
            invoke=invoke,
        )
    paths["result"].write_text(
        result.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    return result
