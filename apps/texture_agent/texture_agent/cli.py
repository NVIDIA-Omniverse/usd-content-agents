# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Texture Agent CLI -- generate and apply textures to USD materials."""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from dotenv import load_dotenv
from world_understanding.agentic.cli import (
    load_cli_config_mapping,
    normalize_cli_step_filters,
    sever_cli_exception_graph,
)

from texture_agent.utils import get_version

load_dotenv()

app = typer.Typer(
    name="texture-agent",
    help="Generate and apply textures to USD materials.",
    no_args_is_help=True,
)


def _is_typer_default(value: object) -> bool:
    return isinstance(value, typer.models.ParameterInfo)


def _setup_logging(verbose: bool = False) -> None:
    if _is_typer_default(verbose):
        verbose = False
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"texture-agent {get_version()}")
        raise typer.Exit()


def _apply_detail_policy_override(cfg: dict, detail_policy: str | None) -> None:
    if detail_policy is None or _is_typer_default(detail_policy):
        return
    from texture_agent.functions.detail_policy import normalize_detail_policy

    texture = cfg.setdefault("texture", {})
    texture["detail_policy"] = normalize_detail_policy(
        detail_policy,
        config_key="--detail-policy",
    )


def _apply_planning_overrides(
    cfg: dict,
    *,
    discovery_mode: str | None = None,
    unit_mode: str | None = None,
    operator_override_cap: int | None = None,
    explicit_material_paths: str | None = None,
    explicit_prim_paths: str | None = None,
    plan_only: bool = False,
) -> None:
    planning = cfg.setdefault("planning", {})
    for key, value in (
        ("discovery_mode", discovery_mode),
        ("unit_mode", unit_mode),
        ("operator_override_cap", operator_override_cap),
    ):
        if value is not None and not _is_typer_default(value):
            planning[key] = value
    for key, value in (
        ("explicit_material_paths", explicit_material_paths),
        ("explicit_prim_paths", explicit_prim_paths),
    ):
        if value is not None and not _is_typer_default(value):
            planning[key] = [item.strip() for item in value.split(",") if item.strip()]
    if plan_only:
        planning["plan_only"] = True


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """Texture Agent: generate and apply textures to USD materials."""
    pass


@app.command()
@sever_cli_exception_graph
def run(
    config: Path = typer.Argument(..., help="Path to the pipeline config YAML"),
    skip: str | None = typer.Option(
        None, "--skip", help="Comma-separated step names to skip"
    ),
    only: str | None = typer.Option(
        None, "--only", help="Comma-separated step names to run exclusively"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show execution plan without running"
    ),
    resume: bool = typer.Option(
        False, "--resume", help="Reuse existing artifacts from the working directory"
    ),
    session_id: str | None = typer.Option(
        None, "--session-id", help="Reuse or override the config session ID"
    ),
    detail_policy: str | None = typer.Option(
        None,
        "--detail-policy",
        help="Override texture detail policy: default or surface_only.",
    ),
    discovery_mode: str | None = typer.Option(
        None,
        "--discovery-mode",
        help="Planning scope: effective_bound, explicit, or all_authored.",
    ),
    unit_mode: str | None = typer.Option(
        None,
        "--unit-mode",
        help="Planning unit mode: per_material, per_group, or per_prim.",
    ),
    operator_override_cap: int | None = typer.Option(
        None,
        "--operator-override-cap",
        min=1,
        max=64,
        help="Intentional texture-unit cap override, never greater than 64.",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
) -> None:
    """Run the full texture pipeline."""
    _setup_logging(verbose)
    logger = logging.getLogger(__name__)

    from texture_agent.config.unified_config import config_to_context, load_config
    from texture_agent.workflows.factory import run_pipeline

    try:
        from texture_agent.config.schema import STEP_ORDER

        skip_list, only_list = normalize_cli_step_filters(
            skip=skip,
            only=only,
            valid_steps=STEP_ORDER,
        )
    except ValueError as error:
        logger.error("Pipeline step filter validation failed: %s", error)
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from None

    try:
        config_data = load_cli_config_mapping(config)
    except (OSError, ValueError) as error:
        logger.error("Pipeline configuration validation failed: %s", error)
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from None

    try:
        cfg = load_config(
            config,
            session_id=session_id,
            config_data=config_data,
        )
        _apply_detail_policy_override(cfg, detail_policy)
        _apply_planning_overrides(
            cfg,
            discovery_mode=discovery_mode,
            unit_mode=unit_mode,
            operator_override_cap=operator_override_cap,
        )
        context = config_to_context(cfg)
        context["resume"] = resume

        context = run_pipeline(context, skip=skip_list, only=only_list, dry_run=dry_run)

        if not dry_run:
            # Print summary
            output_paths = context.get("output_usd_paths", [])
            render_paths = context.get("rendered_image_paths", [])
            typer.echo("\nPipeline complete!")
            if output_paths:
                typer.echo("Output USD files:")
                for p in output_paths:
                    typer.echo(f"  {p}")
            if render_paths:
                typer.echo("Rendered images:")
                for p in render_paths:
                    typer.echo(f"  {p}")
            manifest_path = context.get("artifacts_manifest_path")
            if manifest_path:
                typer.echo("Artifact manifest:")
                typer.echo(f"  {manifest_path}")

    except Exception as e:
        logger.error("Pipeline failed: %s", e)
        if verbose:
            logger.exception("Full traceback:")
        raise typer.Exit(1) from e


@app.command()
def plan(
    config: Path = typer.Argument(..., help="Path to the pipeline config YAML"),
    discovery_mode: str | None = typer.Option(
        None,
        "--discovery-mode",
        help="Planning scope: effective_bound, explicit, or all_authored.",
    ),
    unit_mode: str | None = typer.Option(
        None,
        "--unit-mode",
        help="Planning unit mode: per_material, per_group, or per_prim.",
    ),
    explicit_material_paths: str | None = typer.Option(
        None,
        "--material-paths",
        help="Comma-separated absolute material paths for explicit discovery.",
    ),
    explicit_prim_paths: str | None = typer.Option(
        None,
        "--prim-paths",
        help="Comma-separated absolute prim/subset paths for explicit discovery.",
    ),
    operator_override_cap: int | None = typer.Option(
        None,
        "--operator-override-cap",
        min=1,
        max=64,
        help="Intentional texture-unit cap override, never greater than 64.",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
) -> None:
    """Discover the scene and write texture_plan.json without backend work."""
    _setup_logging(verbose)
    logger = logging.getLogger(__name__)

    from texture_agent.config.unified_config import config_to_context, load_config
    from texture_agent.workflows.factory import run_pipeline

    try:
        cfg = load_config(config)
        _apply_planning_overrides(
            cfg,
            discovery_mode=discovery_mode,
            unit_mode=unit_mode,
            operator_override_cap=operator_override_cap,
            explicit_material_paths=explicit_material_paths,
            explicit_prim_paths=explicit_prim_paths,
            plan_only=True,
        )
        context = config_to_context(cfg)
        context = run_pipeline(
            context,
            only=["discover_materials", "plan_textures"],
        )
        texture_plan = context["texture_plan"]
        typer.echo(f"Texture plan: {context['texture_plan_path']}")
        typer.echo(f"Decision: {texture_plan.decision.state}")
        typer.echo(
            "Selected units: "
            f"{texture_plan.counts.selected_unit_count} "
            f"(effective cap {texture_plan.limits.effective_cap}, "
            f"hard cap {texture_plan.limits.hard_cap})"
        )
        for reason in texture_plan.decision.reasons:
            typer.echo(f"Reason: {reason}")
        for action in texture_plan.decision.recommended_actions:
            typer.echo(f"Action: {action}")
    except Exception as e:
        logger.error("Planning failed: %s", e)
        if verbose:
            logger.exception("Full traceback:")
        raise typer.Exit(1) from e


@app.command()
def discover(
    config: Path = typer.Argument(..., help="Path to the pipeline config YAML"),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
) -> None:
    """Discover and list materials in the input USD."""
    _setup_logging(verbose)
    logger = logging.getLogger(__name__)

    from texture_agent.config.unified_config import config_to_context, load_config
    from texture_agent.tasks import DiscoverMaterialsTask

    try:
        cfg = load_config(config)
        context = config_to_context(cfg)

        task = DiscoverMaterialsTask()
        context = task.run(context)

        materials = context.get("discovered_materials", [])
        typer.echo(f"\nDiscovered {len(materials)} materials:\n")
        typer.echo(f"{'Name':<30} {'Base Color':<25} {'Texture':<8} {'Prims':<6}")
        typer.echo("-" * 69)
        for m in materials:
            color_str = (
                f"({m.base_color[0]:.2f}, {m.base_color[1]:.2f}, {m.base_color[2]:.2f})"
            )
            typer.echo(
                f"{m.name:<30} {color_str:<25} "
                f"{'yes' if m.has_existing_texture else 'no':<8} "
                f"{len(m.bound_prim_paths):<6}"
            )

    except Exception as e:
        logger.error("Discover failed: %s", e)
        if verbose:
            logger.exception("Full traceback:")
        raise typer.Exit(1) from e


@app.command()
def generate(
    config: Path = typer.Argument(..., help="Path to the pipeline config YAML"),
    detail_policy: str | None = typer.Option(
        None,
        "--detail-policy",
        help="Override texture detail policy: default or surface_only.",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
) -> None:
    """Generate and blend textures (without applying to USD)."""
    _setup_logging(verbose)
    logger = logging.getLogger(__name__)

    from texture_agent.config.unified_config import config_to_context, load_config
    from texture_agent.workflows.factory import run_pipeline

    try:
        cfg = load_config(config)
        _apply_detail_policy_override(cfg, detail_policy)
        context = config_to_context(cfg)

        context = run_pipeline(
            context,
            only=[
                "discover_materials",
                "plan_textures",
                "generate_prompts",
                "generate_textures",
                "blend_textures",
            ],
        )

        blended = context.get("blended_textures", {})
        typer.echo(f"\nGenerated and blended {len(blended)} textures:")
        for name, path in blended.items():
            typer.echo(f"  {name}: {path}")

    except Exception as e:
        logger.error("Generate failed: %s", e)
        if verbose:
            logger.exception("Full traceback:")
        raise typer.Exit(1) from e


@app.command("apply")
def apply_cmd(
    config: Path = typer.Argument(..., help="Path to the pipeline config YAML"),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
) -> None:
    """Apply textures to USD (assumes textures already generated)."""
    _setup_logging(verbose)
    logger = logging.getLogger(__name__)

    from texture_agent.config.unified_config import config_to_context, load_config
    from texture_agent.workflows.factory import run_pipeline

    try:
        cfg = load_config(config)
        context = config_to_context(cfg)
        context["resume"] = True
        context["cached_apply_only"] = True

        context = run_pipeline(
            context,
            only=[
                "prepare_uvs",
                "discover_materials",
                "generate_prompts",
                "apply_textures",
            ],
        )

        output_paths = context.get("output_usd_paths", [])
        typer.echo(f"\nApplied textures to {len(output_paths)} USD file(s):")
        for p in output_paths:
            typer.echo(f"  {p}")

    except Exception as e:
        logger.error("Apply failed: %s", e)
        if verbose:
            logger.exception("Full traceback:")
        raise typer.Exit(1) from e
