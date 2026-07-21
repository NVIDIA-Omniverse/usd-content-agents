# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI commands for the large-scene multi-asset pipeline.

Provides analyze, extract, run-agent, collect, and run subcommands under
`material-agent scene`.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Annotated

import typer
import yaml
from rich.console import Console
from rich.table import Table

from .manifest import SceneManifest

logger = logging.getLogger(__name__)
console = Console()

scene_app = typer.Typer(
    name="scene",
    help="Process large USD scenes with multiple sub-assets.",
)

# Per-asset pipeline steps in execution order (used by --from-step)
_ASSET_PIPELINE_STEPS = [
    "optimize_usd",
    "render_preview",
    "identify_asset",
    "generate_reference_image",
    "build_dataset_usd",
    "build_dataset_prepare_dataset",
    "predict",
    "validate_predictions",
    "harmonize_predictions",
    "apply",
    "restore_usd",
    "render",
]


def _steps_before(step_name: str) -> list[str]:
    """Return all pipeline step names that come before *step_name*.

    Raises typer.BadParameter if *step_name* is not a known step.
    """
    if step_name not in _ASSET_PIPELINE_STEPS:
        raise typer.BadParameter(
            f"Unknown step '{step_name}'. "
            f"Valid steps: {', '.join(_ASSET_PIPELINE_STEPS)}"
        )
    idx = _ASSET_PIPELINE_STEPS.index(step_name)
    return _ASSET_PIPELINE_STEPS[:idx]


def _load_scene_config(config: Path) -> dict:
    """Load and validate a scene config YAML."""
    if not config.exists():
        console.print(f"[red]Config file not found:[/red] {config}")
        raise typer.Exit(1)
    with open(config) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise typer.BadParameter(
            f"Scene config must be a YAML mapping: {config}",
            param_hint="config",
        )
    return data


def _get_working_dir(scene_config: dict, config_path: Path) -> Path:
    """Derive working directory from scene config."""
    project = scene_config.get("project", {})
    session_id = project.get("session_id", project.get("name", "scene"))
    config_dir = config_path.parent
    return config_dir / f".{session_id}_scene"


def _get_manifest_path(working_dir: Path) -> Path:
    """Get the manifest file path within working directory."""
    return working_dir / "manifest.json"


def _parse_assets_filter(assets: str | None) -> list[str] | None:
    """Parse comma-separated assets filter string."""
    if not assets:
        return None
    return [a.strip() for a in assets.split(",") if a.strip()]


def _resolve_usd_path(scene_config: dict, config_path: Path) -> Path:
    """Resolve the input USD path from config (relative to config dir)."""
    usd_path_str = scene_config.get("input", {}).get("usd_path", "")
    if not usd_path_str:
        console.print("[red]No input.usd_path in config[/red]")
        raise typer.Exit(1)
    usd_path = Path(usd_path_str)
    if not usd_path.is_absolute():
        usd_path = (config_path.parent / usd_path).resolve()
    if not usd_path.exists():
        console.print(f"[red]USD file not found:[/red] {usd_path}")
        raise typer.Exit(1)
    return usd_path


def _resolve_material_library_yaml(
    scene_config: dict, config_path: Path
) -> Path | None:
    """Resolve the material library YAML path from the scene config.

    The materials section can reference an external YAML file via
    ``materials.path``, or include entries inline.  This function
    resolves the external file path relative to the config directory.

    Returns:
        Resolved Path to the materials YAML, or None if not configured.
    """
    materials_section = scene_config.get("materials", {})
    mat_path_str = materials_section.get("path")
    if not mat_path_str:
        return None

    mat_path = Path(mat_path_str)
    if not mat_path.is_absolute():
        mat_path = (config_path.parent / mat_path).resolve()
    return mat_path


def _setup_logging(verbose: bool) -> None:
    """Configure logging level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _print_manifest_summary(manifest: SceneManifest) -> None:
    """Print a Rich table summarizing the manifest."""
    table = Table(title="Sub-Assets")
    table.add_column("ID", style="dim")
    table.add_column("Name", style="bold")
    table.add_column("Prim Path")
    table.add_column("Meshes", justify="right")
    table.add_column("Vertices", justify="right")
    table.add_column("Instance Group")
    table.add_column("Status")

    status_colors = {
        "pending": "yellow",
        "extracted": "cyan",
        "completed": "green",
        "failed": "red",
        "skipped": "dim",
    }

    for sa in manifest.sub_assets:
        color = status_colors.get(sa.status, "white")
        table.add_row(
            sa.id,
            sa.name,
            sa.prim_path,
            str(sa.mesh_count),
            str(sa.vertex_count),
            sa.instance_group or "",
            f"[{color}]{sa.status}[/{color}]",
        )

    console.print(table)

    if manifest.instance_groups:
        ig_table = Table(title="Instance Groups")
        ig_table.add_column("Group", style="bold")
        ig_table.add_column("Members", justify="right")
        ig_table.add_column("Representative")
        for ig in manifest.instance_groups:
            ig_table.add_row(
                ig.group_name,
                str(ig.instance_count),
                ig.representative_id or "",
            )
        console.print(ig_table)

    if manifest.payload_groups:
        pg_table = Table(title="Payload Groups")
        pg_table.add_column("ID", style="dim")
        pg_table.add_column("Name", style="bold")
        pg_table.add_column("Payload File")
        pg_table.add_column("Instances", justify="right")
        pg_table.add_column("Status")

        for pg in manifest.payload_groups:
            color = status_colors.get(pg.status, "white")
            # Truncate payload file path for display
            payload_display = Path(pg.payload_file).name
            pg_table.add_row(
                pg.id,
                pg.group_name,
                payload_display,
                str(pg.instance_count),
                f"[{color}]{pg.status}[/{color}]",
            )
        console.print(pg_table)


def _print_validation_stats(working_dir: Path) -> None:
    """Aggregate and print validation correction stats across all assets."""
    import json

    reports = list(working_dir.glob("configs/.*/predictions/validate_report.json"))
    if not reports:
        return

    totals = {
        "total": 0,
        "valid": 0,
        "auto_corrected": 0,
        "llm_repaired": 0,
        "failed": 0,
        "no_material": 0,
    }
    corrections: list[tuple[str, str, str]] = []  # (type, old, new)

    for rp in reports:
        try:
            r = json.loads(rp.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        stats = r.get("stats", {})
        for k in totals:
            totals[k] += stats.get(k, 0)
        for c in r.get("auto_corrected", []):
            corrections.append(("auto", c.get("old", "?"), c.get("new", "?")))
        for c in r.get("llm_repaired", []):
            corrections.append(("llm", c.get("old", "?"), c.get("new", "?")))
        for c in r.get("failed", []):
            corrections.append(("failed", c.get("name", "?"), ""))

    fixed = totals["auto_corrected"] + totals["llm_repaired"]
    console.print(
        f"\n  Validation: {totals['total']} predictions checked, "
        f"{totals['valid']} valid, "
        f"[bold]{fixed} corrected[/bold], "
        f"{totals['failed']} unfixable"
    )
    if corrections:
        for typ, old, new in corrections:
            if typ == "failed":
                console.print(f"    [red]UNFIXED:[/red] '{old}'")
            else:
                label = "auto" if typ == "auto" else "LLM"
                console.print(f"    [yellow]{label}:[/yellow] '{old}' -> '{new}'")


@scene_app.command()
def analyze(
    config: Annotated[Path, typer.Argument(help="Path to scene config YAML")],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output manifest path (default: auto)"),
    ] = None,
    no_llm: Annotated[
        bool,
        typer.Option("--no-llm", help="Skip LLM-based split refinement"),
    ] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Verbose output")
    ] = False,
) -> None:
    """Analyze a USD scene and detect sub-assets."""
    _setup_logging(verbose)

    scene_config = _load_scene_config(config)
    usd_path = _resolve_usd_path(scene_config, config)
    working_dir = _get_working_dir(scene_config, config)
    from world_understanding.utils.token_tracking import TokenTracker

    scene_token_tracker = TokenTracker()
    working_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output or _get_manifest_path(working_dir)

    console.print(f"[bold]Analyzing scene:[/bold] {usd_path}")

    # Get analysis options from scene config
    scene_section = scene_config.get("scene", {})
    analyze_opts = scene_section.get("analyze", {})
    filters = scene_section.get("filters", {})

    # LLM config for split refinement (from scene.analyze.llm section)
    llm_config = None
    if not no_llm:
        llm_section = analyze_opts.get("llm")
        if llm_section:
            llm_config = llm_section
        else:
            # Default LLM config if not explicitly configured
            llm_config = {
                "backend": "nim",
                "model": "qwen/qwen3.5-397b-a17b",
                "temperature": 0.1,
                "max_tokens": 256,
            }

    from .analyze import analyze_scene

    manifest = analyze_scene(
        scene_usd_path=usd_path,
        skip_geometry=analyze_opts.get("skip_geometry", False),
        building_block_min_reuse=analyze_opts.get("building_block_min_reuse", 20),
        filters=filters,
        llm_config=llm_config,
        token_tracker=scene_token_tracker,
    )

    manifest.save(manifest_path)
    _print_manifest_summary(manifest)

    console.print(f"\n[green]Manifest saved to:[/green] {manifest_path}")
    processable = manifest.get_processable_assets()
    deduped = len(manifest.sub_assets) - len(processable)
    console.print(
        f"Detected [bold]{len(manifest.sub_assets)}[/bold] sub-assets, "
        f"[bold]{len(manifest.instance_groups)}[/bold] instance groups\n"
        f"  [green]{len(processable)}[/green] unique sub-assets need processing, "
        f"[dim]{deduped} skipped via dedup[/dim]"
    )


@scene_app.command()
def extract(
    config: Annotated[Path, typer.Argument(help="Path to scene config YAML")],
    assets: Annotated[
        str | None,
        typer.Option("--assets", "-a", help="Comma-separated asset names to process"),
    ] = None,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Verbose output")
    ] = False,
) -> None:
    """Extract sub-assets and generate per-asset configs."""
    _setup_logging(verbose)

    scene_config = _load_scene_config(config)
    usd_path = _resolve_usd_path(scene_config, config)
    working_dir = _get_working_dir(scene_config, config)
    manifest_path = _get_manifest_path(working_dir)

    if not manifest_path.exists():
        console.print("[red]Manifest not found. Run 'scene analyze' first.[/red]")
        raise typer.Exit(1)

    manifest = SceneManifest.load(manifest_path)
    names_filter = _parse_assets_filter(assets)

    # Extract sub-assets
    scene_section = scene_config.get("scene", {})
    extract_opts = scene_section.get("extract", {})
    flatten = extract_opts.get("flatten", True)
    extract_workers = extract_opts.get("max_workers", 1)

    console.print(f"[bold]Extracting sub-assets from:[/bold] {usd_path}")

    from .extract import extract_all

    extracted_dir = working_dir / "extracted"
    manifest = extract_all(
        scene_usd_path=usd_path,
        manifest=manifest,
        output_dir=extracted_dir,
        names_filter=names_filter,
        flatten=flatten,
        max_workers=extract_workers,
    )

    # Generate per-asset configs
    console.print("[bold]Generating per-asset configs...[/bold]")

    from .config_gen import generate_all_configs, generate_all_payload_configs

    configs_dir = working_dir / "configs"
    manifest = generate_all_configs(
        manifest=manifest,
        scene_config=scene_config,
        configs_dir=configs_dir,
        scene_config_dir=config.parent.resolve(),
        names_filter=names_filter,
    )

    # Generate per-payload configs
    if manifest.payload_groups:
        console.print("[bold]Generating per-payload configs...[/bold]")
        manifest = generate_all_payload_configs(
            manifest=manifest,
            scene_config=scene_config,
            configs_dir=configs_dir,
            scene_config_dir=config.parent.resolve(),
        )

    manifest.save(manifest_path)
    _print_manifest_summary(manifest)

    console.print(
        f"\n[green]Extraction complete.[/green] Manifest updated: {manifest_path}"
    )


@scene_app.command(name="run-agent")
def run_agent_cmd(
    config: Annotated[Path, typer.Argument(help="Path to scene config YAML")],
    assets: Annotated[
        str | None,
        typer.Option("--assets", "-a", help="Comma-separated asset names to process"),
    ] = None,
    skip: Annotated[
        str | None,
        typer.Option("--skip", help="Comma-separated pipeline steps to skip"),
    ] = None,
    only: Annotated[
        str | None,
        typer.Option(
            "--only", help="Comma-separated pipeline steps to run exclusively"
        ),
    ] = None,
    from_step: Annotated[
        str | None,
        typer.Option(
            "--from-step",
            help="Resume from this step, skipping all earlier steps (e.g. 'predict')",
        ),
    ] = None,
    skip_existing: Annotated[
        bool,
        typer.Option("--skip-existing", help="Skip already completed assets"),
    ] = False,
    workers: Annotated[
        int,
        typer.Option("--workers", "-w", help="Number of parallel workers (default: 1)"),
    ] = 1,
    simulate: Annotated[
        bool,
        typer.Option(
            "--simulate",
            help="Skip rendering/VLM; use mock predictions (round-robin materials)",
        ),
    ] = False,
    simulate_mock_analyze: Annotated[
        bool,
        typer.Option(
            "--simulate-mock-analyze",
            help="Also mock the scene analyze LLM (default: keep real)",
        ),
    ] = False,
    predict_max_workers: Annotated[
        int | None,
        typer.Option(
            "--predict-max-workers",
            help="Override per-asset predict step max_workers",
        ),
    ] = None,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Verbose output")
    ] = False,
) -> None:
    """Run material-agent on each sub-asset (step 3 only)."""
    _setup_logging(verbose)

    scene_config = _load_scene_config(config)
    source_scene_config = scene_config

    # Simulate mode: patch all backends to "mock"
    if simulate:
        from material_agent.api.simulate_config import patch_config_for_simulate

        scene_config = patch_config_for_simulate(
            scene_config, mock_analyze=simulate_mock_analyze
        )
        console.print("[yellow]Simulate mode: all backends patched to 'mock'[/yellow]")

    working_dir = _get_working_dir(scene_config, config)
    manifest_path = _get_manifest_path(working_dir)

    if not manifest_path.exists():
        console.print("[red]Manifest not found. Run 'scene analyze' first.[/red]")
        raise typer.Exit(1)

    manifest = SceneManifest.load(manifest_path)
    names_filter = _parse_assets_filter(assets)
    skip_steps = [s.strip() for s in skip.split(",")] if skip else None
    only_steps = [s.strip() for s in only.split(",")] if only else None

    # --from-step: skip all steps before the given step and use resume mode
    resume = False
    if from_step:
        before = _steps_before(from_step)
        skip_steps = list(set((skip_steps or []) + before))
        resume = True
        console.print(
            f"  [yellow]Resuming from step '{from_step}', "
            f"skipping: {', '.join(before)}[/yellow]"
        )
        # Reset completed assets so they get reprocessed from the given step
        reset_count = 0
        for sa in manifest.sub_assets:
            if sa.status == "completed":
                sa.status = "extracted"
                reset_count += 1
        if reset_count:
            console.print(
                f"  [yellow]Reset {reset_count} completed assets to "
                f"'extracted' for reprocessing[/yellow]"
            )
            manifest.save(manifest_path)

    # Load material names for simulate mode
    material_names: list[str] | None = None
    if simulate:
        from .simulate import load_material_names_from_config

        material_names = load_material_names_from_config(scene_config, config)
        console.print(
            f"[yellow]Simulate mode:[/yellow] using {len(material_names)} "
            f"materials for mock predictions"
        )

    processable = manifest.get_processable_assets(names_filter)
    console.print(f"[bold]Running pipeline for {len(processable)} sub-assets[/bold]")

    from .run import run_all, run_all_payloads_bottomup

    manifest = run_all(
        manifest=manifest,
        manifest_path=manifest_path,
        names_filter=names_filter,
        skip_steps=skip_steps,
        only_steps=only_steps,
        skip_existing=skip_existing,
        max_workers=workers,
        verbose=verbose,
        simulate=simulate,
        material_names=material_names,
        resume=resume,
        from_step=from_step,
        predict_max_workers=predict_max_workers,
        scene_config=source_scene_config,
        scene_config_dir=config.parent.resolve(),
    )

    # Run payload pipelines (bottom-up by depth)
    if manifest.payload_groups:
        payloads_by_depth = manifest.get_payloads_by_depth()
        max_depth = max(payloads_by_depth.keys()) if payloads_by_depth else 0
        total_payloads = sum(len(v) for v in payloads_by_depth.values())
        console.print(
            f"\n[bold]Running pipeline for {total_payloads} payload groups "
            f"(max depth={max_depth})[/bold]"
        )
        configs_dir = _get_working_dir(scene_config, config) / "configs"
        manifest = run_all_payloads_bottomup(
            manifest=manifest,
            manifest_path=manifest_path,
            scene_config=source_scene_config,
            configs_dir=configs_dir,
            scene_config_dir=config.parent.resolve(),
            skip_steps=skip_steps,
            only_steps=only_steps,
            skip_existing=skip_existing,
            max_workers=workers,
            verbose=verbose,
            simulate=simulate,
            material_names=material_names,
            resume=resume,
            from_step=from_step,
            predict_max_workers=predict_max_workers,
        )

    manifest.save(manifest_path)
    _print_manifest_summary(manifest)

    completed = sum(1 for sa in manifest.sub_assets if sa.status == "completed")
    failed = sum(1 for sa in manifest.sub_assets if sa.status == "failed")
    pg_completed = sum(1 for pg in manifest.payload_groups if pg.status == "completed")
    pg_failed = sum(1 for pg in manifest.payload_groups if pg.status == "failed")
    console.print(
        f"\n[green]Run complete:[/green] "
        f"sub-assets: {completed} completed, {failed} failed"
    )
    if manifest.payload_groups:
        console.print(f"  payloads: {pg_completed} completed, {pg_failed} failed")


@scene_app.command()
def collect(
    config: Annotated[Path, typer.Argument(help="Path to scene config YAML")],
    assets: Annotated[
        str | None,
        typer.Option("--assets", "-a", help="Comma-separated asset names to include"),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output USD path (default: auto)"),
    ] = None,
    no_render: Annotated[
        bool,
        typer.Option("--no-render", help="Skip rendering the composed scene"),
    ] = False,
    clear_materials: Annotated[
        bool,
        typer.Option(
            "--clear-materials",
            help="Clear original material bindings before rendering",
        ),
    ] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Verbose output")
    ] = False,
) -> None:
    """Apply unified materials and compose onto the original scene, then render."""
    _setup_logging(verbose)

    scene_config = _load_scene_config(config)
    usd_path = _resolve_usd_path(scene_config, config)
    working_dir = _get_working_dir(scene_config, config)
    manifest_path = _get_manifest_path(working_dir)

    if not manifest_path.exists():
        console.print("[red]Manifest not found. Run 'scene analyze' first.[/red]")
        raise typer.Exit(1)

    manifest = SceneManifest.load(manifest_path)
    names_filter = _parse_assets_filter(assets)

    output_path = output or (working_dir / "output" / "composed_scene.usd")

    # Resolve material library YAML from config
    material_library_yaml = _resolve_material_library_yaml(scene_config, config)
    if not material_library_yaml or not material_library_yaml.exists():
        console.print(
            "[red]Material library YAML not found. "
            "Check materials.path in config.[/red]"
        )
        raise typer.Exit(1)

    # Run cross-asset harmonize before collect (simple mode by default)
    harmonize_config = scene_config.get("scene", {}).get("harmonize", {})
    if harmonize_config.get("enabled", True):
        from .harmonize import harmonize_scene_predictions

        harmonize_llm = harmonize_config.get("llm", {})
        if not harmonize_llm:
            harmonize_llm = scene_config.get("scene", {}).get("reconcile", {}).get(
                "llm", {}
            ) or scene_config.get("scene", {}).get("analyze", {}).get("llm", {})

        harmonize_mode = harmonize_config.get("mode")
        if not harmonize_mode:
            harmonize_mode = "full" if harmonize_llm else "simple"

        console.print(
            f"[bold]Harmonizing predictions across sub-assets "
            f"(mode={harmonize_mode})...[/bold]"
        )
        remap = harmonize_scene_predictions(
            manifest=manifest,
            llm_config=harmonize_llm if harmonize_mode == "full" else None,
            mode=harmonize_mode,
        )
        if remap:
            console.print(f"  [green]Harmonized {len(remap)} predictions[/green]")
        else:
            console.print("  [dim]No cross-asset conflicts[/dim]")

    console.print(f"[bold]Applying materials and composing into:[/bold] {output_path}")

    from .collect import apply_and_compose

    apply_and_compose(
        scene_usd_path=usd_path,
        manifest=manifest,
        output_usd_path=output_path,
        material_library_yaml=material_library_yaml,
        names_filter=names_filter,
    )

    console.print(f"\n[green]Composed scene saved to:[/green] {output_path}")

    # Render the composed scene
    render_config = scene_config.get("steps", {}).get("render", {})
    render_enabled = render_config.get("enabled", True)
    if not no_render and render_enabled:
        console.print("\n[bold]Rendering composed scene...[/bold]")

        from .collect import render_composed_scene

        # Get render config from scene config steps section
        image_width = render_config.get("image_width", 1024)
        image_height = render_config.get("image_height", 1024)
        camera_corners = render_config.get("camera_corners", ["+x+y+z", "-x-y-z"])
        camera_margin = render_config.get("camera_margin", 1.0)
        bg = render_config.get("background_color", [1.0, 1.0, 1.0])

        rendered = render_composed_scene(
            composed_usd_path=output_path,
            output_dir=output_path.parent,
            camera_corners=camera_corners,
            image_width=image_width,
            image_height=image_height,
            camera_margin=camera_margin,
            background_color=tuple(bg),
            clear_materials=clear_materials,
        )

        if rendered:
            console.print(f"[green]Rendered {len(rendered)} images:[/green]")
            for p in rendered:
                console.print(f"  {p}")
        else:
            console.print("[yellow]No renders produced.[/yellow]")


@scene_app.command()
def validate(
    config: Annotated[Path, typer.Argument(help="Path to scene config YAML")],
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Verbose output")
    ] = False,
) -> None:
    """Validate material bindings for a scene pipeline output."""
    _setup_logging(verbose)

    scene_config = _load_scene_config(config)
    working_dir = _get_working_dir(scene_config, config)
    manifest_path = _get_manifest_path(working_dir)

    if not manifest_path.exists():
        console.print("[red]Manifest not found. Run 'scene analyze' first.[/red]")
        raise typer.Exit(1)

    exit_code = _run_validation(config, verbose)
    raise typer.Exit(exit_code)


def _run_validation(config: Path, verbose: bool) -> int:
    """Run scene validation and print results. Returns exit code (0=pass, 1=fail)."""
    from .validate import format_asset_report, validate_scene

    console.print(f"\n[bold]Validating scene:[/bold] {config}")
    report = validate_scene(config, verbose)

    passed = sum(1 for a in report.assets if a.ok and a.status != "inherited")
    inherited = sum(1 for a in report.assets if a.status == "inherited")
    failed = sum(1 for a in report.assets if not a.ok)
    total = len(report.assets)

    for r in report.assets:
        for line in format_asset_report(r, verbose):
            console.print(line)

    if report.payloads:
        console.print(f"\n  {'─' * 40}")
        console.print("  Payload Groups:")
        for pr in report.payloads:
            status_icon = "PASS" if pr.ok else "FAIL"
            summary = (
                f"  [{status_icon}] {pr.name:35s} "
                f"d={pr.depth}  "
                f"preds={pr.predictions_count:>5d}  "
                f"inst={pr.instance_count:>5d}  "
                f"out={'yes' if pr.has_output_usd else 'no':>3s}"
            )
            console.print(summary)
            if verbose or not pr.ok:
                for e in pr.errors:
                    console.print(f"         ERROR: {e}")
                for w in pr.warnings:
                    console.print(f"         WARN:  {w}")

    console.print(f"\n{'=' * 80}")
    pg_passed = sum(1 for p in report.payloads if p.ok)
    pg_failed = sum(1 for p in report.payloads if not p.ok)
    pg_total = len(report.payloads)
    inherited_msg = f", {inherited} inherited" if inherited else ""
    console.print(
        f"  RESULTS: {passed}/{total} assets passed{inherited_msg}, {failed} failed"
    )
    if pg_total > 0:
        console.print(f"  PAYLOADS: {pg_passed}/{pg_total} passed, {pg_failed} failed")
    console.print(f"  Total bindings (layer): {report.total_bindings}")
    console.print(f"  Total de-instanced: {report.total_deinstanced}")
    if report.composed_scene_path:
        console.print(f"  Composed scene: {report.composed_scene_path}")
        console.print(
            f"    Bindings: our={report.composed_our}  "
            f"old={report.composed_old}  none={report.composed_none}"
        )
        if report.composed_instances_checked > 0:
            console.print(
                f"    Instance propagation: "
                f"{report.composed_instance_our}/{report.composed_instances_checked} "
                f"instance proxies have our materials"
            )
        if report.composed_subsets_checked > 0:
            console.print(
                f"    GeomSubset coverage: "
                f"{report.composed_subset_our}/{report.composed_subsets_checked} "
                f"subsets have our materials"
            )
    console.print(f"{'=' * 80}")

    scene_config = _load_scene_config(config)
    working_dir = _get_working_dir(scene_config, config)
    _print_validation_stats(working_dir)

    for e in report.errors:
        console.print(f"  SCENE ERROR: {e}")
    for w in report.warnings:
        console.print(f"  SCENE WARN:  {w}")

    any_failures = failed > 0 or pg_failed > 0 or report.errors
    return 1 if any_failures else 0


@scene_app.command(name="run")
def run_cmd(
    config: Annotated[Path, typer.Argument(help="Path to scene config YAML")],
    assets: Annotated[
        str | None,
        typer.Option("--assets", "-a", help="Comma-separated asset names to process"),
    ] = None,
    skip: Annotated[
        str | None,
        typer.Option("--skip", help="Comma-separated pipeline steps to skip"),
    ] = None,
    only: Annotated[
        str | None,
        typer.Option(
            "--only", help="Comma-separated pipeline steps to run exclusively"
        ),
    ] = None,
    from_step: Annotated[
        str | None,
        typer.Option(
            "--from-step",
            help=(
                "Resume from this per-asset pipeline step, reusing existing "
                "analysis/extraction/dataset (e.g. 'predict')"
            ),
        ),
    ] = None,
    workers: Annotated[
        int,
        typer.Option("--workers", "-w", help="Number of parallel workers (default: 1)"),
    ] = 1,
    skip_existing: Annotated[
        bool,
        typer.Option("--skip-existing", help="Skip already completed assets"),
    ] = False,
    simulate: Annotated[
        bool,
        typer.Option(
            "--simulate",
            help="Skip rendering/VLM; use mock predictions (round-robin materials)",
        ),
    ] = False,
    simulate_mock_analyze: Annotated[
        bool,
        typer.Option(
            "--simulate-mock-analyze",
            help=(
                "Also mock the scene analyze LLM (faster but worse decomposition). "
                "By default --simulate keeps the real analyze LLM."
            ),
        ),
    ] = False,
    clear_materials: Annotated[
        bool,
        typer.Option(
            "--clear-materials",
            help="Clear original material bindings before rendering",
        ),
    ] = False,
    no_render: Annotated[
        bool,
        typer.Option("--no-render", help="Skip rendering the composed scene"),
    ] = False,
    fail_on_validation: Annotated[
        bool,
        typer.Option(
            "--fail-on-validation/--warn-on-validation",
            help=(
                "Exit nonzero when composed-scene validation fails. "
                "Use --warn-on-validation to keep legacy warning-only behavior."
            ),
        ),
    ] = True,
    resume: Annotated[
        bool,
        typer.Option(
            "--resume",
            help=(
                "Resume from existing state: skip analyze/extract if their "
                "outputs already exist, and let per-asset pipelines resume "
                "via their own checkpoint files"
            ),
        ),
    ] = False,
    predict_max_workers: Annotated[
        int | None,
        typer.Option(
            "--predict-max-workers",
            help=(
                "Override per-asset predict step max_workers. "
                "Lower this when running many assets in parallel to avoid "
                "rate limits (e.g. 16 with --workers 16 = 256 concurrent VLM calls)"
            ),
        ),
    ] = None,
    clean: Annotated[
        bool,
        typer.Option("--clean", help="Delete the working directory before starting"),
    ] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Verbose output")
    ] = False,
) -> None:
    """Run the full scene pipeline end-to-end (analyze -> extract -> pipeline -> collect -> validate)."""
    _setup_logging(verbose)

    skip_steps = [s.strip() for s in skip.split(",") if s.strip()] if skip else []
    only_steps = [s.strip() for s in only.split(",") if s.strip()] if only else []
    asset_filter = _parse_assets_filter(assets) or []

    if from_step:
        before = _steps_before(from_step)
        console.print(
            f"[yellow]--from-step '{from_step}': skipping per-asset steps "
            f"{', '.join(before)}[/yellow]"
        )

    if simulate:
        if simulate_mock_analyze:
            console.print(
                "[yellow]Simulate mode: all backends patched to 'mock' "
                "(including analyze LLM)[/yellow]"
            )
        else:
            console.print(
                "[yellow]Simulate mode: all backends patched to 'mock' "
                "(analyze LLM kept real)[/yellow]"
            )

    from material_agent.api import ScenePipelineInput, run_scene_pipeline

    result = run_scene_pipeline(
        ScenePipelineInput(
            config=config,
            assets=asset_filter,
            skip_steps=skip_steps,
            only_steps=only_steps,
            from_step=from_step,
            skip_existing=skip_existing,
            max_workers=workers,
            resume=resume,
            clean=clean,
            no_render=no_render,
            clear_materials=clear_materials,
            validate_output=True,
            fail_on_validation_error=fail_on_validation,
            simulate=simulate,
            simulate_mock_analyze=simulate_mock_analyze,
            predict_max_workers=predict_max_workers,
            verbose=verbose,
        )
    )

    if not result.success:
        console.print(f"[red]Scene pipeline failed:[/red] {result.error}")
        raise typer.Exit(1)

    if result.output_usd_path:
        console.print(
            f"[green]Composed scene saved to:[/green] {result.output_usd_path}"
        )
    if result.rendered_images:
        console.print(f"[green]Rendered {len(result.rendered_images)} images:[/green]")
        for image_path in result.rendered_images:
            console.print(f"  {image_path}")
    if result.stats_report_path:
        console.print(f"[green]Scene stats report:[/green] {result.stats_report_path}")
    console.print(
        f"[green]Sub-assets:[/green] {result.completed_assets} completed, "
        f"{result.failed_assets} failed"
    )
    if result.completed_payloads or result.failed_payloads:
        console.print(
            f"[green]Payloads:[/green] {result.completed_payloads} completed, "
            f"{result.failed_payloads} failed"
        )
    if result.validation_passed is not None:
        status = "passed" if result.validation_passed else "found issues"
        color = "green" if result.validation_passed else "red"
        console.print(f"[{color}]Validation {status}[/{color}]")
    for warning in result.warnings:
        console.print(f"[yellow]{warning}[/yellow]")

    console.print("\n[bold green]Scene pipeline complete![/bold green]")


def _run_cmd_legacy(
    config: Annotated[Path, typer.Argument(help="Path to scene config YAML")],
    assets: Annotated[
        str | None,
        typer.Option("--assets", "-a", help="Comma-separated asset names to process"),
    ] = None,
    skip: Annotated[
        str | None,
        typer.Option("--skip", help="Comma-separated pipeline steps to skip"),
    ] = None,
    only: Annotated[
        str | None,
        typer.Option(
            "--only", help="Comma-separated pipeline steps to run exclusively"
        ),
    ] = None,
    from_step: Annotated[
        str | None,
        typer.Option(
            "--from-step",
            help=(
                "Resume from this per-asset pipeline step, reusing existing "
                "analysis/extraction/dataset (e.g. 'predict')"
            ),
        ),
    ] = None,
    workers: Annotated[
        int,
        typer.Option("--workers", "-w", help="Number of parallel workers (default: 1)"),
    ] = 1,
    skip_existing: Annotated[
        bool,
        typer.Option("--skip-existing", help="Skip already completed assets"),
    ] = False,
    simulate: Annotated[
        bool,
        typer.Option(
            "--simulate",
            help="Skip rendering/VLM; use mock predictions (round-robin materials)",
        ),
    ] = False,
    simulate_mock_analyze: Annotated[
        bool,
        typer.Option(
            "--simulate-mock-analyze",
            help=(
                "Also mock the scene analyze LLM (faster but worse decomposition). "
                "By default --simulate keeps the real analyze LLM."
            ),
        ),
    ] = False,
    clear_materials: Annotated[
        bool,
        typer.Option(
            "--clear-materials",
            help="Clear original material bindings before rendering",
        ),
    ] = False,
    no_render: Annotated[
        bool,
        typer.Option("--no-render", help="Skip rendering the composed scene"),
    ] = False,
    fail_on_validation: Annotated[
        bool,
        typer.Option(
            "--fail-on-validation/--warn-on-validation",
            help=(
                "Exit nonzero when composed-scene validation fails. "
                "Use --warn-on-validation to keep legacy warning-only behavior."
            ),
        ),
    ] = True,
    resume: Annotated[
        bool,
        typer.Option(
            "--resume",
            help=(
                "Resume from existing state: skip analyze/extract if their "
                "outputs already exist, and let per-asset pipelines resume "
                "via their own checkpoint files"
            ),
        ),
    ] = False,
    predict_max_workers: Annotated[
        int | None,
        typer.Option(
            "--predict-max-workers",
            help=(
                "Override per-asset predict step max_workers. "
                "Lower this when running many assets in parallel to avoid "
                "rate limits (e.g. 16 with --workers 16 = 256 concurrent VLM calls)"
            ),
        ),
    ] = None,
    clean: Annotated[
        bool,
        typer.Option("--clean", help="Delete the working directory before starting"),
    ] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Verbose output")
    ] = False,
) -> None:
    """Run the full scene pipeline end-to-end (analyze → extract → pipeline → collect → validate)."""
    _setup_logging(verbose)

    scene_config = _load_scene_config(config)

    # Simulate mode: patch all backends to "mock" in the scene config
    if simulate:
        from material_agent.api.simulate_config import patch_config_for_simulate

        scene_config = patch_config_for_simulate(
            scene_config, mock_analyze=simulate_mock_analyze
        )
        if simulate_mock_analyze:
            console.print(
                "[yellow]Simulate mode: all backends patched to 'mock' "
                "(including analyze LLM)[/yellow]"
            )
        else:
            console.print(
                "[yellow]Simulate mode: all backends patched to 'mock' "
                "(analyze LLM kept real)[/yellow]"
            )

    usd_path = _resolve_usd_path(scene_config, config)
    working_dir = _get_working_dir(scene_config, config)
    from world_understanding.utils.token_tracking import TokenTracker

    scene_token_tracker = TokenTracker()

    if clean and working_dir.exists():
        console.print(f"[yellow]Cleaning working directory: {working_dir}[/yellow]")
        shutil.rmtree(working_dir)

    working_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = _get_manifest_path(working_dir)
    names_filter = _parse_assets_filter(assets)

    # --from-step: compute skip list and decide whether to skip scene-level steps
    skip_steps = [s.strip() for s in skip.split(",")] if skip else None
    only_steps = [s.strip() for s in only.split(",")] if only else None

    if from_step:
        before = _steps_before(from_step)
        skip_steps = list(set((skip_steps or []) + before))
        resume = True
        console.print(
            f"[yellow]--from-step '{from_step}': skipping per-asset steps "
            f"{', '.join(before)}[/yellow]"
        )

    if resume:
        # Check if there is existing state to resume from
        has_manifest = manifest_path.exists()
        extracted_dir_check = working_dir / "extracted"
        has_extracted = extracted_dir_check.exists() and any(
            extracted_dir_check.iterdir()
        )
        if has_manifest or has_extracted:
            console.print(
                "[yellow]Resume mode:[/yellow] will reuse existing "
                "analyze/extract outputs where available"
            )
        else:
            console.print(
                "[yellow]Resume mode:[/yellow] no existing state found, "
                "running full pipeline from scratch"
            )

    # --- Step 1: Analyze ---
    # Skip if resuming and manifest already exists
    if resume and manifest_path.exists():
        console.print(
            "\n[bold blue]Step 1/5:[/bold blue] Analyzing scene... "
            "[dim](skipped — reusing existing manifest)[/dim]"
        )
        manifest = SceneManifest.load(manifest_path)
        console.print(
            f"  [green]Loaded {len(manifest.sub_assets)} sub-assets, "
            f"{len(manifest.instance_groups)} instance groups[/green]"
        )
    else:
        console.print("\n[bold blue]Step 1/5:[/bold blue] Analyzing scene...")
        scene_section = scene_config.get("scene", {})
        analyze_opts = scene_section.get("analyze", {})
        filters = scene_section.get("filters", {})

        # LLM config for split refinement
        llm_section = analyze_opts.get("llm")
        llm_config = llm_section or {
            "backend": "nim",
            "model": "qwen/qwen3.5-397b-a17b",
            "temperature": 0.1,
            "max_tokens": 256,
        }

        from .analyze import analyze_scene

        manifest = analyze_scene(
            scene_usd_path=usd_path,
            skip_geometry=analyze_opts.get("skip_geometry", False),
            building_block_min_reuse=analyze_opts.get("building_block_min_reuse", 20),
            filters=filters,
            llm_config=llm_config,
            token_tracker=scene_token_tracker,
        )
        manifest.save(manifest_path)
        processable = manifest.get_processable_assets()
        deduped = len(manifest.sub_assets) - len(processable)
        console.print(
            f"  [green]Detected {len(manifest.sub_assets)} sub-assets, "
            f"{len(manifest.instance_groups)} instance groups[/green]\n"
            f"  [green]{len(processable)}[/green] unique sub-assets need processing, "
            f"[dim]{deduped} skipped via dedup[/dim]"
        )

    # --- Step 2: Extract + config gen ---
    # Skip if resuming and extracted dir already exists
    scene_section = scene_config.get("scene", {})
    extracted_dir = working_dir / "extracted"
    configs_dir = working_dir / "configs"

    if resume and extracted_dir.exists() and configs_dir.exists():
        console.print(
            "\n[bold blue]Step 2/5:[/bold blue] Extracting sub-assets... "
            "[dim](skipped — reusing existing extraction)[/dim]"
        )
    else:
        console.print("\n[bold blue]Step 2/5:[/bold blue] Extracting sub-assets...")
        extract_opts = scene_section.get("extract", {})
        flatten = extract_opts.get("flatten", True)
        extract_workers = extract_opts.get("max_workers", 1)

        from .extract import extract_all

        manifest = extract_all(
            scene_usd_path=usd_path,
            manifest=manifest,
            output_dir=extracted_dir,
            names_filter=names_filter,
            flatten=flatten,
            max_workers=extract_workers,
        )

        from .config_gen import generate_all_configs, generate_all_payload_configs

        manifest = generate_all_configs(
            manifest=manifest,
            scene_config=scene_config,
            configs_dir=configs_dir,
            scene_config_dir=config.parent.resolve(),
            names_filter=names_filter,
        )

        # Generate per-payload configs
        if manifest.payload_groups:
            console.print("  Generating per-payload configs...")
            manifest = generate_all_payload_configs(
                manifest=manifest,
                scene_config=scene_config,
                configs_dir=configs_dir,
                scene_config_dir=config.parent.resolve(),
            )

        manifest.save(manifest_path)
        console.print("  [green]Extraction and config generation complete[/green]")

    # Reset completed assets when --from-step is used so they get reprocessed.
    # Plain --resume does NOT reset completed assets; per-asset pipelines
    # handle their own resumption via .pipeline_state.json.
    if from_step:
        reset_count = 0
        for sa in manifest.sub_assets:
            if sa.status == "completed":
                sa.status = "extracted"
                reset_count += 1
        for pg in manifest.payload_groups:
            if pg.status == "completed":
                pg.status = "pending"
                reset_count += 1
        if reset_count:
            console.print(
                f"  [yellow]Reset {reset_count} completed assets/payloads "
                f"for reprocessing from step '{from_step}'[/yellow]"
            )
            manifest.save(manifest_path)

    # --- Step 3: Pipeline ---
    console.print(
        "\n[bold blue]Step 3/5:[/bold blue] Running material-agent pipeline..."
    )

    # Load material names for simulate mode
    material_names: list[str] | None = None
    if simulate:
        from .simulate import load_material_names_from_config

        material_names = load_material_names_from_config(scene_config, config)
        console.print(
            f"  [yellow]Simulate mode:[/yellow] using {len(material_names)} "
            f"materials for mock predictions"
        )

    processable = manifest.get_processable_assets(names_filter)
    console.print(f"  Processing {len(processable)} sub-assets...")

    from .run import run_all, run_all_payloads_bottomup

    manifest = run_all(
        manifest=manifest,
        manifest_path=manifest_path,
        names_filter=names_filter,
        skip_steps=skip_steps,
        only_steps=only_steps,
        skip_existing=skip_existing,
        max_workers=workers,
        verbose=verbose,
        simulate=simulate,
        material_names=material_names,
        resume=resume,
        from_step=from_step,
        predict_max_workers=predict_max_workers,
        scene_config=scene_config,
        scene_config_dir=config.parent.resolve(),
    )

    # Run payload pipelines (bottom-up by depth)
    if manifest.payload_groups:
        payloads_by_depth = manifest.get_payloads_by_depth()
        max_depth = max(payloads_by_depth.keys()) if payloads_by_depth else 0
        total_payloads = sum(len(v) for v in payloads_by_depth.values())
        console.print(
            f"  Processing {total_payloads} payload groups (max depth={max_depth})..."
        )
        manifest = run_all_payloads_bottomup(
            manifest=manifest,
            manifest_path=manifest_path,
            scene_config=scene_config,
            configs_dir=configs_dir,
            scene_config_dir=config.parent.resolve(),
            skip_steps=skip_steps,
            only_steps=only_steps,
            skip_existing=skip_existing,
            max_workers=workers,
            verbose=verbose,
            simulate=simulate,
            material_names=material_names,
            resume=resume,
            from_step=from_step,
            predict_max_workers=predict_max_workers,
        )

    manifest.save(manifest_path)

    completed = sum(1 for sa in manifest.sub_assets if sa.status == "completed")
    failed = sum(1 for sa in manifest.sub_assets if sa.status == "failed")
    console.print(f"  [green]{completed} completed, {failed} failed[/green]")
    if manifest.payload_groups:
        pg_completed = sum(
            1 for pg in manifest.payload_groups if pg.status == "completed"
        )
        pg_failed = sum(1 for pg in manifest.payload_groups if pg.status == "failed")
        console.print(
            f"  [green]Payloads: {pg_completed} completed, {pg_failed} failed[/green]"
        )

    # --- Step 3.5: Reconcile predictions (optional) ---
    reconcile_config = scene_config.get("scene", {}).get("reconcile")
    if reconcile_config and reconcile_config.get("enabled", True):
        console.print("\n[bold blue]Step 3.5/5:[/bold blue] Reconciling predictions...")
        from .reconcile import apply_remapping, reconcile_predictions

        reconcile_llm = reconcile_config.get("llm", {})
        if not reconcile_llm:
            # Fall back to analyze LLM config
            reconcile_llm = (
                scene_config.get("scene", {}).get("analyze", {}).get("llm", {})
            )

        # Load material names for context
        mat_names: list[str] | None = None
        mat_yaml = _resolve_material_library_yaml(scene_config, config)
        if mat_yaml and mat_yaml.exists():
            import yaml

            with open(mat_yaml) as f:
                mat_data = yaml.safe_load(f)
            mat_names = [m.get("name", "") for m in mat_data.get("entries", [])]

        remap = reconcile_predictions(
            manifest=manifest,
            llm_config=reconcile_llm,
            materials_list=mat_names,
            token_tracker=scene_token_tracker,
        )
        if remap:
            updated = apply_remapping(manifest, remap)
            console.print(
                f"  [green]Reconciled {updated} predictions "
                f"({len(remap)} remappings)[/green]"
            )
        else:
            console.print("  [dim]No reconciliation needed[/dim]")

    # --- Step 3.6: Harmonize predictions across sub-assets ---
    harmonize_config = scene_config.get("scene", {}).get("harmonize", {})
    if harmonize_config.get("enabled", True):
        from .harmonize import harmonize_scene_predictions

        harmonize_llm = harmonize_config.get("llm", {})
        if not harmonize_llm:
            # Fall back to reconcile or analyze LLM config
            harmonize_llm = scene_config.get("scene", {}).get("reconcile", {}).get(
                "llm", {}
            ) or scene_config.get("scene", {}).get("analyze", {}).get("llm", {})

        # Auto-detect mode: "full" when LLM config is available, "simple" otherwise
        harmonize_mode = harmonize_config.get("mode")
        if not harmonize_mode:
            harmonize_mode = "full" if harmonize_llm else "simple"

        console.print(
            f"\n[bold blue]Step 3.6/5:[/bold blue] Harmonizing predictions "
            f"across sub-assets (mode={harmonize_mode})..."
        )

        remap = harmonize_scene_predictions(
            manifest=manifest,
            llm_config=harmonize_llm if harmonize_mode == "full" else None,
            mode=harmonize_mode,
            token_tracker=scene_token_tracker,
        )
        if remap:
            console.print(
                f"  [green]Harmonized {len(remap)} predictions "
                f"across sub-assets[/green]"
            )
        else:
            console.print("  [dim]No cross-asset conflicts to harmonize[/dim]")

    # --- Step 4: Collect + render ---
    console.print(
        "\n[bold blue]Step 4/5:[/bold blue] Applying materials and composing..."
    )
    output_path = working_dir / "output" / "composed_scene.usd"

    material_library_yaml = _resolve_material_library_yaml(scene_config, config)
    if not material_library_yaml or not material_library_yaml.exists():
        console.print(
            "[red]Material library YAML not found. "
            "Check materials.path in config.[/red]"
        )
        raise typer.Exit(1)

    from .collect import apply_and_compose

    apply_and_compose(
        scene_usd_path=usd_path,
        manifest=manifest,
        output_usd_path=output_path,
        material_library_yaml=material_library_yaml,
        names_filter=names_filter,
    )
    console.print(f"  [green]Composed scene saved to:[/green] {output_path}")

    render_config = scene_config.get("steps", {}).get("render", {})
    render_enabled = render_config.get("enabled", True)
    if not no_render and render_enabled:
        console.print("\n[bold]Rendering composed scene...[/bold]")

        from .collect import render_composed_scene

        image_width = render_config.get("image_width", 1024)
        image_height = render_config.get("image_height", 1024)
        camera_corners = render_config.get("camera_corners", ["+x+y+z", "-x-y-z"])
        camera_margin = render_config.get("camera_margin", 1.0)
        bg = render_config.get("background_color", [1.0, 1.0, 1.0])

        rendered = render_composed_scene(
            composed_usd_path=output_path,
            output_dir=output_path.parent,
            camera_corners=camera_corners,
            image_width=image_width,
            image_height=image_height,
            camera_margin=camera_margin,
            background_color=tuple(bg),
            clear_materials=clear_materials,
        )

        if rendered:
            console.print(f"[green]Rendered {len(rendered)} images:[/green]")
            for p in rendered:
                console.print(f"  {p}")
        else:
            console.print("[yellow]No renders produced.[/yellow]")

    _print_manifest_summary(manifest)
    _print_validation_stats(working_dir)
    from .stats import write_scene_stats_report

    stats_report_path = write_scene_stats_report(
        manifest=manifest,
        working_dir=working_dir,
        output_dir=output_path.parent,
        output_usd_path=output_path,
        scene_operation_token_stats=scene_token_tracker.get_stats(),
    )
    console.print(f"  [green]Scene stats report:[/green] {stats_report_path}")

    # --- Step 5: Validate ---
    console.print("\n[bold blue]Step 5/5:[/bold blue] Validating scene output...")
    validation_exit = _run_validation(config, verbose)
    if validation_exit == 0:
        console.print("  [green]Validation passed[/green]")
    else:
        if fail_on_validation:
            console.print("  [red]Validation failed (see above)[/red]")
            raise typer.Exit(validation_exit)
        console.print("  [yellow]Validation failed (see above)[/yellow]")

    console.print("\n[bold green]Scene pipeline complete![/bold green]")


# ============================================================================
# bundle
# ============================================================================


@scene_app.command()
def bundle(
    config: Annotated[Path, typer.Argument(help="Path to scene config YAML")],
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output", "-o", help="Output directory (default: <working>/output/bundle)"
        ),
    ] = None,
    format: Annotated[
        str,
        typer.Option(
            "--format",
            "-f",
            help="Output USD format: usdc (binary, smaller) or usda (text)",
        ),
    ] = "usdc",
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Verbose output")
    ] = False,
) -> None:
    """Bundle the composed scene into a self-contained directory.

    Creates a flattened USD with the material library copied alongside it,
    with all asset paths rewritten to be relative. The resulting directory
    can be copied to any machine with Kit for rendering — no external
    dependencies needed.
    """
    _setup_logging(verbose)
    scene_config = _load_scene_config(config)
    working_dir = _get_working_dir(scene_config, config)
    output_path = working_dir / "output" / "composed_scene.usd"

    if not output_path.exists():
        console.print(
            "[red]Composed scene not found. Run 'scene run' or 'scene collect' first.[/red]"
        )
        raise typer.Exit(1)

    # Resolve material library directory
    mat_yaml = _resolve_material_library_yaml(scene_config, config)
    if mat_yaml is None or not mat_yaml.exists():
        console.print("[red]Material library YAML not found in config.[/red]")
        raise typer.Exit(1)

    mat_lib_dir = mat_yaml.parent

    bundle_dir = output_dir or (working_dir / "output" / "bundle")
    ext = ".usdc" if format == "usdc" else ".usda"

    console.print(f"\n[bold blue]Bundling scene:[/bold blue] {output_path}")
    console.print(f"  Material library: {mat_lib_dir}")
    console.print(f"  Output: {bundle_dir}")

    from .bundle import create_bundle

    result = create_bundle(
        composed_scene_path=output_path,
        material_library_dir=mat_lib_dir,
        bundle_dir=bundle_dir,
        output_format=ext,
    )

    console.print(f"\n[bold green]Bundle created:[/bold green] {bundle_dir}")
    console.print(f"  USD: {result['usd_file'].name} ({result['usd_size_mb']:.0f} MB)")
    console.print(f"  Library: {result['library_files']} files")
    console.print(f"  Total: {result['total_size_mb']:.0f} MB")
    console.print(f"  Asset paths verified: {result['verified_paths']}")
    if result["missing_paths"]:
        console.print(
            f"  [yellow]WARNING: {result['missing_paths']} unresolved paths[/yellow]"
        )
