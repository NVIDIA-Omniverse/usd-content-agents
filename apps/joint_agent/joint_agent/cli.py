# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Joint Agent CLI interface using Typer and Rich."""

import logging
from copy import deepcopy
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml
from dotenv import load_dotenv
from rich import print
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from world_understanding.agentic.cli import (
    load_cli_config_mapping,
    normalize_cli_step_filters,
    sever_cli_exception_graph,
)
from world_understanding.agentic.config import validate_selected_model_credentials
from world_understanding.agentic.events import get_listener
from world_understanding.utils.credentials import (
    path_is_file_with_safe_diagnostics,
    redact_sensitive_config,
    redact_sensitive_path,
)
from world_understanding.utils.model_auth import (
    MODEL_AUTHENTICATION_FAILURE_MESSAGE,
    is_model_authentication_error,
    public_model_failure_message,
)

from .utils import get_version

__version__ = get_version()

# Load environment variables from .env file. The package __init__ also loads this
# before submodule imports for console-script entry points.
load_dotenv()

# Initialize Typer app and Rich console
app = typer.Typer(
    name="joint-agent",
    help="Joint Agent - LLM hierarchy analysis and VLM classification for articulated bodies",
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
)
console = Console()

_PREPARE_DATASET_FAILURE_MESSAGE = "Dataset preparation failed"
_USD_BUILD_FAILURE_MESSAGE = "USD dataset building failed"
_RIGGED_REFERENCE_FAILURE_MESSAGE = "Rigged-reference validation failed"


class _PipelineConfigShapeError(ValueError):
    """Raised for fixed, credential-safe CLI configuration diagnostics."""


def _validate_run_config_model_credentials(
    raw_config: dict[str, Any],
    config_path: Path,
    skip_steps: list[str],
    only_steps: list[str],
) -> None:
    """Fail before pipeline construction when a selected model key is absent."""
    from joint_agent.config import (
        get_step_defaults,
        normalize_analyze_structure_model_alias,
    )

    credential_config = deepcopy(raw_config)
    steps = (
        credential_config.get("steps")
        if "project" in credential_config
        else credential_config
    )
    if isinstance(steps, dict):
        analyze_config = steps.get("analyze_structure")
        if isinstance(analyze_config, dict):
            steps["analyze_structure"] = normalize_analyze_structure_model_alias(
                analyze_config
            )

    validate_selected_model_credentials(
        credential_config,
        config_path,
        skip_steps,
        only_steps,
        get_step_defaults=get_step_defaults,
        guidance_lines=(
            "Set the required provider credential, configure an endpoint-scoped "
            "api_key/api_key_env with base_url, or skip/disable the affected step.",
        ),
    )


def setup_logging(
    verbose: bool = False,
    log_file: Path | None = None,
    log_level: str = "INFO",
) -> logging.Logger:
    """Setup logging configuration with Rich handler.

    This function now delegates to the shared logging utility.

    Args:
        verbose: Enable verbose output (sets DEBUG level)
        log_file: Optional path to log file
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        Configured logger instance
    """
    from world_understanding.agentic.cli import setup_logging as shared_setup_logging

    return shared_setup_logging(
        agent_name="joint_agent",
        verbose=verbose,
        log_file=log_file,
        log_level=log_level,
    )


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        print(
            f"[bold blue]Joint Agent[/bold blue] version [green]{__version__}[/green]"
        )
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            "-V",
            help="Show version and exit",
            callback=version_callback,
            is_eager=True,
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose output (DEBUG logging)"),
    ] = False,
    log_file: Annotated[
        Path | None,
        typer.Option("--log-file", help="Path to log file"),
    ] = None,
    log_level: Annotated[
        str,
        typer.Option(
            "--log-level",
            help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
        ),
    ] = "INFO",
) -> None:
    """
    Joint Agent - LLM hierarchy analysis and VLM classification for articulated bodies.

    Use [bold]joint-agent --help[/bold] to see available commands.
    """
    # Setup logging
    logger = setup_logging(verbose=verbose, log_file=log_file, log_level=log_level)

    # Store logger in app context for use in commands
    if not hasattr(app, "state"):
        app.state = {}
    app.state["logger"] = logger
    app.state["verbose"] = verbose

    if verbose:
        logger.debug("Verbose mode enabled")
        logger.debug(f"Log level: {log_level}")
        if log_file:
            logger.debug(f"Logging to file: {log_file}")


@app.command()
@sever_cli_exception_graph
def predict(
    config: Annotated[
        Path,
        typer.Argument(
            help="Path to unified YAML configuration file",
        ),
    ],
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose output (DEBUG logging)"),
    ] = False,
    log_file: Annotated[
        Path | None,
        typer.Option("--log-file", help="Path to log file"),
    ] = None,
    log_level: Annotated[
        str,
        typer.Option(
            "--log-level",
            help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
        ),
    ] = "INFO",
) -> None:
    """
    Run asset classification predictions on a dataset.

    This is equivalent to: joint-agent run CONFIG --only predict

    Uses the unified configuration format where all paths are auto-derived from
    project.working_dir. The predict step will run VLM inference to classify assets.

    Example usage:
    ```bash
    joint-agent predict apps/joint_agent/configs/byoa_joint_rigger.yaml
    ```

    Output:
    - {working_dir}/predictions/predictions.jsonl: Classification predictions with reasoning
    - {working_dir}/predictions/report.html: HTML report with visualizations
    """
    # This is just an alias for: run --only predict
    return run(
        config=config,
        skip=None,
        only="predict",
        session_id=None,
        resume=False,
        dry_run=False,
        clean=False,
        verbose=verbose,
        log_file=log_file,
        log_level=log_level,
    )


# Create a sub-app for build-dataset commands
build_dataset_app = typer.Typer(
    name="build-dataset",
    help="Commands for building datasets from various sources",
    rich_markup_mode="rich",
)

# Add build-dataset as a command group to the main app
app.add_typer(build_dataset_app, name="build-dataset")


@build_dataset_app.command(name="prepare-dataset")
@sever_cli_exception_graph
def prepare_dataset(
    config: Annotated[
        Path,
        typer.Argument(
            help="Path to YAML configuration file",
        ),
    ],
    dataset: Annotated[
        Path | None,
        typer.Option(
            "--dataset",
            "-d",
            help="Override dataset path from config",
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose output (DEBUG logging)"),
    ] = False,
    log_file: Annotated[
        Path | None,
        typer.Option("--log-file", help="Path to log file"),
    ] = None,
    log_level: Annotated[
        str,
        typer.Option(
            "--log-level",
            help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
        ),
    ] = "INFO",
) -> None:
    """
    Prepare dataset for classification or prediction.

    This command prepares datasets by organizing data entries
    with images and metadata for VLM classification.

    Example usage:
    ```bash
    # Using config file
    joint-agent build-dataset prepare-dataset path/to/prepare_dataset.yaml

    # Override dataset path
    joint-agent build-dataset prepare-dataset path/to/prepare_dataset.yaml \\
      --dataset ./data/custom
    ```
    """
    # Setup logging for this command
    logger = setup_logging(verbose=verbose, log_file=log_file, log_level=log_level)

    logger.info("Starting prepare dataset workflow")

    # Check if config exists
    safe_config = redact_sensitive_path(config)
    if not config.exists():
        logger.error("Configuration file not found: %s", safe_config)
        console.print(f"[red]Error:[/red] Configuration file not found: {safe_config}")
        raise typer.Exit(1)

    # Display configuration info
    console.print(
        Panel.fit(
            "[bold]Prepare Dataset[/bold]\n\n"
            f"Configuration: {safe_config}\n"
            f"Dataset Override: {redact_sensitive_path(dataset)}\n"
            f"Output: dataset.jsonl saved to dataset directory\n"
            f"Verbose mode: {'ON' if verbose else 'OFF'}",
            border_style="blue",
        )
    )

    logger.info("Configuration file: %s", safe_config)
    if dataset:
        logger.info("Dataset override: %s", redact_sensitive_path(dataset))

    if verbose:
        logger.debug("Verbose mode enabled - detailed logging active")

    # Use API for dataset preparation
    from joint_agent.api import (
        BuildDatasetPrepareDatasetInput,
        build_dataset_prepare_dataset,
    )

    try:
        # Create API parameters
        api_params = BuildDatasetPrepareDatasetInput(
            config=config,
            dataset_override=dataset,
            verbose=verbose,
        )

        # Run the workflow via API
        logger.info("Running prepare dataset workflow...")
        console.print("\n[cyan]Loading config and preparing dataset...[/cyan]")

        result = build_dataset_prepare_dataset(api_params)

        # Check if workflow completed successfully
        if result.success:
            dataset_entries = result.dataset_entries
            failed_models = result.failed_models
            dataset_jsonl_path = result.dataset_jsonl_path

            console.print(
                "\n[bold green]✨ Dataset preparation completed![/bold green]"
            )
            console.print(f"  • Dataset entries: {len(dataset_entries)}")
            console.print(f"  • Failed entries: {len(failed_models)}")
            console.print(
                f"  • Dataset saved to: {redact_sensitive_path(dataset_jsonl_path)}"
            )

            if failed_models:
                safe_failed_models = [
                    str(redact_sensitive_config(name)) for name in failed_models
                ]
                console.print(
                    f"[yellow]Failed entries: {', '.join(safe_failed_models)}[/yellow]"
                )
                logger.info("Failed entries: %s", safe_failed_models)
        else:
            logger.error(_PREPARE_DATASET_FAILURE_MESSAGE)
            console.print(f"[red]Error:[/red] {_PREPARE_DATASET_FAILURE_MESSAGE}")
            raise typer.Exit(1)

    except Exception:
        logger.error(_PREPARE_DATASET_FAILURE_MESSAGE)
        console.print(f"\n[red]Error:[/red] {_PREPARE_DATASET_FAILURE_MESSAGE}")
        raise typer.Exit(1) from None


@build_dataset_app.command(name="usd")
@sever_cli_exception_graph
def usd(
    config: Annotated[
        Path,
        typer.Argument(
            help="Path to the data preparation configuration file.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ],
    source: Annotated[
        Path | None,
        typer.Option(
            "--source",
            "-s",
            help="Path to the USD file or directory (overrides config).",
            exists=False,
            resolve_path=True,
        ),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output directory for dataset (overrides config).",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = None,
    extract_metadata: Annotated[
        bool,
        typer.Option(
            "--extract-metadata/--no-extract-metadata",
            help="Extract prim metadata (materials, transforms, etc.).",
        ),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose output (DEBUG logging)"),
    ] = False,
    log_file: Annotated[
        Path | None,
        typer.Option("--log-file", help="Path to log file"),
    ] = None,
    log_level: Annotated[
        str,
        typer.Option(
            "--log-level",
            help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
        ),
    ] = "INFO",
) -> None:
    """
    Build a dataset from USD file(s) by rendering views of each prim.

    This command will intelligently handle both single file and batch processing:
    - If config has 'usd_path': processes a single USD file
    - If config has 'usd_dir': processes all USD files in that directory

    For batch processing, subdirectories will be created for each USD file.

    Example usage:
    ```bash
    # Single file config (with usd_path)
    joint-agent build-dataset usd path/to/single_usd.yaml

    # Batch processing config (with usd_dir)
    joint-agent build-dataset usd path/to/usd_batch.yaml

    # Override source (file or directory)
    joint-agent build-dataset usd path/to/data_prep.yaml \\
        --source path/to/file_or_dir

    # With metadata extraction
    joint-agent build-dataset usd path/to/data_prep.yaml \\
        --extract-metadata
    ```
    """
    # Setup logging
    logger = setup_logging(verbose=verbose, log_file=log_file, log_level=log_level)

    logger.info("Starting Joint Agent Dataset Build Workflow")

    # Check if config exists
    if not config.exists():
        logger.error("Configuration file not found: %s", redact_sensitive_path(config))
        raise typer.Exit(code=1)

    # Load config to determine if it's single file or batch processing
    try:
        with open(config, encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
    except Exception:
        logger.error("Failed to load configuration")
        raise typer.Exit(code=1) from None

    # Determine if source override points to a directory or file
    is_batch_mode = False
    if source:
        source = Path(source)
        if source.is_dir():
            is_batch_mode = True
    elif "usd_dir" in config_data:
        is_batch_mode = True
    elif "usd_path" not in config_data:
        # Neither usd_path nor usd_dir specified
        logger.error(
            "Configuration must contain either 'usd_path' (for single file) "
            "or 'usd_dir' (for batch processing)"
        )
        raise typer.Exit(code=1)

    # Handle batch processing
    if is_batch_mode:
        logger.info("Detected batch processing mode")

        # Get USD directory
        if source and source.is_dir():
            usd_dir = source
            logger.info(
                "Using USD directory override: %s", redact_sensitive_path(usd_dir)
            )
        elif "usd_dir" in config_data:
            # Resolve path relative to config file location
            config_dir = config.parent
            usd_dir = config_dir / Path(config_data["usd_dir"])
            usd_dir = usd_dir.resolve()
            logger.info("Using usd_dir from config: %s", redact_sensitive_path(usd_dir))
        else:  # pragma: no cover - guarded by is_batch_mode detection above.
            logger.error("Batch mode requires usd_dir in config or --source directory")
            raise typer.Exit(code=1)

        # Get output directory
        if output_dir:
            batch_output_dir = output_dir
        elif "output_dir" in config_data:
            config_dir = config.parent
            batch_output_dir = config_dir / Path(config_data["output_dir"])
            batch_output_dir = batch_output_dir.resolve()
        else:
            batch_output_dir = Path("output")

        # Check if USD directory exists
        if not usd_dir.exists():
            logger.error("USD directory not found: %s", redact_sensitive_path(usd_dir))
            raise typer.Exit(code=1)

        # Use API for batch processing
        from joint_agent.api import BuildDatasetUsdInput, build_dataset_usd

        try:
            # Create API parameters
            api_params = BuildDatasetUsdInput(
                config=config,
                source_override=usd_dir,
                output_dir_override=batch_output_dir,
                extract_metadata=extract_metadata,
                verbose=verbose,
            )

            # Run via API
            api_result = build_dataset_usd(api_params)

            if not api_result.success:
                raise RuntimeError(_USD_BUILD_FAILURE_MESSAGE)

            results = api_result.batch_results
            successful_builds = sum(
                1 for r in results.values() if r.get("status") == "success"
            )
            failed_builds = sum(
                1 for r in results.values() if r.get("status") != "success"
            )

        except Exception:
            logger.error(_USD_BUILD_FAILURE_MESSAGE)
            console.print(f"[red]Error:[/red] {_USD_BUILD_FAILURE_MESSAGE}")
            raise typer.Exit(code=1) from None

        # Display batch results
        table = Table(title="Batch Dataset Build Results", show_header=True)
        table.add_column("USD File", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Prims", justify="right")
        table.add_column("Images", justify="right")
        table.add_column("Output Directory", style="dim")

        for usd_name, result in results.items():
            status = "✓ Success" if result["status"] == "success" else "✗ Failed"
            status_style = "green" if result["status"] == "success" else "red"

            prims = str(result.get("num_prims", "N/A"))
            images = str(result.get("num_images", "N/A"))
            output_path = redact_sensitive_path(Path(result["output_dir"]).name)

            table.add_row(
                redact_sensitive_path(usd_name),
                f"[{status_style}]{status}[/{status_style}]",
                prims,
                images,
                output_path,
            )

        console.print("\n")
        console.print(table)
        console.print("\n")

        if failed_builds == 0:
            console.print(
                Panel.fit(
                    "[bold green]✓[/bold green] All datasets built successfully!",
                    border_style="green",
                )
            )
        elif successful_builds > 0:
            console.print(
                Panel.fit(
                    f"[bold yellow]⚠[/bold yellow] Completed with {failed_builds} failures",
                    border_style="yellow",
                )
            )
        else:
            console.print(
                Panel.fit(
                    "[bold red]✗[/bold red] All builds failed",
                    border_style="red",
                )
            )
            raise typer.Exit(code=1)

    else:
        # Single file processing
        logger.info("Processing single USD file")

        try:
            from joint_agent.api import BuildDatasetUsdInput, build_dataset_usd

            # Create API parameters
            api_params = BuildDatasetUsdInput(
                config=config,
                source_override=source,
                output_dir_override=output_dir,
                extract_metadata=extract_metadata,
                verbose=verbose,
            )

            # Run workflow via API
            logger.info("Executing dataset build workflow")
            result = build_dataset_usd(api_params)

            if not result.success:
                raise RuntimeError(_USD_BUILD_FAILURE_MESSAGE)

            # Create results table
            table = Table(title="Dataset Build Results", show_header=True)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            table.add_row(
                "Dataset Manifest", redact_sensitive_path(result.dataset_path or "N/A")
            )
            table.add_row("Total Prims", str(result.num_prims))
            table.add_row("Total Images", str(result.num_images))

            console.print("\n")
            console.print(table)
            console.print("\n")

            console.print(
                Panel.fit(
                    "[bold green]✓[/bold green] Dataset build completed successfully!",
                    border_style="green",
                )
            )

        except Exception:
            logger.error(_USD_BUILD_FAILURE_MESSAGE)
            console.print(f"[red]Error:[/red] {_USD_BUILD_FAILURE_MESSAGE}")
            raise typer.Exit(code=1) from None


@app.command()
@sever_cli_exception_graph
def run(
    config: Annotated[
        Path,
        typer.Argument(
            help="Path to unified YAML configuration file",
        ),
    ],
    skip: Annotated[
        str | None,
        typer.Option(
            "--skip",
            help="Comma-separated list of steps to skip",
        ),
    ] = None,
    only: Annotated[
        str | None,
        typer.Option(
            "--only",
            help="Comma-separated list of steps to run exclusively",
        ),
    ] = None,
    session_id: Annotated[
        str | None,
        typer.Option(
            "--session-id",
            help="Reuse existing session ID instead of generating a new one",
        ),
    ] = None,
    resume: Annotated[
        bool,
        typer.Option(
            "--resume",
            help="Resume from last successful checkpoint",
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Show pipeline plan without executing",
        ),
    ] = False,
    clean: Annotated[
        bool,
        typer.Option(
            "--clean",
            help="Clean (delete) working directory before starting",
        ),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose output (DEBUG logging)"),
    ] = False,
    log_file: Annotated[
        Path | None,
        typer.Option("--log-file", help="Path to log file"),
    ] = None,
    log_level: Annotated[
        str,
        typer.Option(
            "--log-level",
            help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
        ),
    ] = "INFO",
) -> None:
    """
    Execute the joint-agent pipeline for articulated body classification.

    The pipeline always includes LLM hierarchy analysis (analyze_structure)
    for accurate part naming of robot arms and other articulated mechanisms,
    followed by VLM-based material classification.

    Example usage:
    ```bash
    # Prepare a public bring-your-own-asset config, then edit input.usd_path
    cp apps/joint_agent/configs/byoa_joint_rigger.yaml my_joint_asset.yaml

    # Dry run to see the execution plan
    joint-agent run my_joint_asset.yaml --dry-run

    # Run the configured pipeline
    joint-agent run my_joint_asset.yaml

    # Resume after a transient failure
    joint-agent run my_joint_asset.yaml --resume
    ```
    """
    # Setup logging for this command
    logger = setup_logging(verbose=verbose, log_file=log_file, log_level=log_level)

    # Get event listener for CLI output
    listener = get_listener({}, logger_name="joint_agent.cli")

    logger.info("Starting Joint Agent Pipeline")

    # Check the raw runtime path without exposing it through OS diagnostics.
    if not path_is_file_with_safe_diagnostics(
        config,
        label="pipeline configuration file",
    ):
        safe_config = redact_sensitive_path(config)
        logger.error("Pipeline configuration file not found: %s", safe_config)
        console.print(
            f"[red]Error:[/red] Pipeline configuration file not found: {safe_config}"
        )
        raise typer.Exit(1)

    try:
        from joint_agent.config.schema import STEP_ORDER

        skip_steps, only_steps = normalize_cli_step_filters(
            skip=skip,
            only=only,
            valid_steps=STEP_ORDER,
        )
    except ValueError as error:
        logger.error("Pipeline step filter validation failed: %s", error)
        console.print(f"[red]Error:[/red] {error}")
        raise typer.Exit(1) from None

    # Always ensure analyze_structure is enabled for joint-agent. Keep the
    # override in memory so inline credentials from the source config are never
    # copied to a crash-survivable temporary YAML file.
    source_config_path = config
    config_data: dict[str, Any] = {}
    try:
        try:
            config_data = load_cli_config_mapping(config)

            steps = config_data.setdefault("steps", {})
            if not isinstance(steps, dict):
                raise _PipelineConfigShapeError(
                    "Pipeline configuration 'steps' must be a mapping"
                )
            analyze = steps.setdefault("analyze_structure", {})
            if not isinstance(analyze, dict):
                raise _PipelineConfigShapeError(
                    "Pipeline configuration 'steps.analyze_structure' must be a mapping"
                )
            if analyze.get("enabled") is None:
                analyze["enabled"] = True
        except _PipelineConfigShapeError as error:
            logger.error(str(error))
            console.print(f"[red]Error:[/red] {error}")
            raise typer.Exit(1) from None
        except (OSError, ValueError) as error:
            logger.error(str(error))
            console.print(f"[red]Error:[/red] {error}")
            raise typer.Exit(1) from None
        except Exception:
            # Parser and filesystem diagnostics can contain configuration
            # fragments. Keep the user-visible failure credential-agnostic.
            message = "Unable to load pipeline configuration"
            logger.error(message)
            console.print(f"[red]Error:[/red] {message}")
            raise typer.Exit(1) from None

        if not dry_run:
            try:
                _validate_run_config_model_credentials(
                    config_data, source_config_path, skip_steps, only_steps
                )
            except ValueError as error:
                logger.error("Pipeline configuration validation failed: %s", error)
                console.print(f"[red]Error:[/red] {error}")
                raise typer.Exit(1) from None

        # Display configuration info via event system
        listener.event(
            "pipeline.config.display",
            {
                "config": redact_sensitive_path(config),
                "skip_steps": skip_steps,
                "only_steps": only_steps,
                "resume": resume,
                "dry_run": dry_run,
                "clean": clean,
            },
        )

        if dry_run:
            # Display the in-memory override without persisting it.
            try:
                projected_config = redact_sensitive_config(config_data)
                pipeline_config = (
                    projected_config if isinstance(projected_config, dict) else {}
                )

                console.print("\n[bold cyan]Pipeline Execution Plan:[/bold cyan]\n")

                # Detect config format (unified vs old)
                is_unified = "project" in pipeline_config

                if is_unified:
                    # Unified config format
                    project_name = pipeline_config.get("project", {}).get(
                        "name", "unknown"
                    )
                    working_dir = pipeline_config.get("project", {}).get(
                        "working_dir", f".{project_name}"
                    )

                    console.print(f"[cyan]Project:[/cyan] {project_name}")
                    console.print(f"[cyan]Working Directory:[/cyan] {working_dir}")
                    console.print(
                        f"[cyan]Input USD:[/cyan] {pipeline_config.get('input', {}).get('usd_path', 'N/A')}"
                    )

                    steps_section = pipeline_config.get("steps", {})
                else:
                    # Old config format
                    steps_section = pipeline_config

                # Use centralized step names
                from joint_agent.api.defaults import PIPELINE_STEP_NAMES

                step_names = PIPELINE_STEP_NAMES

                table = Table(title="Steps", show_header=True)
                table.add_column("Step", style="cyan")
                table.add_column("Status", style="yellow")
                table.add_column("Enabled", style="green")

                for step in step_names:
                    if step not in steps_section:
                        continue

                    step_config = steps_section[step]

                    # Check if enabled (for unified format)
                    if is_unified:
                        enabled = step_config.get("enabled")
                        if enabled is None:
                            # Implicitly enable if step has any configuration
                            has_config = any(k != "enabled" for k in step_config.keys())
                            enabled = has_config
                        if not enabled:
                            continue

                    if skip_steps and step in skip_steps:
                        status = "⊘ Skipped"
                        style_name = "dim"
                    elif only_steps and step not in only_steps:
                        status = "⊘ Excluded"
                        style_name = "dim"
                    else:
                        status = "→ Will Run"
                        style_name = "green"

                    enabled = "Yes" if step_config.get("enabled", True) else "No"

                    table.add_row(
                        f"[{style_name}]{step}[/{style_name}]",
                        f"[{style_name}]{status}[/{style_name}]",
                        f"[{style_name}]{enabled}[/{style_name}]",
                    )

                console.print(table)
                console.print("\n[bold green]✓ Dry run complete[/bold green]")
                logger.info("Dry run completed successfully")
                return

            except Exception:
                message = "Unable to render pipeline execution plan"
                logger.error(message)
                console.print(f"\n[red]Error:[/red] {message}")
            # Raise after leaving the handler so the rejected exception graph
            # cannot retain credential-bearing values through ``__context__``.
            raise typer.Exit(1) from None

        # Execute unified pipeline using API
        try:
            from joint_agent.api import PipelineInput, run_pipeline

            logger.info("Creating unified pipeline workflow")

            # Create CLI event listener with Rich formatting
            from joint_agent.api import CLIEventListener

            cli_listener = CLIEventListener(
                logger=logger, console=console, show_events=False
            )

            # Create API parameters
            api_params = PipelineInput(
                config=config_data,
                source_config_path=source_config_path,
                skip_steps=skip_steps,
                only_steps=only_steps,
                session_id=session_id,
                resume=resume,
                dry_run=False,
                clean=clean,
                verbose=False,  # Logging already set up
                event_listener=cli_listener,
            )

            logger.info("Running unified pipeline workflow")
            console.print()

            result = run_pipeline(api_params)

            # Display results from a diagnostic-safe projection. The raw API
            # output remains available to programmatic runtime callers.
            if result.success:
                console.print()

                # Display summary of each step
                if result.step_results:
                    console.print("[bold cyan]Pipeline Results Summary:[/bold cyan]\n")

                    projected_results = redact_sensitive_config(result.step_results)
                    safe_results = (
                        projected_results if isinstance(projected_results, dict) else {}
                    )
                    for step_name, step_output in safe_results.items():
                        console.print(f"[green]✓[/green] {step_name}")
                        if isinstance(step_output, dict):
                            for key, value in step_output.items():
                                if value is not None:
                                    console.print(f"  • {key}: {value}")

                logger.info("Pipeline completed successfully")
            else:
                message = (
                    MODEL_AUTHENTICATION_FAILURE_MESSAGE
                    if is_model_authentication_error(result.error)
                    else "Pipeline execution failed"
                )
                logger.error(message)
                console.print(f"[red]Error:[/red] {message}")
                raise typer.Exit(1) from None

        except typer.Exit:
            raise
        except Exception as error:
            message = public_model_failure_message(error, "Pipeline execution failed")
            logger.error(message)
            console.print(f"\n[red]Error:[/red] {message}")
            if resume:
                console.print(
                    "\n[yellow]Tip:[/yellow] Pipeline checkpoint saved. "
                    "Use --resume to continue."
                )
            raise typer.Exit(1) from None

    finally:
        # Release any inline runtime credentials as soon as this CLI invocation
        # finishes. No modified configuration is written to disk.
        config_data.clear()


@app.command()
@sever_cli_exception_graph
def pipeline(
    config: Annotated[
        Path,
        typer.Argument(
            help="Path to unified YAML configuration file",
        ),
    ],
    skip: Annotated[
        str | None,
        typer.Option(
            "--skip",
            help="Comma-separated list of steps to skip",
        ),
    ] = None,
    only: Annotated[
        str | None,
        typer.Option(
            "--only",
            help="Comma-separated list of steps to run exclusively",
        ),
    ] = None,
    resume: Annotated[
        bool,
        typer.Option(
            "--resume",
            help="Resume from last successful checkpoint",
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Show pipeline plan without executing",
        ),
    ] = False,
    clean: Annotated[
        bool,
        typer.Option(
            "--clean",
            help="Clean (delete) working directory before starting",
        ),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose output (DEBUG logging)"),
    ] = False,
    log_file: Annotated[
        Path | None,
        typer.Option("--log-file", help="Path to log file"),
    ] = None,
    log_level: Annotated[
        str,
        typer.Option(
            "--log-level",
            help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
        ),
    ] = "INFO",
) -> None:
    """
    [DEPRECATED] Execute a multi-step asset agent pipeline.

    **This command is deprecated. Please use 'joint-agent run' instead.**

    This is an alias for the 'run' command and will be removed in a future version.
    """
    # Print deprecation warning
    console.print(
        "[yellow]⚠ Warning:[/yellow] The 'pipeline' command is deprecated and will be removed in a future version."
    )
    console.print("[yellow]           Please use 'joint-agent run' instead.[/yellow]\n")

    # Call the run command with the same arguments
    run(
        config=config,
        skip=skip,
        only=only,
        resume=resume,
        dry_run=dry_run,
        clean=clean,
        verbose=verbose,
        log_file=log_file,
        log_level=log_level,
    )


@app.command(name="validate-rigged-reference")
@sever_cli_exception_graph
def validate_rigged_reference(
    reference_usd: Annotated[
        Path,
        typer.Argument(
            help="Path to the authored rigged USD/USDZ reference.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ],
    articulation_candidates: Annotated[
        Path,
        typer.Argument(
            help="Path to Stage 2 articulation_candidates.json.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ],
    reference_manifest_input: Annotated[
        Path | None,
        typer.Option(
            "--reference-manifest-input",
            help=(
                "Use this pre-authored reference manifest for the comparison "
                "after validating it as a motion-joint subset of the USD. Only "
                "fixed joints may be omitted."
            ),
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            "-o",
            help="Directory for generated validation artifacts.",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = None,
    manifest_output: Annotated[
        Path | None,
        typer.Option(
            "--manifest-output",
            help="Output path for the extracted reference manifest JSON.",
            dir_okay=False,
            resolve_path=True,
        ),
    ] = None,
    validation_output: Annotated[
        Path | None,
        typer.Option(
            "--validation-output",
            help="Output path for the validation comparison JSON.",
            dir_okay=False,
            resolve_path=True,
        ),
    ] = None,
    report_output: Annotated[
        Path | None,
        typer.Option(
            "--report-output",
            help="Output path for the validation HTML report.",
            dir_okay=False,
            resolve_path=True,
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose output"),
    ] = False,
    log_file: Annotated[
        Path | None,
        typer.Option("--log-file", help="Path to log file"),
    ] = None,
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Logging level"),
    ] = "INFO",
) -> None:
    """Validate Stage 2 candidates against an authored rigged USD reference."""
    logger = setup_logging(verbose=verbose, log_file=log_file, log_level=log_level)
    try:
        from joint_agent.functions.rigged_reference_validation import (
            build_effective_reference_manifest_subset,
            compare_articulation_candidates_to_reference,
            extract_reference_articulation_manifest,
            load_json_document,
            write_json_document,
            write_rigged_reference_validation_report_html,
        )

        artifact_dir = output_dir or (
            articulation_candidates.parent / "rigged_reference_validation"
        )
        manifest_path = manifest_output or artifact_dir / "reference_manifest.json"
        validation_path = validation_output or artifact_dir / "validation.json"
        report_path = report_output or artifact_dir / "validation.html"

        logger.info(
            "Extracting reference manifest from %s",
            redact_sensitive_path(reference_usd),
        )
        extracted_manifest = extract_reference_articulation_manifest(reference_usd)
        if reference_manifest_input is None:
            manifest_document = extracted_manifest
        else:
            logger.info(
                "Using supplied reference manifest %s for %s",
                redact_sensitive_path(reference_manifest_input),
                redact_sensitive_path(reference_usd),
            )
            manifest_document = build_effective_reference_manifest_subset(
                extracted_manifest,
                load_json_document(reference_manifest_input),
                allowed_omitted_joint_types=frozenset({"fixed"}),
            )
        candidate_document = load_json_document(articulation_candidates)
        validation_document = compare_articulation_candidates_to_reference(
            manifest_document,
            candidate_document,
        )

        write_json_document(manifest_path, manifest_document)
        write_json_document(validation_path, validation_document)
        write_rigged_reference_validation_report_html(
            report_path,
            validation_document,
        )

        summary = validation_document["summary"]
        table = Table(title="Rigged-Reference Validation", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        for key in (
            "reference_joint_count",
            "candidate_count",
            "matched_reference_count",
            "candidate_recall",
            "joint_type_match_count",
            "body1_match_count",
            "body0_match_count",
            "axis_match_count",
            "missing_candidate_count",
            "extra_candidate_count",
        ):
            table.add_row(key, str(summary.get(key, "N/A")))
        console.print(table)
        console.print(f"Reference manifest: {redact_sensitive_path(manifest_path)}")
        console.print(f"Validation JSON: {redact_sensitive_path(validation_path)}")
        console.print(f"Validation HTML: {redact_sensitive_path(report_path)}")

    except Exception:
        logger.error(_RIGGED_REFERENCE_FAILURE_MESSAGE)
        console.print(f"[red]Error:[/red] {_RIGGED_REFERENCE_FAILURE_MESSAGE}")
        raise typer.Exit(1) from None


@app.command()
@sever_cli_exception_graph
def analyze(
    config: Annotated[
        Path,
        typer.Argument(help="Path to unified YAML configuration file"),
    ],
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose output"),
    ] = False,
    log_file: Annotated[
        Path | None,
        typer.Option("--log-file", help="Path to log file"),
    ] = None,
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Logging level"),
    ] = "INFO",
) -> None:
    """
    Run only the structure analysis step (LLM hierarchy analysis).

    Analyzes the USD hierarchy to produce segment assignments without
    running the full pipeline. Useful for testing or debugging.

    Example usage:
    ```bash
    joint-agent analyze my_joint_asset.yaml
    ```
    """
    return run(
        config=config,
        skip=None,
        only="analyze_structure",
        session_id=None,
        resume=False,
        dry_run=False,
        clean=False,
        verbose=verbose,
        log_file=log_file,
        log_level=log_level,
    )


if __name__ == "__main__":
    app()
