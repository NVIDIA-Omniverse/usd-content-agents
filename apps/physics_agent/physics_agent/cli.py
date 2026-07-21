# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Physics Agent CLI interface using Typer and Rich."""

import logging
from pathlib import Path
from typing import Annotated, Any

import click
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

from physics_agent.tuning.visual_evidence import (
    DEFAULT_JUDGE_GENERATED_FRAMES,
    DEFAULT_JUDGE_REFERENCE_FRAMES,
    DEFAULT_REFERENCE_VIDEO_FRAMES,
    MAX_VISUAL_JUDGE_FRAMES,
)

from .utils import get_version

__version__ = get_version()

# Load environment variables from .env file. The package __init__ also loads this
# before submodule imports for console-script entry points.
load_dotenv()

# Initialize Typer app and Rich console
app = typer.Typer(
    name="physics-agent",
    help="Physics Agent - VLM-based physics property classification for 3D assets",
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
)
console = Console()

_VALID_REFERENCE_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
_VALID_REFERENCE_VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".m4v",
    ".webm",
    ".avi",
    ".mkv",
}

_PREDICT_FAILURE_MESSAGE = "Prediction failed"
_PREPARE_DATASET_FAILURE_MESSAGE = "Dataset preparation failed"
_USD_BUILD_FAILURE_MESSAGE = "USD dataset building failed"
_TUNE_FAILURE_MESSAGE = "Physics tuning failed"
_REFINE_FAILURE_MESSAGE = "Physics refinement failed"


def _validate_run_config_model_credentials(
    raw_config: dict[str, Any],
    config_path: Path,
    skip_steps: list[str],
    only_steps: list[str],
) -> None:
    """Fail before pipeline construction when a selected model key is absent."""
    from physics_agent.config import get_step_defaults

    validate_selected_model_credentials(
        raw_config,
        config_path,
        skip_steps,
        only_steps,
        get_step_defaults=get_step_defaults,
        guidance_lines=(
            "Set the required provider credential, configure an endpoint-scoped "
            "api_key/api_key_env with base_url, or skip/disable the affected step.",
        ),
    )


def _validate_reference_media_paths(
    *,
    label: str,
    paths: list[Path],
    valid_extensions: set[str],
) -> None:
    for path in paths:
        safe_path = redact_sensitive_path(path)
        if not path.exists():
            console.print(f"[red]Error:[/red] {label} not found: {safe_path}")
            raise typer.Exit(1)
        if not path.is_file():
            console.print(
                f"[red]Error:[/red] {label} must be a file, got directory: {safe_path}"
            )
            raise typer.Exit(1)
        if path.suffix.lower() not in valid_extensions:
            allowed = ", ".join(sorted(valid_extensions))
            safe_suffix = redact_sensitive_path(path.suffix)
            console.print(
                f"[red]Error:[/red] {label} has unsupported extension "
                f"{safe_suffix!r}: {safe_path}. Allowed extensions: {allowed}"
            )
            raise typer.Exit(1)


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
        agent_name="physics_agent",
        verbose=verbose,
        log_file=log_file,
        log_level=log_level,
    )


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        print(
            f"[bold blue]Physics Agent[/bold blue] version [green]{__version__}[/green]"
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
    Physics Agent - VLM-based physics property classification for 3D assets.

    Use [bold]physics-agent --help[/bold] to see available commands.
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
    dataset: Annotated[
        Path | None,
        typer.Option(
            "--dataset",
            "-d",
            help="Override dataset path from config (must be a prepared dataset.jsonl)",
        ),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Override output directory for predictions",
        ),
    ] = None,
    resume: Annotated[
        bool,
        typer.Option(
            "--resume",
            help="Resume from existing predictions",
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
    Run asset classification predictions on a prepared dataset.

    This command calls the prediction API directly (`run_predict`) rather than
    routing through the full pipeline. Use this when you already have a prepared
    `dataset.jsonl` and only need to run VLM inference.

    For the full classify/apply workflow (USD optimization, rendering, dataset
    preparation, predict, apply_physics), use `physics-agent run CONFIG`.

    The legacy form `physics-agent run CONFIG --only predict` is also still
    supported and routes through the unified pipeline executor.

    Example usage:
    ```bash
    physics-agent predict apps/physics_agent/configs/lightbulb.yaml
    physics-agent predict apps/physics_agent/configs/lightbulb.yaml --dataset path/to/dataset.jsonl
    ```

    Output:
    - {working_dir}/predictions/predictions.jsonl: Classification predictions
    - {working_dir}/predictions/report.html: HTML report with visualizations
    """
    # Setup logging for this command
    logger = setup_logging(verbose=verbose, log_file=log_file, log_level=log_level)

    logger.info("Starting Physics Agent prediction (direct API)")

    # Validate config path early so the CLI fails fast rather than letting
    # the API raise FileNotFoundError out of __post_init__ with a less
    # actionable trace.
    safe_config = redact_sensitive_path(config)
    if not config.exists():
        logger.error("Configuration file not found: %s", safe_config)
        console.print(f"[red]Error:[/red] Configuration file not found: {safe_config}")
        raise typer.Exit(1)

    console.print(
        Panel.fit(
            "[bold]Predict[/bold]\n\n"
            f"Configuration: {safe_config}\n"
            f"Dataset Override: {redact_sensitive_path(dataset)}\n"
            f"Output Override: {redact_sensitive_path(output_dir)}\n"
            f"Resume: {'ON' if resume else 'OFF'}\n"
            f"Verbose mode: {'ON' if verbose else 'OFF'}",
            border_style="blue",
        )
    )

    try:
        from physics_agent.api import PredictInput, run_predict

        params = PredictInput(
            config=config,
            dataset_override=dataset,
            output_dir_override=output_dir,
            resume=resume,
            verbose=verbose,
        )

        logger.info("Running predict workflow via run_predict API...")
        result = run_predict(params)

        if result.success:
            console.print("\n[bold green]Prediction completed[/bold green]")
            console.print(f"  Predictions: {result.predictions_count}")
            console.print(f"  Failed: {result.failed_count}")
            if result.predictions_path:
                console.print(
                    f"  Saved to: {redact_sensitive_path(result.predictions_path)}"
                )
            if result.token_stats:
                logger.info(
                    "Token stats: %s", redact_sensitive_config(result.token_stats)
                )
            return

        logger.error(_PREDICT_FAILURE_MESSAGE)
        console.print(f"[red]Error:[/red] {_PREDICT_FAILURE_MESSAGE}")
        raise typer.Exit(1)

    except typer.Exit:
        raise
    except Exception:
        logger.error(_PREDICT_FAILURE_MESSAGE)
        console.print(f"\n[red]Error:[/red] {_PREDICT_FAILURE_MESSAGE}")
        raise typer.Exit(1) from None


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
    physics-agent build-dataset prepare-dataset path/to/prepare_dataset_config.yaml

    # Override dataset path
    physics-agent build-dataset prepare-dataset path/to/prepare_dataset_config.yaml \\
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
    from physics_agent.api import (
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
    physics-agent build-dataset usd path/to/single_usd_config.yaml

    # Batch processing config (with usd_dir)
    physics-agent build-dataset usd path/to/usd_batch_config.yaml

    # Override source (file or directory)
    physics-agent build-dataset usd path/to/data_prep_config.yaml \\
        --source path/to/file_or_dir

    # With metadata extraction
    physics-agent build-dataset usd path/to/data_prep_config.yaml \\
        --extract-metadata
    ```
    """
    # Setup logging
    logger = setup_logging(verbose=verbose, log_file=log_file, log_level=log_level)

    logger.info("Starting Physics Agent Dataset Build Workflow")

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
        else:  # pragma: no cover - mode detection guarantees one source
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
        from physics_agent.api import BuildDatasetUsdInput, build_dataset_usd

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
            output_path = Path(result["output_dir"]).name

            table.add_row(
                usd_name,
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
            from physics_agent.api import BuildDatasetUsdInput, build_dataset_usd

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
    Execute a multi-step asset agent pipeline.

    Uses the unified configuration format where all paths are auto-derived from
    project.working_dir and input.usd_path.

    A typical pipeline includes:
    1. build_dataset_usd: Build dataset from USD files
    2. build_dataset_prepare_dataset: Prepare dataset for classification
    3. predict: Run VLM inference for asset classification

    The pipeline automatically connects outputs from one step to inputs of the next.

    Example usage:
    ```bash
    # Run complete pipeline
    physics-agent run apps/physics_agent/configs/lightbulb.yaml

    # Skip USD dataset building (already exists)
    physics-agent run apps/physics_agent/configs/lightbulb.yaml --skip build_dataset_usd

    # Run only prediction step
    physics-agent run apps/physics_agent/configs/lightbulb.yaml --only predict

    # Dry run to see execution plan
    physics-agent run apps/physics_agent/configs/lightbulb.yaml --dry-run
    ```
    """
    # Setup logging for this command
    logger = setup_logging(verbose=verbose, log_file=log_file, log_level=log_level)

    # Get event listener for CLI output
    listener = get_listener({}, logger_name="physics_agent.cli")

    logger.info("Starting Physics Agent Pipeline")

    # Check if config exists without allowing the raw path into OS errors.
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

    # Reject invalid filters before config tasks, output creation, or backend
    # work.
    try:
        from physics_agent.config.schema import STEP_ORDER

        skip_steps, only_steps = normalize_cli_step_filters(
            skip=skip,
            only=only,
            valid_steps=STEP_ORDER,
        )
    except ValueError as error:
        logger.error("Pipeline step filter validation failed: %s", error)
        console.print(f"[red]Error:[/red] {error}")
        raise typer.Exit(1) from None

    try:
        loaded_config = load_cli_config_mapping(config)
    except (OSError, ValueError) as error:
        logger.error("Pipeline configuration validation failed: %s", error)
        console.print(f"[red]Error:[/red] {error}")
        raise typer.Exit(1) from None

    # Display configuration info via event system
    listener.event(
        "pipeline.config.display",
        {
            "config": redact_sensitive_path(config),
            "skip_steps": redact_sensitive_config(skip_steps),
            "only_steps": redact_sensitive_config(only_steps),
            "resume": resume,
            "dry_run": dry_run,
            "clean": clean,
        },
    )

    if dry_run:
        # Use only a redacted presentation copy of the validated mapping.
        try:
            projected_config = redact_sensitive_config(loaded_config)
            pipeline_config = (
                projected_config if isinstance(projected_config, dict) else {}
            )

            console.print("\n[bold cyan]Pipeline Execution Plan:[/bold cyan]\n")

            # Detect config format (unified vs old)
            is_unified = "project" in pipeline_config

            if is_unified:
                # Unified config format
                project_name = pipeline_config.get("project", {}).get("name", "unknown")
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
            from physics_agent.api.defaults import PIPELINE_STEP_NAMES

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
                        # Implicitly enable if step has any configuration besides 'enabled'
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
        raise typer.Exit(1) from None

    try:
        _validate_run_config_model_credentials(
            loaded_config, config, skip_steps, only_steps
        )
    except ValueError as error:
        logger.error("Pipeline configuration validation failed: %s", error)
        console.print(f"[red]Error:[/red] {error}")
        raise typer.Exit(1) from None
    except Exception:
        message = "Unable to load pipeline configuration"
        logger.error(message)
        console.print(f"[red]Error:[/red] {message}")
        raise typer.Exit(1) from None

    # Execute unified pipeline using API
    try:
        from physics_agent.api import PipelineInput, run_pipeline

        logger.info("Creating unified pipeline workflow")

        # Create CLI event listener with Rich formatting
        from physics_agent.api import CLIEventListener

        cli_listener = CLIEventListener(
            logger=logger, console=console, show_events=False
        )

        # Retain the source path here because the API uses its parent as the
        # anchor for relative paths. The validated mapping above remains the
        # dry-run projection without changing established execution semantics.
        api_params = PipelineInput(
            config=config,
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

        # Display a diagnostic projection; retain raw values for API callers.
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
                "\n[yellow]Tip:[/yellow] Pipeline checkpoint saved. Use --resume to continue."
            )
        raise typer.Exit(1) from None


@app.command()
@sever_cli_exception_graph
def tune(
    scenario: Annotated[
        Path | None,
        typer.Argument(
            help="Path to a tuning scenario YAML (e.g. drop_settle.yaml). "
            "Optional when --user-prompt is supplied.",
            exists=False,  # validated below for an actionable error message
        ),
    ] = None,
    user_prompt: Annotated[
        str | None,
        typer.Option(
            "--user-prompt",
            help="Free-form natural-language description of the desired tune "
            "run (e.g. 'make this object bouncy', 'spin a top on a smooth "
            "surface'). Routed through the NL interpreter to author a "
            "Scenario; explicit YAML fields override interpreter output.",
        ),
    ] = None,
    physics_usd: Annotated[
        Path | None,
        typer.Option(
            "--physics-usd",
            help="Path to the physics-authored USD to tune (output of "
            "`apply_physics`). Required unless the scenario YAML defines "
            "`physics_usd:`.",
            exists=False,
        ),
    ] = None,
    reference_images: Annotated[
        list[Path] | None,
        typer.Option(
            "--reference-image",
            help=(
                "Reference image for the VLM judge. Can be specified multiple times."
            ),
        ),
    ] = None,
    reference_videos: Annotated[
        list[Path] | None,
        typer.Option(
            "--reference-video",
            help=(
                "Reference video for the VLM judge. Can be specified multiple times."
            ),
        ),
    ] = None,
    reference_descriptions: Annotated[
        list[str] | None,
        typer.Option(
            "--reference-description",
            help=(
                "Description parallel to --reference-image. Can be specified "
                "multiple times."
            ),
        ),
    ] = None,
    reference_video_descriptions: Annotated[
        list[str] | None,
        typer.Option(
            "--reference-video-description",
            help=(
                "Description parallel to --reference-video. Can be specified "
                "multiple times."
            ),
        ),
    ] = None,
    reference_video_frames: Annotated[
        int,
        typer.Option(
            "--reference-video-frames",
            help="Frames to extract from each reference video for visual judging.",
            min=1,
            max=MAX_VISUAL_JUDGE_FRAMES,
        ),
    ] = DEFAULT_REFERENCE_VIDEO_FRAMES,
    judge_reference_frames: Annotated[
        int,
        typer.Option(
            "--judge-reference-frames",
            help="Max reference images/video frames to send to the VLM judge.",
            min=1,
            max=MAX_VISUAL_JUDGE_FRAMES,
        ),
    ] = DEFAULT_JUDGE_REFERENCE_FRAMES,
    judge_generated_frames: Annotated[
        int,
        typer.Option(
            "--judge-generated-frames",
            help="Max generated render frames to send to the VLM judge.",
            min=1,
            max=MAX_VISUAL_JUDGE_FRAMES,
        ),
    ] = DEFAULT_JUDGE_GENERATED_FRAMES,
    engine: Annotated[
        str,
        typer.Option(
            "--engine",
            help="Simulation backend: 'ovphysx' (PhysX 5 via daemon, production), "
            "'newton' (NVIDIA Newton, GPU/MuJoCo-warp; install with "
            "apps/physics_agent[newton]; supports contact_ke/contact_kd bounce "
            "tuning; no static_friction or restitution tuning yet), or 'fake' "
            "(deterministic, tests / smoke).",
            click_type=click.Choice(["ovphysx", "newton", "fake"]),
        ),
    ] = "ovphysx",
    optimizer: Annotated[
        str,
        typer.Option(
            "--optimizer",
            help="Optimizer: 'auto' (BoTorch when installed, else hard error), "
            "'botorch' (production BO), 'random' (baseline), 'cma-es' (baseline).",
            click_type=click.Choice(["auto", "botorch", "random", "cma-es"]),
        ),
    ] = "auto",
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            "-o",
            help="Directory for best_params.json, history.jsonl, report.md, "
            "tune_results.json, tuned_physics.usd.",
        ),
    ] = Path("output/tune"),
    max_trials: Annotated[
        int,
        typer.Option(
            "--max-trials",
            help="Positive number of optimizer trials to run.",
            min=1,
        ),
    ] = 30,
    seed: Annotated[
        int,
        typer.Option(
            "--seed",
            help="Seed for optimizer + backend (when supported).",
        ),
    ] = 42,
    enable_judge: Annotated[
        bool,
        typer.Option(
            "--judge/--no-judge",
            help="Run the VLM-as-judge over scenario/history/best_params at "
            "the end of tune (default on). --no-judge writes byte-identical "
            "output to the pre-Part-1.1 baseline (no model calls, no judge "
            "artifacts, no refine loop).",
        ),
    ] = True,
    judge_max_iterations: Annotated[
        int,
        typer.Option(
            "--judge-max-iterations",
            help=(
                "Pass-through cap on refine-loop iterations. HAS NO EFFECT "
                "on single-shot `tune` itself (the runner emits "
                "tune.judge.refine_skipped on 'continue'). Use "
                "`physics-agent refine` for true iteration. Default 3."
            ),
            min=1,
        ),
    ] = 3,
    judge_max_tokens: Annotated[
        int | None,
        typer.Option(
            "--judge-max-tokens",
            help=(
                "Max output tokens for judge responses. "
                "Defaults to the physics judge configuration."
            ),
            min=1,
        ),
    ] = None,
    judge_temperature: Annotated[
        float | None,
        typer.Option(
            "--judge-temperature",
            help=(
                "Temperature for judge calls. Defaults to "
                "the scenario judge block or physics judge configuration."
            ),
            min=0.0,
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
    Tune authored physics parameters against a deterministic simulator.

    Runs after `apply_physics` has authored a simulation-ready USD. The
    optimizer iteratively patches the USD with candidate parameter sets and
    asks the configured backend (OvPhysX or the fake test backend) to score
    each candidate. Best-found parameters and a derivative `tuned_physics.usd`
    are written to `--output-dir`.

    Example usage:
    ```bash
    physics-agent tune scenario.yaml --engine ovphysx --optimizer auto
    physics-agent tune scenario.yaml --engine ovphysx --optimizer botorch
    physics-agent tune scenario.yaml --engine ovphysx --optimizer random
    physics-agent tune scenario.yaml --engine ovphysx --optimizer cma-es

    # NL-driven (Part 1.1): no scenario YAML needed; the interpreter authors one
    physics-agent tune --user-prompt "make this object bouncy" --physics-usd asset.usda
    ```
    """
    from physics_agent.tuning import (
        BoTorchUnavailableError,
        OvPhysXUnavailableError,
        TuneInput,
        run_tune,
    )

    logger = setup_logging(verbose=verbose, log_file=log_file, log_level=log_level)
    logger.info("Starting Physics Agent tune workflow")

    user_prompt_text = (user_prompt or "").strip() or None

    # Either a scenario YAML or a user_prompt must be supplied. The runner
    # invokes the NL interpreter when only ``user_prompt`` is set.
    if scenario is None and user_prompt_text is None:
        console.print(
            "[red]Error:[/red] Either a scenario YAML argument or "
            "--user-prompt is required. Pass one or both."
        )
        raise typer.Exit(1)

    if scenario is not None:
        safe_scenario = redact_sensitive_path(scenario)
        if not scenario.exists():
            console.print(f"[red]Error:[/red] Scenario file not found: {safe_scenario}")
            raise typer.Exit(1)
        if not scenario.is_file():
            console.print(
                f"[red]Error:[/red] Scenario must be a YAML file, got directory: "
                f"{safe_scenario}. Pass an existing scenario YAML file."
            )
            raise typer.Exit(1)

    # Allow `physics_usd` to default from the scenario YAML for ergonomics
    # (only when a YAML was supplied). Guard the load so an OSError on
    # ``read_text()`` or a YAML that parses to a non-mapping (legal but
    # not what we expect — e.g. a top-level list or scalar) surface as a
    # clean CLI error, not a Python traceback.
    if physics_usd is None and scenario is not None:
        try:
            loaded = yaml.safe_load(scenario.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            console.print("[red]Error:[/red] Could not load scenario file")
            raise typer.Exit(1) from None
        if loaded is None:
            scenario_data: dict[str, Any] = {}
        elif isinstance(loaded, dict):
            scenario_data = loaded
        else:
            console.print(
                "[red]Error:[/red] Scenario YAML must be a top-level mapping; "
                f"got {type(loaded).__name__}."
            )
            raise typer.Exit(1)
        candidate = scenario_data.get("physics_usd")
        if candidate:
            physics_usd_path = Path(candidate)
            if not physics_usd_path.is_absolute():
                physics_usd_path = (scenario.parent / physics_usd_path).resolve()
            physics_usd = physics_usd_path

    if physics_usd is None:
        console.print(
            "[red]Error:[/red] --physics-usd is required when the scenario "
            "YAML does not set 'physics_usd' (or when no scenario YAML is "
            "supplied — i.e. the --user-prompt-only path). Pass an existing "
            "USD file via --physics-usd."
        )
        raise typer.Exit(1)

    if not physics_usd.exists():
        console.print(
            "[red]Error:[/red] physics USD not found: "
            f"{redact_sensitive_path(physics_usd)}"
        )
        raise typer.Exit(1)
    if not physics_usd.is_file():
        console.print(
            f"[red]Error:[/red] --physics-usd must be a USD file, got "
            f"directory: {redact_sensitive_path(physics_usd)}"
        )
        raise typer.Exit(1)

    if output_dir.exists() and output_dir.is_file():
        console.print(
            f"[red]Error:[/red] --output-dir must be a directory, got existing "
            f"file: {redact_sensitive_path(output_dir)}"
        )
        raise typer.Exit(1)
    reference_images = reference_images or []
    reference_videos = reference_videos or []
    reference_descriptions = reference_descriptions or None
    reference_video_descriptions = reference_video_descriptions or None
    _validate_reference_media_paths(
        label="--reference-image",
        paths=reference_images,
        valid_extensions=_VALID_REFERENCE_IMAGE_EXTENSIONS,
    )
    _validate_reference_media_paths(
        label="--reference-video",
        paths=reference_videos,
        valid_extensions=_VALID_REFERENCE_VIDEO_EXTENSIONS,
    )
    if reference_descriptions is not None and len(reference_descriptions) != len(
        reference_images
    ):
        console.print(
            "[red]Error:[/red] --reference-description must be supplied once "
            "per --reference-image."
        )
        raise typer.Exit(1)
    if reference_video_descriptions is not None and len(
        reference_video_descriptions
    ) != len(reference_videos):
        console.print(
            "[red]Error:[/red] --reference-video-description must be supplied "
            "once per --reference-video."
        )
        raise typer.Exit(1)

    judge_tokens_label = (
        str(judge_max_tokens) if judge_max_tokens is not None else "default"
    )
    judge_temperature_label = (
        f"{judge_temperature:g}" if judge_temperature is not None else "default"
    )
    panel_lines = ["[bold]Physics Agent — tune[/bold]\n"]
    if scenario is not None:
        panel_lines.append(f"Scenario: {redact_sensitive_path(scenario)}")
    if user_prompt_text:
        # Truncate display only — the full prompt is persisted to artifacts.
        truncated = user_prompt_text[:140] + (
            "…" if len(user_prompt_text) > 140 else ""
        )
        panel_lines.append(
            f"User prompt: {redact_sensitive_config(truncated, _path_context=True)!r}"
        )
    panel_lines.extend(
        [
            f"Physics USD: {redact_sensitive_path(physics_usd)}",
            f"Engine: {engine}",
            f"Optimizer: {optimizer}",
            f"Max trials: {max_trials}",
            f"Seed: {seed}",
            f"Reference images: {len(reference_images)}",
            f"Reference videos: {len(reference_videos)}",
            f"Reference video frames: {reference_video_frames}",
            f"Judge reference frames: {judge_reference_frames}",
            f"Judge generated frames: {judge_generated_frames}",
            f"Judge: {'on' if enable_judge else 'off'}"
            f" (max {judge_max_iterations} iterations)"
            if enable_judge
            else "Judge: off",
            f"Judge tokens: {judge_tokens_label}",
            f"Judge temperature: {judge_temperature_label}",
            f"Output: {redact_sensitive_path(output_dir)}",
        ]
    )
    console.print(Panel.fit("\n".join(panel_lines), border_style="blue"))

    try:
        result = run_tune(
            TuneInput(
                scenario=scenario,
                user_prompt=user_prompt_text,
                physics_usd=physics_usd,
                output_dir=output_dir,
                reference_images=reference_images,
                reference_videos=reference_videos,
                reference_descriptions=reference_descriptions,
                reference_video_descriptions=reference_video_descriptions,
                reference_video_frames=reference_video_frames,
                judge_reference_frames=judge_reference_frames,
                judge_generated_frames=judge_generated_frames,
                engine=engine,
                optimizer=optimizer,
                max_trials=max_trials,
                seed=seed,
                enable_judge=enable_judge,
                judge_max_iterations=judge_max_iterations,
                judge_max_tokens=judge_max_tokens,
                judge_temperature=judge_temperature,
                verbose=verbose,
            )
        )
    except BoTorchUnavailableError as e:
        # Surface the actionable install hint on stderr unmodified — the test
        # suite asserts this exact substring is present.
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from e
    except OvPhysXUnavailableError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from e
    except Exception:
        logger.error(_TUNE_FAILURE_MESSAGE)
        console.print(f"[red]Error:[/red] {_TUNE_FAILURE_MESSAGE}")
        raise typer.Exit(1) from None

    if not result.success:
        console.print("[yellow]Tuning ended with warnings[/yellow]")
        # Cancellation still emits artifacts — exit 0 so chained scripts can
        # rely on the artifact paths in result.artifacts.
        if not result.cancelled:
            raise typer.Exit(1)

    table = Table(title="Tune Results", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Engine", result.engine_used)
    table.add_row("Optimizer", result.optimizer_used)
    table.add_row("Trials", str(result.n_trials))
    table.add_row("Best score", f"{result.best_score:.6g}")
    for k in sorted(result.best_params):
        table.add_row(f"best.{k}", f"{result.best_params[k]:.6g}")
    table.add_row("Output dir", redact_sensitive_path(result.output_dir))
    console.print(table)
    console.print(
        Panel.fit(
            "[bold green]Tuning completed[/bold green]\n"
            "Best params written to "
            f"{redact_sensitive_path(result.artifacts.get('best_params.json', '<missing>'))}",
            border_style="green",
        )
    )


@app.command()
@sever_cli_exception_graph
def refine(
    scenario: Annotated[
        Path,
        typer.Argument(
            help="Path to a tuning scenario YAML (e.g. drop_settle.yaml). "
            "The first iteration's bounds + target come from this file; "
            "subsequent iterations are LLM-refined.",
        ),
    ],
    physics_usd: Annotated[
        Path,
        typer.Option(
            "--physics-usd",
            help="Path to the physics-authored USD to tune (output of "
            "`apply_physics`).",
            exists=False,
        ),
    ],
    user_prompt: Annotated[
        str,
        typer.Option(
            "--user-prompt",
            help="Free-form natural-language description of the desired "
            "tune outcome (e.g. 'make this object bouncy').",
        ),
    ],
    reference_images: Annotated[
        list[Path] | None,
        typer.Option(
            "--reference-image",
            help=(
                "Reference image for the VLM judge. Can be specified multiple times."
            ),
        ),
    ] = None,
    reference_videos: Annotated[
        list[Path] | None,
        typer.Option(
            "--reference-video",
            help=(
                "Reference video for the VLM judge. Can be specified multiple times."
            ),
        ),
    ] = None,
    reference_descriptions: Annotated[
        list[str] | None,
        typer.Option(
            "--reference-description",
            help=(
                "Description parallel to --reference-image. Can be specified "
                "multiple times."
            ),
        ),
    ] = None,
    reference_video_descriptions: Annotated[
        list[str] | None,
        typer.Option(
            "--reference-video-description",
            help=(
                "Description parallel to --reference-video. Can be specified "
                "multiple times."
            ),
        ),
    ] = None,
    reference_video_frames: Annotated[
        int,
        typer.Option(
            "--reference-video-frames",
            help="Frames to extract from each reference video for visual judging.",
            min=1,
            max=MAX_VISUAL_JUDGE_FRAMES,
        ),
    ] = DEFAULT_REFERENCE_VIDEO_FRAMES,
    judge_reference_frames: Annotated[
        int,
        typer.Option(
            "--judge-reference-frames",
            help="Max reference images/video frames to send to the VLM judge.",
            min=1,
            max=MAX_VISUAL_JUDGE_FRAMES,
        ),
    ] = DEFAULT_JUDGE_REFERENCE_FRAMES,
    judge_generated_frames: Annotated[
        int,
        typer.Option(
            "--judge-generated-frames",
            help="Max generated render frames to send to the VLM judge.",
            min=1,
            max=MAX_VISUAL_JUDGE_FRAMES,
        ),
    ] = DEFAULT_JUDGE_GENERATED_FRAMES,
    no_visual_evidence: Annotated[
        bool,
        typer.Option(
            "--no-visual-evidence",
            help=(
                "Run the judge without generated/reference image evidence. "
                "Rendering artifacts are still produced when enabled."
            ),
        ),
    ] = False,
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            "-o",
            help="Directory for per-iteration artifacts. Each iteration "
            "writes its own iter_N/ subdirectory plus a final/ snapshot.",
        ),
    ] = Path("output/refine"),
    engine: Annotated[
        str,
        typer.Option(
            "--engine",
            help="Simulation backend (passed through to ``physics-agent tune``): "
            "'ovphysx' (default, daemon-isolated PhysX), 'newton' "
            "(NVIDIA Newton, GPU/MuJoCo-warp; needs apps/physics_agent[newton]; "
            "supports contact_ke/contact_kd bounce tuning; no static_friction "
            "or restitution tuning yet), "
            "or 'fake' (tests).",
            click_type=click.Choice(["ovphysx", "newton", "fake"]),
        ),
    ] = "ovphysx",
    optimizer: Annotated[
        str,
        typer.Option(
            "--optimizer",
            help="Optimizer (passed through to ``physics-agent tune``).",
            click_type=click.Choice(["auto", "botorch", "random", "cma-es"]),
        ),
    ] = "auto",
    max_trials: Annotated[
        int,
        typer.Option(
            "--max-trials",
            help="Trials per iteration.",
            min=1,
        ),
    ] = 30,
    max_iterations: Annotated[
        int,
        typer.Option(
            "--max-iterations",
            help="Hard cap on (tune+judge+refine) iterations.",
            min=1,
        ),
    ] = 5,
    history_window: Annotated[
        int,
        typer.Option(
            "--history-window",
            help=(
                "Number of prior refine iteration rows to pass to the judge "
                "and scenario refiner; 0 disables prompt history."
            ),
            min=0,
        ),
    ] = 20,
    score_threshold: Annotated[
        float,
        typer.Option(
            "--score-threshold",
            help="Combined-score threshold above which the judge approves "
            "(loop terminates).",
        ),
    ] = 0.7,
    judge_max_tokens: Annotated[
        int | None,
        typer.Option(
            "--judge-max-tokens",
            help=(
                "Max output tokens for judge responses. "
                "Defaults to the physics judge configuration."
            ),
            min=1,
        ),
    ] = None,
    judge_temperature: Annotated[
        float | None,
        typer.Option(
            "--judge-temperature",
            help=(
                "Temperature for judge calls. Defaults to "
                "the scenario judge block or physics judge configuration."
            ),
            min=0.0,
        ),
    ] = None,
    seed: Annotated[
        int,
        typer.Option(
            "--seed",
            help="Seed forwarded to the underlying tune step each iteration.",
        ),
    ] = 42,
    chat_backend: Annotated[
        str,
        typer.Option(
            "--chat-backend",
            help="Backend used for scenario refinement and judge VLM calls. "
            "Default ``gemini`` ships in the public install and reads "
            "GOOGLE_API_KEY / GEMINI_API_KEY. ``nim``, ``openai``, and "
            "``anthropic`` are also registered; installed provider plugins "
            "may add more. Available backends are listed at startup if the "
            "chosen one is not registered.",
        ),
    ] = "gemini",
    chat_model: Annotated[
        str,
        typer.Option(
            "--chat-model",
            help="Chat model identifier passed to ``create_chat_model``. "
            "Default targets Gemini 3.1 Pro via the public ``gemini`` "
            "backend (qwen-397b-a17b on NIM hangs on the scenario_refine "
            "prompt under load). When swapping ``--chat-backend`` you "
            "typically want to change this too.",
        ),
    ] = "gemini-3-pro-preview",
    llm_timeout_seconds: Annotated[
        float,
        typer.Option(
            "--llm-timeout-seconds",
            help="Wall-clock deadline (seconds) for each judge / refine "
            "LLM call. Mirrors the tune runner's safeguard so a hung "
            "NIM/ChatNVIDIA call cannot wedge the refine loop. Set 0 to "
            "disable.",
        ),
    ] = 180.0,
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
    Iteratively refine a physics tune via (tune → judge → scenario_refine).

    Each iteration runs ``tune`` against the current scenario, asks the
    VLM judge whether the result is good enough, and — when it isn't —
    asks an LLM to refine the scenario for the next iteration. The
    loop exits on judge approval or on hitting ``--max-iterations``.

    Output layout:
        ``output_dir/iter_{1..N}/{scenario.yaml,prior_refine_history.json,best_params.json,history.jsonl,judge_result.json,...}``
        ``output_dir/final/...``  (snapshot of the winning iteration)
        ``output_dir/refine_summary.json``  (loop-level summary with compact_history)

    Example:
    ```bash
    physics-agent refine \\
        apps/physics_agent/configs/tuning/drop_settle.yaml \\
        --physics-usd path/to/asset_physics.usda \\
        --user-prompt "make it bouncy" \\
        --output-dir /tmp/refine_run \\
        --engine ovphysx --optimizer random --max-trials 4 --max-iterations 3 \\
        --history-window 20
    ```
    """
    logger = setup_logging(verbose=verbose, log_file=log_file, log_level=log_level)
    logger.info("Starting Physics Agent refine workflow")

    if not scenario.exists():
        console.print(
            "[red]Error:[/red] scenario file not found: "
            f"{redact_sensitive_path(scenario)}"
        )
        raise typer.Exit(1)
    # Mirror tune()'s path validation so passing a directory as the
    # scenario / --physics-usd or an existing file as --output-dir surfaces a
    # clean CLI error instead of an IsADirectoryError / FileExistsError
    # crash on first use (CodeRabbit Round 11 thread #6).
    if not scenario.is_file():
        console.print(
            "[red]Error:[/red] scenario must be a YAML file, got directory: "
            f"{redact_sensitive_path(scenario)}"
        )
        raise typer.Exit(1)
    if not physics_usd.exists():
        console.print(
            "[red]Error:[/red] physics USD not found: "
            f"{redact_sensitive_path(physics_usd)}"
        )
        raise typer.Exit(1)
    if not physics_usd.is_file():
        console.print(
            f"[red]Error:[/red] --physics-usd must be a USD file, got "
            f"directory: {redact_sensitive_path(physics_usd)}"
        )
        raise typer.Exit(1)
    user_prompt = (user_prompt or "").strip()
    if not user_prompt:
        console.print("[red]Error:[/red] --user-prompt must be a non-empty string")
        raise typer.Exit(1)

    if output_dir.exists() and output_dir.is_file():
        console.print(
            f"[red]Error:[/red] --output-dir must be a directory, got existing "
            f"file: {redact_sensitive_path(output_dir)}"
        )
        raise typer.Exit(1)
    reference_images = reference_images or []
    reference_videos = reference_videos or []
    visual_evidence_enabled = not no_visual_evidence
    reference_descriptions = reference_descriptions or None
    reference_video_descriptions = reference_video_descriptions or None
    _validate_reference_media_paths(
        label="--reference-image",
        paths=reference_images,
        valid_extensions=_VALID_REFERENCE_IMAGE_EXTENSIONS,
    )
    _validate_reference_media_paths(
        label="--reference-video",
        paths=reference_videos,
        valid_extensions=_VALID_REFERENCE_VIDEO_EXTENSIONS,
    )
    if reference_descriptions is not None and len(reference_descriptions) != len(
        reference_images
    ):
        console.print(
            "[red]Error:[/red] --reference-description must be supplied once "
            "per --reference-image."
        )
        raise typer.Exit(1)
    if reference_video_descriptions is not None and len(
        reference_video_descriptions
    ) != len(reference_videos):
        console.print(
            "[red]Error:[/red] --reference-video-description must be supplied "
            "once per --reference-video."
        )
        raise typer.Exit(1)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    judge_tokens_label = (
        str(judge_max_tokens) if judge_max_tokens is not None else "default"
    )
    judge_temperature_label = (
        f"{judge_temperature:g}" if judge_temperature is not None else "default"
    )

    panel_lines = [
        "[bold]Physics Agent — refine[/bold]\n",
        f"Scenario:        {redact_sensitive_path(scenario)}",
        f"Physics USD:     {redact_sensitive_path(physics_usd)}",
        "User prompt:     "
        f"{redact_sensitive_config(user_prompt, _path_context=True)!r}",
        f"Engine:          {engine}",
        f"Optimizer:       {optimizer}",
        f"Trials per iter: {max_trials}",
        f"Max iterations:  {max_iterations}",
        f"History window:  {history_window}",
        f"Score threshold: {score_threshold:g}",
        f"Judge tokens:    {judge_tokens_label}",
        f"Judge temp:      {judge_temperature_label}",
        f"Visual evidence: {'on' if visual_evidence_enabled else 'off'}",
        f"Reference images:{len(reference_images):>6}",
        f"Reference videos:{len(reference_videos):>6}",
        f"Reference video frames: {reference_video_frames}",
        f"Judge reference frames: {judge_reference_frames}",
        f"Judge generated frames: {judge_generated_frames}",
        f"Chat backend:    {redact_sensitive_config(chat_backend)}",
        f"Chat model:      {redact_sensitive_config(chat_model, _path_context=True)}",
        f"Output:          {redact_sensitive_path(output_dir)}",
    ]
    console.print(Panel.fit("\n".join(panel_lines), border_style="blue"))

    # Build both the chat model for scenario refinement and the VLM for
    # judge calls from the same CLI-selected backend/model. This keeps the
    # default public path self-consistent: a Gemini-backed refine run should
    # not pass chat preflight, spend a tune iteration, and then fail closed
    # because the judge silently fell back to the global NIM VLM default.
    # Hard-fail when the chat model cannot be constructed: a silent
    # no-refine fallback would let the loop repeat the same scenario and make
    # "physics-agent refine" report success without ever refining the task.
    # Operators who want to drive the loop programmatic-only should call
    # ``IterativePhysicsRefinementTask`` directly with ``chat_model=None``.
    try:
        from world_understanding.agentic.config import get_api_key_for_model_config
        from world_understanding.functions.models.backends.registry import (
            chat_backend_requires_api_key,
            list_chat_backends,
            list_vlm_backends,
        )
        from world_understanding.functions.models.chat_models import (
            create_chat_model,
        )
        from world_understanding.functions.models.vision_language_models import (
            create_vlm,
        )
        from world_understanding.utils.credentials import (
            apply_vlm_nim_env_override,
            get_env_api_key_for_backend,
        )

        from physics_agent.api.defaults import (
            DEFAULT_JUDGE_MAX_TOKENS,
            DEFAULT_JUDGE_TEMPERATURE,
            DEFAULT_VLM_REASONING_EFFORT,
        )
        from physics_agent.tuning.visual_evidence import (
            backend_supports_reasoning_effort,
        )

        registered_backends = list_chat_backends()
        if chat_backend not in registered_backends:
            console.print(
                f"[red]Error:[/red] Chat backend {chat_backend!r} is not "
                "registered. Available chat backends: "
                f"{', '.join(registered_backends)}."
            )
            raise typer.Exit(1)
        registered_vlm_backends = list_vlm_backends()
        if chat_backend not in registered_vlm_backends:
            console.print(
                f"[red]Error:[/red] Backend {chat_backend!r} is not "
                "registered as a VLM backend. Refine uses the VLM path for "
                "judge calls, including text-only runs. Available VLM "
                f"backends: {', '.join(registered_vlm_backends)}."
            )
            raise typer.Exit(1)

        api_key = get_env_api_key_for_backend(chat_backend)
        if api_key is None and chat_backend_requires_api_key(chat_backend):
            console.print(
                f"[red]Error:[/red] No API key found for chat_backend "
                f"{chat_backend!r}. Export the matching environment variable "
                "(for example GOOGLE_API_KEY/GEMINI_API_KEY for gemini, NVIDIA_API_KEY for "
                "nim, OPENAI_API_KEY for openai, ANTHROPIC_API_KEY for "
                "anthropic) before running `physics-agent refine`."
            )
            raise typer.Exit(1)
        chat_kwargs: dict[str, Any] = {
            "backend": chat_backend,
            "model": chat_model,
            "temperature": 0.0,
        }
        if api_key is not None:
            chat_kwargs["api_key"] = api_key
        built_chat_model: Any = create_chat_model(**chat_kwargs)
        vlm_config = apply_vlm_nim_env_override(
            {
                "backend": chat_backend,
                "model": chat_model,
                "temperature": (
                    judge_temperature
                    if judge_temperature is not None
                    else DEFAULT_JUDGE_TEMPERATURE
                ),
                "max_tokens": (
                    judge_max_tokens
                    if judge_max_tokens is not None
                    else DEFAULT_JUDGE_MAX_TOKENS
                ),
                "reasoning_effort": DEFAULT_VLM_REASONING_EFFORT,
            }
        )
        if api_key is not None:
            vlm_config["api_key"] = api_key
        vlm_backend = str(vlm_config.get("backend") or "")
        if vlm_backend and vlm_backend != chat_backend:
            console.print(
                "[red]Error:[/red] VLM judge backend would be overridden "
                f"from {chat_backend!r} to {vlm_backend!r} by the VLM NIM "
                "environment override. Refine uses the same model id for "
                "scenario refinement and VLM judging, so this could send "
                f"{chat_model!r} to the wrong backend. Unset the VLM NIM "
                "override, or run refine with --chat-backend nim and an "
                "appropriate NIM model id."
            )
            raise typer.Exit(1)
        if vlm_backend not in registered_vlm_backends:
            console.print(
                f"[red]Error:[/red] Backend {vlm_backend!r} is not "
                f"registered as a VLM backend. Available VLM backends: "
                f"{', '.join(registered_vlm_backends)}."
            )
            raise typer.Exit(1)
        resolved_vlm_api_key = get_api_key_for_model_config(
            vlm_backend,
            vlm_config,
            "vlm",
        )
        if resolved_vlm_api_key:
            vlm_config["api_key"] = resolved_vlm_api_key
        if not backend_supports_reasoning_effort(vlm_backend):
            vlm_config.pop("reasoning_effort", None)
        built_vlm_model: Any = create_vlm(**vlm_config)
    except typer.Exit:
        raise
    except Exception:  # pragma: no cover — provider-dependent
        logger.error(_REFINE_FAILURE_MESSAGE)
        console.print(f"[red]Error:[/red] {_REFINE_FAILURE_MESSAGE}")
        raise typer.Exit(1) from None

    # Round 15 (doyubkim blocker #3): the CLI now delegates to the
    # first-class :func:`physics_agent.api.run_refine` API rather than
    # instantiating :class:`IterativePhysicsRefinementTask` directly.
    # Both paths execute the same orchestrator under the hood; the API
    # surface gives programmatic callers a typed dataclass entry point
    # mirroring material-agent's ``RefineInput`` / ``RefineOutput``.
    from physics_agent.api.refine import RefineInput, run_refine

    refine_params = RefineInput(
        scenario=scenario,
        physics_usd=physics_usd,
        user_prompt=user_prompt,
        output_dir=output_dir,
        reference_images=reference_images,
        reference_videos=reference_videos,
        reference_descriptions=reference_descriptions,
        reference_video_descriptions=reference_video_descriptions,
        reference_video_frames=reference_video_frames,
        judge_reference_frames=judge_reference_frames,
        judge_generated_frames=judge_generated_frames,
        engine=engine,
        optimizer=optimizer,
        max_trials=max_trials,
        seed=seed,
        max_iterations=max_iterations,
        history_window=history_window,
        score_threshold=score_threshold,
        judge_max_tokens=judge_max_tokens,
        judge_temperature=judge_temperature,
        chat_model=built_chat_model,
        vlm_model=built_vlm_model,
        # Refine publishes the winning time-sampled USD instead of a video.
        # Visual judging still renders PNG evidence when reference media or
        # a generated-only freeform judgment requires it.
        force_record_video="off",
        render_winning_trial=False,
        visual_evidence_enabled=visual_evidence_enabled,
        llm_timeout_seconds=llm_timeout_seconds,
    )

    try:
        result = run_refine(refine_params)
    except Exception:
        logger.error(_REFINE_FAILURE_MESSAGE)
        console.print(f"[red]Error:[/red] {_REFINE_FAILURE_MESSAGE}")
        raise typer.Exit(1) from None

    table = Table(title="Refine Results", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Termination", result.termination_reason)
    table.add_row("Iterations", str(result.iteration_count))
    if result.iterations:
        last = result.iterations[-1]
        table.add_row("Last decision", last.judge_decision)
        if last.judge_score is not None:
            table.add_row("Last judge score", f"{last.judge_score:.3f}")
        else:
            table.add_row("Last judge score", "<none>")
        table.add_row("Final metric", str(last.metric_name))
        if last.metric_value is not None:
            table.add_row("Metric value", f"{last.metric_value:.6g}")
    table.add_row("Final dir", redact_sensitive_path(result.final_dir or "<none>"))
    console.print(table)
    # Surface tune-side failures to the shell. The orchestrator catches
    # exceptions from the inner tune step (``error``) and the
    # ``TuneOutput(success=False)`` soft-failure path
    # (``error`` again, or ``cancelled`` when the user/runtime cancelled
    # mid-trial) and records them via ``termination_reason``. Without this
    # check the CLI would print a green "Refinement completed" panel and
    # exit 0 even when every iteration's tune raised. Scripts pinning on
    # the exit code (e.g. CI runners) need a non-zero exit in those cases.
    if result.termination_reason in ("error", "cancelled"):
        colour = "yellow" if result.termination_reason == "cancelled" else "red"
        label = "Cancelled" if result.termination_reason == "cancelled" else "Error"
        console.print(
            f"[{colour}]{label}:[/{colour}] Refinement {result.termination_reason} "
            f"at iteration {result.iteration_count}"
        )
        raise typer.Exit(1)
    console.print(
        Panel.fit(
            "[bold green]Refinement completed[/bold green]\n"
            f"Per-iteration artifacts under {redact_sensitive_path(output_dir)}",
            border_style="green",
        )
    )


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

    **This command is deprecated. Please use 'physics-agent run' instead.**

    This is an alias for the 'run' command and will be removed in a future version.
    """
    # Print deprecation warning
    console.print(
        "[yellow]⚠ Warning:[/yellow] The 'pipeline' command is deprecated and will be removed in a future version."
    )
    console.print(
        "[yellow]           Please use 'physics-agent run' instead.[/yellow]\n"
    )

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


if __name__ == "__main__":
    app()
