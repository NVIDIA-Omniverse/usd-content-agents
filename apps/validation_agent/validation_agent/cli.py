# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validation Agent CLI interface using Typer and Rich."""

from enum import Enum
from logging import Logger
from pathlib import Path
from typing import Annotated

import typer
from dotenv import find_dotenv, load_dotenv
from rich.console import Console
from world_understanding.validation.cli import (
    run_validation_cli_command,
    run_validation_inputs_cli_command,
)
from world_understanding.validation.rendering_backend_contract import (
    VALIDATION_RENDERING_BACKEND_NAMES,
)

from .utils import get_version

__version__ = get_version()


class LogLevel(str, Enum):
    """Supported CLI logging levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class OutputFormat(str, Enum):
    """Supported Validation Agent CLI output formats."""

    TEXT = "text"
    JSON = "json"


app = typer.Typer(
    name="validation-agent",
    help="Validation Agent - runtime validation for generated 3D content",
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=True,
)
console = Console()


def _load_cli_dotenv() -> None:
    """Load .env using the same cwd-first search pattern as other agent CLIs."""

    dotenv_path = find_dotenv(usecwd=True)
    load_dotenv(dotenv_path=dotenv_path or Path.cwd() / ".env")


def setup_logging(
    verbose: bool = False,
    log_file: Path | None = None,
    log_level: str = "INFO",
) -> Logger:
    """Set up agent and world_understanding logging with shared formatting."""

    from world_understanding.agentic.cli import setup_logging as shared_setup_logging

    return shared_setup_logging(
        agent_name="validation_agent",
        verbose=verbose,
        log_file=log_file,
        log_level=log_level,
    )


def version_callback(value: bool | None) -> None:
    """Print version and exit."""

    if value:
        console.print(
            f"[bold blue]Validation Agent[/bold blue] "
            f"version [green]{__version__}[/green]"
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
        LogLevel,
        typer.Option(
            "--log-level",
            help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
        ),
    ] = LogLevel.INFO,
) -> None:
    """Validation Agent command line interface."""
    _load_cli_dotenv()
    logger = setup_logging(
        verbose=verbose,
        log_file=log_file,
        log_level=log_level.value,
    )

    if verbose:
        logger.debug("Verbose mode enabled")
        logger.debug(f"Log level: {log_level.value}")
        if log_file:
            logger.debug(f"Logging to file: {log_file}")


@app.command()
def run(
    config: Annotated[
        Path,
        typer.Argument(help="Path to a Validation Agent V1 JSON/YAML config"),
    ],
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            "-o",
            help="Override the config working directory for report artifacts",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Write a plan-only report without executing templates",
        ),
    ] = False,
    template_overrides: Annotated[
        list[str] | None,
        typer.Option(
            "--template",
            "-t",
            help="Run only this template name; repeat for ordered targeted checks",
        ),
    ] = None,
    focus_prim_overrides: Annotated[
        list[str] | None,
        typer.Option(
            "--focus-prim",
            help="Override manual focus prim path; repeat for multiple prims",
        ),
    ] = None,
    fail_on_warn: Annotated[
        bool,
        typer.Option(
            "--fail-on-warn",
            help="Return a non-zero exit code for warn verdicts",
        ),
    ] = False,
    output_format: Annotated[
        OutputFormat,
        typer.Option(
            "--format",
            "-f",
            help="Output format: text or json",
        ),
    ] = OutputFormat.TEXT,
) -> None:
    """Run Validation Agent V1 from a config file."""

    run_validation_cli_command(
        config_path=config,
        output_dir=output_dir,
        dry_run=dry_run,
        template_overrides=tuple(template_overrides or ()),
        focus_prim_overrides=tuple(focus_prim_overrides or ()),
        fail_on_warn=fail_on_warn,
        output_format=output_format.value,
        console=console,
    )


@app.command()
def validate(
    inputs: Annotated[
        list[Path],
        typer.Argument(help="Input file or directory to validate; repeat as needed"),
    ],
    task_description: Annotated[
        str,
        typer.Option(
            "--task",
            help="Free-form validation task prompt",
        ),
    ],
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            "-o",
            help="Directory for report artifacts",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Write a plan-only report without executing templates",
        ),
    ] = False,
    template_overrides: Annotated[
        list[str] | None,
        typer.Option(
            "--template",
            "-t",
            help="Run only this template name; repeat for ordered targeted checks",
        ),
    ] = None,
    focus_prim_overrides: Annotated[
        list[str] | None,
        typer.Option(
            "--focus-prim",
            help="Manual focus prim path; repeat for multiple prims",
        ),
    ] = None,
    reference_image_paths: Annotated[
        list[Path] | None,
        typer.Option(
            "--reference-image",
            help="Reference image evidence path; repeat for multiple images",
        ),
    ] = None,
    render_backend: Annotated[
        str | None,
        typer.Option(
            "--render-backend",
            help=(
                "Runtime render backend to place in the generated validation request: "
                f"{', '.join(VALIDATION_RENDERING_BACKEND_NAMES)}"
            ),
        ),
    ] = None,
    render_views: Annotated[
        list[str] | None,
        typer.Option(
            "--render-view",
            help="Render view label for runtime visual evidence; repeat as needed",
        ),
    ] = None,
    render_image_width: Annotated[
        int | None,
        typer.Option(
            "--image-width",
            help="Runtime render image width",
        ),
    ] = None,
    render_image_height: Annotated[
        int | None,
        typer.Option(
            "--image-height",
            help="Runtime render image height",
        ),
    ] = None,
    fail_on_warn: Annotated[
        bool,
        typer.Option(
            "--fail-on-warn",
            help="Return a non-zero exit code for warn verdicts",
        ),
    ] = False,
    output_format: Annotated[
        OutputFormat,
        typer.Option(
            "--format",
            "-f",
            help="Output format: text or json",
        ),
    ] = OutputFormat.TEXT,
) -> None:
    """Run Validation Agent V1 from a task prompt and direct inputs."""

    run_validation_inputs_cli_command(
        task_description=task_description,
        inputs=inputs,
        output_dir=output_dir,
        dry_run=dry_run,
        template_overrides=tuple(template_overrides or ()),
        focus_prim_overrides=tuple(focus_prim_overrides or ()),
        fail_on_warn=fail_on_warn,
        output_format=output_format.value,
        console=console,
        reference_image_paths=tuple(reference_image_paths or ()),
        render_backend=render_backend,
        render_views=tuple(render_views or ()),
        render_image_width=render_image_width,
        render_image_height=render_image_height,
    )
