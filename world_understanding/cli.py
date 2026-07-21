# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Command-line interface for World Understanding."""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Annotated, Any

import typer
from dotenv import load_dotenv
from rich import print
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from world_understanding.registry import get_display_registry, get_tool_registry
from world_understanding.render_usd_backend_policy import RENDER_USD_BACKEND_NAMES
from world_understanding.rendering_backend_contract import (
    MAX_REMOTE_RENDER_WORKERS,
    validate_remote_render_max_workers,
    validate_rendering_backend_for_surface,
)
from world_understanding.tools import register_all_tools
from world_understanding.utils.misc_utils import get_version

__version__ = get_version()

load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger("world_understanding.cli")

app = typer.Typer(help="World Understanding CLI (wu)")
console = Console()

_RENDER_USD_DEPRECATED_BACKEND_ALIASES = {"local-ovrtx": "ovrtx"}
_RENDER_USD_ALIAS_REMOVAL_VERSION = "0.6.0"

# Global log level option


def _backend_help(public_backends: str) -> str:
    """Build provider help while keeping optional plugins discoverable at runtime."""
    return f"{public_backends}; installed plugins may add providers"


def _resolve_render_usd_backend(backend: str) -> str:
    """Return the canonical backend selected by ``wu render-usd``."""
    requested_name = backend.strip().lower()
    canonical_name = _RENDER_USD_DEPRECATED_BACKEND_ALIASES.get(
        requested_name,
        requested_name,
    )
    if canonical_name != requested_name:
        console.print(
            "[yellow]Warning:[/yellow] "
            f"--backend {requested_name} is deprecated and will be removed in "
            f"{_RENDER_USD_ALIAS_REMOVAL_VERSION}; use --backend {canonical_name}."
        )
    return validate_rendering_backend_for_surface(
        canonical_name,
        RENDER_USD_BACKEND_NAMES,
        surface="wu render-usd",
    )


def set_log_level(level: str) -> None:
    """Set the logging level."""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")
    logger.setLevel(numeric_level)
    logging.getLogger("world_understanding").setLevel(numeric_level)
    logging.getLogger().setLevel(numeric_level)  # Set root logger level too


def _format_file_size(size_bytes: int) -> str:
    """Format a file size in bytes as a human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def _generate_example_input(input_model: Any) -> dict[str, Any]:
    """Generate example input JSON based on the input model schema."""
    if not input_model:
        return {}

    schema = input_model.model_json_schema()
    properties = schema.get("properties", {})
    required_fields = schema.get("required", [])

    example: dict[str, Any] = {}

    for field_name, field_info in properties.items():
        field_type = field_info.get("type", "string")
        description = field_info.get("description", "")

        # Generate example based on field type and name
        if field_name == "image":
            # Special handling for image fields
            if "properties" in field_info:
                # It's an object with properties
                example[field_name] = {
                    "path": "path/to/image.jpg",
                    "width": 1920,
                    "height": 1080,
                }
            else:
                example[field_name] = "path/to/image.jpg"

        elif field_name == "prompt":
            example[field_name] = "Your prompt text here"

        elif field_name == "backend":
            example[field_name] = field_info.get("default", "nim")

        elif field_name == "model":
            example[field_name] = "meta/llama-4-maverick-17b-128e-instruct"

        elif field_name == "target_color":
            if "RGB" in description or "color" in description.lower():
                example[field_name] = [255, 87, 51]  # RGB example
            else:
                example[field_name] = "#FF5733"

        elif field_name == "colors":
            example[field_name] = ["#FF0000", "#00FF00", "#0000FF"]

        elif field_type == "integer":
            if "min" in field_info and "max" in field_info:
                # Use a value in the middle of the range
                min_val = field_info.get("minimum", field_info.get("min", 0))
                max_val = field_info.get("maximum", field_info.get("max", 100))
                example[field_name] = (min_val + max_val) // 2
            else:
                example[field_name] = field_info.get("default", 5)

        elif field_type == "number":
            example[field_name] = field_info.get("default", 0.7)

        elif field_type == "boolean":
            example[field_name] = field_info.get("default", True)

        elif field_type == "array":
            items = field_info.get("items", {})
            if items.get("type") == "string":
                example[field_name] = ["example1", "example2"]
            elif items.get("type") == "integer":
                example[field_name] = [1, 2, 3]
            else:
                example[field_name] = []

        elif field_type == "object":
            example[field_name] = {}

        else:
            # Default string
            if field_name in required_fields:
                example[field_name] = f"example_{field_name}"
            elif "default" in field_info:
                example[field_name] = field_info["default"]

    return example


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        print(
            f"[bold blue]World Understanding[/bold blue] "
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
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        "-l",
        help="Set global log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    ),
) -> None:
    """World Understanding CLI with configurable logging."""
    try:
        set_log_level(log_level)
        logger.debug(f"Log level set to {log_level}")
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from e


def setup_registry() -> Any:
    """Setup the tool registry with available tools."""
    logger.debug("Setting up tool registry")
    registry = get_tool_registry()

    # Register all tools using the centralized function
    registered_tools = register_all_tools()
    logger.debug(f"Registered {len(registered_tools)} tools")

    return registry


@app.command()
def list_tools(
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
) -> None:
    """List all available tools."""
    if verbose:
        set_log_level("DEBUG")

    logger.info("Listing available tools")
    registry = setup_registry()
    tools = registry.list_tools()
    logger.debug(f"Found {len(tools)} tools")

    table = Table(title="Available Tools")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Version", style="magenta")
    table.add_column("Description", style="green")
    table.add_column("Tags", style="yellow")

    for tool_name in sorted(tools):
        logger.debug(f"Getting info for tool: {tool_name}")
        tool = registry.get(tool_name)
        if tool and hasattr(tool, "spec"):
            spec = tool.spec
            tags = ", ".join(spec.tags)
            table.add_row(spec.name, spec.version, spec.description, tags)
        else:
            logger.warning(f"Tool {tool_name} has no spec")

    console.print(table)
    logger.info(f"Listed {len(tools)} tools")


@app.command()
def tool_info(
    tool_name: str = typer.Argument(..., help="Name of the tool"),
    show_schema: bool = typer.Option(False, "--schema", "-s", help="Show JSON schema"),
    example_input: bool = typer.Option(
        False, "--example-input", "-e", help="Show example input JSON"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
) -> None:
    """Show detailed information about a specific tool."""
    if verbose:
        set_log_level("DEBUG")

    logger.info(f"Getting info for tool: {tool_name}")
    registry = setup_registry()
    tool = registry.get(tool_name)

    if not tool:
        logger.error(f"Tool not found: {tool_name}")
        console.print(f"[red]Tool not found: {tool_name}[/red]")
        raise typer.Exit(1)

    spec = tool.spec
    console.print(f"\n[bold cyan]Tool: {spec.name}[/bold cyan]")
    console.print(f"[yellow]Version:[/yellow] {spec.version}")
    console.print(f"[green]Description:[/green] {spec.description}")
    console.print(f"[magenta]Tags:[/magenta] {', '.join(spec.tags)}")

    if show_schema:
        logger.debug("Showing JSON schema")
        schema = tool.to_json_schema()
        console.print("\n[bold]JSON Schema:[/bold]")
        console.print_json(json.dumps(schema, indent=2))

    if example_input:
        logger.debug("Generating example input JSON")
        console.print("\n[bold]Example Input JSON:[/bold]")
        example = _generate_example_input(spec.input_model)
        console.print_json(json.dumps(example, indent=2))

    # Show input/output models
    console.print("\n[bold]Input Model:[/bold]")
    if spec.input_model:
        schema = spec.input_model.model_json_schema()
        for field_name, field_info in schema.get("properties", {}).items():
            required = field_name in schema.get("required", [])
            req_str = "[red]*[/red]" if required else ""
            console.print(
                f"  {field_name}{req_str}: "
                f"{field_info.get('type', 'unknown')} - "
                f"{field_info.get('description', 'No description')}"
            )

    console.print("\n[bold]Output Model:[/bold]")
    if spec.output_model:
        schema = spec.output_model.model_json_schema()
        for field_name, field_info in schema.get("properties", {}).items():
            console.print(
                f"  {field_name}: {field_info.get('type', 'unknown')} - "
                f"{field_info.get('description', 'No description')}"
            )
    logger.info(f"Displayed info for tool: {tool_name}")


@app.command()
def run_tool(
    tool_name: str = typer.Argument(..., help="Name of the tool to run"),
    inputs_file: Path | None = typer.Option(
        None, "--inputs", "-i", help="Path to JSON file with inputs"
    ),
    output_format: str = typer.Option(
        "text", "--format", "-f", help="Output format: text or json"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
) -> None:
    """Run a specific tool with the given inputs."""
    if verbose:
        set_log_level("DEBUG")

    logger.info(f"Running tool: {tool_name}")
    registry = setup_registry()
    tool = registry.get(tool_name)

    if not tool:
        logger.error(f"Tool not found: {tool_name}")
        console.print(f"[red]Tool not found: {tool_name}[/red]")
        raise typer.Exit(1)

    # Load inputs
    inputs = {}
    if inputs_file:
        logger.debug(f"Loading inputs from file: {inputs_file}")
        with open(inputs_file, encoding="utf-8") as f:
            inputs = json.load(f)
        logger.debug(f"Loaded inputs: {inputs}")
    else:
        logger.debug("No inputs file provided, using empty inputs")

    # Run the tool
    try:
        result = tool.run(inputs)
        logger.info(f"Tool {tool_name} completed successfully")

        if output_format == "json":
            result_json = result.model_dump_json(indent=2)
            console.print_json(result_json)
        else:
            # Use display registry if available
            display_registry = get_display_registry()
            # Convert Pydantic model to dict for display functions
            result_dict = (
                result.model_dump() if hasattr(result, "model_dump") else result
            )
            if not display_registry.display(tool_name, result_dict, console):
                # Fallback to JSON
                result_json = result.model_dump_json(indent=2)
                console.print_json(result_json)

    except Exception as e:
        logger.error(f"Error running tool {tool_name}: {e}")
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1) from e


@app.command()
def chat(
    prompt: str = typer.Argument(..., help="Prompt for the chat model"),
    backend: str = typer.Option(
        "nim", "--backend", "-b", help="Chat model backend: nim, azure, or echo"
    ),
    model: str | None = typer.Option(None, "--model", "-m", help="Model to use"),
    temperature: float = typer.Option(
        0.7, "--temperature", "-t", help="Temperature for generation"
    ),
    max_tokens: int = typer.Option(
        500, "--max-tokens", help="Maximum tokens to generate"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
) -> None:
    """Quick chat interaction using the chat tool."""
    if verbose:
        set_log_level("DEBUG")

    # Setup registry to get chat tool
    registry = setup_registry()
    chat_tool = registry.get("chat")

    if not chat_tool:
        logger.error("Chat tool not found")
        console.print("[red]Chat tool not available[/red]")
        raise typer.Exit(1)

    inputs = {
        "prompt": prompt,
        "backend": backend,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if model:
        inputs["model"] = model

    try:
        result = chat_tool.run(inputs)
        if isinstance(result, dict) and "response" in result:
            console.print(result["response"])
        elif hasattr(result, "response"):
            console.print(result.response)
        else:
            console.print(str(result))

    except Exception as e:
        logger.error(f"Chat error: {e}")
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1) from e


@app.command()
def run_nat(
    config_path: Path = typer.Argument(..., help="Path to NAT config file (YAML/JSON)"),
    question: str = typer.Argument(..., help="Question to ask the NAT workflow"),
    validate_only: bool = typer.Option(
        False, "--validate", help="Only validate the config file without running"
    ),
    interactive: bool = typer.Option(
        False, "--interactive", "-i", help="Interactive mode with cached workflow"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
) -> None:
    """Run a NAT workflow directly from a config file."""
    if verbose:
        set_log_level("DEBUG")

    # Check if file exists
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        console.print(f"[red]Error: Config file not found: {config_path}[/red]")
        raise typer.Exit(1)

    try:
        from world_understanding.nat.runtime_loader import (
            NATWorkflow,
            query_workflow,
            validate_nat_config,
        )
    except ImportError as e:
        logger.error("NAT runtime not available")
        console.print(
            "[red]Error: NAT runtime is not available. "
            "Please ensure NAT is properly installed and configured.[/red]"
        )
        raise typer.Exit(1) from e

    # Validate config if requested
    if validate_only:
        logger.info(f"Validating NAT config: {config_path}")
        if validate_nat_config(config_path):
            console.print(f"[green]✓ Config file is valid: {config_path}[/green]")
        else:
            console.print(f"[red]✗ Config file is invalid: {config_path}[/red]")
            raise typer.Exit(1)
        return

    # Interactive mode with cached workflow
    if interactive:
        logger.info(f"Starting interactive NAT session with: {config_path}")
        console.print(f"[bold]Interactive NAT Session:[/bold] {config_path.name}")
        console.print("[dim]Type 'exit' or 'quit' to end the session[/dim]\n")

        async def run_interactive() -> None:
            async with NATWorkflow(config_path) as workflow:
                # First query with the provided question
                console.print(f"[dim]Question:[/dim] {question}")
                console.print("[dim]Response:[/dim]")
                result = await workflow.query(question)
                console.print(f"{result}\n")

                # Interactive loop
                while True:
                    try:
                        next_question = typer.prompt("\nNext question")
                        if next_question.lower() in ["exit", "quit"]:
                            console.print("[dim]Ending session...[/dim]")
                            break

                        console.print("[dim]Response:[/dim]")
                        result = await workflow.query(next_question)
                        console.print(result)
                    except KeyboardInterrupt:
                        console.print("\n[dim]Session interrupted[/dim]")
                        break
                    except Exception as e:
                        console.print(f"[red]Error:[/red] {str(e)}")
                        if verbose:
                            console.print_exception()

        try:
            asyncio.run(run_interactive())
            logger.info("Interactive NAT session ended")
        except Exception as e:
            logger.error(f"Interactive session failed: {e}")
            console.print(f"\n[red]Error:[/red] {str(e)}")
            if verbose:
                console.print_exception()
            raise typer.Exit(1) from e
    else:
        # Single query mode
        logger.info(f"Running NAT workflow from: {config_path}")
        console.print(f"[bold]Running NAT workflow:[/bold] {config_path.name}")
        console.print(f"[dim]Question:[/dim] {question}\n")

        async def run() -> None:
            console.print("[dim]Response:[/dim]\n")
            result = await query_workflow(config_path, question)
            console.print(result)

        try:
            asyncio.run(run())
            logger.info("NAT workflow completed successfully")
        except Exception as e:
            logger.error(f"NAT workflow execution failed: {e}")
            console.print(f"\n[red]Error:[/red] {str(e)}")
            if verbose:
                console.print_exception()
            raise typer.Exit(1) from e


@app.command()
def vision(
    image: str = typer.Argument(..., help="Path to image file"),
    prompt: str = typer.Option(
        "Describe this image in detail.",
        "--prompt",
        "-p",
        help="Prompt/question about the image",
    ),
    backend: str = typer.Option(
        "nim",
        "--backend",
        "-b",
        help=_backend_help("VLM backend: nim, gemini, openai, anthropic"),
    ),
    model: str | None = typer.Option(None, "--model", "-m", help="Model to use"),
    system_prompt: str = typer.Option(
        "You are a helpful AI assistant that can analyze images.",
        "--system-prompt",
        help="System prompt for the VLM",
    ),
    temperature: float = typer.Option(0.7, "--temperature", "-t"),
    max_tokens: int = typer.Option(1024, "--max-tokens"),
    output_format: str = typer.Option(
        "text", "--format", "-f", help="Output format: text or json"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
) -> None:
    """Analyze images using a Vision-Language Model (VLM)."""
    if verbose:
        set_log_level("DEBUG")

    logger.info(f"Analyzing image: {image}")
    registry = setup_registry()
    tool = registry.get("vlm")

    if not tool:
        logger.error("VLM tool not found")
        console.print("[red]VLM tool not available[/red]")
        raise typer.Exit(1)

    inputs: dict[str, Any] = {
        "prompt": prompt,
        "images": [image],
        "backend": backend,
        "system_prompt": system_prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if model:
        inputs["model"] = model

    try:
        result = tool.run(inputs)

        if output_format == "json":
            result_json = result.model_dump_json(indent=2)
            console.print_json(result_json)
        else:
            # Use display function if available
            display_registry = get_display_registry()
            result_dict = (
                result.model_dump() if hasattr(result, "model_dump") else result
            )
            if not display_registry.display("vlm", result_dict, console):
                # Fallback: print the response text directly
                console.print(result.response)

    except Exception as e:
        logger.error(f"Vision analysis error: {e}")
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1) from e


@app.command()
def detect(
    image: str = typer.Argument(..., help="Path to image file"),
    prompt: str = typer.Argument(..., help="Object description to detect"),
    threshold: float = typer.Option(
        0.3, "--threshold", "-t", help="Detection threshold"
    ),
    output_format: str = typer.Option(
        "text", "--format", "-f", help="Output format: text or json"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
) -> None:
    """Detect objects in images using Grounding DINO."""
    if verbose:
        set_log_level("DEBUG")

    logger.info(f"Detecting '{prompt}' in image: {image}")
    registry = setup_registry()
    tool = registry.get("grounding_dino")

    if not tool:
        logger.error("Grounding DINO tool not found")
        console.print("[red]Grounding DINO tool not available[/red]")
        raise typer.Exit(1)

    inputs = {
        "image_path": image,
        "prompt": prompt,
        "threshold": threshold,
    }

    try:
        result = tool.run(inputs)

        if output_format == "json":
            result_json = result.model_dump_json(indent=2)
            console.print_json(result_json)
        else:
            # Use display function
            display_registry = get_display_registry()
            result_dict = (
                result.model_dump() if hasattr(result, "model_dump") else result
            )
            if not display_registry.display("grounding_dino", result_dict, console):
                # Fallback to JSON
                result_json = result.model_dump_json(indent=2)
                console.print_json(result_json)

    except Exception as e:
        logger.error(f"Detection error: {e}")
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1) from e


@app.command("usd-search")
def usd_search(
    query: str = typer.Argument(..., help="Search query for USD assets"),
    limit: int = typer.Option(
        10, "--limit", "-l", help="Maximum number of results to return"
    ),
    api_host: str = typer.Option(
        None, "--api-host", help="Custom API host URL (uses default if not provided)"
    ),
    file_extensions: str = typer.Option(
        None,
        "--extensions",
        "-e",
        help="Comma-separated file extensions to filter (e.g., 'mdl,usd')",
    ),
    no_metadata: bool = typer.Option(
        False, "--no-metadata", help="Exclude metadata from results"
    ),
    no_images: bool = typer.Option(
        False, "--no-images", help="Exclude images from results"
    ),
    output_format: str = typer.Option(
        "text", "--format", "-f", help="Output format: text or json"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
) -> None:
    """Search for USD assets using semantic search."""
    if verbose:
        set_log_level("DEBUG")

    logger.info(f"Searching for USD assets: '{query}'")
    registry = setup_registry()
    tool = registry.get("usd_search")

    if not tool:
        logger.error("USD search tool not found")
        console.print("[red]USD search tool not available[/red]")
        raise typer.Exit(1)

    # Prepare inputs
    inputs = {
        "query": query,
        "limit": limit,
        "return_metadata": not no_metadata,
        "return_images": not no_images,
    }

    # Add optional parameters
    if api_host:
        inputs["api_host"] = api_host

    if file_extensions:
        # Parse comma-separated extensions
        extensions_list = [ext.strip() for ext in file_extensions.split(",")]
        inputs["file_extension_include"] = extensions_list

    try:
        result = tool.run(inputs)

        if output_format == "json":
            result_json = result.model_dump_json(indent=2)
            console.print_json(result_json)
        else:
            # Display results in human-readable format
            if not result.success:
                console.print(f"[red]Search failed:[/red] {', '.join(result.errors)}")
                raise typer.Exit(1)

            if result.num_results == 0:
                console.print(f"[yellow]No results found for query: '{query}'[/yellow]")
                return

            # Display search info
            console.print("\n[bold]USD Search Results[/bold]")
            console.print(f"Query: [cyan]{query}[/cyan]")
            console.print(
                f"Results: [green]{result.num_results}[/green] of max {limit}"
            )

            if result.processing_time_ms:
                console.print(f"Time: {result.processing_time_ms:.2f}ms")

            if result.file_extensions:
                console.print(f"File types: {', '.join(result.file_extensions)}")

            console.print("")

            # Display each result
            for i, item in enumerate(result.results, 1):
                console.print(f"[bold]{i}. Search Result {i}[/bold]")

                # Show source information if available
                if item.get("source"):
                    source = item["source"]
                    if isinstance(source, dict):
                        if source.get("name"):
                            console.print(f"   [cyan]Name:[/cyan] {source['name']}")
                        if source.get("path"):
                            console.print(f"   [cyan]Path:[/cyan] {source['path']}")
                        if source.get("ext"):
                            console.print(
                                f"   [cyan]Type:[/cyan] {source['ext'].upper()} file"
                            )
                        if source.get("size"):
                            # Convert size to human readable format
                            size_bytes = source["size"]
                            if size_bytes < 1024:
                                size_str = f"{size_bytes} B"
                            elif size_bytes < 1024 * 1024:
                                size_str = f"{size_bytes / 1024:.2f} KB"
                            else:
                                size_str = f"{size_bytes / (1024 * 1024):.2f} MB"
                            console.print(f"   [cyan]Size:[/cyan] {size_str}")
                        if source.get("modified_timestamp"):
                            console.print(
                                f"   [cyan]Modified:[/cyan] {source['modified_timestamp']}"
                            )

                # Show score if available
                if item.get("score") is not None:
                    console.print(f"   [green]Score:[/green] {item['score']:.4f}")

                # Show RRF score from metadata if available
                if item.get("metadata"):
                    metadata = item["metadata"]
                    if isinstance(metadata, dict):
                        if metadata.get("rrf_score") is not None:
                            console.print(
                                f"   [green]RRF Score:[/green] {metadata['rrf_score']:.6f}"
                            )
                        if metadata.get("rrf_rank") is not None:
                            console.print(
                                f"   [green]Rank:[/green] #{metadata['rrf_rank']}"
                            )

                # Show thumbnail availability
                if item.get("thumbnail_exists") is not None:
                    thumb_status = (
                        "✓ Available" if item["thumbnail_exists"] else "✗ Not available"
                    )
                    console.print(f"   [dim]Thumbnail:[/dim] {thumb_status}")

                # Show ID if needed for reference
                if verbose and item.get("id"):
                    console.print(f"   [dim]ID:[/dim] {item['id']}")

                console.print("")

    except Exception as e:
        logger.error(f"USD search error: {e}")
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1) from e


@app.command()
def edit_image(
    image: str = typer.Argument(..., help="Path to the image to edit"),
    prompt: str = typer.Argument(..., help="Text prompt describing the desired edit"),
    output: str = typer.Option(
        None,
        "--output",
        "-o",
        help="Output path for edited image (default: <input>_edited.png)",
    ),
    save_rescaled_input: str = typer.Option(
        None, "--save-rescaled-input", help="Save the rescaled input image to this path"
    ),
    negative_prompt: str = typer.Option(
        "", "--negative", "-n", help="What to avoid in the edit"
    ),
    server_url: str = typer.Option(
        None,
        "--server",
        "-s",
        help="ComfyUI server URL (uses COMFYUI_URL env var if not provided)",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
) -> None:
    """Edit images using text-guided AI without masks (ComfyUI)."""
    if verbose:
        set_log_level("DEBUG")

    logger.info(f"Editing image: {image} with prompt: '{prompt}'")

    # Determine if we need to save rescaled input
    return_rescaled_input = bool(save_rescaled_input)

    registry = setup_registry()
    tool = registry.get("image_edit")

    if not tool:
        logger.error("Image edit tool not found")
        console.print("[red]Image edit tool not available[/red]")
        console.print("Make sure ComfyUI is configured and COMFYUI_URL is set")
        raise typer.Exit(1)

    inputs = {
        "image_path": image,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "return_rescaled_input": return_rescaled_input,
    }

    if server_url:
        inputs["server_url"] = server_url

    try:
        result = tool.run(inputs)

        # Handle custom output paths if provided
        if output and result.edited_image_path:
            import shutil

            shutil.move(result.edited_image_path, output)
            result.edited_image_path = output

        if save_rescaled_input and result.rescaled_input_path:
            import shutil

            shutil.move(result.rescaled_input_path, save_rescaled_input)
            result.rescaled_input_path = save_rescaled_input

        # Display results
        display_registry = get_display_registry()
        result_dict = result.model_dump() if hasattr(result, "model_dump") else result
        if not display_registry.display("image_edit", result_dict, console):
            # Fallback display
            console.print(
                f"[green]✓[/green] Edited image saved: {result.edited_image_path}"
            )
            if result.rescaled_input_path:
                console.print(
                    f"[green]✓[/green] Rescaled input saved: {result.rescaled_input_path}"
                )
            console.print(f"Execution time: {result.execution_time:.2f}s")

    except Exception as e:
        logger.error(f"Image editing error: {e}")
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1) from e


@app.command("convert-usd")
def convert_usd(
    source: str = typer.Argument(..., help="Source USD file path"),
    destination: str = typer.Argument(..., help="Destination USD file path"),
    force: bool = typer.Option(
        False, "--force", "-f", help="Overwrite destination file if it exists"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
) -> None:
    """Convert USD files between different formats (USD <-> USDA).

    The output format is determined by the file extension:
    - .usd: Binary USD format
    - .usda: ASCII USD format
    - .usdc: Crate USD format (binary)

    Examples:
        # Convert binary USD to ASCII
        wu convert-usd scene.usd scene.usda

        # Convert ASCII to binary
        wu convert-usd scene.usda scene.usd

        # Force overwrite existing file
        wu convert-usd scene.usd scene.usda --force
    """
    if verbose:
        set_log_level("DEBUG")

    from pathlib import Path

    try:
        # Import USD here to avoid issues if not available
        try:
            from pxr import Usd
        except ImportError as e:
            console.print("[red]Error: USD Python bindings not available.[/red]")
            console.print(
                "Please install supported USD Python bindings "
                "(Linux ARM64 + Python 3.12: `uv pip install usd-exchange`; "
                "Linux ARM64 + Python 3.13 is currently unsupported; "
                "other supported platforms: `uv pip install usd-core`)."
            )
            raise typer.Exit(1) from e

        source_path = Path(source)
        dest_path = Path(destination)

        # Check if source file exists
        if not source_path.exists():
            console.print(f"[red]Error: Source file not found: {source}[/red]")
            raise typer.Exit(1)

        # Check if destination exists and force flag
        if dest_path.exists() and not force:
            console.print(
                f"[red]Error: Destination file already exists: {destination}[/red]"
            )
            console.print("Use --force to overwrite existing files")
            raise typer.Exit(1)

        # Validate file extensions
        valid_extensions = {".usd", ".usda", ".usdc"}
        source_ext = source_path.suffix.lower()
        dest_ext = dest_path.suffix.lower()

        if source_ext not in valid_extensions:
            console.print(
                f"[red]Error: Unsupported source file extension: {source_ext}[/red]"
            )
            console.print(f"Supported extensions: {', '.join(valid_extensions)}")
            raise typer.Exit(1)

        if dest_ext not in valid_extensions:
            console.print(
                f"[red]Error: Unsupported destination file extension: {dest_ext}[/red]"
            )
            console.print(f"Supported extensions: {', '.join(valid_extensions)}")
            raise typer.Exit(1)

        # Create destination directory if it doesn't exist
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Converting USD file: {source} -> {destination}")
        console.print(
            f"[dim]Converting {source_ext.upper()} -> {dest_ext.upper()}...[/dim]"
        )

        # Open source stage
        source_stage = Usd.Stage.Open(str(source_path))
        if not source_stage:
            console.print(f"[red]Error: Failed to open source USD file: {source}[/red]")
            raise typer.Exit(1)

        # Export to destination format
        # The USD library automatically determines format based on file extension
        success = source_stage.GetRootLayer().Export(str(dest_path))

        if not success:
            console.print(
                f"[red]Error: Failed to convert USD file to: {destination}[/red]"
            )
            raise typer.Exit(1)

        # Verify the output file was created
        if not dest_path.exists():
            console.print(
                f"[red]Error: Output file was not created: {destination}[/red]"
            )
            raise typer.Exit(1)

        # Get file sizes for comparison
        source_size = source_path.stat().st_size
        dest_size = dest_path.stat().st_size

        console.print("\n[bold green]✓ Conversion successful![/bold green]")
        console.print(
            f"[cyan]Source:[/cyan] {source} ({_format_file_size(source_size)})"
        )
        console.print(
            f"[cyan]Destination:[/cyan] {destination} ({_format_file_size(dest_size)})"
        )

        if verbose:
            # Show some additional info about the stage
            console.print("\n[bold]USD Stage Info:[/bold]")
            console.print(f"Root layer: {source_stage.GetRootLayer().identifier}")
            console.print(
                f"Time codes per second: {source_stage.GetTimeCodesPerSecond()}"
            )
            console.print(f"Frame rate: {source_stage.GetFramesPerSecond()}")
            start_time = source_stage.GetStartTimeCode()
            end_time = source_stage.GetEndTimeCode()
            console.print(f"Time range: {start_time} to {end_time}")

            # Count prims
            prim_count = len(list(source_stage.Traverse()))
            console.print(f"Total prims: {prim_count}")

        logger.info("USD conversion completed successfully")

    except Exception as e:
        console.print(f"[red]Conversion error:[/red] {str(e)}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1) from e


@app.command("flatten-usd")
def flatten_usd(
    source: str = typer.Argument(..., help="Source USD file path"),
    destination: str | None = typer.Argument(
        None,
        help="Destination USD file path. Defaults to <source>_flat.<ext>",
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Overwrite destination file if it exists"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
) -> None:
    """Flatten a composed USD stage into a single self-contained file.

    Resolves all sublayers, references, payloads, and inherits into one layer.
    Useful for sharing scenes without dependency chains or for rendering.

    The output format is determined by the destination file extension:
    - .usd: Binary USD format (default)
    - .usda: ASCII USD format
    - .usdc: Crate USD format (binary)

    Examples:
        # Flatten to default name (scene_flat.usd)
        wu flatten-usd scene.usd

        # Flatten to specific output
        wu flatten-usd scene.usd scene_flat.usda

        # Force overwrite
        wu flatten-usd scene.usd output.usd --force
    """
    if verbose:
        set_log_level("DEBUG")

    from pathlib import Path

    try:
        try:
            from pxr import Usd
        except ImportError as e:
            console.print("[red]Error: USD Python bindings not available.[/red]")
            console.print(
                "Please install supported USD Python bindings "
                "(Linux ARM64 + Python 3.12: `uv pip install usd-exchange`; "
                "Linux ARM64 + Python 3.13 is currently unsupported; "
                "other supported platforms: `uv pip install usd-core`)."
            )
            raise typer.Exit(1) from e

        source_path = Path(source)
        if not source_path.exists():
            console.print(f"[red]Error: Source file not found: {source}[/red]")
            raise typer.Exit(1)

        valid_extensions = {".usd", ".usda", ".usdc"}
        if source_path.suffix.lower() not in valid_extensions:
            console.print(
                f"[red]Error: Unsupported source file extension: "
                f"{source_path.suffix}[/red]"
            )
            raise typer.Exit(1)

        # Default destination: <stem>_flat.<ext>
        if destination is None:
            dest_path = source_path.with_name(
                f"{source_path.stem}_flat{source_path.suffix}"
            )
        else:
            dest_path = Path(destination)

        if dest_path.suffix.lower() not in valid_extensions:
            console.print(
                f"[red]Error: Unsupported destination extension: "
                f"{dest_path.suffix}[/red]"
            )
            raise typer.Exit(1)

        if dest_path.exists() and not force:
            console.print(
                f"[red]Error: Destination file already exists: {dest_path}[/red]"
            )
            console.print("Use --force to overwrite existing files")
            raise typer.Exit(1)

        dest_path.parent.mkdir(parents=True, exist_ok=True)

        console.print(f"[dim]Opening {source}...[/dim]")
        stage = Usd.Stage.Open(str(source_path), Usd.Stage.LoadAll)
        if not stage:
            console.print(f"[red]Error: Failed to open USD file: {source}[/red]")
            raise typer.Exit(1)

        console.print("[dim]Flattening stage...[/dim]")
        flat_layer = stage.Flatten()

        console.print(f"[dim]Exporting to {dest_path}...[/dim]")
        success = flat_layer.Export(str(dest_path))
        if not success:
            console.print(f"[red]Error: Export failed for {dest_path}[/red]")
            raise typer.Exit(1)

        if not dest_path.exists():
            console.print(f"[red]Error: Output file was not created: {dest_path}[/red]")
            raise typer.Exit(1)

        source_size = source_path.stat().st_size
        dest_size = dest_path.stat().st_size

        console.print("\n[bold green]Flatten successful![/bold green]")
        console.print(
            f"[cyan]Source:[/cyan] {source} ({_format_file_size(source_size)})"
        )
        console.print(
            f"[cyan]Output:[/cyan] {dest_path} ({_format_file_size(dest_size)})"
        )

        if verbose:
            console.print("\n[bold]Flattened stage info:[/bold]")
            sublayer_count = len(stage.GetRootLayer().subLayerPaths)
            console.print(f"Sublayers resolved: {sublayer_count}")
            prim_count = len(list(stage.Traverse()))
            console.print(f"Total prims: {prim_count}")

        logger.info("USD flatten completed successfully")

    except Exception as e:
        console.print(f"[red]Flatten error:[/red] {str(e)}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1) from e


@app.command("print-usd")
def print_usd(
    usd_path: str = typer.Argument(..., help="Path to the USD file to analyze"),
    start_prim: str = typer.Option(
        None,
        "--start-prim",
        "-p",
        help="Start traversal from specific prim path (e.g., /World/Geo)",
    ),
    show_types: bool = typer.Option(
        False, "--show-types", "-t", help="Show prim types in brackets"
    ),
    show_variants: bool = typer.Option(
        False, "--show-variants", help="Show variant set selections"
    ),
    show_api_schemas: bool = typer.Option(
        False, "--show-api-schemas", help="Show applied API schemas"
    ),
    show_collections: bool = typer.Option(
        False,
        "--show-collections",
        help="Show collections defined on each prim with their includes",
    ),
    show_custom_tokens: bool = typer.Option(
        False,
        "--show-custom-tokens",
        help="Show custom token attributes defined on each prim",
    ),
    show_all: bool = typer.Option(
        False,
        "--show-all",
        "-a",
        help="Enable all --show-xxx options (types, variants, API schemas, collections, custom tokens)",
    ),
    active_only: bool = typer.Option(
        False, "--active-only", help="Show only active prims"
    ),
    max_depth: int | None = typer.Option(
        None, "--max-depth", "-d", help="Maximum depth to traverse"
    ),
    no_info: bool = typer.Option(
        False, "--no-info", help="Don't show stage information header"
    ),
    stats: bool = typer.Option(
        False, "--stats", "-s", help="Show statistics about the scene"
    ),
    query_prim: str = typer.Option(
        None,
        "--query-prim",
        "-q",
        help="Query a specific prim and show its collections and Xform ownership",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
) -> None:
    """Print USD scene hierarchy as a tree with optional metadata.

    Examples:
        # Print basic tree structure
        wu print-usd scene.usd

        # Show prim types and variant selections
        wu print-usd scene.usd --show-types --show-variants

        # Show all information at once
        wu print-usd scene.usd --show-all

        # Show all information with limited depth
        wu print-usd scene.usd --show-all --max-depth 3

        # Start from a specific prim
        wu print-usd scene.usd --start-prim /World/Geometry

        # Query a specific prim's collection membership
        wu print-usd scene.usd --query-prim /World/Geometry/mesh1

        # Show statistics about the scene
        wu print-usd scene.usd --stats
    """
    if verbose:
        set_log_level("DEBUG")

    from pathlib import Path

    from world_understanding.functions.graphics.usd_model import USDModel

    # Check if file exists
    usd_file_path = Path(usd_path)
    if not usd_file_path.exists():
        console.print(f"[red]Error: USD file not found: {usd_path}[/red]")
        raise typer.Exit(1)

    try:
        logger.info(f"Loading USD file: {usd_path}")

        # Load the USD model
        model = USDModel(usd_file_path)

        # If --show-all is specified, enable all show options
        if show_all:
            show_types = True
            show_variants = True
            show_api_schemas = True
            show_collections = True
            show_custom_tokens = True

        # If querying a specific prim
        if query_prim:
            logger.info(f"Querying prim: {query_prim}")
            prim = model.get_prim(query_prim)

            if not prim:
                console.print(f"[red]Error: Prim not found: {query_prim}[/red]")
                raise typer.Exit(1)

            console.print(f"\n[bold cyan]Prim Information: {query_prim}[/bold cyan]")
            console.print(f"[yellow]Name:[/yellow] {prim.name}")
            if prim.type_name:
                console.print(f"[yellow]Type:[/yellow] {prim.type_name}")
            console.print(f"[yellow]Active:[/yellow] {prim.is_active}")
            console.print(f"[yellow]Is Xform:[/yellow] {prim.is_xform}")
            console.print(f"[yellow]Is Instance:[/yellow] {prim.is_instance}")
            console.print(f"[yellow]Depth:[/yellow] {prim.get_depth()}")

            # Show parent
            parent = model.get_parent(query_prim)
            if parent:
                console.print(f"[yellow]Parent:[/yellow] {parent.path}")

            # Show children count
            children = model.get_children(query_prim)
            if children:
                console.print(f"[yellow]Children:[/yellow] {len(children)}")
                if verbose:
                    for child in children[:10]:  # Show first 10
                        console.print(f"  • {child.path}")
                    if len(children) > 10:
                        console.print(f"  ... and {len(children) - 10} more")

            # Show collections this prim belongs to
            collections = model.get_collections_containing_prim(query_prim)
            if collections:
                console.print("\n[bold]Member of Collections:[/bold]")
                for collection in collections:
                    xform = model.get_xform_owning_collection(collection)
                    if xform:
                        console.print(
                            f"  • Collection '[green]{collection.name}[/green]' "
                            f"owned by Xform '[cyan]{xform.path}[/cyan]'"
                        )
                    else:
                        console.print(
                            f"  • Collection '[green]{collection.name}[/green]' "
                            f"defined on '[cyan]{collection.prim_path}[/cyan]'"
                        )

            # Show collections defined ON this prim
            defined_collections = model.get_collections_on_prim(query_prim)
            if defined_collections:
                console.print("\n[bold]Collections Defined on This Prim:[/bold]")
                for collection in defined_collections:
                    console.print(f"  • [green]{collection.name}[/green]")
                    if collection.includes:
                        console.print(f"    Includes: {', '.join(collection.includes)}")
                    if collection.excludes:
                        console.print(f"    Excludes: {', '.join(collection.excludes)}")

            # Show path to root
            if verbose:
                path_to_root = model.get_path_to_root(query_prim)
                console.print("\n[bold]Path to Root:[/bold]")
                for i, path in enumerate(path_to_root):
                    console.print(f"  {'  ' * i}└─ {path}")

            # Show subtree stats if this prim has children
            if children:
                subtree_stats = model.get_subtree_stats(query_prim)
                console.print("\n[bold]Subtree Statistics:[/bold]")
                console.print(f"  Total Prims: {subtree_stats['total_prims']}")
                console.print(f"  Max Depth: {subtree_stats['max_depth']}")
                console.print(f"  Xforms: {subtree_stats['num_xforms']}")
                console.print(f"  Instances: {subtree_stats['num_instances']}")
                if subtree_stats["num_inactive"] > 0:
                    console.print(f"  Inactive: {subtree_stats['num_inactive']}")
        else:
            # Print the tree
            logger.info("Printing USD hierarchy")
            model.print_tree(
                start_path=start_prim,
                show_types=show_types,
                show_variants=show_variants,
                show_api_schemas=show_api_schemas,
                show_collections=show_collections,
                show_custom_tokens=show_custom_tokens,
                active_only=active_only,
                max_depth=max_depth,
                show_info=not no_info,
                show_stats=stats,
            )

            # Print summary if verbose
            if verbose and not stats:  # Don't print summary if stats already shown
                console.print()
                model.print_summary()

        logger.info("USD analysis complete")

    except FileNotFoundError as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1) from e
    except RuntimeError as e:
        console.print(f"[red]USD Error: {str(e)}[/red]")
        raise typer.Exit(1) from e
    except Exception as e:
        console.print(f"[red]Unexpected error: {str(e)}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1) from e


@app.command("render-usd")
def render_usd(
    usd_path: str = typer.Argument(..., help="Path to the USD file to render"),
    camera: str = typer.Option(
        "Camera", "--camera", "-c", help="Camera name or path to render"
    ),
    output: str = typer.Option(
        None,
        "--output",
        "-o",
        help="Output path for single frame/camera (use # for frame numbers in multi-frame)",
    ),
    width: int = typer.Option(1920, "--width", "-w", help="Image width in pixels"),
    height: int | None = typer.Option(
        None, "--height", help="Image height in pixels (defaults to width)"
    ),
    frames: str = typer.Option(
        "0", "--frames", "-f", help="Frames to render (e.g., '0', '0:10')"
    ),
    backend: str = typer.Option(
        "remote",
        "--backend",
        "-b",
        help=(
            "Rendering backend: "
            f"{', '.join(RENDER_USD_BACKEND_NAMES)} (default: remote). "
            "local-ovrtx is a deprecated compatibility alias for ovrtx."
        ),
    ),
    sensors: str | None = typer.Option(
        None,
        "--sensors",
        help=(
            "Comma-separated sensor outputs "
            "(remote only: linear_depth,depth,instance_id_segmentation)"
        ),
    ),
    all_cameras: bool = typer.Option(
        False,
        "--all-cameras",
        help="Render all cameras (requires --output-dir)",
    ),
    remote_max_workers: int = typer.Option(
        4,
        "--remote-max-workers",
        help=(
            "Maximum parallel remote camera requests for --all-cameras "
            "(1-32; default: 4). Set WU_NVCF_GLOBAL_MAX_CONCURRENT_REQUESTS "
            "to the endpoint capacity."
        ),
    ),
    output_dir: str = typer.Option(
        None,
        "--output-dir",
        help="Output directory for multi-camera or multi-frame renders",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
    save_camera_json: bool = typer.Option(
        False,
        "--save-camera-json",
        help="Save camera parameters to JSON file alongside rendered images",
    ),
    focus: str | None = typer.Option(
        None,
        "--focus",
        help="Prim path to focus on; auto-frames camera to fit this prim's bounding box",
    ),
    isolate: str | None = typer.Option(
        None,
        "--isolate",
        help="Comma-separated prim paths; hide all other geometry and render only these",
    ),
    hide: str | None = typer.Option(
        None,
        "--hide",
        help="Comma-separated prim paths or subtrees to hide before rendering",
    ),
    direction: str = typer.Option(
        "+x+y+z",
        "--direction",
        help="Camera direction with optional per-axis weights (e.g. '+x+y+z', '+x-0.5y+z')",
    ),
    margin: float = typer.Option(
        None,
        "--margin",
        help="Camera distance margin multiplier (default: 1.3 for --focus, 1.2 otherwise)",
    ),
    focal_length: float = typer.Option(
        None,
        "--focal-length",
        help="Focal length in mm (default: 50.0). Higher = more zoom.",
    ),
    aperture: float = typer.Option(
        None,
        "--aperture",
        help="Horizontal aperture in mm (default: 36.0)",
    ),
    cam_x: float = typer.Option(
        None, "--cam-x", help="Override camera X position (scene units)"
    ),
    cam_y: float = typer.Option(
        None, "--cam-y", help="Override camera Y position (scene units)"
    ),
    cam_z: float = typer.Option(
        None, "--cam-z", help="Override camera Z position (scene units)"
    ),
    target_x: float = typer.Option(
        None, "--target-x", help="Override look-at target X position (scene units)"
    ),
    target_y: float = typer.Option(
        None, "--target-y", help="Override look-at target Y position (scene units)"
    ),
    target_z: float = typer.Option(
        None, "--target-z", help="Override look-at target Z position (scene units)"
    ),
    near_clip: float | None = typer.Option(
        None,
        "--near-clip",
        help="Override camera near clipping plane distance",
    ),
    far_clip: float | None = typer.Option(
        None,
        "--far-clip",
        help="Override camera far clipping plane distance",
    ),
    dome_light: float = typer.Option(
        None,
        "--dome-light",
        help="Add a dome light with the given intensity (e.g. 1500). Replaces existing lights.",
    ),
    distant_light: float = typer.Option(
        None,
        "--distant-light",
        help="Add a distant light with the given intensity (e.g. 800). Replaces existing lights.",
    ),
    ovrtx_log_level: str = typer.Option(
        "warn",
        "--ovrtx-log-level",
        help="OvRTX log level when --backend ovrtx is used.",
    ),
    ovrtx_venv_dir: str | None = typer.Option(
        None,
        "--ovrtx-venv-dir",
        help="Override the isolated OvRTX virtualenv directory.",
    ),
    ovrtx_num_sensor_updates: int | None = typer.Option(
        None,
        "--ovrtx-num-sensor-updates",
        help="OvRTX progressive renderer steps per frame. Higher is cleaner but slower.",
    ),
    ovrtx_render_mode: str | None = typer.Option(
        None,
        "--ovrtx-render-mode",
        help="OvRTX render mode: rt1, rt2, or pt.",
    ),
    material_target: str | None = typer.Option(
        None,
        "--material-target",
        help=(
            "Render material target: auto, preview_surface, openpbr_materialx, "
            "omnipbr_mdl, or display_color. Use openpbr_materialx to avoid "
            "render-export PreviewSurface fallbacks."
        ),
    ),
) -> None:
    """Render USD files.

    Uses a remote rendering endpoint by default (set RENDER_ENDPOINT). Use
    ``--backend ovrtx`` for the local OvRTX subprocess backend. The legacy
    ``local-ovrtx`` alias is deprecated and will be removed in 0.6.0.

    For single frame + single camera: Use either --output or --output-dir
    For multiple frames or cameras: Use --output-dir only

    Examples:
        # Single frame, single camera to specific file
        wu render-usd scene.usd --output render.png

        # Single frame with custom resolution
        wu render-usd scene.usd --output render.png --width 1024 --height 1024

        # Multiple frames (requires --output-dir)
        wu render-usd scene.usd --frames 0:10 --output-dir renders/

        # All cameras (requires --output-dir)
        wu render-usd scene.usd --all-cameras --output-dir renders/

        # With remote sensor outputs
        wu render-usd scene.usd --output render.png --sensors linear_depth

        # With local OvRTX color rendering
        wu render-usd scene.usd --backend ovrtx --output render.png

        # Render focused on a specific prim (auto-frames camera)
        wu render-usd scene.usd --focus /World/Chair --output chair.png

        # Isolate specific prims (hide everything else)
        wu render-usd scene.usd --isolate /World/Chair,/World/Table --output isolated.png

        # Hide known blockers while keeping the rest of the scene
        wu render-usd scene.usd --hide /World/Wall,/World/Glass --output open_view.png

        # Combine focus and isolate
        wu render-usd scene.usd --focus /World/Chair --isolate /World/Chair --output chair_only.png
    """
    if verbose:
        set_log_level("DEBUG")

    try:
        # Resolve the selector before stage, renderer, network, or output work.
        backend_name = _resolve_render_usd_backend(backend)
        try:
            remote_max_workers = validate_remote_render_max_workers(remote_max_workers)
        except ValueError:
            raise ValueError(
                "--remote-max-workers must be between 1 and "
                f"{MAX_REMOTE_RENDER_WORKERS}"
            ) from None

        # Import here to avoid circular imports.
        from world_understanding.functions.graphics import usd_camera

        # Determine if we're rendering multiple frames
        is_multi_frame = ":" in frames or "," in frames

        # Validation logic for output paths
        if all_cameras or is_multi_frame:
            if not output_dir:
                if all_cameras:
                    console.print(
                        "[red]Error:[/red] --output-dir is required when using --all-cameras"
                    )
                else:
                    console.print(
                        "[red]Error:[/red] --output-dir is required when rendering multiple frames"
                    )
                raise typer.Exit(1)
            if output:
                console.print(
                    "[red]Error:[/red] Cannot use --output with multiple cameras or frames. Use --output-dir instead."
                )
                raise typer.Exit(1)
        else:
            if not output and not output_dir:
                console.print(
                    "[red]Error:[/red] Either --output or --output-dir is required"
                )
                raise typer.Exit(1)
            if output and output_dir:
                console.print(
                    "[red]Error:[/red] Cannot specify both --output and --output-dir. Use one or the other."
                )
                raise typer.Exit(1)

        # Parse isolate paths
        isolate_paths = (
            [p.strip() for p in isolate.split(",") if p.strip()] if isolate else None
        )
        hide_paths = [p.strip() for p in hide.split(",") if p.strip()] if hide else None

        # Bundle camera placement overrides
        cam_overrides = {
            "margin": margin,
            "focal_length": focal_length,
            "aperture": aperture,
            "cam_x": cam_x,
            "cam_y": cam_y,
            "cam_z": cam_z,
            "target_x": target_x,
            "target_y": target_y,
            "target_z": target_z,
            "near_clip": near_clip,
            "far_clip": far_clip,
        }

        # Bundle light overrides
        light_overrides = {
            "dome_light": dome_light,
            "distant_light": distant_light,
        }

        if backend_name == "remote":
            _render_remote(
                usd_path=usd_path,
                camera=camera,
                output=output,
                width=width,
                height=height or width,
                frames=frames,
                sensors=sensors,
                all_cameras=all_cameras,
                remote_max_workers=remote_max_workers,
                output_dir=output_dir,
                verbose=verbose,
                save_camera_json_flag=save_camera_json,
                extract_camera_parameters_fn=usd_camera.extract_camera_parameters,
                save_camera_json_fn=usd_camera.save_camera_json,
                focus=focus,
                isolate_paths=isolate_paths,
                hide_paths=hide_paths,
                direction=direction,
                material_target=material_target,
                **cam_overrides,
                **light_overrides,
            )
        else:
            if sensors:
                console.print(
                    "[red]Error:[/red] OvRTX currently supports color renders only; "
                    "--sensors is not supported with --backend ovrtx."
                )
                raise typer.Exit(1)
            _render_ovrtx(
                usd_path=usd_path,
                camera=camera,
                output=output,
                width=width,
                height=height or width,
                frames=frames,
                sensors=sensors,
                all_cameras=all_cameras,
                output_dir=output_dir,
                verbose=verbose,
                save_camera_json_flag=save_camera_json,
                extract_camera_parameters_fn=usd_camera.extract_camera_parameters,
                save_camera_json_fn=usd_camera.save_camera_json,
                focus=focus,
                isolate_paths=isolate_paths,
                hide_paths=hide_paths,
                direction=direction,
                ovrtx_log_level=ovrtx_log_level,
                ovrtx_venv_dir=ovrtx_venv_dir,
                ovrtx_num_sensor_updates=ovrtx_num_sensor_updates,
                ovrtx_render_mode=ovrtx_render_mode,
                material_target=material_target,
                **cam_overrides,
                **light_overrides,
            )
    except ValueError as e:
        console.print(f"[red]Invalid parameters:[/red] {str(e)}")
        raise typer.Exit(1) from e
    except RuntimeError as e:
        console.print(f"[red]Rendering error:[/red] {str(e)}")
        raise typer.Exit(1) from e
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {str(e)}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1) from e


def _apply_light_overrides(
    stage: Any,
    dome_light: float | None,
    distant_light: float | None,
) -> None:
    """Replace existing lights with user-specified dome and/or distant lights."""
    from pxr import Gf, UsdGeom, UsdLux

    from world_understanding.utils.usd.prim import remove_all_lights

    if dome_light is None and distant_light is None:
        return

    remove_all_lights(stage)
    if dome_light is not None:
        console.print(f"[dim]Adding dome light (intensity={dome_light})[/dim]")
        dome = UsdLux.DomeLight.Define(stage, "/RenderLights/DomeLight")
        dome.GetIntensityAttr().Set(dome_light)
    if distant_light is not None:
        console.print(f"[dim]Adding distant light (intensity={distant_light})[/dim]")
        dl = UsdLux.DistantLight.Define(stage, "/RenderLights/DistantLight")
        dl.GetIntensityAttr().Set(distant_light)
        xform = UsdGeom.Xformable(dl)
        xform.AddRotateXYZOp().Set(Gf.Vec3f(315, 45, 0))


def _expand_isolate_paths(stage: Any, paths: list[str]) -> list[str]:
    """Expand isolate paths to include descendant meshes.

    If a path points to a non-Mesh prim (e.g. Xform), all Mesh
    descendants are included so the entire subtree stays visible.
    """
    from pxr import Usd, UsdGeom

    expanded: set[str] = set()
    for p in paths:
        prim = stage.GetPrimAtPath(p)
        if not prim or not prim.IsValid():
            continue
        if prim.IsA(UsdGeom.Mesh):
            expanded.add(p)
        else:
            for desc in Usd.PrimRange(prim):
                if desc.IsA(UsdGeom.Mesh):
                    expanded.add(desc.GetPath().pathString)
    return list(expanded)


def _hide_render_paths(stage: Any, paths: list[str]) -> tuple[int, list[str]]:
    """Hide the requested imageable prims or subtrees before rendering."""
    from pxr import Usd, UsdGeom

    hidden_count = 0
    missing: list[str] = []
    for p in paths:
        prim = stage.GetPrimAtPath(p)
        if not prim or not prim.IsValid():
            missing.append(p)
            continue

        if prim.IsA(UsdGeom.Imageable):
            # Visibility is inherited for imageable prims, so one opinion at
            # the requested root hides its composed subtree without authoring
            # hundreds of descendant overrides.
            UsdGeom.Imageable(prim).MakeInvisible()
            hidden_count += 1
            continue

        # Non-imageable prims cannot carry visibility. Hide imageable
        # descendants directly so callers can still pass grouping prims.
        for desc in Usd.PrimRange(prim):
            if desc.IsA(UsdGeom.Imageable):
                UsdGeom.Imageable(desc).MakeInvisible()
                hidden_count += 1

    return hidden_count, missing


def _apply_camera_clip_overrides(
    stage: Any,
    camera_paths: list[str] | None,
    near_clip: float | None,
    far_clip: float | None,
) -> int:
    """Apply near/far clipping overrides to existing camera prims."""
    if near_clip is None and far_clip is None:
        return 0

    from pxr import Gf, UsdGeom

    camera_prims = []
    if camera_paths is None:
        camera_prims = [prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Camera)]
    else:
        for camera_path in camera_paths:
            prim = stage.GetPrimAtPath(camera_path)
            if prim and prim.IsValid() and prim.IsA(UsdGeom.Camera):
                camera_prims.append(prim)

    updated = 0
    for prim in camera_prims:
        camera = UsdGeom.Camera(prim)
        clipping_attr = camera.GetClippingRangeAttr()
        current = clipping_attr.Get() or Gf.Vec2f(1.0, 1000000.0)
        next_near = float(near_clip) if near_clip is not None else float(current[0])
        next_far = float(far_clip) if far_clip is not None else float(current[1])
        clipping_attr.Set(Gf.Vec2f(next_near, next_far))
        updated += 1

    return updated


def _collect_camera_paths(stage: Any) -> list[str]:
    """Return all composed camera prim paths in traversal order."""
    from pxr import UsdGeom

    return [
        str(prim.GetPath()) for prim in stage.Traverse() if prim.IsA(UsdGeom.Camera)
    ]


def _export_rendered_stage_for_camera_metadata(
    stage: Any,
    fallback_usd_path: str,
    enabled: bool,
) -> str:
    """Export the mutated render stage when camera JSON needs exact metadata."""
    if not enabled:
        return fallback_usd_path

    import tempfile

    fd, metadata_path = tempfile.mkstemp(
        prefix="wu-render-camera-metadata-",
        suffix=".usda",
    )
    os.close(fd)
    exported = stage.GetRootLayer().Export(metadata_path)
    if not exported:
        Path(metadata_path).unlink(missing_ok=True)
        raise RuntimeError("Failed to export rendered stage for camera metadata")
    return metadata_path


def _prepare_render_stage(
    *,
    usd_path: str,
    camera: str,
    width: int,
    height: int,
    all_cameras: bool,
    focus: str | None,
    isolate_paths: list[str] | None,
    hide_paths: list[str] | None,
    direction: str,
    margin: float | None,
    focal_length: float | None,
    aperture: float | None,
    cam_x: float | None,
    cam_y: float | None,
    cam_z: float | None,
    target_x: float | None,
    target_y: float | None,
    target_z: float | None,
    near_clip: float | None,
    far_clip: float | None,
    dome_light: float | None,
    distant_light: float | None,
    stage_label: str,
    prepare_message: str,
) -> tuple[Any, str]:
    """Open, duplicate, and apply shared render-stage mutations."""
    from pxr import Usd, UsdGeom

    from world_understanding.utils.usd import stage as stage_utils
    from world_understanding.utils.usd.camera import (
        add_corner_view_camera,
        add_focused_corner_view_camera,
    )
    from world_understanding.utils.usd.prim import (
        disable_visibility_except_for_selected_mesh_prims,
    )

    camera_path = camera if camera.startswith("/") else f"/{camera}"

    usd_stage = Usd.Stage.Open(usd_path)
    if not usd_stage:
        console.print(f"[red]Error:[/red] Failed to open USD file: {usd_path}")
        raise typer.Exit(1)

    console.print(f"[dim]{prepare_message}[/dim]")
    original_up_axis = UsdGeom.GetStageUpAxis(usd_stage)
    usd_stage = stage_utils.duplicate_stage(usd_stage, stage_label)
    UsdGeom.SetStageUpAxis(usd_stage, original_up_axis)

    _apply_light_overrides(usd_stage, dome_light, distant_light)

    if isolate_paths:
        console.print(f"[dim]Isolating {len(isolate_paths)} prim(s)...[/dim]")
        expanded = _expand_isolate_paths(usd_stage, isolate_paths)
        disable_visibility_except_for_selected_mesh_prims(usd_stage, expanded)

    if hide_paths:
        hidden_count, missing = _hide_render_paths(usd_stage, hide_paths)
        console.print(f"[dim]Hid {hidden_count} requested prim(s)/subtree(s).[/dim]")
        if missing:
            console.print(
                "[yellow]Warning:[/yellow] hide path(s) not found: "
                + ", ".join(missing)
            )

    def _camera_lens() -> tuple[float, float, float]:
        effective_focal = focal_length or 50.0
        effective_h_aperture = aperture or 36.0
        aspect_ratio = width / height
        if aspect_ratio >= 1.0:
            h_aperture = effective_h_aperture
            v_aperture = effective_h_aperture / aspect_ratio
        else:
            v_aperture = effective_h_aperture
            h_aperture = effective_h_aperture * aspect_ratio
        return effective_focal, h_aperture, v_aperture

    camera_configured_by_helper = False
    if focus:
        focus_prim = usd_stage.GetPrimAtPath(focus)
        if not focus_prim or not focus_prim.IsValid():
            console.print(f"[red]Error:[/red] Focus prim not found: {focus}")
            raise typer.Exit(1)

        effective_focal, h_aperture, v_aperture = _camera_lens()
        camera_path = "/Cameras/FocusedCamera"
        console.print(f"[dim]Creating focused camera on '{focus}'...[/dim]")
        add_focused_corner_view_camera(
            focus_prim,
            camera_path=camera_path,
            direction=direction,
            margin=margin or 1.3,
            focal_length=effective_focal,
            horizontal_aperture=h_aperture,
            vertical_aperture=v_aperture,
            cam_x=cam_x,
            cam_y=cam_y,
            cam_z=cam_z,
            target_x=target_x,
            target_y=target_y,
            target_z=target_z,
            near_clip=near_clip,
            far_clip=far_clip,
        )
        camera_configured_by_helper = True
    elif not all_cameras and not usd_stage.GetPrimAtPath(camera_path).IsValid():
        console.print(
            f"[dim]Camera '{camera_path}' not found in scene. "
            f"Auto-creating corner view camera...[/dim]"
        )
        effective_focal, h_aperture, v_aperture = _camera_lens()
        add_corner_view_camera(
            usd_stage,
            camera_path=camera_path,
            direction=direction,
            margin=margin or 1.2,
            focal_length=effective_focal,
            horizontal_aperture=h_aperture,
            vertical_aperture=v_aperture,
            cam_x=cam_x,
            cam_y=cam_y,
            cam_z=cam_z,
            target_x=target_x,
            target_y=target_y,
            target_z=target_z,
            near_clip=near_clip,
            far_clip=far_clip,
        )
        camera_configured_by_helper = True

    if all_cameras or not camera_configured_by_helper:
        updated_cameras = _apply_camera_clip_overrides(
            usd_stage,
            None if all_cameras else [camera_path],
            near_clip,
            far_clip,
        )
        if updated_cameras:
            console.print(
                f"[dim]Applied clipping overrides to {updated_cameras} camera(s).[/dim]"
            )

    return usd_stage, camera_path


def _print_multi_camera_render_summary(
    result: dict[str, Any],
    *,
    output_dir: str,
    backend_label: str | None = None,
) -> None:
    console.print("\n[bold green]Rendering Complete![/bold green]")
    if backend_label:
        console.print(f"Backend: {backend_label}")
    console.print(f"Total cameras: {result['total_cameras']}")
    console.print(f"Successful: {result['successful_cameras']}")
    console.print(f"Failed: {result['failed_cameras']}")
    console.print(f"Total render time: {result['total_render_time']:.2f} seconds")
    console.print(f"Output directory: {output_dir}")


def _save_all_camera_render_outputs(
    *,
    result: dict[str, Any],
    output_dir: str,
    usd_stage: Any,
    usd_path: str,
    width: int,
    verbose: bool,
    save_camera_json_flag: bool,
    extract_camera_parameters_fn: Any,
    save_camera_json_fn: Any,
) -> None:
    import os

    from world_understanding.utils.usd import stage as stage_utils

    os.makedirs(output_dir, exist_ok=True)
    camera_metadata_usd_path = _export_rendered_stage_for_camera_metadata(
        usd_stage,
        usd_path,
        save_camera_json_flag,
    )
    try:
        for cam_result in result["results"]:
            if cam_result.get("status") != "success" or not cam_result.get("images"):
                continue
            camera_name = stage_utils.sanitize_name_for_filesystem(cam_result["camera"])
            for idx, img in enumerate(cam_result["images"]):
                if len(cam_result["images"]) > 1:
                    filename = f"render_{camera_name}_{idx:04d}.png"
                else:
                    filename = f"render_{camera_name}.png"
                filepath = os.path.join(output_dir, filename)
                img.save(filepath)
                if verbose:
                    console.print(f"  Saved: {filepath}")

                if (
                    save_camera_json_flag
                    and len(cam_result["images"]) == 1
                    and idx == 0
                ):
                    try:
                        camera_params = extract_camera_parameters_fn(
                            usd_path=camera_metadata_usd_path,
                            camera_path=cam_result["camera"],
                            image_width=width,
                            image_height=img.height,
                        )
                        json_filename = f"render_{camera_name}.json"
                        json_path = os.path.join(output_dir, json_filename)
                        save_camera_json_fn(camera_params, json_path)
                        if verbose:
                            console.print(f"  Camera JSON: {json_path}")
                    except Exception as e:
                        if verbose:
                            console.print(
                                "[yellow]  Warning: Failed to save camera JSON: "
                                f"{e}[/yellow]"
                            )

        if verbose and result["results"]:
            console.print("\n[bold]Camera Results:[/bold]")
            for cam_result in result["results"]:
                status_icon = "ok" if cam_result.get("status") == "success" else "FAIL"
                console.print(
                    f"  [{status_icon}] {cam_result['camera']}: "
                    f"{cam_result['frame_count']} frames in "
                    f"{cam_result['render_time']:.2f}s"
                )
                if cam_result.get("error"):
                    console.print(f"    Error: {cam_result['error']}")
    finally:
        if camera_metadata_usd_path != usd_path:
            Path(camera_metadata_usd_path).unlink(missing_ok=True)


def _get_single_camera_result_or_exit(multi_result: dict[str, Any]) -> dict[str, Any]:
    if multi_result.get("successful_cameras", 0) == 0 or not multi_result.get(
        "results"
    ):
        error_msg = "Rendering failed"
        for result in multi_result.get("results", []):
            if result.get("error"):
                error_msg = result["error"]
                break
        console.print(f"[red]Rendering failed:[/red] {error_msg}")
        raise typer.Exit(1)

    return multi_result["results"][0]


def _print_single_camera_render_summary(
    result: dict[str, Any],
    *,
    backend_label: str | None = None,
) -> None:
    console.print("\n[bold green]Rendering Complete![/bold green]")
    if backend_label:
        console.print(f"Backend: {backend_label}")
    console.print(f"Camera: {result['camera']}")
    console.print(f"Frames rendered: {result['frame_count']}")
    console.print(f"Render time: {result['render_time']:.2f} seconds")
    console.print("Output files:")


def _render_ovrtx(
    usd_path: str,
    camera: str,
    output: str | None,
    width: int,
    height: int,
    frames: str,
    sensors: str | None,
    all_cameras: bool,
    output_dir: str | None,
    verbose: bool,
    save_camera_json_flag: bool,
    extract_camera_parameters_fn: Any,
    save_camera_json_fn: Any,
    focus: str | None = None,
    isolate_paths: list[str] | None = None,
    hide_paths: list[str] | None = None,
    direction: str = "+x+y+z",
    margin: float | None = None,
    focal_length: float | None = None,
    aperture: float | None = None,
    cam_x: float | None = None,
    cam_y: float | None = None,
    cam_z: float | None = None,
    target_x: float | None = None,
    target_y: float | None = None,
    target_z: float | None = None,
    near_clip: float | None = None,
    far_clip: float | None = None,
    dome_light: float | None = None,
    distant_light: float | None = None,
    ovrtx_log_level: str = "warn",
    ovrtx_venv_dir: str | None = None,
    ovrtx_num_sensor_updates: int | None = None,
    ovrtx_render_mode: str | None = None,
    material_target: str | None = None,
) -> None:
    """Render USD files using the local OvRTX backend."""
    from world_understanding.functions.graphics.rendering_backend_factory import (
        create_rendering_backend,
    )

    sensor_list = (
        [s.strip() for s in sensors.split(",") if s.strip()] if sensors else None
    )

    usd_stage, camera_path = _prepare_render_stage(
        usd_path=usd_path,
        camera=camera,
        width=width,
        height=height,
        all_cameras=all_cameras,
        focus=focus,
        isolate_paths=isolate_paths,
        hide_paths=hide_paths,
        direction=direction,
        margin=margin,
        focal_length=focal_length,
        aperture=aperture,
        cam_x=cam_x,
        cam_y=cam_y,
        cam_z=cam_z,
        target_x=target_x,
        target_y=target_y,
        target_z=target_z,
        near_clip=near_clip,
        far_clip=far_clip,
        dome_light=dome_light,
        distant_light=distant_light,
        stage_label="ovrtx_render",
        prepare_message="Preparing stage for local OvRTX rendering...",
    )

    ovrtx_kwargs: dict[str, Any] = {
        "log_level": ovrtx_log_level,
        "ovrtx_venv_dir": ovrtx_venv_dir,
    }
    if ovrtx_num_sensor_updates is not None:
        ovrtx_kwargs["num_sensor_updates"] = ovrtx_num_sensor_updates
    if ovrtx_render_mode is not None:
        ovrtx_kwargs["render_mode"] = ovrtx_render_mode
    if material_target is not None:
        ovrtx_kwargs["material_target"] = material_target

    backend = create_rendering_backend("ovrtx", ovrtx_kwargs)
    render_kwargs: dict[str, Any] = {"base_dir": str(Path(usd_path).resolve().parent)}

    if all_cameras:
        console.print(f"[dim]Rendering all cameras from {usd_path} with OvRTX...[/dim]")
        camera_paths = _collect_camera_paths(usd_stage)
        result = backend.render(
            stage=usd_stage,
            image_width=width,
            image_height=height,
            cameras=camera_paths,
            frames=frames,
            sensors=sensor_list,
            **render_kwargs,
        )

        _print_multi_camera_render_summary(
            result,
            output_dir=str(output_dir),
            backend_label="ovrtx",
        )
        _save_all_camera_render_outputs(
            result=result,
            output_dir=str(output_dir),
            usd_stage=usd_stage,
            usd_path=usd_path,
            width=width,
            verbose=verbose,
            save_camera_json_flag=save_camera_json_flag,
            extract_camera_parameters_fn=extract_camera_parameters_fn,
            save_camera_json_fn=save_camera_json_fn,
        )
    else:
        console.print(
            f"[dim]Rendering camera '{camera_path}' from {usd_path} with OvRTX...[/dim]"
        )
        multi_result = backend.render(
            stage=usd_stage,
            image_width=width,
            image_height=height,
            cameras=[camera_path],
            frames=frames,
            sensors=sensor_list,
            **render_kwargs,
        )

        result = _get_single_camera_result_or_exit(multi_result)
        _print_single_camera_render_summary(result, backend_label="ovrtx")

        camera_metadata_usd_path = _export_rendered_stage_for_camera_metadata(
            usd_stage,
            usd_path,
            save_camera_json_flag,
        )
        try:
            _save_render_images(
                result=result,
                output=output,
                output_dir=output_dir,
                camera=camera_path,
                usd_path=camera_metadata_usd_path,
                width=width,
                verbose=verbose,
                save_camera_json_flag=save_camera_json_flag,
                extract_camera_parameters_fn=extract_camera_parameters_fn,
                save_camera_json_fn=save_camera_json_fn,
            )
        finally:
            if camera_metadata_usd_path != usd_path:
                Path(camera_metadata_usd_path).unlink(missing_ok=True)


def _render_remote(
    usd_path: str,
    camera: str,
    output: str | None,
    width: int,
    height: int,
    frames: str,
    sensors: str | None,
    all_cameras: bool,
    output_dir: str | None,
    verbose: bool,
    save_camera_json_flag: bool,
    extract_camera_parameters_fn: Any,
    save_camera_json_fn: Any,
    remote_max_workers: int = 4,
    focus: str | None = None,
    isolate_paths: list[str] | None = None,
    hide_paths: list[str] | None = None,
    direction: str = "+x+y+z",
    margin: float | None = None,
    focal_length: float | None = None,
    aperture: float | None = None,
    cam_x: float | None = None,
    cam_y: float | None = None,
    cam_z: float | None = None,
    target_x: float | None = None,
    target_y: float | None = None,
    target_z: float | None = None,
    near_clip: float | None = None,
    far_clip: float | None = None,
    dome_light: float | None = None,
    distant_light: float | None = None,
    material_target: str | None = None,
) -> None:
    """Render USD files using the REST API-based remote rendering backend."""
    from world_understanding.functions.graphics.rendering_backend_factory import (
        create_rendering_backend,
    )

    sensor_list = (
        [s.strip() for s in sensors.split(",") if s.strip()] if sensors else None
    )

    usd_stage, camera_path = _prepare_render_stage(
        usd_path=usd_path,
        camera=camera,
        width=width,
        height=height,
        all_cameras=all_cameras,
        focus=focus,
        isolate_paths=isolate_paths,
        hide_paths=hide_paths,
        direction=direction,
        margin=margin,
        focal_length=focal_length,
        aperture=aperture,
        cam_x=cam_x,
        cam_y=cam_y,
        cam_z=cam_z,
        target_x=target_x,
        target_y=target_y,
        target_z=target_z,
        near_clip=near_clip,
        far_clip=far_clip,
        dome_light=dome_light,
        distant_light=distant_light,
        stage_label="remote_render",
        prepare_message="Preparing stage for remote rendering upload...",
    )

    backend_config: dict[str, Any] = {"bundle_mdl_assets": True}
    if material_target is not None:
        backend_config["material_target"] = material_target
    backend = create_rendering_backend("remote", backend_config)
    render_kwargs: dict[str, Any] = {
        "base_dir": str(Path(usd_path).resolve().parent),
    }

    if all_cameras:
        console.print(f"[dim]Rendering all cameras from {usd_path}...[/dim]")
        camera_paths = _collect_camera_paths(usd_stage)
        effective_max_workers = min(remote_max_workers, max(1, len(camera_paths)))
        result = backend.render(
            stage=usd_stage,
            image_width=width,
            image_height=height,
            cameras=camera_paths,
            frames=frames,
            sensors=sensor_list,
            max_workers=effective_max_workers,
            **render_kwargs,
        )

        _print_multi_camera_render_summary(result, output_dir=str(output_dir))
        _save_all_camera_render_outputs(
            result=result,
            output_dir=str(output_dir),
            usd_stage=usd_stage,
            usd_path=usd_path,
            width=width,
            verbose=verbose,
            save_camera_json_flag=save_camera_json_flag,
            extract_camera_parameters_fn=extract_camera_parameters_fn,
            save_camera_json_fn=save_camera_json_fn,
        )
    else:
        # The shared remote backend bundles MDL and texture assets for USDZ and
        # complex scenes before rendering this single camera.
        console.print(f"[dim]Rendering camera '{camera_path}' from {usd_path}...[/dim]")
        multi_result = backend.render(
            stage=usd_stage,
            image_width=width,
            image_height=height,
            cameras=[camera_path],
            frames=frames,
            sensors=sensor_list,
            **render_kwargs,
        )

        result = _get_single_camera_result_or_exit(multi_result)
        _print_single_camera_render_summary(result)

        camera_metadata_usd_path = _export_rendered_stage_for_camera_metadata(
            usd_stage,
            usd_path,
            save_camera_json_flag,
        )
        try:
            _save_render_images(
                result=result,
                output=output,
                output_dir=output_dir,
                camera=camera_path,
                usd_path=camera_metadata_usd_path,
                width=width,
                verbose=verbose,
                save_camera_json_flag=save_camera_json_flag,
                extract_camera_parameters_fn=extract_camera_parameters_fn,
                save_camera_json_fn=save_camera_json_fn,
            )
        finally:
            if camera_metadata_usd_path != usd_path:
                Path(camera_metadata_usd_path).unlink(missing_ok=True)


def _save_render_images(
    result: dict[str, Any],
    output: str | None,
    output_dir: str | None,
    camera: str,
    usd_path: str,
    width: int,
    verbose: bool,
    save_camera_json_flag: bool,
    extract_camera_parameters_fn: Any,
    save_camera_json_fn: Any,
) -> None:
    """Save rendered images to disk."""
    import os

    from world_understanding.utils.usd import stage as stage_utils

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        camera_name = stage_utils.sanitize_name_for_filesystem(camera)

        if result.get("images"):
            for idx, img in enumerate(result["images"]):
                if len(result["images"]) > 1:
                    filename = f"render_{camera_name}_{idx:04d}.png"
                else:
                    filename = f"render_{camera_name}.png"
                filepath = os.path.join(output_dir, filename)
                img.save(filepath)
                console.print(f"  {filepath}")

                if save_camera_json_flag and len(result["images"]) == 1 and idx == 0:
                    try:
                        camera_params = extract_camera_parameters_fn(
                            usd_path=usd_path,
                            camera_path=camera,
                            image_width=width,
                            image_height=img.height,
                        )
                        json_filename = f"render_{camera_name}.json"
                        json_path = os.path.join(output_dir, json_filename)
                        save_camera_json_fn(camera_params, json_path)
                        console.print(f"  Camera JSON: {json_path}")
                    except Exception as e:
                        console.print(
                            f"[yellow]Warning: Failed to save camera JSON: {e}[/yellow]"
                        )
                        if verbose:
                            logger.exception("Failed to save camera JSON")
    elif output:
        parent_dir = os.path.dirname(output)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir)

        if result.get("images"):
            for idx, img in enumerate(result["images"]):
                if len(result["images"]) > 1:
                    base, ext = os.path.splitext(output)
                    if "#" in base:
                        num_hashes = base.count("#")
                        frame_num = str(idx).zfill(num_hashes)
                        filepath = base.replace("#" * num_hashes, frame_num) + ext
                    else:
                        filepath = f"{base}_{idx:04d}{ext}"
                else:
                    filepath = output

                img.save(filepath)
                console.print(f"  {filepath}")

                if save_camera_json_flag and len(result["images"]) == 1 and idx == 0:
                    try:
                        camera_params = extract_camera_parameters_fn(
                            usd_path=usd_path,
                            camera_path=camera,
                            image_width=width,
                            image_height=img.height,
                        )
                        json_path = os.path.splitext(output)[0] + ".json"
                        save_camera_json_fn(camera_params, json_path)
                        console.print(f"  Camera JSON: {json_path}")
                    except Exception as e:
                        console.print(
                            f"[yellow]Warning: Failed to save camera JSON: {e}[/yellow]"
                        )
                        if verbose:
                            logger.exception("Failed to save camera JSON")


@app.command("image-gen")
def image_gen(
    prompt: str = typer.Argument(..., help="Text prompt describing the desired image"),
    output: str = typer.Option(
        "generated.png",
        "--output",
        "-o",
        help="Output file path for the generated image",
    ),
    images: list[str] | None = typer.Option(
        None, "--image", "-i", help="Conditioning image(s) (can be repeated)"
    ),
    backend: str = typer.Option(
        "gemini",
        "--backend",
        "-b",
        help=_backend_help("Image generation backend: gemini, openai, nim"),
    ),
    model: str | None = typer.Option(
        None, "--model", "-m", help="Model to use (backend-specific default if omitted)"
    ),
    base_url: str | None = typer.Option(
        None,
        "--base-url",
        help=(
            "Override the API base URL (OpenAI backend only). Useful for "
            "OpenAI-compatible servers such as a locally-hosted NIM container, "
            "e.g. http://localhost:8000/v1"
        ),
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
) -> None:
    """Generate an image from a text prompt using an image generation model.

    Examples:
        # Simple text-to-image
        wu image-gen "A photorealistic red sports car on a mountain road"

        # With conditioning image
        wu image-gen "Apply realistic materials to this 3D render" -i render.png

        # Multiple conditioning images
        wu image-gen "Generate with applied materials" -i target.png -i depth.png

        # Save to specific path
        wu image-gen "A sunset over the ocean" -o sunset.png

        # Use Gemini backend
        wu image-gen "A cute cat" --backend gemini

        # Use OpenAI backend (gpt-image-1)
        wu image-gen "A cute cat" --backend openai

        # Use NIM backend (FLUX) -- cloud API
        wu image-gen "A cute cat" --backend nim

        # Use a locally-hosted NIM image-gen container (OpenAI-compatible)
        wu image-gen "A cute cat" --backend openai \\
            --model black-forest-labs/flux.2-klein-4b \\
            --base-url http://localhost:8000/v1
    """
    if verbose:
        set_log_level("DEBUG")

    from world_understanding.functions.models.image_generation_models import (
        create_image_generation_model,
    )

    try:
        kwargs: dict[str, Any] = {}
        if model:
            kwargs["model"] = model
        if base_url:
            kwargs["base_url"] = base_url

        if backend == "openai" and "api_key" not in kwargs:
            # Resolve through the endpoint-aware helper instead of writing
            # ``OPENAI_API_KEY`` directly into kwargs: a hosted key against
            # an explicit custom ``--base-url`` would otherwise be passed
            # as an explicit pairing to the factory and forwarded to the
            # custom endpoint. The helper keeps the hosted key only for
            # provider-owned URLs.
            from world_understanding.utils.credentials import (
                get_openai_api_key_for_base_url,
                is_local_base_url,
            )

            resolved = get_openai_api_key_for_base_url(base_url, None)
            if resolved:
                kwargs["api_key"] = resolved
            elif base_url and is_local_base_url(base_url):
                # Local OpenAI-compatible servers commonly run no-auth;
                # inject the documented ``not-used`` placeholder so the
                # local image-gen flow keeps working without ``--api-key``.
                kwargs["api_key"] = "not-used"
            elif not base_url:
                console.print(
                    "[red]Error: OPENAI_API_KEY environment variable is not set[/red]"
                )
                raise typer.Exit(1)
            else:
                console.print(
                    "[red]Error: --base-url points at a custom endpoint and no "
                    "endpoint-scoped API key was provided. Set the matching "
                    "endpoint key in config or via the relevant env var.[/red]"
                )
                raise typer.Exit(1)

        if backend == "nim" and "api_key" not in kwargs:
            from world_understanding.utils.credentials import (
                get_nim_api_key_for_base_url,
                is_local_base_url,
            )

            resolved = get_nim_api_key_for_base_url(base_url, None)
            if resolved:
                kwargs["api_key"] = resolved
            elif base_url and is_local_base_url(base_url):
                kwargs["api_key"] = "not-used"
            elif not base_url:
                console.print(
                    "[red]Error: NVIDIA_API_KEY environment variable is not set[/red]"
                )
                raise typer.Exit(1)
            else:
                console.print(
                    "[red]Error: --base-url points at a custom NIM endpoint and "
                    "no endpoint-scoped API key was provided. Set MA_NIM_API_KEY "
                    "for that endpoint.[/red]"
                )
                raise typer.Exit(1)

        if backend not in {"gemini", "nim", "openai"} and "api_key" not in kwargs:
            import world_understanding.functions.models.backends  # noqa: F401
            from world_understanding.functions.models.backends.registry import (
                image_gen_backend_requires_api_key,
            )
            from world_understanding.utils.credentials import (
                get_env_api_key_for_backend,
            )

            if image_gen_backend_requires_api_key(backend):
                resolved = get_env_api_key_for_backend(backend)
                if resolved:
                    kwargs["api_key"] = resolved
                else:
                    console.print(
                        "[red]Error: No API key is configured for image generation "
                        f"backend {backend!r}.[/red]"
                    )
                    raise typer.Exit(1)

        gen_model = create_image_generation_model(backend, **kwargs)
        logger.info("Using %s backend (model=%s)", backend, gen_model.model_name)

        image_paths = images or []
        if image_paths:
            for img_path in image_paths:
                if not Path(img_path).exists():
                    console.print(
                        f"[red]Error: Conditioning image not found: {img_path}[/red]"
                    )
                    raise typer.Exit(1)
            console.print(
                f"Generating with {len(image_paths)} conditioning image(s)..."
            )
        else:
            console.print("Generating from text prompt...")

        result_image = gen_model.generate(
            prompt=prompt,
            images=image_paths if image_paths else None,
        )

        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result_image.save(str(output_path))
        console.print(f"[green]Image saved to: {output_path}[/green]")

    except Exception as e:
        logger.error(f"Image generation error: {e}")
        console.print(f"[red]Error:[/red] {str(e)}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1) from e


@app.command("query-usd")
def query_usd(
    usd_path: str = typer.Argument(..., help="Path to the USD file to query"),
    name: str | None = typer.Option(
        None, "--name", "-n", help="Glob pattern for prim name (e.g. 'Chair*')"
    ),
    path_pattern: str | None = typer.Option(
        None, "--path", help="Glob pattern for prim path"
    ),
    prim_type: str | None = typer.Option(
        None, "--type", "-t", help="Prim type filter (Mesh, Xform, Camera, Light, ...)"
    ),
    has_material: bool = typer.Option(
        False, "--has-material", help="Only prims with bound materials"
    ),
    no_material: bool = typer.Option(
        False, "--no-material", help="Only prims without materials"
    ),
    min_size: float | None = typer.Option(
        None, "--min-size", help="Minimum bounding box volume"
    ),
    max_size: float | None = typer.Option(
        None, "--max-size", help="Maximum bounding box volume"
    ),
    near: str | None = typer.Option(
        None,
        "--near",
        help="Point 'x,y,z' or prim path to measure distance from",
    ),
    radius: float | None = typer.Option(
        None, "--radius", help="Max distance from --near target"
    ),
    overlaps: str | None = typer.Option(
        None, "--overlaps", help="Prim path; return prims whose bboxes overlap"
    ),
    sort: str = typer.Option(
        "name", "--sort", help="Sort by: name, size, distance, type"
    ),
    limit: int | None = typer.Option(None, "--limit", help="Max number of results"),
    start_prim: str | None = typer.Option(
        None, "--start-prim", help="Root prim for traversal scope"
    ),
    active_only: bool = typer.Option(
        False, "--active-only", help="Skip inactive prims"
    ),
    output_format: str = typer.Option(
        "json", "--format", "-f", help="Output format: json, table, paths"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
) -> None:
    """Query prims in a USD scene with spatial filters.

    Search and filter prims by name, type, size, proximity, overlap,
    and material binding. Returns structured JSON by default.

    Examples:
        wu query-usd scene.usd --type Mesh
        wu query-usd scene.usd --name "Chair*" --format table
        wu query-usd scene.usd --near /World/Table --radius 2.0
        wu query-usd scene.usd --overlaps /World/Table --type Mesh
        wu query-usd scene.usd --type Mesh --min-size 0.1 --sort size
    """
    if verbose:
        set_log_level("DEBUG")

    from pxr import Usd

    from world_understanding.functions.graphics.usd_spatial import query_prims

    usd_file = Path(usd_path)
    if not usd_file.exists():
        console.print(f"[red]Error: File not found: {usd_path}[/red]")
        raise typer.Exit(1)

    try:
        stage = Usd.Stage.Open(str(usd_file))
        if not stage:
            console.print(f"[red]Error: Failed to open USD file: {usd_path}[/red]")
            raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error opening USD file: {e}[/red]")
        raise typer.Exit(1) from e

    # Resolve --near: could be "x,y,z" or a prim path
    near_value: list[float] | str | None = None
    if near:
        if near.startswith("/"):
            near_value = near
        else:
            try:
                near_value = [float(x.strip()) for x in near.split(",")]
                if len(near_value) != 3:
                    console.print(
                        "[red]Error: --near point must be 'x,y,z' (3 values)[/red]"
                    )
                    raise typer.Exit(1)
            except ValueError as e:
                console.print(
                    "[red]Error: --near must be a prim path (/...) or 'x,y,z'[/red]"
                )
                raise typer.Exit(1) from e

    # Resolve material filter
    mat_filter: bool | None = None
    if has_material:
        mat_filter = True
    elif no_material:
        mat_filter = False

    results = query_prims(
        stage,
        name_pattern=name,
        path_pattern=path_pattern,
        prim_type=prim_type,
        has_material=mat_filter,
        min_size=min_size,
        max_size=max_size,
        near=near_value,
        radius=radius,
        overlaps=overlaps,
        sort_by=sort,
        limit=limit,
        start_prim=start_prim,
        active_only=active_only,
    )

    # Build query descriptor for JSON output
    query_desc: dict[str, Any] = {}
    if name:
        query_desc["name"] = name
    if path_pattern:
        query_desc["path"] = path_pattern
    if prim_type:
        query_desc["type"] = prim_type
    if near:
        query_desc["near"] = near
    if radius is not None:
        query_desc["radius"] = radius
    if overlaps:
        query_desc["overlaps"] = overlaps

    if output_format == "json":
        output_data = {
            "query": query_desc,
            "results": results,
            "count": len(results),
        }
        console.print(json.dumps(output_data, indent=2))
    elif output_format == "paths":
        for r in results:
            console.print(r["path"])
    elif output_format == "table":
        table = Table(title=f"Query Results ({len(results)} prims)")
        table.add_column("Path", style="cyan")
        table.add_column("Type")
        table.add_column("Volume", justify="right")
        table.add_column("Distance", justify="right")
        table.add_column("Material")
        for r in results:
            vol = f"{r['volume']:.4f}" if r.get("volume") is not None else "-"
            dist = f"{r['distance']:.4f}" if r.get("distance") is not None else "-"
            mat = r.get("material", "-") or "-"
            # Shorten material path to just the name
            if mat != "-":
                mat = mat.rsplit("/", 1)[-1]
            table.add_row(r["path"], r["type"], vol, dist, mat)
        console.print(table)
    else:
        console.print(f"[red]Error: Unknown format: {output_format}[/red]")
        raise typer.Exit(1)


@app.command("inspect-usd")
def inspect_usd(
    usd_path: str = typer.Argument(..., help="Path to the USD file"),
    prim_paths: list[str] = typer.Argument(..., help="Prim path(s) to inspect"),
    world_transform: bool = typer.Option(
        False, "--world-transform", "-w", help="Include composed world-space transform"
    ),
    geometry: bool = typer.Option(
        False, "--geometry", "-g", help="Include geometry stats (vertex/face counts)"
    ),
    properties: bool = typer.Option(
        False, "--properties", "-p", help="Include all authored properties"
    ),
    show_all: bool = typer.Option(False, "--all", "-a", help="Include everything"),
    output_format: str = typer.Option(
        "json", "--format", "-f", help="Output format: json, table"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
) -> None:
    """Inspect specific prims in a USD scene in detail.

    Shows bounding box, hierarchy, material, transforms, geometry stats,
    and authored properties for one or more prims.

    Examples:
        wu inspect-usd scene.usd /World/Chair_01
        wu inspect-usd scene.usd /World/Chair_01 /World/Table --all
        wu inspect-usd scene.usd /World/Chair_01 --geometry --world-transform
    """
    if verbose:
        set_log_level("DEBUG")

    from pxr import Usd

    from world_understanding.functions.graphics.usd_spatial import inspect_prim

    usd_file = Path(usd_path)
    if not usd_file.exists():
        console.print(f"[red]Error: File not found: {usd_path}[/red]")
        raise typer.Exit(1)

    try:
        stage = Usd.Stage.Open(str(usd_file))
        if not stage:
            console.print(f"[red]Error: Failed to open USD file: {usd_path}[/red]")
            raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error opening USD file: {e}[/red]")
        raise typer.Exit(1) from e

    if show_all:
        world_transform = True
        geometry = True
        properties = True

    results = []
    for pp in prim_paths:
        info = inspect_prim(
            stage,
            pp,
            include_world_transform=world_transform,
            include_geometry=geometry,
            include_properties=properties,
        )
        if info is None:
            console.print(f"[yellow]Warning: Prim not found: {pp}[/yellow]")
            continue
        results.append(info)

    if not results:
        console.print("[red]Error: No valid prims found[/red]")
        raise typer.Exit(1)

    if output_format == "json":
        # Single prim: output dict directly; multiple: output list
        output_data = results[0] if len(results) == 1 else results
        console.print(json.dumps(output_data, indent=2))
    elif output_format == "table":
        table = Table(title="Prim Inspection")
        table.add_column("Path", style="cyan")
        table.add_column("Type")
        table.add_column("Active")
        table.add_column("Children", justify="right")
        table.add_column("Volume", justify="right")
        table.add_column("Material")
        for r in results:
            vol = f"{r['volume']:.4f}" if r.get("volume") is not None else "-"
            mat = r.get("material") or "-"
            if mat != "-":
                mat = mat.rsplit("/", 1)[-1]
            table.add_row(
                r["path"],
                r["type"],
                str(r["active"]),
                str(r.get("child_count", 0)),
                vol,
                mat,
            )
        console.print(table)
        # Also print full JSON for detailed fields
        if any(
            r.get("geometry") or r.get("world_transform") or r.get("properties")
            for r in results
        ):
            console.print("\n[bold]Detailed data:[/bold]")
            console.print(json.dumps(results, indent=2))
    else:
        console.print(f"[red]Error: Unknown format: {output_format}[/red]")
        raise typer.Exit(1)


@app.command("scene-summary")
def scene_summary_cmd(
    usd_path: str = typer.Argument(..., help="Path to the USD file"),
    output_format: str = typer.Option(
        "text", "--format", "-f", help="Output format: text, json"
    ),
    start_prim: str | None = typer.Option(
        None, "--start-prim", help="Scope to a subtree"
    ),
    top_n: int = typer.Option(5, "--top-n", help="Number of largest prims to show"),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
) -> None:
    """Show a quick overview of a USD scene.

    Displays composition statistics, spatial extents, material distribution,
    and the largest prims by bounding box volume.

    Examples:
        wu scene-summary scene.usd
        wu scene-summary scene.usd --format json
        wu scene-summary scene.usd --top-n 10
        wu scene-summary scene.usd --start-prim /World/Room
    """
    if verbose:
        set_log_level("DEBUG")

    from pxr import Usd

    from world_understanding.functions.graphics.usd_spatial import scene_summary

    usd_file = Path(usd_path)
    if not usd_file.exists():
        console.print(f"[red]Error: File not found: {usd_path}[/red]")
        raise typer.Exit(1)

    try:
        stage = Usd.Stage.Open(str(usd_file))
        if not stage:
            console.print(f"[red]Error: Failed to open USD file: {usd_path}[/red]")
            raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error opening USD file: {e}[/red]")
        raise typer.Exit(1) from e

    summary = scene_summary(stage, start_prim=start_prim, top_n=top_n)

    if output_format == "json":
        console.print(json.dumps(summary, indent=2))
    elif output_format == "text":
        si = summary["stage_info"]
        comp = summary["composition"]

        console.print(f"\n[bold]Scene:[/bold] {usd_path}")
        console.print(f"  Root layer: {si.get('root_layer') or 'N/A'}")
        console.print(
            f"  Up axis: {si.get('up_axis', 'N/A')}    "
            f"Meters per unit: {si.get('meters_per_unit', 'N/A')}"
        )
        if si.get("start_time") is not None:
            console.print(
                f"  Time range: {si['start_time']} - {si['end_time']} "
                f"({si.get('fps', 'N/A')} fps)"
            )

        console.print("\n[bold]Composition:[/bold]")
        console.print(f"  Total prims: {comp['total_prims']:,}")
        type_parts = [f"{t}: {c:,}" for t, c in list(comp["type_counts"].items())[:8]]
        console.print(f"  {', '.join(type_parts)}")
        if comp["instance_count"]:
            console.print(f"  Instances: {comp['instance_count']:,}")

        extents = summary.get("spatial_extents")
        if extents:
            console.print("\n[bold]Spatial Extents:[/bold]")
            bmin = extents["min"]
            bmax = extents["max"]
            sz = extents["size"]
            console.print(
                f"  Scene bbox: ({bmin[0]:.2f}, {bmin[1]:.2f}, {bmin[2]:.2f}) "
                f"to ({bmax[0]:.2f}, {bmax[1]:.2f}, {bmax[2]:.2f})"
            )
            console.print(f"  Scene size: {sz[0]:.2f} x {sz[1]:.2f} x {sz[2]:.2f}")

        largest = summary.get("largest_prims", [])
        if largest:
            console.print("\n[bold]Largest prims (by bbox volume):[/bold]")
            for p in largest:
                console.print(f"  {p['path']:<50} {p['volume']:.4f}")

        mats = summary.get("materials", [])
        if mats:
            console.print("\n[bold]Materials:[/bold]")
            for m in mats[:10]:
                mat_name = m["material"]
                if mat_name != "(unassigned)":
                    mat_name = mat_name.rsplit("/", 1)[-1]
                console.print(
                    f"  {mat_name:<30} bound to {m['bound_prim_count']} prims"
                )
        console.print()
    else:
        console.print(f"[red]Error: Unknown format: {output_format}[/red]")
        raise typer.Exit(1)


@app.command()
def optimize(
    goal_file: Annotated[Path, typer.Argument(help="Python file defining the Goal")],
    algorithm: Annotated[
        str,
        typer.Option(
            "--algorithm",
            "-a",
            help="Algorithm: cma-es, simulated-annealing, random-search",
        ),
    ] = "cma-es",
    time_budget: Annotated[
        float | None,
        typer.Option("--time-budget", "-t", help="Override time budget (seconds)"),
    ] = None,
    seed: Annotated[int, typer.Option(help="Random seed")] = 42,
    temp_init: Annotated[float, typer.Option(help="SA: initial temperature")] = 5.0,
    temp_final: Annotated[float, typer.Option(help="SA: final temperature")] = 1e-4,
    step_size: Annotated[float, typer.Option(help="SA: perturbation step size")] = 0.5,
    sigma_init: Annotated[float, typer.Option(help="CMA-ES: initial step size")] = 2.0,
) -> None:
    """Run blackbox optimization on a goal defined in a Python file."""
    import importlib.util
    import inspect

    from world_understanding.functions.optimization import (
        Goal,
        cma_es,
        random_search,
        simulated_annealing,
    )

    if not goal_file.exists():
        console.print(f"[red]Goal file not found: {goal_file}[/red]")
        raise typer.Exit(1)

    # Dynamically import the goal file
    spec_mod = importlib.util.spec_from_file_location("_wu_goal", goal_file)
    if spec_mod is None or spec_mod.loader is None:
        console.print(f"[red]Cannot load goal file: {goal_file}[/red]")
        raise typer.Exit(1)
    module = importlib.util.module_from_spec(spec_mod)
    try:
        spec_mod.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception as e:
        console.print(f"[red]Error loading goal file: {e}[/red]")
        raise typer.Exit(1) from e

    # Detect Goal subclass (Variant B) vs flat module (Variant A)
    # Only consider classes defined in the goal file itself (not imported ones)
    goal_instance: Goal | None = None
    for _name, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, Goal) and obj is not Goal and obj.__module__ == "_wu_goal":
            goal_instance = obj()
            break

    if goal_instance is None:
        # Variant A: flat module — wrap into adapter
        required = [
            "METRIC_NAME",
            "METRIC_DIRECTION",
            "TIME_BUDGET",
            "N_DIMS",
            "BOUNDS",
            "evaluate",
        ]
        missing = [r for r in required if not hasattr(module, r)]
        if missing:
            console.print(
                f"[red]Goal file missing required attributes: {missing}[/red]"
            )
            raise typer.Exit(1)

        _module_evaluate = module.evaluate

        class _ModuleGoal(Goal):
            @property
            def metric_name(self) -> str:
                return module.METRIC_NAME

            @property
            def metric_direction(self) -> str:
                return module.METRIC_DIRECTION

            @property
            def time_budget(self) -> float:
                return float(module.TIME_BUDGET)

            @property
            def n_dims(self) -> int:
                return int(module.N_DIMS)

            @property
            def bounds(self) -> tuple[float, float]:
                return tuple(module.BOUNDS)  # type: ignore[return-value]

            def evaluate(self, **context: Any) -> float:
                return _module_evaluate(**context)

        goal_instance = _ModuleGoal()

    # CLI overrides take precedence
    effective_budget = (
        time_budget if time_budget is not None else goal_instance.time_budget
    )

    # Wrap evaluate to negate output for maximization goals
    # (all algorithm functions minimize internally)
    _raw_evaluate = goal_instance.evaluate
    if goal_instance.metric_direction == "maximize":

        def _eval_fn(**ctx: Any) -> float:
            return -_raw_evaluate(**ctx)

    else:
        _eval_fn = _raw_evaluate  # type: ignore[assignment]

    console.print(f"\n[bold]Optimizing:[/bold] {goal_file.name}")
    console.print(f"[dim]Algorithm:[/dim] {algorithm}")
    console.print(f"[dim]Time budget:[/dim] {effective_budget}s")
    console.print(
        f"[dim]Dims:[/dim] {goal_instance.n_dims}  [dim]Bounds:[/dim] {goal_instance.bounds}\n"
    )

    algo = algorithm.lower().replace("_", "-")
    try:
        if algo == "cma-es":
            result = cma_es(
                _eval_fn,
                goal_instance.bounds,
                goal_instance.n_dims,
                effective_budget,
                sigma_init=sigma_init,
                seed=seed,
            )
        elif algo == "simulated-annealing":
            result = simulated_annealing(
                _eval_fn,
                goal_instance.bounds,
                goal_instance.n_dims,
                effective_budget,
                temp_init=temp_init,
                temp_final=temp_final,
                step_size=step_size,
                seed=seed,
            )
        elif algo == "random-search":
            result = random_search(
                _eval_fn,
                goal_instance.bounds,
                goal_instance.n_dims,
                effective_budget,
                seed=seed,
            )
        else:
            console.print(f"[red]Unknown algorithm: {algorithm}[/red]")
            console.print("Choices: cma-es, simulated-annealing, random-search")
            raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Optimization error: {e}[/red]")
        raise typer.Exit(1) from e

    table = Table(title="Optimization Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    # Un-negate for maximize goals (algorithms store negated value internally)
    display_value = (
        -result["best_value"]
        if goal_instance.metric_direction == "maximize"
        else result["best_value"]
    )
    table.add_row(goal_instance.metric_name, f"{display_value:.6f}")
    table.add_row("n_evals", str(result["n_evals"]))
    table.add_row("elapsed (s)", f"{result['elapsed']:.2f}")
    if "generations" in result:
        table.add_row("generations", str(result["generations"]))

    console.print(table)
    console.print(f"\n[dim]best_x:[/dim] {[round(v, 4) for v in result['best_x']]}")


if __name__ == "__main__":
    app()
