# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared utilities for asset agent workflows."""

from importlib.metadata import PackageNotFoundError, version
from typing import Any

from rich.console import Console
from rich.table import Table

console = Console()


# Version info
def get_version() -> str:
    try:
        return version("joint-agent")
    except PackageNotFoundError:
        return "0.0.1-dev"


def display_results(
    results: dict[str, Any], title: str = "Classification Results"
) -> None:
    """Display classification results in a formatted table.

    Args:
        results: Results to display
        title: Title to display (default: "Classification Results")
    """
    console.print(f"\n[bold cyan]{title}[/bold cyan]")
    console.print("=" * 50)

    table = Table(title="Results Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Entries", str(results.get("total_entries", 0)))
    table.add_row("Successful", str(results.get("successful", 0)))
    table.add_row("Failed", str(results.get("failed", 0)))

    if "output_path" in results:
        table.add_row("Output Path", str(results["output_path"]))

    console.print(table)


def format_prediction_output(prediction: dict, include_confidence: bool = True) -> dict:
    """Format a prediction for output.

    Args:
        prediction: Raw prediction from VLM
        include_confidence: Whether to include confidence scores

    Returns:
        Formatted prediction dictionary
    """
    output = {
        "id": prediction["id"],
        "image_path": prediction.get("image_path", ""),
        "classification": prediction.get("vlm_response", ""),
    }

    if include_confidence and "confidence" in prediction:
        output["confidence"] = prediction["confidence"]

    return output
