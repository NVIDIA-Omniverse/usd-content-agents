# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Grounding DINO tool for zero-shot object detection."""

import logging
import os
from typing import Any

from pydantic import Field
from rich.console import Console
from rich.table import Table

from world_understanding.functions.cv.grounding_dino import (
    detect_objects_with_grounding_dino,
)
from world_understanding.tools.base import (
    ExecutionPolicy,
    ToolInput,
    ToolOutput,
    register_tool,
)

logger = logging.getLogger(__name__)


class GroundingDinoInput(ToolInput):
    """Input for Grounding DINO object detection tool."""

    image_path: str = Field(
        ...,
        description="Path to the image to analyze",
    )
    prompt: str = Field(
        ...,
        description="Text description of objects to detect (e.g., 'red pot', 'person wearing hat')",
    )
    threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for detections",
    )
    api_key: str | None = Field(
        default=None,
        description="NVIDIA API key (uses NVIDIA_API_KEY env var if not provided)",
    )


class DetectionResult(ToolOutput):
    """Single detection result."""

    phrase: str = Field(..., description="Detected object phrase")
    bboxes: list[list[int]] = Field(
        ..., description="List of bounding boxes [x, y, width, height]"
    )
    confidence: list[float] = Field(..., description="Confidence scores for each bbox")


class GroundingDinoOutput(ToolOutput):
    """Output for Grounding DINO object detection tool."""

    detections: list[DetectionResult] = Field(
        ...,
        description="List of detected objects with bounding boxes",
    )
    total_detections: int = Field(
        ...,
        description="Total number of objects detected",
    )
    image_width: int = Field(..., description="Width of the analyzed image")
    image_height: int = Field(..., description="Height of the analyzed image")


def _display_detection_results(
    outputs: dict[str, Any], console: Console, indent: str = ""
) -> None:
    """Display detection results in a formatted table."""
    console.print(f"\n{indent}[bold]Grounding DINO Detection Results[/bold]")

    if outputs["total_detections"] == 0:
        console.print(f"{indent}No objects detected matching the prompt.")
        return

    console.print(f"{indent}Total detections: {outputs['total_detections']}")
    console.print(
        f"{indent}Image size: {outputs['image_width']}x{outputs['image_height']}"
    )

    # Create table for detections
    table = Table(title="Detected Objects", show_header=True, header_style="bold cyan")
    table.add_column("Phrase", style="cyan")
    table.add_column("Bounding Box (x,y,w,h)", style="green")
    table.add_column("Confidence", style="yellow")

    for detection in outputs["detections"]:
        for bbox, conf in zip(
            detection["bboxes"], detection["confidence"], strict=False
        ):
            table.add_row(
                detection["phrase"],
                f"[{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]",
                f"{conf:.3f}",
            )

    console.print(table)


@register_tool(
    name="grounding_dino",
    version="0.1.0",
    description="Detect objects in images using Grounding DINO zero-shot detection",
    tags=["vision", "detection", "object-detection", "zero-shot", "gpu"],
    input_model=GroundingDinoInput,
    output_model=GroundingDinoOutput,
    policy=ExecutionPolicy(timeout_s=60.0, device="cuda"),
)
def grounding_dino_tool(inputs: GroundingDinoInput) -> GroundingDinoOutput:
    """Detect objects in images using Grounding DINO."""
    # Get API key (prefer explicit input; fall back to environment)
    api_key = inputs.api_key or os.environ.get("NVIDIA_API_KEY")

    if not api_key:
        raise ValueError(
            "NVIDIA API key required. Set via parameter or NVIDIA_API_KEY env var."
        )

    try:
        # Call the function
        result = detect_objects_with_grounding_dino(
            image=inputs.image_path,
            prompt=inputs.prompt,
            threshold=inputs.threshold,
            api_key=api_key,
        )

        # Convert to output format
        detections = [
            DetectionResult(
                phrase=det["phrase"], bboxes=det["bboxes"], confidence=det["confidence"]
            )
            for det in result["detections"]
        ]

        return GroundingDinoOutput(
            detections=detections,
            total_detections=result["total_detections"],
            image_width=result["image_size"][0],
            image_height=result["image_size"][1],
        )

    except Exception as e:
        logger.error(f"Grounding DINO detection failed: {e}")
        raise


# Attach display function to the tool
grounding_dino_tool._display_function = _display_detection_results
