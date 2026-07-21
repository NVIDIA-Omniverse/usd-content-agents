# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Material Agent adapter for validation-core visual grounding."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from world_understanding.agentic.events import get_listener
from world_understanding.agentic.tasks import Task
from world_understanding.validation import (
    visual_grounding as validation_visual_grounding,
)

__all__ = ["VisualGroundingTask"]


class VisualGroundingTask(Task):
    """Generate visual-to-USD grounding artifacts for harness inspection.

    Input context keys:
        - output_usd_path / usd_path: materialized USD to inspect.
        - rendered_image_path / rendered_image_paths: optional beauty render.
        - visual_grounding_config: optional settings dictionary.

    Output context keys:
        - visual_grounding_packet: packet dictionary.
        - visual_grounding_packet_path: legend JSON path.
        - visual_grounding_html_path: HTML report path.
    """

    def __init__(self) -> None:
        self.name = "VisualGrounding"
        self.description = "Generate object-ID overlays and visible prim legends"

    def run(self, context: dict[str, object], object_store=None) -> dict[str, object]:
        del object_store
        listener = get_listener(context, logger_name=__name__)
        raw_config = context.get("visual_grounding_config")
        if raw_config is None:
            config: dict[str, object] = {}
        elif isinstance(raw_config, Mapping):
            config = dict(raw_config)
        else:
            raise TypeError("visual_grounding_config must be a mapping if provided")

        usd_path = (
            config.get("usd_path")
            or context.get("output_usd_path")
            or context.get("usd_path")
        )
        if not usd_path:
            raise ValueError(
                "VisualGroundingTask requires output_usd_path, usd_path, or "
                "visual_grounding_config.usd_path"
            )

        raw_output_dir = config.get("output_dir")
        if raw_output_dir:
            output_dir = Path(str(raw_output_dir))
        else:
            output_dir = (
                Path(str(usd_path)).expanduser().resolve().parent / "visual_grounding"
            )

        beauty_image_path = config.get("beauty_image_path")
        if not beauty_image_path:
            rendered_paths = context.get("rendered_image_paths")
            if isinstance(rendered_paths, list) and rendered_paths:
                beauty_image_path = rendered_paths[0]
            else:
                beauty_image_path = context.get("rendered_image_path")

        listener.info("Generating visual grounding packet...")
        packet = validation_visual_grounding.generate_visual_grounding_packet(
            usd_path=Path(str(usd_path)),
            output_dir=output_dir,
            prim_path=config.get("prim_path")
            or config.get("root_prim_path")
            or context.get("root_prim_path")
            or context.get("prim_path"),
            beauty_image_path=Path(str(beauty_image_path))
            if beauty_image_path
            else None,
            direction=str(
                config.get("direction", config.get("camera_direction", "+x+y+z"))
            ),
            width=config.get("width"),
            height=config.get("height"),
            rasterizer=str(config.get("rasterizer", "cpu")),
            device=str(config.get("device", "cuda:0")),
            camera_margin=float(config.get("camera_margin", 1.0)),
            focal_length=float(config.get("focal_length", 50.0)),
            horizontal_aperture=float(config.get("horizontal_aperture", 36.0)),
            vertical_aperture=float(config.get("vertical_aperture", 36.0)),
            max_labels=int(config.get("max_labels", 32)),
            label_mode=str(config.get("label_mode", "callout")),
            min_visible_pixels=int(config.get("min_visible_pixels", 64)),
        )

        artifacts = packet["artifacts"]
        context["visual_grounding_packet"] = packet
        context["visual_grounding_packet_path"] = artifacts["legend_json_path"]
        context["visual_grounding_html_path"] = artifacts["html_report_path"]
        context["visual_grounding_overlay_paths"] = {
            key: value
            for key, value in artifacts.items()
            if key.endswith("_path") and value is not None
        }
        listener.info(
            "Visual grounding packet generated: "
            f"{len(packet['visible_entries'])} visible entries"
        )
        return context
