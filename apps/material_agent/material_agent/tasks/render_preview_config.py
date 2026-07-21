# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task for loading render-preview configuration from YAML.

This thin config task maps the material agent step config keys to the
context keys expected by ``RenderScenePreviewTask``.
"""

import logging
from pathlib import Path
from typing import Any

from world_understanding.agentic.config import log_config_source
from world_understanding.agentic.events import get_listener
from world_understanding.agentic.tasks import Task
from world_understanding.utils.credentials import (
    create_directory_with_safe_diagnostics,
    path_exists_with_safe_diagnostics,
    redact_sensitive_config,
    redact_sensitive_path,
    resolve_path_with_safe_diagnostics,
)

from material_agent.tasks.config_loader import load_config_from_context

logger = logging.getLogger(__name__)


class RenderPreviewConfigTask(Task):
    """Load render-preview configuration.

    Mirrors ``RenderConfigTask`` but targets the ``render_preview`` step,
    which uses :class:`RenderScenePreviewTask` from the shared library.

    Input context keys:
        - config_dict: In-memory step configuration (preferred)
        - config_path: YAML path or relative-path anchor

    Output context keys:
        - usd_path: Path to the USD file to render
        - output_dir: Directory for preview images
        - render_config: Dictionary consumed by RenderScenePreviewTask
    """

    def __init__(self) -> None:
        self.name = "RenderPreviewConfig"
        self.description = "Load render-preview configuration"

    def run(self, context: dict[str, Any], object_store=None) -> dict[str, Any]:
        """Load and validate the render-preview configuration.

        Args:
            context: Workflow context
            object_store: Optional object store (not used)

        Returns:
            Updated context with render preview configuration
        """
        listener = get_listener(context, logger_name=__name__)
        config, config_path = load_config_from_context(
            context, missing_path_message="config_path is required in context"
        )
        log_config_source(context, listener.info, label="render-preview")

        # The config is already extracted by the unified pipeline executor
        # so it's a flat dict with all keys at the top level.
        usd_path = config.get("usd_path")
        if not usd_path:
            raise ValueError("usd_path not specified in render_preview config")

        usd_path = Path(usd_path)
        if not usd_path.is_absolute():
            usd_path = resolve_path_with_safe_diagnostics(
                config_path.parent / usd_path,
                label="render-preview USD path",
            )

        if not path_exists_with_safe_diagnostics(
            usd_path,
            label="render-preview USD path",
        ):
            raise FileNotFoundError(
                f"USD file not found: {redact_sensitive_path(usd_path)}"
            )

        output_dir = config.get("output_dir")
        if output_dir:
            output_dir = Path(output_dir)
            if not output_dir.is_absolute():
                output_dir = resolve_path_with_safe_diagnostics(
                    config_path.parent / output_dir,
                    label="render-preview output directory",
                )
        else:
            output_dir = usd_path.parent / "preview"
        create_directory_with_safe_diagnostics(
            output_dir,
            label="render-preview output directory",
        )

        # Build render_config dict for RenderScenePreviewTask
        render_config: dict[str, Any] = {
            "backend": config.get("backend", "remote"),
            "image_width": config.get("image_width", 512),
            "image_height": config.get("image_height", 512),
            "cameras": config.get("cameras", ["+x+y+z"]),
            "camera_margin": config.get("camera_margin", 1.0),
            "background_color": config.get("background_color", [1.0, 1.0, 1.0]),
            "should_reset_materials": config.get("should_reset_materials", True),
            "use_lights": config.get("use_lights", True),
            "flatten_before_render": config.get("flatten_before_render", False),
        }
        if "material_target" in config:
            render_config["material_target"] = config["material_target"]

        listener.info(f"  USD: {redact_sensitive_path(usd_path)}")
        listener.info(f"  Output: {redact_sensitive_path(output_dir)}")
        listener.info(f"  Backend: {redact_sensitive_config(render_config['backend'])}")
        listener.info(
            f"  Size: {render_config['image_width']}x{render_config['image_height']}"
        )
        listener.info(f"  Cameras: {render_config['cameras']}")

        context["usd_path"] = str(usd_path)
        context["output_dir"] = str(output_dir)
        context["render_config"] = render_config

        # Pass through prim_filters if present (same schema as build_dataset_usd)
        prim_filters = config.get("prim_filters")
        if prim_filters:
            context["prim_filters"] = prim_filters
            listener.info(f"  Prim filters: {prim_filters}")

        return context
