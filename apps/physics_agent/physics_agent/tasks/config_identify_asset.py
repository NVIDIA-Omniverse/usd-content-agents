# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration task for the identify_asset pipeline step.

Loads configuration and sets up render config, USD path, and output directory
so that the downstream ``RenderScenePreviewTask`` can render lightweight
whole-scene preview images for asset identification via VLM.
"""

import logging
from pathlib import Path
from typing import Any

from world_understanding.agentic.config import load_config_mapping_from_context
from world_understanding.agentic.tasks import Task
from world_understanding.utils.credentials import (
    create_directory_with_safe_diagnostics,
    redact_sensitive_config,
    redact_sensitive_path,
    resolve_path_with_safe_diagnostics,
)
from world_understanding.utils.object_store import ObjectStore

from physics_agent.api.defaults import IDENTIFY_ASSET_DEFAULTS, apply_defaults

logger = logging.getLogger(__name__)


class IdentifyAssetConfigTask(Task):
    """Load and validate configuration for asset identification.

    Input context keys:
        - config_path: Path to YAML config file

    Output context keys:
        - config: Full configuration dictionary
        - usd_path: Path to the USD file for rendering
        - render_config: Render configuration for RenderScenePreviewTask
        - output_dir: Output directory for identification results
        - vlm_config: VLM configuration
        - identify_system_prompt: System prompt for identification
    """

    def __init__(self):
        """Initialize the config task."""
        self.name = "IdentifyAssetConfig"
        self.description = "Load configuration for asset identification"

    def run(
        self, context: dict[str, Any], object_store: ObjectStore | None = None
    ) -> dict[str, Any]:
        """Load and validate configuration.

        Args:
            context: Workflow context
            object_store: Optional object store

        Returns:
            Updated context with configuration
        """
        config = self._load_config(context)

        # Apply defaults
        config = apply_defaults(config, IDENTIFY_ASSET_DEFAULTS)

        # Resolve paths
        config_path = context.get("config_path")
        if config_path:
            config_dir = Path(config_path).parent
        else:
            config_dir = Path.cwd()

        # Resolve USD path for preview rendering
        usd_path = config.get("usd_path")
        if not usd_path:
            raise ValueError("usd_path is required for asset identification")
        usd_path = str(self._resolve_path(usd_path, config_dir))

        # Resolve output directory
        output_dir = config.get("output_dir")
        if output_dir:
            output_dir = self._resolve_path(output_dir, config_dir)
        else:
            output_dir = config_dir / "identification"
        create_directory_with_safe_diagnostics(
            output_dir,
            label="identify_asset output directory",
            parents=True,
            exist_ok=True,
        )

        # Extract prompts
        prompts = config.get("prompts", {})
        system_prompt = prompts.get(
            "system", IDENTIFY_ASSET_DEFAULTS["prompts"]["system"]
        )

        # Build render_config for RenderScenePreviewTask (self-contained)
        renderer_raw = config.get("renderer", IDENTIFY_ASSET_DEFAULTS["renderer"])
        render_config = {
            "backend": renderer_raw.get("backend", "remote"),
            "image_width": renderer_raw.get("image_width", 512),
            "image_height": renderer_raw.get("image_height", 512),
            "cameras": renderer_raw.get(
                "cameras", ["+x+y+z", "-x+y+z", "-x-y+z", "+x-y+z"]
            ),
            "camera_margin": renderer_raw.get("camera_margin", 3.0),
            "background_color": renderer_raw.get("background_color", [0.0, 0.0, 0.0]),
            "should_reset_materials": renderer_raw.get("should_reset_materials", False),
            "use_lights": renderer_raw.get("use_lights", False),
            "flatten_before_render": renderer_raw.get("flatten_before_render", False),
        }

        # Update context
        context["config"] = config
        context.update(
            {
                "usd_path": usd_path,
                "render_config": render_config,
                "output_dir": str(output_dir),
                "vlm_config": config.get("vlm", {}),
                "llm_config": config.get("llm", {}),
                "identify_system_prompt": system_prompt,
            }
        )

        logger.info("Loaded identification configuration")
        logger.info("USD path: %s", redact_sensitive_path(usd_path))
        logger.info("Output directory: %s", redact_sensitive_path(output_dir))
        logger.info(
            "Renderer backend: %s",
            redact_sensitive_config(render_config.get("backend", "remote")),
        )

        return context

    def _load_config(self, context: dict[str, Any]) -> dict[str, Any]:
        """Load configuration from file or dict."""
        config, _ = load_config_mapping_from_context(
            context,
            allow_empty=context.get("config_dict") is not None,
            missing_path_message="No config_path or config_dict in context",
            parse_error_message="Unable to parse configuration file: {config_path}",
        )
        return config

    def _resolve_path(self, path: str, config_dir: Path) -> Path:
        """Resolve path relative to config directory."""
        path_obj = Path(path)
        if path_obj.is_absolute():
            return path_obj
        return resolve_path_with_safe_diagnostics(
            config_dir / path_obj,
            label="identify_asset configuration path",
        )
