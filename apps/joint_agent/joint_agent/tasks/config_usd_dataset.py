# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""USD dataset configuration task for Joint Agent."""

import logging
from pathlib import Path
from typing import Any

from world_understanding.agentic.config import load_config_mapping_from_context
from world_understanding.agentic.tasks import Task
from world_understanding.utils.credentials import (
    create_directory_with_safe_diagnostics,
    redact_sensitive_path,
    resolve_path_with_safe_diagnostics,
)
from world_understanding.utils.object_store import ObjectStore

logger = logging.getLogger(__name__)


class USDDatasetConfigTask(Task):
    """Load and validate USD dataset building configuration.

    Input context keys:
        - config_path: Path to YAML config file
        OR
        - config_dict: Configuration dictionary

    Output context keys:
        - usd_path: Path to input USD file
        - output_dir: Path to output directory
        - renderer: Renderer configuration
        - prim_filters: Prim filter configuration
    """

    def __init__(self):
        """Initialize the config task."""
        self.name = "USDDatasetConfig"
        self.description = "Load and validate USD dataset configuration"

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

        # Resolve paths
        config_path = context.get("config_path")
        if config_path:
            config_dir = Path(config_path).parent
        else:
            config_dir = Path.cwd()

        # USD path
        usd_path = config.get("usd_path")
        if usd_path:
            usd_path = self._resolve_path(usd_path, config_dir)
        else:
            raise ValueError("No usd_path specified in configuration")

        # Output directory
        output_dir = config.get("output_dir")
        if output_dir:
            output_dir = self._resolve_path(output_dir, config_dir)
        else:
            output_dir = usd_path.parent / "dataset" / "usd"
        create_directory_with_safe_diagnostics(
            output_dir,
            label="USD dataset output directory",
            parents=True,
            exist_ok=True,
        )

        # Renderer configuration
        renderer = config.get("renderer", {})

        # Prim filters
        prim_filters = config.get("prim_filters", {})

        # Update context
        context.update(
            {
                "usd_path": str(usd_path),
                "output_dir": str(output_dir),
                "renderer": renderer,
                "prim_filters": prim_filters,
                "extract_hierarchy": config.get("extract_hierarchy", True),
                "extract_metadata": config.get("extract_metadata", True),
                "skip_existing": config.get("skip_existing", True),
                "batch_size": config.get("batch_size", 4),
                "num_workers": config.get("num_workers", 32),
            }
        )

        logger.info("Loaded configuration for USD dataset building")
        logger.info("USD path: %s", redact_sensitive_path(usd_path))
        logger.info("Output directory: %s", redact_sensitive_path(output_dir))

        return context

    def _load_config(self, context: dict[str, Any]) -> dict[str, Any]:
        """Load configuration from file or dict.

        Args:
            context: Workflow context

        Returns:
            Configuration dictionary
        """
        config, _ = load_config_mapping_from_context(
            context,
            allow_empty=True,
            missing_path_message="No config_path or config_dict in context",
            config_dict_non_mapping_message="config_dict must be a dictionary",
            file_non_mapping_message=("Configuration file must contain a dictionary"),
        )
        return config

    def _resolve_path(self, path: str, config_dir: Path) -> Path:
        """Resolve path relative to config directory.

        Args:
            path: Path string
            config_dir: Configuration directory

        Returns:
            Resolved Path
        """
        path_obj = Path(path)
        if path_obj.is_absolute():
            return path_obj
        return resolve_path_with_safe_diagnostics(
            config_dir / path_obj,
            label="USD dataset configuration path",
        )
