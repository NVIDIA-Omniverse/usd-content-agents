# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare dataset configuration task for Physics Agent."""

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


class PrepareDatasetConfigTask(Task):
    """Load and validate prepare dataset configuration.

    Input context keys:
        - config_path: Path to YAML config file
        OR
        - config_dict: Configuration dictionary

    Output context keys:
        - usd_dir: Path to USD dataset directory
        - dataset: Path to output dataset directory
        - models: List of model subdirectories to process
        - reference_images: List of reference images
        - prompts: Prompt configuration
    """

    def __init__(self):
        """Initialize the config task."""
        self.name = "PrepareDatasetConfig"
        self.description = "Load and validate prepare dataset configuration"

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

        # USD directory (input from build_dataset_usd)
        usd_dir = config.get("usd_dir")
        if usd_dir:
            usd_dir = self._resolve_path(usd_dir, config_dir)
        else:
            raise ValueError("No usd_dir specified in configuration")

        # Dataset output directory
        dataset = config.get("dataset")
        if dataset:
            dataset = self._resolve_path(dataset, config_dir)
        else:
            dataset = usd_dir.parent / "dataset"
        create_directory_with_safe_diagnostics(
            dataset,
            label="prepare_dataset output directory",
            parents=True,
            exist_ok=True,
        )

        # Models list
        models = config.get("models", ["."])

        # Reference images
        reference_images = config.get("reference_images", [])
        if reference_images:
            reference_images = [
                str(self._resolve_path(img, config_dir)) for img in reference_images
            ]

        # Prompts configuration
        prompts = config.get("prompts", {})

        # Update context
        context["config"] = config  # May be needed by downstream tasks
        context.update(
            {
                "usd_dir": str(usd_dir),
                "dataset_path": str(dataset),  # Used by PrepareDatasetTask
                "models": models,
                "reference_images": reference_images,
                "prompts": prompts,
                "include_prim_path_context": config.get(
                    "include_prim_path_context", True
                ),
                "include_geometric_context": config.get(
                    "include_geometric_context", True
                ),
            }
        )

        logger.info("Loaded configuration for prepare dataset")
        logger.info("USD directory: %s", redact_sensitive_path(usd_dir))
        logger.info("Output dataset: %s", redact_sensitive_path(dataset))
        logger.info(
            "Models: %s",
            [redact_sensitive_path(model) for model in models],
        )

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
            allow_empty=context.get("config_dict") is not None,
            missing_path_message="No config_path or config_dict in context",
            parse_error_message="Unable to parse configuration file: {config_path}",
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
            label="prepare_dataset configuration path",
        )
