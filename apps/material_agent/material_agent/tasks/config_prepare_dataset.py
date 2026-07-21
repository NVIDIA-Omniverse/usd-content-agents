# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration loading task for prepare dataset workflows.

NOTE: This is a compatibility shim for the old workflow system.
The unified config system (UnifiedPipelineConfigTask) is preferred.
"""

import logging
from pathlib import Path
from typing import Any

from world_understanding.agentic.config import log_config_source
from world_understanding.agentic.events import get_listener
from world_understanding.agentic.tasks import Task
from world_understanding.utils.credentials import (
    path_exists_with_safe_diagnostics,
    redact_sensitive_path,
)

from material_agent.tasks.config_loader import load_config_from_context

logger = logging.getLogger(__name__)


class PrepareDatasetConfigTask(Task):
    """Compatibility config task for prepare dataset workflows."""

    def __init__(self):
        """Initialize the prepare dataset config loading task."""
        self.name = "PrepareDatasetConfigLoading"
        self.description = "Load prepare dataset configuration from YAML file"

    def run(self, context: dict[str, Any], object_store=None) -> dict[str, Any]:
        """Load prepare dataset configuration.

        Args:
            context: Workflow context containing config_dict or config_path
            object_store: Optional object store (not used)

        Returns:
            Updated context with loaded configuration
        """
        # Get event listener (or logger fallback)
        listener = get_listener(context, logger_name=__name__)

        config, config_path = load_config_from_context(context)
        log_config_source(context, listener.info, label="prepare dataset")

        # Pass through the config
        context["config"] = config
        context["usd_dir"] = Path(config.get("usd_dir", ""))
        context["vector_store_path"] = (
            Path(config["vector_store"]) if config.get("vector_store") else None
        )
        context["dataset_path"] = Path(config.get("dataset", ""))
        context["config_path"] = config_path

        # Use models from config if provided, otherwise discover from usd_dir
        if "models" in config and config["models"]:
            context["models"] = config["models"]
            models = config["models"]
            model_count = len(models) if isinstance(models, list | tuple | dict) else 1
            listener.info(f"Using {model_count} model entries from config")
        else:
            # Discover models from usd_dir (legacy behavior)
            usd_dir = context["usd_dir"]
            if usd_dir and path_exists_with_safe_diagnostics(
                usd_dir,
                label="USD model directory",
            ):
                models = self._discover_models_from_usd_dir(usd_dir)
                context["models"] = models
                listener.info(f"Discovered {len(models)} models from usd_dir")
            else:
                context["models"] = []
                listener.warning("No models found - usd_dir doesn't exist")

        return context

    def _discover_models_from_usd_dir(self, usd_dir: Path) -> list[str]:
        """Discover model numbers from USD directory structure."""
        if not path_exists_with_safe_diagnostics(
            usd_dir,
            label="USD model directory",
        ):
            return []

        models = []
        try:
            items = list(usd_dir.iterdir())
        except OSError as error:
            raise type(error)(
                error.errno,
                "Unable to list USD model directory",
                redact_sensitive_path(usd_dir),
            ) from None
        for item in items:
            try:
                is_directory = item.is_dir()
            except OSError:
                logger.warning(
                    "Unable to inspect USD model entry: %s",
                    redact_sensitive_path(item),
                )
                continue
            if is_directory:
                dataset_json = item / "dataset.json"
                prims_jsonl = item / "prims.jsonl"
                usd_model_json = item / "usd_model.json"

                if (
                    path_exists_with_safe_diagnostics(
                        dataset_json,
                        label="model dataset metadata",
                    )
                    and path_exists_with_safe_diagnostics(
                        prims_jsonl,
                        label="model prim metadata",
                    )
                    and path_exists_with_safe_diagnostics(
                        usd_model_json,
                        label="model USD metadata",
                    )
                ):
                    models.append(item.name)

        models.sort()
        return models
