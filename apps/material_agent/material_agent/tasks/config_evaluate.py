# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration loading task for evaluation workflows.

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
    resolve_path_with_safe_diagnostics,
)

from material_agent.tasks.config_loader import load_config_from_context

logger = logging.getLogger(__name__)


class EvaluateConfigTask(Task):
    """Compatibility config task for evaluation workflows."""

    def __init__(self):
        """Initialize the evaluate config loading task."""
        self.name = "EvaluateConfigLoading"
        self.description = "Load evaluation configuration from YAML file"

    def run(self, context: dict[str, Any], object_store=None) -> dict[str, Any]:
        """Load evaluation configuration.

        Args:
            context: Workflow context containing config_dict or config_path
            object_store: Optional object store (not used)

        Returns:
            Updated context with loaded configuration
        """
        # Get event listener (or logger fallback)
        listener = get_listener(context, logger_name=__name__)

        config, config_path = load_config_from_context(context)
        log_config_source(context, listener.info, label="evaluation")

        # Resolve paths - try both relative to config dir and relative to cwd
        config_dir = resolve_path_with_safe_diagnostics(
            config_path.parent,
            label="evaluation configuration directory",
        )

        def resolve_path(path_str: str | None) -> Path | None:
            """Resolve a path, trying both relative to config dir and cwd."""
            if not path_str:
                return None
            path = Path(path_str)
            if path.is_absolute():
                return path
            # Try relative to cwd first (more common for result paths)
            cwd_path = Path.cwd() / path
            if path_exists_with_safe_diagnostics(
                cwd_path,
                label="evaluation input path",
            ):
                return resolve_path_with_safe_diagnostics(
                    cwd_path,
                    label="evaluation input path",
                )
            # Fall back to relative to config dir
            config_relative = config_dir / path
            return resolve_path_with_safe_diagnostics(
                config_relative,
                label="evaluation input path",
            )

        # Pass through the config
        context["config"] = config

        # Resolve predictions_path
        context["predictions_path"] = resolve_path(config.get("predictions_path"))

        # Resolve dataset_path
        context["dataset_path"] = resolve_path(config.get("dataset_path"))

        context["llm_judge_config"] = config.get("llm_judge", {})

        # Resolve output_dir
        context["output_dir"] = resolve_path(config.get("output_dir"))

        return context
