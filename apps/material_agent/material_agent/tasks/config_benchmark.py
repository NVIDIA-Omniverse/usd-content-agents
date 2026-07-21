# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration loading task for benchmark workflows.

NOTE: This is a compatibility shim for the old workflow system.
The unified config system (UnifiedPipelineConfigTask) is preferred.
"""

import logging
from pathlib import Path
from typing import Any

from world_understanding.agentic.config import (
    clone_config_containers,
    log_config_source,
)
from world_understanding.agentic.events import get_listener
from world_understanding.agentic.tasks import Task
from world_understanding.utils.credentials import (
    path_exists_with_safe_diagnostics,
    read_text_with_safe_diagnostics,
    redact_sensitive_path,
)

from material_agent.api.defaults import BENCHMARK_DEFAULTS
from material_agent.tasks.config_loader import (
    load_config_from_context,
    resolve_config_relative_path,
)

logger = logging.getLogger(__name__)


class BenchmarkConfigTask(Task):
    """Compatibility config task for benchmark workflows."""

    def __init__(self):
        """Initialize the benchmark config loading task."""
        self.name = "BenchmarkConfigLoading"
        self.description = "Load benchmark configuration from YAML file"

    def run(self, context: dict[str, Any], object_store=None) -> dict[str, Any]:
        """Load benchmark configuration.

        Args:
            context: Workflow context containing config_dict or config_path
            object_store: Optional object store (not used)

        Returns:
            Updated context with loaded configuration
        """
        # Get event listener (or logger fallback)
        listener = get_listener(context, logger_name=__name__)

        config, config_path = load_config_from_context(context)
        log_config_source(context, listener.info, label="benchmark")

        # Normalize every benchmark-owned path while the source anchor is still
        # available.  The config is an isolated copy, so callers keep their
        # original relative values.
        for field in ("dataset", "output_dir", "system_prompt_file"):
            if config.get(field):
                config[field] = resolve_config_relative_path(
                    config[field],
                    config_path,
                )

        # Pass through the config
        context["config"] = config
        context["dataset_path"] = config.get("dataset")
        context["output_dir"] = config.get("output_dir")
        context["vlm_config"] = config.get("vlm", {})
        context["llm_config"] = config.get("llm", {})
        if not config.get("llm_judge") and config.get("judge"):
            config["llm_judge"] = clone_config_containers(config["judge"])
        elif not config.get("llm_judge"):
            config["llm_judge"] = clone_config_containers(BENCHMARK_DEFAULTS["judge"])
        context["llm_judge_config"] = config.get("llm_judge", {})
        context["max_workers"] = config.get("max_workers", 64)
        allow_empty_predictions = config.get(
            "allow_empty_predictions",
            BENCHMARK_DEFAULTS["allow_empty_predictions"],
        )
        if not isinstance(allow_empty_predictions, bool):
            raise ValueError(
                "benchmark.allow_empty_predictions must be a boolean, got "
                f"{type(allow_empty_predictions).__name__}"
            )
        context["allow_empty_predictions"] = allow_empty_predictions

        # Load system prompt from file if system_prompt_file is provided
        system_prompt = config.get("system_prompt")
        system_prompt_file = config.get("system_prompt_file")

        if system_prompt_file and not system_prompt:
            # Load from file
            system_prompt_path = Path(system_prompt_file)
            if path_exists_with_safe_diagnostics(
                system_prompt_path,
                label="system prompt file",
            ):
                system_prompt = read_text_with_safe_diagnostics(
                    system_prompt_path,
                    label="system prompt file",
                )
                listener.info(
                    "Loaded system prompt from: "
                    f"{redact_sensitive_path(system_prompt_path)}"
                )
                # Also set it back in config so VLMInferenceTask can find it
                config["system_prompt"] = system_prompt
            else:
                listener.warning(
                    "System prompt file not found: "
                    f"{redact_sensitive_path(system_prompt_path)}, will use default"
                )

        context["system_prompt"] = system_prompt

        return context
