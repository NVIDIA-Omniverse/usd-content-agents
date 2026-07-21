# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration loading task for validate_predictions step."""

import logging
from typing import Any

from world_understanding.agentic.events import get_listener
from world_understanding.agentic.tasks import Task
from world_understanding.utils.credentials import redact_sensitive_path

from material_agent.tasks.config_loader import load_config_from_context

logger = logging.getLogger(__name__)


class ValidatePredictionsConfigTask(Task):
    """Load and validate configuration for prediction validation step.

    Input context keys:
        - config_path: Path to YAML config file

    Output context keys:
        - predictions_path: Path to predictions JSONL file
        - material_names: List of valid material names
        - llm_config: Optional LLM config for repair
        - allow_unknown_material: Whether the "__UNKNOWN__" sentinel is accepted
    """

    def run(self, context: dict[str, Any], object_store=None) -> dict[str, Any]:
        listener = get_listener(context, logger_name=__name__)

        config, _ = load_config_from_context(
            context,
            missing_path_message="config_path is required in context",
            empty_message="Empty configuration",
        )
        if context.get("config_dict") is not None:
            listener.info("Using in-memory validate_predictions config")
        else:
            listener.info("Loading validate_predictions config from file")

        # predictions_path is required (auto-wired by executor)
        if "predictions_path" not in config:
            raise ValueError(
                "predictions_path is required in validate_predictions config"
            )
        context["predictions_path"] = config["predictions_path"]

        # material_names — list of valid names
        if "material_names" not in config:
            raise ValueError(
                "material_names is required in validate_predictions config"
            )
        context["material_names"] = config["material_names"]

        # Optional LLM config for repair
        if "llm" in config:
            context["llm_config"] = config["llm"]

        allow_unknown_material = config.get("allow_unknown_material", True)
        if not isinstance(allow_unknown_material, bool):
            raise ValueError(
                "allow_unknown_material must be a boolean, got "
                f"{type(allow_unknown_material).__name__}"
            )
        context["allow_unknown_material"] = allow_unknown_material

        listener.info(
            f"Predictions: {redact_sensitive_path(context['predictions_path'])}"
        )
        listener.info(f"Material library: {len(context['material_names'])} entries")

        return context
