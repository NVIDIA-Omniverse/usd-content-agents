# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration loading task for validate_usd step."""

import logging
from typing import Any

from world_understanding.agentic.config import (
    load_config_mapping_from_context,
    log_config_source,
)
from world_understanding.agentic.events import get_listener
from world_understanding.agentic.tasks import Task
from world_understanding.agentic.usd_tasks.validate_usd import ON_FAILURE_MODES
from world_understanding.functions.graphics.validate_usd import (
    AVAILABLE_VALIDATION_CATEGORIES,
    DEFAULT_VALIDATION_CATEGORIES,
    normalize_validation_categories,
)
from world_understanding.utils.credentials import redact_sensitive_path

logger = logging.getLogger(__name__)

_INVALID_VALIDATION_CONFIG_MESSAGE = "validation_config must be a mapping"


class ValidateUSDConfigTask(Task):
    """Load and validate configuration for USD validation step.

    Input context keys:
        - config_dict: In-memory configuration dictionary (preferred)
        - config_path: Path to YAML config file (fallback)

    Output context keys:
        - input_usd_path: Path to input USD to validate
        - validation_config: Validation parameters
        - on_failure: "warn" | "block" | "fix"
        - output_dir: Directory for validation outputs
        - original_usd_path: (validate_output only) Path to original input USD
        - baseline_validation: (validate_output only) Cached baseline
    """

    def run(self, context: dict[str, Any], object_store: Any = None) -> dict[str, Any]:
        """Load validation configuration.

        Args:
            context: Workflow context with config_path
            object_store: Optional object store (not used)

        Returns:
            Updated context with configuration values

        Raises:
            FileNotFoundError: If config file not found
            ValueError: If required fields are missing
        """
        listener = get_listener(context)

        config = self._load_config(context, listener)

        # Extract paths (already resolved by UnifiedPipelineConfigTask)
        if "input_usd_path" not in config:
            raise ValueError("input_usd_path is required in validate_usd config")

        context["input_usd_path"] = config["input_usd_path"]

        if "output_dir" in config:
            context["output_dir"] = config["output_dir"]

        # Pass through original_usd_path and baseline_validation
        # (injected by pipeline executor for validate_output)
        if "original_usd_path" in config:
            context["original_usd_path"] = config["original_usd_path"]
        if "baseline_validation" in config:
            context["baseline_validation"] = config["baseline_validation"]

        # on_failure mode
        on_failure = config.get("on_failure", "warn")
        if on_failure not in ON_FAILURE_MODES:
            raise ValueError(
                f"Invalid on_failure mode. Must be one of {ON_FAILURE_MODES}"
            )
        context["on_failure"] = on_failure

        # Build validation config
        validation_config = config.get("validation_config", {})
        if not isinstance(validation_config, dict):
            raise ValueError(_INVALID_VALIDATION_CONFIG_MESSAGE) from None

        # Ensure categories have defaults
        if "categories" not in validation_config:
            validation_config["categories"] = list(DEFAULT_VALIDATION_CATEGORIES)
        else:
            validation_config["categories"] = normalize_validation_categories(
                list(validation_config["categories"])
            )

        # Validate categories
        invalid = [
            c
            for c in validation_config.get("categories", [])
            if c not in AVAILABLE_VALIDATION_CATEGORIES
        ]
        if invalid:
            raise ValueError(
                "Unknown validation categories. "
                f"Available: {AVAILABLE_VALIDATION_CATEGORIES}"
            )

        # Default poll_seconds
        if "poll_seconds" not in validation_config:
            validation_config["poll_seconds"] = 300

        context["validation_config"] = validation_config

        safe_input_path = redact_sensitive_path(context["input_usd_path"])
        listener.info(f"Input USD: {safe_input_path}")
        cats = ", ".join(validation_config.get("categories", []))
        listener.info(f"Categories: {cats}")
        listener.info(f"On failure: {on_failure}")

        return context

    def _load_config(self, context: dict[str, Any], listener: Any) -> dict[str, Any]:
        """Load an isolated mapping without rendering configuration values."""
        config, _ = load_config_mapping_from_context(
            context,
            missing_path_message="config_path is required in context",
            missing_file_message="Configuration file not found: {config_path}",
            read_error_message=(
                "Unable to read validate_usd configuration file: {config_path}"
            ),
            parse_error_message=(
                "Unable to parse validate_usd configuration file: {config_path}"
            ),
            empty_message=(
                "Empty validate_usd configuration dictionary"
                if context.get("config_dict") is not None
                else "Empty configuration file: {config_path}"
            ),
            config_dict_non_mapping_message=(
                "validate_usd config_dict must be a mapping"
            ),
            file_non_mapping_message=("validate_usd configuration must be a mapping"),
        )
        log_config_source(context, listener.info, label="validate_usd")
        return config
