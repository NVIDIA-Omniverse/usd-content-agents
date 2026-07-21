# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration loading task for optimize_usd step."""

import logging
from typing import Any

from pydantic import ValidationError

from world_understanding.agentic.config import (
    load_config_mapping_from_context,
    log_config_source,
)
from world_understanding.agentic.events import get_listener
from world_understanding.agentic.tasks import Task
from world_understanding.agentic.usd_tasks.optimizer_models import (
    SceneOptimizerSettings,
)
from world_understanding.utils.credentials import redact_sensitive_config
from world_understanding.utils.object_store import ObjectStore

logger = logging.getLogger(__name__)

_INVALID_OPTIMIZER_SETTINGS_MESSAGE = "Invalid scene_optimizer_settings in config"
_INVALID_OPTIMIZATION_CONFIG_MESSAGE = "optimization_config must be a mapping"
_INVALID_OPTIMIZATION_BACKEND_MESSAGE = (
    "Invalid optimization backend; expected 'local' or 'remote'"
)
_INVALID_FLATTEN_PROTOTYPES_MESSAGE = "flatten_prototypes must be a boolean"


class OptimizeUSDConfigTask(Task):
    """Load and validate configuration for USD optimization step.

    Input context keys:
        - config_dict: In-memory configuration dictionary (preferred)
        - config_path: Path to YAML config file (fallback)

    Output context keys:
        - input_usd_path: Path to input USD
        - output_usd_path: Path for optimized USD
        - optimization_config: API-specific parameters (optional)
    """

    def run(
        self,
        context: dict[str, Any],
        object_store: ObjectStore | None = None,
    ) -> dict[str, Any]:
        """Load optimization configuration.

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

        # Validate required fields
        if "input_usd_path" not in config:
            raise ValueError("input_usd_path is required in optimize_usd config")
        if "output_usd_path" not in config:
            raise ValueError("output_usd_path is required in optimize_usd config")

        # Extract paths (already resolved by UnifiedPipelineConfigTask)
        context["input_usd_path"] = config["input_usd_path"]
        context["output_usd_path"] = config["output_usd_path"]

        # Get optimization config
        optimization_config = config.get("optimization_config", {})
        if not isinstance(optimization_config, dict):
            listener.error(_INVALID_OPTIMIZATION_CONFIG_MESSAGE)
            raise ValueError(_INVALID_OPTIMIZATION_CONFIG_MESSAGE) from None

        # Parse and validate scene_optimizer_settings if present
        if "scene_optimizer_settings" in optimization_config:
            settings_data = optimization_config["scene_optimizer_settings"]
            if not isinstance(settings_data, dict) or any(
                not isinstance(key, str) for key in settings_data
            ):
                listener.error(_INVALID_OPTIMIZER_SETTINGS_MESSAGE)
                raise ValueError(_INVALID_OPTIMIZER_SETTINGS_MESSAGE) from None
            try:
                settings_model = SceneOptimizerSettings(**settings_data)

                # Convert to dict with snake_case (matches client behavior)
                validated_settings = settings_model.model_dump(
                    by_alias=False, exclude_none=True
                )

                # Replace with validated settings
                optimization_config["scene_optimizer_settings"] = validated_settings

                # Build list of enabled operations (matches client lines 744-750)
                enabled_ops = self._build_enabled_operations(validated_settings)

                # Validate at least one operation (matches client lines 752-755)
                if not enabled_ops:
                    listener.error("At least one operation must be enabled")
                    raise ValueError(
                        "At least one operation must be enabled in scene_optimizer_settings"
                    )

                # Log configuration (matches client format lines 764-776)
                listener.info("Scene optimizer settings validated successfully")
                self._log_optimizer_settings(listener, validated_settings, enabled_ops)

            except ValidationError:
                listener.error(_INVALID_OPTIMIZER_SETTINGS_MESSAGE)
                raise ValueError(_INVALID_OPTIMIZER_SETTINGS_MESSAGE) from None
        else:
            # No scene_optimizer_settings specified - apply defaults from model
            listener.info("No scene_optimizer_settings specified, applying defaults")
            default_settings = SceneOptimizerSettings()
            validated_settings = default_settings.model_dump(
                by_alias=False, exclude_none=True
            )
            optimization_config["scene_optimizer_settings"] = validated_settings

            # Build list of enabled operations (matches client lines 744-750)
            enabled_ops = self._build_enabled_operations(validated_settings)

            # Validate at least one operation (matches client lines 752-755)
            if not enabled_ops:
                listener.error("At least one operation must be enabled")
                raise ValueError(
                    "At least one operation must be enabled in scene_optimizer_settings"
                )

            listener.info("Default scene optimizer settings applied:")
            self._log_optimizer_settings(listener, validated_settings, enabled_ops)

        # Validate backend selection
        backend = optimization_config.get("backend", "local")
        if backend not in ("remote", "local"):
            raise ValueError(_INVALID_OPTIMIZATION_BACKEND_MESSAGE) from None
        listener.info(f"Optimization backend: {redact_sensitive_config(backend)}")

        if backend == "remote":
            # Default poll_seconds to 300 (NVCF max long-polling, 5 min) so the
            # gateway doesn't time out on large scenes.  Users can still override.
            if "poll_seconds" not in optimization_config:
                optimization_config["poll_seconds"] = 300

        context["optimization_config"] = optimization_config

        # Log prototype flattening setting
        # flatten_prototypes: full flatten (convert + inline refs + remove protos)
        # Default is True since optimize_usd is typically used with pre-flattened scenes
        flatten_prototypes = optimization_config.get("flatten_prototypes", True)
        if not isinstance(flatten_prototypes, bool):
            listener.error(_INVALID_FLATTEN_PROTOTYPES_MESSAGE)
            raise ValueError(_INVALID_FLATTEN_PROTOTYPES_MESSAGE) from None
        listener.info(f"Flatten prototypes before optimization: {flatten_prototypes}")

        safe_paths = redact_sensitive_config(
            {
                "input_usd_path": str(context["input_usd_path"]),
                "output_usd_path": str(context["output_usd_path"]),
            }
        )
        listener.info(f"Input USD: {safe_paths['input_usd_path']}")
        listener.info(f"Output USD: {safe_paths['output_usd_path']}")

        return context

    def _load_config(self, context: dict[str, Any], listener: Any) -> dict[str, Any]:
        """Load an isolated mapping without rendering configuration values."""
        config, _ = load_config_mapping_from_context(
            context,
            missing_path_message="config_dict or config_path is required in context",
            missing_file_message="Configuration file not found: {config_path}",
            read_error_message=(
                "Unable to read optimize_usd configuration file: {config_path}"
            ),
            parse_error_message=(
                "Unable to parse optimize_usd configuration file: {config_path}"
            ),
            empty_message=(
                "Empty optimize_usd configuration dictionary"
                if context.get("config_dict") is not None
                else "Empty configuration file: {config_path}"
            ),
            config_dict_non_mapping_message=(
                "optimize_usd config_dict must be a mapping"
            ),
            file_non_mapping_message=("optimize_usd configuration must be a mapping"),
        )
        log_config_source(context, listener.info, label="optimize_usd")
        return config

    def _build_enabled_operations(self, settings: dict[str, Any]) -> list[str]:
        """Build list of enabled operations from settings dict.

        Args:
            settings: Scene optimizer settings dict with snake_case keys

        Returns:
            List of enabled operation names
        """
        enabled_ops = []
        if settings.get("enable_deinstance", True):
            enabled_ops.append("deinstance")
        if settings.get("enable_split_meshes", True):
            enabled_ops.append("split")
        if settings.get("enable_deduplicate", True):
            enabled_ops.append("deduplicate")
        return enabled_ops

    def _log_optimizer_settings(
        self, listener: Any, settings: dict[str, Any], enabled_ops: list[str]
    ) -> None:
        """Log optimizer settings to the listener.

        Args:
            listener: Event listener for logging
            settings: Scene optimizer settings dict with snake_case keys
            enabled_ops: List of enabled operation names
        """
        listener.info(f"  Operations: {' -> '.join(enabled_ops)}")
        listener.info(f"  Generate report: {settings.get('generate_report', True)}")
        listener.info(f"  Capture stats: {settings.get('capture_stats', True)}")
        listener.info(f"  Verbose: {settings.get('verbose', False)}")
        listener.info(f"  Wait for assets: {settings.get('wait_for_assets', False)}")
        listener.info(f"  Stage timeout: {settings.get('stage_timeout', 180.0)}s")
