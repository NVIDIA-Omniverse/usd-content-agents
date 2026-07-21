# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration loading task for apply workflows.

NOTE: This is a compatibility shim for the old workflow system.
The unified config system (UnifiedPipelineConfigTask) is preferred.
"""

import logging
from typing import Any

from world_understanding.agentic.config import log_config_source
from world_understanding.agentic.events import get_listener
from world_understanding.agentic.tasks import Task

from material_agent.materials import material_mapping_with_fallback
from material_agent.tasks.config_loader import load_config_from_context

logger = logging.getLogger(__name__)


class ApplyConfigTask(Task):
    """Compatibility config task for apply workflows.

    Standalone workflows load YAML from ``config_path``. Unified pipelines pass
    ``config_dict`` in memory and retain ``config_path`` only as a path anchor.
    """

    def __init__(self):
        """Initialize the apply config loading task."""
        self.name = "ApplyConfigLoading"
        self.description = "Load apply configuration from YAML file"

    def run(self, context: dict[str, Any], object_store=None) -> dict[str, Any]:
        """Load apply configuration.

        Args:
            context: Workflow context containing config_dict or config_path
            object_store: Optional object store (not used)

        Returns:
            Updated context with loaded configuration
        """
        # Get event listener (or logger fallback)
        listener = get_listener(context, logger_name=__name__)

        config, _ = load_config_from_context(context)
        log_config_source(context, listener.info, label="apply")

        # Pass through the config
        context["config"] = config

        # Extract key fields
        context["input_usd_path"] = config.get("input_usd_path")
        context["predictions_path"] = config.get("predictions_path")
        context["output_usd_path"] = config.get("output_usd_path")
        context["materials_mapping"] = self._mapping_with_fallback(
            config.get("materials_mapping", {})
        )
        context["usd_search_config"] = config.get("usd_search", {})
        context["aws_profile"] = config.get("aws_profile")
        context["layer_only"] = config.get("layer_only", False)
        context["flatten_output"] = config.get("flatten_output", True)
        context["material_profile"] = self._material_profile_from_config(config)
        context["skip_instance_check"] = config.get("skip_instance_check", False)
        allow_empty_predictions = config.get("allow_empty_predictions", False)
        if not isinstance(allow_empty_predictions, bool):
            raise ValueError(
                "apply.allow_empty_predictions must be a boolean, got "
                f"{type(allow_empty_predictions).__name__}"
            )
        context["allow_empty_predictions"] = allow_empty_predictions
        fail_on_unknown_material = config.get("fail_on_unknown_material", False)
        if not isinstance(fail_on_unknown_material, bool):
            raise ValueError(
                "apply.fail_on_unknown_material must be a boolean, got "
                f"{type(fail_on_unknown_material).__name__}"
            )
        context["fail_on_unknown_material"] = fail_on_unknown_material
        context["render_config"] = config.get("render", {})
        context["llm_config"] = config.get("llm", {})

        # Set render_enabled flag based on render config
        render_config = context["render_config"]
        context["render_enabled"] = (
            render_config.get("enabled", False) if render_config else False
        )

        return context

    @staticmethod
    def _mapping_with_fallback(materials_mapping: Any) -> Any:
        """Add the canonical fallback material to legacy direct mappings."""
        return material_mapping_with_fallback(materials_mapping)

    @staticmethod
    def _material_profile_from_config(config: dict[str, Any]) -> str:
        """Return a validated material profile request from legacy apply config."""
        for key in ("material_profile", "shader_target", "material_authoring_target"):
            value = config.get(key)
            if value is None or value == "":
                continue
            if not isinstance(value, str):
                raise ValueError(
                    f"apply.{key} must be a string, got {type(value).__name__}"
                )
            return value
        return "auto"
