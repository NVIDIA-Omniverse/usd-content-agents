# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration validator for Physics Agent."""

import logging
from typing import Any

from world_understanding.agentic.config.unknown_keys import (
    build_nested_config_key_schema,
    warn_unknown_nested_config_keys,
)
from world_understanding.utils.credentials import redact_sensitive_config

from physics_agent.config.schema import (
    REQUIRED_FIELDS,
    REQUIRED_SECTIONS,
    STEP_ORDER,
    get_default_config,
    get_step_defaults,
)
from physics_agent.functions.mass_scale_quality import VALID_MASS_SCALE_POLICIES

logger = logging.getLogger(__name__)

# Allowed values for apply_physics.collision_approx. Shared with the
# per-step ConfigTask so both validation paths stay in sync.
VALID_COLLISION_APPROX = frozenset(
    {
        "convexHull",
        "convexDecomposition",
        "boundingCube",
        "boundingSphere",
        "meshSimplification",
        "none",
    }
)


class ConfigValidator:
    """Validator for Physics Agent configuration."""

    def validate(self, config: dict[str, Any]) -> None:
        """Validate the configuration structure.

        Args:
            config: Configuration dictionary to validate

        Raises:
            ValueError: If configuration is invalid
        """
        # Check required sections
        for section in REQUIRED_SECTIONS:
            if section not in config:
                raise ValueError(f"Missing required section: '{section}'")

        # Check required fields in each section
        for section, fields in REQUIRED_FIELDS.items():
            if section not in config:
                continue
            section_config = config[section]
            if section_config is None:
                section_config = {}
            for field in fields:
                if field not in section_config or section_config[field] is None:
                    raise ValueError(f"Missing required field: '{section}.{field}'")

        key_schema = build_nested_config_key_schema(
            get_default_config(),
            STEP_ORDER,
            get_step_defaults,
        )
        key_schema["steps"]["predict"]["report"] = {}
        warn_unknown_nested_config_keys(
            config,
            key_schema,
            logger,
            strict_paths={("steps", step_name) for step_name in STEP_ORDER},
        )

        # Validate steps section if present
        steps = config.get("steps", {})
        if steps:
            self._validate_steps(steps)

    def _validate_steps(self, steps: dict[str, Any]) -> None:
        """Validate steps configuration.

        Args:
            steps: Steps configuration dictionary
        """
        valid_steps = set(STEP_ORDER)

        for step_name in steps.keys():
            if step_name not in valid_steps:
                logger.warning(
                    "Unknown step '%s' in configuration. Valid steps: %s",
                    redact_sensitive_config(step_name),
                    ", ".join(sorted(valid_steps)),
                )

    def validate_step_requirements(
        self,
        step_name: str,
        step_config: dict[str, Any],
        full_config: dict[str, Any],
    ) -> None:
        """Validate requirements for a specific step.

        Args:
            step_name: Name of the step
            step_config: Step configuration
            full_config: Full configuration dictionary
        """
        # Step-specific validation
        if step_name == "predict":
            # Ensure VLM config is present
            if "vlm" not in step_config:
                logger.warning(
                    "predict step has no 'vlm' configuration - using defaults"
                )

            # Validate output_key if present
            output_key = step_config.get("output_key")
            if output_key and not isinstance(output_key, str):
                raise ValueError(
                    f"predict.output_key must be a string, got {type(output_key)}"
                )
            allow_empty_predictions = step_config.get("allow_empty_predictions", False)
            if not isinstance(allow_empty_predictions, bool):
                raise ValueError(
                    "predict.allow_empty_predictions must be a boolean, got "
                    f"{type(allow_empty_predictions).__name__}"
                )

        elif step_name == "apply_physics":
            collision_approx = step_config.get("collision_approx", "convexHull")
            if collision_approx not in VALID_COLLISION_APPROX:
                raise ValueError(
                    "apply_physics.collision_approx must be one of "
                    f"{sorted(VALID_COLLISION_APPROX)}; got an unsupported value"
                )
            mass_scale_policy = step_config.get("mass_scale_policy", "skip_mass")
            if mass_scale_policy not in VALID_MASS_SCALE_POLICIES:
                raise ValueError(
                    "apply_physics.mass_scale_policy must be one of "
                    f"{sorted(VALID_MASS_SCALE_POLICIES)}; got an unsupported value"
                )
            allow_empty_predictions = step_config.get("allow_empty_predictions", False)
            if not isinstance(allow_empty_predictions, bool):
                raise ValueError(
                    "apply_physics.allow_empty_predictions must be a boolean, got "
                    f"{type(allow_empty_predictions).__name__}"
                )
