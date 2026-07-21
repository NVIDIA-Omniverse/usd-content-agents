# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Apply Physics configuration task."""

import logging
from pathlib import Path
from typing import Any

from world_understanding.agentic.config import load_config_mapping_from_context
from world_understanding.agentic.tasks import Task
from world_understanding.utils.credentials import (
    redact_sensitive_config,
    redact_sensitive_path,
    resolve_path_with_safe_diagnostics,
)
from world_understanding.utils.object_store import ObjectStore

from physics_agent.config.validator import VALID_COLLISION_APPROX
from physics_agent.functions.mass_scale_quality import VALID_MASS_SCALE_POLICIES

logger = logging.getLogger(__name__)


class ApplyPhysicsConfigTask(Task):
    """Load and validate apply_physics step configuration.

    Input context keys:
        - config_path: Path to YAML config file

    Output context keys:
        - usd_path: Input USD file path
        - predictions_path: Path to predictions JSONL
        - output_usd_path: Output path for the physics-augmented USD
        - collision_approx: Collision approximation method
        - mass_scale_policy: warn | skip_mass | fail for mass/scale QA warnings
        - allow_empty_predictions: Allow empty prediction files to produce a
          rigid-body-only USD (default: False)
    """

    def __init__(self) -> None:
        self.name = "ApplyPhysicsConfig"
        self.description = "Load apply physics step configuration"

    def run(
        self, context: dict[str, Any], object_store: ObjectStore | None = None
    ) -> dict[str, Any]:
        config = self._load_config(context)

        # A config_dict may be paired with its original YAML path so an
        # in-memory override preserves the same relative-path semantics.
        if context.get("config_path"):
            config_dir = Path(context["config_path"]).parent
        else:
            config_dir = Path.cwd()

        usd_path_str = config.get("usd_path")
        predictions_path_str = config.get("predictions_path")
        output_usd_path_str = config.get("output_usd_path")
        if not usd_path_str:
            raise ValueError("apply_physics: missing required 'usd_path' in config")
        if not predictions_path_str:
            raise ValueError(
                "apply_physics: missing required 'predictions_path' in config"
            )
        if not output_usd_path_str:
            raise ValueError(
                "apply_physics: missing required 'output_usd_path' in config"
            )

        usd_path = self._resolve_path(usd_path_str, config_dir)
        predictions_path = self._resolve_path(predictions_path_str, config_dir)
        output_usd_path = self._resolve_path(output_usd_path_str, config_dir)
        collision_approx = config.get("collision_approx", "convexHull")
        if collision_approx not in VALID_COLLISION_APPROX:
            raise ValueError(
                "apply_physics.collision_approx must be one of "
                f"{sorted(VALID_COLLISION_APPROX)}; got an unsupported value"
            )
        output_key = config.get("output_key", "classification")
        mass_scale_policy = config.get("mass_scale_policy", "skip_mass")
        if mass_scale_policy not in VALID_MASS_SCALE_POLICIES:
            raise ValueError(
                "apply_physics.mass_scale_policy must be one of "
                f"{sorted(VALID_MASS_SCALE_POLICIES)}; got an unsupported value"
            )
        allow_empty_predictions = config.get("allow_empty_predictions", False)
        if not isinstance(allow_empty_predictions, bool):
            raise ValueError(
                "apply_physics.allow_empty_predictions must be a boolean, got "
                f"{type(allow_empty_predictions).__name__}"
            )

        context.update(
            {
                "usd_path": str(usd_path),
                "predictions_path": str(predictions_path),
                "output_usd_path": str(output_usd_path),
                "collision_approx": collision_approx,
                "output_key": output_key,
                "mass_scale_policy": mass_scale_policy,
                "allow_empty_predictions": allow_empty_predictions,
            }
        )

        logger.info("Input USD: %s", redact_sensitive_path(usd_path))
        logger.info("Predictions: %s", redact_sensitive_path(predictions_path))
        logger.info("Output USD: %s", redact_sensitive_path(output_usd_path))
        logger.info("Collision approx: %s", collision_approx)
        logger.info("Output key: %s", redact_sensitive_config(output_key))
        logger.info("Mass scale policy: %s", mass_scale_policy)
        logger.info("Allow empty predictions: %s", allow_empty_predictions)

        return context

    def _load_config(self, context: dict[str, Any]) -> dict[str, Any]:
        config, _ = load_config_mapping_from_context(
            context,
            allow_empty=True,
            missing_path_message="No config_path or config_dict in context",
            missing_file_message="Config not found: {config_path}",
            parse_error_message=(
                "Unable to parse apply_physics configuration file: {config_path}"
            ),
            config_dict_non_mapping_message=(
                "apply_physics config_dict must be a mapping, got {type_name}"
            ),
            file_non_mapping_message=(
                "apply_physics config must be a YAML mapping, got "
                "{type_name}: {config_path}"
            ),
        )
        return config

    def _resolve_path(self, path: str, config_dir: Path) -> Path:
        p = Path(path)
        if p.is_absolute():
            return p
        return resolve_path_with_safe_diagnostics(
            config_dir / p,
            label="apply_physics configuration path",
        )
