# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration task for the consistency_pass step."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from world_understanding.agentic.config import load_config_mapping_from_context
from world_understanding.agentic.tasks import Task
from world_understanding.utils.credentials import (
    redact_sensitive_path,
    resolve_path_with_safe_diagnostics,
)
from world_understanding.utils.object_store import ObjectStore

logger = logging.getLogger(__name__)


class ConsistencyPassConfigTask(Task):
    """Load configuration for prediction consistency post-processing."""

    def __init__(self) -> None:
        self.name = "ConsistencyPassConfig"
        self.description = "Load prediction consistency-pass configuration"

    def run(
        self, context: dict[str, Any], object_store: ObjectStore | None = None
    ) -> dict[str, Any]:
        config = self._load_config(context)

        config_path = context.get("config_path")
        config_dir = Path(config_path).parent if config_path else Path.cwd()

        predictions_path = config.get("predictions_path")
        if not predictions_path:
            raise ValueError("predictions_path is required in consistency_pass config")
        predictions_path = self._resolve_path(predictions_path, config_dir)

        output_predictions_path = config.get("output_predictions_path")
        if output_predictions_path:
            output_predictions_path = self._resolve_path(
                output_predictions_path, config_dir
            )
        else:
            output_predictions_path = predictions_path.with_name(
                "consistent_predictions.jsonl"
            )

        output_stats_path = config.get("output_stats_path")
        if output_stats_path:
            output_stats_path = self._resolve_path(output_stats_path, config_dir)
        else:
            output_stats_path = output_predictions_path.with_suffix(".stats.json")

        harmonize_motion_profiles = config.get("harmonize_motion_profiles", False)
        if not isinstance(harmonize_motion_profiles, bool):
            raise ValueError("harmonize_motion_profiles must be a boolean")

        context.update(
            {
                "config": config,
                "predictions_path": str(predictions_path),
                "output_predictions_path": str(output_predictions_path),
                "output_stats_path": str(output_stats_path),
                "output_key": config.get("output_key", "classification"),
                "min_group_size": config.get("min_group_size", 2),
                "min_majority_fraction": config.get("min_majority_fraction", 0.6),
                "harmonize_fields": config.get("harmonize_fields", []),
                "harmonize_motion_profiles": harmonize_motion_profiles,
                "add_role": config.get("add_role", True),
                "add_instance_id": config.get("add_instance_id", True),
                "signature_depth": config.get("signature_depth", 2),
            }
        )

        logger.info("Loaded consistency-pass configuration")
        logger.info("Predictions input: %s", redact_sensitive_path(predictions_path))
        logger.info(
            "Predictions output: %s",
            redact_sensitive_path(output_predictions_path),
        )
        return context

    def _load_config(self, context: dict[str, Any]) -> dict[str, Any]:
        config, _ = load_config_mapping_from_context(
            context,
            allow_empty=True,
            missing_path_message="No config_path or config_dict in context",
            config_dict_non_mapping_message="config_dict must be a dictionary",
            file_non_mapping_message=("Configuration file must contain a dictionary"),
        )
        return config

    def _resolve_path(self, path: str | Path, config_dir: Path) -> Path:
        path_obj = Path(path)
        if path_obj.is_absolute():
            return path_obj
        return resolve_path_with_safe_diagnostics(
            config_dir / path_obj,
            label="consistency_pass configuration path",
        )
