# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration task for the analyze_structure step."""

import logging
from pathlib import Path
from typing import Any

from world_understanding.agentic.config import load_config_mapping_from_context
from world_understanding.agentic.tasks import Task
from world_understanding.utils.credentials import (
    create_directory_with_safe_diagnostics,
    redact_sensitive_config,
    redact_sensitive_path,
)

logger = logging.getLogger(__name__)


class AnalyzeStructureConfigTask(Task):
    """Load configuration for the analyze_structure step.

    Reads step config from memory when invoked by the unified executor, with a
    YAML path fallback for direct CLI/API usage.

    Input context keys:
        - config_dict: In-memory configuration dictionary (preferred)
        - config_path: Path to the step config YAML file (fallback)

    Output context keys:
        - usd_path: Path to the USD file to analyze
        - output_dir: Output directory for structure assignments
        - strategy: "auto" | "hierarchy" | "geometric" | "skip"
        - segment_names: Optional list of segment names
        - use_prompt_library: Whether known robot prompt-library data may be used
        - robot_id: Optional explicit prompt-library robot ID
        - vlm_config: LLM/VLM configuration dict (for ModelProvisioningTask)
    """

    def __init__(self) -> None:
        self.name = "AnalyzeStructureConfig"
        self.description = "Load configuration for structure analysis"

    def run(self, context: dict[str, Any], object_store: Any = None) -> dict[str, Any]:
        config = self._load_config(context)

        usd_path = config.get("usd_path")
        if not usd_path:
            raise ValueError("usd_path not provided in config")

        output_dir = config.get("output_dir", ".")
        create_directory_with_safe_diagnostics(
            Path(output_dir),
            label="analyze_structure output directory",
            parents=True,
            exist_ok=True,
        )

        strategy = config.get("strategy", "auto")
        segment_names = config.get("segment_names")
        use_prompt_library = config.get("use_prompt_library", False)
        if not isinstance(use_prompt_library, bool):
            raise ValueError("use_prompt_library must be a boolean")
        robot_id = config.get("robot_id")
        vlm_config = config.get("vlm", config.get("llm", {}))

        logger.info("Structure analysis config:")
        logger.info("  USD: %s", redact_sensitive_path(usd_path))
        logger.info("  Output: %s", redact_sensitive_path(output_dir))
        logger.info("  Strategy: %s", redact_sensitive_config(strategy))
        logger.info(
            "  Segment names: %s",
            redact_sensitive_config(segment_names or "(auto-infer)"),
        )
        logger.info("  Prompt library: %s", "enabled" if use_prompt_library else "off")
        if robot_id:
            logger.info(
                "  Prompt library robot_id: %s",
                redact_sensitive_config(robot_id),
            )

        context.update(
            {
                "usd_path": str(usd_path),
                "output_dir": str(output_dir),
                "strategy": strategy,
                "segment_names": segment_names,
                "use_prompt_library": use_prompt_library,
                "robot_id": robot_id,
                "asset_type": config.get("asset_type"),
                "asset_subtype": config.get("asset_subtype"),
                "asset_confidence": config.get("asset_confidence"),
                "identification_path": config.get("identification_path"),
                # ModelProvisioningTask reads context["config"]["vlm"]
                "config": {"vlm": vlm_config},
            }
        )

        return context

    def _load_config(self, context: dict[str, Any]) -> dict[str, Any]:
        """Load an isolated mapping without rendering configuration values."""
        config, _ = load_config_mapping_from_context(
            context,
            allow_empty=True,
            missing_path_message="config_path not provided in context",
            missing_file_message="Config file not found: {config_path}",
            parse_error_message=(
                "Unable to parse analyze_structure config: {config_path}"
            ),
            config_dict_non_mapping_message=(
                "analyze_structure config_dict must be a mapping"
            ),
            file_non_mapping_message=(
                "analyze_structure configuration must be a mapping"
            ),
        )
        return config
