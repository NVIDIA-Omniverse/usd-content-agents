# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration loading task for cluster_prims and expand_cluster_predictions steps."""

import logging
from pathlib import Path
from typing import Any

from world_understanding.agentic.events import get_listener
from world_understanding.agentic.tasks import Task
from world_understanding.utils.credentials import (
    redact_sensitive_config,
    redact_sensitive_path,
)

from material_agent.tasks.config_loader import load_config_from_context

logger = logging.getLogger(__name__)


def _resolve_config_relative_path(value: str | Path, config_path: Path) -> str:
    """Resolve a task path against its source configuration anchor."""
    path = Path(value)
    if not path.is_absolute():
        path = (config_path.parent / path).resolve()
    return str(path)


class ClusterPrimsConfigTask(Task):
    """Load config for the cluster_prims step.

    Input context keys:
        - config_dict: In-memory configuration dictionary (preferred)
        - config_path: Path to YAML config file (fallback)

    Output context keys:
        - dataset_path: Path to dataset.jsonl
        - working_dir: Pipeline working directory
        - cluster_prims_config: Full cluster_prims config dict
    """

    def run(self, context: dict[str, Any], object_store: Any = None) -> dict[str, Any]:
        listener = get_listener(context, logger_name=__name__)

        config, config_path = load_config_from_context(
            context,
            missing_path_message="config_dict or config_path is required in context",
            missing_file_message="Config not found: {config_path}",
            empty_message="Empty config: {config_path}",
        )

        if "dataset_path" not in config:
            raise ValueError("dataset_path is required in cluster_prims config")
        context["dataset_path"] = _resolve_config_relative_path(
            config["dataset_path"], config_path
        )

        if "working_dir" not in config:
            raise ValueError("working_dir is required in cluster_prims config")
        context["working_dir"] = _resolve_config_relative_path(
            config["working_dir"], config_path
        )

        # Pass through the full config as cluster_prims_config.
        # The executor passes a flat step config, so use it directly rather
        # than looking for a nested key.
        cluster_config = {
            k: v
            for k, v in config.items()
            if k not in ("dataset_path", "working_dir", "enabled")
        }

        context["cluster_prims_config"] = cluster_config

        listener.info(
            f"[cluster_prims] dataset: {redact_sensitive_path(context['dataset_path'])}"
        )
        listener.info(
            f"[cluster_prims] config: "
            f"{redact_sensitive_config(context['cluster_prims_config'])}"
        )
        return context


class ExpandClusterPredictionsConfigTask(Task):
    """Load config for the expand_cluster_predictions step.

    Input context keys:
        - config_dict: In-memory configuration dictionary (preferred)
        - config_path: Path to YAML config file (fallback)

    Output context keys:
        - predictions_path: Path to predictions.jsonl (representatives only)
        - cluster_map_path: Path to clusters/cluster_map.jsonl
    """

    def run(self, context: dict[str, Any], object_store: Any = None) -> dict[str, Any]:
        listener = get_listener(context, logger_name=__name__)

        config, config_path = load_config_from_context(
            context,
            missing_path_message="config_dict or config_path is required in context",
            missing_file_message="Config not found: {config_path}",
            empty_message="Empty config: {config_path}",
        )

        # Propagate cluster_prims_ran so ExpandClusterPredictionsTask can skip itself
        cluster_prims_ran = config.get("cluster_prims_ran", False)
        context["cluster_prims_ran"] = cluster_prims_ran

        if not cluster_prims_ran:
            listener.info(
                "[expand_cluster_predictions] cluster_prims did not run — skipping config load"
            )
            return context

        if "predictions_path" not in config:
            raise ValueError(
                "predictions_path is required in expand_cluster_predictions config"
            )
        context["predictions_path"] = _resolve_config_relative_path(
            config["predictions_path"], config_path
        )

        if "cluster_map_path" not in config:
            raise ValueError(
                "cluster_map_path is required in expand_cluster_predictions config"
            )
        context["cluster_map_path"] = _resolve_config_relative_path(
            config["cluster_map_path"], config_path
        )

        listener.info(
            "[expand_cluster_predictions] predictions: "
            f"{redact_sensitive_path(context['predictions_path'])}"
        )
        listener.info(
            "[expand_cluster_predictions] cluster_map: "
            f"{redact_sensitive_path(context['cluster_map_path'])}"
        )
        return context
