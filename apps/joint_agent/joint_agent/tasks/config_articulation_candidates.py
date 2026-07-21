# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration task for infer_articulation_candidates."""

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

from joint_agent.config.validator import ConfigValidator

logger = logging.getLogger(__name__)


class ArticulationCandidatesConfigTask(Task):
    """Load configuration for Stage 2 articulation candidate inference."""

    def __init__(self) -> None:
        self.name = "ArticulationCandidatesConfig"
        self.description = "Load articulation candidate inference configuration"

    def run(
        self, context: dict[str, Any], object_store: ObjectStore | None = None
    ) -> dict[str, Any]:
        config = self._load_config(context)
        ConfigValidator().validate_step_requirements(
            "infer_articulation_candidates",
            config,
            {},
        )

        config_path = context.get("config_path")
        config_dir = Path(config_path).parent if config_path else Path.cwd()

        predictions_path = config.get("predictions_path")
        if not predictions_path:
            raise ValueError(
                "predictions_path is required in infer_articulation_candidates config"
            )
        predictions_path = self._resolve_path(predictions_path, config_dir)
        prim_metadata_path = config.get("prim_metadata_path")
        if prim_metadata_path:
            prim_metadata_path = self._resolve_path(prim_metadata_path, config_dir)
        dataset_path = config.get("dataset_path")
        if dataset_path:
            dataset_path = self._resolve_path(dataset_path, config_dir)

        output_candidates_path = config.get("output_candidates_path")
        if output_candidates_path:
            output_candidates_path = self._resolve_path(
                output_candidates_path, config_dir
            )
        else:
            output_candidates_path = predictions_path.with_name(
                "articulation_candidates.json"
            )

        output_report_path = config.get("output_report_path")
        if output_report_path:
            output_report_path = self._resolve_path(output_report_path, config_dir)
        else:
            output_report_path = output_candidates_path.with_suffix(".html")

        output_adjudications_path = config.get("output_adjudications_path")
        if output_adjudications_path:
            output_adjudications_path = self._resolve_path(
                output_adjudications_path,
                config_dir,
            )
        else:
            output_adjudications_path = output_candidates_path.with_name(
                "articulation_candidate_adjudications.json"
            )
        adjudication_config = config.get("adjudication", {})
        if adjudication_config is None:
            adjudication_config = {}
        if not isinstance(adjudication_config, dict):
            raise ValueError(
                "infer_articulation_candidates.adjudication must be a dictionary"
            )

        context.update(
            {
                "config": config,
                "predictions_path": str(predictions_path),
                "dataset_path": str(dataset_path) if dataset_path else None,
                "prim_metadata_path": (
                    str(prim_metadata_path) if prim_metadata_path else None
                ),
                "output_candidates_path": str(output_candidates_path),
                "output_report_path": str(output_report_path),
                "output_adjudications_path": str(output_adjudications_path),
                "output_key": config.get("output_key", "classification"),
                "candidate_joint_types": config.get(
                    "candidate_joint_types",
                    ["revolute", "prismatic", "spherical"],
                ),
                "adjudication_config": dict(adjudication_config),
            }
        )

        logger.info("Loaded articulation-candidate inference configuration")
        logger.info("Predictions input: %s", redact_sensitive_path(predictions_path))
        logger.info(
            "Candidates output: %s", redact_sensitive_path(output_candidates_path)
        )
        logger.info("HTML report output: %s", redact_sensitive_path(output_report_path))
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
            label="articulation_candidates configuration path",
        )
