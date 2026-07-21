# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration task for apply_joint_rigger."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from world_understanding.agentic.config import load_config_mapping_from_context
from world_understanding.agentic.tasks import Task
from world_understanding.utils.credentials import resolve_path_with_safe_diagnostics
from world_understanding.utils.object_store import ObjectStore

from joint_agent.joint_rigger_options import (
    CANDIDATE_REQUIRED_JOINT_RIGGER_ADAPTERS,
    DEFAULT_CANDIDATE_READINESS_POLICY,
    DEFAULT_MISSING_DEPENDENCY_POLICY,
    DEFAULT_USD_JOINT_RIGGER_APPLY_COLLISION,
    DEFAULT_USD_JOINT_RIGGER_APPLY_MASSES,
    DEFAULT_USD_JOINT_RIGGER_TEMPLATE,
    PREDICTION_FREE_JOINT_RIGGER_ADAPTERS,
    PREDICTION_OPTIONAL_JOINT_RIGGER_ADAPTERS,
    SUPPORTED_CANDIDATE_READINESS_POLICIES,
    SUPPORTED_INTERNAL_JOINT_RIGGER_ADAPTERS,
    SUPPORTED_MISSING_DEPENDENCY_POLICIES,
    format_allowed_values,
)

logger = logging.getLogger(__name__)


class ApplyJointRiggerConfigTask(Task):
    """Load configuration for the Joint Rigger apply-step boundary."""

    def __init__(self) -> None:
        self.name = "ApplyJointRiggerConfig"
        self.description = "Load Joint Rigger apply-step configuration"

    def run(
        self, context: dict[str, Any], object_store: ObjectStore | None = None
    ) -> dict[str, Any]:
        config = self._load_config(context)

        config_path = context.get("config_path")
        config_dir = Path(config_path).parent if config_path else Path.cwd()

        input_usd_path = self._required_path(config, "input_usd_path", config_dir)
        if "adapter" not in config:
            raise ValueError("adapter is required in apply_joint_rigger config")
        adapter = config["adapter"]
        if adapter not in SUPPORTED_INTERNAL_JOINT_RIGGER_ADAPTERS:
            raise ValueError(
                "adapter must be one of: "
                f"{format_allowed_values(SUPPORTED_INTERNAL_JOINT_RIGGER_ADAPTERS)}"
            )

        predictions_path = self._optional_path(
            config,
            "predictions_path",
            config_dir,
        )
        if (
            predictions_path is None
            and adapter not in PREDICTION_FREE_JOINT_RIGGER_ADAPTERS
            and adapter not in PREDICTION_OPTIONAL_JOINT_RIGGER_ADAPTERS
        ):
            raise ValueError(
                "predictions_path is required in apply_joint_rigger config"
            )
        output_usd_path = self._required_path(config, "output_usd_path", config_dir)
        diagnostics_path = self._required_path(config, "diagnostics_path", config_dir)
        validation_path = self._required_path(config, "validation_path", config_dir)

        articulation_candidates_path = config.get("articulation_candidates_path")
        if (
            isinstance(articulation_candidates_path, str)
            and not articulation_candidates_path.strip()
        ):
            articulation_candidates_path = None
        if articulation_candidates_path:
            articulation_candidates_path = self._resolve_path(
                articulation_candidates_path, config_dir
            )
        if (
            adapter in CANDIDATE_REQUIRED_JOINT_RIGGER_ADAPTERS
            and not articulation_candidates_path
        ):
            raise ValueError(f"articulation_candidates_path is required for {adapter}")
        on_missing_dependency = config.get(
            "on_missing_dependency",
            DEFAULT_MISSING_DEPENDENCY_POLICY,
        )
        if on_missing_dependency not in SUPPORTED_MISSING_DEPENDENCY_POLICIES:
            raise ValueError(
                "on_missing_dependency must be one of: "
                f"{format_allowed_values(SUPPORTED_MISSING_DEPENDENCY_POLICIES)}"
            )
        on_unready_candidates = config.get(
            "on_unready_candidates",
            DEFAULT_CANDIDATE_READINESS_POLICY,
        )
        if on_unready_candidates not in SUPPORTED_CANDIDATE_READINESS_POLICIES:
            raise ValueError(
                "on_unready_candidates must be one of: "
                f"{format_allowed_values(SUPPORTED_CANDIDATE_READINESS_POLICIES)}"
            )
        joint_rigger_template = config.get(
            "joint_rigger_template",
            DEFAULT_USD_JOINT_RIGGER_TEMPLATE,
        )
        if (
            not isinstance(joint_rigger_template, str)
            or not joint_rigger_template.strip()
        ):
            raise ValueError("joint_rigger_template must be a non-empty string")
        apply_masses = self._optional_bool(
            config,
            "apply_masses",
            (
                False
                if adapter == "owned_core"
                else DEFAULT_USD_JOINT_RIGGER_APPLY_MASSES
            ),
        )
        apply_collision = self._optional_bool(
            config,
            "apply_collision",
            (
                False
                if adapter == "owned_core"
                else DEFAULT_USD_JOINT_RIGGER_APPLY_COLLISION
            ),
        )
        if adapter == "owned_core" and (apply_masses or apply_collision):
            raise ValueError(
                "owned_core is topology-only; apply_masses and apply_collision "
                "must both be false"
            )

        context.update(
            {
                "config": config,
                "input_usd_path": str(input_usd_path),
                "predictions_path": (
                    str(predictions_path) if predictions_path is not None else None
                ),
                "articulation_candidates_path": (
                    str(articulation_candidates_path)
                    if articulation_candidates_path
                    else None
                ),
                "output_usd_path": str(output_usd_path),
                "diagnostics_path": str(diagnostics_path),
                "validation_path": str(validation_path),
                "adapter": adapter,
                "on_missing_dependency": on_missing_dependency,
                "on_unready_candidates": on_unready_candidates,
                "joint_rigger_template": joint_rigger_template,
                "apply_masses": apply_masses,
                "apply_collision": apply_collision,
            }
        )

        logger.info("Loaded Joint Rigger apply-step configuration")
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

    def _required_path(
        self, config: dict[str, Any], key: str, config_dir: Path
    ) -> Path:
        value = config.get(key)
        if not value:
            raise ValueError(f"{key} is required in apply_joint_rigger config")
        return self._resolve_path(value, config_dir)

    def _resolve_path(self, path: str | Path, config_dir: Path) -> Path:
        path_obj = Path(path)
        if path_obj.is_absolute():
            return path_obj
        return resolve_path_with_safe_diagnostics(
            config_dir / path_obj,
            label="apply_joint_rigger configuration path",
        )

    def _optional_path(
        self,
        config: dict[str, Any],
        key: str,
        config_dir: Path,
    ) -> Path | None:
        value = config.get(key)
        if value is None or value == "":
            return None
        if not isinstance(value, str | Path):
            raise ValueError(f"{key} must be a path string")
        return self._resolve_path(value, config_dir)

    def _optional_bool(
        self,
        config: dict[str, Any],
        key: str,
        default: bool,
    ) -> bool:
        value = config.get(key, default)
        if not isinstance(value, bool):
            raise ValueError(f"{key} must be a boolean")
        return value
