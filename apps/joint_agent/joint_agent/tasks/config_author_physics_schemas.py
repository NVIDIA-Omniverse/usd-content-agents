# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration task for explicit post-rigger physics schema authoring."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from world_understanding.agentic.config import load_config_mapping_from_context
from world_understanding.agentic.tasks import Task
from world_understanding.utils.credentials import resolve_path_with_safe_diagnostics
from world_understanding.utils.object_store import ObjectStore

logger = logging.getLogger(__name__)

_PATH_KEYS = (
    "input_usd_path",
    "stage2_diagnostics_path",
    "stage2_validation_path",
    "authoring_plan_path",
    "output_usd_path",
    "diagnostics_path",
    "validation_path",
)


class AuthorPhysicsSchemasConfigTask(Task):
    """Load the explicit physics schema authoring boundary configuration."""

    def __init__(self) -> None:
        self.name = "AuthorPhysicsSchemasConfig"
        self.description = "Load explicit post-rigger physics authoring config"

    def run(
        self, context: dict[str, Any], object_store: ObjectStore | None = None
    ) -> dict[str, Any]:
        config = self._load_config(context)
        config_path = context.get("config_path")
        config_dir = Path(config_path).parent if config_path else Path.cwd()

        resolved_paths = {
            key: self._required_path(config, key, config_dir) for key in _PATH_KEYS
        }
        self._require_distinct_artifact_paths(resolved_paths)
        context.update(
            {
                "config": config,
                **{key: str(path) for key, path in resolved_paths.items()},
            }
        )

        logger.info("Loaded explicit physics schema authoring configuration")
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
            raise ValueError(f"{key} is required in author_physics_schemas config")
        if not isinstance(value, str | Path):
            raise ValueError(f"{key} must be a path string")
        path = Path(value)
        if path.is_absolute():
            return path
        return resolve_path_with_safe_diagnostics(
            config_dir / path,
            label="author_physics_schemas configuration path",
        )

    def _require_distinct_artifact_paths(self, paths: dict[str, Path]) -> None:
        by_path: dict[Path, list[str]] = {}
        for key, path in paths.items():
            canonical_path = resolve_path_with_safe_diagnostics(
                path,
                label="author_physics_schemas artifact path",
            )
            by_path.setdefault(canonical_path, []).append(key)
        collisions = [keys for keys in by_path.values() if len(keys) > 1]
        if collisions:
            joined = "; ".join(", ".join(keys) for keys in collisions)
            raise ValueError(
                "author_physics_schemas artifact paths must be distinct: " + joined
            )
