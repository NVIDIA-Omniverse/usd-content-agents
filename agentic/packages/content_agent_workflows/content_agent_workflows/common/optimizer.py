# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Workflow-owned Scene Optimizer request normalization."""

from __future__ import annotations

from copy import deepcopy
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_OPTIMIZATION_CONFIG_DEPTH = 32
MAX_OPTIMIZATION_CONFIG_CONTAINER_ITEMS = 1024
MAX_OPTIMIZATION_CONFIG_TOTAL_ITEMS = 4096


class OptimizerBackend(StrEnum):
    """Scene Optimizer execution backend selected by the workflow."""

    LOCAL = "local"
    REMOTE = "remote"


class OptimizerRequestOptions(BaseModel):
    """Validated workflow options for Scene Optimizer execution."""

    model_config = ConfigDict(extra="forbid")

    optimize: bool = False
    optimizer_backend: OptimizerBackend | None = None
    flatten_prototypes: bool | None = None
    enable_deinstance: bool | None = None
    enable_split: bool | None = None
    enable_deduplicate: bool | None = None
    clear_materials: bool = False
    optimization_config: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_optimizer_operations(self) -> OptimizerRequestOptions:
        if self.optimize and not any(
            _resolved_optimizer_operations(self.resolved_optimization_config()).values()
        ):
            raise ValueError(
                "At least one Scene Optimizer operation must be enabled when "
                "optimize is true."
            )
        return self

    def resolved_optimization_config(self) -> dict[str, Any]:
        config = deepcopy(self.optimization_config)
        if self.optimizer_backend is not None:
            config["backend"] = self.optimizer_backend.value
        if self.flatten_prototypes is not None:
            config["flatten_prototypes"] = self.flatten_prototypes
        settings = _settings_dict(config)
        for value, key in (
            (self.enable_deinstance, "enable_deinstance"),
            (self.enable_split, "enable_split_meshes"),
            (self.enable_deduplicate, "enable_deduplicate"),
        ):
            if value is not None:
                settings[key] = value
        _normalize_aliases(settings)
        if settings:
            config["scene_optimizer_settings"] = settings
        _validate_shape(config)
        return config


def _settings_dict(config: dict[str, Any]) -> dict[str, Any]:
    settings = config.get("scene_optimizer_settings")
    if settings is None:
        return {}
    if not isinstance(settings, dict):
        raise ValueError(
            "optimization_config.scene_optimizer_settings must be an object"
        )
    return deepcopy(settings)


def _normalize_aliases(settings: dict[str, Any]) -> None:
    for alias, canonical in {
        "enableSplitMeshes": "enable_split_meshes",
        "enable_split": "enable_split_meshes",
        "enableDeinstance": "enable_deinstance",
        "enableDeduplicate": "enable_deduplicate",
    }.items():
        if alias in settings:
            settings.setdefault(canonical, settings[alias])
            settings.pop(alias)


def _resolved_optimizer_operations(config: dict[str, Any]) -> dict[str, bool]:
    settings = _settings_dict(config)
    _normalize_aliases(settings)
    result: dict[str, bool] = {}
    for name, key in (
        ("deinstance", "enable_deinstance"),
        ("split", "enable_split_meshes"),
        ("deduplicate", "enable_deduplicate"),
    ):
        value = settings.get(key, True)
        if not isinstance(value, bool):
            raise ValueError(
                f"optimization_config.scene_optimizer_settings.{key} must be a boolean"
            )
        result[name] = value
    return result


def _validate_shape(value: Any, *, depth: int = 0) -> int:
    if depth > MAX_OPTIMIZATION_CONFIG_DEPTH:
        raise ValueError(
            f"optimization_config exceeds maximum depth {MAX_OPTIMIZATION_CONFIG_DEPTH}"
        )
    if not isinstance(value, dict | list):
        return 0
    if len(value) > MAX_OPTIMIZATION_CONFIG_CONTAINER_ITEMS:
        raise ValueError("optimization_config contains too many container items")
    total = len(value)
    children = value.values() if isinstance(value, dict) else value
    if isinstance(value, dict) and not all(isinstance(key, str) for key in value):
        raise ValueError("optimization_config keys must be strings")
    for child in children:
        total += _validate_shape(child, depth=depth + 1)
        if total > MAX_OPTIMIZATION_CONFIG_TOTAL_ITEMS:
            raise ValueError("optimization_config contains too many items")
    return total
