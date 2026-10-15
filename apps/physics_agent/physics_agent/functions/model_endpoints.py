# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The physics pipeline's active model plan for the endpoint record.

Only ``identify_asset`` and ``predict`` call hosted models. Both provision
through ``ModelProvisioningTask``; ``predict`` also builds its optional parsing
``llm`` when that section names a backend.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from world_understanding.agentic.model_endpoints import (
    ActiveModel,
    configured_model_section,
    resolve_provisioned_model_config,
)


def _step_models(step: str, step_config: Mapping[str, Any]) -> list[ActiveModel]:
    active: list[ActiveModel] = []
    if step in {"identify_asset", "predict"}:
        if configured_model_section(step_config.get("vlm")):
            active.append(ActiveModel(step, "vlm", "vlm", step_config["vlm"]))
    if step == "predict" and configured_model_section(step_config.get("llm")):
        # Parsing falls back to the VLM, so this LLM is optional at startup.
        active.append(
            ActiveModel(step, "llm", "llm", step_config["llm"], required=False)
        )
    return active


def physics_active_models(
    steps: Iterable[str], step_configs: Mapping[str, Mapping[str, Any]]
) -> list[ActiveModel]:
    """Every model the given physics steps call, in step order."""

    active: list[ActiveModel] = []
    for step in steps:
        for model in _step_models(step, step_configs.get(step) or {}):
            active.append(
                model._replace(
                    config=resolve_provisioned_model_config(
                        model.interface, model.config
                    )
                )
            )
    return active
