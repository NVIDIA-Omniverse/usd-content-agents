# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The articulation pipeline's active model plan for the endpoint record.

The shared record and preflight live in
``world_understanding.agentic.model_endpoints``; this module lists exactly the
models each articulation step calls at runtime, through the interface that
provisions them.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from world_understanding.agentic.model_endpoints import (
    ActiveModel,
    configured_model_section,
    resolve_provisioned_model_config,
)

from joint_agent.config.model_aliases import normalize_analyze_structure_model_alias

_DEFAULT_ADJUDICATION_MODEL_KEY = "llm"


def _step_models(step: str, step_config: Mapping[str, Any]) -> list[ActiveModel]:
    if step == "identify_asset":
        section = step_config.get("vlm")
        return (
            [ActiveModel(step, "vlm", "vlm", section)]
            if configured_model_section(section)
            else []
        )
    if step == "analyze_structure":
        if step_config.get("strategy") == "skip":
            return []
        # The public ``llm`` alias is provisioned as the runtime VLM.
        normalized = normalize_analyze_structure_model_alias(dict(step_config))
        section = normalized.get("vlm")
        return (
            [ActiveModel(step, "vlm", "vlm", section)]
            if configured_model_section(section)
            else []
        )
    if step == "predict":
        active = []
        if configured_model_section(step_config.get("vlm")):
            active.append(ActiveModel(step, "vlm", "vlm", step_config["vlm"]))
        if configured_model_section(step_config.get("llm")):
            # Parsing falls back to the VLM, so this LLM is optional at startup.
            active.append(
                ActiveModel(step, "llm", "llm", step_config["llm"], required=False)
            )
        return active
    if step == "infer_articulation_candidates":
        adjudication = step_config.get("adjudication")
        if not (isinstance(adjudication, Mapping) and adjudication.get("enabled")):
            return []
        key = str(adjudication.get("model_key") or _DEFAULT_ADJUDICATION_MODEL_KEY)
        section = step_config.get(key)
        if not configured_model_section(section):
            return []
        interface = "vlm" if key.startswith("vlm") else "llm"
        # Called only when topology reconciliation or a compound-edge conflict
        # requires it, so it is recorded but not probed at startup.
        return [ActiveModel(step, key, interface, section, probe_at_startup=False)]
    return []


def joint_active_models(
    steps: Iterable[str], step_configs: Mapping[str, Mapping[str, Any]]
) -> list[ActiveModel]:
    """Every model the given articulation steps call, in step order."""

    active: list[ActiveModel] = []
    for step in steps:
        for model in _step_models(step, step_configs.get(step) or {}):
            # Joint provisions through ModelProvisioningTask, which applies the
            # NIM base-URL environment override.
            active.append(
                model._replace(
                    config=resolve_provisioned_model_config(
                        model.interface, model.config
                    )
                )
            )
    return active
