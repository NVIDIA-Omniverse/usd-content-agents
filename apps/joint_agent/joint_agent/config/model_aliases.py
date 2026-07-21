# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Normalize public model-config aliases to Joint Agent runtime keys."""

from __future__ import annotations

from typing import Any


def normalize_analyze_structure_model_alias(
    step_config: dict[str, Any],
) -> dict[str, Any]:
    """Return a copy with the public ``llm`` alias mapped to runtime ``vlm``.

    Structure analysis provisions its text model through the shared VLM-shaped
    runtime context.  The service API exposes that model as ``llm`` because it
    performs hierarchy naming.  When both keys are present, the public ``llm``
    value wins so an injected/default ``vlm`` cannot shadow the caller's model.
    """
    normalized = step_config.copy()
    llm_config = normalized.pop("llm", None)
    if llm_config is not None:
        normalized["vlm"] = llm_config
    return normalized
