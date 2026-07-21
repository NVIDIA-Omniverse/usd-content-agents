# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helpers for keeping untrusted library data out of prompt instructions."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any

from material_agent.materials import TRUSTED_FALLBACK_GUIDANCE


def format_material_names_for_prompt(
    entries: Iterable[Mapping[str, Any]],
) -> str:
    """Serialize material names plus code-owned fallback guidance for prompts.

    Material descriptions are intentionally excluded. They are user-controlled,
    free-form prose and are not required by the prediction contract, which asks
    the model to choose an exact material name from the library. Reserved fallback
    names receive guidance sourced only from constants in :mod:`materials`.
    """

    material_names = [str(entry["name"]) for entry in entries]
    payload: dict[str, Any] = {"material_names": material_names}
    trusted_fallback_guidance = {
        name: TRUSTED_FALLBACK_GUIDANCE[name]
        for name in material_names
        if name in TRUSTED_FALLBACK_GUIDANCE
    }
    if trusted_fallback_guidance:
        payload["trusted_fallback_guidance"] = trusted_fallback_guidance
    return json.dumps(
        payload,
        ensure_ascii=True,
        indent=2,
    )
