# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Texture Agent capability contract for USD rendering backends."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from world_understanding.rendering_backend_contract import (
    rendering_backend_subset,
    validate_rendering_backend_for_surface,
)

DEFAULT_TEXTURE_RENDERING_BACKEND = "remote"
TEXTURE_PRODUCTION_RENDERING_BACKENDS = rendering_backend_subset("remote", "ovrtx")
TEXTURE_RENDERING_BACKENDS = rendering_backend_subset(
    *TEXTURE_PRODUCTION_RENDERING_BACKENDS,
    "mock",
)
_TEXTURE_RENDERING_STEPS = ("render_previews", "render")


def has_production_visual_evidence(
    backend_type: object,
    *,
    render_count: int,
) -> bool:
    """Return whether saved images qualify as Texture production evidence."""
    return backend_type in TEXTURE_PRODUCTION_RENDERING_BACKENDS and render_count > 0


def validate_texture_rendering_backend(
    backend_type: object,
    *,
    step_name: str,
) -> str:
    """Validate one Texture Agent rendering selector before task side effects."""
    return validate_rendering_backend_for_surface(
        backend_type,
        TEXTURE_RENDERING_BACKENDS,
        surface=f"Texture Agent steps.{step_name}.backend",
    )


def validate_texture_rendering_steps(steps: Mapping[str, Any]) -> None:
    """Validate both exposed Texture Agent USD rendering selectors."""
    for step_name in _TEXTURE_RENDERING_STEPS:
        step_config = steps.get(step_name)
        if not isinstance(step_config, Mapping):
            continue
        validate_texture_rendering_backend(
            step_config.get("backend", DEFAULT_TEXTURE_RENDERING_BACKEND),
            step_name=step_name,
        )
