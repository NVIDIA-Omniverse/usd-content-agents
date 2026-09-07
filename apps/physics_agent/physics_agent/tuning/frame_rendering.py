# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared rendering-backend selection for Physics tuning frame evidence."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from world_understanding.functions.graphics.rendering_backend_factory import (
    validate_rendering_backend_name,
)

DEFAULT_FRAME_RENDERER = "ovrtx"


def resolve_frame_renderer(target: Mapping[str, Any]) -> str:
    """Resolve and validate the tuning frame-evidence renderer.

    ``frame_renderer`` takes precedence over the legacy ``vlm_renderer``
    setting. Only a missing or explicit ``None`` value falls through; other
    falsy values remain explicit configuration and are rejected by the shared
    rendering-backend contract.
    """

    renderer = target.get("frame_renderer")
    if renderer is None:
        renderer = target.get("vlm_renderer")
    if renderer is None:
        renderer = DEFAULT_FRAME_RENDERER
    return validate_rendering_backend_name(renderer)
