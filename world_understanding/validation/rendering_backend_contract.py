# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lightweight USD rendering-backend contract for Validation Agent."""

from __future__ import annotations

from typing import Any

from world_understanding.rendering_backend_contract import rendering_backend_subset

VALIDATION_RENDERING_BACKEND_NAMES: tuple[str, ...] = rendering_backend_subset(
    "remote",
    "ovrtx",
)
SUPPORTED_RENDER_BACKENDS: frozenset[str] = frozenset(
    VALIDATION_RENDERING_BACKEND_NAMES
)


def normalize_validation_rendering_backend(backend: Any) -> Any:
    """Normalize selector spelling without choosing a rendering provider."""
    if backend is None:
        return None
    if not isinstance(backend, str):
        return backend
    normalized = backend.strip().lower()
    return normalized or None
