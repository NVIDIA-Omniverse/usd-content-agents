# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared normalization for Texture renderer result envelopes."""

from __future__ import annotations

from typing import Any


def render_result_items(
    results: Any,
    *,
    producer: str,
) -> list[dict[str, Any]]:
    """Return per-camera results while preserving caller-specific diagnostics."""
    if isinstance(results, dict):
        items = results.get("results")
        if isinstance(items, list) and all(isinstance(item, dict) for item in items):
            return items
        raise ValueError(
            f"{producer} returned a dict without a list-valued 'results' key"
        )

    if isinstance(results, list) and all(isinstance(item, dict) for item in results):
        return results

    raise TypeError(
        f"{producer} returned unsupported result shape "
        f"{type(results).__name__}; expected dict['results'] or list[dict]"
    )
