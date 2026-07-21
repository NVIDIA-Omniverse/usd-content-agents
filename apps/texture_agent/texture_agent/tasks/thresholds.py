# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared helper: validate ``failure_threshold`` config values.

Used by ``GenerateTexturesTask`` and ``BlendTexturesTask`` -- kept in a
private module so neither task has to reach into the other's namespace
for a generic helper.
"""

from __future__ import annotations

import math
from typing import Any


def validate_failure_threshold(value: Any, *, config_key: str) -> float:
    """Coerce + validate a ``failure_threshold`` config value.

    Accepts numeric strings ("0.5") and integers (1) by way of ``float()``.
    Rejects NaN, infinities, uncoercible values, and anything outside
    ``[0.0, 1.0]``. Without this gate a typo'd config (``failure_threshold:
    1.1`` or ``"nan"``) would silently disable the all-failed gate --
    ``nan >= 1.0`` and ``0.5 >= 1.1`` both evaluate ``False`` -- which is
    the very failure mode the threshold exists to catch.

    Always validate at the TOP of a task's ``run`` (before any backend
    dispatch) so a typo fails fast, never after expensive network work.
    """
    try:
        threshold = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{config_key} must be a finite number in [0.0, 1.0], got {value!r}"
        ) from exc
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError(
            f"{config_key} must be a finite number in [0.0, 1.0], got {value!r}"
        )
    return threshold


def raise_if_failure_threshold_exceeded(
    *,
    attempted_count: int,
    errors: list[dict[str, Any]],
    backend_label: str,
    failure_threshold: float,
) -> None:
    """Raise when ``errors / attempted_count`` reaches ``failure_threshold``."""
    if attempted_count <= 0 or not errors:
        return

    failed_materials = {
        str(error.get("material"))
        for error in errors
        if error.get("material") is not None
    }
    failed_count = len(failed_materials) if failed_materials else len(errors)
    failure_rate = failed_count / attempted_count
    if failure_rate < failure_threshold:
        return

    sample_lines = [
        f"{e['material']}: [{e['type']}"
        + (f" {e['status']}" if e.get("status") is not None else "")
        + f"] {e['message']}"
        for e in errors[:3]
    ]
    sample = "\n  - ".join(sample_lines)
    more = f"\n  ... ({len(errors) - 3} more)" if len(errors) > 3 else ""
    threshold_pct = int(failure_threshold * 100)
    raise RuntimeError(
        f"{failed_count}/{attempted_count} texture generation requests "
        f"failed via {backend_label} "
        f"(failure rate {failure_rate:.0%} >= threshold {threshold_pct}%). "
        f"First errors:\n  - {sample}{more}"
    )
