# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Physics-unit helpers compatible with older world-understanding releases."""

from __future__ import annotations

import math

try:
    from world_understanding.utils.physics_units import (
        STANDARD_GRAVITY_M_PER_S2,
        acceleration_m_per_s2_to_stage_units,
        stage_units_per_meter,
        validate_meters_per_unit,
    )
except ModuleNotFoundError as exc:
    if exc.name != "world_understanding.utils.physics_units":
        raise
    # ``physics_units`` first ships after world-understanding 0.5.1. Keep the
    # Physics Agent's existing >=0.5.0 install contract usable until consumers
    # can require the release that contains the shared implementation.
    STANDARD_GRAVITY_M_PER_S2 = 9.81

    def _finite_float(value: float, *, label: str) -> float:
        if isinstance(value, bool):
            raise ValueError(f"{label} must be a finite number, got bool")
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"{label} must be finite, got {result!r}")
        return result

    def validate_meters_per_unit(meters_per_unit: float) -> float:
        """Return a positive, finite ``metersPerUnit`` value."""

        result = _finite_float(meters_per_unit, label="metersPerUnit")
        if result <= 0.0:
            raise ValueError(f"metersPerUnit must be positive, got {result!r}")
        return result

    def stage_units_per_meter(meters_per_unit: float) -> float:
        """Return the number of USD stage units representing one meter."""

        return 1.0 / validate_meters_per_unit(meters_per_unit)

    def acceleration_m_per_s2_to_stage_units(
        acceleration_m_per_s2: float,
        meters_per_unit: float,
    ) -> float:
        """Convert acceleration in m/s^2 to USD distance-units/s^2."""

        acceleration = _finite_float(
            acceleration_m_per_s2,
            label="acceleration_m_per_s2",
        )
        return acceleration * stage_units_per_meter(meters_per_unit)


__all__ = [
    "STANDARD_GRAVITY_M_PER_S2",
    "acceleration_m_per_s2_to_stage_units",
    "stage_units_per_meter",
    "validate_meters_per_unit",
]
