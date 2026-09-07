# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Conversions between SI acceleration and USD stage units."""

from __future__ import annotations

import math

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


def stage_linear_unit_scale(
    source_meters_per_unit: float,
    target_meters_per_unit: float,
) -> float:
    """Scale a stage-linear quantity while changing the stage unit system."""

    source = validate_meters_per_unit(source_meters_per_unit)
    target = validate_meters_per_unit(target_meters_per_unit)
    return source / target


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


def acceleration_stage_units_to_m_per_s2(
    acceleration_stage_units: float,
    meters_per_unit: float,
) -> float:
    """Convert USD distance-units/s^2 acceleration to m/s^2."""

    acceleration = _finite_float(
        acceleration_stage_units,
        label="acceleration_stage_units",
    )
    return acceleration * validate_meters_per_unit(meters_per_unit)


def looks_like_unscaled_standard_gravity(
    acceleration_stage_units: float,
    meters_per_unit: float,
) -> bool:
    """Detect the legacy ``9.81``-regardless-of-MPU authoring fingerprint."""

    acceleration = _finite_float(
        acceleration_stage_units,
        label="acceleration_stage_units",
    )
    mpu = validate_meters_per_unit(meters_per_unit)
    return not math.isclose(mpu, 1.0, rel_tol=0.0, abs_tol=1e-12) and math.isclose(
        abs(acceleration),
        STANDARD_GRAVITY_M_PER_S2,
        rel_tol=1e-6,
        abs_tol=1e-6,
    )


__all__ = [
    "STANDARD_GRAVITY_M_PER_S2",
    "acceleration_m_per_s2_to_stage_units",
    "acceleration_stage_units_to_m_per_s2",
    "looks_like_unscaled_standard_gravity",
    "stage_linear_unit_scale",
    "stage_units_per_meter",
    "validate_meters_per_unit",
]
