# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from world_understanding.utils.physics_units import (
    STANDARD_GRAVITY_M_PER_S2,
    acceleration_m_per_s2_to_stage_units,
    acceleration_stage_units_to_m_per_s2,
    looks_like_unscaled_standard_gravity,
    stage_linear_unit_scale,
    stage_units_per_meter,
    validate_meters_per_unit,
)


@pytest.mark.parametrize(
    ("meters_per_unit", "expected_stage_gravity"),
    [(1.0, 9.81), (0.1, 98.1), (0.01, 981.0), (0.001, 9810.0)],
)
def test_acceleration_round_trip_preserves_si_value(
    meters_per_unit: float,
    expected_stage_gravity: float,
) -> None:
    stage_gravity = acceleration_m_per_s2_to_stage_units(
        STANDARD_GRAVITY_M_PER_S2,
        meters_per_unit,
    )

    assert stage_gravity == pytest.approx(expected_stage_gravity)
    assert acceleration_stage_units_to_m_per_s2(
        stage_gravity,
        meters_per_unit,
    ) == pytest.approx(STANDARD_GRAVITY_M_PER_S2)


def test_stage_acceleration_rescales_in_both_directions() -> None:
    assert stage_linear_unit_scale(0.01, 1.0) == pytest.approx(0.01)
    assert stage_linear_unit_scale(1.0, 0.01) == pytest.approx(100.0)
    assert stage_units_per_meter(0.01) == pytest.approx(100.0)


@pytest.mark.parametrize("meters_per_unit", [0.0, -1.0, float("inf"), float("nan")])
def test_invalid_meters_per_unit_is_rejected(meters_per_unit: float) -> None:
    with pytest.raises(ValueError, match="metersPerUnit"):
        validate_meters_per_unit(meters_per_unit)


def test_legacy_unscaled_gravity_fingerprint_requires_non_meter_stage() -> None:
    assert looks_like_unscaled_standard_gravity(9.81, 0.01)
    assert not looks_like_unscaled_standard_gravity(981.0, 0.01)
    assert not looks_like_unscaled_standard_gravity(9.81, 1.0)
