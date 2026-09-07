# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import math

import pytest

from physics_agent.functions.mass_scale_quality import (
    HIGH_MASS_KG,
    MAX_IMPLIED_FILL_FACTOR,
    build_mass_scale_quality_warnings,
    extract_bbox_metrics_meters,
    has_mass_scale_suspicious_warning,
    merge_quality_warnings,
)


def test_mass_scale_quality_warns_for_high_mass_with_large_bbox():
    prediction = {
        "id": "/World/Robot/oversized_link",
        "classification": {
            "physical_properties": {
                "density": 2700,
                "estimated_mass_kg": 25000,
            }
        },
    }
    dataset_entry = {
        "id": "/World/Robot/oversized_link",
        "metadata": {
            "world_bbox_meters": {
                "size": [8.0, 2.0, 2.0],
            }
        },
    }

    warnings = build_mass_scale_quality_warnings(prediction, dataset_entry)

    assert [warning["code"] for warning in warnings] == ["mass_scale_suspicious"]
    assert warnings[0]["severity"] == "warning"
    assert warnings[0]["details"]["max_dimension_m"] == 8.0
    assert warnings[0]["details"]["estimated_mass_kg"] == 25000
    assert warnings[0]["details"]["thresholds"]["high_mass_kg"] == HIGH_MASS_KG
    assert warnings[0]["message"] == (
        "Predicted mass is very high and the component bounding box is many meters "
        "across. Verify source USD units/scale before using this mass in simulation."
    )


def test_mass_scale_quality_warns_for_implausible_fill_factor_at_small_scale():
    prediction = {
        "id": "/World/banana",
        "classification": {
            "physical_properties": {
                "density": 1000,
                "estimated_mass_kg": 100_000,
            }
        },
    }
    dataset_entry = {
        "id": "/World/banana",
        "metadata": {
            "world_bbox_meters": {
                "size": [0.2, 0.1, 0.05],
            }
        },
    }

    warnings = build_mass_scale_quality_warnings(prediction, dataset_entry)

    assert [warning["code"] for warning in warnings] == ["mass_scale_suspicious"]
    assert warnings[0]["severity"] == "warning"
    assert warnings[0]["details"]["max_dimension_m"] == 0.2
    assert warnings[0]["details"]["implied_fill_factor"] == pytest.approx(100_000)
    assert (
        warnings[0]["details"]["thresholds"]["max_implied_fill_factor"]
        == MAX_IMPLIED_FILL_FACTOR
    )
    assert has_mass_scale_suspicious_warning(
        {**prediction, "quality_warnings": warnings}
    )


def test_mass_scale_quality_handles_subnormal_density_without_zero_division():
    prediction = {
        "id": "prim-1",
        "classification": {
            "physical_properties": {
                "density": 5e-324,
                "estimated_mass_kg": 1.0,
            }
        },
    }
    dataset_entry = {
        "id": "prim-1",
        "metadata": {"world_bbox_meters": {"size": [1.0, 1.0, 0.5]}},
    }

    warnings = build_mass_scale_quality_warnings(prediction, dataset_entry)

    assert [warning["code"] for warning in warnings] == ["mass_scale_suspicious"]
    implied_fill_factor = warnings[0]["details"]["implied_fill_factor"]
    assert math.isfinite(implied_fill_factor)
    assert implied_fill_factor > MAX_IMPLIED_FILL_FACTOR
    json.dumps(warnings, allow_nan=False)


def test_mass_scale_quality_allows_fill_factor_at_upper_bound():
    prediction = {
        "id": "prim-1",
        "classification": {
            "physical_properties": {
                "density": 1000,
                "estimated_mass_kg": 1000 * MAX_IMPLIED_FILL_FACTOR,
            }
        },
    }
    dataset_entry = {
        "id": "prim-1",
        "metadata": {"world_bbox_meters": {"size": [1.0, 1.0, 1.0]}},
    }

    assert build_mass_scale_quality_warnings(prediction, dataset_entry) == []


def test_mass_scale_quality_allows_low_fill_factor_for_sparse_geometry():
    prediction = {
        "id": "prim-1",
        "classification": {
            "physical_properties": {
                "density": 1000,
                "estimated_mass_kg": 5,
            }
        },
    }
    dataset_entry = {
        "id": "prim-1",
        "metadata": {"world_bbox_meters": {"size": [1.0, 1.0, 1.0]}},
    }

    assert build_mass_scale_quality_warnings(prediction, dataset_entry) == []


def test_mass_scale_quality_parses_bbox_metrics_from_prompt_fallback():
    dataset_entry = {
        "id": "prim-1",
        "user_prompt": (
            "Context:\n"
            "Geometric info:\n"
            "  - Dimensions (meters): width=1.500m, height=2.000m, depth=3.000m\n"
            "  - Bounding box volume: 9.000000 m^3"
        ),
    }

    metrics = extract_bbox_metrics_meters(dataset_entry)

    assert metrics["size_m"] == [1.5, 2.0, 3.0]
    assert metrics["max_dimension_m"] == 3.0
    assert metrics["volume_m3"] == 9.0


def test_mass_scale_quality_parses_plain_and_unicode_volume_units():
    for volume_text in (
        "Bounding box volume: 9.000000 m3",
        "Bounding box volume: 9.000000 m³",
    ):
        metrics = extract_bbox_metrics_meters({"user_prompt": volume_text})

        assert metrics["volume_m3"] == 9.0


def test_mass_scale_quality_ignores_normal_handheld_part():
    prediction = {
        "id": "prim-1",
        "classification": {
            "physical_properties": {
                "density": 1200,
                "estimated_mass_kg": 0.25,
            }
        },
    }
    dataset_entry = {
        "id": "prim-1",
        "metadata": {
            "world_bbox_meters": {
                "size": [0.12, 0.04, 0.03],
            }
        },
    }

    assert build_mass_scale_quality_warnings(prediction, dataset_entry) == []


def test_mass_scale_quality_helpers_filter_and_dedupe_warnings():
    warning = {
        "code": "mass_scale_suspicious",
        "severity": "warning",
        "message": "synthetic",
        "details": {"source": "existing"},
    }
    existing = [warning, "not-a-warning", {"code": "other", "severity": "info"}]
    generated = [
        {**warning, "details": {"source": "generated"}},
        {"code": "other", "severity": "info"},
    ]

    assert has_mass_scale_suspicious_warning({"quality_warnings": [warning]})
    assert not has_mass_scale_suspicious_warning(
        {"quality_warnings": [{**warning, "severity": "info"}]}
    )
    merged = merge_quality_warnings(existing, generated)

    assert [item["code"] for item in merged] == [
        "mass_scale_suspicious",
        "other",
    ]
    assert merged[0]["details"] == {"source": "existing"}
