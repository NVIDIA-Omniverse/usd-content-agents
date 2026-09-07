# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Security tests for Material service logging boundaries."""

from __future__ import annotations

import logging

import pytest

from ...service.routers import pipeline_router
from ...service.workers import executor

_HOSTILE = "forged\napi_key=material-secret\r\x1b[2J"


class _HostileInt(int):
    def __str__(self) -> str:
        return _HOSTILE

    def __repr__(self) -> str:
        return _HOSTILE


def test_prim_warning_log_formats_hostile_numeric_subclasses_as_numbers(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger=pipeline_router.__name__)

    pipeline_router._log_prim_count_warning(_HostileInt(101), _HostileInt(100))

    messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == pipeline_router.__name__
    ]
    assert messages == [
        "Input USD prim threshold exceeded: prim_count=101 threshold=100"
    ]
    assert all("\n" not in message and "\r" not in message for message in messages)
    assert all(
        "\x1b" not in message and "material-secret" not in message
        for message in messages
    )


def test_executor_logs_only_bounded_fixed_schema_fields(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger=executor.__name__)
    hostile_stats = {
        "original_prim_count": _HOSTILE,
        "prims_processed": -5,
        "images_generated": 2**80,
        "predictions_made": {"prompt": _HOSTILE},
        "materials_applied": _HOSTILE,
        "scene_assets_completed": _HOSTILE,
        "scene_assets_failed": _HOSTILE,
        "scene_validation_errors": _HOSTILE,
        "scene_validation_warnings": _HOSTILE,
    }
    hostile_coverage = {
        "target_count": _HOSTILE,
        "prepared_count": _HOSTILE,
        "predicted_count": _HOSTILE,
        "bound_count": _HOSTILE,
        "unbound_count": _HOSTILE,
        "warnings": [_HOSTILE],
    }

    executor._log_scene_pipeline_stats(hostile_stats)
    executor._log_pipeline_stats(hostile_stats)
    executor._log_material_coverage(hostile_coverage)

    messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == executor.__name__
    ]
    assert len(messages) == 3
    assert "images_generated=2147483647" in messages[0]
    assert "images_generated=2147483647" in messages[1]
    assert messages[2].endswith("targets=0 prepared=0 predicted=0 bound=0 unbound=0")
    assert all("\n" not in message and "\r" not in message for message in messages)
    assert all(
        "\x1b" not in message and "material-secret" not in message
        for message in messages
    )
