# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Security tests for the executor's final logging boundary."""

from __future__ import annotations

import logging

import pytest

from ...service.workers import executor

_HOSTILE = "forged\nAuthorization: Bearer joint-secret\r\x1b[31m"


def test_pipeline_stats_log_uses_only_bounded_fixed_schema_fields(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger=executor.__name__)

    executor._log_pipeline_stats(
        {
            "prims_processed": _HOSTILE,
            "images_generated": -1,
            "predictions_made": 2**80,
            "articulation_candidates": {"model_output": _HOSTILE},
            "free_form": _HOSTILE,
        }
    )

    messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == executor.__name__
    ]
    assert messages == [
        "Pipeline stats: prims_processed=0 images_generated=0 "
        "predictions_made=2147483647 articulation_candidates=0"
    ]
    assert all("\n" not in message and "\r" not in message for message in messages)
    assert all(
        "\x1b" not in message and "joint-secret" not in message for message in messages
    )
