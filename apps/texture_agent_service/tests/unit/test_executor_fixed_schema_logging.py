# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Security tests for Texture service logging boundaries."""

from __future__ import annotations

import logging

import pytest

from ...service.workers import executor

_HOSTILE = "forged\npassword=texture-secret\r\x1b]0;owned\x07"


class _HostileInt(int):
    def __str__(self) -> str:
        return _HOSTILE

    def __repr__(self) -> str:
        return _HOSTILE


def test_failure_logs_emit_fixed_codes_without_exception_or_path_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.ERROR, logger=executor.__name__)

    executor._log_cancelled_step_drain_failure()
    executor._log_unhandled_pipeline_failure()
    executor._log_step_failure(
        step_index=_HostileInt(2),
        total_steps=_HostileInt(7),
    )
    executor._log_failed_artifact_sync()

    messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == executor.__name__
    ]
    assert messages == [
        "code=cancelled_step_drain_failed",
        "code=pipeline_unhandled_failure",
        "code=pipeline_step_failed step_index=2 total_steps=7",
        "code=failed_step_artifact_sync_failed",
    ]
    assert all("\n" not in message and "\r" not in message for message in messages)
    assert all(
        "\x1b" not in message and "texture-secret" not in message
        for message in messages
    )


def test_pipeline_stats_log_uses_only_bounded_fixed_schema_fields(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger=executor.__name__)

    executor._log_pipeline_stats(
        {
            "materials_found": _HOSTILE,
            "textures_generated": -3,
            "output_usd_count": 2**80,
            "renders_count": {"model_output": _HOSTILE},
            "textures_failed": _HOSTILE,
            "errors": [_HOSTILE],
        }
    )

    messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == executor.__name__
    ]
    assert messages == [
        "Pipeline stats: materials_found=0 textures_generated=0 "
        "output_usd_count=2147483647 renders_count=0 textures_failed=0"
    ]
    assert all("\n" not in message and "\r" not in message for message in messages)
    assert all(
        "\x1b" not in message and "texture-secret" not in message
        for message in messages
    )
