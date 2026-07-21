# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for recursive Physics Agent config-key diagnostics."""

from __future__ import annotations

import logging

from physics_agent.config.validator import ConfigValidator


def test_nested_predict_typo_warns_without_blocking_provider_extensions(
    caplog,
) -> None:
    config = {
        "project": {"name": "nested-key-test"},
        "input": {"usd_path": "asset.usd"},
        "steps": {
            "predict": {
                "max_worker": 8,
                "vlm": {
                    "backend": "custom-provider",
                    "base_url": "https://provider.example.test/v1",
                    "provider_extension": {"custom_option": True},
                },
            }
        },
    }

    with caplog.at_level(logging.WARNING):
        ConfigValidator().validate(config)

    assert "steps.predict.max_worker" in caplog.text
    assert "steps.predict.max_workers" in caplog.text
    assert "base_url" not in caplog.text
    assert "provider_extension" not in caplog.text


def test_non_predict_behavior_typo_reports_full_path_and_suggestion(caplog) -> None:
    config = {
        "project": {"name": "nested-key-test"},
        "input": {"usd_path": "asset.usd"},
        "steps": {"apply_physics": {"collision_aprox": "convexHull"}},
    }

    with caplog.at_level(logging.WARNING):
        ConfigValidator().validate(config)

    assert "steps.apply_physics.collision_aprox" in caplog.text
    assert "steps.apply_physics.collision_approx" in caplog.text
