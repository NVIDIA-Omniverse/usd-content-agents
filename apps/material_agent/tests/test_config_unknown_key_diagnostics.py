# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for recursive Material Agent config-key diagnostics."""

from __future__ import annotations

import logging

from material_agent.config.validator import ConfigValidator


def test_nested_predict_typo_reports_full_path_and_suggestion(caplog) -> None:
    credential_sentinel = "material-config-secret-754"
    signed_query_sentinel = "material-signed-query-secret-754"
    config = {
        "project": {"name": "nested-key-test"},
        "input": {"usd_path": "asset.usd"},
        "output": {},
        "materials": {"path": "materials.yaml"},
        "steps": {
            "predict": {
                "max_worker": 8,
                "unexpected_behavior": True,
                f"https://user:{credential_sentinel}@provider.example.test": True,
                f"?sv=2026-01-01&sig={signed_query_sentinel}": True,
                "vlm": {
                    "backend": "custom-provider",
                    "endpoint": "https://provider.example.test/v1",
                    "provider_extension": {"custom_option": True},
                },
            }
        },
    }

    with caplog.at_level(logging.WARNING):
        ConfigValidator().validate(config)

    assert "steps.predict.max_worker" in caplog.text
    assert "steps.predict.max_workers" in caplog.text
    assert "steps.predict.unexpected_behavior" in caplog.text
    assert credential_sentinel not in caplog.text
    assert signed_query_sentinel not in caplog.text
    assert "<redacted>" in caplog.text
    assert "endpoint" not in caplog.text
    assert "provider_extension" not in caplog.text
    assert "materials.path" not in caplog.text


def test_non_predict_behavior_typo_reports_full_path_and_suggestion(caplog) -> None:
    config = {
        "project": {"name": "nested-key-test"},
        "input": {"usd_path": "asset.usd"},
        "output": {},
        "steps": {"cluster_prims": {"max_worker": 4}},
    }

    with caplog.at_level(logging.WARNING):
        ConfigValidator().validate(config)

    assert "steps.cluster_prims.max_worker" in caplog.text
    assert "steps.cluster_prims.max_workers" in caplog.text


def test_incomplete_step_schemas_remain_open(caplog) -> None:
    config = {
        "project": {"name": "nested-key-test"},
        "input": {"usd_path": "asset.usd"},
        "output": {},
        "steps": {
            "build_dataset_prepare_dataset": {
                "pdf_conversion": {"dpi": 150},
            },
            "validate_predictions": {"allow_unknown_material": True},
        },
    }

    with caplog.at_level(logging.WARNING):
        ConfigValidator().validate(config)

    assert "steps.build_dataset_prepare_dataset.pdf_conversion" not in caplog.text
    assert "steps.validate_predictions.allow_unknown_material" not in caplog.text
