# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Physics identify-asset task defaults."""

import json

from physics_agent.tasks import identify_asset
from physics_agent.tasks.identify_asset import IdentifyAssetTask


class _RecordingVLM:
    def __init__(self):
        self.generate_kwargs = None

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return json.dumps(
            {
                "asset_type": "vehicle",
                "asset_subtype": "forklift",
                "asset_description": "A forklift",
                "confidence": "high",
                "reasoning": "Visible forks and mast",
            }
        )


def test_identify_asset_uses_default_vlm_temperature_when_invoke_kwargs_missing(
    tmp_path, monkeypatch
):
    """Identify-asset fallback should not hardcode the old 0.3 temperature."""
    monkeypatch.setattr(identify_asset, "DEFAULT_VLM_TEMPERATURE", 0.8)
    vlm = _RecordingVLM()

    IdentifyAssetTask().run(
        {
            "vlm": vlm,
            "composition_images": ["/tmp/view.png"],
            "output_dir": str(tmp_path),
        }
    )

    assert vlm.generate_kwargs["temperature"] == 0.8


def test_identify_asset_uses_vlm_config_temperature_when_invoke_kwargs_missing(
    tmp_path, monkeypatch
):
    """VLM config should be the fallback before the module default."""
    monkeypatch.setattr(identify_asset, "DEFAULT_VLM_TEMPERATURE", 0.8)
    vlm = _RecordingVLM()

    IdentifyAssetTask().run(
        {
            "vlm": vlm,
            "composition_images": ["/tmp/view.png"],
            "output_dir": str(tmp_path),
            "vlm_config": {"temperature": 0.6},
        }
    )

    assert vlm.generate_kwargs["temperature"] == 0.6


def test_identify_asset_treats_none_vlm_config_as_empty(tmp_path, monkeypatch):
    """Explicit None VLM config should fall back to the module default."""
    monkeypatch.setattr(identify_asset, "DEFAULT_VLM_TEMPERATURE", 0.8)
    vlm = _RecordingVLM()

    IdentifyAssetTask().run(
        {
            "vlm": vlm,
            "composition_images": ["/tmp/view.png"],
            "output_dir": str(tmp_path),
            "vlm_config": None,
        }
    )

    assert vlm.generate_kwargs["temperature"] == 0.8
