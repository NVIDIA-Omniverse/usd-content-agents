# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused coverage tests for USD renderer provisioning helpers."""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock

import pytest

from world_understanding.agentic.usd_tasks import renderer
from world_understanding.agentic.usd_tasks.renderer import (
    USDRendererProvisioningTask,
    parse_base_mode_settings,
    parse_camera_configuration,
    parse_focus_mode_settings,
    parse_occlusion_settings,
    parse_original_material_settings,
    validate_rendering_modes,
)
from world_understanding.functions.graphics import rendering_backend_factory
from world_understanding.functions.graphics.rendering import CameraFocusMode
from world_understanding.utils.object_store import ObjectStore


class _NoSensorBackend:
    def supports_sensors(self) -> bool:
        return False


class _SensorBackend:
    def __init__(self, supported: list[str]) -> None:
        self.supported = supported

    def supports_sensors(self) -> bool:
        return True

    def get_supported_sensor_modes(self) -> list[str]:
        return list(self.supported)


def test_validate_rendering_modes_edge_cases() -> None:
    assert validate_rendering_modes([], _NoSensorBackend()) == ([], [])

    valid, warnings = validate_rendering_modes(
        ["bad_mode", "depth"],
        _NoSensorBackend(),
    )
    assert valid == []
    assert "Invalid rendering modes" in warnings[0]
    assert "does not support sensor modes" in warnings[1]

    valid, warnings = validate_rendering_modes(
        ["depth", "linear_depth"],
        _SensorBackend(["depth"]),
    )
    assert valid == ["depth"]
    assert "linear_depth" in warnings[0]


def test_parse_per_mode_camera_references_and_string_modes(
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = {
        "rendering_modes": {
            "string_mode": "prim_only",
            "missing_reference": {"use_cameras_from": "not_defined"},
            "list_mode": ["+x"],
        }
    }

    parsed = parse_camera_configuration(config)

    assert sorted(parsed) == ["list_mode"]
    assert parsed["list_mode"][0].direction == "+x"
    assert "not yet defined" in caplog.text


def test_parse_per_mode_render_settings(caplog: pytest.LogCaptureFixture) -> None:
    config = {
        "rendering_modes": {
            "prim_only_original": {
                "skip_occluded_images": True,
                "camera_focus_mode": "stage",
                "use_original_materials": False,
                "base_mode": "prim_only",
            },
            "bad_focus": {"camera_focus_mode": "elsewhere"},
            "string_mode": "prim_only",
        }
    }

    assert parse_occlusion_settings(config) == {"prim_only_original": True}
    assert parse_focus_mode_settings(config) == {
        "prim_only_original": CameraFocusMode.STAGE
    }
    assert "Invalid camera_focus_mode" in caplog.text
    assert parse_original_material_settings(config) == {"prim_only_original": False}
    assert parse_base_mode_settings(config) == {"prim_only_original": "prim_only"}


def test_renderer_provisioning_warp_backend_with_sensor_modes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    class FakeWarpBackend(_SensorBackend):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(["depth"])
            calls.append(kwargs)

    listener = Mock()
    monkeypatch.setattr(
        rendering_backend_factory, "WarpRenderingBackend", FakeWarpBackend
    )
    monkeypatch.setattr(renderer, "get_listener", lambda *args, **kwargs: listener)

    context = {
        "renderer_config": {
            "backend": "warp",
            "device": "cpu",
            "color_boost": 1.25,
            "enable_shadows": False,
            "enable_backface_culling": True,
            "rendering_modes": {
                "prim_with_stage": {},
                "depth": {},
            },
        }
    }
    object_store = Mock(spec=ObjectStore)

    result = USDRendererProvisioningTask().run(context, object_store)

    assert calls == [
        {
            "device": "cpu",
            "color_boost": 1.25,
            "enable_shadows": False,
            "enable_backface_culling": True,
        }
    ]
    assert result["rendering_modes"] == ["prim_with_stage", "depth"]
    assert result["rgb_rendering_modes"] == ["prim_with_stage"]
    assert result["sensor_rendering_modes"] == ["depth"]
    assert result["rendering_backend"].supported == ["depth"]
    listener.info.assert_any_call("Using warp rendering backend")
    assert any(
        "Sensors=['depth']" in call.args[0] for call in listener.info.call_args_list
    )


def test_renderer_provisioning_mock_backend_falls_back_for_invalid_modes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    listener = Mock()
    monkeypatch.setattr(renderer, "get_listener", lambda *args, **kwargs: listener)
    context = {
        "renderer_config": {"backend": "mock"},
        "rendering_modes": "not_a_mode",
    }
    object_store = Mock(spec=ObjectStore)

    result = USDRendererProvisioningTask().run(context, object_store)

    listener.info.assert_any_call("Using mock rendering backend")

    assert result["rendering_modes"] == ["prim_with_stage", "prim_only"]
    assert result["rgb_rendering_modes"] == ["prim_with_stage", "prim_only"]
    assert result["sensor_rendering_modes"] == []
    assert any(
        "Invalid rendering modes" in call.args[0]
        for call in listener.warning.call_args_list
    )
    assert any(
        "No valid rendering modes" in call.args[0]
        for call in listener.warning.call_args_list
    )
