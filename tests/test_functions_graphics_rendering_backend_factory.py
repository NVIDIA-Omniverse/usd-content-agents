# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the shared USD rendering-backend factory."""

from __future__ import annotations

from typing import Any

import pytest

from world_understanding.functions.graphics import rendering_backend_factory as factory
from world_understanding.functions.graphics.mock_rendering import MockRenderingBackend


class _Backend:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


def test_supported_rendering_backends_are_shared_agent_contract() -> None:
    assert factory.SUPPORTED_RENDERING_BACKENDS == {
        "remote",
        "warp",
        "ovrtx",
        "mock",
    }


def test_remote_slot_timeout_preserves_timeout_compatibility() -> None:
    assert issubclass(factory.RemoteRenderingSlotTimeoutError, TimeoutError)


def test_capability_subset_preserves_canonical_order() -> None:
    assert factory.rendering_backend_subset("mock", "remote", "ovrtx") == (
        "remote",
        "ovrtx",
        "mock",
    )


def test_capability_subset_rejects_unknown_and_duplicate_names() -> None:
    with pytest.raises(factory.UnknownRenderingBackendError):
        factory.rendering_backend_subset("remote", "typo")

    with pytest.raises(ValueError, match="Duplicate rendering backend"):
        factory.rendering_backend_subset("remote", "remote")


def test_surface_validation_distinguishes_unknown_from_unsupported() -> None:
    supported = factory.rendering_backend_subset("remote", "ovrtx")

    with pytest.raises(
        factory.UnknownRenderingBackendError,
        match="Supported by wu render-usd: remote, ovrtx",
    ):
        factory.validate_rendering_backend_for_surface(
            "typo",
            supported,
            surface="wu render-usd",
        )

    with pytest.raises(
        factory.UnsupportedRenderingBackendError,
        match="recognized but unsupported by wu render-usd",
    ):
        factory.validate_rendering_backend_for_surface(
            "mock",
            supported,
            surface="wu render-usd",
        )

    assert (
        factory.validate_rendering_backend_for_surface(
            "ovrtx",
            supported,
            surface="wu render-usd",
        )
        == "ovrtx"
    )


def test_create_remote_backend_forwards_only_remote_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NGC_API_KEY", "test-key")
    monkeypatch.setattr(factory, "RemoteRenderingBackend", _Backend)

    backend = factory.create_rendering_backend(
        "remote",
        {
            "base_url": "https://render.example",
            "timeout": 5,
            "material_target": "preview_surface",
            "device": "ignored",
        },
    )

    assert isinstance(backend, _Backend)
    assert backend.kwargs == {
        "api_key": "test-key",
        "base_url": "https://render.example",
        "timeout": 5,
        "material_target": "preview_surface",
    }


def test_create_warp_backend_uses_defaults_and_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(factory, "WarpRenderingBackend", _Backend)

    backend = factory.create_rendering_backend(
        "warp", {"device": "cuda:2", "enable_shadows": False}
    )

    assert isinstance(backend, _Backend)
    assert backend.kwargs == {
        "device": "cuda:2",
        "color_boost": 3.0,
        "enable_shadows": False,
        "enable_backface_culling": True,
    }


def test_create_ovrtx_backend_uses_defaults_and_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(factory, "OvRTXRenderingBackend", _Backend)

    backend = factory.create_rendering_backend(
        "ovrtx",
        {
            "log_level": "debug",
            "num_sensor_updates": 500,
            "render_mode": "pt",
            "material_target": "preview_surface",
        },
    )

    assert isinstance(backend, _Backend)
    assert backend.kwargs == {
        "log_level": "debug",
        "ovrtx_venv_dir": None,
        "num_sensor_updates": 500,
        "render_mode": "pt",
        "material_target": "preview_surface",
    }


def test_create_mock_backend_is_cpu_only() -> None:
    backend = factory.create_rendering_backend("mock", {"ignored": True})

    assert isinstance(backend, MockRenderingBackend)


def test_create_rendering_backend_rejects_unknown_name() -> None:
    with pytest.raises(
        factory.UnknownRenderingBackendError,
        match=(
            "Unknown rendering backend: typo. "
            "Supported backends: mock, ovrtx, remote, warp"
        ),
    ):
        factory.create_rendering_backend("typo")


def test_create_rendering_backend_rejects_non_string_name() -> None:
    with pytest.raises(
        factory.UnknownRenderingBackendError,
        match=r"Unknown rendering backend: \['mock'\]",
    ):
        factory.create_rendering_backend(["mock"])
