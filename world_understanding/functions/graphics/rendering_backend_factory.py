# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Create USD rendering backends from the shared agent configuration contract."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from world_understanding.functions.graphics.rendering import (
    OvRTXRenderingBackend,
    RemoteRenderingBackend,
    RenderingBackend,
    WarpRenderingBackend,
)
from world_understanding.rendering_backend_contract import (
    RENDERING_BACKEND_NAMES,
    SUPPORTED_RENDERING_BACKENDS,
    RemoteRenderingSlotTimeoutError,
    UnknownRenderingBackendError,
    UnsupportedRenderingBackendError,
    rendering_backend_subset,
    validate_rendering_backend_for_surface,
    validate_rendering_backend_name,
)

__all__ = [
    "RENDERING_BACKEND_NAMES",
    "SUPPORTED_RENDERING_BACKENDS",
    "RemoteRenderingSlotTimeoutError",
    "UnknownRenderingBackendError",
    "UnsupportedRenderingBackendError",
    "create_rendering_backend",
    "rendering_backend_subset",
    "validate_rendering_backend_for_surface",
    "validate_rendering_backend_name",
]

_REMOTE_CONFIG_KEYS = (
    "base_url",
    "s3_bucket",
    "s3_region",
    "s3_profile",
    "timeout",
    "max_retries",
    "retry_delay",
    "retry_backoff_factor",
    "retry_jitter",
    "bundle_mdl_assets",
    "use_data_uri",
    "add_preview_fallbacks",
    "material_target",
)
_OVRTX_CONFIG_KEYS = (
    "log_level",
    "ovrtx_venv_dir",
    "num_sensor_updates",
    "render_mode",
    "add_preview_fallbacks",
    "material_target",
)
_OVRTX_DEFAULTS: dict[str, Any] = {
    "log_level": "warn",
    "ovrtx_venv_dir": None,
    "num_sensor_updates": 32,
    "render_mode": "rt2",
}
_WARP_CONFIG_KEYS = (
    "device",
    "color_boost",
    "enable_shadows",
    "enable_backface_culling",
)
_WARP_DEFAULTS: dict[str, Any] = {
    "device": "cuda:0",
    "color_boost": 3.0,
    "enable_shadows": True,
    "enable_backface_culling": True,
}


def _select_config(config: Mapping[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    """Return only constructor options owned by one rendering backend."""
    return {key: config[key] for key in keys if key in config}


def create_rendering_backend(
    backend_type: object,
    config: Mapping[str, Any] | None = None,
) -> RenderingBackend:
    """Create a supported rendering backend and reject unknown names.

    ``remote``, ``warp``, ``ovrtx``, and ``mock`` are the shared backend names
    used by Material and Physics rendering tasks. Backend-specific constructor
    options are allowlisted here so adding a backend or option cannot update one
    selector while silently leaving another behind.

    Args:
        backend_type: Configured backend name.
        config: Rendering configuration containing backend constructor options.

    Raises:
        UnknownRenderingBackendError: If ``backend_type`` is not part of the
            shared contract.
    """
    backend_type = validate_rendering_backend_name(backend_type)

    config = config or {}

    if backend_type == "remote":
        kwargs = _select_config(config, _REMOTE_CONFIG_KEYS)
        kwargs["api_key"] = os.environ.get("NGC_API_KEY")
        return RemoteRenderingBackend(**kwargs)

    if backend_type == "warp":
        kwargs = {**_WARP_DEFAULTS, **_select_config(config, _WARP_CONFIG_KEYS)}
        return WarpRenderingBackend(**kwargs)

    if backend_type == "ovrtx":
        kwargs = {**_OVRTX_DEFAULTS, **_select_config(config, _OVRTX_CONFIG_KEYS)}
        return OvRTXRenderingBackend(**kwargs)

    from world_understanding.functions.graphics.mock_rendering import (
        MockRenderingBackend,
    )

    return MockRenderingBackend()
