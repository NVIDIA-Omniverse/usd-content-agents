# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Texture Agent capability contract for USD rendering backends."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from world_understanding.rendering_backend_contract import (
    rendering_backend_subset,
    validate_rendering_backend_for_surface,
)

DEFAULT_TEXTURE_RENDERING_BACKEND = "remote"
TEXTURE_PRODUCTION_RENDERING_BACKENDS = rendering_backend_subset("ovrtx")
TEXTURE_RENDERING_BACKENDS = rendering_backend_subset(
    "remote",
    "ovrtx",
    "mock",
)
_TEXTURE_RENDERING_STEPS = ("render_previews", "render")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_OVRTX_RENDER_MODES = frozenset({"rt1", "rt2", "pt"})


def ovrtx_request_sha256(
    *,
    source_usd_sha256: object,
    rendered_usd_sha256: object,
    camera_paths: object,
    image_width: object,
    image_height: object,
    render_mode: object,
    num_sensor_updates: object,
) -> str | None:
    """Return a canonical OVRTX request digest, or None for invalid provenance."""
    if (
        not isinstance(source_usd_sha256, str)
        or _SHA256_RE.fullmatch(source_usd_sha256) is None
        or not isinstance(rendered_usd_sha256, Sequence)
        or isinstance(rendered_usd_sha256, str | bytes)
        or not rendered_usd_sha256
        or any(
            not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None
            for digest in rendered_usd_sha256
        )
        or not isinstance(camera_paths, Sequence)
        or isinstance(camera_paths, str | bytes)
        or not camera_paths
        or any(not isinstance(path, str) or not path for path in camera_paths)
        or not isinstance(image_width, int)
        or isinstance(image_width, bool)
        or image_width <= 0
        or not isinstance(image_height, int)
        or isinstance(image_height, bool)
        or image_height <= 0
        or render_mode not in _OVRTX_RENDER_MODES
        or not isinstance(num_sensor_updates, int)
        or isinstance(num_sensor_updates, bool)
        or num_sensor_updates <= 0
    ):
        return None
    payload = {
        "camera_paths": list(camera_paths),
        "image_height": image_height,
        "image_width": image_width,
        "num_sensor_updates": num_sensor_updates,
        "render_mode": render_mode,
        "rendered_usd_sha256": list(rendered_usd_sha256),
        "source_usd_sha256": source_usd_sha256,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def has_production_visual_evidence(
    backend_type: object,
    *,
    render_count: int,
    source_usd_sha256: object = None,
    expected_source_usd_sha256: object = None,
    ovrtx_metadata: Mapping[str, Any] | None = None,
    camera_paths: object = None,
    image_width: object = None,
    image_height: object = None,
    rendered_usd_sha256: object = None,
) -> bool:
    """Return whether saved images have complete digest-bound OVRTX provenance."""
    if backend_type not in TEXTURE_PRODUCTION_RENDERING_BACKENDS or render_count <= 0:
        return False
    if (
        not isinstance(source_usd_sha256, str)
        or not isinstance(expected_source_usd_sha256, str)
        or source_usd_sha256 != expected_source_usd_sha256
        or _SHA256_RE.fullmatch(source_usd_sha256) is None
        or not isinstance(ovrtx_metadata, Mapping)
        or ovrtx_metadata.get("renderer") != "OVRTX"
        or ovrtx_metadata.get("rendered_usd_sha256") != rendered_usd_sha256
    ):
        return False
    expected_request_sha256 = ovrtx_request_sha256(
        source_usd_sha256=source_usd_sha256,
        rendered_usd_sha256=rendered_usd_sha256,
        camera_paths=camera_paths,
        image_width=image_width,
        image_height=image_height,
        render_mode=ovrtx_metadata.get("render_mode"),
        num_sensor_updates=ovrtx_metadata.get("num_sensor_updates"),
    )
    return (
        expected_request_sha256 is not None
        and ovrtx_metadata.get("request_sha256") == expected_request_sha256
    )


def validate_texture_rendering_backend(
    backend_type: object,
    *,
    step_name: str,
) -> str:
    """Validate one Texture Agent rendering selector before task side effects."""
    return validate_rendering_backend_for_surface(
        backend_type,
        TEXTURE_RENDERING_BACKENDS,
        surface=f"Texture Agent steps.{step_name}.backend",
    )


def validate_texture_rendering_steps(steps: Mapping[str, Any]) -> None:
    """Validate both exposed Texture Agent USD rendering selectors."""
    for step_name in _TEXTURE_RENDERING_STEPS:
        step_config = steps.get(step_name)
        if not isinstance(step_config, Mapping):
            continue
        validate_texture_rendering_backend(
            step_config.get("backend", DEFAULT_TEXTURE_RENDERING_BACKEND),
            step_name=step_name,
        )
