# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Attested defaults for the supported official VoMP runtime."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

DEFAULT_VOMP_REVISION: Final = "ac6826f25ca5cfa122eb7bacb74be91b5a9eadf4"
DEFAULT_VOMP_ARTIFACT_SHA256: Final = {
    "config": "e5c123582c148698dc4ba07870d1e7e538a3f5ed92196c7143cd6ef04a601164",
    "geometry_checkpoint_dir": (
        "4b5ff2498751d4a431dae4486312a294ba2723ea332e5eaf01bd461f87f746c1"
    ),
    "matvae_checkpoint_dir": (
        "b091b1760fd2fdda87c9212fbd662d72743c93a7f21e2e134cdb5cd25604817d"
    ),
    "normalization_params_path": (
        "7195a9b0e6520973b63611af20dc5f578728ab37d842deb1ebbbd433b7e654f2"
    ),
}
DEFAULT_VOMP_NUM_VIEWS: Final = 150
DEFAULT_VOMP_IMAGE_WIDTH: Final = 512
DEFAULT_VOMP_IMAGE_HEIGHT: Final = 512
DEFAULT_VOMP_CAMERA_RADIUS: Final = 2.0
DEFAULT_VOMP_FOV_DEGREES: Final = 40.0
DEFAULT_VOMP_SEED: Final = 42
DEFAULT_VOMP_RENDER_MODE: Final = "rt2"
DEFAULT_VOMP_NUM_SENSOR_UPDATES: Final = 32
DEFAULT_VOMP_MATERIAL_TARGET: Final = "auto"
DEFAULT_VOMP_OVRTX_VENV_DIR: Final[str | None] = None
DEFAULT_VOMP_RENDER_CONFIG: Final[Mapping[str, int | float | str | None]] = (
    MappingProxyType(
        {
            "num_views": DEFAULT_VOMP_NUM_VIEWS,
            "image_width": DEFAULT_VOMP_IMAGE_WIDTH,
            "image_height": DEFAULT_VOMP_IMAGE_HEIGHT,
            "radius": DEFAULT_VOMP_CAMERA_RADIUS,
            "fov_degrees": DEFAULT_VOMP_FOV_DEGREES,
            "seed": DEFAULT_VOMP_SEED,
            "render_mode": DEFAULT_VOMP_RENDER_MODE,
            "num_sensor_updates": DEFAULT_VOMP_NUM_SENSOR_UPDATES,
            "material_target": DEFAULT_VOMP_MATERIAL_TARGET,
            "ovrtx_venv_dir": DEFAULT_VOMP_OVRTX_VENV_DIR,
        }
    )
)


__all__ = [
    "DEFAULT_VOMP_ARTIFACT_SHA256",
    "DEFAULT_VOMP_CAMERA_RADIUS",
    "DEFAULT_VOMP_FOV_DEGREES",
    "DEFAULT_VOMP_IMAGE_HEIGHT",
    "DEFAULT_VOMP_IMAGE_WIDTH",
    "DEFAULT_VOMP_MATERIAL_TARGET",
    "DEFAULT_VOMP_NUM_SENSOR_UPDATES",
    "DEFAULT_VOMP_NUM_VIEWS",
    "DEFAULT_VOMP_OVRTX_VENV_DIR",
    "DEFAULT_VOMP_RENDER_CONFIG",
    "DEFAULT_VOMP_RENDER_MODE",
    "DEFAULT_VOMP_REVISION",
    "DEFAULT_VOMP_SEED",
]
