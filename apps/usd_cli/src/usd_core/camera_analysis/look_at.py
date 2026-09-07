# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed facade for deterministic look-at camera placement."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from usd_core.camera_analysis.contracts import (
    CameraPose,
    SceneAnalysisIR,
    VisibilityBackend,
)
from usd_core.camera_analysis.placement import (
    place_cameras_look_at as _place_cameras_look_at,
)


@dataclass(frozen=True)
class LookAtConfig:
    """Validated-by-placement inputs for one look-at selection run."""

    target_path: str
    camera_count: int = 4
    yaw_ranges: str | None = None
    occlusion_threshold: float = 0.4
    min_distance_m: float | None = None
    max_distance_m: float | None = None
    height_offset_m: float = 0.0
    min_height_m: float | None = None  # target-center-relative vertical offset
    max_height_m: float | None = None
    min_look_down_deg: float = 0.0
    max_look_down_deg: float = 90.0
    xy_bounds_m: tuple[tuple[float, float], tuple[float, float]] | None = None
    candidate_count: int = 72
    allow_fewer: bool = False
    seed: int = 0
    focal_length_mm: float = 35.0
    aperture_mm: float = 36.0


@dataclass(frozen=True)
class LookAtResult:
    """Accepted poses and their evidence-bearing placement report."""

    report: dict
    poses: tuple[CameraPose, ...]
    masks: np.ndarray | None = None


def place_cameras_look_at(
    scene: SceneAnalysisIR,
    backend: VisibilityBackend,
    bounds_m: tuple[np.ndarray, np.ndarray],
    *,
    config: LookAtConfig,
) -> LookAtResult:
    """Run the stable placement engine through the typed look-at API."""

    evaluation = _place_cameras_look_at(
        scene,
        backend,
        bounds_m,
        target_path=config.target_path,
        camera_count=config.camera_count,
        yaw_ranges=config.yaw_ranges,
        occlusion_threshold=config.occlusion_threshold,
        min_distance_m=config.min_distance_m,
        max_distance_m=config.max_distance_m,
        height_offset_m=config.height_offset_m,
        min_height_m=config.min_height_m,
        max_height_m=config.max_height_m,
        min_look_down_deg=config.min_look_down_deg,
        max_look_down_deg=config.max_look_down_deg,
        xy_bounds_m=config.xy_bounds_m,
        candidate_count=config.candidate_count,
        allow_fewer=config.allow_fewer,
        seed=config.seed,
        focal_length_mm=config.focal_length_mm,
        aperture_mm=config.aperture_mm,
    )
    return LookAtResult(
        report=evaluation.report,
        poses=evaluation.poses,
        masks=evaluation.masks,
    )


__all__ = ["LookAtConfig", "LookAtResult", "place_cameras_look_at"]
