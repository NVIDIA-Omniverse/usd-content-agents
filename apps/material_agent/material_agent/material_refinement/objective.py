# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Finite feature objective for generated material artifacts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .color import srgb_to_linear
from .contracts import MaterialObjectiveWeights, MaterialRefinementGoal


@dataclass(frozen=True)
class MaterialFeatures:
    """Low-order material features used by the bounded objective."""

    base_color: tuple[float, float, float] | None
    roughness: float | None
    metallic: float | None
    color_source: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "base_color": list(self.base_color)
            if self.base_color is not None
            else None,
            "roughness": self.roughness,
            "metallic": self.metallic,
            "color_source": self.color_source,
        }


@dataclass(frozen=True)
class MaterialObjectiveScore:
    """Normalized lower-is-better score and its auditable components."""

    value: float
    components: dict[str, float]
    normalized_weights: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "value": self.value,
            "components": dict(self.components),
            "normalized_weights": dict(self.normalized_weights),
        }


def _alpha_weighted_mean_rgb(path: Path) -> tuple[float, float, float]:
    with Image.open(path) as image:
        rgba = np.asarray(image.convert("RGBA"), dtype=np.float64) / 255.0
    rgb = srgb_to_linear(rgba[..., :3]).reshape(-1, 3)
    alpha = rgba[..., 3].reshape(-1)
    weight_sum = float(alpha.sum())
    if weight_sum <= 0.0:
        raise ValueError(f"image has no visible pixels: {path.name}")
    mean = (rgb * alpha[:, None]).sum(axis=0) / weight_sum
    values = (float(mean[0]), float(mean[1]), float(mean[2]))
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"image mean is not finite: {path.name}")
    return values


def _mean_orm(path: Path) -> tuple[float, float]:
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
    mean = pixels.reshape(-1, 3).mean(axis=0)
    roughness = float(mean[1])
    metallic = float(mean[2])
    if not math.isfinite(roughness) or not math.isfinite(metallic):
        raise ValueError(f"ORM mean is not finite: {path.name}")
    return roughness, metallic


def resolve_target_features(
    target: MaterialRefinementGoal,
    source: MaterialFeatures | None = None,
) -> MaterialFeatures:
    """Return source-anchored proxy features when no model inference is available."""

    del target
    return MaterialFeatures(
        base_color=source.base_color if source is not None else None,
        roughness=source.roughness if source is not None else None,
        metallic=source.metallic if source is not None else None,
        color_source="source_albedo" if source is not None else None,
    )


def extract_material_features(
    *,
    albedo_path: Path,
    orm_path: Path,
) -> MaterialFeatures:
    """Read generated albedo and ORM maps into objective features."""

    roughness, metallic = _mean_orm(orm_path)
    return MaterialFeatures(
        base_color=_alpha_weighted_mean_rgb(albedo_path),
        roughness=roughness,
        metallic=metallic,
        color_source="generated_albedo",
    )


def score_material_features(
    target: MaterialFeatures,
    generated: MaterialFeatures,
    weights: MaterialObjectiveWeights,
) -> MaterialObjectiveScore:
    """Return a weighted normalized distance in the closed interval [0, 1]."""

    components: dict[str, float] = {}
    raw_weights: dict[str, float] = {}
    if target.base_color is not None:
        if generated.base_color is None:
            raise ValueError("generated material is missing base color")
        squared = sum(
            (actual - expected) ** 2
            for actual, expected in zip(
                generated.base_color,
                target.base_color,
                strict=True,
            )
        )
        components["color"] = math.sqrt(squared / 3.0)
        raw_weights["color"] = weights.color
    if target.roughness is not None:
        if generated.roughness is None:
            raise ValueError("generated material is missing roughness")
        components["roughness"] = abs(generated.roughness - target.roughness)
        raw_weights["roughness"] = weights.roughness
    if target.metallic is not None:
        if generated.metallic is None:
            raise ValueError("generated material is missing metallic")
        components["metallic"] = abs(generated.metallic - target.metallic)
        raw_weights["metallic"] = weights.metallic

    weight_sum = sum(raw_weights.values())
    if weight_sum <= 0.0:
        raise ValueError("objective has no positively weighted target components")
    normalized_weights = {
        name: weight / weight_sum for name, weight in raw_weights.items()
    }
    value = sum(components[name] * normalized_weights[name] for name in components)
    if not math.isfinite(value):
        raise ValueError("material target objective is not finite")
    return MaterialObjectiveScore(
        value=max(0.0, min(1.0, value)),
        components=components,
        normalized_weights=normalized_weights,
    )


def material_feature_distance(
    left: MaterialFeatures,
    right: MaterialFeatures,
    weights: MaterialObjectiveWeights,
) -> float:
    """Return normalized distance across features available on both materials."""

    components: list[tuple[float, float]] = []
    if left.base_color is not None and right.base_color is not None:
        squared = sum(
            (left_value - right_value) ** 2
            for left_value, right_value in zip(
                left.base_color, right.base_color, strict=True
            )
        )
        components.append((math.sqrt(squared / 3.0), weights.color))
    if left.roughness is not None and right.roughness is not None:
        components.append((abs(left.roughness - right.roughness), weights.roughness))
    if left.metallic is not None and right.metallic is not None:
        components.append((abs(left.metallic - right.metallic), weights.metallic))
    weight_sum = sum(weight for _, weight in components)
    if weight_sum <= 0.0:
        raise ValueError(
            "material diversity has no shared positively weighted features"
        )
    distance = sum(value * weight for value, weight in components) / weight_sum
    if not math.isfinite(distance):
        raise ValueError("material diversity distance is not finite")
    return max(0.0, min(1.0, distance))


__all__ = [
    "MaterialFeatures",
    "MaterialObjectiveScore",
    "extract_material_features",
    "material_feature_distance",
    "resolve_target_features",
    "score_material_features",
]
