# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic mask-aware weathering post-processing and evidence metrics."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from apps.texture_gen_service_common import WeatheringControls
from apps.texture_gen_service_common.artifacts import local_path_from_file_uri
from apps.texture_gen_service_common.weathering_intent import (
    WeatheringEffect,
    infer_weathering_effect,
)

_REPETITION_LIMIT = 0.65
_LOCALIZATION_MIN = 0.95
_PBR_CORRELATION_MIN = 0.5


@dataclass(frozen=True)
class WeatheringPlan:
    """Normalized backend execution state derived from prompt intent."""

    effect: WeatheringEffect
    density: float = 0.25
    scale: float = 0.15
    directionality: float = 0.0
    direction_degrees: float = 0.0
    severity: float = 0.7
    editable_mask_uri: str | None = None
    protected_mask_uri: str | None = None


_PLAN_DEFAULTS: dict[WeatheringEffect, tuple[float, float, float]] = {
    "rust": (0.20, 0.12, 0.25),
    "dust": (0.30, 0.16, 0.35),
    "dirt": (0.25, 0.15, 0.20),
    "wear": (0.22, 0.10, 0.15),
}


def plan_weathering_request(
    *,
    text_prompt: str | None,
    controls: WeatheringControls | None,
    strength: float,
) -> WeatheringPlan | None:
    """Derive internal normalized controls from a natural-language request."""

    effect = infer_weathering_effect(text_prompt)
    if effect is None:
        if controls is not None:
            raise ValueError(
                "weathering mask controls require rust, dust, dirt, or wear "
                "intent in conditioning.text_prompt"
            )
        return None

    density, scale, directionality = _PLAN_DEFAULTS[effect]
    direction_degrees = 0.0
    prompt = (text_prompt or "").lower()
    if any(
        phrase in prompt
        for phrase in (
            "bottom",
            "base",
            "lower",
            "feet",
            "gravity",
            "water-trap",
            "water trap",
        )
    ):
        directionality = max(directionality, 0.5)
        direction_degrees = 90.0
    elif any(phrase in prompt for phrase in ("top", "upper", "overhead")):
        directionality = max(directionality, 0.35)
        direction_degrees = -90.0

    return WeatheringPlan(
        effect=effect,
        density=density,
        scale=scale,
        directionality=directionality,
        direction_degrees=direction_degrees,
        severity=strength,
        editable_mask_uri=(controls.editable_mask_uri if controls else None),
        protected_mask_uri=(controls.protected_mask_uri if controls else None),
    )


@dataclass(frozen=True)
class WeatheringOutputs:
    """Weathering-controlled maps and their digest-bound evidence."""

    albedo_uri: str
    orm_uri: str
    mask_uri: str
    metadata: dict[str, Any]
    auxiliary_artifacts: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _local_image_path(uri: str | None, *, label: str) -> Path:
    if not uri:
        raise ValueError(f"{label} is required")
    path = local_path_from_file_uri(uri)
    if path is None:
        path = Path(uri)
    if not path.is_file():
        raise ValueError(f"{label} is not a readable local file: {uri}")
    return path.resolve()


def _rgb(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    with Image.open(path) as image:
        converted = image.convert("RGB")
        if size is not None and converted.size != size:
            converted = converted.resize(size, Image.Resampling.LANCZOS)
        return np.asarray(converted, dtype=np.float32) / 255.0


def _mask_from_uri(
    uri: str | None,
    *,
    size: tuple[int, int],
    default: float,
    label: str,
) -> tuple[np.ndarray, dict[str, Any] | None]:
    if uri is None:
        return np.full((size[1], size[0]), default, dtype=np.float32), None
    path = _local_image_path(uri, label=label)
    with Image.open(path) as image:
        gray = image.convert("L")
        if gray.size != size:
            raise ValueError(
                f"{label} dimensions {gray.size} do not match generated atlas {size}"
            )
        array = np.asarray(gray, dtype=np.float32) / 255.0
    return array, {
        "uri": path.as_uri(),
        "width": size[0],
        "height": size[1],
        "sha256": _sha256(path),
    }


def _effect_score(
    generated: np.ndarray,
    source: np.ndarray,
    effect: str,
) -> np.ndarray:
    delta = np.mean(np.abs(generated - source), axis=2)
    red, green, blue = (generated[:, :, index] for index in range(3))
    if effect == "rust":
        appearance = (
            np.clip(red - blue, 0.0, 1.0) * 0.7
            + np.clip(
                red - 0.8 * green,
                0.0,
                1.0,
            )
            * 0.3
        )
    elif effect == "dust":
        brightness = np.mean(generated, axis=2) - np.mean(source, axis=2)
        chroma = np.max(generated, axis=2) - np.min(generated, axis=2)
        appearance = np.clip(brightness, 0.0, 1.0) * (1.0 - 0.5 * chroma)
    elif effect == "wear":
        appearance = np.clip(np.mean(generated, axis=2) - np.mean(source, axis=2), 0, 1)
    else:
        appearance = delta
    score = 0.7 * delta + 0.3 * appearance
    maximum = float(np.max(score))
    if maximum > 1e-8:
        score /= maximum
    return score.astype(np.float32)


def _shift_without_wrap(array: np.ndarray, dx: int, dy: int) -> np.ndarray:
    shifted = np.zeros_like(array)
    height, width = array.shape
    source_x0 = max(0, -dx)
    source_x1 = min(width, width - dx)
    source_y0 = max(0, -dy)
    source_y1 = min(height, height - dy)
    if source_x0 >= source_x1 or source_y0 >= source_y1:
        return shifted
    target_x0 = source_x0 + dx
    target_x1 = source_x1 + dx
    target_y0 = source_y0 + dy
    target_y1 = source_y1 + dy
    shifted[target_y0:target_y1, target_x0:target_x1] = array[
        source_y0:source_y1,
        source_x0:source_x1,
    ]
    return shifted


def _shape_score(score: np.ndarray, spec: WeatheringPlan) -> np.ndarray:
    height, width = score.shape
    blur_radius = max(0.5, spec.scale * min(width, height) / 24.0)
    score_image = Image.fromarray(np.clip(score * 255.0, 0, 255).astype(np.uint8))
    score = (
        np.asarray(
            score_image.filter(ImageFilter.GaussianBlur(radius=blur_radius)),
            dtype=np.float32,
        )
        / 255.0
    )

    if spec.directionality <= 0.0:
        return score
    radians = math.radians(spec.direction_degrees)
    distance = max(1, round(spec.scale * min(width, height) * spec.directionality))
    directional = score.copy()
    samples = 4
    for step in range(1, samples + 1):
        fraction = step / samples
        dx = round(math.cos(radians) * distance * fraction)
        dy = round(math.sin(radians) * distance * fraction)
        directional += _shift_without_wrap(score, dx, dy)
        directional += _shift_without_wrap(score, -dx, -dy)
    directional /= 1 + 2 * samples
    return score * (1.0 - spec.directionality) + directional * spec.directionality


def _density_mask(score: np.ndarray, density: float) -> np.ndarray:
    if density <= 0.0:
        return np.zeros_like(score)
    if density >= 1.0:
        return np.ones_like(score)
    threshold = float(np.quantile(score, 1.0 - density))
    spread = max(float(np.std(score)) * 0.2, 1.0 / 255.0)
    return np.clip((score - threshold) / spread + 0.5, 0.0, 1.0).astype(np.float32)


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    left_flat = left.astype(np.float64).ravel()
    right_flat = right.astype(np.float64).ravel()
    if float(np.std(left_flat)) <= 1e-12 or float(np.std(right_flat)) <= 1e-12:
        return None
    return float(np.corrcoef(left_flat, right_flat)[0, 1])


def _repetition_peak(mask: np.ndarray, scale: float) -> float:
    image = Image.fromarray(np.clip(mask * 255.0, 0, 255).astype(np.uint8))
    image.thumbnail((256, 256), Image.Resampling.BILINEAR)
    sample = np.asarray(image, dtype=np.float64) / 255.0
    sample -= float(np.mean(sample))
    energy = float(np.sum(sample * sample))
    if energy <= 1e-12:
        return 0.0
    spectrum = np.fft.fft2(sample)
    autocorrelation = np.fft.fftshift(np.fft.ifft2(np.abs(spectrum) ** 2).real / energy)
    height, width = sample.shape
    yy, xx = np.ogrid[:height, :width]
    center_y, center_x = height // 2, width // 2
    radius = np.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)
    minimum_shift = max(8.0, scale * min(width, height))
    valid = (radius >= minimum_shift) & (radius <= min(width, height) * 0.45)
    if not np.any(valid):
        return 0.0
    return max(0.0, float(np.max(autocorrelation[valid])))


def _source_orm(
    *,
    source_orm_path: Path | None,
    size: tuple[int, int],
    roughness: float,
    metalness: float,
) -> np.ndarray:
    if source_orm_path is not None and source_orm_path.is_file():
        return _rgb(source_orm_path, size=size)
    array = np.empty((size[1], size[0], 3), dtype=np.float32)
    array[:, :, 0] = 1.0
    array[:, :, 1] = min(1.0, max(0.0, roughness))
    array[:, :, 2] = min(1.0, max(0.0, metalness))
    return array


def _quality_failures(
    *,
    effect: str,
    localization: float,
    protected_delta: float,
    material_boundary_delta: float,
    repetition_peak: float,
    roughness_correlation: float | None,
    metallic_correlation: float | None,
    metallic_max_delta: float,
) -> list[str]:
    """Evaluate the fail-closed weathering quality contract."""
    failures: list[str] = []
    if localization < _LOCALIZATION_MIN:
        failures.append("weathering changes are not localized to the effect mask")
    if protected_delta > (0.5 / 255.0):
        failures.append("protected pixels changed")
    if material_boundary_delta > (0.5 / 255.0):
        failures.append("weathering crossed a source material boundary")
    if repetition_peak > _REPETITION_LIMIT:
        failures.append("weathering mask has a strong repeated spatial peak")
    if roughness_correlation is None or roughness_correlation < _PBR_CORRELATION_MIN:
        failures.append("roughness does not correlate with localized weathering")
    if effect == "rust":
        if metallic_correlation is None or metallic_correlation > -_PBR_CORRELATION_MIN:
            failures.append("metallic does not decrease with localized rust")
    elif metallic_max_delta > (0.5 / 255.0):
        failures.append("non-rust weathering changed metallic")
    return failures


def apply_weathering_plan(
    *,
    albedo_uri: str,
    orm_uri: str | None,
    source_albedo_path: Path,
    source_orm_path: Path | None,
    source_roughness: float,
    source_metalness: float,
    spec: WeatheringPlan,
    output_dir: Path,
) -> WeatheringOutputs:
    """Apply an internal prompt-derived plan and return localized PBR evidence.

    The generated maps remain candidate appearance inputs. The explicit effect
    mask localizes them; source occlusion is always preserved, rust lowers
    metallic only beneath that mask, and non-rust effects preserve metallic.
    """

    generated_albedo_path = _local_image_path(albedo_uri, label="generated albedo")
    generated_orm_path = _local_image_path(orm_uri, label="generated ORM")
    with Image.open(generated_albedo_path) as image:
        size = image.size
    with Image.open(generated_orm_path) as image:
        if image.size != size:
            raise ValueError(
                f"generated ORM dimensions {image.size} do not match albedo {size}"
            )

    generated_albedo = _rgb(generated_albedo_path)
    source_albedo = _rgb(source_albedo_path, size=size)
    generated_orm = _rgb(generated_orm_path)
    source_orm = _source_orm(
        source_orm_path=source_orm_path,
        size=size,
        roughness=source_roughness,
        metalness=source_metalness,
    )

    if spec.effect == "rust" and float(np.max(source_orm[:, :, 2])) < 0.5:
        raise ValueError("rust weathering requires a metallic source material")

    editable, editable_artifact = _mask_from_uri(
        spec.editable_mask_uri,
        size=size,
        default=1.0,
        label="editable mask",
    )
    protected, protected_artifact = _mask_from_uri(
        spec.protected_mask_uri,
        size=size,
        default=0.0,
        label="protected mask",
    )
    score = _shape_score(
        _effect_score(generated_albedo, source_albedo, spec.effect),
        spec,
    )
    material_editable = np.ones_like(editable)
    if spec.effect == "rust":
        material_editable = (source_orm[:, :, 2] >= 0.5).astype(np.float32)
    mask = (
        _density_mask(score, spec.density)
        * editable
        * (1.0 - protected)
        * material_editable
    )
    mask = np.clip(mask, 0.0, 1.0)
    blend = mask[:, :, None] * spec.severity

    final_albedo = source_albedo * (1.0 - blend) + generated_albedo * blend
    final_orm = source_orm.copy()
    final_orm[:, :, 0] = source_orm[:, :, 0]
    roughness_increase = (1.0 - source_orm[:, :, 1]) * blend[:, :, 0] * 0.75
    candidate_roughness = np.maximum(
        source_orm[:, :, 1],
        generated_orm[:, :, 1],
    )
    final_orm[:, :, 1] = np.maximum(
        source_orm[:, :, 1] + roughness_increase,
        source_orm[:, :, 1] * (1.0 - blend[:, :, 0])
        + candidate_roughness * blend[:, :, 0],
    )
    if spec.effect == "rust":
        final_orm[:, :, 2] = source_orm[:, :, 2] * (1.0 - blend[:, :, 0])
    else:
        final_orm[:, :, 2] = source_orm[:, :, 2]

    output_dir.mkdir(parents=True, exist_ok=True)
    final_albedo_path = output_dir / "weathered_albedo.png"
    final_orm_path = output_dir / "weathered_orm.png"
    mask_path = output_dir / "weathering_mask.png"
    Image.fromarray(np.clip(final_albedo * 255.0, 0, 255).astype(np.uint8)).save(
        final_albedo_path
    )
    Image.fromarray(np.clip(final_orm * 255.0, 0, 255).astype(np.uint8)).save(
        final_orm_path
    )
    Image.fromarray(np.clip(mask * 255.0, 0, 255).astype(np.uint8)).save(mask_path)

    albedo_delta = np.mean(np.abs(final_albedo - source_albedo), axis=2)
    changed = mask > (1.0 / 255.0)
    total_change = float(np.sum(albedo_delta))
    localization = (
        float(np.sum(albedo_delta[changed])) / total_change
        if total_change > 1e-12
        else 1.0
    )
    protected_pixels = protected > 0.0
    if np.any(protected_pixels):
        protected_delta = max(
            float(np.max(np.abs(final_albedo - source_albedo)[protected_pixels])),
            float(np.max(np.abs(final_orm - source_orm)[protected_pixels])),
        )
    else:
        protected_delta = 0.0
    material_protected_pixels = material_editable < 0.5
    if np.any(material_protected_pixels):
        material_boundary_delta = max(
            float(
                np.max(np.abs(final_albedo - source_albedo)[material_protected_pixels])
            ),
            float(np.max(np.abs(final_orm - source_orm)[material_protected_pixels])),
        )
    else:
        material_boundary_delta = 0.0
    roughness_correlation = _pearson(
        mask,
        final_orm[:, :, 1] - source_orm[:, :, 1],
    )
    metallic_correlation = _pearson(
        mask,
        final_orm[:, :, 2] - source_orm[:, :, 2],
    )
    repetition_peak = _repetition_peak(mask, spec.scale)

    failures = _quality_failures(
        effect=spec.effect,
        localization=localization,
        protected_delta=protected_delta,
        material_boundary_delta=material_boundary_delta,
        repetition_peak=repetition_peak,
        roughness_correlation=roughness_correlation,
        metallic_correlation=metallic_correlation,
        metallic_max_delta=float(
            np.max(np.abs(final_orm[:, :, 2] - source_orm[:, :, 2]))
        ),
    )

    evidence: dict[str, Any] = {
        "schema_version": "texture-weathering-evidence.v1",
        "status": "pass" if not failures else "fail",
        "effect": spec.effect,
        "internal_plan": asdict(spec),
        "metrics": {
            "mask_coverage": float(np.mean(mask)),
            "repetition_peak": repetition_peak,
            "localization_ratio": localization,
            "protected_max_delta": protected_delta,
            "material_boundary_max_delta": material_boundary_delta,
            "roughness_mask_correlation": roughness_correlation,
            "metallic_mask_correlation": metallic_correlation,
        },
        "thresholds": {
            "repetition_peak_max": _REPETITION_LIMIT,
            "localization_ratio_min": _LOCALIZATION_MIN,
            "pbr_correlation_min_abs": _PBR_CORRELATION_MIN,
            "protected_max_delta": 0.5 / 255.0,
            "material_boundary_max_delta": 0.5 / 255.0,
        },
        "failures": failures,
        "artifacts": {
            "source_albedo_sha256": _sha256(source_albedo_path),
            "candidate_albedo_sha256": _sha256(generated_albedo_path),
            "candidate_orm_sha256": _sha256(generated_orm_path),
            "final_albedo_sha256": _sha256(final_albedo_path),
            "final_orm_sha256": _sha256(final_orm_path),
            "weathering_mask_sha256": _sha256(mask_path),
        },
    }
    masks: dict[str, Any] = {
        "weathering": {
            "uri": mask_path.as_uri(),
            "width": size[0],
            "height": size[1],
            "sha256": evidence["artifacts"]["weathering_mask_sha256"],
        }
    }
    if editable_artifact is not None:
        masks["editable"] = editable_artifact
    if protected_artifact is not None:
        masks["protected"] = protected_artifact

    return WeatheringOutputs(
        albedo_uri=final_albedo_path.as_uri(),
        orm_uri=final_orm_path.as_uri(),
        mask_uri=mask_path.as_uri(),
        metadata=evidence,
        auxiliary_artifacts={"masks": masks},
    )
