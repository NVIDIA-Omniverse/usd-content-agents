# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Direct texture-map generation for generated material libraries."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from world_understanding.utils.credentials import parse_env_reference

from material_agent.material_library_generation.schema import (
    MaterialRecipe,
    TextureMapSet,
)

_ALBEDO_PROMPT_SUFFIX = (
    "Create a flat, front-facing, seamless tileable PBR albedo/base-color "
    "texture map. No object, no perspective, no cast shadows, no text, no logos."
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TextureGenerationSettings:
    """Settings for direct texture-map generation."""

    texture_size: int = 1024
    backend: str | None = None
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    api_key_env_var: str | None = None
    seed: int | None = None
    color_correct_albedo: bool = True
    albedo_color_correction_strength: float = 1.0


def _color_to_rgb(color: tuple[float, float, float]) -> tuple[int, int, int]:
    return tuple(max(0, min(255, int(round(channel * 255)))) for channel in color)


def _load_reference_images(paths: tuple[str, ...]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for raw_path in paths:
        images.append(Image.open(raw_path).convert("RGB"))
    return images


def _create_image_generation_model(settings: TextureGenerationSettings) -> Any | None:
    if not settings.backend:
        return None

    from world_understanding.functions.models.image_generation_models import (
        create_image_generation_model,
    )

    kwargs: dict[str, Any] = {}
    if settings.model:
        kwargs["model"] = settings.model
    if settings.base_url:
        kwargs["base_url"] = settings.base_url
    api_key = settings.api_key
    if not api_key and settings.api_key_env_var:
        env_name = parse_env_reference(
            settings.api_key_env_var,
            allow_legacy_bare=True,
        )
        api_key = os.getenv(env_name) if env_name else None
    if api_key:
        kwargs["api_key"] = api_key
    return create_image_generation_model(settings.backend, **kwargs)


def _synthesize_albedo(
    recipe: MaterialRecipe, size: int, seed: int | None
) -> Image.Image:
    """Create a deterministic nonblank albedo map for tests/offline validation."""
    seed_value = (seed or 0) + sum(ord(ch) for ch in recipe.material_id)
    base = _color_to_rgb(recipe.base_color_hint)
    image = Image.new("RGB", (size, size), base)
    draw = ImageDraw.Draw(image, "RGBA")

    # Subtle deterministic variation prevents solid-color false confidence while
    # keeping tests stable and cheap.
    for index in range(max(8, size // 32)):
        x0 = _deterministic_span_value(seed_value, index, 1, size)
        y0 = _deterministic_span_value(seed_value, index, 2, size)
        x_width = _deterministic_span_value(
            seed_value,
            index,
            3,
            max(1, size // 16),
            max(2, size // 4),
        )
        y_height = _deterministic_span_value(
            seed_value,
            index,
            4,
            1,
            max(2, size // 32),
        )
        x1 = min(size, x0 + x_width)
        y1 = min(size, y0 + y_height)
        delta = _deterministic_span_value(seed_value, index, 5, -24, 25)
        color = tuple(max(0, min(255, channel + delta)) for channel in base)
        draw.rectangle((x0, y0, x1, y1), fill=(*color, 32))

    return image


def _deterministic_span_value(
    seed_value: int,
    index: int,
    salt: int,
    start: int,
    stop: int | None = None,
) -> int:
    if stop is None:
        stop = start
        start = 0
    if stop <= start:
        return start
    value = seed_value + (index + 1) * 1_103_515_245 + salt * 12_345
    value ^= value >> 16
    return start + (abs(value) % (stop - start))


def _flat_normal(size: int) -> Image.Image:
    return Image.new("RGB", (size, size), (128, 128, 255))


def _flat_orm(recipe: MaterialRecipe, size: int) -> Image.Image:
    roughness = int(round(recipe.pbr_hints.roughness * 255))
    metallic = int(round(recipe.pbr_hints.metallic * 255))
    return Image.new("RGB", (size, size), (255, roughness, metallic))


def _mean_rgb(image: Image.Image) -> tuple[float, float, float]:
    pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    mean = pixels.reshape(-1, 3).mean(axis=0)
    return tuple(float(channel) for channel in mean)


def _match_albedo_mean_to_base_color(
    image: Image.Image,
    base_color: tuple[float, float, float],
    strength: float,
) -> Image.Image:
    """Shift albedo mean toward the planned base color while preserving detail."""
    rgb_image = image.convert("RGB")
    if strength <= 0.0:
        return rgb_image

    pixels = np.asarray(rgb_image, dtype=np.float32) / 255.0
    current_mean = pixels.reshape(-1, 3).mean(axis=0)
    target_mean = np.asarray(base_color, dtype=np.float32)
    correction = (target_mean - current_mean) * float(strength)
    corrected = np.clip(pixels + correction, 0.0, 1.0)

    # Quantization and clipping can leave a small residual. One additional
    # bounded pass keeps generated materials auditable against base_color_hint.
    if strength >= 1.0:
        residual = target_mean - corrected.reshape(-1, 3).mean(axis=0)
        corrected = np.clip(corrected + residual, 0.0, 1.0)

    return Image.fromarray((corrected * 255.0).round().astype(np.uint8), "RGB")


def _generate_albedo_with_model(
    model: Any,
    recipe: MaterialRecipe,
    refs: list[Image.Image],
    size: int,
) -> Image.Image:
    prompts = [
        f"{recipe.appearance_prompt}. {_ALBEDO_PROMPT_SUFFIX}",
        (
            f"Generate only a seamless square PBR albedo texture map for "
            f"{recipe.name}: color={recipe.color or 'unspecified'}, "
            f"material={recipe.material or 'unspecified'}, "
            f"finish={recipe.finish or 'unspecified'}. "
            "Flat texture tile, no product, no scene, no lighting, no text."
        ),
    ]

    last_error: Exception | None = None
    for attempt, prompt in enumerate(prompts, start=1):
        try:
            albedo = model.generate(prompt, images=refs or None)
            if albedo is None:
                raise ValueError("image backend returned None")
            if albedo.size != (size, size):
                albedo = albedo.resize((size, size), Image.Resampling.LANCZOS)
            return albedo.convert("RGB")
        except Exception as exc:
            last_error = exc
            logger.warning(
                "Texture image generation failed for %s (attempt %d/%d): %s",
                recipe.material_id,
                attempt,
                len(prompts),
                exc,
            )

    raise RuntimeError(
        f"Texture image generation failed for {recipe.material_id}"
    ) from last_error


def generate_texture_maps(
    recipe: MaterialRecipe,
    output_dir: str | Path,
    settings: TextureGenerationSettings | None = None,
    image_model: Any | None = None,
) -> TextureMapSet:
    """Generate albedo/normal/ORM maps for one material recipe.

    When an image model or backend is supplied, only the albedo map is generated
    by the image model in this MVP. Normal and ORM maps are synthesized from PBR
    hints so the package remains deterministic and renderer-safe. This keeps the
    workflow as direct material-library synthesis rather than high-level Texture
    Agent texture projection.
    """
    settings = settings or TextureGenerationSettings()
    if settings.texture_size <= 0:
        raise ValueError("texture_size must be positive")
    if not 0.0 <= settings.albedo_color_correction_strength <= 1.0:
        raise ValueError("albedo_color_correction_strength must be in [0, 1]")

    recipe.validate()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model = image_model or _create_image_generation_model(settings)
    if model is not None:
        refs = _load_reference_images(recipe.reference_image_uris)
        try:
            albedo = _generate_albedo_with_model(
                model,
                recipe,
                refs,
                settings.texture_size,
            )
        except RuntimeError as exc:
            logger.warning(
                "Falling back to synthesized albedo for %s after image backend failure: %s",
                recipe.material_id,
                exc,
            )
            albedo = _synthesize_albedo(recipe, settings.texture_size, settings.seed)
    else:
        albedo = _synthesize_albedo(recipe, settings.texture_size, settings.seed)

    if settings.color_correct_albedo:
        before_mean = _mean_rgb(albedo)
        albedo = _match_albedo_mean_to_base_color(
            albedo,
            recipe.base_color_hint,
            settings.albedo_color_correction_strength,
        )
        logger.debug(
            "Color-corrected albedo for %s from mean %s to %s",
            recipe.material_id,
            tuple(round(value, 4) for value in before_mean),
            tuple(round(value, 4) for value in _mean_rgb(albedo)),
        )

    normal = _flat_normal(settings.texture_size)
    orm = _flat_orm(recipe, settings.texture_size)

    albedo_path = output_path / "albedo.png"
    normal_path = output_path / "normal.png"
    orm_path = output_path / "orm.png"
    albedo.save(albedo_path)
    normal.save(normal_path)
    orm.save(orm_path)

    return TextureMapSet(albedo=albedo_path, normal=normal_path, orm=orm_path)
