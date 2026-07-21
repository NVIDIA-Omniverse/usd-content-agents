# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for shared blank-image detection."""

import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from world_understanding.utils.image_blankness import (
    _sample_rgb_pixels,
    analyze_image_blankness,
    image_is_blank,
)


def _nonblank_render_like_image() -> Image.Image:
    image = Image.new("RGB", (64, 64), (220, 220, 220))
    draw = ImageDraw.Draw(image)
    draw.rectangle([4, 4, 28, 60], fill=(40, 70, 160))
    draw.rectangle([34, 8, 58, 58], fill=(180, 60, 30))
    draw.line([0, 63, 63, 0], fill=(0, 0, 0), width=2)
    return image


def test_black_and_white_pngs_are_blank(tmp_path: Path) -> None:
    black_path = tmp_path / "black.png"
    white_path = tmp_path / "white.png"
    Image.new("RGB", (32, 32), (0, 0, 0)).save(black_path)
    Image.new("RGB", (32, 32), (255, 255, 255)).save(white_path)

    assert image_is_blank(black_path)
    assert image_is_blank(white_path)


def test_nonblank_render_like_image_is_not_blank(tmp_path: Path) -> None:
    image_path = tmp_path / "ladder_like_render.png"
    _nonblank_render_like_image().save(image_path)

    stats = analyze_image_blankness(image_path)

    assert not stats.blank
    assert stats.unique_colors >= 4
    assert stats.dominant_color_ratio < 0.99
    assert stats.to_dict()["blank"] is False


def test_near_uniform_png_is_blank_from_bytes(tmp_path: Path) -> None:
    image_path = tmp_path / "near_uniform.png"
    image = Image.new("RGB", (32, 32), (128, 128, 128))
    draw = ImageDraw.Draw(image)
    draw.point((0, 0), fill=(129, 129, 129))
    image.save(image_path)

    assert image_is_blank(image_path.read_bytes())


def test_sparse_high_contrast_object_on_background_is_not_blank() -> None:
    image = Image.new("RGB", (100, 100), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle([40, 40, 44, 44], fill=(0, 0, 0))

    stats = analyze_image_blankness(image)

    assert not stats.blank
    assert stats.strong_minority_pixel_ratio > 0


def test_transparent_background_with_visible_object_is_not_blank() -> None:
    image = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle([20, 20, 27, 27], fill=(20, 80, 220, 255))

    stats = analyze_image_blankness(image)

    assert not stats.blank
    assert stats.alpha_visible_ratio is not None
    assert stats.alpha_visible_ratio > 0


def test_fully_transparent_image_is_blank() -> None:
    image = Image.new("RGBA", (64, 64), (0, 0, 0, 0))

    stats = analyze_image_blankness(image)

    assert stats.blank
    assert stats.reason == "transparent"


def test_tiny_transparent_artifact_does_not_make_solid_image_nonblank() -> None:
    image = Image.new("RGBA", (64, 64), (180, 180, 180, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, 2, 2], fill=(180, 180, 180, 0))

    stats = analyze_image_blankness(image)

    assert stats.blank
    assert stats.reason in {"solid_color", "too_few_unique_colors"}


def test_transparent_rgb_noise_is_masked_before_blankness_metrics() -> None:
    image = Image.new("RGBA", (64, 64))
    for y in range(64):
        for x in range(64):
            image.putpixel((x, y), ((x * 11) % 256, (y * 7) % 256, 127, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, 2, 2], fill=(180, 180, 180, 255))

    stats = analyze_image_blankness(image)

    assert stats.blank
    assert stats.reason in {"too_few_unique_colors", "dominant_color"}


def test_subsampling_does_not_have_deterministic_column_blind_spots() -> None:
    image = Image.new("RGB", (300, 100), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.line([1, 0, 1, 99], fill=(0, 0, 0), width=1)

    stats = analyze_image_blankness(image, max_analysis_pixels=10_000)

    assert not stats.blank
    assert stats.strong_minority_pixel_ratio > 0


def test_subsampling_does_not_have_deterministic_row_blind_spots() -> None:
    image = Image.new("RGB", (300, 100), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.line([0, 1, 299, 1], fill=(0, 0, 0), width=1)

    stats = analyze_image_blankness(image, max_analysis_pixels=10_000)

    assert not stats.blank
    assert stats.strong_minority_pixel_ratio > 0


def test_dominant_color_reason_for_low_contrast_outlier() -> None:
    image = Image.new("RGB", (100, 100), (250, 250, 250))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, 9, 0], fill=(249, 249, 249))

    stats = analyze_image_blankness(image, min_unique=2, max_uniformity=0.99)

    assert stats.blank
    assert stats.reason == "dominant_color"


def test_near_uniform_luminance_reason_for_balanced_low_range_colors() -> None:
    image = Image.new("RGB", (4, 4))
    colors = [(100, 100, 100), (101, 101, 101), (102, 102, 102), (103, 103, 103)]
    for index, color in enumerate(colors * 4):
        image.putpixel((index % 4, index // 4), color)

    stats = analyze_image_blankness(
        image,
        min_unique=4,
        blank_std_threshold=2.0,
        blank_dynamic_range_threshold=3.0,
    )

    assert stats.blank
    assert stats.reason == "near_uniform_luminance"


def test_alpha_content_can_clear_uniform_blank_reason() -> None:
    image = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, 3, 3], fill=(0, 0, 0, 255))

    stats = analyze_image_blankness(image)

    assert not stats.blank
    assert stats.reason is None
    assert stats.alpha_visible_ratio == 0.16


def test_rgba_bytes_and_grayscale_modes_are_normalized() -> None:
    rgba = Image.new("RGBA", (4, 4), (20, 30, 40, 255))
    buffer = io.BytesIO()
    rgba.save(buffer, format="PNG")

    rgba_stats = analyze_image_blankness(memoryview(buffer.getvalue()))
    grayscale_stats = analyze_image_blankness(Image.new("L", (4, 4), 128))

    assert rgba_stats.mode == "RGBA"
    assert grayscale_stats.mode == "RGB"


def test_sample_rgb_pixels_converts_non_rgb_images() -> None:
    sampled = _sample_rgb_pixels(Image.new("L", (2, 2), 7), max_pixels=4)

    assert sampled.shape == (4, 3)
    assert np.all(sampled == 7)
