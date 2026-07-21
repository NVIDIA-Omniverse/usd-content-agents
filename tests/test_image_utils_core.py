# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coverage for image and numpy utility helpers."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from world_understanding.utils import image_utils


def _rgb_test_image() -> Image.Image:
    image = Image.new("RGB", (5, 5), (0, 0, 0))
    pixels = image.load()
    assert pixels is not None
    pixels[1, 1] = (255, 0, 0)
    pixels[2, 2] = (0, 255, 0)
    pixels[3, 3] = (0, 0, 255)
    return image


def test_load_save_and_base64_image_round_trips(tmp_path: Path) -> None:
    image = _rgb_test_image()
    path = tmp_path / "image.png"
    image_utils.save_image(image, path)
    loaded = image_utils.load_image(path)
    assert loaded.size == (5, 5)

    with pytest.raises(FileNotFoundError, match="Image file not found"):
        image_utils.load_image(tmp_path / "missing.png")
    text_path = tmp_path / "not-image.txt"
    text_path.write_text("nope", encoding="utf-8")
    with pytest.raises(OSError, match="Failed to open image"):
        image_utils.load_image(text_path)

    class BadSaver:
        def save(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("cannot save")

    with pytest.raises(OSError, match="Failed to save image"):
        image_utils.save_image(BadSaver(), tmp_path / "bad.png")  # type: ignore[arg-type]

    encoded_png = image_utils.image_to_base64(image)
    assert image_utils.base64_to_image(encoded_png).size == image.size
    data_url = f"data:image/png;base64,{encoded_png}"
    assert image_utils.base64_to_image(data_url).size == image.size
    assert image_utils.image_to_base64(image.convert("RGB"), format="JPEG", quality=80)

    with pytest.raises(OSError, match="Failed to encode image"):
        image_utils.image_to_base64(image, format="not-a-format")
    with pytest.raises(ValueError, match="Invalid base64"):
        image_utils.base64_to_image("not-base64!!!")
    not_an_image = base64.b64encode(b"plain bytes").decode("ascii")
    with pytest.raises(OSError, match="Failed to decode base64"):
        image_utils.base64_to_image(not_an_image)

    output = tmp_path / "saved.png"
    assert image_utils.save_base64_image(encoded_png, str(output)) is True
    assert output.exists()
    assert image_utils.save_base64_image("invalid", str(tmp_path / "bad.png")) is False


def test_numpy_base64_helpers_and_error_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    array = np.array([[1, 2], [3, 4]], dtype=np.int16)
    encoded = image_utils.numpy_to_base64(array, dtype=np.float32)
    decoded = image_utils.base64_to_numpy(
        f"data:application/octet-stream;base64,{encoded}",
        dtype=np.float32,
        shape=(2, 2),
    )
    assert decoded.tolist() == [[1.0, 2.0], [3.0, 4.0]]

    class BadArray:
        def astype(self, _dtype: Any) -> Any:
            raise RuntimeError("bad cast")

    with pytest.raises(ValueError, match="Failed to encode numpy array"):
        image_utils.numpy_to_base64(BadArray(), dtype=np.float32)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Invalid base64 string or shape"):
        image_utils.base64_to_numpy("bad!!!", dtype=np.float32)
    with pytest.raises(ValueError, match="Invalid base64 string or shape"):
        image_utils.base64_to_numpy(encoded, dtype=np.float32, shape=(3, 3))

    def boom(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("frombuffer failed")

    monkeypatch.setattr(image_utils.np, "frombuffer", boom)
    with pytest.raises(ValueError, match="Failed to decode base64 to numpy"):
        image_utils.base64_to_numpy(encoded, dtype=np.float32)

    monkeypatch.undo()
    npy_path = tmp_path / "array.npy"
    assert (
        image_utils.save_base64_numpy(encoded, str(npy_path), dtype=np.float32) is True
    )
    assert npy_path.exists()
    assert image_utils.save_base64_numpy("invalid", str(tmp_path / "bad.npy")) is False


def test_resize_extract_data_url_and_depth_helpers() -> None:
    wide = Image.new("RGB", (400, 100), "red")
    tall = Image.new("RGB", (100, 400), "red")
    small = Image.new("RGB", (10, 8), "red")
    assert image_utils.resize_image(wide, max_long=200, max_short=80).size == (200, 50)
    assert image_utils.resize_image(tall, max_long=200, max_short=80).size == (50, 200)
    assert image_utils.resize_image(small, max_long=200, max_short=80) is small

    png_one = "iVBORw0KGgoAAAANSabc=="
    png_two = "iVBORw0KGgoAAAANSdef=="
    source = (
        "Recording time code: 0.1\n"
        "Camera: ignored\n"
        f"{png_one},\nRenderer plugin: HdStormRendererPlugin\n"
        "Running with Xvfb for GPU rendering...\n"
        f"{png_two}"
    )
    assert image_utils.extract_base64_strings(source) == [
        "iVBORw0KGgoAAAANSabc==",
        "iVBORw0KGgoAAAANSdef==",
    ]
    assert image_utils.extract_base64_strings("\n") == []
    assert image_utils.extract_base64_strings("no images here") == []

    data_url = image_utils.create_data_url(Image.new("RGB", (1, 1), "blue"))
    mime_type, data = image_utils.parse_data_url(data_url)
    assert mime_type == "image/png"
    assert data
    with pytest.raises(ValueError, match="must start"):
        image_utils.parse_data_url("image/png;base64,abc")
    with pytest.raises(ValueError, match="Invalid data URL format"):
        image_utils.parse_data_url("data:image/png;base64abc")

    varying = np.array([[np.inf, 2.0], [4.0, 6.0]], dtype=np.float32)
    processed = image_utils.process_depth_map(varying, min_output_value=0.25)
    assert processed[0, 0] == 0.0
    assert processed[0, 1] == 1.0
    assert processed[1, 1] == 0.25
    same = image_utils.process_depth_map(np.array([[np.inf, 3.0], [3.0, 3.0]]))
    assert same[0, 1] == 1.0
    all_background = image_utils.process_depth_map(np.array([[np.inf]]))
    assert all_background[0, 0] == 0.0


def test_outline_paste_background_and_visibility_helpers() -> None:
    image = _rgb_test_image()

    red_outline = image_utils.extract_red_outline(image)
    green_outline = image_utils.extract_green_outline(image)
    blue_outline = image_utils.extract_blue_outline(image)
    non_black_outline = image_utils.extract_non_black_outline(image)
    assert red_outline.mode == "L"
    assert green_outline.mode == "L"
    assert blue_outline.mode == "L"
    assert non_black_outline.mode == "L"

    with pytest.raises(ValueError, match="Invalid target channel"):
        image_utils.extract_color_outline(image, target_channel="x")

    pasted_outline = image_utils.paste_outline_to_image(
        image, red_outline, outline_color=(12, 34, 56)
    )
    assert pasted_outline.mode == "RGB"

    rgba = Image.new("RGBA", (1, 1), (255, 0, 0, 128))
    assert image_utils.paste_on_background(rgba, (0.0, 1.0, 0.0)).mode == "RGB"
    assert image_utils.paste_on_background(rgba, (300, -10, 30)).getpixel((0, 0))[0] > 0
    with pytest.raises(ValueError, match="RGBA"):
        image_utils.paste_on_background(image)
    with pytest.raises(ValueError, match="exactly 3"):
        image_utils.paste_on_background(rgba, (1, 2))  # type: ignore[arg-type]

    assert image_utils.draw_bounding_box_on_red(image).mode == "L"
    assert image_utils.draw_bounding_box_on_green(image).mode == "L"
    assert image_utils.draw_bounding_box_on_blue(image).mode == "L"
    empty_box = image_utils.draw_bounding_box_on_color(
        Image.new("RGB", (2, 2), "black")
    )
    assert np.array(empty_box).sum() == 0
    with pytest.raises(ValueError, match="Invalid target channel"):
        image_utils.draw_bounding_box_on_color(image, target_channel="x")

    assert bool(image_utils.is_prim_visible_in_image(image, pixel_threshold=1)) is True
    assert (
        bool(
            image_utils.is_prim_visible_in_image(
                image, contour_method="non_black", pixel_threshold=3
            )
        )
        is True
    )
    assert (
        bool(image_utils.is_prim_visible_in_image(image, pixel_threshold=10)) is False
    )
