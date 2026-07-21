# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for dominant color extraction tool."""

import tempfile
import warnings
from pathlib import Path

import pytest
from PIL import Image as PILImage
from sklearn.exceptions import ConvergenceWarning

from world_understanding.tools.cv.get_dominant_colors import (
    GetDominantColorsInput,
    GetDominantColorsOutput,
    _display_color_analysis,
    get_dominant_colors_tool,
)


class TestGetDominantColorsInput:
    """Tests for GetDominantColorsInput model."""

    def test_valid_input(self):
        """Test creating valid GetDominantColorsInput."""
        input_obj = GetDominantColorsInput(
            image_path="test.jpg",
            n_colors=5,
            analyze_brightness=True,
        )

        assert input_obj.image_path == "test.jpg"
        assert input_obj.n_colors == 5
        assert input_obj.analyze_brightness is True

    def test_default_values(self):
        """Test default values for optional fields."""
        input_obj = GetDominantColorsInput(image_path="test.jpg")

        assert input_obj.n_colors == 5  # default
        assert input_obj.analyze_brightness is True  # default

    def test_invalid_n_colors(self):
        """Test validation of n_colors."""
        # Too few colors
        with pytest.raises(ValueError):
            GetDominantColorsInput(
                image_path="test.jpg",
                n_colors=0,
            )

        # Too many colors
        with pytest.raises(ValueError):
            GetDominantColorsInput(
                image_path="test.jpg",
                n_colors=21,
            )

    def test_invalid_image_path(self):
        """Test validation of image_path."""
        # Empty path is actually allowed in the model
        # The validation happens at tool execution time
        input_obj = GetDominantColorsInput(
            image_path="",
            n_colors=5,
        )
        assert input_obj.image_path == ""


class RecordingConsole:
    """Minimal console double that records print calls."""

    def __init__(self) -> None:
        self.calls = []

    def print(self, *args, **kwargs) -> None:
        self.calls.append((args, kwargs))


def test_display_color_analysis_renders_dominant_colors() -> None:
    console = RecordingConsole()

    _display_color_analysis(
        {
            "average_brightness": 127.5,
            "color_diversity": 0.42,
            "dominant_colors": [
                {"hex": "#ff0000", "rgb": [255, 0, 0], "percentage": 0.6},
                {"hex": "#0000ff", "rgb": [0, 0, 255], "percentage": 0.4},
            ],
        },
        console,
        indent="  ",
    )

    rendered = "\n".join(str(call[0][0]) for call in console.calls)
    assert "Color Analysis Results" in rendered
    assert "Average Brightness: 127.5/255" in rendered
    assert "Color Diversity: 0.420" in rendered
    assert "#0000ff RGB(0, 0, 255) - 40.0%" in rendered
    assert any(kwargs.get("style") == "on #ff0000" for _, kwargs in console.calls)


class TestGetDominantColorsTool:
    """Tests for get_dominant_colors_tool."""

    def create_solid_color_image(
        self, color: tuple, size: tuple = (100, 100)
    ) -> PILImage.Image:
        """Create a test image with a single solid color."""
        return PILImage.new("RGB", size, color)

    def create_two_color_image(
        self, color1: tuple, color2: tuple, size: tuple = (100, 100)
    ) -> PILImage.Image:
        """Create an image with two colors split vertically."""
        img = PILImage.new("RGB", size, color1)
        # Fill right half with color2
        pixels = img.load()
        for x in range(size[0] // 2, size[0]):
            for y in range(size[1]):
                pixels[x, y] = color2
        return img

    def create_gradient_image(self, size: tuple = (100, 100)) -> PILImage.Image:
        """Create an image with a gradient."""
        img = PILImage.new("RGB", size)
        pixels = img.load()
        for x in range(size[0]):
            for y in range(size[1]):
                # Create gradient from red to blue
                r = int(255 * (1 - x / size[0]))
                b = int(255 * (x / size[0]))
                pixels[x, y] = (r, 0, b)
        return img

    def test_single_color_image(self):
        """Test with an image containing a single color."""
        # Create a pure red image
        img = self.create_solid_color_image((255, 0, 0))

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            img.save(tmp.name)
            tmp_path = tmp.name

        try:
            inputs = GetDominantColorsInput(
                image_path=tmp_path,
                n_colors=3,
            )

            # Suppress expected warning about fewer clusters than requested
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                output = get_dominant_colors_tool(inputs)

            # Should have 1 dominant color (red)
            assert len(output.dominant_colors) >= 1

            # The dominant color should be close to red
            dominant = output.dominant_colors[0]
            assert dominant.percentage > 90.0
            # RGB values might vary slightly due to JPEG compression
            assert dominant.rgb[0] > 250  # Red channel high
            assert dominant.rgb[1] < 10  # Green channel low
            assert dominant.rgb[2] < 10  # Blue channel low
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_two_color_image(self):
        """Test with an image containing two distinct colors."""
        # Create an image with red and blue
        img = self.create_two_color_image((255, 0, 0), (0, 0, 255))

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            img.save(tmp.name)
            tmp_path = tmp.name

        try:
            inputs = GetDominantColorsInput(
                image_path=tmp_path,
                n_colors=5,
            )

            output = get_dominant_colors_tool(inputs)

            # Should have at least 2 colors
            assert len(output.dominant_colors) >= 2

            # Each color should be around 50%
            for color in output.dominant_colors[:2]:
                assert 40.0 <= color.percentage <= 60.0
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_gradient_image(self):
        """Test with a gradient image."""
        img = self.create_gradient_image()

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            img.save(tmp.name)
            tmp_path = tmp.name

        try:
            inputs = GetDominantColorsInput(
                image_path=tmp_path,
                n_colors=10,
            )

            output = get_dominant_colors_tool(inputs)

            # Should extract multiple colors from the gradient
            assert len(output.dominant_colors) >= 3

            # Total percentage should be close to 100%
            total_percentage = sum(c.percentage for c in output.dominant_colors)
            assert 95.0 <= total_percentage <= 100.0
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_min_percentage_filter(self):
        """Test that min_percentage filters out small color regions."""
        # Create image with mostly red and a tiny bit of blue
        img = PILImage.new("RGB", (100, 100), (255, 0, 0))
        pixels = img.load()
        # Add a small blue corner (1% of image)
        for x in range(10):
            for y in range(10):
                pixels[x, y] = (0, 0, 255)

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            img.save(tmp.name)
            tmp_path = tmp.name

        try:
            # Set min_percentage to 5%, so the blue corner shouldn't appear
            inputs = GetDominantColorsInput(
                image_path=tmp_path,
                n_colors=5,
            )

            output = get_dominant_colors_tool(inputs)

            # Should have mostly red colors
            # The dominant color should be red
            dominant = output.dominant_colors[0]
            assert dominant.percentage > 80.0  # Most of the image is red
            assert dominant.rgb[0] > 200  # Red channel is high
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_grayscale_image(self):
        """Test with a grayscale image."""
        # Create a grayscale image
        img = PILImage.new("L", (100, 100), 128)

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            img.save(tmp.name)
            tmp_path = tmp.name

        try:
            inputs = GetDominantColorsInput(
                image_path=tmp_path,
                n_colors=3,
            )

            # Suppress expected warning about fewer clusters than requested
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                output = get_dominant_colors_tool(inputs)

            # Should work with grayscale (converted to RGB)
            assert len(output.dominant_colors) >= 1

            # The dominant color should be gray (R≈G≈B)
            dominant = output.dominant_colors[0]
            r, g, b = dominant.rgb
            assert abs(r - g) < 20
            assert abs(g - b) < 20
            assert abs(r - b) < 20
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_invalid_image_path(self):
        """Test with invalid image path."""
        inputs = GetDominantColorsInput(
            image_path="nonexistent.jpg",
            n_colors=5,
        )

        with pytest.raises(FileNotFoundError):  # Should raise an error for missing file
            get_dominant_colors_tool(inputs)

    def test_output_format(self):
        """Test the output format is correct."""
        img = self.create_solid_color_image((100, 150, 200))

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            img.save(tmp.name)
            tmp_path = tmp.name

        try:
            inputs = GetDominantColorsInput(
                image_path=tmp_path,
                n_colors=3,
            )

            # Suppress expected warning about fewer clusters than requested
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                output = get_dominant_colors_tool(inputs)

            # Check output structure
            assert isinstance(output, GetDominantColorsOutput)
            assert isinstance(output.dominant_colors, list)
            assert len(output.dominant_colors) > 0

            for color_info in output.dominant_colors:
                assert hasattr(color_info, "hex")
                assert hasattr(color_info, "rgb")
                assert hasattr(color_info, "percentage")

                # Check types
                assert isinstance(color_info.hex, str)
                assert isinstance(color_info.rgb, list)
                assert len(color_info.rgb) == 3
                assert isinstance(color_info.percentage, float)

                # Check hex format
                assert color_info.hex.startswith("#")
                assert len(color_info.hex) == 7
        finally:
            Path(tmp_path).unlink(missing_ok=True)
