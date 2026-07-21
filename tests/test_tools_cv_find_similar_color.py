# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for color matcher tool."""

import tempfile
from pathlib import Path

import pytest
from PIL import Image as PILImage
from PIL import ImageDraw

from world_understanding.tools.cv.find_similar_color import (
    FindSimilarColorInput,
    FindSimilarColorOutput,
    _display_color_match_results,
    find_similar_color_tool,
)


class TestFindSimilarColorInput:
    """Tests for FindSimilarColorInput model."""

    def test_valid_input(self):
        """Test creating valid FindSimilarColorInput."""
        input_obj = FindSimilarColorInput(
            image_path="test.jpg",
            target_color=[255, 0, 0],
            color_tolerance=30,
            min_percentage=5.0,
        )

        assert input_obj.image_path == "test.jpg"
        assert input_obj.target_color == [255, 0, 0]
        assert input_obj.color_tolerance == 30
        assert input_obj.min_percentage == 5.0

    def test_default_values(self):
        """Test default values for optional fields."""
        input_obj = FindSimilarColorInput(
            image_path="test.jpg",
            target_color=[0, 255, 0],
        )

        assert input_obj.color_tolerance == 50  # default
        assert input_obj.min_percentage == 1.0  # default

    def test_invalid_color_values(self):
        """Test validation of RGB color values."""
        # Test color value too high
        with pytest.raises(ValueError, match="RGB values must be between 0 and 255"):
            FindSimilarColorInput(
                image_path="test.jpg",
                target_color=[256, 0, 0],
            )

        # Test color value too low
        with pytest.raises(ValueError, match="RGB values must be between 0 and 255"):
            FindSimilarColorInput(
                image_path="test.jpg",
                target_color=[0, -1, 0],
            )

    def test_invalid_color_length(self):
        """Test validation of color list length."""
        # Too few values
        with pytest.raises(ValueError):
            FindSimilarColorInput(
                image_path="test.jpg",
                target_color=[255, 0],
            )

        # Too many values
        with pytest.raises(ValueError):
            FindSimilarColorInput(
                image_path="test.jpg",
                target_color=[255, 0, 0, 255],
            )

    def test_invalid_tolerance(self):
        """Test validation of color tolerance."""
        # Negative tolerance
        with pytest.raises(ValueError):
            FindSimilarColorInput(
                image_path="test.jpg",
                target_color=[255, 0, 0],
                color_tolerance=-1,
            )

        # Tolerance too high
        with pytest.raises(ValueError):
            FindSimilarColorInput(
                image_path="test.jpg",
                target_color=[255, 0, 0],
                color_tolerance=256,
            )

    def test_invalid_percentage(self):
        """Test validation of min_percentage."""
        # Negative percentage
        with pytest.raises(ValueError):
            FindSimilarColorInput(
                image_path="test.jpg",
                target_color=[255, 0, 0],
                min_percentage=-1.0,
            )

        # Percentage over 100
        with pytest.raises(ValueError):
            FindSimilarColorInput(
                image_path="test.jpg",
                target_color=[255, 0, 0],
                min_percentage=101.0,
            )


class TestFindSimilarColorOutput:
    """Tests for FindSimilarColorOutput model."""

    def test_valid_output(self):
        """Test creating valid FindSimilarColorOutput."""
        output = FindSimilarColorOutput(
            contains_color=True,
            matching_percentage=15.5,
            pixel_count=1550,
            total_pixels=10000,
            target_color_rgb=[255, 0, 0],
            target_color_hex="#ff0000",
            closest_colors=[],
        )

        assert output.contains_color is True
        assert output.matching_percentage == 15.5
        assert output.pixel_count == 1550
        assert output.total_pixels == 10000
        # No message field in the output model


class RecordingConsole:
    """Minimal console double that records print calls."""

    def __init__(self) -> None:
        self.calls = []

    def print(self, *args, **kwargs) -> None:
        self.calls.append((args, kwargs))


def test_display_color_match_results_renders_summary_and_closest_colors() -> None:
    console = RecordingConsole()

    _display_color_match_results(
        {
            "target_color_hex": "#ff0000",
            "target_color_rgb": [255, 0, 0],
            "contains_color": True,
            "matching_percentage": 12.345,
            "closest_colors": [
                {"hex": "#ff0000", "rgb": [255, 0, 0], "percentage": 0.5},
                {"hex": "#00ff00", "rgb": [0, 255, 0], "percentage": 0.25},
            ],
        },
        console,
        indent="  ",
    )

    rendered = "\n".join(str(call[0][0]) for call in console.calls)
    assert "Color Match Results" in rendered
    assert "Target Color: #ff0000 RGB(255, 0, 0)" in rendered
    assert "Status: [green]" in rendered
    assert "Found[/green]" in rendered
    assert "#00ff00 RGB(0, 255, 0) - 25.0%" in rendered
    assert any(kwargs.get("style") == "on #ff0000" for _, kwargs in console.calls)


class TestFindSimilarColorTool:
    """Tests for FindSimilarColorTool."""

    def create_test_image(
        self, color: tuple, size: tuple = (100, 100)
    ) -> PILImage.Image:
        """Create a test image with specified color."""
        img = PILImage.new("RGB", size, color)
        return img

    def create_mixed_image(self) -> PILImage.Image:
        """Create an image with multiple color regions."""
        img = PILImage.new("RGB", (100, 100), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Red rectangle (25% of image)
        draw.rectangle([0, 0, 49, 49], fill=(255, 0, 0))

        # Green rectangle (25% of image)
        draw.rectangle([50, 0, 99, 49], fill=(0, 255, 0))

        # Blue rectangle (25% of image)
        draw.rectangle([0, 50, 49, 99], fill=(0, 0, 255))

        # White remains (25% of image)

        return img

    def test_tool_with_exact_color_match(self):
        """Test tool with exact color match."""
        # Create a red image
        img = self.create_test_image((255, 0, 0))

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            img.save(tmp.name)
            tmp_path = tmp.name

        try:
            # Test with exact red color
            inputs = FindSimilarColorInput(
                image_path=tmp_path,
                target_color=[255, 0, 0],
                color_tolerance=10,
                min_percentage=90.0,
            )

            output = find_similar_color_tool(inputs)

            assert output.contains_color is True
            assert output.matching_percentage >= 90.0
            # Test passes - color found with high percentage
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_tool_with_no_color_match(self):
        """Test tool when color is not found."""
        # Create a blue image
        img = self.create_test_image((0, 0, 255))

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            img.save(tmp.name)
            tmp_path = tmp.name

        try:
            # Look for red in blue image
            inputs = FindSimilarColorInput(
                image_path=tmp_path,
                target_color=[255, 0, 0],
                color_tolerance=10,
                min_percentage=10.0,
            )

            output = find_similar_color_tool(inputs)

            assert output.contains_color is False
            assert output.matching_percentage < 10.0
            # Test passes - color not found
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_tool_with_mixed_colors(self):
        """Test tool with mixed color image."""
        img = self.create_mixed_image()

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            img.save(tmp.name)
            tmp_path = tmp.name

        try:
            # Look for red (should be ~25% of image)
            inputs = FindSimilarColorInput(
                image_path=tmp_path,
                target_color=[255, 0, 0],
                color_tolerance=10,
                min_percentage=20.0,
            )

            output = find_similar_color_tool(inputs)

            assert output.contains_color is True
            assert 20.0 <= output.matching_percentage <= 30.0
            assert output.pixel_count > 0
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_tool_with_tolerance(self):
        """Test color matching with different tolerance levels."""
        # Create an orange image (between red and yellow)
        img = self.create_test_image((255, 128, 0))

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            img.save(tmp.name)
            tmp_path = tmp.name

        try:
            # Low tolerance - should not match
            inputs_low = FindSimilarColorInput(
                image_path=tmp_path,
                target_color=[255, 0, 0],  # Pure red
                color_tolerance=10,
                min_percentage=50.0,
            )

            output_low = find_similar_color_tool(inputs_low)
            assert output_low.contains_color is False

            # High tolerance - should match
            inputs_high = FindSimilarColorInput(
                image_path=tmp_path,
                target_color=[255, 0, 0],  # Pure red
                color_tolerance=150,  # High tolerance
                min_percentage=50.0,
            )

            output_high = find_similar_color_tool(inputs_high)
            assert output_high.contains_color is True
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_tool_with_invalid_image(self):
        """Test tool with invalid image path."""
        inputs = FindSimilarColorInput(
            image_path="nonexistent.jpg",
            target_color=[255, 0, 0],
        )

        with pytest.raises(FileNotFoundError):  # Should raise an error for missing file
            find_similar_color_tool(inputs)

    def test_tool_with_grayscale_image(self):
        """Test tool with grayscale image."""
        # Create a grayscale image
        img = PILImage.new("L", (100, 100), 128)

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            img.save(tmp.name)
            tmp_path = tmp.name

        try:
            # Look for gray color
            inputs = FindSimilarColorInput(
                image_path=tmp_path,
                target_color=[128, 128, 128],
                color_tolerance=10,
                min_percentage=90.0,
            )

            output = find_similar_color_tool(inputs)

            # Should work as grayscale is converted to RGB
            assert output.contains_color is True
            assert output.matching_percentage >= 90.0
        finally:
            Path(tmp_path).unlink(missing_ok=True)
