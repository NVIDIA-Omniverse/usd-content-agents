# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for PDF to images conversion tool and function."""

import tempfile
from pathlib import Path

import pytest
from PIL import Image

from world_understanding.functions.graphics.pdf_to_images import convert_pdf_to_images
from world_understanding.tools.graphics.pdf_to_images import (
    ConvertPdfToImagesInput,
    ConvertPdfToImagesOutput,
    PageImageInfo,
    _display_pdf_conversion,
    convert_pdf_to_images_tool,
)


@pytest.fixture
def sample_pdf_path() -> Path:
    """Create a simple test PDF file using reportlab."""
    from reportlab.lib.colors import Color
    from reportlab.pdfgen import canvas

    # Create a temporary PDF with 3 pages
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        pdf_path = Path(f.name)

    # Create PDF with reportlab
    c = canvas.Canvas(str(pdf_path), pagesize=(200, 200))

    # Page 1: Red background
    c.setFillColor(Color(1, 0.8, 0.8))
    c.rect(0, 0, 200, 200, fill=1, stroke=0)
    c.setFillColor(Color(0, 0, 0))
    c.drawString(50, 100, "Page 1")
    c.showPage()

    # Page 2: Green background
    c.setFillColor(Color(0.8, 1, 0.8))
    c.rect(0, 0, 200, 200, fill=1, stroke=0)
    c.setFillColor(Color(0, 0, 0))
    c.drawString(50, 100, "Page 2")
    c.showPage()

    # Page 3: Blue background
    c.setFillColor(Color(0.8, 0.8, 1))
    c.rect(0, 0, 200, 200, fill=1, stroke=0)
    c.setFillColor(Color(0, 0, 0))
    c.drawString(50, 100, "Page 3")
    c.showPage()

    c.save()

    yield pdf_path

    # Cleanup
    pdf_path.unlink()


# Tests for core function


def test_convert_pdf_basic(sample_pdf_path: Path) -> None:
    """Test basic PDF conversion without saving to disk."""
    results = convert_pdf_to_images(pdf_path=sample_pdf_path)

    assert len(results) == 3
    assert all(r["page_number"] in (1, 2, 3) for r in results)
    assert all(isinstance(r["image"], Image.Image) for r in results)
    assert all(r["width"] > 0 for r in results)
    assert all(r["height"] > 0 for r in results)
    assert all(r["image_path"] is None for r in results)


def test_convert_pdf_with_output_dir(sample_pdf_path: Path) -> None:
    """Test PDF conversion with saving to disk."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        results = convert_pdf_to_images(
            pdf_path=sample_pdf_path, output_dir=output_dir, fmt="png"
        )

        assert len(results) == 3

        # Check all files were created
        for result in results:
            assert result["image_path"] is not None
            image_path = Path(result["image_path"])
            assert image_path.exists()
            assert image_path.suffix == ".png"
            assert image_path.parent == output_dir


def test_convert_pdf_page_range(sample_pdf_path: Path) -> None:
    """Test converting only a specific page range."""
    results = convert_pdf_to_images(pdf_path=sample_pdf_path, first_page=2, last_page=3)

    assert len(results) == 2
    assert results[0]["page_number"] == 2
    assert results[1]["page_number"] == 3


def test_convert_pdf_single_page(sample_pdf_path: Path) -> None:
    """Test converting a single page."""
    results = convert_pdf_to_images(pdf_path=sample_pdf_path, first_page=1, last_page=1)

    assert len(results) == 1
    assert results[0]["page_number"] == 1


def test_convert_pdf_different_dpi(sample_pdf_path: Path) -> None:
    """Test conversion with different DPI settings."""
    results_72 = convert_pdf_to_images(
        pdf_path=sample_pdf_path, dpi=72, first_page=1, last_page=1
    )
    results_300 = convert_pdf_to_images(
        pdf_path=sample_pdf_path, dpi=300, first_page=1, last_page=1
    )

    # Higher DPI should produce larger images
    assert results_300[0]["width"] > results_72[0]["width"]
    assert results_300[0]["height"] > results_72[0]["height"]


def test_convert_pdf_grayscale(sample_pdf_path: Path) -> None:
    """Test grayscale conversion."""
    results = convert_pdf_to_images(
        pdf_path=sample_pdf_path, grayscale=True, first_page=1, last_page=1
    )

    assert len(results) == 1
    img = results[0]["image"]
    assert img.mode == "L"  # Grayscale mode


def test_convert_pdf_jpeg_format(sample_pdf_path: Path) -> None:
    """Test conversion to JPEG format."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        results = convert_pdf_to_images(
            pdf_path=sample_pdf_path,
            output_dir=output_dir,
            fmt="jpeg",
            first_page=1,
            last_page=1,
        )

        assert len(results) == 1
        image_path = Path(results[0]["image_path"])
        assert image_path.suffix == ".jpeg"
        assert image_path.exists()


def test_convert_pdf_invalid_path() -> None:
    """Test error handling for non-existent PDF."""
    with pytest.raises(FileNotFoundError):
        convert_pdf_to_images(pdf_path="nonexistent.pdf")


def test_convert_pdf_invalid_dpi(sample_pdf_path: Path) -> None:
    """Test error handling for invalid DPI."""
    with pytest.raises(ValueError, match="DPI must be between"):
        convert_pdf_to_images(pdf_path=sample_pdf_path, dpi=1000)


def test_convert_pdf_invalid_format(sample_pdf_path: Path) -> None:
    """Test error handling for invalid format."""
    with pytest.raises(ValueError, match="Invalid format"):
        convert_pdf_to_images(pdf_path=sample_pdf_path, fmt="invalid")


def test_convert_pdf_invalid_page_range(sample_pdf_path: Path) -> None:
    """Test error handling for invalid page range."""
    with pytest.raises(ValueError):
        convert_pdf_to_images(pdf_path=sample_pdf_path, first_page=3, last_page=2)


# Tests for tool wrapper


def test_tool_basic_conversion(sample_pdf_path: Path) -> None:
    """Test tool wrapper with basic conversion."""
    inputs = ConvertPdfToImagesInput(pdf_path=str(sample_pdf_path))

    output = convert_pdf_to_images_tool(inputs)

    assert output.total_pages == 3
    assert output.converted_pages == 3
    assert len(output.pages) == 3
    assert output.output_directory is None


def test_tool_with_output_dir(sample_pdf_path: Path) -> None:
    """Test tool wrapper with output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        inputs = ConvertPdfToImagesInput(
            pdf_path=str(sample_pdf_path), output_dir=tmpdir, dpi=150
        )

        output = convert_pdf_to_images_tool(inputs)

        assert output.total_pages == 3
        assert output.converted_pages == 3
        assert output.output_directory == tmpdir

        # Check all pages have paths
        for page in output.pages:
            assert page.image_path is not None
            assert Path(page.image_path).exists()


def test_tool_page_range(sample_pdf_path: Path) -> None:
    """Test tool wrapper with page range."""
    inputs = ConvertPdfToImagesInput(
        pdf_path=str(sample_pdf_path), first_page=2, last_page=3
    )

    output = convert_pdf_to_images_tool(inputs)

    assert output.total_pages == 3
    assert output.converted_pages == 2
    assert len(output.pages) == 2
    assert output.pages[0].page_number == 2
    assert output.pages[1].page_number == 3


def test_tool_grayscale(sample_pdf_path: Path) -> None:
    """Test tool wrapper with grayscale conversion."""
    inputs = ConvertPdfToImagesInput(
        pdf_path=str(sample_pdf_path), grayscale=True, first_page=1, last_page=1
    )

    output = convert_pdf_to_images_tool(inputs)

    assert output.converted_pages == 1
    assert len(output.pages) == 1


class RecordingConsole:
    """Minimal console double that records print calls."""

    def __init__(self) -> None:
        self.calls = []

    def print(self, *args, **kwargs) -> None:
        self.calls.append((args, kwargs))


def test_display_pdf_conversion_renders_pages_and_output_dir() -> None:
    console = RecordingConsole()
    output = ConvertPdfToImagesOutput(
        pages=[
            PageImageInfo(
                page_number=1, image_path="/tmp/page-1.png", width=640, height=480
            ),
            PageImageInfo(page_number=2, image_path=None, width=320, height=240),
        ],
        total_pages=3,
        converted_pages=2,
        output_directory="/tmp/pdf-pages",
    )

    _display_pdf_conversion(output, console, indent="  ")

    rendered = "\n".join(str(call[0][0]) for call in console.calls)
    assert "PDF Conversion Results" in rendered
    assert "Converted: 2/3 pages" in rendered
    assert "Output Directory: /tmp/pdf-pages" in rendered
    assert "Page 1: 640x480px - /tmp/page-1.png" in rendered
    assert "Page 2: 320x240px - in-memory" in rendered


def test_tool_input_validation() -> None:
    """Test tool input validation."""
    # DPI too low
    with pytest.raises(ValueError):
        ConvertPdfToImagesInput(pdf_path="test.pdf", dpi=50)

    # DPI too high
    with pytest.raises(ValueError):
        ConvertPdfToImagesInput(pdf_path="test.pdf", dpi=700)

    # Invalid first_page
    with pytest.raises(ValueError):
        ConvertPdfToImagesInput(pdf_path="test.pdf", first_page=0)
