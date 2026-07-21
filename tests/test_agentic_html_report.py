# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path

import pytest
from PIL import Image

from world_understanding.agentic.utils import html_report as report


def _write_image(path: Path, *, mode: str = "RGB") -> None:
    image = Image.new(mode, (8, 6), (10, 20, 30, 200) if mode == "RGBA" else 127)
    if mode == "P":
        image.putpalette([0, 0, 0, 255, 255, 255] * 128)
    image.save(path)


def test_static_html_helpers_and_pricing_defaults() -> None:
    assert report.escape_html(None) == "N/A"
    assert (
        report.escape_html("A&B <tag> \"quote\" 'single'")
        == "A&amp;B &lt;tag&gt; &quot;quote&quot; &#39;single&#39;"
    )

    css = report.get_common_report_css()
    assert ".image-thumbnail" in css
    assert ".system-prompt-section" in css

    modal = report.get_image_modal_html()
    assert "showImage" in modal
    assert "closeModal" in modal

    assert report.format_system_prompt_section(None) == ""
    system_prompt_html = report.format_system_prompt_section("<system & prompt>")
    assert "&lt;system &amp; prompt&gt;" in system_prompt_html

    pricing = report.get_public_token_pricing_defaults_2026()
    assert pricing["gemini-3-pro-preview"]["prompt_tier_threshold_tokens"] == 200_000
    assert pricing["bedrock-claude-opus-4-1-v1"]["output_per_mtok_usd"] == 150.0

    assert report._format_price_per_mtok_usd(None)
    assert report._format_price_per_mtok_usd("not-a-price")
    assert report._format_price_per_mtok_usd(1.25) == "$1.2500"


def test_process_and_encode_image_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rgb_path = tmp_path / "rgb.png"
    rgba_path = tmp_path / "rgba.png"
    palette_path = tmp_path / "palette.png"
    raw_path = tmp_path / "raw.bin"
    _write_image(rgb_path)
    _write_image(rgba_path, mode="RGBA")
    _write_image(palette_path, mode="P")
    raw_path.write_bytes(b"not really an image")

    png_payload = base64.b64decode(
        report.process_and_encode_image(
            rgb_path, image_max_size=2, image_format="png", image_quality=91
        )
    )
    assert png_payload.startswith(b"\x89PNG")

    rgba_jpeg_payload = base64.b64decode(
        report.process_and_encode_image(rgba_path, image_format="jpeg")
    )
    assert rgba_jpeg_payload.startswith(b"\xff\xd8")

    palette_jpeg_payload = base64.b64decode(
        report.process_and_encode_image(
            palette_path, image_max_size=4, image_format="jpeg", image_quality=70
        )
    )
    assert palette_jpeg_payload.startswith(b"\xff\xd8")

    monkeypatch.setattr(
        report.PILImage,
        "open",
        lambda _path: (_ for _ in ()).throw(RuntimeError("decode failed")),
    )
    assert (
        base64.b64decode(report.process_and_encode_image(raw_path))
        == raw_path.read_bytes()
    )

    with pytest.raises(FileNotFoundError):
        report.process_and_encode_image(tmp_path / "missing.png")


def test_format_images_html_variants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    _write_image(first)
    _write_image(second)

    assert (
        report.format_images_html([]) == '<span style="color: #999;">No images</span>'
    )

    html = report.format_images_html(
        ["first.png", "missing.png", str(second)],
        base_dir=tmp_path,
        image_metadata=[{"vlm_prompt": "look <here>"}],
        image_format="jpeg",
        image_quality=75,
        image_max_size=4,
        max_display=2,
    )
    assert "data:image/jpeg;base64" in html
    assert "image-with-caption" in html
    assert "look &lt;here&gt;" in html
    assert "missing.png (not found)" in html
    assert "+1 more" in html

    no_caption_html = report.format_images_html([str(first)], image_metadata=[{}])
    assert 'class="image-thumbnail"' in no_caption_html
    assert "image-with-caption" not in no_caption_html

    monkeypatch.setattr(
        report,
        "process_and_encode_image",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    error_html = report.format_images_html([str(first)])
    assert "first.png (error)" in error_html


@pytest.mark.parametrize(
    ("model_name", "expected"),
    [
        ("bedrock-claude-opus-4-1", "bedrock-claude-opus-4-1-v1"),
        ("bedrock-claude-sonnet-4-5", "bedrock-claude-sonnet-4-5-v1"),
        ("bedrock-claude-3-7-sonnet", "bedrock-claude-3-7-sonnet-v1"),
        ("claude sonnet 4.5", "claude-sonnet-4.5"),
        ("claude-haiku-4-5", "claude-haiku-4-5-v1"),
        ("claude haiku 4.5", "claude-haiku-4-5-v1"),
        ("us.anthropic.claude-sonnet-4-v1", "us.anthropic.claude-sonnet-4-v1"),
        ("claude sonnet 4", "us.anthropic.claude-sonnet-4-v1"),
        ("gpt-5.2-2026", "gpt-5.2"),
        ("gpt-5.1-mini", "gpt-5.1"),
        ("gpt-5", "gpt-5"),
        ("gemini-2.5-flash-lite", "gemini-2.5-flash-lite"),
        ("gemini-2-5-flash-image", "gemini-2.5-flash-image"),
        ("gemini-2.5-flash", "gemini-2.5-flash"),
        ("gemini-2-5-pro", "gemini-2.5-pro"),
        ("gemini-3.1-pro-preview-image", "gemini-3-pro-image-preview"),
        ("gemini-3.1-pro-preview", "gemini-3-pro-preview"),
        ("gemini-2.0-flash", "gemini-2.5-flash"),
        ("gemini-2-0-pro", "gemini-2.5-pro"),
        ("gemini-1.5-flash", "gemini-2.5-flash"),
        ("gemini-1-5-pro", "gemini-2.5-pro"),
        ("gemini-1.0-pro", "gemini-2.5-pro"),
        ("gemini-custom-flash", "gemini-2.5-flash"),
        ("gemini-custom-pro", "gemini-2.5-pro"),
        ("gemini-custom", "gemini-2.5-flash"),
        ("unknown", None),
    ],
)
def test_canonicalize_pricing_key(model_name: str, expected: str | None) -> None:
    assert report._canonicalize_pricing_key(model_name) == expected


def test_format_cost_estimate_without_usage_lists_reference_prices() -> None:
    default_html = report.format_cost_estimate_section({"invocation_count": 0})
    assert "Token usage was not recorded" in default_html
    assert "gemini-2.5-flash" in default_html

    html = report.format_cost_estimate_section(
        None,
        pricing_defaults={
            "gemini-2.5-flash": {
                "input_per_mtok_usd": "bad",
                "input_per_mtok_usd_long": 0.45,
                "output_per_mtok_usd": None,
                "output_per_mtok_usd_long": 2.5,
                "prompt_tier_threshold_tokens": 1_000,
                "source_url": "https://prices.example/<gemini>",
            }
        },
    )

    assert "Token usage was not recorded" in html
    assert "gemini-2.5-flash" in html
    assert "1k" in html
    assert "https://prices.example/&lt;gemini&gt;" in html


@dataclass
class _Usage:
    model_name: str
    input_tokens: int
    output_tokens: int


def test_format_cost_estimate_with_models_and_tier_splits() -> None:
    html = report.format_cost_estimate_section(
        {
            "invocation_count": 3,
            "by_model": {
                "gemini-3.1-pro-preview": {
                    "input_tokens": 201_500,
                    "output_tokens": 150,
                },
                "gpt-5": {"input_tokens": 1_000, "output_tokens": 500},
                "unknown<model>": {"input_tokens": 10, "output_tokens": 20},
            },
            "all_usages": [
                {
                    "model_name": "gemini-3.1-pro-preview",
                    "input_tokens": 500,
                    "output_tokens": 50,
                },
                _Usage("gemini-3.1-pro-preview", 201_000, 100),
                {},
            ],
        },
        pricing_defaults={
            "gemini-3-pro-preview": {
                "input_per_mtok_usd": 2,
                "input_per_mtok_usd_long": 4,
                "output_per_mtok_usd": 12,
                "output_per_mtok_usd_long": 18,
                "prompt_tier_threshold_tokens": 200_000,
                "source_url": "https://prices.example/gemini",
                "notes": "tiered <pricing>",
            },
            "gpt-5": {
                "input_per_mtok_usd": 1.25,
                "output_per_mtok_usd": 10,
                "source_url": "https://prices.example/gpt",
                "notes": "",
            },
        },
    )

    assert 'data-kind="input_short"' in html
    assert 'data-kind="output_long"' in html
    assert 'data-input-tokens-short="500"' in html
    assert 'data-input-tokens-long="201000"' in html
    assert "tiered &lt;pricing&gt;" in html
    assert "unknown&lt;model&gt;" in html
    assert 'data-kind="input" type="number"' in html


def test_format_cost_estimate_with_aggregate_usage() -> None:
    html = report.format_cost_estimate_section(
        {
            "invocation_count": 1,
            "total_input_tokens": 12_345,
            "total_output_tokens": 678,
        },
        pricing_defaults={},
    )

    assert "aggregate" in html
    assert "12,345" in html
    assert 'data-model-canonical=""' in html


def test_validate_image_options(caplog: pytest.LogCaptureFixture) -> None:
    assert report.validate_image_options(None, None, None) == ("png", 85, None)
    assert report.validate_image_options("jpeg", 100, 128) == ("jpeg", 100, 128)
    assert report.validate_image_options("gif", 0, 0) == ("png", 85, None)
    assert "Invalid image format" in caplog.text
    assert "out of range" in caplog.text
    assert "invalid, ignoring" in caplog.text
