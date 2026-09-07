# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Utilities for generating HTML reports in agentic workflows.

This module provides shared utilities for HTML report generation, including:
- HTML escaping
- Image formatting with base64 encoding
- Image processing (resize, format conversion)
- Common HTML/CSS/JavaScript components
"""

import base64
import io
import logging
from pathlib import Path
from typing import Any

from PIL import Image as PILImage

logger = logging.getLogger(__name__)


def escape_html(text: str | None) -> str:
    """Escape HTML special characters.

    Args:
        text: Text to escape (can be None)

    Returns:
        Escaped HTML string, or "N/A" if input is None
    """
    if text is None:
        return "N/A"

    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def process_and_encode_image(
    img_path: Path,
    image_max_size: int | None = None,
    image_format: str = "png",
    image_quality: int = 85,
) -> str:
    """Process an image (resize, convert format) and return base64-encoded data.

    Args:
        img_path: Path to the image file
        image_max_size: Optional max size in pixels (e.g., 256 for max 256x256)
        image_format: Output format ('png' or 'jpeg')
        image_quality: JPEG quality (1-100, default: 85)

    Returns:
        Base64-encoded image data as string

    Raises:
        Exception: If image processing or encoding fails
    """
    try:
        # Load image
        img = PILImage.open(img_path)

        # Convert RGBA to RGB if needed (JPEG doesn't support transparency)
        if image_format == "jpeg" and img.mode in ("RGBA", "LA", "P"):
            # Create white background
            background = PILImage.new("RGB", img.size, (255, 255, 255))
            if img.mode == "P":
                img = img.convert("RGBA")
            background.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
            img = background

        # Resize if max_size is specified
        if image_max_size:
            img.thumbnail((image_max_size, image_max_size), PILImage.Resampling.LANCZOS)

        # Encode to bytes in the specified format
        img_bytes = io.BytesIO()
        save_format = "JPEG" if image_format == "jpeg" else "PNG"

        if image_format == "jpeg":
            # JPEG with quality control
            img.save(
                img_bytes, format=save_format, quality=image_quality, optimize=True
            )
        else:
            # PNG with optimization
            img.save(img_bytes, format=save_format, optimize=True)

        # Base64 encode
        img_bytes.seek(0)
        img_data = base64.b64encode(img_bytes.getvalue()).decode()
        return img_data

    except Exception as e:
        logger.warning(f"Failed to process image {img_path}: {e}")
        # Fallback: try to read and encode original file
        try:
            with open(img_path, "rb") as f:
                return base64.b64encode(f.read()).decode()
        except Exception as e2:
            logger.error(f"Failed to encode image {img_path}: {e2}")
            raise


def format_images_html(
    images: list[str],
    base_dir: Path | None = None,
    image_metadata: list[dict[str, Any]] | None = None,
    image_max_size: int | None = None,
    image_format: str = "png",
    image_quality: int = 85,
    max_display: int = 1000,
) -> str:
    """Format images as HTML thumbnails with base64 encoding and captions.

    Args:
        images: List of image paths (can be relative or absolute)
        base_dir: Base directory for resolving relative paths (optional)
        image_metadata: Optional list of image metadata with prompts/captions
        image_max_size: Optional max image size in pixels (e.g., 256 for 256x256)
        image_format: Image format ('png' or 'jpeg', default: 'png')
        image_quality: JPEG quality (1-100, default: 85)
        max_display: Maximum number of images to display (default: 1000 - shows all)

    Returns:
        HTML string with image thumbnails and captions
    """
    if not images:
        return '<span style="color: #999;">No images</span>'

    images_html = []
    for idx, img in enumerate(images[:max_display]):
        # Get metadata for this image if available
        metadata = None
        image_prompt = None
        if image_metadata and idx < len(image_metadata):
            metadata = image_metadata[idx]
            image_prompt = metadata.get("vlm_prompt", "")

        try:
            # Resolve image path
            if base_dir:
                img_path = base_dir / img if not Path(img).is_absolute() else Path(img)
            else:
                img_path = Path(img)

            if img_path.exists():
                # Process image: load, resize (optional), convert format (optional), encode
                img_data = process_and_encode_image(
                    img_path,
                    image_max_size=image_max_size,
                    image_format=image_format,
                    image_quality=image_quality,
                )

                # Determine MIME type based on output format
                mime_type = "image/jpeg" if image_format == "jpeg" else "image/png"

                data_url = f"data:{mime_type};base64,{img_data}"

                # Create image with caption if prompt is available
                if image_prompt:
                    # Escape HTML in prompt
                    prompt_html = escape_html(image_prompt)

                    images_html.append(
                        f"""<div class="image-with-caption">
                            <img src="{data_url}" class="image-thumbnail" onclick="showImage('{data_url}')" alt="{escape_html(img)}">
                            <div class="image-caption">{prompt_html}</div>
                        </div>"""
                    )
                else:
                    images_html.append(
                        f'<img src="{data_url}" class="image-thumbnail" onclick="showImage(\'{data_url}\')" alt="{escape_html(img)}">'
                    )
            else:
                images_html.append(
                    f'<span style="color: #999;">{escape_html(img)} (not found)</span>'
                )
        except Exception as e:
            images_html.append(
                f'<span style="color: #F44336;">{escape_html(img)} (error)</span>'
            )
            logger.debug(f"Failed to load image {img}: {e}")

    if len(images) > max_display:
        images_html.append(
            f'<span style="color: #666; font-size: 10px;">+{len(images) - max_display} more</span>'
        )

    return '<div class="image-container">' + "".join(images_html) + "</div>"


def get_image_modal_html() -> str:
    """Get HTML for image modal preview.

    Returns:
        HTML string for modal overlay and JavaScript handlers
    """
    return """
    <!-- Modal for image preview -->
    <div id="imageModal" class="modal">
        <span class="close" onclick="closeModal()">&times;</span>
        <img class="modal-content" id="modalImage">
    </div>

    <script>
        function showImage(src) {
            var modal = document.getElementById('imageModal');
            var modalImg = document.getElementById('modalImage');
            modal.style.display = "block";
            modalImg.src = src;
        }

        function closeModal() {
            document.getElementById('imageModal').style.display = "none";
        }

        // Close modal when clicking outside the image
        window.onclick = function(event) {
            var modal = document.getElementById('imageModal');
            if (event.target == modal) {
                modal.style.display = "none";
            }
        }
    </script>
    """


def get_common_report_css() -> str:
    """Get common CSS styles for HTML reports.

    Returns:
        CSS string with common styles for reports
    """
    return """
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }
        .container {
            max-width: 1600px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 30px;
        }
        h1 {
            color: #333;
            border-bottom: 3px solid #2196F3;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        h2 {
            color: #555;
            margin-top: 30px;
            border-bottom: 1px solid #ddd;
            padding-bottom: 8px;
        }
        h3 {
            color: #666;
            margin-top: 20px;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .metric-card {
            padding: 15px;
            background: #f9f9f9;
            border-radius: 6px;
            border: 1px solid #e0e0e0;
        }
        .metric-label {
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #333;
            margin-top: 5px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            font-size: 12px;
        }
        th {
            background: #2196F3;
            color: white;
            text-align: left;
            padding: 12px;
            font-weight: 600;
            position: sticky;
            top: 0;
            z-index: 10;
        }
        td {
            padding: 12px;
            border-bottom: 1px solid #e0e0e0;
            vertical-align: top;
        }
        tr:hover {
            background: #f9f9f9;
        }
        .image-container {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 5px;
        }
        .image-with-caption {
            display: flex;
            flex-direction: column;
            align-items: center;
            max-width: 150px;
        }
        .image-caption {
            font-size: 11px;
            color: #666;
            margin-top: 4px;
            text-align: center;
            line-height: 1.2;
            word-wrap: break-word;
        }
        .image-thumbnail {
            max-width: 100px;
            max-height: 100px;
            object-fit: cover;
            border: 1px solid #ddd;
            border-radius: 4px;
            cursor: pointer;
        }
        .prompt-cell {
            max-width: 300px;
            max-height: 200px;
            overflow-y: auto;
            font-size: 12px;
            background: #f9f9f9;
            padding: 8px;
            border-radius: 4px;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        .response-cell {
            max-width: 400px;
            max-height: 200px;
            overflow-y: auto;
            font-size: 12px;
            background: #f9f9f9;
            padding: 8px;
            border-radius: 4px;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        .timestamp {
            color: #666;
            font-size: 12px;
            margin-top: 20px;
            text-align: right;
        }
        .system-prompt-section {
            margin: 30px 0;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
            border-left: 4px solid #2196F3;
        }
        .system-prompt-section h3 {
            margin-top: 0;
            color: #333;
            font-size: 14px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .system-prompt-content {
            margin-top: 15px;
            padding: 15px;
            background: white;
            border-radius: 4px;
            font-family: monospace;
            font-size: 10px;
            white-space: pre-wrap;
            word-wrap: break-word;
            max-height: 300px;
            overflow-y: auto;
            border: 1px solid #e0e0e0;
        }
        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            overflow: auto;
            background-color: rgba(0,0,0,0.8);
        }
        .modal-content {
            margin: auto;
            display: block;
            max-width: 90%;
            max-height: 90%;
            margin-top: 50px;
        }
        .close {
            position: absolute;
            top: 15px;
            right: 35px;
            color: #f1f1f1;
            font-size: 40px;
            font-weight: bold;
            cursor: pointer;
        }
    """


def format_system_prompt_section(system_prompt: str | None) -> str:
    """Format system prompt section HTML.

    Args:
        system_prompt: System prompt text (can be None)

    Returns:
        HTML string for system prompt section, or empty string if no prompt
    """
    if not system_prompt:
        return ""

    escaped_prompt = escape_html(system_prompt)
    return f"""
        <div class="system-prompt-section">
            <h3>📝 System Prompt</h3>
            <div class="system-prompt-content">{escaped_prompt}</div>
        </div>
    """


def get_public_token_pricing_defaults_2026() -> dict[str, dict[str, Any]]:
    """Get default per-1M-token pricing (USD) from public docs (2026).

    These are intended as editable defaults in HTML reports; users can override
    them in the report UI. Models without published public pricing are
    intentionally omitted so reports request pricing instead of guessing from a
    similarly named model.

    Returns:
        Dict keyed by a canonical model name with:
            - input_per_mtok_usd: float | None
            - input_per_mtok_usd_long: float | None
            - output_per_mtok_usd: float | None
            - output_per_mtok_usd_long: float | None
            - prompt_tier_threshold_tokens: int | None
            - source_url: str
            - notes: str
    """
    return {
        # Google Gemini (Developer API pricing, 2026)
        "gemini-2.5-flash": {
            "input_per_mtok_usd": 0.30,
            "output_per_mtok_usd": 2.50,
            "source_url": "https://ai.google.dev/gemini-api/docs/pricing",
            "notes": "Gemini 2.5 Flash pricing (public).",
        },
        "gemini-2.5-flash-image": {
            # Gemini pricing is not modality-split; treat as same as the base model.
            "input_per_mtok_usd": 0.30,
            "output_per_mtok_usd": 2.50,
            "source_url": "https://ai.google.dev/gemini-api/docs/pricing",
            "notes": "Gemini 2.5 Flash (image) pricing assumed same as Flash (public).",
        },
        "gemini-2.5-flash-lite": {
            "input_per_mtok_usd": 0.10,
            "output_per_mtok_usd": 0.40,
            "source_url": "https://ai.google.dev/gemini-api/docs/pricing",
            "notes": "Gemini 2.5 Flash-Lite pricing (public).",
        },
        "gemini-2.5-pro": {
            "input_per_mtok_usd": 1.25,
            "output_per_mtok_usd": 10.00,
            "source_url": "https://ai.google.dev/gemini-api/docs/pricing",
            "notes": "Gemini 2.5 Pro pricing (public). Output includes thinking tokens.",
        },
        "gemini-3-pro-preview": {
            "input_per_mtok_usd": 2.00,
            "input_per_mtok_usd_long": 4.00,  # prompts > 200k tokens
            "output_per_mtok_usd": 12.00,
            "output_per_mtok_usd_long": 18.00,  # prompts > 200k tokens
            "prompt_tier_threshold_tokens": 200_000,
            "source_url": "https://ai.google.dev/gemini-api/docs/pricing",
            "notes": "Gemini 3 Pro Preview. Tiered pricing for prompts >200k tokens.",
        },
        # Alias keys used in the report UI.
        "gemini-3.1-pro-preview": {
            "input_per_mtok_usd": 2.00,
            "input_per_mtok_usd_long": 4.00,  # prompts > 200k tokens
            "output_per_mtok_usd": 12.00,
            "output_per_mtok_usd_long": 18.00,  # prompts > 200k tokens
            "prompt_tier_threshold_tokens": 200_000,
            "source_url": "https://ai.google.dev/gemini-api/docs/pricing",
            "notes": "Alias of gemini-3-pro-preview for reference table display.",
        },
        "gemini-3-pro-image-preview": {
            "input_per_mtok_usd": 2.00,
            "input_per_mtok_usd_long": 4.00,  # prompts > 200k tokens
            "output_per_mtok_usd": 12.00,
            "output_per_mtok_usd_long": 18.00,  # prompts > 200k tokens
            "prompt_tier_threshold_tokens": 200_000,
            "source_url": "https://ai.google.dev/gemini-api/docs/pricing",
            "notes": (
                "Gemini 3 Pro (image preview) pricing assumed same as Gemini 3 Pro."
            ),
        },
        # OpenAI (Standard tier prices per 1M tokens)
        "gpt-5": {
            "input_per_mtok_usd": 1.25,
            "output_per_mtok_usd": 10.00,
            "source_url": "https://platform.openai.com/docs/pricing",
            "notes": "OpenAI API pricing (Standard tier).",
        },
        "gpt-5.1": {
            "input_per_mtok_usd": 1.25,
            "output_per_mtok_usd": 10.00,
            "source_url": "https://platform.openai.com/docs/pricing",
            "notes": "OpenAI API pricing (Standard tier).",
        },
        "gpt-5.2": {
            "input_per_mtok_usd": 1.75,
            "output_per_mtok_usd": 14.00,
            "source_url": "https://platform.openai.com/docs/pricing",
            "notes": "OpenAI API pricing (Standard tier).",
        },
        "claude-sonnet-4.5": {
            "input_per_mtok_usd": 3.00,
            "input_per_mtok_usd_long": 6.00,  # prompts > 200k input tokens (1M ctx)
            "output_per_mtok_usd": 15.00,
            "output_per_mtok_usd_long": 22.50,  # prompts > 200k input tokens (1M ctx)
            "prompt_tier_threshold_tokens": 200_000,
            "source_url": "https://docs.anthropic.com/en/docs/about-claude/pricing",
            "notes": "Claude Sonnet 4.5 base pricing; long-context tier applies when 1M context window is enabled and input exceeds 200k.",
        },
        "claude-haiku-4-5-v1": {
            "input_per_mtok_usd": 1.00,
            "output_per_mtok_usd": 5.00,
            "source_url": "https://docs.anthropic.com/en/docs/about-claude/pricing",
            "notes": "Claude Haiku 4.5 base pricing.",
        },
        "us.anthropic.claude-sonnet-4-v1": {
            "input_per_mtok_usd": 3.00,
            "input_per_mtok_usd_long": 6.00,  # prompts > 200k input tokens (1M ctx)
            "output_per_mtok_usd": 15.00,
            "output_per_mtok_usd_long": 22.50,  # prompts > 200k input tokens (1M ctx)
            "prompt_tier_threshold_tokens": 200_000,
            "source_url": "https://docs.anthropic.com/en/docs/about-claude/pricing",
            "notes": "Claude Sonnet 4 base pricing; long-context tier applies when 1M context window is enabled and input exceeds 200k.",
        },
        # Amazon Bedrock pricing varies by region and service tier.
        "bedrock-claude-opus-4-1-v1": {
            "input_per_mtok_usd": 30.00,
            "output_per_mtok_usd": 150.00,
            "source_url": "https://aws.amazon.com/bedrock/pricing/",
            "notes": "Assumes Amazon Bedrock Standard tier in us-east-2. Verify against AWS Bedrock pricing for your region/tier.",
        },
        "bedrock-claude-3-7-sonnet-v1": {
            "input_per_mtok_usd": 6.00,
            "output_per_mtok_usd": 30.00,
            "source_url": "https://aws.amazon.com/bedrock/pricing/",
            "notes": "Assumes Amazon Bedrock Standard tier in us-east-2. Verify against AWS Bedrock pricing for your region/tier.",
        },
        "bedrock-claude-sonnet-4-5-v1": {
            "input_per_mtok_usd": 6.00,
            "output_per_mtok_usd": 30.00,
            "source_url": "https://aws.amazon.com/bedrock/pricing/",
            "notes": "Assumes Amazon Bedrock Standard tier in us-east-2. Verify against AWS Bedrock pricing for your region/tier.",
        },
    }


def _format_price_per_mtok_usd(value: Any) -> str:
    """Render a per-1M-token USD price for HTML tables."""
    if value is None:
        return "—"
    try:
        return f"${float(value):.4f}"
    except (TypeError, ValueError):
        return "—"


def _canonicalize_pricing_key(model_name: str) -> str | None:
    name = model_name.lower()
    # Order matters: more specific first.

    # AWS Bedrock Claude models
    if "bedrock-claude-opus-4-1" in name:
        return "bedrock-claude-opus-4-1-v1"
    if "bedrock-claude-sonnet-4-5" in name:
        return "bedrock-claude-sonnet-4-5-v1"
    if "bedrock-claude-3-7-sonnet" in name:
        return "bedrock-claude-3-7-sonnet-v1"

    # Anthropic Claude models
    if (
        "claude sonnet 4.5" in name
        or "claude-sonnet-4.5" in name
        or "claude-sonnet-4-5" in name
    ):
        return "claude-sonnet-4.5"
    if "claude-haiku-4-5" in name:
        return "claude-haiku-4-5-v1"
    if "claude haiku 4.5" in name:
        return "claude-haiku-4-5-v1"
    if "us.anthropic.claude-sonnet-4-v1" in name:
        return "us.anthropic.claude-sonnet-4-v1"
    if "claude sonnet 4" in name:
        return "us.anthropic.claude-sonnet-4-v1"

    # OpenAI GPT models
    if "gpt-5.6-sol" in name:
        return "gpt-5.6-sol"
    if "gpt-5.2" in name:
        return "gpt-5.2"
    if "gpt-5.1" in name:
        return "gpt-5.1"
    if "gpt-5" in name:
        return "gpt-5"

    # Google Gemini models - order matters: more specific first
    # Gemini 2.5 models
    if "gemini-2.5-flash-lite" in name or "gemini-2-5-flash-lite" in name:
        return "gemini-2.5-flash-lite"
    if "gemini-2.5-flash-image" in name or "gemini-2-5-flash-image" in name:
        return "gemini-2.5-flash-image"
    if "gemini-2.5-flash" in name or "gemini-2-5-flash" in name:
        return "gemini-2.5-flash"
    if "gemini-2.5-pro" in name or "gemini-2-5-pro" in name:
        return "gemini-2.5-pro"

    # Gemini 3 models - check image variant before generic
    if "gemini-3.1-pro-preview-image" in name:
        return "gemini-3-pro-image-preview"
    if "gemini-3.1-pro-preview" in name:
        return "gemini-3-pro-preview"

    # Gemini 2.0 models (map to closest 2.5 equivalent)
    if "gemini-2.0-flash" in name or "gemini-2-0-flash" in name:
        return "gemini-2.5-flash"
    if "gemini-2.0-pro" in name or "gemini-2-0-pro" in name:
        return "gemini-2.5-pro"

    # Gemini 1.5 models (map to closest 2.5 equivalent)
    if "gemini-1.5-flash" in name or "gemini-1-5-flash" in name:
        return "gemini-2.5-flash"
    if "gemini-1.5-pro" in name or "gemini-1-5-pro" in name:
        return "gemini-2.5-pro"

    # Gemini 1.0 models (map to closest 2.5 equivalent)
    if "gemini-1.0-pro" in name or "gemini-1-0-pro" in name:
        return "gemini-2.5-pro"

    # Generic Gemini fallbacks (when version is unspecified or unrecognized)
    if "gemini" in name:
        if "flash" in name:
            return "gemini-2.5-flash"
        if "pro" in name:
            return "gemini-2.5-pro"
        # Default to flash for any unmatched gemini model
        return "gemini-2.5-flash"

    return None


def format_cost_estimate_section(
    token_stats: dict[str, Any] | None,
    pricing_defaults: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Format an interactive cost estimate section based on token usage.

    The resulting HTML contains input boxes for per-1M token prices so users can
    tune costs without regenerating the report.
    """
    if pricing_defaults is None:
        pricing_defaults = get_public_token_pricing_defaults_2026()

    if not token_stats or token_stats.get("invocation_count", 0) == 0:
        pricing_defaults = pricing_defaults or {}
        reference_keys = [
            "gemini-2.5-flash",
            "gemini-2.5-flash-image",
            "gemini-2.5-flash-lite",
            "gemini-2.5-pro",
            "gemini-3.1-pro-preview",
            "gemini-3-pro-image-preview",
            "gemini-3-pro-preview",
            "gpt-5.6-sol",
            "gpt-5",
            "gpt-5.1",
            "gpt-5.2",
            "claude-sonnet-4.5",
            "claude-haiku-4-5-v1",
            "bedrock-claude-opus-4-1-v1",
            "bedrock-claude-3-7-sonnet-v1",
            "us.anthropic.claude-sonnet-4-v1",
            "bedrock-claude-sonnet-4-5-v1",
        ]
        reference_rows = []
        for key in reference_keys:
            d = pricing_defaults.get(key, {})
            in_price_cell = _format_price_per_mtok_usd(d.get("input_per_mtok_usd"))
            in_price_long_cell = _format_price_per_mtok_usd(
                d.get("input_per_mtok_usd_long")
            )
            out_price_cell = _format_price_per_mtok_usd(d.get("output_per_mtok_usd"))
            out_price_long_cell = _format_price_per_mtok_usd(
                d.get("output_per_mtok_usd_long")
            )
            threshold = d.get("prompt_tier_threshold_tokens")
            threshold_cell = f"{threshold // 1000}k" if threshold else "—"
            src = str(d.get("source_url", "") or "")
            src_cell = (
                f"<a href='{escape_html(src)}' target='_blank' rel='noopener noreferrer'>{escape_html(src)}</a>"
                if src
                else "—"
            )
            reference_rows.append(
                f"<tr>"
                f"<td style='font-family: monospace;'>{escape_html(key)}</td>"
                f"<td>{in_price_cell}</td>"
                f"<td>{in_price_long_cell}</td>"
                f"<td>{out_price_cell}</td>"
                f"<td>{out_price_long_cell}</td>"
                f"<td>{threshold_cell}</td>"
                f"<td>{src_cell}</td>"
                f"</tr>"
            )
        return f"""
        <h2>💰 Cost Estimate</h2>
        <div style="color: #666; font-size: 12px;">
            Token usage was not recorded for this run, so an estimate cannot be computed.
            The reference prices below are from public provider docs. Models with tiered pricing show both base and long-context rates (applied when input exceeds the threshold).
        </div>
        <table style="margin-top: 1rem;">
            <thead>
                <tr>
                    <th>Model</th>
                    <th>$/1M Input</th>
                    <th>$/1M Input (Long)</th>
                    <th>$/1M Output</th>
                    <th>$/1M Output (Long)</th>
                    <th>Threshold</th>
                    <th>Source</th>
                </tr>
            </thead>
            <tbody>
                {"".join(reference_rows)}
            </tbody>
        </table>
        """

    by_model: dict[str, Any] = token_stats.get("by_model", {}) or {}
    all_usages = token_stats.get("all_usages", []) or []
    model_usages: dict[str, list[Any]] = {}

    def _usage_get(usage: Any, key: str, default: Any = None) -> Any:
        if isinstance(usage, dict):
            return usage.get(key, default)
        return getattr(usage, key, default)

    if by_model and isinstance(all_usages, list):
        for usage in all_usages:
            model_key = str(_usage_get(usage, "model_name", "unknown") or "unknown")
            model_usages.setdefault(model_key, []).append(usage)

    if by_model:
        rows = [
            {
                "model": str(model_name),
                "input_tokens": int(stats.get("input_tokens", 0)),
                "output_tokens": int(stats.get("output_tokens", 0)),
            }
            for model_name, stats in by_model.items()
        ]
    else:
        rows = [
            {
                "model": "aggregate",
                "input_tokens": int(token_stats.get("total_input_tokens", 0)),
                "output_tokens": int(token_stats.get("total_output_tokens", 0)),
            }
        ]

    # Build rows with default prices (best-effort mapping).
    table_rows: list[str] = []
    sources_seen: dict[str, str] = {}
    for idx, row in enumerate(rows):
        canonical = _canonicalize_pricing_key(row["model"])
        defaults = pricing_defaults.get(canonical, {}) if canonical else {}

        in_price_raw = defaults.get("input_per_mtok_usd", None)
        out_price_raw = defaults.get("output_per_mtok_usd", None)
        has_complete_pricing = in_price_raw is not None and out_price_raw is not None
        in_price = "" if in_price_raw is None else str(float(in_price_raw))
        out_price = "" if out_price_raw is None else str(float(out_price_raw))

        if defaults.get("source_url"):
            sources_seen[str(defaults["source_url"])] = str(defaults.get("notes", ""))

        prompt_tier_threshold = int(
            defaults.get("prompt_tier_threshold_tokens", 0) or 0
        )
        in_price_long = defaults.get("input_per_mtok_usd_long", None)
        out_price_long = defaults.get("output_per_mtok_usd_long", None)

        has_input_tier = prompt_tier_threshold > 0 and in_price_long is not None
        has_output_tier = prompt_tier_threshold > 0 and out_price_long is not None
        has_prompt_tier = has_input_tier or has_output_tier

        if has_input_tier:
            in_price_long = float(in_price_long)
        if has_output_tier:
            out_price_long = float(out_price_long)

        threshold_label = f"{prompt_tier_threshold // 1000}k" if has_prompt_tier else ""

        if has_input_tier:
            input_price_cell_html = f"""
                    <div style="display: flex; flex-direction: column; gap: 4px;">
                        <div style="display: flex; align-items: center; gap: 6px;">
                            <span style="font-size: 11px; color: #666; width: 44px;">≤{threshold_label}</span>
                            <input class="cost-price-input" data-kind="input_short" type="number" step="0.0001" min="0"
                                value="{in_price}" style="width: 90px;" />
                        </div>
                        <div style="display: flex; align-items: center; gap: 6px;">
                            <span style="font-size: 11px; color: #666; width: 44px;">&gt;{threshold_label}</span>
                            <input class="cost-price-input" data-kind="input_long" type="number" step="0.0001" min="0"
                                value="{in_price_long}" style="width: 90px;" />
                        </div>
                    </div>
            """
        else:
            input_price_cell_html = f"""
                    <input class="cost-price-input" data-kind="input" type="number" step="0.0001" min="0"
                        value="{in_price}" style="width: 110px;" />
            """

        if has_output_tier:
            output_price_cell_html = f"""
                    <div style="display: flex; flex-direction: column; gap: 4px;">
                        <div style="display: flex; align-items: center; gap: 6px;">
                            <span style="font-size: 11px; color: #666; width: 44px;">≤{threshold_label}</span>
                            <input class="cost-price-input" data-kind="output_short" type="number" step="0.0001" min="0"
                                value="{out_price}" style="width: 90px;" />
                        </div>
                        <div style="display: flex; align-items: center; gap: 6px;">
                            <span style="font-size: 11px; color: #666; width: 44px;">&gt;{threshold_label}</span>
                            <input class="cost-price-input" data-kind="output_long" type="number" step="0.0001" min="0"
                                value="{out_price_long}" style="width: 90px;" />
                        </div>
                    </div>
            """
        else:
            output_price_cell_html = f"""
                    <input class="cost-price-input" data-kind="output" type="number" step="0.0001" min="0"
                        value="{out_price}" style="width: 110px;" />
            """

        canonical_attr = escape_html(canonical or "")
        prompt_tier_attr = (
            f' data-prompt-tier-threshold="{prompt_tier_threshold}"'
            if has_prompt_tier
            else ""
        )

        tier_split_attr = ""
        if has_prompt_tier and row["model"] in model_usages:
            short_in = 0
            long_in = 0
            short_out = 0
            long_out = 0
            threshold = int(prompt_tier_threshold)
            for usage in model_usages.get(row["model"], []):
                u_in = int(_usage_get(usage, "input_tokens", 0) or 0)
                u_out = int(_usage_get(usage, "output_tokens", 0) or 0)
                if u_in > threshold:
                    long_in += u_in
                    long_out += u_out
                else:
                    short_in += u_in
                    short_out += u_out
            tier_split_attr = (
                f' data-input-tokens-short="{short_in}"'
                f' data-input-tokens-long="{long_in}"'
                f' data-output-tokens-short="{short_out}"'
                f' data-output-tokens-long="{long_out}"'
            )

        table_rows.append(
            f"""
            <tr class="cost-row" data-row-idx="{idx}" data-model-canonical="{canonical_attr}" data-pricing-complete="{str(has_complete_pricing).lower()}"{prompt_tier_attr}{tier_split_attr}>
                <td style="font-family: monospace;">{escape_html(row["model"])}</td>
                <td class="cost-input-tokens" data-value="{row["input_tokens"]}">{row["input_tokens"]:,}</td>
                <td class="cost-output-tokens" data-value="{row["output_tokens"]}">{row["output_tokens"]:,}</td>
                <td>
                    {input_price_cell_html}
                </td>
                <td>
                    {output_price_cell_html}
                </td>
                <td class="cost-estimate-usd" data-value="{"0" if has_complete_pricing else ""}">{"$0.0000" if has_complete_pricing else "Pricing required"}</td>
            </tr>
            """
        )

    sources_html = ""
    if sources_seen:
        items = []
        for url, note in sources_seen.items():
            note_suffix = f" — {escape_html(note)}" if note else ""
            items.append(
                f'<li><a href="{escape_html(url)}" target="_blank" rel="noopener noreferrer">{escape_html(url)}</a>{note_suffix}</li>'
            )
        sources_html = (
            '<div style="margin-top: 8px; font-size: 12px; color: #666;">'
            "<div><strong>Pricing sources (public):</strong></div>"
            f"<ul style='margin: 6px 0 0 18px;'>{''.join(items)}</ul>"
            "</div>"
        )

    # Always include explicit reference pricing for the requested models.
    reference_keys = [
        "gemini-2.5-flash",
        "gemini-2.5-flash-image",
        "gemini-2.5-flash-lite",
        "gemini-2.5-pro",
        "gemini-3.1-pro-preview",
        "gemini-3-pro-image-preview",
        "gemini-3-pro-preview",
        "gpt-5.6-sol",
        "gpt-5",
        "gpt-5.1",
        "gpt-5.2",
        "claude-sonnet-4.5",
        "claude-haiku-4-5-v1",
        "bedrock-claude-opus-4-1-v1",
        "bedrock-claude-3-7-sonnet-v1",
        "us.anthropic.claude-sonnet-4-v1",
        "bedrock-claude-sonnet-4-5-v1",
    ]
    reference_rows = []
    for key in reference_keys:
        d = pricing_defaults.get(key, {})
        in_price_cell = _format_price_per_mtok_usd(d.get("input_per_mtok_usd"))
        in_price_long_cell = _format_price_per_mtok_usd(
            d.get("input_per_mtok_usd_long")
        )
        out_price_cell = _format_price_per_mtok_usd(d.get("output_per_mtok_usd"))
        out_price_long_cell = _format_price_per_mtok_usd(
            d.get("output_per_mtok_usd_long")
        )
        threshold = d.get("prompt_tier_threshold_tokens")
        threshold_cell = f"{threshold // 1000}k" if threshold else "—"
        src = str(d.get("source_url", "") or "")
        src_cell = (
            f"<a href='{escape_html(src)}' target='_blank' rel='noopener noreferrer'>{escape_html(src)}</a>"
            if src
            else "—"
        )
        reference_rows.append(
            f"<tr>"
            f"<td style='font-family: monospace;'>{escape_html(key)}</td>"
            f"<td>{in_price_cell}</td>"
            f"<td>{in_price_long_cell}</td>"
            f"<td>{out_price_cell}</td>"
            f"<td>{out_price_long_cell}</td>"
            f"<td>{threshold_cell}</td>"
            f"<td>{src_cell}</td>"
            f"</tr>"
        )
    reference_table_html = f"""
    <details style="margin-top: 12px;">
        <summary style="cursor: pointer; color: #444; font-size: 12px;">
            Reference per-token prices (2026 public docs)
        </summary>
        <div style="margin-top: 8px; font-size: 12px; color: #666;">
            Prices shown are USD per 1M tokens. Models with tiered pricing show both base and long-context rates (applied when input exceeds the threshold).
        </div>
        <table style="margin-top: 0.75rem;">
            <thead>
                <tr>
                    <th>Model</th>
                    <th>$/1M Input</th>
                    <th>$/1M Input (Long)</th>
                    <th>$/1M Output</th>
                    <th>$/1M Output (Long)</th>
                    <th>Threshold</th>
                    <th>Source</th>
                </tr>
            </thead>
            <tbody>
                {"".join(reference_rows)}
            </tbody>
        </table>
    </details>
    """

    return f"""
    <h2>💰 Cost Estimate</h2>
    <div style="font-size: 12px; color: #666; margin-bottom: 10px;">
        Estimate computed from recorded token usage and editable USD prices (per 1M tokens).
        Adjust the prices below to update totals.
    </div>

    <div class="metrics-grid" style="margin-top: 0;">
        <div class="metric-card">
            <div class="metric-label">Estimated Total Cost (USD)</div>
            <div class="metric-value" id="costTotalUsd">$0.0000</div>
        </div>
    </div>

    <table style="margin-top: 1rem;">
        <thead>
            <tr>
                <th>Model</th>
                <th>Input Tokens</th>
                <th>Output Tokens</th>
                <th>$/1M Input</th>
                <th>$/1M Output</th>
                <th>Estimated Cost</th>
            </tr>
        </thead>
        <tbody>
            {"".join(table_rows)}
        </tbody>
    </table>
    {sources_html}
    {reference_table_html}

    <script>
        function _wuParseFloatOrZero(v) {{
            var x = parseFloat(v);
            return isNaN(x) ? 0.0 : x;
        }}

        function _wuFormatUsd(x) {{
            return "$" + x.toFixed(4);
        }}

        function _wuHasCompletePricing(row) {{
            var inputs = row.querySelectorAll("input.cost-price-input");
            return Array.from(inputs).every(function(input) {{
                return input.value.trim() !== "";
            }});
        }}

        function _wuGetInputPricePerMtok(row, inputTokens) {{
            var shortEl = row.querySelector("input.cost-price-input[data-kind='input_short']");
            var longEl = row.querySelector("input.cost-price-input[data-kind='input_long']");
            if (shortEl && longEl) {{
                var threshold = _wuParseFloatOrZero(row.dataset.promptTierThreshold || "0");
                var shortPrice = _wuParseFloatOrZero(shortEl.value);
                var longPrice = _wuParseFloatOrZero(longEl.value);
                if (threshold > 0.0) {{
                    return (inputTokens > threshold) ? longPrice : shortPrice;
                }}
                // Tiered elements exist but no threshold - use short price as default
                return shortPrice;
            }}

            var singleEl = row.querySelector("input.cost-price-input[data-kind='input']");
            return _wuParseFloatOrZero(singleEl ? singleEl.value : "0");
        }}

        function _wuGetOutputPricePerMtok(row, inputTokens) {{
            var shortEl = row.querySelector("input.cost-price-input[data-kind='output_short']");
            var longEl = row.querySelector("input.cost-price-input[data-kind='output_long']");
            if (shortEl && longEl) {{
                var threshold = _wuParseFloatOrZero(row.dataset.promptTierThreshold || "0");
                var shortPrice = _wuParseFloatOrZero(shortEl.value);
                var longPrice = _wuParseFloatOrZero(longEl.value);
                if (threshold > 0.0) {{
                    return (inputTokens > threshold) ? longPrice : shortPrice;
                }}
                // Tiered elements exist but no threshold - use short price as default
                return shortPrice;
            }}

            var singleEl = row.querySelector("input.cost-price-input[data-kind='output']");
            return _wuParseFloatOrZero(singleEl ? singleEl.value : "0");
        }}

        function _wuRecomputeCostTable() {{
            var total = 0.0;
            var pricedRows = 0;
            var incompleteRows = 0;
            var rows = document.querySelectorAll("tr.cost-row");
            rows.forEach(function(row) {{
                var cell = row.querySelector(".cost-estimate-usd");
                if (!_wuHasCompletePricing(row)) {{
                    incompleteRows += 1;
                    cell.dataset.value = "";
                    cell.textContent = "Pricing required";
                    return;
                }}
                pricedRows += 1;

                var inputTokens = _wuParseFloatOrZero(
                    row.querySelector(".cost-input-tokens").dataset.value
                );
                var outputTokens = _wuParseFloatOrZero(
                    row.querySelector(".cost-output-tokens").dataset.value
                );

                var inputCost = 0.0;
                var inShortEl = row.querySelector("input.cost-price-input[data-kind='input_short']");
                var inLongEl = row.querySelector("input.cost-price-input[data-kind='input_long']");
                if (inShortEl && inLongEl && row.dataset.inputTokensShort && row.dataset.inputTokensLong) {{
                    var shortTokens = _wuParseFloatOrZero(row.dataset.inputTokensShort);
                    var longTokens = _wuParseFloatOrZero(row.dataset.inputTokensLong);
                    var shortPrice = _wuParseFloatOrZero(inShortEl.value);
                    var longPrice = _wuParseFloatOrZero(inLongEl.value);
                    inputCost = (shortTokens / 1000000.0) * shortPrice
                             + (longTokens / 1000000.0) * longPrice;
                }} else {{
                    var inputPrice = _wuGetInputPricePerMtok(row, inputTokens);
                    inputCost = (inputTokens / 1000000.0) * inputPrice;
                }}

                var outputCost = 0.0;
                var outShortEl = row.querySelector("input.cost-price-input[data-kind='output_short']");
                var outLongEl = row.querySelector("input.cost-price-input[data-kind='output_long']");
                if (outShortEl && outLongEl && row.dataset.outputTokensShort && row.dataset.outputTokensLong) {{
                    var shortTokens = _wuParseFloatOrZero(row.dataset.outputTokensShort);
                    var longTokens = _wuParseFloatOrZero(row.dataset.outputTokensLong);
                    var shortPrice = _wuParseFloatOrZero(outShortEl.value);
                    var longPrice = _wuParseFloatOrZero(outLongEl.value);
                    outputCost = (shortTokens / 1000000.0) * shortPrice
                              + (longTokens / 1000000.0) * longPrice;
                }} else {{
                    var outputPrice = _wuGetOutputPricePerMtok(row, inputTokens);
                    outputCost = (outputTokens / 1000000.0) * outputPrice;
                }}

                var cost = inputCost + outputCost;

                total += cost;
                cell.dataset.value = cost.toString();
                cell.textContent = _wuFormatUsd(cost);
            }});

            var totalEl = document.getElementById("costTotalUsd");
            if (totalEl) {{
                if (incompleteRows && !pricedRows) {{
                    totalEl.textContent = "Pricing required";
                }} else {{
                    totalEl.textContent = _wuFormatUsd(total)
                        + (incompleteRows ? " (partial)" : "");
                }}
            }}
        }}

        // Hook up listeners
        document.querySelectorAll("input.cost-price-input").forEach(function(inp) {{
            inp.addEventListener("input", _wuRecomputeCostTable);
        }});

        // Initial compute
        _wuRecomputeCostTable();
    </script>
    """


def validate_image_options(
    image_format: str | None = None,
    image_quality: int | None = None,
    image_max_size: int | None = None,
) -> tuple[str, int, int | None]:
    """Validate and normalize image processing options.

    Args:
        image_format: Image format ('png' or 'jpeg', default: 'png')
        image_quality: JPEG quality (1-100, default: 85)
        image_max_size: Max image size in pixels (default: None)

    Returns:
        Tuple of (validated_format, validated_quality, validated_max_size)
    """
    # Validate format
    if image_format not in ("png", "jpeg", None):
        logger.warning(f"Invalid image format '{image_format}', using 'png'")
        image_format = "png"
    image_format = image_format or "png"

    # Validate quality
    if image_quality is None:
        image_quality = 85
    if not 1 <= image_quality <= 100:
        logger.warning(f"Image quality {image_quality} out of range [1-100], using 85")
        image_quality = 85

    # Validate max size
    if image_max_size is not None and image_max_size < 1:
        logger.warning(f"Image max size {image_max_size} invalid, ignoring")
        image_max_size = None

    return image_format, image_quality, image_max_size
