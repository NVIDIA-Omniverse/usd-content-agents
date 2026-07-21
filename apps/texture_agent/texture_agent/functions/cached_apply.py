# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared context contract for applying previously generated texture caches."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

CACHED_APPLY_ONLY_KEY = "cached_apply_only"


def is_cached_apply_context(context: dict[str, Any]) -> bool:
    """Return whether the current workflow may only consume cached artifacts."""
    planning_config = context.get("planning_config") or {}
    return bool(
        context.get(CACHED_APPLY_ONLY_KEY)
        or planning_config.get("resume_apply_textures")
    )


def is_valid_cached_texture_png(path: str | Path) -> bool:
    """Return whether a cached texture is a safe, decodable non-empty PNG."""
    try:
        with Path(path).open("rb") as stream:
            with Image.open(stream) as image:
                max_pixels = Image.MAX_IMAGE_PIXELS
                if not (
                    image.format == "PNG"
                    and image.width > 0
                    and image.height > 0
                    and (max_pixels is None or image.width * image.height <= max_pixels)
                ):
                    return False
                image.verify()

            # verify() checks PNG chunks and CRCs without decoding IDAT pixels.
            # Reopen the same file descriptor and force a full decode so a
            # structurally valid PNG with corrupt compressed data cannot enter
            # a downloaded USDZ or reach a GPU texture loader.
            stream.seek(0)
            with Image.open(stream) as image:
                image.load()
        return True
    except (
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        OSError,
        SyntaxError,
        ValueError,
    ):
        return False
