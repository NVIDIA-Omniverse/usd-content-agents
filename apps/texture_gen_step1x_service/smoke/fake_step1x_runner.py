# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tiny Step1X-compatible runner for Docker wiring smoke tests.

This script is intentionally not a model replacement. It exercises the same
external-runner contract as the real Step1X runtime by writing the map files
that ``apps.texture_gen_step1x_service.backend.ExternalStep1XRunner`` expects
to find in the job output directory.

The prompt label is diagnostic sugar only; Pillow may omit it if no default
font is available in a stripped-down image.
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)


def _write_albedo(path: Path, prompt: str, size: int) -> None:
    image = Image.new("RGB", (size, size), (196, 150, 52))
    draw = ImageDraw.Draw(image)
    tile = max(16, size // 16)
    for y in range(0, size, tile):
        for x in range(0, size, tile):
            color = (
                (215, 176, 74) if (x // tile + y // tile) % 2 == 0 else (138, 99, 36)
            )
            draw.rectangle((x, y, x + tile - 1, y + tile - 1), fill=color)
    label = (prompt or "fake step1x")[:80]
    draw.rectangle((8, 8, min(size - 8, 520), 44), fill=(30, 30, 30))
    draw.text((16, 18), label, fill=(255, 255, 255))
    image.save(path)


def _write_normal(path: Path, size: int) -> None:
    Image.new("RGB", (size, size), (128, 128, 255)).save(path)


def _write_orm(path: Path, size: int) -> None:
    Image.new("RGB", (size, size), (255, 90, 10)).save(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-asset", type=Path, required=True)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--texture-size", type=int, default=512)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    size = max(64, min(args.texture_size, 2048))
    _write_albedo(args.output_dir / "final_albedo.png", args.prompt, size)
    _write_normal(args.output_dir / "final_normal.png", size)
    _write_orm(args.output_dir / "final_orm.png", size)

    if args.source_asset.exists():
        suffix = args.source_asset.suffix or ".usd"
        shutil.copy2(args.source_asset, args.output_dir / f"edited_fake{suffix}")

    logger.info("fake-step1x wrote maps to %s", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
