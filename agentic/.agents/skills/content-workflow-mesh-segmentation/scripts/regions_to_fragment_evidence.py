#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Map batched image regions through closest-visible fragment ID buffers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--id-buffer-manifest", type=Path, required=True)
    parser.add_argument("--regions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-pixels", type=int, default=1)
    return parser.parse_args()


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _region_mask(raw: dict[str, Any], width: int, height: int) -> np.ndarray:
    image = Image.new("1", (width, height), 0)
    draw = ImageDraw.Draw(image)
    shape = str(raw.get("shape", "polygon")).lower()
    if shape == "polygon":
        points = [tuple(map(float, point)) for point in raw.get("points", [])]
        if len(points) < 3:
            raise ValueError("Polygon regions require at least three points")
        draw.polygon(points, fill=1)
    elif shape in {"box", "rectangle"}:
        box = raw.get("box")
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError("Box regions require box=[x0,y0,x1,y1]")
        draw.rectangle(tuple(map(float, box)), fill=1)
    elif shape == "scribble":
        points = [tuple(map(float, point)) for point in raw.get("points", [])]
        if not points:
            raise ValueError("Scribble regions require points")
        radius = max(0.5, float(raw.get("radius", 3.0)))
        if len(points) > 1:
            draw.line(points, fill=1, width=max(1, int(round(radius * 2.0))))
        for x, y in points:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=1)
    elif shape == "pixels":
        points = [tuple(map(int, point)) for point in raw.get("points", [])]
        if not points:
            raise ValueError("Pixel regions require points")
        for x, y in points:
            if 0 <= x < width and 0 <= y < height:
                draw.point((x, y), fill=1)
    else:
        raise ValueError(f"Unsupported region shape: {shape}")
    return np.asarray(image, dtype=bool)


def main() -> None:
    args = parse_args()
    if args.minimum_pixels <= 0:
        raise ValueError("--minimum-pixels must be positive")
    manifest_path = args.id_buffer_manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    view_by_name = {str(view["name"]): view for view in manifest.get("views", [])}
    if not view_by_name:
        raise ValueError("ID-buffer manifest has no views")
    regions_path = args.regions.resolve()
    payload = json.loads(regions_path.read_text(encoding="utf-8"))
    raw_regions = payload.get("regions")
    if not isinstance(raw_regions, list) or not raw_regions:
        raise ValueError("Region file must contain a nonempty regions list")

    records: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_regions):
        if not isinstance(raw, dict):
            raise ValueError(f"Region {index} is not an object")
        view_id = str(raw.get("view_id", ""))
        if view_id not in view_by_name:
            raise ValueError(f"Region {index} references unknown view: {view_id}")
        polarity = str(raw.get("polarity", "")).lower()
        if polarity not in {"positive", "negative"}:
            raise ValueError(f"Region {index} has invalid polarity: {polarity}")
        view = view_by_name[view_id]
        raw_path = _resolve(
            manifest_path.parent,
            str(view["channels"]["fragment_ids"]["raw"]),
        )
        fragment_ids = np.load(raw_path, allow_pickle=False)
        if fragment_ids.ndim != 2:
            raise ValueError(f"Fragment ID buffer is not 2D: {raw_path}")
        height, width = fragment_ids.shape
        region = _region_mask(raw, width, height)
        valid = region & (fragment_ids >= 0)
        ids, counts = np.unique(fragment_ids[valid], return_counts=True)
        keep = counts >= args.minimum_pixels
        ids = ids[keep].astype(np.int64)
        counts = counts[keep].astype(np.int64)
        total_valid = int(np.count_nonzero(valid))
        candidates = []
        for fragment_id, count in zip(ids, counts, strict=True):
            visible_count = int(np.count_nonzero(fragment_ids == fragment_id))
            candidates.append(
                {
                    "fragment_id": int(fragment_id),
                    "marked_pixel_count": int(count),
                    "marked_pixel_fraction": (
                        float(count / total_valid) if total_valid else 0.0
                    ),
                    "visible_fragment_coverage": (
                        float(count / visible_count) if visible_count else 0.0
                    ),
                }
            )
        records.append(
            {
                "region_index": index,
                "view_id": view_id,
                "polarity": polarity,
                "shape": raw.get("shape", "polygon"),
                "note": raw.get("note"),
                "marked_pixel_count": int(np.count_nonzero(region)),
                "valid_hit_pixel_count": total_valid,
                "candidate_fragment_ids": ids.astype(int).tolist(),
                "candidates": candidates,
            }
        )

    result = {
        "schema_version": "mesh-segmentation-region-fragment-evidence.v1",
        "id_buffer_manifest": str(manifest_path),
        "regions": str(regions_path),
        "closest_visible_hit_only": True,
        "semantic_decision_unit": "immutable_fragment",
        "events": records,
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"output": str(args.output.resolve()), "regions": len(records)}, indent=2
        )
    )


if __name__ == "__main__":
    main()
