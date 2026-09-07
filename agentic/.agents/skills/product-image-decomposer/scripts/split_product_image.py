#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Split a product/exploded image into CAD-ready component crops.

The script accepts an optional manual part plan. Without a plan it performs a
simple connected-component pass against a near-white background, suitable for
clean exploded sheets.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


@dataclass
class Part:
    id: str
    local_id: str
    label: str
    bbox: tuple[int, int, int, int]
    level: int = 1
    parent_id: str | None = None
    role: str = "component"
    has_children: bool = False
    material_hint: str = ""
    cad_notes: str = ""
    function_hint: str = ""
    proportion_notes: str = ""
    interface_notes: str = ""
    expected_features: list[str] | None = None


def _slug(text: str, fallback: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or fallback


def _plan_levels(path: Path) -> int | None:
    data = json.loads(path.read_text())
    raw = data.get("split_levels")
    return int(raw) if raw is not None else None


def _resolve_bbox(
    raw: list[Any],
    image_size: tuple[int, int],
    *,
    bbox_space: str,
    parent_bbox: tuple[int, int, int, int] | None,
) -> tuple[int, int, int, int]:
    if len(raw) != 4:
        raise ValueError("bbox must have 4 values")
    x0, y0, x1, y1 = [float(v) for v in raw]
    space = bbox_space.lower().replace("-", "_")
    pixel_spaces = {
        "absolute",
        "pixel",
        "pixels",
        "source_pixel",
        "source_pixels",
    }
    normalized_spaces = {
        "normalized",
        "relative",
        "source_normalized",
        "source_relative",
    }
    parent_spaces = {
        "parent",
        "parent_normalized",
        "parent_relative",
    }
    if space not in {"source"} | pixel_spaces | normalized_spaces | parent_spaces:
        raise ValueError(f"unsupported bbox_space: {bbox_space!r}")
    base = parent_bbox if space in parent_spaces and parent_bbox is not None else None
    forced_pixels = space in pixel_spaces
    forced_normalized = space in normalized_spaces or space in parent_spaces
    use_normalized = forced_normalized or (
        not forced_pixels and max(abs(x0), abs(y0), abs(x1), abs(y1)) <= 1.5
    )
    if use_normalized:
        if base:
            bx0, by0, bx1, by1 = base
            bw = bx1 - bx0
            bh = by1 - by0
            x0, x1 = bx0 + x0 * bw, bx0 + x1 * bw
            y0, y1 = by0 + y0 * bh, by0 + y1 * bh
        else:
            w, h = image_size
            x0, x1 = x0 * w, x1 * w
            y0, y1 = y0 * h, y1 * h
    elif base:
        bx0, by0, _, _ = base
        x0, x1 = bx0 + x0, bx0 + x1
        y0, y1 = by0 + y0, by0 + y1
    return _clamp_bbox((round(x0), round(y0), round(x1), round(y1)), image_size)


def _load_plan(path: Path, size: tuple[int, int], levels: int) -> list[Part]:
    data = json.loads(path.read_text())
    parts: list[Part] = []

    def visit(
        items: list[dict[str, Any]],
        *,
        level: int,
        parent_id: str | None,
        parent_bbox: tuple[int, int, int, int] | None,
    ) -> None:
        for i, item in enumerate(items, start=1):
            label = str(item.get("label") or item.get("id") or f"part_{i:02d}")
            local_id = _slug(str(item.get("id") or label), f"part_{i:02d}")
            pid = local_id if parent_id is None else f"{parent_id}__{local_id}"
            bbox = _resolve_bbox(
                item["bbox"],
                size,
                bbox_space=str(item.get("bbox_space", "source")),
                parent_bbox=parent_bbox,
            )
            children = item.get("children") or []
            role = str(item.get("role") or ("object" if children else "component"))
            parts.append(
                Part(
                    id=pid,
                    local_id=local_id,
                    label=label,
                    bbox=bbox,
                    level=level,
                    parent_id=parent_id,
                    role=role,
                    has_children=bool(children),
                    material_hint=str(item.get("material_hint", "")),
                    cad_notes=str(item.get("cad_notes", "")),
                    function_hint=str(item.get("function_hint", "")),
                    proportion_notes=str(item.get("proportion_notes", "")),
                    interface_notes=str(item.get("interface_notes", "")),
                    expected_features=[
                        str(v) for v in item.get("expected_features", [])
                    ],
                )
            )
            if children and level < levels:
                visit(children, level=level + 1, parent_id=pid, parent_bbox=bbox)

    visit(data.get("parts", []), level=1, parent_id=None, parent_bbox=None)
    return parts


def _clamp_bbox(
    bbox: tuple[int, int, int, int], size: tuple[int, int]
) -> tuple[int, int, int, int]:
    w, h = size
    x0, y0, x1, y1 = bbox
    x0 = max(0, min(w - 1, x0))
    y0 = max(0, min(h - 1, y0))
    x1 = max(x0 + 1, min(w, x1))
    y1 = max(y0 + 1, min(h, y1))
    return x0, y0, x1, y1


def _expand_bbox(
    bbox: tuple[int, int, int, int], pad: int, size: tuple[int, int]
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    return _clamp_bbox((x0 - pad, y0 - pad, x1 + pad, y1 + pad), size)


def _foreground_mask(img: Image.Image, threshold: int) -> np.ndarray:
    rgb = np.asarray(img.convert("RGB")).astype(np.int16)
    # Distance from white catches grey/chrome edges without requiring cv2.
    dist = np.max(255 - rgb, axis=2)
    mask = dist > threshold
    # Ignore tiny one-pixel antialiasing specks by requiring a 3x3 neighbor.
    padded = np.pad(mask, 1)
    votes = sum(
        padded[1 + dy : 1 + dy + mask.shape[0], 1 + dx : 1 + dx + mask.shape[1]]
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
    )
    return mask & (votes >= 2)


def _components(
    mask: np.ndarray, min_area: int
) -> list[tuple[int, int, int, int, int]]:
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    boxes: list[tuple[int, int, int, int, int]] = []
    for sy in range(h):
        xs = np.flatnonzero(mask[sy] & ~seen[sy])
        for sx in xs:
            if seen[sy, sx] or not mask[sy, sx]:
                continue
            q: deque[tuple[int, int]] = deque([(int(sx), int(sy))])
            seen[sy, sx] = True
            minx = maxx = int(sx)
            miny = maxy = int(sy)
            area = 0
            while q:
                x, y = q.popleft()
                area += 1
                minx = min(minx, x)
                maxx = max(maxx, x)
                miny = min(miny, y)
                maxy = max(maxy, y)
                for nx in (x - 1, x, x + 1):
                    for ny in (y - 1, y, y + 1):
                        if nx < 0 or ny < 0 or nx >= w or ny >= h:
                            continue
                        if seen[ny, nx] or not mask[ny, nx]:
                            continue
                        seen[ny, nx] = True
                        q.append((nx, ny))
            if area >= min_area:
                boxes.append((minx, miny, maxx + 1, maxy + 1, area))
    boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes


def _auto_parts(
    img: Image.Image, threshold: int, min_area: int, pad: int
) -> list[Part]:
    mask = _foreground_mask(img, threshold)
    parts: list[Part] = []
    for i, (x0, y0, x1, y1, area) in enumerate(_components(mask, min_area), start=1):
        bbox = _expand_bbox((x0, y0, x1, y1), pad, img.size)
        parts.append(
            Part(
                id=f"part_{i:02d}",
                local_id=f"part_{i:02d}",
                label=f"Auto part {i:02d}",
                bbox=bbox,
                cad_notes=f"Auto-detected connected component, foreground area {area} px.",
                expected_features=[],
            )
        )
    return parts


def _save_overlay(img: Image.Image, parts: list[Part], out: Path) -> None:
    overlay = img.convert("RGB")
    draw = ImageDraw.Draw(overlay)
    for i, part in enumerate(parts, start=1):
        x0, y0, x1, y1 = part.bbox
        color = (255, 64 + (i * 37) % 160, 32 + (i * 83) % 200)
        draw.rectangle((x0, y0, x1, y1), outline=color, width=4)
        draw.rectangle((x0, max(0, y0 - 22), min(x1, x0 + 260), y0), fill=color)
        draw.text(
            (x0 + 4, max(0, y0 - 20)), f"L{part.level} {i}: {part.id}", fill=(0, 0, 0)
        )
    overlay.save(out)


def _contact_sheet(crops: list[tuple[Part, Path]], out: Path, thumb: int = 220) -> None:
    if not crops:
        return
    cols = min(4, len(crops))
    rows = math.ceil(len(crops) / cols)
    sheet = Image.new("RGB", (cols * thumb, rows * (thumb + 34)), "white")
    draw = ImageDraw.Draw(sheet)
    for idx, (part, path) in enumerate(crops):
        x = (idx % cols) * thumb
        y = (idx // cols) * (thumb + 34)
        crop = Image.open(path).convert("RGB")
        crop.thumbnail((thumb - 16, thumb - 48), Image.Resampling.LANCZOS)
        px = x + (thumb - crop.width) // 2
        py = y + 8
        sheet.paste(crop, (px, py))
        draw.text((x + 8, y + thumb - 32), part.id[:32], fill=(0, 0, 0))
    sheet.save(out)


def _six_view_prompt(part: Part, crop_path: Path) -> str:
    feature_text = ", ".join(part.expected_features or [])
    detail = "; ".join(
        v
        for v in [
            part.material_hint,
            part.cad_notes,
            part.function_hint,
            part.proportion_notes,
            part.interface_notes,
            f"Expected visible features: {feature_text}" if feature_text else "",
        ]
        if v
    )
    role_phrase = "object overview" if part.has_children else "CAD component"
    return (
        "Using the component crop as the only reference, generate a clean six-view CAD "
        f"reference sheet for the {role_phrase} '{part.label}'. Views required: front, back, left, right, "
        "top, and isometric. Keep the object centered in each view, consistent scale, "
        "white background, no labels inside the views, no watermark, no brand text. "
        "Preserve mechanical proportions, holes, radii, seams, material finish, and "
        f"mating interfaces. Component crop path: {crop_path.name}. "
        f"Notes: {detail or 'infer visible geometry only.'}"
    )


def _is_target_leaf(part: Part, effective_levels: int) -> bool:
    return part.level >= effective_levels or not part.has_children


def split_image(
    image_path: Path,
    out_dir: Path,
    plan_path: Path | None,
    threshold: int,
    min_area: int,
    pad: int,
    levels: int | None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    parts_dir = out_dir / "parts"
    six_dir = out_dir / "six_views"
    parts_dir.mkdir(exist_ok=True)
    six_dir.mkdir(exist_ok=True)

    img = Image.open(image_path).convert("RGBA")
    source_copy = out_dir / "source.png"
    if image_path.resolve() != source_copy.resolve():
        shutil.copy2(image_path, source_copy)

    effective_levels = levels or (_plan_levels(plan_path) if plan_path else None) or 1
    if effective_levels < 1:
        raise ValueError("--levels must be >= 1")
    parts = (
        _load_plan(plan_path, img.size, effective_levels)
        if plan_path
        else _auto_parts(img, threshold, min_area, pad)
    )
    crop_records = []
    hierarchical = any(part.level > 1 or part.parent_id for part in parts)
    for part in parts:
        crop = img.crop(part.bbox)
        crop_dir = parts_dir / f"level_{part.level:02d}" if hierarchical else parts_dir
        crop_dir.mkdir(exist_ok=True)
        crop_path = crop_dir / f"{part.id}.png"
        crop.save(crop_path)
        crop_records.append((part, crop_path))

    _save_overlay(img, parts, out_dir / "segmentation_overlay.png")
    _contact_sheet(crop_records, out_dir / "contact_sheet.png")

    manifest = {
        "source_image": str(source_copy),
        "plan": str(plan_path) if plan_path else None,
        "split_levels": effective_levels,
        "parts": [],
    }
    prompt_lines = []
    for part, crop_path in crop_records:
        six_path = six_dir / f"{part.id}_six_view.png"
        prompt = _six_view_prompt(part, crop_path)
        target_leaf = _is_target_leaf(part, effective_levels)
        manifest["parts"].append(
            {
                "id": part.id,
                "local_id": part.local_id,
                "label": part.label,
                "level": part.level,
                "parent_id": part.parent_id,
                "role": part.role,
                "has_children": part.has_children,
                "target_leaf": target_leaf,
                "bbox": list(part.bbox),
                "crop_path": str(crop_path),
                "six_view_path": str(six_path),
                "material_hint": part.material_hint,
                "cad_notes": part.cad_notes,
                "function_hint": part.function_hint,
                "proportion_notes": part.proportion_notes,
                "interface_notes": part.interface_notes,
                "expected_features": part.expected_features or [],
                "six_view_prompt": prompt,
            }
        )
        prompt_lines.append(
            json.dumps(
                {
                    "id": part.id,
                    "level": part.level,
                    "parent_id": part.parent_id,
                    "role": part.role,
                    "target_leaf": target_leaf,
                    "crop_path": str(crop_path),
                    "six_view_path": str(six_path),
                    "prompt": prompt,
                }
            )
        )

    (out_dir / "parts_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    hierarchy = {
        "split_levels": effective_levels,
        "nodes": [
            {
                "id": p.id,
                "local_id": p.local_id,
                "parent_id": p.parent_id,
                "level": p.level,
                "role": p.role,
                "label": p.label,
                "has_children": p.has_children,
                "target_leaf": _is_target_leaf(p, effective_levels),
                "function_hint": p.function_hint,
                "proportion_notes": p.proportion_notes,
                "interface_notes": p.interface_notes,
                "expected_features": p.expected_features or [],
            }
            for p in parts
        ],
    }
    (out_dir / "hierarchy_manifest.json").write_text(
        json.dumps(hierarchy, indent=2) + "\n"
    )
    (out_dir / "six_view_prompts.jsonl").write_text(
        "\n".join(prompt_lines) + ("\n" if prompt_lines else "")
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument(
        "--threshold",
        type=int,
        default=18,
        help="Foreground distance from white for auto mode.",
    )
    parser.add_argument(
        "--min-area", type=int, default=600, help="Smallest auto component in pixels."
    )
    parser.add_argument("--pad", type=int, default=18, help="Crop padding in pixels.")
    parser.add_argument(
        "--levels",
        type=int,
        help="Hierarchy depth to split. Overrides plan split_levels.",
    )
    args = parser.parse_args()

    manifest = split_image(
        args.image,
        args.out,
        args.plan,
        args.threshold,
        args.min_area,
        args.pad,
        args.levels,
    )
    print(f"wrote {len(manifest['parts'])} part crops to {args.out / 'parts'}")
    print(args.out / "parts_manifest.json")


if __name__ == "__main__":
    main()
