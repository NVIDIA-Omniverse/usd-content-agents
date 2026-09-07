#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build optional per-instance crops for revision consistency diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from mesh_geometry import schema_matches
from PIL import Image, ImageDraw

REGIONS_SCHEMA_VERSION = "mesh-segmentation-consistency-regions.v1"
MANIFEST_SCHEMA_VERSION = "mesh-segmentation-consistency-sheet.v1"
REQUIRED_SURFACE_ROLES = {
    "primary_surface",
    "opposing_or_occlusion_revealing",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--neutral-manifest", type=Path, required=True)
    parser.add_argument("--flat-label-manifest", type=Path, required=True)
    parser.add_argument("--selected-only-manifest", type=Path, required=True)
    parser.add_argument("--regions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--padding-fraction", type=float, default=0.12)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--context-views-per-page", type=int, default=2)
    parser.add_argument("--context-image-size", type=int, default=512)
    return parser.parse_args()


def _load_object(path: Path, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_name(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-")
    if not normalized:
        raise ValueError(f"{label} has no safe filename characters")
    return normalized


def _manifest_images(path: Path, label: str) -> dict[str, Path]:
    manifest = _load_object(path, label)
    records = manifest.get("renders")
    if not isinstance(records, list) or not records:
        raise ValueError(f"{label} must contain a nonempty renders array")
    images: dict[str, Path] = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"{label} render {index} must be an object")
        name = record.get("name")
        image = record.get("image")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{label} render {index} lacks name")
        if name in images:
            raise ValueError(f"{label} repeats render name {name!r}")
        if not isinstance(image, str) or not image.strip():
            raise ValueError(f"{label} render {name!r} lacks image")
        image_path = Path(image)
        if not image_path.is_absolute():
            image_path = path.parent / image_path
        image_path = image_path.resolve()
        if not image_path.is_file():
            raise ValueError(f"{label} image is missing: {image_path}")
        images[name] = image_path
    return images


def _bbox(value: object, *, width: int, height: int, label: str) -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(not isinstance(item, int) or isinstance(item, bool) for item in value)
    ):
        raise ValueError(f"{label} bbox must be [x0, y0, x1, y1] integers")
    x0, y0, x1, y1 = value
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(f"{label} bbox is outside the source image")
    return x0, y0, x1, y1


def _square_crop_box(
    bbox: tuple[int, ...],
    *,
    padding_fraction: float,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    side = max(x1 - x0, y1 - y0)
    side = max(1, int(round(side * (1.0 + 2.0 * padding_fraction))))
    center_x = (x0 + x1) / 2.0
    center_y = (y0 + y1) / 2.0
    left = int(round(center_x - side / 2.0))
    top = int(round(center_y - side / 2.0))
    return left, top, left + side, top + side


def _crop(image: Image.Image, box: tuple[int, ...], size: int) -> Image.Image:
    return image.crop(box).resize((size, size), Image.Resampling.LANCZOS)


def main() -> None:
    args = parse_args()
    if not 192 <= args.crop_size <= 1024:
        raise ValueError("--crop-size must be between 192 and 1024")
    if not 0.0 <= args.padding_fraction <= 0.5:
        raise ValueError("--padding-fraction must be between 0 and 0.5")
    if not 1 <= args.columns <= 8:
        raise ValueError("--columns must be between 1 and 8")
    if not 1 <= args.context_views_per_page <= 4:
        raise ValueError("--context-views-per-page must be between 1 and 4")
    if not 256 <= args.context_image_size <= 1024:
        raise ValueError("--context-image-size must be between 256 and 1024")

    neutral_manifest = args.neutral_manifest.resolve()
    flat_manifest = args.flat_label_manifest.resolve()
    selected_manifest = args.selected_only_manifest.resolve()
    regions_path = args.regions.resolve()
    output_dir = args.output_dir.resolve()
    manifest_path = output_dir / "manifest.json"
    sheet_path = output_dir / "comparison_sheet.png"
    if manifest_path.exists() or sheet_path.exists():
        raise FileExistsError("Refusing to overwrite consistency-sheet outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    neutral_images = _manifest_images(neutral_manifest, "neutral manifest")
    flat_images = _manifest_images(flat_manifest, "flat-label manifest")
    selected_images = _manifest_images(
        selected_manifest,
        "selected-only manifest",
    )
    regions = _load_object(regions_path, "consistency regions")
    if not schema_matches(regions.get("schema_version"), REGIONS_SCHEMA_VERSION):
        raise ValueError("Unsupported consistency regions schema")
    raw_instances = regions.get("instances")
    if not isinstance(raw_instances, list) or not raw_instances:
        raise ValueError("consistency regions must contain instances")

    instance_crop_records: list[tuple[str, list[tuple[str, str, Path, Path]]]] = []
    manifest_instances: list[dict[str, Any]] = []
    seen_instances: set[str] = set()
    for instance_index, raw_instance in enumerate(raw_instances):
        if not isinstance(raw_instance, dict):
            raise ValueError(f"instance {instance_index} must be an object")
        instance_id = raw_instance.get("instance_id")
        if not isinstance(instance_id, str) or not instance_id.strip():
            raise ValueError(f"instance {instance_index} lacks instance_id")
        instance_id = instance_id.strip()
        if instance_id in seen_instances:
            raise ValueError(f"duplicate instance_id: {instance_id}")
        seen_instances.add(instance_id)
        instance_slug = _safe_name(instance_id, "instance_id")
        raw_views = raw_instance.get("views")
        if not isinstance(raw_views, list) or len(raw_views) < 2:
            raise ValueError(f"instance {instance_id!r} needs at least two crop views")
        evidence_crops: list[dict[str, Any]] = []
        instance_records: list[tuple[str, str, Path, Path]] = []
        seen_views: set[str] = set()
        surface_roles: set[str] = set()
        for view_index, raw_view in enumerate(raw_views):
            if not isinstance(raw_view, dict):
                raise ValueError(
                    f"instance {instance_id!r} view {view_index} must be an object"
                )
            view_id = raw_view.get("view_id")
            if not isinstance(view_id, str) or not view_id.strip():
                raise ValueError(
                    f"instance {instance_id!r} view {view_index} lacks view_id"
                )
            view_id = view_id.strip()
            if view_id in seen_views:
                raise ValueError(f"instance {instance_id!r} repeats view {view_id!r}")
            seen_views.add(view_id)
            if (
                view_id not in neutral_images
                or view_id not in flat_images
                or view_id not in selected_images
            ):
                raise ValueError(f"view {view_id!r} is missing from one input manifest")
            surface_role = raw_view.get("surface_role")
            if surface_role not in REQUIRED_SURFACE_ROLES:
                raise ValueError(
                    f"instance {instance_id!r} view {view_id!r} needs "
                    "surface_role primary_surface or "
                    "opposing_or_occlusion_revealing"
                )
            surface_roles.add(surface_role)
            with Image.open(flat_images[view_id]) as flat_source:
                flat = flat_source.convert("RGB")
            with Image.open(selected_images[view_id]) as selected_source:
                selected = selected_source.convert("RGB")
            if flat.size != selected.size:
                raise ValueError(
                    f"view {view_id!r} input image dimensions do not match"
                )
            source_bbox = _bbox(
                raw_view.get("bbox"),
                width=flat.width,
                height=flat.height,
                label=f"instance {instance_id!r} view {view_id!r}",
            )
            crop_box = _square_crop_box(
                source_bbox,
                padding_fraction=args.padding_fraction,
            )
            flat_crop = _crop(flat, crop_box, args.crop_size)
            selected_crop = _crop(selected, crop_box, args.crop_size)
            view_slug = _safe_name(view_id, "view_id")
            flat_path = output_dir / f"{instance_slug}__{view_slug}__flat-label.png"
            selected_path = (
                output_dir / f"{instance_slug}__{view_slug}__selected-only.png"
            )
            flat_crop.save(flat_path)
            selected_crop.save(selected_path)
            instance_records.append((view_id, surface_role, flat_path, selected_path))
            evidence_crops.append(
                {
                    "view_id": view_id,
                    "surface_role": surface_role,
                    "source_bbox": list(source_bbox),
                    "square_crop_box": list(crop_box),
                    "flat_label_crop": str(flat_path),
                    "flat_label_crop_sha256": _sha256(flat_path),
                    "selected_only_crop": str(selected_path),
                    "selected_only_crop_sha256": _sha256(selected_path),
                    "crop_size": [args.crop_size, args.crop_size],
                }
            )
        if surface_roles != REQUIRED_SURFACE_ROLES:
            raise ValueError(
                f"instance {instance_id!r} must include both primary and "
                "opposing-or-occlusion-revealing crops"
            )
        manifest_instances.append(
            {
                "instance_id": instance_id,
                "evidence_crops": evidence_crops,
            }
        )
        instance_crop_records.append((instance_id, instance_records))

    # Keep peer instances large and adjacent. A single tall strip is easy for a
    # vision model to downscale until lower-row defects disappear. Each tile
    # contains the primary and opposing views for one instance, with flat-label
    # and selected-only channels side by side.
    max_views = max(len(records) for _, records in instance_crop_records)
    tile_header = 28
    view_label_height = 20
    tile_width = 2 * args.crop_size
    tile_height = tile_header + max_views * (args.crop_size + view_label_height)
    columns = min(args.columns, len(instance_crop_records))
    rows = math.ceil(len(instance_crop_records) / columns)
    sheet = Image.new(
        "RGB",
        (tile_width * columns, tile_height * rows),
        "black",
    )
    draw = ImageDraw.Draw(sheet)
    for index, (instance_id, records) in enumerate(instance_crop_records):
        column = index % columns
        row = index // columns
        tile_x = column * tile_width
        tile_y = row * tile_height
        draw.rectangle(
            (
                tile_x,
                tile_y,
                tile_x + tile_width - 1,
                tile_y + tile_height - 1,
            ),
            outline=(96, 96, 96),
        )
        draw.text((tile_x + 6, tile_y + 7), instance_id, fill="white")
        ordered_records = sorted(
            records,
            key=lambda item: (
                item[1] != "primary_surface",
                item[0],
            ),
        )
        for view_row, (view_id, surface_role, flat_path, selected_path) in enumerate(
            ordered_records
        ):
            crop_y = (
                tile_y + tile_header + view_row * (args.crop_size + view_label_height)
            )
            with Image.open(flat_path) as flat_crop:
                sheet.paste(flat_crop.convert("RGB"), (tile_x, crop_y))
            with Image.open(selected_path) as selected_crop:
                sheet.paste(
                    selected_crop.convert("RGB"),
                    (tile_x + args.crop_size, crop_y),
                )
            role_label = (
                "primary" if surface_role == "primary_surface" else "opposing/occlusion"
            )
            label_y = crop_y + args.crop_size + 3
            draw.text(
                (tile_x + 4, label_y),
                f"{role_label}: {view_id} | flat",
                fill="white",
            )
            draw.text(
                (tile_x + args.crop_size + 4, label_y),
                "selected only",
                fill="white",
            )
    sheet.save(sheet_path)

    # Preserve full-scene context in the exact three-channel layout used by a
    # human reviewer: neutral, flat label, and selected only. Split the views
    # across compact pages so a model does not downscale a long contact sheet
    # until a one-fragment omission disappears.
    context_view_ids: list[str] = []
    for _, records in instance_crop_records:
        for view_id, _, _, _ in records:
            if view_id not in context_view_ids:
                context_view_ids.append(view_id)
    context_pages: list[dict[str, Any]] = []
    context_channels = ["neutral", "flat_label", "selected_only"]
    context_header_height = 24
    for page_index, start in enumerate(
        range(0, len(context_view_ids), args.context_views_per_page)
    ):
        page_view_ids = context_view_ids[start : start + args.context_views_per_page]
        page_width = args.context_image_size * len(context_channels)
        page_height = len(page_view_ids) * (
            args.context_image_size + context_header_height
        )
        page = Image.new("RGB", (page_width, page_height), "black")
        page_draw = ImageDraw.Draw(page)
        for row, view_id in enumerate(page_view_ids):
            y = row * (args.context_image_size + context_header_height)
            sources = (
                ("neutral", neutral_images[view_id]),
                ("flat label", flat_images[view_id]),
                ("selected only", selected_images[view_id]),
            )
            for column, (channel_label, source_path) in enumerate(sources):
                with Image.open(source_path) as source:
                    image = source.convert("RGB").resize(
                        (args.context_image_size, args.context_image_size),
                        Image.Resampling.LANCZOS,
                    )
                x = column * args.context_image_size
                page.paste(image, (x, y))
                page_draw.text(
                    (x + 5, y + args.context_image_size + 5),
                    f"{view_id} | {channel_label}",
                    fill="white",
                )
        page_path = output_dir / f"context_sheet_{page_index:02d}.png"
        page.save(page_path)
        context_pages.append(
            {
                "page_id": f"context-{page_index:02d}",
                "view_ids": page_view_ids,
                "image": str(page_path),
                "image_sha256": _sha256(page_path),
                "size": list(page.size),
            }
        )

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "passed",
        "neutral_render_manifest": str(neutral_manifest),
        "neutral_render_manifest_sha256": _sha256(neutral_manifest),
        "flat_label_render_manifest": str(flat_manifest),
        "flat_label_render_manifest_sha256": _sha256(flat_manifest),
        "selected_only_render_manifest": str(selected_manifest),
        "selected_only_render_manifest_sha256": _sha256(selected_manifest),
        "regions": str(regions_path),
        "regions_sha256": _sha256(regions_path),
        "crop_size": [args.crop_size, args.crop_size],
        "padding_fraction": args.padding_fraction,
        "layout": {
            "mode": "tiled_peer_grid",
            "columns": columns,
            "rows": rows,
            "tile_size": [tile_width, tile_height],
            "sheet_size": list(sheet.size),
            "instance_order": [instance_id for instance_id, _ in instance_crop_records],
            "channels_per_view": ["flat_label", "selected_only"],
            "surface_role_order": [
                "primary_surface",
                "opposing_or_occlusion_revealing",
            ],
        },
        "comparison_sheet": str(sheet_path),
        "comparison_sheet_sha256": _sha256(sheet_path),
        "context_layout": {
            "mode": "full_view_triptych_pages",
            "views_per_page": args.context_views_per_page,
            "image_size": [args.context_image_size, args.context_image_size],
            "channels": context_channels,
            "view_order": context_view_ids,
        },
        "context_pages": context_pages,
        "instances": manifest_instances,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
