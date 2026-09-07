#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build reusable evidence for disconnected topology components."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from mesh_geometry import (
    load_fragment_labels,
    load_usd,
    sha256_file,
    source_metadata,
    write_json,
)
from PIL import Image, ImageDraw, ImageOps

_ISOLATED_VIEW_DIRECTIONS = (
    ("plus_xminus_yplus_z", np.asarray([1.0, -1.0, 1.0])),
    ("minus_xminus_yplus_z", np.asarray([-1.0, -1.0, 1.0])),
    ("plus_xplus_yplus_z", np.asarray([1.0, 1.0, 1.0])),
    ("minus_xplus_yplus_z", np.asarray([-1.0, 1.0, 1.0])),
    ("plus_xminus_yminus_z", np.asarray([1.0, -1.0, -1.0])),
    ("minus_xminus_yminus_z", np.asarray([-1.0, -1.0, -1.0])),
    ("plus_xplus_yminus_z", np.asarray([1.0, 1.0, -1.0])),
    ("minus_xplus_yminus_z", np.asarray([-1.0, 1.0, -1.0])),
)
_GALLERY_SCHEMA_VERSION = "mesh-segmentation-component-gallery.v1"
# These only rank components for human/agent inspection; they never assign
# semantics or exclude a component from the complete gallery.
_NEARBY_DISTANCE_FRACTION = 0.002
_NEARBY_AREA_RATIO_LIMIT = 20.0
_ISOLATED_RENDER_SIZE = (198, 200)
_LOCATOR_RENDER_SIZE = (104, 110)
_CARD_SIZE = (640, 384)
_CONTACT_SHEET_COLUMNS = 2
_CONTACT_SHEET_ROWS = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-usd", type=Path, required=True)
    parser.add_argument("--target")
    parser.add_argument("--fragment-labels", type=Path, required=True)
    parser.add_argument("--id-buffer-manifest", type=Path, required=True)
    parser.add_argument("--neutral-render-manifest", type=Path, required=True)
    parser.add_argument("--appearance-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _resolve_path(value: object, *, base: Path, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty path string")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _render_images(manifest: dict[str, Any], *, base: Path) -> dict[str, Path]:
    raw = manifest.get("renders")
    if not isinstance(raw, list):
        raise ValueError("Neutral render manifest lacks renders")
    images: dict[str, Path] = {}
    for index, record in enumerate(raw):
        if not isinstance(record, dict):
            raise ValueError(f"Neutral render {index} must be an object")
        name = record.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"Neutral render {index} lacks a name")
        image = _resolve_path(
            record.get("image"),
            base=base,
            label=f"neutral render {name!r} image",
        )
        if not image.is_file():
            raise FileNotFoundError(image)
        images[name] = image
    return images


def _render_records(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw = manifest.get("renders")
    if not isinstance(raw, list):
        raise ValueError("Neutral render manifest lacks renders")
    records: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(raw):
        if not isinstance(record, dict):
            raise ValueError(f"Neutral render {index} must be an object")
        name = record.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"Neutral render {index} lacks a name")
        if name in records:
            raise ValueError(f"Neutral render manifest repeats view {name!r}")
        records[name] = record
    return records


def _appearance_images(directory: Path | None, view_ids: set[str]) -> dict[str, Path]:
    if directory is None:
        return {}
    root = directory.resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    images: dict[str, Path] = {}
    for path in sorted(root.glob("*.png")):
        if path.stem in view_ids:
            selected_view_id = path.stem
        else:
            matches = sorted(
                (view_id for view_id in view_ids if path.stem.endswith(view_id)),
                key=lambda view_id: (-len(view_id), view_id),
            )
            if not matches:
                continue
            if len(matches) > 1 and len(matches[0]) == len(matches[1]):
                raise ValueError(
                    f"Appearance image {path.name} matches multiple view IDs: {matches}"
                )
            selected_view_id = matches[0]
        if selected_view_id in images:
            raise ValueError(
                f"Multiple appearance images map to view {selected_view_id!r}: "
                f"{images[selected_view_id].name}, {path.name}"
            )
        images[selected_view_id] = path.resolve()
    return images


def _component_statistics(data: Any) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    component_count = int(data.component_ids.max()) + 1
    for component_id in range(component_count):
        face_ids = np.flatnonzero(data.component_ids == component_id)
        if not len(face_ids):
            raise ValueError(
                "Topology component IDs must be contiguous; "
                f"component {component_id} has no faces"
            )
        point_ids = np.unique(data.triangles[face_ids].reshape(-1))
        points = data.points[point_ids]
        area = float(data.areas[face_ids].sum())
        if area > 0:
            centroid = np.average(
                data.centroids[face_ids],
                axis=0,
                weights=data.areas[face_ids],
            )
        else:
            centroid = data.centroids[face_ids].mean(axis=0)
        records.append(
            {
                "component_id": component_id,
                "face_count": int(len(face_ids)),
                "area": area,
                "centroid": centroid.astype(float).tolist(),
                "bounds_min": points.min(axis=0).astype(float).tolist(),
                "bounds_max": points.max(axis=0).astype(float).tolist(),
            }
        )
    return records


def _nearby_component_ids(
    statistics: list[dict[str, object]],
) -> list[list[int]]:
    """Find similarly scaled components whose 3D bounds nearly touch."""

    bounds_min = np.asarray([record["bounds_min"] for record in statistics])
    bounds_max = np.asarray([record["bounds_max"] for record in statistics])
    areas = np.asarray([record["area"] for record in statistics], dtype=np.float64)
    asset_diagonal = float(
        np.linalg.norm(bounds_max.max(axis=0) - bounds_min.min(axis=0))
    )
    distance_limit = max(asset_diagonal * _NEARBY_DISTANCE_FRACTION, 1.0e-9)
    neighbors: list[list[int]] = []
    for component_id in range(len(statistics)):
        gaps = np.maximum(
            np.maximum(
                bounds_min - bounds_max[component_id],
                bounds_min[component_id] - bounds_max,
            ),
            0.0,
        )
        distances = np.linalg.norm(gaps, axis=1)
        minimum_area = np.minimum(areas, areas[component_id])
        maximum_area = np.maximum(areas, areas[component_id])
        scale_ratio = maximum_area / np.maximum(minimum_area, 1.0e-12)
        candidates = [
            candidate_id
            for candidate_id in range(len(statistics))
            if candidate_id != component_id
            and distances[candidate_id] <= distance_limit
            and scale_ratio[candidate_id] <= _NEARBY_AREA_RATIO_LIMIT
        ]
        neighbors.append(
            sorted(candidates, key=lambda value: (distances[value], value))[:8]
        )
    return neighbors


def _component_fragment_records(
    component_ids: np.ndarray,
    fragment_ids: np.ndarray,
) -> list[dict[str, object]]:
    fragment_components: dict[int, set[int]] = {}
    for component_id, fragment_id in zip(
        component_ids.astype(int),
        fragment_ids.astype(int),
        strict=True,
    ):
        fragment_components.setdefault(fragment_id, set()).add(component_id)
    crossing = sorted(
        fragment_id
        for fragment_id, components in fragment_components.items()
        if len(components) != 1
    )
    if crossing:
        raise ValueError(
            f"Frozen fragments cross disconnected topology components: {crossing[:32]}"
        )
    return [
        {
            "component_id": component_id,
            "fragment_ids": sorted(
                int(value)
                for value in np.unique(fragment_ids[component_ids == component_id])
            ),
        }
        for component_id in range(int(component_ids.max()) + 1)
    ]


def _annotate_components(
    *,
    neutral_path: Path,
    face_ids_path: Path,
    component_ids: np.ndarray,
    output_path: Path,
) -> list[int]:
    face_ids = np.load(face_ids_path, allow_pickle=False)
    if face_ids.ndim != 2 or not np.issubdtype(face_ids.dtype, np.integer):
        raise ValueError(f"Face-ID buffer must be a 2D integer array: {face_ids_path}")
    valid = face_ids >= 0
    if np.any(face_ids[valid] >= len(component_ids)):
        raise ValueError(f"Face-ID buffer exceeds source topology: {face_ids_path}")
    visible_components = np.full(face_ids.shape, -1, dtype=np.int32)
    visible_components[valid] = component_ids[face_ids[valid]]

    image = Image.open(neutral_path).convert("RGB")
    if image.size != (face_ids.shape[1], face_ids.shape[0]):
        raise ValueError(
            f"Neutral image and Face-ID buffer shapes differ for {neutral_path}"
        )
    draw = ImageDraw.Draw(image)
    visible_ids = sorted(int(value) for value in np.unique(visible_components[valid]))
    for component_id in visible_ids:
        rows, columns = np.nonzero(visible_components == component_id)
        if len(columns) == 0:
            continue
        center_x = float(np.median(columns))
        center_y = float(np.median(rows))
        nearest = int(np.argmin((columns - center_x) ** 2 + (rows - center_y) ** 2))
        x = int(columns[nearest])
        y = int(rows[nearest])
        text = str(component_id)
        box = draw.textbbox((x, y), text, anchor="mm")
        draw.rectangle((box[0] - 2, box[1] - 1, box[2] + 2, box[3] + 1), fill="black")
        draw.text((x, y), text, fill=(255, 255, 0), anchor="mm")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return visible_ids


def _view_basis(direction: np.ndarray, *, up_axis: str) -> np.ndarray:
    view = direction / np.linalg.norm(direction)
    up = (
        np.asarray([0.0, 0.0, 1.0])
        if up_axis.upper() == "Z"
        else np.asarray([0.0, 1.0, 0.0])
    )
    right = np.cross(up, view)
    if np.linalg.norm(right) < 1.0e-8:
        right = np.cross(np.asarray([1.0, 0.0, 0.0]), view)
    right /= np.linalg.norm(right)
    screen_up = np.cross(view, right)
    screen_up /= np.linalg.norm(screen_up)
    return np.stack((right, screen_up, view), axis=1)


def _fit_projection(
    projected: np.ndarray,
    *,
    size: tuple[int, int],
    margin: int,
) -> np.ndarray:
    xy = projected[..., :2]
    flat = xy.reshape(-1, 2)
    lower = flat.min(axis=0)
    upper = flat.max(axis=0)
    span = np.maximum(upper - lower, 1.0e-8)
    width, height = size
    scale = min((width - 2 * margin) / span[0], (height - 2 * margin) / span[1])
    fitted = (xy - (lower + upper) * 0.5) * scale
    fitted[..., 0] += width * 0.5
    fitted[..., 1] = height * 0.5 - fitted[..., 1]
    return fitted


def _render_isolated_component(
    data: Any,
    *,
    component_id: int,
) -> tuple[Image.Image, Image.Image, str]:
    """Render occluded geometry itself plus its position in the full asset."""

    face_ids = np.flatnonzero(data.component_ids == component_id)
    if not len(face_ids):
        raise ValueError(f"Topology component {component_id} has no faces to render")
    vertices = data.points[data.triangles[face_ids]].astype(np.float64)
    best_name = ""
    best_basis: np.ndarray | None = None
    best_projected: np.ndarray | None = None
    best_score = -1.0
    for name, direction in _ISOLATED_VIEW_DIRECTIONS:
        basis = _view_basis(direction, up_axis=data.up_axis)
        projected = vertices @ basis
        edges_a = projected[:, 1, :2] - projected[:, 0, :2]
        edges_b = projected[:, 2, :2] - projected[:, 0, :2]
        signed_areas = edges_a[:, 0] * edges_b[:, 1] - edges_a[:, 1] * edges_b[:, 0]
        score = float(np.abs(signed_areas).sum())
        if score > best_score:
            best_name = name
            best_basis = basis
            best_projected = projected
            best_score = score
    assert best_basis is not None and best_projected is not None

    isolated = Image.new("RGB", _ISOLATED_RENDER_SIZE, "black")
    isolated_draw = ImageDraw.Draw(isolated)
    fitted = _fit_projection(best_projected, size=isolated.size, margin=12)
    depths = best_projected[..., 2].mean(axis=1)
    view = best_basis[:, 2]
    facing = np.abs(data.normals[face_ids].astype(np.float64) @ view)
    for local_index in np.argsort(depths):
        brightness = 0.45 + 0.55 * float(facing[local_index])
        color = (
            int(round(255 * brightness)),
            int(round(196 * brightness)),
            0,
        )
        polygon = [tuple(point) for point in fitted[local_index]]
        isolated_draw.polygon(polygon, fill=color, outline=(36, 30, 10))

    locator = Image.new("RGB", _LOCATOR_RENDER_SIZE, "black")
    locator_draw = ImageDraw.Draw(locator)
    all_projected = data.centroids.astype(np.float64) @ best_basis
    fitted_all = _fit_projection(all_projected, size=locator.size, margin=8)
    stride = max(1, len(fitted_all) // 1800)
    for x, y in fitted_all[::stride]:
        locator_draw.point((float(x), float(y)), fill=(68, 68, 68))
    component_projected = fitted_all[face_ids]
    for x, y in component_projected:
        locator_draw.ellipse(
            (float(x) - 1, float(y) - 1, float(x) + 1, float(y) + 1),
            fill=(255, 196, 0),
        )
    return isolated, locator, best_name


def _write_isolated_component_card(
    *,
    data: Any,
    component_id: int,
    nearby_component_ids: list[int],
    output_path: Path,
) -> str:
    isolated, locator, view_id = _render_isolated_component(
        data,
        component_id=component_id,
    )
    canvas = Image.new("RGB", _CARD_SIZE, "black")
    draw = ImageDraw.Draw(canvas)
    isolated = ImageOps.contain(
        isolated,
        (360, 338),
        method=Image.Resampling.LANCZOS,
    )
    locator = ImageOps.contain(
        locator,
        (252, 170),
        method=Image.Resampling.LANCZOS,
    )
    canvas.paste(isolated, (8, 38))
    canvas.paste(locator, (380, 38))
    draw.text((8, 8), f"component {component_id} | isolated {view_id}", fill="white")
    nearby = ",".join(str(value) for value in nearby_component_ids) or "none"
    draw.text((8, 22), f"nearby similar-scale components: {nearby}", fill="white")
    draw.text((380, 8), "global position", fill="white")
    draw.text(
        (380, 226),
        "occluded in registered views\n\ndiagnostic local preview only\n"
        "not semantic evidence",
        fill="white",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return view_id


def _write_component_crop(
    *,
    component_id: int,
    nearby_component_ids: list[int],
    view_id: str | None,
    neutral_path: Path | None,
    appearance_path: Path | None,
    component_mask: np.ndarray | None,
    output_path: Path,
) -> None:
    canvas = Image.new("RGB", _CARD_SIZE, "black")
    draw = ImageDraw.Draw(canvas)
    if neutral_path is None or component_mask is None or not np.any(component_mask):
        draw.text((12, 12), f"component {component_id}\nnot visible", fill="white")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output_path)
        return

    rows, columns = np.nonzero(component_mask)
    height, width = component_mask.shape
    span = max(int(columns.max() - columns.min()), int(rows.max() - rows.min()), 1)
    padding = max(12, int(round(span * 0.30)))
    left = max(int(columns.min()) - padding, 0)
    right = min(int(columns.max()) + padding + 1, width)
    top = max(int(rows.min()) - padding, 0)
    bottom = min(int(rows.max()) + padding + 1, height)

    neutral = np.asarray(Image.open(neutral_path).convert("RGB"), dtype=np.uint8)
    appearance = (
        np.asarray(Image.open(appearance_path).convert("RGB"), dtype=np.uint8)
        if appearance_path is not None
        else neutral
    )
    if appearance.shape != neutral.shape:
        raise ValueError(
            f"Appearance and neutral image shapes differ for {appearance_path}"
        )
    highlight = np.asarray([255, 196, 0], dtype=np.float32)

    def highlighted_crop(
        bounds: tuple[int, int, int, int],
        size: tuple[int, int],
        source: np.ndarray,
        *,
        preserve_component_appearance: bool = False,
    ) -> Image.Image:
        crop_left, crop_top, crop_right, crop_bottom = bounds
        crop = source[crop_top:crop_bottom, crop_left:crop_right].copy()
        crop_mask = component_mask[
            crop_top:crop_bottom,
            crop_left:crop_right,
        ]
        dimmed = np.rint(crop.astype(np.float32) * 0.50).astype(np.uint8)
        if preserve_component_appearance:
            dimmed[crop_mask] = crop[crop_mask]
            dilated = crop_mask.copy()
            dilated[1:, :] |= crop_mask[:-1, :]
            dilated[:-1, :] |= crop_mask[1:, :]
            dilated[:, 1:] |= crop_mask[:, :-1]
            dilated[:, :-1] |= crop_mask[:, 1:]
            dimmed[dilated & ~crop_mask] = highlight.astype(np.uint8)
        else:
            dimmed[crop_mask] = np.rint(
                crop[crop_mask].astype(np.float32) * 0.35 + highlight * 0.65
            ).astype(np.uint8)
        return ImageOps.contain(
            Image.fromarray(dimmed, mode="RGB"),
            size,
            method=Image.Resampling.LANCZOS,
        )

    tile = highlighted_crop(
        (left, top, right, bottom),
        (360, 338),
        appearance,
        preserve_component_appearance=appearance_path is not None,
    )
    canvas.paste(tile, ((376 - tile.width) // 2, 38 + (338 - tile.height) // 2))
    evidence_label = "source appearance" if appearance_path is not None else "neutral"
    draw.text(
        (8, 8),
        f"component {component_id} | {view_id} | {evidence_label}",
        fill="white",
    )
    nearby = ",".join(str(value) for value in nearby_component_ids) or "none"
    draw.text((8, 22), f"nearby similar-scale components: {nearby}", fill="white")

    local_padding = max(32, int(round(span * 2.50)))
    local_left = max(int(columns.min()) - local_padding, 0)
    local_right = min(int(columns.max()) + local_padding + 1, width)
    local_top = max(int(rows.min()) - local_padding, 0)
    local_bottom = min(int(rows.max()) + local_padding + 1, height)
    local_tile = highlighted_crop(
        (local_left, local_top, local_right, local_bottom),
        (252, 154),
        appearance,
        preserve_component_appearance=appearance_path is not None,
    )
    local_tile = ImageOps.expand(local_tile, border=2, fill="white")
    canvas.paste(
        local_tile,
        (632 - local_tile.width, 38 + (158 - local_tile.height) // 2),
    )
    draw.text((380, 8), "local source context", fill="white")

    # Preserve the full registered source view. A box supplies location without
    # recoloring a small component into a different material class.
    context = appearance.copy()
    context_image = Image.fromarray(context, mode="RGB")
    context_draw = ImageDraw.Draw(context_image)
    context_draw.rectangle(
        (left, top, max(right - 1, left), max(bottom - 1, top)),
        outline=(255, 196, 0),
        width=max(3, min(width, height) // 96),
    )
    context_tile = ImageOps.contain(
        context_image,
        (252, 144),
        method=Image.Resampling.LANCZOS,
    )
    context_tile = ImageOps.expand(context_tile, border=2, fill="white")
    canvas.paste(context_tile, (632 - context_tile.width, 376 - context_tile.height))
    draw.text(
        (380, 210),
        "full registered source view",
        fill="white",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _write_component_gallery(
    *,
    data: Any,
    component_ids: np.ndarray,
    statistics: list[dict[str, object]],
    fragment_records: list[dict[str, object]],
    views: list[dict[str, object]],
    provenance: dict[str, object],
    output_dir: Path,
) -> Path:
    gallery_dir = output_dir / "component_gallery"
    gallery_dir.mkdir(parents=True, exist_ok=True)
    prepared: list[dict[str, object]] = []
    for view in views:
        face_ids = np.load(Path(str(view["face_ids"])), allow_pickle=False)
        valid = face_ids >= 0
        visible_components = np.full(face_ids.shape, -1, dtype=np.int32)
        visible_components[valid] = component_ids[face_ids[valid]]
        prepared.append({**view, "component_ids": visible_components})

    gallery_records: list[dict[str, object]] = []
    crop_paths: list[Path] = []
    nearby_components = _nearby_component_ids(statistics)
    for component_id in range(len(statistics)):
        best_view: dict[str, object] | None = None
        best_mask: np.ndarray | None = None
        best_count = 0
        appearance_views = [
            view for view in prepared if isinstance(view.get("appearance_image"), str)
        ]
        for candidate_views in (appearance_views, prepared):
            for view in candidate_views:
                visible_components = view["component_ids"]
                assert isinstance(visible_components, np.ndarray)
                mask = visible_components == component_id
                count = int(mask.sum())
                if count > best_count:
                    best_view = view
                    best_mask = mask
                    best_count = count
            if best_count > 0:
                break
        crop_path = gallery_dir / f"component-{component_id:04d}.png"
        isolated_view_id: str | None = None
        if best_view is None:
            isolated_view_id = _write_isolated_component_card(
                data=data,
                component_id=component_id,
                nearby_component_ids=nearby_components[component_id],
                output_path=crop_path,
            )
        else:
            _write_component_crop(
                component_id=component_id,
                nearby_component_ids=nearby_components[component_id],
                view_id=str(best_view["view_id"]),
                neutral_path=Path(str(best_view["neutral_image"])),
                appearance_path=(
                    Path(str(best_view["appearance_image"]))
                    if isinstance(best_view.get("appearance_image"), str)
                    else None
                ),
                component_mask=best_mask,
                output_path=crop_path,
            )
        crop_paths.append(crop_path)
        gallery_records.append(
            {
                "component_id": component_id,
                "best_view_id": (
                    str(best_view["view_id"]) if best_view is not None else None
                ),
                "isolated_view_id": isolated_view_id,
                "evidence_mode": (
                    "registered_component_context"
                    if best_view is not None
                    else "diagnostic_only_local_preview"
                ),
                "semantic_decision_allowed": best_view is not None,
                "visible_pixel_count": best_count,
                "appearance_image": (
                    str(best_view["appearance_image"])
                    if best_view is not None
                    and isinstance(best_view.get("appearance_image"), str)
                    else None
                ),
                "face_count": statistics[component_id]["face_count"],
                "nearby_similar_scale_component_ids": nearby_components[component_id],
                "fragment_ids": fragment_records[component_id]["fragment_ids"],
                "crop": str(crop_path),
                "crop_sha256": sha256_file(crop_path),
            }
        )

    contact_sheets: list[str] = []
    contact_sheet_artifacts: list[dict[str, str]] = []
    page_size = _CONTACT_SHEET_COLUMNS * _CONTACT_SHEET_ROWS
    for page_index, start in enumerate(range(0, len(crop_paths), page_size)):
        page = Image.new(
            "RGB",
            (
                _CONTACT_SHEET_COLUMNS * _CARD_SIZE[0],
                _CONTACT_SHEET_ROWS * _CARD_SIZE[1],
            ),
            "black",
        )
        for tile_index, crop_path in enumerate(crop_paths[start : start + page_size]):
            tile = Image.open(crop_path).convert("RGB")
            page.paste(
                tile,
                (
                    (tile_index % _CONTACT_SHEET_COLUMNS) * _CARD_SIZE[0],
                    (tile_index // _CONTACT_SHEET_COLUMNS) * _CARD_SIZE[1],
                ),
            )
        page_path = gallery_dir / f"contact-sheet-{page_index:03d}.png"
        page.save(page_path)
        contact_sheets.append(str(page_path))
        contact_sheet_artifacts.append(
            {"path": str(page_path), "sha256": sha256_file(page_path)}
        )

    manifest_path = output_dir / "component_gallery_manifest.json"
    write_json(
        manifest_path,
        {
            "schema_version": _GALLERY_SCHEMA_VERSION,
            **provenance,
            "component_count": len(statistics),
            "selection_rule": (
                "largest_visible_pixel_count_across_registered_views_with_"
                "diagnostic_local_preview_for_occluded_components"
            ),
            "presentation": (
                ("source_appearance_zoom_with_local_and_full_registered_source_context")
                if any(isinstance(view.get("appearance_image"), str) for view in views)
                else "highlighted_zoom_with_local_and_registered_full_view_context"
            ),
            "card_size": list(_CARD_SIZE),
            "local_context_padding": "max(32px, 2.5x component span)",
            "nearby_component_rule": (
                "3D bounds gap <= 0.2% asset diagonal and surface-area ratio <= 20"
            ),
            "occluded_component_fallback": (
                "diagnostic-only local preview; obtain shared OVRTX evidence "
                "before any semantic decision"
            ),
            "components": gallery_records,
            "contact_sheets": contact_sheets,
            "contact_sheet_artifacts": contact_sheet_artifacts,
        },
    )
    return manifest_path


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_usd(args.source_usd, args.target)
    fragment_ids = load_fragment_labels(args.fragment_labels, data.face_count)
    component_ids = data.component_ids.astype("<u4", copy=False)
    component_ids_path = output_dir / "component_ids.u32le"
    component_ids.tofile(component_ids_path)

    statistics = _component_statistics(data)
    write_json(
        output_dir / "component_statistics.json",
        {
            "schema_version": "mesh-segmentation-topology-components.v1",
            **source_metadata(data),
            "component_count": len(statistics),
            "component_ids": str(component_ids_path),
            "component_ids_sha256": sha256_file(component_ids_path),
            "components": statistics,
        },
    )
    fragment_records = _component_fragment_records(component_ids, fragment_ids)
    write_json(
        output_dir / "component_fragment_map.json",
        {
            "schema_version": "mesh-segmentation-component-fragments.v1",
            "source_face_count": data.face_count,
            "component_count": len(statistics),
            "fragment_labels": str(args.fragment_labels.resolve()),
            "fragment_labels_sha256": sha256_file(args.fragment_labels.resolve()),
            "components": fragment_records,
        },
    )

    id_manifest_path = args.id_buffer_manifest.resolve()
    id_manifest = _read_object(id_manifest_path)
    expected_source = source_metadata(data)
    for field_name in (
        "source_sha256",
        "source_face_count",
        "target_prim_path",
        "topology_digest",
    ):
        if id_manifest.get(field_name) != expected_source[field_name]:
            raise ValueError(
                f"ID-buffer manifest {field_name} differs from the source mesh"
            )
    fragment_labels_sha256 = sha256_file(args.fragment_labels.resolve())
    if id_manifest.get("fragment_labels_sha256") != fragment_labels_sha256:
        raise ValueError("ID-buffer manifest fragment labels are stale")
    render_manifest_path = args.neutral_render_manifest.resolve()
    render_manifest = _read_object(render_manifest_path)
    if render_manifest.get("scene_sha256") != expected_source["source_sha256"]:
        raise ValueError("Neutral render manifest scene differs from the source USD")
    neutral_images = _render_images(
        render_manifest,
        base=render_manifest_path.parent,
    )
    neutral_records = _render_records(render_manifest)
    appearance_images = _appearance_images(
        args.appearance_dir,
        set(neutral_images),
    )
    raw_views = id_manifest.get("views")
    if not isinstance(raw_views, list) or not raw_views:
        raise ValueError("ID-buffer manifest lacks views")
    overlay_views: list[dict[str, object]] = []
    for index, record in enumerate(raw_views):
        if not isinstance(record, dict):
            raise ValueError(f"ID-buffer view {index} must be an object")
        name = record.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"ID-buffer view {index} lacks a name")
        channels = record.get("channels")
        face_channel = channels.get("face_ids") if isinstance(channels, dict) else None
        face_ids_path = _resolve_path(
            face_channel.get("raw") if isinstance(face_channel, dict) else None,
            base=id_manifest_path.parent,
            label=f"ID-buffer view {name!r} face IDs",
        )
        neutral_path = neutral_images.get(name)
        if neutral_path is None:
            raise ValueError(f"No neutral render matches ID-buffer view {name!r}")
        neutral_record = neutral_records[name]
        camera_sha256 = record.get("camera_sha256")
        if not isinstance(camera_sha256, str) or camera_sha256 != neutral_record.get(
            "camera_sha256"
        ):
            raise ValueError(f"Camera digest differs for registered view {name!r}")
        id_camera_path = _resolve_path(
            record.get("camera"),
            base=id_manifest_path.parent,
            label=f"ID-buffer view {name!r} camera",
        )
        neutral_camera_path = _resolve_path(
            neutral_record.get("camera"),
            base=render_manifest_path.parent,
            label=f"neutral render {name!r} camera",
        )
        if (
            sha256_file(id_camera_path) != camera_sha256
            or sha256_file(neutral_camera_path) != camera_sha256
        ):
            raise ValueError(f"Camera artifact digest is stale for view {name!r}")
        renderer = neutral_record.get("renderer")
        if renderer in (None, "", {}):
            raise ValueError(f"Neutral render {name!r} lacks OVRTX renderer metadata")
        overlay_path = output_dir / "overlays" / f"{name}.png"
        visible_ids = _annotate_components(
            neutral_path=neutral_path,
            face_ids_path=face_ids_path,
            component_ids=component_ids,
            output_path=overlay_path,
        )
        overlay_views.append(
            {
                "view_id": name,
                "neutral_image": str(neutral_path),
                "neutral_image_sha256": sha256_file(neutral_path),
                "appearance_image": (
                    str(appearance_images[name]) if name in appearance_images else None
                ),
                "appearance_image_sha256": (
                    sha256_file(appearance_images[name])
                    if name in appearance_images
                    else None
                ),
                "face_ids": str(face_ids_path),
                "face_ids_sha256": sha256_file(face_ids_path),
                "overlay": str(overlay_path),
                "overlay_sha256": sha256_file(overlay_path),
                "visible_component_ids": visible_ids,
                "camera_sha256": camera_sha256,
                "renderer": renderer,
            }
        )
    manifest_path = output_dir / "overlay_manifest.json"
    write_json(
        manifest_path,
        {
            "schema_version": "mesh-segmentation-component-overlay.v1",
            "source_asset": str(data.source_path),
            "source_sha256": sha256_file(data.source_path),
            "target_prim_path": data.target_path,
            "component_ids": str(component_ids_path),
            "component_ids_sha256": sha256_file(component_ids_path),
            "views": overlay_views,
        },
    )
    _write_component_gallery(
        data=data,
        component_ids=component_ids,
        statistics=statistics,
        fragment_records=fragment_records,
        views=overlay_views,
        provenance={
            "source_asset": str(data.source_path),
            "source_sha256": expected_source["source_sha256"],
            "target_prim_path": data.target_path,
            "topology_digest": data.topology_digest,
            "fragment_labels": str(args.fragment_labels.resolve()),
            "fragment_labels_sha256": fragment_labels_sha256,
            "id_buffer_manifest": str(id_manifest_path),
            "id_buffer_manifest_sha256": sha256_file(id_manifest_path),
            "neutral_render_manifest": str(render_manifest_path),
            "neutral_render_manifest_sha256": sha256_file(render_manifest_path),
            "registered_views": [
                {
                    "view_id": view["view_id"],
                    "camera_sha256": view["camera_sha256"],
                    "renderer": view["renderer"],
                }
                for view in overlay_views
            ],
        },
        output_dir=output_dir,
    )
    print(manifest_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
