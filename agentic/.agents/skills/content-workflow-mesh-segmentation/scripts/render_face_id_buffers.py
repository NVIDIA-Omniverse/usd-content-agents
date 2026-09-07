#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render registered closest-visible face, fragment, and label ID buffers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp
from mesh_geometry import (
    load_fragment_labels,
    load_labels,
    load_usd,
    sha256_file,
    source_metadata,
    write_json,
)
from PIL import Image
from warp_raycast import ray_intersect_mesh, resolve_warp_device


@wp.kernel
def raycast_face_ids(
    mesh_id: wp.uint64,
    origins: wp.array(dtype=wp.vec3),
    directions: wp.array(dtype=wp.vec3),
    face_ids: wp.array(dtype=wp.int32),
):
    index = wp.tid()
    _distance, _normal, _u, _v, face_id = ray_intersect_mesh(
        mesh_id,
        origins[index],
        directions[index],
        False,
        1.0e6,
    )
    face_ids[index] = face_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-usd", type=Path, required=True)
    parser.add_argument("--target")
    parser.add_argument("--fragment-labels", type=Path, required=True)
    parser.add_argument("--face-labels", type=Path)
    parser.add_argument("--active-segment-id", type=int, default=1)
    parser.add_argument("--camera-json", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--device",
        default="auto",
        help="Warp device; auto uses CUDA when available and otherwise CPU",
    )
    return parser.parse_args()


def _camera_rays(
    camera_path: Path,
) -> tuple[np.ndarray, np.ndarray, int, int, dict[str, Any]]:
    payload = json.loads(camera_path.read_text(encoding="utf-8"))
    width = int(payload["image_width"])
    height = int(payload["image_height"])
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid camera dimensions in {camera_path}")
    camera = payload["camera_state"]
    horizontal_aperture = float(camera["horizontal_aperture"])
    vertical_aperture = horizontal_aperture * height / width
    focal_length = float(camera["focal_length"])
    xs = (
        ((np.arange(width, dtype=np.float32) + 0.5) / width * 2.0 - 1.0)
        * horizontal_aperture
        / (2.0 * focal_length)
    )
    ys = (
        (1.0 - (np.arange(height, dtype=np.float32) + 0.5) / height * 2.0)
        * vertical_aperture
        / (2.0 * focal_length)
    )
    local = np.empty((height, width, 3), dtype=np.float32)
    local[..., 0] = xs[None, :]
    local[..., 1] = ys[:, None]
    local[..., 2] = -1.0
    local /= np.linalg.norm(local, axis=2, keepdims=True)
    transform = np.asarray(payload["camera_world_transform"], dtype=np.float32)
    if transform.shape != (4, 4):
        raise ValueError(f"Invalid camera transform in {camera_path}")
    directions = local.reshape(-1, 3) @ transform[:3, :3]
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    origin = transform[3, :3]
    origins = np.broadcast_to(origin, directions.shape).copy()
    return origins, directions.astype(np.float32), width, height, payload


def _reversible_id_rgb(ids: np.ndarray) -> np.ndarray:
    """Encode nonnegative IDs as exact RGB bytes; black is reserved for misses."""

    encoded = ids.astype(np.int64, copy=False) + 1
    valid = ids >= 0
    rgb = np.zeros((*ids.shape, 3), dtype=np.uint8)
    rgb[..., 0][valid] = (encoded[valid] & 0xFF).astype(np.uint8)
    rgb[..., 1][valid] = ((encoded[valid] >> 8) & 0xFF).astype(np.uint8)
    rgb[..., 2][valid] = ((encoded[valid] >> 16) & 0xFF).astype(np.uint8)
    return rgb


def _hashed_id_rgb(ids: np.ndarray) -> np.ndarray:
    """Create a stable high-contrast preview while preserving raw IDs in NPY."""

    values = ids.astype(np.uint32, copy=False)
    mixed = values.copy()
    mixed ^= mixed >> np.uint32(16)
    mixed *= np.uint32(0x7FEB352D)
    mixed ^= mixed >> np.uint32(15)
    mixed *= np.uint32(0x846CA68B)
    mixed ^= mixed >> np.uint32(16)
    rgb = np.zeros((*ids.shape, 3), dtype=np.uint8)
    valid = ids >= 0
    rgb[..., 0][valid] = (64 + (mixed[valid] & 0xBF)).astype(np.uint8)
    rgb[..., 1][valid] = (64 + ((mixed[valid] >> 8) & 0xBF)).astype(np.uint8)
    rgb[..., 2][valid] = (64 + ((mixed[valid] >> 16) & 0xBF)).astype(np.uint8)
    return rgb


def _flat_label_rgb(
    face_ids: np.ndarray,
    face_labels: np.ndarray,
    active_segment_id: int,
) -> np.ndarray:
    rgb = np.zeros((*face_ids.shape, 3), dtype=np.uint8)
    hit = face_ids >= 0
    rgb[hit] = np.asarray([112, 120, 128], dtype=np.uint8)
    active = np.zeros(face_ids.shape, dtype=bool)
    active[hit] = face_labels[face_ids[hit]] == active_segment_id
    rgb[active] = np.asarray([242, 20, 158], dtype=np.uint8)
    return rgb


def _save_array_and_preview(
    output_dir: Path,
    stem: str,
    suffix: str,
    values: np.ndarray,
    preview: np.ndarray,
) -> dict[str, Any]:
    raw_path = output_dir / f"{stem}_{suffix}.npy"
    preview_path = output_dir / f"{stem}_{suffix}.png"
    np.save(raw_path, values, allow_pickle=False)
    Image.fromarray(preview, mode="RGB").save(preview_path)
    return {
        "raw": str(raw_path),
        "raw_sha256": sha256_file(raw_path),
        "preview": str(preview_path),
        "preview_sha256": sha256_file(preview_path),
        "shape": list(values.shape),
        "dtype": str(values.dtype),
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_usd(args.source_usd, args.target)
    fragment_labels = load_fragment_labels(args.fragment_labels, data.face_count)
    face_labels = (
        load_labels(args.face_labels, data.face_count)
        if args.face_labels is not None
        else None
    )

    device = resolve_warp_device(args.device)
    warp_mesh = wp.Mesh(
        points=wp.array(data.points, dtype=wp.vec3, device=device),
        indices=wp.array(
            data.triangles.astype(np.int32).reshape(-1),
            dtype=wp.int32,
            device=device,
        ),
    )
    records: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for raw_camera in args.camera_json:
        camera_path = raw_camera.resolve()
        if not camera_path.is_file():
            raise FileNotFoundError(f"Camera JSON does not exist: {camera_path}")
        stem = camera_path.stem.removesuffix("_camera")
        if stem in seen_names:
            raise ValueError(f"Duplicate camera view name: {stem}")
        seen_names.add(stem)
        origins, directions, width, height, _payload = _camera_rays(camera_path)
        ray_count = len(directions)
        face_array = wp.empty(ray_count, dtype=wp.int32, device=device)
        wp.launch(
            raycast_face_ids,
            dim=ray_count,
            inputs=[
                warp_mesh.id,
                wp.array(origins, dtype=wp.vec3, device=device),
                wp.array(directions, dtype=wp.vec3, device=device),
                face_array,
            ],
            device=device,
        )
        wp.synchronize_device(device)
        face_ids = face_array.numpy().reshape(height, width).astype(np.int32)
        fragment_ids = np.full(face_ids.shape, -1, dtype=np.int32)
        hit = face_ids >= 0
        fragment_ids[hit] = fragment_labels[face_ids[hit]].astype(np.int32)
        channels: dict[str, Any] = {
            "face_ids": _save_array_and_preview(
                output_dir,
                stem,
                "face_ids",
                face_ids,
                _reversible_id_rgb(face_ids),
            ),
            "fragment_ids": _save_array_and_preview(
                output_dir,
                stem,
                "fragment_ids",
                fragment_ids,
                _hashed_id_rgb(fragment_ids),
            ),
        }
        if face_labels is not None:
            label_ids = np.full(face_ids.shape, -1, dtype=np.int32)
            label_ids[hit] = face_labels[face_ids[hit]].astype(np.int32)
            channels["label_ids"] = _save_array_and_preview(
                output_dir,
                stem,
                "label_ids",
                label_ids,
                _flat_label_rgb(face_ids, face_labels, args.active_segment_id),
            )
        records.append(
            {
                "name": stem,
                "camera": str(camera_path),
                "camera_sha256": sha256_file(camera_path),
                "width": width,
                "height": height,
                "closest_visible_hit_only": True,
                "visible_face_count": int(len(np.unique(face_ids[hit]))),
                "visible_fragment_count": int(len(np.unique(fragment_ids[hit]))),
                "channels": channels,
            }
        )
        print(f"rendered ID buffers for {stem}", flush=True)

    manifest = {
        "schema_version": "mesh-segmentation-id-buffers.v1",
        **source_metadata(data),
        "fragment_labels": str(args.fragment_labels.resolve()),
        "fragment_labels_sha256": sha256_file(args.fragment_labels.resolve()),
        "face_labels": (
            str(args.face_labels.resolve()) if args.face_labels is not None else None
        ),
        "face_labels_sha256": (
            sha256_file(args.face_labels.resolve())
            if args.face_labels is not None
            else None
        ),
        "active_segment_id": args.active_segment_id,
        "device": str(device),
        "face_id_preview_encoding": (
            "RGB little-endian encoding of face_id+1; black means ray miss"
        ),
        "views": records,
    }
    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, manifest)
    print(json.dumps({"manifest": str(manifest_path), "views": len(records)}, indent=2))


if __name__ == "__main__":
    main()
