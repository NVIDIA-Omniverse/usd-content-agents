#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolve batched viewport clicks to signed, closest-hit fragment evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp
from mesh_geometry import (
    load_fragment_labels,
    load_usd,
    sha256_file,
    source_metadata,
    write_json,
)
from warp_raycast import ray_intersect_mesh, resolve_warp_device


@wp.kernel
def raycast_faces(
    mesh_id: wp.uint64,
    origins: wp.array(dtype=wp.vec3),
    directions: wp.array(dtype=wp.vec3),
    distances: wp.array(dtype=wp.float32),
    barycentrics: wp.array(dtype=wp.vec2),
    face_ids: wp.array(dtype=wp.int32),
):
    index = wp.tid()
    distance, _normal, u, v, face_id = ray_intersect_mesh(
        mesh_id,
        origins[index],
        directions[index],
        False,
        1.0e6,
    )
    distances[index] = distance
    barycentrics[index] = wp.vec2(u, v)
    face_ids[index] = face_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-usd", type=Path, required=True)
    parser.add_argument("--target")
    parser.add_argument("--fragment-labels", type=Path, required=True)
    parser.add_argument("--clicks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--device",
        default="auto",
        help="Warp device; auto uses CUDA when available and otherwise CPU",
    )
    parser.add_argument("--edge-threshold", type=float, default=0.025)
    parser.add_argument("--skip-misses", action="store_true")
    return parser.parse_args()


def _near_fragment_boundary(
    face_id: int,
    barycentric: list[float],
    threshold: float,
    triangles: np.ndarray,
    fragment_labels: np.ndarray,
    face_adjacency: np.ndarray,
) -> bool:
    triangle = triangles[face_id]
    local_edges = (
        (triangle[1], triangle[2]),
        (triangle[0], triangle[2]),
        (triangle[0], triangle[1]),
    )
    fragment_id = int(fragment_labels[face_id])
    adjacent_rows = face_adjacency[
        (face_adjacency[:, 0] == face_id) | (face_adjacency[:, 1] == face_id)
    ]
    adjacent_faces = np.where(
        adjacent_rows[:, 0] == face_id,
        adjacent_rows[:, 1],
        adjacent_rows[:, 0],
    )
    for coordinate, edge in zip(barycentric, local_edges, strict=True):
        if coordinate >= threshold:
            continue
        neighbor_triangles = triangles[adjacent_faces]
        shares_edge = np.any(neighbor_triangles == edge[0], axis=1) & np.any(
            neighbor_triangles == edge[1],
            axis=1,
        )
        neighbors = adjacent_faces[shares_edge]
        if not len(neighbors) or any(
            int(fragment_labels[value]) != fragment_id for value in neighbors
        ):
            return True
    return False


def _camera_ray(
    camera_path: Path,
    pixel: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(camera_path.read_text(encoding="utf-8"))
    width = int(payload["image_width"])
    height = int(payload["image_height"])
    x, y = map(float, pixel)
    if not 0.0 <= x < width or not 0.0 <= y < height:
        raise ValueError(f"Pixel {pixel} is outside the {width}x{height} camera")
    camera = payload["camera_state"]
    horizontal_aperture = float(camera["horizontal_aperture"])
    vertical_aperture = horizontal_aperture * height / width
    focal_length = float(camera["focal_length"])
    local = np.asarray(
        [
            ((x + 0.5) / width * 2.0 - 1.0)
            * horizontal_aperture
            / (2.0 * focal_length),
            (1.0 - (y + 0.5) / height * 2.0) * vertical_aperture / (2.0 * focal_length),
            -1.0,
        ],
        dtype=np.float32,
    )
    local /= np.linalg.norm(local)
    transform = np.asarray(payload["camera_world_transform"], dtype=np.float32)
    origin = transform[3, :3].copy()
    direction = local @ transform[:3, :3]
    direction /= np.linalg.norm(direction)
    return origin, direction.astype(np.float32)


def _resolve_camera(clicks_path: Path, raw: dict[str, Any]) -> Path:
    camera = Path(str(raw["camera"]))
    if not camera.is_absolute():
        camera = clicks_path.parent / camera
    camera = camera.resolve()
    if not camera.is_file():
        raise FileNotFoundError(f"Camera JSON does not exist: {camera}")
    return camera


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.edge_threshold < (1.0 / 3.0):
        raise ValueError("--edge-threshold must be in [0, 1/3)")
    data = load_usd(args.source_usd, args.target)
    fragment_labels = load_fragment_labels(args.fragment_labels, data.face_count)
    clicks_path = args.clicks.resolve()
    payload = json.loads(clicks_path.read_text(encoding="utf-8"))
    raw_events = payload.get("events")
    if not isinstance(raw_events, list) or not raw_events:
        raise ValueError("Click file must contain a nonempty events list")

    origins: list[np.ndarray] = []
    directions: list[np.ndarray] = []
    cameras: list[Path] = []
    for index, raw in enumerate(raw_events):
        if not isinstance(raw, dict):
            raise ValueError(f"Click event {index} is not an object")
        polarity = str(raw.get("polarity", "")).lower()
        if polarity not in {"positive", "negative"}:
            raise ValueError(f"Click event {index} has invalid polarity")
        camera = _resolve_camera(clicks_path, raw)
        origin, direction = _camera_ray(camera, list(raw["pixel"]))
        cameras.append(camera)
        origins.append(origin)
        directions.append(direction)

    device = resolve_warp_device(args.device)
    warp_mesh = wp.Mesh(
        points=wp.array(data.points, dtype=wp.vec3, device=device),
        indices=wp.array(
            data.triangles.astype(np.int32).reshape(-1),
            dtype=wp.int32,
            device=device,
        ),
    )
    event_count = len(raw_events)
    distance_array = wp.empty(event_count, dtype=wp.float32, device=device)
    barycentric_array = wp.empty(event_count, dtype=wp.vec2, device=device)
    face_id_array = wp.empty(event_count, dtype=wp.int32, device=device)
    wp.launch(
        raycast_faces,
        dim=event_count,
        inputs=[
            warp_mesh.id,
            wp.array(np.asarray(origins), dtype=wp.vec3, device=device),
            wp.array(np.asarray(directions), dtype=wp.vec3, device=device),
            distance_array,
            barycentric_array,
            face_id_array,
        ],
        device=device,
    )
    wp.synchronize_device(device)
    distances = distance_array.numpy()
    barycentrics = barycentric_array.numpy()
    face_ids = face_id_array.numpy()

    records: list[dict[str, Any]] = []
    accepted_by_fragment: dict[int, str] = {}
    accepted_keys: set[tuple[int, str, str, str]] = set()
    for index, (raw, camera) in enumerate(zip(raw_events, cameras, strict=True)):
        face_id = int(face_ids[index])
        missed = face_id < 0 or not np.isfinite(distances[index])
        if missed and not args.skip_misses:
            raise RuntimeError(f"Click event {index} missed the mesh")
        barycentric = (
            [
                float(1.0 - barycentrics[index].sum()),
                float(barycentrics[index, 0]),
                float(barycentrics[index, 1]),
            ]
            if not missed
            else None
        )
        near_triangle_edge = (
            bool(min(barycentric) < args.edge_threshold)
            if barycentric is not None
            else False
        )
        near_fragment_boundary = (
            _near_fragment_boundary(
                face_id,
                barycentric,
                args.edge_threshold,
                data.triangles,
                fragment_labels,
                data.face_adjacency,
            )
            if barycentric is not None and near_triangle_edge
            else False
        )
        fragment_id = None if missed else int(fragment_labels[face_id])
        polarity = str(raw["polarity"]).lower()
        view_id = str(raw.get("view_id", camera.stem))
        instance_id = str(raw.get("instance_id", "default"))
        record = {
            "event_index": index,
            "camera": str(camera),
            "view_id": view_id,
            "instance_id": instance_id,
            "coverage_role": raw.get("coverage_role"),
            "confuser_id": raw.get("confuser_id"),
            "boundary_pair_id": raw.get("boundary_pair_id"),
            "pixel": [float(value) for value in raw["pixel"]],
            "polarity": polarity,
            "hit_face_id": None if missed else face_id,
            "face_id": None if missed else face_id,
            "fragment_id": fragment_id,
            "distance": None if missed else float(distances[index]),
            "barycentric": barycentric,
            "near_triangle_edge": near_triangle_edge,
            "near_fragment_boundary": near_fragment_boundary,
            "probe_passed": not missed,
            "rejection_reason": (
                "ray_miss"
                if missed
                else "near_fragment_boundary"
                if near_fragment_boundary
                else None
            ),
            "note": raw.get("note"),
        }
        records.append(record)
        if missed or near_fragment_boundary:
            continue
        assert fragment_id is not None
        prior = accepted_by_fragment.get(fragment_id)
        if prior is not None and prior != polarity:
            raise ValueError(
                f"Fragment {fragment_id} received contradictory signed clicks; "
                "inspect it before assigning a semantic label"
            )
        accepted_by_fragment[fragment_id] = polarity
        key = (fragment_id, polarity, view_id, instance_id)
        if key in accepted_keys:
            record["rejection_reason"] = "duplicate_fragment_view_polarity"
        accepted_keys.add(key)

    output = {
        "schema_version": "mesh-segmentation-fragment-evidence.v1",
        **source_metadata(data),
        "fragment_labels": str(args.fragment_labels.resolve()),
        "fragment_labels_sha256": sha256_file(args.fragment_labels.resolve()),
        "fragment_count": int(fragment_labels.max()) + 1,
        "click_source": str(clicks_path),
        "device": str(device),
        "edge_threshold": args.edge_threshold,
        "events": records,
    }
    write_json(args.output.resolve(), output)
    print(
        json.dumps(
            {
                "event_count": len(records),
                "accepted_fragment_count": len(accepted_by_fragment),
                "positive_fragment_count": sum(
                    value == "positive" for value in accepted_by_fragment.values()
                ),
                "negative_fragment_count": sum(
                    value == "negative" for value in accepted_by_fragment.values()
                ),
                "output": str(args.output.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
