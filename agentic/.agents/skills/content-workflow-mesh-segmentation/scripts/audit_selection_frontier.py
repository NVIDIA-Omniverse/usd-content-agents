#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measure geometric completeness signals at the active selection frontier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from mesh_geometry import (
    fragment_atomic_conflicts,
    load_fragment_labels,
    load_labels,
    load_usd,
    sha256_file,
    source_metadata,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-usd", type=Path, required=True)
    parser.add_argument("--target")
    parser.add_argument("--fragment-labels", type=Path, required=True)
    parser.add_argument("--face-labels", type=Path, required=True)
    parser.add_argument("--active-segment-id", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _components(
    face_count: int,
    face_adjacency: np.ndarray,
    selected: np.ndarray,
) -> list[np.ndarray]:
    selected_ids = np.flatnonzero(selected)
    edges = face_adjacency[
        selected[face_adjacency[:, 0]] & selected[face_adjacency[:, 1]]
    ]
    values = trimesh.graph.connected_components(
        edges,
        nodes=selected_ids,
        min_len=1,
    )
    return sorted(
        (np.asarray(value, dtype=np.int64) for value in values),
        key=lambda value: (-len(value), int(value.min())),
    )


def _component_record(
    component_id: int,
    faces: np.ndarray,
    fragment_count: int,
    centroids: np.ndarray,
    boundary_selected: np.ndarray,
    areas: np.ndarray,
) -> dict[str, Any]:
    points = centroids[faces]
    dimensions = points.max(axis=0) - points.min(axis=0)
    positive_dimensions = dimensions[dimensions > 1.0e-8]
    aspect_ratio = (
        float(positive_dimensions.max() / positive_dimensions.min())
        if len(positive_dimensions) >= 2
        else None
    )
    boundary_count = int(np.count_nonzero(boundary_selected[faces]))
    return {
        "component_id": component_id,
        "fragment_count": fragment_count,
        "face_count": int(len(faces)),
        "area_sum": float(areas[faces].sum()),
        "bounds_min": points.min(axis=0).astype(float).tolist(),
        "bounds_max": points.max(axis=0).astype(float).tolist(),
        "centroid_dimensions": dimensions.astype(float).tolist(),
        "centroid_aspect_ratio": aspect_ratio,
        "boundary_face_count": boundary_count,
        "boundary_face_fraction": float(boundary_count / len(faces)),
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_usd(args.source_usd, args.target)
    fragment_labels = load_fragment_labels(args.fragment_labels, data.face_count)
    labels = load_labels(args.face_labels, data.face_count)
    fragment_conflicts = fragment_atomic_conflicts(fragment_labels, labels)
    if len(fragment_conflicts):
        raise ValueError(
            "Face labels split immutable fragments: "
            f"{fragment_conflicts.astype(int).tolist()}"
        )
    selected = labels == args.active_segment_id
    selected_degenerate_ids = np.flatnonzero(selected & ~data.valid_faces)
    if len(selected_degenerate_ids):
        raise ValueError(
            "The active segment includes degenerate source faces: "
            f"{selected_degenerate_ids.astype(int).tolist()}"
        )
    if not np.any(selected):
        raise ValueError("The active segment contains no faces")

    fragment_count = int(fragment_labels.max()) + 1
    fragment_adjacency = np.column_stack(
        (
            fragment_labels[data.face_adjacency[:, 0]],
            fragment_labels[data.face_adjacency[:, 1]],
        )
    )
    fragment_adjacency = fragment_adjacency[
        fragment_adjacency[:, 0] != fragment_adjacency[:, 1]
    ]
    fragment_adjacency.sort(axis=1)
    fragment_adjacency = np.unique(fragment_adjacency, axis=0).astype(np.int64)
    selected_fragments = np.zeros(fragment_count, dtype=bool)
    selected_fragments[fragment_labels[selected]] = True
    boundary_selected_fragments = np.zeros(fragment_count, dtype=bool)
    frontier_fragments = np.zeros(fragment_count, dtype=bool)
    for first, second in fragment_adjacency:
        first_selected = bool(selected_fragments[first])
        second_selected = bool(selected_fragments[second])
        if first_selected == second_selected:
            continue
        boundary_selected_fragments[first if first_selected else second] = True
        frontier_fragments[second if first_selected else first] = True
    boundary_selected = np.zeros(data.face_count, dtype=bool)
    frontier = np.zeros(data.face_count, dtype=bool)
    for first, second in data.face_adjacency:
        first_selected = bool(selected[first])
        second_selected = bool(selected[second])
        if first_selected == second_selected:
            continue
        boundary_selected[first if first_selected else second] = True
        frontier[second if first_selected else first] = True

    fragment_components = _components(
        fragment_count,
        fragment_adjacency,
        selected_fragments,
    )
    components = [
        np.flatnonzero(np.isin(fragment_labels, fragments))
        for fragments in fragment_components
    ]
    component_records = [
        _component_record(
            component_id,
            faces,
            len(fragment_components[component_id]),
            data.centroids,
            boundary_selected,
            data.areas,
        )
        for component_id, faces in enumerate(components)
    ]
    selected_ids_path = output_dir / "selected_face_ids.u32le"
    boundary_ids_path = output_dir / "boundary_selected_face_ids.u32le"
    frontier_ids_path = output_dir / "unselected_frontier_face_ids.u32le"
    selected_fragment_ids_path = output_dir / "selected_fragment_ids.u32le"
    boundary_fragment_ids_path = output_dir / "boundary_selected_fragment_ids.u32le"
    frontier_fragment_ids_path = output_dir / "unselected_frontier_fragment_ids.u32le"
    np.flatnonzero(selected).astype("<u4").tofile(selected_ids_path)
    np.flatnonzero(boundary_selected).astype("<u4").tofile(boundary_ids_path)
    np.flatnonzero(frontier).astype("<u4").tofile(frontier_ids_path)
    np.flatnonzero(selected_fragments).astype("<u4").tofile(selected_fragment_ids_path)
    np.flatnonzero(boundary_selected_fragments).astype("<u4").tofile(
        boundary_fragment_ids_path
    )
    np.flatnonzero(frontier_fragments).astype("<u4").tofile(frontier_fragment_ids_path)

    selected_count = int(np.count_nonzero(selected))
    boundary_count = int(np.count_nonzero(boundary_selected))
    warnings = []
    if len(fragment_components) > 32:
        warnings.append(
            "The active segment has more than 32 disconnected components; "
            "inspect for floaters or repeated instances."
        )
    if boundary_count / selected_count > 0.75:
        warnings.append(
            "More than 75% of selected faces touch the frontier; the result may "
            "be sparse, thin, or under-segmented."
        )
    if any(
        record["centroid_aspect_ratio"] is not None
        and record["centroid_aspect_ratio"] > 8.0
        for record in component_records
    ):
        warnings.append(
            "An elongated selected component needs explicit endpoint validation "
            "from an axial or grazing view."
        )

    payload = {
        "schema_version": "mesh-segmentation-fragment-frontier-audit.v1",
        **source_metadata(data),
        "semantic_decision_unit": "immutable_fragment",
        "fragment_labels": str(args.fragment_labels.resolve()),
        "fragment_labels_sha256": sha256_file(args.fragment_labels.resolve()),
        "fragment_count": fragment_count,
        "face_labels": str(args.face_labels.resolve()),
        "face_labels_sha256": sha256_file(args.face_labels.resolve()),
        "active_segment_id": args.active_segment_id,
        "selected_face_count": selected_count,
        "selected_area_sum": float(data.areas[selected].sum()),
        "selected_fragment_count": int(np.count_nonzero(selected_fragments)),
        "selected_component_count": len(fragment_components),
        "boundary_selected_fragment_count": int(
            np.count_nonzero(boundary_selected_fragments)
        ),
        "unselected_frontier_fragment_count": int(np.count_nonzero(frontier_fragments)),
        "boundary_selected_face_count": boundary_count,
        "unselected_frontier_face_count": int(np.count_nonzero(frontier)),
        "crossing_fragment_adjacency_edge_count": int(
            np.count_nonzero(
                selected_fragments[fragment_adjacency[:, 0]]
                != selected_fragments[fragment_adjacency[:, 1]]
            )
        ),
        "components": component_records,
        "warnings": warnings,
        "selected_face_ids": str(selected_ids_path),
        "boundary_selected_face_ids": str(boundary_ids_path),
        "unselected_frontier_face_ids": str(frontier_ids_path),
        "selected_fragment_ids": str(selected_fragment_ids_path),
        "boundary_selected_fragment_ids": str(boundary_fragment_ids_path),
        "unselected_frontier_fragment_ids": str(frontier_fragment_ids_path),
    }
    write_json(output_dir / "frontier_audit.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
