#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Create deterministic boundary-safe mesh superfacets.

This implementation is based on the iterative face-graph clustering method in:

    Patricio D. Simari, Giulia Picciau, and Leila De Floriani,
    "Fast and Scalable Mesh Superfacets," Computer Graphics Forum,
    33(7):181-190, 2014. https://doi.org/10.1111/cgf.12486

The implementation is a boundary-safe adaptation: it first cuts edges at a
configured dihedral threshold, allocates one seed to every resulting component,
then distributes additional seeds and performs deterministic multi-source
shortest-path assignment with area-weighted center updates.
"""

from __future__ import annotations

import argparse
import colorsys
import json
import math
import time
from pathlib import Path

import numpy as np
import trimesh
from mesh_geometry import (
    MeshData,
    _author_source_mesh,
    _create_stage,
    load_usd,
    sha256_file,
    source_metadata,
    write_json,
)
from pxr import Sdf, UsdGeom, Vt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-usd", type=Path, required=True)
    parser.add_argument("--target")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--patch-count",
        type=int,
        default=1024,
        help=(
            "Extra seeds distributed beyond the mandatory one seed per "
            "hard-boundary component."
        ),
    )
    parser.add_argument("--hard-dihedral-degrees", type=float, default=25.0)
    parser.add_argument("--alpha", type=float, default=200.0)
    parser.add_argument("--eta-convex", type=float, default=1.0)
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Maximum center-update iterations.",
    )
    return parser.parse_args()


def _build_graph(
    data: MeshData,
    *,
    hard_dihedral_degrees: float,
    alpha: float,
    eta_convex: float,
) -> tuple[csr_matrix, np.ndarray, int]:
    topology = trimesh.Trimesh(
        vertices=data.points,
        faces=data.triangles,
        process=False,
    )
    adjacency = np.asarray(topology.face_adjacency, dtype=np.int64)
    shared_edges = np.asarray(topology.face_adjacency_edges, dtype=np.int64)
    valid = data.valid_faces[adjacency[:, 0]] & data.valid_faces[adjacency[:, 1]]
    adjacency = adjacency[valid]
    shared_edges = shared_edges[valid]

    first = adjacency[:, 0]
    second = adjacency[:, 1]
    edge_points = data.points[shared_edges]
    midpoints = edge_points.mean(axis=1)
    edge_lengths = np.linalg.norm(edge_points[:, 0] - edge_points[:, 1], axis=1)
    geodesic = np.linalg.norm(data.centroids[first] - midpoints, axis=1)
    geodesic += np.linalg.norm(data.centroids[second] - midpoints, axis=1)
    normal_dot = np.einsum(
        "ij,ij->i",
        data.normals[first],
        data.normals[second],
    )
    angles = np.arccos(np.clip(normal_dot, -1.0, 1.0)) / math.pi
    center_delta = data.centroids[second] - data.centroids[first]
    concave = np.einsum("ij,ij->i", data.normals[first], center_delta) > 0.0
    eta = np.where(concave, 1.0, eta_convex)
    angular = eta * angles * edge_lengths

    bounds = data.points.max(axis=0) - data.points.min(axis=0)
    bbox_diagonal = float(np.linalg.norm(bounds))
    if bbox_diagonal <= 0.0:
        raise ValueError("Mesh bounding-box diagonal must be positive")
    weights = (geodesic + alpha * angular) / bbox_diagonal
    weights = np.maximum(weights, np.finfo(np.float32).eps).astype(np.float64)
    hard_cut = angles * 180.0 >= hard_dihedral_degrees
    graph_first = first[~hard_cut]
    graph_second = second[~hard_cut]
    graph_weights = weights[~hard_cut]
    matrix = csr_matrix(
        (
            np.concatenate((graph_weights, graph_weights)),
            (
                np.concatenate((graph_first, graph_second)),
                np.concatenate((graph_second, graph_first)),
            ),
        ),
        shape=(data.face_count, data.face_count),
        dtype=np.float64,
    )
    matrix.sort_indices()
    return matrix, adjacency, int(hard_cut.sum())


def _farthest_seeds(
    data: MeshData,
    faces: np.ndarray,
    count: int,
) -> list[int]:
    centroids = data.centroids[faces].astype(np.float64, copy=False)
    area = data.areas[faces].astype(np.float64, copy=False)
    center = (
        np.average(centroids, axis=0, weights=area)
        if float(area.sum()) > 0.0
        else centroids.mean(axis=0)
    )
    first_local = int(np.argmin(np.sum((centroids - center) ** 2, axis=1)))
    seeds = [int(faces[first_local])]
    minimum_squared = np.sum(
        (centroids - centroids[first_local]) ** 2,
        axis=1,
    )
    selected = np.zeros(len(faces), dtype=bool)
    selected[first_local] = True
    for _ in range(1, count):
        next_local = int(np.argmax(np.where(selected, -np.inf, minimum_squared)))
        seeds.append(int(faces[next_local]))
        selected[next_local] = True
        squared = np.sum((centroids - centroids[next_local]) ** 2, axis=1)
        np.minimum(minimum_squared, squared, out=minimum_squared)
    return seeds


def _allocate_extra_patch_counts(
    face_counts: np.ndarray,
    areas: np.ndarray,
    requested: int,
) -> np.ndarray:
    """Allocate optional extra seeds across hard-boundary components."""

    capacity = np.maximum(face_counts - 1, 0)
    total = min(requested, int(capacity.sum()))
    extras = np.zeros(len(face_counts), dtype=np.int64)
    if total == 0:
        return extras

    weights = areas.astype(np.float64, copy=True)
    weights[capacity == 0] = 0.0
    if float(weights.sum()) <= 0.0:
        weights = capacity.astype(np.float64)
    ideal = total * weights / weights.sum()
    extras = np.minimum(np.floor(ideal).astype(np.int64), capacity)
    remaining = total - int(extras.sum())
    remainders = ideal - extras
    while remaining:
        eligible = extras < capacity
        component = int(np.argmax(np.where(eligible, remainders, -np.inf)))
        extras[component] += 1
        remainders[component] -= 1.0
        remaining -= 1
    return extras


def _update_local_centers(
    centroids: np.ndarray,
    areas: np.ndarray,
    assignments: np.ndarray,
    count: int,
) -> np.ndarray:
    weights = areas.astype(np.float64, copy=False)
    area_sums = np.bincount(assignments, weights=weights, minlength=count)
    face_counts = np.bincount(assignments, minlength=count)
    means = np.zeros((count, 3), dtype=np.float64)
    for axis in range(3):
        weighted = np.bincount(
            assignments,
            weights=weights * centroids[:, axis],
            minlength=count,
        )
        unweighted = np.bincount(
            assignments,
            weights=centroids[:, axis],
            minlength=count,
        )
        means[:, axis] = np.divide(
            weighted,
            area_sums,
            out=np.divide(
                unweighted,
                face_counts,
                out=np.zeros_like(unweighted),
                where=face_counts > 0,
            ),
            where=area_sums > 0,
        )

    squared = np.sum((centroids - means[assignments]) ** 2, axis=1)
    minimum = np.full(count, np.inf, dtype=np.float64)
    np.minimum.at(minimum, assignments, squared)
    candidates = squared == minimum[assignments]
    local_face_ids = np.arange(len(assignments), dtype=np.int64)
    centers = np.full(count, len(assignments), dtype=np.int64)
    np.minimum.at(centers, assignments[candidates], local_face_ids[candidates])
    if np.any(centers >= len(assignments)):
        raise RuntimeError("SciPy Superfacets produced an empty fragment")
    return centers


def _classify_component_scipy(
    graph: csr_matrix,
    data: MeshData,
    faces: np.ndarray,
    seed_faces: np.ndarray,
    iterations: int,
) -> tuple[np.ndarray, int]:
    """Partition one hard-boundary component with multi-source Dijkstra."""

    subgraph = graph[faces][:, faces].tocsr()
    local_seeds = np.searchsorted(faces, seed_faces).astype(np.int64)
    assignments = np.zeros(len(faces), dtype=np.int32)
    executed = 0
    for _ in range(iterations):
        executed += 1
        _, _, sources = dijkstra(
            subgraph,
            directed=False,
            indices=local_seeds,
            return_predecessors=True,
            min_only=True,
        )
        seed_to_label = np.full(len(faces), -1, dtype=np.int32)
        seed_to_label[local_seeds] = np.arange(
            len(local_seeds),
            dtype=np.int32,
        )
        assignments = seed_to_label[sources]
        if np.any(assignments < 0):
            raise RuntimeError(
                "SciPy multi-source classification left an unassigned face"
            )
        new_seeds = _update_local_centers(
            data.centroids[faces].astype(np.float64, copy=False),
            data.areas[faces].astype(np.float64, copy=False),
            assignments,
            len(local_seeds),
        )
        if np.array_equal(new_seeds, local_seeds):
            break
        local_seeds = new_seeds
    return assignments, executed


def _classify_scipy(
    graph: csr_matrix,
    data: MeshData,
    hard_component_ids: np.ndarray,
    hard_component_count: int,
    *,
    requested_extra_patch_count: int,
    iterations: int,
) -> tuple[np.ndarray, list[dict[str, object]], int]:
    face_counts = np.bincount(
        hard_component_ids,
        minlength=hard_component_count,
    )
    component_areas = np.bincount(
        hard_component_ids,
        weights=data.areas.astype(np.float64),
        minlength=hard_component_count,
    )
    extras = _allocate_extra_patch_counts(
        face_counts,
        component_areas,
        requested_extra_patch_count,
    )
    order = np.argsort(hard_component_ids, kind="stable")
    offsets = np.concatenate(([0], np.cumsum(face_counts)))
    labels = np.empty(data.face_count, dtype=np.int32)
    next_label = 0
    partitioned_components = 0
    iteration_histogram: dict[int, int] = {}

    for component in range(hard_component_count):
        faces = order[offsets[component] : offsets[component + 1]]
        count = 1 + int(extras[component])
        if count == 1:
            labels[faces] = next_label
            executed = 0
        else:
            partitioned_components += 1
            seed_faces = np.asarray(
                _farthest_seeds(data, faces, count),
                dtype=np.int64,
            )
            assignments, executed = _classify_component_scipy(
                graph,
                data,
                faces,
                seed_faces,
                iterations,
            )
            labels[faces] = next_label + assignments
        iteration_histogram[executed] = iteration_histogram.get(executed, 0) + 1
        next_label += count

    records: list[dict[str, object]] = [
        {
            "partitioned_hard_component_count": partitioned_components,
            "unpartitioned_hard_component_count": (
                hard_component_count - partitioned_components
            ),
            "component_iteration_count_histogram": {
                str(key): value for key, value in sorted(iteration_histogram.items())
            },
            "extra_patch_count": int(extras.sum()),
            "fragment_count": next_label,
        }
    ]
    return labels, records, next_label


def _split_disconnected(
    data: MeshData,
    labels: np.ndarray,
) -> tuple[np.ndarray, int, int]:
    same = labels[data.face_adjacency[:, 0]] == labels[data.face_adjacency[:, 1]]
    edges = data.face_adjacency[same]
    graph = csr_matrix(
        (
            np.ones(2 * len(edges), dtype=np.uint8),
            (
                np.concatenate((edges[:, 0], edges[:, 1])),
                np.concatenate((edges[:, 1], edges[:, 0])),
            ),
        ),
        shape=(data.face_count, data.face_count),
    )
    component_count, component_ids = connected_components(
        graph,
        directed=False,
        return_labels=True,
    )
    original_count = int(len(np.unique(labels)))
    component_patch = np.full(component_count, int(labels.max()) + 1, dtype=np.int64)
    component_first_face = np.full(component_count, data.face_count, dtype=np.int64)
    np.minimum.at(component_patch, component_ids, labels)
    np.minimum.at(
        component_first_face,
        component_ids,
        np.arange(data.face_count, dtype=np.int64),
    )
    order = np.lexsort((component_first_face, component_patch))
    remap = np.empty(component_count, dtype=np.uint32)
    remap[order] = np.arange(component_count, dtype=np.uint32)
    return remap[component_ids], int(component_count), original_count


def _fragment_adjacency(data: MeshData, labels: np.ndarray) -> np.ndarray:
    pairs = labels[data.face_adjacency]
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    if not len(pairs):
        return np.empty((0, 2), dtype=np.uint32)
    return np.unique(np.sort(pairs, axis=1), axis=0).astype(np.uint32)


def _palette(count: int) -> np.ndarray:
    colors = []
    for index in range(count):
        hue = (0.11 + index * 0.6180339887498949) % 1.0
        saturation = 0.58 + 0.18 * ((index * 17) % 3) / 2.0
        value = 0.84 + 0.14 * ((index * 29) % 4) / 3.0
        colors.append(colorsys.hsv_to_rgb(hue, saturation, value))
    return np.asarray(colors, dtype=np.float32)


def _write_fragment_usd(
    path: Path,
    data: MeshData,
    labels: np.ndarray,
    colors: np.ndarray,
) -> None:
    stage = _create_stage(path, data)
    mesh = _author_source_mesh(stage, data)
    primvar = UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
        "displayColor",
        Sdf.ValueTypeNames.Color3fArray,
        UsdGeom.Tokens.uniform,
    )
    primvar.Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(colors[labels])))
    stage.GetRootLayer().Save()


def main() -> None:
    args = parse_args()
    if args.patch_count < 0:
        raise ValueError("--patch-count must be nonnegative")
    if not 0.0 < args.hard_dihedral_degrees <= 180.0:
        raise ValueError("--hard-dihedral-degrees must be in (0, 180]")
    if args.alpha < 0.0:
        raise ValueError("--alpha must be nonnegative")
    if not 0.0 <= args.eta_convex <= 1.0:
        raise ValueError("--eta-convex must be in [0, 1]")
    if args.iterations < 1:
        raise ValueError("--iterations must be positive")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    data = load_usd(args.source_usd, args.target)
    component_count = int(data.component_ids.max()) + 1
    graph, source_adjacency, hard_cut_count = _build_graph(
        data,
        hard_dihedral_degrees=args.hard_dihedral_degrees,
        alpha=args.alpha,
        eta_convex=args.eta_convex,
    )
    hard_component_count, hard_component_ids = connected_components(
        graph,
        directed=False,
        return_labels=True,
    )
    classifier_name = "scipy_multi_source_shortest_path"
    classifier_source = Path(__file__).resolve()
    labels, iteration_records, effective_patch_count = _classify_scipy(
        graph,
        data,
        hard_component_ids,
        int(hard_component_count),
        requested_extra_patch_count=args.patch_count,
        iterations=args.iterations,
    )

    labels, fragment_count, pre_split_count = _split_disconnected(data, labels)
    fragment_adjacency = _fragment_adjacency(data, labels)
    labels_path = output_dir / "fragment_ids.u32le"
    labels.astype("<u4").tofile(labels_path)
    labels_npy = output_dir / "fragment_ids.npy"
    np.save(labels_npy, labels)
    adjacency_path = output_dir / "fragment_adjacency.npy"
    np.save(adjacency_path, fragment_adjacency)

    face_counts = np.bincount(labels, minlength=fragment_count)
    area = np.bincount(
        labels,
        weights=data.areas.astype(np.float64),
        minlength=fragment_count,
    )
    centroids = np.zeros((fragment_count, 3), dtype=np.float64)
    for axis in range(3):
        centroids[:, axis] = np.divide(
            np.bincount(
                labels,
                weights=data.areas * data.centroids[:, axis],
                minlength=fragment_count,
            ),
            area,
            out=np.bincount(
                labels,
                weights=data.centroids[:, axis],
                minlength=fragment_count,
            )
            / np.maximum(face_counts, 1),
            where=area > 0,
        )
    statistics_path = output_dir / "fragment_statistics.npz"
    np.savez_compressed(
        statistics_path,
        face_count=face_counts.astype(np.uint32),
        area=area,
        centroid=centroids,
    )
    colors = _palette(fragment_count)
    colors_path = output_dir / "fragment_colors.npy"
    np.save(colors_path, colors)
    diagnostic = output_dir / "fragments.usdc"
    _write_fragment_usd(diagnostic, data, labels, colors)
    manifest = {
        "schema_version": "mesh-segmentation-fragments.v1",
        **source_metadata(data),
        "algorithm": "boundary_safe_fast_mesh_superfacets",
        "algorithm_reference": {
            "title": "Fast and Scalable Mesh Superfacets",
            "authors": [
                "Patricio D. Simari",
                "Giulia Picciau",
                "Leila De Floriani",
            ],
            "publication": "Computer Graphics Forum 33(7):181-190",
            "year": 2014,
            "doi": "10.1111/cgf.12486",
        },
        "implementation_language": "python",
        "classifier_backend": "scipy",
        "classifier": classifier_name,
        "requested_patch_count": args.patch_count,
        "patch_count_semantics": "extra_seeds_beyond_hard_components",
        "effective_patch_count": effective_patch_count,
        "hard_dihedral_degrees": args.hard_dihedral_degrees,
        "alpha": args.alpha,
        "eta_convex": args.eta_convex,
        "center_update_iteration_limit": args.iterations,
        "topology_component_count": component_count,
        "hard_component_count": int(hard_component_count),
        "hard_cut_edge_count": hard_cut_count,
        "fragment_count_before_connectivity_split": pre_split_count,
        "fragment_count": fragment_count,
        "disconnected_islands_split": fragment_count - pre_split_count,
        "fragment_adjacency_count": int(len(fragment_adjacency)),
        "iterations": iteration_records,
        "runtime_seconds": time.perf_counter() - started,
        "source_face_adjacency_count": int(len(source_adjacency)),
        "fragment_face_count": {
            "minimum": int(face_counts.min()),
            "median": float(np.median(face_counts)),
            "mean": float(face_counts.mean()),
            "maximum": int(face_counts.max()),
        },
        "classifier_source": str(classifier_source),
        "classifier_source_sha256": sha256_file(classifier_source),
        "fragment_ids": str(labels_path),
        "fragment_ids_sha256": sha256_file(labels_path),
        "fragment_ids_npy": str(labels_npy),
        "fragment_adjacency": str(adjacency_path),
        "fragment_statistics": str(statistics_path),
        "fragment_colors": str(colors_path),
        "diagnostic_usd": str(diagnostic),
        "diagnostic_usd_sha256": sha256_file(diagnostic),
    }
    write_json(output_dir / "fragment_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
