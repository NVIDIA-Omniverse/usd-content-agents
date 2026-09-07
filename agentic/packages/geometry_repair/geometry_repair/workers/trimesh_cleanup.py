# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Conservative triangle cleanup without position welding or hole filling."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..attributed_rewrite import (
    ExactTopologyRewrite,
    exact_topology_rewrite,
    rewrite_usd_triangle_meshes_exact,
)
from ..mesh_io import (
    USD_SUFFIXES,
    _topology_vertex_ids,
    load_meshes,
)
from ..models import RepairOperation
from .base import WorkerResult


def _orient_consistently(
    faces: np.ndarray,
    topology_faces: np.ndarray,
) -> tuple[np.ndarray, int, bool]:
    """Orient manifold triangle components without changing face identity."""

    edge_owners: dict[tuple[int, int], list[tuple[int, tuple[int, int]]]] = {}
    for face_index, face in enumerate(topology_faces):
        for start, end in zip(face, np.roll(face, -1), strict=True):
            directed = (int(start), int(end))
            edge_owners.setdefault(tuple(sorted(directed)), []).append((face_index, directed))
    adjacency: dict[int, list[tuple[int, bool]]] = {index: [] for index in range(len(faces))}
    for owners in edge_owners.values():
        if len(owners) != 2:
            continue
        (left, left_edge), (right, right_edge) = owners
        same_direction = left_edge == right_edge
        adjacency[left].append((right, same_direction))
        adjacency[right].append((left, same_direction))
    flips: dict[int, bool] = {}
    conflict = False
    for seed in range(len(faces)):
        if seed in flips:
            continue
        flips[seed] = False
        queue = [seed]
        while queue:
            current = queue.pop()
            for neighbor, must_differ in adjacency[current]:
                expected = flips[current] ^ must_differ
                if neighbor in flips:
                    conflict = conflict or flips[neighbor] != expected
                    continue
                flips[neighbor] = expected
                queue.append(neighbor)
    output = faces.copy()
    flipped = 0
    for index, should_flip in flips.items():
        if should_flip:
            output[index] = output[index][[0, 2, 1]]
            flipped += 1
    return output, flipped, conflict


def _orient_closed_components_outward(
    vertices: np.ndarray,
    faces: np.ndarray,
    topology_faces: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Flip entire closed components whose signed volume points inward."""

    edge_owners: dict[tuple[int, int], list[int]] = {}
    for face_index, face in enumerate(topology_faces):
        for start, end in zip(face, np.roll(face, -1), strict=True):
            edge_owners.setdefault(tuple(sorted((int(start), int(end)))), []).append(face_index)
    adjacency: dict[int, set[int]] = {index: set() for index in range(len(faces))}
    for owners in edge_owners.values():
        if len(owners) == 2:
            adjacency[owners[0]].add(owners[1])
            adjacency[owners[1]].add(owners[0])
    output = faces.copy()
    visited: set[int] = set()
    flipped = 0
    for seed in range(len(faces)):
        if seed in visited:
            continue
        pending = [seed]
        component: list[int] = []
        while pending:
            current = pending.pop()
            if current in visited:
                continue
            visited.add(current)
            component.append(current)
            pending.extend(adjacency[current] - visited)
        component_set = set(component)
        component_edges = [
            owners
            for owners in edge_owners.values()
            if any(owner in component_set for owner in owners)
        ]
        if not component_edges or any(len(owners) != 2 for owners in component_edges):
            continue
        selected = output[np.asarray(component, dtype=np.int64)]
        coordinates = vertices[selected]
        signed_volume = float(
            np.sum(
                np.einsum(
                    "ij,ij->i",
                    coordinates[:, 0],
                    np.cross(coordinates[:, 1], coordinates[:, 2]),
                )
            )
            / 6.0
        )
        if signed_volume < 0.0:
            output[np.asarray(component, dtype=np.int64)] = selected[:, [0, 2, 1]]
            flipped += len(component)
    return output, flipped


class TrimeshCleanupWorker:
    """Remove exact duplicate/degenerate triangles and repair winding safely."""

    name = "trimesh_conservative_cleanup"
    operations = frozenset({"remove_degenerate_faces", "remove_duplicate_faces", "orient_faces"})

    def available(self) -> tuple[bool, str | None]:
        try:
            import trimesh  # noqa: F401

            return True, None
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"

    def execute(
        self,
        *,
        source: Path,
        output: Path,
        operation: RepairOperation,
    ) -> WorkerResult:
        if source.suffix.lower() not in USD_SUFFIXES:
            return WorkerResult(
                status="unavailable",
                failures=["conservative cleanup currently requires a prepared USD stage"],
            )
        meshes, _ = load_meshes(source)
        updates: dict[str, ExactTopologyRewrite] = {}
        deleted: list[str] = []
        warnings: list[str] = []
        operations: list[str] = []
        for mesh in meshes:
            if mesh.is_instance_proxy:
                warnings.append(
                    f"{mesh.path}: retained because instance proxies are read-only; "
                    "run scene_optimizer_deinstance first"
                )
                continue
            if len(mesh.source_face_counts) != len(mesh.triangles) or np.any(
                mesh.source_face_counts != 3
            ):
                warnings.append(
                    f"{mesh.path}: retained because conservative cleanup does not triangulate source n-gons"
                )
                continue
            faces = mesh.triangles.copy()
            if len(faces) == 0 or not np.isfinite(mesh.local_vertices).all():
                continue
            vectors_a = mesh.local_vertices[faces[:, 1]] - mesh.local_vertices[faces[:, 0]]
            vectors_b = mesh.local_vertices[faces[:, 2]] - mesh.local_vertices[faces[:, 0]]
            areas = np.linalg.norm(np.cross(vectors_a, vectors_b), axis=1) * 0.5
            bbox = np.ptp(mesh.local_vertices, axis=0)
            diagonal = float(np.linalg.norm(bbox))
            area_tolerance = max(diagonal**2 * 1e-16, 1e-24)
            nondegenerate = np.isfinite(areas) & (areas > area_tolerance)
            topology_faces = _topology_vertex_ids(mesh.local_vertices, diagonal)[faces]
            canonical = np.sort(topology_faces, axis=1)
            _, unique_indices = np.unique(canonical, axis=0, return_index=True)
            unique = np.zeros(len(faces), dtype=bool)
            unique[unique_indices] = True
            keep = nondegenerate & unique
            removed_count = int(len(faces) - np.count_nonzero(keep))
            cleaned = faces[keep]
            oriented, _consistency_flip_count, conflict = _orient_consistently(
                cleaned,
                topology_faces[keep],
            )
            if conflict:
                return WorkerResult(
                    status="failed",
                    failures=[f"{mesh.path}: orientation constraints are contradictory"],
                )
            oriented, _outward_flip_count = _orient_closed_components_outward(
                mesh.local_vertices,
                oriented,
                topology_faces[keep],
            )
            flip_count = int(np.count_nonzero(np.any(oriented != cleaned, axis=1)))
            if removed_count or flip_count:
                updates[mesh.path] = exact_topology_rewrite(
                    vertices=mesh.local_vertices,
                    source_triangles=faces,
                    output_triangles=oriented,
                    source_face_ids=np.flatnonzero(keep),
                )
            if removed_count:
                deleted.extend(
                    f"{mesh.path}:face:{index}" for index in np.flatnonzero(~keep).tolist()
                )
                operations.append(f"remove_invalid_faces:{mesh.path}:{removed_count}")
            if flip_count:
                operations.append(f"orient_faces:{mesh.path}:{flip_count}")
        if not updates:
            return WorkerResult(
                status="unavailable" if warnings else "completed",
                output_path=str(source),
                changed=False,
                warnings=warnings or ["no conservative cleanup operation was required"],
            )
        rewrite_usd_triangle_meshes_exact(source, output, updates)
        return WorkerResult(
            status="completed",
            output_path=str(output),
            changed=True,
            operations=operations,
            deleted_entities=deleted,
            warnings=warnings,
            metadata={
                "updated_mesh_paths": sorted(updates),
                "correspondence_mode": "exact_source_face_corner",
                "attribute_policy": "preserved_by_exact_domain_remap",
            },
        )
