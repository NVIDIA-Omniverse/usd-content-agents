# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded closure of unambiguous triangular or planar quadrilateral holes."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..attributed_rewrite import (
    GeneratedPatchTopologyRewrite,
    rewrite_usd_triangle_meshes_with_generated_patches,
)
from ..mesh_io import USD_SUFFIXES, _topology_vertex_ids, load_meshes
from ..models import RepairOperation
from .base import WorkerResult


def _boundary_loops(
    faces: np.ndarray,
    topology_ids: np.ndarray,
) -> list[list[int]] | None:
    topology_faces = topology_ids[faces]
    edge_owners: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for face in topology_faces:
        for start, end in zip(face, np.roll(face, -1), strict=True):
            edge_owners.setdefault(tuple(sorted((int(start), int(end)))), []).append(
                (int(start), int(end))
            )
    directed = [owners[0] for owners in edge_owners.values() if len(owners) == 1]
    if not directed:
        return []
    outgoing: dict[int, list[int]] = {}
    incoming: dict[int, list[int]] = {}
    for start, end in directed:
        outgoing.setdefault(start, []).append(end)
        incoming.setdefault(end, []).append(start)
    vertices = set(outgoing) | set(incoming)
    if any(
        len(outgoing.get(vertex, [])) != 1 or len(incoming.get(vertex, [])) != 1
        for vertex in vertices
    ):
        return None
    remaining = set(directed)
    loops: list[list[int]] = []
    while remaining:
        start_edge = min(remaining)
        start = start_edge[0]
        current = start
        loop: list[int] = []
        while True:
            loop.append(current)
            next_vertex = outgoing[current][0]
            edge = (current, next_vertex)
            if edge not in remaining:
                return None
            remaining.remove(edge)
            current = next_vertex
            if current == start:
                break
            if len(loop) > len(directed):
                return None
        loops.append(loop)
    return loops


class TrimeshBoundedHoleFillWorker:
    """Patch only small, closed, unambiguous boundary loops."""

    name = "trimesh_bounded_hole_fill"
    operations = frozenset({"triangular_hole_fill", "planar_quad_hole_fill"})

    def available(self) -> tuple[bool, str | None]:
        return True, None

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
                failures=["bounded hole fill currently requires a canonical USD working copy"],
            )
        max_loop_vertices = int(operation.parameters.get("max_loop_vertices", 4))
        max_loop_count = int(operation.parameters.get("max_loop_count", 8))
        max_perimeter_ratio = float(operation.parameters.get("max_perimeter_ratio", 0.2))
        max_added_area_ratio = float(operation.parameters.get("max_added_area_ratio", 0.02))
        planarity_ratio = float(operation.parameters.get("planarity_ratio", 1e-4))
        if max_loop_vertices not in {3, 4} or not 1 <= max_loop_count <= 32:
            return WorkerResult(status="failed", failures=["invalid bounded-hole topology limits"])
        if not 0.0 < max_perimeter_ratio <= 0.5 or not 0.0 < max_added_area_ratio <= 0.1:
            return WorkerResult(status="failed", failures=["invalid bounded-hole size limits"])
        if not 0.0 < planarity_ratio <= 0.01:
            return WorkerResult(status="failed", failures=["invalid bounded-hole planarity limit"])

        meshes, _ = load_meshes(source)
        updates: dict[str, GeneratedPatchTopologyRewrite] = {}
        records: list[dict[str, object]] = []
        for mesh in (item for item in meshes if item.role == "render"):
            if mesh.is_instance_proxy:
                return WorkerResult(
                    status="unavailable",
                    failures=[f"{mesh.path}: bounded hole fill cannot edit an instance proxy"],
                )
            vertices = np.asarray(mesh.local_vertices, dtype=np.float64)
            faces = np.asarray(mesh.triangles, dtype=np.int64)
            if not len(vertices) or not len(faces):
                continue
            diagonal = float(np.linalg.norm(np.ptp(vertices, axis=0)))
            if not math.isfinite(diagonal) or diagonal <= 1e-12:
                return WorkerResult(status="failed", failures=[f"{mesh.path}: invalid mesh scale"])
            topology_ids = _topology_vertex_ids(vertices, diagonal)
            loops = _boundary_loops(faces, topology_ids)
            if loops is None:
                return WorkerResult(
                    status="unavailable",
                    failures=[f"{mesh.path}: boundaries contain an open chain or branch"],
                )
            if not loops:
                continue
            if len(loops) > max_loop_count or any(
                len(loop) < 3 or len(loop) > max_loop_vertices for loop in loops
            ):
                return WorkerResult(
                    status="unavailable",
                    failures=[
                        f"{mesh.path}: {len(loops)} boundary loops exceed the bounded "
                        f"{max_loop_count}-loop/{max_loop_vertices}-vertex policy"
                    ],
                )
            representatives: dict[int, int] = {}
            for vertex_index, topology_id in enumerate(topology_ids):
                representatives.setdefault(int(topology_id), vertex_index)
            added: list[list[int]] = []
            generated_face_sources: dict[int, tuple[int, ...]] = {}
            total_patch_area = 0.0
            source_area = float(
                np.linalg.norm(
                    np.cross(
                        vertices[faces[:, 1]] - vertices[faces[:, 0]],
                        vertices[faces[:, 2]] - vertices[faces[:, 0]],
                    ),
                    axis=1,
                ).sum()
                * 0.5
            )
            topology_faces = topology_ids[faces]
            edge_owners: dict[tuple[int, int], list[int]] = {}
            for source_face, topology_face in enumerate(topology_faces):
                for start, end in zip(topology_face, np.roll(topology_face, -1), strict=True):
                    edge_owners.setdefault(tuple(sorted((int(start), int(end)))), []).append(
                        source_face
                    )
            for loop in loops:
                indices = [representatives[item] for item in loop]
                points = vertices[indices]
                perimeter = float(
                    np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1).sum()
                )
                if perimeter > diagonal * max_perimeter_ratio:
                    return WorkerResult(
                        status="unavailable",
                        failures=[
                            f"{mesh.path}: boundary perimeter ratio {perimeter / diagonal:.6g} "
                            f"exceeds {max_perimeter_ratio:.6g}"
                        ],
                    )
                if len(indices) == 4:
                    centered = points - points.mean(axis=0)
                    _left, _singular, right = np.linalg.svd(centered, full_matrices=False)
                    deviation = float(np.max(np.abs(centered @ right[-1])))
                    if deviation > diagonal * planarity_ratio:
                        return WorkerResult(
                            status="unavailable",
                            failures=[f"{mesh.path}: quadrilateral boundary is not planar enough"],
                        )
                reversed_loop = [indices[0], *reversed(indices[1:])]
                patch = (
                    [reversed_loop]
                    if len(reversed_loop) == 3
                    else [
                        [reversed_loop[0], reversed_loop[1], reversed_loop[2]],
                        [reversed_loop[0], reversed_loop[2], reversed_loop[3]],
                    ]
                )
                boundary_sources = tuple(
                    sorted(
                        {
                            edge_owners[tuple(sorted((int(start), int(end))))][0]
                            for start, end in zip(loop, np.roll(loop, -1), strict=True)
                        }
                    )
                )
                for triangle in patch:
                    coordinates = vertices[np.asarray(triangle, dtype=np.int64)]
                    total_patch_area += float(
                        np.linalg.norm(
                            np.cross(
                                coordinates[1] - coordinates[0], coordinates[2] - coordinates[0]
                            )
                        )
                        * 0.5
                    )
                    generated_face_sources[len(faces) + len(added)] = boundary_sources
                    added.append(triangle)
            area_ratio = total_patch_area / source_area if source_area > 0.0 else math.inf
            if not math.isfinite(area_ratio) or area_ratio > max_added_area_ratio:
                return WorkerResult(
                    status="unavailable",
                    failures=[
                        f"{mesh.path}: added area ratio {area_ratio:.6g} exceeds "
                        f"{max_added_area_ratio:.6g}"
                    ],
                )
            updated_faces = np.vstack((faces, np.asarray(added, dtype=np.int64)))
            updates[mesh.path] = GeneratedPatchTopologyRewrite(
                vertices=vertices,
                triangles=updated_faces,
                generated_face_sources=generated_face_sources,
            )
            records.append(
                {
                    "mesh_path": mesh.path,
                    "loop_count": len(loops),
                    "loop_vertex_counts": [len(loop) for loop in loops],
                    "added_face_count": len(added),
                    "added_area_ratio": area_ratio,
                }
            )
        if not updates:
            return WorkerResult(
                status="unavailable",
                failures=["no eligible bounded boundary loop was found"],
            )
        try:
            rewrite_usd_triangle_meshes_with_generated_patches(source, output, updates)
        except ValueError as exc:
            output.unlink(missing_ok=True)
            return WorkerResult(
                status="unavailable",
                failures=[f"generated patch attribute transfer refused: {exc}"],
            )
        return WorkerResult(
            status="completed",
            output_path=str(output),
            changed=True,
            operations=["fill_bounded_triangular_or_planar_quad_holes"],
            metadata={
                "holes": records,
                "attribute_policy": "exact_boundary_agreement_or_refuse",
                "generated_normals": "reauthored_geometric_face_normals_when_authored",
            },
        )
