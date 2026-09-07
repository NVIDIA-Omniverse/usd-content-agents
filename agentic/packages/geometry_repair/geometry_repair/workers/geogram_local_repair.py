# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Permissive native-kernel repair using a bounded Geogram vorpalite process."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

from ..attributed_rewrite import (
    ExactTopologyRewrite,
    exact_topology_rewrite,
    rewrite_usd_triangle_meshes_exact,
)
from ..hard_mesh_policy import verify_approved_executable_digest
from ..mesh_io import USD_SUFFIXES, load_meshes, update_usd_triangle_meshes
from ..models import RepairOperation
from .base import WorkerResult

GEOGRAM_EXECUTABLE_SHA256_ENVIRONMENT_VARIABLE = "GEOMETRY_REPAIR_GEOGRAM_EXECUTABLE_SHA256"


def _verified_vorpalite(
    *,
    expected_version: str,
) -> tuple[Path | None, str | None, str | None]:
    executable_name = shutil.which("vorpalite")
    if executable_name is None:
        return None, None, "Geogram vorpalite is not installed or not on PATH"
    executable = Path(executable_name)
    digest, digest_reason = verify_approved_executable_digest(
        executable,
        digest_environment_variable=GEOGRAM_EXECUTABLE_SHA256_ENVIRONMENT_VARIABLE,
    )
    if digest is None:
        return (
            None,
            None,
            "Geogram vorpalite identity rejected: "
            + (digest_reason or "executable digest could not be verified"),
        )
    try:
        completed = subprocess.run(
            [executable, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10.0,
        )
    except (OSError, subprocess.SubprocessError, UnicodeError) as exc:
        return (
            None,
            None,
            f"could not verify the audited vorpalite version: {type(exc).__name__}: {exc}",
        )
    version_text = f"{completed.stdout}\n{completed.stderr}"
    expected_banner = f"vorpalite {expected_version}"
    if completed.returncode != 0 or expected_banner not in version_text:
        return (
            None,
            None,
            f"geometry repair requires the audited {expected_banner} executable",
        )
    return executable, digest, None


def _triangle_geometry_equivalent(
    vertices: np.ndarray,
    faces: np.ndarray,
    candidate_vertices: np.ndarray,
    candidate_faces: np.ndarray,
) -> bool:
    from scipy.spatial import cKDTree

    source = np.asarray(vertices, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    candidate = np.asarray(candidate_vertices, dtype=np.float64)[
        np.asarray(candidate_faces, dtype=np.int64)
    ]
    if len(source) != len(candidate):
        return False
    if not len(source):
        return True
    diagonal = float(np.linalg.norm(np.ptp(source.reshape((-1, 3)), axis=0)))
    tolerance = max(diagonal * 2e-6, 2e-7)
    source_centroids = source.mean(axis=1)
    candidate_centroids = candidate.mean(axis=1)
    distances, source_ids = cKDTree(source_centroids).query(candidate_centroids, k=1)
    if np.any(distances > tolerance) or len(np.unique(source_ids)) != len(source):
        return False
    for candidate_index, source_index in enumerate(source_ids):
        source_triangle = source[int(source_index)]
        candidate_triangle = candidate[candidate_index]
        source_order = np.lexsort(
            (source_triangle[:, 2], source_triangle[:, 1], source_triangle[:, 0])
        )
        candidate_order = np.lexsort(
            (candidate_triangle[:, 2], candidate_triangle[:, 1], candidate_triangle[:, 0])
        )
        if (
            np.max(
                np.linalg.norm(
                    source_triangle[source_order] - candidate_triangle[candidate_order],
                    axis=1,
                )
            )
            > tolerance
        ):
            return False
    return True


def _exact_source_subset_rewrite(
    source_vertices: np.ndarray,
    source_faces: np.ndarray,
    candidate_vertices: np.ndarray,
    candidate_faces: np.ndarray,
) -> ExactTopologyRewrite | None:
    """Recover unique source face/corner identity or refuse generated topology."""

    from scipy.spatial import cKDTree

    source_points = np.asarray(source_vertices, dtype=np.float64).reshape((-1, 3))
    source_triangles = np.asarray(source_faces, dtype=np.int64).reshape((-1, 3))
    candidate_points = np.asarray(candidate_vertices, dtype=np.float64).reshape((-1, 3))
    candidate_triangles = np.asarray(candidate_faces, dtype=np.int64).reshape((-1, 3))
    if not len(source_triangles) or not len(candidate_triangles):
        return None
    source_geometry = source_points[source_triangles]
    candidate_geometry = candidate_points[candidate_triangles]
    diagonal = float(np.linalg.norm(np.ptp(source_points, axis=0)))
    tolerance = max(diagonal * 2e-6, 2e-7)
    centroid_tree = cKDTree(source_geometry.mean(axis=1))
    source_face_ids: list[int] = []
    used_source_faces: set[int] = set()
    for candidate_triangle in candidate_geometry:
        nearby = centroid_tree.query_ball_point(candidate_triangle.mean(axis=0), tolerance)
        matches: list[tuple[int, list[int]]] = []
        for source_face_id in sorted(int(value) for value in nearby):
            if source_face_id in used_source_faces:
                continue
            source_triangle = source_geometry[source_face_id]
            mapped_vertices: list[int] = []
            used_corners: set[int] = set()
            for candidate_corner in candidate_triangle:
                distances = np.linalg.norm(source_triangle - candidate_corner, axis=1)
                matching_corners = [
                    int(value)
                    for value in np.flatnonzero(distances <= tolerance)
                    if int(value) not in used_corners
                ]
                if len(matching_corners) != 1:
                    break
                source_corner = matching_corners[0]
                used_corners.add(source_corner)
                mapped_vertices.append(int(source_triangles[source_face_id, source_corner]))
            if len(mapped_vertices) == 3:
                matches.append((source_face_id, mapped_vertices))
        if len(matches) != 1:
            return None
        source_face_id, mapped_vertices = matches[0]
        used_source_faces.add(source_face_id)
        source_face_ids.append(source_face_id)
    ordered_source_face_ids = np.asarray(sorted(source_face_ids), dtype=np.int64)
    return exact_topology_rewrite(
        vertices=source_points,
        source_triangles=source_triangles,
        output_triangles=source_triangles[ordered_source_face_ids],
        source_face_ids=ordered_source_face_ids,
    )


class GeogramLocalRepairWorker:
    """Repair intersections and local topology without filling holes or deleting parts."""

    name = "geogram_local_repair"
    version = "1.10.0"
    operations = frozenset({"local_topology_repair", "intersection_repair"})

    def available(self) -> tuple[bool, str | None]:
        executable, digest, reason = _verified_vorpalite(expected_version=self.version)
        return executable is not None and digest is not None, reason

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
                failures=["Geogram local repair currently requires a prepared USD stage"],
            )
        import trimesh

        executable, executable_sha256, reason = _verified_vorpalite(expected_version=self.version)
        if executable is None or executable_sha256 is None:
            return WorkerResult(
                status="unavailable",
                failures=[reason or "vorpalite executable identity is unavailable"],
            )
        identity_metadata: dict[str, object] = {
            "vorpalite_version": self.version,
            "vorpalite_sha256": executable_sha256,
        }

        def worker_metadata(**values: object) -> dict[str, object]:
            return {**identity_metadata, **values}

        meshes, _ = load_meshes(source)
        inspection_issue_ids = {
            "mesh:self_intersections_not_evaluated",
            "mesh:coplanar_overlaps_not_evaluated",
        }
        inspection_only = (
            bool(operation.issue_ids) and set(operation.issue_ids) <= inspection_issue_ids
        )
        epsilon_ratio = float(operation.parameters.get("epsilon_ratio", 1e-8))
        if not 0.0 <= epsilon_ratio <= 1e-5:
            return WorkerResult(
                status="failed",
                failures=["epsilon_ratio must be between 0 and 1e-5 of bbox diagonal"],
                metadata=worker_metadata(),
            )
        enable_intersection = bool(
            set(operation.issue_ids)
            & {
                "mesh:self_intersections",
                "mesh:self_intersections_not_evaluated",
                "mesh:coplanar_overlaps",
                "mesh:coplanar_overlaps_not_evaluated",
                "mesh:over_connected_edges",
            }
        )
        work_dir = output.parent / "geogram"
        work_dir.mkdir(parents=True, exist_ok=True)
        updates: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        exact_updates: dict[str, ExactTopologyRewrite] = {}
        inspection_changed_paths: list[str] = []
        logs: list[dict[str, object]] = []
        environment = os.environ.copy()
        environment.update(
            {
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
            }
        )
        for index, mesh in enumerate(meshes):
            if not len(mesh.triangles) or not np.isfinite(mesh.local_vertices).all():
                continue
            input_path = work_dir / f"mesh_{index:04d}_input.obj"
            output_path = work_dir / f"mesh_{index:04d}_output.obj"
            stdout_path = work_dir / f"mesh_{index:04d}_stdout.log"
            stderr_path = work_dir / f"mesh_{index:04d}_stderr.log"
            trimesh.Trimesh(
                vertices=mesh.local_vertices,
                faces=mesh.triangles,
                process=False,
            ).export(input_path, file_type="obj")
            command = [
                str(executable),
                str(input_path),
                str(output_path),
                "profile=repair",
                "pre=true",
                f"pre:repair={'false' if inspection_only else 'true'}",
                f"pre:intersect={'true' if enable_intersection else 'false'}",
                f"pre:epsilon={(0.0 if inspection_only else epsilon_ratio) * 100.0:.12g}%",
                "pre:max_hole_area=0%",
                "pre:max_hole_edges=0",
                "pre:min_comp_area=0%",
                "pre:remove_internal_shells=false",
                "remesh=false",
                "post=false",
                "sys:max_threads=1",
                "log:quiet=true",
            ]
            with (
                stdout_path.open("w", encoding="utf-8") as stdout,
                stderr_path.open("w", encoding="utf-8") as stderr,
            ):
                completed = subprocess.run(
                    command,
                    check=False,
                    stdout=stdout,
                    stderr=stderr,
                    env=environment,
                    timeout=300.0,
                )
            logs.append(
                {
                    "mesh_path": mesh.path,
                    "command": command,
                    "return_code": completed.returncode,
                    "stdout_path": str(stdout_path),
                    "stderr_path": str(stderr_path),
                }
            )
            if completed.returncode != 0 or not output_path.is_file():
                return WorkerResult(
                    status="failed",
                    failures=[f"{mesh.path}: vorpalite exited {completed.returncode}"],
                    metadata=worker_metadata(invocations=logs),
                )
            repaired_scene = trimesh.load_scene(output_path, process=False)
            repaired_parts = [
                geometry
                for geometry in repaired_scene.geometry.values()
                if isinstance(geometry, trimesh.Trimesh) and len(geometry.faces)
            ]
            if len(repaired_parts) != 1:
                return WorkerResult(
                    status="failed",
                    failures=[
                        f"{mesh.path}: vorpalite output has {len(repaired_parts)} mesh payloads"
                    ],
                    metadata=worker_metadata(invocations=logs),
                )
            repaired = repaired_parts[0]
            vertices = np.asarray(repaired.vertices, dtype=np.float64).reshape((-1, 3))
            faces = np.asarray(repaired.faces, dtype=np.int64).reshape((-1, 3))
            if (
                not len(faces)
                or not np.isfinite(vertices).all()
                or np.any(faces < 0)
                or np.any(faces >= len(vertices))
            ):
                return WorkerResult(
                    status="failed",
                    failures=[f"{mesh.path}: vorpalite output failed bounded array checks"],
                    metadata=worker_metadata(invocations=logs),
                )
            if inspection_only:
                equivalent = _triangle_geometry_equivalent(
                    mesh.local_vertices,
                    mesh.triangles,
                    vertices,
                    faces,
                )
                if not equivalent:
                    inspection_changed_paths.append(mesh.path)
            else:
                attributed = bool(
                    mesh.material_subset_count
                    or mesh.material_binding_count
                    or mesh.has_face_varying_data
                    or mesh.authored_uv_count
                    or mesh.authored_normal_count
                )
                if attributed:
                    exact_rewrite = _exact_source_subset_rewrite(
                        mesh.local_vertices,
                        mesh.triangles,
                        vertices,
                        faces,
                    )
                    if exact_rewrite is None:
                        return WorkerResult(
                            status="unavailable",
                            failures=[
                                f"{mesh.path}: Geogram output is not a unique exact subset "
                                "of source triangles; attributed topology mutation requires "
                                "worker-native source face/corner evidence"
                            ],
                            metadata=worker_metadata(invocations=logs),
                        )
                    exact_updates[mesh.path] = exact_rewrite
                else:
                    updates[mesh.path] = (vertices, faces)
        if inspection_only:
            if inspection_changed_paths:
                return WorkerResult(
                    status="unavailable",
                    failures=[
                        "Geogram detected geometry that requires mutation, but attributed source "
                        "USD cannot be rewritten without an exact correspondence map: "
                        + ", ".join(inspection_changed_paths)
                    ],
                    metadata=worker_metadata(
                        inspection_only=True,
                        invocations=logs,
                    ),
                )
            return WorkerResult(
                status="completed",
                output_path=str(source),
                changed=False,
                operations=["geogram_exact_intersection_inspection"],
                verified_issue_ids=sorted(set(operation.issue_ids) & inspection_issue_ids),
                metadata=worker_metadata(
                    inspection_only=True,
                    source_geometry_retained=True,
                    invocations=logs,
                ),
            )
        if not updates and not exact_updates:
            return WorkerResult(
                status="unavailable",
                failures=["Geogram found no eligible triangle meshes"],
                metadata=worker_metadata(invocations=logs),
            )
        if updates:
            update_usd_triangle_meshes(source, output, updates)
        if exact_updates:
            exact_source = output if updates else source
            exact_target = output
            if updates:
                exact_target = output.with_name(f".{output.stem}.exact{output.suffix}")
                exact_target.unlink(missing_ok=True)
            rewrite_usd_triangle_meshes_exact(exact_source, exact_target, exact_updates)
            if exact_target != output:
                exact_target.replace(output)
        return WorkerResult(
            status="completed",
            output_path=str(output),
            changed=True,
            operations=[
                "geogram_pre_repair",
                *(["geogram_intersection_repair"] if enable_intersection else []),
            ],
            verified_issue_ids=sorted(
                set(operation.issue_ids)
                & {
                    "mesh:self_intersections_not_evaluated",
                    "mesh:coplanar_overlaps_not_evaluated",
                }
            ),
            metadata=worker_metadata(
                epsilon_ratio=epsilon_ratio,
                hole_filling_enabled=False,
                component_removal_enabled=False,
                internal_shell_removal_enabled=False,
                source_face_indices_changed=True,
                attributed_exact_rewrite_count=len(exact_updates),
                invocations=logs,
            ),
        )
