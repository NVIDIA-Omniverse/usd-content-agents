# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent source-to-candidate geometry and identity comparison."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import trimesh

from .artifacts import atomic_write_json, file_sha256
from .mesh_io import MeshData, load_meshes, measure_asset
from .models import DriftBand, FidelityReport, ProtectedFeature, RepairBudgets
from .protected_features import evaluate_feature_probes
from .surface import (
    closest_surface,
    deterministic_surface_samples,
    oriented_bbox_extents,
    silhouette_iou,
    unique_geometry_points,
)


def _relative_drift(source: float | None, candidate: float | None) -> float | None:
    if source is None or candidate is None:
        return None
    if abs(source) <= 1e-18:
        return 0.0 if abs(candidate) <= 1e-18 else None
    return abs(candidate - source) / abs(source)


def _limits(band: DriftBand, budgets: RepairBudgets) -> tuple[float, float]:
    if band in {"identity", "conservative"}:
        return budgets.conservative_p99_ratio, budgets.conservative_volume_drift
    if band == "moderate":
        return budgets.moderate_p99_ratio, budgets.moderate_volume_drift
    return budgets.reconstructive_p99_ratio, budgets.reconstructive_volume_drift


def _canonical_relative_paths(paths: list[str]) -> dict[str, str]:
    """Strip package-only common ancestors while retaining semantic descendants."""

    if not paths:
        return {}
    if len(paths) == 1:
        return {paths[0]: "/"}
    components = [tuple(part for part in path.split("/") if part) for path in paths]
    common_count = 0
    for items in zip(*components, strict=False):
        if len(set(items)) != 1:
            break
        common_count += 1
    common_count = min(common_count, min(len(items) - 1 for items in components))
    return {
        path: "/" + "/".join(items[common_count:])
        for path, items in zip(paths, components, strict=True)
    }


def _relative_part_paths(metrics) -> set[str]:
    paths = list(metrics.source_part_paths)
    return set(_canonical_relative_paths(paths).values())


def _coverage_ratio(distances: np.ndarray, tolerance_m: float) -> float | None:
    if not len(distances):
        return None
    return float(np.count_nonzero(distances <= tolerance_m) / len(distances))


def _mapping_coverage(source_count: int, candidate_count: int) -> float:
    if source_count == 0:
        return 1.0 if candidate_count == 0 else 0.0
    return min(source_count, candidate_count) / max(source_count, candidate_count, 1)


def _exact_part_correspondence(
    source_by_path: dict[str, MeshData],
    candidate_by_path: dict[str, MeshData],
) -> tuple[list[dict[str, object]], int]:
    records: list[dict[str, object]] = []
    failed = 0
    for key in sorted(source_by_path.keys() | candidate_by_path.keys()):
        source_mesh = source_by_path.get(key)
        candidate_mesh = candidate_by_path.get(key)
        if source_mesh is None or candidate_mesh is None:
            records.append(
                {
                    "relative_path": key,
                    "source_path": source_mesh.path if source_mesh else None,
                    "candidate_path": candidate_mesh.path if candidate_mesh else None,
                    "status": "missing",
                }
            )
            failed += 1
            continue
        semantics_preserved = (
            source_mesh.material_subset_count == candidate_mesh.material_subset_count
            and source_mesh.material_binding_count == candidate_mesh.material_binding_count
        )
        status = "pass" if semantics_preserved else "fail"
        failed += status != "pass"
        records.append(
            {
                "relative_path": key,
                "source_path": source_mesh.path,
                "candidate_path": candidate_mesh.path,
                "status": status,
                "source_sample_count": 0,
                "candidate_sample_count": 0,
                "source_coverage_ratio": 1.0,
                "candidate_coverage_ratio": 1.0,
                "source_distance_p99_m": 0.0,
                "candidate_distance_p99_m": 0.0,
                "source_face_count": len(source_mesh.triangles),
                "candidate_face_count": len(candidate_mesh.triangles),
                "material_subset_count_preserved": (
                    source_mesh.material_subset_count == candidate_mesh.material_subset_count
                ),
                "material_binding_count_preserved": (
                    source_mesh.material_binding_count == candidate_mesh.material_binding_count
                ),
                "evidence_mode": "exact_world_geometry",
            }
        )
    return records, failed


def _part_surface_samples(mesh: MeshData, limit: int) -> np.ndarray:
    triangles = np.asarray(mesh.triangles, dtype=np.int64).reshape((-1, 3))
    valid = np.all((triangles >= 0) & (triangles < len(mesh.world_vertices_m)), axis=1)
    coordinates = mesh.world_vertices_m[triangles[valid]]
    if not len(coordinates):
        return np.empty((0, 3), dtype=np.float64)
    cross = np.cross(coordinates[:, 1] - coordinates[:, 0], coordinates[:, 2] - coordinates[:, 0])
    coordinates = coordinates[np.linalg.norm(cross, axis=1) > 1e-18]
    if not len(coordinates):
        return np.empty((0, 3), dtype=np.float64)
    if len(coordinates) > limit:
        indices = np.linspace(0, len(coordinates) - 1, limit, dtype=np.int64)
        coordinates = coordinates[indices]
    return np.concatenate((coordinates.mean(axis=1), coordinates.reshape((-1, 3))), axis=0)


def _part_correspondence(
    source: Path,
    candidate: Path,
    *,
    tolerance_m: float,
    sample_limit: int,
    allow_generated_candidate_faces: bool = False,
    exact_world_geometry_match: bool = False,
) -> tuple[list[dict[str, object]], int]:
    """Measure source/candidate coverage independently for every semantic mesh path."""

    source_meshes, _ = load_meshes(source, include_guide_purpose=True)
    candidate_meshes, _ = load_meshes(candidate, include_guide_purpose=True)
    source_by_path = _relative_meshes(source_meshes)
    candidate_by_path = _relative_meshes(candidate_meshes)
    if source_by_path is None or candidate_by_path is None:
        return [], max(len(source_meshes), len(candidate_meshes), 1)
    if exact_world_geometry_match:
        return _exact_part_correspondence(source_by_path, candidate_by_path)
    records: list[dict[str, object]] = []
    ambiguous = 0
    all_keys = sorted(source_by_path.keys() | candidate_by_path.keys())
    per_part_limit = max(16, sample_limit // max(len(all_keys), 1) // 4)
    for key in all_keys:
        source_mesh = source_by_path.get(key)
        candidate_mesh = candidate_by_path.get(key)
        if source_mesh is None or candidate_mesh is None:
            records.append(
                {
                    "relative_path": key,
                    "source_path": source_mesh.path if source_mesh else None,
                    "candidate_path": candidate_mesh.path if candidate_mesh else None,
                    "status": "missing",
                }
            )
            ambiguous += 1
            continue
        source_semantics_preserved = (
            source_mesh.material_subset_count == candidate_mesh.material_subset_count
            and source_mesh.material_binding_count == candidate_mesh.material_binding_count
        )
        source_points = _part_surface_samples(source_mesh, per_part_limit)
        candidate_points = _part_surface_samples(candidate_mesh, per_part_limit)
        try:
            source_surface = trimesh.Trimesh(
                vertices=source_mesh.world_vertices_m,
                faces=source_mesh.triangles,
                process=False,
            )
            candidate_surface = trimesh.Trimesh(
                vertices=candidate_mesh.world_vertices_m,
                faces=candidate_mesh.triangles,
                process=False,
            )
            _nearest, source_distances, _face_ids = trimesh.proximity.closest_point(
                candidate_surface,
                source_points,
            )
            _nearest, candidate_distances, _face_ids = trimesh.proximity.closest_point(
                source_surface,
                candidate_points,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            records.append(
                {
                    "relative_path": key,
                    "source_path": source_mesh.path,
                    "candidate_path": candidate_mesh.path,
                    "status": "not_evaluated",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            ambiguous += 1
            continue
        source_coverage = _coverage_ratio(source_distances, tolerance_m)
        candidate_coverage = _coverage_ratio(candidate_distances, tolerance_m)
        status = (
            "pass"
            if source_coverage is not None
            and candidate_coverage is not None
            and source_coverage >= 0.99
            and (
                candidate_coverage >= 0.99
                or (
                    allow_generated_candidate_faces
                    and len(candidate_mesh.triangles) >= len(source_mesh.triangles)
                )
            )
            and source_semantics_preserved
            else "fail"
        )
        if status != "pass":
            ambiguous += 1
        records.append(
            {
                "relative_path": key,
                "source_path": source_mesh.path,
                "candidate_path": candidate_mesh.path,
                "status": status,
                "source_sample_count": len(source_points),
                "candidate_sample_count": len(candidate_points),
                "source_coverage_ratio": source_coverage,
                "candidate_coverage_ratio": candidate_coverage,
                "source_distance_p99_m": (
                    float(np.percentile(source_distances, 99.0)) if len(source_distances) else None
                ),
                "candidate_distance_p99_m": (
                    float(np.percentile(candidate_distances, 99.0))
                    if len(candidate_distances)
                    else None
                ),
                "source_face_count": len(source_mesh.triangles),
                "candidate_face_count": len(candidate_mesh.triangles),
                "material_subset_count_preserved": (
                    source_mesh.material_subset_count == candidate_mesh.material_subset_count
                ),
                "material_binding_count_preserved": (
                    source_mesh.material_binding_count == candidate_mesh.material_binding_count
                ),
            }
        )
    return records, ambiguous


def compare_explicit_mesh_part_pairs(
    source_path: str | Path,
    candidate_path: str | Path,
    *,
    path_pairs: list[tuple[str, str]],
    tolerance_m: float,
    sample_limit: int,
) -> tuple[list[dict[str, object]], int]:
    """Validate caller-proven part identities by independent surface coverage."""

    source_meshes, _ = load_meshes(source_path, include_guide_purpose=True)
    candidate_meshes, _ = load_meshes(candidate_path, include_guide_purpose=True)
    source_by_path = {mesh.path: mesh for mesh in source_meshes}
    candidate_by_path = {mesh.path: mesh for mesh in candidate_meshes}
    source_pair_paths = [source for source, _candidate in path_pairs]
    candidate_pair_paths = [candidate for _source, candidate in path_pairs]
    if (
        len(set(source_pair_paths)) != len(source_pair_paths)
        or len(set(candidate_pair_paths)) != len(candidate_pair_paths)
        or set(source_pair_paths) != source_by_path.keys()
        or set(candidate_pair_paths) != candidate_by_path.keys()
    ):
        return [], max(len(source_meshes), len(candidate_meshes), 1)

    records: list[dict[str, object]] = []
    failed = 0
    per_part_limit = max(16, sample_limit // max(len(path_pairs), 1) // 4)
    for source_path_value, candidate_path_value in sorted(path_pairs):
        source_mesh = source_by_path[source_path_value]
        candidate_mesh = candidate_by_path[candidate_path_value]
        source_points = _part_surface_samples(source_mesh, per_part_limit)
        candidate_points = _part_surface_samples(candidate_mesh, per_part_limit)
        try:
            source_surface = trimesh.Trimesh(
                vertices=source_mesh.world_vertices_m,
                faces=source_mesh.triangles,
                process=False,
            )
            candidate_surface = trimesh.Trimesh(
                vertices=candidate_mesh.world_vertices_m,
                faces=candidate_mesh.triangles,
                process=False,
            )
            _nearest, source_distances, _face_ids = trimesh.proximity.closest_point(
                candidate_surface,
                source_points,
            )
            _nearest, candidate_distances, _face_ids = trimesh.proximity.closest_point(
                source_surface,
                candidate_points,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            records.append(
                {
                    "source_path": source_path_value,
                    "candidate_path": candidate_path_value,
                    "status": "not_evaluated",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "mapping_authority": "caller_proven_identity",
                }
            )
            failed += 1
            continue
        source_coverage = _coverage_ratio(source_distances, tolerance_m)
        candidate_coverage = _coverage_ratio(candidate_distances, tolerance_m)
        material_counts_preserved = (
            source_mesh.material_subset_count == candidate_mesh.material_subset_count
            and source_mesh.material_binding_count == candidate_mesh.material_binding_count
        )
        status = (
            "pass"
            if source_coverage is not None
            and candidate_coverage is not None
            and source_coverage >= 0.99
            and candidate_coverage >= 0.99
            and material_counts_preserved
            else "fail"
        )
        failed += status != "pass"
        records.append(
            {
                "source_path": source_path_value,
                "candidate_path": candidate_path_value,
                "status": status,
                "source_sample_count": len(source_points),
                "candidate_sample_count": len(candidate_points),
                "source_coverage_ratio": source_coverage,
                "candidate_coverage_ratio": candidate_coverage,
                "source_distance_p99_m": (
                    float(np.percentile(source_distances, 99.0)) if len(source_distances) else None
                ),
                "candidate_distance_p99_m": (
                    float(np.percentile(candidate_distances, 99.0))
                    if len(candidate_distances)
                    else None
                ),
                "source_face_count": len(source_mesh.triangles),
                "candidate_face_count": len(candidate_mesh.triangles),
                "material_subset_count_preserved": (
                    source_mesh.material_subset_count == candidate_mesh.material_subset_count
                ),
                "material_binding_count_preserved": (
                    source_mesh.material_binding_count == candidate_mesh.material_binding_count
                ),
                "mapping_authority": "caller_proven_identity",
            }
        )
    return records, failed


@lru_cache(maxsize=8)
def _cached_measure_asset(path_string: str, size_bytes: int, mtime_ns: int):
    del size_bytes, mtime_ns
    return measure_asset(path_string)


def _measure_asset(path: Path):
    stat = path.stat()
    return _cached_measure_asset(str(path), stat.st_size, stat.st_mtime_ns)


def _relative_meshes(meshes: list[MeshData]) -> dict[str, MeshData] | None:
    """Key mesh arrays independently of package-only USD wrapper prims."""

    if not meshes:
        return {}
    relative = _canonical_relative_paths([mesh.path for mesh in meshes])
    keyed: dict[str, MeshData] = {}
    for mesh in meshes:
        key = relative[mesh.path]
        if key in keyed:
            return None
        keyed[key] = mesh
    return keyed


def _mesh_semantics_match(source: MeshData, candidate: MeshData) -> bool:
    return bool(
        source.role == candidate.role
        and source.material_subset_count == candidate.material_subset_count
        and source.material_binding_count == candidate.material_binding_count
        and source.has_face_varying_data == candidate.has_face_varying_data
        and source.authored_normal_count == candidate.authored_normal_count
        and source.non_finite_normal_count == candidate.non_finite_normal_count
        and source.authored_uv_count == candidate.authored_uv_count
        and source.non_finite_uv_count == candidate.non_finite_uv_count
    )


def _canonical_surface_triangles(
    mesh: MeshData,
    *,
    origin: np.ndarray,
    tolerance: float,
    area_tolerance_twice: float,
) -> np.ndarray:
    triangles = mesh.triangles
    if not len(triangles):
        return np.empty((0, 9), dtype=np.int64)
    valid_indices = np.all((triangles >= 0) & (triangles < len(mesh.world_vertices_m)), axis=1)
    coordinates = mesh.world_vertices_m[triangles[valid_indices]]
    if not len(coordinates) or not np.isfinite(coordinates).all():
        return np.empty((0, 9), dtype=np.int64)
    cross = np.cross(
        coordinates[:, 1] - coordinates[:, 0],
        coordinates[:, 2] - coordinates[:, 0],
    )
    areas_twice = np.linalg.norm(cross, axis=1)
    coordinates = coordinates[areas_twice > area_tolerance_twice]
    if not len(coordinates):
        return np.empty((0, 9), dtype=np.int64)
    quantized = np.rint((coordinates - origin) / tolerance).astype(np.int64)
    order = np.lexsort(
        (quantized[:, :, 2], quantized[:, :, 1], quantized[:, :, 0]),
        axis=1,
    )
    canonical = np.take_along_axis(quantized, order[:, :, None], axis=1).reshape((-1, 9))
    return np.unique(canonical, axis=0)


def _canonical_index_triangles(
    mesh: MeshData,
    *,
    area_tolerance_twice: float,
) -> np.ndarray:
    triangles = mesh.triangles
    if not len(triangles):
        return np.empty((0, 3), dtype=np.int64)
    valid_indices = np.all((triangles >= 0) & (triangles < len(mesh.world_vertices_m)), axis=1)
    triangles = triangles[valid_indices]
    coordinates = mesh.world_vertices_m[triangles]
    if not len(coordinates) or not np.isfinite(coordinates).all():
        return np.empty((0, 3), dtype=np.int64)
    cross = np.cross(
        coordinates[:, 1] - coordinates[:, 0],
        coordinates[:, 2] - coordinates[:, 0],
    )
    triangles = triangles[np.linalg.norm(cross, axis=1) > area_tolerance_twice]
    return np.unique(np.sort(triangles, axis=1), axis=0)


def _missing_index_surfaces_match(
    source: MeshData,
    candidate: MeshData,
    *,
    source_indices: np.ndarray,
    candidate_indices: np.ndarray,
    tolerance: float,
    area_tolerance_twice: float,
) -> bool:
    """Match index-different triangles by unordered world-space vertices."""

    from scipy.spatial import cKDTree

    source_set = {tuple(int(value) for value in row) for row in source_indices}
    candidate_set = {tuple(int(value) for value in row) for row in candidate_indices}

    def contains_missing(
        missing: set[tuple[int, int, int]],
        query_mesh: MeshData,
        target_mesh: MeshData,
    ) -> bool:
        if not missing:
            return True
        target_triangles = target_mesh.triangles
        valid = np.all(
            (target_triangles >= 0) & (target_triangles < len(target_mesh.world_vertices_m)),
            axis=1,
        )
        target_coordinates = target_mesh.world_vertices_m[target_triangles[valid]]
        if not len(target_coordinates) or not np.isfinite(target_coordinates).all():
            return False
        cross = np.cross(
            target_coordinates[:, 1] - target_coordinates[:, 0],
            target_coordinates[:, 2] - target_coordinates[:, 0],
        )
        target_coordinates = target_coordinates[
            np.linalg.norm(cross, axis=1) > area_tolerance_twice
        ]
        if not len(target_coordinates):
            return False
        tree = cKDTree(target_coordinates.mean(axis=1))
        for triangle in missing:
            query = query_mesh.world_vertices_m[np.asarray(triangle, dtype=np.int64)]
            candidates = tree.query_ball_point(query.mean(axis=0), r=tolerance * 1.01)
            if not candidates:
                return False
            matched = False
            for index in candidates:
                target = target_coordinates[int(index)]
                distances = np.linalg.norm(query[:, None, :] - target[None, :, :], axis=2)
                if bool(
                    np.all(np.min(distances, axis=0) <= tolerance)
                    and np.all(np.min(distances, axis=1) <= tolerance)
                ):
                    matched = True
                    break
            if not matched:
                return False
        return True

    return contains_missing(
        source_set - candidate_set,
        source,
        candidate,
    ) and contains_missing(
        candidate_set - source_set,
        candidate,
        source,
    )


def _world_geometry_evidence(
    source: Path,
    candidate: Path,
) -> tuple[
    bool,
    bool,
    dict[str, MeshData] | None,
    dict[str, MeshData] | None,
]:
    """Prove exact array or exact nondegenerate surface preservation.

    Ordered-array identity handles packaging-only edits. Surface-set identity also
    permits deletion of exact duplicate/zero-area faces and winding correction, but
    still requires every semantic part and nondegenerate world triangle to match.
    """

    try:
        source_meshes, _ = load_meshes(source, include_guide_purpose=True)
        candidate_meshes, _ = load_meshes(candidate, include_guide_purpose=True)
    except (OSError, RuntimeError, ValueError):
        return False, False, None, None
    source_by_path = _relative_meshes(source_meshes)
    candidate_by_path = _relative_meshes(candidate_meshes)
    if source_by_path is None or candidate_by_path is None:
        return False, False, None, None
    if source_by_path.keys() != candidate_by_path.keys():
        return False, False, source_by_path, candidate_by_path
    all_vertices = [
        mesh.world_vertices_m
        for mesh in (*source_meshes, *candidate_meshes)
        if len(mesh.world_vertices_m)
    ]
    diagonal = 0.0
    if all_vertices:
        vertices = np.concatenate(all_vertices, axis=0)
        if not np.isfinite(vertices).all():
            return False, False, source_by_path, candidate_by_path
        diagonal = float(np.linalg.norm(np.ptp(vertices, axis=0)))
    # USD points commonly use float32. Treat its round-trip precision as exact
    # packaging preservation; this remains 10,000x tighter than the conservative
    # repair drift band.
    absolute_tolerance = max(diagonal * 1e-7, 1e-12)
    surface_tolerance = max(diagonal * 1e-6, 1e-12)
    area_tolerance_twice = max(diagonal * diagonal * 2e-16, 2e-24)
    origin = (
        np.min(np.concatenate(all_vertices, axis=0), axis=0)
        if all_vertices
        else np.zeros(3, dtype=np.float64)
    )
    exact_arrays = True
    for key, source_mesh in source_by_path.items():
        candidate_mesh = candidate_by_path[key]
        if not _mesh_semantics_match(source_mesh, candidate_mesh):
            return False, False, source_by_path, candidate_by_path
        vertices_aligned = bool(
            source_mesh.world_vertices_m.shape == candidate_mesh.world_vertices_m.shape
            and np.allclose(
                source_mesh.world_vertices_m,
                candidate_mesh.world_vertices_m,
                rtol=0.0,
                atol=absolute_tolerance,
            )
        )
        arrays_match = bool(
            vertices_aligned
            and source_mesh.triangles.shape == candidate_mesh.triangles.shape
            and np.array_equal(source_mesh.triangles, candidate_mesh.triangles)
        )
        exact_arrays = exact_arrays and arrays_match
        if arrays_match:
            continue
        if vertices_aligned:
            source_indices = _canonical_index_triangles(
                source_mesh,
                area_tolerance_twice=area_tolerance_twice,
            )
            candidate_indices = _canonical_index_triangles(
                candidate_mesh,
                area_tolerance_twice=area_tolerance_twice,
            )
            if source_indices.shape == candidate_indices.shape and np.array_equal(
                source_indices, candidate_indices
            ):
                continue
            if _missing_index_surfaces_match(
                source_mesh,
                candidate_mesh,
                source_indices=source_indices,
                candidate_indices=candidate_indices,
                tolerance=absolute_tolerance,
                area_tolerance_twice=area_tolerance_twice,
            ):
                continue
        source_surface = _canonical_surface_triangles(
            source_mesh,
            origin=origin,
            tolerance=surface_tolerance,
            area_tolerance_twice=area_tolerance_twice,
        )
        candidate_surface = _canonical_surface_triangles(
            candidate_mesh,
            origin=origin,
            tolerance=surface_tolerance,
            area_tolerance_twice=area_tolerance_twice,
        )
        if source_surface.shape != candidate_surface.shape or not np.array_equal(
            source_surface, candidate_surface
        ):
            return False, False, source_by_path, candidate_by_path
    return exact_arrays, True, source_by_path, candidate_by_path


def _world_geometry_relation(source: Path, candidate: Path) -> tuple[bool, bool]:
    """Compatibility wrapper returning only exact-array and exact-surface status."""

    exact_arrays, exact_surface, _source_meshes, _candidate_meshes = _world_geometry_evidence(
        source, candidate
    )
    return exact_arrays, exact_surface


def _exact_world_geometry_match(source: Path, candidate: Path) -> bool:
    """Compatibility wrapper for exact ordered-array geometry checks."""

    return _world_geometry_relation(source, candidate)[0]


def _exact_geometry_report(
    source: Path,
    candidate: Path,
    *,
    drift_band: DriftBand,
    budgets: RepairBudgets,
    protected_features: list[ProtectedFeature],
    output_path: str | Path | None,
    source_by_path: dict[str, MeshData] | None = None,
    candidate_by_path: dict[str, MeshData] | None = None,
) -> FidelityReport:
    """Return complete evidence after exact world geometry and semantics are proved."""

    if source_by_path is not None and candidate_by_path is not None:
        part_correspondence, ambiguous_part_count = _exact_part_correspondence(
            source_by_path,
            candidate_by_path,
        )
    else:
        part_correspondence, ambiguous_part_count = _part_correspondence(
            source,
            candidate,
            tolerance_m=1.0e-12,
            sample_limit=budgets.sample_point_limit,
            exact_world_geometry_match=True,
        )
    failures = (
        [f"{ambiguous_part_count} exact semantic part mappings were not preserved"]
        if ambiguous_part_count
        else []
    )
    silhouettes = {
        "front": 1.0,
        "back": 1.0,
        "left": 1.0,
        "right": 1.0,
        "top": 1.0,
        "bottom": 1.0,
    }
    report = FidelityReport(
        source_path=str(source),
        candidate_path=str(candidate),
        drift_band=drift_band,
        status="fail" if failures else "pass",
        exact_world_geometry_match=True,
        exact_world_surface_match=True,
        surface_distance_median_m=0.0,
        surface_distance_p95_m=0.0,
        surface_distance_p99_m=0.0,
        surface_distance_max_m=0.0,
        normal_angle_median_deg=0.0,
        normal_angle_p95_deg=0.0,
        normal_angle_max_deg=0.0,
        p99_bbox_ratio=0.0,
        bbox_max_drift_m=0.0,
        bbox_max_drift_ratio=0.0,
        centroid_drift_ratio=0.0,
        oriented_bbox_extent_drift_ratio=0.0,
        surface_area_drift_ratio=0.0,
        volume_drift_ratio=0.0,
        genus_delta=0.0,
        silhouette_iou_by_view=silhouettes,
        silhouette_iou_min=1.0,
        source_face_coverage_ratio=1.0,
        candidate_face_coverage_ratio=1.0,
        material_mapping_coverage_ratio=1.0,
        part_mapping_coverage_ratio=1.0,
        part_correspondence=part_correspondence,
        ambiguous_part_mapping_count=ambiguous_part_count,
        correspondence_mode="identity",
        protected_features_passed=[feature.name for feature in protected_features],
        failures=failures,
        report_path=str(Path(output_path).resolve()) if output_path else None,
    )
    if output_path:
        atomic_write_json(output_path, report)
    return report


def compare_geometry(
    source_path: str | Path,
    candidate_path: str | Path,
    *,
    drift_band: DriftBand,
    budgets: RepairBudgets,
    protected_features: list[ProtectedFeature] | None = None,
    expected_issue_ids: list[str] | None = None,
    generated_patch_area_ratio_limit: float | None = None,
    output_path: str | Path | None = None,
) -> FidelityReport:
    """Measure candidate drift directly against the normalized source."""

    source = Path(source_path).expanduser().resolve()
    candidate = Path(candidate_path).expanduser().resolve()
    identity = file_sha256(source) == file_sha256(candidate)
    source_by_path: dict[str, MeshData] | None = None
    candidate_by_path: dict[str, MeshData] | None = None
    if identity:
        exact_world_geometry_match = True
        exact_world_surface_match = True
        try:
            source_meshes, _ = load_meshes(source, include_guide_purpose=True)
            source_by_path = _relative_meshes(source_meshes)
            candidate_by_path = source_by_path
        except (OSError, RuntimeError, ValueError):
            pass
    else:
        (
            exact_world_geometry_match,
            exact_world_surface_match,
            source_by_path,
            candidate_by_path,
        ) = _world_geometry_evidence(
            source,
            candidate,
        )
    if exact_world_geometry_match:
        return _exact_geometry_report(
            source,
            candidate,
            drift_band=drift_band,
            budgets=budgets,
            protected_features=list(protected_features or []),
            output_path=output_path,
            source_by_path=source_by_path,
            candidate_by_path=candidate_by_path,
        )
    source_metrics = _measure_asset(source)
    candidate_metrics = source_metrics if exact_world_geometry_match else _measure_asset(candidate)
    source_samples = (
        deterministic_surface_samples(source, budgets.sample_point_limit)
        if not exact_world_surface_match
        else None
    )
    candidate_samples = (
        deterministic_surface_samples(candidate, budgets.sample_point_limit)
        if not exact_world_surface_match
        else None
    )
    normal_angles = np.empty((0,), dtype=np.float64)
    oriented_normals_reliable = (
        source_metrics.mesh.inconsistent_orientation_edge_count == 0
        and candidate_metrics.mesh.inconsistent_orientation_edge_count == 0
        and source_metrics.mesh.inverted_shell_status == "pass"
        and candidate_metrics.mesh.inverted_shell_status == "pass"
    )
    if exact_world_surface_match:
        source_to_candidate = np.zeros(1, dtype=np.float64)
        candidate_to_source = np.zeros(1, dtype=np.float64)
        normal_angles = np.zeros(1, dtype=np.float64)
    else:
        assert source_samples is not None and candidate_samples is not None
        _nearest, source_to_candidate, source_target_normals = closest_surface(
            candidate,
            source_samples.points,
        )
        _nearest, candidate_to_source, candidate_target_normals = closest_surface(
            source,
            candidate_samples.points,
        )
    distances = np.concatenate((source_to_candidate, candidate_to_source))
    if (
        not exact_world_surface_match
        and source_samples is not None
        and candidate_samples is not None
        and len(source_samples.normals)
        and len(candidate_samples.normals)
    ):
        source_dots = np.clip(
            np.einsum("ij,ij->i", source_samples.normals, source_target_normals),
            -1.0,
            1.0,
        )
        candidate_dots = np.clip(
            np.einsum("ij,ij->i", candidate_samples.normals, candidate_target_normals),
            -1.0,
            1.0,
        )
        if not oriented_normals_reliable:
            source_dots = np.abs(source_dots)
            candidate_dots = np.abs(candidate_dots)
        normal_angles = np.degrees(np.arccos(np.concatenate((source_dots, candidate_dots))))
    diagonal = source_metrics.mesh.bbox_diagonal_m or 0.0
    if len(distances):
        median, p95, p99, maximum = [
            float(value) for value in np.percentile(distances, [50.0, 95.0, 99.0, 100.0])
        ]
        p99_ratio = p99 / diagonal if diagonal > 0.0 else None
    else:
        median = p95 = p99 = maximum = None
        p99_ratio = None
    if len(normal_angles):
        normal_median, normal_p95, normal_maximum = [
            float(value) for value in np.percentile(normal_angles, [50.0, 95.0, 100.0])
        ]
    else:
        normal_median = normal_p95 = normal_maximum = None

    source_min = source_metrics.mesh.bbox_min_m
    source_max = source_metrics.mesh.bbox_max_m
    candidate_min = candidate_metrics.mesh.bbox_min_m
    candidate_max = candidate_metrics.mesh.bbox_max_m
    bbox_drift = None
    centroid_drift_ratio = None
    if source_min and source_max and candidate_min and candidate_max:
        bbox_drift = float(
            np.max(
                np.abs(
                    np.asarray([source_min, source_max])
                    - np.asarray([candidate_min, candidate_max])
                )
            )
        )
        source_center = (np.asarray(source_min) + np.asarray(source_max)) * 0.5
        candidate_center = (np.asarray(candidate_min) + np.asarray(candidate_max)) * 0.5
        centroid_drift_ratio = (
            float(np.linalg.norm(candidate_center - source_center)) / diagonal
            if diagonal > 0.0
            else None
        )

    if exact_world_surface_match:
        obb_drift_ratio = 0.0
    else:
        source_obb = oriented_bbox_extents(unique_geometry_points(source))
        candidate_obb = oriented_bbox_extents(unique_geometry_points(candidate))
        obb_drift_ratio = None
        if source_obb is not None and candidate_obb is not None and diagonal > 0.0:
            obb_drift_ratio = float(np.max(np.abs(source_obb - candidate_obb)) / diagonal)

    area_drift = _relative_drift(
        source_metrics.mesh.surface_area_m2, candidate_metrics.mesh.surface_area_m2
    )
    volume_drift = _relative_drift(
        source_metrics.mesh.enclosed_volume_m3,
        candidate_metrics.mesh.enclosed_volume_m3,
    )
    part_delta = len(candidate_metrics.source_part_paths) - len(source_metrics.source_part_paths)
    component_delta = (
        candidate_metrics.mesh.connected_component_count
        - source_metrics.mesh.connected_component_count
    )
    boundary_loop_delta = (
        candidate_metrics.mesh.boundary_loop_count - source_metrics.mesh.boundary_loop_count
    )
    genus_delta = (
        candidate_metrics.mesh.genus - source_metrics.mesh.genus
        if source_metrics.mesh.genus is not None and candidate_metrics.mesh.genus is not None
        else None
    )
    material_delta = (
        candidate_metrics.mesh.material_subset_count - source_metrics.mesh.material_subset_count
    )
    material_binding_delta = (
        candidate_metrics.material_binding_count - source_metrics.material_binding_count
    )
    source_parts = _relative_part_paths(source_metrics)
    candidate_parts = _relative_part_paths(candidate_metrics)
    missing_parts = sorted(source_parts - candidate_parts)
    added_parts = sorted(candidate_parts - source_parts)
    failures: list[str] = []
    warnings: list[str] = []
    if not oriented_normals_reliable and not exact_world_surface_match:
        warnings.append(
            "Normal-angle drift used unoriented axes because source or candidate winding "
            "is not reliable; topology diagnosis remains authoritative."
        )
    p99_limit, volume_limit = _limits(drift_band, budgets)
    expected = set(expected_issue_ids or [])
    generated_boundary_patch = (
        "mesh:boundary_edges" in expected
        and generated_patch_area_ratio_limit is not None
        and area_drift is not None
        and area_drift <= generated_patch_area_ratio_limit
    )
    winding_repair = bool(
        expected.intersection({"mesh:inconsistent_orientation", "mesh:inverted_shells"})
    )
    if p99_ratio is None:
        failures.append("surface drift could not be measured")
    elif p99_ratio > p99_limit and not generated_boundary_patch:
        failures.append(f"p99 surface drift ratio {p99_ratio:.6g} exceeds {p99_limit:.6g}")
    elif p99_ratio > p99_limit:
        warnings.append(
            "symmetric surface drift includes an explicitly classified generated boundary patch"
        )
    if volume_drift is not None and volume_drift > volume_limit and not winding_repair:
        failures.append(f"volume drift ratio {volume_drift:.6g} exceeds {volume_limit:.6g}")
    elif volume_drift is None and not exact_world_surface_match:
        warnings.append(
            "enclosed-volume drift could not be measured because source or candidate "
            "volume is unavailable"
        )
    if area_drift is not None and area_drift > max(volume_limit, p99_limit * 10.0):
        if generated_boundary_patch or expected.intersection(
            {"mesh:degenerate_faces", "mesh:duplicate_faces"}
        ):
            pass
        else:
            failures.append("surface-area drift exceeds the selected drift band")
    elif area_drift is None and not exact_world_surface_match:
        warnings.append(
            "surface-area drift could not be measured because source or candidate "
            "area is unavailable"
        )
    if part_delta:
        failures.append(f"source part count changed by {part_delta}")
    if missing_parts or added_parts:
        failures.append("source-relative part paths changed")
    reconstructive_topology = drift_band == "reconstructive" and bool(
        expected.intersection(
            {
                "mesh:boundary_edges",
                "mesh:coplanar_overlaps",
                "mesh:coplanar_overlaps_not_evaluated",
                "mesh:non_manifold_vertices",
                "mesh:over_connected_edges",
                "mesh:self_intersections",
                "mesh:self_intersections_not_evaluated",
            }
        )
    )
    if component_delta and not expected.intersection(
        {"mesh:degenerate_faces", "mesh:duplicate_faces"}
    ):
        (warnings if reconstructive_topology else failures).append(
            f"connected-component count changed by {component_delta}"
        )
    if material_delta:
        failures.append(f"material subset count changed by {material_delta}")
    if material_binding_delta:
        failures.append(f"material binding count changed by {material_binding_delta}")
    expected_topology_change = expected.intersection(
        {
            "mesh:boundary_edges",
            "mesh:degenerate_faces",
            "mesh:duplicate_faces",
            "mesh:inconsistent_orientation",
        }
    )
    if boundary_loop_delta:
        message = f"boundary-loop count changed by {boundary_loop_delta}"
        if generated_boundary_patch and boundary_loop_delta < 0:
            warnings.append(message)
        elif "mesh:boundary_edges" in expected_topology_change:
            (warnings if boundary_loop_delta < 0 else failures).append(message)
        elif not expected_topology_change:
            (warnings if reconstructive_topology else failures).append(message)
    if genus_delta is not None and abs(genus_delta) > 1e-9:
        (warnings if reconstructive_topology else failures).append(
            f"mesh genus changed by {genus_delta:g}"
        )
    if obb_drift_ratio is not None and obb_drift_ratio > max(p99_limit * 2.0, 1e-6):
        failures.append("oriented-bbox extent drift exceeds the selected drift band")
    normal_limit = (
        10.0
        if drift_band in {"identity", "conservative"}
        else 20.0
        if drift_band == "moderate"
        else 35.0
    )
    if (
        normal_p95 is not None
        and normal_p95 > normal_limit
        and not winding_repair
        and not generated_boundary_patch
    ):
        failures.append(
            f"p95 normal-angle drift {normal_p95:.3f} degrees exceeds {normal_limit:.3f}"
        )

    silhouettes = (
        {
            "front": 1.0,
            "back": 1.0,
            "left": 1.0,
            "right": 1.0,
            "top": 1.0,
            "bottom": 1.0,
        }
        if exact_world_surface_match
        else silhouette_iou(source, candidate)
    )
    silhouette_min = min(silhouettes.values(), default=None)
    silhouette_limit = (
        0.995
        if drift_band in {"identity", "conservative"}
        else 0.98
        if drift_band == "moderate"
        else 0.95
    )
    if silhouette_min is not None and silhouette_min < silhouette_limit:
        failures.append(
            f"minimum silhouette IoU {silhouette_min:.6f} is below {silhouette_limit:.6f}"
        )
    elif drift_band == "reconstructive" and silhouette_min is not None and silhouette_min < 0.98:
        warnings.append(
            f"minimum silhouette IoU {silhouette_min:.6f} is below the moderate 0.980000 "
            "band and requires explicit reconstructive visual review"
        )
    distance_tolerance = max(diagonal * p99_limit, 1e-9)
    source_face_coverage = (
        1.0
        if exact_world_surface_match
        else _coverage_ratio(source_to_candidate, distance_tolerance)
    )
    candidate_face_coverage = (
        1.0
        if exact_world_surface_match
        else _coverage_ratio(candidate_to_source, distance_tolerance)
    )
    if source_face_coverage is not None and source_face_coverage < 0.99:
        failures.append("source surface coverage fell below 99 percent")
    if (
        candidate_face_coverage is not None
        and candidate_face_coverage < 0.99
        and not generated_boundary_patch
    ):
        failures.append("candidate surface coverage fell below 99 percent")

    if exact_world_geometry_match:
        correspondence_mode = "identity"
    elif exact_world_surface_match:
        correspondence_mode = "exact_surface"
    else:
        correspondence_mode = "per_part_surface_projection"
    part_correspondence, ambiguous_part_count = _part_correspondence(
        source,
        candidate,
        tolerance_m=distance_tolerance,
        sample_limit=budgets.sample_point_limit,
        allow_generated_candidate_faces=generated_boundary_patch,
        exact_world_geometry_match=exact_world_geometry_match,
    )
    if ambiguous_part_count:
        failures.append(
            f"{ambiguous_part_count} semantic part mappings failed independent surface coverage"
        )

    passed_features: list[str] = []
    unmeasured_features: list[str] = []
    candidate_paths = set(candidate_metrics.source_part_paths)
    candidate_relative_paths = _relative_part_paths(candidate_metrics)
    source_root = str(source_metrics.default_prim_path or "").rstrip("/")
    probe_results = evaluate_feature_probes(candidate, protected_features or [])
    source_probe_results = {
        result.feature_name: result
        for result in evaluate_feature_probes(source, protected_features or [])
    }
    candidate_probe_results = {result.feature_name: result for result in probe_results}
    for feature in protected_features or []:
        candidate_probe = candidate_probe_results[feature.name]
        source_probe = source_probe_results[feature.name]
        if exact_world_geometry_match:
            passed_features.append(feature.name)
        elif (
            feature.probe is not None
            and candidate_probe.status == "pass"
            and source_probe.status != "fail"
        ):
            passed_features.append(feature.name)
        elif feature.probe is not None and candidate_probe.status == "fail":
            messages = [
                f"protected feature {feature.name!r}: {failure}"
                for failure in candidate_probe.failures
            ]
            (failures if feature.required else warnings).extend(messages)
        elif feature.probe is not None and source_probe.status == "fail":
            failures.append(
                f"protected feature {feature.name!r} probe does not pass on the normalized source"
            )
        elif (
            feature.kind == "part"
            and feature.minimum_size_m is None
            and feature.minimum_clearance_m is None
            and feature.tolerance_m is None
            and feature.scope_path
            and (
                feature.scope_path in candidate_paths
                or (
                    source_root
                    and feature.scope_path.startswith(f"{source_root}/")
                    and feature.scope_path[len(source_root) :] in candidate_relative_paths
                )
            )
        ):
            passed_features.append(feature.name)
        else:
            unmeasured_features.append(feature.name)
            if feature.required:
                warnings.append(
                    f"protected feature {feature.name!r} was not deterministically measured"
                )
    if drift_band == "reconstructive" and not failures:
        warnings.append(
            "Reconstructive geometry passed measured drift and correspondence gates but requires "
            "explicit human or ground-truth acceptance before production certification."
        )
    status = "fail" if failures else "conditional" if warnings else "pass"
    report = FidelityReport(
        source_path=str(source),
        candidate_path=str(candidate),
        drift_band=drift_band,
        status=status,
        exact_world_geometry_match=exact_world_geometry_match,
        exact_world_surface_match=exact_world_surface_match,
        sample_count_source=0 if source_samples is None else len(source_samples.points),
        sample_count_candidate=0 if candidate_samples is None else len(candidate_samples.points),
        surface_distance_median_m=median,
        surface_distance_p95_m=p95,
        surface_distance_p99_m=p99,
        surface_distance_max_m=maximum,
        normal_angle_median_deg=normal_median,
        normal_angle_p95_deg=normal_p95,
        normal_angle_max_deg=normal_maximum,
        p99_bbox_ratio=p99_ratio,
        bbox_max_drift_m=bbox_drift,
        bbox_max_drift_ratio=bbox_drift / diagonal if bbox_drift is not None and diagonal else None,
        centroid_drift_ratio=centroid_drift_ratio,
        oriented_bbox_extent_drift_ratio=obb_drift_ratio,
        surface_area_drift_ratio=area_drift,
        volume_drift_ratio=volume_drift,
        part_count_delta=part_delta,
        component_count_delta=component_delta,
        boundary_loop_count_delta=boundary_loop_delta,
        genus_delta=genus_delta,
        material_subset_count_delta=material_delta,
        material_binding_count_delta=material_binding_delta,
        missing_part_paths=missing_parts,
        added_part_paths=added_parts,
        silhouette_iou_by_view=silhouettes,
        silhouette_iou_min=silhouette_min,
        source_face_coverage_ratio=source_face_coverage,
        candidate_face_coverage_ratio=candidate_face_coverage,
        material_mapping_coverage_ratio=_mapping_coverage(
            source_metrics.mesh.material_subset_count + source_metrics.material_binding_count,
            candidate_metrics.mesh.material_subset_count + candidate_metrics.material_binding_count,
        ),
        part_mapping_coverage_ratio=(
            len(source_parts & candidate_parts) / len(source_parts) if source_parts else 1.0
        ),
        part_correspondence=part_correspondence,
        ambiguous_part_mapping_count=ambiguous_part_count,
        correspondence_mode=correspondence_mode,
        protected_features_passed=passed_features,
        protected_features_unmeasured=unmeasured_features,
        protected_feature_probes=probe_results,
        failures=failures,
        warnings=warnings,
        report_path=str(Path(output_path).resolve()) if output_path else None,
    )
    if output_path:
        atomic_write_json(output_path, report)
    return report
