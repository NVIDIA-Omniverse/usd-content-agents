# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic max-coverage and look-at-object camera placement."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np

from usd_core.camera_analysis.cancellation import check_cancelled
from usd_core.camera_analysis.contracts import (
    MAX_ANALYSIS_TRIANGLES,
    CameraPose,
    SceneAnalysisIR,
)
from usd_core.camera_analysis.coverage import (
    MAX_CAMERAS,
    MAX_COVERAGE_RAYS,
    GridSurface,
    frustum_mask,
    sample_surface,
    validate_grid_workload,
    validate_visibility_region_workload,
    validated_bounds,
    visibility_mask,
    visibility_regions,
)
from usd_core.camera_analysis.scene import is_path_at_or_below, transform_points
from usd_core.camera_analysis.warp_scoring import greedy_select_masks

MAX_CANDIDATES = 512
CAMERA_CLEARANCE_M = 0.05
TARGET_SURFACE_SAMPLES_PER_SHAPE = 32
_SURFACE_INSET_FRACTION = 1.0e-4
_MESH_AREA_CHUNK_TRIANGLES = 65_536
# `_mesh_surface_samples` makes two bounded area passes per distinct shared
# resource/surface metric. Permit work equivalent to one maximum-size legal mesh.
MAX_LOOK_AT_MESH_AREA_TRIANGLE_VISITS = MAX_ANALYSIS_TRIANGLES * 2


def validate_look_down_envelope(
    minimum_deg: float | None, maximum_deg: float | None
) -> tuple[float, float]:
    """Return a finite, ordered pitch envelope below the XY horizon."""

    minimum = 0.0 if minimum_deg is None else float(minimum_deg)
    maximum = 90.0 if maximum_deg is None else float(maximum_deg)
    if (
        not math.isfinite(minimum)
        or not math.isfinite(maximum)
        or minimum < 0.0
        or maximum > 90.0
        or maximum < minimum
    ):
        raise ValueError(
            "look-down envelope must be finite, ordered, and within 0..90 degrees"
        )
    return minimum, maximum


def validate_height_envelope(
    minimum_m: float | None, maximum_m: float | None
) -> tuple[float | None, float | None]:
    """Validate optional target-relative vertical-offset bounds."""

    minimum = None if minimum_m is None else float(minimum_m)
    maximum = None if maximum_m is None else float(maximum_m)
    if minimum is not None and not math.isfinite(minimum):
        raise ValueError("minimum target-relative camera height must be finite")
    if maximum is not None and not math.isfinite(maximum):
        raise ValueError("maximum target-relative camera height must be finite")
    if minimum is not None and maximum is not None and maximum < minimum:
        raise ValueError("maximum target-relative camera height must be >= the minimum")
    return minimum, maximum


def validate_xy_bounds(
    value: tuple[tuple[float, float], tuple[float, float]] | None,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Validate optional inclusive canonical-metre camera-position bounds."""

    if value is None:
        return None
    try:
        if len(value) != 2 or any(len(axis) != 2 for axis in value):
            raise ValueError
        converted = tuple((float(axis[0]), float(axis[1])) for axis in value)
    except (TypeError, ValueError, IndexError) as exc:
        raise ValueError(
            "XY bounds must contain ordered (min, max) pairs for X and Y"
        ) from exc
    for axis_name, (minimum, maximum) in zip(("X", "Y"), converted, strict=True):
        if (
            not math.isfinite(minimum)
            or not math.isfinite(maximum)
            or maximum < minimum
        ):
            raise ValueError(f"{axis_name} bounds must be finite and ordered")
    return converted  # type: ignore[return-value]


def _look_down_angle_deg(position_m: np.ndarray, target_m: np.ndarray) -> float:
    delta = np.asarray(position_m, dtype=np.float64) - np.asarray(
        target_m, dtype=np.float64
    )
    return math.degrees(math.atan2(float(delta[2]), float(np.linalg.norm(delta[:2]))))


def _geometry_clearance_mask(
    poses: list[CameraPose],
    backend,
    *,
    clearance_m: float = CAMERA_CLEARANCE_M,
) -> np.ndarray:
    """Reject near-surface and confidently inside poses through public backend rays.

    The six-axis probe is intentionally backend-only: it neither rebuilds geometry nor
    adds a second intersection implementation.  Opposed outward-facing hits on the
    same shape identify a point inside a consistently oriented closed surface.  If a
    backend cannot return normals, the conservative near-surface guard still applies.
    """

    if not poses:
        return np.zeros(0, dtype=bool)
    check_cancelled()
    positions = np.asarray([pose.position_m for pose in poses], dtype=np.float64)
    axes = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float64,
    )
    origins = np.repeat(positions, len(axes), axis=0)
    directions = np.tile(axes, (len(poses), 1))
    hits = backend.evaluate_rays(origins, directions, include_normals=True)
    check_cancelled()
    distances = np.asarray(hits.distances_m, dtype=np.float64).reshape(len(poses), 6)
    shape_ids = np.asarray(hits.shape_ids, dtype=np.int64).reshape(len(poses), 6)
    rejected = np.any(
        (distances >= 0.0) & (distances <= float(clearance_m) + 1.0e-12), axis=1
    )
    if hits.normals is not None:
        normals = np.asarray(hits.normals, dtype=np.float64).reshape(len(poses), 6, 3)
        alignment = np.einsum("nij,ij->ni", normals, axes)
        for first, second in ((0, 1), (2, 3), (4, 5)):
            rejected |= (
                (shape_ids[:, first] >= 0)
                & (shape_ids[:, first] == shape_ids[:, second])
                & (alignment[:, first] > 1.0e-6)
                & (alignment[:, second] > 1.0e-6)
            )
    return ~rejected


def validate_placement_workload(
    bounds_m: tuple[np.ndarray, np.ndarray],
    *,
    grid: int,
    patch_size_m: float | None,
    candidate_count: int,
) -> tuple[int, int]:
    """Bound floor, visibility, and clearance rays before any backend allocation."""

    rows, columns = validate_grid_workload(
        bounds_m,
        grid=grid,
        cell_size_m=patch_size_m,
        view_count=candidate_count,
        operation="candidate evaluation",
    )
    ray_budget = rows * columns * (candidate_count + 2) + 6 * candidate_count
    if ray_budget > MAX_COVERAGE_RAYS:
        raise ValueError(
            f"candidate evaluation would cast up to {ray_budget:,} rays including "
            f"clearance probes; limit is {MAX_COVERAGE_RAYS:,}"
        )
    return rows, columns


@dataclass(frozen=True)
class PlacementEvaluation:
    report: dict
    poses: tuple[CameraPose, ...]
    masks: np.ndarray | None = None


def _pose(
    position: np.ndarray,
    target: np.ndarray,
    *,
    focal_length_mm: float,
    aperture_mm: float,
) -> CameraPose:
    forward = np.asarray(target, dtype=np.float64) - np.asarray(
        position, dtype=np.float64
    )
    length = float(np.linalg.norm(forward))
    if length <= 1.0e-8:
        raise ValueError("camera position and look-at target must differ")
    forward /= length
    world_up = np.asarray([0.0, 0.0, 1.0])
    if abs(float(np.dot(forward, world_up))) > 0.999:
        world_up = np.asarray([0.0, 1.0, 0.0])
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)
    return CameraPose(
        prim_path=None,
        position_m=tuple(float(item) for item in position),
        right=tuple(float(item) for item in right),
        up=tuple(float(item) for item in up),
        forward=tuple(float(item) for item in forward),
        focal_length_mm=focal_length_mm,
        horizontal_aperture_mm=aperture_mm,
        vertical_aperture_mm=aperture_mm,
    )


def _stage_point(
    scene: SceneAnalysisIR, point_m: tuple[float, float, float]
) -> list[float]:
    return [
        float(item)
        for item in transform_points(np.asarray([point_m]), scene.canonical_to_stage)[0]
    ]


def _camera_record(
    scene: SceneAnalysisIR,
    pose: CameraPose,
    target_m: np.ndarray,
    *,
    index: int,
) -> dict:
    return {
        "id": f"camera-{index + 1:03d}",
        "position": _stage_point(scene, pose.position_m),
        "look_at": _stage_point(scene, tuple(float(item) for item in target_m)),
        "canonical_position_m": list(pose.position_m),
        "focal_length": pose.focal_length_mm,
        "horizontal_aperture": pose.horizontal_aperture_mm,
        "vertical_aperture": pose.vertical_aperture_mm,
    }


def _coverage_candidates(
    surface: GridSurface,
    *,
    candidate_count: int,
    height_m: float,
    standoff_m: float,
    seed: int,
    focal_length_mm: float,
    aperture_mm: float,
) -> tuple[list[CameraPose], np.ndarray]:
    points = surface.accessible_points_m
    target = np.nanmean(points, axis=0)
    floor_z = float(np.nanmedian(points[:, 2]))
    target[2] = floor_z
    expanded_bounds = (
        surface.x_min_m - standoff_m,
        surface.x_max_m + standoff_m,
        surface.y_min_m - standoff_m,
        surface.y_max_m + standoff_m,
    )
    phase = random.Random(seed).random() * 2.0 * math.pi
    poses: list[CameraPose] = []
    # Two deterministic rings provide both tighter and wider viewpoints while
    # preserving a stable tie order for the greedy selector.
    for index in range(candidate_count):
        check_cancelled()
        ring = index % 2
        ordinal = index // 2
        ring_count = (candidate_count + (1 - ring)) // 2
        angle = phase + 2.0 * math.pi * ordinal / max(1, ring_count)
        direction = np.asarray([math.cos(angle), math.sin(angle)])
        exit_distances = []
        if abs(float(direction[0])) > 1.0e-12:
            x_boundary = (
                expanded_bounds[1] if direction[0] > 0.0 else expanded_bounds[0]
            )
            exit_distances.append((x_boundary - target[0]) / direction[0])
        if abs(float(direction[1])) > 1.0e-12:
            y_boundary = (
                expanded_bounds[3] if direction[1] > 0.0 else expanded_bounds[2]
            )
            exit_distances.append((y_boundary - target[1]) / direction[1])
        positive_exits = [value for value in exit_distances if value >= 0.0]
        boundary_radius = min(positive_exits) if positive_exits else 0.0
        # Both rings remain strictly outside the standoff-expanded AABB. The
        # outer ring adds viewpoint diversity without weakening the clearance.
        radius = max(0.25, boundary_radius) * (1.02 if ring == 0 else 1.22)
        position = np.asarray(
            [
                target[0] + radius * direction[0],
                target[1] + radius * direction[1],
                floor_z + height_m,
            ]
        )
        poses.append(
            _pose(
                position,
                target,
                focal_length_mm=focal_length_mm,
                aperture_mm=aperture_mm,
            )
        )
    return poses, target


def place_max_coverage(
    scene: SceneAnalysisIR,
    backend,
    bounds_m: tuple[np.ndarray, np.ndarray],
    *,
    scope_path: str,
    target_coverage: float = 0.95,
    per_cell: int = 1,
    max_cameras: int = 8,
    grid: int = 32,
    cell_size_m: float | None = None,
    patch_size_m: float | None = None,
    candidate_count: int = 64,
    height_m: float = 3.0,
    standoff_m: float = 0.5,
    minimum_gain: float = 0.0,
    min_look_down_deg: float | None = None,
    max_look_down_deg: float | None = None,
    seed: int = 0,
    focal_length_mm: float = 18.0,
    aperture_mm: float = 36.0,
) -> PlacementEvaluation:
    check_cancelled()
    if not math.isfinite(target_coverage) or not 0.0 <= target_coverage <= 1.0:
        raise ValueError("target coverage must be between 0 and 1")
    if not 1 <= per_cell <= MAX_CAMERAS:
        raise ValueError(f"per-cell redundancy must be between 1 and {MAX_CAMERAS}")
    if not 1 <= max_cameras <= MAX_CAMERAS:
        raise ValueError(f"max cameras must be between 1 and {MAX_CAMERAS}")
    if not 4 <= candidate_count <= MAX_CANDIDATES:
        raise ValueError(f"candidate count must be between 4 and {MAX_CANDIDATES}")
    for value, label in ((height_m, "height"), (standoff_m, "standoff")):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{label} must be finite and nonnegative")
    if height_m <= 0.0:
        raise ValueError("height must be positive")
    if not math.isfinite(minimum_gain) or not 0.0 <= minimum_gain <= 1.0:
        raise ValueError("minimum gain must be between 0 and 1")
    if (
        not math.isfinite(focal_length_mm)
        or not math.isfinite(aperture_mm)
        or focal_length_mm <= 0.0
        or aperture_mm <= 0.0
    ):
        raise ValueError("focal length and aperture must be positive")
    if cell_size_m is not None and patch_size_m is not None:
        raise ValueError("cell size and patch size are aliases; provide only one")
    effective_patch_size = patch_size_m if patch_size_m is not None else cell_size_m
    if effective_patch_size is not None and (
        not math.isfinite(float(effective_patch_size))
        or float(effective_patch_size) <= 0.0
    ):
        raise ValueError("patch size must be finite and positive")
    look_down_envelope = validate_look_down_envelope(
        min_look_down_deg, max_look_down_deg
    )

    validate_placement_workload(
        bounds_m,
        grid=grid,
        patch_size_m=effective_patch_size,
        candidate_count=candidate_count,
    )
    check_cancelled()
    surface = sample_surface(
        scene,
        backend,
        bounds_m,
        scope_path=scope_path,
        grid=grid,
        cell_size_m=effective_patch_size,
    )
    raw_candidates, target = _coverage_candidates(
        surface,
        candidate_count=candidate_count,
        height_m=height_m,
        standoff_m=standoff_m,
        seed=seed,
        focal_length_mm=focal_length_mm,
        aperture_mm=aperture_mm,
    )
    envelope_candidates: list[CameraPose] = []
    envelope_indices: list[int] = []
    for index, candidate in enumerate(raw_candidates):
        check_cancelled()
        angle = _look_down_angle_deg(np.asarray(candidate.position_m), target)
        if look_down_envelope[0] - 1.0e-12 <= angle <= look_down_envelope[1] + 1.0e-12:
            envelope_candidates.append(candidate)
            envelope_indices.append(index)
    check_cancelled()
    geometry_keep = _geometry_clearance_mask(envelope_candidates, backend)
    candidates = [
        candidate
        for candidate, keep in zip(envelope_candidates, geometry_keep, strict=True)
        if bool(keep)
    ]
    candidate_indices = [
        index
        for index, keep in zip(envelope_indices, geometry_keep, strict=True)
        if bool(keep)
    ]
    accessible_count = int(surface.accessible.sum())
    if candidates:
        candidate_masks: list[np.ndarray] = []
        for candidate in candidates:
            check_cancelled()
            candidate_masks.append(visibility_mask(candidate, surface, backend))
        masks = np.stack(candidate_masks)
        check_cancelled()
        scores = greedy_select_masks(
            masks,
            surface.accessible,
            per_cell=per_cell,
            max_cameras=max_cameras,
            target_coverage=target_coverage,
            minimum_gain=minimum_gain,
            device=getattr(backend, "device", "cpu"),
        )
        selected = list(scores.selected)
        marginal_counts = list(scores.marginal_counts)
        accessible_count = scores.accessible_count
        covered_count = scores.covered_count
        visible_counts = scores.visible_counts
        stop_reason = scores.stop_reason
    else:
        masks = np.zeros((0, surface.rows, surface.columns), dtype=bool)
        selected = []
        marginal_counts = []
        covered_count = 0
        visible_counts = np.zeros(0, dtype=np.int64)
        stop_reason = "no_valid_candidate"

    selected_masks = (
        masks[selected]
        if selected
        else np.zeros((0, surface.rows, surface.columns), dtype=bool)
    )
    validate_visibility_region_workload(selected_masks, surface.accessible)
    ratio = float(covered_count / accessible_count)
    records = []
    for output_index, (candidate_index, marginal) in enumerate(
        zip(selected, marginal_counts, strict=False)
    ):
        check_cancelled()
        record = _camera_record(
            scene, candidates[candidate_index], target, index=output_index
        )
        mask = masks[candidate_index] & surface.accessible
        record.update(
            {
                "candidate_index": candidate_indices[candidate_index],
                "marginal_cells_at_selection": marginal,
                "visible_cells": int(visible_counts[candidate_index]),
                "visibility": {
                    "schema": "usd-cli.visibility-grid.v1",
                    "coordinate_space": "canonical_meter_z_up",
                    "encoding": "grid-boundary-v1",
                    "regions": visibility_regions(mask, surface),
                    "area_m2": float(
                        visible_counts[candidate_index] * surface.cell_area_m2
                    ),
                },
            }
        )
        records.append(record)

    report = {
        "schema": "usd-cli.camera-placement.v1",
        "method": "max_coverage",
        "source_digest": scene.source_digest,
        "analysis_policy": scene.policy.as_dict(),
        "analysis_policy_digest": scene.policy.digest,
        "scope_path": scope_path,
        "seed": seed,
        "target_coverage": target_coverage,
        "per_cell": per_cell,
        "achieved_coverage": ratio,
        "passed": ratio + 1.0e-12 >= target_coverage,
        "stop_reason": stop_reason,
        "accessible_cells": accessible_count,
        "covered_cells": covered_count,
        "uncovered_cells": accessible_count - covered_count,
        "candidate_count": candidate_count,
        "valid_candidate_count": len(candidates),
        "constraint_rejected_candidate_count": len(raw_candidates)
        - len(envelope_candidates),
        "geometry_rejected_candidate_count": len(envelope_candidates) - len(candidates),
        "selected_count": len(selected),
        "cameras": records,
        "grid": {
            "rows": surface.rows,
            "columns": surface.columns,
            "cell_width_m": surface.cell_width_m,
            "cell_height_m": surface.cell_height_m,
        },
        "config": {
            "height_m": height_m,
            "standoff_m": standoff_m,
            "minimum_gain": minimum_gain,
            "patch_size_m": effective_patch_size,
            "min_look_down_deg": look_down_envelope[0],
            "max_look_down_deg": look_down_envelope[1],
            "camera_clearance_m": CAMERA_CLEARANCE_M,
            "focal_length": focal_length_mm,
            "aperture": aperture_mm,
        },
        "backend": {"name": "newton_warp", **backend.versions},
    }
    return PlacementEvaluation(
        report=report,
        poses=tuple(candidates[index] for index in selected),
        masks=selected_masks,
    )


def parse_yaw_ranges(value: str | None) -> tuple[tuple[float, float], ...]:
    if value is None or not value.strip():
        return ((0.0, 360.0),)
    ranges: list[tuple[float, float]] = []
    for item in value.split(";"):
        parts = [part.strip() for part in item.split(",")]
        if len(parts) != 2:
            raise ValueError("yaw ranges must look like '0,180;270,360'")
        try:
            start, end = (float(part) for part in parts)
        except ValueError as exc:
            raise ValueError("yaw ranges must contain numeric degree pairs") from exc
        if not math.isfinite(start) or not math.isfinite(end):
            raise ValueError("yaw ranges must be finite")
        if end < start:
            raise ValueError(
                "each yaw range end must be greater than or equal to its start"
            )
        if end - start > 360.0 + 1.0e-9:
            raise ValueError("a yaw range cannot span more than 360 degrees")
        ranges.append((start, end))
    return tuple(ranges)


def _yaw_allowed(angle: float, ranges: tuple[tuple[float, float], ...]) -> bool:
    normalized = angle % 360.0
    for start, end in ranges:
        if end - start >= 360.0 - 1.0e-9:
            return True
        start_n, end_n = start % 360.0, end % 360.0
        if start_n <= end_n:
            if start_n - 1.0e-9 <= normalized <= end_n + 1.0e-9:
                return True
        elif normalized >= start_n - 1.0e-9 or normalized <= end_n + 1.0e-9:
            return True
    return False


def _unit_surface_directions(count: int) -> np.ndarray:
    """Return deterministic, near-uniform unit directions without random state."""

    ordinal = np.arange(count, dtype=np.float64)
    z = 1.0 - 2.0 * (ordinal + 0.5) / count
    azimuth = ordinal * (math.pi * (3.0 - math.sqrt(5.0)))
    radial = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack((radial * np.cos(azimuth), radial * np.sin(azimuth), z))


def _interior_barycentric(ordinal: int) -> np.ndarray:
    """Return a stable low-discrepancy point inset from all triangle edges."""

    first = ((ordinal + 0.5) * 0.7548776662466927) % 1.0
    second = ((ordinal + 0.5) * 0.5698402909980532) % 1.0
    root = math.sqrt(first)
    weights = np.asarray(
        [1.0 - root, root * (1.0 - second), root * second], dtype=np.float64
    )
    # A strictly positive barycentric margin avoids shared edges and vertices,
    # where a closest-hit implementation may make an arbitrary adjacent-face choice.
    return weights * 0.85 + 0.05


def _mesh_surface_samples(
    resource,
    count: int,
    *,
    prim_path: str,
    surface_linear: np.ndarray,
) -> np.ndarray:
    """Area-stratify mesh samples in canonical world space using bounded passes."""

    # A shared mesh resource can occur below a non-uniform scale or shear. Triangle
    # weights therefore have to be measured after the occurrence's linear transform;
    # local-space areas can otherwise differ arbitrarily from the surface seen by the
    # visibility backend. Retain the IR's packed dtypes and promote only bounded
    # local/transformed chunks, so this remains safe for multi-million-face resources.
    vertices = np.asarray(resource.vertices)
    triangles = np.asarray(resource.indices).reshape(-1, 3)
    linear = np.asarray(surface_linear, dtype=np.float64)
    if linear.shape != (3, 3) or not np.all(np.isfinite(linear)):
        raise ValueError(f"target mesh {prim_path} has an invalid surface transform")

    def transformed_areas(local_chunk: np.ndarray) -> np.ndarray:
        world_chunk = local_chunk @ linear
        return 0.5 * np.linalg.norm(
            np.cross(
                world_chunk[:, 1] - world_chunk[:, 0],
                world_chunk[:, 2] - world_chunk[:, 0],
            ),
            axis=1,
        )

    total_area = 0.0
    for start in range(0, len(triangles), _MESH_AREA_CHUNK_TRIANGLES):
        check_cancelled()
        chunk = np.asarray(
            vertices[triangles[start : start + _MESH_AREA_CHUNK_TRIANGLES]],
            dtype=np.float64,
        )
        areas = transformed_areas(chunk)
        total_area += float(areas.sum(dtype=np.float64))
    if not math.isfinite(total_area) or total_area <= 1.0e-18:
        raise ValueError(f"target mesh {prim_path} has no sampleable surface area")

    thresholds = (np.arange(count, dtype=np.float64) + 0.5) * total_area / count
    samples = np.empty((count, 3), dtype=np.float64)
    cumulative_area = 0.0
    sample_index = 0
    for start in range(0, len(triangles), _MESH_AREA_CHUNK_TRIANGLES):
        check_cancelled()
        triangle_indices = triangles[start : start + _MESH_AREA_CHUNK_TRIANGLES]
        chunk = np.asarray(vertices[triangle_indices], dtype=np.float64)
        areas = transformed_areas(chunk)
        area_prefix = np.cumsum(areas, dtype=np.float64)
        chunk_area = float(area_prefix[-1])
        chunk_end = cumulative_area + chunk_area
        while sample_index < count and thresholds[sample_index] <= chunk_end:
            relative = thresholds[sample_index] - cumulative_area
            triangle_index = min(
                int(np.searchsorted(area_prefix, relative, side="left")),
                len(chunk) - 1,
            )
            samples[sample_index] = (
                _interior_barycentric(sample_index) @ chunk[triangle_index]
            )
            sample_index += 1
        cumulative_area = chunk_end
        if sample_index == count:
            break
    if sample_index != count:  # pragma: no cover - guarded by the finite area sum
        raise ValueError(
            f"target mesh {prim_path} could not be sampled deterministically"
        )
    return samples


def _mesh_surface_cache_key(
    resource_id: str,
    linear: np.ndarray,
    *,
    prim_path: str,
) -> tuple[str, tuple[float, ...]]:
    """Key area-equivalent occurrences exactly as target sampling does."""

    surface_linear = np.asarray(linear, dtype=np.float64)
    if surface_linear.shape != (3, 3) or not np.all(np.isfinite(surface_linear)):
        raise ValueError(f"target mesh {prim_path} has an invalid surface transform")
    # Translation and world rotation do not change triangle areas. Key shared
    # occurrences by their local-space metric so rigidly rotated instances reuse
    # the expensive two-pass scan, while anisotropic and sheared occurrences retain
    # their own canonical-world weighting.
    surface_metric = surface_linear @ surface_linear.T
    metric_scale = float(np.max(np.abs(surface_metric), initial=0.0))
    normalized_metric = (
        surface_metric / metric_scale if metric_scale > 0.0 else surface_metric
    )
    metric_key = tuple(
        float(value) for value in np.round(normalized_metric, decimals=12).reshape(-1)
    )
    return resource_id, metric_key


def validate_look_at_workload(
    scene: SceneAnalysisIR,
    target_path: str,
    *,
    candidate_count: int,
) -> tuple:
    """Bound target rays and metric-weighted mesh sampling before allocation."""

    if (
        isinstance(candidate_count, bool)
        or not isinstance(candidate_count, int | np.integer)
        or candidate_count < 1
    ):
        raise ValueError("look-at candidate count must be a positive integer")
    target_shapes = tuple(
        shape
        for shape in scene.shapes
        if shape.analysis_role != "helper"
        and is_path_at_or_below(shape.prim_path, target_path)
    )
    if not target_shapes:
        raise ValueError(f"target {target_path} contains no supported visible geometry")
    target_sample_count = len(target_shapes) * TARGET_SURFACE_SAMPLES_PER_SHAPE
    if (target_sample_count + 6) * candidate_count > MAX_COVERAGE_RAYS:
        raise ValueError("look-at candidate ray budget exceeds the analysis limit")

    resources = {resource.resource_id: resource for resource in scene.meshes}
    seen_metrics: set[tuple[str, tuple[float, ...]]] = set()
    area_triangle_visits = 0
    for shape in target_shapes:
        check_cancelled()
        if shape.kind != "mesh":
            continue
        resource_id = shape.mesh_resource_id or ""
        resource = resources.get(resource_id)
        if resource is None:
            raise ValueError(
                f"target mesh {shape.prim_path} references a missing mesh resource"
            )
        linear = np.asarray(shape.transform, dtype=np.float64)[:3, :3]
        cache_key = _mesh_surface_cache_key(
            resource_id, linear, prim_path=shape.prim_path
        )
        if cache_key in seen_metrics:
            continue
        seen_metrics.add(cache_key)
        area_triangle_visits += 2 * (len(resource.indices) // 3)
        if area_triangle_visits > MAX_LOOK_AT_MESH_AREA_TRIANGLE_VISITS:
            raise ValueError(
                "look_at_surface_sampling_workload_limit: deterministic world-area "
                "target sampling requires more than "
                f"{MAX_LOOK_AT_MESH_AREA_TRIANGLE_VISITS:,} aggregate triangle "
                "visits across distinct resource/surface metrics (reduce varied-scale "
                "mesh instances or split the target)"
            )
    return target_shapes


def _analytic_surface_samples(shape, count: int) -> np.ndarray:
    """Sample one supported analytic GPrim in its canonical local space."""

    inset = 1.0 - _SURFACE_INSET_FRACTION
    directions = _unit_surface_directions(count)
    parameters = shape.parameters
    if shape.kind == "sphere":
        return directions * (float(parameters["radius"]) * inset)
    if shape.kind == "box":
        half_extents = np.asarray(
            [parameters["hx"], parameters["hy"], parameters["hz"]],
            dtype=np.float64,
        )
        distances = np.min(
            np.divide(
                half_extents,
                np.abs(directions),
                out=np.full_like(directions, np.inf),
                where=np.abs(directions) > 1.0e-15,
            ),
            axis=1,
        )
        return directions * distances[:, None] * inset
    if shape.kind == "capsule":
        radius = float(parameters["radius"])
        half_height = float(parameters["half_height"])
        # Cylinder area is 4*pi*r*h and the two caps total 4*pi*r^2. Allocate
        # the bounded samples proportionally, while guaranteeing both caps and
        # the barrel are represented even at extreme aspect ratios.
        barrel_count = int(round(count * half_height / (half_height + radius)))
        barrel_count = min(count - 2, max(1, barrel_count))
        cap_count = count - barrel_count
        points = np.empty((count, 3), dtype=np.float64)
        golden = math.pi * (3.0 - math.sqrt(5.0))
        for index in range(barrel_count):
            angle = index * golden
            height_fraction = ((index + 0.5) * 0.6180339887498949 % 1.0) * 2.0 - 1.0
            points[index] = (
                radius * inset * math.cos(angle),
                radius * inset * math.sin(angle),
                half_height * height_fraction,
            )
        cap_ordinals = [0, 0]
        for offset in range(cap_count):
            sign_index = offset % 2
            sign = -1.0 if sign_index == 0 else 1.0
            ordinal = cap_ordinals[sign_index]
            cap_ordinals[sign_index] += 1
            samples_on_cap = (cap_count + (1 - sign_index)) // 2
            axial = (ordinal + 0.5) / samples_on_cap
            radial = math.sqrt(max(0.0, 1.0 - axial * axial))
            angle = offset * golden
            points[barrel_count + offset] = (
                radius * inset * radial * math.cos(angle),
                radius * inset * radial * math.sin(angle),
                sign * (half_height + radius * inset * axial),
            )
        return points
    if shape.kind == "cylinder":
        radius = float(parameters["radius"])
        half_height = float(parameters["half_height"])
        points = np.empty((count, 3), dtype=np.float64)
        for index in range(count):
            angle = index * (math.pi * (3.0 - math.sqrt(5.0)))
            if index % 3:
                height_fraction = ((index * 0.6180339887498949) % 1.0) * 2.0 - 1.0
                points[index] = (
                    radius * inset * math.cos(angle),
                    radius * inset * math.sin(angle),
                    half_height * height_fraction,
                )
            else:
                disk_radius = radius * math.sqrt((index + 0.5) / count) * inset
                points[index] = (
                    disk_radius * math.cos(angle),
                    disk_radius * math.sin(angle),
                    (-half_height if (index // 3) % 2 else half_height) * inset,
                )
        return points
    if shape.kind == "cone":
        radius = float(parameters["radius"])
        half_height = float(parameters["half_height"])
        points = np.empty((count, 3), dtype=np.float64)
        for index in range(count):
            angle = index * (math.pi * (3.0 - math.sqrt(5.0)))
            if index % 4:
                apex_to_base = 0.1 + 0.8 * ((index * 0.6180339887498949) % 1.0)
                local_radius = radius * apex_to_base * inset
                points[index] = (
                    local_radius * math.cos(angle),
                    local_radius * math.sin(angle),
                    half_height * (1.0 - 2.0 * apex_to_base),
                )
            else:
                disk_radius = radius * math.sqrt((index + 0.5) / count) * inset
                points[index] = (
                    disk_radius * math.cos(angle),
                    disk_radius * math.sin(angle),
                    -half_height * inset,
                )
        return points
    if shape.kind == "plane":
        ordinal = np.arange(count, dtype=np.float64)
        x = ((ordinal + 0.5) / count * 2.0 - 1.0) * parameters["width"] * 0.5
        y = (
            (((ordinal + 0.5) * 0.6180339887498949 % 1.0) * 2.0 - 1.0)
            * parameters["length"]
            * 0.5
        )
        return np.column_stack((x * inset, y * inset, np.zeros(count)))
    raise ValueError(
        f"unsupported target surface kind {shape.kind!r} at {shape.prim_path}"
    )


def _target_surface_samples(scene: SceneAnalysisIR, target_shapes: tuple) -> np.ndarray:
    """Build deterministic world-space samples from exact target descendants."""

    resources = {resource.resource_id: resource for resource in scene.meshes}
    mesh_sample_cache: dict[tuple[str, tuple[float, ...]], np.ndarray] = {}
    samples: list[np.ndarray] = []
    for shape in target_shapes:
        check_cancelled()
        if shape.kind == "mesh":
            resource_id = shape.mesh_resource_id or ""
            resource = resources.get(resource_id)
            if resource is None:
                raise ValueError(
                    f"target mesh {shape.prim_path} references a missing mesh resource"
                )
            linear = np.asarray(shape.transform, dtype=np.float64)[:3, :3]
            cache_key = _mesh_surface_cache_key(
                resource_id, linear, prim_path=shape.prim_path
            )
            local_samples = mesh_sample_cache.get(cache_key)
            if local_samples is None:
                local_samples = _mesh_surface_samples(
                    resource,
                    TARGET_SURFACE_SAMPLES_PER_SHAPE,
                    prim_path=shape.prim_path,
                    surface_linear=linear,
                )
                mesh_sample_cache[cache_key] = local_samples
        else:
            local_samples = _analytic_surface_samples(
                shape, TARGET_SURFACE_SAMPLES_PER_SHAPE
            )
        samples.append(transform_points(local_samples, shape.transform))
    check_cancelled()
    return np.concatenate(samples, axis=0)


def place_cameras_look_at(
    scene: SceneAnalysisIR,
    backend,
    bounds_m: tuple[np.ndarray, np.ndarray],
    *,
    target_path: str,
    camera_count: int = 4,
    yaw_ranges: str | None = None,
    occlusion_threshold: float = 0.4,
    min_distance_m: float | None = None,
    max_distance_m: float | None = None,
    height_offset_m: float = 0.0,
    min_height_m: float | None = None,
    max_height_m: float | None = None,
    min_look_down_deg: float | None = None,
    max_look_down_deg: float | None = None,
    xy_bounds_m: tuple[tuple[float, float], tuple[float, float]] | None = None,
    candidate_count: int = 72,
    allow_fewer: bool = False,
    seed: int = 0,
    focal_length_mm: float = 35.0,
    aperture_mm: float = 36.0,
) -> PlacementEvaluation:
    check_cancelled()
    if not 1 <= camera_count <= MAX_CAMERAS:
        raise ValueError(f"camera count must be between 1 and {MAX_CAMERAS}")
    if not 4 <= candidate_count <= MAX_CANDIDATES:
        raise ValueError(f"candidate count must be between 4 and {MAX_CANDIDATES}")
    if not math.isfinite(occlusion_threshold) or not 0.0 <= occlusion_threshold <= 1.0:
        raise ValueError("occlusion threshold must be between 0 and 1")
    if (
        not math.isfinite(focal_length_mm)
        or not math.isfinite(aperture_mm)
        or focal_length_mm <= 0.0
        or aperture_mm <= 0.0
    ):
        raise ValueError("focal length and aperture must be positive")
    height_envelope = validate_height_envelope(min_height_m, max_height_m)
    look_down_envelope = validate_look_down_envelope(
        min_look_down_deg, max_look_down_deg
    )
    xy_bounds = validate_xy_bounds(xy_bounds_m)
    ranges = parse_yaw_ranges(yaw_ranges)
    minimum, maximum = validated_bounds(bounds_m)
    center = (minimum + maximum) * 0.5
    radius = 0.5 * float(np.linalg.norm(maximum - minimum))
    minimum_distance = (
        min_distance_m if min_distance_m is not None else max(0.5, radius * 1.5)
    )
    maximum_distance = (
        max_distance_m
        if max_distance_m is not None
        else max(minimum_distance, radius * 2.5)
    )
    if (
        not math.isfinite(minimum_distance)
        or not math.isfinite(maximum_distance)
        or minimum_distance <= 0.0
        or maximum_distance < minimum_distance
    ):
        raise ValueError(
            "look-at distances must be positive and max-distance >= min-distance"
        )
    if not math.isfinite(height_offset_m):
        raise ValueError("height offset must be finite")
    target_shapes = validate_look_at_workload(
        scene,
        target_path,
        candidate_count=candidate_count,
    )
    target_shape_ids = np.asarray(
        sorted(shape.shape_id for shape in target_shapes), dtype=np.int64
    )
    target_samples = _target_surface_samples(scene, target_shapes)
    phase = random.Random(seed).random() * 360.0
    base_height_offset = float(height_offset_m)
    minimum_height_offset = (
        base_height_offset if height_envelope[0] is None else float(height_envelope[0])
    )
    maximum_height_offset = (
        base_height_offset if height_envelope[1] is None else float(height_envelope[1])
    )
    if height_envelope[0] is None:
        minimum_height_offset = min(base_height_offset, maximum_height_offset)
    if height_envelope[1] is None:
        maximum_height_offset = max(base_height_offset, minimum_height_offset)

    envelope_candidates: list[CameraPose] = []
    envelope_bearings: list[float] = []
    envelope_indices: list[int] = []
    envelope_angles: list[float] = []
    constraint_rejected_count = 0
    duplicate_rejected_count = 0
    position_keys: set[tuple[float, float, float]] = set()
    direction_keys: set[tuple[float, float, float]] = set()
    for index in range(candidate_count):
        check_cancelled()
        bearing = (phase + 360.0 * index / candidate_count) % 360.0
        if not _yaw_allowed(bearing, ranges):
            constraint_rejected_count += 1
            continue
        distance_fraction = (index % 3) / 2.0
        height_fraction = ((index // 3) % 3) / 2.0
        distance = minimum_distance + distance_fraction * (
            maximum_distance - minimum_distance
        )
        height_offset = minimum_height_offset + height_fraction * (
            maximum_height_offset - minimum_height_offset
        )
        if abs(height_offset) > distance + 1.0e-12:
            constraint_rejected_count += 1
            continue
        horizontal_distance = math.sqrt(
            max(0.0, distance * distance - height_offset**2)
        )
        radians = math.radians(bearing)
        position = np.asarray(
            [
                center[0] + horizontal_distance * math.cos(radians),
                center[1] + horizontal_distance * math.sin(radians),
                center[2] + height_offset,
            ]
        )
        look_down_angle = _look_down_angle_deg(position, center)
        if not (
            look_down_envelope[0] - 1.0e-12
            <= look_down_angle
            <= look_down_envelope[1] + 1.0e-12
        ):
            constraint_rejected_count += 1
            continue
        if xy_bounds is not None and not (
            xy_bounds[0][0] - 1.0e-12 <= position[0] <= xy_bounds[0][1] + 1.0e-12
            and xy_bounds[1][0] - 1.0e-12 <= position[1] <= xy_bounds[1][1] + 1.0e-12
        ):
            constraint_rejected_count += 1
            continue
        pose = _pose(
            position,
            center,
            focal_length_mm=focal_length_mm,
            aperture_mm=aperture_mm,
        )
        position_key = tuple(round(float(value), 10) for value in pose.position_m)
        direction_key = tuple(round(float(value), 10) for value in pose.forward)
        if position_key in position_keys or direction_key in direction_keys:
            duplicate_rejected_count += 1
            continue
        position_keys.add(position_key)
        direction_keys.add(direction_key)
        envelope_candidates.append(pose)
        envelope_bearings.append(bearing)
        envelope_indices.append(index)
        envelope_angles.append(look_down_angle)

    check_cancelled()
    geometry_keep = _geometry_clearance_mask(envelope_candidates, backend)
    candidates: list[CameraPose] = []
    bearings: list[float] = []
    candidate_indices: list[int] = []
    look_down_angles: list[float] = []
    occlusions: list[float] = []
    visible_samples: list[int] = []
    in_frame_samples: list[int] = []
    for pose, bearing, source_index, look_down_angle, keep in zip(
        envelope_candidates,
        envelope_bearings,
        envelope_indices,
        envelope_angles,
        geometry_keep,
        strict=True,
    ):
        check_cancelled()
        if not bool(keep):
            continue
        position = np.asarray(pose.position_m, dtype=np.float64)
        in_frame, delta, distances = frustum_mask(pose, target_samples)
        in_frame_indices = np.flatnonzero(in_frame)
        visible = np.zeros(len(target_samples), dtype=bool)
        if len(in_frame_indices):
            check_cancelled()
            selected_delta = delta[in_frame_indices]
            selected_distances = distances[in_frame_indices]
            directions = selected_delta / selected_distances[:, None]
            origins = np.repeat(position[None, :], len(in_frame_indices), axis=0)
            hits = backend.evaluate_rays(origins, directions)
            check_cancelled()
            visible[in_frame_indices] = np.isin(hits.shape_ids, target_shape_ids)
        visible_count = int(visible.sum())
        occlusion = 1.0 - visible_count / len(target_samples)
        if occlusion <= occlusion_threshold + 1.0e-12:
            candidates.append(pose)
            bearings.append(bearing)
            candidate_indices.append(source_index)
            look_down_angles.append(look_down_angle)
            occlusions.append(occlusion)
            visible_samples.append(visible_count)
            in_frame_samples.append(int(in_frame.sum()))

    selected: list[int] = []
    if candidates:
        first = min(
            range(len(candidates)), key=lambda item: (occlusions[item], bearings[item])
        )
        selected.append(first)
        directions = np.asarray(
            [
                np.asarray(pose.position_m, dtype=np.float64) - center
                for pose in candidates
            ]
        )
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        minimum_separation = np.full(len(candidates), np.inf, dtype=np.float64)
        is_selected = np.zeros(len(candidates), dtype=bool)
        is_selected[first] = True
        while len(selected) < min(camera_count, len(candidates)):
            check_cancelled()
            latest = directions[selected[-1]]
            separation = np.degrees(
                np.arccos(np.clip(directions @ latest, -1.0, 1.0))
            )
            np.minimum(minimum_separation, separation, out=minimum_separation)
            remaining = np.flatnonzero(~is_selected)
            best = max(
                (int(index) for index in remaining),
                key=lambda item: (
                    float(minimum_separation[item]),
                    -occlusions[item],
                    -bearings[item],
                ),
            )
            selected.append(best)
            is_selected[best] = True

    enough = len(selected) >= camera_count
    if not enough and not allow_fewer:
        chosen: list[int] = []
        stop_reason = (
            "insufficient_unoccluded_views" if candidates else "no_valid_candidate"
        )
    else:
        chosen = selected[:camera_count]
        if len(chosen) == camera_count:
            stop_reason = "target_met"
        elif chosen:
            stop_reason = "allow_fewer"
        else:
            stop_reason = "no_valid_candidate"
    records = []
    for output_index, candidate_index in enumerate(chosen):
        check_cancelled()
        record = _camera_record(
            scene, candidates[candidate_index], center, index=output_index
        )
        record.update(
            {
                "candidate_index": candidate_indices[candidate_index],
                "bearing_deg": bearings[candidate_index],
                "look_down_deg": look_down_angles[candidate_index],
                "occlusion_fraction": occlusions[candidate_index],
                "in_frame_target_samples": in_frame_samples[candidate_index],
                "visible_target_samples": visible_samples[candidate_index],
                "target_sample_count": len(target_samples),
            }
        )
        records.append(record)
    report = {
        "schema": "usd-cli.camera-placement.v1",
        "method": "look_at",
        "source_digest": scene.source_digest,
        "analysis_policy": scene.policy.as_dict(),
        "analysis_policy_digest": scene.policy.digest,
        "target_path": target_path,
        "seed": seed,
        "requested_count": camera_count,
        "selected_count": len(chosen),
        "passed": len(chosen) == camera_count,
        "stop_reason": stop_reason,
        "candidate_count": candidate_count,
        "valid_candidate_count": len(candidates),
        "envelope_candidate_count": len(envelope_candidates),
        "constraint_rejected_candidate_count": constraint_rejected_count,
        "duplicate_rejected_candidate_count": duplicate_rejected_count,
        "geometry_rejected_candidate_count": int((~geometry_keep).sum()),
        "occlusion_threshold": occlusion_threshold,
        "yaw_ranges": [list(item) for item in ranges],
        "cameras": records,
        "config": {
            "min_distance_m": minimum_distance,
            "max_distance_m": maximum_distance,
            "height_offset_m": height_offset_m,
            "height_reference": "target_center_z",
            "min_height_m": height_envelope[0],
            "max_height_m": height_envelope[1],
            "effective_min_height_m": minimum_height_offset,
            "effective_max_height_m": maximum_height_offset,
            "min_look_down_deg": look_down_envelope[0],
            "max_look_down_deg": look_down_envelope[1],
            "xy_bounds_m": (
                [list(axis) for axis in xy_bounds] if xy_bounds is not None else None
            ),
            "camera_clearance_m": CAMERA_CLEARANCE_M,
            "allow_fewer": allow_fewer,
            "focal_length": focal_length_mm,
            "aperture": aperture_mm,
        },
        "backend": {"name": "newton_warp", **backend.versions},
    }
    return PlacementEvaluation(
        report=report,
        poses=tuple(candidates[index] for index in chosen),
        masks=None,
    )
