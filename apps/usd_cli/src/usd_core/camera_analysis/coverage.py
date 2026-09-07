# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic floor sampling, visibility, and coverage reporting."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from usd_core.camera_analysis.cancellation import check_cancelled
from usd_core.camera_analysis.contracts import CameraPose, SceneAnalysisIR
from usd_core.camera_analysis.scene import camera_pose, is_path_at_or_below
from usd_core.camera_analysis.warp_scoring import score_coverage_masks

MAX_CELLS = 100_000
MAX_CAMERAS = 64
MAX_COVERAGE_RAYS = 2_000_000
MAX_VISIBILITY_BOUNDARY_EDGES = 200_000
FLOOR_PLANARITY_TOLERANCE_M = 1.0e-4


@dataclass(frozen=True)
class GridSurface:
    rows: int
    columns: int
    x_min_m: float
    x_max_m: float
    y_min_m: float
    y_max_m: float
    points_m: np.ndarray
    accessible: np.ndarray
    shape_ids: np.ndarray | None = None

    @property
    def cell_width_m(self) -> float:
        return (self.x_max_m - self.x_min_m) / self.columns

    @property
    def cell_height_m(self) -> float:
        return (self.y_max_m - self.y_min_m) / self.rows

    @property
    def cell_area_m2(self) -> float:
        return self.cell_width_m * self.cell_height_m

    @property
    def accessible_points_m(self) -> np.ndarray:
        return self.points_m.reshape(-1, 3)[self.accessible.reshape(-1)]


@dataclass(frozen=True)
class CoverageEvaluation:
    report: dict
    grid: GridSurface
    camera_masks: np.ndarray
    overlap_counts: np.ndarray


def _grid_shape(
    width: float, height: float, grid: int, cell_size_m: float | None
) -> tuple[int, int]:
    if (
        not math.isfinite(width)
        or not math.isfinite(height)
        or width <= 0.0
        or height <= 0.0
    ):
        raise ValueError("coverage scope must have nonzero horizontal extent")
    if cell_size_m is not None:
        if not math.isfinite(cell_size_m) or cell_size_m <= 0.0:
            raise ValueError("cell size must be finite and positive")
        if width / cell_size_m > MAX_CELLS or height / cell_size_m > MAX_CELLS:
            raise ValueError(
                f"coverage grid would exceed the {MAX_CELLS:,}-cell limit "
                "(increase --cell-size)"
            )
        columns = max(1, int(math.ceil(width / cell_size_m)))
        rows = max(1, int(math.ceil(height / cell_size_m)))
    else:
        if (
            isinstance(grid, bool)
            or not isinstance(grid, int | np.integer)
            or grid < 2
            or grid > 512
        ):
            raise ValueError("grid must be between 2 and 512")
        if width >= height:
            columns = grid
            rows = max(2, int(round(grid * height / width)))
        else:
            rows = grid
            columns = max(2, int(round(grid * width / height)))
    cells = rows * columns
    if cells > MAX_CELLS:
        raise ValueError(
            f"coverage grid would allocate {cells:,} cells; limit is {MAX_CELLS:,} "
            "(increase --cell-size or reduce --grid)"
        )
    return rows, columns


def validated_bounds(
    bounds_m: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    try:
        if len(bounds_m) != 2:
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise ValueError("coverage bounds must be a (minimum, maximum) pair") from exc
    try:
        minimum, maximum = (np.asarray(value, dtype=np.float64) for value in bounds_m)
    except (TypeError, ValueError) as exc:
        raise ValueError("coverage bounds must be finite ordered 3D points") from exc
    if (
        minimum.shape != (3,)
        or maximum.shape != (3,)
        or not np.all(np.isfinite(minimum))
        or not np.all(np.isfinite(maximum))
        or np.any(maximum < minimum)
    ):
        raise ValueError("coverage bounds must be finite ordered 3D points")
    return minimum, maximum


def validate_grid_workload(
    bounds_m: tuple[np.ndarray, np.ndarray],
    *,
    grid: int,
    cell_size_m: float | None,
    view_count: int,
    operation: str,
) -> tuple[int, int]:
    """Reject worst-case grid/ray/mask work before surface or backend allocation."""

    if (
        isinstance(view_count, bool)
        or not isinstance(view_count, int | np.integer)
        or view_count < 0
    ):
        raise ValueError("view count must be a nonnegative integer")
    minimum, maximum = validated_bounds(bounds_m)
    rows, columns = _grid_shape(
        float(maximum[0] - minimum[0]),
        float(maximum[1] - minimum[1]),
        grid,
        cell_size_m,
    )
    # At most two rays per cell identify the bottom operational surface and its
    # unobstructed top hit, followed by at most one visibility ray per cell and
    # requested view. This also bounds the dense mask stack.
    ray_budget = rows * columns * (view_count + 2)
    if ray_budget > MAX_COVERAGE_RAYS:
        raise ValueError(
            f"{operation} would cast up to {ray_budget:,} rays; limit is "
            f"{MAX_COVERAGE_RAYS:,} (reduce cameras/candidates/grid)"
        )
    return rows, columns


def _admitted_scene_ceiling_m(scene: SceneAnalysisIR) -> float:
    """Conservative maximum Z of all admitted, non-helper analysis shapes.

    Floor sampling needs only a safe ray-origin ceiling, not exact global XY
    bounds.  Reducing each packed mesh resource once avoids multiplying vertex
    scans by its differently transformed occurrences; transforming the resource
    AABB support remains conservative for every affine occurrence.
    """

    resources = {resource.resource_id: resource for resource in scene.meshes}
    mesh_bounds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    ceiling_m: float | None = None
    for shape in scene.shapes:
        check_cancelled()
        if shape.analysis_role == "helper":
            continue
        if shape.kind == "mesh":
            resource_id = shape.mesh_resource_id or ""
            resource = resources.get(resource_id)
            if resource is None:
                raise ValueError(
                    f"analysis shape {shape.prim_path} has no mesh bounds resource"
                )
            local_bounds = mesh_bounds.get(resource_id)
            if local_bounds is None:
                vertices = np.asarray(resource.vertices)
                if (
                    vertices.ndim != 2
                    or vertices.shape[1] != 3
                    or len(vertices) == 0
                    or vertices.dtype.kind not in "fiu"
                    or not np.all(np.isfinite(vertices))
                ):
                    raise ValueError(
                        f"analysis shape {shape.prim_path} has malformed or "
                        "non-finite mesh bounds vertices"
                    )
                local_bounds = (
                    np.asarray(vertices.min(axis=0), dtype=np.float64),
                    np.asarray(vertices.max(axis=0), dtype=np.float64),
                )
                mesh_bounds[resource_id] = local_bounds
        else:
            parameters = shape.parameters
            if shape.kind == "box":
                extent = np.asarray(
                    [parameters["hx"], parameters["hy"], parameters["hz"]],
                    dtype=np.float64,
                )
            elif shape.kind == "sphere":
                extent = np.full(3, parameters["radius"], dtype=np.float64)
            elif shape.kind == "capsule":
                extent = np.asarray(
                    [
                        parameters["radius"],
                        parameters["radius"],
                        parameters["half_height"] + parameters["radius"],
                    ],
                    dtype=np.float64,
                )
            elif shape.kind in {"cylinder", "cone"}:
                extent = np.asarray(
                    [
                        parameters["radius"],
                        parameters["radius"],
                        parameters["half_height"],
                    ],
                    dtype=np.float64,
                )
            elif shape.kind == "plane":
                extent = np.asarray(
                    [parameters["width"] * 0.5, parameters["length"] * 0.5, 0.0],
                    dtype=np.float64,
                )
            else:  # pragma: no cover - SceneAnalysisIR owns the closed kind set
                raise ValueError(
                    f"analysis shape {shape.prim_path} has unsupported bounds kind "
                    f"{shape.kind!r}"
                )
            if not np.all(np.isfinite(extent)) or np.any(extent < 0.0):
                raise ValueError(
                    f"analysis shape {shape.prim_path} has malformed or non-finite "
                    "bounds parameters"
                )
            local_bounds = -extent, extent

        matrix = np.asarray(shape.transform, dtype=np.float64)
        if (
            matrix.shape != (4, 4)
            or not np.all(np.isfinite(matrix))
            or not np.allclose(matrix[:3, 3], 0.0, atol=1.0e-10)
            or not math.isclose(float(matrix[3, 3]), 1.0, abs_tol=1.0e-10)
        ):
            raise ValueError(
                f"analysis shape {shape.prim_path} has a malformed or non-finite "
                "affine transform"
            )
        z_coefficients = matrix[:3, 2]
        support = np.where(z_coefficients >= 0.0, local_bounds[1], local_bounds[0])
        shape_ceiling_m = float(support @ z_coefficients + matrix[3, 2])
        if not math.isfinite(shape_ceiling_m):
            raise ValueError(
                f"analysis shape {shape.prim_path} has non-finite world bounds"
            )
        ceiling_m = (
            shape_ceiling_m if ceiling_m is None else max(ceiling_m, shape_ceiling_m)
        )
    if ceiling_m is None:
        raise ValueError("analysis scene contains no admitted non-helper geometry")
    return ceiling_m


def sample_surface(
    scene: SceneAnalysisIR,
    backend,
    bounds_m: tuple[np.ndarray, np.ndarray],
    *,
    scope_path: str,
    grid: int = 32,
    cell_size_m: float | None = None,
) -> GridSurface:
    """Probe the topmost eligible surface below each XY grid cell."""

    check_cancelled()
    minimum, maximum = validated_bounds(bounds_m)
    width, height = float(maximum[0] - minimum[0]), float(maximum[1] - minimum[1])
    rows, columns = _grid_shape(width, height, grid, cell_size_m)
    dx, dy = width / columns, height / rows
    xs = minimum[0] + (np.arange(columns, dtype=np.float64) + 0.5) * dx
    ys = minimum[1] + (np.arange(rows, dtype=np.float64) + 0.5) * dy
    xx, yy = np.meshgrid(xs, ys)
    clearance = max(1.0, float(maximum[2] - minimum[2]) * 0.1)
    # ``bounds_m`` deliberately retains the requested scope's XY footprint, but
    # its Z maximum can be only the top of a floor Gprim.  Start above every
    # admitted non-helper shape so an external shelf or machine can still reject
    # the floor cells it covers.
    probe_ceiling_m = max(float(maximum[2]), _admitted_scene_ceiling_m(scene))
    origins = np.column_stack(
        (
            xx.reshape(-1),
            yy.reshape(-1),
            np.full(rows * columns, probe_ceiling_m + clearance),
        )
    )
    directions = np.tile(np.asarray([0.0, 0.0, -1.0]), (len(origins), 1))
    check_cancelled()
    hits = backend.evaluate_rays(origins, directions)
    check_cancelled()
    path_by_id = scene.shape_path_by_id
    scoped_shape_ids = {
        shape_id
        for shape_id, prim_path in path_by_id.items()
        if scope_path == "/" or is_path_at_or_below(prim_path, scope_path)
    }
    scoped_shape_id_array = np.asarray(sorted(scoped_shape_ids), dtype=np.int64)
    scoped_floor_ids = {
        shape.shape_id
        for shape in scene.shapes
        # Floor and target membership may intentionally overlap (for example a
        # look-at target used as the requested visibility surface).  The compact
        # ShapeIR role retains target precedence, so derive floor membership from
        # the policy paths rather than losing that second membership.
        if shape.analysis_role != "helper"
        and any(
            is_path_at_or_below(shape.prim_path, floor_path)
            for floor_path in scene.policy.floor_paths
        )
        and shape.shape_id in scoped_shape_ids
    }
    if scene.policy.floor_paths and not scoped_floor_ids:
        raise ValueError(
            f"scope {scope_path} contains no shape selected by the floor policy"
        )

    eligible_downward = (
        (hits.shape_ids >= 0)
        & (hits.distances_m >= 0.0)
        & np.isin(hits.shape_ids, scoped_shape_id_array)
    )
    if scoped_floor_ids:
        # Obstacles stay in the visibility model, so a shelf or machine above a
        # declared floor rejects the covered cell instead of becoming walkable.
        accepted = eligible_downward & np.isin(
            hits.shape_ids,
            np.asarray(sorted(scoped_floor_ids), dtype=np.int32),
        )
    else:
        # A broad container scope is not itself a floor declaration. Infer the
        # bottom supporting shape from the opposite ray and require the top hit
        # to be that same shape. This prevents shelf and obstacle tops from
        # silently becoming accessible while retaining backwards-compatible
        # operation for unlabelled floor-only scenes.
        lower_origins = origins.copy()
        lower_origins[:, 2] = minimum[2] - clearance
        check_cancelled()
        upward = backend.evaluate_rays(lower_origins, -directions)
        check_cancelled()
        eligible_upward = (
            (upward.shape_ids >= 0)
            & (upward.distances_m >= 0.0)
            & np.isin(upward.shape_ids, scoped_shape_id_array)
        )
        accepted = (
            eligible_downward & eligible_upward & (hits.shape_ids == upward.shape_ids)
        )
    points = origins.copy()
    points[:, 2] -= np.where(accepted, hits.distances_m, 0.0)
    points[~accepted] = np.nan
    if not np.any(accepted):
        raise ValueError(
            f"scope {scope_path} has no top-facing surface samples on the requested grid"
        )
    return GridSurface(
        rows=rows,
        columns=columns,
        x_min_m=float(minimum[0]),
        x_max_m=float(maximum[0]),
        y_min_m=float(minimum[1]),
        y_max_m=float(maximum[1]),
        points_m=points.reshape(rows, columns, 3),
        accessible=accepted.reshape(rows, columns),
        shape_ids=np.where(accepted, hits.shape_ids, -1).reshape(rows, columns),
    )


def frustum_mask(
    camera: CameraPose, points: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    position = np.asarray(camera.position_m, dtype=np.float64)
    delta = points - position
    distances = np.linalg.norm(delta, axis=1)
    forward = delta @ np.asarray(camera.forward, dtype=np.float64)
    horizontal = delta @ np.asarray(camera.right, dtype=np.float64)
    vertical = delta @ np.asarray(camera.up, dtype=np.float64)
    near, far = camera.clipping_range_m
    horizontal_min, horizontal_max = camera.horizontal_tan_bounds
    vertical_min, vertical_max = camera.vertical_tan_bounds
    mask = (
        (distances > 1.0e-8)
        & (forward >= near)
        & (forward <= far)
        & (horizontal >= forward * horizontal_min)
        & (horizontal <= forward * horizontal_max)
        & (vertical >= forward * vertical_min)
        & (vertical <= forward * vertical_max)
    )
    return mask, delta, distances


def visibility_mask(camera: CameraPose, grid: GridSurface, backend) -> np.ndarray:
    """Return a rows×columns mask for frustum-visible, unoccluded cells."""

    check_cancelled()
    flat_accessible = grid.accessible.reshape(-1)
    points = grid.points_m.reshape(-1, 3)
    accessible_indices = np.flatnonzero(flat_accessible)
    accessible_points = points[accessible_indices]
    in_frustum, delta, distances = frustum_mask(camera, accessible_points)
    candidate_indices = np.flatnonzero(in_frustum)
    visible_accessible = np.zeros(len(accessible_points), dtype=bool)
    if len(candidate_indices):
        check_cancelled()
        selected_delta = delta[candidate_indices]
        selected_distance = distances[candidate_indices]
        directions = selected_delta / selected_distance[:, None]
        origins = np.repeat(
            np.asarray(camera.position_m, dtype=np.float64)[None, :],
            len(candidate_indices),
            axis=0,
        )
        hits = backend.evaluate_rays(origins, directions)
        check_cancelled()
        epsilon = np.maximum(1.0e-4, selected_distance * 1.0e-4)
        visible_accessible[candidate_indices] = (hits.distances_m < 0.0) | (
            hits.distances_m >= selected_distance - epsilon
        )
    result = np.zeros(grid.rows * grid.columns, dtype=bool)
    result[accessible_indices] = visible_accessible
    return result.reshape(grid.rows, grid.columns)


def _signed_area(loop: list[tuple[int, int]]) -> float:
    area = 0.0
    for index, ((x0, y0), (x1, y1)) in enumerate(zip(loop, loop[1:], strict=False)):
        if index % 1024 == 0:
            check_cancelled()
        area += x0 * y1 - x1 * y0
    return 0.5 * area


def _contains(loop: list[tuple[int, int]], point: tuple[int, int]) -> bool:
    x, y = point
    inside = False
    for index, ((x0, y0), (x1, y1)) in enumerate(zip(loop, loop[1:], strict=False)):
        if index % 1024 == 0:
            check_cancelled()
        if (y0 > y) == (y1 > y):
            continue
        crossing = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
        if crossing > x:
            inside = not inside
    return inside


def _boundary_loops(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    rows, columns = mask.shape
    edges: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for index, (row, column) in enumerate(zip(*np.nonzero(mask), strict=False)):
        if index % 1024 == 0:
            check_cancelled()
        if row == 0 or not mask[row - 1, column]:
            edges[(column, row)].append((column + 1, row))
        if column == columns - 1 or not mask[row, column + 1]:
            edges[(column + 1, row)].append((column + 1, row + 1))
        if row == rows - 1 or not mask[row + 1, column]:
            edges[(column + 1, row + 1)].append((column, row + 1))
        if column == 0 or not mask[row, column - 1]:
            edges[(column, row + 1)].append((column, row))
    for values in edges.values():
        values.sort()

    loops: list[list[tuple[int, int]]] = []
    # Recomputing ``min(edges)`` for every disconnected component is quadratic
    # (a checkerboard has one component per visible cell).  One sorted key list
    # preserves the exact deterministic start order while making extraction
    # O(E log E) overall.
    ordered_starts = sorted(edges)
    start_index = 0
    traversed_edges = 0
    while edges:
        while ordered_starts[start_index] not in edges:
            start_index += 1
        start = ordered_starts[start_index]
        current = start
        previous: tuple[int, int] | None = None
        loop = [start]
        while True:
            if traversed_edges % 1024 == 0:
                check_cancelled()
            candidates = edges.get(current)
            if not candidates:
                raise RuntimeError("visibility boundary is not closed")
            if previous is None or len(candidates) == 1:
                following = candidates.pop(0)
            else:
                incoming = (current[0] - previous[0], current[1] - previous[1])

                def turn(
                    candidate: tuple[int, int],
                    current: tuple[int, int] = current,
                    incoming: tuple[int, int] = incoming,
                ) -> tuple[float, tuple[int, int]]:
                    outgoing = (candidate[0] - current[0], candidate[1] - current[1])
                    angle = math.atan2(
                        incoming[0] * outgoing[1] - incoming[1] * outgoing[0],
                        incoming[0] * outgoing[0] + incoming[1] * outgoing[1],
                    )
                    return (-angle, candidate)

                following = min(candidates, key=turn)
                candidates.remove(following)
            if not candidates:
                del edges[current]
            previous, current = current, following
            loop.append(current)
            traversed_edges += 1
            if current == start:
                break
        if len(loop) >= 4:
            loops.append(loop)
    return loops


def _visibility_boundary_edge_count(mask: np.ndarray) -> int:
    """Count exact grid-boundary edges without materializing polygon objects."""

    values = np.asarray(mask, dtype=bool)
    if values.ndim != 2:
        raise ValueError("visibility mask must be two-dimensional")
    if not values.size:
        return 0
    return int(
        np.count_nonzero(values[0])
        + np.count_nonzero(values[-1])
        + np.count_nonzero(values[:, 0])
        + np.count_nonzero(values[:, -1])
        + np.count_nonzero(values[1:] != values[:-1])
        + np.count_nonzero(values[:, 1:] != values[:, :-1])
    )


def validate_visibility_region_workload(
    masks: np.ndarray,
    accessible: np.ndarray,
    *,
    max_boundary_edges: int = MAX_VISIBILITY_BOUNDARY_EDGES,
) -> int:
    """Bound lossless polygon output before constructing nested report objects."""

    check_cancelled()
    values = np.asarray(masks)
    accessible_values = np.asarray(accessible, dtype=bool)
    if values.ndim != 3 or values.shape[1:] != accessible_values.shape:
        raise ValueError("visibility masks and accessible grid shapes differ")
    if not isinstance(max_boundary_edges, int) or max_boundary_edges < 0:
        raise ValueError("visibility boundary edge limit must be nonnegative")
    total = 0
    for camera_index in range(len(values)):
        check_cancelled()
        total += _visibility_boundary_edge_count(
            np.asarray(values[camera_index], dtype=bool) & accessible_values
        )
        if total > max_boundary_edges:
            raise ValueError(
                "visibility_region_complexity_limit: lossless visibility polygons "
                f"exceed {max_boundary_edges:,} boundary edges across cameras "
                "(reduce --grid or camera count)"
            )
    return total


def visibility_regions(mask: np.ndarray, grid: GridSurface) -> list[dict]:
    """Lossless grid-boundary polygons, including disjoint outlines and holes."""

    validate_visibility_region_workload(
        np.asarray(mask, dtype=bool)[None, ...],
        np.ones((grid.rows, grid.columns), dtype=bool),
    )
    loops = _boundary_loops(mask)
    loop_areas = [_signed_area(loop) for loop in loops]
    outlines = [
        loop for loop, area in zip(loops, loop_areas, strict=True) if area > 0.0
    ]
    outline_areas = [area for area in loop_areas if area > 0.0]
    holes = [loop for loop, area in zip(loops, loop_areas, strict=True) if area < 0.0]
    hole_areas = [-area for area in loop_areas if area < 0.0]
    outline_bounds = [
        (
            min(point[0] for point in outline),
            max(point[0] for point in outline),
            min(point[1] for point in outline),
            max(point[1] for point in outline),
        )
        for outline in outlines
    ]
    assigned: dict[int, list[list[tuple[int, int]]]] = defaultdict(list)
    assigned_areas: dict[int, float] = defaultdict(float)
    for hole_index, hole in enumerate(holes):
        check_cancelled()
        point = hole[0]
        candidates = [
            (outline_areas[index], index)
            for index, (outline, bounds) in enumerate(
                zip(outlines, outline_bounds, strict=True)
            )
            if bounds[0] <= point[0] <= bounds[1]
            and bounds[2] <= point[1] <= bounds[3]
            and _contains(outline, point)
        ]
        if candidates:
            owner = min(candidates)[1]
            assigned[owner].append(hole)
            assigned_areas[owner] += hole_areas[hole_index]

    covered_z = grid.points_m[:, :, 2][mask]
    z = float(np.nanmedian(covered_z)) if len(covered_z) else 0.0

    def world(loop: list[tuple[int, int]]) -> list[list[float]]:
        points = []
        for index, (column, row) in enumerate(loop):
            if index % 1024 == 0:
                check_cancelled()
            points.append(
                [
                    grid.x_min_m + column * grid.cell_width_m,
                    grid.y_min_m + row * grid.cell_height_m,
                    z,
                ]
            )
        return points

    regions = []
    for index, outline in enumerate(outlines):
        check_cancelled()
        outline_area = outline_areas[index] * grid.cell_area_m2
        hole_area = assigned_areas[index] * grid.cell_area_m2
        regions.append(
            {
                "outline": world(outline),
                "holes": [world(hole) for hole in assigned[index]],
                "area_m2": outline_area - hole_area,
            }
        )
    return regions


def evaluate_coverage(
    scene: SceneAnalysisIR,
    backend,
    bounds_m: tuple[np.ndarray, np.ndarray],
    *,
    scope_path: str,
    camera_paths: list[str] | None = None,
    target_coverage: float = 0.95,
    per_cell: int = 1,
    grid: int = 32,
    cell_size_m: float | None = None,
) -> CoverageEvaluation:
    check_cancelled()
    if not math.isfinite(target_coverage) or not 0.0 <= target_coverage <= 1.0:
        raise ValueError("target coverage must be between 0 and 1")
    if per_cell < 1 or per_cell > MAX_CAMERAS:
        raise ValueError(f"per-cell redundancy must be between 1 and {MAX_CAMERAS}")
    cameras_by_path = {camera.prim_path: camera for camera in scene.cameras}
    selected_paths = (
        sorted(
            path
            for path, camera in cameras_by_path.items()
            if camera.projection == "perspective"
        )
        if camera_paths is None
        else list(camera_paths)
    )
    if not selected_paths:
        raise ValueError(
            "no eligible perspective cameras selected or present on the stage"
        )
    if len(selected_paths) > MAX_CAMERAS:
        raise ValueError(f"camera count exceeds limit {MAX_CAMERAS}")
    if len(set(selected_paths)) != len(selected_paths):
        raise ValueError("camera selection contains duplicate paths")
    missing = [path for path in selected_paths if path not in cameras_by_path]
    if missing:
        raise ValueError(f"not a camera or not found in analysis scene: {missing[0]}")
    unsupported = [
        path
        for path in selected_paths
        if cameras_by_path[path].projection != "perspective"
    ]
    if unsupported:
        camera = cameras_by_path[unsupported[0]]
        raise ValueError(
            "unsupported_camera_projection: "
            f"camera {camera.prim_path} uses {camera.projection!r}; "
            "coverage requires 'perspective'"
        )

    validate_grid_workload(
        bounds_m,
        grid=grid,
        cell_size_m=cell_size_m,
        view_count=len(selected_paths),
        operation="coverage evaluation",
    )
    sampled = sample_surface(
        scene,
        backend,
        bounds_m,
        scope_path=scope_path,
        grid=grid,
        cell_size_m=cell_size_m,
    )
    camera_masks: list[np.ndarray] = []
    for path in selected_paths:
        check_cancelled()
        camera_masks.append(
            visibility_mask(camera_pose(cameras_by_path[path]), sampled, backend)
        )
    masks = np.stack(camera_masks)
    check_cancelled()
    validate_visibility_region_workload(masks, sampled.accessible)
    scores = score_coverage_masks(
        masks,
        sampled.accessible,
        per_cell=per_cell,
        device=getattr(backend, "device", "cpu"),
    )
    counts = scores.counts
    accessible_count = scores.accessible_count
    covered_count = scores.covered_count
    ratio = covered_count / accessible_count
    accessible_z = sampled.accessible_points_m[:, 2]
    surface_z_m = float(np.median(accessible_z))
    surface_max_deviation_m = float(np.max(np.abs(accessible_z - surface_z_m)))
    if sampled.shape_ids is None:  # pragma: no cover - sample_surface owns this field
        raise RuntimeError("sampled coverage surface omitted its shape identities")
    path_by_id = scene.shape_path_by_id
    accessible_surface_paths = sorted(
        {
            path_by_id[int(shape_id)]
            for shape_id in sampled.shape_ids[sampled.accessible]
            if int(shape_id) in path_by_id
        }
    )

    camera_results = []
    for index, path in enumerate(selected_paths):
        check_cancelled()
        mask = masks[index] & sampled.accessible
        visible_count = int(scores.visible_counts[index])
        marginal = int(scores.marginal_counts[index])
        camera_results.append(
            {
                "camera": path,
                "visible_cells": visible_count,
                "visible_fraction": float(visible_count / accessible_count),
                "marginal_cells": marginal,
                "marginal_fraction": float(marginal / accessible_count),
                "visibility": {
                    "schema": "usd-cli.visibility-grid.v1",
                    "coordinate_space": "canonical_meter_z_up",
                    "encoding": "grid-boundary-v1",
                    "regions": visibility_regions(mask, sampled),
                    "area_m2": float(visible_count * sampled.cell_area_m2),
                },
            }
        )

    report = {
        "schema": "usd-cli.camera-coverage.v1",
        "source_digest": scene.source_digest,
        "analysis_policy": scene.policy.as_dict(),
        "analysis_policy_digest": scene.policy.digest,
        "scope_path": scope_path,
        "target_coverage": target_coverage,
        "per_cell": per_cell,
        "coverage_fraction": ratio,
        "passed": ratio + 1.0e-12 >= target_coverage,
        "accessible_cells": accessible_count,
        "covered_cells": covered_count,
        "uncovered_cells": accessible_count - covered_count,
        "accessible_area_m2": accessible_count * sampled.cell_area_m2,
        "covered_area_m2": covered_count * sampled.cell_area_m2,
        "overlap_histogram": {
            str(key): scores.histogram[key] for key in sorted(scores.histogram)
        },
        "grid": {
            "rows": sampled.rows,
            "columns": sampled.columns,
            "cell_width_m": sampled.cell_width_m,
            "cell_height_m": sampled.cell_height_m,
            "coordinate_space": "canonical_meter_z_up",
            "surface_z_m": surface_z_m,
            "surface_max_deviation_m": surface_max_deviation_m,
            "surface_planarity_tolerance_m": FLOOR_PLANARITY_TOLERANCE_M,
            "surface_planar": bool(
                surface_max_deviation_m <= FLOOR_PLANARITY_TOLERANCE_M
            ),
            "accessible_surface_paths": accessible_surface_paths,
        },
        "cameras": camera_results,
        "backend": {"name": "newton_warp", **backend.versions},
    }
    return CoverageEvaluation(
        report=report, grid=sampled, camera_masks=masks, overlap_counts=counts
    )
