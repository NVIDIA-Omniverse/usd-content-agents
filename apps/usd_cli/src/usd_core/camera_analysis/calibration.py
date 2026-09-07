# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Canonical camera calibration and floor-homography export."""

from __future__ import annotations

import math

import numpy as np

from usd_core.camera_analysis.contracts import SceneAnalysisIR
from usd_core.camera_analysis.identifiers import camera_stable_identity


def _lists(value: np.ndarray) -> list:
    return [[float(item) for item in row] for row in np.asarray(value)]


def _project_cv(
    points_world: np.ndarray, cv_from_world: np.ndarray, intrinsics: np.ndarray
) -> np.ndarray:
    homogeneous = np.column_stack((points_world, np.ones(len(points_world)))).T
    camera = cv_from_world @ homogeneous
    pixels = intrinsics @ camera[:3]
    return (pixels[:2] / pixels[2]).T


def _projection_verification(
    canonical_from_camera: np.ndarray,
    cv_from_world: np.ndarray,
    intrinsics: np.ndarray,
    usd_projection: np.ndarray,
    resolution: tuple[int, int],
) -> dict:
    """Compare exported K/extrinsics against OpenUSD's independent frustum matrix."""

    local_cv = np.asarray(
        [
            [-0.5, -0.5, 2.0],
            [0.5, -0.5, 2.0],
            [-0.5, 0.5, 2.0],
            [0.5, 0.5, 2.0],
            [0.0, 0.0, 5.0],
        ],
        dtype=np.float64,
    )
    # CV (+Z forward, +Y down) -> USD camera (+X right, +Y up, -Z forward).
    usd_local = local_cv * np.asarray([1.0, -1.0, -1.0])
    world = (
        np.column_stack((usd_local, np.ones(len(usd_local)))) @ canonical_from_camera
    )
    clip = np.column_stack((usd_local, np.ones(len(usd_local)))) @ usd_projection
    ndc = clip[:, :2] / clip[:, 3, None]
    # Projection matrices produce OpenGL-style NDC (+Y up).  Image coordinates
    # use a top-left origin, hence the Y inversion.
    resolution_width, resolution_height = (float(value) for value in resolution)
    expected = np.column_stack(
        (
            (ndc[:, 0] + 1.0) * 0.5 * resolution_width,
            (1.0 - ndc[:, 1]) * 0.5 * resolution_height,
        )
    )
    actual = _project_cv(world[:, :3], cv_from_world, intrinsics)
    errors = np.linalg.norm(actual - expected, axis=1)
    return {
        "sample_count": len(errors),
        "reference": "OpenUSD GfCamera frustum projection matrix",
        "max_error_px": float(errors.max(initial=0.0)),
        "rms_error_px": float(math.sqrt(float(np.mean(errors * errors)))),
        "passed": bool(float(errors.max(initial=0.0)) <= 1.0e-7),
    }


def _homography(
    cv_from_world: np.ndarray,
    intrinsics: np.ndarray,
    floor_bounds_m: tuple[np.ndarray, np.ndarray] | None,
    floor_surface: dict | None = None,
) -> dict:
    if floor_bounds_m is None:
        return {"eligible": False, "reason": "floor_scope_not_supplied"}
    minimum, maximum = (np.asarray(value, dtype=np.float64) for value in floor_bounds_m)
    if minimum.shape != (3,) or maximum.shape != (3,):
        raise ValueError("floor bounds must contain two 3D canonical points")
    if not np.all(np.isfinite(minimum)) or not np.all(np.isfinite(maximum)):
        raise ValueError("floor bounds contain NaN or infinity")
    if (
        np.any(maximum < minimum)
        or maximum[0] <= minimum[0]
        or maximum[1] <= minimum[1]
    ):
        raise ValueError("floor bounds must have positive canonical XY extent")
    if floor_surface is None:
        # Backwards-compatible low-level calibration: absent measured coverage,
        # the top of a caller-supplied floor slab is the only declared plane.
        floor_z = float(maximum[2])
        surface_metadata = {
            "source": "floor_scope_bounds_top",
            "max_deviation_m": 0.0,
            "planarity_tolerance_m": 0.0,
        }
    else:
        if not isinstance(floor_surface, dict):
            raise ValueError("floor surface metadata must be an object")
        try:
            floor_z = float(floor_surface["surface_z_m"])
            max_deviation = float(floor_surface["surface_max_deviation_m"])
            tolerance = float(floor_surface["surface_planarity_tolerance_m"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("floor surface metadata is incomplete") from exc
        if (
            not math.isfinite(floor_z)
            or not math.isfinite(max_deviation)
            or not math.isfinite(tolerance)
            or max_deviation < 0.0
            or tolerance < 0.0
        ):
            raise ValueError("floor surface metadata contains invalid values")
        surface_metadata = {
            "source": "accessible_coverage_samples_median",
            "max_deviation_m": max_deviation,
            "planarity_tolerance_m": tolerance,
        }
        if max_deviation > tolerance:
            return {
                "eligible": False,
                "reason": "nonplanar_accessible_surface",
                "floor_z_m": floor_z,
                **surface_metadata,
            }
    projection = intrinsics @ cv_from_world[:3, :]
    matrix = np.column_stack(
        (
            projection[:, 0],
            projection[:, 1],
            projection[:, 2] * floor_z + projection[:, 3],
        )
    )
    condition = float(np.linalg.cond(matrix))
    if not math.isfinite(condition) or condition >= 1.0e12:
        return {"eligible": False, "reason": "degenerate_floor_homography"}
    corners = np.asarray(
        [
            [minimum[0], minimum[1]],
            [maximum[0], minimum[1]],
            [maximum[0], maximum[1]],
            [minimum[0], maximum[1]],
            [(minimum[0] + maximum[0]) * 0.5, (minimum[1] + maximum[1]) * 0.5],
        ],
        dtype=np.float64,
    )
    floor_h = np.column_stack((corners, np.ones(len(corners))))
    via_h = (matrix @ floor_h.T).T
    if np.any(np.abs(via_h[:, 2]) <= 1.0e-12):
        return {"eligible": False, "reason": "floor_plane_intersects_camera_horizon"}
    via_h = via_h[:, :2] / via_h[:, 2, None]
    world = np.column_stack((corners, np.full(len(corners), floor_z)))
    direct = _project_cv(world, cv_from_world, intrinsics)
    errors = np.linalg.norm(via_h - direct, axis=1)
    return {
        "eligible": True,
        "direction": "canonical_floor_xy_to_image_uv",
        "floor_z_m": floor_z,
        **surface_metadata,
        "floor_plane": {
            "coordinate_space": "canonical_meter_z_up",
            "origin_m": [0.0, 0.0, floor_z],
            "u_axis": [1.0, 0.0, 0.0],
            "v_axis": [0.0, 1.0, 0.0],
            "normal": [0.0, 0.0, 1.0],
        },
        "condition_number": condition,
        "image_from_floor": _lists(matrix),
        "floor_from_image": _lists(np.linalg.inv(matrix)),
        "verification": {
            "sample_count": len(errors),
            "max_error_px": float(errors.max(initial=0.0)),
            "rms_error_px": float(math.sqrt(float(np.mean(errors * errors)))),
            "passed": bool(float(errors.max(initial=0.0)) <= 1.0e-7),
        },
    }


def camera_calibration(
    stage,
    scene: SceneAnalysisIR,
    camera_path: str,
    *,
    resolution: tuple[int, int],
    floor_bounds_m: tuple[np.ndarray, np.ndarray] | None = None,
    floor_surface: dict | None = None,
    visibility: dict | None = None,
) -> dict:
    from pxr import Sdf, Usd, UsdGeom

    width, height = resolution
    if width < 1 or height < 1:
        raise ValueError("camera export resolution must be positive")
    prim = stage.GetPrimAtPath(Sdf.Path(camera_path))
    if not prim.IsValid() or not prim.IsA(UsdGeom.Camera):
        raise ValueError(f"not a camera: {camera_path}")
    camera = UsdGeom.Camera(prim)
    camera_ir = next(
        (item for item in scene.cameras if item.prim_path == camera_path), None
    )
    if camera_ir is None:
        raise ValueError(f"camera is excluded from the analysis scene: {camera_path}")
    time = Usd.TimeCode.Default()
    raw_focal = camera.GetFocalLengthAttr().Get(time)
    raw_horizontal_aperture = camera.GetHorizontalApertureAttr().Get(time)
    raw_vertical_aperture = camera.GetVerticalApertureAttr().Get(time)
    raw_horizontal_offset = camera.GetHorizontalApertureOffsetAttr().Get(time)
    raw_vertical_offset = camera.GetVerticalApertureOffsetAttr().Get(time)
    focal = float(50.0 if raw_focal is None else raw_focal)
    horizontal_aperture = float(
        20.955 if raw_horizontal_aperture is None else raw_horizontal_aperture
    )
    vertical_aperture = float(
        15.2908 if raw_vertical_aperture is None else raw_vertical_aperture
    )
    horizontal_offset = float(
        0.0 if raw_horizontal_offset is None else raw_horizontal_offset
    )
    vertical_offset = float(0.0 if raw_vertical_offset is None else raw_vertical_offset)
    projection_token = str(
        camera.GetProjectionAttr().Get(time) or UsdGeom.Tokens.perspective
    )
    values = (
        focal,
        horizontal_aperture,
        vertical_aperture,
        horizontal_offset,
        vertical_offset,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"camera {camera_path} has NaN or infinite lens parameters")
    if focal <= 0.0 or horizontal_aperture <= 0.0 or vertical_aperture <= 0.0:
        raise ValueError(
            f"camera {camera_path} has nonpositive focal length or aperture"
        )
    fx = focal / horizontal_aperture * width
    fy = focal / vertical_aperture * height
    # These signs match GfCamera::ComputeProjectionMatrix.  Positive horizontal
    # filmback offset moves the projected optical axis left; positive vertical
    # offset moves it down after converting NDC (+Y up) to image coordinates.
    cx = width * (0.5 - horizontal_offset / horizontal_aperture)
    cy = height * (0.5 + vertical_offset / vertical_aperture)
    intrinsics = np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])

    canonical_from_camera = np.asarray(camera_ir.transform, dtype=np.float64)
    linear = canonical_from_camera[:3, :3]
    norms = np.linalg.norm(linear, axis=1)
    rotation = linear / np.where(norms > 1.0e-12, norms, 1.0)[:, None]
    rigid = bool(
        np.all(norms > 1.0e-12)
        and np.allclose(norms, np.ones(3), atol=1.0e-7, rtol=1.0e-7)
        and np.allclose(rotation @ rotation.T, np.eye(3), atol=1.0e-7, rtol=1.0e-7)
        and math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1.0e-7)
    )
    camera_from_canonical = np.linalg.inv(canonical_from_camera)
    usd_to_cv = np.diag([1.0, -1.0, -1.0, 1.0])
    cv_from_world = usd_to_cv @ camera_from_canonical.T
    eligible = projection_token == str(UsdGeom.Tokens.perspective) and rigid
    reasons = []
    if projection_token != str(UsdGeom.Tokens.perspective):
        reasons.append("unsupported_projection")
    if not rigid:
        reasons.append("non_rigid_camera_transform")
    stable_id, id_source = camera_stable_identity(prim, camera_path)
    clipping = camera.GetClippingRangeAttr().Get() or (0.01, 1.0e6)
    lens_unit_m = scene.meters_per_unit / 10.0
    result = {
        "schema": "usd-cli.camera-calibration.v1",
        "id": str(stable_id),
        "id_source": id_source,
        "path": camera_path,
        "resolution": {"width": width, "height": height},
        "projection": projection_token,
        "distortion": {"model": "none", "coefficients": []},
        "intrinsics": {
            "K": _lists(intrinsics),
            "focal_length_raw": focal,
            "horizontal_aperture_raw": horizontal_aperture,
            "vertical_aperture_raw": vertical_aperture,
            "horizontal_aperture_offset_raw": horizontal_offset,
            "vertical_aperture_offset_raw": vertical_offset,
            "raw_unit": "tenths_of_scene_unit",
            "raw_unit_m": lens_unit_m,
            "focal_length_m": focal * lens_unit_m,
            "horizontal_aperture_m": horizontal_aperture * lens_unit_m,
            "vertical_aperture_m": vertical_aperture * lens_unit_m,
        },
        "extrinsics": {
            "canonical_world_from_camera": _lists(canonical_from_camera),
            "camera_from_canonical_world": _lists(camera_from_canonical),
            "layout": "row_major",
            "vector_convention": "row_vector_postmultiply",
            "camera_axes": "+X right, +Y up, -Z forward",
            "coordinate_space": "canonical_meter_z_up",
        },
        "image_axes": "+u right, +v down",
        "clipping_range_m": [
            float(clipping[0]) * scene.meters_per_unit,
            float(clipping[1]) * scene.meters_per_unit,
        ],
        "calibration": {"eligible": eligible, "reasons": reasons},
        "homography": (
            _homography(
                cv_from_world,
                intrinsics,
                floor_bounds_m,
                floor_surface=floor_surface,
            )
            if eligible
            else {"eligible": False, "reason": reasons[0]}
        ),
    }
    if eligible:
        gf_projection = camera.GetCamera(time).frustum.ComputeProjectionMatrix()
        projection_array = np.asarray(
            [
                [float(gf_projection[row][column]) for column in range(4)]
                for row in range(4)
            ],
            dtype=np.float64,
        )
        projection_verification = _projection_verification(
            canonical_from_camera,
            cv_from_world,
            intrinsics,
            projection_array,
            (width, height),
        )
        result["calibration"]["projection_verification"] = projection_verification
        if not projection_verification["passed"]:
            result["calibration"]["eligible"] = False
            result["calibration"]["reasons"].append("projection_verification_failed")
            result["homography"] = {
                "eligible": False,
                "reason": "projection_verification_failed",
            }
    if visibility is not None:
        result["visibility"] = visibility
    return result
