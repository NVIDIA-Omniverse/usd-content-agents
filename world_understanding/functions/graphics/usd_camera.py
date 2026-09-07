# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""USD camera parameter extraction and projection utilities."""

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    from pxr import Gf, Sdf, Usd, UsdGeom
except ImportError as e:
    raise ImportError(
        "USD Python bindings (pxr) are required. Install a supported provider "
        "(Linux ARM64 + Python 3.12: `uv pip install usd-exchange`; Linux "
        "ARM64 + Python 3.13 is currently unsupported; other supported "
        "platforms: `uv pip install usd-core`)."
    ) from e


def _matrix4_to_list(m: Gf.Matrix4d) -> list[list[float]]:
    """Convert Gf.Matrix4d to nested list format."""
    return [
        [float(m[0][0]), float(m[0][1]), float(m[0][2]), float(m[0][3])],
        [float(m[1][0]), float(m[1][1]), float(m[1][2]), float(m[1][3])],
        [float(m[2][0]), float(m[2][1]), float(m[2][2]), float(m[2][3])],
        [float(m[3][0]), float(m[3][1]), float(m[3][2]), float(m[3][3])],
    ]


def _list_to_matrix4(values: list[list[float]]) -> Gf.Matrix4d:
    """Convert nested list to Gf.Matrix4d."""
    m = Gf.Matrix4d(1.0)
    for r in range(4):
        for c in range(4):
            m[r][c] = float(values[r][c])
    return m


def extract_camera_parameters(
    usd_path: str,
    camera_path: str,
    image_width: int,
    image_height: int | None = None,
    time_code: float | None = None,
) -> dict[str, Any]:
    """
    Extract camera parameters from a USD file.

    This function reads camera properties from a USD file and computes
    intrinsic and extrinsic parameters compatible with computer vision
    applications.

    Args:
        usd_path: Path to the USD file
        camera_path: Path to the camera prim (e.g., "/World/Camera")
        image_width: Rendered image width in pixels
        image_height: Rendered image height in pixels. If None, computed
                     from camera aspect ratio
        time_code: USD time code for animated cameras. If None, uses default

    Returns:
        Dictionary containing:
            - camera_path: Camera prim path
            - projection: "perspective" or "orthographic"
            - image_width: Image width in pixels
            - image_height: Image height in pixels
            - pixel_aspect_ratio: Pixel aspect ratio (usually 1.0)
            - near: Near clipping plane distance
            - far: Far clipping plane distance
            - focal_length_mm: Focal length in millimeters
            - horizontal_aperture_mm: Horizontal aperture in mm
            - vertical_aperture_mm: Vertical aperture in mm
            - horizontal_aperture_offset_mm: Horizontal aperture offset
            - vertical_aperture_offset_mm: Vertical aperture offset
            - fov_x_rad: Horizontal field of view in radians
            - fov_y_rad: Vertical field of view in radians
            - K: 3x3 intrinsic matrix [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
            - camera_world_transform: 4x4 camera-to-world transform
            - world_to_camera: 4x4 world-to-camera transform

    Raises:
        ValueError: If USD file or camera prim not found
        RuntimeError: If camera parameters cannot be extracted

    Example:
        >>> params = extract_camera_parameters(
        ...     "scene.usda",
        ...     "/World/Camera",
        ...     1920, 1080
        ... )
        >>> print(f"FOV: {math.degrees(params['fov_x_rad']):.1f}°")
    """
    # Open the USD stage
    stage = Usd.Stage.Open(usd_path)
    if not stage:
        raise ValueError(f"Failed to open USD file: {usd_path}")

    return extract_camera_parameters_from_stage(
        stage=stage,
        camera_path=camera_path,
        image_width=image_width,
        image_height=image_height,
        time_code=time_code,
    )


def extract_camera_parameters_from_stage(
    stage: Usd.Stage,
    camera_path: str,
    image_width: int,
    image_height: int | None = None,
    time_code: float | None = None,
    *,
    xform_cache: UsdGeom.XformCache | None = None,
) -> dict[str, Any]:
    """Extract camera parameters from an already-open USD stage."""

    # Get the camera prim
    cam_prim = stage.GetPrimAtPath(Sdf.Path(camera_path))
    if not cam_prim or not cam_prim.IsA(UsdGeom.Camera):
        raise ValueError(
            f"Camera prim not found or not a UsdGeom.Camera: {camera_path}"
        )

    # Set time code
    time = Usd.TimeCode.Default() if time_code is None else Usd.TimeCode(time_code)

    # Get camera and transforms
    cam = UsdGeom.Camera(cam_prim)
    if xform_cache is None:
        xform_cache = UsdGeom.XformCache(time)
    elif xform_cache.GetTime() != time:
        raise ValueError("xform_cache time does not match time_code")
    c2w: Gf.Matrix4d = xform_cache.GetLocalToWorldTransform(cam_prim)
    w2c: Gf.Matrix4d = c2w.GetInverse()

    # Camera attributes
    proj_token = cam.GetProjectionAttr().Get(time)
    projection = str(proj_token) if proj_token else "perspective"

    focal_length_mm = float(cam.GetFocalLengthAttr().Get(time) or 50.0)
    horiz_ap_mm = float(cam.GetHorizontalApertureAttr().Get(time) or 36.0)
    vert_ap_mm = float(cam.GetVerticalApertureAttr().Get(time) or 24.0)
    horiz_offset_mm = float(cam.GetHorizontalApertureOffsetAttr().Get(time) or 0.0)
    vert_offset_mm = float(cam.GetVerticalApertureOffsetAttr().Get(time) or 0.0)

    # Clipping range
    clipping = cam.GetClippingRangeAttr().Get(time)
    if clipping is not None and len(clipping) == 2:
        znear, zfar = float(clipping[0]), float(clipping[1])
    else:
        znear, zfar = 0.1, 10000.0

    # Compute image height if not provided
    if image_height is None:
        aspect_ratio = horiz_ap_mm / vert_ap_mm
        image_height = int(image_width / aspect_ratio)

    # Pixel aspect ratio (usually 1.0 for square pixels)
    pixel_aspect_ratio = 1.0

    # Compute intrinsic parameters
    # USD camera looks down -Z, pixel origin is top-left
    fx = focal_length_mm / horiz_ap_mm * float(image_width)
    fy = focal_length_mm / vert_ap_mm * (float(image_height) / pixel_aspect_ratio)

    # Principal point with aperture offset
    cx = (float(image_width) * 0.5) - (horiz_offset_mm / horiz_ap_mm) * float(
        image_width
    )
    cy = (float(image_height) * 0.5) - (vert_offset_mm / vert_ap_mm) * (
        float(image_height) / pixel_aspect_ratio
    )

    # Field of view in radians
    hfov = 2.0 * math.atan((horiz_ap_mm * 0.5) / max(1e-8, focal_length_mm))
    vfov = 2.0 * math.atan((vert_ap_mm * 0.5) / max(1e-8, focal_length_mm))

    # Intrinsic matrix K
    K = [
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ]

    params: dict[str, Any] = {
        "camera_path": camera_path,
        "projection": projection,
        "image_width": int(image_width),
        "image_height": int(image_height),
        "pixel_aspect_ratio": pixel_aspect_ratio,
        "near": znear,
        "far": zfar,
        "focal_length_mm": focal_length_mm,
        "horizontal_aperture_mm": horiz_ap_mm,
        "vertical_aperture_mm": vert_ap_mm,
        "horizontal_aperture_offset_mm": horiz_offset_mm,
        "vertical_aperture_offset_mm": vert_offset_mm,
        "fov_x_rad": hfov,
        "fov_y_rad": vfov,
        "K": K,
        "camera_world_transform": _matrix4_to_list(c2w),  # camera->world
        "world_to_camera": _matrix4_to_list(w2c),  # world->camera
    }

    if projection == "orthographic":
        params["is_orthographic_like"] = True

    return params


def project_point(
    world_point: tuple[float, float, float] | list[float] | np.ndarray,
    camera_params: dict[str, Any],
) -> tuple[float, float, float]:
    """
    Project a 3D world point to 2D pixel coordinates.

    Args:
        world_point: 3D point in world coordinates (x, y, z)
        camera_params: Camera parameters dict from extract_camera_parameters()

    Returns:
        Tuple of (u, v, depth) where:
            - u, v: Pixel coordinates (origin at top-left)
            - depth: Camera space Z coordinate (positive = in front)

    Example:
        >>> params = extract_camera_parameters("scene.usda", "/World/Camera", 1920, 1080)
        >>> u, v, depth = project_point([1.0, 2.0, 3.0], params)
        >>> print(f"Point projects to pixel ({u:.1f}, {v:.1f}) at depth {depth:.1f}")
    """
    # Convert to numpy for easier manipulation
    world_pt = np.array(world_point, dtype=np.float64)
    if world_pt.shape != (3,):
        raise ValueError(f"world_point must be 3D, got shape {world_pt.shape}")

    # Get world-to-camera transform
    w2c = np.array(camera_params["world_to_camera"], dtype=np.float64)

    # Transform to camera space (homogeneous coordinates)
    world_homo = np.append(world_pt, 1.0)
    cam_homo = w2c @ world_homo
    cam_pt = cam_homo[:3]

    # Extract camera coordinates
    Xc, Yc, Zc = float(cam_pt[0]), float(cam_pt[1]), float(cam_pt[2])

    # Get intrinsic parameters
    K = camera_params["K"]
    fx = float(K[0][0])
    fy = float(K[1][1])
    cx = float(K[0][2])
    cy = float(K[1][2])

    if camera_params.get("projection", "perspective") == "orthographic":
        # Orthographic projection
        u = fx * Xc + cx
        v = fy * (-Yc) + cy  # Flip Y for image coordinates
    else:
        # Perspective projection
        # USD camera looks down -Z
        if abs(Zc) < 1e-12:
            return float("nan"), float("nan"), float(Zc)

        inv_neg_Z = -1.0 / Zc
        u = fx * (Xc * inv_neg_Z) + cx
        v = fy * (-Yc * inv_neg_Z) + cy  # Flip Y for image coordinates

    return float(u), float(v), float(Zc)


def unproject_pixel(
    pixel_coord: tuple[float, float] | list[float] | np.ndarray,
    camera_params: dict[str, Any],
    depth: float | None = None,
) -> (
    tuple[tuple[float, float, float], tuple[float, float, float]]
    | tuple[float, float, float]
):
    """
    Unproject a 2D pixel to a 3D ray or point.

    Args:
        pixel_coord: 2D pixel coordinates (u, v) with origin at top-left
        camera_params: Camera parameters dict from extract_camera_parameters()
        depth: If provided, returns a 3D point at this camera-space depth.
               If None, returns ray origin and direction.

    Returns:
        If depth is None:
            Tuple of (origin, direction) where both are 3D world coordinates
        If depth is provided:
            3D world point at the specified depth

    Example:
        >>> params = extract_camera_parameters("scene.usda", "/World/Camera", 1920, 1080)
        >>> # Get ray for pixel
        >>> origin, direction = unproject_pixel([960, 540], params)
        >>> # Get 3D point at depth 5.0
        >>> point = unproject_pixel([960, 540], params, depth=5.0)
    """
    # Convert to numpy
    pixel = np.array(pixel_coord, dtype=np.float64)
    if pixel.shape != (2,):
        raise ValueError(f"pixel_coord must be 2D, got shape {pixel.shape}")

    u, v = float(pixel[0]), float(pixel[1])

    # Get camera-to-world transform
    c2w = np.array(camera_params["camera_world_transform"], dtype=np.float64)

    # Get intrinsic parameters
    K = camera_params["K"]
    fx = float(K[0][0])
    fy = float(K[1][1])
    cx = float(K[0][2])
    cy = float(K[1][2])

    if camera_params.get("projection", "perspective") == "orthographic":
        # Orthographic unprojection
        Xc = (u - cx) / fx
        Yc = -(v - cy) / fy  # Flip Y from image to camera coordinates
        origin_cam = np.array([Xc, Yc, 0.0, 1.0])
        dir_cam = np.array([0.0, 0.0, -1.0, 0.0])  # Look down -Z
    else:
        # Perspective unprojection
        x = (u - cx) / max(1e-12, fx)
        y = -(v - cy) / max(1e-12, fy)  # Flip Y from image to camera coordinates
        origin_cam = np.array([0.0, 0.0, 0.0, 1.0])
        dir_cam = np.array([x, y, -1.0, 0.0])  # Direction in camera space

    # Transform to world space
    origin_world = (c2w @ origin_cam)[:3]
    dir_world = (c2w @ dir_cam)[:3]

    # Normalize direction
    dir_norm = np.linalg.norm(dir_world)
    if dir_norm > 1e-12:
        dir_world = dir_world / dir_norm

    # Convert to tuples
    origin_tuple = tuple(float(x) for x in origin_world)
    dir_tuple = tuple(float(x) for x in dir_world)

    if depth is not None:
        # Return 3D point at specified depth
        # For perspective: depth is along ray
        # For orthographic: depth is along -Z direction
        if camera_params.get("projection", "perspective") == "orthographic":
            # Move along camera Z axis
            cam_point = np.array(
                [Xc, Yc, -depth, 1.0]
            )  # -depth because camera looks down -Z
            world_point = (c2w @ cam_point)[:3]
        else:
            # Move along ray
            world_point = origin_world + depth * dir_world

        return tuple(float(x) for x in world_point)
    else:
        # Return ray
        return origin_tuple, dir_tuple


def save_camera_json(camera_params: dict[str, Any], output_path: str | Path) -> None:
    """Save camera parameters to JSON file."""
    output_path = Path(output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(camera_params, f, indent=2)


def load_camera_json(json_path: str | Path) -> dict[str, Any]:
    """Load camera parameters from JSON file."""
    json_path = Path(json_path)
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)
