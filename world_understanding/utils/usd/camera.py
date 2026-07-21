# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
import math
from collections.abc import Iterable

from pxr import Gf, Usd, UsdGeom

from .prim import get_bbox_from_prim, traverse_prims

logger = logging.getLogger(__name__)

# DEFAULT_CAMERA_ORDERING = ["-x", "+x", "-y", "+y", "-z", "+z"]
DEFAULT_CAMERA_ORDERING = [
    "+x+y+z",
    "-x+y+z",
    "-x-y+z",
    "+x-y+z",  # All +z (northern hemisphere)
    "+x+y-z",
    "-x+y-z",
    "-x-y-z",
    "+x-y-z",  # All -z (southern hemisphere)
]

DEFAULT_CAMERA_ORDERING_ROTATION_INDICES = {
    0: [0, 1, 2, 3, 4, 5, 6, 7],
    90: [1, 2, 3, 0, 5, 6, 7, 4],
    180: [2, 3, 0, 1, 6, 7, 4, 5],
    270: [3, 0, 1, 2, 7, 4, 5, 6],
}

DEFAULT_CAMERA_ORDERING_ROTATIONS = {
    0: DEFAULT_CAMERA_ORDERING,
    90: [
        "-x+y+z",
        "-x-y+z",
        "+x-y+z",
        "+x+y+z",  # All +z (northern hemisphere)
        "-x+y-z",
        "-x-y-z",
        "+x-y-z",
        "+x+y-z",  # All -z (southern hemisphere)
    ],
    180: [
        "-x-y+z",
        "+x-y+z",
        "+x+y+z",
        "-x+y+z",  # All +z (northern hemisphere)
        "-x-y-z",
        "+x-y-z",
        "+x+y-z",
        "-x+y-z",  # All -z (southern hemisphere)
    ],
    270: [
        "+x-y+z",
        "+x+y+z",
        "-x+y+z",
        "-x-y+z",  # All +z (northern hemisphere)
        "+x-y-z",
        "+x+y-z",
        "-x+y-z",
        "-x-y-z",  # All -z (southern hemisphere)
    ],
}


def compute_camera_framing_position_sides(
    bbox_min: tuple[float, float, float],
    bbox_max: tuple[float, float, float],
    direction: str = "+x",  # One of "+x", "-x", "+y", "-y", "+z", "-z"
    margin: float = 1.0,
    min_distance: float = 0,
    focal_length: float = 60.0,
    horizontal_aperture: float = 36.0,
    vertical_aperture: float = 36.0,
    max_scene_size: float | None = None,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """
    Computes the position of a camera that fully frames a box from any cardinal
    direction (side view).

    Args:
        bbox_min (tuple): (x, y, z) minimum corner of the scene bounding box
        bbox_max (tuple): (x, y, z) maximum corner of the scene bounding box
        direction (str): Camera viewing direction, one of "+x", "-x", "+y", "-y",
            "+z", "-z"
        margin (float): Optional multiplier to slightly expand framing
        min_distance (float): Minimum distance from the scene
        focal_length (float): Camera focal length in mm
        horizontal_aperture (float): Horizontal aperture in mm
        vertical_aperture (float): Vertical aperture in mm
        max_scene_size (float): Optional maximum size limit for the scene after
            margin is applied. If set, the effective size will be capped at this value.

    Returns:
        camera_position (tuple): (x, y, z) position of the camera
        look_at_point (tuple): (x, y, z) point camera is looking at
    """
    if direction not in ["+x", "-x", "+y", "-y", "+z", "-z"]:
        raise ValueError("direction must be one of '+x', '-x', '+y', '-y', '+z', '-z'")

    # Compute scene center
    center_x = (bbox_min[0] + bbox_max[0]) / 2.0
    center_y = (bbox_min[1] + bbox_max[1]) / 2.0
    center_z = (bbox_min[2] + bbox_max[2]) / 2.0

    # Determine which dimensions to use for FoV calculations based on viewing
    # direction
    if direction in ["+x", "-x"]:
        # Looking along X axis, use Y and Z for FoV
        size1 = bbox_max[1] - bbox_min[1]  # Y dimension
        size2 = bbox_max[2] - bbox_min[2]  # Z dimension
        center1 = center_y
        center2 = center_z
    elif direction in ["+y", "-y"]:
        # Looking along Y axis, use X and Z for FoV
        size1 = bbox_max[0] - bbox_min[0]  # X dimension
        size2 = bbox_max[2] - bbox_min[2]  # Z dimension
        center1 = center_x
        center2 = center_z
    else:  # direction in ["+z", "-z"]
        # Looking along Z axis, use X and Y for FoV
        size1 = bbox_max[0] - bbox_min[0]  # X dimension
        size2 = bbox_max[1] - bbox_min[1]  # Y dimension
        center1 = center_x
        center2 = center_y

    # Apply margins
    size1 *= margin
    size2 *= margin

    # Apply max_scene_size limit if specified
    if max_scene_size is not None:
        size1 = min(size1, max_scene_size)
        size2 = min(size2, max_scene_size)

    # Calculate FoV in radians from aperture and focal length
    fov1 = 2 * math.atan(horizontal_aperture / (2 * focal_length))
    fov2 = 2 * math.atan(vertical_aperture / (2 * focal_length))

    # Compute required distance from field of view and bbox size
    required_distance1 = (size1 / 2.0) / math.tan(fov1 / 2.0)
    required_distance2 = (size2 / 2.0) / math.tan(fov2 / 2.0)

    # Take the max required distance to fit both axes
    required_distance = max(required_distance1, required_distance2)

    # Set camera position based on direction
    if direction == "+x":
        camera_x = bbox_max[0] + max(required_distance, min_distance)
        camera_y = center1
        camera_z = center2
        look_at_x = center_x
        look_at_y = center1
        look_at_z = center2
    elif direction == "-x":
        camera_x = bbox_min[0] - max(required_distance, min_distance)
        camera_y = center1
        camera_z = center2
        look_at_x = center_x
        look_at_y = center1
        look_at_z = center2
    elif direction == "+y":
        camera_x = center1
        camera_y = bbox_max[1] + max(required_distance, min_distance)
        camera_z = center2
        look_at_x = center1
        look_at_y = center_y
        look_at_z = center2
    elif direction == "-y":
        camera_x = center1
        camera_y = bbox_min[1] - max(required_distance, min_distance)
        camera_z = center2
        look_at_x = center1
        look_at_y = center_y
        look_at_z = center2
    elif direction == "+z":
        camera_x = center1
        camera_y = center2
        camera_z = bbox_min[2] - max(required_distance, min_distance)
        look_at_x = center1
        look_at_y = center2
        look_at_z = center_z
    else:  # direction == "-z"
        camera_x = center1
        camera_y = center2
        camera_z = bbox_max[2] + max(required_distance, min_distance)
        look_at_x = center1
        look_at_y = center2
        look_at_z = center_z

    return (camera_x, camera_y, camera_z), (look_at_x, look_at_y, look_at_z)


def _parse_direction_weights(direction: str) -> tuple[float, float, float]:
    """Parse a direction string into (sx, sy, sz) weight floats.

    Accepts both simple corner strings like ``"+x+y+z"`` and weighted forms
    like ``"+x-0.5y+z"`` or ``"+0.5x-1y+0.5z"``.

    Returns:
        Tuple of (sx, sy, sz) weights.
    """
    direction = direction.lower().replace(" ", "")
    weights: dict[str, float] = {}
    index = 0
    while index < len(direction):
        start = index
        if direction[index] in "+-":
            index += 1
        while index < len(direction) and (
            direction[index].isdigit() or direction[index] == "."
        ):
            index += 1
        if index < len(direction) and direction[index] in "xyz":
            val_str = direction[start:index]
            axis = direction[index]
            index += 1
            if val_str in ("", "+"):
                val = 1.0
            elif val_str == "-":
                val = -1.0
            else:
                try:
                    val = float(val_str)
                except ValueError:
                    logger.debug(
                        "Ignoring malformed camera direction weight token: %s%s",
                        val_str,
                        axis,
                    )
                    continue
            weights[axis] = val
        else:
            index = start + 1
    return weights.get("x", 1.0), weights.get("y", 1.0), weights.get("z", 1.0)


def compute_camera_framing_position_corners(
    bbox_min: tuple[float, float, float],
    bbox_max: tuple[float, float, float],
    direction: str = "+x+y+z",  # e.g. "+x+y+z", "+x-0.5y+z"
    margin: float = 1.0,
    min_distance: float = 0,
    focal_length: float = 60.0,
    horizontal_aperture: float = 36.0,
    vertical_aperture: float = 36.0,
    max_scene_size: float | None = None,
    cam_x: float | None = None,
    cam_y: float | None = None,
    cam_z: float | None = None,
    target_x: float | None = None,
    target_y: float | None = None,
    target_z: float | None = None,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """
    Computes the position of a camera that fully frames a box from a corner
    direction (corner view).

    The *direction* string supports both the 8 canonical corners (e.g.
    ``"+x+y+z"``) and arbitrary per-axis weights (e.g. ``"+x-0.5y+z"``).

    Individual camera or look-at coordinates can be overridden with the
    ``cam_x/y/z`` and ``target_x/y/z`` arguments.  When an override is
    supplied the corresponding auto-computed value is replaced.

    Args:
        bbox_min: (x, y, z) minimum corner of the scene bounding box
        bbox_max: (x, y, z) maximum corner of the scene bounding box
        direction: Camera viewing direction (e.g. "+x+y+z", "+x-0.5y+z")
        margin: Multiplier to slightly expand framing
        min_distance: Minimum distance from the scene
        focal_length: Camera focal length in mm
        horizontal_aperture: Horizontal aperture in mm
        vertical_aperture: Vertical aperture in mm
        max_scene_size: Optional maximum size limit for the scene after
            margin is applied.
        cam_x: Override camera X position (scene units)
        cam_y: Override camera Y position (scene units)
        cam_z: Override camera Z position (scene units)
        target_x: Override look-at target X position (scene units)
        target_y: Override look-at target Y position (scene units)
        target_z: Override look-at target Z position (scene units)

    Returns:
        camera_position: (x, y, z) position of the camera
        look_at_point: (x, y, z) point camera is looking at
    """
    sx, sy, sz = _parse_direction_weights(direction)

    # Compute scene center
    center_x = (bbox_min[0] + bbox_max[0]) / 2.0
    center_y = (bbox_min[1] + bbox_max[1]) / 2.0
    center_z = (bbox_min[2] + bbox_max[2]) / 2.0

    # Compute box size for FoV
    size_x = bbox_max[0] - bbox_min[0]
    size_y = bbox_max[1] - bbox_min[1]
    size_z = bbox_max[2] - bbox_min[2]

    # Apply margins
    size_x *= margin
    size_y *= margin
    size_z *= margin

    # Apply max_scene_size limit if specified
    if max_scene_size is not None:
        size_x = min(size_x, max_scene_size)
        size_y = min(size_y, max_scene_size)
        size_z = min(size_z, max_scene_size)

    # Calculate FoV in radians from aperture and focal length
    fov_x = 2 * math.atan(horizontal_aperture / (2 * focal_length))
    fov_y = 2 * math.atan(vertical_aperture / (2 * focal_length))

    # Compute the bounding "cube" that encompasses the bounding box of the target
    cube_size = max(size_x, size_y, size_z)
    half = cube_size / 2.0

    # Project the box diagonal onto the image plane
    cube_diag = cube_size * math.sqrt(3)
    proj_diag = cube_diag / math.sqrt(2)

    # Compute required distance so that the projected diagonal fits in the FoV
    min_fov = min(fov_x, fov_y)
    required_distance = (proj_diag / 2.0) / math.tan(min_fov / 2.0)
    required_distance = max(required_distance, min_distance)

    # Place the camera along the weighted direction vector from center
    x = center_x + sx * half
    y = center_y + sy * half
    z = center_z + sz * half

    # Move the camera out from the corner along the vector from center to corner
    corner_vec = [x - center_x, y - center_y, z - center_z]
    norm = math.sqrt(sum(c**2 for c in corner_vec))
    if norm == 0:
        raise ValueError("Bounding box is degenerate (zero size)")
    unit_vec = [c / norm for c in corner_vec]
    camera_x = x + unit_vec[0] * required_distance
    camera_y = y + unit_vec[1] * required_distance
    camera_z = z + unit_vec[2] * required_distance

    # Apply per-axis overrides
    if cam_x is not None:
        camera_x = cam_x
    if cam_y is not None:
        camera_y = cam_y
    if cam_z is not None:
        camera_z = cam_z

    look_x = target_x if target_x is not None else center_x
    look_y = target_y if target_y is not None else center_y
    look_z = target_z if target_z is not None else center_z

    return (camera_x, camera_y, camera_z), (look_x, look_y, look_z)


def _setup_side_view_camera(
    stage: Usd.Stage,
    camera_path: str,
    bbox_min: Gf.Vec3d,
    bbox_max: Gf.Vec3d,
    direction: str,
    margin: float,
    min_distance: float,
    focal_length: float,
    horizontal_aperture: float,
    vertical_aperture: float,
    near_clip_margin: float = 0.01,
    far_clip_margin: float = 0.01,
    near_clip: float | None = None,
    far_clip: float | None = None,
    max_scene_size: float | None = None,
    time: Usd.TimeCode = Usd.TimeCode.Default(),
) -> UsdGeom.Camera:
    """Internal helper to set up a camera with the given parameters.

    This function encapsulates the common camera setup logic used by both
    add_side_view_camera and add_focused_side_view_camera.

    Args:
        stage: The USD stage
        camera_path: Path where the camera prim will be created
        bbox_min: Minimum corner of the bounding box to frame
        bbox_max: Maximum corner of the bounding box to frame
        direction: Camera viewing direction, one of "+x", "-x", "+y", "-y",
            "+z", "-z"
        margin: Margin multiplier for framing
        min_distance: Minimum distance for camera placement
        focal_length: Camera focal length in mm
        horizontal_aperture: Camera horizontal aperture in mm
        vertical_aperture: Camera vertical aperture in mm
        near_clip: Near clipping plane distance (computed if None)
        far_clip: Far clipping plane distance (computed if None)
        max_scene_size: Optional maximum size limit for the scene after
            margin is applied. If set, the effective size will be capped at this value.
        time: Time at which to set the camera attributes (default: Default)
    Returns:
        The camera prim
    """
    # Create a camera prim if it doesn't exist
    camera_exists = stage.GetPrimAtPath(camera_path)
    if not camera_exists:
        camera_prim = UsdGeom.Camera.Define(stage, camera_path)
    else:
        camera_prim = UsdGeom.Camera(stage.GetPrimAtPath(camera_path))

    # Set camera attributes
    if not camera_exists:
        camera_prim.CreateFocalLengthAttr(focal_length)
        camera_prim.CreateHorizontalApertureAttr(horizontal_aperture)
        camera_prim.CreateVerticalApertureAttr(vertical_aperture)
    else:
        camera_prim.GetFocalLengthAttr().Set(focal_length, time=time)
        camera_prim.GetHorizontalApertureAttr().Set(horizontal_aperture, time=time)
        camera_prim.GetVerticalApertureAttr().Set(vertical_aperture, time=time)

    # Use the helper function to calculate camera position
    camera_position, look_at_point = compute_camera_framing_position_sides(
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        direction=direction,
        margin=margin,
        min_distance=min_distance,
        focal_length=focal_length,
        horizontal_aperture=horizontal_aperture,
        vertical_aperture=vertical_aperture,
        max_scene_size=max_scene_size,
    )

    # Create a transform op for the camera
    if not camera_exists:
        cam_prim = UsdGeom.Xformable(camera_prim)
        xform_op = cam_prim.AddTransformOp()
    else:
        cam_prim = UsdGeom.Xformable(camera_prim)
        xform_op = cam_prim.GetTransformOp()

    # Set the camera transform
    cam_pos_gf = Gf.Vec3d(*camera_position)  # Unpack tuple into Vec3d
    look_at_gf = Gf.Vec3d(*look_at_point)  # Unpack look_at point into Vec3d

    # Calculate forward direction (z-axis) - camera looks down negative z-axis in USD
    forward = (cam_pos_gf - look_at_gf).GetNormalized()

    # Get the stage's up axis setting
    stage_up_axis = UsdGeom.GetStageUpAxis(stage)

    # Choose up vector based on stage's up axis and viewing direction
    if stage_up_axis == UsdGeom.Tokens.y:
        stage_up_vec = Gf.Vec3d(0, 1, 0)  # Y-up
        fallback_up_vec = Gf.Vec3d(0, 0, 1)  # Z as fallback
    else:  # Z-up (default)
        stage_up_vec = Gf.Vec3d(0, 0, 1)  # Z-up
        fallback_up_vec = Gf.Vec3d(0, 1, 0)  # Y as fallback

    # Use stage up vector, but switch to fallback if parallel to forward
    # (i.e., when looking along the up axis)
    up = stage_up_vec
    if direction in ["+z", "-z"] and stage_up_axis == UsdGeom.Tokens.z:
        # Looking along Z-up axis, use fallback
        up = fallback_up_vec
    elif direction in ["+y", "-y"] and stage_up_axis == UsdGeom.Tokens.y:
        # Looking along Y-up axis, use fallback
        up = fallback_up_vec

    # Calculate right vector (x-axis) via cross product
    right = Gf.Cross(up, forward).GetNormalized()

    # Recalculate up vector (y-axis) to ensure orthogonality
    up = Gf.Cross(forward, right).GetNormalized()

    # Create rotation matrix from these basis vectors
    rotation = Gf.Matrix4d(
        right[0],
        right[1],
        right[2],
        0,
        up[0],
        up[1],
        up[2],
        0,
        forward[0],
        forward[1],
        forward[2],
        0,
        0,
        0,
        0,
        1,
    )

    # Create translation matrix
    translation = Gf.Matrix4d().SetTranslate(cam_pos_gf)

    # Combine rotation and translation
    matrix = rotation * translation
    xform_op.Set(matrix, time=time)

    # Calculate clipping planes based on final position
    # User overrides take precedence if provided
    near_clip_final = near_clip
    far_clip_final = far_clip

    if near_clip_final is None or far_clip_final is None:
        # Project all 8 bbox corners onto the camera view axis and use
        # the min/max signed depth as the clipping range. USD clipping
        # planes are measured along the camera's forward axis, NOT
        # Euclidean distance — corners off-axis sit closer along the
        # view ray than their straight-line distance, so Euclidean
        # distance can place ``near`` past the actual scene front and
        # clip foreground geometry. (Example: a unit cube viewed from
        # ``+x`` at the default framing has Euclidean ``min ≈ 3.58``
        # while the front face is only ``3.33`` units down the view
        # axis.) This works for any direction — single-axis
        # (``+x``/``-z``) and compound corner directions
        # (``+x+y+z``, ``-x-0.5y+z``, …) alike. The previous single-axis
        # if/elif chain silently fell through for compound directions,
        # leaving ``dist_to_front``/``dist_to_back`` at stale loop-
        # iteration values that downstream guard-clamped into a thin
        # slab — an invisible bug that broke every corner-camera scene.
        cx, cy, cz = (
            float(camera_position[0]),
            float(camera_position[1]),
            float(camera_position[2]),
        )
        # ``forward`` is the camera's +Z basis (cam_pos - look_at,
        # normalized); USD cameras look along -Z, so the view direction
        # (camera into scene) is ``-forward``. Signed depth along view =
        # dot(corner - cam_pos, -forward).
        view_x = -float(forward[0])
        view_y = -float(forward[1])
        view_z = -float(forward[2])
        corner_depths: list[float] = []
        for x in (float(bbox_min[0]), float(bbox_max[0])):
            for y in (float(bbox_min[1]), float(bbox_max[1])):
                for z in (float(bbox_min[2]), float(bbox_max[2])):
                    dx = x - cx
                    dy = y - cy
                    dz = z - cz
                    corner_depths.append(dx * view_x + dy * view_y + dz * view_z)
        dist_to_front = max(1e-6, min(corner_depths))
        dist_to_back = max(dist_to_front + 1e-6, max(corner_depths))

        if near_clip_final is None:
            # Near plane slightly closer than scene front
            near_clip_final = max(0.01, dist_to_front * (1.0 - near_clip_margin))

        if far_clip_final is None:
            # Far plane slightly farther than scene back
            far_clip_final = dist_to_back * (1.0 + far_clip_margin)

    # Ensure near < far
    if near_clip_final >= far_clip_final:
        # Add 10% or 0.01, whichever is larger
        near_adjust = max(0.01, abs(near_clip_final) * 0.1)
        far_clip_final = near_clip_final + near_adjust

    # Set the final clipping range
    if not camera_exists:
        camera_prim.CreateClippingRangeAttr(Gf.Vec2f(near_clip_final, far_clip_final))
        camera_prim.GetClippingRangeAttr().Set(
            Gf.Vec2f(near_clip_final, far_clip_final), time=time
        )
    else:
        camera_prim.GetClippingRangeAttr().Set(
            Gf.Vec2f(near_clip_final, far_clip_final), time=time
        )

    return camera_prim


def add_side_view_camera(
    stage: Usd.Stage,
    camera_path: str = "/Cameras/SideViewCamera",
    direction: str = "-z",  # One of "+x", "-x", "+y", "-y", "+z", "-z"
    margin: float = 1.0,
    min_distance: float = 0,
    focal_length: float = 60.0,
    horizontal_aperture: float = 36.0,
    vertical_aperture: float = 36.0,
    near_clip_margin: float = 0.01,
    far_clip_margin: float = 0.01,
    near_clip: float | None = None,
    far_clip: float | None = None,
    max_scene_size: float | None = None,
    time: Usd.TimeCode = Usd.TimeCode.Default(),
    included_purposes: Iterable[str] | None = None,
) -> UsdGeom.Camera:
    """Create a camera looking at the scene from any cardinal direction.

    This function creates a camera positioned around the scene looking at it from
    the specified direction. The camera's distance is automatically adjusted to
    ensure the entire scene is visible based on the scene's bounding box.

    Args:
        stage: The USD stage containing the scene
        camera_path: Path of the camera prim (default: "/Cameras/SideViewCamera")
        direction: Camera viewing direction, one of "+x", "-x", "+y", "-y",
            "+z", "-z" (default: "-z" for top-down view)
        margin: Margin multiplier applied to ensure the entire scene is visible
            (default: 1.0)
        min_distance: Minimum distance between scene and camera position
            (default: 0)
        focal_length: Camera focal length in mm (default: 60.0)
        horizontal_aperture: Camera horizontal aperture in mm (default: 36.0)
        vertical_aperture: Camera vertical aperture in mm (default: 36.0)
        near_clip: Near clipping plane distance (default: None, computed)
        far_clip: Far clipping plane distance (default: None, computed)
        max_scene_size: Optional maximum size limit for the scene after
            margin is applied. If set, the effective size will be capped at this value.
        time: Time at which to set the camera attributes (default: Default)
        included_purposes: Optional UsdGeom purposes included in scene framing.
            Defaults to ``default`` only.

    Returns:
        The camera prim
    """
    # Compute the bounding box of the scene.
    scene_bbox = get_bbox_from_prim(
        stage.GetPseudoRoot(),
        time=time,
        included_purposes=included_purposes,
    )
    aligned_range = scene_bbox.ComputeAlignedRange()

    # Get the bounding box extents
    bbox_min = aligned_range.GetMin()
    bbox_max = aligned_range.GetMax()

    # Use the helper function to set up the camera
    # distance parameter is incorporated in min_distance

    return _setup_side_view_camera(
        stage=stage,
        camera_path=camera_path,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        direction=direction,
        margin=margin,
        min_distance=min_distance,
        focal_length=focal_length,
        horizontal_aperture=horizontal_aperture,
        vertical_aperture=vertical_aperture,
        near_clip_margin=near_clip_margin,
        far_clip_margin=far_clip_margin,
        near_clip=near_clip,
        far_clip=far_clip,
        max_scene_size=max_scene_size,
        time=time,
    )


def add_focused_side_view_camera(
    prim_to_focus: Usd.Prim,
    camera_path: str = "/Cameras/FocusedSideViewCamera",
    direction: str = "-z",  # One of "+x", "-x", "+y", "-y", "+z", "-z"
    margin: float = 1.0,
    min_distance: float = 0,
    focal_length: float = 60.0,
    horizontal_aperture: float = 36.0,
    vertical_aperture: float = 36.0,
    near_clip_margin: float = 0.01,
    far_clip_margin: float = 0.01,
    near_clip: float | None = None,
    far_clip: float | None = None,
    max_scene_size: float | None = None,
    time: Usd.TimeCode = Usd.TimeCode.Default(),
    included_purposes: Iterable[str] | None = None,
) -> UsdGeom.Camera:
    """Create a camera focused on a specific prim from any cardinal direction.

    This function creates a camera positioned to frame a specific prim from the
    specified direction. The camera's distance is automatically adjusted to
    ensure the prim is fully visible based on its bounding box.

    Args:
        prim_to_focus: The USD prim to focus the camera on
        camera_path: Path of the camera prim (default: "/Cameras/FocusedSideViewCamera")
        direction: Camera viewing direction, one of "+x", "-x", "+y", "-y",
            "+z", "-z" (default: "-z" for top-down view)
        margin: Margin multiplier applied to ensure the prim is fully visible
            (default: 1.0)
        min_distance: Minimum distance between prim and camera position
            (default: 0)
        focal_length: Camera focal length in mm (default: 60.0)
        horizontal_aperture: Camera horizontal aperture in mm (default: 36.0)
        vertical_aperture: Camera vertical aperture in mm (default: 36.0)
        near_clip: Near clipping plane distance (default: None, computed)
        far_clip: Far clipping plane distance (default: None, computed)
        max_scene_size: Optional maximum size limit for the scene after
            margin is applied. If set, the effective size will be capped at this value.
        time: Time at which to set the camera attributes (default: Default)
        included_purposes: Optional UsdGeom purposes included in prim framing.
            Defaults to ``default`` only.

    Returns:
        The camera prim
    """
    # Get the stage from the prim
    stage = prim_to_focus.GetStage()

    # Get the bounding box of the prim to focus on
    bbox = get_bbox_from_prim(
        prim_to_focus,
        time=time,
        included_purposes=included_purposes,
    )
    aligned_range = bbox.ComputeAlignedRange()

    # Get the bounding box extents
    bbox_min = aligned_range.GetMin()
    bbox_max = aligned_range.GetMax()

    # Use the helper function to set up the camera

    return _setup_side_view_camera(
        stage=stage,
        camera_path=camera_path,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        direction=direction,
        margin=margin,
        min_distance=min_distance,
        focal_length=focal_length,
        horizontal_aperture=horizontal_aperture,
        vertical_aperture=vertical_aperture,
        near_clip_margin=near_clip_margin,
        far_clip_margin=far_clip_margin,
        near_clip=near_clip,
        far_clip=far_clip,
        max_scene_size=max_scene_size,
        time=time,
    )


def print_camera_prim_info(camera_prim: UsdGeom.Camera) -> None:
    """
    Print information about a camera prim.

    This function prints attribute values of a camera prim including focal length,
    aperture settings, clipping range, projection type, and camera position.

    Args:
        camera_prim: The USD camera prim
    """
    # Get camera position from its transformation matrix
    xformable = UsdGeom.Xformable(camera_prim)
    transform_matrix = xformable.GetLocalTransformation()
    position = transform_matrix.ExtractTranslation()
    print(f"  Position: ({position[0]}, {position[1]}, {position[2]})")

    # Extract rotation as forward, up, right vectors
    # Forward is the negative Z axis (3rd column) of the rotation part of the matrix
    forward = Gf.Vec3d(
        -transform_matrix[0][2], -transform_matrix[1][2], -transform_matrix[2][2]
    ).GetNormalized()
    # Up is the Y axis (2nd column) of the rotation part of the matrix
    up = Gf.Vec3d(
        transform_matrix[0][1], transform_matrix[1][1], transform_matrix[2][1]
    ).GetNormalized()
    # Right is the X axis (1st column) of the rotation part of the matrix
    right = Gf.Vec3d(
        transform_matrix[0][0], transform_matrix[1][0], transform_matrix[2][0]
    ).GetNormalized()

    print("  Rotation:")
    print(f"    Forward: ({forward[0]:.4f}, {forward[1]:.4f}, {forward[2]:.4f})")
    print(f"    Up: ({up[0]:.4f}, {up[1]:.4f}, {up[2]:.4f})")
    print(f"    Right: ({right[0]:.4f}, {right[1]:.4f}, {right[2]:.4f})")


def _setup_corner_view_camera(
    stage: Usd.Stage,
    camera_path: str,
    bbox_min: Gf.Vec3d,
    bbox_max: Gf.Vec3d,
    direction: str,
    margin: float,
    min_distance: float,
    focal_length: float,
    horizontal_aperture: float,
    vertical_aperture: float,
    near_clip_margin: float = 0.01,
    far_clip_margin: float = 0.01,
    near_clip: float | None = None,
    far_clip: float | None = None,
    max_scene_size: float | None = None,
    time: Usd.TimeCode = Usd.TimeCode.Default(),
    cam_x: float | None = None,
    cam_y: float | None = None,
    cam_z: float | None = None,
    target_x: float | None = None,
    target_y: float | None = None,
    target_z: float | None = None,
) -> UsdGeom.Camera:
    """Internal helper to set up a camera at a corner view.

    Args:
        stage: The USD stage
        camera_path: Path where the camera prim will be created
        bbox_min: Minimum corner of the bounding box to frame
        bbox_max: Maximum corner of the bounding box to frame
        direction: Camera viewing direction, one of the 8 corners
        margin: Margin multiplier for framing
        min_distance: Minimum distance for camera placement
        focal_length: Camera focal length in mm
        horizontal_aperture: Camera horizontal aperture in mm
        vertical_aperture: Camera vertical aperture in mm
        near_clip: Near clipping plane distance (computed if None)
        far_clip: Far clipping plane distance (computed if None)
        max_scene_size: Optional maximum size limit for the scene after
            margin is applied. If set, the effective size will be capped at this value.
        time: Time at which to set the camera attributes (default: Default)
    Returns:
        The camera prim
    """
    # Create a camera prim if it doesn't exist
    camera_exists = stage.GetPrimAtPath(camera_path)
    if not camera_exists:
        camera_prim = UsdGeom.Camera.Define(stage, camera_path)
    else:
        camera_prim = UsdGeom.Camera(stage.GetPrimAtPath(camera_path))

    # Set camera attributes
    if not camera_exists:
        camera_prim.CreateFocalLengthAttr(focal_length)
        camera_prim.CreateHorizontalApertureAttr(horizontal_aperture)
        camera_prim.CreateVerticalApertureAttr(vertical_aperture)
    else:
        camera_prim.GetFocalLengthAttr().Set(focal_length, time=time)
        camera_prim.GetHorizontalApertureAttr().Set(horizontal_aperture, time=time)
        camera_prim.GetVerticalApertureAttr().Set(vertical_aperture, time=time)

    # Use the helper function to calculate camera position (corner view)
    camera_position, look_at_point = compute_camera_framing_position_corners(
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        direction=direction,
        margin=margin,
        min_distance=min_distance,
        focal_length=focal_length,
        horizontal_aperture=horizontal_aperture,
        vertical_aperture=vertical_aperture,
        max_scene_size=max_scene_size,
        cam_x=cam_x,
        cam_y=cam_y,
        cam_z=cam_z,
        target_x=target_x,
        target_y=target_y,
        target_z=target_z,
    )

    # Create a transform op for the camera
    if not camera_exists:
        cam_prim = UsdGeom.Xformable(camera_prim)
        xform_op = cam_prim.AddTransformOp()
    else:
        cam_prim = UsdGeom.Xformable(camera_prim)
        xform_op = cam_prim.GetTransformOp()

    # Set the camera transform
    cam_pos_gf = Gf.Vec3d(*camera_position)  # Unpack tuple into Vec3d
    look_at_gf = Gf.Vec3d(*look_at_point)  # Unpack look_at point into Vec3d

    # Calculate forward direction (z-axis) - camera looks down negative z-axis in USD
    forward = (cam_pos_gf - look_at_gf).GetNormalized()

    # Get the stage's up axis setting
    stage_up_axis = UsdGeom.GetStageUpAxis(stage)

    # Choose up vector based on stage's up axis
    if stage_up_axis == UsdGeom.Tokens.y:
        up = Gf.Vec3d(0, 1, 0)  # Y-up
        fallback_up = Gf.Vec3d(0, 0, 1)  # Z as fallback
    else:  # Z-up (default)
        up = Gf.Vec3d(0, 0, 1)  # Z-up
        fallback_up = Gf.Vec3d(0, 1, 0)  # Y as fallback

    # If forward is parallel to up (looking along the up axis), use fallback
    if abs(forward[0]) < 1e-6 and abs(forward[1]) < 1e-6:
        # Forward is along Z, so if up is also Z, use fallback
        if stage_up_axis != UsdGeom.Tokens.y:
            up = fallback_up
    elif abs(forward[0]) < 1e-6 and abs(forward[2]) < 1e-6:
        # Forward is along Y, so if up is also Y, use fallback
        if stage_up_axis == UsdGeom.Tokens.y:
            up = fallback_up

    # Calculate right vector (x-axis) via cross product
    right = Gf.Cross(up, forward).GetNormalized()

    # Recalculate up vector (y-axis) to ensure orthogonality
    up = Gf.Cross(forward, right).GetNormalized()

    # Create rotation matrix from these basis vectors
    rotation = Gf.Matrix4d(
        right[0],
        right[1],
        right[2],
        0,
        up[0],
        up[1],
        up[2],
        0,
        forward[0],
        forward[1],
        forward[2],
        0,
        0,
        0,
        0,
        1,
    )

    # Create translation matrix
    translation = Gf.Matrix4d().SetTranslate(cam_pos_gf)

    # Combine rotation and translation
    matrix = rotation * translation
    xform_op.Set(matrix, time=time)

    # Calculate clipping planes based on final position
    near_clip_final = near_clip
    far_clip_final = far_clip

    if near_clip_final is None or far_clip_final is None:
        # Project bbox corners onto the camera view axis (NOT Euclidean
        # distance). USD clipping planes are measured along the camera's
        # forward axis; off-axis corners that sit at larger Euclidean
        # distance still need to fall within [near, far] along view, and
        # using Euclidean distance for ``near`` can place it past actual
        # scene-front geometry. View direction (camera into scene) is
        # ``-forward`` since USD cameras look along -Z.
        view_x = -float(forward[0])
        view_y = -float(forward[1])
        view_z = -float(forward[2])
        cx = float(cam_pos_gf[0])
        cy = float(cam_pos_gf[1])
        cz = float(cam_pos_gf[2])
        depths = [
            (corner[0] - cx) * view_x
            + (corner[1] - cy) * view_y
            + (corner[2] - cz) * view_z
            for corner in [
                (bbox_min[0], bbox_min[1], bbox_min[2]),
                (bbox_min[0], bbox_min[1], bbox_max[2]),
                (bbox_min[0], bbox_max[1], bbox_min[2]),
                (bbox_min[0], bbox_max[1], bbox_max[2]),
                (bbox_max[0], bbox_min[1], bbox_min[2]),
                (bbox_max[0], bbox_min[1], bbox_max[2]),
                (bbox_max[0], bbox_max[1], bbox_min[2]),
                (bbox_max[0], bbox_max[1], bbox_max[2]),
            ]
        ]
        min_depth = min(depths)
        max_depth = max(depths)
        if near_clip_final is None:
            near_clip_final = max(0.01, min_depth * (1.0 - near_clip_margin))
        if far_clip_final is None:
            far_clip_final = max_depth * (1.0 + far_clip_margin)

    # Ensure near < far
    if near_clip_final >= far_clip_final:
        near_adjust = max(0.01, abs(near_clip_final) * 0.1)
        far_clip_final = near_clip_final + near_adjust

    # Set the final clipping range
    if not camera_exists:
        camera_prim.CreateClippingRangeAttr(Gf.Vec2f(near_clip_final, far_clip_final))
        camera_prim.GetClippingRangeAttr().Set(
            Gf.Vec2f(near_clip_final, far_clip_final), time=time
        )
    else:
        camera_prim.GetClippingRangeAttr().Set(
            Gf.Vec2f(near_clip_final, far_clip_final), time=time
        )

    return camera_prim


def add_corner_view_camera(
    stage: Usd.Stage,
    camera_path: str = "/Cameras/CornerViewCamera",
    direction: str = "+x+y+z",  # e.g. "+x+y+z", "+x-0.5y+z"
    margin: float = 1.0,
    min_distance: float = 0,
    focal_length: float = 60.0,
    horizontal_aperture: float = 36.0,
    vertical_aperture: float = 36.0,
    near_clip_margin: float = 0.01,
    far_clip_margin: float = 0.01,
    near_clip: float | None = None,
    far_clip: float | None = None,
    max_scene_size: float | None = None,
    time: Usd.TimeCode = Usd.TimeCode.Default(),
    cam_x: float | None = None,
    cam_y: float | None = None,
    cam_z: float | None = None,
    target_x: float | None = None,
    target_y: float | None = None,
    target_z: float | None = None,
    included_purposes: Iterable[str] | None = None,
) -> UsdGeom.Camera:
    """Create a camera looking at the scene from any corner direction.

    This function creates a camera positioned at a corner of the scene looking at
    its center. The camera's distance is automatically adjusted to ensure the
    entire scene is visible based on the scene's bounding box.

    Args:
        stage: The USD stage containing the scene
        camera_path: Path of the camera prim (default: "/Cameras/CornerViewCamera")
        direction: Camera viewing direction, one of the 8 corners
        margin: Margin multiplier applied to ensure the entire scene is visible
        min_distance: Minimum distance between scene and camera position
        focal_length: Camera focal length in mm (default: 60.0)
        horizontal_aperture: Camera horizontal aperture in mm (default: 36.0)
        vertical_aperture: Camera vertical aperture in mm (default: 36.0)
        near_clip: Near clipping plane distance (default: None, computed)
        far_clip: Far clipping plane distance (default: None, computed)
        max_scene_size: Optional maximum size limit for the scene after
            margin is applied. If set, the effective size will be capped at this value.
        time: Time at which to set the camera attributes (default: Default)
        included_purposes: Optional UsdGeom purposes included in scene framing.
            Defaults to ``default`` only.

    Returns:
        The camera prim
    """
    scene_bbox = get_bbox_from_prim(
        stage.GetPseudoRoot(),
        time=time,
        included_purposes=included_purposes,
    )
    aligned_range = scene_bbox.ComputeAlignedRange()
    bbox_min = aligned_range.GetMin()
    bbox_max = aligned_range.GetMax()

    return _setup_corner_view_camera(
        stage=stage,
        camera_path=camera_path,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        direction=direction,
        margin=margin,
        min_distance=min_distance,
        focal_length=focal_length,
        horizontal_aperture=horizontal_aperture,
        vertical_aperture=vertical_aperture,
        near_clip_margin=near_clip_margin,
        far_clip_margin=far_clip_margin,
        near_clip=near_clip,
        far_clip=far_clip,
        max_scene_size=max_scene_size,
        time=time,
        cam_x=cam_x,
        cam_y=cam_y,
        cam_z=cam_z,
        target_x=target_x,
        target_y=target_y,
        target_z=target_z,
    )


def add_focused_corner_view_camera(
    prim_to_focus: Usd.Prim,
    camera_path: str = "/Cameras/FocusedCornerViewCamera",
    direction: str = "+x+y+z",  # e.g. "+x+y+z", "+x-0.5y+z"
    margin: float = 1.0,
    min_distance: float = 0,
    focal_length: float = 60.0,
    horizontal_aperture: float = 36.0,
    vertical_aperture: float = 36.0,
    near_clip_margin: float = 0.01,
    far_clip_margin: float = 0.01,
    near_clip: float | None = None,
    far_clip: float | None = None,
    max_scene_size: float | None = None,
    time: Usd.TimeCode = Usd.TimeCode.Default(),
    cam_x: float | None = None,
    cam_y: float | None = None,
    cam_z: float | None = None,
    target_x: float | None = None,
    target_y: float | None = None,
    target_z: float | None = None,
    included_purposes: Iterable[str] | None = None,
) -> UsdGeom.Camera:
    """Create a camera focused on a specific prim from any corner direction.

    This function creates a camera positioned at a corner of the bounding box of
    the given prim, looking at its center. The camera's distance is automatically
    adjusted to ensure the prim is fully visible based on its bounding box.

    Args:
        prim_to_focus: The USD prim to focus the camera on
        camera_path: Path of the camera prim (default: "/Cameras/FocusedCornerViewCamera")
        direction: Camera viewing direction, one of the 8 corners
        distance: Minimum distance to position the camera (default: 100)
        margin: Margin multiplier applied to ensure the prim is fully visible
        min_distance: Minimum distance between prim and camera position
        focal_length: Camera focal length in mm (default: 60.0)
        horizontal_aperture: Camera horizontal aperture in mm (default: 36.0)
        vertical_aperture: Camera vertical aperture in mm (default: 36.0)
        near_clip: Near clipping plane distance (default: None, computed)
        far_clip: Far clipping plane distance (default: None, computed)
        max_scene_size: Optional maximum size limit for the scene after
            margin is applied. If set, the effective size will be capped at this value.
        time: Time at which to set the camera attributes (default: Default)
        included_purposes: Optional UsdGeom purposes included in prim framing.
            Defaults to ``default`` only.

    Returns:
        The camera prim
    """
    stage = prim_to_focus.GetStage()

    bbox = get_bbox_from_prim(
        prim_to_focus,
        time=time,
        included_purposes=included_purposes,
    )
    aligned_range = bbox.ComputeAlignedRange()
    bbox_min = aligned_range.GetMin()
    bbox_max = aligned_range.GetMax()

    return _setup_corner_view_camera(
        stage=stage,
        camera_path=camera_path,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        direction=direction,
        margin=margin,
        min_distance=min_distance,
        focal_length=focal_length,
        horizontal_aperture=horizontal_aperture,
        vertical_aperture=vertical_aperture,
        near_clip_margin=near_clip_margin,
        far_clip_margin=far_clip_margin,
        near_clip=near_clip,
        far_clip=far_clip,
        max_scene_size=max_scene_size,
        time=time,
        cam_x=cam_x,
        cam_y=cam_y,
        cam_z=cam_z,
        target_x=target_x,
        target_y=target_y,
        target_z=target_z,
    )


def get_all_cameras(stage: Usd.Stage) -> list[UsdGeom.Camera]:
    """Get all cameras in the stage."""
    return [
        UsdGeom.Camera(prim)
        for prim in traverse_prims(stage)
        if prim.IsA(UsdGeom.Camera)
    ]


def get_all_camera_paths(stage: Usd.Stage) -> list[str]:
    """Get all camera paths in the stage."""
    return [
        prim.GetPath() for prim in traverse_prims(stage) if prim.IsA(UsdGeom.Camera)
    ]
