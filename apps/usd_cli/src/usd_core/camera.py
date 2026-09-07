# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Camera placement + projection (PR-6) — pure pxr + numpy, runs anywhere.

Vendored/adapted from world-understanding (Q2 decision: copy minimal code):
- look-at matrix + clipping: `world_understanding/utils/usd/camera.py`
- intrinsics / projection: `world_understanding/functions/graphics/usd_camera.py`

Fit uses a direction-independent bounding-sphere framing so `orbit` can place the camera
at any angle and still frame the scene — the "3DAL adaptive camera" intent.
"""

from __future__ import annotations

import math

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom

DEFAULT_FOCAL = 60.0
DEFAULT_APERTURE = 36.0


def camera_lens_mm_to_raw(stage: Usd.Stage, value_mm: float) -> float:
    """Convert documented millimetres to USD tenths-of-a-scene-unit."""

    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    value = float(value_mm)
    if (
        not math.isfinite(meters_per_unit)
        or meters_per_unit <= 0.0
        or not math.isfinite(value)
    ):
        raise ValueError("camera lens units and millimetre value must be finite")
    return value / (meters_per_unit * 100.0)


# ── up-axis helpers ────────────────────────────────────────────────────────────
def _up_vec(stage: Usd.Stage) -> Gf.Vec3d:
    return Gf.Vec3d(0, 1, 0) if UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.y \
        else Gf.Vec3d(0, 0, 1)


# ── look-at (Gram-Schmidt, respects stage up axis) ──────────────────────────────
def look_at_matrix(cam_pos, look_at, up: Gf.Vec3d) -> Gf.Matrix4d:
    """Camera-to-world transform with -Z pointing at `look_at`. Adapted verbatim from
    utils/usd/camera.py:444-498."""
    cam_pos_gf = Gf.Vec3d(*cam_pos)
    look_at_gf = Gf.Vec3d(*look_at)
    forward = (cam_pos_gf - look_at_gf).GetNormalized()  # camera's +Z (looks down -Z)

    # Guard against forward parallel to up (looking straight along the up axis).
    if abs(Gf.Dot(forward, up.GetNormalized())) > 0.999:
        up = Gf.Vec3d(0, 0, 1) if abs(up[1]) > 0.5 else Gf.Vec3d(0, 1, 0)

    right = Gf.Cross(up, forward).GetNormalized()
    up_ortho = Gf.Cross(forward, right).GetNormalized()
    rotation = Gf.Matrix4d(
        right[0], right[1], right[2], 0,
        up_ortho[0], up_ortho[1], up_ortho[2], 0,
        forward[0], forward[1], forward[2], 0,
        0, 0, 0, 1,
    )
    return rotation * Gf.Matrix4d().SetTranslate(cam_pos_gf)


# ── orbit (spherical around a target) ───────────────────────────────────────────
def orbit_position(target, distance: float, azimuth_deg: float, elevation_deg: float,
                   up_axis_y: bool = True):
    """Position on a sphere of `distance` around `target`. azimuth 0 = facing the target
    from +Z (Y-up) / +X (Z-up); elevation lifts toward the up axis."""
    az, el = math.radians(azimuth_deg), math.radians(max(-89.0, min(89.0, elevation_deg)))
    h = math.cos(el)
    if up_axis_y:
        d = (h * math.sin(az), math.sin(el), h * math.cos(az))
    else:  # Z-up
        d = (h * math.cos(az), h * math.sin(az), math.sin(el))
    return (target[0] + d[0] * distance, target[1] + d[1] * distance, target[2] + d[2] * distance)


# ── fit distance (bounding sphere, direction-independent) ───────────────────────
def fit_distance(size, focal: float = DEFAULT_FOCAL, h_ap: float = DEFAULT_APERTURE,
                 v_ap: float = DEFAULT_APERTURE, margin: float = 1.15) -> float:
    """Distance at which a bounding sphere of the given size fills the frame.

    fov from aperture/focal (utils/usd/camera.py:137-138); distance = R / sin(fov/2),
    using the tighter axis so nothing is clipped from any orbit angle."""
    radius = 0.5 * math.sqrt(sum(s * s for s in size)) * margin
    fov = min(2 * math.atan(h_ap / (2 * focal)), 2 * math.atan(v_ap / (2 * focal)))
    return max(radius / max(1e-6, math.sin(fov / 2)), 1e-3)


def clipping_range(bbox_min, bbox_max, cam_pos, look_at, margin: float = 0.1):
    """Near/far along the view axis from the 8 bbox corners (utils/usd/camera.py:523-552)."""
    forward = (Gf.Vec3d(*cam_pos) - Gf.Vec3d(*look_at)).GetNormalized()
    view = (-forward[0], -forward[1], -forward[2])  # USD cameras look down -Z
    depths = []
    for x in (bbox_min[0], bbox_max[0]):
        for y in (bbox_min[1], bbox_max[1]):
            for z in (bbox_min[2], bbox_max[2]):
                d = (x - cam_pos[0], y - cam_pos[1], z - cam_pos[2])
                depths.append(d[0] * view[0] + d[1] * view[1] + d[2] * view[2])
    # Keep the near plane in the same stage-space scale as the fitted camera.
    # A fixed 0.01 floor clips unit-normalized sub-centimeter assets completely
    # (for example a 0.6 mm CAD prop viewed from roughly 1 mm away).
    front = max(1e-6, min(depths))
    back = max(front + 1e-6, max(depths))
    near = max(1e-6, front * (1.0 - margin))
    far = back * (1.0 + margin)
    # Zero-extent bounds at the camera plane otherwise produce a numerically
    # useless ~1e-6 clipping interval. Preserve small CAD assets while giving
    # render backends a stable minimum depth span for degenerate geometry.
    far = max(far, near + 1e-5)
    return near, far


# ── author / fetch a camera prim ────────────────────────────────────────────────
def author_camera(stage: Usd.Stage, path: str, cam_pos, look_at,
                  focal: float = DEFAULT_FOCAL, h_ap: float = DEFAULT_APERTURE,
                  v_ap: float = DEFAULT_APERTURE, near=None, far=None) -> UsdGeom.Camera:
    """Define-or-update a UsdGeom.Camera at `path`, positioned looking at `look_at`."""
    prim = stage.GetPrimAtPath(Sdf.Path(path))
    cam = UsdGeom.Camera(prim) if prim and prim.IsA(UsdGeom.Camera) else UsdGeom.Camera.Define(stage, path)
    cam.CreateFocalLengthAttr(focal)
    cam.CreateHorizontalApertureAttr(h_ap)
    cam.CreateVerticalApertureAttr(v_ap)
    if near is not None and far is not None:
        cam.CreateClippingRangeAttr(Gf.Vec2f(float(near), float(far)))

    xf = UsdGeom.Xformable(cam.GetPrim())
    op = xf.GetTransformOp() or xf.AddTransformOp()
    op.Set(look_at_matrix(cam_pos, look_at, _up_vec(stage)))
    # Tag the tool's auto-created render cameras so `save` can strip them and they
    # don't leak into a materials/physics deliverable as collateral geometry. The
    # `usd_cam` prefix covers the single render camera AND `usd_cam_orbit_NN` (orbit
    # renders leaked into deliverables when only the exact name was tagged). The
    # usd_cam* namespace is reserved for the tool (`camera create` rejects it,
    # along with the legacy ov_cam*/dsc_cam* prefixes older builds authored).
    # EVERY camera this function authors additionally carries usdAuthoredCamera:
    # such cameras are fully described by a protocol-v2 camera_def, so renders can
    # strip them from the bundle too (user camera churn was invalidating the
    # viewpoint-independent bundle cache in round 7). Flat keys:
    # SetCustomDataByKey treats ':' as a nested-dict path separator.
    cam.GetPrim().SetCustomDataByKey("usdAuthoredCamera", True)
    if cam.GetPrim().GetName().startswith(("usd_cam", "ov_cam")):
        cam.GetPrim().SetCustomDataByKey("usdManagedCamera", True)
    return cam


def list_cameras(stage: Usd.Stage) -> list[str]:
    # includes cameras inside native instances (proxies) — listing is read-only
    return [p.GetPath().pathString
            for p in stage.Traverse(Usd.TraverseInstanceProxies())
            if p.IsA(UsdGeom.Camera)]


# ── intrinsics + projection (for raycast --screen and ref-label overlays) ────────
def camera_params(stage: Usd.Stage, camera_path: str, width: int, height: int) -> dict:
    """Intrinsics K + world->camera, adapted from usd_camera.py:extract_camera_parameters."""
    cam = UsdGeom.Camera(stage.GetPrimAtPath(Sdf.Path(camera_path)))
    tc = Usd.TimeCode.Default()
    c2w = UsdGeom.XformCache(tc).GetLocalToWorldTransform(cam.GetPrim())
    w2c = c2w.GetInverse()
    focal = float(cam.GetFocalLengthAttr().Get(tc) or DEFAULT_FOCAL)
    h_ap = float(cam.GetHorizontalApertureAttr().Get(tc) or DEFAULT_APERTURE)
    # Square pixels: the beauty backends conform the *vertical* aperture to the image aspect
    # (v_ap = h_ap*height/width) so OVRTX and analytic views are not stretched. The
    # camera's stored v_ap is often left equal to h_ap, so deriving fy from it would stretch
    # the analytic AOVs relative to the beauty pass. Match the backends: fy == fx.
    fx = fy = focal / h_ap * width
    cx, cy = width * 0.5, height * 0.5
    return {"fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "w2c": np.array([[w2c[r][c] for c in range(4)] for r in range(4)], dtype=np.float64)}


def project_point(world_point, params: dict):
    """World point -> (u, v, depth). depth>0 means in front. usd_camera.py:project_point."""
    wp = np.array([world_point[0], world_point[1], world_point[2], 1.0])
    cam = wp @ params["w2c"]  # USD is row-vector convention: v' = v · M
    xc, yc, zc = float(cam[0]), float(cam[1]), float(cam[2])
    if abs(zc) < 1e-9:
        return float("nan"), float("nan"), zc
    inv = -1.0 / zc  # camera looks down -Z
    u = params["fx"] * (xc * inv) + params["cx"]
    v = params["fy"] * (-yc * inv) + params["cy"]  # flip Y for image space
    return u, v, -zc
