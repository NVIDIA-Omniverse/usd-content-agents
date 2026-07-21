# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""USD rendering functions using NVIDIA Warp GPU raytracer.

This module provides rendering functions that use the warp-lang library's
GPU raytracer (from the Newton physics project) for in-process CUDA-based
rendering. Unlike OvRTX, this requires no Vulkan display server, no
subprocess isolation, and no separate venv — rendering runs directly in
the current Python process on any CUDA-capable GPU.

The raytracer uses diffuse-only shading with configurable color boosting
to compensate for the lack of PBR materials.

    Requires:
    - warp-lang (``pip install warp-lang``)
    - Newton warp_raytrace module from ``world-understanding[warp]``
    - NVIDIA GPU with CUDA
"""

import inspect
import logging
import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from world_understanding.functions.graphics.render_ovrtx import _parse_frames

if TYPE_CHECKING:
    from pxr import Usd  # pragma: no cover

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _RenderMesh:
    """Mesh data in both legacy Warp and Newton ModelBuilder forms."""

    warp_mesh: Any
    vertices: np.ndarray
    indices: np.ndarray


# ---------------------------------------------------------------------------
# Lazy warp imports — so the module can be imported even without warp
# ---------------------------------------------------------------------------


def _import_warp():
    """Lazily import warp and Newton's warp_raytrace module.

    Returns:
        Tuple of (wp, RenderContext, mesh_shape_type_int, RenderLightType).

    Raises:
        ImportError: If warp-lang or Newton warp_raytrace is not available.
    """
    try:
        import warp as wp
    except ImportError as exc:
        raise ImportError(
            "warp-lang is required for WarpRenderingBackend. "
            "Install with: pip install warp-lang"
        ) from exc

    try:
        from newton._src.sensors.warp_raytrace import RenderContext
        from newton._src.sensors.warp_raytrace.types import RenderLightType
    except ImportError as exc:
        raise ImportError(
            "Newton is required for WarpRenderingBackend. "
            "Install with: uv pip install 'world-understanding[warp]'"
        ) from exc

    # RenderShapeType was removed in newton >= a6069e84 and replaced by GeoType.
    # Support both old and new newton versions.
    try:
        from newton._src.sensors.warp_raytrace import RenderShapeType

        mesh_shape_type_int = int(RenderShapeType.MESH)
    except ImportError:
        from newton._src.geometry import GeoType

        mesh_shape_type_int = int(GeoType.MESH)

    return wp, RenderContext, mesh_shape_type_int, RenderLightType


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _triangulate(
    face_vertex_counts: np.ndarray, face_vertex_indices: np.ndarray
) -> np.ndarray:
    """Fan-triangulate polygon faces to flat triangle index array.

    Args:
        face_vertex_counts: Per-face vertex counts (e.g., [4, 3, 4]).
        face_vertex_indices: Flat vertex index array.

    Returns:
        Flat int32 array of triangle indices (v0, v1, v2, v0, v1, v2, ...).
    """
    triangles = []
    idx = 0
    for count in face_vertex_counts:
        v0 = face_vertex_indices[idx]
        for i in range(1, count - 1):
            triangles.extend(
                [v0, face_vertex_indices[idx + i], face_vertex_indices[idx + i + 1]]
            )
        idx += count
    return np.array(triangles, dtype=np.int32)


def _gf_matrix_to_transform_7f(m) -> list[float]:
    """Convert a Gf.Matrix4d to 7 floats [px, py, pz, qx, qy, qz, qw].

    This is the format expected by ``wp.transformf``.

    Args:
        m: A ``Gf.Matrix4d`` world transform matrix.

    Returns:
        List of 7 floats: [tx, ty, tz, qx, qy, qz, qw].
    """
    from pxr import Gf

    t = m.ExtractTranslation()
    gf_quat = Gf.Transform(m).GetRotation().GetQuat()
    real = gf_quat.GetReal()
    imag = gf_quat.GetImaginary()

    # Normalize quaternion
    length = math.sqrt(real**2 + imag[0] ** 2 + imag[1] ** 2 + imag[2] ** 2)
    if length > 0:
        real /= length
        imag = [imag[0] / length, imag[1] / length, imag[2] / length]
    else:
        imag = [0.0, 0.0, 0.0]
        real = 1.0

    return [
        float(t[0]),
        float(t[1]),
        float(t[2]),
        float(imag[0]),
        float(imag[1]),
        float(imag[2]),
        float(real),
    ]


def _unpack_color_image(packed: np.ndarray, world_idx: int, cam_idx: int) -> np.ndarray:
    """Unpack uint32 ABGR-packed color image to RGBA uint8 array for PIL.

    The warp raytracer packs RGBA into uint32 as:
        bits 0-7: R, 8-15: G, 16-23: B, 24-31: A

    Args:
        packed: numpy array of shape (worlds, cameras, H, W) with dtype uint32.
        world_idx: World index (typically 0).
        cam_idx: Camera index.

    Returns:
        RGBA uint8 array of shape (H, W, 4).
    """
    p = packed[world_idx, cam_idx]  # (H, W) uint32
    r = (p & 0xFF).astype(np.uint8)
    g = ((p >> 8) & 0xFF).astype(np.uint8)
    b = ((p >> 16) & 0xFF).astype(np.uint8)
    a = np.full_like(r, 255)
    return np.stack([r, g, b, a], axis=-1)


def _unpack_depth_image(depth: np.ndarray, world_idx: int, cam_idx: int) -> np.ndarray:
    """Extract depth image for a specific world/camera.

    Args:
        depth: numpy array of shape (worlds, cameras, H, W) with dtype float32.
        world_idx: World index (typically 0).
        cam_idx: Camera index.

    Returns:
        float32 array of shape (H, W).
    """
    return depth[world_idx, cam_idx].copy()


def _unpack_normal_image(
    normal: np.ndarray, world_idx: int, cam_idx: int
) -> np.ndarray:
    """Extract normal image for a specific world/camera.

    Args:
        normal: numpy array of shape (worlds, cameras, H, W, 3) with dtype float32.
        world_idx: World index (typically 0).
        cam_idx: Camera index.

    Returns:
        float32 array of shape (H, W, 3).
    """
    return normal[world_idx, cam_idx].copy()


# ---------------------------------------------------------------------------
# Scene extraction from USD
# ---------------------------------------------------------------------------


def _extract_meshes(stage: "Usd.Stage", time_code, device: str):
    """Extract all Mesh prims from the stage and create render mesh data.

    Traverses the stage for UsdGeom.Mesh prims with render or default purpose.
    Each mesh is triangulated and uploaded to the GPU, while retaining CPU-side
    vertex/index arrays for Newton >=1.2's Model-based renderer path.

    Args:
        stage: USD stage.
        time_code: USD TimeCode for attribute evaluation.
        device: Warp device string (e.g., "cuda:0").

    Returns:
        Tuple of (warp_meshes, mesh_prims) where:
            - warp_meshes: List of _RenderMesh objects
            - mesh_prims: List of corresponding Usd.Prim objects
    """
    wp, _, _, _ = _import_warp()
    from pxr import UsdGeom

    warp_meshes = []
    mesh_prims = []

    for prim in stage.TraverseAll():
        if not prim.IsA(UsdGeom.Mesh) or prim.IsInstanceProxy():
            continue

        # Check purpose (render/default only)
        imageable = UsdGeom.Imageable(prim)
        purpose = imageable.ComputePurpose()
        if purpose not in (UsdGeom.Tokens.default_, UsdGeom.Tokens.render):
            continue

        mesh = UsdGeom.Mesh(prim)
        points_attr = mesh.GetPointsAttr()
        fvc_attr = mesh.GetFaceVertexCountsAttr()
        fvi_attr = mesh.GetFaceVertexIndicesAttr()

        if (
            not points_attr.HasValue()
            or not fvc_attr.HasValue()
            or not fvi_attr.HasValue()
        ):
            continue

        points = np.array(points_attr.Get(time_code), dtype=np.float32)
        fvc = np.array(fvc_attr.Get(time_code))
        fvi = np.array(fvi_attr.Get(time_code))

        if len(points) == 0 or len(fvc) == 0 or len(fvi) == 0:
            continue

        tri_idx = _triangulate(fvc, fvi)
        if len(tri_idx) == 0:
            continue

        wm = wp.Mesh(
            points=wp.array(points, dtype=wp.vec3f, device=device),
            indices=wp.array(tri_idx, dtype=wp.int32, device=device),
        )
        warp_meshes.append(_RenderMesh(wm, points, tri_idx))
        mesh_prims.append(prim)

    logger.debug("Extracted %d meshes from USD stage", len(warp_meshes))
    return warp_meshes, mesh_prims


def _get_display_color(
    prim, time_code, boost: float = 3.0
) -> tuple[float, float, float, float]:
    """Get primvars:displayColor at time, boosted for diffuse-only shading.

    Args:
        prim: USD prim to query.
        time_code: USD TimeCode.
        boost: Multiplier to compensate for diffuse-only shading (default 3.0).

    Returns:
        RGBA tuple with values in [0, 1].
    """
    attr = prim.GetAttribute("primvars:displayColor")
    if attr and attr.HasValue():
        val = attr.Get(time_code)
        if val and len(val) > 0:
            c = val[0]
            return (
                min(float(c[0]) * boost, 1.0),
                min(float(c[1]) * boost, 1.0),
                min(float(c[2]) * boost, 1.0),
                1.0,
            )
    return (0.8, 0.8, 0.8, 1.0)


def _is_visible(prim, time_code) -> bool:
    """Check if prim is visible at the given USD time code."""
    from pxr import UsdGeom

    return (
        UsdGeom.Imageable(prim).ComputeVisibility(time_code) != UsdGeom.Tokens.invisible
    )


# ---------------------------------------------------------------------------
# Light setup
# ---------------------------------------------------------------------------


def _setup_lights(stage: "Usd.Stage", ctx, time_code, device: str) -> None:
    """Configure lights on the RenderContext.

    Checks for existing UsdLux lights in the stage. If found, extracts
    their direction. If none found, sets up 3 default directional lights
    (key + fill + rim).

    Args:
        stage: USD stage.
        ctx: RenderContext to configure.
        time_code: USD TimeCode for attribute evaluation.
        device: Warp device string.
    """
    wp, _, _, RenderLightType = _import_warp()
    from pxr import Gf, UsdGeom, UsdLux

    DIR = int(RenderLightType.DIRECTIONAL)

    # Look for existing lights
    light_dirs = []
    xform_cache = UsdGeom.XformCache(time_code)

    for prim in stage.Traverse():
        if not (
            prim.IsA(UsdLux.BoundableLightBase)
            or prim.IsA(UsdLux.NonboundableLightBase)
        ):
            continue

        if prim.IsA(UsdLux.DistantLight):
            # DistantLight emits along -Z in local space
            mat = xform_cache.GetLocalToWorldTransform(prim)
            light_dir = Gf.Vec3d(mat.TransformDir(Gf.Vec3d(0, 0, -1))).GetNormalized()
            light_dirs.append(
                (float(light_dir[0]), float(light_dir[1]), float(light_dir[2]))
            )

    if light_dirs:
        # Use existing lights as directional lights
        num_lights = len(light_dirs)
        ctx.lights_active = wp.array([True] * num_lights, dtype=wp.bool, device=device)
        ctx.lights_type = wp.array([DIR] * num_lights, dtype=wp.int32, device=device)
        ctx.lights_cast_shadow = wp.array(
            [True] * num_lights, dtype=wp.bool, device=device
        )
        ctx.lights_position = wp.array(
            [(0.0, 0.0, 0.0)] * num_lights, dtype=wp.vec3f, device=device
        )
        ctx.lights_orientation = wp.array(light_dirs, dtype=wp.vec3f, device=device)
        logger.debug("Using %d existing lights from USD stage", num_lights)
    else:
        # Default 3-light setup: key + fill + rim
        # Key light: 45 deg from above-right-front
        key_dir = (0.5, -0.707, -0.5)
        # Fill light: opposite of key, softer
        fill_dir = (-0.5, 0.707, 0.5)
        # Rim light: from behind/above
        rim_dir = (0.0, -0.5, 0.866)

        ctx.lights_active = wp.array([True, True, True], dtype=wp.bool, device=device)
        ctx.lights_type = wp.array([DIR, DIR, DIR], dtype=wp.int32, device=device)
        ctx.lights_cast_shadow = wp.array(
            [True, True, True], dtype=wp.bool, device=device
        )
        ctx.lights_position = wp.array(
            [(0.0, 0.0, 0.0)] * 3, dtype=wp.vec3f, device=device
        )
        ctx.lights_orientation = wp.array(
            [key_dir, fill_dir, rim_dir], dtype=wp.vec3f, device=device
        )
        logger.debug("No lights in stage — using 3 default directional lights")


# ---------------------------------------------------------------------------
# RenderContext setup
# ---------------------------------------------------------------------------


def _setup_render_context(
    warp_meshes: list,
    mesh_prims: list,
    time_code,
    device: str,
    enable_shadows: bool = True,
    enable_backface_culling: bool = True,
    color_boost: float = 3.0,
):
    """Create and configure a RenderContext with the extracted scene data.

    Args:
        warp_meshes: List of _RenderMesh objects.
        mesh_prims: List of corresponding USD prims.
        time_code: USD TimeCode for initial attribute evaluation.
        device: Warp device string.
        enable_shadows: Whether to enable shadow rays.
        enable_backface_culling: Whether to enable backface culling.
        color_boost: Color boost factor for diffuse compensation.

    Returns:
        Configured RenderContext.
    """
    wp, RenderContext, mesh_shape_type_int, _ = _import_warp()

    num_meshes = len(warp_meshes)

    # Newton >=0.2.3 renamed Options→Config and options=→config=. Newton
    # 1.4 moved Config from the constructor to render(), so detect the
    # supported constructor contract instead of using the class attribute as a
    # proxy for the accepted keyword arguments.
    options_cls = getattr(RenderContext, "Options", None) or RenderContext.Config
    render_config = options_cls(
        enable_global_world=False,
        enable_textures=False,
        enable_shadows=enable_shadows,
        enable_ambient_lighting=True,
        enable_particles=False,
        enable_backface_culling=enable_backface_culling,
        max_distance=1000.0,
    )
    constructor_parameters = inspect.signature(RenderContext).parameters
    accepts_arbitrary_keywords = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in constructor_parameters.values()
    )
    if hasattr(RenderContext, "Options"):
        constructor_config_key = "options"
    elif "config" in constructor_parameters or accepts_arbitrary_keywords:
        constructor_config_key = "config"
    else:
        constructor_config_key = None

    constructor_kwargs: dict[str, Any] = {
        "world_count": 1,
        "device": device,
    }
    if constructor_config_key is not None:
        constructor_kwargs[constructor_config_key] = render_config
    ctx = RenderContext(**constructor_kwargs)
    ctx._wu_render_config = render_config
    ctx._wu_render_config_on_render = constructor_config_key is None

    if not hasattr(ctx, "utils"):
        from newton._src.sensors.warp_raytrace import Utils

        # Newton 1.4 stopped owning Utils on RenderContext. Reattach the
        # package's utility object so the renderer can preserve one code path
        # for camera rays and output allocation across supported versions.
        ctx.utils = Utils(ctx, render_config=render_config)

    if not hasattr(ctx.utils, "compute_mesh_bounds"):
        return _setup_newton_model_render_context(
            ctx=ctx,
            render_meshes=warp_meshes,
            mesh_prims=mesh_prims,
            time_code=time_code,
            device=device,
            color_boost=color_boost,
        )

    # -- Mesh data --
    ctx.mesh_ids = wp.array(
        [m.warp_mesh.id for m in warp_meshes], dtype=wp.uint64, device=device
    )
    ctx.mesh_bounds = wp.empty((num_meshes, 2), dtype=wp.vec3f, ndim=2, device=device)
    ctx.utils.compute_mesh_bounds()

    # Dummy arrays for texture/texcoord (kernel signature requires them,
    # but they are never accessed with enable_textures=False / materials=-1)
    ctx.mesh_face_offsets = wp.zeros(1, dtype=wp.int32, device=device)
    ctx.mesh_face_vertices = wp.zeros(1, dtype=wp.vec3i, device=device)
    ctx.mesh_texcoord = wp.zeros(1, dtype=wp.vec2f, device=device)
    ctx.mesh_texcoord_offsets = wp.zeros(1, dtype=wp.int32, device=device)
    ctx.material_texture_ids = wp.array([-1], dtype=wp.int32, device=device)
    ctx.material_texture_repeat = wp.zeros(1, dtype=wp.vec2f, device=device)
    ctx.material_rgba = wp.zeros(1, dtype=wp.vec4f, device=device)
    ctx.texture_offsets = wp.zeros(1, dtype=wp.int32, device=device)
    ctx.texture_data = wp.zeros(1, dtype=wp.uint32, device=device)
    ctx.texture_height = wp.zeros(1, dtype=wp.int32, device=device)
    ctx.texture_width = wp.zeros(1, dtype=wp.int32, device=device)

    # -- Shape data (all meshes; visibility controlled via shape_enabled per frame) --
    ctx.shape_types = wp.array(
        [mesh_shape_type_int] * num_meshes, dtype=wp.int32, device=device
    )
    ctx.shape_mesh_indices = wp.array(
        list(range(num_meshes)), dtype=wp.int32, device=device
    )
    ctx.shape_sizes = wp.array(
        [(1.0, 1.0, 1.0)] * num_meshes, dtype=wp.vec3f, device=device
    )
    ctx.shape_materials = wp.array([-1] * num_meshes, dtype=wp.int32, device=device)
    ctx.shape_world_index = wp.array([0] * num_meshes, dtype=wp.int32, device=device)
    ctx.shape_count_total = num_meshes

    # World transforms for mesh prims
    from pxr import UsdGeom

    xform_cache = UsdGeom.XformCache(time_code)
    shape_xforms = [
        _gf_matrix_to_transform_7f(xform_cache.GetLocalToWorldTransform(p))
        for p in mesh_prims
    ]
    data = np.array(shape_xforms, dtype=np.float32)
    ctx.shape_transforms = wp.array(data, dtype=wp.transformf, device=device)

    # Initial visibility (all visible)
    visible = list(range(num_meshes))
    ctx.shape_enabled = wp.array(
        np.array(visible, dtype=np.uint32), dtype=wp.uint32, device=device
    )
    ctx.shape_count_enabled = len(visible)

    # Initial colors
    colors = [_get_display_color(p, time_code, boost=color_boost) for p in mesh_prims]
    ctx.shape_colors = wp.array(colors, dtype=wp.vec4f, device=device)

    return ctx


def _setup_newton_model_render_context(
    *,
    ctx,
    render_meshes: list[_RenderMesh],
    mesh_prims: list,
    time_code,
    device: str,
    color_boost: float,
):
    """Initialize Newton >=1.2 RenderContext, which renders Model/State BVHs."""
    import newton
    from pxr import UsdGeom

    wp, _, _, _ = _import_warp()

    # Render-only USD meshes are static global shapes in Newton's model. The
    # raytracer must include the global world when rendering world 0.
    ctx._wu_render_config.enable_global_world = True

    builder = newton.ModelBuilder()
    xform_cache = UsdGeom.XformCache(time_code)
    for render_mesh, prim in zip(render_meshes, mesh_prims, strict=True):
        color = _get_display_color(prim, time_code, boost=color_boost)
        mesh = newton.Mesh(
            render_mesh.vertices,
            render_mesh.indices,
            compute_inertia=False,
            color=color[:3],
        )
        cfg = builder.ShapeConfig(
            density=0.0,
            collision_group=0,
            has_shape_collision=False,
            has_particle_collision=False,
        )
        xform_7f = _gf_matrix_to_transform_7f(
            xform_cache.GetLocalToWorldTransform(prim)
        )
        xform = wp.transform(xform_7f[:3], xform_7f[3:])
        builder.add_shape_mesh(
            body=-1,
            xform=xform,
            mesh=mesh,
            cfg=cfg,
            color=color[:3],
            label=str(prim.GetPath()),
        )

    model = builder.finalize(device=device)
    state = model.state()
    ctx.init_from_model(model, load_textures=False)
    ctx._wu_render_model = model
    ctx._wu_render_state = state
    ctx._wu_base_shape_flags = [int(flag) for flag in model.shape_flags.numpy()]
    _update_newton_model_render_context(
        ctx,
        mesh_prims=mesh_prims,
        time_code=time_code,
        device=device,
        color_boost=color_boost,
    )
    return ctx


def _update_render_context_for_frame(
    ctx,
    *,
    mesh_prims: list,
    time_code,
    device: str,
    color_boost: float,
) -> int:
    if hasattr(ctx, "_wu_render_model"):
        return _update_newton_model_render_context(
            ctx,
            mesh_prims=mesh_prims,
            time_code=time_code,
            device=device,
            color_boost=color_boost,
        )

    wp, _, _, _ = _import_warp()
    visible = [i for i, p in enumerate(mesh_prims) if _is_visible(p, time_code)]
    ctx.shape_enabled = wp.array(
        np.array(visible, dtype=np.uint32), dtype=wp.uint32, device=device
    )
    ctx.shape_count_enabled = len(visible)

    # Force BVH rebuild when visibility changes.
    ctx.bvh_shapes = None
    ctx.bvh_shapes_lowers = None
    ctx.bvh_shapes_uppers = None
    ctx.bvh_shapes_groups = None
    ctx.bvh_shapes_group_roots = None

    colors = [_get_display_color(p, time_code, boost=color_boost) for p in mesh_prims]
    ctx.shape_colors = wp.array(colors, dtype=wp.vec4f, device=device)
    return len(visible)


def _update_newton_model_render_context(
    ctx,
    *,
    mesh_prims: list,
    time_code,
    device: str,
    color_boost: float,
) -> int:
    from pxr import UsdGeom

    try:
        from newton.geometry import ShapeFlags
    except ImportError:
        from newton._src.geometry import ShapeFlags

    wp, _, _, _ = _import_warp()
    model = ctx._wu_render_model
    state = ctx._wu_render_state

    visible = {i for i, prim in enumerate(mesh_prims) if _is_visible(prim, time_code)}
    visible_bit = int(ShapeFlags.VISIBLE)
    flags = []
    for index, base_flag in enumerate(ctx._wu_base_shape_flags):
        if index in visible:
            flags.append(base_flag | visible_bit)
        else:
            flags.append(base_flag & ~visible_bit)
    model.shape_flags = wp.array(flags, dtype=wp.int32, device=device)

    xform_cache = UsdGeom.XformCache(time_code)
    shape_xforms = [
        _gf_matrix_to_transform_7f(xform_cache.GetLocalToWorldTransform(p))
        for p in mesh_prims
    ]
    model.shape_transform = wp.array(
        np.array(shape_xforms, dtype=np.float32), dtype=wp.transform, device=device
    )

    colors = [
        _get_display_color(p, time_code, boost=color_boost)[:3] for p in mesh_prims
    ]
    ctx.shape_colors = wp.array(colors, dtype=wp.vec3f, device=device)

    if hasattr(model, "bvh_build_shapes"):
        model.bvh_build_shapes(state)
    else:
        from newton.geometry import build_bvh_shape

        build_bvh_shape(model, state)
    return len(visible)


def _render_context_render(ctx, **render_kwargs: Any) -> None:
    if getattr(ctx, "_wu_render_config_on_render", False):
        render_kwargs.setdefault("config", ctx._wu_render_config)
    if hasattr(ctx, "_wu_render_model"):
        ctx.render(ctx._wu_render_model, ctx._wu_render_state, **render_kwargs)
    else:
        ctx.render(**render_kwargs)


def _clear_render_outputs(render_kwargs: dict[str, Any]) -> None:
    for output_name in ("color_image", "depth_image", "normal_image"):
        output = render_kwargs.get(output_name)
        if output is not None:
            output.zero_()


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------


def _compute_camera_fov(stage: "Usd.Stage", camera_path: str, time_code) -> float:
    """Compute vertical FOV in radians from a UsdGeom.Camera.

    Args:
        stage: USD stage.
        camera_path: Path to camera prim.
        time_code: USD TimeCode.

    Returns:
        Vertical FOV in radians.
    """
    from pxr import UsdGeom

    cam = UsdGeom.Camera(stage.GetPrimAtPath(camera_path))
    focal = float(cam.GetFocalLengthAttr().Get(time_code))
    v_aperture = float(cam.GetVerticalApertureAttr().Get(time_code))

    if focal <= 0 or v_aperture <= 0:
        # Fallback to reasonable defaults
        logger.warning(
            "Invalid camera parameters for %s (focal=%f, vAperture=%f). "
            "Using default 45-degree FOV.",
            camera_path,
            focal,
            v_aperture,
        )
        return math.radians(45.0)

    return 2.0 * math.atan(v_aperture / (2.0 * focal))


def _get_camera_transforms(
    stage: "Usd.Stage",
    camera_paths: list[str],
    time_code,
) -> list[list[float]]:
    """Get world transforms for cameras at a given time code.

    Args:
        stage: USD stage.
        camera_paths: List of camera prim paths.
        time_code: USD TimeCode.

    Returns:
        List of 7-float transform lists [tx, ty, tz, qx, qy, qz, qw].
    """
    from pxr import UsdGeom

    xfc = UsdGeom.XformCache(time_code)
    cam_xforms = []
    for cam_path in camera_paths:
        prim = stage.GetPrimAtPath(cam_path)
        if not prim.IsValid():
            logger.warning("Camera prim not found: %s", cam_path)
            # Identity transform as fallback
            cam_xforms.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
            continue
        cam_mat = xfc.GetLocalToWorldTransform(prim)
        cam_xforms.append(_gf_matrix_to_transform_7f(cam_mat))
    return cam_xforms


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------


def render_all_cameras(
    stage: "Usd.Stage",
    image_width: int = 1024,
    image_height: int = 1024,
    cameras: list[str] | None = None,
    frames: str = "0",
    sensors: list[str] | None = None,
    device: str = "cuda:0",
    color_boost: float = 3.0,
    enable_shadows: bool = True,
    enable_backface_culling: bool = True,
) -> dict[str, Any]:
    """Render multiple cameras from a USD stage using Warp GPU raytracer.

    This function extracts geometry from the USD stage, sets up a
    RenderContext, and renders all requested cameras/frames in-process
    on the CUDA GPU. No subprocess, no Vulkan, no DISPLAY required.

    Args:
        stage: USD stage to render.
        image_width: Output image width in pixels.
        image_height: Output image height in pixels.
        cameras: List of camera prim paths. If None, uses ["/Camera"].
        frames: Frame specification (e.g., "0", "0:10", "0,5,10").
        sensors: Optional sensor names (e.g., ["depth", "normal"]).
        device: Warp CUDA device string. Default: "cuda:0".
        color_boost: Multiplier for displayColor to compensate for
            diffuse-only shading. Default: 3.0.
        enable_shadows: Whether to cast shadow rays. Default: True.
        enable_backface_culling: Whether to enable backface culling.
            Default: True.

    Returns:
        Dict matching RenderingBackend.render() contract with keys:
            total_cameras, successful_cameras, failed_cameras,
            total_render_time, results (list of per-camera dicts).
    """
    wp, _, _, _ = _import_warp()

    if cameras is None or len(cameras) == 0:
        cameras = ["/Camera"]

    frame_list = _parse_frames(frames)
    sensors = sensors or []
    total_start_time = time.time()

    # Ensure warp is initialized
    wp.init()

    # Extract meshes from stage (using time 0 for geometry)
    from pxr import Usd

    tc0 = Usd.TimeCode(frame_list[0] if frame_list else 0)
    warp_meshes, mesh_prims = _extract_meshes(stage, tc0, device)

    if not warp_meshes:
        logger.warning("No meshes found in USD stage")
        total_render_time = time.time() - total_start_time
        return {
            "total_cameras": len(cameras),
            "successful_cameras": 0,
            "failed_cameras": len(cameras),
            "total_render_time": total_render_time,
            "results": [
                {
                    "camera": cam,
                    "images": [],
                    "sensors": {},
                    "render_time": total_render_time,
                    "frame_count": 0,
                    "error": "No meshes found in stage",
                }
                for cam in cameras
            ],
        }

    num_cameras = len(cameras)
    num_meshes = len(warp_meshes)

    # Set up RenderContext
    ctx = _setup_render_context(
        warp_meshes=warp_meshes,
        mesh_prims=mesh_prims,
        time_code=tc0,
        device=device,
        enable_shadows=enable_shadows,
        enable_backface_culling=enable_backface_culling,
        color_boost=color_boost,
    )

    # Set up lights
    _setup_lights(stage, ctx, tc0, device)

    # Compute per-camera FOVs (each camera may have different optics)
    per_camera_fovs = [_compute_camera_fov(stage, cam, tc0) for cam in cameras]

    # Pre-compute camera rays (FOV is constant across frames)
    camera_fovs = wp.array(per_camera_fovs, dtype=wp.float32, device=device)
    if hasattr(ctx.utils, "compute_camera_rays_pinhole"):
        camera_rays = ctx.utils.compute_camera_rays_pinhole(
            image_width,
            image_height,
            camera_fovs=camera_fovs,
        )
    else:
        camera_rays = ctx.utils.compute_pinhole_camera_rays(
            image_width, image_height, camera_fovs
        )

    # Create output buffers
    color_image = ctx.utils.create_color_image_output(
        image_width, image_height, num_cameras
    )

    depth_image = None
    if "depth" in sensors:
        depth_image = ctx.utils.create_depth_image_output(
            image_width, image_height, num_cameras
        )

    normal_image = None
    if "normal" in sensors:
        normal_image = ctx.utils.create_normal_image_output(
            image_width, image_height, num_cameras
        )

    # Per-camera result accumulators
    cam_data: list[dict[str, Any]] = [
        {"images": [], "sensor_data": {s: {} for s in sensors}} for _ in cameras
    ]

    # Render loop
    logger.info(
        "Warp rendering %d camera(s), %d frame(s), %d mesh(es) at %dx%d",
        num_cameras,
        len(frame_list),
        num_meshes,
        image_width,
        image_height,
    )

    for frame_num in frame_list:
        t0 = time.time()
        tc = Usd.TimeCode(frame_num)

        visible_count = _update_render_context_for_frame(
            ctx,
            mesh_prims=mesh_prims,
            time_code=tc,
            device=device,
            color_boost=color_boost,
        )

        # Camera transforms at this frame
        cam_xforms = _get_camera_transforms(stage, cameras, tc)
        xform_data = np.array(cam_xforms, dtype=np.float32)
        camera_transforms = wp.array(xform_data, dtype=wp.transformf, device=device)
        camera_transforms = camera_transforms.reshape((num_cameras, 1))

        # Render
        render_kwargs: dict[str, Any] = {
            "camera_transforms": camera_transforms,
            "camera_rays": camera_rays,
            "color_image": color_image,
        }
        if depth_image is not None:
            render_kwargs["depth_image"] = depth_image
        if normal_image is not None:
            render_kwargs["normal_image"] = normal_image

        if visible_count == 0:
            _clear_render_outputs(render_kwargs)
        else:
            _render_context_render(ctx, **render_kwargs)
        wp.synchronize_device(device)

        elapsed = time.time() - t0

        # Extract images from GPU
        color_np = color_image.numpy()
        for cam_idx in range(num_cameras):
            rgba = _unpack_color_image(color_np, 0, cam_idx)
            pil_img = Image.fromarray(rgba).convert("RGBA")
            cam_data[cam_idx]["images"].append(pil_img)

        # Extract sensor data
        if depth_image is not None and "depth" in sensors:
            depth_np = depth_image.numpy()
            for cam_idx in range(num_cameras):
                cam_data[cam_idx]["sensor_data"]["depth"][frame_num] = (
                    _unpack_depth_image(depth_np, 0, cam_idx)
                )

        if normal_image is not None and "normal" in sensors:
            normal_np = normal_image.numpy()
            for cam_idx in range(num_cameras):
                cam_data[cam_idx]["sensor_data"]["normal"][frame_num] = (
                    _unpack_normal_image(normal_np, 0, cam_idx)
                )

        logger.debug(
            "Frame %d: %.3fs, %d/%d meshes visible",
            frame_num,
            elapsed,
            visible_count,
            num_meshes,
        )

    total_render_time = time.time() - total_start_time

    # Build results in the standard RenderingBackend format
    results = []
    successful_cameras = 0
    failed_cameras = 0

    for cam_idx, camera in enumerate(cameras):
        images = cam_data[cam_idx]["images"]
        sensor_data = cam_data[cam_idx]["sensor_data"]

        if images:
            successful_cameras += 1
            results.append(
                {
                    "camera": camera,
                    "images": images,
                    "sensors": sensor_data,
                    "render_time": total_render_time,
                    "frame_count": len(images),
                }
            )
        else:
            failed_cameras += 1
            results.append(
                {
                    "camera": camera,
                    "images": [],
                    "sensors": {},
                    "render_time": total_render_time,
                    "frame_count": 0,
                    "error": "No images produced",
                }
            )

    logger.info(
        "Warp render complete: %d/%d cameras, %.2fs total (%.3fs/frame)",
        successful_cameras,
        len(cameras),
        total_render_time,
        total_render_time / max(len(frame_list), 1),
    )

    return {
        "total_cameras": len(cameras),
        "successful_cameras": successful_cameras,
        "failed_cameras": failed_cameras,
        "total_render_time": total_render_time,
        "results": results,
    }
