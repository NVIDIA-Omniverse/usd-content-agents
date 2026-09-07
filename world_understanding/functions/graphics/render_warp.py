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
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import wraps
from threading import Lock
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from world_understanding.functions.graphics.render_ovrtx import _parse_frames

if TYPE_CHECKING:
    from pxr import Usd  # pragma: no cover

logger = logging.getLogger(__name__)

# Newton/WARP kernel compilation writes shared cache metadata and is not safe
# when two service threads enter different render specializations at once. A
# single process-wide boundary also keeps the in-process CUDA renderer from
# overlapping work on one device. Containers do not share this cache unless
# explicitly mounted.
_WARP_RENDER_LOCK = Lock()


def _serialize_warp_render[**P, R](function: Callable[P, R]) -> Callable[P, R]:
    """Serialize in-process WARP renders, including first-use compilation."""

    @wraps(function)
    def serialized(*args: P.args, **kwargs: P.kwargs) -> R:
        with _WARP_RENDER_LOCK:
            return function(*args, **kwargs)

    return serialized


# Newton 1.4 captures RenderContext.Config.max_distance with wp.static, so each
# distinct value compiles another render kernel. Keep the specialization set
# finite while covering every practically representable WARP scene scale.
_RAY_DISTANCE_BUCKETS = (
    1.0e3,
    1.0e4,
    1.0e5,
    1.0e6,
    1.0e7,
    1.0e8,
    1.0e9,
    1.0e10,
    1.0e12,
    1.0e15,
    1.0e18,
    1.0e21,
    1.0e24,
    1.0e27,
    1.0e30,
    1.0e33,
    1.0e36,
    1.0e38,
)


@dataclass(frozen=True)
class _RenderMesh:
    """Mesh data in both legacy Warp and Newton ModelBuilder forms."""

    warp_mesh: Any
    vertices: np.ndarray
    indices: np.ndarray
    vertex_basis: np.ndarray = field(
        default_factory=lambda: np.eye(3, dtype=np.float32)
    )
    vertices_in_world_space: bool = False


@dataclass(frozen=True)
class _MeshTransform:
    """Newton-compatible decomposition of a USD mesh world transform."""

    transform_7f: list[float]
    scale: tuple[float, float, float]
    vertex_basis: np.ndarray


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


def _gf_translation_rotation_to_transform_7f(
    translation: Any, rotation: Any
) -> list[float]:
    """Convert a Gf translation and rotation to a normalized Warp transform."""
    gf_quat = rotation.GetQuat()
    real = gf_quat.GetReal()
    imag = gf_quat.GetImaginary()

    length = math.sqrt(real**2 + imag[0] ** 2 + imag[1] ** 2 + imag[2] ** 2)
    if length > 0:
        real /= length
        imag = [imag[0] / length, imag[1] / length, imag[2] / length]
    else:
        imag = [0.0, 0.0, 0.0]
        real = 1.0

    return [
        float(translation[0]),
        float(translation[1]),
        float(translation[2]),
        float(imag[0]),
        float(imag[1]),
        float(imag[2]),
        float(real),
    ]


def _gf_matrix_to_transform_7f(m: Any) -> list[float]:
    """Convert the rigid part of a Gf.Matrix4d to a Warp transform.

    This is the format expected by ``wp.transformf``.

    Args:
        m: A ``Gf.Matrix4d`` world transform matrix.

    Returns:
        List of 7 floats: [tx, ty, tz, qx, qy, qz, qw].
    """
    from pxr import Gf

    t = m.ExtractTranslation()
    return _gf_translation_rotation_to_transform_7f(t, Gf.Transform(m).GetRotation())


def _gf_matrix_to_mesh_transform(m: Any) -> _MeshTransform:
    """Decompose a USD affine transform without dropping scale or shear.

    Newton mesh shapes represent a signed axis-aligned scale followed by a
    rigid transform. ``Gf.Transform`` factors an arbitrary affine linear part
    as ``P^-1 * S * P * R``. Baking ``P^-1`` into local mesh vertices lets
    Newton apply ``S`` and ``P * R`` natively, preserving non-uniform scale,
    mirroring, and static shear exactly.

    Args:
        m: A finite ``Gf.Matrix4d`` world transform matrix.

    Returns:
        Newton-compatible rigid transform, signed scale, and vertex basis.

    Raises:
        ValueError: If the matrix is non-finite or cannot be decomposed
            losslessly for WARP rendering.
    """
    from pxr import Gf

    matrix_values = np.asarray(m, dtype=np.float64)
    if not np.isfinite(matrix_values).all():
        raise ValueError("USD mesh transform contains non-finite values")

    transform = Gf.Transform(m)
    pivot_orientation = Gf.Matrix3d(1.0)
    pivot_orientation.SetRotate(transform.GetPivotOrientation())
    rotation = Gf.Matrix3d(1.0)
    rotation.SetRotate(transform.GetRotation())

    combined_rotation_matrix = pivot_orientation * rotation
    combined_rotation = combined_rotation_matrix.ExtractRotation()
    transform_7f = _gf_translation_rotation_to_transform_7f(
        m.ExtractTranslation(), combined_rotation
    )
    scale_value = transform.GetScale()
    scale = (
        float(scale_value[0]),
        float(scale_value[1]),
        float(scale_value[2]),
    )
    vertex_basis = np.asarray(pivot_orientation, dtype=np.float64).T

    # Gf.Transform can return a plausible-looking factorization for a singular
    # matrix even when that factorization no longer reconstructs the authored
    # linear transform (for example, a rotated zero-scale axis). Refuse that
    # lossy result instead of rendering silently displaced or stretched geometry.
    linear_part = matrix_values[:3, :3]
    reconstructed_linear = (
        vertex_basis
        @ np.diag(scale)
        @ np.asarray(combined_rotation_matrix, dtype=np.float64)
    )
    linear_magnitude = max(1.0, float(np.max(np.abs(linear_part))))
    if not np.allclose(
        reconstructed_linear,
        linear_part,
        rtol=1.0e-6,
        atol=1.0e-6 * linear_magnitude,
    ):
        raise ValueError(
            "USD mesh transform cannot be decomposed losslessly for WARP rendering"
        )

    if not np.isfinite(np.asarray((*transform_7f, *scale))).all():
        raise ValueError("USD mesh transform decomposition contains non-finite values")

    return _MeshTransform(
        transform_7f=transform_7f,
        scale=scale,
        vertex_basis=np.ascontiguousarray(vertex_basis, dtype=np.float32),
    )


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
    xform_cache = UsdGeom.XformCache(time_code)

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

        mesh_transform = _gf_matrix_to_mesh_transform(
            xform_cache.GetLocalToWorldTransform(prim)
        )
        transformed_points = np.ascontiguousarray(
            points @ mesh_transform.vertex_basis, dtype=np.float32
        )

        wm = wp.Mesh(
            points=wp.array(transformed_points, dtype=wp.vec3f, device=device),
            indices=wp.array(tri_idx, dtype=wp.int32, device=device),
        )
        warp_meshes.append(
            _RenderMesh(
                wm,
                transformed_points,
                tri_idx,
                vertex_basis=mesh_transform.vertex_basis,
            )
        )
        mesh_prims.append(prim)

    logger.debug("Extracted %d meshes from USD stage", len(warp_meshes))
    return warp_meshes, mesh_prims


def _get_mesh_shape_data(
    render_meshes: list[_RenderMesh], mesh_prims: list[Any], time_code: Any
) -> tuple[list[list[float]], list[tuple[float, float, float]]]:
    """Return Newton shape transforms/scales for one USD evaluation time.

    A fixed vertex basis can represent static shear and ordinary animated TRS.
    Animated shear changes that basis and cannot be represented by Newton's
    rigid transform plus per-axis scale without rebuilding mesh geometry, so it
    fails explicitly rather than silently rendering incorrect frames.
    """
    from pxr import UsdGeom

    xform_cache = UsdGeom.XformCache(time_code)
    shape_transforms: list[list[float]] = []
    shape_scales: list[tuple[float, float, float]] = []
    for render_mesh, prim in zip(render_meshes, mesh_prims, strict=True):
        if render_mesh.vertices_in_world_space:
            shape_transforms.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
            shape_scales.append((1.0, 1.0, 1.0))
            continue
        mesh_transform = _gf_matrix_to_mesh_transform(
            xform_cache.GetLocalToWorldTransform(prim)
        )
        if not np.allclose(
            mesh_transform.vertex_basis,
            render_mesh.vertex_basis,
            rtol=1.0e-5,
            atol=1.0e-6,
        ):
            raise ValueError(
                "WARP rendering does not support time-varying shear or scale-axis "
                f"orientation on mesh {prim.GetPath()} at {time_code}"
            )
        shape_transforms.append(mesh_transform.transform_7f)
        shape_scales.append(mesh_transform.scale)

    return shape_transforms, shape_scales


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


def _create_render_context(
    render_context_type: Any,
    *,
    device: str,
    enable_shadows: bool,
    enable_backface_culling: bool,
    max_distance: float = 1000.0,
) -> Any:
    """Construct a render context across Newton configuration API versions."""

    config_type = (
        getattr(render_context_type, "Options", None) or render_context_type.Config
    )
    config = config_type(
        enable_global_world=False,
        enable_textures=False,
        enable_shadows=enable_shadows,
        enable_ambient_lighting=True,
        enable_particles=False,
        enable_backface_culling=enable_backface_culling,
        max_distance=max_distance,
    )

    constructor_parameters = inspect.signature(render_context_type).parameters
    constructor_kwargs: dict[str, Any] = {"world_count": 1, "device": device}
    if "options" in constructor_parameters:
        constructor_kwargs["options"] = config
    elif "config" in constructor_parameters:
        constructor_kwargs["config"] = config

    ctx = render_context_type(**constructor_kwargs)
    ctx._wu_render_config = config
    if not hasattr(ctx, "config"):
        ctx.config = config

    render_method = getattr(ctx, "render", None)
    ctx._wu_render_config_per_call = render_method is not None and (
        "config" in inspect.signature(render_method).parameters
    )
    return ctx


def _ensure_render_context_utils(ctx: Any) -> None:
    if hasattr(ctx, "utils"):
        return

    from newton._src.sensors.warp_raytrace import Utils

    ctx.utils = Utils(ctx, ctx._wu_render_config)


def _compute_render_camera_rays(
    ctx: Any,
    width: int,
    height: int,
    camera_fovs: Any,
) -> Any:
    compute_rays = getattr(ctx.utils, "compute_camera_rays_pinhole", None)
    if compute_rays is not None:
        return compute_rays(width, height, camera_fovs=camera_fovs)
    return ctx.utils.compute_pinhole_camera_rays(width, height, camera_fovs)


def _create_render_output(
    ctx: Any,
    output_type: str,
    width: int,
    height: int,
    camera_count: int,
) -> Any:
    """Create an output buffer across Newton render-context API versions."""

    factory_name = f"create_{output_type}_image_output"
    factory = getattr(ctx, factory_name, None)
    if factory is None:
        factory = getattr(ctx.utils, factory_name)
    return factory(width, height, camera_count)


def _setup_render_context(
    warp_meshes: list,
    mesh_prims: list,
    time_code,
    device: str,
    enable_shadows: bool = True,
    enable_backface_culling: bool = True,
    color_boost: float = 3.0,
    max_distance: float = 1000.0,
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
        max_distance: Maximum ray distance in stage units.

    Returns:
        Configured RenderContext.
    """
    wp, RenderContext, mesh_shape_type_int, _ = _import_warp()

    num_meshes = len(warp_meshes)

    ctx = _create_render_context(
        RenderContext,
        device=device,
        enable_shadows=enable_shadows,
        enable_backface_culling=enable_backface_culling,
        max_distance=max_distance,
    )
    _ensure_render_context_utils(ctx)

    if not hasattr(getattr(ctx, "utils", None), "compute_mesh_bounds"):
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

    shape_xforms, shape_scales = _get_mesh_shape_data(
        warp_meshes, mesh_prims, time_code
    )

    # -- Shape data (all meshes; visibility controlled via shape_enabled per frame) --
    ctx.shape_types = wp.array(
        [mesh_shape_type_int] * num_meshes, dtype=wp.int32, device=device
    )
    ctx.shape_mesh_indices = wp.array(
        list(range(num_meshes)), dtype=wp.int32, device=device
    )
    ctx.shape_sizes = wp.array(shape_scales, dtype=wp.vec3f, device=device)
    ctx.shape_materials = wp.array([-1] * num_meshes, dtype=wp.int32, device=device)
    ctx.shape_world_index = wp.array([0] * num_meshes, dtype=wp.int32, device=device)
    ctx.shape_count_total = num_meshes

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

    wp, _, _, _ = _import_warp()

    # Render-only USD meshes are static global shapes in Newton's model. The
    # raytracer must include the global world when rendering world 0.
    ctx._wu_render_config.enable_global_world = True

    builder = newton.ModelBuilder()
    shape_xforms, shape_scales = _get_mesh_shape_data(
        render_meshes, mesh_prims, time_code
    )
    for render_mesh, prim, xform_7f, shape_scale in zip(
        render_meshes, mesh_prims, shape_xforms, shape_scales, strict=True
    ):
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
        xform = wp.transform(xform_7f[:3], xform_7f[3:])
        builder.add_shape_mesh(
            body=-1,
            xform=xform,
            mesh=mesh,
            scale=shape_scale,
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
        render_meshes=render_meshes,
        mesh_prims=mesh_prims,
        time_code=time_code,
        device=device,
        color_boost=color_boost,
    )
    return ctx


def _update_render_context_for_frame(
    ctx,
    *,
    render_meshes: list[_RenderMesh],
    mesh_prims: list,
    time_code,
    device: str,
    color_boost: float,
) -> int:
    if hasattr(ctx, "_wu_render_model"):
        return _update_newton_model_render_context(
            ctx,
            render_meshes=render_meshes,
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

    shape_xforms, shape_scales = _get_mesh_shape_data(
        render_meshes, mesh_prims, time_code
    )
    ctx.shape_transforms = wp.array(
        np.array(shape_xforms, dtype=np.float32),
        dtype=wp.transformf,
        device=device,
    )
    ctx.shape_sizes = wp.array(shape_scales, dtype=wp.vec3f, device=device)

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
    render_meshes: list[_RenderMesh],
    mesh_prims: list,
    time_code,
    device: str,
    color_boost: float,
) -> int:
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

    shape_xforms, shape_scales = _get_mesh_shape_data(
        render_meshes, mesh_prims, time_code
    )
    model.shape_transform = wp.array(
        np.array(shape_xforms, dtype=np.float32), dtype=wp.transform, device=device
    )
    model.shape_scale = wp.array(shape_scales, dtype=wp.vec3f, device=device)

    colors = [
        _get_display_color(p, time_code, boost=color_boost)[:3] for p in mesh_prims
    ]
    ctx.shape_colors = wp.array(colors, dtype=wp.vec3f, device=device)

    _build_model_shape_bvh(model, state)
    return len(visible)


def _render_context_render(ctx: Any, **render_kwargs: Any) -> None:
    if hasattr(ctx, "_wu_render_model"):
        if getattr(ctx, "_wu_render_config_per_call", False):
            render_kwargs.setdefault("config", ctx._wu_render_config)
        ctx.render(ctx._wu_render_model, ctx._wu_render_state, **render_kwargs)
    else:
        ctx.render(**render_kwargs)


def _build_model_shape_bvh(model: Any, state: Any) -> None:
    build_shapes = getattr(model, "bvh_build_shapes", None)
    if build_shapes is not None:
        build_shapes(state)
        return

    from newton.geometry import build_bvh_shape

    build_bvh_shape(model, state)


def _clear_render_outputs(render_kwargs: dict[str, Any]) -> None:
    for output_name in ("color_image", "depth_image", "normal_image"):
        output = render_kwargs.get(output_name)
        if output is not None:
            output.zero_()


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------


def _compute_max_camera_distance(
    stage: "Usd.Stage",
    camera_paths: list[str],
    frame_numbers: list[int | float],
    image_width: int = 1,
    image_height: int = 1,
) -> float:
    """Return a conservative, cache-bounded Newton ray-distance limit.

    USD far clipping is a camera-space Z plane, while Newton limits distance
    along normalized rays. The most oblique image-corner ray therefore needs a
    longer interval than the authored far value. Newton also specializes this
    value into its WARP render kernel, so the result is rounded up to one of a
    fixed set of buckets instead of compiling once per scene-specific camera.
    """
    from pxr import Usd, UsdGeom

    if image_width <= 0 or image_height <= 0:
        raise ValueError("WARP render dimensions must be positive")

    aspect_ratio = float(image_width) / float(image_height)
    required_distance = 0.0
    for frame_number in frame_numbers:
        time_code = Usd.TimeCode(frame_number)
        for camera_path in camera_paths:
            prim = stage.GetPrimAtPath(camera_path)
            if not prim.IsValid() or not prim.IsA(UsdGeom.Camera):
                continue
            clipping_range = UsdGeom.Camera(prim).GetClippingRangeAttr().Get(time_code)
            if clipping_range is None:
                continue
            near_distance = float(clipping_range[0])
            far_distance = float(clipping_range[1])
            if (
                math.isfinite(near_distance)
                and math.isfinite(far_distance)
                and 0.0 <= near_distance < far_distance
            ):
                vertical_fov = _compute_camera_fov(stage, camera_path, time_code)
                half_height = math.tan(vertical_fov * 0.5)
                corner_ray_length = math.sqrt(
                    1.0 + half_height * half_height + (half_height * aspect_ratio) ** 2
                )
                ray_distance = far_distance * corner_ray_length
                if math.isfinite(ray_distance):
                    required_distance = max(required_distance, ray_distance)

    if required_distance > 0.0:
        # Leave a small floating-point margin before selecting the bucket so a
        # float32 specialization cannot round just below the required bound.
        required_distance *= 1.0 + 1.0e-6
        for bucket in _RAY_DISTANCE_BUCKETS:
            if required_distance <= bucket:
                return bucket
        raise ValueError(
            "Required WARP ray distance exceeds the largest supported "
            f"finite bucket ({_RAY_DISTANCE_BUCKETS[-1]:.0e})"
        )

    logger.warning(
        "No valid camera clipping range found; using a 1000-unit WARP ray limit"
    )
    return 1000.0


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


@_serialize_warp_render
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
        max_distance=_compute_max_camera_distance(
            stage,
            cameras,
            frame_list,
            image_width=image_width,
            image_height=image_height,
        ),
    )

    # Set up lights
    _setup_lights(stage, ctx, tc0, device)

    # Compute per-camera FOVs (each camera may have different optics)
    per_camera_fovs = [_compute_camera_fov(stage, cam, tc0) for cam in cameras]

    # Pre-compute camera rays (FOV is constant across frames)
    camera_fovs = wp.array(per_camera_fovs, dtype=wp.float32, device=device)
    camera_rays = _compute_render_camera_rays(
        ctx,
        image_width,
        image_height,
        camera_fovs,
    )

    # Create output buffers
    color_image = _create_render_output(
        ctx, "color", image_width, image_height, num_cameras
    )

    depth_image = None
    if "depth" in sensors:
        depth_image = _create_render_output(
            ctx, "depth", image_width, image_height, num_cameras
        )

    normal_image = None
    if "normal" in sensors:
        normal_image = _create_render_output(
            ctx, "normal", image_width, image_height, num_cameras
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
            render_meshes=warp_meshes,
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
