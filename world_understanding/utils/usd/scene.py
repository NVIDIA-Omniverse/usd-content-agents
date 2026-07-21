# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""USD scene-construction helpers: ground plane authoring with physics.

The geometry pattern mirrors the private
``apps/material_agent/internal/scripts/render_turntable_direct.py:_add_ground_plane``
helper but adds the ``UsdPhysics.CollisionAPI`` and
``UsdPhysics.MaterialAPI`` schemas so the plane participates in
simulation. We don't import that internal script — physics_agent and
material_agent must not couple — but the geometry layout is mirrored
verbatim so render output stays visually consistent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - static typing import.
    from pxr import Usd

# Default ground-plane footprint multiplier — the plane extent is
# ``max_dim * _GROUND_PLANE_EXTENT_MULTIPLIER`` when ``extent`` is None.
# Mirrors render_turntable_direct.py:217 (``max_dim * 1.45``) but we
# scale it up for physics so a body that slides off-axis still lands
# on collidable geometry.
_GROUND_PLANE_EXTENT_MULTIPLIER = 4.0
_GROUND_PLANE_OFFSET_MULTIPLIER = 0.015  # plane sits this fraction below bbox_min

# Default material name for the physics ground plane.
_DEFAULT_PLANE_PATH = "/World/GroundPlane"
_DEFAULT_MATERIAL_PATH = "/World/GroundPlaneMaterial"
_GROUND_VISUAL_COLOR = (0.14, 0.14, 0.14)
_GROUND_VISUAL_ROUGHNESS = 0.86
_GROUND_VISUAL_SPECULAR_COLOR = (0.0, 0.0, 0.0)


def add_ground_plane(
    stage: Usd.Stage,
    *,
    center: tuple[float, float, float] | None = None,
    extent: float | None = None,
    friction: float = 0.5,
    restitution: float = 0.0,
    plane_path: str = _DEFAULT_PLANE_PATH,
    material_path: str = _DEFAULT_MATERIAL_PATH,
) -> Usd.Prim:
    """Author a Mesh ground plane with collision + physics material.

    The plane is a flat quad whose normal aligns with the stage's
    up-axis (``UsdGeom.GetStageUpAxis``). Its position defaults to the
    stage's bbox_min on the up-axis (with a small offset below); its
    extent defaults to ``max(scene_size) * 4``.

    Args:
        stage: Open USD stage to author into.
        center: Optional ``(x, y, z)``. When ``None``, derives from
            ``world_understanding.utils.usd.stage.get_scene_extent``.
        extent: Half-side length in meters. When ``None``, derives from
            scene extent. Set to a fixed value (e.g. 10.0) for tests
            that need deterministic geometry.
        friction: Static + dynamic friction (same value for both —
            ovphysx tensor binding combines them). 0..2 typical.
        restitution: 0..1; 0 == no bounce, 1 == elastic.
        plane_path: USD prim path for the plane mesh.
        material_path: USD prim path for the ``UsdShade.Material``
            carrying ``UsdPhysics.MaterialAPI``.

    Returns:
        The plane prim. The mesh has ``UsdPhysics.CollisionAPI``
        applied; a ``UsdShade.Material`` is bound that carries a
        ``UsdPhysics.MaterialAPI`` with the requested friction /
        restitution.

    Raises:
        ValueError: when ``center`` / ``extent`` cannot be derived
            (empty stage with no defaults).
    """
    from pxr import (  # type: ignore[import-untyped]
        Gf,
        Sdf,
        UsdGeom,
        UsdPhysics,
        UsdShade,
    )

    up_axis = UsdGeom.GetStageUpAxis(stage)

    # Derive center / extent from scene bbox when not supplied.
    if center is None or extent is None:
        from world_understanding.utils.usd.stage import get_scene_extent

        try:
            ext_info = get_scene_extent(stage)
            bbox = ext_info.get("bounding_box") or {}
            bbox_min = bbox.get("min")
            bbox_max = bbox.get("max")
        except Exception:
            bbox_min, bbox_max = None, None
        if bbox_min is None or bbox_max is None:
            raise ValueError(
                "stage bbox is empty and no center/extent supplied; "
                "pass explicit center=(x,y,z) and extent=float."
            )
        size = [float(bbox_max[i]) - float(bbox_min[i]) for i in range(3)]
        max_dim = max(*size, 1.0)
        if center is None:
            center = (
                0.5 * (float(bbox_max[0]) + float(bbox_min[0])),
                0.5 * (float(bbox_max[1]) + float(bbox_min[1])),
                0.5 * (float(bbox_max[2]) + float(bbox_min[2])),
            )
        if extent is None:
            extent = max_dim * _GROUND_PLANE_EXTENT_MULTIPLIER

    # Compute plane vertices. The plane sits slightly below the lowest
    # scene point on the up-axis so a body initialised at bbox_min
    # doesn't penetrate the plane on frame 0.
    extent_f = float(extent)
    cx, cy, cz = (float(c) for c in center)

    # The plane's "ground" coordinate on the up-axis. We default to 0
    # which matches the apply_physics convention (UsdPhysics.Scene
    # gravity expects body to be ABOVE ground at y=0 / z=0).
    if up_axis == UsdGeom.Tokens.z:
        ground_z = 0.0
        points = [
            Gf.Vec3f(cx - extent_f, cy - extent_f, ground_z),
            Gf.Vec3f(cx + extent_f, cy - extent_f, ground_z),
            Gf.Vec3f(cx + extent_f, cy + extent_f, ground_z),
            Gf.Vec3f(cx - extent_f, cy + extent_f, ground_z),
        ]
    else:
        ground_y = 0.0
        points = [
            Gf.Vec3f(cx - extent_f, ground_y, cz - extent_f),
            Gf.Vec3f(cx - extent_f, ground_y, cz + extent_f),
            Gf.Vec3f(cx + extent_f, ground_y, cz + extent_f),
            Gf.Vec3f(cx + extent_f, ground_y, cz - extent_f),
        ]

    mesh = UsdGeom.Mesh.Define(stage, Sdf.Path(plane_path))
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateExtentAttr([points[0], points[2]])

    # Physics: collision schema on the mesh + a material with friction.
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())

    material = UsdShade.Material.Define(stage, Sdf.Path(material_path))
    material_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    material_api.CreateStaticFrictionAttr(float(friction))
    material_api.CreateDynamicFrictionAttr(float(friction))
    material_api.CreateRestitutionAttr(float(restitution))

    # Visual: keep the synthetic physics ground matte. Without an explicit
    # surface shader, RTX renderers can fall back to glossy material defaults
    # under the default HDRI dome, producing mirror-like ground highlights.
    shader = UsdShade.Shader.Define(stage, f"{material_path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*_GROUND_VISUAL_COLOR)
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(
        _GROUND_VISUAL_ROUGHNESS
    )
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(1)
    shader.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*_GROUND_VISUAL_SPECULAR_COLOR)
    )
    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)

    return mesh.GetPrim()


__all__ = ["add_ground_plane"]
