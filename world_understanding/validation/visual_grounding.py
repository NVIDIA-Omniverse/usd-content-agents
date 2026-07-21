# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Visual-to-USD grounding packet generation for validation evidence.

This module renders flat object-ID views from USD mesh geometry, overlays
readable numeric prim labels on object-ID and beauty images, and writes legends
that map visible pixels back to USD prim paths and current material bindings.
It is domain-neutral evidence generation; applications decide how to interpret
the labels.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pxr import Gf, Usd, UsdGeom, UsdShade

from world_understanding.functions.graphics import render_warp
from world_understanding.functions.graphics.rendering import hide_prims_outside_subtree
from world_understanding.utils.usd.camera import (
    add_corner_view_camera,
    add_focused_corner_view_camera,
    add_focused_side_view_camera,
    add_side_view_camera,
    compute_camera_framing_position_corners,
    compute_camera_framing_position_sides,
)
from world_understanding.utils.usd.prim import get_bbox_from_prim

_SIDE_VIEW_DIRECTIONS = {"+x", "-x", "+y", "-y", "+z", "-z"}
_CALLOUT_MIN_GUTTER_X = 72
_CALLOUT_MIN_GUTTER_Y = 24


@dataclass(frozen=True)
class MeshRecord:
    numeric_id: int
    prim_path: str
    material_path: str | None
    parent_path: str
    points_world: np.ndarray
    triangles: np.ndarray
    color_rgb: tuple[int, int, int]


@dataclass(frozen=True)
class CameraSpec:
    position: np.ndarray
    target: np.ndarray
    right: np.ndarray
    up: np.ndarray
    forward: np.ndarray
    focal_length: float
    horizontal_aperture: float
    vertical_aperture: float

    @property
    def tan_half_fov_x(self) -> float:
        return math.tan(math.atan(self.horizontal_aperture / (2.0 * self.focal_length)))

    @property
    def tan_half_fov_y(self) -> float:
        return math.tan(math.atan(self.vertical_aperture / (2.0 * self.focal_length)))


def _as_np3(value: Iterable[float]) -> np.ndarray:
    return np.array([float(v) for v in value], dtype=np.float64)


def _is_under_root(prim: Usd.Prim, root_path: str | None) -> bool:
    if not root_path:
        return True
    prim_path = str(prim.GetPath())
    return prim_path == root_path or prim_path.startswith(f"{root_path}/")


def _is_visible(prim: Usd.Prim) -> bool:
    imageable = UsdGeom.Imageable(prim)
    return imageable.ComputeVisibility() != UsdGeom.Tokens.invisible


def _triangulate(
    face_vertex_counts: Iterable[int], face_vertex_indices: Iterable[int]
) -> np.ndarray:
    triangles: list[tuple[int, int, int]] = []
    indices = [int(i) for i in face_vertex_indices]
    offset = 0
    for count_value in face_vertex_counts:
        count = int(count_value)
        if count >= 3:
            first = indices[offset]
            for i in range(1, count - 1):
                triangles.append((first, indices[offset + i], indices[offset + i + 1]))
        offset += count
    return np.array(triangles, dtype=np.int32)


def _stable_color(numeric_id: int) -> tuple[int, int, int]:
    # Hash-like palette with enough contrast against black background.
    hue = (numeric_id * 0.618033988749895) % 1.0
    saturation = 0.78
    value = 0.95
    i = int(hue * 6.0)
    f = hue * 6.0 - i
    p = value * (1.0 - saturation)
    q = value * (1.0 - f * saturation)
    t = value * (1.0 - (1.0 - f) * saturation)
    channel_sets = [
        (value, t, p),
        (q, value, p),
        (p, value, t),
        (p, q, value),
        (t, p, value),
        (value, p, q),
    ]
    rgb = channel_sets[i % 6]
    return tuple(int(round(c * 255)) for c in rgb)


def _bound_material_path(prim: Usd.Prim) -> str | None:
    try:
        material, _relationship = UsdShade.MaterialBindingAPI(
            prim
        ).ComputeBoundMaterial()
    except Exception:
        return None
    if material and material.GetPrim().IsValid():
        return str(material.GetPath())
    return None


def _extract_mesh_records(stage: Usd.Stage, root_path: str | None) -> list[MeshRecord]:
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    records: list[MeshRecord] = []
    numeric_id = 1

    for prim in stage.TraverseAll():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        if prim.IsInstanceProxy():
            continue
        if not _is_under_root(prim, root_path):
            continue
        if not _is_visible(prim):
            continue
        imageable = UsdGeom.Imageable(prim)
        purpose = imageable.ComputePurpose()
        if purpose not in (UsdGeom.Tokens.default_, UsdGeom.Tokens.render):
            continue

        mesh = UsdGeom.Mesh(prim)
        points_value = mesh.GetPointsAttr().Get()
        counts_value = mesh.GetFaceVertexCountsAttr().Get()
        indices_value = mesh.GetFaceVertexIndicesAttr().Get()
        if not points_value or not counts_value or not indices_value:
            continue

        local_points = np.array(points_value, dtype=np.float64)
        triangles = _triangulate(counts_value, indices_value)
        if local_points.size == 0 or triangles.size == 0:
            continue

        world_matrix = xform_cache.GetLocalToWorldTransform(prim)
        world_points = np.array(
            [
                tuple(
                    world_matrix.Transform(
                        Gf.Vec3d(float(p[0]), float(p[1]), float(p[2]))
                    )
                )
                for p in local_points
            ],
            dtype=np.float64,
        )

        records.append(
            MeshRecord(
                numeric_id=numeric_id,
                prim_path=str(prim.GetPath()),
                material_path=_bound_material_path(prim),
                parent_path=str(prim.GetParent().GetPath()),
                points_world=world_points,
                triangles=triangles,
                color_rgb=_stable_color(numeric_id),
            )
        )
        numeric_id += 1

    return records


def _mesh_records_from_prims(mesh_prims: list[Usd.Prim]) -> list[MeshRecord]:
    records: list[MeshRecord] = []
    for shape_index, prim in enumerate(mesh_prims):
        numeric_id = shape_index + 1
        records.append(
            MeshRecord(
                numeric_id=numeric_id,
                prim_path=str(prim.GetPath()),
                material_path=_bound_material_path(prim),
                parent_path=str(prim.GetParent().GetPath()),
                points_world=np.empty((0, 3), dtype=np.float64),
                triangles=np.empty((0, 3), dtype=np.int32),
                color_rgb=_stable_color(numeric_id),
            )
        )
    return records


def _extract_world_warp_meshes(
    stage: Usd.Stage,
    root_path: str | None,
    time_code: Usd.TimeCode,
    device: str,
) -> tuple[list[object], list[Usd.Prim]]:
    wp, _render_context_cls, _mesh_shape_type_int, _render_light_type = (
        render_warp._import_warp()
    )
    xform_cache = UsdGeom.XformCache(time_code)
    warp_meshes: list[object] = []
    mesh_prims: list[Usd.Prim] = []

    for prim in stage.TraverseAll():
        if not prim.IsA(UsdGeom.Mesh) or prim.IsInstanceProxy():
            continue
        if not _is_under_root(prim, root_path):
            continue
        if not _is_visible(prim):
            continue
        imageable = UsdGeom.Imageable(prim)
        purpose = imageable.ComputePurpose()
        if purpose not in (UsdGeom.Tokens.default_, UsdGeom.Tokens.render):
            continue

        mesh = UsdGeom.Mesh(prim)
        points_value = mesh.GetPointsAttr().Get(time_code)
        counts_value = mesh.GetFaceVertexCountsAttr().Get(time_code)
        indices_value = mesh.GetFaceVertexIndicesAttr().Get(time_code)
        if not points_value or not counts_value or not indices_value:
            continue

        local_points = np.array(points_value, dtype=np.float64)
        triangles = _triangulate(counts_value, indices_value)
        if local_points.size == 0 or triangles.size == 0:
            continue

        world_matrix = xform_cache.GetLocalToWorldTransform(prim)
        world_points = np.array(
            [
                tuple(
                    world_matrix.Transform(
                        Gf.Vec3d(float(point[0]), float(point[1]), float(point[2]))
                    )
                )
                for point in local_points
            ],
            dtype=np.float32,
        )
        triangle_indices = triangles.reshape(-1).astype(np.int32)
        warp_mesh = wp.Mesh(
            points=wp.array(world_points, dtype=wp.vec3f, device=device),
            indices=wp.array(
                triangle_indices,
                dtype=wp.int32,
                device=device,
            ),
        )
        warp_meshes.append(
            render_warp._RenderMesh(
                warp_mesh=warp_mesh,
                vertices=world_points,
                indices=triangle_indices,
            )
        )
        mesh_prims.append(prim)

    return warp_meshes, mesh_prims


def _target_prim_or_root(stage: Usd.Stage, root_path: str | None) -> Usd.Prim:
    if root_path:
        prim = stage.GetPrimAtPath(root_path)
        if prim and prim.IsValid():
            return prim
        raise ValueError(f"Prim path not found: {root_path}")
    return stage.GetPseudoRoot()


def _make_camera(
    stage: Usd.Stage,
    root_path: str | None,
    direction: str,
    margin: float,
    focal_length: float,
    horizontal_aperture: float,
    vertical_aperture: float,
) -> CameraSpec:
    focus_prim = _target_prim_or_root(stage, root_path)
    bbox = get_bbox_from_prim(focus_prim)
    aligned = bbox.ComputeAlignedRange()
    bbox_min = aligned.GetMin()
    bbox_max = aligned.GetMax()
    framing_fn = (
        compute_camera_framing_position_sides
        if direction in _SIDE_VIEW_DIRECTIONS
        else compute_camera_framing_position_corners
    )
    cam_position, target = framing_fn(
        bbox_min=(bbox_min[0], bbox_min[1], bbox_min[2]),
        bbox_max=(bbox_max[0], bbox_max[1], bbox_max[2]),
        direction=direction,
        margin=margin,
        min_distance=0,
        focal_length=focal_length,
        horizontal_aperture=horizontal_aperture,
        vertical_aperture=vertical_aperture,
    )

    position_np = _as_np3(cam_position)
    target_np = _as_np3(target)
    forward = target_np - position_np
    forward /= np.linalg.norm(forward)

    stage_up_axis = UsdGeom.GetStageUpAxis(stage)
    world_up = np.array(
        [0.0, 1.0, 0.0] if stage_up_axis == UsdGeom.Tokens.y else [0.0, 0.0, 1.0]
    )
    if abs(float(np.dot(forward, world_up))) > 0.98:
        world_up = np.array(
            [0.0, 0.0, 1.0] if stage_up_axis == UsdGeom.Tokens.y else [0.0, 1.0, 0.0]
        )
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)

    return CameraSpec(
        position=position_np,
        target=target_np,
        right=right,
        up=up,
        forward=forward,
        focal_length=focal_length,
        horizontal_aperture=horizontal_aperture,
        vertical_aperture=vertical_aperture,
    )


def _add_probe_camera_to_stage(
    stage: Usd.Stage,
    root_path: str | None,
    direction: str,
    margin: float,
    focal_length: float,
    horizontal_aperture: float,
    vertical_aperture: float,
) -> str:
    camera_path = "/VisualGroundingProbeCamera"
    if root_path:
        focus_prim = _target_prim_or_root(stage, root_path)
        add_camera_fn = (
            add_focused_side_view_camera
            if direction in _SIDE_VIEW_DIRECTIONS
            else add_focused_corner_view_camera
        )
        add_camera_fn(
            prim_to_focus=focus_prim,
            camera_path=camera_path,
            direction=direction,
            margin=margin,
            min_distance=0,
            focal_length=focal_length,
            horizontal_aperture=horizontal_aperture,
            vertical_aperture=vertical_aperture,
            near_clip_margin=0.1,
            far_clip_margin=0.1,
        )
    else:
        add_camera_fn = (
            add_side_view_camera
            if direction in _SIDE_VIEW_DIRECTIONS
            else add_corner_view_camera
        )
        add_camera_fn(
            stage=stage,
            camera_path=camera_path,
            direction=direction,
            margin=margin,
            min_distance=0,
            focal_length=focal_length,
            horizontal_aperture=horizontal_aperture,
            vertical_aperture=vertical_aperture,
            near_clip_margin=0.1,
            far_clip_margin=0.1,
        )
    return camera_path


def _render_id_buffer_with_warp(
    stage: Usd.Stage,
    root_path: str | None,
    width: int,
    height: int,
    direction: str,
    margin: float,
    focal_length: float,
    horizontal_aperture: float,
    vertical_aperture: float,
    device: str,
) -> tuple[np.ndarray, list[MeshRecord]]:
    if root_path:
        hide_prims_outside_subtree(stage, root_path)
    camera_path = _add_probe_camera_to_stage(
        stage=stage,
        root_path=root_path,
        direction=direction,
        margin=margin,
        focal_length=focal_length,
        horizontal_aperture=horizontal_aperture,
        vertical_aperture=vertical_aperture,
    )

    wp, _render_context_cls, _mesh_shape_type_int, _render_light_type = (
        render_warp._import_warp()
    )
    wp.init()
    time_code = Usd.TimeCode(0)
    warp_meshes, mesh_prims = _extract_world_warp_meshes(
        stage=stage,
        root_path=root_path,
        time_code=time_code,
        device=device,
    )
    if not warp_meshes:
        raise RuntimeError("Warp extracted no meshes from the stage")

    ctx = render_warp._setup_render_context(
        warp_meshes=warp_meshes,
        mesh_prims=mesh_prims,
        time_code=time_code,
        device=device,
        enable_shadows=False,
        enable_backface_culling=False,
        color_boost=1.0,
    )
    identity_transforms = np.array(
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]] * len(mesh_prims),
        dtype=np.float32,
    )
    ctx.shape_transforms = wp.array(
        identity_transforms,
        dtype=wp.transformf,
        device=device,
    )

    visible = [
        i
        for i, prim in enumerate(mesh_prims)
        if render_warp._is_visible(prim, time_code)
    ]
    ctx.shape_enabled = wp.array(
        np.array(visible, dtype=np.uint32), dtype=wp.uint32, device=device
    )
    ctx.shape_count_enabled = len(visible)
    ctx.bvh_shapes = None
    ctx.bvh_shapes_lowers = None
    ctx.bvh_shapes_uppers = None
    ctx.bvh_shapes_groups = None
    ctx.bvh_shapes_group_roots = None

    camera_fov = render_warp._compute_camera_fov(stage, camera_path, time_code)
    camera_rays = ctx.utils.compute_pinhole_camera_rays(
        width,
        height,
        wp.array([camera_fov], dtype=wp.float32, device=device),
    )
    camera_xforms = render_warp._get_camera_transforms(stage, [camera_path], time_code)
    camera_transforms = wp.array(
        np.array(camera_xforms, dtype=np.float32),
        dtype=wp.transformf,
        device=device,
    ).reshape((1, 1))
    shape_index_image = ctx.create_shape_index_image_output(width, height, 1)
    ctx.render(
        camera_transforms=camera_transforms,
        camera_rays=camera_rays,
        shape_index_image=shape_index_image,
    )
    wp.synchronize_device(device)

    shape_indices = shape_index_image.numpy()[0, 0]
    max_uint = np.iinfo(shape_indices.dtype).max
    id_buffer = np.where(
        shape_indices == max_uint, 0, shape_indices.astype(np.int64) + 1
    )
    return id_buffer.astype(np.int32), _mesh_records_from_prims(mesh_prims)


def _project_points(
    points_world: np.ndarray, camera: CameraSpec, width: int, height: int
) -> tuple[np.ndarray, np.ndarray]:
    rel = points_world - camera.position
    cam_x = rel @ camera.right
    cam_y = rel @ camera.up
    depth = rel @ camera.forward

    safe_depth = np.maximum(depth, 1.0e-6)
    ndc_x = cam_x / (safe_depth * camera.tan_half_fov_x)
    ndc_y = cam_y / (safe_depth * camera.tan_half_fov_y)
    screen_x = (ndc_x + 1.0) * 0.5 * float(width - 1)
    screen_y = (1.0 - ndc_y) * 0.5 * float(height - 1)
    return np.stack([screen_x, screen_y, depth], axis=1), depth


def _rasterize_triangle(
    tri: np.ndarray,
    numeric_id: int,
    z_buffer: np.ndarray,
    id_buffer: np.ndarray,
) -> None:
    height, width = z_buffer.shape
    if np.any(tri[:, 2] <= 1.0e-5):
        return

    x0, y0, z0 = tri[0]
    x1, y1, z1 = tri[1]
    x2, y2, z2 = tri[2]

    min_x = max(int(math.floor(min(x0, x1, x2))), 0)
    max_x = min(int(math.ceil(max(x0, x1, x2))), width - 1)
    min_y = max(int(math.floor(min(y0, y1, y2))), 0)
    max_y = min(int(math.ceil(max(y0, y1, y2))), height - 1)
    if min_x > max_x or min_y > max_y:
        return

    area = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
    if abs(area) < 1.0e-8:
        return

    xs = np.arange(min_x, max_x + 1, dtype=np.float64) + 0.5
    ys = np.arange(min_y, max_y + 1, dtype=np.float64) + 0.5
    grid_x, grid_y = np.meshgrid(xs, ys)

    w0 = ((x1 - grid_x) * (y2 - grid_y) - (y1 - grid_y) * (x2 - grid_x)) / area
    w1 = ((x2 - grid_x) * (y0 - grid_y) - (y2 - grid_y) * (x0 - grid_x)) / area
    w2 = 1.0 - w0 - w1
    inside = (w0 >= -1.0e-6) & (w1 >= -1.0e-6) & (w2 >= -1.0e-6)
    if not np.any(inside):
        return

    depth = w0 * z0 + w1 * z1 + w2 * z2
    region_z = z_buffer[min_y : max_y + 1, min_x : max_x + 1]
    region_id = id_buffer[min_y : max_y + 1, min_x : max_x + 1]
    update = inside & (depth < region_z)
    region_z[update] = depth[update]
    region_id[update] = numeric_id


def _rasterize(
    records: list[MeshRecord],
    camera: CameraSpec,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    z_buffer = np.full((height, width), np.inf, dtype=np.float64)
    id_buffer = np.zeros((height, width), dtype=np.int32)

    for record in records:
        projected, _depth = _project_points(record.points_world, camera, width, height)
        for tri_indices in record.triangles:
            tri = projected[tri_indices]
            _rasterize_triangle(tri, record.numeric_id, z_buffer, id_buffer)

    return id_buffer, z_buffer


def _segmentation_image(
    id_buffer: np.ndarray, records_by_id: dict[int, MeshRecord]
) -> Image.Image:
    height, width = id_buffer.shape
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    for numeric_id, record in records_by_id.items():
        rgb[id_buffer == numeric_id] = record.color_rgb
    return Image.fromarray(rgb, mode="RGB")


def _visible_surface_anchor(
    *,
    xs: np.ndarray,
    ys: np.ndarray,
    bbox: list[int],
) -> list[float]:
    """Choose a label anchor that is guaranteed to be on visible prim pixels."""
    bbox_center_x = (float(bbox[0]) + float(bbox[2])) / 2.0
    bbox_center_y = (float(bbox[1]) + float(bbox[3])) / 2.0
    centroid_x = float(np.mean(xs))
    centroid_y = float(np.mean(ys))

    x_values = xs.astype(np.float64)
    y_values = ys.astype(np.float64)
    distance_to_bbox_center = (x_values - bbox_center_x) ** 2 + (
        y_values - bbox_center_y
    ) ** 2
    distance_to_centroid = (x_values - centroid_x) ** 2 + (y_values - centroid_y) ** 2
    selected_index = int(np.lexsort((distance_to_centroid, distance_to_bbox_center))[0])
    return [float(xs[selected_index]), float(ys[selected_index])]


def _visible_entries(
    id_buffer: np.ndarray,
    records_by_id: dict[int, MeshRecord],
    min_visible_pixels: int,
) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for numeric_id in sorted(records_by_id):
        ys, xs = np.where(id_buffer == numeric_id)
        visible_pixels = int(len(xs))
        if visible_pixels < min_visible_pixels:
            continue
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        bbox_center = [
            (float(bbox[0]) + float(bbox[2])) / 2.0,
            (float(bbox[1]) + float(bbox[3])) / 2.0,
        ]
        visible_centroid = [float(np.mean(xs)), float(np.mean(ys))]
        label_xy = _visible_surface_anchor(xs=xs, ys=ys, bbox=bbox)
        record = records_by_id[numeric_id]
        entries.append(
            {
                "id": numeric_id,
                "prim_path": record.prim_path,
                "material_path": record.material_path,
                "parent_path": record.parent_path,
                "visible_pixels": visible_pixels,
                "bbox_xyxy": bbox,
                "bbox_center_xy": bbox_center,
                "visible_centroid_xy": visible_centroid,
                "label_xy": label_xy,
                "label_anchor_mode": "center_most_visible_pixel",
                "color_rgb": list(record.color_rgb),
            }
        )
    entries.sort(key=lambda item: int(item["visible_pixels"]), reverse=True)
    return entries


def _draw_labeled_overlay(
    base: Image.Image,
    entries: list[dict[str, object]],
    max_labels: int,
    label_mode: str,
) -> Image.Image:
    image = base.convert("RGBA")
    if label_mode == "center":
        _draw_centered_labels(image, entries[:max_labels], rounded=True)
    else:
        image = _draw_callout_label_overlay(
            image,
            entries[:max_labels],
            rounded=True,
        )
    return image.convert("RGB")


def _draw_beauty_label_overlay(
    beauty_path: Path,
    entries: list[dict[str, object]],
    max_labels: int,
    target_size: tuple[int, int],
    label_mode: str,
) -> Image.Image | None:
    if not beauty_path.exists():
        return None
    image = Image.open(beauty_path).convert("RGBA").resize(target_size, Image.LANCZOS)
    if label_mode == "center":
        _draw_centered_labels(image, entries[:max_labels], rounded=False)
    else:
        image = _draw_callout_label_overlay(
            image,
            entries[:max_labels],
            rounded=False,
        )
    return image.convert("RGB")


def _load_label_font(size: int = 14) -> ImageFont.ImageFont:
    for font_path in (
        "DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ):
        try:
            return ImageFont.truetype(font_path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_centered_labels(
    image: Image.Image,
    entries: list[dict[str, object]],
    rounded: bool,
) -> None:
    draw = ImageDraw.Draw(image, "RGBA")
    font = _load_label_font()
    for entry in entries:
        x, y = entry["label_xy"]  # type: ignore[index]
        label = str(entry["id"])
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        left = int(max(2, min(float(x) - text_w / 2.0 - 3, image.width - text_w - 8)))
        top = int(max(2, min(float(y) - text_h / 2.0 - 3, image.height - text_h - 8)))
        box = [left, top, left + text_w + 6, top + text_h + 6]
        if rounded:
            draw.rounded_rectangle(
                box,
                radius=3,
                fill=(0, 0, 0, 210),
                outline=(255, 255, 255, 230),
                width=1,
            )
        else:
            draw.rectangle(
                box,
                fill=(0, 0, 0, 180),
                outline=(255, 255, 255, 230),
                width=1,
            )
        draw.text((left + 3, top + 2), label, font=font, fill=(255, 255, 255, 255))


def _draw_callout_label_overlay(
    image: Image.Image,
    entries: list[dict[str, object]],
    rounded: bool,
) -> Image.Image:
    if not entries:
        return image

    font = _load_label_font()
    measurement = ImageDraw.Draw(image, "RGBA")
    max_label_w = 0
    max_label_h = 0
    for entry in entries:
        label = str(entry["id"])
        bbox = measurement.textbbox((0, 0), label, font=font)
        max_label_w = max(max_label_w, bbox[2] - bbox[0])
        max_label_h = max(max_label_h, bbox[3] - bbox[1])

    gutter_x = max(_CALLOUT_MIN_GUTTER_X, max_label_w + 48)
    gutter_y = max(_CALLOUT_MIN_GUTTER_Y, max_label_h + 14)
    canvas = Image.new(
        "RGBA",
        (image.width + gutter_x * 2, image.height + gutter_y * 2),
        (0, 0, 0, 255),
    )
    canvas.paste(image, (gutter_x, gutter_y))
    shifted_entries = [
        _shift_overlay_entry(entry, x_offset=gutter_x, y_offset=gutter_y)
        for entry in entries
    ]
    _draw_callout_labels(canvas, shifted_entries, rounded=rounded, font=font)
    return canvas


def _shift_overlay_entry(
    entry: dict[str, object],
    x_offset: int,
    y_offset: int,
) -> dict[str, object]:
    shifted = dict(entry)
    label_xy = entry["label_xy"]  # type: ignore[index]
    shifted["label_xy"] = [
        float(label_xy[0]) + float(x_offset),
        float(label_xy[1]) + float(y_offset),
    ]
    bbox = entry["bbox_xyxy"]  # type: ignore[index]
    shifted["bbox_xyxy"] = [
        int(bbox[0]) + x_offset,
        int(bbox[1]) + y_offset,
        int(bbox[2]) + x_offset,
        int(bbox[3]) + y_offset,
    ]
    return shifted


def _draw_callout_labels(
    image: Image.Image,
    entries: list[dict[str, object]],
    rounded: bool,
    font: ImageFont.ImageFont | None = None,
) -> None:
    if not entries:
        return

    draw = ImageDraw.Draw(image, "RGBA")
    if font is None:
        font = _load_label_font()
    union = _union_bbox(entries)
    object_center_x = (union[0] + union[2]) / 2.0
    left_entries: list[dict[str, object]] = []
    right_entries: list[dict[str, object]] = []
    for entry in entries:
        x, _y = entry["label_xy"]  # type: ignore[index]
        if float(x) < object_center_x:
            left_entries.append(entry)
        else:
            right_entries.append(entry)

    _draw_callout_side(
        draw=draw,
        image=image,
        font=font,
        entries=left_entries,
        side="left",
        union=union,
        rounded=rounded,
    )
    _draw_callout_side(
        draw=draw,
        image=image,
        font=font,
        entries=right_entries,
        side="right",
        union=union,
        rounded=rounded,
    )


def _union_bbox(entries: list[dict[str, object]]) -> tuple[int, int, int, int]:
    bboxes = [entry["bbox_xyxy"] for entry in entries]
    x0 = min(int(bbox[0]) for bbox in bboxes)  # type: ignore[index]
    y0 = min(int(bbox[1]) for bbox in bboxes)  # type: ignore[index]
    x1 = max(int(bbox[2]) for bbox in bboxes)  # type: ignore[index]
    y1 = max(int(bbox[3]) for bbox in bboxes)  # type: ignore[index]
    return x0, y0, x1, y1


def _spread_positions(
    desired: list[float],
    min_y: float,
    max_y: float,
    min_gap: float,
) -> list[float]:
    if not desired:
        return []
    values = [min(max(y, min_y), max_y) for y in desired]
    values.sort()
    for i in range(1, len(values)):
        values[i] = max(values[i], values[i - 1] + min_gap)
    overflow = values[-1] - max_y
    if overflow > 0:
        values = [value - overflow for value in values]
    for i in range(len(values) - 2, -1, -1):
        values[i] = min(values[i], values[i + 1] - min_gap)
    underflow = min_y - values[0]
    if underflow > 0:
        values = [value + underflow for value in values]
    return values


def _draw_callout_side(
    draw: ImageDraw.ImageDraw,
    image: Image.Image,
    font: ImageFont.ImageFont,
    entries: list[dict[str, object]],
    side: str,
    union: tuple[int, int, int, int],
    rounded: bool,
) -> None:
    if not entries:
        return

    entries.sort(key=lambda entry: float(entry["label_xy"][1]))  # type: ignore[index]
    text_metrics: list[tuple[dict[str, object], int, int]] = []
    for entry in entries:
        label = str(entry["id"])
        bbox = draw.textbbox((0, 0), label, font=font)
        text_metrics.append((entry, bbox[2] - bbox[0], bbox[3] - bbox[1]))

    max_label_w = max(width for _entry, width, _height in text_metrics)
    max_label_h = max(height for _entry, _width, height in text_metrics)
    min_gap = max_label_h + 8
    min_y = max(4.0, float(union[1]) - 28.0)
    max_y = min(float(image.height - max_label_h - 10), float(union[3]) + 28.0)
    if max_y - min_y < min_gap * max(len(entries) - 1, 0):
        min_y = 4.0
        max_y = float(image.height - max_label_h - 10)

    desired = [float(entry["label_xy"][1]) for entry in entries]  # type: ignore[index]
    label_ys = _spread_positions(desired, min_y, max_y, min_gap)

    if side == "left":
        label_x = max(4, union[0] - max_label_w - 26)
    else:
        label_x = min(image.width - max_label_w - 10, union[2] + 18)

    for (entry, text_w, text_h), label_y in zip(text_metrics, label_ys, strict=True):
        anchor_x, anchor_y = entry["label_xy"]  # type: ignore[index]
        label = str(entry["id"])
        left = int(label_x)
        top = int(label_y)
        box = [left, top, left + text_w + 6, top + text_h + 6]
        line_start_x = box[2] if side == "left" else box[0]
        line_start_y = top + (text_h + 6) // 2
        line_fill = (255, 255, 255, 205)
        shadow_fill = (0, 0, 0, 190)
        draw.line(
            [
                (line_start_x + 1, line_start_y + 1),
                (float(anchor_x) + 1, float(anchor_y) + 1),
            ],
            fill=shadow_fill,
            width=2,
        )
        draw.line(
            [(line_start_x, line_start_y), (float(anchor_x), float(anchor_y))],
            fill=line_fill,
            width=1,
        )
        draw.ellipse(
            [
                float(anchor_x) - 2,
                float(anchor_y) - 2,
                float(anchor_x) + 2,
                float(anchor_y) + 2,
            ],
            fill=(255, 255, 255, 230),
            outline=(0, 0, 0, 230),
        )
        if rounded:
            draw.rounded_rectangle(
                box,
                radius=3,
                fill=(0, 0, 0, 220),
                outline=(255, 255, 255, 235),
                width=1,
            )
        else:
            draw.rectangle(
                box,
                fill=(0, 0, 0, 190),
                outline=(255, 255, 255, 235),
                width=1,
            )
        draw.text((left + 3, top + 2), label, font=font, fill=(255, 255, 255, 255))


def _write_legend_csv(path: Path, entries: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "visible_pixels",
                "bbox_xyxy",
                "prim_path",
                "material_path",
                "parent_path",
                "color_rgb",
            ],
        )
        writer.writeheader()
        for entry in entries:
            writer.writerow(
                {
                    "id": entry["id"],
                    "visible_pixels": entry["visible_pixels"],
                    "bbox_xyxy": json.dumps(entry["bbox_xyxy"]),
                    "prim_path": entry["prim_path"],
                    "material_path": entry["material_path"] or "",
                    "parent_path": entry["parent_path"],
                    "color_rgb": json.dumps(entry["color_rgb"]),
                }
            )


def _write_html_report(
    output_dir: Path,
    args: argparse.Namespace,
    entries: list[dict[str, object]],
    elapsed_seconds: float,
    total_meshes: int,
) -> None:
    beauty_block = ""
    if (output_dir / "materialized_labeled_overlay.png").exists():
        beauty_block = """
          <figure>
            <img src="materialized_labeled_overlay.png" alt="Materialized render with labels">
            <figcaption>Materialized render with projected IDs</figcaption>
          </figure>
        """
    rows = "\n".join(
        "<tr>"
        f"<td>{entry['id']}</td>"
        f"<td>{entry['visible_pixels']}</td>"
        f"<td><code>{html.escape(str(entry['prim_path']))}</code></td>"
        f"<td><code>{html.escape(str(entry['material_path'] or ''))}</code></td>"
        f"<td><code>{html.escape(str(entry['parent_path']))}</code></td>"
        "</tr>"
        for entry in entries
    )
    doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Visual Grounding Probe</title>
  <style>
    body {{
      margin: 24px;
      background: #101214;
      color: #e9edf1;
      font-family: Arial, sans-serif;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 16px;
      align-items: start;
    }}
    figure {{ margin: 0; }}
    img {{
      width: 100%;
      height: auto;
      background: #000;
      border: 1px solid #343a40;
    }}
    figcaption {{
      margin-top: 6px;
      color: #b8c0ca;
      font-size: 13px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 18px;
      font-size: 13px;
    }}
    th, td {{
      border-bottom: 1px solid #30363d;
      padding: 8px;
      vertical-align: top;
      text-align: left;
    }}
    th {{ color: #d7dee8; background: #1a1f25; position: sticky; top: 0; }}
    code {{ color: #f1f5f9; white-space: nowrap; }}
    .meta {{ color: #b8c0ca; margin-bottom: 16px; }}
  </style>
</head>
<body>
  <h1>Visual Grounding Probe</h1>
  <p class="meta">
    USD: <code>{html.escape(str(args.usd))}</code><br>
    Root: <code>{html.escape(str(args.prim_path or "full stage"))}</code><br>
    Direction: <code>{html.escape(args.direction)}</code>,
    rasterizer: <code>{html.escape(args.rasterizer)}</code>,
    labels: <code>{html.escape(args.label_mode)}</code>,
    visible meshes: {len(entries)} / {total_meshes},
    raster time: {elapsed_seconds:.2f}s
  </p>
  <div class="grid">
    {beauty_block}
    <figure>
      <img src="object_id_segmentation.png" alt="Object-ID segmentation">
      <figcaption>Flat object-ID segmentation</figcaption>
    </figure>
    <figure>
      <img src="object_id_labeled_overlay.png" alt="Object-ID labeled overlay">
      <figcaption>Readable numeric labels over visible prim regions</figcaption>
    </figure>
  </div>
  <table>
    <thead>
      <tr>
        <th>ID</th>
        <th>Visible Pixels</th>
        <th>Prim Path</th>
        <th>Material Binding</th>
        <th>Parent</th>
      </tr>
    </thead>
    <tbody>
      {rows}
    </tbody>
  </table>
</body>
</html>
"""
    (output_dir / "index.html").write_text(doc, encoding="utf-8")


def generate_visual_grounding_packet(
    *,
    usd_path: str | Path,
    output_dir: str | Path,
    prim_path: str | None = None,
    beauty_image_path: str | Path | None = None,
    direction: str = "+x+y+z",
    width: int | None = None,
    height: int | None = None,
    rasterizer: str = "cpu",
    device: str = "cuda:0",
    camera_margin: float = 1.0,
    focal_length: float = 50.0,
    horizontal_aperture: float = 36.0,
    vertical_aperture: float = 36.0,
    max_labels: int = 32,
    label_mode: str = "callout",
    min_visible_pixels: int = 64,
) -> dict[str, object]:
    """Generate visual grounding artifacts for a USD stage.

    The returned packet is intentionally file-oriented so a coding harness can
    attach images to its context while still having a machine-readable legend
    that maps visible IDs back to USD prims and current material bindings.
    """
    usd_path = Path(usd_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    beauty_path = (
        Path(beauty_image_path).expanduser().resolve() if beauty_image_path else None
    )
    if width is None or height is None:
        if beauty_path and beauty_path.exists():
            with Image.open(beauty_path) as beauty:
                image_width, image_height = beauty.size
            width = width or image_width
            height = height or image_height
        else:
            width = width or 768
            height = height or width
    if width is None or height is None or width <= 0 or height <= 0:
        raise ValueError("width and height must be positive integers")

    if rasterizer not in {"cpu", "warp"}:
        raise ValueError("rasterizer must be 'cpu' or 'warp'")
    if label_mode not in {"callout", "center"}:
        raise ValueError("label_mode must be 'callout' or 'center'")

    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Failed to open USD: {usd_path}")

    if rasterizer == "warp":
        id_buffer, records = _render_id_buffer_with_warp(
            stage=stage,
            root_path=prim_path,
            width=width,
            height=height,
            direction=direction,
            margin=camera_margin,
            focal_length=focal_length,
            horizontal_aperture=horizontal_aperture,
            vertical_aperture=vertical_aperture,
            device=device,
        )
        camera_payload: dict[str, object] = {
            "mode": "warp_shape_index_image",
            "device": device,
        }
    else:
        records = _extract_mesh_records(stage, prim_path)
        if not records:
            raise RuntimeError("No mesh records found for requested USD/root prim")

        camera = _make_camera(
            stage=stage,
            root_path=prim_path,
            direction=direction,
            margin=camera_margin,
            focal_length=focal_length,
            horizontal_aperture=horizontal_aperture,
            vertical_aperture=vertical_aperture,
        )
        id_buffer, _z_buffer = _rasterize(records, camera, width, height)
        camera_payload = {
            "mode": "cpu_triangle_rasterizer",
            "position": camera.position.tolist(),
            "target": camera.target.tolist(),
            "forward": camera.forward.tolist(),
            "right": camera.right.tolist(),
            "up": camera.up.tolist(),
            "focal_length": camera.focal_length,
            "horizontal_aperture": camera.horizontal_aperture,
            "vertical_aperture": camera.vertical_aperture,
        }

    records_by_id = {record.numeric_id: record for record in records}
    entries = _visible_entries(id_buffer, records_by_id, min_visible_pixels)
    segmentation = _segmentation_image(id_buffer, records_by_id)
    labeled_overlay = _draw_labeled_overlay(
        segmentation,
        entries,
        max_labels,
        label_mode,
    )
    segmentation_path = output_dir / "object_id_segmentation.png"
    labeled_overlay_path = output_dir / "object_id_labeled_overlay.png"
    segmentation.save(segmentation_path)
    labeled_overlay.save(labeled_overlay_path)

    beauty_overlay_path: Path | None = None
    if beauty_path:
        beauty_overlay = _draw_beauty_label_overlay(
            beauty_path,
            entries,
            max_labels,
            target_size=(width, height),
            label_mode=label_mode,
        )
        if beauty_overlay:
            beauty_overlay_path = output_dir / "materialized_labeled_overlay.png"
            beauty_overlay.save(beauty_overlay_path)

    elapsed = time.time() - start
    legend_path = output_dir / "legend.json"
    legend_csv_path = output_dir / "legend.csv"
    html_path = output_dir / "index.html"
    payload = {
        "schema_version": "material-visual-grounding-packet/v1",
        "source_task": "world_understanding.validation.visual_grounding.generate_visual_grounding_packet",
        "usd": str(usd_path),
        "prim_path": prim_path,
        "direction": direction,
        "image_size": [width, height],
        "raster_seconds": elapsed,
        "rasterizer": rasterizer,
        "label_mode": label_mode,
        "total_meshes": len(records),
        "visible_entries": entries,
        "camera": camera_payload,
        "artifacts": {
            "segmentation_image_path": str(segmentation_path),
            "object_id_labeled_overlay_path": str(labeled_overlay_path),
            "materialized_labeled_overlay_path": (
                str(beauty_overlay_path) if beauty_overlay_path else None
            ),
            "beauty_labeled_overlay_path": (
                str(beauty_overlay_path) if beauty_overlay_path else None
            ),
            "legend_json_path": str(legend_path),
            "legend_csv_path": str(legend_csv_path),
            "html_report_path": str(html_path),
        },
    }
    legend_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_legend_csv(legend_csv_path, entries)
    html_args = argparse.Namespace(
        usd=usd_path,
        prim_path=prim_path,
        direction=direction,
        rasterizer=rasterizer,
        label_mode=label_mode,
        beauty_image=beauty_path,
    )
    _write_html_report(output_dir, html_args, entries, elapsed, len(records))
    return payload


__all__ = [
    "CameraSpec",
    "MeshRecord",
    "generate_visual_grounding_packet",
]
