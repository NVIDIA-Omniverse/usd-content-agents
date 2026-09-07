# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OVRTX-to-VoMP orchestration for rigid-body mass authoring."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from physics_agent.integrations.vomp import (
    VompApplyResult,
    VompIntegrationError,
    _matrix4_array,
    _preflight_vomp_mass_target,
    _reload_and_hash_composition_layers,
    _require_sha256,
    _transform_points,
    _validate_static_mesh_geometry,
    _validate_static_transform_chain,
    _validate_vomp_input_layer,
    _verify_composition_hashes,
    apply_vomp_mass_properties,
)
from physics_agent.integrations.vomp_defaults import (
    DEFAULT_VOMP_CAMERA_RADIUS,
    DEFAULT_VOMP_FOV_DEGREES,
    DEFAULT_VOMP_IMAGE_HEIGHT,
    DEFAULT_VOMP_IMAGE_WIDTH,
    DEFAULT_VOMP_MATERIAL_TARGET,
    DEFAULT_VOMP_NUM_SENSOR_UPDATES,
    DEFAULT_VOMP_NUM_VIEWS,
    DEFAULT_VOMP_OVRTX_VENV_DIR,
    DEFAULT_VOMP_RENDER_MODE,
    DEFAULT_VOMP_SEED,
)
from physics_agent.integrations.vomp_deformable import (
    DEFAULT_MAX_DEFORMABLE_VOXELS,
    MaterialReductionPolicy,
    VompDeformableApplyResult,
    _positive_integer,
    _preflight_vomp_deformable_target,
    apply_vomp_volume_deformable,
)
from physics_agent.integrations.vomp_runtime import (
    ExternalVompRunner,
    VompRunner,
    VompRunRequest,
    VompRuntimeConfig,
    _validate_runtime,
)


@dataclass(frozen=True)
class VompRenderConfig:
    """Calibrated OVRTX evidence settings compatible with VoMP training."""

    num_views: int = DEFAULT_VOMP_NUM_VIEWS
    image_width: int = DEFAULT_VOMP_IMAGE_WIDTH
    image_height: int = DEFAULT_VOMP_IMAGE_HEIGHT
    radius: float = DEFAULT_VOMP_CAMERA_RADIUS
    fov_degrees: float = DEFAULT_VOMP_FOV_DEGREES
    seed: int = DEFAULT_VOMP_SEED
    render_mode: str = DEFAULT_VOMP_RENDER_MODE
    num_sensor_updates: int = DEFAULT_VOMP_NUM_SENSOR_UPDATES
    material_target: str = DEFAULT_VOMP_MATERIAL_TARGET
    ovrtx_venv_dir: str | None = DEFAULT_VOMP_OVRTX_VENV_DIR


@dataclass(frozen=True)
class VompPreparedEvidence:
    """Mesh, calibrated renders, and provenance prepared for VoMP."""

    artifact_dir: Path
    mesh_path: Path
    metadata_path: Path
    render_stage_path: Path
    manifest_path: Path
    manifest: dict[str, Any]
    composition_hashes: Mapping[Path, str]


@dataclass(frozen=True)
class VompPipelineResult:
    """Complete OVRTX -> VoMP -> USD mass-authoring result."""

    apply_result: VompApplyResult
    evidence: VompPreparedEvidence
    vomp_npz_path: Path
    worker_manifest_path: Path
    worker_log_path: Path


@dataclass(frozen=True)
class VompDeformablePipelineResult:
    """Complete OVRTX -> VoMP -> volume-deformable authoring result."""

    apply_result: VompDeformableApplyResult
    evidence: VompPreparedEvidence
    vomp_npz_path: Path
    worker_manifest_path: Path
    worker_log_path: Path


@dataclass(frozen=True)
class _WorldMesh:
    vertices_m: np.ndarray
    triangles: np.ndarray
    center_m: np.ndarray
    scale_m: float
    source_mesh_count: int
    unique_mesh_count: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.stem}_",
        suffix=path.suffix,
        dir=path.parent,
        delete=False,
    )
    temporary = Path(stream.name)
    try:
        with stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_render_config(config: VompRenderConfig) -> None:
    for name, value in (
        ("num_views", config.num_views),
        ("image_width", config.image_width),
        ("image_height", config.image_height),
        ("num_sensor_updates", config.num_sensor_updates),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise VompIntegrationError(f"VoMP render {name} must be a positive integer")
    if not math.isfinite(config.radius) or config.radius <= 0.5:
        raise VompIntegrationError("VoMP camera radius must be finite and above 0.5")
    if not math.isfinite(config.fov_degrees) or not 1.0 < config.fov_degrees < 179.0:
        raise VompIntegrationError("VoMP camera FOV must be between 1 and 179 degrees")
    if config.render_mode not in {"rt1", "rt2", "pt"}:
        raise VompIntegrationError("VoMP OVRTX render_mode must be rt1, rt2, or pt")
    if config.material_target not in {"auto", "preview_surface", "openpbr_materialx"}:
        raise VompIntegrationError("VoMP OVRTX material_target is unsupported")


def _sample_views(config: VompRenderConfig) -> list[dict[str, float | int]]:
    random = np.random.RandomState(config.seed)
    offset = (float(random.rand()), float(random.rand()))

    def radical_inverse(base: int, number: int) -> float:
        value = 0.0
        inverse_base = 1.0 / base
        inverse_power = inverse_base
        while number > 0:
            value += (number % base) * inverse_power
            number //= base
            inverse_power *= inverse_base
        return value

    views: list[dict[str, float | int]] = []
    for index in range(config.num_views):
        u = index / config.num_views + offset[0] / config.num_views
        v = radical_inverse(2, index) + offset[1]
        u = 2.0 * u if u < 0.25 else (2.0 / 3.0) * u + 1.0 / 3.0
        pitch = math.acos(1.0 - 2.0 * u) - math.pi / 2.0
        yaw = v * 2.0 * math.pi
        views.append(
            {
                "index": index,
                "yaw": yaw,
                "pitch": pitch,
                "radius": config.radius,
                "fovDegrees": config.fov_degrees,
            }
        )
    return views


def _cross_2d(points: np.ndarray, a: int, b: int, c: int) -> float:
    first = points[b] - points[a]
    second = points[c] - points[a]
    return float(first[0] * second[1] - first[1] * second[0])


def _triangulate_faces(
    points: np.ndarray,
    face_counts: np.ndarray,
    face_indices: np.ndarray,
    hole_indices: set[int],
) -> np.ndarray:
    if len(face_counts) and np.all(face_counts == 3):
        expected_indices = len(face_counts) * 3
        if len(face_indices) < expected_indices:
            raise VompIntegrationError("VoMP target mesh has malformed face indices")
        if len(face_indices) > expected_indices:
            raise VompIntegrationError("VoMP target mesh has trailing face indices")
        all_triangles = face_indices.reshape((-1, 3))
        included = np.fromiter(
            (index not in hole_indices for index in range(len(all_triangles))),
            dtype=np.bool_,
            count=len(all_triangles),
        )
        triangle_faces = all_triangles[included]
        if not len(triangle_faces):
            return np.empty((0, 3), dtype=np.int64)
        if np.any(triangle_faces < 0) or np.any(triangle_faces >= len(points)):
            raise VompIntegrationError("VoMP target mesh has invalid indices")
        repeated_indices = (
            (triangle_faces[:, 0] == triangle_faces[:, 1])
            | (triangle_faces[:, 1] == triangle_faces[:, 2])
            | (triangle_faces[:, 2] == triangle_faces[:, 0])
        )
        if np.any(repeated_indices):
            raise VompIntegrationError(
                "VoMP target mesh has a face with repeated vertex indices"
            )

        triangle_points = points[triangle_faces]
        linear_scale = np.max(np.ptp(triangle_points, axis=1), axis=1)
        if np.any(~np.isfinite(linear_scale)) or np.any(linear_scale <= 0.0):
            raise VompIntegrationError("VoMP target mesh has a degenerate face")
        length_tolerance = np.maximum(
            linear_scale * 1e-9,
            np.spacing(linear_scale) * 32.0,
        )
        edges = np.stack(
            (
                triangle_points[:, 1] - triangle_points[:, 0],
                triangle_points[:, 2] - triangle_points[:, 1],
                triangle_points[:, 0] - triangle_points[:, 2],
            ),
            axis=1,
        )
        if np.any(np.linalg.norm(edges, axis=2) <= length_tolerance[:, np.newaxis]):
            raise VompIntegrationError(
                "VoMP target mesh has a face with repeated vertex positions"
            )
        normal_lengths = np.linalg.norm(
            np.cross(edges[:, 0], -edges[:, 2]),
            axis=1,
        )
        area_tolerance = np.maximum(
            linear_scale * linear_scale * 1e-12,
            np.spacing(linear_scale) * linear_scale * 32.0,
        )
        if np.any(~np.isfinite(normal_lengths)) or np.any(
            normal_lengths <= area_tolerance
        ):
            raise VompIntegrationError("VoMP target mesh has a zero-area face")
        return np.asarray(triangle_faces, dtype=np.int64).copy()

    triangles: list[tuple[int, int, int]] = []
    cursor = 0
    for face_number, count_value in enumerate(face_counts):
        count = int(count_value)
        if count < 3:
            raise VompIntegrationError(
                "VoMP target contains a face with fewer than 3 vertices"
            )
        face = face_indices[cursor : cursor + count]
        if len(face) != count:
            raise VompIntegrationError("VoMP target mesh has malformed face indices")
        cursor += count
        if face_number in hole_indices:
            continue
        if np.any(face < 0) or np.any(face >= len(points)):
            raise VompIntegrationError("VoMP target mesh has invalid indices")
        if len({int(index) for index in face}) != count:
            raise VompIntegrationError(
                "VoMP target mesh has a face with repeated vertex indices"
            )

        face_points = points[face]
        local_points = face_points - face_points[0]
        linear_scale = float(np.max(np.ptp(face_points, axis=0)))
        if not math.isfinite(linear_scale) or linear_scale <= 0.0:
            raise VompIntegrationError("VoMP target mesh has a degenerate face")
        length_tolerance = max(
            linear_scale * 1e-9,
            math.ulp(linear_scale) * 32.0,
        )
        area_tolerance = max(
            linear_scale * linear_scale * 1e-12,
            math.ulp(linear_scale) * linear_scale * 32.0,
        )
        pairwise_distances = np.linalg.norm(
            face_points[:, np.newaxis, :] - face_points[np.newaxis, :, :],
            axis=2,
        )
        distinct_distances = pairwise_distances[np.triu_indices(count, k=1)]
        if np.any(distinct_distances <= length_tolerance):
            raise VompIntegrationError(
                "VoMP target mesh has a face with repeated vertex positions"
            )

        normal = np.sum(
            np.cross(local_points, np.roll(local_points, -1, axis=0)),
            axis=0,
        )
        normal_length = float(np.linalg.norm(normal))
        if not math.isfinite(normal_length) or normal_length <= area_tolerance:
            raise VompIntegrationError("VoMP target mesh has a zero-area face")
        unit_normal = normal / normal_length
        # UsdGeom.Mesh stores points as point3f[]; allow its coordinate-scale
        # quantization without weakening the topology tolerances above.
        coordinate_magnitude = float(np.max(np.abs(face_points)))
        planarity_tolerance = max(
            length_tolerance,
            float(np.finfo(np.float32).eps) * 8.0 * coordinate_magnitude,
        )
        if np.any(np.abs(local_points @ unit_normal) > planarity_tolerance):
            raise VompIntegrationError(
                "VoMP target mesh has a non-planar polygon; tessellate it first"
            )

        drop_axis = int(np.argmax(np.abs(unit_normal)))
        projected = np.delete(local_points, drop_axis, axis=1)

        for first_edge in range(count):
            first_next = (first_edge + 1) % count
            for second_edge in range(first_edge + 1, count):
                second_next = (second_edge + 1) % count
                if first_next == second_edge or second_next == first_edge:
                    continue
                first_start = projected[first_edge]
                first_end = projected[first_next]
                second_start = projected[second_edge]
                second_end = projected[second_next]
                boxes_overlap = bool(
                    np.all(
                        np.maximum(
                            np.minimum(first_start, first_end),
                            np.minimum(second_start, second_end),
                        )
                        <= np.minimum(
                            np.maximum(first_start, first_end),
                            np.maximum(second_start, second_end),
                        )
                        + length_tolerance
                    )
                )
                if boxes_overlap and (
                    _cross_2d(projected, first_edge, first_next, second_edge)
                    * _cross_2d(projected, first_edge, first_next, second_next)
                    <= area_tolerance * area_tolerance
                    and _cross_2d(projected, second_edge, second_next, first_edge)
                    * _cross_2d(projected, second_edge, second_next, first_next)
                    <= area_tolerance * area_tolerance
                ):
                    raise VompIntegrationError(
                        "VoMP target mesh has a self-intersecting polygon; "
                        "tessellate it first"
                    )

        fan_turns = np.asarray(
            [_cross_2d(projected, 0, index, index + 1) for index in range(1, count - 1)]
        )
        if np.any(np.abs(fan_turns) <= area_tolerance) or not (
            np.all(fan_turns > 0.0) or np.all(fan_turns < 0.0)
        ):
            raise VompIntegrationError(
                "VoMP target mesh has a concave polygon with an unsafe fan; "
                "tessellate it first"
            )
        first = int(face[0])
        triangles.extend(
            (first, int(face[index]), int(face[index + 1]))
            for index in range(1, count - 1)
        )
    if cursor != len(face_indices):
        raise VompIntegrationError("VoMP target mesh has trailing face indices")
    return np.asarray(triangles, dtype=np.int64).reshape((-1, 3))


def _target_world_mesh(target: Any, meters_per_unit: float) -> _WorldMesh:
    from pxr import Usd, UsdGeom

    render_time = Usd.TimeCode(0)
    xform_cache = UsdGeom.XformCache(render_time)
    vertex_chunks: list[np.ndarray] = []
    triangle_chunks: list[np.ndarray] = []
    seen_geometry: set[str] = set()
    source_mesh_count = 0
    vertex_offset = 0
    predicate = Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)
    for prim in Usd.PrimRange(target, predicate):
        if not prim.IsA(UsdGeom.Boundable):
            continue
        imageable = UsdGeom.Imageable(prim)
        if imageable.ComputeVisibility(render_time) == UsdGeom.Tokens.invisible:
            continue
        purpose = imageable.ComputePurpose()
        if purpose not in {UsdGeom.Tokens.default_, UsdGeom.Tokens.render}:
            continue
        if not prim.IsA(UsdGeom.Mesh):
            raise VompIntegrationError(
                "VoMP target contains visible unsupported renderable geometry at "
                f"{prim.GetPath()}; tessellate it to UsdGeom.Mesh or exclude it "
                "from rendering"
            )
        mesh = UsdGeom.Mesh(prim)
        _validate_static_mesh_geometry(mesh)
        subdivision_scheme = mesh.GetSubdivisionSchemeAttr().Get(render_time)
        if subdivision_scheme != UsdGeom.Tokens.none:
            raise VompIntegrationError(
                f"VoMP target mesh {prim.GetPath()} uses subdivision scheme "
                f"{subdivision_scheme or 'unknown'}; author subdivisionScheme=none "
                "or tessellate the evaluated surface first"
            )
        _validate_static_transform_chain(prim, UsdGeom)
        source_mesh_count += 1
        raw_points = np.asarray(mesh.GetPointsAttr().Get(render_time), dtype=np.float64)
        face_counts = np.asarray(
            mesh.GetFaceVertexCountsAttr().Get(render_time), dtype=np.int64
        )
        face_indices = np.asarray(
            mesh.GetFaceVertexIndicesAttr().Get(render_time), dtype=np.int64
        )
        if raw_points.ndim != 2 or raw_points.shape[1:] != (3,) or not len(raw_points):
            raise VompIntegrationError(
                f"VoMP target mesh {prim.GetPath()} has no valid points"
            )
        if not np.all(np.isfinite(raw_points)):
            raise VompIntegrationError(
                f"VoMP target mesh {prim.GetPath()} has non-finite points"
            )
        holes = {
            int(value) for value in (mesh.GetHoleIndicesAttr().Get(render_time) or [])
        }
        triangles = _triangulate_faces(raw_points, face_counts, face_indices, holes)
        if not len(triangles):
            continue

        matrix = xform_cache.GetLocalToWorldTransform(prim)
        transformed = _transform_points(matrix, raw_points) * meters_per_unit
        linear = _matrix4_array(matrix)[:3, :3].T
        left_handed = (
            mesh.GetOrientationAttr().Get(render_time) == UsdGeom.Tokens.leftHanded
        )
        reflected = float(np.linalg.det(linear)) < 0.0
        if left_handed != reflected:
            triangles = triangles[:, [0, 2, 1]]

        digest = hashlib.sha256()
        digest.update(np.ascontiguousarray(transformed).tobytes())
        digest.update(np.ascontiguousarray(triangles).tobytes())
        geometry_digest = digest.hexdigest()
        if geometry_digest in seen_geometry:
            continue
        seen_geometry.add(geometry_digest)
        vertex_chunks.append(transformed)
        triangle_chunks.append(triangles + vertex_offset)
        vertex_offset += len(transformed)

    if not vertex_chunks or not triangle_chunks:
        raise VompIntegrationError(
            f"VoMP target prim {target.GetPath()} has no visible render meshes"
        )
    vertices = np.concatenate(vertex_chunks, axis=0)
    triangles = np.concatenate(triangle_chunks, axis=0)
    minimum = np.min(vertices, axis=0)
    maximum = np.max(vertices, axis=0)
    center = (minimum + maximum) / 2.0
    scale = float(np.max(maximum - minimum))
    if not math.isfinite(scale) or scale <= 0.0:
        raise VompIntegrationError("VoMP target world mesh has a degenerate bound")
    return _WorldMesh(
        vertices_m=vertices,
        triangles=triangles,
        center_m=center,
        scale_m=scale,
        source_mesh_count=source_mesh_count,
        unique_mesh_count=len(vertex_chunks),
    )


def _write_ascii_ply(path: Path, mesh: _WorldMesh) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="ascii",
        prefix=f".{path.stem}_",
        suffix=path.suffix,
        dir=path.parent,
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write("ply\nformat ascii 1.0\n")
        stream.write(f"element vertex {len(mesh.vertices_m)}\n")
        stream.write("property double x\nproperty double y\nproperty double z\n")
        stream.write(f"element face {len(mesh.triangles)}\n")
        stream.write("property list uchar int vertex_indices\nend_header\n")
        np.savetxt(stream, mesh.vertices_m, fmt="%.17g %.17g %.17g")
        face_rows = np.empty((len(mesh.triangles), 4), dtype=np.int64)
        face_rows[:, 0] = 3
        face_rows[:, 1:] = mesh.triangles
        np.savetxt(stream, face_rows, fmt="%d %d %d %d")
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _add_camera(
    stage: Any,
    view: dict[str, float | int],
    *,
    center_m: np.ndarray,
    scale_m: float,
    meters_per_unit: float,
) -> str:
    from pxr import Gf, UsdGeom

    index = int(view["index"])
    yaw = float(view["yaw"])
    pitch = float(view["pitch"])
    radius = float(view["radius"])
    normalized_eye = np.asarray(
        [
            radius * math.cos(yaw) * math.cos(pitch),
            radius * math.sin(yaw) * math.cos(pitch),
            radius * math.sin(pitch),
        ],
        dtype=np.float64,
    )
    eye_m = center_m + normalized_eye * scale_m
    eye = Gf.Vec3d(*(eye_m / meters_per_unit))
    target = Gf.Vec3d(*(center_m / meters_per_unit))
    forward = (eye - target).GetNormalized()
    up_hint = Gf.Vec3d(0.0, 0.0, 1.0)
    if abs(Gf.Dot(forward, up_hint)) > 0.999:
        up_hint = Gf.Vec3d(0.0, 1.0, 0.0)
    right = Gf.Cross(up_hint, forward).GetNormalized()
    up = Gf.Cross(forward, right).GetNormalized()
    rotation = Gf.Matrix4d(
        right[0],
        right[1],
        right[2],
        0.0,
        up[0],
        up[1],
        up[2],
        0.0,
        forward[0],
        forward[1],
        forward[2],
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )
    transform = rotation * Gf.Matrix4d().SetTranslate(eye)
    camera_path = f"/__PhysicsAgentVompCameras/Camera_{index:03d}"
    camera = UsdGeom.Camera.Define(stage, camera_path)
    camera.CreateProjectionAttr(UsdGeom.Tokens.perspective)
    camera.CreateHorizontalApertureAttr(32.0)
    camera.CreateVerticalApertureAttr(32.0)
    camera.CreateFocalLengthAttr(
        16.0 / math.tan(math.radians(float(view["fovDegrees"])) / 2.0)
    )
    camera.CreateClippingRangeAttr(
        Gf.Vec2f(
            max(scale_m * 0.01 / meters_per_unit, 1e-5),
            scale_m * (radius + 4.0) / meters_per_unit,
        )
    )
    UsdGeom.Xformable(camera).AddTransformOp().Set(transform)
    return camera_path


def _render_digest(metadata_path: Path, image_paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in (metadata_path, *image_paths):
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _ovrtx_render_metadata(render_result: dict[str, Any]) -> dict[str, Any]:
    raw_results = render_result.get("results", [])
    if not isinstance(raw_results, list):
        raise VompIntegrationError("OVRTX returned invalid VoMP render metadata")

    blank_frames = render_result.get("blank_render_frames", [])
    if not isinstance(blank_frames, list):
        raise VompIntegrationError("OVRTX returned invalid blank-frame metadata")
    blank_frames = list(blank_frames)
    cameras: list[dict[str, Any]] = []
    for item in raw_results:
        if not isinstance(item, dict):
            raise VompIntegrationError("OVRTX returned invalid VoMP camera metadata")
        camera_blanks = item.get("blank_render_frames", [])
        if not isinstance(camera_blanks, list):
            raise VompIntegrationError("OVRTX returned invalid blank-frame metadata")
        images = item.get("images", [])
        if not isinstance(images, list):
            raise VompIntegrationError("OVRTX returned invalid VoMP camera metadata")
        for blank_frame in camera_blanks:
            if blank_frame not in blank_frames:
                blank_frames.append(blank_frame)
        cameras.append(
            {
                "camera": item.get("camera"),
                "status": item.get("status"),
                "error": item.get("error"),
                "renderTimeSeconds": item.get("render_time"),
                "frameCount": item.get("frame_count"),
                "imageFrames": item.get("image_frames"),
                "imageCount": len(images),
                "warnings": item.get("warnings", []),
                "blankRenderFrames": camera_blanks,
            }
        )
    metadata = {
        "schemaVersion": 1,
        "totalCameras": render_result.get("total_cameras"),
        "successfulCameras": render_result.get("successful_cameras"),
        "failedCameras": render_result.get("failed_cameras"),
        "totalRenderTimeSeconds": render_result.get("total_render_time"),
        "warnings": render_result.get("warnings", []),
        "blankRenderFrames": blank_frames,
        "cameras": cameras,
    }
    try:
        encoded = json.dumps(metadata, allow_nan=False)
        normalized = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise VompIntegrationError(
            "OVRTX returned invalid VoMP render metadata"
        ) from exc
    return cast(dict[str, Any], normalized)


def _validate_mesh_transform_association(
    render_manifest: dict[str, Any],
    inference_manifest: dict[str, Any],
) -> None:
    geometry = render_manifest.get("geometry")
    transform = inference_manifest.get("meshTransform")
    if not isinstance(geometry, dict) or not isinstance(transform, dict):
        raise VompIntegrationError("VoMP worker omitted its mesh normalization frame")
    expected_center_raw = geometry.get("centerM")
    returned_center_raw = transform.get("centerM")
    expected_scale_raw = geometry.get("scaleM")
    returned_scale_raw = transform.get("scaleM")
    centers = (expected_center_raw, returned_center_raw)
    scales = (expected_scale_raw, returned_scale_raw)
    if any(
        not isinstance(center, list)
        or len(center) != 3
        or any(
            isinstance(value, bool) or not isinstance(value, int | float)
            for value in center
        )
        for center in centers
    ) or any(
        isinstance(value, bool) or not isinstance(value, int | float)
        for value in scales
    ):
        raise VompIntegrationError(
            "VoMP worker returned an invalid mesh normalization frame"
        )
    try:
        expected_center = np.asarray(expected_center_raw, dtype=np.float64)
        returned_center = np.asarray(returned_center_raw, dtype=np.float64)
        expected_scale = float(cast(int | float, expected_scale_raw))
        returned_scale = float(cast(int | float, returned_scale_raw))
    except (TypeError, ValueError, OverflowError) as exc:
        raise VompIntegrationError(
            "VoMP worker returned an invalid mesh normalization frame"
        ) from exc
    if (
        expected_center.shape != (3,)
        or returned_center.shape != (3,)
        or not np.all(np.isfinite(expected_center))
        or not np.all(np.isfinite(returned_center))
        or not math.isfinite(expected_scale)
        or not math.isfinite(returned_scale)
        or expected_scale <= 0.0
        or returned_scale <= 0.0
    ):
        raise VompIntegrationError(
            "VoMP worker returned an invalid mesh normalization frame"
        )
    tolerance = max(1e-12, expected_scale * 1e-9)
    if not np.allclose(
        returned_center,
        expected_center,
        rtol=1e-9,
        atol=tolerance,
    ) or not math.isclose(
        returned_scale,
        expected_scale,
        rel_tol=1e-9,
        abs_tol=tolerance,
    ):
        raise VompIntegrationError(
            "VoMP mesh normalization does not match the OVRTX evidence frame"
        )


def _prepare_artifact_directory(work_root: Path) -> Path:
    artifact_dir = work_root / "evidence"
    if artifact_dir.is_symlink() or artifact_dir.is_file():
        artifact_dir.unlink()
    elif artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True)
    return artifact_dir


def _preflight_vomp_deformable_evidence_target(stage: Any, target: Any) -> float:
    """Reject targets the Newton-first deformable adapter cannot publish."""

    meters_per_unit, _, _ = _preflight_vomp_deformable_target(stage, target)
    return float(meters_per_unit)


def prepare_vomp_evidence(
    usd_path: str | Path,
    *,
    target_prim_path: str,
    artifact_dir: str | Path,
    render_config: VompRenderConfig | None = None,
    target_contract: Literal["rigid", "volume-deformable"] = "rigid",
) -> VompPreparedEvidence:
    """Render calibrated OVRTX views and export target geometry in world meters."""

    from pxr import Sdf, Usd, UsdGeom, UsdLux
    from world_understanding.functions.graphics.rendering_backend_factory import (
        create_rendering_backend,
    )
    from world_understanding.functions.graphics.usd_camera import (
        extract_camera_parameters_from_stage,
    )

    config = render_config or VompRenderConfig()
    _validate_render_config(config)
    source = Path(usd_path).expanduser().resolve()
    output_dir = Path(artifact_dir).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError("VoMP render input USD was not found")
    _validate_vomp_input_layer(source)
    output_dir.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.Open(str(source))
    if not stage:
        raise VompIntegrationError("Unable to open the VoMP render input USD")
    composition_hashes = _reload_and_hash_composition_layers(stage)
    try:
        target_path = Sdf.Path(target_prim_path)
    except Exception as exc:
        raise VompIntegrationError(
            "VoMP target_prim_path is not a valid USD path"
        ) from exc
    if (
        not target_path.IsAbsolutePath()
        or not target_path.IsPrimPath()
        or target_path == Sdf.Path.absoluteRootPath
    ):
        raise VompIntegrationError(
            "VoMP target_prim_path must be an absolute prim path"
        )
    target = stage.GetPrimAtPath(target_path)
    if not target or not target.IsValid():
        raise VompIntegrationError("VoMP target prim was not found")
    if target_contract == "rigid":
        meters_per_unit, _, _, _ = _preflight_vomp_mass_target(stage, target)
    elif target_contract == "volume-deformable":
        meters_per_unit = _preflight_vomp_deformable_evidence_target(stage, target)
    else:
        raise VompIntegrationError(
            "VoMP evidence target_contract must be rigid or volume-deformable"
        )

    mesh = _target_world_mesh(target, meters_per_unit)
    mesh_path = output_dir / "target_world_m.ply"
    _write_ascii_ply(mesh_path, mesh)

    session_layer = Sdf.Layer.CreateAnonymous("physics_agent_vomp_render_session.usda")
    render_stage = Usd.Stage.Open(stage.GetRootLayer(), session_layer)
    if not render_stage:
        raise VompIntegrationError("Unable to create the VoMP render stage")
    render_stage.SetEditTarget(session_layer)
    if render_stage.GetPrimAtPath("/__PhysicsAgentVompCameras"):
        raise VompIntegrationError(
            "VoMP render input already defines /__PhysicsAgentVompCameras, "
            "which Physics Agent reserves for its calibrated cameras"
        )
    render_target = render_stage.GetPrimAtPath(target_path)
    predicate = Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)
    for prim in Usd.PrimRange.Stage(render_stage, predicate):
        if prim.GetPath().HasPrefix(target_path):
            continue
        if prim.IsA(UsdGeom.Boundable) and not prim.HasAPI(UsdLux.LightAPI):
            visibility_prim = prim
            while visibility_prim.IsInstanceProxy():
                visibility_prim = visibility_prim.GetParent()
            UsdGeom.Imageable(visibility_prim).CreateVisibilityAttr(
                UsdGeom.Tokens.invisible
            )
    UsdGeom.Scope.Define(render_stage, "/__PhysicsAgentVompCameras")
    views = _sample_views(config)
    camera_paths = [
        _add_camera(
            render_stage,
            view,
            center_m=mesh.center_m,
            scale_m=mesh.scale_m,
            meters_per_unit=meters_per_unit,
        )
        for view in views
    ]
    if UsdGeom.Imageable(render_target).ComputeVisibility() == UsdGeom.Tokens.invisible:
        raise VompIntegrationError(
            "VoMP target prim is not visible for OVRTX rendering"
        )

    render_stage_path = output_dir / "ovrtx_input.usda"
    flattened = render_stage.Flatten()
    if not flattened.Export(str(render_stage_path)):
        raise VompIntegrationError("Unable to export the prepared VoMP render stage")
    exported_render_stage = Usd.Stage.Open(str(render_stage_path))
    if not exported_render_stage:
        raise VompIntegrationError("Unable to reopen the prepared VoMP render stage")

    backend = create_rendering_backend(
        "ovrtx",
        {
            "log_level": "warn",
            "ovrtx_venv_dir": config.ovrtx_venv_dir,
            "num_sensor_updates": config.num_sensor_updates,
            "render_mode": config.render_mode,
            "material_target": config.material_target,
        },
    )
    try:
        render_result = backend.render(
            stage=exported_render_stage,
            image_width=config.image_width,
            image_height=config.image_height,
            cameras=camera_paths,
            frames="0",
            base_dir=str(source.parent),
        )
    except Exception as exc:
        raise VompIntegrationError(
            "OVRTX failed while rendering VoMP evidence"
        ) from exc
    renderer_metadata = _ovrtx_render_metadata(render_result)
    renderer_metadata_path = output_dir / "ovrtx_render_result.json"
    _write_json_atomic(renderer_metadata_path, renderer_metadata)
    if renderer_metadata["blankRenderFrames"]:
        raise VompIntegrationError(
            "OVRTX marked one or more VoMP evidence frames as blank; "
            "renderer metadata was preserved in ovrtx_render_result.json"
        )
    if (
        int(render_result.get("successful_cameras", 0)) != len(camera_paths)
        or int(render_result.get("failed_cameras", 0)) != 0
    ):
        raise VompIntegrationError("OVRTX did not render every requested VoMP camera")

    render_dir = output_dir / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)
    results_by_camera = {
        str(item.get("camera")): item for item in render_result.get("results", [])
    }
    frames: list[dict[str, Any]] = []
    image_paths: list[Path] = []
    camera_xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    for view, camera_path in zip(views, camera_paths, strict=True):
        camera_result = results_by_camera.get(camera_path)
        images = camera_result.get("images", []) if camera_result else []
        if len(images) != 1:
            raise VompIntegrationError("OVRTX returned an invalid VoMP camera result")
        image_path = render_dir / f"{int(view['index']):03d}.png"
        with tempfile.NamedTemporaryFile(
            prefix=f".{image_path.stem}_",
            suffix=image_path.suffix,
            dir=render_dir,
            delete=False,
        ) as stream:
            temporary_image = Path(stream.name)
        try:
            images[0].save(temporary_image)
            temporary_image.replace(image_path)
        finally:
            temporary_image.unlink(missing_ok=True)
        parameters = extract_camera_parameters_from_stage(
            stage=exported_render_stage,
            camera_path=camera_path,
            image_width=config.image_width,
            image_height=config.image_height,
            xform_cache=camera_xform_cache,
        )
        camera_to_world = np.asarray(
            parameters["camera_world_transform"], dtype=np.float64
        ).T
        normalized = camera_to_world.copy()
        normalized[:3, 3] = (
            camera_to_world[:3, 3] * meters_per_unit - mesh.center_m
        ) / mesh.scale_m
        expected = np.asarray(
            [
                float(view["radius"])
                * math.cos(float(view["yaw"]))
                * math.cos(float(view["pitch"])),
                float(view["radius"])
                * math.sin(float(view["yaw"]))
                * math.cos(float(view["pitch"])),
                float(view["radius"]) * math.sin(float(view["pitch"])),
            ]
        )
        if not np.allclose(normalized[:3, 3], expected, rtol=0.0, atol=1e-7):
            raise VompIntegrationError(
                "Prepared OVRTX camera calibration is inconsistent"
            )
        frames.append(
            {
                "file_path": image_path.name,
                "transform_matrix": normalized.tolist(),
                "camera_angle_x": float(parameters["fov_x_rad"]),
            }
        )
        image_paths.append(image_path)

    metadata_path = output_dir / "renders_metadata.json"
    _write_json_atomic(metadata_path, frames)
    for layer_path, expected_hash in composition_hashes.items():
        if not layer_path.is_file() or _sha256(layer_path) != expected_hash:
            raise VompIntegrationError(
                "input USD composition changed during VoMP evidence rendering"
            )
    manifest: dict[str, Any] = {
        "schemaVersion": 1,
        "backend": "world_understanding.ovrtx",
        "sourceUsdSha256": composition_hashes.get(source) or _sha256(source),
        "sourceCompositionSha256": sorted(set(composition_hashes.values())),
        "renderStageSha256": _sha256(render_stage_path),
        "rendererMetadataSha256": _sha256(renderer_metadata_path),
        "rendererMetadata": renderer_metadata,
        "meshSha256": _sha256(mesh_path),
        "renderDigestSha256": _render_digest(metadata_path, image_paths),
        "targetPrimPath": str(target_path),
        "geometry": {
            "sourceMeshCount": mesh.source_mesh_count,
            "uniqueMeshCount": mesh.unique_mesh_count,
            "vertexCount": int(len(mesh.vertices_m)),
            "triangleCount": int(len(mesh.triangles)),
            "centerM": mesh.center_m.tolist(),
            "scaleM": mesh.scale_m,
        },
        "cameraSampling": {
            "algorithm": "vomp_hammersley_v1",
            "seed": config.seed,
            "numViews": config.num_views,
            "radius": config.radius,
            "fovDegrees": config.fov_degrees,
        },
        "render": {
            "imageWidth": config.image_width,
            "imageHeight": config.image_height,
            "renderMode": config.render_mode,
            "numSensorUpdates": config.num_sensor_updates,
            "materialTarget": config.material_target,
            "successfulCameras": len(image_paths),
        },
    }
    manifest_path = output_dir / "ovrtx_manifest.json"
    _write_json_atomic(manifest_path, manifest)
    return VompPreparedEvidence(
        artifact_dir=output_dir,
        mesh_path=mesh_path,
        metadata_path=metadata_path,
        render_stage_path=render_stage_path,
        manifest_path=manifest_path,
        manifest=manifest,
        composition_hashes=dict(composition_hashes),
    )


def _inference_provenance(manifest: dict[str, Any]) -> dict[str, Any]:
    runtime = cast(dict[str, Any], manifest.get("runtime") or {})
    artifacts = cast(dict[str, Any], runtime.get("artifacts") or {})
    safe_artifacts = {
        key: {
            "sha256": value.get("sha256"),
            "sizeBytes": value.get("sizeBytes"),
        }
        for key, value in artifacts.items()
        if isinstance(value, dict)
    }
    return {
        "protocolVersion": manifest.get("protocolVersion"),
        "outputNpzSha256": manifest.get("outputNpzSha256"),
        "sampleCount": manifest.get("sampleCount"),
        "completeVoxelField": manifest.get("completeVoxelField"),
        "voxelSizeNormalized": manifest.get("voxelSizeNormalized"),
        "meshTransform": manifest.get("meshTransform"),
        "deduplication": manifest.get("deduplication"),
        "runtime": {
            "vompRevision": runtime.get("vompRevision"),
            "pythonVersion": runtime.get("pythonVersion"),
            "torchVersion": runtime.get("torchVersion"),
            "attentionBackend": runtime.get("attentionBackend"),
            "cudaAvailable": runtime.get("cudaAvailable"),
            "cudaVersion": runtime.get("cudaVersion"),
            "artifacts": safe_artifacts,
        },
    }


def run_vomp_mass_pipeline(
    usd_path: str | Path,
    output_usd_path: str | Path,
    *,
    target_prim_path: str,
    work_dir: str | Path,
    runtime_config: VompRuntimeConfig | None = None,
    runner: VompRunner | None = None,
    render_config: VompRenderConfig | None = None,
    provenance_path: str | Path | None = None,
) -> VompPipelineResult:
    """Run OVRTX evidence capture, official VoMP, and strict mass authoring."""

    if runner is None:
        if runtime_config is None:
            raise VompIntegrationError("run_vomp_mass_pipeline requires a VoMP runtime")
        _validate_runtime(runtime_config)
        runner = ExternalVompRunner(runtime_config)
    config = render_config or VompRenderConfig()
    _validate_render_config(config)
    work_root = Path(work_dir).expanduser().resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    artifact_dir = _prepare_artifact_directory(work_root)
    evidence = prepare_vomp_evidence(
        usd_path,
        target_prim_path=target_prim_path,
        artifact_dir=artifact_dir,
        render_config=config,
    )
    npz_path = artifact_dir / "vomp_voxel_materials.npz"
    run_result = runner.run(
        VompRunRequest(
            mesh_path=evidence.mesh_path,
            metadata_path=evidence.metadata_path,
            output_dir=evidence.artifact_dir,
            output_npz_path=npz_path,
            num_views=config.num_views,
            seed=config.seed,
        )
    )
    _verify_composition_hashes(evidence.composition_hashes)
    worker_npz_sha256 = run_result.manifest.get("outputNpzSha256")
    if not isinstance(worker_npz_sha256, str):
        raise VompIntegrationError("VoMP worker omitted its NPZ SHA-256 digest")
    worker_npz_sha256 = _require_sha256(
        worker_npz_sha256,
        label="VoMP worker outputNpzSha256",
    )
    if (
        not run_result.output_npz_path.is_file()
        or _sha256(run_result.output_npz_path) != worker_npz_sha256
    ):
        raise VompIntegrationError(
            "VoMP worker NPZ failed pipeline digest verification"
        )
    worker_manifest_path = artifact_dir / "vomp_inference_manifest.json"
    _write_json_atomic(worker_manifest_path, run_result.manifest)
    _validate_mesh_transform_association(evidence.manifest, run_result.manifest)
    apply_result = apply_vomp_mass_properties(
        usd_path,
        run_result.output_npz_path,
        output_usd_path,
        target_prim_path=target_prim_path,
        voxel_size_m=run_result.voxel_size_m,
        coordinate_unit_meters=run_result.coordinate_unit_meters,
        complete_voxel_field=True,
        coordinate_offset_m=run_result.coordinate_offset_m,
        provenance_path=provenance_path,
        evidence_provenance={
            "rendering": evidence.manifest,
            "inference": _inference_provenance(run_result.manifest),
        },
        expected_npz_sha256=worker_npz_sha256,
        expected_composition_hashes=evidence.composition_hashes,
    )
    return VompPipelineResult(
        apply_result=apply_result,
        evidence=evidence,
        vomp_npz_path=run_result.output_npz_path,
        worker_manifest_path=worker_manifest_path,
        worker_log_path=run_result.worker_log_path,
    )


def run_vomp_volume_deformable_pipeline(
    usd_path: str | Path,
    output_usd_path: str | Path,
    *,
    target_prim_path: str,
    work_dir: str | Path,
    runtime_config: VompRuntimeConfig | None = None,
    runner: VompRunner | None = None,
    render_config: VompRenderConfig | None = None,
    material_reduction: MaterialReductionPolicy = "reject",
    max_deformable_voxels: int = DEFAULT_MAX_DEFORMABLE_VOXELS,
    provenance_path: str | Path | None = None,
) -> VompDeformablePipelineResult:
    """Run OVRTX, official VoMP, and Newton-first deformable authoring."""

    deformable_voxel_limit = _positive_integer(
        max_deformable_voxels,
        label="max_deformable_voxels",
    )
    if runner is None:
        if runtime_config is None:
            raise VompIntegrationError(
                "run_vomp_volume_deformable_pipeline requires a VoMP runtime"
            )
        if runtime_config.max_complete_voxels > deformable_voxel_limit:
            runtime_config = replace(
                runtime_config,
                max_complete_voxels=deformable_voxel_limit,
            )
        _validate_runtime(runtime_config)
        runner = ExternalVompRunner(runtime_config)
    config = render_config or VompRenderConfig()
    _validate_render_config(config)
    if material_reduction not in {"reject", "homogeneous-volume-average"}:
        raise VompIntegrationError(
            "material_reduction must be 'reject' or 'homogeneous-volume-average'"
        )
    work_root = Path(work_dir).expanduser().resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    artifact_dir = _prepare_artifact_directory(work_root)
    evidence = prepare_vomp_evidence(
        usd_path,
        target_prim_path=target_prim_path,
        artifact_dir=artifact_dir,
        render_config=config,
        target_contract="volume-deformable",
    )
    npz_path = artifact_dir / "vomp_voxel_materials.npz"
    run_result = runner.run(
        VompRunRequest(
            mesh_path=evidence.mesh_path,
            metadata_path=evidence.metadata_path,
            output_dir=evidence.artifact_dir,
            output_npz_path=npz_path,
            num_views=config.num_views,
            seed=config.seed,
        )
    )
    _verify_composition_hashes(evidence.composition_hashes)
    worker_npz_sha256 = run_result.manifest.get("outputNpzSha256")
    if not isinstance(worker_npz_sha256, str):
        raise VompIntegrationError("VoMP worker omitted its NPZ SHA-256 digest")
    worker_npz_sha256 = _require_sha256(
        worker_npz_sha256,
        label="VoMP worker outputNpzSha256",
    )
    if (
        not run_result.output_npz_path.is_file()
        or _sha256(run_result.output_npz_path) != worker_npz_sha256
    ):
        raise VompIntegrationError(
            "VoMP worker NPZ failed pipeline digest verification"
        )
    worker_manifest_path = artifact_dir / "vomp_inference_manifest.json"
    _write_json_atomic(worker_manifest_path, run_result.manifest)
    _validate_mesh_transform_association(evidence.manifest, run_result.manifest)
    apply_result = apply_vomp_volume_deformable(
        usd_path,
        run_result.output_npz_path,
        output_usd_path,
        target_prim_path=target_prim_path,
        voxel_size_m=run_result.voxel_size_m,
        coordinate_unit_meters=run_result.coordinate_unit_meters,
        complete_voxel_field=True,
        coordinate_offset_m=run_result.coordinate_offset_m,
        material_reduction=material_reduction,
        max_deformable_voxels=max_deformable_voxels,
        provenance_path=provenance_path,
        evidence_provenance={
            "rendering": evidence.manifest,
            "inference": _inference_provenance(run_result.manifest),
        },
        expected_npz_sha256=worker_npz_sha256,
        expected_composition_hashes=evidence.composition_hashes,
    )
    return VompDeformablePipelineResult(
        apply_result=apply_result,
        evidence=evidence,
        vomp_npz_path=run_result.output_npz_path,
        worker_manifest_path=worker_manifest_path,
        worker_log_path=run_result.worker_log_path,
    )


__all__ = [
    "VompDeformablePipelineResult",
    "VompPipelineResult",
    "VompPreparedEvidence",
    "VompRenderConfig",
    "prepare_vomp_evidence",
    "run_vomp_mass_pipeline",
    "run_vomp_volume_deformable_pipeline",
]
