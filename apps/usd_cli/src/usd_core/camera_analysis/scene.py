# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""USD traversal and canonical scene adaptation for camera analysis."""

from __future__ import annotations

import hashlib
import math
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass, replace

import numpy as np

from usd_core.camera_analysis.cancellation import check_cancelled
from usd_core.camera_analysis.contracts import (
    MAX_ANALYSIS_GEOMETRY_BYTES,
    MAX_ANALYSIS_SHAPES,
    MAX_ANALYSIS_TRIANGLES,
    MAX_ANALYSIS_VERTICES,
    AnalysisRole,
    CameraIR,
    CameraPose,
    Matrix4,
    MeshResource,
    SceneAnalysisIR,
    SceneAnalysisPolicy,
    ShapeIR,
)

_PURPOSES = frozenset({"default", "render"})
_MAX_RAW_FACE_INDICES = MAX_ANALYSIS_TRIANGLES * 3
_MESH_BOUNDS_CHUNK_VERTICES = 65_536
# Exact occurrence bounds scan each shared resource once per distinct linear
# transform. Keep aggregate scan work no larger than one fully populated legal
# mesh resource, even when a stage contains many differently transformed instances.
MAX_ANALYSIS_BOUNDS_VERTEX_VISITS = MAX_ANALYSIS_VERTICES


@dataclass(frozen=True)
class _RawMeshData:
    points: object | None
    face_counts: object | None
    face_indices: object | None
    holes: object


@dataclass
class _RawSceneBudget:
    shapes: int = 0
    points: int = 0
    face_indices: int = 0
    estimated_triangles: int = 0
    estimated_geometry_bytes: int = 0

    def ensure_shape_capacity(self, *, prim_path: str) -> None:
        if self.shapes >= MAX_ANALYSIS_SHAPES:
            raise ValueError(
                f"analysis scene has more than {MAX_ANALYSIS_SHAPES:,} supported "
                f"shapes at {prim_path}"
            )

    def account_shape(self, *, prim_path: str) -> None:
        self.ensure_shape_capacity(prim_path=prim_path)
        self.shapes += 1

    def account_mesh(self, data: _RawMeshData, *, prim_path: str) -> None:
        try:
            point_count = 0 if data.points is None else len(data.points)
            face_count = 0 if data.face_counts is None else len(data.face_counts)
            index_count = 0 if data.face_indices is None else len(data.face_indices)
        except TypeError as exc:
            raise ValueError(f"mesh {prim_path} has unsized raw topology") from exc

        next_points = self.points + point_count
        if next_points > MAX_ANALYSIS_VERTICES:
            raise ValueError(
                f"analysis scene raw mesh points would total {next_points:,}; limit "
                f"is {MAX_ANALYSIS_VERTICES:,}"
            )
        next_indices = self.face_indices + index_count
        if next_indices > _MAX_RAW_FACE_INDICES:
            raise ValueError(
                f"analysis scene raw mesh face indices would total {next_indices:,}; "
                f"limit is {_MAX_RAW_FACE_INDICES:,}"
            )
        if face_count > MAX_ANALYSIS_TRIANGLES - self.estimated_triangles:
            raise ValueError(
                f"analysis scene raw mesh faces exceed the "
                f"{MAX_ANALYSIS_TRIANGLES:,}-triangle limit"
            )
        try:
            counts = () if data.face_counts is None else data.face_counts
            triangle_count = 0
            for index, count in enumerate(counts):
                if index % 1024 == 0:
                    check_cancelled()
                triangle_count += max(0, int(count) - 2)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"mesh {prim_path} has malformed raw face counts") from exc
        next_triangles = self.estimated_triangles + triangle_count
        if next_triangles > MAX_ANALYSIS_TRIANGLES:
            raise ValueError(
                f"analysis scene raw topology would generate up to "
                f"{next_triangles:,} triangles; limit is {MAX_ANALYSIS_TRIANGLES:,}"
            )
        next_bytes = next_points * 12 + next_triangles * 12
        if next_bytes > MAX_ANALYSIS_GEOMETRY_BYTES:
            raise ValueError(
                f"analysis scene raw topology needs an estimated {next_bytes:,} "
                f"geometry bytes; limit is {MAX_ANALYSIS_GEOMETRY_BYTES:,}"
            )
        self.points = next_points
        self.face_indices = next_indices
        self.estimated_triangles = next_triangles
        self.estimated_geometry_bytes = next_bytes


def _matrix_tuple(value: np.ndarray) -> Matrix4:
    return tuple(tuple(float(item) for item in row) for row in value)


def _np_matrix(value) -> np.ndarray:
    return np.asarray(
        [[float(value[row][column]) for column in range(4)] for row in range(4)],
        dtype=np.float64,
    )


def _axis_map(up_axis: str) -> np.ndarray:
    """Row-vector rotation from a USD axis convention to canonical Z-up."""

    if up_axis.upper() == "Y":
        # (x, y, z) -> (x, -z, y), a proper rotation with +Y mapped to +Z.
        return np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
            dtype=np.float64,
        )
    return np.eye(3, dtype=np.float64)


def _point_space_matrices(
    meters_per_unit: float, up_axis: str
) -> tuple[np.ndarray, np.ndarray]:
    stage_to_canonical = np.eye(4, dtype=np.float64)
    stage_to_canonical[:3, :3] = float(meters_per_unit) * _axis_map(up_axis)
    return stage_to_canonical, np.linalg.inv(stage_to_canonical)


def _canonical_transform(
    world_transform, meters_per_unit: float, up_axis: str
) -> np.ndarray:
    """Map local metre coordinates to canonical Z-up world metres."""

    world = _np_matrix(world_transform)
    if not np.all(np.isfinite(world)):
        raise ValueError("USD transform contains a non-finite value")
    if not np.allclose(world[:3, 3], 0.0, atol=1.0e-12) or not math.isclose(
        float(world[3, 3]), 1.0, abs_tol=1.0e-12
    ):
        raise ValueError("USD transform is not affine")
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = world[:3, :3] @ _axis_map(up_axis)
    result[3, :3] = world[3, :3] * meters_per_unit @ _axis_map(up_axis)
    return result


def _source_digest(stage) -> str:
    """Digest the exact composed stage visible to analysis.

    Flattening includes root, session, references, payloads, variants, and current
    in-memory opinions.  This is intentionally evidence-oriented rather than a cheap
    pathname hash.
    """

    check_cancelled()
    flattened_layer = stage.Flatten()
    check_cancelled()
    digest = hashlib.sha256()
    # A multi-million-element USDC can expand to several times its packed size
    # when serialized as USDA text. Export one temporary crate and stream it so
    # source evidence never needs a second, unbounded text/bytes copy in memory.
    with tempfile.TemporaryDirectory(prefix="usd-cli-camera-digest-") as directory:
        path = f"{directory}/composed.usdc"
        if not flattened_layer.Export(path):
            raise RuntimeError("could not serialize the composed stage for hashing")
        check_cancelled()
        with open(path, "rb") as stream:  # noqa: PTH123 - temporary binary stream
            while chunk := stream.read(4 * 1024 * 1024):
                digest.update(chunk)
                check_cancelled()
    return "sha256:" + digest.hexdigest()


def _triangulate(
    face_counts: Iterable[int],
    face_indices: Iterable[int],
    *,
    holes: Iterable[int] = (),
    left_handed: bool = False,
) -> np.ndarray:
    # USD mesh topology is int32. Keep it in packed arrays and preallocate the
    # exact output rather than constructing Python lists containing as many as
    # three Python ``int`` objects per triangle (which can exceed a GiB well
    # before the supported packed-geometry limit).
    check_cancelled()
    indices = np.asarray(face_indices, dtype=np.int32)
    counts = np.asarray(face_counts, dtype=np.int32)
    hole_values = np.asarray(holes, dtype=np.int32)
    check_cancelled()
    if indices.ndim != 1 or counts.ndim != 1 or hole_values.ndim != 1:
        raise ValueError("mesh topology arrays must be one-dimensional")
    hole_set = {int(index) for index in hole_values}
    invalid_holes = sorted(
        index for index in hole_set if index < 0 or index >= len(counts)
    )
    if invalid_holes:
        raise ValueError(f"mesh hole index {invalid_holes[0]} does not name a face")

    triangle_count = 0
    for face_index, raw_count in enumerate(counts):
        if face_index % 1024 == 0:
            check_cancelled()
        count = int(raw_count)
        if count < 3:
            raise ValueError(f"mesh face {face_index} has fewer than three vertices")
        if face_index not in hole_set:
            triangle_count += count - 2
    check_cancelled()
    triangles = np.empty((triangle_count, 3), dtype=np.int32)
    offset = 0
    triangle_offset = 0
    for face_index, raw_count in enumerate(counts):
        if face_index % 1024 == 0:
            check_cancelled()
        count = int(raw_count)
        if offset + count > len(indices):
            raise ValueError("mesh face topology is malformed")
        if face_index not in hole_set:
            anchor = indices[offset]
            output = triangles[triangle_offset : triangle_offset + count - 2]
            output[:, 0] = anchor
            if left_handed:
                output[:, 1] = indices[offset + 2 : offset + count]
                output[:, 2] = indices[offset + 1 : offset + count - 1]
            else:
                output[:, 1] = indices[offset + 1 : offset + count - 1]
                output[:, 2] = indices[offset + 2 : offset + count]
            triangle_offset += count - 2
        offset += count
    if offset != len(indices):
        raise ValueError("mesh face counts do not consume all face indices")
    return triangles.reshape(-1)


def _validate_polygon_faces(
    vertices: np.ndarray,
    face_counts: Iterable[int],
    face_indices: Iterable[int],
    *,
    holes: Iterable[int] = (),
    prim_path: str,
) -> None:
    """Prove fan triangulation preserves each admitted polygon's surface."""

    counts = np.asarray(face_counts, dtype=np.int32)
    indices = np.asarray(face_indices, dtype=np.int32)
    hole_set = {int(value) for value in np.asarray(holes, dtype=np.int32)}
    offset = 0
    for face_index, raw_count in enumerate(counts):
        if face_index % 1024 == 0:
            check_cancelled()
        count = int(raw_count)
        if count < 0 or offset + count > len(indices):
            raise ValueError(f"mesh {prim_path} face topology is malformed")
        face = indices[offset : offset + count]
        offset += count
        if count <= 3 or face_index in hole_set:
            continue
        points = np.asarray(vertices[face], dtype=np.float64)
        extent = float(np.ptp(points, axis=0).max(initial=0.0))
        tolerance = max(1.0e-9, extent * 1.0e-6)
        origin = points[0]
        normal = None
        for index in range(1, count - 1):
            candidate = np.cross(points[index] - origin, points[index + 1] - origin)
            length = float(np.linalg.norm(candidate))
            if length > tolerance * tolerance:
                normal = candidate / length
                break
        if normal is None or np.any(np.abs((points - origin) @ normal) > tolerance):
            raise ValueError(
                f"unsupported_nonplanar_mesh_face: {prim_path} face {face_index} "
                "cannot be parity-safe fan triangulated"
            )
        projected = np.delete(points, int(np.argmax(np.abs(normal))), axis=1)
        minimum_turn = math.inf
        maximum_turn = -math.inf
        for index in range(count):
            first = projected[(index + 1) % count] - projected[index]
            second = projected[(index + 2) % count] - projected[(index + 1) % count]
            turn = float(first[0] * second[1] - first[1] * second[0])
            if abs(turn) > tolerance * tolerance:
                minimum_turn = min(minimum_turn, turn)
                maximum_turn = max(maximum_turn, turn)
        if not math.isfinite(minimum_turn) or minimum_turn < 0.0 < maximum_turn:
            raise ValueError(
                f"unsupported_concave_mesh_face: {prim_path} face {face_index} "
                "cannot be parity-safe fan triangulated"
            )
    if offset != len(indices):
        raise ValueError(f"mesh {prim_path} face counts do not consume all indices")


def _raw_mesh_data(mesh) -> _RawMeshData:
    holes_attr = getattr(mesh, "GetHoleIndicesAttr", lambda: None)()
    return _RawMeshData(
        points=mesh.GetPointsAttr().Get(),
        face_counts=mesh.GetFaceVertexCountsAttr().Get(),
        face_indices=mesh.GetFaceVertexIndicesAttr().Get(),
        holes=holes_attr.Get() if holes_attr else (),
    )


def _validate_mesh_subdivision(mesh) -> str:
    """Require an exact polygonal surface until limit-surface tessellation exists."""

    scheme = str(mesh.GetSubdivisionSchemeAttr().Get() or "catmullClark")
    if scheme != "none":
        raise ValueError(
            f"unsupported_subdivision_scheme: {mesh.GetPath()} uses {scheme!r}; "
            "camera analysis requires subdivisionScheme='none'"
        )
    return scheme


def _mesh_resource(
    mesh,
    meters_per_unit: float,
    *,
    raw: _RawMeshData | None = None,
) -> tuple[str, np.ndarray, np.ndarray] | None:
    subdivision_scheme = _validate_mesh_subdivision(mesh)
    data = raw if raw is not None else _raw_mesh_data(mesh)
    points = data.points
    counts = data.face_counts
    indices = data.face_indices
    if points is None or len(points) == 0:
        return None
    if counts is None or indices is None or (len(counts) == 0 and len(indices) == 0):
        return None
    # Vt point arrays expose NumPy's array protocol; a packed copy avoids a
    # second multi-million-element Python list at the ingestion boundary.
    check_cancelled()
    vertices = np.array(points, dtype=np.float32, copy=True)
    if vertices.shape != (len(points), 3):
        raise ValueError(f"mesh {mesh.GetPath()} points must be an N-by-3 array")
    vertices *= np.float32(meters_per_unit)
    if not np.all(np.isfinite(vertices)):
        raise ValueError(f"mesh {mesh.GetPath()} contains a non-finite point")
    orientation = str(mesh.GetOrientationAttr().Get() or "rightHanded")
    if orientation not in {"rightHanded", "leftHanded"}:
        raise ValueError(
            f"mesh {mesh.GetPath()} has unsupported orientation {orientation!r}"
        )
    raw_indices = np.asarray(indices, dtype=np.int32)
    if len(raw_indices) and (
        int(raw_indices.min()) < 0 or int(raw_indices.max()) >= len(vertices)
    ):
        raise ValueError("mesh face topology references a missing point")
    check_cancelled()
    _validate_polygon_faces(
        vertices,
        counts,
        raw_indices,
        holes=data.holes or (),
        prim_path=str(mesh.GetPath()),
    )
    check_cancelled()
    triangles = _triangulate(
        counts,
        raw_indices,
        holes=data.holes or (),
        left_handed=orientation == "leftHanded",
    )
    if not len(triangles):
        return None
    digest = hashlib.sha256()
    digest.update(subdivision_scheme.encode("ascii"))
    digest.update(orientation.encode("ascii"))
    digest.update(vertices.tobytes(order="C"))
    digest.update(triangles.tobytes(order="C"))
    return "mesh:" + digest.hexdigest(), vertices, triangles


def imageable_analysis_policy(prim) -> tuple[bool, str]:
    """Return whether one composed prim participates and its computed purpose."""

    from pxr import UsdGeom

    imageable = UsdGeom.Imageable(prim)
    if not imageable:
        return True, "default"
    purpose_token = imageable.ComputePurpose() or UsdGeom.Tokens.default_
    purpose = str(purpose_token)
    effective_visibility = getattr(imageable, "ComputeEffectiveVisibility", None)
    if effective_visibility is None:
        visibility = imageable.ComputeVisibility()
    else:
        try:
            visibility = effective_visibility(purpose_token)
        except TypeError:
            # Older usd-core releases expose only the default-purpose overload.
            visibility = effective_visibility()
    if visibility == UsdGeom.Tokens.invisible:
        return False, purpose
    return purpose in _PURPOSES, purpose


def _axis_rotation(axis: str) -> np.ndarray:
    """Local row rotation that maps Newton's +Z primitive axis to USD's axis."""

    axis = axis.upper()
    if axis not in {"X", "Y", "Z"}:
        raise ValueError(f"unsupported analytic primitive axis {axis!r}")
    if axis == "X":
        return np.asarray(
            [
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    if axis == "Y":
        return np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    return np.eye(4, dtype=np.float64)


def _positive_length(value: float, *, attribute: str, prim_path: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{prim_path} has nonpositive or non-finite {attribute}")
    return result


def _analytic_shape(
    prim, meters_per_unit: float
) -> tuple[str, dict[str, float], np.ndarray] | None:
    from pxr import UsdGeom

    if prim.IsA(UsdGeom.Cube):
        raw_size = UsdGeom.Cube(prim).GetSizeAttr().Get()
        size = (
            _positive_length(
                2.0 if raw_size is None else raw_size,
                attribute="size",
                prim_path=prim.GetPath().pathString,
            )
            * meters_per_unit
        )
        return "box", {"hx": size * 0.5, "hy": size * 0.5, "hz": size * 0.5}, np.eye(4)
    if prim.IsA(UsdGeom.Sphere):
        raw_radius = UsdGeom.Sphere(prim).GetRadiusAttr().Get()
        radius = (
            _positive_length(
                1.0 if raw_radius is None else raw_radius,
                attribute="radius",
                prim_path=prim.GetPath().pathString,
            )
            * meters_per_unit
        )
        return "sphere", {"radius": radius}, np.eye(4)
    for schema_name, kind in (
        ("Capsule", "capsule"),
        ("Cylinder", "cylinder"),
        ("Cone", "cone"),
    ):
        schema_type = getattr(UsdGeom, schema_name, None)
        if schema_type is None or not prim.IsA(schema_type):
            continue
        schema = schema_type(prim)
        raw_radius = schema.GetRadiusAttr().Get()
        raw_height = schema.GetHeightAttr().Get()
        default_radius, default_height = (0.5, 1.0) if kind == "capsule" else (1.0, 2.0)
        radius = (
            _positive_length(
                default_radius if raw_radius is None else raw_radius,
                attribute="radius",
                prim_path=prim.GetPath().pathString,
            )
            * meters_per_unit
        )
        height = (
            _positive_length(
                default_height if raw_height is None else raw_height,
                attribute="height",
                prim_path=prim.GetPath().pathString,
            )
            * meters_per_unit
        )
        axis = str(schema.GetAxisAttr().Get() or "Z")
        return (
            kind,
            {"radius": radius, "half_height": height * 0.5},
            _axis_rotation(axis),
        )
    plane_type = getattr(UsdGeom, "Plane", None)
    if plane_type is not None and prim.IsA(plane_type):
        plane = plane_type(prim)
        raw_width = plane.GetWidthAttr().Get()
        raw_length = plane.GetLengthAttr().Get()
        width = (
            _positive_length(
                2.0 if raw_width is None else raw_width,
                attribute="width",
                prim_path=prim.GetPath().pathString,
            )
            * meters_per_unit
        )
        length = (
            _positive_length(
                2.0 if raw_length is None else raw_length,
                attribute="length",
                prim_path=prim.GetPath().pathString,
            )
            * meters_per_unit
        )
        axis = str(plane.GetAxisAttr().Get() or "Z")
        return "plane", {"width": width, "length": length}, _axis_rotation(axis)
    return None


def _camera_ir(camera, canonical: np.ndarray, meters_per_unit: float) -> CameraIR:
    from pxr import UsdGeom

    path = camera.GetPath().pathString
    projection = str(camera.GetProjectionAttr().Get() or UsdGeom.Tokens.perspective)
    # OpenUSD lens attributes are authored in tenths of a scene unit, not mm.
    # raw * metersPerUnit / 10 converts to metres, hence raw * mpu * 100 to mm.
    lens_raw_to_mm = meters_per_unit * 100.0
    horizontal_offset = (
        float(camera.GetHorizontalApertureOffsetAttr().Get() or 0.0) * lens_raw_to_mm
    )
    vertical_offset = (
        float(camera.GetVerticalApertureOffsetAttr().Get() or 0.0) * lens_raw_to_mm
    )
    if not math.isfinite(horizontal_offset) or not math.isfinite(vertical_offset):
        raise ValueError(f"camera {path} has a non-finite aperture offset")

    raw_focal = camera.GetFocalLengthAttr().Get()
    focal = (
        _positive_length(
            raw_focal if raw_focal is not None else 50.0,
            attribute="focalLength",
            prim_path=path,
        )
        * lens_raw_to_mm
    )
    raw_horizontal_aperture = camera.GetHorizontalApertureAttr().Get()
    horizontal_aperture = (
        _positive_length(
            raw_horizontal_aperture if raw_horizontal_aperture is not None else 20.955,
            attribute="horizontalAperture",
            prim_path=path,
        )
        * lens_raw_to_mm
    )
    raw_vertical_aperture = camera.GetVerticalApertureAttr().Get()
    vertical_aperture = (
        _positive_length(
            raw_vertical_aperture if raw_vertical_aperture is not None else 15.2908,
            attribute="verticalAperture",
            prim_path=path,
        )
        * lens_raw_to_mm
    )
    raw_clipping = camera.GetClippingRangeAttr().Get()
    clipping = raw_clipping if raw_clipping is not None else (1.0, 1.0e6)
    near_m, far_m = (
        float(clipping[0]) * meters_per_unit,
        float(clipping[1]) * meters_per_unit,
    )
    if (
        not math.isfinite(near_m)
        or not math.isfinite(far_m)
        or near_m <= 0.0
        or far_m <= near_m
    ):
        raise ValueError(f"camera {path} has an invalid clipping range")
    return CameraIR(
        prim_path=path,
        transform=_matrix_tuple(canonical),
        focal_length_mm=focal,
        horizontal_aperture_mm=horizontal_aperture,
        vertical_aperture_mm=vertical_aperture,
        clipping_range_m=(near_m, far_m),
        horizontal_aperture_offset_mm=horizontal_offset,
        vertical_aperture_offset_mm=vertical_offset,
        projection=projection,
    )


def normalize_scene_analysis_policy(
    policy: SceneAnalysisPolicy | None,
) -> SceneAnalysisPolicy:
    from pxr import Sdf

    if policy is None:
        return SceneAnalysisPolicy()
    if not isinstance(policy, SceneAnalysisPolicy):
        raise TypeError("scene analysis policy must be a SceneAnalysisPolicy")
    normalized: dict[str, tuple[str, ...]] = {}
    for field_name in (
        "scope_paths",
        "include_paths",
        "exclude_paths",
        "helper_paths",
        "target_paths",
        "floor_paths",
    ):
        values = getattr(policy, field_name)
        if not isinstance(values, tuple) or any(
            not isinstance(value, str) for value in values
        ):
            raise ValueError(
                f"scene analysis {field_name} must be a tuple of USD paths"
            )
        paths: set[str] = set()
        for value in values:
            path = Sdf.Path(value)
            if (
                not path.IsAbsolutePath()
                or (not path.IsPrimPath() and not path.IsAbsoluteRootPath())
                or path.ContainsPrimVariantSelection()
            ):
                raise ValueError(
                    f"scene analysis {field_name} contains invalid prim path {value!r}"
                )
            paths.add(path.pathString)
        normalized[field_name] = tuple(sorted(paths))
    return SceneAnalysisPolicy(**normalized)


def scene_analysis_policy_from_dict(value: dict) -> SceneAnalysisPolicy:
    """Parse one exact serialized policy and apply canonical path normalization."""

    fields = (
        "scope_paths",
        "include_paths",
        "exclude_paths",
        "helper_paths",
        "target_paths",
        "floor_paths",
    )
    if not isinstance(value, dict) or set(value) != set(fields):
        raise ValueError("scene analysis policy must contain its exact path fields")
    values: dict[str, tuple[str, ...]] = {}
    for field_name in fields:
        paths = value[field_name]
        if not isinstance(paths, list) or any(
            not isinstance(path, str) for path in paths
        ):
            raise ValueError(f"scene analysis {field_name} must be a list of USD paths")
        values[field_name] = tuple(paths)
    return normalize_scene_analysis_policy(SceneAnalysisPolicy(**values))


def _matches_policy_path(path: str, roots: tuple[str, ...]) -> bool:
    return any(is_path_at_or_below(path, root) for root in roots)


def _analysis_role(path: str, policy: SceneAnalysisPolicy) -> AnalysisRole:
    if _matches_policy_path(path, policy.helper_paths):
        return "helper"
    if _matches_policy_path(path, policy.target_paths):
        return "target"
    if _matches_policy_path(path, policy.floor_paths):
        return "floor"
    return "obstacle"


def build_scene_analysis_ir(
    stage,
    *,
    policy: SceneAnalysisPolicy | None = None,
    include_geometry: bool = True,
) -> SceneAnalysisIR:
    """Build the authoritative composed-scene IR without importing Newton or Warp.

    ``include_geometry=False`` retains the composed camera inventory, stage units,
    canonical transforms, policy, and source digest while deliberately skipping shape
    ingestion.  Calibration-only interchange uses that mode so unsupported render
    geometry cannot block an operation that produces no visibility evidence.
    """

    from pxr import Usd, UsdGeom

    check_cancelled()
    policy = normalize_scene_analysis_policy(policy)
    # ``GetStageMetersPerUnit`` already supplies the USD schema fallback when the
    # metadata is unauthored.  Do not use truthiness here: OpenUSD permits an
    # explicitly authored zero, which must fail the positive-unit contract below
    # instead of silently being reinterpreted as centimetres.
    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    if not math.isfinite(meters_per_unit) or meters_per_unit <= 0.0:
        raise ValueError("stage metersPerUnit must be finite and positive")
    up_axis = str(UsdGeom.GetStageUpAxis(stage) or UsdGeom.Tokens.y).upper()
    if up_axis not in {"Y", "Z"}:
        raise ValueError(f"unsupported stage up axis {up_axis!r} (expected Y or Z)")
    stage_to_canonical, canonical_to_stage = _point_space_matrices(
        meters_per_unit, up_axis
    )
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())

    point_instancer_type = getattr(UsdGeom, "PointInstancer", None)

    mesh_by_id: dict[str, MeshResource] = {}
    mesh_resource_by_source: dict[str, tuple[str, np.ndarray, np.ndarray] | None] = {}
    shapes: list[ShapeIR] = []
    cameras: list[CameraIR] = []
    raw_budget = _RawSceneBudget()

    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        check_cancelled()
        allowed, purpose = imageable_analysis_policy(prim)
        if not allowed:
            continue
        path = prim.GetPath().pathString

        if prim.IsA(UsdGeom.Camera):
            world = xform_cache.GetLocalToWorldTransform(prim)
            canonical = _canonical_transform(world, meters_per_unit, up_axis)
            cameras.append(_camera_ir(UsdGeom.Camera(prim), canonical, meters_per_unit))
            continue
        if not include_geometry:
            continue

        world = xform_cache.GetLocalToWorldTransform(prim)
        canonical = _canonical_transform(world, meters_per_unit, up_axis)

        if policy.include_paths and not _matches_policy_path(
            path, policy.include_paths
        ):
            continue
        if _matches_policy_path(path, policy.exclude_paths):
            continue
        if point_instancer_type is not None and prim.IsA(point_instancer_type):
            # Point-instancer occurrence transforms and identities are not represented
            # by instance-proxy traversal. Apply visibility and explicit selection
            # policy first: ignored helpers and excluded geometry must not poison an
            # otherwise supported analysis, while any admitted instancer still fails
            # closed instead of silently tracing only its prototype.
            if _analysis_role(path, policy) == "helper":
                continue
            raise ValueError(
                f"unsupported_point_instancer: {path} requires explicit occurrence expansion"
            )

        mesh_resource_id: str | None = None
        parameters: dict[str, float] = {}
        local_prefix = np.eye(4, dtype=np.float64)
        if prim.IsA(UsdGeom.Mesh):
            raw_budget.ensure_shape_capacity(prim_path=path)
            source_prim = prim.GetPrimInPrototype() if prim.IsInstanceProxy() else prim
            source_path = source_prim.GetPath().pathString
            if source_path not in mesh_resource_by_source:
                mesh = UsdGeom.Mesh(prim)
                # Validate the authored surface contract before reading or
                # materializing potentially multi-million-element topology.
                _validate_mesh_subdivision(mesh)
                raw = _raw_mesh_data(mesh)
                raw_budget.account_mesh(raw, prim_path=path)
                mesh_resource_by_source[source_path] = _mesh_resource(
                    mesh,
                    meters_per_unit,
                    raw=raw,
                )
            resource = mesh_resource_by_source[source_path]
            if resource is None:
                continue
            mesh_resource_id, vertices, indices = resource
            mesh_by_id.setdefault(
                mesh_resource_id,
                MeshResource(mesh_resource_id, vertices, indices),
            )
            kind = "mesh"
        else:
            analytic = _analytic_shape(prim, meters_per_unit)
            if analytic is None:
                if prim.IsA(UsdGeom.Boundable):
                    role = _analysis_role(path, policy)
                    if role == "helper":
                        # Explicit helpers are a documented ignore-only policy
                        # for unsupported visualization geometry. They cannot
                        # silently become analysis obstacles.
                        continue
                    type_name = str(prim.GetTypeName() or "Boundable")
                    raise ValueError(f"unsupported_geometry_type: {path} ({type_name})")
                continue
            kind, parameters, local_prefix = analytic
        raw_budget.account_shape(prim_path=path)
        shape_transform = local_prefix @ canonical
        shapes.append(
            ShapeIR(
                shape_id=len(shapes),
                prim_path=path,
                kind=kind,
                transform=_matrix_tuple(shape_transform),
                mesh_resource_id=mesh_resource_id,
                parameters=parameters,
                purpose=purpose,
                instance_proxy=bool(prim.IsInstanceProxy()),
                analysis_role=_analysis_role(path, policy),
                in_scope=_matches_policy_path(path, policy.scope_paths),
            )
        )

    check_cancelled()
    shapes = [
        replace(shape, shape_id=shape_id)
        for shape_id, shape in enumerate(
            sorted(shapes, key=lambda item: item.prim_path)
        )
    ]
    cameras.sort(key=lambda item: item.prim_path)
    check_cancelled()
    return SceneAnalysisIR(
        meters_per_unit=meters_per_unit,
        source_up_axis=up_axis,
        canonical_up_axis="Z",
        stage_to_canonical=_matrix_tuple(stage_to_canonical),
        canonical_to_stage=_matrix_tuple(canonical_to_stage),
        meshes=tuple(mesh_by_id[key] for key in sorted(mesh_by_id)),
        shapes=tuple(shapes),
        cameras=tuple(cameras),
        source_digest=_source_digest(stage),
        policy=policy,
    )


def transform_points(points: np.ndarray, matrix: Matrix4) -> np.ndarray:
    """Transform N×3 row-vector points by a homogeneous matrix."""

    values = np.asarray(points, dtype=np.float64)
    homogeneous = np.concatenate(
        (values, np.ones((len(values), 1), dtype=np.float64)), axis=1
    )
    return (homogeneous @ np.asarray(matrix, dtype=np.float64))[:, :3]


def transform_directions(directions: np.ndarray, matrix: Matrix4) -> np.ndarray:
    values = np.asarray(directions, dtype=np.float64)
    result = values @ np.asarray(matrix, dtype=np.float64)[:3, :3]
    lengths = np.linalg.norm(result, axis=1)
    if np.any(lengths <= 1.0e-12):
        raise ValueError("cannot transform a zero-length direction")
    return result / lengths[:, None]


def analysis_bounds(
    scene: SceneAnalysisIR, *, roots: tuple[str, ...] = ()
) -> tuple[np.ndarray, np.ndarray]:
    """Bounds of exact admitted, non-helper IR shapes below ``roots``.

    Unlike a raw USD subtree bbox, this cannot be expanded by excluded or helper
    geometry that the analytic backend deliberately does not treat as an obstacle.
    """

    resources = {resource.resource_id: resource for resource in scene.meshes}
    mesh_scan_keys: set[tuple[str, bytes]] = set()
    mesh_vertex_visits = 0
    # Count every distinct scan before transforming the first vertex. A shared
    # maximum-size mesh under thousands of different occurrence transforms is
    # legal in the packed IR, but must not turn one bounds query into billions of
    # CPU vertex transforms.
    for shape in scene.shapes:
        check_cancelled()
        if shape.analysis_role == "helper" or (
            roots
            and not any(is_path_at_or_below(shape.prim_path, root) for root in roots)
        ):
            continue
        if shape.kind != "mesh":
            continue
        resource_id = shape.mesh_resource_id or ""
        resource = resources.get(resource_id)
        if resource is None or not len(resource.vertices):
            raise ValueError(
                f"analysis shape {shape.prim_path} has no mesh bounds resource"
            )
        matrix = np.asarray(shape.transform, dtype=np.float64)
        linear = np.ascontiguousarray(matrix[:3, :3])
        cache_key = (resource_id, linear.tobytes())
        if cache_key in mesh_scan_keys:
            continue
        mesh_scan_keys.add(cache_key)
        mesh_vertex_visits += len(resource.vertices)
        if mesh_vertex_visits > MAX_ANALYSIS_BOUNDS_VERTEX_VISITS:
            raise ValueError(
                "analysis_bounds_workload_limit: exact transformed mesh bounds "
                f"require more than {MAX_ANALYSIS_BOUNDS_VERTEX_VISITS:,} aggregate "
                "vertex visits across distinct resource/linear transforms "
                "(reduce transformed instances or split the analysis scope)"
            )

    mesh_bounds: dict[tuple[str, bytes], tuple[np.ndarray, np.ndarray]] = {}
    world_minimum: np.ndarray | None = None
    world_maximum: np.ndarray | None = None
    for shape in scene.shapes:
        check_cancelled()
        if shape.analysis_role == "helper" or (
            roots
            and not any(is_path_at_or_below(shape.prim_path, root) for root in roots)
        ):
            continue
        if shape.kind == "mesh":
            resource_id = shape.mesh_resource_id or ""
            resource = resources.get(resource_id)
            if resource is None or not len(resource.vertices):
                raise ValueError(
                    f"analysis shape {shape.prim_path} has no mesh bounds resource"
                )
            matrix = np.asarray(shape.transform, dtype=np.float64)
            linear = np.ascontiguousarray(matrix[:3, :3])
            translation = matrix[3, :3]
            cache_key = (resource_id, linear.tobytes())
            cached_bounds = mesh_bounds.get(cache_key)
            if cached_bounds is None:
                # A transformed local AABB is only conservative for a sparse mesh and
                # can be arbitrarily wider than its actual surface. Reduce transformed
                # vertices in bounded chunks instead. Translation is applied below, so
                # occurrences that differ only by translation reuse this exact scan.
                vertices = np.asarray(resource.vertices)
                transformed_minimum: np.ndarray | None = None
                transformed_maximum: np.ndarray | None = None
                for start in range(0, len(vertices), _MESH_BOUNDS_CHUNK_VERTICES):
                    check_cancelled()
                    chunk = np.asarray(
                        vertices[start : start + _MESH_BOUNDS_CHUNK_VERTICES],
                        dtype=np.float64,
                    )
                    transformed = chunk @ linear
                    chunk_minimum = transformed.min(axis=0)
                    chunk_maximum = transformed.max(axis=0)
                    transformed_minimum = (
                        chunk_minimum
                        if transformed_minimum is None
                        else np.minimum(transformed_minimum, chunk_minimum)
                    )
                    transformed_maximum = (
                        chunk_maximum
                        if transformed_maximum is None
                        else np.maximum(transformed_maximum, chunk_maximum)
                    )
                if transformed_minimum is None or transformed_maximum is None:
                    raise ValueError(
                        f"analysis shape {shape.prim_path} has no mesh bounds vertices"
                    )
                cached_bounds = transformed_minimum, transformed_maximum
                mesh_bounds[cache_key] = cached_bounds
            shape_minimum = cached_bounds[0] + translation
            shape_maximum = cached_bounds[1] + translation
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
            local_minimum, local_maximum = -extent, extent
            corners = np.asarray(
                [
                    [x, y, z]
                    for x in (local_minimum[0], local_maximum[0])
                    for y in (local_minimum[1], local_maximum[1])
                    for z in (local_minimum[2], local_maximum[2])
                ],
                dtype=np.float64,
            )
            transformed = transform_points(corners, shape.transform)
            shape_minimum = transformed.min(axis=0)
            shape_maximum = transformed.max(axis=0)
        world_minimum = (
            shape_minimum
            if world_minimum is None
            else np.minimum(world_minimum, shape_minimum)
        )
        world_maximum = (
            shape_maximum
            if world_maximum is None
            else np.maximum(world_maximum, shape_maximum)
        )
    if world_minimum is None or world_maximum is None:
        scope = ", ".join(roots) if roots else "the analysis scene"
        raise ValueError(f"{scope} contains no admitted non-helper geometry")
    return world_minimum, world_maximum


def canonical_bounds(range3d, scene: SceneAnalysisIR) -> tuple[np.ndarray, np.ndarray]:
    """Canonical axis-aligned bounds for a USD world-space ``Gf.Range3d``."""

    minimum, maximum = range3d.GetMin(), range3d.GetMax()
    corners = np.asarray(
        [
            [x, y, z]
            for x in (minimum[0], maximum[0])
            for y in (minimum[1], maximum[1])
            for z in (minimum[2], maximum[2])
        ],
        dtype=np.float64,
    )
    transformed = transform_points(corners, scene.stage_to_canonical)
    return transformed.min(axis=0), transformed.max(axis=0)


def camera_pose(camera: CameraIR) -> CameraPose:
    if camera.projection != "perspective":
        raise ValueError(
            "unsupported_camera_projection: "
            f"camera {camera.prim_path} uses {camera.projection!r}; "
            "camera analysis requires 'perspective'"
        )
    matrix = np.asarray(camera.transform, dtype=np.float64)
    position = matrix[3, :3]
    axes = matrix[:3, :3]
    lengths = np.linalg.norm(axes, axis=1)
    if np.any(~np.isfinite(matrix)) or np.any(lengths <= 1.0e-12):
        raise ValueError(
            f"camera {camera.prim_path} has a singular or non-finite transform"
        )
    normalized = axes / lengths[:, None]
    if not np.allclose(normalized @ normalized.T, np.eye(3), atol=2.0e-5, rtol=2.0e-5):
        raise ValueError(f"camera {camera.prim_path} has a sheared transform")
    if not math.isclose(float(np.linalg.det(normalized)), 1.0, abs_tol=2.0e-5):
        raise ValueError(f"camera {camera.prim_path} has a reflected transform")

    return CameraPose(
        prim_path=camera.prim_path,
        position_m=tuple(float(item) for item in position),
        right=tuple(float(item) for item in normalized[0]),
        up=tuple(float(item) for item in normalized[1]),
        forward=tuple(float(item) for item in -normalized[2]),
        focal_length_mm=camera.focal_length_mm,
        horizontal_aperture_mm=camera.horizontal_aperture_mm,
        vertical_aperture_mm=camera.vertical_aperture_mm,
        clipping_range_m=camera.clipping_range_m,
        horizontal_aperture_offset_mm=camera.horizontal_aperture_offset_mm,
        vertical_aperture_offset_mm=camera.vertical_aperture_offset_mm,
    )


def is_path_at_or_below(path: str, root: str) -> bool:
    """Exact USD descendant check; `/Shelf2` is not below `/Shelf`."""

    from pxr import Sdf

    return Sdf.Path(path).HasPrefix(Sdf.Path(root))
