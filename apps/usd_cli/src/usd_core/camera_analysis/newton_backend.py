# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Newton/Warp closest-hit backend for camera analysis.

Imports are intentionally local to backend construction.  Ordinary usd-cli startup and
help must not import Warp, initialize CUDA, or require the camera-analysis extra.
"""

from __future__ import annotations

import importlib.metadata
import math

import numpy as np

from usd_core.camera_analysis.cancellation import check_cancelled
from usd_core.camera_analysis.contracts import (
    MAX_ANALYSIS_GEOMETRY_BYTES,
    MAX_ANALYSIS_SHAPES,
    MAX_ANALYSIS_TRIANGLES,
    MAX_ANALYSIS_VERTICES,
    CameraBatchRequest,
    CameraObservation,
    MeshResource,
    RayHits,
    SceneAnalysisIR,
)

MAX_MODEL_SHAPES = MAX_ANALYSIS_SHAPES
MAX_MODEL_VERTICES = MAX_ANALYSIS_VERTICES
MAX_MODEL_TRIANGLES = MAX_ANALYSIS_TRIANGLES
MAX_MODEL_GEOMETRY_BYTES = MAX_ANALYSIS_GEOMETRY_BYTES
MAX_CAMERA_BATCH = 64
QUALIFIED_NEWTON_VERSION_PREFIX = "1.5."
QUALIFIED_WARP_VERSION_PREFIX = "1.16."


class CameraAnalysisUnavailable(RuntimeError):
    """The optional analytic backend cannot be used in this environment."""


def backend_versions() -> dict[str, str | bool]:
    versions: dict[str, str | bool] = {}
    for distribution, key in (("newton", "newton"), ("warp-lang", "warp")):
        try:
            versions[key] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[key] = "missing"
    versions["qualified"] = str(versions.get("newton", "")).startswith(
        QUALIFIED_NEWTON_VERSION_PREFIX
    ) and str(versions.get("warp", "")).startswith(
        QUALIFIED_WARP_VERSION_PREFIX
    )
    return versions


def require_qualified_backend_versions() -> dict[str, str | bool]:
    """Return the installed Newton/Warp identity or reject it before analysis.

    Camera analysis is an optional-provider path. Its reports can be used to
    author a rig or publish visibility evidence, so merely annotating an
    incompatible runtime as ``qualified=false`` is not safe: callers must stop
    before building the analysis model or producing any artifact.
    """

    versions = backend_versions()
    if versions.get("qualified") is True:
        return versions
    raise CameraAnalysisUnavailable(
        "camera analysis requires qualified Newton 1.5.x and Warp 1.16.x; "
        f"found newton={versions.get('newton', 'missing')!s}, "
        f"warp={versions.get('warp', 'missing')!s}. Install "
        "`usd-cli[camera-analysis]` for the qualified runtime"
    )


def _lazy_imports():
    try:
        import newton
        import warp as wp
    except ImportError as exc:
        raise CameraAnalysisUnavailable(
            "camera analysis requires the optional Newton/Warp backend; install "
            "`usd-cli[camera-analysis]`"
        ) from exc
    if not hasattr(newton, "intersect_ray") or not hasattr(newton, "ModelBuilder"):
        raise CameraAnalysisUnavailable(
            "installed Newton does not expose the required public ModelBuilder/intersect_ray APIs; "
            "install `usd-cli[camera-analysis]`"
        )
    return newton, wp


def _rotation_quaternion(rotation_row: np.ndarray) -> tuple[float, float, float, float]:
    """Quaternion (x, y, z, w) for a row-vector rotation matrix."""

    matrix = rotation_row.T  # the standard conversion below is column-vector based
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (matrix[2, 1] - matrix[1, 2]) / scale
        qy = (matrix[0, 2] - matrix[2, 0]) / scale
        qz = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            qw = (matrix[2, 1] - matrix[1, 2]) / scale
            qx = 0.25 * scale
            qy = (matrix[0, 1] + matrix[1, 0]) / scale
            qz = (matrix[0, 2] + matrix[2, 0]) / scale
        elif index == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            qw = (matrix[0, 2] - matrix[2, 0]) / scale
            qx = (matrix[0, 1] + matrix[1, 0]) / scale
            qy = 0.25 * scale
            qz = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            qw = (matrix[1, 0] - matrix[0, 1]) / scale
            qx = (matrix[0, 2] + matrix[2, 0]) / scale
            qy = (matrix[1, 2] + matrix[2, 1]) / scale
            qz = 0.25 * scale
    quaternion = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    length = float(np.linalg.norm(quaternion))
    if length <= 1.0e-12:
        raise ValueError("affine transform has no representable rotation")
    quaternion /= length
    return tuple(float(item) for item in quaternion)


def _decompose_rigid_scale(
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    linear = matrix[:3, :3]
    scales = np.linalg.norm(linear, axis=1)
    if np.any(scales <= 1.0e-10):
        return None
    rotation = linear / scales[:, None]
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=2.0e-5, rtol=2.0e-5):
        return None
    determinant = float(np.linalg.det(rotation))
    if math.isclose(determinant, -1.0, abs_tol=2.0e-5):
        # Keep a proper rotation for Warp and move the reflection into one signed
        # scale component.  Axis zero is deterministic and every supported
        # analytic primitive is symmetric across its local X plane.
        rotation = rotation.copy()
        scales = scales.copy()
        rotation[0] *= -1.0
        scales[0] *= -1.0
    elif not math.isclose(determinant, 1.0, abs_tol=2.0e-5):
        return None
    return matrix[3, :3], rotation, scales


def _affine_matrix(value, *, prim_path: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{prim_path} has a malformed or non-finite affine transform")
    if not np.allclose(matrix[:3, 3], 0.0, atol=1.0e-10) or not math.isclose(
        float(matrix[3, 3]), 1.0, abs_tol=1.0e-10
    ):
        raise ValueError(f"{prim_path} has a non-affine transform")
    return matrix


def _validate_capsule_scale(scale: np.ndarray, *, prim_path: str) -> None:
    values = np.abs(np.asarray(scale, dtype=np.float64))
    if not all(
        math.isclose(
            float(value),
            float(values[0]),
            rel_tol=1.0e-6,
            abs_tol=1.0e-10,
        )
        for value in values[1:]
    ):
        raise ValueError(
            f"unsupported_affine_transform: {prim_path} has nonuniform capsule "
            "scale; Newton capsule caps require uniform scale"
        )


def _validate_radial_scale(
    scale: np.ndarray, *, prim_path: str, shape_kind: str
) -> None:
    """Reject elliptical radial cross-sections before Newton allocates a model."""

    values = np.abs(np.asarray(scale, dtype=np.float64))
    if not math.isclose(
        float(values[0]),
        float(values[1]),
        rel_tol=1.0e-6,
        abs_tol=1.0e-10,
    ):
        raise ValueError(
            f"unsupported_affine_transform: {prim_path} has an elliptical "
            f"{shape_kind} cross-section"
        )


class NewtonVisibilityBackend:
    """A reusable Newton model and batched closest-hit query surface."""

    max_rays = 2_000_000
    max_camera_rays = 4_194_304
    chunk_size = 250_000
    max_shapes = MAX_MODEL_SHAPES
    max_mesh_vertices = MAX_MODEL_VERTICES
    max_mesh_triangles = MAX_MODEL_TRIANGLES
    max_geometry_bytes = MAX_MODEL_GEOMETRY_BYTES
    max_camera_batch = MAX_CAMERA_BATCH

    def __init__(self, scene: SceneAnalysisIR, device: str = "cpu") -> None:
        self.scene = scene
        check_cancelled()
        self._versions = require_qualified_backend_versions()
        check_cancelled()
        self._preflight_scene()
        check_cancelled()
        self.newton, self.wp = _lazy_imports()
        try:
            self.device = self.wp.get_device(device)
        except Exception as exc:  # noqa: BLE001 - normalize backend diagnostics
            raise ValueError(
                f"unknown or unavailable Warp device {device!r}: {exc}"
            ) from exc
        check_cancelled()
        self._shape_ir_by_newton_id: dict[int, int] = {}
        self._model = self._build_model()
        self._sensors: dict[bool, object] = {}

    def _preflight_scene(self) -> None:
        """Validate all model inputs and limits before Newton/Warp allocate."""

        check_cancelled()
        valid_roles = {"obstacle", "floor", "target", "helper"}
        shape_ids: set[int] = set()
        for shape in self.scene.shapes:
            check_cancelled()
            if shape.analysis_role not in valid_roles:
                raise ValueError(
                    f"unsupported SceneAnalysisIR analysis role {shape.analysis_role!r}"
                )
            if isinstance(shape.shape_id, bool) or not isinstance(
                shape.shape_id, int | np.integer
            ):
                raise ValueError(
                    f"SceneAnalysisIR shape id is not an integer: {shape.shape_id!r}"
                )
            shape_id = int(shape.shape_id)
            if shape_id < 0 or shape_id in shape_ids:
                raise ValueError(
                    f"duplicate or negative SceneAnalysisIR shape id {shape.shape_id!r}"
                )
            shape_ids.add(shape_id)
        self._analysis_shapes = tuple(
            shape for shape in self.scene.shapes if shape.analysis_role != "helper"
        )
        if not self._analysis_shapes:
            raise ValueError(
                "scene has no supported non-helper geometry for camera analysis"
            )
        if len(self._analysis_shapes) > self.max_shapes:
            raise ValueError(
                f"analysis model has {len(self._analysis_shapes):,} shapes; limit is "
                f"{self.max_shapes:,}"
            )

        used_resource_ids = {
            shape.mesh_resource_id
            for shape in self._analysis_shapes
            if shape.mesh_resource_id is not None
        }
        resources: dict[str, MeshResource] = {}
        total_vertices = 0
        total_triangles = 0
        geometry_bytes = 0
        for resource in self.scene.meshes:
            check_cancelled()
            if not resource.resource_id or resource.resource_id in resources:
                raise ValueError(
                    f"duplicate or empty mesh resource id {resource.resource_id!r}"
                )
            resources[resource.resource_id] = resource
            if resource.resource_id not in used_resource_ids:
                continue
            vertices = np.asarray(resource.vertices)
            indices = np.asarray(resource.indices)
            if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
                raise ValueError(
                    f"mesh resource {resource.resource_id} vertices must have shape (N, 3)"
                )
            if vertices.dtype.kind not in "fiu" or not np.all(np.isfinite(vertices)):
                raise ValueError(
                    f"mesh resource {resource.resource_id} has non-finite vertices"
                )
            if indices.ndim != 1 or len(indices) == 0 or len(indices) % 3:
                raise ValueError(
                    f"mesh resource {resource.resource_id} indices must be a nonempty flat triangle list"
                )
            if indices.dtype.kind not in "iu":
                raise ValueError(
                    f"mesh resource {resource.resource_id} indices must be integers"
                )
            if int(indices.min()) < 0 or int(indices.max()) >= len(vertices):
                raise ValueError(
                    f"mesh resource {resource.resource_id} references a missing vertex"
                )
            total_vertices += len(vertices)
            total_triangles += len(indices) // 3
            geometry_bytes += vertices.nbytes + indices.nbytes

        required_parameters = {
            "box": ("hx", "hy", "hz"),
            "sphere": ("radius",),
            "capsule": ("radius", "half_height"),
            "cylinder": ("radius", "half_height"),
            "cone": ("radius", "half_height"),
            "plane": ("width", "length"),
        }
        for shape in self._analysis_shapes:
            check_cancelled()
            matrix = _affine_matrix(shape.transform, prim_path=shape.prim_path)
            decomposition = _decompose_rigid_scale(matrix)
            if shape.kind == "mesh":
                resource = resources.get(shape.mesh_resource_id or "")
                if resource is None:
                    raise ValueError(
                        f"mesh shape {shape.prim_path} references a missing mesh resource"
                    )
                if decomposition is None:
                    determinant = float(np.linalg.det(matrix[:3, :3]))
                    if not math.isfinite(determinant) or abs(determinant) <= 1.0e-12:
                        raise ValueError(
                            f"unsupported_affine_transform: {shape.prim_path} has a singular transform"
                        )
                    # This occurrence needs a baked mesh instead of the reusable
                    # resource, so account for the extra Newton allocation.
                    vertices = np.asarray(resource.vertices)
                    indices = np.asarray(resource.indices)
                    total_vertices += len(vertices)
                    total_triangles += len(indices) // 3
                    geometry_bytes += vertices.nbytes + indices.nbytes
            elif shape.kind in required_parameters:
                if decomposition is None:
                    raise ValueError(
                        f"unsupported_affine_transform: {shape.prim_path} has shear "
                        "or a singular transform that Newton's analytic shape "
                        "transform cannot represent"
                    )
                if shape.kind == "capsule":
                    _validate_capsule_scale(decomposition[2], prim_path=shape.prim_path)
                elif shape.kind in {"cylinder", "cone"}:
                    _validate_radial_scale(
                        decomposition[2],
                        prim_path=shape.prim_path,
                        shape_kind=shape.kind,
                    )
                for parameter in required_parameters[shape.kind]:
                    value = float(shape.parameters.get(parameter, math.nan))
                    if not math.isfinite(value) or value <= 0.0:
                        raise ValueError(
                            f"{shape.prim_path} has invalid analytic parameter {parameter!r}"
                        )
            else:
                raise ValueError(
                    f"unsupported SceneAnalysisIR shape kind {shape.kind!r}"
                )

        if total_vertices > self.max_mesh_vertices:
            raise ValueError(
                f"analysis model would allocate {total_vertices:,} mesh vertices; limit is "
                f"{self.max_mesh_vertices:,}"
            )
        if total_triangles > self.max_mesh_triangles:
            raise ValueError(
                f"analysis model would allocate {total_triangles:,} mesh triangles; limit is "
                f"{self.max_mesh_triangles:,}"
            )
        if geometry_bytes > self.max_geometry_bytes:
            raise ValueError(
                f"analysis model geometry needs at least {geometry_bytes:,} bytes; limit is "
                f"{self.max_geometry_bytes:,}"
            )

    @property
    def versions(self) -> dict[str, str | bool]:
        return dict(self._versions)

    def _build_model(self):
        check_cancelled()
        builder = self.newton.ModelBuilder(up_axis=self.newton.Axis.Z)
        used_resource_ids = {
            shape.mesh_resource_id
            for shape in self._analysis_shapes
            if shape.mesh_resource_id is not None
        }
        resources = {
            resource.resource_id: resource
            for resource in self.scene.meshes
            if resource.resource_id in used_resource_ids
        }
        meshes: dict[str, object] = {}
        for resource_id, resource in resources.items():
            check_cancelled()
            meshes[resource_id] = self.newton.Mesh(
                resource.vertices,
                resource.indices,
                compute_inertia=False,
                is_solid=False,
            )

        for shape in self._analysis_shapes:
            check_cancelled()
            matrix = _affine_matrix(shape.transform, prim_path=shape.prim_path)
            decomposition = _decompose_rigid_scale(matrix)
            mesh = meshes.get(shape.mesh_resource_id or "")
            if decomposition is None:
                if shape.kind != "mesh" or mesh is None:
                    raise ValueError(
                        f"unsupported_affine_transform: {shape.prim_path} has shear "
                        "or a singular transform that Newton's analytic shape "
                        "transform cannot represent"
                    )
                # Preserve sharing for every ordinary instance.  Only a genuinely
                # sheared occurrence gets a deterministic derived mesh.
                resource = resources[shape.mesh_resource_id or ""]
                derived_vertices = resource.vertices.astype(np.float64) @ matrix[:3, :3]
                mesh = self.newton.Mesh(
                    derived_vertices.astype(np.float32),
                    resource.indices,
                    compute_inertia=False,
                    is_solid=False,
                )
                position = matrix[3, :3]
                rotation = np.eye(3)
                scale = np.ones(3)
            else:
                position, rotation, scale = decomposition
            geometry_scale = scale if shape.kind == "mesh" else np.abs(scale)
            xform = self.wp.transform(
                tuple(float(item) for item in position),
                _rotation_quaternion(rotation),
            )
            kwargs = {"body": -1, "xform": xform, "label": shape.prim_path}
            if shape.kind == "mesh":
                newton_id = builder.add_shape_mesh(
                    **kwargs,
                    mesh=mesh,
                    scale=tuple(float(item) for item in scale),
                )
            elif shape.kind == "box":
                newton_id = builder.add_shape_box(
                    **kwargs,
                    hx=shape.parameters["hx"] * float(geometry_scale[0]),
                    hy=shape.parameters["hy"] * float(geometry_scale[1]),
                    hz=shape.parameters["hz"] * float(geometry_scale[2]),
                )
            elif shape.kind == "sphere":
                radii = np.asarray(
                    [
                        shape.parameters["radius"] * float(value)
                        for value in geometry_scale
                    ]
                )
                if np.allclose(radii, radii[0], atol=1.0e-7, rtol=1.0e-6):
                    newton_id = builder.add_shape_sphere(
                        **kwargs, radius=float(radii[0])
                    )
                else:
                    newton_id = builder.add_shape_ellipsoid(
                        **kwargs,
                        rx=float(radii[0]),
                        ry=float(radii[1]),
                        rz=float(radii[2]),
                    )
            elif shape.kind == "capsule":
                _validate_capsule_scale(geometry_scale, prim_path=shape.prim_path)
                newton_id = builder.add_shape_capsule(
                    **kwargs,
                    radius=shape.parameters["radius"] * float(geometry_scale[0]),
                    half_height=shape.parameters["half_height"]
                    * float(geometry_scale[2]),
                )
            elif shape.kind == "cylinder":
                _validate_radial_scale(
                    geometry_scale,
                    prim_path=shape.prim_path,
                    shape_kind=shape.kind,
                )
                newton_id = builder.add_shape_cylinder(
                    **kwargs,
                    radius=shape.parameters["radius"] * float(geometry_scale[0]),
                    half_height=shape.parameters["half_height"]
                    * float(geometry_scale[2]),
                )
            elif shape.kind == "cone":
                _validate_radial_scale(
                    geometry_scale,
                    prim_path=shape.prim_path,
                    shape_kind=shape.kind,
                )
                newton_id = builder.add_shape_cone(
                    **kwargs,
                    radius=shape.parameters["radius"] * float(geometry_scale[0]),
                    half_height=shape.parameters["half_height"]
                    * float(geometry_scale[2]),
                )
            elif shape.kind == "plane":
                newton_id = builder.add_shape_plane(
                    **kwargs,
                    width=shape.parameters["width"] * float(geometry_scale[0]),
                    length=shape.parameters["length"] * float(geometry_scale[1]),
                )
            else:  # pragma: no cover - IR construction owns the closed kind set
                raise ValueError(
                    f"unsupported SceneAnalysisIR shape kind {shape.kind!r}"
                )
            self._shape_ir_by_newton_id[int(newton_id)] = shape.shape_id
        if not self._shape_ir_by_newton_id:
            raise ValueError(
                "scene has no supported visible geometry for camera analysis"
            )
        check_cancelled()
        model = builder.finalize(device=self.device)
        check_cancelled()
        return model

    def evaluate_rays(
        self,
        origins_m: np.ndarray,
        directions: np.ndarray,
        *,
        include_normals: bool = False,
    ) -> RayHits:
        check_cancelled()
        try:
            origin_count = len(origins_m)
            direction_count = len(directions)
        except TypeError as exc:
            raise ValueError(
                "ray origins and directions must both have shape (N, 3)"
            ) from exc
        count = origin_count
        if count > self.max_rays or direction_count > self.max_rays:
            requested = max(count, direction_count)
            raise ValueError(
                f"ray request has {requested:,} rays; limit is {self.max_rays:,} "
                "(reduce grid/candidates or split the request)"
            )
        origins = np.asarray(origins_m)
        ray_directions = np.asarray(directions)
        if (
            origins.ndim != 2
            or origins.shape[1] != 3
            or ray_directions.shape != origins.shape
        ):
            raise ValueError("ray origins and directions must both have shape (N, 3)")
        if np.any(~np.isfinite(origins)) or np.any(~np.isfinite(ray_directions)):
            raise ValueError("ray inputs must be finite")
        origins = origins.astype(np.float32, copy=False)
        ray_directions = ray_directions.astype(np.float32, copy=False)
        lengths = np.linalg.norm(ray_directions, axis=1)
        if np.any(lengths <= 1.0e-12):
            raise ValueError("ray directions must be nonzero")
        ray_directions = ray_directions / lengths[:, None]
        check_cancelled()
        distances = np.full(count, -1.0, dtype=np.float32)
        shape_ids = np.full(count, -1, dtype=np.int32)
        normals = np.zeros((count, 3), dtype=np.float32) if include_normals else None

        for start in range(0, count, self.chunk_size):
            check_cancelled()
            stop = min(count, start + self.chunk_size)
            size = stop - start
            wp_origins = self.wp.array(
                origins[start:stop], dtype=self.wp.vec3, device=self.device
            )
            wp_directions = self.wp.array(
                ray_directions[start:stop], dtype=self.wp.vec3, device=self.device
            )
            worlds = self.wp.full(size, -1, dtype=self.wp.int32, device=self.device)
            out_distance = self.wp.empty(
                size, dtype=self.wp.float32, device=self.device
            )
            out_shape = self.wp.empty(size, dtype=self.wp.int32, device=self.device)
            out_normal = (
                self.wp.empty(size, dtype=self.wp.vec3, device=self.device)
                if include_normals
                else None
            )
            self.newton.intersect_ray(
                self._model,
                ray_origins=wp_origins,
                ray_directions=wp_directions,
                ray_worlds=worlds,
                enable_global_world=True,
                out_dist=out_distance,
                out_shape_id=out_shape,
                out_normal=out_normal,
            )
            self.wp.synchronize_device(self.device)
            check_cancelled()
            distances[start:stop] = out_distance.numpy()
            raw_ids = out_shape.numpy()
            shape_ids[start:stop] = np.asarray(
                [self._shape_ir_by_newton_id.get(int(item), -1) for item in raw_ids],
                dtype=np.int32,
            )
            if normals is not None and out_normal is not None:
                normals[start:stop] = out_normal.numpy()
        missing = shape_ids < 0
        distances[missing] = -1.0
        if normals is not None:
            normals[missing] = 0.0
        return RayHits(distances_m=distances, shape_ids=shape_ids, normals=normals)

    def _validate_camera_request(self, request: CameraBatchRequest) -> list[np.ndarray]:
        channel_flags = {
            "include_depth": request.include_depth,
            "include_shape_ids": request.include_shape_ids,
            "include_normals": request.include_normals,
            "include_albedo": request.include_albedo,
        }
        invalid_flag = next(
            (
                name
                for name, value in channel_flags.items()
                if not isinstance(value, bool)
            ),
            None,
        )
        if invalid_flag is not None:
            raise ValueError(f"camera observation {invalid_flag} must be boolean")
        camera_count = len(request.cameras)
        if camera_count == 0:
            raise ValueError(
                "camera observation request must contain at least one camera"
            )
        if camera_count > self.max_camera_batch:
            raise ValueError(
                f"camera observation has {camera_count} cameras; limit is {self.max_camera_batch}"
            )
        if (
            isinstance(request.width, bool)
            or isinstance(request.height, bool)
            or not isinstance(request.width, int | np.integer)
            or not isinstance(request.height, int | np.integer)
            or request.width <= 0
            or request.height <= 0
        ):
            raise ValueError(
                "camera observation width and height must be positive integers"
            )
        pixel_count = camera_count * int(request.width) * int(request.height)
        if pixel_count > self.max_camera_rays:
            raise ValueError(
                f"camera observation would trace {pixel_count:,} rays; limit is "
                f"{self.max_camera_rays:,}"
            )
        channel_bytes = (
            (4 if request.include_depth else 0)
            + (4 if request.include_shape_ids else 0)
            + (12 if request.include_normals else 0)
            + (4 if request.include_albedo else 0)
        )
        if channel_bytes == 0:
            raise ValueError(
                "camera observation must request at least one output channel"
            )
        # Depth is also the validity/clipping mask for shape, normal, and albedo-only
        # requests. Account for that internal allocation before creating the sensor.
        working_channel_bytes = channel_bytes + (0 if request.include_depth else 4)
        working_bytes = pixel_count * working_channel_bytes
        if working_bytes > self.max_geometry_bytes:
            raise ValueError(
                f"camera observation outputs and clipping mask need "
                f"{working_bytes:,} bytes; limit is {self.max_geometry_bytes:,}"
            )

        transforms: list[np.ndarray] = []
        for index, camera in enumerate(request.cameras):
            position = np.asarray(camera.position_m, dtype=np.float64)
            rotation = np.asarray(
                [
                    camera.right,
                    camera.up,
                    -np.asarray(camera.forward, dtype=np.float64),
                ],
                dtype=np.float64,
            )
            intrinsics = np.asarray(
                [
                    camera.focal_length_mm,
                    camera.horizontal_aperture_mm,
                    camera.vertical_aperture_mm,
                    *camera.clipping_range_m,
                    camera.horizontal_aperture_offset_mm,
                    camera.vertical_aperture_offset_mm,
                ],
                dtype=np.float64,
            )
            if (
                position.shape != (3,)
                or rotation.shape != (3, 3)
                or not np.all(np.isfinite(position))
                or not np.all(np.isfinite(rotation))
                or not np.all(np.isfinite(intrinsics))
                or np.any(intrinsics[:3] <= 0.0)
                or intrinsics[3] <= 0.0
                or intrinsics[4] <= intrinsics[3]
            ):
                raise ValueError(
                    f"camera observation camera {index} has invalid pose or intrinsics"
                )
            if not np.allclose(
                rotation @ rotation.T, np.eye(3), atol=2.0e-5, rtol=2.0e-5
            ):
                raise ValueError(
                    f"camera observation camera {index} has non-orthogonal axes"
                )
            if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=2.0e-5):
                raise ValueError(
                    f"camera observation camera {index} has a reflected basis"
                )
            transforms.append(
                np.asarray(
                    [*position, *_rotation_quaternion(rotation)],
                    dtype=np.float32,
                )
            )
        return transforms

    def render_cameras(self, request: CameraBatchRequest) -> CameraObservation:
        """Render requested structured channels through public SensorTiledCamera APIs."""

        check_cancelled()
        transform_values = self._validate_camera_request(request)
        sensor_type = getattr(
            getattr(self.newton, "sensors", None), "SensorTiledCamera", None
        )
        if sensor_type is None:
            raise CameraAnalysisUnavailable(
                "installed Newton does not expose the public sensors.SensorTiledCamera API"
            )
        load_textures = bool(request.include_albedo)
        if load_textures not in self._sensors:
            check_cancelled()
            self._sensors[load_textures] = sensor_type(
                self._model, load_textures=load_textures
            )
        sensor = self._sensors[load_textures]

        camera_count = len(request.cameras)
        utils = sensor.utils
        camera_rays = utils.compute_camera_rays_pinhole(
            int(request.width),
            int(request.height),
            focal_length=np.asarray(
                [camera.focal_length_mm for camera in request.cameras], dtype=np.float32
            ),
            horizontal_aperture=np.asarray(
                [camera.horizontal_aperture_mm for camera in request.cameras],
                dtype=np.float32,
            ),
            vertical_aperture=np.asarray(
                [camera.vertical_aperture_mm for camera in request.cameras],
                dtype=np.float32,
            ),
            horizontal_aperture_offset=np.asarray(
                [camera.horizontal_aperture_offset_mm for camera in request.cameras],
                dtype=np.float32,
            ),
            vertical_aperture_offset=np.asarray(
                [camera.vertical_aperture_offset_mm for camera in request.cameras],
                dtype=np.float32,
            ),
        )
        camera_transforms = self.wp.array(
            transform_values,
            dtype=self.wp.transform,
            device=self.device,
        ).reshape((camera_count, 1))
        depth_output = utils.create_depth_image_output(
            request.width, request.height, camera_count
        )
        shape_output = (
            utils.create_shape_index_image_output(
                request.width, request.height, camera_count
            )
            if request.include_shape_ids
            else None
        )
        normal_output = (
            utils.create_normal_image_output(
                request.width, request.height, camera_count
            )
            if request.include_normals
            else None
        )
        albedo_output = (
            utils.create_albedo_image_output(
                request.width, request.height, camera_count
            )
            if request.include_albedo
            else None
        )
        max_distance = max(
            camera.clipping_range_m[1]
            * math.sqrt(
                1.0
                + max(abs(value) for value in camera.horizontal_tan_bounds) ** 2
                + max(abs(value) for value in camera.vertical_tan_bounds) ** 2
            )
            for camera in request.cameras
        )
        render_config = sensor_type.RenderConfig(
            enable_global_world=True,
            enable_textures=load_textures,
            enable_shadows=False,
            enable_particles=False,
            enable_backface_culling=False,
            output_color_space=self.newton.utils.ColorSpace.SRGB,
            max_distance=float(max_distance),
        )
        check_cancelled()
        sensor.update(
            self._model.state(),
            camera_transforms=camera_transforms,
            camera_rays=camera_rays,
            depth_image=depth_output,
            shape_index_image=shape_output,
            normal_image=normal_output,
            albedo_image=albedo_output,
            render_config=render_config,
        )
        self.wp.synchronize_device(self.device)
        check_cancelled()

        sensor_depth = depth_output.numpy()[0].astype(np.float32, copy=True)
        sensor_depth[(~np.isfinite(sensor_depth)) | (sensor_depth <= 0.0)] = -1.0
        pixel_x = (np.arange(request.width, dtype=np.float32) + 0.5) / float(
            request.width
        )
        pixel_y = (np.arange(request.height, dtype=np.float32) + 0.5) / float(
            request.height
        )
        focal = np.asarray(
            [camera.focal_length_mm for camera in request.cameras], dtype=np.float32
        )[:, None]
        horizontal_tangent = (
            (pixel_x[None, :] - 0.5)
            * np.asarray(
                [camera.horizontal_aperture_mm for camera in request.cameras],
                dtype=np.float32,
            )[:, None]
            + np.asarray(
                [camera.horizontal_aperture_offset_mm for camera in request.cameras],
                dtype=np.float32,
            )[:, None]
        ) / focal
        vertical_tangent = (
            (0.5 - pixel_y[None, :])
            * np.asarray(
                [camera.vertical_aperture_mm for camera in request.cameras],
                dtype=np.float32,
            )[:, None]
            + np.asarray(
                [camera.vertical_aperture_offset_mm for camera in request.cameras],
                dtype=np.float32,
            )[:, None]
        ) / focal
        forward_cosine = np.reciprocal(
            np.sqrt(
                1.0
                + vertical_tangent[:, :, None] ** 2
                + horizontal_tangent[:, None, :] ** 2
            )
        )
        forward_depth = sensor_depth * forward_cosine
        near = np.asarray(
            [camera.clipping_range_m[0] for camera in request.cameras],
            dtype=np.float32,
        )[:, None, None]
        far = np.asarray(
            [camera.clipping_range_m[1] for camera in request.cameras],
            dtype=np.float32,
        )[:, None, None]
        clipped = (forward_depth < near) | (forward_depth > far)
        sensor_depth[clipped] = -1.0
        shape_ids = None
        if shape_output is not None:
            raw_shape_ids = shape_output.numpy()[0]
            shape_ids = np.full(raw_shape_ids.shape, -1, dtype=np.int32)
            for newton_id, scene_id in sorted(self._shape_ir_by_newton_id.items()):
                shape_ids[raw_shape_ids == np.uint32(newton_id)] = scene_id
            missing = (sensor_depth < 0.0) | (shape_ids < 0)
            sensor_depth[missing] = -1.0
            shape_ids[missing] = -1
        normals = None
        if normal_output is not None:
            normals = normal_output.numpy()[0].astype(np.float32, copy=True)
            normals[sensor_depth < 0.0] = 0.0
        albedo = None
        if albedo_output is not None:
            packed = albedo_output.numpy()[0].astype(np.uint32, copy=True)
            albedo = np.stack(
                [
                    (packed >> np.uint32(shift)) & np.uint32(0xFF)
                    for shift in (0, 8, 16, 24)
                ],
                axis=-1,
            ).astype(np.uint8, copy=False)
            albedo[sensor_depth < 0.0] = 0
        check_cancelled()
        return CameraObservation(
            depth_m=sensor_depth if request.include_depth else None,
            shape_ids=shape_ids,
            normals=normals,
            albedo_rgba=albedo,
        )
