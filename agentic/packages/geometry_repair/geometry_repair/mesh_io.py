# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Format-neutral mesh extraction and conservative USD mesh updates."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .models import GeometryMetrics, MeshMetrics, MeshRecord

USD_SUFFIXES = {".usd", ".usda", ".usdc", ".usdz"}
MESH_SUFFIXES = {".obj", ".stl", ".ply", ".glb", ".gltf", ".3mf", ".off"}
_FLATTENED_PROTOTYPE_NAME = re.compile(r"^Flattened_Prototype_(?:canonical_)?\d+$")
_FLATTENED_PROTOTYPE_TEXT = re.compile(r"Flattened_Prototype_(?:canonical_)?\d+")


def _gltf_document(path: Path) -> dict[str, Any]:
    """Read only the structured glTF root needed for capability gating."""

    if path.suffix.lower() == ".gltf":
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        with path.open("rb") as stream:
            header = stream.read(12)
            if len(header) != 12:
                raise ValueError("GLB header is truncated")
            magic, version, declared_length = struct.unpack("<4sII", header)
            if magic != b"glTF" or version != 2:
                raise ValueError("GLB must use the glTF 2.0 container")
            if declared_length != path.stat().st_size:
                raise ValueError("GLB declared length does not match the source file")
            chunk_header = stream.read(8)
            if len(chunk_header) != 8:
                raise ValueError("GLB JSON chunk header is truncated")
            chunk_length, chunk_type = struct.unpack("<II", chunk_header)
            if chunk_type != 0x4E4F534A:
                raise ValueError("GLB first chunk is not structured JSON")
            encoded = stream.read(chunk_length)
            if len(encoded) != chunk_length:
                raise ValueError("GLB JSON chunk is truncated")
            payload = json.loads(encoded.rstrip(b" \t\r\n\x00").decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("glTF root must be a JSON object")
    return payload


def _gltf_dynamic_features(path: Path) -> list[str]:
    payload = _gltf_document(path)
    features: list[str] = []
    if payload.get("animations"):
        features.append("animations")
    if payload.get("skins"):
        features.append("skins")
    if any(
        primitive.get("targets")
        for mesh in payload.get("meshes") or []
        if isinstance(mesh, dict)
        for primitive in mesh.get("primitives") or []
        if isinstance(primitive, dict)
    ):
        features.append("morph_targets")
    return features


def _usd_mesh_role(prim: Any, purpose: str) -> str:
    from pxr import UsdGeom

    lowered_name = prim.GetName().lower()
    collision_markers = ("collision", "_coll_", "collider", "_col_")
    visual_markers = ("_vis", "visual", "render")
    helper_markers = ("helper", "locator", "gizmo", "debug")
    has_collision_marker = any(marker in lowered_name for marker in collision_markers)
    if any(marker in lowered_name for marker in visual_markers) and not has_collision_marker:
        return "render"
    if purpose in {str(UsdGeom.Tokens.guide), str(UsdGeom.Tokens.proxy)} or has_collision_marker:
        return "collision"
    if any(marker in lowered_name for marker in helper_markers):
        return "helper"
    return "render"


def _usd_mesh_has_enabled_collision(prim: Any, purpose: str) -> bool:
    """Return whether a prim participates in collision, including dual-role meshes."""

    from pxr import UsdGeom, UsdPhysics

    if purpose in {str(UsdGeom.Tokens.guide), str(UsdGeom.Tokens.proxy)}:
        return True
    if not prim.HasAPI(UsdPhysics.CollisionAPI):
        return False
    enabled = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
    return enabled is not False


@dataclass(frozen=True)
class MeshData:
    """Internal mesh arrays and source identity used by workers and metrics."""

    path: str
    local_vertices: np.ndarray
    world_vertices_m: np.ndarray
    triangles: np.ndarray
    source_face_counts: np.ndarray
    source_face_indices: np.ndarray
    purpose: str | None
    role: str
    is_instance_proxy: bool
    material_subset_count: int
    material_binding_count: int
    has_face_varying_data: bool
    authored_normal_count: int
    non_finite_normal_count: int
    authored_uv_count: int
    non_finite_uv_count: int
    transform_non_finite: bool
    transform_singular: bool
    transform_non_uniform: bool
    transform_sheared: bool
    transform_reflected: bool


def _transform_flags(matrix: np.ndarray) -> tuple[bool, bool, bool, bool, bool]:
    """Classify a 4x4 affine transform without modifying source geometry."""

    transform = np.asarray(matrix, dtype=np.float64).reshape((4, 4))
    non_finite = not bool(np.isfinite(transform).all())
    if non_finite:
        return True, True, False, False, False
    linear = transform[:3, :3]
    singular_values = np.linalg.svd(linear, compute_uv=False)
    scale_reference = max(float(np.max(singular_values)), 1.0)
    singular = bool(float(np.min(singular_values)) <= scale_reference * 1e-12)
    column_norms = np.linalg.norm(linear, axis=0)
    positive_norms = column_norms[column_norms > scale_reference * 1e-12]
    non_uniform = bool(
        len(positive_norms) > 1
        and float(np.max(positive_norms) - np.min(positive_norms))
        > max(float(np.max(positive_norms)) * 1e-8, 1e-12)
    )
    normalized = np.zeros_like(linear)
    valid = column_norms > scale_reference * 1e-12
    normalized[:, valid] = linear[:, valid] / column_norms[valid]
    gram = normalized.T @ normalized
    off_diagonal = gram - np.diag(np.diag(gram))
    sheared = bool(np.max(np.abs(off_diagonal), initial=0.0) > 1e-8)
    reflected = bool(np.linalg.det(linear) < 0.0)
    return non_finite, singular, non_uniform, sheared, reflected


def _triangulate(
    counts: np.ndarray, indices: np.ndarray, point_count: int
) -> tuple[np.ndarray, int, int]:
    triangles: list[tuple[int, int, int]] = []
    invalid_indices = 0
    non_triangular = 0
    offset = 0
    for raw_count in counts:
        count = int(raw_count)
        face = indices[offset : offset + max(count, 0)]
        offset += max(count, 0)
        if count < 3 or len(face) != count:
            invalid_indices += max(1, count)
            continue
        if np.any(face < 0) or np.any(face >= point_count):
            invalid_indices += int(np.count_nonzero((face < 0) | (face >= point_count)))
            continue
        if count != 3:
            non_triangular += 1
        for index in range(1, count - 1):
            triangles.append((int(face[0]), int(face[index]), int(face[index + 1])))
    if offset != len(indices):
        invalid_indices += abs(len(indices) - offset)
    return (
        np.asarray(triangles, dtype=np.int64).reshape((-1, 3)),
        invalid_indices,
        non_triangular,
    )


def _tessellate_usd_analytic_prim(prim: Any) -> tuple[np.ndarray, np.ndarray] | None:
    """Create a deterministic analysis mesh for a native USD analytic Gprim."""

    import trimesh
    from pxr import UsdGeom

    geometry: Any
    axis: str | None = None
    if prim.IsA(UsdGeom.Cube):
        size = float(UsdGeom.Cube(prim).GetSizeAttr().Get() or 2.0)
        geometry = trimesh.creation.box(extents=(size, size, size))
    elif prim.IsA(UsdGeom.Sphere):
        radius = float(UsdGeom.Sphere(prim).GetRadiusAttr().Get() or 1.0)
        geometry = trimesh.creation.icosphere(subdivisions=3, radius=radius)
    elif prim.IsA(UsdGeom.Cylinder):
        schema = UsdGeom.Cylinder(prim)
        radius = float(schema.GetRadiusAttr().Get() or 1.0)
        height = float(schema.GetHeightAttr().Get() or 2.0)
        axis = str(schema.GetAxisAttr().Get() or UsdGeom.Tokens.z)
        geometry = trimesh.creation.cylinder(radius=radius, height=height, sections=48)
    elif prim.IsA(UsdGeom.Cone):
        schema = UsdGeom.Cone(prim)
        radius = float(schema.GetRadiusAttr().Get() or 1.0)
        height = float(schema.GetHeightAttr().Get() or 2.0)
        axis = str(schema.GetAxisAttr().Get() or UsdGeom.Tokens.z)
        geometry = trimesh.creation.cone(radius=radius, height=height, sections=48)
    elif prim.IsA(UsdGeom.Capsule):
        schema = UsdGeom.Capsule(prim)
        radius = float(schema.GetRadiusAttr().Get() or 1.0)
        height = float(schema.GetHeightAttr().Get() or 2.0)
        axis = str(schema.GetAxisAttr().Get() or UsdGeom.Tokens.z)
        geometry = trimesh.creation.capsule(radius=radius, height=height, count=(24, 24))
    else:
        return None

    if axis is not None and axis.upper() != "Z":
        target = {
            "X": np.asarray((1.0, 0.0, 0.0), dtype=np.float64),
            "Y": np.asarray((0.0, 1.0, 0.0), dtype=np.float64),
        }.get(axis.upper())
        if target is None:
            raise ValueError(f"Unsupported USD analytic primitive axis {axis!r}")
        transform = trimesh.geometry.align_vectors(
            np.asarray((0.0, 0.0, 1.0), dtype=np.float64),
            target,
        )
        geometry.apply_transform(transform)
    return (
        np.asarray(geometry.vertices, dtype=np.float64).reshape((-1, 3)),
        np.asarray(geometry.faces, dtype=np.int64).reshape((-1, 3)),
    )


def _load_usd_meshes(
    path: Path,
    *,
    include_guide_purpose: bool = False,
) -> tuple[list[MeshData], dict[str, Any]]:
    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError(f"Usd.Stage.Open returned None for {path}")
    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    up_axis = str(UsdGeom.GetStageUpAxis(stage))
    transform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    meshes: list[MeshData] = []
    collision_paths: list[str] = []
    helper_paths: list[str] = []
    material_binding_count = 0
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        purpose = str(UsdGeom.Imageable(prim).ComputePurpose())
        role = _usd_mesh_role(prim, purpose)
        if _usd_mesh_has_enabled_collision(prim, purpose):
            collision_paths.append(str(prim.GetPath()))
        if role == "helper":
            helper_paths.append(str(prim.GetPath()))
        if role != "render" and not include_guide_purpose:
            continue
        points = np.asarray(
            [[float(value) for value in point] for point in (mesh.GetPointsAttr().Get() or [])],
            dtype=np.float64,
        ).reshape((-1, 3))
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get() or [], dtype=np.int64)
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get() or [], dtype=np.int64)
        triangles, _, _ = _triangulate(counts, indices, len(points))
        matrix = transform_cache.GetLocalToWorldTransform(prim)
        matrix_array = np.asarray(
            [[float(matrix[row][column]) for column in range(4)] for row in range(4)],
            dtype=np.float64,
        )
        transform_flags = _transform_flags(matrix_array)
        world = np.asarray(
            [
                [float(coord) * meters_per_unit for coord in matrix.Transform(Gf.Vec3d(*point))]
                for point in points
            ],
            dtype=np.float64,
        ).reshape((-1, 3))
        if up_axis.upper() == "Y" and len(world):
            world = np.column_stack((world[:, 0], -world[:, 2], world[:, 1]))
        elif up_axis.upper() == "X" and len(world):
            world = np.column_stack((-world[:, 2], world[:, 1], world[:, 0]))
        subsets = UsdGeom.Subset.GetAllGeomSubsets(mesh)
        face_varying_data = False
        authored_uv_count = 0
        non_finite_uv_count = 0
        for primvar in UsdGeom.PrimvarsAPI(prim).GetPrimvars():
            if primvar.GetInterpolation() in {
                UsdGeom.Tokens.faceVarying,
                UsdGeom.Tokens.uniform,
            }:
                face_varying_data = True
            name = str(primvar.GetPrimvarName()).lower()
            if name in {"st", "uv", "texcoord", "texcoords"}:
                values = primvar.Get() or []
                authored_uv_count += len(values)
                for value in values:
                    coordinates = [float(item) for item in value]
                    if not all(math.isfinite(item) for item in coordinates):
                        non_finite_uv_count += 1
        authored_normals = mesh.GetNormalsAttr().Get() or []
        authored_normal_count = len(authored_normals)
        non_finite_normal_count = sum(
            not all(math.isfinite(float(item)) for item in normal) for normal in authored_normals
        )
        if (
            mesh.GetNormalsInterpolation()
            in {
                UsdGeom.Tokens.faceVarying,
                UsdGeom.Tokens.uniform,
            }
            and mesh.GetNormalsAttr().HasAuthoredValueOpinion()
        ):
            face_varying_data = True
        bindings = [
            relationship
            for relationship in prim.GetRelationships()
            if relationship.GetName().startswith("material:binding")
        ]
        material_binding_count += len(bindings)
        meshes.append(
            MeshData(
                path=str(prim.GetPath()),
                local_vertices=points,
                world_vertices_m=world,
                triangles=triangles,
                source_face_counts=counts,
                source_face_indices=indices,
                purpose=purpose,
                role=role,
                is_instance_proxy=prim.IsInstanceProxy(),
                material_subset_count=len(subsets),
                material_binding_count=len(bindings),
                has_face_varying_data=face_varying_data,
                authored_normal_count=authored_normal_count,
                non_finite_normal_count=non_finite_normal_count,
                authored_uv_count=authored_uv_count,
                non_finite_uv_count=non_finite_uv_count,
                transform_non_finite=transform_flags[0],
                transform_singular=transform_flags[1],
                transform_non_uniform=transform_flags[2],
                transform_sheared=transform_flags[3],
                transform_reflected=transform_flags[4],
            )
        )
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        analytic = _tessellate_usd_analytic_prim(prim)
        if analytic is None:
            continue
        purpose = str(UsdGeom.Imageable(prim).ComputePurpose())
        role = _usd_mesh_role(prim, purpose)
        path_text = str(prim.GetPath())
        if _usd_mesh_has_enabled_collision(prim, purpose):
            collision_paths.append(path_text)
        if role == "helper":
            helper_paths.append(path_text)
        if role != "render" and not include_guide_purpose:
            continue
        points, triangles = analytic
        matrix = transform_cache.GetLocalToWorldTransform(prim)
        matrix_array = np.asarray(
            [[float(matrix[row][column]) for column in range(4)] for row in range(4)],
            dtype=np.float64,
        )
        transform_flags = _transform_flags(matrix_array)
        world = np.asarray(
            [
                [float(coord) * meters_per_unit for coord in matrix.Transform(Gf.Vec3d(*point))]
                for point in points
            ],
            dtype=np.float64,
        ).reshape((-1, 3))
        if up_axis.upper() == "Y" and len(world):
            world = np.column_stack((world[:, 0], -world[:, 2], world[:, 1]))
        elif up_axis.upper() == "X" and len(world):
            world = np.column_stack((-world[:, 2], world[:, 1], world[:, 0]))
        bindings = [
            relationship
            for relationship in prim.GetRelationships()
            if relationship.GetName().startswith("material:binding")
        ]
        material_binding_count += len(bindings)
        meshes.append(
            MeshData(
                path=path_text,
                local_vertices=points,
                world_vertices_m=world,
                triangles=triangles,
                source_face_counts=np.full(len(triangles), 3, dtype=np.int64),
                source_face_indices=triangles.reshape(-1),
                purpose=purpose,
                role=role,
                is_instance_proxy=prim.IsInstanceProxy(),
                material_subset_count=0,
                material_binding_count=len(bindings),
                has_face_varying_data=False,
                authored_normal_count=0,
                non_finite_normal_count=0,
                authored_uv_count=0,
                non_finite_uv_count=0,
                transform_non_finite=transform_flags[0],
                transform_singular=transform_flags[1],
                transform_non_uniform=transform_flags[2],
                transform_sheared=transform_flags[3],
                transform_reflected=transform_flags[4],
            )
        )
    roots = [prim for prim in stage.GetPseudoRoot().GetChildren() if prim.IsValid()]
    return meshes, {
        "source_format": path.suffix.lower().lstrip("."),
        "up_axis": up_axis,
        "meters_per_unit": meters_per_unit,
        "root_count": len(roots),
        "default_prim_path": str(stage.GetDefaultPrim().GetPath())
        if stage.GetDefaultPrim()
        else None,
        "material_binding_count": material_binding_count,
        "source_collision_paths": collision_paths,
        "source_helper_paths": helper_paths,
    }


def _load_external_meshes(path: Path) -> tuple[list[MeshData], dict[str, Any]]:
    import trimesh

    scene = trimesh.load_scene(path, process=False)
    meshes: list[MeshData] = []
    node_records = []
    for node_name in sorted(scene.graph.nodes_geometry):
        transform, geometry_name = scene.graph[node_name]
        node_records.append((node_name, geometry_name, transform))
    if not node_records:
        node_records = [
            (name, name, np.eye(4, dtype=np.float64)) for name in sorted(scene.geometry)
        ]
    for index, (node_name, geometry_name, transform) in enumerate(node_records):
        geometry = scene.geometry.get(geometry_name)
        if not isinstance(geometry, trimesh.Trimesh):
            continue
        vertices = np.asarray(geometry.vertices, dtype=np.float64).reshape((-1, 3))
        transform_array = np.asarray(transform, dtype=np.float64)
        transform_flags = _transform_flags(transform_array)
        world_vertices = trimesh.transform_points(vertices, transform_array)
        faces = np.asarray(geometry.faces, dtype=np.int64).reshape((-1, 3))
        raw_normals = geometry.vertex_attributes.get("normals")
        normal_values = (
            np.asarray(raw_normals, dtype=np.float64).reshape((-1, 3))
            if raw_normals is not None
            else np.empty((0, 3), dtype=np.float64)
        )
        raw_uvs = getattr(geometry.visual, "uv", None)
        uv_values = (
            np.asarray(raw_uvs, dtype=np.float64).reshape((-1, 2))
            if raw_uvs is not None
            else np.empty((0, 2), dtype=np.float64)
        )
        safe_node_name = str(node_name or f"mesh_{index:04d}").replace("/", "_")
        meshes.append(
            MeshData(
                path=f"/{safe_node_name}_{index:04d}",
                local_vertices=vertices,
                world_vertices_m=world_vertices,
                triangles=faces,
                source_face_counts=np.full(len(faces), 3, dtype=np.int64),
                source_face_indices=faces.reshape(-1),
                purpose="default",
                role="render",
                is_instance_proxy=False,
                material_subset_count=0,
                material_binding_count=1 if geometry.visual is not None else 0,
                has_face_varying_data=geometry.visual is not None,
                authored_normal_count=len(normal_values),
                non_finite_normal_count=int(
                    np.count_nonzero(~np.isfinite(normal_values).all(axis=1))
                ),
                authored_uv_count=len(uv_values),
                non_finite_uv_count=int(np.count_nonzero(~np.isfinite(uv_values).all(axis=1))),
                transform_non_finite=transform_flags[0],
                transform_singular=transform_flags[1],
                transform_non_uniform=transform_flags[2],
                transform_sheared=transform_flags[3],
                transform_reflected=transform_flags[4],
            )
        )
    return meshes, {
        "source_format": path.suffix.lower().lstrip("."),
        "up_axis": None,
        "meters_per_unit": None,
        "root_count": len(meshes),
        "default_prim_path": None,
        "material_binding_count": sum(mesh.material_binding_count for mesh in meshes),
        "source_collision_paths": [],
        "source_helper_paths": [],
    }


def load_meshes(
    path: str | Path,
    *,
    include_guide_purpose: bool = False,
) -> tuple[list[MeshData], dict[str, Any]]:
    """Load USD or common polygonal mesh input without automatic processing.

    Guide-purpose USD geometry is excluded by default because it is commonly used for
    collision proxies and should not affect render-geometry diagnosis. Collision-only
    consumers must opt in explicitly.
    """

    source = Path(path).expanduser().resolve()
    suffix = source.suffix.lower()
    if suffix in USD_SUFFIXES:
        return _load_usd_meshes(
            source,
            include_guide_purpose=include_guide_purpose,
        )
    if suffix in MESH_SUFFIXES:
        return _load_external_meshes(source)
    raise ValueError(f"Unsupported mesh diagnostic format: {suffix or '<none>'}")


def external_mesh_to_usd_stage(
    source: str | Path,
    target: str | Path,
    *,
    meters_per_unit: float | None = None,
    up_axis: str | None = None,
) -> Path:
    """Create a source-mapped USD working layer from a supported mesh file.

    Unknown physical units or up axis remain caller-owned. Only glTF's specified
    meter/Y-up frame is inferred automatically; other formats require explicit
    values before a physical repair profile can mutate their working copy.
    """

    import trimesh
    from pxr import Gf, Sdf, Tf, Usd, UsdGeom, UsdShade, Vt

    source_path = Path(source).expanduser().resolve()
    target_path = Path(target).expanduser().resolve()
    if source_path.suffix.lower() not in MESH_SUFFIXES:
        raise ValueError(f"unsupported external mesh format: {source_path.suffix}")
    if source_path.suffix.lower() in {".gltf", ".glb"}:
        dynamic_features = _gltf_dynamic_features(source_path)
        if dynamic_features:
            raise ValueError(
                "external mesh normalization refuses dynamic glTF features until a "
                "semantics-preserving USD converter is selected: " + ", ".join(dynamic_features)
            )
    scene = trimesh.load_scene(source_path, process=False)
    source_units = str(scene.units or "").lower() or None
    if meters_per_unit is None and source_path.suffix.lower() in {".gltf", ".glb"}:
        meters_per_unit = 1.0
    if meters_per_unit is None and source_units:
        try:
            meters_per_unit = float(trimesh.units.unit_conversion(source_units, "meters"))
        except (KeyError, ValueError):
            meters_per_unit = None
    if up_axis is None and source_path.suffix.lower() in {".gltf", ".glb"}:
        up_axis = "Y"
    if meters_per_unit is None or up_axis is None:
        missing = []
        if meters_per_unit is None:
            missing.append("source_meters_per_unit")
        if up_axis is None:
            missing.append("source_up_axis")
        raise ValueError("external mesh physical normalization requires " + " and ".join(missing))
    if not math.isfinite(meters_per_unit) or meters_per_unit <= 0.0:
        raise ValueError("meters_per_unit must be finite and positive")
    normalized_axis = str(up_axis).upper()
    if normalized_axis not in {"X", "Y", "Z"}:
        raise ValueError("up_axis must be X, Y, or Z")

    target_path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(target_path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage_axis = "Z" if normalized_axis == "X" else normalized_axis
    UsdGeom.SetStageUpAxis(stage, stage_axis)
    root = UsdGeom.Xform.Define(stage, "/ImportedAsset")
    root.GetPrim().SetCustomDataByKey("geometryRepairSourcePath", str(source_path))
    root.GetPrim().SetCustomDataByKey("geometryRepairSourceUnits", source_units or "unknown")
    root.GetPrim().SetCustomDataByKey("geometryRepairSourceUpAxis", normalized_axis)
    if normalized_axis == "X":
        # OpenUSD stage metadata supports only Y-up and Z-up. Map source X-up
        # into canonical Z-up at the working-layer root without changing mesh
        # topology, node transforms, or source provenance.
        root.AddRotateYOp().Set(-90.0)
    stage.SetDefaultPrim(root.GetPrim())
    looks = UsdGeom.Scope.Define(stage, "/ImportedAsset/Looks")
    looks.GetPrim().SetCustomDataByKey("geometryRepairGeneratedWorkingScope", True)

    base = scene.graph.base_frame
    parents = dict(scene.graph.transforms.parents)
    node_data = scene.graph.transforms.node_data
    edge_data = scene.graph.transforms.edge_data
    pending = {str(node) for node in scene.graph.nodes if str(node) != str(base)}
    paths: dict[str, Sdf.Path] = {str(base): Sdf.Path("/ImportedAsset")}
    used_children: dict[str, set[str]] = {}
    material_count = 0
    while pending:
        progressed = False
        for node_name in sorted(pending):
            parent_name = str(parents.get(node_name, base))
            if parent_name not in paths:
                continue
            parent_path = paths[parent_name]
            valid_name = str(Tf.MakeValidIdentifier(node_name)) or "Node"
            siblings = used_children.setdefault(str(parent_path), set())
            candidate_name = valid_name
            suffix = 1
            while candidate_name in siblings:
                suffix += 1
                candidate_name = f"{valid_name}_{suffix:03d}"
            siblings.add(candidate_name)
            prim_path = parent_path.AppendChild(candidate_name)
            geometry_name = node_data.get(node_name, {}).get("geometry")
            geometry = scene.geometry.get(geometry_name) if geometry_name is not None else None
            schema = (
                UsdGeom.Mesh.Define(stage, prim_path)
                if isinstance(geometry, trimesh.Trimesh)
                else UsdGeom.Xform.Define(stage, prim_path)
            )
            prim = schema.GetPrim()
            prim.SetCustomDataByKey("geometryRepairSourceNode", node_name)
            if geometry_name is not None:
                prim.SetCustomDataByKey("geometryRepairSourceGeometry", str(geometry_name))
            local = np.asarray(
                edge_data.get((parent_name, node_name), {}).get("matrix", np.eye(4)),
                dtype=np.float64,
            ).copy()
            local[:3, 3] *= meters_per_unit
            UsdGeom.Xformable(prim).AddTransformOp().Set(Gf.Matrix4d(*local.T.reshape(-1).tolist()))
            if isinstance(geometry, trimesh.Trimesh):
                mesh = UsdGeom.Mesh(prim)
                vertices = (
                    np.asarray(geometry.vertices, dtype=np.float64).reshape((-1, 3))
                    * meters_per_unit
                )
                faces = np.asarray(geometry.faces, dtype=np.int32).reshape((-1, 3))
                mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices.astype(np.float32)))
                mesh.CreateFaceVertexCountsAttr(
                    Vt.IntArray.FromNumpy(np.full(len(faces), 3, dtype=np.int32))
                )
                mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.reshape(-1)))
                mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
                raw_normals = geometry.vertex_attributes.get("normals")
                if raw_normals is not None:
                    normals = np.asarray(raw_normals, dtype=np.float32).reshape((-1, 3))
                    if len(normals) == len(vertices):
                        mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals))
                        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
                raw_uv = getattr(geometry.visual, "uv", None)
                if raw_uv is not None:
                    uv = np.asarray(raw_uv, dtype=np.float32).reshape((-1, 2))
                    if len(uv) == len(vertices):
                        primvar = UsdGeom.PrimvarsAPI(prim).CreatePrimvar(
                            "st",
                            Sdf.ValueTypeNames.TexCoord2fArray,
                            UsdGeom.Tokens.vertex,
                        )
                        primvar.Set(Vt.Vec2fArray.FromNumpy(uv))
                material = getattr(geometry.visual, "material", None)
                if material is not None:
                    material_count += 1
                    material_name = (
                        str(Tf.MakeValidIdentifier(str(getattr(material, "name", "Material"))))
                        or "Material"
                    )
                    material_path = Sdf.Path("/ImportedAsset/Looks").AppendChild(
                        f"{material_name}_{material_count:04d}"
                    )
                    usd_material = UsdShade.Material.Define(stage, material_path)
                    shader = UsdShade.Shader.Define(
                        stage, material_path.AppendChild("PreviewSurface")
                    )
                    shader.CreateIdAttr("UsdPreviewSurface")
                    color = np.asarray(
                        getattr(material, "main_color", [180, 180, 180, 255]),
                        dtype=np.float64,
                    ).reshape(-1)
                    if len(color) >= 3:
                        color = np.clip(color[:3] / 255.0, 0.0, 1.0)
                        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
                            Gf.Vec3f(*[float(value) for value in color])
                        )
                    usd_material.CreateSurfaceOutput().ConnectToSource(
                        shader.ConnectableAPI(), "surface"
                    )
                    UsdShade.MaterialBindingAPI.Apply(prim).Bind(usd_material)
            paths[node_name] = prim_path
            pending.remove(node_name)
            progressed = True
        if not progressed:
            raise ValueError(f"external mesh scene graph contains an unresolved cycle: {pending}")
    layer_data = dict(stage.GetRootLayer().customLayerData or {})
    layer_data.update(
        {
            "geometryRepairExternalMeshWorkingCopy": True,
            "geometryRepairSourceFormat": source_path.suffix.lower().lstrip("."),
            "geometryRepairSourceMetersPerUnit": float(meters_per_unit),
            "geometryRepairSourceUpAxis": normalized_axis,
        }
    )
    stage.GetRootLayer().customLayerData = layer_data
    stage.GetRootLayer().Save()
    return target_path


def _component_count(triangles: np.ndarray) -> int:
    if len(triangles) == 0:
        return 0
    parent = list(range(len(triangles)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    owners: dict[tuple[int, int], int] = {}
    for face_index, face in enumerate(triangles):
        for start, end in zip(face, np.roll(face, -1), strict=True):
            edge = tuple(sorted((int(start), int(end))))
            previous = owners.setdefault(edge, face_index)
            union(previous, face_index)
    return len({find(index) for index in range(len(triangles))})


def _face_components(triangles: np.ndarray) -> list[np.ndarray]:
    """Return deterministic edge-connected face-index components."""

    if len(triangles) == 0:
        return []
    import trimesh

    edges = np.vstack(
        (
            triangles[:, [0, 1]],
            triangles[:, [1, 2]],
            triangles[:, [2, 0]],
        )
    )
    edges.sort(axis=1)
    owners = np.tile(np.arange(len(triangles), dtype=np.int64), 3)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    sorted_edges = edges[order]
    sorted_owners = owners[order]
    shared = np.all(sorted_edges[1:] == sorted_edges[:-1], axis=1)
    adjacency = np.column_stack((sorted_owners[:-1][shared], sorted_owners[1:][shared]))
    adjacency = adjacency[adjacency[:, 0] != adjacency[:, 1]]
    components = trimesh.graph.connected_components(
        adjacency,
        nodes=np.arange(len(triangles), dtype=np.int64),
        engine="scipy",
    )
    normalized = [np.sort(np.asarray(component, dtype=np.int64)) for component in components]
    return sorted(normalized, key=lambda component: int(component[0]))


def _boundary_components(
    edge_occurrences: dict[tuple[int, int], list[tuple[int, int]]],
) -> tuple[int, int]:
    boundary_edges = [edge for edge, owners in edge_occurrences.items() if len(owners) == 1]
    if not boundary_edges:
        return 0, 0
    adjacency: dict[int, set[int]] = {}
    for left, right in boundary_edges:
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)
    visited: set[int] = set()
    loops = 0
    chains = 0
    for seed in sorted(adjacency):
        if seed in visited:
            continue
        pending = [seed]
        component: set[int] = set()
        while pending:
            current = pending.pop()
            if current in component:
                continue
            component.add(current)
            pending.extend(adjacency[current] - component)
        visited.update(component)
        if component and all(len(adjacency[vertex]) == 2 for vertex in component):
            loops += 1
        else:
            chains += 1
    return loops, chains


def _triangle_quality(
    vertices: np.ndarray,
    triangles: np.ndarray,
    diagonal: float,
) -> tuple[int, float | None, float | None]:
    if not len(triangles):
        return 0, None, None
    coordinates = vertices[triangles]
    edges = np.stack(
        (
            coordinates[:, 1] - coordinates[:, 0],
            coordinates[:, 2] - coordinates[:, 1],
            coordinates[:, 0] - coordinates[:, 2],
        ),
        axis=1,
    )
    lengths = np.linalg.norm(edges, axis=2)
    areas = np.linalg.norm(np.cross(edges[:, 0], -edges[:, 2]), axis=1) * 0.5
    valid = np.isfinite(areas) & (areas > max(diagonal * diagonal * 1e-16, 1e-24))
    if not np.any(valid):
        return 0, None, None
    lengths = lengths[valid]
    areas = areas[valid]
    denominator = 4.0 * math.sqrt(3.0) * areas
    aspect = np.sum(lengths * lengths, axis=1) / denominator
    finite = aspect[np.isfinite(aspect)]
    if not len(finite):
        return len(triangles), None, None
    minimum_altitude = np.divide(
        2.0 * areas,
        np.maximum(lengths.max(axis=1), 1e-30),
    )
    needle = int(
        np.count_nonzero((aspect > 1000.0) | (minimum_altitude < max(diagonal * 1e-8, 1e-12)))
    )
    return needle, float(np.percentile(finite, 95.0)), float(np.max(finite))


def _component_metrics(
    vertices: np.ndarray,
    triangles: np.ndarray,
    face_areas: np.ndarray,
    total_area: float,
) -> tuple[int, int | None, float | None, int, int, str, int, str]:
    """Measure component size, shell orientation, and bounded nesting."""

    components = _face_components(triangles)
    if not components:
        return 0, None, None, 0, 0, "not_evaluated", 0, "not_evaluated"
    face_counts = [len(component) for component in components]
    area_ratios = [
        float(face_areas[component].sum() / total_area) if total_area > 0.0 else 0.0
        for component in components
    ]
    tiny_count = sum(
        ratio < 1e-4 or count <= 2 for ratio, count in zip(area_ratios, face_counts, strict=True)
    )
    inverted = 0
    inverted_status = "pass"
    nested = 0
    nested_status = "pass"
    component_meshes = []
    if len(components) > 256:
        return (
            len(components),
            min(face_counts),
            min(area_ratios),
            tiny_count,
            0,
            "not_evaluated",
            0,
            "not_evaluated",
        )
    try:
        import trimesh

        for component in components:
            faces = triangles[component]
            used, inverse = np.unique(faces.reshape(-1), return_inverse=True)
            part = trimesh.Trimesh(
                vertices=vertices[used],
                faces=inverse.reshape((-1, 3)),
                process=False,
            )
            if part.is_watertight:
                coordinates = np.asarray(part.vertices)[np.asarray(part.faces)]
                signed_volume = float(
                    np.einsum(
                        "ij,ij->i",
                        coordinates[:, 0],
                        np.cross(coordinates[:, 1], coordinates[:, 2]),
                    ).sum()
                    / 6.0
                )
                if math.isfinite(signed_volume) and signed_volume < 0.0:
                    inverted += 1
            component_meshes.append(part)
        closed = [mesh for mesh in component_meshes if mesh.is_watertight]
        for inner_index, inner in enumerate(closed):
            if not len(inner.vertices):
                continue
            vertex_indices = np.linspace(
                0,
                len(inner.vertices) - 1,
                num=min(16, len(inner.vertices)),
                dtype=np.int64,
            )
            probe = np.asarray(inner.vertices[vertex_indices], dtype=np.float64)
            for outer_index, outer in enumerate(closed):
                if inner_index == outer_index:
                    continue
                outer_min, outer_max = outer.bounds
                inner_min, inner_max = inner.bounds
                bbox_inside = bool(
                    np.all(inner_min > outer_min + 1e-12) and np.all(inner_max < outer_max - 1e-12)
                )
                if bbox_inside and bool(np.all(outer.contains(probe))):
                    nested += 1
                    break
    except Exception:
        inverted_status = "not_evaluated"
        nested_status = "not_evaluated"
        inverted = 0
        nested = 0
    if inverted:
        inverted_status = "fail"
    if nested:
        nested_status = "fail"
    return (
        len(components),
        min(face_counts),
        min(area_ratios),
        tiny_count,
        inverted,
        inverted_status,
        nested,
        nested_status,
    )


def _triangles_overlap_coplanar(
    left: np.ndarray,
    right: np.ndarray,
    tolerance: float,
) -> bool:
    left_normal = np.cross(left[1] - left[0], left[2] - left[0])
    right_normal = np.cross(right[1] - right[0], right[2] - right[0])
    left_length = float(np.linalg.norm(left_normal))
    right_length = float(np.linalg.norm(right_normal))
    if left_length <= tolerance or right_length <= tolerance:
        return False
    left_unit = left_normal / left_length
    right_unit = right_normal / right_length
    if abs(float(np.dot(left_unit, right_unit))) < 1.0 - 1e-8:
        return False
    if float(np.max(np.abs((right - left[0]) @ left_unit))) > tolerance:
        return False
    drop_axis = int(np.argmax(np.abs(left_unit)))
    left_2d = np.delete(left, drop_axis, axis=1)
    right_2d = np.delete(right, drop_axis, axis=1)
    axes = []
    for triangle in (left_2d, right_2d):
        edge_vectors = np.roll(triangle, -1, axis=0) - triangle
        axes.extend(np.column_stack((-edge_vectors[:, 1], edge_vectors[:, 0])))
    for axis in axes:
        length = float(np.linalg.norm(axis))
        if length <= tolerance:
            continue
        direction = axis / length
        left_projection = left_2d @ direction
        right_projection = right_2d @ direction
        overlap = min(float(left_projection.max()), float(right_projection.max())) - max(
            float(left_projection.min()), float(right_projection.min())
        )
        if overlap <= tolerance:
            return False
    return True


def _coplanar_overlaps(
    vertices: np.ndarray,
    triangles: np.ndarray,
    diagonal: float,
    *,
    pair_limit: int = 10_000,
    triangle_limit: int = 25_000,
) -> tuple[str, int]:
    import trimesh

    if not len(triangles):
        return "not_evaluated", 0
    if len(triangles) > triangle_limit:
        return "not_evaluated", 0
    coordinates = vertices[triangles]
    tolerance = max(diagonal * 1e-10, 1e-12)
    minimum = coordinates.min(axis=1)
    maximum = coordinates.max(axis=1)
    bounds = np.column_stack((minimum - tolerance, maximum + tolerance))
    try:
        tree = trimesh.util.bounds_tree(bounds)
    except Exception:
        return "not_evaluated", 0
    count = 0
    tested = 0
    for index in range(len(triangles)):
        for other in sorted(int(item) for item in tree.intersection(bounds[index])):
            if other <= index:
                continue
            if not {int(value) for value in triangles[index]}.isdisjoint(
                int(value) for value in triangles[other]
            ):
                continue
            tested += 1
            if tested > pair_limit:
                return "not_evaluated", count
            if _triangles_overlap_coplanar(coordinates[index], coordinates[other], tolerance):
                count += 1
    return ("fail" if count else "pass"), count


def _non_manifold_vertex_count(triangles: np.ndarray) -> int:
    """Count bow-tie vertices whose incident faces form multiple edge fans."""

    incident: dict[int, list[int]] = {}
    for face_index, face in enumerate(triangles):
        if len({int(value) for value in face}) != 3:
            continue
        for vertex in face:
            incident.setdefault(int(vertex), []).append(face_index)
    non_manifold = 0
    for vertex, face_indices in incident.items():
        if len(face_indices) <= 1:
            continue
        neighbors = {face_index: set() for face_index in face_indices}
        edge_owners: dict[int, int] = {}
        for face_index in face_indices:
            for other in triangles[face_index]:
                other_vertex = int(other)
                if other_vertex == vertex:
                    continue
                previous = edge_owners.setdefault(other_vertex, face_index)
                neighbors[previous].add(face_index)
                neighbors[face_index].add(previous)
        visited: set[int] = set()
        fan_count = 0
        for face_index in face_indices:
            if face_index in visited:
                continue
            fan_count += 1
            pending = [face_index]
            while pending:
                current = pending.pop()
                if current in visited:
                    continue
                visited.add(current)
                pending.extend(neighbors[current] - visited)
        if fan_count > 1:
            non_manifold += 1
    return non_manifold


def _duplicate_vertex_count(vertices: np.ndarray, diagonal: float) -> int:
    if len(vertices) == 0:
        return 0
    tolerance = max(diagonal * 1e-9, 1e-12)
    quantized = np.round(vertices / tolerance).astype(np.int64)
    unique_count = len(np.unique(quantized, axis=0))
    return max(0, len(vertices) - unique_count)


def _near_duplicate_vertex_count(vertices: np.ndarray, diagonal: float) -> int:
    if len(vertices) == 0 or not np.isfinite(vertices).all():
        return 0
    exact = _duplicate_vertex_count(vertices, diagonal)
    tolerance = max(diagonal * 1e-7, 1e-10)
    quantized = np.round(vertices / tolerance).astype(np.int64)
    near = max(0, len(vertices) - len(np.unique(quantized, axis=0)))
    return max(0, near - exact)


def _topology_vertex_ids(vertices: np.ndarray, diagonal: float) -> np.ndarray:
    """Map coincident positions to analysis-only IDs without mutating the mesh."""

    if not len(vertices) or not np.isfinite(vertices).all():
        return np.arange(len(vertices), dtype=np.int64)
    tolerance = max(diagonal * 1e-9, 1e-12)
    quantized = np.round(vertices / tolerance).astype(np.int64)
    _, inverse = np.unique(quantized, axis=0, return_inverse=True)
    return inverse.astype(np.int64)


def positional_topology_mesh(
    vertices: np.ndarray,
    triangles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a seam-welded derived mesh for analysis or collision generation.

    This never mutates source or render geometry. It only joins vertices that are
    coincident within the same scale-relative tolerance used by diagnosis, then
    removes collapsed and exact duplicate triangles.
    """

    source_vertices = np.asarray(vertices, dtype=np.float64).reshape((-1, 3))
    source_triangles = np.asarray(triangles, dtype=np.int64).reshape((-1, 3))
    if not len(source_vertices) or not len(source_triangles):
        return source_vertices.copy(), source_triangles.copy()
    if not np.isfinite(source_vertices).all():
        raise ValueError("positional topology input contains non-finite vertices")
    valid = np.all((source_triangles >= 0) & (source_triangles < len(source_vertices)), axis=1)
    source_triangles = source_triangles[valid]
    if not len(source_triangles):
        return source_vertices.copy(), source_triangles

    diagonal = float(np.linalg.norm(source_vertices.max(axis=0) - source_vertices.min(axis=0)))
    topology_ids = _topology_vertex_ids(source_vertices, diagonal)
    vertex_count = int(topology_ids.max()) + 1
    welded_vertices = np.zeros((vertex_count, 3), dtype=np.float64)
    np.add.at(welded_vertices, topology_ids, source_vertices)
    counts = np.bincount(topology_ids, minlength=vertex_count)
    welded_vertices /= counts[:, None]

    welded_triangles = topology_ids[source_triangles]
    nondegenerate = (
        (welded_triangles[:, 0] != welded_triangles[:, 1])
        & (welded_triangles[:, 1] != welded_triangles[:, 2])
        & (welded_triangles[:, 2] != welded_triangles[:, 0])
    )
    welded_triangles = welded_triangles[nondegenerate]
    if not len(welded_triangles):
        return welded_vertices, welded_triangles
    face_keys = np.sort(welded_triangles, axis=1)
    _, first_indices = np.unique(face_keys, axis=0, return_index=True)
    return welded_vertices, welded_triangles[np.sort(first_indices)]


def _triangles_intersect(left: np.ndarray, right: np.ndarray, tolerance: float) -> bool:
    left_edges = np.roll(left, -1, axis=0) - left
    right_edges = np.roll(right, -1, axis=0) - right
    left_normal = np.cross(left_edges[0], left_edges[1])
    right_normal = np.cross(right_edges[0], right_edges[1])
    axes = np.asarray(
        [
            left_normal,
            right_normal,
            *(np.cross(a, b) for a in left_edges for b in right_edges),
            *(np.cross(left_normal, edge) for edge in left_edges),
            *(np.cross(right_normal, edge) for edge in right_edges),
        ],
        dtype=np.float64,
    )
    lengths = np.linalg.norm(axes, axis=1)
    axes = axes[lengths > tolerance]
    lengths = lengths[lengths > tolerance]
    if not len(axes):
        return False
    normalized = axes / lengths[:, None]
    left_projection = left @ normalized.T
    right_projection = right @ normalized.T
    separated = (left_projection.max(axis=0) < right_projection.min(axis=0) - tolerance) | (
        right_projection.max(axis=0) < left_projection.min(axis=0) - tolerance
    )
    return not bool(np.any(separated))


def _self_intersections(
    vertices: np.ndarray,
    triangles: np.ndarray,
    diagonal: float,
    *,
    triangle_limit: int = 25_000,
    broad_pair_limit: int = 100_000,
    candidate_limit: int = 10_000,
) -> tuple[str, int, int, int, str | None]:
    import trimesh

    if not len(triangles):
        return "not_evaluated", 0, 0, 0, "mesh has no valid triangles"
    if len(triangles) > triangle_limit:
        return (
            "not_evaluated",
            0,
            0,
            0,
            f"triangle count {len(triangles)} exceeds exact-check limit {triangle_limit}",
        )
    coordinates = vertices[triangles]
    if not np.isfinite(coordinates).all():
        return "not_evaluated", 0, 0, 0, "mesh contains non-finite triangle coordinates"
    minimum = coordinates.min(axis=1)
    maximum = coordinates.max(axis=1)
    tolerance = max(diagonal * 1e-10, 1e-12)
    bounds = np.column_stack((minimum - tolerance, maximum + tolerance))
    try:
        tree = trimesh.util.bounds_tree(bounds)
    except Exception as exc:
        return (
            "not_evaluated",
            0,
            0,
            0,
            f"self-intersection R-tree construction failed: {type(exc).__name__}: {exc}",
        )
    intersections = 0
    broad_pairs = 0
    candidate_pairs = 0
    quantized_coordinates = np.round(coordinates / tolerance).astype(np.int64)
    vertex_sets = [
        frozenset(tuple(int(value) for value in point) for point in triangle)
        for triangle in quantized_coordinates
    ]
    for index in range(len(triangles)):
        for other in sorted(int(item) for item in tree.intersection(bounds[index])):
            if other <= index:
                continue
            if not vertex_sets[index].isdisjoint(vertex_sets[other]):
                continue
            broad_pairs += 1
            if broad_pairs > broad_pair_limit:
                return (
                    "not_evaluated",
                    intersections,
                    broad_pairs,
                    candidate_pairs,
                    f"non-adjacent broad-phase pair count exceeds exact-check limit {broad_pair_limit}",
                )
            candidate_pairs += 1
            if candidate_pairs > candidate_limit:
                return (
                    "not_evaluated",
                    intersections,
                    broad_pairs,
                    candidate_pairs,
                    f"candidate pair count exceeds exact-check limit {candidate_limit}",
                )
            if _triangles_intersect(coordinates[index], coordinates[other], tolerance):
                intersections += 1
    return (
        "fail" if intersections else "pass",
        intersections,
        broad_pairs,
        candidate_pairs,
        None,
    )


def measure_mesh(
    mesh: MeshData,
    *,
    exact_pair_limit: int = 10_000,
) -> MeshMetrics:
    """Measure source indices and analysis-only positional seam connectivity."""

    vertices = mesh.world_vertices_m
    triangles = mesh.triangles
    finite_mask = np.isfinite(vertices).all(axis=1) if len(vertices) else np.array([], dtype=bool)
    non_finite = int(len(vertices) - np.count_nonzero(finite_mask))
    valid_triangles = (
        triangles[np.all((triangles >= 0) & (triangles < len(vertices)), axis=1)]
        if len(triangles)
        else triangles
    )
    invalid_index_count = int((len(triangles) - len(valid_triangles)) * 3)

    if len(vertices) and np.any(finite_mask):
        finite_vertices = vertices[finite_mask]
        bbox_min = finite_vertices.min(axis=0)
        bbox_max = finite_vertices.max(axis=0)
        diagonal = float(np.linalg.norm(bbox_max - bbox_min))
    else:
        bbox_min = bbox_max = None
        diagonal = 0.0

    topology_ids = _topology_vertex_ids(vertices, diagonal)
    topology_triangles = topology_ids[valid_triangles] if len(valid_triangles) else valid_triangles
    edge_occurrences: dict[tuple[int, int], list[tuple[int, int]]] = {}
    indexed_edge_counts: dict[tuple[int, int], int] = {}
    duplicate_faces = 0
    seen_faces: set[tuple[int, int, int]] = set()
    degenerate = 0
    surface_area = 0.0
    signed_volume = 0.0
    face_areas = np.zeros(len(valid_triangles), dtype=np.float64)
    area_tolerance = max(diagonal * diagonal * 1e-16, 1e-24)
    for face_index, (face, topology_face) in enumerate(
        zip(
            valid_triangles,
            topology_triangles,
            strict=True,
        )
    ):
        key = tuple(sorted(int(item) for item in topology_face))
        if key in seen_faces:
            duplicate_faces += 1
        seen_faces.add(key)
        a, b, c = vertices[face]
        cross = np.cross(b - a, c - a)
        area = float(np.linalg.norm(cross) * 0.5)
        if not math.isfinite(area) or area <= area_tolerance:
            degenerate += 1
        else:
            face_areas[face_index] = area
            surface_area += area
            signed_volume += float(np.dot(a, np.cross(b, c)) / 6.0)
        for start, end in zip(
            topology_face,
            np.roll(topology_face, -1),
            strict=True,
        ):
            directed = (int(start), int(end))
            edge = tuple(sorted(directed))
            edge_occurrences.setdefault(edge, []).append(directed)
        for start, end in zip(face, np.roll(face, -1), strict=True):
            edge = tuple(sorted((int(start), int(end))))
            indexed_edge_counts[edge] = indexed_edge_counts.get(edge, 0) + 1

    indexed_boundary = sum(count == 1 for count in indexed_edge_counts.values())
    boundary = sum(len(items) == 1 for items in edge_occurrences.values())
    boundary_loops, boundary_chains = _boundary_components(edge_occurrences)
    over_connected = sum(len(items) > 2 for items in edge_occurrences.values())
    non_manifold_vertices = _non_manifold_vertex_count(topology_triangles)
    inconsistent = 0
    for items in edge_occurrences.values():
        if len(items) != 2:
            continue
        if items[0] == items[1]:
            inconsistent += 1
    watertight = (
        bool(valid_triangles.size)
        and boundary == 0
        and over_connected == 0
        and non_manifold_vertices == 0
    )
    volume = abs(signed_volume) if watertight and degenerate == 0 else None
    needle_count, aspect_p95, aspect_max = _triangle_quality(
        vertices,
        valid_triangles,
        diagonal,
    )
    analysis_vertices, analysis_triangles = positional_topology_mesh(
        vertices,
        valid_triangles,
    )
    analysis_face_areas = (
        np.linalg.norm(
            np.cross(
                analysis_vertices[analysis_triangles[:, 1]]
                - analysis_vertices[analysis_triangles[:, 0]],
                analysis_vertices[analysis_triangles[:, 2]]
                - analysis_vertices[analysis_triangles[:, 0]],
            ),
            axis=1,
        )
        * 0.5
        if len(analysis_triangles)
        else np.empty((0,), dtype=np.float64)
    )
    (
        component_count,
        smallest_component_faces,
        smallest_component_area_ratio,
        tiny_component_count,
        inverted_shell_count,
        inverted_shell_status,
        nested_shell_count,
        nested_shell_status,
    ) = _component_metrics(
        analysis_vertices,
        analysis_triangles,
        analysis_face_areas,
        float(analysis_face_areas.sum()),
    )
    coplanar_overlap_status, coplanar_overlap_count = _coplanar_overlaps(
        vertices,
        valid_triangles,
        diagonal,
        pair_limit=exact_pair_limit,
    )
    (
        self_intersection_status,
        self_intersection_count,
        self_intersection_broad_phase_pairs,
        self_intersection_candidate_pairs,
        self_intersection_reason,
    ) = _self_intersections(
        vertices,
        valid_triangles,
        diagonal,
        broad_pair_limit=max(exact_pair_limit * 4, exact_pair_limit),
        candidate_limit=exact_pair_limit,
    )
    _, source_invalid, non_triangular = _triangulate(
        mesh.source_face_counts, mesh.source_face_indices, len(vertices)
    )
    used_vertices = np.unique(valid_triangles.reshape(-1)) if len(valid_triangles) else np.array([])
    vertex_count_for_euler = (
        len(np.unique(analysis_triangles.reshape(-1))) if len(analysis_triangles) else 0
    )
    analysis_edges = {
        tuple(sorted((int(start), int(end))))
        for face in analysis_triangles
        for start, end in zip(face, np.roll(face, -1), strict=True)
    }
    edge_count_for_euler = len(analysis_edges)
    euler = (
        int(vertex_count_for_euler - edge_count_for_euler + len(analysis_triangles))
        if len(analysis_triangles)
        else None
    )
    genus = None
    if watertight and euler is not None:
        candidate_genus = (2.0 * component_count - float(euler)) / 2.0
        if candidate_genus >= -1e-9:
            genus = max(0.0, candidate_genus)
    finite_vertices = vertices[np.all(np.isfinite(vertices), axis=1)]
    geometric_dimension: int | None = None
    minimum_bbox_extent = None
    if len(finite_vertices):
        extents = np.ptp(finite_vertices, axis=0)
        minimum_bbox_extent = float(np.min(extents))
        centered = finite_vertices - np.mean(finite_vertices, axis=0)
        singular_values = np.linalg.svd(centered, compute_uv=False)
        rank_tolerance = max(diagonal * 1e-9, np.finfo(float).eps)
        geometric_dimension = int(np.count_nonzero(singular_values > rank_tolerance))
    volume_tolerance = diagonal**3 * 1e-12
    zero_thickness_status = (
        "fail"
        if geometric_dimension is not None
        and geometric_dimension < 3
        and (bool(watertight) or indexed_boundary == 0)
        else "fail"
        if bool(watertight) and volume is not None and abs(volume) <= volume_tolerance
        else "not_evaluated"
        if geometric_dimension is not None and geometric_dimension < 3
        else "pass"
        if geometric_dimension is not None
        else "not_evaluated"
    )
    return MeshMetrics(
        mesh_count=1,
        point_count=len(vertices),
        face_count=len(mesh.source_face_counts),
        triangle_count=len(valid_triangles),
        invalid_index_count=invalid_index_count + source_invalid,
        non_finite_vertex_count=non_finite,
        degenerate_face_count=degenerate,
        duplicate_face_count=duplicate_faces,
        duplicate_vertex_count=_duplicate_vertex_count(vertices, diagonal),
        near_duplicate_vertex_count=_near_duplicate_vertex_count(vertices, diagonal),
        unused_vertex_count=max(0, len(vertices) - len(used_vertices)),
        indexed_boundary_edge_count=indexed_boundary,
        boundary_edge_count=boundary,
        boundary_loop_count=boundary_loops,
        open_boundary_chain_count=boundary_chains,
        over_connected_edge_count=over_connected,
        non_manifold_vertex_count=non_manifold_vertices,
        inconsistent_orientation_edge_count=inconsistent,
        non_triangular_face_count=non_triangular,
        needle_triangle_count=needle_count,
        triangle_aspect_ratio_p95=aspect_p95,
        triangle_aspect_ratio_max=aspect_max,
        connected_component_count=component_count,
        smallest_component_face_count=smallest_component_faces,
        smallest_component_area_ratio=smallest_component_area_ratio,
        tiny_component_count=tiny_component_count,
        inverted_shell_count=inverted_shell_count,
        inverted_shell_status=inverted_shell_status,
        nested_shell_count=nested_shell_count,
        nested_shell_status=nested_shell_status,
        coplanar_overlap_count=coplanar_overlap_count,
        coplanar_overlap_status=coplanar_overlap_status,
        material_subset_count=mesh.material_subset_count,
        authored_normal_count=mesh.authored_normal_count,
        non_finite_normal_count=mesh.non_finite_normal_count,
        authored_uv_count=mesh.authored_uv_count,
        non_finite_uv_count=mesh.non_finite_uv_count,
        non_finite_transform_count=int(mesh.transform_non_finite),
        singular_transform_count=int(mesh.transform_singular),
        non_uniform_transform_count=int(mesh.transform_non_uniform),
        sheared_transform_count=int(mesh.transform_sheared),
        reflected_transform_count=int(mesh.transform_reflected),
        surface_area_m2=surface_area,
        enclosed_volume_m3=volume,
        bbox_min_m=bbox_min.tolist() if bbox_min is not None else None,
        bbox_max_m=bbox_max.tolist() if bbox_max is not None else None,
        bbox_diagonal_m=diagonal,
        minimum_bbox_extent_m=minimum_bbox_extent,
        geometric_dimension=geometric_dimension,
        zero_thickness_status=zero_thickness_status,
        watertight=watertight,
        euler_characteristic=euler,
        genus=genus,
        self_intersection_status=self_intersection_status,
        self_intersection_count=self_intersection_count,
        self_intersection_broad_phase_pairs=self_intersection_broad_phase_pairs,
        self_intersection_candidate_pairs=self_intersection_candidate_pairs,
        self_intersection_reason=self_intersection_reason,
    )


def _sum_optional(values: list[float | None]) -> float | None:
    return (
        sum(float(value) for value in values)
        if all(value is not None for value in values)
        else None
    )


def _inter_part_intersections(
    meshes: list[MeshData],
    *,
    pair_limit: int = 1_000,
) -> tuple[str, int, str | None]:
    """Bounded triangle intersection check across distinct source mesh paths."""

    import trimesh

    if len(meshes) < 2:
        return "pass", 0, None
    tested = 0
    intersections = 0
    for left_index, left in enumerate(meshes):
        left_triangles = left.triangles[
            np.all(
                (left.triangles >= 0) & (left.triangles < len(left.world_vertices_m)),
                axis=1,
            )
        ]
        if not len(left_triangles):
            continue
        left_coordinates = left.world_vertices_m[left_triangles]
        left_min = left_coordinates.min(axis=1)
        left_max = left_coordinates.max(axis=1)
        for right in meshes[left_index + 1 :]:
            right_triangles = right.triangles[
                np.all(
                    (right.triangles >= 0) & (right.triangles < len(right.world_vertices_m)),
                    axis=1,
                )
            ]
            if not len(right_triangles):
                continue
            right_coordinates = right.world_vertices_m[right_triangles]
            right_min = right_coordinates.min(axis=1)
            right_max = right_coordinates.max(axis=1)
            combined = np.vstack((left.world_vertices_m, right.world_vertices_m))
            finite = combined[np.isfinite(combined).all(axis=1)]
            diagonal = (
                float(np.linalg.norm(finite.max(axis=0) - finite.min(axis=0)))
                if len(finite)
                else 0.0
            )
            tolerance = max(diagonal * 1e-10, 1e-12)
            if np.any(left_max.max(axis=0) < right_min.min(axis=0) - tolerance) or np.any(
                right_max.max(axis=0) < left_min.min(axis=0) - tolerance
            ):
                continue
            right_bounds = np.column_stack((right_min - tolerance, right_max + tolerance))
            try:
                right_tree = trimesh.util.bounds_tree(right_bounds)
            except Exception as exc:
                return (
                    "not_evaluated",
                    intersections,
                    f"inter-part R-tree construction failed: {type(exc).__name__}: {exc}",
                )
            for triangle_index, triangle in enumerate(left_coordinates):
                query = np.concatenate(
                    (
                        left_min[triangle_index] - tolerance,
                        left_max[triangle_index] + tolerance,
                    )
                )
                candidates = sorted(int(candidate) for candidate in right_tree.intersection(query))
                for candidate in candidates:
                    tested += 1
                    if tested > pair_limit:
                        return (
                            "not_evaluated",
                            intersections,
                            f"inter-part candidate pair count exceeds limit {pair_limit}",
                        )
                    if _triangles_intersect(
                        triangle,
                        right_coordinates[candidate],
                        tolerance,
                    ):
                        intersections += 1
    return ("fail" if intersections else "pass"), intersections, None


def measure_asset(
    path: str | Path,
    *,
    unresolved_dependency_count: int = 0,
    unresolved_geometry_dependency_count: int = 0,
    unresolved_material_dependency_count: int = 0,
) -> GeometryMetrics:
    """Measure an asset while preserving per-mesh identity and aggregate facts."""

    meshes, stage = load_meshes(path)
    records: list[MeshRecord] = []
    # Exact Python triangle predicates are bounded per asset, not merely per
    # mesh. Give a single dense mesh enough candidates to exclude spatially
    # adjacent-but-topologically-disjoint triangles while keeping assemblies
    # under one deterministic 50k-pair aggregate ceiling.
    mesh_count = max(len(meshes), 1)
    per_mesh_ceiling = max(50, 50_000 // mesh_count)
    largest_triangle_count = max((len(mesh.triangles) for mesh in meshes), default=0)
    exact_pair_limit = min(
        per_mesh_ceiling,
        max(4_000, 2 * largest_triangle_count),
    )
    metrics = [measure_mesh(mesh, exact_pair_limit=exact_pair_limit) for mesh in meshes]
    for mesh, measured in zip(meshes, metrics, strict=True):
        failures = (
            measured.invalid_index_count
            + measured.non_finite_vertex_count
            + measured.degenerate_face_count
            + measured.duplicate_face_count
            + measured.over_connected_edge_count
            + measured.non_manifold_vertex_count
            + measured.inconsistent_orientation_edge_count
            + measured.self_intersection_count
        )
        warning = (
            measured.boundary_edge_count
            or measured.non_triangular_face_count
            or measured.self_intersection_status == "not_evaluated"
        )
        records.append(
            MeshRecord(
                path=mesh.path,
                purpose=mesh.purpose,
                role=mesh.role,
                is_instance_proxy=mesh.is_instance_proxy,
                metrics=measured,
                topology_status="fail" if failures else "warning" if warning else "pass",
                has_face_material_subsets=mesh.material_subset_count > 0,
                has_authored_normals=mesh.authored_normal_count > 0,
                has_authored_uvs=mesh.authored_uv_count > 0,
                has_face_varying_data=mesh.has_face_varying_data,
            )
        )
    aggregate = MeshMetrics(
        mesh_count=len(metrics),
        point_count=sum(item.point_count for item in metrics),
        face_count=sum(item.face_count for item in metrics),
        triangle_count=sum(item.triangle_count for item in metrics),
        invalid_index_count=sum(item.invalid_index_count for item in metrics),
        non_finite_vertex_count=sum(item.non_finite_vertex_count for item in metrics),
        degenerate_face_count=sum(item.degenerate_face_count for item in metrics),
        duplicate_face_count=sum(item.duplicate_face_count for item in metrics),
        duplicate_vertex_count=sum(item.duplicate_vertex_count for item in metrics),
        near_duplicate_vertex_count=sum(item.near_duplicate_vertex_count for item in metrics),
        unused_vertex_count=sum(item.unused_vertex_count for item in metrics),
        indexed_boundary_edge_count=sum(item.indexed_boundary_edge_count for item in metrics),
        boundary_edge_count=sum(item.boundary_edge_count for item in metrics),
        boundary_loop_count=sum(item.boundary_loop_count for item in metrics),
        open_boundary_chain_count=sum(item.open_boundary_chain_count for item in metrics),
        over_connected_edge_count=sum(item.over_connected_edge_count for item in metrics),
        non_manifold_vertex_count=sum(item.non_manifold_vertex_count for item in metrics),
        inconsistent_orientation_edge_count=sum(
            item.inconsistent_orientation_edge_count for item in metrics
        ),
        non_triangular_face_count=sum(item.non_triangular_face_count for item in metrics),
        needle_triangle_count=sum(item.needle_triangle_count for item in metrics),
        triangle_aspect_ratio_p95=max(
            (item.triangle_aspect_ratio_p95 for item in metrics if item.triangle_aspect_ratio_p95),
            default=None,
        ),
        triangle_aspect_ratio_max=max(
            (item.triangle_aspect_ratio_max for item in metrics if item.triangle_aspect_ratio_max),
            default=None,
        ),
        connected_component_count=sum(item.connected_component_count for item in metrics),
        smallest_component_face_count=min(
            (
                item.smallest_component_face_count
                for item in metrics
                if item.smallest_component_face_count is not None
            ),
            default=None,
        ),
        smallest_component_area_ratio=min(
            (
                item.smallest_component_area_ratio
                for item in metrics
                if item.smallest_component_area_ratio is not None
            ),
            default=None,
        ),
        tiny_component_count=sum(item.tiny_component_count for item in metrics),
        inverted_shell_count=sum(item.inverted_shell_count for item in metrics),
        inverted_shell_status=(
            "fail"
            if any(item.inverted_shell_status == "fail" for item in metrics)
            else "not_evaluated"
            if any(item.inverted_shell_status == "not_evaluated" for item in metrics)
            else "pass"
        ),
        nested_shell_count=sum(item.nested_shell_count for item in metrics),
        nested_shell_status=(
            "fail"
            if any(item.nested_shell_status == "fail" for item in metrics)
            else "not_evaluated"
            if any(item.nested_shell_status == "not_evaluated" for item in metrics)
            else "pass"
        ),
        coplanar_overlap_count=sum(item.coplanar_overlap_count for item in metrics),
        coplanar_overlap_status=(
            "fail"
            if any(item.coplanar_overlap_status == "fail" for item in metrics)
            else "not_evaluated"
            if any(item.coplanar_overlap_status == "not_evaluated" for item in metrics)
            else "pass"
        ),
        material_subset_count=sum(item.material_subset_count for item in metrics),
        authored_normal_count=sum(item.authored_normal_count for item in metrics),
        non_finite_normal_count=sum(item.non_finite_normal_count for item in metrics),
        authored_uv_count=sum(item.authored_uv_count for item in metrics),
        non_finite_uv_count=sum(item.non_finite_uv_count for item in metrics),
        non_finite_transform_count=sum(item.non_finite_transform_count for item in metrics),
        singular_transform_count=sum(item.singular_transform_count for item in metrics),
        non_uniform_transform_count=sum(item.non_uniform_transform_count for item in metrics),
        sheared_transform_count=sum(item.sheared_transform_count for item in metrics),
        reflected_transform_count=sum(item.reflected_transform_count for item in metrics),
        minimum_bbox_extent_m=min(
            (
                item.minimum_bbox_extent_m
                for item in metrics
                if item.minimum_bbox_extent_m is not None
            ),
            default=None,
        ),
        geometric_dimension=min(
            (item.geometric_dimension for item in metrics if item.geometric_dimension is not None),
            default=None,
        ),
        zero_thickness_status=(
            "fail"
            if any(item.zero_thickness_status == "fail" for item in metrics)
            else "not_evaluated"
            if any(item.zero_thickness_status == "not_evaluated" for item in metrics)
            else "pass"
        ),
        surface_area_m2=_sum_optional([item.surface_area_m2 for item in metrics]),
        enclosed_volume_m3=_sum_optional([item.enclosed_volume_m3 for item in metrics]),
        watertight=all(item.watertight is True for item in metrics) if metrics else None,
        euler_characteristic=(
            sum(int(item.euler_characteristic) for item in metrics)
            if metrics and all(item.euler_characteristic is not None for item in metrics)
            else None
        ),
        genus=(
            sum(float(item.genus) for item in metrics)
            if metrics and all(item.genus is not None for item in metrics)
            else None
        ),
        self_intersection_status=(
            "fail"
            if any(item.self_intersection_status == "fail" for item in metrics)
            else "not_evaluated"
            if any(item.self_intersection_status == "not_evaluated" for item in metrics)
            else "pass"
        ),
        self_intersection_count=sum(item.self_intersection_count for item in metrics),
        self_intersection_broad_phase_pairs=sum(
            item.self_intersection_broad_phase_pairs for item in metrics
        ),
        self_intersection_candidate_pairs=sum(
            item.self_intersection_candidate_pairs for item in metrics
        ),
        self_intersection_reason="; ".join(
            item.self_intersection_reason for item in metrics if item.self_intersection_reason
        )
        or None,
    )
    bounds_min = [item.bbox_min_m for item in metrics if item.bbox_min_m is not None]
    bounds_max = [item.bbox_max_m for item in metrics if item.bbox_max_m is not None]
    if bounds_min and bounds_max:
        aggregate.bbox_min_m = np.min(np.asarray(bounds_min), axis=0).tolist()
        aggregate.bbox_max_m = np.max(np.asarray(bounds_max), axis=0).tolist()
        aggregate.bbox_diagonal_m = float(
            np.linalg.norm(np.asarray(aggregate.bbox_max_m) - np.asarray(aggregate.bbox_min_m))
        )
    inter_part_status, inter_part_count, inter_part_reason = _inter_part_intersections(meshes)
    return GeometryMetrics(
        source_format=str(stage["source_format"]),
        up_axis=stage.get("up_axis"),
        meters_per_unit=stage.get("meters_per_unit"),
        root_count=int(stage.get("root_count") or 0),
        default_prim_path=stage.get("default_prim_path"),
        mesh=aggregate,
        meshes=records,
        source_part_paths=[mesh.path for mesh in meshes],
        source_collision_paths=list(stage.get("source_collision_paths") or []),
        source_helper_paths=list(stage.get("source_helper_paths") or []),
        collision_source_mesh_count=len(stage.get("source_collision_paths") or []),
        helper_source_mesh_count=len(stage.get("source_helper_paths") or []),
        instance_proxy_mesh_count=sum(mesh.is_instance_proxy for mesh in meshes),
        material_binding_count=int(stage.get("material_binding_count") or 0),
        unresolved_dependency_count=unresolved_dependency_count,
        unresolved_geometry_dependency_count=unresolved_geometry_dependency_count,
        unresolved_material_dependency_count=unresolved_material_dependency_count,
        self_intersection_status=aggregate.self_intersection_status,
        inter_part_intersection_status=inter_part_status,
        inter_part_intersection_count=inter_part_count,
        inter_part_intersection_reason=inter_part_reason,
    )


def audit_asset_intersections(
    path: str | Path,
    *,
    output_path: str | Path | None = None,
    budget: Any | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Run the scalable non-mutating audit over all render parts.

    The full report remains authoritative. Callers may project its result into
    legacy metrics, but must retain skipped and indeterminate status evidence.
    """

    from .artifacts import atomic_write_json
    from .scalable_audit import ScalableAuditBudget, audit_triangle_mesh

    meshes, _stage = load_meshes(path)
    vertices: list[np.ndarray] = []
    triangles: list[np.ndarray] = []
    part_ids: list[str] = []
    ranges: list[dict[str, Any]] = []
    vertex_offset = 0
    triangle_offset = 0
    for mesh in meshes:
        mesh_vertices = np.asarray(mesh.world_vertices_m, dtype=np.float64)
        mesh_triangles = np.asarray(mesh.triangles, dtype=np.int64)
        vertices.append(mesh_vertices)
        triangles.append(mesh_triangles + vertex_offset)
        part_ids.extend([mesh.path] * len(mesh_triangles))
        ranges.append(
            {
                "path": mesh.path,
                "triangle_start": triangle_offset,
                "triangle_end": triangle_offset + len(mesh_triangles),
            }
        )
        vertex_offset += len(mesh_vertices)
        triangle_offset += len(mesh_triangles)
    combined_vertices = np.vstack(vertices) if vertices else np.empty((0, 3), dtype=np.float64)
    combined_triangles = np.vstack(triangles) if triangles else np.empty((0, 3), dtype=np.int64)
    selected_budget = budget if budget is not None else ScalableAuditBudget()
    report = audit_triangle_mesh(
        combined_vertices,
        combined_triangles,
        part_ids=part_ids,
        budget=selected_budget,
    )
    payload = {
        "schema_version": "geometry-repair.asset-scalable-audit.v1",
        "source_path": str(Path(path).expanduser().resolve()),
        "mesh_triangle_ranges": ranges,
        "audit": report.model_dump(mode="json"),
    }
    if output_path is not None:
        atomic_write_json(output_path, payload)
    return report, payload


def copy_usd_stage(source: Path, target: Path) -> Path:
    """Copy/export a USD stage to a mutable standalone layer."""

    target.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() != ".usdz" and target.suffix.lower() == source.suffix.lower():
        shutil.copy2(source, target)
        return target
    from pxr import Usd

    stage = Usd.Stage.Open(str(source))
    if stage is None:
        raise RuntimeError(f"Could not open USD source {source}")
    if not stage.GetRootLayer().Export(str(target)):
        raise RuntimeError(f"Could not export mutable USD copy to {target}")
    return target


def _canonicalize_flattened_prototypes(path: Path) -> None:
    """Give OpenUSD-generated flattened prototypes stable content-based ordering."""

    from pxr import Sdf, Usd

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError(f"Could not reopen flattened USD stage {path}")
    layer = stage.GetRootLayer()
    prototypes = [
        prim_spec
        for prim_spec in layer.rootPrims
        if _FLATTENED_PROTOTYPE_NAME.fullmatch(prim_spec.name)
    ]
    if not prototypes:
        return

    usage: dict[str, list[tuple[str, int]]] = {str(prim_spec.path): [] for prim_spec in prototypes}
    for prim in stage.TraverseAll():
        references = prim.GetMetadata("references")
        if references is None:
            continue
        for index, reference in enumerate(references.GetAppliedItems()):
            target = str(reference.primPath)
            if not reference.assetPath and target in usage:
                usage[target].append((str(prim.GetPath()), index))

    records: list[tuple[str, tuple[tuple[str, int], ...], str]] = []
    for prim_spec in prototypes:
        scratch = Sdf.Layer.CreateAnonymous(".usda")
        Sdf.CreatePrimInLayer(scratch, "/Prototype")
        if not Sdf.CopySpec(layer, prim_spec.path, scratch, "/Prototype"):
            raise RuntimeError(f"Could not fingerprint flattened prototype {prim_spec.path}")
        canonical_text = _FLATTENED_PROTOTYPE_TEXT.sub(
            "Flattened_Prototype",
            scratch.ExportToString(),
        )
        records.append(
            (
                hashlib.sha256(canonical_text.encode()).hexdigest(),
                tuple(sorted(usage[str(prim_spec.path)])),
                str(prim_spec.path),
            )
        )

    name_mapping: dict[str, str] = {}
    for index, (_digest, _usage, source_path) in enumerate(sorted(records)):
        name_mapping[source_path.removeprefix("/")] = f"Flattened_Prototype_canonical_{index:04d}"

    canonical_text = layer.ExportToString()
    temporary_names: dict[str, str] = {}
    for index, source_name in enumerate(sorted(name_mapping)):
        temporary_name = f"__GeometryRepairPrototypeTmp_{index:04d}"
        if temporary_name in canonical_text:
            raise RuntimeError(f"Prototype temporary name collision in {path}")
        pattern = rf"(?<![A-Za-z0-9_]){re.escape(source_name)}(?![A-Za-z0-9_])"
        canonical_text, replacement_count = re.subn(
            pattern,
            temporary_name,
            canonical_text,
        )
        if replacement_count == 0:
            raise RuntimeError(f"Could not locate flattened prototype {source_name} in {path}")
        temporary_names[temporary_name] = name_mapping[source_name]
    for temporary_name, target_name in temporary_names.items():
        canonical_text = canonical_text.replace(temporary_name, target_name)

    expected_names = set(name_mapping.values())
    actual_names = set(_FLATTENED_PROTOTYPE_TEXT.findall(canonical_text))
    if actual_names != expected_names:
        raise RuntimeError(
            f"Flattened prototype canonicalization mismatch in {path}: "
            f"expected {sorted(expected_names)}, found {sorted(actual_names)}"
        )

    transformed = Sdf.Layer.CreateAnonymous(".usda")
    if not transformed.ImportFromString(canonical_text):
        raise RuntimeError(f"Could not parse canonicalized flattened stage {path}")

    ordered = Sdf.Layer.CreateAnonymous(".usda")
    for key in transformed.pseudoRoot.ListInfoKeys():
        if key != "primOrder":
            ordered.pseudoRoot.SetInfo(key, transformed.pseudoRoot.GetInfo(key))
    for prim_spec in sorted(transformed.rootPrims, key=lambda item: item.name):
        if not Sdf.CopySpec(transformed, prim_spec.path, ordered, prim_spec.path):
            raise RuntimeError(f"Could not order flattened root prim {prim_spec.path}")
    layer.TransferContent(ordered)
    layer.Save()


def flatten_usd_stage(source: Path, target: Path) -> Path:
    """Export a self-contained composed USD layer for final handoff."""

    from pxr import Usd

    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.Open(str(source))
    if stage is None:
        raise RuntimeError(f"Could not open USD source {source}")
    flattened = stage.Flatten(addSourceFileComment=False)
    if not flattened.Export(str(target)):
        raise RuntimeError(f"Could not export flattened USD stage to {target}")
    _canonicalize_flattened_prototypes(target)
    return target


def separate_non_render_geometry(path: str | Path) -> dict[str, list[str]]:
    """Mark collision/helper meshes as non-rendering without deleting source identity."""

    from pxr import Usd, UsdGeom

    target = Path(path).expanduser().resolve()
    stage = Usd.Stage.Open(str(target))
    if stage is None:
        raise RuntimeError(f"Could not open final render stage {target}")
    separated: dict[str, list[str]] = {"collision": [], "helper": []}
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        imageable = UsdGeom.Imageable(prim)
        purpose = str(imageable.ComputePurpose())
        role = _usd_mesh_role(prim, purpose)
        if role == "render":
            continue
        imageable.CreatePurposeAttr().Set(UsdGeom.Tokens.guide)
        imageable.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
        prim.SetCustomDataByKey("geometryRepairSourceRole", role)
        separated[role].append(str(prim.GetPath()))
    layer_data = dict(stage.GetRootLayer().customLayerData or {})
    layer_data["geometryRepairSeparatedSourceCollisionCount"] = len(separated["collision"])
    layer_data["geometryRepairSeparatedSourceHelperCount"] = len(separated["helper"])
    stage.GetRootLayer().customLayerData = layer_data
    stage.GetRootLayer().Save()
    return separated


def update_usd_triangle_meshes(
    source: str | Path,
    target: str | Path,
    updates: dict[str, tuple[np.ndarray, np.ndarray]],
) -> Path:
    """Apply local-space point/triangle updates while preserving USD hierarchy."""

    from pxr import Usd, UsdGeom, Vt

    source_path = Path(source).resolve()
    target_path = Path(target).resolve()
    copy_usd_stage(source_path, target_path)
    stage = Usd.Stage.Open(str(target_path))
    if stage is None:
        raise RuntimeError(f"Could not open repair working copy {target_path}")
    for prim_path, (vertices, triangles) in updates.items():
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsA(UsdGeom.Mesh):
            raise RuntimeError(f"Repair target mesh no longer exists: {prim_path}")
        if prim.IsInstanceProxy():
            raise RuntimeError(
                f"Repair target is a read-only instance proxy: {prim_path}; "
                "run the typed Scene Optimizer deinstance prerequisite first"
            )
        mesh = UsdGeom.Mesh(prim)
        mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(vertices.astype(np.float32)))
        mesh.GetFaceVertexCountsAttr().Set(
            Vt.IntArray.FromNumpy(np.full(len(triangles), 3, dtype=np.int32))
        )
        mesh.GetFaceVertexIndicesAttr().Set(
            Vt.IntArray.FromNumpy(triangles.astype(np.int32).reshape(-1))
        )
        normals = mesh.GetNormalsAttr()
        if normals and normals.HasAuthoredValueOpinion():
            normals.Clear()
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
        mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    stage.GetRootLayer().Save()
    return target_path
