# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic SimReady package authoring for geometry-stage USD assets.

The authoring policy is intentionally explicit. Geometry workflows often emit
separate meshes for material zones, task references, and measurement helpers;
those meshes must not silently become independent rigid bodies or colliders.
This module groups physical members, keeps static and visual-only members
separate, and marks helper geometry with guide purpose before validation.
"""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SIMREADY_AUTHORING_SCHEMA = "cad-verifier/simready-authoring@1"


def _remove_failed_artifacts(paths: tuple[Path, ...]) -> list[str]:
    errors: list[str] = []
    for path in dict.fromkeys(paths):
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            errors.append(f"Could not remove failed SimReady artifact {path}: {exc}")
    return errors


def _safe_name(value: str) -> str:
    result = "".join(
        character if character.isalnum() or character == "_" else "_" for character in value
    )
    if result and not (result[0].isalpha() or result[0] == "_"):
        result = f"_{result}"
    return result or "unnamed"


def _require_unique_safe_names(values: list[str], *, field: str) -> None:
    by_safe_name: dict[str, str] = {}
    for value in values:
        safe_name = _safe_name(value)
        previous = by_safe_name.get(safe_name)
        if previous is not None:
            raise ValueError(
                f"SimReady policy {field} names {previous!r} and {value!r} both "
                f"resolve to USD prim name {safe_name!r}."
            )
        by_safe_name[safe_name] = value


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _normalize_member_names(values: Any, *, field: str) -> tuple[str, ...]:
    if values is None:
        return ()
    if not isinstance(values, list) or not all(isinstance(item, str) and item for item in values):
        raise ValueError(f"SimReady policy {field} must be a list of non-empty strings.")
    if len(values) != len(set(values)):
        raise ValueError(f"SimReady policy {field} contains duplicate members.")
    return tuple(values)


def _normalize_vector3(values: Any, *, field: str) -> tuple[float, float, float]:
    if values is None:
        return (0.0, 0.0, 0.0)
    if not isinstance(values, list) or len(values) != 3:
        raise ValueError(f"SimReady policy {field} must contain three numbers.")
    result = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"SimReady policy {field} values must be finite.")
    return result


def _body_pose_adjustment(value: Any, *, body_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"SimReady body {body_name} pose_adjustment must be an object.")
    translation = _normalize_vector3(
        value.get("translation"), field=f"body {body_name} pose_adjustment.translation"
    )
    rotation = _normalize_vector3(
        value.get("rotate_xyz_degrees"),
        field=f"body {body_name} pose_adjustment.rotate_xyz_degrees",
    )
    rationale = str(value.get("rationale") or "").strip()
    if (any(translation) or any(rotation)) and not rationale:
        raise ValueError(
            f"SimReady body {body_name} pose_adjustment requires a rationale when nonzero."
        )
    return {
        "translation": translation,
        "rotate_xyz_degrees": rotation,
        "rationale": rationale,
    }


def _joint_local_rotation(
    body_to_world: Any,
    default_to_world: Any,
    *,
    body_name: str,
    Gf: Any,
) -> Any:
    """Return a body-local joint frame aligned to the source default prim."""

    def normalized(direction: Any, *, label: str) -> Any:
        length = float(direction.GetLength())
        if not math.isfinite(length) or length <= 1e-12:
            raise ValueError(f"SimReady joint {body_name} {label} direction is degenerate.")
        return direction / length

    def require_orthonormal(frame: tuple[Any, Any, Any], *, label: str) -> None:
        if any(
            abs(float(Gf.Dot(frame[left], frame[right]))) > 1e-6
            for left, right in ((0, 1), (0, 2), (1, 2))
        ) or float(Gf.Dot(Gf.Cross(frame[0], frame[1]), frame[2])) < (1.0 - 1e-6):
            raise ValueError(
                f"SimReady joint body {body_name} {label} cannot represent a shared "
                "orthonormal joint frame."
            )

    for transform, label in (
        (body_to_world, "body"),
        (default_to_world, "default prim"),
    ):
        determinant = float(transform.GetDeterminant())
        if not math.isfinite(determinant) or math.isclose(
            determinant,
            0.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"SimReady joint body {body_name} {label} transform must be invertible."
            )

    bases = (
        Gf.Vec3d(1.0, 0.0, 0.0),
        Gf.Vec3d(0.0, 1.0, 0.0),
        Gf.Vec3d(0.0, 0.0, 1.0),
    )
    world_frame = tuple(
        normalized(
            default_to_world.TransformDir(direction),
            label=f"default-frame axis {index}",
        )
        for index, direction in enumerate(bases)
    )
    require_orthonormal(world_frame, label="default frame")
    world_to_body = body_to_world.GetInverse()
    local_frame = tuple(
        normalized(
            world_to_body.TransformDir(direction),
            label=f"local-frame axis {index}",
        )
        for index, direction in enumerate(world_frame)
    )
    require_orthonormal(local_frame, label="local frame")

    matrix = Gf.Matrix3d(1.0)
    for row, direction in enumerate(local_frame):
        matrix.SetRow(row, direction)
    rotation = matrix.ExtractRotation()
    for base, expected_local, expected_world in zip(
        bases,
        local_frame,
        world_frame,
        strict=True,
    ):
        observed_local = rotation.TransformDir(base)
        observed_world = normalized(
            body_to_world.TransformDir(observed_local),
            label="reprojected joint-frame",
        )
        if any(
            abs(float(observed_local[axis]) - float(expected_local[axis])) > 1e-6
            or abs(float(observed_world[axis]) - float(expected_world[axis])) > 1e-6
            for axis in range(3)
        ):
            raise ValueError(
                f"SimReady joint body {body_name} could not preserve the common "
                "joint-frame orientation."
            )
    quaternion = rotation.GetQuat()
    return Gf.Quatf(
        float(quaternion.GetReal()),
        Gf.Vec3f(*(float(value) for value in quaternion.GetImaginary())),
    )


def _geometry_member_fingerprint(member: Any) -> str | None:
    """Fingerprint authored Gprim geometry while ignoring authoring-owned state."""

    from pxr import Usd, UsdGeom

    member_path = str(member.GetPath())
    payload: list[dict[str, Any]] = []
    for prim in Usd.PrimRange(member):
        if not prim.IsA(UsdGeom.Gprim):
            continue
        attributes = []
        for attribute in sorted(prim.GetAttributes(), key=lambda item: item.GetName()):
            name = attribute.GetName()
            if name == "xformOpOrder" or name.startswith(("xformOp:", "physics:")):
                continue
            if not attribute.HasAuthoredValueOpinion():
                continue
            attributes.append(
                {
                    "name": name,
                    "type": str(attribute.GetTypeName()),
                    "default": repr(attribute.Get()),
                    "samples": [
                        [float(sample), repr(attribute.Get(Usd.TimeCode(sample)))]
                        for sample in attribute.GetTimeSamples()
                    ],
                }
            )
        prim_path = str(prim.GetPath())
        payload.append(
            {
                "relative_path": prim_path[len(member_path) :] or ".",
                "type": prim.GetTypeName(),
                "attributes": attributes,
            }
        )
    if not payload:
        return None
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _geometry_fingerprints(default_prim: Any) -> dict[str, str]:
    """Return geometry-only fingerprints keyed by direct default-prim member."""

    fingerprints: dict[str, str] = {}
    for member in default_prim.GetChildren():
        fingerprint = _geometry_member_fingerprint(member)
        if fingerprint is not None:
            fingerprints[member.GetName()] = fingerprint
    return fingerprints


def _remove_api(prim: Any, api_schema: Any) -> bool:
    if not prim.HasAPI(api_schema):
        return False
    prim.RemoveAPI(api_schema)
    return True


def _valid_physics_material_target(stage: Any, prim: Any, UsdPhysics: Any) -> bool:
    relationship = prim.GetRelationship("material:binding:physics")
    if not relationship:
        return False
    return any(
        (target_prim := stage.GetPrimAtPath(target)).IsValid()
        and target_prim.HasAPI(UsdPhysics.MaterialAPI)
        for target in relationship.GetTargets()
    )


def _valid_visual_material_target(prim: Any, UsdShade: Any) -> bool:
    try:
        material, _relationship = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    except Exception:
        return False
    return bool(material and material.GetPrim().IsValid())


def _resolve_member(default_prim: Any, member_name: str) -> Any:
    stage = default_prim.GetStage()
    if member_name.startswith("/"):
        prim = stage.GetPrimAtPath(member_name)
    else:
        prim = default_prim.GetChild(member_name)
    if not prim or not prim.IsValid():
        raise ValueError(f"SimReady policy member does not exist: {member_name}")
    if prim.GetParent() != default_prim:
        raise ValueError(
            f"SimReady policy members must be direct children of the default prim: {prim.GetPath()}"
        )
    return prim


def _move_members(
    stage: Any,
    default_prim: Any,
    destination: Any,
    members: tuple[str, ...],
) -> dict[str, str]:
    from pxr import Sdf

    moved: dict[str, str] = {}
    edits = Sdf.BatchNamespaceEdit()
    for member_name in members:
        prim = _resolve_member(default_prim, member_name)
        source_path = str(prim.GetPath())
        prim_name = prim.GetName()
        destination_path = f"{destination.GetPath()}/{prim_name}"
        edits.Add(Sdf.Path(source_path), Sdf.Path(destination_path))
        moved[source_path] = destination_path
    if members and not stage.GetRootLayer().Apply(edits):
        raise RuntimeError(f"Failed to reparent members below {destination.GetPath()}.")
    return moved


def _remap_property_paths(stage: Any, member_mapping: dict[str, str]) -> int:
    """Retarget relationships and connections after root-layer namespace edits."""

    from pxr import Sdf

    path_mapping = sorted(
        ((Sdf.Path(source), Sdf.Path(target)) for source, target in member_mapping.items()),
        key=lambda item: len(str(item[0])),
        reverse=True,
    )

    def remap(path: Any) -> Any:
        for source, target in path_mapping:
            if path.HasPrefix(source):
                return path.ReplacePrefix(source, target)
        return path

    changed = 0
    for prim in stage.TraverseAll():
        for relationship in prim.GetRelationships():
            targets = relationship.GetTargets()
            remapped = [remap(target) for target in targets]
            if remapped != targets:
                relationship.SetTargets(remapped)
                changed += 1
        for attribute in prim.GetAttributes():
            connections = attribute.GetConnections()
            remapped = [remap(connection) for connection in connections]
            if remapped != connections:
                attribute.SetConnections(remapped)
                changed += 1
    return changed


def _member_world_transforms(default_prim: Any, members: tuple[str, ...]) -> dict[str, Any]:
    from pxr import Gf, Usd, UsdGeom

    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    return {
        str(prim.GetPath()): Gf.Matrix4d(cache.GetLocalToWorldTransform(prim))
        for member in members
        for prim in [_resolve_member(default_prim, member)]
    }


def _geometry_center_in_default_space(default_prim: Any, members: tuple[str, ...]) -> Any:
    from pxr import Gf, Usd, UsdGeom

    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    minima = [math.inf, math.inf, math.inf]
    maxima = [-math.inf, -math.inf, -math.inf]
    for member in members:
        aligned = bbox_cache.ComputeWorldBound(
            _resolve_member(default_prim, member)
        ).ComputeAlignedBox()
        low = aligned.GetMin()
        high = aligned.GetMax()
        for axis in range(3):
            minima[axis] = min(minima[axis], float(low[axis]))
            maxima[axis] = max(maxima[axis], float(high[axis]))
    if not all(math.isfinite(value) for value in minima + maxima):
        raise ValueError("Cannot recenter a SimReady body with non-finite geometry bounds.")
    center_world = Gf.Vec3d(*((minima[axis] + maxima[axis]) * 0.5 for axis in range(3)))
    default_to_world = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(
        default_prim
    )
    return default_to_world.GetInverse().Transform(center_world)


def _rebase_body_origin(
    stage: Any,
    body_prim: Any,
    member_mapping: dict[str, str],
    member_world_transforms: dict[str, Any],
    center_in_default: Any,
) -> None:
    from pxr import Gf, Usd, UsdGeom

    UsdGeom.Xformable(body_prim).AddTranslateOp(opSuffix="simreadyBodyOrigin").Set(
        Gf.Vec3d(*center_in_default)
    )
    body_to_world = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(body_prim)
    world_to_body = body_to_world.GetInverse()
    for source_path, destination_path in member_mapping.items():
        destination = stage.GetPrimAtPath(destination_path)
        local_to_body = member_world_transforms[source_path] * world_to_body
        UsdGeom.Xformable(destination).MakeMatrixXform().Set(local_to_body)
    verification_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    for source_path, destination_path in member_mapping.items():
        actual = verification_cache.GetLocalToWorldTransform(stage.GetPrimAtPath(destination_path))
        expected = member_world_transforms[source_path]
        delta = max(
            abs(float(actual[row][column]) - float(expected[row][column]))
            for row in range(4)
            for column in range(4)
        )
        if delta > 1e-9:
            raise RuntimeError(
                f"Rebasing {destination_path} changed its world transform by {delta:.3e}."
            )


def _create_grasp_line(
    stage: Any,
    default_prim: Any,
    grasp: dict[str, Any],
    fixes: list[str],
) -> str:
    from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt

    raw_points = grasp.get("points")
    if not isinstance(raw_points, list) or len(raw_points) < 2:
        raise ValueError("SimReady policy grasp.points must contain at least two points.")
    points: list[tuple[float, float, float]] = []
    for raw_point in raw_points:
        if not isinstance(raw_point, list) or len(raw_point) != 3:
            raise ValueError("Each SimReady grasp point must contain three coordinates.")
        point = tuple(float(value) for value in raw_point)
        if not all(math.isfinite(value) for value in point):
            raise ValueError("SimReady grasp points must be finite.")
        points.append(point)

    curve_path = f"{default_prim.GetPath()}/grasp_identifier_01"
    existing = stage.GetPrimAtPath(curve_path)
    if existing and existing.IsValid():
        stage.RemovePrim(curve_path)
    curve = UsdGeom.BasisCurves.Define(stage, curve_path)
    curve.CreateTypeAttr(UsdGeom.Tokens.linear)
    curve.CreateCurveVertexCountsAttr(Vt.IntArray([len(points)]))
    curve.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*point) for point in points]))
    line_length = math.sqrt(sum((points[-1][index] - points[0][index]) ** 2 for index in range(3)))
    width = min(max(line_length * 0.035, 0.0005), 0.004)
    curve.CreateWidthsAttr(Vt.FloatArray([width]))
    curve.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(0.1, 0.9, 0.2)]))
    curve.CreatePurposeAttr(UsdGeom.Tokens.guide)
    minima = [min(point[index] for point in points) - width for index in range(3)]
    maxima = [max(point[index] for point in points) + width for index in range(3)]
    curve.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*minima), Gf.Vec3f(*maxima)]))
    curve.GetPrim().SetCustomDataByKey(
        "geometry_agent:grasp_target", str(grasp.get("target") or "")
    )
    curve.GetPrim().SetCustomDataByKey(
        "geometry_agent:grasp_rationale", str(grasp.get("rationale") or "")
    )

    looks_path = f"{default_prim.GetPath()}/Looks"
    UsdGeom.Scope.Define(stage, looks_path)
    material = UsdShade.Material.Define(stage, f"{looks_path}/grasp_guide")
    shader = UsdShade.Shader.Define(stage, f"{looks_path}/grasp_guide/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.1, 0.9, 0.2))
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.02, 0.2, 0.04))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(curve.GetPrim()).Bind(material)
    fixes.append(f"author_semantic_grasp_line:{curve_path}")
    return curve_path


def _coacd_options(value: Any, *, body_name: str) -> dict[str, Any]:
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ValueError(f"SimReady body {body_name} collision_proxy_options must be an object.")
    options = {
        "threshold": float(value.get("threshold", 0.035)),
        "max_convex_hull": int(value.get("max_convex_hull", 24)),
        "preprocess_resolution": int(value.get("preprocess_resolution", 50)),
        "resolution": int(value.get("resolution", 2000)),
        "max_ch_vertex": int(value.get("max_ch_vertex", 128)),
        "seed": int(value.get("seed", 0)),
    }
    if not math.isfinite(options["threshold"]) or not 0.001 <= options["threshold"] <= 1.0:
        raise ValueError(f"SimReady body {body_name} CoACD threshold must be in [0.001, 1.0].")
    for field in ("max_convex_hull", "preprocess_resolution", "resolution", "max_ch_vertex"):
        if options[field] <= 0:
            raise ValueError(f"SimReady body {body_name} CoACD {field} must be positive.")
    return options


def _triangulate_polygon_face(
    face: list[int],
    vertices: list[tuple[float, float, float]],
    *,
    mesh_path: str,
) -> list[list[int]]:
    """Triangulate one simple planar face without assuming convexity."""

    cleaned: list[int] = []
    for index in face:
        if index < 0 or index >= len(vertices):
            raise ValueError(
                f"Collision proxy source {mesh_path} references vertex {index} "
                "outside its points array."
            )
        if not cleaned or cleaned[-1] != index:
            cleaned.append(index)
    if len(cleaned) > 1 and cleaned[0] == cleaned[-1]:
        cleaned.pop()
    if len(set(cleaned)) < 3:
        raise ValueError(
            f"Collision proxy source {mesh_path} has a face with fewer than "
            "three distinct vertices."
        )

    points = [vertices[index] for index in cleaned]
    origin = points[0]
    local_points = [tuple(point[axis] - origin[axis] for axis in range(3)) for point in points]
    normal = (
        sum(
            (local_points[index][1] - local_points[(index + 1) % len(local_points)][1])
            * (local_points[index][2] + local_points[(index + 1) % len(local_points)][2])
            for index in range(len(local_points))
        ),
        sum(
            (local_points[index][2] - local_points[(index + 1) % len(local_points)][2])
            * (local_points[index][0] + local_points[(index + 1) % len(local_points)][0])
            for index in range(len(local_points))
        ),
        sum(
            (local_points[index][0] - local_points[(index + 1) % len(local_points)][0])
            * (local_points[index][1] + local_points[(index + 1) % len(local_points)][1])
            for index in range(len(local_points))
        ),
    )
    linear_scale = max(
        max(point[axis] for point in local_points) - min(point[axis] for point in local_points)
        for axis in range(3)
    )
    if not math.isfinite(linear_scale) or linear_scale <= 0.0:
        raise ValueError(f"Collision proxy source {mesh_path} has a degenerate face.")
    area_epsilon = max(
        linear_scale * linear_scale * 1e-12,
        math.ulp(linear_scale) * linear_scale * 32.0,
    )
    drop_axis = max(range(3), key=lambda axis: abs(normal[axis]))
    if abs(normal[drop_axis]) <= area_epsilon:
        raise ValueError(f"Collision proxy source {mesh_path} has a zero-area face.")
    axes = tuple(axis for axis in range(3) if axis != drop_axis)
    projected = [(point[axes[0]], point[axes[1]]) for point in local_points]

    def cross(a: int, b: int, c: int) -> float:
        pa, pb, pc = projected[a], projected[b], projected[c]
        return (pb[0] - pa[0]) * (pc[1] - pa[1]) - (pb[1] - pa[1]) * (pc[0] - pa[0])

    signed_area = 0.5 * sum(
        projected[index][0] * projected[(index + 1) % len(projected)][1]
        - projected[(index + 1) % len(projected)][0] * projected[index][1]
        for index in range(len(projected))
    )
    if abs(signed_area) <= area_epsilon:
        raise ValueError(f"Collision proxy source {mesh_path} has a zero-area face.")
    orientation = 1.0 if signed_area > 0.0 else -1.0

    def inside_triangle(point: int, a: int, b: int, c: int) -> bool:
        return all(
            orientation * value >= -area_epsilon
            for value in (cross(a, b, point), cross(b, c, point), cross(c, a, point))
        )

    remaining = list(range(len(cleaned)))
    triangles: list[list[int]] = []
    while len(remaining) > 3:
        ear_found = False
        for offset, current in enumerate(remaining):
            previous = remaining[offset - 1]
            following = remaining[(offset + 1) % len(remaining)]
            if orientation * cross(previous, current, following) <= area_epsilon:
                continue
            if any(
                inside_triangle(candidate, previous, current, following)
                for candidate in remaining
                if candidate not in {previous, current, following}
            ):
                continue
            triangles.append([cleaned[previous], cleaned[current], cleaned[following]])
            del remaining[offset]
            ear_found = True
            break
        if ear_found:
            continue

        collinear = next(
            (
                offset
                for offset, current in enumerate(remaining)
                if abs(
                    cross(
                        remaining[offset - 1],
                        current,
                        remaining[(offset + 1) % len(remaining)],
                    )
                )
                <= area_epsilon
            ),
            None,
        )
        if collinear is None:
            raise ValueError(
                f"Collision proxy source {mesh_path} has a self-intersecting "
                "or untriangulatable face."
            )
        del remaining[collinear]

    triangles.append([cleaned[index] for index in remaining])
    return triangles


def _triangulated_mesh_in_reference_space(
    stage: Any,
    mesh_prim: Any,
    reference_prim: Any,
) -> tuple[Any, Any]:
    import numpy as np
    from pxr import Gf, Usd, UsdGeom

    mesh = UsdGeom.Mesh(mesh_prim)
    points = mesh.GetPointsAttr().Get() or []
    counts = list(mesh.GetFaceVertexCountsAttr().Get() or [])
    indices = list(mesh.GetFaceVertexIndicesAttr().Get() or [])
    if not points or not counts or not indices:
        raise ValueError(f"Collision proxy source mesh is empty: {mesh_prim.GetPath()}")
    if list(mesh.GetHoleIndicesAttr().Get() or []):
        raise ValueError(
            f"Collision proxy source {mesh_prim.GetPath()} contains polygon holes; "
            "triangulate it explicitly before CoACD authoring."
        )

    triangles: list[list[int]] = []
    vertices = [tuple(float(value) for value in point) for point in points]
    offset = 0
    for count in counts:
        face = indices[offset : offset + count]
        offset += count
        if count < 3:
            raise ValueError(
                f"Collision proxy source {mesh_prim.GetPath()} has a face with "
                "fewer than three vertices."
            )
        triangles.extend(
            _triangulate_polygon_face(
                face,
                vertices,
                mesh_path=str(mesh_prim.GetPath()),
            )
        )
    if offset != len(indices):
        raise ValueError(
            f"Collision proxy source {mesh_prim.GetPath()} has inconsistent "
            "face counts and indices."
        )
    if not triangles:
        raise ValueError(
            f"Collision proxy source has no triangulatable faces: {mesh_prim.GetPath()}"
        )

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    mesh_to_world = xform_cache.GetLocalToWorldTransform(mesh_prim)
    world_to_reference = xform_cache.GetLocalToWorldTransform(reference_prim).GetInverse()
    vertices = np.asarray(
        [
            list(world_to_reference.Transform(mesh_to_world.Transform(Gf.Vec3d(*point))))
            for point in points
        ],
        dtype=np.float64,
    )
    return vertices, np.asarray(triangles, dtype=np.int32)


def _author_coacd_collision_proxies(
    stage: Any,
    body_prim: Any,
    member_paths: list[str],
    *,
    body_name: str,
    options: dict[str, Any],
) -> dict[str, Any]:
    try:
        import coacd
    except ImportError as exc:
        raise RuntimeError(
            "CoACD collision proxies were requested but the optional 'coacd' package "
            "is not installed. Install cad-verifier[collision]."
        ) from exc
    coacd.set_log_level("warn")

    from pxr import Gf, Usd, UsdGeom, Vt

    proxy_root = UsdGeom.Xform.Define(stage, f"{body_prim.GetPath()}/CollisionProxies").GetPrim()
    UsdGeom.Imageable(proxy_root).CreatePurposeAttr(UsdGeom.Tokens.guide)
    proxy_paths: list[str] = []
    sources: list[dict[str, Any]] = []
    for member_path in member_paths:
        member = stage.GetPrimAtPath(member_path)
        mesh_prims = [prim for prim in Usd.PrimRange(member) if prim.IsA(UsdGeom.Mesh)]
        for mesh_index, mesh_prim in enumerate(mesh_prims):
            vertices, triangles = _triangulated_mesh_in_reference_space(stage, mesh_prim, body_prim)
            parts = coacd.run_coacd(
                coacd.Mesh(vertices, triangles),
                threshold=options["threshold"],
                max_convex_hull=options["max_convex_hull"],
                preprocess_resolution=options["preprocess_resolution"],
                resolution=options["resolution"],
                max_ch_vertex=options["max_ch_vertex"],
                seed=options["seed"],
            )
            if not parts:
                raise RuntimeError(f"CoACD produced no hulls for {mesh_prim.GetPath()}.")
            source_name = _safe_name(f"{Path(member_path).name}_{mesh_index:02d}")
            for hull_index, (hull_vertices, hull_faces) in enumerate(parts):
                hull_path = f"{proxy_root.GetPath()}/{source_name}_hull_{hull_index:03d}"
                proxy = UsdGeom.Mesh.Define(stage, hull_path)
                points = [Gf.Vec3f(*(float(value) for value in vertex)) for vertex in hull_vertices]
                faces = [list(int(value) for value in face) for face in hull_faces]
                proxy.CreatePointsAttr(Vt.Vec3fArray(points))
                proxy.CreateFaceVertexCountsAttr(Vt.IntArray([len(face) for face in faces]))
                proxy.CreateFaceVertexIndicesAttr(
                    Vt.IntArray([value for face in faces for value in face])
                )
                proxy.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
                minima = [
                    min(float(vertex[index]) for vertex in hull_vertices) for index in range(3)
                ]
                maxima = [
                    max(float(vertex[index]) for vertex in hull_vertices) for index in range(3)
                ]
                proxy.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*minima), Gf.Vec3f(*maxima)]))
                imageable = UsdGeom.Imageable(proxy.GetPrim())
                imageable.CreatePurposeAttr(UsdGeom.Tokens.guide)
                imageable.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
                proxy.GetPrim().SetCustomDataByKey(
                    "geometry_agent:collision_proxy_backend", "coacd"
                )
                proxy.GetPrim().SetCustomDataByKey(
                    "geometry_agent:collision_proxy_source",
                    str(mesh_prim.GetPath()),
                )
                proxy_paths.append(hull_path)
            sources.append(
                {
                    "member_path": member_path,
                    "mesh_path": str(mesh_prim.GetPath()),
                    "hull_count": len(parts),
                    "source_vertex_count": int(vertices.shape[0]),
                    "source_triangle_count": int(triangles.shape[0]),
                }
            )
    if not proxy_paths:
        raise RuntimeError(f"CoACD produced no collision proxies for body {body_name}.")
    return {
        "backend": "coacd",
        "body_name": body_name,
        "proxy_root": str(proxy_root.GetPath()),
        "proxy_paths": proxy_paths,
        "hull_count": len(proxy_paths),
        "options": options,
        "sources": sources,
    }


def _flatten_source_for_authoring(source: Path, target: Path) -> None:
    """Resolve composition at the source anchor and export one editable layer."""

    from pxr import Usd

    source_stage = Usd.Stage.Open(str(source))
    if source_stage is None:
        raise RuntimeError(f"Usd.Stage.Open returned None for source: {source}")
    flattened = source_stage.Flatten()
    if flattened is None or not flattened.Export(str(target)):
        raise RuntimeError(f"Failed to flatten source USD for authoring: {source}")


def author_simready_package(
    usd_path: str | Path,
    output_path: str | Path,
    *,
    policy: dict[str, Any],
) -> dict[str, Any]:
    """Author a staged SimReady USD package from an explicit semantic policy.

    Policy members refer to direct children of the source stage's default prim.
    Supported member buckets are ``bodies[*].members``, ``static_members``,
    ``visual_only_members``, and ``guide_members``. Every source geometry member
    must appear in exactly one bucket; omissions are rejected rather than
    guessed. Source coordinates must already be Z-up meters; changing metadata
    without converting geometry and physics quantities is rejected. Joint
    ``position`` values and joint axes use the source default prim's coordinate
    frame and are converted independently into each body's local frame after
    recentering and pose adjustments.
    """

    source = Path(usd_path).resolve()
    target = Path(output_path).resolve()
    fixes: list[str] = []
    sidecar = target.with_suffix(".json")
    if not source.is_file():
        cleanup_errors = _remove_failed_artifacts((target, sidecar))
        return {
            "schema": SIMREADY_AUTHORING_SCHEMA,
            "status": "fail",
            "message": f"USD source does not exist: {source}",
            "artifacts": {},
            "fixes": fixes,
            "cleanup_errors": cleanup_errors,
        }
    if not isinstance(policy, dict):
        _remove_failed_artifacts((target, sidecar))
        raise TypeError("SimReady authoring policy must be a dictionary.")
    if source == target:
        return {
            "schema": SIMREADY_AUTHORING_SCHEMA,
            "status": "fail",
            "message": "SimReady output must differ from the source USD.",
            "source_usd": str(source),
            "artifacts": {},
            "fixes": fixes,
        }

    run_token = uuid.uuid4().hex
    source_suffix = source.suffix.lower() or ".usd"
    edit_target = target.with_name(f".{target.stem}.{run_token}.authoring{source_suffix}")
    publish_target = target.with_name(
        f".{target.stem}.{run_token}.publish{target.suffix or '.usd'}"
    )
    sidecar_target = target.with_name(f".{target.stem}.{run_token}.metadata.json")
    transient_paths = (edit_target, publish_target, sidecar_target)

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        cleanup_errors = _remove_failed_artifacts((target, sidecar, *transient_paths))
        if cleanup_errors:
            raise RuntimeError("; ".join(cleanup_errors))
        _flatten_source_for_authoring(source, edit_target)
        fixes.append("flatten_source_composition")

        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

        stage = Usd.Stage.Open(str(edit_target))
        if stage is None:
            raise RuntimeError("Usd.Stage.Open returned None.")
        default_prim = stage.GetDefaultPrim()
        if not default_prim or not default_prim.IsValid():
            roots = [prim for prim in stage.GetPseudoRoot().GetChildren() if prim.IsValid()]
            if len(roots) != 1:
                raise ValueError("SimReady authoring requires one resolvable default prim.")
            stage.SetDefaultPrim(roots[0])
            default_prim = roots[0]
            fixes.append(f"set_default_prim:{default_prim.GetPath()}")

        source_up_axis = UsdGeom.GetStageUpAxis(stage)
        source_meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
        normalization_errors: list[str] = []
        if source_up_axis != UsdGeom.Tokens.z:
            normalization_errors.append(f"upAxis={source_up_axis}")
        if not math.isclose(
            source_meters_per_unit,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            normalization_errors.append(f"metersPerUnit={source_meters_per_unit:.17g}")
        if normalization_errors:
            raise ValueError(
                "SimReady authoring requires source USD geometry normalized to "
                "Z-up meters before physical authoring; observed "
                + ", ".join(normalization_errors)
                + ". Convert coordinates and physics quantities before authoring."
            )

        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        fixes.extend(["set_up_axis_z", "set_meters_per_unit_1"])

        instance_paths = [str(prim.GetPath()) for prim in stage.Traverse() if prim.IsInstance()]
        for instance_path in instance_paths:
            instance_prim = stage.GetPrimAtPath(instance_path)
            instance_prim.SetInstanceable(False)
        if instance_paths:
            fixes.append(f"deinstance_for_neutral_colliders:{len(instance_paths)}")
            flattened_layer = stage.Flatten()
            root_layer = stage.GetRootLayer()
            root_layer.TransferContent(flattened_layer)
            root_layer.Save()
            stage.Reload()
            default_prim = stage.GetDefaultPrim()
            if not default_prim or not default_prim.IsValid():
                raise RuntimeError("Flattened SimReady staging layer lost its default prim.")
            fixes.append("flatten_deinstanced_staging_layer")

        source_fingerprints = _geometry_fingerprints(default_prim)
        bodies = policy.get("bodies")
        if not isinstance(bodies, list) or not bodies:
            raise ValueError("SimReady policy bodies must contain at least one body definition.")

        body_specs: list[dict[str, Any]] = []
        assigned: list[str] = []
        supported_approximations = {
            "none",
            "convexHull",
            "convexDecomposition",
            "meshSimplification",
            "boundingSphere",
            "boundingCube",
        }
        for index, raw_body in enumerate(bodies):
            if not isinstance(raw_body, dict):
                raise ValueError(f"SimReady policy body {index} must be an object.")
            name = str(raw_body.get("name") or "").strip()
            if not name:
                raise ValueError(f"SimReady policy body {index} requires a name.")
            members = _normalize_member_names(
                raw_body.get("members"), field=f"bodies[{index}].members"
            )
            if not members:
                raise ValueError(f"SimReady policy body {name} has no members.")
            attached_visual_members = _normalize_member_names(
                raw_body.get("visual_only_members"),
                field=f"bodies[{index}].visual_only_members",
            )
            mass_kg = float(raw_body.get("mass_kg", 0.1))
            if not math.isfinite(mass_kg) or mass_kg <= 0.0:
                raise ValueError(f"SimReady body {name} mass_kg must be positive and finite.")
            kinematic = bool(raw_body.get("kinematic", False))
            collision_approximation = str(
                raw_body.get("collision_approximation") or ("none" if kinematic else "convexHull")
            )
            if collision_approximation not in supported_approximations:
                raise ValueError(
                    f"SimReady body {name} collision_approximation must be one of "
                    f"{sorted(supported_approximations)}."
                )
            collision_proxy_backend = str(raw_body.get("collision_proxy_backend") or "").strip()
            if collision_proxy_backend not in {"", "coacd"}:
                raise ValueError(
                    f"SimReady body {name} collision_proxy_backend must be 'coacd' when set."
                )
            collision_proxy_options = (
                _coacd_options(raw_body.get("collision_proxy_options"), body_name=name)
                if collision_proxy_backend
                else {}
            )
            pose_adjustment = _body_pose_adjustment(raw_body.get("pose_adjustment"), body_name=name)
            recenter_origin = bool(raw_body.get("recenter_origin", not kinematic))
            body_specs.append(
                {
                    "name": name,
                    "members": members,
                    "visual_only_members": attached_visual_members,
                    "mass_kg": mass_kg,
                    "kinematic": kinematic,
                    "collision_approximation": collision_approximation,
                    "collision_proxy_backend": collision_proxy_backend or None,
                    "collision_proxy_options": collision_proxy_options,
                    "pose_adjustment": pose_adjustment,
                    "recenter_origin": recenter_origin,
                }
            )
            assigned.extend(members)
            assigned.extend(attached_visual_members)

        _require_unique_safe_names(
            [str(body_spec["name"]) for body_spec in body_specs],
            field="body",
        )

        static_members = _normalize_member_names(
            policy.get("static_members"), field="static_members"
        )
        visual_only_members = _normalize_member_names(
            policy.get("visual_only_members"), field="visual_only_members"
        )
        guide_members = _normalize_member_names(policy.get("guide_members"), field="guide_members")
        assigned.extend(static_members)
        assigned.extend(visual_only_members)
        assigned.extend(guide_members)
        if len(assigned) != len(set(assigned)):
            duplicates = sorted({member for member in assigned if assigned.count(member) > 1})
            raise ValueError(
                "SimReady policy assigns members more than once: " + ", ".join(duplicates)
            )
        source_members = set(source_fingerprints)
        assigned_members = set(assigned)
        if source_members != assigned_members:
            missing = sorted(source_members - assigned_members)
            unknown = sorted(assigned_members - source_members)
            details = []
            if missing:
                details.append("unassigned=" + ",".join(missing))
            if unknown:
                details.append("unknown=" + ",".join(unknown))
            raise ValueError("SimReady policy geometry coverage mismatch: " + "; ".join(details))

        sim_bodies = UsdGeom.Xform.Define(stage, f"{default_prim.GetPath()}/SimBodies").GetPrim()
        member_mapping: dict[str, str] = {}
        body_paths: list[str] = []
        collider_member_paths: list[str] = []
        collider_approximation_by_root: dict[str, str] = {}
        collision_proxy_records: list[dict[str, Any]] = []
        pose_adjustment_records: list[dict[str, Any]] = []
        body_origin_records: list[dict[str, Any]] = []
        for body_spec in body_specs:
            body = UsdGeom.Xform.Define(
                stage, f"{sim_bodies.GetPath()}/{_safe_name(body_spec['name'])}"
            ).GetPrim()
            all_body_members = body_spec["members"] + body_spec["visual_only_members"]
            world_transforms = _member_world_transforms(default_prim, all_body_members)
            center_in_default = (
                _geometry_center_in_default_space(default_prim, body_spec["members"])
                if body_spec["recenter_origin"]
                else None
            )
            physical_mapping = _move_members(stage, default_prim, body, body_spec["members"])
            visual_mapping = _move_members(
                stage,
                default_prim,
                body,
                body_spec["visual_only_members"],
            )
            body_mapping = {**physical_mapping, **visual_mapping}
            member_mapping.update(body_mapping)
            if center_in_default is not None:
                _rebase_body_origin(
                    stage,
                    body,
                    body_mapping,
                    world_transforms,
                    center_in_default,
                )
                origin_record = {
                    "body_name": body_spec["name"],
                    "body_path": str(body.GetPath()),
                    "origin_in_default_space": [float(value) for value in center_in_default],
                    "world_geometry_preserved": True,
                }
                body_origin_records.append(origin_record)
                fixes.append(f"recenter_body_origin:{body_spec['name']}")
            pose_adjustment = body_spec["pose_adjustment"]
            if pose_adjustment:
                xformable = UsdGeom.Xformable(body)
                translation = pose_adjustment["translation"]
                rotation = pose_adjustment["rotate_xyz_degrees"]
                if any(translation):
                    xformable.AddTranslateOp(opSuffix="simreadyPoseAdjustment").Set(
                        Gf.Vec3d(*translation)
                    )
                if any(rotation):
                    xformable.AddRotateXYZOp(opSuffix="simreadyPoseAdjustment").Set(
                        Gf.Vec3f(*rotation)
                    )
                pose_record = {
                    "body_name": body_spec["name"],
                    "body_path": str(body.GetPath()),
                    "translation": list(translation),
                    "rotate_xyz_degrees": list(rotation),
                    "rationale": pose_adjustment["rationale"],
                }
                pose_adjustment_records.append(pose_record)
                fixes.append(f"author_body_pose_adjustment:{body_spec['name']}")
            if body_spec["collision_proxy_backend"] == "coacd":
                proxy_record = _author_coacd_collision_proxies(
                    stage,
                    body,
                    list(physical_mapping.values()),
                    body_name=body_spec["name"],
                    options=body_spec["collision_proxy_options"],
                )
                collision_proxy_records.append(proxy_record)
                collider_member_paths.append(proxy_record["proxy_root"])
                collider_approximation_by_root[proxy_record["proxy_root"]] = "convexHull"
                fixes.append(
                    f"author_coacd_collision_hulls:{body_spec['name']}:{proxy_record['hull_count']}"
                )
            else:
                collider_member_paths.extend(physical_mapping.values())
                collider_approximation_by_root.update(
                    {
                        path: body_spec["collision_approximation"]
                        for path in physical_mapping.values()
                    }
                )
            body_paths.append(str(body.GetPath()))

        bucket_paths: dict[str, str] = {}
        for bucket_name, members in (
            ("Static", static_members),
            ("VisualOnly", visual_only_members),
            ("Guides", guide_members),
        ):
            if not members:
                continue
            bucket = UsdGeom.Xform.Define(
                stage, f"{default_prim.GetPath()}/{bucket_name}"
            ).GetPrim()
            member_mapping.update(_move_members(stage, default_prim, bucket, members))
            bucket_paths[bucket_name] = str(bucket.GetPath())

        remapped_property_count = _remap_property_paths(stage, member_mapping)
        if remapped_property_count:
            fixes.append(f"remap_namespace_property_paths:{remapped_property_count}")

        source_joint_paths = [
            str(prim.GetPath()) for prim in stage.Traverse() if prim.IsA(UsdPhysics.Joint)
        ]
        for joint_path in sorted(source_joint_paths, key=len, reverse=True):
            stage.RemovePrim(joint_path)
        if source_joint_paths:
            fixes.append(f"remove_source_physics_joints:{len(source_joint_paths)}")

        removed_counts = {"rigid_body": 0, "mass": 0, "collision": 0, "mesh_collision": 0}
        for prim in list(stage.Traverse()):
            removed_counts["rigid_body"] += int(_remove_api(prim, UsdPhysics.RigidBodyAPI))
            removed_counts["mass"] += int(_remove_api(prim, UsdPhysics.MassAPI))
            removed_counts["collision"] += int(_remove_api(prim, UsdPhysics.CollisionAPI))
            removed_counts["mesh_collision"] += int(_remove_api(prim, UsdPhysics.MeshCollisionAPI))
        fixes.append("reset_source_physics_schemas")

        material_policy = dict(policy.get("physics_material") or {})
        physics_scope = UsdGeom.Scope.Define(stage, f"{default_prim.GetPath()}/PhysicsMaterials")
        default_material = UsdShade.Material.Define(
            stage, f"{physics_scope.GetPath()}/geometry_agent_default"
        )
        material_api = UsdPhysics.MaterialAPI.Apply(default_material.GetPrim())
        material_api.CreateStaticFrictionAttr(float(material_policy.get("static_friction", 0.8)))
        material_api.CreateDynamicFrictionAttr(float(material_policy.get("dynamic_friction", 0.65)))
        material_api.CreateRestitutionAttr(float(material_policy.get("restitution", 0.05)))
        material_api.CreateDensityAttr(float(material_policy.get("density_kg_m3", 1000.0)))
        fixes.append(f"author_fallback_physics_material:{default_material.GetPath()}")

        looks_scope = UsdGeom.Scope.Define(stage, f"{default_prim.GetPath()}/Looks")
        default_visual_material = UsdShade.Material.Define(
            stage, f"{looks_scope.GetPath()}/geometry_agent_default_visual"
        )
        default_visual_shader = UsdShade.Shader.Define(
            stage, f"{default_visual_material.GetPath()}/PreviewSurface"
        )
        default_visual_shader.CreateIdAttr("UsdPreviewSurface")
        default_visual_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(0.55, 0.58, 0.62)
        )
        default_visual_shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        default_visual_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.58)
        default_visual_material.CreateSurfaceOutput().ConnectToSource(
            default_visual_shader.ConnectableAPI(), "surface"
        )

        fallback_visual_material_bindings: list[str] = []
        for prim in stage.Traverse():
            if not prim.IsA(UsdGeom.Gprim):
                continue
            imageable = UsdGeom.Imageable(prim)
            if imageable.ComputeVisibility() == UsdGeom.Tokens.invisible:
                continue
            if imageable.ComputePurpose() in {UsdGeom.Tokens.guide, UsdGeom.Tokens.proxy}:
                continue
            if _valid_visual_material_target(prim, UsdShade):
                continue
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(default_visual_material)
            fallback_visual_material_bindings.append(str(prim.GetPath()))
        if fallback_visual_material_bindings:
            fixes.append(f"bind_fallback_visual_material:{len(fallback_visual_material_bindings)}")
        else:
            stage.RemovePrim(default_visual_material.GetPath())

        physical_roots = [
            (stage.GetPrimAtPath(path), collider_approximation_by_root[path])
            for path in collider_member_paths
        ]
        static_path = bucket_paths.get("Static")
        if static_path:
            physical_roots.append((stage.GetPrimAtPath(static_path), "none"))
        collider_paths: list[str] = []
        collider_approximations: dict[str, str] = {}
        fallback_material_bindings: list[str] = []
        for physical_root, approximation in physical_roots:
            for prim in stage.Traverse():
                if prim != physical_root and not prim.GetPath().HasPrefix(physical_root.GetPath()):
                    continue
                if not prim.IsA(UsdGeom.Gprim):
                    continue
                UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
                collider_paths.append(str(prim.GetPath()))
                if prim.IsA(UsdGeom.Mesh):
                    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(approximation)
                    collider_approximations[str(prim.GetPath())] = approximation
                if not _valid_physics_material_target(stage, prim, UsdPhysics):
                    prim.CreateRelationship("material:binding:physics").SetTargets(
                        [default_material.GetPath()]
                    )
                    fallback_material_bindings.append(str(prim.GetPath()))
        fixes.append(f"author_gprim_colliders:{len(collider_paths)}")

        for body_spec, body_path in zip(body_specs, body_paths, strict=True):
            body_prim = stage.GetPrimAtPath(body_path)
            rigid_body = UsdPhysics.RigidBodyAPI.Apply(body_prim)
            rigid_body.CreateRigidBodyEnabledAttr(True)
            rigid_body.CreateKinematicEnabledAttr(body_spec["kinematic"])
            UsdPhysics.MassAPI.Apply(body_prim).CreateMassAttr(body_spec["mass_kg"])
            body_prim.SetCustomDataByKey("geometry_agent:body_name", body_spec["name"])
            body_prim.SetCustomDataByKey(
                "geometry_agent:body_members",
                json.dumps(list(body_spec["members"])),
            )
            body_prim.SetCustomDataByKey(
                "geometry_agent:body_visual_only_members",
                json.dumps(list(body_spec["visual_only_members"])),
            )
            if body_spec["collision_proxy_backend"]:
                body_prim.SetCustomDataByKey(
                    "geometry_agent:collision_proxy_backend",
                    body_spec["collision_proxy_backend"],
                )
            fixes.append(f"author_rigid_body:{body_path}")

        body_paths_by_name = {
            body_spec["name"]: body_path
            for body_spec, body_path in zip(body_specs, body_paths, strict=True)
        }
        joint_paths: list[str] = []
        raw_joints = policy.get("joints") or []
        if not isinstance(raw_joints, list):
            raise ValueError("SimReady policy joints must be a list.")
        if raw_joints:
            joint_names: list[str] = []
            for index, raw_joint in enumerate(raw_joints):
                if not isinstance(raw_joint, dict):
                    raise ValueError(f"SimReady policy joint {index} must be an object.")
                joint_name = str(raw_joint.get("name") or "").strip()
                if not joint_name:
                    raise ValueError(f"SimReady policy joint {index} requires a name.")
                joint_names.append(joint_name)
            _require_unique_safe_names(joint_names, field="joint")

            joint_scope = UsdGeom.Scope.Define(stage, f"{default_prim.GetPath()}/Joints")
            joint_xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
            default_to_world = joint_xform_cache.GetLocalToWorldTransform(default_prim)
            has_moving_joint = False
            for index, raw_joint in enumerate(raw_joints):
                joint_name = str(raw_joint.get("name") or "").strip()
                joint_type = str(raw_joint.get("type") or "fixed").strip().lower()
                body0_name = str(raw_joint.get("body0") or "")
                body1_name = str(raw_joint.get("body1") or "")
                if not joint_name or body0_name not in body_paths_by_name:
                    raise ValueError(f"SimReady policy joint {index} has an invalid name/body0.")
                if body1_name not in body_paths_by_name:
                    raise ValueError(f"SimReady policy joint {index} has an invalid body1.")
                joint_path = f"{joint_scope.GetPath()}/{_safe_name(joint_name)}"
                if joint_type == "fixed":
                    joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
                elif joint_type == "revolute":
                    has_moving_joint = True
                    joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
                    axis = str(raw_joint.get("axis") or "Z").upper()
                    if axis not in {"X", "Y", "Z"}:
                        raise ValueError(f"SimReady revolute joint {joint_name} has invalid axis.")
                    joint.CreateAxisAttr(axis)
                    if raw_joint.get("lower_limit_degrees") is not None:
                        joint.CreateLowerLimitAttr(float(raw_joint["lower_limit_degrees"]))
                    if raw_joint.get("upper_limit_degrees") is not None:
                        joint.CreateUpperLimitAttr(float(raw_joint["upper_limit_degrees"]))
                elif joint_type == "prismatic":
                    has_moving_joint = True
                    joint = UsdPhysics.PrismaticJoint.Define(stage, joint_path)
                    axis = str(raw_joint.get("axis") or "Z").upper()
                    if axis not in {"X", "Y", "Z"}:
                        raise ValueError(f"SimReady prismatic joint {joint_name} has invalid axis.")
                    joint.CreateAxisAttr(axis)
                    if raw_joint.get("lower_limit") is not None:
                        joint.CreateLowerLimitAttr(float(raw_joint["lower_limit"]))
                    if raw_joint.get("upper_limit") is not None:
                        joint.CreateUpperLimitAttr(float(raw_joint["upper_limit"]))
                else:
                    raise ValueError(
                        f"SimReady policy joint {joint_name} has unsupported type {joint_type}."
                    )
                joint.CreateBody0Rel().SetTargets([body_paths_by_name[body0_name]])
                joint.CreateBody1Rel().SetTargets([body_paths_by_name[body1_name]])
                joint.CreateCollisionEnabledAttr(bool(raw_joint.get("collision_enabled", False)))
                position_values = _normalize_vector3(
                    raw_joint.get("position"),
                    field=f"joint {joint_name} position",
                )
                position_in_default = Gf.Vec3d(*position_values)
                position_in_world = default_to_world.Transform(position_in_default)
                local_positions = []
                local_rotations = []
                for body_name in (body0_name, body1_name):
                    body_prim = stage.GetPrimAtPath(body_paths_by_name[body_name])
                    body_to_world = joint_xform_cache.GetLocalToWorldTransform(body_prim)
                    world_to_body = body_to_world.GetInverse()
                    local_positions.append(world_to_body.Transform(position_in_world))
                    local_rotations.append(
                        _joint_local_rotation(
                            body_to_world,
                            default_to_world,
                            body_name=body_name,
                            Gf=Gf,
                        )
                    )
                joint.CreateLocalPos0Attr(Gf.Vec3f(*(float(value) for value in local_positions[0])))
                joint.CreateLocalPos1Attr(Gf.Vec3f(*(float(value) for value in local_positions[1])))
                joint.CreateLocalRot0Attr(local_rotations[0])
                joint.CreateLocalRot1Attr(local_rotations[1])
                joint_paths.append(joint_path)
                fixes.append(f"author_{joint_type}_joint:{joint_path}")
            if has_moving_joint:
                UsdPhysics.ArticulationRootAPI.Apply(default_prim)
                fixes.append(f"author_articulation_root:{default_prim.GetPath()}")

        guide_path = bucket_paths.get("Guides")
        if guide_path:
            UsdGeom.Imageable(stage.GetPrimAtPath(guide_path)).CreatePurposeAttr(
                UsdGeom.Tokens.guide
            )
            fixes.append(f"mark_guide_geometry:{guide_path}")

        grasp_path = _create_grasp_line(
            stage,
            default_prim,
            dict(policy.get("grasp") or {}),
            fixes,
        )

        generated_at = datetime.now(UTC)
        asset_name = str(policy.get("asset_name") or target.stem)
        profile_name = str(policy.get("profile") or "Prop-Robotics-Neutral")
        profile_version = str(policy.get("profile_version") or "1.0.0")
        source_file = str(policy.get("source_file") or source.name)
        asset_type = str(policy.get("asset_type") or "rigid_prop")
        simready_metadata = {
            "profile": profile_name,
            "profile_version": profile_version,
            "asset_name": asset_name,
            "asset_type": asset_type,
            "source_file": source_file,
            "usd_date_generated": generated_at.date().isoformat(),
            "generator": "cad_verifier.author_simready_package",
            "description": str(policy.get("description") or f"SimReady asset {asset_name}"),
        }
        layer_data = dict(stage.GetRootLayer().customLayerData or {})
        layer_data.update(
            {
                "SimReady_Metadata": simready_metadata,
                "asset_name": asset_name,
                "asset_type": asset_type,
                "source_file": source_file,
                "usd_date_generated": generated_at.date().isoformat(),
            }
        )
        stage.GetRootLayer().customLayerData = layer_data
        fixes.append("author_simready_metadata")

        current_fingerprints: dict[str, str] = {}
        for original_path, new_path in member_mapping.items():
            prim = stage.GetPrimAtPath(new_path)
            fingerprint = _geometry_member_fingerprint(prim)
            if fingerprint is not None:
                current_fingerprints[Path(original_path).name] = fingerprint
        geometry_preserved = source_fingerprints == current_fingerprints
        if not geometry_preserved:
            mismatched_members = sorted(
                member
                for member in set(source_fingerprints) | set(current_fingerprints)
                if source_fingerprints.get(member) != current_fingerprints.get(member)
            )
            raise RuntimeError(
                "SimReady namespace authoring changed source mesh geometry for: "
                + ", ".join(mismatched_members)
            )

        root_layer = stage.GetRootLayer()
        root_layer.Save()
        if not root_layer.Export(str(publish_target)):
            raise RuntimeError(f"Failed to export authored SimReady USD: {publish_target}")
        if not publish_target.is_file() or publish_target.stat().st_size <= 0:
            raise RuntimeError(f"Authored SimReady USD is empty: {publish_target}")
        validation_stage = Usd.Stage.Open(str(publish_target))
        if validation_stage is None:
            raise RuntimeError("Authored SimReady USD could not be reopened for validation.")
        validation_default_prim = validation_stage.GetDefaultPrim()
        if not validation_default_prim or not validation_default_prim.IsValid():
            raise RuntimeError("Authored SimReady USD validation found no valid default prim.")

        sidecar_target.write_text(
            json.dumps(
                {
                    "SimReady_Metadata": simready_metadata,
                    "member_mapping": member_mapping,
                    "body_paths": body_paths,
                    "joint_paths": joint_paths,
                    "collider_paths": collider_paths,
                    "grasp_path": grasp_path,
                    "collision_proxies": collision_proxy_records,
                    "pose_adjustments": pose_adjustment_records,
                    "body_origins": body_origin_records,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        edit_target.unlink(missing_ok=True)
        publish_target.replace(target)
        try:
            sidecar_target.replace(sidecar)
        except Exception:
            target.unlink(missing_ok=True)
            raise
        return {
            "schema": SIMREADY_AUTHORING_SCHEMA,
            "status": "pass",
            "message": f"Authored SimReady package with {len(body_paths)} rigid body/bodies.",
            "source_usd": str(source),
            "artifacts": {
                "simready_usd": str(target),
                "simready_metadata": str(sidecar),
            },
            "profile": profile_name,
            "profile_version": profile_version,
            "geometry_preserved": geometry_preserved,
            "body_paths": body_paths,
            "joint_paths": joint_paths,
            "collider_paths": collider_paths,
            "collider_approximations": collider_approximations,
            "collision_proxies": collision_proxy_records,
            "pose_adjustments": pose_adjustment_records,
            "body_origins": body_origin_records,
            "guide_path": guide_path,
            "grasp_path": grasp_path,
            "member_mapping": member_mapping,
            "fallback_material_bindings": fallback_material_bindings,
            "fallback_visual_material_bindings": fallback_visual_material_bindings,
            "removed_source_api_counts": removed_counts,
            "removed_source_joint_count": len(source_joint_paths),
            "fixes": fixes,
            "policy": _json_ready(policy),
        }
    except Exception as exc:
        cleanup_errors = _remove_failed_artifacts(
            (target, sidecar, edit_target, publish_target, sidecar_target)
        )
        return {
            "schema": SIMREADY_AUTHORING_SCHEMA,
            "status": "fail",
            "message": f"SimReady package authoring failed: {exc}",
            "source_usd": str(source),
            "artifacts": {},
            "fixes": fixes,
            "policy": _json_ready(policy),
            "error": f"{type(exc).__name__}: {exc}",
            "cleanup_errors": cleanup_errors,
        }
