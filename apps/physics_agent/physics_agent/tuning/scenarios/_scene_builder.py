# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parent-side USD scene builders for tune scenarios.

Both scenarios (drop_settle and freeform) share the same skeleton:

  patched physics USD  ──►  derivative scene USD with:
                              • UsdPhysics.Scene (gravity)
                              • Ground plane (collision + friction)
                              • Body translated to its initial pose
                              • Camera(s) framed scale-aware on the body

This module is **parent-side only** — it imports ``pxr`` from the parent's
OpenUSD provider. The ovphysx daemon process never imports this code; it only
reads the resulting ``.usda`` file from disk.

**Stage-unit / up-axis awareness.** The simulator (PhysX via ovphysx)
operates on raw USD numbers — it does not consult ``metersPerUnit`` or
``upAxis`` to convert. So a scene whose USD geometry is in centimeters
needs gravity in cm/s², not m/s², and body placement / ground plane on
the Z axis (not Y) when the stage authors ``upAxis=Z``. This module
honors both:

  * ``GetStageMetersPerUnit`` is read from the source stage; callers
    pass distances and gravity in meters and we scale into stage units
    at the boundary.
  * ``GetStageUpAxis`` is read from the source stage; the body, ground
    plane, and gravity direction all align with that axis.

Result reporting (``bbox_size_m``, ``rest_position``,
``drop_height_m_resolved``) stays in **meters** for the user-facing
API even when the simulator was driven in stage units internally.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

from physics_agent.physics_units import (
    acceleration_m_per_s2_to_stage_units,
    stage_units_per_meter,
    validate_meters_per_unit,
)

if TYPE_CHECKING:
    from pxr import Usd

logger = logging.getLogger(__name__)


def _stage_units_per_meter(stage: Usd.Stage) -> float:
    """Return ``1.0 / metersPerUnit``. Multiplying a
    meter quantity by this gives the equivalent in stage units."""
    from pxr import UsdGeom  # type: ignore[import-untyped]

    return stage_units_per_meter(float(UsdGeom.GetStageMetersPerUnit(stage)))


def _stage_up_axis_index(stage: Usd.Stage) -> int:
    """Return 1 for Y-up, 2 for Z-up. USD only defines those two."""
    from pxr import UsdGeom  # type: ignore[import-untyped]

    return 2 if UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z else 1


def _gravity_direction_for_up_axis(stage: Usd.Stage, gravity: float) -> Any:
    """Build the gravity ``Vec3f`` aligned with the stage's up-axis.

    A negative ``gravity`` value points 'down' along the up-axis (the
    apply_physics convention).
    """
    from pxr import Gf, UsdGeom  # type: ignore[import-untyped]

    sign = -1.0 if gravity < 0.0 else 1.0
    if UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z:
        return Gf.Vec3f(0.0, 0.0, sign)
    return Gf.Vec3f(0.0, sign, 0.0)


def _bake_metric_units(stage: Usd.Stage) -> None:
    """Rewrite the in-memory stage to ``metersPerUnit == 1.0`` by
    rescaling non-instanced mesh data and every existing
    ``xformOp:translate`` / ``xformOp:transform`` by ``mpu``. Native
    instances keep their read-only prototypes and receive an equivalent
    uniform scale on the instance root.

    Background: ovphysx applies gravity in stage units regardless of
    the authored ``physicsGravityMagnitude``, so a centimeter-scale
    source USD ends up with effective gravity ~9.81 cm/s² (100× too
    slow). Rewriting the stage to metric makes the simulator's
    stage-unit gravity equivalent to real Earth gravity.

    A separate parent scale-op is unsuitable because it also scales the
    scenario-authored launch translation. For native instances, the scale is
    instead appended on the instance root while ``_set_body_translation``
    orders the launch translation first. The prototype geometry is converted
    without scaling that metric translation or mutating its source layer.

    Idempotent: a stage that's already metric is left untouched.
    """
    from pxr import Gf, UsdGeom, Vt  # type: ignore[import-untyped]

    mpu = validate_meters_per_unit(float(UsdGeom.GetStageMetersPerUnit(stage)))
    if abs(mpu - 1.0) < 1e-9:
        return

    s = float(mpu)
    for prim in stage.Traverse():
        # Standard stage traversal does not enter instance prototypes, and
        # instance proxies are read-only. Preserve both the source layer and
        # the instance by applying the stage-unit conversion at the instance
        # root instead. ``AddScaleOp`` appends the scale to the existing stack;
        # scenario-authored translations are later ordered first by
        # ``_set_body_translation``, so launch height remains in metric units.
        is_instance = prim.IsInstance()
        if is_instance:
            xformable = UsdGeom.Xformable(prim)
            if not xformable:
                raise ValueError(
                    f"cannot metric-bake non-Xformable instance root {prim.GetPath()}"
                )
            property_names = {str(name) for name in prim.GetPropertyNames()}
            metric_scale_suffix = "wuMetricBake"
            suffix_index = 2
            while f"xformOp:scale:{metric_scale_suffix}" in property_names:
                metric_scale_suffix = f"wuMetricBake{suffix_index}"
                suffix_index += 1
            metric_scale_op = xformable.AddScaleOp(opSuffix=metric_scale_suffix)
            metric_scale_op.Set(Gf.Vec3f(s, s, s))

        if prim.IsInstanceProxy():
            continue
        # Mesh points: rescale every vertex.
        # An instance root can itself be a Mesh; its new root scale already
        # covers those authored points, so rewriting them would scale twice.
        if not is_instance and prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            pts_attr = mesh.GetPointsAttr()
            pts = pts_attr.Get()
            if pts is not None and len(pts) > 0:
                scaled = Vt.Vec3fArray(
                    [Gf.Vec3f(p[0] * s, p[1] * s, p[2] * s) for p in pts]
                )
                pts_attr.Set(scaled)
            # Extent should also rescale.
            ext_attr = mesh.GetExtentAttr()
            ext = ext_attr.Get()
            if ext is not None and len(ext) == 2:
                ext_attr.Set(
                    [
                        Gf.Vec3f(ext[0][0] * s, ext[0][1] * s, ext[0][2] * s),
                        Gf.Vec3f(ext[1][0] * s, ext[1][1] * s, ext[1][2] * s),
                    ]
                )

        # Xformable prims: rescale translation magnitude. We keep
        # rotation and scale ops as-is; only translate magnitudes care
        # about the unit system. ``xformOp:transform`` matrices have
        # their translation column scaled too.
        xformable = UsdGeom.Xformable(prim)
        if xformable:
            for op in xformable.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    v = op.Get()
                    if v is not None:
                        op.Set(type(v)(v[0] * s, v[1] * s, v[2] * s))
                elif op.GetOpType() == UsdGeom.XformOp.TypeTransform:
                    m = op.Get()
                    if m is not None:
                        # Scale the translation row of the 4×4 matrix.
                        new_m = type(m)(m)
                        new_m[3][0] *= s
                        new_m[3][1] *= s
                        new_m[3][2] *= s
                        op.Set(new_m)

    UsdGeom.SetStageMetersPerUnit(stage, 1.0)


def _ground_plane_root_for_body(stage: Usd.Stage, body_prim: Usd.Prim) -> str:
    """Return a stage-root prim path that is **outside** the body_prim
    subtree, suitable as the parent for the ground plane and its
    physics material.

    Rationale (CodeRabbit Round 12 CX P1#1): ``add_ground_plane`` defaults
    its plane and material paths to ``/World/...``. When the patched
    physics USD's stage default-prim is also ``/World`` (the canonical
    SimReady asset layout), authoring under that path makes the plane a
    descendant of the rigid-body parent, so the plane inherits the
    ``UsdPhysics.RigidBodyAPI`` and the asset ends up "welded to its own
    ground" — it never falls onto a static plane.

    The returned path is a top-level Scope sibling of body_prim. It's
    materialised here (as ``/SceneRoot``) so downstream consumers can
    inspect the resulting scene USD without us having to thread the
    string through every callsite.
    """
    from pxr import Sdf, UsdGeom  # type: ignore[import-untyped]

    root = "/SceneRoot"
    body_path = body_prim.GetPath()
    # Defensive: if a SimReady asset has already authored a /SceneRoot,
    # pick a unique sibling so we never collide.
    candidates = [root, "/_Stage", "/_TuneScene", "/_GroundRoot"]
    for candidate in candidates:
        candidate_path = Sdf.Path(candidate)
        if not body_path.HasPrefix(candidate_path) and not candidate_path.HasPrefix(
            body_path
        ):
            existing = stage.GetPrimAtPath(candidate_path)
            if not existing or not existing.IsValid():
                UsdGeom.Scope.Define(stage, candidate_path)
                return candidate
            # Already exists and is outside body_prim — reuse it.
            return candidate
    # Last-ditch fallback: a guaranteed-unique sibling under the root.
    fallback = f"/_TuneScene_{abs(hash(str(body_path))) & 0xFFFFFF:06x}"
    UsdGeom.Scope.Define(stage, Sdf.Path(fallback))
    return fallback


def _stage_default_body_prim(stage: Usd.Stage) -> Usd.Prim:
    """Find the rigid body prim to drive for single-body tune scenes.

    Tune v1 assumes the patched physics USD is already clean single-body
    input. Preserve the historical traversal-order contract: the first
    authored rigid body is the one the simulator, recorder, and metrics
    drive. Multi-body or nested-body inputs should be normalized upstream
    or selected explicitly by future scenario metadata rather than guessed
    here.
    """
    from pxr import UsdPhysics  # type: ignore[import-untyped]

    rigid_bodies = [
        prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    if not rigid_bodies:
        raise ValueError(
            "no prim with UsdPhysics.RigidBodyAPI found in the patched physics "
            "USD; apply_physics must run before tune"
        )
    return rigid_bodies[0]


def _reset_drop_settle_velocities(stage: Usd.Stage) -> None:
    """Author the workflow's gravity-only initial-state contract.

    Source assets may carry stale linear or angular velocities.  A drop-settle
    trial measures response from rest, so every rigid body in the derivative
    scenario must start at zero velocity before the low-level simulator loads it.
    """

    from pxr import Gf, UsdPhysics  # type: ignore[import-untyped]

    zero = Gf.Vec3f(0.0, 0.0, 0.0)
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        rigid_body = UsdPhysics.RigidBodyAPI(prim)
        velocity = rigid_body.CreateVelocityAttr()
        angular_velocity = rigid_body.CreateAngularVelocityAttr()
        # Remove same-layer time samples as well as the default before
        # authoring the stronger scenario initial state.
        velocity.Clear()
        angular_velocity.Clear()
        velocity.Set(zero)
        angular_velocity.Set(zero)


def _bbox_size_stage_units(prim: Usd.Prim) -> tuple[float, float, float]:
    """Bbox size in raw USD stage units along (x, y, z) world axes."""
    bmin, bmax = _bbox_minmax_stage_units(prim)
    return (
        float(bmax[0]) - float(bmin[0]),
        float(bmax[1]) - float(bmin[1]),
        float(bmax[2]) - float(bmin[2]),
    )


def _bbox_minmax_stage_units(
    prim: Usd.Prim,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Bbox min and max in raw USD stage units (world space).

    Useful when the placement code needs to know where the body's
    bottom currently sits on the up-axis — e.g. to translate so that
    bottom lands on the ground, regardless of whether the asset's
    local origin is at the bbox center or at one corner.
    """
    from world_understanding.utils.usd.prim import get_bbox_from_prim

    bbox = get_bbox_from_prim(prim)
    box_range = bbox.ComputeAlignedRange()
    bmin = box_range.GetMin()
    bmax = box_range.GetMax()
    return (
        (float(bmin[0]), float(bmin[1]), float(bmin[2])),
        (float(bmax[0]), float(bmax[1]), float(bmax[2])),
    )


def _bbox_minmax_pose_local_stage_units(
    prim: Usd.Prim,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Bbox min/max in the selected body's unposed local frame.

    Freeform ground-clearance scoring rotates these bbox corners by the
    recorded body pose. The values therefore must not already include the
    selected body's authored translate/rotate/scale pose ops, otherwise a
    transformed asset gets rotated or scaled twice. Descendant transforms stay
    included because they are part of the shape under the selected body.
    """
    from pxr import Usd, UsdGeom, UsdPhysics  # type: ignore[import-untyped]

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    combined = _combined_relative_bound_values(
        prim,
        _acceptance_bound_prims(prim, UsdGeom, UsdPhysics),
        cache,
    )
    if combined is None:
        box_range = cache.ComputeUntransformedBound(prim).ComputeAlignedRange()
        if box_range.IsEmpty():
            return _bbox_minmax_stage_units(prim)
        bmin = box_range.GetMin()
        bmax = box_range.GetMax()
        combined = [
            float(bmin[0]),
            float(bmin[1]),
            float(bmin[2]),
            float(bmax[0]),
            float(bmax[1]),
            float(bmax[2]),
        ]
    if not all(math.isfinite(v) for v in combined):
        return _bbox_minmax_stage_units(prim)
    return (
        (combined[0], combined[1], combined[2]),
        (combined[3], combined[4], combined[5]),
    )


def _combined_relative_bound_values(
    body_prim: Usd.Prim,
    measurement_prims: list[Any],
    cache: Any,
) -> list[float] | None:
    combined: list[float] | None = None
    for measurement_prim in measurement_prims:
        if measurement_prim == body_prim:
            bbox = cache.ComputeUntransformedBound(measurement_prim)
        else:
            bbox = cache.ComputeRelativeBound(measurement_prim, body_prim)
        box_range = bbox.ComputeAlignedRange()
        if box_range.IsEmpty():
            continue
        bmin = box_range.GetMin()
        bmax = box_range.GetMax()
        values = [
            float(bmin[0]),
            float(bmin[1]),
            float(bmin[2]),
            float(bmax[0]),
            float(bmax[1]),
            float(bmax[2]),
        ]
        if not all(math.isfinite(v) for v in values):
            continue
        if combined is None:
            combined = values
        else:
            combined = [
                min(combined[0], values[0]),
                min(combined[1], values[1]),
                min(combined[2], values[2]),
                max(combined[3], values[3]),
                max(combined[4], values[4]),
                max(combined[5], values[5]),
            ]
    return combined


def _acceptance_bound_prims(prim: Usd.Prim, UsdGeom: Any, UsdPhysics: Any) -> list[Any]:
    body_path = str(prim.GetPath())
    colliders: list[Any] = []
    visible_shape_prims: list[Any] = []
    stage = prim.GetStage()
    for candidate in stage.Traverse():
        candidate_path = str(candidate.GetPath())
        if candidate_path != body_path and not candidate_path.startswith(
            f"{body_path}/"
        ):
            continue
        if candidate.HasAPI(UsdPhysics.CollisionAPI):
            enabled = UsdPhysics.CollisionAPI(candidate).GetCollisionEnabledAttr().Get()
            if enabled is not False:
                colliders.append(candidate)
            continue
        if candidate.IsA(UsdGeom.Boundable) and not _is_helper_or_hidden_shape(
            candidate, UsdGeom
        ):
            visible_shape_prims.append(candidate)
    return colliders or visible_shape_prims or [prim]


# Upper bound on the deduplicated vertex cloud held in memory before convex
# hull reduction. Beyond this the builder falls back to conservative bbox
# corners rather than materializing an arbitrarily large point set (and its
# O(N log N) deterministic sort) inside a scene-tool request. Hull reduction
# pays off for dense NON-convex meshes, whose hulls are small; a dense convex
# mesh (a tessellated globe) keeps essentially every vertex on its hull, hits
# the 50k serialization cap, and lands on the conservative bbox path anyway —
# the bound only controls how much transient memory (~100 MB at 400k) is
# spent finding that out.
_MAX_HULL_INPUT_POINTS = 400_000
_MAX_SERIALIZED_SUPPORT_POINTS = 50_000
_GROUND_CLEARANCE_SUPPORT_SCHEMA_VERSION = (
    "physics-agent.ground-clearance-support-decision.v1"
)
_GROUND_CLEARANCE_SUPPORT_POLICY_VERSION = "raw-hull-input-bound.v1"


def _measurement_mesh_points(prim: Usd.Prim, UsdGeom: Any) -> Any:
    """Read a mesh point array behind one testable, non-copying boundary."""

    return UsdGeom.Mesh(prim).GetPointsAttr().Get() or []


def _measurement_prim_signature(
    body_prim: Usd.Prim,
    measurement_prims: list[Any],
) -> list[dict[str, Any]]:
    """Return the cheap authored-state portion of a support-cache identity.

    Geometry bytes are bound by the caller-provided base key.  These fields
    make body selection, target selection, transforms, collision enablement,
    approximation, and stage units explicit so a caller cannot accidentally
    reuse pose-local support points for a different validation body.
    """

    from pxr import Usd, UsdGeom, UsdPhysics  # type: ignore[import-untyped]

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    signature: list[dict[str, Any]] = []
    for prim in measurement_prims:
        relative_transform, _reset = xform_cache.ComputeRelativeTransform(
            prim, body_prim
        )
        approximation: str | None = None
        if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            value = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
            approximation = str(value) if value is not None else None
        collision_enabled: bool | None = None
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            value = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
            collision_enabled = bool(value) if value is not None else None
        signature.append(
            {
                "prim_path": str(prim.GetPath()),
                "type_name": str(prim.GetTypeName()),
                "collision_enabled": collision_enabled,
                "collision_approximation": approximation,
                "relative_transform": [
                    round(float(relative_transform[row][column]), 12)
                    for row in range(4)
                    for column in range(4)
                ],
            }
        )
    return sorted(signature, key=lambda item: str(item["prim_path"]))


def _support_decision_cache_key(
    *,
    base_key: str,
    body_prim: Usd.Prim,
    measurement_signature: list[dict[str, Any]],
    max_points: int,
) -> str:
    from pxr import UsdGeom  # type: ignore[import-untyped]

    stage = body_prim.GetStage()
    payload = {
        "base_key": base_key,
        "body_prim_path": str(body_prim.GetPath()),
        "measurement_prims": measurement_signature,
        "stage_meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(stage)),
        "stage_up_axis": str(UsdGeom.GetStageUpAxis(stage)),
        "policy_version": _GROUND_CLEARANCE_SUPPORT_POLICY_VERSION,
        "processing_bound": _MAX_HULL_INPUT_POINTS,
        "serialization_cap": max_points,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _support_decision(
    *,
    body_prim: Usd.Prim,
    measurement_prims: list[Any],
    max_points: int,
) -> tuple[list[list[float]] | None, dict[str, Any]]:
    """Select ground-clearance measurement support and explain the choice."""

    from pxr import Gf, Usd, UsdGeom, UsdPhysics  # type: ignore[import-untyped]

    started = time.perf_counter()
    body_path = str(body_prim.GetPath())
    measurement_paths = [str(prim.GetPath()) for prim in measurement_prims]
    approximations: set[str] = set()
    raw_point_count = 0
    hull_input_estimate = 0
    point_count_estimate_complete = True
    records: list[tuple[Any, Any, str | None]] = []
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())

    def finish(
        *,
        points: list[list[float]] | None,
        selected_support_type: str,
        reason_code: str,
        reason: str,
        observed_unique_point_count: int | None,
    ) -> tuple[list[list[float]] | None, dict[str, Any]]:
        decision = {
            "schema_version": _GROUND_CLEARANCE_SUPPORT_SCHEMA_VERSION,
            "policy_version": _GROUND_CLEARANCE_SUPPORT_POLICY_VERSION,
            "body_prim_path": body_path,
            "measurement_prim_paths": measurement_paths,
            "authored_collision_approximations": sorted(approximations),
            "selected_support_type": selected_support_type,
            "exact": points is not None,
            "reason_code": reason_code,
            "reason": reason,
            "raw_point_count_estimate": raw_point_count,
            "hull_input_point_count_estimate": hull_input_estimate,
            "point_count_estimate_complete": point_count_estimate_complete,
            "observed_unique_point_count": observed_unique_point_count,
            "processing_bound": _MAX_HULL_INPUT_POINTS,
            "serialization_cap": max_points,
            "selected_point_count": len(points) if points is not None else 8,
            "primary_selection_duration_seconds": time.perf_counter() - started,
        }
        return points, decision

    # Preflight the Vt arrays before constructing transformed Python tuples or
    # invoking QHull.  boundingCube contributes only eight exact AABB corners,
    # so its raw mesh vertices deliberately do not consume the hull-input
    # budget.  Equality with the configured bound remains allowed.
    for prim_index, measurement_prim in enumerate(measurement_prims):
        if not measurement_prim.IsA(UsdGeom.Mesh):
            return finish(
                points=None,
                selected_support_type="conservative_bbox_corners",
                reason_code="non_mesh_collider",
                reason=(
                    f"Measurement prim {measurement_prim.GetPath()} is not a mesh; "
                    "ground clearance uses conservative body bounds."
                ),
                observed_unique_point_count=None,
            )
        approximation: str | None = None
        if measurement_prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            value = (
                UsdPhysics.MeshCollisionAPI(measurement_prim)
                .GetApproximationAttr()
                .Get()
            )
            approximation = str(value) if value is not None else None
            if approximation:
                approximations.add(approximation)
            if approximation == "boundingSphere":
                return finish(
                    points=None,
                    selected_support_type="conservative_bbox_corners",
                    reason_code="bounding_sphere_requires_conservative_support",
                    reason=(
                        "A boundingSphere collider has no finite exact support "
                        "point set; ground clearance uses conservative body bounds."
                    ),
                    observed_unique_point_count=None,
                )
        point_count = len(_measurement_mesh_points(measurement_prim, UsdGeom))
        raw_point_count += point_count
        if approximation != "boundingCube":
            hull_input_estimate += point_count
            if hull_input_estimate > _MAX_HULL_INPUT_POINTS:
                point_count_estimate_complete = prim_index == len(measurement_prims) - 1
                logger.warning(
                    "Ground-clearance support estimate %d exceeded the %d-point "
                    "processing bound; using conservative bbox corners. The "
                    "authored collider is unchanged.",
                    hull_input_estimate,
                    _MAX_HULL_INPUT_POINTS,
                )
                count_description = (
                    "Estimated support input"
                    if point_count_estimate_complete
                    else "Observed support input lower bound"
                )
                return finish(
                    points=None,
                    selected_support_type="conservative_bbox_corners",
                    reason_code="raw_point_estimate_exceeded",
                    reason=(
                        f"{count_description} {hull_input_estimate} exceeds the "
                        f"{_MAX_HULL_INPUT_POINTS}-point processing bound; the "
                        "authored collider remains unchanged."
                    ),
                    observed_unique_point_count=None,
                )
        relative_transform, _reset = xform_cache.ComputeRelativeTransform(
            measurement_prim, body_prim
        )
        # Do not retain every Vt point array while preflighting a multi-mesh
        # body. Re-read one array at a time only after the aggregate is known
        # to fit the processing budget.
        records.append((measurement_prim, relative_transform, approximation))

    unique_points: set[tuple[float, float, float]] = set()
    used_bounding_cube = False
    for measurement_prim, relative_transform, approximation in records:
        points = _measurement_mesh_points(measurement_prim, UsdGeom)
        if approximation == "boundingCube":
            used_bounding_cube = True
            local_min = [math.inf, math.inf, math.inf]
            local_max = [-math.inf, -math.inf, -math.inf]
            finite_point_count = 0
            # The boundingCube path is exempt from the hull-input limit because
            # it emits only eight corners. Compute its extrema in one streaming
            # pass without duplicating the full authored point array.
            for point in points:
                values = tuple(float(point[index]) for index in range(3))
                if not all(math.isfinite(value) for value in values):
                    continue
                finite_point_count += 1
                for index, value in enumerate(values):
                    local_min[index] = min(local_min[index], value)
                    local_max[index] = max(local_max[index], value)
            if finite_point_count == 0:
                continue
            for corner_x in (local_min[0], local_max[0]):
                for corner_y in (local_min[1], local_max[1]):
                    for corner_z in (local_min[2], local_max[2]):
                        transformed = relative_transform.Transform(
                            Gf.Vec3d(corner_x, corner_y, corner_z)
                        )
                        values = tuple(float(transformed[index]) for index in range(3))
                        if all(math.isfinite(value) for value in values):
                            unique_points.add(
                                tuple(round(value, 12) for value in values)
                            )
            continue
        for point in points:
            transformed = relative_transform.Transform(
                Gf.Vec3d(float(point[0]), float(point[1]), float(point[2]))
            )
            values = tuple(float(transformed[index]) for index in range(3))
            if not all(math.isfinite(value) for value in values):
                continue
            unique_points.add(tuple(round(value, 12) for value in values))
            if len(unique_points) > _MAX_HULL_INPUT_POINTS:
                logger.warning(
                    "Ground-clearance support cloud exceeded the %d-point "
                    "processing bound after deduplication; using conservative "
                    "bbox corners. The authored collider is unchanged.",
                    _MAX_HULL_INPUT_POINTS,
                )
                return finish(
                    points=None,
                    selected_support_type="conservative_bbox_corners",
                    reason_code="unique_point_bound_exceeded",
                    reason=(
                        f"Observed unique support count exceeded the "
                        f"{_MAX_HULL_INPUT_POINTS}-point processing bound; the "
                        "authored collider remains unchanged."
                    ),
                    observed_unique_point_count=len(unique_points),
                )

    if not unique_points:
        return finish(
            points=None,
            selected_support_type="conservative_bbox_corners",
            reason_code="no_finite_support_points",
            reason=(
                "No finite collider support points were available; ground "
                "clearance uses conservative body bounds."
            ),
            observed_unique_point_count=0,
        )
    if len(unique_points) > max_points:
        reduced, reduction_reason = _convex_hull_vertices_with_reason(
            unique_points, max_points=max_points
        )
        if reduced is None:
            code = reduction_reason or "convex_hull_reduction_failed"
            return finish(
                points=None,
                selected_support_type="conservative_bbox_corners",
                reason_code=code,
                reason=(
                    f"Convex-hull reduction could not produce at most "
                    f"{max_points} support points ({code}); ground clearance "
                    "uses conservative body bounds and the authored collider "
                    "is unchanged."
                ),
                observed_unique_point_count=len(unique_points),
            )
        return finish(
            points=reduced,
            selected_support_type="convex_hull_vertices",
            reason_code="convex_hull_reduction",
            reason=(
                f"Reduced {len(unique_points)} unique collider vertices to "
                f"{len(reduced)} exact convex-hull support vertices."
            ),
            observed_unique_point_count=len(unique_points),
        )
    selected = [list(point) for point in sorted(unique_points)]
    return finish(
        points=selected,
        selected_support_type=(
            "bounding_cube_corners"
            if used_bounding_cube
            and all(record[2] == "boundingCube" for record in records)
            else "mesh_vertices"
        ),
        reason_code="exact_support_points",
        reason=f"Selected {len(selected)} exact collider support points.",
        observed_unique_point_count=len(unique_points),
    )


def _select_ground_clearance_support(
    body_prim: Usd.Prim,
    measurement_prims: list[Any],
    *,
    max_points: int = _MAX_SERIALIZED_SUPPORT_POINTS,
    support_cache: dict[str, dict[str, Any]] | None = None,
    support_cache_base_key: str | None = None,
) -> tuple[list[list[float]] | None, dict[str, Any]]:
    """Return support points plus a structured, optionally reused decision."""

    lookup_started = time.perf_counter()
    cache_key = None
    if support_cache is not None and support_cache_base_key:
        signature = _measurement_prim_signature(body_prim, measurement_prims)
        cache_key = _support_decision_cache_key(
            base_key=support_cache_base_key,
            body_prim=body_prim,
            measurement_signature=signature,
            max_points=max_points,
        )
        cached = support_cache.get(cache_key)
        if (
            isinstance(cached, dict)
            and isinstance(cached.get("decision"), dict)
            and cached["decision"].get("schema_version")
            == _GROUND_CLEARANCE_SUPPORT_SCHEMA_VERSION
            and cached["decision"].get("policy_version")
            == _GROUND_CLEARANCE_SUPPORT_POLICY_VERSION
        ):
            cached["reuse_count"] = int(cached.get("reuse_count") or 0) + 1
            decision = deepcopy(cached["decision"])
            decision.update(
                {
                    "cache_status": "reused",
                    "reuse_count": cached["reuse_count"],
                    "selection_duration_seconds": time.perf_counter() - lookup_started,
                }
            )
            points = deepcopy(cached.get("geometry_points_local_stage"))
            logger.info(
                "Reused ground-clearance support decision %s for body %s.",
                decision.get("decision_id"),
                body_prim.GetPath(),
            )
            return points, decision

    points, decision = _support_decision(
        body_prim=body_prim,
        measurement_prims=measurement_prims,
        max_points=max_points,
    )
    decision_id_payload = {
        "cache_key": cache_key,
        "body_prim_path": decision["body_prim_path"],
        "measurement_prim_paths": decision["measurement_prim_paths"],
        "policy_version": decision["policy_version"],
        "reason_code": decision["reason_code"],
    }
    decision["decision_id"] = hashlib.sha256(
        json.dumps(decision_id_payload, sort_keys=True).encode()
    ).hexdigest()
    decision["cache_key"] = cache_key
    decision["cache_status"] = "primary"
    decision["reuse_count"] = 0
    decision["selection_duration_seconds"] = time.perf_counter() - lookup_started
    if support_cache is not None and cache_key is not None:
        support_cache[cache_key] = {
            "decision": deepcopy(decision),
            "geometry_points_local_stage": deepcopy(points),
            "reuse_count": 0,
        }
    return points, decision


def _acceptance_mesh_points_pose_local_stage_units(
    body_prim: Usd.Prim,
    measurement_prims: list[Any],
    *,
    max_points: int = 50_000,
) -> list[list[float]] | None:
    """Return exact support vertices in the selected body's unposed frame.

    Rotating an axis-aligned bbox around an oblique or curved collider can put
    empty bbox corners below the ground and report false penetration. Mesh
    vertices are exact or near-exact support candidates for every
    mesh-derived collision approximation (triangle mesh, convex hull,
    convex decomposition, simplification, SDF) — the simulated shape tracks
    the source vertices. ``boundingCube`` simulates the mesh's local AABB,
    so its eight corners ARE the exact support points and are emitted in
    place of the vertices. The one true exception is ``boundingSphere``:
    the simulated sphere circumscribes the mesh and dips below any finite
    point set between corners, so no vertex-derived support set is exact
    for it. When the vertex count exceeds ``max_points``, reduce to the
    convex hull instead of giving up: for any direction, the extreme
    (lowest) vertex is always a hull vertex, so the reduction is lossless
    for ground-clearance scoring while keeping the serialized point set
    compact. Return ``None`` when the collider contains non-mesh shapes or
    a boundingSphere approximation, the cloud exceeds the in-memory
    processing bound, or hull reduction is
    unavailable/degenerate/over the serialization cap; the caller then
    measures against conservative bbox corners under the absolute default
    limit and records a diagnostic.
    """
    points, _decision = _select_ground_clearance_support(
        body_prim,
        measurement_prims,
        max_points=max_points,
    )
    return points


def _convex_hull_vertices(
    points: set[tuple[float, float, float]],
    *,
    max_points: int,
) -> list[list[float]] | None:
    """Reduce a vertex cloud to its convex hull vertices, or None on failure.

    ``QJ`` joggles degenerate (coplanar/collinear) inputs so thin shells and
    flat plates still produce a hull instead of raising QhullError. Every
    ``None`` path logs its cause: the caller's conservative-bbox fallback is
    behaviorally identical for a degenerate coplanar mesh and for an
    out-of-memory hull build, and triage needs to tell them apart.
    """
    vertices, _reason = _convex_hull_vertices_with_reason(points, max_points=max_points)
    return vertices


def _convex_hull_vertices_with_reason(
    points: set[tuple[float, float, float]],
    *,
    max_points: int,
) -> tuple[list[list[float]] | None, str | None]:
    """Reduce to hull vertices and return a stable failure reason code."""

    try:
        import numpy
        from scipy.spatial import ConvexHull  # type: ignore[import-untyped]
    except ImportError:
        logger.warning(
            "Convex-hull reduction unavailable (scipy/numpy not importable); "
            "falling back to conservative bbox corners."
        )
        return None, "convex_hull_unavailable"
    # The whole allocation-and-reduction sequence is guarded, not only the
    # ConvexHull call: for clouds near the input bound, the list/array/unique
    # allocations can raise MemoryError first, and that must degrade to the
    # documented conservative-bbox fallback instead of aborting validation.
    try:
        # numpy.unique sorts lexicographically (rows compared element-wise),
        # the same deterministic order as sorting the tuples in Python but
        # in C.
        array = numpy.unique(
            numpy.asarray(list(points), dtype=numpy.float64).reshape(-1, 3), axis=0
        )
        hull = ConvexHull(array, qhull_options="QJ")
        vertices = array[numpy.sort(hull.vertices)]
        if len(vertices) > max_points:
            logger.warning(
                "Convex hull kept %d vertices, above the %d serialization "
                "cap; falling back to conservative bbox corners.",
                len(vertices),
                max_points,
            )
            return None, "convex_hull_serialization_cap_exceeded"
        return [[float(v) for v in row] for row in vertices], None
    except MemoryError as error:
        logger.warning(
            "Convex-hull reduction of %d vertices ran out of memory (%s); "
            "falling back to conservative bbox corners.",
            len(points),
            error,
        )
        return None, "convex_hull_memory_error"
    except Exception as error:  # noqa: BLE001 - fallback path logs the class
        logger.warning(
            "Convex-hull reduction of %d vertices failed (%s: %s); falling "
            "back to conservative bbox corners.",
            len(points),
            type(error).__name__,
            error,
        )
        return None, "convex_hull_reduction_failed"


def _is_helper_or_hidden_shape(prim: Usd.Prim, UsdGeom: Any) -> bool:
    name = prim.GetName().lower()
    if any(
        token in name
        for token in ("bbox", "bounding", "bounds", "helper", "guide", "debug")
    ):
        return True
    if prim.IsA(UsdGeom.Imageable):
        imageable = UsdGeom.Imageable(prim)
        if imageable.ComputeVisibility() == UsdGeom.Tokens.invisible:
            return True
        if imageable.ComputePurpose() == UsdGeom.Tokens.guide:
            return True
    return False


def _body_pose_scale(prim: Usd.Prim) -> tuple[float, float, float]:
    """Return effective scale from the selected body to stage space."""

    from pxr import Usd, UsdGeom  # type: ignore[import-untyped]

    if not prim.IsA(UsdGeom.Xformable):
        return (1.0, 1.0, 1.0)
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    scale = []
    for row in range(3):
        axis_scale = math.sqrt(
            float(matrix[row][0]) ** 2
            + float(matrix[row][1]) ** 2
            + float(matrix[row][2]) ** 2
        )
        scale.append(axis_scale)
    if not all(math.isfinite(value) and value > 0.0 for value in scale):
        return (1.0, 1.0, 1.0)
    return (scale[0], scale[1], scale[2])


def _bbox_size_meters(prim: Usd.Prim) -> tuple[float, float, float]:
    """Bbox size in METERS along (x, y, z) world axes.

    Honors the stage's ``metersPerUnit`` so a centimeter-scale source
    USD doesn't get reported as a 100× larger scene. Used for the
    user-facing ``bbox_size_m`` result field; the simulator itself
    consumes stage units (see :func:`_bbox_size_stage_units`).
    """
    sx, sy, sz = _bbox_size_stage_units(prim)
    stage = prim.GetStage()
    if stage is None:
        return (sx, sy, sz)
    from pxr import UsdGeom  # type: ignore[import-untyped]

    mpu = validate_meters_per_unit(float(UsdGeom.GetStageMetersPerUnit(stage)))
    return (sx * mpu, sy * mpu, sz * mpu)


def _get_body_translation(prim: Usd.Prim) -> tuple[float, float, float]:
    """Read the current translate op on ``prim``. Returns (0, 0, 0) when
    no translate op is authored.

    Companion to :func:`_set_body_translation`. The bbox-min-aware
    placement paths in drop_settle and freeform compute a target
    world-space bottom position. Since the body's bbox already
    reflects any pre-existing translate, computing the new translate as
    ``current_translate + (target - current_world_bottom)`` preserves
    the body's authored offset — replacing the translate op outright
    (the historical bug kimbyn flagged 2026-05-12) placed pre-translated
    assets below the ground plane.
    """
    from pxr import UsdGeom  # type: ignore[import-untyped]

    xformable = UsdGeom.Xformable(prim)
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            v = op.Get()
            if v is None:
                return (0.0, 0.0, 0.0)
            return (float(v[0]), float(v[1]), float(v[2]))
    return (0.0, 0.0, 0.0)


def _set_body_translation(
    prim: Usd.Prim, translate: tuple[float, float, float]
) -> None:
    """Replace any existing translate op on a rigid body with the given
    (x, y, z). Preserves other xformOps (rotation, etc.).

    drop_settle authors translate-only; freeform authors translate +
    optional rotation."""
    from pxr import Gf, UsdGeom  # type: ignore[import-untyped]

    xformable = UsdGeom.Xformable(prim)
    ops = xformable.GetOrderedXformOps()
    translate_op = None
    for op in ops:
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translate_op = op
            break
    if translate_op is None:
        translate_op = xformable.AddTranslateOp()
        # Maintain the existing op order; AddTranslateOp appends, but we
        # want translate first for simplicity.
        new_ops = [translate_op] + [op for op in ops if op != translate_op]
        xformable.SetXformOpOrder(new_ops)
    translate_op.Set(Gf.Vec3d(*translate))


def _set_body_rotation(
    prim: Usd.Prim, rotation_xyz_radians: tuple[float, float, float]
) -> None:
    """Replace any existing rotate op on the body with an XYZ Euler
    rotation in radians.

    Used by freeform scenarios where ``target.initial_pose.rotation``
    is supplied. drop_settle skips rotation authoring."""
    import math

    from pxr import Gf, UsdGeom  # type: ignore[import-untyped]

    xformable = UsdGeom.Xformable(prim)
    ops = xformable.GetOrderedXformOps()
    rotate_op = None
    # USD only defines six Tait-Bryan rotate triples (no axis repeats);
    # the previous list included the bogus ``TypeRotateXYX`` which would
    # AttributeError the moment a freeform target supplied a rotation.
    rotate_op_types = (
        UsdGeom.XformOp.TypeRotateXYZ,
        UsdGeom.XformOp.TypeRotateXZY,
        UsdGeom.XformOp.TypeRotateYXZ,
        UsdGeom.XformOp.TypeRotateYZX,
        UsdGeom.XformOp.TypeRotateZXY,
        UsdGeom.XformOp.TypeRotateZYX,
    )
    for op in ops:
        if op.GetOpType() in rotate_op_types:
            rotate_op = op
            break
    if rotate_op is None:
        rotate_op = xformable.AddRotateXYZOp()
    # USD rotateXYZ expects degrees.
    rx, ry, rz = (math.degrees(float(v)) for v in rotation_xyz_radians)
    rotate_op.Set(Gf.Vec3f(rx, ry, rz))


def _author_physics_scene(stage: Usd.Stage, *, gravity_m_per_s2: float) -> None:
    """Author a UsdPhysics.Scene at /PhysicsScene aligned with the
    stage's up-axis.

    The direction vector aligns with ``UsdGeom.GetStageUpAxis``: ``-Z``
    for Z-up stages, ``-Y`` for Y-up. ``gravity_m_per_s2`` is converted
    from SI to stage acceleration units before authoring. The built-in scene
    builders first call ``_bake_metric_units``, so their conversion is a
    no-op while gravity, body placement, and rest_position remain on one
    self-consistent metric basis.

    A negative ``gravity_m_per_s2`` points 'down' along the up-axis
    (matches the apply_physics convention).

    Any pre-existing ``UsdPhysics.Scene`` prims that are NOT at
    ``/PhysicsScene`` are deactivated so the harness owns the single
    authoritative scene for the simulation. This matters for assets
    whose ``apply_physics`` output nests its own ``PhysicsScene`` under
    a wrapper default prim (e.g. Isaac SimReady ``/RootNode`` assets):
    without dedup, the simulator ends up with two scenes — the asset's
    nested one and the harness's stage-root one — and the rigid body
    binds to a different scene from the GroundPlane, so the body never
    collides with the ground and falls indefinitely.
    """
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics  # type: ignore[import-untyped]

    target_path = Sdf.Path("/PhysicsScene")
    # ``TraverseAll`` does NOT descend into instance proxies, so a
    # nested ``UsdPhysics.Scene`` under an instanced subtree would
    # silently slip past the dedup loop. Use the instance-proxy-aware
    # range with the all-prims predicate (so inactive prims are also
    # visited — an explicitly-inactive scene needs no further action,
    # but checking ``IsActive`` below is still cheap and lets the
    # ``IsInstanceProxy`` branch own its own warning).
    proxy_traversal = Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)
    for prim in Usd.PrimRange.Stage(stage, proxy_traversal):
        if prim.IsA(UsdPhysics.Scene) and prim.GetPath() != target_path:
            if prim.IsInstanceProxy():
                # Instance proxies are read-only views into a prototype;
                # ``SetActive(False)`` on a proxy is a silent no-op in
                # USD, so we'd believe we'd deactivated it but the prim
                # would still arbitrate alongside the harness scene at
                # simulation time. The proper fix is to deinstance
                # upstream (the scene_builder's input is expected to be
                # deinstanced; non-deinstanced wrappers like Isaac
                # SimReady need ``apply_physics`` to author into the
                # uninstanced layer first). Warn loudly so a regression
                # in that contract surfaces in logs even when bodies
                # appear to fall correctly.
                logger.warning(
                    "Skipping instance-proxy UsdPhysics.Scene at %s "
                    "(read-only — cannot SetActive(False)). Remediation: "
                    "enable upstream deinstancing — for the physics-agent "
                    "CLI, set ``optimize_usd.enabled: true`` with "
                    "``scene_optimizer_settings.enable_deinstance: true`` "
                    "in your pipeline config so the materialized USD is "
                    "deinstanced before apply_physics runs; for direct "
                    "scene_builder callers, deinstance the patched USD "
                    "before invoking build_drop_settle_scene / "
                    "build_freeform_scene. Without this, the asset's "
                    "nested scene may still arbitrate alongside the "
                    "harness scene at simulation time even when bodies "
                    "appear to fall correctly.",
                    prim.GetPath(),
                )
                continue
            if prim.IsActive():
                prim.SetActive(False)
                logger.debug(
                    "Deactivated pre-existing UsdPhysics.Scene at %s "
                    "to defer to harness-owned %s",
                    prim.GetPath(),
                    target_path,
                )

    direction = _gravity_direction_for_up_axis(stage, gravity_m_per_s2)
    scene = UsdPhysics.Scene.Define(stage, target_path)
    scene.CreateGravityDirectionAttr().Set(direction)
    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    gravity_stage_units = acceleration_m_per_s2_to_stage_units(
        abs(float(gravity_m_per_s2)),
        meters_per_unit,
    )
    scene.CreateGravityMagnitudeAttr().Set(gravity_stage_units)


def _author_cameras(
    stage: Usd.Stage,
    body_prim: Usd.Prim,
    directions: list[str],
    *,
    ground_bias_fraction: float | None = None,
) -> list[str]:
    """Author one focused camera per direction. Returns the camera
    prim paths.

    Direction tokens are routed to the matching helper:

    * Cardinal axes (``+x``, ``-x``, ..., ``-z``) → side-view helper.
      The previous default ``"-z"`` produces a *top-down* view in a
      Z-up scene because Z is the up-axis there — useful only for
      orthographic top renders, not for visualising a fall.
    * Corner triples (``+x+y+z``, ``-x-y-z``, ``+x-0.5y+z``, ...) →
      ``add_focused_corner_view_camera``, which gives a tilted
      isometric perspective showing all three axes. This matches
      material_agent's render-task convention
      (``apps/material_agent/material_agent/tasks/render.py``,
      ``camera_corners=["+x+y+z","-x-y-z"]``) and works correctly
      regardless of the stage's up-axis since the direction is
      body-local.

    drop_settle / freeform default to ``["+x+y+z"]`` so the recording
    is visually meaningful end-to-end. Callers can still request
    cardinal views explicitly when they want orthographic frames.

    ``ground_bias_fraction`` (0.0–1.0) lerps the camera's look-at on
    the stage's up-axis from the body's bbox center (0.0) toward the
    ground plane at stage-up coord 0 (1.0). Default ``None`` preserves
    the bbox-center look-at — callers who want the ground higher in
    the frame (e.g. ``drop_settle`` runs where the body falls offscreen
    by mid-trial) pass ``0.5``–``0.8``. Non-up-axis coordinates of the
    look-at are unchanged so the body stays framed horizontally. YAML
    or LLM-authored strings are coerced via ``float()`` so a quoted
    ``"0.75"`` is accepted; non-numeric values raise ``ValueError``.

    For cardinal directions (``+x`` etc.), the side-view camera helper
    does not accept ``target_x/y/z`` overrides, so the bias is dropped
    on the cardinal path and a single ``WARNING`` is logged per
    ``_author_cameras`` call (not per cardinal direction). The
    single-warning ergonomic prevents spam in scenarios that author
    several cardinal cameras (e.g. ``["+x", "+y", "+z"]`` for
    orthographic side views) while still surfacing the dropped bias
    once. Per-direction warnings are intentionally avoided to keep log
    volume low for multi-camera scenarios.
    """
    from world_understanding.utils.usd.camera import (
        add_focused_corner_view_camera,
        add_focused_side_view_camera,
    )

    # Pre-compute the up-axis look-at override (if requested) once
    # per call: the body's bbox is the same for every direction.
    target_kwargs: dict[str, float] = {}
    if ground_bias_fraction is not None:
        # Reject ``bool`` explicitly — ``isinstance(True, int) is True`` in
        # Python and ``float(True) == 1.0`` / ``float(False) == 0.0``,
        # both of which pass the [0.0, 1.0] bounds check below but match
        # neither plausible caller intent (``True`` ≠ "turn on bias",
        # ``False`` ≠ "no bias"). Fail loud so LLM-authored ``true``
        # tokens surface as configuration errors instead of running with
        # camera aimed at the ground.
        if isinstance(ground_bias_fraction, bool):
            raise ValueError(
                "ground_bias_fraction must be a float in [0.0, 1.0]; "
                f"got bool {ground_bias_fraction!r} — bool is rejected to "
                "avoid silent float(True)=1.0 / float(False)=0.0 coercion"
            )
        # Coerce strings to float so YAML quoted values and LLM-authored
        # ``"0.75"`` strings are accepted; non-numeric values raise the
        # same ``ValueError`` as out-of-range numerics so callers get a
        # consistent error type.
        try:
            ground_bias_fraction = float(ground_bias_fraction)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"ground_bias_fraction must be a float in [0.0, 1.0]; "
                f"got {ground_bias_fraction!r}"
            ) from exc
        if not 0.0 <= ground_bias_fraction <= 1.0:
            raise ValueError(
                "ground_bias_fraction must be in [0.0, 1.0]; "
                f"got {ground_bias_fraction}"
            )
        from pxr import Usd as _Usd  # type: ignore[import-untyped]
        from pxr import UsdGeom  # type: ignore[import-untyped]

        cache = UsdGeom.BBoxCache(_Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        bbox = cache.ComputeWorldBound(body_prim).ComputeAlignedRange()
        up_idx = _stage_up_axis_index(stage)
        body_center_up = (bbox.GetMin()[up_idx] + bbox.GetMax()[up_idx]) / 2.0
        # Ground plane is authored at stage-up coord 0. Contract:
        # ``build_drop_settle_scene`` and ``build_freeform_scene`` both
        # call ``add_ground_plane(center=(0.0, 0.0, 0.0), ...)``
        # unconditionally, and ``add_ground_plane`` (in
        # ``world_understanding.utils.usd.scene``) places the plane at
        # the given center on the stage's up-axis. If a future scenario
        # ever wants an elevated ground (drop-onto-platform), this lerp
        # would need to read the actual plane's up-coord from the
        # composed stage instead of assuming 0.0.
        ground_up = 0.0
        target_up = body_center_up + ground_bias_fraction * (ground_up - body_center_up)
        target_kwargs[("target_x", "target_y", "target_z")[up_idx]] = float(target_up)

    # A direction is a "corner" if it names ≥ 2 distinct axes. Cardinal
    # directions name exactly one. Tokens like "+x-0.5y+z" count as
    # corner because they span more than one axis — the corner helper
    # accepts that fractional form.
    def _is_corner(direction: str) -> bool:
        axes = {c for c in direction.lower() if c in ("x", "y", "z")}
        return len(axes) >= 2

    def _sanitize_prim_name(direction: str) -> str:
        """Build a USD-safe prim name from a direction string.

        Documented direction strings like ``+x-0.5y+z`` (fractional axes
        for diagonal weighting) contain the literal ``.`` character,
        which is not legal in USD prim names — ``Sdf.Path`` rejects them
        as ill-formed. Map ``+`` / ``-`` to readable prefixes and replace
        every other non-alphanumeric character with ``_`` so we always
        produce a valid prim name. (kimbyn 2026-05-12)

        Round 20 adversarial follow-up: a non-leading-sign direction like
        ``0.5x+0.5y+0.5z`` (no ``+`` / ``-`` prefix) would sanitize to
        ``0_5xplus_0_5yplus_0_5z`` — Sdf rejects prim names starting with
        a digit. The documented format always uses a leading sign, but
        LLM-authored scenarios (`interpret_user_prompt_tuning`,
        `scenario_refine`) can hallucinate anything, so prefix an
        underscore when the sanitized output would start with a digit
        rather than rely on the caller honoring the format.
        """
        s = direction.replace("+", "plus_").replace("-", "minus_")
        sanitized = "".join(c if c.isalnum() or c == "_" else "_" for c in s)
        # Collapse repeated underscores for readability.
        while "__" in sanitized:
            sanitized = sanitized.replace("__", "_")
        sanitized = sanitized.strip("_") or "axis"
        if sanitized[0].isdigit():
            sanitized = "_" + sanitized
        return sanitized

    # ``add_focused_side_view_camera`` does NOT accept ``target_x/y/z``
    # (only the corner helper does), so the bias kwargs cannot ride
    # along the cardinal path without TypeError. Surface that the bias
    # is silently inapplicable for cardinal directions, but only warn
    # once per call regardless of how many cardinal directions appear.
    warned_side_bias_ignored = False
    paths: list[str] = []
    for d in directions:
        cam_path = f"/Cameras/{_sanitize_prim_name(d)}"
        if _is_corner(d):
            add_focused_corner_view_camera(
                prim_to_focus=body_prim,
                camera_path=cam_path,
                direction=d,
                margin=1.4,
                near_clip=0.01,
                far_clip=100000.0,
                **target_kwargs,
            )
        else:
            if target_kwargs and not warned_side_bias_ignored:
                logger.warning(
                    "ground_bias_fraction=%s is ignored for cardinal "
                    "camera direction %r (only corner directions like "
                    "'+x+y+z' support look-at overrides — the side-view "
                    "helper has no target_x/y/z parameter)",
                    ground_bias_fraction,
                    d,
                )
                warned_side_bias_ignored = True
            add_focused_side_view_camera(
                prim_to_focus=body_prim,
                camera_path=cam_path,
                direction=d,
                margin=1.4,
                near_clip=0.01,
                far_clip=100000.0,
            )
        paths.append(cam_path)
    return paths


def _export_scene_portably(
    scene_stage: Usd.Stage,
    patched_physics_usd: Path,
    output_scene_usd: Path,
    approved_dependency_roots: Iterable[str | Path] | None,
) -> None:
    """Publish a composed simulation scene without flattening its instances.

    The scenario additions are authored into the source stage's root layer.
    Exporting that layer directly to a different directory leaves relative
    sublayers, references, payloads, and package-member paths anchored at the
    old location. Flattening would keep the composed geometry but materialize
    instance proxies, changing the scene the runtime is supposed to validate.

    Reuse the resolver-backed portable exporter instead. Passing the root layer
    explicitly preserves authored composition, while the exporter copies and
    re-anchors the complete dependency closure into a sibling sidecar. The
    approved roots are a required trust boundary for copied filesystem data;
    direct callers default to the input asset's directory, and service callers
    can provide the narrower session/input roots they already trust.
    """
    from world_understanding.functions.graphics.so_export import (
        export_stage_portably,
    )

    from physics_agent.functions.apply_physics import (
        _is_runtime_resolved_asset_path,
    )

    dependency_roots: Iterable[str | Path] = (
        approved_dependency_roots
        if approved_dependency_roots is not None
        else (patched_physics_usd.parent,)
    )
    export_stage_portably(
        scene_stage,
        output_scene_usd,
        approved_dependency_roots=dependency_roots,
        export_layer=scene_stage.GetRootLayer(),
        is_runtime_asset_path=_is_runtime_resolved_asset_path,
    )


def build_drop_settle_scene(
    patched_physics_usd: Path,
    output_scene_usd: Path,
    *,
    drop_height_m: float | None = None,
    placement_mode: str = "drop",
    gravity: float = -9.81,
    ground_friction: float = 0.5,
    cameras: list[str] | None = None,
    camera_ground_bias_fraction: float | None = None,
    body_prim_path_hint: str | None = None,
    body_pattern_hint: str | None = None,
    placement_prim_path_hint: str | None = None,
    approved_dependency_roots: Iterable[str | Path] | None = None,
    ground_clearance_support_cache: dict[str, dict[str, Any]] | None = None,
    ground_clearance_support_cache_key: str | None = None,
) -> dict[str, Any]:
    """Build a drop-settle or explicitly mounted simulation scene.

    The body's bbox-min on the **stage's up-axis** (Y or Z, read from
    ``UsdGeom.GetStageUpAxis``) is translated to ``drop_height_m``.
    **``drop_height_m`` is the GAP between the body's bottom and the
    ground, NOT the body's absolute coordinate.** When ``None`` it
    defaults to the body's own height.

    Stage-unit awareness: ``drop_height_m`` and ``gravity`` are passed
    in the user-facing units (meters and m/s²); the scene builder
    converts both into the stage's native units before authoring so
    the simulator runs self-consistently on a centimeter-scale USD.

    Args:
        patched_physics_usd: Output of
            :func:`physics_agent.tuning.usd_patch.patch_physics_usd`
            (the body has UsdPhysics schemas, mass, friction, etc.).
        output_scene_usd: Path to write the derivative scene to.
        drop_height_m: Gap (in meters) between body bottom and ground.
            Defaults to the body's bbox_height (so the body sits one
            own-height above the ground).
        placement_mode: ``drop`` preserves the existing ground-placement
            behavior. ``mounted`` preserves authored metric transforms and
            joint frames, including world anchors. It accepts no drop gap
            other than zero and checks settling around the initial body pose.
        gravity: Signed gravity in m/s² (negative = downward).
        ground_friction: Static + dynamic friction on the ground plane
            (0..2 typical). 0.5 is a reasonable concrete-like default.
        cameras: List of cardinal directions for cameras (default
            ``["+x+y+z"]`` — single tilted corner view; works for
            both Y-up and Z-up assets).
        camera_ground_bias_fraction: Optional 0.0–1.0 bias for the
            camera's look-at on the up-axis. ``None`` (default) keeps
            the look-at at the body's bbox center; ``0.75`` shifts it
            75% of the way from the center toward the ground, giving
            the falling body more vertical room above it in the frame.
        body_prim_path_hint: Optional rigid-body prim to track. The
            traversal-order default remains unchanged when omitted.
        body_pattern_hint: Optional ovphysx tensor-binding pattern.
            Defaults to the selected rigid-body path.
        placement_prim_path_hint: Optional common ancestor to translate
            onto the ground. Use this for multibody assets so relative
            body poses and joint frames remain unchanged.
        approved_dependency_roots: Filesystem roots whose USD layers and
            assets may be copied into the portable simulation-scene sidecar.
            Defaults to ``patched_physics_usd.parent``. Service callers should
            pass their trusted session/input roots explicitly.
        ground_clearance_support_cache: Optional parent-owned in-memory cache.
            Cached support points are never loaded from a run artifact.
        ground_clearance_support_cache_key: Caller-owned stable identity that
            binds the immutable source geometry and collider decision. Reuse is
            disabled when omitted.

    Returns:
        Dict with ``body_prim_path`` (USD path to the rigid body),
        ``body_pattern`` (ovphysx-style pattern equal to the prim path),
        ``rest_position`` (expected resting position [x, y, z]),
        ``drop_height_m_resolved`` (the actual gap used),
        ``bbox_size_m`` (body bbox size in world meters),
        ``camera_paths`` (list of authored camera prim paths).
    """
    scene_build_started = time.perf_counter()
    from pxr import Usd, UsdGeom, UsdPhysics  # type: ignore[import-untyped]
    from world_understanding.functions.graphics.so_export import (
        _normalize_dependency_roots,
    )
    from world_understanding.utils.usd.scene import add_ground_plane

    if placement_mode not in {"drop", "mounted"}:
        raise ValueError("placement_mode must be drop or mounted")
    if placement_mode == "mounted" and drop_height_m not in (None, 0.0):
        raise ValueError("mounted placement requires drop_height_m to be zero or omitted")

    dependency_roots = _normalize_dependency_roots(
        tuple(approved_dependency_roots)
        if approved_dependency_roots is not None
        else (Path(patched_physics_usd).resolve().parent,)
    )
    # Open the patched USD into an in-memory stage. We add ground +
    # camera + UsdPhysics.Scene + body translation in memory, then
    # export to ``output_scene_usd`` without saving the original
    # patched file. This avoids the sublayer-resolution edge cases
    # that bit us when using subLayerPaths.
    scene_stage = Usd.Stage.Open(str(patched_physics_usd))
    if scene_stage is None:
        raise ValueError(f"could not open patched physics USD {patched_physics_usd}")

    # Rewrite the in-memory stage to metric so PhysX's stage-unit gravity
    # equals real Earth m/s². Without this a centimeter-scale source
    # USD ends up with effective gravity 100× too slow.
    if placement_mode == "mounted":
        # The legacy metric baker changes geometry/xforms but does not rebase
        # all dimensional joint/mass properties. Never move world anchors or
        # silently change a mounted mechanism's coordinate system.
        if not math.isclose(
            float(UsdGeom.GetStageMetersPerUnit(scene_stage)),
            1.0,
            rel_tol=0,
            abs_tol=1e-9,
        ):
            raise ValueError("mounted placement requires metersPerUnit=1")
    else:
        _bake_metric_units(scene_stage)

    if body_prim_path_hint:
        body_prim = scene_stage.GetPrimAtPath(body_prim_path_hint)
        if not body_prim or not body_prim.IsValid():
            raise ValueError(
                f"body_prim_path_hint {body_prim_path_hint!r} was not found in the patched USD"
            )
    else:
        body_prim = _stage_default_body_prim(scene_stage)
    if placement_prim_path_hint:
        placement_prim = scene_stage.GetPrimAtPath(placement_prim_path_hint)
        if not placement_prim or not placement_prim.IsValid():
            raise ValueError(
                "placement_prim_path_hint "
                f"{placement_prim_path_hint!r} was not found in the patched USD"
            )
        if not body_prim.GetPath().HasPrefix(placement_prim.GetPath()):
            raise ValueError(
                f"body {body_prim.GetPath()} is not below placement prim "
                f"{placement_prim.GetPath()}"
            )
        if not UsdGeom.Xformable(placement_prim):
            # USD silently ignores xform ops on non-Xformable prims (Scope or
            # typeless groupings): the drop translation would never happen
            # while every downstream gate assumed it did. Fail closed.
            raise ValueError(
                "placement_prim_path_hint "
                f"{placement_prim_path_hint!r} is not UsdGeomXformable; the "
                "drop placement would be silently discarded"
            )
    else:
        placement_prim = body_prim
    _reset_drop_settle_velocities(scene_stage)
    up_idx = _stage_up_axis_index(scene_stage)  # 1 for Y-up, 2 for Z-up
    # Post-bake the stage is metric, so stage-unit == meter.
    units_per_meter = _stage_units_per_meter(scene_stage)
    mpu = 1.0 / units_per_meter

    # Pre-translation world bbox tells us where the body's bottom
    # currently sits on the up-axis. Different assets put their local
    # origin in different places — a centered lightbulb has bbox_min
    # at -bbox/2, a parent-xform-rooted asset like the SimReady ladder
    # has bbox_min at 0. We translate by ``drop_h - bbox_min_axis`` so
    # the bottom lands exactly at drop_h regardless of origin convention.
    bbox_size_stage = _bbox_size_stage_units(placement_prim)
    bbox_min_stage, _bbox_max_stage = _bbox_minmax_stage_units(placement_prim)
    acceptance_bound_prims = _acceptance_bound_prims(body_prim, UsdGeom, UsdPhysics)
    bbox_min_local_tuple, bbox_max_local_tuple = _bbox_minmax_pose_local_stage_units(
        body_prim
    )
    geometry_points_local_stage, support_decision = _select_ground_clearance_support(
        body_prim,
        acceptance_bound_prims,
        support_cache=ground_clearance_support_cache,
        support_cache_base_key=ground_clearance_support_cache_key,
    )
    bbox_local_stage_scale = [float(v) for v in _body_pose_scale(body_prim)]
    bbox_size_m = tuple(s * mpu for s in bbox_size_stage)
    bbox_height_stage = bbox_size_stage[up_idx]
    bbox_height_m = bbox_size_m[up_idx]
    if bbox_height_stage <= 0:
        raise ValueError(
            f"body bbox height is non-positive ({bbox_height_stage}); "
            "apply_physics may have produced a degenerate geometry"
        )

    # Default drop_height_m == bbox_height. Caller passes meters; we
    # operate in stage units inside the sim. Post metric-bake the two
    # are equivalent.
    if placement_mode == "mounted":
        drop_h_m = 0.0
    else:
        drop_h_m = float(drop_height_m) if drop_height_m is not None else bbox_height_m
    if drop_h_m < 0:
        raise ValueError(f"drop_height_m must be >= 0, got {drop_h_m}")
    drop_h_stage = drop_h_m * units_per_meter

    # Translate so bbox_min on the up-axis lands at drop_h. The bbox we
    # just read already includes any pre-existing translate authored on
    # the body, so the right move is to ADD the gap-closing delta to the
    # current translate (not replace it with the raw delta — that's the
    # bug kimbyn flagged 2026-05-12 that pushed pre-translated assets
    # below the ground plane).
    current_translation_stage = _get_body_translation(placement_prim)
    bbox_min_local_stage = [float(v) for v in bbox_min_local_tuple]
    bbox_max_local_stage = [float(v) for v in bbox_max_local_tuple]
    delta_axis_stage = drop_h_stage - bbox_min_stage[up_idx]
    new_translation_stage = list(current_translation_stage)
    new_translation_stage[up_idx] = current_translation_stage[up_idx] + delta_axis_stage
    if placement_mode == "drop":
        _set_body_translation(
            placement_prim,
            (
                new_translation_stage[0],
                new_translation_stage[1],
                new_translation_stage[2],
            ),
        )

    # The metric-baked stage receives gravity in meters/s^2.
    _author_physics_scene(scene_stage, gravity_m_per_s2=gravity)

    # Ground plane: ``add_ground_plane`` already aligns its normal with
    # the stage's up-axis. Its extent is in stage units, so feed it the
    # stage-unit max bbox dim.
    #
    # Round 12 (CX P1#1): ``add_ground_plane`` defaults to
    # ``/World/GroundPlane`` and ``/World/GroundPlaneMaterial``; when the
    # asset's stage default-prim happens to also be ``/World`` (the
    # canonical SimReady asset path), the plane lands as a child of the
    # rigid-body parent and inherits its RigidBodyAPI — i.e. the asset
    # ends up "welded to its own ground" and never falls. Place the plane
    # and its material under a sibling path that is definitely outside
    # the body_prim subtree so the plane stays static.
    _ground_root = _ground_plane_root_for_body(scene_stage, placement_prim)
    add_ground_plane(
        scene_stage,
        center=(0.0, 0.0, 0.0),
        extent=max(bbox_size_stage) * 8.0,
        friction=ground_friction,
        restitution=0.0,
        plane_path=f"{_ground_root}/GroundPlane",
        material_path=f"{_ground_root}/GroundPlaneMaterial",
    )

    # Camera(s).
    camera_directions = list(cameras) if cameras else ["+x+y+z"]
    camera_paths = _author_cameras(
        scene_stage,
        placement_prim,
        camera_directions,
        ground_bias_fraction=camera_ground_bias_fraction,
    )

    # Default prim already set by patched USD; keep it. Publish the root layer
    # with a localized dependency closure: direct root-layer export would leave
    # relative arcs anchored at the source, while flattening would materialize
    # instance proxies and change runtime semantics.
    _export_scene_portably(
        scene_stage,
        patched_physics_usd,
        output_scene_usd,
        dependency_roots,
    )

    # rest_position is what ``settle_distance`` compares the final body
    # translate against. The body settles when its bbox-min on the
    # up-axis sits at the ground (z=0); the corresponding translate
    # value is ``-bbox_min_pre`` (the inverse of the body's local
    # origin offset from its bbox-min). For a corner-origin asset like
    # SimReady's ladder bbox_min_pre = 0 → rest_translate = 0; for a
    # centered single-mesh asset like a lightbulb bbox_min_pre = -h/2
    # → rest_translate = h/2.
    if placement_mode == "mounted":
        # A mounted body need not rest on the synthetic ground. Its complete
        # authored frame remains the reference, including nonzero X/Y/Z and
        # transformed parents. No joint relationship or local frame is edited.
        body_world_transform = UsdGeom.Xformable(
            body_prim
        ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        rest_position_stage = list(body_world_transform.ExtractTranslation())
    elif placement_prim == body_prim:
        rest_position_stage = [0.0, 0.0, 0.0]
        rest_position_stage[up_idx] = -bbox_min_stage[up_idx]
    else:
        body_world_transform = UsdGeom.Xformable(
            body_prim
        ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        body_world_translation = body_world_transform.ExtractTranslation()
        rest_position_stage = [float(value) for value in body_world_translation]
        # The common placement root is currently at launch height. At ground
        # contact that root, and therefore every child body, is lower by the
        # requested gap along the stage up axis.
        rest_position_stage[up_idx] -= drop_h_stage

    # Authoritative world-up vector for downstream consumers. The
    # judge derives this via ``infer_world_up(rest_position)`` today,
    # but a corner-origin asset (where ``bbox_min`` already sits at
    # origin) yields a zero rest_position and infer_world_up falls
    # back to legacy Y-up — wrong on a Z-up stage. Stash the actual
    # axis here so the judge can use it directly when the inference
    # would be ambiguous.
    world_up_stage = [0.0, 0.0, 0.0]
    world_up_stage[up_idx] = 1.0

    body_path_str = str(body_prim.GetPath())
    result = {
        "placement_mode": placement_mode,
        "body_prim_path": body_path_str,
        "body_pattern": str(body_pattern_hint or body_path_str),
        "placement_prim_path": str(placement_prim.GetPath()),
        # rest_position is in STAGE UNITS — same units the simulator
        # writes to the trajectory and that ``settle_distance`` reads.
        # Converting to meters here would mismatch and produce a wrong
        # score on non-meter stages.
        "rest_position": rest_position_stage,
        "world_up": world_up_stage,
        "drop_height_m_resolved": drop_h_m,
        "bbox_size_m": list(bbox_size_m),
        "bbox_min_local_stage": bbox_min_local_stage,
        "bbox_max_local_stage": bbox_max_local_stage,
        "bbox_local_stage_scale": bbox_local_stage_scale,
        "bbox_local_stage_space": "pose_local",
        "meters_per_unit": mpu,
        "gravity_magnitude_m_per_s2": abs(float(gravity)),
        "camera_paths": camera_paths,
        "ground_clearance_support_decision": support_decision,
    }
    if geometry_points_local_stage is not None:
        result["geometry_points_local_stage"] = geometry_points_local_stage
        result["ground_clearance_geometry"] = "collider_mesh_vertices"
    else:
        result["ground_clearance_geometry"] = "conservative_bbox_corners"
    total_scene_build_seconds = time.perf_counter() - scene_build_started
    support_selection_seconds = support_decision.get("selection_duration_seconds")
    if not isinstance(support_selection_seconds, int | float) or isinstance(
        support_selection_seconds, bool
    ):
        support_selection_seconds = 0.0
    # Keep benchmark phases mutually exclusive: support selection is reported
    # independently from the rest of scene construction/export.
    result["scene_build_duration_seconds"] = max(
        0.0,
        total_scene_build_seconds - float(support_selection_seconds),
    )
    return result


def build_freeform_scene(
    patched_physics_usd: Path,
    output_scene_usd: Path,
    *,
    target: dict[str, Any],
    body_prim_path_hint: str | None = None,
    approved_dependency_roots: Iterable[str | Path] | None = None,
) -> dict[str, Any]:
    """Build a freeform simulation scene from an LLM-authored target dict.

    Reads:
      target.gravity (default -9.81)
      target.surface.friction (default 0.5)
      target.initial_pose.position (default [0, bbox_height, 0])
      target.initial_pose.rotation (default [0, 0, 0] XYZ Euler radians)
      target.cameras (default ["+x+y+z"] — tilted corner view)
      target.camera_ground_bias_fraction (0.0–1.0, optional; lerps the
        corner-camera look-at along the up-axis toward the ground.
        Identical contract to :func:`build_drop_settle_scene`'s
        ``camera_ground_bias_fraction`` kwarg — see ``_author_cameras``
        for the full semantics and cardinal-camera caveat.)

    ``approved_dependency_roots`` bounds which filesystem dependencies may
    be copied into the portable scene sidecar. It defaults to the patched
    asset's directory; service callers should pass trusted input roots.

    Returns the same shape as :func:`build_drop_settle_scene` — callers
    treat scenes uniformly. ``rest_position`` is the body's initial
    position (used by the trajectory metric module to compute settle
    distance for freeform-but-still-settled cases).
    """
    from pxr import Usd  # type: ignore[import-untyped]
    from world_understanding.functions.graphics.so_export import (
        _normalize_dependency_roots,
    )
    from world_understanding.utils.usd.scene import add_ground_plane

    dependency_roots = _normalize_dependency_roots(
        tuple(approved_dependency_roots)
        if approved_dependency_roots is not None
        else (Path(patched_physics_usd).resolve().parent,)
    )
    # Same in-memory pattern as drop_settle: open the patched USD,
    # author additions, export to scene_path without modifying the
    # source on disk.
    scene_stage = Usd.Stage.Open(str(patched_physics_usd))
    if scene_stage is None:
        raise ValueError(f"could not open patched physics USD {patched_physics_usd}")

    # Same metric-bake as drop_settle so freeform inherits the same
    # gravity-magnitude correctness on cm-scale USDs.
    _bake_metric_units(scene_stage)

    if body_prim_path_hint:
        body_prim = scene_stage.GetPrimAtPath(body_prim_path_hint)
        if not body_prim:
            raise ValueError(
                f"body_prim_path_hint {body_prim_path_hint!r} not found in "
                "the patched USD"
            )
    else:
        body_prim = _stage_default_body_prim(scene_stage)

    up_idx = _stage_up_axis_index(scene_stage)
    units_per_meter = _stage_units_per_meter(scene_stage)
    mpu = 1.0 / units_per_meter

    bbox_size_stage = _bbox_size_stage_units(body_prim)
    bbox_min_stage, _bbox_max_stage = _bbox_minmax_stage_units(body_prim)
    bbox_size_m = tuple(s * mpu for s in bbox_size_stage)
    bbox_height_stage = max(bbox_size_stage[up_idx], 1e-6)
    current_translation_stage = _get_body_translation(body_prim)
    bbox_min_local_tuple, bbox_max_local_tuple = _bbox_minmax_pose_local_stage_units(
        body_prim
    )
    bbox_min_local_stage = [float(v) for v in bbox_min_local_tuple]
    bbox_max_local_stage = [float(v) for v in bbox_max_local_tuple]
    bbox_local_stage_scale = [float(v) for v in _body_pose_scale(body_prim)]

    gravity = float(target.get("gravity", -9.81))
    surface = target.get("surface") or {}
    friction = float(surface.get("friction", 0.5))

    initial_pose = target.get("initial_pose") or {}
    pos_m = initial_pose.get("position")
    if pos_m is None:
        # Default: body bottom at drop_h = bbox_height_m above the
        # ground. We mirror drop_settle's bbox-min-aware placement so
        # the bottom lands at the right height regardless of the
        # asset's local origin convention (centered single-mesh has
        # bbox_min at -h/2, parent-Xform-rooted SimReady asset has
        # bbox_min at 0). bbox_min already includes the body's
        # pre-existing translate, so the gap-closing delta is ADDED to
        # the current translate (kimbyn 2026-05-12 — replacing the op
        # outright placed pre-translated assets below the ground).
        delta_axis_stage = bbox_height_stage - bbox_min_stage[up_idx]
        new_translation_stage = list(current_translation_stage)
        new_translation_stage[up_idx] = (
            current_translation_stage[up_idx] + delta_axis_stage
        )
        pos_stage = tuple(new_translation_stage)
        # Report the equivalent meter-coords back through the result dict.
        pos_m_tuple = tuple(c * mpu for c in pos_stage)
    else:
        # Explicit user-supplied absolute position — REPLACE semantics
        # (the user authored the world position they want, not a delta).
        pos_m_tuple = (float(pos_m[0]), float(pos_m[1]), float(pos_m[2]))
        pos_stage = tuple(c * units_per_meter for c in pos_m_tuple)
    _set_body_translation(body_prim, pos_stage)

    rot = initial_pose.get("rotation")
    if rot is not None:
        _set_body_rotation(body_prim, (float(rot[0]), float(rot[1]), float(rot[2])))

    _author_physics_scene(scene_stage, gravity_m_per_s2=gravity)
    # Round 12 (CX P1#1): same body_prim subtree avoidance as drop_settle.
    _ground_root = _ground_plane_root_for_body(scene_stage, body_prim)
    add_ground_plane(
        scene_stage,
        center=(0.0, 0.0, 0.0),
        extent=max(bbox_size_stage) * 8.0,
        friction=friction,
        restitution=0.0,
        plane_path=f"{_ground_root}/GroundPlane",
        material_path=f"{_ground_root}/GroundPlaneMaterial",
    )

    camera_directions = list(target.get("cameras") or ["+x+y+z"])
    camera_paths = _author_cameras(
        scene_stage,
        body_prim,
        camera_directions,
        ground_bias_fraction=target.get("camera_ground_bias_fraction"),
    )

    _export_scene_portably(
        scene_stage,
        patched_physics_usd,
        output_scene_usd,
        dependency_roots,
    )

    body_path_str = str(body_prim.GetPath())
    # Round 12 (CX P2#4): freeform also needs to surface ``world_up`` so
    # downstream programmatic scoring (``trajectory_summary``) and the
    # judge use the actual stage up-axis instead of the legacy Y-up
    # default. A Z-up asset with the wrong axis flagged a yaw spin around
    # Z as ``fell_over``.
    world_up_stage = [0.0, 0.0, 0.0]
    world_up_stage[up_idx] = 1.0
    return {
        "body_prim_path": body_path_str,
        "body_pattern": body_path_str,
        # rest_position is in STAGE UNITS so settle_distance compares
        # apples-to-apples against the trajectory's stage-unit positions.
        "rest_position": list(pos_stage),
        "world_up": world_up_stage,
        "bbox_size_m": list(bbox_size_m),
        "bbox_min_local_stage": bbox_min_local_stage,
        "bbox_max_local_stage": bbox_max_local_stage,
        "bbox_local_stage_scale": bbox_local_stage_scale,
        "bbox_local_stage_space": "pose_local",
        "meters_per_unit": mpu,
        "camera_paths": camera_paths,
        "gravity": gravity,
        "gravity_magnitude_m_per_s2": abs(float(gravity)),
        "ground_friction": friction,
    }


__all__ = ["build_drop_settle_scene", "build_freeform_scene"]
