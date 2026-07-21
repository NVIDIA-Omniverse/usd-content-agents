# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fail-closed validation for owned Joint Rigger authoring.

The topology validators prove that explicit joint plans can be authored without
reshaping the source stage. The schema validators snapshot hierarchy, topology,
and transforms around evidence-backed physics writes. OpenUSD imports stay
inside call sites so JSON-contract consumers do not require the runtime.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, NoReturn, cast

from world_understanding.functions.physics.joint_rigger.models import (
    FieldDecisionV1,
    JointDiagnosticV1,
    JointPlanV1,
    JointRiggerContractError,
    JointRiggerDiagnosticsV1,
    JointRiggerPlanV1,
    canonical_sha256,
)

_FRAME_TOLERANCE = 1e-5
_SHARED_ANCHOR_DISTANCE_TOLERANCE = 1e-6
_FLOAT32_MAX = float.fromhex("0x1.fffffep+127")
_MATRIX_TOLERANCE = 1e-9
_TOPOLOGY_HASH_LENGTH = 12
_TOPOLOGY_AUTHOR_VERSION = "world-understanding-joint-topology-author-v1"
_JOINT_SCOPE_NAME = "Joints"
# These are trust-boundary ceilings, not request tuning knobs.  Making them
# caller-configurable would let an untrusted plan disable the fail-closed work
# bound.  Tests may lower them to exercise the rejection paths; production
# invocations always use these fixed maxima.
_INACTIVE_SCAN_MAX_ACTIVATION_ROUNDS = 1024
_INACTIVE_SCAN_MAX_PRIM_VISITS = 1_000_000
_STAGE_SNAPSHOT_MAX_PRIM_VISITS = 1_000_000
_SOURCE_JOINT_SCAN_MAX_PRIM_VISITS = 1_000_000
_SOURCE_JOINT_SCAN_MAX_PATHS = 16_384
_JOINT_FACT_ATTRIBUTES = (
    "physics:axis",
    "physics:localPos0",
    "physics:localPos1",
    "physics:localRot0",
    "physics:localRot1",
    "physics:lowerLimit",
    "physics:upperLimit",
    "physics:coneAngle0Limit",
    "physics:coneAngle1Limit",
)
_SUPPORTED_JOINT_TYPES = frozenset({"revolute", "prismatic", "spherical"})
_JOINT_SCHEMA_TYPE_NAMES = {
    "revolute": "PhysicsRevoluteJoint",
    "prismatic": "PhysicsPrismaticJoint",
    "spherical": "PhysicsSphericalJoint",
}
_SOURCE_BACKED_PROVENANCE_SOURCES = frozenset(
    {
        "accepted_manifest",
        "authored_metadata",
        "authored_reference",
        "source_metadata",
    }
)

type _Vector3 = tuple[float, float, float]
type _Quaternion = tuple[float, tuple[float, float, float]]


@dataclass(frozen=True)
class _PrimSnapshot:
    """Source-prim facts that topology authoring is forbidden to change."""

    type_name: str
    active: bool
    defined: bool
    instanceable: bool
    applied_schemas: tuple[str, ...]
    raw_api_schemas: str | None
    authored_prim_specs: tuple[tuple[str, str, tuple[tuple[str, str], ...]], ...]
    authored_property_specs: tuple[tuple[str, tuple[tuple[str, str, str], ...]], ...]
    world_transform: tuple[float, ...] | None


@dataclass(frozen=True)
class _StageSnapshot:
    """Structural source-stage baseline for the no-reshape invariant."""

    default_prim_path: str
    meters_per_unit: float
    up_axis: str
    prims: dict[str, _PrimSnapshot]


@dataclass(frozen=True)
class _PreparedJoint:
    """All deterministic USD facts preflighted for one explicit joint plan."""

    source: JointPlanV1
    joint_path: str
    axis_token: str | None
    local_pos0: _Vector3
    local_pos1: _Vector3
    local_rot0: _Quaternion | None
    local_rot1: _Quaternion | None
    anchor_stage: _Vector3
    lower_limit: float | None
    upper_limit: float | None
    drive_instance: str | None


@dataclass(frozen=True)
class _TopologyPreflight:
    """Complete pre-authoring state shared by the author and validator."""

    joints: tuple[_PreparedJoint, ...]
    joints_scope_path: str
    create_joints_scope: bool
    plan_sha256: str
    snapshot: _StageSnapshot


def validate_joint_topology_plan(stage: Any, plan: JointRiggerPlanV1) -> None:
    """Prove that ``plan`` can be authored without changing source structure."""

    if not isinstance(plan, JointRiggerPlanV1):
        raise TypeError("plan must be a JointRiggerPlanV1")
    _preflight_topology_authoring(stage, plan)


def validate_authored_joint_topology(
    stage: Any,
    plan: JointRiggerPlanV1,
    diagnostics: JointRiggerDiagnosticsV1 | None = None,
) -> None:
    """Validate an already-authored stage against one exact shared plan."""

    if not isinstance(plan, JointRiggerPlanV1):
        raise TypeError("plan must be a JointRiggerPlanV1")
    if diagnostics is not None and not isinstance(
        diagnostics,
        JointRiggerDiagnosticsV1,
    ):
        raise TypeError("diagnostics must be a JointRiggerDiagnosticsV1 or None")
    preflight = _preflight_topology_authoring(
        stage,
        plan,
        allow_existing_joint_paths=True,
    )
    _validate_authored_preflight(stage, preflight, diagnostics=diagnostics)


def _preflight_topology_authoring(
    stage: Any,
    plan: JointRiggerPlanV1,
    *,
    allow_existing_joint_paths: bool = False,
) -> _TopologyPreflight:
    """Resolve every authored fact before a stage can be mutated."""

    Gf, Sdf, Tf, Usd, UsdGeom, UsdPhysics = _pxr_modules()
    _validate_supported_plan_shape(plan)
    default_prim = stage.GetDefaultPrim()
    if (
        not default_prim
        or not default_prim.IsValid()
        or not default_prim.IsActive()
        or not default_prim.IsDefined()
    ):
        _fail("invalid_default_prim", "input stage must define an active defaultPrim")
    default_path = default_prim.GetPath()
    if (
        not default_path.IsAbsolutePath()
        or not default_path.IsPrimPath()
        or default_path.IsAbsoluteRootPath()
    ):
        _fail(
            "invalid_default_prim",
            f"input stage defaultPrim path is invalid: {default_path}",
        )
    if default_prim.IsInstance() or default_prim.IsInstanceProxy():
        _fail(
            "invalid_default_prim",
            "input stage defaultPrim cannot be an instance for no-reshape authoring",
        )

    existing_joint_paths = _existing_joint_paths(
        stage,
        Sdf=Sdf,
        Usd=Usd,
        UsdPhysics=UsdPhysics,
    )
    joints_scope_path = default_path.AppendChild(_JOINT_SCOPE_NAME)
    authored_paths = tuple(
        _deterministic_joint_path(
            default_path=default_path,
            joint=joint,
            Sdf=Sdf,
            Tf=Tf,
        )
        for joint in plan.joints
    )
    if len(authored_paths) != len(set(authored_paths)):
        _fail(
            "joint_target_collision",
            "multiple topology entries resolve to one deterministic joint path",
        )
    _reject_nested_joint_paths(authored_paths, Sdf=Sdf)
    planned_path_values = {str(path) for path in authored_paths}
    if allow_existing_joint_paths:
        if existing_joint_paths != planned_path_values:
            _fail(
                "authored_graph_mismatch",
                "authored stage joint paths do not exactly match the plan: "
                f"observed={sorted(existing_joint_paths)}, "
                f"expected={sorted(planned_path_values)}",
            )
    elif existing_joint_paths:
        _fail(
            "source_already_rigged",
            "owned topology authoring requires a source with no existing USD "
            f"physics joints; found {sorted(existing_joint_paths)}",
        )

    existing_scope = stage.GetPrimAtPath(joints_scope_path)
    create_joints_scope = not existing_scope or not existing_scope.IsValid()
    if not create_joints_scope and (
        not existing_scope.IsActive()
        or not existing_scope.IsDefined()
        or existing_scope.IsInstance()
        or existing_scope.IsInstanceProxy()
        or not existing_scope.IsA(UsdGeom.Scope)
    ):
        _fail(
            "joint_scope_conflict",
            "owned topology authoring requires an active, defined, non-instance "
            f"UsdGeom.Scope at {joints_scope_path}",
        )

    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    if not math.isfinite(meters_per_unit) or meters_per_unit <= 0.0:
        _fail(
            "invalid_stage_units",
            f"input stage metersPerUnit must be positive and finite; got "
            f"{meters_per_unit!r}",
        )

    prepared = tuple(
        _prepare_joint(
            stage,
            joint,
            joint_path=joint_path,
            meters_per_unit=meters_per_unit,
            allow_existing_joint_path=allow_existing_joint_paths,
            Gf=Gf,
            Sdf=Sdf,
            UsdGeom=UsdGeom,
        )
        for joint, joint_path in zip(plan.joints, authored_paths, strict=True)
    )
    snapshot = _capture_stage_snapshot(stage, UsdGeom=UsdGeom)
    return _TopologyPreflight(
        joints=prepared,
        joints_scope_path=str(joints_scope_path),
        create_joints_scope=create_joints_scope,
        plan_sha256=canonical_sha256(plan),
        snapshot=snapshot,
    )


def _existing_joint_paths(
    stage: Any,
    *,
    Sdf: Any,
    Usd: Any,
    UsdPhysics: Any,
) -> set[str]:
    """Return composed joint paths, including native-instance internals.

    ``Stage.TraverseAll`` deliberately stops at native instances.  Inspect both
    instance proxies and prototype namespaces so an existing joint cannot hide
    behind instancing during either source preflight or exact-graph validation.
    """

    paths: set[str] = set()
    prim_visits = 0

    def add_path(path: str) -> None:
        if path in paths:
            return
        if len(paths) >= _SOURCE_JOINT_SCAN_MAX_PATHS:
            _fail(
                "source_joint_path_limit_exceeded",
                "existing-joint inspection exceeds the fixed "
                f"{_SOURCE_JOINT_SCAN_MAX_PATHS}-path retention limit",
            )
        paths.add(path)

    def collect_joint(prim: Any) -> bool:
        if not prim.IsA(UsdPhysics.Joint):
            return False
        add_path(str(prim.GetPath()))
        return True

    def inspect(prims: Any, *, phase: str) -> None:
        nonlocal prim_visits
        for prim in prims:
            prim_visits += 1
            if prim_visits > _SOURCE_JOINT_SCAN_MAX_PRIM_VISITS:
                _fail(
                    "source_joint_scan_limit_exceeded",
                    "existing-joint inspection exceeds the fixed "
                    f"{_SOURCE_JOINT_SCAN_MAX_PRIM_VISITS}-prim visit limit "
                    f"during {phase}",
                )
            collect_joint(prim)

    inspect(
        Usd.PrimRange.Stage(
            stage,
            Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate),
        ),
        phase="composed-stage scan",
    )
    for prototype in stage.GetPrototypes():
        inspect(
            Usd.PrimRange.AllPrims(prototype),
            phase="prototype scan",
        )
    inactive_paths = _paths_with_inactive_ancestors_enabled(
        stage,
        matches=collect_joint,
        Sdf=Sdf,
        Usd=Usd,
    )
    for path in inactive_paths:
        add_path(path)
    return paths


def _bounded_traverse_all(
    stage: Any,
    *,
    failure_code: str,
    purpose: str,
) -> Iterator[Any]:
    """Stream composed prims under one fixed public trust-boundary ceiling."""

    prim_visits = 0
    for prim in stage.TraverseAll():
        prim_visits += 1
        if prim_visits > _STAGE_SNAPSHOT_MAX_PRIM_VISITS:
            _fail(
                failure_code,
                f"{purpose} exceeds the fixed "
                f"{_STAGE_SNAPSHOT_MAX_PRIM_VISITS}-prim visit limit",
            )
        yield prim


def _paths_with_inactive_ancestors_enabled(
    stage: Any,
    *,
    matches: Callable[[Any], bool],
    Sdf: Any,
    Usd: Any,
) -> set[str]:
    """Inspect inactive composed subtrees without scanning unrelated layer specs.

    An inactive prim remains visible to ``PrimAllPrimsPredicate``, but traversal
    prunes its descendants.  Open the same composition with a private, stronger
    session layer, expand native instances, and enable each inactive composed
    ancestor until the complete currently selected namespace is visible.  This
    respects references, variants, population masks, load rules, and muted
    layers while leaving the caller's stage and authored layers untouched.
    """

    prim_visits = 0

    def consume_prim_visit(*, phase: str) -> None:
        nonlocal prim_visits
        prim_visits += 1
        if prim_visits > _INACTIVE_SCAN_MAX_PRIM_VISITS:
            _fail(
                "source_joint_scan_limit_exceeded",
                f"inactive-joint inspection exceeded its {phase} work budget",
            )

    def contains_inactive_prim(prims: Any) -> bool:
        for prim in prims:
            consume_prim_visit(phase="initial composed-prim")
            if not prim.IsActive():
                return True
        return False

    has_inactive_prim = contains_inactive_prim(
        Usd.PrimRange.Stage(
            stage,
            Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate),
        )
    ) or any(
        contains_inactive_prim(Usd.PrimRange.AllPrims(prototype))
        for prototype in stage.GetPrototypes()
    )
    if not has_inactive_prim:
        return set()

    scan_stage = Usd.Stage.CreateInMemory(
        "joint-rigger-inactive-scan.usda",
        stage.GetPathResolverContext(),
        Usd.Stage.LoadNone,
    )
    scan_root = scan_stage.GetRootLayer()
    scan_session = scan_stage.GetSessionLayer()
    source_session = stage.GetSessionLayer()
    scan_stage.SetPopulationMask(stage.GetPopulationMask())
    muted_layers = stage.GetMutedLayers()
    if muted_layers:
        scan_stage.MuteAndUnmuteLayers(muted_layers, [])
    scan_stage.SetLoadRules(stage.GetLoadRules())
    with Sdf.ChangeBlock():
        scan_root.subLayerPaths = [stage.GetRootLayer().identifier]
        if source_session is not None:
            scan_session.subLayerPaths = [source_session.identifier]
    scan_stage.SetEditTarget(scan_session)

    enabled_paths: set[str] = set()
    expanded_instance_paths: set[str] = set()
    activation_rounds = 0
    while True:
        composed_prims = []
        for prim in Usd.PrimRange.Stage(
            scan_stage,
            Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate),
        ):
            consume_prim_visit(phase="composed-prim")
            composed_prims.append(prim)
        inactive_paths = {
            str(prim.GetPath())
            for prim in composed_prims
            if not prim.IsActive() and not prim.IsInstanceProxy()
        }
        instance_paths = {
            str(prim.GetPath())
            for prim in composed_prims
            if prim.IsInstance() and not prim.IsInstanceProxy()
        }
        pending_paths = inactive_paths - enabled_paths
        pending_instances = instance_paths - expanded_instance_paths
        if not pending_paths and not pending_instances:
            break
        activation_rounds += 1
        if activation_rounds > _INACTIVE_SCAN_MAX_ACTIVATION_ROUNDS:
            _fail(
                "source_joint_scan_limit_exceeded",
                "inactive-joint inspection exceeded its activation-round budget",
            )
        for path in sorted(pending_instances):
            instance = scan_stage.GetPrimAtPath(path)
            for _ in Usd.PrimRange.AllPrims(instance.GetPrototype()):
                consume_prim_visit(phase="instance-expansion")
        with Sdf.ChangeBlock():
            for path in sorted(pending_instances):
                _require_scan_instance_expansion(scan_stage, path)
                expanded_instance_paths.add(path)
            for path in sorted(pending_paths):
                _require_scan_activation(scan_stage, path)
                enabled_paths.add(path)

    return {str(prim.GetPath()) for prim in composed_prims if matches(prim)}


def _require_scan_instance_expansion(stage: Any, path: str) -> None:
    """Fail closed when a private scan cannot expand one native instance."""

    if not stage.OverridePrim(path).SetInstanceable(False):
        _fail(
            "source_joint_scan_failed",
            f"could not inspect instance subtree at {path}",
        )


def _require_scan_activation(stage: Any, path: str) -> None:
    """Fail closed when a private scan cannot activate one composed subtree."""

    if not stage.OverridePrim(path).SetActive(True):
        _fail(
            "source_joint_scan_failed",
            f"could not inspect inactive composed subtree at {path}",
        )


def _validate_supported_plan_shape(plan: JointRiggerPlanV1) -> None:
    if not plan.joints:
        _fail("empty_topology", "owned topology authoring requires at least one joint")
    if plan.rigid_bodies:
        _fail(
            "physics_schema_fields_unsupported",
            "WP-R2 does not author rigid-body, mass, or collider plans",
        )
    if plan.articulation_root is not None:
        _fail(
            "physics_schema_fields_unsupported",
            "WP-R2 does not author an articulation-root schema",
        )
    for joint in plan.joints:
        joint_id = joint.topology.joint_id
        if joint.state is not None or joint.mimic is not None:
            _fail(
                "physics_schema_fields_unsupported",
                f"joint {joint_id!r} carries WP-R3 state or mimic fields",
            )
        if joint.topology.joint_type not in _SUPPORTED_JOINT_TYPES:
            _fail(
                "unsupported_joint_type",
                f"joint {joint_id!r} has unsupported type "
                f"{joint.topology.joint_type!r}",
            )
        if joint.drive is not None:
            if joint.topology.joint_type == "spherical":
                _fail(
                    "unsupported_drive_instance",
                    f"spherical joint {joint_id!r} cannot represent a scalar drive",
                )
            _require_source_backed_provenance(
                joint.drive.provenance,
                label=f"joint {joint_id!r} drive",
            )
        if joint.joint_friction is not None:
            if joint.topology.joint_type == "spherical":
                _fail(
                    "joint_friction_not_applicable",
                    f"spherical joint {joint_id!r} cannot represent scalar friction",
                )
            _require_source_backed_provenance(
                joint.joint_friction.provenance,
                label=f"joint {joint_id!r} friction",
            )
        if joint.limit is not None:
            _require_source_backed_provenance(
                joint.limit.provenance,
                label=f"joint {joint_id!r} limit",
            )
        if joint.anchor is not None:
            _require_source_backed_provenance(
                joint.anchor.provenance,
                label=f"joint {joint_id!r} anchor",
            )


def _require_source_backed_provenance(provenance: Any, *, label: str) -> None:
    if (
        provenance.source not in _SOURCE_BACKED_PROVENANCE_SOURCES
        or provenance.artifact is None
        or provenance.prim_path is None
        or not provenance.properties
    ):
        _fail(
            "optional_field_not_source_backed",
            f"{label} must use artifact-backed provenance and identify an artifact, "
            "prim_path, and properties",
        )


def _deterministic_joint_path(
    *,
    default_path: Any,
    joint: JointPlanV1,
    Sdf: Any,
    Tf: Any,
) -> Any:
    topology = joint.topology
    payload = json.dumps(
        {
            "axis_stage": topology.axis_stage,
            "body0": topology.body0,
            "body1": topology.body1,
            "joint_id": topology.joint_id,
            "joint_type": topology.joint_type,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:_TOPOLOGY_HASH_LENGTH]
    identifier = str(Tf.MakeValidIdentifier(topology.joint_id) or "joint")
    identifier = identifier[:96]
    joint_name = f"{identifier}_{digest}"
    path = default_path.AppendChild(_JOINT_SCOPE_NAME).AppendChild(joint_name)
    if not path.IsAbsolutePath() or not path.IsPrimPath():  # pragma: no cover
        _fail(
            "invalid_joint_path",
            f"could not derive a valid joint path for {topology.joint_id!r}",
        )
    return Sdf.Path(path)


def _reject_nested_joint_paths(paths: tuple[Any, ...], *, Sdf: Any) -> None:
    for index, left in enumerate(paths):
        for right in paths[index + 1 :]:
            left_path = Sdf.Path(left)
            right_path = Sdf.Path(right)
            if left_path.HasPrefix(right_path) or right_path.HasPrefix(left_path):
                _fail(
                    "joint_target_collision",
                    f"planned joint paths must not be nested: {left}, {right}",
                )


def _prepare_joint(
    stage: Any,
    joint: JointPlanV1,
    *,
    joint_path: Any,
    meters_per_unit: float,
    allow_existing_joint_path: bool,
    Gf: Any,
    Sdf: Any,
    UsdGeom: Any,
) -> _PreparedJoint:
    topology = joint.topology
    if not allow_existing_joint_path and stage.GetPrimAtPath(joint_path).IsValid():
        _fail(
            "joint_target_collision",
            f"refusing to overwrite an existing prim at {joint_path}",
        )
    body0_prim = _require_endpoint(
        stage,
        topology.body0,
        label=f"joint {topology.joint_id!r} body0",
        Sdf=Sdf,
        UsdGeom=UsdGeom,
    )
    body1_prim = _require_endpoint(
        stage,
        topology.body1,
        label=f"joint {topology.joint_id!r} body1",
        Sdf=Sdf,
        UsdGeom=UsdGeom,
    )
    _require_static_transform_chain(
        body0_prim,
        label=f"joint {topology.joint_id!r} body0",
        UsdGeom=UsdGeom,
    )
    _require_static_transform_chain(
        body1_prim,
        label=f"joint {topology.joint_id!r} body1",
        UsdGeom=UsdGeom,
    )
    xform_cache = UsdGeom.XformCache()
    body0_xform = xform_cache.GetLocalToWorldTransform(body0_prim)
    body1_xform = xform_cache.GetLocalToWorldTransform(body1_prim)
    _require_invertible_transform(
        body0_xform,
        label=f"joint {topology.joint_id!r} body0",
    )
    _require_invertible_transform(
        body1_xform,
        label=f"joint {topology.joint_id!r} body1",
    )

    if joint.anchor is None:
        anchor = body1_xform.Transform(Gf.Vec3d(0.0, 0.0, 0.0))
    else:
        anchor = Gf.Vec3d(*joint.anchor.position_stage)
    local_pos0 = body0_xform.GetInverse().Transform(anchor)
    local_pos1 = body1_xform.GetInverse().Transform(anchor)

    axis_token: str | None = None
    local_rot0: _Quaternion | None = None
    local_rot1: _Quaternion | None = None
    if topology.axis_stage is not None:
        axis_stage = Gf.Vec3d(*topology.axis_stage)
        axis_token, _ = _axis_token_and_vector(topology.axis_stage, Gf=Gf)
        stage_frame = _stage_joint_frame(axis_stage, Gf=Gf)
        local_rot0 = _local_joint_frame_rotation(
            body0_xform,
            stage_frame=stage_frame,
            axis_token=axis_token,
            label=f"joint {topology.joint_id!r} body0",
            Gf=Gf,
        )
        local_rot1 = _local_joint_frame_rotation(
            body1_xform,
            stage_frame=stage_frame,
            axis_token=axis_token,
            label=f"joint {topology.joint_id!r} body1",
            Gf=Gf,
        )

    lower_limit = None
    upper_limit = None
    if joint.limit is not None:
        lower_limit = joint.limit.lower
        upper_limit = joint.limit.upper
        if topology.joint_type == "prismatic":
            lower_limit = _optional_divide(lower_limit, meters_per_unit)
            upper_limit = _optional_divide(upper_limit, meters_per_unit)

    drive_instance = None
    if joint.drive is not None:
        drive_instance = "angular" if topology.joint_type == "revolute" else "linear"
        for field in (
            "stiffness",
            "damping",
            "max_force",
            "target_position",
            "target_velocity",
            "max_joint_velocity",
        ):
            _require_float32_value(
                getattr(joint.drive, field),
                label=f"joint {topology.joint_id!r} drive {field}",
            )
    if joint.joint_friction is not None:
        _require_float32_value(
            joint.joint_friction.coefficient,
            label=f"joint {topology.joint_id!r} friction coefficient",
        )

    local_pos0_value = _float32_vector(
        _vec3_tuple(local_pos0),
        label=f"joint {topology.joint_id!r} body0 local anchor",
    )
    local_pos1_value = _float32_vector(
        _vec3_tuple(local_pos1),
        label=f"joint {topology.joint_id!r} body1 local anchor",
    )
    anchor_value = _vec3_tuple(anchor)
    local_pos0_value, local_pos1_value = _reconcile_float32_local_anchors(
        body0_xform,
        body1_xform,
        requested_anchor=anchor_value,
        local_pos0=local_pos0_value,
        local_pos1=local_pos1_value,
        explicit_anchor=joint.anchor is not None,
        label=f"joint {topology.joint_id!r}",
        Gf=Gf,
    )
    reprojected_anchor0 = _vec3_tuple(
        body0_xform.Transform(Gf.Vec3d(*local_pos0_value))
    )
    reprojected_anchor1 = _vec3_tuple(
        body1_xform.Transform(Gf.Vec3d(*local_pos1_value))
    )
    if not (
        _vector_distance(reprojected_anchor0, reprojected_anchor1)
        <= _SHARED_ANCHOR_DISTANCE_TOLERANCE
        and _anchor_vectors_close(reprojected_anchor0, anchor_value)
        and _anchor_vectors_close(reprojected_anchor1, anchor_value)
        and (
            joint.anchor is None
            or (
                _vector_distance(reprojected_anchor0, anchor_value)
                <= _SHARED_ANCHOR_DISTANCE_TOLERANCE
                and _vector_distance(reprojected_anchor1, anchor_value)
                <= _SHARED_ANCHOR_DISTANCE_TOLERANCE
            )
        )
    ):
        _fail(
            "authored_value_out_of_range",
            f"joint {topology.joint_id!r} local anchors do not preserve the "
            "requested stage anchor after USD float32 storage: "
            f"requested={anchor_value!r}, body0={reprojected_anchor0!r}, "
            f"body1={reprojected_anchor1!r}",
        )
    local_rot0 = _float32_quaternion(
        local_rot0,
        label=f"joint {topology.joint_id!r} body0 local rotation",
    )
    local_rot1 = _float32_quaternion(
        local_rot1,
        label=f"joint {topology.joint_id!r} body1 local rotation",
    )
    lower_limit = _optional_float32_value(
        lower_limit,
        label=f"joint {topology.joint_id!r} lower limit",
    )
    upper_limit = _optional_float32_value(
        upper_limit,
        label=f"joint {topology.joint_id!r} upper limit",
    )
    return _PreparedJoint(
        source=joint,
        joint_path=str(joint_path),
        axis_token=axis_token,
        local_pos0=local_pos0_value,
        local_pos1=local_pos1_value,
        local_rot0=local_rot0,
        local_rot1=local_rot1,
        anchor_stage=anchor_value,
        lower_limit=lower_limit,
        upper_limit=upper_limit,
        drive_instance=drive_instance,
    )


def _require_endpoint(
    stage: Any,
    path_value: str,
    *,
    label: str,
    Sdf: Any,
    UsdGeom: Any,
) -> Any:
    path = Sdf.Path(path_value)
    if (
        str(path) != path_value
        or not path.IsAbsolutePath()
        or not path.IsPrimPath()
        or path.IsAbsoluteRootPath()
    ):
        _fail("endpoint_missing", f"{label} is not an exact absolute prim path")
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        _fail("endpoint_missing", f"{label} does not resolve: {path_value}")
    if not prim.IsActive() or not prim.IsDefined():
        _fail(
            "endpoint_inactive_or_undefined",
            f"{label} must be active and defined: {path_value}",
        )
    if prim.IsPrototype() or prim.IsInPrototype():
        _fail(
            "endpoint_prototype",
            f"{label} is in a prototype namespace and cannot be targeted without "
            "reshaping",
        )
    if prim.IsInstanceProxy():
        _fail(
            "endpoint_instance_proxy",
            f"{label} is an instance proxy and cannot be targeted without reshaping",
        )
    if not UsdGeom.Xformable(prim):
        _fail("endpoint_not_xformable", f"{label} is not transformable: {path_value}")
    return prim


def _require_static_transform_chain(prim: Any, *, label: str, UsdGeom: Any) -> None:
    current = prim
    while current.IsValid() and not current.IsPseudoRoot():
        xformable = UsdGeom.Xformable(current)
        if xformable:
            samples = sorted(
                {
                    float(sample)
                    for op in xformable.GetOrderedXformOps()
                    for sample in op.GetAttr().GetTimeSamples()
                }
            )
            if samples:
                _fail(
                    "time_varying_endpoint_transform",
                    f"{label} has a time-sampled transform at {current.GetPath()}: "
                    f"{samples}",
                )
        current = current.GetParent()


def _require_invertible_transform(matrix: Any, *, label: str) -> None:
    values = _matrix_values(matrix)
    determinant = float(matrix.GetDeterminant())
    if (
        any(not math.isfinite(value) for value in values)
        or not math.isfinite(determinant)
        or math.isclose(determinant, 0.0, abs_tol=1e-12)
    ):
        _fail(
            "singular_endpoint_transform",
            f"{label} transform must be finite and invertible",
        )


def _axis_token_and_vector(axis: _Vector3, *, Gf: Any) -> tuple[str, Any]:
    cardinal = {
        "X": (1.0, 0.0, 0.0),
        "Y": (0.0, 1.0, 0.0),
        "Z": (0.0, 0.0, 1.0),
    }
    for token, base in cardinal.items():
        if all(
            math.isclose(abs(left), right, rel_tol=0.0, abs_tol=1e-9)
            for left, right in zip(axis, base, strict=True)
        ):
            return token, Gf.Vec3d(*base)
    # USD exposes only cardinal axis tokens. The authored local rotations map
    # this canonical X basis onto the exact non-cardinal signed stage frame.
    return "X", Gf.Vec3d(1.0, 0.0, 0.0)


def _axis_frame_bases(axis_token: str, *, Gf: Any) -> tuple[Any, Any, Any]:
    bases = {
        "X": (
            Gf.Vec3d(1.0, 0.0, 0.0),
            Gf.Vec3d(0.0, 1.0, 0.0),
            Gf.Vec3d(0.0, 0.0, 1.0),
        ),
        "Y": (
            Gf.Vec3d(0.0, 1.0, 0.0),
            Gf.Vec3d(0.0, 0.0, 1.0),
            Gf.Vec3d(1.0, 0.0, 0.0),
        ),
        "Z": (
            Gf.Vec3d(0.0, 0.0, 1.0),
            Gf.Vec3d(1.0, 0.0, 0.0),
            Gf.Vec3d(0.0, 1.0, 0.0),
        ),
    }
    return bases[axis_token]


def _stage_joint_frame(axis_stage: Any, *, Gf: Any) -> tuple[Any, Any, Any]:
    axis = _normalized_direction(axis_stage, label="planned stage axis")
    cardinals = (
        Gf.Vec3d(1.0, 0.0, 0.0),
        Gf.Vec3d(0.0, 1.0, 0.0),
        Gf.Vec3d(0.0, 0.0, 1.0),
    )
    reference = min(cardinals, key=lambda value: abs(float(Gf.Dot(axis, value))))
    secondary = _normalized_direction(
        reference - axis * Gf.Dot(axis, reference),
        label="planned stage frame secondary direction",
    )
    tertiary = _normalized_direction(
        Gf.Cross(axis, secondary),
        label="planned stage frame tertiary direction",
    )
    return axis, secondary, tertiary


def _local_joint_frame_rotation(
    world_transform: Any,
    *,
    stage_frame: tuple[Any, Any, Any],
    axis_token: str,
    label: str,
    Gf: Any,
) -> _Quaternion:
    inverse = world_transform.GetInverse()
    local_frame = tuple(
        _normalized_direction(
            inverse.TransformDir(direction),
            label=f"{label} local frame direction {index}",
        )
        for index, direction in enumerate(stage_frame)
    )
    if any(
        abs(float(Gf.Dot(local_frame[left], local_frame[right]))) > _FRAME_TOLERANCE
        for left, right in ((0, 1), (0, 2), (1, 2))
    ) or float(Gf.Dot(Gf.Cross(local_frame[0], local_frame[1]), local_frame[2])) < (
        1.0 - _FRAME_TOLERANCE
    ):
        _fail(
            "unsupported_endpoint_joint_frame",
            f"{label} transform cannot represent one shared orthonormal joint frame",
        )

    row_indices = {
        "X": (0, 1, 2),
        "Y": (1, 2, 0),
        "Z": (2, 0, 1),
    }[axis_token]
    matrix = Gf.Matrix3d(1.0)
    for row_index, direction in zip(row_indices, local_frame, strict=True):
        matrix.SetRow(row_index, direction)
    rotation = matrix.ExtractRotation()
    for index, (base, expected_local, expected_stage) in enumerate(
        zip(
            _axis_frame_bases(axis_token, Gf=Gf),
            local_frame,
            stage_frame,
            strict=True,
        )
    ):
        observed_local = rotation.TransformDir(base)
        observed_stage = world_transform.TransformDir(observed_local)
        if not _directions_close(
            observed_local, expected_local
        ) or not _directions_close(
            observed_stage,
            expected_stage,
        ):
            _fail(
                "unsupported_endpoint_joint_frame",
                f"{label} could not represent shared joint frame direction {index}",
            )
    return _rotation_tuple(rotation)


def _validate_authored_preflight(
    stage: Any,
    preflight: _TopologyPreflight,
    *,
    diagnostics: JointRiggerDiagnosticsV1 | None,
    additional_allowed_applied_schemas: Mapping[str, frozenset[str]] | None = None,
    additional_expected_applied_schema_order: Mapping[str, tuple[str, ...]]
    | None = None,
    additional_allowed_authored_properties: Mapping[str, frozenset[str]] | None = None,
    additional_expected_attribute_specs: Mapping[str, Mapping[str, tuple[str, str]]]
    | None = None,
    additional_expected_relationship_targets: Mapping[str, Mapping[str, str]]
    | None = None,
    plan_sha256_override: str | None = None,
) -> None:
    Gf, Sdf, _, _, UsdGeom, UsdPhysics = _pxr_modules()
    if preflight.create_joints_scope:
        _validate_new_joints_scope_snapshot(
            _capture_stage_snapshot(stage, UsdGeom=UsdGeom),
            preflight,
            Sdf=Sdf,
            failure_code="authored_graph_mismatch",
        )
    schema_by_type = {
        "revolute": UsdPhysics.RevoluteJoint,
        "prismatic": UsdPhysics.PrismaticJoint,
        "spherical": UsdPhysics.SphericalJoint,
    }
    diagnostics_by_id: dict[str, JointDiagnosticV1] = {}
    if diagnostics is not None:
        diagnostics_by_id = {
            item.joint_id: item for item in diagnostics.joint_diagnostics
        }
        expected_ids = {item.source.topology.joint_id for item in preflight.joints}
        if set(diagnostics_by_id) != expected_ids:
            _fail(
                "authored_graph_mismatch",
                "joint diagnostics do not exactly cover the authored plan",
            )

    for expected in preflight.joints:
        topology = expected.source.topology
        schema = schema_by_type[topology.joint_type]
        prim = stage.GetPrimAtPath(Sdf.Path(expected.joint_path))
        if (
            not prim
            or not prim.IsValid()
            or not prim.IsActive()
            or not prim.IsDefined()
            or not prim.IsA(schema)
        ):
            _fail(
                "authored_graph_mismatch",
                f"missing expected {topology.joint_type} joint at "
                f"{expected.joint_path}",
            )
        joint = schema(prim)
        additional_allowed_properties = (
            additional_allowed_authored_properties or {}
        ).get(expected.joint_path, frozenset())
        _reject_time_sampled_joint_attributes(
            joint,
            prim,
            expected,
            UsdPhysics=UsdPhysics,
            additional_allowed=additional_allowed_properties,
        )
        _require_single_target(joint.GetBody0Rel(), topology.body0, field="body0")
        _require_single_target(joint.GetBody1Rel(), topology.body1, field="body1")
        _require_authored_float32_vector(
            joint.GetLocalPos0Attr(),
            expected.local_pos0,
            label=f"{expected.joint_path} localPos0",
        )
        _require_authored_float32_vector(
            joint.GetLocalPos1Attr(),
            expected.local_pos1,
            label=f"{expected.joint_path} localPos1",
        )

        if topology.axis_stage is None:
            for attribute in (
                joint.GetAxisAttr(),
                joint.GetLocalRot0Attr(),
                joint.GetLocalRot1Attr(),
            ):
                if attribute.HasAuthoredValueOpinion():
                    _fail(
                        "authored_graph_mismatch",
                        f"spherical joint authored unexpected {attribute.GetName()}",
                    )
        else:
            axis_token = expected.axis_token
            if axis_token is None:  # pragma: no cover - preflight invariant
                _fail(
                    "authored_graph_mismatch",
                    f"preflight omitted axis token at {expected.joint_path}",
                )
            if joint.GetAxisAttr().Get() != axis_token:
                _fail(
                    "authored_graph_mismatch",
                    f"joint axis token mismatch at {expected.joint_path}",
                )
            body0 = stage.GetPrimAtPath(topology.body0)
            body1 = stage.GetPrimAtPath(topology.body1)
            xform_cache = UsdGeom.XformCache()
            stage_frame = _stage_joint_frame(Gf.Vec3d(*topology.axis_stage), Gf=Gf)
            frame_bases = _axis_frame_bases(axis_token, Gf=Gf)
            for body_index, body in enumerate((body0, body1)):
                rotation_attribute = (
                    joint.GetLocalRot0Attr()
                    if body_index == 0
                    else joint.GetLocalRot1Attr()
                )
                if not rotation_attribute.HasAuthoredValueOpinion():
                    _fail(
                        "authored_graph_mismatch",
                        f"{expected.joint_path} localRot{body_index} is not authored",
                    )
                local_rotation = rotation_attribute.Get()
                if local_rotation is None:
                    _fail(
                        "authored_graph_mismatch",
                        f"{expected.joint_path} localRot{body_index} has no value",
                    )
                expected_rotation = (
                    expected.local_rot0 if body_index == 0 else expected.local_rot1
                )
                if expected_rotation is None:  # pragma: no cover - invariant
                    _fail(
                        "authored_graph_mismatch",
                        f"preflight omitted localRot{body_index} at "
                        f"{expected.joint_path}",
                    )
                _require_float32_quaternion_value(
                    local_rotation,
                    expected_rotation,
                    label=f"{expected.joint_path} localRot{body_index}",
                )
                rotation = Gf.Rotation(local_rotation)
                world_transform = xform_cache.GetLocalToWorldTransform(body)
                for frame_index, (base, stage_direction) in enumerate(
                    zip(frame_bases, stage_frame, strict=True)
                ):
                    world_direction = world_transform.TransformDir(
                        rotation.TransformDir(base)
                    )
                    _require_close_vector(
                        world_direction,
                        stage_direction,
                        label=(
                            f"{expected.joint_path} body{body_index} frame "
                            f"direction {frame_index}"
                        ),
                        normalized=True,
                    )

        if topology.joint_type == "spherical":
            _require_authored_value(
                joint.GetConeAngle0LimitAttr(),
                None,
                label="coneAngle0Limit",
            )
            _require_authored_value(
                joint.GetConeAngle1LimitAttr(),
                None,
                label="coneAngle1Limit",
            )
        else:
            _require_authored_value(
                joint.GetLowerLimitAttr(),
                expected.lower_limit,
                label="lowerLimit",
            )
            _require_authored_value(
                joint.GetUpperLimitAttr(),
                expected.upper_limit,
                label="upperLimit",
            )
        additional_allowed = (additional_allowed_applied_schemas or {}).get(
            expected.joint_path, frozenset()
        )
        additional_schema_order = (additional_expected_applied_schema_order or {}).get(
            expected.joint_path
        )
        _validate_drive(
            prim,
            expected,
            UsdPhysics=UsdPhysics,
            additional_allowed=additional_allowed,
        )
        # Combined R2+R3 validation already checked the exact R3 value against
        # its resolved physics plan. The topology projection owns no friction
        # fact, so treat that cross-layer property like the allowed drive API.
        if "physxJoint:jointFriction" not in additional_allowed_properties:
            _validate_joint_friction(prim, expected)
        _validate_joint_applied_schemas(
            prim,
            expected,
            Sdf=Sdf,
            additional_allowed=additional_allowed,
            additional_expected_order=additional_schema_order,
        )
        _validate_joint_authored_properties(
            joint,
            prim,
            expected,
            Sdf=Sdf,
            UsdPhysics=UsdPhysics,
            additional_allowed=additional_allowed_properties,
            additional_expected_attribute_specs=(
                additional_expected_attribute_specs or {}
            ).get(expected.joint_path, {}),
            additional_expected_relationship_targets=(
                additional_expected_relationship_targets or {}
            ).get(expected.joint_path, {}),
        )

        if prim.GetCustomDataByKey("jointRigger:jointId") != topology.joint_id:
            _fail(
                "authored_graph_mismatch",
                f"joint id customData mismatch at {expected.joint_path}",
            )
        expected_plan_sha256 = plan_sha256_override or preflight.plan_sha256
        if prim.GetCustomDataByKey("jointRigger:planSha256") != expected_plan_sha256:
            _fail(
                "authored_graph_mismatch",
                f"plan identity customData mismatch at {expected.joint_path}",
            )
        if (
            prim.GetCustomDataByKey("jointRigger:authoringVersion")
            != _TOPOLOGY_AUTHOR_VERSION
        ):
            _fail(
                "authored_graph_mismatch",
                f"authoring version customData mismatch at {expected.joint_path}",
            )
        expected_decisions: str | None = None
        if diagnostics is not None:
            diagnostic = diagnostics_by_id[topology.joint_id]
            if diagnostic.authored_prim_path != expected.joint_path:
                _fail(
                    "authored_graph_mismatch",
                    f"diagnostics path mismatch for {topology.joint_id!r}",
                )
            expected_decisions = json.dumps(
                [
                    item.model_dump(mode="json", exclude_none=True)
                    for item in diagnostic.field_decisions
                ],
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            if (
                prim.GetCustomDataByKey("jointRigger:fieldDecisions")
                != expected_decisions
            ):
                _fail(
                    "authored_graph_mismatch",
                    f"field-decision provenance mismatch at {expected.joint_path}",
                )
        _validate_joint_authored_metadata(
            prim,
            expected,
            plan_sha256=expected_plan_sha256,
            expected_field_decisions=expected_decisions,
        )


def _validate_no_reshape(
    stage: Any,
    preflight: _TopologyPreflight,
    *,
    normalize_layer_identifiers: bool = False,
    layer_identifier_remap: Mapping[Path, Path] | None = None,
) -> None:
    _, Sdf, _, _, UsdGeom, _ = _pxr_modules()
    observed = _capture_stage_snapshot(stage, UsdGeom=UsdGeom)
    expected_snapshot = preflight.snapshot
    if normalize_layer_identifiers:
        observed = _normalized_snapshot_layer_identifiers(
            observed,
            layer_identifier_remap=layer_identifier_remap,
        )
        expected_snapshot = _normalized_snapshot_layer_identifiers(
            expected_snapshot,
            layer_identifier_remap=layer_identifier_remap,
        )
    observed = _without_allowed_new_inert_overs(
        observed,
        expected_snapshot,
        allowed_paths=_planned_authoring_ancestor_paths(preflight, Sdf=Sdf),
        Sdf=Sdf,
    )
    if (
        observed.default_prim_path != expected_snapshot.default_prim_path
        or not math.isclose(
            observed.meters_per_unit,
            expected_snapshot.meters_per_unit,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or observed.up_axis != expected_snapshot.up_axis
    ):
        _fail(
            "no_reshape_violation",
            "topology authoring changed source stage metadata",
        )
    if preflight.create_joints_scope and preflight.joints_scope_path in observed.prims:
        _validate_new_joints_scope_snapshot(
            observed,
            preflight,
            Sdf=Sdf,
            failure_code="no_reshape_violation",
        )
    allowed_new_paths = {item.joint_path for item in preflight.joints}
    if preflight.create_joints_scope:
        allowed_new_paths.add(preflight.joints_scope_path)
    unexpected = set(observed.prims) - set(expected_snapshot.prims) - allowed_new_paths
    missing = set(expected_snapshot.prims) - set(observed.prims)
    if unexpected or missing:
        _fail(
            "no_reshape_violation",
            "topology authoring changed source prim paths: "
            f"added={sorted(unexpected)}, removed={sorted(missing)}",
        )
    for path, expected in expected_snapshot.prims.items():
        if observed.prims[path] != expected:
            _fail(
                "no_reshape_violation",
                f"topology authoring changed source prim facts at {path}",
            )


def _validate_new_joints_scope_snapshot(
    observed: _StageSnapshot,
    preflight: _TopologyPreflight,
    *,
    Sdf: Any,
    failure_code: str,
) -> None:
    """Require the newly authored container to be one clean ``def Scope``."""

    path = preflight.joints_scope_path
    scope = observed.prims.get(path)
    expected_spec = (
        path,
        (
            ("specifier", _stable_usd_info_value(Sdf.SpecifierDef)),
            ("typeName", _stable_usd_info_value("Scope")),
        ),
    )
    if (
        scope is None
        or scope.type_name != "Scope"
        or not scope.active
        or not scope.defined
        or scope.instanceable
        or scope.applied_schemas
        or scope.raw_api_schemas is not None
        or len(scope.authored_prim_specs) != 1
        or scope.authored_prim_specs[0][1:] != expected_spec
        or scope.authored_property_specs
        or scope.world_transform is not None
    ):
        _fail(
            failure_code,
            f"new joint scope at {path} is not the exact author-owned UsdGeom.Scope",
        )


def _normalized_snapshot_layer_identifiers(
    snapshot: _StageSnapshot,
    *,
    layer_identifier_remap: Mapping[Path, Path] | None,
) -> _StageSnapshot:
    """Resolve descriptor aliases without weakening any authored layer facts."""

    def normalized(identifier: str) -> str:
        path = Path(identifier)
        if not path.is_absolute():
            return identifier
        resolved = path.resolve(strict=False)
        if layer_identifier_remap is not None:
            for projected_path, logical_path in layer_identifier_remap.items():
                if resolved == projected_path.resolve(strict=False):
                    return str(logical_path)
        return str(resolved)

    prims = {
        path: replace(
            prim,
            authored_prim_specs=tuple(
                (normalized(identifier), spec_path, info)
                for identifier, spec_path, info in prim.authored_prim_specs
            ),
            authored_property_specs=tuple(
                (
                    property_name,
                    tuple(
                        (normalized(identifier), spec_path, text)
                        for identifier, spec_path, text in specs
                    ),
                )
                for property_name, specs in prim.authored_property_specs
            ),
        )
        for path, prim in snapshot.prims.items()
    }
    return replace(snapshot, prims=prims)


def _planned_authoring_ancestor_paths(
    preflight: _TopologyPreflight,
    *,
    Sdf: Any,
) -> set[str]:
    """Return exact ancestors that USD may need to over for joint authorship."""

    planned_paths = [item.joint_path for item in preflight.joints]
    planned_paths.append(preflight.joints_scope_path)
    ancestors: set[str] = set()
    for path_value in planned_paths:
        path = Sdf.Path(path_value).GetParentPath()
        while path != Sdf.Path.absoluteRootPath:
            ancestors.add(str(path))
            path = path.GetParentPath()
    return ancestors


def _without_allowed_new_inert_overs(
    observed: _StageSnapshot,
    expected: _StageSnapshot,
    *,
    allowed_paths: set[str],
    Sdf: Any,
) -> _StageSnapshot:
    """Remove only new inert overs required to contain planned joint prims."""

    inert_over_info = (("specifier", _stable_usd_info_value(Sdf.SpecifierOver)),)
    prims = dict(observed.prims)
    for path in allowed_paths:
        observed_prim = prims.get(path)
        expected_prim = expected.prims.get(path)
        if observed_prim is None or expected_prim is None:
            continue
        expected_specs = set(expected_prim.authored_prim_specs)
        retained_specs = tuple(
            spec
            for spec in observed_prim.authored_prim_specs
            if spec in expected_specs or spec[2] != inert_over_info
        )
        if retained_specs != observed_prim.authored_prim_specs:
            prims[path] = replace(
                observed_prim,
                authored_prim_specs=retained_specs,
            )
    return replace(observed, prims=prims)


def _capture_stage_snapshot(stage: Any, *, UsdGeom: Any) -> _StageSnapshot:
    xform_cache = UsdGeom.XformCache()
    prims: dict[str, _PrimSnapshot] = {}
    for prim in _bounded_traverse_all(
        stage,
        failure_code="stage_snapshot_scan_limit_exceeded",
        purpose="joint-rigger stage snapshot",
    ):
        xformable = UsdGeom.Xformable(prim)
        transform = None
        if xformable:
            transform = _matrix_values(xform_cache.GetLocalToWorldTransform(prim))
        raw_api_schemas = prim.GetMetadata("apiSchemas")
        prims[str(prim.GetPath())] = _PrimSnapshot(
            type_name=str(prim.GetTypeName()),
            active=bool(prim.IsActive()),
            defined=bool(prim.IsDefined()),
            instanceable=bool(prim.IsInstanceable()),
            applied_schemas=tuple(str(item) for item in prim.GetAppliedSchemas()),
            raw_api_schemas=(None if raw_api_schemas is None else str(raw_api_schemas)),
            authored_prim_specs=tuple(
                (
                    str(spec.layer.identifier),
                    str(spec.path),
                    tuple(
                        (str(key), _stable_usd_info_value(spec.GetInfo(key)))
                        for key in sorted(spec.ListInfoKeys(), key=str)
                    ),
                )
                for spec in prim.GetPrimStack()
            ),
            authored_property_specs=tuple(
                (
                    str(prop.GetName()),
                    tuple(
                        (
                            str(spec.layer.identifier),
                            str(spec.path),
                            str(spec.GetAsText()),
                        )
                        for spec in prop.GetPropertyStack()
                    ),
                )
                for prop in sorted(
                    prim.GetAuthoredProperties(),
                    key=lambda item: str(item.GetName()),
                )
            ),
            world_transform=transform,
        )
    default_prim = stage.GetDefaultPrim()
    return _StageSnapshot(
        default_prim_path=str(default_prim.GetPath()),
        meters_per_unit=float(UsdGeom.GetStageMetersPerUnit(stage)),
        up_axis=str(UsdGeom.GetStageUpAxis(stage)),
        prims=prims,
    )


def _stable_usd_info_value(value: Any) -> str:
    """Return an address-free representation of one authored USD info value."""

    value_type = f"{type(value).__module__}.{type(value).__qualname__}"
    if isinstance(value, dict):
        items = sorted(
            (
                _stable_usd_info_value(key),
                _stable_usd_info_value(item),
            )
            for key, item in value.items()
        )
        body = ",".join(f"{key}={item}" for key, item in items)
        return f"{value_type}:{{{body}}}"
    if isinstance(value, list | tuple):
        body = ",".join(_stable_usd_info_value(item) for item in value)
        return f"{value_type}:[{body}]"
    return f"{value_type}:{value}"


def _validate_drive(
    prim: Any,
    expected: _PreparedJoint,
    *,
    UsdPhysics: Any,
    additional_allowed: frozenset[str] = frozenset(),
) -> None:
    source_drive = expected.source.drive
    drive_tokens = sorted(
        str(token)
        for token in prim.GetAppliedSchemas()
        if str(token).startswith("PhysicsDriveAPI:")
    )
    if source_drive is None:
        if set(drive_tokens) - additional_allowed:
            _fail(
                "authored_graph_mismatch",
                f"joint {expected.joint_path} authored an unexpected drive",
            )
        return
    expected_token = f"PhysicsDriveAPI:{expected.drive_instance}"
    if drive_tokens != [expected_token]:
        _fail(
            "authored_graph_mismatch",
            f"joint {expected.joint_path} drive instances do not match",
        )
    drive = UsdPhysics.DriveAPI.Get(prim, expected.drive_instance)
    values = {
        "type": (drive.GetTypeAttr(), source_drive.drive_type),
        "stiffness": (drive.GetStiffnessAttr(), source_drive.stiffness),
        "damping": (drive.GetDampingAttr(), source_drive.damping),
        "maxForce": (drive.GetMaxForceAttr(), source_drive.max_force),
        "targetPosition": (
            drive.GetTargetPositionAttr(),
            source_drive.target_position,
        ),
        "targetVelocity": (
            drive.GetTargetVelocityAttr(),
            source_drive.target_velocity,
        ),
    }
    for label, (attribute, expected_value) in values.items():
        if not attribute.HasAuthoredValueOpinion():
            _fail(
                "authored_graph_mismatch",
                f"joint {expected.joint_path} drive omitted {label}",
            )
        observed = attribute.Get()
        if observed is None:
            _fail(
                "authored_graph_mismatch",
                f"joint {expected.joint_path} drive {label} has no value",
            )
        if isinstance(expected_value, str):
            matches = str(observed) == expected_value
        else:
            matches = float(observed) == _float32_round_trip(
                expected_value,
                label=f"{expected.joint_path} drive {label}",
            )
        if not matches:
            _fail(
                "authored_graph_mismatch",
                f"joint {expected.joint_path} drive {label} does not match",
            )
    max_velocity = prim.GetAttribute("physxJoint:maxJointVelocity")
    _require_authored_value(
        max_velocity,
        source_drive.max_joint_velocity,
        label="maxJointVelocity",
    )


def _validate_joint_friction(prim: Any, expected: _PreparedJoint) -> None:
    friction = expected.source.joint_friction
    _require_authored_value(
        prim.GetAttribute("physxJoint:jointFriction"),
        friction.coefficient if friction is not None else None,
        label="jointFriction",
    )


def _expected_joint_api_schemas(expected: _PreparedJoint) -> tuple[str, ...]:
    """Return the one canonical raw API-schema list authored by WP-R2."""

    allowed: list[str] = []
    if expected.drive_instance is not None:
        allowed.append(f"PhysicsDriveAPI:{expected.drive_instance}")
    if (
        expected.source.joint_friction is not None
        or expected.source.drive is not None
        and expected.source.drive.max_joint_velocity is not None
    ):
        allowed.append("PhysxJointAPI")
    return tuple(allowed)


def _validate_joint_applied_schemas(
    prim: Any,
    expected: _PreparedJoint,
    *,
    Sdf: Any,
    additional_allowed: frozenset[str] = frozenset(),
    additional_expected_order: tuple[str, ...] | None = None,
) -> None:
    """Require exact effective and canonical raw list-op schema authorship."""

    topology_tokens = _expected_joint_api_schemas(expected)
    allowed = set(topology_tokens) | set(additional_allowed)
    if additional_allowed and additional_expected_order is None:
        _fail(
            "authored_graph_mismatch",
            f"joint {expected.joint_path} additional applied schemas lack a "
            "canonical raw ordering contract",
        )
    expected_raw_order = topology_tokens + (additional_expected_order or ())
    if (
        len(expected_raw_order) != len(set(expected_raw_order))
        or set(expected_raw_order) != allowed
    ):
        _fail(
            "authored_graph_mismatch",
            f"joint {expected.joint_path} applied-schema ordering contract does "
            "not exactly cover the allowed schemas",
        )
    observed = _applied_schema_tokens(prim)
    if observed != allowed:
        _fail(
            "authored_graph_mismatch",
            f"joint {expected.joint_path} applied schemas do not exactly match the "
            f"plan: unexpected={sorted(observed - allowed)}, "
            f"missing={sorted(allowed - observed)}",
        )
    prim_stack = tuple(prim.GetPrimStack())
    if len(prim_stack) != 1:
        _fail(
            "authored_graph_mismatch",
            f"joint {expected.joint_path} must have exactly one authored PrimSpec; "
            f"found {len(prim_stack)}",
        )
    spec = prim_stack[0]
    expected_info_keys = {"customData", "specifier", "typeName"}
    if allowed:
        expected_info_keys.add("apiSchemas")
    observed_info_keys = {str(key) for key in spec.ListInfoKeys()}
    if observed_info_keys != expected_info_keys:
        _fail(
            "authored_graph_mismatch",
            f"joint {expected.joint_path} PrimSpec metadata does not exactly match "
            f"the author-owned shape: unexpected="
            f"{sorted(observed_info_keys - expected_info_keys)}, missing="
            f"{sorted(expected_info_keys - observed_info_keys)}",
        )
    if (
        spec.specifier != Sdf.SpecifierDef
        or str(spec.typeName)
        != (_JOINT_SCHEMA_TYPE_NAMES[expected.source.topology.joint_type])
    ):
        _fail(
            "authored_graph_mismatch",
            f"joint {expected.joint_path} has unexpected raw PrimSpec type authorship",
        )
    if not allowed:
        return
    raw_schemas = spec.GetInfo("apiSchemas")
    prepended_tokens = (
        tuple(str(item) for item in raw_schemas.prependedItems)
        if isinstance(raw_schemas, Sdf.TokenListOp)
        else ()
    )
    if not isinstance(raw_schemas, Sdf.TokenListOp) or (
        raw_schemas.isExplicit
        or tuple(str(item) for item in raw_schemas.explicitItems)
        or tuple(str(item) for item in raw_schemas.addedItems)
        or tuple(str(item) for item in raw_schemas.appendedItems)
        or tuple(str(item) for item in raw_schemas.deletedItems)
        or tuple(str(item) for item in raw_schemas.orderedItems)
        or len(prepended_tokens) != len(set(prepended_tokens))
        or prepended_tokens != expected_raw_order
    ):
        _fail(
            "authored_graph_mismatch",
            f"joint {expected.joint_path} raw apiSchemas list-op does not exactly "
            "match canonical author-owned prepends",
        )


def _validate_joint_authored_metadata(
    prim: Any,
    expected: _PreparedJoint,
    *,
    plan_sha256: str,
    expected_field_decisions: str | None,
) -> None:
    """Reject every customData key outside the exact WP-R2 namespace shape."""

    field_decisions = prim.GetCustomDataByKey("jointRigger:fieldDecisions")
    if expected_field_decisions is None:
        field_decisions = _require_canonical_field_decisions(
            field_decisions,
            joint_path=expected.joint_path,
        )
    expected_custom_data = {
        "jointRigger": {
            "authoringVersion": _TOPOLOGY_AUTHOR_VERSION,
            "fieldDecisions": expected_field_decisions or field_decisions,
            "jointId": expected.source.topology.joint_id,
            "planSha256": plan_sha256,
        }
    }
    # Applied-schema validation already requires exactly one authored PrimSpec.
    # Inspect that authored opinion rather than composed schema fallback metadata.
    authored_custom_data = prim.GetPrimStack()[0].GetInfo("customData")
    if authored_custom_data != expected_custom_data:
        _fail(
            "authored_graph_mismatch",
            f"joint {expected.joint_path} customData does not exactly match the "
            "author-owned metadata",
        )


def _require_canonical_field_decisions(value: Any, *, joint_path: str) -> str:
    """Validate report-less field-decision metadata without weakening its shape."""

    if not isinstance(value, str):
        _fail(
            "authored_graph_mismatch",
            f"joint {joint_path} field-decision customData is not a string",
        )
    try:
        payload = json.loads(value)
        if not isinstance(payload, list):
            raise ValueError("field decisions must be a list")
        decisions = tuple(
            FieldDecisionV1.model_validate_json(
                json.dumps(item, separators=(",", ":"), ensure_ascii=False)
            )
            for item in payload
        )
        canonical = json.dumps(
            [item.model_dump(mode="json", exclude_none=True) for item in decisions],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        _fail(
            "authored_graph_mismatch",
            f"joint {joint_path} field-decision customData is invalid: {exc}",
        )
    if canonical != value:
        _fail(
            "authored_graph_mismatch",
            f"joint {joint_path} field-decision customData is not canonical",
        )
    return value


def _validate_joint_authored_properties(
    joint: Any,
    prim: Any,
    expected: _PreparedJoint,
    *,
    Sdf: Any,
    UsdPhysics: Any,
    additional_allowed: frozenset[str] = frozenset(),
    additional_expected_attribute_specs: Mapping[str, tuple[str, str]] | None = None,
    additional_expected_relationship_targets: Mapping[str, str] | None = None,
) -> None:
    """Require the joint to contain exactly the plan-owned properties."""

    allowed = _joint_allowed_authored_properties(
        joint,
        prim,
        expected,
        UsdPhysics=UsdPhysics,
        additional_allowed=additional_allowed,
    )
    observed = {str(prop.GetName()) for prop in prim.GetAuthoredProperties()}
    if observed != allowed:
        _fail(
            "authored_graph_mismatch",
            f"joint {expected.joint_path} authored properties do not exactly "
            f"match the plan: unexpected={sorted(observed - allowed)}, "
            f"missing={sorted(allowed - observed)}",
        )
    _validate_joint_raw_property_specs(
        joint,
        prim,
        expected,
        allowed=allowed,
        Sdf=Sdf,
        UsdPhysics=UsdPhysics,
        additional_expected_attribute_specs=(additional_expected_attribute_specs or {}),
        additional_expected_relationship_targets=(
            additional_expected_relationship_targets or {}
        ),
    )


def _validate_joint_raw_property_specs(
    joint: Any,
    prim: Any,
    expected: _PreparedJoint,
    *,
    allowed: set[str],
    Sdf: Any,
    UsdPhysics: Any,
    additional_expected_attribute_specs: Mapping[str, tuple[str, str]] | None = None,
    additional_expected_relationship_targets: Mapping[str, str] | None = None,
) -> None:
    """Require canonical raw AttributeSpec and RelationshipSpec authorship."""

    additional_attribute_specs = additional_expected_attribute_specs or {}
    additional_relationship_targets = additional_expected_relationship_targets or {}
    spec = tuple(prim.GetPrimStack())[0]
    properties = {str(prop.name): prop for prop in spec.properties}
    if set(properties) != allowed:
        _fail(
            "authored_graph_mismatch",
            f"joint {expected.joint_path} raw property specs do not exactly match "
            f"the plan: unexpected={sorted(set(properties) - allowed)}, "
            f"missing={sorted(allowed - set(properties))}",
        )

    relationship_targets = {
        str(joint.GetBody0Rel().GetName()): expected.source.topology.body0,
        str(joint.GetBody1Rel().GetName()): expected.source.topology.body1,
    }
    if set(relationship_targets).intersection(additional_relationship_targets):
        _fail(
            "authored_graph_mismatch",
            f"joint {expected.joint_path} additional relationship contract "
            "overlaps topology-owned relationships",
        )
    relationship_targets.update(additional_relationship_targets)
    attribute_shapes: dict[str, tuple[Any, Any]] = {
        str(joint.GetLocalPos0Attr().GetName()): (
            Sdf.ValueTypeNames.Point3f,
            Sdf.VariabilityVarying,
        ),
        str(joint.GetLocalPos1Attr().GetName()): (
            Sdf.ValueTypeNames.Point3f,
            Sdf.VariabilityVarying,
        ),
    }
    if expected.axis_token is not None:
        attribute_shapes.update(
            {
                str(joint.GetAxisAttr().GetName()): (
                    Sdf.ValueTypeNames.Token,
                    Sdf.VariabilityUniform,
                ),
                str(joint.GetLocalRot0Attr().GetName()): (
                    Sdf.ValueTypeNames.Quatf,
                    Sdf.VariabilityVarying,
                ),
                str(joint.GetLocalRot1Attr().GetName()): (
                    Sdf.ValueTypeNames.Quatf,
                    Sdf.VariabilityVarying,
                ),
            }
        )
    if expected.lower_limit is not None:
        attribute_shapes[str(joint.GetLowerLimitAttr().GetName())] = (
            Sdf.ValueTypeNames.Float,
            Sdf.VariabilityVarying,
        )
    if expected.upper_limit is not None:
        attribute_shapes[str(joint.GetUpperLimitAttr().GetName())] = (
            Sdf.ValueTypeNames.Float,
            Sdf.VariabilityVarying,
        )
    if expected.drive_instance is not None:
        drive = UsdPhysics.DriveAPI.Get(prim, expected.drive_instance)
        attribute_shapes.update(
            {
                str(drive.GetTypeAttr().GetName()): (
                    Sdf.ValueTypeNames.Token,
                    Sdf.VariabilityUniform,
                ),
                **{
                    str(attribute.GetName()): (
                        Sdf.ValueTypeNames.Float,
                        Sdf.VariabilityVarying,
                    )
                    for attribute in (
                        drive.GetStiffnessAttr(),
                        drive.GetDampingAttr(),
                        drive.GetMaxForceAttr(),
                        drive.GetTargetPositionAttr(),
                        drive.GetTargetVelocityAttr(),
                    )
                },
            }
        )
        if expected.source.drive is not None and (
            expected.source.drive.max_joint_velocity is not None
        ):
            attribute_shapes["physxJoint:maxJointVelocity"] = (
                Sdf.ValueTypeNames.Float,
                Sdf.VariabilityVarying,
            )
    if expected.source.joint_friction is not None:
        attribute_shapes["physxJoint:jointFriction"] = (
            Sdf.ValueTypeNames.Float,
            Sdf.VariabilityVarying,
        )
    variability_by_name = {
        "uniform": Sdf.VariabilityUniform,
        "varying": Sdf.VariabilityVarying,
    }
    type_by_name = {
        "float": Sdf.ValueTypeNames.Float,
        "token": Sdf.ValueTypeNames.Token,
    }
    if set(attribute_shapes).intersection(additional_attribute_specs):
        _fail(
            "authored_graph_mismatch",
            f"joint {expected.joint_path} additional attribute contract overlaps "
            "topology-owned attributes",
        )
    for name, (type_name, variability_name) in additional_attribute_specs.items():
        try:
            attribute_shapes[name] = (
                type_by_name[type_name],
                variability_by_name[variability_name],
            )
        except KeyError:
            _fail(
                "authored_graph_mismatch",
                f"joint {expected.joint_path} additional attribute {name!r} has "
                "an unsupported raw specification contract",
            )
    expected_property_names = set(relationship_targets) | set(attribute_shapes)
    if expected_property_names != allowed:
        _fail(
            "authored_graph_mismatch",
            f"joint {expected.joint_path} raw property specification contract "
            f"does not exactly cover the authored allowlist: unexpected="
            f"{sorted(expected_property_names - allowed)}, missing="
            f"{sorted(allowed - expected_property_names)}",
        )

    for name, target in relationship_targets.items():
        property_spec = properties[name]
        if not isinstance(property_spec, Sdf.RelationshipSpec) or (
            {str(key) for key in property_spec.ListInfoKeys()}
            != {"custom", "targetPaths", "variability"}
            or property_spec.custom
            or property_spec.variability != Sdf.VariabilityUniform
        ):
            _fail(
                "authored_graph_mismatch",
                f"joint {expected.joint_path} relationship {name!r} has "
                "noncanonical raw metadata",
            )
        targets = property_spec.GetInfo("targetPaths")
        if not isinstance(targets, Sdf.PathListOp) or (
            not targets.isExplicit
            or tuple(str(item) for item in targets.explicitItems) != (target,)
            or tuple(targets.addedItems)
            or tuple(targets.prependedItems)
            or tuple(targets.appendedItems)
            or tuple(targets.deletedItems)
            or tuple(targets.orderedItems)
        ):
            _fail(
                "authored_graph_mismatch",
                f"joint {expected.joint_path} relationship {name!r} has a "
                "noncanonical raw target list-op",
            )

    for name, (type_name, variability) in attribute_shapes.items():
        property_spec = properties[name]
        if not isinstance(property_spec, Sdf.AttributeSpec) or (
            {str(key) for key in property_spec.ListInfoKeys()}
            != {"custom", "default", "typeName", "variability"}
            or property_spec.custom
            or property_spec.typeName != type_name
            or property_spec.variability != variability
        ):
            _fail(
                "authored_graph_mismatch",
                f"joint {expected.joint_path} attribute {name!r} has "
                "noncanonical raw metadata",
            )


def _joint_allowed_authored_properties(
    joint: Any,
    prim: Any,
    expected: _PreparedJoint,
    *,
    UsdPhysics: Any,
    additional_allowed: frozenset[str] = frozenset(),
) -> set[str]:
    """Return the exact property allowlist for one prepared joint."""

    allowed = set(additional_allowed)
    allowed.update(
        {
            str(joint.GetBody0Rel().GetName()),
            str(joint.GetBody1Rel().GetName()),
            str(joint.GetLocalPos0Attr().GetName()),
            str(joint.GetLocalPos1Attr().GetName()),
        }
    )
    if expected.axis_token is not None:
        allowed.update(
            {
                str(joint.GetAxisAttr().GetName()),
                str(joint.GetLocalRot0Attr().GetName()),
                str(joint.GetLocalRot1Attr().GetName()),
            }
        )
    if expected.lower_limit is not None:
        allowed.add(str(joint.GetLowerLimitAttr().GetName()))
    if expected.upper_limit is not None:
        allowed.add(str(joint.GetUpperLimitAttr().GetName()))
    if expected.drive_instance is not None:
        drive = UsdPhysics.DriveAPI.Get(prim, expected.drive_instance)
        allowed.update(
            str(attribute.GetName())
            for attribute in (
                drive.GetTypeAttr(),
                drive.GetStiffnessAttr(),
                drive.GetDampingAttr(),
                drive.GetMaxForceAttr(),
                drive.GetTargetPositionAttr(),
                drive.GetTargetVelocityAttr(),
            )
        )
        if expected.source.drive is not None and (
            expected.source.drive.max_joint_velocity is not None
        ):
            allowed.add("physxJoint:maxJointVelocity")
    if expected.source.joint_friction is not None:
        allowed.add("physxJoint:jointFriction")
    return allowed


def _reject_time_sampled_joint_attributes(
    joint: Any,
    prim: Any,
    expected: _PreparedJoint,
    *,
    UsdPhysics: Any,
    additional_allowed: frozenset[str] = frozenset(),
) -> None:
    """Reject animation on every attribute the owned plan is allowed to author."""

    allowed = _joint_allowed_authored_properties(
        joint,
        prim,
        expected,
        UsdPhysics=UsdPhysics,
        additional_allowed=additional_allowed,
    )
    for property_name in sorted(allowed):
        attribute = prim.GetAttribute(property_name)
        if not attribute or not attribute.IsValid():
            continue
        samples = sorted(float(sample) for sample in attribute.GetTimeSamples())
        if samples:
            _fail(
                "authored_graph_mismatch",
                f"joint {expected.joint_path} plan-owned attribute "
                f"{property_name!r} has time samples: {samples}",
            )


def _applied_schema_tokens(prim: Any) -> set[str]:
    """Return registered schemas plus raw authored optional-runtime tokens."""

    tokens = {str(token) for token in prim.GetAppliedSchemas()}
    metadata = prim.GetMetadata("apiSchemas")
    if metadata is not None:
        try:
            tokens.update(str(token) for token in metadata.GetAppliedItems())
        except AttributeError:
            if isinstance(metadata, list | tuple):
                tokens.update(str(token) for token in metadata)
    return tokens


def _require_single_target(relationship: Any, expected: str, *, field: str) -> None:
    targets = [str(target) for target in relationship.GetTargets()]
    if targets != [expected]:
        _fail(
            "authored_graph_mismatch",
            f"authored {field} targets do not match: {targets} != [{expected}]",
        )


def _require_authored_value(
    attribute: Any,
    expected: float | None,
    *,
    label: str,
) -> None:
    authored = bool(attribute and attribute.HasAuthoredValueOpinion())
    if expected is None:
        if authored:
            _fail("authored_graph_mismatch", f"authored unexpected {label}")
        return
    stored_expected = _float32_round_trip(expected, label=label)
    observed = attribute.Get() if authored else None
    if observed is None or float(observed) != stored_expected:
        _fail("authored_graph_mismatch", f"authored {label} does not match plan")


def _require_authored_float32_vector(
    attribute: Any,
    expected: _Vector3,
    *,
    label: str,
) -> None:
    if not attribute or not attribute.HasAuthoredValueOpinion():
        _fail("authored_graph_mismatch", f"authored {label} is missing")
    observed = attribute.Get()
    if observed is None or _vec3_tuple(observed) != expected:
        _fail("authored_graph_mismatch", f"authored {label} does not match plan")


def _require_float32_quaternion_value(
    observed: Any,
    expected: _Quaternion,
    *,
    label: str,
) -> None:
    imaginary = observed.GetImaginary()
    value = (
        float(observed.GetReal()),
        (float(imaginary[0]), float(imaginary[1]), float(imaginary[2])),
    )
    if value != expected:
        _fail("authored_graph_mismatch", f"authored {label} does not match plan")


def _require_close_vector(
    actual: Any,
    expected: Any,
    *,
    label: str,
    normalized: bool = False,
) -> None:
    actual_values = _vec3_tuple(actual)
    expected_values = _vec3_tuple(expected)
    if normalized:
        actual_values = _normalized_tuple(actual_values, label=label)
        expected_values = _normalized_tuple(expected_values, label=label)
    if any(
        not math.isclose(left, right, rel_tol=1e-6, abs_tol=_FRAME_TOLERANCE)
        for left, right in zip(actual_values, expected_values, strict=True)
    ):
        _fail(
            "authored_graph_mismatch",
            f"authored {label} mismatch: {actual_values} != {expected_values}",
        )


def _normalized_direction(vector: Any, *, label: str) -> Any:
    length = float(vector.GetLength())
    if not math.isfinite(length) or math.isclose(length, 0.0, abs_tol=1e-12):
        _fail("singular_endpoint_transform", f"{label} cannot be normalized")
    return vector / length


def _directions_close(left: Any, right: Any) -> bool:
    left_values = _normalized_tuple(_vec3_tuple(left), label="observed direction")
    right_values = _normalized_tuple(_vec3_tuple(right), label="expected direction")
    return all(
        math.isclose(
            left_component,
            right_component,
            rel_tol=0.0,
            abs_tol=_FRAME_TOLERANCE,
        )
        for left_component, right_component in zip(
            left_values,
            right_values,
            strict=True,
        )
    )


def _normalized_tuple(value: _Vector3, *, label: str) -> _Vector3:
    length = math.sqrt(sum(component * component for component in value))
    if not math.isfinite(length) or math.isclose(length, 0.0, abs_tol=1e-12):
        _fail("authored_graph_mismatch", f"{label} cannot be normalized")
    return cast(_Vector3, tuple(component / length for component in value))


def _rotation_tuple(rotation: Any) -> _Quaternion:
    quaternion = rotation.GetQuat()
    imaginary = quaternion.GetImaginary()
    return (
        float(quaternion.GetReal()),
        (float(imaginary[0]), float(imaginary[1]), float(imaginary[2])),
    )


def _vec3_tuple(value: Any) -> _Vector3:
    return (float(value[0]), float(value[1]), float(value[2]))


@dataclass(frozen=True)
class JointRiggerStageSnapshot:
    """Immutable source-structure snapshot around a schema-only write.

    ``prims`` records path, parent, concrete type, and composition-state flags.
    ``joint_topology`` records the type, endpoint relationships, axis, local
    frames, and limits for every USD physics joint. ``world_transforms``
    records default-time world matrices for every xformable prim, not only the
    planned rigid bodies.
    """

    prims: tuple[tuple[str, str, str, bool, bool, bool, bool], ...]
    joint_topology: tuple[tuple[Any, ...], ...]
    world_transforms: tuple[tuple[str, tuple[float, ...]], ...]


@dataclass(frozen=True)
class JointRiggerPhysicsSchemaSnapshot:
    """Exact composed physics-schema state for one opened stage artifact.

    The snapshot covers every applied ``Physics``/``Physx`` API token and every
    authored physics, state, drive, and PhysX property, including authored
    connection state and composed connection targets. It is intentionally
    separate from :class:`JointRiggerStageSnapshot`: schema authoring is allowed
    to change these rows, while hierarchy, joint topology, and transforms must
    remain unchanged.
    """

    prims: tuple[
        tuple[
            str,
            tuple[str, ...],
            tuple[tuple[str, str, tuple[Any, ...]], ...],
        ],
        ...,
    ]


def capture_joint_rigger_stage_snapshot(stage: Any) -> JointRiggerStageSnapshot:
    """Capture hierarchy, joint topology, and default-time world transforms."""

    if stage is None:
        raise JointRiggerContractError("invalid_stage", "stage must not be None")

    try:
        from pxr import UsdGeom, UsdPhysics
    except ImportError as exc:  # pragma: no cover - optional-runtime guard
        raise JointRiggerContractError(
            "openusd_unavailable",
            "OpenUSD bindings are required for physics schema validation",
        ) from exc

    prim_rows: list[tuple[str, str, str, bool, bool, bool, bool]] = []
    topology_rows: list[tuple[Any, ...]] = []
    transform_rows: list[tuple[str, tuple[float, ...]]] = []
    xform_cache = UsdGeom.XformCache()

    try:
        for prim in _bounded_traverse_all(
            stage,
            failure_code="stage_snapshot_scan_limit_exceeded",
            purpose="joint-rigger stage snapshot",
        ):
            path = str(prim.GetPath())
            parent = prim.GetParent()
            parent_path = "" if parent.IsPseudoRoot() else str(parent.GetPath())
            prim_rows.append(
                (
                    path,
                    parent_path,
                    str(prim.GetTypeName()),
                    bool(prim.IsActive()),
                    bool(prim.IsDefined()),
                    bool(prim.IsInstance()),
                    bool(prim.IsInstanceable()),
                )
            )

            if prim.IsA(UsdPhysics.Joint):
                topology_rows.append(
                    (
                        path,
                        str(prim.GetTypeName()),
                        _relationship_targets(prim, "physics:body0"),
                        _relationship_targets(prim, "physics:body1"),
                        tuple(
                            _attribute_snapshot(prim, name)
                            for name in _JOINT_FACT_ATTRIBUTES
                        ),
                    )
                )

            xformable = UsdGeom.Xformable(prim)
            if xformable:
                matrix = xform_cache.GetLocalToWorldTransform(prim)
                transform_rows.append((path, _matrix_values(matrix)))
    except JointRiggerContractError:
        raise
    except Exception as exc:
        raise JointRiggerContractError(
            "stage_traversal_failed",
            f"could not traverse stage: {exc}",
        ) from exc

    return JointRiggerStageSnapshot(
        prims=tuple(sorted(prim_rows)),
        joint_topology=tuple(sorted(topology_rows)),
        world_transforms=tuple(sorted(transform_rows)),
    )


def capture_joint_rigger_physics_schema_snapshot(
    stage: Any,
) -> JointRiggerPhysicsSchemaSnapshot:
    """Capture exact composed values for the owned physics-schema namespace."""

    if stage is None:
        raise JointRiggerContractError("invalid_stage", "stage must not be None")

    prim_rows: list[
        tuple[
            str,
            tuple[str, ...],
            tuple[tuple[str, str, tuple[Any, ...]], ...],
        ]
    ] = []
    property_prefixes = ("physics:", "physx", "state:", "drive:")
    for prim in _bounded_traverse_all(
        stage,
        failure_code="stage_snapshot_scan_limit_exceeded",
        purpose="joint-rigger physics-schema snapshot",
    ):
        schemas = tuple(
            sorted(
                token
                for token in _applied_schema_tokens(prim)
                if token.startswith(("Physics", "Physx"))
            )
        )
        properties: list[tuple[str, str, tuple[Any, ...]]] = []
        for prop in prim.GetAuthoredProperties():
            name = str(prop.GetName())
            if not name.startswith(property_prefixes):
                continue
            attribute = prim.GetAttribute(name)
            if attribute:
                properties.append((name, "attribute", _attribute_snapshot(prim, name)))
                continue
            relationship = prim.GetRelationship(name)
            if relationship:
                properties.append(
                    (
                        name,
                        "relationship",
                        tuple(str(path) for path in relationship.GetTargets()),
                    )
                )
        if schemas or properties:
            prim_rows.append(
                (
                    str(prim.GetPath()),
                    schemas,
                    tuple(sorted(properties)),
                )
            )
    return JointRiggerPhysicsSchemaSnapshot(prims=tuple(sorted(prim_rows)))


def validate_joint_rigger_stage_preservation(
    before: JointRiggerStageSnapshot,
    after: JointRiggerStageSnapshot,
) -> None:
    """Fail when schema authoring changed source structure or transforms."""

    if before.prims != after.prims:
        raise JointRiggerContractError(
            "source_hierarchy_changed",
            _first_difference("prim structure", before.prims, after.prims),
        )
    if before.joint_topology != after.joint_topology:
        raise JointRiggerContractError(
            "joint_topology_changed",
            _first_difference(
                "joint topology",
                before.joint_topology,
                after.joint_topology,
            ),
        )

    before_transforms = dict(before.world_transforms)
    after_transforms = dict(after.world_transforms)
    if before_transforms.keys() != after_transforms.keys():
        raise JointRiggerContractError(
            "world_transform_set_changed",
            "xformable prim paths changed during schema authoring",
        )
    for path in sorted(before_transforms):
        left = before_transforms[path]
        right = after_transforms[path]
        if not all(
            math.isclose(
                first,
                second,
                rel_tol=_MATRIX_TOLERANCE,
                abs_tol=_MATRIX_TOLERANCE,
            )
            for first, second in zip(left, right, strict=True)
        ):
            raise JointRiggerContractError(
                "world_transform_changed",
                f"default-time world transform changed at {path}",
            )


def physics_schema_counts(stage: Any) -> dict[str, int]:
    """Count composed applied physics API schemas on one stage artifact."""

    if stage is None:
        raise JointRiggerContractError("invalid_stage", "stage must not be None")
    counts: dict[str, int] = {}
    for prim in _bounded_traverse_all(
        stage,
        failure_code="physics_schema_count_scan_limit_exceeded",
        purpose="physics-schema counting",
    ):
        for token in _applied_schema_tokens(prim):
            if token.startswith(("Physics", "Physx")):
                counts[token] = counts.get(token, 0) + 1
    return dict(sorted(counts.items()))


def _relationship_targets(prim: Any, name: str) -> tuple[str, ...]:
    relationship = prim.GetRelationship(name)
    if not relationship:
        return ()
    return tuple(str(path) for path in relationship.GetTargets())


def _attribute_snapshot(prim: Any, name: str) -> tuple[Any, ...]:
    attribute = prim.GetAttribute(name)
    if not attribute:
        return (name, False, False, (), None, ())
    samples = tuple(float(value) for value in attribute.GetTimeSamples())
    return (
        name,
        bool(attribute.HasAuthoredValueOpinion()),
        bool(attribute.HasAuthoredConnections()),
        tuple(str(path) for path in attribute.GetConnections()),
        _snapshot_value(attribute.Get()),
        tuple((sample, _snapshot_value(attribute.Get(sample))) for sample in samples),
    )


def _snapshot_value(value: Any) -> Any:
    if hasattr(value, "GetReal") and hasattr(value, "GetImaginary"):
        imaginary = value.GetImaginary()
        return (
            float(value.GetReal()),
            float(imaginary[0]),
            float(imaginary[1]),
            float(imaginary[2]),
        )
    if isinstance(value, str | bool | int | float) or value is None:
        return value
    try:
        return tuple(_snapshot_value(item) for item in value)
    except TypeError:
        return str(value)


def _matrix_values(matrix: Any) -> tuple[float, ...]:
    return tuple(float(matrix[row][column]) for row in range(4) for column in range(4))


def _anchor_vectors_close(left: _Vector3, right: _Vector3) -> bool:
    """Compare stage anchors with absolute tolerance plus double-ULP headroom."""

    for left_component, right_component in zip(left, right, strict=True):
        if not math.isfinite(left_component) or not math.isfinite(right_component):
            return False
        magnitude = max(abs(left_component), abs(right_component))
        tolerance = max(_FRAME_TOLERANCE, 8.0 * math.ulp(magnitude))
        if not math.isclose(
            left_component,
            right_component,
            rel_tol=0.0,
            abs_tol=tolerance,
        ):
            return False
    return True


def _vector_distance(left: _Vector3, right: _Vector3) -> float:
    """Return the Euclidean distance between two finite stage-space vectors."""

    return math.sqrt(
        sum(
            (left_component - right_component) ** 2
            for left_component, right_component in zip(left, right, strict=True)
        )
    )


def _reconcile_float32_local_anchors(
    body0_xform: Any,
    body1_xform: Any,
    *,
    requested_anchor: _Vector3,
    local_pos0: _Vector3,
    local_pos1: _Vector3,
    explicit_anchor: bool,
    label: str,
    Gf: Any,
) -> tuple[_Vector3, _Vector3]:
    """Choose one stage anchor representable by both endpoint float3 frames.

    USD physics stores local joint positions as ``float3``. Independently
    rounding the two inverse transforms can therefore produce different stage
    anchors even when both are individually close to the requested point. The
    two independently reprojected points are deterministic representable
    candidates. Re-quantize both endpoints around each candidate and select the
    closest contract-valid pair without changing either tolerance.
    """

    def reproject(
        xform: Any,
        local_position: _Vector3,
    ) -> _Vector3:
        return _vec3_tuple(xform.Transform(Gf.Vec3d(*local_position)))

    initial_anchor0 = reproject(body0_xform, local_pos0)
    initial_anchor1 = reproject(body1_xform, local_pos1)
    explicit_drift = max(
        _vector_distance(initial_anchor0, requested_anchor),
        _vector_distance(initial_anchor1, requested_anchor),
    )
    if _vector_distance(
        initial_anchor0, initial_anchor1
    ) <= _SHARED_ANCHOR_DISTANCE_TOLERANCE and (
        not explicit_anchor or explicit_drift <= _SHARED_ANCHOR_DISTANCE_TOLERANCE
    ):
        return local_pos0, local_pos1

    best_score: tuple[float, float, float, int] | None = None
    best_pair: tuple[_Vector3, _Vector3] | None = None
    inverse0 = body0_xform.GetInverse()
    inverse1 = body1_xform.GetInverse()
    for candidate_index, candidate_anchor in enumerate(
        (initial_anchor0, initial_anchor1)
    ):
        candidate = Gf.Vec3d(*candidate_anchor)
        candidate_local0 = _float32_vector(
            _vec3_tuple(inverse0.Transform(candidate)),
            label=f"{label} body0 reconciled local anchor",
        )
        candidate_local1 = _float32_vector(
            _vec3_tuple(inverse1.Transform(candidate)),
            label=f"{label} body1 reconciled local anchor",
        )
        candidate_anchor0 = reproject(body0_xform, candidate_local0)
        candidate_anchor1 = reproject(body1_xform, candidate_local1)
        if not (
            _anchor_vectors_close(candidate_anchor0, requested_anchor)
            and _anchor_vectors_close(candidate_anchor1, requested_anchor)
        ):
            continue
        mutual_distance = _vector_distance(candidate_anchor0, candidate_anchor1)
        if mutual_distance > _SHARED_ANCHOR_DISTANCE_TOLERANCE:
            continue
        drift0 = _vector_distance(candidate_anchor0, requested_anchor)
        drift1 = _vector_distance(candidate_anchor1, requested_anchor)
        if explicit_anchor and max(drift0, drift1) > (
            _SHARED_ANCHOR_DISTANCE_TOLERANCE
        ):
            continue
        score = (
            max(drift0, drift1),
            mutual_distance,
            drift0 + drift1,
            candidate_index,
        )
        if best_score is None or score < best_score:
            best_score = score
            best_pair = (candidate_local0, candidate_local1)

    return best_pair if best_pair is not None else (local_pos0, local_pos1)


def _optional_divide(value: float | None, divisor: float) -> float | None:
    if value is None:
        return None
    try:
        converted = float(value)
        result = converted / divisor
    except OverflowError:
        _fail(
            "authored_value_out_of_range",
            f"converted prismatic limit is not representable: {value!r} / {divisor!r}",
        )
    if not math.isfinite(result) or (converted != 0.0 and result == 0.0):
        _fail(
            "authored_value_out_of_range",
            f"converted prismatic limit is not representable: {value!r} / {divisor!r}",
        )
    return result


def _require_float32_value(value: float | None, *, label: str) -> None:
    if value is None:
        return
    _float32_round_trip(value, label=label)


def _float32_round_trip(value: float, *, label: str) -> float:
    converted = float(value)
    if not math.isfinite(converted) or abs(converted) > _FLOAT32_MAX:
        _fail(
            "authored_value_out_of_range",
            f"{label} must be finite and representable as a USD float; got {value!r}",
        )
    stored = float(struct.unpack(">f", struct.pack(">f", converted))[0])
    if not math.isfinite(stored) or (converted != 0.0 and stored == 0.0):
        _fail(
            "authored_value_out_of_range",
            f"{label} does not survive USD float32 storage; got {value!r}",
        )
    return stored


def _optional_float32_value(value: float | None, *, label: str) -> float | None:
    if value is None:
        return None
    return _float32_round_trip(value, label=label)


def _float32_vector(value: _Vector3, *, label: str) -> _Vector3:
    return cast(
        _Vector3,
        tuple(
            _float32_round_trip(component, label=f"{label}[{index}]")
            for index, component in enumerate(value)
        ),
    )


def _float32_quaternion(
    value: _Quaternion | None,
    *,
    label: str,
) -> _Quaternion | None:
    if value is None:
        return None
    real, imaginary = value
    return (
        _float32_round_trip(real, label=f"{label}.real"),
        _float32_vector(imaginary, label=f"{label}.imaginary"),
    )


def _pxr_modules() -> tuple[Any, Any, Any, Any, Any, Any]:
    try:
        from pxr import Gf, Sdf, Tf, Usd, UsdGeom, UsdPhysics
    except ImportError as exc:
        _fail(
            "runtime_dependency_unavailable",
            f"OpenUSD pxr bindings are required for topology authoring: {exc}",
        )
    return Gf, Sdf, Tf, Usd, UsdGeom, UsdPhysics


def _fail(code: str, detail: str) -> NoReturn:
    raise JointRiggerContractError(code, detail)


def _first_difference(label: str, left: tuple[Any, ...], right: tuple[Any, ...]) -> str:
    left_set = set(left)
    right_set = set(right)
    removed = sorted(left_set - right_set, key=repr)
    added = sorted(right_set - left_set, key=repr)
    return (
        f"{label} changed; removed={removed[:1]!r}, added={added[:1]!r}, "
        f"before_count={len(left)}, after_count={len(right)}"
    )


__all__ = [
    "JointRiggerPhysicsSchemaSnapshot",
    "JointRiggerStageSnapshot",
    "capture_joint_rigger_physics_schema_snapshot",
    "capture_joint_rigger_stage_snapshot",
    "physics_schema_counts",
    "validate_authored_joint_topology",
    "validate_joint_rigger_stage_preservation",
    "validate_joint_topology_plan",
]
