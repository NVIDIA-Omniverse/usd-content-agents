# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Evidence-backed USD physics schema authoring for Joint Rigger plans.

This module is intentionally narrower than a physics repair utility.  It
consumes an already-authorized Joint Rigger plan, verifies that the
stage contains exactly the represented joint graph, and applies only the
explicitly owned schema subset.  Missing or ambiguous evidence is rejected
before the first write.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, NoReturn

from world_understanding.functions.physics.joint_rigger.artifacts import (
    _route_cleanup_failures,
)
from world_understanding.functions.physics.joint_rigger.models import (
    DIAGNOSTICS_SCHEMA_VERSION,
    ArticulationRootPlanV1,
    FieldDecisionV1,
    JointDiagnosticV1,
    JointPlanV1,
    JointRiggerContractError,
    JointRiggerDiagnosticsV1,
    JointRiggerPlanV1,
    JointRiggerPlanV2,
)
from world_understanding.functions.physics.joint_rigger.validation import (
    _SHARED_ANCHOR_DISTANCE_TOLERANCE,
    _existing_joint_paths,
    _float32_round_trip,
    _paths_with_inactive_ancestors_enabled,
    capture_joint_rigger_stage_snapshot,
    validate_joint_rigger_stage_preservation,
)

_BACKEND_NAME = "world_understanding.physics_schemas"
_BACKEND_VERSION = "1"
_VALUE_TOLERANCE = 1e-6
# One validation may inspect clips for many owned prims.  This high fixed
# ceiling supports normal frame-sharded clip sets while bounding adversarial
# metadata cross-products before further layer opens.
_MAX_R3_VALUE_CLIP_INSPECTIONS = 4096
# Bound the aggregate ancestor walk even when no clip asset is ultimately
# opened.  This prevents deep proxy hierarchies from multiplying path work.
_MAX_R3_VALUE_CLIP_ANCESTOR_VISITS = 65_536
# Match the fixed composed-stage scan ceiling used by the topology validator.
# The three production UR10 inputs visit only 272-273 prims, but a compact
# instanced stage can expand far beyond its sealed source bytes.
_MAX_R3_INSTANCE_PROXY_PRIM_VISITS = 1_000_000
# Retained paths carry Python/Sdf objects and feed the clip-ancestor audit, so
# cap them independently of the cheap streaming stage traversal.
_MAX_R3_INSTANCE_PROXY_OWNED_PATHS = 16_384
# Articulation-root discovery performs a separate composed/prototype scan from
# value-clip coverage.  Bound both streamed work and retained matching paths.
_MAX_R3_ARTICULATION_ROOT_PRIM_VISITS = 1_000_000
_MAX_R3_ARTICULATION_ROOT_PATHS = 16_384
_NESTED_BODY_RESET_OP_SUFFIX = "jointRiggerPreserveWorld"
_NESTED_BODY_RESET_OP_NAME = f"xformOp:transform:{_NESTED_BODY_RESET_OP_SUFFIX}"
_AXES = {
    "x": (1.0, 0.0, 0.0),
    "y": (0.0, 1.0, 0.0),
    "z": (0.0, 0.0, 1.0),
}
_JOINT_TYPE_NAMES = {
    "revolute": "PhysicsRevoluteJoint",
    "prismatic": "PhysicsPrismaticJoint",
    "spherical": "PhysicsSphericalJoint",
}
_MOTIONS = {"revolute": "angular", "prismatic": "linear"}
_EVIDENCE_BACKED_SOURCES = frozenset(
    {
        "accepted_manifest",
        "authored_metadata",
        "authored_reference",
        "owner_approved_plan",
        "source_metadata",
    }
)

type JointRiggerPhysicsPlan = JointRiggerPlanV1 | JointRiggerPlanV2


@dataclass(frozen=True)
class _JointContext:
    plan: JointPlanV1
    prim: Any
    motion: str | None
    axis_token: str | None


@dataclass(frozen=True)
class _Preflight:
    graph_roots: tuple[str, ...]
    joints: dict[str, _JointContext]
    nested_body_world_matrices: dict[str, tuple[float, ...]]


@dataclass(frozen=True)
class _R3RawAuthorshipContract:
    """Canonical R3 opinions authored at one mapped edit-target prim."""

    schema_order: tuple[str, ...]
    attribute_specs: dict[str, tuple[str, str]]
    attribute_defaults: dict[str, Any]
    relationship_targets: dict[str, str]
    preserved_schema_order: tuple[str, ...] = ()

    @property
    def schema_tokens(self) -> frozenset[str]:
        return frozenset(self.schema_order) | frozenset(self.preserved_schema_order)

    @property
    def authored_properties(self) -> frozenset[str]:
        return frozenset(self.attribute_specs | self.relationship_targets)


def author_physics_schemas(
    stage: Any,
    plan: JointRiggerPhysicsPlan,
    *,
    backend_name: str = _BACKEND_NAME,
    backend_version: str | None = _BACKEND_VERSION,
) -> JointRiggerDiagnosticsV1:
    """Apply the plan's evidence-backed physics subset to ``stage``.

    The function mutates the supplied stage only after a complete preflight.
    Callers that need artifact-level atomicity should invoke it on a staged
    copy and publish that copy only after this function succeeds.

    The owned subset is rigid-body enablement, complete mass/inertia, exact
    supported collision-owner prims, exact articulation roots, default Joint
    State, and explicit motion-compatible drive or mimic opinions. A static
    nested rigid body without an existing transform-stack reset is represented
    by one reset matrix that preserves its default-time world transform. The
    author never changes hierarchy, geometry, composition arcs, joint
    topology, or world transforms, and never authors a ``PhysicsScene``,
    materials, contacts, grasp data, or inferred candidates.
    """

    if stage is None:
        _fail("invalid_stage", "stage must not be None")
    if not isinstance(plan, JointRiggerPlanV1 | JointRiggerPlanV2):
        _fail("invalid_plan", "plan must be a JointRiggerPlanV1 or V2 instance")
    if not backend_name.strip():
        _fail("invalid_backend_name", "backend_name must not be blank")
    if backend_version is not None and not backend_version.strip():
        _fail("invalid_backend_version", "backend_version must not be blank")

    validate_physics_plan_evidence(plan)
    try:
        before = capture_joint_rigger_stage_snapshot(stage)
        preflight = _preflight(stage, plan)
    except Exception as exc:
        if isinstance(exc, JointRiggerContractError):
            raise
        raise JointRiggerContractError(
            "physics_schema_preflight_failed",
            "OpenUSD could not inspect the stage before physics schema "
            f"authoring: {type(exc).__name__}: {exc}",
        ) from exc
    diagnostics = _diagnostics(
        plan,
        backend_name=backend_name,
        backend_version=backend_version,
    )
    edit_layer, edit_layer_backup = _backup_edit_layer(stage)
    try:
        _normalize_compatible_explicit_api_schemas(
            stage,
            _r3_raw_authorship_contract(stage, plan, preflight),
        )
        _apply(stage, plan, preflight)
        _validate_authored(stage, plan, preflight)
        after = capture_joint_rigger_stage_snapshot(stage)
        validate_joint_rigger_stage_preservation(before, after)
    except BaseException as exc:
        rollback_exc: BaseException | None = None
        try:
            _rollback_edit_layer(edit_layer, edit_layer_backup)
        except BaseException as caught_rollback_exc:
            rollback_exc = caught_rollback_exc
        if rollback_exc is not None:
            if not isinstance(exc, Exception):
                _route_cleanup_failures(
                    [("Physics schema rollback also failed", rollback_exc)],
                    primary_error=exc,
                    label="Physics schema rollback failed",
                )
                raise
            if not isinstance(rollback_exc, Exception):
                _route_cleanup_failures(
                    [("Physics schema authoring also failed", exc)],
                    primary_error=rollback_exc,
                    label="Physics schema authoring failed",
                )
                raise rollback_exc from exc
            failures = ExceptionGroup(
                "Physics schema authoring and rollback both failed",
                [exc, rollback_exc],
            )
            raise JointRiggerContractError(
                "physics_schema_rollback_failed",
                "could not restore the active edit layer after "
                f"{type(exc).__name__}: {exc}; rollback failed: "
                f"{type(rollback_exc).__name__}: {rollback_exc}",
            ) from failures
        if not isinstance(exc, Exception):
            raise
        if isinstance(exc, JointRiggerContractError):
            raise
        raise JointRiggerContractError(
            "physics_schema_authoring_failed",
            f"OpenUSD rejected an owned schema write: {exc}",
        ) from exc

    return diagnostics


def validate_physics_plan_evidence(plan: JointRiggerPhysicsPlan) -> None:
    """Validate the complete R3 evidence boundary without opening or writing USD.

    This pure check is intentionally usable from backend ``probe`` methods. It
    rejects source-data-limited plans before a staging artifact is allocated,
    while stage-specific graph and schema checks remain in :func:`_preflight`.
    """

    if not isinstance(plan, JointRiggerPlanV1 | JointRiggerPlanV2):
        _fail("invalid_plan", "plan must be a JointRiggerPlanV1 or V2 instance")
    if not plan.joints:
        _fail("physics_plan_incomplete", "at least one planned joint is required")

    endpoint_paths = {
        endpoint
        for joint in plan.joints
        for endpoint in (joint.topology.body0, joint.topology.body1)
    }
    body_paths = {body.prim_path for body in plan.rigid_bodies}
    if body_paths != endpoint_paths:
        _fail(
            "body_coverage_mismatch",
            "rigid_bodies must cover exactly the joint endpoint union; "
            f"missing={sorted(endpoint_paths - body_paths)}, "
            f"extra={sorted(body_paths - endpoint_paths)}",
        )

    graph_roots = _graph_roots(plan, body_paths)
    if isinstance(plan, JointRiggerPlanV1):
        articulation_root = plan.articulation_root
        if articulation_root is None:
            _fail(
                "articulation_root_missing",
                "the unique directed graph root requires explicit provenance",
            )
        assert articulation_root is not None
        if articulation_root.prim_path != graph_roots[0]:
            _fail(
                "articulation_root_mismatch",
                f"planned root {articulation_root.prim_path} is not graph root "
                f"{graph_roots[0]}",
            )
        _require_plan_provenance(
            articulation_root.provenance,
            label="articulation root",
        )
    else:
        planned_roots = tuple(root.prim_path for root in plan.articulation_roots)
        if not planned_roots:
            _fail(
                "articulation_roots_missing",
                "every directed graph component requires explicit root provenance",
            )
        if planned_roots != graph_roots:
            _fail(
                "articulation_roots_mismatch",
                "planned articulation roots do not exactly match graph component "
                f"roots: planned={list(planned_roots)}, "
                f"graph={list(graph_roots)}",
            )
        for articulation_root in plan.articulation_roots:
            _require_plan_provenance(
                articulation_root.provenance,
                label=f"articulation root {articulation_root.prim_path}",
            )

    for body in plan.rigid_bodies:
        _require_plan_provenance(body.provenance, label=f"rigid body {body.prim_path}")
        if body.mass is None:
            _fail(
                "mass_evidence_missing",
                f"complete mass and inertia evidence is required for {body.prim_path}",
            )
        assert body.mass is not None
        _require_plan_provenance(
            body.mass.provenance,
            label=f"mass and inertia at {body.prim_path}",
        )
        if not body.colliders:
            _fail(
                "collider_evidence_missing",
                "at least one exact supported collision-owner prim is required for "
                f"{body.prim_path}",
            )
        for collider in body.colliders:
            _require_plan_provenance(
                collider.provenance,
                label=f"collider {collider.prim_path}",
            )

    for joint in plan.joints:
        joint_id = joint.topology.joint_id
        for label, value in (
            ("limit", joint.limit),
            ("anchor", joint.anchor),
            ("joint friction", joint.joint_friction),
            ("state", joint.state),
            ("drive", joint.drive),
            ("mimic", joint.mimic),
        ):
            if value is not None:
                _require_plan_provenance(
                    value.provenance,
                    label=f"joint {joint_id} {label}",
                )
        if joint.topology.joint_type != "spherical" and joint.state is None:
            _fail(
                "joint_state_evidence_missing",
                f"explicit Joint State disposition is required for {joint_id}",
            )


def _require_plan_provenance(provenance: Any, *, label: str) -> None:
    if provenance.source not in _EVIDENCE_BACKED_SOURCES:
        _fail(
            "physics_evidence_not_source_backed",
            f"{label} uses unsupported provenance source {provenance.source!r}",
        )


def _backup_edit_layer(stage: Any) -> tuple[Any, Any]:
    """Return the active edit layer and a detached exact-content backup."""

    try:
        from pxr import Sdf

        edit_target = stage.GetEditTarget()
        layer = edit_target.GetLayer()
        if layer is None or not layer.permissionToEdit:
            _fail(
                "physics_schema_edit_layer_unavailable",
                "the active stage edit layer is missing or read-only",
            )
        backup = Sdf.Layer.CreateAnonymous("joint-rigger-physics-rollback.usda")
        backup.TransferContent(layer)
        return layer, backup
    except Exception as exc:
        if isinstance(exc, JointRiggerContractError):
            raise
        raise JointRiggerContractError(
            "physics_schema_edit_layer_unavailable",
            f"could not snapshot the active stage edit layer: {exc}",
        ) from exc


def _rollback_edit_layer(
    layer: Any,
    backup: Any,
) -> None:
    """Restore every opinion and preserve an exact failure for caller routing."""

    expected = backup.ExportToString()
    layer.TransferContent(backup)
    if layer.ExportToString() != expected:
        raise RuntimeError("restored edit-layer content differs from its backup")


def validate_authored_physics_schemas(
    stage: Any,
    plan: JointRiggerPhysicsPlan,
    *,
    backend_name: str = _BACKEND_NAME,
    backend_version: str | None = _BACKEND_VERSION,
) -> JointRiggerDiagnosticsV1:
    """Validate the exact owned schema subset without mutating ``stage``.

    This is the saved-artifact counterpart to :func:`author_physics_schemas`.
    It is useful to artifact backends that must close and reopen a staged layer
    before deriving the final identity and reports.
    """

    if stage is None:
        _fail("invalid_stage", "stage must not be None")
    if not isinstance(plan, JointRiggerPlanV1 | JointRiggerPlanV2):
        _fail("invalid_plan", "plan must be a JointRiggerPlanV1 or V2 instance")
    if not backend_name.strip():
        _fail("invalid_backend_name", "backend_name must not be blank")
    if backend_version is not None and not backend_version.strip():
        _fail("invalid_backend_version", "backend_version must not be blank")

    validate_physics_plan_evidence(plan)
    try:
        before = capture_joint_rigger_stage_snapshot(stage)
        preflight = _preflight(stage, plan)
        _validate_authored(stage, plan, preflight)
        after = capture_joint_rigger_stage_snapshot(stage)
        validate_joint_rigger_stage_preservation(before, after)
    except Exception as exc:
        if isinstance(exc, JointRiggerContractError):
            raise
        raise JointRiggerContractError(
            "physics_schema_validation_failed",
            "OpenUSD could not inspect the authored physics schemas: "
            f"{type(exc).__name__}: {exc}",
        ) from exc
    return _diagnostics(
        plan,
        backend_name=backend_name,
        backend_version=backend_version,
    )


_R3_OWNED_SCHEMA_FAMILIES = frozenset(
    {
        "PhysicsArticulationRootAPI",
        "PhysicsCollisionAPI",
        "PhysicsDriveAPI",
        "PhysicsJointStateAPI",
        "PhysicsMassAPI",
        "PhysicsMeshCollisionAPI",
        "PhysicsRigidBodyAPI",
        "PhysxJointAPI",
        "PhysxMimicJointAPI",
    }
)
_R3_OWNED_ATTRIBUTE_NAMES = frozenset(
    {
        "physics:approximation",
        "physics:centerOfMass",
        "physics:collisionEnabled",
        "physics:diagonalInertia",
        "physics:kinematicEnabled",
        "physics:mass",
        "physics:principalAxes",
        "physics:rigidBodyEnabled",
    }
)
_R3_OWNED_PROPERTY_PREFIXES = (
    "drive:",
    "physxJoint:",
    "physxMimicJoint:",
    "state:",
)
_R2_JOINT_VALUE_ATTRIBUTE_NAMES = frozenset(
    {
        "physics:axis",
        "physics:breakForce",
        "physics:breakTorque",
        "physics:collisionEnabled",
        "physics:excludeFromArticulation",
        "physics:jointEnabled",
        "physics:localPos0",
        "physics:localPos1",
        "physics:localRot0",
        "physics:localRot1",
        "physics:lowerLimit",
        "physics:upperLimit",
        "physics:coneAngle0Limit",
        "physics:coneAngle1Limit",
    }
)
_R2_BODY_VALUE_ATTRIBUTE_NAMES = frozenset(
    {
        "physics:angularVelocity",
        "physics:density",
        "physics:startsAsleep",
        "physics:velocity",
    }
)
_SOURCE_COLLIDER_MASS_ATTRIBUTES = ("physics:mass", "physics:density")
_UNSUPPORTED_SOURCE_COLLIDER_MASS_ATTRIBUTES = (
    "physics:centerOfMass",
    "physics:diagonalInertia",
    "physics:principalAxes",
)


def _is_r3_owned_schema_token(token: str) -> bool:
    return token.partition(":")[0] in _R3_OWNED_SCHEMA_FAMILIES


def _is_r3_owned_property_name(name: str) -> bool:
    return name in _R3_OWNED_ATTRIBUTE_NAMES or name.startswith(
        _R3_OWNED_PROPERTY_PREFIXES
    )


def _canonical_r3_attribute_default(
    value: Any,
    *,
    type_name: str,
    label: str,
) -> Any:
    """Return the exact value produced by the owned USD storage type."""

    if type_name == "bool":
        if not isinstance(value, bool):  # pragma: no cover - internal invariant
            _fail("physics_raw_contract_ambiguous", f"{label} is not a bool")
        return value
    if type_name == "token":
        return str(value)
    if type_name == "float":
        return _float32_round_trip(float(value), label=label)
    if type_name in {"float3", "point3f", "quatf"}:
        components = tuple(float(item) for item in value)
        expected_length = 4 if type_name == "quatf" else 3
        if len(components) != expected_length:  # pragma: no cover - model invariant
            _fail(
                "physics_raw_contract_ambiguous",
                f"{label} does not have {expected_length} components",
            )
        return tuple(
            _float32_round_trip(component, label=f"{label}[{index}]")
            for index, component in enumerate(components)
        )
    _fail(  # pragma: no cover - contract builder uses a closed type set
        "physics_raw_contract_ambiguous",
        f"{label} has unsupported USD type {type_name!r}",
    )


def _r3_raw_authorship_contract(
    stage: Any,
    plan: JointRiggerPhysicsPlan,
    preflight: _Preflight,
) -> dict[str, _R3RawAuthorshipContract]:
    """Derive raw R3 opinions in the exact order used by :func:`_apply`."""

    schema_order: dict[str, list[str]] = {}
    attribute_specs: dict[str, dict[str, tuple[str, str]]] = {}
    attribute_defaults: dict[str, dict[str, Any]] = {}
    relationship_targets: dict[str, dict[str, str]] = {}

    from pxr import UsdGeom, UsdPhysics

    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    kilograms_per_unit = float(UsdPhysics.GetStageKilogramsPerUnit(stage))

    def add_schema(path: str, token: str) -> None:
        tokens = schema_order.setdefault(path, [])
        if token in tokens:  # pragma: no cover - preflight/model invariant
            _fail(
                "physics_raw_contract_ambiguous",
                f"R3 schema {token!r} is assigned more than once at {path}",
            )
        tokens.append(token)

    def add_attribute(
        path: str,
        name: str,
        type_name: str,
        variability: str,
        default: Any,
    ) -> None:
        attributes = attribute_specs.setdefault(path, {})
        expected = (type_name, variability)
        previous = attributes.setdefault(name, expected)
        if previous != expected:  # pragma: no cover - preflight/model invariant
            _fail(
                "physics_raw_contract_ambiguous",
                f"R3 attribute {name!r} has conflicting raw shapes at {path}",
            )
        defaults = attribute_defaults.setdefault(path, {})
        canonical_default = _canonical_r3_attribute_default(
            default,
            type_name=type_name,
            label=f"{path}.{name}",
        )
        previous_default = defaults.setdefault(name, canonical_default)
        if not _stored_values_equal(previous_default, canonical_default):
            _fail(
                "physics_raw_contract_ambiguous",
                f"R3 attribute {name!r} has conflicting defaults at {path}",
            )

    def add_relationship(path: str, name: str, target: str) -> None:
        relationships = relationship_targets.setdefault(path, {})
        previous = relationships.setdefault(name, target)
        if previous != target:  # pragma: no cover - preflight/model invariant
            _fail(
                "physics_raw_contract_ambiguous",
                f"R3 relationship {name!r} has conflicting targets at {path}",
            )

    # Keep this traversal in lockstep with _apply: body and collider roles can
    # intentionally aggregate on one prim before the articulation-root role.
    for body in plan.rigid_bodies:
        path = body.prim_path
        add_schema(path, "PhysicsRigidBodyAPI")
        add_schema(path, "PhysicsMassAPI")
        add_attribute(path, "physics:rigidBodyEnabled", "bool", "varying", True)
        add_attribute(path, "physics:kinematicEnabled", "bool", "varying", False)
        assert body.mass is not None
        mass_value, center_of_mass_value, inertia_value = _mass_stage_values(
            body.mass,
            meters_per_unit=meters_per_unit,
            kilograms_per_unit=kilograms_per_unit,
        )
        add_attribute(path, "physics:mass", "float", "varying", mass_value)
        if center_of_mass_value is not None:
            add_attribute(
                path,
                "physics:centerOfMass",
                "point3f",
                "varying",
                center_of_mass_value,
            )
        add_attribute(
            path,
            "physics:diagonalInertia",
            "float3",
            "varying",
            inertia_value,
        )
        if body.mass.principal_axes is not None:
            add_attribute(
                path,
                "physics:principalAxes",
                "quatf",
                "varying",
                body.mass.principal_axes,
            )

        for collider in body.colliders:
            collider_path = collider.prim_path
            add_schema(collider_path, "PhysicsCollisionAPI")
            add_attribute(
                collider_path,
                "physics:collisionEnabled",
                "bool",
                "varying",
                True,
            )
            if collider.has_mesh_collision_api:
                add_schema(collider_path, "PhysicsMeshCollisionAPI")
            if collider.mesh_approximation is not None:
                add_attribute(
                    collider_path,
                    "physics:approximation",
                    "token",
                    "uniform",
                    collider.mesh_approximation,
                )

    for graph_root in preflight.graph_roots:
        add_schema(graph_root, "PhysicsArticulationRootAPI")

    for joint in plan.joints:
        path = joint.topology.joint_id
        context = preflight.joints[path]
        # A passive spherical joint intentionally owns no scalar state/control
        # schema.  Keep an explicit empty contract for it so cross-layer raw
        # scans still reject R3-family tokens and properties that the plan did
        # not authorize.
        schema_order.setdefault(path, [])
        if joint.state is not None:
            assert context.motion is not None
            add_schema(path, f"PhysicsJointStateAPI:{context.motion}")
            add_attribute(
                path,
                f"state:{context.motion}:physics:position",
                "float",
                "varying",
                joint.state.position,
            )
            add_attribute(
                path,
                f"state:{context.motion}:physics:velocity",
                "float",
                "varying",
                joint.state.velocity,
            )
        if joint.drive is not None:
            assert context.motion is not None
            add_schema(path, f"PhysicsDriveAPI:{context.motion}")
            for name, value, type_name in _drive_specs(context):
                add_attribute(
                    path,
                    name,
                    type_name,
                    "uniform" if type_name == "token" else "varying",
                    value,
                )
        if (
            joint.joint_friction is not None
            or joint.drive is not None
            and joint.drive.max_joint_velocity is not None
        ):
            add_schema(path, "PhysxJointAPI")
            if joint.drive is not None and joint.drive.max_joint_velocity is not None:
                add_attribute(
                    path,
                    "physxJoint:maxJointVelocity",
                    "float",
                    "varying",
                    joint.drive.max_joint_velocity,
                )
            if joint.joint_friction is not None:
                add_attribute(
                    path,
                    "physxJoint:jointFriction",
                    "float",
                    "varying",
                    joint.joint_friction.coefficient,
                )
        if joint.drive is None and joint.mimic is not None:
            assert context.axis_token is not None
            instance = f"rot{context.axis_token.upper()}"
            add_schema(path, f"PhysxMimicJointAPI:{instance}")
            namespace = f"physxMimicJoint:{instance}"
            reference = preflight.joints[joint.mimic.reference_joint_id]
            assert reference.axis_token is not None
            for name, value, type_name in _mimic_specs(
                context,
                reference_axis=f"rot{reference.axis_token.upper()}",
            ):
                add_attribute(path, name, type_name, "varying", value)
            add_relationship(
                path,
                f"{namespace}:referenceJoint",
                joint.mimic.reference_joint_id,
            )

    paths = (
        set(schema_order)
        | set(attribute_specs)
        | set(attribute_defaults)
        | set(relationship_targets)
    )
    contracts = {
        path: _R3RawAuthorshipContract(
            schema_order=tuple(schema_order.get(path, ())),
            preserved_schema_order=(),
            attribute_specs=dict(attribute_specs.get(path, {})),
            attribute_defaults=dict(attribute_defaults.get(path, {})),
            relationship_targets=dict(relationship_targets.get(path, {})),
        )
        for path in sorted(paths)
    }
    return _with_valid_source_physics(stage, plan, contracts)


def _with_valid_source_physics(
    stage: Any,
    plan: JointRiggerPhysicsPlan,
    contracts: dict[str, _R3RawAuthorshipContract],
) -> dict[str, _R3RawAuthorshipContract]:
    """Add the one explicit source-only physics form preserved by R3.

    A planned descendant collider may retain a source-authored ``MassAPI``
    carrying positive, static ``mass`` and/or ``density`` facts. The planned
    rigid body's complete mass/inertia remains authoritative. No other
    unplanned R3 schema or property is inferred as compatible.
    """

    collider_owners = {
        collider.prim_path: body.prim_path
        for body in plan.rigid_bodies
        for collider in body.colliders
        if collider.prim_path != body.prim_path
    }
    updated = dict(contracts)
    for scene_path, body_path in sorted(collider_owners.items()):
        contract = updated[scene_path]
        prim = stage.GetPrimAtPath(scene_path)
        composed_r3_tokens = tuple(
            token
            for token in _composed_raw_api_schema_items(prim)
            if _is_r3_owned_schema_token(token)
        )
        unexpected = tuple(
            token for token in composed_r3_tokens if token not in contract.schema_tokens
        )
        authored_mass_names = tuple(
            name
            for name in _SOURCE_COLLIDER_MASS_ATTRIBUTES
            if (attribute := prim.GetAttribute(name))
            and _has_authored_value(attribute, owner=scene_path)
        )
        if not unexpected and not authored_mass_names:
            continue
        if unexpected != ("PhysicsMassAPI",):
            # Preserve the established cross-layer raw validator and reason
            # code for every unrecognized schema/property shape. Density is
            # not in the R3-owned namespace, so reject it here when no exact
            # source MassAPI contract can account for it.
            if "physics:density" in authored_mass_names:
                _fail(
                    "physics_schema_conflict",
                    f"planned collider {scene_path} has source density without "
                    "one exact preservable PhysicsMassAPI",
                )
            continue
        if not scene_path.startswith(f"{body_path}/"):
            _fail(
                "physics_schema_conflict",
                f"source MassAPI collider {scene_path} is not a strict descendant "
                f"of planned body {body_path}",
            )
        source_specs, source_defaults = _source_collider_mass_contract(
            stage,
            prim,
            scene_path=scene_path,
        )
        updated[scene_path] = _R3RawAuthorshipContract(
            schema_order=contract.schema_order,
            preserved_schema_order=("PhysicsMassAPI",),
            attribute_specs={**contract.attribute_specs, **source_specs},
            attribute_defaults={**contract.attribute_defaults, **source_defaults},
            relationship_targets=contract.relationship_targets,
        )
    return updated


def _source_collider_mass_contract(
    stage: Any,
    prim: Any,
    *,
    scene_path: str,
) -> tuple[dict[str, tuple[str, str]], dict[str, Any]]:
    """Validate exact static source mass facts retained on one collider."""

    from pxr import Sdf

    edit_target = stage.GetEditTarget()
    layer = edit_target.GetLayer()
    spec_path = edit_target.MapToSpecPath(Sdf.Path(scene_path))
    prim_spec = None if layer is None else layer.GetPrimAtPath(spec_path)
    if not isinstance(prim_spec, Sdf.PrimSpec):
        _fail(
            "physics_schema_conflict",
            f"source MassAPI collider {scene_path} has no active edit-target PrimSpec",
        )
    schema_specs: list[Any] = []
    source_schema_list_op = None
    for spec in prim.GetPrimStack():
        if "apiSchemas" not in {str(key) for key in spec.ListInfoKeys()}:
            continue
        raw_schemas = spec.GetInfo("apiSchemas")
        if not isinstance(raw_schemas, Sdf.TokenListOp):
            _fail(
                "physics_schema_conflict",
                f"source MassAPI at {scene_path} has non-list-op apiSchemas",
            )
        tokens = {
            str(token)
            for bucket in (
                raw_schemas.explicitItems,
                raw_schemas.addedItems,
                raw_schemas.prependedItems,
                raw_schemas.appendedItems,
                raw_schemas.deletedItems,
                raw_schemas.orderedItems,
            )
            for token in bucket
        }
        if "PhysicsMassAPI" in tokens:
            schema_specs.append(spec)
            source_schema_list_op = raw_schemas
    if len(schema_specs) != 1 or (
        str(schema_specs[0].layer.identifier),
        str(schema_specs[0].path),
    ) != (str(layer.identifier), str(spec_path)):
        _fail(
            "physics_schema_conflict",
            f"source MassAPI at {scene_path} has ambiguous layer ownership",
        )
    if source_schema_list_op is None:  # pragma: no cover - guarded above
        _fail(
            "physics_schema_conflict",
            f"source MassAPI at {scene_path} has no inspectable apiSchemas opinion",
        )
    schema_buckets = tuple(
        tuple(str(token) for token in getattr(source_schema_list_op, bucket))
        for bucket in (
            "explicitItems",
            "addedItems",
            "prependedItems",
            "appendedItems",
            "deletedItems",
            "orderedItems",
        )
    )
    populated = tuple(bucket for bucket in schema_buckets if bucket)
    all_tokens = tuple(token for bucket in schema_buckets for token in bucket)
    if (
        len(populated) != 1
        or len(all_tokens) != len(set(all_tokens))
        or (
            not source_schema_list_op.isExplicit
            and not tuple(source_schema_list_op.prependedItems)
        )
    ):
        _fail(
            "physics_schema_conflict",
            f"source MassAPI at {scene_path} has ambiguous apiSchemas authorship",
        )

    for name in _UNSUPPORTED_SOURCE_COLLIDER_MASS_ATTRIBUTES:
        attribute = prim.GetAttribute(name)
        if attribute and _has_authored_value(attribute, owner=scene_path):
            _require_static(attribute, owner=scene_path)
            _fail(
                "physics_schema_conflict",
                f"source MassAPI at {scene_path} has unsupported {name}",
            )

    specs: dict[str, tuple[str, str]] = {}
    defaults: dict[str, Any] = {}
    for name in _SOURCE_COLLIDER_MASS_ATTRIBUTES:
        if prim.GetRelationship(name):
            _fail(
                "physics_schema_conflict",
                f"source MassAPI property {scene_path}.{name} is a Relationship",
            )
        attribute = prim.GetAttribute(name)
        if not attribute or not _has_authored_value(attribute, owner=scene_path):
            continue
        _require_static(attribute, owner=scene_path)
        property_stack = tuple(attribute.GetPropertyStack())
        source_property = prim_spec.properties.get(name)
        if (
            len(property_stack) != 1
            or property_stack[0] != source_property
            or not isinstance(source_property, Sdf.AttributeSpec)
            or source_property.custom
            or source_property.typeName != Sdf.ValueTypeNames.Float
            or source_property.variability != Sdf.VariabilityVarying
            or {str(key) for key in source_property.ListInfoKeys()}
            != {"custom", "default", "typeName", "variability"}
        ):
            _fail(
                "physics_schema_conflict",
                f"source MassAPI attribute {scene_path}.{name} has ambiguous raw "
                "authorship",
            )
        value = attribute.Get()
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            _fail(
                "physics_schema_conflict",
                f"source MassAPI attribute {scene_path}.{name} must be a positive "
                "finite float",
            )
        specs[name] = ("float", "varying")
        defaults[name] = _float32_round_trip(
            float(value),
            label=f"{scene_path}.{name}",
        )
    if not specs:
        _fail(
            "physics_schema_conflict",
            f"source MassAPI at {scene_path} has no explicit mass or density",
        )
    return specs, defaults


def _composed_raw_api_schema_items(prim: Any) -> tuple[str, ...]:
    """Return every composed API token, including optional-runtime schemas."""

    metadata = prim.GetMetadata("apiSchemas")
    if metadata is None:
        return ()
    try:
        return tuple(str(token) for token in metadata.GetAppliedItems())
    except AttributeError as exc:  # pragma: no cover - OpenUSD type invariant
        raise JointRiggerContractError(
            "physics_schema_list_op_ambiguous",
            f"{prim.GetPath()} has non-list-op composed apiSchemas metadata",
        ) from exc


def _normalize_compatible_explicit_api_schemas(
    stage: Any,
    contracts: dict[str, _R3RawAuthorshipContract],
) -> None:
    """Normalize only explicit schema opinions covered by the R3 contract.

    OpenUSD appends a newly applied schema to an existing explicit list-op.  A
    flattened Isaac stage can legitimately carry explicit foreign APIs,
    compatible plan-owned APIs, and the validated source-collider ``MassAPI``
    form. Convert one duplicate-free, single-contributor explicit opinion to
    an equivalent prepend before applying R3. Every R3 token and its relative
    order must already be covered by the contract, and the composed schema
    sequence must remain byte-for-byte equivalent across normalization.
    """

    from pxr import Sdf

    edit_target = stage.GetEditTarget()
    layer = edit_target.GetLayer()
    if layer is None:  # pragma: no cover - edit-layer backup already proved it
        _fail(
            "physics_schema_edit_layer_unavailable",
            "cannot normalize apiSchemas without an active edit layer",
        )
    for scene_path, contract in contracts.items():
        if not contract.schema_order:
            continue
        spec_path = edit_target.MapToSpecPath(Sdf.Path(scene_path))
        if spec_path == Sdf.Path.emptyPath:
            _fail(
                "physics_schema_list_op_ambiguous",
                f"R3 prim {scene_path} cannot be mapped into the active edit target",
            )
        prim_spec = layer.GetPrimAtPath(spec_path)
        if not isinstance(prim_spec, Sdf.PrimSpec):
            continue
        if "apiSchemas" not in {str(key) for key in prim_spec.ListInfoKeys()}:
            continue
        raw_schemas = prim_spec.GetInfo("apiSchemas")
        if not isinstance(raw_schemas, Sdf.TokenListOp):
            continue
        buckets = tuple(
            tuple(str(token) for token in getattr(raw_schemas, bucket))
            for bucket in (
                "explicitItems",
                "addedItems",
                "prependedItems",
                "appendedItems",
                "deletedItems",
                "orderedItems",
            )
        )
        populated = tuple(bucket for bucket in buckets if bucket)
        all_items = tuple(token for bucket in buckets for token in bucket)
        if len(populated) > 1 or len(all_items) != len(set(all_items)):
            _fail(
                "physics_schema_list_op_ambiguous",
                f"R3 prim {scene_path} has an ambiguous apiSchemas opinion",
            )
        if not raw_schemas.isExplicit:
            continue
        prim = stage.GetPrimAtPath(scene_path)
        contributing = tuple(
            spec
            for spec in prim.GetPrimStack()
            if "apiSchemas" in {str(key) for key in spec.ListInfoKeys()}
        )
        if len(contributing) != 1 or (
            str(contributing[0].layer.identifier),
            str(contributing[0].path),
        ) != (str(layer.identifier), str(spec_path)):
            # Rewriting an explicit opinion could unmask another layer.  Leave
            # the raw form untouched so the strict postwrite validator rejects
            # it and the transaction restores the original layer exactly.
            continue
        explicit = tuple(str(token) for token in raw_schemas.explicitItems)
        if not explicit:
            continue
        explicit_owned = tuple(
            token for token in explicit if _is_r3_owned_schema_token(token)
        )
        unexpected_owned = tuple(
            token for token in explicit_owned if token not in contract.schema_tokens
        )
        if unexpected_owned:
            _fail(
                "physics_schema_list_op_ambiguous",
                f"R3 prim {scene_path} explicit apiSchemas contains incompatible "
                f"physics tokens: {list(unexpected_owned)}",
            )
        for expected_order in (
            contract.schema_order,
            contract.preserved_schema_order,
        ):
            observed_order = tuple(
                token for token in explicit_owned if token in set(expected_order)
            )
            existing_expected_order = tuple(
                token for token in expected_order if token in set(observed_order)
            )
            if observed_order != existing_expected_order:
                _fail(
                    "physics_schema_list_op_ambiguous",
                    f"R3 prim {scene_path} explicit apiSchemas has incompatible "
                    "physics token order",
                )
        before = _composed_raw_api_schema_items(prim)
        registered_before = tuple(str(token) for token in prim.GetAppliedSchemas())
        prim_spec.SetInfo(
            "apiSchemas",
            Sdf.TokenListOp.Create(prependedItems=list(explicit)),
        )
        after = _composed_raw_api_schema_items(prim)
        registered_after = tuple(str(token) for token in prim.GetAppliedSchemas())
        if after != before or registered_after != registered_before:
            _fail(
                "physics_schema_list_op_ambiguous",
                f"normalizing foreign apiSchemas at {scene_path} changed composed "
                f"tokens: before={before}, after={after}, "
                f"registered_before={registered_before}, "
                f"registered_after={registered_after}",
            )


def _validate_contributing_r3_attribute_specs(
    prim: Any,
    scene_path: str,
    contract: _R3RawAuthorshipContract,
    *,
    Sdf: Any,
    type_by_name: dict[str, Any],
    variability_by_name: dict[str, Any],
) -> None:
    """Validate every authored layer opinion for plan-owned attributes."""

    for name, (type_name, variability_name) in contract.attribute_specs.items():
        expected_default = contract.attribute_defaults[name]
        for prim_spec in prim.GetPrimStack():
            properties = {str(prop.name): prop for prop in prim_spec.properties}
            property_spec = properties.get(name)
            if property_spec is None:
                continue
            if not isinstance(property_spec, Sdf.AttributeSpec):
                _fail(
                    "authored_graph_mismatch",
                    f"R3 prim {scene_path} contributing property {name!r} is not "
                    "an AttributeSpec",
                )
            if (
                property_spec.custom
                or property_spec.typeName != type_by_name[type_name]
                or property_spec.variability != variability_by_name[variability_name]
            ):
                _fail(
                    "authored_graph_mismatch",
                    f"R3 prim {scene_path} contributing raw attribute {name!r} "
                    "has noncanonical type, variability, or customness",
                )
            info_keys = {str(key) for key in property_spec.ListInfoKeys()}
            time_samples = tuple(
                float(value)
                for value in property_spec.layer.ListTimeSamplesForPath(
                    property_spec.path
                )
            )
            if time_samples:
                _fail(
                    "authored_graph_mismatch",
                    f"R3 prim {scene_path} contributing raw attribute {name!r} "
                    f"has time samples: {time_samples}",
                )
            if "spline" in info_keys:
                _fail(
                    "authored_graph_mismatch",
                    f"R3 prim {scene_path} contributing raw attribute {name!r} "
                    "has an authored spline",
                )
            if "connectionPaths" in info_keys:
                _fail(
                    "authored_graph_mismatch",
                    f"R3 prim {scene_path} contributing raw attribute {name!r} "
                    "has an authored connection list-op",
                )
            if "default" in info_keys and not _stored_values_equal(
                property_spec.GetInfo("default"),
                expected_default,
            ):
                _fail(
                    "authored_graph_mismatch",
                    f"R3 prim {scene_path} contributing raw attribute {name!r} "
                    "has a noncanonical default",
                )


def _clip_asset_carries_owned_samples(
    asset_path: Any,
    *,
    source_layer: Any,
    clip_prim_path: Any,
    relative_prim_path: Any,
    attribute_names: frozenset[str],
    include_r3_owned_property_names: bool,
    include_xform_property_names: bool,
    resolver_context: Any,
    Sdf: Any,
) -> tuple[bool, str]:
    """Inspect one concrete value-clip layer for owned temporal values."""

    # ``resolvedPath`` is cached data carried by Sdf.AssetPath and can be
    # stale or forged independently of the authored locator.  Resolve only
    # the authored path against the layer that owns this raw clips opinion so
    # in-memory validation matches save/reopen behavior.
    authored_path = str(getattr(asset_path, "authoredPath", ""))
    try:
        from pxr import Ar

        with Ar.ResolverContextBinder(resolver_context):
            resolved_path = Sdf.ComputeAssetPathRelativeToLayer(
                source_layer,
                authored_path,
            )
            clip_layer = Sdf.Layer.FindOrOpen(resolved_path)
        if clip_layer is None:
            _fail(
                "authored_graph_mismatch",
                "R3 value-clip asset cannot be inspected: "
                f"{authored_path!r} from {source_layer.identifier!r}",
            )

        target_prim_path = clip_prim_path.AppendPath(relative_prim_path)
        inspected_names = set(attribute_names)
        if include_r3_owned_property_names:
            target_prim_spec = clip_layer.GetPrimAtPath(target_prim_path)
            if target_prim_spec is not None:
                inspected_names.update(
                    str(prop.name)
                    for prop in target_prim_spec.properties
                    if isinstance(prop, Sdf.AttributeSpec)
                    and _is_r3_owned_property_name(str(prop.name))
                )
        if include_xform_property_names:
            target_prim_spec = clip_layer.GetPrimAtPath(target_prim_path)
            if target_prim_spec is not None:
                inspected_names.update(
                    str(prop.name)
                    for prop in target_prim_spec.properties
                    if isinstance(prop, Sdf.AttributeSpec)
                    and (
                        str(prop.name) == "xformOpOrder"
                        or str(prop.name).startswith("xformOp:")
                    )
                )
        for name in sorted(inspected_names):
            property_path = target_prim_path.AppendProperty(name)
            property_spec = clip_layer.GetPropertyAtPath(property_path)
            if not isinstance(property_spec, Sdf.AttributeSpec):
                continue
            time_samples = tuple(clip_layer.ListTimeSamplesForPath(property_path))
            info_keys = {str(key) for key in property_spec.ListInfoKeys()}
            if time_samples or "spline" in info_keys:
                return True, f"{clip_layer.identifier}:{property_path}"
        return False, clip_layer.identifier
    except Exception as exc:
        if isinstance(exc, JointRiggerContractError):
            raise
        raise JointRiggerContractError(
            "authored_graph_mismatch",
            "R3 value-clip asset inspection failed for "
            f"{authored_path!r} from {source_layer.identifier!r}: "
            f"{type(exc).__name__}: {exc}",
        ) from exc


def _validate_contributing_r3_value_clips(
    stage: Any,
    scene_path: str,
    attribute_names: frozenset[str],
    *,
    Sdf: Any,
    inspection_keys: set[tuple[str, str, str, str, tuple[str, ...]]],
    include_r3_owned_property_names: bool,
    include_xform_property_names: bool,
    resolver_context: Any,
    ancestor_visits: int = 0,
) -> int:
    """Reject ancestor clips that can supply plan-owned attribute values.

    A stronger canonical default masks value-clip samples from normal composed
    attribute queries.  Therefore every raw ``clips`` opinion in each
    contributing ancestor PrimSpec is inspected, including opinions hidden by
    a stronger clips dictionary.  Concrete clip assets are rejected only when
    they carry temporal values at an owned property path; sibling subtrees and
    assets containing unrelated properties remain valid.

    Template and manifest-only clip forms cannot be enumerated from one raw
    metadata opinion without evaluating a potentially external asset pattern.
    They fail closed only when their clip anchor covers this owned prim.
    """

    if (
        not attribute_names
        and not include_r3_owned_property_names
        and not include_xform_property_names
    ):
        return ancestor_visits
    target_path = Sdf.Path(scene_path)
    ancestor_path = target_path
    while ancestor_path != Sdf.Path.absoluteRootPath:
        ancestor_visits += 1
        if ancestor_visits > _MAX_R3_VALUE_CLIP_ANCESTOR_VISITS:
            _fail(
                "r3_value_clip_ancestor_scan_limit_exceeded",
                "R3 value-clip audit exceeds the fixed "
                f"{_MAX_R3_VALUE_CLIP_ANCESTOR_VISITS}-ancestor visit limit",
            )
        ancestor_prim = stage.GetPrimAtPath(ancestor_path)
        if ancestor_prim:
            relative_path = target_path.MakeRelativePath(ancestor_path)
            raw_opinions: list[tuple[Any, str, dict[str, Any]]] = []
            for prim_spec in ancestor_prim.GetPrimStack():
                info_keys = {str(key) for key in prim_spec.ListInfoKeys()}
                if "clips" not in info_keys:
                    continue
                raw_clips = prim_spec.GetInfo("clips")
                if not isinstance(raw_clips, dict):
                    _fail(
                        "authored_graph_mismatch",
                        f"R3 ancestor {ancestor_path} has malformed raw clips metadata",
                    )
                for raw_clip_set, settings in raw_clips.items():
                    clip_set = str(raw_clip_set)
                    if not isinstance(settings, dict):
                        _fail(
                            "authored_graph_mismatch",
                            f"R3 ancestor {ancestor_path} clip set "
                            f"{clip_set!r} is malformed",
                        )
                    raw_opinions.append((prim_spec, clip_set, settings))

            composed_clips = ancestor_prim.GetMetadata("clips") or {}
            prim_paths_by_set: dict[str, set[str]] = {}
            composed_settings_by_set: dict[str, dict[str, Any]] = {}
            winning_asset_settings_by_set: dict[str, dict[str, Any]] = {}
            for _, clip_set, settings in raw_opinions:
                if (
                    "assetPaths" in settings
                    and clip_set not in winning_asset_settings_by_set
                ):
                    # GetPrimStack is strongest-to-weakest.  Nested clips
                    # dictionary fields compose independently, so the first
                    # opinion that authors assetPaths is the exact winner for
                    # that field.
                    winning_asset_settings_by_set[clip_set] = settings
                if "primPath" in settings:
                    prim_paths_by_set.setdefault(clip_set, set()).add(
                        str(settings["primPath"])
                    )
            if isinstance(composed_clips, dict):
                for raw_clip_set, settings in composed_clips.items():
                    if isinstance(settings, dict):
                        clip_set = str(raw_clip_set)
                        composed_settings_by_set[clip_set] = settings
                        if "primPath" in settings:
                            prim_paths_by_set.setdefault(clip_set, set()).add(
                                str(settings["primPath"])
                            )

            for prim_spec, clip_set, settings in raw_opinions:
                has_manifest_only = (
                    "manifestAssetPath" in settings and "assetPaths" not in settings
                )
                if "templateAssetPath" in settings or has_manifest_only:
                    _fail(
                        "authored_graph_mismatch",
                        f"R3 ancestor {ancestor_path} clip set "
                        f"{clip_set!r} has an uninspectable template or "
                        f"manifest-only source covering owned prim {scene_path}",
                    )
                if "assetPaths" not in settings:
                    continue
                raw_asset_paths = settings["assetPaths"]
                if not isinstance(raw_asset_paths, Sdf.AssetPathArray):
                    _fail(
                        "authored_graph_mismatch",
                        f"R3 ancestor {ancestor_path} clip set {clip_set!r} "
                        "has malformed assetPaths metadata; expected an "
                        "AssetPathArray",
                    )
                asset_paths = tuple(raw_asset_paths)
                if not asset_paths:
                    continue
                raw_prim_paths = (
                    {str(settings["primPath"])}
                    if "primPath" in settings
                    else prim_paths_by_set.get(clip_set, set())
                )
                composed_settings = composed_settings_by_set.get(clip_set, {})
                if (
                    "primPath" in settings
                    and winning_asset_settings_by_set.get(clip_set) is settings
                    and "primPath" in composed_settings
                ):
                    # This raw assetPaths opinion is the composed winner.  Its
                    # effective primPath can come from a stronger partial
                    # dictionary, so inspect that active association in
                    # addition to the opinion's own latent complete pair.
                    raw_prim_paths.add(str(composed_settings["primPath"]))
                if not raw_prim_paths:
                    _fail(
                        "authored_graph_mismatch",
                        f"R3 ancestor {ancestor_path} clip set {clip_set!r} "
                        "has clip assets but no inspectable primPath for owned "
                        f"prim {scene_path}",
                    )
                clip_prim_paths = []
                for raw_prim_path in sorted(raw_prim_paths):
                    clip_prim_path = Sdf.Path(raw_prim_path)
                    if (
                        not clip_prim_path.IsAbsolutePath()
                        or not clip_prim_path.IsPrimPath()
                    ):
                        _fail(
                            "authored_graph_mismatch",
                            f"R3 ancestor {ancestor_path} clip set "
                            f"{clip_set!r} has an invalid primPath",
                        )
                    clip_prim_paths.append(clip_prim_path)

                # Every authored clip asset is a latent temporal source.  Do
                # not let a missing/inactive ``active`` entry hide an owned
                # sample that a later list edit can activate.
                for asset_path in asset_paths:
                    for clip_prim_path in clip_prim_paths:
                        inspection_key = (
                            str(prim_spec.layer.identifier),
                            str(getattr(asset_path, "path", "") or asset_path),
                            str(clip_prim_path),
                            str(relative_path),
                            tuple(sorted(attribute_names)),
                        )
                        if inspection_key in inspection_keys:
                            continue
                        if len(inspection_keys) >= _MAX_R3_VALUE_CLIP_INSPECTIONS:
                            _fail(
                                "authored_graph_mismatch",
                                "R3 value-clip audit exceeds the bounded "
                                f"inspection limit of "
                                f"{_MAX_R3_VALUE_CLIP_INSPECTIONS}",
                            )
                        inspection_keys.add(inspection_key)
                        carries_owned, source = _clip_asset_carries_owned_samples(
                            asset_path,
                            source_layer=prim_spec.layer,
                            clip_prim_path=clip_prim_path,
                            relative_prim_path=relative_path,
                            attribute_names=attribute_names,
                            include_r3_owned_property_names=(
                                include_r3_owned_property_names
                            ),
                            include_xform_property_names=(include_xform_property_names),
                            resolver_context=resolver_context,
                            Sdf=Sdf,
                        )
                        if carries_owned:
                            _fail(
                                "authored_graph_mismatch",
                                f"R3 prim {scene_path} has plan-owned temporal "
                                f"values supplied by ancestor {ancestor_path} "
                                f"clip set {clip_set!r}: {source}",
                            )
        ancestor_path = ancestor_path.GetParentPath()
    return ancestor_visits


def _validate_plan_owned_value_clips(
    stage: Any,
    attribute_names_by_path: dict[str, frozenset[str]],
    *,
    include_r3_owned_property_names_at: frozenset[str] = frozenset(),
    include_xform_property_names_at: frozenset[str] = frozenset(),
) -> None:
    """Audit concrete value-clip sources for every plan-owned attribute path."""

    from pxr import Sdf

    inspection_keys: set[tuple[str, str, str, str, tuple[str, ...]]] = set()
    scene_paths = (
        set(attribute_names_by_path)
        | set(include_r3_owned_property_names_at)
        | set(include_xform_property_names_at)
    )
    ancestor_visits = 0
    for scene_path in sorted(scene_paths):
        attribute_names = attribute_names_by_path.get(scene_path, frozenset())
        ancestor_visits = _validate_contributing_r3_value_clips(
            stage,
            scene_path,
            attribute_names,
            Sdf=Sdf,
            inspection_keys=inspection_keys,
            include_r3_owned_property_names=(
                scene_path in include_r3_owned_property_names_at
            ),
            include_xform_property_names=(
                scene_path in include_xform_property_names_at
            ),
            resolver_context=stage.GetPathResolverContext(),
            ancestor_visits=ancestor_visits,
        )


def _complete_plan_owned_value_clip_attributes(
    stage: Any,
    plan: JointRiggerPhysicsPlan,
    contracts: dict[str, _R3RawAuthorshipContract],
) -> tuple[dict[str, frozenset[str]], frozenset[str]]:
    """Return the complete R2 topology/transform plus R3 attribute boundary."""

    from pxr import Sdf, Usd, UsdGeom

    from world_understanding.functions.physics.joint_rigger.reference import (
        _collider_geometry_attributes,
    )

    attribute_names_by_path = {
        path: set(contract.attribute_specs) for path, contract in contracts.items()
    }
    transform_paths: set[str] = set()
    instance_collider_paths: set[str] = set()
    for body in plan.rigid_bodies:
        attribute_names_by_path.setdefault(body.prim_path, set()).update(
            _R2_BODY_VALUE_ATTRIBUTE_NAMES
        )
        for collider in body.colliders:
            collider_prim = stage.GetPrimAtPath(collider.prim_path)
            collider_names = attribute_names_by_path.setdefault(
                collider.prim_path,
                set(),
            )
            collider_names.update({"physics:collisionEnabled", "physics:approximation"})
            if collider_prim.IsA(UsdGeom.Gprim):
                collider_names.update(_collider_geometry_attributes(collider_prim))

            current = collider_prim
            while current.IsValid() and not current.IsPseudoRoot():
                transform_paths.add(str(current.GetPath()))
                if str(current.GetPath()) == body.prim_path:
                    break
                current = current.GetParent()

            if collider_prim.IsInstance():
                instance_collider_paths.add(collider.prim_path)

    if instance_collider_paths:
        instance_roots = {Sdf.Path(path) for path in instance_collider_paths}
        covered_instance_paths = set(instance_roots)
        prim_visits = 0
        covered_proxy_paths = 0
        for proxy_prim in _stage_prims_with_instance_proxies(stage, Usd=Usd):
            prim_visits += 1
            if prim_visits > _MAX_R3_INSTANCE_PROXY_PRIM_VISITS:
                _fail(
                    "r3_instance_proxy_scan_limit_exceeded",
                    "R3 instance-proxy audit exceeds the fixed "
                    f"{_MAX_R3_INSTANCE_PROXY_PRIM_VISITS}-prim visit limit",
                )
            if not proxy_prim.IsInstanceProxy():
                continue
            proxy_path = proxy_prim.GetPath()
            # PrimRange.Stage yields preorder traversal.  Propagating coverage
            # from each parent therefore bounds membership work to O(1) per
            # visited proxy instead of repeatedly walking deep ancestor chains.
            if proxy_path.GetParentPath() not in covered_instance_paths:
                continue
            covered_proxy_paths += 1
            if covered_proxy_paths > _MAX_R3_INSTANCE_PROXY_OWNED_PATHS:
                _fail(
                    "r3_instance_proxy_owned_path_limit_exceeded",
                    "R3 instance-proxy audit exceeds the fixed "
                    f"{_MAX_R3_INSTANCE_PROXY_OWNED_PATHS}-covered-path limit",
                )
            covered_instance_paths.add(proxy_path)
            proxy_path_text = str(proxy_path)
            if UsdGeom.Xformable(proxy_prim):
                transform_paths.add(proxy_path_text)
            if proxy_prim.IsA(UsdGeom.Gprim):
                attribute_names_by_path.setdefault(proxy_path_text, set()).update(
                    _collider_geometry_attributes(proxy_prim)
                )
    for joint in plan.joints:
        topology = joint.topology
        attribute_names_by_path.setdefault(topology.joint_id, set()).update(
            _R2_JOINT_VALUE_ATTRIBUTE_NAMES
        )
        for body_path in (topology.body0, topology.body1):
            current = stage.GetPrimAtPath(body_path)
            while current.IsValid() and not current.IsPseudoRoot():
                transform_paths.add(str(current.GetPath()))
                xformable = UsdGeom.Xformable(current)
                if xformable:
                    owned_names = attribute_names_by_path.setdefault(
                        str(current.GetPath()),
                        set(),
                    )
                    owned_names.update(
                        str(op.GetAttr().GetName())
                        for op in xformable.GetOrderedXformOps()
                    )
                    xform_order = xformable.GetXformOpOrderAttr()
                    if xform_order and xform_order.HasAuthoredValueOpinion():
                        owned_names.add(str(xform_order.GetName()))
                current = current.GetParent()
    return (
        {
            path: frozenset(attribute_names)
            for path, attribute_names in attribute_names_by_path.items()
            if attribute_names
        },
        frozenset(transform_paths),
    )


def _stage_prims_with_instance_proxies(stage: Any, *, Usd: Any) -> Any:
    """Return one proxy-expanded stage range for the bounded R3 audit."""

    return Usd.PrimRange.Stage(
        stage,
        Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate),
    )


def _validate_contributing_r3_property_names(
    prim: Any,
    scene_path: str,
    contract: _R3RawAuthorshipContract,
) -> None:
    """Reject plan-owned property names outside the contract on every layer."""

    for prim_spec in prim.GetPrimStack():
        unexpected = sorted(
            name
            for name in (str(prop.name) for prop in prim_spec.properties)
            if _is_r3_owned_property_name(name)
            and name not in contract.authored_properties
        )
        if unexpected:
            _fail(
                "authored_graph_mismatch",
                f"R3 prim {scene_path} contributing raw properties include "
                f"unexpected plan-owned names: {unexpected}",
            )


def _validate_contributing_r3_relationship_specs(
    prim: Any,
    scene_path: str,
    contract: _R3RawAuthorshipContract,
    *,
    Sdf: Any,
) -> None:
    """Validate relationship shape without constraining source-layer list-ops."""

    for name in contract.relationship_targets:
        for prim_spec in prim.GetPrimStack():
            properties = {str(prop.name): prop for prop in prim_spec.properties}
            property_spec = properties.get(name)
            if property_spec is None:
                continue
            if not isinstance(property_spec, Sdf.RelationshipSpec):
                _fail(
                    "authored_graph_mismatch",
                    f"R3 prim {scene_path} contributing property {name!r} is not "
                    "a RelationshipSpec",
                )
            if (
                property_spec.custom
                or property_spec.variability != Sdf.VariabilityUniform
            ):
                _fail(
                    "authored_graph_mismatch",
                    f"R3 prim {scene_path} contributing raw relationship {name!r} "
                    "has noncanonical variability or customness",
                )


def _validate_contributing_r3_schema_tokens(
    prim: Any,
    scene_path: str,
    contract: _R3RawAuthorshipContract,
    *,
    Sdf: Any,
) -> None:
    """Reject unexpected R3-family tokens in every contributing list-op."""

    for prim_spec in prim.GetPrimStack():
        info_keys = {str(key) for key in prim_spec.ListInfoKeys()}
        if "apiSchemas" not in info_keys:
            continue
        raw_schemas = prim_spec.GetInfo("apiSchemas")
        if not isinstance(raw_schemas, Sdf.TokenListOp):
            _fail(
                "authored_graph_mismatch",
                f"R3 prim {scene_path} has a noncanonical contributing raw "
                "apiSchemas opinion",
            )
        tokens = {
            str(token)
            for bucket in (
                raw_schemas.explicitItems,
                raw_schemas.addedItems,
                raw_schemas.prependedItems,
                raw_schemas.appendedItems,
                raw_schemas.deletedItems,
                raw_schemas.orderedItems,
            )
            for token in bucket
        }
        unexpected = {
            token
            for token in tokens
            if _is_r3_owned_schema_token(token) and token not in contract.schema_tokens
        }
        if unexpected:
            _fail(
                "authored_graph_mismatch",
                f"R3 prim {scene_path} contributing raw apiSchemas contains "
                f"unexpected same-family tokens: {sorted(unexpected)}",
            )


def _validate_owned_physics_raw_authorship(
    stage: Any,
    plan: JointRiggerPhysicsPlan,
    preflight: _Preflight,
) -> None:
    """Require canonical raw R3 opinions in the mapped active edit target."""

    from pxr import Sdf

    contracts = _r3_raw_authorship_contract(stage, plan, preflight)
    value_clip_attributes, transform_paths = _complete_plan_owned_value_clip_attributes(
        stage,
        plan,
        contracts,
    )
    _validate_plan_owned_value_clips(
        stage,
        value_clip_attributes,
        include_r3_owned_property_names_at=frozenset(contracts),
        include_xform_property_names_at=transform_paths,
    )
    edit_target = stage.GetEditTarget()
    layer = edit_target.GetLayer()
    if layer is None:  # pragma: no cover - preflight already checks the layer
        _fail(
            "authored_graph_mismatch",
            "R3 raw authorship has no active edit-target layer",
        )
    type_by_name = {
        "bool": Sdf.ValueTypeNames.Bool,
        "float": Sdf.ValueTypeNames.Float,
        "float3": Sdf.ValueTypeNames.Float3,
        "point3f": Sdf.ValueTypeNames.Point3f,
        "quatf": Sdf.ValueTypeNames.Quatf,
        "token": Sdf.ValueTypeNames.Token,
    }
    variability_by_name = {
        "uniform": Sdf.VariabilityUniform,
        "varying": Sdf.VariabilityVarying,
    }
    for scene_path, contract in contracts.items():
        prim = stage.GetPrimAtPath(scene_path)
        _validate_contributing_r3_schema_tokens(
            prim,
            scene_path,
            contract,
            Sdf=Sdf,
        )
        composed_owned_schemas = frozenset(
            token
            for token in _applied_schema_tokens(prim)
            if _is_r3_owned_schema_token(token)
        )
        if composed_owned_schemas != contract.schema_tokens:
            _fail(
                "authored_graph_mismatch",
                f"R3 prim {scene_path} composed raw schema families do not exactly "
                f"match the plan: unexpected="
                f"{sorted(composed_owned_schemas - contract.schema_tokens)}, "
                f"missing={sorted(contract.schema_tokens - composed_owned_schemas)}",
            )
        _validate_contributing_r3_property_names(
            prim,
            scene_path,
            contract,
        )
        _validate_contributing_r3_attribute_specs(
            prim,
            scene_path,
            contract,
            Sdf=Sdf,
            type_by_name=type_by_name,
            variability_by_name=variability_by_name,
        )
        _validate_contributing_r3_relationship_specs(
            prim,
            scene_path,
            contract,
            Sdf=Sdf,
        )
        if not contract.schema_order and not contract.authored_properties:
            # Empty contracts (currently passive spherical joints) require no
            # edit-target footprint.  The composed and every-layer scans above
            # still prove the absence of all R3-owned schemas/properties.
            continue
        spec_path = edit_target.MapToSpecPath(Sdf.Path(scene_path))
        if spec_path == Sdf.Path.emptyPath:
            _fail(
                "authored_graph_mismatch",
                f"R3 prim {scene_path} cannot be mapped into the active edit target",
            )
        prim_spec = layer.GetPrimAtPath(spec_path)
        if not isinstance(prim_spec, Sdf.PrimSpec):
            _fail(
                "authored_graph_mismatch",
                f"R3 prim {scene_path} has no authored edit-target PrimSpec",
            )

        raw_schemas = (
            prim_spec.GetInfo("apiSchemas")
            if "apiSchemas" in {str(key) for key in prim_spec.ListInfoKeys()}
            else None
        )
        if raw_schemas is None or not isinstance(raw_schemas, Sdf.TokenListOp):
            _fail(
                "authored_graph_mismatch",
                f"R3 prim {scene_path} lacks a canonical raw apiSchemas list-op",
            )
        assert raw_schemas is not None
        buckets = {
            "explicit": tuple(str(item) for item in raw_schemas.explicitItems),
            "added": tuple(str(item) for item in raw_schemas.addedItems),
            "prepended": tuple(str(item) for item in raw_schemas.prependedItems),
            "appended": tuple(str(item) for item in raw_schemas.appendedItems),
            "deleted": tuple(str(item) for item in raw_schemas.deletedItems),
            "ordered": tuple(str(item) for item in raw_schemas.orderedItems),
        }
        owned_by_bucket = {
            name: tuple(token for token in tokens if _is_r3_owned_schema_token(token))
            for name, tokens in buckets.items()
        }
        noncanonical_buckets = {
            name: tokens
            for name, tokens in owned_by_bucket.items()
            if name != "prepended" and tokens
        }
        prepended_owned = owned_by_bucket["prepended"]
        planned_order = tuple(
            token for token in prepended_owned if token in set(contract.schema_order)
        )
        preserved_order = tuple(
            token
            for token in prepended_owned
            if token in set(contract.preserved_schema_order)
        )
        if (
            noncanonical_buckets
            or len(prepended_owned) != len(set(prepended_owned))
            or set(prepended_owned) != set(contract.schema_tokens)
            or planned_order != contract.schema_order
            or preserved_order != contract.preserved_schema_order
        ):
            _fail(
                "authored_graph_mismatch",
                f"R3 prim {scene_path} raw apiSchemas list-op does not exactly "
                "match canonical author-owned and source-preserved prepends",
            )

        properties = {str(prop.name): prop for prop in prim_spec.properties}
        observed_managed = {
            name
            for name in properties
            if _is_r3_owned_property_name(name) or name in contract.authored_properties
        }
        if observed_managed != contract.authored_properties:
            _fail(
                "authored_graph_mismatch",
                f"R3 prim {scene_path} raw property specs do not exactly match "
                f"the plan: unexpected="
                f"{sorted(observed_managed - contract.authored_properties)}, "
                f"missing={sorted(contract.authored_properties - observed_managed)}",
            )

        for name, (type_name, variability_name) in contract.attribute_specs.items():
            property_spec = properties[name]
            expected_default = contract.attribute_defaults[name]
            composed_attribute = prim.GetAttribute(name)
            composed_default = composed_attribute.Get() if composed_attribute else None
            if not isinstance(property_spec, Sdf.AttributeSpec) or (
                {str(key) for key in property_spec.ListInfoKeys()}
                != {"custom", "default", "typeName", "variability"}
                or property_spec.custom
                or property_spec.typeName != type_by_name[type_name]
                or property_spec.variability != variability_by_name[variability_name]
                or not _stored_values_equal(
                    property_spec.GetInfo("default"),
                    expected_default,
                )
                or not composed_attribute
                or composed_attribute.GetTypeName() != type_by_name[type_name]
                or composed_attribute.GetVariability()
                != variability_by_name[variability_name]
                or composed_attribute.IsCustom()
                or not _stored_values_equal(composed_default, expected_default)
            ):
                _fail(
                    "authored_graph_mismatch",
                    f"R3 prim {scene_path} attribute {name!r} has "
                    "noncanonical raw metadata or default",
                )

        for name, target in contract.relationship_targets.items():
            property_spec = properties[name]
            if not isinstance(property_spec, Sdf.RelationshipSpec) or (
                {str(key) for key in property_spec.ListInfoKeys()}
                != {"custom", "targetPaths", "variability"}
                or property_spec.custom
                or property_spec.variability != Sdf.VariabilityUniform
            ):
                _fail(
                    "authored_graph_mismatch",
                    f"R3 prim {scene_path} relationship {name!r} has "
                    "noncanonical raw metadata",
                )
            target_spec_path = edit_target.MapToSpecPath(Sdf.Path(target))
            if target_spec_path == Sdf.Path.emptyPath:
                _fail(
                    "authored_graph_mismatch",
                    f"R3 relationship target {target!r} at {scene_path} cannot "
                    "be mapped into the active edit target",
                )
            targets = property_spec.GetInfo("targetPaths")
            if not isinstance(targets, Sdf.PathListOp) or (
                not targets.isExplicit
                or tuple(str(item) for item in targets.explicitItems)
                != (str(target_spec_path),)
                or tuple(targets.addedItems)
                or tuple(targets.prependedItems)
                or tuple(targets.appendedItems)
                or tuple(targets.deletedItems)
                or tuple(targets.orderedItems)
            ):
                _fail(
                    "authored_graph_mismatch",
                    f"R3 prim {scene_path} relationship {name!r} has a "
                    "noncanonical raw target list-op",
                )


def _existing_articulation_root_paths(
    stage: Any,
    *,
    Sdf: Any,
    Usd: Any,
) -> set[str]:
    """Return composed roots, including instance and inactive subtrees."""

    def has_articulation_root(prim: Any) -> bool:
        return "PhysicsArticulationRootAPI" in _applied_schema_tokens(prim)

    paths: set[str] = set()
    prim_visits = 0

    def add_path(path: str) -> None:
        if path in paths:
            return
        if len(paths) >= _MAX_R3_ARTICULATION_ROOT_PATHS:
            _fail(
                "r3_articulation_root_path_limit_exceeded",
                "R3 articulation-root discovery exceeds the fixed "
                f"{_MAX_R3_ARTICULATION_ROOT_PATHS}-path retention limit",
            )
        paths.add(path)

    def collect_articulation_root(prim: Any) -> bool:
        if not has_articulation_root(prim):
            return False
        add_path(str(prim.GetPath()))
        return True

    def inspect(prims: Any, *, phase: str) -> None:
        nonlocal prim_visits
        for prim in prims:
            prim_visits += 1
            if prim_visits > _MAX_R3_ARTICULATION_ROOT_PRIM_VISITS:
                _fail(
                    "r3_articulation_root_scan_limit_exceeded",
                    "R3 articulation-root discovery exceeds the fixed "
                    f"{_MAX_R3_ARTICULATION_ROOT_PRIM_VISITS}-prim visit "
                    f"limit during {phase}",
                )
            collect_articulation_root(prim)

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
    try:
        inactive_paths = _paths_with_inactive_ancestors_enabled(
            stage,
            matches=collect_articulation_root,
            Sdf=Sdf,
            Usd=Usd,
        )
    except JointRiggerContractError as exc:
        if exc.code not in {
            "source_joint_scan_limit_exceeded",
            "source_joint_scan_failed",
        }:
            raise
        reason_code = (
            "r3_articulation_root_scan_limit_exceeded"
            if exc.code == "source_joint_scan_limit_exceeded"
            else "r3_articulation_root_scan_failed"
        )
        raise JointRiggerContractError(
            reason_code,
            f"R3 articulation-root inactive-subtree discovery failed: {exc.detail}",
        ) from exc
    for path in inactive_paths:
        add_path(path)
    return paths


def _preflight(stage: Any, plan: JointRiggerPhysicsPlan) -> _Preflight:
    try:
        from pxr import Sdf, Usd, UsdGeom, UsdPhysics
    except ImportError as exc:  # pragma: no cover - optional-runtime guard
        raise JointRiggerContractError(
            "openusd_unavailable",
            "OpenUSD bindings are required for physics schema authoring",
        ) from exc

    validate_physics_plan_evidence(plan)
    endpoint_paths = {
        endpoint
        for joint in plan.joints
        for endpoint in (joint.topology.body0, joint.topology.body1)
    }
    body_paths = {body.prim_path for body in plan.rigid_bodies}
    assert body_paths == endpoint_paths
    graph_roots = _graph_roots(plan, body_paths)
    planned_roots = tuple(root.prim_path for root in _articulation_roots(plan))
    assert planned_roots == graph_roots

    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    kilograms_per_unit = float(UsdPhysics.GetStageKilogramsPerUnit(stage))
    if not math.isfinite(meters_per_unit) or meters_per_unit <= 0.0:
        _fail(
            "invalid_stage_units",
            f"metersPerUnit must be positive and finite; got {meters_per_unit!r}",
        )
    if not math.isfinite(kilograms_per_unit) or kilograms_per_unit <= 0.0:
        _fail(
            "invalid_stage_units",
            f"kilogramsPerUnit must be positive and finite; got {kilograms_per_unit!r}",
        )

    nested_body_world_matrices: dict[str, tuple[float, ...]] = {}
    xform_cache = UsdGeom.XformCache()
    for path in sorted(body_paths):
        prim = _require_target_prim(stage, path, kind="rigid body")
        if prim.IsInstanceable():
            _fail(
                "rigid_body_uneditable",
                f"rigid body cannot be authored on an instanceable prim: {path}",
            )
        xformable = UsdGeom.Xformable(prim)
        if not xformable:
            _fail("rigid_body_not_xformable", f"rigid body is not Xformable: {path}")
        nested_owners = [
            other
            for other in body_paths
            if other != path and path.startswith(f"{other}/")
        ]
        _require_static_transform_chain(prim)
        if nested_owners and not xformable.GetResetXformStack():
            if prim.GetAttribute(_NESTED_BODY_RESET_OP_NAME) or prim.GetRelationship(
                _NESTED_BODY_RESET_OP_NAME
            ):
                _fail(
                    "nested_body_reset_conflict",
                    "nested body already has the reserved Joint Rigger reset "
                    f"property: {path}.{_NESTED_BODY_RESET_OP_NAME}",
                )
            world_matrix = xform_cache.GetLocalToWorldTransform(prim)
            matrix_values = tuple(
                float(world_matrix[row][column])
                for row in range(4)
                for column in range(4)
            )
            if any(not math.isfinite(value) for value in matrix_values):
                _fail(
                    "nested_body_transform_invalid",
                    f"nested body has a non-finite world transform: {path}",
                )
            nested_body_world_matrices[path] = matrix_values

    for body in plan.rigid_bodies:
        _preflight_body(
            stage,
            body,
            body_paths=body_paths,
            meters_per_unit=meters_per_unit,
            kilograms_per_unit=kilograms_per_unit,
            UsdGeom=UsdGeom,
        )

    existing_roots = _existing_articulation_root_paths(stage, Sdf=Sdf, Usd=Usd)
    if not existing_roots.issubset(graph_roots):
        _fail(
            "articulation_root_ambiguous",
            f"stage has articulation roots outside the planned graph roots: "
            f"{sorted(existing_roots)}",
        )

    planned_joint_paths: set[str] = set()
    for joint in plan.joints:
        topology = joint.topology
        path_validation = Sdf.Path.IsValidPathString(topology.joint_id)
        path_is_valid = (
            path_validation[0]
            if isinstance(path_validation, tuple)
            else path_validation
        )
        if not path_is_valid:
            _fail(
                "joint_path_required",
                f"stage schema authoring requires joint_id to be an exact "
                f"absolute prim path: {topology.joint_id!r}",
            )
        joint_path = Sdf.Path(topology.joint_id)
        if (
            str(joint_path) != topology.joint_id
            or not joint_path.IsAbsolutePath()
            or not joint_path.IsPrimPath()
            or joint_path.IsAbsoluteRootPath()
            or joint_path.ContainsPrimVariantSelection()
        ):
            _fail(
                "joint_path_required",
                f"stage schema authoring requires joint_id to be an exact "
                f"absolute prim path: {topology.joint_id!r}",
            )
        planned_joint_paths.add(topology.joint_id)

    stage_joint_paths = _existing_joint_paths(
        stage,
        Sdf=Sdf,
        Usd=Usd,
        UsdPhysics=UsdPhysics,
    )
    extra_joint_paths = stage_joint_paths - planned_joint_paths
    if extra_joint_paths:
        _fail(
            "unplanned_joint_schema",
            f"stage has active, defined joints outside the exact plan graph: "
            f"{sorted(extra_joint_paths)}",
        )

    contexts: dict[str, _JointContext] = {}
    xform_cache = UsdGeom.XformCache()
    for joint in plan.joints:
        topology = joint.topology
        prim = _require_target_prim(stage, topology.joint_id, kind="joint")
        expected_type = _JOINT_TYPE_NAMES[topology.joint_type]
        if str(prim.GetTypeName()) != expected_type:
            _fail(
                "joint_type_mismatch",
                f"{topology.joint_id} has type {prim.GetTypeName()!s}, expected "
                f"{expected_type}",
            )
        _require_relationship_target(prim, "physics:body0", topology.body0)
        _require_relationship_target(prim, "physics:body1", topology.body1)
        axis_token = _preflight_axis(
            stage,
            prim,
            joint,
            xform_cache=xform_cache,
        )
        _preflight_anchor(
            stage,
            prim,
            joint,
            xform_cache=xform_cache,
        )
        _preflight_limits(
            prim,
            joint,
            meters_per_unit=meters_per_unit,
        )
        context = _JointContext(
            plan=joint,
            prim=prim,
            motion=_MOTIONS.get(topology.joint_type),
            axis_token=axis_token,
        )
        contexts[topology.joint_id] = context

    for context in contexts.values():
        _preflight_joint_state(context)
        _preflight_joint_control(context, contexts)
    return _Preflight(
        graph_roots=graph_roots,
        joints=contexts,
        nested_body_world_matrices=nested_body_world_matrices,
    )


def _articulation_roots(
    plan: JointRiggerPhysicsPlan,
) -> tuple[ArticulationRootPlanV1, ...]:
    if isinstance(plan, JointRiggerPlanV1):
        return () if plan.articulation_root is None else (plan.articulation_root,)
    return plan.articulation_roots


def _graph_roots(
    plan: JointRiggerPhysicsPlan,
    body_paths: set[str],
) -> tuple[str, ...]:
    incoming = dict.fromkeys(body_paths, 0)
    adjacency: dict[str, list[str]] = {path: [] for path in body_paths}
    for joint in plan.joints:
        body0 = joint.topology.body0
        body1 = joint.topology.body1
        incoming[body1] += 1
        adjacency[body0].append(body1)
        if incoming[body1] > 1:
            _fail(
                "ambiguous_joint_graph",
                f"body has multiple incoming joints: {body1}",
            )

    roots = tuple(sorted(path for path, count in incoming.items() if count == 0))
    if isinstance(plan, JointRiggerPlanV1) and len(roots) != 1:
        _fail(
            "ambiguous_joint_graph",
            f"expected one directed graph root, found {list(roots)}",
        )

    visited: set[str] = set()
    pending = list(reversed(roots))
    while pending:
        path = pending.pop()
        if path in visited:
            continue
        visited.add(path)
        pending.extend(sorted(adjacency[path], reverse=True))
    if visited != body_paths:
        missing = sorted(body_paths - visited)
        if isinstance(plan, JointRiggerPlanV1):
            _fail(
                "disconnected_joint_graph",
                f"graph root does not reach bodies {missing}",
            )
        _fail(
            "cyclic_joint_graph",
            f"directed joint graph contains a rootless cycle: {missing}",
        )
    return roots


def _is_instance_root_xform(prim: Any, *, UsdGeom: Any) -> bool:
    """Return whether ``prim`` is an editable Xform instance root."""

    return bool(
        prim.IsA(UsdGeom.Xform) and prim.IsInstance() and not prim.IsInstanceProxy()
    )


def _supports_mesh_collision_api(prim: Any, *, UsdGeom: Any) -> bool:
    """Return whether R3 owns MeshCollisionAPI on this exact prim kind."""

    return bool(
        prim.IsA(UsdGeom.Mesh) or _is_instance_root_xform(prim, UsdGeom=UsdGeom)
    )


def _preflight_body(
    stage: Any,
    body: Any,
    *,
    body_paths: set[str],
    meters_per_unit: float,
    kilograms_per_unit: float,
    UsdGeom: Any,
) -> None:
    prim = stage.GetPrimAtPath(body.prim_path)
    _preflight_attr(
        prim,
        "physics:rigidBodyEnabled",
        True,
        expected_type="bool",
    )
    _preflight_attr(
        prim,
        "physics:kinematicEnabled",
        False,
        expected_type="bool",
    )
    for name, expected, value_type in (
        ("physics:startsAsleep", False, "bool"),
        ("physics:velocity", (0.0, 0.0, 0.0), "vector3f"),
        ("physics:angularVelocity", (0.0, 0.0, 0.0), "vector3f"),
    ):
        _preflight_attr(
            prim,
            name,
            expected,
            expected_type=value_type,
        )
    if body.mass is None:
        _fail(
            "mass_evidence_missing",
            f"complete mass and inertia evidence is required for {body.prim_path}",
        )
    if not body.colliders:
        _fail(
            "collider_evidence_missing",
            "at least one exact supported collision-owner prim is required for "
            f"{body.prim_path}",
        )

    for unsupported in ("physics:density",):
        attr = prim.GetAttribute(unsupported)
        if attr and _has_authored_value(attr, owner=str(prim.GetPath())):
            _require_static(attr, owner=str(prim.GetPath()))
            _fail(
                "mass_schema_conflict",
                f"unrepresented authored {unsupported} exists at {prim.GetPath()}",
            )

    mass_value, center_of_mass_value, inertia_value = _mass_stage_values(
        body.mass,
        meters_per_unit=meters_per_unit,
        kilograms_per_unit=kilograms_per_unit,
    )
    _preflight_attr(
        prim,
        "physics:mass",
        mass_value,
        expected_type="float",
    )
    if center_of_mass_value is None:
        center_of_mass = prim.GetAttribute("physics:centerOfMass")
        if center_of_mass and _has_authored_value(
            center_of_mass,
            owner=str(prim.GetPath()),
        ):
            _require_static(center_of_mass, owner=str(prim.GetPath()))
            _fail(
                "mass_schema_conflict",
                f"unplanned center of mass exists at {prim.GetPath()}",
            )
    else:
        _preflight_attr(
            prim,
            "physics:centerOfMass",
            center_of_mass_value,
            expected_type="point3f",
        )
    _preflight_attr(
        prim,
        "physics:diagonalInertia",
        inertia_value,
        expected_type="float3",
    )
    if body.mass.principal_axes is None:
        principal = prim.GetAttribute("physics:principalAxes")
        if principal and _has_authored_value(
            principal,
            owner=str(prim.GetPath()),
        ):
            _require_static(principal, owner=str(prim.GetPath()))
            _fail(
                "mass_schema_conflict",
                f"unplanned principal axes exist at {prim.GetPath()}",
            )
    else:
        _preflight_attr(
            prim,
            "physics:principalAxes",
            body.mass.principal_axes,
            expected_type="quatf",
        )

    for collider in body.colliders:
        collider_prim = _require_target_prim(
            stage,
            collider.prim_path,
            kind="collider",
        )
        is_gprim = collider_prim.IsA(UsdGeom.Gprim)
        is_instance_root_xform = _is_instance_root_xform(
            collider_prim,
            UsdGeom=UsdGeom,
        )
        if not is_gprim and not is_instance_root_xform:
            _fail(
                "collider_not_gprim",
                "planned collider is neither a GPrim nor an Xform instance root: "
                f"{collider.prim_path}",
            )
        owners = [
            path
            for path in body_paths
            if collider.prim_path == path or collider.prim_path.startswith(f"{path}/")
        ]
        nearest_owner = max(owners, key=lambda path: len(path.split("/")))
        if nearest_owner != body.prim_path:
            _fail(
                "collider_ownership_ambiguous",
                f"{collider.prim_path} belongs to nearest planned body "
                f"{nearest_owner}, not {body.prim_path}",
            )
        is_mesh = collider_prim.IsA(UsdGeom.Mesh)
        supports_mesh_collision = is_mesh or is_instance_root_xform
        if is_instance_root_xform and (
            not collider.has_mesh_collision_api or collider.mesh_approximation is None
        ):
            _fail(
                "instance_root_collider_evidence_incomplete",
                "Xform instance-root colliders require explicit "
                "PhysicsMeshCollisionAPI evidence and an authored approximation: "
                f"{collider.prim_path}",
            )
        if not supports_mesh_collision and collider.has_mesh_collision_api:
            _fail(
                "mesh_collision_api_not_applicable",
                "unsupported collision owner carries PhysicsMeshCollisionAPI "
                "evidence: "
                f"{collider.prim_path}",
            )
        tokens = _applied_schema_tokens(collider_prim)
        approximation = collider_prim.GetAttribute("physics:approximation")
        has_approximation = bool(
            approximation
            and _has_authored_value(
                approximation,
                owner=collider.prim_path,
            )
        )
        has_mesh_schema = "PhysicsMeshCollisionAPI" in tokens
        if not supports_mesh_collision or not collider.has_mesh_collision_api:
            if has_mesh_schema or has_approximation:
                _fail(
                    "collider_schema_conflict",
                    "collider has unplanned PhysicsMeshCollisionAPI evidence: "
                    f"{collider.prim_path}",
                )
        elif collider.mesh_approximation is None and has_approximation:
            _fail(
                "collider_schema_conflict",
                f"bare MeshCollisionAPI plan has an unplanned approximation at "
                f"{collider.prim_path}",
            )
        _preflight_attr(
            collider_prim,
            "physics:collisionEnabled",
            True,
            expected_type="bool",
        )
        if supports_mesh_collision and collider.mesh_approximation is not None:
            _preflight_attr(
                collider_prim,
                "physics:approximation",
                collider.mesh_approximation,
                expected_type="token",
            )


def _preflight_axis(
    stage: Any,
    prim: Any,
    joint: JointPlanV1,
    *,
    xform_cache: Any,
) -> str | None:
    topology = joint.topology
    if topology.joint_type == "spherical":
        return None
    axis_attr = prim.GetAttribute("physics:axis")
    if not axis_attr or not _has_authored_value(
        axis_attr,
        owner=topology.joint_id,
    ):
        _fail(
            "axis_unresolved",
            f"joint lacks an authored axis: {topology.joint_id}",
        )
    _require_static(axis_attr, owner=topology.joint_id)
    axis_token = str(axis_attr.Get()).strip().lower()
    base = _AXES.get(axis_token)
    if base is None:
        _fail(
            "axis_unresolved",
            f"unsupported physics:axis {axis_attr.Get()!r} at {topology.joint_id}",
        )

    from pxr import Gf

    frame_axes: list[tuple[float, float, float]] = []
    for index, body_path in enumerate((topology.body0, topology.body1)):
        vector = Gf.Vec3d(*base)
        rotation_attr = prim.GetAttribute(f"physics:localRot{index}")
        if rotation_attr and _has_authored_value(
            rotation_attr,
            owner=topology.joint_id,
        ):
            _require_static(rotation_attr, owner=topology.joint_id)
            rotation = rotation_attr.Get()
            if rotation is None:
                _fail(
                    "axis_unresolved",
                    f"physics:localRot{index} has no value at {topology.joint_id}",
                )
            vector = Gf.Rotation(rotation).TransformDir(vector)
        body = stage.GetPrimAtPath(body_path)
        vector = xform_cache.GetLocalToWorldTransform(body).TransformDir(vector)
        frame_axes.append(_normalized_vector(vector, owner=topology.joint_id))
    if _dot(frame_axes[0], frame_axes[1]) < 1.0 - _VALUE_TOLERANCE:
        _fail(
            "contradictory_joint_frames",
            f"joint endpoint frames establish different signed axes: "
            f"{topology.joint_id}",
        )
    assert topology.axis_stage is not None
    if _dot(frame_axes[0], topology.axis_stage) < 1.0 - _VALUE_TOLERANCE:
        _fail(
            "axis_mismatch",
            f"authored joint axis does not match plan axis_stage at "
            f"{topology.joint_id}",
        )
    return axis_token


def _preflight_anchor(
    stage: Any,
    prim: Any,
    joint: JointPlanV1,
    *,
    xform_cache: Any,
) -> None:
    from pxr import Gf

    attributes = (
        prim.GetAttribute("physics:localPos0"),
        prim.GetAttribute("physics:localPos1"),
    )
    authored = tuple(
        bool(
            attribute
            and _has_authored_value(
                attribute,
                owner=joint.topology.joint_id,
            )
        )
        for attribute in attributes
    )
    if any(authored) and not all(authored):
        _fail(
            "joint_anchor_incomplete",
            f"joint must author both local anchor positions: {joint.topology.joint_id}",
        )
    if not all(authored):
        if joint.anchor is not None:
            _fail(
                "joint_anchor_mismatch",
                f"planned anchor is absent at {joint.topology.joint_id}",
            )
        return

    positions: list[tuple[float, float, float]] = []
    for index, (attribute, body_path) in enumerate(
        zip(
            attributes,
            (joint.topology.body0, joint.topology.body1),
            strict=True,
        )
    ):
        _require_static(attribute, owner=joint.topology.joint_id)
        value = attribute.Get()
        if value is None:
            _fail(
                "joint_anchor_incomplete",
                f"physics:localPos{index} has no value at {joint.topology.joint_id}",
            )
        body = stage.GetPrimAtPath(body_path)
        position = xform_cache.GetLocalToWorldTransform(body).Transform(
            Gf.Vec3d(*value)
        )
        positions.append(
            (
                float(position[0]),
                float(position[1]),
                float(position[2]),
            )
        )
    if _distance(positions[0], positions[1]) > _SHARED_ANCHOR_DISTANCE_TOLERANCE:
        _fail(
            "contradictory_joint_frames",
            f"localPos0/localPos1 establish different anchors at "
            f"{joint.topology.joint_id}",
        )
    if joint.anchor is not None and (
        _distance(positions[0], joint.anchor.position_stage)
        > _SHARED_ANCHOR_DISTANCE_TOLERANCE
    ):
        _fail(
            "joint_anchor_mismatch",
            f"authored joint anchor differs from the plan at {joint.topology.joint_id}",
        )


def _preflight_limits(
    prim: Any,
    joint: JointPlanV1,
    *,
    meters_per_unit: float,
) -> None:
    limit = joint.limit
    for name in ("physics:lowerLimit", "physics:upperLimit"):
        attr = prim.GetAttribute(name)
        if attr and _has_authored_value(attr, owner=joint.topology.joint_id):
            _require_static(attr, owner=joint.topology.joint_id)
    if limit is None:
        authored = [
            name
            for name in ("physics:lowerLimit", "physics:upperLimit")
            if (attr := prim.GetAttribute(name))
            and _has_authored_value(attr, owner=joint.topology.joint_id)
        ]
        if authored:
            _fail(
                "limit_evidence_mismatch",
                f"stage has limits omitted by the plan at {joint.topology.joint_id}: "
                f"{authored}",
            )
        return

    divisor = meters_per_unit if joint.topology.joint_type == "prismatic" else 1.0
    for field, value in (("lowerLimit", limit.lower), ("upperLimit", limit.upper)):
        name = f"physics:{field}"
        attr = prim.GetAttribute(name)
        if value is None:
            if attr and _has_authored_value(
                attr,
                owner=joint.topology.joint_id,
            ):
                _fail(
                    "limit_evidence_mismatch",
                    f"unplanned {name} exists at {joint.topology.joint_id}",
                )
            continue
        if not attr or not _has_authored_value(
            attr,
            owner=joint.topology.joint_id,
        ):
            _fail(
                "limit_evidence_mismatch",
                f"planned {name} is absent at {joint.topology.joint_id}",
            )
        expected = float(value) / divisor
        if not _values_equal(attr.Get(), expected):
            _fail(
                "limit_evidence_mismatch",
                f"{name} differs from the plan at {joint.topology.joint_id}",
            )


def _preflight_joint_state(context: _JointContext) -> None:
    prim = context.prim
    joint = context.plan
    state_properties = {
        str(prop.GetName())
        for prop in prim.GetAuthoredProperties()
        if str(prop.GetName()).startswith("state:")
    }
    state_tokens = {
        token
        for token in _applied_schema_tokens(prim)
        if token.startswith("PhysicsJointStateAPI:")
    }
    if context.motion is None:
        if joint.state is not None or state_tokens or state_properties:
            _fail(
                "joint_state_not_applicable",
                f"spherical joint cannot carry scalar Joint State: "
                f"{joint.topology.joint_id}",
            )
        return
    if joint.state is None:
        _fail(
            "joint_state_evidence_missing",
            f"explicit Joint State disposition is required for "
            f"{joint.topology.joint_id}",
        )
    state = joint.state
    assert state is not None
    expected_token = f"PhysicsJointStateAPI:{context.motion}"
    extras = state_tokens - {expected_token}
    if extras:
        _fail(
            "joint_state_schema_conflict",
            f"incompatible Joint State instances at {joint.topology.joint_id}: "
            f"{sorted(extras)}",
        )
    names = (
        f"state:{context.motion}:physics:position",
        f"state:{context.motion}:physics:velocity",
    )
    extra_properties = state_properties - set(names)
    if extra_properties:
        _fail(
            "joint_state_schema_conflict",
            f"incompatible Joint State properties at {joint.topology.joint_id}: "
            f"{sorted(extra_properties)}",
        )
    existing_complete = expected_token in state_tokens and all(
        (attr := prim.GetAttribute(name))
        and _has_authored_value(attr, owner=joint.topology.joint_id)
        for name in names
    )
    if not existing_complete and (
        not math.isclose(state.position, 0.0, abs_tol=0.0)
        or not math.isclose(state.velocity, 0.0, abs_tol=0.0)
    ):
        _fail(
            "unsafe_new_joint_state",
            f"a new Joint State must use zero rest values at {joint.topology.joint_id}",
        )
    _preflight_attr(
        prim,
        names[0],
        state.position,
        expected_type="float",
    )
    _preflight_attr(
        prim,
        names[1],
        state.velocity,
        expected_type="float",
    )
    _require_position_inside_authored_limits(
        prim,
        state.position,
        owner=joint.topology.joint_id,
        code="joint_state_outside_limits",
    )


def _preflight_physx_joint(
    context: _JointContext,
    *,
    schema_tokens: set[str],
    authored_properties: set[str],
    conflict_code: str,
) -> None:
    joint = context.plan
    expected: dict[str, float] = {}
    if joint.drive is not None and joint.drive.max_joint_velocity is not None:
        expected["physxJoint:maxJointVelocity"] = joint.drive.max_joint_velocity
    if joint.joint_friction is not None:
        if context.motion is None:
            _fail(
                "joint_friction_not_applicable",
                f"spherical joint cannot carry scalar joint friction: "
                f"{joint.topology.joint_id}",
            )
        coefficient = joint.joint_friction.coefficient
        if not math.isfinite(coefficient) or coefficient < 0.0:
            _fail(
                "invalid_joint_friction",
                f"joint friction must be finite and nonnegative at "
                f"{joint.topology.joint_id}",
            )
        expected["physxJoint:jointFriction"] = coefficient
    if not expected:
        if schema_tokens or authored_properties:
            _fail(
                conflict_code,
                f"unplanned PhysxJointAPI opinions at {joint.topology.joint_id}: "
                f"{sorted(schema_tokens | authored_properties)}",
            )
        return
    token_extras = schema_tokens - {"PhysxJointAPI"}
    if token_extras:
        _fail(
            conflict_code,
            f"unrepresented PhysxJointAPI schemas at {joint.topology.joint_id}: "
            f"{sorted(token_extras)}",
        )
    property_extras = authored_properties - set(expected)
    if property_extras:
        _fail(
            conflict_code,
            f"unrepresented PhysxJointAPI properties at {joint.topology.joint_id}: "
            f"{sorted(property_extras)}",
        )
    for name, value in expected.items():
        _preflight_attr(
            prim=context.prim, name=name, expected=value, expected_type="float"
        )


def _preflight_joint_control(
    context: _JointContext,
    contexts: dict[str, _JointContext],
) -> None:
    prim = context.prim
    joint = context.plan
    tokens = _applied_schema_tokens(prim)
    drive_tokens = {token for token in tokens if token.startswith("PhysicsDriveAPI:")}
    mimic_tokens = {
        token for token in tokens if token.startswith("PhysxMimicJointAPI:")
    }
    physx_joint_tokens = {
        token
        for token in tokens
        if token == "PhysxJointAPI" or token.startswith("PhysxJointAPI:")
    }
    authored_properties = {str(prop.GetName()) for prop in prim.GetAuthoredProperties()}
    drive_properties = {
        name for name in authored_properties if name.startswith("drive:")
    }
    mimic_properties = {
        name for name in authored_properties if name.startswith("physxMimicJoint:")
    }
    physx_joint_properties = {
        name for name in authored_properties if name.startswith("physxJoint:")
    }
    if joint.drive is None and joint.mimic is None:
        conflicts = drive_tokens | mimic_tokens | drive_properties | mimic_properties
        if conflicts:
            _fail(
                "passive_control_schema_conflict",
                f"passive joint has drive/mimic schemas at "
                f"{joint.topology.joint_id}: "
                f"{sorted(conflicts)}",
            )
        _preflight_physx_joint(
            context,
            schema_tokens=physx_joint_tokens,
            authored_properties=physx_joint_properties,
            conflict_code="passive_control_schema_conflict",
        )
        return
    if context.motion is None:
        _fail(
            "joint_control_not_applicable",
            f"spherical joint has no supported scalar control schema: "
            f"{joint.topology.joint_id}",
        )

    if joint.drive is not None:
        if mimic_tokens or mimic_properties:
            _fail(
                "drive_schema_conflict",
                f"drive plan conflicts with mimic schema at {joint.topology.joint_id}",
            )
        expected = f"PhysicsDriveAPI:{context.motion}"
        extras = drive_tokens - {expected}
        if extras:
            _fail(
                "drive_schema_conflict",
                f"incompatible drive instances at {joint.topology.joint_id}: "
                f"{sorted(extras)}",
            )
        drive = joint.drive
        if drive.stiffness < 0.0 or drive.damping < 0.0 or drive.max_force < 0.0:
            _fail(
                "invalid_drive_values",
                f"stiffness, damping, and max_force must be nonnegative at "
                f"{joint.topology.joint_id}",
            )
        if drive.max_joint_velocity is not None and drive.max_joint_velocity < 0.0:
            _fail(
                "invalid_drive_values",
                f"max_joint_velocity must be nonnegative at {joint.topology.joint_id}",
            )
        specs = _drive_specs(context)
        extra_properties = drive_properties - {name for name, _, _ in specs}
        if extra_properties:
            _fail(
                "drive_schema_conflict",
                f"incompatible drive properties at {joint.topology.joint_id}: "
                f"{sorted(extra_properties)}",
            )
        existing_complete = expected in drive_tokens and all(
            (attr := prim.GetAttribute(name))
            and _has_authored_value(attr, owner=joint.topology.joint_id)
            for name, _, _ in specs
        )
        if not existing_complete and (
            not math.isclose(drive.target_position, 0.0, abs_tol=0.0)
            or not math.isclose(drive.target_velocity, 0.0, abs_tol=0.0)
        ):
            _fail(
                "unsafe_new_drive_target",
                f"new drives require zero rest targets at {joint.topology.joint_id}",
            )
        for name, value, value_type in specs:
            _preflight_attr(prim, name, value, expected_type=value_type)
        _preflight_physx_joint(
            context,
            schema_tokens=physx_joint_tokens,
            authored_properties=physx_joint_properties,
            conflict_code="drive_schema_conflict",
        )
        _require_position_inside_authored_limits(
            prim,
            drive.target_position,
            owner=joint.topology.joint_id,
            code="drive_target_outside_limits",
        )
        return

    assert joint.mimic is not None
    if joint.joint_friction is not None:
        _fail(
            "mimic_schema_conflict",
            f"mimic plan cannot carry joint friction at {joint.topology.joint_id}",
        )
    if joint.topology.joint_type != "revolute":
        _fail(
            "mimic_not_applicable",
            f"mimic is limited to revolute joints: {joint.topology.joint_id}",
        )
    reference = contexts[joint.mimic.reference_joint_id]
    if reference.plan.topology.joint_type != "revolute":
        _fail(
            "mimic_not_applicable",
            f"mimic reference is not revolute: {joint.mimic.reference_joint_id}",
        )
    if reference.plan.mimic is not None:
        _fail(
            "mimic_chain_unsupported",
            f"mimic reference cannot itself be a mimic: "
            f"{joint.mimic.reference_joint_id}",
        )
    assert joint.topology.axis_stage is not None
    assert reference.plan.topology.axis_stage is not None
    if _dot(joint.topology.axis_stage, reference.plan.topology.axis_stage) < (
        1.0 - _VALUE_TOLERANCE
    ):
        _fail(
            "mimic_axis_mismatch",
            f"mimic and reference signed axes differ at {joint.topology.joint_id}",
        )
    _require_complete_zero_spanning_limits(context)
    _require_complete_zero_spanning_limits(reference)
    if drive_tokens or drive_properties or physx_joint_tokens or physx_joint_properties:
        _fail(
            "mimic_schema_conflict",
            f"mimic plan conflicts with drive schema at {joint.topology.joint_id}",
        )
    assert context.axis_token is not None
    assert reference.axis_token is not None
    instance = f"rot{context.axis_token.upper()}"
    reference_instance = f"rot{reference.axis_token.upper()}"
    expected_token = f"PhysxMimicJointAPI:{instance}"
    extras = mimic_tokens - {expected_token}
    if extras:
        _fail(
            "mimic_schema_conflict",
            f"incompatible mimic instances at {joint.topology.joint_id}: "
            f"{sorted(extras)}",
        )
    namespace = f"physxMimicJoint:{instance}"
    specs = _mimic_specs(
        context,
        reference_axis=reference_instance,
    )
    expected_properties = {
        *(name for name, _, _ in specs),
        f"{namespace}:referenceJoint",
    }
    extra_properties = mimic_properties - expected_properties
    if extra_properties:
        _fail(
            "mimic_schema_conflict",
            f"incompatible mimic properties at {joint.topology.joint_id}: "
            f"{sorted(extra_properties)}",
        )
    for name, value, value_type in specs:
        _preflight_attr(prim, name, value, expected_type=value_type)
    relationship_name = f"{namespace}:referenceJoint"
    if prim.GetAttribute(relationship_name):
        _fail(
            "mimic_schema_conflict",
            f"planned mimic relationship name is already an Attribute at "
            f"{joint.topology.joint_id}: {relationship_name}",
        )
    relationship = prim.GetRelationship(relationship_name)
    if relationship:
        targets = tuple(str(path) for path in relationship.GetTargets())
        expected_targets = (joint.mimic.reference_joint_id,)
        if targets and targets != expected_targets:
            _fail(
                "mimic_schema_conflict",
                f"conflicting mimic reference at {joint.topology.joint_id}: {targets}",
            )


def _apply(stage: Any, plan: JointRiggerPhysicsPlan, preflight: _Preflight) -> None:
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    for path in sorted(
        preflight.nested_body_world_matrices,
        key=lambda item: (item.count("/"), item),
    ):
        prim = stage.GetPrimAtPath(path)
        xformable = UsdGeom.Xformable(prim)
        matrix_op = xformable.AddTransformOp(
            UsdGeom.XformOp.PrecisionDouble,
            _NESTED_BODY_RESET_OP_SUFFIX,
        )
        matrix = Gf.Matrix4d(*preflight.nested_body_world_matrices[path])
        if not matrix_op.Set(matrix) or not xformable.SetXformOpOrder(
            [matrix_op],
            resetXformStack=True,
        ):
            _fail(
                "nested_body_reset_failed",
                f"could not author the world-preserving reset matrix at {path}",
            )

    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    kilograms_per_unit = float(UsdPhysics.GetStageKilogramsPerUnit(stage))
    for body in plan.rigid_bodies:
        prim = stage.GetPrimAtPath(body.prim_path)
        rigid_body = UsdPhysics.RigidBodyAPI.Apply(prim)
        _require_application(rigid_body, "RigidBodyAPI", prim)
        rigid_body.CreateRigidBodyEnabledAttr(True)
        rigid_body.CreateKinematicEnabledAttr(False)

        assert body.mass is not None
        mass_value, center_of_mass_value, inertia_value = _mass_stage_values(
            body.mass,
            meters_per_unit=meters_per_unit,
            kilograms_per_unit=kilograms_per_unit,
        )
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        _require_application(mass_api, "MassAPI", prim)
        mass_api.CreateMassAttr(float(mass_value))
        if center_of_mass_value is not None:
            mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*center_of_mass_value))
        mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*inertia_value))
        if body.mass.principal_axes is not None:
            real, x, y, z = body.mass.principal_axes
            mass_api.CreatePrincipalAxesAttr(
                Gf.Quatf(float(real), Gf.Vec3f(float(x), float(y), float(z)))
            )

        for collider in body.colliders:
            collider_prim = stage.GetPrimAtPath(collider.prim_path)
            collision = UsdPhysics.CollisionAPI.Apply(collider_prim)
            _require_application(collision, "CollisionAPI", collider_prim)
            collision.CreateCollisionEnabledAttr(True)
            if (
                _supports_mesh_collision_api(
                    collider_prim,
                    UsdGeom=UsdGeom,
                )
                and collider.has_mesh_collision_api
            ):
                mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(collider_prim)
                _require_application(
                    mesh_collision,
                    "MeshCollisionAPI",
                    collider_prim,
                )
                if collider.mesh_approximation is not None:
                    mesh_collision.CreateApproximationAttr(collider.mesh_approximation)

    for graph_root in preflight.graph_roots:
        root = stage.GetPrimAtPath(graph_root)
        root_api = UsdPhysics.ArticulationRootAPI.Apply(root)
        _require_application(root_api, "ArticulationRootAPI", root)

    for context in preflight.joints.values():
        joint = context.plan
        prim = context.prim
        if joint.state is not None:
            assert context.motion is not None
            token = f"PhysicsJointStateAPI:{context.motion}"
            _require_application(prim.AddAppliedSchema(token), token, prim)
            prim.CreateAttribute(
                f"state:{context.motion}:physics:position",
                Sdf.ValueTypeNames.Float,
                custom=False,
            ).Set(float(joint.state.position))
            prim.CreateAttribute(
                f"state:{context.motion}:physics:velocity",
                Sdf.ValueTypeNames.Float,
                custom=False,
            ).Set(float(joint.state.velocity))

        if joint.drive is not None:
            assert context.motion is not None
            drive = joint.drive
            drive_api = UsdPhysics.DriveAPI.Apply(prim, context.motion)
            _require_application(
                drive_api,
                f"DriveAPI:{context.motion}",
                prim,
            )
            drive_api.CreateTypeAttr(drive.drive_type)
            drive_api.CreateStiffnessAttr(float(drive.stiffness))
            drive_api.CreateDampingAttr(float(drive.damping))
            drive_api.CreateMaxForceAttr(float(drive.max_force))
            drive_api.CreateTargetPositionAttr(float(drive.target_position))
            drive_api.CreateTargetVelocityAttr(float(drive.target_velocity))
        elif joint.mimic is not None:
            _apply_mimic(context, preflight.joints, Sdf=Sdf)
        if (
            joint.joint_friction is not None
            or joint.drive is not None
            and joint.drive.max_joint_velocity is not None
        ):
            _require_application(
                prim.AddAppliedSchema("PhysxJointAPI"),
                "PhysxJointAPI",
                prim,
            )
            if joint.drive is not None and joint.drive.max_joint_velocity is not None:
                prim.CreateAttribute(
                    "physxJoint:maxJointVelocity",
                    Sdf.ValueTypeNames.Float,
                    custom=False,
                ).Set(float(joint.drive.max_joint_velocity))
            if joint.joint_friction is not None:
                prim.CreateAttribute(
                    "physxJoint:jointFriction",
                    Sdf.ValueTypeNames.Float,
                    custom=False,
                ).Set(float(joint.joint_friction.coefficient))


def _apply_mimic(
    context: _JointContext,
    contexts: dict[str, _JointContext],
    *,
    Sdf: Any,
) -> None:
    mimic = context.plan.mimic
    assert mimic is not None
    assert context.axis_token is not None
    reference = contexts[mimic.reference_joint_id]
    assert reference.axis_token is not None
    instance = f"rot{context.axis_token.upper()}"
    reference_instance = f"rot{reference.axis_token.upper()}"
    token = f"PhysxMimicJointAPI:{instance}"
    _require_application(context.prim.AddAppliedSchema(token), token, context.prim)
    for name, value, value_type in _mimic_specs(
        context,
        reference_axis=reference_instance,
    ):
        sdf_type = (
            Sdf.ValueTypeNames.Token
            if value_type == "token"
            else Sdf.ValueTypeNames.Float
        )
        context.prim.CreateAttribute(name, sdf_type, custom=False).Set(value)
    namespace = f"physxMimicJoint:{instance}"
    context.prim.CreateRelationship(
        f"{namespace}:referenceJoint",
        custom=False,
    ).SetTargets([Sdf.Path(mimic.reference_joint_id)])


def _validate_authored(
    stage: Any,
    plan: JointRiggerPhysicsPlan,
    preflight: _Preflight,
) -> None:
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    for path, expected_matrix in preflight.nested_body_world_matrices.items():
        xformable = UsdGeom.Xformable(stage.GetPrimAtPath(path))
        op_order = tuple(str(token) for token in xformable.GetXformOpOrderAttr().Get())
        if op_order != ("!resetXformStack!", _NESTED_BODY_RESET_OP_NAME):
            _fail(
                "postwrite_validation_failed",
                f"nested body does not have the canonical reset stack: {path}",
            )
        ordered_ops = xformable.GetOrderedXformOps()
        if len(ordered_ops) != 1:
            _fail(
                "postwrite_validation_failed",
                f"nested body reset stack is ambiguous: {path}",
            )
        actual = ordered_ops[0].Get()
        actual_matrix = tuple(
            float(actual[row][column]) for row in range(4) for column in range(4)
        )
        if not all(
            math.isclose(
                expected,
                observed,
                rel_tol=_VALUE_TOLERANCE,
                abs_tol=_VALUE_TOLERANCE,
            )
            for expected, observed in zip(
                expected_matrix,
                actual_matrix,
                strict=True,
            )
        ):
            _fail(
                "postwrite_validation_failed",
                f"nested body reset matrix does not preserve its world pose: {path}",
            )

    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    kilograms_per_unit = float(UsdPhysics.GetStageKilogramsPerUnit(stage))
    for body in plan.rigid_bodies:
        prim = stage.GetPrimAtPath(body.prim_path)
        _require_schema(prim, "PhysicsRigidBodyAPI")
        _require_schema(prim, "PhysicsMassAPI")
        _require_exact_attr(prim, "physics:rigidBodyEnabled", True)
        _require_exact_attr(prim, "physics:kinematicEnabled", False)
        assert body.mass is not None
        mass_value, center_of_mass_value, inertia_value = _mass_stage_values(
            body.mass,
            meters_per_unit=meters_per_unit,
            kilograms_per_unit=kilograms_per_unit,
        )
        _require_exact_attr(prim, "physics:mass", mass_value)
        if center_of_mass_value is not None:
            _require_exact_attr(
                prim,
                "physics:centerOfMass",
                center_of_mass_value,
            )
        _require_exact_attr(prim, "physics:diagonalInertia", inertia_value)
        if body.mass.principal_axes is not None:
            _require_exact_attr(
                prim,
                "physics:principalAxes",
                body.mass.principal_axes,
            )
        for collider in body.colliders:
            collider_prim = stage.GetPrimAtPath(collider.prim_path)
            _require_schema(collider_prim, "PhysicsCollisionAPI")
            _require_exact_attr(collider_prim, "physics:collisionEnabled", True)
            if (
                _supports_mesh_collision_api(
                    collider_prim,
                    UsdGeom=UsdGeom,
                )
                and collider.has_mesh_collision_api
            ):
                _require_schema(collider_prim, "PhysicsMeshCollisionAPI")
                if collider.mesh_approximation is not None:
                    _require_exact_attr(
                        collider_prim,
                        "physics:approximation",
                        collider.mesh_approximation,
                    )
                else:
                    _require_absent_authored_attr(
                        collider_prim,
                        "physics:approximation",
                    )
            else:
                _require_schema_absent(collider_prim, "PhysicsMeshCollisionAPI")
                _require_absent_authored_attr(
                    collider_prim,
                    "physics:approximation",
                )

    roots = _existing_articulation_root_paths(stage, Sdf=Sdf, Usd=Usd)
    expected_roots = set(preflight.graph_roots)
    if roots != expected_roots:
        _fail(
            "postwrite_validation_failed",
            f"expected articulation roots at {sorted(expected_roots)}, "
            f"got {sorted(roots)}",
        )

    expected_joint_paths = set(preflight.joints)
    observed_joint_paths = _existing_joint_paths(
        stage,
        Sdf=Sdf,
        Usd=Usd,
        UsdPhysics=UsdPhysics,
    )
    if observed_joint_paths != expected_joint_paths:
        _fail(
            "postwrite_validation_failed",
            "authored stage joints do not exactly match the plan: "
            f"unexpected={sorted(observed_joint_paths - expected_joint_paths)}, "
            f"missing={sorted(expected_joint_paths - observed_joint_paths)}",
        )

    for context in preflight.joints.values():
        joint = context.plan
        prim = context.prim
        if joint.state is not None:
            assert context.motion is not None
            _require_schema(prim, f"PhysicsJointStateAPI:{context.motion}")
            _require_exact_attr(
                prim,
                f"state:{context.motion}:physics:position",
                joint.state.position,
            )
            _require_exact_attr(
                prim,
                f"state:{context.motion}:physics:velocity",
                joint.state.velocity,
            )
        if joint.drive is not None:
            assert context.motion is not None
            _require_schema(prim, f"PhysicsDriveAPI:{context.motion}")
            for name, value, _ in _drive_specs(context):
                _require_exact_attr(prim, name, value)
        elif joint.mimic is not None:
            assert context.axis_token is not None
            reference = preflight.joints[joint.mimic.reference_joint_id]
            assert reference.axis_token is not None
            instance = f"rot{context.axis_token.upper()}"
            reference_instance = f"rot{reference.axis_token.upper()}"
            _require_schema(prim, f"PhysxMimicJointAPI:{instance}")
            for name, value, _ in _mimic_specs(
                context,
                reference_axis=reference_instance,
            ):
                _require_exact_attr(prim, name, value)
            targets = tuple(
                str(path)
                for path in prim.GetRelationship(
                    f"physxMimicJoint:{instance}:referenceJoint"
                ).GetTargets()
            )
            if targets != (joint.mimic.reference_joint_id,):
                _fail(
                    "postwrite_validation_failed",
                    f"mimic reference mismatch at {joint.topology.joint_id}",
                )
        if (
            joint.joint_friction is not None
            or joint.drive is not None
            and joint.drive.max_joint_velocity is not None
        ):
            _require_schema(prim, "PhysxJointAPI")
            if joint.drive is not None and joint.drive.max_joint_velocity is not None:
                _require_exact_attr(
                    prim,
                    "physxJoint:maxJointVelocity",
                    joint.drive.max_joint_velocity,
                )
            if joint.joint_friction is not None:
                _require_exact_attr(
                    prim,
                    "physxJoint:jointFriction",
                    joint.joint_friction.coefficient,
                )

    _validate_owned_physics_raw_authorship(stage, plan, preflight)


def _diagnostics(
    plan: JointRiggerPhysicsPlan,
    *,
    backend_name: str,
    backend_version: str | None,
) -> JointRiggerDiagnosticsV1:
    top_level: list[FieldDecisionV1] = []
    for body in plan.rigid_bodies:
        prefix = f"rigid_bodies[{body.prim_path}]"
        top_level.append(
            FieldDecisionV1(
                field=f"{prefix}.rigid_body",
                disposition="accepted",
                provenance=body.provenance,
                detail="Applied enabled dynamic RigidBodyAPI to the exact endpoint.",
            )
        )
        assert body.mass is not None
        top_level.extend(
            (
                FieldDecisionV1(
                    field=f"{prefix}.mass.mass_kg",
                    disposition="accepted",
                    provenance=body.mass.provenance,
                    detail="Applied the explicit SI-backed mass.",
                ),
                _planned_leaf_decision(
                    f"{prefix}.mass.center_of_mass_m",
                    present=body.mass.center_of_mass_m is not None,
                    provenance=body.mass.provenance,
                    accepted_detail=(
                        "Applied the explicit body-local SI center of mass."
                    ),
                    absent_reason="not_planned",
                ),
                FieldDecisionV1(
                    field=f"{prefix}.mass.diagonal_inertia_kg_m2",
                    disposition="accepted",
                    provenance=body.mass.provenance,
                    detail="Applied the explicit SI-backed diagonal inertia.",
                ),
                _planned_leaf_decision(
                    f"{prefix}.mass.principal_axes",
                    present=body.mass.principal_axes is not None,
                    provenance=body.mass.provenance,
                    accepted_detail="Applied the explicit principal axes.",
                    absent_reason="not_planned",
                ),
            )
        )
        for collider in body.colliders:
            collider_prefix = f"{prefix}.colliders[{collider.prim_path}]"
            top_level.extend(
                (
                    FieldDecisionV1(
                        field=f"{collider_prefix}.collision",
                        disposition="accepted",
                        provenance=collider.provenance,
                        detail=(
                            "Applied CollisionAPI to the exact planned supported "
                            "collision-owner prim."
                        ),
                    ),
                    _planned_leaf_decision(
                        f"{collider_prefix}.mesh_collision_api",
                        present=collider.has_mesh_collision_api,
                        provenance=collider.provenance,
                        accepted_detail=(
                            "Applied the explicitly represented "
                            "PhysicsMeshCollisionAPI."
                        ),
                        absent_reason="not_planned",
                    ),
                    _planned_leaf_decision(
                        f"{collider_prefix}.mesh_approximation",
                        present=collider.mesh_approximation is not None,
                        provenance=collider.provenance,
                        accepted_detail="Applied the explicit mesh approximation.",
                        absent_reason="not_planned",
                    ),
                )
            )
    roots = _articulation_roots(plan)
    assert roots
    for root in roots:
        field = (
            "articulation_root"
            if isinstance(plan, JointRiggerPlanV1)
            else f"articulation_roots[{root.prim_path}]"
        )
        top_level.append(
            FieldDecisionV1(
                field=field,
                disposition="accepted",
                provenance=root.provenance,
                detail=(
                    "Applied ArticulationRootAPI to the exact directed graph "
                    "component root."
                ),
            )
        )

    joint_diagnostics: list[JointDiagnosticV1] = []
    for joint in plan.joints:
        decisions: list[FieldDecisionV1] = []
        for field, provenance in joint.topology.field_provenance.items():
            decisions.append(
                FieldDecisionV1(
                    field=f"topology.{field}",
                    disposition="accepted",
                    provenance=provenance,
                    detail="Matched the already-authored joint topology.",
                )
            )
        limit = joint.limit
        for field in ("lower", "upper"):
            decisions.append(
                _planned_leaf_decision(
                    f"limit.{field}",
                    present=limit is not None and getattr(limit, field) is not None,
                    provenance=limit.provenance if limit is not None else None,
                    accepted_detail="Matched the exact authored scalar limit.",
                    absent_reason="not_planned",
                )
            )
        decisions.append(
            _planned_leaf_decision(
                "limit.unit",
                present=limit is not None,
                provenance=limit.provenance if limit is not None else None,
                accepted_detail="Matched the explicit limit unit.",
                absent_reason="not_planned",
            )
        )
        decisions.append(
            _planned_leaf_decision(
                "anchor.position_stage",
                present=joint.anchor is not None,
                provenance=(
                    joint.anchor.provenance if joint.anchor is not None else None
                ),
                accepted_detail="Matched the authored joint anchor frames.",
                absent_reason="not_planned",
            )
        )
        scalar_reason = (
            "not_applicable"
            if joint.topology.joint_type == "spherical"
            else "not_planned"
        )
        decisions.append(
            _planned_leaf_decision(
                "joint_friction.coefficient",
                present=joint.joint_friction is not None,
                provenance=(
                    joint.joint_friction.provenance
                    if joint.joint_friction is not None
                    else None
                ),
                accepted_detail="Applied the explicit PhysX joint friction.",
                absent_reason=scalar_reason,
            )
        )
        _append_object_leaf_decisions(
            decisions,
            prefix="state",
            value=joint.state,
            fields=("position", "velocity"),
            absent_reason=scalar_reason,
            accepted_detail="Applied the explicit default Joint State.",
        )
        _append_object_leaf_decisions(
            decisions,
            prefix="drive",
            value=joint.drive,
            fields=(
                "drive_type",
                "stiffness",
                "damping",
                "max_force",
                "target_position",
                "target_velocity",
                "max_joint_velocity",
            ),
            optional_fields=frozenset({"max_joint_velocity"}),
            absent_reason=scalar_reason,
            accepted_detail="Applied the complete motion-compatible drive.",
        )
        _append_object_leaf_decisions(
            decisions,
            prefix="mimic",
            value=joint.mimic,
            fields=(
                "reference_joint_id",
                "gearing",
                "offset",
                "natural_frequency",
                "damping_ratio",
            ),
            absent_reason=(
                "not_applicable"
                if joint.topology.joint_type != "revolute"
                else "not_planned"
            ),
            accepted_detail="Applied the explicit revolute mimic relationship.",
        )
        decisions.append(
            FieldDecisionV1(
                field="usd.joint_prim_path",
                disposition="defaulted",
                reason_code="deterministic_joint_path",
                detail=joint.topology.joint_id,
            )
        )
        reasons = tuple(
            decision.reason_code
            for decision in decisions
            if decision.reason_code is not None
        )
        joint_diagnostics.append(
            JointDiagnosticV1(
                joint_id=joint.topology.joint_id,
                field_decisions=tuple(decisions),
                reason_codes=reasons,
            )
        )
    return JointRiggerDiagnosticsV1(
        schema_version=DIAGNOSTICS_SCHEMA_VERSION,
        backend_name=backend_name,
        backend_version=backend_version,
        field_decisions=tuple(top_level),
        joint_diagnostics=tuple(joint_diagnostics),
    )


def _planned_leaf_decision(
    field: str,
    *,
    present: bool,
    provenance: Any | None,
    accepted_detail: str,
    absent_reason: str,
) -> FieldDecisionV1:
    if present:
        assert provenance is not None
        return FieldDecisionV1(
            field=field,
            disposition="accepted",
            provenance=provenance,
            detail=accepted_detail,
        )
    return FieldDecisionV1(
        field=field,
        disposition="ignored",
        reason_code=absent_reason,
        detail=(
            "The field is not applicable to this planned joint."
            if absent_reason == "not_applicable"
            else "No explicit evidence was present in the plan."
        ),
    )


def _append_object_leaf_decisions(
    decisions: list[FieldDecisionV1],
    *,
    prefix: str,
    value: Any | None,
    fields: tuple[str, ...],
    absent_reason: str,
    accepted_detail: str,
    optional_fields: frozenset[str] = frozenset(),
) -> None:
    for field in fields:
        present = value is not None and not (
            field in optional_fields and getattr(value, field) is None
        )
        decisions.append(
            _planned_leaf_decision(
                f"{prefix}.{field}",
                present=present,
                provenance=value.provenance if value is not None else None,
                accepted_detail=accepted_detail,
                absent_reason=absent_reason,
            )
        )


def _drive_specs(context: _JointContext) -> tuple[tuple[str, Any, str], ...]:
    drive = context.plan.drive
    assert drive is not None
    assert context.motion is not None
    namespace = f"drive:{context.motion}:physics"
    return (
        (f"{namespace}:type", drive.drive_type, "token"),
        (f"{namespace}:stiffness", drive.stiffness, "float"),
        (f"{namespace}:damping", drive.damping, "float"),
        (f"{namespace}:maxForce", drive.max_force, "float"),
        (f"{namespace}:targetPosition", drive.target_position, "float"),
        (f"{namespace}:targetVelocity", drive.target_velocity, "float"),
    )


def _mimic_specs(
    context: _JointContext,
    *,
    reference_axis: str,
) -> tuple[tuple[str, Any, str], ...]:
    mimic = context.plan.mimic
    assert mimic is not None
    assert context.axis_token is not None
    namespace = f"physxMimicJoint:rot{context.axis_token.upper()}"
    return (
        (f"{namespace}:referenceJointAxis", reference_axis, "token"),
        (f"{namespace}:gearing", mimic.gearing, "float"),
        (f"{namespace}:offset", mimic.offset, "float"),
        (f"{namespace}:naturalFrequency", mimic.natural_frequency, "float"),
        (f"{namespace}:dampingRatio", mimic.damping_ratio, "float"),
    )


def _mass_stage_values(
    mass: Any,
    *,
    meters_per_unit: float,
    kilograms_per_unit: float,
) -> tuple[
    float,
    tuple[float, float, float] | None,
    tuple[float, float, float],
]:
    mass_value = float(mass.mass_kg) / kilograms_per_unit
    inertia_divisor = kilograms_per_unit * meters_per_unit * meters_per_unit
    if not math.isfinite(inertia_divisor) or inertia_divisor <= 0.0:
        _fail(
            "invalid_stage_units",
            "stage unit metadata cannot represent finite positive inertia values",
        )
    inertia = tuple(
        float(value) / inertia_divisor for value in mass.diagonal_inertia_kg_m2
    )
    center_of_mass_m = getattr(mass, "center_of_mass_m", None)
    center_of_mass = (
        tuple(float(value) / meters_per_unit for value in center_of_mass_m)
        if center_of_mass_m is not None
        else None
    )
    if (
        not math.isfinite(mass_value)
        or mass_value <= 0.0
        or any(not math.isfinite(value) or value <= 0.0 for value in inertia)
        or (
            center_of_mass is not None
            and any(not math.isfinite(value) for value in center_of_mass)
        )
    ):
        _fail(
            "mass_unit_conversion_invalid",
            "SI mass/center/inertia cannot be represented in the stage unit metadata",
        )
    return (
        mass_value,
        (
            (center_of_mass[0], center_of_mass[1], center_of_mass[2])
            if center_of_mass is not None
            else None
        ),
        (inertia[0], inertia[1], inertia[2]),
    )


def _require_complete_zero_spanning_limits(context: _JointContext) -> None:
    limit = context.plan.limit
    if limit is None or limit.lower is None or limit.upper is None:
        _fail(
            "mimic_limits_incomplete",
            f"mimic requires complete finite limits at "
            f"{context.plan.topology.joint_id}",
        )
    assert limit is not None and limit.lower is not None and limit.upper is not None
    if limit.lower > 0.0 or limit.upper < 0.0 or limit.lower == limit.upper:
        _fail(
            "mimic_limits_incompatible",
            f"mimic limits must span the zero rest pose at "
            f"{context.plan.topology.joint_id}",
        )


def _require_position_inside_authored_limits(
    prim: Any,
    position: float,
    *,
    owner: str,
    code: str,
) -> None:
    lower_attr = prim.GetAttribute("physics:lowerLimit")
    upper_attr = prim.GetAttribute("physics:upperLimit")
    lower = (
        float(lower_attr.Get())
        if lower_attr and _has_authored_value(lower_attr, owner=owner)
        else None
    )
    upper = (
        float(upper_attr.Get())
        if upper_attr and _has_authored_value(upper_attr, owner=owner)
        else None
    )
    if lower is not None and position < lower - _VALUE_TOLERANCE:
        _fail(code, f"position {position} is below {lower} at {owner}")
    if upper is not None and position > upper + _VALUE_TOLERANCE:
        _fail(code, f"position {position} is above {upper} at {owner}")


def _require_target_prim(stage: Any, path: str, *, kind: str) -> Any:
    prim = stage.GetPrimAtPath(path)
    if (
        not prim
        or not prim.IsValid()
        or not prim.IsActive()
        or not prim.IsDefined()
        or prim.IsInstanceProxy()
    ):
        _fail(
            f"{kind.replace(' ', '_')}_unresolved",
            f"{kind} is not an active, defined, editable prim: {path}",
        )
    return prim


def _require_relationship_target(prim: Any, name: str, expected: str) -> None:
    relationship = prim.GetRelationship(name)
    targets = (
        tuple(str(path) for path in relationship.GetTargets()) if relationship else ()
    )
    if targets != (expected,):
        _fail(
            "joint_topology_mismatch",
            f"{name} at {prim.GetPath()} has targets {targets}, expected {(expected,)}",
        )


def _require_static_transform_chain(prim: Any) -> None:
    """Require static authored ancestry, including beyond resetXformStack.

    A reset permits the v1 nested-body layout; it does not widen the temporal
    eligibility contract to admit animated ancestors above that reset.
    """

    from pxr import UsdGeom

    target = str(prim.GetPath())
    current = prim
    while current.IsValid() and not current.IsPseudoRoot():
        xformable = UsdGeom.Xformable(current)
        if xformable:
            order_attribute = current.GetAttribute("xformOpOrder")
            if order_attribute:
                _require_unconnected_attribute(order_attribute, owner=target)
                order_samples = tuple(
                    float(value) for value in order_attribute.GetTimeSamples()
                )
                if order_samples:
                    _fail(
                        "time_varying_body_transform",
                        f"body {target} has xformOpOrder samples at "
                        f"{current.GetPath()}: {order_samples}",
                    )
            for op in xformable.GetOrderedXformOps():
                attribute = op.GetAttr()
                _require_unconnected_attribute(attribute, owner=target)
                samples = tuple(float(value) for value in attribute.GetTimeSamples())
                if samples:
                    _fail(
                        "time_varying_body_transform",
                        f"body {target} has transform samples at "
                        f"{current.GetPath()}: {samples}",
                    )
        current = current.GetParent()


def _preflight_attr(
    prim: Any,
    name: str,
    expected: Any,
    *,
    expected_type: str,
) -> None:
    if prim.GetRelationship(name):
        _fail(
            "physics_schema_conflict",
            f"owned Attribute name is already a Relationship at "
            f"{prim.GetPath()}: {name}",
        )
    attr = prim.GetAttribute(name)
    if not attr:
        return
    actual_type = str(attr.GetTypeName())
    property_stack = tuple(attr.GetPropertyStack())
    if any(not hasattr(spec, "typeName") for spec in property_stack):
        _fail(
            "physics_schema_conflict",
            f"{name} at {prim.GetPath()} has a non-AttributeSpec opinion",
        )
    authored_types = {
        str(spec.typeName) for spec in property_stack if str(spec.typeName)
    }
    if actual_type != expected_type or any(
        value != expected_type for value in authored_types
    ):
        _fail(
            "physics_schema_conflict",
            f"{name} at {prim.GetPath()} has type {actual_type} and authored "
            f"types {sorted(authored_types)}, expected {expected_type}",
        )
    if not _has_authored_value(attr, owner=str(prim.GetPath())):
        return
    _require_static(attr, owner=str(prim.GetPath()))
    if not _values_equal(attr.Get(), expected):
        _fail(
            "physics_schema_conflict",
            f"{name} at {prim.GetPath()} conflicts with explicit plan value",
        )


def _require_static(attr: Any, *, owner: str) -> None:
    _require_unconnected_attribute(attr, owner=owner)
    if attr.HasSpline():
        _fail(
            "time_sampled_owned_property",
            f"{owner} has spline-authored {attr.GetName()}",
        )
    samples = tuple(float(value) for value in attr.GetTimeSamples())
    if samples:
        _fail(
            "time_sampled_owned_property",
            f"{owner} has time-sampled {attr.GetName()}: {samples}",
        )


def _require_application(applied: Any, schema: str, prim: Any) -> None:
    if not applied:
        _fail(
            "physics_schema_apply_failed",
            f"could not apply {schema} at {prim.GetPath()}",
        )


def _require_schema(prim: Any, token: str) -> None:
    if token not in _applied_schema_tokens(prim):
        _fail(
            "postwrite_validation_failed",
            f"{prim.GetPath()} lacks required {token}",
        )


def _require_exact_attr(prim: Any, name: str, expected: Any) -> None:
    attr = prim.GetAttribute(name)
    if (
        not attr
        or not _has_authored_value(attr, owner=str(prim.GetPath()))
        or not _values_equal(attr.Get(), expected)
    ):
        _fail(
            "postwrite_validation_failed",
            f"{name} is absent or differs at {prim.GetPath()}",
        )


def _require_absent_authored_attr(prim: Any, name: str) -> None:
    attr = prim.GetAttribute(name)
    if attr and _has_authored_value(attr, owner=str(prim.GetPath())):
        _fail(
            "postwrite_validation_failed",
            f"{name} is unexpectedly authored at {prim.GetPath()}",
        )


def _require_schema_absent(prim: Any, token: str) -> None:
    if token in _applied_schema_tokens(prim):
        _fail(
            "postwrite_validation_failed",
            f"{prim.GetPath()} has unplanned {token}",
        )


def _has_authored_value(attr: Any, *, owner: str) -> bool:
    _require_unconnected_attribute(attr, owner=owner)
    return bool(attr.HasAuthoredValueOpinion())


def _require_unconnected_attribute(attr: Any, *, owner: str) -> None:
    if attr.HasAuthoredConnections():
        connections = tuple(str(path) for path in attr.GetConnections())
        _fail(
            "connected_owned_property",
            f"{owner} has authored connections on {attr.GetName()}: {connections}",
        )


def _applied_schema_tokens(prim: Any) -> set[str]:
    tokens = {str(token) for token in prim.GetAppliedSchemas()}
    metadata = prim.GetMetadata("apiSchemas")
    if metadata is not None:
        try:
            tokens.update(str(token) for token in metadata.GetAppliedItems())
        except AttributeError:
            if isinstance(metadata, list | tuple):
                tokens.update(str(token) for token in metadata)
    return tokens


def _normalized_vector(value: Any, *, owner: str) -> tuple[float, float, float]:
    vector = tuple(float(value[index]) for index in range(3))
    if any(not math.isfinite(component) for component in vector):
        _fail("axis_not_finite", f"axis contains non-finite values at {owner}")
    length = math.sqrt(sum(component * component for component in vector))
    if not math.isfinite(length) or math.isclose(length, 0.0, abs_tol=1e-12):
        _fail("axis_unresolved", f"axis cannot be normalized at {owner}")
    return tuple(component / length for component in vector)  # type: ignore[return-value]


def _dot(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return sum(first * second for first, second in zip(left, right, strict=True))


def _distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return math.sqrt(
        sum((first - second) ** 2 for first, second in zip(left, right, strict=True))
    )


def _values_equal(left: Any, right: Any) -> bool:
    left_value = _plain_value(left)
    right_value = _plain_value(right)
    if isinstance(left_value, tuple) and isinstance(right_value, tuple):
        return len(left_value) == len(right_value) and all(
            _values_equal(first, second)
            for first, second in zip(left_value, right_value, strict=True)
        )
    if isinstance(left_value, bool) or isinstance(right_value, bool):
        return left_value is right_value
    if isinstance(left_value, int | float) and isinstance(right_value, int | float):
        return math.isclose(
            float(left_value),
            float(right_value),
            rel_tol=_VALUE_TOLERANCE,
            abs_tol=_VALUE_TOLERANCE,
        )
    return bool(left_value == right_value)


def _stored_values_equal(left: Any, right: Any) -> bool:
    """Compare already-canonical USD storage values without numeric tolerance."""

    return _stored_plain_values_equal(_plain_value(left), _plain_value(right))


def _stored_plain_values_equal(left: Any, right: Any) -> bool:
    """Recursively compare canonical values, including the sign of zero."""

    if isinstance(left, tuple) or isinstance(right, tuple):
        if not isinstance(left, tuple) or not isinstance(right, tuple):
            return False
        return len(left) == len(right) and all(
            _stored_plain_values_equal(first, second)
            for first, second in zip(left, right, strict=True)
        )
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, int | float) and isinstance(right, int | float):
        left_float = float(left)
        right_float = float(right)
        if left_float == 0.0 and right_float == 0.0:
            return math.copysign(1.0, left_float) == math.copysign(
                1.0,
                right_float,
            )
        return bool(left == right)
    return bool(left == right)


def _plain_value(value: Any) -> Any:
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
        return tuple(_plain_value(item) for item in value)
    except TypeError:
        return str(value)


def _fail(code: str, detail: str) -> NoReturn:
    raise JointRiggerContractError(code, detail)


__all__ = [
    "author_physics_schemas",
    "validate_authored_physics_schemas",
    "validate_physics_plan_evidence",
]
