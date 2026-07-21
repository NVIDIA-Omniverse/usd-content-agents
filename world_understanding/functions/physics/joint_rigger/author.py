# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Owned, app-independent authoring of explicit Joint Rigger topology plans."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from world_understanding.functions.physics.joint_rigger.artifacts import (
    JointRiggerArtifactTargets,
    _route_cleanup_failures,
    validate_artifact_targets,
)
from world_understanding.functions.physics.joint_rigger.facade import (
    JointRiggerArtifactError,
    JointRiggerBackendIncompatibleError,
    author_joint_rig,
)
from world_understanding.functions.physics.joint_rigger.models import (
    DIAGNOSTICS_SCHEMA_VERSION,
    PLAN_SCHEMA_VERSION,
    RESULT_SCHEMA_VERSION,
    FieldDecisionV1,
    JointDiagnosticV1,
    JointPlanV1,
    JointRiggerContractError,
    JointRiggerDiagnosticsV1,
    JointRiggerInputV1,
    JointRiggerInputV2,
    JointRiggerPlanV1,
    JointRiggerPlanV2,
    JointRiggerResultV1,
    canonical_json,
    canonical_sha256,
)
from world_understanding.functions.physics.joint_rigger.reference import (
    identify_usd_artifact,
)
from world_understanding.functions.physics.joint_rigger.rigid_links import (
    author_aggregate_rigid_links,
    request_has_aggregate_links,
    validate_authored_rigid_links,
)
from world_understanding.functions.physics.joint_rigger.source_binding import (
    BoundInputDirectory,
    SealedSourceBinding,
    bound_input_dependency_snapshots,
    close_source_binding,
    copy_regular_file_to_new_path,
    create_sealed_source_binding,
    freeze_bound_projection_root,
    materialize_bound_input,
    remove_bound_input_directory,
    require_sealed_source_binding,
    restore_bound_projection_paths,
    write_new_text_file,
)
from world_understanding.functions.physics.joint_rigger.validation import (
    _TOPOLOGY_AUTHOR_VERSION,
    _preflight_topology_authoring,
    _TopologyPreflight,
    _validate_authored_preflight,
    _validate_no_reshape,
)

TOPOLOGY_AUTHOR_NAME = "owned_topology"
TOPOLOGY_AUTHOR_VERSION = _TOPOLOGY_AUTHOR_VERSION
_RAW_USD_EXTENSIONS = frozenset({".usd", ".usda", ".usdc"})


@dataclass(frozen=True)
class OwnedTopologyBackend:
    """Facade backend for one immutable local source USD and explicit plan."""

    source_usd_path: Path

    name: ClassVar[str] = TOPOLOGY_AUTHOR_NAME
    backend_name: ClassVar[str] = TOPOLOGY_AUTHOR_NAME
    backend_version: ClassVar[str] = TOPOLOGY_AUTHOR_VERSION
    # ``author`` repeats every probe check inside the same sealed projection it
    # uses for authoring.  The facade may therefore skip a redundant standalone
    # probe call while direct callers retain the full ``probe`` API.
    author_runs_probe_checks: ClassVar[bool] = True
    supports_joint_rigger_input_v2: ClassVar[bool] = True
    supports_aggregate_rigid_links: ClassVar[bool] = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_usd_path", Path(self.source_usd_path))

    def probe(self, request: JointRiggerInputV1) -> None:
        """Validate dependency, request shape, source identity, and full topology."""

        _validate_supported_request(request)
        _validate_raw_usd_path(self.source_usd_path, label="source USD")
        topology_plan = _topology_phase_plan(request.plan)
        has_aggregate_links = request_has_aggregate_links(request)
        with _bound_source_projection(
            self.source_usd_path,
            request,
            editable_root=isinstance(request, JointRiggerInputV2),
        ) as (binding, bound_source, _, _):
            stage = _open_stage(bound_source, label="bound source USD")
            try:
                if isinstance(request, JointRiggerInputV2):
                    if has_aggregate_links:
                        author_aggregate_rigid_links(stage, request)
                    else:
                        validate_authored_rigid_links(stage, request)
                    _author_v2_articulation_roots(stage, request)
                _preflight_topology_authoring(stage, topology_plan)
            finally:
                del stage
            require_sealed_source_binding(binding)

    def author(
        self,
        request: JointRiggerInputV1,
        artifact_targets: JointRiggerArtifactTargets,
    ) -> JointRiggerResultV1:
        """Copy the source root, author into staging, and emit bound v1 reports."""

        _validate_supported_request(request)
        _validate_raw_usd_path(self.source_usd_path, label="source USD")
        publication_output = artifact_targets.publication_output_path
        if publication_output is None:  # pragma: no cover - artifact invariant
            raise JointRiggerBackendIncompatibleError(
                "Facade omitted publication_output_path"
            )
        if (
            artifact_targets.sidecar_path is not None
            or artifact_targets.publication_sidecar_path is not None
        ):
            raise JointRiggerBackendIncompatibleError(
                "Owned WP-R2 topology authoring currently supports raw USD roots "
                "without a publication sidecar"
            )
        _validate_raw_usd_path(
            publication_output,
            label="publication output USD",
            require_file=False,
        )
        _validate_raw_usd_path(
            artifact_targets.output_path,
            label="physical output USD",
            require_file=False,
        )
        _validate_format_preserving_output(
            source_path=self.source_usd_path,
            publication_path=publication_output,
            physical_path=artifact_targets.output_path,
        )
        topology_plan = _topology_phase_plan(request.plan)
        with _bound_source_projection(
            self.source_usd_path,
            request,
            editable_root=True,
        ) as (binding, bound_source, bound_directory, restore_paths):
            _validate_publication_targets_against_binding(
                artifact_targets,
                source_path=self.source_usd_path,
                binding=binding,
            )
            _validate_dependency_relocation(
                source_path=self.source_usd_path,
                publication_path=publication_output,
                has_external_dependencies=bool(binding.dependencies),
            )
            stage = _open_stage(bound_source, label="bound editable source USD")
            try:
                if isinstance(request, JointRiggerInputV2):
                    author_aggregate_rigid_links(stage, request)
                    _author_v2_articulation_roots(stage, request)
                source_preflight = _preflight_topology_authoring(
                    stage,
                    topology_plan,
                )
                try:
                    diagnostics = _topology_diagnostics_for_request(
                        request,
                        _author_topology_stage(stage, topology_plan),
                    )
                except JointRiggerContractError as exc:
                    raise JointRiggerArtifactError(
                        "Owned topology authoring failed post-preflight validation: "
                        f"{exc.code}: {exc.detail}"
                    ) from exc
                if not stage.GetRootLayer().Save():
                    raise JointRiggerArtifactError(
                        "OpenUSD could not save the authored topology root layer"
                    )
            finally:
                del stage

            def validate_final_projection(validation_path: Path) -> None:
                verification_stage = _open_stage(
                    validation_path,
                    label="descriptor-pinned saved output USD",
                )
                try:
                    try:
                        if isinstance(request, JointRiggerInputV2):
                            validate_authored_rigid_links(
                                verification_stage,
                                request,
                            )
                            _validate_v2_articulation_roots(
                                verification_stage,
                                request,
                            )
                        _validate_authored_saved_stage(
                            verification_stage,
                            topology_plan,
                            diagnostics,
                            source_preflight=source_preflight,
                            normalize_layer_identifiers=True,
                            layer_identifier_remap=restore_paths,
                        )
                    except JointRiggerContractError as exc:
                        raise JointRiggerArtifactError(
                            "Final descriptor-pinned topology validation failed: "
                            f"{exc.code}: {exc.detail}"
                        ) from exc
                finally:
                    del verification_stage

            def restore_final_projection_paths(output_descriptor: int) -> None:
                restore_bound_projection_paths(
                    bound_source,
                    projection_root=bound_directory.path / "filesystem",
                    logical_output_parent=publication_output.parent,
                    restore_paths=restore_paths,
                    output_descriptor=output_descriptor,
                )

            with freeze_bound_projection_root(
                bound_source,
                validate_frozen_projection=validate_final_projection,
                prepare_before_freeze=restore_final_projection_paths,
            ) as frozen_source:
                require_sealed_source_binding(binding)
                copy_regular_file_to_new_path(
                    bound_source,
                    artifact_targets.output_path,
                    label="authored topology root",
                    frozen_source=frozen_source,
                )

            output_artifact = identify_usd_artifact(
                artifact_targets.output_path,
                uri=str(publication_output),
            )
            _require_source_identity(self.source_usd_path, request)
            result = JointRiggerResultV1(
                schema_version=RESULT_SCHEMA_VERSION,
                status="succeeded",
                input_sha256=canonical_sha256(request),
                plan_sha256=canonical_sha256(request.plan),
                output_artifact=output_artifact,
                diagnostics=diagnostics,
            )
            write_new_text_file(
                artifact_targets.diagnostics_path,
                canonical_json(diagnostics),
                label="Joint Rigger diagnostics",
            )
            write_new_text_file(
                artifact_targets.result_path,
                canonical_json(result),
                label="Joint Rigger result",
            )
            return result


def author_joint_topology(
    request: JointRiggerInputV1,
    *,
    source_usd_path: str | Path,
    artifact_targets: JointRiggerArtifactTargets,
) -> JointRiggerResultV1:
    """Author one explicit ready plan through the shared atomic facade.

    This owned backend supports V1 topology and V2 exact rigid-link membership
    (including aggregate namespace authoring) for raw USD roots. USDZ and root relocation
    with relative composition dependencies fail closed until shared packaging
    helpers can preserve those dependencies without importing an app package.
    """

    if not isinstance(request, JointRiggerInputV1):
        raise TypeError("request must be a JointRiggerInputV1")
    if not isinstance(artifact_targets, JointRiggerArtifactTargets):
        raise TypeError("artifact_targets must be JointRiggerArtifactTargets")
    source_path = Path(source_usd_path)
    return author_joint_rig(
        request,
        OwnedTopologyBackend(source_path),
        artifact_targets,
    )


def _topology_phase_plan(
    plan: JointRiggerPlanV1 | JointRiggerPlanV2,
) -> JointRiggerPlanV1:
    """Project a V1/V2 request onto fields owned by topology-only authoring."""

    if isinstance(plan, JointRiggerPlanV1):
        # V1 is already the exact plan consumed by this backend. Returning it
        # intact preserves the pre-V2 contract, including source-backed drive,
        # friction, state, mimic, rigid-body, and articulation-root opinions.
        return plan
    return JointRiggerPlanV1(
        schema_version=PLAN_SCHEMA_VERSION,
        # V2 adds link/root structure around the same JointPlanV1 records. Keep
        # every joint field so topology preflight still accepts or rejects the
        # request exactly as V1 did instead of silently stripping opinions.
        joints=plan.joints,
    )


def _topology_diagnostics_for_request(
    request: JointRiggerInputV1,
    diagnostics: JointRiggerDiagnosticsV1,
) -> JointRiggerDiagnosticsV1:
    """Bind V2 articulation-root dispositions to the topology result."""

    if not isinstance(request, JointRiggerInputV2):
        return diagnostics
    decisions = tuple(
        decision
        for decision in diagnostics.field_decisions
        if decision.field != "articulation_root"
    )
    if request.plan.articulation_roots:
        decisions += tuple(
            FieldDecisionV1(
                field=f"articulation_roots[{root.prim_path}]",
                disposition="accepted",
                provenance=root.provenance,
                detail=(
                    "Applied ArticulationRootAPI to the exact directed graph "
                    "component root."
                ),
            )
            for root in request.plan.articulation_roots
        )
    else:
        decisions += (
            FieldDecisionV1(
                field="articulation_roots",
                disposition="ignored",
                reason_code="not_provided",
            ),
        )
    return JointRiggerDiagnosticsV1(
        schema_version=diagnostics.schema_version,
        backend_name=diagnostics.backend_name,
        backend_version=diagnostics.backend_version,
        field_decisions=decisions,
        joint_diagnostics=diagnostics.joint_diagnostics,
        errors=diagnostics.errors,
        warnings=diagnostics.warnings,
    )


def _author_v2_articulation_roots(stage: Any, request: JointRiggerInputV2) -> None:
    """Apply only the exact V2 articulation-root schemas on a private stage."""

    from pxr import UsdPhysics

    _validate_v2_articulation_roots(stage, request, allow_missing=True)
    for root in request.plan.articulation_roots:
        prim = stage.GetPrimAtPath(root.prim_path)
        api = UsdPhysics.ArticulationRootAPI.Apply(prim)
        if not api:
            raise JointRiggerArtifactError(
                f"Could not apply ArticulationRootAPI to {root.prim_path}"
            )
    _validate_v2_articulation_roots(stage, request)


def _validate_v2_articulation_roots(
    stage: Any,
    request: JointRiggerInputV2,
    *,
    allow_missing: bool = False,
) -> None:
    """Require the saved stage to expose exactly the planned V2 roots."""

    from pxr import Sdf, Usd

    from world_understanding.functions.physics.joint_rigger.schemas import (
        _existing_articulation_root_paths,
    )

    expected = {root.prim_path for root in request.plan.articulation_roots}
    observed = _existing_articulation_root_paths(stage, Sdf=Sdf, Usd=Usd)
    extras = observed - expected
    missing = expected - observed
    if extras or (missing and not allow_missing):
        raise JointRiggerContractError(
            "authored_articulation_roots_mismatch",
            "authored articulation roots do not exactly match the V2 plan: "
            f"missing={sorted(missing)}, extra={sorted(extras)}",
        )


def _author_topology_stage(
    stage: Any,
    plan: JointRiggerPlanV1,
) -> JointRiggerDiagnosticsV1:
    preflight = _preflight_topology_authoring(stage, plan)
    diagnostics = _build_diagnostics(plan, preflight)
    try:
        _author_preflight(stage, preflight, diagnostics)
        _validate_authored_preflight(stage, preflight, diagnostics=diagnostics)
        _validate_no_reshape(stage, preflight)
    except BaseException as exc:
        rollback_exc: BaseException | None = None
        try:
            _rollback_preflight(stage, preflight)
            _validate_no_reshape(stage, preflight)
        except BaseException as caught_rollback_exc:
            rollback_exc = caught_rollback_exc
        if rollback_exc is not None:
            if not isinstance(exc, Exception):
                _route_cleanup_failures(
                    [("Owned topology rollback also failed", rollback_exc)],
                    primary_error=exc,
                    label="Owned topology rollback failed",
                )
                raise
            if not isinstance(rollback_exc, Exception):
                _route_cleanup_failures(
                    [("Owned topology authoring also failed", exc)],
                    primary_error=rollback_exc,
                    label="Owned topology authoring failed",
                )
                raise rollback_exc from exc
            failures = ExceptionGroup(
                "Owned topology authoring and rollback both failed",
                [exc, rollback_exc],
            )
            raise JointRiggerArtifactError(
                "Owned topology authoring could not restore the pre-authoring "
                f"stage after {type(exc).__name__}: {exc}; rollback failed: "
                f"{type(rollback_exc).__name__}: {rollback_exc}"
            ) from failures
        raise
    return diagnostics


def _author_preflight(
    stage: Any,
    preflight: _TopologyPreflight,
    diagnostics: JointRiggerDiagnosticsV1,
) -> None:
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    if preflight.create_joints_scope:
        scope = UsdGeom.Scope.Define(stage, Sdf.Path(preflight.joints_scope_path))
        if not scope or not scope.GetPrim().IsValid():
            raise JointRiggerArtifactError(
                f"Could not define joint scope {preflight.joints_scope_path}"
            )
    schemas = {
        "revolute": UsdPhysics.RevoluteJoint,
        "prismatic": UsdPhysics.PrismaticJoint,
        "spherical": UsdPhysics.SphericalJoint,
    }
    diagnostics_by_id = {item.joint_id: item for item in diagnostics.joint_diagnostics}
    for prepared in preflight.joints:
        topology = prepared.source.topology
        joint = schemas[topology.joint_type].Define(
            stage,
            Sdf.Path(prepared.joint_path),
        )
        prim = joint.GetPrim()
        if not prim or not prim.IsValid():
            raise JointRiggerArtifactError(
                f"Could not define {topology.joint_type} joint {prepared.joint_path}"
            )
        _require_set(
            joint.CreateBody0Rel().SetTargets([Sdf.Path(topology.body0)]),
            f"{prepared.joint_path} body0",
        )
        _require_set(
            joint.CreateBody1Rel().SetTargets([Sdf.Path(topology.body1)]),
            f"{prepared.joint_path} body1",
        )
        _require_set(
            joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*prepared.local_pos0)),
            f"{prepared.joint_path} localPos0",
        )
        _require_set(
            joint.CreateLocalPos1Attr().Set(Gf.Vec3f(*prepared.local_pos1)),
            f"{prepared.joint_path} localPos1",
        )
        if prepared.axis_token is not None:
            if prepared.local_rot0 is None or prepared.local_rot1 is None:
                raise JointRiggerArtifactError(
                    f"Preflight omitted local axis frames for {prepared.joint_path}"
                )
            _require_set(
                joint.CreateAxisAttr().Set(prepared.axis_token),
                f"{prepared.joint_path} axis",
            )
            _require_set(
                joint.CreateLocalRot0Attr().Set(_quatf(prepared.local_rot0, Gf=Gf)),
                f"{prepared.joint_path} localRot0",
            )
            _require_set(
                joint.CreateLocalRot1Attr().Set(_quatf(prepared.local_rot1, Gf=Gf)),
                f"{prepared.joint_path} localRot1",
            )
        if prepared.lower_limit is not None:
            _require_set(
                joint.CreateLowerLimitAttr().Set(float(prepared.lower_limit)),
                f"{prepared.joint_path} lowerLimit",
            )
        if prepared.upper_limit is not None:
            _require_set(
                joint.CreateUpperLimitAttr().Set(float(prepared.upper_limit)),
                f"{prepared.joint_path} upperLimit",
            )
        _author_drive(prim, prepared, UsdPhysics=UsdPhysics)
        _author_physx_joint(prim, prepared, Sdf=Sdf)

        diagnostic = diagnostics_by_id[topology.joint_id]
        decisions_payload = json.dumps(
            [
                item.model_dump(mode="json", exclude_none=True)
                for item in diagnostic.field_decisions
            ],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        _require_set(
            prim.SetCustomDataByKey("jointRigger:jointId", topology.joint_id),
            f"{prepared.joint_path} joint id customData",
        )
        _require_set(
            prim.SetCustomDataByKey(
                "jointRigger:planSha256",
                preflight.plan_sha256,
            ),
            f"{prepared.joint_path} plan identity customData",
        )
        _require_set(
            prim.SetCustomDataByKey(
                "jointRigger:authoringVersion",
                TOPOLOGY_AUTHOR_VERSION,
            ),
            f"{prepared.joint_path} version customData",
        )
        _require_set(
            prim.SetCustomDataByKey("jointRigger:fieldDecisions", decisions_payload),
            f"{prepared.joint_path} field decisions customData",
        )


def _author_drive(prim: Any, prepared: Any, *, UsdPhysics: Any) -> None:
    drive_plan = prepared.source.drive
    if drive_plan is None:
        return
    drive = UsdPhysics.DriveAPI.Apply(prim, prepared.drive_instance)
    if not drive:
        raise JointRiggerArtifactError(
            f"Could not apply drive schema to {prepared.joint_path}"
        )
    attributes = (
        (drive.CreateTypeAttr(), drive_plan.drive_type, "type"),
        (drive.CreateStiffnessAttr(), drive_plan.stiffness, "stiffness"),
        (drive.CreateDampingAttr(), drive_plan.damping, "damping"),
        (drive.CreateMaxForceAttr(), drive_plan.max_force, "maxForce"),
        (
            drive.CreateTargetPositionAttr(),
            drive_plan.target_position,
            "targetPosition",
        ),
        (
            drive.CreateTargetVelocityAttr(),
            drive_plan.target_velocity,
            "targetVelocity",
        ),
    )
    for attribute, value, label in attributes:
        _require_set(
            attribute.Set(value),
            f"{prepared.joint_path} drive {label}",
        )


def _author_physx_joint(prim: Any, prepared: Any, *, Sdf: Any) -> None:
    drive_plan = prepared.source.drive
    friction_plan = prepared.source.joint_friction
    max_joint_velocity = (
        drive_plan.max_joint_velocity if drive_plan is not None else None
    )
    if max_joint_velocity is None and friction_plan is None:
        return
    _require_set(
        prim.AddAppliedSchema("PhysxJointAPI"),
        f"{prepared.joint_path} PhysxJointAPI",
    )
    if max_joint_velocity is not None:
        attribute = prim.CreateAttribute(
            "physxJoint:maxJointVelocity",
            Sdf.ValueTypeNames.Float,
            custom=False,
        )
        _require_set(
            attribute.Set(float(max_joint_velocity)),
            f"{prepared.joint_path} maxJointVelocity",
        )
    if friction_plan is not None:
        attribute = prim.CreateAttribute(
            "physxJoint:jointFriction",
            Sdf.ValueTypeNames.Float,
            custom=False,
        )
        _require_set(
            attribute.Set(float(friction_plan.coefficient)),
            f"{prepared.joint_path} jointFriction",
        )


def _build_diagnostics(
    plan: JointRiggerPlanV1,
    preflight: _TopologyPreflight,
) -> JointRiggerDiagnosticsV1:
    joint_diagnostics = tuple(
        _joint_diagnostic(prepared) for prepared in preflight.joints
    )
    return JointRiggerDiagnosticsV1(
        schema_version=DIAGNOSTICS_SCHEMA_VERSION,
        backend_name=TOPOLOGY_AUTHOR_NAME,
        backend_version=TOPOLOGY_AUTHOR_VERSION,
        field_decisions=(
            FieldDecisionV1(
                field="legacy_component_names",
                disposition="ignored",
                reason_code="legacy_component_name_compatibility_not_requested",
                detail="Owned topology authoring consumed only the structured plan.",
            ),
            FieldDecisionV1(
                field="articulation_root",
                disposition="ignored",
                reason_code="not_provided",
                detail="WP-R2 authors no articulation-root schema.",
            ),
            FieldDecisionV1(
                field="rigid_bodies",
                disposition="ignored",
                reason_code="not_provided",
                detail="WP-R2 authors no rigid-body, mass, or collider schemas.",
            ),
        ),
        joint_diagnostics=joint_diagnostics,
    )


def _joint_diagnostic(prepared: Any) -> JointDiagnosticV1:
    joint = prepared.source
    topology = joint.topology
    decisions: list[FieldDecisionV1] = [
        FieldDecisionV1(
            field=f"topology.{field}",
            disposition="accepted",
            provenance=topology.field_provenance[field],
        )
        for field in ("joint_type", "body0", "body1")
    ]
    decisions.append(
        FieldDecisionV1(
            field="topology.axis_stage",
            disposition=("accepted" if topology.axis_stage is not None else "ignored"),
            provenance=(
                topology.field_provenance["axis_stage"]
                if topology.axis_stage is not None
                else None
            ),
            reason_code=(
                None if topology.axis_stage is not None else "not_applicable_spherical"
            ),
            detail=(
                "Spherical joints do not author a scalar axis."
                if topology.axis_stage is None
                else ""
            ),
        )
    )
    if joint.limit is None:
        for field in ("lower", "upper", "unit"):
            decisions.append(
                FieldDecisionV1(
                    field=f"limit.{field}",
                    disposition="ignored",
                    reason_code="not_provided",
                    detail="No source-backed scalar limit was supplied.",
                )
            )
    else:
        for field, value in (
            ("lower", joint.limit.lower),
            ("upper", joint.limit.upper),
        ):
            decisions.append(
                FieldDecisionV1(
                    field=f"limit.{field}",
                    disposition="accepted" if value is not None else "ignored",
                    provenance=joint.limit.provenance if value is not None else None,
                    reason_code=None if value is not None else "not_provided",
                )
            )
        decisions.append(
            FieldDecisionV1(
                field="limit.unit",
                disposition="accepted",
                provenance=joint.limit.provenance,
            )
        )
    decisions.append(
        FieldDecisionV1(
            field="anchor.position_stage",
            disposition="accepted" if joint.anchor is not None else "defaulted",
            provenance=joint.anchor.provenance if joint.anchor is not None else None,
            reason_code=(
                None if joint.anchor is not None else "inferred_body1_world_origin"
            ),
            detail=(
                "No source-backed anchor was supplied; the shared anchor uses "
                f"body1's world origin {prepared.anchor_stage!r}."
                if joint.anchor is None
                else ""
            ),
        )
    )
    decisions.extend(_drive_decisions(joint))
    friction = joint.joint_friction
    decisions.append(
        FieldDecisionV1(
            field="joint_friction.coefficient",
            disposition="accepted" if friction is not None else "ignored",
            provenance=friction.provenance if friction is not None else None,
            reason_code=None if friction is not None else "not_planned",
            detail=(
                "Authored the explicit PhysX joint-friction coefficient."
                if friction is not None
                else "No source-backed joint friction was supplied."
            ),
        )
    )
    decisions.extend(
        (
            FieldDecisionV1(
                field="state",
                disposition="ignored",
                reason_code="outside_wp_r2_scope",
                detail="Joint state belongs to WP-R3 and was not provided.",
            ),
            FieldDecisionV1(
                field="mimic",
                disposition="ignored",
                reason_code="outside_wp_r2_scope",
                detail="Mimic schemas belong to WP-R3 and were not provided.",
            ),
            FieldDecisionV1(
                field="usd.joint_prim_path",
                disposition="defaulted",
                reason_code="deterministic_joint_path",
                detail=prepared.joint_path,
            ),
            FieldDecisionV1(
                field="usd.local_frames",
                disposition="defaulted",
                reason_code="derived_from_stage_axis_and_anchor",
                detail="Local positions and rotations were derived without reshaping.",
            ),
        )
    )
    reason_codes = tuple(
        decision.reason_code
        for decision in decisions
        if decision.reason_code is not None
    )
    return JointDiagnosticV1(
        joint_id=topology.joint_id,
        field_decisions=tuple(decisions),
        reason_codes=reason_codes,
    )


def _drive_decisions(joint: JointPlanV1) -> tuple[FieldDecisionV1, ...]:
    if joint.drive is None:
        return tuple(
            FieldDecisionV1(
                field=f"drive.{field}",
                disposition="ignored",
                reason_code="not_provided",
                detail="No source-backed drive was supplied.",
            )
            for field in (
                "drive_type",
                "stiffness",
                "damping",
                "max_force",
                "target_position",
                "target_velocity",
                "max_joint_velocity",
            )
        )
    drive = joint.drive
    decisions = [
        FieldDecisionV1(
            field=f"drive.{field}",
            disposition="accepted",
            provenance=drive.provenance,
        )
        for field in (
            "drive_type",
            "stiffness",
            "damping",
            "max_force",
            "target_position",
            "target_velocity",
        )
    ]
    decisions.append(
        FieldDecisionV1(
            field="drive.max_joint_velocity",
            disposition=(
                "accepted" if drive.max_joint_velocity is not None else "ignored"
            ),
            provenance=(
                drive.provenance if drive.max_joint_velocity is not None else None
            ),
            reason_code=(
                None if drive.max_joint_velocity is not None else "not_provided"
            ),
        )
    )
    return tuple(decisions)


def _validate_authored_saved_stage(
    stage: Any,
    plan: JointRiggerPlanV1,
    diagnostics: JointRiggerDiagnosticsV1,
    *,
    source_preflight: _TopologyPreflight,
    normalize_layer_identifiers: bool = False,
    layer_identifier_remap: Mapping[Path, Path] | None = None,
) -> None:
    authored_preflight = _preflight_topology_authoring(
        stage,
        plan,
        allow_existing_joint_paths=True,
    )
    _validate_authored_preflight(
        stage,
        authored_preflight,
        diagnostics=diagnostics,
    )
    _validate_no_reshape(
        stage,
        source_preflight,
        normalize_layer_identifiers=normalize_layer_identifiers,
        layer_identifier_remap=layer_identifier_remap,
    )


def _rollback_preflight(stage: Any, preflight: _TopologyPreflight) -> None:
    from pxr import Sdf

    rollback_paths = [prepared.joint_path for prepared in reversed(preflight.joints)]
    if preflight.create_joints_scope:
        rollback_paths.append(preflight.joints_scope_path)
    removal_failures: list[str] = []
    edit_layer = stage.GetEditTarget().GetLayer()
    for path_value in rollback_paths:
        path = Sdf.Path(path_value)
        prim = stage.GetPrimAtPath(path)
        prim_spec = edit_layer.GetPrimAtPath(path)
        if (not prim or not prim.IsValid()) and prim_spec is None:
            continue
        try:
            removed = stage.RemovePrim(path)
        except Exception as exc:
            removal_failures.append(f"{path_value}: {type(exc).__name__}: {exc}")
            continue
        if not removed:
            removal_failures.append(f"{path_value}: RemovePrim reported failure")

    remaining_paths = []
    for path_value in rollback_paths:
        path = Sdf.Path(path_value)
        prim = stage.GetPrimAtPath(path)
        if (prim and prim.IsValid()) or edit_layer.GetPrimAtPath(path) is not None:
            remaining_paths.append(path_value)
    if removal_failures or remaining_paths:
        details = []
        if removal_failures:
            details.append(f"removal failures={removal_failures}")
        if remaining_paths:
            details.append(f"authored paths remain={remaining_paths}")
        raise JointRiggerArtifactError(
            "Owned topology rollback could not remove every authored prim: "
            + "; ".join(details)
        )


def _validate_supported_request(request: JointRiggerInputV1) -> None:
    if not isinstance(request, JointRiggerInputV1):
        raise TypeError("request must be a JointRiggerInputV1")
    if request.legacy_component_names is not None:
        raise JointRiggerBackendIncompatibleError(
            "Owned topology authoring never consumes legacy component_name input"
        )


@contextmanager
def _bound_source_projection(
    source_path: Path,
    request: JointRiggerInputV1,
    *,
    editable_root: bool,
) -> Iterator[tuple[SealedSourceBinding, Path, BoundInputDirectory, dict[Path, Path]]]:
    """Yield one descriptor-derived source closure and clean every resource."""

    binding: SealedSourceBinding | None = None
    bound_directory: BoundInputDirectory | None = None
    primary_error: BaseException | None = None
    try:
        binding = create_sealed_source_binding(
            source_path,
            expected=request.source_asset,
        )
        bound_source, bound_directory, restore_paths = materialize_bound_input(
            descriptor=binding.descriptor,
            expected_sha256=binding.sha256,
            logical_input_path=source_path,
            dependencies=bound_input_dependency_snapshots(binding),
            editable_root=editable_root,
        )
        yield binding, bound_source, bound_directory, restore_paths
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        cleanup_errors: list[tuple[str, BaseException]] = []
        if bound_directory is not None:
            try:
                remove_bound_input_directory(bound_directory)
            except BaseException as exc:
                cleanup_errors.append(("Bound projection cleanup failed", exc))
        if binding is not None:
            try:
                cleanup_errors.extend(
                    ("Bound source descriptor cleanup failed", error)
                    for error in close_source_binding(binding)
                )
            except BaseException as exc:
                cleanup_errors.append(("Bound source descriptor cleanup failed", exc))
        _route_cleanup_failures(
            cleanup_errors,
            primary_error=primary_error,
            label="Bound source cleanup failed",
        )


def _validate_publication_targets_against_binding(
    artifact_targets: JointRiggerArtifactTargets,
    *,
    source_path: Path,
    binding: SealedSourceBinding,
) -> None:
    """Protect the exact bound source closure from final publication targets.

    ``validate_artifact_targets`` accepts caller-facing layouts, whose physical
    and publication coordinates are identical.  Backend staging targets
    intentionally differ, so this guard constructs the caller-facing
    publication view before applying the shared overlap validation.
    """

    publication_output = artifact_targets.publication_output_path
    publication_diagnostics = artifact_targets.publication_diagnostics_path
    publication_result = artifact_targets.publication_result_path
    if (
        publication_output is None
        or publication_diagnostics is None
        or publication_result is None
    ):  # pragma: no cover - normalized target invariant
        raise JointRiggerBackendIncompatibleError(
            "Facade omitted a required publication artifact path"
        )
    require_sealed_source_binding(binding)
    read_paths: list[tuple[str, Path]] = [
        ("bound source USD", source_path),
        ("captured resolved bound source USD", binding.path),
        (
            "current resolved bound source USD",
            source_path.expanduser().resolve(strict=False),
        ),
    ]
    for index, dependency in enumerate(binding.dependencies):
        label = f"bound source USD dependency[{index}]"
        read_paths.append((label, dependency.path))
        read_paths.extend((label, path) for path in dependency.projection_paths)
    publication_targets = JointRiggerArtifactTargets(
        output_path=publication_output,
        diagnostics_path=publication_diagnostics,
        result_path=publication_result,
        sidecar_path=artifact_targets.publication_sidecar_path,
    )
    try:
        validate_artifact_targets(publication_targets, read_paths=read_paths)
    except (OSError, ValueError) as exc:
        raise JointRiggerArtifactError(
            f"Owned topology publication targets overlap bound inputs: {exc}"
        ) from exc


def _validate_raw_usd_path(
    path: Path,
    *,
    label: str,
    require_file: bool = True,
) -> None:
    if path.suffix.lower() not in _RAW_USD_EXTENSIONS:
        raise JointRiggerBackendIncompatibleError(
            f"{label} raw USD must use .usd, .usda, or .usdc for WP-R2; got {path}"
        )
    if require_file and (path.is_symlink() or not path.is_file()):
        raise JointRiggerArtifactError(f"{label} must be a regular file: {path}")


def _validate_dependency_relocation(
    *,
    source_path: Path,
    publication_path: Path,
    has_external_dependencies: bool,
) -> None:
    if has_external_dependencies and (
        source_path.parent.expanduser().resolve(strict=False)
        != publication_path.parent.expanduser().resolve(strict=False)
    ):
        raise JointRiggerBackendIncompatibleError(
            "WP-R2 cannot relocate a raw USD root with composition dependencies; "
            "publish beside the source or use a future shared packaging adapter"
        )


def _validate_format_preserving_output(
    *,
    source_path: Path,
    publication_path: Path,
    physical_path: Path,
) -> None:
    source_suffix = source_path.suffix.lower()
    publication_suffix = publication_path.suffix.lower()
    physical_suffix = physical_path.suffix.lower()
    if publication_suffix != source_suffix or physical_suffix != source_suffix:
        raise JointRiggerBackendIncompatibleError(
            "WP-R2 raw USD authoring preserves the source layer format and requires "
            "matching source and output suffixes; explicit USD format conversion is "
            "not supported: "
            f"source={source_suffix}, publication={publication_suffix}, "
            f"physical={physical_suffix}"
        )


def _require_source_identity(path: Path, request: JointRiggerInputV1) -> None:
    observed = identify_usd_artifact(path, uri=request.source_asset.uri)
    if observed != request.source_asset:
        raise JointRiggerArtifactError(
            "Source USD or its dependency closure does not match source_asset"
        )


def _open_stage(path: Path, *, label: str) -> Any:
    try:
        from pxr import Usd
    except ImportError as exc:
        raise JointRiggerContractError(
            "runtime_dependency_unavailable",
            f"OpenUSD pxr bindings are required for topology authoring: {exc}",
        ) from exc
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise JointRiggerArtifactError(f"Could not open {label}: {path}")
    return stage


def _quatf(value: Any, *, Gf: Any) -> Any:
    real, imaginary = value
    return Gf.Quatf(float(real), Gf.Vec3f(*imaginary))


def _require_set(result: Any, label: str) -> None:
    if result is False:
        raise JointRiggerArtifactError(f"Could not author {label}")


__all__ = [
    "TOPOLOGY_AUTHOR_NAME",
    "TOPOLOGY_AUTHOR_VERSION",
    "OwnedTopologyBackend",
    "author_joint_topology",
]
