# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Atomic R2 topology plus R3 physics-schema artifact coverage."""

from __future__ import annotations

import hashlib
import importlib
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

import pytest
from pxr import Gf, Sdf, Ts, Usd, UsdGeom, UsdPhysics

from world_understanding.functions.physics.joint_rigger import (
    DIAGNOSTICS_SCHEMA_VERSION,
    INPUT_SCHEMA_VERSION,
    INPUT_SCHEMA_VERSION_V2,
    PLAN_SCHEMA_VERSION,
    PLAN_SCHEMA_VERSION_V2,
    ArticulationRootPlanV1,
    ColliderPlanV1,
    FieldDecisionV1,
    FieldProvenanceV1,
    JointDiagnosticV1,
    JointDriveV1,
    JointFrictionV1,
    JointLimitV1,
    JointMimicV1,
    JointPlanV1,
    JointRiggerArtifactError,
    JointRiggerArtifactTargets,
    JointRiggerContractError,
    JointRiggerDiagnosticsV1,
    JointRiggerInputV1,
    JointRiggerInputV2,
    JointRiggerPlanV1,
    JointRiggerPlanV2,
    JointStateV1,
    JointTopologyV1,
    MassPropertiesV1,
    OwnedTopologyAndPhysicsBackend,
    RigidBodyPlanV1,
    RigidLinkMemberPlanV1,
    RigidLinkPlanV1,
    author_joint_rig_with_physics,
    author_joint_topology,
    author_physics_schemas,
    canonical_sha256,
    identify_usd_artifact,
    validate_authored_joint_rig_with_physics,
    validate_authored_physics_schemas,
)

combined_module = importlib.import_module(
    "world_understanding.functions.physics.joint_rigger.combined"
)
artifacts_module = importlib.import_module(
    "world_understanding.functions.physics.joint_rigger.artifacts"
)
schemas_module = importlib.import_module(
    "world_understanding.functions.physics.joint_rigger.schemas"
)
validation_module = importlib.import_module(
    "world_understanding.functions.physics.joint_rigger.validation"
)
rigid_links_module = importlib.import_module(
    "world_understanding.functions.physics.joint_rigger.rigid_links"
)


def _v2_plan_from_v1(plan: JointRiggerPlanV1) -> JointRiggerPlanV2:
    roots = () if plan.articulation_root is None else (plan.articulation_root,)
    return JointRiggerPlanV2(
        schema_version=PLAN_SCHEMA_VERSION_V2,
        joints=plan.joints,
        rigid_bodies=plan.rigid_bodies,
        articulation_roots=roots,
    )


def _v2_topology_phase_plan(plan: JointRiggerPlanV2) -> JointRiggerPlanV2:
    topology_plan = combined_module._topology_phase_plan(plan)
    return JointRiggerPlanV2(
        schema_version=PLAN_SCHEMA_VERSION_V2,
        joints=topology_plan.joints,
        articulation_roots=(),
    )


def test_combined_author_identity_covers_the_final_schema_artifact(
    tmp_path: Path,
) -> None:
    source, request = _fixture(tmp_path, joint_friction=0.25)

    topology_targets = _targets(tmp_path, "topology-only")
    topology_request = JointRiggerInputV1(
        schema_version=INPUT_SCHEMA_VERSION,
        source_asset=request.source_asset,
        plan=combined_module._topology_phase_plan(request.plan),
        conflict_policy="error",
    )
    topology_result = author_joint_topology(
        topology_request,
        source_usd_path=source,
        artifact_targets=topology_targets,
    )
    topology_identity = topology_result.output_artifact
    assert topology_identity is not None
    assert topology_identity.root_sha256 == _sha256(topology_targets.output_path)

    # A separately published R2 root has already been identified. Mutating that
    # root with R3 necessarily makes the earlier result stale.
    resolved_plan = combined_module._resolve_physics_plan(
        request.plan,
        topology_result.diagnostics,
    )
    topology_targets.output_path.chmod(0o600)
    topology_stage = Usd.Stage.Open(str(topology_targets.output_path))
    assert topology_stage is not None
    author_physics_schemas(topology_stage, resolved_plan)
    assert topology_stage.GetRootLayer().Save()
    del topology_stage
    assert topology_identity.root_sha256 != _sha256(topology_targets.output_path)

    combined_targets = _targets(tmp_path, "combined")
    result = author_joint_rig_with_physics(
        request,
        source_usd_path=source,
        artifact_targets=combined_targets,
    )

    assert result.input_sha256 == canonical_sha256(request)
    assert result.plan_sha256 == canonical_sha256(request.plan)
    assert result.output_artifact == identify_usd_artifact(
        combined_targets.output_path,
        uri=str(combined_targets.output_path),
    )
    assert result.output_artifact.root_sha256 == _sha256(combined_targets.output_path)
    assert result.diagnostics.backend_name == "owned_topology_and_physics"
    assert result.diagnostics.backend_version == (
        "world-understanding-joint-rig-author-v1"
    )
    assert [item.joint_id for item in result.diagnostics.joint_diagnostics] == ["hinge"]
    combined_joint = result.diagnostics.joint_diagnostics[0]
    authored_path = combined_joint.authored_prim_path
    assert authored_path is not None
    assert authored_path.startswith("/World/Joints/hinge_")
    joint_decisions = {item.field: item for item in combined_joint.field_decisions}
    assert not {"state", "drive", "mimic"}.intersection(joint_decisions)
    assert joint_decisions["state.position"].disposition == "accepted"
    assert request.plan.joints[0].state is not None
    assert joint_decisions["state.position"].provenance == (
        request.plan.joints[0].state.provenance
    )
    assert joint_decisions["joint_friction.coefficient"].disposition == "accepted"
    assert request.plan.joints[0].joint_friction is not None
    assert joint_decisions["joint_friction.coefficient"].provenance == (
        request.plan.joints[0].joint_friction.provenance
    )
    assert joint_decisions["drive.drive_type"].disposition == "ignored"
    assert joint_decisions["mimic.reference_joint_id"].disposition == "ignored"
    top_level = {item.field: item for item in result.diagnostics.field_decisions}
    assert "rigid_bodies[/World/base].mass.mass_kg" in top_level
    assert "rigid_bodies[/World/base].mass_inertia" not in top_level
    assert request.plan.rigid_bodies[0].mass is not None
    assert (
        top_level["rigid_bodies[/World/base].mass.mass_kg"].provenance
        == request.plan.rigid_bodies[0].mass.provenance
    )

    final_stage = Usd.Stage.Open(str(combined_targets.output_path))
    assert final_stage is not None
    final_resolved_plan = combined_module._resolve_physics_plan(
        request.plan,
        result.diagnostics,
    )
    validate_authored_physics_schemas(
        final_stage,
        final_resolved_plan,
        backend_name=result.diagnostics.backend_name,
        backend_version=result.diagnostics.backend_version,
    )
    joint_prim = final_stage.GetPrimAtPath(authored_path)
    assert joint_prim.GetAttribute("physxJoint:jointFriction").Get() == pytest.approx(
        0.25
    )
    assert joint_prim.GetCustomDataByKey("jointRigger:jointId") == "hinge"
    assert joint_prim.GetCustomDataByKey("jointRigger:planSha256") == canonical_sha256(
        request.plan
    )
    for body_path in ("/World/base", "/World/link"):
        schemas = set(final_stage.GetPrimAtPath(body_path).GetAppliedSchemas())
        assert {"PhysicsRigidBodyAPI", "PhysicsMassAPI"} <= schemas
    del final_stage


def test_v2_combined_authoring_aggregates_flat_siblings_deterministically(
    tmp_path: Path,
) -> None:
    source, request = _aggregate_fixture(tmp_path)
    source_bytes = source.read_bytes()
    first_targets = _targets(tmp_path, "aggregate-first")
    second_targets = _targets(tmp_path, "aggregate-second")

    first_result = author_joint_rig_with_physics(
        request,
        source_usd_path=source,
        artifact_targets=first_targets,
    )
    second_result = author_joint_rig_with_physics(
        request,
        source_usd_path=source,
        artifact_targets=second_targets,
    )

    assert source.read_bytes() == source_bytes
    assert (
        first_targets.output_path.read_bytes()
        == second_targets.output_path.read_bytes()
    )
    assert first_result.input_sha256 == canonical_sha256(request)
    assert first_result.plan_sha256 == canonical_sha256(request.plan)
    assert first_result.output_artifact is not None
    assert second_result.output_artifact is not None
    assert (
        first_result.output_artifact.root_sha256
        == second_result.output_artifact.root_sha256
    )
    validate_authored_joint_rig_with_physics(
        request,
        first_result,
        output_usd_path=first_targets.output_path,
    )

    source_stage = Usd.Stage.Open(str(source))
    output_stage = Usd.Stage.Open(str(first_targets.output_path))
    assert source_stage is not None
    assert output_stage is not None
    assert source_stage.GetPrimAtPath("/World/panel_a").IsValid()
    assert source_stage.GetPrimAtPath("/World/panel_b").IsValid()
    assert not source_stage.GetPrimAtPath("/World/drawer").IsValid()
    assert not output_stage.GetPrimAtPath("/World/panel_a").IsValid()
    assert not output_stage.GetPrimAtPath("/World/panel_b").IsValid()

    aggregate = output_stage.GetPrimAtPath("/World/drawer")
    assert aggregate.IsA(UsdGeom.Xform)
    assert {str(child.GetPath()) for child in aggregate.GetAllChildren()} == {
        "/World/drawer/panel_a",
        "/World/drawer/panel_b",
    }
    assert aggregate.GetCustomDataByKey("jointRigger:rigidLinkId") == "drawer"
    assert aggregate.HasAPI(UsdPhysics.RigidBodyAPI)
    for collider_path in (
        "/World/drawer/panel_a/collision",
        "/World/drawer/panel_b/collision",
    ):
        assert output_stage.GetPrimAtPath(collider_path).HasAPI(UsdPhysics.CollisionAPI)

    joints = [prim for prim in output_stage.Traverse() if prim.IsA(UsdPhysics.Joint)]
    assert len(joints) == 1
    joint = UsdPhysics.Joint(joints[0])
    assert tuple(str(path) for path in joint.GetBody0Rel().GetTargets()) == (
        "/World/base",
    )
    assert tuple(str(path) for path in joint.GetBody1Rel().GetTargets()) == (
        "/World/drawer",
    )
    del output_stage
    del source_stage


def test_v2_two_components_support_aggregate_and_nested_existing_links(
    tmp_path: Path,
) -> None:
    source, request = _multi_root_fixture(
        tmp_path,
        stem="aggregate-nested",
        edges=(
            ("drawer_slide", "/World/base", "/World/drawer"),
            (
                "appliance_door",
                "/World/appliance",
                "/World/appliance/door",
            ),
        ),
        aggregate_body_path="/World/drawer",
    )
    targets = _targets(tmp_path, "aggregate-nested-output")

    result = author_joint_rig_with_physics(
        request,
        source_usd_path=source,
        artifact_targets=targets,
    )
    validate_authored_joint_rig_with_physics(
        request,
        result,
        output_usd_path=targets.output_path,
    )

    stage = Usd.Stage.Open(str(targets.output_path))
    assert stage is not None
    roots = {
        str(prim.GetPath())
        for prim in stage.Traverse()
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    }
    assert roots == {"/World/appliance", "/World/base"}
    assert stage.GetPrimAtPath("/World/drawer").HasAPI(UsdPhysics.RigidBodyAPI)
    assert stage.GetPrimAtPath("/World/appliance/door").HasAPI(UsdPhysics.RigidBodyAPI)
    assert not stage.GetPrimAtPath("/World/panel_a").IsValid()
    assert stage.GetPrimAtPath("/World/drawer/panel_a").IsValid()
    assert (
        len(tuple(prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.Joint))) == 2
    )
    del stage


def test_v2_kitchen_shape_authors_six_joints_and_five_exact_roots(
    tmp_path: Path,
) -> None:
    edges = (
        ("drawer_slide", "/World/cabinet", "/World/cabinet/drawer"),
        (
            "drawer_handle",
            "/World/cabinet/drawer",
            "/World/cabinet/drawer/handle",
        ),
        *tuple(
            (
                f"appliance_{index}_door",
                f"/World/appliance_{index}",
                f"/World/appliance_{index}/door",
            )
            for index in range(1, 5)
        ),
    )
    source, request = _multi_root_fixture(
        tmp_path,
        stem="kitchen-six-five",
        edges=edges,
    )
    targets = _targets(tmp_path, "kitchen-six-five-output")

    result = author_joint_rig_with_physics(
        request,
        source_usd_path=source,
        artifact_targets=targets,
    )
    validate_authored_joint_rig_with_physics(
        request,
        result,
        output_usd_path=targets.output_path,
    )

    assert len(result.diagnostics.joint_diagnostics) == 6
    root_fields = {
        decision.field
        for decision in result.diagnostics.field_decisions
        if decision.field.startswith("articulation_roots[")
    }
    assert root_fields == {
        "articulation_roots[/World/cabinet]",
        *(f"articulation_roots[/World/appliance_{index}]" for index in range(1, 5)),
    }
    stage = Usd.Stage.Open(str(targets.output_path))
    assert stage is not None
    assert {
        str(prim.GetPath())
        for prim in stage.Traverse()
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    } == {
        "/World/cabinet",
        *(f"/World/appliance_{index}" for index in range(1, 5)),
    }
    assert (
        len(tuple(prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.Joint))) == 6
    )
    del stage


def test_v2_existing_links_are_identity_only_and_bind_v2_plan_hash(
    tmp_path: Path,
) -> None:
    source, v1_request = _fixture(tmp_path)
    source_bytes = source.read_bytes()
    v2_request = JointRiggerInputV2(
        schema_version=INPUT_SCHEMA_VERSION_V2,
        source_asset=v1_request.source_asset,
        plan=_v2_plan_from_v1(v1_request.plan),
        rigid_links=tuple(
            RigidLinkPlanV1(
                link_id=body_path.rsplit("/", maxsplit=1)[-1],
                body_authoring="existing",
                body_prim_path=body_path,
                members=(
                    RigidLinkMemberPlanV1(
                        source_prim_path=body_path,
                        authored_prim_path=body_path,
                    ),
                ),
            )
            for body_path in ("/World/base", "/World/link")
        ),
    )
    v1_targets = _targets(tmp_path, "identity-v1")
    v2_targets = _targets(tmp_path, "identity-v2")

    v1_result = author_joint_rig_with_physics(
        v1_request,
        source_usd_path=source,
        artifact_targets=v1_targets,
    )
    v2_result = author_joint_rig_with_physics(
        v2_request,
        source_usd_path=source,
        artifact_targets=v2_targets,
    )

    assert source.read_bytes() == source_bytes
    assert v1_result.output_artifact is not None
    assert v2_result.output_artifact is not None
    assert v1_result.plan_sha256 == canonical_sha256(v1_request.plan)
    assert v2_result.plan_sha256 == canonical_sha256(v2_request.plan)
    assert v1_result.plan_sha256 != v2_result.plan_sha256
    assert v1_result.input_sha256 != v2_result.input_sha256
    stage = Usd.Stage.Open(str(v2_targets.output_path))
    assert stage is not None
    for body_path in ("/World/base", "/World/link"):
        assert (
            stage.GetPrimAtPath(body_path).GetCustomDataByKey("jointRigger:rigidLinkId")
            is None
        )
    del stage


def test_v2_stale_aggregate_mapping_fails_before_publication(tmp_path: Path) -> None:
    source, request = _aggregate_fixture(tmp_path)
    aggregate = next(
        link for link in request.rigid_links if link.body_authoring == "aggregate"
    )
    stale_aggregate = RigidLinkPlanV1(
        link_id=aggregate.link_id,
        body_authoring="aggregate",
        body_prim_path=aggregate.body_prim_path,
        members=(
            RigidLinkMemberPlanV1(
                source_prim_path="/World/missing_panel",
                authored_prim_path="/World/drawer/missing_panel",
            ),
            aggregate.members[1],
        ),
    )
    stale_request = JointRiggerInputV2(
        schema_version=INPUT_SCHEMA_VERSION_V2,
        source_asset=request.source_asset,
        plan=request.plan,
        rigid_links=tuple(
            stale_aggregate if link.link_id == aggregate.link_id else link
            for link in request.rigid_links
        ),
    )
    targets = _targets(tmp_path, "stale-aggregate")

    with pytest.raises(JointRiggerContractError, match="aggregate_source_missing"):
        author_joint_rig_with_physics(
            stale_request,
            source_usd_path=source,
            artifact_targets=targets,
        )

    assert not targets.output_path.exists()
    assert not targets.diagnostics_path.exists()
    assert not targets.result_path.exists()


def test_topology_only_backend_authors_exact_v2_aggregate_links(
    tmp_path: Path,
) -> None:
    source, request = _aggregate_fixture(tmp_path)
    topology_request = JointRiggerInputV2(
        schema_version=INPUT_SCHEMA_VERSION_V2,
        source_asset=request.source_asset,
        plan=_v2_topology_phase_plan(request.plan),
        rigid_links=request.rigid_links,
    )
    targets = _targets(tmp_path, "topology-only-aggregate")

    result = author_joint_topology(
        topology_request,
        source_usd_path=source,
        artifact_targets=targets,
    )

    assert result.input_sha256 == canonical_sha256(topology_request)
    assert result.plan_sha256 == canonical_sha256(topology_request.plan)
    stage = Usd.Stage.Open(str(targets.output_path))
    assert stage is not None
    rigid_links_module.validate_authored_rigid_links(stage, topology_request)
    topology_plan = combined_module._topology_phase_plan(topology_request.plan)
    validation_module.validate_authored_joint_topology(
        stage,
        topology_plan,
        result.diagnostics,
    )
    assert stage.GetPrimAtPath("/World/drawer/panel_a").IsValid()
    assert stage.GetPrimAtPath("/World/drawer/panel_b").IsValid()
    assert not stage.GetPrimAtPath("/World/panel_a").IsValid()
    assert not stage.GetPrimAtPath("/World/panel_b").IsValid()


def test_v2_saved_output_validation_rejects_aggregate_metadata_tampering(
    tmp_path: Path,
) -> None:
    source, request = _aggregate_fixture(tmp_path)
    targets = _targets(tmp_path, "aggregate-tamper")
    result = author_joint_rig_with_physics(
        request,
        source_usd_path=source,
        artifact_targets=targets,
    )

    targets.output_path.chmod(0o600)
    stage = Usd.Stage.Open(str(targets.output_path))
    assert stage is not None
    stage.GetPrimAtPath("/World/drawer").SetCustomDataByKey(
        "jointRigger:rigidLinkId",
        "tampered",
    )
    assert stage.GetRootLayer().Save()
    del stage
    tampered_result = result.model_copy(
        update={
            "output_artifact": identify_usd_artifact(
                targets.output_path,
                uri=str(targets.output_path),
            )
        }
    )

    with pytest.raises(JointRiggerContractError, match="authored_aggregate_mismatch"):
        validate_authored_joint_rig_with_physics(
            request,
            tampered_result,
            output_usd_path=targets.output_path,
        )


def test_v2_saved_output_rejects_aggregate_member_transform_drift(
    tmp_path: Path,
) -> None:
    source, request = _aggregate_fixture(tmp_path)
    targets = _targets(tmp_path, "aggregate-member-drift")
    result = author_joint_rig_with_physics(
        request,
        source_usd_path=source,
        artifact_targets=targets,
    )

    targets.output_path.chmod(0o600)
    stage = Usd.Stage.Open(str(targets.output_path))
    assert stage is not None
    panel = UsdGeom.Xformable(stage.GetPrimAtPath("/World/drawer/panel_a"))
    translate_ops = panel.GetOrderedXformOps()
    assert len(translate_ops) == 1
    assert translate_ops[0].Set(Gf.Vec3d(99.0, -0.5, 0.0))
    assert stage.GetRootLayer().Save()
    del stage
    tampered_result = result.model_copy(
        update={
            "output_artifact": identify_usd_artifact(
                targets.output_path,
                uri=str(targets.output_path),
            )
        }
    )

    with pytest.raises(JointRiggerContractError, match="aggregate_member_changed"):
        validate_authored_joint_rig_with_physics(
            request,
            tampered_result,
            output_usd_path=targets.output_path,
        )


@pytest.mark.parametrize(
    "failure_hook",
    ("_author_aggregate_metadata", "_validate_authored_aggregates"),
)
def test_v2_aggregate_private_stage_rollback_restores_namespace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_hook: str,
) -> None:
    source, request = _aggregate_fixture(tmp_path)
    source_bytes = source.read_bytes()
    stage = Usd.Stage.Open(str(source))
    assert stage is not None
    layer_before = stage.GetRootLayer().ExportToString()

    def fail_after_private_mutation(*_args: object, **_kwargs: object) -> None:
        raise JointRiggerContractError("injected_failure", failure_hook)

    monkeypatch.setattr(
        rigid_links_module,
        failure_hook,
        fail_after_private_mutation,
    )
    with pytest.raises(JointRiggerContractError, match="injected_failure"):
        rigid_links_module.author_aggregate_rigid_links(stage, request)

    assert stage.GetRootLayer().ExportToString() == layer_before
    assert stage.GetPrimAtPath("/World/panel_a").IsValid()
    assert stage.GetPrimAtPath("/World/panel_b").IsValid()
    assert not stage.GetPrimAtPath("/World/drawer").IsValid()
    del stage
    assert source.read_bytes() == source_bytes


@pytest.mark.parametrize(
    ("source_case", "reason"),
    (
        ("reference", "aggregate_dependency_authorship_unsupported"),
        ("payload", "aggregate_dependency_authorship_unsupported"),
        (
            "variant",
            "aggregate_(dependency_authorship|variant)_unsupported",
        ),
        ("instance", "aggregate_instance_unsupported"),
        ("body_collision", "aggregate_body_collision"),
    ),
)
def test_v2_aggregate_rejects_unsafe_source_composition_before_publication(
    tmp_path: Path,
    source_case: str,
    reason: str,
) -> None:
    source, request = _aggregate_fixture(tmp_path, source_case=source_case)
    source_bytes = source.read_bytes()
    targets = _targets(tmp_path, f"unsafe-{source_case}")

    with pytest.raises(JointRiggerContractError, match=reason):
        author_joint_rig_with_physics(
            request,
            source_usd_path=source,
            artifact_targets=targets,
        )

    assert source.read_bytes() == source_bytes
    assert not targets.output_path.exists()
    assert not targets.diagnostics_path.exists()
    assert not targets.result_path.exists()


@pytest.mark.parametrize("approximation", ["none", "convexHull"])
def test_combined_author_round_trips_xform_instance_root_colliders(
    tmp_path: Path,
    approximation: Literal["none", "convexHull"],
) -> None:
    source, request = _fixture(
        tmp_path,
        instance_root_approximation=approximation,
    )
    targets = _targets(tmp_path, f"instance-root-{approximation}")

    result = author_joint_rig_with_physics(
        request,
        source_usd_path=source,
        artifact_targets=targets,
    )

    validate_authored_joint_rig_with_physics(
        request,
        result,
        output_usd_path=targets.output_path,
    )
    stage = Usd.Stage.Open(str(targets.output_path))
    assert stage is not None
    for body_path in ("/World/base", "/World/link"):
        collider = stage.GetPrimAtPath(f"{body_path}/collision")
        assert collider.IsInstance()
        assert collider.HasAPI(UsdPhysics.CollisionAPI)
        assert collider.HasAPI(UsdPhysics.MeshCollisionAPI)
        assert collider.GetAttribute("physics:approximation").Get() == approximation


def test_combined_resolution_maps_mimic_reference_to_authored_path(
    tmp_path: Path,
) -> None:
    _, request = _fixture(tmp_path)
    source_joint = request.plan.joints[0]
    assert source_joint.state is not None
    provenance = source_joint.state.provenance
    follower = JointPlanV1(
        topology=JointTopologyV1(
            joint_id="follower",
            joint_type="revolute",
            body0=source_joint.topology.body0,
            body1=source_joint.topology.body1,
            axis_stage=source_joint.topology.axis_stage,
            field_provenance=source_joint.topology.field_provenance,
        ),
        mimic=JointMimicV1(
            reference_joint_id="hinge",
            gearing=-1.0,
            offset=0.0,
            natural_frequency=4.0,
            damping_ratio=0.7,
            provenance=provenance,
        ),
    )
    plan = JointRiggerPlanV1(
        schema_version=PLAN_SCHEMA_VERSION,
        joints=(*request.plan.joints, follower),
    )
    diagnostics = JointRiggerDiagnosticsV1(
        schema_version=DIAGNOSTICS_SCHEMA_VERSION,
        backend_name="owned_topology",
        joint_diagnostics=(
            JointDiagnosticV1(
                joint_id="hinge",
                field_decisions=(
                    FieldDecisionV1(
                        field="usd.joint_prim_path",
                        disposition="defaulted",
                        reason_code="deterministic_joint_path",
                        detail="/World/Joints/hinge_000000000000",
                    ),
                ),
            ),
            JointDiagnosticV1(
                joint_id="follower",
                field_decisions=(
                    FieldDecisionV1(
                        field="usd.joint_prim_path",
                        disposition="defaulted",
                        reason_code="deterministic_joint_path",
                        detail="/World/Joints/follower_000000000000",
                    ),
                ),
            ),
        ),
    )

    resolved = combined_module._resolve_physics_plan(plan, diagnostics)

    resolved_by_path = {joint.topology.joint_id: joint for joint in resolved.joints}
    resolved_follower = resolved_by_path["/World/Joints/follower_000000000000"]
    assert resolved_follower.mimic is not None
    assert (
        resolved_follower.mimic.reference_joint_id == "/World/Joints/hinge_000000000000"
    )


@pytest.mark.parametrize(
    "failure_point",
    ["topology", "physics", "validation", "combined_readback"],
)
def test_combined_author_preserves_old_bundle_on_phase_or_validation_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    source, request = _fixture(tmp_path)
    targets = _targets(tmp_path, "existing")
    author_joint_rig_with_physics(
        request,
        source_usd_path=source,
        artifact_targets=targets,
    )
    expected = {
        targets.output_path: targets.output_path.read_bytes(),
        targets.diagnostics_path: targets.diagnostics_path.read_bytes(),
        targets.result_path: targets.result_path.read_bytes(),
    }

    def fail(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise JointRiggerContractError(
            "injected_failure",
            f"injected {failure_point} failure",
        )

    if failure_point == "topology":
        monkeypatch.setattr(combined_module, "_author_topology_stage", fail)
    elif failure_point == "physics":
        monkeypatch.setattr(combined_module, "author_physics_schemas", fail)
    elif failure_point == "validation":
        monkeypatch.setattr(
            combined_module,
            "validate_authored_physics_schemas",
            fail,
        )
    else:
        monkeypatch.setattr(
            combined_module,
            "validate_authored_joint_rig_with_physics",
            fail,
        )

    with pytest.raises(
        JointRiggerArtifactError,
        match="injected_failure",
    ):
        author_joint_rig_with_physics(
            request,
            source_usd_path=source,
            artifact_targets=targets,
        )

    assert {path: path.read_bytes() for path in expected} == expected
    staged = [path for path in tmp_path.iterdir() if ".stage-" in path.name]
    # Every fallible combined validation now runs against the frozen private
    # source projection before any facade staging path is created.
    assert not staged


@pytest.mark.parametrize(
    ("failure_point", "expected_prefix"),
    [
        (
            "authoring",
            "Combined authoring failed post-preflight validation",
        ),
        (
            "final_descriptor_validation",
            "Final descriptor-pinned combined validation failed",
        ),
        (
            "saved_post_author_validation",
            "Saved combined output failed post-author validation",
        ),
    ],
)
def test_combined_author_frames_artifact_failures_and_preserves_old_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
    expected_prefix: str,
) -> None:
    source, request = _fixture(tmp_path)
    targets = _targets(tmp_path, "artifact-error-framing")
    author_joint_rig_with_physics(
        request,
        source_usd_path=source,
        artifact_targets=targets,
    )
    expected = {
        targets.output_path: targets.output_path.read_bytes(),
        targets.diagnostics_path: targets.diagnostics_path.read_bytes(),
        targets.result_path: targets.result_path.read_bytes(),
    }
    injected = JointRiggerArtifactError(f"injected {failure_point} failure")

    def fail(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise injected

    if failure_point == "authoring":
        monkeypatch.setattr(combined_module, "_resolve_physics_plan", fail)
    elif failure_point == "final_descriptor_validation":
        monkeypatch.setattr(combined_module, "_validate_saved_combined_stage", fail)
    else:
        monkeypatch.setattr(
            combined_module,
            "validate_authored_joint_rig_with_physics",
            fail,
        )

    with pytest.raises(JointRiggerArtifactError) as raised:
        author_joint_rig_with_physics(
            request,
            source_usd_path=source,
            artifact_targets=targets,
        )

    assert str(raised.value) == (f"{expected_prefix}: injected {failure_point} failure")
    assert raised.value.__cause__ is injected
    assert {path: path.read_bytes() for path in expected} == expected
    staged = [path for path in tmp_path.iterdir() if ".stage-" in path.name]
    assert not staged


@pytest.mark.parametrize(
    "failure_point",
    [
        "post_copy",
        "diagnostics_write",
        "result_write",
        "report_mid_bind",
        "outer_cleanup",
    ],
)
def test_combined_late_failure_removes_descriptor_bound_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    source, request = _fixture(tmp_path)
    targets = _targets(tmp_path, "late-failure")
    author_joint_rig_with_physics(
        request,
        source_usd_path=source,
        artifact_targets=targets,
    )
    expected = {
        targets.output_path: targets.output_path.read_bytes(),
        targets.diagnostics_path: targets.diagnostics_path.read_bytes(),
        targets.result_path: targets.result_path.read_bytes(),
    }

    if failure_point == "post_copy":
        original_copy = combined_module.copy_regular_file_to_new_path

        def copy_then_fail(*args: Any, **kwargs: Any) -> None:
            original_copy(*args, **kwargs)
            raise JointRiggerArtifactError("injected post-copy failure")

        monkeypatch.setattr(
            combined_module,
            "copy_regular_file_to_new_path",
            copy_then_fail,
        )
    elif failure_point in {"diagnostics_write", "result_write"}:
        failed_label = {
            "diagnostics_write": "Joint Rigger diagnostics",
            "result_write": "Joint Rigger result",
        }[failure_point]
        original_write = combined_module.write_new_text_file

        def fail_selected_write(
            path: Path,
            payload: str,
            *,
            label: str,
            bind_created_file: Any = None,
        ) -> None:
            if label == failed_label:
                raise JointRiggerArtifactError(f"injected {failure_point} failure")
            original_write(
                path,
                payload,
                label=label,
                bind_created_file=bind_created_file,
            )

        monkeypatch.setattr(
            combined_module,
            "write_new_text_file",
            fail_selected_write,
        )
    elif failure_point == "report_mid_bind":
        original_bind = artifacts_module._bind_staging_promotion_source

        def fail_report_mid_bind(*args: Any, **kwargs: Any) -> None:
            if Path(kwargs["path"]).suffix == ".json":
                raise JointRiggerArtifactError("injected report_mid_bind failure")
            original_bind(*args, **kwargs)

        monkeypatch.setattr(
            artifacts_module,
            "_bind_staging_promotion_source",
            fail_report_mid_bind,
        )
    else:
        original_projection = combined_module._bound_source_projection

        @contextmanager
        def projection_with_failed_exit(*args: Any, **kwargs: Any) -> Any:
            with original_projection(*args, **kwargs) as projection:
                yield projection
            raise JointRiggerArtifactError("injected outer cleanup failure")

        monkeypatch.setattr(
            combined_module,
            "_bound_source_projection",
            projection_with_failed_exit,
        )

    with pytest.raises(JointRiggerArtifactError, match="injected"):
        author_joint_rig_with_physics(
            request,
            source_usd_path=source,
            artifact_targets=targets,
        )

    assert {path: path.read_bytes() for path in expected} == expected
    assert not [path for path in tmp_path.iterdir() if ".stage-" in path.name]


def test_combined_public_readback_is_read_only(tmp_path: Path) -> None:
    source, request = _fixture(tmp_path)
    targets = _targets(tmp_path, "readback")
    result = author_joint_rig_with_physics(
        request,
        source_usd_path=source,
        artifact_targets=targets,
    )
    before_bytes = targets.output_path.read_bytes()
    before_mode = targets.output_path.stat().st_mode

    validate_authored_joint_rig_with_physics(
        request,
        result,
        output_usd_path=targets.output_path,
    )

    assert targets.output_path.read_bytes() == before_bytes
    assert targets.output_path.stat().st_mode == before_mode


def test_combined_facade_uses_one_sealed_author_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, request = _fixture(tmp_path)
    targets = _targets(tmp_path, "one-lifecycle")
    original_projection = combined_module._bound_source_projection
    projection_count = 0

    def counted_projection(*args: Any, **kwargs: Any) -> Any:
        nonlocal projection_count
        projection_count += 1
        return original_projection(*args, **kwargs)

    monkeypatch.setattr(
        combined_module,
        "_bound_source_projection",
        counted_projection,
    )

    result = author_joint_rig_with_physics(
        request,
        source_usd_path=source,
        artifact_targets=targets,
    )

    assert result.status == "succeeded"
    assert projection_count == 1


def test_direct_combined_backend_requires_facade_staging_cleanup(
    tmp_path: Path,
) -> None:
    source, request = _fixture(tmp_path)
    targets = _targets(tmp_path, "direct-without-facade")

    with pytest.raises(
        RuntimeError,
        match="requires facade-owned staging cleanup",
    ):
        OwnedTopologyAndPhysicsBackend(source).author(request, targets)

    assert not targets.output_path.exists()
    assert not targets.diagnostics_path.exists()
    assert not targets.result_path.exists()


@pytest.mark.parametrize(
    "target_field",
    ["output_path", "diagnostics_path", "result_path"],
)
def test_direct_combined_backend_rejects_bound_source_target_alias(
    tmp_path: Path,
    target_field: str,
) -> None:
    source, request = _fixture(tmp_path)
    source_bytes = source.read_bytes()
    target_values = {
        "output_path": tmp_path / "direct.usda",
        "diagnostics_path": tmp_path / "direct.diagnostics.json",
        "result_path": tmp_path / "direct.result.json",
    }
    target_values[target_field] = source
    targets = JointRiggerArtifactTargets(**target_values)

    with pytest.raises(
        JointRiggerArtifactError,
        match=rf"{target_field} must not alias bound source USD",
    ):
        OwnedTopologyAndPhysicsBackend(source).author(request, targets)

    assert source.read_bytes() == source_bytes
    for field, path in target_values.items():
        if field != target_field:
            assert not path.exists()


def test_combined_public_readback_accepts_owned_drive_schema_tokens(
    tmp_path: Path,
) -> None:
    source, request = _fixture(tmp_path)
    joint = request.plan.joints[0]
    assert joint.state is not None
    drive = JointDriveV1(
        drive_type="force",
        stiffness=20.0,
        damping=2.0,
        max_force=100.0,
        target_position=0.0,
        target_velocity=0.0,
        max_joint_velocity=3.0,
        provenance=joint.state.provenance,
    )
    plan = request.plan.model_copy(
        update={"joints": (joint.model_copy(update={"drive": drive}),)}
    )
    request = request.model_copy(update={"plan": plan})
    targets = _targets(tmp_path, "drive-readback")

    result = author_joint_rig_with_physics(
        request,
        source_usd_path=source,
        artifact_targets=targets,
    )

    validate_authored_joint_rig_with_physics(
        request,
        result,
        output_usd_path=targets.output_path,
    )


def test_topology_readback_accepts_static_r3_owned_joint_property(
    tmp_path: Path,
) -> None:
    source, request = _fixture(tmp_path)
    targets = _targets(tmp_path, "static-r3-property")
    result = author_joint_rig_with_physics(
        request,
        source_usd_path=source,
        artifact_targets=targets,
    )
    stage = Usd.Stage.Open(str(targets.output_path))
    assert stage is not None
    joint_path = result.diagnostics.joint_diagnostics[0].authored_prim_path
    assert joint_path is not None
    state_position = stage.GetPrimAtPath(joint_path).GetAttribute(
        "state:angular:physics:position"
    )
    assert state_position.HasAuthoredValueOpinion()
    assert state_position.GetTimeSamples() == []

    _validate_topology_readback_with_r3_allowlists(stage, request, result)


def test_topology_readback_rejects_time_sampled_r3_owned_joint_property(
    tmp_path: Path,
) -> None:
    source, request = _fixture(tmp_path)
    targets = _targets(tmp_path, "sampled-r3-property")
    result = author_joint_rig_with_physics(
        request,
        source_usd_path=source,
        artifact_targets=targets,
    )
    stage = Usd.Stage.Open(str(targets.output_path))
    assert stage is not None
    joint_path = result.diagnostics.joint_diagnostics[0].authored_prim_path
    assert joint_path is not None
    state_position = stage.GetPrimAtPath(joint_path).GetAttribute(
        "state:angular:physics:position"
    )
    assert state_position.Set(1.0, Usd.TimeCode(1.0))

    with pytest.raises(JointRiggerContractError) as error:
        _validate_topology_readback_with_r3_allowlists(stage, request, result)

    assert error.value.code == "time_sampled_owned_property"
    assert "state:angular:physics:position" in error.value.detail
    assert "time-sampled" in error.value.detail
    assert "(1.0,)" in error.value.detail


def test_combined_public_readback_accepts_owned_mimic_schema_tokens(
    tmp_path: Path,
) -> None:
    source, request = _mimic_fixture(tmp_path)
    targets = _targets(tmp_path, "mimic-readback")

    result = author_joint_rig_with_physics(
        request,
        source_usd_path=source,
        artifact_targets=targets,
    )

    validate_authored_joint_rig_with_physics(
        request,
        result,
        output_usd_path=targets.output_path,
    )


@pytest.mark.parametrize(
    ("fixture_kind", "mutation"),
    (
        ("state", "custom"),
        ("state", "uniform_variability"),
        ("drive", "unexpected_metadata"),
        ("mimic", "relationship_list_op"),
        ("drive", "api_order"),
        ("drive", "api_ordered_items"),
    ),
)
def test_combined_public_readback_rejects_noncanonical_r3_raw_authorship(
    tmp_path: Path,
    fixture_kind: str,
    mutation: str,
) -> None:
    if fixture_kind == "mimic":
        source, request = _mimic_fixture(tmp_path)
    else:
        source, request = _fixture(tmp_path)
        if fixture_kind == "drive":
            request = _request_with_drive(request)
    targets = _targets(tmp_path, f"raw-r3-{fixture_kind}-{mutation}")
    result = author_joint_rig_with_physics(
        request,
        source_usd_path=source,
        artifact_targets=targets,
    )
    diagnostics_index = 1 if fixture_kind == "mimic" else 0
    joint_path = result.diagnostics.joint_diagnostics[
        diagnostics_index
    ].authored_prim_path
    assert joint_path is not None

    targets.output_path.chmod(0o600)
    stage = Usd.Stage.Open(str(targets.output_path))
    assert stage is not None
    prim_spec = stage.GetRootLayer().GetPrimAtPath(joint_path)
    assert prim_spec is not None
    if mutation == "custom":
        attribute = prim_spec.properties["state:angular:physics:position"]
        assert isinstance(attribute, Sdf.AttributeSpec)
        attribute.custom = True
        del attribute
    elif mutation == "uniform_variability":
        attribute = prim_spec.properties["state:angular:physics:position"]
        assert isinstance(attribute, Sdf.AttributeSpec)
        attribute.SetInfo("variability", Sdf.VariabilityUniform)
        del attribute
    elif mutation == "unexpected_metadata":
        attribute = prim_spec.properties["drive:angular:physics:stiffness"]
        assert isinstance(attribute, Sdf.AttributeSpec)
        attribute.SetInfo("documentation", "adversarial metadata")
        del attribute
    elif mutation == "relationship_list_op":
        relationship = prim_spec.properties["physxMimicJoint:rotZ:referenceJoint"]
        assert isinstance(relationship, Sdf.RelationshipSpec)
        target_list = relationship.GetInfo("targetPaths")
        assert isinstance(target_list, Sdf.PathListOp)
        explicit_targets = list(target_list.explicitItems)
        relationship.targetPathList.ClearEdits()
        relationship.targetPathList.prependedItems = explicit_targets
        del target_list, relationship
    elif mutation in {"api_order", "api_ordered_items"}:
        schemas = prim_spec.GetInfo("apiSchemas")
        assert isinstance(schemas, Sdf.TokenListOp)
        if mutation == "api_order":
            schemas.prependedItems = list(reversed(schemas.prependedItems))
        else:
            schemas.orderedItems = list(schemas.prependedItems)
        prim_spec.SetInfo("apiSchemas", schemas)
        del schemas
    else:  # pragma: no cover - parameter table and mutation logic stay in lockstep
        raise AssertionError(f"unhandled raw R3 mutation: {mutation}")
    assert stage.GetRootLayer().Save()
    del prim_spec, stage

    current_result = _result_for_current_output(result, targets.output_path)
    with pytest.raises(JointRiggerContractError) as error:
        validate_authored_joint_rig_with_physics(
            request,
            current_result,
            output_usd_path=targets.output_path,
        )

    assert error.value.code == "authored_graph_mismatch"
    assert "noncanonical raw metadata" in error.value.detail or (
        "raw" in error.value.detail
    )


def test_combined_public_readback_rejects_unplanned_joint_property(
    tmp_path: Path,
) -> None:
    source, request = _fixture(tmp_path)
    targets = _targets(tmp_path, "unexpected-joint-property")
    result = author_joint_rig_with_physics(
        request,
        source_usd_path=source,
        artifact_targets=targets,
    )
    joint_path = result.diagnostics.joint_diagnostics[0].authored_prim_path
    assert joint_path is not None
    targets.output_path.chmod(0o600)
    stage = Usd.Stage.Open(str(targets.output_path))
    assert stage is not None
    unexpected = stage.GetPrimAtPath(joint_path).CreateAttribute(
        "unexpected:jointProperty",
        Sdf.ValueTypeNames.Float,
        custom=True,
    )
    assert unexpected.Set(1.0)
    assert stage.GetRootLayer().Save()
    del stage
    assert result.output_artifact is not None
    current_identity = identify_usd_artifact(
        targets.output_path,
        uri=result.output_artifact.uri,
    )
    current_result = result.model_copy(update={"output_artifact": current_identity})

    with pytest.raises(JointRiggerContractError) as error:
        validate_authored_joint_rig_with_physics(
            request,
            current_result,
            output_usd_path=targets.output_path,
        )

    assert error.value.code == "authored_graph_mismatch"
    assert "unexpected:jointProperty" in error.value.detail


@pytest.mark.parametrize("identity_field", ["input_sha256", "plan_sha256", "output"])
def test_combined_public_readback_rejects_result_identity_mismatch(
    tmp_path: Path,
    identity_field: str,
) -> None:
    source, request = _fixture(tmp_path)
    targets = _targets(tmp_path, "identity")
    result = author_joint_rig_with_physics(
        request,
        source_usd_path=source,
        artifact_targets=targets,
    )
    if identity_field == "output":
        assert result.output_artifact is not None
        bad_output = result.output_artifact.model_copy(update={"root_sha256": "0" * 64})
        bad_result = result.model_copy(update={"output_artifact": bad_output})
    else:
        bad_result = result.model_copy(update={identity_field: "0" * 64})

    with pytest.raises(JointRiggerArtifactError, match="identity"):
        validate_authored_joint_rig_with_physics(
            request,
            bad_result,
            output_usd_path=targets.output_path,
        )


def test_combined_public_readback_rejects_non_succeeded_result(
    tmp_path: Path,
) -> None:
    source, request = _fixture(tmp_path)
    targets = _targets(tmp_path, "failed-result")
    result = author_joint_rig_with_physics(
        request,
        source_usd_path=source,
        artifact_targets=targets,
    )
    failed_result = result.model_copy(
        update={"status": "failed", "output_artifact": None}
    )

    with pytest.raises(JointRiggerArtifactError, match="succeeded result"):
        validate_authored_joint_rig_with_physics(
            request,
            failed_result,
            output_usd_path=targets.output_path,
        )


def test_combined_public_readback_rejects_owned_schema_mutation(
    tmp_path: Path,
) -> None:
    source, request = _fixture(tmp_path)
    targets = _targets(tmp_path, "schema-mutation")
    result = author_joint_rig_with_physics(
        request,
        source_usd_path=source,
        artifact_targets=targets,
    )
    targets.output_path.chmod(0o600)
    stage = Usd.Stage.Open(str(targets.output_path))
    assert stage is not None
    mass = UsdPhysics.MassAPI(stage.GetPrimAtPath("/World/base"))
    assert mass.GetMassAttr().Set(23.0)
    assert stage.GetRootLayer().Save()
    del stage
    assert result.output_artifact is not None
    current_identity = identify_usd_artifact(
        targets.output_path,
        uri=result.output_artifact.uri,
    )
    current_result = result.model_copy(update={"output_artifact": current_identity})

    with pytest.raises(JointRiggerContractError, match="physics_schema_conflict"):
        validate_authored_joint_rig_with_physics(
            request,
            current_result,
            output_usd_path=targets.output_path,
        )


def test_combined_public_readback_rejects_sub_tolerance_r3_default_drift(
    tmp_path: Path,
) -> None:
    source, request = _fixture(tmp_path)
    targets = _targets(tmp_path, "sub-tolerance-r3-default-drift")
    result = author_joint_rig_with_physics(
        request,
        source_usd_path=source,
        artifact_targets=targets,
    )
    joint_path = result.diagnostics.joint_diagnostics[0].authored_prim_path
    assert joint_path is not None

    targets.output_path.chmod(0o600)
    stage = Usd.Stage.Open(str(targets.output_path))
    assert stage is not None
    prim_spec = stage.GetRootLayer().GetPrimAtPath(joint_path)
    assert prim_spec is not None
    state = prim_spec.properties["state:angular:physics:position"]
    assert isinstance(state, Sdf.AttributeSpec)
    state.SetInfo("default", 5e-7)
    assert stage.GetRootLayer().Save()
    del state, prim_spec, stage

    current_result = _result_for_current_output(result, targets.output_path)
    with pytest.raises(JointRiggerContractError) as error:
        validate_authored_joint_rig_with_physics(
            request,
            current_result,
            output_usd_path=targets.output_path,
        )

    assert error.value.code == "authored_graph_mismatch"
    assert "noncanonical default" in error.value.detail


def test_combined_public_readback_rejects_cross_layer_r3_family_token(
    tmp_path: Path,
) -> None:
    source, request = _fixture(tmp_path)
    targets = _targets(tmp_path, "cross-layer-r3-family-token")
    result = author_joint_rig_with_physics(
        request,
        source_usd_path=source,
        artifact_targets=targets,
    )

    malformed_layer_path = tmp_path / "cross-layer-malformed-r3.usda"
    malformed_layer = Sdf.Layer.CreateNew(str(malformed_layer_path))
    assert malformed_layer is not None
    prim_spec = Sdf.CreatePrimInLayer(malformed_layer, "/World/base")
    schemas = Sdf.TokenListOp()
    schemas.prependedItems = ["PhysicsRigidBodyAPI:adversarial"]
    prim_spec.SetInfo("apiSchemas", schemas)
    assert malformed_layer.Save()
    del schemas, prim_spec, malformed_layer

    targets.output_path.chmod(0o600)
    stage = Usd.Stage.Open(str(targets.output_path))
    assert stage is not None
    stage.GetRootLayer().subLayerPaths.append(malformed_layer_path.name)
    assert stage.GetRootLayer().Save()
    del stage

    current_result = _result_for_current_output(result, targets.output_path)
    with pytest.raises(JointRiggerContractError) as error:
        validate_authored_joint_rig_with_physics(
            request,
            current_result,
            output_usd_path=targets.output_path,
        )

    assert error.value.code == "authored_graph_mismatch"
    assert "contributing raw apiSchemas" in error.value.detail
    assert "PhysicsRigidBodyAPI:adversarial" in error.value.detail


def test_combined_public_readback_rejects_cross_layer_r3_attribute_metadata(
    tmp_path: Path,
) -> None:
    source, request = _fixture(tmp_path)
    targets = _targets(tmp_path, "cross-layer-r3-attribute-metadata")
    result = author_joint_rig_with_physics(
        request,
        source_usd_path=source,
        artifact_targets=targets,
    )

    malformed_layer_path = tmp_path / "cross-layer-malformed-r3-attribute.usda"
    malformed_layer = Sdf.Layer.CreateNew(str(malformed_layer_path))
    assert malformed_layer is not None
    prim_spec = Sdf.CreatePrimInLayer(malformed_layer, "/World/base")
    Sdf.AttributeSpec(
        prim_spec,
        "physics:mass",
        Sdf.ValueTypeNames.Float,
        Sdf.VariabilityVarying,
        True,
    )
    assert malformed_layer.Save()
    del prim_spec, malformed_layer

    targets.output_path.chmod(0o600)
    stage = Usd.Stage.Open(str(targets.output_path))
    assert stage is not None
    stage.GetRootLayer().subLayerPaths.append(malformed_layer_path.name)
    assert stage.GetRootLayer().Save()
    del stage

    current_result = _result_for_current_output(result, targets.output_path)
    with pytest.raises(JointRiggerContractError) as error:
        validate_authored_joint_rig_with_physics(
            request,
            current_result,
            output_usd_path=targets.output_path,
        )

    assert error.value.code == "authored_graph_mismatch"
    assert "contributing raw attribute" in error.value.detail
    assert "physics:mass" in error.value.detail


def test_combined_public_readback_rejects_cross_layer_unexpected_owned_property(
    tmp_path: Path,
) -> None:
    source, request = _fixture(tmp_path)
    targets = _targets(tmp_path, "cross-layer-r3-unexpected-property")
    result = author_joint_rig_with_physics(
        request,
        source_usd_path=source,
        artifact_targets=targets,
    )

    malformed_layer_path = tmp_path / "unexpected-r3-property.usda"
    malformed_layer = Sdf.Layer.CreateNew(str(malformed_layer_path))
    assert malformed_layer is not None
    prim_spec = Sdf.CreatePrimInLayer(malformed_layer, "/World/base/collision")
    rogue = Sdf.AttributeSpec(
        prim_spec,
        "physics:mass",
        Sdf.ValueTypeNames.Float,
        Sdf.VariabilityVarying,
        False,
    )
    rogue.default = 1.0
    assert malformed_layer.Save()
    del rogue, prim_spec, malformed_layer

    targets.output_path.chmod(0o600)
    stage = Usd.Stage.Open(str(targets.output_path))
    assert stage is not None
    stage.GetRootLayer().subLayerPaths.append(malformed_layer_path.name)
    assert stage.GetRootLayer().Save()
    del stage

    current_result = _result_for_current_output(result, targets.output_path)
    with pytest.raises(JointRiggerContractError) as error:
        validate_authored_joint_rig_with_physics(
            request,
            current_result,
            output_usd_path=targets.output_path,
        )

    assert error.value.code == "authored_graph_mismatch"
    assert "unexpected plan-owned names" in error.value.detail
    assert "physics:mass" in error.value.detail


@pytest.mark.parametrize("violation", ["time_samples", "spline", "connections"])
def test_combined_public_readback_rejects_cross_layer_r3_attribute_state(
    tmp_path: Path,
    violation: str,
) -> None:
    source, request = _fixture(tmp_path)
    targets = _targets(tmp_path, f"cross-layer-r3-attribute-{violation}")
    result = author_joint_rig_with_physics(
        request,
        source_usd_path=source,
        artifact_targets=targets,
    )

    malformed_layer_path = tmp_path / f"malformed-r3-attribute-{violation}.usda"
    malformed_layer = Sdf.Layer.CreateNew(str(malformed_layer_path))
    assert malformed_layer is not None
    prim_spec = Sdf.CreatePrimInLayer(malformed_layer, "/World/base")
    attribute = Sdf.AttributeSpec(
        prim_spec,
        "physics:mass",
        Sdf.ValueTypeNames.Float,
        Sdf.VariabilityVarying,
        False,
    )
    spline_property_path: Sdf.Path | None = None
    if violation == "time_samples":
        malformed_layer.SetTimeSample(attribute.path, 1.0, 999.0)
    elif violation == "spline":
        malformed_stage = Usd.Stage.Open(malformed_layer)
        assert malformed_stage is not None
        malformed_attribute = malformed_stage.GetPrimAtPath("/World/base").GetAttribute(
            "physics:mass"
        )
        spline = Ts.Spline("float")
        knot = Ts.Knot("float")
        knot.SetTime(1.0)
        knot.SetValue(3.0)
        spline.SetKnot(knot)
        assert malformed_attribute.SetSpline(spline)
        assert malformed_attribute.HasSpline()
        assert malformed_attribute.GetTimeSamples() == []
        assert "spline" in {str(key) for key in attribute.ListInfoKeys()}
        spline_property_path = attribute.path
        del malformed_attribute, malformed_stage
    else:
        attribute.connectionPathList.ClearEditsAndMakeExplicit()
        assert "connectionPaths" in {str(key) for key in attribute.ListInfoKeys()}
    assert malformed_layer.Save()
    del attribute, prim_spec, malformed_layer
    if spline_property_path is not None:
        persisted_layer = Sdf.Layer.OpenAsAnonymous(str(malformed_layer_path))
        assert persisted_layer is not None
        persisted_attribute = persisted_layer.GetPropertyAtPath(spline_property_path)
        assert isinstance(persisted_attribute, Sdf.AttributeSpec)
        assert "spline" in {str(key) for key in persisted_attribute.ListInfoKeys()}
        del persisted_attribute, persisted_layer

    targets.output_path.chmod(0o600)
    stage = Usd.Stage.Open(str(targets.output_path))
    assert stage is not None
    stage.GetRootLayer().subLayerPaths.append(malformed_layer_path.name)
    assert stage.GetRootLayer().Save()
    del stage

    composed_has_spline: bool | None = None
    if violation == "spline":
        composed_stage = Usd.Stage.Open(str(targets.output_path))
        assert composed_stage is not None
        composed_attribute = composed_stage.GetPrimAtPath("/World/base").GetAttribute(
            "physics:mass"
        )
        composed_has_spline = composed_attribute.HasSpline()
        del composed_attribute, composed_stage

    current_result = _result_for_current_output(result, targets.output_path)
    with pytest.raises(JointRiggerContractError) as error:
        validate_authored_joint_rig_with_physics(
            request,
            current_result,
            output_usd_path=targets.output_path,
        )

    expected_code = {
        "time_samples": "authored_graph_mismatch",
        "spline": (
            "time_sampled_owned_property"
            if composed_has_spline
            else "authored_graph_mismatch"
        ),
        "connections": "connected_owned_property",
    }[violation]
    assert error.value.code == expected_code
    if violation == "time_samples":
        assert "contributing raw attribute" in error.value.detail
    expected_detail = {
        "time_samples": "time samples",
        "spline": "spline",
        "connections": "connection",
    }[violation]
    assert expected_detail in error.value.detail


@pytest.mark.parametrize(
    "owned_channel",
    [
        "joint-local-pos",
        "endpoint-xform-op",
        "joint-enabled",
        "joint-break-force",
    ],
)
def test_combined_public_readback_rejects_masked_r2_value_clip_samples(
    tmp_path: Path,
    owned_channel: str,
) -> None:
    source, request = _fixture(tmp_path)
    targets = _targets(tmp_path, f"masked-r2-clip-{owned_channel}")
    result = author_joint_rig_with_physics(
        request,
        source_usd_path=source,
        artifact_targets=targets,
    )
    joint_path = result.diagnostics.joint_diagnostics[0].authored_prim_path
    assert joint_path is not None

    if owned_channel == "joint-local-pos":
        scene_prim_path = joint_path
        clip_prim_path = "/Clip" + joint_path.removeprefix("/World")
        attribute_name = "physics:localPos0"
        value_type = Sdf.ValueTypeNames.Point3f
        sample_value = Gf.Vec3f(17.0, 18.0, 19.0)
        active = True
    elif owned_channel == "endpoint-xform-op":
        scene_prim_path = "/World/link"
        clip_prim_path = "/Clip/link"
        attribute_name = "xformOp:translate"
        value_type = Sdf.ValueTypeNames.Double3
        sample_value = Gf.Vec3d(17.0, 18.0, 19.0)
        active = True
    elif owned_channel == "joint-enabled":
        scene_prim_path = joint_path
        clip_prim_path = "/Clip" + joint_path.removeprefix("/World")
        attribute_name = "physics:jointEnabled"
        value_type = Sdf.ValueTypeNames.Bool
        sample_value = False
        active = True
    else:
        scene_prim_path = joint_path
        clip_prim_path = "/Clip" + joint_path.removeprefix("/World")
        attribute_name = "physics:breakForce"
        value_type = Sdf.ValueTypeNames.Float
        sample_value = 3.0
        active = False

    clip_asset_path = tmp_path / f"{owned_channel}-samples.usda"
    clip_layer = Sdf.Layer.CreateNew(str(clip_asset_path))
    clip_prim_spec = Sdf.CreatePrimInLayer(clip_layer, clip_prim_path)
    clip_attribute = Sdf.AttributeSpec(
        clip_prim_spec,
        attribute_name,
        value_type,
        Sdf.VariabilityVarying,
        False,
    )
    clip_layer.SetTimeSample(clip_attribute.path, 1.0, sample_value)
    assert clip_layer.Save()
    del clip_attribute, clip_prim_spec, clip_layer

    clip_metadata_path = tmp_path / f"{owned_channel}-metadata.usda"
    clip_metadata_stage = Usd.Stage.CreateNew(str(clip_metadata_path))
    assert clip_metadata_stage is not None
    UsdGeom.Xform.Define(clip_metadata_stage, "/World")
    clips = Usd.ClipsAPI(clip_metadata_stage.GetPrimAtPath("/World"))
    clips.SetClipAssetPaths([Sdf.AssetPath(clip_asset_path.name)], "r2_masked")
    assert clips.SetClipPrimPath("/Clip", "r2_masked")
    if active:
        clips.SetClipActive([(0.0, 0.0)], "r2_masked")
    assert clip_metadata_stage.GetRootLayer().Save()
    del clips, clip_metadata_stage

    targets.output_path.chmod(0o600)
    stage = Usd.Stage.Open(str(targets.output_path))
    assert stage is not None
    stage.GetRootLayer().subLayerPaths.append(clip_metadata_path.name)
    assert stage.GetRootLayer().Save()
    del stage

    persisted_stage = Usd.Stage.Open(str(targets.output_path))
    assert persisted_stage is not None
    persisted_attribute = persisted_stage.GetPrimAtPath(scene_prim_path).GetAttribute(
        attribute_name
    )
    assert persisted_attribute
    if owned_channel in {"joint-local-pos", "endpoint-xform-op", "joint-break-force"}:
        assert persisted_attribute.GetTimeSamples() == []
    else:
        assert persisted_attribute.GetTimeSamples()
    del persisted_attribute, persisted_stage

    current_result = _result_for_current_output(result, targets.output_path)
    with pytest.raises(JointRiggerContractError) as error:
        validate_authored_joint_rig_with_physics(
            request,
            current_result,
            output_usd_path=targets.output_path,
        )

    assert error.value.code == "authored_graph_mismatch"
    assert attribute_name in error.value.detail
    assert clip_asset_path.name in error.value.detail


def test_combined_public_readback_rejects_result_diagnostic_mutation(
    tmp_path: Path,
) -> None:
    source, request = _fixture(tmp_path)
    targets = _targets(tmp_path, "diagnostic-mutation")
    result = author_joint_rig_with_physics(
        request,
        source_usd_path=source,
        artifact_targets=targets,
    )
    diagnostics = result.diagnostics.model_copy(
        update={"backend_version": "mutated-backend"}
    )
    bad_result = result.model_copy(update={"diagnostics": diagnostics})

    with pytest.raises(JointRiggerArtifactError, match="diagnostics"):
        validate_authored_joint_rig_with_physics(
            request,
            bad_result,
            output_usd_path=targets.output_path,
        )


def test_combined_public_readback_rejects_stage_diagnostic_mutation(
    tmp_path: Path,
) -> None:
    source, request = _fixture(tmp_path)
    targets = _targets(tmp_path, "stage-diagnostic-mutation")
    result = author_joint_rig_with_physics(
        request,
        source_usd_path=source,
        artifact_targets=targets,
    )
    authored_path = result.diagnostics.joint_diagnostics[0].authored_prim_path
    assert authored_path is not None
    targets.output_path.chmod(0o600)
    stage = Usd.Stage.Open(str(targets.output_path))
    assert stage is not None
    prim = stage.GetPrimAtPath(authored_path)
    prim.SetCustomDataByKey("jointRigger:fieldDecisions", "[]")
    assert stage.GetRootLayer().Save()
    del stage
    assert result.output_artifact is not None
    current_identity = identify_usd_artifact(
        targets.output_path,
        uri=result.output_artifact.uri,
    )
    current_result = result.model_copy(update={"output_artifact": current_identity})

    with pytest.raises(JointRiggerContractError, match="field-decision provenance"):
        validate_authored_joint_rig_with_physics(
            request,
            current_result,
            output_usd_path=targets.output_path,
        )


@pytest.mark.parametrize("substitution", ["root", "dependency"])
def test_combined_public_readback_rejects_aba_path_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    substitution: Literal["root", "dependency"],
) -> None:
    dependency, request = _fixture(tmp_path)
    if substitution == "root":
        source = dependency
    else:
        source = tmp_path / "dependency-root.usda"
        source_layer = Sdf.Layer.CreateNew(str(source))
        source_layer.defaultPrim = "World"
        source_layer.subLayerPaths.append(dependency.name)
        assert source_layer.Save()
        del source_layer
        request = request.model_copy(
            update={"source_asset": identify_usd_artifact(source, uri=str(source))}
        )
    targets = _targets(tmp_path, f"identity-aba-{substitution}")
    result = author_joint_rig_with_physics(
        request,
        source_usd_path=source,
        artifact_targets=targets,
    )
    substituted_path = targets.output_path if substitution == "root" else dependency
    valid = tmp_path / f"valid-{substitution}.usda"
    invalid = tmp_path / f"invalid-{substitution}.usda"
    shutil.copy2(substituted_path, valid)
    substituted_path.chmod(0o600)
    invalid_stage = Usd.Stage.Open(str(substituted_path))
    assert invalid_stage is not None
    if substitution == "root":
        assert (
            invalid_stage.GetPrimAtPath("/World/base")
            .GetAttribute("physics:mass")
            .Set(99.0)
        )
    else:
        link = UsdGeom.Xformable(invalid_stage.GetPrimAtPath("/World/link"))
        translate_ops = link.GetOrderedXformOps()
        assert len(translate_ops) == 1
        assert translate_ops[0].Set(Gf.Vec3d(99.0, 0.0, 0.0))
    assert invalid_stage.GetRootLayer().Save()
    del invalid_stage
    shutil.copy2(substituted_path, invalid)
    assert result.output_artifact is not None
    invalid_identity = identify_usd_artifact(
        targets.output_path,
        uri=result.output_artifact.uri,
    )
    invalid_result = result.model_copy(update={"output_artifact": invalid_identity})
    invalid_bytes = invalid.read_bytes()

    identify = combined_module.identify_usd_artifact
    create_binding = combined_module.create_sealed_source_binding
    open_stage = combined_module._open_stage
    identity_calls = 0
    binding_calls = 0
    private_stage_paths: list[Path] = []
    live_path_is_valid = False
    aba_restored = False

    def substitute_around_validation(*args: Any, **kwargs: Any) -> Any:
        nonlocal aba_restored, identity_calls, live_path_is_valid
        identity_calls += 1
        if identity_calls == 2:
            invalid.replace(substituted_path)
            aba_restored = True
            live_path_is_valid = False
        identity = identify(*args, **kwargs)
        if identity_calls == 1:
            valid.replace(substituted_path)
            live_path_is_valid = True
        return identity

    def bind_then_substitute(*args: Any, **kwargs: Any) -> Any:
        nonlocal binding_calls, live_path_is_valid
        binding_calls += 1
        binding = create_binding(*args, **kwargs)
        valid.replace(substituted_path)
        live_path_is_valid = True
        return binding

    def restore_aba_before_private_open(path: Path, *, label: str) -> Any:
        nonlocal aba_restored, live_path_is_valid
        if path != targets.output_path:
            private_stage_paths.append(path)
            stage = open_stage(path, label=label)
            if live_path_is_valid:
                invalid.replace(substituted_path)
                aba_restored = True
                live_path_is_valid = False
            return stage
        return open_stage(path, label=label)

    monkeypatch.setattr(
        combined_module,
        "identify_usd_artifact",
        substitute_around_validation,
    )
    monkeypatch.setattr(
        combined_module,
        "create_sealed_source_binding",
        bind_then_substitute,
    )
    monkeypatch.setattr(combined_module, "_open_stage", restore_aba_before_private_open)

    with pytest.raises(JointRiggerContractError) as error:
        validate_authored_joint_rig_with_physics(
            request,
            invalid_result,
            output_usd_path=targets.output_path,
        )
    assert (
        error.value.code
        == {
            "root": "physics_schema_conflict",
            "dependency": "contradictory_joint_frames",
        }[substitution]
    )
    assert binding_calls == 1
    assert identity_calls == 0
    assert len(private_stage_paths) == 1
    assert private_stage_paths[0].as_posix().startswith("/proc/self/fd/")
    assert aba_restored
    assert substituted_path.read_bytes() == invalid_bytes


def test_combined_public_readback_type_guards(tmp_path: Path) -> None:
    source, request = _fixture(tmp_path)
    targets = _targets(tmp_path, "type-guards")
    result = author_joint_rig_with_physics(
        request,
        source_usd_path=source,
        artifact_targets=targets,
    )

    with pytest.raises(TypeError, match="request"):
        validate_authored_joint_rig_with_physics(  # type: ignore[arg-type]
            object(),
            result,
            output_usd_path=targets.output_path,
        )
    with pytest.raises(TypeError, match="result"):
        validate_authored_joint_rig_with_physics(  # type: ignore[arg-type]
            request,
            object(),
            output_usd_path=targets.output_path,
        )


def test_combined_probe_rejects_missing_evidence_before_opening_the_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, request = _fixture(tmp_path)
    first_body = request.plan.rigid_bodies[0].model_copy(update={"mass": None})
    plan = request.plan.model_copy(
        update={
            "rigid_bodies": (first_body, *request.plan.rigid_bodies[1:]),
        }
    )
    request = request.model_copy(update={"plan": plan})
    opened = False

    def reject_open(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        nonlocal opened
        opened = True
        raise AssertionError("probe opened USD before pure evidence validation")

    monkeypatch.setattr(combined_module, "_open_stage", reject_open)

    with pytest.raises(JointRiggerContractError) as caught:
        combined_module.OwnedTopologyAndPhysicsBackend(source).probe(request)

    assert caught.value.code == "mass_evidence_missing"
    assert opened is False


def test_combined_probe_opens_only_a_descriptor_bound_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, request = _fixture(tmp_path)
    original_open = combined_module._open_stage
    opened_paths: list[Path] = []

    def track_open(path: Path, *, label: str) -> Any:
        opened_paths.append(path)
        assert path != source
        return original_open(path, label=label)

    monkeypatch.setattr(combined_module, "_open_stage", track_open)

    combined_module.OwnedTopologyAndPhysicsBackend(source).probe(request)

    assert len(opened_paths) == 1
    assert "joint-rigger-bound-input" in opened_paths[0].as_posix()
    assert not opened_paths[0].exists()


def test_combined_authors_from_sealed_root_during_transient_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, request = _fixture(tmp_path, source_marker="original")
    alternate = tmp_path / "alternate.usda"
    shutil.copy2(source, alternate)
    alternate_layer = Sdf.Layer.FindOrOpen(str(alternate))
    assert alternate_layer is not None
    alternate_layer.customLayerData = {"sealedSource": "alternate"}
    assert alternate_layer.Save()
    del alternate_layer
    original_bytes = source.read_bytes()
    alternate_bytes = alternate.read_bytes()
    original_author = combined_module._author_topology_stage
    swapped = False

    def author_while_live_source_is_swapped(stage: Any, plan: Any) -> Any:
        nonlocal swapped
        source.write_bytes(alternate_bytes)
        swapped = True
        try:
            return original_author(stage, plan)
        finally:
            source.write_bytes(original_bytes)

    monkeypatch.setattr(
        combined_module,
        "_author_topology_stage",
        author_while_live_source_is_swapped,
    )
    targets = _targets(tmp_path, "sealed-root")

    result = author_joint_rig_with_physics(
        request,
        source_usd_path=source,
        artifact_targets=targets,
    )

    assert result.status == "succeeded"
    assert swapped
    assert source.read_bytes() == original_bytes
    output_layer = Sdf.Layer.FindOrOpen(str(targets.output_path))
    assert output_layer is not None
    assert output_layer.customLayerData["sealedSource"] == "original"
    output_bytes = targets.output_path.read_bytes()
    assert b"joint-rigger-bound-input" not in output_bytes
    assert b"/proc/self/fd/" not in output_bytes


def test_combined_relative_dependency_relocation_fails_closed(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "output"
    source_dir.mkdir()
    output_dir.mkdir()
    dependency, dependency_request = _fixture(source_dir)
    source = source_dir / "root.usda"
    root_layer = Sdf.Layer.CreateNew(str(source))
    root_layer.defaultPrim = "World"
    root_layer.subLayerPaths.append(dependency.name)
    assert root_layer.Save()
    del root_layer
    source_artifact = identify_usd_artifact(source, uri=str(source))
    request = dependency_request.model_copy(update={"source_asset": source_artifact})
    targets = JointRiggerArtifactTargets(
        output_path=output_dir / "rigged.usda",
        diagnostics_path=output_dir / "diagnostics.json",
        result_path=output_dir / "result.json",
    )

    with pytest.raises(
        combined_module.JointRiggerBackendIncompatibleError,
        match="cannot relocate.*composition dependencies",
    ):
        author_joint_rig_with_physics(
            request,
            source_usd_path=source,
            artifact_targets=targets,
        )

    assert not targets.output_path.exists()
    assert not targets.diagnostics_path.exists()
    assert not targets.result_path.exists()


def test_combined_authors_from_sealed_dependency_during_transient_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_dir = tmp_path / "dependency-swap"
    source_dir.mkdir()
    dependency, dependency_request = _fixture(source_dir)
    alternate = source_dir / "alternate.usda"
    shutil.copy2(dependency, alternate)
    alternate_stage = Usd.Stage.Open(str(alternate))
    assert alternate_stage is not None
    alternate_link = UsdGeom.Xformable(alternate_stage.GetPrimAtPath("/World/link"))
    alternate_ops = alternate_link.GetOrderedXformOps()
    assert len(alternate_ops) == 1
    assert alternate_ops[0].Set(Gf.Vec3d(99.0, 0.0, 0.0))
    assert alternate_stage.GetRootLayer().Save()
    del alternate_stage
    source = source_dir / "root.usda"
    root_layer = Sdf.Layer.CreateNew(str(source))
    root_layer.defaultPrim = "World"
    root_layer.subLayerPaths.append(dependency.name)
    assert root_layer.Save()
    del root_layer
    source_artifact = identify_usd_artifact(source, uri=str(source))
    request = dependency_request.model_copy(update={"source_asset": source_artifact})
    original_dependency = dependency.read_bytes()
    alternate_dependency = alternate.read_bytes()
    original_author = combined_module._author_topology_stage
    swapped = False

    def author_while_live_dependency_is_swapped(stage: Any, plan: Any) -> Any:
        nonlocal swapped
        dependency.write_bytes(alternate_dependency)
        swapped = True
        try:
            return original_author(stage, plan)
        finally:
            dependency.write_bytes(original_dependency)

    monkeypatch.setattr(
        combined_module,
        "_author_topology_stage",
        author_while_live_dependency_is_swapped,
    )
    targets = _targets(source_dir, "sealed-dependency")

    result = author_joint_rig_with_physics(
        request,
        source_usd_path=source,
        artifact_targets=targets,
    )

    assert result.status == "succeeded"
    assert result.output_artifact == identify_usd_artifact(
        targets.output_path,
        uri=str(targets.output_path),
    )
    assert swapped
    assert dependency.read_bytes() == original_dependency
    output_stage = Usd.Stage.Open(str(targets.output_path))
    assert output_stage is not None
    authored_path = result.diagnostics.joint_diagnostics[0].authored_prim_path
    assert authored_path is not None
    joint = UsdPhysics.RevoluteJoint(output_stage.GetPrimAtPath(authored_path))
    assert tuple(joint.GetLocalPos0Attr().Get()) == pytest.approx((1.0, 0.0, 0.0))
    assert tuple(joint.GetLocalPos1Attr().Get()) == pytest.approx((0.0, 0.0, 0.0))
    output_bytes = targets.output_path.read_bytes()
    assert b"joint-rigger-bound-input" not in output_bytes
    assert b"/proc/self/fd/" not in output_bytes


def test_combined_facade_cleans_staging_after_connected_attribute_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, request = _fixture(tmp_path)
    targets = _targets(tmp_path, "connected-existing")
    author_joint_rig_with_physics(
        request,
        source_usd_path=source,
        artifact_targets=targets,
    )
    expected = {
        targets.output_path: targets.output_path.read_bytes(),
        targets.diagnostics_path: targets.diagnostics_path.read_bytes(),
        targets.result_path: targets.result_path.read_bytes(),
    }
    original_author = combined_module._author_topology_stage

    def author_with_connection(stage: Any, plan: JointRiggerPlanV1) -> Any:
        diagnostics = original_author(stage, plan)
        path = diagnostics.joint_diagnostics[0].authored_prim_path
        assert path is not None
        driver = stage.GetPrimAtPath("/World").CreateAttribute(
            "outputs:axis",
            Sdf.ValueTypeNames.Token,
            custom=True,
        )
        axis = stage.GetPrimAtPath(path).GetAttribute("physics:axis")
        assert axis.AddConnection(driver.GetPath())
        return diagnostics

    monkeypatch.setattr(
        combined_module,
        "_author_topology_stage",
        author_with_connection,
    )

    with pytest.raises(
        JointRiggerArtifactError,
        match="connected_owned_property",
    ):
        author_joint_rig_with_physics(
            request,
            source_usd_path=source,
            artifact_targets=targets,
        )

    assert {path: path.read_bytes() for path in expected} == expected
    assert not [path for path in tmp_path.iterdir() if ".stage-" in path.name]


def test_combined_final_frozen_validation_rejects_pre_freeze_root_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, request = _fixture(tmp_path)
    targets = _targets(tmp_path, "pre-freeze-replacement")
    validated_copy = tmp_path / "validated-combined.usda"
    real_freeze = combined_module.freeze_bound_projection_root
    replacement_injected = False

    def replace_root_before_freeze(path: Path, **kwargs: Any) -> Any:
        nonlocal replacement_injected
        assert not replacement_injected
        replacement_injected = True
        shutil.copy2(path, validated_copy)
        path.rename(path.with_name(f"{path.name}.validated-away"))
        shutil.copy2(source, path)
        return real_freeze(path, **kwargs)

    monkeypatch.setattr(
        combined_module,
        "freeze_bound_projection_root",
        replace_root_before_freeze,
    )

    with pytest.raises(
        JointRiggerArtifactError,
        match="Final descriptor-pinned combined validation failed",
    ):
        author_joint_rig_with_physics(
            request,
            source_usd_path=source,
            artifact_targets=targets,
        )

    assert replacement_injected
    validated_stage = Usd.Stage.Open(str(validated_copy))
    assert validated_stage is not None
    assert any(prim.IsA(UsdPhysics.Joint) for prim in validated_stage.Traverse())
    assert not targets.output_path.exists()
    assert not targets.diagnostics_path.exists()
    assert not targets.result_path.exists()
    assert not any(tmp_path.rglob(".*.stage-*"))


def test_combined_copy_uses_the_validated_frozen_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, request = _fixture(tmp_path)
    targets = _targets(tmp_path, "frozen-copy")
    real_copy = combined_module.copy_regular_file_to_new_path
    observed_frozen_descriptor: int | None = None

    def capture_copy(
        source_path: Path,
        target_path: Path,
        *,
        label: str,
        frozen_source: Any = None,
        bind_created_file: Any = None,
    ) -> None:
        nonlocal observed_frozen_descriptor
        assert frozen_source is not None
        observed_frozen_descriptor = frozen_source.descriptor
        real_copy(
            source_path,
            target_path,
            label=label,
            frozen_source=frozen_source,
            bind_created_file=bind_created_file,
        )

    monkeypatch.setattr(
        combined_module,
        "copy_regular_file_to_new_path",
        capture_copy,
    )

    result = author_joint_rig_with_physics(
        request,
        source_usd_path=source,
        artifact_targets=targets,
    )

    assert result.status == "succeeded"
    assert observed_frozen_descriptor is not None
    assert targets.output_path.exists()


def _fixture(
    tmp_path: Path,
    *,
    source_marker: str | None = None,
    instance_root_approximation: Literal["none", "convexHull"] | None = None,
    joint_friction: float | None = None,
) -> tuple[Path, JointRiggerInputV1]:
    source = tmp_path / "source.usda"
    stage = Usd.Stage.CreateNew(str(source))
    assert stage is not None
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)

    if instance_root_approximation is not None:
        UsdGeom.Scope.Define(stage, "/World/ColliderPrototypes")
    for index, body_path in enumerate(("/World/base", "/World/link")):
        body = UsdGeom.Xform.Define(stage, body_path)
        body.AddTranslateOp().Set(Gf.Vec3d(float(index), 0.0, 0.0))
        evidence = body.GetPrim().CreateAttribute(
            "jointEvidence:axis",
            Sdf.ValueTypeNames.Float3,
            custom=True,
        )
        evidence.Set(Gf.Vec3f(0.0, 0.0, 1.0))
        collider_path = f"{body_path}/collision"
        if instance_root_approximation is None:
            UsdGeom.Cube.Define(stage, collider_path)
        else:
            prototype_path = f"/World/ColliderPrototypes/prototype_{index}"
            UsdGeom.Xform.Define(stage, prototype_path)
            UsdGeom.Cube.Define(stage, f"{prototype_path}/shape")
            collider = UsdGeom.Xform.Define(stage, collider_path).GetPrim()
            collider.GetReferences().AddInternalReference(prototype_path)
            collider.SetInstanceable(True)
            assert collider.IsInstance()
    if source_marker is not None:
        stage.GetRootLayer().customLayerData = {"sealedSource": source_marker}
    assert stage.GetRootLayer().Save()
    del stage

    source_artifact = identify_usd_artifact(source, uri=str(source))
    provenance = FieldProvenanceV1(
        source="authored_metadata",
        artifact=source_artifact,
        prim_path="/World/link",
        properties=("jointEvidence:axis",),
        evidence="The fixture authors this exact evidence property.",
    )
    topology = JointTopologyV1(
        joint_id="hinge",
        joint_type="revolute",
        body0="/World/base",
        body1="/World/link",
        axis_stage=(0.0, 0.0, 1.0),
        field_provenance={
            "joint_type": provenance,
            "body0": provenance,
            "body1": provenance,
            "axis_stage": provenance,
        },
    )
    body_plans = tuple(
        RigidBodyPlanV1(
            prim_path=body_path,
            mass=MassPropertiesV1(
                mass_kg=2.0,
                diagonal_inertia_kg_m2=(1.0, 1.0, 1.0),
                provenance=provenance,
            ),
            colliders=(
                ColliderPlanV1(
                    prim_path=f"{body_path}/collision",
                    mesh_approximation=instance_root_approximation,
                    provenance=provenance,
                ),
            ),
            provenance=provenance,
        )
        for body_path in ("/World/base", "/World/link")
    )
    plan = JointRiggerPlanV1(
        schema_version=PLAN_SCHEMA_VERSION,
        joints=(
            JointPlanV1(
                topology=topology,
                joint_friction=(
                    JointFrictionV1(
                        coefficient=joint_friction,
                        provenance=provenance,
                    )
                    if joint_friction is not None
                    else None
                ),
                state=JointStateV1(
                    position=0.0,
                    velocity=0.0,
                    provenance=provenance,
                ),
            ),
        ),
        rigid_bodies=body_plans,
        articulation_root=ArticulationRootPlanV1(
            prim_path="/World/base",
            provenance=provenance,
        ),
    )
    return source, JointRiggerInputV1(
        schema_version=INPUT_SCHEMA_VERSION,
        source_asset=source_artifact,
        plan=plan,
        conflict_policy="error",
    )


def _aggregate_fixture(
    tmp_path: Path,
    *,
    source_case: str | None = None,
) -> tuple[Path, JointRiggerInputV2]:
    source = tmp_path / "aggregate-source.usda"
    stage = Usd.Stage.CreateNew(str(source))
    assert stage is not None
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)

    base = UsdGeom.Xform.Define(stage, "/World/base")
    base.AddTranslateOp().Set(Gf.Vec3d(-1.0, 0.0, 0.0))
    UsdGeom.Cube.Define(stage, "/World/base/collision")
    for panel_path, translate in (
        ("/World/panel_a", Gf.Vec3d(1.0, -0.5, 0.0)),
        ("/World/panel_b", Gf.Vec3d(1.0, 0.5, 0.0)),
    ):
        panel = UsdGeom.Xform.Define(stage, panel_path)
        panel.AddTranslateOp().Set(translate)
        UsdGeom.Cube.Define(stage, f"{panel_path}/collision")
    evidence = stage.GetPrimAtPath("/World/panel_a").CreateAttribute(
        "jointEvidence:axis",
        Sdf.ValueTypeNames.Float3,
        custom=True,
    )
    assert evidence.Set(Gf.Vec3f(0.0, 0.0, 1.0))
    panel_a = stage.GetPrimAtPath("/World/panel_a")
    if source_case == "reference":
        panel_a.GetReferences().AddInternalReference("/World/base")
    elif source_case == "payload":
        panel_a.GetPayloads().AddInternalPayload("/World/base")
    elif source_case == "variant":
        variant_set = panel_a.GetVariantSets().AddVariantSet("shape")
        assert variant_set.AddVariant("default")
        assert variant_set.SetVariantSelection("default")
        with variant_set.GetVariantEditContext():
            UsdGeom.Scope.Define(stage, "/World/panel_a/variant_child")
    elif source_case == "instance":
        UsdGeom.Xform.Define(stage, "/World/prototype")
        UsdGeom.Cube.Define(stage, "/World/prototype/shape")
        panel_a.GetReferences().AddInternalReference("/World/prototype")
        assert panel_a.SetInstanceable(True)
    elif source_case == "body_collision":
        UsdGeom.Xform.Define(stage, "/World/drawer")
    elif source_case is not None:
        raise AssertionError(f"unknown aggregate fixture source_case: {source_case}")
    assert stage.GetRootLayer().Save()
    del stage

    source_artifact = identify_usd_artifact(source, uri=str(source))
    provenance = FieldProvenanceV1(
        source="authored_metadata",
        artifact=source_artifact,
        prim_path="/World/panel_a",
        properties=("jointEvidence:axis",),
        evidence="The source panel authors the aggregate joint evidence.",
    )
    plan = JointRiggerPlanV2(
        schema_version=PLAN_SCHEMA_VERSION_V2,
        joints=(
            JointPlanV1(
                topology=JointTopologyV1(
                    joint_id="drawer_slide",
                    joint_type="prismatic",
                    body0="/World/base",
                    body1="/World/drawer",
                    axis_stage=(0.0, 0.0, 1.0),
                    field_provenance=dict.fromkeys(
                        ("joint_type", "body0", "body1", "axis_stage"),
                        provenance,
                    ),
                ),
                state=JointStateV1(
                    position=0.0,
                    velocity=0.0,
                    provenance=provenance,
                ),
            ),
        ),
        rigid_bodies=(
            RigidBodyPlanV1(
                prim_path="/World/base",
                mass=MassPropertiesV1(
                    mass_kg=2.0,
                    diagonal_inertia_kg_m2=(1.0, 1.0, 1.0),
                    provenance=provenance,
                ),
                colliders=(
                    ColliderPlanV1(
                        prim_path="/World/base/collision",
                        provenance=provenance,
                    ),
                ),
                provenance=provenance,
            ),
            RigidBodyPlanV1(
                prim_path="/World/drawer",
                mass=MassPropertiesV1(
                    mass_kg=1.0,
                    diagonal_inertia_kg_m2=(0.5, 0.5, 0.5),
                    provenance=provenance,
                ),
                colliders=tuple(
                    ColliderPlanV1(
                        prim_path=f"/World/drawer/{panel}/collision",
                        provenance=provenance,
                    )
                    for panel in ("panel_a", "panel_b")
                ),
                provenance=provenance,
            ),
        ),
        articulation_roots=(
            ArticulationRootPlanV1(
                prim_path="/World/base",
                provenance=provenance,
            ),
        ),
    )
    return source, JointRiggerInputV2(
        schema_version=INPUT_SCHEMA_VERSION_V2,
        source_asset=source_artifact,
        plan=plan,
        rigid_links=(
            RigidLinkPlanV1(
                link_id="drawer",
                body_authoring="aggregate",
                body_prim_path="/World/drawer",
                members=tuple(
                    RigidLinkMemberPlanV1(
                        source_prim_path=f"/World/{panel}",
                        authored_prim_path=f"/World/drawer/{panel}",
                    )
                    for panel in ("panel_a", "panel_b")
                ),
            ),
            RigidLinkPlanV1(
                link_id="base",
                body_authoring="existing",
                body_prim_path="/World/base",
                members=(
                    RigidLinkMemberPlanV1(
                        source_prim_path="/World/base",
                        authored_prim_path="/World/base",
                    ),
                ),
            ),
        ),
        conflict_policy="error",
    )


def _multi_root_fixture(
    tmp_path: Path,
    *,
    stem: str,
    edges: tuple[tuple[str, str, str], ...],
    aggregate_body_path: str | None = None,
) -> tuple[Path, JointRiggerInputV2]:
    source = tmp_path / f"{stem}-source.usda"
    stage = Usd.Stage.CreateNew(str(source))
    assert stage is not None
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)

    body_paths = {endpoint for _, body0, body1 in edges for endpoint in (body0, body1)}
    for body_path in sorted(body_paths, key=lambda path: (path.count("/"), path)):
        if body_path == aggregate_body_path:
            continue
        body = UsdGeom.Xform.Define(stage, body_path)
        if any(
            body_path.startswith(f"{candidate}/")
            for candidate in body_paths
            if candidate != body_path
        ):
            body.SetResetXformStack(True)
        UsdGeom.Cube.Define(stage, f"{body_path}/collision")

    if aggregate_body_path is not None:
        assert aggregate_body_path == "/World/drawer"
        for panel in ("panel_a", "panel_b"):
            panel_path = f"/World/{panel}"
            UsdGeom.Xform.Define(stage, panel_path)
            UsdGeom.Cube.Define(stage, f"{panel_path}/collision")
    assert stage.GetRootLayer().Save()
    del stage

    source_artifact = identify_usd_artifact(source, uri=str(source))
    provenance = FieldProvenanceV1(
        source="owner_approved_plan",
        evidence=f"The owner approved the {stem} acceptance fixture.",
    )
    joints = tuple(
        JointPlanV1(
            topology=JointTopologyV1(
                joint_id=joint_id,
                joint_type="prismatic",
                body0=body0,
                body1=body1,
                axis_stage=(0.0, 0.0, 1.0),
                field_provenance=dict.fromkeys(
                    ("joint_type", "body0", "body1", "axis_stage"),
                    provenance,
                ),
            ),
            state=JointStateV1(
                position=0.0,
                velocity=0.0,
                provenance=provenance,
            ),
        )
        for joint_id, body0, body1 in edges
    )
    incoming = dict.fromkeys(body_paths, 0)
    for _, _, body1 in edges:
        incoming[body1] += 1
    root_paths = tuple(sorted(path for path, count in incoming.items() if count == 0))
    rigid_bodies = tuple(
        RigidBodyPlanV1(
            prim_path=body_path,
            mass=MassPropertiesV1(
                mass_kg=1.0,
                diagonal_inertia_kg_m2=(1.0, 1.0, 1.0),
                provenance=provenance,
            ),
            colliders=(
                tuple(
                    ColliderPlanV1(
                        prim_path=(f"{aggregate_body_path}/{panel}/collision"),
                        provenance=provenance,
                    )
                    for panel in ("panel_a", "panel_b")
                )
                if body_path == aggregate_body_path
                else (
                    ColliderPlanV1(
                        prim_path=f"{body_path}/collision",
                        provenance=provenance,
                    ),
                )
            ),
            provenance=provenance,
        )
        for body_path in sorted(body_paths)
    )
    plan = JointRiggerPlanV2(
        schema_version=PLAN_SCHEMA_VERSION_V2,
        joints=joints,
        rigid_bodies=rigid_bodies,
        articulation_roots=tuple(
            ArticulationRootPlanV1(
                prim_path=root_path,
                provenance=provenance,
            )
            for root_path in root_paths
        ),
    )
    links: list[RigidLinkPlanV1] = []
    for body_path in sorted(body_paths):
        link_id = body_path.removeprefix("/World/").replace("/", "__")
        if body_path == aggregate_body_path:
            links.append(
                RigidLinkPlanV1(
                    link_id=link_id,
                    body_authoring="aggregate",
                    body_prim_path=body_path,
                    members=tuple(
                        RigidLinkMemberPlanV1(
                            source_prim_path=f"/World/{panel}",
                            authored_prim_path=f"{body_path}/{panel}",
                        )
                        for panel in ("panel_a", "panel_b")
                    ),
                )
            )
        else:
            links.append(
                RigidLinkPlanV1(
                    link_id=link_id,
                    body_authoring="existing",
                    body_prim_path=body_path,
                    members=(
                        RigidLinkMemberPlanV1(
                            source_prim_path=body_path,
                            authored_prim_path=body_path,
                        ),
                    ),
                )
            )
    return source, JointRiggerInputV2(
        schema_version=INPUT_SCHEMA_VERSION_V2,
        source_asset=source_artifact,
        plan=plan,
        rigid_links=tuple(links),
    )


def _targets(tmp_path: Path, stem: str) -> JointRiggerArtifactTargets:
    return JointRiggerArtifactTargets(
        output_path=tmp_path / f"{stem}.usda",
        diagnostics_path=tmp_path / f"{stem}.diagnostics.json",
        result_path=tmp_path / f"{stem}.result.json",
    )


def _validate_topology_readback_with_r3_allowlists(
    stage: Any,
    request: JointRiggerInputV1,
    result: Any,
) -> None:
    """Exercise topology readback with the real R3-owned property boundary."""

    topology_plan = combined_module._topology_phase_plan(request.plan)
    preflight = validation_module._preflight_topology_authoring(
        stage,
        topology_plan,
        allow_existing_joint_paths=True,
    )
    resolved_plan = combined_module._resolve_physics_plan(
        request.plan,
        result.diagnostics,
    )
    contract = schemas_module._r3_raw_authorship_contract(
        stage,
        resolved_plan,
        schemas_module._preflight(stage, resolved_plan),
    )
    validation_module._validate_authored_preflight(
        stage,
        preflight,
        diagnostics=result.diagnostics,
        additional_allowed_applied_schemas={
            path: item.schema_tokens for path, item in contract.items()
        },
        additional_expected_applied_schema_order={
            path: item.schema_order for path, item in contract.items()
        },
        additional_allowed_authored_properties={
            path: item.authored_properties for path, item in contract.items()
        },
        additional_expected_attribute_specs={
            path: item.attribute_specs for path, item in contract.items()
        },
        additional_expected_relationship_targets={
            path: item.relationship_targets for path, item in contract.items()
        },
        plan_sha256_override=canonical_sha256(request.plan),
    )


def _request_with_drive(request: JointRiggerInputV1) -> JointRiggerInputV1:
    joint = request.plan.joints[0]
    assert joint.state is not None
    drive = JointDriveV1(
        drive_type="force",
        stiffness=20.0,
        damping=2.0,
        max_force=100.0,
        target_position=0.0,
        target_velocity=0.0,
        max_joint_velocity=3.0,
        provenance=joint.state.provenance,
    )
    plan = request.plan.model_copy(
        update={"joints": (joint.model_copy(update={"drive": drive}),)}
    )
    return request.model_copy(update={"plan": plan})


def _result_for_current_output(result: Any, output_path: Path) -> Any:
    assert result.output_artifact is not None
    current_identity = identify_usd_artifact(
        output_path,
        uri=result.output_artifact.uri,
    )
    return result.model_copy(update={"output_artifact": current_identity})


def _mimic_fixture(tmp_path: Path) -> tuple[Path, JointRiggerInputV1]:
    source = tmp_path / "mimic-source.usda"
    stage = Usd.Stage.CreateNew(str(source))
    assert stage is not None
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    bodies = ("/World/base", "/World/arm", "/World/hand")
    for index, body_path in enumerate(bodies):
        body = UsdGeom.Xform.Define(stage, body_path)
        body.AddTranslateOp().Set(Gf.Vec3d(float(index), 0.0, 0.0))
        evidence = body.GetPrim().CreateAttribute(
            "jointEvidence:axis",
            Sdf.ValueTypeNames.Float3,
            custom=True,
        )
        evidence.Set(Gf.Vec3f(0.0, 0.0, 1.0))
        UsdGeom.Cube.Define(stage, f"{body_path}/collision")
    assert stage.GetRootLayer().Save()
    del stage

    source_artifact = identify_usd_artifact(source, uri=str(source))
    provenance = FieldProvenanceV1(
        source="authored_metadata",
        artifact=source_artifact,
        prim_path="/World/arm",
        properties=("jointEvidence:axis",),
        evidence="The fixture authors the exact shared-axis evidence.",
    )
    joints: list[JointPlanV1] = []
    for joint_id, body0, body1 in (
        ("first", bodies[0], bodies[1]),
        ("second", bodies[1], bodies[2]),
    ):
        mimic = (
            JointMimicV1(
                reference_joint_id="first",
                gearing=-1.0,
                offset=0.0,
                natural_frequency=4.0,
                damping_ratio=0.7,
                provenance=provenance,
            )
            if joint_id == "second"
            else None
        )
        joints.append(
            JointPlanV1(
                topology=JointTopologyV1(
                    joint_id=joint_id,
                    joint_type="revolute",
                    body0=body0,
                    body1=body1,
                    axis_stage=(0.0, 0.0, 1.0),
                    field_provenance={
                        "joint_type": provenance,
                        "body0": provenance,
                        "body1": provenance,
                        "axis_stage": provenance,
                    },
                ),
                state=JointStateV1(
                    position=0.0,
                    velocity=0.0,
                    provenance=provenance,
                ),
                limit=JointLimitV1(
                    lower=-90.0,
                    upper=90.0,
                    unit="degrees",
                    provenance=provenance,
                ),
                mimic=mimic,
            )
        )
    body_plans = tuple(
        RigidBodyPlanV1(
            prim_path=body_path,
            mass=MassPropertiesV1(
                mass_kg=2.0,
                diagonal_inertia_kg_m2=(1.0, 1.0, 1.0),
                provenance=provenance,
            ),
            colliders=(
                ColliderPlanV1(
                    prim_path=f"{body_path}/collision",
                    provenance=provenance,
                ),
            ),
            provenance=provenance,
        )
        for body_path in bodies
    )
    plan = JointRiggerPlanV1(
        schema_version=PLAN_SCHEMA_VERSION,
        joints=tuple(joints),
        rigid_bodies=body_plans,
        articulation_root=ArticulationRootPlanV1(
            prim_path=bodies[0],
            provenance=provenance,
        ),
    )
    return source, JointRiggerInputV1(
        schema_version=INPUT_SCHEMA_VERSION,
        source_asset=source_artifact,
        plan=plan,
        conflict_policy="error",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
