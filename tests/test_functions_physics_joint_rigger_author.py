# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic fixture coverage for the owned shared topology author."""

from __future__ import annotations

import json
import math
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("pxr")
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from world_understanding.functions.physics.joint_rigger import (
    INPUT_SCHEMA_VERSION,
    PLAN_SCHEMA_VERSION,
    ArtifactIdentityV1,
    FieldProvenanceV1,
    JointAnchorV1,
    JointFrictionV1,
    JointLimitV1,
    JointPlanV1,
    JointRiggerArtifactError,
    JointRiggerArtifactTargets,
    JointRiggerBackendIncompatibleError,
    JointRiggerContractError,
    JointRiggerInputV1,
    JointRiggerPlanV1,
    JointStateV1,
    JointTopologyV1,
    LegacyComponentAssignmentV1,
    LegacyComponentNameCompatibilityV1,
    OwnedTopologyBackend,
    RigidBodyPlanV1,
    author_joint_rig,
    author_joint_topology,
    canonical_json,
    canonical_sha256,
    extract_reference_input,
    identify_usd_artifact,
    validate_authored_joint_topology,
    validate_joint_topology_plan,
)
from world_understanding.functions.physics.joint_rigger import (
    artifacts as artifacts_module,
)
from world_understanding.functions.physics.joint_rigger import author as author_module
from world_understanding.functions.physics.joint_rigger import (
    validation as validation_module,
)

type _Vector3 = tuple[float, float, float]


def _create_source_stage(
    path: Path,
    body_paths: tuple[str, ...],
    *,
    meters_per_unit: float = 1.0,
    rotations: Mapping[str, tuple[str, float]] | None = None,
    translations: Mapping[str, _Vector3] | None = None,
) -> None:
    stage = Usd.Stage.CreateNew(str(path))
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage, meters_per_unit)
    rotations = rotations or {}
    translations = translations or {}
    for body_path in body_paths:
        body = UsdGeom.Xform.Define(stage, body_path)
        if body_path in translations:
            body.AddTranslateOp().Set(Gf.Vec3d(*translations[body_path]))
        rotation = rotations.get(body_path)
        if rotation is not None:
            axis, degrees = rotation
            getattr(body, f"AddRotate{axis.upper()}Op")().Set(degrees)
        UsdGeom.Cube.Define(stage, f"{body_path}/visual")
    assert stage.GetRootLayer().Save()


def _add_native_instance_with_hidden_joint(stage: Any, asset_path: Path) -> None:
    asset = Usd.Stage.CreateNew(str(asset_path))
    model = UsdGeom.Xform.Define(asset, "/Model")
    asset.SetDefaultPrim(model.GetPrim())
    UsdGeom.Xform.Define(asset, "/Model/Base")
    UsdGeom.Xform.Define(asset, "/Model/Link")
    hidden = UsdPhysics.RevoluteJoint.Define(asset, "/Model/HiddenJoint")
    assert hidden.CreateBody0Rel().SetTargets([Sdf.Path("/Model/Base")])
    assert hidden.CreateBody1Rel().SetTargets([Sdf.Path("/Model/Link")])
    assert asset.GetRootLayer().Save()
    del asset

    instance = stage.OverridePrim("/World/InstancedRig")
    assert instance.GetReferences().AddReference(str(asset_path))
    assert instance.SetInstanceable(True)
    proxy = stage.GetPrimAtPath("/World/InstancedRig/HiddenJoint")
    assert proxy.IsValid()
    assert proxy.IsInstanceProxy()
    assert proxy.IsA(UsdPhysics.Joint)
    assert not any(
        prim.IsA(UsdPhysics.Joint) and prim.IsInstanceProxy()
        for prim in stage.TraverseAll()
    )
    assert any(
        prim.IsA(UsdPhysics.Joint)
        for prototype in stage.GetPrototypes()
        for prim in Usd.PrimRange.AllPrims(prototype)
    )


def _author_reference_joint(
    stage: Any,
    *,
    index: int,
    joint_id: str,
    joint_type: str,
    body0: str,
    body1: str,
    axis_stage: _Vector3 | None,
    anchor_stage: _Vector3,
    lower: float | None = None,
    upper: float | None = None,
    drive: bool = False,
    joint_friction: float | None = None,
) -> None:
    scope = UsdGeom.Scope.Define(stage, "/World/ReferenceJoints")
    assert scope.GetPrim().IsValid()
    schema = {
        "revolute": UsdPhysics.RevoluteJoint,
        "prismatic": UsdPhysics.PrismaticJoint,
        "spherical": UsdPhysics.SphericalJoint,
    }[joint_type]
    path = Sdf.Path(f"/World/ReferenceJoints/joint_{index}")
    joint = schema.Define(stage, path)
    assert joint.CreateBody0Rel().SetTargets([Sdf.Path(body0)])
    assert joint.CreateBody1Rel().SetTargets([Sdf.Path(body1)])
    body0_prim = stage.GetPrimAtPath(body0)
    body1_prim = stage.GetPrimAtPath(body1)
    cache = UsdGeom.XformCache()
    body0_xform = cache.GetLocalToWorldTransform(body0_prim)
    body1_xform = cache.GetLocalToWorldTransform(body1_prim)
    anchor = Gf.Vec3d(*anchor_stage)
    assert joint.CreateLocalPos0Attr().Set(
        Gf.Vec3f(*body0_xform.GetInverse().Transform(anchor))
    )
    assert joint.CreateLocalPos1Attr().Set(
        Gf.Vec3f(*body1_xform.GetInverse().Transform(anchor))
    )
    if axis_stage is not None:
        axis_token, _ = _axis_token(axis_stage)
        assert joint.CreateAxisAttr().Set(axis_token)
        world_axis = Gf.Vec3d(*axis_stage)
        stage_frame = validation_module._stage_joint_frame(world_axis, Gf=Gf)
        local_rot0 = validation_module._local_joint_frame_rotation(
            body0_xform,
            stage_frame=stage_frame,
            axis_token=axis_token,
            label=f"reference joint {joint_id!r} body0",
            Gf=Gf,
        )
        local_rot1 = validation_module._local_joint_frame_rotation(
            body1_xform,
            stage_frame=stage_frame,
            axis_token=axis_token,
            label=f"reference joint {joint_id!r} body1",
            Gf=Gf,
        )
        assert joint.CreateLocalRot0Attr().Set(
            Gf.Quatf(local_rot0[0], Gf.Vec3f(*local_rot0[1]))
        )
        assert joint.CreateLocalRot1Attr().Set(
            Gf.Quatf(local_rot1[0], Gf.Vec3f(*local_rot1[1]))
        )
    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    if lower is not None:
        authored_lower = lower / meters_per_unit if joint_type == "prismatic" else lower
        assert joint.CreateLowerLimitAttr().Set(float(authored_lower))
    if upper is not None:
        authored_upper = upper / meters_per_unit if joint_type == "prismatic" else upper
        assert joint.CreateUpperLimitAttr().Set(float(authored_upper))
    if drive:
        instance = "angular" if joint_type == "revolute" else "linear"
        drive_api = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), instance)
        assert drive_api
        assert drive_api.CreateTypeAttr().Set("force")
        assert drive_api.CreateStiffnessAttr().Set(12.0)
        assert drive_api.CreateDampingAttr().Set(3.0)
        assert drive_api.CreateMaxForceAttr().Set(40.0)
        assert drive_api.CreateTargetPositionAttr().Set(5.0)
        assert drive_api.CreateTargetVelocityAttr().Set(0.5)
    if joint_friction is not None:
        assert joint.GetPrim().AddAppliedSchema("PhysxJointAPI")
        assert (
            joint.GetPrim()
            .CreateAttribute(
                "physxJoint:jointFriction",
                Sdf.ValueTypeNames.Float,
                custom=False,
            )
            .Set(joint_friction)
        )
    joint.GetPrim().SetCustomDataByKey("fixture:jointId", joint_id)


def _axis_token(axis: _Vector3) -> tuple[str, Any]:
    cardinal = {
        "X": (1.0, 0.0, 0.0),
        "Y": (0.0, 1.0, 0.0),
        "Z": (0.0, 0.0, 1.0),
    }
    for token, base in cardinal.items():
        if tuple(abs(value) for value in axis) == base:
            return token, Gf.Vec3d(*base)
    return "X", Gf.Vec3d(1.0, 0.0, 0.0)


def _paired_fixture(
    tmp_path: Path,
    *,
    name: str,
    body_paths: tuple[str, ...],
    joints: tuple[dict[str, Any], ...],
    meters_per_unit: float = 1.0,
    rotations: Mapping[str, tuple[str, float]] | None = None,
    translations: Mapping[str, _Vector3] | None = None,
) -> tuple[Path, Path, JointRiggerInputV1]:
    source = tmp_path / f"{name}-source.usda"
    reference = tmp_path / f"{name}-reference.usda"
    _create_source_stage(
        source,
        body_paths,
        meters_per_unit=meters_per_unit,
        rotations=rotations,
        translations=translations,
    )
    shutil.copy2(source, reference)
    stage = Usd.Stage.Open(str(reference))
    assert stage is not None
    for index, joint in enumerate(joints):
        _author_reference_joint(stage, index=index, **joint)
    assert stage.GetRootLayer().Save()
    del stage
    request = extract_reference_input(
        source,
        reference,
        source_uri=str(source),
        reference_uri=f"fixture://{name}/rigged.usda",
    )
    reference.unlink()
    return source, reference, request


def _targets(tmp_path: Path, name: str) -> JointRiggerArtifactTargets:
    return JointRiggerArtifactTargets(
        output_path=tmp_path / f"{name}.usda",
        diagnostics_path=tmp_path / f"{name}.diagnostics.json",
        result_path=tmp_path / f"{name}.result.json",
    )


def _source_prim_snapshot(stage: Any) -> dict[str, tuple[Any, ...]]:
    cache = UsdGeom.XformCache()
    snapshot = {}
    for prim in stage.TraverseAll():
        xformable = UsdGeom.Xformable(prim)
        matrix = None
        if xformable:
            transform = cache.GetLocalToWorldTransform(prim)
            matrix = tuple(
                float(transform[row][column]) for row in range(4) for column in range(4)
            )
        snapshot[str(prim.GetPath())] = (
            str(prim.GetTypeName()),
            tuple(str(item) for item in prim.GetAppliedSchemas()),
            tuple(sorted(str(prop.GetName()) for prop in prim.GetAuthoredProperties())),
            bool(prim.IsInstanceable()),
            matrix,
        )
    return snapshot


def _authored_graph(stage: Any) -> dict[str, tuple[Any, ...]]:
    cache = UsdGeom.XformCache()
    graph: dict[str, tuple[Any, ...]] = {}
    for prim in stage.TraverseAll():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint_type = next(
            name
            for name, schema in (
                ("revolute", UsdPhysics.RevoluteJoint),
                ("prismatic", UsdPhysics.PrismaticJoint),
                ("spherical", UsdPhysics.SphericalJoint),
            )
            if prim.IsA(schema)
        )
        joint = {
            "revolute": UsdPhysics.RevoluteJoint,
            "prismatic": UsdPhysics.PrismaticJoint,
            "spherical": UsdPhysics.SphericalJoint,
        }[joint_type](prim)
        body0 = str(joint.GetBody0Rel().GetTargets()[0])
        body1 = str(joint.GetBody1Rel().GetTargets()[0])
        axis = None
        if joint_type != "spherical":
            token = str(joint.GetAxisAttr().Get())
            base = {
                "X": Gf.Vec3d(1.0, 0.0, 0.0),
                "Y": Gf.Vec3d(0.0, 1.0, 0.0),
                "Z": Gf.Vec3d(0.0, 0.0, 1.0),
            }[token]
            local = Gf.Rotation(joint.GetLocalRot0Attr().Get()).TransformDir(base)
            world = cache.GetLocalToWorldTransform(
                stage.GetPrimAtPath(body0)
            ).TransformDir(local)
            world.Normalize()
            axis = tuple(float(value) for value in world)
        graph[str(prim.GetCustomDataByKey("jointRigger:jointId"))] = (
            joint_type,
            body0,
            body1,
            axis,
        )
    return graph


@pytest.mark.parametrize(
    "fixture_name",
    ["drawer", "hinge", "file_cabinet", "caster", "trolley"],
)
def test_fixture_ladder_matches_reference_graph_without_reshaping(
    tmp_path: Path,
    fixture_name: str,
) -> None:
    if fixture_name == "drawer":
        bodies = ("/World/Case", "/World/Drawer")
        joints = (
            {
                "joint_id": "drawer-slide",
                "joint_type": "prismatic",
                "body0": bodies[0],
                "body1": bodies[1],
                "axis_stage": (1.0, 0.0, 0.0),
                "anchor_stage": (25.0, 0.0, 0.0),
                "lower": 0.0,
                "upper": 0.5,
            },
        )
        meters_per_unit = 0.01
        rotations = None
        translations = {bodies[1]: (25.0, 0.0, 0.0)}
    elif fixture_name == "hinge":
        bodies = ("/World/Frame", "/World/Door")
        joints = (
            {
                "joint_id": "door-hinge",
                "joint_type": "revolute",
                "body0": bodies[0],
                "body1": bodies[1],
                "axis_stage": (0.0, 0.0, -1.0),
                "anchor_stage": (0.0, 2.0, 1.0),
                "lower": -10.0,
                "upper": 100.0,
                "drive": True,
            },
        )
        meters_per_unit = 1.0
        rotations = {bodies[0]: ("Y", 20.0), bodies[1]: ("X", -35.0)}
        translations = {bodies[1]: (0.0, 2.0, 0.0)}
    elif fixture_name == "file_cabinet":
        bodies = (
            "/World/Cabinet",
            "/World/Cabinet/Drawer_1",
            "/World/Cabinet/Drawer_2",
            "/World/Cabinet/Drawer_3",
        )
        joints = tuple(
            {
                "joint_id": f"file drawer {index}",
                "joint_type": "prismatic",
                "body0": bodies[0],
                "body1": body,
                "axis_stage": (0.0, -1.0, 0.0),
                "anchor_stage": (0.0, float(index), 0.0),
            }
            for index, body in enumerate(bodies[1:], start=1)
        )
        meters_per_unit = 1.0
        rotations = None
        translations = {
            body: (0.0, float(index), 0.0)
            for index, body in enumerate(bodies[1:], start=1)
        }
    elif fixture_name == "caster":
        bodies = (
            "/World/Cart",
            "/World/Cart/CasterFrame",
            "/World/Cart/CasterFrame/Wheel",
        )
        joints = (
            {
                "joint_id": "caster swivel",
                "joint_type": "revolute",
                "body0": bodies[0],
                "body1": bodies[1],
                "axis_stage": (0.0, 0.0, 1.0),
                "anchor_stage": (1.0, 0.0, 0.0),
            },
            {
                "joint_id": "wheel roll",
                "joint_type": "revolute",
                "body0": bodies[1],
                "body1": bodies[2],
                "axis_stage": (-1.0, 0.0, 0.0),
                "anchor_stage": (1.0, 0.0, -1.0),
            },
        )
        meters_per_unit = 1.0
        rotations = {bodies[1]: ("Z", 30.0), bodies[2]: ("Y", 45.0)}
        translations = {bodies[1]: (1.0, 0.0, 0.0), bodies[2]: (0.0, 0.0, -1.0)}
    else:
        bodies = (
            "/World/Trolley",
            "/World/Trolley/TableSlide",
            "/World/Trolley/CasterFrame",
            "/World/Trolley/CasterFrame/Wheel",
        )
        joints = (
            {
                "joint_id": "table slide",
                "joint_type": "prismatic",
                "body0": bodies[0],
                "body1": bodies[1],
                "axis_stage": (0.0, 1.0, 0.0),
                "anchor_stage": (0.0, 0.5, 1.0),
                "lower": 0.0,
                "upper": 0.4,
            },
            {
                "joint_id": "trolley caster swivel",
                "joint_type": "revolute",
                "body0": bodies[0],
                "body1": bodies[2],
                "axis_stage": (0.0, 0.0, 1.0),
                "anchor_stage": (1.0, 0.0, 0.0),
            },
            {
                "joint_id": "trolley wheel roll",
                "joint_type": "revolute",
                "body0": bodies[2],
                "body1": bodies[3],
                "axis_stage": (1.0, 0.0, 0.0),
                "anchor_stage": (1.0, 0.0, -1.0),
            },
        )
        meters_per_unit = 1.0
        rotations = {bodies[2]: ("Z", 15.0), bodies[3]: ("Y", -20.0)}
        translations = {
            bodies[1]: (0.0, 0.5, 1.0),
            bodies[2]: (1.0, 0.0, 0.0),
            bodies[3]: (0.0, 0.0, -1.0),
        }

    source, reference, request = _paired_fixture(
        tmp_path,
        name=fixture_name,
        body_paths=bodies,
        joints=joints,
        meters_per_unit=meters_per_unit,
        rotations=rotations,
        translations=translations,
    )
    source_bytes = source.read_bytes()
    source_stage = Usd.Stage.Open(str(source))
    assert source_stage is not None
    source_snapshot = _source_prim_snapshot(source_stage)
    targets = _targets(tmp_path, f"{fixture_name}-output")

    result = author_joint_topology(
        request,
        source_usd_path=source,
        artifact_targets=targets,
    )

    assert result.status == "succeeded"
    assert source.read_bytes() == source_bytes
    output_stage = Usd.Stage.Open(str(targets.output_path))
    assert output_stage is not None
    validate_authored_joint_topology(output_stage, request.plan, result.diagnostics)
    replayed = extract_reference_input(
        source,
        targets.output_path,
        source_uri=str(source),
        reference_uri=str(targets.output_path),
    )
    expected_topologies = sorted(
        (
            joint.topology.joint_type,
            joint.topology.body0,
            joint.topology.body1,
            joint.topology.axis_stage,
        )
        for joint in request.plan.joints
    )
    replayed_topologies = sorted(
        (
            joint.topology.joint_type,
            joint.topology.body0,
            joint.topology.body1,
            joint.topology.axis_stage,
        )
        for joint in replayed.plan.joints
    )
    assert len(replayed_topologies) == len(expected_topologies)
    for replayed_topology, expected_topology in zip(
        replayed_topologies,
        expected_topologies,
        strict=True,
    ):
        assert replayed_topology[:3] == expected_topology[:3]
        if expected_topology[3] is None:
            assert replayed_topology[3] is None
        else:
            assert replayed_topology[3] == pytest.approx(
                expected_topology[3],
                abs=1e-5,
            )
    observed_graph = _authored_graph(output_stage)
    expected_graph = {
        joint.topology.joint_id: (
            joint.topology.joint_type,
            joint.topology.body0,
            joint.topology.body1,
            joint.topology.axis_stage,
        )
        for joint in request.plan.joints
    }
    assert set(observed_graph) == set(expected_graph)
    for joint_id, expected in expected_graph.items():
        observed = observed_graph[joint_id]
        assert observed[:3] == expected[:3]
        if expected[3] is None:
            assert observed[3] is None
        else:
            assert observed[3] == pytest.approx(expected[3], abs=1e-5)
    output_snapshot = _source_prim_snapshot(output_stage)
    assert {path: output_snapshot[path] for path in source_snapshot} == source_snapshot
    assert all(
        diagnostic.authored_prim_path is not None
        and diagnostic.authored_prim_path.startswith("/World/Joints/")
        for diagnostic in result.diagnostics.joint_diagnostics
    )
    expected_diagnostic_fields = {
        "topology.joint_type",
        "topology.body0",
        "topology.body1",
        "topology.axis_stage",
        "limit.lower",
        "limit.upper",
        "limit.unit",
        "anchor.position_stage",
        "joint_friction.coefficient",
        "drive.drive_type",
        "drive.stiffness",
        "drive.damping",
        "drive.max_force",
        "drive.target_position",
        "drive.target_velocity",
        "drive.max_joint_velocity",
        "state",
        "mimic",
        "usd.joint_prim_path",
        "usd.local_frames",
    }
    for diagnostic in result.diagnostics.joint_diagnostics:
        assert {
            decision.field for decision in diagnostic.field_decisions
        } == expected_diagnostic_fields
    assert {
        decision.field: (decision.disposition, decision.reason_code)
        for decision in result.diagnostics.field_decisions
    } == {
        "articulation_root": ("ignored", "not_provided"),
        "legacy_component_names": (
            "ignored",
            "legacy_component_name_compatibility_not_requested",
        ),
        "rigid_bodies": ("ignored", "not_provided"),
    }
    assert not reference.exists()  # Offline evidence is never a runtime input.


def test_source_backed_limits_anchor_and_drive_are_preserved(
    tmp_path: Path,
) -> None:
    bodies = ("/World/Frame", "/World/Door")
    source, _, request = _paired_fixture(
        tmp_path,
        name="driven-hinge",
        body_paths=bodies,
        joints=(
            {
                "joint_id": "driven hinge",
                "joint_type": "revolute",
                "body0": bodies[0],
                "body1": bodies[1],
                "axis_stage": (0.0, 0.0, -1.0),
                "anchor_stage": (1.0, 2.0, 3.0),
                "lower": -15.0,
                "upper": 95.0,
                "drive": True,
                "joint_friction": 0.2,
            },
        ),
        rotations={bodies[1]: ("Y", 25.0)},
        translations={bodies[1]: (1.0, 0.0, 0.0)},
    )
    joint = request.plan.joints[0]
    assert joint.limit is not None
    assert joint.anchor is not None
    assert joint.drive is not None
    assert joint.joint_friction is not None
    drive = joint.drive.model_copy(update={"max_joint_velocity": 33.0})
    updated_joint = joint.model_copy(update={"drive": drive})
    request = request.model_copy(
        update={"plan": request.plan.model_copy(update={"joints": (updated_joint,)})}
    )
    targets = _targets(tmp_path, "driven-hinge-output")

    result = author_joint_topology(
        request,
        source_usd_path=source,
        artifact_targets=targets,
    )

    diagnostic = result.diagnostics.joint_diagnostics[0]
    assert diagnostic.authored_prim_path is not None
    decisions = {item.field: item for item in diagnostic.field_decisions}
    expected_provenance = {
        **{
            f"topology.{field}": provenance
            for field, provenance in joint.topology.field_provenance.items()
        },
        **dict.fromkeys(
            ("limit.lower", "limit.upper", "limit.unit"),
            joint.limit.provenance,
        ),
        "anchor.position_stage": joint.anchor.provenance,
        "joint_friction.coefficient": joint.joint_friction.provenance,
        **{
            f"drive.{field}": drive.provenance
            for field in (
                "drive_type",
                "stiffness",
                "damping",
                "max_force",
                "target_position",
                "target_velocity",
                "max_joint_velocity",
            )
        },
    }
    accepted = {
        field: decision.provenance
        for field, decision in decisions.items()
        if decision.disposition == "accepted"
    }
    assert accepted == expected_provenance
    stage = Usd.Stage.Open(str(targets.output_path))
    assert stage is not None
    prim = stage.GetPrimAtPath(diagnostic.authored_prim_path)
    authored = UsdPhysics.RevoluteJoint(prim)
    assert authored.GetLowerLimitAttr().Get() == pytest.approx(-15.0)
    assert authored.GetUpperLimitAttr().Get() == pytest.approx(95.0)
    drive_api = UsdPhysics.DriveAPI.Get(prim, "angular")
    assert drive_api.GetTypeAttr().Get() == "force"
    assert drive_api.GetStiffnessAttr().Get() == pytest.approx(12.0)
    assert drive_api.GetDampingAttr().Get() == pytest.approx(3.0)
    assert drive_api.GetMaxForceAttr().Get() == pytest.approx(40.0)
    assert drive_api.GetTargetPositionAttr().Get() == pytest.approx(5.0)
    assert drive_api.GetTargetVelocityAttr().Get() == pytest.approx(0.5)
    assert prim.GetAttribute("physxJoint:maxJointVelocity").Get() == pytest.approx(33.0)
    assert prim.GetAttribute("physxJoint:jointFriction").Get() == pytest.approx(0.2)
    assert "PhysxJointAPI" in prim.GetMetadata("apiSchemas").GetAppliedItems()
    assert json.loads(str(prim.GetCustomDataByKey("jointRigger:fieldDecisions"))) == [
        decision.model_dump(mode="json", exclude_none=True)
        for decision in diagnostic.field_decisions
    ]


@pytest.mark.parametrize(
    ("joint_type", "axis_stage"),
    [
        ("revolute", (0.0, 0.0, 1.0)),
        ("prismatic", (1.0, 0.0, 0.0)),
    ],
)
def test_passive_scalar_joint_friction_authors_without_a_drive(
    tmp_path: Path,
    joint_type: str,
    axis_stage: _Vector3,
) -> None:
    bodies = ("/World/Base", "/World/Link")
    source, _, request = _paired_fixture(
        tmp_path,
        name=f"passive-{joint_type}-friction",
        body_paths=bodies,
        joints=(
            {
                "joint_id": f"passive {joint_type}",
                "joint_type": joint_type,
                "body0": bodies[0],
                "body1": bodies[1],
                "axis_stage": axis_stage,
                "anchor_stage": (0.0, 0.0, 0.0),
                "joint_friction": 0.0,
            },
        ),
    )
    planned_joint = request.plan.joints[0]
    assert planned_joint.drive is None
    assert planned_joint.joint_friction is not None
    targets = _targets(tmp_path, f"passive-{joint_type}-friction-output")

    result = author_joint_topology(
        request,
        source_usd_path=source,
        artifact_targets=targets,
    )

    diagnostic = result.diagnostics.joint_diagnostics[0]
    assert diagnostic.authored_prim_path is not None
    output_stage = Usd.Stage.Open(str(targets.output_path))
    assert output_stage is not None
    prim = output_stage.GetPrimAtPath(diagnostic.authored_prim_path)
    schemas = validation_module._applied_schema_tokens(prim)
    assert schemas == {"PhysxJointAPI"}
    assert prim.GetAttribute("physxJoint:jointFriction").Get() == 0.0
    assert not any(token.startswith("PhysicsDriveAPI:") for token in schemas)
    validate_authored_joint_topology(
        stage=output_stage,
        plan=request.plan,
        diagnostics=result.diagnostics,
    )


@pytest.mark.parametrize(
    "tamper",
    ["value", "missing", "time_sample", "unknown_property", "unknown_schema"],
)
def test_joint_friction_readback_fails_closed_on_authored_physx_drift(
    tmp_path: Path,
    tamper: str,
) -> None:
    bodies = ("/World/Base", "/World/Link")
    source, _, request = _paired_fixture(
        tmp_path,
        name=f"friction-readback-{tamper}",
        body_paths=bodies,
        joints=(
            {
                "joint_id": "friction readback",
                "joint_type": "revolute",
                "body0": bodies[0],
                "body1": bodies[1],
                "axis_stage": (0.0, 0.0, 1.0),
                "anchor_stage": (0.0, 0.0, 0.0),
                "joint_friction": 0.15,
            },
        ),
    )
    targets = _targets(tmp_path, f"friction-readback-{tamper}-output")
    result = author_joint_topology(
        request,
        source_usd_path=source,
        artifact_targets=targets,
    )
    path = result.diagnostics.joint_diagnostics[0].authored_prim_path
    assert path is not None
    stage = Usd.Stage.Open(str(targets.output_path))
    assert stage is not None
    prim = stage.GetPrimAtPath(path)
    friction = prim.GetAttribute("physxJoint:jointFriction")
    if tamper == "value":
        assert friction.Set(0.25)
    elif tamper == "missing":
        assert prim.RemoveProperty("physxJoint:jointFriction")
    elif tamper == "time_sample":
        assert friction.Set(0.15, Usd.TimeCode(1.0))
    elif tamper == "unknown_property":
        assert prim.CreateAttribute(
            "physxJoint:solverFoo",
            Sdf.ValueTypeNames.Float,
            custom=False,
        ).Set(1.0)
    else:
        assert prim.AddAppliedSchema("PhysxJointAPI:rogue")

    with pytest.raises(JointRiggerContractError) as caught:
        validate_authored_joint_topology(stage, request.plan, result.diagnostics)

    assert caught.value.code == "authored_graph_mismatch"


def test_facade_publishes_bound_output_and_exact_reports(tmp_path: Path) -> None:
    source = tmp_path / "integration-source.usda"
    bodies = ("/World/Base", "/World/Link")
    _create_source_stage(source, bodies)
    request = _simple_request(source, body0=bodies[0], body1=bodies[1])
    targets = _targets(tmp_path, "integration-output")
    for path in (
        targets.output_path,
        targets.diagnostics_path,
        targets.result_path,
    ):
        path.write_text("previous artifact", encoding="utf-8")

    result = author_joint_topology(
        request,
        source_usd_path=source,
        artifact_targets=targets,
    )

    assert result.input_sha256 == canonical_sha256(request)
    assert result.plan_sha256 == canonical_sha256(request.plan)
    assert result.output_artifact == identify_usd_artifact(
        targets.output_path,
        uri=str(targets.output_path),
    )
    assert targets.diagnostics_path.read_text(encoding="utf-8") == canonical_json(
        result.diagnostics
    )
    assert targets.result_path.read_text(encoding="utf-8") == canonical_json(result)
    stage = Usd.Stage.Open(str(targets.output_path))
    assert stage is not None
    validate_authored_joint_topology(stage, request.plan, result.diagnostics)
    assert not any(tmp_path.glob(".*.stage-*"))
    assert not any(tmp_path.glob(".*.rollback-*"))
    assert not any(tmp_path.glob(".*.sealed-*"))


def test_facade_publication_failure_restores_previous_complete_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "publication-rollback-source.usda"
    bodies = ("/World/Base", "/World/Link")
    _create_source_stage(source, bodies)
    request = _simple_request(source, body0=bodies[0], body1=bodies[1])
    targets = _targets(tmp_path, "publication-rollback-output")
    previous = {
        targets.output_path: b"previous output",
        targets.diagnostics_path: b"previous diagnostics",
        targets.result_path: b"previous result",
    }
    for path, payload in previous.items():
        path.write_bytes(payload)
    source_bytes = source.read_bytes()
    original_replace = artifacts_module._replace_entry
    failed = False

    def fail_result_promotion(source_entry: Any, target_entry: Any) -> None:
        nonlocal failed
        if not failed and target_entry.path == targets.result_path:
            failed = True
            raise OSError("forced owned-topology result promotion failure")
        original_replace(source_entry, target_entry)

    monkeypatch.setattr(artifacts_module, "_replace_entry", fail_result_promotion)

    with pytest.raises(
        JointRiggerArtifactError,
        match="forced owned-topology result promotion failure",
    ):
        author_joint_topology(
            request,
            source_usd_path=source,
            artifact_targets=targets,
        )

    assert failed
    assert source.read_bytes() == source_bytes
    for path, payload in previous.items():
        assert path.read_bytes() == payload
    assert not any(tmp_path.glob(".*.stage-*"))
    assert not any(tmp_path.glob(".*.rollback-*"))
    assert not any(tmp_path.glob(".*.sealed-*"))


def test_prismatic_meters_convert_to_stage_units_and_names_are_stable(
    tmp_path: Path,
) -> None:
    bodies = ("/World/Case", "/World/Drawer")
    source, _, request = _paired_fixture(
        tmp_path,
        name="centimeter-drawer",
        body_paths=bodies,
        joints=(
            {
                "joint_id": "drawer 01",
                "joint_type": "prismatic",
                "body0": bodies[0],
                "body1": bodies[1],
                "axis_stage": (1.0, 0.0, 0.0),
                "anchor_stage": (0.0, 0.0, 0.0),
                "lower": 0.0,
                "upper": 0.5,
            },
        ),
        meters_per_unit=0.01,
    )
    first = _targets(tmp_path, "drawer-first")
    second = _targets(tmp_path, "drawer-second")

    first_result = author_joint_topology(
        request,
        source_usd_path=source,
        artifact_targets=first,
    )
    second_result = author_joint_topology(
        request,
        source_usd_path=source,
        artifact_targets=second,
    )

    first_path = first_result.diagnostics.joint_diagnostics[0].authored_prim_path
    second_path = second_result.diagnostics.joint_diagnostics[0].authored_prim_path
    assert first_path == second_path
    assert first_path is not None
    stage = Usd.Stage.Open(str(first.output_path))
    assert stage is not None
    joint = UsdPhysics.PrismaticJoint(stage.GetPrimAtPath(first_path))
    assert joint.GetLowerLimitAttr().Get() == pytest.approx(0.0)
    assert joint.GetUpperLimitAttr().Get() == pytest.approx(50.0)


def test_prismatic_limit_conversion_overflow_fails_before_publication(
    tmp_path: Path,
) -> None:
    source = tmp_path / "tiny-units.usda"
    bodies = ("/World/Case", "/World/Drawer")
    _create_source_stage(source, bodies, meters_per_unit=1e-320)
    identity = identify_usd_artifact(source, uri=str(source))
    topology = JointTopologyV1(
        joint_id="overflowing-drawer",
        joint_type="prismatic",
        body0=bodies[0],
        body1=bodies[1],
        axis_stage=(1.0, 0.0, 0.0),
        field_provenance={
            field: _source_provenance(identity, field=field, prim_path=bodies[1])
            for field in ("joint_type", "body0", "body1", "axis_stage")
        },
    )
    limit = JointLimitV1(
        lower=0.0,
        upper=1.0,
        unit="meters",
        provenance=_source_provenance(
            identity,
            field="limit",
            prim_path=bodies[1],
        ),
    )
    request = JointRiggerInputV1(
        schema_version=INPUT_SCHEMA_VERSION,
        source_asset=identity,
        plan=JointRiggerPlanV1(
            schema_version=PLAN_SCHEMA_VERSION,
            joints=(JointPlanV1(topology=topology, limit=limit),),
        ),
    )
    targets = _targets(tmp_path, "overflow-output")

    with pytest.raises(JointRiggerContractError) as error:
        author_joint_topology(
            request,
            source_usd_path=source,
            artifact_targets=targets,
        )

    assert error.value.code == "authored_value_out_of_range"
    assert not targets.output_path.exists()
    assert not targets.diagnostics_path.exists()
    assert not targets.result_path.exists()


def test_float32_underflow_fails_before_publication(tmp_path: Path) -> None:
    bodies = ("/World/Frame", "/World/Door")
    source, _, request = _paired_fixture(
        tmp_path,
        name="underflowing-drive",
        body_paths=bodies,
        joints=(
            {
                "joint_id": "underflowing drive",
                "joint_type": "revolute",
                "body0": bodies[0],
                "body1": bodies[1],
                "axis_stage": (0.0, 0.0, 1.0),
                "anchor_stage": (0.0, 0.0, 0.0),
                "drive": True,
            },
        ),
    )
    planned_joint = request.plan.joints[0]
    assert planned_joint.drive is not None
    underflowing_drive = planned_joint.drive.model_copy(update={"stiffness": 1e-50})
    request = request.model_copy(
        update={
            "plan": request.plan.model_copy(
                update={
                    "joints": (
                        planned_joint.model_copy(update={"drive": underflowing_drive}),
                    )
                }
            )
        }
    )
    targets = _targets(tmp_path, "underflowing-drive-output")

    with pytest.raises(JointRiggerContractError) as error:
        author_joint_topology(
            request,
            source_usd_path=source,
            artifact_targets=targets,
        )

    assert error.value.code == "authored_value_out_of_range"
    assert "does not survive USD float32 storage" in str(error.value)
    assert not targets.output_path.exists()
    assert not targets.diagnostics_path.exists()
    assert not targets.result_path.exists()


def test_representable_small_float32_value_round_trips_exactly(
    tmp_path: Path,
) -> None:
    bodies = ("/World/Frame", "/World/Door")
    source, _, request = _paired_fixture(
        tmp_path,
        name="small-representable-drive",
        body_paths=bodies,
        joints=(
            {
                "joint_id": "small representable drive",
                "joint_type": "revolute",
                "body0": bodies[0],
                "body1": bodies[1],
                "axis_stage": (0.0, 0.0, 1.0),
                "anchor_stage": (0.0, 0.0, 0.0),
                "drive": True,
            },
        ),
    )
    planned_joint = request.plan.joints[0]
    assert planned_joint.drive is not None
    small_drive = planned_joint.drive.model_copy(update={"target_velocity": 1e-20})
    request = request.model_copy(
        update={
            "plan": request.plan.model_copy(
                update={
                    "joints": (planned_joint.model_copy(update={"drive": small_drive}),)
                }
            )
        }
    )
    targets = _targets(tmp_path, "small-representable-drive-output")
    result = author_joint_topology(
        request,
        source_usd_path=source,
        artifact_targets=targets,
    )
    path = result.diagnostics.joint_diagnostics[0].authored_prim_path
    assert path is not None
    stage = Usd.Stage.Open(str(targets.output_path))
    assert stage is not None
    drive = UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath(path), "angular")
    target_velocity = drive.GetTargetVelocityAttr()
    assert target_velocity.Get() != 0.0
    validate_authored_joint_topology(stage, request.plan, result.diagnostics)

    assert target_velocity.Set(0.0)
    with pytest.raises(JointRiggerContractError) as error:
        validate_authored_joint_topology(stage, request.plan, result.diagnostics)
    assert error.value.code == "authored_graph_mismatch"


def test_non_cardinal_signed_axis_and_default_anchor_are_diagnosed(
    tmp_path: Path,
) -> None:
    source = tmp_path / "diagonal-source.usda"
    bodies = ("/World/Base", "/World/Link")
    _create_source_stage(
        source,
        bodies,
        rotations={bodies[0]: ("Z", 25.0), bodies[1]: ("Y", -30.0)},
        translations={bodies[1]: (2.0, 3.0, 4.0)},
    )
    identity = identify_usd_artifact(source, uri=str(source))
    axis = 1.0 / math.sqrt(2.0)
    provenance = {
        field: _source_provenance(identity, field=field, prim_path=bodies[1])
        for field in ("joint_type", "body0", "body1", "axis_stage")
    }
    topology = JointTopologyV1(
        joint_id="diagonal-axis",
        joint_type="revolute",
        body0=bodies[0],
        body1=bodies[1],
        axis_stage=(axis, -axis, 0.0),
        field_provenance=provenance,
    )
    request = JointRiggerInputV1(
        schema_version=INPUT_SCHEMA_VERSION,
        source_asset=identity,
        plan=JointRiggerPlanV1(
            schema_version=PLAN_SCHEMA_VERSION,
            joints=(JointPlanV1(topology=topology),),
        ),
    )
    targets = _targets(tmp_path, "diagonal-output")
    source_stage = Usd.Stage.Open(str(source))
    assert source_stage is not None
    validate_joint_topology_plan(source_stage, request.plan)

    result = author_joint_topology(
        request,
        source_usd_path=source,
        artifact_targets=targets,
    )

    diagnostic = result.diagnostics.joint_diagnostics[0]
    decisions = {item.field: item for item in diagnostic.field_decisions}
    assert decisions["anchor.position_stage"].disposition == "defaulted"
    assert (
        decisions["anchor.position_stage"].reason_code == "inferred_body1_world_origin"
    )
    stage = Usd.Stage.Open(str(targets.output_path))
    observed = _authored_graph(stage)[topology.joint_id][3]
    assert observed == pytest.approx(topology.axis_stage, abs=1e-5)


def test_spherical_topology_authors_without_scalar_axis_or_limits(
    tmp_path: Path,
) -> None:
    source = tmp_path / "spherical-source.usda"
    bodies = ("/World/Base", "/World/Ball")
    _create_source_stage(source, bodies)
    source_stage = Usd.Stage.Open(str(source))
    assert source_stage is not None
    UsdGeom.Scope.Define(source_stage, "/World/Joints")
    assert source_stage.GetRootLayer().Save()
    del source_stage
    identity = identify_usd_artifact(source, uri=str(source))
    topology = JointTopologyV1(
        joint_id="ball joint",
        joint_type="spherical",
        body0=bodies[0],
        body1=bodies[1],
        field_provenance={
            field: _source_provenance(identity, field=field, prim_path=bodies[1])
            for field in ("joint_type", "body0", "body1")
        },
    )
    request = JointRiggerInputV1(
        schema_version=INPUT_SCHEMA_VERSION,
        source_asset=identity,
        plan=JointRiggerPlanV1(
            schema_version=PLAN_SCHEMA_VERSION,
            joints=(JointPlanV1(topology=topology),),
        ),
    )
    targets = _targets(tmp_path, "spherical-output")

    result = author_joint_topology(
        request,
        source_usd_path=source,
        artifact_targets=targets,
    )

    diagnostic = result.diagnostics.joint_diagnostics[0]
    assert diagnostic.authored_prim_path is not None
    decisions = {item.field: item for item in diagnostic.field_decisions}
    assert decisions["topology.axis_stage"].disposition == "ignored"
    assert decisions["topology.axis_stage"].reason_code == "not_applicable_spherical"
    assert decisions["limit.lower"].disposition == "ignored"
    stage = Usd.Stage.Open(str(targets.output_path))
    assert stage is not None
    joint = UsdPhysics.SphericalJoint(
        stage.GetPrimAtPath(diagnostic.authored_prim_path)
    )
    assert not joint.GetAxisAttr().HasAuthoredValueOpinion()
    assert not joint.GetLocalRot0Attr().HasAuthoredValueOpinion()
    assert not joint.GetLocalRot1Attr().HasAuthoredValueOpinion()
    assert stage.GetPrimAtPath("/World/Joints").IsA(UsdGeom.Scope)


@pytest.mark.parametrize("container_type", ["Xform", "Cube"])
def test_existing_joint_container_must_be_a_scope(
    tmp_path: Path,
    container_type: str,
) -> None:
    source = tmp_path / f"joint-container-{container_type.lower()}.usda"
    bodies = ("/World/Base", "/World/Link")
    _create_source_stage(source, bodies)
    stage = Usd.Stage.Open(str(source))
    assert stage is not None
    schema = {"Xform": UsdGeom.Xform, "Cube": UsdGeom.Cube}[container_type]
    schema.Define(stage, "/World/Joints")
    assert stage.GetRootLayer().Save()
    del stage
    request = _simple_request(source, body0=bodies[0], body1=bodies[1])
    targets = _targets(tmp_path, f"joint-container-{container_type.lower()}-output")

    with pytest.raises(JointRiggerContractError) as error:
        author_joint_topology(
            request,
            source_usd_path=source,
            artifact_targets=targets,
        )

    assert error.value.code == "joint_scope_conflict"
    assert "UsdGeom.Scope" in str(error.value)
    assert not targets.output_path.exists()
    assert not targets.diagnostics_path.exists()
    assert not targets.result_path.exists()


@pytest.mark.parametrize(
    ("case", "reason_code"),
    [
        ("missing_endpoint", "endpoint_missing"),
        ("inactive_endpoint", "endpoint_inactive_or_undefined"),
        ("non_xformable_endpoint", "endpoint_not_xformable"),
        ("time_sampled_endpoint", "time_varying_endpoint_transform"),
        ("singular_endpoint", "singular_endpoint_transform"),
        ("invalid_stage_units", "invalid_stage_units"),
        ("existing_joint", "source_already_rigged"),
    ],
)
def test_ambiguous_or_incompatible_topology_fails_without_partial_artifacts(
    tmp_path: Path,
    case: str,
    reason_code: str,
) -> None:
    source = tmp_path / f"{case}.usda"
    bodies = ("/World/Base", "/World/Link")
    _create_source_stage(source, bodies)
    stage = Usd.Stage.Open(str(source))
    assert stage is not None
    if case == "time_sampled_endpoint":
        body = UsdGeom.Xformable(stage.GetPrimAtPath(bodies[1]))
        body.AddTranslateOp().Set(Gf.Vec3d(1.0, 0.0, 0.0), Usd.TimeCode(1.0))
    elif case == "inactive_endpoint":
        stage.GetPrimAtPath(bodies[1]).SetActive(False)
    elif case == "non_xformable_endpoint":
        stage.RemovePrim(bodies[1])
        UsdGeom.Scope.Define(stage, bodies[1])
    elif case == "singular_endpoint":
        body = UsdGeom.Xformable(stage.GetPrimAtPath(bodies[1]))
        body.AddScaleOp().Set(Gf.Vec3d(0.0, 1.0, 1.0))
    elif case == "invalid_stage_units":
        UsdGeom.SetStageMetersPerUnit(stage, 0.0)
    elif case == "existing_joint":
        existing = UsdPhysics.RevoluteJoint.Define(stage, "/World/existing")
        assert existing.CreateBody0Rel().SetTargets([Sdf.Path(bodies[0])])
        assert existing.CreateBody1Rel().SetTargets([Sdf.Path(bodies[1])])
    assert stage.GetRootLayer().Save()
    del stage
    source_bytes = source.read_bytes()
    identity = identify_usd_artifact(source, uri=str(source))
    body1 = "/World/Missing" if case == "missing_endpoint" else bodies[1]
    topology = JointTopologyV1(
        joint_id="ambiguous",
        joint_type="revolute",
        body0=bodies[0],
        body1=body1,
        axis_stage=(0.0, 0.0, 1.0),
        field_provenance={
            field: _source_provenance(identity, field=field, prim_path=body1)
            for field in ("joint_type", "body0", "body1", "axis_stage")
        },
    )
    request = JointRiggerInputV1(
        schema_version=INPUT_SCHEMA_VERSION,
        source_asset=identity,
        plan=JointRiggerPlanV1(
            schema_version=PLAN_SCHEMA_VERSION,
            joints=(JointPlanV1(topology=topology),),
        ),
    )
    targets = _targets(tmp_path, f"{case}-output")
    for path in (
        targets.output_path,
        targets.diagnostics_path,
        targets.result_path,
    ):
        path.write_text("previous artifact", encoding="utf-8")

    with pytest.raises(JointRiggerContractError) as error:
        author_joint_topology(
            request,
            source_usd_path=source,
            artifact_targets=targets,
        )

    assert error.value.code == reason_code
    assert source.read_bytes() == source_bytes
    for path in (
        targets.output_path,
        targets.diagnostics_path,
        targets.result_path,
    ):
        assert path.read_text(encoding="utf-8") == "previous artifact"
    assert not any(tmp_path.glob(".*.stage-*"))


def test_wp_r3_plan_fields_and_non_source_backed_optional_facts_fail_closed(
    tmp_path: Path,
) -> None:
    source = tmp_path / "unsupported.usda"
    bodies = ("/World/Base", "/World/Link")
    _create_source_stage(source, bodies)
    identity = identify_usd_artifact(source, uri=str(source))
    topology = JointTopologyV1(
        joint_id="unsupported",
        joint_type="revolute",
        body0=bodies[0],
        body1=bodies[1],
        axis_stage=(0.0, 0.0, 1.0),
        field_provenance={
            field: _source_provenance(identity, field=field, prim_path=bodies[1])
            for field in ("joint_type", "body0", "body1", "axis_stage")
        },
    )
    state = JointStateV1(
        position=0.0,
        velocity=0.0,
        provenance=_source_provenance(
            identity,
            field="state",
            prim_path=bodies[1],
        ),
    )
    state_request = JointRiggerInputV1(
        schema_version=INPUT_SCHEMA_VERSION,
        source_asset=identity,
        plan=JointRiggerPlanV1(
            schema_version=PLAN_SCHEMA_VERSION,
            joints=(JointPlanV1(topology=topology, state=state),),
        ),
    )
    body_request = JointRiggerInputV1(
        schema_version=INPUT_SCHEMA_VERSION,
        source_asset=identity,
        plan=JointRiggerPlanV1(
            schema_version=PLAN_SCHEMA_VERSION,
            joints=(JointPlanV1(topology=topology),),
            rigid_bodies=(
                RigidBodyPlanV1(
                    prim_path=bodies[1],
                    provenance=_source_provenance(
                        identity,
                        field="rigid_body",
                        prim_path=bodies[1],
                    ),
                ),
            ),
        ),
    )
    anchor = JointAnchorV1(
        position_stage=(0.0, 0.0, 0.0),
        provenance=FieldProvenanceV1(
            source="owner_approved_plan",
            artifact=identity,
            prim_path=bodies[1],
            properties=("owner:anchor",),
            evidence="Owner supplied an anchor without source artifact evidence.",
        ),
    )
    anchor_request = JointRiggerInputV1(
        schema_version=INPUT_SCHEMA_VERSION,
        source_asset=identity,
        plan=JointRiggerPlanV1(
            schema_version=PLAN_SCHEMA_VERSION,
            joints=(JointPlanV1(topology=topology, anchor=anchor),),
        ),
    )
    friction_request = JointRiggerInputV1(
        schema_version=INPUT_SCHEMA_VERSION,
        source_asset=identity,
        plan=JointRiggerPlanV1(
            schema_version=PLAN_SCHEMA_VERSION,
            joints=(
                JointPlanV1(
                    topology=topology,
                    joint_friction=JointFrictionV1(
                        coefficient=0.15,
                        provenance=FieldProvenanceV1(
                            source="owner_approved_plan",
                            evidence="Owner supplied friction without source evidence.",
                        ),
                    ),
                ),
            ),
        ),
    )

    for index, (request, expected_code) in enumerate(
        (
            (state_request, "physics_schema_fields_unsupported"),
            (body_request, "physics_schema_fields_unsupported"),
            (anchor_request, "optional_field_not_source_backed"),
            (friction_request, "optional_field_not_source_backed"),
        )
    ):
        targets = _targets(tmp_path, f"unsupported-{index}")
        with pytest.raises(JointRiggerContractError) as error:
            author_joint_topology(
                request,
                source_usd_path=source,
                artifact_targets=targets,
            )
        assert error.value.code == expected_code
        for artifact_path in (
            targets.output_path,
            targets.diagnostics_path,
            targets.result_path,
        ):
            assert not artifact_path.exists(), (
                f"rejected case {index} published {artifact_path}"
            )


def test_legacy_component_name_compatibility_fails_before_authoring(
    tmp_path: Path,
) -> None:
    source = tmp_path / "legacy-compatibility-source.usda"
    bodies = ("/World/Base", "/World/Link")
    _create_source_stage(source, bodies)
    topology_request = _simple_request(source, body0=bodies[0], body1=bodies[1])
    request = JointRiggerInputV1(
        schema_version=INPUT_SCHEMA_VERSION,
        source_asset=topology_request.source_asset,
        plan=topology_request.plan,
        legacy_component_names=LegacyComponentNameCompatibilityV1(
            assignments=(
                LegacyComponentAssignmentV1(
                    prim_path=bodies[1],
                    component_name="drawer",
                    source_field="component_name",
                ),
            ),
        ),
    )
    targets = _targets(tmp_path, "legacy-compatibility-output")

    with pytest.raises(
        JointRiggerBackendIncompatibleError,
        match="never consumes legacy component_name",
    ):
        author_joint_topology(
            request,
            source_usd_path=source,
            artifact_targets=targets,
        )

    assert not targets.output_path.exists()
    assert not targets.diagnostics_path.exists()
    assert not targets.result_path.exists()


def test_post_author_validation_failure_rolls_back_before_facade_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "rollback-source.usda"
    bodies = ("/World/Base", "/World/Link")
    _create_source_stage(source, bodies)
    request = _simple_request(source, body0=bodies[0], body1=bodies[1])
    source_bytes = source.read_bytes()
    targets = _targets(tmp_path, "rollback-output")
    for path in (
        targets.output_path,
        targets.diagnostics_path,
        targets.result_path,
    ):
        path.write_text("previous artifact", encoding="utf-8")

    def fail_validation(*args: Any, **kwargs: Any) -> None:
        raise JointRiggerContractError(
            "injected_validation_failure",
            "exercise topology rollback",
        )

    monkeypatch.setattr(author_module, "_validate_authored_preflight", fail_validation)

    with pytest.raises(JointRiggerArtifactError) as error:
        author_joint_topology(
            request,
            source_usd_path=source,
            artifact_targets=targets,
        )

    assert isinstance(error.value.__cause__, JointRiggerContractError)
    assert error.value.__cause__.code == "injected_validation_failure"
    assert source.read_bytes() == source_bytes
    for path in (
        targets.output_path,
        targets.diagnostics_path,
        targets.result_path,
    ):
        assert path.read_text(encoding="utf-8") == "previous artifact"
    assert not any(tmp_path.glob(".*.stage-*"))


@pytest.mark.parametrize(
    ("remove_result", "expected_detail"),
    [
        (False, "RemovePrim reported failure"),
        (True, "authored paths remain"),
    ],
    ids=["false-return", "lying-success"],
)
def test_rollback_requires_remove_success_and_absent_authored_paths(
    tmp_path: Path,
    remove_result: bool,
    expected_detail: str,
) -> None:
    source = tmp_path / "rollback-verification-source.usda"
    bodies = ("/World/Base", "/World/Link")
    _create_source_stage(source, bodies)
    request = _simple_request(source, body0=bodies[0], body1=bodies[1])
    stage = Usd.Stage.Open(str(source))
    assert stage is not None
    preflight = validation_module._preflight_topology_authoring(stage, request.plan)
    diagnostics = author_module._build_diagnostics(request.plan, preflight)
    author_module._author_preflight(stage, preflight, diagnostics)
    remove_calls: list[str] = []

    class NonRemovingStage:
        def __getattr__(self, name: str) -> Any:
            return getattr(stage, name)

        def RemovePrim(self, path: Any) -> bool:
            remove_calls.append(str(path))
            return remove_result

    with pytest.raises(JointRiggerArtifactError, match=expected_detail):
        author_module._rollback_preflight(NonRemovingStage(), preflight)

    assert remove_calls == [
        *(prepared.joint_path for prepared in reversed(preflight.joints)),
        preflight.joints_scope_path,
    ]
    for path in remove_calls:
        assert stage.GetPrimAtPath(path).IsValid()


def test_owned_topology_backend_probe_accepts_complete_bound_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "probe-source.usda"
    bodies = ("/World/Base", "/World/Link")
    _create_source_stage(source, bodies)
    request = _simple_request(source, body0=bodies[0], body1=bodies[1])

    OwnedTopologyBackend(source).probe(request)


def test_rollback_ignores_preflight_paths_that_were_never_authored(
    tmp_path: Path,
) -> None:
    source = tmp_path / "empty-rollback-source.usda"
    bodies = ("/World/Base", "/World/Link")
    _create_source_stage(source, bodies)
    request = _simple_request(source, body0=bodies[0], body1=bodies[1])
    stage = Usd.Stage.Open(str(source))
    assert stage is not None
    preflight = validation_module._preflight_topology_authoring(stage, request.plan)

    author_module._rollback_preflight(stage, preflight)

    assert not stage.GetPrimAtPath(preflight.joints[0].joint_path).IsValid()
    assert not stage.GetPrimAtPath(preflight.joints_scope_path).IsValid()


def test_keyboard_interrupt_during_authoring_rolls_back_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "interrupted-authoring-source.usda"
    bodies = ("/World/Base", "/World/Link")
    _create_source_stage(source, bodies)
    request = _simple_request(source, body0=bodies[0], body1=bodies[1])
    stage = Usd.Stage.Open(str(source))
    assert stage is not None

    def interrupt_validation(*_args: Any, **_kwargs: Any) -> None:
        raise KeyboardInterrupt("forced authoring interruption")

    monkeypatch.setattr(
        author_module,
        "_validate_authored_preflight",
        interrupt_validation,
    )

    with pytest.raises(KeyboardInterrupt, match="forced authoring interruption"):
        author_module._author_topology_stage(stage, request.plan)

    preflight = validation_module._preflight_topology_authoring(stage, request.plan)
    assert not stage.GetPrimAtPath(preflight.joints[0].joint_path).IsValid()
    assert not stage.GetPrimAtPath(preflight.joints_scope_path).IsValid()


def test_post_save_no_reshape_rejects_source_mutation_after_reopen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "post-save-source.usda"
    bodies = ("/World/Base", "/World/Link")
    _create_source_stage(source, bodies)
    request = _simple_request(source, body0=bodies[0], body1=bodies[1])
    source_bytes = source.read_bytes()
    targets = _targets(tmp_path, "post-save-output")
    original_author = author_module._author_topology_stage
    original_saved_validation = author_module._validate_authored_saved_stage
    reopened_meters_per_unit: list[float] = []

    def author_then_mutate_source_metadata(stage: Any, plan: Any) -> Any:
        diagnostics = original_author(stage, plan)
        UsdGeom.SetStageMetersPerUnit(stage, 2.0)
        return diagnostics

    def observe_saved_stage(
        stage: Any,
        plan: Any,
        diagnostics: Any,
        *,
        source_preflight: Any,
        normalize_layer_identifiers: bool = False,
        layer_identifier_remap: Mapping[Path, Path] | None = None,
    ) -> None:
        reopened_meters_per_unit.append(float(UsdGeom.GetStageMetersPerUnit(stage)))
        original_saved_validation(
            stage,
            plan,
            diagnostics,
            source_preflight=source_preflight,
            normalize_layer_identifiers=normalize_layer_identifiers,
            layer_identifier_remap=layer_identifier_remap,
        )

    monkeypatch.setattr(
        author_module,
        "_author_topology_stage",
        author_then_mutate_source_metadata,
    )
    monkeypatch.setattr(
        author_module,
        "_validate_authored_saved_stage",
        observe_saved_stage,
    )

    with pytest.raises(JointRiggerArtifactError) as error:
        author_joint_topology(
            request,
            source_usd_path=source,
            artifact_targets=targets,
        )

    assert reopened_meters_per_unit == [2.0]
    assert isinstance(error.value.__cause__, JointRiggerContractError)
    assert error.value.__cause__.code == "no_reshape_violation"
    assert source.read_bytes() == source_bytes
    for path in (
        targets.output_path,
        targets.diagnostics_path,
        targets.result_path,
    ):
        assert not path.exists()
    assert not any(tmp_path.glob(".*.stage-*"))


def test_author_and_rollback_failures_retain_both_exception_traces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "dual-failure-source.usda"
    bodies = ("/World/Base", "/World/Link")
    _create_source_stage(source, bodies)
    request = _simple_request(source, body0=bodies[0], body1=bodies[1])
    targets = _targets(tmp_path, "dual-failure-output")

    def fail_validation(*args: Any, **kwargs: Any) -> None:
        raise JointRiggerContractError(
            "injected_validation_failure",
            "exercise retained authoring failure",
        )

    def fail_rollback(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("exercise retained rollback failure")

    monkeypatch.setattr(author_module, "_validate_authored_preflight", fail_validation)
    monkeypatch.setattr(author_module, "_rollback_preflight", fail_rollback)

    with pytest.raises(JointRiggerArtifactError) as caught:
        author_joint_topology(
            request,
            source_usd_path=source,
            artifact_targets=targets,
        )

    cause = caught.value.__cause__
    assert isinstance(cause, ExceptionGroup)
    assert len(cause.exceptions) == 2
    assert isinstance(cause.exceptions[0], JointRiggerContractError)
    assert cause.exceptions[0].code == "injected_validation_failure"
    assert isinstance(cause.exceptions[1], RuntimeError)
    assert not targets.output_path.exists()
    assert not targets.diagnostics_path.exists()
    assert not targets.result_path.exists()


def test_fatal_authoring_failure_remains_primary_when_rollback_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "fatal-authoring-rollback-failure.usda"
    bodies = ("/World/Base", "/World/Link")
    _create_source_stage(source, bodies)
    request = _simple_request(source, body0=bodies[0], body1=bodies[1])
    stage = Usd.Stage.Open(str(source))
    assert stage is not None
    primary = KeyboardInterrupt("fatal authoring interruption")
    rollback_failure = RuntimeError("ordinary rollback failure")

    def fail_validation(*_args: Any, **_kwargs: Any) -> None:
        raise primary

    def fail_rollback(*_args: Any, **_kwargs: Any) -> None:
        raise rollback_failure

    monkeypatch.setattr(author_module, "_validate_authored_preflight", fail_validation)
    monkeypatch.setattr(author_module, "_rollback_preflight", fail_rollback)

    with pytest.raises(KeyboardInterrupt) as caught:
        author_module._author_topology_stage(stage, request.plan)

    assert caught.value is primary
    notes = "\n".join(getattr(primary, "__notes__", ()))
    assert "Owned topology rollback also failed" in notes
    assert "RuntimeError: ordinary rollback failure" in notes


def test_fatal_rollback_failure_outranks_ordinary_authoring_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "fatal-rollback-authoring-failure.usda"
    bodies = ("/World/Base", "/World/Link")
    _create_source_stage(source, bodies)
    request = _simple_request(source, body0=bodies[0], body1=bodies[1])
    stage = Usd.Stage.Open(str(source))
    assert stage is not None
    primary = JointRiggerContractError(
        "ordinary_authoring_failure",
        "ordinary authoring failure",
    )
    rollback_failure = SystemExit("fatal rollback interruption")

    def fail_validation(*_args: Any, **_kwargs: Any) -> None:
        raise primary

    def fail_rollback(*_args: Any, **_kwargs: Any) -> None:
        raise rollback_failure

    monkeypatch.setattr(author_module, "_validate_authored_preflight", fail_validation)
    monkeypatch.setattr(author_module, "_rollback_preflight", fail_rollback)

    with pytest.raises(SystemExit) as caught:
        author_module._author_topology_stage(stage, request.plan)

    assert caught.value is rollback_failure
    assert caught.value.__cause__ is primary
    notes = "\n".join(getattr(rollback_failure, "__notes__", ()))
    assert "Owned topology authoring also failed" in notes
    assert "ordinary authoring failure" in notes


def test_fatal_authoring_failure_remains_primary_over_fatal_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "dual-fatal-authoring-failure.usda"
    bodies = ("/World/Base", "/World/Link")
    _create_source_stage(source, bodies)
    request = _simple_request(source, body0=bodies[0], body1=bodies[1])
    stage = Usd.Stage.Open(str(source))
    assert stage is not None
    primary = KeyboardInterrupt("fatal authoring interruption")
    rollback_failure = SystemExit("fatal rollback interruption")

    def fail_validation(*_args: Any, **_kwargs: Any) -> None:
        raise primary

    def fail_rollback(*_args: Any, **_kwargs: Any) -> None:
        raise rollback_failure

    monkeypatch.setattr(author_module, "_validate_authored_preflight", fail_validation)
    monkeypatch.setattr(author_module, "_rollback_preflight", fail_rollback)

    with pytest.raises(KeyboardInterrupt) as caught:
        author_module._author_topology_stage(stage, request.plan)

    assert caught.value is primary
    notes = "\n".join(getattr(primary, "__notes__", ()))
    assert "Owned topology rollback also failed" in notes
    assert "SystemExit: fatal rollback interruption" in notes


def test_owned_topology_authors_from_sealed_root_during_transient_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.usda"
    alternate = tmp_path / "alternate.usda"
    bodies = ("/World/Base", "/World/Link")
    _create_source_stage(source, bodies)
    _create_source_stage(alternate, bodies)
    source_layer = Sdf.Layer.FindOrOpen(str(source))
    alternate_layer = Sdf.Layer.FindOrOpen(str(alternate))
    assert source_layer is not None
    assert alternate_layer is not None
    source_layer.customLayerData = {"sealedSource": "original"}
    alternate_layer.customLayerData = {"sealedSource": "alternate"}
    assert source_layer.Save()
    assert alternate_layer.Save()
    del source_layer, alternate_layer
    request = _simple_request(source, body0=bodies[0], body1=bodies[1])
    original_bytes = source.read_bytes()
    alternate_bytes = alternate.read_bytes()
    original_author = author_module._author_topology_stage
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
        author_module,
        "_author_topology_stage",
        author_while_live_source_is_swapped,
    )
    targets = _targets(tmp_path, "sealed-root-output")

    result = author_joint_topology(
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


def test_owned_topology_authors_from_sealed_dependency_during_transient_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.usda"
    dependency = tmp_path / "geometry.usda"
    alternate = tmp_path / "alternate-geometry.usda"
    bodies = ("/World/Base", "/World/Link")
    _create_source_stage(
        dependency,
        bodies,
        translations={bodies[1]: (1.0, 0.0, 0.0)},
    )
    _create_source_stage(
        alternate,
        bodies,
        translations={bodies[1]: (99.0, 0.0, 0.0)},
    )
    root_layer = Sdf.Layer.CreateNew(str(source))
    root_layer.defaultPrim = "World"
    root_layer.subLayerPaths.append(dependency.name)
    assert root_layer.Save()
    del root_layer
    request = _simple_request(source, body0=bodies[0], body1=bodies[1])
    original_dependency = dependency.read_bytes()
    alternate_dependency = alternate.read_bytes()
    original_author = author_module._author_topology_stage
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
        author_module,
        "_author_topology_stage",
        author_while_live_dependency_is_swapped,
    )
    targets = _targets(tmp_path, "sealed-dependency-output")

    result = author_joint_topology(
        request,
        source_usd_path=source,
        artifact_targets=targets,
    )

    assert result.status == "succeeded"
    assert swapped
    assert dependency.read_bytes() == original_dependency
    output_stage = Usd.Stage.Open(str(targets.output_path))
    assert output_stage is not None
    diagnostic = result.diagnostics.joint_diagnostics[0]
    assert diagnostic.authored_prim_path is not None
    joint = UsdPhysics.RevoluteJoint(
        output_stage.GetPrimAtPath(diagnostic.authored_prim_path)
    )
    assert tuple(joint.GetLocalPos0Attr().Get()) == pytest.approx((1.0, 0.0, 0.0))
    assert tuple(joint.GetLocalPos1Attr().Get()) == pytest.approx((0.0, 0.0, 0.0))
    output_bytes = targets.output_path.read_bytes()
    assert b"joint-rigger-bound-input" not in output_bytes
    assert b"/proc/self/fd/" not in output_bytes


def test_owned_topology_restores_sealed_symlink_dependency_to_backing_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.usda"
    dependency = tmp_path / "real-geometry.usda"
    dependency_alias = tmp_path / "geometry-alias.usda"
    alternate = tmp_path / "alternate-geometry.usda"
    bodies = ("/World/Base", "/World/Link")
    _create_source_stage(
        dependency,
        bodies,
        translations={bodies[1]: (1.0, 0.0, 0.0)},
    )
    _create_source_stage(
        alternate,
        bodies,
        translations={bodies[1]: (99.0, 0.0, 0.0)},
    )
    dependency_alias.symlink_to(dependency.name)
    root_layer = Sdf.Layer.CreateNew(str(source))
    root_layer.defaultPrim = "World"
    root_layer.subLayerPaths.append(dependency_alias.name)
    assert root_layer.Save()
    del root_layer
    request = _simple_request(source, body0=bodies[0], body1=bodies[1])
    original_author = author_module._author_topology_stage
    retargeted = False

    def author_while_alias_is_retargeted(stage: Any, plan: Any) -> Any:
        nonlocal retargeted
        dependency_alias.unlink()
        dependency_alias.symlink_to(alternate.name)
        retargeted = True
        try:
            return original_author(stage, plan)
        finally:
            dependency_alias.unlink()
            dependency_alias.symlink_to(dependency.name)

    monkeypatch.setattr(
        author_module,
        "_author_topology_stage",
        author_while_alias_is_retargeted,
    )
    targets = _targets(tmp_path, "sealed-alias-output")

    result = author_joint_topology(
        request,
        source_usd_path=source,
        artifact_targets=targets,
    )

    assert result.status == "succeeded"
    assert retargeted
    output_layer = Sdf.Layer.FindOrOpen(str(targets.output_path))
    assert output_layer is not None
    assert output_layer.subLayerPaths == [dependency.name]
    output_stage = Usd.Stage.Open(str(targets.output_path))
    assert output_stage is not None
    assert output_stage.GetPrimAtPath(bodies[1]).IsValid()
    output_bytes = targets.output_path.read_bytes()
    assert dependency_alias.name.encode() not in output_bytes
    assert b"joint-rigger-bound-input" not in output_bytes
    assert b"/proc/self/fd/" not in output_bytes


def test_relative_dependency_relocation_fails_closed(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "output"
    source_dir.mkdir()
    output_dir.mkdir()
    dependency = source_dir / "dependency.usda"
    dependency.write_text('#usda 1.0\n\ndef Xform "Dependency" {}\n', encoding="utf-8")
    source = source_dir / "root.usda"
    bodies = ("/World/Base", "/World/Link")
    _create_source_stage(source, bodies)
    stage = Usd.Stage.Open(str(source))
    assert stage is not None
    stage.GetRootLayer().subLayerPaths.append(dependency.name)
    assert stage.GetRootLayer().Save()
    del stage
    request = _simple_request(source, body0=bodies[0], body1=bodies[1])
    targets = JointRiggerArtifactTargets(
        output_path=output_dir / "rigged.usda",
        diagnostics_path=output_dir / "diagnostics.json",
        result_path=output_dir / "result.json",
    )

    with pytest.raises(
        JointRiggerBackendIncompatibleError,
        match="cannot relocate.*composition dependencies",
    ):
        author_joint_topology(
            request,
            source_usd_path=source,
            artifact_targets=targets,
        )

    assert not targets.output_path.exists()
    assert not targets.diagnostics_path.exists()
    assert not targets.result_path.exists()


@pytest.mark.parametrize(
    "target_field",
    ["output_path", "diagnostics_path", "result_path"],
)
@pytest.mark.parametrize("protected_kind", ["source", "dependency"])
def test_direct_owned_backend_protects_bound_closure_with_remote_request_uri(
    tmp_path: Path,
    target_field: str,
    protected_kind: str,
) -> None:
    """The exported backend must not rely on request URIs for local protection."""

    dependency = tmp_path / "dependency.usda"
    dependency.write_text('#usda 1.0\n\ndef Xform "Dependency" {}\n', encoding="utf-8")
    source = tmp_path / "source.usda"
    bodies = ("/World/Base", "/World/Link")
    _create_source_stage(source, bodies)
    stage = Usd.Stage.Open(str(source))
    assert stage is not None
    stage.GetRootLayer().subLayerPaths.append(dependency.name)
    assert stage.GetRootLayer().Save()
    del stage
    request = _simple_request(
        source,
        body0=bodies[0],
        body1=bodies[1],
        source_uri="https://assets.example.invalid/logical/source.usda",
    )
    protected = source if protected_kind == "source" else dependency
    protected_bytes = protected.read_bytes()
    target_values = {
        "output_path": tmp_path / "generated.usda",
        "diagnostics_path": tmp_path / "diagnostics.json",
        "result_path": tmp_path / "result.json",
    }
    target_values[target_field] = protected
    targets = JointRiggerArtifactTargets(**target_values)

    with pytest.raises(
        JointRiggerArtifactError,
        match=rf"{target_field} must not alias bound source USD",
    ):
        author_joint_rig(request, OwnedTopologyBackend(source), targets)

    assert protected.read_bytes() == protected_bytes
    for field, path in target_values.items():
        if field != target_field:
            assert not path.exists()
    assert not any(tmp_path.rglob(".*.stage-*"))


def test_publication_protects_captured_source_after_parent_symlink_retarget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The publication guard must use the path captured with the sealed inode."""

    source_a_dir = tmp_path / "source-a"
    source_b_dir = tmp_path / "source-b"
    source_a_dir.mkdir()
    source_b_dir.mkdir()
    source_a = source_a_dir / "source.usda"
    source_b = source_b_dir / "source.usda"
    bodies = ("/World/Base", "/World/Link")
    _create_source_stage(source_b, bodies)
    shutil.copy2(source_b, source_a)
    source_a_bytes = source_a.read_bytes()
    source_b_bytes = source_b.read_bytes()
    assert source_a_bytes == source_b_bytes

    source_parent_alias = tmp_path / "source-parent"
    source_parent_alias.symlink_to(source_b_dir, target_is_directory=True)
    source = source_parent_alias / source_b.name
    request = _simple_request(
        source,
        body0=bodies[0],
        body1=bodies[1],
        source_uri="https://assets.example.invalid/logical/source.usda",
    )
    targets = JointRiggerArtifactTargets(
        output_path=source_a,
        diagnostics_path=tmp_path / "diagnostics.json",
        result_path=tmp_path / "result.json",
    )
    real_create_binding = author_module.create_sealed_source_binding
    binding_count = 0

    def retarget_around_author_binding(path: Path, *, expected: Any) -> Any:
        nonlocal binding_count
        binding_count += 1
        if binding_count != 1:
            return real_create_binding(path, expected=expected)

        source_parent_alias.unlink()
        source_parent_alias.symlink_to(source_a_dir, target_is_directory=True)
        try:
            binding = real_create_binding(path, expected=expected)
            assert binding.path == source_a
        finally:
            source_parent_alias.unlink()
            source_parent_alias.symlink_to(source_b_dir, target_is_directory=True)
        return binding

    monkeypatch.setattr(
        author_module,
        "create_sealed_source_binding",
        retarget_around_author_binding,
    )

    with pytest.raises(
        JointRiggerArtifactError,
        match="output_path must not alias captured resolved bound source USD",
    ):
        author_joint_rig(request, OwnedTopologyBackend(source), targets)

    assert binding_count == 1
    assert source_parent_alias.resolve() == source_b_dir
    assert source_a.read_bytes() == source_a_bytes
    assert source_b.read_bytes() == source_b_bytes
    assert not targets.diagnostics_path.exists()
    assert not targets.result_path.exists()
    assert not any(tmp_path.rglob(".*.stage-*"))


def test_final_frozen_validation_rejects_pre_freeze_root_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.usda"
    bodies = ("/World/Base", "/World/Link")
    _create_source_stage(source, bodies)
    request = _simple_request(source, body0=bodies[0], body1=bodies[1])
    targets = _targets(tmp_path, "pre-freeze-replacement")
    validated_copy = tmp_path / "validated-authored.usda"
    real_freeze = author_module.freeze_bound_projection_root
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
        author_module,
        "freeze_bound_projection_root",
        replace_root_before_freeze,
    )

    with pytest.raises(
        JointRiggerArtifactError,
        match="Final descriptor-pinned topology validation failed",
    ):
        author_joint_topology(
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


def test_final_frozen_validation_rejects_transient_parent_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.usda"
    bodies = ("/World/Base", "/World/Link")
    _create_source_stage(source, bodies)
    request = _simple_request(source, body0=bodies[0], body1=bodies[1])
    targets = _targets(tmp_path, "transient-parent-swap")
    real_freeze = author_module.freeze_bound_projection_root
    replacement_injected = False

    def replace_parent_during_validation(path: Path, **kwargs: Any) -> Any:
        nonlocal replacement_injected
        assert not replacement_injected
        replacement_injected = True
        parent = path.parent
        container = parent.parent
        validated_parent = container / f"{parent.name}.validated"
        unvalidated_parent = container / f"{parent.name}.unvalidated"
        container.chmod(0o700)
        shutil.copytree(parent, validated_parent, copy_function=shutil.copy2)
        shutil.copy2(source, path)
        validate = kwargs["validate_frozen_projection"]

        def validate_while_lexical_parent_is_swapped(
            validation_path: Path,
        ) -> None:
            parent.rename(unvalidated_parent)
            validated_parent.rename(parent)
            try:
                validate(validation_path)
            finally:
                parent.rename(validated_parent)
                unvalidated_parent.rename(parent)

        kwargs["validate_frozen_projection"] = validate_while_lexical_parent_is_swapped
        return real_freeze(path, **kwargs)

    monkeypatch.setattr(
        author_module,
        "freeze_bound_projection_root",
        replace_parent_during_validation,
    )

    with pytest.raises(
        JointRiggerArtifactError,
        match="Final descriptor-pinned topology validation failed",
    ):
        author_joint_topology(
            request,
            source_usd_path=source,
            artifact_targets=targets,
        )

    assert replacement_injected
    assert not targets.output_path.exists()
    assert not targets.diagnostics_path.exists()
    assert not targets.result_path.exists()
    assert not any(tmp_path.rglob(".*.stage-*"))


def test_bound_projection_runs_descriptor_cleanup_after_fatal_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.usda"
    bodies = ("/World/Base", "/World/Link")
    _create_source_stage(source, bodies)
    request = _simple_request(source, body0=bodies[0], body1=bodies[1])
    real_remove = author_module.remove_bound_input_directory
    real_close = author_module.close_source_binding
    bound_directories: list[Path] = []
    close_called = False

    def interrupt_projection_cleanup(path: Path) -> None:
        bound_directories.append(path)
        raise KeyboardInterrupt("projection cleanup interrupted")

    def interrupt_after_descriptor_cleanup(binding: Any) -> list[Exception]:
        nonlocal close_called
        close_called = True
        errors = real_close(binding)
        assert not errors
        raise SystemExit("descriptor cleanup interrupted")

    monkeypatch.setattr(
        author_module,
        "remove_bound_input_directory",
        interrupt_projection_cleanup,
    )
    monkeypatch.setattr(
        author_module,
        "close_source_binding",
        interrupt_after_descriptor_cleanup,
    )
    try:
        with pytest.raises(KeyboardInterrupt) as raised:
            with author_module._bound_source_projection(
                source,
                request,
                editable_root=False,
            ):
                pass
    finally:
        for directory in bound_directories:
            real_remove(directory)

    assert close_called
    assert raised.value.args == ("projection cleanup interrupted",)
    assert any(
        "descriptor cleanup interrupted" in note
        for note in getattr(raised.value, "__notes__", ())
    )


def test_bound_projection_preserves_active_fatal_and_cleanup_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.usda"
    bodies = ("/World/Base", "/World/Link")
    _create_source_stage(source, bodies)
    request = _simple_request(source, body0=bodies[0], body1=bodies[1])
    real_remove = author_module.remove_bound_input_directory
    real_close = author_module.close_source_binding
    bound_directories: list[Path] = []

    def fail_projection_cleanup(path: Path) -> None:
        bound_directories.append(path)
        raise RuntimeError("projection cleanup failed")

    def fail_descriptor_cleanup(binding: Any) -> list[Exception]:
        errors = real_close(binding)
        assert not errors
        return [OSError("descriptor close failed")]

    monkeypatch.setattr(
        author_module,
        "remove_bound_input_directory",
        fail_projection_cleanup,
    )
    monkeypatch.setattr(
        author_module,
        "close_source_binding",
        fail_descriptor_cleanup,
    )
    primary = KeyboardInterrupt("authoring interrupted")
    try:
        with pytest.raises(KeyboardInterrupt) as raised:
            with author_module._bound_source_projection(
                source,
                request,
                editable_root=False,
            ):
                raise primary
    finally:
        for directory in bound_directories:
            real_remove(directory)

    assert raised.value is primary
    notes = getattr(primary, "__notes__", ())
    assert any("projection cleanup failed" in note for note in notes)
    assert any("descriptor close failed" in note for note in notes)


def test_validation_detects_authored_graph_tampering(tmp_path: Path) -> None:
    bodies = ("/World/Base", "/World/Door")
    source, _, request = _paired_fixture(
        tmp_path,
        name="tamper",
        body_paths=bodies,
        joints=(
            {
                "joint_id": "hinge",
                "joint_type": "revolute",
                "body0": bodies[0],
                "body1": bodies[1],
                "axis_stage": (0.0, 0.0, 1.0),
                "anchor_stage": (0.0, 0.0, 0.0),
            },
        ),
    )
    targets = _targets(tmp_path, "tamper-output")
    result = author_joint_topology(
        request,
        source_usd_path=source,
        artifact_targets=targets,
    )
    path = result.diagnostics.joint_diagnostics[0].authored_prim_path
    assert path is not None
    stage = Usd.Stage.Open(str(targets.output_path))
    assert stage is not None
    joint = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(path))
    assert joint.CreateBody1Rel().SetTargets([Sdf.Path(bodies[0])])

    with pytest.raises(JointRiggerContractError) as error:
        validate_authored_joint_topology(stage, request.plan, result.diagnostics)

    assert error.value.code == "authored_graph_mismatch"

    joint.GetPrim().SetCustomDataByKey(
        "jointRigger:authoringVersion",
        "tampered-author-v0",
    )
    assert joint.CreateBody1Rel().SetTargets([Sdf.Path(bodies[1])])
    with pytest.raises(JointRiggerContractError) as version_error:
        validate_authored_joint_topology(stage, request.plan, result.diagnostics)
    assert version_error.value.code == "authored_graph_mismatch"


def test_source_joint_hidden_in_native_instance_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "native-instance-source.usda"
    bodies = ("/World/Base", "/World/Link")
    _create_source_stage(source, bodies)
    stage = Usd.Stage.Open(str(source))
    assert stage is not None
    _add_native_instance_with_hidden_joint(stage, tmp_path / "rigged-model.usda")
    assert stage.GetRootLayer().Save()
    request = _simple_request(source, body0=bodies[0], body1=bodies[1])

    with pytest.raises(JointRiggerContractError) as error:
        validate_joint_topology_plan(stage, request.plan)

    assert error.value.code == "source_already_rigged"
    assert "HiddenJoint" in str(error.value)


def test_exact_graph_detects_joint_hidden_in_native_instance(tmp_path: Path) -> None:
    source = tmp_path / "native-instance-exact-graph-source.usda"
    bodies = ("/World/Base", "/World/Link")
    _create_source_stage(source, bodies)
    request = _simple_request(source, body0=bodies[0], body1=bodies[1])
    targets = _targets(tmp_path, "native-instance-exact-graph-output")
    result = author_joint_topology(
        request,
        source_usd_path=source,
        artifact_targets=targets,
    )
    stage = Usd.Stage.Open(str(targets.output_path))
    assert stage is not None
    _add_native_instance_with_hidden_joint(
        stage,
        tmp_path / "exact-graph-rigged-model.usda",
    )

    with pytest.raises(JointRiggerContractError) as error:
        validate_authored_joint_topology(stage, request.plan, result.diagnostics)

    assert error.value.code == "authored_graph_mismatch"
    assert "HiddenJoint" in str(error.value)


@pytest.mark.parametrize(
    "extra_behavior",
    [
        "joint_enabled",
        "collision_enabled",
        "break_force",
        "raw_drive_attribute",
        "raw_physx_property",
        "unplanned_joint_friction",
        "unexpected_physx_api",
        "unexpected_api",
    ],
)
def test_validation_rejects_unplanned_authored_joint_behavior(
    tmp_path: Path,
    extra_behavior: str,
) -> None:
    source = tmp_path / f"extra-{extra_behavior}-source.usda"
    bodies = ("/World/Base", "/World/Link")
    _create_source_stage(source, bodies)
    request = _simple_request(source, body0=bodies[0], body1=bodies[1])
    targets = _targets(tmp_path, f"extra-{extra_behavior}-output")
    result = author_joint_topology(
        request,
        source_usd_path=source,
        artifact_targets=targets,
    )
    path = result.diagnostics.joint_diagnostics[0].authored_prim_path
    assert path is not None
    stage = Usd.Stage.Open(str(targets.output_path))
    assert stage is not None
    prim = stage.GetPrimAtPath(path)
    joint = UsdPhysics.RevoluteJoint(prim)
    if extra_behavior == "joint_enabled":
        assert joint.CreateJointEnabledAttr().Set(False)
    elif extra_behavior == "collision_enabled":
        assert joint.CreateCollisionEnabledAttr().Set(True)
    elif extra_behavior == "break_force":
        assert joint.CreateBreakForceAttr().Set(12.0)
    elif extra_behavior == "raw_drive_attribute":
        attribute = prim.CreateAttribute(
            "drive:angular:physics:stiffness",
            Sdf.ValueTypeNames.Float,
            custom=False,
        )
        assert attribute.Set(7.0)
    elif extra_behavior == "raw_physx_property":
        assert prim.AddAppliedSchema("PhysxJointAPI")
        assert prim.CreateAttribute(
            "physxJoint:solverFoo",
            Sdf.ValueTypeNames.Float,
            custom=False,
        ).Set(1.0)
    elif extra_behavior == "unplanned_joint_friction":
        assert prim.AddAppliedSchema("PhysxJointAPI")
        assert prim.CreateAttribute(
            "physxJoint:jointFriction",
            Sdf.ValueTypeNames.Float,
            custom=False,
        ).Set(0.15)
    elif extra_behavior == "unexpected_physx_api":
        assert prim.AddAppliedSchema("PhysxJointAPI:rogue")
    else:
        assert UsdPhysics.RigidBodyAPI.Apply(prim)

    with pytest.raises(JointRiggerContractError) as error:
        validate_authored_joint_topology(stage, request.plan, result.diagnostics)

    assert error.value.code == "authored_graph_mismatch"


@pytest.mark.parametrize(
    "tamper",
    [
        "missing_local_pos0",
        "changed_local_pos1",
        "changed_local_rot0",
        "unexpected_lower_limit",
    ],
)
def test_validation_rejects_non_round_tripping_joint_values(
    tmp_path: Path,
    tamper: str,
) -> None:
    source = tmp_path / f"round-trip-{tamper}-source.usda"
    bodies = ("/World/Base", "/World/Link")
    _create_source_stage(source, bodies)
    request = _simple_request(source, body0=bodies[0], body1=bodies[1])
    targets = _targets(tmp_path, f"round-trip-{tamper}-output")
    result = author_joint_topology(
        request,
        source_usd_path=source,
        artifact_targets=targets,
    )
    path = result.diagnostics.joint_diagnostics[0].authored_prim_path
    assert path is not None
    stage = Usd.Stage.Open(str(targets.output_path))
    assert stage is not None
    joint = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(path))
    if tamper == "missing_local_pos0":
        assert joint.GetPrim().RemoveProperty(joint.GetLocalPos0Attr().GetName())
    elif tamper == "changed_local_pos1":
        assert joint.GetLocalPos1Attr().Set(Gf.Vec3f(99.0, 0.0, 0.0))
    elif tamper == "changed_local_rot0":
        assert joint.GetLocalRot0Attr().Set(Gf.Quatf(0.0, Gf.Vec3f(1.0, 0.0, 0.0)))
    else:
        assert joint.CreateLowerLimitAttr().Set(-1.0)

    with pytest.raises(JointRiggerContractError) as error:
        validate_authored_joint_topology(stage, request.plan, result.diagnostics)

    assert error.value.code == "authored_graph_mismatch"


def test_owned_topology_backend_rejects_usdz_without_partial_artifacts(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.usda"
    bodies = ("/World/Base", "/World/Link")
    _create_source_stage(source, bodies)
    identity = identify_usd_artifact(source, uri=str(source))
    topology = JointTopologyV1(
        joint_id="joint",
        joint_type="revolute",
        body0=bodies[0],
        body1=bodies[1],
        axis_stage=(0.0, 0.0, 1.0),
        field_provenance={
            field: _source_provenance(identity, field=field, prim_path=bodies[1])
            for field in ("joint_type", "body0", "body1", "axis_stage")
        },
    )
    request = JointRiggerInputV1(
        schema_version=INPUT_SCHEMA_VERSION,
        source_asset=identity,
        plan=JointRiggerPlanV1(
            schema_version=PLAN_SCHEMA_VERSION,
            joints=(JointPlanV1(topology=topology),),
        ),
    )
    targets = JointRiggerArtifactTargets(
        output_path=tmp_path / "output.usdz",
        diagnostics_path=tmp_path / "diagnostics.json",
        result_path=tmp_path / "result.json",
    )

    with pytest.raises(JointRiggerBackendIncompatibleError, match="raw USD"):
        author_joint_topology(
            request,
            source_usd_path=source,
            artifact_targets=targets,
        )

    assert not targets.output_path.exists()
    assert not targets.diagnostics_path.exists()
    assert not targets.result_path.exists()

    packaged_source = tmp_path / "source.usdz"
    packaged_source.write_bytes(source.read_bytes())
    raw_targets = _targets(tmp_path, "raw-output")
    with pytest.raises(JointRiggerBackendIncompatibleError, match="raw USD"):
        author_joint_topology(
            request,
            source_usd_path=packaged_source,
            artifact_targets=raw_targets,
        )
    assert not raw_targets.output_path.exists()
    assert not raw_targets.diagnostics_path.exists()
    assert not raw_targets.result_path.exists()


def test_owned_topology_backend_rejects_symlinked_source_root(
    tmp_path: Path,
) -> None:
    real_source = tmp_path / "real-source.usda"
    source_alias = tmp_path / "source-alias.usda"
    bodies = ("/World/Base", "/World/Link")
    _create_source_stage(real_source, bodies)
    source_alias.symlink_to(real_source.name)
    source_bytes = real_source.read_bytes()
    request = _simple_request(real_source, body0=bodies[0], body1=bodies[1])
    targets = _targets(tmp_path, "symlinked-source-output")

    with pytest.raises(
        JointRiggerArtifactError,
        match="source USD must be a regular file",
    ):
        author_joint_topology(
            request,
            source_usd_path=source_alias,
            artifact_targets=targets,
        )

    assert source_alias.is_symlink()
    assert real_source.read_bytes() == source_bytes
    assert not targets.output_path.exists()
    assert not targets.diagnostics_path.exists()
    assert not targets.result_path.exists()


def test_owned_topology_backend_rejects_cross_format_raw_usd_output(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.usda"
    bodies = ("/World/Base", "/World/Link")
    _create_source_stage(source, bodies)
    request = _simple_request(source, body0=bodies[0], body1=bodies[1])
    targets = JointRiggerArtifactTargets(
        output_path=tmp_path / "output.usdc",
        diagnostics_path=tmp_path / "diagnostics.json",
        result_path=tmp_path / "result.json",
    )

    with pytest.raises(
        JointRiggerBackendIncompatibleError,
        match="matching source and output suffixes",
    ):
        author_joint_topology(
            request,
            source_usd_path=source,
            artifact_targets=targets,
        )

    assert not targets.output_path.exists()
    assert not targets.diagnostics_path.exists()
    assert not targets.result_path.exists()


@pytest.mark.parametrize(
    ("concrete_format", "magic"),
    [("usda", b"#usda"), ("usdc", b"PXR-USDC")],
)
def test_generic_usd_output_preserves_concrete_source_encoding(
    tmp_path: Path,
    concrete_format: str,
    magic: bytes,
) -> None:
    bodies = ("/World/Base", "/World/Link")
    concrete_source = tmp_path / f"source.{concrete_format}"
    _create_source_stage(concrete_source, bodies)
    source = tmp_path / "source.usd"
    concrete_source.rename(source)
    assert source.read_bytes().startswith(magic)
    request = _simple_request(source, body0=bodies[0], body1=bodies[1])
    targets = JointRiggerArtifactTargets(
        output_path=tmp_path / "output.usd",
        diagnostics_path=tmp_path / "diagnostics.json",
        result_path=tmp_path / "result.json",
    )

    result = author_joint_topology(
        request,
        source_usd_path=source,
        artifact_targets=targets,
    )

    assert result.status == "succeeded"
    assert targets.output_path.read_bytes().startswith(magic)
    output_stage = Usd.Stage.Open(str(targets.output_path))
    assert output_stage is not None
    assert any(prim.IsA(UsdPhysics.Joint) for prim in output_stage.Traverse())


@pytest.mark.parametrize(
    ("publication_suffix", "physical_suffix"),
    [(".usda", ".usdc"), (".usd", ".usda")],
)
def test_format_preserving_diagnostic_reports_both_output_suffixes(
    tmp_path: Path,
    publication_suffix: str,
    physical_suffix: str,
) -> None:
    with pytest.raises(JointRiggerBackendIncompatibleError) as error:
        author_module._validate_format_preserving_output(
            source_path=tmp_path / "source.usda",
            publication_path=tmp_path / f"published{publication_suffix}",
            physical_path=tmp_path / f"staged{physical_suffix}",
        )

    assert (
        f"source=.usda, publication={publication_suffix}, physical={physical_suffix}"
    ) in str(error.value)


@pytest.mark.parametrize("anchor_mode", ["explicit", "body1_default"])
def test_float32_local_anchor_reprojection_fails_before_authoring(
    tmp_path: Path,
    anchor_mode: str,
) -> None:
    source = tmp_path / f"large-anchor-{anchor_mode}.usda"
    bodies = ("/World/Base", "/World/Link")
    translations = (
        {bodies[1]: (100000001.0, 0.0, 0.0)} if anchor_mode == "body1_default" else None
    )
    _create_source_stage(source, bodies, translations=translations)
    request = _simple_request(source, body0=bodies[0], body1=bodies[1])
    if anchor_mode == "explicit":
        joint = request.plan.joints[0]
        anchor = JointAnchorV1(
            position_stage=(100000001.0, 0.0, 0.0),
            provenance=_source_provenance(
                request.source_asset,
                field="anchor",
                prim_path=bodies[1],
            ),
        )
        request = request.model_copy(
            update={
                "plan": request.plan.model_copy(
                    update={"joints": (joint.model_copy(update={"anchor": anchor}),)}
                )
            }
        )
    source_bytes = source.read_bytes()
    stage = Usd.Stage.Open(str(source))
    assert stage is not None

    with pytest.raises(JointRiggerContractError) as validation_error:
        validate_joint_topology_plan(stage, request.plan)

    assert validation_error.value.code == "authored_value_out_of_range"
    assert "do not preserve the requested stage anchor" in str(validation_error.value)

    targets = _targets(tmp_path, f"large-anchor-{anchor_mode}-output")
    with pytest.raises(JointRiggerContractError) as author_error:
        author_joint_topology(
            request,
            source_usd_path=source,
            artifact_targets=targets,
        )

    assert author_error.value.code == "authored_value_out_of_range"
    assert source.read_bytes() == source_bytes
    assert not targets.output_path.exists()
    assert not targets.diagnostics_path.exists()
    assert not targets.result_path.exists()


def test_default_anchor_reconciles_endpoint_float32_reprojections(
    tmp_path: Path,
) -> None:
    source = tmp_path / "reconciled-default-anchor.usda"
    bodies = ("/World/Base", "/World/Link")
    scale = float(Gf.Vec3f(0.28)[0])
    local_midpoint = 100.0 + 2.0**-18
    requested_x = scale * local_midpoint
    _create_source_stage(
        source,
        bodies,
        translations={bodies[1]: (requested_x, 0.0, 0.0)},
    )
    stage = Usd.Stage.Open(str(source))
    assert stage is not None
    base = UsdGeom.Xformable(stage.GetPrimAtPath(bodies[0]))
    assert base.AddScaleOp().Set(Gf.Vec3f(scale))
    assert stage.GetRootLayer().Save()
    request = _simple_request(source, body0=bodies[0], body1=bodies[1])

    preflight = validation_module._preflight_topology_authoring(stage, request.plan)
    prepared = preflight.joints[0]
    cache = UsdGeom.XformCache()
    body0_xform = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(bodies[0]))
    body1_xform = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(bodies[1]))
    independently_rounded_anchor0 = tuple(
        float(value)
        for value in body0_xform.Transform(
            Gf.Vec3d(
                *validation_module._float32_vector(
                    (local_midpoint, 0.0, 0.0),
                    label="independently rounded body0 anchor",
                )
            )
        )
    )
    independently_rounded_anchor1 = tuple(
        float(value) for value in body1_xform.Transform(Gf.Vec3d())
    )
    assert (
        validation_module._vector_distance(
            independently_rounded_anchor0,
            independently_rounded_anchor1,
        )
        > validation_module._SHARED_ANCHOR_DISTANCE_TOLERANCE
    )

    authored_anchor0 = tuple(
        float(value) for value in body0_xform.Transform(Gf.Vec3d(*prepared.local_pos0))
    )
    authored_anchor1 = tuple(
        float(value) for value in body1_xform.Transform(Gf.Vec3d(*prepared.local_pos1))
    )
    requested_anchor = (requested_x, 0.0, 0.0)
    assert prepared.local_pos1 != (0.0, 0.0, 0.0)
    assert (
        validation_module._vector_distance(
            authored_anchor0,
            authored_anchor1,
        )
        <= validation_module._SHARED_ANCHOR_DISTANCE_TOLERANCE
    )
    assert all(
        abs(observed - requested) <= validation_module._FRAME_TOLERANCE
        for authored in (authored_anchor0, authored_anchor1)
        for observed, requested in zip(authored, requested_anchor, strict=True)
    )

    targets = _targets(tmp_path, "reconciled-default-anchor-output")
    result = author_joint_topology(
        request,
        source_usd_path=source,
        artifact_targets=targets,
    )
    output_stage = Usd.Stage.Open(str(targets.output_path))
    assert output_stage is not None
    validate_authored_joint_topology(
        output_stage,
        request.plan,
        result.diagnostics,
    )


def test_exactly_representable_large_anchor_round_trips() -> None:
    stage = Usd.Stage.CreateInMemory()
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.Xform.Define(stage, "/World/Base")
    UsdGeom.Xform.Define(stage, "/World/Link")
    identity = ArtifactIdentityV1(
        uri="fixture://large-anchor-exact.usda",
        root_sha256="0" * 64,
    )
    topology = JointTopologyV1(
        joint_id="exact large anchor",
        joint_type="revolute",
        body0="/World/Base",
        body1="/World/Link",
        axis_stage=(0.0, 0.0, 1.0),
        field_provenance={
            field: _source_provenance(
                identity,
                field=field,
                prim_path="/World/Link",
            )
            for field in ("joint_type", "body0", "body1", "axis_stage")
        },
    )
    anchor = JointAnchorV1(
        position_stage=(100000000.0, 0.0, 0.0),
        provenance=_source_provenance(
            identity,
            field="anchor",
            prim_path="/World/Link",
        ),
    )
    plan = JointRiggerPlanV1(
        schema_version=PLAN_SCHEMA_VERSION,
        joints=(JointPlanV1(topology=topology, anchor=anchor),),
    )

    validate_joint_topology_plan(stage, plan)
    diagnostics = author_module._author_topology_stage(stage, plan)
    validate_authored_joint_topology(stage, plan, diagnostics)


@pytest.mark.parametrize("endpoint_kind", ["prototype_root", "prototype_child"])
def test_prototype_namespace_endpoint_fails_before_authoring(
    tmp_path: Path,
    endpoint_kind: str,
) -> None:
    source = tmp_path / f"{endpoint_kind}.usda"
    bodies = ("/World/Base", "/World/Link")
    _create_source_stage(source, bodies)
    stage = Usd.Stage.Open(str(source))
    assert stage is not None
    model = UsdGeom.Xform.Define(stage, "/Model")
    UsdGeom.Xform.Define(stage, "/Model/PrototypeLink")
    instance = stage.OverridePrim("/World/Instance")
    assert instance.GetReferences().AddInternalReference(model.GetPath())
    assert instance.SetInstanceable(True)
    assert stage.GetRootLayer().Save()
    prototypes = stage.GetPrototypes()
    assert len(prototypes) == 1
    prototype = prototypes[0]
    endpoint = (
        prototype
        if endpoint_kind == "prototype_root"
        else stage.GetPrimAtPath(prototype.GetPath().AppendChild("PrototypeLink"))
    )
    assert endpoint.IsPrototype() or endpoint.IsInPrototype()
    endpoint_path = str(endpoint.GetPath())
    request = _simple_request(
        source,
        body0=bodies[0],
        body1=endpoint_path,
    )
    source_bytes = source.read_bytes()

    with pytest.raises(JointRiggerContractError) as validation_error:
        validate_joint_topology_plan(stage, request.plan)

    assert validation_error.value.code == "endpoint_prototype"

    targets = _targets(tmp_path, f"{endpoint_kind}-output")
    with pytest.raises(JointRiggerContractError) as author_error:
        author_joint_topology(
            request,
            source_usd_path=source,
            artifact_targets=targets,
        )

    assert author_error.value.code == "endpoint_prototype"
    assert source.read_bytes() == source_bytes
    assert not targets.output_path.exists()
    assert not targets.diagnostics_path.exists()
    assert not targets.result_path.exists()


@pytest.mark.parametrize(
    ("meters_per_unit", "limit_value"),
    [(1e-300, 1e300), (1e300, 1e-300)],
    ids=["nonfinite", "underflow"],
)
def test_prismatic_limit_conversion_fails_before_float32_and_authoring(
    tmp_path: Path,
    meters_per_unit: float,
    limit_value: float,
) -> None:
    source = tmp_path / f"prismatic-limit-{meters_per_unit!r}.usda"
    bodies = ("/World/Base", "/World/Link")
    _create_source_stage(
        source,
        bodies,
        meters_per_unit=meters_per_unit,
    )
    request = _simple_request(source, body0=bodies[0], body1=bodies[1])
    original_joint = request.plan.joints[0]
    topology = JointTopologyV1(
        joint_id=original_joint.topology.joint_id,
        joint_type="prismatic",
        body0=original_joint.topology.body0,
        body1=original_joint.topology.body1,
        axis_stage=original_joint.topology.axis_stage,
        field_provenance=original_joint.topology.field_provenance,
    )
    limit = JointLimitV1(
        lower=limit_value,
        unit="meters",
        provenance=_source_provenance(
            request.source_asset,
            field="limit",
            prim_path=bodies[1],
        ),
    )
    request = request.model_copy(
        update={
            "plan": request.plan.model_copy(
                update={"joints": (JointPlanV1(topology=topology, limit=limit),)}
            )
        }
    )
    source_bytes = source.read_bytes()
    stage = Usd.Stage.Open(str(source))
    assert stage is not None

    with pytest.raises(JointRiggerContractError) as validation_error:
        validate_joint_topology_plan(stage, request.plan)

    assert validation_error.value.code == "authored_value_out_of_range"
    assert "converted prismatic limit is not representable" in str(
        validation_error.value
    )

    targets = _targets(tmp_path, f"prismatic-limit-{meters_per_unit!r}-output")
    with pytest.raises(JointRiggerContractError) as author_error:
        author_joint_topology(
            request,
            source_usd_path=source,
            artifact_targets=targets,
        )

    assert author_error.value.code == "authored_value_out_of_range"
    assert source.read_bytes() == source_bytes
    assert not targets.output_path.exists()
    assert not targets.diagnostics_path.exists()
    assert not targets.result_path.exists()


@pytest.mark.parametrize(
    "tamper",
    ["time_sampled_extension", "inactive_joint", "undefined_joint"],
)
def test_author_rolls_back_invalid_authored_joint_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    bodies = ("/World/Base", "/World/Link")
    source, _, request = _paired_fixture(
        tmp_path,
        name=f"authored-state-{tamper}",
        body_paths=bodies,
        joints=(
            {
                "joint_id": "authored state",
                "joint_type": "revolute",
                "body0": bodies[0],
                "body1": bodies[1],
                "axis_stage": (0.0, 0.0, 1.0),
                "anchor_stage": (0.0, 0.0, 0.0),
                "lower": -10.0,
                "upper": 100.0,
                "drive": True,
            },
        ),
    )
    planned_joint = request.plan.joints[0]
    assert planned_joint.drive is not None
    drive = planned_joint.drive.model_copy(update={"max_joint_velocity": 4.0})
    request = request.model_copy(
        update={
            "plan": request.plan.model_copy(
                update={"joints": (planned_joint.model_copy(update={"drive": drive}),)}
            )
        }
    )
    real_author_preflight = author_module._author_preflight

    def author_then_tamper(stage: Any, preflight: Any, diagnostics: Any) -> None:
        real_author_preflight(stage, preflight, diagnostics)
        prim = stage.GetPrimAtPath(preflight.joints[0].joint_path)
        if tamper == "time_sampled_extension":
            attribute = prim.GetAttribute("physxJoint:maxJointVelocity")
            assert attribute.Set(attribute.Get(), Usd.TimeCode(1.0))
        elif tamper == "inactive_joint":
            assert prim.SetActive(False)
        else:
            spec = stage.GetRootLayer().GetPrimAtPath(preflight.joints[0].joint_path)
            assert spec is not None
            spec.specifier = Sdf.SpecifierOver

    monkeypatch.setattr(author_module, "_author_preflight", author_then_tamper)
    source_bytes = source.read_bytes()
    targets = _targets(tmp_path, f"authored-state-{tamper}-output")

    with pytest.raises(JointRiggerArtifactError) as error:
        author_joint_topology(
            request,
            source_usd_path=source,
            artifact_targets=targets,
        )

    assert isinstance(error.value.__cause__, JointRiggerContractError)
    assert error.value.__cause__.code == "authored_graph_mismatch"
    assert source.read_bytes() == source_bytes
    assert not targets.output_path.exists()
    assert not targets.diagnostics_path.exists()
    assert not targets.result_path.exists()


def _source_provenance(
    identity: ArtifactIdentityV1,
    *,
    field: str,
    prim_path: str,
) -> FieldProvenanceV1:
    return FieldProvenanceV1(
        source="source_metadata",
        artifact=identity,
        prim_path=prim_path,
        properties=(field,),
        evidence=f"Synthetic source evidence for {field}.",
    )


def _simple_request(
    source: Path,
    *,
    body0: str,
    body1: str,
    source_uri: str | None = None,
) -> JointRiggerInputV1:
    identity = identify_usd_artifact(source, uri=source_uri or str(source))
    topology = JointTopologyV1(
        joint_id="simple joint",
        joint_type="revolute",
        body0=body0,
        body1=body1,
        axis_stage=(0.0, 0.0, 1.0),
        field_provenance={
            field: _source_provenance(identity, field=field, prim_path=body1)
            for field in ("joint_type", "body0", "body1", "axis_stage")
        },
    )
    return JointRiggerInputV1(
        schema_version=INPUT_SCHEMA_VERSION,
        source_asset=identity,
        plan=JointRiggerPlanV1(
            schema_version=PLAN_SCHEMA_VERSION,
            joints=(JointPlanV1(topology=topology),),
        ),
    )
