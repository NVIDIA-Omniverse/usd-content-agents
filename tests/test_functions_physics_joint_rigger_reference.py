# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the app-independent Joint Rigger reference oracle."""

from __future__ import annotations

import hashlib
import math
import os
import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

import pytest

Gf = pytest.importorskip("pxr.Gf")
Ar = pytest.importorskip("pxr.Ar")
Sdf = pytest.importorskip("pxr.Sdf")
Usd = pytest.importorskip("pxr.Usd")
UsdGeom = pytest.importorskip("pxr.UsdGeom")
UsdPhysics = pytest.importorskip("pxr.UsdPhysics")
UsdShade = pytest.importorskip("pxr.UsdShade")
UsdUtils = pytest.importorskip("pxr.UsdUtils")
Ts = pytest.importorskip("pxr.Ts")

import world_understanding.functions.physics.joint_rigger.reference as rv  # noqa: E402
from world_understanding.functions.physics.joint_rigger.models import (  # noqa: E402
    ArtifactIdentityV1,
    JointRiggerContractError,
    JointRiggerInputV1,
    canonical_json,
    canonical_sha256,
)
from world_understanding.functions.physics.joint_rigger.reference import (  # noqa: E402
    extract_reference_input,
    identify_usd_artifact,
    local_usd_dependency_paths,
    write_reference_input,
)

SOURCE_URI = "fixture://drawer-hinge/source"
REFERENCE_URI = "fixture://drawer-hinge/reference"
_BODY_FALLBACK_CASES = (
    (
        "physics:rigidBodyEnabled",
        Sdf.ValueTypeNames.Bool,
        True,
        False,
        "unsupported_rigid_body_property",
    ),
    (
        "physics:kinematicEnabled",
        Sdf.ValueTypeNames.Bool,
        False,
        True,
        "unsupported_rigid_body_property",
    ),
    (
        "physics:startsAsleep",
        Sdf.ValueTypeNames.Bool,
        False,
        True,
        "unsupported_rigid_body_property",
    ),
    (
        "physics:velocity",
        Sdf.ValueTypeNames.Vector3f,
        Gf.Vec3f(0.0),
        Gf.Vec3f(1.0, 0.0, 0.0),
        "unsupported_rigid_body_property",
    ),
    (
        "physics:angularVelocity",
        Sdf.ValueTypeNames.Vector3f,
        Gf.Vec3f(0.0),
        Gf.Vec3f(0.0, 1.0, 0.0),
        "unsupported_rigid_body_property",
    ),
    (
        "physics:collisionEnabled",
        Sdf.ValueTypeNames.Bool,
        True,
        False,
        "unsupported_collision_property",
    ),
)
_MASS_PROPERTY_CASES = (
    ("physics:mass", Sdf.ValueTypeNames.Float, 1.0),
    ("physics:density", Sdf.ValueTypeNames.Float, 1000.0),
    ("physics:centerOfMass", Sdf.ValueTypeNames.Point3f, Gf.Vec3f(0.0)),
    (
        "physics:diagonalInertia",
        Sdf.ValueTypeNames.Vector3f,
        Gf.Vec3f(1.0),
    ),
    (
        "physics:principalAxes",
        Sdf.ValueTypeNames.Quatf,
        Gf.Quatf(1.0, Gf.Vec3f(0.0)),
    ),
)


def test_extract_reference_input_preserves_graph_optional_facts_and_identity(
    tmp_path: Path,
) -> None:
    source, reference = _write_pair(tmp_path)
    source_before = _sha256(source)
    reference_before = _sha256(reference)

    result = extract_reference_input(
        source,
        reference,
        source_uri=SOURCE_URI,
        reference_uri=REFERENCE_URI,
    )

    assert result.source_asset.uri == SOURCE_URI
    assert result.source_asset.root_sha256 == source_before
    assert len(result.source_asset.dependency_bundle_sha256 or "") == 64
    assert [joint.topology.joint_id for joint in result.plan.joints] == [
        "/World/Joints/door",
        "/World/Joints/drawer",
        "/World/Joints/spherical",
    ]

    drawer = _joint(result, "/World/Joints/drawer")
    assert drawer.topology.joint_type == "prismatic"
    assert drawer.topology.body0 == "/World/base"
    assert drawer.topology.body1 == "/World/drawer"
    assert drawer.topology.axis_stage == (1.0, 0.0, 0.0)
    assert set(drawer.topology.field_provenance) == {
        "joint_type",
        "body0",
        "body1",
        "axis_stage",
    }
    axis_provenance = drawer.topology.field_provenance["axis_stage"]
    assert axis_provenance.source == "authored_reference"
    assert axis_provenance.artifact is not None
    assert axis_provenance.artifact.uri == REFERENCE_URI
    assert axis_provenance.properties == ("physics:axis",)
    assert drawer.limit is not None
    assert drawer.limit.unit == "meters"
    assert drawer.limit.lower == 0.0
    assert drawer.limit.upper == pytest.approx(0.5)
    assert drawer.anchor is not None
    assert drawer.anchor.position_stage == (10.0, 0.0, 0.0)
    assert drawer.anchor.provenance.properties == (
        "physics:localPos0",
        "physics:localPos1",
    )
    assert drawer.drive is not None
    assert drawer.drive.drive_type == "force"
    assert drawer.drive.stiffness == 25.0

    spherical = _joint(result, "/World/Joints/spherical")
    assert spherical.topology.joint_type == "spherical"
    assert spherical.topology.axis_stage is None
    assert set(spherical.topology.field_provenance) == {
        "joint_type",
        "body0",
        "body1",
    }

    body_by_path = {body.prim_path: body for body in result.plan.rigid_bodies}
    assert set(body_by_path) == {
        "/World/ball",
        "/World/base",
        "/World/door",
        "/World/drawer",
    }
    base = body_by_path["/World/base"]
    assert base.mass is not None
    assert base.mass.mass_kg == 2.0
    assert base.mass.diagonal_inertia_kg_m2 == pytest.approx((0.01, 0.02, 0.03))
    assert [collider.prim_path for collider in base.colliders] == ["/World/base"]
    assert base.colliders[0].provenance.properties == ("PhysicsCollisionAPI",)
    assert result.plan.articulation_root is not None
    assert result.plan.articulation_root.prim_path == "/World/base"

    assert _sha256(source) == source_before
    assert _sha256(reference) == reference_before


def test_optional_joint_facts_are_absent_without_authored_evidence(
    tmp_path: Path,
) -> None:
    source, reference = _write_pair(tmp_path)

    result = extract_reference_input(
        source,
        reference,
        source_uri=SOURCE_URI,
        reference_uri=REFERENCE_URI,
    )

    door = _joint(result, "/World/Joints/door")
    assert door.limit is None
    assert door.anchor is None
    assert door.joint_friction is None
    assert door.drive is None
    assert door.state is None
    assert door.mimic is None
    payload = result.model_dump(mode="json", exclude_none=True)
    serialized_door = next(
        item
        for item in payload["plan"]["joints"]
        if item["topology"]["joint_id"] == "/World/Joints/door"
    )
    assert set(serialized_door) == {"topology"}


def test_axis_provenance_lists_only_authored_local_rotations(tmp_path: Path) -> None:
    source, reference = _write_pair(tmp_path)
    stage = Usd.Stage.Open(str(reference))
    drawer = UsdPhysics.PrismaticJoint(stage.GetPrimAtPath("/World/Joints/drawer"))
    drawer.GetLocalRot0Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0)))
    stage.GetRootLayer().Save()

    result = _extract(source, reference)

    axis_provenance = _joint(result, "/World/Joints/drawer").topology.field_provenance[
        "axis_stage"
    ]
    assert axis_provenance.properties == (
        "physics:axis",
        "physics:localRot0",
    )


def test_relative_joint_frame_twist_fails_closed(tmp_path: Path) -> None:
    source, reference = _write_pair(tmp_path)
    stage = Usd.Stage.Open(str(reference))
    drawer = UsdPhysics.PrismaticJoint(stage.GetPrimAtPath("/World/Joints/drawer"))
    twist = Gf.Quatf(Gf.Rotation(Gf.Vec3d(1.0, 0.0, 0.0), 90.0).GetQuat())
    drawer.GetLocalRot0Attr().Set(twist)
    drawer.GetLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0)))
    assert stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "unsupported_joint_frame_twist"
    assert "/World/Joints/drawer" in caught.value.detail


def test_matching_joint_frame_twist_is_representable_as_shared_axis(
    tmp_path: Path,
) -> None:
    source, reference = _write_pair(tmp_path)
    stage = Usd.Stage.Open(str(reference))
    drawer = UsdPhysics.PrismaticJoint(stage.GetPrimAtPath("/World/Joints/drawer"))
    twist = Gf.Quatf(Gf.Rotation(Gf.Vec3d(1.0, 0.0, 0.0), 90.0).GetQuat())
    drawer.GetLocalRot0Attr().Set(twist)
    drawer.GetLocalRot1Attr().Set(twist)
    assert stage.GetRootLayer().Save()

    result = _extract(source, reference)

    extracted = _joint(result, "/World/Joints/drawer")
    assert extracted.topology.axis_stage == (1.0, 0.0, 0.0)
    assert extracted.topology.field_provenance["axis_stage"].properties == (
        "physics:axis",
        "physics:localRot0",
        "physics:localRot1",
    )


@pytest.mark.parametrize(
    ("field", "quaternion", "expected_detail"),
    [
        ("localRot0", Gf.Quatf(0.0, Gf.Vec3f(0.0)), "zero or near-zero"),
        ("localRot1", Gf.Quatf(1e-13, Gf.Vec3f(0.0)), "zero or near-zero"),
        ("localRot0", Gf.Quatf(math.nan, Gf.Vec3f(0.0)), "non-finite"),
        (
            "localRot1",
            Gf.Quatf(0.0, Gf.Vec3f(math.inf, 0.0, 0.0)),
            "non-finite",
        ),
        ("localRot0", Gf.Quatf(2.0, Gf.Vec3f(0.0)), "unit quaternion"),
    ],
)
def test_invalid_authored_joint_frame_quaternions_fail_closed(
    tmp_path: Path,
    field: str,
    quaternion: Any,
    expected_detail: str,
) -> None:
    source, reference = _write_pair(tmp_path)
    stage = Usd.Stage.Open(str(reference))
    drawer = UsdPhysics.PrismaticJoint(stage.GetPrimAtPath("/World/Joints/drawer"))
    attributes = {
        "localRot0": drawer.GetLocalRot0Attr(),
        "localRot1": drawer.GetLocalRot1Attr(),
    }
    assert attributes[field].Set(quaternion)
    assert stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "invalid_joint_frame_rotation"
    assert field in caught.value.detail
    assert expected_detail in caught.value.detail


def test_normalized_float_quaternion_and_sign_equivalent_frame_are_accepted(
    tmp_path: Path,
) -> None:
    source, reference = _write_pair(tmp_path)
    stage = Usd.Stage.Open(str(reference))
    drawer = UsdPhysics.PrismaticJoint(stage.GetPrimAtPath("/World/Joints/drawer"))
    quaternion = Gf.Quatf(Gf.Rotation(Gf.Vec3d(1.0, 2.0, 3.0), 37.0).GetQuat())
    sign_equivalent = Gf.Quatf(
        -quaternion.GetReal(),
        -quaternion.GetImaginary(),
    )
    assert drawer.GetLocalRot0Attr().Set(quaternion)
    assert drawer.GetLocalRot1Attr().Set(sign_equivalent)
    assert stage.GetRootLayer().Save()

    result = _extract(source, reference)

    axis = _joint(result, "/World/Joints/drawer").topology.axis_stage
    assert axis is not None
    assert math.isclose(math.sqrt(sum(component**2 for component in axis)), 1.0)


@pytest.mark.parametrize(
    "mutation",
    ["meters_per_unit", "kilograms_per_unit", "up_axis"],
)
def test_paired_stage_units_and_spatial_metadata_must_match(
    tmp_path: Path,
    mutation: str,
) -> None:
    source, reference = _write_pair(tmp_path)
    source_stage = Usd.Stage.Open(str(source))
    if mutation == "meters_per_unit":
        UsdGeom.SetStageMetersPerUnit(source_stage, 1.0)
    elif mutation == "kilograms_per_unit":
        UsdPhysics.SetStageKilogramsPerUnit(source_stage, 100.0)
    else:
        UsdGeom.SetStageUpAxis(source_stage, UsdGeom.Tokens.z)
    assert source_stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "stage_metadata_mismatch"
    assert mutation.split("_")[0] in caught.value.detail


def test_matching_nondefault_stage_mass_units_are_accepted(tmp_path: Path) -> None:
    source, reference = _write_pair(tmp_path)
    for path in (source, reference):
        stage = Usd.Stage.Open(str(path))
        UsdPhysics.SetStageKilogramsPerUnit(stage, 2.0)
        assert stage.GetRootLayer().Save()

    result = _extract(source, reference)

    base = next(
        body for body in result.plan.rigid_bodies if body.prim_path == "/World/base"
    )
    assert base.mass is not None
    assert base.mass.mass_kg == 4.0
    assert _joint(result, "/World/Joints/drawer").drive is not None


@pytest.mark.parametrize("mutation", ["translate", "rotate"])
def test_paired_endpoint_world_transforms_must_match(
    tmp_path: Path,
    mutation: str,
) -> None:
    source, reference = _write_pair(tmp_path)
    reference_stage = Usd.Stage.Open(str(reference))
    drawer = UsdGeom.Xformable(reference_stage.GetPrimAtPath("/World/drawer"))
    if mutation == "translate":
        drawer.AddTranslateOp().Set(Gf.Vec3d(1.0, 0.0, 0.0))
    else:
        drawer.AddRotateZOp().Set(15.0)
    assert reference_stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "endpoint_transform_mismatch"
    assert "/World/drawer" in caught.value.detail


def test_matching_nonidentity_endpoint_world_transforms_are_accepted(
    tmp_path: Path,
) -> None:
    source, reference = _write_pair(tmp_path)
    for path in (source, reference):
        stage = Usd.Stage.Open(str(path))
        UsdGeom.Xformable(stage.GetPrimAtPath("/World")).AddTranslateOp().Set(
            Gf.Vec3d(1.0, 2.0, 3.0)
        )
        assert stage.GetRootLayer().Save()

    result = _extract(source, reference)

    anchor = _joint(result, "/World/Joints/drawer").anchor
    assert anchor is not None
    assert anchor.position_stage == (11.0, 2.0, 3.0)


def test_large_coordinate_endpoint_drift_uses_absolute_tolerance(
    tmp_path: Path,
) -> None:
    source, reference = _write_pair(tmp_path)
    for path, translation in (
        (source, 1_000_000.0),
        (reference, 1_000_001.0),
    ):
        stage = Usd.Stage.Open(str(path))
        UsdGeom.Xformable(stage.GetPrimAtPath("/World/base")).AddTranslateOp().Set(
            Gf.Vec3d(translation, 0.0, 0.0)
        )
        assert stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(
            source,
            reference,
            joint_paths=("/World/Joints/door",),
            allowed_omitted_joint_types=("revolute", "prismatic", "spherical"),
        )

    assert caught.value.code == "endpoint_transform_mismatch"
    assert "/World/base" in caught.value.detail


def test_matching_large_coordinate_endpoint_transforms_are_accepted(
    tmp_path: Path,
) -> None:
    source, reference = _write_pair(tmp_path)
    for path in (source, reference):
        stage = Usd.Stage.Open(str(path))
        UsdGeom.Xformable(stage.GetPrimAtPath("/World/base")).AddTranslateOp().Set(
            Gf.Vec3d(1_000_000.0, 0.0, 0.0)
        )
        assert stage.GetRootLayer().Save()

    result = _extract(
        source,
        reference,
        joint_paths=("/World/Joints/door",),
        allowed_omitted_joint_types=("revolute", "prismatic", "spherical"),
    )

    assert [joint.topology.joint_id for joint in result.plan.joints] == [
        "/World/Joints/door"
    ]


@pytest.mark.parametrize("artifact", ["source", "reference"])
@pytest.mark.parametrize("prim_path", ["/World/drawer", "/World"])
def test_time_sampled_endpoint_transform_chains_fail_closed(
    tmp_path: Path,
    artifact: str,
    prim_path: str,
) -> None:
    source, reference = _write_pair(tmp_path)
    stage_path = source if artifact == "source" else reference
    stage = Usd.Stage.Open(str(stage_path))
    translate = UsdGeom.Xformable(stage.GetPrimAtPath(prim_path)).AddTranslateOp()
    translate.Set(Gf.Vec3d(0.0), Usd.TimeCode(1.0))
    translate.Set(Gf.Vec3d(1.0, 0.0, 0.0), Usd.TimeCode(2.0))
    assert stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "time_varying_endpoint_transform"
    assert artifact in caught.value.detail
    assert prim_path in caught.value.detail


def test_static_endpoint_transform_guard_skips_nonxformable_prims(
    tmp_path: Path,
) -> None:
    source, _ = _write_pair(tmp_path)
    stage = Usd.Stage.Open(str(source))
    UsdGeom.Scope.Define(stage, "/World/metadata")

    rv._require_static_endpoint_transform(
        stage,
        path="/World/metadata",
        endpoint="body0",
        joint_path="/World/Joints/example",
        stage_label="source",
        UsdGeom=UsdGeom,
    )


def test_reference_articulation_root_may_be_an_ancestor_of_planned_bodies(
    tmp_path: Path,
) -> None:
    source, reference = _write_pair(tmp_path, articulation_root_on_ancestor=True)

    result = extract_reference_input(
        source,
        reference,
        source_uri=SOURCE_URI,
        reference_uri=REFERENCE_URI,
    )

    assert result.plan.articulation_root is not None
    assert result.plan.articulation_root.prim_path == "/World"


def test_reference_articulation_root_must_resolve_in_paired_source(
    tmp_path: Path,
) -> None:
    source, reference = _write_pair(tmp_path, articulation_root_on_ancestor=True)
    value = extract_reference_input(
        source,
        reference,
        source_uri=SOURCE_URI,
        reference_uri=REFERENCE_URI,
    )
    assert value.plan.articulation_root is not None
    reference_identity = value.plan.articulation_root.provenance.artifact
    assert reference_identity is not None
    reference_stage = Usd.Stage.Open(str(reference))

    with pytest.raises(JointRiggerContractError) as caught:
        rv._extract_articulation_root(
            reference_stage,
            source_stage=Usd.Stage.CreateInMemory(),
            body_paths={"/World/base", "/World/drawer"},
            joint_paths={"/World/Joints/drawer"},
            reference_identity=reference_identity,
            UsdPhysics=UsdPhysics,
        )

    assert caught.value.code == "articulation_root_not_in_source"


@pytest.mark.parametrize(
    "root_path",
    ["/World/Joints/door", "/World/Joints"],
)
def test_joint_associated_articulation_roots_fail_when_v1_cannot_represent_them(
    tmp_path: Path,
    root_path: str,
) -> None:
    source, reference = _write_pair(tmp_path)
    stage = Usd.Stage.Open(str(reference))
    stage.GetPrimAtPath("/World/base").RemoveAppliedSchema("PhysicsArticulationRootAPI")
    UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath(root_path))
    assert stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "unsupported_joint_articulation_root"
    assert root_path in caught.value.detail


@pytest.mark.parametrize("keep_body_root", [False, True])
def test_unrelated_articulation_roots_do_not_enter_the_selected_plan(
    tmp_path: Path,
    keep_body_root: bool,
) -> None:
    source, reference = _write_pair(tmp_path)
    for stage_path in (source, reference):
        stage = Usd.Stage.Open(str(stage_path))
        if stage_path == reference and not keep_body_root:
            stage.GetPrimAtPath("/World/base").RemoveAppliedSchema(
                "PhysicsArticulationRootAPI"
            )
        unrelated = UsdGeom.Xform.Define(stage, "/World/unrelatedRoot").GetPrim()
        UsdPhysics.ArticulationRootAPI.Apply(unrelated)
        assert stage.GetRootLayer().Save()

    result = _extract(source, reference)

    if keep_body_root:
        assert result.plan.articulation_root is not None
        assert result.plan.articulation_root.prim_path == "/World/base"
    else:
        assert result.plan.articulation_root is None


def test_body_and_joint_associated_articulation_roots_are_ambiguous(
    tmp_path: Path,
) -> None:
    source, reference = _write_pair(tmp_path)
    stage = Usd.Stage.Open(str(reference))
    UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath("/World/Joints/door"))
    assert stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "contradictory_articulation_roots"
    assert "/World/base" in caught.value.detail
    assert "/World/Joints/door" in caught.value.detail


def test_reference_export_round_trips_and_is_byte_deterministic(
    tmp_path: Path,
) -> None:
    source_a, reference_a = _write_pair(tmp_path / "a", reverse=False)
    first = extract_reference_input(
        source_a,
        reference_a,
        source_uri=SOURCE_URI,
        reference_uri=REFERENCE_URI,
    )
    second = extract_reference_input(
        source_a,
        reference_a,
        source_uri=SOURCE_URI,
        reference_uri=REFERENCE_URI,
    )

    assert canonical_json(first) == canonical_json(second)
    assert canonical_sha256(first) == canonical_sha256(second)
    output = tmp_path / "golden" / "joint_rigger_input.json"
    write_reference_input(output, first)
    assert output.read_text(encoding="utf-8") == canonical_json(first)
    loaded = JointRiggerInputV1.model_validate_json(output.read_text(encoding="utf-8"))
    assert loaded == first
    write_reference_input(output, loaded)
    assert output.read_bytes() == canonical_json(first).encode("utf-8")

    # A physically different layer must retain its different artifact identity,
    # but traversal/definition order never leaks into the semantic tuple order.
    source_b, reference_b = _write_pair(tmp_path / "b", reverse=True)
    reordered = extract_reference_input(
        source_b,
        reference_b,
        source_uri=SOURCE_URI,
        reference_uri=REFERENCE_URI,
    )
    assert [item.topology.joint_id for item in reordered.plan.joints] == [
        item.topology.joint_id for item in first.plan.joints
    ]


def test_identify_usd_artifact_binds_root_and_used_layer_closure(
    tmp_path: Path,
) -> None:
    source, _ = _write_pair(tmp_path)
    dependency = _add_inert_sublayer(source, "dependency.usda")

    first = identify_usd_artifact(source, uri=SOURCE_URI)
    second = identify_usd_artifact(source, uri=SOURCE_URI)

    assert first == second
    assert first.root_sha256 == _sha256(source)
    assert first.dependency_bundle_sha256 is not None
    dependency.write_text(
        dependency.read_text(encoding="utf-8") + "\n# identity mutation\n",
        encoding="utf-8",
    )
    changed = identify_usd_artifact(source, uri=SOURCE_URI)
    assert changed.root_sha256 == first.root_sha256
    assert changed.dependency_bundle_sha256 != first.dependency_bundle_sha256


def test_dependency_helpers_reject_stale_cached_root_without_mutation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root.usda"
    old_dependency = tmp_path / "old.usda"
    new_dependency = tmp_path / "new.usda"
    _write_value_layer(old_dependency, 1)
    _write_value_layer(new_dependency, 2)
    _write_sublayer_root(root, old_dependency.name)
    cached_stage = Usd.Stage.Open(str(root))
    identify_usd_artifact(root, uri="fixture://cache-refresh")

    _write_sublayer_root(root, new_dependency.name)

    for operation in (
        lambda: local_usd_dependency_paths(root),
        lambda: identify_usd_artifact(root, uri="fixture://cache-refresh"),
    ):
        with pytest.raises(JointRiggerContractError) as caught:
            operation()
        assert caught.value.code == "artifact_dependency_cache_stale"
        assert "fresh process" in caught.value.detail
    assert cached_stage.GetPrimAtPath("/World").GetAttribute("identityValue").Get() == 1
    assert not cached_stage.GetRootLayer().dirty


def test_dependency_helpers_reject_stale_cached_nested_layer_without_mutation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root.usda"
    middle = tmp_path / "middle.usda"
    old_dependency = tmp_path / "old.usda"
    new_dependency = tmp_path / "new.usda"
    _write_value_layer(old_dependency, 1)
    _write_value_layer(new_dependency, 2)
    _write_sublayer_root(middle, old_dependency.name)
    _write_sublayer_root(root, middle.name)
    cached_stage = Usd.Stage.Open(str(root))

    _write_sublayer_root(middle, new_dependency.name)

    with pytest.raises(JointRiggerContractError) as caught:
        local_usd_dependency_paths(root)

    assert caught.value.code == "artifact_dependency_cache_stale"
    assert str(middle) in caught.value.detail
    assert cached_stage.GetPrimAtPath("/World").GetAttribute("identityValue").Get() == 1
    assert not cached_stage.GetRootLayer().dirty


def test_dependency_inventory_rejects_multiple_stale_cached_layers(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root.usda"
    middle = tmp_path / "middle.usda"
    old_dependency = tmp_path / "old.usda"
    new_dependency = tmp_path / "new.usda"
    _write_value_layer(old_dependency, 1)
    _write_value_layer(new_dependency, 2)
    _write_sublayer_root(middle, old_dependency.name)
    cached_middle_stage = Usd.Stage.Open(str(middle))
    _write_sublayer_root(root, old_dependency.name)
    cached_root_stage = Usd.Stage.Open(str(root))

    _write_sublayer_root(middle, new_dependency.name)
    _write_sublayer_root(root, middle.name)

    with pytest.raises(JointRiggerContractError) as caught:
        local_usd_dependency_paths(root)

    assert caught.value.code == "artifact_dependency_cache_stale"
    assert (
        cached_middle_stage.GetPrimAtPath("/World").GetAttribute("identityValue").Get()
        == 1
    )
    assert (
        cached_root_stage.GetPrimAtPath("/World").GetAttribute("identityValue").Get()
        == 1
    )
    assert not cached_middle_stage.GetRootLayer().dirty
    assert not cached_root_stage.GetRootLayer().dirty


def test_dependency_inventory_accepts_clean_cached_layers_without_mutation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root.usda"
    dependency = tmp_path / "dependency.usda"
    _write_value_layer(dependency, 1)
    _write_sublayer_root(root, dependency.name)
    cached_stage = Usd.Stage.Open(str(root))
    root_layer = cached_stage.GetRootLayer()
    before = root_layer.ExportToString()

    paths = set(local_usd_dependency_paths(root))

    assert paths == {root.resolve(), dependency.resolve()}
    assert root_layer.ExportToString() == before
    assert not root_layer.dirty


def test_dependency_refresh_rejects_dirty_cached_layer_without_discarding_edits(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root.usda"
    _write_sublayer_root(root)
    cached_stage = Usd.Stage.Open(str(root))
    UsdGeom.Scope.Define(cached_stage, "/Unsaved")
    root_layer = cached_stage.GetRootLayer()
    assert root_layer.dirty

    with pytest.raises(JointRiggerContractError) as caught:
        local_usd_dependency_paths(root)

    assert caught.value.code == "artifact_dependency_cache_dirty"
    assert root_layer.dirty
    assert cached_stage.GetPrimAtPath("/Unsaved").IsValid()


def test_dependency_refresh_rejects_initially_dirty_layer_directly() -> None:
    class DirtyLayer:
        dirty = True

    class LayerApi:
        @staticmethod
        def OpenAsAnonymous(_identifier: str) -> None:
            raise AssertionError("dirty layers must fail before a fresh read")

    class FakeSdf:
        Layer = LayerApi

    with pytest.raises(JointRiggerContractError) as caught:
        rv._require_layers_current_for_read(
            [DirtyLayer()],
            identifiers=["dirty.usda"],
            Sdf=FakeSdf,
        )

    assert caught.value.code == "artifact_dependency_cache_dirty"


def test_dependency_inventory_detects_layer_becoming_dirty_between_checks() -> None:
    class RacingLayer:
        dirty = False
        unsaved_value: str | None = None

        def ExportToString(self) -> str:
            return self.unsaved_value or "disk value"

    layer = RacingLayer()

    class FreshLayer:
        @staticmethod
        def ExportToString() -> str:
            return "disk value"

    class LayerApi:
        reload_called = False

        @staticmethod
        def OpenAsAnonymous(_identifier: str) -> FreshLayer:
            # A writer wins after the first dirty check and before comparison.
            layer.unsaved_value = "must survive"
            layer.dirty = True
            return FreshLayer()

        @classmethod
        def ReloadLayers(cls, *_args: Any) -> bool:
            cls.reload_called = True
            raise AssertionError("non-destructive validation must never reload")

    class FakeSdf:
        Layer = LayerApi

    with pytest.raises(JointRiggerContractError) as caught:
        rv._require_layers_current_for_read(
            [layer],
            identifiers=["racing.usda"],
            Sdf=FakeSdf,
        )

    assert caught.value.code == "artifact_dependency_cache_dirty"
    assert not LayerApi.reload_called
    assert layer.unsaved_value == "must survive"
    assert layer.dirty


def test_dependency_inventory_detects_layer_becoming_dirty_during_export() -> None:
    class RacingLayer:
        dirty = False

        @staticmethod
        def ExportToString() -> str:
            return "same bytes"

    layer = RacingLayer()

    class FreshLayer:
        @staticmethod
        def ExportToString() -> str:
            # A writer wins after the pre-export dirty check. Keeping the same
            # serialized bytes proves dirty state itself must be rechecked.
            layer.dirty = True
            return "same bytes"

    class LayerApi:
        @staticmethod
        def OpenAsAnonymous(_identifier: str) -> FreshLayer:
            return FreshLayer()

    class FakeSdf:
        Layer = LayerApi

    with pytest.raises(JointRiggerContractError) as caught:
        rv._require_layers_current_for_read(
            [layer],
            identifiers=["racing-export.usda"],
            Sdf=FakeSdf,
        )

    assert caught.value.code == "artifact_dependency_cache_dirty"
    assert "while comparing" in caught.value.detail
    assert layer.dirty


@pytest.mark.parametrize(
    ("scenario", "expected_code"),
    [
        ("missing_root", "artifact_dependency_enumeration_failed"),
        ("anonymous_root", None),
        ("missing_identifier", "artifact_dependency_refresh_failed"),
        ("fresh_open_failure", "artifact_dependency_refresh_failed"),
        ("stale_cache", "artifact_dependency_cache_stale"),
    ],
)
def test_dependency_refresh_defensive_layer_states(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
    expected_code: str | None,
) -> None:
    class Layer:
        anonymous = scenario == "anonymous_root"
        identifier = "" if scenario == "missing_identifier" else "fixture.usda"
        resolvedPath = None
        realPath = None
        dirty = False

        @staticmethod
        def ExportToString() -> str:
            return "cached"

    root_layer = None if scenario == "missing_root" else Layer()

    class LayerApi:
        @staticmethod
        def FindOrOpen(_path: str) -> Layer | None:
            return root_layer

        @staticmethod
        def OpenAsAnonymous(_identifier: str) -> Any:
            if scenario == "fresh_open_failure":
                return None

            class FreshLayer:
                @staticmethod
                def ExportToString() -> str:
                    return "fresh" if scenario == "stale_cache" else "cached"

            return FreshLayer()

    class FakeSdf:
        Layer = LayerApi

    class FakeAr:
        @staticmethod
        def IsPackageRelativePath(_path: str) -> bool:
            return False

        @staticmethod
        def SplitPackageRelativePathOuter(_path: str) -> tuple[str, str]:
            raise AssertionError("non-package paths must not be split")

    class FakeUsdUtils:
        @staticmethod
        def ComputeAllDependencies(
            _path: str,
        ) -> tuple[list[Any], list[Any], list[Any]]:
            return [], [], []

    pxr = __import__("pxr")
    monkeypatch.setattr(pxr, "Ar", FakeAr)
    monkeypatch.setattr(pxr, "Sdf", FakeSdf)
    monkeypatch.setattr(pxr, "UsdUtils", FakeUsdUtils)

    if expected_code is None:
        assert rv._fresh_usd_dependency_inventory(tmp_path / "root.usda") == (
            [],
            [],
            [],
        )
        return

    with pytest.raises(JointRiggerContractError) as caught:
        rv._fresh_usd_dependency_inventory(tmp_path / "root.usda")

    assert caught.value.code == expected_code


def test_identify_usd_artifact_binds_dependency_locator_to_content(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root.usda"
    first_dependency = tmp_path / "first.usda"
    second_dependency = tmp_path / "second.usda"
    _write_value_layer(first_dependency, 1)
    _write_value_layer(second_dependency, 2)
    root_layer = Sdf.Layer.CreateNew(str(root))
    root_layer.subLayerPaths = [first_dependency.name, second_dependency.name]
    assert root_layer.Save()

    before = identify_usd_artifact(root, uri="fixture://dependency-swap")
    first_bytes = first_dependency.read_bytes()
    second_bytes = second_dependency.read_bytes()
    first_dependency.write_bytes(second_bytes)
    second_dependency.write_bytes(first_bytes)
    _reload_layers(root, first_dependency, second_dependency)
    after = identify_usd_artifact(root, uri="fixture://dependency-swap")

    assert after.root_sha256 == before.root_sha256
    assert after.dependency_bundle_sha256 != before.dependency_bundle_sha256


@pytest.mark.parametrize("root_suffix", [".usda", ".usdc"])
@pytest.mark.parametrize(
    ("dependency_kind", "expected"),
    [
        ("device", "error:dependency_artifact_invalid"),
        ("fifo", "error:dependency_artifact_invalid"),
        ("symlink_fifo", "error:dependency_artifact_invalid"),
        ("symlink", "ok"),
        ("regular", "ok"),
    ],
)
def test_identify_usd_artifact_preflights_local_dependency_special_files(
    tmp_path: Path,
    root_suffix: str,
    dependency_kind: str,
    expected: str,
) -> None:
    root = tmp_path / f"root{root_suffix}"
    dependency = tmp_path / "dependency.usda"
    if dependency_kind == "device":
        if not Path("/dev/zero").exists():
            pytest.skip("requires the Linux /dev/zero character device")
        locator = "/dev/zero:SDF_FORMAT_ARGS:format=usda"
    elif dependency_kind in {"fifo", "symlink_fifo"}:
        fifo = tmp_path / "dependency.fifo"
        os.mkfifo(fifo)
        if dependency_kind == "symlink_fifo":
            alias = tmp_path / "dependency-alias.usda"
            alias.symlink_to(fifo.name)
            locator = alias.name
        else:
            locator = f"{fifo.name}:SDF_FORMAT_ARGS:format=usda"
    else:
        _write_value_layer(dependency, 1)
        if dependency_kind == "symlink":
            alias = tmp_path / "dependency-alias.usda"
            alias.symlink_to(dependency.name)
            locator = alias.name
        else:
            locator = dependency.name
    root_layer = Sdf.Layer.CreateNew(str(root))
    root_layer.subLayerPaths = [locator]
    assert root_layer.Save()

    result = _identify_artifact_subprocess(root)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == expected


def test_identify_usd_artifact_binds_symlink_locator_to_target_content(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root.usda"
    first_target = tmp_path / "first-target.usda"
    second_target = tmp_path / "second-target.usda"
    first_link = tmp_path / "first-link.usda"
    second_link = tmp_path / "second-link.usda"
    _write_value_layer(first_target, 1)
    _write_value_layer(second_target, 2)
    first_link.symlink_to(first_target.name)
    second_link.symlink_to(second_target.name)
    root_layer = Sdf.Layer.CreateNew(str(root))
    root_layer.subLayerPaths = [first_link.name, second_link.name]
    assert root_layer.Save()

    before = identify_usd_artifact(root, uri="fixture://symlink-swap")
    first_link.unlink()
    second_link.unlink()
    first_link.symlink_to(second_target.name)
    second_link.symlink_to(first_target.name)
    after = identify_usd_artifact(root, uri="fixture://symlink-swap")

    assert after.root_sha256 == before.root_sha256
    assert after.dependency_bundle_sha256 != before.dependency_bundle_sha256


def test_extract_reference_input_preflights_reference_before_composition(
    tmp_path: Path,
) -> None:
    if not Path("/dev/zero").exists():
        pytest.skip("requires the Linux /dev/zero character device")
    source, reference = _write_pair(tmp_path)
    reference_layer = Sdf.Layer.FindOrOpen(str(reference))
    assert reference_layer is not None
    reference_layer.subLayerPaths.append("/dev/zero:SDF_FORMAT_ARGS:format=usda")
    assert reference_layer.Save()

    result = _extract_reference_subprocess(source, reference)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == (
        "error:dependency_artifact_invalid"
    )


def test_local_dependency_paths_preflights_before_inventory(tmp_path: Path) -> None:
    if not Path("/dev/zero").exists():
        pytest.skip("requires the Linux /dev/zero character device")
    root = tmp_path / "root.usda"
    root_layer = Sdf.Layer.CreateNew(str(root))
    root_layer.subLayerPaths = ["/dev/zero:SDF_FORMAT_ARGS:format=usda"]
    assert root_layer.Save()

    result = _local_dependency_paths_subprocess(root)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == (
        "error:dependency_artifact_invalid"
    )


@pytest.mark.parametrize(
    ("operation", "expected_code"),
    [
        ("identify", "artifact_dependency_mutated"),
        ("local_paths", "artifact_dependency_mutated"),
    ],
)
def test_private_projection_rejects_post_copy_dependency_fifo_swap(
    tmp_path: Path,
    operation: str,
    expected_code: str,
) -> None:
    root = tmp_path / "root.usda"
    dependency = tmp_path / "dependency.usda"
    _write_value_layer(dependency, 1)
    _write_sublayer_root(root, dependency.name)

    result = _post_projection_fifo_swap_subprocess(
        operation,
        root,
        dependency,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == f"error:{expected_code}"


@pytest.mark.parametrize("operation", ["identify", "local_paths"])
def test_private_projection_preserves_root_code_after_fifo_swap(
    tmp_path: Path,
    operation: str,
) -> None:
    root = tmp_path / "root.usda"
    _write_sublayer_root(root)

    result = _post_projection_fifo_swap_subprocess(operation, root, root)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "error:artifact_mutated"


@pytest.mark.parametrize(
    ("target_name", "expected_code"),
    [
        ("source", "source_dependency_artifact_mutated"),
        ("reference", "reference_dependency_artifact_mutated"),
    ],
)
def test_extraction_projection_rejects_post_copy_dependency_fifo_swap(
    tmp_path: Path,
    target_name: str,
    expected_code: str,
) -> None:
    source, reference = _write_pair(tmp_path)
    source_dependency = _add_inert_sublayer(source, "source-dependency.usda")
    reference_dependency = _add_inert_sublayer(
        reference,
        "reference-dependency.usda",
    )
    dependency = source_dependency if target_name == "source" else reference_dependency

    result = _post_projection_fifo_swap_subprocess(
        "extract",
        source,
        dependency,
        secondary=reference,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == f"error:{expected_code}"


@pytest.mark.parametrize(
    ("target_name", "expected_code"),
    [
        ("source", "source_artifact_mutated"),
        ("reference", "reference_artifact_mutated"),
    ],
)
def test_extraction_projection_preserves_root_code_after_fifo_swap(
    tmp_path: Path,
    target_name: str,
    expected_code: str,
) -> None:
    source, reference = _write_pair(tmp_path)
    target = source if target_name == "source" else reference

    result = _post_projection_fifo_swap_subprocess(
        "extract",
        source,
        target,
        secondary=reference,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == f"error:{expected_code}"


def test_private_projection_preflight_terminates_for_cyclic_sublayers(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.usda"
    second = tmp_path / "second.usda"
    _write_sublayer_root(first, second.name)
    _write_sublayer_root(second, first.name)

    result = _preflight_dependency_subprocess(first)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "ok"


def test_projection_seals_active_resolver_asset_and_preserves_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene_directory = tmp_path / "scene"
    scene_directory.mkdir()
    root = scene_directory / "root.usda"
    resolver_asset = tmp_path / "OmniPBR.mdl"
    resolver_asset.write_bytes(b"resolver-backed MDL")
    _write_resolver_asset_root(root, resolver_asset.name)
    monkeypatch.chdir(tmp_path)

    identity = identify_usd_artifact(root, uri="fixture://resolver-backed")
    structure = rv._capture_dependency_structure(
        root,
        logical_artifact_path=root,
    )
    frozen_records = tuple(
        rv._CapturedDependencyIdentityRecord(
            kind=record.kind,
            locator=record.locator,
            sha256=(
                _sha256(root)
                if record.backing_path is None
                else _sha256(record.backing_path)
            ),
            backing_path=record.backing_path,
        )
        for record in structure
    )
    captured = rv._artifact_identity_from_captured_records(
        logical_artifact_path=root,
        uri=identity.uri,
        root_sha256=identity.root_sha256,
        records=frozen_records,
    )

    assert set(local_usd_dependency_paths(root)) == {
        root.resolve(),
        resolver_asset.resolve(),
    }
    assert any(
        record.kind == "asset" and record.backing_path == resolver_asset.resolve()
        for record in structure
    )
    assert captured == identity


def test_projection_identity_includes_transitive_opaque_dependencies(
    tmp_path: Path,
) -> None:
    assert rv._BUNDLE_SCHEMA == "world-understanding-usd-dependency-bundle-v3"
    root = tmp_path / "root.usda"
    main = tmp_path / "Main.mdl"
    peer = tmp_path / "Peer.mdl"
    textures = tmp_path / "textures"
    texture = textures / "albedo.png"
    textures.mkdir()
    _write_resolver_asset_root(root, main.name)
    main.write_text("mdl 1.7;\nimport Peer::*;\n", encoding="utf-8")
    peer.write_text(
        'mdl 1.7;\nimport ::df::*;\ntexture_2d("textures/albedo.png");\n',
        encoding="utf-8",
    )
    texture.write_bytes(b"real texture bytes")

    identity = identify_usd_artifact(root, uri="fixture://opaque-closure")
    structure = rv._capture_dependency_structure(
        root,
        logical_artifact_path=root,
    )
    captured = rv._artifact_identity_from_captured_records(
        logical_artifact_path=root,
        uri=identity.uri,
        root_sha256=identity.root_sha256,
        records=tuple(
            rv._CapturedDependencyIdentityRecord(
                kind=record.kind,
                locator=record.locator,
                sha256=(
                    _sha256(root)
                    if record.backing_path is None
                    else _sha256(record.backing_path)
                ),
                backing_path=record.backing_path,
            )
            for record in structure
        ),
    )

    assert set(local_usd_dependency_paths(root)) == {
        root.resolve(),
        main.resolve(),
        peer.resolve(),
        texture.resolve(),
    }
    assert {
        (record.kind, record.locator)
        for record in structure
        if record.kind == "opaque_asset"
    } == {
        ("opaque_asset", "Peer.mdl"),
        ("opaque_asset", "textures/albedo.png"),
    }
    assert captured == identity

    peer.write_text(
        'mdl 1.7;\nimport ::df::*;\ntexture_2d("textures/albedo.png");\n'
        "// changed bytes\n",
        encoding="utf-8",
    )
    changed = identify_usd_artifact(root, uri=identity.uri)
    assert changed.root_sha256 == identity.root_sha256
    assert changed.dependency_bundle_sha256 != identity.dependency_bundle_sha256


def test_projection_identity_supports_runtime_using_and_sibling_resources(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle"
    materials = bundle / "materials"
    textures = bundle / "textures"
    materials.mkdir(parents=True)
    textures.mkdir()
    root = bundle / "root.usda"
    main = materials / "Main.mdl"
    texture = textures / "albedo.png"
    _write_resolver_asset_root(root, "materials/Main.mdl")
    main.write_text(
        "mdl 1.7;\n"
        "using ::OmniPBR import OmniPBR;\n"
        'texture_2d("../textures/albedo.png");\n',
        encoding="utf-8",
    )
    texture.write_bytes(b"real sibling texture bytes")

    identity = identify_usd_artifact(root, uri="fixture://opaque-using-runtime")

    assert set(local_usd_dependency_paths(root)) == {
        root.resolve(),
        main.resolve(),
        texture.resolve(),
    }
    assert identity.root_sha256 == _sha256(root)


@pytest.mark.parametrize("symlink_kind", ["leaf", "ancestor"])
def test_projection_rejects_opaque_dependency_symlink_escape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    symlink_kind: Literal["leaf", "ancestor"],
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    root = bundle / "root.usda"
    main = bundle / "Main.mdl"
    textures = bundle / "textures"
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_texture = outside / "albedo.png"
    outside_texture.write_bytes(b"off-bundle texture bytes")
    _write_resolver_asset_root(root, main.name)
    main.write_text(
        'mdl 1.7;\ntexture_2d("textures/albedo.png");\n',
        encoding="utf-8",
    )
    if symlink_kind == "leaf":
        textures.mkdir()
        (textures / "albedo.png").symlink_to(outside_texture)
    else:
        textures.symlink_to(outside, target_is_directory=True)
    _forbid_file_payload_read(monkeypatch, outside_texture)

    with pytest.raises(JointRiggerContractError) as caught:
        identify_usd_artifact(root, uri=f"fixture://opaque-{symlink_kind}-escape")

    assert caught.value.code == "artifact_dependency_preflight_failed"
    assert "opaque material dependency traverses a symlink" in str(caught.value)


def test_projection_rejects_in_tree_opaque_dependency_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = tmp_path / "bundle"
    textures = bundle / "textures"
    textures.mkdir(parents=True)
    root = bundle / "root.usda"
    main = bundle / "Main.mdl"
    actual = textures / "actual.png"
    actual.write_bytes(b"in-tree texture bytes")
    (textures / "albedo.png").symlink_to(actual.name)
    _write_resolver_asset_root(root, main.name)
    main.write_text(
        'mdl 1.7;\ntexture_2d("textures/albedo.png");\n',
        encoding="utf-8",
    )
    _forbid_file_payload_read(monkeypatch, actual)

    with pytest.raises(JointRiggerContractError) as caught:
        identify_usd_artifact(root, uri="fixture://opaque-in-tree-symlink")

    assert caught.value.code == "artifact_dependency_preflight_failed"
    assert "opaque material dependency traverses a symlink" in str(caught.value)


@pytest.mark.parametrize("symlink_kind", ["leaf", "ancestor"])
def test_projection_rejects_initial_opaque_document_symlink_before_hashing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    symlink_kind: Literal["leaf", "ancestor"],
) -> None:
    bundle = tmp_path / "bundle"
    outside = tmp_path / "outside"
    bundle.mkdir()
    outside.mkdir()
    root = bundle / "root.usda"
    if symlink_kind == "leaf":
        outside_main = outside / "Main.mdl"
        locator = "Main.mdl"
        (bundle / locator).symlink_to(outside_main)
    else:
        outside_materials = outside / "materials"
        outside_materials.mkdir()
        outside_main = outside_materials / "Main.mdl"
        locator = "materials/Main.mdl"
        (bundle / "materials").symlink_to(
            outside_materials,
            target_is_directory=True,
        )
    outside_main.write_text("mdl 1.7;\n", encoding="utf-8")
    _write_resolver_asset_root(root, locator)
    _forbid_file_payload_read(monkeypatch, outside_main)

    with pytest.raises(JointRiggerContractError) as caught:
        identify_usd_artifact(
            root,
            uri=f"fixture://initial-opaque-{symlink_kind}-escape",
        )

    assert caught.value.code == "artifact_dependency_preflight_failed"
    assert "opaque material dependency traverses a symlink" in str(caught.value)


@pytest.mark.parametrize("document_kind", ["initial", "descendant"])
def test_projection_rejects_oversized_opaque_document_before_hashing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    document_kind: Literal["initial", "descendant"],
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    root = bundle / "root.usda"
    main = bundle / "Main.mdl"
    _write_resolver_asset_root(root, main.name)
    if document_kind == "initial":
        oversized = main
    else:
        main.write_text("mdl 1.7;\nimport Child::*;\n", encoding="utf-8")
        oversized = bundle / "Child.mdl"
    oversized.write_bytes(b"x" * 33)
    monkeypatch.setattr(rv, "_MAX_OPAQUE_DOCUMENT_BYTES", 32)
    _forbid_file_payload_read(monkeypatch, oversized)

    with pytest.raises(JointRiggerContractError) as caught:
        identify_usd_artifact(
            root,
            uri=f"fixture://oversized-opaque-{document_kind}",
        )

    assert caught.value.code == "artifact_dependency_preflight_failed"
    assert "exceeds the 32-byte limit" in str(caught.value)


def test_opaque_projection_skips_duplicate_pending_document(tmp_path: Path) -> None:
    document = tmp_path / "Main.mdl"
    document.write_text("mdl 1.7;\n", encoding="utf-8")
    projection = _test_projection(tmp_path / "projection")
    root = tmp_path / "root.usda"

    class DuplicateIterationClosure(set[Path]):
        def __iter__(self) -> Iterator[Path]:
            values = tuple(super().__iter__())
            return iter((*values, *values))

    closure = DuplicateIterationClosure({document})
    rv._populate_opaque_projection_closure(
        projection,
        root=root,
        closure=closure,
    )

    assert set(projection.files) == {document}
    assert projection.opaque_dependencies[root] == set()


def test_opaque_projection_enforces_document_count_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "First.mdl"
    second = tmp_path / "Second.mdl"
    first.write_text("mdl 1.7;\n", encoding="utf-8")
    second.write_text("mdl 1.7;\n", encoding="utf-8")
    projection = _test_projection(tmp_path / "projection")
    monkeypatch.setattr(rv, "_MAX_OPAQUE_DEPENDENCY_FILES", 1)

    with pytest.raises(JointRiggerContractError) as caught:
        rv._populate_opaque_projection_closure(
            projection,
            root=tmp_path / "root.usda",
            closure={first, second},
        )

    assert caught.value.code == "artifact_dependency_preflight_failed"
    assert "exceeds the 1-file limit" in str(caught.value)
    assert len(projection.files) == 1


def test_opaque_projection_enforces_reference_count_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = tmp_path / "Main.mdl"
    document.write_text(
        'mdl 1.7;\ntexture_2d("first.png");\ntexture_2d("second.png");\n',
        encoding="utf-8",
    )
    projection = _test_projection(tmp_path / "projection")
    monkeypatch.setattr(rv, "_MAX_OPAQUE_DEPENDENCY_REFERENCES", 1)

    with pytest.raises(JointRiggerContractError) as caught:
        rv._populate_opaque_projection_closure(
            projection,
            root=tmp_path / "root.usda",
            closure={document},
        )

    assert caught.value.code == "artifact_dependency_preflight_failed"
    assert "exceeds the 1-reference limit" in str(caught.value)


def test_opaque_projection_rejects_file_outside_allowed_root(tmp_path: Path) -> None:
    allowed_root = tmp_path / "bundle"
    allowed_root.mkdir()
    outside = tmp_path / "outside.mdl"
    outside.write_text("mdl 1.7;\n", encoding="utf-8")
    projection = _test_projection(tmp_path / "projection")

    with pytest.raises(JointRiggerContractError) as caught:
        rv._copy_precomposition_file(
            projection,
            outside,
            opaque_allowed_root=allowed_root,
        )

    assert caught.value.code == "artifact_dependency_preflight_failed"
    assert "escapes its artifact tree" in str(caught.value)
    assert outside not in projection.files


def test_opaque_projection_allows_symlink_ancestor_outside_allowed_root(
    tmp_path: Path,
) -> None:
    actual = tmp_path / "actual"
    bundle = actual / "bundle"
    bundle.mkdir(parents=True)
    alias = tmp_path / "alias"
    alias.symlink_to(actual, target_is_directory=True)
    allowed_root = alias / "bundle"
    document = allowed_root / "Main.mdl"
    (bundle / "Main.mdl").write_text("mdl 1.7;\n", encoding="utf-8")
    projection = _test_projection(tmp_path / "projection")

    record = rv._copy_precomposition_file(
        projection,
        document,
        opaque_allowed_root=allowed_root,
    )

    assert [hop[0] for hop in record.symlink_hops] == [alias]
    assert record.projected_path.read_text(encoding="utf-8") == "mdl 1.7;\n"


def test_dependency_inventory_does_not_duplicate_represented_opaque_asset(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root.usda"
    main = tmp_path / "Main.mdl"
    peer = tmp_path / "Peer.mdl"
    root.write_text(
        "\n".join(
            (
                "#usda 1.0",
                "",
                'def Scope "World"',
                "{",
                "    custom asset test:main = @Main.mdl@",
                "    custom asset test:peer = @Peer.mdl@",
                "}",
                "",
            )
        ),
        encoding="utf-8",
    )
    main.write_text("mdl 1.7;\nimport Peer::*;\n", encoding="utf-8")
    peer.write_text("mdl 1.7;\nimport ::df::*;\n", encoding="utf-8")

    structure = rv._capture_dependency_structure(
        root,
        logical_artifact_path=root,
    )

    assert [
        record.kind for record in structure if record.backing_path == peer.resolve()
    ] == ["asset"]


def test_projection_rejects_resolver_asset_fifo_swap(tmp_path: Path) -> None:
    scene_directory = tmp_path / "scene"
    scene_directory.mkdir()
    root = scene_directory / "root.usda"
    resolver_asset = tmp_path / "OmniPBR.mdl"
    resolver_asset.write_bytes(b"resolver-backed MDL")
    _write_resolver_asset_root(root, resolver_asset.name)

    result = _post_projection_fifo_swap_subprocess(
        "identify",
        root,
        resolver_asset,
        resolver_root=tmp_path,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == (
        "error:artifact_dependency_mutated"
    )


def test_projection_isolates_late_created_unresolved_fifo(tmp_path: Path) -> None:
    scene_directory = tmp_path / "scene"
    resolver_root = tmp_path / "resolver-root"
    scene_directory.mkdir()
    resolver_root.mkdir()
    root = scene_directory / "root.usda"
    _write_sublayer_root(root, "late.usda")

    result = _post_projection_fifo_swap_subprocess(
        "identify",
        root,
        resolver_root / "late.usda",
        resolver_root=resolver_root,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == (
        "error:unresolved_artifact_dependency"
    )


def test_projection_identifier_mapping_preserves_passthrough_and_package_args(
    tmp_path: Path,
) -> None:
    projection = _test_projection(tmp_path)
    projected_package = projection.mirror_root / "tmp" / "library.usdz"
    package_identifier = Ar.JoinPackageRelativePath(
        str(projected_package),
        "nested/root.usda",
    )
    package_identifier = Sdf.Layer.CreateIdentifier(
        package_identifier,
        {"format": "usda"},
    )

    mapped = projection.original_identifier(
        package_identifier,
        Ar=Ar,
        Sdf=Sdf,
    )
    mapped_path, mapped_arguments = Sdf.Layer.SplitIdentifier(mapped)
    mapped_outer, mapped_inner = Ar.SplitPackageRelativePathOuter(mapped_path)

    assert mapped_outer == "/tmp/library.usdz"
    assert mapped_inner == "nested/root.usda"
    assert mapped_arguments == {"format": "usda"}
    for passthrough in (
        "",
        "resolver://remote/layer.usda",
        "relative/layer.usda",
        str(tmp_path / "outside.usda"),
    ):
        assert (
            projection.original_identifier(passthrough, Ar=Ar, Sdf=Sdf) == passthrough
        )

    class RaisingLayerApi:
        @staticmethod
        def SplitIdentifier(_identifier: str) -> tuple[str, dict[str, str]]:
            raise ValueError("invalid identifier")

    class RaisingSdf:
        Layer = RaisingLayerApi

    assert projection.original_identifier("invalid", Ar=Ar, Sdf=RaisingSdf) == "invalid"


def test_projection_locator_preserves_packages_args_and_resolver_passthrough(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projection = _test_projection(tmp_path / "projection-owner")
    owner = tmp_path / "scene" / "root.usda"
    owner.parent.mkdir()
    owner.write_text("#usda 1.0\n", encoding="utf-8")

    for remote_locator in (
        "resolver://remote/asset.mdl",
        "s:opaque-resolver-asset",
        Sdf.Layer.CreateIdentifier(
            "s:opaque-resolver-layer",
            {"format": "usda"},
        ),
    ):
        assert (
            rv._projected_local_locator(
                remote_locator,
                owner_path=owner,
                projection=projection,
                Ar=Ar,
                Sdf=Sdf,
            )
            is None
        )

    for windows_locator in ("C:/assets/missing.usda", r"C:\assets\missing.usda"):
        assert (
            rv._projected_local_locator(
                windows_locator,
                owner_path=owner,
                projection=projection,
                Ar=Ar,
                Sdf=Sdf,
            )
            is not None
        )

    file_uri_asset = owner.parent / "file uri asset.mdl"
    file_uri_asset.write_bytes(b"canonical file URI asset")
    file_uri = rv._projected_local_locator(
        file_uri_asset.as_uri(),
        owner_path=owner,
        projection=projection,
        Ar=Ar,
        Sdf=Sdf,
    )
    assert file_uri is not None
    assert file_uri.path == file_uri_asset
    assert Path(file_uri.rewritten) == projection.projected_path(file_uri_asset)

    for invalid_file_uri in (
        "embedded\x00nul.usda",
        "file:relative.usda",
        f"file:{file_uri_asset}",
        f"{file_uri_asset.as_uri()}/../{file_uri_asset.name}",
        f"file://remote-host{file_uri_asset}",
        "file:///tmp/embedded%00nul.usda",
        "file://[bad/path",
        "file:////tmp/double-root.usda",
    ):
        with pytest.raises(JointRiggerContractError) as invalid_file:
            rv._projected_local_locator(
                invalid_file_uri,
                owner_path=owner,
                projection=projection,
                Ar=Ar,
                Sdf=Sdf,
            )
        assert invalid_file.value.code == "artifact_dependency_preflight_failed"

    missing_package = tmp_path / "missing.usdz"
    missing_identifier = Sdf.Layer.CreateIdentifier(
        Ar.JoinPackageRelativePath(str(missing_package), "nested/root.usda"),
        {"format": "usda"},
    )
    missing = rv._projected_local_locator(
        missing_identifier,
        owner_path=owner,
        projection=projection,
        Ar=Ar,
        Sdf=Sdf,
    )
    assert missing is not None
    assert missing.path is None
    missing_path, missing_arguments = Sdf.Layer.SplitIdentifier(missing.rewritten)
    missing_outer, missing_inner = Ar.SplitPackageRelativePathOuter(missing_path)
    assert Path(missing_outer) == projection.projected_path(missing_package)
    assert missing_inner == "nested/root.usda"
    assert missing_arguments == {"format": "usda"}

    sibling_asset = owner.parent / "sibling.mdl"
    sibling_asset.write_bytes(b"owner-relative asset")
    sibling = rv._projected_local_locator(
        sibling_asset.name,
        owner_path=owner,
        projection=projection,
        Ar=Ar,
        Sdf=Sdf,
    )
    assert sibling is not None
    assert sibling.path == sibling_asset
    assert Path(sibling.rewritten) == projection.projected_path(sibling_asset)

    unresolved_identifier = Sdf.Layer.CreateIdentifier(
        Ar.JoinPackageRelativePath("Unresolved.usdz", "nested/root.usda"),
        {"format": "usda"},
    )
    unresolved = rv._projected_local_locator(
        unresolved_identifier,
        owner_path=owner,
        projection=projection,
        Ar=Ar,
        Sdf=Sdf,
    )
    assert unresolved is not None
    assert unresolved.path is None
    unresolved_path, unresolved_arguments = Sdf.Layer.SplitIdentifier(
        unresolved.rewritten
    )
    unresolved_outer, unresolved_inner = Ar.SplitPackageRelativePathOuter(
        unresolved_path
    )
    assert Path(unresolved_outer) == projection.projected_path(
        owner.parent / "Unresolved.usdz"
    )
    assert unresolved_inner == "nested/root.usda"
    assert unresolved_arguments == {"format": "usda"}

    search_package = tmp_path / "Search.usdz"
    search_package.write_bytes(b"resolver package")
    monkeypatch.chdir(tmp_path)
    search_identifier = Sdf.Layer.CreateIdentifier(
        Ar.JoinPackageRelativePath(search_package.name, "nested/root.usda"),
        {"format": "usda"},
    )
    resolved = rv._projected_local_locator(
        search_identifier,
        owner_path=owner,
        projection=projection,
        Ar=Ar,
        Sdf=Sdf,
    )
    assert resolved is not None
    assert resolved.path == search_package
    resolved_path, resolved_arguments = Sdf.Layer.SplitIdentifier(resolved.rewritten)
    resolved_outer, resolved_inner = Ar.SplitPackageRelativePathOuter(resolved_path)
    assert Path(resolved_outer) == projection.projected_path(search_package)
    assert resolved_inner == "nested/root.usda"
    assert resolved_arguments == {"format": "usda"}


@pytest.mark.parametrize(
    ("resolved_value", "expected_relative"),
    [
        ("", "search-root-only.mdl"),
        ("relative", "relative"),
        ("resolver://remote/asset", None),
        ("s:opaque-resolved-asset", None),
    ],
)
def test_projection_locator_isolates_local_unresolved_resolver_results(
    tmp_path: Path,
    resolved_value: str,
    expected_relative: str | None,
) -> None:
    projection = _test_projection(tmp_path)
    owner = tmp_path / "nested" / "root.usda"

    class Resolver:
        @staticmethod
        def Resolve(_path: str) -> str:
            return resolved_value

    class FakeAr:
        IsPackageRelativePath = staticmethod(Ar.IsPackageRelativePath)
        SplitPackageRelativePathOuter = staticmethod(Ar.SplitPackageRelativePathOuter)
        JoinPackageRelativePath = staticmethod(Ar.JoinPackageRelativePath)

        @staticmethod
        def GetResolver() -> Resolver:
            return Resolver()

    result = rv._projected_local_locator(
        "search-root-only.mdl",
        owner_path=owner,
        projection=projection,
        Ar=FakeAr,
        Sdf=Sdf,
    )
    if expected_relative is None:
        assert result is None
        return
    assert result is not None
    assert result.path is None
    assert Path(result.rewritten) == projection.projected_path(
        owner.parent / expected_relative
    )


def test_projection_locator_seals_canonical_file_uri_resolver_result(
    tmp_path: Path,
) -> None:
    projection = _test_projection(tmp_path / "projection")
    owner = tmp_path / "nested" / "root.usda"
    resolved_asset = tmp_path / "resolver asset.mdl"
    resolved_asset.write_bytes(b"resolver file URI")

    class Resolver:
        @staticmethod
        def Resolve(_path: str) -> str:
            return resolved_asset.as_uri()

    class FakeAr:
        IsPackageRelativePath = staticmethod(Ar.IsPackageRelativePath)
        SplitPackageRelativePathOuter = staticmethod(Ar.SplitPackageRelativePathOuter)
        JoinPackageRelativePath = staticmethod(Ar.JoinPackageRelativePath)

        @staticmethod
        def GetResolver() -> Resolver:
            return Resolver()

    result = rv._projected_local_locator(
        "search-root-only.mdl",
        owner_path=owner,
        projection=projection,
        Ar=FakeAr,
        Sdf=Sdf,
    )

    assert result is not None
    assert result.path == resolved_asset
    assert Path(result.rewritten) == projection.projected_path(resolved_asset)


def test_projection_preserves_opaque_resolver_package_with_arguments(
    tmp_path: Path,
) -> None:
    projection = _test_projection(tmp_path / "projection")
    owner = tmp_path / "nested" / "root.usda"
    locator = Sdf.Layer.CreateIdentifier(
        Ar.JoinPackageRelativePath("Search.usdz", "nested/root.usda"),
        {"format": "usda"},
    )

    class Resolver:
        @staticmethod
        def Resolve(path: str) -> str:
            assert path == "Search.usdz"
            return "s:opaque-package"

    class FakeAr:
        IsPackageRelativePath = staticmethod(Ar.IsPackageRelativePath)
        SplitPackageRelativePathOuter = staticmethod(Ar.SplitPackageRelativePathOuter)
        JoinPackageRelativePath = staticmethod(Ar.JoinPackageRelativePath)

        @staticmethod
        def GetResolver() -> Resolver:
            return Resolver()

    assert (
        rv._projected_local_locator(
            locator,
            owner_path=owner,
            projection=projection,
            Ar=FakeAr,
            Sdf=Sdf,
        )
        is None
    )


def test_projection_locator_wraps_resolver_failure(
    tmp_path: Path,
) -> None:
    projection = _test_projection(tmp_path)
    owner = tmp_path / "nested" / "root.usda"

    class Resolver:
        @staticmethod
        def Resolve(_path: str) -> str:
            raise RuntimeError("resolver failed")

    class FakeAr:
        IsPackageRelativePath = staticmethod(Ar.IsPackageRelativePath)

        @staticmethod
        def GetResolver() -> Resolver:
            return Resolver()

    with pytest.raises(JointRiggerContractError) as caught:
        rv._projected_local_locator(
            "search-root-only.mdl",
            owner_path=owner,
            projection=projection,
            Ar=FakeAr,
            Sdf=Sdf,
        )

    assert caught.value.code == "artifact_dependency_preflight_failed"


def test_projection_preflight_covers_limits_and_absent_absolute_assets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "root.usda"
    _write_resolver_asset_root(root, str(tmp_path / "absent.mdl"))
    rv._preflight_local_dependency_locators(root)

    projection = _test_projection(tmp_path / "limit")
    fake_state = (1, 1, 0, 1, 1, 1, 1)
    fake_record = rv._ProjectedLocalFile(
        lexical_path=root,
        backing_path=root,
        projected_path=projection.projected_path(root),
        expected_state=fake_state,
        symlink_hops=(),
        sha256="0" * 64,
    )

    def next_dependency(
        _projection: Any,
        _record: Any,
        *,
        owner_path: Path,
        **_kwargs: Any,
    ) -> tuple[Any, ...]:
        dependency = owner_path.with_name(owner_path.name + ".next")
        return (rv._ProjectedLocator(dependency, None, True, dependency.name),)

    monkeypatch.setattr(rv, "_MAX_DEPENDENCY_VALIDATION_PASSES", 1)
    monkeypatch.setattr(
        rv,
        "_copy_precomposition_file",
        lambda *_args, **_kwargs: fake_record,
    )
    monkeypatch.setattr(
        rv, "_precomposition_layer_suffix", lambda *_args, **_kwargs: ".usda"
    )
    monkeypatch.setattr(rv, "_inspect_precomposition_layer", next_dependency)
    with pytest.raises(JointRiggerContractError) as caught:
        rv._populate_projection_root(projection, root)

    assert caught.value.code == "artifact_dependency_preflight_failed"


@pytest.mark.parametrize("race", ["short_read", "growth"])
def test_projection_copy_rejects_descriptor_races_and_cleans_partial_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    race: str,
) -> None:
    target = tmp_path / "dependency.usda"
    target.write_bytes(b"stable dependency bytes")
    projection = _test_projection(tmp_path / "projection")
    real_pread = rv.os.pread
    expected_size = target.stat().st_size

    if race == "short_read":
        monkeypatch.setattr(rv.os, "pread", lambda *_args: b"")
    else:

        def growing_pread(descriptor: int, count: int, offset: int) -> bytes:
            if offset == expected_size:
                return b"x"
            return real_pread(descriptor, count, offset)

        monkeypatch.setattr(rv.os, "pread", growing_pread)

    with pytest.raises(JointRiggerContractError) as caught:
        rv._copy_precomposition_file(projection, target)

    assert caught.value.code == "dependency_artifact_invalid"
    assert not projection.projected_path(target).exists()


@pytest.mark.parametrize(
    ("scenario", "allow_invalid", "expected_code"),
    [
        ("missing_allowed", True, None),
        ("missing_rejected", False, "artifact_dependency_preflight_failed"),
        ("empty_locator", False, None),
        ("remote_passthrough", False, None),
        ("export_failed", False, "artifact_dependency_preflight_failed"),
    ],
)
def test_projected_layer_inspection_defensive_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
    allow_invalid: bool,
    expected_code: str | None,
) -> None:
    owner = tmp_path / "dependency.data"
    projected = tmp_path / "projected.data"
    record = rv._ProjectedLocalFile(
        lexical_path=owner,
        backing_path=owner,
        projected_path=projected,
        expected_state=(1, 1, 0, 1, 1, 1, 1),
        symlink_hops=(),
        sha256="0" * 64,
    )

    class FileFormat:
        formatId = "usda"

    class Layer:
        @staticmethod
        def GetFileFormat() -> FileFormat:
            return FileFormat()

        @staticmethod
        def Export(_path: str, *, args: dict[str, str]) -> bool:
            assert args == {"format": "usda"}
            return False

    class LayerApi:
        SplitIdentifier = staticmethod(Sdf.Layer.SplitIdentifier)

        @staticmethod
        def CreateIdentifier(path: str, _arguments: dict[str, str]) -> str:
            return path + ":format=usda"

        @staticmethod
        def OpenAsAnonymous(_identifier: str) -> Layer | None:
            if scenario.startswith("missing"):
                return None
            return Layer()

        @staticmethod
        def Find(_identifier: str) -> None:
            return None

    class FakeSdf:
        Layer = LayerApi

    class FakeUsdUtils:
        @staticmethod
        def ModifyAssetPaths(
            _layer: Layer,
            callback: Any,
            *,
            keepEmptyPathsInArrays: bool,
        ) -> None:
            assert keepEmptyPathsInArrays
            if scenario == "empty_locator":
                assert callback("") == ""
            elif scenario == "remote_passthrough":
                remote = "resolver://remote/layer.usda"
                assert callback(remote) == remote
            elif scenario == "export_failed":
                assert callback("original") == "projected"

    if scenario == "export_failed":
        monkeypatch.setattr(
            rv,
            "_projected_local_locator",
            lambda *_args, **_kwargs: rv._ProjectedLocator(
                path=None,
                format_hint=None,
                inspect_layer=False,
                rewritten="projected",
            ),
        )

    def operation() -> tuple[Any, ...]:
        return rv._inspect_precomposition_layer(
            _test_projection(tmp_path / "projection"),
            record,
            owner_path=owner,
            format_hint="usda",
            allow_invalid_layer=allow_invalid,
            Ar=Ar,
            Sdf=FakeSdf,
            UsdUtils=FakeUsdUtils,
        )

    if expected_code is None:
        assert operation() == ()
    else:
        with pytest.raises(JointRiggerContractError) as caught:
            operation()
        assert caught.value.code == expected_code


def test_projected_generic_usd_rewrite_uses_inferred_export_format(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = tmp_path / "owner.usd"
    projected = tmp_path / "projected.usd"
    layer = Sdf.Layer.CreateNew(str(projected))
    assert layer is not None
    assert str(layer.GetFileFormat().formatId) == "usd"
    layer.subLayerPaths.append("original.usda")
    layer.Save()

    record = rv._ProjectedLocalFile(
        lexical_path=owner,
        backing_path=owner,
        projected_path=projected,
        expected_state=(1, 1, 0, 1, 1, 1, 1),
        symlink_hops=(),
        sha256="0" * 64,
    )
    monkeypatch.setattr(
        rv,
        "_projected_local_locator",
        lambda *_args, **_kwargs: rv._ProjectedLocator(
            path=None,
            format_hint=None,
            inspect_layer=False,
            rewritten="projected.usda",
        ),
    )

    dependencies = rv._inspect_precomposition_layer(
        _test_projection(tmp_path / "projection"),
        record,
        owner_path=owner,
        format_hint=None,
        allow_invalid_layer=False,
        Ar=Ar,
        Sdf=Sdf,
        UsdUtils=UsdUtils,
    )

    assert len(dependencies) == 1
    rewritten = Sdf.Layer.OpenAsAnonymous(str(projected))
    assert rewritten is not None
    assert list(rewritten.subLayerPaths) == ["projected.usda"]


def test_projection_cache_and_hash_defensive_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RacingCachedLayer:
        dirty_reads = 0

        @property
        def dirty(self) -> bool:
            self.dirty_reads += 1
            return self.dirty_reads >= 2

        @staticmethod
        def ExportToString() -> str:
            return "same"

    cached = RacingCachedLayer()

    class ProjectedLayer:
        @staticmethod
        def ExportToString() -> str:
            return "same"

    class RacingLayerApi:
        @staticmethod
        def CreateIdentifier(path: str, _arguments: dict[str, str]) -> str:
            return path + ":format=usda"

        @staticmethod
        def Find(_identifier: str) -> RacingCachedLayer:
            return cached

    class RacingSdf:
        Layer = RacingLayerApi

    with pytest.raises(JointRiggerContractError) as dirty:
        rv._require_cached_layer_matches_projection(
            tmp_path / "cached.data",
            ProjectedLayer(),
            format_hint="usda",
            Sdf=RacingSdf,
        )
    assert dirty.value.code == "artifact_dependency_cache_dirty"

    record = rv._ProjectedLocalFile(
        lexical_path=tmp_path / "package.usdz",
        backing_path=tmp_path / "package.usdz",
        projected_path=tmp_path / "projected.usdz",
        expected_state=(1, 1, 0, 1, 1, 1, 1),
        symlink_hops=(),
        sha256="0" * 64,
    )

    class MissingPackageLayerApi:
        @staticmethod
        def Find(_identifier: str) -> object:
            return object()

        @staticmethod
        def OpenAsAnonymous(_identifier: str) -> None:
            return None

    class MissingPackageSdf:
        Layer = MissingPackageLayerApi

    with pytest.raises(JointRiggerContractError) as missing_package:
        rv._require_cached_package_matches_projection(
            record.lexical_path,
            record,
            Sdf=MissingPackageSdf,
        )
    assert missing_package.value.code == "artifact_dependency_refresh_failed"

    target = tmp_path / "hash-target"
    target.write_bytes(b"stable")
    descriptor = os.open(target, os.O_RDONLY)
    real_pread = rv.os.pread
    try:
        monkeypatch.setattr(rv.os, "pread", lambda *_args: b"")
        with pytest.raises(JointRiggerContractError) as short_read:
            rv._precomposition_descriptor_sha256(
                descriptor,
                size=target.stat().st_size,
                label="short",
            )
        assert short_read.value.code == "dependency_artifact_invalid"

        monkeypatch.setattr(
            rv.os,
            "pread",
            lambda fd, count, offset: (
                b"x"
                if offset == target.stat().st_size
                else real_pread(fd, count, offset)
            ),
        )
        with pytest.raises(JointRiggerContractError) as growth:
            rv._precomposition_descriptor_sha256(
                descriptor,
                size=target.stat().st_size,
                label="growth",
            )
        assert growth.value.code == "dependency_artifact_invalid"
    finally:
        os.close(descriptor)

    projection = _test_projection(tmp_path / "root-mismatch")
    with pytest.raises(JointRiggerContractError) as root_mismatch:
        rv._require_projected_root_matches_hash(
            tmp_path / "missing.usda",
            projection=projection,
            expected_sha256="0" * 64,
            code="artifact_mutated",
        )
    assert root_mismatch.value.code == "artifact_mutated"
    assert (
        rv._precomposition_layer_suffix(
            tmp_path / "dependency.data",
            format_hint="unsupported",
        )
        is None
    )
    assert (
        rv._precomposition_layer_suffix(
            tmp_path / "dependency.data",
            format_hint="usda",
        )
        == ".usda"
    )
    direct_fifo = tmp_path / "direct-dependency.fifo"
    os.mkfifo(direct_fifo)
    with pytest.raises(JointRiggerContractError) as special_file:
        rv._open_precomposition_regular_file(direct_fifo)
    assert special_file.value.code == "dependency_artifact_invalid"


def test_precomposition_file_state_guards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "dependency.usda"
    target.write_bytes(b"stable")
    real_fstat = rv.os.fstat

    class ChangedStat:
        def __init__(self, original: Any, **changes: int) -> None:
            self.original = original
            self.changes = changes

        def __getattr__(self, name: str) -> Any:
            if name in self.changes:
                return self.changes[name]
            return getattr(self.original, name)

    monkeypatch.setattr(
        rv.os,
        "fstat",
        lambda descriptor: ChangedStat(
            real_fstat(descriptor),
            st_size=real_fstat(descriptor).st_size + 1,
        ),
    )
    with pytest.raises(JointRiggerContractError) as opened_race:
        rv._open_precomposition_regular_file(target)
    assert opened_race.value.code == "dependency_artifact_invalid"

    monkeypatch.setattr(rv.os, "fstat", real_fstat)
    descriptor, state, backing_path, _ = rv._open_precomposition_regular_file(target)
    changed_state = (*state[:4], state[4] + 1, *state[5:])
    try:
        with pytest.raises(JointRiggerContractError) as retained_race:
            rv._require_precomposition_file_unchanged(
                backing_path,
                descriptor=descriptor,
                expected_state=changed_state,
            )
    finally:
        os.close(descriptor)
    assert retained_race.value.code == "dependency_artifact_invalid"


def test_precomposition_symlink_cycle_and_race_guards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.symlink_to(second.name)
    second.symlink_to(first.name)
    with pytest.raises(JointRiggerContractError) as cycle:
        rv._resolve_precomposition_symlinks(first)
    assert cycle.value.code == "dependency_artifact_invalid"

    regular = tmp_path / "regular"
    regular.write_bytes(b"stable")
    with monkeypatch.context() as context:
        context.setattr(rv, "_MAX_DEPENDENCY_VALIDATION_PASSES", 1)
        with pytest.raises(JointRiggerContractError) as limit:
            rv._resolve_precomposition_symlinks(regular)
    assert limit.value.code == "dependency_artifact_invalid"

    target = tmp_path / "target"
    alternate = tmp_path / "alternate"
    alias = tmp_path / "alias"
    target.write_bytes(b"target")
    alternate.write_bytes(b"alternate")
    alias.symlink_to(target.name)
    real_stat = rv.os.stat
    alias_stats = 0

    class ChangedStat:
        def __init__(self, original: Any) -> None:
            self.original = original

        def __getattr__(self, name: str) -> Any:
            value = getattr(self.original, name)
            return value + 1 if name == "st_mtime_ns" else value

    def racing_stat(path: Any, *, follow_symlinks: bool = True) -> Any:
        nonlocal alias_stats
        result = real_stat(path, follow_symlinks=follow_symlinks)
        if Path(path) == alias and not follow_symlinks:
            alias_stats += 1
            if alias_stats == 2:
                return ChangedStat(result)
        return result

    with monkeypatch.context() as context:
        context.setattr(rv.os, "stat", racing_stat)
        with pytest.raises(JointRiggerContractError) as changed_during_resolve:
            rv._resolve_precomposition_symlinks(alias)
    assert changed_during_resolve.value.code == "dependency_artifact_invalid"

    _, hops = rv._resolve_precomposition_symlinks(alias)
    alias.unlink()
    alias.symlink_to(alternate.name)
    with pytest.raises(JointRiggerContractError) as changed_after_resolve:
        rv._require_precomposition_symlinks_unchanged(
            hops,
            locator="fixture-locator",
        )
    assert changed_after_resolve.value.code == "dependency_artifact_invalid"


def test_projected_dependency_must_belong_to_retained_projection(
    tmp_path: Path,
) -> None:
    target = tmp_path / "dependency.bin"
    target.write_bytes(b"dependency")
    projection = _test_projection(tmp_path / "projection")

    with pytest.raises(JointRiggerContractError) as caught:
        rv._resolved_usd_dependency(
            "asset",
            str(target),
            artifact_path=tmp_path / "root.usda",
            Ar=Ar,
            read_identifier=str(target),
            projection=projection,
            Sdf=Sdf,
        )

    assert caught.value.code == "dependency_artifact_missing"

    unrelated_read_path = tmp_path / "unrelated.bin"
    unrelated_read_path.write_bytes(b"must not be hashed")
    dependency = rv._ResolvedUsdDependency(
        kind="asset",
        identifier=str(target),
        lexical_path=target,
        local_path=target,
        package_relative=False,
        read_identifier=str(unrelated_read_path),
    )
    assert rv._resolved_dependency_sha256(dependency) == _sha256(target)


@pytest.mark.parametrize(
    ("locator", "expected"),
    [
        ("", False),
        ("relative/asset.usda", False),
        ("s:opaque-asset", True),
        ("custom-resolver:asset", True),
        ("C:/assets/asset.usda", False),
        (r"C:\assets\asset.usda", False),
        ("C:drive-relative.usda", True),
        ("file:///tmp/asset.usda", False),
        ("file:relative.usda", False),
        ("file://[bad/path", True),
    ],
)
def test_remote_resolver_locator_classification(
    locator: str,
    expected: bool,
) -> None:
    assert rv._is_remote_resolver_locator(locator) is expected


def test_resolved_dependency_preserves_opaque_identifiers_and_file_uris(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "root.usda"
    artifact.write_text("#usda 1.0\n", encoding="utf-8")
    dependency = tmp_path / "dependency file.usda"
    dependency.write_text("#usda 1.0\n", encoding="utf-8")

    file_uri = rv._resolved_usd_dependency(
        "layer",
        dependency.as_uri(),
        artifact_path=artifact,
        Ar=Ar,
        Sdf=Sdf,
    )
    assert file_uri.identifier == dependency.as_uri()
    assert file_uri.lexical_path == dependency
    assert file_uri.local_path == dependency

    projection = _test_projection(tmp_path / "projection")
    opaque_identifiers = (
        ("s:opaque-resolved-layer", False),
        (
            Sdf.Layer.CreateIdentifier(
                Ar.JoinPackageRelativePath(
                    "s:opaque-package.usdz",
                    "nested/root.usda",
                ),
                {"format": "usda"},
            ),
            True,
        ),
    )
    for opaque_identifier, package_relative in opaque_identifiers:
        opaque = rv._resolved_usd_dependency(
            "layer",
            opaque_identifier,
            artifact_path=artifact,
            Ar=Ar,
            read_identifier=opaque_identifier,
            projection=projection,
            Sdf=Sdf,
        )
        assert opaque.identifier == opaque_identifier
        assert opaque.read_identifier == opaque_identifier
        assert opaque.lexical_path is None
        assert opaque.local_path is None
        assert opaque.captured_sha256 is None
        assert opaque.package_relative is package_relative

    for invalid_file_uri in (
        "embedded\x00nul.usda",
        "file:relative.usda",
        f"file:{dependency}",
        f"{dependency.as_uri()}/../{dependency.name}",
        f"file://remote-host{dependency}",
        "file:///tmp/embedded%00nul.usda",
        "file://[bad/path",
        "file:////tmp/double-root.usda",
    ):
        with pytest.raises(JointRiggerContractError) as invalid_file:
            rv._resolved_usd_dependency(
                "layer",
                invalid_file_uri,
                artifact_path=artifact,
                Ar=Ar,
                Sdf=Sdf,
            )
        assert invalid_file.value.code == "dependency_artifact_invalid"


def test_retained_identity_recheck_rejects_bundle_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "root.usda"
    _write_sublayer_root(root)
    root_sha256 = _sha256(root)
    expected = ArtifactIdentityV1(
        uri="fixture://bundle-mismatch",
        root_sha256=root_sha256,
        dependency_bundle_sha256="0" * 64,
    )

    with pytest.raises(JointRiggerContractError) as caught:
        rv._require_artifact_identity_unchanged(
            root,
            object(),
            expected,
            dependencies=(),
            projection=_test_projection(tmp_path / "projection"),
            missing_code="artifact_missing",
            root_mutated_code="artifact_mutated",
            dependency_mutated_code="artifact_dependency_mutated",
        )

    assert caught.value.code == "artifact_dependency_mutated"


def test_dependency_paths_optionally_preserve_lexical_symlink_aliases(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root.usda"
    real_dependency = tmp_path / "real-dependency.usda"
    dependency_alias = tmp_path / "dependency-alias.usda"
    _write_value_layer(real_dependency, 1)
    dependency_alias.symlink_to(real_dependency)
    _write_sublayer_root(root, dependency_alias.name)

    assert set(local_usd_dependency_paths(root)) == {
        root.resolve(),
        real_dependency.resolve(),
    }
    assert set(local_usd_dependency_paths(root, include_lexical_aliases=True)) == {
        root.resolve(),
        real_dependency.resolve(),
        dependency_alias.absolute(),
    }


@pytest.mark.parametrize("symlink_kind", ["leaf", "ancestor"])
def test_dependency_paths_preserve_every_symlink_hop(
    tmp_path: Path,
    symlink_kind: Literal["leaf", "ancestor"],
) -> None:
    root = tmp_path / "root.usda"
    real_directory = tmp_path / "real"
    real_directory.mkdir()
    real_dependency = real_directory / "dependency.usda"
    _write_value_layer(real_dependency, 1)
    if symlink_kind == "leaf":
        intermediate_alias = tmp_path / "intermediate-alias.usda"
        authored_alias = tmp_path / "authored-alias.usda"
        intermediate_alias.symlink_to(real_dependency)
        authored_alias.symlink_to(intermediate_alias)
        authored_locator = authored_alias
        expected_aliases = {intermediate_alias, authored_alias}
    else:
        intermediate_directory = tmp_path / "intermediate-directory"
        authored_directory = tmp_path / "authored-directory"
        intermediate_directory.symlink_to(real_directory, target_is_directory=True)
        authored_directory.symlink_to(
            intermediate_directory,
            target_is_directory=True,
        )
        authored_locator = authored_directory / real_dependency.name
        expected_aliases = {
            intermediate_directory / real_dependency.name,
            authored_locator,
        }
    _write_sublayer_root(root, authored_locator.relative_to(tmp_path).as_posix())

    assert set(
        local_usd_dependency_paths(root, include_lexical_aliases=True)
    ) == expected_aliases | {
        root.resolve(),
        real_dependency,
    }


def test_identify_usd_artifact_is_stable_after_tree_relocation(tmp_path: Path) -> None:
    original = tmp_path / "original"
    source, _ = _write_pair(original)
    _add_inert_sublayer(source, "dependency.usda")
    texture = original / "texture.bin"
    texture.write_bytes(b"texture")
    stage = Usd.Stage.Open(str(source))
    stage.GetDefaultPrim().CreateAttribute(
        "test:asset",
        Sdf.ValueTypeNames.Asset,
    ).Set(Sdf.AssetPath(texture.name))
    assert stage.GetRootLayer().Save()
    relocated = tmp_path / "relocated"
    shutil.copytree(original, relocated)

    before = identify_usd_artifact(source, uri=SOURCE_URI)
    after = identify_usd_artifact(relocated / source.name, uri=SOURCE_URI)

    assert after == before


def test_local_dependency_paths_and_identity_include_external_assets(
    tmp_path: Path,
) -> None:
    source, reference = _write_pair(tmp_path)
    dependency = _add_inert_sublayer(source, "dependency.usda")
    texture = tmp_path / "texture.bin"
    texture.write_bytes(b"texture-a")
    source_stage = Usd.Stage.Open(str(source))
    source_stage.GetDefaultPrim().CreateAttribute(
        "test:asset",
        Sdf.ValueTypeNames.Asset,
    ).Set(Sdf.AssetPath(texture.name))
    source_stage.GetDefaultPrim().CreateAttribute(
        "test:duplicateAsset",
        Sdf.ValueTypeNames.Asset,
    ).Set(Sdf.AssetPath(texture.name))
    assert source_stage.GetRootLayer().Save()

    paths = local_usd_dependency_paths(source)
    assert paths == tuple(
        sorted(
            {source.resolve(), dependency.resolve(), texture.resolve()},
            key=lambda item: item.as_posix(),
        )
    )
    assert local_usd_dependency_paths(source) == paths

    before = identify_usd_artifact(source, uri=SOURCE_URI)
    texture.write_bytes(b"texture-b")
    after = identify_usd_artifact(source, uri=SOURCE_URI)
    assert after.root_sha256 == before.root_sha256
    assert after.dependency_bundle_sha256 != before.dependency_bundle_sha256

    reference_stage = Usd.Stage.Open(str(reference))
    reference_stage.GetDefaultPrim().CreateAttribute(
        "test:asset",
        Sdf.ValueTypeNames.Asset,
    ).Set(Sdf.AssetPath(texture.name))
    assert reference_stage.GetRootLayer().Save()
    extracted = _extract(source, reference)
    assert extracted.source_asset == identify_usd_artifact(source, uri=SOURCE_URI)


def test_dependency_enumeration_fails_closed_on_unresolved_or_invalid_usd(
    tmp_path: Path,
) -> None:
    broken = tmp_path / "broken.usda"
    broken.write_text(
        "#usda 1.0\n( subLayers = [@missing.usda@] )\n",
        encoding="utf-8",
    )
    for operation in (
        lambda: local_usd_dependency_paths(broken),
        lambda: identify_usd_artifact(broken, uri="fixture://broken"),
    ):
        with pytest.raises(JointRiggerContractError) as caught:
            operation()
        assert caught.value.code == "unresolved_artifact_dependency"
        assert str(tmp_path / "missing.usda") in caught.value.detail
        assert "joint-rigger-composition-" not in caught.value.detail
        assert caught.value.unresolved_dependency_paths == (
            str(tmp_path / "missing.usda"),
        )

    invalid = tmp_path / "invalid.usda"
    invalid.write_text("not usd", encoding="utf-8")
    with pytest.raises(JointRiggerContractError) as caught:
        local_usd_dependency_paths(invalid)
    assert caught.value.code == "artifact_dependency_enumeration_failed"


def test_package_dependencies_map_to_outer_file_and_bind_package_entries(
    tmp_path: Path,
) -> None:
    package_source = tmp_path / "package-source"
    package_source.mkdir()
    root = package_source / "root.usda"
    dependency = package_source / "dependency.usda"
    texture = package_source / "texture.bin"
    dependency.write_text('#usda 1.0\ndef Xform "Dependency" {}\n', encoding="utf-8")
    texture.write_bytes(b"texture")
    root.write_text(
        "#usda 1.0\n"
        "( subLayers = [@dependency.usda@] )\n"
        'def Xform "Root" {\n'
        "    custom asset test:asset = @texture.bin@\n"
        "}\n",
        encoding="utf-8",
    )
    package = tmp_path / "asset.usdz"
    assert UsdUtils.CreateNewUsdzPackage(str(root), str(package))

    assert local_usd_dependency_paths(package) == (package.resolve(),)
    identity = identify_usd_artifact(package, uri="fixture://package")
    assert identity.root_sha256 == _sha256(package)
    assert identity.dependency_bundle_sha256 is not None


def test_dependency_inventory_rejects_stale_package_without_mutation(
    tmp_path: Path,
) -> None:
    def create_package(directory: Path, dependency_name: str) -> Path:
        directory.mkdir()
        dependency = directory / dependency_name
        dependency.write_text(
            f'#usda 1.0\ndef Xform "{dependency.stem}" {{}}\n',
            encoding="utf-8",
        )
        root = directory / "root.usda"
        _write_sublayer_root(root, dependency.name)
        package = directory / "asset.usdz"
        assert UsdUtils.CreateNewUsdzPackage(str(root), str(package))
        return package

    old_package = create_package(tmp_path / "old", "old.usda")
    new_package = create_package(tmp_path / "new", "new.usda")
    live_directory = tmp_path / "live"
    live_directory.mkdir()
    live_package = live_directory / "asset.usdz"
    shutil.copy2(old_package, live_package)
    cached_stage = Usd.Stage.Open(str(live_package))
    identify_usd_artifact(live_package, uri="fixture://package-refresh")

    shutil.copy2(new_package, live_package)

    for operation in (
        lambda: local_usd_dependency_paths(live_package),
        lambda: identify_usd_artifact(
            live_package,
            uri="fixture://package-refresh",
        ),
    ):
        with pytest.raises(JointRiggerContractError) as caught:
            operation()
        assert caught.value.code == "artifact_dependency_cache_stale"
    assert cached_stage.GetPrimAtPath("/old").IsValid()
    assert not cached_stage.GetPrimAtPath("/new").IsValid()
    assert not cached_stage.GetRootLayer().dirty


def test_identify_usd_artifact_rejects_dependency_mutation_during_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, _ = _write_pair(tmp_path)
    dependency = _add_inert_sublayer(source, "dependency.usda")
    real_identity = rv._artifact_identity
    calls = 0

    def mutate_after_first_identity(*args: Any, **kwargs: Any) -> ArtifactIdentityV1:
        nonlocal calls
        identity = real_identity(*args, **kwargs)
        calls += 1
        if calls == 1:
            dependency.write_text(
                dependency.read_text(encoding="utf-8") + "\n# changed mid-read\n",
                encoding="utf-8",
            )
        return identity

    monkeypatch.setattr(rv, "_artifact_identity", mutate_after_first_identity)
    with pytest.raises(JointRiggerContractError) as caught:
        identify_usd_artifact(source, uri=SOURCE_URI)

    assert caught.value.code == "artifact_dependency_mutated"


def test_reference_selection_requires_explicit_omission_policy(tmp_path: Path) -> None:
    source, reference = _write_pair(tmp_path, include_fixed=True)
    moving_paths = (
        "/World/Joints/drawer",
        "/World/Joints/door",
        "/World/Joints/spherical",
    )

    with pytest.raises(JointRiggerContractError) as caught:
        extract_reference_input(
            source,
            reference,
            source_uri=SOURCE_URI,
            reference_uri=REFERENCE_URI,
            joint_paths=moving_paths,
        )
    assert caught.value.code == "unapproved_omitted_joint"

    result = extract_reference_input(
        source,
        reference,
        source_uri=SOURCE_URI,
        reference_uri=REFERENCE_URI,
        joint_paths=moving_paths,
        allowed_omitted_joint_types=("fixed",),
    )
    assert len(result.plan.joints) == 3


def test_source_joint_absent_from_reference_fails_closed(tmp_path: Path) -> None:
    source, reference = _write_pair(tmp_path)
    source_stage = Usd.Stage.Open(str(source))
    UsdGeom.Scope.Define(source_stage, "/World/Joints")
    legacy = UsdPhysics.FixedJoint.Define(
        source_stage,
        "/World/Joints/legacy",
    )
    legacy.CreateBody0Rel().SetTargets(["/World/base"])
    legacy.CreateBody1Rel().SetTargets(["/World/door"])
    assert source_stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "source_joint_not_in_reference"
    assert "/World/Joints/legacy" in caught.value.detail
    assert "/World/base" in caught.value.detail
    assert "/World/door" in caught.value.detail


def test_matching_source_joint_can_be_explicitly_omitted(tmp_path: Path) -> None:
    source, reference = _write_pair(tmp_path, include_fixed=True)
    source_stage = Usd.Stage.Open(str(source))
    UsdGeom.Scope.Define(source_stage, "/World/Joints")
    fixed = UsdPhysics.FixedJoint.Define(source_stage, "/World/Joints/fixed")
    fixed.CreateBody0Rel().SetTargets(["/World/base"])
    fixed.CreateBody1Rel().SetTargets(["/World/door"])
    assert source_stage.GetRootLayer().Save()

    result = _extract(
        source,
        reference,
        joint_paths=(
            "/World/Joints/drawer",
            "/World/Joints/door",
            "/World/Joints/spherical",
        ),
        allowed_omitted_joint_types=("fixed",),
    )

    assert len(result.plan.joints) == 3


def test_different_source_joint_cannot_be_explicitly_omitted(tmp_path: Path) -> None:
    source, reference = _write_pair(tmp_path, include_fixed=True)
    source_stage = Usd.Stage.Open(str(source))
    UsdGeom.Scope.Define(source_stage, "/World/Joints")
    fixed = UsdPhysics.FixedJoint.Define(source_stage, "/World/Joints/fixed")
    fixed.CreateBody0Rel().SetTargets(["/World/base"])
    fixed.CreateBody1Rel().SetTargets(["/World/drawer"])
    assert source_stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(
            source,
            reference,
            joint_paths=(
                "/World/Joints/drawer",
                "/World/Joints/door",
                "/World/Joints/spherical",
            ),
            allowed_omitted_joint_types=("fixed",),
        )

    assert caught.value.code == "source_joint_differs_from_reference"
    assert "/World/Joints/fixed" in caught.value.detail
    assert "physics:body1" in caught.value.detail


@pytest.mark.parametrize("composed_state", ["inactive", "undefined"])
def test_discovered_joints_must_be_active_and_defined(
    tmp_path: Path,
    composed_state: str,
) -> None:
    source, reference = _write_pair(tmp_path)
    stage = Usd.Stage.Open(str(reference))
    door = stage.GetPrimAtPath("/World/Joints/door")
    if composed_state == "inactive":
        door.SetActive(False)
    else:
        door.SetSpecifier(Sdf.SpecifierOver)
    assert stage.GetRootLayer().Save()

    result = _extract(source, reference)

    assert [joint.topology.joint_id for joint in result.plan.joints] == [
        "/World/Joints/drawer",
        "/World/Joints/spherical",
    ]


@pytest.mark.parametrize(
    ("composed_state", "reason_code"),
    [
        ("inactive", "selected_joint_inactive"),
        ("undefined", "selected_joint_undefined"),
    ],
)
def test_explicit_inactive_or_undefined_joint_selection_fails_closed(
    tmp_path: Path,
    composed_state: str,
    reason_code: str,
) -> None:
    source, reference = _write_pair(tmp_path)
    stage = Usd.Stage.Open(str(reference))
    door = stage.GetPrimAtPath("/World/Joints/door")
    if composed_state == "inactive":
        door.SetActive(False)
    else:
        door.SetSpecifier(Sdf.SpecifierOver)
    assert stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(
            source,
            reference,
            joint_paths=("/World/Joints/door",),
            allowed_omitted_joint_types=("prismatic", "spherical"),
        )

    assert caught.value.code == reason_code
    assert "/World/Joints/door" in caught.value.detail


@pytest.mark.parametrize("source_kind", ["preexisting_joint", "rigged_as_source"])
def test_selected_joint_paths_must_be_absent_from_paired_source(
    tmp_path: Path,
    source_kind: str,
) -> None:
    source, reference = _write_pair(tmp_path)
    if source_kind == "rigged_as_source":
        source = reference
        kwargs: dict[str, Any] = {}
    else:
        source_stage = Usd.Stage.Open(str(source))
        UsdGeom.Scope.Define(source_stage, "/World/Joints")
        _define_joint(source_stage, "door")
        assert source_stage.GetRootLayer().Save()
        kwargs = {
            "joint_paths": ("/World/Joints/door",),
            "allowed_omitted_joint_types": ("prismatic", "spherical"),
        }

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference, **kwargs)

    assert caught.value.code == "selected_joint_present_in_source"
    assert "/World/Joints/door" in caught.value.detail


def test_unrigged_source_has_no_selected_joint_path_evidence(tmp_path: Path) -> None:
    source, reference = _write_pair(tmp_path)

    result = _extract(source, reference)

    assert len(result.plan.joints) == 3


def test_joint_inside_instance_proxy_fails_instead_of_being_omitted(
    tmp_path: Path,
) -> None:
    source, reference = _write_pair(tmp_path)
    source_asset = tmp_path / "source_instance.usda"
    reference_asset = tmp_path / "reference_instance.usda"
    _write_instance_asset(source_asset)
    _write_instance_asset(reference_asset, include_joint=True)
    _add_instance_reference(source, source_asset, "/World/Instance")
    _add_instance_reference(reference, reference_asset, "/World/Instance")

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "unsupported_instance_proxy_physics"
    assert "/World/Instance/Joints/hinge" in caught.value.detail


@pytest.mark.parametrize(
    "proxy_artifacts",
    [
        ("source",),
        ("reference",),
        ("source", "reference"),
    ],
    ids=("source-only", "reference-only", "both"),
)
def test_selected_endpoint_instance_proxy_fails_regardless_of_pairing(
    tmp_path: Path,
    proxy_artifacts: tuple[str, ...],
) -> None:
    source, reference = _write_pair(tmp_path)
    for label, stage_path in (("source", source), ("reference", reference)):
        if label in proxy_artifacts:
            asset_path = tmp_path / f"{label}_endpoint_instance.usda"
            _write_instance_asset(asset_path)
            _add_instance_reference(stage_path, asset_path, "/World/Instance")
        else:
            stage = Usd.Stage.Open(str(stage_path))
            UsdGeom.Xform.Define(stage, "/World/Instance")
            UsdGeom.Cube.Define(stage, "/World/Instance/body0")
            assert stage.GetRootLayer().Save()

    reference_stage = Usd.Stage.Open(str(reference))
    joint = UsdPhysics.RevoluteJoint.Define(
        reference_stage,
        "/World/Joints/instanceEndpoint",
    )
    joint.CreateBody0Rel().SetTargets(["/World/base"])
    joint.CreateBody1Rel().SetTargets(["/World/Instance/body0"])
    joint.CreateAxisAttr("Z")
    assert reference_stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(
            source,
            reference,
            joint_paths=("/World/Joints/instanceEndpoint",),
            allowed_omitted_joint_types=("revolute", "prismatic", "spherical"),
        )

    assert caught.value.code == "unsupported_instance_proxy_physics"
    assert "/World/Instance/body0" in caught.value.detail


def test_collider_inside_instance_proxy_fails_instead_of_being_omitted(
    tmp_path: Path,
) -> None:
    source, reference = _write_pair(tmp_path)
    source_asset = tmp_path / "source_collider_instance.usda"
    reference_asset = tmp_path / "reference_collider_instance.usda"
    _write_instance_asset(source_asset)
    _write_instance_asset(reference_asset, include_collider=True)
    _add_instance_reference(source, source_asset, "/World/base/Instance")
    _add_instance_reference(reference, reference_asset, "/World/base/Instance")

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "unsupported_instance_proxy_physics"
    assert "/World/base/Instance/collider" in caught.value.detail


def test_harmless_instance_proxy_geometry_is_accepted(tmp_path: Path) -> None:
    source, reference = _write_pair(tmp_path)
    asset = tmp_path / "visual_instance.usda"
    _write_instance_asset(asset)
    _add_instance_reference(source, asset, "/World/VisualInstance")
    _add_instance_reference(reference, asset, "/World/VisualInstance")

    result = _extract(source, reference)

    assert len(result.plan.joints) == 3
    assert all(
        not body.prim_path.startswith("/World/VisualInstance")
        for body in result.plan.rigid_bodies
    )


@pytest.mark.parametrize("proxy_stage", ["source", "reference"])
def test_articulation_root_instance_proxy_fails_closed(
    tmp_path: Path,
    proxy_stage: str,
) -> None:
    source, reference = _write_pair(tmp_path)
    root_path = "/World/Instance/articulation"
    body_path = f"{root_path}/body"
    for label, stage_path in (("source", source), ("reference", reference)):
        if label == proxy_stage:
            asset = tmp_path / f"{label}_articulation_instance.usda"
            _write_instance_asset(asset, include_articulation_root=True)
            _add_instance_reference(stage_path, asset, "/World/Instance")
        else:
            stage = Usd.Stage.Open(str(stage_path))
            root = UsdGeom.Xform.Define(stage, root_path).GetPrim()
            UsdGeom.Cube.Define(stage, body_path)
            if label == "reference":
                UsdPhysics.ArticulationRootAPI.Apply(root)
            assert stage.GetRootLayer().Save()

    reference_stage = Usd.Stage.Open(str(reference))
    source_stage = Usd.Stage.Open(str(source))
    with pytest.raises(JointRiggerContractError) as caught:
        rv._extract_articulation_root(
            reference_stage,
            source_stage=source_stage,
            body_paths={body_path},
            joint_paths=set(),
            reference_identity=_identity(reference),
            UsdPhysics=UsdPhysics,
        )

    assert caught.value.code == "unsupported_instance_proxy_physics"
    assert root_path in caught.value.detail


@pytest.mark.parametrize(
    ("mutation", "reason_code"),
    [
        ("missing_body0", "missing_body0"),
        ("multiple_body0", "multiple_body0_targets"),
        ("same_endpoints", "same_body_endpoints"),
        ("missing_source_endpoint", "endpoint_not_in_source"),
        ("missing_axis", "axis_unresolved"),
        ("contradictory_axis_frames", "contradictory_joint_frames"),
        ("partial_anchor", "incomplete_optional_schema"),
        ("invalid_limit_range", "invalid_limit_range"),
    ],
)
def test_reference_extraction_fails_closed_with_stable_reason_codes(
    tmp_path: Path,
    mutation: str,
    reason_code: str,
) -> None:
    source, reference = _write_pair(tmp_path)
    _mutate_pair(source, reference, mutation)

    with pytest.raises(JointRiggerContractError) as caught:
        extract_reference_input(
            source,
            reference,
            source_uri=SOURCE_URI,
            reference_uri=REFERENCE_URI,
        )

    assert caught.value.code == reason_code
    assert reason_code in str(caught.value)


def test_selected_joint_must_exist_and_unsupported_type_is_not_silent(
    tmp_path: Path,
) -> None:
    source, reference = _write_pair(tmp_path, include_fixed=True)
    with pytest.raises(JointRiggerContractError) as missing:
        extract_reference_input(
            source,
            reference,
            source_uri=SOURCE_URI,
            reference_uri=REFERENCE_URI,
            joint_paths=("/World/Joints/not_present",),
        )
    assert missing.value.code == "selected_joint_missing"

    with pytest.raises(JointRiggerContractError) as unsupported:
        extract_reference_input(
            source,
            reference,
            source_uri=SOURCE_URI,
            reference_uri=REFERENCE_URI,
        )
    assert unsupported.value.code == "unsupported_joint_type"


def test_empty_selection_blank_allowance_duplicate_paths_and_blank_uri_fail(
    tmp_path: Path,
) -> None:
    source, reference = _write_pair(tmp_path)
    reference_stage = Usd.Stage.Open(str(reference))
    reference_stage.RemovePrim("/World/Joints")
    reference_stage.GetRootLayer().Save()
    with pytest.raises(JointRiggerContractError) as empty:
        _extract(source, reference)
    assert empty.value.code == "no_supported_joints"

    source, reference = _write_pair(tmp_path / "selection")
    with pytest.raises(JointRiggerContractError) as blank_allowance:
        _extract(source, reference, allowed_omitted_joint_types=("",))
    assert blank_allowance.value.code == "invalid_joint_selection"
    with pytest.raises(JointRiggerContractError) as duplicate:
        _extract(
            source,
            reference,
            joint_paths=("/World/Joints/door", "/World/Joints/door"),
        )
    assert duplicate.value.code == "duplicate_joint_selection"
    with pytest.raises(JointRiggerContractError) as blank_uri:
        extract_reference_input(
            source,
            reference,
            source_uri=" ",
            reference_uri=REFERENCE_URI,
        )
    assert blank_uri.value.code == "invalid_artifact_identity"


def test_missing_and_unopenable_stage_errors_are_stable(tmp_path: Path) -> None:
    source, reference = _write_pair(tmp_path / "valid")
    with pytest.raises(JointRiggerContractError) as missing_source:
        _extract(tmp_path / "missing-source.usda", reference)
    assert missing_source.value.code == "source_artifact_missing"
    with pytest.raises(JointRiggerContractError) as missing_reference:
        _extract(source, tmp_path / "missing-reference.usda")
    assert missing_reference.value.code == "reference_artifact_missing"

    bad_source = tmp_path / "bad-source.usda"
    bad_source.write_text("not usd", encoding="utf-8")
    with pytest.raises(JointRiggerContractError) as source_open:
        _extract(bad_source, reference)
    assert source_open.value.code == "source_stage_open_failed"
    bad_reference = tmp_path / "bad-reference.usda"
    bad_reference.write_text("not usd", encoding="utf-8")
    with pytest.raises(JointRiggerContractError) as reference_open:
        _extract(source, bad_reference)
    assert reference_open.value.code == "reference_stage_open_failed"


@pytest.mark.parametrize(
    ("target_name", "reason_code"),
    [
        ("source.usda", "source_artifact_mutated"),
        ("reference.usda", "reference_artifact_mutated"),
    ],
)
def test_extraction_detects_mid_read_artifact_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_name: str,
    reason_code: str,
) -> None:
    source, reference = _write_pair(tmp_path)
    real_hash = rv._file_sha256
    counts = 0

    def changing_hash(path: Path, *, code: str) -> str:
        nonlocal counts
        value = real_hash(path, code=code)
        if Path(path).name == target_name:
            counts += 1
            if counts >= 2:
                return "0" * 64 if value != "0" * 64 else "1" * 64
        return value

    monkeypatch.setattr(rv, "_file_sha256", changing_hash)
    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)
    assert caught.value.code == reason_code


@pytest.mark.parametrize(
    ("target_name", "reason_code"),
    [
        ("source", "source_dependency_artifact_mutated"),
        ("reference", "reference_dependency_artifact_mutated"),
    ],
)
def test_extraction_detects_mid_read_dependency_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_name: str,
    reason_code: str,
) -> None:
    source, reference = _write_pair(tmp_path)
    source_dependency = _add_inert_sublayer(source, "source-dependency.usda")
    reference_dependency = _add_inert_sublayer(
        reference,
        "reference-dependency.usda",
    )
    target = source_dependency if target_name == "source" else reference_dependency
    real_extract = rv._extract_joint_plan
    mutated = False

    def mutate_during_extraction(*args: Any, **kwargs: Any) -> Any:
        nonlocal mutated
        if not mutated:
            target.write_text(
                target.read_text(encoding="utf-8") + "\n# changed mid-read\n",
                encoding="utf-8",
            )
            mutated = True
        return real_extract(*args, **kwargs)

    monkeypatch.setattr(rv, "_extract_joint_plan", mutate_during_extraction)
    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == reason_code


def test_private_joint_type_and_unsupported_plan_guards(tmp_path: Path) -> None:
    source, reference = _write_pair(tmp_path, include_fixed=True)
    source_stage = Usd.Stage.Open(str(source))
    reference_stage = Usd.Stage.Open(str(reference))
    fixed = reference_stage.GetPrimAtPath("/World/Joints/fixed")
    identity = _identity(reference)
    with pytest.raises(JointRiggerContractError) as caught:
        rv._extract_joint_plan(
            fixed,
            source_stage=source_stage,
            reference_stage=reference_stage,
            reference_identity=identity,
            source_xform_cache=UsdGeom.XformCache(Usd.TimeCode.Default()),
            xform_cache=UsdGeom.XformCache(Usd.TimeCode.Default()),
            UsdPhysics=UsdPhysics,
            UsdGeom=UsdGeom,
        )
    assert caught.value.code == "unsupported_joint_type"

    class UnknownSchemas:
        pass

    class TypePrim:
        def __init__(self, name: str) -> None:
            self.name = name

        def GetTypeName(self) -> str:
            return self.name

    assert rv._joint_type(TypePrim("PhysicsCustomJoint"), UnknownSchemas) == "custom"
    assert rv._joint_type(TypePrim("PhysicsJoint"), UnknownSchemas) == "unknown"


@pytest.mark.parametrize(
    ("mutation", "reason_code"),
    [
        ("root_endpoint", "invalid_body0_path"),
        ("unsupported_axis", "axis_unresolved"),
        ("nonfinite_limit", "invalid_limit_value"),
        ("blocked_local_rotation", "axis_unresolved"),
        ("blocked_anchor", "incomplete_optional_schema"),
        ("contradictory_anchor", "contradictory_joint_frames"),
    ],
)
def test_reference_property_guardrails(
    tmp_path: Path,
    mutation: str,
    reason_code: str,
) -> None:
    source, reference = _write_pair(tmp_path)
    stage = Usd.Stage.Open(str(reference))
    drawer = UsdPhysics.PrismaticJoint(stage.GetPrimAtPath("/World/Joints/drawer"))
    if mutation == "root_endpoint":
        drawer.GetBody0Rel().SetTargets([Sdf.Path("/World/base.invalidProperty")])
    elif mutation == "unsupported_axis":
        drawer.GetAxisAttr().Set("Q")
    elif mutation == "nonfinite_limit":
        drawer.GetUpperLimitAttr().Set(math.nan)
    elif mutation == "blocked_local_rotation":
        drawer.GetLocalRot0Attr().Block()
    elif mutation == "blocked_anchor":
        drawer.GetLocalPos0Attr().Block()
    elif mutation == "contradictory_anchor":
        drawer.GetLocalPos1Attr().Set(Gf.Vec3f(20.0, 0.0, 0.0))
    stage.GetRootLayer().Save()
    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)
    assert caught.value.code == reason_code


@pytest.mark.parametrize(
    ("fields", "value", "reported_field"),
    [
        (("localPos0",), math.nan, "localPos0"),
        (("localPos1",), math.inf, "localPos1"),
        (("localPos0", "localPos1"), -math.inf, "localPos0"),
    ],
)
def test_nonfinite_authored_anchor_values_have_stable_contract_errors(
    tmp_path: Path,
    fields: tuple[str, ...],
    value: float,
    reported_field: str,
) -> None:
    source, reference = _write_pair(tmp_path)
    stage = Usd.Stage.Open(str(reference))
    drawer = UsdPhysics.PrismaticJoint(stage.GetPrimAtPath("/World/Joints/drawer"))
    attributes = {
        "localPos0": drawer.GetLocalPos0Attr(),
        "localPos1": drawer.GetLocalPos1Attr(),
    }
    for field in fields:
        attributes[field].Set(Gf.Vec3f(value, 0.0, 0.0))
    assert stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "invalid_anchor_value"
    assert reported_field in caught.value.detail


def test_finite_anchor_that_transforms_nonfinite_has_stable_contract_error(
    tmp_path: Path,
) -> None:
    source, reference = _write_pair(tmp_path)
    for path in (source, reference):
        stage = Usd.Stage.Open(str(path))
        for body_path in ("/World/base", "/World/ball"):
            UsdGeom.Xformable(stage.GetPrimAtPath(body_path)).AddScaleOp(
                precision=UsdGeom.XformOp.PrecisionDouble
            ).Set(Gf.Vec3d(1e308, 1.0, 1.0))
        if path == reference:
            joint = UsdPhysics.SphericalJoint(
                stage.GetPrimAtPath("/World/Joints/spherical")
            )
            joint.CreateLocalPos0Attr(Gf.Vec3f(2.0, 0.0, 0.0))
            joint.CreateLocalPos1Attr(Gf.Vec3f(2.0, 0.0, 0.0))
        assert stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(
            source,
            reference,
            joint_paths=("/World/Joints/spherical",),
            allowed_omitted_joint_types=("prismatic", "revolute"),
        )

    assert caught.value.code == "invalid_anchor_value"
    assert "transforms to a non-finite stage position" in caught.value.detail


@pytest.mark.parametrize("axis", ["X", "Y", "Z"])
def test_authored_spherical_axis_fails_closed(
    tmp_path: Path,
    axis: str,
) -> None:
    source, reference = _write_pair(tmp_path)
    stage = Usd.Stage.Open(str(reference))
    joint = UsdPhysics.SphericalJoint(stage.GetPrimAtPath("/World/Joints/spherical"))
    attribute = joint.CreateAxisAttr(axis)
    assert attribute.HasAuthoredValueOpinion()
    assert stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "unsupported_spherical_axis"
    assert "/World/Joints/spherical" in caught.value.detail
    assert "physics:axis" in caught.value.detail


def test_time_sampled_spherical_axis_preserves_static_contract_error(
    tmp_path: Path,
) -> None:
    source, reference = _write_pair(tmp_path)
    stage = Usd.Stage.Open(str(reference))
    joint = UsdPhysics.SphericalJoint(stage.GetPrimAtPath("/World/Joints/spherical"))
    attribute = joint.CreateAxisAttr("X")
    attribute.Set("Y", Usd.TimeCode(1.0))
    assert attribute.GetTimeSamples() == [1.0]
    assert stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "time_sampled_static_property"
    assert "physics:axis" in caught.value.detail


@pytest.mark.parametrize("index", [0, 1])
def test_authored_spherical_frame_rotation_fails_closed(
    tmp_path: Path,
    index: int,
) -> None:
    source, reference = _write_pair(tmp_path)
    stage = Usd.Stage.Open(str(reference))
    joint = UsdPhysics.SphericalJoint(stage.GetPrimAtPath("/World/Joints/spherical"))
    create_rotation = (
        joint.CreateLocalRot0Attr if index == 0 else joint.CreateLocalRot1Attr
    )
    attribute = create_rotation(Gf.Quatf(1.0))
    assert attribute.HasAuthoredValueOpinion()
    assert stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "unsupported_spherical_orientation"
    assert f"physics:localRot{index}" in caught.value.detail


@pytest.mark.parametrize("index", [0, 1])
def test_connected_spherical_frame_rotation_fails_closed(
    tmp_path: Path,
    index: int,
) -> None:
    source, reference = _write_pair(tmp_path)
    stage = Usd.Stage.Open(str(reference))
    driver = stage.GetPrimAtPath("/World/ball").CreateAttribute(
        "driverRotation",
        Sdf.ValueTypeNames.Quatf,
    )
    assert driver.Set(Gf.Quatf(1.0))
    joint = UsdPhysics.SphericalJoint(stage.GetPrimAtPath("/World/Joints/spherical"))
    rotation = joint.GetLocalRot0Attr() if index == 0 else joint.GetLocalRot1Attr()
    assert rotation.AddConnection(driver.GetPath())
    assert not rotation.HasAuthoredValueOpinion()
    assert rotation.HasAuthoredConnections()
    assert stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "unsupported_spherical_orientation"
    assert f"physics:localRot{index}" in caught.value.detail


def test_unauthored_spherical_frame_rotations_are_accepted(tmp_path: Path) -> None:
    source, reference = _write_pair(tmp_path)
    stage = Usd.Stage.Open(str(reference))
    joint = UsdPhysics.SphericalJoint(stage.GetPrimAtPath("/World/Joints/spherical"))
    for rotation in (joint.GetLocalRot0Attr(), joint.GetLocalRot1Attr()):
        assert not rotation.HasAuthoredValueOpinion()
        assert not rotation.HasAuthoredConnections()

    result = _extract(source, reference)

    assert _joint(result, "/World/Joints/spherical").topology.joint_type == "spherical"


@pytest.mark.parametrize(
    ("field_case", "expected_code", "expected_detail"),
    [
        ("topology", "unsupported_attribute_connection", "physics:axis"),
        ("anchor0", "unsupported_attribute_connection", "physics:localPos0"),
        ("anchor1", "unsupported_attribute_connection", "physics:localPos1"),
        ("limit", "unsupported_attribute_connection", "physics:lowerLimit"),
        (
            "drive",
            "unsupported_attribute_connection",
            "drive:angular:physics:stiffness",
        ),
        ("state", "unsupported_optional_schema", "PhysicsJointStateAPI:angular"),
        ("rigid_body", "unsupported_attribute_connection", "rigidBodyEnabled"),
        ("mass", "unsupported_attribute_connection", "physics:mass"),
        ("collider", "unsupported_attribute_connection", "physics:approximation"),
    ],
)
def test_connected_value_only_physics_attributes_fail_closed(
    tmp_path: Path,
    field_case: str,
    expected_code: str,
    expected_detail: str,
) -> None:
    source, reference = _write_pair(tmp_path)
    stage = Usd.Stage.Open(str(reference))
    door_prim = stage.GetPrimAtPath("/World/Joints/door")
    door = UsdPhysics.RevoluteJoint(door_prim)
    base = stage.GetPrimAtPath("/World/base")
    if field_case == "topology":
        attribute = door.GetAxisAttr()
    elif field_case == "anchor0":
        attribute = door.GetLocalPos0Attr()
    elif field_case == "anchor1":
        attribute = door.GetLocalPos1Attr()
    elif field_case == "limit":
        attribute = door.GetLowerLimitAttr()
    elif field_case == "drive":
        attribute = UsdPhysics.DriveAPI.Apply(
            door_prim,
            "angular",
        ).GetStiffnessAttr()
    elif field_case == "state":
        assert door_prim.AddAppliedSchema("PhysicsJointStateAPI:angular")
        attribute = door_prim.CreateAttribute(
            "state:angular:physics:position",
            Sdf.ValueTypeNames.Float,
            custom=False,
        )
    elif field_case == "rigid_body":
        attribute = UsdPhysics.RigidBodyAPI(base).GetRigidBodyEnabledAttr()
    elif field_case == "mass":
        attribute = UsdPhysics.MassAPI(base).GetMassAttr()
    else:
        attribute = UsdPhysics.MeshCollisionAPI.Apply(
            base,
        ).GetApproximationAttr()
    _connect_value_only_attribute(stage, attribute)
    assert stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == expected_code
    assert expected_detail in caught.value.detail


@pytest.mark.parametrize(
    "property_name",
    ["physxJoint:maxJointVelocity", "physxJoint:jointFriction"],
)
def test_connected_physx_joint_opinion_fails_closed(
    tmp_path: Path,
    property_name: str,
) -> None:
    source, reference = _write_pair(tmp_path)
    stage = Usd.Stage.Open(str(reference))
    prim = stage.GetPrimAtPath("/World/Joints/drawer")
    assert prim.AddAppliedSchema("PhysxJointAPI")
    attribute = prim.CreateAttribute(
        property_name,
        Sdf.ValueTypeNames.Float,
        custom=False,
    )
    assert attribute.Set(3.5)
    _connect_value_only_attribute(stage, attribute)
    assert stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "unsupported_attribute_connection"
    assert property_name in caught.value.detail


@pytest.mark.parametrize(
    "property_name",
    [
        "physics:lowerLimit",
        "physics:upperLimit",
        "physics:coneAngle0Limit",
        "physics:coneAngle1Limit",
    ],
)
def test_all_authored_spherical_limit_opinions_fail_closed(
    tmp_path: Path,
    property_name: str,
) -> None:
    source, reference = _write_pair(tmp_path)
    stage = Usd.Stage.Open(str(reference))
    stage.GetPrimAtPath("/World/Joints/spherical").CreateAttribute(
        property_name,
        Sdf.ValueTypeNames.Float,
        custom=False,
    ).Set(45.0)
    assert stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "unsupported_spherical_limit"
    assert property_name in caught.value.detail


@pytest.mark.parametrize(
    "property_name",
    ["physics:coneAngle0Limit", "physics:coneAngle1Limit"],
)
def test_spherical_limit_opinions_fail_on_non_spherical_joints(
    tmp_path: Path,
    property_name: str,
) -> None:
    source, reference = _write_pair(tmp_path)
    stage = Usd.Stage.Open(str(reference))
    stage.GetPrimAtPath("/World/Joints/door").CreateAttribute(
        property_name,
        Sdf.ValueTypeNames.Float,
        custom=False,
    ).Set(45.0)
    assert stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "unsupported_spherical_limit"
    assert property_name in caught.value.detail


@pytest.mark.parametrize(
    ("property_name", "value"),
    [
        ("physics:jointEnabled", True),
        ("physics:collisionEnabled", False),
        ("physics:excludeFromArticulation", False),
    ],
)
def test_explicit_base_joint_fallback_values_are_accepted(
    tmp_path: Path,
    property_name: str,
    value: bool,
) -> None:
    source, reference = _write_pair(tmp_path)
    stage = Usd.Stage.Open(str(reference))
    prim = stage.GetPrimAtPath("/World/Joints/door")
    prim.CreateAttribute(property_name, Sdf.ValueTypeNames.Bool, custom=False).Set(
        value
    )
    assert stage.GetRootLayer().Save()

    result = _extract(source, reference)

    assert _joint(result, "/World/Joints/door").topology.joint_type == "revolute"


@pytest.mark.parametrize(
    ("property_name", "value"),
    [
        ("physics:jointEnabled", False),
        ("physics:collisionEnabled", True),
        ("physics:excludeFromArticulation", True),
    ],
)
def test_nondefault_base_joint_properties_fail_closed(
    tmp_path: Path,
    property_name: str,
    value: bool,
) -> None:
    source, reference = _write_pair(tmp_path)
    stage = Usd.Stage.Open(str(reference))
    stage.GetPrimAtPath("/World/Joints/door").CreateAttribute(
        property_name,
        Sdf.ValueTypeNames.Bool,
        custom=False,
    ).Set(value)
    assert stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "unsupported_joint_property"
    assert property_name in caught.value.detail


@pytest.mark.parametrize("with_target", [False, True], ids=("empty", "targeted"))
def test_unrepresented_joint_proxy_relationship_fails_closed(
    tmp_path: Path,
    with_target: bool,
) -> None:
    source, reference = _write_pair(tmp_path)
    stage = Usd.Stage.Open(str(reference))
    joint = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath("/World/Joints/door"))
    relationship = joint.CreateProxyPrimRel()
    if with_target:
        relationship.SetTargets(["/World/door"])
    assert relationship.IsAuthored()
    assert stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "unsupported_joint_relationship"
    assert "/World/Joints/door" in caught.value.detail
    assert "proxyPrim" in caught.value.detail


def test_time_sampled_base_joint_fallback_fails_closed(tmp_path: Path) -> None:
    source, reference = _write_pair(tmp_path)
    stage = Usd.Stage.Open(str(reference))
    attribute = stage.GetPrimAtPath("/World/Joints/door").CreateAttribute(
        "physics:jointEnabled",
        Sdf.ValueTypeNames.Bool,
        custom=False,
    )
    attribute.Set(True)
    attribute.Set(True, Usd.TimeCode(1.0))
    assert stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "time_sampled_static_property"
    assert "physics:jointEnabled" in caught.value.detail


@pytest.mark.parametrize("property_name", ["physics:breakForce", "physics:breakTorque"])
def test_finite_joint_break_thresholds_fail_closed(
    tmp_path: Path,
    property_name: str,
) -> None:
    source, reference = _write_pair(tmp_path)
    stage = Usd.Stage.Open(str(reference))
    stage.GetPrimAtPath("/World/Joints/door").CreateAttribute(
        property_name,
        Sdf.ValueTypeNames.Float,
        custom=False,
    ).Set(100.0)
    assert stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "unsupported_joint_property"
    assert property_name in caught.value.detail


@pytest.mark.parametrize("value", [math.nan, -math.inf])
def test_invalid_joint_break_thresholds_fail_closed(
    tmp_path: Path,
    value: float,
) -> None:
    source, reference = _write_pair(tmp_path)
    stage = Usd.Stage.Open(str(reference))
    stage.GetPrimAtPath("/World/Joints/door").CreateAttribute(
        "physics:breakForce",
        Sdf.ValueTypeNames.Float,
        custom=False,
    ).Set(value)
    assert stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "invalid_joint_property"
    assert "physics:breakForce" in caught.value.detail


def test_explicit_unbreakable_joint_threshold_is_equivalent_to_omission(
    tmp_path: Path,
) -> None:
    source, reference = _write_pair(tmp_path)
    stage = Usd.Stage.Open(str(reference))
    stage.GetPrimAtPath("/World/Joints/door").CreateAttribute(
        "physics:breakForce",
        Sdf.ValueTypeNames.Float,
        custom=False,
    ).Set(math.inf)
    assert stage.GetRootLayer().Save()

    result = _extract(source, reference)

    assert _joint(result, "/World/Joints/door").topology.joint_type == "revolute"


@pytest.mark.parametrize("property_name", ["physics:breakForce", "physics:breakTorque"])
def test_time_sampled_unbreakable_joint_thresholds_fail_closed(
    tmp_path: Path,
    property_name: str,
) -> None:
    source, reference = _write_pair(tmp_path)
    stage = Usd.Stage.Open(str(reference))
    attribute = stage.GetPrimAtPath("/World/Joints/door").CreateAttribute(
        property_name,
        Sdf.ValueTypeNames.Float,
        custom=False,
    )
    attribute.Set(math.inf)
    attribute.Set(math.inf, Usd.TimeCode(1.0))
    assert stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "time_sampled_static_property"
    assert property_name in caught.value.detail


@pytest.mark.parametrize(
    ("family", "field"),
    [
        ("joint_frame", "axis"),
        ("joint_frame", "localRot0"),
        ("joint_frame", "localRot1"),
        ("anchor", "localPos0"),
        ("anchor", "localPos1"),
        ("limit", "lowerLimit"),
        ("limit", "upperLimit"),
        ("drive", "drive_type"),
        ("drive", "stiffness"),
        ("drive", "damping"),
        ("drive", "max_force"),
        ("drive", "target_position"),
        ("drive", "target_velocity"),
        ("physx_velocity", "maxJointVelocity"),
        ("physx_friction", "jointFriction"),
        ("mass", "mass"),
        ("mass", "centerOfMass"),
        ("mass", "diagonalInertia"),
        ("mass", "principalAxes"),
        ("mesh_collision", "approximation"),
    ],
)
def test_all_v1_static_extractor_fields_reject_time_samples(
    tmp_path: Path,
    family: str,
    field: str,
) -> None:
    source, reference = _write_pair(tmp_path)
    stage = Usd.Stage.Open(str(reference))
    attribute, value = _static_extractor_attribute(stage, family, field)
    _author_time_sample(attribute, value)
    assert stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "time_sampled_static_property"
    assert str(attribute.GetName()) in caught.value.detail


def test_static_attribute_guard_rejects_ancestor_value_clip_metadata() -> None:
    stage = Usd.Stage.CreateInMemory()
    world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    body = UsdGeom.Xform.Define(stage, "/World/body").GetPrim()
    attribute = body.CreateAttribute(
        "physics:mass",
        Sdf.ValueTypeNames.Float,
        custom=False,
    )
    assert attribute.Set(2.0)
    clips = Usd.ClipsAPI(world)
    clips.SetClipAssetPaths([Sdf.AssetPath("latent-mass.usda")], "mass")
    assert clips.SetClipPrimPath("/Clip", "mass")

    with pytest.raises(JointRiggerContractError) as caught:
        rv._require_static_attribute(attribute, owner_path="/World/body")

    assert caught.value.code == "time_sampled_static_property"
    assert "physics:mass" in caught.value.detail
    assert "value_clip_sources=" in caught.value.detail
    assert "/World" in caught.value.detail


def test_revolute_limits_and_invalid_prismatic_stage_units(tmp_path: Path) -> None:
    source, reference = _write_pair(tmp_path / "revolute")
    stage = Usd.Stage.Open(str(reference))
    door = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath("/World/Joints/door"))
    door.CreateLowerLimitAttr(-45.0)
    door.CreateUpperLimitAttr(90.0)
    stage.GetRootLayer().Save()
    result = _extract(source, reference)
    plan = _joint(result, "/World/Joints/door")
    assert plan.limit is not None
    assert (plan.limit.lower, plan.limit.upper, plan.limit.unit) == (
        -45.0,
        90.0,
        "degrees",
    )

    source, reference = _write_pair(tmp_path / "units")
    stage = Usd.Stage.Open(str(reference))
    stage.SetMetadata("metersPerUnit", 0.0)
    stage.GetRootLayer().Save()
    with pytest.raises(JointRiggerContractError) as invalid_units:
        _extract(source, reference)
    assert invalid_units.value.code == "invalid_stage_units"


def test_limit_value_block_is_incomplete_contract_evidence(tmp_path: Path) -> None:
    source, reference = _write_pair(tmp_path)
    stage = Usd.Stage.Open(str(reference))
    drawer = UsdPhysics.PrismaticJoint(stage.GetPrimAtPath("/World/Joints/drawer"))
    attribute = drawer.GetLowerLimitAttr()
    assert attribute.Set(Sdf.ValueBlock())
    assert attribute.HasAuthoredValueOpinion()
    assert attribute.Get() is None
    assert stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "incomplete_optional_schema"
    assert "physics:lowerLimit" in caught.value.detail


@pytest.mark.parametrize(
    ("mutation", "reason_code"),
    [
        ("multiple", "unsupported_optional_schema"),
        ("incomplete", "incomplete_optional_schema"),
        ("invalid_type", "invalid_drive_type"),
    ],
)
def test_drive_schema_guardrails(
    tmp_path: Path,
    mutation: str,
    reason_code: str,
) -> None:
    source, reference = _write_pair(tmp_path)
    stage = Usd.Stage.Open(str(reference))
    prim = stage.GetPrimAtPath("/World/Joints/drawer")
    drive = UsdPhysics.DriveAPI.Get(prim, "linear")
    if mutation == "multiple":
        UsdPhysics.DriveAPI.Apply(prim, "angular")
    elif mutation == "incomplete":
        drive.GetDampingAttr().Clear()
    else:
        drive.GetTypeAttr().Set("invalid")
    stage.GetRootLayer().Save()
    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)
    assert caught.value.code == reason_code


@pytest.mark.parametrize(
    "field",
    [
        "drive_type",
        "stiffness",
        "damping",
        "max_force",
        "target_position",
        "target_velocity",
    ],
)
def test_drive_value_blocks_are_incomplete_contract_evidence(
    tmp_path: Path,
    field: str,
) -> None:
    source, reference = _write_pair(tmp_path)
    stage = Usd.Stage.Open(str(reference))
    attribute, _ = _static_extractor_attribute(stage, "drive", field)
    assert attribute.Set(Sdf.ValueBlock())
    assert attribute.HasAuthoredValueOpinion()
    assert attribute.Get() is None
    assert stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "incomplete_optional_schema"
    assert str(attribute.GetName()) in caught.value.detail


@pytest.mark.parametrize("field", ["stiffness", "target_position"])
def test_nonfinite_drive_values_have_stable_contract_errors(
    tmp_path: Path,
    field: str,
) -> None:
    source, reference = _write_pair(tmp_path)
    stage = Usd.Stage.Open(str(reference))
    attribute, _ = _static_extractor_attribute(stage, "drive", field)
    assert attribute.Set(math.nan)
    assert stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "invalid_drive_value"
    assert field in caught.value.detail


def test_negative_drive_value_has_stable_contract_error(tmp_path: Path) -> None:
    source, reference = _write_pair(tmp_path)
    stage = Usd.Stage.Open(str(reference))
    attribute, _ = _static_extractor_attribute(stage, "drive", "stiffness")
    assert attribute.Set(-1.0)
    assert stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "invalid_drive_value"
    assert "negative stiffness" in caught.value.detail


@pytest.mark.parametrize(
    ("mutation", "reason_code"),
    [
        ("remove_matching_api", "drive_property_without_api"),
        ("mismatched_instance", "drive_property_without_api"),
        ("malformed_raw_property", "drive_property_without_api"),
        ("unrepresented_property", "unsupported_optional_schema"),
    ],
)
def test_authored_drive_properties_require_matching_represented_api(
    tmp_path: Path,
    mutation: str,
    reason_code: str,
) -> None:
    source, reference = _write_pair(tmp_path)
    stage = Usd.Stage.Open(str(reference))
    prim = stage.GetPrimAtPath("/World/Joints/drawer")
    if mutation == "remove_matching_api":
        prim.RemoveAppliedSchema("PhysicsDriveAPI:linear")
    else:
        name = {
            "mismatched_instance": "drive:angular:physics:stiffness",
            "malformed_raw_property": "drive:linear:raw",
            "unrepresented_property": "drive:linear:physics:feedForward",
        }[mutation]
        prim.CreateAttribute(name, Sdf.ValueTypeNames.Float, custom=False).Set(1.0)
    assert stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == reason_code
    assert "drive:" in caught.value.detail


def test_joint_without_drive_api_or_properties_has_no_drive(tmp_path: Path) -> None:
    source, reference = _write_pair(tmp_path)

    door = _joint(_extract(source, reference), "/World/Joints/door")

    assert door.drive is None


def test_revolute_angular_drive_is_preserved(tmp_path: Path) -> None:
    source, reference = _write_pair(tmp_path)
    stage = Usd.Stage.Open(str(reference))
    _apply_complete_drive(stage.GetPrimAtPath("/World/Joints/door"), "angular")
    stage.GetRootLayer().Save()

    result = _extract(source, reference)

    drive = _joint(result, "/World/Joints/door").drive
    assert drive is not None
    assert drive.drive_type == "force"
    assert drive.stiffness == 25.0


def test_physx_max_joint_velocity_is_preserved_with_drive_provenance(
    tmp_path: Path,
) -> None:
    source, reference = _write_pair(tmp_path)
    stage = Usd.Stage.Open(str(reference))
    prim = stage.GetPrimAtPath("/World/Joints/drawer")
    assert prim.AddAppliedSchema("PhysxJointAPI")
    prim.CreateAttribute(
        "physxJoint:maxJointVelocity",
        Sdf.ValueTypeNames.Float,
        custom=False,
    ).Set(3.5)
    prim.CreateAttribute(
        "physxJoint:jointFriction",
        Sdf.ValueTypeNames.Float,
        custom=False,
    ).Set(0.25)
    assert stage.GetRootLayer().Save()

    joint = _joint(_extract(source, reference), "/World/Joints/drawer")
    drive = joint.drive
    assert drive is not None
    assert drive.max_joint_velocity == 3.5
    assert "physxJoint:maxJointVelocity" in drive.provenance.properties
    assert "physxJoint:jointFriction" not in drive.provenance.properties
    assert joint.joint_friction is not None
    assert joint.joint_friction.coefficient == 0.25
    assert joint.joint_friction.provenance.properties == ("physxJoint:jointFriction",)


@pytest.mark.parametrize(
    ("joint_path", "coefficient", "driven"),
    [
        ("/World/Joints/door", 0.0, False),
        ("/World/Joints/drawer", 0.15, True),
    ],
)
def test_physx_joint_friction_is_independent_for_passive_and_driven_scalar_joints(
    tmp_path: Path,
    joint_path: str,
    coefficient: float,
    driven: bool,
) -> None:
    source, reference = _write_pair(tmp_path)
    stage = Usd.Stage.Open(str(reference))
    prim = stage.GetPrimAtPath(joint_path)
    assert prim.AddAppliedSchema("PhysxJointAPI")
    prim.CreateAttribute(
        "physxJoint:jointFriction",
        Sdf.ValueTypeNames.Float,
        custom=False,
    ).Set(coefficient)
    assert stage.GetRootLayer().Save()

    joint = _joint(_extract(source, reference), joint_path)

    assert (joint.drive is not None) is driven
    assert joint.joint_friction is not None
    assert joint.joint_friction.coefficient == pytest.approx(coefficient)
    assert joint.joint_friction.provenance.prim_path == joint_path
    assert joint.joint_friction.provenance.properties == ("physxJoint:jointFriction",)


@pytest.mark.parametrize(
    ("mutation", "reason_code"),
    [
        ("schema_only", "unsupported_optional_schema"),
        ("without_schema", "unsupported_optional_schema"),
        ("without_drive", "drive_property_without_api"),
        ("unknown_opinion", "unsupported_optional_schema"),
        ("instance_schema", "unsupported_optional_schema"),
        ("invalid_value", "invalid_drive_value"),
    ],
)
def test_physx_joint_opinions_fail_closed_when_unrepresented(
    tmp_path: Path,
    mutation: str,
    reason_code: str,
) -> None:
    source, reference = _write_pair(tmp_path)
    stage = Usd.Stage.Open(str(reference))
    prim = stage.GetPrimAtPath("/World/Joints/drawer")
    if mutation != "without_schema":
        assert prim.AddAppliedSchema(
            "PhysxJointAPI:unsupported"
            if mutation == "instance_schema"
            else "PhysxJointAPI"
        )
    if mutation == "without_drive":
        prim.RemoveAppliedSchema("PhysicsDriveAPI:linear")
    if mutation != "schema_only":
        if mutation == "unknown_opinion":
            prim.CreateAttribute(
                "physxJoint:solverFoo",
                Sdf.ValueTypeNames.Float,
                custom=False,
            ).Set(1.0)
        else:
            prim.CreateAttribute(
                "physxJoint:maxJointVelocity",
                Sdf.ValueTypeNames.Float,
                custom=False,
            ).Set(-1.0 if mutation == "invalid_value" else 3.5)
    assert stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)
    assert caught.value.code == reason_code


@pytest.mark.parametrize(
    ("mutation", "reason_code"),
    [
        ("without_schema", "unsupported_optional_schema"),
        ("spherical", "joint_friction_not_applicable"),
        ("negative", "invalid_joint_friction"),
        ("nonfinite", "invalid_joint_friction"),
        ("blocked", "invalid_joint_friction"),
    ],
)
def test_physx_joint_friction_fails_closed_for_invalid_authored_evidence(
    tmp_path: Path,
    mutation: str,
    reason_code: str,
) -> None:
    source, reference = _write_pair(tmp_path)
    stage = Usd.Stage.Open(str(reference))
    joint_path = (
        "/World/Joints/spherical" if mutation == "spherical" else "/World/Joints/door"
    )
    prim = stage.GetPrimAtPath(joint_path)
    if mutation != "without_schema":
        assert prim.AddAppliedSchema("PhysxJointAPI")
    attribute = prim.CreateAttribute(
        "physxJoint:jointFriction",
        Sdf.ValueTypeNames.Float,
        custom=False,
    )
    value: Any = {
        "without_schema": 0.1,
        "spherical": 0.1,
        "negative": -0.1,
        "nonfinite": math.nan,
        "blocked": Sdf.ValueBlock(),
    }[mutation]
    assert attribute.Set(value)
    assert stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == reason_code
    assert (
        "joint friction" in caught.value.detail.lower() or mutation == "without_schema"
    )


@pytest.mark.parametrize(
    ("joint_path", "instance"),
    [
        ("/World/Joints/drawer", "angular"),
        ("/World/Joints/door", "linear"),
        ("/World/Joints/spherical", "angular"),
    ],
)
def test_drive_instance_must_match_supported_joint_type(
    tmp_path: Path,
    joint_path: str,
    instance: str,
) -> None:
    source, reference = _write_pair(tmp_path)
    stage = Usd.Stage.Open(str(reference))
    prim = stage.GetPrimAtPath(joint_path)
    if joint_path == "/World/Joints/drawer":
        _remove_drive_schema_and_properties(prim, "linear")
    _apply_complete_drive(prim, instance)
    stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "unsupported_drive_instance"


@pytest.mark.parametrize(
    "schema",
    [
        "PhysicsJointStateAPI:angular",
        "PhysxMimicJointAPI:rotZ",
    ],
)
def test_unrepresented_joint_state_and_mimic_schemas_fail_closed(
    tmp_path: Path,
    schema: str,
) -> None:
    source, reference = _write_pair(tmp_path)
    stage = Usd.Stage.Open(str(reference))
    stage.GetPrimAtPath("/World/Joints/door").AddAppliedSchema(schema)
    stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "unsupported_optional_schema"
    assert schema in caught.value.detail


@pytest.mark.parametrize(
    "schema",
    [
        "PhysicsRigidBodyAPI",
        "PhysicsMassAPI",
        "PhysicsCollisionAPI",
        "PhysxUnsupportedJointAPI",
    ],
)
def test_every_other_applied_physics_api_on_selected_joint_fails_closed(
    tmp_path: Path,
    schema: str,
) -> None:
    source, reference = _write_pair(tmp_path)
    stage = Usd.Stage.Open(str(reference))
    assert stage.GetPrimAtPath("/World/Joints/door").AddAppliedSchema(schema)
    assert stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "unsupported_optional_schema"
    assert schema in caught.value.detail


@pytest.mark.parametrize(
    ("property_name", "value_type", "value"),
    [
        (
            "state:angular:physics:position",
            Sdf.ValueTypeNames.Float,
            0.0,
        ),
        (
            "physxMimicJoint:rotZ:gearing",
            Sdf.ValueTypeNames.Float,
            1.0,
        ),
    ],
)
def test_raw_joint_state_and_mimic_properties_fail_closed_without_api(
    tmp_path: Path,
    property_name: str,
    value_type: Any,
    value: Any,
) -> None:
    source, reference = _write_pair(tmp_path)
    stage = Usd.Stage.Open(str(reference))
    stage.GetPrimAtPath("/World/Joints/door").CreateAttribute(
        property_name,
        value_type,
        custom=False,
    ).Set(value)
    assert stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "unsupported_optional_schema"
    assert property_name in caught.value.detail


@pytest.mark.parametrize(
    (
        "property_name",
        "value_type",
        "fallback",
        "_nondefault",
        "_unsupported_code",
    ),
    _BODY_FALLBACK_CASES,
)
def test_explicit_rigid_body_and_collision_fallbacks_are_accepted(
    tmp_path: Path,
    property_name: str,
    value_type: Any,
    fallback: Any,
    _nondefault: Any,
    _unsupported_code: str,
) -> None:
    source, reference = _write_pair(tmp_path)
    stage = Usd.Stage.Open(str(reference))
    stage.GetPrimAtPath("/World/base").CreateAttribute(
        property_name,
        value_type,
        custom=False,
    ).Set(fallback)
    assert stage.GetRootLayer().Save()

    result = _extract(source, reference)

    assert any(body.prim_path == "/World/base" for body in result.plan.rigid_bodies)


@pytest.mark.parametrize(
    (
        "property_name",
        "value_type",
        "_fallback",
        "nondefault",
        "unsupported_code",
    ),
    _BODY_FALLBACK_CASES,
)
def test_nondefault_rigid_body_and_collision_properties_fail_closed(
    tmp_path: Path,
    property_name: str,
    value_type: Any,
    _fallback: Any,
    nondefault: Any,
    unsupported_code: str,
) -> None:
    source, reference = _write_pair(tmp_path)
    stage = Usd.Stage.Open(str(reference))
    stage.GetPrimAtPath("/World/base").CreateAttribute(
        property_name,
        value_type,
        custom=False,
    ).Set(nondefault)
    assert stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == unsupported_code
    assert property_name in caught.value.detail


@pytest.mark.parametrize("with_target", [False, True], ids=("empty", "targeted"))
def test_unrepresented_rigid_body_simulation_owner_fails_closed(
    tmp_path: Path,
    with_target: bool,
) -> None:
    source, reference = _write_pair(tmp_path)
    stage = Usd.Stage.Open(str(reference))
    if with_target:
        UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    relationship = UsdPhysics.RigidBodyAPI(
        stage.GetPrimAtPath("/World/door")
    ).CreateSimulationOwnerRel()
    if with_target:
        relationship.SetTargets(["/World/PhysicsScene"])
    assert relationship.IsAuthored()
    assert stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "unsupported_rigid_body_relationship"
    assert "/World/door" in caught.value.detail
    assert "physics:simulationOwner" in caught.value.detail


@pytest.mark.parametrize("artifact", ["source", "reference"])
def test_endpoint_simulation_owner_without_rigid_body_api_fails_closed(
    tmp_path: Path,
    artifact: str,
) -> None:
    source, reference = _write_pair(tmp_path)
    stage_path = source if artifact == "source" else reference
    stage = Usd.Stage.Open(str(stage_path))
    prim = stage.GetPrimAtPath("/World/door")
    if artifact == "reference":
        _remove_rigid_body_schema_and_properties(prim)
    relationship = prim.CreateRelationship(
        "physics:simulationOwner",
        custom=False,
    )
    relationship.SetTargets(["/World/base"])
    assert stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "unsupported_rigid_body_relationship"
    assert "/World/door" in caught.value.detail
    assert "physics:simulationOwner" in caught.value.detail


@pytest.mark.parametrize("with_target", [False, True], ids=("empty", "targeted"))
def test_unrepresented_collider_simulation_owner_fails_closed(
    tmp_path: Path,
    with_target: bool,
) -> None:
    source, reference = _write_pair(tmp_path)
    collider_path = "/World/base/childCollider"
    source_stage = Usd.Stage.Open(str(source))
    UsdGeom.Cube.Define(source_stage, collider_path)
    assert source_stage.GetRootLayer().Save()
    reference_stage = Usd.Stage.Open(str(reference))
    collider = UsdGeom.Cube.Define(reference_stage, collider_path).GetPrim()
    collision_api = UsdPhysics.CollisionAPI.Apply(collider)
    collision_api.CreateCollisionEnabledAttr(True)
    if with_target:
        UsdPhysics.Scene.Define(reference_stage, "/World/PhysicsScene")
    relationship = collision_api.CreateSimulationOwnerRel()
    if with_target:
        relationship.SetTargets(["/World/PhysicsScene"])
    assert relationship.IsAuthored()
    assert reference_stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "unsupported_collision_relationship"
    assert collider_path in caught.value.detail
    assert "physics:simulationOwner" in caught.value.detail


@pytest.mark.parametrize("artifact", ["source", "reference"])
def test_descendant_simulation_owner_without_collision_api_fails_closed(
    tmp_path: Path,
    artifact: str,
) -> None:
    source, reference = _write_pair(tmp_path)
    collider_path = "/World/base/relationshipOnlyCollider"
    for stage_path in (source, reference):
        stage = Usd.Stage.Open(str(stage_path))
        prim = UsdGeom.Cube.Define(stage, collider_path).GetPrim()
        if stage_path == (source if artifact == "source" else reference):
            prim.CreateRelationship(
                "physics:simulationOwner",
                custom=False,
            ).SetTargets(["/World/base"])
        assert stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "unsupported_collision_relationship"
    assert collider_path in caught.value.detail
    assert "physics:simulationOwner" in caught.value.detail


def test_source_only_descendant_simulation_owner_fails_closed(
    tmp_path: Path,
) -> None:
    source, reference = _write_pair(tmp_path)
    source_stage = Usd.Stage.Open(str(source))
    collider_path = "/World/base/sourceOnlyCollider"
    prim = UsdGeom.Cube.Define(source_stage, collider_path).GetPrim()
    prim.CreateRelationship(
        "physics:simulationOwner",
        custom=False,
    ).SetTargets(["/World/base"])
    assert source_stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "unsupported_collision_relationship"
    assert collider_path in caught.value.detail
    assert "physics:simulationOwner" in caught.value.detail


@pytest.mark.parametrize(
    (
        "property_name",
        "value_type",
        "fallback",
        "_nondefault",
        "_unsupported_code",
    ),
    _BODY_FALLBACK_CASES,
)
def test_time_sampled_rigid_body_and_collision_fallbacks_fail_closed(
    tmp_path: Path,
    property_name: str,
    value_type: Any,
    fallback: Any,
    _nondefault: Any,
    _unsupported_code: str,
) -> None:
    source, reference = _write_pair(tmp_path)
    stage = Usd.Stage.Open(str(reference))
    attribute = stage.GetPrimAtPath("/World/base").CreateAttribute(
        property_name,
        value_type,
        custom=False,
    )
    _author_time_sample(attribute, fallback)
    assert stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "time_sampled_static_property"
    assert property_name in caught.value.detail


def test_malformed_vector_fallback_is_rejected_without_conversion_error() -> None:
    assert not rv._matches_schema_fallback(object(), (0.0, 0.0, 0.0))


@pytest.mark.parametrize(
    ("property_name", "value_type", "value"),
    [
        ("physics:rigidBodyEnabled", Sdf.ValueTypeNames.Bool, True),
        ("physics:velocity", Sdf.ValueTypeNames.Vector3f, Gf.Vec3f(0.0)),
    ],
)
def test_rigid_body_properties_without_rigid_body_api_fail_closed(
    tmp_path: Path,
    property_name: str,
    value_type: Any,
    value: Any,
) -> None:
    source, reference = _write_pair(tmp_path)
    stage = Usd.Stage.Open(str(reference))
    prim = stage.GetPrimAtPath("/World/door")
    prim.RemoveAppliedSchema("PhysicsRigidBodyAPI")
    prim.CreateAttribute(property_name, value_type, custom=False).Set(value)
    assert stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "rigid_body_property_without_api"
    assert property_name in caught.value.detail


def test_principal_axes_alone_is_incomplete_mass_evidence(tmp_path: Path) -> None:
    source, reference = _write_pair(tmp_path)
    stage = Usd.Stage.Open(str(reference))
    mass = UsdPhysics.MassAPI.Apply(stage.GetPrimAtPath("/World/ball"))
    mass.CreatePrincipalAxesAttr(Gf.Quatf(1.0, Gf.Vec3f(0.0)))
    assert stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "incomplete_optional_schema"
    assert "/World/ball" in caught.value.detail


@pytest.mark.parametrize(
    "field",
    ["mass", "centerOfMass", "diagonalInertia", "principalAxes"],
)
def test_mass_value_blocks_are_incomplete_contract_evidence(
    tmp_path: Path,
    field: str,
) -> None:
    source, reference = _write_pair(tmp_path)
    stage = Usd.Stage.Open(str(reference))
    attribute, _ = _static_extractor_attribute(stage, "mass", field)
    assert attribute.Set(Sdf.ValueBlock())
    assert attribute.HasAuthoredValueOpinion()
    assert attribute.Get() is None
    assert stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "incomplete_optional_schema"
    assert str(attribute.GetName()) in caught.value.detail


@pytest.mark.parametrize("field", ["mass", "centerOfMass", "diagonalInertia"])
def test_nonfinite_mass_values_have_stable_contract_errors(
    tmp_path: Path,
    field: str,
) -> None:
    source, reference = _write_pair(tmp_path)
    stage = Usd.Stage.Open(str(reference))
    attribute, value = _static_extractor_attribute(stage, "mass", field)
    invalid = math.nan if field == "mass" else Gf.Vec3f(math.nan, value[1], value[2])
    assert attribute.Set(invalid)
    assert stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "invalid_mass_properties"
    assert "/World/base" in caught.value.detail


@pytest.mark.parametrize(
    ("mutation", "detail"),
    [
        ("nonpositive_mass", "mass must be positive"),
        ("nonpositive_inertia", "inertia components must be positive"),
        ("inertia_triangle", "inertia violates the inertia triangle"),
        ("unnormalized_axes", "principal axes must be normalized"),
    ],
)
def test_invalid_mass_shapes_have_stable_contract_errors(
    tmp_path: Path,
    mutation: str,
    detail: str,
) -> None:
    source, reference = _write_pair(tmp_path)
    stage = Usd.Stage.Open(str(reference))
    mass = UsdPhysics.MassAPI(stage.GetPrimAtPath("/World/base"))
    if mutation == "nonpositive_mass":
        mass.GetMassAttr().Set(0.0)
    elif mutation == "nonpositive_inertia":
        mass.GetDiagonalInertiaAttr().Set(Gf.Vec3f(0.0, 1.0, 1.0))
    elif mutation == "inertia_triangle":
        mass.GetDiagonalInertiaAttr().Set(Gf.Vec3f(1.0, 1.0, 3.0))
    else:
        mass.CreatePrincipalAxesAttr(Gf.Quatf(2.0, Gf.Vec3f(0.0)))
    assert stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "invalid_mass_properties"
    assert detail in caught.value.detail


def test_mass_si_conversion_overflow_has_stable_contract_error(
    tmp_path: Path,
) -> None:
    source, reference = _write_pair(tmp_path)
    for path in (source, reference):
        stage = Usd.Stage.Open(str(path))
        UsdPhysics.SetStageKilogramsPerUnit(stage, 1e308)
        assert stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "invalid_mass_properties"
    assert "conversion to SI produced non-finite" in caught.value.detail


@pytest.mark.parametrize("invalid_field", ["inertia", "principal_axes"])
def test_type_impossible_mass_runtime_shapes_have_stable_contract_errors(
    tmp_path: Path,
    invalid_field: str,
) -> None:
    source, reference = _write_pair(tmp_path)

    class Attribute:
        def __init__(self, name: str, value: Any, *, authored: bool = True) -> None:
            self._name = name
            self._value = value
            self._authored = authored

        def HasAuthoredValueOpinion(self) -> bool:
            return self._authored

        def HasAuthoredConnections(self) -> bool:
            return False

        def Get(self) -> Any:
            return self._value

        def GetTimeSamples(self) -> list[float]:
            return []

        def GetName(self) -> str:
            return self._name

    class MassAPI:
        def GetMassAttr(self) -> Attribute:
            return Attribute("physics:mass", 1.0)

        def GetDiagonalInertiaAttr(self) -> Attribute:
            value = object() if invalid_field == "inertia" else (1.0, 1.0, 1.0)
            return Attribute("physics:diagonalInertia", value)

        def GetPrincipalAxesAttr(self) -> Attribute:
            return Attribute(
                "physics:principalAxes",
                object(),
                authored=invalid_field == "principal_axes",
            )

    class Physics:
        MassAPI = staticmethod(lambda prim: MassAPI())

    class Prim:
        def HasAPI(self, schema: Any) -> bool:
            return True

        def GetAttribute(self, name: str) -> Attribute:
            return Attribute(name, None, authored=False)

    with pytest.raises(JointRiggerContractError) as caught:
        rv._extract_mass(
            Prim(),
            body_path="/World/body",
            reference_identity=_identity(reference),
            kilograms_per_unit=1.0,
            meters_per_unit=1.0,
            UsdPhysics=Physics,
        )

    assert caught.value.code == "invalid_mass_properties"
    assert (
        "not a three-component value"
        if invalid_field == "inertia"
        else "invalid principal axes"
    ) in caught.value.detail


def test_mass_api_schema_without_rigid_body_fails_closed(tmp_path: Path) -> None:
    source, reference = _write_pair(tmp_path)
    stage = Usd.Stage.Open(str(reference))
    prim = stage.GetPrimAtPath("/World/door")
    _remove_rigid_body_schema_and_properties(prim)
    UsdPhysics.MassAPI.Apply(prim)
    assert stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "mass_without_rigid_body"
    assert "PhysicsMassAPI" in caught.value.detail


def test_collision_api_schema_without_rigid_body_fails_closed(tmp_path: Path) -> None:
    source, reference = _write_pair(tmp_path)
    stage = Usd.Stage.Open(str(reference))
    prim = stage.GetPrimAtPath("/World/door")
    _remove_rigid_body_schema_and_properties(prim)
    assert prim.HasAPI(UsdPhysics.CollisionAPI)
    assert stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "collision_without_rigid_body"
    assert "/World/door" in caught.value.detail
    assert "PhysicsCollisionAPI" in caught.value.detail


@pytest.mark.parametrize("evidence", ["api", "property", "simulation_owner"])
def test_descendant_collider_evidence_without_rigid_body_fails_closed(
    tmp_path: Path,
    evidence: str,
) -> None:
    source, reference = _write_pair(tmp_path)
    child_path = "/World/door/unownedCollider"
    for stage_path in (source, reference):
        stage = Usd.Stage.Open(str(stage_path))
        if stage_path == reference:
            door = stage.GetPrimAtPath("/World/door")
            _remove_rigid_body_schema_and_properties(door)
            _remove_collision_schema_and_properties(door)
        child = UsdGeom.Cube.Define(stage, child_path).GetPrim()
        if stage_path == reference:
            if evidence == "api":
                UsdPhysics.CollisionAPI.Apply(child)
            elif evidence == "property":
                child.CreateAttribute(
                    "physics:collisionEnabled",
                    Sdf.ValueTypeNames.Bool,
                    custom=False,
                ).Set(True)
            else:
                child.CreateRelationship(
                    "physics:simulationOwner",
                    custom=False,
                ).SetTargets(["/World/base"])
        assert stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "collision_without_rigid_body"
    assert child_path in caught.value.detail


@pytest.mark.parametrize("composed_state", ["inactive", "undefined"])
def test_inactive_unowned_collider_evidence_is_ignored(
    tmp_path: Path,
    composed_state: str,
) -> None:
    source, reference = _write_pair(tmp_path)
    stage = Usd.Stage.Open(str(reference))
    door = stage.GetPrimAtPath("/World/door")
    _remove_rigid_body_schema_and_properties(door)
    _remove_collision_schema_and_properties(door)
    child = UsdGeom.Cube.Define(stage, "/World/door/ignoredCollider").GetPrim()
    UsdPhysics.CollisionAPI.Apply(child)
    if composed_state == "inactive":
        child.SetActive(False)
    else:
        child.SetSpecifier(Sdf.SpecifierOver)
    assert stage.GetRootLayer().Save()

    result = _extract(source, reference)

    assert all(body.prim_path != "/World/door" for body in result.plan.rigid_bodies)


def test_reference_only_nested_body_and_collider_fail_closed(
    tmp_path: Path,
) -> None:
    source, reference = _write_pair(tmp_path)
    nested_body_path = "/World/door/nestedBody"
    collider_path = f"{nested_body_path}/collider"
    for stage_path in (source, reference):
        stage = Usd.Stage.Open(str(stage_path))
        if stage_path == reference:
            door = stage.GetPrimAtPath("/World/door")
            _remove_rigid_body_schema_and_properties(door)
            _remove_collision_schema_and_properties(door)
        nested_body = UsdGeom.Xform.Define(stage, nested_body_path).GetPrim()
        UsdGeom.Cube.Define(stage, collider_path)
        if stage_path == reference:
            UsdPhysics.RigidBodyAPI.Apply(nested_body)
            UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath(collider_path))
        assert stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "unrepresented_descendant_rigid_body"
    assert nested_body_path in caught.value.detail


@pytest.mark.parametrize(
    ("missing_fact", "expected_fact"),
    [
        ("collision_api", "PhysicsCollisionAPI"),
        ("collision_property", "physics:collisionEnabled"),
        ("mass_api", "PhysicsMassAPI"),
    ],
)
def test_unplanned_nested_body_requires_exact_descendant_physics(
    tmp_path: Path,
    missing_fact: str,
    expected_fact: str,
) -> None:
    source, reference = _write_pair(tmp_path)
    nested_body_path = "/World/door/nestedBody"
    collider_path = f"{nested_body_path}/collider"
    for stage_path in (source, reference):
        stage = Usd.Stage.Open(str(stage_path))
        nested = UsdGeom.Xform.Define(stage, nested_body_path).GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(nested)
        collider = UsdGeom.Cube.Define(stage, collider_path).GetPrim()
        if missing_fact == "collision_property":
            collision = UsdPhysics.CollisionAPI.Apply(collider)
            if stage_path == reference:
                collision.CreateCollisionEnabledAttr(True)
        elif stage_path == reference and missing_fact == "collision_api":
            UsdPhysics.CollisionAPI.Apply(collider)
        elif stage_path == reference:
            UsdPhysics.MassAPI.Apply(collider)
        assert stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "unrepresented_descendant_rigid_body"
    assert collider_path in caught.value.detail
    assert expected_fact in caught.value.detail


@pytest.mark.parametrize("missing_fact", ["api", "property"])
def test_unplanned_nested_reference_body_requires_exact_source_facts(
    tmp_path: Path,
    missing_fact: str,
) -> None:
    source, reference = _write_pair(tmp_path)
    nested_path = "/World/door/nestedBody"
    for stage_path in (source, reference):
        stage = Usd.Stage.Open(str(stage_path))
        nested = UsdGeom.Xform.Define(stage, nested_path).GetPrim()
        if stage_path == reference or missing_fact == "property":
            body = UsdPhysics.RigidBodyAPI.Apply(nested)
            if stage_path == reference and missing_fact == "property":
                body.CreateRigidBodyEnabledAttr(True)
        assert stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "unrepresented_descendant_rigid_body"
    assert nested_path in caught.value.detail
    expected = "PhysicsRigidBodyAPI" if missing_fact == "api" else "rigidBodyEnabled"
    assert expected in caught.value.detail


def test_matching_unplanned_nested_body_is_replay_safe(tmp_path: Path) -> None:
    source, reference = _write_pair(tmp_path)
    nested_path = "/World/door/nestedBody"
    for stage_path in (source, reference):
        stage = Usd.Stage.Open(str(stage_path))
        nested = UsdGeom.Xform.Define(stage, nested_path).GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(nested).CreateRigidBodyEnabledAttr(True)
        collider = UsdGeom.Cube.Define(stage, f"{nested_path}/collider").GetPrim()
        UsdPhysics.CollisionAPI.Apply(collider).CreateCollisionEnabledAttr(True)
        assert stage.GetRootLayer().Save()

    result = _extract(source, reference)

    assert all(body.prim_path != nested_path for body in result.plan.rigid_bodies)


@pytest.mark.parametrize(
    ("property_name", "value_type", "value"),
    _MASS_PROPERTY_CASES,
)
def test_mass_properties_without_api_or_rigid_body_fail_closed(
    tmp_path: Path,
    property_name: str,
    value_type: Any,
    value: Any,
) -> None:
    source, reference = _write_pair(tmp_path)
    stage = Usd.Stage.Open(str(reference))
    prim = stage.GetPrimAtPath("/World/door")
    _remove_rigid_body_schema_and_properties(prim)
    assert not prim.HasAPI(UsdPhysics.MassAPI)
    prim.CreateAttribute(property_name, value_type, custom=False).Set(value)
    assert stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "mass_without_rigid_body"
    assert property_name in caught.value.detail


@pytest.mark.parametrize(
    ("property_name", "value_type", "value"),
    _MASS_PROPERTY_CASES,
)
def test_mass_properties_without_api_on_rigid_body_fail_closed(
    tmp_path: Path,
    property_name: str,
    value_type: Any,
    value: Any,
) -> None:
    source, reference = _write_pair(tmp_path)
    stage = Usd.Stage.Open(str(reference))
    prim = stage.GetPrimAtPath("/World/door")
    assert prim.HasAPI(UsdPhysics.RigidBodyAPI)
    assert not prim.HasAPI(UsdPhysics.MassAPI)
    prim.CreateAttribute(property_name, value_type, custom=False).Set(value)
    assert stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "mass_property_without_api"
    assert property_name in caught.value.detail


def test_rigid_body_without_mass_schema_or_properties_has_no_mass(
    tmp_path: Path,
) -> None:
    source, reference = _write_pair(tmp_path)

    result = _extract(source, reference)

    door = next(
        body for body in result.plan.rigid_bodies if body.prim_path == "/World/door"
    )
    assert door.mass is None


def test_bare_mass_api_on_rigid_body_fails_closed(tmp_path: Path) -> None:
    source, reference = _write_pair(tmp_path)
    stage = Usd.Stage.Open(str(reference))
    mass = UsdPhysics.MassAPI.Apply(stage.GetPrimAtPath("/World/door"))
    assert not mass.GetMassAttr().HasAuthoredValueOpinion()
    assert stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "incomplete_optional_schema"
    assert "/World/door" in caught.value.detail
    assert "MassAPI" in caught.value.detail


def test_source_mass_fact_absent_from_reference_fails_closed(tmp_path: Path) -> None:
    source, reference = _write_pair(tmp_path)
    source_stage = Usd.Stage.Open(str(source))
    mass = UsdPhysics.MassAPI.Apply(source_stage.GetPrimAtPath("/World/door"))
    mass.CreateMassAttr(3.0)
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(1.0))
    assert source_stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "source_physics_not_in_reference"
    assert "/World/door PhysicsMassAPI" in caught.value.detail
    assert "/World/door physics:mass" in caught.value.detail


@pytest.mark.parametrize(
    ("mutation", "expected_fact"),
    [
        ("source_only_rigid_body_api", "PhysicsRigidBodyAPI"),
        ("source_only_collision_api", "PhysicsCollisionAPI"),
        ("rigid_body_property_mismatch", "physics:rigidBodyEnabled"),
        ("collision_property_mismatch", "physics:collisionEnabled"),
        ("mass_property_mismatch", "physics:mass"),
        ("articulation_root_mismatch", "PhysicsArticulationRootAPI"),
    ],
)
def test_source_physics_facts_must_be_a_reference_subset(
    tmp_path: Path,
    mutation: str,
    expected_fact: str,
) -> None:
    source, reference = _write_pair(tmp_path)
    source_stage = Usd.Stage.Open(str(source))
    if mutation.startswith("source_only_"):
        path = "/World/base/sourceOnlyPhysics"
        prim = UsdGeom.Xform.Define(source_stage, path).GetPrim()
        if mutation == "source_only_rigid_body_api":
            UsdPhysics.RigidBodyAPI.Apply(prim)
        else:
            UsdPhysics.CollisionAPI.Apply(prim)
    elif mutation == "rigid_body_property_mismatch":
        UsdPhysics.RigidBodyAPI.Apply(
            source_stage.GetPrimAtPath("/World/door")
        ).CreateRigidBodyEnabledAttr(False)
    elif mutation == "collision_property_mismatch":
        UsdPhysics.CollisionAPI.Apply(
            source_stage.GetPrimAtPath("/World/door")
        ).CreateCollisionEnabledAttr(False)
    elif mutation == "mass_property_mismatch":
        mass = UsdPhysics.MassAPI.Apply(source_stage.GetPrimAtPath("/World/base"))
        mass.CreateMassAttr(3.0)
    else:
        UsdPhysics.ArticulationRootAPI.Apply(source_stage.GetPrimAtPath("/World"))
    assert source_stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "source_physics_not_in_reference"
    assert expected_fact in caught.value.detail


def test_matching_source_physics_subset_is_accepted(tmp_path: Path) -> None:
    source, reference = _write_pair(tmp_path)
    source_stage = Usd.Stage.Open(str(source))
    base = source_stage.GetPrimAtPath("/World/base")
    UsdPhysics.RigidBodyAPI.Apply(base).CreateRigidBodyEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(base).CreateMassAttr(2.0)
    UsdPhysics.CollisionAPI.Apply(base).CreateCollisionEnabledAttr(True)
    UsdPhysics.ArticulationRootAPI.Apply(base)
    assert source_stage.GetRootLayer().Save()

    result = _extract(source, reference)

    base_plan = next(
        body for body in result.plan.rigid_bodies if body.prim_path == "/World/base"
    )
    assert base_plan.mass is not None
    assert base_plan.mass.mass_kg == 2.0


@pytest.mark.parametrize(
    ("mutation", "expected_fact"),
    [
        ("source_only_api", "PhysicsRigidBodyAPI"),
        ("reference_only_api", "PhysicsRigidBodyAPI"),
        ("property_mismatch", "physics:rigidBodyEnabled"),
    ],
)
def test_selected_endpoint_ancestor_source_physics_must_match_reference(
    tmp_path: Path,
    mutation: str,
    expected_fact: str,
) -> None:
    source, reference = _write_pair(tmp_path)
    if mutation != "reference_only_api":
        source_stage = Usd.Stage.Open(str(source))
        UsdPhysics.RigidBodyAPI.Apply(
            source_stage.GetPrimAtPath("/World")
        ).CreateRigidBodyEnabledAttr(True)
        assert source_stage.GetRootLayer().Save()
    if mutation in {"reference_only_api", "property_mismatch"}:
        reference_stage = Usd.Stage.Open(str(reference))
        UsdPhysics.RigidBodyAPI.Apply(
            reference_stage.GetPrimAtPath("/World")
        ).CreateRigidBodyEnabledAttr(mutation == "reference_only_api")
        assert reference_stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "source_physics_not_in_reference"
    assert "/World" in caught.value.detail
    assert expected_fact in caught.value.detail


def test_matching_preexisting_selected_endpoint_ancestor_physics_is_allowed(
    tmp_path: Path,
) -> None:
    source, reference = _write_pair(tmp_path)
    for stage_path in (source, reference):
        stage = Usd.Stage.Open(str(stage_path))
        UsdPhysics.RigidBodyAPI.Apply(
            stage.GetPrimAtPath("/World")
        ).CreateRigidBodyEnabledAttr(True)
        assert stage.GetRootLayer().Save()

    result = _extract(source, reference)

    assert all(body.prim_path != "/World" for body in result.plan.rigid_bodies)


@pytest.mark.parametrize("direction", ["source_only", "reference_only"])
@pytest.mark.parametrize(
    ("fact_kind", "target_path", "expected_fact"),
    [
        ("filtered_pairs", "/World", "PhysicsFilteredPairsAPI"),
        (
            "physx_rigid_body",
            "/World/base/unmodeledCollider",
            "PhysxRigidBodyAPI",
        ),
    ],
)
def test_unmodeled_physics_facts_fail_closed_in_both_artifact_directions(
    tmp_path: Path,
    direction: str,
    fact_kind: str,
    target_path: str,
    expected_fact: str,
) -> None:
    source, reference = _write_pair(tmp_path)
    if fact_kind == "physx_rigid_body":
        for stage_path in (source, reference):
            stage = Usd.Stage.Open(str(stage_path))
            collider = UsdGeom.Cube.Define(stage, target_path).GetPrim()
            if stage_path == reference:
                UsdPhysics.CollisionAPI.Apply(collider).CreateCollisionEnabledAttr(True)
            assert stage.GetRootLayer().Save()
    selected_path = source if direction == "source_only" else reference
    stage = Usd.Stage.Open(str(selected_path))
    _apply_unmodeled_physics_fact(
        stage,
        target_path,
        fact_kind,
        "/World/base" if fact_kind == "filtered_pairs" else True,
    )
    assert stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "unrepresented_physics_facts"
    assert target_path in caught.value.detail
    assert expected_fact in caught.value.detail


def test_reference_only_bare_unmodeled_api_on_selected_endpoint_fails_closed(
    tmp_path: Path,
) -> None:
    source, reference = _write_pair(tmp_path)
    stage = Usd.Stage.Open(str(reference))
    endpoint_path = "/World/door"
    assert stage.GetPrimAtPath(endpoint_path).AddAppliedSchema(
        "PhysicsFilteredPairsAPI"
    )
    assert stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "unrepresented_physics_facts"
    assert endpoint_path in caught.value.detail
    assert "PhysicsFilteredPairsAPI" in caught.value.detail


@pytest.mark.parametrize(
    ("case", "source_target", "reference_target"),
    [
        ("source_only", "/World/base", None),
        ("reference_only", None, "/World/base"),
        ("mismatch", "/World/base", "/World/door"),
        ("exact", "/World/base", "/World/base"),
    ],
)
def test_every_selected_body_descendant_requires_exact_physics_parity(
    tmp_path: Path,
    case: str,
    source_target: str | None,
    reference_target: str | None,
) -> None:
    source, reference = _write_pair(tmp_path)
    descendant_path = "/World/base/nonColliderMetadata"
    for stage_path, target in (
        (source, source_target),
        (reference, reference_target),
    ):
        stage = Usd.Stage.Open(str(stage_path))
        UsdGeom.Scope.Define(stage, descendant_path)
        if target is not None:
            _apply_unmodeled_physics_fact(
                stage,
                descendant_path,
                "filtered_pairs",
                target,
            )
        assert stage.GetRootLayer().Save()

    if case == "exact":
        assert _extract(source, reference).plan.rigid_bodies
        return

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "unrepresented_physics_facts"
    assert descendant_path in caught.value.detail
    assert "PhysicsFilteredPairsAPI" in caught.value.detail


@pytest.mark.parametrize(
    ("case", "source_friction", "reference_friction"),
    [
        ("source_only", 0.2, None),
        ("reference_only", None, 0.2),
        ("mismatch", 0.2, 0.8),
        ("exact", 0.2, 0.2),
    ],
)
def test_direct_physics_material_binding_follows_external_material_parity(
    tmp_path: Path,
    case: str,
    source_friction: float | None,
    reference_friction: float | None,
) -> None:
    source, reference = _write_pair(tmp_path)
    material_path = "/World/Looks/physicsMaterial"
    for stage_path, friction in (
        (source, source_friction),
        (reference, reference_friction),
    ):
        _bind_physics_material(
            stage_path,
            owner_path="/World/base",
            material_path=material_path,
            friction=friction,
        )

    if case == "exact":
        assert _extract(source, reference).plan.rigid_bodies
        return

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "unrepresented_physics_facts"
    assert material_path in caught.value.detail
    assert (
        "PhysicsMaterialAPI" in caught.value.detail
        or "physics:staticFriction" in caught.value.detail
    )


def test_physics_binding_relationship_requires_material_binding_api_parity(
    tmp_path: Path,
) -> None:
    source, reference = _write_pair(tmp_path)
    material_path = "/World/Looks/physicsMaterial"
    for stage_path in (source, reference):
        _bind_physics_material(
            stage_path,
            owner_path="/World/base",
            material_path=material_path,
            friction=0.2,
        )
    stage = Usd.Stage.Open(str(source))
    assert stage.GetPrimAtPath("/World/base").RemoveAppliedSchema("MaterialBindingAPI")
    assert stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "unsupported_physics_material_binding"
    assert "MaterialBindingAPI" in caught.value.detail
    assert "source=False" in caught.value.detail


@pytest.mark.parametrize("matching", [False, True])
def test_physics_binding_strength_metadata_requires_exact_parity(
    tmp_path: Path,
    matching: bool,
) -> None:
    source, reference = _write_pair(tmp_path)
    material_path = "/World/Looks/physicsMaterial"
    for stage_path in (source, reference):
        _bind_physics_material(
            stage_path,
            owner_path="/World/base",
            material_path=material_path,
            friction=0.2,
        )
    for stage_path, strength in (
        (source, UsdShade.Tokens.strongerThanDescendants),
        (
            reference,
            UsdShade.Tokens.strongerThanDescendants
            if matching
            else UsdShade.Tokens.weakerThanDescendants,
        ),
    ):
        stage = Usd.Stage.Open(str(stage_path))
        relationship = stage.GetPrimAtPath("/World/base").GetRelationship(
            "material:binding:physics"
        )
        assert UsdShade.MaterialBindingAPI.SetMaterialBindingStrength(
            relationship,
            strength,
        )
        assert stage.GetRootLayer().Save()

    if matching:
        assert _extract(source, reference).plan.rigid_bodies
        return

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "unrepresented_physics_material_binding"
    assert "material:binding:physics differs" in caught.value.detail


def test_all_purpose_binding_to_physics_material_uses_external_material_parity(
    tmp_path: Path,
) -> None:
    source, reference = _write_pair(tmp_path)
    material_path = "/World/Looks/fallbackPhysicsMaterial"
    _bind_physics_material(
        source,
        owner_path="/World/base",
        material_path=material_path,
        friction=0.2,
        all_purpose=True,
    )
    _bind_physics_material(
        reference,
        owner_path="/World/base",
        material_path=material_path,
        friction=0.8,
        all_purpose=True,
    )

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "unrepresented_physics_facts"
    assert "physics:staticFriction" in caught.value.detail


def test_malformed_all_purpose_binding_to_physics_material_fails_closed(
    tmp_path: Path,
) -> None:
    source, reference = _write_pair(tmp_path)
    material_path = "/World/Looks/fallbackPhysicsMaterial"
    for stage_path in (source, reference):
        stage = Usd.Stage.Open(str(stage_path))
        UsdGeom.Scope.Define(stage, "/World/Looks")
        material = UsdShade.Material.Define(stage, material_path)
        UsdPhysics.MaterialAPI.Apply(material.GetPrim()).CreateStaticFrictionAttr(0.2)
        assert stage.GetRootLayer().Save()
    stage = Usd.Stage.Open(str(reference))
    owner = stage.GetPrimAtPath("/World/base")
    UsdShade.MaterialBindingAPI.Apply(owner)
    relationship = owner.CreateRelationship("material:binding", custom=False)
    assert relationship.SetTargets(["/World/door", material_path])
    assert stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "unsupported_physics_material_binding"
    assert "unsupported shape or targets" in caught.value.detail


def test_collection_physics_material_binding_and_definition_exact_parity(
    tmp_path: Path,
) -> None:
    source, reference = _write_pair(tmp_path)
    for stage_path in (source, reference):
        _bind_physics_material(
            stage_path,
            owner_path="/World",
            material_path="/World/Looks/collectionPhysicsMaterial",
            friction=0.4,
            collection_name="rigidBodies",
            collection_members=("/World/base",),
        )

    assert _extract(source, reference).plan.rigid_bodies


def test_collection_physics_material_membership_must_preexist_exactly(
    tmp_path: Path,
) -> None:
    source, reference = _write_pair(tmp_path)
    _bind_physics_material(
        source,
        owner_path="/World",
        material_path="/World/Looks/collectionPhysicsMaterial",
        friction=0.4,
        collection_name="rigidBodies",
        collection_members=("/World/base",),
    )
    _bind_physics_material(
        reference,
        owner_path="/World",
        material_path="/World/Looks/collectionPhysicsMaterial",
        friction=0.4,
        collection_name="rigidBodies",
        collection_members=("/World/door",),
    )

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "unrepresented_physics_material_binding"
    assert "collection definition differs" in caught.value.detail
    assert "collection:rigidBodies:includes" in caught.value.detail


def test_malformed_collection_physics_material_binding_fails_closed(
    tmp_path: Path,
) -> None:
    source, reference = _write_pair(tmp_path)
    material_path = "/World/Looks/physicsMaterial"
    for stage_path in (source, reference):
        stage = Usd.Stage.Open(str(stage_path))
        UsdGeom.Scope.Define(stage, "/World/Looks")
        material = UsdShade.Material.Define(stage, material_path)
        UsdPhysics.MaterialAPI.Apply(material.GetPrim()).CreateStaticFrictionAttr(0.4)
        assert stage.GetRootLayer().Save()
    stage = Usd.Stage.Open(str(reference))
    relationship = stage.GetPrimAtPath("/World/base").CreateRelationship(
        "material:binding:collection:physics:broken",
        custom=False,
    )
    assert relationship.SetTargets([material_path])
    assert stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "unsupported_physics_material_binding"
    assert "unsupported shape or targets" in caught.value.detail


def test_collection_physics_binding_precedence_requires_exact_parity(
    tmp_path: Path,
) -> None:
    source, reference = _write_pair(tmp_path)
    orders: dict[Path, list[str]] = {}
    bound_materials: dict[Path, str] = {}
    for stage_path, reverse in ((source, False), (reference, True)):
        stage = Usd.Stage.Open(str(stage_path))
        world = stage.GetPrimAtPath("/World")
        binding_api = UsdShade.MaterialBindingAPI.Apply(world)
        UsdGeom.Scope.Define(stage, "/World/Looks")
        relationship_names = []
        for index, friction in ((1, 0.1), (2, 0.9)):
            material = UsdShade.Material.Define(stage, f"/World/Looks/material{index}")
            UsdPhysics.MaterialAPI.Apply(material.GetPrim()).CreateStaticFrictionAttr(
                friction
            )
            collection = Usd.CollectionAPI.Apply(world, f"collection{index}")
            assert collection.GetIncludesRel().SetTargets(["/World/base"])
            binding_name = f"binding{index}"
            assert binding_api.Bind(
                collection,
                material,
                bindingName=binding_name,
                materialPurpose="physics",
            )
            relationship_names.append(
                f"material:binding:collection:physics:{binding_name}"
            )
        order = list(reversed(relationship_names)) if reverse else relationship_names
        world.SetPropertyOrder(order)
        orders[stage_path] = order
        assert stage.GetRootLayer().Save()
        bound = UsdShade.MaterialBindingAPI(
            stage.GetPrimAtPath("/World/base")
        ).ComputeBoundMaterial("physics")[0]
        bound_materials[stage_path] = str(bound.GetPath())

    assert bound_materials[source] != bound_materials[reference]
    assert orders[source] != orders[reference]
    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "unrepresented_physics_material_binding"
    assert "precedence differs" in caught.value.detail


def test_nested_collection_physics_membership_requires_recursive_parity(
    tmp_path: Path,
) -> None:
    source, reference = _write_pair(tmp_path)
    for stage_path, nested_member in (
        (source, "/World/base"),
        (reference, "/World/door"),
    ):
        stage = Usd.Stage.Open(str(stage_path))
        world = stage.GetPrimAtPath("/World")
        UsdGeom.Scope.Define(stage, "/World/Looks")
        material = UsdShade.Material.Define(stage, "/World/Looks/nestedMaterial")
        UsdPhysics.MaterialAPI.Apply(material.GetPrim()).CreateStaticFrictionAttr(0.2)
        inner = Usd.CollectionAPI.Apply(world, "inner")
        assert inner.GetIncludesRel().SetTargets([nested_member])
        outer = Usd.CollectionAPI.Apply(world, "outer")
        assert outer.GetIncludesRel().SetTargets([inner.GetCollectionPath()])
        assert UsdShade.MaterialBindingAPI.Apply(world).Bind(
            outer,
            material,
            bindingName="outerBinding",
            materialPurpose="physics",
        )
        assert stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "unrepresented_physics_material_binding"
    assert "effective collection membership differs" in caught.value.detail
    assert "/World.collection:inner" in caught.value.detail


def test_physics_binding_target_must_be_a_material_in_both_stages(
    tmp_path: Path,
) -> None:
    source, reference = _write_pair(tmp_path)
    material_path = "/World/Looks/physicsMaterial"
    for stage_path, use_material in ((source, True), (reference, False)):
        stage = Usd.Stage.Open(str(stage_path))
        UsdGeom.Scope.Define(stage, "/World/Looks")
        target = (
            UsdShade.Material.Define(stage, material_path).GetPrim()
            if use_material
            else UsdGeom.Scope.Define(stage, material_path).GetPrim()
        )
        UsdPhysics.MaterialAPI.Apply(target).CreateStaticFrictionAttr(0.2)
        owner = stage.GetPrimAtPath("/World/base")
        UsdShade.MaterialBindingAPI.Apply(owner)
        relationship = owner.CreateRelationship(
            "material:binding:physics",
            custom=False,
        )
        assert relationship.SetTargets([material_path])
        assert stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "unsupported_physics_material_binding"
    assert "must be a UsdShade.Material" in caught.value.detail
    assert "Scope" in caught.value.detail


def test_selected_joint_rejects_arbitrary_unmodeled_physics_property(
    tmp_path: Path,
) -> None:
    source, reference = _write_pair(tmp_path)
    stage = Usd.Stage.Open(str(reference))
    joint_path = "/World/Joints/door"
    attribute = stage.GetPrimAtPath(joint_path).CreateAttribute(
        "physics:customUnmodeled",
        Sdf.ValueTypeNames.Float,
        custom=True,
    )
    assert attribute.Set(3.0)
    assert stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "unsupported_optional_schema"
    assert joint_path in caught.value.detail
    assert "physics:customUnmodeled" in caught.value.detail


def test_unmodeled_physics_attribute_metadata_requires_exact_parity(
    tmp_path: Path,
) -> None:
    source, reference = _write_pair(tmp_path)
    for stage_path, documentation in (
        (source, "source metadata"),
        (reference, "reference metadata"),
    ):
        stage = Usd.Stage.Open(str(stage_path))
        _apply_unmodeled_physics_fact(
            stage,
            "/World/base",
            "physx_rigid_body",
            True,
        )
        attribute = stage.GetPrimAtPath("/World/base").GetAttribute(
            "physxRigidBody:disableGravity"
        )
        attribute.SetDocumentation(documentation)
        assert stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "unrepresented_physics_facts"
    assert "physxRigidBody:disableGravity" in caught.value.detail
    assert "mismatched_properties" in caught.value.detail


def test_modeled_physics_attribute_metadata_requires_exact_source_parity(
    tmp_path: Path,
) -> None:
    source, reference = _write_pair(tmp_path)
    for stage_path, documentation in (
        (source, "source metadata"),
        (reference, "reference metadata"),
    ):
        stage = Usd.Stage.Open(str(stage_path))
        attribute = UsdPhysics.RigidBodyAPI.Apply(
            stage.GetPrimAtPath("/World/base")
        ).CreateRigidBodyEnabledAttr(True)
        attribute.SetDocumentation(documentation)
        assert stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "source_physics_not_in_reference"
    assert "physics:rigidBodyEnabled" in caught.value.detail


@pytest.mark.parametrize(
    ("fact_kind", "target_path", "source_value", "reference_value", "property_name"),
    [
        (
            "filtered_pairs",
            "/World",
            "/World/base",
            "/World/door",
            "physics:filteredPairs",
        ),
        (
            "physx_rigid_body",
            "/World/base/unmodeledCollider",
            True,
            False,
            "physxRigidBody:disableGravity",
        ),
    ],
)
def test_unmodeled_physics_property_and_relationship_values_require_exact_parity(
    tmp_path: Path,
    fact_kind: str,
    target_path: str,
    source_value: Any,
    reference_value: Any,
    property_name: str,
) -> None:
    source, reference = _write_pair(tmp_path)
    for stage_path, value in (
        (source, source_value),
        (reference, reference_value),
    ):
        stage = Usd.Stage.Open(str(stage_path))
        if fact_kind == "physx_rigid_body":
            collider = UsdGeom.Cube.Define(stage, target_path).GetPrim()
            if stage_path == reference:
                UsdPhysics.CollisionAPI.Apply(collider).CreateCollisionEnabledAttr(True)
        _apply_unmodeled_physics_fact(
            stage,
            target_path,
            fact_kind,
            value,
        )
        assert stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "unrepresented_physics_facts"
    assert property_name in caught.value.detail
    assert "mismatched_properties" in caught.value.detail


@pytest.mark.parametrize(
    ("fact_kind", "target_path", "property_name"),
    [
        (
            "drive_namespace",
            "/World/door",
            "drive:angular:physics:stiffness",
        ),
        (
            "state_namespace",
            "/World/base/namespacedCollider",
            "state:angular:physics:position",
        ),
    ],
)
def test_namespaced_unmodeled_physics_properties_require_exact_parity(
    tmp_path: Path,
    fact_kind: str,
    target_path: str,
    property_name: str,
) -> None:
    source, reference = _write_pair(tmp_path)
    for stage_path, value in ((source, 1.0), (reference, 2.0)):
        stage = Usd.Stage.Open(str(stage_path))
        if fact_kind == "state_namespace":
            collider = UsdGeom.Cube.Define(stage, target_path).GetPrim()
            if stage_path == reference:
                UsdPhysics.CollisionAPI.Apply(collider).CreateCollisionEnabledAttr(True)
        _apply_unmodeled_physics_fact(
            stage,
            target_path,
            fact_kind,
            value,
        )
        assert stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "unrepresented_physics_facts"
    assert property_name in caught.value.detail
    assert "mismatched_properties" in caught.value.detail


def test_exact_preexisting_namespaced_physics_properties_are_replay_safe(
    tmp_path: Path,
) -> None:
    source, reference = _write_pair(tmp_path)
    collider_path = "/World/base/namespacedCollider"
    for stage_path in (source, reference):
        stage = Usd.Stage.Open(str(stage_path))
        collider = UsdGeom.Cube.Define(stage, collider_path).GetPrim()
        if stage_path == reference:
            UsdPhysics.CollisionAPI.Apply(collider).CreateCollisionEnabledAttr(True)
        _apply_unmodeled_physics_fact(
            stage,
            "/World/door",
            "drive_namespace",
            3.0,
        )
        _apply_unmodeled_physics_fact(
            stage,
            collider_path,
            "state_namespace",
            4.0,
        )
        if stage_path == reference:
            state_token = "PhysicsJointStateAPI:angular"
            tokens = rv._applied_schema_tokens(collider)
            collider.SetMetadata(
                "apiSchemas",
                Sdf.TokenListOp.CreateExplicit(
                    [state_token, *(token for token in tokens if token != state_token)]
                ),
            )
            drive_prim = stage.GetPrimAtPath("/World/door")
            drive_token = "PhysicsDriveAPI:angular"
            drive_tokens = rv._applied_schema_tokens(drive_prim)
            drive_prim.SetMetadata(
                "apiSchemas",
                Sdf.TokenListOp.CreateExplicit(
                    [
                        drive_token,
                        *(token for token in drive_tokens if token != drive_token),
                    ]
                ),
            )
        assert stage.GetRootLayer().Save()

    result = _extract(source, reference)

    base = next(
        body for body in result.plan.rigid_bodies if body.prim_path == "/World/base"
    )
    assert collider_path in {collider.prim_path for collider in base.colliders}


def test_exact_preexisting_unmodeled_physics_facts_are_replay_safe(
    tmp_path: Path,
) -> None:
    source, reference = _write_pair(tmp_path)
    collider_path = "/World/base/retainedPhysxCollider"
    for stage_path in (source, reference):
        stage = Usd.Stage.Open(str(stage_path))
        collider = UsdGeom.Cube.Define(stage, collider_path).GetPrim()
        if stage_path == reference:
            UsdPhysics.CollisionAPI.Apply(collider).CreateCollisionEnabledAttr(True)
        _apply_unmodeled_physics_fact(
            stage,
            "/World",
            "filtered_pairs",
            "/World/base",
        )
        _apply_unmodeled_physics_fact(
            stage,
            collider_path,
            "physx_rigid_body",
            False,
        )
        if stage_path == reference:
            retained_token = "PhysxRigidBodyAPI"
            tokens = rv._applied_schema_tokens(collider)
            collider.SetMetadata(
                "apiSchemas",
                Sdf.TokenListOp.CreateExplicit(
                    [
                        retained_token,
                        *(token for token in tokens if token != retained_token),
                    ]
                ),
            )
        assert stage.GetRootLayer().Save()

    result = _extract(source, reference)

    base = next(
        body for body in result.plan.rigid_bodies if body.prim_path == "/World/base"
    )
    assert collider_path in {collider.prim_path for collider in base.colliders}


def test_exact_preexisting_custom_physics_api_and_relationship_are_replay_safe(
    tmp_path: Path,
) -> None:
    source, reference = _write_pair(tmp_path)
    endpoint_path = "/World/door"
    for stage_path in (source, reference):
        stage = Usd.Stage.Open(str(stage_path))
        prim = stage.GetPrimAtPath(endpoint_path)
        assert prim.AddAppliedSchema("PhysicsRetainedCustomAPI")
        relationship = prim.CreateRelationship(
            "physics:retainedCustomTargets",
            custom=True,
        )
        assert relationship.SetTargets(["/World/base"])
        if stage_path == reference:
            retained_token = "PhysicsRetainedCustomAPI"
            tokens = rv._applied_schema_tokens(prim)
            prim.SetMetadata(
                "apiSchemas",
                Sdf.TokenListOp.CreateExplicit(
                    [
                        retained_token,
                        *(token for token in tokens if token != retained_token),
                    ]
                ),
            )
        assert stage.GetRootLayer().Save()

    result = _extract(source, reference)

    assert any(body.prim_path == endpoint_path for body in result.plan.rigid_bodies)


def test_reordered_unmodeled_physics_api_facts_fail_closed(tmp_path: Path) -> None:
    source, reference = _write_pair(tmp_path)
    endpoint_path = "/World/door"
    custom_tokens = ("PhysicsRetainedCustomAAPI", "PhysicsRetainedCustomBAPI")
    for stage_path, ordered_custom in (
        (source, custom_tokens),
        (reference, tuple(reversed(custom_tokens))),
    ):
        stage = Usd.Stage.Open(str(stage_path))
        prim = stage.GetPrimAtPath(endpoint_path)
        for token in custom_tokens:
            assert prim.AddAppliedSchema(token)
        existing = [
            token
            for token in rv._applied_schema_tokens(prim)
            if token not in custom_tokens
        ]
        prim.SetMetadata(
            "apiSchemas",
            Sdf.TokenListOp.CreateExplicit([*ordered_custom, *existing]),
        )
        assert rv._applied_schema_tokens(prim)[:2] == ordered_custom
        assert stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "unrepresented_physics_facts"
    assert "reference_append_apis" in caught.value.detail
    assert "PhysicsRetainedCustomAAPI" in caught.value.detail


@pytest.mark.parametrize("matching_order", [True, False])
def test_source_modeled_api_keeps_relative_order_around_reference_only_modeled_api(
    tmp_path: Path,
    matching_order: bool,
) -> None:
    source, reference = _write_pair(tmp_path)
    endpoint_path = "/World/base"
    custom_token = "PhysicsRetainedOrderedAPI"

    source_stage = Usd.Stage.Open(str(source))
    source_prim = source_stage.GetPrimAtPath(endpoint_path)
    UsdPhysics.RigidBodyAPI.Apply(source_prim)
    assert source_prim.AddAppliedSchema(custom_token)
    source_prim.SetMetadata(
        "apiSchemas",
        Sdf.TokenListOp.CreateExplicit([custom_token, "PhysicsRigidBodyAPI"]),
    )
    assert source_stage.GetRootLayer().Save()

    reference_stage = Usd.Stage.Open(str(reference))
    reference_prim = reference_stage.GetPrimAtPath(endpoint_path)
    assert reference_prim.AddAppliedSchema(custom_token)
    reference_modeled = [
        token
        for token in rv._applied_schema_tokens(reference_prim)
        if token != custom_token
    ]
    reference_order = (
        [custom_token, *reference_modeled]
        if matching_order
        else [*reference_modeled, custom_token]
    )
    reference_prim.SetMetadata(
        "apiSchemas",
        Sdf.TokenListOp.CreateExplicit(reference_order),
    )
    assert reference_stage.GetRootLayer().Save()

    if matching_order:
        assert _extract(source, reference).plan.rigid_bodies
        return

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "unrepresented_physics_facts"
    assert "PhysicsRetainedOrderedAPI" in caught.value.detail
    assert "reference_append_apis" in caught.value.detail


def test_reference_only_modeled_api_cannot_interleave_source_api_prefix(
    tmp_path: Path,
) -> None:
    source, reference = _write_pair(tmp_path)
    endpoint_path = "/World/base"
    custom_token = "PhysicsRetainedInterleaveAPI"

    source_stage = Usd.Stage.Open(str(source))
    source_prim = source_stage.GetPrimAtPath(endpoint_path)
    UsdPhysics.RigidBodyAPI.Apply(source_prim)
    assert source_prim.AddAppliedSchema(custom_token)
    source_prim.SetMetadata(
        "apiSchemas",
        Sdf.TokenListOp.CreateExplicit([custom_token, "PhysicsRigidBodyAPI"]),
    )
    assert source_stage.GetRootLayer().Save()

    reference_stage = Usd.Stage.Open(str(reference))
    reference_prim = reference_stage.GetPrimAtPath(endpoint_path)
    assert reference_prim.AddAppliedSchema(custom_token)
    reference_modeled = [
        token
        for token in rv._applied_schema_tokens(reference_prim)
        if token != custom_token
    ]
    reference_only = next(
        token for token in reference_modeled if token != "PhysicsRigidBodyAPI"
    )
    remaining = [
        token
        for token in reference_modeled
        if token not in {"PhysicsRigidBodyAPI", reference_only}
    ]
    reference_prim.SetMetadata(
        "apiSchemas",
        Sdf.TokenListOp.CreateExplicit(
            [
                custom_token,
                reference_only,
                "PhysicsRigidBodyAPI",
                *remaining,
            ]
        ),
    )
    assert reference_stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "unrepresented_physics_facts"
    assert "source_prefix_matches=False" in caught.value.detail
    assert reference_only in caught.value.detail


@pytest.mark.parametrize("canonical_order", [True, False])
def test_reference_only_modeled_apis_require_planned_application_order(
    tmp_path: Path,
    canonical_order: bool,
) -> None:
    source, reference = _write_pair(tmp_path)
    endpoint_path = "/World/base"
    reference_stage = Usd.Stage.Open(str(reference))
    reference_prim = reference_stage.GetPrimAtPath(endpoint_path)
    UsdPhysics.MeshCollisionAPI.Apply(reference_prim)
    canonical_tokens = [
        "PhysicsRigidBodyAPI",
        "PhysicsMassAPI",
        "PhysicsCollisionAPI",
        "PhysicsMeshCollisionAPI",
        "PhysicsArticulationRootAPI",
    ]
    reference_prim.SetMetadata(
        "apiSchemas",
        Sdf.TokenListOp.CreateExplicit(
            canonical_tokens if canonical_order else list(reversed(canonical_tokens))
        ),
    )
    assert reference_stage.GetRootLayer().Save()

    if canonical_order:
        assert _extract(source, reference).plan.rigid_bodies
        return

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "unrepresented_physics_facts"
    assert "expected_reference_append_apis" in caught.value.detail
    assert "PhysicsMassAPI" in caught.value.detail


def test_complete_descendant_collider_mass_lifts_in_identity_owner_frame(
    tmp_path: Path,
) -> None:
    source, reference = _write_pair(tmp_path)
    child_path = "/World/drawer/massCollider"
    _define_mass_contributor(source, child_path, author_physics=False)
    _define_mass_contributor(reference, child_path, author_physics=True)

    result = _extract(source, reference)

    body = next(
        item for item in result.plan.rigid_bodies if item.prim_path == "/World/drawer"
    )
    assert body.mass is not None
    assert body.mass.mass_kg == pytest.approx(2.0)
    assert body.mass.center_of_mass_m == pytest.approx((0.01, 0.02, 0.03))
    assert body.mass.diagonal_inertia_kg_m2 == pytest.approx((0.01, 0.015, 0.02))
    assert body.mass.principal_axes == (1.0, 0.0, 0.0, 0.0)
    assert body.mass.provenance.prim_path == child_path
    assert set(body.mass.provenance.properties) == {
        "physics:mass",
        "physics:centerOfMass",
        "physics:diagonalInertia",
        "physics:principalAxes",
    }
    assert body.mass.provenance.derivation is not None
    assert "centerOfMass_owner_m=R*centerOfMass_contributor_m" in (
        body.mass.provenance.derivation
    )
    assert "stage_mass_and_length_units_to_si" in body.mass.provenance.derivation


def test_descendant_mass_lifts_translation_rotation_and_inertia_frame(
    tmp_path: Path,
) -> None:
    source, reference = _write_pair(tmp_path)
    child_path = "/World/drawer/rotatedMassCollider"
    transform = Gf.Matrix4d(1.0)
    transform.SetRotate(Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), 90.0))
    transform.SetTranslateOnly(Gf.Vec3d(10.0, 20.0, 30.0))
    for path, author_physics in ((source, False), (reference, True)):
        _define_mass_contributor(
            path,
            child_path,
            author_physics=author_physics,
            transform=transform,
            center_of_mass=(1.0, 0.0, 0.0),
            principal_axes=(math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0),
        )

    result = _extract(source, reference)

    body = next(
        item for item in result.plan.rigid_bodies if item.prim_path == "/World/drawer"
    )
    assert body.mass is not None
    assert body.mass.center_of_mass_m == pytest.approx((0.1, 0.21, 0.3))
    assert body.mass.diagonal_inertia_kg_m2 == pytest.approx((0.01, 0.015, 0.02))
    assert body.mass.principal_axes == pytest.approx((0.5, 0.5, 0.5, 0.5))
    assert body.mass.principal_axes[0] > 0.0
    assert body.mass.provenance.derivation is not None
    assert "translation_stage=(10, 20, 30)" in body.mass.provenance.derivation


def test_descendant_mass_lift_uses_gf_row_vector_order_for_transformed_owner(
    tmp_path: Path,
) -> None:
    source, reference = _write_pair(tmp_path)
    owner_path = "/World/ball"
    child_path = f"{owner_path}/massCollider"
    owner_matrix = Gf.Matrix4d(1.0)
    owner_matrix.SetRotate(Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), 37.0))
    owner_matrix.SetTranslateOnly(Gf.Vec3d(11.0, -7.0, 5.0))
    contributor_matrix = Gf.Matrix4d(1.0)
    contributor_matrix.SetRotate(Gf.Rotation(Gf.Vec3d(1.0, 0.0, 0.0), -23.0))
    contributor_matrix.SetTranslateOnly(Gf.Vec3d(2.0, 3.0, -4.0))
    center_stage = (0.25, -0.5, 1.75)
    for stage_path, author_physics in ((source, False), (reference, True)):
        stage = Usd.Stage.Open(str(stage_path))
        owner = UsdGeom.Xformable(stage.GetPrimAtPath(owner_path))
        assert owner.AddTransformOp().Set(owner_matrix)
        assert stage.GetRootLayer().Save()
        _define_mass_contributor(
            stage_path,
            child_path,
            author_physics=author_physics,
            transform=contributor_matrix,
            center_of_mass=center_stage,
            principal_axes=(1.0, 0.0, 0.0, 0.0),
        )

    result = _extract(source, reference)

    body = next(
        item for item in result.plan.rigid_bodies if item.prim_path == owner_path
    )
    assert body.mass is not None
    expected_center_stage = contributor_matrix.Transform(Gf.Vec3d(*center_stage))
    expected_center_m = tuple(float(value) * 0.01 for value in expected_center_stage)
    assert body.mass.center_of_mass_m == pytest.approx(expected_center_m)
    expected_rotation = contributor_matrix.ExtractRotation().GetQuat()
    expected_imaginary = expected_rotation.GetImaginary()
    assert body.mass.principal_axes == pytest.approx(
        (
            float(expected_rotation.GetReal()),
            *(float(value) for value in expected_imaginary),
        )
    )

    stage = Usd.Stage.Open(str(reference))
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    owner_world = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(owner_path))
    contributor_world = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(child_path))
    assert Gf.IsClose(contributor_world, contributor_matrix * owner_world, 1e-12)
    contributor_to_owner = contributor_world * owner_world.GetInverse()
    reversed_order = owner_world.GetInverse() * contributor_world
    assert Gf.IsClose(contributor_to_owner, contributor_matrix, 1e-12)
    assert not Gf.IsClose(reversed_order, contributor_matrix, 1e-12)
    reversed_center = reversed_order.Transform(Gf.Vec3d(*center_stage))
    assert (reversed_center - expected_center_stage).GetLength() > 1.0


def test_descendant_mass_lift_does_not_round_trip_center_through_stage_units(
    tmp_path: Path,
) -> None:
    source, reference = _write_pair(tmp_path)
    child_path = "/World/drawer/largeTranslationMassCollider"
    transform = Gf.Matrix4d(1.0)
    transform.SetTranslateOnly(Gf.Vec3d(100_000_000.0, 0.0, 0.0))
    for path, author_physics in ((source, False), (reference, True)):
        _define_mass_contributor(
            path,
            child_path,
            author_physics=author_physics,
            transform=transform,
            center_of_mass=(0.1, 0.0, 0.0),
            principal_axes=(1.0, 0.0, 0.0, 0.0),
        )

    result = _extract(source, reference)

    body = next(
        item for item in result.plan.rigid_bodies if item.prim_path == "/World/drawer"
    )
    assert body.mass is not None
    assert body.mass.center_of_mass_m is not None
    center_stage = float(Gf.Vec3f(0.1, 0.0, 0.0)[0])
    expected_direct_si = center_stage * 0.01 + 100_000_000.0 * 0.01
    old_si_stage_si = ((center_stage * 0.01) / 0.01 + 100_000_000.0) * 0.01
    assert old_si_stage_si != expected_direct_si
    assert body.mass.center_of_mass_m[0] == expected_direct_si


@pytest.mark.parametrize("transform_kind", ["scale", "shear"])
def test_descendant_mass_lift_rejects_nonrigid_transform(
    tmp_path: Path,
    transform_kind: str,
) -> None:
    source, reference = _write_pair(tmp_path)
    child_path = "/World/drawer/nonrigidMassCollider"
    transform = Gf.Matrix4d(1.0)
    if transform_kind == "scale":
        transform.SetScale(Gf.Vec3d(1.0, 2.0, 1.0))
    else:
        transform.SetRow(0, Gf.Vec4d(1.0, 0.25, 0.0, 0.0))
    for path, author_physics in ((source, False), (reference, True)):
        _define_mass_contributor(
            path,
            child_path,
            author_physics=author_physics,
            transform=transform,
        )

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "descendant_mass_transform_not_rigid"
    assert child_path in caught.value.detail
    assert "scale, shear, or reflection" in caught.value.detail


def test_descendant_mass_lift_rejects_reflection_explicitly(tmp_path: Path) -> None:
    source, reference = _write_pair(tmp_path)
    child_path = "/World/drawer/reflectedMassCollider"
    reflection = Gf.Matrix4d(1.0)
    reflection.SetScale(Gf.Vec3d(-1.0, 1.0, 1.0))
    for path, author_physics in ((source, False), (reference, True)):
        _define_mass_contributor(
            path,
            child_path,
            author_physics=author_physics,
            transform=reflection,
        )

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "descendant_mass_transform_not_rigid"
    assert child_path in caught.value.detail
    assert "reflection" in caught.value.detail
    assert "handedness=-1.0" in caught.value.detail


def test_descendant_mass_lift_rejects_time_sampled_transform(tmp_path: Path) -> None:
    source, reference = _write_pair(tmp_path)
    child_path = "/World/drawer/animatedMassCollider"
    transform = Gf.Matrix4d(1.0)
    for path, author_physics in ((source, False), (reference, True)):
        _define_mass_contributor(
            path,
            child_path,
            author_physics=author_physics,
            transform=transform,
            time_sampled_transform=True,
        )

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "descendant_mass_transform_time_sampled"
    assert child_path in caught.value.detail


def test_descendant_mass_lift_rejects_spline_transform(tmp_path: Path) -> None:
    source, reference = _write_pair(tmp_path)
    child_path = "/World/drawer/splineMassCollider"
    _define_mass_contributor(source, child_path, author_physics=False)
    _define_mass_contributor(reference, child_path, author_physics=True)
    stage = Usd.Stage.Open(str(reference))
    child = UsdGeom.Xformable(stage.GetPrimAtPath(child_path))
    attribute = child.AddRotateXOp().GetAttr()
    spline = Ts.Spline("float")
    knot = Ts.Knot("float")
    knot.SetTime(1.0)
    knot.SetValue(15.0)
    spline.SetKnot(knot)
    assert attribute.SetSpline(spline)
    assert attribute.GetTimeSamples() == []
    assert stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "descendant_mass_transform_time_sampled"
    assert "spline=True" in caught.value.detail
    assert child_path in caught.value.detail


def test_descendant_mass_lift_rejects_spline_authored_mass(tmp_path: Path) -> None:
    source, reference = _write_pair(tmp_path)
    child_path = "/World/drawer/splineMassCollider"
    _define_mass_contributor(source, child_path, author_physics=False)
    _define_mass_contributor(reference, child_path, author_physics=True)
    stage = Usd.Stage.Open(str(reference))
    mass_attribute = UsdPhysics.MassAPI(stage.GetPrimAtPath(child_path)).GetMassAttr()
    spline = Ts.Spline("float")
    knot = Ts.Knot("float")
    knot.SetTime(1.0)
    knot.SetValue(3.0)
    spline.SetKnot(knot)
    assert mass_attribute.Get() == 2.0
    assert mass_attribute.SetSpline(spline)
    assert mass_attribute.HasSpline()
    assert mass_attribute.GetTimeSamples() == []
    assert stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "time_sampled_static_property"
    assert child_path in caught.value.detail
    assert "physics:mass" in caught.value.detail
    assert "spline=True" in caught.value.detail
    assert "raw_splines=" in caught.value.detail


def test_descendant_mass_lift_rejects_reset_below_animated_ancestor(
    tmp_path: Path,
) -> None:
    source, reference = _write_pair(tmp_path)
    owner_path = "/World/ball"
    child_path = f"{owner_path}/resetMassCollider"
    for stage_path, author_physics in ((source, False), (reference, True)):
        _define_mass_contributor(
            stage_path,
            child_path,
            author_physics=author_physics,
        )
        stage = Usd.Stage.Open(str(stage_path))
        rotate_attribute = (
            UsdGeom.Xformable(stage.GetPrimAtPath("/World")).AddRotateZOp().GetAttr()
        )
        assert rotate_attribute.Set(0.0)
        spline = Ts.Spline("float")
        knot = Ts.Knot("float")
        knot.SetTime(1.0)
        knot.SetValue(45.0)
        spline.SetKnot(knot)
        assert rotate_attribute.SetSpline(spline)
        assert rotate_attribute.HasSpline()
        assert rotate_attribute.GetTimeSamples() == []
        contributor = UsdGeom.Xformable(stage.GetPrimAtPath(child_path))
        contributor.SetResetXformStack(True)
        assert contributor.GetResetXformStack()
        assert stage.GetRootLayer().Save()

    stage = Usd.Stage.Open(str(reference))
    owner = stage.GetPrimAtPath(owner_path)
    contributor = stage.GetPrimAtPath(child_path)
    default_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    animated_cache = UsdGeom.XformCache(Usd.TimeCode(1.0))
    default_relative = (
        default_cache.GetLocalToWorldTransform(contributor)
        * default_cache.GetLocalToWorldTransform(owner).GetInverse()
    )
    animated_relative = (
        animated_cache.GetLocalToWorldTransform(contributor)
        * animated_cache.GetLocalToWorldTransform(owner).GetInverse()
    )
    default_axis = default_relative.TransformDir(Gf.Vec3d(1.0, 0.0, 0.0))
    animated_axis = animated_relative.TransformDir(Gf.Vec3d(1.0, 0.0, 0.0))
    assert tuple(default_axis) == pytest.approx((1.0, 0.0, 0.0))
    assert tuple(animated_axis) == pytest.approx((math.sqrt(0.5), -math.sqrt(0.5), 0.0))

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "descendant_mass_reset_xform_unsupported"
    assert child_path in caught.value.detail
    assert owner_path in caught.value.detail


def test_descendant_mass_lift_rejects_instanceable_contributor(
    tmp_path: Path,
) -> None:
    source, reference = _write_pair(tmp_path)
    child_path = "/World/drawer/instanceMassCollider"
    _define_mass_contributor(
        source,
        child_path,
        author_physics=False,
        instanceable=True,
    )
    _define_mass_contributor(
        reference,
        child_path,
        author_physics=True,
        instanceable=True,
    )

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "descendant_mass_instance_unsupported"
    assert child_path in caught.value.detail


def test_descendant_mass_lift_requires_collider_contributor(tmp_path: Path) -> None:
    source, reference = _write_pair(tmp_path)
    child_path = "/World/drawer/nonColliderMass"
    _define_mass_contributor(source, child_path, author_physics=False)
    _define_mass_contributor(
        reference,
        child_path,
        author_physics=True,
        collision=False,
    )

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "descendant_mass_contributor_not_collider"
    assert child_path in caught.value.detail


@pytest.mark.parametrize(
    "missing",
    ["mass", "centerOfMass", "diagonalInertia", "principalAxes"],
)
def test_descendant_mass_lift_rejects_every_incomplete_frame(
    tmp_path: Path,
    missing: str,
) -> None:
    source, reference = _write_pair(tmp_path)
    child_path = "/World/drawer/incompleteMassCollider"
    _define_mass_contributor(source, child_path, author_physics=False)
    _define_mass_contributor(
        reference,
        child_path,
        author_physics=True,
        missing=missing,
    )

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "incomplete_descendant_mass_properties"
    assert f"physics:{missing}" in caught.value.detail


def test_body_and_descendant_mass_evidence_conflict_fails_closed(
    tmp_path: Path,
) -> None:
    source, reference = _write_pair(tmp_path)
    child_path = "/World/base/conflictingMassCollider"
    _define_mass_contributor(source, child_path, author_physics=False)
    _define_mass_contributor(reference, child_path, author_physics=True)

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "body_descendant_mass_conflict"
    assert "/World/base" in caught.value.detail
    assert child_path in caught.value.detail
    assert "'replay_status': 'reference_only'" in caught.value.detail


def test_incomplete_body_and_descendant_mass_evidence_conflict_fails_closed(
    tmp_path: Path,
) -> None:
    source, reference = _write_pair(tmp_path)
    child_path = "/World/drawer/conflictingMassCollider"
    stage = Usd.Stage.Open(str(reference))
    UsdPhysics.MassAPI.Apply(stage.GetPrimAtPath("/World/drawer"))
    assert stage.GetRootLayer().Save()
    _define_mass_contributor(source, child_path, author_physics=False)
    _define_mass_contributor(reference, child_path, author_physics=True)

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "body_descendant_mass_conflict"
    assert "PhysicsMassAPI" in caught.value.detail
    assert child_path in caught.value.detail


def test_multiple_descendant_mass_contributors_fail_closed(tmp_path: Path) -> None:
    source, reference = _write_pair(tmp_path)
    child_paths = (
        "/World/drawer/massColliderA",
        "/World/drawer/massColliderB",
    )
    for child_path in child_paths:
        _define_mass_contributor(source, child_path, author_physics=False)
        _define_mass_contributor(reference, child_path, author_physics=True)

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "multiple_descendant_mass_contributors"
    assert all(path in caught.value.detail for path in child_paths)


def test_matching_preexisting_descendant_mass_is_not_duplicated(
    tmp_path: Path,
) -> None:
    source, reference = _write_pair(tmp_path)
    child_path = "/World/drawer/preexistingMassCollider"
    _define_mass_contributor(source, child_path, author_physics=True)
    _define_mass_contributor(reference, child_path, author_physics=True)

    result = _extract(source, reference)

    body = next(
        item for item in result.plan.rigid_bodies if item.prim_path == "/World/drawer"
    )
    assert body.mass is None


def test_lifted_mass_contributor_validity_guard_is_deterministic(
    tmp_path: Path,
) -> None:
    source, reference = _write_pair(tmp_path)
    child_path = "/World/drawer/massCollider"
    _define_mass_contributor(source, child_path, author_physics=False)
    _define_mass_contributor(reference, child_path, author_physics=True)
    result = _extract(source, reference)
    source_stage = Usd.Stage.Open(str(source))
    reference_stage = Usd.Stage.Open(str(reference))
    assert reference_stage.RemovePrim(child_path)

    with pytest.raises(JointRiggerContractError) as caught:
        rv._require_unmodeled_physics_facts_preexisting(
            source_stage,
            reference_stage=reference_stage,
            body_paths={body.prim_path for body in result.plan.rigid_bodies},
            rigid_bodies=result.plan.rigid_bodies,
            articulation_root=result.plan.articulation_root,
            UsdPhysics=UsdPhysics,
        )

    assert caught.value.code == "lifted_mass_contributor_missing"
    assert caught.value.detail == (
        f"planned lifted mass contributor {child_path} does not resolve to a valid "
        "reference prim"
    )


def test_changed_preexisting_descendant_mass_fails_with_complete_evidence(
    tmp_path: Path,
) -> None:
    source, reference = _write_pair(tmp_path)
    child_path = "/World/drawer/changedMassCollider"
    _define_mass_contributor(source, child_path, author_physics=True)
    _define_mass_contributor(reference, child_path, author_physics=True)
    stage = Usd.Stage.Open(str(reference))
    UsdPhysics.MassAPI(stage.GetPrimAtPath(child_path)).GetMassAttr().Set(3.0)
    assert stage.GetRootLayer().Save()

    source_stage = Usd.Stage.Open(str(source))
    reference_stage = Usd.Stage.Open(str(reference))
    owner_path = "/World/drawer"
    with pytest.raises(JointRiggerContractError) as caught:
        rv._extract_body_mass(
            reference_stage.GetPrimAtPath(owner_path),
            owned_prims=(reference_stage.GetPrimAtPath(child_path),),
            source_stage=source_stage,
            body_path=owner_path,
            reference_identity=_identity(reference),
            kilograms_per_unit=float(
                UsdPhysics.GetStageKilogramsPerUnit(reference_stage)
            ),
            meters_per_unit=float(UsdGeom.GetStageMetersPerUnit(reference_stage)),
            UsdPhysics=UsdPhysics,
        )

    assert caught.value.code == "descendant_mass_source_conflict"
    assert child_path in caught.value.detail
    assert "'replay_status': 'source_conflict'" in caught.value.detail
    assert "'reference_evidence':" in caught.value.detail
    assert "'source_evidence':" in caught.value.detail

    with pytest.raises(JointRiggerContractError) as preflight:
        _extract(source, reference)
    assert preflight.value.code == "source_physics_not_in_reference"
    assert f"{child_path} physics:mass" in preflight.value.detail


def test_reference_only_and_replay_preserved_descendant_mass_cannot_mix(
    tmp_path: Path,
) -> None:
    source, reference = _write_pair(tmp_path)
    preserved_path = "/World/drawer/preservedMassCollider"
    reference_only_path = "/World/drawer/referenceOnlyMassCollider"
    _define_mass_contributor(source, preserved_path, author_physics=True)
    _define_mass_contributor(reference, preserved_path, author_physics=True)
    _define_mass_contributor(source, reference_only_path, author_physics=False)
    _define_mass_contributor(reference, reference_only_path, author_physics=True)

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "multiple_descendant_mass_contributors"
    assert preserved_path in caught.value.detail
    assert reference_only_path in caught.value.detail
    assert "'replay_status': 'matching_preexisting'" in caught.value.detail
    assert "'replay_status': 'reference_only'" in caught.value.detail


@pytest.mark.parametrize(
    ("evidence", "reason_code"),
    [
        ("api_only", "incomplete_descendant_mass_properties"),
        ("density", "unsupported_mass_property"),
        ("property_without_api", "mass_property_without_api"),
    ],
)
def test_descendant_mass_evidence_fails_closed(
    tmp_path: Path,
    evidence: str,
    reason_code: str,
) -> None:
    source, reference = _write_pair(tmp_path)
    child_path = "/World/ball/massChild"
    for path in (source, reference):
        stage = Usd.Stage.Open(str(path))
        child = UsdGeom.Cube.Define(stage, child_path).GetPrim()
        if path == reference:
            if evidence in {"api_only", "density"}:
                mass = UsdPhysics.MassAPI.Apply(child)
                if evidence == "density":
                    mass.CreateDensityAttr(123.0)
            else:
                child.CreateAttribute(
                    "physics:density",
                    Sdf.ValueTypeNames.Float,
                    custom=False,
                ).Set(123.0)
        assert stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == reason_code
    assert child_path in caught.value.detail
    expected_evidence = "MassAPI" if evidence == "api_only" else "physics:density"
    assert expected_evidence in caught.value.detail


def test_matching_preexisting_nested_body_mass_is_replay_safe(
    tmp_path: Path,
) -> None:
    source, reference = _write_pair(tmp_path)
    child_path = "/World/ball/preexistingMassChild"
    for stage_path in (source, reference):
        stage = Usd.Stage.Open(str(stage_path))
        child = UsdGeom.Cube.Define(stage, child_path).GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(child).CreateRigidBodyEnabledAttr(True)
        UsdPhysics.MassAPI.Apply(child).CreateMassAttr(1.5)
        assert stage.GetRootLayer().Save()

    result = _extract(source, reference)

    assert all(body.prim_path != child_path for body in result.plan.rigid_bodies)


def test_reference_only_nested_body_mass_is_not_silently_ignored(
    tmp_path: Path,
) -> None:
    source, reference = _write_pair(tmp_path)
    child_path = "/World/ball/referenceOnlyMassChild"
    for stage_path in (source, reference):
        stage = Usd.Stage.Open(str(stage_path))
        child = UsdGeom.Cube.Define(stage, child_path).GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(child).CreateRigidBodyEnabledAttr(True)
        if stage_path == reference:
            UsdPhysics.MassAPI.Apply(child).CreateMassAttr(1.5)
        assert stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "unrepresented_descendant_rigid_body"
    assert child_path in caught.value.detail
    assert "PhysicsMassAPI" in caught.value.detail


def test_descendant_mass_below_nonrigid_endpoint_fails_closed(
    tmp_path: Path,
) -> None:
    source, reference = _write_pair(tmp_path)
    child_path = "/World/drawer/massChild"
    for stage_path in (source, reference):
        stage = Usd.Stage.Open(str(stage_path))
        UsdGeom.Cube.Define(stage, child_path)
        if stage_path == reference:
            endpoint = stage.GetPrimAtPath("/World/drawer")
            _remove_rigid_body_schema_and_properties(endpoint)
            _remove_collision_schema_and_properties(endpoint)
            mass = UsdPhysics.MassAPI.Apply(stage.GetPrimAtPath(child_path))
            mass.CreateMassAttr(1.0)
            mass.CreateDiagonalInertiaAttr(Gf.Vec3f(1.0))
        assert stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "unsupported_descendant_mass"
    assert child_path in caught.value.detail
    assert "/World/drawer" in caught.value.detail


def test_rigid_body_mass_and_articulation_optional_paths(tmp_path: Path) -> None:
    source, reference = _write_pair(tmp_path / "optional")
    stage = Usd.Stage.Open(str(reference))
    door = stage.GetPrimAtPath("/World/door")
    _remove_rigid_body_schema_and_properties(door)
    _remove_collision_schema_and_properties(door)
    base_mass = UsdPhysics.MassAPI(stage.GetPrimAtPath("/World/base"))
    base_mass.CreateCenterOfMassAttr(Gf.Vec3f(25.0, 0.0, 0.0))
    base_mass.CreatePrincipalAxesAttr(Gf.Quatf(-1.0, Gf.Vec3f(0.0)))
    stage.GetPrimAtPath("/World/base").RemoveAppliedSchema("PhysicsArticulationRootAPI")
    stage.GetRootLayer().Save()
    result = _extract(source, reference)
    body_paths = {body.prim_path for body in result.plan.rigid_bodies}
    assert "/World/door" not in body_paths
    assert result.plan.articulation_root is None
    base = next(
        body for body in result.plan.rigid_bodies if body.prim_path == "/World/base"
    )
    assert base.mass is not None
    assert base.mass.center_of_mass_m == (0.25, 0.0, 0.0)
    assert base.mass.principal_axes == (1.0, 0.0, 0.0, 0.0)

    source, reference = _write_pair(tmp_path / "partial")
    stage = Usd.Stage.Open(str(reference))
    partial = UsdPhysics.MassAPI.Apply(stage.GetPrimAtPath("/World/door"))
    partial.CreateMassAttr(1.0)
    stage.GetRootLayer().Save()
    with pytest.raises(JointRiggerContractError) as incomplete:
        _extract(source, reference)
    assert incomplete.value.code == "incomplete_optional_schema"


def test_owner_center_of_mass_without_mass_inertia_pair_fails_closed(
    tmp_path: Path,
) -> None:
    source, reference = _write_pair(tmp_path)
    body_path = "/World/drawer"
    stage = Usd.Stage.Open(str(reference))
    mass = UsdPhysics.MassAPI.Apply(stage.GetPrimAtPath(body_path))
    mass.CreateCenterOfMassAttr(Gf.Vec3f(1.0, 2.0, 3.0))
    assert stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "incomplete_optional_schema"
    assert caught.value.detail == (
        f"{body_path} has incomplete MassAPI; "
        "required=['physics:mass', 'physics:diagonalInertia']; "
        "authored=['physics:centerOfMass']; "
        "missing=['physics:mass', 'physics:diagonalInertia']"
    )


@pytest.mark.parametrize(
    ("body_path", "property_name", "value_type", "value"),
    [
        (
            "/World/base",
            "physics:density",
            Sdf.ValueTypeNames.Float,
            1000.0,
        ),
    ],
)
def test_unrepresented_mass_properties_fail_closed(
    tmp_path: Path,
    body_path: str,
    property_name: str,
    value_type: Any,
    value: Any,
) -> None:
    source, reference = _write_pair(tmp_path)
    stage = Usd.Stage.Open(str(reference))
    prim = stage.GetPrimAtPath(body_path)
    UsdPhysics.MassAPI.Apply(prim)
    prim.CreateAttribute(
        property_name,
        value_type,
        custom=False,
    ).Set(value)
    assert stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "unsupported_mass_property"
    assert property_name in caught.value.detail


def test_invalid_mass_units_inertia_shape_and_multiple_roots(tmp_path: Path) -> None:
    source, reference = _write_pair(tmp_path / "units")
    stage = Usd.Stage.Open(str(reference))
    UsdPhysics.SetStageKilogramsPerUnit(stage, 0.0)
    stage.GetRootLayer().Save()
    with pytest.raises(JointRiggerContractError) as units:
        _extract(
            source,
            reference,
            joint_paths=("/World/Joints/door",),
            allowed_omitted_joint_types=("prismatic", "spherical"),
        )
    assert units.value.code == "invalid_stage_units"

    class Attribute:
        def __init__(self, name: str, value: Any, *, authored: bool = True) -> None:
            self.name = name
            self.value = value
            self.authored = authored

        def HasAuthoredValueOpinion(self) -> bool:
            return self.authored

        def HasAuthoredConnections(self) -> bool:
            return False

        def Get(self) -> Any:
            return self.value

        def GetTimeSamples(self) -> list[float]:
            return []

        def GetName(self) -> str:
            return self.name

    class MassAPI:
        def GetMassAttr(self) -> Attribute:
            return Attribute("physics:mass", 1.0)

        def GetDiagonalInertiaAttr(self) -> Attribute:
            return Attribute("physics:diagonalInertia", (1.0, 2.0))

        def GetPrincipalAxesAttr(self) -> Attribute:
            return Attribute("physics:principalAxes", Gf.Quatf())

        def GetCenterOfMassAttr(self) -> Attribute:
            return Attribute("physics:centerOfMass", None, authored=False)

        def GetDensityAttr(self) -> Attribute:
            return Attribute("physics:density", None, authored=False)

    class Physics:
        MassAPI = staticmethod(lambda prim: MassAPI())

    class Prim:
        def HasAPI(self, schema: Any) -> bool:
            return True

        def GetAttribute(self, name: str) -> Attribute:
            return Attribute(name, None, authored=False)

    with pytest.raises(JointRiggerContractError) as defensive_units:
        rv._extract_mass(
            Prim(),
            body_path="/World/body",
            reference_identity=_identity(reference),
            kilograms_per_unit=0.0,
            meters_per_unit=1.0,
            UsdPhysics=Physics,
        )
    assert defensive_units.value.code == "invalid_stage_units"

    with pytest.raises(JointRiggerContractError) as inertia:
        rv._extract_mass(
            Prim(),
            body_path="/World/body",
            reference_identity=_identity(reference),
            kilograms_per_unit=1.0,
            meters_per_unit=1.0,
            UsdPhysics=Physics,
        )
    assert inertia.value.code == "invalid_mass_properties"

    source, reference = _write_pair(tmp_path / "roots")
    stage = Usd.Stage.Open(str(reference))
    UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath("/World/drawer"))
    stage.GetRootLayer().Save()
    with pytest.raises(JointRiggerContractError) as roots:
        _extract(source, reference)
    assert roots.value.code == "contradictory_articulation_roots"


def test_mesh_collider_approximation_is_preserved_or_rejected(tmp_path: Path) -> None:
    source, reference = _write_pair(tmp_path / "valid")
    stage = Usd.Stage.Open(str(reference))
    prim = stage.GetPrimAtPath("/World/base")
    mesh_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
    mesh_api.CreateApproximationAttr("convexHull")
    prim.SetMetadata(
        "apiSchemas",
        Sdf.TokenListOp.CreateExplicit(
            [
                "PhysicsRigidBodyAPI",
                "PhysicsMassAPI",
                "PhysicsCollisionAPI",
                "PhysicsMeshCollisionAPI",
                "PhysicsArticulationRootAPI",
            ]
        ),
    )
    stage.GetRootLayer().Save()
    result = _extract(source, reference)
    base = next(
        body for body in result.plan.rigid_bodies if body.prim_path == "/World/base"
    )
    assert base.colliders[0].has_mesh_collision_api
    assert base.colliders[0].mesh_approximation == "convexHull"
    assert base.colliders[0].provenance.properties == (
        "PhysicsCollisionAPI",
        "PhysicsMeshCollisionAPI",
        "physics:approximation",
    )

    source, reference = _write_pair(tmp_path / "invalid")
    stage = Usd.Stage.Open(str(reference))
    prim = stage.GetPrimAtPath("/World/base")
    mesh_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
    mesh_api.CreateApproximationAttr("unsupported")
    prim.SetMetadata(
        "apiSchemas",
        Sdf.TokenListOp.CreateExplicit(
            [
                "PhysicsRigidBodyAPI",
                "PhysicsMassAPI",
                "PhysicsCollisionAPI",
                "PhysicsMeshCollisionAPI",
                "PhysicsArticulationRootAPI",
            ]
        ),
    )
    stage.GetRootLayer().Save()
    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)
    assert caught.value.code == "unsupported_collider_approximation"


def test_mesh_collision_api_without_authored_approximation_is_preserved(
    tmp_path: Path,
) -> None:
    source, reference = _write_pair(tmp_path)
    stage = Usd.Stage.Open(str(reference))
    prim = stage.GetPrimAtPath("/World/base")
    api = UsdPhysics.MeshCollisionAPI.Apply(prim)
    assert not api.GetApproximationAttr().HasAuthoredValueOpinion()
    prim.SetMetadata(
        "apiSchemas",
        Sdf.TokenListOp.CreateExplicit(
            [
                "PhysicsRigidBodyAPI",
                "PhysicsMassAPI",
                "PhysicsCollisionAPI",
                "PhysicsMeshCollisionAPI",
                "PhysicsArticulationRootAPI",
            ]
        ),
    )
    assert stage.GetRootLayer().Save()

    result = _extract(source, reference)

    base = next(
        body for body in result.plan.rigid_bodies if body.prim_path == "/World/base"
    )
    assert base.colliders[0].mesh_collision_api is True
    assert base.colliders[0].mesh_approximation is None
    assert base.colliders[0].provenance.properties == (
        "PhysicsCollisionAPI",
        "PhysicsMeshCollisionAPI",
    )


def test_joint_collision_enabled_is_not_collider_api_evidence() -> None:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    joint = UsdPhysics.FixedJoint.Define(stage, "/World/nestedJoint")
    joint.CreateCollisionEnabledAttr(False)

    evidence = rv._collision_api_evidence(
        joint.GetPrim(),
        UsdPhysics=UsdPhysics,
    )

    assert evidence == ()


@pytest.mark.parametrize(
    "evidence",
    ["mesh_schema", "collision_enabled", "mesh_approximation"],
)
def test_collision_evidence_without_collision_api_fails_closed(
    tmp_path: Path,
    evidence: str,
) -> None:
    source, reference = _write_pair(tmp_path)
    stage = Usd.Stage.Open(str(reference))
    prim = UsdGeom.Cube.Define(stage, "/World/base/orphanCollider").GetPrim()
    if evidence == "mesh_schema":
        UsdPhysics.MeshCollisionAPI.Apply(prim)
    elif evidence == "collision_enabled":
        prim.CreateAttribute(
            "physics:collisionEnabled",
            Sdf.ValueTypeNames.Bool,
            custom=False,
        ).Set(True)
    else:
        prim.CreateAttribute(
            "physics:approximation",
            Sdf.ValueTypeNames.Token,
            custom=False,
        ).Set("convexHull")
    assert not prim.HasAPI(UsdPhysics.CollisionAPI)
    assert stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "collision_evidence_without_api"
    assert "/World/base/orphanCollider" in caught.value.detail


def test_mesh_approximation_requires_mesh_collision_api(tmp_path: Path) -> None:
    source, reference = _write_pair(tmp_path)
    stage = Usd.Stage.Open(str(reference))
    prim = UsdGeom.Cube.Define(stage, "/World/base/collider").GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)
    prim.CreateAttribute(
        "physics:approximation",
        Sdf.ValueTypeNames.Token,
        custom=False,
    ).Set("convexHull")
    assert not prim.HasAPI(UsdPhysics.MeshCollisionAPI)
    assert stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "mesh_collision_property_without_api"


def test_source_backed_child_collider_is_preserved(tmp_path: Path) -> None:
    source, reference = _write_pair(tmp_path)
    source_stage = Usd.Stage.Open(str(source))
    UsdGeom.Cube.Define(source_stage, "/World/base/childCollider")
    assert source_stage.GetRootLayer().Save()
    reference_stage = Usd.Stage.Open(str(reference))
    collider = UsdGeom.Cube.Define(
        reference_stage,
        "/World/base/childCollider",
    ).GetPrim()
    UsdPhysics.CollisionAPI.Apply(collider).CreateCollisionEnabledAttr(True)
    assert reference_stage.GetRootLayer().Save()

    result = _extract(source, reference)

    base = next(
        body for body in result.plan.rigid_bodies if body.prim_path == "/World/base"
    )
    assert [collider.prim_path for collider in base.colliders] == [
        "/World/base",
        "/World/base/childCollider",
    ]


@pytest.mark.parametrize("geometry_kind", ["cube_size", "mesh_points"])
def test_source_collider_geometry_must_match_reference(
    tmp_path: Path,
    geometry_kind: str,
) -> None:
    source, reference = _write_pair(tmp_path)
    collider_path = "/World/base/geometryCollider"
    for stage_path in (source, reference):
        stage = Usd.Stage.Open(str(stage_path))
        if geometry_kind == "cube_size":
            collider_schema = UsdGeom.Cube.Define(stage, collider_path)
            collider_schema.CreateSizeAttr(1.0 if stage_path == source else 2.0)
        else:
            collider_schema = UsdGeom.Mesh.Define(stage, collider_path)
            offset = 0.0 if stage_path == source else 1.0
            collider_schema.CreatePointsAttr(
                [
                    Gf.Vec3f(offset, 0.0, 0.0),
                    Gf.Vec3f(1.0 + offset, 0.0, 0.0),
                    Gf.Vec3f(offset, 1.0, 0.0),
                ]
            )
        if stage_path == reference:
            UsdPhysics.CollisionAPI.Apply(
                collider_schema.GetPrim()
            ).CreateCollisionEnabledAttr(True)
        assert stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "source_collider_geometry_mismatch"
    assert collider_path in caught.value.detail
    expected_attribute = "size" if geometry_kind == "cube_size" else "points"
    assert expected_attribute in caught.value.detail


@pytest.mark.parametrize("transform_owner", ["collider", "parent"])
def test_source_collider_transform_chain_must_match_reference(
    tmp_path: Path,
    transform_owner: str,
) -> None:
    source, reference = _write_pair(tmp_path)
    parent_path = "/World/base/colliderParent"
    collider_path = f"{parent_path}/collider"
    for stage_path in (source, reference):
        stage = Usd.Stage.Open(str(stage_path))
        parent = UsdGeom.Xform.Define(stage, parent_path)
        collider = UsdGeom.Cube.Define(stage, collider_path)
        xformable = parent if transform_owner == "parent" else collider
        xformable.AddTranslateOp().Set(
            Gf.Vec3d(1.0 if stage_path == source else 2.0, 0.0, 0.0)
        )
        if stage_path == reference:
            UsdPhysics.CollisionAPI.Apply(
                collider.GetPrim()
            ).CreateCollisionEnabledAttr(True)
        assert stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "source_collider_transform_mismatch"
    assert collider_path in caught.value.detail


@pytest.mark.parametrize("time_varying_fact", ["geometry", "transform"])
def test_time_varying_source_backed_collider_fails_closed(
    tmp_path: Path,
    time_varying_fact: str,
) -> None:
    source, reference = _write_pair(tmp_path)
    collider_path = "/World/base/timeVaryingCollider"
    for stage_path in (source, reference):
        stage = Usd.Stage.Open(str(stage_path))
        collider = UsdGeom.Cube.Define(stage, collider_path)
        if time_varying_fact == "geometry":
            attribute = collider.CreateSizeAttr()
        else:
            attribute = collider.AddTranslateOp().GetAttr()
        attribute.Set(1.0 if time_varying_fact == "geometry" else Gf.Vec3d(1.0), 0.0)
        attribute.Set(2.0 if time_varying_fact == "geometry" else Gf.Vec3d(2.0), 1.0)
        if stage_path == reference:
            UsdPhysics.CollisionAPI.Apply(
                collider.GetPrim()
            ).CreateCollisionEnabledAttr(True)
        assert stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    expected_code = (
        "time_varying_collider_geometry"
        if time_varying_fact == "geometry"
        else "time_varying_collider_transform"
    )
    assert caught.value.code == expected_code
    assert collider_path in caught.value.detail


def test_matching_static_collider_geometry_and_transform_are_replay_safe(
    tmp_path: Path,
) -> None:
    source, reference = _write_pair(tmp_path)
    collider_path = "/World/base/matchingCollider"
    for stage_path in (source, reference):
        stage = Usd.Stage.Open(str(stage_path))
        collider = UsdGeom.Cube.Define(stage, collider_path)
        collider.CreateSizeAttr(2.5)
        collider.AddTranslateOp().Set(Gf.Vec3d(3.0, 4.0, 5.0))
        if stage_path == reference:
            UsdPhysics.CollisionAPI.Apply(
                collider.GetPrim()
            ).CreateCollisionEnabledAttr(True)
        assert stage.GetRootLayer().Save()

    result = _extract(source, reference)

    base = next(
        body for body in result.plan.rigid_bodies if body.prim_path == "/World/base"
    )
    assert collider_path in {collider.prim_path for collider in base.colliders}


def test_reference_only_collider_is_rejected(tmp_path: Path) -> None:
    source, reference = _write_pair(tmp_path)
    reference_stage = Usd.Stage.Open(str(reference))
    collider = UsdGeom.Cube.Define(
        reference_stage,
        "/World/base/referenceOnlyCollider",
    ).GetPrim()
    UsdPhysics.CollisionAPI.Apply(collider).CreateCollisionEnabledAttr(True)
    assert reference_stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "collider_not_in_source"
    assert "/World/base/referenceOnlyCollider" in caught.value.detail


@pytest.mark.parametrize(
    ("composed_state", "reason_code"),
    [
        ("inactive", "source_collider_inactive"),
        ("undefined", "source_collider_undefined"),
    ],
)
def test_inactive_or_undefined_source_collider_is_rejected(
    tmp_path: Path,
    composed_state: str,
    reason_code: str,
) -> None:
    source, reference = _write_pair(tmp_path)
    source_stage = Usd.Stage.Open(str(source))
    source_collider = UsdGeom.Cube.Define(
        source_stage,
        "/World/base/sourceStateCollider",
    ).GetPrim()
    if composed_state == "inactive":
        source_collider.SetActive(False)
    else:
        source_collider.SetSpecifier(Sdf.SpecifierOver)
    assert source_stage.GetRootLayer().Save()
    reference_stage = Usd.Stage.Open(str(reference))
    reference_collider = UsdGeom.Cube.Define(
        reference_stage,
        "/World/base/sourceStateCollider",
    ).GetPrim()
    UsdPhysics.CollisionAPI.Apply(reference_collider).CreateCollisionEnabledAttr(True)
    assert reference_stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == reason_code
    assert "/World/base/sourceStateCollider" in caught.value.detail


@pytest.mark.parametrize(
    ("source_schema", "reference_schema"),
    [
        (UsdGeom.Sphere, UsdGeom.Cube),
        (UsdGeom.Xform, UsdGeom.Xform),
    ],
    ids=("different-gprim-types", "not-gprim-geometry"),
)
def test_incompatible_source_collider_type_is_rejected(
    tmp_path: Path,
    source_schema: Any,
    reference_schema: Any,
) -> None:
    source, reference = _write_pair(tmp_path)
    source_stage = Usd.Stage.Open(str(source))
    source_schema.Define(source_stage, "/World/base/incompatibleCollider")
    assert source_stage.GetRootLayer().Save()
    reference_stage = Usd.Stage.Open(str(reference))
    reference_collider = reference_schema.Define(
        reference_stage,
        "/World/base/incompatibleCollider",
    ).GetPrim()
    UsdPhysics.CollisionAPI.Apply(reference_collider).CreateCollisionEnabledAttr(True)
    assert reference_stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "source_collider_type_mismatch"
    assert "/World/base/incompatibleCollider" in caught.value.detail


@pytest.mark.parametrize("composed_state", ["inactive", "undefined"])
def test_inactive_or_undefined_orphan_collision_evidence_is_not_selected(
    tmp_path: Path,
    composed_state: str,
) -> None:
    source, reference = _write_pair(tmp_path)
    stage = Usd.Stage.Open(str(reference))
    prim = UsdGeom.Cube.Define(stage, "/World/base/ignoredOrphan").GetPrim()
    UsdPhysics.MeshCollisionAPI.Apply(prim)
    if composed_state == "inactive":
        prim.SetActive(False)
    else:
        prim.SetSpecifier(Sdf.SpecifierOver)
    assert stage.GetRootLayer().Save()

    result = _extract(source, reference)

    base = next(
        body for body in result.plan.rigid_bodies if body.prim_path == "/World/base"
    )
    assert all(
        collider.prim_path != "/World/base/ignoredOrphan" for collider in base.colliders
    )


def test_gprim_without_collision_schema_or_properties_is_not_a_collider(
    tmp_path: Path,
) -> None:
    source, reference = _write_pair(tmp_path)
    stage = Usd.Stage.Open(str(reference))
    UsdGeom.Cube.Define(stage, "/World/base/visualOnly")
    assert stage.GetRootLayer().Save()

    result = _extract(source, reference)

    base = next(
        body for body in result.plan.rigid_bodies if body.prim_path == "/World/base"
    )
    assert all(
        collider.prim_path != "/World/base/visualOnly" for collider in base.colliders
    )


def test_matching_nested_non_endpoint_body_keeps_ownership_out_of_plan(
    tmp_path: Path,
) -> None:
    source, reference = _write_pair(tmp_path)
    for stage_path in (source, reference):
        stage = Usd.Stage.Open(str(stage_path))
        nested_body = UsdGeom.Xform.Define(stage, "/World/base/nestedBody").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(nested_body).CreateRigidBodyEnabledAttr(True)
        nested_collider = UsdGeom.Cube.Define(
            stage,
            "/World/base/nestedBody/collider",
        ).GetPrim()
        UsdPhysics.CollisionAPI.Apply(nested_collider).CreateCollisionEnabledAttr(True)
        assert stage.GetRootLayer().Save()

    result = _extract(source, reference)

    base = next(
        body for body in result.plan.rigid_bodies if body.prim_path == "/World/base"
    )
    assert [collider.prim_path for collider in base.colliders] == ["/World/base"]


@pytest.mark.parametrize(
    ("drift", "expected_code"),
    [
        ("type", "source_collider_type_mismatch"),
        ("geometry", "source_collider_geometry_mismatch"),
        ("transform", "source_collider_transform_mismatch"),
        ("missing_xform", "source_collider_transform_mismatch"),
    ],
)
def test_retained_nested_body_collider_requires_complete_source_parity(
    tmp_path: Path,
    drift: str,
    expected_code: str,
) -> None:
    source, reference = _write_pair(tmp_path)
    nested_body_path = "/World/base/retainedNestedBody"
    transform_path = f"{nested_body_path}/colliderFrame"
    collider_path = f"{transform_path}/collider"
    for stage_path in (source, reference):
        stage = Usd.Stage.Open(str(stage_path))
        nested_body = UsdGeom.Xform.Define(stage, nested_body_path)
        UsdPhysics.RigidBodyAPI.Apply(nested_body.GetPrim()).CreateRigidBodyEnabledAttr(
            True
        )
        transform = (
            None
            if drift == "missing_xform" and stage_path == source
            else UsdGeom.Xform.Define(stage, transform_path)
        )
        if transform is None:
            stage.DefinePrim(transform_path)
        elif drift == "transform":
            transform.AddTranslateOp().Set(
                Gf.Vec3d(1.0 if stage_path == source else 2.0, 0.0, 0.0)
            )
        collider_schema = (
            UsdGeom.Sphere.Define(stage, collider_path)
            if drift == "type" and stage_path == source
            else UsdGeom.Cube.Define(stage, collider_path)
        )
        if drift == "geometry":
            collider_schema.CreateSizeAttr(1.0 if stage_path == source else 2.0)
        collider = collider_schema.GetPrim()
        UsdPhysics.CollisionAPI.Apply(collider).CreateCollisionEnabledAttr(True)
        assert stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == expected_code
    assert collider_path in caught.value.detail


def test_retained_nested_body_collider_exact_geometry_and_transform_are_safe(
    tmp_path: Path,
) -> None:
    source, reference = _write_pair(tmp_path)
    nested_body_path = "/World/base/retainedNestedBody"
    transform_path = f"{nested_body_path}/colliderFrame"
    collider_path = f"{transform_path}/collider"
    for stage_path in (source, reference):
        stage = Usd.Stage.Open(str(stage_path))
        nested_body = UsdGeom.Xform.Define(stage, nested_body_path)
        nested_body.AddTranslateOp().Set(Gf.Vec3d(1.0, 2.0, 3.0))
        UsdPhysics.RigidBodyAPI.Apply(nested_body.GetPrim()).CreateRigidBodyEnabledAttr(
            True
        )
        transform = UsdGeom.Xform.Define(stage, transform_path)
        transform.AddRotateZOp().Set(30.0)
        collider = UsdGeom.Cube.Define(stage, collider_path)
        collider.CreateSizeAttr(2.5)
        collider.AddTranslateOp().Set(Gf.Vec3d(4.0, 5.0, 6.0))
        UsdPhysics.CollisionAPI.Apply(collider.GetPrim()).CreateCollisionEnabledAttr(
            True
        )
        assert stage.GetRootLayer().Save()

    result = _extract(source, reference)

    assert all(body.prim_path != nested_body_path for body in result.plan.rigid_bodies)
    assert all(
        collider.prim_path != collider_path
        for body in result.plan.rigid_bodies
        for collider in body.colliders
    )


@pytest.mark.parametrize("composed_state", ["inactive", "undefined"])
def test_inactive_or_undefined_colliders_do_not_leak_into_body_plans(
    tmp_path: Path,
    composed_state: str,
) -> None:
    source, reference = _write_pair(tmp_path)
    stage = Usd.Stage.Open(str(reference))
    collider = UsdGeom.Cube.Define(
        stage,
        "/World/base/ignoredCollider",
    ).GetPrim()
    UsdPhysics.CollisionAPI.Apply(collider).CreateCollisionEnabledAttr(True)
    if composed_state == "inactive":
        collider.SetActive(False)
    else:
        collider.SetSpecifier(Sdf.SpecifierOver)
    assert stage.GetRootLayer().Save()

    result = _extract(source, reference)

    base = next(
        body for body in result.plan.rigid_bodies if body.prim_path == "/World/base"
    )
    assert [item.prim_path for item in base.colliders] == ["/World/base"]


def test_ar_asset_hash_uses_bounded_chunked_reads() -> None:
    payload = bytes(range(251)) * ((rv._AR_ASSET_READ_CHUNK_SIZE * 2 // 251) + 17)

    class FakeAsset:
        def __init__(self) -> None:
            self.reads: list[tuple[int, int]] = []

        @staticmethod
        def GetSize() -> int:
            return len(payload)

        def Read(self, count: int, offset: int) -> bytes:
            assert 0 < count <= rv._AR_ASSET_READ_CHUNK_SIZE
            self.reads.append((count, offset))
            return payload[offset : offset + count]

        @staticmethod
        def GetBuffer() -> None:
            raise AssertionError("whole-asset buffers must not be requested")

    asset = FakeAsset()
    digest = rv._ar_asset_sha256(asset, identifier="fake://large-package-entry")

    assert digest == hashlib.sha256(payload).hexdigest()
    assert len(asset.reads) >= 3
    assert [offset for _, offset in asset.reads] == [
        sum(count for count, _ in asset.reads[:index])
        for index in range(len(asset.reads))
    ]
    assert sum(count for count, _ in asset.reads) == len(payload)


@pytest.mark.parametrize("failure", ["short", "exception"])
def test_ar_asset_hash_rejects_incomplete_reads(failure: str) -> None:
    class FailingAsset:
        @staticmethod
        def GetSize() -> int:
            return 16

        @staticmethod
        def Read(count: int, _offset: int) -> bytes:
            if failure == "exception":
                raise OSError("forced resolver read failure")
            return b"x" * (count - 1)

    with pytest.raises(JointRiggerContractError) as caught:
        rv._ar_asset_sha256(FailingAsset(), identifier="fake://broken-entry")

    assert caught.value.code == "dependency_artifact_read_failed"
    expected = "short read" if failure == "short" else "forced resolver read failure"
    assert expected in caught.value.detail


def test_artifact_identity_fallback_and_numeric_defensive_helpers(
    tmp_path: Path,
) -> None:
    source, _ = _write_pair(tmp_path)

    class Layer:
        resolvedPath = ""
        realPath = ""
        identifier = "memory://fallback"
        anonymous = False

        def ExportToString(self) -> str:
            return "#usda 1.0"

    class Stage:
        def GetUsedLayers(self) -> list[Layer]:
            return [Layer()]

    identity = rv._artifact_identity(
        source,
        "fixture://fallback",
        Stage(),
        root_sha256=_sha256(source),
    )
    assert len(identity.dependency_bundle_sha256 or "") == 64

    class FileLayer(Layer):
        realPath = str(source)
        identifier = str(source)

    class FileStage:
        def GetRootLayer(self) -> FileLayer:
            return FileLayer()

        def GetUsedLayers(self) -> list[FileLayer]:
            return [FileLayer()]

    file_identity = rv._artifact_identity(
        source,
        "fixture://file-fallback",
        FileStage(),
        root_sha256=_sha256(source),
    )
    assert len(file_identity.dependency_bundle_sha256 or "") == 64

    class MissingLocatorLayer(Layer):
        identifier = ""

    class MissingLocatorStage:
        def GetUsedLayers(self) -> list[MissingLocatorLayer]:
            return [MissingLocatorLayer()]

    with pytest.raises(JointRiggerContractError) as missing_locator:
        rv._artifact_identity(
            source,
            "fixture://missing-locator",
            MissingLocatorStage(),
            root_sha256=_sha256(source),
        )
    assert missing_locator.value.code == "invalid_artifact_identity"

    class AnonymousLayer:
        anonymous = True

    class AnonymousStage:
        def GetUsedLayers(self) -> list[AnonymousLayer]:
            return [AnonymousLayer()]

    anonymous_identity = rv._artifact_identity(
        source,
        "fixture://anonymous",
        AnonymousStage(),
        root_sha256=_sha256(source),
    )
    assert len(anonymous_identity.dependency_bundle_sha256 or "") == 64
    assert (
        rv._canonical_layer_locator(
            f"{source}[nested/root.usda]",
            artifact_path=source,
        )
        == "$artifact[nested/root.usda]"
    )
    package_with_arguments = Sdf.Layer.CreateIdentifier(
        Ar.JoinPackageRelativePath(str(source), "nested/root.usda"),
        {"format": "usda"},
    )
    expected_package_with_arguments = Sdf.Layer.CreateIdentifier(
        Ar.JoinPackageRelativePath("$artifact", "nested/root.usda"),
        {"format": "usda"},
    )
    assert (
        rv._canonical_layer_locator(
            package_with_arguments,
            artifact_path=source,
        )
        == expected_package_with_arguments
    )
    assert (
        rv._canonical_layer_locator(
            "relative/dependency.usda",
            artifact_path=source,
        )
        == "relative/dependency.usda"
    )
    bracketed_directory = tmp_path / "asset[1]"
    bracketed_directory.mkdir()
    bracketed_root = bracketed_directory / "root.usda"
    bracketed_dependency = bracketed_directory / "dependency.usda"
    assert (
        rv._canonical_layer_locator(
            str(bracketed_dependency),
            artifact_path=bracketed_root,
        )
        == "dependency.usda"
    )

    relative = rv._resolved_usd_dependency(
        "asset",
        source.name,
        artifact_path=source,
        Ar=Ar,
    )
    assert relative.local_path == source.resolve()
    with pytest.raises(JointRiggerContractError) as empty_identifier:
        rv._resolved_usd_dependency(
            "asset",
            "",
            artifact_path=source,
            Ar=Ar,
        )
    assert empty_identifier.value.code == "invalid_artifact_identity"
    with pytest.raises(JointRiggerContractError) as missing_dependency:
        rv._resolved_usd_dependency(
            "asset",
            "missing.bin",
            artifact_path=source,
            Ar=Ar,
        )
    assert missing_dependency.value.code == "dependency_artifact_missing"
    with pytest.raises(JointRiggerContractError) as unopenable_dependency:
        rv._resolved_dependency_sha256(
            rv._ResolvedUsdDependency(
                kind="asset",
                identifier=str(source.with_suffix(".usdz")) + "[missing.bin]",
                lexical_path=None,
                local_path=None,
                package_relative=True,
            )
        )
    assert unopenable_dependency.value.code == "dependency_artifact_missing"

    class MissingStage:
        class Stage:
            @staticmethod
            def Open(path: str) -> None:
                return None

    with pytest.raises(JointRiggerContractError) as missing_stage:
        rv._open_stage(source, Usd=MissingStage, label="source")
    assert missing_stage.value.code == "source_stage_open_failed"

    with pytest.raises(JointRiggerContractError) as not_finite:
        rv._normalized_vector((math.nan, 0.0, 0.0), joint_path="/World/J")
    assert not_finite.value.code == "axis_not_finite"
    with pytest.raises(JointRiggerContractError) as zero:
        rv._normalized_vector((0.0, 0.0, 0.0), joint_path="/World/J")
    assert zero.value.code == "axis_unresolved"
    with pytest.raises(JointRiggerContractError) as invalid_anchor_shape:
        rv._finite_vector3(
            object(),
            code="invalid_anchor_value",
            detail="invalid anchor shape",
        )
    assert invalid_anchor_shape.value.code == "invalid_anchor_value"
    assert rv._optional_number(None) is None
    assert math.isnan(rv._optional_number(True) or 0.0)
    assert math.isnan(rv._optional_number(object()) or 0.0)


@pytest.mark.parametrize(
    "locator",
    [
        "s:assets/../rig.usda",
        Sdf.Layer.CreateIdentifier(
            Ar.JoinPackageRelativePath(
                "s:packages/../rig.usdz",
                "nested/../root.usda",
            ),
            {"format": "usda", "target": "variant/a:b"},
        ),
    ],
)
def test_canonical_layer_locator_preserves_opaque_resolver_dot_segments(
    tmp_path: Path,
    locator: str,
) -> None:
    artifact = tmp_path / "root.usda"

    assert rv._canonical_layer_locator(locator, artifact_path=artifact) == locator


def test_canonical_layer_locator_unifies_local_paths_and_file_uris(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "root.usda"
    dependency = tmp_path / "dependencies" / "layer.usda"

    local_locator = rv._canonical_layer_locator(
        str(dependency),
        artifact_path=artifact,
    )
    file_uri_locator = rv._canonical_layer_locator(
        dependency.as_uri(),
        artifact_path=artifact,
    )

    assert local_locator == "dependencies/layer.usda"
    assert file_uri_locator == local_locator


def test_canonical_layer_locator_rebuilds_file_uri_package_and_arguments(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "root.usda"
    package = tmp_path / "packages" / "rig.usdz"
    package_inner = "nested/../root.usda"
    arguments = {"format": "usda", "target": "variant/a:b"}
    file_uri_locator = Sdf.Layer.CreateIdentifier(
        Ar.JoinPackageRelativePath(package.as_uri(), package_inner),
        arguments,
    )
    local_locator = Sdf.Layer.CreateIdentifier(
        Ar.JoinPackageRelativePath(str(package), package_inner),
        arguments,
    )
    expected = Sdf.Layer.CreateIdentifier(
        Ar.JoinPackageRelativePath("packages/rig.usdz", package_inner),
        arguments,
    )

    assert (
        rv._canonical_layer_locator(file_uri_locator, artifact_path=artifact)
        == expected
    )
    assert (
        rv._canonical_layer_locator(local_locator, artifact_path=artifact) == expected
    )


def test_canonical_layer_locator_rejects_malformed_file_uri_package(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "root.usda"
    malformed = Sdf.Layer.CreateIdentifier(
        Ar.JoinPackageRelativePath(
            "file://localhost/tmp/rig.usdz",
            "nested/root.usda",
        ),
        {"format": "usda"},
    )

    with pytest.raises(JointRiggerContractError) as caught:
        rv._canonical_layer_locator(malformed, artifact_path=artifact)

    assert caught.value.code == "invalid_artifact_identity"
    assert "exact canonical absolute file URI" in caught.value.detail


def test_projected_local_locator_ignores_empty_authored_path(tmp_path: Path) -> None:
    assert (
        rv._projected_local_locator(
            "",
            owner_path=tmp_path / "root.usda",
            projection=_test_projection(tmp_path / "projection"),
            Ar=Ar,
            Sdf=Sdf,
        )
        is None
    )


def test_captured_dependency_structure_builds_frozen_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, _ = _write_pair(tmp_path)
    dependency = _add_inert_sublayer(source, "identity_dependency.usda")
    logical_source = Path("/published/source.usda")

    structure = rv._capture_dependency_structure(
        source,
        logical_artifact_path=logical_source,
    )

    assert any(
        record.kind == "stage_root_layer"
        and record.locator == "$artifact"
        and record.backing_path is None
        for record in structure
    )
    dependency_record = next(
        record for record in structure if record.backing_path == dependency.resolve()
    )
    assert dependency_record.kind == "used_layer"
    frozen_records = tuple(
        rv._CapturedDependencyIdentityRecord(
            kind=record.kind,
            locator=record.locator,
            sha256=(
                _sha256(source)
                if record.backing_path is None
                else _sha256(record.backing_path)
            ),
            backing_path=record.backing_path,
        )
        for record in structure
    )
    identity = rv._artifact_identity_from_captured_records(
        logical_artifact_path=logical_source,
        uri="fixture://frozen-identity",
        root_sha256=_sha256(source),
        records=frozen_records,
    )
    assert identity.uri == "fixture://frozen-identity"
    assert len(identity.dependency_bundle_sha256 or "") == 64

    monkeypatch.setattr(
        rv,
        "_enumerate_usd_dependencies",
        lambda _path: (
            rv._ResolvedUsdDependency(
                kind="asset",
                identifier="resolver://remote-only",
                lexical_path=None,
                local_path=None,
                package_relative=False,
            ),
        ),
    )
    with pytest.raises(JointRiggerContractError) as unbound:
        rv._capture_dependency_structure(
            source,
            logical_artifact_path=logical_source,
        )
    assert unbound.value.code == "unbound_dependency_artifact"


def test_local_path_alias_chain_handles_relative_targets_and_cycles(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.usda"
    target.write_text("#usda 1.0\n", encoding="utf-8")
    alias = tmp_path / "relative.usda"
    alias.symlink_to(target.name)

    assert rv._local_path_alias_chain(alias) == (alias, target)

    first = tmp_path / "first.usda"
    second = tmp_path / "second.usda"
    first.symlink_to(second.name)
    second.symlink_to(first.name)
    with pytest.raises(JointRiggerContractError) as cycle:
        rv._local_path_alias_chain(first)
    assert cycle.value.code == "dependency_artifact_invalid"


@pytest.mark.parametrize(
    ("schema", "property_name"),
    [
        ("PhysxJointAPI:unsupported", "physxJoint:maxJointVelocity"),
        ("PhysxJointAPI", "physxJoint:solverFoo"),
    ],
)
def test_physx_joint_extractor_rejects_unrepresented_schema_and_property(
    schema: str,
    property_name: str,
) -> None:
    stage = Usd.Stage.CreateInMemory()
    prim = UsdGeom.Xform.Define(stage, "/Joint").GetPrim()
    assert prim.AddAppliedSchema(schema)
    prim.CreateAttribute(
        property_name,
        Sdf.ValueTypeNames.Float,
        custom=False,
    ).Set(1.0)

    with pytest.raises(JointRiggerContractError) as caught:
        rv._extract_physx_joint_opinions(
            prim,
            joint_type="revolute",
            joint_path="/Joint",
            has_drive=True,
            reference_identity=ArtifactIdentityV1(
                uri="fixture://physx-joint",
                root_sha256="a" * 64,
            ),
        )
    assert caught.value.code == "unsupported_optional_schema"


def test_composed_joint_mismatch_reports_each_property_dimension() -> None:
    stages: list[Any] = []

    def prim_pair() -> tuple[Any, Any]:
        source_stage = Usd.Stage.CreateInMemory()
        reference_stage = Usd.Stage.CreateInMemory()
        stages.extend((source_stage, reference_stage))
        return (
            source_stage.DefinePrim("/Joint", "Xform"),
            reference_stage.DefinePrim("/Joint", "Xform"),
        )

    source, reference = prim_pair()
    reference.SetTypeName("Scope")
    assert "typeName" in (rv._composed_joint_mismatch(source, reference) or "")

    source, reference = prim_pair()
    assert source.AddAppliedSchema("PhysicsCustomAPI")
    assert "appliedSchemas" in (rv._composed_joint_mismatch(source, reference) or "")

    source, reference = prim_pair()
    source.CreateAttribute("physics:test", Sdf.ValueTypeNames.Int).Set(1)
    assert "authoredProperties" in (
        rv._composed_joint_mismatch(source, reference) or ""
    )

    source, reference = prim_pair()
    source.CreateAttribute("physics:test", Sdf.ValueTypeNames.Int).Set(1)
    reference.CreateRelationship("physics:test").SetTargets(["/Target"])
    assert "changes property kind" in (
        rv._composed_joint_mismatch(source, reference) or ""
    )

    source, reference = prim_pair()
    source.CreateAttribute("physics:test", Sdf.ValueTypeNames.Int).Set(1)
    reference.CreateAttribute("physics:test", Sdf.ValueTypeNames.Int).Set(2)
    assert "attribute physics:test differs" == rv._composed_joint_mismatch(
        source,
        reference,
    )

    source, reference = prim_pair()
    source.CreateAttribute("physics:test", Sdf.ValueTypeNames.Int).Set(1)
    reference.CreateAttribute("physics:test", Sdf.ValueTypeNames.Int).Set(1)
    assert rv._composed_joint_mismatch(source, reference) is None


def test_ancestor_and_nested_simulation_owner_relationships_fail_closed(
    tmp_path: Path,
) -> None:
    source, reference = _write_pair(tmp_path / "ancestor")
    stage = Usd.Stage.Open(str(reference))
    stage.GetPrimAtPath("/World").CreateRelationship(
        "physics:simulationOwner",
        custom=False,
    ).SetTargets(["/World/base"])
    assert stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as ancestor:
        _extract(source, reference)
    assert ancestor.value.code == "unsupported_rigid_body_relationship"
    assert "ancestor" in ancestor.value.detail

    source, reference = _write_pair(tmp_path / "nested")
    nested_path = "/World/base/nestedBody"
    for stage_path in (source, reference):
        stage = Usd.Stage.Open(str(stage_path))
        nested = UsdGeom.Xform.Define(stage, nested_path).GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(nested)
        if stage_path == reference:
            nested.CreateRelationship(
                "physics:simulationOwner",
                custom=False,
            ).SetTargets(["/World/base"])
        assert stage.GetRootLayer().Save()

    with pytest.raises(JointRiggerContractError) as nested:
        _extract(source, reference)
    assert nested.value.code == "unsupported_rigid_body_relationship"
    assert "nested" in nested.value.detail


def test_registered_api_fallbacks_cover_synthetic_prim_behavior() -> None:
    class Prim:
        @staticmethod
        def GetMetadata(_name: str) -> None:
            return None

        @staticmethod
        def GetAppliedSchemas() -> tuple[str, ...]:
            return ("PhysicsSyntheticAPI",)

        @staticmethod
        def HasAPI(schema: Any) -> bool:
            return schema is expected_schema

    expected_schema = object()
    prim = Prim()

    assert rv._applied_schema_tokens(prim) == ("PhysicsSyntheticAPI",)
    assert rv._physics_api_facts(
        prim,
        (("PhysicsFallbackAPI", expected_schema),),
    ) == ("PhysicsSyntheticAPI", "PhysicsFallbackAPI")


def test_low_level_value_mass_asset_and_quaternion_guards() -> None:
    class Uncomparable:
        def __eq__(self, _other: object) -> bool:
            raise TypeError("not comparable")

    class NegativeAsset:
        @staticmethod
        def GetSize() -> int:
            return -1

    assert not rv._usd_values_equal(Uncomparable(), Uncomparable())
    with pytest.raises(JointRiggerContractError) as negative_size:
        rv._ar_asset_sha256(NegativeAsset(), identifier="fake://negative")
    assert negative_size.value.code == "dependency_artifact_read_failed"
    with pytest.raises(JointRiggerContractError) as malformed_quaternion:
        rv._validated_joint_frame_rotation(
            object(),
            joint_path="/World/Joints/test",
            field_name="physics:localRot0",
            Gf=Gf,
        )
    assert malformed_quaternion.value.code == "invalid_joint_frame_rotation"


def test_material_binding_filter_and_parser_edge_cases() -> None:
    source_stage = Usd.Stage.CreateInMemory()
    reference_stage = Usd.Stage.CreateInMemory()
    source_owner = source_stage.DefinePrim("/World/Owner", "Xform")
    reference_owner = reference_stage.DefinePrim("/World/Owner", "Xform")
    source_owner.CreateRelationship("material:binding:preview").SetTargets(
        ["/World/Looks/Visual"]
    )
    reference_owner.CreateRelationship("material:binding:preview").SetTargets(
        ["/World/Looks/Visual"]
    )
    assert not rv._require_physics_material_bindings_preexisting(
        source_stage,
        reference_stage=reference_stage,
        source_prims={"/World/Owner": source_owner},
        reference_prims={"/World/Owner": reference_owner},
        replay_scope_paths={"/World/Owner"},
    )

    visual_source = UsdShade.Material.Define(
        source_stage,
        "/World/Looks/Visual",
    )
    visual_reference = UsdShade.Material.Define(
        reference_stage,
        "/World/Looks/Visual",
    )
    assert UsdShade.MaterialBindingAPI.Apply(source_owner).Bind(visual_source)
    assert UsdShade.MaterialBindingAPI.Apply(reference_owner).Bind(visual_reference)
    assert not rv._require_physics_material_bindings_preexisting(
        source_stage,
        reference_stage=reference_stage,
        source_prims={"/World/Owner": source_owner},
        reference_prims={"/World/Owner": reference_owner},
        replay_scope_paths={"/World/Owner"},
    )
    assert not rv._material_binding_may_target_physics(
        None,
        source_stage,
        reference_stage=reference_stage,
        relationship_name="material:binding",
    )
    assert not rv._material_target_has_physics_facts(
        source_stage,
        reference_stage=reference_stage,
        material_path="/World/Looks/Missing",
    )
    assert (
        rv._material_binding_target(
            None,
            relationship_name="material:binding:physics",
            strict=True,
        )
        is None
    )

    one_sided_source_stage = Usd.Stage.CreateInMemory()
    one_sided_reference_stage = Usd.Stage.CreateInMemory()
    one_sided_source = one_sided_source_stage.DefinePrim("/Owner", "Xform")
    one_sided_reference = one_sided_reference_stage.DefinePrim("/Owner", "Xform")
    one_sided_source.CreateRelationship(
        "material:binding:physics",
    ).SetTargets(["/Looks/Physics"])
    with pytest.raises(JointRiggerContractError) as one_sided:
        rv._require_physics_material_bindings_preexisting(
            one_sided_source_stage,
            reference_stage=one_sided_reference_stage,
            source_prims={"/Owner": one_sided_source},
            reference_prims={"/Owner": one_sided_reference},
            replay_scope_paths={"/Owner"},
        )
    assert one_sided.value.code == "unrepresented_physics_material_binding"

    parser_stage = Usd.Stage.CreateInMemory()
    attribute_owner = parser_stage.DefinePrim("/AttributeOwner", "Xform")
    attribute_owner.CreateAttribute(
        "material:binding:physics",
        Sdf.ValueTypeNames.String,
    ).Set("not a relationship")
    assert (
        rv._material_binding_target(
            attribute_owner,
            relationship_name="material:binding:physics",
            strict=False,
        )
        is None
    )
    with pytest.raises(JointRiggerContractError) as property_kind:
        rv._material_binding_target(
            attribute_owner,
            relationship_name="material:binding:physics",
            strict=True,
        )
    assert property_kind.value.code == "unsupported_physics_material_binding"

    invalid_material_owner = parser_stage.DefinePrim("/InvalidMaterial", "Xform")
    invalid_material_owner.CreateRelationship(
        "material:binding:physics",
    ).SetTargets(["/World/Looks/Material.outputs:surface"])
    assert (
        rv._material_binding_target(
            invalid_material_owner,
            relationship_name="material:binding:physics",
            strict=False,
        )
        is None
    )
    with pytest.raises(JointRiggerContractError) as invalid_material:
        rv._material_binding_target(
            invalid_material_owner,
            relationship_name="material:binding:physics",
            strict=True,
        )
    assert invalid_material.value.code == "unsupported_physics_material_binding"

    invalid_collection_owner = parser_stage.DefinePrim(
        "/InvalidCollection",
        "Xform",
    )
    invalid_collection_owner.CreateRelationship(
        "material:binding:collection:physics:test",
    ).SetTargets(["/World/NotACollectionProperty", "/World/Looks/Material"])
    assert (
        rv._material_binding_target(
            invalid_collection_owner,
            relationship_name="material:binding:collection:physics:test",
            strict=False,
        )
        is None
    )
    with pytest.raises(JointRiggerContractError) as invalid_collection:
        rv._material_binding_target(
            invalid_collection_owner,
            relationship_name="material:binding:collection:physics:test",
            strict=True,
        )
    assert invalid_collection.value.code == "unsupported_physics_material_binding"


def test_material_target_and_collection_definition_guards() -> None:
    source_stage = Usd.Stage.CreateInMemory()
    reference_stage = Usd.Stage.CreateInMemory()
    with pytest.raises(JointRiggerContractError) as missing_material:
        rv._require_material_target_resolved(
            source_stage,
            reference_stage=reference_stage,
            owner_path="/World/Owner",
            relationship_name="material:binding:physics",
            material_path="/World/Looks/Missing",
        )
    assert missing_material.value.code == "unrepresented_physics_material_binding"

    checked = {("/World", "alreadyChecked")}
    rv._require_collection_definition_preexisting(
        source_stage,
        reference_stage=reference_stage,
        owner_path="/World/Owner",
        relationship_name="material:binding:collection:physics:test",
        collection_prim_path="/World",
        collection_instance="alreadyChecked",
        checked_collections=checked,
    )
    with pytest.raises(JointRiggerContractError) as missing_collection:
        rv._require_collection_definition_preexisting(
            source_stage,
            reference_stage=reference_stage,
            owner_path="/World/Owner",
            relationship_name="material:binding:collection:physics:test",
            collection_prim_path="/World/Missing",
            collection_instance="test",
            checked_collections=set(),
        )
    assert missing_collection.value.code == "unsupported_physics_material_binding"

    source_world = source_stage.DefinePrim("/World", "Xform")
    reference_world = reference_stage.DefinePrim("/World", "Xform")
    with pytest.raises(JointRiggerContractError) as missing_api:
        rv._require_collection_definition_preexisting(
            source_stage,
            reference_stage=reference_stage,
            owner_path="/World/Owner",
            relationship_name="material:binding:collection:physics:test",
            collection_prim_path="/World",
            collection_instance="test",
            checked_collections=set(),
        )
    assert missing_api.value.code == "unsupported_physics_material_binding"

    for prim in (source_world, reference_world):
        invalid = Usd.CollectionAPI.Apply(prim, "invalid")
        invalid.CreateExpansionRuleAttr("notAnExpansionRule")
    with pytest.raises(JointRiggerContractError) as invalid_collection:
        rv._require_collection_definition_preexisting(
            source_stage,
            reference_stage=reference_stage,
            owner_path="/World/Owner",
            relationship_name="material:binding:collection:physics:invalid",
            collection_prim_path="/World",
            collection_instance="invalid",
            checked_collections=set(),
        )
    assert invalid_collection.value.code == "unsupported_physics_material_binding"

    nested_source_stage = Usd.Stage.CreateInMemory()
    nested_reference_stage = Usd.Stage.CreateInMemory()
    for stage in (nested_source_stage, nested_reference_stage):
        world = stage.DefinePrim("/World", "Xform")
        stage.DefinePrim("/World/Member", "Xform")
        inner = Usd.CollectionAPI.Apply(world, "inner")
        inner.GetIncludesRel().SetTargets(["/World/Member"])
        outer = Usd.CollectionAPI.Apply(world, "outer")
        outer.GetIncludesRel().SetTargets([inner.GetCollectionPath()])
    nested_checked: set[tuple[str, str]] = set()
    rv._require_collection_definition_preexisting(
        nested_source_stage,
        reference_stage=nested_reference_stage,
        owner_path="/World/Owner",
        relationship_name="material:binding:collection:physics:outer",
        collection_prim_path="/World",
        collection_instance="outer",
        checked_collections=nested_checked,
    )
    assert nested_checked == {("/World", "outer"), ("/World", "inner")}


def test_collection_definition_closure_limits_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_stage = Usd.Stage.CreateInMemory()
    reference_stage = Usd.Stage.CreateInMemory()
    for stage in (source_stage, reference_stage):
        world = stage.DefinePrim("/World", "Xform")
        nested_paths = []
        for index in range(3):
            nested = Usd.CollectionAPI.Apply(world, f"nested{index}")
            nested_paths.append(nested.GetCollectionPath())
        outer = Usd.CollectionAPI.Apply(world, "outer")
        outer.GetIncludesRel().SetTargets(nested_paths)

    monkeypatch.setattr(rv, "_MAX_PHYSICS_MATERIAL_COLLECTION_DEFINITIONS", 2)
    with pytest.raises(JointRiggerContractError) as count_limit:
        rv._require_collection_definition_preexisting(
            source_stage,
            reference_stage=reference_stage,
            owner_path="/World/Owner",
            relationship_name="material:binding:collection:physics:outer",
            collection_prim_path="/World",
            collection_instance="outer",
            checked_collections=set(),
        )
    assert count_limit.value.code == "unsupported_physics_material_binding"
    assert "maximum definition count of 2" in count_limit.value.detail

    for stage in (source_stage, reference_stage):
        world = stage.GetPrimAtPath("/World")
        target_path = Sdf.Path("/World")
        for index in range(4):
            nested = Usd.CollectionAPI.Apply(world, f"chain{index}")
            nested.GetIncludesRel().SetTargets([target_path])
            target_path = nested.GetCollectionPath()

    monkeypatch.setattr(rv, "_MAX_PHYSICS_MATERIAL_COLLECTION_DEPTH", 2)
    monkeypatch.setattr(
        rv,
        "_MAX_PHYSICS_MATERIAL_COLLECTION_DEFINITIONS",
        256,
    )
    with pytest.raises(JointRiggerContractError) as depth_limit:
        rv._require_collection_definition_preexisting(
            source_stage,
            reference_stage=reference_stage,
            owner_path="/World/Owner",
            relationship_name="material:binding:collection:physics:chain3",
            collection_prim_path="/World",
            collection_instance="chain3",
            checked_collections=set(),
        )
    assert depth_limit.value.code == "unsupported_physics_material_binding"
    assert "maximum nesting depth of 2" in depth_limit.value.detail


def test_nearest_body_owner_uses_path_ancestors_without_scanning_bodies() -> None:
    class MembershipOnlyBodyPaths(set[str]):
        def __iter__(self) -> Iterator[str]:
            raise AssertionError("body ownership must not scan every body")

    body_paths = MembershipOnlyBodyPaths(
        {
            "/World/base",
            "/World/base/nested",
            "/World/unrelated",
        }
    )

    assert (
        rv._nearest_body_owner("/World/base/nested/collider", body_paths)
        == "/World/base/nested"
    )
    assert rv._nearest_body_owner("/World/base/visual", body_paths) == "/World/base"
    assert rv._nearest_body_owner("/World/baseball", body_paths) is None

    class FakePrim:
        def __init__(self, path: str) -> None:
            self.path = path

        def GetPath(self) -> str:
            return self.path

    class SinglePassPrims:
        def __init__(self, *paths: str) -> None:
            self.prims = tuple(FakePrim(path) for path in paths)
            self.iterations = 0

        def __iter__(self) -> Iterator[FakePrim]:
            self.iterations += 1
            if self.iterations > 1:
                raise AssertionError("collider prims must be grouped in one pass")
            return iter(self.prims)

    prims = SinglePassPrims(
        "/World/base",
        "/World/base/visual",
        "/World/base/nested/collider",
        "/World/unrelated/visual",
        "/World/unowned",
    )
    grouped = rv._prims_by_nearest_body_owner(prims, body_paths)

    assert prims.iterations == 1
    assert {
        owner: tuple(prim.path for prim in owned_prims)
        for owner, owned_prims in grouped.items()
    } == {
        "/World/base": ("/World/base", "/World/base/visual"),
        "/World/base/nested": ("/World/base/nested/collider",),
        "/World/unrelated": ("/World/unrelated/visual",),
    }


def test_authored_attribute_connection_identity_is_compared() -> None:
    source_stage = Usd.Stage.CreateInMemory()
    reference_stage = Usd.Stage.CreateInMemory()
    source_prim = source_stage.DefinePrim("/Owner", "Xform")
    reference_prim = reference_stage.DefinePrim("/Owner", "Xform")
    source_attribute = source_prim.CreateAttribute(
        "physics:value",
        Sdf.ValueTypeNames.Float,
    )
    reference_attribute = reference_prim.CreateAttribute(
        "physics:value",
        Sdf.ValueTypeNames.Float,
    )
    source_attribute.Set(1.0)
    reference_attribute.Set(1.0)
    source_driver = source_prim.CreateAttribute(
        "sourceDriver",
        Sdf.ValueTypeNames.Float,
    )
    reference_driver = reference_prim.CreateAttribute(
        "referenceDriver",
        Sdf.ValueTypeNames.Float,
    )
    source_attribute.AddConnection(source_driver.GetPath())
    reference_attribute.AddConnection(reference_driver.GetPath())

    assert not rv._matching_authored_attribute(
        source_attribute,
        reference_attribute,
    )


def test_collider_geometry_world_transform_and_nested_ownership_guards() -> None:
    source_stage = Usd.Stage.CreateInMemory()
    reference_stage = Usd.Stage.CreateInMemory()
    source_cube = UsdGeom.Cube.Define(source_stage, "/Cube").GetPrim()
    reference_cube = UsdGeom.Cube.Define(reference_stage, "/Cube").GetPrim()
    source_cube.CreateAttribute(
        "collisionShapeHint",
        Sdf.ValueTypeNames.Float,
    ).Set(1.0)
    with pytest.raises(JointRiggerContractError) as geometry:
        rv._require_matching_collider_geometry(
            source_cube,
            reference_cube,
            path="/Cube",
        )
    assert geometry.value.code == "source_collider_geometry_mismatch"

    transformed_source = Usd.Stage.CreateInMemory()
    transformed_reference = Usd.Stage.CreateInMemory()
    for stage, translation in (
        (transformed_source, 1.0),
        (transformed_reference, 2.0),
    ):
        world = UsdGeom.Xform.Define(stage, "/World")
        world.AddTranslateOp().Set(Gf.Vec3d(translation, 0.0, 0.0))
        UsdGeom.Xform.Define(stage, "/World/Body")
        UsdGeom.Cube.Define(stage, "/World/Body/Collider")
    with pytest.raises(JointRiggerContractError) as world_transform:
        rv._require_matching_collider_transforms(
            transformed_source,
            transformed_reference,
            body_path="/World/Body",
            path="/World/Body/Collider",
            UsdGeom=UsdGeom,
        )
    assert world_transform.value.code == "source_collider_transform_mismatch"
    assert "world transform" in world_transform.value.detail

    ownership_stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(ownership_stage, "/World")
    UsdGeom.Xform.Define(ownership_stage, "/World/Body")
    nested = UsdGeom.Xform.Define(
        ownership_stage,
        "/World/Body/Nested",
    ).GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(nested)
    collider = UsdGeom.Cube.Define(
        ownership_stage,
        "/World/Body/Nested/Collider",
    ).GetPrim()
    UsdPhysics.CollisionAPI.Apply(collider)
    assert not rv._unowned_collision_evidence(
        ownership_stage,
        body_path="/World/Body",
        all_rigid_body_paths={"/World/Body/Nested"},
        UsdPhysics=UsdPhysics,
    )


@pytest.mark.parametrize(
    "race",
    ["nonregular", "before_open", "short_read", "growth", "post_hash"],
)
def test_file_sha256_rejects_inode_and_content_races(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    race: str,
) -> None:
    if race == "nonregular":
        target = tmp_path / "directory"
        target.mkdir()
    else:
        target = tmp_path / "artifact.usda"
        target.write_bytes(b"stable payload")

    real_fstat = rv.os.fstat
    real_pread = rv.os.pread

    class ChangedStat:
        def __init__(self, original: Any, **changes: int) -> None:
            self.original = original
            self.changes = changes

        def __getattr__(self, name: str) -> Any:
            if name in self.changes:
                return self.changes[name]
            return getattr(self.original, name)

    if race == "before_open":
        monkeypatch.setattr(
            rv.os,
            "fstat",
            lambda descriptor: ChangedStat(
                real_fstat(descriptor),
                st_size=real_fstat(descriptor).st_size + 1,
            ),
        )
    elif race == "short_read":
        monkeypatch.setattr(rv.os, "pread", lambda *_args: b"")
    elif race == "growth":
        expected_size = target.stat().st_size

        def growing_pread(descriptor: int, count: int, offset: int) -> bytes:
            if offset == expected_size:
                return b"x"
            return real_pread(descriptor, count, offset)

        monkeypatch.setattr(rv.os, "pread", growing_pread)
    elif race == "post_hash":
        calls = 0

        def changing_fstat(descriptor: int) -> Any:
            nonlocal calls
            calls += 1
            current = real_fstat(descriptor)
            if calls == 2:
                return ChangedStat(
                    current,
                    st_mtime_ns=current.st_mtime_ns + 1,
                )
            return current

        monkeypatch.setattr(rv.os, "fstat", changing_fstat)

    with pytest.raises(JointRiggerContractError) as caught:
        rv._file_sha256(target, code="artifact_raced")
    assert caught.value.code == "artifact_raced"


# OpenUSD process startup can exceed 15 seconds in the fully parallel CI suite.
# Keep special-file probes bounded while allowing for that runner contention.
_SPECIAL_FILE_SUBPROCESS_TIMEOUT_SECONDS = 60.0


def _identify_artifact_subprocess(path: Path) -> subprocess.CompletedProcess[str]:
    script = "\n".join(
        (
            "import sys",
            "from pathlib import Path",
            "from world_understanding.functions.physics.joint_rigger.models "
            "import JointRiggerContractError",
            "from world_understanding.functions.physics.joint_rigger.reference "
            "import identify_usd_artifact",
            "try:",
            "    identify_usd_artifact(Path(sys.argv[1]), uri='fixture://child')",
            "except JointRiggerContractError as exc:",
            "    print(f'error:{exc.code}')",
            "else:",
            "    print('ok')",
        )
    )
    return subprocess.run(
        [sys.executable, "-c", script, str(path)],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
        timeout=_SPECIAL_FILE_SUBPROCESS_TIMEOUT_SECONDS,
    )


def _extract_reference_subprocess(
    source: Path,
    reference: Path,
) -> subprocess.CompletedProcess[str]:
    script = "\n".join(
        (
            "import sys",
            "from pathlib import Path",
            "from world_understanding.functions.physics.joint_rigger.models "
            "import JointRiggerContractError",
            "from world_understanding.functions.physics.joint_rigger.reference "
            "import extract_reference_input",
            "try:",
            "    extract_reference_input(",
            "        Path(sys.argv[1]),",
            "        Path(sys.argv[2]),",
            "        source_uri='fixture://source',",
            "        reference_uri='fixture://reference',",
            "    )",
            "except JointRiggerContractError as exc:",
            "    print(f'error:{exc.code}')",
            "else:",
            "    print('ok')",
        )
    )
    return subprocess.run(
        [sys.executable, "-c", script, str(source), str(reference)],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
        timeout=_SPECIAL_FILE_SUBPROCESS_TIMEOUT_SECONDS,
    )


def _local_dependency_paths_subprocess(
    path: Path,
) -> subprocess.CompletedProcess[str]:
    script = "\n".join(
        (
            "import sys",
            "from pathlib import Path",
            "from world_understanding.functions.physics.joint_rigger.models "
            "import JointRiggerContractError",
            "from world_understanding.functions.physics.joint_rigger.reference "
            "import local_usd_dependency_paths",
            "try:",
            "    local_usd_dependency_paths(Path(sys.argv[1]))",
            "except JointRiggerContractError as exc:",
            "    print(f'error:{exc.code}')",
            "else:",
            "    print('ok')",
        )
    )
    return subprocess.run(
        [sys.executable, "-c", script, str(path)],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
        timeout=_SPECIAL_FILE_SUBPROCESS_TIMEOUT_SECONDS,
    )


def _post_projection_fifo_swap_subprocess(
    operation: str,
    primary: Path,
    dependency: Path,
    *,
    secondary: Path | None = None,
    resolver_root: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    script = "\n".join(
        (
            "import os",
            "import sys",
            "from pathlib import Path",
            "import world_understanding.functions.physics.joint_rigger.reference as rv",
            "from world_understanding.functions.physics.joint_rigger.models "
            "import JointRiggerContractError",
            "operation = sys.argv[1]",
            "primary = Path(sys.argv[2])",
            "secondary = Path(sys.argv[3]) if sys.argv[3] else None",
            "dependency = Path(sys.argv[4])",
            "resolver_root = Path(sys.argv[5]) if sys.argv[5] else None",
            "if resolver_root is not None:",
            "    os.chdir(resolver_root)",
            "swapped = False",
            "def swap_dependency():",
            "    global swapped",
            "    if swapped:",
            "        return",
            "    dependency.unlink(missing_ok=True)",
            "    os.mkfifo(dependency)",
            "    swapped = True",
            "if operation == 'local_paths':",
            "    real_inventory = rv._fresh_usd_dependency_inventory",
            "    def hooked_inventory(path):",
            "        swap_dependency()",
            "        return real_inventory(path)",
            "    rv._fresh_usd_dependency_inventory = hooked_inventory",
            "else:",
            "    real_open_stage = rv._open_stage",
            "    def hooked_open_stage(path, **kwargs):",
            "        swap_dependency()",
            "        return real_open_stage(path, **kwargs)",
            "    rv._open_stage = hooked_open_stage",
            "try:",
            "    if operation == 'identify':",
            "        rv.identify_usd_artifact(primary, uri='fixture://child')",
            "    elif operation == 'local_paths':",
            "        rv.local_usd_dependency_paths(primary)",
            "    elif operation == 'extract':",
            "        assert secondary is not None",
            "        rv.extract_reference_input(",
            "            primary,",
            "            secondary,",
            "            source_uri='fixture://source',",
            "            reference_uri='fixture://reference',",
            "        )",
            "    else:",
            "        raise AssertionError(f'unknown operation: {operation}')",
            "except JointRiggerContractError as exc:",
            "    print(f'error:{exc.code}')",
            "else:",
            "    print('ok')",
        )
    )
    return subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            operation,
            str(primary),
            str(secondary) if secondary is not None else "",
            str(dependency),
            str(resolver_root) if resolver_root is not None else "",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
        timeout=_SPECIAL_FILE_SUBPROCESS_TIMEOUT_SECONDS,
    )


def _preflight_dependency_subprocess(path: Path) -> subprocess.CompletedProcess[str]:
    script = "\n".join(
        (
            "import sys",
            "from pathlib import Path",
            "import world_understanding.functions.physics.joint_rigger.reference as rv",
            "from world_understanding.functions.physics.joint_rigger.models "
            "import JointRiggerContractError",
            "try:",
            "    rv._preflight_local_dependency_locators(Path(sys.argv[1]))",
            "except JointRiggerContractError as exc:",
            "    print(f'error:{exc.code}')",
            "else:",
            "    print('ok')",
        )
    )
    return subprocess.run(
        [sys.executable, "-c", script, str(path)],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
        timeout=_SPECIAL_FILE_SUBPROCESS_TIMEOUT_SECONDS,
    )


def _write_pair(
    directory: Path,
    *,
    reverse: bool = False,
    include_fixed: bool = False,
    articulation_root_on_ancestor: bool = False,
) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    source_path = directory / "source.usda"
    reference_path = directory / "reference.usda"
    for path, rigged in ((source_path, False), (reference_path, True)):
        stage = Usd.Stage.CreateNew(str(path))
        UsdGeom.SetStageMetersPerUnit(stage, 0.01)
        root = UsdGeom.Xform.Define(stage, "/World").GetPrim()
        bodies = ("base", "drawer", "door", "ball")
        for name in reversed(bodies) if reverse else bodies:
            UsdGeom.Cube.Define(stage, f"/World/{name}")
        stage.SetDefaultPrim(root)
        if rigged:
            UsdGeom.Scope.Define(stage, "/World/Joints")
            for body_path in (
                "/World/base",
                "/World/drawer",
                "/World/door",
                "/World/ball",
            ):
                prim = stage.GetPrimAtPath(body_path)
                UsdPhysics.RigidBodyAPI.Apply(prim).CreateRigidBodyEnabledAttr(True)
                if body_path == "/World/base":
                    mass = UsdPhysics.MassAPI.Apply(prim)
                    mass.CreateMassAttr(2.0)
                    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(100.0, 200.0, 300.0))
                UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
            articulation_root = (
                root
                if articulation_root_on_ancestor
                else stage.GetPrimAtPath("/World/base")
            )
            UsdPhysics.ArticulationRootAPI.Apply(articulation_root)
            names = ["drawer", "door", "spherical"]
            if reverse:
                names.reverse()
            for name in names:
                _define_joint(stage, name)
            if include_fixed:
                fixed = UsdPhysics.FixedJoint.Define(stage, "/World/Joints/fixed")
                fixed.CreateBody0Rel().SetTargets(["/World/base"])
                fixed.CreateBody1Rel().SetTargets(["/World/door"])
        stage.GetRootLayer().Save()
    return source_path, reference_path


def _write_instance_asset(
    path: Path,
    *,
    include_joint: bool = False,
    include_collider: bool = False,
    include_articulation_root: bool = False,
) -> None:
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/Asset").GetPrim()
    stage.SetDefaultPrim(root)
    UsdGeom.Cube.Define(stage, "/Asset/body0")
    UsdGeom.Cube.Define(stage, "/Asset/body1")
    collider = UsdGeom.Cube.Define(stage, "/Asset/collider").GetPrim()
    articulation = UsdGeom.Xform.Define(stage, "/Asset/articulation").GetPrim()
    UsdGeom.Cube.Define(stage, "/Asset/articulation/body")
    if include_joint:
        UsdGeom.Scope.Define(stage, "/Asset/Joints")
        joint = UsdPhysics.RevoluteJoint.Define(stage, "/Asset/Joints/hinge")
        joint.CreateBody0Rel().SetTargets(["/Asset/body0"])
        joint.CreateBody1Rel().SetTargets(["/Asset/body1"])
        joint.CreateAxisAttr("Z")
    if include_collider:
        UsdPhysics.CollisionAPI.Apply(collider).CreateCollisionEnabledAttr(True)
    if include_articulation_root:
        UsdPhysics.ArticulationRootAPI.Apply(articulation)
    assert stage.GetRootLayer().Save()


def _add_instance_reference(
    stage_path: Path,
    asset_path: Path,
    instance_path: str,
) -> None:
    stage = Usd.Stage.Open(str(stage_path))
    prim = stage.DefinePrim(instance_path, "Xform")
    prim.GetReferences().AddReference(str(asset_path))
    prim.SetInstanceable(True)
    assert stage.GetRootLayer().Save()


def _add_inert_sublayer(root_path: Path, name: str) -> Path:
    dependency_path = root_path.parent / name
    dependency_stage = Usd.Stage.CreateNew(str(dependency_path))
    UsdGeom.Scope.Define(dependency_stage, "/IdentityDependency")
    assert dependency_stage.GetRootLayer().Save()
    root_stage = Usd.Stage.Open(str(root_path))
    root_stage.GetRootLayer().subLayerPaths.append(name)
    assert root_stage.GetRootLayer().Save()
    return dependency_path


def _write_value_layer(path: Path, value: int) -> None:
    stage = Usd.Stage.CreateNew(str(path))
    prim = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    prim.CreateAttribute("identityValue", Sdf.ValueTypeNames.Int).Set(value)
    assert stage.GetRootLayer().Save()


def _write_sublayer_root(path: Path, sublayer: str | None = None) -> None:
    metadata = f"( subLayers = [@{sublayer}@] )\n" if sublayer else ""
    path.write_text(
        f'#usda 1.0\n{metadata}def Scope "Layer" {{}}\n',
        encoding="utf-8",
    )


def _forbid_file_payload_read(
    monkeypatch: pytest.MonkeyPatch,
    path: Path,
) -> None:
    """Fail if projection reads bytes from one security-test operand."""

    expected = path.stat()
    expected_identity = (expected.st_dev, expected.st_ino)
    real_pread = rv.os.pread

    def guarded_pread(descriptor: int, size: int, offset: int) -> bytes:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) == expected_identity:
            raise AssertionError(f"projection read rejected operand bytes: {path}")
        return real_pread(descriptor, size, offset)

    monkeypatch.setattr(rv.os, "pread", guarded_pread)


def _write_resolver_asset_root(path: Path, locator: str) -> None:
    path.write_text(
        "\n".join(
            (
                "#usda 1.0",
                "",
                'def Scope "World"',
                "{",
                f"    custom asset test:resolverAsset = @{locator}@",
                "}",
                "",
            )
        ),
        encoding="utf-8",
    )


def _test_projection(directory: Path) -> Any:
    mirror_root = directory / "absolute"
    mirror_root.mkdir(parents=True)
    return rv._UsdCompositionProjection(
        mirror_root=mirror_root,
        files={},
        closures={},
        layer_dependencies={},
    )


def _reload_layers(*paths: Path) -> None:
    expected = {str(path) for path in paths}
    for layer in Sdf.Layer.GetLoadedLayers():
        if layer.realPath in expected:
            layer.Reload()


def _define_joint(stage: Any, name: str) -> None:
    if name == "drawer":
        joint = UsdPhysics.PrismaticJoint.Define(stage, "/World/Joints/drawer")
        joint.CreateBody0Rel().SetTargets(["/World/base"])
        joint.CreateBody1Rel().SetTargets(["/World/drawer"])
        joint.CreateAxisAttr("X")
        joint.CreateLowerLimitAttr(0.0)
        joint.CreateUpperLimitAttr(50.0)
        joint.CreateLocalPos0Attr(Gf.Vec3f(10.0, 0.0, 0.0))
        joint.CreateLocalPos1Attr(Gf.Vec3f(10.0, 0.0, 0.0))
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(25.0)
        drive.CreateDampingAttr(2.0)
        drive.CreateMaxForceAttr(100.0)
        drive.CreateTargetPositionAttr(0.0)
        drive.CreateTargetVelocityAttr(0.0)
    elif name == "door":
        joint = UsdPhysics.RevoluteJoint.Define(stage, "/World/Joints/door")
        joint.CreateBody0Rel().SetTargets(["/World/base"])
        joint.CreateBody1Rel().SetTargets(["/World/door"])
        joint.CreateAxisAttr("Z")
    elif name == "spherical":
        joint = UsdPhysics.SphericalJoint.Define(stage, "/World/Joints/spherical")
        joint.CreateBody0Rel().SetTargets(["/World/base"])
        joint.CreateBody1Rel().SetTargets(["/World/ball"])
    else:  # pragma: no cover - helper guard
        raise AssertionError(name)


def _apply_complete_drive(prim: Any, instance: str) -> None:
    drive = UsdPhysics.DriveAPI.Apply(prim, instance)
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(25.0)
    drive.CreateDampingAttr(2.0)
    drive.CreateMaxForceAttr(100.0)
    drive.CreateTargetPositionAttr(0.0)
    drive.CreateTargetVelocityAttr(0.0)


def _connect_value_only_attribute(stage: Any, attribute: Any) -> None:
    attribute.Clear()
    driver = stage.GetPrimAtPath("/World").CreateAttribute(
        "connectionDriver",
        attribute.GetTypeName(),
        custom=True,
    )
    assert attribute.AddConnection(driver.GetPath())
    assert not attribute.HasAuthoredValueOpinion()
    assert attribute.HasAuthoredConnections()


def _apply_unmodeled_physics_fact(
    stage: Any,
    prim_path: str,
    fact_kind: str,
    value: Any,
) -> None:
    prim = stage.GetPrimAtPath(prim_path)
    assert prim and prim.IsValid()
    if fact_kind == "filtered_pairs":
        assert prim.AddAppliedSchema("PhysicsFilteredPairsAPI")
        relationship = prim.CreateRelationship("physics:filteredPairs", custom=False)
        assert relationship.SetTargets([value])
        return
    if fact_kind == "physx_rigid_body":
        assert prim.AddAppliedSchema("PhysxRigidBodyAPI")
        attribute = prim.CreateAttribute(
            "physxRigidBody:disableGravity",
            Sdf.ValueTypeNames.Bool,
            custom=False,
        )
        assert attribute.Set(value)
        return
    if fact_kind == "drive_namespace":
        assert prim.AddAppliedSchema("PhysicsDriveAPI:angular")
        attribute = prim.CreateAttribute(
            "drive:angular:physics:stiffness",
            Sdf.ValueTypeNames.Float,
            custom=False,
        )
        assert attribute.Set(value)
        return
    if fact_kind == "state_namespace":
        assert prim.AddAppliedSchema("PhysicsJointStateAPI:angular")
        attribute = prim.CreateAttribute(
            "state:angular:physics:position",
            Sdf.ValueTypeNames.Float,
            custom=False,
        )
        assert attribute.Set(value)
        return
    raise AssertionError(fact_kind)


def _bind_physics_material(
    stage_path: Path,
    *,
    owner_path: str,
    material_path: str,
    friction: float | None,
    all_purpose: bool = False,
    collection_name: str | None = None,
    collection_members: tuple[str, ...] = (),
) -> None:
    stage = Usd.Stage.Open(str(stage_path))
    UsdGeom.Scope.Define(stage, str(Sdf.Path(material_path).GetParentPath()))
    material = UsdShade.Material.Define(stage, material_path)
    if friction is not None:
        UsdPhysics.MaterialAPI.Apply(material.GetPrim()).CreateStaticFrictionAttr(
            friction
        )
    owner = stage.GetPrimAtPath(owner_path)
    binding_api = UsdShade.MaterialBindingAPI.Apply(owner)
    if collection_name is None:
        if all_purpose:
            assert binding_api.Bind(material)
        else:
            assert binding_api.Bind(material, materialPurpose="physics")
    else:
        collection = Usd.CollectionAPI.Apply(owner, collection_name)
        assert collection.GetIncludesRel().SetTargets(collection_members)
        assert binding_api.Bind(
            collection,
            material,
            bindingName=f"{collection_name}Binding",
            materialPurpose="physics",
        )
    assert stage.GetRootLayer().Save()


def _remove_drive_schema_and_properties(prim: Any, instance: str) -> None:
    prim.RemoveAppliedSchema(f"PhysicsDriveAPI:{instance}")
    prefix = f"drive:{instance}:"
    for prop in tuple(prim.GetAuthoredProperties()):
        name = str(prop.GetName())
        if name.startswith(prefix):
            prim.RemoveProperty(name)


def _remove_rigid_body_schema_and_properties(prim: Any) -> None:
    prim.RemoveAppliedSchema("PhysicsRigidBodyAPI")
    for name in (
        "physics:rigidBodyEnabled",
        "physics:kinematicEnabled",
        "physics:startsAsleep",
        "physics:velocity",
        "physics:angularVelocity",
    ):
        prim.RemoveProperty(name)


def _remove_collision_schema_and_properties(prim: Any) -> None:
    prim.RemoveAppliedSchema("PhysicsCollisionAPI")
    prim.RemoveAppliedSchema("PhysicsMeshCollisionAPI")
    for name in ("physics:collisionEnabled", "physics:approximation"):
        prim.RemoveProperty(name)


def _define_mass_contributor(
    stage_path: Path,
    child_path: str,
    *,
    author_physics: bool,
    transform: Any | None = None,
    center_of_mass: tuple[float, float, float] = (1.0, 2.0, 3.0),
    principal_axes: tuple[float, float, float, float] = (-1.0, 0.0, 0.0, 0.0),
    missing: str | None = None,
    collision: bool = True,
    instanceable: bool = False,
    time_sampled_transform: bool = False,
) -> None:
    stage = Usd.Stage.Open(str(stage_path))
    child = UsdGeom.Cube.Define(stage, child_path)
    if transform is not None:
        operation = child.AddTransformOp()
        assert operation.Set(transform)
        if time_sampled_transform:
            assert operation.Set(transform, Usd.TimeCode(1.0))
    assert child.GetPrim().SetInstanceable(instanceable)
    if author_physics:
        if collision:
            UsdPhysics.CollisionAPI.Apply(child.GetPrim()).CreateCollisionEnabledAttr(
                True
            )
        mass = UsdPhysics.MassAPI.Apply(child.GetPrim())
        if missing != "mass":
            mass.CreateMassAttr(2.0)
        if missing != "centerOfMass":
            mass.CreateCenterOfMassAttr(Gf.Vec3f(*center_of_mass))
        if missing != "diagonalInertia":
            mass.CreateDiagonalInertiaAttr(Gf.Vec3f(100.0, 150.0, 200.0))
        if missing != "principalAxes":
            real, x, y, z = principal_axes
            mass.CreatePrincipalAxesAttr(Gf.Quatf(real, Gf.Vec3f(x, y, z)))
    assert stage.GetRootLayer().Save()


def _static_extractor_attribute(
    stage: Any,
    family: str,
    field: str,
) -> tuple[Any, Any]:
    drawer_prim = stage.GetPrimAtPath("/World/Joints/drawer")
    drawer = UsdPhysics.PrismaticJoint(drawer_prim)
    if family == "joint_frame":
        getters = {
            "axis": drawer.GetAxisAttr,
            "localRot0": drawer.GetLocalRot0Attr,
            "localRot1": drawer.GetLocalRot1Attr,
        }
        values = {
            "axis": "X",
            "localRot0": Gf.Quatf(1.0, Gf.Vec3f(0.0)),
            "localRot1": Gf.Quatf(1.0, Gf.Vec3f(0.0)),
        }
        return getters[field](), values[field]
    if family == "anchor":
        getters = {
            "localPos0": drawer.GetLocalPos0Attr,
            "localPos1": drawer.GetLocalPos1Attr,
        }
        return getters[field](), Gf.Vec3f(10.0, 0.0, 0.0)
    if family == "limit":
        getters = {
            "lowerLimit": drawer.GetLowerLimitAttr,
            "upperLimit": drawer.GetUpperLimitAttr,
        }
        values = {"lowerLimit": 0.0, "upperLimit": 50.0}
        return getters[field](), values[field]
    if family == "drive":
        drive = UsdPhysics.DriveAPI.Get(drawer_prim, "linear")
        getters = {
            "drive_type": drive.GetTypeAttr,
            "stiffness": drive.GetStiffnessAttr,
            "damping": drive.GetDampingAttr,
            "max_force": drive.GetMaxForceAttr,
            "target_position": drive.GetTargetPositionAttr,
            "target_velocity": drive.GetTargetVelocityAttr,
        }
        attribute = getters[field]()
        return attribute, attribute.Get()
    if family == "physx_velocity":
        assert drawer_prim.AddAppliedSchema("PhysxJointAPI")
        attribute = drawer_prim.CreateAttribute(
            "physxJoint:maxJointVelocity",
            Sdf.ValueTypeNames.Float,
            custom=False,
        )
        return attribute, 3.5
    if family == "physx_friction":
        assert drawer_prim.AddAppliedSchema("PhysxJointAPI")
        attribute = drawer_prim.CreateAttribute(
            "physxJoint:jointFriction",
            Sdf.ValueTypeNames.Float,
            custom=False,
        )
        return attribute, 0.15
    if family == "mass":
        mass = UsdPhysics.MassAPI(stage.GetPrimAtPath("/World/base"))
        if field == "principalAxes":
            attribute = mass.CreatePrincipalAxesAttr(Gf.Quatf(1.0, Gf.Vec3f(0.0)))
        elif field == "centerOfMass":
            attribute = mass.CreateCenterOfMassAttr(Gf.Vec3f(1.0, 2.0, 3.0))
        else:
            getters = {
                "mass": mass.GetMassAttr,
                "diagonalInertia": mass.GetDiagonalInertiaAttr,
            }
            attribute = getters[field]()
        return attribute, attribute.Get()
    if family == "mesh_collision":
        mesh = UsdPhysics.MeshCollisionAPI.Apply(stage.GetPrimAtPath("/World/base"))
        attribute = mesh.CreateApproximationAttr("convexHull")
        return attribute, "convexHull"
    raise AssertionError(family)  # pragma: no cover - parameter guard


def _author_time_sample(attribute: Any, value: Any) -> None:
    attribute.SetVariability(Sdf.VariabilityVarying)
    assert attribute.Set(value)
    assert attribute.Set(value, Usd.TimeCode(1.0))
    assert attribute.GetTimeSamples() == [1.0]


def _mutate_pair(source: Path, reference: Path, mutation: str) -> None:
    source_stage = Usd.Stage.Open(str(source))
    reference_stage = Usd.Stage.Open(str(reference))
    drawer = UsdPhysics.PrismaticJoint(
        reference_stage.GetPrimAtPath("/World/Joints/drawer")
    )
    if mutation == "missing_body0":
        drawer.GetBody0Rel().ClearTargets(True)
    elif mutation == "multiple_body0":
        drawer.GetBody0Rel().SetTargets(["/World/base", "/World/door"])
    elif mutation == "same_endpoints":
        drawer.GetBody0Rel().SetTargets(["/World/drawer"])
    elif mutation == "missing_source_endpoint":
        source_stage.RemovePrim("/World/drawer")
    elif mutation == "missing_axis":
        drawer.GetAxisAttr().Clear()
    elif mutation == "contradictory_axis_frames":
        for stage in (source_stage, reference_stage):
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/drawer")).AddRotateZOp().Set(
                90.0
            )
    elif mutation == "partial_anchor":
        drawer.GetLocalPos1Attr().Clear()
    elif mutation == "invalid_limit_range":
        drawer.GetLowerLimitAttr().Set(75.0)
        drawer.GetUpperLimitAttr().Set(25.0)
    else:  # pragma: no cover - parameter guard
        raise AssertionError(mutation)
    source_stage.GetRootLayer().Save()
    reference_stage.GetRootLayer().Save()


def _joint(value: JointRiggerInputV1, joint_id: str) -> Any:
    return next(
        item for item in value.plan.joints if item.topology.joint_id == joint_id
    )


def _extract(
    source: Path,
    reference: Path,
    **kwargs: Any,
) -> JointRiggerInputV1:
    return extract_reference_input(
        source,
        reference,
        source_uri=SOURCE_URI,
        reference_uri=REFERENCE_URI,
        **kwargs,
    )


def _identity(path: Path) -> ArtifactIdentityV1:
    return ArtifactIdentityV1(uri="fixture://identity", root_sha256=_sha256(path))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
