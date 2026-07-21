# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for evidence-backed shared Joint Rigger physics schema authoring."""

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import pytest
from pydantic import ValidationError

Ar = pytest.importorskip("pxr.Ar")
Gf = pytest.importorskip("pxr.Gf")
Sdf = pytest.importorskip("pxr.Sdf")
Tf = pytest.importorskip("pxr.Tf")
Ts = pytest.importorskip("pxr.Ts")
Usd = pytest.importorskip("pxr.Usd")
UsdGeom = pytest.importorskip("pxr.UsdGeom")
UsdPhysics = pytest.importorskip("pxr.UsdPhysics")

from world_understanding.functions.physics.joint_rigger import (  # noqa: E402
    PLAN_SCHEMA_VERSION,
    ArticulationRootPlanV1,
    ArtifactIdentityV1,
    ColliderPlanV1,
    FieldProvenanceV1,
    JointAnchorV1,
    JointDriveV1,
    JointFrictionV1,
    JointLimitV1,
    JointMimicV1,
    JointPlanV1,
    JointRiggerContractError,
    JointRiggerPlanV1,
    JointStateV1,
    JointTopologyV1,
    MassPropertiesV1,
    RigidBodyPlanV1,
    author_physics_schemas,
    capture_joint_rigger_physics_schema_snapshot,
    capture_joint_rigger_stage_snapshot,
    physics_schema_counts,
    validate_authored_physics_schemas,
    validate_joint_rigger_stage_preservation,
    validate_physics_plan_evidence,
)

schemas_module = importlib.import_module(
    "world_understanding.functions.physics.joint_rigger.schemas"
)
validation_module = importlib.import_module(
    "world_understanding.functions.physics.joint_rigger.validation"
)

_PROVENANCE = FieldProvenanceV1(
    source="owner_approved_plan",
    evidence="Fixture values explicitly approved for schema-authoring tests.",
)

_RAW_R3_AUTHORSHIP_CASES = (
    ("state", "custom"),
    ("state", "uniform_variability"),
    ("drive", "unexpected_metadata"),
    ("friction", "friction_unexpected_metadata"),
    ("mimic", "relationship_list_op"),
    ("drive", "api_order"),
    ("drive", "api_ordered_items"),
    ("body", "body_mass_custom"),
    ("body", "body_api_ordered_items"),
    ("root", "root_api_appended"),
    ("collider", "collider_unexpected_metadata"),
    ("collider", "collider_api_explicit"),
    ("body", "single_apply_instance_token"),
    ("drive", "bare_multi_apply_token"),
)


def _raw_r3_fixture(fixture_kind: str) -> tuple[Any, JointRiggerPlanV1, str]:
    if fixture_kind == "mimic":
        stage, plan = _mimic_fixture()
        path = "/World/Joints/second"
    else:
        stage, plan = _revolute_fixture(
            with_drive=fixture_kind in {"drive", "body", "root", "collider"},
            joint_friction=0.15 if fixture_kind == "friction" else None,
        )
        path = {
            "state": "/World/Joints/hinge",
            "drive": "/World/Joints/hinge",
            "friction": "/World/Joints/hinge",
            "body": "/World/base",
            "root": "/World/base",
            "collider": "/World/base/collision",
        }[fixture_kind]
    author_physics_schemas(stage, plan)
    return stage, plan, path


def _mutate_raw_r3_authorship(stage: Any, path: str, mutation: str) -> None:
    prim_spec = stage.GetRootLayer().GetPrimAtPath(path)
    assert prim_spec is not None
    if mutation == "custom":
        attribute = prim_spec.properties["state:angular:physics:position"]
        assert isinstance(attribute, Sdf.AttributeSpec)
        attribute.custom = True
    elif mutation == "uniform_variability":
        attribute = prim_spec.properties["state:angular:physics:position"]
        assert isinstance(attribute, Sdf.AttributeSpec)
        attribute.SetInfo("variability", Sdf.VariabilityUniform)
    elif mutation == "unexpected_metadata":
        attribute = prim_spec.properties["drive:angular:physics:stiffness"]
        assert isinstance(attribute, Sdf.AttributeSpec)
        attribute.SetInfo("documentation", "adversarial metadata")
    elif mutation == "friction_unexpected_metadata":
        attribute = prim_spec.properties["physxJoint:jointFriction"]
        assert isinstance(attribute, Sdf.AttributeSpec)
        attribute.SetInfo("documentation", "adversarial metadata")
    elif mutation == "relationship_list_op":
        relationship = prim_spec.properties["physxMimicJoint:rotZ:referenceJoint"]
        assert isinstance(relationship, Sdf.RelationshipSpec)
        target_list = relationship.GetInfo("targetPaths")
        assert isinstance(target_list, Sdf.PathListOp)
        explicit_targets = list(target_list.explicitItems)
        relationship.targetPathList.ClearEdits()
        relationship.targetPathList.prependedItems = explicit_targets
    elif mutation in {"api_order", "api_ordered_items"}:
        schemas = prim_spec.GetInfo("apiSchemas")
        assert isinstance(schemas, Sdf.TokenListOp)
        if mutation == "api_order":
            schemas.prependedItems = list(reversed(schemas.prependedItems))
        else:
            schemas.orderedItems = list(schemas.prependedItems)
        prim_spec.SetInfo("apiSchemas", schemas)
    elif mutation == "body_mass_custom":
        attribute = prim_spec.properties["physics:mass"]
        assert isinstance(attribute, Sdf.AttributeSpec)
        attribute.custom = True
    elif mutation == "body_api_ordered_items":
        schemas = prim_spec.GetInfo("apiSchemas")
        assert isinstance(schemas, Sdf.TokenListOp)
        schemas.orderedItems = list(schemas.prependedItems)
        prim_spec.SetInfo("apiSchemas", schemas)
    elif mutation == "root_api_appended":
        schemas = prim_spec.GetInfo("apiSchemas")
        assert isinstance(schemas, Sdf.TokenListOp)
        prepended = list(schemas.prependedItems)
        prepended.remove("PhysicsArticulationRootAPI")
        schemas.prependedItems = prepended
        schemas.appendedItems = ["PhysicsArticulationRootAPI"]
        prim_spec.SetInfo("apiSchemas", schemas)
    elif mutation == "collider_unexpected_metadata":
        attribute = prim_spec.properties["physics:collisionEnabled"]
        assert isinstance(attribute, Sdf.AttributeSpec)
        attribute.SetInfo("documentation", "adversarial collider metadata")
    elif mutation == "collider_api_explicit":
        schemas = prim_spec.GetInfo("apiSchemas")
        assert isinstance(schemas, Sdf.TokenListOp)
        prim_spec.SetInfo(
            "apiSchemas",
            Sdf.TokenListOp.CreateExplicit(list(schemas.prependedItems)),
        )
    elif mutation in {"single_apply_instance_token", "bare_multi_apply_token"}:
        schemas = prim_spec.GetInfo("apiSchemas")
        assert isinstance(schemas, Sdf.TokenListOp)
        malformed_token = (
            "PhysicsRigidBodyAPI:adversarial"
            if mutation == "single_apply_instance_token"
            else "PhysicsDriveAPI"
        )
        schemas.prependedItems = [*schemas.prependedItems, malformed_token]
        prim_spec.SetInfo("apiSchemas", schemas)
    else:  # pragma: no cover - parameter table and helper stay in lockstep
        raise AssertionError(f"unhandled raw R3 mutation: {mutation}")


@pytest.mark.parametrize(("fixture_kind", "mutation"), _RAW_R3_AUTHORSHIP_CASES)
def test_public_schema_validator_rejects_noncanonical_raw_r3_authorship(
    fixture_kind: str,
    mutation: str,
) -> None:
    stage, plan, path = _raw_r3_fixture(fixture_kind)
    _mutate_raw_r3_authorship(stage, path, mutation)
    malformed = stage.GetRootLayer().ExportToString()

    with pytest.raises(JointRiggerContractError) as caught:
        validate_authored_physics_schemas(stage, plan)

    assert caught.value.code == "authored_graph_mismatch"
    assert "raw" in caught.value.detail
    assert stage.GetRootLayer().ExportToString() == malformed


@pytest.mark.parametrize(("fixture_kind", "mutation"), _RAW_R3_AUTHORSHIP_CASES)
def test_idempotent_schema_authoring_enforces_canonical_raw_r3_authorship(
    fixture_kind: str,
    mutation: str,
) -> None:
    stage, plan, path = _raw_r3_fixture(fixture_kind)
    _mutate_raw_r3_authorship(stage, path, mutation)
    malformed = stage.GetRootLayer().ExportToString()

    if mutation in {"relationship_list_op", "collider_api_explicit"}:
        # SetTargets and the validated explicit-source rule deliberately
        # canonicalize these two raw list-ops.
        author_physics_schemas(stage, plan)
        assert stage.GetRootLayer().ExportToString() != malformed
        validate_authored_physics_schemas(stage, plan)
    else:
        with pytest.raises(JointRiggerContractError) as caught:
            author_physics_schemas(stage, plan)

        normalization_rejections = {
            "api_ordered_items",
            "body_api_ordered_items",
            "root_api_appended",
        }
        if mutation in normalization_rejections:
            assert caught.value.code == "physics_schema_list_op_ambiguous"
            assert "ambiguous apiSchemas opinion" in caught.value.detail
        else:
            assert caught.value.code == "authored_graph_mismatch"
            assert "raw" in caught.value.detail
        assert stage.GetRootLayer().ExportToString() == malformed


def test_author_physics_schemas_applies_exact_owned_subset_idempotently() -> None:
    stage, plan = _revolute_fixture(with_drive=True)
    before = capture_joint_rigger_stage_snapshot(stage)
    assert physics_schema_counts(stage) == {}

    diagnostics = author_physics_schemas(stage, plan)

    after = capture_joint_rigger_stage_snapshot(stage)
    validate_joint_rigger_stage_preservation(before, after)
    assert diagnostics.backend_name == "world_understanding.physics_schemas"
    assert diagnostics.errors == ()
    assert {item.field for item in diagnostics.field_decisions} == {
        "articulation_root",
        "rigid_bodies[/World/base].colliders[/World/base/collision].collision",
        "rigid_bodies[/World/base].colliders[/World/base/collision].mesh_approximation",
        "rigid_bodies[/World/base].colliders[/World/base/collision].mesh_collision_api",
        "rigid_bodies[/World/base].mass.center_of_mass_m",
        "rigid_bodies[/World/base].mass.diagonal_inertia_kg_m2",
        "rigid_bodies[/World/base].mass.mass_kg",
        "rigid_bodies[/World/base].mass.principal_axes",
        "rigid_bodies[/World/base].rigid_body",
        "rigid_bodies[/World/link].colliders[/World/link/collision].collision",
        "rigid_bodies[/World/link].colliders[/World/link/collision].mesh_approximation",
        "rigid_bodies[/World/link].colliders[/World/link/collision].mesh_collision_api",
        "rigid_bodies[/World/link].mass.center_of_mass_m",
        "rigid_bodies[/World/link].mass.diagonal_inertia_kg_m2",
        "rigid_bodies[/World/link].mass.mass_kg",
        "rigid_bodies[/World/link].mass.principal_axes",
        "rigid_bodies[/World/link].rigid_body",
    }
    for decision in diagnostics.field_decisions:
        if decision.disposition == "accepted":
            assert decision.provenance == _PROVENANCE
    joint_diagnostic = diagnostics.joint_diagnostics[0]
    assert joint_diagnostic.authored_prim_path == "/World/Joints/hinge"
    assert (
        "authored_prim_path"
        not in diagnostics.model_dump(mode="json")["joint_diagnostics"][0]
    )
    decisions = {item.field: item for item in joint_diagnostic.field_decisions}
    assert set(decisions) == {
        "anchor.position_stage",
        "drive.damping",
        "drive.drive_type",
        "drive.max_force",
        "drive.max_joint_velocity",
        "drive.stiffness",
        "drive.target_position",
        "drive.target_velocity",
        "joint_friction.coefficient",
        "limit.lower",
        "limit.unit",
        "limit.upper",
        "mimic.damping_ratio",
        "mimic.gearing",
        "mimic.natural_frequency",
        "mimic.offset",
        "mimic.reference_joint_id",
        "state.position",
        "state.velocity",
        "topology.axis_stage",
        "topology.body0",
        "topology.body1",
        "topology.joint_type",
        "usd.joint_prim_path",
    }
    assert decisions["state.position"].disposition == "accepted"
    assert decisions["joint_friction.coefficient"].disposition == "ignored"
    assert decisions["joint_friction.coefficient"].reason_code == "not_planned"
    assert decisions["drive.stiffness"].disposition == "accepted"
    assert decisions["mimic.reference_joint_id"].disposition == "ignored"
    assert decisions["mimic.reference_joint_id"].reason_code == "not_planned"
    for decision in decisions.values():
        if decision.disposition == "accepted":
            assert decision.provenance == _PROVENANCE

    base = stage.GetPrimAtPath("/World/base")
    link = stage.GetPrimAtPath("/World/link")
    for prim in (base, link):
        assert prim.HasAPI(UsdPhysics.RigidBodyAPI)
        assert prim.HasAPI(UsdPhysics.MassAPI)
        assert prim.GetAttribute("physics:rigidBodyEnabled").Get() is True
        assert prim.GetAttribute("physics:kinematicEnabled").Get() is False
        assert prim.GetAttribute("physics:mass").Get() == pytest.approx(2.0)
        assert tuple(prim.GetAttribute("physics:diagonalInertia").Get()) == (
            pytest.approx(100.0),
            pytest.approx(150.0),
            pytest.approx(200.0),
        )
        collider = stage.GetPrimAtPath(f"{prim.GetPath()}/collision")
        assert collider.HasAPI(UsdPhysics.CollisionAPI)
        assert collider.GetAttribute("physics:collisionEnabled").Get() is True
    assert base.HasAPI(UsdPhysics.ArticulationRootAPI)
    assert not link.HasAPI(UsdPhysics.ArticulationRootAPI)

    joint = stage.GetPrimAtPath("/World/Joints/hinge")
    schemas = _schema_tokens(joint)
    assert "PhysicsJointStateAPI:angular" in schemas
    assert "PhysicsDriveAPI:angular" in schemas
    assert "PhysxJointAPI" in schemas
    assert joint.GetAttribute("state:angular:physics:position").Get() == 0.0
    assert joint.GetAttribute("state:angular:physics:velocity").Get() == 0.0
    assert joint.GetAttribute("drive:angular:physics:type").Get() == "force"
    assert joint.GetAttribute("physxJoint:maxJointVelocity").Get() == 3.0
    assert not any(prim.IsA(UsdPhysics.Scene) for prim in stage.TraverseAll())
    assert physics_schema_counts(stage) == {
        "PhysicsArticulationRootAPI": 1,
        "PhysicsCollisionAPI": 2,
        "PhysicsDriveAPI:angular": 1,
        "PhysicsJointStateAPI:angular": 1,
        "PhysicsMassAPI": 2,
        "PhysicsRigidBodyAPI": 2,
        "PhysxJointAPI": 1,
    }
    schema_snapshot = capture_joint_rigger_physics_schema_snapshot(stage)
    assert validate_authored_physics_schemas(stage, plan) == diagnostics
    assert capture_joint_rigger_physics_schema_snapshot(stage) == schema_snapshot
    none_version_diagnostics = author_physics_schemas(
        stage,
        plan,
        backend_version=None,
    )
    assert none_version_diagnostics.backend_version is None
    none_version_validation = validate_authored_physics_schemas(
        stage,
        plan,
        backend_version=None,
    )
    assert none_version_validation.backend_version is None

    first_authored = stage.GetRootLayer().ExportToString()
    counts = physics_schema_counts(stage)
    second_diagnostics = author_physics_schemas(stage, plan)
    assert second_diagnostics == diagnostics
    assert physics_schema_counts(stage) == counts
    assert stage.GetRootLayer().ExportToString() == first_authored


@pytest.mark.parametrize(
    ("tamper", "reason_code"),
    [
        ("value", "physics_schema_conflict"),
        ("time_sample", "time_sampled_owned_property"),
        ("remove", "postwrite_validation_failed"),
    ],
)
def test_center_of_mass_authors_decides_and_rejects_readback_drift(
    tamper: str,
    reason_code: str,
) -> None:
    stage, original_plan = _revolute_fixture(with_drive=True)
    body = original_plan.rigid_bodies[0]
    assert body.mass is not None
    mass = body.mass.model_copy(update={"center_of_mass_m": (0.01, -0.02, 0.03)})
    plan = _replace_first_body(original_plan, mass=mass)

    diagnostics = author_physics_schemas(stage, plan)

    prim = stage.GetPrimAtPath(body.prim_path)
    center = prim.GetAttribute("physics:centerOfMass")
    assert str(center.GetTypeName()) == "point3f"
    assert tuple(center.Get()) == pytest.approx((1.0, -2.0, 3.0))
    decision = next(
        item
        for item in diagnostics.field_decisions
        if item.field == f"rigid_bodies[{body.prim_path}].mass.center_of_mass_m"
    )
    assert decision.disposition == "accepted"
    assert decision.provenance == mass.provenance
    assert validate_authored_physics_schemas(stage, plan) == diagnostics

    if tamper == "value":
        assert center.Set(Gf.Vec3f(1.0, -2.0, 3.5))
    elif tamper == "time_sample":
        assert center.Set(Gf.Vec3f(1.0, -2.0, 3.0), Usd.TimeCode(1.0))
    else:
        assert prim.RemoveProperty("physics:centerOfMass")

    with pytest.raises(JointRiggerContractError) as caught:
        validate_authored_physics_schemas(stage, plan)

    assert caught.value.code == reason_code


def test_lifted_descendant_mass_plan_preflights_applies_validates_and_replays() -> None:
    stage, original_plan = _revolute_fixture(with_drive=True)
    body = original_plan.rigid_bodies[0]
    assert body.mass is not None
    contributor_path = f"{body.prim_path}/collision"
    lifted_provenance = FieldProvenanceV1(
        source="authored_reference",
        artifact=ArtifactIdentityV1(
            uri="fixture://lifted-mass/reference.usda",
            root_sha256="a" * 64,
        ),
        prim_path=contributor_path,
        properties=(
            "physics:mass",
            "physics:centerOfMass",
            "physics:diagonalInertia",
            "physics:principalAxes",
        ),
        derivation="rigid_body_frame_lift(contributor_to_owner)",
        evidence="Complete descendant collider mass frame.",
    )
    lifted_mass = MassPropertiesV1(
        mass_kg=body.mass.mass_kg,
        center_of_mass_m=(0.01, -0.02, 0.03),
        diagonal_inertia_kg_m2=body.mass.diagonal_inertia_kg_m2,
        principal_axes=(math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)),
        provenance=lifted_provenance,
    )
    plan = _replace_first_body(original_plan, mass=lifted_mass)
    contributor = stage.GetPrimAtPath(contributor_path)
    assert not contributor.HasAPI(UsdPhysics.MassAPI)

    validate_physics_plan_evidence(plan)
    diagnostics = author_physics_schemas(stage, plan)

    owner = stage.GetPrimAtPath(body.prim_path)
    assert owner.HasAPI(UsdPhysics.MassAPI)
    assert tuple(owner.GetAttribute("physics:centerOfMass").Get()) == pytest.approx(
        (1.0, -2.0, 3.0)
    )
    authored_axes = owner.GetAttribute("physics:principalAxes").Get()
    assert authored_axes.GetReal() == pytest.approx(math.sqrt(0.5))
    assert tuple(authored_axes.GetImaginary()) == pytest.approx(
        (0.0, 0.0, math.sqrt(0.5))
    )
    assert not contributor.HasAPI(UsdPhysics.MassAPI)
    assert validate_authored_physics_schemas(stage, plan) == diagnostics

    first_authored = stage.GetRootLayer().ExportToString()
    assert author_physics_schemas(stage, plan) == diagnostics
    assert validate_authored_physics_schemas(stage, plan) == diagnostics
    assert stage.GetRootLayer().ExportToString() == first_authored


def test_author_normalizes_foreign_explicit_api_before_owned_prepends() -> None:
    stage, plan = _revolute_fixture(with_drive=True)
    base = stage.GetPrimAtPath("/World/base")
    base_spec = stage.GetRootLayer().GetPrimAtPath("/World/base")
    assert base_spec is not None
    base_spec.SetInfo(
        "apiSchemas",
        Sdf.TokenListOp.CreateExplicit(["IsaacLinkAPI"]),
    )
    assert tuple(base.GetMetadata("apiSchemas").GetAppliedItems()) == ("IsaacLinkAPI",)

    diagnostics = author_physics_schemas(stage, plan)

    schemas = base_spec.GetInfo("apiSchemas")
    assert isinstance(schemas, Sdf.TokenListOp)
    assert not schemas.isExplicit
    assert tuple(str(item) for item in schemas.prependedItems) == (
        "IsaacLinkAPI",
        "PhysicsRigidBodyAPI",
        "PhysicsMassAPI",
        "PhysicsArticulationRootAPI",
    )
    assert not tuple(schemas.explicitItems)
    assert not tuple(schemas.addedItems)
    assert not tuple(schemas.appendedItems)
    assert not tuple(schemas.deletedItems)
    assert not tuple(schemas.orderedItems)
    assert tuple(base.GetMetadata("apiSchemas").GetAppliedItems()) == (
        "IsaacLinkAPI",
        "PhysicsRigidBodyAPI",
        "PhysicsMassAPI",
        "PhysicsArticulationRootAPI",
    )
    assert validate_authored_physics_schemas(stage, plan) == diagnostics
    first_authored = stage.GetRootLayer().ExportToString()
    assert author_physics_schemas(stage, plan) == diagnostics
    assert stage.GetRootLayer().ExportToString() == first_authored


def test_composed_raw_api_schema_items_is_empty_without_metadata() -> None:
    stage = Usd.Stage.CreateInMemory()
    prim = UsdGeom.Xform.Define(stage, "/World/plain").GetPrim()
    assert prim.GetMetadata("apiSchemas") is None

    assert schemas_module._composed_raw_api_schema_items(prim) == ()


def test_normalizer_rejects_scene_path_outside_mapped_edit_target() -> None:
    asset_stage = Usd.Stage.CreateInMemory()
    asset_root = UsdGeom.Xform.Define(asset_stage, "/Asset").GetPrim()
    asset_stage.SetDefaultPrim(asset_root)
    stage = Usd.Stage.CreateInMemory()
    scene_root = UsdGeom.Xform.Define(stage, "/Scene").GetPrim()
    assert scene_root.GetReferences().AddReference(
        asset_stage.GetRootLayer().identifier,
        "/Asset",
    )
    reference_nodes = scene_root.GetPrimIndex().rootNode.children
    assert len(reference_nodes) == 1
    edit_target = Usd.EditTarget(asset_stage.GetRootLayer(), reference_nodes[0])
    assert edit_target.MapToSpecPath(Sdf.Path("/Outside")) == Sdf.Path.emptyPath
    stage.SetEditTarget(edit_target)
    contract = schemas_module._R3RawAuthorshipContract(
        schema_order=("PhysicsRigidBodyAPI",),
        attribute_specs={},
        attribute_defaults={},
        relationship_targets={},
    )

    with pytest.raises(JointRiggerContractError) as caught:
        schemas_module._normalize_compatible_explicit_api_schemas(
            stage,
            {"/Outside": contract},
        )

    assert caught.value.code == "physics_schema_list_op_ambiguous"
    assert "cannot be mapped into the active edit target" in caught.value.detail


def test_normalizer_skips_non_token_list_op_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_pxr = importlib.import_module("pxr")
    writes: list[tuple[str, Any]] = []

    class UnexpectedMetadataPrimSpec:
        def ListInfoKeys(self) -> tuple[str, ...]:
            return ("apiSchemas",)

        def GetInfo(self, name: str) -> tuple[str, ...]:
            assert name == "apiSchemas"
            return ("PhysicsRigidBodyAPI",)

        def SetInfo(self, name: str, value: Any) -> None:
            writes.append((name, value))

    prim_spec = UnexpectedMetadataPrimSpec()
    layer = SimpleNamespace(GetPrimAtPath=lambda _path: prim_spec)
    edit_target = SimpleNamespace(
        GetLayer=lambda: layer,
        MapToSpecPath=lambda path: path,
    )
    stage = SimpleNamespace(GetEditTarget=lambda: edit_target)
    sdf_runtime_shim = SimpleNamespace(
        Path=Sdf.Path,
        PrimSpec=UnexpectedMetadataPrimSpec,
        TokenListOp=Sdf.TokenListOp,
    )
    monkeypatch.setattr(real_pxr, "Sdf", sdf_runtime_shim)
    contract = schemas_module._R3RawAuthorshipContract(
        schema_order=("PhysicsRigidBodyAPI",),
        attribute_specs={},
        attribute_defaults={},
        relationship_targets={},
    )

    schemas_module._normalize_compatible_explicit_api_schemas(
        stage,
        {"/World/base": contract},
    )

    assert writes == []


def test_author_rolls_back_when_foreign_schema_normalization_changes_composition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage, plan = _revolute_fixture(with_drive=True)
    base_spec = stage.GetRootLayer().GetPrimAtPath("/World/base")
    assert base_spec is not None
    base_spec.SetInfo(
        "apiSchemas",
        Sdf.TokenListOp.CreateExplicit(["IsaacLinkAPI"]),
    )
    before = stage.GetRootLayer().ExportToString()
    real_composed_items = schemas_module._composed_raw_api_schema_items
    base_results = iter(
        [
            ("IsaacLinkAPI",),
            ("IsaacLinkAPI", "UnexpectedSchemaAPI"),
        ]
    )

    def composed_items(prim: Any) -> tuple[str, ...]:
        if str(prim.GetPath()) == "/World/base":
            return next(base_results)
        return real_composed_items(prim)

    monkeypatch.setattr(
        schemas_module,
        "_composed_raw_api_schema_items",
        composed_items,
    )

    with pytest.raises(JointRiggerContractError) as caught:
        author_physics_schemas(stage, plan)

    assert caught.value.code == "physics_schema_list_op_ambiguous"
    assert "changed composed tokens" in caught.value.detail
    assert "UnexpectedSchemaAPI" in caught.value.detail
    assert stage.GetRootLayer().ExportToString() == before


def test_normalizer_canonicalizes_compatible_plan_owned_explicit_schemas() -> None:
    stage, plan = _revolute_fixture(with_drive=True)
    base = stage.GetPrimAtPath("/World/base")
    base_spec = stage.GetRootLayer().GetPrimAtPath("/World/base")
    assert base_spec is not None
    base_spec.SetInfo(
        "apiSchemas",
        Sdf.TokenListOp.CreateExplicit(["IsaacLinkAPI", "PhysicsRigidBodyAPI"]),
    )
    before_applied = tuple(base.GetMetadata("apiSchemas").GetAppliedItems())
    preflight = schemas_module._preflight(stage, plan)

    schemas_module._normalize_compatible_explicit_api_schemas(
        stage,
        schemas_module._r3_raw_authorship_contract(stage, plan, preflight),
    )

    normalized = base_spec.GetInfo("apiSchemas")
    assert isinstance(normalized, Sdf.TokenListOp)
    assert not normalized.isExplicit
    assert tuple(str(token) for token in normalized.prependedItems) == (
        "IsaacLinkAPI",
        "PhysicsRigidBodyAPI",
    )
    assert tuple(base.GetMetadata("apiSchemas").GetAppliedItems()) == before_applied


def test_author_preserves_valid_explicit_source_collider_mass_contract() -> None:
    stage, plan = _revolute_fixture()
    path = "/World/base/collision"
    prim = stage.GetPrimAtPath(path)
    prim_spec = stage.GetRootLayer().GetPrimAtPath(path)
    assert prim_spec is not None
    prim_spec.SetInfo(
        "apiSchemas",
        Sdf.TokenListOp.CreateExplicit(
            [
                "PhysicsCollisionAPI",
                "NewtonCollisionAPI",
                "PhysicsMassAPI",
                "MaterialBindingAPI",
            ]
        ),
    )
    prim.CreateAttribute(
        "physics:mass",
        Sdf.ValueTypeNames.Float,
        custom=False,
    ).Set(0.2)
    prim.CreateAttribute(
        "physics:density",
        Sdf.ValueTypeNames.Float,
        custom=False,
    ).Set(10.0)

    diagnostics = author_physics_schemas(stage, plan)

    schemas = prim_spec.GetInfo("apiSchemas")
    assert isinstance(schemas, Sdf.TokenListOp)
    assert not schemas.isExplicit
    assert tuple(str(token) for token in schemas.prependedItems) == (
        "PhysicsCollisionAPI",
        "NewtonCollisionAPI",
        "PhysicsMassAPI",
        "MaterialBindingAPI",
    )
    assert prim.GetAttribute("physics:mass").Get() == pytest.approx(0.2)
    assert prim.GetAttribute("physics:density").Get() == pytest.approx(10.0)
    assert validate_authored_physics_schemas(stage, plan) == diagnostics
    first_authored = stage.GetRootLayer().ExportToString()
    assert author_physics_schemas(stage, plan) == diagnostics
    assert stage.GetRootLayer().ExportToString() == first_authored


@pytest.mark.parametrize(
    "violation",
    [
        "unexpected_schema",
        "blocked",
        "connected",
        "time_varying",
        "nonpositive",
        "unsupported_inertia",
        "ambiguous_list_op",
    ],
)
def test_incompatible_explicit_source_collider_physics_rolls_back(
    violation: str,
) -> None:
    stage, plan = _revolute_fixture()
    path = "/World/base/collision"
    prim = stage.GetPrimAtPath(path)
    prim_spec = stage.GetRootLayer().GetPrimAtPath(path)
    assert prim_spec is not None
    source_tokens = ["PhysicsCollisionAPI", "PhysicsMassAPI"]
    if violation == "unexpected_schema":
        source_tokens.append("PhysicsRigidBodyAPI")
    prim_spec.SetInfo(
        "apiSchemas",
        Sdf.TokenListOp.CreateExplicit(source_tokens),
    )
    mass = prim.CreateAttribute(
        "physics:mass",
        Sdf.ValueTypeNames.Float,
        custom=False,
    )
    mass.Set(0.2)
    if violation == "blocked":
        mass.Block()
    elif violation == "connected":
        driver = prim.CreateAttribute(
            "source:mass",
            Sdf.ValueTypeNames.Float,
        )
        driver.Set(0.2)
        mass.AddConnection(driver.GetPath())
    elif violation == "time_varying":
        mass.Set(0.3, Usd.TimeCode(1.0))
    elif violation == "nonpositive":
        mass.Set(0.0)
    elif violation == "unsupported_inertia":
        prim.CreateAttribute(
            "physics:diagonalInertia",
            Sdf.ValueTypeNames.Float3,
            custom=False,
        ).Set((1.0, 1.0, 1.0))
    elif violation == "ambiguous_list_op":
        schemas = prim_spec.GetInfo("apiSchemas")
        assert isinstance(schemas, Sdf.TokenListOp)
        schemas.appendedItems = ["MaterialBindingAPI"]
        prim_spec.SetInfo("apiSchemas", schemas)
    before = stage.GetRootLayer().ExportToString()

    with pytest.raises(JointRiggerContractError):
        author_physics_schemas(stage, plan)

    assert stage.GetRootLayer().ExportToString() == before


@pytest.mark.parametrize("shape", ["duplicate", "mixed"])
def test_normalizer_rejects_ambiguous_api_schema_list_ops(shape: str) -> None:
    stage, plan = _revolute_fixture(with_drive=True)
    base_spec = stage.GetRootLayer().GetPrimAtPath("/World/base")
    assert base_spec is not None
    schemas = Sdf.TokenListOp()
    if shape == "duplicate":
        schemas.addedItems = ["IsaacLinkAPI", "IsaacLinkAPI"]
    else:
        schemas.prependedItems = ["IsaacLinkAPI"]
        schemas.appendedItems = ["MaterialBindingAPI"]
    base_spec.SetInfo("apiSchemas", schemas)
    preflight = schemas_module._preflight(stage, plan)

    with pytest.raises(JointRiggerContractError) as caught:
        schemas_module._normalize_compatible_explicit_api_schemas(
            stage,
            schemas_module._r3_raw_authorship_contract(stage, plan, preflight),
        )

    assert caught.value.code == "physics_schema_list_op_ambiguous"


def test_author_rolls_back_when_foreign_explicit_normalization_unmasks_tokens() -> None:
    stage, plan = _revolute_fixture(with_drive=True)
    weaker = Sdf.Layer.CreateAnonymous("weaker-foreign-schema.usda")
    stage.GetRootLayer().subLayerPaths.append(weaker.identifier)
    weaker_spec = Sdf.CreatePrimInLayer(weaker, "/World/base")
    weaker_spec.SetInfo(
        "apiSchemas",
        Sdf.TokenListOp.Create(prependedItems=["MaterialBindingAPI"]),
    )
    base_spec = stage.GetRootLayer().GetPrimAtPath("/World/base")
    assert base_spec is not None
    base_spec.SetInfo(
        "apiSchemas",
        Sdf.TokenListOp.CreateExplicit(["CollectionAPI:source"]),
    )
    before = stage.GetRootLayer().ExportToString()
    weaker_before = weaker.ExportToString()

    with pytest.raises(JointRiggerContractError) as caught:
        author_physics_schemas(stage, plan)

    assert caught.value.code == "authored_graph_mismatch"
    assert "raw apiSchemas list-op" in caught.value.detail
    assert stage.GetRootLayer().ExportToString() == before
    assert weaker.ExportToString() == weaker_before


@pytest.mark.parametrize(
    ("field", "attribute_name"),
    [
        ("stiffness", "drive:angular:physics:stiffness"),
        ("damping", "drive:angular:physics:damping"),
        ("max_force", "drive:angular:physics:maxForce"),
        ("max_joint_velocity", "physxJoint:maxJointVelocity"),
    ],
)
def test_zero_drive_values_remain_authorable_under_the_public_contract(
    field: str,
    attribute_name: str,
) -> None:
    stage, plan = _revolute_fixture(with_drive=True)
    original_drive = plan.joints[0].drive
    assert original_drive is not None
    drive = JointDriveV1.model_validate(
        {
            **original_drive.model_dump(mode="python"),
            field: 0.0,
        }
    )
    plan = _replace_first_joint(plan, drive=drive)

    diagnostics = author_physics_schemas(stage, plan)

    joint = stage.GetPrimAtPath("/World/Joints/hinge")
    assert joint.GetAttribute(attribute_name).Get() == 0.0
    assert validate_authored_physics_schemas(stage, plan) == diagnostics


def test_joint_friction_round_trips_for_passive_and_driven_scalar_joints() -> None:
    fixtures = (
        (*_revolute_fixture(with_drive=False, joint_friction=0.0), "angular", False),
        (*_revolute_fixture(with_drive=True, joint_friction=0.15), "angular", True),
        (*_prismatic_fixture(with_drive=False, joint_friction=0.25), "linear", False),
        (*_prismatic_fixture(with_drive=True, joint_friction=0.5), "linear", True),
    )

    for stage, plan, motion, driven in fixtures:
        before = capture_joint_rigger_stage_snapshot(stage)
        diagnostics = author_physics_schemas(stage, plan)
        joint_path = plan.joints[0].topology.joint_id
        prim = stage.GetPrimAtPath(joint_path)
        schemas = _schema_tokens(prim)
        friction = plan.joints[0].joint_friction
        assert friction is not None
        assert "PhysxJointAPI" in schemas
        assert (f"PhysicsDriveAPI:{motion}" in schemas) is driven
        assert prim.GetAttribute("physxJoint:jointFriction").Get() == pytest.approx(
            friction.coefficient
        )
        decision = {
            item.field: item
            for item in diagnostics.joint_diagnostics[0].field_decisions
        }["joint_friction.coefficient"]
        assert decision.disposition == "accepted"
        assert decision.provenance == friction.provenance
        validate_joint_rigger_stage_preservation(
            before,
            capture_joint_rigger_stage_snapshot(stage),
        )
        assert validate_authored_physics_schemas(stage, plan) == diagnostics
        authored = stage.GetRootLayer().ExportToString()
        assert author_physics_schemas(stage, plan) == diagnostics
        assert stage.GetRootLayer().ExportToString() == authored


@pytest.mark.parametrize(
    "tamper",
    ["value", "missing", "time_sample", "unknown_property", "unknown_schema"],
)
def test_joint_friction_schema_readback_rejects_every_owned_physx_drift(
    tamper: str,
) -> None:
    stage, plan = _revolute_fixture(with_drive=False, joint_friction=0.15)
    author_physics_schemas(stage, plan)
    prim = stage.GetPrimAtPath("/World/Joints/hinge")
    friction = prim.GetAttribute("physxJoint:jointFriction")
    if tamper == "value":
        friction.Set(0.25)
    elif tamper == "missing":
        prim.RemoveProperty("physxJoint:jointFriction")
    elif tamper == "time_sample":
        friction.Set(0.15, Usd.TimeCode(1.0))
    elif tamper == "unknown_property":
        prim.CreateAttribute(
            "physxJoint:solverFoo",
            Sdf.ValueTypeNames.Float,
            custom=False,
        ).Set(1.0)
    else:
        prim.AddAppliedSchema("PhysxJointAPI:rogue")

    with pytest.raises(JointRiggerContractError) as caught:
        validate_authored_physics_schemas(stage, plan)

    expected_codes = {
        "value": "physics_schema_conflict",
        "missing": "postwrite_validation_failed",
        "time_sample": "time_sampled_owned_property",
        "unknown_property": "passive_control_schema_conflict",
        "unknown_schema": "passive_control_schema_conflict",
    }
    assert caught.value.code == expected_codes[tamper]


def test_prismatic_linear_state_and_drive_round_trip() -> None:
    stage, plan = _prismatic_fixture()

    diagnostics = author_physics_schemas(stage, plan)

    joint = stage.GetPrimAtPath("/World/Joints/slider")
    schemas = _schema_tokens(joint)
    assert "PhysicsJointStateAPI:linear" in schemas
    assert "PhysicsDriveAPI:linear" in schemas
    assert "PhysxJointAPI" in schemas
    assert joint.GetAttribute("physics:lowerLimit").Get() == pytest.approx(-25.0)
    assert joint.GetAttribute("physics:upperLimit").Get() == pytest.approx(50.0)
    assert joint.GetAttribute("state:linear:physics:position").Get() == 0.0
    assert joint.GetAttribute("state:linear:physics:velocity").Get() == 0.0
    assert joint.GetAttribute("drive:linear:physics:stiffness").Get() == 20.0
    assert joint.GetAttribute("drive:linear:physics:targetPosition").Get() == 0.0
    assert validate_authored_physics_schemas(stage, plan) == diagnostics


def test_raw_r3_authorship_uses_the_mapped_active_edit_target() -> None:
    stage, plan = _revolute_fixture(with_drive=True)
    authored_layer = Sdf.Layer.CreateAnonymous("r3-authored-schemas.usda")
    stage.GetRootLayer().subLayerPaths.append(authored_layer.identifier)
    stage.SetEditTarget(Usd.EditTarget(authored_layer))

    diagnostics = author_physics_schemas(stage, plan)

    joint_spec = authored_layer.GetPrimAtPath("/World/Joints/hinge")
    assert joint_spec is not None
    assert "state:angular:physics:position" in joint_spec.properties
    assert validate_authored_physics_schemas(stage, plan) == diagnostics
    assert len(stage.GetPrimAtPath("/World/Joints/hinge").GetPrimStack()) == 2


def test_mapped_reference_targets_use_spec_namespace() -> None:
    asset_stage, asset_plan = _mimic_fixture()
    asset_stage.SetDefaultPrim(asset_stage.GetPrimAtPath("/World"))

    def remap(path: str) -> str:
        assert path == "/World" or path.startswith("/World/")
        return f"/Scene{path.removeprefix('/World')}"

    joints = []
    for joint in asset_plan.joints:
        topology = joint.topology.model_copy(
            update={
                "joint_id": remap(joint.topology.joint_id),
                "body0": remap(joint.topology.body0),
                "body1": remap(joint.topology.body1),
            }
        )
        mimic = joint.mimic
        if mimic is not None:
            mimic = mimic.model_copy(
                update={"reference_joint_id": remap(mimic.reference_joint_id)}
            )
        joints.append(joint.model_copy(update={"topology": topology, "mimic": mimic}))
    bodies = tuple(
        body.model_copy(
            update={
                "prim_path": remap(body.prim_path),
                "colliders": tuple(
                    collider.model_copy(update={"prim_path": remap(collider.prim_path)})
                    for collider in body.colliders
                ),
            }
        )
        for body in asset_plan.rigid_bodies
    )
    assert asset_plan.articulation_root is not None
    plan = asset_plan.model_copy(
        update={
            "joints": tuple(joints),
            "rigid_bodies": bodies,
            "articulation_root": asset_plan.articulation_root.model_copy(
                update={"prim_path": remap(asset_plan.articulation_root.prim_path)}
            ),
        }
    )

    stage = Usd.Stage.CreateInMemory()
    scene = UsdGeom.Xform.Define(stage, "/Scene").GetPrim()
    stage.SetDefaultPrim(scene)
    UsdGeom.SetStageMetersPerUnit(
        stage,
        UsdGeom.GetStageMetersPerUnit(asset_stage),
    )
    UsdPhysics.SetStageKilogramsPerUnit(
        stage,
        UsdPhysics.GetStageKilogramsPerUnit(asset_stage),
    )
    assert scene.GetReferences().AddReference(
        asset_stage.GetRootLayer().identifier,
        "/World",
    )
    reference_nodes = scene.GetPrimIndex().rootNode.children
    assert len(reference_nodes) == 1
    edit_target = Usd.EditTarget(asset_stage.GetRootLayer(), reference_nodes[0])
    assert edit_target.MapToSpecPath(Sdf.Path("/Scene/Joints/first")) == Sdf.Path(
        "/World/Joints/first"
    )
    stage.SetEditTarget(edit_target)

    diagnostics = author_physics_schemas(stage, plan)

    relationship = asset_stage.GetRootLayer().GetRelationshipAtPath(
        "/World/Joints/second.physxMimicJoint:rotZ:referenceJoint"
    )
    assert relationship is not None
    targets = relationship.GetInfo("targetPaths")
    assert isinstance(targets, Sdf.PathListOp)
    assert tuple(str(path) for path in targets.explicitItems) == (
        "/World/Joints/first",
    )
    assert validate_authored_physics_schemas(stage, plan) == diagnostics


def test_raw_r3_default_cannot_hide_below_a_stronger_composed_value() -> None:
    stage, plan = _revolute_fixture(with_drive=True)
    author_physics_schemas(stage, plan)
    mass = stage.GetPrimAtPath("/World/base").GetAttribute("physics:mass")
    expected = mass.Get()
    root_spec = stage.GetRootLayer().GetPrimAtPath("/World/base")
    assert root_spec is not None
    mass_spec = root_spec.properties["physics:mass"]
    assert isinstance(mass_spec, Sdf.AttributeSpec)
    mass_spec.SetInfo("default", float(expected) + 5e-7)

    stage.SetEditTarget(stage.GetSessionLayer())
    assert mass.Set(expected)
    stage.SetEditTarget(stage.GetRootLayer())
    assert mass.Get() == expected
    root_before = stage.GetRootLayer().ExportToString()
    session_before = stage.GetSessionLayer().ExportToString()

    with pytest.raises(JointRiggerContractError) as caught:
        validate_authored_physics_schemas(stage, plan)

    assert caught.value.code == "authored_graph_mismatch"
    assert "noncanonical default" in caught.value.detail
    assert stage.GetRootLayer().ExportToString() == root_before
    assert stage.GetSessionLayer().ExportToString() == session_before


def test_plan_owned_composed_default_rejects_sub_tolerance_drift() -> None:
    stage, plan = _revolute_fixture(with_drive=True)
    author_physics_schemas(stage, plan)
    root_before = stage.GetRootLayer().ExportToString()

    stage.SetEditTarget(stage.GetSessionLayer())
    state = stage.GetPrimAtPath("/World/Joints/hinge").GetAttribute(
        "state:angular:physics:position"
    )
    assert state.Set(5e-7)
    session_before = stage.GetSessionLayer().ExportToString()
    stage.SetEditTarget(stage.GetRootLayer())

    with pytest.raises(JointRiggerContractError) as caught:
        validate_authored_physics_schemas(stage, plan)

    assert caught.value.code == "authored_graph_mismatch"
    assert "noncanonical default" in caught.value.detail
    assert stage.GetRootLayer().ExportToString() == root_before
    assert stage.GetSessionLayer().ExportToString() == session_before


@pytest.mark.parametrize("storage_type", ["float3", "point3f", "quatf"])
def test_plan_owned_compound_default_rejects_sub_tolerance_drift(
    storage_type: str,
) -> None:
    stage, original_plan = _revolute_fixture(with_drive=True)
    plan = original_plan
    if storage_type in {"point3f", "quatf"}:
        body = plan.rigid_bodies[0]
        assert body.mass is not None
        update = (
            {"center_of_mass_m": (0.01, 0.02, 0.03)}
            if storage_type == "point3f"
            else {"principal_axes": (1.0, 0.0, 0.0, 0.0)}
        )
        mass = body.mass.model_copy(update=update)
        plan = plan.model_copy(
            update={
                "rigid_bodies": (
                    body.model_copy(update={"mass": mass}),
                    *plan.rigid_bodies[1:],
                )
            }
        )
    author_physics_schemas(stage, plan)

    prim_spec = stage.GetRootLayer().GetPrimAtPath("/World/base")
    assert prim_spec is not None
    if storage_type == "float3":
        attribute = prim_spec.properties["physics:diagonalInertia"]
        drifted_default = Gf.Vec3f(100.00001, 150.0, 200.0)
    elif storage_type == "point3f":
        attribute = prim_spec.properties["physics:centerOfMass"]
        drifted_default = Gf.Vec3f(1.0000005, 2.0, 3.0)
    else:
        attribute = prim_spec.properties["physics:principalAxes"]
        drifted_default = Gf.Quatf(1.0, Gf.Vec3f(5e-7, 0.0, 0.0))
    assert isinstance(attribute, Sdf.AttributeSpec)
    attribute.SetInfo("default", drifted_default)
    malformed = stage.GetRootLayer().ExportToString()

    with pytest.raises(JointRiggerContractError) as caught:
        validate_authored_physics_schemas(stage, plan)

    assert caught.value.code == "authored_graph_mismatch"
    assert "noncanonical default" in caught.value.detail
    assert stage.GetRootLayer().ExportToString() == malformed


@pytest.mark.parametrize("canonical_negative", [False, True], ids=["plus", "minus"])
@pytest.mark.parametrize("storage_type", ["float", "float3", "quatf"])
def test_stored_value_comparison_distinguishes_signed_zero_recursively(
    canonical_negative: bool,
    storage_type: str,
) -> None:
    positive: Any
    negative: Any
    if storage_type == "float":
        positive = 0.0
        negative = -0.0
    elif storage_type == "float3":
        positive = Gf.Vec3f(1.0, 0.0, 2.0)
        negative = Gf.Vec3f(1.0, -0.0, 2.0)
    else:
        assert storage_type == "quatf"
        positive = Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0))
        negative = Gf.Quatf(1.0, Gf.Vec3f(-0.0, 0.0, 0.0))

    canonical = negative if canonical_negative else positive
    opposite = positive if canonical_negative else negative
    assert schemas_module._stored_values_equal(canonical, canonical)
    assert not schemas_module._stored_values_equal(canonical, opposite)


@pytest.mark.parametrize("canonical_negative", [False, True], ids=["plus", "minus"])
@pytest.mark.parametrize("storage_type", ["float", "quatf"])
@pytest.mark.parametrize(
    ("opinion_strength", "operation"),
    (
        ("active", validate_authored_physics_schemas),
        ("stronger", author_physics_schemas),
        ("stronger", validate_authored_physics_schemas),
        ("hidden_weaker", author_physics_schemas),
        ("hidden_weaker", validate_authored_physics_schemas),
    ),
    ids=[
        "active-validate",
        "stronger-author",
        "stronger-validate",
        "hidden-weaker-author",
        "hidden-weaker-validate",
    ],
)
def test_public_contract_rejects_signed_zero_default_mismatch(
    canonical_negative: bool,
    storage_type: str,
    operation: Any,
    opinion_strength: str,
) -> None:
    stage, plan = _revolute_fixture(with_drive=True)
    canonical_zero = -0.0 if canonical_negative else 0.0
    opposite_zero = 0.0 if canonical_negative else -0.0
    if storage_type == "float":
        joint = plan.joints[0]
        assert joint.state is not None
        plan = _replace_first_joint(
            plan,
            state=joint.state.model_copy(update={"position": canonical_zero}),
        )
        path = "/World/Joints/hinge"
        name = "state:angular:physics:position"
        opposite_default: Any = opposite_zero
    else:
        assert storage_type == "quatf"
        body = plan.rigid_bodies[0]
        assert body.mass is not None
        plan = _replace_first_body(
            plan,
            mass=body.mass.model_copy(
                update={"principal_axes": (1.0, canonical_zero, 0.0, 0.0)}
            ),
        )
        path = "/World/base"
        name = "physics:principalAxes"
        opposite_default = Gf.Quatf(
            1.0,
            Gf.Vec3f(opposite_zero, 0.0, 0.0),
        )

    root_layer = stage.GetRootLayer()
    session_layer = stage.GetSessionLayer()
    author_physics_schemas(stage, plan)
    layer = root_layer
    if opinion_strength in {"stronger", "hidden_weaker"}:
        stage.SetEditTarget(session_layer)
        author_physics_schemas(stage, plan)
        if opinion_strength == "stronger":
            layer = session_layer
            stage.SetEditTarget(root_layer)
    prim_spec = layer.GetPrimAtPath(path)
    assert prim_spec is not None
    attribute = prim_spec.properties[name]
    assert isinstance(attribute, Sdf.AttributeSpec)
    canonical_plain = schemas_module._plain_value(attribute.GetInfo("default"))
    canonical_component = (
        canonical_plain if storage_type == "float" else canonical_plain[1]
    )
    assert canonical_component == 0.0
    assert math.copysign(1.0, canonical_component) == math.copysign(
        1.0,
        canonical_zero,
    )
    if storage_type == "quatf":
        prim_spec.RemoveProperty(attribute)
        attribute = Sdf.AttributeSpec(
            prim_spec,
            name,
            Sdf.ValueTypeNames.Quatf,
            Sdf.VariabilityVarying,
            False,
        )
        attribute.default = opposite_default
    else:
        attribute.SetInfo("default", opposite_default)
    opposite_plain = schemas_module._plain_value(attribute.GetInfo("default"))
    opposite_component = (
        opposite_plain if storage_type == "float" else opposite_plain[1]
    )
    assert opposite_component == 0.0
    assert math.copysign(1.0, opposite_component) == math.copysign(
        1.0,
        opposite_zero,
    )
    serialized_line = next(
        line for line in layer.ExportToString().splitlines() if name in line
    )
    assert ("-0" in serialized_line) is (math.copysign(1.0, opposite_zero) < 0.0)
    root_before = root_layer.ExportToString()
    session_before = session_layer.ExportToString()

    with pytest.raises(JointRiggerContractError) as caught:
        operation(stage, plan)

    assert caught.value.code == "authored_graph_mismatch"
    assert "noncanonical default" in caught.value.detail
    assert name in caught.value.detail
    assert root_layer.ExportToString() == root_before
    assert session_layer.ExportToString() == session_before


def test_raw_r3_contract_covers_every_owned_attribute_storage_type_and_role() -> None:
    fixtures = [
        _revolute_fixture(with_drive=True, mesh=True),
        _mimic_fixture(),
    ]
    observed_types: set[str] = set()
    observed_names: set[str] = set()

    for stage, original_plan in fixtures:
        body = original_plan.rigid_bodies[0]
        assert body.mass is not None
        mass = body.mass.model_copy(
            update={
                "center_of_mass_m": (0.01, 0.02, 0.03),
                "principal_axes": (1.0, 0.0, 0.0, 0.0),
            }
        )
        plan = original_plan.model_copy(
            update={
                "rigid_bodies": (
                    body.model_copy(update={"mass": mass}),
                    *original_plan.rigid_bodies[1:],
                )
            }
        )
        author_physics_schemas(stage, plan)
        contract = schemas_module._r3_raw_authorship_contract(
            stage,
            plan,
            schemas_module._preflight(stage, plan),
        )

        for path, item in contract.items():
            assert set(item.attribute_defaults) == set(item.attribute_specs)
            prim = stage.GetPrimAtPath(path)
            prim_spec = stage.GetRootLayer().GetPrimAtPath(path)
            assert prim_spec is not None
            for name, (type_name, _) in item.attribute_specs.items():
                observed_types.add(type_name)
                observed_names.add(name)
                attribute_spec = prim_spec.properties[name]
                assert isinstance(attribute_spec, Sdf.AttributeSpec)
                assert schemas_module._stored_values_equal(
                    attribute_spec.GetInfo("default"),
                    item.attribute_defaults[name],
                )
                assert schemas_module._stored_values_equal(
                    prim.GetAttribute(name).Get(),
                    item.attribute_defaults[name],
                )

    assert observed_types == {
        "bool",
        "float",
        "float3",
        "point3f",
        "quatf",
        "token",
    }
    assert {
        "physics:approximation",
        "physics:collisionEnabled",
        "physics:centerOfMass",
        "physics:diagonalInertia",
        "physics:mass",
        "physics:principalAxes",
        "physics:rigidBodyEnabled",
        "state:angular:physics:position",
        "drive:angular:physics:stiffness",
        "physxJoint:maxJointVelocity",
        "physxMimicJoint:rotZ:gearing",
        "physxMimicJoint:rotZ:referenceJointAxis",
    }.issubset(observed_names)


@pytest.mark.parametrize(
    ("path", "malformed_token"),
    (
        ("/World/base", "PhysicsRigidBodyAPI:adversarial"),
        ("/World/base", "PhysicsArticulationRootAPI:adversarial"),
        ("/World/base/collision", "PhysicsCollisionAPI:adversarial"),
        ("/World/Joints/hinge", "PhysicsJointStateAPI"),
    ),
    ids=["body", "root", "collider", "joint"],
)
@pytest.mark.parametrize(
    "operation",
    [author_physics_schemas, validate_authored_physics_schemas],
    ids=["author", "validate"],
)
@pytest.mark.parametrize("contributing_strength", ["weaker", "stronger"])
def test_composed_r3_schema_family_contract_rejects_cross_layer_tokens(
    path: str,
    malformed_token: str,
    operation: Any,
    contributing_strength: str,
) -> None:
    stage, plan = _revolute_fixture(with_drive=True)
    author_physics_schemas(stage, plan)
    if contributing_strength == "weaker":
        stage.SetEditTarget(stage.GetSessionLayer())
        author_physics_schemas(stage, plan)
        prim_spec = stage.GetRootLayer().GetPrimAtPath(path)
        assert prim_spec is not None
        schemas = prim_spec.GetInfo("apiSchemas")
        assert isinstance(schemas, Sdf.TokenListOp)
        schemas.prependedItems = [*schemas.prependedItems, malformed_token]
    else:
        prim_spec = Sdf.CreatePrimInLayer(stage.GetSessionLayer(), path)
        schemas = Sdf.TokenListOp()
        schemas.prependedItems = [malformed_token]
    prim_spec.SetInfo("apiSchemas", schemas)
    root_before = stage.GetRootLayer().ExportToString()
    session_before = stage.GetSessionLayer().ExportToString()

    with pytest.raises(JointRiggerContractError) as caught:
        operation(stage, plan)

    assert caught.value.code == "authored_graph_mismatch"
    assert "contributing raw apiSchemas" in caught.value.detail
    assert malformed_token in caught.value.detail
    assert stage.GetRootLayer().ExportToString() == root_before
    assert stage.GetSessionLayer().ExportToString() == session_before


def test_composed_r3_schema_family_contract_allows_unrelated_weaker_api() -> None:
    stage, plan = _revolute_fixture(with_drive=True)
    author_physics_schemas(stage, plan)
    stage.SetEditTarget(stage.GetSessionLayer())
    author_physics_schemas(stage, plan)

    prim_spec = stage.GetRootLayer().GetPrimAtPath("/World/base")
    assert prim_spec is not None
    schemas = prim_spec.GetInfo("apiSchemas")
    assert isinstance(schemas, Sdf.TokenListOp)
    schemas.prependedItems = [*schemas.prependedItems, "PhysicsMaterialAPI"]
    prim_spec.SetInfo("apiSchemas", schemas)

    validate_authored_physics_schemas(stage, plan)


@pytest.mark.parametrize("masking_list_op", ["explicit", "delete"])
def test_contributing_r3_schema_scan_rejects_tokens_hidden_by_list_ops(
    masking_list_op: str,
) -> None:
    stage, plan = _revolute_fixture(with_drive=True)
    author_physics_schemas(stage, plan)
    path = "/World/base"
    malformed_token = "PhysicsRigidBodyAPI:adversarial"

    weaker_layer = Sdf.Layer.CreateAnonymous("weaker-malformed-r3.usda")
    stage.GetRootLayer().subLayerPaths.append(weaker_layer.identifier)
    weaker_spec = Sdf.CreatePrimInLayer(weaker_layer, path)
    weaker_schemas = Sdf.TokenListOp()
    weaker_schemas.prependedItems = [malformed_token]
    weaker_spec.SetInfo("apiSchemas", weaker_schemas)

    contract = schemas_module._r3_raw_authorship_contract(
        stage,
        plan,
        schemas_module._preflight(stage, plan),
    )[path]
    session_spec = Sdf.CreatePrimInLayer(stage.GetSessionLayer(), path)
    masking_schemas = Sdf.TokenListOp()
    if masking_list_op == "explicit":
        masking_schemas.explicitItems = list(contract.schema_order)
    else:
        masking_schemas.deletedItems = [malformed_token]
    session_spec.SetInfo("apiSchemas", masking_schemas)

    composed_owned = {
        token
        for token in schemas_module._applied_schema_tokens(stage.GetPrimAtPath(path))
        if schemas_module._is_r3_owned_schema_token(token)
    }
    assert composed_owned == contract.schema_tokens

    with pytest.raises(JointRiggerContractError) as caught:
        validate_authored_physics_schemas(stage, plan)

    assert caught.value.code == "authored_graph_mismatch"
    assert "contributing raw apiSchemas" in caught.value.detail
    assert malformed_token in caught.value.detail


@pytest.mark.parametrize("nonactive_bucket", ["explicit", "ordered"])
def test_contributing_r3_schema_scan_allows_expected_nonactive_bucket_forms(
    nonactive_bucket: str,
) -> None:
    stage, plan = _revolute_fixture(with_drive=True)
    author_physics_schemas(stage, plan)
    path = "/World/base"
    contract = schemas_module._r3_raw_authorship_contract(
        stage,
        plan,
        schemas_module._preflight(stage, plan),
    )[path]
    session_spec = Sdf.CreatePrimInLayer(stage.GetSessionLayer(), path)
    schemas = Sdf.TokenListOp()
    if nonactive_bucket == "explicit":
        schemas.explicitItems = list(contract.schema_order)
    else:
        schemas.orderedItems = list(contract.schema_order)
    session_spec.SetInfo("apiSchemas", schemas)

    validate_authored_physics_schemas(stage, plan)


@pytest.mark.parametrize(
    ("path", "name", "value_type", "expected_variability", "mutation"),
    (
        (
            "/World/base",
            "physics:mass",
            Sdf.ValueTypeNames.Float,
            Sdf.VariabilityVarying,
            "custom",
        ),
        (
            "/World/base/collision",
            "physics:collisionEnabled",
            Sdf.ValueTypeNames.Bool,
            Sdf.VariabilityVarying,
            "variability",
        ),
        (
            "/World/Joints/hinge",
            "state:angular:physics:position",
            Sdf.ValueTypeNames.Float,
            Sdf.VariabilityVarying,
            "default",
        ),
        (
            "/World/base",
            "physics:mass",
            Sdf.ValueTypeNames.Float,
            Sdf.VariabilityVarying,
            "kind",
        ),
        (
            "/World/Joints/hinge",
            "state:angular:physics:position",
            Sdf.ValueTypeNames.Float,
            Sdf.VariabilityVarying,
            "type",
        ),
    ),
    ids=[
        "body-custom",
        "collider-variability",
        "joint-default",
        "body-kind",
        "joint-type",
    ],
)
@pytest.mark.parametrize(
    "operation",
    [author_physics_schemas, validate_authored_physics_schemas],
    ids=["author", "validate"],
)
@pytest.mark.parametrize("contributing_strength", ["weaker", "stronger"])
def test_r3_attribute_contract_rejects_noncanonical_contributing_specs(
    path: str,
    name: str,
    value_type: Any,
    expected_variability: Any,
    mutation: str,
    operation: Any,
    contributing_strength: str,
) -> None:
    stage, plan = _revolute_fixture(with_drive=True)
    author_physics_schemas(stage, plan)
    if contributing_strength == "weaker":
        stage.SetEditTarget(stage.GetSessionLayer())
        author_physics_schemas(stage, plan)
        prim_spec = stage.GetRootLayer().GetPrimAtPath(path)
        assert prim_spec is not None
        attribute = prim_spec.properties[name]
        assert isinstance(attribute, Sdf.AttributeSpec)
    else:
        prim_spec = Sdf.CreatePrimInLayer(stage.GetSessionLayer(), path)
        attribute = None

    if mutation in {"kind", "type"}:
        if attribute is not None:
            prim_spec.RemoveProperty(attribute)
        if mutation == "kind":
            Sdf.RelationshipSpec(prim_spec, name, False)
        else:
            Sdf.AttributeSpec(
                prim_spec,
                name,
                Sdf.ValueTypeNames.Double,
                expected_variability,
                False,
            )
    else:
        if attribute is None:
            attribute = Sdf.AttributeSpec(
                prim_spec,
                name,
                value_type,
                expected_variability,
                False,
            )

        if mutation == "custom":
            attribute.custom = True
        elif mutation == "variability":
            attribute.SetInfo("variability", Sdf.VariabilityUniform)
        else:
            assert mutation == "default"
            attribute.SetInfo("default", 5e-7)
    root_before = stage.GetRootLayer().ExportToString()
    session_before = stage.GetSessionLayer().ExportToString()

    with pytest.raises(JointRiggerContractError) as caught:
        operation(stage, plan)

    expected_code = (
        "physics_schema_conflict"
        if mutation in {"kind", "type"}
        else "authored_graph_mismatch"
    )
    assert caught.value.code == expected_code
    if expected_code == "authored_graph_mismatch":
        assert "contributing" in caught.value.detail
    assert name in caught.value.detail
    assert stage.GetRootLayer().ExportToString() == root_before
    assert stage.GetSessionLayer().ExportToString() == session_before


@pytest.mark.parametrize("contributing_strength", ["weaker", "stronger"])
def test_r3_attribute_contract_allows_partial_no_default_source_specs(
    contributing_strength: str,
) -> None:
    stage, plan = _revolute_fixture(with_drive=True)
    author_physics_schemas(stage, plan)
    path = "/World/base"
    name = "physics:mass"

    if contributing_strength == "weaker":
        stage.SetEditTarget(stage.GetSessionLayer())
        author_physics_schemas(stage, plan)
        prim_spec = stage.GetRootLayer().GetPrimAtPath(path)
        assert prim_spec is not None
        attribute = prim_spec.properties[name]
        assert isinstance(attribute, Sdf.AttributeSpec)
        attribute.ClearInfo("default")
    else:
        prim_spec = Sdf.CreatePrimInLayer(stage.GetSessionLayer(), path)
        Sdf.AttributeSpec(
            prim_spec,
            name,
            Sdf.ValueTypeNames.Float,
            Sdf.VariabilityVarying,
            False,
        )
    unrelated = Sdf.AttributeSpec(
        prim_spec,
        "source:unrelatedNote",
        Sdf.ValueTypeNames.String,
        Sdf.VariabilityUniform,
        True,
    )
    unrelated.default = "preserved"

    validate_authored_physics_schemas(stage, plan)


@pytest.mark.parametrize(
    "violation",
    ["time_samples", "spline", "connections", "empty_connections"],
)
@pytest.mark.parametrize(
    "operation",
    [author_physics_schemas, validate_authored_physics_schemas],
    ids=["author", "validate"],
)
@pytest.mark.parametrize(
    "contributing_strength",
    ["weaker", "stronger_masked"],
)
def test_r3_attribute_contract_rejects_hidden_temporal_and_connection_opinions(
    violation: str,
    operation: Any,
    contributing_strength: str,
) -> None:
    stage, plan = _revolute_fixture(with_drive=True)
    extra_layer = None
    composed_has_spline = False
    if contributing_strength == "weaker":
        author_physics_schemas(stage, plan)
        stage.SetEditTarget(stage.GetSessionLayer())
        author_physics_schemas(stage, plan)
        contributing_layer = stage.GetRootLayer()
        prim_spec = contributing_layer.GetPrimAtPath("/World/base")
        assert prim_spec is not None
        attribute = prim_spec.properties["physics:mass"]
        assert isinstance(attribute, Sdf.AttributeSpec)
    else:
        extra_layer = Sdf.Layer.CreateAnonymous("r3-active-weaker.usda")
        stage.GetRootLayer().subLayerPaths.append(extra_layer.identifier)
        stage.SetEditTarget(extra_layer)
        author_physics_schemas(stage, plan)
        stage.SetEditTarget(stage.GetSessionLayer())
        author_physics_schemas(stage, plan)
        contributing_layer = stage.GetRootLayer()
        prim_spec = contributing_layer.GetPrimAtPath("/World/base")
        assert prim_spec is not None
        attribute = Sdf.AttributeSpec(
            prim_spec,
            "physics:mass",
            Sdf.ValueTypeNames.Float,
            Sdf.VariabilityVarying,
            False,
        )
        stage.SetEditTarget(extra_layer)

    if violation == "time_samples":
        contributing_layer.SetTimeSample(attribute.path, 1.0, 999.0)
    elif violation == "spline":
        spline = Ts.Spline("float")
        knot = Ts.Knot("float")
        knot.SetTime(1.0)
        knot.SetValue(3.0)
        spline.SetKnot(knot)
        active_layer = stage.GetEditTarget().GetLayer()
        stage.SetEditTarget(contributing_layer)
        composed_attribute = stage.GetPrimAtPath("/World/base").GetAttribute(
            "physics:mass"
        )
        assert composed_attribute.SetSpline(spline)
        stage.SetEditTarget(active_layer)
        # OpenUSD releases differ on whether a raw Ts spline contributes to
        # the composed query.  The raw AttributeSpec remains the portable
        # source of truth that the validator must reject either way.
        composed_has_spline = composed_attribute.HasSpline()
        assert composed_attribute.GetTimeSamples() == []
        assert "spline" in {str(key) for key in attribute.ListInfoKeys()}
    elif violation == "connections":
        attribute.connectionPathList.prependedItems = [
            Sdf.Path("/World/link.physics:mass")
        ]
    else:
        assert violation == "empty_connections"
        attribute.connectionPathList.ClearEditsAndMakeExplicit()
        assert "connectionPaths" in {str(key) for key in attribute.ListInfoKeys()}
    layers = [stage.GetRootLayer(), stage.GetSessionLayer()]
    if extra_layer is not None:
        layers.append(extra_layer)
    before = tuple(layer.ExportToString() for layer in layers)

    with pytest.raises(JointRiggerContractError) as caught:
        operation(stage, plan)

    expected_code = {
        "time_samples": "authored_graph_mismatch",
        "spline": (
            "time_sampled_owned_property"
            if composed_has_spline
            else "authored_graph_mismatch"
        ),
        "connections": "connected_owned_property",
        "empty_connections": "connected_owned_property",
    }[violation]
    assert caught.value.code == expected_code
    if violation == "time_samples":
        assert "contributing raw attribute" in caught.value.detail
    assert "physics:mass" in caught.value.detail
    expected_detail = {
        "time_samples": "time samples",
        "spline": "spline",
        "connections": "connection",
        "empty_connections": "connection",
    }[violation]
    assert expected_detail in caught.value.detail
    assert tuple(layer.ExportToString() for layer in layers) == before


def _write_real_value_clip(
    path: Path,
    *,
    attribute_name: str,
    prim_path: str = "/Clip/base",
    value_type: Any = Sdf.ValueTypeNames.Float,
    sample_value: Any = 17.0,
) -> Path:
    layer = Sdf.Layer.CreateNew(str(path))
    prim_spec = Sdf.CreatePrimInLayer(layer, prim_path)
    attribute = Sdf.AttributeSpec(
        prim_spec,
        attribute_name,
        value_type,
        Sdf.VariabilityVarying,
        False,
    )
    layer.SetTimeSample(attribute.path, 1.0, sample_value)
    assert layer.Save()
    return path


def _author_real_value_clip_metadata(
    stage: Any,
    layer: Any,
    *,
    anchor_path: str,
    asset_path: Path,
    clip_set: str = "r3",
    clip_prim_path: str = "/Clip",
) -> None:
    previous_target = stage.GetEditTarget()
    stage.SetEditTarget(layer)
    clips = Usd.ClipsAPI(stage.GetPrimAtPath(anchor_path))
    clips.SetClipAssetPaths([Sdf.AssetPath(str(asset_path))], clip_set)
    assert clips.SetClipPrimPath(clip_prim_path, clip_set)
    clips.SetClipActive([(0.0, 0.0)], clip_set)
    clips.SetClipTimes([(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)], clip_set)
    stage.SetEditTarget(previous_target)


@pytest.mark.parametrize(
    "operation",
    [author_physics_schemas, validate_authored_physics_schemas],
    ids=["author", "validate"],
)
@pytest.mark.parametrize("clip_strength", ["weaker", "stronger", "metadata_masked"])
def test_r3_attribute_contract_rejects_real_masked_value_clip_samples(
    tmp_path: Path,
    operation: Any,
    clip_strength: str,
) -> None:
    stage, plan = _revolute_fixture(with_drive=True)
    author_physics_schemas(stage, plan)
    canonical_mass = (
        stage.GetPrimAtPath("/World/base").GetAttribute("physics:mass").Get()
    )
    owned_clip = _write_real_value_clip(
        tmp_path / f"owned-{clip_strength}.usda",
        attribute_name="physics:mass",
    )

    if clip_strength == "weaker":
        stage.SetEditTarget(stage.GetSessionLayer())
        author_physics_schemas(stage, plan)
        _author_real_value_clip_metadata(
            stage,
            stage.GetRootLayer(),
            anchor_path="/World",
            asset_path=owned_clip,
        )
        stage.SetEditTarget(stage.GetSessionLayer())
    elif clip_strength == "stronger":
        stage.SetEditTarget(stage.GetSessionLayer())
        author_physics_schemas(stage, plan)
        _author_real_value_clip_metadata(
            stage,
            stage.GetSessionLayer(),
            anchor_path="/World",
            asset_path=owned_clip,
        )
        stage.SetEditTarget(stage.GetRootLayer())
    else:
        stage.SetEditTarget(stage.GetSessionLayer())
        author_physics_schemas(stage, plan)
        weaker_layer = Sdf.Layer.CreateAnonymous("masked-owned-value-clip.usda")
        stage.GetRootLayer().subLayerPaths.append(weaker_layer.identifier)
        _author_real_value_clip_metadata(
            stage,
            weaker_layer,
            anchor_path="/World",
            asset_path=owned_clip,
        )
        unrelated_clip = _write_real_value_clip(
            tmp_path / "stronger-unrelated.usda",
            attribute_name="source:temperature",
        )
        _author_real_value_clip_metadata(
            stage,
            stage.GetSessionLayer(),
            anchor_path="/World",
            asset_path=unrelated_clip,
        )
        stage.SetEditTarget(stage.GetSessionLayer())

    mass = stage.GetPrimAtPath("/World/base").GetAttribute("physics:mass")
    assert mass.Get() == canonical_mass
    assert mass.GetTimeSamples() == []
    before = tuple(
        layer.ExportToString()
        for layer in (stage.GetRootLayer(), stage.GetSessionLayer())
    )

    with pytest.raises(JointRiggerContractError) as caught:
        operation(stage, plan)

    assert caught.value.code == "authored_graph_mismatch"
    assert "value" in caught.value.detail
    assert "clip" in caught.value.detail
    assert "physics:mass" in caught.value.detail
    assert (
        tuple(
            layer.ExportToString()
            for layer in (stage.GetRootLayer(), stage.GetSessionLayer())
        )
        == before
    )


@pytest.mark.parametrize(
    "latent_case", ["split_dictionary", "no_active", "inactive_index"]
)
def test_r3_attribute_contract_rejects_latent_raw_clip_sources(
    tmp_path: Path,
    latent_case: str,
) -> None:
    stage, plan = _revolute_fixture(with_drive=True)
    author_physics_schemas(stage, plan)
    stage.SetEditTarget(stage.GetSessionLayer())
    author_physics_schemas(stage, plan)
    owned_clip = _write_real_value_clip(
        tmp_path / f"latent-owned-{latent_case}.usda",
        attribute_name="physics:mass",
    )
    unrelated_clip = _write_real_value_clip(
        tmp_path / f"latent-unrelated-{latent_case}.usda",
        attribute_name="source:temperature",
    )
    clip_set = f"r3_{latent_case}"

    if latent_case == "split_dictionary":
        weaker_layer = Sdf.Layer.CreateAnonymous("split-clips-dictionary.usda")
        stage.GetRootLayer().subLayerPaths.append(weaker_layer.identifier)
        stage.SetEditTarget(weaker_layer)
        Usd.ClipsAPI(stage.GetPrimAtPath("/World")).SetClipAssetPaths(
            [Sdf.AssetPath(str(owned_clip))],
            clip_set,
        )
        stage.SetEditTarget(stage.GetSessionLayer())
        assert Usd.ClipsAPI(stage.GetPrimAtPath("/World")).SetClipPrimPath(
            "/Clip",
            clip_set,
        )
    else:
        stage.SetEditTarget(stage.GetRootLayer())
        clips = Usd.ClipsAPI(stage.GetPrimAtPath("/World"))
        asset_paths = [Sdf.AssetPath(str(owned_clip))]
        if latent_case == "inactive_index":
            asset_paths = [
                Sdf.AssetPath(str(unrelated_clip)),
                Sdf.AssetPath(str(owned_clip)),
            ]
        clips.SetClipAssetPaths(asset_paths, clip_set)
        assert clips.SetClipPrimPath("/Clip", clip_set)
        if latent_case == "inactive_index":
            clips.SetClipActive([(0.0, 0.0)], clip_set)
    stage.SetEditTarget(stage.GetSessionLayer())
    mass = stage.GetPrimAtPath("/World/base").GetAttribute("physics:mass")
    assert mass.GetTimeSamples() == []

    with pytest.raises(JointRiggerContractError) as caught:
        validate_authored_physics_schemas(stage, plan)

    assert caught.value.code == "authored_graph_mismatch"
    assert "clip" in caught.value.detail
    assert "physics:mass" in caught.value.detail


def test_r3_attribute_contract_allows_latent_unrelated_clip_source(
    tmp_path: Path,
) -> None:
    stage, plan = _revolute_fixture(with_drive=True)
    author_physics_schemas(stage, plan)
    unrelated_clip = _write_real_value_clip(
        tmp_path / "latent-unrelated-only.usda",
        attribute_name="source:temperature",
    )
    clips = Usd.ClipsAPI(stage.GetPrimAtPath("/World"))
    clips.SetClipAssetPaths(
        [Sdf.AssetPath(str(unrelated_clip))],
        "r3_latent_unrelated",
    )
    assert clips.SetClipPrimPath("/Clip", "r3_latent_unrelated")

    validate_authored_physics_schemas(stage, plan)


@pytest.mark.parametrize(
    "authored_kind",
    ["owned", "unrelated"],
)
def test_r3_clip_asset_resolution_ignores_forged_resolved_path(
    tmp_path: Path,
    authored_kind: str,
) -> None:
    stage, plan = _revolute_fixture(with_drive=True)
    author_physics_schemas(stage, plan)
    owned_clip = _write_real_value_clip(
        tmp_path / "resolved-owned.usda",
        attribute_name="physics:mass",
    )
    unrelated_clip = _write_real_value_clip(
        tmp_path / "resolved-unrelated.usda",
        attribute_name="source:temperature",
    )
    authored_path, forged_resolved_path = (
        (owned_clip, unrelated_clip)
        if authored_kind == "owned"
        else (unrelated_clip, owned_clip)
    )
    clips = Usd.ClipsAPI(stage.GetPrimAtPath("/World"))
    clips.SetClipAssetPaths(
        [Sdf.AssetPath(str(authored_path), str(forged_resolved_path))],
        "r3_forged_resolution",
    )
    assert clips.SetClipPrimPath("/Clip", "r3_forged_resolution")
    saved_stage_path = tmp_path / f"forged-{authored_kind}-stage.usda"
    assert stage.GetRootLayer().Export(str(saved_stage_path))
    reopened_stage = Usd.Stage.Open(str(saved_stage_path))
    assert reopened_stage

    for candidate_stage in (stage, reopened_stage):
        if authored_kind == "owned":
            with pytest.raises(JointRiggerContractError) as caught:
                validate_authored_physics_schemas(candidate_stage, plan)

            assert caught.value.code == "authored_graph_mismatch"
            assert str(owned_clip) in caught.value.detail
        else:
            validate_authored_physics_schemas(candidate_stage, plan)


@pytest.mark.parametrize("clip_kind", ["owned", "unrelated"])
def test_r3_clip_asset_resolution_uses_the_stage_resolver_context(
    tmp_path: Path,
    clip_kind: str,
) -> None:
    search_dir = tmp_path / "resolver-search"
    stage_dir = tmp_path / "resolver-stage"
    search_dir.mkdir()
    stage_dir.mkdir()
    clip_path = _write_real_value_clip(
        search_dir / "context-clip.usda",
        attribute_name=(
            "physics:mass" if clip_kind == "owned" else "source:temperature"
        ),
    )
    assert not (stage_dir / clip_path.name).exists()

    source_stage, plan = _revolute_fixture(with_drive=True)
    author_physics_schemas(source_stage, plan)
    clips = Usd.ClipsAPI(source_stage.GetPrimAtPath("/World"))
    clips.SetClipAssetPaths([Sdf.AssetPath(clip_path.name)], "resolver_context")
    assert clips.SetClipPrimPath("/Clip", "resolver_context")
    stage_path = stage_dir / "context-stage.usda"
    assert source_stage.GetRootLayer().Export(str(stage_path))
    resolver_context = Ar.ResolverContext(Ar.DefaultResolverContext([str(search_dir)]))
    stage = Usd.Stage.Open(
        str(stage_path),
        pathResolverContext=resolver_context,
    )
    assert stage is not None
    assert stage.GetPathResolverContext() == resolver_context

    if clip_kind == "owned":
        with pytest.raises(JointRiggerContractError) as caught:
            validate_authored_physics_schemas(stage, plan)

        assert caught.value.code == "authored_graph_mismatch"
        assert "context-clip.usda" in caught.value.detail
    else:
        validate_authored_physics_schemas(stage, plan)


def test_r3_clip_audit_preserves_each_complete_raw_asset_prim_path_pair(
    tmp_path: Path,
) -> None:
    stage, plan = _revolute_fixture(with_drive=True)
    author_physics_schemas(stage, plan)
    stage.SetEditTarget(stage.GetSessionLayer())
    author_physics_schemas(stage, plan)
    weak_clip = _write_real_value_clip(
        tmp_path / "pair-weak-owned-name.usda",
        attribute_name="physics:mass",
        prim_path="/Other/base",
    )
    strong_clip = _write_real_value_clip(
        tmp_path / "pair-strong-unrelated.usda",
        attribute_name="source:temperature",
        prim_path="/Clip/base",
    )
    _author_real_value_clip_metadata(
        stage,
        stage.GetRootLayer(),
        anchor_path="/World",
        asset_path=weak_clip,
        clip_set="r3_exact_pair",
        clip_prim_path="/Clip",
    )
    _author_real_value_clip_metadata(
        stage,
        stage.GetSessionLayer(),
        anchor_path="/World",
        asset_path=strong_clip,
        clip_set="r3_exact_pair",
        clip_prim_path="/Other",
    )
    stage.SetEditTarget(stage.GetSessionLayer())
    mass = stage.GetPrimAtPath("/World/base").GetAttribute("physics:mass")
    assert mass.GetTimeSamples() == []

    validate_authored_physics_schemas(stage, plan)


def test_r3_clip_audit_uses_composed_path_for_the_winning_raw_asset(
    tmp_path: Path,
) -> None:
    stage, plan = _revolute_fixture(with_drive=True)
    author_physics_schemas(stage, plan)
    stage.SetEditTarget(stage.GetSessionLayer())
    author_physics_schemas(stage, plan)
    owned_clip = _write_real_value_clip(
        tmp_path / "winning-asset-partial-path.usda",
        attribute_name="physics:mass",
        prim_path="/Strong/base",
    )
    _author_real_value_clip_metadata(
        stage,
        stage.GetRootLayer(),
        anchor_path="/World",
        asset_path=owned_clip,
        clip_set="r3_partial_path_override",
        clip_prim_path="/Weak",
    )
    stage.SetEditTarget(stage.GetSessionLayer())
    assert Usd.ClipsAPI(stage.GetPrimAtPath("/World")).SetClipPrimPath(
        "/Strong",
        "r3_partial_path_override",
    )
    mass = stage.GetPrimAtPath("/World/base").GetAttribute("physics:mass")
    assert mass.GetTimeSamples() == []

    with pytest.raises(JointRiggerContractError) as caught:
        validate_authored_physics_schemas(stage, plan)

    assert caught.value.code == "authored_graph_mismatch"
    assert "physics:mass" in caught.value.detail
    assert "winning-asset-partial-path.usda" in caught.value.detail


@pytest.mark.parametrize("fixture_kind", ["active", "empty-spherical"])
def test_r3_clip_audit_rejects_unplanned_owned_prefix_attributes(
    tmp_path: Path,
    fixture_kind: str,
) -> None:
    if fixture_kind == "active":
        stage, plan = _revolute_fixture(with_drive=False)
        joint_path = "/World/Joints/hinge"
        clip_prim_path = "/Clip/Joints/hinge"
    else:
        stage, plan = _spherical_fixture()
        joint_path = "/World/Joints/ball"
        clip_prim_path = "/Clip/Joints/ball"
    author_physics_schemas(stage, plan)
    unplanned_clip = _write_real_value_clip(
        tmp_path / f"unplanned-{fixture_kind}.usda",
        attribute_name="drive:angular:physics:rogue",
        prim_path=clip_prim_path,
    )
    _author_real_value_clip_metadata(
        stage,
        stage.GetRootLayer(),
        anchor_path="/World",
        asset_path=unplanned_clip,
        clip_set="r3_unplanned_prefix",
    )

    with pytest.raises(JointRiggerContractError) as caught:
        validate_authored_physics_schemas(stage, plan)

    assert caught.value.code == "authored_graph_mismatch"
    assert joint_path in caught.value.detail
    assert "drive:angular:physics:rogue" in caught.value.detail


@pytest.mark.parametrize(
    "operation",
    [author_physics_schemas, validate_authored_physics_schemas],
    ids=["author-rollback", "validate-read-only"],
)
@pytest.mark.parametrize("owned_channel", ["joint-local-pos", "endpoint-xform-op"])
def test_standalone_schema_validator_rejects_latent_r2_value_clip_samples(
    tmp_path: Path,
    operation: Any,
    owned_channel: str,
) -> None:
    stage, plan = _revolute_fixture(with_drive=False)
    author_physics_schemas(stage, plan)
    if owned_channel == "joint-local-pos":
        scene_prim_path = "/World/Joints/hinge"
        clip_prim_path = "/Clip/Joints/hinge"
        attribute_name = "physics:localPos0"
        value_type = Sdf.ValueTypeNames.Point3f
        sample_value = Gf.Vec3f(17.0, 18.0, 19.0)
    else:
        scene_prim_path = "/World/link"
        clip_prim_path = "/Clip/link"
        attribute_name = "xformOp:translate"
        value_type = Sdf.ValueTypeNames.Double3
        sample_value = Gf.Vec3d(17.0, 18.0, 19.0)

    clip_asset_path = tmp_path / f"standalone-{owned_channel}.usda"
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

    clips = Usd.ClipsAPI(stage.GetPrimAtPath("/World"))
    clips.SetClipAssetPaths(
        [Sdf.AssetPath(str(clip_asset_path))],
        "r2_standalone",
    )
    assert clips.SetClipPrimPath("/Clip", "r2_standalone")
    observed_attribute = stage.GetPrimAtPath(scene_prim_path).GetAttribute(
        attribute_name
    )
    assert observed_attribute
    assert observed_attribute.GetTimeSamples() == []
    before = stage.GetRootLayer().ExportToString()

    with pytest.raises(JointRiggerContractError) as caught:
        operation(stage, plan)

    assert caught.value.code == "authored_graph_mismatch"
    assert attribute_name in caught.value.detail
    assert clip_asset_path.name in caught.value.detail
    assert stage.GetRootLayer().ExportToString() == before


@pytest.mark.parametrize(
    (
        "channel",
        "scene_prim_path",
        "clip_prim_path",
        "attribute_name",
        "value_type",
        "sample_value",
        "active",
    ),
    (
        (
            "joint-enabled",
            "/World/Joints/hinge",
            "/Clip/Joints/hinge",
            "physics:jointEnabled",
            Sdf.ValueTypeNames.Bool,
            False,
            True,
        ),
        (
            "joint-break-force",
            "/World/Joints/hinge",
            "/Clip/Joints/hinge",
            "physics:breakForce",
            Sdf.ValueTypeNames.Float,
            3.0,
            False,
        ),
        (
            "body-starts-asleep",
            "/World/link",
            "/Clip/link",
            "physics:startsAsleep",
            Sdf.ValueTypeNames.Bool,
            True,
            False,
        ),
        (
            "new-transform-op",
            "/World/link",
            "/Clip/link",
            "xformOp:rotateXYZ:clipOnly",
            Sdf.ValueTypeNames.Float3,
            Gf.Vec3f(10.0, 20.0, 30.0),
            False,
        ),
        (
            "collider-geometry",
            "/World/base/collision",
            "/Clip/base/collision",
            "size",
            Sdf.ValueTypeNames.Double,
            3.0,
            False,
        ),
        (
            "collider-transform",
            "/World/base/collision",
            "/Clip/base/collision",
            "xformOp:translate:clipOnly",
            Sdf.ValueTypeNames.Double3,
            Gf.Vec3d(1.0, 2.0, 3.0),
            False,
        ),
    ),
)
def test_standalone_clip_audit_covers_joint_body_transform_and_collider_channels(
    tmp_path: Path,
    channel: str,
    scene_prim_path: str,
    clip_prim_path: str,
    attribute_name: str,
    value_type: Any,
    sample_value: Any,
    active: bool,
) -> None:
    stage, plan = _revolute_fixture(with_drive=False)
    author_physics_schemas(stage, plan)
    clip_asset_path = _write_real_value_clip(
        tmp_path / f"boundary-{channel}.usda",
        attribute_name=attribute_name,
        prim_path=clip_prim_path,
        value_type=value_type,
        sample_value=sample_value,
    )
    clips = Usd.ClipsAPI(stage.GetPrimAtPath("/World"))
    clips.SetClipAssetPaths(
        [Sdf.AssetPath(str(clip_asset_path))],
        f"boundary_{channel.replace('-', '_')}",
    )
    assert clips.SetClipPrimPath(
        "/Clip",
        f"boundary_{channel.replace('-', '_')}",
    )
    if active:
        clips.SetClipActive(
            [(0.0, 0.0)],
            f"boundary_{channel.replace('-', '_')}",
        )
    assert stage.GetPrimAtPath(scene_prim_path)

    with pytest.raises(JointRiggerContractError) as caught:
        validate_authored_physics_schemas(stage, plan)

    assert caught.value.code in {
        "authored_graph_mismatch",
        "time_sampled_owned_property",
        "time_varying_endpoint_transform",
    }
    assert attribute_name in caught.value.detail


def test_standalone_clip_audit_reaches_instance_proxy_collider_geometry(
    tmp_path: Path,
) -> None:
    stage, original_plan = _revolute_fixture(with_drive=False)
    plan = _replace_colliders_with_xform_instance_roots(
        stage,
        original_plan,
        mesh_approximation="convexHull",
        mesh_collision_api=True,
    )
    author_physics_schemas(stage, plan)
    proxy_path = "/World/base/collision/shape"
    proxy = stage.GetPrimAtPath(proxy_path)
    assert proxy and proxy.IsInstanceProxy() and proxy.IsA(UsdGeom.Gprim)
    clip_asset_path = _write_real_value_clip(
        tmp_path / "instance-proxy-geometry.usda",
        attribute_name="size",
        prim_path="/Clip/base/collision/shape",
        value_type=Sdf.ValueTypeNames.Double,
        sample_value=4.0,
    )
    clips = Usd.ClipsAPI(stage.GetPrimAtPath("/World"))
    clips.SetClipAssetPaths(
        [Sdf.AssetPath(str(clip_asset_path))],
        "instance_proxy_geometry",
    )
    assert clips.SetClipPrimPath("/Clip", "instance_proxy_geometry")

    with pytest.raises(JointRiggerContractError) as caught:
        validate_authored_physics_schemas(stage, plan)

    assert caught.value.code == "authored_graph_mismatch"
    assert proxy_path in caught.value.detail
    assert "size" in caught.value.detail


@pytest.mark.parametrize(
    "malformed_asset_paths",
    [7, None, "owned.usda", {"path": "owned.usda"}],
    ids=["integer", "none", "string", "dictionary"],
)
@pytest.mark.parametrize(
    "operation",
    [author_physics_schemas, validate_authored_physics_schemas],
    ids=["author", "validate"],
)
def test_r3_clip_audit_rejects_malformed_asset_paths_container(
    malformed_asset_paths: Any,
    operation: Any,
) -> None:
    stage, plan = _revolute_fixture(with_drive=True)
    author_physics_schemas(stage, plan)
    world_spec = stage.GetRootLayer().GetPrimAtPath("/World")
    assert world_spec is not None
    world_spec.SetInfo(
        "clips",
        {
            "malformed": {
                "assetPaths": malformed_asset_paths,
                "primPath": "/Clip",
            }
        },
    )
    before = stage.GetRootLayer().ExportToString()

    with pytest.raises(JointRiggerContractError) as caught:
        operation(stage, plan)

    assert caught.value.code == "authored_graph_mismatch"
    assert "malformed assetPaths metadata" in caught.value.detail
    assert stage.GetRootLayer().ExportToString() == before


@pytest.mark.parametrize(
    "operation",
    [author_physics_schemas, validate_authored_physics_schemas],
    ids=["author-rollback", "validate-read-only"],
)
def test_r3_clip_audit_normalizes_invalid_layer_open_failures(
    tmp_path: Path,
    operation: Any,
) -> None:
    stage, plan = _revolute_fixture(with_drive=True)
    author_physics_schemas(stage, plan)
    invalid_clip = tmp_path / "invalid-value-clip.usda"
    invalid_clip.write_text("this is not a valid USDA layer", encoding="utf-8")
    clips = Usd.ClipsAPI(stage.GetPrimAtPath("/World"))
    clips.SetClipAssetPaths(
        [Sdf.AssetPath(str(invalid_clip))],
        "r3_invalid_layer",
    )
    assert clips.SetClipPrimPath("/Clip", "r3_invalid_layer")
    before = stage.GetRootLayer().ExportToString()

    with pytest.raises(JointRiggerContractError) as caught:
        operation(stage, plan)

    assert caught.value.code == "authored_graph_mismatch"
    assert "value-clip asset inspection failed" in caught.value.detail
    assert "invalid-value-clip.usda" in caught.value.detail
    assert stage.GetRootLayer().ExportToString() == before


def test_r3_value_clip_audit_fails_closed_at_fixed_inspection_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage, plan = _revolute_fixture(with_drive=True)
    author_physics_schemas(stage, plan)
    clip_paths = [
        _write_real_value_clip(
            tmp_path / f"bounded-unrelated-{index}.usda",
            attribute_name="source:temperature",
        )
        for index in range(2)
    ]
    clips = Usd.ClipsAPI(stage.GetPrimAtPath("/World"))
    clips.SetClipAssetPaths(
        [Sdf.AssetPath(str(path)) for path in clip_paths],
        "r3_bounded",
    )
    assert clips.SetClipPrimPath("/Clip", "r3_bounded")
    monkeypatch.setattr(schemas_module, "_MAX_R3_VALUE_CLIP_INSPECTIONS", 1)

    with pytest.raises(JointRiggerContractError) as caught:
        validate_authored_physics_schemas(stage, plan)

    assert caught.value.code == "authored_graph_mismatch"
    assert "bounded inspection limit of 1" in caught.value.detail


def test_instance_proxy_value_clip_boundary_uses_one_stage_traversal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage, original_plan = _revolute_fixture()
    plan = _replace_colliders_with_xform_instance_roots(
        stage,
        original_plan,
        mesh_approximation="convexHull",
        mesh_collision_api=True,
        nested_depth=3,
    )
    unrelated_instance = UsdGeom.Xform.Define(stage, "/World/unrelated").GetPrim()
    unrelated_instance.GetReferences().AddInternalReference(
        "/World/ColliderPrototypes/prototype_0"
    )
    unrelated_instance.SetInstanceable(True)
    assert unrelated_instance.IsInstance()
    real_range = schemas_module._stage_prims_with_instance_proxies
    range_calls = 0

    def counted_range(stage_arg: Any, *, Usd: Any) -> Any:
        nonlocal range_calls
        range_calls += 1
        return real_range(stage_arg, Usd=Usd)

    monkeypatch.setattr(
        schemas_module,
        "_stage_prims_with_instance_proxies",
        counted_range,
    )

    attributes, transform_paths = (
        schemas_module._complete_plan_owned_value_clip_attributes(stage, plan, {})
    )

    assert range_calls == 1
    for proxy_path in (
        "/World/base/collision/nested_0/nested_1/nested_2/shape",
        "/World/link/collision/nested_0/nested_1/nested_2/shape",
    ):
        assert "size" in attributes[proxy_path]
        assert proxy_path in transform_paths
    assert not any(path.startswith("/World/unrelated/") for path in transform_paths)


def test_instance_proxy_value_clip_boundary_has_global_prim_visit_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage, original_plan = _revolute_fixture()
    plan = _replace_colliders_with_xform_instance_roots(
        stage,
        original_plan,
        mesh_approximation="convexHull",
        mesh_collision_api=True,
    )
    before = stage.GetRootLayer().ExportToString()
    monkeypatch.setattr(schemas_module, "_MAX_R3_INSTANCE_PROXY_PRIM_VISITS", 1)

    with pytest.raises(JointRiggerContractError) as caught:
        schemas_module._complete_plan_owned_value_clip_attributes(stage, plan, {})

    assert caught.value.code == "r3_instance_proxy_scan_limit_exceeded"
    assert "fixed 1-prim visit limit" in caught.value.detail
    assert stage.GetRootLayer().ExportToString() == before


def test_instance_proxy_value_clip_boundary_has_global_owned_path_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage, original_plan = _revolute_fixture()
    plan = _replace_colliders_with_xform_instance_roots(
        stage,
        original_plan,
        mesh_approximation="convexHull",
        mesh_collision_api=True,
    )
    before = stage.GetRootLayer().ExportToString()
    monkeypatch.setattr(schemas_module, "_MAX_R3_INSTANCE_PROXY_OWNED_PATHS", 1)

    with pytest.raises(JointRiggerContractError) as caught:
        author_physics_schemas(stage, plan)

    assert caught.value.code == "r3_instance_proxy_owned_path_limit_exceeded"
    assert "fixed 1-covered-path limit" in caught.value.detail
    assert stage.GetRootLayer().ExportToString() == before


def test_value_clip_audit_has_global_ancestor_visit_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage, plan = _revolute_fixture()
    author_physics_schemas(stage, plan)
    before = stage.GetRootLayer().ExportToString()
    monkeypatch.setattr(schemas_module, "_MAX_R3_VALUE_CLIP_ANCESTOR_VISITS", 3)

    with pytest.raises(JointRiggerContractError) as caught:
        validate_authored_physics_schemas(stage, plan)

    assert caught.value.code == "r3_value_clip_ancestor_scan_limit_exceeded"
    assert "fixed 3-ancestor visit limit" in caught.value.detail
    assert stage.GetRootLayer().ExportToString() == before


def test_r3_attribute_contract_allows_real_clips_that_cannot_affect_owned_paths(
    tmp_path: Path,
) -> None:
    stage, plan = _revolute_fixture(with_drive=True)
    author_physics_schemas(stage, plan)
    unrelated_clip = _write_real_value_clip(
        tmp_path / "ancestor-unrelated.usda",
        attribute_name="source:temperature",
    )
    _author_real_value_clip_metadata(
        stage,
        stage.GetSessionLayer(),
        anchor_path="/World",
        asset_path=unrelated_clip,
    )

    UsdGeom.Xform.Define(stage, "/World/sibling")
    sibling_clip = _write_real_value_clip(
        tmp_path / "sibling-owned-name.usda",
        attribute_name="physics:mass",
        prim_path="/Clip",
    )
    _author_real_value_clip_metadata(
        stage,
        stage.GetSessionLayer(),
        anchor_path="/World/sibling",
        asset_path=sibling_clip,
    )
    stage.SetEditTarget(stage.GetRootLayer())

    validate_authored_physics_schemas(stage, plan)


def test_r3_attribute_contract_fails_closed_for_covered_template_clip(
    tmp_path: Path,
) -> None:
    stage, plan = _revolute_fixture(with_drive=True)
    author_physics_schemas(stage, plan)
    template_clip = _write_real_value_clip(
        tmp_path / "template.1.usda",
        attribute_name="physics:mass",
    )
    assert template_clip.exists()
    stage.SetEditTarget(stage.GetSessionLayer())
    author_physics_schemas(stage, plan)
    clips = Usd.ClipsAPI(stage.GetPrimAtPath("/World"))
    assert clips.SetClipPrimPath("/Clip", "r3_template")
    clips.SetClipTemplateAssetPath(
        str(tmp_path / "template.#.usda"),
        "r3_template",
    )
    clips.SetClipTemplateStartTime(1.0, "r3_template")
    clips.SetClipTemplateEndTime(1.0, "r3_template")
    clips.SetClipTemplateStride(1.0, "r3_template")
    stage.SetEditTarget(stage.GetRootLayer())

    with pytest.raises(JointRiggerContractError) as caught:
        validate_authored_physics_schemas(stage, plan)

    assert caught.value.code == "authored_graph_mismatch"
    assert "template" in caught.value.detail
    assert "covering owned prim /World" in caught.value.detail


@pytest.mark.parametrize(
    ("role", "path", "name"),
    (
        ("body", "/World/base", "drive:angular:physics:rogue"),
        ("collider", "/World/base/collision", "physics:mass"),
        ("joint", "/World/Joints/hinge", "physxMimicJoint:rotZ:rogue"),
    ),
)
@pytest.mark.parametrize(
    "operation",
    [author_physics_schemas, validate_authored_physics_schemas],
    ids=["author", "validate"],
)
@pytest.mark.parametrize("contributing_strength", ["weaker", "stronger"])
def test_r3_contract_rejects_unexpected_owned_property_names_on_any_layer(
    role: str,
    path: str,
    name: str,
    operation: Any,
    contributing_strength: str,
) -> None:
    stage, plan = _revolute_fixture(with_drive=True)
    author_physics_schemas(stage, plan)
    if contributing_strength == "weaker":
        stage.SetEditTarget(stage.GetSessionLayer())
        author_physics_schemas(stage, plan)
        layer = stage.GetRootLayer()
    else:
        layer = stage.GetSessionLayer()
        stage.SetEditTarget(stage.GetRootLayer())
    prim_spec = Sdf.CreatePrimInLayer(layer, path)
    rogue = Sdf.AttributeSpec(
        prim_spec,
        name,
        Sdf.ValueTypeNames.Float,
        Sdf.VariabilityVarying,
        False,
    )
    rogue.default = 1.0
    root_before = stage.GetRootLayer().ExportToString()
    session_before = stage.GetSessionLayer().ExportToString()

    with pytest.raises(JointRiggerContractError) as caught:
        operation(stage, plan)

    if role == "joint":
        assert caught.value.code == "drive_schema_conflict"
        assert "mimic schema" in caught.value.detail
    else:
        assert caught.value.code == "authored_graph_mismatch"
        assert "unexpected plan-owned names" in caught.value.detail
        assert name in caught.value.detail
        assert path in caught.value.detail
    assert stage.GetRootLayer().ExportToString() == root_before
    assert stage.GetSessionLayer().ExportToString() == session_before


@pytest.mark.parametrize("mutation", ["custom", "variability", "kind"])
@pytest.mark.parametrize("contributing_strength", ["weaker", "stronger"])
def test_r3_relationship_contract_rejects_noncanonical_contributing_specs(
    mutation: str,
    contributing_strength: str,
) -> None:
    stage, plan = _mimic_fixture()
    author_physics_schemas(stage, plan)
    path = "/World/Joints/second"
    name = "physxMimicJoint:rotZ:referenceJoint"

    if contributing_strength == "weaker":
        stage.SetEditTarget(stage.GetSessionLayer())
        author_physics_schemas(stage, plan)
        prim_spec = stage.GetRootLayer().GetPrimAtPath(path)
        assert prim_spec is not None
        relationship = prim_spec.properties[name]
        assert isinstance(relationship, Sdf.RelationshipSpec)
    else:
        prim_spec = Sdf.CreatePrimInLayer(stage.GetSessionLayer(), path)
        relationship = None

    if mutation == "kind":
        if relationship is not None:
            prim_spec.RemoveProperty(relationship)
        Sdf.AttributeSpec(
            prim_spec,
            name,
            Sdf.ValueTypeNames.Token,
            Sdf.VariabilityUniform,
            False,
        )
    else:
        if relationship is None:
            relationship = Sdf.RelationshipSpec(prim_spec, name, False)
        if mutation == "custom":
            relationship.custom = True
        else:
            relationship.SetInfo("variability", Sdf.VariabilityVarying)
    root_before = stage.GetRootLayer().ExportToString()
    session_before = stage.GetSessionLayer().ExportToString()

    with pytest.raises(JointRiggerContractError) as caught:
        validate_authored_physics_schemas(stage, plan)

    assert caught.value.code in {"authored_graph_mismatch", "mimic_schema_conflict"}
    assert name in caught.value.detail
    assert stage.GetRootLayer().ExportToString() == root_before
    assert stage.GetSessionLayer().ExportToString() == session_before


@pytest.mark.parametrize("contributing_strength", ["weaker", "stronger"])
def test_r3_relationship_contract_allows_nonactive_target_list_op_forms(
    contributing_strength: str,
) -> None:
    stage, plan = _mimic_fixture()
    author_physics_schemas(stage, plan)
    path = "/World/Joints/second"
    name = "physxMimicJoint:rotZ:referenceJoint"
    expected_target = Sdf.Path("/World/Joints/first")

    if contributing_strength == "weaker":
        stage.SetEditTarget(stage.GetSessionLayer())
        author_physics_schemas(stage, plan)
        prim_spec = stage.GetRootLayer().GetPrimAtPath(path)
        assert prim_spec is not None
        relationship = prim_spec.properties[name]
        assert isinstance(relationship, Sdf.RelationshipSpec)
        relationship.targetPathList.ClearEdits()
    else:
        prim_spec = Sdf.CreatePrimInLayer(stage.GetSessionLayer(), path)
        relationship = Sdf.RelationshipSpec(prim_spec, name, False)
    relationship.targetPathList.prependedItems = [expected_target]

    validate_authored_physics_schemas(stage, plan)


def test_raw_r3_contract_aggregates_body_collider_and_root_on_one_prim() -> None:
    stage, plan = _revolute_fixture(with_drive=True)
    assert stage.RemovePrim("/World/base/collision")
    UsdGeom.Cube.Define(stage, "/World/base")
    plan = _replace_first_body(
        plan,
        colliders=(
            ColliderPlanV1(
                prim_path="/World/base",
                provenance=_PROVENANCE,
            ),
        ),
    )

    author_physics_schemas(stage, plan)

    prim_spec = stage.GetRootLayer().GetPrimAtPath("/World/base")
    assert prim_spec is not None
    schemas = prim_spec.GetInfo("apiSchemas")
    assert isinstance(schemas, Sdf.TokenListOp)
    assert tuple(str(item) for item in schemas.prependedItems) == (
        "PhysicsRigidBodyAPI",
        "PhysicsMassAPI",
        "PhysicsCollisionAPI",
        "PhysicsArticulationRootAPI",
    )
    assert {
        "physics:collisionEnabled",
        "physics:diagonalInertia",
        "physics:kinematicEnabled",
        "physics:mass",
        "physics:rigidBodyEnabled",
    }.issubset(prim_spec.properties.keys())
    validate_authored_physics_schemas(stage, plan)


def _add_hidden_r3_graph_prim(stage: Any, hidden_kind: str) -> str:
    kind, schema = hidden_kind.split("_", maxsplit=1)
    if kind == "instance":
        asset_stage = Usd.Stage.CreateInMemory()
        asset = UsdGeom.Xform.Define(asset_stage, "/Asset")
        asset_stage.SetDefaultPrim(asset.GetPrim())
        hidden_path = "/Asset/Hidden"
        if schema == "joint":
            UsdPhysics.FixedJoint.Define(asset_stage, hidden_path)
        else:
            hidden = UsdGeom.Xform.Define(asset_stage, hidden_path).GetPrim()
            assert UsdPhysics.ArticulationRootAPI.Apply(hidden)
        instance = UsdGeom.Xform.Define(stage, "/World/HiddenInstance").GetPrim()
        assert instance.GetReferences().AddReference(
            asset_stage.GetRootLayer().identifier,
            "/Asset",
        )
        assert instance.SetInstanceable(True)
        observed_path = "/World/HiddenInstance/Hidden"
        assert stage.GetPrimAtPath(observed_path).IsInstanceProxy()
        return observed_path

    assert kind == "inactive"
    inactive = UsdGeom.Xform.Define(stage, "/World/Inactive")
    observed_path = "/World/Inactive/Hidden"
    if schema == "joint":
        UsdPhysics.FixedJoint.Define(stage, observed_path)
    else:
        hidden = UsdGeom.Xform.Define(stage, observed_path).GetPrim()
        assert UsdPhysics.ArticulationRootAPI.Apply(hidden)
    assert inactive.GetPrim().SetActive(False)
    assert not stage.GetPrimAtPath(observed_path).IsValid()
    return observed_path


@pytest.mark.parametrize(
    "operation",
    [author_physics_schemas, validate_authored_physics_schemas],
    ids=["author", "validate"],
)
@pytest.mark.parametrize(
    ("hidden_kind", "reason_code"),
    (
        ("instance_joint", "unplanned_joint_schema"),
        ("instance_root", "articulation_root_ambiguous"),
        ("inactive_joint", "unplanned_joint_schema"),
        ("inactive_root", "articulation_root_ambiguous"),
    ),
)
def test_schema_contract_rejects_instance_and_inactive_hidden_graph_prims(
    operation: Any,
    hidden_kind: str,
    reason_code: str,
) -> None:
    stage, plan = _revolute_fixture(with_drive=True)
    author_physics_schemas(stage, plan)
    observed_path = _add_hidden_r3_graph_prim(stage, hidden_kind)
    before = stage.GetRootLayer().ExportToString()

    with pytest.raises(JointRiggerContractError) as caught:
        operation(stage, plan)

    assert caught.value.code == reason_code
    assert observed_path in caught.value.detail
    assert stage.GetRootLayer().ExportToString() == before


@pytest.mark.parametrize(
    ("operation", "preauthor"),
    [
        (author_physics_schemas, False),
        (validate_authored_physics_schemas, True),
    ],
    ids=["author", "validate"],
)
def test_articulation_root_discovery_has_fixed_prim_visit_limit(
    monkeypatch: pytest.MonkeyPatch,
    operation: Any,
    preauthor: bool,
) -> None:
    stage, plan = _revolute_fixture()
    if preauthor:
        author_physics_schemas(stage, plan)
    before = stage.GetRootLayer().ExportToString()
    monkeypatch.setattr(
        schemas_module,
        "_MAX_R3_ARTICULATION_ROOT_PRIM_VISITS",
        1,
    )

    with pytest.raises(JointRiggerContractError) as caught:
        operation(stage, plan)

    assert caught.value.code == "r3_articulation_root_scan_limit_exceeded"
    assert "fixed 1-prim visit limit" in caught.value.detail
    assert stage.GetRootLayer().ExportToString() == before


@pytest.mark.parametrize(
    ("operation", "preauthor"),
    [
        (author_physics_schemas, False),
        (validate_authored_physics_schemas, True),
    ],
    ids=["author", "validate"],
)
def test_schema_public_entrypoints_bound_preservation_snapshot_scan(
    monkeypatch: pytest.MonkeyPatch,
    operation: Any,
    preauthor: bool,
) -> None:
    stage, plan = _revolute_fixture()
    if preauthor:
        author_physics_schemas(stage, plan)
    before = stage.GetRootLayer().ExportToString()
    monkeypatch.setattr(
        validation_module,
        "_STAGE_SNAPSHOT_MAX_PRIM_VISITS",
        1,
    )

    with pytest.raises(JointRiggerContractError) as caught:
        operation(stage, plan)

    assert caught.value.code == "stage_snapshot_scan_limit_exceeded"
    assert "fixed 1-prim visit limit" in caught.value.detail
    assert stage.GetRootLayer().ExportToString() == before


def test_articulation_root_discovery_has_fixed_path_retention_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage, plan = _revolute_fixture()
    assert UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath("/World"))
    assert UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath("/World/link"))
    before = stage.GetRootLayer().ExportToString()
    monkeypatch.setattr(
        schemas_module,
        "_MAX_R3_ARTICULATION_ROOT_PATHS",
        1,
    )

    with pytest.raises(JointRiggerContractError) as caught:
        author_physics_schemas(stage, plan)

    assert caught.value.code == "r3_articulation_root_path_limit_exceeded"
    assert "fixed 1-path retention limit" in caught.value.detail
    assert stage.GetRootLayer().ExportToString() == before


def test_articulation_root_discovery_normalizes_inactive_scan_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage, plan = _revolute_fixture()
    _add_hidden_r3_graph_prim(stage, "inactive_root")
    before = stage.GetRootLayer().ExportToString()
    monkeypatch.setattr(validation_module, "_INACTIVE_SCAN_MAX_PRIM_VISITS", 1)

    with pytest.raises(JointRiggerContractError) as caught:
        author_physics_schemas(stage, plan)

    assert caught.value.code == "r3_articulation_root_scan_limit_exceeded"
    assert "inactive-subtree discovery failed" in caught.value.detail
    assert "inactive-joint inspection" in caught.value.detail
    assert stage.GetRootLayer().ExportToString() == before


def test_articulation_root_inactive_discovery_uses_same_path_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage, plan = _revolute_fixture()
    for index in range(2):
        inactive = UsdGeom.Xform.Define(stage, f"/World/Inactive_{index}")
        hidden = UsdGeom.Xform.Define(
            stage,
            f"/World/Inactive_{index}/Hidden",
        ).GetPrim()
        assert UsdPhysics.ArticulationRootAPI.Apply(hidden)
        assert inactive.GetPrim().SetActive(False)
    before = stage.GetRootLayer().ExportToString()
    monkeypatch.setattr(
        schemas_module,
        "_MAX_R3_ARTICULATION_ROOT_PATHS",
        1,
    )

    with pytest.raises(JointRiggerContractError) as caught:
        author_physics_schemas(stage, plan)

    assert caught.value.code == "r3_articulation_root_path_limit_exceeded"
    assert stage.GetRootLayer().ExportToString() == before


@pytest.mark.parametrize(
    ("inner_code", "expected_code"),
    [
        ("source_joint_scan_failed", "r3_articulation_root_scan_failed"),
        ("unrelated_contract_error", "unrelated_contract_error"),
    ],
)
def test_articulation_root_discovery_routes_inactive_scan_errors(
    monkeypatch: pytest.MonkeyPatch,
    inner_code: str,
    expected_code: str,
) -> None:
    stage, plan = _revolute_fixture()
    before = stage.GetRootLayer().ExportToString()
    injected = JointRiggerContractError(inner_code, "injected inactive scan failure")

    def fail_inactive_scan(*_args: Any, **_kwargs: Any) -> set[str]:
        raise injected

    monkeypatch.setattr(
        schemas_module,
        "_paths_with_inactive_ancestors_enabled",
        fail_inactive_scan,
    )

    with pytest.raises(JointRiggerContractError) as caught:
        author_physics_schemas(stage, plan)

    assert caught.value.code == expected_code
    if expected_code == inner_code:
        assert caught.value is injected
    else:
        assert caught.value.__cause__ is injected
    assert stage.GetRootLayer().ExportToString() == before


@pytest.mark.parametrize("hidden_schema", ["joint", "root"])
def test_postwrite_hidden_graph_prim_fails_and_rolls_back(
    monkeypatch: pytest.MonkeyPatch,
    hidden_schema: str,
) -> None:
    stage, plan = _revolute_fixture(with_drive=True)
    before = stage.GetRootLayer().ExportToString()
    real_apply = schemas_module._apply

    def apply_then_hide(stage: Any, plan: Any, preflight: Any) -> None:
        real_apply(stage, plan, preflight)
        inactive = UsdGeom.Xform.Define(stage, "/World/PostwriteInactive")
        hidden_path = "/World/PostwriteInactive/Hidden"
        if hidden_schema == "joint":
            UsdPhysics.FixedJoint.Define(stage, hidden_path)
        else:
            hidden = UsdGeom.Xform.Define(stage, hidden_path).GetPrim()
            assert UsdPhysics.ArticulationRootAPI.Apply(hidden)
        assert inactive.GetPrim().SetActive(False)

    monkeypatch.setattr(schemas_module, "_apply", apply_then_hide)

    with pytest.raises(JointRiggerContractError) as caught:
        author_physics_schemas(stage, plan)

    assert caught.value.code == "postwrite_validation_failed"
    assert "PostwriteInactive/Hidden" in caught.value.detail
    assert stage.GetRootLayer().ExportToString() == before


def test_author_physics_schemas_applies_explicit_mesh_and_revolute_mimic() -> None:
    stage, plan = _mimic_fixture()
    before = capture_joint_rigger_stage_snapshot(stage)

    diagnostics = author_physics_schemas(stage, plan)

    validate_joint_rigger_stage_preservation(
        before,
        capture_joint_rigger_stage_snapshot(stage),
    )
    for body in ("/World/base", "/World/arm", "/World/hand"):
        collider = stage.GetPrimAtPath(f"{body}/collision")
        assert collider.HasAPI(UsdPhysics.CollisionAPI)
        assert collider.HasAPI(UsdPhysics.MeshCollisionAPI)
        assert collider.GetAttribute("physics:approximation").Get() == "convexHull"

    second = stage.GetPrimAtPath("/World/Joints/second")
    schemas = _schema_tokens(second)
    assert "PhysxMimicJointAPI:rotZ" in schemas
    assert "PhysicsDriveAPI:angular" not in schemas
    namespace = "physxMimicJoint:rotZ"
    assert second.GetAttribute(f"{namespace}:referenceJointAxis").Get() == "rotZ"
    assert second.GetAttribute(f"{namespace}:gearing").Get() == -1.0
    assert second.GetRelationship(f"{namespace}:referenceJoint").GetTargets() == [
        Sdf.Path("/World/Joints/first")
    ]
    second_diagnostics = next(
        item
        for item in diagnostics.joint_diagnostics
        if item.joint_id == "/World/Joints/second"
    )
    decisions = {item.field: item for item in second_diagnostics.field_decisions}
    assert decisions["mimic.reference_joint_id"].disposition == "accepted"
    assert decisions["mimic.gearing"].provenance == _PROVENANCE
    assert decisions["drive.drive_type"].disposition == "ignored"


def test_spherical_joint_reports_scalar_state_and_control_not_applicable() -> None:
    stage, plan = _spherical_fixture()

    diagnostics = author_physics_schemas(stage, plan)

    joint = stage.GetPrimAtPath("/World/Joints/ball")
    schemas = _schema_tokens(joint)
    assert not any(token.startswith("PhysicsJointStateAPI:") for token in schemas)
    assert not any(token.startswith("PhysicsDriveAPI:") for token in schemas)
    decisions = {
        item.field: item for item in diagnostics.joint_diagnostics[0].field_decisions
    }
    assert decisions["state.position"].disposition == "ignored"
    assert decisions["state.position"].reason_code == "not_applicable"
    assert decisions["joint_friction.coefficient"].reason_code == "not_applicable"
    assert decisions["drive.drive_type"].reason_code == "not_applicable"
    assert decisions["mimic.reference_joint_id"].reason_code == "not_applicable"
    assert "not applicable" in decisions["state.position"].detail


def _set_raw_api_schema_bucket(
    prim_spec: Any,
    token: str,
    bucket: str,
) -> None:
    if bucket == "explicit":
        schemas = Sdf.TokenListOp.CreateExplicit([token])
    else:
        schemas = Sdf.TokenListOp()
        setattr(schemas, f"{bucket}Items", [token])
    prim_spec.SetInfo("apiSchemas", schemas)


def test_spherical_joint_has_an_explicit_empty_r3_raw_contract() -> None:
    stage, plan = _spherical_fixture()
    author_physics_schemas(stage, plan)

    contracts = schemas_module._r3_raw_authorship_contract(
        stage,
        plan,
        schemas_module._preflight(stage, plan),
    )

    contract = contracts["/World/Joints/ball"]
    assert contract.schema_order == ()
    assert contract.attribute_specs == {}
    assert contract.relationship_targets == {}
    validate_authored_physics_schemas(stage, plan)


@pytest.mark.parametrize(
    "operation",
    [author_physics_schemas, validate_authored_physics_schemas],
    ids=["author", "validate"],
)
@pytest.mark.parametrize(
    ("malformed_token", "bucket"),
    (
        ("PhysicsJointStateAPI", "explicit"),
        ("PhysicsDriveAPI", "added"),
        ("PhysxMimicJointAPI", "prepended"),
        ("PhysicsJointStateAPI", "appended"),
        ("PhysicsDriveAPI", "deleted"),
        ("PhysxMimicJointAPI", "ordered"),
    ),
    ids=[
        "state-explicit",
        "drive-added",
        "mimic-prepended",
        "state-appended",
        "drive-deleted",
        "mimic-ordered",
    ],
)
def test_empty_spherical_contract_rejects_hidden_tokens_in_every_list_op_bucket(
    operation: Any,
    malformed_token: str,
    bucket: str,
) -> None:
    stage, plan = _spherical_fixture()
    author_physics_schemas(stage, plan)
    weaker_layer = Sdf.Layer.CreateAnonymous("hidden-spherical-token.usda")
    stage.GetRootLayer().subLayerPaths.append(weaker_layer.identifier)
    weaker_spec = Sdf.CreatePrimInLayer(weaker_layer, "/World/Joints/ball")
    _set_raw_api_schema_bucket(weaker_spec, malformed_token, bucket)
    session_spec = Sdf.CreatePrimInLayer(
        stage.GetSessionLayer(),
        "/World/Joints/ball",
    )
    session_spec.SetInfo(
        "apiSchemas",
        Sdf.TokenListOp.CreateExplicit(["PhysicsMaterialAPI"]),
    )
    stage.SetEditTarget(stage.GetRootLayer())
    composed_owned = {
        token
        for token in _schema_tokens(stage.GetPrimAtPath("/World/Joints/ball"))
        if schemas_module._is_r3_owned_schema_token(token)
    }
    assert composed_owned == set()

    with pytest.raises(JointRiggerContractError) as caught:
        operation(stage, plan)

    assert caught.value.code == "authored_graph_mismatch"
    assert "contributing raw apiSchemas" in caught.value.detail
    assert malformed_token in caught.value.detail


@pytest.mark.parametrize(
    "operation",
    [author_physics_schemas, validate_authored_physics_schemas],
    ids=["author", "validate"],
)
@pytest.mark.parametrize(
    ("contributing_strength", "malformed_token"),
    (
        ("weaker", "PhysicsJointStateAPI"),
        ("stronger", "PhysicsDriveAPI"),
        ("stronger", "PhysxMimicJointAPI"),
    ),
)
def test_empty_spherical_contract_rejects_weaker_and_stronger_owned_tokens(
    operation: Any,
    contributing_strength: str,
    malformed_token: str,
) -> None:
    stage, plan = _spherical_fixture()
    author_physics_schemas(stage, plan)
    if contributing_strength == "weaker":
        root_spec = stage.GetRootLayer().GetPrimAtPath("/World/Joints/ball")
        assert root_spec is not None
        _set_raw_api_schema_bucket(root_spec, malformed_token, "prepended")
        session_spec = Sdf.CreatePrimInLayer(
            stage.GetSessionLayer(),
            "/World/Joints/ball",
        )
        masking = Sdf.TokenListOp()
        masking.deletedItems = [malformed_token]
        session_spec.SetInfo("apiSchemas", masking)
        stage.SetEditTarget(stage.GetSessionLayer())
    else:
        session_spec = Sdf.CreatePrimInLayer(
            stage.GetSessionLayer(),
            "/World/Joints/ball",
        )
        _set_raw_api_schema_bucket(session_spec, malformed_token, "prepended")
        stage.SetEditTarget(stage.GetRootLayer())

    with pytest.raises(JointRiggerContractError) as caught:
        operation(stage, plan)

    assert caught.value.code in {
        "authored_graph_mismatch",
        "drive_schema_conflict",
        "joint_state_schema_mismatch",
    }
    assert malformed_token.partition(":")[0] in caught.value.detail


def test_empty_spherical_contract_allows_unrelated_api_across_layers() -> None:
    stage, plan = _spherical_fixture()
    author_physics_schemas(stage, plan)
    root_spec = stage.GetRootLayer().GetPrimAtPath("/World/Joints/ball")
    assert root_spec is not None
    _set_raw_api_schema_bucket(root_spec, "PhysicsMaterialAPI", "prepended")
    session_spec = Sdf.CreatePrimInLayer(
        stage.GetSessionLayer(),
        "/World/Joints/ball",
    )
    _set_raw_api_schema_bucket(session_spec, "CollectionAPI:review", "prepended")

    validate_authored_physics_schemas(stage, plan)


@pytest.mark.parametrize(
    ("mutation", "reason_code"),
    [
        ("body_coverage", "body_coverage_mismatch"),
        ("logical_joint_id", "joint_path_required"),
        ("invalid_stage_units", "invalid_stage_units"),
        ("mass_missing", "mass_evidence_missing"),
        ("collider_missing", "collider_evidence_missing"),
        ("root_missing", "articulation_root_missing"),
        ("root_mismatch", "articulation_root_mismatch"),
        ("state_missing", "joint_state_evidence_missing"),
        ("nonzero_new_state", "unsafe_new_joint_state"),
        ("collider_not_gprim", "collider_not_gprim"),
        ("rigid_body_conflict", "physics_schema_conflict"),
        ("time_sampled_property", "time_sampled_owned_property"),
        ("time_sampled_transform", "time_varying_body_transform"),
        (
            "time_sampled_ancestor_transform",
            "time_varying_body_transform",
        ),
        ("extra_articulation_root", "articulation_root_ambiguous"),
        ("extra_joint", "unplanned_joint_schema"),
        ("topology_mismatch", "joint_topology_mismatch"),
        ("axis_mismatch", "axis_mismatch"),
        ("nonzero_new_drive", "unsafe_new_drive_target"),
        ("passive_existing_drive", "passive_control_schema_conflict"),
        ("passive_existing_friction", "passive_control_schema_conflict"),
    ],
)
def test_authoring_preflight_fails_before_any_write(
    mutation: str,
    reason_code: str,
) -> None:
    stage, plan = _revolute_fixture(with_drive=mutation == "nonzero_new_drive")
    if mutation == "body_coverage":
        plan = _replace_plan(plan, rigid_bodies=plan.rigid_bodies[:1])
    elif mutation == "logical_joint_id":
        topology = plan.joints[0].topology.model_copy(update={"joint_id": "hinge"})
        plan = _replace_first_joint(plan, topology=topology)
    elif mutation == "invalid_stage_units":
        UsdGeom.SetStageMetersPerUnit(stage, 0.0)
    elif mutation == "mass_missing":
        plan = _replace_first_body(plan, mass=None)
    elif mutation == "collider_missing":
        plan = _replace_first_body(plan, colliders=())
    elif mutation == "root_missing":
        plan = _replace_plan(plan, articulation_root=None)
    elif mutation == "root_mismatch":
        plan = _replace_plan(
            plan,
            articulation_root=ArticulationRootPlanV1(
                prim_path="/World/link",
                provenance=_PROVENANCE,
            ),
        )
    elif mutation == "state_missing":
        plan = _replace_first_joint(plan, state=None)
    elif mutation == "nonzero_new_state":
        plan = _replace_first_joint(
            plan,
            state=JointStateV1(
                position=1.0,
                velocity=0.0,
                provenance=_PROVENANCE,
            ),
        )
    elif mutation == "collider_not_gprim":
        plan = _replace_first_body(
            plan,
            colliders=(
                ColliderPlanV1(
                    prim_path="/World/base",
                    provenance=_PROVENANCE,
                ),
            ),
        )
    elif mutation == "rigid_body_conflict":
        stage.GetPrimAtPath("/World/base").CreateAttribute(
            "physics:rigidBodyEnabled",
            Sdf.ValueTypeNames.Bool,
            custom=False,
        ).Set(False)
    elif mutation == "time_sampled_property":
        attribute = stage.GetPrimAtPath("/World/base").CreateAttribute(
            "physics:rigidBodyEnabled",
            Sdf.ValueTypeNames.Bool,
            custom=False,
        )
        attribute.Set(True)
        attribute.Set(True, Usd.TimeCode(1.0))
    elif mutation == "time_sampled_transform":
        op = UsdGeom.Xformable(stage.GetPrimAtPath("/World/link")).GetOrderedXformOps()[
            0
        ]
        op.Set(Gf.Vec3d(2.0, 0.0, 0.0), Usd.TimeCode(1.0))
    elif mutation == "time_sampled_ancestor_transform":
        op = UsdGeom.Xformable(stage.GetPrimAtPath("/World")).AddTranslateOp()
        op.Set(Gf.Vec3d(0.0, 0.0, 0.0))
        op.Set(Gf.Vec3d(2.0, 0.0, 0.0), Usd.TimeCode(1.0))
    elif mutation == "extra_articulation_root":
        UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath("/World/link"))
    elif mutation == "extra_joint":
        extra = UsdPhysics.FixedJoint.Define(stage, "/World/Joints/extra")
        extra.CreateBody0Rel().SetTargets(["/World/base"])
        extra.CreateBody1Rel().SetTargets(["/World/link"])
    elif mutation == "topology_mismatch":
        joint = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath("/World/Joints/hinge"))
        joint.GetBody0Rel().SetTargets(["/World/link"])
    elif mutation == "axis_mismatch":
        UsdPhysics.RevoluteJoint(
            stage.GetPrimAtPath("/World/Joints/hinge")
        ).GetAxisAttr().Set("X")
    elif mutation == "nonzero_new_drive":
        assert plan.joints[0].drive is not None
        plan = _replace_first_joint(
            plan,
            drive=plan.joints[0].drive.model_copy(update={"target_position": 5.0}),
        )
    elif mutation == "passive_existing_drive":
        _apply_complete_drive(
            stage.GetPrimAtPath("/World/Joints/hinge"),
            motion="angular",
        )
    elif mutation == "passive_existing_friction":
        prim = stage.GetPrimAtPath("/World/Joints/hinge")
        prim.AddAppliedSchema("PhysxJointAPI")
        prim.CreateAttribute(
            "physxJoint:jointFriction",
            Sdf.ValueTypeNames.Float,
            custom=False,
        ).Set(0.15)
    else:  # pragma: no cover - parameter guard
        raise AssertionError(mutation)

    before = stage.GetRootLayer().ExportToString()
    with pytest.raises(JointRiggerContractError) as caught:
        author_physics_schemas(stage, plan)

    assert caught.value.code == reason_code
    assert stage.GetRootLayer().ExportToString() == before


def test_existing_nonzero_state_and_drive_are_preserved_when_exact() -> None:
    stage, plan = _revolute_fixture(with_drive=True)
    state = JointStateV1(
        position=5.0,
        velocity=1.0,
        provenance=_PROVENANCE,
    )
    assert plan.joints[0].drive is not None
    drive = plan.joints[0].drive.model_copy(
        update={"target_position": 5.0, "target_velocity": 1.0}
    )
    plan = _replace_first_joint(plan, state=state, drive=drive)
    joint = stage.GetPrimAtPath("/World/Joints/hinge")
    joint.AddAppliedSchema("PhysicsJointStateAPI:angular")
    joint.CreateAttribute(
        "state:angular:physics:position",
        Sdf.ValueTypeNames.Float,
        custom=False,
    ).Set(5.0)
    joint.CreateAttribute(
        "state:angular:physics:velocity",
        Sdf.ValueTypeNames.Float,
        custom=False,
    ).Set(1.0)
    _apply_complete_drive(
        joint,
        motion="angular",
        target_position=5.0,
        target_velocity=1.0,
        max_joint_velocity=3.0,
    )

    author_physics_schemas(stage, plan)

    assert joint.GetAttribute("state:angular:physics:position").Get() == 5.0
    assert joint.GetAttribute("drive:angular:physics:targetPosition").Get() == 5.0


def test_authored_joint_anchor_must_match_the_explicit_plan() -> None:
    stage, plan = _revolute_fixture()
    joint = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath("/World/Joints/hinge"))
    joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(-1.0, 0.0, 0.0))
    anchor = JointAnchorV1(
        position_stage=(0.0, 0.0, 0.0),
        provenance=_PROVENANCE,
    )
    plan = _replace_first_joint(plan, anchor=anchor)

    diagnostics = author_physics_schemas(stage, plan)

    decisions = {
        item.field: item for item in diagnostics.joint_diagnostics[0].field_decisions
    }
    assert decisions["anchor.position_stage"].disposition == "accepted"
    assert decisions["anchor.position_stage"].provenance == _PROVENANCE


def test_mismatched_joint_anchor_fails_before_authoring() -> None:
    stage, plan = _revolute_fixture()
    joint = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath("/World/Joints/hinge"))
    joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(-1.0, 0.0, 0.0))
    plan = _replace_first_joint(
        plan,
        anchor=JointAnchorV1(
            position_stage=(2.0, 0.0, 0.0),
            provenance=_PROVENANCE,
        ),
    )
    before = stage.GetRootLayer().ExportToString()

    with pytest.raises(JointRiggerContractError) as caught:
        author_physics_schemas(stage, plan)

    assert caught.value.code == "joint_anchor_mismatch"
    assert stage.GetRootLayer().ExportToString() == before


def test_mimic_requires_matching_signed_axis_and_complete_limits() -> None:
    stage, plan = _mimic_fixture(second_axis="X")
    before = stage.GetRootLayer().ExportToString()

    with pytest.raises(JointRiggerContractError) as caught:
        author_physics_schemas(stage, plan)

    assert caught.value.code == "mimic_axis_mismatch"
    assert stage.GetRootLayer().ExportToString() == before


def test_plan_rejects_collider_not_owned_by_nearest_planned_nested_body() -> None:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/base")
    nested = UsdGeom.Xform.Define(stage, "/World/base/link")
    nested.SetResetXformStack(True)
    UsdGeom.Cube.Define(stage, "/World/base/link/collision")
    UsdGeom.Scope.Define(stage, "/World/Joints")
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/World/Joints/hinge")
    joint.CreateBody0Rel().SetTargets(["/World/base"])
    joint.CreateBody1Rel().SetTargets(["/World/base/link"])
    joint.CreateAxisAttr("Z")
    joint.CreateLowerLimitAttr(-45.0)
    joint.CreateUpperLimitAttr(45.0)
    topology = _topology(
        "/World/Joints/hinge",
        "revolute",
        "/World/base",
        "/World/base/link",
        axis=(0.0, 0.0, 1.0),
    )
    wrong_collider = ColliderPlanV1(
        prim_path="/World/base/link/collision",
        provenance=_PROVENANCE,
    )
    before = stage.GetRootLayer().ExportToString()

    with pytest.raises(ValidationError, match="nearest planned rigid body"):
        JointRiggerPlanV1(
            schema_version=PLAN_SCHEMA_VERSION,
            joints=(
                JointPlanV1(
                    topology=topology,
                    limit=_limit(),
                    state=_zero_state(),
                ),
            ),
            rigid_bodies=(
                _body("/World/base", wrong_collider),
                _body(
                    "/World/base/link",
                    ColliderPlanV1(
                        prim_path="/World/base/link",
                        provenance=_PROVENANCE,
                    ),
                ),
            ),
            articulation_root=ArticulationRootPlanV1(
                prim_path="/World/base",
                provenance=_PROVENANCE,
            ),
        )

    assert stage.GetRootLayer().ExportToString() == before


def test_nested_body_without_existing_reset_preserves_world_transforms() -> None:
    stage, plan = _nested_fixture(reset_nested_body=False)
    nested_path = "/World/base/link"
    nested = UsdGeom.Xformable(stage.GetPrimAtPath(nested_path))
    before = capture_joint_rigger_stage_snapshot(stage)
    before_world = dict(before.world_transforms)[nested_path]
    source_ops = {str(op.GetOpName()): op.Get() for op in nested.GetOrderedXformOps()}

    diagnostics = author_physics_schemas(stage, plan)

    validate_joint_rigger_stage_preservation(
        before,
        capture_joint_rigger_stage_snapshot(stage),
    )
    assert nested.GetResetXformStack()
    assert tuple(str(token) for token in nested.GetXformOpOrderAttr().Get()) == (
        "!resetXformStack!",
        "xformOp:transform:jointRiggerPreserveWorld",
    )
    matrix_op = nested.GetOrderedXformOps()[0]
    matrix = matrix_op.Get()
    assert tuple(
        float(matrix[row][column]) for row in range(4) for column in range(4)
    ) == pytest.approx(before_world)
    for name, value in source_ops.items():
        assert stage.GetPrimAtPath(nested_path).GetAttribute(name).Get() == value
    assert validate_authored_physics_schemas(stage, plan) == diagnostics


def test_nested_body_with_existing_reset_preserves_world_transforms() -> None:
    stage, plan = _nested_fixture(reset_nested_body=True)
    nested = UsdGeom.Xformable(stage.GetPrimAtPath("/World/base/link"))
    before_order = tuple(str(token) for token in nested.GetXformOpOrderAttr().Get())
    before = capture_joint_rigger_stage_snapshot(stage)

    author_physics_schemas(stage, plan)

    validate_joint_rigger_stage_preservation(
        before,
        capture_joint_rigger_stage_snapshot(stage),
    )
    assert (
        tuple(str(token) for token in nested.GetXformOpOrderAttr().Get())
        == before_order
    )


def test_nested_body_reset_rejects_reserved_property_conflict() -> None:
    stage, plan = _nested_fixture(reset_nested_body=False)
    nested = stage.GetPrimAtPath("/World/base/link")
    nested.CreateAttribute(
        "xformOp:transform:jointRiggerPreserveWorld",
        Sdf.ValueTypeNames.String,
        custom=True,
    ).Set("source-owned")
    before = stage.GetRootLayer().ExportToString()

    with pytest.raises(JointRiggerContractError) as caught:
        author_physics_schemas(stage, plan)

    assert caught.value.code == "nested_body_reset_conflict"
    assert stage.GetRootLayer().ExportToString() == before


def test_nested_body_reset_rejects_sampled_xform_op_order() -> None:
    stage, plan = _nested_fixture(reset_nested_body=False)
    nested = UsdGeom.Xformable(stage.GetPrimAtPath("/World/base/link"))
    order = nested.GetXformOpOrderAttr()
    assert order.Set(order.Get(), Usd.TimeCode(1.0))
    before = stage.GetRootLayer().ExportToString()

    with pytest.raises(JointRiggerContractError) as caught:
        author_physics_schemas(stage, plan)

    assert caught.value.code == "time_varying_body_transform"
    assert "xformOpOrder samples" in caught.value.detail
    assert stage.GetRootLayer().ExportToString() == before


def test_nested_body_resets_preserve_referenced_geometry_and_composition() -> None:
    stage, plan, source_stage = _nested_reference_fixture()
    source_before = source_stage.GetRootLayer().ExportToString()
    root_spec = stage.GetRootLayer().GetPrimAtPath("/World/base")
    assert root_spec is not None
    references_before = tuple(
        str(reference)
        for reference in root_spec.GetInfo("references").GetAddedOrExplicitItems()
    )
    collider_paths = tuple(
        collider.prim_path for body in plan.rigid_bodies for collider in body.colliders
    )
    sizes_before = {
        path: UsdGeom.Cube(stage.GetPrimAtPath(path)).GetSizeAttr().Get()
        for path in collider_paths
    }
    before = capture_joint_rigger_stage_snapshot(stage)

    diagnostics = author_physics_schemas(stage, plan)

    validate_joint_rigger_stage_preservation(
        before,
        capture_joint_rigger_stage_snapshot(stage),
    )
    assert source_stage.GetRootLayer().ExportToString() == source_before
    root_spec = stage.GetRootLayer().GetPrimAtPath("/World/base")
    assert root_spec is not None
    assert (
        tuple(
            str(reference)
            for reference in root_spec.GetInfo("references").GetAddedOrExplicitItems()
        )
        == references_before
    )
    assert {
        path: UsdGeom.Cube(stage.GetPrimAtPath(path)).GetSizeAttr().Get()
        for path in collider_paths
    } == sizes_before
    for path in (
        "/World/base/wheel",
        "/World/base/wheel/tire",
        "/World/base/drawer",
    ):
        xformable = UsdGeom.Xformable(stage.GetPrimAtPath(path))
        assert xformable.GetResetXformStack()
        assert tuple(str(token) for token in xformable.GetXformOpOrderAttr().Get()) == (
            "!resetXformStack!",
            "xformOp:transform:jointRiggerPreserveWorld",
        )
    assert validate_authored_physics_schemas(stage, plan) == diagnostics


@pytest.mark.parametrize(
    ("family", "property_name", "reason_code"),
    [
        (
            "state",
            "state:linear:physics:position",
            "joint_state_schema_conflict",
        ),
        (
            "drive",
            "drive:angular:physics:stiffness",
            "passive_control_schema_conflict",
        ),
        (
            "mimic",
            "physxMimicJoint:rotZ:gearing",
            "passive_control_schema_conflict",
        ),
    ],
)
def test_unplanned_raw_state_and_control_properties_fail_closed(
    family: str,
    property_name: str,
    reason_code: str,
) -> None:
    stage, plan = _revolute_fixture()
    joint = stage.GetPrimAtPath("/World/Joints/hinge")
    joint.CreateAttribute(
        property_name,
        Sdf.ValueTypeNames.Float,
        custom=False,
    ).Set(1.0)
    before = stage.GetRootLayer().ExportToString()

    with pytest.raises(JointRiggerContractError) as caught:
        author_physics_schemas(stage, plan)

    assert caught.value.code == reason_code, family
    assert stage.GetRootLayer().ExportToString() == before


def test_non_mesh_collider_rejects_mesh_collision_schema_evidence() -> None:
    stage, plan = _revolute_fixture()
    collider = stage.GetPrimAtPath("/World/base/collision")
    UsdPhysics.MeshCollisionAPI.Apply(collider).CreateApproximationAttr("convexHull")
    before = stage.GetRootLayer().ExportToString()

    with pytest.raises(JointRiggerContractError) as caught:
        author_physics_schemas(stage, plan)

    assert caught.value.code == "collider_schema_conflict"
    assert stage.GetRootLayer().ExportToString() == before


@pytest.mark.parametrize(
    ("mesh_collision_api", "mesh_approximation", "expects_api"),
    [
        (None, None, False),
        (True, None, True),
        (None, "convexHull", True),
    ],
)
def test_mesh_collider_preserves_the_three_representable_api_states(
    mesh_collision_api: Literal[True] | None,
    mesh_approximation: Literal["convexHull"] | None,
    expects_api: bool,
) -> None:
    stage, plan = _revolute_fixture(
        mesh=True,
        mesh_collision_api=mesh_collision_api,
        mesh_approximation=mesh_approximation,
    )

    diagnostics = author_physics_schemas(stage, plan)

    decisions = {item.field: item for item in diagnostics.field_decisions}
    for body_path in ("/World/base", "/World/link"):
        collider_path = f"{body_path}/collision"
        collider = stage.GetPrimAtPath(collider_path)
        assert collider.HasAPI(UsdPhysics.CollisionAPI)
        assert collider.HasAPI(UsdPhysics.MeshCollisionAPI) is expects_api
        approximation = collider.GetAttribute("physics:approximation")
        assert bool(approximation and approximation.HasAuthoredValueOpinion()) is (
            mesh_approximation is not None
        )
        if mesh_approximation is not None:
            assert approximation.Get() == mesh_approximation
        prefix = f"rigid_bodies[{body_path}].colliders[{collider_path}]"
        assert decisions[f"{prefix}.mesh_collision_api"].disposition == (
            "accepted" if expects_api else "ignored"
        )
        assert decisions[f"{prefix}.mesh_approximation"].disposition == (
            "accepted" if mesh_approximation is not None else "ignored"
        )


@pytest.mark.parametrize("approximation", ["none", "convexHull"])
def test_xform_instance_root_collider_round_trips_exact_standard_schemas(
    tmp_path: Path,
    approximation: Literal["none", "convexHull"],
) -> None:
    stage, plan = _revolute_fixture()
    plan = _replace_colliders_with_xform_instance_roots(
        stage,
        plan,
        mesh_approximation=approximation,
    )
    before = capture_joint_rigger_stage_snapshot(stage)

    diagnostics = author_physics_schemas(stage, plan)

    validate_joint_rigger_stage_preservation(
        before,
        capture_joint_rigger_stage_snapshot(stage),
    )
    for body_path in ("/World/base", "/World/link"):
        collider_path = f"{body_path}/collision"
        collider = stage.GetPrimAtPath(collider_path)
        assert collider.IsInstance()
        assert collider.HasAPI(UsdPhysics.CollisionAPI)
        assert collider.HasAPI(UsdPhysics.MeshCollisionAPI)
        assert collider.GetAttribute("physics:approximation").Get() == approximation
        prefix = f"rigid_bodies[{body_path}].colliders[{collider_path}]"
        decisions = {item.field: item for item in diagnostics.field_decisions}
        assert decisions[f"{prefix}.collision"].disposition == "accepted"
        assert decisions[f"{prefix}.mesh_collision_api"].disposition == "accepted"
        assert decisions[f"{prefix}.mesh_approximation"].disposition == "accepted"

    output = tmp_path / f"xform-instance-{approximation}.usda"
    assert stage.GetRootLayer().Export(str(output))
    reopened = Usd.Stage.Open(str(output))
    assert reopened is not None
    validate_authored_physics_schemas(reopened, plan)


def test_non_instance_xform_collider_fails_before_any_write() -> None:
    stage, plan = _revolute_fixture()
    collider_path = "/World/base/collision"
    assert stage.RemovePrim(collider_path)
    UsdGeom.Xform.Define(stage, collider_path)
    collider = ColliderPlanV1(
        prim_path=collider_path,
        mesh_approximation="none",
        provenance=_PROVENANCE,
    )
    plan = _replace_first_body(plan, colliders=(collider,))
    before = stage.GetRootLayer().ExportToString()

    with pytest.raises(JointRiggerContractError) as caught:
        author_physics_schemas(stage, plan)

    assert caught.value.code == "collider_not_gprim"
    assert stage.GetRootLayer().ExportToString() == before


def test_xform_instance_root_requires_explicit_approximation() -> None:
    stage, plan = _revolute_fixture()
    plan = _replace_colliders_with_xform_instance_roots(
        stage,
        plan,
        mesh_approximation=None,
        mesh_collision_api=True,
    )
    before = stage.GetRootLayer().ExportToString()

    with pytest.raises(JointRiggerContractError) as caught:
        author_physics_schemas(stage, plan)

    assert caught.value.code == "instance_root_collider_evidence_incomplete"
    assert stage.GetRootLayer().ExportToString() == before


def test_non_mesh_collider_rejects_a_planned_mesh_collision_api() -> None:
    stage, plan = _revolute_fixture()
    collider = (
        plan.rigid_bodies[0]
        .colliders[0]
        .model_copy(update={"mesh_collision_api": True})
    )
    plan = _replace_first_body(plan, colliders=(collider,))
    before = stage.GetRootLayer().ExportToString()

    with pytest.raises(JointRiggerContractError) as caught:
        author_physics_schemas(stage, plan)

    assert caught.value.code == "mesh_collision_api_not_applicable"
    assert stage.GetRootLayer().ExportToString() == before


def test_mesh_collider_rejects_an_existing_unplanned_mesh_api() -> None:
    stage, plan = _revolute_fixture(
        mesh=True,
        mesh_collision_api=None,
        mesh_approximation=None,
    )
    collider = stage.GetPrimAtPath("/World/base/collision")
    UsdPhysics.MeshCollisionAPI.Apply(collider)
    before = stage.GetRootLayer().ExportToString()

    with pytest.raises(JointRiggerContractError) as caught:
        author_physics_schemas(stage, plan)

    assert caught.value.code == "collider_schema_conflict"
    assert stage.GetRootLayer().ExportToString() == before


@pytest.mark.parametrize(
    ("target_kind", "value_type"),
    [
        ("axis", Sdf.ValueTypeNames.Token),
        ("local_pos0", Sdf.ValueTypeNames.Float3),
        ("rigid_body", Sdf.ValueTypeNames.Bool),
        ("state", Sdf.ValueTypeNames.Float),
    ],
)
def test_authored_connections_on_owned_attributes_fail_closed(
    target_kind: str,
    value_type: Any,
) -> None:
    stage, plan = _revolute_fixture()
    joint = stage.GetPrimAtPath("/World/Joints/hinge")
    if target_kind == "axis":
        target = joint.GetAttribute("physics:axis")
    elif target_kind == "local_pos0":
        target = joint.CreateAttribute(
            "physics:localPos0",
            Sdf.ValueTypeNames.Float3,
            custom=False,
        )
    elif target_kind == "rigid_body":
        target = stage.GetPrimAtPath("/World/base").CreateAttribute(
            "physics:rigidBodyEnabled",
            Sdf.ValueTypeNames.Bool,
            custom=False,
        )
    else:
        target = joint.CreateAttribute(
            "state:angular:physics:position",
            Sdf.ValueTypeNames.Float,
            custom=False,
        )
    driver = stage.GetPrimAtPath("/World").CreateAttribute(
        f"outputs:{target_kind}",
        value_type,
        custom=True,
    )
    assert target.AddConnection(driver.GetPath())
    assert target.HasAuthoredConnections()
    before = stage.GetRootLayer().ExportToString()

    with pytest.raises(JointRiggerContractError) as caught:
        author_physics_schemas(stage, plan)

    assert caught.value.code == "connected_owned_property"
    assert stage.GetRootLayer().ExportToString() == before


def test_authored_empty_connection_list_is_rejected_and_snapshotted() -> None:
    stage, plan = _revolute_fixture()
    axis = stage.GetPrimAtPath("/World/Joints/hinge").GetAttribute("physics:axis")
    assert axis.SetConnections([])
    assert axis.HasAuthoredConnections()
    assert axis.GetConnections() == []

    structural = capture_joint_rigger_stage_snapshot(stage)
    topology_row = next(
        row for row in structural.joint_topology if row[0] == "/World/Joints/hinge"
    )
    axis_row = next(row for row in topology_row[4] if row[0] == "physics:axis")
    assert axis_row[2] is True
    assert axis_row[3] == ()
    physics = capture_joint_rigger_physics_schema_snapshot(stage)
    prim_row = next(row for row in physics.prims if row[0] == "/World/Joints/hinge")
    property_row = next(row for row in prim_row[2] if row[0] == "physics:axis")
    assert property_row[2][2] is True
    assert property_row[2][3] == ()

    with pytest.raises(JointRiggerContractError) as caught:
        author_physics_schemas(stage, plan)

    assert caught.value.code == "connected_owned_property"


@pytest.mark.parametrize("failure_point", ["apply", "postvalidation", "preservation"])
def test_direct_schema_authoring_restores_the_edit_layer_on_unexpected_failure(
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    stage, plan = _revolute_fixture(with_drive=True)
    before_text = stage.GetRootLayer().ExportToString()
    before_schema = capture_joint_rigger_physics_schema_snapshot(stage)

    if failure_point == "apply":

        def fail_during_apply(*args: Any, **kwargs: Any) -> None:
            del kwargs
            target_stage = args[0]
            prim = target_stage.GetPrimAtPath("/World/base")
            UsdPhysics.RigidBodyAPI.Apply(prim).CreateRigidBodyEnabledAttr(True)
            raise RuntimeError("injected partial apply failure")

        monkeypatch.setattr(schemas_module, "_apply", fail_during_apply)
        expected_code = "physics_schema_authoring_failed"
    elif failure_point == "postvalidation":

        def fail_postvalidation(*args: Any, **kwargs: Any) -> None:
            del args, kwargs
            raise RuntimeError("injected postvalidation failure")

        monkeypatch.setattr(schemas_module, "_validate_authored", fail_postvalidation)
        expected_code = "physics_schema_authoring_failed"
    else:

        def fail_preservation(*args: Any, **kwargs: Any) -> None:
            del args, kwargs
            raise JointRiggerContractError(
                "injected_preservation_failure",
                "injected preservation failure",
            )

        monkeypatch.setattr(
            schemas_module,
            "validate_joint_rigger_stage_preservation",
            fail_preservation,
        )
        expected_code = "injected_preservation_failure"

    with pytest.raises(JointRiggerContractError) as caught:
        author_physics_schemas(stage, plan)

    assert caught.value.code == expected_code
    assert stage.GetRootLayer().ExportToString() == before_text
    assert capture_joint_rigger_physics_schema_snapshot(stage) == before_schema


def test_fatal_schema_authoring_failure_restores_layer_and_remains_primary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage, plan = _revolute_fixture(with_drive=True)
    before_text = stage.GetRootLayer().ExportToString()
    before_schema = capture_joint_rigger_physics_schema_snapshot(stage)
    primary = KeyboardInterrupt("fatal postvalidation interruption")

    def fail_postvalidation(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise primary

    monkeypatch.setattr(schemas_module, "_validate_authored", fail_postvalidation)

    with pytest.raises(KeyboardInterrupt) as caught:
        author_physics_schemas(stage, plan)

    assert caught.value is primary
    assert stage.GetRootLayer().ExportToString() == before_text
    assert capture_joint_rigger_physics_schema_snapshot(stage) == before_schema


@pytest.mark.parametrize(
    ("primary_kind", "rollback_kind"),
    [
        ("ordinary", "ordinary"),
        ("fatal", "ordinary"),
        ("ordinary", "fatal"),
        ("fatal", "fatal"),
    ],
)
def test_schema_authoring_failure_priority_preserves_exact_exceptions_and_layer(
    monkeypatch: pytest.MonkeyPatch,
    primary_kind: str,
    rollback_kind: str,
) -> None:
    stage, plan = _revolute_fixture(with_drive=True)
    before_text = stage.GetRootLayer().ExportToString()
    before_schema = capture_joint_rigger_physics_schema_snapshot(stage)
    primary: BaseException = (
        RuntimeError("ordinary authoring failure")
        if primary_kind == "ordinary"
        else KeyboardInterrupt("fatal authoring interruption")
    )
    rollback_failure: BaseException = (
        OSError("ordinary rollback failure")
        if rollback_kind == "ordinary"
        else SystemExit("fatal rollback interruption")
    )
    original_rollback = schemas_module._rollback_edit_layer
    rollback_observed = False

    def fail_postvalidation(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise primary

    def restore_then_fail(layer: Any, backup: Any) -> None:
        nonlocal rollback_observed
        original_rollback(layer, backup)
        rollback_observed = True
        assert layer.ExportToString() == before_text
        raise rollback_failure

    monkeypatch.setattr(schemas_module, "_validate_authored", fail_postvalidation)
    monkeypatch.setattr(schemas_module, "_rollback_edit_layer", restore_then_fail)

    if primary_kind == "fatal":
        with pytest.raises(KeyboardInterrupt) as caught:
            author_physics_schemas(stage, plan)
        assert caught.value is primary
        notes = "\n".join(getattr(primary, "__notes__", ()))
        assert "Physics schema rollback also failed" in notes
        assert f"{type(rollback_failure).__name__}: {rollback_failure}" in notes
    elif rollback_kind == "fatal":
        with pytest.raises(SystemExit) as caught:
            author_physics_schemas(stage, plan)
        assert caught.value is rollback_failure
        assert caught.value.__cause__ is primary
        notes = "\n".join(getattr(rollback_failure, "__notes__", ()))
        assert "Physics schema authoring also failed" in notes
        assert f"{type(primary).__name__}: {primary}" in notes
    else:
        with pytest.raises(JointRiggerContractError) as caught:
            author_physics_schemas(stage, plan)
        assert caught.value.code == "physics_schema_rollback_failed"
        cause = caught.value.__cause__
        assert isinstance(cause, ExceptionGroup)
        assert cause.exceptions == (primary, rollback_failure)

    assert rollback_observed
    assert stage.GetRootLayer().ExportToString() == before_text
    assert capture_joint_rigger_physics_schema_snapshot(stage) == before_schema


def test_physics_plan_evidence_rejects_template_defaults_without_a_stage() -> None:
    _, plan = _revolute_fixture()
    defaulted = FieldProvenanceV1(
        source="template_default",
        evidence="A template default is not accepted source evidence.",
    )
    assert plan.rigid_bodies[0].mass is not None
    mass = plan.rigid_bodies[0].mass.model_copy(update={"provenance": defaulted})
    plan = _replace_first_body(plan, mass=mass)

    with pytest.raises(JointRiggerContractError) as caught:
        validate_physics_plan_evidence(plan)

    assert caught.value.code == "physics_evidence_not_source_backed"
    assert "mass and inertia" in caught.value.detail


def test_joint_friction_requires_approved_physics_plan_provenance() -> None:
    _, plan = _revolute_fixture(joint_friction=0.15)
    defaulted = FieldProvenanceV1(
        source="template_default",
        evidence="A template default is not accepted joint-friction evidence.",
    )
    friction = plan.joints[0].joint_friction
    assert friction is not None
    plan = _replace_first_joint(
        plan,
        joint_friction=friction.model_copy(update={"provenance": defaulted}),
    )

    with pytest.raises(JointRiggerContractError) as caught:
        validate_physics_plan_evidence(plan)

    assert caught.value.code == "physics_evidence_not_source_backed"
    assert "joint friction" in caught.value.detail


@pytest.mark.parametrize(
    "conflict", ["mass_attribute", "state_attribute", "mimic_relation"]
)
def test_opposite_property_kind_conflicts_fail_before_any_write(
    conflict: str,
) -> None:
    if conflict == "mimic_relation":
        stage, plan = _mimic_fixture()
        prim = stage.GetPrimAtPath("/World/Joints/second")
        prim.CreateAttribute(
            "physxMimicJoint:rotZ:referenceJoint",
            Sdf.ValueTypeNames.String,
            custom=False,
        ).Set("/World/Joints/first")
        expected_code = "mimic_schema_conflict"
    else:
        stage, plan = _revolute_fixture()
        if conflict == "mass_attribute":
            prim = stage.GetPrimAtPath("/World/base")
            name = "physics:mass"
        else:
            prim = stage.GetPrimAtPath("/World/Joints/hinge")
            name = "state:angular:physics:position"
        prim.CreateRelationship(name, custom=False)
        expected_code = "physics_schema_conflict"
    before = stage.GetRootLayer().ExportToString()

    with pytest.raises(JointRiggerContractError) as caught:
        author_physics_schemas(stage, plan)

    assert caught.value.code == expected_code
    assert stage.GetRootLayer().ExportToString() == before


@pytest.mark.parametrize(
    ("mutation", "reason_code"),
    [
        ("hierarchy", "source_hierarchy_changed"),
        ("topology", "joint_topology_changed"),
        ("axis", "joint_topology_changed"),
        ("transform", "world_transform_changed"),
    ],
)
def test_stage_preservation_validator_reports_exact_changed_boundary(
    mutation: str,
    reason_code: str,
) -> None:
    stage, _ = _revolute_fixture()
    before = capture_joint_rigger_stage_snapshot(stage)
    if mutation == "hierarchy":
        UsdGeom.Scope.Define(stage, "/World/extra")
    elif mutation == "topology":
        UsdPhysics.RevoluteJoint(
            stage.GetPrimAtPath("/World/Joints/hinge")
        ).GetBody1Rel().SetTargets(["/World/base"])
    elif mutation == "axis":
        UsdPhysics.RevoluteJoint(
            stage.GetPrimAtPath("/World/Joints/hinge")
        ).GetAxisAttr().Set("X")
    else:
        UsdGeom.Xformable(stage.GetPrimAtPath("/World/link")).GetOrderedXformOps()[
            0
        ].Set(Gf.Vec3d(2.0, 0.0, 0.0))
    after = capture_joint_rigger_stage_snapshot(stage)

    with pytest.raises(JointRiggerContractError) as caught:
        validate_joint_rigger_stage_preservation(before, after)

    assert caught.value.code == reason_code


def test_validation_helpers_reject_missing_stage() -> None:
    with pytest.raises(JointRiggerContractError) as snapshot:
        capture_joint_rigger_stage_snapshot(None)
    assert snapshot.value.code == "invalid_stage"
    with pytest.raises(JointRiggerContractError) as schema_snapshot:
        capture_joint_rigger_physics_schema_snapshot(None)
    assert schema_snapshot.value.code == "invalid_stage"
    with pytest.raises(JointRiggerContractError) as counts:
        physics_schema_counts(None)
    assert counts.value.code == "invalid_stage"


@pytest.mark.parametrize(
    "operation",
    [author_physics_schemas, validate_authored_physics_schemas],
    ids=["author", "validate"],
)
@pytest.mark.parametrize(
    ("kwargs", "reason_code"),
    [
        ({"backend_name": " "}, "invalid_backend_name"),
        ({"backend_version": " "}, "invalid_backend_version"),
    ],
    ids=["blank-name", "blank-version"],
)
def test_blank_backend_identity_fails_before_authoring(
    operation: Any,
    kwargs: dict[str, str],
    reason_code: str,
) -> None:
    stage, plan = _revolute_fixture()
    before = stage.GetRootLayer().ExportToString()

    with pytest.raises(JointRiggerContractError) as caught:
        operation(stage, plan, **kwargs)

    assert caught.value.code == reason_code
    assert stage.GetRootLayer().ExportToString() == before


@pytest.mark.parametrize(
    ("guard", "reason_code"),
    [
        ("author-stage", "invalid_stage"),
        ("author-plan", "invalid_plan"),
        ("evidence-plan", "invalid_plan"),
        ("evidence-empty", "physics_plan_incomplete"),
        ("validate-stage", "invalid_stage"),
        ("validate-plan", "invalid_plan"),
    ],
)
def test_schema_public_entrypoint_argument_guards(
    guard: str,
    reason_code: str,
) -> None:
    stage, plan = _revolute_fixture()
    with pytest.raises(JointRiggerContractError) as caught:
        if guard == "author-stage":
            author_physics_schemas(None, plan)
        elif guard == "author-plan":
            author_physics_schemas(stage, object())  # type: ignore[arg-type]
        elif guard == "evidence-plan":
            validate_physics_plan_evidence(object())  # type: ignore[arg-type]
        elif guard == "evidence-empty":
            empty = JointRiggerPlanV1.model_construct(
                schema_version=PLAN_SCHEMA_VERSION,
                joints=(),
                rigid_bodies=(),
                articulation_root=None,
            )
            validate_physics_plan_evidence(empty)
        elif guard == "validate-stage":
            validate_authored_physics_schemas(None, plan)
        else:
            validate_authored_physics_schemas(
                stage,
                object(),  # type: ignore[arg-type]
            )
    assert caught.value.code == reason_code


@pytest.mark.parametrize(
    ("operation", "reason_code"),
    [
        (author_physics_schemas, "physics_schema_preflight_failed"),
        (validate_authored_physics_schemas, "physics_schema_validation_failed"),
    ],
    ids=["author", "validate"],
)
def test_schema_public_entrypoints_normalize_openusd_preflight_errors(
    monkeypatch: pytest.MonkeyPatch,
    operation: Any,
    reason_code: str,
) -> None:
    stage, plan = _revolute_fixture()
    before = stage.GetRootLayer().ExportToString()

    def fail_preflight(*_args: Any, **_kwargs: Any) -> None:
        raise Tf.ErrorException("injected OpenUSD preflight failure")

    monkeypatch.setattr(schemas_module, "_preflight", fail_preflight)

    with pytest.raises(JointRiggerContractError) as caught:
        operation(stage, plan)

    assert caught.value.code == reason_code
    assert "ErrorException" in caught.value.detail
    assert stage.GetRootLayer().ExportToString() == before


def test_schema_edit_layer_backup_rejects_read_only_layer() -> None:
    layer = SimpleNamespace(permissionToEdit=False)
    edit_target = SimpleNamespace(GetLayer=lambda: layer)
    stage = SimpleNamespace(GetEditTarget=lambda: edit_target)
    with pytest.raises(JointRiggerContractError) as caught:
        schemas_module._backup_edit_layer(stage)
    assert caught.value.code == "physics_schema_edit_layer_unavailable"


def test_raw_contract_rejects_conflicting_canonical_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage, plan = _revolute_fixture(with_drive=True)
    preflight = schemas_module._preflight(stage, plan)
    monkeypatch.setattr(schemas_module, "_stored_values_equal", lambda *_: False)

    with pytest.raises(JointRiggerContractError) as caught:
        schemas_module._r3_raw_authorship_contract(stage, plan, preflight)

    assert caught.value.code == "physics_raw_contract_ambiguous"
    assert "conflicting defaults" in caught.value.detail


@pytest.mark.parametrize(
    "violation",
    ["wrong_property_kind", "spline", "connectionPaths"],
)
def test_contributing_attribute_spec_low_level_guards(violation: str) -> None:
    class FakeAttributeSpec:
        name = "physics:mass"
        custom = False
        typeName = "float"
        variability = "varying"
        path = "/World/base.physics:mass"
        layer = SimpleNamespace(ListTimeSamplesForPath=lambda _path: ())

        def ListInfoKeys(self) -> list[str]:
            return [violation] if violation != "wrong_property_kind" else []

        def GetInfo(self, _name: str) -> float:
            return 1.0

    property_spec: Any = FakeAttributeSpec()
    if violation == "wrong_property_kind":
        property_spec = SimpleNamespace(name="physics:mass")
    prim_spec = SimpleNamespace(properties=(property_spec,))
    prim = SimpleNamespace(GetPrimStack=lambda: (prim_spec,))
    contract = schemas_module._R3RawAuthorshipContract(
        schema_order=(),
        attribute_specs={"physics:mass": ("float", "varying")},
        attribute_defaults={"physics:mass": 1.0},
        relationship_targets={},
    )
    fake_sdf = SimpleNamespace(AttributeSpec=FakeAttributeSpec)

    with pytest.raises(JointRiggerContractError) as caught:
        schemas_module._validate_contributing_r3_attribute_specs(
            prim,
            "/World/base",
            contract,
            Sdf=fake_sdf,
            type_by_name={"float": "float"},
            variability_by_name={"varying": "varying"},
        )

    assert caught.value.code == "authored_graph_mismatch"
    if violation == "spline":
        assert "spline" in caught.value.detail
    elif violation == "connectionPaths":
        assert "connection" in caught.value.detail


def test_value_clip_asset_missing_after_resolution_fails_closed(tmp_path: Path) -> None:
    source = Sdf.Layer.CreateAnonymous("missing-clip-source.usda")
    missing = tmp_path / "does-not-exist.usda"

    with pytest.raises(JointRiggerContractError) as caught:
        schemas_module._clip_asset_carries_owned_samples(
            Sdf.AssetPath(str(missing)),
            source_layer=source,
            clip_prim_path=Sdf.Path("/Clip"),
            relative_prim_path=Sdf.Path("base"),
            attribute_names=frozenset({"physics:mass"}),
            include_r3_owned_property_names=False,
            include_xform_property_names=False,
            resolver_context=Ar.ResolverContext(),
            Sdf=Sdf,
        )

    assert caught.value.code == "authored_graph_mismatch"
    assert "cannot be inspected" in caught.value.detail


def test_value_clip_audit_returns_for_an_empty_boundary() -> None:
    schemas_module._validate_contributing_r3_value_clips(
        object(),
        "/World/base",
        frozenset(),
        Sdf=Sdf,
        inspection_keys=set(),
        include_r3_owned_property_names=False,
        include_xform_property_names=False,
        resolver_context=Ar.ResolverContext(),
    )


@pytest.mark.parametrize(
    ("raw_clips", "reason_fragment"),
    [
        (7, "malformed raw clips"),
        ({"bad": 7}, "clip set 'bad' is malformed"),
    ],
)
def test_value_clip_audit_rejects_malformed_raw_clip_dictionaries(
    raw_clips: Any,
    reason_fragment: str,
) -> None:
    prim_spec = SimpleNamespace(
        ListInfoKeys=lambda: ("clips",),
        GetInfo=lambda _name: raw_clips,
    )
    prim = SimpleNamespace(
        GetPrimStack=lambda: (prim_spec,),
        GetMetadata=lambda _name: {},
    )
    stage = SimpleNamespace(
        GetPrimAtPath=lambda path: prim if str(path) == "/World/base" else None
    )

    with pytest.raises(JointRiggerContractError) as caught:
        schemas_module._validate_contributing_r3_value_clips(
            stage,
            "/World/base",
            frozenset({"physics:mass"}),
            Sdf=Sdf,
            inspection_keys=set(),
            include_r3_owned_property_names=False,
            include_xform_property_names=False,
            resolver_context=Ar.ResolverContext(),
        )

    assert caught.value.code == "authored_graph_mismatch"
    assert reason_fragment in caught.value.detail


@pytest.mark.parametrize(
    ("clip_case", "reason_fragment"),
    [
        ("no_prim_path", "no inspectable primPath"),
        ("invalid_prim_path", "invalid primPath"),
    ],
)
def test_value_clip_audit_rejects_uninspectable_prim_paths(
    clip_case: str,
    reason_fragment: str,
) -> None:
    settings: dict[str, Any] = {
        "assetPaths": Sdf.AssetPathArray([Sdf.AssetPath("unused.usda")])
    }
    if clip_case == "invalid_prim_path":
        settings["primPath"] = "relative"
    raw_clips = {"bad": settings}
    prim_spec = SimpleNamespace(
        layer=SimpleNamespace(identifier="fake-layer"),
        ListInfoKeys=lambda: ("clips",),
        GetInfo=lambda _name: raw_clips,
    )
    prim = SimpleNamespace(
        GetPrimStack=lambda: (prim_spec,),
        GetMetadata=lambda _name: raw_clips,
    )
    stage = SimpleNamespace(
        GetPrimAtPath=lambda path: prim if str(path) == "/World/base" else None
    )

    with pytest.raises(JointRiggerContractError) as caught:
        schemas_module._validate_contributing_r3_value_clips(
            stage,
            "/World/base",
            frozenset({"physics:mass"}),
            Sdf=Sdf,
            inspection_keys=set(),
            include_r3_owned_property_names=False,
            include_xform_property_names=False,
            resolver_context=Ar.ResolverContext(),
        )

    assert caught.value.code == "authored_graph_mismatch"
    assert reason_fragment in caught.value.detail


def test_value_clip_audit_skips_empty_assets_and_duplicate_inspections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_clips = {
        "empty": {
            "assetPaths": Sdf.AssetPathArray(),
            "primPath": "/Clip",
        },
        "first": {
            "assetPaths": Sdf.AssetPathArray([Sdf.AssetPath("same.usda")]),
            "primPath": "/Clip",
        },
        "second": {
            "assetPaths": Sdf.AssetPathArray([Sdf.AssetPath("same.usda")]),
            "primPath": "/Clip",
        },
    }
    prim_spec = SimpleNamespace(
        layer=SimpleNamespace(identifier="fake-layer"),
        ListInfoKeys=lambda: ("clips",),
        GetInfo=lambda _name: raw_clips,
    )
    prim = SimpleNamespace(
        GetPrimStack=lambda: (prim_spec,),
        GetMetadata=lambda _name: raw_clips,
    )
    stage = SimpleNamespace(
        GetPrimAtPath=lambda path: prim if str(path) == "/World/base" else None
    )
    inspected: list[str] = []

    def inspect(asset_path: Any, **_kwargs: Any) -> tuple[bool, str]:
        inspected.append(str(asset_path))
        return False, "unused"

    monkeypatch.setattr(schemas_module, "_clip_asset_carries_owned_samples", inspect)
    keys: set[tuple[str, str, str, str, tuple[str, ...]]] = set()
    schemas_module._validate_contributing_r3_value_clips(
        stage,
        "/World/base",
        frozenset({"physics:mass"}),
        Sdf=Sdf,
        inspection_keys=keys,
        include_r3_owned_property_names=False,
        include_xform_property_names=False,
        resolver_context=Ar.ResolverContext(),
    )

    assert inspected == ["@same.usda@"]
    assert len(keys) == 1


def test_contributing_relationship_scan_skips_absent_property() -> None:
    contract = schemas_module._R3RawAuthorshipContract(
        schema_order=(),
        attribute_specs={},
        attribute_defaults={},
        relationship_targets={"physxMimicJoint:rotZ:referenceJoint": "/Ref"},
    )
    prim_spec = SimpleNamespace(properties=())
    prim = SimpleNamespace(GetPrimStack=lambda: (prim_spec,))
    schemas_module._validate_contributing_r3_relationship_specs(
        prim,
        "/Joint",
        contract,
        Sdf=Sdf,
    )


def test_contributing_schema_scan_rejects_non_list_op_metadata() -> None:
    prim_spec = SimpleNamespace(
        ListInfoKeys=lambda: ("apiSchemas",),
        GetInfo=lambda _name: ["PhysicsRigidBodyAPI"],
    )
    prim = SimpleNamespace(GetPrimStack=lambda: (prim_spec,))
    contract = schemas_module._R3RawAuthorshipContract(
        schema_order=("PhysicsRigidBodyAPI",),
        attribute_specs={},
        attribute_defaults={},
        relationship_targets={},
    )

    with pytest.raises(JointRiggerContractError) as caught:
        schemas_module._validate_contributing_r3_schema_tokens(
            prim,
            "/World/base",
            contract,
            Sdf=Sdf,
        )

    assert caught.value.code == "authored_graph_mismatch"
    assert "noncanonical" in caught.value.detail


@pytest.mark.parametrize(
    "failure",
    [
        "composed_schema",
        "unmapped_scene",
        "missing_prim_spec",
        "missing_api_schemas",
        "raw_properties",
        "relationship_metadata",
        "unmapped_relationship_target",
    ],
)
def test_owned_raw_authorship_low_level_guards(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    scene_path = "/Scene"
    target_path = "/Target"
    schema_order = ("PhysicsRigidBodyAPI",)
    attribute_specs: dict[str, tuple[str, str]] = {}
    attribute_defaults: dict[str, Any] = {}
    relationship_targets: dict[str, str] = {}
    if failure == "raw_properties":
        attribute_specs["physics:mass"] = ("float", "varying")
        attribute_defaults["physics:mass"] = 1.0
    if failure in {"relationship_metadata", "unmapped_relationship_target"}:
        schema_order = ("PhysxMimicJointAPI:rotZ",)
        relationship_targets["physxMimicJoint:rotZ:referenceJoint"] = target_path
    contract = schemas_module._R3RawAuthorshipContract(
        schema_order=schema_order,
        attribute_specs=attribute_specs,
        attribute_defaults=attribute_defaults,
        relationship_targets=relationship_targets,
    )
    layer = Sdf.Layer.CreateAnonymous(f"raw-guard-{failure}.usda")
    prim_spec = None
    if failure != "missing_prim_spec":
        prim_spec = Sdf.CreatePrimInLayer(layer, scene_path)
    if prim_spec is not None and failure != "missing_api_schemas":
        schemas = Sdf.TokenListOp()
        schemas.prependedItems = list(schema_order)
        prim_spec.SetInfo("apiSchemas", schemas)
    if prim_spec is not None and relationship_targets:
        relationship = Sdf.RelationshipSpec(
            prim_spec,
            "physxMimicJoint:rotZ:referenceJoint",
            failure == "relationship_metadata",
        )
        relationship.targetPathList.explicitItems = [Sdf.Path(target_path)]

    def map_to_spec(path: Any) -> Any:
        if failure == "unmapped_scene":
            return Sdf.Path.emptyPath
        if failure == "unmapped_relationship_target" and str(path) == target_path:
            return Sdf.Path.emptyPath
        return Sdf.Path(str(path))

    edit_target = SimpleNamespace(
        GetLayer=lambda: layer,
        MapToSpecPath=map_to_spec,
    )
    prim = SimpleNamespace()
    stage = SimpleNamespace(
        GetEditTarget=lambda: edit_target,
        GetPrimAtPath=lambda _path: prim,
    )
    monkeypatch.setattr(
        schemas_module,
        "_r3_raw_authorship_contract",
        lambda *_args: {scene_path: contract},
    )
    monkeypatch.setattr(
        schemas_module,
        "_complete_plan_owned_value_clip_attributes",
        lambda *_args: ({}, frozenset()),
    )
    monkeypatch.setattr(
        schemas_module,
        "_validate_plan_owned_value_clips",
        lambda *_args, **_kwargs: None,
    )
    for name in (
        "_validate_contributing_r3_schema_tokens",
        "_validate_contributing_r3_property_names",
        "_validate_contributing_r3_attribute_specs",
        "_validate_contributing_r3_relationship_specs",
    ):
        monkeypatch.setattr(
            schemas_module,
            name,
            lambda *_args, **_kwargs: None,
        )
    observed_schemas = set(schema_order)
    if failure == "composed_schema":
        observed_schemas.clear()
    monkeypatch.setattr(
        schemas_module,
        "_applied_schema_tokens",
        lambda _prim: observed_schemas,
    )

    with pytest.raises(JointRiggerContractError) as caught:
        schemas_module._validate_owned_physics_raw_authorship(
            stage,
            object(),
            object(),
        )

    assert caught.value.code == "authored_graph_mismatch"


@pytest.mark.parametrize(
    ("failure", "reason_code"),
    [
        ("kilograms", "invalid_stage_units"),
        ("instanceable_body", "rigid_body_uneditable"),
        ("non_xformable_body", "rigid_body_not_xformable"),
        ("invalid_joint_path_syntax", "joint_path_required"),
        ("joint_type", "joint_type_mismatch"),
    ],
)
def test_schema_preflight_additional_stage_guards(
    failure: str,
    reason_code: str,
) -> None:
    stage, plan = _revolute_fixture()
    if failure == "kilograms":
        UsdPhysics.SetStageKilogramsPerUnit(stage, 0.0)
    elif failure == "instanceable_body":
        stage.GetPrimAtPath("/World/base").SetInstanceable(True)
    elif failure == "non_xformable_body":
        stage.GetPrimAtPath("/World/base").SetTypeName("Scope")
    elif failure == "invalid_joint_path_syntax":
        topology = plan.joints[0].topology.model_copy(
            update={"joint_id": "/World/<bad>"}
        )
        plan = _replace_first_joint(plan, topology=topology)
    else:
        stage.GetPrimAtPath("/World/Joints/hinge").SetTypeName("PhysicsPrismaticJoint")

    with pytest.raises(JointRiggerContractError) as caught:
        schemas_module._preflight(stage, plan)

    assert caught.value.code == reason_code


def _graph_plan(*edges: tuple[str, str]) -> JointRiggerPlanV1:
    return JointRiggerPlanV1.model_construct(
        schema_version=PLAN_SCHEMA_VERSION,
        joints=tuple(
            SimpleNamespace(topology=SimpleNamespace(body0=body0, body1=body1))
            for body0, body1 in edges
        ),
    )


@pytest.mark.parametrize(
    ("body_paths", "edges", "reason_code"),
    [
        (
            {"/A", "/B", "/C"},
            (("/A", "/C"), ("/B", "/C")),
            "ambiguous_joint_graph",
        ),
        (
            {"/A", "/B"},
            (("/A", "/B"), ("/B", "/A")),
            "ambiguous_joint_graph",
        ),
        (
            {"/A", "/B", "/C"},
            (("/B", "/C"), ("/C", "/B")),
            "disconnected_joint_graph",
        ),
    ],
)
def test_graph_root_rejects_ambiguous_and_disconnected_graphs(
    body_paths: set[str],
    edges: tuple[tuple[str, str], ...],
    reason_code: str,
) -> None:
    with pytest.raises(JointRiggerContractError) as caught:
        schemas_module._graph_roots(_graph_plan(*edges), body_paths)
    assert caught.value.code == reason_code


def test_plan_evidence_accepts_graph_deeper_than_python_recursion_limit() -> None:
    edge_count = sys.getrecursionlimit() + 50
    body_paths = tuple(f"/World/body_{index:05d}" for index in range(edge_count + 1))
    joints = tuple(
        JointPlanV1(
            topology=_topology(
                f"/World/Joints/joint_{index:05d}",
                "spherical",
                body_paths[index],
                body_paths[index + 1],
            )
        )
        for index in range(edge_count)
    )
    bodies = tuple(
        _body(
            path,
            ColliderPlanV1(
                prim_path=f"{path}/collision",
                provenance=_PROVENANCE,
            ),
        )
        for path in body_paths
    )
    plan = JointRiggerPlanV1(
        schema_version=PLAN_SCHEMA_VERSION,
        joints=joints,
        rigid_bodies=bodies,
        articulation_root=ArticulationRootPlanV1(
            prim_path=body_paths[0],
            provenance=_PROVENANCE,
        ),
    )

    validate_physics_plan_evidence(plan)


@pytest.mark.parametrize(
    ("failure", "reason_code"),
    [
        ("mass", "mass_evidence_missing"),
        ("colliders", "collider_evidence_missing"),
        ("center_of_mass", "mass_schema_conflict"),
        ("principal_axes", "mass_schema_conflict"),
        ("ownership", "collider_ownership_ambiguous"),
    ],
)
def test_body_preflight_additional_evidence_guards(
    failure: str,
    reason_code: str,
) -> None:
    stage, plan = _revolute_fixture()
    body = plan.rigid_bodies[0]
    if failure == "mass":
        body = body.model_copy(update={"mass": None})
    elif failure == "colliders":
        body = body.model_copy(update={"colliders": ()})
    elif failure == "center_of_mass":
        stage.GetPrimAtPath(body.prim_path).CreateAttribute(
            "physics:centerOfMass",
            Sdf.ValueTypeNames.Point3f,
            custom=False,
        ).Set(Gf.Vec3f(0.0, 0.0, 0.0))
    elif failure == "principal_axes":
        stage.GetPrimAtPath(body.prim_path).CreateAttribute(
            "physics:principalAxes",
            Sdf.ValueTypeNames.Quatf,
            custom=False,
        ).Set(Gf.Quatf(1.0))
    elif failure == "ownership":
        body = RigidBodyPlanV1.model_construct(
            prim_path=body.prim_path,
            mass=body.mass,
            colliders=(
                ColliderPlanV1(
                    prim_path="/World/link/collision",
                    provenance=_PROVENANCE,
                ),
            ),
            provenance=body.provenance,
        )
    with pytest.raises(JointRiggerContractError) as caught:
        schemas_module._preflight_body(
            stage,
            body,
            body_paths={"/World/base", "/World/link"},
            meters_per_unit=0.01,
            kilograms_per_unit=2.0,
            UsdGeom=UsdGeom,
        )

    assert caught.value.code == reason_code


def test_bare_mesh_collision_plan_rejects_existing_approximation() -> None:
    stage, plan = _revolute_fixture(
        mesh=True,
        mesh_collision_api=True,
        mesh_approximation=None,
    )
    stage.GetPrimAtPath("/World/base/collision").CreateAttribute(
        "physics:approximation",
        Sdf.ValueTypeNames.Token,
        custom=False,
    ).Set("convexHull")

    with pytest.raises(JointRiggerContractError) as caught:
        schemas_module._preflight(stage, plan)

    assert caught.value.code == "collider_schema_conflict"
    assert "bare MeshCollisionAPI" in caught.value.detail


@pytest.mark.parametrize(
    ("failure", "reason_code"),
    [
        ("missing_axis", "axis_unresolved"),
        ("unsupported_axis", "axis_unresolved"),
        ("empty_local_rotation", "axis_unresolved"),
        ("contradictory_frames", "contradictory_joint_frames"),
    ],
)
def test_axis_preflight_additional_guards(
    failure: str,
    reason_code: str,
) -> None:
    stage, plan = _revolute_fixture()
    prim = stage.GetPrimAtPath("/World/Joints/hinge")
    joint_schema = UsdPhysics.RevoluteJoint(prim)
    if failure == "missing_axis":
        prim.GetAttribute("physics:axis").Clear()
    elif failure == "unsupported_axis":
        prim.GetAttribute("physics:axis").Set("Q")
    elif failure == "empty_local_rotation":
        joint_schema.CreateLocalRot0Attr().Set(Sdf.ValueBlock())
    else:
        joint_schema.CreateLocalRot1Attr().Set(Gf.Quatf(0.0, Gf.Vec3f(1.0, 0.0, 0.0)))

    with pytest.raises(JointRiggerContractError) as caught:
        schemas_module._preflight_axis(
            stage,
            prim,
            plan.joints[0],
            xform_cache=UsdGeom.XformCache(),
        )

    assert caught.value.code == reason_code


@pytest.mark.parametrize(
    ("failure", "reason_code"),
    [
        ("one_anchor", "joint_anchor_incomplete"),
        ("planned_but_absent", "joint_anchor_mismatch"),
        ("empty_anchor_value", "joint_anchor_incomplete"),
    ],
)
def test_anchor_preflight_additional_guards(
    failure: str,
    reason_code: str,
) -> None:
    stage, plan = _revolute_fixture()
    prim = stage.GetPrimAtPath("/World/Joints/hinge")
    joint_schema = UsdPhysics.RevoluteJoint(prim)
    joint = plan.joints[0]
    if failure == "one_anchor":
        joint_schema.CreateLocalPos0Attr(Gf.Vec3f(0.0))
    elif failure == "planned_but_absent":
        joint = joint.model_copy(
            update={
                "anchor": JointAnchorV1(
                    position_stage=(0.0, 0.0, 0.0),
                    provenance=_PROVENANCE,
                )
            }
        )
    else:
        joint_schema.CreateLocalPos0Attr().Set(Sdf.ValueBlock())
        joint_schema.CreateLocalPos1Attr().Set(Sdf.ValueBlock())

    with pytest.raises(JointRiggerContractError) as caught:
        schemas_module._preflight_anchor(
            stage,
            prim,
            joint,
            xform_cache=UsdGeom.XformCache(),
        )

    assert caught.value.code == reason_code


@pytest.mark.parametrize(
    ("local_offset", "accepted"),
    [
        (float(Gf.Vec3f(1e-6)[0]), True),
        (float(Gf.Vec3f(1.0000001e-6)[0]), False),
    ],
    ids=["last-float-inside", "first-float-outside"],
)
def test_anchor_preflight_keeps_strict_distance_boundary(
    local_offset: float,
    accepted: bool,
) -> None:
    stage, plan = _revolute_fixture()
    prim = stage.GetPrimAtPath("/World/Joints/hinge")
    joint_schema = UsdPhysics.RevoluteJoint(prim)
    joint_schema.CreateLocalPos0Attr().Set(Gf.Vec3f(1.0, 0.0, 0.0))
    joint_schema.CreateLocalPos1Attr().Set(Gf.Vec3f(local_offset, 0.0, 0.0))
    assert (
        local_offset <= validation_module._SHARED_ANCHOR_DISTANCE_TOLERANCE
    ) is accepted

    if accepted:
        schemas_module._preflight_anchor(
            stage,
            prim,
            plan.joints[0],
            xform_cache=UsdGeom.XformCache(),
        )
    else:
        with pytest.raises(JointRiggerContractError) as caught:
            schemas_module._preflight_anchor(
                stage,
                prim,
                plan.joints[0],
                xform_cache=UsdGeom.XformCache(),
            )
        assert caught.value.code == "contradictory_joint_frames"


@pytest.mark.parametrize(
    "failure",
    [
        "omitted_limits",
        "unplanned_lower",
        "one_sided_absent",
        "planned_absent",
        "planned_mismatch",
    ],
)
def test_limit_preflight_additional_guards_and_one_sided_success(
    failure: str,
) -> None:
    stage, plan = _revolute_fixture()
    prim = stage.GetPrimAtPath("/World/Joints/hinge")
    joint = plan.joints[0]
    if failure == "omitted_limits":
        joint = joint.model_copy(update={"limit": None})
    elif failure in {"unplanned_lower", "one_sided_absent"}:
        limit = JointLimitV1(
            lower=None,
            upper=45.0,
            unit="degrees",
            provenance=_PROVENANCE,
        )
        joint = joint.model_copy(update={"limit": limit})
        if failure == "one_sided_absent":
            prim.GetAttribute("physics:lowerLimit").Clear()
    elif failure == "planned_absent":
        prim.GetAttribute("physics:lowerLimit").Clear()
    else:
        prim.GetAttribute("physics:lowerLimit").Set(-44.0)

    if failure == "one_sided_absent":
        schemas_module._preflight_limits(prim, joint, meters_per_unit=0.01)
        return

    with pytest.raises(JointRiggerContractError) as caught:
        schemas_module._preflight_limits(prim, joint, meters_per_unit=0.01)

    assert caught.value.code == "limit_evidence_mismatch"


@pytest.mark.parametrize(
    ("failure", "reason_code"),
    [
        ("spherical_state", "joint_state_not_applicable"),
        ("missing_state", "joint_state_evidence_missing"),
        ("extra_state_schema", "joint_state_schema_conflict"),
    ],
)
def test_joint_state_preflight_additional_guards(
    failure: str,
    reason_code: str,
) -> None:
    stage, plan = _revolute_fixture()
    prim = stage.GetPrimAtPath("/World/Joints/hinge")
    joint = plan.joints[0]
    motion: str | None = "angular"
    if failure == "spherical_state":
        topology = joint.topology.model_copy(update={"joint_type": "spherical"})
        joint = joint.model_copy(update={"topology": topology})
        motion = None
    elif failure == "missing_state":
        joint = joint.model_copy(update={"state": None})
    else:
        prim.AddAppliedSchema("PhysicsJointStateAPI:linear")
    context = schemas_module._JointContext(
        plan=joint,
        prim=prim,
        motion=motion,
        axis_token="z" if motion is not None else None,
    )

    with pytest.raises(JointRiggerContractError) as caught:
        schemas_module._preflight_joint_state(context)

    assert caught.value.code == reason_code


@pytest.mark.parametrize(
    ("failure", "reason_code"),
    [
        ("spherical_drive", "joint_control_not_applicable"),
        ("spherical_friction", "joint_friction_not_applicable"),
        ("extra_drive_schema", "drive_schema_conflict"),
        ("negative_drive", "invalid_drive_values"),
        ("negative_velocity", "invalid_drive_values"),
        ("negative_friction", "invalid_joint_friction"),
        ("extra_drive_property", "drive_schema_conflict"),
        ("unplanned_max_velocity", "drive_schema_conflict"),
        ("extra_physx_schema", "drive_schema_conflict"),
        ("extra_physx_property", "drive_schema_conflict"),
    ],
)
def test_joint_drive_preflight_additional_guards(
    failure: str,
    reason_code: str,
) -> None:
    stage, plan = _revolute_fixture(with_drive=True)
    prim = stage.GetPrimAtPath("/World/Joints/hinge")
    joint = plan.joints[0]
    assert joint.drive is not None
    motion: str | None = "angular"
    if failure == "spherical_drive":
        topology = joint.topology.model_copy(update={"joint_type": "spherical"})
        joint = joint.model_copy(update={"topology": topology})
        motion = None
    elif failure == "spherical_friction":
        topology = joint.topology.model_copy(update={"joint_type": "spherical"})
        joint = joint.model_copy(
            update={
                "topology": topology,
                "drive": None,
                "joint_friction": JointFrictionV1(
                    coefficient=0.15,
                    provenance=_PROVENANCE,
                ),
            }
        )
        motion = None
    elif failure == "extra_drive_schema":
        prim.AddAppliedSchema("PhysicsDriveAPI:linear")
    elif failure == "negative_drive":
        joint = joint.model_copy(
            update={"drive": joint.drive.model_copy(update={"stiffness": -1.0})}
        )
    elif failure == "negative_velocity":
        joint = joint.model_copy(
            update={
                "drive": joint.drive.model_copy(update={"max_joint_velocity": -1.0})
            }
        )
    elif failure == "negative_friction":
        joint = joint.model_copy(
            update={
                "joint_friction": JointFrictionV1.model_construct(
                    coefficient=-0.1,
                    provenance=_PROVENANCE,
                )
            }
        )
    elif failure == "extra_drive_property":
        prim.CreateAttribute(
            "drive:angular:physics:rogue",
            Sdf.ValueTypeNames.Float,
            custom=False,
        )
    elif failure == "unplanned_max_velocity":
        joint = joint.model_copy(
            update={
                "drive": joint.drive.model_copy(update={"max_joint_velocity": None})
            }
        )
        prim.AddAppliedSchema("PhysxJointAPI")
    elif failure == "extra_physx_schema":
        prim.AddAppliedSchema("PhysxJointAPI:rogue")
    else:
        prim.CreateAttribute(
            "physxJoint:rogue",
            Sdf.ValueTypeNames.Float,
            custom=False,
        )
    context = schemas_module._JointContext(
        plan=joint,
        prim=prim,
        motion=motion,
        axis_token="z" if motion is not None else None,
    )

    with pytest.raises(JointRiggerContractError) as caught:
        schemas_module._preflight_joint_control(
            context,
            {joint.topology.joint_id: context},
        )

    assert caught.value.code == reason_code


@pytest.mark.parametrize(
    ("failure", "reason_code"),
    [
        ("nonrevolute_mimic", "mimic_not_applicable"),
        ("nonrevolute_reference", "mimic_not_applicable"),
        ("mimic_chain", "mimic_chain_unsupported"),
        ("drive_schema", "mimic_schema_conflict"),
        ("extra_mimic_schema", "mimic_schema_conflict"),
        ("extra_mimic_property", "mimic_schema_conflict"),
        ("conflicting_reference", "mimic_schema_conflict"),
    ],
)
def test_joint_mimic_preflight_additional_guards(
    failure: str,
    reason_code: str,
) -> None:
    stage, plan = _mimic_fixture()
    preflight = schemas_module._preflight(stage, plan)
    contexts = dict(preflight.joints)
    target_path = "/World/Joints/second"
    reference_path = "/World/Joints/first"
    target = contexts[target_path]
    reference = contexts[reference_path]
    assert target.plan.mimic is not None
    if failure == "nonrevolute_mimic":
        topology = target.plan.topology.model_copy(update={"joint_type": "prismatic"})
        target = schemas_module._JointContext(
            target.plan.model_copy(update={"topology": topology}),
            target.prim,
            "linear",
            target.axis_token,
        )
    elif failure == "nonrevolute_reference":
        topology = reference.plan.topology.model_copy(
            update={"joint_type": "prismatic"}
        )
        reference = schemas_module._JointContext(
            reference.plan.model_copy(update={"topology": topology}),
            reference.prim,
            "linear",
            reference.axis_token,
        )
    elif failure == "mimic_chain":
        reference = schemas_module._JointContext(
            reference.plan.model_copy(update={"mimic": target.plan.mimic}),
            reference.prim,
            reference.motion,
            reference.axis_token,
        )
    elif failure == "drive_schema":
        target.prim.AddAppliedSchema("PhysicsDriveAPI:angular")
    elif failure == "extra_mimic_schema":
        target.prim.AddAppliedSchema("PhysxMimicJointAPI:rotX")
    elif failure == "extra_mimic_property":
        target.prim.CreateAttribute(
            "physxMimicJoint:rotZ:rogue",
            Sdf.ValueTypeNames.Float,
            custom=False,
        )
    else:
        target.prim.CreateRelationship(
            "physxMimicJoint:rotZ:referenceJoint",
            custom=False,
        ).SetTargets(["/World/Joints/wrong"])
    contexts[target_path] = target
    contexts[reference_path] = reference

    with pytest.raises(JointRiggerContractError) as caught:
        schemas_module._preflight_joint_control(target, contexts)

    assert caught.value.code == reason_code


def test_postwrite_validation_rejects_changed_mimic_reference() -> None:
    stage, plan = _mimic_fixture()
    author_physics_schemas(stage, plan)
    preflight = schemas_module._preflight(stage, plan)
    stage.GetPrimAtPath("/World/Joints/second").GetRelationship(
        "physxMimicJoint:rotZ:referenceJoint"
    ).SetTargets(["/World/Joints/wrong"])

    with pytest.raises(JointRiggerContractError) as caught:
        schemas_module._validate_authored(stage, plan, preflight)

    assert caught.value.code == "postwrite_validation_failed"
    assert "mimic reference mismatch" in caught.value.detail


@pytest.mark.parametrize(
    ("failure", "reason_code"),
    [
        ("invalid_divisor", "invalid_stage_units"),
        ("invalid_mass", "mass_unit_conversion_invalid"),
    ],
)
def test_mass_stage_value_conversion_guards(
    failure: str,
    reason_code: str,
) -> None:
    mass = SimpleNamespace(
        mass_kg=4.0,
        diagonal_inertia_kg_m2=(0.02, 0.03, 0.04),
    )
    meters_per_unit = 0.0 if failure == "invalid_divisor" else 1.0
    if failure == "invalid_mass":
        mass.mass_kg = math.inf

    with pytest.raises(JointRiggerContractError) as caught:
        schemas_module._mass_stage_values(
            mass,
            meters_per_unit=meters_per_unit,
            kilograms_per_unit=1.0,
        )

    assert caught.value.code == reason_code


@pytest.mark.parametrize(
    ("failure", "reason_code"),
    [
        ("incomplete", "mimic_limits_incomplete"),
        ("not_zero_spanning", "mimic_limits_incompatible"),
    ],
)
def test_mimic_limit_guards(failure: str, reason_code: str) -> None:
    stage, plan = _revolute_fixture()
    joint = plan.joints[0]
    if failure == "incomplete":
        joint = joint.model_copy(update={"limit": None})
    else:
        joint = joint.model_copy(
            update={
                "limit": JointLimitV1(
                    lower=1.0,
                    upper=2.0,
                    unit="degrees",
                    provenance=_PROVENANCE,
                )
            }
        )
    context = schemas_module._JointContext(
        joint,
        stage.GetPrimAtPath("/World/Joints/hinge"),
        "angular",
        "z",
    )

    with pytest.raises(JointRiggerContractError) as caught:
        schemas_module._require_complete_zero_spanning_limits(context)

    assert caught.value.code == reason_code


@pytest.mark.parametrize("position", [-46.0, 46.0])
def test_position_limit_guard_rejects_both_sides(position: float) -> None:
    stage, _ = _revolute_fixture()
    prim = stage.GetPrimAtPath("/World/Joints/hinge")
    with pytest.raises(JointRiggerContractError) as caught:
        schemas_module._require_position_inside_authored_limits(
            prim,
            position,
            owner="/World/Joints/hinge",
            code="outside",
        )
    assert caught.value.code == "outside"


def test_target_prim_guard_rejects_missing_prim() -> None:
    stage, _ = _revolute_fixture()
    with pytest.raises(JointRiggerContractError) as caught:
        schemas_module._require_target_prim(stage, "/Missing", kind="test target")
    assert caught.value.code == "test_target_unresolved"


def test_static_attribute_guard_rejects_a_composed_spline() -> None:
    attribute = SimpleNamespace(
        GetName=lambda: "physics:mass",
        HasAuthoredConnections=lambda: False,
        HasSpline=lambda: True,
    )

    with pytest.raises(JointRiggerContractError) as caught:
        schemas_module._require_static(attribute, owner="/World/base")

    assert caught.value.code == "time_sampled_owned_property"
    assert caught.value.detail == ("/World/base has spline-authored physics:mass")


@pytest.mark.parametrize(
    "guard",
    [
        "application",
        "required_schema",
        "exact_attribute",
        "absent_attribute",
        "absent_schema",
    ],
)
def test_postwrite_low_level_guards(guard: str) -> None:
    stage, _ = _revolute_fixture()
    prim = stage.GetPrimAtPath("/World/base/collision")
    with pytest.raises(JointRiggerContractError) as caught:
        if guard == "application":
            schemas_module._require_application(False, "TestAPI", prim)
        elif guard == "required_schema":
            schemas_module._require_schema(prim, "PhysicsRigidBodyAPI")
        elif guard == "exact_attribute":
            schemas_module._require_exact_attr(prim, "physics:missing", True)
        elif guard == "absent_attribute":
            prim.CreateAttribute(
                "physics:approximation",
                Sdf.ValueTypeNames.Token,
                custom=False,
            ).Set("none")
            schemas_module._require_absent_authored_attr(
                prim,
                "physics:approximation",
            )
        else:
            prim.AddAppliedSchema("PhysicsMeshCollisionAPI")
            schemas_module._require_schema_absent(
                prim,
                "PhysicsMeshCollisionAPI",
            )
    assert caught.value.code in {
        "physics_schema_apply_failed",
        "postwrite_validation_failed",
    }


def test_applied_schema_tokens_accepts_legacy_list_metadata() -> None:
    prim = SimpleNamespace(
        GetAppliedSchemas=lambda: (),
        GetMetadata=lambda _name: ["PhysicsRigidBodyAPI"],
    )
    assert schemas_module._applied_schema_tokens(prim) == {"PhysicsRigidBodyAPI"}


@pytest.mark.parametrize(
    ("value", "reason_code"),
    [
        ((math.nan, 0.0, 1.0), "axis_not_finite"),
        ((0.0, 0.0, 0.0), "axis_unresolved"),
    ],
)
def test_normalized_vector_guards(
    value: tuple[float, float, float],
    reason_code: str,
) -> None:
    with pytest.raises(JointRiggerContractError) as caught:
        schemas_module._normalized_vector(value, owner="test")
    assert caught.value.code == reason_code


def test_stored_plain_value_comparison_rejects_tuple_scalar_mismatch() -> None:
    assert not schemas_module._stored_plain_values_equal((1.0,), 1.0)


def test_plain_value_falls_back_to_string_for_noniterable_objects() -> None:
    class NonIterable:
        def __str__(self) -> str:
            return "opaque"

    assert schemas_module._plain_value(NonIterable()) == "opaque"


def _revolute_fixture(
    *,
    with_drive: bool = False,
    joint_friction: float | None = None,
    mesh: bool = False,
    mesh_approximation: Literal["convexHull", "convexDecomposition", "sdf"]
    | None = "convexHull",
    mesh_collision_api: Literal[True] | None = None,
) -> tuple[Any, JointRiggerPlanV1]:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 2.0)
    UsdGeom.Xform.Define(stage, "/World")
    bodies = ("/World/base", "/World/link")
    colliders = []
    for index, path in enumerate(bodies):
        body = UsdGeom.Xform.Define(stage, path)
        body.AddTranslateOp().Set(Gf.Vec3d(float(index), 0.0, 0.0))
        collider_path = f"{path}/collision"
        if mesh:
            UsdGeom.Mesh.Define(stage, collider_path)
        else:
            UsdGeom.Cube.Define(stage, collider_path)
        colliders.append(
            ColliderPlanV1(
                prim_path=collider_path,
                mesh_collision_api=(mesh_collision_api if mesh else None),
                mesh_approximation=(mesh_approximation if mesh else None),
                provenance=_PROVENANCE,
            )
        )
    UsdGeom.Scope.Define(stage, "/World/Joints")
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/World/Joints/hinge")
    joint.CreateBody0Rel().SetTargets([bodies[0]])
    joint.CreateBody1Rel().SetTargets([bodies[1]])
    joint.CreateAxisAttr("Z")
    joint.CreateLowerLimitAttr(-45.0)
    joint.CreateUpperLimitAttr(45.0)
    drive = (
        JointDriveV1(
            drive_type="force",
            stiffness=20.0,
            damping=2.0,
            max_force=100.0,
            target_position=0.0,
            target_velocity=0.0,
            max_joint_velocity=3.0,
            provenance=_PROVENANCE,
        )
        if with_drive
        else None
    )
    plan = JointRiggerPlanV1(
        schema_version=PLAN_SCHEMA_VERSION,
        joints=(
            JointPlanV1(
                topology=_topology(
                    "/World/Joints/hinge",
                    "revolute",
                    bodies[0],
                    bodies[1],
                    axis=(0.0, 0.0, 1.0),
                ),
                limit=_limit(),
                state=_zero_state(),
                joint_friction=(
                    JointFrictionV1(
                        coefficient=joint_friction,
                        provenance=_PROVENANCE,
                    )
                    if joint_friction is not None
                    else None
                ),
                drive=drive,
            ),
        ),
        rigid_bodies=tuple(
            _body(path, collider)
            for path, collider in zip(bodies, colliders, strict=True)
        ),
        articulation_root=ArticulationRootPlanV1(
            prim_path=bodies[0],
            provenance=_PROVENANCE,
        ),
    )
    return stage, plan


def _prismatic_fixture(
    *,
    with_drive: bool = True,
    joint_friction: float | None = None,
) -> tuple[Any, JointRiggerPlanV1]:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 2.0)
    UsdGeom.Xform.Define(stage, "/World")
    bodies = ("/World/base", "/World/link")
    colliders = []
    for index, path in enumerate(bodies):
        body = UsdGeom.Xform.Define(stage, path)
        body.AddTranslateOp().Set(Gf.Vec3d(float(index), 0.0, 0.0))
        collider_path = f"{path}/collision"
        UsdGeom.Cube.Define(stage, collider_path)
        colliders.append(
            ColliderPlanV1(
                prim_path=collider_path,
                provenance=_PROVENANCE,
            )
        )
    UsdGeom.Scope.Define(stage, "/World/Joints")
    joint = UsdPhysics.PrismaticJoint.Define(stage, "/World/Joints/slider")
    joint.CreateBody0Rel().SetTargets([bodies[0]])
    joint.CreateBody1Rel().SetTargets([bodies[1]])
    joint.CreateAxisAttr("X")
    joint.CreateLowerLimitAttr(-25.0)
    joint.CreateUpperLimitAttr(50.0)
    plan = JointRiggerPlanV1(
        schema_version=PLAN_SCHEMA_VERSION,
        joints=(
            JointPlanV1(
                topology=_topology(
                    "/World/Joints/slider",
                    "prismatic",
                    bodies[0],
                    bodies[1],
                    axis=(1.0, 0.0, 0.0),
                ),
                limit=JointLimitV1(
                    lower=-0.25,
                    upper=0.5,
                    unit="meters",
                    provenance=_PROVENANCE,
                ),
                state=_zero_state(),
                joint_friction=(
                    JointFrictionV1(
                        coefficient=joint_friction,
                        provenance=_PROVENANCE,
                    )
                    if joint_friction is not None
                    else None
                ),
                drive=(
                    JointDriveV1(
                        drive_type="force",
                        stiffness=20.0,
                        damping=2.0,
                        max_force=100.0,
                        target_position=0.0,
                        target_velocity=0.0,
                        max_joint_velocity=3.0,
                        provenance=_PROVENANCE,
                    )
                    if with_drive
                    else None
                ),
            ),
        ),
        rigid_bodies=tuple(
            _body(path, collider)
            for path, collider in zip(bodies, colliders, strict=True)
        ),
        articulation_root=ArticulationRootPlanV1(
            prim_path=bodies[0],
            provenance=_PROVENANCE,
        ),
    )
    return stage, plan


def _replace_colliders_with_xform_instance_roots(
    stage: Any,
    plan: JointRiggerPlanV1,
    *,
    mesh_approximation: Literal[
        "none",
        "convexHull",
        "convexDecomposition",
        "sdf",
    ]
    | None,
    mesh_collision_api: Literal[True] | None = None,
    nested_depth: int = 0,
) -> JointRiggerPlanV1:
    prototype_root = "/World/ColliderPrototypes"
    UsdGeom.Scope.Define(stage, prototype_root)
    bodies = []
    for index, body in enumerate(plan.rigid_bodies):
        collider_path = f"{body.prim_path}/collision"
        assert stage.RemovePrim(collider_path)
        prototype_path = f"{prototype_root}/prototype_{index}"
        UsdGeom.Xform.Define(stage, prototype_path)
        prototype_leaf_path = prototype_path
        for depth in range(nested_depth):
            prototype_leaf_path = f"{prototype_leaf_path}/nested_{depth}"
            UsdGeom.Xform.Define(stage, prototype_leaf_path)
        UsdGeom.Cube.Define(stage, f"{prototype_leaf_path}/shape")
        collider_prim = UsdGeom.Xform.Define(stage, collider_path).GetPrim()
        collider_prim.GetReferences().AddInternalReference(prototype_path)
        collider_prim.SetInstanceable(True)
        assert collider_prim.IsInstance()
        collider = ColliderPlanV1(
            prim_path=collider_path,
            mesh_collision_api=mesh_collision_api,
            mesh_approximation=mesh_approximation,
            provenance=_PROVENANCE,
        )
        bodies.append(body.model_copy(update={"colliders": (collider,)}))
    return plan.model_copy(update={"rigid_bodies": tuple(bodies)})


def _mimic_fixture(
    *,
    second_axis: str = "Z",
) -> tuple[Any, JointRiggerPlanV1]:
    stage = Usd.Stage.CreateInMemory()
    bodies = ("/World/base", "/World/arm", "/World/hand")
    UsdGeom.Xform.Define(stage, "/World")
    collider_plans = []
    for path in bodies:
        UsdGeom.Xform.Define(stage, path)
        collider_path = f"{path}/collision"
        UsdGeom.Mesh.Define(stage, collider_path)
        collider_plans.append(
            ColliderPlanV1(
                prim_path=collider_path,
                mesh_approximation="convexHull",
                provenance=_PROVENANCE,
            )
        )
    UsdGeom.Scope.Define(stage, "/World/Joints")
    joint_specs = (
        ("first", bodies[0], bodies[1], "Z"),
        ("second", bodies[1], bodies[2], second_axis),
    )
    joint_plans = []
    for name, body0, body1, axis in joint_specs:
        path = f"/World/Joints/{name}"
        joint = UsdPhysics.RevoluteJoint.Define(stage, path)
        joint.CreateBody0Rel().SetTargets([body0])
        joint.CreateBody1Rel().SetTargets([body1])
        joint.CreateAxisAttr(axis)
        joint.CreateLowerLimitAttr(-90.0)
        joint.CreateUpperLimitAttr(90.0)
        axis_vector = _axis_vector(axis)
        mimic = (
            JointMimicV1(
                reference_joint_id="/World/Joints/first",
                gearing=-1.0,
                offset=0.0,
                natural_frequency=4.0,
                damping_ratio=0.7,
                provenance=_PROVENANCE,
            )
            if name == "second"
            else None
        )
        joint_plans.append(
            JointPlanV1(
                topology=_topology(
                    path,
                    "revolute",
                    body0,
                    body1,
                    axis=axis_vector,
                ),
                limit=JointLimitV1(
                    lower=-90.0,
                    upper=90.0,
                    unit="degrees",
                    provenance=_PROVENANCE,
                ),
                state=_zero_state(),
                mimic=mimic,
            )
        )
    plan = JointRiggerPlanV1(
        schema_version=PLAN_SCHEMA_VERSION,
        joints=tuple(joint_plans),
        rigid_bodies=tuple(
            _body(path, collider)
            for path, collider in zip(bodies, collider_plans, strict=True)
        ),
        articulation_root=ArticulationRootPlanV1(
            prim_path=bodies[0],
            provenance=_PROVENANCE,
        ),
    )
    return stage, plan


def _spherical_fixture() -> tuple[Any, JointRiggerPlanV1]:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    bodies = ("/World/base", "/World/ball")
    collider_plans = []
    for path in bodies:
        UsdGeom.Xform.Define(stage, path)
        collider_path = f"{path}/collision"
        UsdGeom.Sphere.Define(stage, collider_path)
        collider_plans.append(
            ColliderPlanV1(
                prim_path=collider_path,
                provenance=_PROVENANCE,
            )
        )
    UsdGeom.Scope.Define(stage, "/World/Joints")
    joint = UsdPhysics.SphericalJoint.Define(stage, "/World/Joints/ball")
    joint.CreateBody0Rel().SetTargets([bodies[0]])
    joint.CreateBody1Rel().SetTargets([bodies[1]])
    plan = JointRiggerPlanV1(
        schema_version=PLAN_SCHEMA_VERSION,
        joints=(
            JointPlanV1(
                topology=_topology(
                    "/World/Joints/ball",
                    "spherical",
                    bodies[0],
                    bodies[1],
                )
            ),
        ),
        rigid_bodies=tuple(
            _body(path, collider)
            for path, collider in zip(bodies, collider_plans, strict=True)
        ),
        articulation_root=ArticulationRootPlanV1(
            prim_path=bodies[0],
            provenance=_PROVENANCE,
        ),
    )
    return stage, plan


def _nested_fixture(
    *,
    reset_nested_body: bool,
) -> tuple[Any, JointRiggerPlanV1]:
    stage = Usd.Stage.CreateInMemory()
    world = UsdGeom.Xform.Define(stage, "/World")
    world.AddTranslateOp().Set(Gf.Vec3d(1.0, -2.0, 3.0))
    base = UsdGeom.Xform.Define(stage, "/World/base")
    base.AddTranslateOp().Set(Gf.Vec3d(4.0, 5.0, -6.0))
    base.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, 35.0))
    nested = UsdGeom.Xform.Define(stage, "/World/base/link")
    nested.AddTranslateOp().Set(Gf.Vec3d(-7.0, 8.0, 9.0))
    nested.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, 5.0))
    nested.AddScaleOp().Set(Gf.Vec3f(0.75, 1.25, 1.5))
    nested.SetResetXformStack(reset_nested_body)
    collider_paths = (
        "/World/base/collision",
        "/World/base/link/collision",
    )
    for path in collider_paths:
        UsdGeom.Cube.Define(stage, path)
    UsdGeom.Scope.Define(stage, "/World/Joints")
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/World/Joints/hinge")
    joint.CreateBody0Rel().SetTargets(["/World/base"])
    joint.CreateBody1Rel().SetTargets(["/World/base/link"])
    joint.CreateAxisAttr("Z")
    joint.CreateLowerLimitAttr(-45.0)
    joint.CreateUpperLimitAttr(45.0)
    plan = JointRiggerPlanV1(
        schema_version=PLAN_SCHEMA_VERSION,
        joints=(
            JointPlanV1(
                topology=_topology(
                    "/World/Joints/hinge",
                    "revolute",
                    "/World/base",
                    "/World/base/link",
                    axis=(0.0, 0.0, 1.0),
                ),
                limit=_limit(),
                state=_zero_state(),
            ),
        ),
        rigid_bodies=(
            _body(
                "/World/base",
                ColliderPlanV1(
                    prim_path=collider_paths[0],
                    provenance=_PROVENANCE,
                ),
            ),
            _body(
                "/World/base/link",
                ColliderPlanV1(
                    prim_path=collider_paths[1],
                    provenance=_PROVENANCE,
                ),
            ),
        ),
        articulation_root=ArticulationRootPlanV1(
            prim_path="/World/base",
            provenance=_PROVENANCE,
        ),
    )
    return stage, plan


def _nested_reference_fixture() -> tuple[Any, JointRiggerPlanV1, Any]:
    source_stage = Usd.Stage.CreateInMemory()
    asset = UsdGeom.Xform.Define(source_stage, "/Asset")
    source_stage.SetDefaultPrim(asset.GetPrim())
    asset.AddTranslateOp().Set(Gf.Vec3d(2.0, 3.0, 4.0))
    body_paths = (
        "/World/base",
        "/World/base/wheel",
        "/World/base/wheel/tire",
        "/World/base/drawer",
    )
    source_paths = (
        "/Asset",
        "/Asset/wheel",
        "/Asset/wheel/tire",
        "/Asset/drawer",
    )
    for index, path in enumerate(source_paths[1:], start=1):
        body = UsdGeom.Xform.Define(source_stage, path)
        body.AddTranslateOp().Set(
            Gf.Vec3d(float(index), float(index * -2), float(index * 3))
        )
        body.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, float(index * -9)))
    collider_paths = tuple(f"{path}/collision" for path in source_paths)
    for index, path in enumerate(collider_paths, start=1):
        cube = UsdGeom.Cube.Define(source_stage, path)
        cube.GetSizeAttr().Set(float(index))

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    base = UsdGeom.Xform.Define(stage, "/World/base").GetPrim()
    assert base.GetReferences().AddReference(
        source_stage.GetRootLayer().identifier,
        "/Asset",
    )
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))

    UsdGeom.Scope.Define(stage, "/World/Joints")
    joint_specs = (
        ("wheel", body_paths[0], body_paths[1]),
        ("tire", body_paths[1], body_paths[2]),
        ("drawer", body_paths[0], body_paths[3]),
    )
    joints: list[JointPlanV1] = []
    for name, body0, body1 in joint_specs:
        joint_path = f"/World/Joints/{name}"
        joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
        joint.CreateBody0Rel().SetTargets([body0])
        joint.CreateBody1Rel().SetTargets([body1])
        joint.CreateAxisAttr("Z")
        joint.CreateLowerLimitAttr(-45.0)
        joint.CreateUpperLimitAttr(45.0)
        joints.append(
            JointPlanV1(
                topology=_topology(
                    joint_path,
                    "revolute",
                    body0,
                    body1,
                    axis=(0.0, 0.0, 1.0),
                ),
                limit=_limit(),
                state=_zero_state(),
            )
        )

    remapped_colliders = tuple(
        path.replace("/Asset", "/World/base", 1) for path in collider_paths
    )
    plan = JointRiggerPlanV1(
        schema_version=PLAN_SCHEMA_VERSION,
        joints=tuple(joints),
        rigid_bodies=tuple(
            _body(
                body_path,
                ColliderPlanV1(
                    prim_path=collider_path,
                    provenance=_PROVENANCE,
                ),
            )
            for body_path, collider_path in zip(
                body_paths,
                remapped_colliders,
                strict=True,
            )
        ),
        articulation_root=ArticulationRootPlanV1(
            prim_path=body_paths[0],
            provenance=_PROVENANCE,
        ),
    )
    return stage, plan, source_stage


def _body(path: str, collider: ColliderPlanV1) -> RigidBodyPlanV1:
    return RigidBodyPlanV1(
        prim_path=path,
        mass=MassPropertiesV1(
            mass_kg=4.0,
            diagonal_inertia_kg_m2=(0.02, 0.03, 0.04),
            provenance=_PROVENANCE,
        ),
        colliders=(collider,),
        provenance=_PROVENANCE,
    )


def _topology(
    joint_id: str,
    joint_type: str,
    body0: str,
    body1: str,
    *,
    axis: tuple[float, float, float] | None = None,
) -> JointTopologyV1:
    fields = ("joint_type", "body0", "body1")
    provenance = dict.fromkeys(fields, _PROVENANCE)
    if axis is not None:
        provenance["axis_stage"] = _PROVENANCE
    return JointTopologyV1(
        joint_id=joint_id,
        joint_type=joint_type,  # type: ignore[arg-type]
        body0=body0,
        body1=body1,
        axis_stage=axis,
        field_provenance=provenance,
    )


def _limit() -> JointLimitV1:
    return JointLimitV1(
        lower=-45.0,
        upper=45.0,
        unit="degrees",
        provenance=_PROVENANCE,
    )


def _zero_state() -> JointStateV1:
    return JointStateV1(position=0.0, velocity=0.0, provenance=_PROVENANCE)


def _replace_plan(
    plan: JointRiggerPlanV1,
    *,
    joints: tuple[JointPlanV1, ...] | None = None,
    rigid_bodies: tuple[RigidBodyPlanV1, ...] | None = None,
    articulation_root: ArticulationRootPlanV1 | None | object = ...,  # noqa: PYI051
) -> JointRiggerPlanV1:
    root = plan.articulation_root if articulation_root is ... else articulation_root
    assert root is None or isinstance(root, ArticulationRootPlanV1)
    return JointRiggerPlanV1(
        schema_version=plan.schema_version,
        joints=plan.joints if joints is None else joints,
        rigid_bodies=plan.rigid_bodies if rigid_bodies is None else rigid_bodies,
        articulation_root=root,
    )


def _replace_first_body(
    plan: JointRiggerPlanV1,
    **updates: Any,
) -> JointRiggerPlanV1:
    first = plan.rigid_bodies[0].model_copy(update=updates)
    return _replace_plan(plan, rigid_bodies=(first, *plan.rigid_bodies[1:]))


def _replace_first_joint(
    plan: JointRiggerPlanV1,
    **updates: Any,
) -> JointRiggerPlanV1:
    first = plan.joints[0].model_copy(update=updates)
    return _replace_plan(plan, joints=(first, *plan.joints[1:]))


def _apply_complete_drive(
    prim: Any,
    *,
    motion: str,
    target_position: float = 0.0,
    target_velocity: float = 0.0,
    max_joint_velocity: float | None = None,
) -> None:
    drive = UsdPhysics.DriveAPI.Apply(prim, motion)
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(20.0)
    drive.CreateDampingAttr(2.0)
    drive.CreateMaxForceAttr(100.0)
    drive.CreateTargetPositionAttr(target_position)
    drive.CreateTargetVelocityAttr(target_velocity)
    if max_joint_velocity is not None:
        prim.AddAppliedSchema("PhysxJointAPI")
        prim.CreateAttribute(
            "physxJoint:maxJointVelocity",
            Sdf.ValueTypeNames.Float,
            custom=False,
        ).Set(max_joint_velocity)


def _axis_vector(token: str) -> tuple[float, float, float]:
    return {
        "X": (1.0, 0.0, 0.0),
        "Y": (0.0, 1.0, 0.0),
        "Z": (0.0, 0.0, 1.0),
    }[token]


def _schema_tokens(prim: Any) -> set[str]:
    tokens = {str(token) for token in prim.GetAppliedSchemas()}
    metadata = prim.GetMetadata("apiSchemas")
    if metadata is not None:
        tokens.update(str(token) for token in metadata.GetAppliedItems())
    return tokens
