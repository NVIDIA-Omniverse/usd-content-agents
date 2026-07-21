# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Paired-reference coverage for exact Xform instance-root colliders."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Literal

import pytest
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from world_understanding.functions.physics.joint_rigger import (
    JointRiggerContractError,
    JointRiggerInputV1,
    extract_reference_input,
)

reference_module = importlib.import_module(
    "world_understanding.functions.physics.joint_rigger.reference"
)

_SOURCE_URI = "fixture://instance-collider-source"
_REFERENCE_URI = "fixture://instance-collider-reference"
_COLLIDER_PATH = "/World/base/collision"
_PROTOTYPE_PATH = "/World/ColliderPrototypes/base"
_APPROXIMATIONS = ("none", "convexHull", "convexDecomposition", "sdf")


@pytest.mark.parametrize("approximation", _APPROXIMATIONS)
def test_oracle_emits_paired_xform_instance_root_colliders(
    tmp_path: Path,
    approximation: Literal[
        "none",
        "convexHull",
        "convexDecomposition",
        "sdf",
    ],
) -> None:
    source, reference = _write_pair(tmp_path, approximation=approximation)

    result = _extract(source, reference)

    colliders = {
        collider.prim_path: collider
        for body in result.plan.rigid_bodies
        for collider in body.colliders
    }
    assert set(colliders) == {"/World/base/collision", "/World/link/collision"}
    assert all(collider.has_mesh_collision_api for collider in colliders.values())
    assert all(
        collider.mesh_approximation == approximation for collider in colliders.values()
    )


def test_oracle_rejects_bare_api_xform_instance_root(tmp_path: Path) -> None:
    source, reference = _write_pair(tmp_path, approximation="none")
    stage = Usd.Stage.Open(str(reference))
    approximation = stage.GetPrimAtPath(_COLLIDER_PATH).GetAttribute(
        "physics:approximation"
    )
    assert approximation.Clear()
    assert stage.GetRootLayer().Save()
    del stage

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "instance_root_collider_evidence_incomplete"


@pytest.mark.parametrize(
    ("mutated_sides", "expected_code"),
    [
        (("source",), "source_collider_instance_composition_mismatch"),
        (("reference",), "source_collider_instance_composition_mismatch"),
        (("source", "reference"), "source_collider_type_mismatch"),
    ],
)
def test_oracle_rejects_non_instance_xform_colliders(
    tmp_path: Path,
    mutated_sides: tuple[str, ...],
    expected_code: str,
) -> None:
    source, reference = _write_pair(tmp_path, approximation="convexHull")
    paths = {"source": source, "reference": reference}
    for side in mutated_sides:
        stage = Usd.Stage.Open(str(paths[side]))
        collider = stage.GetPrimAtPath(_COLLIDER_PATH)
        collider.SetInstanceable(False)
        assert not collider.IsInstance()
        assert stage.GetRootLayer().Save()
        del stage

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == expected_code


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ("composition", "source_collider_instance_composition_mismatch"),
        ("geometry", "source_collider_geometry_mismatch"),
        ("prototype_transform", "source_collider_transform_mismatch"),
        ("instance_transform", "source_collider_transform_mismatch"),
        ("proxy_physics", "source_physics_not_in_reference"),
    ],
)
def test_oracle_rejects_instance_collider_parity_mismatches(
    tmp_path: Path,
    mutation: str,
    expected_code: str,
) -> None:
    source, reference = _write_pair(tmp_path, approximation="none")
    stage = Usd.Stage.Open(str(source))
    if mutation == "composition":
        UsdGeom.Scope.Define(stage, f"{_PROTOTYPE_PATH}/extra")
    elif mutation == "geometry":
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath(f"{_PROTOTYPE_PATH}/shape"))
        mesh.GetPointsAttr().Set(
            [
                Gf.Vec3f(0.0, 0.0, 0.0),
                Gf.Vec3f(2.0, 0.0, 0.0),
                Gf.Vec3f(0.0, 1.0, 0.0),
            ]
        )
    elif mutation == "prototype_transform":
        prototype_mesh = stage.GetPrimAtPath(f"{_PROTOTYPE_PATH}/shape")
        UsdGeom.Xformable(prototype_mesh).AddTranslateOp().Set(Gf.Vec3d(1.0, 0.0, 0.0))
    elif mutation == "instance_transform":
        instance = stage.GetPrimAtPath(_COLLIDER_PATH)
        UsdGeom.Xformable(instance).AddTranslateOp().Set(Gf.Vec3d(1.0, 0.0, 0.0))
    elif mutation == "proxy_physics":
        prototype_mesh = stage.GetPrimAtPath(f"{_PROTOTYPE_PATH}/shape")
        UsdPhysics.MeshCollisionAPI.Apply(prototype_mesh).CreateApproximationAttr(
            "none"
        )
    else:  # pragma: no cover - parameter guard
        raise AssertionError(mutation)
    assert stage.GetRootLayer().Save()
    del stage

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == expected_code


def test_oracle_normalizes_local_prototype_property_paths(tmp_path: Path) -> None:
    source, reference = _write_pair(
        tmp_path,
        approximation="none",
        offset_source_prototype_id=True,
        source_relationship_target="{prototype}/shape",
        reference_relationship_target="{prototype}/shape",
        source_connection_target="{prototype}/shape.test:source",
        reference_connection_target="{prototype}/shape.test:source",
    )
    source_stage = Usd.Stage.Open(str(source))
    reference_stage = Usd.Stage.Open(str(reference))
    source_prototypes = {
        body_name: source_stage.GetPrimAtPath(
            f"/World/{body_name}/collision"
        ).GetPrototype()
        for body_name in ("base", "link")
    }
    reference_prototypes = {
        body_name: reference_stage.GetPrimAtPath(
            f"/World/{body_name}/collision"
        ).GetPrototype()
        for body_name in ("base", "link")
    }
    source_synthetic_root = Sdf.Path("/__Prototype_source")
    reference_synthetic_root = Sdf.Path("/__Prototype_reference")
    assert reference_module._prototype_path_comparison_key(
        source_synthetic_root.AppendPath("shape.test:source"),
        prototype_path=source_synthetic_root,
    ) == reference_module._prototype_path_comparison_key(
        reference_synthetic_root.AppendPath("shape.test:source"),
        prototype_path=reference_synthetic_root,
    )

    for body_name in ("base", "link"):
        source_prototype = source_prototypes[body_name]
        reference_prototype = reference_prototypes[body_name]
        source_shape = source_prototype.GetChild("shape")
        reference_shape = reference_prototype.GetChild("shape")
        for source_paths, reference_paths in (
            (
                source_shape.GetRelationship("test:prototypeTarget").GetTargets(),
                reference_shape.GetRelationship("test:prototypeTarget").GetTargets(),
            ),
            (
                source_shape.GetAttribute("test:connected").GetConnections(),
                reference_shape.GetAttribute("test:connected").GetConnections(),
            ),
        ):
            assert tuple(
                reference_module._prototype_path_comparison_key(
                    path,
                    prototype_path=source_prototype.GetPath(),
                )
                for path in source_paths
            ) == tuple(
                reference_module._prototype_path_comparison_key(
                    path,
                    prototype_path=reference_prototype.GetPath(),
                )
                for path in reference_paths
            )

    result = _extract(source, reference)

    assert result.plan.rigid_bodies[0].colliders


@pytest.mark.parametrize(
    ("source_target", "reference_target"),
    [
        ("{prototype}", "{prototype}/shape"),
        ("/External/source", "/External/reference"),
    ],
)
def test_oracle_rejects_prototype_relationship_target_mismatches(
    tmp_path: Path,
    source_target: str,
    reference_target: str,
) -> None:
    source, reference = _write_pair(
        tmp_path,
        approximation="none",
        offset_source_prototype_id=True,
        source_relationship_target=source_target,
        reference_relationship_target=reference_target,
    )

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "source_collider_instance_composition_mismatch"


@pytest.mark.parametrize(
    ("source_target", "reference_target"),
    [
        ("{prototype}/shape.test:source", "{prototype}/shape.test:alternate"),
        ("/External.source", "/External.reference"),
    ],
)
def test_oracle_rejects_prototype_attribute_connection_mismatches(
    tmp_path: Path,
    source_target: str,
    reference_target: str,
) -> None:
    source, reference = _write_pair(
        tmp_path,
        approximation="none",
        offset_source_prototype_id=True,
        source_connection_target=source_target,
        reference_connection_target=reference_target,
    )

    with pytest.raises(JointRiggerContractError) as caught:
        _extract(source, reference)

    assert caught.value.code == "source_collider_geometry_mismatch"


def test_instance_composition_helper_rejects_unresolved_prototypes() -> None:
    source_stage = Usd.Stage.CreateInMemory()
    reference_stage = Usd.Stage.CreateInMemory()
    source = UsdGeom.Xform.Define(source_stage, "/Collider").GetPrim()
    reference = UsdGeom.Xform.Define(reference_stage, "/Collider").GetPrim()
    assert not source.GetPrototype()
    assert not reference.GetPrototype()

    with pytest.raises(JointRiggerContractError) as caught:
        reference_module._require_matching_instance_collider_composition(
            source,
            reference,
            path="/Collider",
            UsdGeom=UsdGeom,
        )

    assert caught.value.code == "source_collider_instance_composition_mismatch"


@pytest.mark.parametrize(
    "mutation",
    ("metadata", "nested_instance", "no_geometry"),
)
def test_instance_composition_helper_rejects_prototype_contract_gaps(
    mutation: str,
) -> None:
    include_geometry = mutation == "metadata"
    _source_stage, source = _in_memory_instance_root(
        include_geometry=include_geometry,
        include_nested_instance=mutation == "nested_instance",
        custom_data=mutation == "metadata",
    )
    _reference_stage, reference = _in_memory_instance_root(
        include_geometry=include_geometry,
        include_nested_instance=mutation == "nested_instance",
    )
    assert source.GetPrototype()
    assert reference.GetPrototype()

    with pytest.raises(JointRiggerContractError) as caught:
        reference_module._require_matching_instance_collider_composition(
            source,
            reference,
            path="/Collider",
            UsdGeom=UsdGeom,
        )

    assert caught.value.code == "source_collider_instance_composition_mismatch"


@pytest.mark.parametrize(
    "mutation",
    ("attribute_set", "default_value", "time_sample"),
)
def test_prototype_properties_helper_rejects_attribute_contract_gaps(
    mutation: str,
) -> None:
    source_stage = Usd.Stage.CreateInMemory()
    reference_stage = Usd.Stage.CreateInMemory()
    source = UsdGeom.Xform.Define(source_stage, "/Prototype").GetPrim()
    reference = UsdGeom.Xform.Define(reference_stage, "/Prototype").GetPrim()
    source_attribute = source.CreateAttribute(
        "physics:testValue",
        Sdf.ValueTypeNames.Float,
    )
    if mutation != "attribute_set":
        reference_attribute = reference.CreateAttribute(
            "physics:testValue",
            Sdf.ValueTypeNames.Float,
        )
        source_attribute.Set(1.0 if mutation == "default_value" else 0.0)
        reference_attribute.Set(2.0 if mutation == "default_value" else 0.0)
        if mutation == "time_sample":
            source_attribute.SetVariability(Sdf.VariabilityVarying)
            reference_attribute.SetVariability(Sdf.VariabilityVarying)
            source_attribute.Set(1.0, Usd.TimeCode(1.0))
            reference_attribute.Set(2.0, Usd.TimeCode(1.0))

    with pytest.raises(JointRiggerContractError) as caught:
        reference_module._require_matching_prototype_properties(
            source,
            reference,
            path="/Prototype",
            source_prototype_path=source.GetPath(),
            reference_prototype_path=reference.GetPath(),
        )

    assert caught.value.code == "source_collider_instance_composition_mismatch"


def test_prototype_transform_helper_rejects_transformability_mismatch() -> None:
    source_stage = Usd.Stage.CreateInMemory()
    reference_stage = Usd.Stage.CreateInMemory()
    source = UsdGeom.Xform.Define(source_stage, "/Prototype").GetPrim()
    reference = UsdGeom.Scope.Define(reference_stage, "/Prototype").GetPrim()

    with pytest.raises(JointRiggerContractError) as caught:
        reference_module._require_matching_prototype_collider_transform(
            source,
            reference,
            path="/Prototype",
            UsdGeom=UsdGeom,
        )

    assert caught.value.code == "source_collider_transform_mismatch"


def _in_memory_instance_root(
    *,
    include_geometry: bool,
    include_nested_instance: bool,
    custom_data: bool = False,
) -> tuple[Usd.Stage, Usd.Prim]:
    stage = Usd.Stage.CreateInMemory()
    library_path = "/Library/Collider"
    UsdGeom.Xform.Define(stage, library_path)
    if include_geometry:
        shape = UsdGeom.Cube.Define(stage, f"{library_path}/shape").GetPrim()
        if custom_data:
            shape.SetCustomData({"test:side": "source"})
    if include_nested_instance:
        nested_library_path = "/Library/NestedCollider"
        UsdGeom.Xform.Define(stage, nested_library_path)
        nested = UsdGeom.Xform.Define(stage, f"{library_path}/nested").GetPrim()
        nested.GetReferences().AddInternalReference(nested_library_path)
        nested.SetInstanceable(True)
        assert nested.IsInstance()

    instance = UsdGeom.Xform.Define(stage, "/Collider").GetPrim()
    instance.GetReferences().AddInternalReference(library_path)
    instance.SetInstanceable(True)
    assert instance.IsInstance()
    return stage, instance


def _write_pair(
    directory: Path,
    *,
    approximation: Literal[
        "none",
        "convexHull",
        "convexDecomposition",
        "sdf",
    ],
    offset_source_prototype_id: bool = False,
    source_relationship_target: str | None = None,
    reference_relationship_target: str | None = None,
    source_connection_target: str | None = None,
    reference_connection_target: str | None = None,
) -> tuple[Path, Path]:
    source_path = directory / "source.usda"
    reference_path = directory / "reference.usda"
    for stage_path, rigged in ((source_path, False), (reference_path, True)):
        stage = Usd.Stage.CreateNew(str(stage_path))
        root = UsdGeom.Xform.Define(stage, "/World").GetPrim()
        stage.SetDefaultPrim(root)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
        UsdGeom.Scope.Define(stage, "/World/ColliderPrototypes")

        if not rigged and offset_source_prototype_id:
            UsdGeom.Xform.Define(stage, "/World/ColliderPrototypes/a_offset")
            offset = UsdGeom.Xform.Define(stage, "/World/a_offset").GetPrim()
            offset.GetReferences().AddInternalReference(
                "/World/ColliderPrototypes/a_offset"
            )
            offset.SetInstanceable(True)

        relationship_target = (
            reference_relationship_target if rigged else source_relationship_target
        )
        connection_target = (
            reference_connection_target if rigged else source_connection_target
        )
        for body_name in ("base", "link"):
            body_path = f"/World/{body_name}"
            body = UsdGeom.Xform.Define(stage, body_path).GetPrim()
            prototype_path = f"/World/ColliderPrototypes/{body_name}"
            UsdGeom.Xform.Define(stage, prototype_path)
            mesh = UsdGeom.Mesh.Define(stage, f"{prototype_path}/shape")
            mesh.CreatePointsAttr(
                [
                    Gf.Vec3f(0.0, 0.0, 0.0),
                    Gf.Vec3f(1.0, 0.0, 0.0),
                    Gf.Vec3f(0.0, 1.0, 0.0),
                ]
            )
            mesh.CreateFaceVertexCountsAttr([3])
            mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
            if relationship_target is not None:
                mesh.GetPrim().CreateRelationship("test:prototypeTarget").SetTargets(
                    [relationship_target.format(prototype=prototype_path)]
                )
            if connection_target is not None:
                mesh.GetPrim().CreateAttribute(
                    "test:source",
                    Sdf.ValueTypeNames.Float,
                ).Set(1.0)
                mesh.GetPrim().CreateAttribute(
                    "test:alternate",
                    Sdf.ValueTypeNames.Float,
                ).Set(2.0)
                mesh.GetPrim().CreateAttribute(
                    "test:connected",
                    Sdf.ValueTypeNames.Float,
                ).SetConnections([connection_target.format(prototype=prototype_path)])

            collider = UsdGeom.Xform.Define(
                stage,
                f"{body_path}/collision",
            ).GetPrim()
            collider.GetReferences().AddInternalReference(prototype_path)
            collider.SetInstanceable(True)
            assert collider.IsInstance()

            if rigged:
                UsdPhysics.RigidBodyAPI.Apply(body).CreateRigidBodyEnabledAttr(True)
                UsdPhysics.CollisionAPI.Apply(collider).CreateCollisionEnabledAttr(True)
                UsdPhysics.MeshCollisionAPI.Apply(collider).CreateApproximationAttr(
                    approximation
                )

        if rigged:
            UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath("/World/base"))
            UsdGeom.Scope.Define(stage, "/World/Joints")
            joint = UsdPhysics.RevoluteJoint.Define(stage, "/World/Joints/hinge")
            joint.CreateBody0Rel().SetTargets(["/World/base"])
            joint.CreateBody1Rel().SetTargets(["/World/link"])
            joint.CreateAxisAttr("Z")
        assert stage.GetRootLayer().Save()
    return source_path, reference_path


def _extract(source: Path, reference: Path) -> JointRiggerInputV1:
    return extract_reference_input(
        source,
        reference,
        source_uri=_SOURCE_URI,
        reference_uri=_REFERENCE_URI,
    )
