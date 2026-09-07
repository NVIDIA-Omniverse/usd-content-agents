# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Targeted edge coverage for apply_physics helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pxr import Sdf, Usd, UsdGeom, UsdPhysics

from physics_agent.functions.apply_physics import (
    PhysicsAuthoringError,
    _apply_predictions_to_stage,
    _block_existing_mass,
    _remove_flattened_mass_attributes,
    apply_physics,
    load_predictions,
)


def _stage_with_default() -> Usd.Stage:
    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Cube.Define(stage, "/World/Cube")
    stage.SetDefaultPrim(root.GetPrim())
    return stage


def _mark_deformable(prim: Usd.Prim, semantics: str) -> None:
    if semantics == "schema":
        prim.AddAppliedSchema("PhysicsDeformableBodyAPI")
    elif semantics == "owned-marker":
        prim.SetCustomDataByKey(
            "physicsAgentVompDeformable",
            {"adapter": "physics_agent.vomp_volume_deformable"},
        )
    else:
        prim.AddAppliedSchema(semantics)


@pytest.mark.parametrize(
    "semantics",
    [
        "schema",
        "owned-marker",
        "PhysicsCurvesDeformableSimAPI",
        "PhysicsSurfaceDeformableSimAPI",
        "PhysicsVolumeDeformableSimAPI",
    ],
)
@pytest.mark.parametrize(
    ("location", "relation"),
    [("target", "already has"), ("ancestor", "below"), ("descendant", "contains")],
)
def test_apply_predictions_rejects_deformable_collider_hierarchy(
    tmp_path: Path,
    semantics: str,
    location: str,
    relation: str,
) -> None:
    stage = Usd.Stage.CreateInMemory()
    world = UsdGeom.Xform.Define(stage, "/World")
    ancestor = UsdGeom.Xform.Define(stage, "/World/Ancestor").GetPrim()
    target = UsdGeom.Xform.Define(stage, "/World/Ancestor/Target").GetPrim()
    descendant = UsdGeom.Cube.Define(
        stage, "/World/Ancestor/Target/Descendant"
    ).GetPrim()
    stage.SetDefaultPrim(world.GetPrim())
    conflict = {
        "target": target,
        "ancestor": ancestor,
        "descendant": descendant,
    }[location]
    _mark_deformable(conflict, semantics)

    with pytest.raises(PhysicsAuthoringError, match=rf"{relation}.*deformable"):
        _apply_predictions_to_stage(
            stage,
            tmp_path / "scene.usda",
            [
                {
                    "id": str(target.GetPath()),
                    "classification": {
                        "physical_properties": {
                            "estimated_mass_kg": 1.0,
                            "density": 100.0,
                        },
                    },
                }
            ],
            "convexHull",
            "classification",
            "skip_mass",
            allow_empty_predictions=False,
            author_rigid_body=False,
        )

    assert not target.HasAPI(UsdPhysics.CollisionAPI)
    assert not stage.GetPrimAtPath("/World/PhysicsScene").IsValid()


def test_apply_predictions_rejects_default_body_containing_deformable(
    tmp_path: Path,
) -> None:
    stage = _stage_with_default()
    soft_body = UsdGeom.Xform.Define(stage, "/World/SoftBody").GetPrim()
    _mark_deformable(soft_body, "schema")

    with pytest.raises(
        PhysicsAuthoringError,
        match=r"RigidBodyAPI/MassAPI.*contains deformable",
    ):
        _apply_predictions_to_stage(
            stage,
            tmp_path / "scene.usda",
            [],
            "convexHull",
            "classification",
            "skip_mass",
            allow_empty_predictions=True,
            author_rigid_body=True,
        )

    assert not stage.GetDefaultPrim().HasAPI(UsdPhysics.RigidBodyAPI)


def test_apply_predictions_rejects_explicit_mass_body_containing_deformable(
    tmp_path: Path,
) -> None:
    stage = _stage_with_default()
    mass_body = UsdGeom.Xform.Define(stage, "/MassBody").GetPrim()
    soft_body = UsdGeom.Xform.Define(stage, "/MassBody/SoftBody").GetPrim()
    _mark_deformable(soft_body, "owned-marker")

    with pytest.raises(PhysicsAuthoringError, match=r"MassAPI.*contains deformable"):
        _apply_predictions_to_stage(
            stage,
            tmp_path / "scene.usda",
            [
                {
                    "id": "/World/Cube",
                    "classification": {
                        "component_id": "component_001",
                        "mass_authoring_path": str(mass_body.GetPath()),
                        "physical_properties": {
                            "estimated_mass_kg": 1.0,
                            "density": 100.0,
                        },
                    },
                }
            ],
            "convexHull",
            "classification",
            "skip_mass",
            allow_empty_predictions=False,
            author_rigid_body=True,
        )

    assert not mass_body.HasAPI(UsdPhysics.MassAPI)
    assert not stage.GetPrimAtPath("/World/Cube").HasAPI(UsdPhysics.CollisionAPI)
    assert not stage.GetPrimAtPath("/World/PhysicsScene").IsValid()


def test_apply_predictions_rejects_deformable_skip_mass_target_before_writes(
    tmp_path: Path,
) -> None:
    stage = _stage_with_default()
    mass_body = UsdGeom.Xform.Define(stage, "/MassBody").GetPrim()
    UsdPhysics.MassAPI.Apply(mass_body).CreateMassAttr(5.0)
    soft_body = UsdGeom.Xform.Define(stage, "/MassBody/SoftBody").GetPrim()
    _mark_deformable(soft_body, "schema")

    with pytest.raises(PhysicsAuthoringError, match=r"MassAPI.*contains deformable"):
        _apply_predictions_to_stage(
            stage,
            tmp_path / "scene.usda",
            [
                {
                    "id": "/World/Cube",
                    "quality_warnings": [{"code": "mass_scale_suspicious"}],
                    "classification": {
                        "mass_authoring_path": str(mass_body.GetPath()),
                        "physical_properties": {
                            "estimated_mass_kg": 1.0,
                            "density": 100.0,
                        },
                    },
                }
            ],
            "convexHull",
            "classification",
            "skip_mass",
            allow_empty_predictions=False,
            author_rigid_body=True,
        )

    assert UsdPhysics.MassAPI(mass_body).GetMassAttr().Get() == 5.0
    assert not stage.GetPrimAtPath("/World/Cube").HasAPI(UsdPhysics.CollisionAPI)
    assert not stage.GetPrimAtPath("/World/PhysicsScene").IsValid()


def test_apply_predictions_rechecks_rigid_children_after_deinstancing(
    tmp_path: Path,
) -> None:
    referenced_path = tmp_path / "referenced.usda"
    referenced = Usd.Stage.CreateNew(str(referenced_path))
    model = UsdGeom.Xform.Define(referenced, "/Model")
    referenced.SetDefaultPrim(model.GetPrim())
    link = UsdGeom.Xform.Define(referenced, "/Model/Link").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(link).CreateRigidBodyEnabledAttr(True)
    referenced.GetRootLayer().Save()

    stage = Usd.Stage.CreateInMemory()
    world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    world.GetReferences().AddReference(str(referenced_path), "/Model")
    world.SetInstanceable(True)
    stage.SetDefaultPrim(world)

    _apply_predictions_to_stage(
        stage,
        tmp_path / "scene.usda",
        [],
        "convexHull",
        "classification",
        "skip_mass",
        allow_empty_predictions=True,
        author_rigid_body=True,
    )

    assert not world.IsInstanceable()
    assert not world.HasAPI(UsdPhysics.RigidBodyAPI)
    assert stage.GetPrimAtPath("/World/Link").HasAPI(UsdPhysics.RigidBodyAPI)


def test_apply_predictions_rechecks_rigid_children_after_mass_target_deinstancing(
    tmp_path: Path,
) -> None:
    referenced_path = tmp_path / "referenced_mass_body.usda"
    referenced = Usd.Stage.CreateNew(str(referenced_path))
    model = UsdGeom.Xform.Define(referenced, "/Model")
    referenced.SetDefaultPrim(model.GetPrim())
    link = UsdGeom.Xform.Define(referenced, "/Model/Link").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(link).CreateRigidBodyEnabledAttr(True)
    referenced.GetRootLayer().Save()

    stage = _stage_with_default()
    world = stage.GetDefaultPrim()
    body = UsdGeom.Xform.Define(stage, "/World/Body").GetPrim()
    body.GetReferences().AddReference(str(referenced_path), "/Model")
    body.SetInstanceable(True)

    _apply_predictions_to_stage(
        stage,
        tmp_path / "scene.usda",
        [
            {
                "id": "/World/Cube",
                "classification": {
                    "component_id": "component_001",
                    "mass_authoring_path": "/World/Body",
                    "component_estimated_mass_kg": 1.0,
                    "physical_properties": {
                        "estimated_mass_kg": 1.0,
                        "density": 100.0,
                    },
                },
            }
        ],
        "convexHull",
        "classification",
        "skip_mass",
        allow_empty_predictions=False,
        author_rigid_body=True,
    )

    assert not body.IsInstanceable()
    assert UsdPhysics.MassAPI(body).GetMassAttr().Get() == 1.0
    assert not world.HasAPI(UsdPhysics.RigidBodyAPI)
    assert stage.GetPrimAtPath("/World/Body/Link").HasAPI(UsdPhysics.RigidBodyAPI)


def test_load_predictions_skips_blank_lines(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        "\n" + json.dumps({"id": "/World/Cube"}) + "\n\n",
        encoding="utf-8",
    )
    assert load_predictions(str(predictions)) == [{"id": "/World/Cube"}]


def test_mass_cleanup_edges(tmp_path: Path) -> None:
    stage = _stage_with_default()
    cube = stage.GetPrimAtPath("/World/Cube")
    UsdPhysics.MassAPI.Apply(cube).CreateMassAttr(1.0)
    layer = stage.GetRootLayer()

    _remove_flattened_mass_attributes(layer, {"/Missing", "/World/Cube"})
    cube_spec = layer.GetPrimAtPath("/World/Cube")
    assert "physics:mass" not in cube_spec.properties

    assert _block_existing_mass(stage, "/Missing") is False
    assert _block_existing_mass(stage, "/World") is False


def test_apply_predictions_skips_bad_records_when_allowed(tmp_path: Path) -> None:
    stage = _stage_with_default()
    applied, skipped, _cleared, _skipped_mass, _body_path = _apply_predictions_to_stage(
        stage,
        tmp_path / "scene.usda",
        [
            {},
            {"id": "/World/Cube", "classification": "not-a-dict"},
            {
                "id": "/World/Cube",
                "quality_warnings": [{"code": "mass_scale_suspicious"}],
                "classification": "not-a-dict",
            },
            {
                "id": "/World/Cube",
                "quality_warnings": [{"code": "mass_scale_suspicious"}],
                "classification": {},
            },
            {
                "id": "/Missing",
                "quality_warnings": [{"code": "mass_scale_suspicious"}],
                "classification": {
                    "physical_properties": {
                        "estimated_mass_kg": 1.0,
                        "density": 100.0,
                    }
                },
            },
        ],
        "convexHull",
        "classification",
        "skip_mass",
        allow_empty_predictions=True,
    )
    assert applied == 0
    assert skipped == 5


def test_apply_predictions_blocks_existing_aggregate_mass_on_suspicious_scale(
    tmp_path: Path,
) -> None:
    stage = _stage_with_default()
    root = stage.GetDefaultPrim()
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(5.0)

    applied, skipped, _cleared, skipped_mass, body_path = _apply_predictions_to_stage(
        stage,
        tmp_path / "scene.usda",
        [
            {
                "id": "/World/Cube",
                "quality_warnings": [{"code": "mass_scale_suspicious"}],
                "classification": {
                    "physical_properties": {
                        "estimated_mass_kg": 10.0,
                        "density": 100.0,
                    },
                },
            }
        ],
        "convexHull",
        "classification",
        "skip_mass",
        allow_empty_predictions=False,
    )

    assert applied == 1
    assert skipped == 0
    assert skipped_mass == {"/World"}
    assert body_path == "/World"
    mass_attr = UsdPhysics.MassAPI(root).GetMassAttr()
    assert mass_attr.Get() is None


def test_apply_predictions_deduplicates_explicit_component_mass(
    tmp_path: Path,
) -> None:
    stage = _stage_with_default()
    UsdGeom.Cube.Define(stage, "/World/CubeB")

    applied, skipped, _cleared, _skipped_mass, _body_path = _apply_predictions_to_stage(
        stage,
        tmp_path / "scene.usda",
        [
            {
                "id": "/World/Cube",
                "classification": {
                    "component_id": "component_001",
                    "mass_authoring_path": "/World",
                    "component_estimated_mass_kg": 1.0,
                    "physical_properties": {
                        "estimated_mass_kg": 1.0,
                        "density": 100.0,
                    },
                },
            },
            {
                "id": "/World/CubeB",
                "classification": {
                    "component_id": "component_001",
                    "mass_authoring_path": "/World",
                    "component_estimated_mass_kg": 1.0,
                    "physical_properties": {
                        "estimated_mass_kg": 1.0,
                        "density": 100.0,
                    },
                },
            },
        ],
        "convexHull",
        "classification",
        "skip_mass",
        allow_empty_predictions=False,
    )

    assert applied == 2
    assert skipped == 0
    assert UsdPhysics.MassAPI(stage.GetDefaultPrim()).GetMassAttr().Get() == 1.0


def test_apply_predictions_plans_mass_after_prior_record_adds_collision(
    tmp_path: Path,
) -> None:
    stage = _stage_with_default()
    body = UsdGeom.Xform.Define(stage, "/World/Body").GetPrim()

    applied, skipped, _cleared, _skipped_mass, _body_path = _apply_predictions_to_stage(
        stage,
        tmp_path / "scene.usda",
        [
            {
                "id": "/World/Cube",
                "classification": {
                    "component_id": "component_001",
                    "mass_authoring_path": "/World/Body",
                    "component_estimated_mass_kg": 1.0,
                    "physical_properties": {
                        "estimated_mass_kg": 1.0,
                        "density": 100.0,
                    },
                },
            },
            {
                "id": "/World/Cube",
                "classification": {
                    "collision_mode": "preserve_existing",
                    "component_id": "component_002",
                    "mass_authoring_path": "/World/Body",
                    "component_estimated_mass_kg": 2.0,
                    "physical_properties": {
                        "estimated_mass_kg": 2.0,
                        "density": 100.0,
                    },
                },
            },
        ],
        "convexHull",
        "classification",
        "skip_mass",
        allow_empty_predictions=False,
    )

    assert applied == 2
    assert skipped == 0
    assert UsdPhysics.MassAPI(body).GetMassAttr().Get() == 3.0


def test_apply_predictions_skips_nonpositive_component_mass(
    tmp_path: Path,
) -> None:
    stage = _stage_with_default()
    body = UsdGeom.Xform.Define(stage, "/World/Body").GetPrim()

    applied, skipped, _cleared, _skipped_mass, _body_path = _apply_predictions_to_stage(
        stage,
        tmp_path / "scene.usda",
        [
            {
                "id": "/World/Cube",
                "classification": {
                    "component_id": "component_001",
                    "mass_authoring_path": "/World/Body",
                    "component_estimated_mass_kg": 0.0,
                    "physical_properties": {
                        "estimated_mass_kg": 1.0,
                        "density": 100.0,
                    },
                },
            }
        ],
        "convexHull",
        "classification",
        "skip_mass",
        allow_empty_predictions=False,
    )

    assert applied == 1
    assert skipped == 0
    assert not body.HasAPI(UsdPhysics.MassAPI)


def test_apply_predictions_falls_back_from_invalid_component_mass(
    tmp_path: Path,
) -> None:
    stage = _stage_with_default()
    body = UsdGeom.Xform.Define(stage, "/World/Body").GetPrim()

    applied, skipped, _cleared, _skipped_mass, _body_path = _apply_predictions_to_stage(
        stage,
        tmp_path / "scene.usda",
        [
            {
                "id": "/World/Cube",
                "classification": {
                    "component_id": "component_001",
                    "mass_authoring_path": "/World/Body",
                    "component_estimated_mass_kg": "unknown",
                    "physical_properties": {
                        "estimated_mass_kg": 1.25,
                        "density": 100.0,
                    },
                },
            }
        ],
        "convexHull",
        "classification",
        "skip_mass",
        allow_empty_predictions=False,
    )

    assert applied == 1
    assert skipped == 0
    assert UsdPhysics.MassAPI(body).GetMassAttr().Get() == 1.25


def test_apply_predictions_clears_suspicious_component_mass_authoring_path(
    tmp_path: Path,
) -> None:
    stage = _stage_with_default()
    cube = stage.GetPrimAtPath("/World/Cube")
    body = UsdGeom.Xform.Define(stage, "/World/Body").GetPrim()
    UsdPhysics.MassAPI.Apply(cube).CreateMassAttr(2.0)
    UsdPhysics.MassAPI.Apply(body).CreateMassAttr(5.0)

    applied, skipped, _cleared, skipped_mass, _body_path = _apply_predictions_to_stage(
        stage,
        tmp_path / "scene.usda",
        [
            {
                "id": "/World/Cube",
                "quality_warnings": [{"code": "mass_scale_suspicious"}],
                "classification": {
                    "component_id": "component_001",
                    "mass_authoring_path": "/World/Body",
                    "physical_properties": {
                        "estimated_mass_kg": 1.25,
                        "density": 100.0,
                    },
                },
            }
        ],
        "convexHull",
        "classification",
        "skip_mass",
        allow_empty_predictions=False,
    )

    assert applied == 1
    assert skipped == 0
    assert skipped_mass == {"/World", "/World/Cube", "/World/Body"}
    assert not UsdPhysics.MassAPI(cube).GetMassAttr().HasAuthoredValueOpinion()
    assert not UsdPhysics.MassAPI(body).GetMassAttr().HasAuthoredValueOpinion()


def test_apply_predictions_deinstances_explicit_body_mass_target(
    tmp_path: Path,
) -> None:
    stage = _stage_with_default()
    body = UsdGeom.Xform.Define(stage, "/World/Body").GetPrim()
    body.SetInstanceable(True)

    _apply_predictions_to_stage(
        stage,
        tmp_path / "scene.usda",
        [
            {
                "id": "/World/Cube",
                "classification": {
                    "component_id": "component_001",
                    "mass_authoring_path": "/World/Body",
                    "component_estimated_mass_kg": 1.0,
                    "physical_properties": {
                        "estimated_mass_kg": 1.0,
                        "density": 100.0,
                    },
                },
            }
        ],
        "convexHull",
        "classification",
        "skip_mass",
        allow_empty_predictions=False,
    )

    assert body.IsInstanceable() is False
    assert UsdPhysics.MassAPI(body).GetMassAttr().Get() == 1.0


def test_apply_predictions_can_skip_default_rigid_body_for_static_intent(
    tmp_path: Path,
) -> None:
    stage = _stage_with_default()

    applied, skipped, _cleared, _skipped_mass, _body_path = _apply_predictions_to_stage(
        stage,
        tmp_path / "scene.usda",
        [
            {
                "id": "/World/Cube",
                "classification": {
                    "physical_properties": {
                        "estimated_mass_kg": 1.0,
                        "density": 100.0,
                    },
                },
            }
        ],
        "convexHull",
        "classification",
        "skip_mass",
        allow_empty_predictions=False,
        author_rigid_body=False,
    )

    assert applied == 1
    assert skipped == 0
    default_prim = stage.GetDefaultPrim()
    assert not default_prim.HasAPI(UsdPhysics.RigidBodyAPI)
    assert not default_prim.HasAPI(UsdPhysics.MassAPI)
    assert stage.GetPrimAtPath("/World/Cube").HasAPI(UsdPhysics.CollisionAPI)


def test_apply_predictions_skips_default_body_when_enabled_ancestor_exists(
    tmp_path: Path,
) -> None:
    stage = Usd.Stage.CreateInMemory()
    world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    UsdGeom.Xform.Define(stage, "/World/Group")
    asset = UsdGeom.Xform.Define(stage, "/World/Group/Asset").GetPrim()
    UsdGeom.Cube.Define(stage, "/World/Group/Asset/Cube")
    stage.SetDefaultPrim(asset)
    UsdPhysics.RigidBodyAPI.Apply(world).CreateRigidBodyEnabledAttr(True)

    applied, skipped, _cleared, _skipped_mass, _body_path = _apply_predictions_to_stage(
        stage,
        tmp_path / "scene.usda",
        [
            {
                "id": "/World/Group/Asset/Cube",
                "classification": {
                    "physical_properties": {
                        "estimated_mass_kg": 1.0,
                        "density": 100.0,
                    },
                },
            }
        ],
        "convexHull",
        "classification",
        "skip_mass",
        allow_empty_predictions=False,
        author_rigid_body=True,
    )

    assert applied == 1
    assert skipped == 0
    assert world.HasAPI(UsdPhysics.RigidBodyAPI)
    assert not asset.HasAPI(UsdPhysics.RigidBodyAPI)
    assert stage.GetPrimAtPath("/World/Group/Asset/Cube").HasAPI(
        UsdPhysics.CollisionAPI
    )


def test_apply_predictions_allows_non_xform_default_for_static_intent(
    tmp_path: Path,
) -> None:
    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Scope.Define(stage, "/World")
    UsdGeom.Cube.Define(stage, "/World/Cube")
    stage.SetDefaultPrim(root.GetPrim())

    applied, skipped, _cleared, _skipped_mass, _body_path = _apply_predictions_to_stage(
        stage,
        tmp_path / "scene.usda",
        [
            {
                "id": "/World/Cube",
                "classification": {
                    "physical_properties": {
                        "estimated_mass_kg": 1.0,
                        "density": 100.0,
                    },
                },
            }
        ],
        "convexHull",
        "classification",
        "skip_mass",
        allow_empty_predictions=False,
        author_rigid_body=False,
    )

    assert applied == 1
    assert skipped == 0
    default_prim = stage.GetDefaultPrim()
    assert not default_prim.HasAPI(UsdPhysics.RigidBodyAPI)
    assert stage.GetPrimAtPath("/World/Cube").HasAPI(UsdPhysics.CollisionAPI)


def test_apply_predictions_preserves_existing_collider_schema(
    tmp_path: Path,
) -> None:
    stage = _stage_with_default()
    cube = stage.GetPrimAtPath("/World/Cube")
    UsdPhysics.CollisionAPI.Apply(cube).CreateCollisionEnabledAttr(True)

    _apply_predictions_to_stage(
        stage,
        tmp_path / "scene.usda",
        [
            {
                "id": "/World/Cube",
                "classification": {
                    "collision_mode": "preserve_existing",
                    "physical_properties": {
                        "estimated_mass_kg": 0.0,
                        "density": 0.0,
                    },
                },
            }
        ],
        "convexHull",
        "classification",
        "skip_mass",
        allow_empty_predictions=False,
    )

    assert cube.HasAPI(UsdPhysics.CollisionAPI)
    assert not cube.HasAPI(UsdPhysics.MeshCollisionAPI)


def test_apply_predictions_uses_record_collision_approximation(
    tmp_path: Path,
) -> None:
    stage = _stage_with_default()

    _apply_predictions_to_stage(
        stage,
        tmp_path / "scene.usda",
        [
            {
                "id": "/World/Cube",
                "classification": {
                    "collision_approximation": "convexDecomposition",
                    "physical_properties": {
                        "estimated_mass_kg": 0.0,
                        "density": 0.0,
                    },
                },
            }
        ],
        "convexHull",
        "classification",
        "skip_mass",
        allow_empty_predictions=False,
    )

    mesh_api = UsdPhysics.MeshCollisionAPI(stage.GetPrimAtPath("/World/Cube"))
    assert mesh_api.GetApproximationAttr().Get() == "convexDecomposition"


def test_apply_predictions_rejects_preserve_existing_without_collider(
    tmp_path: Path,
) -> None:
    stage = _stage_with_default()

    with pytest.raises(PhysicsAuthoringError, match="preserve_existing"):
        _apply_predictions_to_stage(
            stage,
            tmp_path / "scene.usda",
            [
                {
                    "id": "/World/Cube",
                    "classification": {
                        "collision_mode": "preserve_existing",
                        "physical_properties": {
                            "estimated_mass_kg": 0.0,
                            "density": 0.0,
                        },
                    },
                }
            ],
            "convexHull",
            "classification",
            "skip_mass",
            allow_empty_predictions=False,
        )


def test_apply_predictions_rejects_preserve_existing_disabled_collider(
    tmp_path: Path,
) -> None:
    stage = _stage_with_default()
    cube = stage.GetPrimAtPath("/World/Cube")
    UsdPhysics.CollisionAPI.Apply(cube).CreateCollisionEnabledAttr(False)

    with pytest.raises(PhysicsAuthoringError, match="disabled"):
        _apply_predictions_to_stage(
            stage,
            tmp_path / "scene.usda",
            [
                {
                    "id": "/World/Cube",
                    "classification": {
                        "collision_mode": "preserve_existing",
                        "physical_properties": {
                            "estimated_mass_kg": 0.0,
                            "density": 0.0,
                        },
                    },
                }
            ],
            "convexHull",
            "classification",
            "skip_mass",
            allow_empty_predictions=False,
        )


def test_block_existing_mass_removes_root_authored_value() -> None:
    stage = _stage_with_default()
    cube = stage.GetPrimAtPath("/World/Cube")
    UsdPhysics.MassAPI.Apply(cube).CreateMassAttr(2.0)

    assert _block_existing_mass(stage, "/World/Cube") is True
    root_spec = stage.GetRootLayer().GetPrimAtPath("/World/Cube")
    assert "physics:mass" not in root_spec.properties


def test_block_existing_mass_deinstances_target() -> None:
    stage = _stage_with_default()
    cube = stage.GetPrimAtPath("/World/Cube")
    cube.SetInstanceable(True)
    UsdPhysics.MassAPI.Apply(cube).CreateMassAttr(2.0)

    assert _block_existing_mass(stage, "/World/Cube") is True
    assert cube.IsInstanceable() is False


def test_apply_physics_does_not_mutate_open_source_layer(tmp_path: Path) -> None:
    """apply_physics must be side-effect-free on its input layer.

    Regression: apply_physics opens its input through the process-global
    Sdf.Layer registry and authors on it. When a long-lived caller (a Content
    Workbench session) holds that same source layer open, the in-memory edits
    leaked into later re-inspections of the source, which then rejected an
    author-on-targets decision because the source "already had" colliders. The
    authored physics belongs only in the distinct output file.
    """
    source = tmp_path / "asset.usda"
    stage = Usd.Stage.CreateNew(str(source))
    root = UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Cube.Define(stage, "/World/Cube")
    stage.SetDefaultPrim(root.GetPrim())
    stage.GetRootLayer().Save()
    del stage

    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        json.dumps(
            {
                "id": "/World/Cube",
                "classification": {
                    "material": "generic",
                    "component_type": "body",
                    "physical_properties": {"density": 850.0},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "asset_physics.usda"

    # Hold the source layer open for the duration of the call, mimicking a
    # persistent session that keeps the source stage resident in the registry.
    held = Usd.Stage.Open(str(source))
    apply_physics(str(source), str(predictions), str(output), collision_approx="none")

    # The still-registered source layer must be reverted to its on-disk
    # (physics-free) state: not dirty and carrying no authored physics schema.
    source_layer = Sdf.Layer.Find(str(source))
    assert source_layer is not None
    assert not source_layer.dirty
    assert "PhysicsCollisionAPI" not in source_layer.ExportToString()
    assert held is not None  # keep the source layer registered across the call

    # The output, however, must carry the authored collider.
    out_stage = Usd.Stage.Open(str(output))
    out_cube = out_stage.GetPrimAtPath("/World/Cube")
    assert "PhysicsCollisionAPI" in out_cube.GetAppliedSchemas()
