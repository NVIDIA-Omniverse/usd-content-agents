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
    _copy_usdz_asset_for_flattened_output,
    _is_relative_to,
    _remove_flattened_mass_attributes,
    _rewrite_flattened_usdz_asset_paths,
    load_predictions,
)


def _stage_with_default() -> Usd.Stage:
    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Cube.Define(stage, "/World/Cube")
    stage.SetDefaultPrim(root.GetPrim())
    return stage


def test_load_predictions_skips_blank_lines(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        "\n" + json.dumps({"id": "/World/Cube"}) + "\n\n",
        encoding="utf-8",
    )
    assert load_predictions(str(predictions)) == [{"id": "/World/Cube"}]


def test_mass_cleanup_and_relative_path_edges(tmp_path: Path) -> None:
    stage = _stage_with_default()
    cube = stage.GetPrimAtPath("/World/Cube")
    UsdPhysics.MassAPI.Apply(cube).CreateMassAttr(1.0)
    layer = stage.GetRootLayer()

    _remove_flattened_mass_attributes(layer, {"/Missing", "/World/Cube"})
    cube_spec = layer.GetPrimAtPath("/World/Cube")
    assert "physics:mass" not in cube_spec.properties

    assert _is_relative_to(tmp_path / "inside", tmp_path) is True
    assert _is_relative_to(Path("/definitely/outside"), tmp_path) is False

    assert _block_existing_mass(stage, "/Missing") is False
    assert _block_existing_mass(stage, "/World") is False


def test_usdz_asset_rewrite_edges(tmp_path: Path) -> None:
    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()
    output = tmp_path / "out.usda"
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_asset = outside / "texture.png"
    outside_asset.write_bytes(b"png")

    unchanged = Sdf.AssetPath(str(outside_asset))
    assert (
        _copy_usdz_asset_for_flattened_output(unchanged, extract_dir, output).path
        == unchanged.path
    )

    target = outside / "linked.png"
    target.write_bytes(b"png")
    (extract_dir / "linked.png").symlink_to(target)
    symlinked = Sdf.AssetPath("linked.png")
    assert (
        _copy_usdz_asset_for_flattened_output(symlinked, extract_dir, output).path
        == symlinked.path
    )

    escape_source_dir = extract_dir / "escape"
    escape_source_dir.mkdir()
    (escape_source_dir / "texture.png").write_bytes(b"png")
    escape_target_dir = tmp_path / "escape-target"
    escape_target_dir.mkdir()
    assets_escape = tmp_path / "out_assets" / "escape"
    assets_escape.parent.mkdir()
    assets_escape.symlink_to(escape_target_dir)
    escaping = Sdf.AssetPath("escape/texture.png")
    assert (
        _copy_usdz_asset_for_flattened_output(escaping, extract_dir, output).path
        == escaping.path
    )

    texture = extract_dir / "texture.png"
    texture.write_bytes(b"png")
    stage = Usd.Stage.CreateInMemory()
    prim = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    attr = prim.CreateAttribute("inputs:textures", Sdf.ValueTypeNames.AssetArray)
    attr.Set(Sdf.AssetPathArray([Sdf.AssetPath("texture.png")]))

    _rewrite_flattened_usdz_asset_paths(stage.GetRootLayer(), extract_dir, output)
    rewritten = attr.Get()[0].path
    assert rewritten == "out_assets/texture.png"
    assert (tmp_path / "out_assets" / "texture.png").exists()


def test_apply_predictions_skips_bad_records_when_allowed(tmp_path: Path) -> None:
    stage = _stage_with_default()
    applied, skipped, _cleared, _skipped_mass, _body_path = _apply_predictions_to_stage(
        stage,
        tmp_path / "scene.usda",
        [
            {},
            {"id": "/World/Cube", "classification": "not-a-dict"},
        ],
        "convexHull",
        "classification",
        "skip_mass",
        allow_empty_predictions=True,
    )
    assert applied == 0
    assert skipped == 2


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
