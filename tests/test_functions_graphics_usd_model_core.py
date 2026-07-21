# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coverage for the queryable USDModel scene index."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pxr import Gf, Sdf, Usd, UsdGeom, Vt

from world_understanding.functions.graphics.usd_model import (
    CollectionInfo,
    USDModel,
    USDPrimNode,
    VariantSelection,
)


def _write_scene(path: Path) -> Path:
    stage = Usd.Stage.CreateNew(str(path))
    stage.SetStartTimeCode(1)
    stage.SetEndTimeCode(10)
    stage.SetFramesPerSecond(24)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)

    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    group = UsdGeom.Xform.Define(stage, "/World/Group")
    mesh = UsdGeom.Mesh.Define(stage, "/World/Group/Cube")
    mesh.CreatePointsAttr(
        Vt.Vec3fArray(
            [
                Gf.Vec3f(0, 0, 0),
                Gf.Vec3f(1, 0, 0),
                Gf.Vec3f(0, 1, 0),
            ]
        )
    )
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    mesh.GetPrim().CreateAttribute("ri:visibility", Sdf.ValueTypeNames.Token).Set(
        "camera"
    )
    variant_set = mesh.GetPrim().GetVariantSets().AddVariantSet("lod")
    variant_set.AddVariant("high")
    variant_set.SetVariantSelection("high")

    hidden = UsdGeom.Xform.Define(stage, "/World/Group/Hidden")
    collection = Usd.CollectionAPI.Apply(world.GetPrim(), "renderable")
    collection.GetIncludesRel().AddTarget(group.GetPath())
    collection.GetExcludesRel().AddTarget(hidden.GetPath())

    assert stage.GetRootLayer().Save()
    return path


def test_usd_model_loads_indexes_queries_tree_and_json(tmp_path: Path) -> None:
    scene_path = _write_scene(tmp_path / "scene.usda")
    model = USDModel(str(scene_path))

    assert model.root_layer == str(scene_path)
    assert model.default_prim_path == "/World"
    assert model.start_time == 1
    assert model.end_time == 10
    assert model.fps == 24
    assert model.up_axis == "Z"
    assert model.meters_per_unit == pytest.approx(0.01)

    cube = model.get_prim("/World/Group/Cube")
    assert cube is not None
    assert cube.get_depth() == 3
    assert cube.is_descendant_of("/World")
    assert cube.is_ancestor_of("/World/Group/Cube/Child")
    assert cube.to_dict()["variant_selections"] == [
        {"set_name": "lod", "selection": "high"}
    ]
    assert cube.custom_tokens["ri:visibility"] == "camera"

    assert model.get_prims_by_type("Mesh") == [cube]
    assert model.get_prims_by_type("Missing") == []
    assert model.get_prims_by_name("Cube") == [cube]
    assert model.get_prims_by_name("Missing") == []
    assert {prim.path for prim in model.get_all_xforms()} >= {"/World", "/World/Group"}
    assert model.get_all_meshes() == [cube]
    assert model.get_parent("/World/Group/Cube").path == "/World/Group"
    assert model.get_parent("/World") is None
    assert [child.path for child in model.get_children("/World")] == ["/World/Group"]
    assert model.get_children("/Missing") == []
    assert [prim.path for prim in model.get_ancestors("/World/Group/Cube")] == [
        "/World/Group",
        "/World",
    ]
    assert [prim.path for prim in model.get_ancestors("/World/Group/Cube", True)] == [
        "/World/Group/Cube",
        "/World/Group",
        "/World",
    ]
    assert model.get_ancestors("/Missing") == []
    assert [prim.path for prim in model.get_descendants("/World", True)] == [
        "/World",
        "/World/Group",
        "/World/Group/Cube",
        "/World/Group/Hidden",
    ]

    collections = model.get_collections_containing_prim("/World/Group/Cube")
    assert len(collections) == 1
    assert collections[0].to_dict() == {
        "name": "renderable",
        "prim_path": "/World",
        "includes": ["/World/Group"],
        "excludes": ["/World/Group/Hidden"],
    }
    assert model.get_collections_containing_prim("/World/Group/Hidden") == []
    assert model.get_xform_owning_collection(collections[0]).path == "/World"
    assert model.get_collections_on_prim("/World") == collections
    assert model.get_collections_on_prim("/Missing") == []
    assert model.find_prim_in_collection_of_xform("/World/Group/Cube") == [
        (collections[0], model.get_prim("/World"))
    ]
    assert model.get_path_to_root("/World/Group/Cube") == [
        "/World/Group/Cube",
        "/World/Group",
        "/World",
    ]
    assert model.get_path_to_root("/Missing") == []

    stats = model.get_subtree_stats("/World")
    assert stats["total_prims"] == 4
    assert stats["type_counts"]["Mesh"] == 1
    assert stats["num_xforms"] == 4

    tree = model.print_tree_to_str(
        show_variants=True,
        show_api_schemas=True,
        show_collections=True,
        show_custom_tokens=True,
        show_stats=True,
    )
    assert "Stage Information:" in tree
    assert "World [Xform] <CollectionAPI:renderable>" in tree
    assert "{renderable:[/World/Group]}" in tree
    assert "Cube [Mesh] {lod=high}" in tree
    assert "|ri:visibility=camera|" in tree
    assert "Statistics:" in tree
    assert model.print_tree_to_str(start_path="/Missing").endswith(
        "Error: Invalid prim path: /Missing"
    )
    assert "Cube" not in model.print_tree_to_str(start_path="/World", max_depth=1)

    data_without_hierarchy = model.to_dict(include_hierarchy=False)
    assert "prims" not in data_without_hierarchy
    data_with_hierarchy = model.to_dict()
    assert data_with_hierarchy["statistics"]["type_distribution"]["Mesh"] == 1
    parsed = json.loads(model.to_json(indent=None))
    assert parsed["stage_info"]["default_prim_path"] == "/World"
    json_path = tmp_path / "model.json"
    model.save_json(str(json_path), indent=0)
    loaded = USDModel.load_json(json_path)
    assert loaded.default_prim_path == "/World"
    assert loaded.get_prim("/World/Group/Cube").custom_tokens["ri:visibility"] == (
        "camera"
    )


def test_usd_model_prints_and_restores_manual_branch_shapes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    data = {
        "file_path": str(tmp_path / "missing.usda"),
        "stage_info": {
            "root_layer": "root.usda",
            "default_prim_path": None,
            "start_time": 0,
            "end_time": 1,
            "fps": 30,
            "up_axis": "Y",
            "meters_per_unit": 1,
        },
        "collections": [
            {
                "name": "manual",
                "prim_path": "/World/Scope",
                "includes": ["/World/Scope"],
            }
        ],
        "prims": {
            "/World": {
                "path": "/World",
                "name": "World",
                "type_name": "Xform",
                "is_xform": True,
                "children_paths": ["/World/Scope"],
            },
            "/World/Scope": {
                "path": "/World/Scope",
                "name": "Scope",
                "type_name": "Scope",
                "parent_path": "/World",
                "children_paths": ["/World/Scope/Hidden", "/World/Scope/Proto"],
                "defined_collections": [
                    {
                        "name": "manual",
                        "prim_path": "/World/Scope",
                        "includes": ["/World/Scope"],
                    }
                ],
            },
            "/World/Scope/Hidden": {
                "path": "/World/Scope/Hidden",
                "name": "Hidden",
                "type_name": None,
                "is_active": False,
                "is_instance": True,
                "parent_path": "/World/Scope",
            },
            "/World/Scope/Proto": {
                "path": "/World/Scope/Proto",
                "name": "Proto",
                "type_name": "Mesh",
                "is_in_prototype": True,
                "parent_path": "/World/Scope",
            },
        },
    }
    model = USDModel.from_dict(data)
    assert model.get_xform_owning_collection(model.collections[0]).path == "/World"
    assert (
        model.get_xform_owning_collection(CollectionInfo("orphan", "/Missing")) is None
    )
    assert model.get_collections_containing_prim("/World/Scope/Proto")[0].name == (
        "manual"
    )
    assert model.find_prim_in_collection_of_xform("/World/Scope/Proto")

    assert "Hidden (inactive)" in model.print_tree_to_str(
        show_info=False,
        show_stats=True,
    )
    assert "Hidden" not in model.print_tree_to_str(show_info=False, active_only=True)
    assert "Total Instances: 1" in model._get_statistics_str()
    assert "Total Prototype Prims: 1" in model._get_statistics_str()
    empty_collection = USDPrimNode(
        "/World/EmptyCollectionOwner",
        "EmptyCollectionOwner",
        defined_collections=[CollectionInfo("empty", "/World/EmptyCollectionOwner")],
    )
    assert (
        "empty:[]"
        in model._get_prim_tree_lines(
            empty_collection,
            "",
            True,
            False,
            False,
            False,
            True,
            False,
            False,
            None,
            0,
        )[0]
    )
    model.print_summary()
    model.print_stage_info()
    model.print_tree(show_info=False)
    model._print_prim_tree(
        model.get_prim("/World"),
        "",
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        None,
        0,
    )
    model._print_statistics()
    captured = capsys.readouterr().out
    assert "USD Model Summary" in captured
    assert "Stage Information" in captured
    assert "Statistics:" in captured

    missing_stage_model = USDModel(tmp_path / "does-not-exist.usda", load_stage=False)
    missing_stage_model._load_stage_metadata()
    missing_stage_model._build_prim_hierarchy()
    assert missing_stage_model.prims == {}

    monkeypatch.setattr(
        USDModel, "load", lambda self: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    assert USDModel.from_dict(
        {**data, "file_path": str(tmp_path)}, load_stage=True
    ).prims


def test_usd_model_dataclasses_and_load_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    collection = CollectionInfo("set", "/World", ["/World/A"], ["/World/A/Hidden"])
    assert collection.contains_prim("/World/A/Mesh")
    assert not collection.contains_prim("/World/A/Hidden/Mesh")
    assert not collection.contains_prim("/Other")
    assert VariantSelection("lod", "low").to_dict() == {
        "set_name": "lod",
        "selection": "low",
    }
    assert USDPrimNode("/World/A", "A").is_descendant_of("/World")
    assert not USDPrimNode("/World/A", "A").is_ancestor_of("/Other")

    with pytest.raises(FileNotFoundError):
        USDModel(tmp_path / "missing.usda")

    broken = tmp_path / "broken.usda"
    broken.write_text("#usda 1.0\n")
    monkeypatch.setattr(Usd.Stage, "Open", lambda _path: None)
    with pytest.raises(RuntimeError, match="Failed to open USD file"):
        USDModel(broken)
